"""Built-in HTTP transcription server for mlx-qwen3-asr.

Turns any Apple Silicon Mac into a speech-to-text API endpoint.
Requires optional dependencies: ``pip install mlx-qwen3-asr[serve]``

Usage::

    mlx-qwen3-asr serve --port 8765 --api-key mykey123
"""

import asyncio
import logging
import tempfile
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger("mlx_qwen3_asr.server")


# ---------------------------------------------------------------------------
# Job model
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    """In-memory representation of a transcription job."""

    job_id: str
    status: JobStatus
    created_at: float
    api_key: str = ""
    completed_at: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    temp_path: Optional[str] = None

    # Request parameters
    language: Optional[str] = None
    timestamps: bool = False
    context: str = ""


# ---------------------------------------------------------------------------
# Server config
# ---------------------------------------------------------------------------

@dataclass
class ServerConfig:
    """Server configuration populated from CLI flags / env vars."""

    host: str = "0.0.0.0"
    port: int = 8765
    api_keys: list[str] = field(default_factory=list)
    model: str = "Qwen/Qwen3-ASR-0.6B"
    dtype: str = "float16"
    rate_limit: int = 60
    max_file_size_mb: int = 200
    max_duration_sec: int = 1800
    max_queue_depth: int = 10
    job_ttl_sec: int = 3600


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Simple sliding-window rate limiter keyed by API key."""

    def __init__(self, max_requests: int, window_sec: float = 60.0) -> None:
        self._max = max_requests
        self._window = window_sec
        self._timestamps: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self._window
        ts = self._timestamps[key]
        # Prune expired
        self._timestamps[key] = [t for t in ts if t > cutoff]
        if len(self._timestamps[key]) >= self._max:
            return False
        self._timestamps[key].append(now)
        return True

    def retry_after(self, key: str) -> float:
        ts = self._timestamps.get(key, [])
        if not ts:
            return 0.0
        return max(0.0, ts[0] + self._window - time.monotonic())


# ---------------------------------------------------------------------------
# Shared app state
# ---------------------------------------------------------------------------

@dataclass
class _AppState:
    """Mutable state shared across the application."""

    jobs: dict[str, Job] = field(default_factory=dict)
    job_queue: Optional[asyncio.Queue] = None
    rate_limiter: Optional[_RateLimiter] = None
    start_time: float = 0.0
    api_keys_set: set[str] = field(default_factory=set)
    session: object = None  # Session instance
    config: Optional[ServerConfig] = None
    inference_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    sync_inflight: int = 0  # count of /v1 requests waiting for inference


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app(config: ServerConfig):
    """Create and return the FastAPI application.

    This factory is the main integration point. It:
    - Loads the ASR model into a Session on startup
    - Runs a background worker that processes jobs sequentially
    - Runs a background sweeper that expires old jobs
    """
    try:
        from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
    except ImportError as exc:
        raise ImportError(
            "Server dependencies not installed. "
            'Install with: pip install "mlx-qwen3-asr[serve]"'
        ) from exc

    state = _AppState(
        job_queue=asyncio.Queue(maxsize=config.max_queue_depth),
        rate_limiter=_RateLimiter(config.rate_limit),
        start_time=time.monotonic(),
        api_keys_set=set(config.api_keys),
        config=config,
    )

    # ---- Auth helper ----

    def _check_auth(request: Request) -> str:
        """Validate Bearer token and return the API key."""
        auth = request.headers.get("authorization", "")
        if not auth:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        parts = auth.split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid Authorization format")
        key = parts[1]
        if key not in state.api_keys_set:
            raise HTTPException(status_code=403, detail="Invalid API key")
        return key

    # ---- Background tasks ----

    async def _job_worker() -> None:
        """Sequential job processor — pulls from queue, runs transcription."""
        while True:
            job_id = await state.job_queue.get()
            job = state.jobs.get(job_id)
            if job is None or job.status != JobStatus.QUEUED:
                state.job_queue.task_done()
                continue

            job.status = JobStatus.PROCESSING
            try:
                result = await _run_transcription(job)
                job.result = result
                job.status = JobStatus.COMPLETED
                job.completed_at = time.time()
            except Exception as exc:
                logger.exception("Job %s failed", job_id)
                job.status = JobStatus.FAILED
                job.completed_at = time.time()
                job.error = _sanitize_error(exc)
            finally:
                _cleanup_temp(job)
                state.job_queue.task_done()

    async def _run_transcription(job: Job) -> dict:
        """Run transcription via Session.transcribe_async (thread offload)."""
        session = state.session
        async with state.inference_lock:
            result = await session.transcribe_async(
                job.temp_path,
                language=job.language,
                context=job.context,
                return_timestamps=job.timestamps,
                return_chunks=True,
            )
        return _result_to_dict(result)

    async def _ttl_sweeper() -> None:
        """Periodically remove expired jobs."""
        while True:
            await asyncio.sleep(60)
            now = time.time()
            expired = [
                jid
                for jid, j in state.jobs.items()
                if j.status in (JobStatus.COMPLETED, JobStatus.FAILED)
                and j.completed_at is not None
                and (now - j.completed_at) > config.job_ttl_sec
            ]
            for jid in expired:
                j = state.jobs.pop(jid, None)
                if j:
                    _cleanup_temp(j)

    # ---- Lifespan ----

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Loading model %s (dtype=%s)...", config.model, config.dtype)
        import mlx.core as mx

        from .session import Session

        dtype_map = {
            "float16": mx.float16,
            "float32": mx.float32,
            "bfloat16": mx.bfloat16,
        }
        dtype = dtype_map.get(config.dtype, mx.float16)
        state.session = Session(config.model, dtype=dtype)
        logger.info("Model loaded: %s", config.model)

        worker_task = asyncio.create_task(_job_worker())
        sweeper_task = asyncio.create_task(_ttl_sweeper())

        yield

        worker_task.cancel()
        sweeper_task.cancel()
        # Await cancellation to ensure cleanup runs
        for task in (worker_task, sweeper_task):
            try:
                await task
            except asyncio.CancelledError:
                pass

    app = FastAPI(
        title="mlx-qwen3-asr",
        description="Speech-to-text API powered by Qwen3-ASR on Apple Silicon",
        lifespan=lifespan,
    )

    # Store state on app for test access
    app.state.server = state

    # ---- Endpoints ----

    @app.get("/health")
    async def health() -> dict:
        queued = sum(1 for j in state.jobs.values() if j.status == JobStatus.QUEUED)
        processing = sum(
            1 for j in state.jobs.values() if j.status == JobStatus.PROCESSING
        )
        return {
            "status": "ok",
            "model": config.model,
            "dtype": config.dtype,
            "uptime_seconds": round(time.monotonic() - state.start_time),
            "queued_jobs": queued,
            "processing_jobs": processing,
            "max_queue_depth": config.max_queue_depth,
        }

    @app.post("/transcribe", status_code=202)
    async def transcribe(
        request: Request,
        audio: UploadFile = File(...),
        language: Optional[str] = Form(None),
        timestamps: Optional[str] = Form(None),
        context: Optional[str] = Form(None),
    ) -> dict:
        key = _check_auth(request)

        # Rate limit (submission only)
        if not state.rate_limiter.is_allowed(key):
            retry = state.rate_limiter.retry_after(key)
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(int(retry) + 1)},
            )

        # File size check
        contents = await audio.read()
        size_mb = len(contents) / (1024 * 1024)
        if size_mb > config.max_file_size_mb:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large ({size_mb:.1f} MB). "
                    f"Max: {config.max_file_size_mb} MB"
                ),
            )

        # Write to temp file
        suffix = Path(audio.filename or "upload.wav").suffix or ".wav"
        tmp = tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False, prefix="mlx_asr_"
        )
        try:
            tmp.write(contents)
            tmp.flush()
            tmp.close()
        except Exception:
            tmp.close()
            Path(tmp.name).unlink(missing_ok=True)
            raise

        # Create job
        job_id = f"j_{uuid.uuid4().hex}"
        job = Job(
            job_id=job_id,
            status=JobStatus.QUEUED,
            created_at=time.time(),
            api_key=key,
            temp_path=tmp.name,
            language=language,
            timestamps=_parse_bool(timestamps),
            context=context or "",
        )
        state.jobs[job_id] = job

        # Atomic enqueue with backpressure
        try:
            state.job_queue.put_nowait(job_id)
        except asyncio.QueueFull:
            # Clean up the job and temp file we just created
            state.jobs.pop(job_id, None)
            _cleanup_temp(job)
            raise HTTPException(
                status_code=503,
                detail="Server at capacity",
                headers={"Retry-After": "10"},
            )

        return {
            "job_id": job_id,
            "status": job.status.value,
            "created_at": _format_time(job.created_at),
        }

    @app.get("/jobs/{job_id}")
    async def get_job(request: Request, job_id: str) -> dict:
        key = _check_auth(request)
        # Polling does NOT count against rate limit

        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        # Enforce job ownership — only the submitting key can read the job
        if job.api_key and job.api_key != key:
            raise HTTPException(status_code=404, detail="Job not found")

        resp: dict = {
            "job_id": job.job_id,
            "status": job.status.value,
            "created_at": _format_time(job.created_at),
        }

        if job.status == JobStatus.COMPLETED:
            resp["completed_at"] = _format_time(job.completed_at)
            resp["result"] = job.result
        elif job.status == JobStatus.FAILED:
            resp["completed_at"] = _format_time(job.completed_at)
            resp["error"] = job.error

        return resp

    # ---- OpenAI-compatible endpoint ----

    @app.post("/v1/audio/transcriptions")
    async def openai_transcriptions(
        request: Request,
        file: UploadFile = File(...),
        model: Optional[str] = Form(None),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: Optional[str] = Form("json"),
        temperature: Optional[float] = Form(None),
    ):
        """OpenAI-compatible transcription endpoint.

        Synchronous — blocks until transcription completes and returns the
        result directly.  Accepts the same fields as OpenAI's
        ``POST /v1/audio/transcriptions``.
        """
        key = _check_auth(request)

        # Rate limit
        if not state.rate_limiter.is_allowed(key):
            retry = state.rate_limiter.retry_after(key)
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(int(retry) + 1)},
            )

        # Validate response_format
        fmt = (response_format or "json").strip().lower()
        if fmt not in _OPENAI_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid response_format '{response_format}'. "
                    f"Supported: {', '.join(sorted(_OPENAI_FORMATS))}"
                ),
            )

        # File size check
        contents = await file.read()
        size_mb = len(contents) / (1024 * 1024)
        if size_mb > config.max_file_size_mb:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large ({size_mb:.1f} MB). "
                    f"Max: {config.max_file_size_mb} MB"
                ),
            )

        # Write temp file
        suffix = Path(file.filename or "upload.wav").suffix or ".wav"
        tmp = tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False, prefix="mlx_asr_"
        )
        try:
            tmp.write(contents)
            tmp.flush()
            tmp.close()
        except Exception:
            tmp.close()
            Path(tmp.name).unlink(missing_ok=True)
            raise

        # Timestamps needed for verbose_json, srt, vtt
        need_timestamps = fmt in ("verbose_json", "srt", "vtt")

        # Backpressure: count queued jobs + in-flight /v1 requests
        if state.job_queue.qsize() + state.sync_inflight >= config.max_queue_depth:
            Path(tmp.name).unlink(missing_ok=True)
            raise HTTPException(
                status_code=503,
                detail="Server at capacity",
                headers={"Retry-After": "10"},
            )

        # Synchronous transcription — serialized via inference lock
        state.sync_inflight += 1
        try:
            async with state.inference_lock:
                result = await state.session.transcribe_async(
                    tmp.name,
                    language=language,
                    context=prompt or "",
                    return_timestamps=need_timestamps,
                    return_chunks=True,
                )
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=_sanitize_error(exc)
            )
        finally:
            state.sync_inflight -= 1
            Path(tmp.name).unlink(missing_ok=True)

        return _openai_format_response(result, fmt)

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result_to_dict(result: object) -> dict:
    """Convert TranscriptionResult to a JSON-serializable dict.

    Passes through the library output as-is — no translation layer.
    None fields are omitted from the response.
    """
    from .transcribe import TranscriptionResult

    if isinstance(result, TranscriptionResult):
        d = asdict(result)
    else:
        d = dict(result)  # type: ignore[arg-type]
    # Strip None values for cleaner responses
    return {k: v for k, v in d.items() if v is not None}


def _parse_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.lower() in ("true", "1", "yes")


def _format_time(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    from datetime import datetime, timezone

    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _sanitize_error(exc: Exception) -> str:
    """Return a safe error message without leaking internal paths."""
    msg = str(exc)
    # Strip common path prefixes that leak server internals
    for prefix in ("/tmp/", "/var/", "/Users/", "/home/"):
        if prefix in msg:
            msg = type(exc).__name__ + ": transcription failed"
            break
    return msg


def _cleanup_temp(job: Job) -> None:
    if job.temp_path:
        Path(job.temp_path).unlink(missing_ok=True)
        job.temp_path = None


# ---------------------------------------------------------------------------
# OpenAI-compatible response formatting
# ---------------------------------------------------------------------------

_OPENAI_FORMATS = frozenset({"json", "text", "verbose_json", "srt", "vtt"})


def _openai_format_response(result: object, fmt: str):
    """Format a TranscriptionResult as an OpenAI-compatible response."""
    from fastapi.responses import PlainTextResponse

    from .transcribe import TranscriptionResult

    if not isinstance(result, TranscriptionResult):
        return {"text": str(result)}

    if fmt == "text":
        return PlainTextResponse(result.text)

    if fmt == "json":
        return {"text": result.text}

    if fmt == "verbose_json":
        resp: dict = {
            "task": "transcribe",
            "language": (result.language or "").lower(),
            "duration": _estimate_duration(result),
            "text": result.text,
        }
        if result.segments:
            resp["words"] = [
                {"word": s["text"], "start": s["start"], "end": s["end"]}
                for s in result.segments
            ]
        if result.chunks:
            resp["segments"] = [
                {
                    "id": c.get("chunk_index", i),
                    "start": c["start"],
                    "end": c["end"],
                    "text": c["text"],
                }
                for i, c in enumerate(result.chunks)
            ]
        return resp

    # srt / vtt — need subtitle grouping
    from .writers import group_subtitle_segments

    if not result.segments:
        if fmt == "vtt":
            return PlainTextResponse("WEBVTT\n\n", media_type="text/plain")
        return PlainTextResponse("", media_type="text/plain")

    grouped = group_subtitle_segments(
        result.segments, language=result.language or ""
    )

    if fmt == "srt":
        return PlainTextResponse(_format_srt(grouped), media_type="text/plain")

    return PlainTextResponse(_format_vtt(grouped), media_type="text/plain")


def _estimate_duration(result: object) -> float:
    """Estimate audio duration from transcription result timestamps."""
    segments = getattr(result, "segments", None)
    if segments:
        return max((s.get("end", 0.0) for s in segments), default=0.0)
    chunks = getattr(result, "chunks", None)
    if chunks:
        return max((c.get("end", 0.0) for c in chunks), default=0.0)
    return 0.0


def _format_srt(segments: list[dict]) -> str:
    """Format grouped subtitle segments as an SRT string."""
    lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_ts_srt(seg['start'])} --> {_ts_srt(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def _format_vtt(segments: list[dict]) -> str:
    """Format grouped subtitle segments as a WebVTT string."""
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{_ts_vtt(seg['start'])} --> {_ts_vtt(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def _ts_srt(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ts_vtt(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# ---------------------------------------------------------------------------
# Entry point (called from CLI)
# ---------------------------------------------------------------------------

def _validate_config(config: ServerConfig) -> None:
    """Validate server config at startup. Raises SystemExit on invalid values."""
    errors: list[str] = []
    if not config.api_keys:
        errors.append("at least one API key is required (--api-key or MLX_ASR_API_KEY)")
    if config.rate_limit < 1:
        errors.append("--rate-limit must be >= 1")
    if config.max_queue_depth < 1:
        errors.append("--max-queue-depth must be >= 1")
    if config.max_file_size_mb < 1:
        errors.append("--max-file-size must be >= 1")
    if config.max_duration_sec < 1:
        errors.append("--max-duration must be >= 1")
    if config.job_ttl_sec < 1:
        errors.append("--job-ttl must be >= 1")
    if config.port < 1 or config.port > 65535:
        errors.append("--port must be between 1 and 65535")
    if errors:
        raise SystemExit("Error: " + "; ".join(errors))


def run_server(config: ServerConfig) -> None:
    """Start the server with uvicorn."""
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "Server dependencies not installed. "
            'Install with: pip install "mlx-qwen3-asr[serve]"'
        ) from exc

    _validate_config(config)

    app = create_app(config)
    logger.info(
        "Starting server on %s:%d (model=%s, max_queue=%d)",
        config.host, config.port, config.model, config.max_queue_depth,
    )
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")
