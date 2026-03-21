"""Tests for mlx_qwen3_asr/server.py — transcription HTTP server."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlx_qwen3_asr.server import (
    Job,
    JobStatus,
    ServerConfig,
    _AppState,
    _cleanup_temp,
    _estimate_duration,
    _format_srt,
    _format_time,
    _format_vtt,
    _parse_bool,
    _RateLimiter,
    _result_to_dict,
    _sanitize_error,
    _validate_config,
    create_app,
)

# Skip all tests if fastapi/httpx not installed
fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from httpx import ASGITransport, AsyncClient  # noqa: E402

# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------


class TestParseBool:
    def test_true_values(self):
        assert _parse_bool("true") is True
        assert _parse_bool("True") is True
        assert _parse_bool("TRUE") is True
        assert _parse_bool("1") is True
        assert _parse_bool("yes") is True
        assert _parse_bool("YES") is True

    def test_false_values(self):
        assert _parse_bool("false") is False
        assert _parse_bool("0") is False
        assert _parse_bool("no") is False
        assert _parse_bool(None) is False
        assert _parse_bool("anything") is False


class TestFormatTime:
    def test_none(self):
        assert _format_time(None) is None

    def test_epoch(self):
        result = _format_time(0.0)
        assert result == "1970-01-01T00:00:00Z"

    def test_z_suffix(self):
        """Timestamps must end with Z, not +00:00."""
        result = _format_time(1710936000.0)
        assert result is not None
        assert result.endswith("Z")
        assert "+00:00" not in result


class TestResultToDict:
    def test_strips_none_fields(self):
        from mlx_qwen3_asr.transcribe import TranscriptionResult

        result = TranscriptionResult(text="hello", language="English")
        d = _result_to_dict(result)
        assert d == {"text": "hello", "language": "English"}
        assert "segments" not in d
        assert "chunks" not in d

    def test_preserves_non_none_fields(self):
        from mlx_qwen3_asr.transcribe import TranscriptionResult

        segments = [{"text": "hello", "start": 0.0, "end": 1.0}]
        result = TranscriptionResult(
            text="hello", language="English", segments=segments
        )
        d = _result_to_dict(result)
        assert d["segments"] == segments


class TestSanitizeError:
    def test_safe_message_passes_through(self):
        exc = RuntimeError("Audio is corrupted")
        assert _sanitize_error(exc) == "Audio is corrupted"

    def test_path_leak_is_sanitized(self):
        exc = RuntimeError("Failed to open /tmp/mlx_asr_abc123.wav")
        result = _sanitize_error(exc)
        assert "/tmp/" not in result
        assert "RuntimeError" in result

    def test_home_path_is_sanitized(self):
        exc = ValueError("No such file: /Users/alice/secret/model.bin")
        result = _sanitize_error(exc)
        assert "/Users/" not in result
        assert "ValueError" in result


class TestCleanupTemp:
    def test_removes_file(self, tmp_path):
        f = tmp_path / "test.wav"
        f.write_bytes(b"data")
        job = Job(
            job_id="j_test",
            status=JobStatus.COMPLETED,
            created_at=time.time(),
            temp_path=str(f),
        )
        _cleanup_temp(job)
        assert not f.exists()
        assert job.temp_path is None

    def test_handles_missing_file(self):
        job = Job(
            job_id="j_test",
            status=JobStatus.COMPLETED,
            created_at=time.time(),
            temp_path="/nonexistent/file.wav",
        )
        _cleanup_temp(job)
        assert job.temp_path is None

    def test_handles_no_temp_path(self):
        job = Job(
            job_id="j_test",
            status=JobStatus.COMPLETED,
            created_at=time.time(),
        )
        _cleanup_temp(job)


# ---------------------------------------------------------------------------
# Rate limiter tests
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_allows_under_limit(self):
        rl = _RateLimiter(max_requests=3, window_sec=60.0)
        assert rl.is_allowed("key1") is True
        assert rl.is_allowed("key1") is True
        assert rl.is_allowed("key1") is True

    def test_blocks_over_limit(self):
        rl = _RateLimiter(max_requests=2, window_sec=60.0)
        assert rl.is_allowed("key1") is True
        assert rl.is_allowed("key1") is True
        assert rl.is_allowed("key1") is False

    def test_keys_are_independent(self):
        rl = _RateLimiter(max_requests=1, window_sec=60.0)
        assert rl.is_allowed("key1") is True
        assert rl.is_allowed("key2") is True
        assert rl.is_allowed("key1") is False

    def test_retry_after_returns_positive(self):
        rl = _RateLimiter(max_requests=1, window_sec=60.0)
        rl.is_allowed("key1")
        rl.is_allowed("key1")
        retry = rl.retry_after("key1")
        assert retry > 0


# ---------------------------------------------------------------------------
# ServerConfig + validation tests
# ---------------------------------------------------------------------------


class TestServerConfig:
    def test_defaults(self):
        config = ServerConfig(api_keys=["key1"])
        assert config.host == "0.0.0.0"
        assert config.port == 8765
        assert config.rate_limit == 60
        assert config.max_file_size_mb == 200
        assert config.max_duration_sec == 1800
        assert config.max_queue_depth == 10
        assert config.job_ttl_sec == 3600

    def test_multiple_keys(self):
        config = ServerConfig(api_keys=["key1", "key2", "key3"])
        assert len(config.api_keys) == 3


class TestValidateConfig:
    def test_valid_config_passes(self):
        config = ServerConfig(api_keys=["key1"])
        _validate_config(config)  # should not raise

    def test_no_api_keys(self):
        config = ServerConfig(api_keys=[])
        with pytest.raises(SystemExit, match="at least one API key"):
            _validate_config(config)

    def test_zero_rate_limit(self):
        config = ServerConfig(api_keys=["k"], rate_limit=0)
        with pytest.raises(SystemExit, match="rate-limit"):
            _validate_config(config)

    def test_zero_queue_depth(self):
        config = ServerConfig(api_keys=["k"], max_queue_depth=0)
        with pytest.raises(SystemExit, match="max-queue-depth"):
            _validate_config(config)

    def test_negative_values(self):
        config = ServerConfig(api_keys=["k"], max_file_size_mb=-1)
        with pytest.raises(SystemExit, match="max-file-size"):
            _validate_config(config)

    def test_invalid_port(self):
        config = ServerConfig(api_keys=["k"], port=99999)
        with pytest.raises(SystemExit, match="port"):
            _validate_config(config)


# ---------------------------------------------------------------------------
# Test helpers for HTTP integration tests
# ---------------------------------------------------------------------------

def _mock_session():
    """Create a mock Session returning a fixed TranscriptionResult."""
    from mlx_qwen3_asr.transcribe import TranscriptionResult

    mock = MagicMock()
    mock.transcribe_async = AsyncMock(
        return_value=TranscriptionResult(
            text="Hello world",
            language="English",
            chunks=[{
                "text": "Hello world",
                "start": 0.0,
                "end": 2.5,
                "chunk_index": 0,
                "language": "English",
            }],
        )
    )
    return mock


async def _run_worker(s: _AppState) -> None:
    """Background worker for tests — processes jobs sequentially."""
    while True:
        job_id = await s.job_queue.get()
        job = s.jobs.get(job_id)
        if job is None or job.status != JobStatus.QUEUED:
            s.job_queue.task_done()
            continue
        job.status = JobStatus.PROCESSING
        try:
            result = await s.session.transcribe_async(
                job.temp_path,
                language=job.language,
                context=job.context,
                return_timestamps=job.timestamps,
                return_chunks=True,
            )
            job.result = _result_to_dict(result)
            job.status = JobStatus.COMPLETED
            job.completed_at = time.time()
        except Exception as exc:
            job.status = JobStatus.FAILED
            job.completed_at = time.time()
            job.error = str(exc)
        finally:
            _cleanup_temp(job)
            s.job_queue.task_done()


def _create_test_app(
    api_keys: list[str] | None = None,
    rate_limit: int = 60,
    max_file_size_mb: int = 200,
    max_queue_depth: int = 10,
    mock_session_obj: object | None = None,
):
    """Create a test app that skips model loading and uses a mock Session.

    IMPORTANT: httpx ASGITransport does not trigger lifespan events.
    For tests that need the job worker, use ``_create_test_app_with_worker``
    instead (an async context manager).
    """
    config = ServerConfig(
        api_keys=api_keys or ["testkey"],
        rate_limit=rate_limit,
        max_file_size_mb=max_file_size_mb,
        max_queue_depth=max_queue_depth,
    )
    app = create_app(config)

    # Inject mock session directly into app state (no lifespan needed)
    session = mock_session_obj or _mock_session()
    app.state.server.session = session

    return app


@asynccontextmanager
async def _create_test_app_with_worker(
    api_keys: list[str] | None = None,
    rate_limit: int = 60,
    max_file_size_mb: int = 200,
    max_queue_depth: int = 10,
    mock_session_obj: object | None = None,
):
    """Create a test app with a running background worker.

    Use as: ``async with _create_test_app_with_worker() as app: ...``
    """
    app = _create_test_app(
        api_keys=api_keys,
        rate_limit=rate_limit,
        max_file_size_mb=max_file_size_mb,
        max_queue_depth=max_queue_depth,
        mock_session_obj=mock_session_obj,
    )
    worker_task = asyncio.create_task(_run_worker(app.state.server))
    try:
        yield app
    finally:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# HTTP integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_no_auth_required():
    """Health endpoint works without auth."""
    app = _create_test_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model"] == "Qwen/Qwen3-ASR-0.6B"
        assert "queued_jobs" in data
        assert "processing_jobs" in data
        assert "max_queue_depth" in data


@pytest.mark.asyncio
async def test_transcribe_requires_auth():
    """Transcribe endpoint rejects unauthenticated requests."""
    app = _create_test_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/transcribe",
            files={"audio": ("test.wav", b"RIFF" * 10, "audio/wav")},
        )
        assert resp.status_code == 401


@pytest.mark.asyncio
async def test_transcribe_rejects_bad_key():
    """Transcribe endpoint rejects invalid API key."""
    app = _create_test_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/transcribe",
            headers={"Authorization": "Bearer wrongkey"},
            files={"audio": ("test.wav", b"RIFF" * 10, "audio/wav")},
        )
        assert resp.status_code == 403


@pytest.mark.asyncio
async def test_transcribe_rejects_oversized_file():
    """Transcribe rejects files exceeding max_file_size_mb."""
    app = _create_test_app(max_file_size_mb=1)
    big_data = b"x" * (2 * 1024 * 1024)  # 2 MB

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/transcribe",
            headers={"Authorization": "Bearer testkey"},
            files={"audio": ("test.wav", big_data, "audio/wav")},
        )
        assert resp.status_code == 413


@pytest.mark.asyncio
async def test_jobs_requires_auth():
    """Jobs endpoint rejects unauthenticated requests."""
    app = _create_test_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/jobs/j_nonexistent")
        assert resp.status_code == 401


@pytest.mark.asyncio
async def test_jobs_returns_404_for_unknown():
    """Jobs endpoint returns 404 for unknown job IDs."""
    app = _create_test_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j_nonexistent",
            headers={"Authorization": "Bearer testkey"},
        )
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_rate_limit_returns_429():
    """Rate limiter returns 429 when exceeded."""
    app = _create_test_app(rate_limit=1)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        headers = {"Authorization": "Bearer testkey"}
        audio = ("test.wav", b"RIFF" * 10, "audio/wav")

        # First request uses the rate limit
        await client.post("/transcribe", headers=headers, files={"audio": audio})
        # Second request should be rate-limited
        resp2 = await client.post(
            "/transcribe", headers=headers, files={"audio": audio}
        )
        assert resp2.status_code == 429
        assert "Retry-After" in resp2.headers


@pytest.mark.asyncio
async def test_full_job_lifecycle():
    """End-to-end: submit job, poll until completed, verify result."""
    async with _create_test_app_with_worker() as app:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            headers = {"Authorization": "Bearer testkey"}

            # Submit
            resp = await client.post(
                "/transcribe",
                headers=headers,
                files={"audio": ("test.wav", b"RIFF" * 100, "audio/wav")},
            )
            assert resp.status_code == 202
            data = resp.json()
            assert "job_id" in data
            assert data["status"] == "queued"
            job_id = data["job_id"]

            # Job ID should be full UUID length (j_ + 32 hex chars)
            assert len(job_id) == 34

            # Poll until done (max 5 seconds)
            for _ in range(50):
                resp = await client.get(f"/jobs/{job_id}", headers=headers)
                assert resp.status_code == 200
                data = resp.json()
                if data["status"] in ("completed", "failed"):
                    break
                await asyncio.sleep(0.1)

            assert data["status"] == "completed"
            assert data["result"]["text"] == "Hello world"
            assert data["result"]["language"] == "English"
            assert "completed_at" in data
            # Timestamps should use Z suffix
            assert data["completed_at"].endswith("Z")


@pytest.mark.asyncio
async def test_job_passes_language_and_timestamps():
    """Verify that language and timestamps params reach the Session."""
    from mlx_qwen3_asr.transcribe import TranscriptionResult

    session = MagicMock()
    captured_kwargs = {}

    async def capture_transcribe(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return TranscriptionResult(text="ok", language="Japanese")

    session.transcribe_async = capture_transcribe

    async with _create_test_app_with_worker(mock_session_obj=session) as app:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            headers = {"Authorization": "Bearer testkey"}
            resp = await client.post(
                "/transcribe",
                headers=headers,
                files={"audio": ("test.wav", b"RIFF" * 10, "audio/wav")},
                data={"language": "ja", "timestamps": "true", "context": "ASR test"},
            )
            assert resp.status_code == 202
            job_id = resp.json()["job_id"]

            # Wait for processing
            for _ in range(50):
                resp = await client.get(f"/jobs/{job_id}", headers=headers)
                if resp.json()["status"] in ("completed", "failed"):
                    break
                await asyncio.sleep(0.1)

    assert captured_kwargs.get("language") == "ja"
    assert captured_kwargs.get("return_timestamps") is True
    assert captured_kwargs.get("context") == "ASR test"


@pytest.mark.asyncio
async def test_backpressure_returns_503():
    """Queue full returns 503 via atomic put_nowait."""
    from mlx_qwen3_asr.transcribe import TranscriptionResult

    session = MagicMock()

    async def slow_transcribe(*args, **kwargs):
        await asyncio.sleep(10)
        return TranscriptionResult(text="done", language="English")

    session.transcribe_async = slow_transcribe

    async with _create_test_app_with_worker(
        max_queue_depth=1, mock_session_obj=session
    ) as app:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            headers = {"Authorization": "Bearer testkey"}
            audio = ("test.wav", b"RIFF" * 10, "audio/wav")

            # Fill the queue (1 processing + queue is full)
            resp1 = await client.post(
                "/transcribe", headers=headers, files={"audio": audio}
            )
            assert resp1.status_code == 202

            # Give worker time to pick it up (frees queue slot)
            await asyncio.sleep(0.1)

            # This fills the queue slot again
            resp2 = await client.post(
                "/transcribe", headers=headers, files={"audio": audio}
            )
            assert resp2.status_code == 202

            # This should get 503
            resp3 = await client.post(
                "/transcribe", headers=headers, files={"audio": audio}
            )
            assert resp3.status_code == 503
            assert "Retry-After" in resp3.headers


@pytest.mark.asyncio
async def test_multiple_api_keys():
    """Multiple API keys all work."""
    async with _create_test_app_with_worker(api_keys=["key1", "key2"]) as app:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            audio = ("test.wav", b"RIFF" * 10, "audio/wav")

            resp1 = await client.post(
                "/transcribe",
                headers={"Authorization": "Bearer key1"},
                files={"audio": audio},
            )
            assert resp1.status_code == 202

            resp2 = await client.post(
                "/transcribe",
                headers={"Authorization": "Bearer key2"},
                files={"audio": audio},
            )
            assert resp2.status_code == 202


@pytest.mark.asyncio
async def test_job_isolation_cross_key():
    """Key2 cannot read key1's job — returns 404."""
    async with _create_test_app_with_worker(api_keys=["key1", "key2"]) as app:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # key1 submits a job
            resp = await client.post(
                "/transcribe",
                headers={"Authorization": "Bearer key1"},
                files={"audio": ("test.wav", b"RIFF" * 10, "audio/wav")},
            )
            assert resp.status_code == 202
            job_id = resp.json()["job_id"]

            # key1 can read their own job
            resp = await client.get(
                f"/jobs/{job_id}",
                headers={"Authorization": "Bearer key1"},
            )
            assert resp.status_code == 200

            # key2 cannot read key1's job
            resp = await client.get(
                f"/jobs/{job_id}",
                headers={"Authorization": "Bearer key2"},
            )
            assert resp.status_code == 404


@pytest.mark.asyncio
async def test_failed_job_returns_error():
    """Failed transcription returns error in job response."""
    session = MagicMock()
    session.transcribe_async = AsyncMock(
        side_effect=RuntimeError("Audio is corrupted")
    )

    async with _create_test_app_with_worker(mock_session_obj=session) as app:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            headers = {"Authorization": "Bearer testkey"}

            resp = await client.post(
                "/transcribe",
                headers=headers,
                files={"audio": ("bad.wav", b"RIFF" * 10, "audio/wav")},
            )
            assert resp.status_code == 202
            job_id = resp.json()["job_id"]

            for _ in range(50):
                resp = await client.get(f"/jobs/{job_id}", headers=headers)
                data = resp.json()
                if data["status"] in ("completed", "failed"):
                    break
                await asyncio.sleep(0.1)

            assert data["status"] == "failed"
            assert "error" in data
            assert "completed_at" in data


@pytest.mark.asyncio
async def test_temp_file_cleaned_after_completion():
    """Temp file is deleted after job completes."""
    async with _create_test_app_with_worker() as app:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            headers = {"Authorization": "Bearer testkey"}

            resp = await client.post(
                "/transcribe",
                headers=headers,
                files={"audio": ("test.wav", b"RIFF" * 100, "audio/wav")},
            )
            job_id = resp.json()["job_id"]

            # Wait for completion
            for _ in range(50):
                resp = await client.get(f"/jobs/{job_id}", headers=headers)
                if resp.json()["status"] in ("completed", "failed"):
                    break
                await asyncio.sleep(0.1)

            # The job's temp_path should be cleaned up
            job = app.state.server.jobs[job_id]
            assert job.temp_path is None


@pytest.mark.asyncio
async def test_backpressure_cleans_up_on_503():
    """When 503 is returned, the temp file and job entry are cleaned up."""
    from mlx_qwen3_asr.transcribe import TranscriptionResult

    session = MagicMock()

    async def slow_transcribe(*args, **kwargs):
        await asyncio.sleep(10)
        return TranscriptionResult(text="done", language="English")

    session.transcribe_async = slow_transcribe

    async with _create_test_app_with_worker(
        max_queue_depth=1, mock_session_obj=session
    ) as app:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            headers = {"Authorization": "Bearer testkey"}
            audio = ("test.wav", b"RIFF" * 10, "audio/wav")

            # Fill up
            await client.post("/transcribe", headers=headers, files={"audio": audio})
            await asyncio.sleep(0.1)
            await client.post("/transcribe", headers=headers, files={"audio": audio})

            # Count jobs before 503
            jobs_before = len(app.state.server.jobs)

            # This gets 503
            resp = await client.post(
                "/transcribe", headers=headers, files={"audio": audio}
            )
            assert resp.status_code == 503

            # No extra job was left in the store
            assert len(app.state.server.jobs) == jobs_before


# ---------------------------------------------------------------------------
# CLI serve subcommand tests
# ---------------------------------------------------------------------------


def test_cli_serve_help():
    """``mlx-qwen3-asr serve --help`` exits cleanly."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "mlx_qwen3_asr.cli", "serve", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Start the transcription HTTP server" in result.stdout
    assert "--api-key" in result.stdout
    assert "--port" in result.stdout
    assert "--max-queue-depth" in result.stdout
    assert "--job-ttl" in result.stdout


def test_cli_serve_parses_all_flags():
    """Serve subcommand parses all flags into ServerConfig."""

    from mlx_qwen3_asr.cli import _parse_serve_args

    with patch("mlx_qwen3_asr.server.run_server") as mock_run:
        _parse_serve_args([
            "--host", "127.0.0.1",
            "--port", "9999",
            "--api-key", "k1,k2",
            "--model", "Qwen/Qwen3-ASR-1.7B",
            "--dtype", "bfloat16",
            "--rate-limit", "30",
            "--max-file-size", "100",
            "--max-duration", "600",
            "--max-queue-depth", "5",
            "--job-ttl", "7200",
        ])

    config = mock_run.call_args[0][0]
    assert config.host == "127.0.0.1"
    assert config.port == 9999
    assert config.api_keys == ["k1", "k2"]
    assert config.model == "Qwen/Qwen3-ASR-1.7B"
    assert config.dtype == "bfloat16"
    assert config.rate_limit == 30
    assert config.max_file_size_mb == 100
    assert config.max_duration_sec == 600
    assert config.max_queue_depth == 5
    assert config.job_ttl_sec == 7200


def test_cli_legacy_audio_arg_still_works():
    """Legacy ``mlx-qwen3-asr audio.wav`` still works."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "mlx_qwen3_asr.cli", "nonexistent.wav"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "File not found: nonexistent.wav" in result.stderr


def test_run_server_requires_api_key():
    """run_server exits if no API keys provided."""
    from mlx_qwen3_asr.server import run_server

    config = ServerConfig(api_keys=[])
    with pytest.raises(SystemExit, match="at least one API key"):
        run_server(config)


def test_run_server_rejects_zero_rate_limit():
    """run_server exits if rate_limit is 0."""
    from mlx_qwen3_asr.server import run_server

    config = ServerConfig(api_keys=["k"], rate_limit=0)
    with pytest.raises(SystemExit, match="rate-limit"):
        run_server(config)


def test_run_server_rejects_zero_queue_depth():
    """run_server exits if max_queue_depth is 0."""
    from mlx_qwen3_asr.server import run_server

    config = ServerConfig(api_keys=["k"], max_queue_depth=0)
    with pytest.raises(SystemExit, match="max-queue-depth"):
        run_server(config)


# ---------------------------------------------------------------------------
# OpenAI-compatible response helpers
# ---------------------------------------------------------------------------


class TestEstimateDuration:
    def test_from_segments(self):
        from mlx_qwen3_asr.transcribe import TranscriptionResult

        result = TranscriptionResult(
            text="hi", language="en",
            segments=[{"text": "hi", "start": 0.0, "end": 2.5}],
        )
        assert _estimate_duration(result) == 2.5

    def test_from_chunks(self):
        from mlx_qwen3_asr.transcribe import TranscriptionResult

        result = TranscriptionResult(
            text="hi", language="en",
            chunks=[{"text": "hi", "start": 0.0, "end": 5.0}],
        )
        assert _estimate_duration(result) == 5.0

    def test_no_timestamps(self):
        from mlx_qwen3_asr.transcribe import TranscriptionResult

        result = TranscriptionResult(text="hi", language="en")
        assert _estimate_duration(result) == 0.0


class TestFormatSrt:
    def test_basic(self):
        segments = [
            {"text": "Hello world", "start": 0.0, "end": 1.0},
            {"text": "How are you", "start": 1.5, "end": 2.5},
        ]
        srt = _format_srt(segments)
        assert "1\n00:00:00,000 --> 00:00:01,000\nHello world" in srt
        assert "2\n00:00:01,500 --> 00:00:02,500\nHow are you" in srt

    def test_empty(self):
        assert _format_srt([]) == ""


class TestFormatVtt:
    def test_basic(self):
        segments = [
            {"text": "Hello world", "start": 0.0, "end": 1.0},
        ]
        vtt = _format_vtt(segments)
        assert vtt.startswith("WEBVTT")
        assert "00:00:00.000 --> 00:00:01.000" in vtt
        assert "Hello world" in vtt


# ---------------------------------------------------------------------------
# OpenAI-compatible endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_json_default():
    """OpenAI compat returns JSON with text field by default."""
    app = _create_test_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer testkey"},
            files={"file": ("test.wav", b"RIFF" * 100, "audio/wav")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data == {"text": "Hello world"}


@pytest.mark.asyncio
async def test_openai_text_format():
    """OpenAI compat returns plain text when response_format=text."""
    app = _create_test_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer testkey"},
            files={"file": ("test.wav", b"RIFF" * 100, "audio/wav")},
            data={"response_format": "text"},
        )
        assert resp.status_code == 200
        assert resp.text == "Hello world"
        assert "text/plain" in resp.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_openai_verbose_json():
    """OpenAI compat verbose_json includes words and duration."""
    from mlx_qwen3_asr.transcribe import TranscriptionResult

    session = MagicMock()
    session.transcribe_async = AsyncMock(
        return_value=TranscriptionResult(
            text="Hello world",
            language="English",
            segments=[
                {"text": "Hello", "start": 0.0, "end": 0.5},
                {"text": "world", "start": 0.5, "end": 1.0},
            ],
        )
    )
    app = _create_test_app(mock_session_obj=session)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer testkey"},
            files={"file": ("test.wav", b"RIFF" * 100, "audio/wav")},
            data={"response_format": "verbose_json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["task"] == "transcribe"
        assert data["language"] == "english"
        assert data["text"] == "Hello world"
        assert data["duration"] == 1.0
        assert len(data["words"]) == 2
        assert data["words"][0] == {"word": "Hello", "start": 0.0, "end": 0.5}


@pytest.mark.asyncio
async def test_openai_srt_format():
    """OpenAI compat returns SRT subtitles."""
    from mlx_qwen3_asr.transcribe import TranscriptionResult

    session = MagicMock()
    session.transcribe_async = AsyncMock(
        return_value=TranscriptionResult(
            text="Hello world",
            language="English",
            segments=[
                {"text": "Hello", "start": 0.0, "end": 0.5},
                {"text": "world", "start": 0.5, "end": 1.0},
            ],
        )
    )
    app = _create_test_app(mock_session_obj=session)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer testkey"},
            files={"file": ("test.wav", b"RIFF" * 100, "audio/wav")},
            data={"response_format": "srt"},
        )
        assert resp.status_code == 200
        assert "text/plain" in resp.headers.get("content-type", "")
        assert "00:00:00,000" in resp.text
        assert "-->" in resp.text


@pytest.mark.asyncio
async def test_openai_vtt_format():
    """OpenAI compat returns WebVTT subtitles."""
    from mlx_qwen3_asr.transcribe import TranscriptionResult

    session = MagicMock()
    session.transcribe_async = AsyncMock(
        return_value=TranscriptionResult(
            text="Hello world",
            language="English",
            segments=[
                {"text": "Hello", "start": 0.0, "end": 0.5},
                {"text": "world", "start": 0.5, "end": 1.0},
            ],
        )
    )
    app = _create_test_app(mock_session_obj=session)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer testkey"},
            files={"file": ("test.wav", b"RIFF" * 100, "audio/wav")},
            data={"response_format": "vtt"},
        )
        assert resp.status_code == 200
        assert "WEBVTT" in resp.text
        assert "00:00:00.000" in resp.text


@pytest.mark.asyncio
async def test_openai_requires_auth():
    """OpenAI compat endpoint requires authentication."""
    app = _create_test_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", b"RIFF" * 10, "audio/wav")},
        )
        assert resp.status_code == 401


@pytest.mark.asyncio
async def test_openai_rejects_bad_key():
    """OpenAI compat endpoint rejects invalid API key."""
    app = _create_test_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer wrongkey"},
            files={"file": ("test.wav", b"RIFF" * 10, "audio/wav")},
        )
        assert resp.status_code == 403


@pytest.mark.asyncio
async def test_openai_rate_limited():
    """OpenAI compat respects rate limiting."""
    app = _create_test_app(rate_limit=1)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        headers = {"Authorization": "Bearer testkey"}
        audio = ("test.wav", b"RIFF" * 10, "audio/wav")

        # First request uses the rate limit
        await client.post(
            "/v1/audio/transcriptions", headers=headers, files={"file": audio}
        )
        # Second should be limited
        resp = await client.post(
            "/v1/audio/transcriptions", headers=headers, files={"file": audio}
        )
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers


@pytest.mark.asyncio
async def test_openai_prompt_maps_to_context():
    """OpenAI 'prompt' field maps to our 'context' parameter."""
    from mlx_qwen3_asr.transcribe import TranscriptionResult

    captured = {}

    async def capture(*args, **kwargs):
        captured.update(kwargs)
        return TranscriptionResult(text="ok", language="English")

    session = MagicMock()
    session.transcribe_async = capture

    app = _create_test_app(mock_session_obj=session)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer testkey"},
            files={"file": ("test.wav", b"RIFF" * 10, "audio/wav")},
            data={"prompt": "Medical terminology"},
        )

    assert captured.get("context") == "Medical terminology"


@pytest.mark.asyncio
async def test_openai_verbose_json_enables_timestamps():
    """verbose_json format automatically enables timestamp generation."""
    from mlx_qwen3_asr.transcribe import TranscriptionResult

    captured = {}

    async def capture(*args, **kwargs):
        captured.update(kwargs)
        return TranscriptionResult(
            text="ok", language="English",
            segments=[{"text": "ok", "start": 0.0, "end": 0.5}],
        )

    session = MagicMock()
    session.transcribe_async = capture

    app = _create_test_app(mock_session_obj=session)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer testkey"},
            files={"file": ("test.wav", b"RIFF" * 10, "audio/wav")},
            data={"response_format": "verbose_json"},
        )

    assert captured.get("return_timestamps") is True


@pytest.mark.asyncio
async def test_openai_invalid_format():
    """OpenAI compat rejects invalid response_format."""
    app = _create_test_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer testkey"},
            files={"file": ("test.wav", b"RIFF" * 10, "audio/wav")},
            data={"response_format": "xml"},
        )
        assert resp.status_code == 400


@pytest.mark.asyncio
async def test_openai_file_size_limit():
    """OpenAI compat respects file size limits."""
    app = _create_test_app(max_file_size_mb=1)
    big_data = b"x" * (2 * 1024 * 1024)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer testkey"},
            files={"file": ("test.wav", big_data, "audio/wav")},
        )
        assert resp.status_code == 413


@pytest.mark.asyncio
async def test_openai_model_field_accepted():
    """OpenAI compat accepts model field without error."""
    app = _create_test_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer testkey"},
            files={"file": ("test.wav", b"RIFF" * 100, "audio/wav")},
            data={"model": "Qwen/Qwen3-ASR-1.7B"},
        )
        assert resp.status_code == 200
        assert resp.json()["text"] == "Hello world"


@pytest.mark.asyncio
async def test_openai_500_on_transcription_failure():
    """OpenAI compat returns 500 when transcription fails."""
    session = MagicMock()
    session.transcribe_async = AsyncMock(
        side_effect=RuntimeError("Audio is corrupted")
    )
    app = _create_test_app(mock_session_obj=session)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer testkey"},
            files={"file": ("bad.wav", b"RIFF" * 10, "audio/wav")},
        )
        assert resp.status_code == 500
        assert "detail" in resp.json()


@pytest.mark.asyncio
async def test_openai_verbose_json_includes_segments():
    """verbose_json includes segments from chunks."""
    from mlx_qwen3_asr.transcribe import TranscriptionResult

    session = MagicMock()
    session.transcribe_async = AsyncMock(
        return_value=TranscriptionResult(
            text="Hello world",
            language="English",
            segments=[
                {"text": "Hello", "start": 0.0, "end": 0.5},
                {"text": "world", "start": 0.5, "end": 1.0},
            ],
            chunks=[{
                "text": "Hello world",
                "start": 0.0,
                "end": 1.0,
                "chunk_index": 0,
                "language": "English",
            }],
        )
    )
    app = _create_test_app(mock_session_obj=session)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer testkey"},
            files={"file": ("test.wav", b"RIFF" * 100, "audio/wav")},
            data={"response_format": "verbose_json"},
        )
        data = resp.json()
        assert "segments" in data
        assert data["segments"][0]["id"] == 0
        assert data["segments"][0]["text"] == "Hello world"
        assert "words" in data


@pytest.mark.asyncio
async def test_openai_backpressure_returns_503():
    """OpenAI compat returns 503 when server is at capacity."""
    from mlx_qwen3_asr.transcribe import TranscriptionResult

    session = MagicMock()

    async def slow_transcribe(*args, **kwargs):
        await asyncio.sleep(10)
        return TranscriptionResult(text="done", language="English")

    session.transcribe_async = slow_transcribe

    async with _create_test_app_with_worker(
        max_queue_depth=1, mock_session_obj=session
    ) as app:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            headers = {"Authorization": "Bearer testkey"}
            audio = ("test.wav", b"RIFF" * 10, "audio/wav")

            # Fill native queue: submit a job to occupy the worker
            resp1 = await client.post(
                "/transcribe", headers=headers, files={"audio": audio}
            )
            assert resp1.status_code == 202
            await asyncio.sleep(0.1)

            # Fill queue slot
            resp2 = await client.post(
                "/transcribe", headers=headers, files={"audio": audio}
            )
            assert resp2.status_code == 202

            # OpenAI endpoint should get 503
            resp3 = await client.post(
                "/v1/audio/transcriptions", headers=headers, files={"file": audio}
            )
            assert resp3.status_code == 503
            assert "Retry-After" in resp3.headers


@pytest.mark.asyncio
async def test_openai_verbose_json_no_segments():
    """verbose_json omits words key when no timestamps available."""
    from mlx_qwen3_asr.transcribe import TranscriptionResult

    session = MagicMock()
    session.transcribe_async = AsyncMock(
        return_value=TranscriptionResult(text="Hello", language="English")
    )
    app = _create_test_app(mock_session_obj=session)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer testkey"},
            files={"file": ("test.wav", b"RIFF" * 100, "audio/wav")},
            data={"response_format": "verbose_json"},
        )
        data = resp.json()
        assert data["task"] == "transcribe"
        assert data["text"] == "Hello"
        assert "words" not in data
        assert "segments" not in data
