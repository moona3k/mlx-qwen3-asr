"""Command-line interface for mlx-qwen3-asr."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from ._version import __version__
from .config import DEFAULT_MODEL_ID
from .diarization import (
    DEFAULT_DIARIZATION_DEVICE,
    SUPPORTED_DIARIZATION_DEVICES,
)

_FFMPEG_REQUIRED_SUFFIXES = {
    ".aac",
    ".aiff",
    ".avi",
    ".flac",
    ".m4a",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".webm",
    ".wma",
}


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class _ChunkProgressPrinter:
    def __init__(self, *, enabled: bool, start_time: float):
        self._enabled = bool(enabled)
        self._start_time = float(start_time)

    def __call__(self, payload: dict) -> None:
        if not self._enabled:
            return
        event = str(payload.get("event", ""))
        if event not in {"chunk_completed", "completed"}:
            return

        elapsed = max(0.0, time.time() - self._start_time)
        progress = float(payload.get("progress", 0.0) or 0.0)
        total_chunks = int(payload.get("total_chunks", 0) or 0)
        chunk_index = int(payload.get("chunk_index", 0) or 0)
        total_audio_sec = float(payload.get("audio_duration_sec", 0.0) or 0.0)
        eta = None
        if progress > 0.0:
            eta = max(0.0, (elapsed / progress) - elapsed)

        if event == "completed":
            print(
                (
                    f"Progress: 100.0% ({_format_duration(total_audio_sec)} audio) "
                    f"in {_format_duration(elapsed)}"
                ),
                file=sys.stderr,
            )
            return

        print(
            (
                f"Progress: chunk {chunk_index}/{max(total_chunks, 1)} "
                f"({progress * 100.0:.1f}%) ETA {_format_duration(eta)}"
            ),
            file=sys.stderr,
        )


def _print_languages() -> None:
    from .tokenizer import known_language_aliases

    aliases = known_language_aliases()
    print("Supported language aliases:")
    for language, values in aliases.items():
        print(f"- {language}: {', '.join(values)}")


def _ffmpeg_install_hint() -> str:
    if sys.platform == "darwin":
        return "brew install ffmpeg"
    if sys.platform.startswith("linux"):
        return "sudo apt-get update && sudo apt-get install -y ffmpeg"
    if sys.platform.startswith("win"):
        return "winget install Gyan.FFmpeg"
    return "Install ffmpeg and ensure it is available on PATH."


def _has_ffmpeg_binary() -> bool:
    return shutil.which("ffmpeg") is not None


def _has_module_spec(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def _input_likely_requires_ffmpeg(path: str) -> bool:
    suffix = Path(path).suffix.strip().lower()
    return suffix in _FFMPEG_REQUIRED_SUFFIXES


def _preflight_ffmpeg_for_inputs(audio_paths: list[str]) -> None:
    """Fail early for known ffmpeg-dependent media when ffmpeg is unavailable."""
    if _has_ffmpeg_binary():
        return
    ffmpeg_inputs = [p for p in audio_paths if _input_likely_requires_ffmpeg(p)]
    if not ffmpeg_inputs:
        return
    first = ffmpeg_inputs[0]
    print(
        (
            "Error: input media appears to require ffmpeg decoding "
            f"(example: {first})."
        ),
        file=sys.stderr,
    )
    print(
        f"Install ffmpeg and retry. Suggested command: {_ffmpeg_install_hint()}",
        file=sys.stderr,
    )
    raise SystemExit(1)


def _run_doctor() -> int:
    """Print environment diagnostics and return a shell-style exit code."""
    failures = 0
    warnings = 0

    print(f"mlx-qwen3-asr doctor (version {__version__})")
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {sys.platform}")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        print(f"[OK] ffmpeg: {ffmpeg_path}")
    else:
        warnings += 1
        print("[WARN] ffmpeg: not found on PATH (WAV input works without it)")
        print(f"       fix for mp3/m4a/mp4 and other formats: {_ffmpeg_install_hint()}")

    mlx_ok = _has_module_spec("mlx")
    if mlx_ok:
        print("[OK] mlx: installed")
    else:
        failures += 1
        print("[FAIL] mlx: not installed")
        print("       fix: pip install mlx")

    pyannote_ok = _has_module_spec("pyannote.audio")
    torch_ok = _has_module_spec("torch")
    torchcodec_ok = _has_module_spec("torchcodec")
    if pyannote_ok and torch_ok and torchcodec_ok:
        print("[OK] diarize extras: pyannote.audio + torch + torchcodec installed")
    else:
        warnings += 1
        print("[WARN] diarize extras: missing (optional)")
        print('       fix: pip install "mlx-qwen3-asr[diarize]"')

    token = (
        os.environ.get("PYANNOTE_AUTH_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or ""
    )
    if token:
        print("[OK] diarize auth token: set")
    else:
        warnings += 1
        print("[WARN] diarize auth token: not set")
        print("       needed for default pyannote/speaker-diarization-community-1")
        print("       fix: export PYANNOTE_AUTH_TOKEN=hf_...")

    if failures:
        print(f"doctor result: FAIL ({failures} failure(s), {warnings} warning(s))")
        return 1
    print(f"doctor result: OK ({warnings} warning(s))")
    return 0


def _preflight_diarization_runtime(
    device: str = DEFAULT_DIARIZATION_DEVICE,
) -> None:
    """Fail fast when the diarization backend or model access is unusable.

    Args:
        device: Device the transcription run will use, so preflight warms the
            same cache entry instead of a second one.
    """
    if not _has_module_spec("pyannote.audio"):
        print(
            "Error: --diarize requires optional dependency 'pyannote.audio'.",
            file=sys.stderr,
        )
        print(
            'Install with: pip install "mlx-qwen3-asr[diarize]"',
            file=sys.stderr,
        )
        raise SystemExit(1)
    if not _has_module_spec("torch"):
        print(
            "Error: --diarize requires PyTorch via pyannote dependencies.",
            file=sys.stderr,
        )
        print(
            'Install with: pip install "mlx-qwen3-asr[diarize]"',
            file=sys.stderr,
        )
        raise SystemExit(1)

    token = (
        os.environ.get("PYANNOTE_AUTH_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or ""
    )
    if not token:
        print(
            (
                "Info: the default pyannote/speaker-diarization-community-1 "
                "model requires accepting Hugging Face terms and setting "
                "PYANNOTE_AUTH_TOKEN (or HF_TOKEN), unless PYANNOTE_MODEL_ID "
                "points to a local or ungated model."
            ),
            file=sys.stderr,
        )

    try:
        _ensure_diarization_backend_ready(device)
    except (ImportError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _ensure_diarization_backend_ready(
    device: str = DEFAULT_DIARIZATION_DEVICE,
) -> None:
    """Validate diarization backend/model access before transcription starts.

    Args:
        device: Device the transcription run will use. Preflight must warm the
            same device, otherwise the cached pipeline is keyed to a different
            one and transcription pays for a second load.
    """
    # Intentional private import: we want to fail fast before spending time
    # on ASR transcription when diarization backend access is invalid.
    from .diarization import _load_pyannote_pipeline

    _load_pyannote_pipeline(device=device)


def _emit_new_stable_text(
    state_text: str,
    emitted_text: str,
) -> str:
    current = str(state_text or "")
    emitted = str(emitted_text or "")
    if not current:
        return emitted
    if current.startswith(emitted):
        delta = current[len(emitted):]
        if delta:
            print(delta, end="", flush=True)
        return current
    # Fallback when stable text gets desynced unexpectedly.
    print(f"\n{current}", end="", flush=True)
    return current


def _parse_serve_args(argv: list[str]) -> None:
    """Parse and run the ``serve`` subcommand."""
    parser = argparse.ArgumentParser(
        prog="mlx-qwen3-asr serve",
        description="Start the transcription HTTP server",
    )
    parser.add_argument(
        "--host", default=os.environ.get("MLX_ASR_HOST", "0.0.0.0"),
        help="Bind address (default: 0.0.0.0, env: MLX_ASR_HOST)",
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("MLX_ASR_PORT", "8765")),
        help="Server port (default: 8765, env: MLX_ASR_PORT)",
    )
    parser.add_argument(
        "--api-key", default=os.environ.get("MLX_ASR_API_KEY", ""),
        help="API key(s), comma-separated (env: MLX_ASR_API_KEY)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_ID,
        help=f"Model name or path (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--dtype", default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Model dtype (default: float16)",
    )
    parser.add_argument(
        "--rate-limit", type=int, default=60,
        help="Max submissions per minute per key (default: 60)",
    )
    parser.add_argument(
        "--max-file-size", type=int, default=2048,
        help="Max upload size in MB (default: 2048 / 2 GB)",
    )
    parser.add_argument(
        "--max-duration", type=int, default=28800,
        help="Max audio duration in seconds (default: 28800 / 8 hours)",
    )
    parser.add_argument(
        "--max-queue-depth", type=int, default=10,
        help="Max queued jobs before 503 (default: 10)",
    )
    parser.add_argument(
        "--job-ttl", type=int, default=3600,
        help="Seconds to keep completed jobs (default: 3600)",
    )
    args = parser.parse_args(argv)

    api_keys = [k.strip() for k in args.api_key.split(",") if k.strip()]

    from .server import ServerConfig, run_server

    config = ServerConfig(
        host=args.host,
        port=args.port,
        api_keys=api_keys,
        model=args.model,
        dtype=args.dtype,
        rate_limit=args.rate_limit,
        max_file_size_mb=args.max_file_size,
        max_duration_sec=args.max_duration,
        max_queue_depth=args.max_queue_depth,
        job_ttl_sec=args.job_ttl,
    )
    run_server(config)


def _build_parser() -> argparse.ArgumentParser:
    """Build the transcription argument parser (the ``serve`` parser is separate)."""
    parser = argparse.ArgumentParser(
        prog="mlx-qwen3-asr",
        description="Qwen3-ASR speech recognition on Apple Silicon via MLX",
    )

    parser.add_argument(
        "audio",
        nargs="*",
        help="Audio file(s) to transcribe",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help=f"Model name or path (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--context",
        default="",
        help=(
            "Domain-specific context string for the system prompt. "
            "Provide space-separated terms (e.g., '交易 停滞') to bias "
            "transcription toward domain vocabulary."
        ),
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Force language (e.g., English, Chinese)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=".",
        help="Output directory (default: current directory)",
    )
    parser.add_argument(
        "--output-format", "-f",
        default="txt",
        choices=["txt", "json", "srt", "vtt", "tsv", "all"],
        help="Output format (default: txt)",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Request word-level timestamps via native MLX forced aligner.",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help=(
            "Attach speaker labels to transcript output (offline only). "
            "Speaker segments are written by the json format; use -f json."
        ),
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Optional fixed speaker count override for --diarize.",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=1,
        help="Minimum speaker count for --diarize auto mode (default: 1).",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=8,
        help="Maximum speaker count for --diarize auto mode (default: 8).",
    )
    parser.add_argument(
        "--diarize-device",
        choices=SUPPORTED_DIARIZATION_DEVICES,
        default=DEFAULT_DIARIZATION_DEVICE,
        help=(
            "Device for the pyannote diarization pipeline "
            "(default: auto -> mps/cuda when available, else cpu)."
        ),
    )
    parser.add_argument(
        "--forced-aligner",
        default="Qwen/Qwen3-ForcedAligner-0.6B",
        help="Forced aligner model (default: Qwen/Qwen3-ForcedAligner-0.6B)",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Model dtype (default: float16)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=(
            "Maximum tokens to generate per chunk "
            "(default: adaptive 128-512 tokens, ~12 tokens/sec of audio)"
        ),
    )
    parser.add_argument(
        "--draft-model",
        default=None,
        help="Optional draft model for speculative decoding (e.g., Qwen/Qwen3-ASR-0.6B)",
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        default=4,
        help="Speculative decode draft width (default: 4)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable non-verbose progress updates",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Run experimental chunked streaming decode instead of offline transcribe",
    )
    parser.add_argument(
        "--mic",
        action="store_true",
        help="Capture and transcribe live microphone audio (experimental)",
    )
    parser.add_argument(
        "--mic-device",
        default=None,
        help="Microphone device name/index for --mic",
    )
    parser.add_argument(
        "--mic-duration-sec",
        type=float,
        default=None,
        help="Optional microphone capture duration in seconds",
    )
    parser.add_argument(
        "--mic-sample-rate",
        type=int,
        default=16000,
        help="Microphone sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--stream-chunk-sec",
        type=float,
        default=2.0,
        help="Streaming chunk size in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--stream-max-context-sec",
        type=float,
        default=30.0,
        help="Streaming max context window in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--stream-endpointing-mode",
        choices=["fixed", "energy"],
        default="fixed",
        help="Streaming chunk endpointing strategy (default: fixed)",
    )
    parser.add_argument(
        "--stream-finalization-mode",
        choices=["accuracy", "latency"],
        default="accuracy",
        help="Streaming finish policy (default: accuracy)",
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="Print known language aliases/codes and exit",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Run environment diagnostics (ffmpeg/deps/tokens) and exit.",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress transcription text on stdout",
    )
    output_group.add_argument(
        "--stdout-only",
        "--no-output-file",
        action="store_true",
        help="Print transcription to stdout without writing output files",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser


def _output_formats(args: argparse.Namespace) -> list[str]:
    if args.output_format == "all":
        return ["txt", "json", "srt", "vtt", "tsv"]
    return [args.output_format]


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject flag combinations that no mode supports. Exits with status 1."""
    if args.mic and args.audio:
        print("Error: --mic cannot be used with audio file arguments.", file=sys.stderr)
        raise SystemExit(1)
    if not args.mic and not args.audio:
        parser.error("audio is required unless --mic, --list-languages, or --doctor is used")

    wants_subtitles = any(fmt in {"srt", "vtt"} for fmt in _output_formats(args))
    checks = [
        (args.streaming and args.timestamps, "--streaming does not support --timestamps."),
        (args.streaming and args.diarize, "--streaming does not support --diarize."),
        (args.mic and args.timestamps, "--mic does not support --timestamps."),
        (args.mic and args.diarize, "--mic does not support --diarize."),
        (args.mic and args.streaming, "--mic implies streaming mode; do not pass --streaming."),
        (
            args.streaming and args.draft_model is not None,
            "--streaming does not support --draft-model yet.",
        ),
        (args.mic and args.draft_model is not None, "--mic does not support --draft-model."),
        (args.mic_sample_rate <= 0, "--mic-sample-rate must be > 0."),
        (
            args.mic_duration_sec is not None and args.mic_duration_sec <= 0,
            "--mic-duration-sec must be > 0.",
        ),
        (
            args.num_speakers is not None and args.num_speakers <= 0,
            "--num-speakers must be > 0.",
        ),
        (args.min_speakers <= 0, "--min-speakers must be > 0."),
        (args.max_speakers < args.min_speakers, "--max-speakers must be >= --min-speakers."),
        (
            wants_subtitles and (args.streaming or args.mic),
            "subtitle formats (srt/vtt) require offline transcription.",
        ),
    ]
    for failed, message in checks:
        if failed:
            print(f"Error: {message}", file=sys.stderr)
            raise SystemExit(1)


def _write_outputs(
    result,  # noqa: ANN001
    stem: str,
    formats: list[str],
    output_dir: Path,
    verbose: bool,
) -> bool:
    """Write ``result`` in every requested format. Returns True if any write failed."""
    from .writers import get_writer

    failed = False
    for fmt in formats:
        out_path = output_dir / f"{stem}.{fmt}"
        try:
            get_writer(fmt)(result, str(out_path))
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            failed = True
            continue
        if verbose:
            print(f"Written: {out_path}", file=sys.stderr)
    return failed


def _run_mic(args: argparse.Namespace, dtype):  # noqa: ANN001
    """Capture from the microphone until Ctrl+C or ``--mic-duration-sec`` elapses."""
    import numpy as np

    from .streaming import feed_audio, finish_streaming, init_streaming
    from .transcribe import TranscriptionResult

    mic_started = time.time()
    try:
        import sounddevice as sd
    except ImportError as exc:
        print(
            "Error: --mic requires the optional dependency 'sounddevice'.",
            file=sys.stderr,
        )
        print(
            "Install with: pip install sounddevice",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    if args.verbose:
        print("Listening on microphone... Press Ctrl+C to stop.", file=sys.stderr)

    chunk_samples = max(1, int(args.stream_chunk_sec * args.mic_sample_rate))
    state = init_streaming(
        model=args.model,
        context=args.context,
        chunk_size_sec=args.stream_chunk_sec,
        max_context_sec=args.stream_max_context_sec,
        sample_rate=args.mic_sample_rate,
        dtype=dtype,
        max_new_tokens=args.max_new_tokens,
        endpointing_mode=args.stream_endpointing_mode,
        finalization_mode=args.stream_finalization_mode,
        language=args.language,
    )
    emitted_stable = ""
    captured_samples = 0
    try:
        with sd.InputStream(
            samplerate=args.mic_sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_samples,
            device=args.mic_device,
        ) as stream:
            while True:
                if args.mic_duration_sec is not None:
                    elapsed = time.time() - mic_started
                    if elapsed >= args.mic_duration_sec:
                        break
                    remaining_sec = max(0.0, args.mic_duration_sec - elapsed)
                    frames = min(
                        chunk_samples,
                        max(1, int(remaining_sec * args.mic_sample_rate)),
                    )
                else:
                    frames = chunk_samples

                data, overflowed = stream.read(frames)
                if overflowed and args.verbose:
                    print("Warning: microphone overflow detected.", file=sys.stderr)
                chunk = np.asarray(data, dtype=np.float32).reshape(-1)
                captured_samples += int(chunk.shape[0])
                state = feed_audio(chunk, state)
                if not args.quiet:
                    emitted_stable = _emit_new_stable_text(state.stable_text, emitted_stable)
    except KeyboardInterrupt:
        if args.verbose:
            print("\nStopping microphone capture...", file=sys.stderr)

    state = finish_streaming(state)
    result = TranscriptionResult(
        text=state.text,
        language=state.language,
        segments=None,
        chunks=None,
    )
    if not args.quiet:
        if result.text.startswith(emitted_stable):
            tail = result.text[len(emitted_stable):]
            if tail:
                print(tail, end="", flush=True)
            print()
        else:
            print(f"\n{result.text}")
    elapsed = time.time() - mic_started
    if args.verbose:
        # Live capture runs at wall-clock speed, so a real-time factor is
        # meaningless here; report what was captured instead.
        captured_sec = captured_samples / float(args.mic_sample_rate)
        print(f"\nLanguage: {result.language}", file=sys.stderr)
        print(f"Captured: {captured_sec:.2f}s of audio", file=sys.stderr)
        print(f"Time: {elapsed:.2f}s", file=sys.stderr)
    return result


def _transcribe_file(
    args: argparse.Namespace,
    audio_path: str,
    *,
    dtype,  # noqa: ANN001
    aligner,  # noqa: ANN001
    timestamps: bool,
):
    """Transcribe one file in offline or streaming mode and return the result."""
    import numpy as np

    from .audio import SAMPLE_RATE, load_audio
    from .streaming import feed_audio, finish_streaming, init_streaming
    from .transcribe import TranscriptionResult, transcribe

    start_time = time.time()
    show_progress = not args.verbose and not args.no_progress
    if not args.streaming:
        return transcribe(
            audio=audio_path,
            model=args.model,
            draft_model=args.draft_model,
            context=args.context,
            language=args.language,
            return_timestamps=timestamps,
            diarize=args.diarize,
            diarization_num_speakers=args.num_speakers,
            diarization_min_speakers=args.min_speakers,
            diarization_max_speakers=args.max_speakers,
            diarization_device=args.diarize_device,
            return_chunks=True,
            forced_aligner=aligner,
            dtype=dtype,
            max_new_tokens=args.max_new_tokens,
            num_draft_tokens=args.num_draft_tokens,
            verbose=args.verbose,
            on_progress=_ChunkProgressPrinter(enabled=show_progress, start_time=start_time),
        )

    audio_np = np.asarray(load_audio(audio_path), dtype=np.float32)
    chunk_samples = max(1, int(args.stream_chunk_sec * SAMPLE_RATE))
    state = init_streaming(
        model=args.model,
        context=args.context,
        chunk_size_sec=args.stream_chunk_sec,
        max_context_sec=args.stream_max_context_sec,
        sample_rate=SAMPLE_RATE,
        dtype=dtype,
        max_new_tokens=args.max_new_tokens,
        endpointing_mode=args.stream_endpointing_mode,
        finalization_mode=args.stream_finalization_mode,
        language=args.language,
    )
    total_chunks = max(1, int(np.ceil(len(audio_np) / chunk_samples)))
    for i in range(0, len(audio_np), chunk_samples):
        state = feed_audio(audio_np[i : i + chunk_samples], state)
        if show_progress:
            chunk_idx = (i // chunk_samples) + 1
            progress_ratio = min(1.0, chunk_idx / total_chunks)
            elapsed = max(0.0, time.time() - start_time)
            eta = (elapsed / progress_ratio - elapsed) if progress_ratio > 0 else None
            print(
                f"Progress: chunk {chunk_idx}/{total_chunks} "
                f"({progress_ratio * 100.0:.1f}%) ETA {_format_duration(eta)}",
                file=sys.stderr,
            )
    state = finish_streaming(state)
    if show_progress:
        audio_dur = _format_duration(len(audio_np) / SAMPLE_RATE)
        elapsed_dur = _format_duration(time.time() - start_time)
        print(f"Progress: 100.0% ({audio_dur} audio) in {elapsed_dur}", file=sys.stderr)
    return TranscriptionResult(
        text=state.text,
        language=state.language,
        segments=None,
        chunks=[
            {
                "text": state.text,
                "start": 0.0,
                "end": float(len(audio_np) / SAMPLE_RATE),
                "chunk_index": 0,
                "language": state.language,
            }
        ],
    )


def main():
    """CLI entry point for mlx-qwen3-asr."""
    # Subcommands: "serve" starts the HTTP server; "transcribe" is the explicit
    # spelling of the default file-transcription mode.
    argv = sys.argv[1:]
    if argv and argv[0] == "serve":
        _parse_serve_args(argv[1:])
        return
    if argv and argv[0] == "transcribe":
        argv = argv[1:]

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.doctor:
        code = _run_doctor()
        if code != 0:
            raise SystemExit(code)
        return
    if args.list_languages:
        _print_languages()
        return

    _validate_args(parser, args)
    if args.audio and not args.mic:
        _preflight_ffmpeg_for_inputs(args.audio)
    if args.diarize and not args.streaming and not args.mic:
        _preflight_diarization_runtime(args.diarize_device)

    # Lazy imports keep --help fast.
    import mlx.core as mx

    from .forced_aligner import ForcedAligner

    dtype = {"float16": mx.float16, "float32": mx.float32, "bfloat16": mx.bfloat16}[args.dtype]
    formats = _output_formats(args)
    wants_subtitles = any(fmt in {"srt", "vtt"} for fmt in formats)
    timestamps = bool(args.timestamps or wants_subtitles or args.diarize)
    reasons = [
        reason
        for enabled, reason in (
            (wants_subtitles, "subtitle output was requested"),
            (args.diarize, "diarization was requested"),
        )
        if enabled and not args.timestamps
    ]
    if reasons:
        print(f"Info: auto-enabling timestamps because {' and '.join(reasons)}.", file=sys.stderr)
    if not Path(args.model).exists():
        print(
            f"Model: {args.model}. First run may download model files from Hugging Face "
            "(~1.2 GB for 0.6B, larger for 1.7B).",
            file=sys.stderr,
        )

    output_dir = Path(args.output_dir)
    write_output_files = not args.stdout_only
    if write_output_files:
        output_dir.mkdir(parents=True, exist_ok=True)
    aligner = (
        ForcedAligner(model_path=args.forced_aligner, dtype=dtype, backend="mlx")
        if timestamps and not args.streaming and not args.mic
        else None
    )
    had_error = False

    if args.mic:
        result = _run_mic(args, dtype)
        if write_output_files:
            stem = datetime.now().strftime("microphone-%Y%m%d-%H%M%S")
            had_error = _write_outputs(result, stem, formats, output_dir, args.verbose)
        if had_error:
            raise SystemExit(1)
        return

    for audio_path in args.audio:
        if not Path(audio_path).exists():
            print(f"Error: File not found: {audio_path}", file=sys.stderr)
            had_error = True
            continue
        if args.verbose:
            print(f"\nTranscribing: {audio_path}")
        start_time = time.time()
        try:
            result = _transcribe_file(
                args, audio_path, dtype=dtype, aligner=aligner, timestamps=timestamps
            )
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            had_error = True
            continue
        elapsed = time.time() - start_time

        if not args.quiet:
            print(result.text)
        if args.verbose:
            chunks = result.chunks or []
            duration = float(chunks[-1].get("end", 0.0)) if chunks else 0.0
            print(f"\nLanguage: {result.language}")
            print(f"Time: {elapsed:.2f}s")
            if duration > 0:
                print(f"RTF: {elapsed / duration:.4f}x")
        if write_output_files and _write_outputs(
            result, Path(audio_path).stem, formats, output_dir, args.verbose
        ):
            had_error = True

    if had_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
