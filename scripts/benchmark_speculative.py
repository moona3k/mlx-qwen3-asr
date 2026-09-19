#!/usr/bin/env python3
"""Benchmark baseline vs speculative decoding for mlx-qwen3-asr."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

try:
    import _repo_path  # noqa: F401
except ModuleNotFoundError:  # invoked as ``python -m scripts.<name>``
    from scripts import _repo_path  # noqa: F401


def _maybe_reexec_venv() -> None:
    repo = Path(__file__).resolve().parents[1]
    venv_python = repo / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    if os.environ.get("_MLX_QWEN3_ASR_REEXEC") == "1":
        return
    env = dict(os.environ)
    env["_MLX_QWEN3_ASR_REEXEC"] = "1"
    os.execve(
        str(venv_python),
        [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
        env,
    )


_maybe_reexec_venv()

import mlx.core as mx

from mlx_qwen3_asr import load_audio, transcribe


def _dtype_from_name(name: str) -> mx.Dtype:
    mapping = {
        "float16": mx.float16,
        "float32": mx.float32,
        "bfloat16": mx.bfloat16,
    }
    return mapping[name]


def _summarize(latencies: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "min": min(latencies),
        "max": max(latencies),
        "std": statistics.pstdev(latencies) if len(latencies) > 1 else 0.0,
    }


def _run_mode(
    *,
    audio_path: str,
    model: str,
    draft_model: str | None,
    num_draft_tokens: int,
    dtype: mx.Dtype,
    language: str | None,
    max_new_tokens: int,
    warmup_runs: int,
    runs: int,
) -> tuple[dict[str, float], str]:
    for _ in range(max(0, warmup_runs)):
        transcribe(
            audio_path,
            model=model,
            draft_model=draft_model,
            language=language,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            num_draft_tokens=num_draft_tokens,
            verbose=False,
        )

    latencies: list[float] = []
    last_text = ""
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        result = transcribe(
            audio_path,
            model=model,
            draft_model=draft_model,
            language=language,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            num_draft_tokens=num_draft_tokens,
            verbose=False,
        )
        latencies.append(time.perf_counter() - t0)
        last_text = result.text

    return _summarize(latencies), last_text


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark baseline and speculative decoding on one audio file."
    )
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-ASR-1.7B",
        help="Target model for final output (default: Qwen/Qwen3-ASR-1.7B)",
    )
    parser.add_argument(
        "--draft-model",
        default="Qwen/Qwen3-ASR-0.6B",
        help="Draft model for speculative decoding (default: Qwen/Qwen3-ASR-0.6B)",
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        default=4,
        help="Speculative draft width (default: 4)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32", "bfloat16"],
        default="float16",
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--language", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if args.num_draft_tokens < 1:
        raise ValueError(
            f"--num-draft-tokens must be >= 1, got {args.num_draft_tokens}"
        )

    dtype = _dtype_from_name(args.dtype)
    audio = load_audio(str(audio_path))
    duration_sec = float(audio.shape[0]) / 16000.0

    baseline_stats, baseline_text = _run_mode(
        audio_path=str(audio_path),
        model=args.model,
        draft_model=None,
        num_draft_tokens=args.num_draft_tokens,
        dtype=dtype,
        language=args.language,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=args.warmup_runs,
        runs=args.runs,
    )
    speculative_stats, speculative_text = _run_mode(
        audio_path=str(audio_path),
        model=args.model,
        draft_model=args.draft_model,
        num_draft_tokens=args.num_draft_tokens,
        dtype=dtype,
        language=args.language,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=args.warmup_runs,
        runs=args.runs,
    )

    text_match = baseline_text == speculative_text
    speedup = baseline_stats["mean"] / speculative_stats["mean"]

    payload = {
        "audio_path": str(audio_path),
        "audio_duration_sec": duration_sec,
        "dtype": args.dtype,
        "target_model": args.model,
        "draft_model": args.draft_model,
        "num_draft_tokens": args.num_draft_tokens,
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "max_new_tokens": args.max_new_tokens,
        "baseline": {
            "latency_sec": baseline_stats,
            "rtf": baseline_stats["mean"] / duration_sec if duration_sec > 0 else 0.0,
        },
        "speculative": {
            "latency_sec": speculative_stats,
            "rtf": speculative_stats["mean"] / duration_sec if duration_sec > 0 else 0.0,
        },
        "speedup_mean": speedup,
        "text_match": text_match,
        "baseline_preview": baseline_text[:200],
        "speculative_preview": speculative_text[:200],
    }

    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
