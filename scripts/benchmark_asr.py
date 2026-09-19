#!/usr/bin/env python3
"""Minimal ASR benchmark harness for latency + RTF tracking."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import _repo_path  # noqa: F401


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark mlx-qwen3-asr on one audio file."
    )
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-ASR-0.6B",
        help="Model name/path",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32", "bfloat16"],
        default="float16",
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--language", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    dtype = _dtype_from_name(args.dtype)
    audio = load_audio(str(audio_path))
    duration_sec = float(audio.shape[0]) / 16000.0

    for _ in range(max(0, args.warmup_runs)):
        transcribe(
            str(audio_path),
            model=args.model,
            language=args.language,
            dtype=dtype,
            max_new_tokens=args.max_new_tokens,
            verbose=False,
        )

    latencies: list[float] = []
    last_text = ""
    for _ in range(max(1, args.runs)):
        t0 = time.perf_counter()
        result = transcribe(
            str(audio_path),
            model=args.model,
            language=args.language,
            dtype=dtype,
            max_new_tokens=args.max_new_tokens,
            verbose=False,
        )
        dt = time.perf_counter() - t0
        latencies.append(dt)
        last_text = result.text

    mean_sec = statistics.mean(latencies)
    median_sec = statistics.median(latencies)
    min_sec = min(latencies)
    max_sec = max(latencies)
    std_sec = statistics.pstdev(latencies) if len(latencies) > 1 else 0.0
    rtf = mean_sec / duration_sec if duration_sec > 0 else 0.0

    payload = {
        "audio_path": str(audio_path),
        "audio_duration_sec": duration_sec,
        "model": args.model,
        "dtype": args.dtype,
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "latency_sec": {
            "mean": mean_sec,
            "median": median_sec,
            "min": min_sec,
            "max": max_sec,
            "std": std_sec,
        },
        "rtf": rtf,
        "transcript_preview": last_text[:200],
    }

    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
