#!/usr/bin/env python3
"""Benchmark rolling streaming latency for mlx-qwen3-asr."""
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
import numpy as np

from mlx_qwen3_asr import load_audio, load_model
from mlx_qwen3_asr.streaming import (
    feed_audio,
    finish_streaming,
    init_streaming,
    streaming_metrics,
)


def _dtype_from_name(name: str) -> mx.Dtype:
    mapping = {
        "float16": mx.float16,
        "float32": mx.float32,
        "bfloat16": mx.bfloat16,
    }
    return mapping[name]


def _chunk_audio(audio: np.ndarray, chunk_size_samples: int) -> list[np.ndarray]:
    chunks: list[np.ndarray] = []
    for i in range(0, len(audio), chunk_size_samples):
        chunks.append(audio[i : i + chunk_size_samples])
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark streaming decode latency.")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32", "bfloat16"],
        default="float16",
    )
    parser.add_argument("--chunk-size-sec", type=float, default=2.0)
    parser.add_argument("--max-context-sec", type=float, default=30.0)
    parser.add_argument(
        "--endpointing-mode",
        choices=["fixed", "energy"],
        default="fixed",
        help="Chunk endpointing strategy (default: fixed)",
    )
    parser.add_argument("--unfixed-chunk-num", type=int, default=2)
    parser.add_argument("--unfixed-token-num", type=int, default=5)
    parser.add_argument(
        "--finalization-mode",
        choices=["accuracy", "latency"],
        default="accuracy",
        help="Finish policy: accuracy runs tail refinement fallback; latency skips it.",
    )
    parser.add_argument(
        "--disable-tail-refine",
        action="store_true",
        help="Deprecated alias for --finalization-mode latency",
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    dtype = _dtype_from_name(args.dtype)
    model, _ = load_model(args.model, dtype=dtype)

    audio_np = np.array(load_audio(str(audio_path)), dtype=np.float32)
    duration_sec = len(audio_np) / 16000.0
    chunk_size_samples = max(1, int(args.chunk_size_sec * 16000))
    chunks = _chunk_audio(audio_np, chunk_size_samples)

    def run_once() -> tuple[float, list[float], float, dict[str, float | int]]:
        finalization_mode = "latency" if args.disable_tail_refine else args.finalization_mode
        state = init_streaming(
            model=args.model,
            unfixed_chunk_num=args.unfixed_chunk_num,
            unfixed_token_num=args.unfixed_token_num,
            chunk_size_sec=args.chunk_size_sec,
            max_context_sec=args.max_context_sec,
            sample_rate=16000,
            endpointing_mode=args.endpointing_mode,
            finalization_mode=finalization_mode,
        )

        chunk_latencies: list[float] = []
        started = time.perf_counter()
        for chunk in chunks:
            t0 = time.perf_counter()
            feed_audio(chunk, state, model=model)
            chunk_latencies.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        finish_streaming(state, model=model)
        finish_latency = time.perf_counter() - t1
        total = time.perf_counter() - started
        return total, chunk_latencies, finish_latency, streaming_metrics(state)

    for _ in range(max(0, args.warmup_runs)):
        run_once()

    total_latencies: list[float] = []
    per_chunk_means: list[float] = []
    per_chunk_p95: list[float] = []
    finish_latencies: list[float] = []
    partial_stabilities: list[float] = []
    rewrite_rates: list[float] = []
    finalization_deltas: list[int] = []

    for _ in range(max(1, args.runs)):
        total, chunk_latencies, finish_latency, quality = run_once()
        total_latencies.append(total)
        finish_latencies.append(finish_latency)
        per_chunk_means.append(statistics.mean(chunk_latencies) if chunk_latencies else 0.0)
        per_chunk_p95.append(
            float(np.percentile(chunk_latencies, 95)) if chunk_latencies else 0.0
        )
        partial_stabilities.append(float(quality["partial_stability"]))
        rewrite_rates.append(float(quality["rewrite_rate"]))
        finalization_deltas.append(int(quality["finalization_delta_chars"]))

    mean_total = statistics.mean(total_latencies)
    payload = {
        "audio_path": str(audio_path),
        "audio_duration_sec": duration_sec,
        "model": args.model,
        "dtype": args.dtype,
        "chunk_size_sec": args.chunk_size_sec,
        "max_context_sec": args.max_context_sec,
        "endpointing_mode": args.endpointing_mode,
        "unfixed_chunk_num": args.unfixed_chunk_num,
        "unfixed_token_num": args.unfixed_token_num,
        "finalization_mode": "latency" if args.disable_tail_refine else args.finalization_mode,
        "enable_tail_refine": not args.disable_tail_refine
        and args.finalization_mode == "accuracy",
        "num_chunks": len(chunks),
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "latency_sec": {
            "total_mean": mean_total,
            "total_median": statistics.median(total_latencies),
            "total_min": min(total_latencies),
            "total_max": max(total_latencies),
            "chunk_mean_mean": statistics.mean(per_chunk_means),
            "chunk_p95_mean": statistics.mean(per_chunk_p95),
            "finish_mean": statistics.mean(finish_latencies),
        },
        "streaming_quality": {
            "partial_stability_mean": statistics.mean(partial_stabilities),
            "rewrite_rate_mean": statistics.mean(rewrite_rates),
            "finalization_delta_chars_mean": statistics.mean(finalization_deltas),
            "finalization_delta_chars_max": max(finalization_deltas) if finalization_deltas else 0,
        },
        "rtf": mean_total / duration_sec if duration_sec > 0 else 0.0,
    }

    print(json.dumps(payload, indent=2))

    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
