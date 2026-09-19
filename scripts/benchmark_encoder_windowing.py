#!/usr/bin/env python3
"""Benchmark dense masked vs per-window encoder-layer execution."""

from __future__ import annotations

import argparse
import json
import statistics
import time

try:
    import _repo_path  # noqa: F401
except ModuleNotFoundError:  # invoked as ``python -m scripts.<name>``
    from scripts import _repo_path  # noqa: F401

import mlx.core as mx
import numpy as np

from mlx_qwen3_asr.model import (
    AudioEncoderLayer,
    _apply_windowed_encoder_layers,
    _create_windowed_mask,
)


def _run_dense(
    x: mx.array,
    layers: list[AudioEncoderLayer],
    cu_seqlens: list[int],
) -> mx.array:
    mask = _create_windowed_mask(seq_len=x.shape[1], cu_seqlens=cu_seqlens, dtype=x.dtype)
    out = x
    for layer in layers:
        out = layer(out, mask=mask)
    return out


def _run_windowed(
    x: mx.array,
    layers: list[AudioEncoderLayer],
    cu_seqlens: list[int],
) -> mx.array:
    return _apply_windowed_encoder_layers(x, layers, cu_seqlens)


def _timeit(fn, runs: int) -> tuple[float, float, float, float]:
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        y = fn()
        mx.eval(y)
        latencies.append(time.perf_counter() - t0)
    return (
        statistics.mean(latencies),
        statistics.median(latencies),
        min(latencies),
        max(latencies),
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq-len", type=int, default=832)
    p.add_argument("--window-len", type=int, default=104)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--ffn-dim", type=int, default=1024)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--runs", type=int, default=8)
    p.add_argument("--json-output", type=str, default=None)
    args = p.parse_args()

    x = mx.random.normal((1, args.seq_len, args.d_model), dtype=mx.float32)
    cu_seqlens = list(range(0, args.seq_len, args.window_len))
    if cu_seqlens[-1] != args.seq_len:
        cu_seqlens.append(args.seq_len)
    elif len(cu_seqlens) == 1:
        cu_seqlens = [0, args.seq_len]

    layers = [
        AudioEncoderLayer(args.d_model, args.num_heads, args.ffn_dim)
        for _ in range(args.layers)
    ]

    # Warmup + correctness check.
    for _ in range(args.warmup):
        mx.eval(_run_dense(x, layers, cu_seqlens))
        mx.eval(_run_windowed(x, layers, cu_seqlens))

    dense_out = _run_dense(x, layers, cu_seqlens)
    window_out = _run_windowed(x, layers, cu_seqlens)
    mx.eval(dense_out, window_out)
    dense_np = np.array(dense_out)
    window_np = np.array(window_out)
    max_abs_diff = float(np.max(np.abs(dense_np - window_np)))
    mean_abs_diff = float(np.mean(np.abs(dense_np - window_np)))

    dense_stats = _timeit(lambda: _run_dense(x, layers, cu_seqlens), runs=args.runs)
    window_stats = _timeit(lambda: _run_windowed(x, layers, cu_seqlens), runs=args.runs)

    payload = {
        "shape": {
            "batch": 1,
            "seq_len": args.seq_len,
            "window_len": args.window_len,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "ffn_dim": args.ffn_dim,
            "layers": args.layers,
            "num_windows": len(cu_seqlens) - 1,
        },
        "accuracy": {
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
        },
        "dense_mask": {
            "mean_sec": dense_stats[0],
            "median_sec": dense_stats[1],
            "min_sec": dense_stats[2],
            "max_sec": dense_stats[3],
        },
        "windowed_segments": {
            "mean_sec": window_stats[0],
            "median_sec": window_stats[1],
            "min_sec": window_stats[2],
            "max_sec": window_stats[3],
        },
        "speedup_x": dense_stats[0] / window_stats[0] if window_stats[0] > 0 else None,
    }

    print(json.dumps(payload, indent=2))
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
