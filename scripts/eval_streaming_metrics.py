#!/usr/bin/env python3
"""Evaluate streaming stability/rollback diagnostics on an audio file."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
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
    return [audio[i : i + chunk_size_samples] for i in range(0, len(audio), chunk_size_samples)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate streaming quality diagnostics.")
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
    )
    parser.add_argument("--unfixed-chunk-num", type=int, default=2)
    parser.add_argument("--unfixed-token-num", type=int, default=5)
    parser.add_argument(
        "--finalization-mode",
        choices=["accuracy", "latency"],
        default="accuracy",
    )
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    dtype = _dtype_from_name(args.dtype)
    model, _ = load_model(args.model, dtype=dtype)
    audio_np = np.array(load_audio(str(audio_path)), dtype=np.float32)
    chunk_size_samples = max(1, int(args.chunk_size_sec * 16000))
    chunks = _chunk_audio(audio_np, chunk_size_samples)

    state = init_streaming(
        model=args.model,
        unfixed_chunk_num=args.unfixed_chunk_num,
        unfixed_token_num=args.unfixed_token_num,
        chunk_size_sec=args.chunk_size_sec,
        max_context_sec=args.max_context_sec,
        sample_rate=16000,
        endpointing_mode=args.endpointing_mode,
        finalization_mode=args.finalization_mode,
    )

    per_chunk: list[dict[str, float | int | bool]] = []
    for i, chunk in enumerate(chunks, start=1):
        prev_chunks_processed = int(state.chunk_id)
        feed_audio(chunk, state, model=model)
        item = dict(streaming_metrics(state))
        item["chunk_index"] = i
        item["decoded"] = int(state.chunk_id) > prev_chunks_processed
        per_chunk.append(item)

    finish_streaming(state, model=model)
    final_metrics = streaming_metrics(state)

    payload = {
        "audio_path": str(audio_path),
        "audio_duration_sec": len(audio_np) / 16000.0,
        "model": args.model,
        "dtype": args.dtype,
        "chunk_size_sec": args.chunk_size_sec,
        "max_context_sec": args.max_context_sec,
        "endpointing_mode": args.endpointing_mode,
        "unfixed_chunk_num": args.unfixed_chunk_num,
        "unfixed_token_num": args.unfixed_token_num,
        "finalization_mode": args.finalization_mode,
        "num_chunks": len(chunks),
        "per_chunk_metrics": per_chunk,
        "final_metrics": final_metrics,
        "final_text": state.text,
        "final_language": state.language,
    }

    print(json.dumps(payload, indent=2))
    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
