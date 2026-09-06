#!/usr/bin/env python3
"""Evaluate native log-mel parity against HF WhisperFeatureExtractor (optional)."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


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

from mlx_qwen3_asr.audio import (
    SAMPLE_RATE,
    compute_features,
    load_audio,
    log_mel_spectrogram,
)


def _make_random_cases(seed: int, seconds: list[float]) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    for sec in seconds:
        n = int(sec * SAMPLE_RATE)
        out.append(rng.standard_normal(n).astype(np.float32))
    return out


def _hf_reference_features(audio_np: np.ndarray) -> tuple[np.ndarray, int]:
    """Compute reference features via HF WhisperFeatureExtractor.

    This is an optional research/evaluation lane; runtime inference no longer
    depends on `transformers`.
    """
    try:
        from transformers import WhisperFeatureExtractor
    except ImportError as exc:
        raise RuntimeError(
            "HF mel parity script requires optional dependency `transformers`. "
            "Install with: pip install transformers"
        ) from exc

    extractor = WhisperFeatureExtractor(
        feature_size=128,
        sampling_rate=SAMPLE_RATE,
        chunk_length=30,
        n_fft=400,
        hop_length=160,
    )
    result = extractor(
        audio_np.astype(np.float32),
        sampling_rate=SAMPLE_RATE,
        return_tensors="np",
        padding="do_not_pad",
        truncation=False,
        return_attention_mask=False,
    )
    mel = result["input_features"][0]  # (128, n_frames)
    return mel, int(mel.shape[-1])


def _compare_one(audio_np: np.ndarray) -> dict:
    # Current production path (custom for do_not_pad)
    mel_cur, lens_cur = compute_features(audio_np, padding="do_not_pad")

    # Explicit HF reference path (optional dependency, evaluation only).
    mel_hf, lens_hf = _hf_reference_features(audio_np)

    # Direct custom function (sanity, should match compute_features do_not_pad)
    mel_direct = np.array(log_mel_spectrogram(mx.array(audio_np)))

    cur = np.array(mel_cur[0])
    ref = np.array(mel_hf)

    return {
        "frames_cur": int(lens_cur.item()),
        "frames_hf": int(lens_hf),
        "shape_cur": list(cur.shape),
        "shape_hf": list(ref.shape),
        "direct_shape": list(mel_direct.shape),
        "mae": float(np.mean(np.abs(cur - ref))),
        "max_abs": float(np.max(np.abs(cur - ref))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate native mel parity against HF WhisperFeatureExtractor "
            "(requires optional `transformers`)."
        )
    )
    parser.add_argument(
        "--seconds",
        default="1,2,5,10,31",
        help="Comma-separated random clip durations in seconds.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixture", default="tests/fixtures/test_speech.wav")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--fail-mae-above", type=float, default=1e-5)
    parser.add_argument("--fail-max-abs-above", type=float, default=2e-4)
    args = parser.parse_args()

    seconds = [float(x.strip()) for x in args.seconds.split(",") if x.strip()]
    payload = {
        "seed": args.seed,
        "seconds": seconds,
        "fixture": args.fixture,
        "thresholds": {
            "mae": args.fail_mae_above,
            "max_abs": args.fail_max_abs_above,
        },
        "cases": [],
    }

    random_cases = _make_random_cases(args.seed, seconds)
    for sec, audio_np in zip(seconds, random_cases, strict=True):
        item = _compare_one(audio_np)
        item["case"] = f"random_{sec}s"
        payload["cases"].append(item)

    fixture_path = Path(args.fixture)
    if fixture_path.exists():
        fixture_audio = np.array(load_audio(str(fixture_path)), dtype=np.float32)
        item = _compare_one(fixture_audio)
        item["case"] = f"fixture_{fixture_path.name}"
        payload["cases"].append(item)

    maes = [c["mae"] for c in payload["cases"]]
    max_abs = [c["max_abs"] for c in payload["cases"]]
    payload["summary"] = {
        "max_mae": max(maes) if maes else 0.0,
        "max_max_abs": max(max_abs) if max_abs else 0.0,
        "passed": (max(maes) <= args.fail_mae_above if maes else True)
        and (max(max_abs) <= args.fail_max_abs_above if max_abs else True),
    }

    print(json.dumps(payload, indent=2))

    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0 if payload["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
