#!/usr/bin/env python3
"""Compare the forced aligner's MLX audio encoder against the fp32 PyTorch reference.

The forced aligner has its own audio encoder weights (``Qwen/Qwen3-ForcedAligner-0.6B``),
so encoder-parity evidence for the ASR model does not cover it. This lane measures the
encoder-output error directly, which is the measurement closest to any encoder change
(``MEM-2026-09-19-010``): word-level timestamp match rate downstream flips on
borderline fp16 argmax decisions in both directions and hides small drifts.

Per clip it records the mean/max absolute error over all audio tokens, the mean
absolute error on the last token (where tail padding differences surface first),
and the relative error against the reference magnitude.

Reference attention. ``qwen-asr`` 0.0.6 defines ``_prepare_attention_mask`` (the
block-diagonal ``n_window_infer`` mask the encoder was trained with) but never
calls it; ``cu_seqlens`` only reaches the flash-attention-2 kernel. On CPU/sdpa the
shipped reference therefore attends across the whole clip, which diverges from the
trained model once a clip exceeds one 800-frame window (8 s). The default here
(``--reference-attention windowed``) applies that mask so the reference matches
what FA2 computes; ``as-shipped`` reproduces the unmasked CPU behaviour for
comparison. Measured 2026-09-19: windowed gives 0.08% relative error on every
clip; as-shipped gives 5-13% on every clip over 8 s and 0.08% under.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import _repo_path  # noqa: F401
except ModuleNotFoundError:  # invoked as ``python -m scripts.<name>``
    from scripts import _repo_path  # noqa: F401

import mlx.core as mx
import numpy as np

from mlx_qwen3_asr.audio import compute_features, load_audio

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from eval_aligner_parity import (  # noqa: E402
    SPLIT_ARCHIVES,
    _collect_samples,
    _dtype_from_name,
    _ensure_split,
    _load_qwen_asr_reference,
)

SUITE = "aligner-encoder-parity-vs-qwen-asr-fp32"


def _git_head_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out or None


def _host_chip() -> str | None:
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return platform.machine() or None
    return out or None


def compare_encoder_outputs(mlx_out: np.ndarray, ref_out: np.ndarray) -> dict[str, float | int]:
    """Error statistics for one clip: MLX encoder output vs fp32 reference.

    Both inputs are ``(n_tokens, hidden)``. Raises ``ValueError`` on a shape
    mismatch because a token-count difference is a real divergence, not noise.
    """
    if mlx_out.shape != ref_out.shape:
        raise ValueError(
            f"Encoder output shape mismatch: mlx={mlx_out.shape} reference={ref_out.shape}"
        )
    if mlx_out.ndim != 2 or mlx_out.shape[0] == 0:
        raise ValueError(f"Expected non-empty (n_tokens, hidden) outputs, got {mlx_out.shape}")
    diff = np.abs(mlx_out.astype(np.float64) - ref_out.astype(np.float64))
    ref_mag = float(np.abs(ref_out.astype(np.float64)).mean())
    if ref_mag <= 0.0:
        raise ValueError("Reference encoder output is all zeros; cannot compute relative error")
    mae = float(diff.mean())
    return {
        "n_tokens": int(mlx_out.shape[0]),
        "hidden": int(mlx_out.shape[1]),
        "mae": mae,
        "max_abs_err": float(diff.max()),
        "last_token_mae": float(diff[-1].mean()),
        "reference_mean_abs": ref_mag,
        "relative_mae": mae / ref_mag,
    }


def summarize_rows(rows: list[dict]) -> dict[str, float | int]:
    """Aggregate per-clip statistics; means are unweighted over clips."""
    if not rows:
        return {"clips": 0}
    return {
        "clips": len(rows),
        "mae_mean": float(np.mean([r["mae"] for r in rows])),
        "mae_max": float(np.max([r["mae"] for r in rows])),
        "max_abs_err_max": float(np.max([r["max_abs_err"] for r in rows])),
        "last_token_mae_mean": float(np.mean([r["last_token_mae"] for r in rows])),
        "last_token_mae_max": float(np.max([r["last_token_mae"] for r in rows])),
        "relative_mae_mean": float(np.mean([r["relative_mae"] for r in rows])),
        "relative_mae_max": float(np.max([r["relative_mae"] for r in rows])),
    }


def threshold_failures(
    summary: dict[str, float | int],
    *,
    fail_mae_mean_above: float | None,
    fail_last_token_mae_max_above: float | None,
    fail_relative_mae_max_above: float | None,
) -> list[str]:
    failures: list[str] = []
    if not summary.get("clips"):
        return ["Aligner encoder parity: no clips were compared"]
    checks = (
        ("mae_mean", fail_mae_mean_above),
        ("last_token_mae_max", fail_last_token_mae_max_above),
        ("relative_mae_max", fail_relative_mae_max_above),
    )
    for key, threshold in checks:
        if threshold is None:
            continue
        value = float(summary[key])
        if value > threshold:
            failures.append(
                f"Aligner encoder parity gate failed: {key}={value:.6g} > threshold={threshold:.6g}"
            )
    return failures


def apply_reference_window_mask(encoder) -> None:
    """Make the PyTorch reference encoder honour ``cu_seqlens`` on CPU/sdpa.

    Wraps each encoder layer so the block-diagonal mask from the encoder's own
    ``_prepare_attention_mask`` is passed when the caller did not supply one.
    The mask is built once per encoder forward (keyed on the ``cu_seqlens``
    boundaries and sequence length) and shared across layers. Idempotent.
    """
    if getattr(encoder, "_mlx_qwen3_asr_windowed", False):
        return

    cache: dict[tuple, object] = {}

    def _mask_for(hidden_states, cu_seqlens):
        key = (int(hidden_states.shape[0]), tuple(int(v) for v in cu_seqlens.tolist()))
        if key not in cache:
            cache.clear()
            cache[key] = encoder._prepare_attention_mask(hidden_states, cu_seqlens)
        return cache[key]

    def _wrap(layer):
        original = layer.forward

        def forward(hidden_states, cu_seqlens, attention_mask=None, **kwargs):
            if attention_mask is None:
                attention_mask = _mask_for(hidden_states, cu_seqlens)
            return original(hidden_states, cu_seqlens, attention_mask=attention_mask, **kwargs)

        layer.forward = forward

    for layer in encoder.layers:
        _wrap(layer)
    encoder._mlx_qwen3_asr_windowed = True


def _reference_encoder_output(ref_backend, audio: np.ndarray) -> np.ndarray:
    import torch

    feature_extractor = ref_backend.processor.feature_extractor
    feats = feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding="do_not_pad",
        return_attention_mask=True,
    )
    with torch.inference_mode():
        out = ref_backend.model.thinker.get_audio_features(
            feats["input_features"].float(), feats["attention_mask"]
        )
    return out.to(torch.float32).cpu().numpy()


def _mlx_encoder_output(backend, audio: np.ndarray) -> np.ndarray:
    mel, feature_lens = compute_features(audio.astype(np.float32))
    features, _ = backend.model.audio_tower(mel.astype(backend.dtype), feature_lens)
    return np.array(features.astype(mx.float32))[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aligner audio-encoder output parity: MLX vs fp32 qwen-asr."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-ForcedAligner-0.6B")
    parser.add_argument(
        "--dtype", choices=["float16", "float32", "bfloat16"], default="float16"
    )
    parser.add_argument("--subset", choices=sorted(SPLIT_ARCHIVES), default="test-clean")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument(
        "--data-dir",
        default=str(Path.home() / ".cache" / "mlx-qwen3-asr" / "datasets"),
    )
    parser.add_argument(
        "--reference-attention",
        choices=["windowed", "as-shipped"],
        default="windowed",
        help=(
            "windowed: apply the reference encoder's own block-diagonal n_window_infer "
            "mask (what flash-attention-2 computes; the trained behaviour). "
            "as-shipped: leave qwen-asr's CPU/sdpa path unmasked (full attention)."
        ),
    )
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--fail-mae-mean-above", type=float, default=None)
    parser.add_argument("--fail-last-token-mae-max-above", type=float, default=None)
    parser.add_argument("--fail-relative-mae-max-above", type=float, default=None)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be >= 1")

    started = time.perf_counter()
    data_dir = Path(args.data_dir).expanduser().resolve()
    split_root = _ensure_split(data_dir, args.subset)
    selected = _collect_samples(split_root, max_samples=args.samples)
    if not selected:
        raise RuntimeError(f"No samples found under {split_root}")

    from mlx_qwen3_asr.forced_aligner import _MLXForcedAlignerBackend

    mlx_backend = _MLXForcedAlignerBackend(args.model, _dtype_from_name(args.dtype))
    ref_backend = _load_qwen_asr_reference(args.model)
    if args.reference_attention == "windowed":
        apply_reference_window_mask(ref_backend.model.thinker.audio_tower)

    import torch
    import transformers

    rows: list[dict] = []
    for sample in selected:
        audio = np.array(load_audio(str(sample.audio_path))).astype(np.float32)
        mlx_out = _mlx_encoder_output(mlx_backend, audio)
        ref_out = _reference_encoder_output(ref_backend, audio)
        stats = compare_encoder_outputs(mlx_out, ref_out)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "audio_path": str(sample.audio_path),
                "duration_sec": len(audio) / 16000.0,
                **stats,
            }
        )

    summary = summarize_rows(rows)
    repo_root = Path(__file__).resolve().parents[1]
    payload = {
        "suite": SUITE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_head_commit(repo_root),
        "model": args.model,
        "subset": args.subset,
        "mlx_dtype": args.dtype,
        "mlx_version": mx.__version__,
        "host": _host_chip(),
        "reference_stack": {
            "package": "qwen-asr",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": "cpu",
            "dtype": "float32",
            "attention": args.reference_attention,
            "attention_impl": str(
                ref_backend.model.thinker.audio_tower.config._attn_implementation
            ),
        },
        "summary": summary,
        "rows": rows,
        "elapsed_sec": time.perf_counter() - started,
    }

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    failures = threshold_failures(
        summary,
        fail_mae_mean_above=args.fail_mae_mean_above,
        fail_last_token_mae_max_above=args.fail_last_token_mae_max_above,
        fail_relative_mae_max_above=args.fail_relative_mae_max_above,
    )
    if failures:
        for msg in failures:
            print(msg, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
