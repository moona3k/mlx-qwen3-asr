#!/usr/bin/env python3
"""Probe MLX vs reference logit behavior on multilingual parity mismatches.

This script helps classify whether token mismatches are likely caused by
near-tie decoding (small top1-top2 margin) or larger model-path divergence.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import _repo_path  # noqa: F401
import mlx.core as mx
import numpy as np

from mlx_qwen3_asr.audio import load_audio
from mlx_qwen3_asr.load_models import load_model

EOS_IDS = {151643, 151645}


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    language: Optional[str]
    audio_path: Path


def _dtype_from_name(name: str) -> mx.Dtype:
    mapping = {
        "float16": mx.float16,
        "float32": mx.float32,
        "bfloat16": mx.bfloat16,
    }
    return mapping[name]


def _read_audio_path(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "soundfile is required for logit parity probe. "
            "Install with: pip install 'mlx-qwen3-asr[eval]'"
        ) from exc

    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1).astype(np.float32)
    else:
        audio = audio.astype(np.float32)
    return audio, int(sr)


def _resample_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    return np.array(load_audio((audio, sr))).astype(np.float32)


def _parse_manifest(path: Path) -> dict[str, ManifestRow]:
    out: dict[str, ManifestRow] = {}
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        row = line.strip()
        if not row:
            continue
        obj = json.loads(row)
        sample_id = str(obj.get("sample_id", f"manifest-{i:05d}"))
        audio_path = Path(obj["audio_path"]).expanduser().resolve()
        out[sample_id] = ManifestRow(
            sample_id=sample_id,
            language=obj.get("language"),
            audio_path=audio_path,
        )
    return out


def _first_mismatch(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return -1


def _trim_trailing_eos(tokens: list[int], step_infos: list[dict[str, Any]]) -> None:
    while tokens and tokens[-1] in EOS_IDS:
        tokens.pop()
        if step_infos:
            step_infos.pop()


def _topk_info(logits: mx.array, k: int) -> dict[str, Any]:
    # Use NumPy top-k for diagnostics portability across MLX versions.
    flat_np = np.asarray(logits.reshape(-1), dtype=np.float32)
    kk = max(1, min(k, int(flat_np.shape[0])))
    top_ids_np = np.argpartition(flat_np, -kk)[-kk:]
    top_ids_np = top_ids_np[np.argsort(flat_np[top_ids_np])[::-1]]
    top_ids = [int(x) for x in top_ids_np.tolist()]
    top_vals = [float(flat_np[i]) for i in top_ids]
    margin = top_vals[0] - top_vals[1] if len(top_vals) >= 2 else float("inf")
    return {
        "top_token_id": top_ids[0] if top_ids else -1,
        "top_logit": top_vals[0] if top_vals else float("-inf"),
        "margin_top1_top2": float(margin),
        "topk_token_ids": top_ids,
        "topk_logits": top_vals,
    }


def _build_decode_positions(seq_len: int, max_new_tokens: int, dtype: mx.Dtype) -> mx.array:
    next_pos_base = mx.arange(
        seq_len,
        seq_len + max(max_new_tokens - 1, 0),
        dtype=dtype,
    )
    next_pos_3d = mx.stack([next_pos_base, next_pos_base, next_pos_base], axis=0)
    return next_pos_3d[None, :, :]


def _decode_mlx_with_logits(
    *,
    model: Any,
    input_ids: mx.array,
    audio_features: mx.array,
    position_ids: mx.array,
    max_new_tokens: int,
    topk: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    max_seq_len = int(input_ids.shape[1] + max_new_tokens)
    cache = model.create_cache(max_seq_len=max_seq_len)
    logits = model.prefill(
        input_ids=input_ids,
        audio_features=audio_features,
        position_ids=position_ids,
        cache=cache,
    )

    token = int(mx.argmax(logits.reshape(-1)).item())
    generated = [token]
    step_infos = [_topk_info(logits, topk)]

    seq_len = int(input_ids.shape[1])
    next_pos_3d = _build_decode_positions(
        seq_len=seq_len,
        max_new_tokens=max_new_tokens,
        dtype=position_ids.dtype,
    )

    for step in range(1, max_new_tokens):
        if token in EOS_IDS:
            break
        next_ids = mx.array([[token]])
        next_position_ids = next_pos_3d[:, :, step - 1 : step]
        logits = model.step(
            input_ids=next_ids,
            position_ids=next_position_ids,
            cache=cache,
        )
        token = int(mx.argmax(logits.reshape(-1)).item())
        generated.append(token)
        step_infos.append(_topk_info(logits, topk))

    _trim_trailing_eos(generated, step_infos)
    return generated, step_infos


def _collect_sample_ids(
    *,
    parity_json: Optional[Path],
    requested_sample_ids: list[str],
    num_samples: int,
) -> list[str]:
    if requested_sample_ids:
        return requested_sample_ids[: max(1, num_samples)]
    if parity_json is None:
        raise ValueError("--sample-id is required when --parity-json is not provided.")

    payload = json.loads(parity_json.read_text(encoding="utf-8"))
    out: list[str] = []
    for row in payload.get("rows", []):
        if not bool(row.get("token_match", False)):
            out.append(str(row["sample_id"]))
        if len(out) >= max(1, num_samples):
            break
    if not out:
        raise RuntimeError(f"No mismatches found in parity artifact: {parity_json}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe multilingual parity logit behavior.")
    parser.add_argument("--manifest-jsonl", required=True)
    parser.add_argument("--parity-json", default=None)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32", "bfloat16"],
        default="float16",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--near-tie-margin", type=float, default=0.5)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    try:
        import qwen_asr
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Logit parity probe requires torch and qwen-asr. "
            "Install with: pip install 'mlx-qwen3-asr[aligner]'"
        ) from exc

    manifest = _parse_manifest(Path(args.manifest_jsonl).expanduser().resolve())
    parity_json = Path(args.parity_json).expanduser().resolve() if args.parity_json else None
    sample_ids = _collect_sample_ids(
        parity_json=parity_json,
        requested_sample_ids=[str(x) for x in args.sample_id],
        num_samples=max(1, args.num_samples),
    )

    dtype = _dtype_from_name(args.dtype)
    mlx_model, _ = load_model(args.model, dtype=dtype)
    ref = qwen_asr.Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="cpu",
        max_new_tokens=args.max_new_tokens,
    )

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for sample_id in sample_ids:
        info = manifest.get(sample_id)
        if info is None:
            raise KeyError(f"Sample '{sample_id}' not found in manifest.")

        audio, sr = _read_audio_path(info.audio_path)
        audio16 = _resample_16k(audio, sr)

        prompt = ref._build_text_prompt(context="", force_language=info.language)  # noqa: SLF001
        inputs = ref.processor(
            text=[prompt],
            audio=[audio16],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(ref.model.device).to(ref.model.dtype)

        with torch.no_grad():
            ref_out = ref.model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                output_scores=True,
            )

        prompt_len = int(inputs["input_ids"].shape[1])
        ref_tokens = [int(x) for x in ref_out.sequences[0, prompt_len:].tolist()]
        ref_steps = []
        if not hasattr(ref_out, "scores") or ref_out.scores is None:
            raise RuntimeError(
                "Reference generation did not return score tensors. "
                "This probe requires `scores` to compute top-k margin diagnostics."
            )
        for score in ref_out.scores:
            values, ids = torch.topk(score[0], k=min(args.topk, score.shape[-1]))
            top_ids = [int(x) for x in ids.tolist()]
            top_vals = [float(x) for x in values.tolist()]
            margin = top_vals[0] - top_vals[1] if len(top_vals) >= 2 else float("inf")
            ref_steps.append(
                {
                    "top_token_id": top_ids[0] if top_ids else -1,
                    "top_logit": top_vals[0] if top_vals else float("-inf"),
                    "margin_top1_top2": float(margin),
                    "topk_token_ids": top_ids,
                    "topk_logits": top_vals,
                }
            )
        _trim_trailing_eos(ref_tokens, ref_steps)

        input_ids = mx.array(inputs["input_ids"].cpu().numpy().astype(np.int32))
        mel = mx.array(inputs["input_features"].cpu().numpy().astype(np.float32))
        feature_lens = mx.array(
            inputs["feature_attention_mask"].sum(-1).cpu().numpy().astype(np.int32)
        )
        audio_features, _ = mlx_model.audio_tower(mel.astype(dtype), feature_lens)
        seq_len = input_ids.shape[1]
        pos = mx.arange(seq_len)[None, :]
        position_ids = mx.stack([pos, pos, pos], axis=1)
        mlx_tokens, mlx_steps = _decode_mlx_with_logits(
            model=mlx_model,
            input_ids=input_ids,
            audio_features=audio_features,
            position_ids=position_ids,
            max_new_tokens=args.max_new_tokens,
            topk=max(1, args.topk),
        )

        mismatch_idx = _first_mismatch(mlx_tokens, ref_tokens)
        probe_idx = mismatch_idx
        if probe_idx < 0:
            probe_idx = max(0, min(len(mlx_steps), len(ref_steps)) - 1)
        probe_idx = min(probe_idx, len(mlx_steps) - 1, len(ref_steps) - 1)

        mlx_probe = mlx_steps[probe_idx]
        ref_probe = ref_steps[probe_idx]
        near_tie = bool(
            mlx_probe["margin_top1_top2"] <= args.near_tie_margin
            or ref_probe["margin_top1_top2"] <= args.near_tie_margin
        )

        rows.append(
            {
                "sample_id": sample_id,
                "language": info.language,
                "audio_path": str(info.audio_path),
                "mismatch_index": mismatch_idx,
                "common_prefix_len": mismatch_idx if mismatch_idx >= 0 else len(ref_tokens),
                "ref_token_count": len(ref_tokens),
                "mlx_token_count": len(mlx_tokens),
                "ref_token_at_probe": (
                    int(ref_tokens[probe_idx]) if 0 <= probe_idx < len(ref_tokens) else None
                ),
                "mlx_token_at_probe": (
                    int(mlx_tokens[probe_idx]) if 0 <= probe_idx < len(mlx_tokens) else None
                ),
                "probe_step_index": probe_idx,
                "near_tie_margin_threshold": float(args.near_tie_margin),
                "near_tie_at_probe": near_tie,
                "ref_probe": ref_probe,
                "mlx_probe": mlx_probe,
                "ref_token_surface_at_probe": ref.processor.tokenizer.decode(
                    [ref_probe["top_token_id"]],
                    skip_special_tokens=False,
                ),
                "mlx_token_surface_at_probe": ref.processor.tokenizer.decode(
                    [mlx_probe["top_token_id"]],
                    skip_special_tokens=False,
                ),
            }
        )

    payload = {
        "suite": "logit-parity-probe-v1",
        "model": args.model,
        "dtype": args.dtype,
        "samples": len(rows),
        "near_tie_margin": float(args.near_tie_margin),
        "mismatch_rows": int(sum(1 for r in rows if int(r["mismatch_index"]) >= 0)),
        "near_tie_rows": int(sum(1 for r in rows if bool(r["near_tie_at_probe"]))),
        "elapsed_sec": time.perf_counter() - started,
        "rows": rows,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.json_output:
        out = Path(args.json_output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
