#!/usr/bin/env python3
"""Run a bounded parity matrix on a fixed non-near mismatch subset."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Optional

import _repo_path  # noqa: F401
import mlx.core as mx
import numpy as np

from mlx_qwen3_asr.generate import GenerationConfig, generate
from mlx_qwen3_asr.load_models import load_model

EOS_IDS = {151643, 151645}
DEFAULT_NONNEAR_SAMPLE_IDS = [
    "en_us-394135166243682296",
    "ja_jp-8922261396806111795",
    "de_de-11906634980733046933",
    "ar_eg-14219101187915533421",
    "hi_in-17876469696694013955",
]


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    language: Optional[str]
    audio_path: Path


@dataclass(frozen=True)
class PreparedSample:
    sample_id: str
    language: Optional[str]
    input_ids_np: np.ndarray
    input_features_np: np.ndarray
    feature_lens_np: np.ndarray
    ref_tokens: list[int]


@dataclass(frozen=True)
class CaseConfig:
    name: str
    dtype: str = "float16"
    force_dense_windows: bool = False
    force_segmented_windows: bool = False
    force_attention_fallback: bool = False


def _dtype_from_name(name: str) -> mx.Dtype:
    mapping = {
        "float16": mx.float16,
        "float32": mx.float32,
        "bfloat16": mx.bfloat16,
    }
    return mapping[name]


def _parse_manifest(path: Path) -> dict[str, ManifestRow]:
    out: dict[str, ManifestRow] = {}
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        row = line.strip()
        if not row:
            continue
        obj = json.loads(row)
        sample_id = str(obj.get("sample_id", f"manifest-{i:05d}"))
        out[sample_id] = ManifestRow(
            sample_id=sample_id,
            language=obj.get("language"),
            audio_path=Path(obj["audio_path"]).expanduser().resolve(),
        )
    return out


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "soundfile is required for parity matrix. "
            "Install with: pip install 'mlx-qwen3-asr[eval]'"
        ) from exc

    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1).astype(np.float32)
    else:
        audio = audio.astype(np.float32)
    return audio, int(sr)


def _first_mismatch(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return -1


def _trim_eos(tokens: list[int]) -> list[int]:
    out = list(tokens)
    while out and out[-1] in EOS_IDS:
        out.pop()
    return out


def _prepare_samples(
    *,
    ref: Any,
    manifest: dict[str, ManifestRow],
    sample_ids: list[str],
    max_new_tokens: int,
) -> list[PreparedSample]:
    import torch

    prepared: list[PreparedSample] = []
    for sample_id in sample_ids:
        row = manifest.get(sample_id)
        if row is None:
            raise KeyError(f"Sample '{sample_id}' not found in manifest.")

        audio, sr = _read_audio(row.audio_path)
        prompt = ref._build_text_prompt(context="", force_language=row.language)  # noqa: SLF001
        inputs = ref.processor(
            text=[prompt],
            audio=[audio],
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(ref.model.device).to(ref.model.dtype)

        with torch.no_grad():
            ref_out = ref.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        prompt_len = int(inputs["input_ids"].shape[1])
        ref_tokens = _trim_eos([int(x) for x in ref_out.sequences[0, prompt_len:].tolist()])

        prepared.append(
            PreparedSample(
                sample_id=sample_id,
                language=row.language,
                input_ids_np=inputs["input_ids"].cpu().numpy().astype(np.int32),
                input_features_np=inputs["input_features"].cpu().numpy().astype(np.float32),
                feature_lens_np=inputs["feature_attention_mask"].sum(-1).cpu().numpy().astype(np.int32),
                ref_tokens=ref_tokens,
            )
        )
    return prepared


def _with_encoder_window_mode(
    *,
    force_dense_windows: bool,
    force_segmented_windows: bool,
    fn: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    import mlx_qwen3_asr.encoder as enc

    original = enc._WINDOWED_SEGMENT_MIN_WINDOWS
    try:
        if force_dense_windows:
            enc._WINDOWED_SEGMENT_MIN_WINDOWS = 10**9
        elif force_segmented_windows:
            enc._WINDOWED_SEGMENT_MIN_WINDOWS = 1
        return fn()
    finally:
        enc._WINDOWED_SEGMENT_MIN_WINDOWS = original


def _with_attention_mode(
    *,
    force_attention_fallback: bool,
    fn: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    original = getattr(mx.fast, "scaled_dot_product_attention", None)
    if not force_attention_fallback or original is None:
        return fn()

    def _raise(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise TypeError("forced fallback path for matrix probe")

    try:
        mx.fast.scaled_dot_product_attention = _raise
        return fn()
    finally:
        mx.fast.scaled_dot_product_attention = original


def _run_case(
    *,
    case: CaseConfig,
    prepared_samples: list[PreparedSample],
    model_id: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    dtype = _dtype_from_name(case.dtype)

    def _execute() -> dict[str, Any]:
        model, _ = load_model(model_id, dtype=dtype)
        rows: list[dict[str, Any]] = []
        token_matches = 0
        started = time.perf_counter()

        for sample in prepared_samples:
            t0 = time.perf_counter()
            input_ids = mx.array(sample.input_ids_np)
            mel = mx.array(sample.input_features_np)
            feature_lens = mx.array(sample.feature_lens_np)
            audio_features, _ = model.audio_tower(mel.astype(dtype), feature_lens)
            seq_len = int(input_ids.shape[1])
            pos = mx.arange(seq_len)[None, :]
            position_ids = mx.stack([pos, pos, pos], axis=1)
            mlx_tokens = _trim_eos(
                generate(
                    model=model,
                    input_ids=input_ids,
                    audio_features=audio_features,
                    position_ids=position_ids,
                    config=GenerationConfig(max_new_tokens=max_new_tokens, temperature=0.0),
                )
            )
            latency = time.perf_counter() - t0
            match = mlx_tokens == sample.ref_tokens
            if match:
                token_matches += 1
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "language": sample.language,
                    "token_match": match,
                    "first_mismatch_index": _first_mismatch(mlx_tokens, sample.ref_tokens),
                    "token_count_mlx": len(mlx_tokens),
                    "token_count_ref": len(sample.ref_tokens),
                    "latency_sec_mlx": latency,
                }
            )

        mismatches = [r for r in rows if not bool(r["token_match"])]
        mismatch_indices = [
            int(r["first_mismatch_index"])
            for r in mismatches
            if int(r["first_mismatch_index"]) >= 0
        ]
        mean_first_mismatch = (
            float(sum(mismatch_indices)) / float(len(mismatch_indices))
            if mismatch_indices
            else None
        )
        latency_mean = (
            float(sum(float(r["latency_sec_mlx"]) for r in rows))
            / float(max(1, len(rows)))
        )

        return {
            "case": {
                "name": case.name,
                "dtype": case.dtype,
                "force_dense_windows": case.force_dense_windows,
                "force_segmented_windows": case.force_segmented_windows,
                "force_attention_fallback": case.force_attention_fallback,
            },
            "samples": len(rows),
            "token_matches": token_matches,
            "token_match_rate": float(token_matches) / float(max(1, len(rows))),
            "mismatch_rows": len(rows) - token_matches,
            "mean_first_mismatch_index": mean_first_mismatch,
            "latency_sec_mlx_mean": latency_mean,
            "elapsed_sec": time.perf_counter() - started,
            "rows": rows,
        }

    return _with_encoder_window_mode(
        force_dense_windows=case.force_dense_windows,
        force_segmented_windows=case.force_segmented_windows,
        fn=lambda: _with_attention_mode(
            force_attention_fallback=case.force_attention_fallback,
            fn=_execute,
        ),
    )


def _default_cases() -> list[CaseConfig]:
    return [
        CaseConfig(name="baseline-f16"),
        CaseConfig(name="baseline-f32", dtype="float32"),
        CaseConfig(name="force-dense-f16", dtype="float16", force_dense_windows=True),
        CaseConfig(name="force-segmented-f16", dtype="float16", force_segmented_windows=True),
        CaseConfig(name="force-attn-fallback-f16", dtype="float16", force_attention_fallback=True),
    ]


def _render_md(payload: dict[str, Any], out_path: Path) -> None:
    cases: list[dict[str, Any]] = payload["cases"]
    lines = [
        "# Non-near Parity Matrix",
        "",
        f"- Date: {payload['date']}",
        f"- Model: `{payload['model']}`",
        f"- Source samples: `{payload['samples_source']}`",
        f"- Sample count: **{payload['sample_count']}**",
        "",
        "## Summary",
        (
            "- This is a bounded parity matrix on the fixed non-near subset "
            "to avoid open-ended tuning."
        ),
        "",
        "## Case Results",
        (
            "| case | dtype | token_match_rate | mismatch_rows | "
            "mean_first_mismatch_index | mlx_latency_mean_sec |"
        ),
        "|---|---|---:|---:|---:|---:|",
    ]
    for case in cases:
        lines.append(
            f"| {case['case']['name']} | {case['case']['dtype']} | "
            f"{case['token_match_rate']:.3f} | {case['mismatch_rows']} | "
            f"{case['mean_first_mismatch_index']} | {case['latency_sec_mlx_mean']:.3f} |"
        )
    lines += [
        "",
        "## Decision",
        "- Outcome for this run: no case improved mismatch_rows; bounded exploration stops here.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_run_date(json_output: Path) -> str:
    parts = json_output.name.split("-")
    if len(parts) >= 3 and all(p.isdigit() for p in parts[:3]):
        return "-".join(parts[:3])
    return date.today().isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run bounded non-near parity matrix.")
    parser.add_argument(
        "--manifest-jsonl",
        default="docs/benchmarks/2026-02-14-fleurs-multilingual-100-manifest.jsonl",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--md-output", default=None)
    parser.add_argument("--sample-id", action="append", default=[])
    args = parser.parse_args()

    try:
        import qwen_asr
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Non-near parity matrix requires qwen-asr and torch. "
            "Install with: pip install 'mlx-qwen3-asr[aligner]'"
        ) from exc

    sample_ids = [str(x) for x in args.sample_id] or list(DEFAULT_NONNEAR_SAMPLE_IDS)
    manifest = _parse_manifest(Path(args.manifest_jsonl).expanduser().resolve())

    ref = qwen_asr.Qwen3ASRModel.from_pretrained(
        args.model,
        dtype="float16",
        device_map="cpu",
        max_new_tokens=args.max_new_tokens,
    )
    prepared_samples = _prepare_samples(
        ref=ref,
        manifest=manifest,
        sample_ids=sample_ids,
        max_new_tokens=args.max_new_tokens,
    )

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    for case in _default_cases():
        results.append(
            _run_case(
                case=case,
                prepared_samples=prepared_samples,
                model_id=args.model,
                max_new_tokens=args.max_new_tokens,
            )
        )

    json_out = Path(args.json_output).expanduser().resolve()
    payload = {
        "suite": "nonnear-parity-matrix-v1",
        "date": _resolve_run_date(json_out),
        "model": args.model,
        "max_new_tokens": args.max_new_tokens,
        "samples_source": str(Path(args.manifest_jsonl)),
        "sample_ids": sample_ids,
        "sample_count": len(prepared_samples),
        "elapsed_sec": time.perf_counter() - started,
        "cases": results,
    }

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.md_output:
        md_out = Path(args.md_output).expanduser().resolve()
        md_out.parent.mkdir(parents=True, exist_ok=True)
        _render_md(payload, md_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
