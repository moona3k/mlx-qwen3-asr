#!/usr/bin/env python3
"""Evaluate WER/CER quality from a JSONL manifest with references."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import _repo_path  # noqa: F401
except ModuleNotFoundError:  # invoked as ``python -m scripts.<name>``
    from scripts import _repo_path  # noqa: F401

import mlx.core as mx
import numpy as np

from mlx_qwen3_asr import load_model, transcribe

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from eval import metrics as _metrics  # noqa: E402
from eval.provenance import runtime_provenance  # noqa: E402

edit_distance = _metrics.edit_distance
_normalize_quality_text = _metrics.normalize_quality_text
_wer_tokens = _metrics.wer_tokens
_cer_tokens = _metrics.cer_tokens
_is_char_primary_language = _metrics.is_char_primary_language


@dataclass(frozen=True)
class ManifestSample:
    sample_id: str
    subset: str
    speaker_id: str
    language: Optional[str]
    audio_path: Path
    reference_text: str


def _dtype_from_name(name: str) -> mx.Dtype:
    mapping = {
        "float16": mx.float16,
        "float32": mx.float32,
        "bfloat16": mx.bfloat16,
    }
    return mapping[name]


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "soundfile is required for manifest quality evaluation. "
            "Install with: pip install 'mlx-qwen3-asr[eval]'"
        ) from exc

    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1).astype(np.float32)
    else:
        audio = audio.astype(np.float32)
    return audio, int(sr)


def _parse_manifest(path: Path) -> list[ManifestSample]:
    rows: list[ManifestSample] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        row = line.strip()
        if not row:
            continue
        obj = json.loads(row)
        audio_path = Path(obj["audio_path"]).expanduser().resolve()
        reference_text = str(obj.get("reference_text", "")).strip()
        if not reference_text:
            raise ValueError(
                f"Manifest row {i} missing non-empty reference_text: {path}"
            )
        rows.append(
            ManifestSample(
                sample_id=str(obj.get("sample_id", f"manifest-{i:05d}")),
                subset=str(obj.get("subset", "manifest")),
                speaker_id=str(obj.get("speaker_id", "unknown")),
                language=obj.get("language"),
                audio_path=audio_path,
                reference_text=reference_text,
            )
        )
    return rows


def _threshold_failures(
    *,
    wer: float,
    cer: float,
    primary_error_rate: float,
    fail_wer_above: float | None,
    fail_cer_above: float | None,
    fail_primary_above: float | None,
) -> list[str]:
    failures: list[str] = []
    if fail_wer_above is not None and wer > fail_wer_above:
        failures.append(
            f"WER regression gate failed: wer={wer:.6f} > threshold={fail_wer_above:.6f}"
        )
    if fail_cer_above is not None and cer > fail_cer_above:
        failures.append(
            f"CER regression gate failed: cer={cer:.6f} > threshold={fail_cer_above:.6f}"
        )
    if fail_primary_above is not None and primary_error_rate > fail_primary_above:
        failures.append(
            "Primary regression gate failed: "
            f"primary_error_rate={primary_error_rate:.6f} > threshold={fail_primary_above:.6f}"
        )
    return failures


def _error_rate_from_counts(errors: int, total: int) -> float:
    return float(errors) / float(max(1, total))


def _select_language_arg(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None
    value = str(language).strip()
    if not value:
        return None
    lowered = value.lower()
    if lowered in {"unknown", "auto", "none"}:
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate WER/CER from a JSONL manifest.")
    parser.add_argument("--manifest-jsonl", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32", "bfloat16"],
        default="float16",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--fail-wer-above", type=float, default=None)
    parser.add_argument("--fail-cer-above", type=float, default=None)
    parser.add_argument("--fail-primary-above", type=float, default=None)
    args = parser.parse_args()

    started = time.perf_counter()
    manifest_path = Path(args.manifest_jsonl).expanduser().resolve()
    samples = _parse_manifest(manifest_path)
    if args.limit is not None:
        samples = samples[: max(0, int(args.limit))]
    if not samples:
        raise RuntimeError(f"No samples to evaluate from manifest: {manifest_path}")

    dtype = _dtype_from_name(args.dtype)
    model, _ = load_model(args.model, dtype=dtype)

    total_wer_err = 0
    total_wer_den = 0
    total_cer_err = 0
    total_cer_den = 0
    total_primary_err = 0
    total_primary_den = 0
    latencies: list[float] = []
    rows: list[dict] = []
    by_language: dict[str, dict[str, float]] = {}

    for i, sample in enumerate(samples, start=1):
        audio, sr = _read_audio(sample.audio_path)
        language_arg = _select_language_arg(sample.language)

        t0 = time.perf_counter()
        result = transcribe(
            (audio, sr),
            model=model,
            language=language_arg,
            max_new_tokens=args.max_new_tokens,
            verbose=False,
        )
        latency = time.perf_counter() - t0
        latencies.append(latency)

        ref_norm = _normalize_quality_text(sample.reference_text)
        hyp_norm = _normalize_quality_text(result.text)
        ref_wer_tokens = _wer_tokens(ref_norm)
        hyp_wer_tokens = _wer_tokens(hyp_norm)
        ref_cer_tokens = _cer_tokens(ref_norm)
        hyp_cer_tokens = _cer_tokens(hyp_norm)

        wer_err = int(edit_distance(ref_wer_tokens, hyp_wer_tokens))
        cer_err = int(edit_distance(ref_cer_tokens, hyp_cer_tokens))
        wer_den = len(ref_wer_tokens)
        cer_den = len(ref_cer_tokens)

        total_wer_err += wer_err
        total_wer_den += wer_den
        total_cer_err += cer_err
        total_cer_den += cer_den

        char_primary = _is_char_primary_language(sample.language)
        primary_metric = "cer" if char_primary else "wer"
        primary_err = cer_err if char_primary else wer_err
        primary_den = cer_den if char_primary else wer_den
        total_primary_err += primary_err
        total_primary_den += primary_den

        lang_key = str(sample.language or "unknown")
        lang_stats = by_language.setdefault(
            lang_key,
            {
                "samples": 0,
                "wer_errors": 0,
                "wer_denominator": 0,
                "cer_errors": 0,
                "cer_denominator": 0,
                "primary_errors": 0,
                "primary_denominator": 0,
                "latency_sec_mean": 0.0,
            },
        )
        lang_stats["samples"] += 1
        lang_stats["wer_errors"] += wer_err
        lang_stats["wer_denominator"] += wer_den
        lang_stats["cer_errors"] += cer_err
        lang_stats["cer_denominator"] += cer_den
        lang_stats["primary_errors"] += primary_err
        lang_stats["primary_denominator"] += primary_den
        lang_stats["latency_sec_mean"] += latency

        rows.append(
            {
                "index": i,
                "sample_id": sample.sample_id,
                "subset": sample.subset,
                "speaker_id": sample.speaker_id,
                "language": sample.language,
                "audio_path": str(sample.audio_path),
                "duration_sec": len(audio) / float(sr),
                "reference_raw": sample.reference_text,
                "hypothesis_raw": result.text,
                "reference_normalized": ref_norm,
                "hypothesis_normalized": hyp_norm,
                "wer_errors": wer_err,
                "wer_denominator": wer_den,
                "wer": _error_rate_from_counts(wer_err, wer_den),
                "cer_errors": cer_err,
                "cer_denominator": cer_den,
                "cer": _error_rate_from_counts(cer_err, cer_den),
                "primary_metric": primary_metric,
                "primary_error_rate": _error_rate_from_counts(primary_err, primary_den),
                "latency_sec": latency,
            }
        )

    for stats in by_language.values():
        n = max(1, int(stats["samples"]))
        stats["wer"] = _error_rate_from_counts(
            int(stats["wer_errors"]), int(stats["wer_denominator"])
        )
        stats["cer"] = _error_rate_from_counts(
            int(stats["cer_errors"]), int(stats["cer_denominator"])
        )
        stats["primary_error_rate"] = _error_rate_from_counts(
            int(stats["primary_errors"]), int(stats["primary_denominator"])
        )
        stats["latency_sec_mean"] = float(stats["latency_sec_mean"]) / float(n)

    wer = _error_rate_from_counts(total_wer_err, total_wer_den)
    cer = _error_rate_from_counts(total_cer_err, total_cer_den)
    primary_error_rate = _error_rate_from_counts(total_primary_err, total_primary_den)
    payload = {
        "suite": "manifest-quality-v1",
        "manifest_jsonl": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "model": args.model,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "samples": len(samples),
        "wer": wer,
        "cer": cer,
        "primary_error_rate": primary_error_rate,
        "latency_sec_mean": float(np.mean(latencies)) if latencies else 0.0,
        "elapsed_sec": time.perf_counter() - started,
        "runtime": runtime_provenance(),
        "by_language": by_language,
        "rows": rows,
    }

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    failures = _threshold_failures(
        wer=wer,
        cer=cer,
        primary_error_rate=primary_error_rate,
        fail_wer_above=args.fail_wer_above,
        fail_cer_above=args.fail_cer_above,
        fail_primary_above=args.fail_primary_above,
    )
    if failures:
        for msg in failures:
            print(msg, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
