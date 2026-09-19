#!/usr/bin/env python3
"""Evaluate streaming quality over a JSONL manifest.

Emits stability/latency diagnostics for every sample and endpointing mode,
and, when the manifest carries ``reference_text``, scores ``final_text``
against the references with the same normalization as
``eval_manifest_quality.py``. The reference score is the gate that matters:
the pre-0.4.2 streaming decoder passed every stability threshold while
producing 56% primary error (``docs/EVAL_GAPS.md``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    import _repo_path  # noqa: F401
except ModuleNotFoundError:  # invoked as ``python -m scripts.<name>``
    from scripts import _repo_path  # noqa: F401

import mlx.core as mx
import numpy as np

from mlx_qwen3_asr import load_audio, load_model
from mlx_qwen3_asr.streaming import (
    feed_audio,
    finish_streaming,
    init_streaming,
    streaming_metrics,
)

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from eval.metrics import score_hypothesis  # noqa: E402

SCHEMA_VERSION = "streaming-manifest-quality-v1.2"


@dataclass(frozen=True)
class ManifestSample:
    sample_id: str
    subset: str
    speaker_id: str
    language: str | None
    audio_path: Path
    reference_text: str | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out if out else None


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dtype_from_name(name: str) -> mx.Dtype:
    mapping = {
        "float16": mx.float16,
        "float32": mx.float32,
        "bfloat16": mx.bfloat16,
    }
    return mapping[name]


def _chunk_audio(audio: np.ndarray, chunk_size_samples: int) -> list[np.ndarray]:
    return [audio[i : i + chunk_size_samples] for i in range(0, len(audio), chunk_size_samples)]


def _parse_manifest(path: Path) -> list[ManifestSample]:
    rows: list[ManifestSample] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        row = line.strip()
        if not row:
            continue
        obj = json.loads(row)
        audio_path_value = obj.get("audio_path")
        if not audio_path_value:
            raise ValueError(f"Manifest row {i} missing audio_path: {path}")
        audio_path = Path(str(audio_path_value)).expanduser().resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Manifest row {i} references missing audio: {audio_path}")
        reference_raw = obj.get("reference_text")
        reference_text = str(reference_raw).strip() if reference_raw is not None else None
        rows.append(
            ManifestSample(
                sample_id=str(obj.get("sample_id", f"manifest-{i:05d}")),
                subset=str(obj.get("subset", "manifest")),
                speaker_id=str(obj.get("speaker_id", "unknown")),
                language=(None if obj.get("language") is None else str(obj.get("language"))),
                audio_path=audio_path,
                reference_text=reference_text or None,
            )
        )
    return rows


def _load_references(manifest_path: Path) -> dict[str, tuple[str, str | None]]:
    """Map ``sample_id`` to ``(reference_text, language)`` without touching audio.

    Used to re-score committed artifacts whose rows carry ``final_text`` but
    not the reference; the manifest audio need not exist locally.
    """
    refs: dict[str, tuple[str, str | None]] = {}
    for i, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        row = line.strip()
        if not row:
            continue
        obj = json.loads(row)
        reference = str(obj.get("reference_text", "")).strip()
        if not reference:
            continue
        sample_id = str(obj.get("sample_id", f"manifest-{i:05d}"))
        language = None if obj.get("language") is None else str(obj.get("language"))
        refs[sample_id] = (reference, language)
    return refs


def _error_rate(errors: int, total: int) -> float:
    return float(errors) / float(max(1, total))


def _score_rows_against_references(
    rows: list[dict],
    references: dict[str, tuple[str, str | None]],
) -> dict[str, object] | None:
    """Score ``final_text`` per row and aggregate WER/CER/primary by mode.

    Mutates each scored row in place to add the per-sample error fields.
    Returns ``None`` when no row has a reference. Rows without a reference
    are left unscored and counted in ``unscored_rows``.
    """
    totals: dict[str, dict[str, int]] = {}
    scored = 0
    unscored = 0
    for row in rows:
        ref = references.get(str(row.get("sample_id")))
        if ref is None:
            unscored += 1
            continue
        reference_text, language = ref
        score = score_hypothesis(reference_text, str(row.get("final_text", "")), language)
        row["reference_raw"] = reference_text
        row["reference_normalized"] = score["reference_normalized"]
        row["hypothesis_normalized"] = score["hypothesis_normalized"]
        row["wer_errors"] = score["wer_errors"]
        row["wer_denominator"] = score["wer_denominator"]
        row["wer"] = _error_rate(int(score["wer_errors"]), int(score["wer_denominator"]))
        row["cer_errors"] = score["cer_errors"]
        row["cer_denominator"] = score["cer_denominator"]
        row["cer"] = _error_rate(int(score["cer_errors"]), int(score["cer_denominator"]))
        row["primary_metric"] = score["primary_metric"]
        row["primary_error_rate"] = _error_rate(
            int(score["primary_errors"]), int(score["primary_denominator"])
        )
        scored += 1

        mode = str(row.get("endpointing_mode", "unknown"))
        for key in ("all", mode):
            bucket = totals.setdefault(
                key,
                {
                    "wer_errors": 0,
                    "wer_denominator": 0,
                    "cer_errors": 0,
                    "cer_denominator": 0,
                    "primary_errors": 0,
                    "primary_denominator": 0,
                    "evaluations": 0,
                },
            )
            bucket["evaluations"] += 1
            for field in (
                "wer_errors",
                "wer_denominator",
                "cer_errors",
                "cer_denominator",
                "primary_errors",
                "primary_denominator",
            ):
                bucket[field] += int(score[field])

    if scored == 0:
        return None

    def _rates(bucket: dict[str, int]) -> dict[str, float | int]:
        return {
            "evaluations": bucket["evaluations"],
            "wer": _error_rate(bucket["wer_errors"], bucket["wer_denominator"]),
            "cer": _error_rate(bucket["cer_errors"], bucket["cer_denominator"]),
            "primary_error_rate": _error_rate(
                bucket["primary_errors"], bucket["primary_denominator"]
            ),
        }

    by_mode = {mode: _rates(bucket) for mode, bucket in sorted(totals.items()) if mode != "all"}
    return {
        "note": (
            "final_text scored against manifest reference_text with "
            "eval_manifest_quality normalization"
        ),
        "scored_rows": scored,
        "unscored_rows": unscored,
        "aggregate": _rates(totals["all"]),
        "by_mode": by_mode,
        "worst_mode_primary_error_rate": max(
            float(stats["primary_error_rate"]) for stats in by_mode.values()
        ),
    }


def _load_offline_quality(path: Path, *, manifest_path: Path) -> dict[str, float | str]:
    """Read the offline ``eval_manifest_quality`` artifact used as the ceiling anchor.

    Refuses artifacts produced from a different manifest: comparing streaming
    on one dataset against offline on another would make the ceiling
    meaningless.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    offline_manifest = Path(str(payload.get("manifest_jsonl", ""))).name
    if offline_manifest and offline_manifest != manifest_path.name:
        raise ValueError(
            "Offline quality artifact was produced from a different manifest: "
            f"{offline_manifest} != {manifest_path.name}"
        )
    return {
        "source": str(path),
        "model": str(payload.get("model", "")),
        "wer": float(payload["wer"]),
        "cer": float(payload["cer"]),
        "primary_error_rate": float(payload["primary_error_rate"]),
    }


def _split_modes(raw: str) -> list[str]:
    modes = [m.strip() for m in raw.split(",") if m.strip()]
    invalid = [m for m in modes if m not in {"fixed", "energy"}]
    if invalid:
        bad = ", ".join(sorted(set(invalid)))
        raise ValueError(f"Unsupported endpointing mode(s): {bad}")
    if not modes:
        raise ValueError("At least one endpointing mode is required")
    return modes


def _summarize_rows(rows: list[dict]) -> dict[str, dict[str, float | int]]:
    if not rows:
        return {
            "aggregate": {
                "evaluations": 0,
                "partial_stability_mean": 0.0,
                "partial_stability_min": 0.0,
                "rewrite_rate_mean": 0.0,
                "rewrite_rate_max": 0.0,
                "finalization_delta_chars_mean": 0.0,
                "finalization_delta_chars_max": 0,
                "latency_sec_mean": 0.0,
                "latency_sec_p95": 0.0,
                "rtf_mean": 0.0,
                "rtf_p95": 0.0,
            },
            "by_mode": {},
        }

    def _stats(chunk: list[dict]) -> dict[str, float | int]:
        stabilities = [float(r["partial_stability"]) for r in chunk]
        rewrites = [float(r["rewrite_rate"]) for r in chunk]
        final_deltas = [int(r["finalization_delta_chars"]) for r in chunk]
        latencies = [float(r["latency_sec"]) for r in chunk]
        rtfs = [float(r["rtf"]) for r in chunk]
        return {
            "evaluations": len(chunk),
            "partial_stability_mean": float(statistics.mean(stabilities)),
            "partial_stability_min": float(min(stabilities)),
            "rewrite_rate_mean": float(statistics.mean(rewrites)),
            "rewrite_rate_max": float(max(rewrites)),
            "finalization_delta_chars_mean": float(statistics.mean(final_deltas)),
            "finalization_delta_chars_max": int(max(final_deltas)),
            "latency_sec_mean": float(statistics.mean(latencies)),
            "latency_sec_p95": float(np.percentile(latencies, 95)),
            "rtf_mean": float(statistics.mean(rtfs)),
            "rtf_p95": float(np.percentile(rtfs, 95)),
        }

    by_mode: dict[str, list[dict]] = {}
    for row in rows:
        by_mode.setdefault(str(row["endpointing_mode"]), []).append(row)

    return {
        "aggregate": _stats(rows),
        "by_mode": {mode: _stats(mode_rows) for mode, mode_rows in sorted(by_mode.items())},
    }


def _threshold_failures(
    *,
    aggregate: dict[str, float | int],
    fail_partial_stability_below: float | None,
    fail_rewrite_rate_above: float | None,
    fail_finalization_delta_chars_above: int | None,
    quality: dict[str, object] | None = None,
    fail_primary_above: float | None = None,
    fail_primary_above_offline_pp: float | None = None,
) -> list[str]:
    failures: list[str] = []
    stability_mean = float(aggregate.get("partial_stability_mean", 0.0))
    rewrite_mean = float(aggregate.get("rewrite_rate_mean", 0.0))
    final_delta_max = int(aggregate.get("finalization_delta_chars_max", 0))

    if (
        fail_partial_stability_below is not None
        and stability_mean < fail_partial_stability_below
    ):
        failures.append(
            "Streaming stability gate failed: "
            f"partial_stability_mean={stability_mean:.6f} "
            f"< threshold={fail_partial_stability_below:.6f}"
        )
    if fail_rewrite_rate_above is not None and rewrite_mean > fail_rewrite_rate_above:
        failures.append(
            "Streaming rewrite gate failed: "
            f"rewrite_rate_mean={rewrite_mean:.6f} "
            f"> threshold={fail_rewrite_rate_above:.6f}"
        )
    if (
        fail_finalization_delta_chars_above is not None
        and final_delta_max > fail_finalization_delta_chars_above
    ):
        failures.append(
            "Streaming finalization gate failed: "
            f"finalization_delta_chars_max={final_delta_max} "
            f"> threshold={fail_finalization_delta_chars_above}"
        )

    wants_reference_gate = (
        fail_primary_above is not None or fail_primary_above_offline_pp is not None
    )
    if wants_reference_gate and quality is None:
        failures.append(
            "Streaming reference gate failed: a primary-error threshold was requested "
            "but no manifest row carried reference_text, so final_text was not scored"
        )
        return failures
    if quality is None:
        return failures

    worst_primary = float(quality["worst_mode_primary_error_rate"])
    if fail_primary_above is not None and worst_primary > fail_primary_above:
        failures.append(
            "Streaming reference gate failed: "
            f"worst_mode_primary_error_rate={worst_primary:.6f} "
            f"> threshold={fail_primary_above:.6f}"
        )
    if fail_primary_above_offline_pp is not None:
        offline = quality.get("offline")
        if not isinstance(offline, dict):
            failures.append(
                "Streaming reference gate failed: --fail-primary-above-offline-pp "
                "requires --offline-quality-json"
            )
        else:
            offline_primary = float(offline["primary_error_rate"])
            ceiling = offline_primary + fail_primary_above_offline_pp / 100.0
            if worst_primary > ceiling:
                failures.append(
                    "Streaming reference gate failed: "
                    f"worst_mode_primary_error_rate={worst_primary:.6f} "
                    f"> offline {offline_primary:.6f} + "
                    f"{fail_primary_above_offline_pp:.2f}pp = {ceiling:.6f}"
                )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate streaming diagnostics over a JSONL manifest."
    )
    parser.add_argument("--manifest-jsonl", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32", "bfloat16"],
        default="float16",
    )
    parser.add_argument(
        "--endpointing-modes",
        default="fixed,energy",
        help="Comma-separated endpointing modes (fixed,energy)",
    )
    parser.add_argument("--chunk-size-sec", type=float, default=2.0)
    parser.add_argument("--max-context-sec", type=float, default=30.0)
    parser.add_argument("--unfixed-chunk-num", type=int, default=2)
    parser.add_argument("--unfixed-token-num", type=int, default=5)
    parser.add_argument(
        "--finalization-mode",
        choices=["accuracy", "latency"],
        default="accuracy",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--fail-partial-stability-below", type=float, default=None)
    parser.add_argument("--fail-rewrite-rate-above", type=float, default=None)
    parser.add_argument("--fail-finalization-delta-chars-above", type=int, default=None)
    parser.add_argument(
        "--offline-quality-json",
        default=None,
        help=(
            "eval_manifest_quality.py artifact for the same manifest; recorded under "
            "quality_vs_reference.offline and used as the anchor for "
            "--fail-primary-above-offline-pp"
        ),
    )
    parser.add_argument(
        "--fail-primary-above",
        type=float,
        default=None,
        help="Fail if the worst endpointing mode's primary error rate exceeds this",
    )
    parser.add_argument(
        "--fail-primary-above-offline-pp",
        type=float,
        default=None,
        help=(
            "Fail if the worst endpointing mode's primary error rate exceeds the "
            "offline artifact's by more than this many percentage points"
        ),
    )
    args = parser.parse_args()

    if args.chunk_size_sec <= 0:
        raise ValueError("--chunk-size-sec must be > 0")
    if args.max_context_sec <= 0:
        raise ValueError("--max-context-sec must be > 0")
    if args.fail_primary_above_offline_pp is not None and not args.offline_quality_json:
        parser.error("--fail-primary-above-offline-pp requires --offline-quality-json")

    started = time.perf_counter()
    manifest_path = Path(args.manifest_jsonl).expanduser().resolve()
    offline_quality: dict[str, float | str] | None = None
    if args.offline_quality_json:
        offline_quality = _load_offline_quality(
            Path(args.offline_quality_json).expanduser().resolve(),
            manifest_path=manifest_path,
        )
    samples = _parse_manifest(manifest_path)
    if args.limit is not None:
        samples = samples[: max(0, int(args.limit))]
    if not samples:
        raise RuntimeError(f"No samples to evaluate from manifest: {manifest_path}")

    endpointing_modes = _split_modes(args.endpointing_modes)
    dtype = _dtype_from_name(args.dtype)
    model, _ = load_model(args.model, dtype=dtype)

    chunk_size_samples = max(1, int(args.chunk_size_sec * 16000))
    rows: list[dict] = []

    for sample_index, sample in enumerate(samples, start=1):
        audio_np = np.array(load_audio(str(sample.audio_path)), dtype=np.float32)
        duration_sec = len(audio_np) / 16000.0
        chunks = _chunk_audio(audio_np, chunk_size_samples)

        for mode in endpointing_modes:
            state = init_streaming(
                model=args.model,
                unfixed_chunk_num=args.unfixed_chunk_num,
                unfixed_token_num=args.unfixed_token_num,
                chunk_size_sec=args.chunk_size_sec,
                max_context_sec=args.max_context_sec,
                sample_rate=16000,
                endpointing_mode=mode,
                finalization_mode=args.finalization_mode,
            )

            t0 = time.perf_counter()
            for chunk in chunks:
                feed_audio(chunk, state, model=model)
            finish_streaming(state, model=model)
            latency_sec = time.perf_counter() - t0

            metrics = dict(streaming_metrics(state))
            rows.append(
                {
                    "index": sample_index,
                    "sample_id": sample.sample_id,
                    "subset": sample.subset,
                    "speaker_id": sample.speaker_id,
                    "language": sample.language,
                    "audio_path": str(sample.audio_path),
                    "duration_sec": duration_sec,
                    "endpointing_mode": mode,
                    "num_chunks": len(chunks),
                    "partial_stability": float(metrics.get("partial_stability", 0.0)),
                    "rewrite_rate": float(metrics.get("rewrite_rate", 0.0)),
                    "finalization_delta_chars": int(
                        metrics.get("finalization_delta_chars", 0)
                    ),
                    "latency_sec": latency_sec,
                    "rtf": (latency_sec / duration_sec) if duration_sec > 0 else 0.0,
                    "final_text": str(getattr(state, "text", "")),
                    "final_language": str(getattr(state, "language", "unknown")),
                }
            )

    summary = _summarize_rows(rows)
    aggregate = summary["aggregate"]
    references = {
        s.sample_id: (s.reference_text, s.language) for s in samples if s.reference_text
    }
    quality = _score_rows_against_references(rows, references)
    if quality is not None and offline_quality is not None:
        quality["offline"] = offline_quality
    repo_root = Path(__file__).resolve().parents[1]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "suite": "streaming-manifest-quality-v1",
        "generated_at_utc": _iso_utc_now(),
        "git_commit": _git_head_commit(repo_root),
        "manifest_jsonl": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "model": args.model,
        "dtype": args.dtype,
        "endpointing_modes": endpointing_modes,
        "chunk_size_sec": args.chunk_size_sec,
        "max_context_sec": args.max_context_sec,
        "unfixed_chunk_num": args.unfixed_chunk_num,
        "unfixed_token_num": args.unfixed_token_num,
        "finalization_mode": args.finalization_mode,
        "samples": len(samples),
        "evaluations": len(rows),
        "aggregate": aggregate,
        "by_mode": summary["by_mode"],
        "quality_vs_reference": quality,
        "rows": rows,
        "elapsed_sec": time.perf_counter() - started,
    }

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    failures = _threshold_failures(
        aggregate=aggregate,
        fail_partial_stability_below=args.fail_partial_stability_below,
        fail_rewrite_rate_above=args.fail_rewrite_rate_above,
        fail_finalization_delta_chars_above=args.fail_finalization_delta_chars_above,
        quality=quality,
        fail_primary_above=args.fail_primary_above,
        fail_primary_above_offline_pp=args.fail_primary_above_offline_pp,
    )
    if failures:
        for msg in failures:
            print(msg, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
