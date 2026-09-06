#!/usr/bin/env python3
"""Evaluate diarization quality (DER/JER) from a JSONL manifest.

Manifest row schema:
- audio_path: str (required)
- reference_speaker_segments: list[dict] (required)
  - each segment: {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.2}
- sample_id: str (optional)
- language: str (optional)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

from mlx_qwen3_asr import load_model, transcribe


@dataclass(frozen=True)
class ManifestSample:
    sample_id: str
    audio_path: Path
    language: Optional[str]
    reference_speaker_segments: list[dict]


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
            "soundfile is required for diarization evaluation. "
            "Install with: pip install 'mlx-qwen3-asr[eval]'"
        ) from exc

    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1).astype(np.float32)
    else:
        audio = audio.astype(np.float32)
    return audio, int(sr)


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


def _sanitize_segments(segments: list[dict]) -> list[dict]:
    cleaned: list[dict] = []
    for segment in segments:
        speaker = str(segment.get("speaker", "")).strip()
        if not speaker:
            continue
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        if not math.isfinite(start) or not math.isfinite(end):
            continue
        start = max(0.0, start)
        end = max(start, end)
        if end <= start:
            continue
        cleaned.append({"speaker": speaker, "start": start, "end": end})
    cleaned.sort(key=lambda x: (float(x["start"]), float(x["end"]), str(x["speaker"])))
    return cleaned


def _parse_manifest(path: Path) -> list[ManifestSample]:
    rows: list[ManifestSample] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        obj = json.loads(raw)
        audio_path = Path(obj["audio_path"]).expanduser().resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Manifest row {i} references missing audio: {audio_path}")
        reference = obj.get("reference_speaker_segments")
        if not isinstance(reference, list):
            raise ValueError(
                "Manifest row {i} requires list field 'reference_speaker_segments'.".format(
                    i=i
                )
            )
        reference_clean = _sanitize_segments(reference)
        if not reference_clean:
            raise ValueError(
                f"Manifest row {i} has no usable reference_speaker_segments after validation."
            )
        rows.append(
            ManifestSample(
                sample_id=str(obj.get("sample_id", f"manifest-{i:05d}")),
                audio_path=audio_path,
                language=obj.get("language"),
                reference_speaker_segments=reference_clean,
            )
        )
    return rows


def _segments_to_masks(
    segments: list[dict],
    *,
    timeline_sec: float,
    frame_step_sec: float,
) -> tuple[list[str], list[np.ndarray]]:
    n_frames = max(1, int(math.ceil(timeline_sec / frame_step_sec)))
    by_speaker: dict[str, np.ndarray] = {}
    for item in segments:
        speaker = str(item["speaker"])
        mask = by_speaker.setdefault(speaker, np.zeros((n_frames,), dtype=bool))
        start = float(item["start"])
        end = float(item["end"])
        lo = max(0, min(n_frames, int(math.floor(start / frame_step_sec))))
        hi = max(0, min(n_frames, int(math.ceil(end / frame_step_sec))))
        if hi > lo:
            mask[lo:hi] = True
    speakers = sorted(by_speaker.keys())
    masks = [by_speaker[s] for s in speakers]
    return speakers, masks


def _overlap_duration(a: np.ndarray, b: np.ndarray, *, frame_step_sec: float) -> float:
    return float(np.sum(a & b)) * frame_step_sec


def _best_row_to_col_assignment(weights: np.ndarray) -> list[int]:
    """Return best one-to-one row->col assignment maximizing total weight.

    Returns list of length rows with each entry set to:
    - col index [0, cols), or
    - -1 for unmatched row.
    """
    rows, cols = weights.shape
    if rows == 0:
        return []
    if cols == 0:
        return [-1] * rows

    if cols > 12:
        # Practical fallback for very large speaker sets.
        assignment = [-1] * rows
        used: set[int] = set()
        pairs: list[tuple[float, int, int]] = []
        for r in range(rows):
            for c in range(cols):
                pairs.append((float(weights[r, c]), r, c))
        pairs.sort(reverse=True)
        for score, r, c in pairs:
            if score <= 0.0:
                break
            if assignment[r] != -1 or c in used:
                continue
            assignment[r] = c
            used.add(c)
        return assignment

    @lru_cache(maxsize=None)
    def _solve(row: int, used_mask: int) -> tuple[float, tuple[int, ...]]:
        if row >= rows:
            return 0.0, ()

        best_score, best_assign = _solve(row + 1, used_mask)
        best_col = -1
        for col in range(cols):
            bit = 1 << col
            if used_mask & bit:
                continue
            child_score, child_assign = _solve(row + 1, used_mask | bit)
            score = float(weights[row, col]) + child_score
            if score > best_score:
                best_score = score
                best_assign = child_assign
                best_col = col
        return best_score, (best_col, *best_assign)

    _score, packed = _solve(0, 0)
    return list(packed)


def _build_collar_keep_mask(
    *,
    n_frames: int,
    frame_step_sec: float,
    reference_segments: list[dict],
    collar_sec: float,
) -> np.ndarray:
    keep = np.ones((n_frames,), dtype=bool)
    if collar_sec <= 0.0:
        return keep
    boundaries: list[float] = []
    for item in reference_segments:
        boundaries.append(float(item["start"]))
        boundaries.append(float(item["end"]))
    for b in boundaries:
        lo = int(math.floor(max(0.0, b - collar_sec) / frame_step_sec))
        hi = int(math.ceil(max(0.0, b + collar_sec) / frame_step_sec))
        lo = max(0, min(n_frames, lo))
        hi = max(0, min(n_frames, hi))
        if hi > lo:
            keep[lo:hi] = False
    return keep


def compute_sample_diarization_metrics(
    *,
    reference_segments: list[dict],
    hypothesis_segments: list[dict],
    audio_duration_sec: float,
    frame_step_sec: float,
    collar_sec: float,
    ignore_overlap: bool,
) -> dict:
    reference = _sanitize_segments(reference_segments)
    hypothesis = _sanitize_segments(hypothesis_segments)
    timeline_sec = max(
        float(audio_duration_sec),
        max((float(x["end"]) for x in reference), default=0.0),
        max((float(x["end"]) for x in hypothesis), default=0.0),
        frame_step_sec,
    )
    ref_speakers, ref_masks = _segments_to_masks(
        reference,
        timeline_sec=timeline_sec,
        frame_step_sec=frame_step_sec,
    )
    hyp_speakers, hyp_masks = _segments_to_masks(
        hypothesis,
        timeline_sec=timeline_sec,
        frame_step_sec=frame_step_sec,
    )
    n_frames = (
        len(ref_masks[0])
        if ref_masks
        else max(1, int(math.ceil(timeline_sec / frame_step_sec)))
    )

    overlap = np.zeros((len(hyp_speakers), len(ref_speakers)), dtype=np.float64)
    for hi, hmask in enumerate(hyp_masks):
        for ri, rmask in enumerate(ref_masks):
            overlap[hi, ri] = _overlap_duration(hmask, rmask, frame_step_sec=frame_step_sec)
    hyp_to_ref_idx = _best_row_to_col_assignment(overlap)
    hyp_to_ref: dict[str, str] = {}
    for hi, ri in enumerate(hyp_to_ref_idx):
        if ri >= 0 and ri < len(ref_speakers):
            hyp_to_ref[hyp_speakers[hi]] = ref_speakers[ri]

    mapped_hyp_masks: dict[str, np.ndarray] = {}
    for hi, speaker in enumerate(hyp_speakers):
        target = hyp_to_ref.get(speaker)
        if target is None:
            continue
        prev = mapped_hyp_masks.get(target)
        mapped_hyp_masks[target] = (
            hyp_masks[hi].copy() if prev is None else (prev | hyp_masks[hi])
        )

    ref_active_count = np.zeros((n_frames,), dtype=np.int32)
    hyp_active_count = np.zeros((n_frames,), dtype=np.int32)
    intersection_count = np.zeros((n_frames,), dtype=np.int32)

    for rmask in ref_masks:
        ref_active_count += rmask.astype(np.int32)
    for hmask in hyp_masks:
        hyp_active_count += hmask.astype(np.int32)
    for speaker, rmask in zip(ref_speakers, ref_masks, strict=True):
        hmask = mapped_hyp_masks.get(speaker)
        if hmask is None:
            continue
        intersection_count += (rmask & hmask).astype(np.int32)

    keep_mask = _build_collar_keep_mask(
        n_frames=n_frames,
        frame_step_sec=frame_step_sec,
        reference_segments=reference,
        collar_sec=collar_sec,
    )
    if ignore_overlap:
        keep_mask &= ref_active_count <= 1

    miss = np.maximum(0, ref_active_count - intersection_count)
    false_alarm = np.maximum(0, hyp_active_count - intersection_count)
    confusion = np.minimum(ref_active_count, hyp_active_count) - intersection_count
    errors = miss + false_alarm + confusion

    ref_time = float(np.sum(ref_active_count[keep_mask])) * frame_step_sec
    miss_time = float(np.sum(miss[keep_mask])) * frame_step_sec
    false_alarm_time = float(np.sum(false_alarm[keep_mask])) * frame_step_sec
    confusion_time = float(np.sum(confusion[keep_mask])) * frame_step_sec
    error_time = float(np.sum(errors[keep_mask])) * frame_step_sec
    der = (error_time / ref_time) if ref_time > 0.0 else 0.0

    # JER: one-to-one reference->hypothesis assignment by Jaccard.
    jaccard = np.zeros((len(ref_speakers), len(hyp_speakers)), dtype=np.float64)
    for ri, rmask in enumerate(ref_masks):
        for hi, hmask in enumerate(hyp_masks):
            inter = _overlap_duration(rmask, hmask, frame_step_sec=frame_step_sec)
            union = (
                float(np.sum(rmask)) * frame_step_sec
                + float(np.sum(hmask)) * frame_step_sec
                - inter
            )
            jaccard[ri, hi] = (inter / union) if union > 0.0 else 0.0
    ref_to_hyp_idx = _best_row_to_col_assignment(jaccard)
    jer_error_sum = 0.0
    for ri, hi in enumerate(ref_to_hyp_idx):
        if hi < 0 or hi >= len(hyp_speakers):
            jer_error_sum += 1.0
            continue
        jer_error_sum += 1.0 - float(jaccard[ri, hi])
    jer_den = len(ref_speakers)
    jer = (jer_error_sum / jer_den) if jer_den > 0 else 0.0

    return {
        "der": der,
        "jer": jer,
        "der_numerator_sec": error_time,
        "der_denominator_sec": ref_time,
        "miss_sec": miss_time,
        "false_alarm_sec": false_alarm_time,
        "confusion_sec": confusion_time,
        "jer_error_sum": jer_error_sum,
        "jer_denominator": jer_den,
        "num_ref_speakers": len(ref_speakers),
        "num_hyp_speakers": len(hyp_speakers),
        "hyp_to_ref_map": hyp_to_ref,
    }


def _threshold_failures(
    *,
    der: float,
    jer: float,
    fail_der_above: float | None,
    fail_jer_above: float | None,
) -> list[str]:
    failures: list[str] = []
    if fail_der_above is not None and der > fail_der_above:
        failures.append(
            f"DER regression gate failed: der={der:.6f} > threshold={fail_der_above:.6f}"
        )
    if fail_jer_above is not None and jer > fail_jer_above:
        failures.append(
            f"JER regression gate failed: jer={jer:.6f} > threshold={fail_jer_above:.6f}"
        )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate diarization DER/JER from a JSONL manifest."
    )
    parser.add_argument("--manifest-jsonl", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32", "bfloat16"],
        default="float16",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=1)
    parser.add_argument("--max-speakers", type=int, default=8)
    parser.add_argument("--frame-step-sec", type=float, default=0.02)
    parser.add_argument("--collar-sec", type=float, default=0.25)
    parser.add_argument(
        "--ignore-overlap",
        action="store_true",
        help="Exclude overlap reference frames from DER scoring.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--fail-der-above", type=float, default=None)
    parser.add_argument("--fail-jer-above", type=float, default=None)
    args = parser.parse_args()

    if args.frame_step_sec <= 0.0:
        raise ValueError("--frame-step-sec must be > 0.")
    if args.collar_sec < 0.0:
        raise ValueError("--collar-sec must be >= 0.")

    started = time.perf_counter()
    manifest_path = Path(args.manifest_jsonl).expanduser().resolve()
    samples = _parse_manifest(manifest_path)
    if args.limit is not None:
        samples = samples[: max(0, int(args.limit))]
    if not samples:
        raise RuntimeError(f"No samples to evaluate from manifest: {manifest_path}")

    dtype = _dtype_from_name(args.dtype)
    model, _ = load_model(args.model, dtype=dtype)

    rows: list[dict] = []
    der_num = 0.0
    der_den = 0.0
    jer_num = 0.0
    jer_den = 0
    latencies: list[float] = []

    for i, sample in enumerate(samples, start=1):
        audio, sr = _read_audio(sample.audio_path)
        language_arg = _select_language_arg(sample.language)

        t0 = time.perf_counter()
        result = transcribe(
            (audio, sr),
            model=model,
            language=language_arg,
            diarize=True,
            diarization_num_speakers=args.num_speakers,
            diarization_min_speakers=args.min_speakers,
            diarization_max_speakers=args.max_speakers,
            max_new_tokens=args.max_new_tokens,
            verbose=False,
        )
        latency = time.perf_counter() - t0
        latencies.append(latency)
        predicted_segments = _sanitize_segments(result.speaker_segments or [])
        metrics = compute_sample_diarization_metrics(
            reference_segments=sample.reference_speaker_segments,
            hypothesis_segments=predicted_segments,
            audio_duration_sec=float(len(audio) / max(1, sr)),
            frame_step_sec=args.frame_step_sec,
            collar_sec=args.collar_sec,
            ignore_overlap=bool(args.ignore_overlap),
        )

        der_num += float(metrics["der_numerator_sec"])
        der_den += float(metrics["der_denominator_sec"])
        jer_num += float(metrics["jer_error_sum"])
        jer_den += int(metrics["jer_denominator"])

        rows.append(
            {
                "index": i,
                "sample_id": sample.sample_id,
                "audio_path": str(sample.audio_path),
                "language": sample.language,
                "duration_sec": float(len(audio) / max(1, sr)),
                "reference_speaker_segments": sample.reference_speaker_segments,
                "hypothesis_speaker_segments": predicted_segments,
                "num_ref_speakers": metrics["num_ref_speakers"],
                "num_hyp_speakers": metrics["num_hyp_speakers"],
                "hyp_to_ref_map": metrics["hyp_to_ref_map"],
                "der": metrics["der"],
                "jer": metrics["jer"],
                "miss_sec": metrics["miss_sec"],
                "false_alarm_sec": metrics["false_alarm_sec"],
                "confusion_sec": metrics["confusion_sec"],
                "latency_sec": latency,
            }
        )

    der = (der_num / der_den) if der_den > 0.0 else 0.0
    jer = (jer_num / float(jer_den)) if jer_den > 0 else 0.0
    payload = {
        "suite": "diarization-quality-v1",
        "manifest_jsonl": str(manifest_path),
        "model": args.model,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "frame_step_sec": args.frame_step_sec,
        "collar_sec": args.collar_sec,
        "ignore_overlap": bool(args.ignore_overlap),
        "samples": len(samples),
        "der": der,
        "jer": jer,
        "der_numerator_sec": der_num,
        "der_denominator_sec": der_den,
        "jer_error_sum": jer_num,
        "jer_denominator": jer_den,
        "latency_sec_mean": float(np.mean(latencies)) if latencies else 0.0,
        "elapsed_sec": time.perf_counter() - started,
        "rows": rows,
    }

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.json_output:
        out = Path(args.json_output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    failures = _threshold_failures(
        der=der,
        jer=jer,
        fail_der_above=args.fail_der_above,
        fail_jer_above=args.fail_jer_above,
    )
    if failures:
        for msg in failures:
            print(msg, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
