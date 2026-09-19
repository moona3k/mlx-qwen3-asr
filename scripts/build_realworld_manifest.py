#!/usr/bin/env python3
"""Build deterministic real-world ASR manifests from HF parquet shards.

This script currently supports:
- `ami_ihm_test` (meeting speech, conversational)
- `earnings22_chunked_test` (accented earnings-call speech, chunked)
"""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import dataclass
from pathlib import Path

import _repo_path  # noqa: F401
import numpy as np
from huggingface_hub import HfApi, hf_hub_download

from mlx_qwen3_asr.audio import load_audio


@dataclass(frozen=True)
class SourceSpec:
    name: str
    repo_id: str
    parquet_prefix: str
    text_field: str
    speaker_field: str
    sample_id_field: str
    subset: str
    language: str
    duration_start_field: str
    duration_end_field: str


@dataclass(frozen=True)
class Candidate:
    source: str
    sample_id: str
    speaker_id: str
    reference_text: str
    duration_sec: float
    audio_bytes: bytes
    subset: str
    language: str


SOURCES: dict[str, SourceSpec] = {
    "ami_ihm_test": SourceSpec(
        name="ami_ihm_test",
        repo_id="edinburghcstr/ami",
        parquet_prefix="ihm/test-",
        text_field="text",
        speaker_field="speaker_id",
        sample_id_field="audio_id",
        subset="ami-ihm-test",
        language="English",
        duration_start_field="begin_time",
        duration_end_field="end_time",
    ),
    "earnings22_chunked_test": SourceSpec(
        name="earnings22_chunked_test",
        repo_id="distil-whisper/earnings22",
        parquet_prefix="chunked/test-",
        text_field="transcription",
        speaker_field="file_id",
        sample_id_field="segment_id",
        subset="earnings22-chunked-test",
        language="English",
        duration_start_field="start_ts",
        duration_end_field="end_ts",
    ),
}


def _list_parquet_files(spec: SourceSpec) -> list[str]:
    api = HfApi()
    files = api.list_repo_files(spec.repo_id, repo_type="dataset")
    return sorted(
        path
        for path in files
        if path.startswith(spec.parquet_prefix) and path.endswith(".parquet")
    )


def _duration_from_row(row: dict, spec: SourceSpec) -> float:
    start = row.get(spec.duration_start_field)
    end = row.get(spec.duration_end_field)
    if start is None or end is None:
        return 0.0
    return max(0.0, float(end) - float(start))


def _audio_bytes_from_row(row: dict) -> bytes | None:
    audio_obj = row.get("audio")
    if not isinstance(audio_obj, dict):
        return None
    raw = audio_obj.get("bytes")
    if raw is None:
        return None
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, bytearray):
        return bytes(raw)
    return None


def _load_candidates(
    *,
    spec: SourceSpec,
    samples_per_source: int,
    min_duration_sec: float,
    max_duration_sec: float,
    candidate_multiplier: int,
    min_unique_speakers: int,
    max_shards: int | None,
) -> list[Candidate]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "build_realworld_manifest requires pyarrow. "
            "Install with: pip install pyarrow"
        ) from exc

    target_candidates = max(1, samples_per_source * max(1, candidate_multiplier))
    target_unique_speakers = max(1, int(min_unique_speakers))
    parquet_files = _list_parquet_files(spec)
    if max_shards is not None:
        parquet_files = parquet_files[: max(0, int(max_shards))]
    if not parquet_files:
        raise RuntimeError(f"No parquet shards found for source={spec.name}")

    cols = [
        spec.text_field,
        spec.speaker_field,
        spec.sample_id_field,
        spec.duration_start_field,
        spec.duration_end_field,
        "audio",
    ]
    out: list[Candidate] = []
    seen_speakers: set[str] = set()
    for shard_idx, filename in enumerate(parquet_files):
        shard_path = hf_hub_download(
            repo_id=spec.repo_id,
            repo_type="dataset",
            filename=filename,
        )
        table = pq.read_table(shard_path, columns=cols)
        rows = table.to_pylist()
        for row_idx, row in enumerate(rows):
            text = str(row.get(spec.text_field, "")).strip()
            if not text:
                continue
            duration_sec = _duration_from_row(row, spec)
            if duration_sec < min_duration_sec or duration_sec > max_duration_sec:
                continue
            audio_bytes = _audio_bytes_from_row(row)
            if not audio_bytes:
                continue
            sample_id = str(
                row.get(spec.sample_id_field) or f"{spec.name}-{shard_idx:03d}-{row_idx:06d}"
            )
            speaker_id = str(row.get(spec.speaker_field) or "unknown")
            seen_speakers.add(speaker_id)
            out.append(
                Candidate(
                    source=spec.name,
                    sample_id=sample_id,
                    speaker_id=speaker_id,
                    reference_text=text,
                    duration_sec=duration_sec,
                    audio_bytes=audio_bytes,
                    subset=spec.subset,
                    language=spec.language,
                )
            )
            if len(out) >= target_candidates and len(seen_speakers) >= target_unique_speakers:
                return out
    return out


def _sample_speaker_round_robin(
    candidates: list[Candidate], samples: int, seed: int
) -> list[Candidate]:
    if samples <= 0 or not candidates:
        return []
    by_speaker: dict[str, list[Candidate]] = {}
    for item in candidates:
        by_speaker.setdefault(item.speaker_id, []).append(item)

    rng = np.random.default_rng(seed)
    speakers = sorted(by_speaker)
    if not speakers:
        return []

    for speaker in speakers:
        items = by_speaker[speaker]
        if len(items) > 1:
            order = rng.permutation(len(items))
            by_speaker[speaker] = [items[int(i)] for i in order]

    # Rotate start speaker for deterministic seed-dependent spread.
    offset = int(seed % len(speakers))
    speakers = speakers[offset:] + speakers[:offset]

    selected: list[Candidate] = []
    while len(selected) < samples:
        made_progress = False
        for speaker in speakers:
            items = by_speaker[speaker]
            if not items:
                continue
            selected.append(items.pop(0))
            made_progress = True
            if len(selected) >= samples:
                break
        if not made_progress:
            break
    return selected


def _to_audio_16k(audio_bytes: bytes) -> np.ndarray:
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "build_realworld_manifest requires soundfile. "
            "Install with: pip install soundfile"
        ) from exc
    array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if array.ndim > 1:
        array = np.mean(array, axis=1).astype(np.float32)
    else:
        array = array.astype(np.float32)
    return np.asarray(load_audio((array, int(sr))), dtype=np.float32)


def _safe_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build deterministic JSONL manifest from real-world HF parquet datasets."
    )
    parser.add_argument(
        "--sources",
        default="ami_ihm_test,earnings22_chunked_test",
        help=f"Comma-separated source names. Supported: {','.join(sorted(SOURCES))}",
    )
    parser.add_argument("--samples-per-source", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260215)
    parser.add_argument("--candidate-multiplier", type=int, default=8)
    parser.add_argument("--min-unique-speakers", type=int, default=4)
    parser.add_argument("--max-shards", type=int, default=2)
    parser.add_argument("--min-duration-sec", type=float, default=1.0)
    parser.add_argument("--max-duration-sec", type=float, default=25.0)
    parser.add_argument(
        "--output-manifest",
        default="docs/benchmarks/2026-02-15-realworld-manifest-40.jsonl",
    )
    parser.add_argument(
        "--output-audio-dir",
        default=str(Path.home() / ".cache" / "mlx-qwen3-asr" / "datasets" / "realworld-audio"),
    )
    args = parser.parse_args()

    source_names = [s.strip() for s in str(args.sources).split(",") if s.strip()]
    if not source_names:
        raise ValueError("At least one source must be provided.")
    unknown = [s for s in source_names if s not in SOURCES]
    if unknown:
        raise ValueError(f"Unknown source(s): {', '.join(unknown)}")
    if args.samples_per_source <= 0:
        raise ValueError("--samples-per-source must be > 0")
    if args.candidate_multiplier <= 0:
        raise ValueError("--candidate-multiplier must be > 0")
    if args.min_unique_speakers <= 0:
        raise ValueError("--min-unique-speakers must be > 0")
    if args.min_duration_sec <= 0.0:
        raise ValueError("--min-duration-sec must be > 0")
    if args.max_duration_sec <= args.min_duration_sec:
        raise ValueError("--max-duration-sec must be > --min-duration-sec")

    audio_root = Path(args.output_audio_dir).expanduser().resolve()
    manifest_path = Path(args.output_manifest).expanduser().resolve()
    audio_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "build_realworld_manifest requires soundfile. "
            "Install with: pip install soundfile"
        ) from exc

    rows_out: list[dict[str, object]] = []
    summary_sources: list[dict[str, object]] = []
    for idx, source_name in enumerate(source_names):
        spec = SOURCES[source_name]
        candidates = _load_candidates(
            spec=spec,
            samples_per_source=int(args.samples_per_source),
            min_duration_sec=float(args.min_duration_sec),
            max_duration_sec=float(args.max_duration_sec),
            candidate_multiplier=int(args.candidate_multiplier),
            min_unique_speakers=int(args.min_unique_speakers),
            max_shards=args.max_shards if args.max_shards > 0 else None,
        )
        picked = _sample_speaker_round_robin(
            candidates,
            samples=int(args.samples_per_source),
            seed=int(args.seed) + idx,
        )

        source_dir = audio_root / _safe_component(spec.name)
        source_dir.mkdir(parents=True, exist_ok=True)
        for sample_idx, item in enumerate(picked):
            safe_id = _safe_component(item.sample_id) or f"{spec.name}_{sample_idx:04d}"
            wav_path = (source_dir / f"{safe_id}.wav").resolve()
            audio_16k = _to_audio_16k(item.audio_bytes)
            sf.write(str(wav_path), audio_16k, samplerate=16000)

            rows_out.append(
                {
                    "sample_id": f"{spec.name}:{item.sample_id}",
                    "subset": spec.subset,
                    "speaker_id": item.speaker_id,
                    "language": spec.language,
                    "audio_path": str(wav_path),
                    "condition": "realworld",
                    "source": spec.name,
                    "duration_sec": item.duration_sec,
                    "reference_text": item.reference_text,
                }
            )
        summary_sources.append(
            {
                "source": spec.name,
                "repo_id": spec.repo_id,
                "subset": spec.subset,
                "candidates": len(candidates),
                "samples_written": len(picked),
                "unique_speakers": len({x.speaker_id for x in picked}),
            }
        )

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows_out:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    payload = {
        "output_manifest": str(manifest_path),
        "output_audio_dir": str(audio_root),
        "sources": summary_sources,
        "total_samples": len(rows_out),
        "seed": int(args.seed),
        "samples_per_source": int(args.samples_per_source),
        "candidate_multiplier": int(args.candidate_multiplier),
        "min_unique_speakers": int(args.min_unique_speakers),
        "max_shards": int(args.max_shards),
        "duration_filter_sec": {
            "min": float(args.min_duration_sec),
            "max": float(args.max_duration_sec),
        },
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
