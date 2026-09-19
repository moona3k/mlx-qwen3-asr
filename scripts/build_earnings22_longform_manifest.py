#!/usr/bin/env python3
"""Build deterministic non-synthetic long-form manifest from Earnings22 full split."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import _repo_path  # noqa: F401
import numpy as np
from huggingface_hub import HfApi, hf_hub_download

from mlx_qwen3_asr.audio import load_audio


@dataclass(frozen=True)
class Candidate:
    file_id: str
    language_family: str
    country: str
    duration_sec: float
    transcription: str
    audio_bytes: bytes
    audio_name: str


def _list_full_shards() -> list[str]:
    api = HfApi()
    files = api.list_repo_files("distil-whisper/earnings22", repo_type="dataset")
    return sorted(
        f for f in files if f.startswith("full/test-") and f.endswith(".parquet")
    )


def _safe_duration(value: object) -> float:
    try:
        return float(str(value).strip())
    except Exception:  # pragma: no cover
        return 0.0


def _load_candidates(
    *,
    min_duration_sec: float,
    max_duration_sec: float,
) -> list[Candidate]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "build_earnings22_longform_manifest requires pyarrow. "
            "Install with: pip install pyarrow"
        ) from exc

    out: list[Candidate] = []
    cols = [
        "audio",
        "file_id",
        "language_family",
        "country_by_ticker",
        "file_length",
        "transcription",
    ]
    for shard in _list_full_shards():
        shard_path = hf_hub_download(
            repo_id="distil-whisper/earnings22",
            repo_type="dataset",
            filename=shard,
        )
        table = pq.read_table(shard_path, columns=cols)
        for row in table.to_pylist():
            duration_sec = _safe_duration(row.get("file_length"))
            if duration_sec < min_duration_sec or duration_sec > max_duration_sec:
                continue
            audio = row.get("audio")
            if not isinstance(audio, dict):
                continue
            audio_bytes = audio.get("bytes")
            if not isinstance(audio_bytes, (bytes, bytearray)) or not audio_bytes:
                continue
            transcription = str(row.get("transcription", "")).strip()
            if not transcription:
                continue
            out.append(
                Candidate(
                    file_id=str(row.get("file_id") or "unknown"),
                    language_family=str(row.get("language_family") or "unknown"),
                    country=str(row.get("country_by_ticker") or "unknown"),
                    duration_sec=float(duration_sec),
                    transcription=transcription,
                    audio_bytes=bytes(audio_bytes),
                    audio_name=str(audio.get("path") or "audio.mp3"),
                )
            )
    return out


def _sample_family_round_robin(
    candidates: list[Candidate], *, samples: int, seed: int
) -> list[Candidate]:
    if samples <= 0 or not candidates:
        return []
    by_family: dict[str, list[Candidate]] = {}
    for c in candidates:
        by_family.setdefault(c.language_family, []).append(c)
    rng = np.random.default_rng(seed)
    families = sorted(by_family)
    if not families:
        return []

    for family in families:
        items = by_family[family]
        order = rng.permutation(len(items))
        by_family[family] = [items[int(i)] for i in order]

    start = int(seed % len(families))
    families = families[start:] + families[:start]

    out: list[Candidate] = []
    while len(out) < samples:
        progress = False
        for family in families:
            items = by_family[family]
            if not items:
                continue
            out.append(items.pop(0))
            progress = True
            if len(out) >= samples:
                break
        if not progress:
            break
    return out


def _decode_to_wav_16k(audio_bytes: bytes, audio_name: str, out_wav: Path) -> None:
    suffix = Path(audio_name).suffix or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(audio_bytes)
    try:
        arr = np.asarray(load_audio(str(tmp_path)), dtype=np.float32)
        try:
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "build_earnings22_longform_manifest requires soundfile. "
                "Install with: pip install soundfile"
            ) from exc
        sf.write(str(out_wav), arr, samplerate=16000)
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build long-form non-synthetic manifest from Earnings22 full split."
    )
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260215)
    parser.add_argument("--min-duration-sec", type=float, default=850.0)
    parser.add_argument("--max-duration-sec", type=float, default=1500.0)
    parser.add_argument(
        "--output-manifest",
        default="docs/benchmarks/2026-02-15-earnings22-full-longform3-manifest.jsonl",
    )
    parser.add_argument(
        "--output-audio-dir",
        default=str(
            Path.home() / ".cache" / "mlx-qwen3-asr" / "datasets" / "earnings22-full-longform"
        ),
    )
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be > 0")
    if args.min_duration_sec <= 0:
        raise ValueError("--min-duration-sec must be > 0")
    if args.max_duration_sec <= args.min_duration_sec:
        raise ValueError("--max-duration-sec must be > --min-duration-sec")

    candidates = _load_candidates(
        min_duration_sec=float(args.min_duration_sec),
        max_duration_sec=float(args.max_duration_sec),
    )
    if not candidates:
        raise RuntimeError("No candidates matched the duration range.")

    selected = _sample_family_round_robin(
        candidates, samples=int(args.samples), seed=int(args.seed)
    )
    if not selected:
        raise RuntimeError("No long-form samples selected.")

    out_dir = Path(args.output_audio_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.output_manifest).expanduser().resolve()
    manifest.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for i, c in enumerate(selected):
        wav_path = (out_dir / f"earnings22-full-{i:03d}-{c.file_id}.wav").resolve()
        _decode_to_wav_16k(c.audio_bytes, c.audio_name, wav_path)
        rows.append(
            {
                "sample_id": f"earnings22-full:{c.file_id}",
                "subset": "earnings22-full-test",
                "speaker_id": c.file_id,
                "language": "English",
                "audio_path": str(wav_path),
                "condition": "realworld-longform",
                "source": "earnings22_full_test",
                "duration_sec": c.duration_sec,
                "country": c.country,
                "language_family": c.language_family,
                "reference_text": c.transcription,
            }
        )

    with manifest.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    summary = {
        "output_manifest": str(manifest),
        "output_audio_dir": str(out_dir),
        "samples": len(rows),
        "duration_filter_sec": {
            "min": float(args.min_duration_sec),
            "max": float(args.max_duration_sec),
        },
        "selected_mean_duration_sec": float(np.mean([r["duration_sec"] for r in rows])),
        "selected_total_duration_sec": float(np.sum([r["duration_sec"] for r in rows])),
        "unique_language_families": sorted({r["language_family"] for r in rows}),
        "unique_countries": sorted({r["country"] for r in rows}),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
