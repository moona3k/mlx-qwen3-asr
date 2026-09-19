#!/usr/bin/env python3
"""Build deterministic long-form manifest by concatenating short manifest rows."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import _repo_path  # noqa: F401
import numpy as np

from mlx_qwen3_asr.audio import load_audio


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    subset: str
    speaker_id: str
    language: str
    audio_path: Path
    reference_text: str


def _parse_manifest(path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        row = line.strip()
        if not row:
            continue
        obj = json.loads(row)
        audio_path = Path(obj["audio_path"]).expanduser().resolve()
        rows.append(
            ManifestRow(
                sample_id=str(obj.get("sample_id", f"manifest-{i:05d}")),
                subset=str(obj.get("subset", "manifest")),
                speaker_id=str(obj.get("speaker_id", "unknown")),
                language=str(obj.get("language", "unknown")),
                audio_path=audio_path,
                reference_text=str(obj.get("reference_text", "")),
            )
        )
    return rows


def _read_audio_16k(path: Path) -> np.ndarray:
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "build_longform_manifest requires soundfile. "
            "Install with: pip install 'mlx-qwen3-asr[eval]'"
        ) from exc
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1).astype(np.float32)
    return np.array(load_audio((audio, int(sr)))).astype(np.float32)


def _build_concat(
    rows: list[ManifestRow],
    target_samples: int,
    silence_samples: int,
    start_cursor: int,
) -> tuple[np.ndarray, list[str], list[str], int]:
    if not rows:
        raise ValueError("Cannot build longform concat from empty row list.")
    out_parts: list[np.ndarray] = []
    source_ids: list[str] = []
    source_texts: list[str] = []
    cursor = start_cursor
    total = 0

    while total < target_samples:
        row = rows[cursor % len(rows)]
        cursor += 1
        clip = _read_audio_16k(row.audio_path)
        if clip.size == 0:
            continue
        out_parts.append(clip)
        total += int(clip.size)
        source_ids.append(row.sample_id)
        if row.reference_text:
            source_texts.append(row.reference_text.strip())
        if silence_samples > 0 and total < target_samples:
            out_parts.append(np.zeros(silence_samples, dtype=np.float32))
            total += silence_samples

    merged = np.concatenate(out_parts).astype(np.float32)
    return merged, source_ids, source_texts, cursor


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build long-form manifest by deterministic concatenation."
    )
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument(
        "--output-manifest",
        default="docs/benchmarks/longform-manifest.jsonl",
    )
    parser.add_argument(
        "--output-audio-dir",
        default=str(Path.home() / ".cache" / "mlx-qwen3-asr" / "datasets" / "longform-audio"),
    )
    parser.add_argument("--target-duration-sec", type=float, default=75.0)
    parser.add_argument("--clips-per-language", type=int, default=1)
    parser.add_argument("--silence-sec", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260215)
    args = parser.parse_args()

    rows = _parse_manifest(Path(args.input_manifest).expanduser().resolve())
    by_language: dict[str, list[ManifestRow]] = {}
    for row in rows:
        by_language.setdefault(row.language, []).append(row)
    for lang in by_language:
        by_language[lang] = sorted(by_language[lang], key=lambda r: r.sample_id)

    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "build_longform_manifest requires soundfile. "
            "Install with: pip install 'mlx-qwen3-asr[eval]'"
        ) from exc

    target_samples = max(1, int(round(args.target_duration_sec * 16000)))
    silence_samples = max(0, int(round(args.silence_sec * 16000)))
    audio_root = Path(args.output_audio_dir).expanduser().resolve()
    audio_root.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.output_manifest).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    out_rows: list[dict[str, object]] = []
    for lang_idx, (language, items) in enumerate(sorted(by_language.items())):
        if not items:
            continue
        cursor = int((args.seed + lang_idx) % max(1, len(items)))
        safe_lang = "".join(ch if ch.isalnum() else "_" for ch in language).strip("_").lower()
        lang_dir = audio_root / safe_lang
        lang_dir.mkdir(parents=True, exist_ok=True)

        for clip_idx in range(max(1, args.clips_per_language)):
            merged, src_ids, src_texts, cursor = _build_concat(
                items,
                target_samples=target_samples,
                silence_samples=silence_samples,
                start_cursor=cursor,
            )
            sample_id = f"longform-{safe_lang}-{clip_idx:03d}"
            wav_path = (lang_dir / f"{sample_id}.wav").resolve()
            sf.write(str(wav_path), merged, samplerate=16000)

            out_rows.append(
                {
                    "sample_id": sample_id,
                    "subset": f"longform-{safe_lang}",
                    "speaker_id": f"mixed-{safe_lang}",
                    "language": language,
                    "audio_path": str(wav_path),
                    "condition": "longform-concat",
                    "source_sample_ids": src_ids,
                    "source_count": len(src_ids),
                    "target_duration_sec": args.target_duration_sec,
                    "duration_sec": float(merged.size) / 16000.0,
                    "reference_text": " ".join(src_texts).strip(),
                }
            )

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    payload = {
        "input_manifest": str(Path(args.input_manifest).expanduser().resolve()),
        "output_manifest": str(manifest_path),
        "output_audio_dir": str(audio_root),
        "languages": sorted(by_language.keys()),
        "clips_per_language": args.clips_per_language,
        "target_duration_sec": args.target_duration_sec,
        "silence_sec": args.silence_sec,
        "samples_total": len(out_rows),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
