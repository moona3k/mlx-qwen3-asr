#!/usr/bin/env python3
"""Build deterministic diarization manifests by mixing single-speaker clips.

Input manifest row requirements:
- `audio_path`
- `speaker_id`

Optional fields propagated when available:
- `sample_id`
- `language`
- `reference_text`
"""

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
    speaker_id: str
    language: str
    audio_path: Path
    reference_text: str


def _safe_component(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return safe.strip("_")


def _parse_manifest(path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        obj = json.loads(raw)
        speaker_id = str(obj.get("speaker_id", "")).strip()
        if not speaker_id:
            raise ValueError(f"Manifest row {i} missing non-empty speaker_id.")
        audio_path = Path(obj["audio_path"]).expanduser().resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Manifest row {i} references missing audio: {audio_path}")
        rows.append(
            ManifestRow(
                sample_id=str(obj.get("sample_id", f"manifest-{i:05d}")),
                speaker_id=speaker_id,
                language=str(obj.get("language", "unknown")),
                audio_path=audio_path,
                reference_text=str(obj.get("reference_text", "")),
            )
        )
    return rows


def _build_mix_plan(
    rows: list[ManifestRow],
    *,
    num_mixes: int,
    segments_per_mix: int,
    seed: int,
) -> list[list[ManifestRow]]:
    if num_mixes <= 0 or segments_per_mix <= 0 or not rows:
        return []

    by_speaker: dict[str, list[ManifestRow]] = {}
    for row in rows:
        by_speaker.setdefault(row.speaker_id, []).append(row)
    speakers = sorted(by_speaker.keys())
    if not speakers:
        return []

    rng = np.random.default_rng(seed)
    for speaker in speakers:
        items = by_speaker[speaker]
        if len(items) > 1:
            order = rng.permutation(len(items))
            by_speaker[speaker] = [items[int(i)] for i in order]

    cursor_by_speaker = {speaker: 0 for speaker in speakers}
    start_idx = int(seed % len(speakers))
    plans: list[list[ManifestRow]] = []

    for mix_idx in range(num_mixes):
        picked: list[ManifestRow] = []
        used: set[str] = set()
        cursor = start_idx + (mix_idx * segments_per_mix)
        attempts = 0
        max_attempts = max(len(speakers) * 6, segments_per_mix * 4)

        while len(picked) < segments_per_mix and attempts < max_attempts:
            speaker = speakers[cursor % len(speakers)]
            cursor += 1
            attempts += 1
            if speaker in used and len(used) < len(speakers):
                continue
            items = by_speaker.get(speaker) or []
            if not items:
                continue
            pos = cursor_by_speaker[speaker] % len(items)
            cursor_by_speaker[speaker] += 1
            picked.append(items[pos])
            used.add(speaker)

        # Fallback: allow repeated speakers if required to fill the mix.
        if len(picked) < segments_per_mix:
            fallback_cursor = 0
            max_fallback = max(segments_per_mix * len(speakers) * 3, 16)
            while len(picked) < segments_per_mix and fallback_cursor < max_fallback:
                speaker = speakers[(start_idx + mix_idx + fallback_cursor) % len(speakers)]
                fallback_cursor += 1
                items = by_speaker.get(speaker) or []
                if not items:
                    continue
                pos = cursor_by_speaker[speaker] % len(items)
                cursor_by_speaker[speaker] += 1
                picked.append(items[pos])

        if picked:
            plans.append(picked)
    return plans


def _read_audio_16k(path: Path) -> np.ndarray:
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "build_diarization_manifest requires soundfile. "
            "Install with: pip install 'mlx-qwen3-asr[eval]'"
        ) from exc
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1).astype(np.float32)
    return np.array(load_audio((audio, int(sr)))).astype(np.float32)


def _build_mixed_audio(
    rows: list[ManifestRow],
    *,
    silence_sec: float,
) -> tuple[np.ndarray, list[dict], list[str], str]:
    silence_samples = max(0, int(round(16000 * silence_sec)))
    silence = np.zeros((silence_samples,), dtype=np.float32)

    parts: list[np.ndarray] = []
    reference_segments: list[dict] = []
    source_ids: list[str] = []
    text_parts: list[str] = []
    cursor_sec = 0.0

    for idx, row in enumerate(rows):
        audio = _read_audio_16k(row.audio_path)
        if audio.size == 0:
            continue
        start = cursor_sec
        end = start + (float(audio.size) / 16000.0)
        reference_segments.append(
            {
                "speaker": row.speaker_id,
                "start": round(start, 6),
                "end": round(end, 6),
            }
        )
        source_ids.append(row.sample_id)
        if row.reference_text.strip():
            text_parts.append(f"[{row.speaker_id}] {row.reference_text.strip()}")
        parts.append(audio)
        cursor_sec = end
        if idx != len(rows) - 1 and silence_samples > 0:
            parts.append(silence)
            cursor_sec += float(silence_samples) / 16000.0

    if not parts:
        return np.zeros((1,), dtype=np.float32), [], source_ids, ""

    merged = np.concatenate(parts).astype(np.float32)
    peak = float(np.max(np.abs(merged))) if merged.size else 0.0
    if peak > 0.98:
        merged = (merged * (0.98 / peak)).astype(np.float32)
    return merged, reference_segments, source_ids, " ".join(text_parts).strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build deterministic diarization manifest from single-speaker clips."
    )
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument(
        "--output-manifest",
        default="docs/benchmarks/2026-02-15-diarization-manifest-20.jsonl",
    )
    parser.add_argument(
        "--output-audio-dir",
        default=str(Path.home() / ".cache" / "mlx-qwen3-asr" / "datasets" / "diarization-audio"),
    )
    parser.add_argument("--num-mixes", type=int, default=20)
    parser.add_argument("--segments-per-mix", type=int, default=2)
    parser.add_argument("--silence-sec", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260215)
    args = parser.parse_args()

    if args.num_mixes <= 0:
        raise ValueError("--num-mixes must be > 0")
    if args.segments_per_mix <= 0:
        raise ValueError("--segments-per-mix must be > 0")
    if args.silence_sec < 0.0:
        raise ValueError("--silence-sec must be >= 0")

    input_manifest = Path(args.input_manifest).expanduser().resolve()
    output_manifest = Path(args.output_manifest).expanduser().resolve()
    output_audio_dir = Path(args.output_audio_dir).expanduser().resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_audio_dir.mkdir(parents=True, exist_ok=True)

    rows = _parse_manifest(input_manifest)
    plans = _build_mix_plan(
        rows,
        num_mixes=int(args.num_mixes),
        segments_per_mix=int(args.segments_per_mix),
        seed=int(args.seed),
    )
    if not plans:
        raise RuntimeError("No mixes were generated. Check input manifest and arguments.")

    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "build_diarization_manifest requires soundfile. "
            "Install with: pip install 'mlx-qwen3-asr[eval]'"
        ) from exc

    out_rows: list[dict[str, object]] = []
    speakers_seen: set[str] = set()
    for mix_idx, plan in enumerate(plans):
        mixed, ref_segments, source_ids, reference_text = _build_mixed_audio(
            plan,
            silence_sec=float(args.silence_sec),
        )
        if not ref_segments:
            continue
        speaker_set = sorted({str(x["speaker"]) for x in ref_segments})
        speakers_seen.update(speaker_set)
        language_set = {row.language for row in plan if row.language.strip()}
        language = sorted(language_set)[0] if len(language_set) == 1 else "unknown"
        sample_id = f"diarization-mix-{mix_idx:04d}"
        wav_path = (output_audio_dir / f"{sample_id}.wav").resolve()
        sf.write(str(wav_path), mixed, samplerate=16000)
        out_rows.append(
            {
                "sample_id": sample_id,
                "subset": "diarization-mix",
                "speaker_id": "mixed",
                "language": language,
                "audio_path": str(wav_path),
                "condition": "diarization-mix-sequential",
                "source_sample_ids": source_ids,
                "source_speakers": speaker_set,
                "source_count": len(source_ids),
                "duration_sec": float(mixed.size) / 16000.0,
                "reference_text": reference_text,
                "reference_speaker_segments": ref_segments,
            }
        )

    if not out_rows:
        raise RuntimeError("Generated mixes were empty. No output rows written.")

    with output_manifest.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    payload = {
        "input_manifest": str(input_manifest),
        "output_manifest": str(output_manifest),
        "output_audio_dir": str(output_audio_dir),
        "num_input_rows": len(rows),
        "num_mixes_requested": int(args.num_mixes),
        "num_mixes_written": len(out_rows),
        "segments_per_mix": int(args.segments_per_mix),
        "silence_sec": float(args.silence_sec),
        "seed": int(args.seed),
        "unique_speakers_covered": len(speakers_seen),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
