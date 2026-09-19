#!/usr/bin/env python3
"""Evaluate mlx-qwen3-asr on LibriSpeech with deterministic sampling."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

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

from eval.metrics import compute_cer, compute_wer, normalize_text  # noqa: E402

OPENSLR_BASE = "https://www.openslr.org/resources/12"
SPLIT_ARCHIVES = {
    "test-clean": "test-clean.tar.gz",
    "test-other": "test-other.tar.gz",
}


@dataclass(frozen=True)
class LibriSample:
    sample_id: str
    speaker_id: str
    audio_path: Path
    text: str


def _dtype_from_name(name: str) -> mx.Dtype:
    mapping = {
        "float16": mx.float16,
        "float32": mx.float32,
        "bfloat16": mx.bfloat16,
    }
    return mapping[name]


def _download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as src, dst.open("wb") as out:  # noqa: S310
        out.write(src.read())


def _extract_archive(archive_path: Path, dst_dir: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=dst_dir)


def _ensure_split(data_dir: Path, subset: str) -> Path:
    if subset not in SPLIT_ARCHIVES:
        supported = ", ".join(sorted(SPLIT_ARCHIVES))
        raise ValueError(f"Unsupported subset '{subset}'. Supported: {supported}")

    split_root = data_dir / "LibriSpeech" / subset
    if split_root.exists():
        return split_root

    archive_name = SPLIT_ARCHIVES[subset]
    archive_path = data_dir / archive_name
    if not archive_path.exists():
        url = f"{OPENSLR_BASE}/{archive_name}"
        print(f"Downloading {url} -> {archive_path}")
        _download_file(url, archive_path)

    print(f"Extracting {archive_path} -> {data_dir}")
    _extract_archive(archive_path, data_dir)
    if not split_root.exists():
        raise FileNotFoundError(f"Expected split path not found after extract: {split_root}")
    return split_root


def _speaker_id_from_sample_id(sample_id: str) -> str:
    parts = sample_id.split("-", 1)
    return parts[0] if parts else sample_id


def _collect_all_samples(split_root: Path) -> list[LibriSample]:
    samples: list[LibriSample] = []
    for trans_path in sorted(split_root.rglob("*.trans.txt")):
        with trans_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = line.strip()
                if not row:
                    continue
                sample_id, text = row.split(" ", 1)
                audio_path = trans_path.parent / f"{sample_id}.flac"
                if audio_path.exists():
                    samples.append(
                        LibriSample(
                            sample_id=sample_id,
                            speaker_id=_speaker_id_from_sample_id(sample_id),
                            audio_path=audio_path,
                            text=text,
                        )
                    )
    return samples


def _collect_samples(
    split_root: Path,
    max_samples: int,
    sampling: str,
) -> list[LibriSample]:
    all_samples = _collect_all_samples(split_root)
    if sampling == "sequential":
        return all_samples[:max_samples]
    if sampling != "speaker_round_robin":
        raise ValueError(
            f"Unsupported sampling strategy '{sampling}'. "
            "Supported: sequential, speaker_round_robin"
        )

    by_speaker: dict[str, list[LibriSample]] = {}
    for sample in all_samples:
        by_speaker.setdefault(sample.speaker_id, []).append(sample)

    speakers = sorted(by_speaker)
    selected: list[LibriSample] = []
    idx = 0
    while len(selected) < max_samples:
        added_this_round = False
        for speaker in speakers:
            speaker_samples = by_speaker[speaker]
            if idx < len(speaker_samples):
                selected.append(speaker_samples[idx])
                added_this_round = True
                if len(selected) >= max_samples:
                    break
        if not added_this_round:
            break
        idx += 1
    return selected


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - dependency message path
        raise RuntimeError(
            "soundfile is required for LibriSpeech evaluation. "
            "Install with: pip install 'mlx-qwen3-asr[eval]'"
        ) from exc

    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1).astype(np.float32)
    else:
        audio = audio.astype(np.float32)
    return audio, int(sr)


def _threshold_failures(
    *,
    wer: float,
    cer: float,
    fail_wer_above: float | None,
    fail_cer_above: float | None,
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
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate model quality on LibriSpeech samples.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-ASR-0.6B",
        help="Model ID or local path.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32", "bfloat16"],
        default="float16",
    )
    parser.add_argument(
        "--subset",
        choices=sorted(SPLIT_ARCHIVES),
        default="test-clean",
        help="LibriSpeech subset to evaluate.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of deterministic samples to evaluate.",
    )
    parser.add_argument(
        "--sampling",
        choices=["speaker_round_robin", "sequential"],
        default="speaker_round_robin",
        help=(
            "Deterministic sampling strategy. "
            "speaker_round_robin balances across speakers (default)."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path.home() / ".cache" / "mlx-qwen3-asr" / "datasets"),
        help="Cache directory for downloaded LibriSpeech archives.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens per sample.",
    )
    parser.add_argument(
        "--fail-wer-above",
        type=float,
        default=None,
        help="Fail with non-zero exit code if WER exceeds this threshold.",
    )
    parser.add_argument(
        "--fail-cer-above",
        type=float,
        default=None,
        help="Fail with non-zero exit code if CER exceeds this threshold.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional JSON output path.",
    )
    args = parser.parse_args()

    started = time.perf_counter()

    data_dir = Path(args.data_dir).expanduser().resolve()
    split_root = _ensure_split(data_dir, args.subset)
    selected = _collect_samples(
        split_root,
        max_samples=max(1, args.samples),
        sampling=args.sampling,
    )
    if not selected:
        raise RuntimeError(f"No samples found under {split_root}")

    dtype = _dtype_from_name(args.dtype)
    model, _ = load_model(args.model, dtype=dtype)

    references: list[str] = []
    hypotheses: list[str] = []
    sample_rows: list[dict] = []
    latencies: list[float] = []
    total_audio_sec = 0.0

    for idx, sample in enumerate(selected, start=1):
        audio, sr = _read_audio(sample.audio_path)
        total_audio_sec += len(audio) / float(sr)

        t0 = time.perf_counter()
        result = transcribe(
            (audio, sr),
            model=model,
            language="English",
            max_new_tokens=args.max_new_tokens,
            verbose=False,
        )
        latency = time.perf_counter() - t0
        latencies.append(latency)

        reference = normalize_text(sample.text)
        hypothesis = normalize_text(result.text)
        references.append(reference)
        hypotheses.append(hypothesis)

        sample_rows.append(
            {
                "index": idx,
                "sample_id": sample.sample_id,
                "speaker_id": sample.speaker_id,
                "audio_path": str(sample.audio_path),
                "reference": reference,
                "hypothesis": hypothesis,
                "latency_sec": latency,
            }
        )

    wer = compute_wer(references, hypotheses)
    cer = compute_cer(references, hypotheses)
    elapsed = time.perf_counter() - started
    mean_latency = float(np.mean(latencies)) if latencies else 0.0

    speaker_counts: dict[str, int] = {}
    for sample in selected:
        speaker_counts[sample.speaker_id] = speaker_counts.get(sample.speaker_id, 0) + 1

    payload = {
        "suite": "librispeech-v1",
        "subset": args.subset,
        "sampling": args.sampling,
        "samples": len(selected),
        "unique_speakers": len(speaker_counts),
        "speaker_counts": speaker_counts,
        "model": args.model,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "wer": wer,
        "cer": cer,
        "mean_latency_sec": mean_latency,
        "audio_duration_sec_total": total_audio_sec,
        "rtf": (sum(latencies) / total_audio_sec) if total_audio_sec > 0 else 0.0,
        "elapsed_sec": elapsed,
        "rows": sample_rows,
    }

    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    failures = _threshold_failures(
        wer=wer,
        cer=cer,
        fail_wer_above=args.fail_wer_above,
        fail_cer_above=args.fail_cer_above,
    )
    if failures:
        for msg in failures:
            print(msg, file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
