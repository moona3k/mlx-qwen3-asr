#!/usr/bin/env python3
"""Evaluate forced-aligner parity between MLX and qwen_asr backends."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_qwen3_asr.audio import load_audio
from mlx_qwen3_asr.forced_aligner import ForcedAligner

OPENSLR_BASE = "https://www.openslr.org/resources/12"
SPLIT_ARCHIVES = {
    "test-clean": "test-clean.tar.gz",
    "test-other": "test-other.tar.gz",
}


@dataclass(frozen=True)
class LibriSample:
    sample_id: str
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


def _collect_samples(split_root: Path, max_samples: int) -> list[LibriSample]:
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
                        LibriSample(sample_id=sample_id, audio_path=audio_path, text=text)
                    )
                if len(samples) >= max_samples:
                    return samples
    return samples


def _norm_text_list(items) -> list[str]:
    out = []
    for it in items:
        tok = getattr(it, "text", "")
        tok = str(tok).strip()
        if tok:
            out.append(tok.lower())
    return out


def _load_qwen_asr_reference(model_path: str):
    try:
        from qwen_asr import Qwen3ForcedAligner  # type: ignore
    except ImportError as exc:  # pragma: no cover - integration dependency gate
        raise RuntimeError(
            "Aligner parity reference requires optional dependency `qwen-asr`. "
            "Install with: pip install qwen-asr"
        ) from exc
    return Qwen3ForcedAligner.from_pretrained(model_path, device_map="cpu")


def _align_qwen_asr(
    ref_backend,
    audio: np.ndarray,
    text: str,
    language: str,
):
    out = ref_backend.align(
        audio=[(audio.astype(np.float32), 16000)],
        text=[text],
        language=[language],
    )
    if not out:
        return []
    first = out[0]
    items = getattr(first, "items", None)
    return items if items is not None else first


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate MLX/qwen_asr aligner parity.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-ForcedAligner-0.6B",
        help="Forced aligner model ID or local path.",
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
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of deterministic samples to evaluate.",
    )
    parser.add_argument(
        "--language",
        default="English",
        help="Language label passed to both aligner backends.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path.home() / ".cache" / "mlx-qwen3-asr" / "datasets"),
        help="Cache directory for downloaded LibriSpeech archives.",
    )
    parser.add_argument(
        "--fail-text-match-rate-below",
        type=float,
        default=None,
        help="Fail when word-text match rate falls below threshold (0..1).",
    )
    parser.add_argument(
        "--fail-timing-mae-ms-above",
        type=float,
        default=None,
        help="Fail when matched-word timing MAE (ms) exceeds threshold.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional JSON output path.",
    )
    args = parser.parse_args()

    dtype = _dtype_from_name(args.dtype)
    data_dir = Path(args.data_dir).expanduser().resolve()
    split_root = _ensure_split(data_dir, args.subset)
    selected = _collect_samples(split_root, max_samples=max(1, args.samples))
    if not selected:
        raise RuntimeError(f"No samples found under {split_root}")

    aligner_mlx = ForcedAligner(model_path=args.model, dtype=dtype, backend="mlx")
    aligner_qwen = _load_qwen_asr_reference(args.model)

    rows: list[dict] = []
    mlx_lat: list[float] = []
    qwen_lat: list[float] = []
    start_err_ms: list[float] = []
    end_err_ms: list[float] = []

    for sample in selected:
        audio = np.array(load_audio(str(sample.audio_path))).astype(np.float32)

        t0 = time.perf_counter()
        mlx_items = aligner_mlx.align(audio, sample.text, args.language)
        mlx_lat.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        qwen_items = _align_qwen_asr(aligner_qwen, audio, sample.text, args.language)
        qwen_lat.append(time.perf_counter() - t1)

        mlx_words = _norm_text_list(mlx_items)
        qwen_words = _norm_text_list(qwen_items)
        text_match = mlx_words == qwen_words

        row = {
            "sample_id": sample.sample_id,
            "audio_path": str(sample.audio_path),
            "n_words_mlx": len(mlx_words),
            "n_words_qwen_asr": len(qwen_words),
            "text_match": text_match,
            "latency_sec_mlx": mlx_lat[-1],
            "latency_sec_qwen_asr": qwen_lat[-1],
        }

        if text_match and mlx_items:
            s_err = []
            e_err = []
            for a, b in zip(mlx_items, qwen_items, strict=True):
                s_err.append(abs(float(a.start_time) - float(b.start_time)) * 1000.0)
                e_err.append(abs(float(a.end_time) - float(b.end_time)) * 1000.0)
            row["start_mae_ms"] = float(np.mean(s_err))
            row["end_mae_ms"] = float(np.mean(e_err))
            start_err_ms.extend(s_err)
            end_err_ms.extend(e_err)
        else:
            row["start_mae_ms"] = None
            row["end_mae_ms"] = None

        rows.append(row)

    matched = [r for r in rows if r["text_match"]]
    text_match_rate = len(matched) / len(rows) if rows else 0.0

    start_mae = float(np.mean(start_err_ms)) if start_err_ms else None
    end_mae = float(np.mean(end_err_ms)) if end_err_ms else None
    all_err = (start_err_ms + end_err_ms)
    all_mae = float(np.mean(all_err)) if all_err else None

    payload = {
        "suite": "aligner-parity-v1",
        "subset": args.subset,
        "samples": len(rows),
        "model": args.model,
        "dtype": args.dtype,
        "language": args.language,
        "text_match_rate": text_match_rate,
        "timing_mae_ms_start": start_mae,
        "timing_mae_ms_end": end_mae,
        "timing_mae_ms_all": all_mae,
        "latency_sec_mlx_mean": float(np.mean(mlx_lat)) if mlx_lat else None,
        "latency_sec_qwen_asr_mean": float(np.mean(qwen_lat)) if qwen_lat else None,
        "latency_speedup_qwen_over_mlx": (
            (float(np.mean(qwen_lat)) / float(np.mean(mlx_lat)))
            if mlx_lat and qwen_lat and float(np.mean(mlx_lat)) > 0
            else None
        ),
        "rows": rows,
    }

    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if (
        args.fail_text_match_rate_below is not None
        and text_match_rate < args.fail_text_match_rate_below
    ):
        print(
            "Aligner parity text-match gate failed: "
            f"rate={text_match_rate:.6f} < threshold={args.fail_text_match_rate_below:.6f}",
            file=sys.stderr,
        )
        return 2

    if args.fail_timing_mae_ms_above is not None and (
        all_mae is None or all_mae > args.fail_timing_mae_ms_above
    ):
        shown = "None" if all_mae is None else f"{all_mae:.6f}"
        print(
            "Aligner parity timing gate failed: "
            f"mae_ms={shown} > threshold={args.fail_timing_mae_ms_above:.6f}",
            file=sys.stderr,
        )
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
