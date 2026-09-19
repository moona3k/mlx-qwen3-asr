#!/usr/bin/env python3
"""Build deterministic multilingual JSONL manifests for parity runs.

Uses direct Hugging Face Hub artifacts (TSV + audio tar) so it works with
`datasets>=4`, where script-based dataset loaders are removed.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
from pathlib import Path

import _repo_path  # noqa: F401
import numpy as np
from huggingface_hub import HfApi, hf_hub_download

from mlx_qwen3_asr.audio import load_audio

LANGUAGE_LABELS = {
    "en_us": "English",
    "zh_cn": "Chinese",
    "cmn_hans_cn": "Chinese",
    "yue_hant_hk": "Cantonese",
    "ja_jp": "Japanese",
    "de_de": "German",
    "fr_fr": "French",
    "es_419": "Spanish",
    "ru_ru": "Russian",
    "ar_eg": "Arabic",
    "hi_in": "Hindi",
    "ko_kr": "Korean",
    "it_it": "Italian",
    "pt_br": "Portuguese",
    "tr_tr": "Turkish",
    "vi_vn": "Vietnamese",
    "th_th": "Thai",
    "nl_nl": "Dutch",
    "pl_pl": "Polish",
}

LANGUAGE_ALIASES = {
    "en": "en_us",
    "zh": "cmn_hans_cn",
    "zh_cn": "cmn_hans_cn",
    "cmn_cn": "cmn_hans_cn",
    "cantonese": "yue_hant_hk",
    "yue_hk": "yue_hant_hk",
    "zh_hk": "yue_hant_hk",
    "ja": "ja_jp",
    "de": "de_de",
    "fr": "fr_fr",
    "es": "es_419",
    "ru": "ru_ru",
    "ar": "ar_eg",
    "hi": "hi_in",
    "ko": "ko_kr",
}

FLEURS_TSV_COLUMNS = [
    "speaker_id",
    "audio_file",
    "transcription",
    "raw_transcription",
    "tokenized_transcription",
    "num_samples",
    "gender",
]


def _language_label_for_config(config_name: str) -> str:
    return LANGUAGE_LABELS.get(config_name, config_name.replace("_", "-"))


def _select_indices(total: int, take: int, seed: int) -> list[int]:
    if total <= 0 or take <= 0:
        return []
    if take >= total:
        return list(range(total))
    rng = np.random.default_rng(seed)
    selected = sorted(int(i) for i in rng.choice(total, size=take, replace=False))
    return selected


def _sanitize_id(value: object) -> str:
    raw = str(value)
    out = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw).strip("_")
    return out or "sample"


def _available_fleurs_configs(dataset: str) -> set[str]:
    api = HfApi()
    files = api.list_repo_files(dataset, repo_type="dataset")
    return {
        p.split("/")[1]
        for p in files
        if p.startswith("data/") and p.endswith("/test.tsv")
    }


def _resolve_language_configs(requested: list[str], available: set[str]) -> list[str]:
    out: list[str] = []
    unknown: list[str] = []
    for entry in requested:
        key = entry.strip()
        if not key:
            continue
        normalized = LANGUAGE_ALIASES.get(key, key)
        if normalized not in available:
            unknown.append(key)
            continue
        out.append(normalized)
    if unknown:
        sample = ", ".join(sorted(list(available))[:20])
        raise ValueError(
            "Unsupported FLEURS configs: "
            + ", ".join(unknown)
            + ". Example valid configs: "
            + sample
        )
    return out


def _parse_fleurs_tsv(tsv_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with tsv_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for fields in reader:
            if not fields:
                continue
            padded = fields + [""] * max(0, len(FLEURS_TSV_COLUMNS) - len(fields))
            row = {
                key: padded[idx]
                for idx, key in enumerate(FLEURS_TSV_COLUMNS)
            }
            if row["audio_file"]:
                rows.append(row)
    return rows


def _load_audio_from_tar(
    tar: tarfile.TarFile,
    split: str,
    audio_file: str,
) -> tuple[np.ndarray, int]:
    member_name = f"{split}/{audio_file}"
    member = tar.getmember(member_name)
    extracted = tar.extractfile(member)
    if extracted is None:
        raise FileNotFoundError(f"Failed to extract member: {member_name}")

    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "build_multilingual_manifest requires soundfile. "
            "Install with: pip install 'mlx-qwen3-asr[eval]'"
        ) from exc

    buf = io.BytesIO(extracted.read())
    audio, sr = sf.read(buf, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1).astype(np.float32)
    return audio.astype(np.float32), int(sr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build multilingual manifest JSONL from HF FLEURS files."
    )
    parser.add_argument("--dataset", default="google/fleurs")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--languages",
        default="en_us,zh_cn,ja_jp,de_de,fr_fr,es_419,ru_ru,ar_eg,hi_in,ko_kr",
        help="Comma-separated FLEURS language configs.",
    )
    parser.add_argument("--samples-per-language", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260214)
    parser.add_argument(
        "--audio-dir",
        default=str(Path.home() / ".cache" / "mlx-qwen3-asr" / "datasets" / "fleurs-audio"),
    )
    parser.add_argument(
        "--output-manifest",
        default="docs/benchmarks/fleurs-multilingual-manifest.jsonl",
    )
    args = parser.parse_args()

    audio_root = Path(args.audio_dir).expanduser().resolve()
    manifest_path = Path(args.output_manifest).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    audio_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    requested_langs = [x.strip() for x in args.languages.split(",") if x.strip()]
    available_configs = _available_fleurs_configs(args.dataset)
    langs = _resolve_language_configs(requested_langs, available_configs)
    for lang_idx, lang_cfg in enumerate(langs):
        tsv_file = f"data/{lang_cfg}/{args.split}.tsv"
        tar_file = f"data/{lang_cfg}/audio/{args.split}.tar.gz"
        tsv_path = Path(
            hf_hub_download(
                repo_id=args.dataset,
                repo_type="dataset",
                filename=tsv_file,
            )
        )
        tar_path = Path(
            hf_hub_download(
                repo_id=args.dataset,
                repo_type="dataset",
                filename=tar_file,
            )
        )

        metadata = _parse_fleurs_tsv(tsv_path)
        picks = _select_indices(len(metadata), args.samples_per_language, seed=args.seed + lang_idx)
        label = _language_label_for_config(lang_cfg)
        subset = f"fleurs-{args.split}-{lang_cfg}"
        lang_audio_dir = audio_root / lang_cfg
        lang_audio_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(tar_path, "r:gz") as tar:
            for pick_idx in picks:
                item = metadata[int(pick_idx)]
                audio_file = item["audio_file"]
                arr, sr = _load_audio_from_tar(tar, split=args.split, audio_file=audio_file)
                audio_16k = np.array(load_audio((arr, sr))).astype(np.float32)

                sample_suffix = _sanitize_id(Path(audio_file).stem)
                sample_id = f"{lang_cfg}-{sample_suffix}"
                wav_path = (lang_audio_dir / f"{sample_id}.wav").resolve()

                try:
                    import soundfile as sf
                except ImportError as exc:  # pragma: no cover
                    raise RuntimeError(
                        "build_multilingual_manifest requires soundfile. "
                        "Install with: pip install 'mlx-qwen3-asr[eval]'"
                    ) from exc
                sf.write(str(wav_path), audio_16k, samplerate=16000)

                rows.append(
                    {
                        "sample_id": sample_id,
                        "subset": subset,
                        "speaker_id": item.get("speaker_id", "unknown"),
                        "language": label,
                        "audio_path": str(wav_path),
                        "condition": "manifest-fleurs",
                        "dataset": args.dataset,
                        "dataset_config": lang_cfg,
                        "dataset_split": args.split,
                        "reference_text": item.get("transcription", ""),
                    }
                )

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    payload = {
        "manifest_path": str(manifest_path),
        "audio_dir": str(audio_root),
        "dataset": args.dataset,
        "split": args.split,
        "languages": langs,
        "samples_per_language": args.samples_per_language,
        "samples_total": len(rows),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
