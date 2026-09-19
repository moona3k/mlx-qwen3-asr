#!/usr/bin/env python3
"""Convert Qwen3-ASR weights and publish quantized artifacts to HuggingFace.

Two modes:

1. Convert and upload in one go (CI workflow ``publish-quantized.yml``)::

    HF_TOKEN=... python scripts/publish_quantized.py \
      --source-model Qwen/Qwen3-ASR-0.6B \
      --repo-id moona3k/mlx-qwen3-asr-0.6b-4bit \
      --bits 4

2. Upload a directory that was already produced by ``scripts/convert.py`` and
   validated locally (keeps its ``README.md`` model card if present)::

    HF_TOKEN=... python scripts/publish_quantized.py \
      --from-dir ~/.cache/mlx-qwen3-asr/publish/Qwen3-ASR-0.6B-4bit-g64 \
      --repo-id moona3k/mlx-qwen3-asr-0.6b-4bit

Pass ``--output-dir`` in mode 1 to keep the converted files instead of a
temporary directory, and ``--skip-upload`` to stop after conversion.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REQUIRED_FILES = ("config.json", "vocab.json", "merges.txt", "quantization_config.json")


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def _convert(args: argparse.Namespace, out_dir: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    convert_script = project_root / "scripts" / "convert.py"
    _run(
        [
            sys.executable,
            str(convert_script),
            "--model",
            args.source_model,
            "--output-dir",
            str(out_dir),
            "--quantize",
            str(args.bits),
            "--group-size",
            str(args.group_size),
            "--dtype",
            args.dtype,
        ],
        cwd=project_root,
    )


def _default_card(args: argparse.Namespace) -> str:
    return (
        f"# {args.repo_id}\n\n"
        f"Quantized MLX conversion of `{args.source_model}` for "
        "[mlx-qwen3-asr](https://github.com/moona3k/mlx-qwen3-asr).\n\n"
        f"- bits: {args.bits}\n"
        f"- group_size: {args.group_size}\n"
        f"- dtype: {args.dtype}\n"
    )


def _validate_dir(model_dir: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (model_dir / name).exists()]
    if not any(model_dir.glob("*.safetensors")):
        missing.append("*.safetensors")
    if missing:
        raise SystemExit(f"{model_dir} is missing: {', '.join(missing)}")


def _upload(model_dir: Path, args: argparse.Namespace, token: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message
        or f"Add {args.bits}-bit MLX quantized weights from {args.source_model}",
    )
    print(f"Uploaded quantized model to https://huggingface.co/{args.repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert and upload quantized MLX Qwen3-ASR model to HuggingFace"
    )
    parser.add_argument(
        "--source-model",
        default="Qwen/Qwen3-ASR-0.6B",
        help="Source HuggingFace model ID or local model path",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target HuggingFace repo ID, e.g. moona3k/mlx-qwen3-asr-0.6b-4bit",
    )
    parser.add_argument("--bits", type=int, choices=[4, 8], default=None, help="Quantization bits")
    parser.add_argument("--group-size", type=int, default=64, help="Quantization group size")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Intermediate conversion dtype",
    )
    parser.add_argument(
        "--from-dir",
        default=None,
        help="Upload this already-converted directory instead of converting",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Keep converted files here instead of a temporary directory",
    )
    parser.add_argument("--skip-upload", action="store_true", help="Convert only")
    parser.add_argument("--commit-message", default=None, help="Override the upload commit message")
    parser.add_argument("--private", action="store_true", help="Create the target repo as private")
    args = parser.parse_args()

    if args.from_dir is None and args.bits is None:
        parser.error("--bits is required unless --from-dir is given")

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token and not args.skip_upload:
        raise RuntimeError("Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) before publishing.")

    if args.from_dir is not None:
        model_dir = Path(args.from_dir).expanduser().resolve()
        _validate_dir(model_dir)
        if not (model_dir / "README.md").exists():
            (model_dir / "README.md").write_text(_default_card(args), encoding="utf-8")
        if not args.skip_upload:
            _upload(model_dir, args, token or "")
        return

    if args.output_dir is not None:
        out_dir = Path(args.output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        _convert(args, out_dir)
        _validate_dir(out_dir)
        if not (out_dir / "README.md").exists():
            (out_dir / "README.md").write_text(_default_card(args), encoding="utf-8")
        if not args.skip_upload:
            _upload(out_dir, args, token or "")
        return

    with tempfile.TemporaryDirectory(prefix="mlx-qwen3-asr-quant-") as tmpdir:
        out_dir = Path(tmpdir) / "model"
        _convert(args, out_dir)
        _validate_dir(out_dir)
        (out_dir / "README.md").write_text(_default_card(args), encoding="utf-8")
        if not args.skip_upload:
            _upload(out_dir, args, token or "")


if __name__ == "__main__":
    main()
