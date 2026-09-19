#!/usr/bin/env python3
"""Convert and optionally quantize Qwen3-ASR weights for MLX.

Usage:
    python scripts/convert.py --model Qwen/Qwen3-ASR-1.7B --output-dir ./mlx-model
    python scripts/convert.py --model Qwen/Qwen3-ASR-1.7B --quantize 4 --output-dir ./mlx-model-4bit
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import _repo_path  # noqa: F401
import mlx.core as mx
import mlx.utils as mlx_utils

from mlx_qwen3_asr.config import Qwen3ASRConfig
from mlx_qwen3_asr.convert import quantize_model, remap_weights
from mlx_qwen3_asr.load_models import _load_safetensors, _resolve_path
from mlx_qwen3_asr.model import Qwen3ASRModel


def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen3-ASR weights for MLX"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-ASR-1.7B",
        help="HuggingFace repo ID or local path",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for converted weights",
    )
    parser.add_argument(
        "--quantize",
        type=int,
        choices=[4, 8],
        default=None,
        help="Quantize to N bits (4 or 8)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Quantization group size (default: 64)",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Weight dtype (default: float16)",
    )
    args = parser.parse_args()

    dtype_map = {
        "float16": mx.float16,
        "float32": mx.float32,
        "bfloat16": mx.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Resolve and load
    model_path = _resolve_path(args.model)
    print(f"Loading model from {model_path}")

    with open(model_path / "config.json") as f:
        raw_config = json.load(f)
    config = Qwen3ASRConfig.from_dict(raw_config)

    weights = _load_safetensors(model_path)
    weights = remap_weights(weights)
    print(f"Loaded {len(weights)} weight tensors")

    # Cast weights to target dtype
    if dtype != mx.float32:
        weights = {k: v.astype(dtype) for k, v in weights.items()}

    # Instantiate model
    model = Qwen3ASRModel(config)
    model.load_weights(list(weights.items()))

    # Quantize if requested
    if args.quantize:
        print(f"Quantizing to {args.quantize}-bit (group_size={args.group_size})")
        quantize_model(model, bits=args.quantize, group_size=args.group_size)

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save weights (tree_flatten produces flat key-value pairs from nested params)
    weight_path = output_dir / "weights.safetensors"
    flat_weights = dict(mlx_utils.tree_flatten(model.parameters()))
    mx.save_safetensors(str(weight_path), flat_weights)
    print(f"Saved weights to {weight_path}")

    # Copy config and tokenizer files
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                   "vocab.json", "merges.txt", "special_tokens_map.json"]:
        src = model_path / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)

    if args.quantize:
        quant_cfg = {
            "bits": int(args.quantize),
            "group_size": int(args.group_size),
        }
        (output_dir / "quantization_config.json").write_text(
            json.dumps(quant_cfg, indent=2),
            encoding="utf-8",
        )

    print(f"Conversion complete: {output_dir}")


if __name__ == "__main__":
    main()
