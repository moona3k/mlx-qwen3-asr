"""Weight remapping and conversion for Qwen3-ASR HuggingFace -> MLX."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def remap_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Remap HuggingFace weight keys to MLX model keys.

    Two transformations:
    1. Strip 'thinker.' prefix from all keys
       HF: thinker.audio_tower.conv2d1.weight -> audio_tower.conv2d1.weight
       HF: thinker.model.layers.0.self_attn.q_proj.weight -> model.layers.0.self_attn.q_proj.weight
       HF: thinker.lm_head.weight -> lm_head.weight

    2. Transpose Conv2d weights from PyTorch to MLX format
       PyTorch: (out_channels, in_channels, kH, kW)
       MLX:     (out_channels, kH, kW, in_channels)
       Transform: transpose(0, 2, 3, 1)

    Returns:
        Remapped weights dict ready for model.load_weights()
    """
    remapped = {}

    for key, value in weights.items():
        # Strip thinker. prefix
        new_key = key
        had_thinker_prefix = new_key.startswith("thinker.")
        if new_key.startswith("thinker."):
            new_key = new_key[len("thinker."):]

        # Transpose Conv2d weights
        # Conv2d weight keys match pattern: audio_tower.conv2d{1,2,3}.weight
        if (
            had_thinker_prefix
            and "conv2d" in new_key
            and new_key.endswith(".weight")
            and value.ndim == 4
        ):
            # PyTorch (out, in, kH, kW) -> MLX (out, kH, kW, in)
            value = value.transpose(0, 2, 3, 1)

        remapped[new_key] = value

    return remapped


def quantize_model(
    model: nn.Module,
    bits: int = 4,
    group_size: int = 64,
    encoder_bits: int | None = None,
) -> nn.Module:
    """Quantize model Linear and Embedding layers.

    Args:
        model: The model to quantize
        bits: Quantization bits for the text decoder (4 or 8)
        group_size: Quantization group size
        encoder_bits: Bits for the audio encoder (``audio_tower.*``). ``None``
            uses ``bits``; ``16`` leaves the encoder unquantized. On the 0.6B
            model the encoder accounts for most of the 4-bit quality loss
            (LibriSpeech WER 2.63% all-4-bit vs 2.37% with an 8-bit encoder,
            fp16 2.33%), so ``bits=4, encoder_bits=8`` is the recommended
            4-bit recipe.

    Returns:
        Quantized model (in-place modification)
    """
    enc_bits = bits if encoder_bits is None else int(encoder_bits)

    def _is_encoder(path: str) -> bool:
        return path.startswith("audio_tower")

    def _quantizable(module: nn.Module) -> bool:
        return isinstance(module, (nn.Linear, nn.Embedding))

    if enc_bits == bits:
        nn.quantize(model, bits=bits, group_size=group_size)
        return model

    nn.quantize(
        model,
        bits=bits,
        group_size=group_size,
        class_predicate=lambda path, m: _quantizable(m) and not _is_encoder(path),
    )
    if enc_bits < 16:
        nn.quantize(
            model,
            bits=enc_bits,
            group_size=group_size,
            class_predicate=lambda path, m: _quantizable(m) and _is_encoder(path),
        )
    return model
