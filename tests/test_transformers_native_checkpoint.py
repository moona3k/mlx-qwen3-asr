"""Tests for Hugging Face transformers>=5.24 native Qwen3-ASR checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from mlx_qwen3_asr.convert import remap_weights
from mlx_qwen3_asr.tokenizer import _NativeQwenBPETokenizer


def test_remap_weights_transformers_native_layout() -> None:
    """Native transformers>=5.24 Qwen3ASRForConditionalGeneration keys map cleanly."""
    conv_pt = mx.zeros((480, 1, 3, 3))
    weights = {
        "model.audio_tower.conv2d1.weight": conv_pt,
        "model.audio_tower.conv2d1.bias": mx.zeros((480,)),
        "model.audio_tower.layers.0.fc1.weight": mx.zeros((4096, 1024)),
        "model.multi_modal_projector.linear_1.weight": mx.zeros((1024, 1024)),
        "model.multi_modal_projector.linear_1.bias": mx.zeros((1024,)),
        "model.multi_modal_projector.linear_2.weight": mx.zeros((2048, 1024)),
        "model.multi_modal_projector.linear_2.bias": mx.zeros((2048,)),
        "model.language_model.embed_tokens.weight": mx.zeros((16, 2048)),
        "model.language_model.layers.0.self_attn.q_proj.weight": mx.zeros((2048, 2048)),
        "model.language_model.norm.weight": mx.zeros((2048,)),
    }

    remapped = remap_weights(weights)

    assert remapped["audio_tower.conv2d1.weight"].shape == (480, 3, 3, 1)
    assert "audio_tower.conv2d1.bias" in remapped
    assert "audio_tower.layers.0.fc1.weight" in remapped
    assert "audio_tower.proj1.weight" in remapped
    assert "audio_tower.proj1.bias" in remapped
    assert "audio_tower.proj2.weight" in remapped
    assert "audio_tower.proj2.bias" in remapped
    assert "model.embed_tokens.weight" in remapped
    assert "model.layers.0.self_attn.q_proj.weight" in remapped
    assert "model.norm.weight" in remapped


def test_native_bpe_tokenizer_loads_tokenizer_json_fallback(tmp_path: Path) -> None:
    """Tokenizer loads cleanly when only tokenizer.json (no vocab.json/merges.txt) exists."""
    tok_json = {
        "model": {
            "type": "BPE",
            "vocab": {"h": 0, "i": 1, "hi": 2, "!": 3},
            "merges": [["h", "i"]],
        },
        "added_tokens": [
            {"id": 151643, "content": "<|endoftext|>", "special": True},
            {"id": 151645, "content": "<|im_end|>", "special": True},
        ],
    }
    (tmp_path / "tokenizer.json").write_text(json.dumps(tok_json), encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "<|endoftext|>"}),
        encoding="utf-8",
    )

    tok = _NativeQwenBPETokenizer(str(tmp_path))
    ids = tok.encode("hi!<|im_end|>")
    assert ids == [2, 3, 151645]
    assert tok.decode(ids, skip_special_tokens=True) == "hi!"
