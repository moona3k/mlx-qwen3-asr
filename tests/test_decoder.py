"""Tests for mlx_qwen3_asr/decoder.py."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

import mlx_qwen3_asr.decoder as decmod
from mlx_qwen3_asr.config import TextDecoderConfig


def _tiny_decoder() -> decmod.TextDecoder:
    cfg = TextDecoderConfig(
        vocab_size=32,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=128,
    )
    return decmod.TextDecoder(cfg)


def test_step_many_with_prefix_uses_prefix_causal_mask(monkeypatch):
    decoder = _tiny_decoder()
    cache = decmod.KVCache(num_layers=1)
    cache.offset = 5

    captured = {}

    def _fake_sdpa(q, k, v, mask=None, scale=None):  # noqa: ANN001
        captured["mask"] = mask
        return mx.zeros_like(q)

    monkeypatch.setattr(decmod, "_scaled_dot_product_attention", _fake_sdpa)

    # Multi-token decode with existing prefix cache.
    L = 3
    embeds = mx.zeros((1, L, 256), dtype=mx.float32)
    pos = mx.arange(L, dtype=mx.int32)[None, :]
    position_ids = mx.stack([pos, pos, pos], axis=1)
    _ = decoder(inputs_embeds=embeds, position_ids=position_ids, cache=cache)

    mask = captured.get("mask")
    assert mask is not None
    assert tuple(mask.shape) == (1, 1, 3, 8)  # (B,H,L,prefix+L)

    m = np.array(mask[0, 0], dtype=np.float32)
    # Prefix columns are fully visible.
    np.testing.assert_allclose(m[:, :5], 0.0, atol=0.0)
    # New-token block must be causal.
    assert m[0, 5] == 0.0
    assert m[0, 6] < -1e8
    assert m[0, 7] < -1e8
    assert m[1, 5] == 0.0
    assert m[1, 6] == 0.0
    assert m[1, 7] < -1e8
    assert m[2, 5] == 0.0
    assert m[2, 6] == 0.0
    assert m[2, 7] == 0.0


def test_single_token_with_prefix_uses_no_mask(monkeypatch):
    decoder = _tiny_decoder()
    cache = decmod.KVCache(num_layers=1)
    cache.offset = 7

    captured = {}

    def _fake_sdpa(q, k, v, mask=None, scale=None):  # noqa: ANN001
        captured["mask"] = mask
        return mx.zeros_like(q)

    monkeypatch.setattr(decmod, "_scaled_dot_product_attention", _fake_sdpa)

    embeds = mx.zeros((1, 1, 256), dtype=mx.float32)
    pos = mx.array([[0]], dtype=mx.int32)
    position_ids = mx.stack([pos, pos, pos], axis=1)
    _ = decoder(inputs_embeds=embeds, position_ids=position_ids, cache=cache)

    assert "mask" in captured
    assert captured["mask"] is None


def test_causal_mask_caches_are_bounded():
    """Distinct prefill lengths must not accumulate (L, L) masks forever."""
    decmod._CAUSAL_MASK_CACHE.clear()
    decmod._CAUSAL_MASK_WITH_PREFIX_CACHE.clear()
    limit = decmod._MASK_CACHE_ENTRIES
    for seq_len in range(1, limit + 10):
        decmod._create_causal_mask(seq_len, mx.float32)
        decmod._create_causal_mask_with_prefix(seq_len, prefix_len=4, dtype=mx.float32)
    assert len(decmod._CAUSAL_MASK_CACHE) == limit
    assert len(decmod._CAUSAL_MASK_WITH_PREFIX_CACHE) == limit
    # Most recent entries are the ones retained, and are reused on the next call.
    mask = decmod._create_causal_mask(limit + 9, mx.float32)
    assert decmod._create_causal_mask(limit + 9, mx.float32) is mask
