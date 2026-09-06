"""Text decoder and KV-cache components for Qwen3-ASR in MLX."""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .attention import _scaled_dot_product_attention
from .cache_utils import LRUCache
from .config import TextDecoderConfig
from .mrope import InterleavedMRoPE, apply_rotary_pos_emb


class TextAttention(nn.Module):
    """Causal self-attention with MRoPE and Q/K norms for the text decoder.

    Uses RMSNorm on queries and keys (Qwen3 innovation). Supports grouped
    query attention (GQA) when num_kv_heads < num_heads.

    Args:
        config: TextDecoderConfig with model hyperparameters.
    """

    def __init__(self, config: TextDecoderConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        # Per-head RMSNorm on queries and keys (Qwen3 innovation)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        layer_idx: int = 0,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Input hidden states, shape (B, L, D).
            cos: MRoPE cosine embeddings, shape (B, L, head_dim).
            sin: MRoPE sine embeddings, shape (B, L, head_dim).
            mask: Optional causal attention mask, broadcastable to (B, H, L, S).
            cache: Optional KV cache for autoregressive generation.
            layer_idx: Index of this layer in the decoder stack (for cache).

        Returns:
            Output tensor, shape (B, L, D).
        """
        B, L, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape: (B, L, num_heads * head_dim) -> (B, L, num_heads, head_dim)
        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_kv_heads, self.head_dim)
        v = v.reshape(B, L, self.num_kv_heads, self.head_dim)

        # Apply Q/K norms (operates on last dim)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transpose to (B, H, L, D) for attention
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Apply MRoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Update KV cache if provided
        if cache is not None:
            k, v = cache.update(k, v, layer_idx)

        # Scaled dot-product attention
        out = _scaled_dot_product_attention(q, k, v, mask=mask)

        # (B, H, L, Dh) -> (B, L, H, Dh) -> (B, L, D)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (Qwen3-style).

    Computes: down_proj(silu(gate_proj(x)) * up_proj(x))

    Args:
        hidden_size: Input/output dimension.
        intermediate_size: Intermediate (expanded) dimension.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor, shape (..., hidden_size).

        Returns:
            Output tensor, shape (..., hidden_size).
        """
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TextDecoderLayer(nn.Module):
    """Pre-norm transformer decoder layer with RMSNorm.

    Uses RMSNorm (NOT LayerNorm) as is standard for Qwen3/LLaMA-family models.

    Args:
        config: TextDecoderConfig with model hyperparameters.
    """

    def __init__(self, config: TextDecoderConfig):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = TextAttention(config)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        layer_idx: int = 0,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Input hidden states, shape (B, L, D).
            cos: MRoPE cosine embeddings.
            sin: MRoPE sine embeddings.
            mask: Optional causal attention mask.
            cache: Optional KV cache.
            layer_idx: Index of this layer (for cache).

        Returns:
            Output tensor, shape (B, L, D).
        """
        # Self-attention block
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cos, sin, mask=mask, cache=cache, layer_idx=layer_idx)
        x = residual + x

        # Feed-forward block
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class TextDecoder(nn.Module):
    """Full text decoder: embedding -> transformer with MRoPE -> norm.

    Note: The LM head is NOT included here; it lives in the top-level
    Qwen3ASRModel to match the HuggingFace weight layout.

    Args:
        config: TextDecoderConfig with model hyperparameters.
    """

    def __init__(self, config: TextDecoderConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            TextDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # MRoPE with interleaved frequency assignment
        mrope_section = [24, 20, 20]
        if isinstance(config.rope_scaling, dict):
            mrope_section = config.rope_scaling.get("mrope_section", mrope_section)

        self.rotary_emb = InterleavedMRoPE(
            head_dim=config.head_dim,
            base=config.rope_theta,
            mrope_section=list(mrope_section),
        )

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        """Forward pass through the text decoder.

        Exactly one of ``input_ids`` or ``inputs_embeds`` must be provided.

        Args:
            input_ids: Token IDs, shape (B, L).
            inputs_embeds: Pre-computed embeddings, shape (B, L, D).
            position_ids: MRoPE position IDs, shape (B, 3, L).
            attention_mask: Causal mask, broadcastable to (B, H, L, S).
            cache: Optional KV cache for autoregressive generation.

        Returns:
            Hidden states, shape (B, L, D).
        """
        if inputs_embeds is not None:
            h = inputs_embeds
        elif input_ids is not None:
            h = self.embed_tokens(input_ids)
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        if position_ids is None:
            raise ValueError("position_ids must be provided for text decoding")

        # Compute MRoPE cos/sin
        cos, sin = self.rotary_emb(position_ids, dtype=h.dtype)

        # Prepare causal mask if not provided
        if attention_mask is None:
            L = h.shape[1]
            if cache is not None and cache.offset > 0:
                # Single-token decode with prefix cache does not need an explicit mask.
                # Multi-token decode (e.g., speculative verification) must preserve
                # causality within the newly appended token block.
                if L == 1:
                    attention_mask = None
                else:
                    attention_mask = _create_causal_mask_with_prefix(
                        seq_len=L,
                        prefix_len=cache.offset,
                        dtype=h.dtype,
                    )
            else:
                # Full causal mask
                attention_mask = _create_causal_mask(L, h.dtype)

        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            h = layer(h, cos, sin, mask=attention_mask, cache=cache, layer_idx=i)

        # Final norm
        h = self.norm(h)
        return h


class KVCache:
    """Per-layer key-value cache for autoregressive generation.

    Stores cached key and value tensors for each layer, concatenating new
    entries along the sequence dimension on each step.

    Args:
        num_layers: Number of decoder layers.
    """

    def __init__(self, num_layers: int, max_seq_len: Optional[int] = None):
        self.keys: list[Optional[mx.array]] = [None] * num_layers
        self.values: list[Optional[mx.array]] = [None] * num_layers
        self.offset: int = 0
        self.max_seq_len = max_seq_len

    def update(
        self,
        key: mx.array,
        value: mx.array,
        layer_idx: int,
    ) -> tuple[mx.array, mx.array]:
        """Append new key/value entries to the cache.

        Args:
            key: New key tensor, shape (B, H, L_new, D).
            value: New value tensor, shape (B, H, L_new, D).
            layer_idx: Which layer this cache belongs to.

        Returns:
            Tuple of (full_keys, full_values) including cached history.
        """
        if self.max_seq_len is None:
            if self.keys[layer_idx] is not None:
                prev_key = self.keys[layer_idx]
                prev_value = self.values[layer_idx]
                assert prev_key is not None
                assert prev_value is not None
                self.keys[layer_idx] = mx.concatenate(
                    [prev_key, key], axis=2
                )
                self.values[layer_idx] = mx.concatenate(
                    [prev_value, value], axis=2
                )
            else:
                self.keys[layer_idx] = key
                self.values[layer_idx] = value
            full_k = self.keys[layer_idx]
            full_v = self.values[layer_idx]
            assert full_k is not None
            assert full_v is not None
        else:
            # Preallocated cache mode: write new tokens into a fixed buffer.
            B, H, L_new, D = key.shape
            start = self.offset
            end = start + L_new
            if end > self.max_seq_len:
                raise ValueError(
                    f"KV cache overflow: end={end}, max_seq_len={self.max_seq_len}"
                )

            if self.keys[layer_idx] is None:
                self.keys[layer_idx] = mx.zeros(
                    (B, H, self.max_seq_len, D), dtype=key.dtype
                )
                self.values[layer_idx] = mx.zeros(
                    (B, H, self.max_seq_len, D), dtype=value.dtype
                )

            key_cache = self.keys[layer_idx]
            value_cache = self.values[layer_idx]
            assert key_cache is not None
            assert value_cache is not None

            # Follow mlx-lm's cache pattern: in-place writes into preallocated
            # buffers avoid constructing a fresh array on each token step.
            key_cache[..., start:end, :] = key
            value_cache[..., start:end, :] = value

            full_k = key_cache[:, :, :end, :]
            full_v = value_cache[:, :, :end, :]

        # Update offset after the last layer processes its step
        if layer_idx == len(self.keys) - 1:
            self.offset += key.shape[2]

        return full_k, full_v

    @property
    def seq_len(self) -> int:
        """Return the current cached sequence length."""
        return self.offset

    def trim(self, num_tokens: int) -> None:
        """Trim recently appended tokens from all layer caches.

        Used by speculative decoding when some drafted tokens are rejected and
        should not remain in cache state.

        Args:
            num_tokens: Number of latest cached tokens to remove.
        """
        if num_tokens < 0:
            raise ValueError(f"num_tokens must be >= 0, got {num_tokens}")
        if num_tokens == 0:
            return
        if num_tokens > self.offset:
            raise ValueError(
                f"Cannot trim {num_tokens} tokens from cache with length {self.offset}"
            )

        new_len = self.offset - num_tokens

        if self.max_seq_len is None:
            for i in range(len(self.keys)):
                key_cache = self.keys[i]
                value_cache = self.values[i]
                if key_cache is not None:
                    self.keys[i] = key_cache[:, :, :new_len, :]
                if value_cache is not None:
                    self.values[i] = value_cache[:, :, :new_len, :]

        self.offset = new_len


def _create_causal_mask(seq_len: int, dtype: mx.Dtype = mx.float32) -> mx.array:
    """Create an additive causal mask for self-attention.

    Masked positions are filled with -inf (or a very large negative number);
    unmasked positions are 0.

    Args:
        seq_len: Sequence length.
        dtype: Output data type.

    Returns:
        Causal mask, shape (1, 1, seq_len, seq_len).
    """
    cache_key = (seq_len, str(dtype))
    cached = _CAUSAL_MASK_CACHE.get(cache_key)
    if cached is not None:
        return cached

    mask = mx.full((seq_len, seq_len), -1e9, dtype=dtype)
    mask = mx.triu(mask, k=1)  # zero on and below diagonal, -1e9 above
    mask = mask[None, None, :, :]  # (1, 1, L, L)
    _CAUSAL_MASK_CACHE.put(cache_key, mask)
    return mask


# Every distinct prefill length produces an (L, L) mask; bound the caches so a
# long-running server transcribing varied audio does not accumulate them.
_MASK_CACHE_ENTRIES = 16
_CAUSAL_MASK_CACHE: LRUCache[tuple[int, str], mx.array] = LRUCache(_MASK_CACHE_ENTRIES)


def _create_causal_mask_with_prefix(
    seq_len: int,
    prefix_len: int,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Create causal mask for appending ``seq_len`` tokens after cached prefix.

    The prefix region (already in cache) is fully visible to all new queries.
    Causality is enforced only among the newly appended tokens.
    """
    cache_key = (seq_len, prefix_len, str(dtype))
    cached = _CAUSAL_MASK_WITH_PREFIX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Left block: cached prefix keys, always visible.
    left = mx.zeros((seq_len, prefix_len), dtype=dtype)

    # Right block: causal mask among new tokens.
    right = mx.full((seq_len, seq_len), -1e9, dtype=dtype)
    right = mx.triu(right, k=1)

    mask = mx.concatenate([left, right], axis=1)[None, None, :, :]
    _CAUSAL_MASK_WITH_PREFIX_CACHE.put(cache_key, mask)
    return mask


_CAUSAL_MASK_WITH_PREFIX_CACHE: LRUCache[tuple[int, int, str], mx.array] = LRUCache(
    _MASK_CACHE_ENTRIES
)
