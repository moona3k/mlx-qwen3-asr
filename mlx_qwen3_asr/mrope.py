"""Interleaved Multi-dimensional Rotary Position Embedding (MRoPE) for Qwen3-ASR.

Critical correctness module. The standard nn.RoPE does NOT work for Qwen3-ASR.
The model uses interleaved frequency assignment across 3 spatial dimensions
(temporal, height, width) with sections [24, 20, 20].

Frequency assignment uses STRIDE-3 INTERLEAVING (NOT chunking):
    - freq index 0 -> section 0 (temporal)
    - freq index 1 -> section 1 (height)
    - freq index 2 -> section 2 (width)
    - freq index 3 -> section 0 (temporal)
    - freq index 4 -> section 1 (height)
    - ... and so on

Total frequencies: (24 + 20 + 20) = 64 = head_dim // 2
Each frequency maps to 2 dimensions in the rotation, so 64 * 2 = 128 = head_dim.

Official behavior follows Qwen's `apply_interleaved_mrope` logic:
start from the temporal frequencies and overwrite selected indices for
height/width dimensions. Uncovered indices remain temporal-derived (not zeroed).

Reference: apply_interleaved_mrope() in the official Qwen3-ASR repo.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

MROPE_SECTION = [24, 20, 20]  # temporal, height, width


class InterleavedMRoPE:
    """Multi-dimensional RoPE with interleaved frequency assignment.

    Sections [24, 20, 20] assign frequencies to temporal/height/width dims
    using stride-3 interleaving (NOT chunking).

    Args:
        head_dim: Dimension per attention head (128 for Qwen3-ASR).
        base: RoPE base frequency (1000000.0 for Qwen3-ASR checkpoints).
        mrope_section: Frequency count per spatial dimension [24, 20, 20].
    """

    def __init__(
        self,
        head_dim: int,
        base: float = 1000000.0,
        mrope_section: list[int] | None = None,
    ):
        self.head_dim = head_dim
        self.base = base
        self.mrope_section = mrope_section or list(MROPE_SECTION)
        self.half_dim = head_dim // 2

        assert sum(self.mrope_section) == self.half_dim, (
            f"Sum of mrope_section {self.mrope_section} = {sum(self.mrope_section)} "
            f"must equal head_dim // 2 = {self.half_dim}"
        )

        # Precompute inverse frequencies: 1 / (base ^ (2i / head_dim))
        inv_freq = 1.0 / (
            base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
        )
        # Materialize now: this buffer is not a module parameter, and MLX >= 0.31
        # cannot evaluate a graph built on another thread (see encoder.py).
        mx.eval(inv_freq)
        self._inv_freq = inv_freq  # shape: (half_dim,)

        # Official overwrite masks:
        # freqs_t = freqs[0] (temporal), then overwrite selected indices with
        # freqs[1] / freqs[2] values.
        self._overwrite_masks: list[mx.array] = []
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            stop = min(length, self.half_dim)
            indices: np.ndarray = np.arange(offset, stop, 3, dtype=np.int32)
            mask: np.ndarray = np.zeros(self.half_dim, dtype=bool)
            mask[indices] = True
            # Shape (1, 1, half_dim) for broadcast over (B, L, half_dim)
            self._overwrite_masks.append(mx.array(mask[None, None, :]))

    def __call__(
        self,
        position_ids: mx.array,
        dtype: mx.Dtype = mx.float32,
    ) -> tuple[mx.array, mx.array]:
        """Compute interleaved MRoPE cos/sin embeddings.

        Args:
            position_ids: Shape (batch, 3, seq_len) -- one row per spatial
                dimension (temporal, height, width).
            dtype: Output dtype.

        Returns:
            Tuple of (cos, sin), each of shape (batch, seq_len, head_dim).
        """
        batch, n_dims, seq_len = position_ids.shape
        assert n_dims == 3, f"Expected 3 spatial dims, got {n_dims}"

        # Compute per-dimension frequencies over the full half_dim.
        # position_ids: (B, 3, L) -> (3, B, L, 1)
        pos = position_ids.astype(mx.float32).transpose(1, 0, 2)[..., None]
        freqs = pos * self._inv_freq[None, None, None, :]  # (3, B, L, half_dim)

        # Official interleaving:
        # freqs_t = freqs[0]; overwrite selected indices from H/W dimensions.
        freqs_t = freqs[0]
        for dim, mask in enumerate(self._overwrite_masks, start=1):
            freqs_t = mx.where(mask, freqs[dim], freqs_t)

        emb = mx.concatenate([freqs_t, freqs_t], axis=-1)  # (B, L, head_dim)
        cos_full = mx.cos(emb).astype(dtype)
        sin_full = mx.sin(emb).astype(dtype)

        return cos_full, sin_full


def apply_rotary_pos_emb(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> tuple[mx.array, mx.array]:
    """Apply rotary position embeddings to query and key tensors.

    Uses the standard rotation formula:
        x_rot = x * cos + rotate_half(x) * sin

    where rotate_half splits the last dimension in half and applies:
        [-x2, x1] concatenation.

    Args:
        q: Query tensor, shape (batch, n_heads, seq_len, head_dim).
        k: Key tensor, shape (batch, n_heads, seq_len, head_dim).
        cos: Cosine embeddings from InterleavedMRoPE,
            shape (batch, seq_len, head_dim).
        sin: Sine embeddings from InterleavedMRoPE,
            shape (batch, seq_len, head_dim).

    Returns:
        Tuple of (q_embed, k_embed) with rotary embeddings applied,
        same shapes as inputs.
    """
    # Expand dims for multi-head broadcasting:
    # (batch, seq_len, head_dim) -> (batch, 1, seq_len, head_dim)
    if cos.ndim == 3:
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]

    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed, k_embed


def _rotate_half(x: mx.array) -> mx.array:
    """Rotate half of the hidden dims of the input.

    Splits the last dimension in half and returns [-x2, x1].

    Args:
        x: Input tensor, shape (..., head_dim).

    Returns:
        Rotated tensor, same shape as input.
    """
    mid = x.shape[-1] // 2
    x1 = x[..., :mid]
    x2 = x[..., mid:]
    return mx.concatenate([-x2, x1], axis=-1)
