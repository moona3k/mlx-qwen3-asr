"""Shared attention helpers for Qwen3-ASR modules."""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx


def scalar_int(value: mx.array) -> int:
    """Return a 0-d integer array as a Python int (a host sync).

    ``mx.array.item()`` is typed ``int | float | complex`` in newer MLX stubs;
    this keeps call sites honest about the conversion.
    """
    return int(value.item())  # type: ignore[arg-type]



def _scaled_dot_product_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    mask: Optional[mx.array] = None,
    scale: Optional[float] = None,
) -> mx.array:
    """Compute scaled dot-product attention with an optional mask.

    Tries ``mx.fast.scaled_dot_product_attention`` first (fused kernel),
    falling back to a manual implementation when unavailable.

    Args:
        q: Query tensor, shape (B, H, L, D).
        k: Key tensor, shape (B, H, S, D).
        v: Value tensor, shape (B, H, S, D).
        mask: Optional attention mask broadcastable to (B, H, L, S).
        scale: Scaling factor; defaults to 1/sqrt(D).

    Returns:
        Attention output, shape (B, H, L, D).
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    if hasattr(mx.fast, "scaled_dot_product_attention"):
        try:
            return mx.fast.scaled_dot_product_attention(
                q, k, v, scale=scale, mask=mask
            )
        except (TypeError, ValueError):
            # Fall through to a compatibility path below.
            pass
        except RuntimeError as e:
            # Only fall back for known fused-kernel compatibility issues.
            # Unexpected runtime failures (e.g., OOM) should still surface.
            msg = str(e).lower()
            if "not implemented" in msg or "unsupported" in msg:
                pass
            else:
                raise

    # Manual fallback
    if q.shape[1] != k.shape[1]:
        if q.shape[1] % k.shape[1] != 0:
            raise ValueError(
                f"Incompatible attention heads: q={q.shape[1]}, k={k.shape[1]}"
            )
        groups = q.shape[1] // k.shape[1]
        k = mx.repeat(k, groups, axis=1)
        v = mx.repeat(v, groups, axis=1)

    scores = (q @ k.transpose(0, 1, 3, 2)) * scale
    if mask is not None:
        scores = scores + mask
    weights = mx.softmax(scores, axis=-1)
    return weights @ v
