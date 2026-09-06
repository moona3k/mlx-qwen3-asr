"""Audio encoder components for Qwen3-ASR in MLX."""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .attention import _scaled_dot_product_attention, scalar_int
from .config import AudioEncoderConfig

# Use segmented per-window execution only once we have enough windows that
# dense block-mask attention becomes a clear overhead.
_WINDOWED_SEGMENT_MIN_WINDOWS = 20


class SinusoidalPositionEmbedding(nn.Module):
    """Fixed (non-learned) sinusoidal position embeddings.

    Computed at init time and stored as a frozen buffer -- these are NOT
    part of the saved weights.

    Uses the official Qwen3-ASR formula matching ``SinusoidsPositionEmbedding``
    from the HuggingFace reference::

        log_timescale_increment = log(10000) / (channels // 2 - 1)
        inv_timescales = exp(-i * log_timescale_increment)

    This differs from the standard Transformer PE formula by using
    ``half_dim - 1`` in the denominator (not ``half_dim``).

    Args:
        num_positions: Maximum number of positions to support.
        embedding_dim: Dimensionality of each position embedding vector.
    """

    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__()
        self.num_positions = num_positions
        self.embedding_dim = embedding_dim

        half_dim = embedding_dim // 2
        # Match official: log_timescale_increment = log(10000) / (half_dim - 1)
        log_timescale_increment = math.log(10000.0) / (half_dim - 1)
        inv_timescales = mx.exp(
            -log_timescale_increment * mx.arange(half_dim, dtype=mx.float32)
        )
        positions = mx.arange(0, num_positions, dtype=mx.float32)
        # Outer product: (num_positions, half_dim)
        scaled_time = positions[:, None] * inv_timescales[None, :]
        # Concatenate sin and cos: (num_positions, embedding_dim)
        pe = mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=-1)
        # If embedding_dim is odd, trim the last column
        if embedding_dim % 2 == 1:
            pe = pe[:, :embedding_dim]
        # Materialize now. ``_pe`` is not a parameter, so ``mx.eval(model.parameters())``
        # never reaches it, and an unevaluated graph is bound to the stream of the
        # thread that built it: MLX >= 0.31 refuses to evaluate it from another
        # thread ("There is no Stream(gpu, 1) in current thread").
        mx.eval(pe)
        self._pe = pe
        self.freeze()

    def __call__(self, positions: int | mx.array) -> mx.array:
        """Return sinusoidal embeddings by length or explicit indices.

        Args:
            positions:
                - int: returns embeddings for 0..positions-1.
                - mx.array: returns embeddings indexed by that tensor.

        Returns:
            Embeddings of shape (positions, embedding_dim).
        """
        if isinstance(positions, int):
            return self._pe[:positions]
        return self._pe[positions]


class AudioAttention(nn.Module):
    """Bidirectional multi-head attention for the audio encoder.

    Uses bias on all projections. No causal mask, no RoPE.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor, shape (B, L, D).
            mask: Optional attention mask, broadcastable to (B, H, L, S).
                Used to mask padded positions in the encoder.

        Returns:
            Output tensor, shape (B, L, D).
        """
        B, L, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (B, L, H, Dh) then transpose to (B, H, L, Dh)
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Bidirectional attention with optional padding mask
        out = _scaled_dot_product_attention(q, k, v, mask=mask)

        # (B, H, L, Dh) -> (B, L, H, Dh) -> (B, L, D)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(out)


class AudioEncoderLayer(nn.Module):
    """Pre-norm transformer layer for the audio encoder.

    Uses LayerNorm (NOT RMSNorm) and GELU activation, following the
    Whisper / Qwen3-ASR audio encoder convention.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        encoder_ffn_dim: Feed-forward intermediate dimension.
    """

    def __init__(self, d_model: int, num_heads: int, encoder_ffn_dim: int):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = AudioAttention(d_model, num_heads)
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, encoder_ffn_dim, bias=True)
        self.fc2 = nn.Linear(encoder_ffn_dim, d_model, bias=True)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor, shape (B, L, D).
            mask: Optional attention mask for padding, broadcastable to (B, H, L, S).

        Returns:
            Output tensor, shape (B, L, D).
        """
        # Self-attention block (pre-norm)
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, mask=mask)
        x = residual + x

        # Feed-forward block (pre-norm)
        residual = x
        x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.fc2(x)
        x = residual + x

        # Match upstream torch audio encoder behavior: clamp fp16 activations
        # after each layer to avoid inf/nan propagation.
        if x.dtype == mx.float16:
            clamp_value = float(np.finfo(np.float16).max - 1000.0)
            x = mx.clip(x, -clamp_value, clamp_value)

        return x


class AudioEncoder(nn.Module):
    """Full audio encoder: Conv2d stem -> sinusoidal pos -> transformer -> projection.

    The Conv2d stem applies three stride-2 convolutions for 8x temporal
    downsampling. The output is projected to the text decoder's hidden size
    via a 2-layer MLP.

    Args:
        config: AudioEncoderConfig with model hyperparameters.
    """

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.config = config
        dhs = config.downsample_hidden_size  # 480 for 1.7B

        # Conv2d stem: 3 layers, each stride=2, total 8x temporal downsample
        self.conv2d1 = nn.Conv2d(1, dhs, kernel_size=3, stride=2, padding=1)
        self.conv2d2 = nn.Conv2d(dhs, dhs, kernel_size=3, stride=2, padding=1)
        self.conv2d3 = nn.Conv2d(dhs, dhs, kernel_size=3, stride=2, padding=1)

        # After 3x stride-2 on 128 mel bins: 128 -> 64 -> 32 -> 16
        freq_after_conv = config.num_mel_bins // 8
        self.conv_out = nn.Linear(dhs * freq_after_conv, config.d_model, bias=False)

        # Sinusoidal position embeddings (fixed, not learned)
        self.embed_positions = SinusoidalPositionEmbedding(
            config.max_source_positions, config.d_model
        )

        # Transformer encoder layers
        self.layers = [
            AudioEncoderLayer(
                config.d_model,
                config.encoder_attention_heads,
                config.encoder_ffn_dim,
            )
            for _ in range(config.encoder_layers)
        ]

        # Post-transformer layer norm
        self.ln_post = nn.LayerNorm(config.d_model)

        # Output projection MLP: d_model -> d_model -> output_dim
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.proj2 = nn.Linear(config.d_model, config.output_dim)

    def _apply_conv_stem(self, x: mx.array) -> mx.array:
        """Apply the 3-layer Conv2d stem with GELU activations.

        Args:
            x: Input tensor in NHWC format, shape ``(B, H=mel, W=time, C)``.

        Returns:
            Output tensor in NHWC, shape ``(B, mel_down, time_down, dhs)``.
        """
        x = nn.gelu(self.conv2d1(x))
        x = nn.gelu(self.conv2d2(x))
        x = nn.gelu(self.conv2d3(x))
        return x

    def get_output_lengths(self, input_lengths: mx.array) -> mx.array:
        """Compute output token counts matching the official encoder formula.

        The encoder splits mel frames into chunks of ``n_window * 2``
        (typically 100) frames each. Full chunks produce 13 tokens; the
        tail chunk (if any) produces fewer. The formula matches
        ``_get_feat_extract_output_lengths`` from the official Qwen3-ASR
        codebase.

        Args:
            input_lengths: Mel frame counts, shape ``(B,)``.

        Returns:
            Output token counts, shape ``(B,)``.
        """
        chunk_size = self.config.n_window * 2  # e.g. 100
        # Tail chunk: frames not filling a full chunk
        tail_frames = input_lengths % chunk_size
        # Apply 3x stride-2 conv downsampling to tail
        tail_after_conv1 = (tail_frames - 1) // 2 + 1
        tail_after_conv2 = (tail_after_conv1 - 1) // 2 + 1
        tail_tokens = (tail_after_conv2 - 1) // 2 + 1
        # When tail_frames == 0 there is no tail chunk -> 0 tokens.
        # (MLX integer // truncates towards zero for negatives, unlike
        # Python/PyTorch which floor towards -inf, so (0-1)//2 gives 0
        # not -1, producing an off-by-one without this guard.)
        tail_tokens = mx.where(
            tail_frames > 0, tail_tokens, mx.zeros_like(tail_tokens)
        )
        # Full chunks produce the conv-downsampled length of chunk_size.
        full_after_conv1 = (chunk_size - 1) // 2 + 1
        full_after_conv2 = (full_after_conv1 - 1) // 2 + 1
        full_chunk_tokens = (full_after_conv2 - 1) // 2 + 1
        n_full_chunks = input_lengths // chunk_size
        return tail_tokens + n_full_chunks * full_chunk_tokens

    def __call__(
        self,
        input_features: mx.array,
        feature_lens: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Encode audio mel spectrograms into feature vectors.

        Implements the official Qwen3-ASR encoder pipeline:

        1. Split mel into chunks of ``n_window * 2`` frames
        2. Per-chunk Conv2d stem with channel-major reshape
        3. Per-chunk sinusoidal position embeddings
        4. Windowed block-diagonal attention across transformer layers
        5. Post-processing: LayerNorm, GELU projection

        Args:
            input_features: Mel spectrogram, shape ``(B, n_mels, n_frames)``.
                May be padded (e.g. to 3000 frames / 30 seconds).
            feature_lens: Actual frame counts per sample (before padding),
                shape ``(B,)``.

        Returns:
            Tuple of ``(audio_features, output_lens)``:
                audio_features: shape ``(B, max_tokens, output_dim)``
                output_lens: shape ``(B,)``, valid token count per sample
        """
        B = input_features.shape[0]
        chunk_size = self.config.n_window * 2
        n_window_infer = self.config.n_window_infer

        all_features = []
        all_output_lens = []

        for b in range(B):
            feat_len = scalar_int(feature_lens[b])
            mel = input_features[b, :, :feat_len]  # (128, actual_frames)
            features = self._encode_single(mel, chunk_size, n_window_infer)
            all_features.append(features)
            all_output_lens.append(features.shape[0])

        # Pad to max length and stack into batch
        max_len = max(all_output_lens)
        padded = []
        for f in all_features:
            if f.shape[0] < max_len:
                pad_arr = mx.zeros(
                    (max_len - f.shape[0], f.shape[-1]), dtype=f.dtype
                )
                f = mx.concatenate([f, pad_arr], axis=0)
            padded.append(f)

        output = mx.stack(padded)  # (B, max_len, output_dim)
        output_lens = mx.array(all_output_lens)
        return output, output_lens

    def _encode_single(
        self,
        mel: mx.array,
        chunk_size: int,
        n_window_infer: int,
    ) -> mx.array:
        """Encode a single (unbatched) mel spectrogram.

        Implements the per-sample encoder forward pass matching the official
        ``Qwen3ASRAudioEncoder.forward()`` logic:

        - Split mel into chunks of ``chunk_size`` frames
        - Process each chunk through the Conv2d stem independently
        - Use channel-major reshape: ``(1, F', T', C) -> (1, T', C*F')``
        - Apply per-chunk sinusoidal position embeddings (each chunk starts
          from position 0)
        - Run transformer with windowed (block-diagonal) attention

        Args:
            mel: Mel spectrogram, shape ``(n_mels, n_frames)``, already
                trimmed to actual length (no padding).
            chunk_size: Number of mel frames per chunk (``n_window * 2``).
            n_window_infer: Number of mel frames per attention window.

        Returns:
            Encoded features, shape ``(n_tokens, output_dim)``.
        """
        total_frames = mel.shape[1]

        # --- Per-chunk Conv2d processing ---
        # Batch full-size chunks together so long audio avoids many Python-level
        # Conv2d calls while keeping tail-chunk semantics identical.
        chunk_token_lens: list[int] = []
        chunk_conv_outputs: list[mx.array] = []
        n_full_chunks = total_frames // chunk_size

        if n_full_chunks > 0:
            full_frames = n_full_chunks * chunk_size
            full_mel = mel[:, :full_frames]  # (n_mels, n_full * chunk_size)
            full_mel = full_mel.reshape(
                int(mel.shape[0]),
                n_full_chunks,
                chunk_size,
            ).transpose(1, 0, 2)  # (n_full, n_mels, chunk_size)

            # MLX NHWC batch: (n_full, H=n_mels, W=chunk_size, C=1)
            x_full = self._apply_conv_stem(full_mel[:, :, :, None])

            # Channel-major reshape to match conv_out weight layout.
            # PyTorch does: (B,C,F,T) -> permute(0,3,1,2) -> (B,T,C,F) -> view(B,T,C*F)
            # MLX NHWC:    (B,F',T',C) -> transpose(0,2,3,1) -> (B,T',C,F')
            _, F_d, T_d, C_d = x_full.shape
            x_full = x_full.transpose(0, 2, 3, 1)  # (n_full, T', C=dhs, F'=freq_down)
            x_full = x_full.reshape(n_full_chunks, T_d, C_d * F_d)
            chunk_conv_outputs.append(x_full.reshape(n_full_chunks * T_d, C_d * F_d))
            chunk_token_lens.extend([int(T_d)] * n_full_chunks)

        # Tail chunk uses exact per-chunk semantics (no extra right-padding).
        if n_full_chunks * chunk_size < total_frames:
            chunk_mel = mel[:, n_full_chunks * chunk_size :]  # (n_mels, chunk_len)
            x_tail = self._apply_conv_stem(chunk_mel[None, :, :, None])
            _, F_d, T_d, C_d = x_tail.shape
            x_tail = x_tail.transpose(0, 2, 3, 1).reshape(1, T_d, C_d * F_d)
            chunk_token_lens.append(int(T_d))
            chunk_conv_outputs.append(x_tail[0])

        # Concatenate all chunks and project to d_model
        x = mx.concatenate(chunk_conv_outputs, axis=0)  # (total_tokens, dhs*freq_down)
        x = self.conv_out(x)  # (total_tokens, d_model)

        # --- Per-chunk sinusoidal position embeddings ---
        # Each chunk gets PE starting from position 0 (matching official)
        max_chunk_tokens = max(chunk_token_lens)
        pe = self.embed_positions(max_chunk_tokens)  # (max_chunk_tokens, d_model)

        pe_parts: list[mx.array] = []
        for ct in chunk_token_lens:
            pe_parts.append(pe[:ct])
        pe_full = mx.concatenate(pe_parts, axis=0)  # (total_tokens, d_model)
        # The PE table is float32 and outside parameters(), so load_model's dtype
        # cast never touches it. Adding it unconverted would promote every later
        # encoder (and decoder) activation to float32, tripling latency.
        x = x + pe_full.astype(x.dtype)

        # --- Windowed attention ---
        total_tokens = x.shape[0]
        tokens_per_full_chunk = chunk_token_lens[0]
        tokens_per_window = tokens_per_full_chunk * (n_window_infer // chunk_size)

        # Build cu_seqlens for attention windows
        cu_seqlens: list[int] = [0]
        pos = 0
        while pos < total_tokens:
            window_end = min(pos + tokens_per_window, total_tokens)
            cu_seqlens.append(window_end)
            pos = window_end

        num_windows = len(cu_seqlens) - 1
        # Add batch dim for transformer layers: (1, total_tokens, d_model)
        x = x[None, :, :]
        if num_windows >= _WINDOWED_SEGMENT_MIN_WINDOWS:
            x = _apply_windowed_encoder_layers(x, self.layers, cu_seqlens)
        else:
            mask = _create_windowed_mask(total_tokens, cu_seqlens, x.dtype)
            for layer in self.layers:
                x = layer(x, mask=mask)

        x = x[0]  # remove batch dim: (total_tokens, d_model)

        # --- Post-processing ---
        x = self.ln_post(x)
        x = nn.gelu(self.proj1(x))
        x = self.proj2(x)

        return x  # (total_tokens, output_dim)


def _create_windowed_mask(
    seq_len: int,
    cu_seqlens: list[int],
    dtype: mx.Dtype = mx.float32,
) -> Optional[mx.array]:
    """Create a block-diagonal attention mask for windowed encoder attention.

    Tokens within the same window can attend to each other; tokens in
    different windows cannot. This matches the ``cu_seqlens``-based
    windowed attention in the official Qwen3-ASR encoder.

    Args:
        seq_len: Total sequence length.
        cu_seqlens: Cumulative sequence lengths defining window boundaries.
            E.g. ``[0, 104, 208, 260]`` means three windows: [0..103],
            [104..207], [208..259].
        dtype: Output dtype.

    Returns:
        Additive attention mask of shape ``(1, 1, seq_len, seq_len)`` where
        cross-window positions use ``finfo(dtype).min``, or ``None`` if there is only
        one window (no masking needed).
    """
    # Single window -- no mask needed
    if len(cu_seqlens) <= 2:
        return None

    # Assign each position to a window via boundary counting
    boundaries = mx.array(cu_seqlens[1:-1])  # internal boundaries
    positions = mx.arange(seq_len)

    # window_id[i] = number of internal boundaries <= position i
    window_ids = mx.sum(
        positions[:, None] >= boundaries[None, :], axis=1
    )  # (seq_len,)

    # Tokens attend iff they share a window
    same_window = window_ids[:, None] == window_ids[None, :]  # (L, L)
    masked_value = float(mx.finfo(dtype).min)
    mask = mx.where(
        same_window,
        mx.array(0.0, dtype=dtype),
        mx.array(masked_value, dtype=dtype),
    )
    return mask[None, None, :, :]  # (1, 1, L, L)


def _apply_windowed_encoder_layers(
    x: mx.array,
    layers: list[AudioEncoderLayer],
    cu_seqlens: list[int],
) -> mx.array:
    """Apply encoder layers independently per attention window.

    This is mathematically equivalent to a full-sequence forward pass with a
    block-diagonal additive mask derived from ``cu_seqlens``. It avoids
    materializing dense ``(L, L)`` masks and prevents cross-window attention
    compute for long sequences.

    Args:
        x: Encoder hidden states, shape ``(1, L, D)``.
        layers: Audio encoder layers to apply.
        cu_seqlens: Cumulative window boundaries (inclusive start, exclusive end).

    Returns:
        Updated hidden states, shape ``(1, L, D)``.
    """
    # Single window: keep fast-path identical to prior behavior.
    if len(cu_seqlens) <= 2:
        for layer in layers:
            x = layer(x, mask=None)
        return x

    for layer in layers:
        parts: list[mx.array] = []
        for s, e in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True):
            parts.append(layer(x[:, s:e, :], mask=None))
        x = mx.concatenate(parts, axis=1)
    return x
