"""Tests for mlx_qwen3_asr/model.py.

All tests use tiny configs (small d_model, 2 layers, 2 heads) to keep tests
fast and avoid needing real model weights.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import mlx_qwen3_asr.encoder as encoder_module
from mlx_qwen3_asr.config import AudioEncoderConfig, Qwen3ASRConfig, TextDecoderConfig
from mlx_qwen3_asr.model import (
    AudioAttention,
    AudioEncoder,
    AudioEncoderLayer,
    KVCache,
    Qwen3ASRModel,
    SinusoidalPositionEmbedding,
    SwiGLU,
    TextAttention,
    TextDecoderLayer,
    _apply_windowed_encoder_layers,
    _create_causal_mask,
    _create_windowed_mask,
    _scaled_dot_product_attention,
)

# ---------------------------------------------------------------------------
# Tiny configs for fast tests
# ---------------------------------------------------------------------------

def _tiny_audio_config() -> AudioEncoderConfig:
    return AudioEncoderConfig(
        num_mel_bins=128,
        encoder_layers=2,
        encoder_attention_heads=2,
        encoder_ffn_dim=128,
        d_model=64,
        output_dim=96,
        max_source_positions=100,
        downsample_hidden_size=32,
    )


def _tiny_text_config() -> TextDecoderConfig:
    return TextDecoderConfig(
        vocab_size=256,
        hidden_size=96,
        intermediate_size=192,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=48,  # hidden_size / num_heads is not forced, head_dim is separate
        rms_norm_eps=1e-6,
    )


def _tiny_asr_config() -> Qwen3ASRConfig:
    return Qwen3ASRConfig(
        audio_config=_tiny_audio_config(),
        text_config=_tiny_text_config(),
    )


# ---------------------------------------------------------------------------
# SinusoidalPositionEmbedding
# ---------------------------------------------------------------------------


class TestSinusoidalPositionEmbedding:
    """Test SinusoidalPositionEmbedding output shape."""

    def test_output_shape(self):
        emb = SinusoidalPositionEmbedding(num_positions=100, embedding_dim=64)
        positions = mx.arange(10)
        result = emb(positions)
        assert result.shape == (10, 64)

    def test_single_position(self):
        emb = SinusoidalPositionEmbedding(num_positions=50, embedding_dim=32)
        result = emb(mx.array([0]))
        assert result.shape == (1, 32)

    def test_odd_embedding_dim(self):
        """For odd embedding_dim=33, half_dim=16 so concat gives 32 columns.
        The trim pe[:, :33] on 32 columns just returns 32. This is fine since
        odd dims are not used in practice."""
        emb = SinusoidalPositionEmbedding(num_positions=10, embedding_dim=33)
        result = emb(mx.arange(5))
        # half_dim = 33 // 2 = 16, concat(sin, cos) = 32 cols.
        # pe[:, :33] on 32 cols => 32.
        assert result.shape == (5, 32)


# ---------------------------------------------------------------------------
# AudioAttention
# ---------------------------------------------------------------------------


class TestAudioAttention:
    """Test AudioAttention output shape."""

    def test_output_shape(self):
        d_model, num_heads = 64, 2
        attn = AudioAttention(d_model, num_heads)
        x = mx.random.normal((1, 10, d_model))
        result = attn(x)
        mx.eval(result)
        assert result.shape == (1, 10, d_model)

    def test_batch_output_shape(self):
        d_model, num_heads = 64, 4
        attn = AudioAttention(d_model, num_heads)
        x = mx.random.normal((2, 8, d_model))
        result = attn(x)
        mx.eval(result)
        assert result.shape == (2, 8, d_model)


class TestScaledDotProductAttention:
    """Test fused-attention fallback behavior."""

    def test_runtime_unsupported_falls_back_to_manual(self, monkeypatch):
        def _unsupported(*args, **kwargs):
            raise RuntimeError("unsupported fused kernel for this mask layout")

        monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", _unsupported)

        q = mx.random.normal((1, 2, 4, 8))
        k = mx.random.normal((1, 2, 4, 8))
        v = mx.random.normal((1, 2, 4, 8))
        out = _scaled_dot_product_attention(q, k, v, mask=None)
        mx.eval(out)
        assert out.shape == (1, 2, 4, 8)

    def test_runtime_errors_are_not_silently_swallowed(self, monkeypatch):
        def _oom(*args, **kwargs):
            raise RuntimeError("out of memory")

        monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", _oom)

        q = mx.random.normal((1, 2, 4, 8))
        k = mx.random.normal((1, 2, 4, 8))
        v = mx.random.normal((1, 2, 4, 8))
        with pytest.raises(RuntimeError, match="out of memory"):
            _scaled_dot_product_attention(q, k, v, mask=None)


# ---------------------------------------------------------------------------
# AudioEncoderLayer
# ---------------------------------------------------------------------------


class TestAudioEncoderLayer:
    """Test AudioEncoderLayer output shape."""

    def test_output_shape(self):
        d_model, num_heads, ffn_dim = 64, 2, 128
        layer = AudioEncoderLayer(d_model, num_heads, ffn_dim)
        x = mx.random.normal((1, 10, d_model))
        result = layer(x)
        mx.eval(result)
        assert result.shape == (1, 10, d_model)

    def test_float16_path_clamps_extreme_values(self, monkeypatch):
        layer = AudioEncoderLayer(d_model=8, num_heads=2, encoder_ffn_dim=16)

        monkeypatch.setattr(layer, "self_attn_layer_norm", lambda x: x)
        monkeypatch.setattr(
            layer,
            "self_attn",
            lambda x, mask=None: mx.full(x.shape, 70000.0, dtype=x.dtype),
        )
        monkeypatch.setattr(layer, "final_layer_norm", lambda x: x)
        monkeypatch.setattr(layer, "fc1", lambda x: x)
        monkeypatch.setattr(layer, "fc2", lambda x: x)

        x = mx.zeros((1, 2, 8), dtype=mx.float16)
        out = layer(x)
        mx.eval(out)

        clamp = np.finfo(np.float16).max - 1000.0
        assert bool(mx.all(mx.isfinite(out)).item())
        assert float(mx.max(out).item()) <= clamp + 1e-3
        assert float(mx.min(out).item()) >= -clamp - 1e-3


# ---------------------------------------------------------------------------
# AudioEncoder.get_output_lengths
# ---------------------------------------------------------------------------


class TestAudioEncoderGetOutputLengths:
    """Test AudioEncoder.get_output_lengths with known lengths."""

    @staticmethod
    def _official_output_length(input_frames: int, chunk_size: int = 100) -> int:
        """Match upstream _get_feat_extract_output_lengths behavior.

        Formula from Qwen3-ASR upstream:
        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        """
        tail_frames = input_frames % chunk_size
        tail_after_conv1 = (tail_frames - 1) // 2 + 1
        tail_after_conv2 = (tail_after_conv1 - 1) // 2 + 1
        tail_tokens = (tail_after_conv2 - 1) // 2 + 1
        if tail_frames == 0:
            tail_tokens = 0
        return tail_tokens + (input_frames // chunk_size) * 13

    def test_known_lengths(self):
        cfg = _tiny_audio_config()
        encoder = AudioEncoder(cfg)

        lengths = mx.array([100])
        result = encoder.get_output_lengths(lengths)
        mx.eval(result)
        assert result.item() == self._official_output_length(100)

    def test_400_frames(self):
        cfg = _tiny_audio_config()
        encoder = AudioEncoder(cfg)
        lengths = mx.array([400])
        result = encoder.get_output_lengths(lengths)
        mx.eval(result)
        assert result.item() == self._official_output_length(400)

    def test_1000_frames(self):
        cfg = _tiny_audio_config()
        encoder = AudioEncoder(cfg)
        lengths = mx.array([1000])
        result = encoder.get_output_lengths(lengths)
        mx.eval(result)
        assert result.item() == self._official_output_length(1000)

    def test_batch(self):
        cfg = _tiny_audio_config()
        encoder = AudioEncoder(cfg)
        lengths = mx.array([100, 400, 1000])
        result = encoder.get_output_lengths(lengths)
        mx.eval(result)
        expected = [
            self._official_output_length(100),
            self._official_output_length(400),
            self._official_output_length(1000),
        ]
        np.testing.assert_array_equal(np.array(result), expected)

    def test_small_input(self):
        cfg = _tiny_audio_config()
        encoder = AudioEncoder(cfg)
        # 1 -> (1+1)//2=1 -> 1 -> 1
        lengths = mx.array([1])
        result = encoder.get_output_lengths(lengths)
        mx.eval(result)
        assert result.item() == 1

    def test_chunk_boundaries(self):
        cfg = _tiny_audio_config()
        encoder = AudioEncoder(cfg)
        lengths = mx.array([99, 100, 101, 199, 200, 201])
        result = encoder.get_output_lengths(lengths)
        mx.eval(result)
        expected = [self._official_output_length(int(x)) for x in [99, 100, 101, 199, 200, 201]]
        np.testing.assert_array_equal(np.array(result), expected)

    def test_matches_actual_encoder_output(self):
        cfg = _tiny_audio_config()
        encoder = AudioEncoder(cfg)
        chunk_size = cfg.n_window * 2

        for frames in [1, 2, 3, 10, 50, 99, 100, 101, 199, 200, 399, 400, 401]:
            mel = mx.random.normal((cfg.num_mel_bins, frames))
            encoded = encoder._encode_single(
                mel,
                chunk_size=chunk_size,
                n_window_infer=cfg.n_window_infer,
            )
            predicted = encoder.get_output_lengths(mx.array([frames]))
            mx.eval(encoded, predicted)
            assert int(predicted.item()) == encoded.shape[0]


# ---------------------------------------------------------------------------
# AudioEncoder._encode_single optimization parity
# ---------------------------------------------------------------------------


def _encode_single_reference_loop(
    encoder: AudioEncoder,
    mel: mx.array,
    *,
    chunk_size: int,
    n_window_infer: int,
) -> mx.array:
    """Reference chunk-by-chunk implementation for encoder parity tests."""
    total_frames = mel.shape[1]
    chunk_token_lens: list[int] = []
    chunk_conv_outputs: list[mx.array] = []

    for start in range(0, total_frames, chunk_size):
        end = min(start + chunk_size, total_frames)
        chunk_mel = mel[:, start:end]
        x = encoder._apply_conv_stem(chunk_mel[None, :, :, None])
        _, F_d, T_d, C_d = x.shape
        x = x.transpose(0, 2, 3, 1).reshape(1, T_d, C_d * F_d)
        chunk_token_lens.append(int(T_d))
        chunk_conv_outputs.append(x[0])

    x = mx.concatenate(chunk_conv_outputs, axis=0)
    x = encoder.conv_out(x)

    max_chunk_tokens = max(chunk_token_lens)
    pe = encoder.embed_positions(max_chunk_tokens)
    pe_parts = [pe[:ct] for ct in chunk_token_lens]
    x = x + mx.concatenate(pe_parts, axis=0)

    total_tokens = int(x.shape[0])
    tokens_per_full_chunk = int(chunk_token_lens[0])
    tokens_per_window = tokens_per_full_chunk * (n_window_infer // chunk_size)

    cu_seqlens: list[int] = [0]
    pos = 0
    while pos < total_tokens:
        window_end = min(pos + tokens_per_window, total_tokens)
        cu_seqlens.append(window_end)
        pos = window_end

    x = x[None, :, :]
    num_windows = len(cu_seqlens) - 1
    if num_windows >= encoder_module._WINDOWED_SEGMENT_MIN_WINDOWS:
        x = _apply_windowed_encoder_layers(x, encoder.layers, cu_seqlens)
    else:
        mask = _create_windowed_mask(total_tokens, cu_seqlens, x.dtype)
        for layer in encoder.layers:
            x = layer(x, mask=mask)

    x = x[0]
    x = encoder.ln_post(x)
    x = nn.gelu(encoder.proj1(x))
    x = encoder.proj2(x)
    return x


class TestAudioEncoderEncodeSingleOptimized:
    def test_matches_reference_loop_with_tail_chunk(self):
        cfg = _tiny_audio_config()
        encoder = AudioEncoder(cfg)
        chunk_size = cfg.n_window * 2

        # Force multiple full chunks + a tail chunk.
        mel = mx.random.normal((cfg.num_mel_bins, chunk_size * 3 + 37))
        optimized = encoder._encode_single(
            mel,
            chunk_size=chunk_size,
            n_window_infer=cfg.n_window_infer,
        )
        reference = _encode_single_reference_loop(
            encoder,
            mel,
            chunk_size=chunk_size,
            n_window_infer=cfg.n_window_infer,
        )
        mx.eval(optimized, reference)
        np.testing.assert_allclose(
            np.array(optimized),
            np.array(reference),
            atol=1e-5,
            rtol=1e-5,
        )

# ---------------------------------------------------------------------------
# TextAttention
# ---------------------------------------------------------------------------


class TestTextAttention:
    """Test TextAttention output shape with small config."""

    def test_output_shape(self):
        # Need head_dim such that sections sum to half_dim
        # head_dim=48 -> half_dim=24 -> need sections summing to 24
        # We'll use a head_dim of 128 for MRoPE compatibility or mock cos/sin
        cfg_for_attn = TextDecoderConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=32,
        )
        attn = TextAttention(cfg_for_attn)
        batch, seq_len = 1, 5
        x = mx.random.normal((batch, seq_len, 64))
        # cos/sin shape: (batch, seq_len, head_dim)
        cos = mx.ones((batch, seq_len, 32))
        sin = mx.zeros((batch, seq_len, 32))
        result = attn(x, cos, sin)
        mx.eval(result)
        assert result.shape == (batch, seq_len, 64)


# ---------------------------------------------------------------------------
# SwiGLU
# ---------------------------------------------------------------------------


class TestSwiGLU:
    """Test SwiGLU output shape."""

    def test_output_shape(self):
        hidden_size, intermediate_size = 64, 128
        mlp = SwiGLU(hidden_size, intermediate_size)
        x = mx.random.normal((1, 10, hidden_size))
        result = mlp(x)
        mx.eval(result)
        assert result.shape == (1, 10, hidden_size)

    def test_2d_input(self):
        hidden_size, intermediate_size = 32, 64
        mlp = SwiGLU(hidden_size, intermediate_size)
        x = mx.random.normal((5, hidden_size))
        result = mlp(x)
        mx.eval(result)
        assert result.shape == (5, hidden_size)


# ---------------------------------------------------------------------------
# TextDecoderLayer
# ---------------------------------------------------------------------------


class TestTextDecoderLayer:
    """Test TextDecoderLayer output shape."""

    def test_output_shape(self):
        cfg = TextDecoderConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=32,
        )
        layer = TextDecoderLayer(cfg)
        batch, seq_len = 1, 5
        x = mx.random.normal((batch, seq_len, 64))
        cos = mx.ones((batch, seq_len, 32))
        sin = mx.zeros((batch, seq_len, 32))
        result = layer(x, cos, sin)
        mx.eval(result)
        assert result.shape == (batch, seq_len, 64)


# ---------------------------------------------------------------------------
# KVCache
# ---------------------------------------------------------------------------


class TestKVCache:
    """Test KVCache update and offset tracking."""

    def test_initial_state(self):
        cache = KVCache(num_layers=4)
        assert cache.offset == 0
        assert cache.seq_len == 0
        assert len(cache.keys) == 4
        assert all(k is None for k in cache.keys)

    def test_first_update(self):
        cache = KVCache(num_layers=2)
        k = mx.random.normal((1, 2, 5, 16))  # (B, H, L, D)
        v = mx.random.normal((1, 2, 5, 16))

        # Update layer 0
        k_out, v_out = cache.update(k, v, layer_idx=0)
        assert k_out.shape == (1, 2, 5, 16)
        assert cache.offset == 0  # offset only updates after last layer

        # Update layer 1 (last layer)
        k_out, v_out = cache.update(k, v, layer_idx=1)
        assert cache.offset == 5
        assert cache.seq_len == 5

    def test_sequential_updates(self):
        cache = KVCache(num_layers=2)
        k1 = mx.random.normal((1, 2, 5, 16))
        v1 = mx.random.normal((1, 2, 5, 16))

        # First step: prefill with 5 tokens
        cache.update(k1, v1, layer_idx=0)
        cache.update(k1, v1, layer_idx=1)
        assert cache.offset == 5

        # Second step: one new token
        k2 = mx.random.normal((1, 2, 1, 16))
        v2 = mx.random.normal((1, 2, 1, 16))
        k_out, v_out = cache.update(k2, v2, layer_idx=0)
        assert k_out.shape == (1, 2, 6, 16)  # 5 + 1 = 6

        cache.update(k2, v2, layer_idx=1)
        assert cache.offset == 6

    def test_preallocated_mode(self):
        cache = KVCache(num_layers=2, max_seq_len=8)
        k1 = mx.random.normal((1, 2, 5, 16))
        v1 = mx.random.normal((1, 2, 5, 16))
        k2 = mx.random.normal((1, 2, 1, 16))
        v2 = mx.random.normal((1, 2, 1, 16))

        k_out, v_out = cache.update(k1, v1, layer_idx=0)
        assert k_out.shape == (1, 2, 5, 16)
        assert v_out.shape == (1, 2, 5, 16)
        cache.update(k1, v1, layer_idx=1)
        assert cache.offset == 5

        k_out, v_out = cache.update(k2, v2, layer_idx=0)
        assert k_out.shape == (1, 2, 6, 16)
        assert v_out.shape == (1, 2, 6, 16)
        cache.update(k2, v2, layer_idx=1)
        assert cache.offset == 6

    def test_preallocated_overflow_raises(self):
        cache = KVCache(num_layers=1, max_seq_len=2)
        k = mx.random.normal((1, 2, 3, 16))
        v = mx.random.normal((1, 2, 3, 16))
        with pytest.raises(ValueError, match="KV cache overflow"):
            cache.update(k, v, layer_idx=0)

    def test_trim_non_preallocated(self):
        cache = KVCache(num_layers=1)
        k = mx.random.normal((1, 2, 5, 16))
        v = mx.random.normal((1, 2, 5, 16))
        cache.update(k, v, layer_idx=0)
        assert cache.offset == 5

        cache.trim(2)
        assert cache.offset == 3
        assert cache.keys[0].shape[2] == 3
        assert cache.values[0].shape[2] == 3

    def test_trim_preallocated(self):
        cache = KVCache(num_layers=1, max_seq_len=8)
        k = mx.random.normal((1, 2, 5, 16))
        v = mx.random.normal((1, 2, 5, 16))
        cache.update(k, v, layer_idx=0)
        assert cache.offset == 5

        cache.trim(2)
        assert cache.offset == 3
        # Buffer stays preallocated; offset controls active prefix.
        assert cache.keys[0].shape[2] == 8
        assert cache.values[0].shape[2] == 8

    def test_trim_raises_on_invalid_count(self):
        cache = KVCache(num_layers=1)
        with pytest.raises(ValueError, match="must be >= 0"):
            cache.trim(-1)
        with pytest.raises(ValueError, match="Cannot trim"):
            cache.trim(1)


# ---------------------------------------------------------------------------
# _create_causal_mask
# ---------------------------------------------------------------------------


class TestCreateCausalMask:
    """Test _create_causal_mask shape and values."""

    def test_shape(self):
        mask = _create_causal_mask(4)
        assert mask.shape == (1, 1, 4, 4)

    def test_values(self):
        mask = _create_causal_mask(3)
        mx.eval(mask)
        m = np.array(mask[0, 0])
        # Diagonal and below should be 0
        assert m[0, 0] == 0.0
        assert m[1, 0] == 0.0
        assert m[1, 1] == 0.0
        assert m[2, 0] == 0.0
        assert m[2, 1] == 0.0
        assert m[2, 2] == 0.0
        # Above diagonal should be large negative
        assert m[0, 1] == pytest.approx(-1e9)
        assert m[0, 2] == pytest.approx(-1e9)
        assert m[1, 2] == pytest.approx(-1e9)

    def test_seq_len_1(self):
        mask = _create_causal_mask(1)
        mx.eval(mask)
        assert mask.shape == (1, 1, 1, 1)
        assert np.array(mask[0, 0, 0, 0]) == 0.0


# ---------------------------------------------------------------------------
# Windowed encoder execution
# ---------------------------------------------------------------------------


class TestWindowedEncoderExecution:
    """Windowed per-segment execution should match dense masked execution."""

    def test_mask_uses_dtype_min_for_blocked_positions(self):
        mask = _create_windowed_mask(seq_len=6, cu_seqlens=[0, 3, 6], dtype=mx.float16)
        assert mask is not None
        mx.eval(mask)
        m = np.array(mask[0, 0])

        # Same-window positions stay unmasked.
        assert m[0, 1] == 0.0
        assert m[4, 5] == 0.0
        # Cross-window positions match dtype minimum (upstream torch behavior).
        assert m[0, 4] == pytest.approx(np.finfo(np.float16).min)
        assert m[5, 2] == pytest.approx(np.finfo(np.float16).min)

    def test_matches_dense_masked_layers(self):
        d_model, num_heads, ffn_dim = 64, 2, 128
        layers = [AudioEncoderLayer(d_model, num_heads, ffn_dim) for _ in range(2)]
        x = mx.random.normal((1, 11, d_model))
        cu_seqlens = [0, 4, 9, 11]

        mask = _create_windowed_mask(seq_len=11, cu_seqlens=cu_seqlens, dtype=x.dtype)
        dense = x
        for layer in layers:
            dense = layer(dense, mask=mask)

        windowed = _apply_windowed_encoder_layers(x, layers, cu_seqlens)

        mx.eval(dense, windowed)
        np.testing.assert_allclose(
            np.array(dense),
            np.array(windowed),
            atol=1e-5,
            rtol=1e-5,
        )


# ---------------------------------------------------------------------------
# Encoder dtype preservation
# ---------------------------------------------------------------------------


class TestEncoderDtype:
    def test_float16_input_stays_float16_through_encoder(self):
        """The float32 PE table must not promote encoder activations."""
        enc = AudioEncoder(_tiny_audio_config())
        params = enc.parameters()
        enc.update(
            __import__("mlx.utils").utils.tree_map(
                lambda a: a.astype(mx.float16) if a.dtype == mx.float32 else a, params
            )
        )
        mel = mx.zeros((1, 128, 100), dtype=mx.float16)
        out, _ = enc(mel, mx.array([100], dtype=mx.int32))
        mx.eval(out)
        assert out.dtype == mx.float16


# ---------------------------------------------------------------------------
# Thread portability (issue #16)
# ---------------------------------------------------------------------------


class TestThreadPortability:
    """A model built on one thread must run on another.

    MLX >= 0.31 binds unevaluated arrays to the thread that created them. Any
    init-time buffer that is not a parameter (sinusoidal PE, MRoPE inverse
    frequencies) must therefore be materialized in ``__init__``; otherwise the
    first forward pass from a different thread fails with
    ``There is no Stream(gpu, 1) in current thread``. Older MLX passes this
    test regardless, so CI on current MLX is the real guard.
    """

    def test_forward_on_other_thread_after_init_on_main(self):
        from concurrent.futures import ThreadPoolExecutor

        cfg = _tiny_asr_config()
        cfg.text_config.head_dim = 128
        model = Qwen3ASRModel(cfg)
        mx.eval(model.parameters())  # mirrors load_model(); leaves buffers lazy

        def run() -> tuple[tuple[int, ...], tuple[int, ...]]:
            # Inputs are built on the inference thread, as a real caller would.
            mel = mx.zeros((1, cfg.audio_config.num_mel_bins, 100), dtype=mx.float32)
            feature_lens = mx.array([100], dtype=mx.int32)
            input_ids = mx.array([[1, 2, 3]], dtype=mx.int32)
            position_ids = mx.zeros((1, 3, 3), dtype=mx.int32)
            audio_features, _ = model.audio_tower(mel, feature_lens)
            logits = model(input_ids, position_ids=position_ids)
            mx.eval(audio_features, logits)
            return tuple(audio_features.shape), tuple(logits.shape)

        with ThreadPoolExecutor(max_workers=1) as pool:
            audio_shape, logits_shape = pool.submit(run).result()

        assert audio_shape[0] == 1 and audio_shape[2] == cfg.audio_config.output_dim
        assert logits_shape == (1, 3, cfg.text_config.vocab_size)


# ---------------------------------------------------------------------------
# Qwen3ASRModel instantiation
# ---------------------------------------------------------------------------


class TestQwen3ASRModelInstantiation:
    """Test Qwen3ASRModel instantiation with tiny config."""

    def test_instantiation(self):
        cfg = _tiny_asr_config()
        # Need head_dim with valid MRoPE sections. Use head_dim=128 for compatibility.
        cfg.text_config.head_dim = 128
        model = Qwen3ASRModel(cfg)
        assert model.audio_token_id == cfg.audio_token_id
        assert model.num_audio_encoder_layers == 2
        assert model.num_text_decoder_layers == 2

    def test_create_cache(self):
        cfg = _tiny_asr_config()
        cfg.text_config.head_dim = 128
        model = Qwen3ASRModel(cfg)
        cache = model.create_cache()
        assert isinstance(cache, KVCache)
        assert len(cache.keys) == cfg.text_config.num_hidden_layers

    def test_default_config_instantiation(self):
        """Instantiation with full default config should not error."""
        cfg = Qwen3ASRConfig()
        model = Qwen3ASRModel(cfg)
        assert model.num_audio_encoder_layers == 24
        assert model.num_text_decoder_layers == 28

    def test_classify_num_changes_output_head_size(self):
        cfg = _tiny_asr_config()
        cfg.text_config.head_dim = 128
        cfg.classify_num = 64
        model = Qwen3ASRModel(cfg)
        assert model.lm_head.weight.shape[0] == 64
        assert model.model.embed_tokens.weight.shape[0] == cfg.text_config.vocab_size

    def test_inject_audio_features_raises_on_out_of_bounds_placeholders(self):
        cfg = _tiny_asr_config()
        cfg.text_config.head_dim = 128
        model = Qwen3ASRModel(cfg)

        embeds = mx.random.normal((1, 6, cfg.text_config.hidden_size))
        audio_features = mx.random.normal((1, 2, cfg.text_config.hidden_size))
        audio_mask = mx.array([[False, True, True, True, False, False]])

        with pytest.raises(ValueError, match="Audio injection out of bounds"):
            model._inject_audio_features(embeds, audio_features, audio_mask)

    def test_inject_audio_features_raises_on_dimension_mismatch(self):
        cfg = _tiny_asr_config()
        cfg.text_config.head_dim = 128
        model = Qwen3ASRModel(cfg)

        embeds = mx.random.normal((1, 4, cfg.text_config.hidden_size))
        audio_features = mx.random.normal((1, 2, cfg.text_config.hidden_size + 1))
        audio_mask = mx.array([[False, True, True, False]])

        with pytest.raises(ValueError, match="Audio injection channel mismatch"):
            model._inject_audio_features(embeds, audio_features, audio_mask)

    def test_inject_audio_features_no_audio_placeholders_is_noop(self):
        cfg = _tiny_asr_config()
        cfg.text_config.head_dim = 128
        model = Qwen3ASRModel(cfg)

        embeds = mx.random.normal((1, 5, cfg.text_config.hidden_size))
        audio_features = mx.zeros((1, 0, cfg.text_config.hidden_size))
        audio_mask = mx.array([[False, False, False, False, False]])

        out = model._inject_audio_features(embeds, audio_features, audio_mask)
        mx.eval(out)
        np.testing.assert_allclose(np.array(out), np.array(embeds), atol=0.0, rtol=0.0)

    def test_step_rejects_out_of_vocab_token_ids(self):
        cfg = _tiny_asr_config()
        cfg.text_config.head_dim = 128
        model = Qwen3ASRModel(cfg)
        cache = model.create_cache()
        bad_token = cfg.text_config.vocab_size
        input_ids = mx.array([[bad_token]], dtype=mx.int32)
        position_ids = mx.zeros((1, 3, 1), dtype=mx.int32)

        with pytest.raises(ValueError, match="input_ids out of bounds for embed_tokens"):
            model.step(input_ids=input_ids, position_ids=position_ids, cache=cache)

    def test_step_rejects_non_integer_input_ids(self):
        cfg = _tiny_asr_config()
        cfg.text_config.head_dim = 128
        model = Qwen3ASRModel(cfg)
        cache = model.create_cache()
        input_ids = mx.array([[1.0]], dtype=mx.float32)
        position_ids = mx.zeros((1, 3, 1), dtype=mx.int32)

        with pytest.raises(ValueError, match="input_ids must use an integer dtype"):
            model.step(input_ids=input_ids, position_ids=position_ids, cache=cache)

    def test_step_unchecked_path_skips_range_validation(self, monkeypatch):
        cfg = _tiny_asr_config()
        cfg.text_config.head_dim = 128
        model = Qwen3ASRModel(cfg)
        cache = model.create_cache()
        input_ids = mx.array([[1]], dtype=mx.int32)
        position_ids = mx.zeros((1, 3, 1), dtype=mx.int32)

        def _fail_range_check(_: mx.array) -> None:
            raise AssertionError("range validation should be skipped in unchecked path")

        monkeypatch.setattr(model, "_validate_input_ids_range", _fail_range_check)

        logits = model.step(
            input_ids=input_ids,
            position_ids=position_ids,
            cache=cache,
            validate_input_ids=False,
        )
        assert tuple(logits.shape) == (1, 1, cfg.text_config.vocab_size)

    def test_step_unchecked_path_still_rejects_non_integer_dtype(self):
        cfg = _tiny_asr_config()
        cfg.text_config.head_dim = 128
        model = Qwen3ASRModel(cfg)
        cache = model.create_cache()
        input_ids = mx.array([[1.0]], dtype=mx.float32)
        position_ids = mx.zeros((1, 3, 1), dtype=mx.int32)

        with pytest.raises(ValueError, match="input_ids must use an integer dtype"):
            model.step(
                input_ids=input_ids,
                position_ids=position_ids,
                cache=cache,
                validate_input_ids=False,
            )
