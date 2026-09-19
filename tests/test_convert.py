"""Tests for mlx_qwen3_asr/convert.py."""

import mlx.core as mx
import numpy as np

from mlx_qwen3_asr.convert import remap_weights


class TestRemapWeightsPrefix:
    """Test remap_weights() strips 'thinker.' prefix."""

    def test_strips_thinker_prefix(self):
        weights = {
            "thinker.audio_tower.conv2d1.bias": mx.zeros((10,)),
            "thinker.model.layers.0.self_attn.q_proj.weight": mx.zeros((64, 64)),
            "thinker.lm_head.weight": mx.zeros((100, 64)),
        }
        remapped = remap_weights(weights)
        assert "audio_tower.conv2d1.bias" in remapped
        assert "model.layers.0.self_attn.q_proj.weight" in remapped
        assert "lm_head.weight" in remapped
        # Old keys should not be present
        assert "thinker.audio_tower.conv2d1.bias" not in remapped

    def test_leaves_non_thinker_keys_unchanged(self):
        weights = {
            "audio_tower.conv2d1.bias": mx.zeros((10,)),
            "model.norm.weight": mx.zeros((64,)),
        }
        remapped = remap_weights(weights)
        assert "audio_tower.conv2d1.bias" in remapped
        assert "model.norm.weight" in remapped


class TestRemapWeightsConv2d:
    """Test remap_weights() transposes Conv2d weights correctly."""

    def test_transposes_conv2d_weights(self):
        """Conv2d weights (4D with 'conv2d' in key) should be transposed
        from PyTorch (out, in, kH, kW) to MLX (out, kH, kW, in).
        """
        # PyTorch format: (out_channels=8, in_channels=3, kH=3, kW=3)
        pt_weight = mx.arange(8 * 3 * 3 * 3).reshape(8, 3, 3, 3).astype(mx.float32)
        weights = {"thinker.audio_tower.conv2d1.weight": pt_weight}
        remapped = remap_weights(weights)

        result = remapped["audio_tower.conv2d1.weight"]
        # Should be (out=8, kH=3, kW=3, in=3) after transpose(0, 2, 3, 1)
        assert result.shape == (8, 3, 3, 3)

        # Verify the transpose is correct: value at [o, i, h, w] in PT
        # should be at [o, h, w, i] in MLX
        mx.eval(pt_weight, result)
        pt_np = np.array(pt_weight)
        result_np = np.array(result)
        for o in range(2):  # Spot-check a few values
            for i in range(3):
                for h in range(3):
                    for w in range(3):
                        assert pt_np[o, i, h, w] == result_np[o, h, w, i]

    def test_does_not_transpose_non_conv2d_4d(self):
        """4D tensors without 'conv2d' in the key should NOT be transposed."""
        weight = mx.zeros((4, 3, 3, 3))
        weights = {"thinker.some_other_layer.weight": weight}
        remapped = remap_weights(weights)
        result = remapped["some_other_layer.weight"]
        assert result.shape == (4, 3, 3, 3)

    def test_does_not_transpose_2d_with_conv2d_name(self):
        """2D tensors with 'conv2d' in the key (like bias) should not be transposed."""
        bias = mx.zeros((10,))
        weights = {"thinker.audio_tower.conv2d1.bias": bias}
        remapped = remap_weights(weights)
        result = remapped["audio_tower.conv2d1.bias"]
        assert result.shape == (10,)

    def test_does_not_transpose_non_thinker_conv2d_weight(self):
        """Already-converted local MLX weights should not be transposed again."""
        mlx_weight = mx.zeros((8, 3, 3, 1))
        weights = {"audio_tower.conv2d1.weight": mlx_weight}
        remapped = remap_weights(weights)
        result = remapped["audio_tower.conv2d1.weight"]
        assert result.shape == (8, 3, 3, 1)

    def test_conv2d2_and_conv2d3(self):
        """All three conv2d layers should be transposed."""
        for name in ["conv2d1", "conv2d2", "conv2d3"]:
            w = mx.zeros((16, 8, 3, 3))
            weights = {f"thinker.audio_tower.{name}.weight": w}
            remapped = remap_weights(weights)
            key = f"audio_tower.{name}.weight"
            assert key in remapped
            # After transpose(0, 2, 3, 1): (16, 3, 3, 8)
            assert remapped[key].shape == (16, 3, 3, 8)


class TestRemapWeightsMixed:
    """Test remap_weights() with a mix of weight types."""

    def test_mixed_weights(self):
        weights = {
            "thinker.audio_tower.conv2d1.weight": mx.zeros((32, 1, 3, 3)),
            "thinker.audio_tower.conv2d1.bias": mx.zeros((32,)),
            "thinker.model.layers.0.self_attn.q_proj.weight": mx.zeros((64, 64)),
            "thinker.lm_head.weight": mx.zeros((100, 64)),
        }
        remapped = remap_weights(weights)
        # Conv2d weight transposed
        assert remapped["audio_tower.conv2d1.weight"].shape == (32, 3, 3, 1)
        # Bias unchanged
        assert remapped["audio_tower.conv2d1.bias"].shape == (32,)
        # 2D weights unchanged
        assert remapped["model.layers.0.self_attn.q_proj.weight"].shape == (64, 64)
        assert remapped["lm_head.weight"].shape == (100, 64)


class TestQuantizeModelEncoderBits:
    def _model(self):
        import mlx.nn as nn

        class _Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.audio_tower = nn.Sequential(nn.Linear(64, 64))
                self.model = nn.Sequential(nn.Linear(64, 64))
                self.embed = nn.Embedding(16, 64)

        return _Tiny(), nn

    def test_default_quantizes_encoder_and_decoder_at_same_width(self):
        from mlx_qwen3_asr.convert import quantize_model

        model, nn = self._model()
        quantize_model(model, bits=4, group_size=64)
        assert isinstance(model.audio_tower.layers[0], nn.QuantizedLinear)
        assert model.audio_tower.layers[0].bits == 4
        assert model.model.layers[0].bits == 4
        assert isinstance(model.embed, nn.QuantizedEmbedding)

    def test_encoder_bits_sets_a_separate_width(self):
        from mlx_qwen3_asr.convert import quantize_model

        model, nn = self._model()
        quantize_model(model, bits=4, group_size=64, encoder_bits=8)
        assert model.audio_tower.layers[0].bits == 8
        assert model.model.layers[0].bits == 4
        assert model.embed.bits == 4

    def test_encoder_bits_16_leaves_encoder_in_float(self):
        from mlx_qwen3_asr.convert import quantize_model

        model, nn = self._model()
        quantize_model(model, bits=4, group_size=64, encoder_bits=16)
        assert isinstance(model.audio_tower.layers[0], nn.Linear)
        assert not isinstance(model.audio_tower.layers[0], nn.QuantizedLinear)
        assert model.model.layers[0].bits == 4
