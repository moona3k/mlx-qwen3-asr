"""Tests for mlx_qwen3_asr/load_models.py."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as mlx_utils

from mlx_qwen3_asr.config import (
    ACCURACY_MODEL_ID,
    AudioEncoderConfig,
    Qwen3ASRConfig,
    TextDecoderConfig,
)
from mlx_qwen3_asr.load_models import (
    _cast_tree_dtype,
    _infer_quantization_params,
    _is_quantized_weights,
    _load_model_with_resolved_path,
    _materialize_tied_lm_head_weights,
    _ModelHolder,
    _quantize_model_for_loaded_weights,
    _quantized_module_paths,
    _read_quantization_config,
    _resolve_path,
)
from mlx_qwen3_asr.model import Qwen3ASRModel


def _tiny_config(*, tie_word_embeddings: bool = True) -> Qwen3ASRConfig:
    """Build a small valid ASR config for loader tests."""
    return Qwen3ASRConfig(
        audio_config=AudioEncoderConfig(
            num_mel_bins=128,
            encoder_layers=1,
            encoder_attention_heads=2,
            encoder_ffn_dim=64,
            d_model=32,
            output_dim=256,
            max_source_positions=100,
            downsample_hidden_size=24,
        ),
        text_config=TextDecoderConfig(
            vocab_size=128,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=128,
            tie_word_embeddings=tie_word_embeddings,
        ),
    )


def _tiny_config_dict(*, tie_word_embeddings: bool = True) -> dict:
    """Return the nested JSON config shape used by checkpoint directories."""
    return {
        "thinker_config": {
            "audio_config": {
                "num_mel_bins": 128,
                "encoder_layers": 1,
                "encoder_attention_heads": 2,
                "encoder_ffn_dim": 64,
                "d_model": 32,
                "output_dim": 256,
                "max_source_positions": 100,
                "downsample_hidden_size": 24,
            },
            "text_config": {
                "vocab_size": 128,
                "hidden_size": 256,
                "intermediate_size": 512,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 128,
                "tie_word_embeddings": tie_word_embeddings,
            },
        }
    }


def _write_tiny_config(model_dir: Path, *, tie_word_embeddings: bool = True) -> None:
    """Write a tiny nested config.json into a checkpoint fixture directory."""
    (model_dir / "config.json").write_text(
        json.dumps(_tiny_config_dict(tie_word_embeddings=tie_word_embeddings)),
        encoding="utf-8",
    )


class TestCastTreeDtype:
    """Test recursive dtype casting of parameter trees."""

    def test_casts_nested_arrays(self):
        tree = {
            "a": mx.ones((2, 2), dtype=mx.float32),
            "b": {
                "c": [mx.zeros((1,), dtype=mx.float32), {"d": mx.array([3.0])}],
            },
            "int_array": mx.array([1, 2, 3], dtype=mx.int32),
            "name": "keep-me",
        }

        casted = _cast_tree_dtype(tree, mx.float16)
        leaves = mlx_utils.tree_flatten(casted)

        for _, value in leaves:
            if isinstance(value, mx.array) and mx.issubdtype(value.dtype, mx.floating):
                assert value.dtype == mx.float16

        assert casted["int_array"].dtype == mx.int32

        assert casted["name"] == "keep-me"


class TestTiedLmHeadWeights:
    def test_materializes_missing_tied_lm_head_weight(self):
        embedding = mx.zeros((128, 48), dtype=mx.float16)
        weights = {"model.embed_tokens.weight": embedding}

        patched = _materialize_tied_lm_head_weights(weights, _tiny_config())

        assert patched["lm_head.weight"] is embedding
        assert "lm_head.weight" not in weights

    def test_materializes_quantized_tied_lm_head_aux_tensors(self):
        embedding = mx.zeros((128, 6), dtype=mx.uint32)
        scales = mx.ones((128, 1), dtype=mx.float16)
        biases = mx.zeros((128, 1), dtype=mx.float16)
        weights = {
            "model.embed_tokens.weight": embedding,
            "model.embed_tokens.scales": scales,
            "model.embed_tokens.biases": biases,
        }

        patched = _materialize_tied_lm_head_weights(weights, _tiny_config())

        assert patched["lm_head.weight"] is embedding
        assert patched["lm_head.scales"] is scales
        assert patched["lm_head.biases"] is biases

    def test_preserves_explicit_lm_head_weight(self):
        embedding = mx.zeros((128, 48), dtype=mx.float16)
        lm_head = mx.ones((128, 48), dtype=mx.float16)
        weights = {
            "model.embed_tokens.weight": embedding,
            "lm_head.weight": lm_head,
        }

        patched = _materialize_tied_lm_head_weights(weights, _tiny_config())

        assert patched is weights
        assert patched["lm_head.weight"] is lm_head

    def test_materializes_missing_aux_tensors_for_explicit_tied_lm_head(self):
        embedding = mx.zeros((128, 6), dtype=mx.uint32)
        lm_head = mx.ones((128, 6), dtype=mx.uint32)
        scales = mx.ones((128, 1), dtype=mx.float16)
        biases = mx.zeros((128, 1), dtype=mx.float16)
        weights = {
            "model.embed_tokens.weight": embedding,
            "model.embed_tokens.scales": scales,
            "model.embed_tokens.biases": biases,
            "lm_head.weight": lm_head,
        }

        patched = _materialize_tied_lm_head_weights(weights, _tiny_config())

        assert patched["lm_head.weight"] is lm_head
        assert patched["lm_head.scales"] is scales
        assert patched["lm_head.biases"] is biases

    def test_does_not_materialize_aux_tensors_for_shape_mismatched_lm_head(self):
        embedding = mx.zeros((128, 6), dtype=mx.uint32)
        lm_head = mx.ones((128, 48), dtype=mx.float16)
        scales = mx.ones((128, 1), dtype=mx.float16)
        weights = {
            "model.embed_tokens.weight": embedding,
            "model.embed_tokens.scales": scales,
            "lm_head.weight": lm_head,
        }

        patched = _materialize_tied_lm_head_weights(weights, _tiny_config())

        assert patched is weights
        assert "lm_head.scales" not in patched

    def test_does_not_materialize_when_embeddings_are_not_tied(self):
        embedding = mx.zeros((128, 48), dtype=mx.float16)
        weights = {"model.embed_tokens.weight": embedding}

        patched = _materialize_tied_lm_head_weights(
            weights,
            _tiny_config(tie_word_embeddings=False),
        )

        assert patched is weights
        assert "lm_head.weight" not in patched


class TestLoadModelWithCommunityLayouts:
    def test_loads_tied_checkpoint_without_explicit_lm_head(self, tmp_path: Path):
        model_dir = tmp_path / "tied-bf16"
        model_dir.mkdir()
        _write_tiny_config(model_dir)

        model = Qwen3ASRModel(_tiny_config())
        weights = dict(mlx_utils.tree_flatten(model.parameters()))
        weights.pop("lm_head.weight")
        mx.save_safetensors(str(model_dir / "model.safetensors"), weights)

        loaded, _, _ = _load_model_with_resolved_path(str(model_dir), dtype=mx.float16)
        params = dict(mlx_utils.tree_flatten(loaded.parameters()))

        assert "lm_head.weight" in params
        assert params["lm_head.weight"].shape == params["model.embed_tokens.weight"].shape

    def test_tied_lm_head_shares_the_embedding_array_after_cast(self, tmp_path: Path):
        """load_weights + dtype cast untie the head; loader must re-tie it."""
        model_dir = tmp_path / "tied-fp32"
        model_dir.mkdir()
        _write_tiny_config(model_dir)
        model = Qwen3ASRModel(_tiny_config())
        weights = dict(mlx_utils.tree_flatten(model.parameters()))
        mx.save_safetensors(str(model_dir / "model.safetensors"), weights)

        loaded, _, _ = _load_model_with_resolved_path(str(model_dir), dtype=mx.float16)

        assert loaded.lm_head.weight is loaded.model.embed_tokens.weight
        assert loaded.lm_head.weight.dtype == mx.float16

    def test_loads_partially_quantized_checkpoint(self, tmp_path: Path):
        model_dir = tmp_path / "partial-quant"
        model_dir.mkdir()
        _write_tiny_config(model_dir)
        (model_dir / "quantization_config.json").write_text(
            '{"bits": 4, "group_size": 64}',
            encoding="utf-8",
        )

        model = Qwen3ASRModel(_tiny_config())
        quantized_paths = {
            "model.embed_tokens",
            "model.layers.0.self_attn.q_proj",
            "lm_head",
        }
        nn.quantize(
            model,
            bits=4,
            group_size=64,
            class_predicate=lambda path, _module: path in quantized_paths,
        )
        weights = dict(mlx_utils.tree_flatten(model.parameters()))
        for key in ("lm_head.weight", "lm_head.scales", "lm_head.biases"):
            weights.pop(key)
        mx.save_safetensors(str(model_dir / "model.safetensors"), weights)

        loaded, _, _ = _load_model_with_resolved_path(str(model_dir), dtype=mx.float16)
        params = dict(mlx_utils.tree_flatten(loaded.parameters()))

        assert "lm_head.scales" in params
        assert "model.layers.0.self_attn.q_proj.scales" in params
        assert not any(
            key.startswith("audio_tower.") and key.endswith((".scales", ".biases"))
            for key in params
        )


class TestResolvePath:
    """Test model path resolution logic."""

    def test_uses_local_path_when_config_exists(self, tmp_path: Path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")

        resolved = _resolve_path(str(model_dir))
        assert resolved == model_dir

    def test_downloads_from_hub_for_nonlocal_path(self, monkeypatch):
        expected = "/tmp/fake-model-dir"

        def fake_snapshot_download(repo_id, allow_patterns):  # noqa: ANN001
            assert repo_id == ACCURACY_MODEL_ID
            assert "*.safetensors" in allow_patterns
            return expected

        monkeypatch.setattr(
            "huggingface_hub.snapshot_download",
            fake_snapshot_download,
        )

        resolved = _resolve_path(ACCURACY_MODEL_ID)
        assert resolved == Path(expected)


class TestModelHolder:
    def test_get_resolved_path_uses_cached_resolve(self, monkeypatch):
        _ModelHolder.clear()
        sentinel_model = object()
        sentinel_cfg = object()

        def fake_loader(path_or_hf_repo, dtype):  # noqa: ANN001
            assert path_or_hf_repo == ACCURACY_MODEL_ID
            return sentinel_model, sentinel_cfg, Path("/tmp/qwen3-resolved")

        monkeypatch.setattr(
            "mlx_qwen3_asr.load_models._load_model_with_resolved_path",
            fake_loader,
        )

        model, cfg = _ModelHolder.get(ACCURACY_MODEL_ID, dtype=mx.float16)
        assert model is sentinel_model
        assert cfg is sentinel_cfg
        assert _ModelHolder.get_resolved_path(ACCURACY_MODEL_ID, dtype=mx.float16) == (
            "/tmp/qwen3-resolved"
        )

        _ModelHolder.clear()

    def test_caches_multiple_models_without_eviction(self, monkeypatch):
        _ModelHolder.clear()
        calls: list[tuple[str, mx.Dtype]] = []
        store: dict[str, object] = {}

        def fake_loader(path_or_hf_repo, dtype):  # noqa: ANN001
            calls.append((path_or_hf_repo, dtype))
            model = store.setdefault(f"m:{path_or_hf_repo}", object())
            cfg = store.setdefault(f"c:{path_or_hf_repo}", object())
            return model, cfg, Path(f"/tmp/{path_or_hf_repo.replace('/', '_')}")

        monkeypatch.setattr(
            "mlx_qwen3_asr.load_models._load_model_with_resolved_path",
            fake_loader,
        )

        m_a_1, _ = _ModelHolder.get("Qwen/A", dtype=mx.float16)
        m_b_1, _ = _ModelHolder.get("Qwen/B", dtype=mx.float16)
        m_a_2, _ = _ModelHolder.get("Qwen/A", dtype=mx.float16)

        assert m_a_1 is m_a_2
        assert m_a_1 is not m_b_1
        assert calls == [("Qwen/A", mx.float16), ("Qwen/B", mx.float16)]
        assert _ModelHolder.get_resolved_path("Qwen/B", dtype=mx.float16) == "/tmp/Qwen_B"

        _ModelHolder.clear()

    def test_cache_key_isolated_by_dtype(self, monkeypatch):
        _ModelHolder.clear()
        calls: list[tuple[str, mx.Dtype]] = []
        store: dict[tuple[str, str], tuple[object, object]] = {}

        def fake_loader(path_or_hf_repo, dtype):  # noqa: ANN001
            calls.append((path_or_hf_repo, dtype))
            key = (path_or_hf_repo, str(dtype))
            model, cfg = store.setdefault(key, (object(), object()))
            return model, cfg, Path(f"/tmp/{path_or_hf_repo.replace('/', '_')}")

        monkeypatch.setattr(
            "mlx_qwen3_asr.load_models._load_model_with_resolved_path",
            fake_loader,
        )

        m_f16, _ = _ModelHolder.get("Qwen/A", dtype=mx.float16)
        m_f32, _ = _ModelHolder.get("Qwen/A", dtype=mx.float32)
        m_f16_2, _ = _ModelHolder.get("Qwen/A", dtype=mx.float16)

        assert m_f16 is m_f16_2
        assert m_f16 is not m_f32
        assert calls == [("Qwen/A", mx.float16), ("Qwen/A", mx.float32)]

        _ModelHolder.clear()

    def test_lru_eviction_when_capacity_exceeded(self, monkeypatch):
        _ModelHolder.clear()
        _ModelHolder.set_cache_capacity(1)
        calls: list[tuple[str, mx.Dtype]] = []

        def fake_loader(path_or_hf_repo, dtype):  # noqa: ANN001
            calls.append((path_or_hf_repo, dtype))
            model, cfg = object(), object()
            return model, cfg, Path(f"/tmp/{path_or_hf_repo.replace('/', '_')}")

        monkeypatch.setattr(
            "mlx_qwen3_asr.load_models._load_model_with_resolved_path",
            fake_loader,
        )

        try:
            m_a_1, _ = _ModelHolder.get("Qwen/A", dtype=mx.float16)
            _ = _ModelHolder.get("Qwen/B", dtype=mx.float16)
            m_a_2, _ = _ModelHolder.get("Qwen/A", dtype=mx.float16)

            assert m_a_1 is not m_a_2
            assert calls == [
                ("Qwen/A", mx.float16),
                ("Qwen/B", mx.float16),
                ("Qwen/A", mx.float16),
            ]
        finally:
            _ModelHolder.set_cache_capacity(4)
            _ModelHolder.clear()


class _FakeModel:
    def __init__(self):
        self._params = {
            "layer.weight": mx.zeros((4, 64), dtype=mx.float32),
            "other.weight": mx.zeros((8, 128), dtype=mx.float32),
        }

    def parameters(self):
        return self._params


class TestQuantizationHelpers:
    def test_is_quantized_weights(self):
        assert _is_quantized_weights({"a.weight": mx.zeros((1, 1))}) is False
        assert _is_quantized_weights({"a.scales": mx.zeros((1, 1))}) is True

    def test_infer_quantization_params(self):
        # layer.weight input_dim=64, packed_cols=8 -> bits=4
        # layer.scales cols=1 -> group_size=64
        weights = {
            "layer.weight": mx.zeros((4, 8), dtype=mx.uint32),
            "layer.scales": mx.zeros((4, 1), dtype=mx.float16),
            "layer.biases": mx.zeros((4, 1), dtype=mx.float16),
            # Add one noisy candidate that should be ignored for group-size mode.
            "other.weight": mx.zeros((8, 16), dtype=mx.uint32),
            "other.scales": mx.zeros((8, 16), dtype=mx.float16),
            "other.biases": mx.zeros((8, 16), dtype=mx.float16),
        }
        bits, group_size = _infer_quantization_params(weights, _FakeModel())
        assert bits == 4
        assert group_size == 64

    def test_read_quantization_config(self, tmp_path: Path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        cfg_path = model_dir / "quantization_config.json"
        cfg_path.write_text('{"bits": 4, "group_size": 64}', encoding="utf-8")
        cfg = _read_quantization_config(model_dir)
        assert cfg == {"bits": 4, "group_size": 64}

    def test_read_quantization_config_returns_none_on_invalid_json(self, tmp_path: Path, caplog):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        cfg_path = model_dir / "quantization_config.json"
        cfg_path.write_text("{not-json", encoding="utf-8")

        cfg = _read_quantization_config(model_dir)
        assert cfg is None
        assert "Failed to parse quantization metadata" in caplog.text

    def test_quantized_module_paths_come_from_saved_scales(self):
        weights = {
            "audio_tower.conv_out.weight": mx.zeros((1, 1)),
            "model.embed_tokens.weight": mx.zeros((1, 1)),
            "model.embed_tokens.scales": mx.zeros((1, 1)),
            "model.layers.0.self_attn.q_proj.scales": mx.zeros((1, 1)),
            "lm_head.scales": mx.zeros((1, 1)),
        }

        assert _quantized_module_paths(weights) == {
            "model.embed_tokens",
            "model.layers.0.self_attn.q_proj",
            "lm_head",
        }

    def test_quantize_model_only_converts_modules_with_saved_scales(self):
        model = Qwen3ASRModel(_tiny_config())
        weights = {
            "model.embed_tokens.scales": mx.zeros((1, 1)),
            "model.layers.0.self_attn.q_proj.scales": mx.zeros((1, 1)),
            "lm_head.scales": mx.zeros((1, 1)),
        }

        _quantize_model_for_loaded_weights(model, weights, bits=4, group_size=64)
        params = dict(mlx_utils.tree_flatten(model.parameters()))

        assert "model.embed_tokens.scales" in params
        assert "model.layers.0.self_attn.q_proj.scales" in params
        assert "lm_head.scales" in params
        assert not any(
            key.startswith("audio_tower.") and key.endswith((".scales", ".biases"))
            for key in params
        )
        assert "model.layers.0.self_attn.k_proj.scales" not in params
