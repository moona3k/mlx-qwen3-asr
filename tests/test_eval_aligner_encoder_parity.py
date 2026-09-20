"""Unit tests for scripts/eval_aligner_encoder_parity.py pure helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_script_module():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    script_path = scripts_dir / "eval_aligner_encoder_parity.py"
    spec = importlib.util.spec_from_file_location("eval_aligner_encoder_parity_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_aligner_encoder_parity_script"] = module
    spec.loader.exec_module(module)
    return module


def test_compare_encoder_outputs_statistics():
    mod = _load_script_module()
    ref = np.array([[1.0, -1.0], [2.0, -2.0], [4.0, -4.0]], dtype=np.float32)
    mlx = ref.copy()
    mlx[-1, 0] += 0.6  # only the last token differs
    stats = mod.compare_encoder_outputs(mlx, ref)

    assert stats["n_tokens"] == 3
    assert stats["hidden"] == 2
    assert stats["max_abs_err"] == pytest.approx(0.6)
    assert stats["mae"] == pytest.approx(0.6 / 6)
    assert stats["last_token_mae"] == pytest.approx(0.3)
    assert stats["reference_mean_abs"] == pytest.approx(14.0 / 6)
    assert stats["relative_mae"] == pytest.approx((0.6 / 6) / (14.0 / 6))


def test_compare_encoder_outputs_rejects_shape_mismatch():
    mod = _load_script_module()
    with pytest.raises(ValueError, match="shape mismatch"):
        mod.compare_encoder_outputs(np.zeros((3, 4)), np.zeros((4, 4)))


def test_summarize_and_thresholds():
    mod = _load_script_module()
    rows = [
        {"mae": 0.001, "max_abs_err": 0.01, "last_token_mae": 0.002, "relative_mae": 0.0007},
        {"mae": 0.003, "max_abs_err": 0.05, "last_token_mae": 0.004, "relative_mae": 0.0009},
    ]
    summary = mod.summarize_rows(rows)
    assert summary["clips"] == 2
    assert summary["mae_mean"] == pytest.approx(0.002)
    assert summary["last_token_mae_max"] == pytest.approx(0.004)
    assert summary["relative_mae_max"] == pytest.approx(0.0009)

    assert (
        mod.threshold_failures(
            summary,
            fail_mae_mean_above=0.01,
            fail_last_token_mae_max_above=0.01,
            fail_relative_mae_max_above=0.002,
        )
        == []
    )
    failures = mod.threshold_failures(
        summary,
        fail_mae_mean_above=None,
        fail_last_token_mae_max_above=None,
        fail_relative_mae_max_above=0.0005,
    )
    assert len(failures) == 1 and "relative_mae_max" in failures[0]
    assert mod.threshold_failures(
        {"clips": 0},
        fail_mae_mean_above=None,
        fail_last_token_mae_max_above=None,
        fail_relative_mae_max_above=None,
    ) == ["Aligner encoder parity: no clips were compared"]


def test_compare_encoder_outputs_rejects_zero_reference():
    mod = _load_script_module()
    with pytest.raises(ValueError, match="all zeros"):
        mod.compare_encoder_outputs(np.ones((2, 2)), np.zeros((2, 2)))


def test_apply_reference_window_mask_supplies_mask_once_and_is_idempotent():
    """Exercise the real torch dispatch: Module.__call__ -> instance forward."""
    torch = pytest.importorskip("torch")
    mod = _load_script_module()
    calls: list[object] = []
    mask_builds: list[tuple] = []

    class _Layer(torch.nn.Module):
        def forward(self, hidden_states, cu_seqlens, attention_mask=None, **kwargs):  # noqa: ANN001
            calls.append(attention_mask)
            return (hidden_states,)

    class _Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([_Layer(), _Layer()])

        def _prepare_attention_mask(self, hidden_states, cu_seqlens):  # noqa: ANN001
            mask_builds.append(tuple(cu_seqlens.tolist()))
            return torch.zeros(1, 1, hidden_states.shape[0], hidden_states.shape[0])

    encoder = _Encoder()
    mod.apply_reference_window_mask(encoder)
    mod.apply_reference_window_mask(encoder)  # second call must not double-wrap

    hidden = torch.zeros(150, 4)
    cu = torch.tensor([0, 104, 150], dtype=torch.int32)
    for layer in encoder.layers:
        layer(hidden, cu)  # positional call, as Qwen3ASRAudioEncoder.forward does
    assert len(calls) == 2 and all(isinstance(m, torch.Tensor) for m in calls)
    assert calls[0] is calls[1], "mask should be built once per forward and shared"
    assert mask_builds == [(0, 104, 150)]

    # A new sequence rebuilds; an explicit mask passes through untouched.
    encoder.layers[0](torch.zeros(50, 4), torch.tensor([0, 50], dtype=torch.int32))
    assert mask_builds[-1] == (0, 50)
    encoder.layers[0](hidden, cu, attention_mask="explicit")
    assert calls[-1] == "explicit"
