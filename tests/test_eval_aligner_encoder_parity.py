"""Unit tests for scripts/eval_aligner_encoder_parity.py pure helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_apply_reference_window_mask_supplies_mask_and_is_idempotent():
    mod = _load_script_module()
    calls: list[object] = []

    class _Layer:
        def forward(self, hidden_states, cu_seqlens, attention_mask=None, **kwargs):  # noqa: ANN001
            calls.append(attention_mask)
            return (hidden_states,)

    encoder = SimpleNamespace(
        layers=[_Layer(), _Layer()],
        _prepare_attention_mask=lambda hidden_states, cu_seqlens: ("mask", tuple(cu_seqlens)),
    )
    mod.apply_reference_window_mask(encoder)
    mod.apply_reference_window_mask(encoder)  # second call must not double-wrap

    for layer in encoder.layers:
        layer.forward("h", [0, 104, 150])
    assert calls == [("mask", (0, 104, 150))] * 2

    # An explicit mask passes through untouched.
    encoder.layers[0].forward("h", [0, 5], attention_mask="explicit")
    assert calls[-1] == "explicit"
