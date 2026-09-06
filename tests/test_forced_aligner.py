"""Tests for forced aligner text/timestamp processing utilities."""

from __future__ import annotations

import builtins
import random
import sys
import types

import numpy as np
import pytest

import mlx_qwen3_asr.forced_aligner as famod
from mlx_qwen3_asr.forced_aligner import ForcedAlignTextProcessor


@pytest.fixture(autouse=True)
def _reset_text_processor_caches():
    ForcedAlignTextProcessor._nagisa = None
    ForcedAlignTextProcessor._ko_tokenizer = None
    ForcedAlignTextProcessor._ko_tokenizer_error = None
    yield
    ForcedAlignTextProcessor._nagisa = None
    ForcedAlignTextProcessor._ko_tokenizer = None
    ForcedAlignTextProcessor._ko_tokenizer_error = None


def test_clean_token_keeps_letters_digits_apostrophe():
    assert ForcedAlignTextProcessor.clean_token("can't!!! 123") == "can't123"


def test_tokenize_space_lang_splits_mixed_cjk():
    text = "hello 世界! test"
    assert ForcedAlignTextProcessor.tokenize_space_lang(text) == [
        "hello",
        "世",
        "界",
        "test",
    ]


def test_tokenize_text_japanese_uses_nagisa_when_available(monkeypatch):
    class _Tagged:
        words = ["こんにちは", "!", "世界"]

    nagisa_mod = types.ModuleType("nagisa")
    nagisa_mod.tagging = lambda text: _Tagged()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nagisa", nagisa_mod)

    tokens = ForcedAlignTextProcessor.tokenize_text("ignored", "ja")
    assert tokens == ["こんにちは", "世界"]


def test_tokenize_text_japanese_missing_dep_raises_clear_error(monkeypatch):
    monkeypatch.delitem(sys.modules, "nagisa", raising=False)
    original_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "nagisa":
            raise ImportError("nagisa not available")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(RuntimeError, match="Japanese tokenization requires optional dependency"):
        ForcedAlignTextProcessor.tokenize_text("こんにちは", "japanese")


def test_tokenize_text_korean_uses_soynlp_and_caches(monkeypatch):
    state = {"ctor_calls": 0, "scores_size": 0}

    class _DummyLTokenizer:
        def __init__(self, scores):
            state["ctor_calls"] += 1
            state["scores_size"] = len(scores)

        def tokenize(self, text):
            return ["안녕", "하세요!"]

    soynlp_mod = types.ModuleType("soynlp")
    soynlp_tokenizer_mod = types.ModuleType("soynlp.tokenizer")
    soynlp_tokenizer_mod.LTokenizer = _DummyLTokenizer  # type: ignore[attr-defined]
    soynlp_mod.tokenizer = soynlp_tokenizer_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "soynlp", soynlp_mod)
    monkeypatch.setitem(sys.modules, "soynlp.tokenizer", soynlp_tokenizer_mod)

    tokens1 = ForcedAlignTextProcessor.tokenize_text("안녕하세요", "ko")
    tokens2 = ForcedAlignTextProcessor.tokenize_text("안녕하세요", "korean")

    assert tokens1 == ["안녕", "하세요"]
    assert tokens2 == ["안녕", "하세요"]
    assert state["ctor_calls"] == 1
    assert state["scores_size"] > 0


def test_tokenize_text_korean_missing_dep_raises_clear_error(monkeypatch):
    monkeypatch.delitem(sys.modules, "soynlp", raising=False)
    monkeypatch.delitem(sys.modules, "soynlp.tokenizer", raising=False)
    original_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name.startswith("soynlp"):
            raise ImportError("soynlp not available")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(RuntimeError, match="Korean tokenization requires optional dependency"):
        ForcedAlignTextProcessor.tokenize_text("안녕하세요", "korean")


def test_fix_timestamp_repairs_non_monotonic_sequence():
    raw = np.array([0, 80, 160, 120, 240, 320], dtype=np.int32)
    fixed = ForcedAlignTextProcessor.fix_timestamp(raw)
    assert len(fixed) == len(raw)
    assert all(fixed[i] <= fixed[i + 1] for i in range(len(fixed) - 1))


def _lis_non_decreasing_indices_reference(arr: list[float]) -> list[int]:
    """Legacy O(n^2) DP reference used for behavioral parity checks."""
    n = len(arr)
    if n == 0:
        return []
    dp = [1] * n
    parent = [-1] * n
    for i in range(n):
        for j in range(i):
            if arr[j] <= arr[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
    max_len = max(dp)
    end_idx = dp.index(max_len)
    out: list[int] = []
    cur = end_idx
    while cur != -1:
        out.append(cur)
        cur = parent[cur]
    out.reverse()
    return out


def test_lis_non_decreasing_indices_matches_legacy_reference():
    rng = random.Random(20260214)
    for n in range(1, 40):
        for _ in range(150):
            arr = [float(rng.randint(-40, 40)) for _ in range(n)]
            got = ForcedAlignTextProcessor._lis_non_decreasing_indices(arr)
            want = _lis_non_decreasing_indices_reference(arr)
            assert got == want


def test_fix_timestamp_preserves_duplicate_non_decreasing_lis_behavior():
    raw = np.array([0, 10, 10, 9, 11], dtype=np.int32)
    fixed = ForcedAlignTextProcessor.fix_timestamp(raw)
    assert fixed == [0, 10, 10, 10, 11]


def test_parse_timestamp_ms_pairs_words():
    words = ["hello", "world"]
    timestamp_ms = np.array([0, 120, 130, 300], dtype=np.int32)
    aligned = ForcedAlignTextProcessor.parse_timestamp_ms(words, timestamp_ms)
    assert len(aligned) == 2
    assert aligned[0].text == "hello"
    assert aligned[0].start_time == 0.0
    assert aligned[0].end_time == 0.12
    assert aligned[1].text == "world"
    assert aligned[1].start_time == 0.13
    assert aligned[1].end_time == 0.3


def test_forced_aligner_rejects_invalid_backend():
    aligner = famod.ForcedAligner(backend="invalid")
    with pytest.raises(RuntimeError, match="Unsupported aligner backend"):
        aligner._ensure_loaded()


def test_forced_aligner_rejects_legacy_qwen_backend():
    aligner = famod.ForcedAligner(backend="qwen_asr")
    with pytest.raises(RuntimeError, match="Expected: mlx"):
        aligner._ensure_loaded()


def test_forced_aligner_default_backend_is_native_mlx():
    aligner = famod.ForcedAligner()
    assert aligner.backend == famod.ALIGNER_BACKEND_MLX


def test_forced_aligner_mlx_backend_reports_load_failure(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("mlx init failed")

    monkeypatch.setattr(famod, "_MLXForcedAlignerBackend", _boom)
    aligner = famod.ForcedAligner(backend=famod.ALIGNER_BACKEND_MLX)
    with pytest.raises(RuntimeError, match="Failed to load MLX aligner backend"):
        aligner._ensure_loaded()
