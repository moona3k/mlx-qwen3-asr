"""Tests for ASR evaluation metrics."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_metrics_module():
    path = Path("scripts/eval/metrics.py")
    spec = importlib.util.spec_from_file_location("eval_metrics_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_METRICS = _load_metrics_module()
compute_cer = _METRICS.compute_cer
compute_wer = _METRICS.compute_wer
edit_distance = _METRICS.edit_distance
normalize_text = _METRICS.normalize_text


def test_normalize_text():
    assert normalize_text("Hello,  WORLD!!") == "hello world"
    assert normalize_text("Don't-stop") == "don't stop"


def test_edit_distance():
    assert edit_distance(["a", "b", "c"], ["a", "x", "c"]) == 1
    assert edit_distance([], ["x"]) == 1
    assert edit_distance(["x"], []) == 1


def test_compute_wer_and_cer():
    refs = ["hello world", "fast speech"]
    hyps = ["hello wurld", "fast"]

    wer = compute_wer(refs, hyps)
    cer = compute_cer(refs, hyps)

    # WER: 2 substitutions/deletions over 4 words.
    assert wer == 0.5
    assert 0.0 < cer < 1.0


def test_normalize_quality_text_multilingual():
    norm = _METRICS.normalize_quality_text
    assert norm("Héllo, Wörld’s!  ") == "héllo wörld's"
    assert norm("你好，世界。") == "你好 世界"
    assert norm("") == ""


def test_wer_tokens_falls_back_to_characters_without_whitespace():
    assert _METRICS.wer_tokens("你好世界") == ["你", "好", "世", "界"]
    assert _METRICS.wer_tokens("a b") == ["a", "b"]
    assert _METRICS.cer_tokens("a b") == ["a", "b"]


def test_is_char_primary_language():
    is_char = _METRICS.is_char_primary_language
    assert is_char("Chinese") and is_char("ja_jp") and is_char("ko-KR") and is_char("cmn_hans_cn")
    assert not is_char("English") and not is_char(None) and not is_char("")


def test_score_hypothesis_primary_metric_by_language():
    score = _METRICS.score_hypothesis
    en = score("hello world again", "hello world", "English")
    assert en["primary_metric"] == "wer"
    assert (en["primary_errors"], en["primary_denominator"]) == (1, 3)
    zh = score("你好世界", "你好世", "Chinese")
    assert zh["primary_metric"] == "cer"
    assert (zh["primary_errors"], zh["primary_denominator"]) == (1, 4)
    assert zh["wer_errors"] == 1  # character fallback tokens
