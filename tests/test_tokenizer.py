"""Tests for mlx_qwen3_asr/tokenizer.py.

Tokenizer tests use a tiny local vocab/merges fixture.
"""

import json

import mlx_qwen3_asr.tokenizer as tokmod
from mlx_qwen3_asr.tokenizer import (
    _TokenizerHolder,
    canonicalize_language,
    known_language_aliases,
    parse_asr_output,
)


class TestParseASROutputStandard:
    """Test parse_asr_output() with standard format."""

    def test_english(self):
        text = "language English<asr_text>hello world"
        lang, transcript = parse_asr_output(text)
        assert lang == "English"
        assert transcript == "hello world"

    def test_chinese(self):
        text = "language Chinese<asr_text>你好世界"
        lang, transcript = parse_asr_output(text)
        assert lang == "Chinese"
        assert transcript == "你好世界"

    def test_japanese(self):
        text = "language Japanese<asr_text>こんにちは"
        lang, transcript = parse_asr_output(text)
        assert lang == "Japanese"
        assert transcript == "こんにちは"


class TestParseASROutputSpecialTokens:
    """Test parse_asr_output() cleans trailing special tokens."""

    def test_cleans_im_end(self):
        text = "language English<asr_text>hello world<|im_end|>"
        lang, transcript = parse_asr_output(text)
        assert transcript == "hello world"

    def test_cleans_endoftext(self):
        text = "language English<asr_text>hello world<|endoftext|>"
        lang, transcript = parse_asr_output(text)
        assert transcript == "hello world"

    def test_cleans_multiple_special_tokens(self):
        text = "language English<asr_text>hello world<|im_end|><|endoftext|>"
        lang, transcript = parse_asr_output(text)
        assert transcript == "hello world"

    def test_does_not_strip_non_trailing_special_tokens(self):
        text = "language English<asr_text>hello<|im_end|>world"
        lang, transcript = parse_asr_output(text)
        assert transcript == "hello<|im_end|>world"


class TestParseASROutputFallback:
    """Test parse_asr_output() fallback with no <asr_text> marker."""

    def test_no_marker(self):
        text = "just some text without marker"
        lang, transcript = parse_asr_output(text)
        assert lang == "unknown"
        assert transcript == "just some text without marker"

    def test_empty_string(self):
        lang, transcript = parse_asr_output("")
        assert lang == "unknown"
        assert transcript == ""

    def test_whitespace_only(self):
        lang, transcript = parse_asr_output("   ")
        assert lang == "unknown"
        assert transcript == ""


class TestParseASROutputEdgeCases:
    """Test parse_asr_output() edge cases."""

    def test_empty_transcription(self):
        text = "language English<asr_text>"
        lang, transcript = parse_asr_output(text)
        assert lang == "English"
        assert transcript == ""

    def test_no_language_prefix(self):
        """If the text before <asr_text> doesn't start with 'language '."""
        text = "English<asr_text>hello"
        lang, transcript = parse_asr_output(text)
        assert lang == "English"
        assert transcript == "hello"

    def test_whitespace_in_transcript(self):
        text = "language English<asr_text>  hello world  "
        lang, transcript = parse_asr_output(text)
        assert transcript == "hello world"

    def test_language_none_empty_audio(self):
        text = "language None<asr_text>"
        lang, transcript = parse_asr_output(text)
        assert lang == ""
        assert transcript == ""

    def test_forced_language_treats_output_as_plain_text(self):
        text = "hello world<|im_end|>"
        lang, transcript = parse_asr_output(text, user_language="English")
        assert lang == "English"
        assert transcript == "hello world"

    def test_forced_language_canonicalizes_common_aliases(self):
        text = "hello world<|im_end|>"
        lang, transcript = parse_asr_output(text, user_language="de_de")
        assert lang == "German"
        assert transcript == "hello world"

    def test_forced_language_strips_asr_prefix_if_present(self):
        text = "language German<asr_text>hallo welt<|im_end|>"
        lang, transcript = parse_asr_output(text, user_language="de")
        assert lang == "German"
        assert transcript == "hallo welt"

    def test_char_repetition_cleanup(self):
        text = "language English<asr_text>" + ("a" * 30)
        lang, transcript = parse_asr_output(text)
        assert lang == "English"
        assert transcript == "a"

    def test_pattern_repetition_cleanup(self):
        text = "language English<asr_text>" + ("ab" * 25)
        lang, transcript = parse_asr_output(text)
        assert lang == "English"
        assert transcript == "ab"


def test_tokenizer_holder_caches_by_model_path(monkeypatch):
    created = []

    class _DummyTokenizer:
        def __init__(self, model_path: str):
            created.append(model_path)
            self.model_path = model_path

    monkeypatch.setattr(tokmod, "Tokenizer", _DummyTokenizer)
    _TokenizerHolder.clear()

    t1 = _TokenizerHolder.get("repo/a")
    t2 = _TokenizerHolder.get("repo/a")
    t3 = _TokenizerHolder.get("repo/b")

    assert t1 is t2
    assert t1 is not t3
    assert created == ["repo/a", "repo/b"]


def test_tokenizer_holder_lru_eviction(monkeypatch):
    created = []

    class _DummyTokenizer:
        def __init__(self, model_path: str):
            created.append(model_path)
            self.model_path = model_path

    monkeypatch.setattr(tokmod, "Tokenizer", _DummyTokenizer)
    _TokenizerHolder.clear()
    _TokenizerHolder.set_cache_capacity(1)
    try:
        t1 = _TokenizerHolder.get("repo/a")
        _ = _TokenizerHolder.get("repo/b")
        t2 = _TokenizerHolder.get("repo/a")

        assert t1 is not t2
        assert created == ["repo/a", "repo/b", "repo/a"]
    finally:
        _TokenizerHolder.set_cache_capacity(8)
        _TokenizerHolder.clear()


def test_canonicalize_language_handles_codes_and_names():
    assert canonicalize_language("de") == "German"
    assert canonicalize_language("fr_fr") == "French"
    assert canonicalize_language("English") == "English"
    assert canonicalize_language("xx") == "xx"


def test_known_language_aliases_contains_canonical_and_code_forms():
    aliases = known_language_aliases()
    assert "English" in aliases
    assert "en" in aliases["English"]
    assert "english" in aliases["English"]


def test_parse_asr_output_canonicalizes_detected_language():
    text = "language english<asr_text>hello world"
    lang, transcript = parse_asr_output(text)
    assert lang == "English"
    assert transcript == "hello world"


def _write_min_tokenizer_files(tmp_path):
    vocab = {
        "h": 0,
        "e": 1,
        "l": 2,
        "o": 3,
        "w": 4,
        "r": 5,
        "d": 6,
        "!": 7,
        "Ġ": 8,
        "Ċ": 9,
        "he": 10,
        "hel": 11,
        "hell": 12,
        "hello": 13,
        "wo": 14,
        "wor": 15,
        "worl": 16,
        "world": 17,
        "Ġh": 18,
        "Ġhe": 19,
        "Ġhel": 20,
        "Ġhell": 21,
        "Ġhello": 22,
        "Ġw": 23,
        "Ġwo": 24,
        "Ġwor": 25,
        "Ġworl": 26,
        "Ġworld": 27,
    }
    merges = "\n".join(
        [
            "#version: 0.2",
            "h e",
            "he l",
            "hel l",
            "hell o",
            "w o",
            "wo r",
            "wor l",
            "worl d",
            "Ġ h",
            "Ġh e",
            "Ġhe l",
            "Ġhel l",
            "Ġhell o",
            "Ġ w",
            "Ġw o",
            "Ġwo r",
            "Ġwor l",
            "Ġworl d",
        ]
    )
    tok_cfg = {
        "errors": "replace",
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        "added_tokens_decoder": {
            "151643": {"content": "<|endoftext|>", "special": True},
            "151644": {"content": "<|im_start|>", "special": True},
            "151645": {"content": "<|im_end|>", "special": True},
            "151669": {"content": "<|audio_start|>", "special": True},
            "151670": {"content": "<|audio_end|>", "special": True},
            "151676": {"content": "<|audio_pad|>", "special": True},
            "151704": {"content": "<asr_text>", "special": False},
        },
    }
    (tmp_path / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (tmp_path / "merges.txt").write_text(merges, encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(tok_cfg), encoding="utf-8")
    return str(tmp_path)


def test_native_tokenizer_encode_decode_added_tokens(tmp_path):
    model_dir = _write_min_tokenizer_files(tmp_path)
    tok = tokmod.Tokenizer(model_dir)

    text = "<|audio_start|><|audio_pad|><|audio_end|> hello world<asr_text>"
    ids = tok.encode(text)

    assert ids[:3] == [151669, 151676, 151670]
    assert ids[-1] == 151704
    assert tok.decode(ids) == text


def test_native_tokenizer_skip_special_tokens(tmp_path):
    model_dir = _write_min_tokenizer_files(tmp_path)
    tok = tokmod.Tokenizer(model_dir)

    ids = [151669, 151676, 151670, 151704]
    assert tok.decode(ids, skip_special_tokens=True) == "<asr_text>"


class TestBuildPromptTokensContext:
    """Test that context parameter flows into the system message."""

    def test_default_context_produces_empty_system_content(self, tmp_path):
        model_dir = _write_min_tokenizer_files(tmp_path)
        tok = tokmod.Tokenizer(model_dir)
        tokens = tok.build_prompt_tokens(n_audio_tokens=2)
        # System message: <|im_start|> + "system\n" + <|im_end|>
        assert tokens[0] == 151644  # <|im_start|>
        # The next tokens should be the system prefix immediately followed by
        # <|im_end|> (no content tokens between them).
        im_end_idx = tokens.index(151645)
        system_prefix = tok.encode("system\n")
        assert tokens[1 : 1 + len(system_prefix)] == system_prefix
        # With empty context, <|im_end|> follows right after system prefix.
        assert im_end_idx == 1 + len(system_prefix)

    def test_custom_context_injects_tokens(self, tmp_path):
        model_dir = _write_min_tokenizer_files(tmp_path)
        tok = tokmod.Tokenizer(model_dir)
        tokens_empty = tok.build_prompt_tokens(n_audio_tokens=2, context="")
        tokens_ctx = tok.build_prompt_tokens(n_audio_tokens=2, context="hello")
        # Custom context should produce more tokens in the system message.
        assert len(tokens_ctx) > len(tokens_empty)
        # The context tokens should appear between system prefix and <|im_end|>.
        ctx_encoded = tok.encode("hello")
        system_prefix = tok.encode("system\n")
        start = 1 + len(system_prefix)
        assert tokens_ctx[start : start + len(ctx_encoded)] == ctx_encoded

    def test_followup_accepts_context_kwarg(self, tmp_path):
        model_dir = _write_min_tokenizer_files(tmp_path)
        tok = tokmod.Tokenizer(model_dir)
        # Should not raise — context is accepted but not used in follow-up.
        tokens = tok.build_followup_prompt_tokens(
            n_audio_tokens=2, context="hello"
        )
        assert isinstance(tokens, list)

    def test_audio_tokens_present_with_context(self, tmp_path):
        model_dir = _write_min_tokenizer_files(tmp_path)
        tok = tokmod.Tokenizer(model_dir)
        tokens = tok.build_prompt_tokens(n_audio_tokens=3, context="hello world")
        assert tokens.count(151676) == 3  # <|audio_pad|> count
        assert 151669 in tokens  # <|audio_start|>
        assert 151670 in tokens  # <|audio_end|>
