"""Tests for mlx_qwen3_asr/transcribe.py."""

from __future__ import annotations

import asyncio
import importlib
import warnings
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx_qwen3_asr.config import DEFAULT_MODEL_ID
from mlx_qwen3_asr.forced_aligner import AlignedWord
from mlx_qwen3_asr.transcribe import _to_audio_np, transcribe, transcribe_async, transcribe_batch


class _DummyModel:
    audio_token_id = 151676
    config = SimpleNamespace(text_config=SimpleNamespace(vocab_size=151936))

    def audio_tower(self, mel: mx.array, feature_lens: mx.array):
        n = 2
        feats = mx.zeros((1, n, 2048), dtype=mx.float16)
        lens = mx.array([n], dtype=mx.int32)
        return feats, lens


class _DummyTokenizer:
    def __init__(self, model_path: str):
        self.model_path = model_path

    def build_prompt_tokens(
        self, n_audio_tokens: int, language: str | None = None, context: str = "",
    ) -> list[int]:
        return [1, 2, 3]

    def decode(self, ids: list[int]) -> str:
        return "language English<asr_text>hello world"


class _RecordingLanguageTokenizer(_DummyTokenizer):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.languages: list[str | None] = []
        self.contexts: list[str] = []

    def build_prompt_tokens(
        self, n_audio_tokens: int, language: str | None = None, context: str = "",
    ) -> list[int]:
        self.languages.append(language)
        self.contexts.append(context)
        return [1, 2, 3]


class _DummyTokenizerHolder:
    @staticmethod
    def get(model_path: str):
        return _DummyTokenizer(model_path)


class _DummyAligner:
    def align(self, audio: np.ndarray, text: str, language: str):
        return [AlignedWord(text="hello", start_time=0.1, end_time=0.4)]


def test_transcribe_basic(monkeypatch):
    tmod = importlib.import_module("mlx_qwen3_asr.transcribe")

    monkeypatch.setattr(tmod, "_TokenizerHolder", _DummyTokenizerHolder)
    monkeypatch.setattr(tmod._ModelHolder, "get", lambda *a, **k: (_DummyModel(), None))
    monkeypatch.setattr(
        tmod,
        "compute_features",
        lambda audio: (mx.zeros((1, 128, 100), dtype=mx.float32), mx.array([100], dtype=mx.int32)),
    )
    monkeypatch.setattr(tmod, "generate", lambda **kwargs: [10, 11, 12])

    result = transcribe(np.zeros(3200, dtype=np.float32))
    assert result.language == "English"
    assert result.text == "hello world"
    assert result.segments is None
    assert result.speaker_segments is None


def test_transcribe_with_timestamps(monkeypatch):
    tmod = importlib.import_module("mlx_qwen3_asr.transcribe")

    monkeypatch.setattr(tmod, "_TokenizerHolder", _DummyTokenizerHolder)
    monkeypatch.setattr(tmod._ModelHolder, "get", lambda *a, **k: (_DummyModel(), None))
    monkeypatch.setattr(
        tmod,
        "compute_features",
        lambda audio: (mx.zeros((1, 128, 100), dtype=mx.float32), mx.array([100], dtype=mx.int32)),
    )
    monkeypatch.setattr(tmod, "generate", lambda **kwargs: [10, 11, 12])
    monkeypatch.setattr(
        tmod,
        "split_audio_into_chunks",
        lambda audio, sr: [
            (np.zeros(16000, dtype=np.float32), 0.0),
            (np.zeros(16000, dtype=np.float32), 1.5),
        ],
    )
    monkeypatch.setattr(
        tmod,
        "parse_asr_output",
        lambda raw, user_language=None: ("English", "hello world"),
    )

    result = transcribe(
        np.zeros(32000, dtype=np.float32),
        return_timestamps=True,
        forced_aligner=_DummyAligner(),
    )

    assert result.language == "English"
    assert result.text == "hello world hello world"
    assert result.segments == [
        {"text": "hello", "start": 0.1, "end": 0.4},
        {"text": "hello", "start": 1.6, "end": 1.9},
    ]
    assert result.speaker_segments is None


def test_transcribe_diarize_returns_speaker_segments(monkeypatch):
    tmod = importlib.import_module("mlx_qwen3_asr.transcribe")

    monkeypatch.setattr(tmod, "_TokenizerHolder", _DummyTokenizerHolder)
    monkeypatch.setattr(tmod._ModelHolder, "get", lambda *a, **k: (_DummyModel(), None))
    monkeypatch.setattr(
        tmod._ModelHolder,
        "get_resolved_path",
        lambda path, dtype=mx.float16: path,
    )
    monkeypatch.setattr(
        tmod,
        "compute_features",
        lambda audio: (mx.zeros((1, 128, 100), dtype=mx.float32), mx.array([100], dtype=mx.int32)),
    )
    monkeypatch.setattr(tmod, "generate", lambda **kwargs: [10, 11, 12])
    monkeypatch.setattr(
        tmod,
        "parse_asr_output",
        lambda raw, user_language=None: ("English", "hello world"),
    )
    monkeypatch.setattr(
        tmod,
        "infer_speaker_turns",
        lambda audio, sr, config: [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": float(len(audio) / sr)}
        ],
    )

    result = transcribe(
        np.zeros(32000, dtype=np.float32),
        diarize=True,
        forced_aligner=_DummyAligner(),
    )

    assert result.language == "English"
    assert result.text == "hello world"
    assert result.segments is None
    assert result.speaker_segments is not None
    assert len(result.speaker_segments) >= 1
    assert result.speaker_segments[0]["speaker"] == "SPEAKER_00"


def test_transcribe_diarize_labels_word_segments_when_return_timestamps(monkeypatch):
    tmod = importlib.import_module("mlx_qwen3_asr.transcribe")

    monkeypatch.setattr(tmod, "_TokenizerHolder", _DummyTokenizerHolder)
    monkeypatch.setattr(tmod._ModelHolder, "get", lambda *a, **k: (_DummyModel(), None))
    monkeypatch.setattr(
        tmod._ModelHolder,
        "get_resolved_path",
        lambda path, dtype=mx.float16: path,
    )
    monkeypatch.setattr(
        tmod,
        "compute_features",
        lambda audio: (mx.zeros((1, 128, 100), dtype=mx.float32), mx.array([100], dtype=mx.int32)),
    )
    monkeypatch.setattr(tmod, "generate", lambda **kwargs: [10, 11, 12])
    monkeypatch.setattr(
        tmod,
        "parse_asr_output",
        lambda raw, user_language=None: ("English", "hello world"),
    )
    monkeypatch.setattr(
        tmod,
        "infer_speaker_turns",
        lambda audio, sr, config: [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": float(len(audio) / sr)}
        ],
    )

    result = transcribe(
        np.zeros(16000, dtype=np.float32),
        return_timestamps=True,
        diarize=True,
        forced_aligner=_DummyAligner(),
    )

    assert result.segments is not None
    assert result.segments[0]["speaker"] == "SPEAKER_00"
    assert result.speaker_segments is not None


def test_transcribe_diarize_speaker_segments_cover_turns_when_words_sparse(monkeypatch):
    tmod = importlib.import_module("mlx_qwen3_asr.transcribe")

    monkeypatch.setattr(tmod, "_TokenizerHolder", _DummyTokenizerHolder)
    monkeypatch.setattr(tmod._ModelHolder, "get", lambda *a, **k: (_DummyModel(), None))
    monkeypatch.setattr(
        tmod._ModelHolder,
        "get_resolved_path",
        lambda path, dtype=mx.float16: path,
    )
    monkeypatch.setattr(
        tmod,
        "compute_features",
        lambda audio: (mx.zeros((1, 128, 100), dtype=mx.float32), mx.array([100], dtype=mx.int32)),
    )
    monkeypatch.setattr(tmod, "generate", lambda **kwargs: [10, 11, 12])
    monkeypatch.setattr(
        tmod,
        "parse_asr_output",
        lambda raw, user_language=None: ("English", "hello world"),
    )
    monkeypatch.setattr(
        tmod,
        "infer_speaker_turns",
        lambda audio, sr, config: [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 0.5},
            {"speaker": "SPEAKER_01", "start": 0.5, "end": 1.0},
        ],
    )

    result = transcribe(
        np.zeros(16000, dtype=np.float32),
        diarize=True,
        forced_aligner=_DummyAligner(),
    )

    assert result.speaker_segments is not None
    assert len(result.speaker_segments) == 2
    assert result.speaker_segments[0]["speaker"] == "SPEAKER_00"
    assert result.speaker_segments[0]["text"] == "hello"
    assert result.speaker_segments[1]["speaker"] == "SPEAKER_01"
    assert result.speaker_segments[1]["text"] == ""


def test_transcribe_rejects_invalid_diarization_bounds():
    with pytest.raises(ValueError, match="diarization_max_speakers"):
        transcribe(
            np.zeros(3200, dtype=np.float32),
            diarize=True,
            diarization_min_speakers=3,
            diarization_max_speakers=2,
        )


def test_transcribe_joins_cjk_chunks_without_spaces(monkeypatch):
    tmod = importlib.import_module("mlx_qwen3_asr.transcribe")

    monkeypatch.setattr(tmod, "_TokenizerHolder", _DummyTokenizerHolder)
    monkeypatch.setattr(tmod._ModelHolder, "get", lambda *a, **k: (_DummyModel(), None))
    monkeypatch.setattr(
        tmod._ModelHolder,
        "get_resolved_path",
        lambda path, dtype=mx.float16: path,
    )
    monkeypatch.setattr(
        tmod,
        "compute_features",
        lambda audio: (mx.zeros((1, 128, 100), dtype=mx.float32), mx.array([100], dtype=mx.int32)),
    )
    monkeypatch.setattr(tmod, "generate", lambda **kwargs: [10, 11, 12])
    monkeypatch.setattr(
        tmod,
        "split_audio_into_chunks",
        lambda audio, sr: [
            (np.zeros(16000, dtype=np.float32), 0.0),
            (np.zeros(16000, dtype=np.float32), 1.0),
        ],
    )
    monkeypatch.setattr(
        tmod,
        "parse_asr_output",
        lambda raw, user_language=None: ("Chinese", "你好"),
    )

    result = transcribe(np.zeros(32000, dtype=np.float32))
    assert result.language == "Chinese"
    assert result.text == "你好你好"


def test_transcribe_uses_default_model_id(monkeypatch):
    tmod = importlib.import_module("mlx_qwen3_asr.transcribe")
    called_paths = []

    def _fake_get(path, **kwargs):  # noqa: ANN001
        called_paths.append(path)
        return _DummyModel(), None

    monkeypatch.setattr(tmod, "_TokenizerHolder", _DummyTokenizerHolder)
    monkeypatch.setattr(tmod._ModelHolder, "get", _fake_get)
    monkeypatch.setattr(
        tmod._ModelHolder,
        "get_resolved_path",
        lambda path, dtype=mx.float16: path,
    )
    monkeypatch.setattr(
        tmod,
        "compute_features",
        lambda audio: (mx.zeros((1, 128, 100), dtype=mx.float32), mx.array([100], dtype=mx.int32)),
    )
    monkeypatch.setattr(tmod, "generate", lambda **kwargs: [10, 11, 12])

    _ = transcribe(np.zeros(3200, dtype=np.float32))
    assert called_paths == [DEFAULT_MODEL_ID]


def test_transcribe_uses_resolved_model_path_for_tokenizer(monkeypatch):
    tmod = importlib.import_module("mlx_qwen3_asr.transcribe")
    token_paths = []

    class _RecordingTokenizerHolder:
        @staticmethod
        def get(model_path: str):
            token_paths.append(model_path)
            return _DummyTokenizer(model_path)

    monkeypatch.setattr(tmod, "_TokenizerHolder", _RecordingTokenizerHolder)
    monkeypatch.setattr(tmod._ModelHolder, "get", lambda *a, **k: (_DummyModel(), None))
    monkeypatch.setattr(
        tmod._ModelHolder,
        "get_resolved_path",
        lambda path, dtype=mx.float16: "/tmp/qwen3-resolved-model",
    )
    monkeypatch.setattr(
        tmod,
        "compute_features",
        lambda audio: (mx.zeros((1, 128, 100), dtype=mx.float32), mx.array([100], dtype=mx.int32)),
    )
    monkeypatch.setattr(tmod, "generate", lambda **kwargs: [10, 11, 12])

    _ = transcribe(np.zeros(3200, dtype=np.float32))
    assert token_paths == ["/tmp/qwen3-resolved-model"]


def test_transcribe_canonicalizes_forced_language(monkeypatch):
    tmod = importlib.import_module("mlx_qwen3_asr.transcribe")
    tokenizer = _RecordingLanguageTokenizer("repo/a")

    class _RecordingTokenizerHolder:
        @staticmethod
        def get(model_path: str):
            return tokenizer

    monkeypatch.setattr(tmod, "_TokenizerHolder", _RecordingTokenizerHolder)
    monkeypatch.setattr(tmod._ModelHolder, "get", lambda *a, **k: (_DummyModel(), None))
    monkeypatch.setattr(
        tmod._ModelHolder,
        "get_resolved_path",
        lambda path, dtype=mx.float16: path,
    )
    monkeypatch.setattr(
        tmod,
        "compute_features",
        lambda audio: (mx.zeros((1, 128, 100), dtype=mx.float32), mx.array([100], dtype=mx.int32)),
    )
    monkeypatch.setattr(tmod, "generate", lambda **kwargs: [10, 11, 12])
    monkeypatch.setattr(
        tmod,
        "parse_asr_output",
        lambda raw, user_language=None: (user_language or "unknown", "hallo welt"),
    )

    result = transcribe(np.zeros(3200, dtype=np.float32), language="de_de")

    assert tokenizer.languages == ["German"]
    assert result.language == "German"
    assert result.text == "hallo welt"


def test_transcribe_passes_context_to_tokenizer(monkeypatch):
    tmod = importlib.import_module("mlx_qwen3_asr.transcribe")
    tokenizer = _RecordingLanguageTokenizer("repo/a")

    class _RecordingTokenizerHolder:
        @staticmethod
        def get(model_path: str):
            return tokenizer

    monkeypatch.setattr(tmod, "_TokenizerHolder", _RecordingTokenizerHolder)
    monkeypatch.setattr(tmod._ModelHolder, "get", lambda *a, **k: (_DummyModel(), None))
    monkeypatch.setattr(
        tmod._ModelHolder,
        "get_resolved_path",
        lambda path, dtype=mx.float16: path,
    )
    monkeypatch.setattr(
        tmod,
        "compute_features",
        lambda audio: (mx.zeros((1, 128, 100), dtype=mx.float32), mx.array([100], dtype=mx.int32)),
    )
    monkeypatch.setattr(tmod, "generate", lambda **kwargs: [10, 11, 12])

    result = transcribe(np.zeros(3200, dtype=np.float32), context="交易 停滞")

    assert tokenizer.contexts == ["交易 停滞"]
    assert result.text == "hello world"


def test_transcribe_default_context_is_empty(monkeypatch):
    tmod = importlib.import_module("mlx_qwen3_asr.transcribe")
    tokenizer = _RecordingLanguageTokenizer("repo/a")

    class _RecordingTokenizerHolder:
        @staticmethod
        def get(model_path: str):
            return tokenizer

    monkeypatch.setattr(tmod, "_TokenizerHolder", _RecordingTokenizerHolder)
    monkeypatch.setattr(tmod._ModelHolder, "get", lambda *a, **k: (_DummyModel(), None))
    monkeypatch.setattr(
        tmod._ModelHolder,
        "get_resolved_path",
        lambda path, dtype=mx.float16: path,
    )
    monkeypatch.setattr(
        tmod,
        "compute_features",
        lambda audio: (mx.zeros((1, 128, 100), dtype=mx.float32), mx.array([100], dtype=mx.int32)),
    )
    monkeypatch.setattr(tmod, "generate", lambda **kwargs: [10, 11, 12])

    transcribe(np.zeros(3200, dtype=np.float32))

    assert tokenizer.contexts == [""]


def test_transcribe_batch_per_audio_context(monkeypatch):
    tmod = importlib.import_module("mlx_qwen3_asr.transcribe")
    tokenizer = _RecordingLanguageTokenizer("repo/a")

    class _RecordingTokenizerHolder:
        @staticmethod
        def get(model_path: str):
            return tokenizer

    monkeypatch.setattr(tmod, "_TokenizerHolder", _RecordingTokenizerHolder)
    monkeypatch.setattr(tmod._ModelHolder, "get", lambda *a, **k: (_DummyModel(), None))
    monkeypatch.setattr(
        tmod._ModelHolder,
        "get_resolved_path",
        lambda path, dtype=mx.float16: path,
    )
    monkeypatch.setattr(
        tmod,
        "compute_features",
        lambda audio: (mx.zeros((1, 128, 100), dtype=mx.float32), mx.array([100], dtype=mx.int32)),
    )
    monkeypatch.setattr(tmod, "generate", lambda **kwargs: [10, 11, 12])

    results = transcribe_batch(
        [np.zeros(3200, dtype=np.float32), np.zeros(3200, dtype=np.float32)],
        context=["ctx_a", "ctx_b"],
    )

    assert len(results) == 2
    assert tokenizer.contexts == ["ctx_a", "ctx_b"]


def test_transcribe_batch_broadcasts_single_context(monkeypatch):
    tmod = importlib.import_module("mlx_qwen3_asr.transcribe")
    tokenizer = _RecordingLanguageTokenizer("repo/a")

    class _RecordingTokenizerHolder:
        @staticmethod
        def get(model_path: str):
            return tokenizer

    monkeypatch.setattr(tmod, "_TokenizerHolder", _RecordingTokenizerHolder)
    monkeypatch.setattr(tmod._ModelHolder, "get", lambda *a, **k: (_DummyModel(), None))
    monkeypatch.setattr(
        tmod._ModelHolder,
        "get_resolved_path",
        lambda path, dtype=mx.float16: path,
    )
    monkeypatch.setattr(
        tmod,
        "compute_features",
        lambda audio: (mx.zeros((1, 128, 100), dtype=mx.float32), mx.array([100], dtype=mx.int32)),
    )
    monkeypatch.setattr(tmod, "generate", lambda **kwargs: [10, 11, 12])

    results = transcribe_batch(
        [np.zeros(3200, dtype=np.float32), np.zeros(3200, dtype=np.float32)],
        context="shared_ctx",
    )

    assert len(results) == 2
    assert tokenizer.contexts == ["shared_ctx", "shared_ctx"]


def test_transcribe_batch_context_length_mismatch():
    import pytest

    with pytest.raises(ValueError, match="context list length"):
        transcribe_batch(
            [np.zeros(3200, dtype=np.float32)],
            context=["a", "b"],
        )


def test_transcribe_uses_speculative_path_when_draft_model_is_set(monkeypatch):
    tmod = importlib.import_module("mlx_qwen3_asr.transcribe")

    get_calls = []

    def _fake_get(path, **kwargs):  # noqa: ANN001
        get_calls.append(path)
        return _DummyModel(), None

    monkeypatch.setattr(tmod, "_TokenizerHolder", _DummyTokenizerHolder)
    monkeypatch.setattr(tmod._ModelHolder, "get", _fake_get)
    monkeypatch.setattr(
        tmod._ModelHolder,
        "get_resolved_path",
        lambda path, dtype=mx.float16: path,
    )
    monkeypatch.setattr(
        tmod,
        "compute_features",
        lambda audio: (mx.zeros((1, 128, 100), dtype=mx.float32), mx.array([100], dtype=mx.int32)),
    )
    monkeypatch.setattr(tmod, "generate", lambda **kwargs: [10, 11, 12])

    calls = {"spec": 0}

    def _fake_speculative(**kwargs):  # noqa: ANN003
        calls["spec"] += 1
        return [10, 11, 12]

    monkeypatch.setattr(tmod, "generate_speculative", _fake_speculative)

    result = transcribe(
        np.zeros(3200, dtype=np.float32),
        model="Qwen/Qwen3-ASR-1.7B",
        draft_model="Qwen/Qwen3-ASR-0.6B",
    )

    assert result.language == "English"
    assert result.text == "hello world"
    assert calls["spec"] == 1
    assert get_calls == ["Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-0.6B"]


def test_to_audio_np_uses_canonical_downmix_for_numpy():
    stereo = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
    out = _to_audio_np(stereo)
    np.testing.assert_allclose(out, np.array([2.0, 6.0], dtype=np.float32), atol=1e-6)


def test_to_audio_np_normalizes_integer_pcm_numpy():
    pcm = np.array([16384, -16384], dtype=np.int16)
    out = _to_audio_np(pcm)
    np.testing.assert_allclose(out, np.array([0.5, -0.5], dtype=np.float32), atol=1e-6)


def test_to_audio_np_uses_numpy_loader_path(monkeypatch):
    tmod = importlib.import_module("mlx_qwen3_asr.transcribe")
    calls = []

    def _fake_load_audio_np(source, sr):  # noqa: ANN001
        calls.append((type(source).__name__, sr))
        return np.array([0.25, -0.25], dtype=np.float32)

    monkeypatch.setattr(tmod, "load_audio_np", _fake_load_audio_np)
    out = tmod._to_audio_np(np.array([1.0, -1.0], dtype=np.float32))

    np.testing.assert_allclose(out, np.array([0.25, -0.25], dtype=np.float32), atol=1e-6)
    assert calls == [("ndarray", 16000)]


def test_transcribe_warns_on_unknown_forced_language(monkeypatch):
    tmod = importlib.import_module("mlx_qwen3_asr.transcribe")

    monkeypatch.setattr(tmod, "_TokenizerHolder", _DummyTokenizerHolder)
    monkeypatch.setattr(tmod._ModelHolder, "get", lambda *a, **k: (_DummyModel(), None))
    monkeypatch.setattr(
        tmod._ModelHolder,
        "get_resolved_path",
        lambda path, dtype=mx.float16: path,
    )
    monkeypatch.setattr(
        tmod,
        "compute_features",
        lambda audio: (mx.zeros((1, 128, 100), dtype=mx.float32), mx.array([100], dtype=mx.int32)),
    )
    monkeypatch.setattr(tmod, "generate", lambda **kwargs: [10, 11, 12])

    with pytest.warns(UserWarning, match="Unsupported language"):
        result = transcribe(np.zeros(3200, dtype=np.float32), language="xx_unsupported")

    assert result.language == "xx_unsupported"


def test_transcribe_does_not_warn_when_language_supported_by_model_config(monkeypatch):
    tmod = importlib.import_module("mlx_qwen3_asr.transcribe")

    class _ModelWithSupport(_DummyModel):
        config = SimpleNamespace(
            text_config=SimpleNamespace(vocab_size=151936),
            support_languages=["xx-custom"],
        )

    monkeypatch.setattr(tmod, "_TokenizerHolder", _DummyTokenizerHolder)
    monkeypatch.setattr(tmod._ModelHolder, "get", lambda *a, **k: (_ModelWithSupport(), None))
    monkeypatch.setattr(
        tmod._ModelHolder,
        "get_resolved_path",
        lambda path, dtype=mx.float16: path,
    )
    monkeypatch.setattr(
        tmod,
        "compute_features",
        lambda audio: (mx.zeros((1, 128, 100), dtype=mx.float32), mx.array([100], dtype=mx.int32)),
    )
    monkeypatch.setattr(tmod, "generate", lambda **kwargs: [10, 11, 12])

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        result = transcribe(np.zeros(3200, dtype=np.float32), language="xx_custom")

    assert result.language == "xx_custom"
    assert not [w for w in records if issubclass(w.category, UserWarning)]


def test_transcribe_can_return_chunk_metadata(monkeypatch):
    tmod = importlib.import_module("mlx_qwen3_asr.transcribe")

    monkeypatch.setattr(tmod, "_TokenizerHolder", _DummyTokenizerHolder)
    monkeypatch.setattr(tmod._ModelHolder, "get", lambda *a, **k: (_DummyModel(), None))
    monkeypatch.setattr(
        tmod._ModelHolder,
        "get_resolved_path",
        lambda path, dtype=mx.float16: path,
    )
    monkeypatch.setattr(
        tmod,
        "compute_features",
        lambda audio: (mx.zeros((1, 128, 100), dtype=mx.float32), mx.array([100], dtype=mx.int32)),
    )
    monkeypatch.setattr(tmod, "generate", lambda **kwargs: [10, 11, 12])
    monkeypatch.setattr(
        tmod,
        "split_audio_into_chunks",
        lambda audio, sr: [
            (np.zeros(16000, dtype=np.float32), 0.0),
            (np.zeros(8000, dtype=np.float32), 1.0),
        ],
    )

    texts = iter([("english", "hello"), ("english", "world")])
    monkeypatch.setattr(tmod, "parse_asr_output", lambda raw, user_language=None: next(texts))

    result = transcribe(np.zeros(24000, dtype=np.float32), return_chunks=True)

    assert result.language == "English"
    assert result.text == "hello world"
    assert result.chunks is not None
    assert len(result.chunks) == 2
    assert result.chunks[0]["start"] == 0.0
    assert result.chunks[0]["end"] == 1.0
    assert result.chunks[1]["end"] == 1.5


def test_transcribe_progress_callback_emits_events(monkeypatch):
    tmod = importlib.import_module("mlx_qwen3_asr.transcribe")

    monkeypatch.setattr(tmod, "_TokenizerHolder", _DummyTokenizerHolder)
    monkeypatch.setattr(tmod._ModelHolder, "get", lambda *a, **k: (_DummyModel(), None))
    monkeypatch.setattr(
        tmod._ModelHolder,
        "get_resolved_path",
        lambda path, dtype=mx.float16: path,
    )
    monkeypatch.setattr(
        tmod,
        "compute_features",
        lambda audio: (mx.zeros((1, 128, 100), dtype=mx.float32), mx.array([100], dtype=mx.int32)),
    )
    monkeypatch.setattr(tmod, "generate", lambda **kwargs: [10, 11, 12])
    monkeypatch.setattr(
        tmod,
        "split_audio_into_chunks",
        lambda audio, sr: [
            (np.zeros(16000, dtype=np.float32), 0.0),
            (np.zeros(16000, dtype=np.float32), 1.0),
        ],
    )

    events: list[dict] = []
    result = transcribe(
        np.zeros(32000, dtype=np.float32),
        on_progress=events.append,
    )

    assert result.text == "hello world hello world"
    kinds = [event["event"] for event in events]
    assert "chunks_prepared" in kinds
    assert "chunk_started" in kinds
    assert "chunk_completed" in kinds
    assert kinds[-1] == "completed"
    assert events[-1]["progress"] == 1.0


def test_transcribe_diarize_progress_emits_diarization_event(monkeypatch):
    tmod = importlib.import_module("mlx_qwen3_asr.transcribe")

    monkeypatch.setattr(tmod, "_TokenizerHolder", _DummyTokenizerHolder)
    monkeypatch.setattr(tmod._ModelHolder, "get", lambda *a, **k: (_DummyModel(), None))
    monkeypatch.setattr(
        tmod._ModelHolder,
        "get_resolved_path",
        lambda path, dtype=mx.float16: path,
    )
    monkeypatch.setattr(
        tmod,
        "compute_features",
        lambda audio: (mx.zeros((1, 128, 100), dtype=mx.float32), mx.array([100], dtype=mx.int32)),
    )
    monkeypatch.setattr(tmod, "generate", lambda **kwargs: [10, 11, 12])
    monkeypatch.setattr(
        tmod,
        "infer_speaker_turns",
        lambda audio, sr, config: [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": float(len(audio) / sr)}
        ],
    )

    events: list[dict] = []
    _ = transcribe(
        np.zeros(3200, dtype=np.float32),
        diarize=True,
        forced_aligner=_DummyAligner(),
        on_progress=events.append,
    )

    kinds = [event["event"] for event in events]
    assert "diarization_completed" in kinds


def test_transcribe_batch_reuses_loaded_components(monkeypatch):
    tmod = importlib.import_module("mlx_qwen3_asr.transcribe")
    get_calls = []

    def _fake_get(path, **kwargs):  # noqa: ANN001
        get_calls.append(path)
        return _DummyModel(), None

    monkeypatch.setattr(tmod, "_TokenizerHolder", _DummyTokenizerHolder)
    monkeypatch.setattr(tmod._ModelHolder, "get", _fake_get)
    monkeypatch.setattr(
        tmod._ModelHolder,
        "get_resolved_path",
        lambda path, dtype=mx.float16: path,
    )
    monkeypatch.setattr(
        tmod,
        "compute_features",
        lambda audio: (mx.zeros((1, 128, 100), dtype=mx.float32), mx.array([100], dtype=mx.int32)),
    )
    monkeypatch.setattr(tmod, "generate", lambda **kwargs: [10, 11, 12])

    outs = transcribe_batch(
        [np.zeros(3200, dtype=np.float32), np.zeros(3200, dtype=np.float32)],
        model="Qwen/Qwen3-ASR-0.6B",
    )

    assert len(outs) == 2
    assert outs[0].text == "hello world"
    assert outs[1].text == "hello world"
    assert get_calls == ["Qwen/Qwen3-ASR-0.6B"]


def test_transcribe_async_wrapper(monkeypatch):
    tmod = importlib.import_module("mlx_qwen3_asr.transcribe")

    monkeypatch.setattr(tmod, "_TokenizerHolder", _DummyTokenizerHolder)
    monkeypatch.setattr(tmod._ModelHolder, "get", lambda *a, **k: (_DummyModel(), None))
    monkeypatch.setattr(
        tmod._ModelHolder,
        "get_resolved_path",
        lambda path, dtype=mx.float16: path,
    )
    monkeypatch.setattr(
        tmod,
        "compute_features",
        lambda audio: (mx.zeros((1, 128, 100), dtype=mx.float32), mx.array([100], dtype=mx.int32)),
    )
    monkeypatch.setattr(tmod, "generate", lambda **kwargs: [10, 11, 12])

    result = asyncio.run(transcribe_async(np.zeros(3200, dtype=np.float32)))
    assert result.text == "hello world"
