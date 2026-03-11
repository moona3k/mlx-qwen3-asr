"""Tests for mlx_qwen3_asr/streaming.py."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np

import mlx_qwen3_asr.streaming as smod
from mlx_qwen3_asr.config import DEFAULT_MODEL_ID
from mlx_qwen3_asr.streaming import (
    UNFIXED_TOKEN_NUM,
    StreamingState,
    _append_chunk_text,
    _split_stable_unstable,
    feed_audio,
    finish_streaming,
    init_streaming,
    streaming_metrics,
)


class TestInitStreaming:
    def test_default_state(self):
        state = init_streaming()
        assert isinstance(state, StreamingState)
        assert state.unfixed_chunk_num == 2
        assert state.unfixed_token_num == 5
        assert state.chunk_size_samples == 32000
        assert state.max_context_samples == 480000
        assert state._model_path == DEFAULT_MODEL_ID
        assert state.text == ""
        assert state.language == "unknown"
        assert state.chunk_id == 0
        assert len(state.buffer) == 0
        assert len(state.audio_accum) == 0
        assert state.stable_text == ""

    def test_custom_chunk_size(self):
        state = init_streaming(chunk_size_sec=5.0)
        assert state.chunk_size_samples == 80000

    def test_custom_sample_rate(self):
        state = init_streaming(chunk_size_sec=2.0, sample_rate=8000)
        assert state.chunk_size_samples == 16000

    def test_custom_context_window(self):
        state = init_streaming(max_context_sec=5.0, sample_rate=8000)
        assert state.max_context_samples == 40000

    def test_custom_unfixed_controls(self):
        state = init_streaming(unfixed_chunk_num=3, unfixed_token_num=7)
        assert state.unfixed_chunk_num == 3
        assert state.unfixed_token_num == 7

    def test_custom_model(self):
        state = init_streaming(model="my/custom-model")
        assert state._model_path == "my/custom-model"

    def test_invalid_chunk_size_raises(self):
        with np.testing.assert_raises(ValueError):
            init_streaming(chunk_size_sec=0.0)

    def test_invalid_context_window_raises(self):
        with np.testing.assert_raises(ValueError):
            init_streaming(max_context_sec=0.0)

    def test_invalid_sample_rate_raises(self):
        with np.testing.assert_raises(ValueError):
            init_streaming(sample_rate=0)

    def test_context_must_be_at_least_chunk(self):
        with np.testing.assert_raises(ValueError):
            init_streaming(chunk_size_sec=2.0, max_context_sec=1.0)

    def test_invalid_max_new_tokens_raises(self):
        with np.testing.assert_raises(ValueError):
            init_streaming(max_new_tokens=0)

    def test_tail_refine_flag(self):
        state = init_streaming(enable_tail_refine=False)
        assert state.enable_tail_refine is False
        assert state.finalization_mode == "latency"

    def test_finalization_mode_accuracy(self):
        state = init_streaming(finalization_mode="accuracy")
        assert state.finalization_mode == "accuracy"
        assert state.enable_tail_refine is True

    def test_finalization_mode_latency(self):
        state = init_streaming(finalization_mode="latency")
        assert state.finalization_mode == "latency"
        assert state.enable_tail_refine is False

    def test_invalid_finalization_mode_raises(self):
        with np.testing.assert_raises(ValueError):
            init_streaming(finalization_mode="fast")

    def test_invalid_endpointing_mode_raises(self):
        with np.testing.assert_raises(ValueError):
            init_streaming(endpointing_mode="vad")


class TestStreamingStateDefaults:
    def test_default_buffer(self):
        state = StreamingState()
        assert isinstance(state.buffer, np.ndarray)
        assert len(state.buffer) == 0
        assert state.buffer.dtype == np.float32

    def test_default_audio_accum(self):
        state = StreamingState()
        assert isinstance(state.audio_accum, np.ndarray)
        assert len(state.audio_accum) == 0

    def test_default_text(self):
        assert StreamingState().text == ""

    def test_default_language(self):
        assert StreamingState().language == "unknown"

    def test_default_chunk_id(self):
        assert StreamingState().chunk_id == 0

    def test_default_unfixed_controls(self):
        state = StreamingState()
        assert state.unfixed_chunk_num == 2
        assert state.unfixed_token_num == 5

    def test_default_chunk_size_samples(self):
        assert StreamingState().chunk_size_samples == 32000

    def test_default_max_context_samples(self):
        assert StreamingState().max_context_samples == 480000

    def test_default_stable_text(self):
        assert StreamingState().stable_text == ""


class TestSplitStableUnstable:
    def test_short_text_all_unstable(self):
        stable, unstable = _split_stable_unstable("", "hello world")
        assert stable == ""
        assert unstable == "hello world"

    def test_long_text_splits_correctly(self):
        text = "one two three four five six seven eight"
        stable, unstable = _split_stable_unstable("", text)
        assert stable == "one two three"
        assert unstable == "four five six seven eight"

    def test_preserves_previous_stable_text(self):
        prev_stable = "this is already stable text that is long"
        stable, unstable = _split_stable_unstable(prev_stable, "hello world")
        assert stable == prev_stable
        assert unstable == "hello world"

    def test_empty_text(self):
        stable, unstable = _split_stable_unstable("", "")
        assert stable == ""
        assert unstable == ""

    def test_exactly_unfixed_tokens(self):
        words = ["word"] * UNFIXED_TOKEN_NUM
        text = " ".join(words)
        stable, unstable = _split_stable_unstable("", text)
        assert stable == ""
        assert unstable == text

    def test_one_more_than_unfixed(self):
        words = [f"word{i}" for i in range(UNFIXED_TOKEN_NUM + 1)]
        text = " ".join(words)
        stable, unstable = _split_stable_unstable("", text)
        assert stable == words[0]
        assert unstable == " ".join(words[1:])

    def test_custom_unfixed_tokens(self):
        stable, unstable = _split_stable_unstable("", "a b c d e f g h", unfixed_tokens=3)
        assert stable == "a b c d e"
        assert unstable == "f g h"

    def test_cjk_without_spaces_splits_by_character(self):
        text = "こんにちは世界ありがとう"
        stable, unstable = _split_stable_unstable("", text, unfixed_tokens=3)
        assert stable == text[:-3]
        assert unstable == text[-3:]

    def test_stable_grows_monotonically(self):
        prev_stable = "one two three"
        text = "one two three four five six seven eight nine ten"
        stable, _ = _split_stable_unstable(prev_stable, text)
        assert len(stable) >= len(prev_stable)


class TestAppendChunkText:
    def test_exact_repeat_is_deduped(self):
        assert _append_chunk_text("a b c", "a b c", "English") == "a b c"

    def test_suffix_prefix_overlap_appends_only_tail(self):
        assert _append_chunk_text("a b c", "b c d", "English") == "a b c d"

    def test_cjk_overlap_appends_without_spaces(self):
        assert _append_chunk_text("你好世界", "世界和平", "Chinese") == "你好世界和平"

    def test_rewrite_superset_prefers_replacement(self):
        cur = "The quick brown fox jumps over the lazy."
        new = "The quick brown fox jumps over the lazy dog."
        assert _append_chunk_text(cur, new, "English") == new


class TestFeedAudio:
    def test_feed_audio_reuses_same_state_object(self, monkeypatch):
        calls = []

        def fake_decode(audio, state, model=None):  # noqa: ANN001
            calls.append((len(audio), model))
            return "hello world", "English"

        monkeypatch.setattr(smod, "_decode_chunk_incremental", fake_decode)

        state = init_streaming(chunk_size_sec=1.0, sample_rate=10)
        out = feed_audio(np.ones(10, dtype=np.float32), state)
        assert out is state
        assert state.chunk_id == 1
        assert calls[0][0] == 10

    def test_feed_audio_decodes_only_new_chunk(self, monkeypatch):
        call_lengths = []

        def fake_decode(audio, state, model=None):  # noqa: ANN001
            call_lengths.append(len(audio))
            return "a b c d e f g", "English"

        monkeypatch.setattr(smod, "_decode_chunk_incremental", fake_decode)

        state = init_streaming(chunk_size_sec=1.0, max_context_sec=2.0, sample_rate=10)
        feed_audio(np.ones(10, dtype=np.float32), state)
        feed_audio(np.ones(10, dtype=np.float32), state)
        feed_audio(np.ones(10, dtype=np.float32), state)

        assert call_lengths == [10, 10, 10]

    def test_feed_audio_energy_endpointing_prefers_silence_boundary(self, monkeypatch):
        call_lengths = []

        def fake_decode(audio, state, model=None):  # noqa: ANN001
            call_lengths.append(len(audio))
            return "ok", "English"

        monkeypatch.setattr(smod, "_decode_chunk_incremental", fake_decode)

        state = init_streaming(
            chunk_size_sec=1.0,
            sample_rate=10,
            endpointing_mode="energy",
            endpoint_lookback_sec=0.5,
            endpoint_frame_ms=100.0,
            endpoint_min_chunk_sec=0.2,
        )
        audio = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=np.float32)
        feed_audio(audio, state)

        assert call_lengths == [9]
        assert len(state.buffer) == 1

    def test_feed_audio_energy_endpointing_falls_back_without_silence(self, monkeypatch):
        call_lengths = []

        def fake_decode(audio, state, model=None):  # noqa: ANN001
            call_lengths.append(len(audio))
            return "ok", "English"

        monkeypatch.setattr(smod, "_decode_chunk_incremental", fake_decode)

        state = init_streaming(
            chunk_size_sec=1.0,
            sample_rate=10,
            endpointing_mode="energy",
            endpoint_lookback_sec=0.5,
            endpoint_frame_ms=100.0,
            endpoint_min_chunk_sec=0.2,
        )
        audio = np.ones(10, dtype=np.float32)
        feed_audio(audio, state)

        assert call_lengths == [10]

    def test_feed_audio_consumes_multiple_full_chunks_in_single_call(self, monkeypatch):
        call_lengths = []

        def fake_decode(audio, state, model=None):  # noqa: ANN001
            call_lengths.append(len(audio))
            return "chunk", "English"

        monkeypatch.setattr(smod, "_decode_chunk_incremental", fake_decode)

        state = init_streaming(chunk_size_sec=1.0, sample_rate=10)
        feed_audio(np.ones(25, dtype=np.float32), state)

        assert call_lengths == [10, 10]
        assert state.chunk_id == 2
        assert len(state.buffer) == 5

    def test_feed_audio_caps_audio_accum_memory(self, monkeypatch):
        monkeypatch.setattr(smod, "_decode_chunk_incremental", lambda *_a, **_k: ("ok", "English"))

        state = init_streaming(chunk_size_sec=1.0, max_context_sec=2.0, sample_rate=10)
        feed_audio(np.ones(10, dtype=np.float32), state)
        feed_audio(np.ones(10, dtype=np.float32), state)
        feed_audio(np.ones(10, dtype=np.float32), state)

        assert len(state.audio_accum) == 20

    def test_feed_audio_honors_unfixed_chunk_warmup(self, monkeypatch):
        monkeypatch.setattr(
            smod,
            "_decode_chunk_incremental",
            lambda *_a, **_k: ("one two three four", "English"),
        )

        state = init_streaming(
            chunk_size_sec=1.0,
            sample_rate=10,
            unfixed_chunk_num=1,
            unfixed_token_num=2,
        )

        feed_audio(np.ones(10, dtype=np.float32), state)
        assert state.stable_text == ""

        feed_audio(np.ones(10, dtype=np.float32), state)
        assert state.stable_text == "one two"

    def test_feed_audio_accepts_int16_pcm(self, monkeypatch):
        captured = {}

        def fake_decode(audio, state, model=None):  # noqa: ANN001
            arr = np.asarray(audio)
            captured["dtype"] = arr.dtype
            captured["max_abs"] = float(np.max(np.abs(arr)))
            return "hello", "English"

        monkeypatch.setattr(smod, "_decode_chunk_incremental", fake_decode)

        state = init_streaming(chunk_size_sec=1.0, sample_rate=10)
        feed_audio(np.full((10,), 16384, dtype=np.int16), state)

        assert captured["dtype"] == np.float32
        assert np.isclose(captured["max_abs"], 0.5)

    def test_feed_audio_downmixes_multichannel_input(self, monkeypatch):
        call_lengths = []

        def fake_decode(audio, state, model=None):  # noqa: ANN001
            call_lengths.append(len(audio))
            return "hello", "English"

        monkeypatch.setattr(smod, "_decode_chunk_incremental", fake_decode)

        state = init_streaming(chunk_size_sec=0.5, sample_rate=10)
        feed_audio(np.ones((2, 5), dtype=np.float32), state)

        assert call_lengths == [5]

    def test_feed_audio_none_raises(self):
        state = init_streaming(chunk_size_sec=1.0, sample_rate=10)
        with np.testing.assert_raises(ValueError):
            feed_audio(None, state)  # type: ignore[arg-type]

    def test_feed_audio_empty_array_is_noop(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            smod,
            "_decode_chunk_incremental",
            lambda *_a, **_k: (calls.append(1), "English"),  # type: ignore[misc]
        )

        state = init_streaming(chunk_size_sec=1.0, sample_rate=10)
        out = feed_audio(np.array([], dtype=np.float32), state)
        assert out is state
        assert len(state.buffer) == 0
        assert len(state.audio_accum) == 0
        assert calls == []


class TestFinishStreaming:
    def test_finish_streaming_skips_decode_when_no_pending_buffer(self, monkeypatch):
        def fail_decode(*_args, **_kwargs):  # noqa: ANN001
            raise AssertionError("finish_streaming should not decode without pending buffer")

        monkeypatch.setattr(smod, "_decode_chunk_incremental", fail_decode)

        state = init_streaming(chunk_size_sec=1.0, sample_rate=10)
        state.audio_accum = np.ones(10, dtype=np.float32)
        state.buffer = np.array([], dtype=np.float32)
        state.text = "hello world"
        state.stable_text = "hello world"

        out = finish_streaming(state)
        assert out is state
        assert state.text == "hello world"
        assert state.stable_text == "hello world"

    def test_finish_streaming_decodes_pending_tail(self, monkeypatch):
        calls = []

        def fake_decode(audio, state, model=None):  # noqa: ANN001
            calls.append(len(audio))
            return "tail text", "English"

        monkeypatch.setattr(smod, "_decode_chunk_incremental", fake_decode)

        state = init_streaming(
            chunk_size_sec=1.0,
            sample_rate=10,
            unfixed_chunk_num=0,
            unfixed_token_num=1,
        )
        state.buffer = np.ones(5, dtype=np.float32)
        state.audio_accum = np.ones(5, dtype=np.float32)
        state.stable_text = "prefix"

        out = finish_streaming(state)
        assert out is state
        assert calls == [5]
        assert len(state.buffer) == 0
        assert state.language == "English"
        assert state.stable_text == state.text

    def test_finish_streaming_falls_back_when_tail_makes_no_progress(self, monkeypatch):
        monkeypatch.setattr(
            smod,
            "_decode_chunk_incremental",
            lambda *_a, **_k: ("same text", "English"),
        )

        calls = {"full": 0}

        def fake_transcribe(audio, model, max_new_tokens, verbose, context=""):  # noqa: ANN001
            calls["full"] += 1
            return SimpleNamespace(text="same text plus tail", language="English")

        transcribe_module = __import__("mlx_qwen3_asr.transcribe", fromlist=["transcribe"])
        monkeypatch.setattr(transcribe_module, "transcribe", fake_transcribe)

        state = init_streaming(chunk_size_sec=1.0, sample_rate=10)
        state.text = "same text"
        state.language = "English"
        state.buffer = np.ones(3, dtype=np.float32)
        state.audio_accum = np.ones(13, dtype=np.float32)

        out = finish_streaming(state)
        assert out is state
        assert calls["full"] == 1
        assert state.text == "same text plus tail"
        assert len(state.buffer) == 0
        assert state.stable_text == state.text

    def test_finish_streaming_fallback_uses_bounded_refine_window(self, monkeypatch):
        monkeypatch.setattr(
            smod,
            "_decode_chunk_incremental",
            lambda *_a, **_k: ("prefix", "English"),
        )

        seen = {}

        def fake_transcribe(audio, model, max_new_tokens, verbose, context=""):  # noqa: ANN001
            seen["len"] = len(np.asarray(audio))
            return SimpleNamespace(text="tail", language="English")

        transcribe_module = __import__("mlx_qwen3_asr.transcribe", fromlist=["transcribe"])
        monkeypatch.setattr(transcribe_module, "transcribe", fake_transcribe)

        state = init_streaming(chunk_size_sec=1.0, sample_rate=10)
        state.text = "prefix"
        state.language = "English"
        state.buffer = np.ones(3, dtype=np.float32)
        state.audio_accum = np.ones(25, dtype=np.float32)

        out = finish_streaming(state)
        assert out is state
        assert seen["len"] == 13  # chunk_size_samples (10) + tail (3)
        assert state.text == "prefix tail"
        assert len(state.buffer) == 0

    def test_finish_streaming_skip_fallback_when_disabled(self, monkeypatch):
        monkeypatch.setattr(
            smod,
            "_decode_chunk_incremental",
            lambda *_a, **_k: ("same text", "English"),
        )

        def fail_transcribe(*_a, **_k):  # noqa: ANN001
            raise AssertionError("fallback transcribe should be disabled")

        transcribe_module = __import__("mlx_qwen3_asr.transcribe", fromlist=["transcribe"])
        monkeypatch.setattr(transcribe_module, "transcribe", fail_transcribe)

        state = init_streaming(chunk_size_sec=1.0, sample_rate=10, enable_tail_refine=False)
        state.text = "same text"
        state.language = "English"
        state.buffer = np.ones(3, dtype=np.float32)
        state.audio_accum = np.ones(13, dtype=np.float32)

        out = finish_streaming(state)
        assert out is state
        assert state.text == "same text"
        assert len(state.buffer) == 0


class TestStreamingMetrics:
    def test_streaming_metrics_defaults(self):
        metrics = streaming_metrics(init_streaming())
        assert metrics["chunks_processed"] == 0
        assert metrics["text_chars"] == 0
        assert metrics["stable_chars"] == 0
        assert metrics["partial_stability"] == 1.0
        assert metrics["text_updates"] == 0
        assert metrics["rewrite_events"] == 0
        assert metrics["rewrite_rate"] == 0.0
        assert metrics["finalization_delta_chars"] == 0

    def test_feed_audio_tracks_rewrite_events(self, monkeypatch):
        outputs = iter(
            [
                ("one two three four five six", "English"),
                ("one two three seven eight nine", "English"),
            ]
        )
        monkeypatch.setattr(
            smod,
            "_decode_chunk_incremental",
            lambda *_a, **_k: next(outputs),
        )

        state = init_streaming(chunk_size_sec=1.0, sample_rate=10, unfixed_chunk_num=0)
        feed_audio(np.ones(10, dtype=np.float32), state)
        feed_audio(np.ones(10, dtype=np.float32), state)
        metrics = streaming_metrics(state)

        assert metrics["chunks_processed"] == 2
        assert metrics["text_updates"] == 2
        assert metrics["rewrite_events"] == 1
        assert metrics["rewrite_rate"] == 0.5

    def test_finish_streaming_tracks_finalization_delta_chars(self, monkeypatch):
        monkeypatch.setattr(
            smod,
            "_decode_chunk_incremental",
            lambda *_a, **_k: ("world", "English"),
        )

        state = init_streaming(chunk_size_sec=1.0, sample_rate=10)
        state.text = "hello"
        state.language = "English"
        state.buffer = np.ones(3, dtype=np.float32)
        state.audio_accum = np.ones(13, dtype=np.float32)

        finish_streaming(state)
        metrics = streaming_metrics(state)

        assert state.text == "hello world"
        assert metrics["finalization_delta_chars"] == 6


class TestIncrementalDecode:
    def test_decode_chunk_reuses_single_cache_instance(self, monkeypatch):
        state = init_streaming(chunk_size_sec=1.0, sample_rate=10, max_new_tokens=4)

        class _FakeModel:
            audio_token_id = 151676

            def __init__(self):
                self.create_cache_calls = 0
                self.prefill_cache_ids = []
                self.prefill_starts = []

            def create_cache(self):
                self.create_cache_calls += 1
                return object()

            def audio_tower(self, mel, feature_lens):  # noqa: ANN001
                return mx.zeros((1, 2, 8), dtype=mx.float16), mx.array([2], dtype=mx.int32)

            def prefill(self, input_ids, audio_features, position_ids, cache):  # noqa: ANN001
                self.prefill_cache_ids.append(id(cache))
                self.prefill_starts.append(int(np.array(position_ids)[0, 0, 0]))
                return mx.array([[[0.0, 1.0, 0.0]]], dtype=mx.float32)

        class _FakeTokenizer:
            EOS_TOKEN_IDS = [2]

            def build_prompt_tokens(self, n_audio_tokens, language=None, context=""):  # noqa: ANN001
                return [11, 12]

            def build_followup_prompt_tokens(self, n_audio_tokens, language=None, context=""):  # noqa: ANN001
                return [13]

            def decode(self, ids):  # noqa: ANN001
                return "language English<asr_text>hello"

        fake_model = _FakeModel()
        fake_tokenizer = _FakeTokenizer()

        monkeypatch.setattr(
            smod,
            "_ensure_stream_runtime",
            lambda _state, _model: (fake_model, fake_tokenizer, mx.float16),
        )
        monkeypatch.setattr(
            smod,
            "compute_features",
            lambda _audio: (
                mx.zeros((1, 128, 10), dtype=mx.float32),
                mx.array([10], dtype=mx.int32),
            ),
        )
        monkeypatch.setattr(
            smod,
            "_decode_tokens_incremental",
            lambda **_kwargs: [1],
        )
        monkeypatch.setattr(
            smod,
            "parse_asr_output",
            lambda _raw, user_language=None: ("English", "hello"),
        )

        text_1, lang_1 = smod._decode_chunk_incremental(np.ones(10, dtype=np.float32), state)
        text_2, lang_2 = smod._decode_chunk_incremental(np.ones(10, dtype=np.float32), state)

        assert (text_1, lang_1) == ("hello", "English")
        assert (text_2, lang_2) == ("hello", "English")
        assert fake_model.create_cache_calls == 1
        assert len(fake_model.prefill_cache_ids) == 2
        assert fake_model.prefill_cache_ids[0] == fake_model.prefill_cache_ids[1]
        assert fake_model.prefill_starts == [0, 3]
        assert state._next_position == 5
