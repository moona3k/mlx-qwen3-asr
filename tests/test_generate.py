"""Tests for mlx_qwen3_asr/generate.py."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_qwen3_asr.generate import (
    FINISH_REASON_EOS,
    FINISH_REASON_LENGTH,
    FINISH_REASON_REPETITION,
    REPETITION_THRESHOLD,
    GenerationConfig,
    _build_decode_positions,
    _detect_repetition,
    _sample,
    generate,
    generate_speculative,
    generate_with_info,
    resolve_max_new_tokens,
)

# ---------------------------------------------------------------------------
# GenerationConfig
# ---------------------------------------------------------------------------


class TestGenerationConfig:
    """Test GenerationConfig defaults."""

    def test_default_max_new_tokens(self):
        cfg = GenerationConfig()
        assert cfg.max_new_tokens == 4096

    def test_default_temperature(self):
        cfg = GenerationConfig()
        assert cfg.temperature == 0.0

    def test_default_eos_token_ids(self):
        cfg = GenerationConfig()
        assert cfg.eos_token_ids == [151643, 151645]

    def test_default_eval_interval(self):
        cfg = GenerationConfig()
        assert cfg.eval_interval == 1

    def test_default_num_draft_tokens(self):
        cfg = GenerationConfig()
        assert cfg.num_draft_tokens == 4


class TestResolveMaxNewTokens:
    """Test duration-aware max token budget resolution."""

    def test_explicit_override_is_preserved(self):
        assert resolve_max_new_tokens(4096, audio_duration_sec=5.0) == 4096

    def test_short_audio_uses_floor(self):
        assert resolve_max_new_tokens(None, audio_duration_sec=5.0) == 128

    def test_thirty_second_chunk_is_duration_scaled(self):
        assert resolve_max_new_tokens(None, audio_duration_sec=30.0) == 392

    def test_long_chunk_is_capped(self):
        assert resolve_max_new_tokens(None, audio_duration_sec=120.0) == 512

    def test_rejects_negative_override(self):
        with pytest.raises(ValueError, match="max_new_tokens"):
            resolve_max_new_tokens(-1, audio_duration_sec=5.0)


# ---------------------------------------------------------------------------
# _detect_repetition
# ---------------------------------------------------------------------------


class TestDetectRepetition:
    """Test _detect_repetition() logic."""

    def test_short_sequence_returns_false(self):
        """Sequences shorter than threshold should return False."""
        tokens = [1, 2, 3, 4, 5]
        assert _detect_repetition(tokens) is False

    def test_no_repetition(self):
        """Varied tokens should not trigger repetition."""
        tokens = list(range(REPETITION_THRESHOLD + 10))
        assert _detect_repetition(tokens) is False

    def test_single_token_repeated(self):
        """A single token repeated >= threshold times should be detected."""
        tokens = [42] * (REPETITION_THRESHOLD + 5)
        assert _detect_repetition(tokens) is True

    def test_single_token_just_at_threshold(self):
        tokens = [42] * REPETITION_THRESHOLD
        assert _detect_repetition(tokens) is True

    def test_single_token_below_threshold(self):
        """Single token repeated fewer than threshold times (but sequence >= threshold).
        Must ensure no pattern detection triggers either."""
        # Use enough varied tokens, then repeat below both single and pattern thresholds.
        # Pattern threshold for len-2 is max(2, 20//2) = 10, so keep repeats < 10.
        varied = list(range(REPETITION_THRESHOLD))
        tokens = varied + [42] * 5  # 5 consecutive repeats, well below 20
        assert _detect_repetition(tokens) is False

    def test_pattern_repetition(self):
        """Pattern of 2+ tokens repeated many times should be detected."""
        pattern = [10, 20]
        # threshold // pattern_len = 20 // 2 = 10, need at least max(2, 10) = 10
        tokens = pattern * 12  # 12 repetitions, length 24 >= 20
        assert _detect_repetition(tokens) is True

    def test_pattern_3_tokens(self):
        """Pattern of 3 tokens repeated enough times."""
        pattern = [10, 20, 30]
        # threshold // 3 = 6 (rounded down), need >= max(2, 6) = 6
        # Use 8 repetitions, length = 24 >= 20
        tokens = pattern * 8
        assert _detect_repetition(tokens) is True

    def test_pattern_not_enough_repeats(self):
        """Pattern repeated fewer times than needed should return False."""
        pattern = [10, 20]
        # Only 3 repetitions, well below threshold
        varied = list(range(100, 120))
        tokens = varied + pattern * 3
        assert _detect_repetition(tokens) is False

    def test_empty_list(self):
        assert _detect_repetition([]) is False

    def test_exactly_threshold_length_no_repetition(self):
        tokens = list(range(REPETITION_THRESHOLD))
        assert _detect_repetition(tokens) is False


# ---------------------------------------------------------------------------
# _sample
# ---------------------------------------------------------------------------


class TestSample:
    """Test _sample() sampling logic."""

    def test_greedy_returns_argmax(self):
        """Temperature 0.0 (greedy) should return the argmax index."""
        logits = mx.array([[[0.1, 0.5, 0.3, 0.9, 0.2]]])
        token = _sample(logits, temperature=0.0)
        assert token == 3  # index of 0.9

    def test_greedy_with_negative_values(self):
        logits = mx.array([[[-10.0, -5.0, -1.0, -0.5, -2.0]]])
        token = _sample(logits, temperature=0.0)
        assert token == 3  # index of -0.5

    def test_greedy_with_2d_input(self):
        """_sample reshapes to 1D, so 2D input should work."""
        logits = mx.array([[0.1, 0.5, 0.9, 0.3]])
        token = _sample(logits, temperature=0.0)
        assert token == 2  # index of 0.9

    def test_temperature_sampling_returns_valid_index(self):
        """Temperature > 0 should return a valid index within vocab size."""
        vocab_size = 100
        logits = mx.random.normal((1, 1, vocab_size))
        token = _sample(logits, temperature=1.0)
        assert 0 <= token < vocab_size

    def test_negative_temperature_acts_as_greedy(self):
        """Negative temperature should act as greedy (temperature <= 0.0)."""
        logits = mx.array([[[0.1, 0.5, 0.9, 0.3]]])
        token = _sample(logits, temperature=-1.0)
        assert token == 2


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


class TestGenerate:
    """Test top-level generate() orchestration."""

    def test_generate_uses_model_prefill_and_step_interfaces(self):
        class _DummyModel:
            def __init__(self):
                self.calls = []
                self.cache_obj = object()

            def create_cache(self, max_seq_len=None):  # noqa: ANN001
                self.calls.append(("create_cache", max_seq_len))
                return self.cache_obj

            def prefill(self, input_ids, audio_features, position_ids, cache):  # noqa: ANN001
                self.calls.append(
                    (
                        "prefill",
                        tuple(input_ids.shape),
                        tuple(audio_features.shape),
                        tuple(position_ids.shape),
                        cache is self.cache_obj,
                    )
                )
                # greedy -> token 1
                return mx.array([[[0.0, 1.0, 0.0]]], dtype=mx.float32)

            def step(self, input_ids, position_ids, cache):  # noqa: ANN001
                self.calls.append(
                    (
                        "step",
                        tuple(input_ids.shape),
                        tuple(position_ids.shape),
                        cache is self.cache_obj,
                    )
                )
                # greedy -> eos token 2
                return mx.array([[[0.0, 0.0, 1.0]]], dtype=mx.float32)

        model = _DummyModel()
        input_ids = mx.array([[10, 20, 30, 40, 50]])
        audio_features = mx.zeros((1, 8, 4))
        position_ids = mx.zeros((1, 3, 5), dtype=mx.int32)
        config = GenerationConfig(max_new_tokens=3, temperature=0.0, eos_token_ids=[2])

        out = generate(
            model=model,
            input_ids=input_ids,
            audio_features=audio_features,
            position_ids=position_ids,
            config=config,
        )

        assert out == [1]
        assert model.calls[0] == ("create_cache", 8)
        assert model.calls[1][0] == "prefill"
        assert model.calls[2][0] == "step"

    def test_generate_with_info_reports_eos(self):
        class _DummyModel:
            def create_cache(self, max_seq_len=None):  # noqa: ANN001
                return object()

            def prefill(self, input_ids, audio_features, position_ids, cache):  # noqa: ANN001
                return mx.array([[[0.0, 1.0, 0.0]]], dtype=mx.float32)

            def step(self, input_ids, position_ids, cache):  # noqa: ANN001
                return mx.array([[[0.0, 0.0, 1.0]]], dtype=mx.float32)

        result = generate_with_info(
            model=_DummyModel(),
            input_ids=mx.array([[1, 2, 3]]),
            audio_features=mx.zeros((1, 2, 4)),
            position_ids=mx.zeros((1, 3, 3), dtype=mx.int32),
            config=GenerationConfig(max_new_tokens=3, temperature=0.0, eos_token_ids=[2]),
        )

        assert result.tokens == [1]
        assert result.finish_reason == FINISH_REASON_EOS
        assert result.truncated is False

    def test_generate_with_info_reports_length_without_eos(self):
        class _DummyModel:
            def create_cache(self, max_seq_len=None):  # noqa: ANN001
                return object()

            def prefill(self, input_ids, audio_features, position_ids, cache):  # noqa: ANN001
                return mx.array([[[0.0, 1.0, 0.0]]], dtype=mx.float32)

            def step(self, input_ids, position_ids, cache):  # noqa: ANN001
                return mx.array([[[0.0, 1.0, 0.0]]], dtype=mx.float32)

        result = generate_with_info(
            model=_DummyModel(),
            input_ids=mx.array([[1, 2, 3]]),
            audio_features=mx.zeros((1, 2, 4)),
            position_ids=mx.zeros((1, 3, 3), dtype=mx.int32),
            config=GenerationConfig(max_new_tokens=3, temperature=0.0, eos_token_ids=[999]),
        )

        assert result.tokens == [1, 1, 1]
        assert result.finish_reason == FINISH_REASON_LENGTH
        assert result.truncated is True

    def test_generate_with_info_reports_repetition_before_length(self):
        class _DummyModel:
            def create_cache(self, max_seq_len=None):  # noqa: ANN001
                return object()

            def prefill(self, input_ids, audio_features, position_ids, cache):  # noqa: ANN001
                return mx.array([[[0.0, 1.0, 0.0]]], dtype=mx.float32)

            def step(self, input_ids, position_ids, cache):  # noqa: ANN001
                return mx.array([[[0.0, 1.0, 0.0]]], dtype=mx.float32)

        result = generate_with_info(
            model=_DummyModel(),
            input_ids=mx.array([[1, 2, 3]]),
            audio_features=mx.zeros((1, 2, 4)),
            position_ids=mx.zeros((1, 3, 3), dtype=mx.int32),
            config=GenerationConfig(
                max_new_tokens=REPETITION_THRESHOLD + 5,
                temperature=0.0,
                eos_token_ids=[999],
            ),
        )

        assert result.finish_reason == FINISH_REASON_REPETITION
        assert result.truncated is False

    def test_generate_with_info_reports_length_when_cap_and_repetition_coincide(self):
        class _DummyModel:
            def create_cache(self, max_seq_len=None):  # noqa: ANN001
                return object()

            def prefill(self, input_ids, audio_features, position_ids, cache):  # noqa: ANN001
                return mx.array([[[0.0, 1.0, 0.0]]], dtype=mx.float32)

            def step(self, input_ids, position_ids, cache):  # noqa: ANN001
                return mx.array([[[0.0, 1.0, 0.0]]], dtype=mx.float32)

        result = generate_with_info(
            model=_DummyModel(),
            input_ids=mx.array([[1, 2, 3]]),
            audio_features=mx.zeros((1, 2, 4)),
            position_ids=mx.zeros((1, 3, 3), dtype=mx.int32),
            config=GenerationConfig(
                max_new_tokens=REPETITION_THRESHOLD,
                temperature=0.0,
                eos_token_ids=[999],
            ),
        )

        assert result.finish_reason == FINISH_REASON_LENGTH
        assert result.truncated is True

    def test_generate_handles_max_new_tokens_one(self):
        class _DummyModel:
            def create_cache(self, max_seq_len=None):  # noqa: ANN001
                return object()

            def prefill(self, input_ids, audio_features, position_ids, cache):  # noqa: ANN001
                return mx.array([[[0.0, 1.0, 0.0]]], dtype=mx.float32)

            def step(self, input_ids, position_ids, cache):  # noqa: ANN001
                raise AssertionError("step() should not be called for max_new_tokens=1")

        model = _DummyModel()
        out = generate(
            model=model,
            input_ids=mx.array([[1, 2, 3]]),
            audio_features=mx.zeros((1, 2, 4)),
            position_ids=mx.zeros((1, 3, 3), dtype=mx.int32),
            config=GenerationConfig(max_new_tokens=1, temperature=0.0, eos_token_ids=[999]),
        )
        assert out == [1]

    def test_generate_handles_max_new_tokens_zero(self):
        class _DummyModel:
            def create_cache(self, max_seq_len=None):  # noqa: ANN001
                raise AssertionError("create_cache() should not be called for max_new_tokens=0")

        model = _DummyModel()
        out = generate(
            model=model,
            input_ids=mx.array([[1, 2, 3]]),
            audio_features=mx.zeros((1, 2, 4)),
            position_ids=mx.zeros((1, 3, 3), dtype=mx.int32),
            config=GenerationConfig(max_new_tokens=0, temperature=0.0),
        )
        assert out == []

    def test_generate_rejects_negative_max_new_tokens(self):
        class _DummyModel:
            pass

        with pytest.raises(ValueError, match="max_new_tokens"):
            generate(
                model=_DummyModel(),
                input_ids=mx.array([[1, 2, 3]]),
                audio_features=mx.zeros((1, 2, 4)),
                position_ids=mx.zeros((1, 3, 3), dtype=mx.int32),
                config=GenerationConfig(max_new_tokens=-1, temperature=0.0),
            )


class TestBuildDecodePositions:
    def test_returns_empty_tail_for_small_generation(self):
        pos = _build_decode_positions(seq_len=5, max_new_tokens=1, dtype=mx.int32)
        assert tuple(pos.shape) == (1, 3, 0)


class _SpecCache:
    def __init__(self):
        self.offset = 0
        self.trim_calls: list[int] = []

    def trim(self, num_tokens: int):
        self.trim_calls.append(num_tokens)
        self.offset -= num_tokens


class _SpecDummyModel:
    def __init__(self, transitions: dict[int, int], first_token: int, vocab_size: int = 32):
        self.transitions = transitions
        self.first_token = first_token
        self.vocab_size = vocab_size
        self.last_cache: _SpecCache | None = None

    def create_cache(self, max_seq_len=None):  # noqa: ANN001
        self.last_cache = _SpecCache()
        return self.last_cache

    def _logits(self, next_tokens: list[int]) -> mx.array:
        arr = np.full((1, len(next_tokens), self.vocab_size), -1e9, dtype=np.float32)
        for i, tok in enumerate(next_tokens):
            arr[0, i, tok] = 0.0
        return mx.array(arr)

    def prefill(self, input_ids, audio_features, position_ids, cache):  # noqa: ANN001
        cache.offset += int(input_ids.shape[1])
        return self._logits([self.first_token])

    def step(self, input_ids, position_ids, cache):  # noqa: ANN001
        tok = int(np.array(input_ids)[0, 0])
        cache.offset += 1
        return self._logits([self.transitions[tok]])

    def step_many(self, input_ids, position_ids, cache):  # noqa: ANN001
        toks = np.array(input_ids)[0].tolist()
        cache.offset += int(len(toks))
        next_tokens = [self.transitions[int(t)] for t in toks]
        return self._logits(next_tokens)


class TestGenerateSpeculative:
    def _dummy_inputs(self):
        input_ids = mx.array([[10, 20, 30]])
        audio_features = mx.zeros((1, 2, 4))
        position_ids = mx.zeros((1, 3, 3), dtype=mx.int32)
        return input_ids, audio_features, position_ids

    def test_matches_greedy_when_target_and_draft_agree(self):
        transitions = {i: i + 1 for i in range(0, 20)}
        target = _SpecDummyModel(transitions=transitions, first_token=1)
        draft = _SpecDummyModel(transitions=transitions, first_token=1)

        input_ids, audio_features, position_ids = self._dummy_inputs()
        cfg = GenerationConfig(
            max_new_tokens=6,
            temperature=0.0,
            eos_token_ids=[999],
            num_draft_tokens=3,
        )

        greedy = generate(
            model=target,
            input_ids=input_ids,
            audio_features=audio_features,
            position_ids=position_ids,
            config=cfg,
        )
        speculative = generate_speculative(
            model=target,
            draft_model=draft,
            input_ids=input_ids,
            audio_features=audio_features,
            draft_audio_features=audio_features,
            position_ids=position_ids,
            config=cfg,
        )

        assert speculative == greedy

    def test_matches_greedy_with_partial_rejections(self):
        target_transitions = {i: i + 1 for i in range(0, 40)}
        draft_transitions = {i: i + 1 for i in range(0, 40)}
        target_transitions[2] = 9  # diverges from draft here
        target_transitions[9] = 4
        target = _SpecDummyModel(transitions=target_transitions, first_token=1)
        draft = _SpecDummyModel(transitions=draft_transitions, first_token=1)

        input_ids, audio_features, position_ids = self._dummy_inputs()
        cfg = GenerationConfig(
            max_new_tokens=7,
            temperature=0.0,
            eos_token_ids=[999],
            num_draft_tokens=3,
        )

        greedy = generate(
            model=target,
            input_ids=input_ids,
            audio_features=audio_features,
            position_ids=position_ids,
            config=cfg,
        )
        speculative = generate_speculative(
            model=target,
            draft_model=draft,
            input_ids=input_ids,
            audio_features=audio_features,
            draft_audio_features=audio_features,
            position_ids=position_ids,
            config=cfg,
        )

        assert speculative == greedy
        assert any(v > 0 for v in target.last_cache.trim_calls)
        assert any(v > 0 for v in draft.last_cache.trim_calls)
        assert target.last_cache.trim_calls == draft.last_cache.trim_calls

    def test_rejects_non_greedy_mode(self):
        transitions = {i: i + 1 for i in range(0, 20)}
        target = _SpecDummyModel(transitions=transitions, first_token=1)
        draft = _SpecDummyModel(transitions=transitions, first_token=1)
        input_ids, audio_features, position_ids = self._dummy_inputs()
        cfg = GenerationConfig(max_new_tokens=4, temperature=0.7)

        with pytest.raises(ValueError, match="greedy mode"):
            generate_speculative(
                model=target,
                draft_model=draft,
                input_ids=input_ids,
                audio_features=audio_features,
                draft_audio_features=audio_features,
                position_ids=position_ids,
                config=cfg,
            )

    def test_handles_max_new_tokens_one(self):
        transitions = {i: i + 1 for i in range(0, 20)}
        target = _SpecDummyModel(transitions=transitions, first_token=1)
        draft = _SpecDummyModel(transitions=transitions, first_token=1)
        input_ids, audio_features, position_ids = self._dummy_inputs()
        cfg = GenerationConfig(max_new_tokens=1, temperature=0.0, eos_token_ids=[999])

        out = generate_speculative(
            model=target,
            draft_model=draft,
            input_ids=input_ids,
            audio_features=audio_features,
            draft_audio_features=audio_features,
            position_ids=position_ids,
            config=cfg,
        )
        assert out == [1]

    def test_handles_max_new_tokens_zero(self):
        transitions = {i: i + 1 for i in range(0, 20)}
        target = _SpecDummyModel(transitions=transitions, first_token=1)
        draft = _SpecDummyModel(transitions=transitions, first_token=1)
        input_ids, audio_features, position_ids = self._dummy_inputs()
        cfg = GenerationConfig(max_new_tokens=0, temperature=0.0, eos_token_ids=[999])

        out = generate_speculative(
            model=target,
            draft_model=draft,
            input_ids=input_ids,
            audio_features=audio_features,
            draft_audio_features=audio_features,
            position_ids=position_ids,
            config=cfg,
        )
        assert out == []

    def test_rejects_negative_max_new_tokens(self):
        transitions = {i: i + 1 for i in range(0, 20)}
        target = _SpecDummyModel(transitions=transitions, first_token=1)
        draft = _SpecDummyModel(transitions=transitions, first_token=1)
        input_ids, audio_features, position_ids = self._dummy_inputs()
        cfg = GenerationConfig(max_new_tokens=-1, temperature=0.0)

        with pytest.raises(ValueError, match="max_new_tokens"):
            generate_speculative(
                model=target,
                draft_model=draft,
                input_ids=input_ids,
                audio_features=audio_features,
                draft_audio_features=audio_features,
                position_ids=position_ids,
                config=cfg,
            )
