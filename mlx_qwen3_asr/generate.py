"""Autoregressive text generation for Qwen3-ASR."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx

from .model import Qwen3ASRModel
from .runtime_utils import supports_kwarg

# Repetition detection constants (from official repo)
REPETITION_THRESHOLD = 20
AUTO_MAX_NEW_TOKENS_FLOOR = 128
AUTO_MAX_NEW_TOKENS_CAP = 512
AUTO_MAX_NEW_TOKENS_PER_SECOND = 12.0
AUTO_MAX_NEW_TOKENS_MARGIN = 32
FINISH_REASON_EOS = "eos"
FINISH_REASON_REPETITION = "repetition"
FINISH_REASON_LENGTH = "length"


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 4096
    temperature: float = 0.0  # greedy by default
    eos_token_ids: list[int] = field(default_factory=lambda: [151643, 151645])
    eval_interval: int = 1
    num_draft_tokens: int = 4


@dataclass(frozen=True)
class GenerationResult:
    """Generated token IDs plus decode termination metadata.

    `tokens` and `generated_tokens` exclude the trailing EOS sentinel when the
    decoder emitted one — so for an EOS-terminated decode that produced N
    visible tokens plus an EOS, both values are N. The invariant
    ``truncated <=> generated_tokens == max_new_tokens`` therefore holds: a
    truncated run never had an EOS to strip.
    """

    tokens: list[int]
    finish_reason: str
    generated_tokens: int
    max_new_tokens: int

    @property
    def truncated(self) -> bool:
        """Whether decoding stopped because the token budget was exhausted."""
        return self.finish_reason == FINISH_REASON_LENGTH


def resolve_max_new_tokens(
    max_new_tokens: Optional[int],
    *,
    audio_duration_sec: float,
) -> int:
    """Resolve the decode token budget for one audio chunk.

    Args:
        max_new_tokens: Explicit caller-provided budget. When set, this value
            is returned unchanged after validation.
        audio_duration_sec: Duration of the audio chunk used for the adaptive
            default when `max_new_tokens` is unset.

    Returns:
        Integer token budget. The adaptive default is
        `ceil(duration * AUTO_MAX_NEW_TOKENS_PER_SECOND) +
        AUTO_MAX_NEW_TOKENS_MARGIN`, clamped to
        `[AUTO_MAX_NEW_TOKENS_FLOOR, AUTO_MAX_NEW_TOKENS_CAP]`.

    Raises:
        ValueError: If `max_new_tokens` is negative.
    """
    if max_new_tokens is not None:
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
        return int(max_new_tokens)

    duration = max(0.0, float(audio_duration_sec))
    adaptive = math.ceil(duration * AUTO_MAX_NEW_TOKENS_PER_SECOND)
    adaptive += AUTO_MAX_NEW_TOKENS_MARGIN
    return min(
        AUTO_MAX_NEW_TOKENS_CAP,
        max(AUTO_MAX_NEW_TOKENS_FLOOR, adaptive),
    )


def coerce_generation_result(
    output: GenerationResult | list[int],
    config: GenerationConfig,
) -> GenerationResult:
    """Normalize legacy token-list output into a `GenerationResult`.

    Args:
        output: Either a metadata-rich generation result or the legacy
            `list[int]` returned by older generation hooks.
        config: Generation configuration used for the decode turn.

    Returns:
        `GenerationResult` with best-effort finish metadata. Legacy list
        output is classified as `length` when it reaches the token budget,
        otherwise `repetition` when repetition is observable, otherwise `eos`.
    """
    if isinstance(output, GenerationResult):
        return output
    tokens = list(output)
    if len(tokens) >= config.max_new_tokens:
        finish_reason = FINISH_REASON_LENGTH
    elif detect_repetition(tokens):
        finish_reason = FINISH_REASON_REPETITION
    else:
        finish_reason = FINISH_REASON_EOS
    return GenerationResult(
        tokens=tokens,
        finish_reason=finish_reason,
        generated_tokens=len(tokens),
        max_new_tokens=config.max_new_tokens,
    )


def generate(
    model: Qwen3ASRModel,
    input_ids: mx.array,
    audio_features: mx.array,
    position_ids: mx.array,
    config: GenerationConfig = None,
) -> list[int]:
    """Generate text tokens autoregressively.

    Two phases:
    1. Prefill: Process full input sequence (text prompt + audio features)
    2. Decode: Generate one token at a time with KV cache

    Args:
        model: The Qwen3ASR model
        input_ids: Input token IDs with audio placeholders, shape (1, seq_len)
        audio_features: Encoded audio features from audio encoder, shape (1, n_audio, dim)
        position_ids: MRoPE position IDs, shape (1, 3, seq_len)
        config: Generation configuration

    Returns:
        List of generated token IDs (excluding the input)
    """
    return generate_with_info(
        model=model,
        input_ids=input_ids,
        audio_features=audio_features,
        position_ids=position_ids,
        config=config,
    ).tokens


def generate_with_info(
    model: Qwen3ASRModel,
    input_ids: mx.array,
    audio_features: mx.array,
    position_ids: mx.array,
    config: GenerationConfig = None,
) -> GenerationResult:
    """Generate text tokens with decode termination metadata.

    Args:
        model: Qwen3-ASR model.
        input_ids: Input token IDs with audio placeholders, shape `(1, seq_len)`.
        audio_features: Encoded audio features from the audio tower.
        position_ids: MRoPE position IDs, shape `(1, 3, seq_len)`.
        config: Generation configuration. Defaults to `GenerationConfig()`.

    Returns:
        `GenerationResult` containing generated token IDs, finish reason,
        generated token count, and the effective token budget.

    Raises:
        ValueError: If `config.max_new_tokens` is negative.
    """
    if config is None:
        config = GenerationConfig()
    if config.max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {config.max_new_tokens}")
    if config.max_new_tokens == 0:
        return GenerationResult(
            tokens=[],
            finish_reason=FINISH_REASON_LENGTH,
            generated_tokens=0,
            max_new_tokens=0,
        )

    max_seq_len = int(input_ids.shape[1] + config.max_new_tokens)
    cache = model.create_cache(max_seq_len=max_seq_len)
    unchecked_step_kw = (
        {"validate_input_ids": False}
        if supports_kwarg(getattr(model, "step", None), "validate_input_ids")
        else {}
    )

    # Phase 1: Prefill prompt and populate cache.
    logits = model.prefill(
        input_ids=input_ids,
        audio_features=audio_features,
        position_ids=position_ids,
        cache=cache,
    )

    # Sample first token
    token = _sample(logits, config.temperature)
    generated = [token]

    # Phase 2: Autoregressive decode
    seq_len = input_ids.shape[1]
    next_pos_3d = _build_decode_positions(
        seq_len=seq_len,
        max_new_tokens=config.max_new_tokens,
        dtype=position_ids.dtype,
    )

    for step in range(1, config.max_new_tokens):
        # Check EOS
        if token in config.eos_token_ids:
            break

        # Check repetition
        if _detect_repetition(generated):
            break

        # Next token input
        next_ids = mx.array([[token]])

        # Reuse precomputed decode positions to avoid per-step allocation.
        next_position_ids = next_pos_3d[:, :, step - 1 : step]

        logits = model.step(
            input_ids=next_ids,
            position_ids=next_position_ids,
            cache=cache,
            **unchecked_step_kw,
        )
        token = _sample(logits, config.temperature)
        generated.append(token)

        _periodic_eval(cache=cache, step=step, eval_interval=config.eval_interval)

    return _finalize_generation_result(generated, config)


def generate_speculative(
    model: Qwen3ASRModel,
    draft_model: Qwen3ASRModel,
    input_ids: mx.array,
    audio_features: mx.array,
    draft_audio_features: mx.array,
    position_ids: mx.array,
    config: GenerationConfig = None,
) -> list[int]:
    """Generate text with greedy speculative decoding.

    Uses a small draft model to propose tokens and verifies them in batches with
    the target model. Output is guaranteed to match greedy decoding of the
    target model.
    """
    return generate_speculative_with_info(
        model=model,
        draft_model=draft_model,
        input_ids=input_ids,
        audio_features=audio_features,
        draft_audio_features=draft_audio_features,
        position_ids=position_ids,
        config=config,
    ).tokens


def generate_speculative_with_info(
    model: Qwen3ASRModel,
    draft_model: Qwen3ASRModel,
    input_ids: mx.array,
    audio_features: mx.array,
    draft_audio_features: mx.array,
    position_ids: mx.array,
    config: GenerationConfig = None,
) -> GenerationResult:
    """Generate with greedy speculative decoding and finish metadata.

    Args:
        model: Target Qwen3-ASR model.
        draft_model: Smaller draft model used to propose tokens.
        input_ids: Input token IDs with audio placeholders, shape `(1, seq_len)`.
        audio_features: Encoded audio features for the target model.
        draft_audio_features: Encoded audio features for the draft model.
        position_ids: MRoPE position IDs, shape `(1, 3, seq_len)`.
        config: Generation configuration. Defaults to `GenerationConfig()`.

    Returns:
        `GenerationResult` containing generated token IDs, finish reason,
        generated token count, and the effective token budget.

    Raises:
        ValueError: If `config.max_new_tokens` is negative, temperature is not
            greedy, or `config.num_draft_tokens` is less than one.
    """
    if config is None:
        config = GenerationConfig()
    if config.max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {config.max_new_tokens}")
    if config.max_new_tokens == 0:
        return GenerationResult(
            tokens=[],
            finish_reason=FINISH_REASON_LENGTH,
            generated_tokens=0,
            max_new_tokens=0,
        )
    if config.temperature != 0.0:
        raise ValueError("Speculative decoding currently supports greedy mode only.")
    if config.num_draft_tokens < 1:
        raise ValueError(
            f"num_draft_tokens must be >= 1, got {config.num_draft_tokens}"
        )

    max_seq_len = int(input_ids.shape[1] + config.max_new_tokens)
    target_cache = model.create_cache(max_seq_len=max_seq_len)
    draft_cache = draft_model.create_cache(max_seq_len=max_seq_len)
    unchecked_target_step_kw = (
        {"validate_input_ids": False}
        if supports_kwarg(getattr(model, "step", None), "validate_input_ids")
        else {}
    )
    unchecked_draft_step_kw = (
        {"validate_input_ids": False}
        if supports_kwarg(getattr(draft_model, "step", None), "validate_input_ids")
        else {}
    )
    unchecked_step_many_kw = (
        {"validate_input_ids": False}
        if supports_kwarg(getattr(model, "step_many", None), "validate_input_ids")
        else {}
    )

    target_logits = model.prefill(
        input_ids=input_ids,
        audio_features=audio_features,
        position_ids=position_ids,
        cache=target_cache,
    )
    _ = draft_model.prefill(
        input_ids=input_ids,
        audio_features=draft_audio_features,
        position_ids=position_ids,
        cache=draft_cache,
    )

    token = _sample(target_logits, config.temperature)
    generated = [token]

    seq_len = int(input_ids.shape[1])
    next_pos_3d = _build_decode_positions(
        seq_len=seq_len,
        max_new_tokens=config.max_new_tokens,
        dtype=position_ids.dtype,
    )

    step = 1
    while step < config.max_new_tokens:
        if token in config.eos_token_ids:
            break
        if _detect_repetition(generated):
            break

        remaining = config.max_new_tokens - step
        num_draft = min(config.num_draft_tokens, remaining - 1) if remaining > 1 else 0
        if num_draft <= 0:
            next_ids = mx.array([[token]])
            next_position_ids = next_pos_3d[:, :, step - 1 : step]
            logits = model.step(
                input_ids=next_ids,
                position_ids=next_position_ids,
                cache=target_cache,
                **unchecked_target_step_kw,
            )
            token = _sample(logits, config.temperature)
            generated.append(token)
            _periodic_eval(
                cache=target_cache,
                step=step,
                eval_interval=config.eval_interval,
            )
            step += 1
            continue

        draft_tokens: list[int] = []
        draft_input = token
        for d_i in range(num_draft):
            d_ids = mx.array([[draft_input]])
            d_pos = next_pos_3d[:, :, step - 1 + d_i : step + d_i]
            d_logits = draft_model.step(
                input_ids=d_ids,
                position_ids=d_pos,
                cache=draft_cache,
                **unchecked_draft_step_kw,
            )
            draft_input = _sample(d_logits, config.temperature)
            draft_tokens.append(draft_input)

        verify_ids = mx.array([[token, *draft_tokens]])
        verify_pos = next_pos_3d[:, :, step - 1 : step + num_draft]
        verify_logits = model.step_many(
            input_ids=verify_ids,
            position_ids=verify_pos,
            cache=target_cache,
            **unchecked_step_many_kw,
        )
        verify_pred_ids = mx.argmax(verify_logits, axis=-1)
        verify_pred = [int(x) for x in verify_pred_ids[0].tolist()]

        accepted = 0
        while accepted < num_draft and verify_pred[accepted] == draft_tokens[accepted]:
            accepted += 1

        # Rewind unaccepted speculative steps so both caches match accepted path.
        target_cache.trim(num_draft - accepted)
        draft_cache.trim(num_draft - accepted)

        stop = False
        for i in range(accepted):
            token = draft_tokens[i]
            generated.append(token)
            step += 1
            if step >= config.max_new_tokens:
                stop = True
                break
            if token in config.eos_token_ids:
                stop = True
                break
            if _detect_repetition(generated):
                stop = True
                break
        if stop:
            break

        token = verify_pred[accepted]
        generated.append(token)
        step += 1

        _periodic_eval(
            cache=target_cache,
            step=step,
            eval_interval=config.eval_interval,
        )
        _periodic_eval(
            cache=draft_cache,
            step=step,
            eval_interval=config.eval_interval,
        )

    return _finalize_generation_result(generated, config)


def _finalize_generation_result(
    generated: list[int],
    config: GenerationConfig,
) -> GenerationResult:
    finish_reason = _infer_finish_reason(generated, config)
    if generated and generated[-1] in config.eos_token_ids:
        generated = generated[:-1]
    return GenerationResult(
        tokens=generated,
        finish_reason=finish_reason,
        generated_tokens=len(generated),
        max_new_tokens=config.max_new_tokens,
    )


def _infer_finish_reason(
    generated: list[int],
    config: GenerationConfig,
) -> str:
    if generated and generated[-1] in config.eos_token_ids:
        return FINISH_REASON_EOS
    if len(generated) >= config.max_new_tokens:
        return FINISH_REASON_LENGTH
    if _detect_repetition(generated):
        return FINISH_REASON_REPETITION
    raise AssertionError("generation stopped without EOS, repetition, or token cap")


def _sample(logits: mx.array, temperature: float) -> int:
    """Sample a token from logits.

    Args:
        logits: Shape (1, 1, vocab_size) or (1, vocab_size)
        temperature: 0.0 for greedy, > 0 for sampling

    Returns:
        Selected token ID
    """
    logits = logits.reshape(-1)  # (vocab_size,)

    if temperature <= 0.0:
        # Greedy
        return mx.argmax(logits).item()
    else:
        # Temperature sampling — pass logits directly (categorical expects log-probs)
        return mx.random.categorical(logits / temperature).item()


def _build_decode_positions(
    seq_len: int,
    max_new_tokens: int,
    dtype: mx.Dtype,
) -> mx.array:
    next_pos_base = mx.arange(
        seq_len,
        seq_len + max(max_new_tokens - 1, 0),
        dtype=dtype,
    )
    next_pos_3d = mx.stack([next_pos_base, next_pos_base, next_pos_base], axis=0)
    return next_pos_3d[None, :, :]


def _periodic_eval(
    cache: object,
    step: int,
    eval_interval: int,
) -> None:
    # Materialize cache periodically to avoid graph buildup while reducing
    # per-step synchronization overhead.
    if eval_interval <= 0 or (step % eval_interval) != 0:
        return
    if not hasattr(cache, "keys") or not hasattr(cache, "values"):
        return
    cache_tensors = [c for c in cache.keys if c is not None] + [
        c for c in cache.values if c is not None
    ]
    if cache_tensors:
        mx.eval(cache_tensors)


def detect_repetition(tokens: list[int], threshold: int = REPETITION_THRESHOLD) -> bool:
    """Detect repetitive token patterns in generated output.

    Args:
        tokens: Generated token IDs.
        threshold: Repetition threshold used by the two-stage detector.

    Returns:
        `True` when a repeated single token or short token pattern is detected.
    """
    return _detect_repetition(tokens, threshold=threshold)


def _detect_repetition(tokens: list[int], threshold: int = REPETITION_THRESHOLD) -> bool:
    """Detect repetitive token patterns.

    Two-stage detection (from official repo):
    1. Single token repeated > threshold times consecutively
    2. Pattern of 2-10 tokens repeated > threshold/pattern_len times

    Returns:
        True if repetition detected
    """
    if len(tokens) < threshold:
        return False

    # Stage 1: Single token repetition
    count = 1
    for i in range(len(tokens) - 1, 0, -1):
        if tokens[i] == tokens[i - 1]:
            count += 1
            if count >= threshold:
                return True
        else:
            break

    # Stage 2: Pattern repetition (check patterns of length 2-10)
    for pattern_len in range(2, min(11, len(tokens) // 2 + 1)):
        pattern = tokens[-pattern_len:]
        repeat_count = 1
        pos = len(tokens) - pattern_len * 2
        while pos >= 0:
            if tokens[pos:pos + pattern_len] == pattern:
                repeat_count += 1
                pos -= pattern_len
            else:
                break
        if repeat_count >= max(2, threshold // pattern_len):
            return True

    return False
