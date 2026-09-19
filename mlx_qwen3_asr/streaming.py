"""Streaming ASR: windowed re-decode with text-prefix rollback.

This follows the official Qwen3-ASR streaming recipe
(``qwen_asr.inference.qwen3_asr.streaming_transcribe``): every time a chunk
arrives, the *whole* accumulated audio window is encoded again and decoded
with the previously generated text, minus its last ``unfixed_token_num``
tokens, forced into the prompt as a prefix. The model therefore always sees
the full acoustic context it was trained on and only has to generate the new
tail, so partial results are stable and the final text matches offline
quality.

The window is bounded by ``max_context_sec``. When the next chunk would
overflow it, the current window's text is committed and a fresh window starts
with that chunk; the committed text and the live window text are joined into
``state.text``.

An earlier design fed each chunk through the encoder on its own and appended
it to a live KV cache as a follow-up chat turn. That is linear in cost but the
model was never trained on it: the multilingual-100 lane measured 56% primary
error against 9.5% offline (``docs/benchmarks/2026-09-19-streaming-manifest-*``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import numpy as np

from .attention import scalar_int
from .audio import compute_features
from .config import DEFAULT_MODEL_ID
from .generate import (
    AUTO_MAX_NEW_TOKENS_FLOOR,
    detect_repetition,
    resolve_max_new_tokens,
)
from .load_models import _ModelHolder
from .model import Qwen3ASRModel
from .tokenizer import (
    Tokenizer,
    _TokenizerHolder,
    canonicalize_language,
    is_no_space_language,
    parse_asr_output,
)

# Streaming constants (from official repo)
UNFIXED_CHUNK_NUM = 2     # Chunks decoded without a text prefix at window start
UNFIXED_TOKEN_NUM = 5     # Trailing tokens rolled back before each re-decode

_DEFAULT_EOS_TOKEN_IDS = (151643, 151645)
_ENDPOINTING_MODES = {"fixed", "energy"}
_REPLACEMENT_CHAR = "\ufffd"


@dataclass
class StreamingState:
    """State for streaming ASR session.

    Attributes:
        buffer: Pending audio samples not yet processed.
        audio_accum: Audio of the current decode window (bounded by
            ``max_context_samples``); re-encoded on every chunk.
        text: Current best transcription (committed windows + live window).
        committed_text: Text of windows that have already been flushed out of
            ``audio_accum``; never re-decoded.
        window_text: Parsed text of the live window.
        language: Detected language.
        chunk_id: Number of chunks processed over the whole stream.
        unfixed_chunk_num: Chunks at the start of each window decoded without
            a text prefix.
        unfixed_token_num: Trailing generated tokens rolled back before each
            re-decode; also the number of trailing units reported unstable.
        chunk_size_samples: Samples per chunk.
        max_context_samples: Max samples retained in ``audio_accum``.
        sample_rate: Sample rate of the incoming PCM.
        stable_text: Text considered stable (won't change).
        max_new_tokens: Per-decode token budget. When the caller left it
            unset, this is the adaptive floor and the effective budget scales
            with the window duration; an explicit value is a hard cap.
        finalization_mode: Retained for API compatibility (`accuracy` or
            `latency`); both modes decode the pending tail at finish.
        enable_tail_refine: Retained for API compatibility; the window
            re-decode at finish subsumes the former tail refinement pass.
    """

    buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    audio_accum: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    text: str = ""
    committed_text: str = ""
    window_text: str = ""
    context: str = ""
    language: str = "unknown"
    forced_language: Optional[str] = None
    chunk_id: int = 0
    unfixed_chunk_num: int = UNFIXED_CHUNK_NUM
    unfixed_token_num: int = UNFIXED_TOKEN_NUM
    chunk_size_samples: int = 32000  # 2 seconds at 16kHz
    max_context_samples: int = 480000  # 30 seconds at 16kHz
    sample_rate: int = 16000
    stable_text: str = ""
    max_new_tokens: int = AUTO_MAX_NEW_TOKENS_FLOOR
    finalization_mode: str = "accuracy"
    enable_tail_refine: bool = True
    endpointing_mode: str = "fixed"
    endpoint_lookback_samples: int = 4800  # 300ms at 16kHz
    endpoint_frame_samples: int = 320  # 20ms at 16kHz
    endpoint_min_chunk_samples: int = 8000  # 500ms at 16kHz
    _model_path: str = DEFAULT_MODEL_ID
    _adaptive_token_budget: bool = True
    _window_raw_ids: list[int] = field(default_factory=list)
    _window_chunk_id: int = 0
    _model_obj: Optional[Qwen3ASRModel] = None
    _tokenizer: Optional[Tokenizer] = None
    _dtype: mx.Dtype = mx.float16
    _resolved_model_path: Optional[str] = None
    _text_updates: int = 0
    _rewrite_events: int = 0
    _pre_finalize_text: str = ""
    _finalization_delta_chars: int = 0


def init_streaming(
    model: str = DEFAULT_MODEL_ID,
    context: str = "",
    unfixed_chunk_num: int = UNFIXED_CHUNK_NUM,
    unfixed_token_num: int = UNFIXED_TOKEN_NUM,
    chunk_size_sec: float = 2.0,
    max_context_sec: float = 30.0,
    sample_rate: int = 16000,
    dtype: mx.Dtype = mx.float16,
    max_new_tokens: Optional[int] = None,
    finalization_mode: str = "accuracy",
    enable_tail_refine: Optional[bool] = None,
    endpointing_mode: str = "fixed",
    endpoint_lookback_sec: float = 0.3,
    endpoint_frame_ms: float = 20.0,
    endpoint_min_chunk_sec: float = 0.5,
    language: Optional[str] = None,
) -> StreamingState:
    """Initialize a streaming ASR session."""
    if chunk_size_sec <= 0:
        raise ValueError(f"chunk_size_sec must be > 0, got: {chunk_size_sec}")
    if max_context_sec <= 0:
        raise ValueError(f"max_context_sec must be > 0, got: {max_context_sec}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be > 0, got: {sample_rate}")
    if max_context_sec < chunk_size_sec:
        raise ValueError(
            f"max_context_sec must be >= chunk_size_sec, got: "
            f"{max_context_sec} < {chunk_size_sec}"
        )
    effective_max_new_tokens = resolve_max_new_tokens(
        max_new_tokens,
        audio_duration_sec=chunk_size_sec,
    )
    if effective_max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be > 0, got: {max_new_tokens}")
    mode = str(finalization_mode).strip().lower()
    if mode not in {"accuracy", "latency"}:
        raise ValueError(
            f"finalization_mode must be 'accuracy' or 'latency', got: {finalization_mode}"
        )
    ep_mode = str(endpointing_mode).strip().lower()
    if ep_mode not in _ENDPOINTING_MODES:
        raise ValueError(
            f"endpointing_mode must be one of {sorted(_ENDPOINTING_MODES)}, "
            f"got: {endpointing_mode}"
        )
    if endpoint_lookback_sec < 0:
        raise ValueError(
            f"endpoint_lookback_sec must be >= 0, got: {endpoint_lookback_sec}"
        )
    if endpoint_frame_ms <= 0:
        raise ValueError(f"endpoint_frame_ms must be > 0, got: {endpoint_frame_ms}")
    if endpoint_min_chunk_sec <= 0:
        raise ValueError(
            f"endpoint_min_chunk_sec must be > 0, got: {endpoint_min_chunk_sec}"
        )

    # Backward-compatible override for callers still passing enable_tail_refine.
    if enable_tail_refine is None:
        tail_refine = mode == "accuracy"
    else:
        tail_refine = bool(enable_tail_refine)
        mode = "accuracy" if tail_refine else "latency"

    return StreamingState(
        context=context or "",
        unfixed_chunk_num=int(unfixed_chunk_num),
        unfixed_token_num=int(unfixed_token_num),
        chunk_size_samples=int(chunk_size_sec * sample_rate),
        max_context_samples=int(max_context_sec * sample_rate),
        sample_rate=int(sample_rate),
        _model_path=model,
        _dtype=dtype,
        max_new_tokens=int(effective_max_new_tokens),
        _adaptive_token_budget=max_new_tokens is None,
        finalization_mode=mode,
        enable_tail_refine=tail_refine,
        endpointing_mode=ep_mode,
        endpoint_lookback_samples=max(0, int(endpoint_lookback_sec * sample_rate)),
        endpoint_frame_samples=max(1, int((endpoint_frame_ms / 1000.0) * sample_rate)),
        endpoint_min_chunk_samples=max(1, int(endpoint_min_chunk_sec * sample_rate)),
        forced_language=canonicalize_language(language),
    )


def feed_audio(
    pcm: np.ndarray,
    state: StreamingState,
    model: Optional[Qwen3ASRModel] = None,
) -> StreamingState:
    """Feed audio to the streaming session, re-decoding the window per chunk."""
    if pcm is None:
        raise ValueError("pcm must not be None")

    x = _sanitize_stream_pcm(pcm)
    if x.size == 0:
        return state

    state.buffer = np.concatenate([state.buffer, x])

    while len(state.buffer) >= state.chunk_size_samples:
        decode_samples = _select_decode_samples(state)
        if decode_samples <= 0:
            break
        chunk_audio = state.buffer[:decode_samples]
        state.buffer = state.buffer[decode_samples:]

        _extend_window(state, chunk_audio)
        prev_text = state.text
        _decode_and_update(state, model)

        if state.text != prev_text:
            state._text_updates += 1
        if prev_text and state.text and not state.text.startswith(prev_text):
            state._rewrite_events += 1

        if state.chunk_id < state.unfixed_chunk_num:
            stable = state.stable_text
        else:
            stable, _ = _split_stable_unstable(
                state.stable_text,
                state.text,
                unfixed_tokens=state.unfixed_token_num,
            )
        state.chunk_id += 1
        state.stable_text = stable

    return state


def finish_streaming(
    state: StreamingState,
    model: Optional[Qwen3ASRModel] = None,
) -> StreamingState:
    """Finalize the session by decoding the window with any pending tail audio."""
    state._pre_finalize_text = state.text
    if len(state.buffer) == 0:
        state._finalization_delta_chars = 0
        if len(state.audio_accum) > 0:
            state.stable_text = state.text
        return state

    tail = state.buffer
    state.buffer = np.array([], dtype=np.float32)
    _extend_window(state, tail)
    _decode_and_update(state, model)
    state.chunk_id += 1

    state.stable_text = state.text
    state._finalization_delta_chars = len(state.text) - len(state._pre_finalize_text)
    return state


def streaming_metrics(state: StreamingState) -> dict[str, float | int]:
    """Return lightweight streaming quality diagnostics for a session state."""
    text_chars = int(len(state.text))
    stable_chars = int(len(state.stable_text))
    text_updates = int(state._text_updates)
    rewrite_events = int(state._rewrite_events)
    return {
        "chunks_processed": int(state.chunk_id),
        "text_chars": text_chars,
        "stable_chars": stable_chars,
        "partial_stability": float(stable_chars / text_chars) if text_chars > 0 else 1.0,
        "text_updates": text_updates,
        "rewrite_events": rewrite_events,
        "rewrite_rate": float(rewrite_events / text_updates) if text_updates > 0 else 0.0,
        "finalization_delta_chars": int(state._finalization_delta_chars),
    }


def _extend_window(state: StreamingState, audio: np.ndarray) -> None:
    """Append audio to the live window, committing the window first if it would overflow."""
    overflow = len(state.audio_accum) + len(audio) > state.max_context_samples
    if len(state.audio_accum) > 0 and overflow:
        _commit_window(state)
    state.audio_accum = np.concatenate([state.audio_accum, audio])


def _commit_window(state: StreamingState) -> None:
    """Freeze the live window's text and start an empty window."""
    state.committed_text = _join_texts(state.committed_text, state.window_text, state.language)
    state.window_text = ""
    state.audio_accum = np.array([], dtype=np.float32)
    state._window_raw_ids = []
    state._window_chunk_id = 0


def _decode_and_update(state: StreamingState, model: Optional[Qwen3ASRModel]) -> None:
    """Re-decode the live window and refresh ``text``/``language``."""
    window_text, window_language = _decode_window(state.audio_accum, state, model=model)
    if window_language and window_language != "unknown" and state.language == "unknown":
        state.language = window_language
    state.window_text = window_text
    state.text = _join_texts(state.committed_text, window_text, state.language)


def _join_texts(committed: str, window: str, language: str) -> str:
    left = str(committed or "").strip()
    right = str(window or "").strip()
    if not left:
        return right
    if not right:
        return left
    joiner = "" if is_no_space_language(language) else " "
    return f"{left}{joiner}{right}"


def _infer_model_dtype(model: Qwen3ASRModel) -> mx.Dtype:
    weight = model.model.embed_tokens.weight
    if isinstance(weight, mx.array) and mx.issubdtype(weight.dtype, mx.floating):
        return weight.dtype
    return mx.float16


def _ensure_stream_runtime(
    state: StreamingState,
    model: Optional[Qwen3ASRModel],
) -> tuple[Qwen3ASRModel, Tokenizer, mx.Dtype]:
    if model is not None:
        model_obj = model
        state._model_obj = model_obj
        resolved = getattr(model_obj, "_resolved_model_path", None)
        if resolved is not None:
            state._resolved_model_path = str(resolved)
    else:
        if state._model_obj is None:
            model_obj, _ = _ModelHolder.get(state._model_path, dtype=state._dtype)
            state._model_obj = model_obj
            state._resolved_model_path = _ModelHolder.get_resolved_path(
                state._model_path,
                dtype=state._dtype,
            )
        model_obj = state._model_obj

    if model_obj is None:
        raise RuntimeError("Streaming runtime model resolution failed.")

    state._dtype = _infer_model_dtype(model_obj)
    if state._tokenizer is None:
        tok_path = state._resolved_model_path or state._model_path
        state._tokenizer = _TokenizerHolder.get(tok_path)
    return model_obj, state._tokenizer, state._dtype


def _build_position_ids(start: int, length: int, dtype: mx.Dtype = mx.int32) -> mx.array:
    positions = mx.arange(start, start + length, dtype=dtype)[None, :]
    return mx.stack([positions, positions, positions], axis=1)


def _decode_tokens_incremental(
    *,
    model: Qwen3ASRModel,
    cache: object,
    initial_logits: mx.array,
    start_pos: int,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    pos_dtype: mx.Dtype,
) -> list[int]:
    logits = initial_logits
    generated: list[int] = []
    position = int(start_pos)
    for _ in range(max_new_tokens):
        token = scalar_int(mx.argmax(logits.reshape(-1)))
        if token in eos_token_ids:
            break

        generated.append(token)
        if detect_repetition(generated):
            break

        next_ids = mx.array([[token]])
        next_pos = _build_position_ids(position, 1, dtype=pos_dtype)
        logits = model.step(
            input_ids=next_ids,
            position_ids=next_pos,
            cache=cache,
            validate_input_ids=False,
        )
        position += 1

    return generated


def _rollback_prefix_ids(
    raw_ids: list[int],
    unfixed_tokens: int,
    tokenizer: Tokenizer,
) -> list[int]:
    """Drop the last ``unfixed_tokens`` ids, backing off further if the cut splits a character."""
    end = max(0, len(raw_ids) - max(0, int(unfixed_tokens)))
    while end > 0 and _REPLACEMENT_CHAR in tokenizer.decode(raw_ids[:end]):
        end -= 1
    return list(raw_ids[:end])


def _decode_window(
    window_audio: np.ndarray,
    state: StreamingState,
    model: Optional[Qwen3ASRModel] = None,
) -> tuple[str, str]:
    """Encode the whole window and decode it with the rolled-back text prefix.

    Returns ``(text, language)`` for the window. Updates
    ``state._window_raw_ids`` with the raw generated ids (prefix + new) so the
    next call can roll back from them.
    """
    model_obj, tokenizer, dtype = _ensure_stream_runtime(state, model)

    mel, feature_lens = compute_features(
        np.asarray(window_audio, dtype=np.float32),
        sr=state.sample_rate,
    )
    audio_features, _ = model_obj.audio_tower(mel.astype(dtype), feature_lens)
    n_audio_tokens = int(audio_features.shape[1])

    prompt_tokens = tokenizer.build_prompt_tokens(
        n_audio_tokens=n_audio_tokens,
        language=state.forced_language,
        context=state.context,
    )
    if state._window_chunk_id < state.unfixed_chunk_num:
        prefix_ids: list[int] = []
    else:
        prefix_ids = _rollback_prefix_ids(
            state._window_raw_ids, state.unfixed_token_num, tokenizer
        )

    input_ids = mx.array([list(prompt_tokens) + prefix_ids])
    position_ids = _build_position_ids(0, int(input_ids.shape[1]))
    cache = model_obj.create_cache()
    logits = model_obj.prefill(
        input_ids=input_ids,
        audio_features=audio_features,
        position_ids=position_ids,
        cache=cache,
    )

    budget = int(state.max_new_tokens)
    if state._adaptive_token_budget:
        window_sec = float(len(window_audio)) / float(max(1, state.sample_rate))
        budget = max(budget, int(resolve_max_new_tokens(None, audio_duration_sec=window_sec)))
    eos_token_ids = tuple(getattr(tokenizer, "EOS_TOKEN_IDS", _DEFAULT_EOS_TOKEN_IDS))
    generated = _decode_tokens_incremental(
        model=model_obj,
        cache=cache,
        initial_logits=logits,
        start_pos=int(input_ids.shape[1]),
        max_new_tokens=budget,
        eos_token_ids=eos_token_ids,
        pos_dtype=position_ids.dtype,
    )
    # Release the window-sized tensors before the next chunk; long sessions
    # otherwise accumulate Metal allocations (see the offline chunk loop).
    del cache, logits, audio_features, mel
    mx.clear_cache()

    raw_ids = prefix_ids + generated
    state._window_raw_ids = raw_ids
    state._window_chunk_id += 1

    raw_text = tokenizer.decode(raw_ids)
    lang, text = parse_asr_output(raw_text, user_language=state.forced_language)
    return text, lang


def _select_decode_samples(state: StreamingState) -> int:
    """Choose how many buffered samples to decode in this turn."""
    chunk = int(state.chunk_size_samples)
    if len(state.buffer) < chunk:
        return 0
    if state.endpointing_mode != "energy":
        return chunk
    return _select_energy_endpoint_samples(state)


def _select_energy_endpoint_samples(state: StreamingState) -> int:
    """Find a low-energy boundary near the fixed chunk boundary.

    Keeps decode latency bounded by never selecting a boundary beyond the fixed
    chunk size and falling back to fixed-size behavior when no silence-like
    boundary is detected.
    """
    chunk = int(state.chunk_size_samples)
    if len(state.buffer) < chunk:
        return 0

    frame = max(1, int(state.endpoint_frame_samples))
    hop = max(1, frame // 2)
    min_chunk = max(1, int(state.endpoint_min_chunk_samples))
    lookback = max(0, int(state.endpoint_lookback_samples))

    search_start = max(min_chunk, chunk - lookback)
    search_end = chunk
    if search_end - search_start < frame:
        return chunk

    segment = state.buffer[search_start:search_end]
    seg_rms = _frame_rms(segment, frame, hop)
    ref_rms = _frame_rms(state.buffer[:chunk], frame, hop)
    if seg_rms.size == 0 or ref_rms.size == 0:
        return chunk

    threshold = float(np.quantile(ref_rms, 0.20))
    ref_median = float(np.median(ref_rms))
    if ref_median <= 1e-8:
        return chunk
    # Require a meaningful low-energy dip; otherwise keep fixed-size chunking.
    if float(np.min(seg_rms)) > (ref_median * 0.8):
        return chunk
    silence_like = np.where(seg_rms <= (threshold + 1e-8))[0]
    if silence_like.size == 0:
        return chunk

    # Pick the latest silence-like frame near the boundary (min latency impact).
    idx = int(silence_like[-1])
    boundary = search_start + (idx * hop) + (frame // 2)
    boundary = max(min_chunk, min(chunk, boundary))
    if boundary <= 0:
        return chunk
    return int(boundary)


def _frame_rms(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """Compute simple frame-wise RMS values for endpoint detection."""
    n = int(len(x))
    if n < frame:
        return np.array([], dtype=np.float32)
    vals = []
    for start in range(0, n - frame + 1, hop):
        seg = x[start : start + frame]
        vals.append(float(np.sqrt(np.mean(seg ** 2))))
    return np.asarray(vals, dtype=np.float32)


def _sanitize_stream_pcm(pcm: np.ndarray) -> np.ndarray:
    """Normalize streaming PCM to mono float32 waveform."""
    x = np.asarray(pcm)
    if x.ndim == 0:
        raise ValueError("pcm must be 1-D or 2-D audio array")
    if x.ndim > 2:
        raise ValueError(f"pcm must be 1-D or 2-D, got shape {x.shape}")

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        x = x.astype(np.float32)
        if info.min >= 0:
            midpoint = (info.max + 1) / 2.0
            x = (x - midpoint) / midpoint
        else:
            x = x / float(max(abs(info.min), info.max))
    else:
        x = x.astype(np.float32, copy=False)

    if x.ndim == 2:
        n0, n1 = int(x.shape[0]), int(x.shape[1])
        if n0 <= 8 and n1 <= 8:
            if n0 == n1:
                channel_axis = 1
            else:
                channel_axis = 0 if n0 < n1 else 1
        elif n0 <= 8:
            channel_axis = 0
        elif n1 <= 8:
            channel_axis = 1
        else:
            channel_axis = 1
        x = x.mean(axis=channel_axis)

    return np.asarray(x, dtype=np.float32)


def _split_text_units(text: str) -> tuple[list[str], str]:
    """Split text into rollback units and return the join delimiter."""
    if any(ch.isspace() for ch in text):
        return text.split(), " "
    return list(text), ""


def _split_stable_unstable(
    prev_stable: str,
    new_text: str,
    unfixed_tokens: int = UNFIXED_TOKEN_NUM,
) -> tuple[str, str]:
    """Split transcription into stable and unstable parts."""
    units, joiner = _split_text_units(new_text)

    if len(units) <= unfixed_tokens:
        return prev_stable, new_text

    stable_units = units[:-unfixed_tokens]
    unstable_units = units[-unfixed_tokens:]

    stable = joiner.join(stable_units)
    unstable = joiner.join(unstable_units)

    if len(stable) < len(prev_stable):
        stable = prev_stable

    return stable, unstable


def _append_chunk_text(current: str, addition: str, language: str) -> str:
    curr = str(current or "").strip()
    add = str(addition or "").strip()
    if not add:
        return curr
    if not curr:
        return add
    if curr == add or curr.endswith(add):
        return curr
    if add.startswith(curr):
        return add

    joiner = "" if is_no_space_language(language) else " "
    if joiner == " ":
        curr_units = curr.split()
        add_units = add.split()
    else:
        curr_units = list(curr)
        add_units = list(add)

    # If the new segment appears to be a full rewrite/superset of the current
    # text (same prefix and at least as long), prefer replacement over append.
    prefix_check = 3 if joiner == " " else 6
    pref_n = min(prefix_check, len(curr_units), len(add_units))
    if (
        pref_n > 0
        and curr_units[:pref_n] == add_units[:pref_n]
        and len(add_units) >= len(curr_units)
    ):
        return add

    max_overlap = min(len(curr_units), len(add_units))
    overlap = 0
    for k in range(max_overlap, 0, -1):
        if curr_units[-k:] == add_units[:k]:
            overlap = k
            break

    if overlap > 0:
        merged_units = curr_units + add_units[overlap:]
        return joiner.join(merged_units)

    return f"{curr}{joiner}{add}"
