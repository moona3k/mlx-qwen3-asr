"""High-level transcription pipeline for Qwen3-ASR."""

from __future__ import annotations

import asyncio
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union

import mlx.core as mx
import numpy as np

from .audio import SAMPLE_RATE, compute_features, load_audio_np
from .chunking import split_audio_into_chunks
from .config import DEFAULT_MODEL_ID
from .diarization import (
    DiarizationConfig,
    build_speaker_segments_from_turns,
    diarize_chunk_items,
    diarize_word_segments,
    infer_speaker_turns,
    validate_diarization_config,
)
from .forced_aligner import ForcedAligner
from .generate import GenerationConfig, generate, generate_speculative
from .load_models import _ModelHolder
from .model import Qwen3ASRModel
from .tokenizer import (
    Tokenizer,
    _TokenizerHolder,
    canonicalize_language,
    known_language_names,
    language_is_known,
    parse_asr_output,
)

AudioInput = Union[str, Path, np.ndarray, mx.array, tuple[np.ndarray, int]]
ProgressCallback = Callable[[dict[str, Any]], None]
CJK_LANG_ALIASES = {
    "chinese",
    "zh",
    "zh-cn",
    "zh-tw",
    "cantonese",
    "yue",
    "japanese",
    "ja",
    "jp",
    "korean",
    "ko",
    "kr",
}


@dataclass(frozen=True)
class TranscriptionResult:
    """Result of audio transcription.

    Attributes:
        text: The transcribed text
        language: Detected or forced language
        segments: Optional list of segments with timestamps [{text, start, end}, ...]
        chunks: Optional list of chunk-level transcripts
            [{text, start, end, chunk_index, language}, ...]
        speaker_segments: Optional list of speaker-attributed spans
            [{speaker, start, end, text}, ...]
    """
    text: str
    language: str
    segments: Optional[list[dict]] = None
    chunks: Optional[list[dict]] = None
    speaker_segments: Optional[list[dict]] = None


@dataclass(frozen=True)
class TranscribeOptions:
    """Shared transcribe runtime options used across sync/async/session entry points."""

    context: str = ""
    language: Optional[str] = None
    return_timestamps: bool = False
    diarize: bool = False
    diarization_num_speakers: Optional[int] = None
    diarization_min_speakers: int = 1
    diarization_max_speakers: int = 8
    return_chunks: bool = False
    forced_aligner: Optional[Union[str, ForcedAligner]] = None
    dtype: mx.Dtype = mx.float16
    max_new_tokens: int = 1024
    num_draft_tokens: int = 4
    verbose: bool = False
    on_progress: Optional[ProgressCallback] = None


def _join_chunk_texts(texts: list[str], language: str) -> str:
    """Join per-chunk text while preserving languages without whitespace delimiters."""
    if not texts:
        return ""
    normalized = (language or "").strip().lower()
    joiner = "" if normalized in CJK_LANG_ALIASES else " "
    return joiner.join(texts)


def _build_transcribe_options(
    *,
    context: str = "",
    language: Optional[str] = None,
    return_timestamps: bool = False,
    diarize: bool = False,
    diarization_num_speakers: Optional[int] = None,
    diarization_min_speakers: int = 1,
    diarization_max_speakers: int = 8,
    return_chunks: bool = False,
    forced_aligner: Optional[Union[str, ForcedAligner]] = None,
    dtype: mx.Dtype = mx.float16,
    max_new_tokens: int = 1024,
    num_draft_tokens: int = 4,
    verbose: bool = False,
    on_progress: Optional[ProgressCallback] = None,
) -> TranscribeOptions:
    return TranscribeOptions(
        context=context or "",
        language=language,
        return_timestamps=return_timestamps,
        diarize=diarize,
        diarization_num_speakers=diarization_num_speakers,
        diarization_min_speakers=diarization_min_speakers,
        diarization_max_speakers=diarization_max_speakers,
        return_chunks=return_chunks,
        forced_aligner=forced_aligner,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        num_draft_tokens=num_draft_tokens,
        verbose=verbose,
        on_progress=on_progress,
    )


def _transcribe_options_to_kwargs(
    options: TranscribeOptions,
    *,
    include_dtype: bool = True,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "context": options.context,
        "language": options.language,
        "return_timestamps": options.return_timestamps,
        "diarize": options.diarize,
        "diarization_num_speakers": options.diarization_num_speakers,
        "diarization_min_speakers": options.diarization_min_speakers,
        "diarization_max_speakers": options.diarization_max_speakers,
        "return_chunks": options.return_chunks,
        "forced_aligner": options.forced_aligner,
        "max_new_tokens": options.max_new_tokens,
        "num_draft_tokens": options.num_draft_tokens,
        "verbose": options.verbose,
        "on_progress": options.on_progress,
    }
    if include_dtype:
        kwargs["dtype"] = options.dtype
    return kwargs


def _resolve_model_components(
    model: Union[str, Qwen3ASRModel],
    *,
    dtype: mx.Dtype,
) -> tuple[Qwen3ASRModel, str]:
    if isinstance(model, str):
        model_obj, _ = _ModelHolder.get(model, dtype=dtype)
        model_path = _ModelHolder.get_resolved_path(model, dtype=dtype)
    else:
        model_obj = model
        model_path = DEFAULT_MODEL_ID
    return model_obj, model_path


def _resolve_runtime_options(
    options: TranscribeOptions,
) -> tuple[bool, Optional[ForcedAligner], Optional[DiarizationConfig]]:
    diarization_config = _resolve_diarization_config(
        diarize=options.diarize,
        diarization_num_speakers=options.diarization_num_speakers,
        diarization_min_speakers=options.diarization_min_speakers,
        diarization_max_speakers=options.diarization_max_speakers,
    )
    effective_return_timestamps = bool(
        options.return_timestamps or diarization_config is not None
    )
    aligner = _resolve_aligner(effective_return_timestamps, options.forced_aligner)
    return effective_return_timestamps, aligner, diarization_config


def transcribe(
    audio: AudioInput,
    *,
    model: Union[str, Qwen3ASRModel] = DEFAULT_MODEL_ID,
    draft_model: Optional[Union[str, Qwen3ASRModel]] = None,
    context: str = "",
    language: Optional[str] = None,
    return_timestamps: bool = False,
    diarize: bool = False,
    diarization_num_speakers: Optional[int] = None,
    diarization_min_speakers: int = 1,
    diarization_max_speakers: int = 8,
    return_chunks: bool = False,
    forced_aligner: Optional[Union[str, ForcedAligner]] = None,
    dtype: mx.Dtype = mx.float16,
    max_new_tokens: int = 1024,
    num_draft_tokens: int = 4,
    verbose: bool = False,
    on_progress: Optional[ProgressCallback] = None,
) -> TranscriptionResult:
    """Transcribe audio to text using Qwen3-ASR.

    Pipeline:
    1. Load audio -> mono 16kHz
    2. Split into chunks if > 1200s
    3. Per chunk: mel spectrogram -> encode -> build prompt -> decode -> parse
    4. Merge chunks
    5. Optional: forced aligner for word-level timestamps
    6. Optional: diarization speaker attribution

    Args:
        audio: Audio source - file path, numpy array, mx.array, or (array, sr) tuple
        model: Model name/path or pre-loaded model instance
        draft_model: Optional smaller model for speculative decoding.
            Must share tokenizer/vocabulary with ``model``.
        context: Domain-specific context string injected as the system prompt
            content. Provide space-separated terms (e.g., ``"交易 停滞"``) to
            bias the decoder toward domain vocabulary. Empty string by default,
            matching the official Qwen3-ASR implementation.
        language: Force language detection (e.g., "English", "Chinese")
        return_timestamps: Whether to return word-level timestamps.
        diarize: Whether to attach speaker labels to transcript output.
        diarization_num_speakers: Optional fixed speaker count override.
        diarization_min_speakers: Lower bound for auto speaker estimation.
        diarization_max_speakers: Upper bound for auto speaker estimation.
        return_chunks: Whether to return chunk-level transcript metadata.
        forced_aligner: Path/name of forced aligner model or prebuilt aligner object.
        dtype: Model dtype
        max_new_tokens: Maximum tokens to generate per chunk
        num_draft_tokens: Number of speculative draft tokens per verify step
        verbose: Print progress information
        on_progress: Optional callback receiving progress event dictionaries.

    Returns:
        TranscriptionResult with text, language, and optional segments
    """
    options = _build_transcribe_options(
        context=context,
        language=language,
        return_timestamps=return_timestamps,
        diarize=diarize,
        diarization_num_speakers=diarization_num_speakers,
        diarization_min_speakers=diarization_min_speakers,
        diarization_max_speakers=diarization_max_speakers,
        return_chunks=return_chunks,
        forced_aligner=forced_aligner,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        num_draft_tokens=num_draft_tokens,
        verbose=verbose,
        on_progress=on_progress,
    )
    _, aligner, diarization_config = _resolve_runtime_options(options)

    # Load model
    model_obj, model_path = _resolve_model_components(model, dtype=options.dtype)

    draft_model_obj = _resolve_draft_model(
        draft_model=draft_model,
        dtype=options.dtype,
        target_model=model_obj,
    )

    # Load tokenizer (cached by model path)
    tokenizer = _TokenizerHolder.get(model_path)

    # Load audio
    audio_np = _to_audio_np(audio)

    if options.verbose:
        duration = len(audio_np) / SAMPLE_RATE
        print(f"Audio duration: {duration:.1f}s ({len(audio_np)} samples)")

    return _transcribe_loaded_components(
        audio_np=audio_np,
        model_obj=model_obj,
        tokenizer=tokenizer,
        dtype=options.dtype,
        draft_model_obj=draft_model_obj,
        context=options.context,
        language=options.language,
        aligner=aligner,
        return_timestamps=options.return_timestamps,
        diarization_config=diarization_config,
        return_chunks=options.return_chunks,
        max_new_tokens=options.max_new_tokens,
        num_draft_tokens=options.num_draft_tokens,
        verbose=options.verbose,
        on_progress=options.on_progress,
    )


async def transcribe_async(
    audio: AudioInput,
    *,
    model: Union[str, Qwen3ASRModel] = DEFAULT_MODEL_ID,
    draft_model: Optional[Union[str, Qwen3ASRModel]] = None,
    context: str = "",
    language: Optional[str] = None,
    return_timestamps: bool = False,
    diarize: bool = False,
    diarization_num_speakers: Optional[int] = None,
    diarization_min_speakers: int = 1,
    diarization_max_speakers: int = 8,
    return_chunks: bool = False,
    forced_aligner: Optional[Union[str, ForcedAligner]] = None,
    dtype: mx.Dtype = mx.float16,
    max_new_tokens: int = 1024,
    num_draft_tokens: int = 4,
    verbose: bool = False,
    on_progress: Optional[ProgressCallback] = None,
) -> TranscriptionResult:
    """Async wrapper for ``transcribe`` using ``asyncio.to_thread``."""
    options = _build_transcribe_options(
        context=context,
        language=language,
        return_timestamps=return_timestamps,
        diarize=diarize,
        diarization_num_speakers=diarization_num_speakers,
        diarization_min_speakers=diarization_min_speakers,
        diarization_max_speakers=diarization_max_speakers,
        return_chunks=return_chunks,
        forced_aligner=forced_aligner,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        num_draft_tokens=num_draft_tokens,
        verbose=verbose,
        on_progress=on_progress,
    )
    return await asyncio.to_thread(
        transcribe,
        audio,
        model=model,
        draft_model=draft_model,
        **_transcribe_options_to_kwargs(options),
    )


def transcribe_batch(
    audios: list[AudioInput],
    *,
    model: Union[str, Qwen3ASRModel] = DEFAULT_MODEL_ID,
    draft_model: Optional[Union[str, Qwen3ASRModel]] = None,
    context: Union[str, list[str]] = "",
    language: Optional[str] = None,
    return_timestamps: bool = False,
    diarize: bool = False,
    diarization_num_speakers: Optional[int] = None,
    diarization_min_speakers: int = 1,
    diarization_max_speakers: int = 8,
    return_chunks: bool = False,
    forced_aligner: Optional[Union[str, ForcedAligner]] = None,
    dtype: mx.Dtype = mx.float16,
    max_new_tokens: int = 1024,
    num_draft_tokens: int = 4,
    verbose: bool = False,
    on_progress: Optional[ProgressCallback] = None,
) -> list[TranscriptionResult]:
    """Transcribe multiple audio inputs while reusing loaded model/tokenizer.

    Args:
        context: Domain-specific context string or list of per-audio context
            strings. When a single string is given it applies to all audios.
            When a list is given it must have the same length as *audios*.
    """
    if not audios:
        return []

    # Normalize context to per-audio list.
    if isinstance(context, str):
        contexts = [context or ""] * len(audios)
    else:
        if len(context) != len(audios):
            raise ValueError(
                f"context list length ({len(context)}) must match "
                f"audios length ({len(audios)})"
            )
        contexts = [c or "" for c in context]

    options = _build_transcribe_options(
        language=language,
        return_timestamps=return_timestamps,
        diarize=diarize,
        diarization_num_speakers=diarization_num_speakers,
        diarization_min_speakers=diarization_min_speakers,
        diarization_max_speakers=diarization_max_speakers,
        return_chunks=return_chunks,
        forced_aligner=forced_aligner,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        num_draft_tokens=num_draft_tokens,
        verbose=verbose,
        on_progress=on_progress,
    )
    _, aligner, diarization_config = _resolve_runtime_options(options)
    model_obj, model_path = _resolve_model_components(model, dtype=options.dtype)

    draft_model_obj = _resolve_draft_model(
        draft_model=draft_model,
        dtype=options.dtype,
        target_model=model_obj,
    )
    tokenizer = _TokenizerHolder.get(model_path)

    outputs: list[TranscriptionResult] = []
    total = len(audios)
    for index, audio in enumerate(audios, start=1):
        if options.on_progress is not None:
            _emit_progress(
                options.on_progress,
                {
                    "event": "batch_file_started",
                    "file_index": index,
                    "file_total": total,
                },
            )
        audio_np = _to_audio_np(audio)
        result = _transcribe_loaded_components(
            audio_np=audio_np,
            model_obj=model_obj,
            tokenizer=tokenizer,
            dtype=options.dtype,
            draft_model_obj=draft_model_obj,
            context=contexts[index - 1],
            language=options.language,
            aligner=aligner,
            return_timestamps=options.return_timestamps,
            diarization_config=diarization_config,
            return_chunks=options.return_chunks,
            max_new_tokens=options.max_new_tokens,
            num_draft_tokens=options.num_draft_tokens,
            verbose=options.verbose,
            on_progress=_batch_progress_adapter(
                on_progress=options.on_progress,
                file_index=index,
                file_total=total,
            ),
        )
        outputs.append(result)
        if options.on_progress is not None:
            _emit_progress(
                options.on_progress,
                {
                    "event": "batch_file_completed",
                    "file_index": index,
                    "file_total": total,
                },
            )
    return outputs


async def transcribe_batch_async(
    audios: list[AudioInput],
    *,
    model: Union[str, Qwen3ASRModel] = DEFAULT_MODEL_ID,
    draft_model: Optional[Union[str, Qwen3ASRModel]] = None,
    context: Union[str, list[str]] = "",
    language: Optional[str] = None,
    return_timestamps: bool = False,
    diarize: bool = False,
    diarization_num_speakers: Optional[int] = None,
    diarization_min_speakers: int = 1,
    diarization_max_speakers: int = 8,
    return_chunks: bool = False,
    forced_aligner: Optional[Union[str, ForcedAligner]] = None,
    dtype: mx.Dtype = mx.float16,
    max_new_tokens: int = 1024,
    num_draft_tokens: int = 4,
    verbose: bool = False,
    on_progress: Optional[ProgressCallback] = None,
) -> list[TranscriptionResult]:
    """Async wrapper for ``transcribe_batch`` using ``asyncio.to_thread``."""
    options = _build_transcribe_options(
        language=language,
        return_timestamps=return_timestamps,
        diarize=diarize,
        diarization_num_speakers=diarization_num_speakers,
        diarization_min_speakers=diarization_min_speakers,
        diarization_max_speakers=diarization_max_speakers,
        return_chunks=return_chunks,
        forced_aligner=forced_aligner,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        num_draft_tokens=num_draft_tokens,
        verbose=verbose,
        on_progress=on_progress,
    )
    kwargs = _transcribe_options_to_kwargs(options)
    # Override scalar context from options with the original (possibly list)
    # context so per-audio context propagates correctly.
    kwargs["context"] = context
    return await asyncio.to_thread(
        transcribe_batch,
        audios,
        model=model,
        draft_model=draft_model,
        **kwargs,
    )


def _to_audio_np(audio: AudioInput) -> np.ndarray:
    """Convert supported audio inputs to float32 numpy waveform at 16kHz."""
    if isinstance(audio, mx.array):
        return load_audio_np(np.array(audio), sr=SAMPLE_RATE)
    return load_audio_np(audio, sr=SAMPLE_RATE)


def _warn_if_unsupported_language(language: Optional[str], model_obj: Qwen3ASRModel) -> None:
    if language is None:
        return
    raw = str(language).strip()
    if not raw:
        return
    if language_is_known(raw):
        return

    # Preserve flexibility for model-provided language-code lists in config.
    support_languages = getattr(model_obj.config, "support_languages", None) or []
    normalized = raw.casefold().replace("_", "-")
    supported_codes = {
        str(code).strip().casefold().replace("_", "-")
        for code in support_languages
        if str(code).strip()
    }
    if normalized in supported_codes:
        return

    known = ", ".join(known_language_names())
    warnings.warn(
        f"Unsupported language '{raw}'. Known language aliases map to: {known}. "
        "Continuing anyway; recognition quality may degrade.",
        stacklevel=3,
    )


def _resolve_aligner(
    return_timestamps: bool,
    forced_aligner: Optional[Union[str, ForcedAligner]],
) -> Optional[ForcedAligner]:
    if not return_timestamps:
        return None
    if forced_aligner is None:
        return ForcedAligner()
    if isinstance(forced_aligner, str):
        return ForcedAligner(forced_aligner)
    return forced_aligner


def _resolve_draft_model(
    draft_model: Optional[Union[str, Qwen3ASRModel]],
    dtype: mx.Dtype,
    target_model: Qwen3ASRModel,
) -> Optional[Qwen3ASRModel]:
    if draft_model is None:
        return None
    if isinstance(draft_model, str):
        draft_model_obj, _ = _ModelHolder.get(draft_model, dtype=dtype)
    else:
        draft_model_obj = draft_model

    if (
        draft_model_obj.config.text_config.vocab_size
        != target_model.config.text_config.vocab_size
    ):
        raise ValueError(
            "Draft model vocabulary size mismatch: "
            f"target={target_model.config.text_config.vocab_size}, "
            f"draft={draft_model_obj.config.text_config.vocab_size}"
        )
    if draft_model_obj.audio_token_id != target_model.audio_token_id:
        raise ValueError(
            "Draft model audio token mismatch: "
            f"target={target_model.audio_token_id}, "
            f"draft={draft_model_obj.audio_token_id}"
        )
    return draft_model_obj


def _resolve_diarization_config(
    *,
    diarize: bool,
    diarization_num_speakers: Optional[int],
    diarization_min_speakers: int,
    diarization_max_speakers: int,
) -> Optional[DiarizationConfig]:
    if not diarize:
        if diarization_num_speakers is not None:
            raise ValueError(
                "diarization_num_speakers requires diarize=True."
            )
        return None
    return validate_diarization_config(
        num_speakers=diarization_num_speakers,
        min_speakers=diarization_min_speakers,
        max_speakers=diarization_max_speakers,
    )


def _transcribe_loaded_components(
    *,
    audio_np: np.ndarray,
    model_obj: Qwen3ASRModel,
    tokenizer: Tokenizer,
    dtype: mx.Dtype,
    draft_model_obj: Optional[Qwen3ASRModel],
    context: str = "",
    language: Optional[str],
    aligner: Optional[ForcedAligner],
    return_timestamps: bool,
    diarization_config: Optional[DiarizationConfig],
    return_chunks: bool,
    max_new_tokens: int,
    num_draft_tokens: int,
    verbose: bool,
    on_progress: Optional[ProgressCallback] = None,
) -> TranscriptionResult:
    """Transcribe using already-loaded model/tokenizer components."""
    chunks = split_audio_into_chunks(audio_np, sr=SAMPLE_RATE)
    forced_language = canonicalize_language(language)
    _warn_if_unsupported_language(language, model_obj)
    total_audio_sec = float(len(audio_np) / SAMPLE_RATE) if len(audio_np) > 0 else 0.0

    if verbose and len(chunks) > 1:
        print(f"Split into {len(chunks)} chunks")

    all_texts = []
    all_segments: list[dict] = []
    all_chunk_items: list[dict] = []
    detected_language = forced_language or "unknown"
    processed_sec = 0.0
    needs_alignment = bool(return_timestamps or diarization_config is not None)

    _emit_progress(
        on_progress,
        {
            "event": "chunks_prepared",
            "total_chunks": len(chunks),
            "audio_duration_sec": total_audio_sec,
            "progress": 0.0,
        },
    )

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        num_draft_tokens=num_draft_tokens,
    )

    for chunk_idx, (chunk_audio, offset) in enumerate(chunks):
        chunk_duration_sec = float(len(chunk_audio) / SAMPLE_RATE)
        _emit_progress(
            on_progress,
            {
                "event": "chunk_started",
                "chunk_index": chunk_idx + 1,
                "total_chunks": len(chunks),
                "chunk_offset_sec": float(offset),
                "chunk_duration_sec": chunk_duration_sec,
                "audio_duration_sec": total_audio_sec,
                "processed_audio_sec": processed_sec,
                "progress": _safe_progress(processed_sec, total_audio_sec),
            },
        )
        if verbose:
            print(
                f"Processing chunk {chunk_idx + 1}/{len(chunks)} "
                f"(offset={offset:.1f}s, duration={len(chunk_audio)/SAMPLE_RATE:.1f}s)"
            )

        mel, feature_lens = compute_features(chunk_audio)
        audio_features, _ = model_obj.audio_tower(mel.astype(dtype), feature_lens)
        draft_audio_features = None
        if draft_model_obj is not None:
            draft_audio_features, _ = draft_model_obj.audio_tower(
                mel.astype(dtype), feature_lens
            )
        n_audio_tokens = audio_features.shape[1]

        if verbose:
            print(f"  Audio features: {n_audio_tokens} tokens")

        prompt_tokens = tokenizer.build_prompt_tokens(
            n_audio_tokens=n_audio_tokens,
            language=forced_language,
            context=context,
        )
        input_ids = mx.array([prompt_tokens])

        seq_len = input_ids.shape[1]
        positions = mx.arange(seq_len)[None, :]
        position_ids = mx.stack([positions, positions, positions], axis=1)

        if draft_model_obj is None:
            output_tokens = generate(
                model=model_obj,
                input_ids=input_ids,
                audio_features=audio_features,
                position_ids=position_ids,
                config=gen_config,
            )
        else:
            output_tokens = generate_speculative(
                model=model_obj,
                draft_model=draft_model_obj,
                input_ids=input_ids,
                audio_features=audio_features,
                draft_audio_features=draft_audio_features,
                position_ids=position_ids,
                config=gen_config,
            )

        raw_text = tokenizer.decode(output_tokens)
        lang, text = parse_asr_output(raw_text, user_language=forced_language)
        lang = canonicalize_language(lang) or lang

        if detected_language == "unknown":
            detected_language = lang

        all_texts.append(text)
        if return_chunks:
            all_chunk_items.append(
                {
                    "text": text,
                    "start": float(offset),
                    "end": float(offset + chunk_duration_sec),
                    "chunk_index": int(chunk_idx),
                    "language": lang,
                }
            )

        if needs_alignment and aligner is not None and text.strip():
            align_lang = forced_language or lang
            if align_lang and align_lang != "unknown":
                aligned = aligner.align(chunk_audio, text, align_lang)
                for item in aligned:
                    all_segments.append(
                        {
                            "text": item.text,
                            "start": item.start_time + offset,
                            "end": item.end_time + offset,
                        }
                    )

        if verbose:
            print(f"  [{lang}] {text[:100]}{'...' if len(text) > 100 else ''}")

        processed_sec += chunk_duration_sec
        _emit_progress(
            on_progress,
            {
                "event": "chunk_completed",
                "chunk_index": chunk_idx + 1,
                "total_chunks": len(chunks),
                "chunk_offset_sec": float(offset),
                "chunk_duration_sec": chunk_duration_sec,
                "audio_duration_sec": total_audio_sec,
                "processed_audio_sec": processed_sec,
                "progress": _safe_progress(processed_sec, total_audio_sec),
                "language": lang,
                "segment_count": len(all_segments) if return_timestamps else 0,
            },
        )

    final_language = canonicalize_language(detected_language) or detected_language
    out_segments: Optional[list[dict]] = all_segments if return_timestamps else None
    speaker_segments: Optional[list[dict]] = None
    labeled_segments: list[dict] = []
    if diarization_config is not None:
        speaker_turns = infer_speaker_turns(
            audio_np,
            sr=SAMPLE_RATE,
            config=diarization_config,
        )
        labeled_segments, speaker_segments = diarize_word_segments(
            all_segments,
            config=diarization_config,
            speaker_turns=speaker_turns,
        )
        speaker_segments = build_speaker_segments_from_turns(
            speaker_turns=speaker_turns,
            word_segments=labeled_segments or all_segments,
        )
        if return_timestamps:
            out_segments = labeled_segments
        if not speaker_segments:
            fallback_chunks = all_chunk_items
            if not fallback_chunks:
                fallback_chunks = [
                    {
                        "text": _join_chunk_texts(all_texts, final_language),
                        "start": 0.0,
                        "end": total_audio_sec,
                    }
                ]
            speaker_segments = diarize_chunk_items(
                fallback_chunks,
                config=diarization_config,
                speaker_turns=speaker_turns,
            )
        _emit_progress(
            on_progress,
            {
                "event": "diarization_completed",
                "speaker_segment_count": len(speaker_segments or []),
                "word_segment_count": len(labeled_segments),
            },
        )

    _emit_progress(
        on_progress,
        {
            "event": "completed",
            "total_chunks": len(chunks),
            "audio_duration_sec": total_audio_sec,
            "processed_audio_sec": processed_sec,
            "progress": 1.0,
            "language": final_language,
        },
    )
    return TranscriptionResult(
        text=_join_chunk_texts(all_texts, final_language),
        language=final_language,
        segments=out_segments,
        chunks=all_chunk_items if return_chunks else None,
        speaker_segments=speaker_segments,
    )


def _safe_progress(processed_audio_sec: float, total_audio_sec: float) -> float:
    if total_audio_sec <= 0:
        return 0.0
    return float(min(max(processed_audio_sec / total_audio_sec, 0.0), 1.0))


def _emit_progress(
    on_progress: Optional[ProgressCallback],
    payload: dict[str, Any],
) -> None:
    if on_progress is None:
        return
    on_progress(dict(payload))


def _batch_progress_adapter(
    *,
    on_progress: Optional[ProgressCallback],
    file_index: int,
    file_total: int,
) -> Optional[ProgressCallback]:
    if on_progress is None:
        return None

    def _callback(payload: dict[str, Any]) -> None:
        enriched = dict(payload)
        enriched["file_index"] = file_index
        enriched["file_total"] = file_total
        on_progress(enriched)

    return _callback
