"""Explicit session API for model/tokenizer ownership."""

from __future__ import annotations

import asyncio
from typing import Any, Optional, Union

import mlx.core as mx
import numpy as np

from . import streaming as streaming_mod
from .config import DEFAULT_MODEL_ID
from .forced_aligner import ForcedAligner
from .load_models import _resolve_path, load_model
from .model import Qwen3ASRModel
from .tokenizer import Tokenizer
from .transcribe import (
    AudioInput,
    ProgressCallback,
    TranscriptionResult,
    _build_transcribe_options,
    _resolve_aligner,
    _resolve_diarization_config,
    _resolve_draft_model,
    _to_audio_np,
    _transcribe_loaded_components,
    _transcribe_options_to_kwargs,
)


class Session:
    """Explicit transcription session holding model and tokenizer state.

    This is the power-user path that avoids hidden process-global holders.
    """

    def __init__(
        self,
        model: Union[str, Qwen3ASRModel] = DEFAULT_MODEL_ID,
        *,
        dtype: mx.Dtype = mx.float16,
        tokenizer_model: Optional[str] = None,
    ) -> None:
        self.dtype = dtype

        if isinstance(model, str):
            self.model_id = model
            self.model, self.config = load_model(model, dtype=dtype)
            resolved_path = getattr(self.model, "_resolved_model_path", None)
            tok_path = tokenizer_model or resolved_path or str(_resolve_path(model))
        else:
            source_model_id = getattr(model, "_source_model_id", None)
            resolved_model_path = getattr(model, "_resolved_model_path", None)
            tok_path = tokenizer_model or resolved_model_path or source_model_id
            if tok_path is None:
                raise ValueError(
                    "tokenizer_model is required when passing a pre-loaded model "
                    "without source metadata."
                )
            self.model_id = str(source_model_id or tok_path)
            self.model = model
            self.config = None
            tok_path = str(tok_path)

        self.tokenizer = Tokenizer(tok_path)

    def transcribe(
        self,
        audio: AudioInput,
        *,
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
        max_new_tokens: int = 1024,
        num_draft_tokens: int = 4,
        verbose: bool = False,
        on_progress: Optional[ProgressCallback] = None,
    ) -> TranscriptionResult:
        """Transcribe audio using this session's loaded model/tokenizer."""
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
            dtype=self.dtype,
            max_new_tokens=max_new_tokens,
            num_draft_tokens=num_draft_tokens,
            verbose=verbose,
            on_progress=on_progress,
        )
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
        draft_model_obj = _resolve_draft_model(
            draft_model=draft_model,
            dtype=self.dtype,
            target_model=self.model,
        )
        audio_np = _to_audio_np(audio)
        return _transcribe_loaded_components(
            audio_np=audio_np,
            model_obj=self.model,
            tokenizer=self.tokenizer,
            dtype=self.dtype,
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
        self,
        audio: AudioInput,
        *,
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
            dtype=self.dtype,
            max_new_tokens=max_new_tokens,
            num_draft_tokens=num_draft_tokens,
            verbose=verbose,
            on_progress=on_progress,
        )
        return await asyncio.to_thread(
            self.transcribe,
            audio,
            draft_model=draft_model,
            **_transcribe_options_to_kwargs(options, include_dtype=False),
        )

    def init_streaming(
        self,
        *,
        context: str = "",
        language: Optional[str] = None,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
        chunk_size_sec: float = 2.0,
        max_context_sec: float = 30.0,
        sample_rate: int = 16000,
        max_new_tokens: int = 1024,
        finalization_mode: str = "accuracy",
        enable_tail_refine: Optional[bool] = None,
        endpointing_mode: str = "fixed",
        endpoint_lookback_sec: float = 0.3,
        endpoint_frame_ms: float = 20.0,
        endpoint_min_chunk_sec: float = 0.5,
    ) -> streaming_mod.StreamingState:
        """Create streaming state bound to this session's model settings."""
        return streaming_mod.init_streaming(
            model=self.model_id,
            context=context,
            language=language,
            unfixed_chunk_num=unfixed_chunk_num,
            unfixed_token_num=unfixed_token_num,
            chunk_size_sec=chunk_size_sec,
            max_context_sec=max_context_sec,
            sample_rate=sample_rate,
            dtype=self.dtype,
            max_new_tokens=max_new_tokens,
            finalization_mode=finalization_mode,
            enable_tail_refine=enable_tail_refine,
            endpointing_mode=endpointing_mode,
            endpoint_lookback_sec=endpoint_lookback_sec,
            endpoint_frame_ms=endpoint_frame_ms,
            endpoint_min_chunk_sec=endpoint_min_chunk_sec,
        )

    def feed_audio(
        self,
        pcm: np.ndarray,
        state: streaming_mod.StreamingState,
    ) -> streaming_mod.StreamingState:
        """Feed streaming audio using this session's loaded model."""
        return streaming_mod.feed_audio(pcm=pcm, state=state, model=self.model)

    def finish_streaming(
        self,
        state: streaming_mod.StreamingState,
    ) -> streaming_mod.StreamingState:
        """Finalize streaming decode using this session's loaded model."""
        return streaming_mod.finish_streaming(state=state, model=self.model)

    @property
    def model_info(self) -> dict[str, Any]:
        """Return lightweight runtime metadata for this session."""
        cfg = getattr(self.model, "config", None)
        text_cfg = getattr(cfg, "text_config", None)
        return {
            "model_id": self.model_id,
            "resolved_model_path": getattr(self.model, "_resolved_model_path", None),
            "dtype": str(self.dtype),
            "vocab_size": getattr(text_cfg, "vocab_size", None),
            "support_languages": list(getattr(cfg, "support_languages", []) or []),
        }
