"""Diarization helpers built around optional pyannote integration.

This module keeps the existing transcript attribution contract while delegating
speaker-turn inference to pyannote when ``diarize=True``.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from inspect import signature
from typing import Any, Optional

import numpy as np

DEFAULT_SPEAKER_LABEL = "SPEAKER_00"
DEFAULT_PYANNOTE_MODEL_ID = "pyannote/speaker-diarization-community-1"


@dataclass(frozen=True)
class DiarizationConfig:
    """Configuration for runtime diarization behavior."""

    num_speakers: Optional[int] = None
    min_speakers: int = 1
    max_speakers: int = 8


def validate_diarization_config(
    *,
    num_speakers: Optional[int],
    min_speakers: int,
    max_speakers: int,
) -> DiarizationConfig:
    """Validate diarization configuration values."""
    if num_speakers is not None and num_speakers < 1:
        raise ValueError("diarization_num_speakers must be >= 1.")
    if min_speakers < 1:
        raise ValueError("diarization_min_speakers must be >= 1.")
    if max_speakers < min_speakers:
        raise ValueError(
            "diarization_max_speakers must be >= diarization_min_speakers."
        )
    return DiarizationConfig(
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )


def infer_speaker_turns(
    audio: np.ndarray,
    *,
    sr: int,
    config: DiarizationConfig,
    _pipeline: Optional[object] = None,
) -> list[dict]:
    """Infer speaker turns using pyannote pipeline inference."""
    if sr <= 0:
        raise ValueError("sr must be > 0.")
    if audio.size == 0:
        return []

    pipeline = _pipeline or _load_pyannote_pipeline()
    kwargs = _speaker_count_kwargs(config)
    audio_np = np.asarray(audio, dtype=np.float32).reshape(-1)

    try:
        diarization = pipeline(_pyannote_input(audio_np, sr), **kwargs)
    except TypeError as exc:
        # Some pyannote versions reject speaker-count kwargs.
        if not _is_retryable_kwargs_type_error(exc):
            raise _diarization_runtime_error(exc) from exc
        warnings.warn(
            "pyannote backend rejected speaker-count kwargs; retrying without them.",
            stacklevel=2,
        )
        try:
            diarization = pipeline(_pyannote_input(audio_np, sr))
        except Exception as exc:
            raise _diarization_runtime_error(exc) from exc
    except Exception as exc:
        raise _diarization_runtime_error(exc) from exc

    turns = _annotation_to_turns(diarization, duration=float(audio_np.shape[0] / sr))
    if not turns:
        return [
            {
                "speaker": DEFAULT_SPEAKER_LABEL,
                "start": 0.0,
                "end": float(audio_np.shape[0] / sr),
            }
        ]
    return turns


def diarize_word_segments(
    segments: list[dict],
    *,
    config: DiarizationConfig,
    speaker_turns: Optional[list[dict]] = None,
) -> tuple[list[dict], list[dict]]:
    """Assign speaker labels to word-level segments."""
    _ = config
    if not segments:
        return [], []

    turns = speaker_turns or []
    labeled: list[dict] = []
    for seg in segments:
        item = dict(seg)
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start))
        item["speaker"] = _speaker_for_interval(start, end, turns)
        labeled.append(item)
    return labeled, _merge_speaker_segments(labeled)


def build_speaker_segments_from_turns(
    *,
    speaker_turns: list[dict],
    word_segments: Optional[list[dict]] = None,
    max_gap_sec: float = 0.2,
) -> list[dict]:
    """Build transcript speaker segments from diarization turns.

    Unlike ``_merge_speaker_segments``, this keeps empty-text turns so output
    time coverage tracks diarization output even when ASR words are sparse.
    """
    if not speaker_turns:
        return []

    turns = sorted(
        (
            {
                "speaker": str(t.get("speaker", DEFAULT_SPEAKER_LABEL)),
                "start": float(t.get("start", 0.0)),
                "end": float(t.get("end", t.get("start", 0.0))),
            }
            for t in speaker_turns
        ),
        key=lambda x: (x["start"], x["end"]),
    )
    words = sorted(
        (dict(w) for w in (word_segments or [])),
        key=lambda x: (float(x.get("start", 0.0)), float(x.get("end", 0.0))),
    )

    out: list[dict] = []
    wi = 0
    for turn in turns:
        start = max(0.0, float(turn["start"]))
        end = max(start, float(turn["end"]))
        speaker = str(turn["speaker"])
        while wi < len(words) and float(words[wi].get("end", 0.0)) <= start:
            wi += 1

        text_parts: list[str] = []
        wj = wi
        while wj < len(words) and float(words[wj].get("start", 0.0)) < end:
            w = words[wj]
            ws = float(w.get("start", 0.0))
            we = float(w.get("end", ws))
            overlap = max(0.0, min(end, we) - max(start, ws))
            if overlap > 0.0:
                token = str(w.get("text", "")).strip()
                if token:
                    text_parts.append(token)
            wj += 1

        out.append(
            {
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": " ".join(text_parts).strip(),
            }
        )

    merged: list[dict] = []
    for item in out:
        if not merged:
            merged.append(dict(item))
            continue
        prev = merged[-1]
        gap = float(item["start"]) - float(prev["end"])
        if prev["speaker"] == item["speaker"] and gap <= max_gap_sec:
            prev["end"] = max(float(prev["end"]), float(item["end"]))
            prev_text = str(prev.get("text", "")).strip()
            next_text = str(item.get("text", "")).strip()
            if prev_text and next_text:
                prev["text"] = f"{prev_text} {next_text}".strip()
            elif next_text:
                prev["text"] = next_text
            else:
                prev["text"] = prev_text
        else:
            merged.append(dict(item))
    return merged


def diarize_chunk_items(
    chunks: list[dict],
    *,
    config: DiarizationConfig,
    speaker_turns: Optional[list[dict]] = None,
) -> list[dict]:
    """Fallback speaker segments derived from chunk-level transcript items."""
    _ = config
    if not chunks:
        return []
    turns = speaker_turns or []
    items: list[dict] = []
    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        start = float(chunk.get("start", 0.0))
        end = float(chunk.get("end", start))
        items.append(
            {
                "speaker": _speaker_for_interval(start, end, turns),
                "start": start,
                "end": max(end, start),
                "text": text,
            }
        )
    return _merge_speaker_segments(items)


_PYANNOTE_PIPELINE_CACHE: dict[tuple[str, str], object] = {}


def _load_pyannote_pipeline() -> object:
    model_id = os.environ.get("PYANNOTE_MODEL_ID", DEFAULT_PYANNOTE_MODEL_ID)
    token = (
        os.environ.get("PYANNOTE_AUTH_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_TOKEN")
        or ""
    )
    key = (model_id, token)
    cached = _PYANNOTE_PIPELINE_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise ImportError(
            "Diarization requires optional dependency 'pyannote.audio'. "
            "Install with: pip install \"mlx-qwen3-asr[diarize]\""
        ) from exc

    kwargs = _pyannote_auth_kwargs(Pipeline.from_pretrained, token)
    try:
        pipeline = Pipeline.from_pretrained(model_id, **kwargs)
    except Exception as exc:
        raise _pyannote_pipeline_init_error(model_id, _format_exception(exc)) from exc
    if pipeline is None:
        raise _pyannote_pipeline_init_error(
            model_id,
            "Pipeline.from_pretrained returned None; pyannote could not download "
            "or load the pipeline configuration.",
        )
    _PYANNOTE_PIPELINE_CACHE[key] = pipeline
    return pipeline


def _pyannote_pipeline_init_error(model_id: str, root_cause: str) -> RuntimeError:
    return RuntimeError(
        f"Failed to initialize pyannote pipeline '{model_id}'. "
        "If this model is gated, accept its Hugging Face terms and set "
        "PYANNOTE_AUTH_TOKEN (or HF_TOKEN). You can also override "
        "PYANNOTE_MODEL_ID. "
        f"Root cause: {root_cause}"
    )


def _pyannote_auth_kwargs(from_pretrained: Any, token: str) -> dict[str, Any]:
    """Build auth kwargs for pyannote 4 while tolerating pyannote 3-style APIs."""
    if not token:
        return {}
    try:
        params = signature(from_pretrained).parameters
    except (TypeError, ValueError):
        return {"token": token}

    if "use_auth_token" in params and "token" not in params:
        return {"use_auth_token": token}
    return {"token": token}


def _pyannote_input(audio_np: np.ndarray, sr: int) -> dict[str, Any]:
    try:
        # Intentional: diarization is optional and this torch import is gated
        # behind `[diarize]` + `diarize=True`, never in the core ASR path.
        import torch
    except ImportError as exc:
        raise ImportError(
            "Diarization requires PyTorch via pyannote dependencies. "
            "Install with: pip install \"mlx-qwen3-asr[diarize]\""
        ) from exc

    waveform = torch.from_numpy(audio_np).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": sr}


def _is_retryable_kwargs_type_error(exc: TypeError) -> bool:
    msg = str(exc).casefold()
    if "unexpected keyword" not in msg:
        return False
    return any(
        key in msg
        for key in (
            "num_speakers",
            "min_speakers",
            "max_speakers",
            "speaker",
        )
    )


def _speaker_count_kwargs(config: DiarizationConfig) -> dict[str, int]:
    if config.num_speakers is not None:
        return {"num_speakers": int(config.num_speakers)}
    return {
        "min_speakers": int(config.min_speakers),
        "max_speakers": int(config.max_speakers),
    }


def _diarization_runtime_error(exc: Exception) -> RuntimeError:
    return RuntimeError(
        "pyannote diarization inference failed. "
        "Verify the installed '[diarize]' extra and any required Hugging Face "
        "token/terms for your diarization model. "
        f"Root cause: {_format_exception(exc)}"
    )


def _format_exception(exc: Exception) -> str:
    msg = str(exc).strip()
    if msg:
        return f"{type(exc).__name__}: {msg}"
    return type(exc).__name__


def _pyannote_annotation_candidates(output: Any) -> list[Any]:
    """Return pyannote 4 annotation candidates in timestamp-attribution order."""
    # Community-1 exposes exclusive diarization specifically to simplify
    # reconciliation with transcription timestamps, which matches this module's
    # single-speaker attribution contract.
    candidates: list[Any] = []
    exclusive = getattr(output, "exclusive_speaker_diarization", None)
    if exclusive is not None:
        candidates.append(exclusive)
    regular = getattr(output, "speaker_diarization", None)
    if regular is not None:
        candidates.append(regular)
    candidates.append(output)
    return candidates


def _annotation_to_turns(annotation: Any, *, duration: float) -> list[dict]:
    if annotation is None:
        return []

    raw: list[tuple[str, float, float]] = []
    for candidate in _pyannote_annotation_candidates(annotation):
        raw = _raw_annotation_turns(candidate)
        if raw:
            break

    if not raw:
        return []

    raw.sort(key=lambda x: (x[1], x[2]))
    label_map: dict[str, str] = {}
    turns: list[dict] = []

    for label, start, end in raw:
        if label not in label_map:
            label_map[label] = f"SPEAKER_{len(label_map):02d}"
        speaker = label_map[label]
        turns.append({"speaker": speaker, "start": max(0.0, start), "end": min(duration, end)})

    return _merge_speaker_turns(turns)


def _raw_annotation_turns(annotation: Any) -> list[tuple[str, float, float]]:
    raw: list[tuple[str, float, float]] = []

    if hasattr(annotation, "itertracks"):
        for segment, _, label in annotation.itertracks(yield_label=True):
            start = float(getattr(segment, "start", 0.0))
            end = float(getattr(segment, "end", start))
            if end > start:
                raw.append((str(label), start, end))
    elif isinstance(annotation, list):
        for item in annotation:
            if not isinstance(item, dict):
                continue
            label = str(item.get("speaker", item.get("label", DEFAULT_SPEAKER_LABEL)))
            start = float(item.get("start", 0.0))
            end = float(item.get("end", start))
            if end > start:
                raw.append((label, start, end))

    return raw


def _merge_speaker_turns(turns: list[dict], *, max_gap_sec: float = 0.2) -> list[dict]:
    if not turns:
        return []
    merged: list[dict] = []
    for t in turns:
        if not merged:
            merged.append(dict(t))
            continue
        prev = merged[-1]
        if (
            prev["speaker"] == t["speaker"]
            and float(t["start"]) - float(prev["end"]) <= max_gap_sec
        ):
            prev["end"] = max(float(prev["end"]), float(t["end"]))
        else:
            merged.append(dict(t))
    return merged


def _speaker_for_interval(start: float, end: float, turns: list[dict]) -> str:
    if not turns:
        return DEFAULT_SPEAKER_LABEL
    start = float(start)
    end = float(max(end, start))
    best_speaker = str(turns[0].get("speaker", DEFAULT_SPEAKER_LABEL))
    best_overlap = -1.0
    for t in turns:
        ts = float(t.get("start", 0.0))
        te = float(t.get("end", ts))
        overlap = max(0.0, min(end, te) - max(start, ts))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = str(t.get("speaker", DEFAULT_SPEAKER_LABEL))
    if best_overlap > 0.0:
        return best_speaker

    mid = 0.5 * (start + end)
    best_dist = float("inf")
    for t in turns:
        ts = float(t.get("start", 0.0))
        te = float(t.get("end", ts))
        tmid = 0.5 * (ts + te)
        dist = abs(mid - tmid)
        if dist < best_dist:
            best_dist = dist
            best_speaker = str(t.get("speaker", DEFAULT_SPEAKER_LABEL))
    return best_speaker


def _merge_speaker_segments(items: list[dict], *, max_gap_sec: float = 0.8) -> list[dict]:
    """Merge adjacent same-speaker items into longer contiguous turns."""
    if not items:
        return []

    merged: list[dict] = []
    for item in items:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        speaker = str(item.get("speaker", DEFAULT_SPEAKER_LABEL))
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start))
        end = max(end, start)

        if not merged:
            merged.append(
                {"speaker": speaker, "start": start, "end": end, "text": text}
            )
            continue

        prev = merged[-1]
        gap = start - float(prev["end"])
        if speaker == prev["speaker"] and gap <= max_gap_sec:
            prev["end"] = max(float(prev["end"]), end)
            prev["text"] = f"{prev['text']} {text}".strip()
        else:
            merged.append(
                {"speaker": speaker, "start": start, "end": end, "text": text}
            )
    return merged
