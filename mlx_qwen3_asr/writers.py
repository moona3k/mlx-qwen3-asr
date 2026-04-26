"""Output format writers for transcription results."""

from __future__ import annotations

import json
import re
from typing import Callable

from .transcribe import TranscriptionResult

_CJK_LANG_ALIASES = {
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


def write_txt(result: TranscriptionResult, output_path: str) -> None:
    """Write plain text transcription."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result.text)
        f.write("\n")


def write_json(result: TranscriptionResult, output_path: str) -> None:
    """Write JSON formatted transcription with metadata."""
    data = {
        "text": result.text,
        "language": result.language,
    }
    if result.finish_reason:
        data["finish_reason"] = result.finish_reason
    if result.finish_reason or result.truncated:
        data["truncated"] = result.truncated
    if result.segments:
        data["segments"] = result.segments
    if result.speaker_segments:
        data["speaker_segments"] = result.speaker_segments
    if result.chunks:
        data["chunks"] = result.chunks

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_srt(result: TranscriptionResult, output_path: str) -> None:
    """Write SRT subtitle format. Requires segments with timestamps."""
    if not result.segments:
        raise ValueError("SRT output requires timestamp segments. Re-run with --timestamps.")
    subtitle_segments = group_subtitle_segments(result.segments, language=result.language)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(subtitle_segments, 1):
            start = _format_timestamp_srt(seg["start"])
            end = _format_timestamp_srt(seg["end"])
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{seg['text']}\n\n")


def write_vtt(result: TranscriptionResult, output_path: str) -> None:
    """Write WebVTT subtitle format. Requires segments with timestamps."""
    if not result.segments:
        raise ValueError("VTT output requires timestamp segments. Re-run with --timestamps.")
    subtitle_segments = group_subtitle_segments(result.segments, language=result.language)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")

        for seg in subtitle_segments:
            start = _format_timestamp_vtt(seg["start"])
            end = _format_timestamp_vtt(seg["end"])
            f.write(f"{start} --> {end}\n")
            f.write(f"{seg['text']}\n\n")


def write_tsv(result: TranscriptionResult, output_path: str) -> None:
    """Write TSV format with start, end, text columns."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("start\tend\ttext\n")

        if not result.segments:
            f.write(f"0\t-1\t{result.text}\n")
            return

        for seg in result.segments:
            start_ms = int(round(seg["start"] * 1000))
            end_ms = int(round(seg["end"] * 1000))
            f.write(f"{start_ms}\t{end_ms}\t{seg['text']}\n")


def get_writer(fmt: str) -> Callable:
    """Get writer function by format name.

    Args:
        fmt: Format string - one of 'txt', 'json', 'srt', 'vtt', 'tsv'

    Returns:
        Writer function with signature (result, output_path) -> None
    """
    writers = {
        "txt": write_txt,
        "json": write_json,
        "srt": write_srt,
        "vtt": write_vtt,
        "tsv": write_tsv,
    }
    if fmt not in writers:
        raise ValueError(f"Unknown format '{fmt}'. Supported: {', '.join(writers.keys())}")
    return writers[fmt]


def group_subtitle_segments(
    segments: list[dict],
    *,
    language: str = "",
    max_words: int = 10,
    max_chars: int = 42,
    max_duration_sec: float = 6.0,
    max_gap_sec: float = 0.8,
) -> list[dict]:
    """Group word-level segments into subtitle-friendly phrases."""
    if not segments:
        return []

    grouped: list[dict] = []
    current: list[dict] = []

    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        if end < start:
            end = start

        item = {"text": text, "start": start, "end": end}
        if not current:
            current.append(item)
            continue

        current_text = _join_subtitle_tokens(current, language=language)
        next_candidate = _join_subtitle_tokens([*current, item], language=language)
        gap = start - float(current[-1]["end"])
        duration = end - float(current[0]["start"])
        should_break = False
        if gap >= max_gap_sec:
            should_break = True
        if duration > max_duration_sec:
            should_break = True
        if len(current) >= max_words:
            should_break = True
        if len(next_candidate) > max_chars and len(current_text) > 0:
            should_break = True
        if _ends_sentence(str(current[-1]["text"])):
            should_break = True

        if should_break:
            grouped.append(
                {
                    "text": current_text,
                    "start": float(current[0]["start"]),
                    "end": float(current[-1]["end"]),
                }
            )
            current = [item]
        else:
            current.append(item)

    if current:
        grouped.append(
            {
                "text": _join_subtitle_tokens(current, language=language),
                "start": float(current[0]["start"]),
                "end": float(current[-1]["end"]),
            }
        )
    return grouped


def _ends_sentence(text: str) -> bool:
    return bool(re.search(r"[.!?。！？…]$", str(text or "").strip()))


def _join_subtitle_tokens(tokens: list[dict], *, language: str) -> str:
    parts = [
        str(item.get("text", "")).strip()
        for item in tokens
        if str(item.get("text", "")).strip()
    ]
    lang = str(language or "").strip().lower()
    if lang in _CJK_LANG_ALIASES:
        return "".join(parts)

    joined = " ".join(parts).strip()
    joined = re.sub(r"\s+([,.;:!?])", r"\1", joined)
    joined = re.sub(r"([(\[{])\s+", r"\1", joined)
    return joined


def _format_timestamp_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    total_ms = max(0, int(round(seconds * 1000.0)))
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, millis = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as VTT timestamp: HH:MM:SS.mmm"""
    total_ms = max(0, int(round(seconds * 1000.0)))
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, millis = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
