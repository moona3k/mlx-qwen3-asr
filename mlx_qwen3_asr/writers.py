"""Output format writers for transcription results."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Callable, Optional

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
    subtitle_segments = group_subtitle_segments(
        result.segments, language=result.language, text=result.text
    )

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
    subtitle_segments = group_subtitle_segments(
        result.segments, language=result.language, text=result.text
    )

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
    text: Optional[str] = None,
    max_words: int = 10,
    max_chars: int = 42,
    max_duration_sec: float = 6.0,
    max_gap_sec: float = 0.8,
) -> list[dict]:
    """Group word-level segments into subtitle-friendly cues.

    A cue ends at a sentence boundary, a speaker change, a pause of at least
    ``max_gap_sec``, or when adding the next word would exceed ``max_chars``
    display cells (CJK characters count double) or ``max_duration_sec``. A
    clause boundary (comma, semicolon) also ends a cue once it is at least half
    full. ``max_words`` applies only to space-delimited languages; for CJK the
    aligner emits one segment per character, so a word cap would cut phrases
    at arbitrary points (issue #15).

    Args:
        segments: Timed word segments ``[{text, start, end, speaker?}, ...]``.
        language: Transcript language; CJK languages are joined without spaces.
        text: Full transcript. The forced aligner drops punctuation, so when
            given, the transcript's punctuation is re-attached to the segments
            before grouping so sentence and clause boundaries can be used.
        max_words: Maximum words per cue for space-delimited languages.
        max_chars: Maximum display width per cue.
        max_duration_sec: Maximum cue duration in seconds.
        max_gap_sec: Silence that forces a new cue.

    Returns:
        List of ``{text, start, end}`` cues.
    """
    if not segments:
        return []
    if text:
        segments = restore_punctuation(segments, text)
    cjk = _is_cjk_language(language)

    grouped: list[dict] = []
    current: list[dict] = []

    def _flush() -> None:
        grouped.append(
            {
                "text": _join_subtitle_tokens(current, language=language),
                "start": float(current[0]["start"]),
                "end": float(current[-1]["end"]),
            }
        )

    for seg in segments:
        seg_text = str(seg.get("text", "")).strip()
        if not seg_text:
            continue
        start = float(seg.get("start", 0.0))
        end = max(float(seg.get("end", start)), start)
        item = {"text": seg_text, "start": start, "end": end, "speaker": seg.get("speaker")}
        if not current:
            current.append(item)
            continue

        current_text = _join_subtitle_tokens(current, language=language)
        candidate_text = _join_subtitle_tokens([*current, item], language=language)
        last_text = str(current[-1]["text"])
        should_break = (
            start - float(current[-1]["end"]) >= max_gap_sec
            or end - float(current[0]["start"]) > max_duration_sec
            or item["speaker"] != current[-1]["speaker"]
            or (not cjk and len(current) >= max_words)
            or _display_width(candidate_text) > max_chars
            or _ends_sentence(last_text)
            or (_ends_clause(last_text) and _display_width(current_text) * 2 >= max_chars)
        )
        if should_break:
            _flush()
            current = [item]
        else:
            current.append(item)

    if current:
        _flush()
    return grouped


def restore_punctuation(segments: list[dict], text: str) -> list[dict]:
    """Re-attach the transcript's punctuation to aligner word segments.

    The forced aligner tokenizes the transcript into bare words (or CJK
    characters) and drops punctuation. This walks ``text`` alongside the
    segments and copies leading punctuation (opening quotes) onto the first
    word of a phrase and trailing punctuation (commas, periods, closing quotes)
    onto the last word. If the segments do not match the transcript character
    for character (ignoring case and whitespace), the input is returned
    unchanged rather than guessing.

    Args:
        segments: Timed segments in transcript order.
        text: Full transcript the segments were aligned against.

    Returns:
        New segment dicts with punctuation restored, or the original list when
        the two sequences cannot be matched.
    """
    n = len(text)
    i = 0
    restored: list[dict] = []
    for seg in segments:
        chars = [c for c in str(seg.get("text", "")) if not c.isspace()]
        if not chars:
            restored.append(seg)
            continue
        leading: list[str] = []
        while i < n and (text[i].isspace() or _is_punct(text[i])):
            if not text[i].isspace():
                leading.append(text[i])
            i += 1
        for c in chars:
            if i < n and text[i].casefold() == c.casefold():
                i += 1
            else:
                return segments
        trailing: list[str] = []
        while i < n and _is_punct(text[i]) and not _is_opening_punct(text[i]):
            trailing.append(text[i])
            i += 1
        restored.append(
            {**seg, "text": "".join(leading) + str(seg["text"]).strip() + "".join(trailing)}
        )
    return restored


def _is_punct(char: str) -> bool:
    return unicodedata.category(char).startswith("P")


def _is_opening_punct(char: str) -> bool:
    return unicodedata.category(char) in {"Ps", "Pi"}


def _is_cjk_language(language: str) -> bool:
    return str(language or "").strip().lower() in _CJK_LANG_ALIASES


def _display_width(text: str) -> int:
    """Terminal-style display width: wide and fullwidth characters count double."""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in text)


_CLOSING_PUNCT = re.escape("\"'”’」』）)】》〉]")
_ENDS_SENTENCE = re.compile(r"[.!?。！？…][" + _CLOSING_PUNCT + r"]*$")
_ENDS_CLAUSE = re.compile(r"[,;:，、；：][" + _CLOSING_PUNCT + r"]*$")


def _ends_sentence(text: str) -> bool:
    return bool(_ENDS_SENTENCE.search(str(text or "").strip()))


def _ends_clause(text: str) -> bool:
    return bool(_ENDS_CLAUSE.search(str(text or "").strip()))


def _join_subtitle_tokens(tokens: list[dict], *, language: str) -> str:
    parts = [
        str(item.get("text", "")).strip()
        for item in tokens
        if str(item.get("text", "")).strip()
    ]
    if _is_cjk_language(language):
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
