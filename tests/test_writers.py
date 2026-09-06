"""Tests for mlx_qwen3_asr/writers.py."""

import json

import pytest

from mlx_qwen3_asr.transcribe import TranscriptionResult
from mlx_qwen3_asr.writers import (
    _format_timestamp_srt,
    _format_timestamp_vtt,
    get_writer,
    group_subtitle_segments,
    restore_punctuation,
    write_json,
    write_srt,
    write_tsv,
    write_txt,
    write_vtt,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_result():
    return TranscriptionResult(text="Hello world", language="English")


@pytest.fixture
def result_with_segments():
    return TranscriptionResult(
        text="Hello world. How are you?",
        language="English",
        segments=[
            {"text": "Hello world.", "start": 0.0, "end": 2.5},
            {"text": "How are you?", "start": 2.5, "end": 5.0},
        ],
    )


# ---------------------------------------------------------------------------
# write_txt
# ---------------------------------------------------------------------------


class TestWriteTxt:
    """Test write_txt() writes text + newline."""

    def test_writes_text_with_newline(self, simple_result, tmp_path):
        path = str(tmp_path / "output.txt")
        write_txt(simple_result, path)
        with open(path) as f:
            content = f.read()
        assert content == "Hello world\n"

    def test_unicode_text(self, tmp_path):
        result = TranscriptionResult(text="你好世界", language="Chinese")
        path = str(tmp_path / "output.txt")
        write_txt(result, path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert content == "你好世界\n"


# ---------------------------------------------------------------------------
# write_json
# ---------------------------------------------------------------------------


class TestWriteJson:
    """Test write_json() writes valid JSON with text and language."""

    def test_basic_json(self, simple_result, tmp_path):
        path = str(tmp_path / "output.json")
        write_json(simple_result, path)
        with open(path) as f:
            data = json.load(f)
        assert data["text"] == "Hello world"
        assert data["language"] == "English"
        assert "truncated" not in data
        assert "segments" not in data

    def test_json_includes_truncated_with_finish_reason(self, tmp_path):
        result = TranscriptionResult(
            text="Hello world",
            language="English",
            finish_reason="length",
            truncated=True,
        )
        path = str(tmp_path / "output.json")
        write_json(result, path)
        with open(path) as f:
            data = json.load(f)
        assert data["finish_reason"] == "length"
        assert data["truncated"] is True

    def test_json_with_segments(self, result_with_segments, tmp_path):
        path = str(tmp_path / "output.json")
        write_json(result_with_segments, path)
        with open(path) as f:
            data = json.load(f)
        assert data["text"] == "Hello world. How are you?"
        assert len(data["segments"]) == 2
        assert data["segments"][0]["start"] == 0.0
        assert data["segments"][1]["end"] == 5.0

    def test_json_with_speaker_segments(self, tmp_path):
        result = TranscriptionResult(
            text="Hello world",
            language="English",
            speaker_segments=[
                {
                    "speaker": "SPEAKER_00",
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Hello world",
                }
            ],
        )
        path = str(tmp_path / "output.json")
        write_json(result, path)
        with open(path) as f:
            data = json.load(f)
        assert "speaker_segments" in data
        assert data["speaker_segments"][0]["speaker"] == "SPEAKER_00"


# ---------------------------------------------------------------------------
# write_srt
# ---------------------------------------------------------------------------


class TestWriteSrt:
    """Test write_srt() with and without segments."""

    def test_with_segments(self, result_with_segments, tmp_path):
        path = str(tmp_path / "output.srt")
        write_srt(result_with_segments, path)
        with open(path) as f:
            content = f.read()

        # Check numbering starts at 1
        assert content.startswith("1\n")
        # Check timestamp format: HH:MM:SS,mmm
        assert "00:00:00,000 --> 00:00:02,500" in content
        assert "2\n" in content
        assert "00:00:02,500 --> 00:00:05,000" in content
        # Check text
        assert "Hello world." in content
        assert "How are you?" in content

    def test_raises_without_segments(self, simple_result, tmp_path):
        path = str(tmp_path / "output.srt")
        with pytest.raises(ValueError, match="requires timestamp segments"):
            write_srt(simple_result, path)


# ---------------------------------------------------------------------------
# write_vtt
# ---------------------------------------------------------------------------


class TestWriteVtt:
    """Test write_vtt() with and without segments."""

    def test_webvtt_header(self, result_with_segments, tmp_path):
        path = str(tmp_path / "output.vtt")
        write_vtt(result_with_segments, path)
        with open(path) as f:
            content = f.read()
        assert content.startswith("WEBVTT\n")

    def test_with_segments(self, result_with_segments, tmp_path):
        path = str(tmp_path / "output.vtt")
        write_vtt(result_with_segments, path)
        with open(path) as f:
            content = f.read()
        # VTT uses period for millis separator
        assert "00:00:00.000 --> 00:00:02.500" in content
        assert "00:00:02.500 --> 00:00:05.000" in content

    def test_raises_without_segments(self, simple_result, tmp_path):
        path = str(tmp_path / "output.vtt")
        with pytest.raises(ValueError, match="requires timestamp segments"):
            write_vtt(simple_result, path)


class TestSubtitleGrouping:
    def test_groups_word_level_segments_into_phrases(self):
        segments = [
            {"text": "Hello", "start": 0.0, "end": 0.4},
            {"text": "world.", "start": 0.41, "end": 0.8},
            {"text": "How", "start": 1.2, "end": 1.5},
            {"text": "are", "start": 1.51, "end": 1.7},
            {"text": "you?", "start": 1.71, "end": 2.0},
        ]
        grouped = group_subtitle_segments(segments, language="English")
        assert len(grouped) == 2
        assert grouped[0]["text"] == "Hello world."
        assert grouped[1]["text"] == "How are you?"

    @staticmethod
    def _char_segments(text: str, step: float = 0.2) -> list[dict]:
        """Mimic the forced aligner: one unpunctuated segment per CJK character."""
        chars = [c for c in text if c.isalnum()]
        return [
            {"text": c, "start": i * step, "end": (i + 1) * step}
            for i, c in enumerate(chars)
        ]

    def test_cjk_is_not_cut_every_ten_characters(self):
        # Issue #15: max_words counted each aligned character as a word.
        text = "今天我们继续学习真实义品这个概念非常重要"  # 19 chars, no punctuation
        grouped = group_subtitle_segments(self._char_segments(text), language="Chinese")
        assert [g["text"] for g in grouped] == [text]

    def test_cjk_wraps_by_display_width(self):
        text = "字" * 45
        grouped = group_subtitle_segments(self._char_segments(text), language="Chinese")
        # 42 display cells / 2 per CJK character = 21 characters per cue.
        assert [len(g["text"]) for g in grouped] == [21, 21, 3]
        assert "".join(g["text"] for g in grouped) == text

    def test_cjk_breaks_at_restored_sentence_and_clause_boundaries(self):
        text = "大家好，今天我们继续学习真实义品。上一次我们讲到三取空这一段，这个概念非常重要。"
        grouped = group_subtitle_segments(
            self._char_segments(text), language="Chinese", text=text
        )
        assert [g["text"] for g in grouped] == [
            "大家好，今天我们继续学习真实义品。",  # short clause merges with next
            "上一次我们讲到三取空这一段，",  # clause boundary once half full
            "这个概念非常重要。",
        ]
        assert grouped[0]["end"] == grouped[1]["start"]

    def test_cjk_pause_still_splits(self):
        segments = [
            {"text": "你好", "start": 0.0, "end": 1.0},
            {"text": "再见", "start": 2.5, "end": 3.0},
        ]
        grouped = group_subtitle_segments(segments, language="Chinese")
        assert [g["text"] for g in grouped] == ["你好", "再见"]

    def test_latin_word_cap_is_unchanged(self):
        segments = [
            {"text": "a", "start": i * 0.1, "end": (i + 1) * 0.1} for i in range(12)
        ]
        grouped = group_subtitle_segments(segments, language="English")
        assert [g["text"] for g in grouped] == [" ".join(["a"] * 10), "a a"]

    def test_latin_sentence_breaks_after_punctuation_restored(self):
        words = "The quick brown fox jumps It ran".split()
        segments = [
            {"text": w, "start": i * 0.3, "end": (i + 1) * 0.3} for i, w in enumerate(words)
        ]
        grouped = group_subtitle_segments(
            segments, language="English", text="The quick brown fox jumps. It ran!"
        )
        assert [g["text"] for g in grouped] == ["The quick brown fox jumps.", "It ran!"]

    def test_speaker_change_starts_new_cue(self):
        segments = [
            {"text": "你好", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"text": "大家好", "start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"},
        ]
        grouped = group_subtitle_segments(segments, language="Chinese")
        assert [g["text"] for g in grouped] == ["你好", "大家好"]

    def test_input_segments_are_not_mutated(self):
        segments = [{"text": "你好", "start": 0.0, "end": 1.0}]
        snapshot = json.dumps(segments)
        group_subtitle_segments(segments, language="Chinese", text="「你好！」")
        assert json.dumps(segments) == snapshot


class TestRestorePunctuation:
    def test_attaches_leading_and_trailing_punctuation(self):
        segments = [
            {"text": "你", "start": 0.0, "end": 0.1},
            {"text": "好", "start": 0.1, "end": 0.2},
            {"text": "再", "start": 0.3, "end": 0.4},
            {"text": "见", "start": 0.4, "end": 0.5},
        ]
        restored = restore_punctuation(segments, "「你好。」再见！")
        assert [s["text"] for s in restored] == ["「你", "好。」", "再", "见！"]
        assert restored[0]["start"] == 0.0 and restored[1]["end"] == 0.2

    def test_is_case_and_whitespace_insensitive(self):
        segments = [
            {"text": "hello", "start": 0, "end": 1},
            {"text": "World", "start": 1, "end": 2},
        ]
        restored = restore_punctuation(segments, "Hello,  world.")
        assert [s["text"] for s in restored] == ["hello,", "World."]

    def test_returns_input_unchanged_on_mismatch(self):
        segments = [{"text": "正确", "start": 0.0, "end": 1.0}]
        assert restore_punctuation(segments, "错误。") is segments

    def test_keeps_existing_punctuation_when_text_is_absent(self):
        segments = [
            {"text": "你好，", "start": 0.0, "end": 1.0},
            {"text": "世界！", "start": 1.0, "end": 2.0},
        ]
        assert group_subtitle_segments(segments, language="Chinese") == [
            {"text": "你好，世界！", "start": 0.0, "end": 2.0}
        ]


class TestSubtitleWritersRestorePunctuation:
    @pytest.mark.parametrize(
        ("writer", "suffix", "header"),
        [(write_srt, "srt", "1\n"), (write_vtt, "vtt", "WEBVTT\n\n")],
    )
    def test_cues_carry_transcript_punctuation(self, writer, suffix, header, tmp_path):
        text = "今天我们继续学习。这个概念非常重要！"
        segments = TestSubtitleGrouping._char_segments(text)
        result = TranscriptionResult(text=text, language="Chinese", segments=segments)
        path = tmp_path / f"out.{suffix}"
        writer(result, str(path))
        content = path.read_text(encoding="utf-8")
        assert content.startswith(header)
        cues = [line for line in content.splitlines() if line and "-->" not in line]
        cues = [c for c in cues if not c.isdigit() and c != "WEBVTT"]
        assert cues == ["今天我们继续学习。", "这个概念非常重要！"]


# ---------------------------------------------------------------------------
# write_tsv
# ---------------------------------------------------------------------------


class TestWriteTsv:
    """Test write_tsv() with and without segments."""

    def test_with_segments(self, result_with_segments, tmp_path):
        path = str(tmp_path / "output.tsv")
        write_tsv(result_with_segments, path)
        with open(path) as f:
            lines = f.readlines()
        # Header
        assert lines[0].strip() == "start\tend\ttext"
        # First segment
        parts = lines[1].strip().split("\t")
        assert parts[0] == "0"  # 0.0 * 1000 = 0
        assert parts[1] == "2500"  # 2.5 * 1000 = 2500
        assert parts[2] == "Hello world."

    def test_without_segments(self, simple_result, tmp_path):
        path = str(tmp_path / "output.tsv")
        write_tsv(simple_result, path)
        with open(path) as f:
            lines = f.readlines()
        assert lines[0].strip() == "start\tend\ttext"
        parts = lines[1].strip().split("\t")
        assert parts[0] == "0"
        assert parts[1] == "-1"
        assert parts[2] == "Hello world"

    def test_rounds_milliseconds(self, tmp_path):
        result = TranscriptionResult(
            text="x",
            language="English",
            segments=[{"text": "x", "start": 1.2346, "end": 2.3456}],
        )
        path = str(tmp_path / "output.tsv")
        write_tsv(result, path)
        with open(path) as f:
            lines = f.readlines()
        parts = lines[1].strip().split("\t")
        assert parts[0] == "1235"
        assert parts[1] == "2346"


# ---------------------------------------------------------------------------
# get_writer
# ---------------------------------------------------------------------------


class TestGetWriter:
    """Test get_writer() returns correct function."""

    def test_txt(self):
        assert get_writer("txt") is write_txt

    def test_json(self):
        assert get_writer("json") is write_json

    def test_srt(self):
        assert get_writer("srt") is write_srt

    def test_vtt(self):
        assert get_writer("vtt") is write_vtt

    def test_tsv(self):
        assert get_writer("tsv") is write_tsv

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown format"):
            get_writer("xml")


# ---------------------------------------------------------------------------
# Timestamp formatting
# ---------------------------------------------------------------------------


class TestFormatTimestampSrt:
    """Test _format_timestamp_srt() correctness."""

    def test_zero(self):
        assert _format_timestamp_srt(0.0) == "00:00:00,000"

    def test_seconds(self):
        assert _format_timestamp_srt(1.5) == "00:00:01,500"

    def test_minutes(self):
        assert _format_timestamp_srt(65.25) == "00:01:05,250"

    def test_hours(self):
        assert _format_timestamp_srt(3661.5) == "01:01:01,500"

    def test_large_value(self):
        # 99 hours, 59 minutes, 59 seconds, 999 millis
        assert _format_timestamp_srt(359999.999) == "99:59:59,999"

    def test_rounding_carries_to_next_second(self):
        assert _format_timestamp_srt(1.9996) == "00:00:02,000"


class TestFormatTimestampVtt:
    """Test _format_timestamp_vtt() correctness."""

    def test_zero(self):
        assert _format_timestamp_vtt(0.0) == "00:00:00.000"

    def test_seconds(self):
        assert _format_timestamp_vtt(1.5) == "00:00:01.500"

    def test_minutes(self):
        assert _format_timestamp_vtt(65.25) == "00:01:05.250"

    def test_uses_period_not_comma(self):
        """VTT uses period for millis separator, not comma like SRT."""
        result = _format_timestamp_vtt(1.5)
        assert "." in result
        assert "," not in result

    def test_rounding_carries_to_next_second(self):
        assert _format_timestamp_vtt(1.9996) == "00:00:02.000"
