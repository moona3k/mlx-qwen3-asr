"""Tests for mlx_qwen3_asr/writers.py."""

import json

import pytest

from mlx_qwen3_asr.transcribe import TranscriptionResult
from mlx_qwen3_asr.writers import (
    _format_timestamp_srt,
    _format_timestamp_vtt,
    get_writer,
    group_subtitle_segments,
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
