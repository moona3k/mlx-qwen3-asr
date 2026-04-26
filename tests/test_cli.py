"""Tests for mlx_qwen3_asr/cli.py."""

from __future__ import annotations

import importlib
import subprocess
import sys

import numpy as np
import pytest


def test_cli_timestamps_flag_is_accepted():
    result = subprocess.run(
        [sys.executable, "-m", "mlx_qwen3_asr.cli", "--timestamps", "dummy.wav"],
        capture_output=True,
        text=True,
        check=False,
    )

    # The command should parse and run; missing file is handled at runtime.
    assert result.returncode == 1
    assert "File not found: dummy.wav" in result.stderr


def test_cli_rejects_legacy_aligner_backend_flag(monkeypatch, capsys, tmp_path):
    cli = __import__("mlx_qwen3_asr.cli", fromlist=["main"])
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx-qwen3-asr",
            "--timestamps",
            "--aligner-backend",
            "qwen_asr",
            str(audio_path),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "unrecognized arguments: --aligner-backend" in capsys.readouterr().err


def test_cli_continues_batch_on_non_runtime_errors(monkeypatch, capsys, tmp_path):
    cli = __import__("mlx_qwen3_asr.cli", fromlist=["main"])
    transcribe_mod = importlib.import_module("mlx_qwen3_asr.transcribe")
    writers_mod = importlib.import_module("mlx_qwen3_asr.writers")

    bad_audio = tmp_path / "bad.wav"
    good_audio = tmp_path / "good.wav"
    bad_audio.write_bytes(b"RIFF")
    good_audio.write_bytes(b"RIFF")

    class _DummyResult:
        text = "good"
        language = "English"
        segments = None

    def _fake_transcribe(**kwargs):  # noqa: ANN003
        if kwargs["audio"] == str(bad_audio):
            raise ValueError("decode failed")
        return _DummyResult()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx-qwen3-asr",
            str(bad_audio),
            str(good_audio),
        ],
    )
    monkeypatch.setattr(transcribe_mod, "transcribe", _fake_transcribe)
    monkeypatch.setattr(writers_mod, "get_writer", lambda fmt: (lambda result, out_path: None))

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "decode failed" in captured.err
    assert "good" in captured.out


def test_cli_streaming_rejects_timestamps(monkeypatch, capsys, tmp_path):
    cli = __import__("mlx_qwen3_asr.cli", fromlist=["main"])
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx-qwen3-asr",
            "--streaming",
            "--timestamps",
            str(audio_path),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert "--streaming does not support --timestamps" in capsys.readouterr().err


def test_cli_streaming_rejects_diarize(monkeypatch, capsys, tmp_path):
    cli = __import__("mlx_qwen3_asr.cli", fromlist=["main"])
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx-qwen3-asr",
            "--streaming",
            "--diarize",
            str(audio_path),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert "--streaming does not support --diarize" in capsys.readouterr().err


def test_cli_streaming_mode_uses_streaming_pipeline(monkeypatch, capsys, tmp_path):
    cli = __import__("mlx_qwen3_asr.cli", fromlist=["main"])
    audio_mod = importlib.import_module("mlx_qwen3_asr.audio")
    stream_mod = importlib.import_module("mlx_qwen3_asr.streaming")
    transcribe_mod = importlib.import_module("mlx_qwen3_asr.transcribe")
    writers_mod = importlib.import_module("mlx_qwen3_asr.writers")

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF")

    calls = {"feed": 0, "finish": 0}

    class _DummyState:
        text = ""
        language = "unknown"

    def fake_load_audio(path):  # noqa: ANN001
        assert path == str(audio_path)
        return np.zeros(12, dtype=np.float32)

    def fake_init_streaming(**kwargs):  # noqa: ANN003
        assert kwargs["finalization_mode"] == "latency"
        return _DummyState()

    def fake_feed_audio(chunk, state):  # noqa: ANN001
        calls["feed"] += 1
        state.text = f"chunk{calls['feed']}"
        state.language = "English"
        return state

    def fake_finish_streaming(state):  # noqa: ANN001
        calls["finish"] += 1
        state.text = "final streaming text"
        state.language = "English"
        return state

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx-qwen3-asr",
            "--streaming",
            "--stream-chunk-sec",
            "1.0",
            "--stream-finalization-mode",
            "latency",
            str(audio_path),
        ],
    )
    monkeypatch.setattr(audio_mod, "load_audio", fake_load_audio)
    monkeypatch.setattr(stream_mod, "init_streaming", fake_init_streaming)
    monkeypatch.setattr(stream_mod, "feed_audio", fake_feed_audio)
    monkeypatch.setattr(stream_mod, "finish_streaming", fake_finish_streaming)

    def _fake_writer(result, out_path):  # noqa: ANN001, ARG001
        assert isinstance(result, transcribe_mod.TranscriptionResult)

    monkeypatch.setattr(writers_mod, "get_writer", lambda fmt: _fake_writer)

    cli.main()
    out = capsys.readouterr()
    assert "final streaming text" in out.out
    assert calls["feed"] == 1
    assert calls["finish"] == 1


def test_cli_offline_forwards_context(monkeypatch, tmp_path):
    cli = __import__("mlx_qwen3_asr.cli", fromlist=["main"])
    transcribe_mod = importlib.import_module("mlx_qwen3_asr.transcribe")
    writers_mod = importlib.import_module("mlx_qwen3_asr.writers")

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF")
    calls = {}

    class _DummyResult:
        text = "offline text"
        language = "English"
        segments = None
        chunks = [{"start": 0.0, "end": 1.0}]

    def _fake_transcribe(**kwargs):  # noqa: ANN003
        calls.update(kwargs)
        return _DummyResult()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx-qwen3-asr",
            "--context",
            "交易 停滞",
            str(audio_path),
        ],
    )
    monkeypatch.setattr(transcribe_mod, "transcribe", _fake_transcribe)
    monkeypatch.setattr(writers_mod, "get_writer", lambda fmt: (lambda result, out_path: None))

    cli.main()
    assert calls["context"] == "交易 停滞"


def test_cli_streaming_forwards_context(monkeypatch, tmp_path):
    cli = __import__("mlx_qwen3_asr.cli", fromlist=["main"])
    audio_mod = importlib.import_module("mlx_qwen3_asr.audio")
    stream_mod = importlib.import_module("mlx_qwen3_asr.streaming")
    writers_mod = importlib.import_module("mlx_qwen3_asr.writers")

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF")
    calls = {}

    class _DummyState:
        text = "stream text"
        language = "English"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx-qwen3-asr",
            "--streaming",
            "--context",
            "earnings EBITDA",
            str(audio_path),
        ],
    )
    monkeypatch.setattr(audio_mod, "load_audio", lambda path: np.zeros(12, dtype=np.float32))
    monkeypatch.setattr(
        stream_mod,
        "init_streaming",
        lambda **kwargs: calls.update(kwargs) or _DummyState(),
    )
    monkeypatch.setattr(stream_mod, "feed_audio", lambda chunk, state: state)
    monkeypatch.setattr(stream_mod, "finish_streaming", lambda state: state)
    monkeypatch.setattr(writers_mod, "get_writer", lambda fmt: (lambda result, out_path: None))

    cli.main()
    assert calls["context"] == "earnings EBITDA"


def test_cli_stdout_only_does_not_write_files(monkeypatch, capsys, tmp_path):
    cli = __import__("mlx_qwen3_asr.cli", fromlist=["main"])
    transcribe_mod = importlib.import_module("mlx_qwen3_asr.transcribe")
    writers_mod = importlib.import_module("mlx_qwen3_asr.writers")

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF")
    write_calls = {"count": 0}

    class _DummyResult:
        text = "stdout text"
        language = "English"
        segments = None
        chunks = [{"start": 0.0, "end": 1.0}]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx-qwen3-asr",
            "--stdout-only",
            "-o",
            str(tmp_path / "out"),
            str(audio_path),
        ],
    )
    monkeypatch.setattr(transcribe_mod, "transcribe", lambda **kwargs: _DummyResult())

    def _fake_writer(result, out_path):  # noqa: ANN001, ARG001
        write_calls["count"] += 1

    monkeypatch.setattr(writers_mod, "get_writer", lambda fmt: _fake_writer)

    cli.main()
    captured = capsys.readouterr()
    assert "stdout text" in captured.out
    assert write_calls["count"] == 0


def test_cli_quiet_suppresses_stdout(monkeypatch, capsys, tmp_path):
    cli = __import__("mlx_qwen3_asr.cli", fromlist=["main"])
    transcribe_mod = importlib.import_module("mlx_qwen3_asr.transcribe")
    writers_mod = importlib.import_module("mlx_qwen3_asr.writers")

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF")

    class _DummyResult:
        text = "quiet text"
        language = "English"
        segments = None
        chunks = [{"start": 0.0, "end": 1.0}]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx-qwen3-asr",
            "--quiet",
            str(audio_path),
        ],
    )
    monkeypatch.setattr(transcribe_mod, "transcribe", lambda **kwargs: _DummyResult())
    monkeypatch.setattr(writers_mod, "get_writer", lambda fmt: (lambda result, out_path: None))

    cli.main()
    captured = capsys.readouterr()
    assert "quiet text" not in captured.out


def test_cli_list_languages_exits_without_audio(monkeypatch, capsys):
    cli = __import__("mlx_qwen3_asr.cli", fromlist=["main"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx-qwen3-asr",
            "--list-languages",
        ],
    )
    cli.main()
    captured = capsys.readouterr()
    assert "Supported language aliases" in captured.out


def test_cli_doctor_runs_without_audio(monkeypatch):
    cli = __import__("mlx_qwen3_asr.cli", fromlist=["main"])
    called = {"value": False}

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx-qwen3-asr",
            "--doctor",
        ],
    )
    monkeypatch.setattr(
        cli,
        "_run_doctor",
        lambda: called.__setitem__("value", True) or 0,
    )

    cli.main()
    assert called["value"] is True


def test_cli_doctor_nonzero_exits(monkeypatch):
    cli = __import__("mlx_qwen3_asr.cli", fromlist=["main"])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx-qwen3-asr",
            "--doctor",
        ],
    )
    monkeypatch.setattr(cli, "_run_doctor", lambda: 1)

    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1


def test_cli_ffmpeg_preflight_rejects_video_when_missing_binary(monkeypatch, capsys, tmp_path):
    cli = __import__("mlx_qwen3_asr.cli", fromlist=["main"])
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"not-real-media")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx-qwen3-asr",
            str(video_path),
        ],
    )
    monkeypatch.setattr(cli.shutil, "which", lambda cmd: None)

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "require ffmpeg decoding" in err
    assert "Install ffmpeg and retry" in err


def test_cli_ffmpeg_preflight_allows_wav_when_missing_binary(monkeypatch, capsys, tmp_path):
    cli = __import__("mlx_qwen3_asr.cli", fromlist=["main"])
    transcribe_mod = importlib.import_module("mlx_qwen3_asr.transcribe")
    writers_mod = importlib.import_module("mlx_qwen3_asr.writers")

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF")

    class _DummyResult:
        text = "ok"
        language = "English"
        segments = None
        chunks = [{"start": 0.0, "end": 0.5}]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx-qwen3-asr",
            str(audio_path),
        ],
    )
    monkeypatch.setattr(cli.shutil, "which", lambda cmd: None)
    monkeypatch.setattr(transcribe_mod, "transcribe", lambda **kwargs: _DummyResult())
    monkeypatch.setattr(writers_mod, "get_writer", lambda fmt: (lambda result, out_path: None))

    cli.main()
    assert "ok" in capsys.readouterr().out


def test_cli_diarize_preflight_requires_pyannote(monkeypatch, capsys):
    cli = __import__("mlx_qwen3_asr.cli", fromlist=["main"])

    def _fake_find_spec(name):  # noqa: ANN001
        if name == "pyannote.audio":
            return None
        return object()

    monkeypatch.setattr(cli.importlib.util, "find_spec", _fake_find_spec)

    with pytest.raises(SystemExit) as exc:
        cli._preflight_diarization_runtime()  # noqa: SLF001

    assert exc.value.code == 1
    assert "requires optional dependency 'pyannote.audio'" in capsys.readouterr().err


def test_cli_diarize_preflight_emits_token_hint(monkeypatch, capsys):
    cli = __import__("mlx_qwen3_asr.cli", fromlist=["main"])

    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object())
    monkeypatch.delenv("PYANNOTE_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.setattr(cli, "_ensure_diarization_backend_ready", lambda: None)

    cli._preflight_diarization_runtime()  # noqa: SLF001
    assert "default pyannote/speaker-diarization-community-1" in capsys.readouterr().err


def test_cli_diarize_preflight_checks_backend_readiness(monkeypatch):
    cli = __import__("mlx_qwen3_asr.cli", fromlist=["main"])
    called = {"value": False}

    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(
        cli,
        "_ensure_diarization_backend_ready",
        lambda: called.__setitem__("value", True),
    )

    cli._preflight_diarization_runtime()  # noqa: SLF001
    assert called["value"] is True


def test_cli_diarize_preflight_fails_fast_when_backend_unavailable(monkeypatch, capsys):
    cli = __import__("mlx_qwen3_asr.cli", fromlist=["main"])

    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(
        cli,
        "_ensure_diarization_backend_ready",
        lambda: (_ for _ in ()).throw(RuntimeError("backend unavailable")),
    )

    with pytest.raises(SystemExit) as exc:
        cli._preflight_diarization_runtime()  # noqa: SLF001

    assert exc.value.code == 1
    assert "backend unavailable" in capsys.readouterr().err


def test_cli_auto_enables_timestamps_for_subtitles(monkeypatch, tmp_path):
    cli = __import__("mlx_qwen3_asr.cli", fromlist=["main"])
    transcribe_mod = importlib.import_module("mlx_qwen3_asr.transcribe")
    writers_mod = importlib.import_module("mlx_qwen3_asr.writers")

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF")
    calls = {}

    class _DummyResult:
        text = "hello"
        language = "English"
        segments = [{"text": "hello", "start": 0.0, "end": 0.5}]
        chunks = [{"start": 0.0, "end": 0.5}]

    def _fake_transcribe(**kwargs):  # noqa: ANN003
        calls["return_timestamps"] = kwargs["return_timestamps"]
        return _DummyResult()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx-qwen3-asr",
            "-f",
            "srt",
            str(audio_path),
        ],
    )
    monkeypatch.setattr(transcribe_mod, "transcribe", _fake_transcribe)
    monkeypatch.setattr(writers_mod, "get_writer", lambda fmt: (lambda result, out_path: None))

    cli.main()
    assert calls["return_timestamps"] is True


def test_cli_diarize_auto_enables_timestamps_and_forwards_args(monkeypatch, tmp_path):
    cli = __import__("mlx_qwen3_asr.cli", fromlist=["main"])
    transcribe_mod = importlib.import_module("mlx_qwen3_asr.transcribe")
    writers_mod = importlib.import_module("mlx_qwen3_asr.writers")

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF")
    calls = {}

    class _DummyResult:
        text = "hello"
        language = "English"
        segments = [{"text": "hello", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"}]
        speaker_segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 0.5, "text": "hello"}
        ]
        chunks = [{"start": 0.0, "end": 0.5}]

    def _fake_transcribe(**kwargs):  # noqa: ANN003
        calls.update(kwargs)
        return _DummyResult()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx-qwen3-asr",
            "--diarize",
            "--num-speakers",
            "2",
            "--min-speakers",
            "1",
            "--max-speakers",
            "4",
            str(audio_path),
        ],
    )
    monkeypatch.setattr(cli, "_preflight_diarization_runtime", lambda: None)
    monkeypatch.setattr(transcribe_mod, "transcribe", _fake_transcribe)
    monkeypatch.setattr(writers_mod, "get_writer", lambda fmt: (lambda result, out_path: None))

    cli.main()
    assert calls["return_timestamps"] is True
    assert calls["diarize"] is True
    assert calls["diarization_num_speakers"] == 2
    assert calls["diarization_min_speakers"] == 1
    assert calls["diarization_max_speakers"] == 4


def test_cli_streaming_rejects_subtitle_formats(monkeypatch, capsys, tmp_path):
    cli = __import__("mlx_qwen3_asr.cli", fromlist=["main"])
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx-qwen3-asr",
            "--streaming",
            "-f",
            "srt",
            str(audio_path),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert "require offline transcription" in capsys.readouterr().err
