"""Tests for release-gate helpers in scripts/quality_gate.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_quality_gate_module():
    repo = Path(__file__).resolve().parents[1]
    module_path = repo / "scripts" / "quality_gate.py"
    spec = importlib.util.spec_from_file_location("quality_gate_test_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["quality_gate_test_module"] = module
    spec.loader.exec_module(module)
    return module


def test_streaming_quality_gate_requires_audio_fixture(monkeypatch, tmp_path):
    qg = _load_quality_gate_module()
    monkeypatch.delenv("STREAMING_QUALITY_AUDIO", raising=False)
    monkeypatch.setenv("STREAMING_QUALITY_ENDPOINTING_MODES", "fixed")

    step = qg._run_streaming_quality_gate(
        repo=tmp_path,
        python_bin="python",
        strict_release=False,
    )

    assert not step.passed
    assert "STREAMING_QUALITY_AUDIO is required" in step.note


def test_streaming_quality_gate_rejects_empty_mode_list(monkeypatch, tmp_path):
    qg = _load_quality_gate_module()
    fixture = tmp_path / "tests" / "fixtures" / "test_speech.wav"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_bytes(b"RIFF")
    monkeypatch.setenv("STREAMING_QUALITY_ENDPOINTING_MODES", " , ")

    step = qg._run_streaming_quality_gate(
        repo=tmp_path,
        python_bin="python",
        strict_release=False,
    )

    assert not step.passed
    assert "No endpointing modes configured" in step.note


def test_streaming_quality_gate_passes_for_good_metrics(monkeypatch, tmp_path):
    qg = _load_quality_gate_module()
    fixture = tmp_path / "tests" / "fixtures" / "test_speech.wav"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_bytes(b"RIFF")

    def _fake_run(cmd, _cwd, env=None):  # noqa: ANN001, ANN002, ARG001
        json_output = Path(cmd[cmd.index("--json-output") + 1])
        payload = {
            "final_metrics": {
                "partial_stability": 0.95,
                "rewrite_rate": 0.04,
                "finalization_delta_chars": 3,
            }
        }
        json_output.write_text(json.dumps(payload), encoding="utf-8")
        return qg.StepResult(
            name="python",
            cmd=" ".join(cmd),
            passed=True,
            duration_sec=0.01,
            returncode=0,
        )

    monkeypatch.setattr(qg, "_run", _fake_run)
    monkeypatch.setenv("STREAMING_QUALITY_ENDPOINTING_MODES", "fixed,energy")
    monkeypatch.setenv("STREAMING_QUALITY_FAIL_PARTIAL_STABILITY_BELOW", "0.90")
    monkeypatch.setenv("STREAMING_QUALITY_FAIL_REWRITE_RATE_ABOVE", "0.10")
    monkeypatch.setenv("STREAMING_QUALITY_FAIL_FINALIZATION_DELTA_CHARS_ABOVE", "8")

    step = qg._run_streaming_quality_gate(
        repo=tmp_path,
        python_bin="python",
        strict_release=False,
    )

    assert step.passed
    assert "fixed: stability=0.9500" in step.note
    assert "energy: stability=0.9500" in step.note


def test_streaming_quality_gate_fails_threshold(monkeypatch, tmp_path):
    qg = _load_quality_gate_module()
    fixture = tmp_path / "tests" / "fixtures" / "test_speech.wav"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_bytes(b"RIFF")

    def _fake_run(cmd, _cwd, env=None):  # noqa: ANN001, ANN002, ARG001
        json_output = Path(cmd[cmd.index("--json-output") + 1])
        payload = {
            "final_metrics": {
                "partial_stability": 0.2,
                "rewrite_rate": 0.6,
                "finalization_delta_chars": 100,
            }
        }
        json_output.write_text(json.dumps(payload), encoding="utf-8")
        return qg.StepResult(
            name="python",
            cmd=" ".join(cmd),
            passed=True,
            duration_sec=0.01,
            returncode=0,
        )

    monkeypatch.setattr(qg, "_run", _fake_run)
    monkeypatch.setenv("STREAMING_QUALITY_ENDPOINTING_MODES", "fixed")
    monkeypatch.setenv("STREAMING_QUALITY_FAIL_PARTIAL_STABILITY_BELOW", "0.90")
    monkeypatch.setenv("STREAMING_QUALITY_FAIL_REWRITE_RATE_ABOVE", "0.10")
    monkeypatch.setenv("STREAMING_QUALITY_FAIL_FINALIZATION_DELTA_CHARS_ABOVE", "8")

    step = qg._run_streaming_quality_gate(
        repo=tmp_path,
        python_bin="python",
        strict_release=False,
    )

    assert not step.passed
    assert "partial_stability=0.2000 < 0.9000" in step.note


def test_realworld_longform_quality_gate_requires_manifest(monkeypatch, tmp_path):
    qg = _load_quality_gate_module()
    monkeypatch.delenv("REALWORLD_LONGFORM_EVAL_JSONL", raising=False)

    step = qg._run_realworld_longform_quality_gate(
        repo=tmp_path,
        python_bin="python",
        strict_release=False,
    )

    assert not step.passed
    assert "requires REALWORLD_LONGFORM_EVAL_JSONL" in step.note


def test_realworld_longform_quality_gate_fails_missing_audio(monkeypatch, tmp_path):
    qg = _load_quality_gate_module()
    manifest = tmp_path / "longform.jsonl"
    missing_audio = tmp_path / "missing.wav"
    row = {
        "sample_id": "s1",
        "subset": "earnings22-full-test",
        "speaker_id": "spk",
        "language": "English",
        "audio_path": str(missing_audio),
        "reference_text": "hello world",
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setenv("REALWORLD_LONGFORM_EVAL_JSONL", str(manifest))

    step = qg._run_realworld_longform_quality_gate(
        repo=tmp_path,
        python_bin="python",
        strict_release=False,
    )

    assert not step.passed
    assert "missing local audio files" in step.note
    assert "build_earnings22_longform_manifest.py" in step.note


def test_realworld_longform_quality_gate_passes_and_uses_strict_default(
    monkeypatch,
    tmp_path,
):
    qg = _load_quality_gate_module()
    manifest = tmp_path / "longform.jsonl"
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    row = {
        "sample_id": "s1",
        "subset": "earnings22-full-test",
        "speaker_id": "spk",
        "language": "English",
        "audio_path": str(audio),
        "reference_text": "hello world",
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setenv("REALWORLD_LONGFORM_EVAL_JSONL", str(manifest))

    called: dict[str, object] = {}

    def _fake_run(cmd, _cwd, env=None):  # noqa: ANN001, ANN002, ARG001
        called["cmd"] = cmd
        return qg.StepResult(
            name="python",
            cmd=" ".join(cmd),
            passed=True,
            duration_sec=0.01,
            returncode=0,
        )

    monkeypatch.setattr(qg, "_run", _fake_run)

    step = qg._run_realworld_longform_quality_gate(
        repo=tmp_path,
        python_bin="python",
        strict_release=True,
    )

    assert step.passed
    cmd = called["cmd"]
    assert isinstance(cmd, list)
    idx = cmd.index("--fail-primary-above")
    assert cmd[idx + 1] == "0.20"


def test_streaming_manifest_quality_gate_requires_manifest(monkeypatch, tmp_path):
    qg = _load_quality_gate_module()
    monkeypatch.delenv("STREAMING_MANIFEST_QUALITY_EVAL_JSONL", raising=False)

    step = qg._run_streaming_manifest_quality_gate(
        repo=tmp_path,
        python_bin="python",
        strict_release=False,
    )

    assert not step.passed
    assert "requires STREAMING_MANIFEST_QUALITY_EVAL_JSONL" in step.note


def test_streaming_manifest_quality_gate_fails_missing_audio(monkeypatch, tmp_path):
    qg = _load_quality_gate_module()
    manifest = tmp_path / "streaming_manifest.jsonl"
    row = {
        "sample_id": "s1",
        "subset": "manifest",
        "speaker_id": "spk",
        "audio_path": str(tmp_path / "missing.wav"),
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setenv("STREAMING_MANIFEST_QUALITY_EVAL_JSONL", str(manifest))

    step = qg._run_streaming_manifest_quality_gate(
        repo=tmp_path,
        python_bin="python",
        strict_release=False,
    )

    assert not step.passed
    assert "missing local audio files" in step.note


def test_streaming_manifest_quality_gate_passes_and_uses_threshold_defaults(
    monkeypatch,
    tmp_path,
):
    qg = _load_quality_gate_module()
    manifest = tmp_path / "streaming_manifest.jsonl"
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    row = {
        "sample_id": "s1",
        "subset": "manifest",
        "speaker_id": "spk",
        "audio_path": str(audio),
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setenv("STREAMING_MANIFEST_QUALITY_EVAL_JSONL", str(manifest))

    called: dict[str, object] = {}

    def _fake_run(cmd, _cwd, env=None):  # noqa: ANN001, ANN002, ARG001
        called["cmd"] = cmd
        return qg.StepResult(
            name="python",
            cmd=" ".join(cmd),
            passed=True,
            duration_sec=0.01,
            returncode=0,
        )

    monkeypatch.setattr(qg, "_run", _fake_run)

    step = qg._run_streaming_manifest_quality_gate(
        repo=tmp_path,
        python_bin="python",
        strict_release=True,
    )

    assert step.passed
    cmd = called["cmd"]
    assert isinstance(cmd, list)
    idx = cmd.index("--fail-partial-stability-below")
    assert cmd[idx + 1] == "0.85"
    idx = cmd.index("--fail-rewrite-rate-above")
    assert cmd[idx + 1] == "0.85"
