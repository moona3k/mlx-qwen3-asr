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

    (step,) = qg._run_streaming_manifest_quality_gates(
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

    (step,) = qg._run_streaming_manifest_quality_gates(
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
    offline = tmp_path / "offline.json"
    offline.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("STREAMING_MANIFEST_QUALITY_EVAL_JSONL", str(manifest))
    monkeypatch.setenv("STREAMING_MANIFEST_QUALITY_EVAL_OFFLINE_JSON", str(offline))

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

    (step,) = qg._run_streaming_manifest_quality_gates(
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
    idx = cmd.index("--offline-quality-json")
    assert cmd[idx + 1] == str(offline.resolve())
    idx = cmd.index("--fail-primary-above-offline-pp")
    assert cmd[idx + 1] == "3.0"


def test_streaming_manifest_quality_gate_strict_requires_offline_artifact(
    monkeypatch,
    tmp_path,
):
    qg = _load_quality_gate_module()
    manifest = tmp_path / "streaming_manifest.jsonl"
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    manifest.write_text(
        json.dumps({"sample_id": "s1", "audio_path": str(audio)}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("STREAMING_MANIFEST_QUALITY_EVAL_JSONL", str(manifest))
    monkeypatch.delenv("STREAMING_MANIFEST_QUALITY_EVAL_OFFLINE_JSON", raising=False)

    (step,) = qg._run_streaming_manifest_quality_gates(
        repo=tmp_path,
        python_bin="python",
        strict_release=True,
    )

    assert not step.passed
    assert "STREAMING_MANIFEST_QUALITY_EVAL_OFFLINE_JSON" in step.note


def test_streaming_manifest_quality_gate_strict_defaults_to_committed_manifest(
    monkeypatch,
    tmp_path,
):
    qg = _load_quality_gate_module()
    monkeypatch.delenv("STREAMING_MANIFEST_QUALITY_EVAL_JSONL", raising=False)
    monkeypatch.delenv("STREAMING_MANIFEST_QUALITY_EVAL_OFFLINE_JSON", raising=False)

    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    default_manifest = tmp_path / qg.STREAMING_MANIFEST_STRICT_DEFAULT_MANIFEST
    default_manifest.parent.mkdir(parents=True)
    default_manifest.write_text(
        json.dumps({"sample_id": "s1", "audio_path": str(audio), "reference_text": "x"}) + "\n",
        encoding="utf-8",
    )
    default_offline = tmp_path / qg.STREAMING_MANIFEST_STRICT_DEFAULT_OFFLINE_JSON
    default_offline.write_text("{}", encoding="utf-8")

    called: dict[str, object] = {}

    def _fake_run(cmd, _cwd, env=None):  # noqa: ANN001, ANN002, ARG001
        called["cmd"] = cmd
        return qg.StepResult(
            name="python", cmd=" ".join(cmd), passed=True, duration_sec=0.0, returncode=0
        )

    monkeypatch.setattr(qg, "_run", _fake_run)

    (step,) = qg._run_streaming_manifest_quality_gates(
        repo=tmp_path,
        python_bin="python",
        strict_release=True,
    )

    assert step.passed
    cmd = called["cmd"]
    assert isinstance(cmd, list)
    assert cmd[cmd.index("--manifest-jsonl") + 1] == str(default_manifest.resolve())
    assert cmd[cmd.index("--offline-quality-json") + 1] == str(default_offline.resolve())
    assert cmd[cmd.index("--fail-primary-above-offline-pp") + 1] == "3.0"


def test_streaming_manifest_quality_gate_strict_runs_every_committed_lane(
    monkeypatch,
    tmp_path,
):
    qg = _load_quality_gate_module()
    monkeypatch.delenv("STREAMING_MANIFEST_QUALITY_EVAL_JSONL", raising=False)
    monkeypatch.delenv("STREAMING_MANIFEST_QUALITY_EVAL_OFFLINE_JSON", raising=False)
    monkeypatch.setenv(
        "STREAMING_MANIFEST_QUALITY_EVAL_JSON_OUTPUT", str(tmp_path / "out" / "streaming.json")
    )

    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    assert len(qg.STREAMING_MANIFEST_STRICT_DEFAULT_LANES) == 2
    for manifest_rel, offline_rel in qg.STREAMING_MANIFEST_STRICT_DEFAULT_LANES:
        manifest = tmp_path / manifest_rel
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps({"sample_id": "s1", "audio_path": str(audio), "reference_text": "x"})
            + "\n",
            encoding="utf-8",
        )
        (tmp_path / offline_rel).write_text("{}", encoding="utf-8")

    cmds: list[list[str]] = []

    def _fake_run(cmd, _cwd, env=None):  # noqa: ANN001, ANN002, ARG001
        cmds.append(cmd)
        return qg.StepResult(
            name="python", cmd=" ".join(cmd), passed=True, duration_sec=0.0, returncode=0
        )

    monkeypatch.setattr(qg, "_run", _fake_run)

    steps = qg._run_streaming_manifest_quality_gates(
        repo=tmp_path,
        python_bin="python",
        strict_release=True,
    )

    assert len(steps) == 2 and all(s.passed for s in steps)
    manifests = [c[c.index("--manifest-jsonl") + 1] for c in cmds]
    offlines = [c[c.index("--offline-quality-json") + 1] for c in cmds]
    outputs = [c[c.index("--json-output") + 1] for c in cmds]
    expected = [
        (str((tmp_path / m).resolve()), str((tmp_path / o).resolve()))
        for m, o in qg.STREAMING_MANIFEST_STRICT_DEFAULT_LANES
    ]
    assert list(zip(manifests, offlines, strict=True)) == expected
    assert all(c[c.index("--fail-primary-above-offline-pp") + 1] == "3.0" for c in cmds)
    # Shared JSON output must not be overwritten by the second lane.
    assert len(set(outputs)) == 2
    assert all(Path(o).stem.startswith("streaming-") for o in outputs)


def test_streaming_manifest_quality_gate_strict_rejects_offline_env_without_manifest(
    monkeypatch,
    tmp_path,
):
    qg = _load_quality_gate_module()
    monkeypatch.delenv("STREAMING_MANIFEST_QUALITY_EVAL_JSONL", raising=False)
    monkeypatch.setenv("STREAMING_MANIFEST_QUALITY_EVAL_OFFLINE_JSON", str(tmp_path / "o.json"))
    def _must_not_run(*_args, **_kwargs):
        raise AssertionError("must not run")

    monkeypatch.setattr(qg, "_run", _must_not_run)

    steps = qg._run_streaming_manifest_quality_gates(
        repo=tmp_path, python_bin="python", strict_release=True
    )
    assert len(steps) == 1 and not steps[0].passed
    assert "set both or neither" in steps[0].note


def test_streaming_manifest_quality_gate_strict_reports_missing_committed_offline(
    monkeypatch,
    tmp_path,
):
    qg = _load_quality_gate_module()
    monkeypatch.delenv("STREAMING_MANIFEST_QUALITY_EVAL_JSONL", raising=False)
    monkeypatch.delenv("STREAMING_MANIFEST_QUALITY_EVAL_OFFLINE_JSON", raising=False)
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    manifest = tmp_path / qg.STREAMING_MANIFEST_STRICT_DEFAULT_MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"sample_id": "s1", "audio_path": str(audio)}) + "\n", encoding="utf-8"
    )
    # Deliberately no offline artifact.
    steps = qg._run_streaming_manifest_quality_gates(
        repo=tmp_path, python_bin="python", strict_release=True
    )
    assert len(steps) == 1 and not steps[0].passed
    assert "Committed offline artifact missing" in steps[0].note


def test_streaming_manifest_quality_gate_strict_empty_pp_env_keeps_default_ceiling(
    monkeypatch,
    tmp_path,
):
    qg = _load_quality_gate_module()
    manifest = tmp_path / "m.jsonl"
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    manifest.write_text(
        json.dumps({"sample_id": "s1", "audio_path": str(audio)}) + "\n", encoding="utf-8"
    )
    offline = tmp_path / "offline.json"
    offline.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("STREAMING_MANIFEST_QUALITY_EVAL_JSONL", str(manifest))
    monkeypatch.setenv("STREAMING_MANIFEST_QUALITY_EVAL_OFFLINE_JSON", str(offline))
    monkeypatch.setenv("STREAMING_MANIFEST_QUALITY_EVAL_FAIL_PRIMARY_ABOVE_OFFLINE_PP", "")

    cmds: list[list[str]] = []
    monkeypatch.setattr(
        qg,
        "_run",
        lambda cmd, _cwd, env=None: (  # noqa: ARG005
            cmds.append(cmd),
            qg.StepResult(name="python", cmd="", passed=True, duration_sec=0.0, returncode=0),
        )[1],
    )
    qg._run_streaming_manifest_quality_gates(
        repo=tmp_path, python_bin="python", strict_release=True
    )
    assert cmds[0][cmds[0].index("--fail-primary-above-offline-pp") + 1] == "3.0"


def test_streaming_manifest_quality_gate_explicit_manifest_runs_single_lane(
    monkeypatch,
    tmp_path,
):
    qg = _load_quality_gate_module()
    manifest = tmp_path / "custom.jsonl"
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    manifest.write_text(
        json.dumps({"sample_id": "s1", "audio_path": str(audio)}) + "\n", encoding="utf-8"
    )
    offline = tmp_path / "offline.json"
    offline.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("STREAMING_MANIFEST_QUALITY_EVAL_JSONL", str(manifest))
    monkeypatch.setenv("STREAMING_MANIFEST_QUALITY_EVAL_OFFLINE_JSON", str(offline))
    monkeypatch.delenv("STREAMING_MANIFEST_QUALITY_EVAL_JSON_OUTPUT", raising=False)

    cmds: list[list[str]] = []
    monkeypatch.setattr(
        qg,
        "_run",
        lambda cmd, _cwd, env=None: (  # noqa: ARG005
            cmds.append(cmd),
            qg.StepResult(name="python", cmd="", passed=True, duration_sec=0.0, returncode=0),
        )[1],
    )

    steps = qg._run_streaming_manifest_quality_gates(
        repo=tmp_path, python_bin="python", strict_release=True
    )
    assert len(steps) == 1
    assert cmds[0][cmds[0].index("--manifest-jsonl") + 1] == str(manifest.resolve())


def test_streaming_manifest_quality_gate_non_strict_has_no_reference_ceiling(
    monkeypatch,
    tmp_path,
):
    qg = _load_quality_gate_module()
    manifest = tmp_path / "streaming_manifest.jsonl"
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    manifest.write_text(
        json.dumps({"sample_id": "s1", "audio_path": str(audio)}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("STREAMING_MANIFEST_QUALITY_EVAL_JSONL", str(manifest))
    monkeypatch.delenv("STREAMING_MANIFEST_QUALITY_EVAL_OFFLINE_JSON", raising=False)

    called: dict[str, object] = {}

    def _fake_run(cmd, _cwd, env=None):  # noqa: ANN001, ANN002, ARG001
        called["cmd"] = cmd
        return qg.StepResult(
            name="python", cmd=" ".join(cmd), passed=True, duration_sec=0.0, returncode=0
        )

    monkeypatch.setattr(qg, "_run", _fake_run)

    (step,) = qg._run_streaming_manifest_quality_gates(
        repo=tmp_path,
        python_bin="python",
        strict_release=False,
    )

    assert step.passed
    cmd = called["cmd"]
    assert isinstance(cmd, list)
    assert "--offline-quality-json" not in cmd
    assert "--fail-primary-above-offline-pp" not in cmd
