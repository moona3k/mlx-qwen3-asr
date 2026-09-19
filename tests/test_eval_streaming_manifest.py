"""Unit tests for scripts/eval_streaming_manifest.py helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "eval_streaming_manifest.py"
    module_name = "eval_streaming_manifest_script"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_split_modes_rejects_invalid_mode():
    mod = _load_script_module()
    with pytest.raises(ValueError, match="Unsupported endpointing mode"):
        mod._split_modes("fixed,vad")  # noqa: SLF001


def test_parse_manifest_requires_audio_path(tmp_path: Path):
    mod = _load_script_module()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"sample_id": "s1"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing audio_path"):
        mod._parse_manifest(manifest)  # noqa: SLF001


def test_sha256_file(tmp_path: Path):
    mod = _load_script_module()
    data = tmp_path / "data.bin"
    data.write_bytes(b"abc123")
    assert mod._sha256_file(data) == (
        "6ca13d52ca70c883e0f0bb101e425a89e8624de51db2d2392593af6a84118090"
    )


def test_main_emits_multifile_payload(monkeypatch, tmp_path: Path):
    mod = _load_script_module()

    a1 = tmp_path / "a1.wav"
    a2 = tmp_path / "a2.wav"
    a1.write_bytes(b"RIFF")
    a2.write_bytes(b"RIFF")

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"sample_id": "s1", "audio_path": str(a1), "subset": "set-a"}),
                json.dumps({"sample_id": "s2", "audio_path": str(a2), "subset": "set-b"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def _fake_load_model(_name, dtype=None):  # noqa: ANN001, ARG001
        return object(), {"dtype": str(dtype)}

    def _fake_load_audio(_path):  # noqa: ANN001
        return np.zeros(32000, dtype=np.float32)

    def _fake_init_streaming(**kwargs):  # noqa: ANN003
        return SimpleNamespace(
            chunk_id=0,
            text="",
            language="unknown",
            endpointing_mode=kwargs["endpointing_mode"],
        )

    def _fake_feed_audio(_chunk, state, model=None):  # noqa: ANN001, ANN002, ARG001
        state.chunk_id += 1

    def _fake_finish_streaming(state, model=None):  # noqa: ANN001, ARG001
        state.text = f"done-{state.endpointing_mode}"
        state.language = "English"

    def _fake_streaming_metrics(state):  # noqa: ANN001
        if state.endpointing_mode == "energy":
            return {
                "partial_stability": 0.94,
                "rewrite_rate": 0.04,
                "finalization_delta_chars": 2,
            }
        return {
            "partial_stability": 0.90,
            "rewrite_rate": 0.06,
            "finalization_delta_chars": 3,
        }

    monkeypatch.setattr(mod, "load_model", _fake_load_model)
    monkeypatch.setattr(mod, "load_audio", _fake_load_audio)
    monkeypatch.setattr(mod, "init_streaming", _fake_init_streaming)
    monkeypatch.setattr(mod, "feed_audio", _fake_feed_audio)
    monkeypatch.setattr(mod, "finish_streaming", _fake_finish_streaming)
    monkeypatch.setattr(mod, "streaming_metrics", _fake_streaming_metrics)
    monkeypatch.setattr(mod, "_git_head_commit", lambda _repo_root: "deadbeef")
    monkeypatch.setattr(mod, "_iso_utc_now", lambda: "2026-02-16T11:22:33+00:00")

    output = tmp_path / "streaming_manifest.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_streaming_manifest.py",
            "--manifest-jsonl",
            str(manifest),
            "--endpointing-modes",
            "fixed,energy",
            "--json-output",
            str(output),
        ],
    )

    rc = mod.main()
    assert rc == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == mod.SCHEMA_VERSION
    assert payload["quality_vs_reference"] is None
    assert payload["suite"] == "streaming-manifest-quality-v1"
    assert payload["generated_at_utc"] == "2026-02-16T11:22:33+00:00"
    assert payload["git_commit"] == "deadbeef"
    assert payload["manifest_sha256"] == mod._sha256_file(manifest)  # noqa: SLF001
    assert payload["samples"] == 2
    assert payload["evaluations"] == 4
    assert set(payload["by_mode"].keys()) == {"fixed", "energy"}
    assert payload["aggregate"]["partial_stability_mean"] > 0.0


def test_main_fails_threshold(monkeypatch, tmp_path: Path):
    mod = _load_script_module()

    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"sample_id": "s1", "audio_path": str(audio)}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "load_model", lambda _name, dtype=None: (object(), {}))
    monkeypatch.setattr(mod, "load_audio", lambda _path: np.zeros(16000, dtype=np.float32))
    monkeypatch.setattr(
        mod,
        "init_streaming",
        lambda **kwargs: SimpleNamespace(
            chunk_id=0,
            text="",
            language="unknown",
            endpointing_mode=kwargs["endpointing_mode"],
        ),
    )
    monkeypatch.setattr(
        mod,
        "feed_audio",
        lambda _chunk, state, model=None: setattr(state, "chunk_id", state.chunk_id + 1),
    )
    monkeypatch.setattr(mod, "finish_streaming", lambda state, model=None: None)
    monkeypatch.setattr(
        mod,
        "streaming_metrics",
        lambda _state: {
            "partial_stability": 0.10,
            "rewrite_rate": 0.80,
            "finalization_delta_chars": 100,
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_streaming_manifest.py",
            "--manifest-jsonl",
            str(manifest),
            "--endpointing-modes",
            "fixed",
            "--fail-partial-stability-below",
            "0.85",
            "--fail-rewrite-rate-above",
            "0.30",
            "--fail-finalization-delta-chars-above",
            "32",
        ],
    )

    rc = mod.main()
    assert rc == 2


_BENCHMARKS = Path(__file__).resolve().parents[1] / "docs" / "benchmarks"
_MULTILINGUAL_MANIFEST = _BENCHMARKS / "2026-09-07-fleurs-multilingual-100-manifest.jsonl"
_OFFLINE_MULTILINGUAL = _BENCHMARKS / "2026-09-07-manifest-quality-multilingual100-0p6b.json"
_PRE_FIX_ARTIFACT = (
    _BENCHMARKS / "2026-09-19-streaming-manifest-multilingual100-incremental-kv.json"
)
_POST_FIX_ARTIFACT = _BENCHMARKS / "2026-09-19-streaming-manifest-multilingual100.json"


def _rescore_committed_artifact(mod, artifact: Path) -> tuple[list[dict], dict]:
    rows = json.loads(artifact.read_text(encoding="utf-8"))["rows"]
    references = mod._load_references(_MULTILINGUAL_MANIFEST)  # noqa: SLF001
    quality = mod._score_rows_against_references(rows, references)  # noqa: SLF001
    assert quality is not None
    quality["offline"] = mod._load_offline_quality(  # noqa: SLF001
        _OFFLINE_MULTILINGUAL, manifest_path=_MULTILINGUAL_MANIFEST
    )
    return rows, quality


def test_reference_gate_would_have_failed_pre_fix_streaming_decoder():
    """The gate must catch the 2026-09-19 regression from the committed artifact.

    The incremental-KV decoder scored partial_stability 1.0 and passed every
    stability threshold while producing 56% primary error. Re-scoring its
    ``final_text`` against the manifest references through the new gate must
    fail at offline + 3pp; otherwise the gate does not do what it claims.
    """
    mod = _load_script_module()
    rows, quality = _rescore_committed_artifact(mod, _PRE_FIX_ARTIFACT)

    assert quality["scored_rows"] == len(rows) == 200
    assert quality["unscored_rows"] == 0
    # Matches the hand-computed block committed in the artifact.
    committed = json.loads(_PRE_FIX_ARTIFACT.read_text(encoding="utf-8"))[
        "quality_vs_reference"
    ]
    for mode in ("fixed", "energy"):
        assert quality["by_mode"][mode]["primary_error_rate"] == pytest.approx(
            committed["by_mode"][mode]["primary_error_rate"], abs=1e-9
        )
    assert quality["worst_mode_primary_error_rate"] > 0.5

    aggregate = json.loads(_PRE_FIX_ARTIFACT.read_text(encoding="utf-8"))["aggregate"]
    failures = mod._threshold_failures(  # noqa: SLF001
        aggregate=aggregate,
        fail_partial_stability_below=0.85,
        fail_rewrite_rate_above=0.85,
        fail_finalization_delta_chars_above=32,
        quality=quality,
        fail_primary_above_offline_pp=3.0,
    )
    reference_failures = [f for f in failures if f.startswith("Streaming reference gate")]
    assert len(reference_failures) == 1, failures
    assert "worst_mode_primary_error_rate=0.576482" in reference_failures[0]
    assert "offline 0.095350 + 3.00pp" in reference_failures[0]

    # The stability and rewrite thresholds let it through (stability was a
    # perfect 1.0). Only the single-sample finalization max (72 chars) would
    # have tripped, and only in strict mode, which never ran this lane.
    stability_only = mod._threshold_failures(  # noqa: SLF001
        aggregate=aggregate,
        fail_partial_stability_below=0.85,
        fail_rewrite_rate_above=0.85,
        fail_finalization_delta_chars_above=None,
    )
    assert stability_only == []
    assert aggregate["partial_stability_mean"] == 1.0


def test_reference_gate_passes_post_fix_streaming_decoder():
    mod = _load_script_module()
    _rows, quality = _rescore_committed_artifact(mod, _POST_FIX_ARTIFACT)

    assert quality["worst_mode_primary_error_rate"] < 0.12
    aggregate = json.loads(_POST_FIX_ARTIFACT.read_text(encoding="utf-8"))["aggregate"]
    failures = mod._threshold_failures(  # noqa: SLF001
        aggregate=aggregate,
        fail_partial_stability_below=0.85,
        fail_rewrite_rate_above=0.85,
        fail_finalization_delta_chars_above=32,
        quality=quality,
        fail_primary_above_offline_pp=3.0,
    )
    assert failures == []


_LONGFORM_MANIFEST = _BENCHMARKS / "2026-09-07-fleurs-longform-10x75-manifest.jsonl"
_OFFLINE_LONGFORM = _BENCHMARKS / "2026-09-07-manifest-quality-longform10-0p6b.json"
_POST_FIX_LONGFORM_ARTIFACT = _BENCHMARKS / "2026-09-19-streaming-manifest-longform10.json"


def test_reference_gate_passes_post_fix_longform_lane():
    """The second strict lane (10 x 75 s) must pass at offline + 3pp today."""
    mod = _load_script_module()
    rows = json.loads(_POST_FIX_LONGFORM_ARTIFACT.read_text(encoding="utf-8"))["rows"]
    references = mod._load_references(_LONGFORM_MANIFEST)  # noqa: SLF001
    quality = mod._score_rows_against_references(rows, references)  # noqa: SLF001
    assert quality is not None and quality["scored_rows"] == len(rows) == 20
    quality["offline"] = mod._load_offline_quality(  # noqa: SLF001
        _OFFLINE_LONGFORM, manifest_path=_LONGFORM_MANIFEST
    )
    committed = json.loads(_POST_FIX_LONGFORM_ARTIFACT.read_text(encoding="utf-8"))[
        "quality_vs_reference"
    ]
    for mode in ("fixed", "energy"):
        assert quality["by_mode"][mode]["primary_error_rate"] == pytest.approx(
            committed["by_mode"][mode]["primary_error_rate"], abs=1e-9
        )
    failures = mod._threshold_failures(  # noqa: SLF001
        aggregate={},
        fail_partial_stability_below=None,
        fail_rewrite_rate_above=None,
        fail_finalization_delta_chars_above=None,
        quality=quality,
        fail_primary_above_offline_pp=3.0,
    )
    assert failures == []
    # Headroom is real but not large: record it so a drift shows up in review.
    headroom_pp = (
        quality["offline"]["primary_error_rate"] + 0.03
        - quality["worst_mode_primary_error_rate"]
    ) * 100
    assert 1.0 < headroom_pp < 2.0


def test_load_offline_quality_rejects_other_manifest(tmp_path: Path):
    mod = _load_script_module()
    with pytest.raises(ValueError, match="different manifest"):
        mod._load_offline_quality(  # noqa: SLF001
            _OFFLINE_MULTILINGUAL, manifest_path=tmp_path / "other.jsonl"
        )


def test_reference_gate_requires_references_when_threshold_requested():
    mod = _load_script_module()
    failures = mod._threshold_failures(  # noqa: SLF001
        aggregate={},
        fail_partial_stability_below=None,
        fail_rewrite_rate_above=None,
        fail_finalization_delta_chars_above=None,
        quality=None,
        fail_primary_above=0.2,
    )
    assert len(failures) == 1
    assert "no manifest row carried reference_text" in failures[0]


def test_main_scores_references_and_gates_on_offline_ceiling(monkeypatch, tmp_path: Path):
    mod = _load_script_module()

    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sample_id": "s1",
                        "audio_path": str(audio),
                        "language": "English",
                        "reference_text": "hello world again",
                    }
                ),
                json.dumps(
                    {
                        "sample_id": "s2",
                        "audio_path": str(audio),
                        "language": "Chinese",
                        "reference_text": "你好世界",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    offline = tmp_path / "offline.json"
    offline.write_text(
        json.dumps(
            {
                "manifest_jsonl": str(manifest),
                "model": "Qwen/Qwen3-ASR-0.6B",
                "wer": 0.1,
                "cer": 0.05,
                "primary_error_rate": 0.10,
            }
        ),
        encoding="utf-8",
    )

    hypotheses = {"s1": "hello world", "s2": "你好世界"}

    monkeypatch.setattr(mod, "load_model", lambda _name, dtype=None: (object(), {}))
    monkeypatch.setattr(mod, "load_audio", lambda _path: np.zeros(16000, dtype=np.float32))

    def _fake_init_streaming(**kwargs):  # noqa: ANN003
        return SimpleNamespace(text="", language="unknown", sample_id=None)

    current: dict[str, str] = {}

    def _fake_feed_audio(_chunk, state, model=None):  # noqa: ANN001, ANN002, ARG001
        state.sample_id = current["sample_id"]

    def _fake_finish_streaming(state, model=None):  # noqa: ANN001, ARG001
        state.text = hypotheses[state.sample_id]

    real_parse = mod._parse_manifest  # noqa: SLF001

    def _tracking_parse(path):  # noqa: ANN001
        samples = real_parse(path)
        # Track which sample is being streamed via load_audio ordering.
        order = iter(samples)

        def _load(_p):  # noqa: ANN001
            current["sample_id"] = next(order).sample_id
            return np.zeros(16000, dtype=np.float32)

        monkeypatch.setattr(mod, "load_audio", _load)
        return samples

    monkeypatch.setattr(mod, "_parse_manifest", _tracking_parse)
    monkeypatch.setattr(mod, "init_streaming", _fake_init_streaming)
    monkeypatch.setattr(mod, "feed_audio", _fake_feed_audio)
    monkeypatch.setattr(mod, "finish_streaming", _fake_finish_streaming)
    monkeypatch.setattr(
        mod,
        "streaming_metrics",
        lambda _state: {
            "partial_stability": 1.0,
            "rewrite_rate": 0.0,
            "finalization_delta_chars": 0,
        },
    )

    output = tmp_path / "out.json"
    argv = [
        "eval_streaming_manifest.py",
        "--manifest-jsonl",
        str(manifest),
        "--endpointing-modes",
        "fixed",
        "--offline-quality-json",
        str(offline),
        "--json-output",
        str(output),
        "--fail-primary-above-offline-pp",
    ]
    # s1 drops one of three words (WER 1/3); s2 is exact. Primary errors are
    # pooled: 1 / (3 + 4) = 0.142857. Offline 0.10 + 3pp = 0.13 fails; + 5pp passes.
    monkeypatch.setattr(sys, "argv", [*argv, "3.0"])
    assert mod.main() == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == mod.SCHEMA_VERSION
    quality = payload["quality_vs_reference"]
    assert quality["scored_rows"] == 2
    assert quality["by_mode"]["fixed"]["primary_error_rate"] == pytest.approx(1 / 7)
    assert quality["offline"]["primary_error_rate"] == 0.10
    s1 = next(r for r in payload["rows"] if r["sample_id"] == "s1")
    assert s1["primary_metric"] == "wer"
    assert s1["wer"] == pytest.approx(1 / 3)
    s2 = next(r for r in payload["rows"] if r["sample_id"] == "s2")
    assert s2["primary_metric"] == "cer"
    assert s2["cer"] == 0.0

    monkeypatch.setattr(sys, "argv", [*argv, "5.0"])
    assert mod.main() == 0
