"""Tests for diarization helpers."""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest

from mlx_qwen3_asr.diarization import (
    DEFAULT_DIARIZATION_DEVICE,
    DEFAULT_PYANNOTE_MODEL_ID,
    DEFAULT_SPEAKER_LABEL,
    _resolve_diarization_device,
    build_speaker_segments_from_turns,
    diarize_chunk_items,
    diarize_word_segments,
    infer_speaker_turns,
    validate_diarization_config,
)


class _FakeSegment:
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


class _FakeAnnotation:
    def __init__(self, turns: list[tuple[float, float, str]]):
        self._turns = turns

    def itertracks(self, yield_label: bool = False):
        _ = yield_label
        for start, end, label in self._turns:
            yield _FakeSegment(start, end), None, label


class _RecordingPipeline:
    def __init__(self, annotation):
        self.annotation = annotation
        self.calls: list[tuple[dict, dict]] = []

    def __call__(self, payload, **kwargs):  # noqa: ANN001
        self.calls.append((payload, kwargs))
        return self.annotation


class _FakeDiarizeOutput:
    def __init__(self, *, speaker_diarization, exclusive_speaker_diarization=None):
        self.speaker_diarization = speaker_diarization
        self.exclusive_speaker_diarization = exclusive_speaker_diarization


def test_validate_diarization_config_rejects_invalid_bounds():
    with pytest.raises(ValueError, match="diarization_max_speakers"):
        validate_diarization_config(
            num_speakers=None,
            min_speakers=3,
            max_speakers=2,
        )


def test_infer_speaker_turns_fixed_speaker_count_forwards_num_speakers(monkeypatch):
    dmod = importlib.import_module("mlx_qwen3_asr.diarization")
    monkeypatch.setattr(
        dmod,
        "_pyannote_input",
        lambda audio, sr: {"waveform": audio[None, :], "sample_rate": sr},
    )

    audio = np.zeros((16000,), dtype=np.float32)
    cfg = validate_diarization_config(
        num_speakers=2,
        min_speakers=1,
        max_speakers=4,
    )
    pipe = _RecordingPipeline(_FakeAnnotation([(0.0, 0.5, "A"), (0.5, 1.0, "B")]))
    turns = infer_speaker_turns(audio, sr=16000, config=cfg, _pipeline=pipe)

    assert len(turns) == 2
    assert turns[0]["speaker"] == "SPEAKER_00"
    assert turns[1]["speaker"] == "SPEAKER_01"
    assert pipe.calls[0][1] == {"num_speakers": 2}


def test_infer_speaker_turns_auto_mode_forwards_min_max_speakers(monkeypatch):
    dmod = importlib.import_module("mlx_qwen3_asr.diarization")
    monkeypatch.setattr(
        dmod,
        "_pyannote_input",
        lambda audio, sr: {"waveform": audio[None, :], "sample_rate": sr},
    )

    audio = np.zeros((16000,), dtype=np.float32)
    cfg = validate_diarization_config(
        num_speakers=None,
        min_speakers=1,
        max_speakers=3,
    )
    pipe = _RecordingPipeline(_FakeAnnotation([(0.0, 1.0, "speaker-a")]))
    turns = infer_speaker_turns(audio, sr=16000, config=cfg, _pipeline=pipe)

    assert len(turns) == 1
    assert turns[0]["speaker"] == "SPEAKER_00"
    assert pipe.calls[0][1] == {"min_speakers": 1, "max_speakers": 3}


def test_infer_speaker_turns_returns_default_when_annotation_is_empty(monkeypatch):
    dmod = importlib.import_module("mlx_qwen3_asr.diarization")
    monkeypatch.setattr(
        dmod,
        "_pyannote_input",
        lambda audio, sr: {"waveform": audio[None, :], "sample_rate": sr},
    )

    audio = np.zeros((8000,), dtype=np.float32)
    cfg = validate_diarization_config(
        num_speakers=None,
        min_speakers=1,
        max_speakers=2,
    )
    pipe = _RecordingPipeline(_FakeAnnotation([]))

    turns = infer_speaker_turns(audio, sr=8000, config=cfg, _pipeline=pipe)

    assert turns == [{"speaker": DEFAULT_SPEAKER_LABEL, "start": 0.0, "end": 1.0}]


def test_infer_speaker_turns_merges_adjacent_same_speaker(monkeypatch):
    dmod = importlib.import_module("mlx_qwen3_asr.diarization")
    monkeypatch.setattr(
        dmod,
        "_pyannote_input",
        lambda audio, sr: {"waveform": audio[None, :], "sample_rate": sr},
    )

    audio = np.zeros((16000,), dtype=np.float32)
    cfg = validate_diarization_config(
        num_speakers=1,
        min_speakers=1,
        max_speakers=2,
    )
    pipe = _RecordingPipeline(
        _FakeAnnotation(
            [
                (0.0, 0.4, "same"),
                (0.45, 0.8, "same"),
            ]
        )
    )

    turns = infer_speaker_turns(audio, sr=16000, config=cfg, _pipeline=pipe)

    assert turns == [{"speaker": "SPEAKER_00", "start": 0.0, "end": 0.8}]


def test_infer_speaker_turns_unwraps_pyannote4_exclusive_diarization(monkeypatch):
    dmod = importlib.import_module("mlx_qwen3_asr.diarization")
    monkeypatch.setattr(
        dmod,
        "_pyannote_input",
        lambda audio, sr: {"waveform": audio[None, :], "sample_rate": sr},
    )

    audio = np.zeros((16000,), dtype=np.float32)
    cfg = validate_diarization_config(
        num_speakers=None,
        min_speakers=1,
        max_speakers=2,
    )
    regular = _FakeAnnotation([(0.0, 1.0, "regular")])
    exclusive = _FakeAnnotation([(0.0, 0.5, "exclusive-a"), (0.5, 1.0, "exclusive-b")])
    pipe = _RecordingPipeline(
        _FakeDiarizeOutput(
            speaker_diarization=regular,
            exclusive_speaker_diarization=exclusive,
        )
    )

    turns = infer_speaker_turns(audio, sr=16000, config=cfg, _pipeline=pipe)

    assert turns == [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 0.5},
        {"speaker": "SPEAKER_01", "start": 0.5, "end": 1.0},
    ]


def test_infer_speaker_turns_unwraps_pyannote4_regular_diarization(monkeypatch):
    dmod = importlib.import_module("mlx_qwen3_asr.diarization")
    monkeypatch.setattr(
        dmod,
        "_pyannote_input",
        lambda audio, sr: {"waveform": audio[None, :], "sample_rate": sr},
    )

    audio = np.zeros((16000,), dtype=np.float32)
    cfg = validate_diarization_config(
        num_speakers=None,
        min_speakers=1,
        max_speakers=2,
    )
    pipe = _RecordingPipeline(
        _FakeDiarizeOutput(
            speaker_diarization=_FakeAnnotation([(0.0, 1.0, "regular")]),
        )
    )

    turns = infer_speaker_turns(audio, sr=16000, config=cfg, _pipeline=pipe)

    assert turns == [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0}]


def test_infer_speaker_turns_falls_back_when_pyannote4_exclusive_is_empty(monkeypatch):
    dmod = importlib.import_module("mlx_qwen3_asr.diarization")
    monkeypatch.setattr(
        dmod,
        "_pyannote_input",
        lambda audio, sr: {"waveform": audio[None, :], "sample_rate": sr},
    )

    audio = np.zeros((16000,), dtype=np.float32)
    cfg = validate_diarization_config(
        num_speakers=None,
        min_speakers=1,
        max_speakers=2,
    )
    pipe = _RecordingPipeline(
        _FakeDiarizeOutput(
            speaker_diarization=_FakeAnnotation([(0.0, 1.0, "regular")]),
            exclusive_speaker_diarization=_FakeAnnotation([]),
        )
    )

    turns = infer_speaker_turns(audio, sr=16000, config=cfg, _pipeline=pipe)

    assert turns == [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0}]


def test_infer_speaker_turns_raises_helpful_error_when_dependency_missing(monkeypatch):
    dmod = importlib.import_module("mlx_qwen3_asr.diarization")

    def _raise_import_error(device=DEFAULT_DIARIZATION_DEVICE):
        _ = device
        raise ImportError("missing pyannote")

    monkeypatch.setattr(dmod, "_load_pyannote_pipeline", _raise_import_error)

    cfg = validate_diarization_config(
        num_speakers=1,
        min_speakers=1,
        max_speakers=2,
    )

    with pytest.raises(ImportError, match="missing pyannote"):
        infer_speaker_turns(np.zeros((8000,), dtype=np.float32), sr=8000, config=cfg)


def test_infer_speaker_turns_wraps_pipeline_runtime_errors(monkeypatch):
    dmod = importlib.import_module("mlx_qwen3_asr.diarization")
    monkeypatch.setattr(
        dmod,
        "_pyannote_input",
        lambda audio, sr: {"waveform": audio[None, :], "sample_rate": sr},
    )

    class _FailingPipeline:
        def __call__(self, payload, **kwargs):  # noqa: ANN001
            _ = payload, kwargs
            raise RuntimeError("backend exploded")

    cfg = validate_diarization_config(
        num_speakers=1,
        min_speakers=1,
        max_speakers=2,
    )

    with pytest.raises(RuntimeError, match="Root cause: RuntimeError: backend exploded"):
        infer_speaker_turns(
            np.zeros((8000,), dtype=np.float32),
            sr=8000,
            config=cfg,
            _pipeline=_FailingPipeline(),
        )


def test_infer_speaker_turns_retries_on_speaker_kwargs_type_error(monkeypatch):
    dmod = importlib.import_module("mlx_qwen3_asr.diarization")
    monkeypatch.setattr(
        dmod,
        "_pyannote_input",
        lambda audio, sr: {"waveform": audio[None, :], "sample_rate": sr},
    )

    class _RetryPipeline:
        def __init__(self):
            self.calls = 0

        def __call__(self, payload, **kwargs):  # noqa: ANN001
            _ = payload
            self.calls += 1
            if self.calls == 1:
                raise TypeError("got an unexpected keyword argument 'min_speakers'")
            assert kwargs == {}
            return _FakeAnnotation([(0.0, 0.8, "spk")])

    cfg = validate_diarization_config(
        num_speakers=None,
        min_speakers=1,
        max_speakers=2,
    )
    pipe = _RetryPipeline()

    with pytest.warns(UserWarning, match="rejected speaker-count kwargs"):
        turns = infer_speaker_turns(
            np.zeros((8000,), dtype=np.float32),
            sr=8000,
            config=cfg,
            _pipeline=pipe,
        )

    assert pipe.calls == 2
    assert turns == [{"speaker": "SPEAKER_00", "start": 0.0, "end": 0.8}]


def test_infer_speaker_turns_does_not_retry_on_non_kwargs_type_error(monkeypatch):
    dmod = importlib.import_module("mlx_qwen3_asr.diarization")
    monkeypatch.setattr(
        dmod,
        "_pyannote_input",
        lambda audio, sr: {"waveform": audio[None, :], "sample_rate": sr},
    )

    class _TypeErrorPipeline:
        def __init__(self):
            self.calls = 0

        def __call__(self, payload, **kwargs):  # noqa: ANN001
            _ = payload, kwargs
            self.calls += 1
            raise TypeError("shape mismatch in backend")

    cfg = validate_diarization_config(
        num_speakers=1,
        min_speakers=1,
        max_speakers=2,
    )
    pipe = _TypeErrorPipeline()

    with pytest.raises(RuntimeError, match="Root cause: TypeError: shape mismatch"):
        infer_speaker_turns(
            np.zeros((8000,), dtype=np.float32),
            sr=8000,
            config=cfg,
            _pipeline=pipe,
        )
    assert pipe.calls == 1


def test_load_pyannote_pipeline_uses_pyannote4_token_kwarg(monkeypatch):
    dmod = importlib.import_module("mlx_qwen3_asr.diarization")
    calls = {}

    class _FakePipeline:
        @classmethod
        def from_pretrained(cls, model_id, *, token=None):  # noqa: ANN001
            calls["model_id"] = model_id
            calls["token"] = token
            return object()

    fake_audio_module = types.ModuleType("pyannote.audio")
    fake_audio_module.Pipeline = _FakePipeline
    fake_pkg = types.ModuleType("pyannote")
    fake_pkg.audio = fake_audio_module

    monkeypatch.setitem(sys.modules, "pyannote", fake_pkg)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio_module)
    monkeypatch.setenv("PYANNOTE_AUTH_TOKEN", "hf_test")
    monkeypatch.delenv("PYANNOTE_MODEL_ID", raising=False)
    diarization = importlib.import_module("mlx_qwen3_asr.diarization")
    monkeypatch.setattr(diarization, "_UNAVAILABLE_DIARIZATION_DEVICES", set())
    dmod._PYANNOTE_PIPELINE_CACHE.clear()  # noqa: SLF001

    dmod._load_pyannote_pipeline()  # noqa: SLF001

    assert calls == {
        "model_id": DEFAULT_PYANNOTE_MODEL_ID,
        "token": "hf_test",
    }


def test_load_pyannote_pipeline_defaults_to_token_for_kwargs_signature(monkeypatch):
    dmod = importlib.import_module("mlx_qwen3_asr.diarization")
    calls = {}

    class _FakePipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):  # noqa: ANN001
            calls["model_id"] = model_id
            calls["kwargs"] = kwargs
            return object()

    fake_audio_module = types.ModuleType("pyannote.audio")
    fake_audio_module.Pipeline = _FakePipeline
    fake_pkg = types.ModuleType("pyannote")
    fake_pkg.audio = fake_audio_module

    monkeypatch.setitem(sys.modules, "pyannote", fake_pkg)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio_module)
    monkeypatch.setenv("PYANNOTE_AUTH_TOKEN", "hf_test")
    monkeypatch.delenv("PYANNOTE_MODEL_ID", raising=False)
    diarization = importlib.import_module("mlx_qwen3_asr.diarization")
    monkeypatch.setattr(diarization, "_UNAVAILABLE_DIARIZATION_DEVICES", set())
    dmod._PYANNOTE_PIPELINE_CACHE.clear()  # noqa: SLF001

    dmod._load_pyannote_pipeline()  # noqa: SLF001

    assert calls == {
        "model_id": DEFAULT_PYANNOTE_MODEL_ID,
        "kwargs": {"token": "hf_test"},
    }


def test_load_pyannote_pipeline_falls_back_to_pyannote3_auth_kwarg(monkeypatch):
    dmod = importlib.import_module("mlx_qwen3_asr.diarization")
    calls = {}

    class _FakePipeline:
        @classmethod
        def from_pretrained(cls, model_id, *, use_auth_token=None):  # noqa: ANN001
            calls["model_id"] = model_id
            calls["use_auth_token"] = use_auth_token
            return object()

    fake_audio_module = types.ModuleType("pyannote.audio")
    fake_audio_module.Pipeline = _FakePipeline
    fake_pkg = types.ModuleType("pyannote")
    fake_pkg.audio = fake_audio_module

    monkeypatch.setitem(sys.modules, "pyannote", fake_pkg)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio_module)
    monkeypatch.setenv("PYANNOTE_AUTH_TOKEN", "hf_test")
    monkeypatch.setenv("PYANNOTE_MODEL_ID", "pyannote/speaker-diarization-3.1")
    dmod._PYANNOTE_PIPELINE_CACHE.clear()  # noqa: SLF001

    dmod._load_pyannote_pipeline()  # noqa: SLF001

    assert calls == {
        "model_id": "pyannote/speaker-diarization-3.1",
        "use_auth_token": "hf_test",
    }


def test_load_pyannote_pipeline_wraps_from_pretrained_errors(monkeypatch):
    dmod = importlib.import_module("mlx_qwen3_asr.diarization")

    class _FakePipeline:
        @classmethod
        def from_pretrained(cls, model_id, *, token=None):  # noqa: ANN001
            _ = model_id, token
            raise RuntimeError("401 unauthorized")

    fake_audio_module = types.ModuleType("pyannote.audio")
    fake_audio_module.Pipeline = _FakePipeline
    fake_pkg = types.ModuleType("pyannote")
    fake_pkg.audio = fake_audio_module

    monkeypatch.setitem(sys.modules, "pyannote", fake_pkg)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio_module)
    monkeypatch.delenv("PYANNOTE_MODEL_ID", raising=False)
    diarization = importlib.import_module("mlx_qwen3_asr.diarization")
    monkeypatch.setattr(diarization, "_UNAVAILABLE_DIARIZATION_DEVICES", set())
    dmod._PYANNOTE_PIPELINE_CACHE.clear()  # noqa: SLF001

    with pytest.raises(RuntimeError, match="Root cause: RuntimeError: 401 unauthorized"):
        dmod._load_pyannote_pipeline()  # noqa: SLF001


def test_load_pyannote_pipeline_rejects_none_return(monkeypatch):
    dmod = importlib.import_module("mlx_qwen3_asr.diarization")

    class _FakePipeline:
        @classmethod
        def from_pretrained(cls, model_id, *, token=None):  # noqa: ANN001
            _ = model_id, token
            return None

    fake_audio_module = types.ModuleType("pyannote.audio")
    fake_audio_module.Pipeline = _FakePipeline
    fake_pkg = types.ModuleType("pyannote")
    fake_pkg.audio = fake_audio_module

    monkeypatch.setitem(sys.modules, "pyannote", fake_pkg)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio_module)
    monkeypatch.delenv("PYANNOTE_MODEL_ID", raising=False)
    diarization = importlib.import_module("mlx_qwen3_asr.diarization")
    monkeypatch.setattr(diarization, "_UNAVAILABLE_DIARIZATION_DEVICES", set())
    dmod._PYANNOTE_PIPELINE_CACHE.clear()  # noqa: SLF001

    with pytest.raises(
        RuntimeError,
        match=r"Root cause: Pipeline\.from_pretrained returned None",
    ):
        dmod._load_pyannote_pipeline()  # noqa: SLF001


def test_diarize_word_segments_adds_speaker_labels():
    cfg = validate_diarization_config(
        num_speakers=None,
        min_speakers=1,
        max_speakers=8,
    )
    words = [
        {"text": "hello", "start": 0.1, "end": 0.3},
        {"text": "world", "start": 0.35, "end": 0.6},
    ]
    labeled, speakers = diarize_word_segments(words, config=cfg)
    assert labeled[0]["speaker"] == DEFAULT_SPEAKER_LABEL
    assert speakers[0]["speaker"] == DEFAULT_SPEAKER_LABEL
    assert speakers[0]["text"] == "hello world"


def test_diarize_chunk_items_returns_fallback_speaker_segments():
    cfg = validate_diarization_config(
        num_speakers=None,
        min_speakers=1,
        max_speakers=8,
    )
    chunks = [
        {"text": "hello", "start": 0.0, "end": 0.8},
        {"text": "world", "start": 1.0, "end": 1.5},
    ]
    speaker_segments = diarize_chunk_items(chunks, config=cfg)
    assert len(speaker_segments) == 1
    assert speaker_segments[0]["speaker"] == DEFAULT_SPEAKER_LABEL
    assert speaker_segments[0]["text"] == "hello world"


def test_diarize_word_segments_uses_turn_overlap():
    cfg = validate_diarization_config(
        num_speakers=2,
        min_speakers=1,
        max_speakers=4,
    )
    turns = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
        {"speaker": "SPEAKER_01", "start": 1.0, "end": 2.0},
    ]
    words = [
        {"text": "hello", "start": 0.1, "end": 0.4},
        {"text": "world", "start": 1.2, "end": 1.6},
    ]
    labeled, speaker_segments = diarize_word_segments(
        words,
        config=cfg,
        speaker_turns=turns,
    )
    assert labeled[0]["speaker"] == "SPEAKER_00"
    assert labeled[1]["speaker"] == "SPEAKER_01"
    assert len(speaker_segments) == 2


def test_build_speaker_segments_from_turns_keeps_empty_turn_text():
    turns = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
        {"speaker": "SPEAKER_01", "start": 1.0, "end": 2.0},
    ]
    words = [{"text": "hello", "start": 0.1, "end": 0.4, "speaker": "SPEAKER_00"}]

    speaker_segments = build_speaker_segments_from_turns(
        speaker_turns=turns,
        word_segments=words,
    )

    assert len(speaker_segments) == 2
    assert speaker_segments[0]["speaker"] == "SPEAKER_00"
    assert speaker_segments[0]["text"] == "hello"
    assert speaker_segments[1]["speaker"] == "SPEAKER_01"
    assert speaker_segments[1]["text"] == ""


class _FakeBackendFlag:
    def __init__(self, available: bool):
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeTorch:
    """Minimal torch stand-in for device resolution tests."""

    def __init__(self, *, mps: bool = False, cuda: bool = False):
        self.backends = types.SimpleNamespace(mps=_FakeBackendFlag(mps))
        self.cuda = _FakeBackendFlag(cuda)

    def device(self, name: str) -> str:
        return f"device:{name}"


class _MovablePipeline:
    """Fake pipeline that can fail accelerator transfers but accept CPU ones.

    Mirrors pyannote's ``Pipeline.to``, which moves sub-models in a loop: a
    mid-loop failure leaves earlier components on the accelerator, so the
    recovery path must be able to move what is left back to CPU.
    """

    def __init__(self, fail_accelerator: bool = False, fail_cpu: bool = False):
        self.moved_to: list[str] = []
        self._fail_accelerator = fail_accelerator
        self._fail_cpu = fail_cpu

    def to(self, device):
        is_cpu = str(device).endswith("cpu")
        if is_cpu and self._fail_cpu:
            raise RuntimeError("cpu transfer is broken too")
        if not is_cpu and self._fail_accelerator:
            self.moved_to.append(f"{device}:partial")
            raise RuntimeError("backend does not support this op")
        self.moved_to.append(device)
        return self


def test_validate_diarization_config_defaults_device_to_auto():
    cfg = validate_diarization_config(
        num_speakers=None, min_speakers=1, max_speakers=8
    )
    assert cfg.device == "auto"


def test_validate_diarization_config_rejects_unknown_device():
    with pytest.raises(ValueError, match="diarization_device"):
        validate_diarization_config(
            num_speakers=None, min_speakers=1, max_speakers=8, device="tpu"
        )


@pytest.mark.parametrize(
    ("mps", "cuda", "expected"),
    [
        (True, False, "mps"),
        (False, True, "cuda"),
        (False, False, "cpu"),
        (True, True, "mps"),
    ],
)
def test_resolve_diarization_device_auto_prefers_fastest(mps, cuda, expected):
    torch_stub = _FakeTorch(mps=mps, cuda=cuda)
    assert _resolve_diarization_device("auto", torch_stub) == expected


def test_resolve_diarization_device_respects_explicit_choice():
    torch_stub = _FakeTorch(mps=True)
    assert _resolve_diarization_device("cpu", torch_stub) == "cpu"


def test_resolve_diarization_device_survives_torch_without_mps_attr():
    torch_stub = _FakeTorch()
    torch_stub.backends = types.SimpleNamespace()
    assert _resolve_diarization_device("auto", torch_stub) == "cpu"


def _isolate_pyannote_env(monkeypatch):
    """Keep cache-key assertions independent of the developer's environment."""
    for var in ("PYANNOTE_AUTH_TOKEN", "HUGGINGFACE_TOKEN", "HF_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("PYANNOTE_MODEL_ID", raising=False)
    diarization = importlib.import_module("mlx_qwen3_asr.diarization")
    monkeypatch.setattr(diarization, "_UNAVAILABLE_DIARIZATION_DEVICES", set())


def _install_fake_pyannote(monkeypatch, pipeline):
    fake_module = types.ModuleType("pyannote.audio")
    fake_module.Pipeline = types.SimpleNamespace(
        from_pretrained=lambda model_id, **kwargs: pipeline
    )
    monkeypatch.setitem(sys.modules, "pyannote", types.ModuleType("pyannote"))
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_module)


def test_load_pyannote_pipeline_moves_to_resolved_device(monkeypatch):
    diarization = importlib.import_module("mlx_qwen3_asr.diarization")
    monkeypatch.setattr(diarization, "_PYANNOTE_PIPELINE_CACHE", {})
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch(mps=True))
    pipeline = _MovablePipeline()
    _install_fake_pyannote(monkeypatch, pipeline)

    loaded = diarization._load_pyannote_pipeline("auto")

    assert loaded is pipeline
    assert pipeline.moved_to == ["device:mps"]


def test_load_pyannote_pipeline_skips_move_for_cpu(monkeypatch):
    diarization = importlib.import_module("mlx_qwen3_asr.diarization")
    monkeypatch.setattr(diarization, "_PYANNOTE_PIPELINE_CACHE", {})
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch(mps=True))
    pipeline = _MovablePipeline()
    _install_fake_pyannote(monkeypatch, pipeline)

    diarization._load_pyannote_pipeline("cpu")

    assert pipeline.moved_to == []


def test_load_pyannote_pipeline_falls_back_when_move_fails(monkeypatch):
    _isolate_pyannote_env(monkeypatch)
    diarization = importlib.import_module("mlx_qwen3_asr.diarization")
    cache: dict = {}
    monkeypatch.setattr(diarization, "_PYANNOTE_PIPELINE_CACHE", cache)
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch(mps=True))
    pipeline = _MovablePipeline(fail_accelerator=True)
    _install_fake_pyannote(monkeypatch, pipeline)

    with pytest.warns(UserWarning, match="falling back to CPU"):
        loaded = diarization._load_pyannote_pipeline("auto")

    assert loaded is pipeline
    # The partially moved pipeline must be pulled back to CPU before caching,
    # otherwise a later CPU caller inherits a mixed-device pipeline.
    assert pipeline.moved_to == ["device:mps:partial", "device:cpu"]
    assert set(cache) == {(DEFAULT_PYANNOTE_MODEL_ID, "", "cpu")}


def test_load_pyannote_pipeline_raises_when_cpu_recovery_also_fails(monkeypatch):
    _isolate_pyannote_env(monkeypatch)
    diarization = importlib.import_module("mlx_qwen3_asr.diarization")
    cache: dict = {}
    monkeypatch.setattr(diarization, "_PYANNOTE_PIPELINE_CACHE", cache)
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch(mps=True))
    pipeline = _MovablePipeline(fail_accelerator=True, fail_cpu=True)
    _install_fake_pyannote(monkeypatch, pipeline)

    with pytest.warns(UserWarning), pytest.raises(RuntimeError, match="mixed devices"):
        diarization._load_pyannote_pipeline("auto")

    assert cache == {}, "an unusable pipeline must never be cached"


def test_load_pyannote_pipeline_caches_per_device(monkeypatch):
    _isolate_pyannote_env(monkeypatch)
    diarization = importlib.import_module("mlx_qwen3_asr.diarization")
    cache: dict = {}
    monkeypatch.setattr(diarization, "_PYANNOTE_PIPELINE_CACHE", cache)
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch(mps=True))
    _install_fake_pyannote(monkeypatch, _MovablePipeline())

    diarization._load_pyannote_pipeline("cpu")
    diarization._load_pyannote_pipeline("mps")

    assert {key[2] for key in cache} == {"cpu", "mps"}


def test_load_pyannote_pipeline_does_not_retry_a_device_that_already_failed(monkeypatch):
    """A failed accelerator must be remembered for the rest of the process.

    The fallback pipeline is cached under the CPU key, so without this the next
    `auto` call misses its own key, reloads the pipeline, retries the same
    doomed transfer, and warns again - every single call.
    """
    _isolate_pyannote_env(monkeypatch)
    diarization = importlib.import_module("mlx_qwen3_asr.diarization")
    monkeypatch.setattr(diarization, "_PYANNOTE_PIPELINE_CACHE", {})
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch(mps=True))

    loads = {"count": 0}

    def _load(model_id, **kwargs):
        loads["count"] += 1
        return _MovablePipeline(fail_accelerator=True)

    fake_module = types.ModuleType("pyannote.audio")
    fake_module.Pipeline = types.SimpleNamespace(from_pretrained=_load)
    monkeypatch.setitem(sys.modules, "pyannote", types.ModuleType("pyannote"))
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_module)

    with pytest.warns(UserWarning, match="falling back to CPU"):
        first = diarization._load_pyannote_pipeline("auto")
    second = diarization._load_pyannote_pipeline("auto")

    assert first is second
    assert loads["count"] == 1, "the failed accelerator was retried"
    assert diarization._UNAVAILABLE_DIARIZATION_DEVICES == {"mps"}


def test_explicit_request_for_a_known_bad_device_resolves_to_cpu(monkeypatch):
    _isolate_pyannote_env(monkeypatch)
    diarization = importlib.import_module("mlx_qwen3_asr.diarization")
    monkeypatch.setattr(diarization, "_UNAVAILABLE_DIARIZATION_DEVICES", {"mps"})
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch(mps=True))

    _, resolved = diarization._resolve_torch_device("mps")

    assert resolved == "cpu"
