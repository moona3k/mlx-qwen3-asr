"""Tests for explicit Session API."""

import asyncio

import mlx.core as mx
import numpy as np

import mlx_qwen3_asr.session as sessmod
from mlx_qwen3_asr.transcribe import TranscriptionResult


def test_session_loads_model_and_tokenizer_from_resolved_path(monkeypatch):
    created = {}

    class _DummyTokenizer:
        def __init__(self, path):  # noqa: ANN001
            created["tokenizer_path"] = path

    class _DummyModel:
        pass

    monkeypatch.setattr(sessmod, "Tokenizer", _DummyTokenizer)
    monkeypatch.setattr(sessmod, "load_model", lambda model, dtype: (_DummyModel(), object()))
    monkeypatch.setattr(sessmod, "_resolve_path", lambda model: "/tmp/resolved-model")

    session = sessmod.Session("repo/a", dtype=mx.float16)
    assert session.model_id == "repo/a"
    assert session.dtype == mx.float16
    assert created["tokenizer_path"] == "/tmp/resolved-model"


def test_session_transcribe_passes_explicit_components(monkeypatch):
    class _DummyTokenizer:
        def __init__(self, path):  # noqa: ANN001
            self.path = path

    class _DummyModel:
        pass

    calls = {}

    monkeypatch.setattr(sessmod, "Tokenizer", _DummyTokenizer)
    monkeypatch.setattr(sessmod, "load_model", lambda model, dtype: (_DummyModel(), object()))
    monkeypatch.setattr(sessmod, "_resolve_path", lambda model: "/tmp/resolved-model")
    monkeypatch.setattr(sessmod, "_to_audio_np", lambda audio: np.zeros(160, dtype=np.float32))
    monkeypatch.setattr(sessmod, "_resolve_aligner", lambda rt, fa: "ALIGNER")
    monkeypatch.setattr(sessmod, "_resolve_draft_model", lambda **kwargs: None)

    def fake_transcribe_loaded_components(**kwargs):  # noqa: ANN003
        calls.update(kwargs)
        return TranscriptionResult(text="ok", language="English")

    monkeypatch.setattr(sessmod, "_transcribe_loaded_components", fake_transcribe_loaded_components)

    session = sessmod.Session("repo/a", dtype=mx.float32)
    out = session.transcribe(
        np.zeros(160, dtype=np.float32),
        language="English",
        return_timestamps=True,
        max_new_tokens=77,
        verbose=True,
    )

    assert out.text == "ok"
    assert calls["dtype"] == mx.float32
    assert calls["language"] == "English"
    assert calls["aligner"] == "ALIGNER"
    assert calls["return_timestamps"] is True
    assert calls["max_new_tokens"] == 77
    assert calls["num_draft_tokens"] == 4
    assert calls["verbose"] is True


def test_session_transcribe_forwards_diarization_config(monkeypatch):
    class _DummyTokenizer:
        def __init__(self, path):  # noqa: ANN001
            self.path = path

    class _DummyModel:
        pass

    calls = {}

    monkeypatch.setattr(sessmod, "Tokenizer", _DummyTokenizer)
    monkeypatch.setattr(sessmod, "load_model", lambda model, dtype: (_DummyModel(), object()))
    monkeypatch.setattr(sessmod, "_resolve_path", lambda model: "/tmp/resolved-model")
    monkeypatch.setattr(sessmod, "_to_audio_np", lambda audio: np.zeros(160, dtype=np.float32))
    monkeypatch.setattr(sessmod, "_resolve_aligner", lambda rt, fa: "ALIGNER")
    monkeypatch.setattr(sessmod, "_resolve_draft_model", lambda **kwargs: None)

    def _fake_diarization_config(**kwargs):  # noqa: ANN003
        return {"enabled": kwargs["diarize"]}

    def fake_transcribe_loaded_components(**kwargs):  # noqa: ANN003
        calls.update(kwargs)
        return TranscriptionResult(text="ok", language="English")

    monkeypatch.setattr(sessmod, "_resolve_diarization_config", _fake_diarization_config)
    monkeypatch.setattr(sessmod, "_transcribe_loaded_components", fake_transcribe_loaded_components)

    session = sessmod.Session("repo/a", dtype=mx.float32)
    out = session.transcribe(
        np.zeros(160, dtype=np.float32),
        diarize=True,
        diarization_num_speakers=2,
    )

    assert out.text == "ok"
    assert calls["diarization_config"] == {"enabled": True}


def test_session_transcribe_supports_draft_model(monkeypatch):
    class _DummyTokenizer:
        def __init__(self, path):  # noqa: ANN001
            self.path = path

    class _DummyModel:
        pass

    calls = {}

    monkeypatch.setattr(sessmod, "Tokenizer", _DummyTokenizer)
    monkeypatch.setattr(sessmod, "load_model", lambda model, dtype: (_DummyModel(), object()))
    monkeypatch.setattr(sessmod, "_resolve_path", lambda model: "/tmp/resolved-model")
    monkeypatch.setattr(sessmod, "_to_audio_np", lambda audio: np.zeros(160, dtype=np.float32))
    monkeypatch.setattr(sessmod, "_resolve_aligner", lambda rt, fa: None)
    monkeypatch.setattr(sessmod, "_resolve_draft_model", lambda **kwargs: "DRAFT")

    def fake_transcribe_loaded_components(**kwargs):  # noqa: ANN003
        calls.update(kwargs)
        return TranscriptionResult(text="ok", language="English")

    monkeypatch.setattr(sessmod, "_transcribe_loaded_components", fake_transcribe_loaded_components)

    session = sessmod.Session("repo/a", dtype=mx.float16)
    out = session.transcribe(
        np.zeros(160, dtype=np.float32),
        draft_model="repo/draft",
        num_draft_tokens=7,
    )

    assert out.text == "ok"
    assert calls["draft_model_obj"] == "DRAFT"
    assert calls["num_draft_tokens"] == 7


def test_session_transcribe_forwards_context(monkeypatch):
    class _DummyTokenizer:
        def __init__(self, path):  # noqa: ANN001
            self.path = path

    class _DummyModel:
        pass

    calls = {}

    monkeypatch.setattr(sessmod, "Tokenizer", _DummyTokenizer)
    monkeypatch.setattr(sessmod, "load_model", lambda model, dtype: (_DummyModel(), object()))
    monkeypatch.setattr(sessmod, "_resolve_path", lambda model: "/tmp/resolved-model")
    monkeypatch.setattr(sessmod, "_to_audio_np", lambda audio: np.zeros(160, dtype=np.float32))
    monkeypatch.setattr(sessmod, "_resolve_aligner", lambda rt, fa: None)
    monkeypatch.setattr(sessmod, "_resolve_draft_model", lambda **kwargs: None)

    def fake_transcribe_loaded_components(**kwargs):  # noqa: ANN003
        calls.update(kwargs)
        return TranscriptionResult(text="ok", language="English")

    monkeypatch.setattr(sessmod, "_transcribe_loaded_components", fake_transcribe_loaded_components)

    session = sessmod.Session("repo/a", dtype=mx.float16)
    out = session.transcribe(
        np.zeros(160, dtype=np.float32),
        context="交易 停滞",
    )

    assert out.text == "ok"
    assert calls["context"] == "交易 停滞"


def test_session_with_preloaded_model_requires_tokenizer_metadata():
    class _DummyModel:
        pass

    with np.testing.assert_raises(ValueError):
        sessmod.Session(_DummyModel())


def test_session_with_preloaded_model_uses_embedded_tokenizer_metadata(monkeypatch):
    created = {}

    class _DummyTokenizer:
        def __init__(self, path):  # noqa: ANN001
            created["tokenizer_path"] = path

    class _DummyModel:
        _resolved_model_path = "/tmp/preloaded-tokenizer-model"
        _source_model_id = "repo/preloaded"

    monkeypatch.setattr(sessmod, "Tokenizer", _DummyTokenizer)
    session = sessmod.Session(_DummyModel())
    assert session.model_id == "repo/preloaded"
    assert created["tokenizer_path"] == "/tmp/preloaded-tokenizer-model"


def test_session_init_streaming_forwards_session_defaults(monkeypatch):
    class _DummyTokenizer:
        def __init__(self, path):  # noqa: ANN001
            self.path = path

    class _DummyModel:
        pass

    calls = {}

    monkeypatch.setattr(sessmod, "Tokenizer", _DummyTokenizer)
    monkeypatch.setattr(sessmod, "load_model", lambda model, dtype: (_DummyModel(), object()))
    monkeypatch.setattr(sessmod, "_resolve_path", lambda model: "/tmp/resolved-model")

    def fake_streaming_init(**kwargs):  # noqa: ANN003
        calls.update(kwargs)
        return object()

    monkeypatch.setattr(sessmod.streaming_mod, "init_streaming", fake_streaming_init)

    session = sessmod.Session("repo/a", dtype=mx.float32)
    state = session.init_streaming(
        chunk_size_sec=1.5,
        max_context_sec=10.0,
        max_new_tokens=77,
        finalization_mode="latency",
        enable_tail_refine=None,
    )

    assert state is not None
    assert calls["model"] == "repo/a"
    assert calls["dtype"] == mx.float32
    assert calls["chunk_size_sec"] == 1.5
    assert calls["max_context_sec"] == 10.0
    assert calls["max_new_tokens"] == 77
    assert calls["finalization_mode"] == "latency"
    assert calls["enable_tail_refine"] is None


def test_session_init_streaming_forwards_context(monkeypatch):
    class _DummyTokenizer:
        def __init__(self, path):  # noqa: ANN001
            self.path = path

    class _DummyModel:
        pass

    calls = {}

    monkeypatch.setattr(sessmod, "Tokenizer", _DummyTokenizer)
    monkeypatch.setattr(sessmod, "load_model", lambda model, dtype: (_DummyModel(), object()))
    monkeypatch.setattr(sessmod, "_resolve_path", lambda model: "/tmp/resolved-model")

    def fake_streaming_init(**kwargs):  # noqa: ANN003
        calls.update(kwargs)
        return object()

    monkeypatch.setattr(sessmod.streaming_mod, "init_streaming", fake_streaming_init)

    session = sessmod.Session("repo/a", dtype=mx.float16)
    state = session.init_streaming(context="earnings EBITDA")

    assert state is not None
    assert calls["context"] == "earnings EBITDA"


def test_session_streaming_methods_use_loaded_model(monkeypatch):
    class _DummyTokenizer:
        def __init__(self, path):  # noqa: ANN001
            self.path = path

    class _DummyModel:
        pass

    calls = {"feed": None, "finish": None}

    monkeypatch.setattr(sessmod, "Tokenizer", _DummyTokenizer)
    monkeypatch.setattr(sessmod, "load_model", lambda model, dtype: (_DummyModel(), object()))
    monkeypatch.setattr(sessmod, "_resolve_path", lambda model: "/tmp/resolved-model")
    monkeypatch.setattr(sessmod.streaming_mod, "init_streaming", lambda **kwargs: object())

    def fake_feed_audio(pcm, state, model):  # noqa: ANN001
        calls["feed"] = {"pcm_len": len(pcm), "state": state, "model": model}
        return "FEED_OUT"

    def fake_finish_streaming(state, model):  # noqa: ANN001
        calls["finish"] = {"state": state, "model": model}
        return "FINISH_OUT"

    monkeypatch.setattr(sessmod.streaming_mod, "feed_audio", fake_feed_audio)
    monkeypatch.setattr(sessmod.streaming_mod, "finish_streaming", fake_finish_streaming)

    session = sessmod.Session("repo/a", dtype=mx.float16)
    state = object()
    out_feed = session.feed_audio(np.zeros(12, dtype=np.float32), state)
    out_finish = session.finish_streaming(state)

    assert out_feed == "FEED_OUT"
    assert out_finish == "FINISH_OUT"
    assert calls["feed"]["pcm_len"] == 12
    assert calls["feed"]["state"] is state
    assert calls["feed"]["model"] is session.model
    assert calls["finish"]["state"] is state
    assert calls["finish"]["model"] is session.model


def test_session_model_info_exposes_runtime_metadata(monkeypatch):
    class _DummyTokenizer:
        def __init__(self, path):  # noqa: ANN001
            self.path = path

    class _DummyModel:
        _resolved_model_path = "/tmp/resolved-model"

        class config:  # noqa: D106
            support_languages = ["en", "ja"]

            class text_config:  # noqa: D106
                vocab_size = 151936

    monkeypatch.setattr(sessmod, "Tokenizer", _DummyTokenizer)
    monkeypatch.setattr(sessmod, "load_model", lambda model, dtype: (_DummyModel(), object()))
    monkeypatch.setattr(sessmod, "_resolve_path", lambda model: "/tmp/resolved-model")

    session = sessmod.Session("repo/a", dtype=mx.float16)
    info = session.model_info
    assert info["model_id"] == "repo/a"
    assert info["resolved_model_path"] == "/tmp/resolved-model"
    assert info["dtype"] == str(mx.float16)
    assert info["vocab_size"] == 151936
    assert info["support_languages"] == ["en", "ja"]


def test_session_transcribe_async_wrapper(monkeypatch):
    class _DummyTokenizer:
        def __init__(self, path):  # noqa: ANN001
            self.path = path

    class _DummyModel:
        pass

    monkeypatch.setattr(sessmod, "Tokenizer", _DummyTokenizer)
    monkeypatch.setattr(sessmod, "load_model", lambda model, dtype: (_DummyModel(), object()))
    monkeypatch.setattr(sessmod, "_resolve_path", lambda model: "/tmp/resolved-model")

    session = sessmod.Session("repo/a", dtype=mx.float16)
    monkeypatch.setattr(
        session,
        "transcribe",
        lambda *a, **k: sessmod.TranscriptionResult(text="ok-async", language="English"),
    )

    out = asyncio.run(session.transcribe_async(np.zeros(10, dtype=np.float32)))
    assert out.text == "ok-async"


def test_session_transcribe_async_runs_on_the_given_executor(monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    class _DummyTokenizer:
        def __init__(self, path):  # noqa: ANN001
            self.path = path

    monkeypatch.setattr(sessmod, "Tokenizer", _DummyTokenizer)
    monkeypatch.setattr(sessmod, "load_model", lambda model, dtype: (object(), object()))
    monkeypatch.setattr(sessmod, "_resolve_path", lambda model: "/tmp/resolved-model")
    session = sessmod.Session("repo/a", dtype=mx.float16)
    seen: list[str] = []

    def _record(*a, **k):
        seen.append(threading.current_thread().name)
        return sessmod.TranscriptionResult(text="ok", language="English")

    monkeypatch.setattr(session, "transcribe", _record)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="owner") as pool:
        asyncio.run(session.transcribe_async(np.zeros(10, dtype=np.float32), executor=pool))
    assert seen and seen[0].startswith("owner")


def _stub_session(monkeypatch, calls):
    """Session with every heavy dependency replaced, capturing the diarization config."""

    class _DummyTokenizer:
        def __init__(self, path):  # noqa: ANN001
            self.path = path

    class _DummyModel:
        pass

    monkeypatch.setattr(sessmod, "Tokenizer", _DummyTokenizer)
    monkeypatch.setattr(sessmod, "load_model", lambda model, dtype: (_DummyModel(), object()))
    monkeypatch.setattr(sessmod, "_resolve_path", lambda model: "/tmp/resolved-model")
    monkeypatch.setattr(sessmod, "_to_audio_np", lambda audio: np.zeros(160, dtype=np.float32))
    monkeypatch.setattr(sessmod, "_resolve_aligner", lambda rt, fa: "ALIGNER")
    monkeypatch.setattr(sessmod, "_resolve_draft_model", lambda **kwargs: None)

    real_resolve = sessmod._resolve_diarization_config

    def spy_resolve(**kwargs):  # noqa: ANN003
        calls.update(kwargs)
        return real_resolve(**kwargs)

    monkeypatch.setattr(sessmod, "_resolve_diarization_config", spy_resolve)
    monkeypatch.setattr(
        sessmod,
        "_transcribe_loaded_components",
        lambda **kwargs: TranscriptionResult(text="ok", language="English"),
    )
    return sessmod.Session("repo/a", dtype=mx.float32)


def test_session_transcribe_forwards_the_requested_diarization_device(monkeypatch):
    """Accepting the option is not the same as honouring it.

    `Session.transcribe` built the options with the device but then called
    `_resolve_diarization_config` without it, so an explicit device was silently
    replaced by `auto` - the caller got the accelerator they asked not to use.
    """
    calls: dict = {}
    session = _stub_session(monkeypatch, calls)

    session.transcribe(
        np.zeros(160, dtype=np.float32),
        diarize=True,
        diarization_device="cpu",
    )

    assert calls["diarization_device"] == "cpu"


def test_session_transcribe_async_forwards_the_requested_diarization_device(monkeypatch):
    calls: dict = {}
    session = _stub_session(monkeypatch, calls)

    asyncio.run(
        session.transcribe_async(
            np.zeros(160, dtype=np.float32),
            diarize=True,
            diarization_device="cpu",
        )
    )

    assert calls["diarization_device"] == "cpu"
