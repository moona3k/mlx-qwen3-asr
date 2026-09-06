# CLAUDE.md

Agent instructions for `mlx-qwen3-asr`. `AGENTS.md` points here.

## North Star

**`pip install mlx-qwen3-asr` is the definitive way to run Qwen3-ASR on Apple Silicon.**

Ground-up MLX reimplementation of Qwen3-ASR. Same HuggingFace weights, same
output quality, every layer rewritten for Metal. Not a wrapper, not a binding.

- One-command setup: no model conversion, no CUDA, no PyTorch, no transformers
- Both models validated: 0.6B (fast, default) and 1.7B (accuracy), benchmarked across 10 languages
- Native forced aligner for word timestamps; optional pyannote diarization
- Long audio, streaming, speculative decoding, 4/8-bit quantization
- Built-in HTTP server with an OpenAI-compatible endpoint (`[serve]` extra)
- Production posture: bounds checks, model cache, Session API, typed core, benchmark-gated changes

Not a multi-model toolkit (that is mlx-audio) and not a training framework.

## Status (v0.3.5, September 2026)

Published on PyPI and in use by third parties (external issues and PRs land
regularly). Full suite is 660+ tests and runs in under 10 s. Real-model
regressions are caught by the nightly LibriSpeech lane.

Next (see `docs/ROADMAP.md`): aligner quality hardening, publishing quantized
artifacts, long-form robustness benchmarks, streaming-manifest artifacts.

## Architecture

Qwen3-ASR is an encoder-decoder model:

1. Audio -> mel spectrogram (128 bins) -> Conv2d stem (8x downsample) -> windowed transformer encoder
2. Audio features are injected into the text embedding sequence at `<|audio_pad|>` positions
3. Text decoder (Qwen3-style, interleaved MRoPE, GQA) generates `language X<asr_text>...`
4. Optional: a separate 0.6B forced-aligner model yields word timestamps

Read `docs/ARCHITECTURE.md` before touching model code and check the official
Qwen3-ASR repo for the reference implementation.

### Correctness invariants

Violating any of these produces silently wrong output:

1. MRoPE is interleaved, not chunked: sections [24,20,20] with stride-3 frequency assignment
2. Encoder uses LayerNorm + bias; decoder uses RMSNorm + no bias
3. Conv2d weights transpose from PyTorch `(out,in,kH,kW)` to MLX `(out,kH,kW,in)`
4. Sinusoidal position embeddings are computed, not loaded
5. Audio token IDs: pad=151676, start=151669, end=151670
6. The core path (audio -> mel -> encoder -> decoder -> text) never imports torch; torch is reachable only behind the `[diarize]` extra

### MLX runtime rules

- Every array built in an `__init__` and stored outside `parameters()` must be
  materialized with `mx.eval` (see `SinusoidalPositionEmbedding`,
  `InterleavedMRoPE`). MLX 0.31+ cannot evaluate a lazy graph from another
  thread; a model loaded on one thread must run on any thread.
- Non-parameter buffers must be cast to the activation dtype at the use site;
  `load_model()`'s dtype cast only reaches `parameters()`.
- Module-level array caches use `cache_utils.LRUCache`, never a bare dict.
- Release per-chunk tensors and call `mx.clear_cache()` between chunks of long audio.

### Module map

```
mlx_qwen3_asr/
├── __init__.py        Public API: transcribe*, Session, load_model, load_audio, ForcedAligner
├── transcribe.py      Pipeline: TranscribeOptions, chunk loop, timestamps, diarization glue
├── session.py         Session: explicit model/tokenizer ownership, async with executor
├── streaming.py       KV-cache streaming, context trimming, tail refinement
├── cli.py             CLI: transcribe (default), serve, --mic, --doctor
├── server.py          FastAPI server: async jobs, OpenAI-compatible endpoint, limits
├── audio.py           Audio I/O, native WAV fast path, MLX mel spectrogram
├── chunking.py        Energy-based splitting into 30 s chunks
├── encoder.py         Audio encoder (Conv2d stem, sinusoidal PE, windowed attention)
├── decoder.py         Text decoder (GQA, SwiGLU, KVCache, bounded mask caches)
├── mrope.py           Interleaved MRoPE (critical correctness)
├── attention.py       Shared SDPA helper (fused + fallback)
├── model.py           Qwen3ASRModel: audio injection, prefill/step/step_many
├── generate.py        Greedy and speculative decoding, repetition stop, token budgets
├── forced_aligner.py  Native MLX forced aligner + LIS timestamp correction
├── diarization.py     Optional pyannote integration, device selection, speaker segments
├── tokenizer.py       Native BPE tokenizer, language aliases, join_text_parts, output parsing
├── load_models.py     HF download, weight loading, quantization, _ModelHolder cache
├── convert.py         Weight key remapping + Conv2d transpose
├── writers.py         txt/json/srt/vtt/tsv writers, subtitle cue grouping
├── cache_utils.py     LRUCache
├── config.py          Dataclass configs (no MLX imports)
└── assets/            mel_filters.npz, korean_dict_jieba.dict
```

### Key technical decisions

Full rationale lives in `docs/DECISIONS.md`.

| Decision | Choice |
|----------|--------|
| Standalone package, not mlx-audio | mlx-audio lacks MRoPE and pulls heavy deps |
| Tokenizer | Native BPE from vocab.json/merges.txt |
| Audio loading | Native WAV parser; ffmpeg subprocess for other formats |
| Mel spectrogram | Custom MLX implementation |
| Forced aligner | Native MLX only |
| Diarization | Optional pyannote behind `[diarize]`; `auto` device with CPU fallback |
| Server | FastAPI + uvicorn behind `[serve]`; one dedicated inference thread |
| Text joining | `tokenizer.join_text_parts`; Chinese/Japanese have no spaces, Korean does |

## Code conventions

- Python 3.10+, type hints on public functions, Google-style docstrings on public classes/functions
- `mlx.nn.Module` for model components; `@dataclass(frozen=True)` for outputs
- Flat module-level functions unless state genuinely needs a class
- One explicitly typed signature per behaviour (`transcribe`, `transcribe_batch`,
  `Session.transcribe`); wrappers forward `**kwargs`; internals pass `TranscribeOptions`
- Production code never branches on what a test double supports; give doubles the real signature
- Naming: modules `snake_case.py`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`, private `_prefixed`

## Development

```bash
uv pip install -e ".[dev,serve]"            # .venv is Python 3.11; use uv, not pip
python -m pytest tests/ -q                  # full suite, ~5 s
ruff check .
python scripts/quality_gate.py --mode fast  # ruff + mypy typed core + pytest; run before PRs
RUN_REFERENCE_PARITY=1 python scripts/quality_gate.py --mode release
python -c "import mlx_qwen3_asr as m; print(m.transcribe('tests/fixtures/test_speech.wav').text)"
mlx-qwen3-asr tests/fixtures/test_speech.wav --verbose
```

The repo venv pins an older MLX than users install. Before touching threading,
dtype handling or the audio front end, also run the suite in a venv with the
newest `mlx` release; CI (`macos-14`, latest mlx) is the real guard.

Real-model checks when a change can affect numerics: `scripts/eval_librispeech.py
--samples 100 --sampling speaker_round_robin` before and after, comparing WER and
the per-sample hypotheses in the JSON output.

## Git commit messages

Commits are the project's institutional memory. For multi-file or non-obvious
changes use the full format below; typos, one-line fixes and dependency bumps
use `type: summary` plus one sentence of why.

```
<type>: <imperative summary>

## What Changed
## Root Intent      (the failure mode or gap that motivated the change)
## Seed Prompt      (a briefing dense enough for another engineer to reproduce the diff)
## Files Changed

Co-Authored-By: Claude <noreply@anthropic.com>
```

Types: `feat` `fix` `perf` `refactor` `docs` `test` `chore`.

## Publishing to PyPI

1. Bump `mlx_qwen3_asr/_version.py`; run `python scripts/update_readme_stats.py`
2. `python scripts/quality_gate.py --mode fast`
3. `uv build`, then check the wheel contains every module in the map above, `py.typed` and both assets
4. `TWINE_USERNAME=__token__ TWINE_PASSWORD=<token> uv tool run twine upload dist/*`
5. `pip install mlx-qwen3-asr==VERSION && mlx-qwen3-asr --version`
6. `git tag v<VERSION> && git push origin v<VERSION>`

PyPI account moona3k@gmail.com, token scoped to the project. Pure Python wheel. License string `Apache-2.0`.

## Documentation

| File | Purpose |
|------|---------|
| `docs/ARCHITECTURE.md` | Model architecture and constants |
| `docs/DECISIONS.md` | Numbered decision records with rationale |
| `docs/ROADMAP.md` | Status by priority and next exploration queue |
| `docs/QUALITY_GATE.md` | Fast / release / strict gates; CI workflows |
| `docs/BENCHMARKS.md`, `docs/BENCHMARKING.md` | Measured results and protocol |
| `docs/GOLDEN_DATASET.md`, `docs/EVAL_GAPS.md` | Quality lanes and remaining gaps |
| `docs/COMPARISON.md`, `docs/RESEARCH.md`, `docs/MODEL_WATCH.md` | Alternatives, sources, model watchlist |
| `docs/server/` | Server README, API spec, ADR, deployment |
| `docs/memory/operating-memory.md` | Agent memory front door; events in `docs/memory/events/` |
| `docs/archive/` | Dated notes kept for provenance, not maintained |

`.github/workflows/` is the source of truth for CI: `ci.yml` (lint + fast
gate on PRs), `nightly-regression.yml`, `long-media-regression.yml`,
`reference-parity.yml`, `quantization-matrix.yml`, `publish-quantized.yml`,
`package-analytics.yml`.

## Continuous learning

Start at `docs/memory/operating-memory.md`. For non-trivial work append an
event to `docs/memory/events/YYYY-MM.md` with `Decision`, `Reuse next time`
and `Evidence`; include the failed paths when the miss was meaningful. Update
`operating-memory.md` only when active guidance changes. Decision rationale
belongs in commit messages and `docs/DECISIONS.md`.

## References

- Official Qwen3-ASR: https://github.com/QwenLM/Qwen3-ASR
- Paper: https://arxiv.org/abs/2601.21337
- mlx-whisper (pattern): https://github.com/ml-explore/mlx-examples/tree/main/whisper
- Swift port (third-party): https://github.com/ivan-digital/qwen3-asr-swift
