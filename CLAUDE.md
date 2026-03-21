# CLAUDE.md

Agent instructions for `mlx-qwen3-asr`.

## North Star

**`pip install mlx-qwen3-asr` — the definitive speech recognition package for Apple Silicon.**

Ground-up MLX reimplementation of Qwen3-ASR. Same HuggingFace weights, same output quality, every layer rewritten for Metal. Not a wrapper, not a binding — a standalone package.

- **One-command setup** — no ffmpeg install, no model conversion, no CUDA
- **Both models validated** — 0.6B (fast, default) and 1.7B (accuracy), benchmarked across 10 languages
- **Native forced aligner** — word-level timestamps via MLX, no PyTorch dependency
- **Zero transformers dependency** — custom mel spectrogram, native BPE tokenizer, no HF transformers at runtime
- **52 languages** — validated, not just English
- **Fast** — Metal-optimized inference, sub-200ms latency on short clips
- **Production-grade** — bounds checks, multi-slot model cache, Session API, proper error paths
- **No PyTorch in the core path** — core ASR pipeline (audio → mel → encoder → decoder → text) must never import torch
- **Built-in server** — `mlx-qwen3-asr serve` turns any Mac into a transcription API endpoint (optional `[serve]` extra)

### What this is NOT

- Not a multi-model toolkit (that's mlx-audio)
- Not a training framework

## Current Status (v0.1.0)

Published on PyPI. 441 tests (440 passed, 1 skipped). Core pipeline stable.

### What's next

1. **Transcription server** — built-in HTTP server (`mlx-qwen3-asr serve`) with async jobs, API key auth, backpressure. See `docs/server/` for spec.
2. **Swift port** — once Python proves every decision, native Swift+MLX for apps and system integration (separate repo: `qwen3-asr-swift`)

### What's done

- PyPI v0.1.0 published, 1.7B validated, both models benchmarked across 10 languages
- **Native BPE tokenizer** — `transformers` fully removed from dependencies; loads vocab.json/merges.txt directly
- **KV-cache streaming** — linear-complexity incremental decoding with context trimming, tail refinement, CJK-aware stable/unstable splitting
- Custom mel spectrogram (default path; no HF dependency)
- Native MLX forced aligner with LIS timestamp correction (no PyTorch dependency)
- `model.prefill()` / `model.step()` / `model.step_many()` clean interface
- `Session` object + multi-slot `_ModelHolder` cache keyed by `(path, dtype)`
- Audio stays numpy through feature extraction, converts to `mx.array` at model entry
- Bounds checks on audio injection, narrowed exception handling, language validation with warnings
- Streaming quality metrics: `partial_stability`, `rewrite_rate`, `finalization_delta_chars`
- 138 benchmark artifacts across 13 eval scripts, MLX-vs-PyTorch head-to-head (MLX wins: 15.99% vs 16.69% WER)
- Typed-core mypy gate, release quality gate (ruff + pytest + reference parity + LibriSpeech + manifest quality + benchmarks)

## Architecture

Qwen3-ASR is an encoder-decoder model:
1. Audio → mel spectrogram (128 bins) → Conv2d stem (8x downsample) → 24 transformer encoder layers → audio features
2. Audio features injected into text embedding sequence at `<|audio_pad|>` placeholder positions
3. Text decoder (28 Qwen3-style layers with interleaved MRoPE) autoregressively generates transcription
4. Optional: Forced aligner (separate 0.6B model) provides word-level timestamps

### Correctness invariants

These are non-negotiable. Violating any one produces silently wrong output:

1. **MRoPE is interleaved, not chunked** — sections [24,20,20] with stride-3 frequency assignment across temporal/height/width dimensions
2. **Encoder uses LayerNorm + bias** — text decoder uses RMSNorm + no bias. Mixing them breaks inference silently
3. **Conv2d weight transpose** — PyTorch `(out,in,kH,kW)` → MLX `(out,kH,kW,in)` via `transpose(0,2,3,1)`
4. **Sinusoidal position embeddings are computed, not loaded** — they're not in the weight files
5. **Audio token IDs** — pad=151676, start=151669, end=151670

### Module map

```
mlx_qwen3_asr/
├── __init__.py          # Public API: transcribe(), load_model(), Session
├── __main__.py          # python -m mlx_qwen3_asr
├── _version.py          # Single source of truth for version
├── config.py            # Dataclass configs (no MLX imports)
├── audio.py             # Audio I/O (load_audio_np) + custom mel spectrogram (MLX)
├── mrope.py             # Interleaved Multi-RoPE (critical correctness)
├── attention.py         # Shared SDPA helper (fused + fallback)
├── encoder.py           # Audio encoder: SinusoidalPE, windowed attention, Conv2d stem
├── decoder.py           # Text decoder: GQA, SwiGLU, KVCache, causal mask
├── model.py             # Qwen3ASRModel: glue, audio injection, prefill/step/step_many
├── convert.py           # Weight key remapping + Conv2d transpose
├── load_models.py       # HuggingFace download + multi-slot model cache
├── generate.py          # Autoregressive decode loop
├── transcribe.py        # High-level pipeline (main entry point)
├── tokenizer.py         # Native BPE tokenizer (vocab.json/merges.txt) + language validation
├── session.py           # Session: explicit model/tokenizer lifecycle
├── chunking.py          # Energy-based long audio splitting
├── forced_aligner.py    # MLX forced aligner + LIS timestamp correction
├── streaming.py         # KV-cache streaming with context trimming + tail refinement
├── writers.py           # Output format writers (txt, srt, vtt, json, tsv)
├── server.py            # Optional HTTP server (FastAPI, async jobs, API key auth)
├── cli.py               # CLI entry point (transcribe + serve subcommands)
└── assets/
    ├── mel_filters.npz        # Pre-computed 128-bin Slaney mel filterbanks
    └── korean_dict_jieba.dict # Korean tokenizer dictionary for aligner
```

### Key technical decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Standalone vs mlx-audio | Standalone | mlx-audio lacks MRoPE, too many deps |
| Tokenizer | Native BPE (vocab.json/merges.txt) | Zero external tokenizer deps |
| Audio loading | ffmpeg subprocess + fast WAV path | Handles all formats, no heavy deps |
| Weight format | safetensors | MLX ecosystem standard |
| Mel spectrogram | Custom MLX (default) | HF fallback for non-16kHz only |
| Forced aligner | Native MLX (default) | PyTorch `qwen-asr` as optional fallback |
| Server | FastAPI + uvicorn (optional `[serve]`) | Async, lightweight, OpenAPI docs free |

## Code Conventions

- **Python 3.10+** — f-strings, type hints, dataclasses throughout
- **`mlx.nn.Module`** for all model components (never torch)
- **Type hints** on all public functions; **Google-style docstrings** on public classes/functions
- **`@dataclass(frozen=True)`** for output types (e.g., `TranscriptionResult`, `AlignedWord`)
- **No classes where functions suffice** — prefer flat module-level functions
- **Naming**: modules `snake_case.py`, classes `PascalCase`, functions `snake_case`, constants `UPPER_SNAKE_CASE`, private `_prefixed`

## Git Commit Messages

Every commit is a self-contained unit of knowledge. The message is not just a changelog entry — it's the **DNA of the change**: the intent, the reasoning, and a seed prompt that encapsulates everything needed to reproduce the work from scratch.

A well-written commit message is **hydrated with the most pertinent context from every angle** — the problem that triggered it, the architectural constraints that shaped it, the alternatives that were rejected, and the precise technical details that make it work. It should carry all the traces: enough signal that any future agent can reconstruct not just *what* changed, but *why* it had to change, *what else was considered*, and *how* to arrive at the same solution independently.

The goal is **maximum information density per line**. Every sentence should earn its place. No filler, no boilerplate — just distilled, essential context that compounds in value over time as the git history becomes the project's institutional memory.

### Structure

```
<type>: <concise summary> (imperative mood)

## What Changed
Detailed breakdown of every file/section modified.

## Root Intent
Why this commit exists — the underlying problem or goal,
not the mechanical description of what was done. Include
the failure mode or gap that motivated the work.

## Seed Prompt
The generative kernel of this change. A dense, high-context
instruction that carries enough architectural awareness and
domain knowledge to reproduce an equivalent diff from first
principles. Not a step-by-step script — a briefing that
transmits the essential understanding.

Write it as if encoding the change's DNA into a single
message: what to do, why it matters, what constraints to
respect, and what pitfalls to avoid. A skilled engineer
with codebase access but zero prior context should be able
to produce an equivalent diff from this alone.

## Files Changed
Summary with line counts for quick scanning.

Co-Authored-By: Claude <noreply@anthropic.com>
```

### Types

`feat` | `fix` | `refactor` | `docs` | `test` | `chore`

### When to use full format vs simplified

- **Full format**: multi-file changes, architectural decisions, new modules, non-obvious fixes
- **Simplified** (type + summary + brief why): typos, single-line fixes, dependency bumps

## Development Environment

```bash
# Setup — existing venv at .venv (Python 3.11), no pip installed — use uv
uv pip install -e ".[dev]"

# Tests (441 total; 440 passed, 1 skipped)
python -m pytest tests/              # full suite
python -m pytest tests/ -x -q        # stop on first failure

# Lint
ruff check mlx_qwen3_asr/

# Quality gate (run before PRs)
python scripts/quality_gate.py --mode fast

# Release gate (includes reference parity — requires qwen-asr)
RUN_REFERENCE_PARITY=1 python scripts/quality_gate.py --mode release

# Quick smoke test
python -c "import mlx_qwen3_asr; print(mlx_qwen3_asr.transcribe('tests/fixtures/test_audio.wav'))"

# CLI
mlx-qwen3-asr tests/fixtures/test_audio.wav --verbose
```

## Publishing to PyPI

1. Bump version in `mlx_qwen3_asr/_version.py`
2. Run quality gate: `python scripts/quality_gate.py --mode fast`
3. Build: `uv build`
4. Verify wheel: confirm 22 modules + `py.typed` + both assets
5. Upload: `TWINE_USERNAME=__token__ TWINE_PASSWORD=<token> uv tool run twine upload dist/*`
6. Verify: `pip install mlx-qwen3-asr==VERSION && mlx-qwen3-asr --help`
7. Tag: `git tag v<VERSION> && git push origin v<VERSION>`

PyPI account: moona3k@gmail.com. Token scope: `mlx-qwen3-asr` project. Pure Python wheel (`py3-none-any`). License: `Apache-2.0` (string format in pyproject.toml, not table).

## Documentation

| File | Purpose |
|------|---------|
| `docs/ARCHITECTURE.md` | Qwen3-ASR architecture deep dive |
| `docs/DECISIONS.md` | Key decisions and rationale |
| `docs/QUALITY_GATE.md` | Merge/release quality gates |
| `docs/BENCHMARKS.md` | Measured results and methodology |
| `docs/GOLDEN_DATASET.md` | Golden dataset policy and evaluation commands |
| `docs/COMPARISON.md` | Comparison with alternatives (mlx-audio, whisper, etc.) |
| `docs/RESEARCH.md` | Research findings, model analysis |
| `docs/BENCHMARKING.md` | Runtime measurement protocol and methodology |
| `docs/memory/operating-memory.md` | Agent memory front door (protocol + compacted guidance) |
| `docs/memory/events/` | Append-only implementation memory events |
| `docs/server/README.md` | Server feature overview and quick start |
| `docs/server/API-SPEC.md` | Full server API specification |
| `docs/server/ADR-001-transcription-server.md` | Server architecture decision record |
| `docs/server/DEPLOYMENT.md` | Server deployment and operations guide |

## Continuous Learning

Start memory workflow at `docs/memory/operating-memory.md` (single front door).

Guidance:

1. For non-trivial and meaningful work, append an event in
   `docs/memory/events/YYYY-MM.md`.
2. Minimum bar for event entries: include `Decision`, `Reuse next time`, and
   `Evidence`.
3. Update `docs/memory/operating-memory.md` only when active guidance changes.
4. Include failed paths/root cause when the miss is meaningful.
5. If a pattern repeats, promote it to `Distilled Learnings`.

Optional fields when useful: `Scope`, `What worked`, `What did not work`,
`Risk left`, `Revisit trigger`, `ROI`.

Cross-session learnings still go in `MEMORY.md` (auto-loaded). Decision
rationale belongs in commit messages and `docs/DECISIONS.md`.

**Before touching model code**: read `docs/ARCHITECTURE.md` and check the official Qwen3-ASR repo for reference implementation.

**After hitting a dead end**: identify root cause, note what didn't work, document if a future agent would hit the same wall.

## References

- Official Qwen3-ASR: https://github.com/QwenLM/Qwen3-ASR
- Paper: https://arxiv.org/abs/2601.21337
- mlx-whisper (pattern): https://github.com/ml-explore/mlx-examples/tree/main/whisper
- Swift port (third-party): https://github.com/ivan-digital/qwen3-asr-swift
