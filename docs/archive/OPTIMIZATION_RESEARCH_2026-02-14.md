# Qwen3-ASR Deep Research and MLX Optimization Plan (2026-02-14)

## Status Update (2026-02-14)

This document includes early-session research notes and should be read with
the current gate docs:

- `docs/QUALITY_GATE.md`
- `docs/GOLDEN_DATASET.md`
- `docs/BENCHMARKING.md`

Important: this file is a time-capsule of exploratory notes from a single
session. Several items below were hypotheses at the time and may now be
resolved. Treat `docs/RESEARCH.md` + `docs/ROADMAP.md` as current truth.

Resolved-status notes (to avoid stale interpretation):
- Earlier MRoPE mismatch hypothesis is resolved in the current codebase and
  covered by parity gates/tests.
- Earlier "forced aligner path is WIP" note is partially outdated:
  - native MLX aligner backend now exists as experimental,
  - native MLX is now the default backend; `qwen_asr` remains available as
    the official-reference option.
- Quantization quality statements based on small-sample runs in this file have
  been superseded by speaker-balanced `n=100` artifacts in `docs/benchmarks/`.

Implemented optimization finding on 2026-02-14:

- Tokenizer caching across repeated `transcribe()` calls reduced measured
  latency by about 68% on local M4 Pro benchmarks for
  `Qwen/Qwen3-ASR-0.6B` + `tests/fixtures/test_speech.wav`.
- Decoder-path improvements:
  - preallocated KV cache writes (in-place assignment) to remove repeated concat growth,
  - direct grouped-query fused attention without explicit K/V head repetition.
- Current measured operating point:
  - short fixture: ~`0.53s` mean latency (`RTF ~0.21`)
  - 10-second clip: ~`0.94-0.95s` mean latency (`RTF ~0.095`)
- Quantized operating point (4-bit, validated load path):
  - short fixture: ~`0.159s` mean latency (`RTF ~0.063`)
  - 10-second clip: ~`0.283s` mean latency (`RTF ~0.028`)
  - LibriSpeech test-clean sample (20 utterances): same WER as fp16 in local run.

## Scope

This note focuses on how to make `mlx-qwen3-asr` both:

1. **Correct** against the official Qwen3-ASR implementation.
2. **Fast** in the style of `faster-whisper` (and ideally better on Apple Silicon with MLX).

All findings are based on official Qwen sources, official model configs, and official performance docs.

## Verified Upstream Facts (Official Sources)

- Qwen3-ASR and Qwen3-ForcedAligner were released on **2026-01-29**.
- Qwen3-ASR supports **52 languages/dialects** (30 languages + 22 Chinese dialects).
- Official toolkit supports:
  - offline and streaming ASR
  - forced alignment timestamps (11 languages)
  - vLLM backend for highest throughput
- Official technical report claims for 0.6B:
  - TTFT as low as 92ms
  - up to 2000 seconds of audio/s at concurrency 128 (vLLM environment)

## Critical Correctness Findings for This Repo

Historical note: this section records findings captured during the exploratory
session window; some items here were later resolved. See the resolved-status
notes at the top and `docs/RESEARCH.md` for current-state truth.

### 1) MRoPE behavior mismatch with official code

Official `apply_interleaved_mrope` keeps temporal frequencies as the base and overwrites only selected indices for H/W dimensions. It does **not** zero out "uncovered" indices.

Current local `mlx_qwen3_asr/mrope.py` assumes uncovered indices remain zeros and encodes that into both implementation and tests.

Observed numerical mismatch from a direct comparison script:

- differing cosine indices: `61, 62, 125, 126`
- local cos: `0`
- official-style cos: `1`

Impact:

- This is a direct positional embedding discrepancy in text attention.
- It can reduce transcription quality even when everything else looks healthy.

### 2) Public docs still have stale architecture numbers

`README.md` model-variant table currently lists old dimensions/layer counts for 1.7B.
Official HF `config.json` for `Qwen/Qwen3-ASR-1.7B` currently reports:

- audio: 24 layers, 16 heads, `d_model=1024`, `output_dim=2048`, `n_window=50`, `n_window_infer=800`
- text: 28 layers, 16 heads, 8 KV heads, `hidden=2048`, `intermediate=6144`, `rope_theta=1e6`

This should be updated to keep docs aligned with official model artifacts.

### 3) Official inference behavior not fully mirrored yet

Official inference includes:

- stronger repetition cleanup in post-processing (`detect_and_fix_repetitions`)
- forced aligner path (currently intentionally guarded as WIP in this repo)
- streaming and long-audio behavior tuned around official prompt/output conventions

## Performance Research: What "fast-whisper style" means in practice

From `faster-whisper` and CTranslate2 docs, the recurring speed principles are:

1. **Batch aggressively** (especially on GPU backends).
2. **Quantize aggressively** (int8/int4 where quality allows).
3. **Use async pipelines** (decode/preprocess in parallel with model execution).
4. **Keep shapes stable** to maximize kernel reuse and reduce overhead.

For MLX specifically, official MLX docs add:

1. Use `mx.compile` on steady-state hot paths.
2. Avoid frequent eval barriers; evaluate at natural boundaries.
3. Use streams and unified memory to overlap CPU and GPU work.
4. Bucket input shapes to avoid recompilation churn.

## Optimization Roadmap (Prioritized)

## Phase 0: Correctness Parity First (must-do before speed work)

1. Fix MRoPE to match official `apply_interleaved_mrope`.
2. Replace/extend MRoPE tests with a golden test against official behavior.
3. Update README architecture table to match current official configs.
4. Add parity tests for parse/output behavior against official formatting.

Acceptance:

- MRoPE golden tests pass.
- No known architecture mismatch in docs.

## Phase 1: Low-risk speedups with minimal architecture change

1. Cache `WhisperFeatureExtractor` instance (avoid per-chunk re-instantiation).
2. Add shape bucketing for mel frame lengths and prompt lengths.
3. Compile hot decode step (`mx.compile`) on stable bucketed shapes.
4. Reduce avoidable `mx.eval` calls in decode loop.

Expected:

- meaningful TTFT and per-chunk latency reduction with low risk.

## Phase 2: Throughput architecture (fast-whisper-like)

1. Add **batched transcription path**:
   - sort by audio length
   - batch by token budget
   - restore original order after decode
2. Add request-level worker queue with bounded concurrency.
3. Overlap pipeline stages:
   - CPU: audio decode + feature prep
   - GPU: encoder/decoder

Expected:

- significant throughput gains for multi-file and server workloads.

## Phase 3: Quantization and kernel-level acceleration

1. Mixed-precision strategy:
   - keep sensitive paths in fp16/bf16
   - quantize linear/embedding-heavy decoder components
2. Evaluate 4-bit/8-bit quality-speed tradeoffs per model size (0.6B, 1.7B).
3. Investigate MLX custom/fused kernels for repeated decoder ops if needed.

Expected:

- biggest memory and throughput gains, with quality tuning required.

## Phase 4: Feature parity that also improves product value

1. Implement official-style forced aligner path.
2. Implement official streaming mode semantics and knobs:
   - chunk size (2s baseline)
   - unfixed chunk/token rollback
3. Add optional VAD-guided long-audio chunking mode.

## Benchmarking Protocol (for disciplined optimization)

Track these metrics per model size (0.6B and 1.7B), on the same machine:

- TTFT
- real-time factor (RTF)
- audio seconds processed / wall-clock second
- peak memory
- quality (WER/CER on fixed eval sets)

Test scenarios:

1. single short utterance (latency)
2. 10-minute file (long-form stability)
3. N-file batch (throughput)
4. streaming 2-second chunks (partial-output stability)

## Suggested Immediate Next Sprint (highest ROI)

1. Fix MRoPE to official behavior and rewrite its tests.
2. Update README model-variant table to official config values.
3. Cache feature extractor + add decode shape bucketing + compile hot decode path.
4. Add benchmark harness and lock in baseline numbers before larger refactors.

## Primary Sources

- Qwen3-ASR official repo: https://github.com/QwenLM/Qwen3-ASR
- Official MRoPE implementation (`apply_interleaved_mrope`):  
  https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py
- Qwen3-ASR technical report (arXiv 2601.21337): https://arxiv.org/abs/2601.21337
- HF model configs:
  - https://huggingface.co/Qwen/Qwen3-ASR-1.7B/raw/main/config.json
  - https://huggingface.co/Qwen/Qwen3-ASR-0.6B/raw/main/config.json
  - https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B/raw/main/config.json
- faster-whisper official repo: https://github.com/SYSTRAN/faster-whisper
- CTranslate2 performance docs: https://opennmt.net/CTranslate2/performance.html
- CTranslate2 quantization docs: https://opennmt.net/CTranslate2/quantization.html
- MLX compile docs: https://ml-explore.github.io/mlx/build/html/usage/compile.html
- MLX lazy eval docs: https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html
- MLX streams docs: https://ml-explore.github.io/mlx/build/html/usage/using_streams.html
