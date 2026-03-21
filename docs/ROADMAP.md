# Roadmap to North Star

Target: `pip install mlx-qwen3-asr` is the obvious/default way to run
Qwen3-ASR on Apple Silicon.

## Status by Priority

1. Correctness validation vs official PyTorch (token parity, greedy)
- In progress.
- Added optional integration test scaffold:
  - `tests/test_reference_parity.py`
  - Enabled via `RUN_REFERENCE_PARITY=1`.
- Added manual CI workflow: `.github/workflows/reference-parity.yml`.
- Added explicit gate runner: `scripts/quality_gate.py` with fast/release modes.
- Added policy doc: `docs/QUALITY_GATE.md`.
- Added upstream-style output parse hardening:
  - repetition cleanup in decoded text,
  - `language None` empty-audio handling,
  - forced-language parse path support.
- Added scheduled regression lane:
  - `.github/workflows/nightly-regression.yml`
  - `scripts/eval_librispeech.py`
  - `docs/GOLDEN_DATASET.md`

2. Long audio preprocessing (no 30s feature truncation)
- Done.
- `compute_features()` now uses `truncation=False` and defaults to
  `padding="do_not_pad"`.

3. Quantized model artifacts on HuggingFace (4-bit / 8-bit)
- In progress (runtime path now validated; artifact publishing pending).
- Code-level quantization utility exists (`mlx_qwen3_asr.convert.quantize_model`).
- Added publishing script: `scripts/publish_quantized.py`.
- Added manual CI workflow: `.github/workflows/publish-quantized.yml`.
- Fixed quantized runtime loading in `load_model()`:
  - detects quantized tensors,
  - quantizes model modules before loading weights,
  - supports optional `quantization_config.json` metadata.
- Latest validated benchmark point from refreshed matrix run (4-bit, 0.6B, Apple M4 Pro):
  - short fixture: mean `0.1187s`, RTF `0.0469`
  - 10s clip: mean `0.2286s`, RTF `0.0229`
- Quantization sweep complete (`fp16`, `4bit-g64`, `4bit-g32`, `8bit-g64`):
  - selected `4bit-g64` as current recommended default profile.
- Added reproducible experiment workflow:
  - `.github/workflows/quantization-matrix.yml` (manual CI matrix runs).

Performance progress:
- Done (low-risk optimization): tokenizer caching in `transcribe()` hot path.
- Measured local result (Apple M4 Pro, `Qwen/Qwen3-ASR-0.6B`, short fixture):
  - mean latency `1.7217s` -> `0.5464s` (`-68.3%`)
  - RTF `0.6796` -> `0.2157`
- Done (decoder-path optimization):
  - preallocated KV cache updates,
  - direct GQA fused attention path (no explicit K/V repeat).
- Done (decoder micro-optimization follow-up):
  - switched preallocated KV cache write path to in-place slice assignment
    after same-session A/B benchmarking (neutral-to-better overall).
- Done (I/O startup optimization):
  - native fast-path WAV loader for PCM/float `.wav` inputs, with ffmpeg
    fallback for unsupported formats, reducing short-clip overhead.
- Done (cold-start tokenizer optimization):
  - runtime tokenizer path is now native in-repo byte-level BPE
    (no `transformers` dependency in core transcription flow).
  - tokenizer receives resolved local snapshot path for deterministic local
    vocab/merges loading.
- Done (long-context encoder optimization):
  - added hybrid execution strategy for audio-encoder windowed attention:
    dense block-mask for small window counts, segmented per-window execution for
    long contexts (`num_windows >= 20`).
  - benchmark sweep shows crossover around `16-20` windows and substantial gains
    on long contexts (up to `~4.17x` in synthetic long-sequence benchmark).
- Current measured fp16 point from refreshed matrix run
  (Apple M4 Pro, `Qwen/Qwen3-ASR-0.6B`, float16):
  - short fixture: mean `0.4996s`, RTF `0.1972`
  - 10s clip: mean `0.9088s`, RTF `0.0909`

4. Forced aligner timestamps
- In progress.
- Timestamps now default to native MLX backend.
- Runtime aligner is now native-only (`mlx`).
- `qwen-asr` remains in optional parity/evaluation scripts as a reference lane.
- Native aligner groundwork now landed:
  - ported official text-unit preprocessing and LIS-based timestamp correction
    utilities into `mlx_qwen3_asr/forced_aligner.py`,
  - added regression coverage in `tests/test_forced_aligner.py`.
- Native MLX backend path is the runtime timestamp backend.
- Initial smoke benchmark on fixture audio shows strong latency upside
  (~`4.73x` mean vs `qwen_asr`) with matching sample word spans.
- Deterministic parity lane is now in place (`scripts/eval_aligner_parity.py`)
  and currently passes on `test-clean` (50 samples):
  - text-match rate: `1.0`,
  - timing MAE: `5.69ms`,
  - mean latency speedup: `~2.64x` vs `qwen_asr`.
- Native JA/KO tokenizer parity is now wired for the MLX backend:
  - Japanese via `nagisa`,
  - Korean via `soynlp` + official Korean tokenizer dictionary asset.
- Native MLX aligner quality hardening remains an active optimization lane.

5. Discoverability (README polish + PyPI)
- In progress.
- README validation section added; package metadata links updated.
- Benchmark process documented (`scripts/benchmark_asr.py`, `docs/BENCHMARKING.md`).
- Default model for `transcribe()`/CLI is now `Qwen/Qwen3-ASR-0.6B` to keep
  one-line install/run fast and reliable on typical Apple Silicon machines.

## Next Exploration Queue

Near-term work should remain correctness-gated and benchmark-driven:

1. **Transcription server** (`mlx-qwen3-asr serve`)
- Status: implemented and shipped. 46 tests, hardened from external review.
- Scope: FastAPI + uvicorn behind `[serve]` optional extra. Single-tenant,
  API key auth, async job model, sequential FIFO, in-memory job store with TTL.
  File upload only (no URL ingestion in v1). Backpressure via atomic queue admission.
  Job ownership enforced per API key. Error sanitization. Config validation.
- Gate: passed — server starts, accepts uploads, returns correct transcriptions,
  handles auth/rate-limiting/backpressure/job-isolation. Integration tests pass.
- Spec: `docs/server/ADR-001-transcription-server.md`, `docs/server/API-SPEC.md`

2. Native MLX forced aligner (timestamps) quality hardening
- Goal: continue quality hardening now that runtime PyTorch dependency is removed.
- Gate: word-level timing quality must be competitive with current `qwen-asr` backend.

2. Quantized model publication lane
- Goal: publish vetted `4bit-g64` and `8bit-g64` artifacts to HuggingFace.
- Gate: published artifacts must reproduce current local WER/RTF envelope.

3. Long-form robustness benchmark expansion
- Goal: extend golden eval + latency coverage beyond the current short fixture and 10s clip.
- Gate: no quality regressions on >30s and multi-minute real-world clips.

4. Evaluation coverage expansion (quality lanes)
- Goal: close remaining WER/quality gaps tracked in `docs/EVAL_GAPS.md`
  (currently centered on publishing streaming-manifest artifacts from maintained datasets).
- Gate: each lane must produce versioned benchmark artifacts and remain reproducible.

5. Decode API cleanliness and cache lifecycle rigor
- Goal: keep model/generation boundaries explicit (`prefill/step` + session ownership)
  to support maintainability and future low-risk optimizations.
- Gate: no parity regression and no additional hidden global state.

6. Speculative decoding prototype for 1.7B (paper-backed)
- Status: prototype implemented with strict-parity verification path and benchmark harness.
- Result (current): parity passed, latency regressed on tested short/10s clips.
- Next gate: require measurable latency win before any default-path adoption.
- Sources:
  - https://arxiv.org/abs/2211.17192
  - https://arxiv.org/abs/2507.18181
  - https://arxiv.org/abs/2507.21522

7. Streaming policy (deprioritized)
- Keep streaming explicitly experimental and avoid major roadmap allocation
  until the core offline quality/speed/timestamp track is fully production-grade.
- Current lane status: multi-file streaming evaluator and optional gate hook are
  implemented; remaining work is artifact publication cadence.

8. Speaker diarization (optional extra)
- Status: shipped as optional offline integration in API/CLI runtime.
- Runtime path: `pyannote.audio` backend behind `mlx-qwen3-asr[diarize]`.
- Guardrail: core ASR path remains native MLX and torch-free.
- Focus: maintain stable diarization output contract and benchmark gates,
  without carrying speculative native diarization scaffolding.

## Acceptance Gates

- Token parity: deterministic greedy parity test passes on reference fixtures.
- Reliability: long audio regression tests pass (`>30s` mel lengths).
- Golden quality: LibriSpeech sampled WER/CER trend remains within policy threshold.
- Packaging: reproducible release checklist (version bump, build, publish).
