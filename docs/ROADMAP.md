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
- Done (2026-09-19). Published `moona3k/mlx-qwen3-asr-{0.6b,1.7b}-{4bit,8bit}`
  with model cards (`docs/model-cards/`) and per-sample evals
  (`docs/benchmarks/2026-09-19-quantized-artifacts-*.json`). 8-bit is
  hypothesis-identical to fp16 for both sizes; 4-bit uses an 8-bit audio
  encoder (`--encoder-bits 8`, the encoder carried most of the all-4-bit loss)
  and scores 2.37% WER (0.6B, fp16 2.33%) and 1.73% (1.7B, fp16 1.94%).
  Mixed widths need the per-module loader shipped in 0.4.3.
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

## Handoff (2026-09-19): what separates "fixed" from "won't regress"

Ordered by how much a regression would cost. Each item has an acceptance
criterion so the next agent can tell when it is done.

1. **Reference-scored streaming gate.** Done 2026-09-19.
   `scripts/eval_streaming_manifest.py` scores `final_text` against manifest
   references (`quality_vs_reference`, schema v1.2) and the strict release
   gate runs the lane by default against the multilingual-100 manifest with a
   ceiling of offline primary error + 3pp.
   `test_reference_gate_would_have_failed_pre_fix_streaming_decoder` re-scores
   the committed pre-#26 artifact and asserts the gate rejects it (57.6% vs a
   12.5% ceiling) while the post-#26 artifact passes (11.4%).
   The long-form 10 x 75 s manifest is under the same ceiling (1.25pp headroom).
2. **Forced aligner audit.** Done 2026-09-19. Word timestamps: 100% text
   match, 5.57 ms MAE on MLX 0.30.6 and 0.32.2 (February: 5.69 ms). New
   `scripts/eval_aligner_encoder_parity.py` measures the aligner encoder
   against fp32 PyTorch: 0.089% max relative error on 50 clips, both MLX
   versions; wired into `RUN_ALIGNER_PARITY=1`. Finding: the CPU `qwen-asr`
   reference never applies the encoder's 800-frame attention window
   (`_prepare_attention_mask` is defined, not called), so as-shipped
   comparisons show 5-13% error on clips over 8 s that vanish with the mask.
   Artifact: `docs/benchmarks/2026-09-19-aligner-parity-50.md`.
   Reported upstream as QwenLM/Qwen3-ASR#213 (repro: 24.9% encoder
   divergence on a 20 s clip, 4/22 long-clip transcripts change).
3. **Streaming prefix reuse.** Done 2026-09-19 (Decision 30). Encoder output
   and decoder KV for complete 8 s attention windows are cached across chunks,
   keyed on the log-mel global max. Long-form RTF 0.104 -> 0.083 (same-day
   cache-off baseline), multilingual-100 0.090 -> 0.081; 2/20 and 3/200
   hypotheses changed, each equal or better; both strict lanes pass. The
   premise that the encoder dominated was wrong: it was 18% of per-chunk
   time, prefill 31%, generation 50%. Caching both encoder and KV addressed
   the first two; generation is the floor of the re-decode recipe, so the
   "toward 0.06" target is not reachable without changing the recipe
   (fewer rolled-back tokens, or speculative decoding of the tail).
4. **Window-commit at silence.** Done 2026-09-19 (Decision 31). The commit
   cut moves to the quietest >= 120 ms pause in the last 2 s of the window and
   the audio after it is carried into the new window; new windows inherit
   the detected language. Long-form lane, hard cut vs silence cut, same code:
   fixed 12.28% -> 11.87%, energy 12.08% -> 12.23% (the rise is numeral
   formatting variance on one English row, not a boundary artifact); the
   boundary artifacts visible in the hard-cut hypotheses ("for. Cost-saving",
   "EPC. Earlier", "que.", a spurious "The middle") are gone; zero adjacent
   duplicate words either way; +9.6% RTF interleaved. The language carry
   alone fixed a 59% Hindi row where a fresh window had drifted to Indonesian
   and looped.
5. **Publishing token.** Decided 2026-09-19: publishing stays on a
   maintainer machine; the token lives in the gitignored `.secrets/hf_token`
   (or `HF_TOKEN`, or the `huggingface-cli login` file), resolved by
   `scripts/publish_quantized.py::resolve_hf_token`; a test asserts the file
   is gitignored. `publish-quantized.yml` remains usable by anyone who adds a
   repo secret, but it is not on the release path.
6. **Broaden the hardware/MLX matrix.** Half done 2026-09-19: every eval and
   benchmark artifact now records `runtime` (chip, memory, macOS, Python, MLX,
   commit) via `scripts/eval/provenance.py`, including the nightly uploads;
   `docs/BENCHMARKS.md` has a hardware matrix with the two known hosts (M4 Pro
   48 GB; `macos-14` runner 7 GB).
   - Still needed: one committed result from an 8-16 GB Apple Silicon machine.
     Requires hardware the maintainer does not have; the matrix section gives
     the two commands and the file-naming rule for a contributor.

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
- Status 2026-09-19: audited on MLX 0.30.6 and 0.32.2, 100% text match,
  5.57 ms MAE, encoder output within 0.09% of fp32 reference
  (`docs/benchmarks/2026-09-19-aligner-parity-50.md`).

2. Quantized model publication lane
- Done 2026-09-19 (see Status item 3). Remaining: re-publish when the source
  checkpoints or the quantization recipe change. Publishing runs locally with
  the token in the gitignored `.secrets/hf_token` (handoff item 5).

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

7. Streaming
- 2026-09-19: rebuilt on the official re-feed + prefix-rollback recipe after
  the maintained-manifest lane measured 56% primary error for the incremental
  KV-cache design (offline 9.5%). Now 11.4% (multilingual-100) and 12.3%
  (long-form 10x75 s) at RTF 0.08 / 0.18. See Decision 29 and
  `docs/benchmarks/2026-09-19-streaming-manifest-*.json`.
- Next: encoder-output caching across re-decodes; overlap at window commit;
  gate the manifest lane in strict release mode.

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
