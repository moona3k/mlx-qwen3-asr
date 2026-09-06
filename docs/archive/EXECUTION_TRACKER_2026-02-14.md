# Execution Tracker (2026-02-14)

North-star execution notes for the current optimization/refactor wave.

## Goal

Ship the highest-quality native MLX Qwen3-ASR path while preserving correctness gates.

## Current Priorities

1. Decode/session architecture decoupling for production-grade core inference.
2. Keep all claims benchmark-backed and quality-gate clean.
3. Prioritize offline quality/speed/timestamp rigor over streaming feature expansion.

## Verified Findings

### 1) Custom mel parity status (updated)

Measured against current `compute_features()` (HF WhisperFeatureExtractor):

- Frame count mismatch:
  - HF path returns `N` frames for `N*160` samples (e.g., 2s -> 200).
  - Current custom `log_mel_spectrogram()` returns `N+1` frames (e.g., 2s -> 201).
- Value mismatch after truncating custom to HF frame count:
  - MAE around `0.153-0.155` on normalized log-mel.
  - Correlation around `0.09-0.12`.

Status update:
- Fixed by (a) regenerating filterbank to match Whisper exactly and
  (b) trimming final STFT frame before normalization.
- Parity now passes on deterministic random + fixture set:
  - max MAE: `2.83e-07`
  - max abs diff: `1.13e-04`
  - exact frame-count parity.
- Artifacts:
  - `docs/benchmarks/2026-02-14-mel-parity.json`
  - `docs/benchmarks/2026-02-14-mel-parity.md`

### 2) Mel filter mismatch root cause (resolved)

Comparing `assets/mel_filters.npz` against `WhisperFeatureExtractor.mel_filters`
(transposed to shape `(128, 201)`):

- MAE: `0.0002265`
- Max abs diff: `0.05816`

Resolution:
- `scripts/generate_mel_filters.py` now generates Whisper-compatible filters
  (`mel_filter_bank(..., norm="slaney", mel_scale="slaney")`).
- `compute_features(..., padding="do_not_pad")` now uses the corrected custom
  mel path by default.

### 3) Official streaming semantics context

From upstream `qwen_asr/inference/qwen3_asr.py`:

- Streaming path re-feeds accumulated audio context each chunk.
- Uses `unfixed_chunk_num` and `unfixed_token_num` rollback controls.
- Streaming in official stack is backend-limited (`vllm`) and no timestamps.

We already aligned our rolling streaming controls to the same knobs and added
bounded-context behavior for stable per-chunk cost.

### 4) Encoder long-context execution strategy

- Dense block-mask encoder attention is fine for small window counts but scales
  poorly as sequence length grows.
- A segmented per-window execution path is numerically equivalent (max-abs diff
  around `1e-6` in synthetic checks) and significantly faster for long contexts.
- Hybrid threshold selected from benchmark sweep:
  - use segmented path when `num_windows >= 20`.
- Artifacts:
  - `docs/benchmarks/2026-02-14-encoder-windowing-threshold.json`
  - `docs/benchmarks/2026-02-14-encoder-windowing-threshold.md`

### 5) Output parse robustness hardening

- Implemented upstream-style repetition cleanup in ASR output parsing.
- Added explicit handling for:
  - `language None<asr_text>` empty-audio case,
  - forced-language parsing path,
  - consistent trailing special-token cleanup.
- Added regression tests in tokenizer/transcribe lanes.

### 6) Streaming input hardening and dead-state cleanup

- Removed unused `previous_tokens` field from `StreamingState`.
- Added explicit input normalization in `feed_audio(...)`:
  - accepts non-1D arrays (flattens),
  - converts `int16` PCM to `float32` in `[-1, 1]`.
- Added init-time validation for streaming parameters:
  - `chunk_size_sec > 0`, `max_context_sec > 0`, `sample_rate > 0`.
- Added regression tests for these behaviors.

### 7) Speculative decoding prototype (target/draft) landed

- Added cache trim support in `KVCache` and model `step_many(...)` API for
  batched verification steps.
- Added experimental speculative generation path (`generate_speculative`) with
  strict greedy parity behavior and cache rewind logic.
- Wired optional draft-model controls through Python API/CLI:
  - `transcribe(..., draft_model=..., num_draft_tokens=...)`
  - `mlx-qwen3-asr --draft-model ... --num-draft-tokens ...`
- Added benchmark harness:
  - `scripts/benchmark_speculative.py`
- Current measured status:
  - parity: `text_match=true`,
  - performance: slower on tested short/10s workloads vs baseline.
  - therefore remains experimental and non-default.

### 8) Multi-model cache residency hardening

- `_ModelHolder` cache now keys by `(model_path, dtype)` instead of storing
  only one global model instance.
- Why this matters:
  - speculative runs often use both target and draft models,
  - single-entry cache caused avoidable reload thrash across repeated calls.
- Added regression coverage to ensure two model IDs remain cached without
  evicting each other in steady-state usage.

### 9) Final paper/repo review synthesis (actionability)

Primary-source refresh across Qwen3-ASR, Swift ports, and ASR decoding papers:

- Lossless speculative decoding papers remain directionally relevant, but local
  evidence still shows no win on current short/10s workloads; keep experimental.
- Whisper/WhisperX-style long-form segmentation + alignment remains the highest
  practical algorithmic lane for post-recording quality stability.
- No paper-backed shortcut was found that can safely replace the current
  correctness-first gates for MRoPE/audio injection/long-context handling.

### 10) Post-change correctness checkpoint

- Ran release gate after final cache/research pass:
  - `RUN_REFERENCE_PARITY=1 python scripts/quality_gate.py --mode release`
  - Result: PASS (full pytest + reference parity lane green).

### 11) Native max_length mel path hardening

- `compute_features(..., padding="max_length")` now uses the native custom mel
  path (with deterministic zero-padding to 3000 frames for short clips), rather
  than routing through HF extractor.
- Compatibility fallback to HF remains for uncommon padding modes.
- Result:
  - reduced HF extractor dependency in common feature paths,
  - preserved `feature_lens` behavior and existing tests.

### 12) Quantization matrix refresh with speaker-balanced defaults

- Ran `scripts/benchmark_quantization_matrix.py` with:
  - configs: `fp16,4:64,8:64`
  - benchmark runs: `5`
  - eval lane: `test-clean`, `n=100`, `speaker_round_robin`
- Artifacts:
  - `docs/benchmarks/2026-02-14-quant-matrix-speaker100.json`
  - `docs/benchmarks/2026-02-14-quant-matrix-speaker100.md`
- Snapshot:
  - `4bit-g64` long speedup vs fp16: `4.68x`, WER delta `+0.004316`
  - `8bit-g64` long speedup vs fp16: `3.11x`, WER delta `+0.000432`

### 13) Audio frontend cache cleanup (native path hot-loop)

- Added cache for mel filterbank asset loading (`_mel_filters_np`) so repeated
  chunk processing does not re-open/parse `mel_filters.npz`.
- Added cache for Hann window creation (`_hann_window`) used by STFT.
- Added regression tests to ensure cache behavior is active.

### 14) Latest post-change release checkpoint

- Re-ran full release gate after native max_length path + audio frontend cache
  cleanups:
  - `RUN_REFERENCE_PARITY=1 python scripts/quality_gate.py --mode release`
  - Result: PASS.

### 15) End-to-end docs alignment pass

- Synchronized README and benchmarking docs to the latest committed
  speaker-balanced matrix artifact:
  - `docs/benchmarks/2026-02-14-quant-matrix-speaker100.{json,md}`
- Updated runtime/quality snapshot numbers to match current artifact values and
  removed stale latency rows from older runs.
- Verified benchmark-path references: only expected generated outputs are absent
  (`latest*`, `nightly*`, CI upload filenames, and one-off local golden paths).

### 16) Forced-aligner LIS hotpath optimized with legacy-exact behavior

- `ForcedAlignTextProcessor.fix_timestamp(...)` now uses an O(n log n)
  LIS solver (Fenwick tree + coordinate compression) instead of O(n^2) DP.
- Kept exact legacy tie semantics to avoid timestamp behavior drift:
  - earliest predecessor for equal-length candidates,
  - earliest end index for global LIS length.
- Added regression coverage:
  - randomized equivalence against the legacy O(n^2) LIS reference,
  - duplicate-heavy non-decreasing edge-case check.

Post-change validation:
- Fast gate: PASS (`285 passed, 1 skipped`).
- Release gate (with parity lane): PASS (`286 passed` + `reference parity passed`).

### 17) End-to-end docs/truth alignment + CLI timestamp UX clarity

- README was refreshed for stricter claim honesty:
  - benchmark scope and machine context are explicit,
  - timestamp parity claims now include dataset/scope + artifact pointers,
  - limitations section now states streaming/speculative/native-aligner status.
- Comparison/decision docs were updated to match current implementation state:
  - dual timestamp backend policy (`mlx` default, `qwen_asr` optional reference),
  - implementation comparison now avoids stale hard claims and emphasizes
    artifact-backed evidence.
- CLI timestamp help/error messaging now clarifies backend dependency behavior:
  - `qwen_asr` backend requires optional `qwen-asr`,
  - users can choose `--aligner-backend mlx` to avoid that dependency.
- Post-change validation:
  - fast gate: PASS (`286 passed, 1 skipped`)
  - release gate: PASS (`287 passed` + reference parity lane pass)

### 18) Fused-attention fallback hardening (error visibility)

- Tightened `_scaled_dot_product_attention(...)` fused-kernel fallback policy:
  - still falls back on `TypeError` / `ValueError`,
  - for `RuntimeError`, only falls back when error clearly indicates unsupported
    fused-kernel compatibility (`not implemented` / `unsupported`),
  - otherwise re-raises to avoid hiding real runtime failures (e.g., OOM).
- Added regression tests:
  - compatibility-style runtime error falls back and returns valid output,
  - non-compatibility runtime error propagates.

Post-change validation:
- fast gate: PASS (`288 passed, 1 skipped`)
- release gate: PASS (`289 passed` + reference parity lane pass)

### 19) Release gate + native aligner parity lane checkpoint

- Ran full release gate with aligner parity lane enabled:

```bash
RUN_REFERENCE_PARITY=1 RUN_ALIGNER_PARITY=1 ALIGNER_PARITY_SAMPLES=10 \
python scripts/quality_gate.py --mode release
```

- Result: PASS.
- Aligner parity snapshot (`test-clean`, `n=10`, English):
  - text match rate: `1.0000`
  - timing MAE all: `1.6667 ms`
  - mean latency: `mlx=0.2972s`, `qwen_asr=0.9129s`
  - relative speed (`qwen_asr / mlx`): `3.07x`

### 20) README artifact-reference integrity gate

- Added `tests/test_docs_integrity.py`:
  - validates dated benchmark artifact references in `README.md` resolve to
    real committed files (including brace-expanded forms like `{json,md}`).
- Purpose:
  - prevent doc drift where performance/quality claims reference missing
    artifacts,
  - keep README evidence links auditable as numbers evolve.

Post-change validation:
- fast gate: PASS (`289 passed, 1 skipped`)
- release gate: PASS (`290 passed` + reference parity lane pass)

### 21) Native-first timestamp backend default (Python path)

- Switched forced-aligner default backend from `qwen_asr` to native `mlx`.
- CLI `--aligner-backend` default now `mlx`; `qwen_asr` remains explicit opt-in.
- Updated docs/tests to match native-first policy.

### 22) Targeted bug-fix pass from external audit feedback

Validated and addressed:

1. Speculative cache rollback bug:
- fixed draft cache trim to match target trim (`num_draft - accepted`).

2. Decode-position robustness:
- `_build_decode_positions(...)` now always returns an array (empty tail for
  `max_new_tokens <= 1`) to remove `None` shape hazards.

3. Session API parity:
- added `draft_model` + `num_draft_tokens` to `Session.transcribe(...)`.

4. Session tokenizer safety:
- pre-loaded model path now uses embedded source metadata;
- explicit error when metadata/tokenizer path is unavailable.

5. Streaming guards:
- reject `max_context_sec < chunk_size_sec`,
- treat empty PCM input as explicit no-op.

All fixes include regression coverage in `tests/test_generate.py`,
`tests/test_session.py`, and `tests/test_streaming.py`.

### 23) Expanded reference parity coverage lane (new)

- Added `scripts/eval_reference_parity_suite.py` for broader token-level parity:
  - deterministic `test-clean` + `test-other` sampling,
  - optional synthetic long/multi-speaker mixes,
  - optional external multilingual manifest input.
- Wired as optional release-gate lane via `RUN_REFERENCE_PARITY_SUITE=1`.
- Smoke run artifacts:
  - `docs/benchmarks/2026-02-14-reference-parity-suite-smoke.json`
  - `docs/benchmarks/2026-02-14-reference-parity-suite-smoke.md`

Honest finding:
- single-fixture parity does not extrapolate to broad token-exact parity.
- smoke lane currently shows mismatches on harder/longer samples and should
  remain exploratory until those are resolved.

## Decision Gates

### Gate A: Mel backend switch

Status: COMPLETE

Switched only after all were true:

1. Frame-count parity is exact across short and long clips.
2. Value parity threshold:
   - MAE <= `1e-3` (target), or a justified threshold validated by transcript parity.
3. Transcript parity:
   - No regression in reference parity lane.
   - No measurable degradation in sampled LibriSpeech WER/CER lane.

### Gate B: Streaming architecture claims

Keep streaming marked experimental until:

1. Resumable decode cache path is implemented and benchmarked.
2. Streaming benchmark lane shows predictable latency under long sessions.
3. Regression tests cover multilingual rollback behavior and state transitions.

## Immediate Next Tasks

1. (Complete) Added explicit `Session` API skeleton for explicit state ownership.
2. (Complete) Introduced model-level `prefill/step` interfaces so `generate()` no longer reaches into model internals.
3. (In progress) Expand golden-set benchmark coverage (quality + latency) for release and quantized artifacts.
4. Keep streaming experimental; only revisit deeper streaming work after core gates are saturated.

### 24) Reference parity suite hardening (coverage + multilingual honesty)

- Extended `scripts/eval_reference_parity_suite.py` with:
  - optional deterministic noise variants (`--include-noise-variants`, `--noise-snrs-db`, `--noise-seed`),
  - dual parity metrics:
    - strict token parity (`token_match_rate`),
    - Unicode-safe normalized text parity (`text_match_rate`),
  - per-row debug fields (`ref_text_raw`, `mlx_text_raw`, normalized variants),
  - optional text-threshold fail gate (`--fail-text-match-rate-below`).
- Fixed a multilingual evaluation bias:
  - removed English-only normalization from this lane,
  - replaced with script-local Unicode-preserving normalization so CJK/non-Latin
    comparisons are not falsely collapsed to empty strings.
- Wired optional parity-suite env controls through `scripts/quality_gate.py`:
  - `REFERENCE_PARITY_SUITE_MANIFEST_JSONL`,
  - `REFERENCE_PARITY_SUITE_INCLUDE_NOISE_VARIANTS`,
  - `REFERENCE_PARITY_SUITE_NOISE_SNRS_DB`,
  - `REFERENCE_PARITY_SUITE_NOISE_SEED`,
  - `REFERENCE_PARITY_SUITE_FAIL_TEXT_MATCH_RATE_BELOW`,
  - `REFERENCE_PARITY_SUITE_JSON_OUTPUT`.
- Added helper unit coverage:
  - deterministic noise-variant generation,
  - Unicode-safe normalization behavior.
- Re-ran release gate with parity-suite lane enabled (small mixed smoke):
  - artifact: `docs/benchmarks/2026-02-14-reference-parity-suite-smoke-v2.json`
  - token match rate: `0.40`
  - text match rate: `0.60`
  - lane remains exploratory (gap-finding), not a hard release blocker.

### 25) Broader English parity sweep (22-sample mixed lane)

- Ran expanded parity suite on deterministic LibriSpeech sample plus synthetic
  long/noise variants:
  - `--subsets test-clean,test-other`
  - `--samples-per-subset 5`
  - `--include-long-mixes --long-mixes 2 --long-mix-segments 4`
  - `--include-noise-variants --noise-snrs-db 10`
- Artifacts:
  - `docs/benchmarks/2026-02-14-reference-parity-suite-english22.json`
  - `docs/benchmarks/2026-02-14-reference-parity-suite-english22.md`
- Result summary:
  - token match rate: `16/22 = 0.7273`
  - text match rate: `18/22 = 0.8182`
  - largest remaining gap: synthetic long mixed-speaker clips (`0.0` token/text).

### 26) Reproducible multilingual parity manifest workflow + smoke run

- Added `scripts/build_multilingual_manifest.py`:
  - builds deterministic JSONL manifests from FLEURS (`google/fleurs`) across
    selected language configs,
  - resolves user-friendly aliases (`zh_cn -> cmn_hans_cn`, etc),
  - validates configs against available Hub files,
  - writes local 16k mono WAVs + manifest rows with language labels.
- Added helper tests in `tests/test_build_multilingual_manifest.py`.
- Important compatibility fix:
  - avoided `datasets` script-loader path (fails on `datasets>=4` for FLEURS),
  - switched to direct Hub TSV/audio-tar consumption (`hf_hub_download`).
- Ran multilingual smoke parity artifact set:
  - manifest: `docs/benchmarks/2026-02-14-fleurs-multilingual-smoke-manifest.jsonl`
  - parity JSON: `docs/benchmarks/2026-02-14-reference-parity-suite-multilingual-smoke.json`
  - parity summary: `docs/benchmarks/2026-02-14-reference-parity-suite-multilingual-smoke.md`
- Smoke result (4 languages, 1 sample each):
  - token match rate: `0.50` (2/4)
  - text match rate: `0.50` (2/4)
  - matches: English + Chinese; mismatches: Japanese + German.
- Verified via release-gate wiring:
  - `RUN_REFERENCE_PARITY=1 RUN_REFERENCE_PARITY_SUITE=1` with
    `REFERENCE_PARITY_SUITE_SUBSETS=''` and
    `REFERENCE_PARITY_SUITE_MANIFEST_JSONL=...` completed successfully.

### 27) Multilingual parity scale-up (20 and 100 sample lanes)

- Built deterministic multilingual manifests (10 languages):
  - `docs/benchmarks/2026-02-14-fleurs-multilingual-20-manifest.jsonl`
  - `docs/benchmarks/2026-02-14-fleurs-multilingual-100-manifest.jsonl`
- Ran manifest-only parity suite for each:
  - `docs/benchmarks/2026-02-14-reference-parity-suite-multilingual-20.json`
  - `docs/benchmarks/2026-02-14-reference-parity-suite-multilingual-100.json`
- Added mismatch taxonomy analyzer:
  - script: `scripts/analyze_reference_parity_mismatches.py`
  - tests: `tests/test_analyze_reference_parity_mismatches.py`
  - analysis artifacts:
    - `docs/benchmarks/2026-02-14-reference-parity-suite-multilingual-20-analysis.{json,md}`
    - `docs/benchmarks/2026-02-14-reference-parity-suite-multilingual-100-analysis.{json,md}`
    - `docs/benchmarks/2026-02-14-reference-parity-suite-multilingual-smoke-analysis.{json,md}`
- Results:
  - 20-sample lane: token/text parity `0.55`
  - 100-sample lane: token parity `0.64`, text parity `0.67`
  - hardest languages on 100-sample lane: Arabic/French/Hindi
    (token parity `0.40` each).

### 28) External evaluation-feedback validation (coverage gaps)

- External feedback was validated as directionally correct:
  - we have strong parity infrastructure, but quality lanes still have blind spots.
- Captured as a persistent doc so future follow-up is explicit:
  - `docs/EVAL_GAPS.md`
- Gaps recorded there include:
  - 1.7B WER lane,
  - `test-other` WER lane,
  - non-English quality lane (not only parity),
  - long-form quality lane (`>30s`, multi-minute),
  - direct MLX-vs-PyTorch quality comparison lane,
  - real-world audio lane.

### 29) External bug-audit triage + edge-case hardening

Validated and addressed the latest external review claims against current `main`.

Implemented fixes:
- Canonicalized in-memory audio handling via `load_audio(...)` for all
  `transcribe(...)` input types (including `np.ndarray`/`mx.array`), so array
  inputs now follow the same mono + dtype normalization path as tuple/file
  inputs.
- Added integer PCM normalization for array/tuple audio sources in `audio.py`
  to align with file decode semantics.
- Added explicit short-audio validation in `log_mel_spectrogram(...)` before
  clamping/reduction to avoid opaque zero-size reduction errors.
- Added defensive validation in `split_audio_into_chunks(...)`:
  - `sr > 0`
  - `max_chunk_sec > 0`
- Fixed generation boundary semantics:
  - `max_new_tokens == 0` now returns zero generated tokens for both greedy
    and speculative paths.
  - `max_new_tokens < 0` now raises `ValueError`.
- Streaming multichannel input now downmixes to mono instead of flattening
  interleaved samples.

Confirmed already-fixed / outdated claims:
- speculative cache rollback trim mismatch: already fixed before this pass
  (`target_cache.trim(...)` and `draft_cache.trim(...)` are symmetric).
- streaming `max_context_sec >= chunk_size_sec` validation: already enforced.
- streaming empty-array guard: already present (`feed_audio` no-op on empty).

Static typing status snapshot:
- Ran `mypy` baseline over `mlx_qwen3_asr scripts tests`.
- Current result: `50` errors in `12` files.
- This remains an advisory lane (not yet a required CI/release gate) until
  baseline debt is reduced in focused follow-up work.

Post-change validation:
- fast gate: PASS (`325 passed, 1 skipped`)

### 30) Long-form parity scale-up (10 languages, serial shard execution)

- Built deterministic long-form concatenation tooling:
  - script: `scripts/build_longform_manifest.py`
  - tests: `tests/test_build_longform_manifest.py`
- Evaluator manifest parsing now preserves `source_sample_ids` metadata:
  - `scripts/eval_reference_parity_suite.py`
  - regression: `tests/test_eval_reference_parity_suite.py`
- Created long-form manifests:
  - `docs/benchmarks/2026-02-14-fleurs-longform-smoke2-manifest.jsonl`
  - `docs/benchmarks/2026-02-14-fleurs-longform-10x75-manifest.jsonl`
- Ran long-form smoke parity with high token budget:
  - `docs/benchmarks/2026-02-14-reference-parity-suite-longform-smoke2-1024.json`
- Ran full 10-sample long-form lane serially (2-sample shards) to avoid prior
  process instability:
  - `docs/benchmarks/2026-02-15-reference-parity-suite-longform10-shard00.json`
  - `docs/benchmarks/2026-02-15-reference-parity-suite-longform10-shard01.json`
  - `docs/benchmarks/2026-02-15-reference-parity-suite-longform10-shard02.json`
  - `docs/benchmarks/2026-02-15-reference-parity-suite-longform10-shard03.json`
  - `docs/benchmarks/2026-02-15-reference-parity-suite-longform10-shard04.json`
- Aggregated run summary:
  - `docs/benchmarks/2026-02-15-reference-parity-suite-longform10.json`
  - `docs/benchmarks/2026-02-15-reference-parity-suite-longform10.md`
- Result snapshot:
  - token parity: `0.0` (0/10)
  - normalized text parity: `0.0` (0/10)
  - first mismatch index: mean `14.6`, median `13.5`, min `0`, max `28`
  - latency (mean): `mlx=11.55s`, `qwen_asr-ref=48.39s` (`4.19x` ref/mlx)

Interpretation:
- strict token parity is not a sufficient standalone long-form quality proxy;
  long-sequence greedy decode can diverge early even when transcript quality is
  close. This lane is now retained as a stress/parity signal, not a quality gate.

### 31) WER/CER quality lane made default in release gate

- Added explicit CER threshold support to `scripts/eval_librispeech.py`:
  - new arg: `--fail-cer-above`
  - shared threshold check helper for WER/CER.
- Added unit tests for threshold behavior:
  - `tests/test_eval_librispeech.py`
- Updated release gate (`scripts/quality_gate.py`) to run deterministic quality
  eval by default:
  - `scripts/eval_librispeech.py`
  - defaults:
    - subset: `test-clean`
    - samples: `20`
    - sampling: `speaker_round_robin`
    - thresholds: `WER<=0.10`, `CER<=0.06`
  - disable override: `RUN_QUALITY_EVAL=0`
- Updated docs:
  - `docs/QUALITY_GATE.md`

Intent:
- token parity remains necessary for correctness drift detection,
- WER/CER is now first-class in default release validation to prevent
  over-indexing on token-exact metrics alone.

### 32) Release-gate quality lane validated at default sample size

- Ran release gate with default quality lane semantics and 20 deterministic
  LibriSpeech samples:

```bash
RUN_QUALITY_EVAL=1 QUALITY_EVAL_SAMPLES=20 \
python scripts/quality_gate.py --mode release
```

- Result: PASS.
- Quality snapshot (`Qwen/Qwen3-ASR-0.6B`, `test-clean`, `n=20`):
  - WER: `0.020446`
  - CER: `0.004052`
  - mean latency: `1.40s` per sample
- Saved standalone artifact:
  - `docs/benchmarks/2026-02-15-librispeech-release-gate-20.json`

### 33) Manifest-based multilingual/long-form quality lane (WER/CER)

- Added `scripts/eval_manifest_quality.py`:
  - evaluates JSONL manifests containing `audio_path` + `reference_text`,
  - uses Unicode-safe normalization for multilingual text,
  - computes both WER and CER,
  - computes language-aware primary metric:
    - CER for Chinese/Japanese/Korean,
    - WER for other languages.
- Added helper tests:
  - `tests/test_eval_manifest_quality.py`
- Wired optional release-gate lane:
  - env flag: `RUN_MANIFEST_QUALITY_EVAL=1`
  - required manifest: `MANIFEST_QUALITY_EVAL_JSONL=...`
  - integrated via `scripts/quality_gate.py`
  - documented in `docs/QUALITY_GATE.md`

Run executed on long-form multilingual manifest (`n=10`):

```bash
python scripts/eval_manifest_quality.py \
  --manifest-jsonl docs/benchmarks/2026-02-14-fleurs-longform-10x75-manifest.jsonl \
  --model Qwen/Qwen3-ASR-0.6B \
  --dtype float16 \
  --max-new-tokens 1024 \
  --json-output docs/benchmarks/2026-02-15-manifest-quality-longform10.json
```

Results snapshot:
- samples: `10`
- WER: `0.1671`
- CER: `0.0704`
- primary error rate: `0.1156`
- mean latency: `11.13s`

Artifacts:
- `docs/benchmarks/2026-02-15-manifest-quality-longform10.json`
- `docs/benchmarks/2026-02-15-manifest-quality-longform10.md`

Interpretation:
- this closes a major gap where long-form quality was inferred only from token
  parity; we now have direct WER/CER for multilingual long-form synthetic data.
- remaining gap is scale/domain breadth (real-world audio + larger sample size).

### 34) Multilingual short-form quality lane (FLEURS manifest, n=100)

- Ran manifest-quality evaluator on deterministic multilingual short-form set:

```bash
python scripts/eval_manifest_quality.py \
  --manifest-jsonl docs/benchmarks/2026-02-14-fleurs-multilingual-100-manifest.jsonl \
  --model Qwen/Qwen3-ASR-0.6B \
  --dtype float16 \
  --max-new-tokens 256 \
  --json-output docs/benchmarks/2026-02-15-manifest-quality-multilingual100.json
```

- Aggregate results (`10 languages x 10`):
  - samples: `100`
  - WER: `0.1589`
  - CER: `0.0541`
  - primary error rate: `0.0937`
  - mean latency: `1.35s`
- Notable per-language primary metrics:
  - English: `0.0463`
  - Chinese: `0.0437` (CER primary)
  - Japanese: `0.0846` (CER primary)
  - Korean: `0.0675` (CER primary)
  - Hindi: `0.1667`
  - Arabic: `0.2150`

Artifacts:
- `docs/benchmarks/2026-02-15-manifest-quality-multilingual100.json`
- `docs/benchmarks/2026-02-15-manifest-quality-multilingual100.md`

Impact:
- this upgrades non-English quality validation from parity-only to direct
  reference-based metrics on a broader multilingual sample.

### 35) Strict release profile promoted to executable gate policy

- Added strict release profile support in `scripts/quality_gate.py`:
  - `RUN_STRICT_RELEASE=1` enables truth-quality + parity + perf lanes by
    default in `--mode release`.
  - release parity test now executes with `RUN_REFERENCE_PARITY=1` and fails
    when `torch`/`qwen_asr` deps are missing (unless explicitly overridden).
  - strict defaults:
    - manifest quality lane on,
    - multilingual parity suite on with practical floor thresholds,
    - perf benchmark lane on with RTF/latency thresholds.
- Updated `docs/QUALITY_GATE.md` with exact strict-profile command and env
  controls.
- Validation:
  - fast gate: PASS,
  - strict release smoke: PASS (all lanes exercised).

Why this matters:
- moves the highest-bar intent from prose into repeatable pass/fail checks,
- reduces risk of accidental release with missing multilingual/perf evidence.

### 36) Multilingual language-alias hardening for forced-language path

- Added language canonicalization in `mlx_qwen3_asr/tokenizer.py`:
  - new helper: `canonicalize_language(...)`
  - normalizes common codes/aliases (e.g. `de_de`, `fr`, `zh-cn`) to canonical
    names used in prompt forcing (e.g. `German`, `French`, `Chinese`).
- Updated prompt construction:
  - `Tokenizer.build_prompt_tokens(...)` now uses canonicalized language labels
    before emitting `language {X}<asr_text>`.
- Hardened forced-language parse behavior:
  - `parse_asr_output(..., user_language=...)` now strips an unexpected
    `language ...<asr_text>` prefix if the model emits it, rather than treating
    the entire string as plain transcript.
  - forced language output is canonicalized consistently.
- Updated transcription pipeline:
  - `mlx_qwen3_asr/transcribe.py` canonicalizes `language` once and reuses it
    for prompt forcing, parser input, and aligner calls.
- Added regression tests:
  - `tests/test_tokenizer.py` (forced-language canonicalization + prefix strip),
  - `tests/test_transcribe.py` (forced language code normalized before prompt).

Validation:
- targeted tests: PASS (`tests/test_tokenizer.py`, `tests/test_transcribe.py`),
- full fast gate: PASS (`349 passed, 1 skipped`).

### 37) Multilingual triage checkpoint: no aggregate movement after lang-canon

Re-ran multilingual lanes after language canonicalization hardening:

- manifest quality lane:
  - `docs/benchmarks/2026-02-15-manifest-quality-multilingual100-post-langcanon.json`
- parity lane:
  - `docs/benchmarks/2026-02-15-reference-parity-suite-multilingual100-post-langcanon.json`

Result:
- aggregate quality unchanged vs prior multilingual-100 baseline:
  - WER delta: `+0.000000`
  - CER delta: `+0.000000`
  - primary error delta: `+0.000000`
- aggregate parity unchanged:
  - token-match delta: `+0.0000`
  - text-match delta: `+0.0000`

Interpretation:
- canonicalized forced-language handling improves API robustness but does not
  move multilingual benchmark aggregates on the current FLEURS-100 lane.

### 38) Precision sensitivity check (fp16 vs fp32) for multilingual parity

Ran parity suite on multilingual-20 with `dtype=float32`:

- artifact:
  - `docs/benchmarks/2026-02-15-reference-parity-suite-multilingual20-fp32.json`

Compared against prior fp16 multilingual-20 artifact:
- token match: `0.55` vs `0.55` (no change)
- text match: `0.55` vs `0.55` (no change)
- MLX latency: slightly slower in fp32.

Conclusion:
- residual multilingual parity gap is unlikely to be a simple floating-point
  precision issue in current decode path.

### 39) Checkpoint summary artifact for triage decisions

Added machine/human-readable checkpoint comparison:

- `docs/benchmarks/2026-02-15-multilingual-triage-checkpoint.json`
- `docs/benchmarks/2026-02-15-multilingual-triage-checkpoint.md`

Purpose:
- keep decision history explicit before moving to deeper multilingual work
  (likely model-path parity investigation rather than frontend/postprocess only).

### 40) Logit-level mismatch probe (multilingual100 parity gaps)

Added a dedicated diagnostic script:

- `scripts/eval_logit_parity_probe.py`

Purpose:
- probe multilingual parity mismatch rows at the first mismatch position and
  compare MLX vs reference top-k logits/margins to distinguish:
  - near-tie decode flips (argmax boundary behavior),
  - stronger logit/path divergence.

Artifacts:
- `docs/benchmarks/2026-02-15-logit-parity-probe-multilingual100-mismatches.json`
- `docs/benchmarks/2026-02-15-logit-parity-probe-multilingual100-mismatches.md`

Observed on current multilingual-100 mismatch rows (`n=36`):
- near-tie rows (`ref_margin<=0.5` OR `mlx_margin<=0.5`): `31/36` (`86.1%`)
- strict top1/top2 cross-swaps: `30/36` (`83.3%`)
- only `5/36` rows remained non-near-tie at probe position.

Interpretation:
- most mismatches are top-2 rank flips at narrow margins rather than broad
  candidate-set divergence; this supports prioritizing targeted model-path
  parity investigation on the small non-near subset before making larger
  decoding changes.

### 41) FP32 recheck of non-near mismatch subset

Ran logit probe on the five highest-priority non-near rows from section 40
with `dtype=float32`:

- `docs/benchmarks/2026-02-15-logit-parity-probe-nonnear5-fp32.json`
- `docs/benchmarks/2026-02-15-logit-parity-probe-nonnear5-fp32.md`

Observed:
- mismatches remain `5/5`,
- near-tie rows remain `0/5`,
- strict top1/top2 cross-swaps still present on `3/5`.

Conclusion:
- this residual subset is not explained by fp16 precision effects; follow-up
  should target model-path/logit-source parity for these specific rows.

### 42) Non-near subset stage-localization (encoder + teacher-forced decode)

Added stage-level parity artifact on the five non-near rows:

- `docs/benchmarks/2026-02-15-nonnear5-stage-parity.json`
- `docs/benchmarks/2026-02-15-nonnear5-stage-parity.md`

Method:
- same input features/prompt path as reference,
- compared MLX vs reference audio-encoder outputs,
- ran teacher-forced decode parity using reference-generated token prefixes.

Observed:
- audio-feature cosine on this subset: ~`0.89` to `0.96` (close, not identical),
- teacher-forced top1 parity remains high (`~0.94` to `~0.98`),
- each sample still has a deterministic first top1 divergence step matching the
  non-near mismatch location.

Interpretation:
- residual gaps are reproducible beyond free-running decode; evidence points to
  model-path numeric/representation parity (not just decode policy effects).

### 43) Cache semantics ruled out on non-near subset

Added cache-consistency artifact:

- `docs/benchmarks/2026-02-15-cache-vs-recompute-nonnear5.json`
- `docs/benchmarks/2026-02-15-cache-vs-recompute-nonnear5.md`

Method:
- for each non-near row, compared:
  - normal cached decode path,
  - full-prefix recompute decode path (rebuild cache every step).

Result:
- `5/5` rows exactly equal; first mismatch index `-1` for all rows.

Interpretation:
- residual multilingual non-near mismatches are not caused by KV-cache update
  semantics in MLX decode.

### 44) Audio encoder fp16 clamp parity fix (upstream alignment)

Aligned `AudioEncoderLayer` with upstream torch behavior by adding the
post-layer fp16 clamp (`finfo(float16).max - 1000`) in:

- `mlx_qwen3_asr/encoder.py`

Added regression coverage:

- `tests/test_model.py::TestAudioEncoderLayer::test_float16_path_clamps_extreme_values`

Follow-up probe on non-near subset:

- `docs/benchmarks/2026-02-15-logit-parity-probe-nonnear5-post-enc-clamp.json`
- `docs/benchmarks/2026-02-15-logit-parity-probe-nonnear5-post-enc-clamp.md`
- comparison: `docs/benchmarks/2026-02-15-logit-parity-probe-nonnear5-post-enc-clamp-compare.json`

Observed on this subset:
- mismatch index / top-token identity at probe step unchanged (`5/5` rows).

Interpretation:
- clamp alignment is still the correct upstream-compatibility fix and protects
  fp16 stability, but it does not explain the current residual non-near
  multilingual parity gap by itself.

### 45) Audio encoder layerwise drift localization (non-near subset)

Added layerwise audio-encoder parity artifact:

- `docs/benchmarks/2026-02-15-audio-encoder-layerwise-parity-nonnear5.json`
- `docs/benchmarks/2026-02-15-audio-encoder-layerwise-parity-nonnear5.md`

Method:
- compared MLX and reference audio encoder hidden states per layer (L1..L18)
  on the five non-near mismatch rows, using identical preprocessed features.

Observed trend:
- high similarity at shallow layers (often `~0.99` cosine at L1),
- gradual cosine decay across depth,
- largest decay on the Japanese sample:
  - L1 `0.9889` → L18 `0.9100` → final `0.8933`.

Interpretation:
- residual non-near mismatches are consistent with gradual encoder
  representation drift, not a single decode/cache failure point.

### 46) Windowed-mask value alignment (finfo.min) in encoder attention

Aligned `_create_windowed_mask(...)` blocked-value semantics with upstream:

- from hardcoded `-1e9` to `finfo(dtype).min` in
  `mlx_qwen3_asr/encoder.py`

Added regression coverage:

- `tests/test_model.py::TestWindowedEncoderExecution::test_mask_uses_dtype_min_for_blocked_positions`

Follow-up non-near probe artifacts:

- `docs/benchmarks/2026-02-15-logit-parity-probe-nonnear5-post-maskmin.json`
- `docs/benchmarks/2026-02-15-logit-parity-probe-nonnear5-post-maskmin.md`
- `docs/benchmarks/2026-02-15-logit-parity-probe-nonnear5-post-maskmin-compare.json`

Observed:
- mismatch probe fields unchanged on this subset (`5/5` still non-near).

Interpretation:
- this fixes an upstream semantic mismatch and is the safer mask behavior, but
  it is not the dominant cause of the current residual non-near parity gap.

### 47) Bounded non-near parity matrix (hard stop gate)

Added reproducible bounded matrix runner:

- `scripts/eval_nonnear_parity_matrix.py`

Artifacts:

- `docs/benchmarks/2026-02-15-nonnear-parity-matrix.json`
- `docs/benchmarks/2026-02-15-nonnear-parity-matrix.md`

Case set (fixed, intentionally small):
- baseline-f16
- baseline-f32
- force-dense-f16
- force-segmented-f16
- force-attn-fallback-f16

Result on non-near subset (`n=5`):
- all cases remained `0/5` token parity,
- mismatch rows unchanged (`5`),
- mean first mismatch index unchanged (`28.6`).

Decision:
- no improvement under this bounded matrix, so exploratory parity tuning stops
  here for this phase; preserve current implementation and carry forward the
  documented residual gap for future targeted work.
