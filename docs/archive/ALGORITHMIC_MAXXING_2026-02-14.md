# Algorithmic Maxxing Review (2026-02-14)

Focused review of paper-backed inference algorithms that can materially improve
`mlx-qwen3-asr` quality/speed without over-engineering.

## Sources Reviewed

- Qwen3-ASR Technical Report: https://arxiv.org/abs/2601.21337
- Official Qwen3-ASR repo/code: https://github.com/QwenLM/Qwen3-ASR
- Whisper paper (robust long-form ASR context): https://arxiv.org/abs/2212.04356
- WhisperX (long-form + alignment pipeline): https://arxiv.org/abs/2303.00747
- Speculative decoding (lossless acceleration): https://arxiv.org/abs/2211.17192
- SpecASR (ASR-specialized speculative decoding): https://arxiv.org/abs/2507.18181
- Token Map Drafting (model-free speculative ASR): https://arxiv.org/abs/2507.21522
- Distil-Whisper (distilled ASR baseline): https://arxiv.org/abs/2311.00430
- MLX compile docs (implementation guidance): https://ml-explore.github.io/mlx/build/html/usage/compile.html

## Changes Applied This Session

1. Long-context encoder hybrid execution
- Added segmented per-window encoder execution for large window counts while
  preserving dense-mask path for shorter contexts.
- Benchmark-backed threshold: `num_windows >= 20`.
- Result: avoids long-context dense-mask blow-up while preserving short-input
  latency behavior.

2. Upstream-style repetition cleanup in output parsing
- Added repetition post-processing aligned with official Qwen utilities:
  - long single-character run collapse,
  - long repeated-pattern collapse.
- Upstream reference:
  - `detect_and_fix_repetitions(...)` in
    `qwen_asr/inference/utils.py`:
    https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/utils.py
- Added edge-case handling for:
  - `language None<asr_text>` empty-audio case,
  - forced language parse path,
  - consistent special-token stripping.

3. Multi-model residency cache for speculative path
- `_ModelHolder` now caches by `(path, dtype)` instead of a single global slot.
- This removes repeated target/draft reload churn across repeated speculative
  calls and is a required foundation for any serious speculative benchmarking.

4. Forced-aligner timestamp repair complexity drop (legacy behavior preserved)
- Replaced `fix_timestamp(...)` LIS core from O(n^2) DP to O(n log n) using
  coordinate compression + Fenwick tree.
- Critical constraint: tie semantics remain identical to legacy behavior:
  - predecessor choice stays "earliest index with best length",
  - chosen LIS endpoint stays earliest index at global max length.
- Why this matters:
  - avoids quadratic blow-up in long timestamp sequences,
  - preserves existing alignment outputs (no behavior drift).
- Validation:
  - new randomized parity test compares LIS indices against legacy O(n^2)
    reference implementation across hundreds of random inputs per length bucket.

Related algorithmic references:
- Fredman 1975 (LIS complexity lower/upper bounds):
  https://doi.org/10.1016/0012-365X(75)90103-X
- Fenwick 1994 (binary indexed tree data structure):
  https://doi.org/10.1002/spe.4380240306

## High-Value Next Algorithms (Paper-Backed)

1. Speculative decoding for 1.7B using 0.6B draft model
- Why: lossless decoding acceleration in autoregressive models (Leviathan et al.).
- Fit here: Qwen3-ASR decode is autoregressive and already has explicit cache
  interfaces (`prefill/step`) that make this implementable.
- Practical gate:
  - token-for-token parity with baseline greedy decode,
  - wall-time speedup on 1.7B short/long clips.

2. VAD + overlap merge for long-form offline transcription
- Why: Whisper/WhisperX long-form robustness is improved by segmentation around
  speech boundaries and alignment-aware merging.
- Fit here: complements current chunker for difficult acoustic conditions.
- Practical gate:
  - no WER/CER regression on existing lane,
  - improved stability on long noisy recordings.

3. Confidence-aware decode safeguards
- Why: ASR hallucination/repetition tail issues are often controllable with
  decode-time confidence heuristics and bounded retries.
- Fit here: can be implemented as optional conservative mode for enterprise use.
- Practical gate:
- reduces pathological repetition incidents,
- no measurable average quality regression.

4. ASR-specialized speculative decoding variants
- Why: 2025 papers show stronger ASR-specific speedups than generic speculative
  decoding baselines in some settings.
- Fit here:
  - `SpecASR` ideas map onto our existing `prefill/step` API and cache ownership.
  - Token-map drafting may be useful for enterprise domain-specific deployments.
- Practical gate:
- must preserve exact greedy token parity vs baseline on release lane,
- must beat current 1.7B latency on short and 10s clips.

5. Isotonic regression (PAVA) for monotonic timestamp projection (deferred)
- Why: PAVA gives an O(n) projection onto monotonic sequences and can minimize
  squared error against raw timestamp logits.
- Reference lineage:
  - Ayer et al. 1955 (classic isotonic formulation):
    https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-26/issue-4/An-Empirical-Distribution-Function-for-Sampling-with-Incomplete-Information/10.1214/aoms/1177728423.full
- Why deferred:
  - current production behavior is intentionally aligned to legacy LIS repair;
    changing objective from LIS-anchor interpolation to isotonic projection is
    likely to shift timestamps and would require a new parity-quality gate.

Status update:
- Prototype path implemented with strict parity checks.
- Current benchmark evidence on tested short/10s workloads shows no speed win yet,
  so it remains experimental.

Final review note:
- Research was refreshed against recent ASR speculative papers. None provided a
  drop-in change that beats our current baseline without adding substantial
  complexity/risk. The correct next step is stronger throughput evaluation on
  larger decode-heavy workloads before deeper speculative algorithm work.

## Not Prioritized (Current Policy)

- Production streaming architecture work remains explicitly deprioritized until
  offline quality/speed/timestamp goals are fully saturated.
