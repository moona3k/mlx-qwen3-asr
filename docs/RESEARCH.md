# Research: Qwen3-ASR

Verified notes for this repo, based on official artifacts as of **2026-02-14**.

Latest academic best-practice cross-check:
- `docs/archive/ACADEMIC_BEST_PRACTICES_2026-02-15.md` (archived)

## Primary Sources

- Qwen3-ASR paper: https://arxiv.org/abs/2601.21337
- Official codebase: https://github.com/QwenLM/Qwen3-ASR
- Official Python package (`qwen-asr`): https://github.com/QwenLM/Qwen3-ASR/tree/main/qwen_asr
- Apple MLX examples: https://github.com/ml-explore/mlx-examples
- Apple MLX-LM cache implementation: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py
- MLX-Audio (existing community MLX Qwen3-ASR path): https://github.com/Blaizzy/mlx-audio
- Swift MLX implementation: https://github.com/ivan-digital/qwen3-asr-swift
- HF model cards/configs:
  - https://huggingface.co/Qwen/Qwen3-ASR-1.7B
  - https://huggingface.co/Qwen/Qwen3-ASR-0.6B
  - https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B

## Model Family Snapshot

| Model | Core Positioning | Notable Claim (official) |
|---|---|---|
| Qwen3-ASR-1.7B | Accuracy-first ASR | SOTA among open-source ASR models |
| Qwen3-ASR-0.6B | Efficiency-first ASR | 92 ms TTFT, 2000 s/s at concurrency 128 |
| Qwen3-ForcedAligner-0.6B | NAR forced alignment | 11-language alignment, 80 ms segment resolution |

## Config Facts (from HF `config.json`)

### Qwen3-ASR-1.7B
- Audio: `encoder_layers=24`, `encoder_attention_heads=16`, `d_model=1024`, `output_dim=2048`
- Text: `hidden_size=2048`, `num_hidden_layers=28`, GQA `num_attention_heads=16`, `num_key_value_heads=8`
- Audio chunk params: `n_window=50`, `n_window_infer=800`, `conv_chunksize=500`

### Qwen3-ASR-0.6B
- Audio: `encoder_layers=18`, `encoder_attention_heads=14`, `d_model=896`, `output_dim=1024`
- Text: `hidden_size=1024`, `num_hidden_layers=28`, GQA `16/8`
- Audio chunk params: `n_window=50`, `n_window_infer=800`, `conv_chunksize=500`

### Qwen3-ForcedAligner-0.6B
- Top-level extras: `timestamp_token_id=151705`, `timestamp_segment_time=80`, `classify_num=5000`
- Audio: `encoder_layers=24`, `encoder_attention_heads=16`, `d_model=1024`, `output_dim=1024`
- Text: `hidden_size=1024`, `num_hidden_layers=28`, GQA `16/8`

## Encoder Length Formula (Critical for Correctness)

Official upstream uses per-100-frame chunk logic:

```text
input_lengths_leave = input_lengths % 100
feat_lengths = (input_lengths_leave - 1) // 2 + 1
output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
```

This means exact multiples of 100 frames map to `13 * n_chunks` tokens
(e.g., 400 -> 52, 1000 -> 130), not simple full-sequence 8x downsampling.

## Benchmarks and Claims (from paper/model cards)

- 1.7B reports LibriSpeech test-clean WER 1.51 and WenetSpeech test-net CER 4.97.
- 0.6B reports LibriSpeech test-clean WER 1.72 and WenetSpeech test-net CER 5.38.
- Forced aligner reports 42.9 ms average alignment shift (AAS), outperforming listed baselines.

### Snapshot Tables (for quick reference)

| Model | LibriSpeech clean (WER) | WenetSpeech test-net (CER) |
|---|---:|---:|
| Qwen3-ASR-1.7B | 1.51 | 4.97 |
| Qwen3-ASR-0.6B | 1.72 | 5.38 |

| Forced alignment model | AAS (ms) |
|---|---:|
| Qwen3-ForcedAligner-0.6B | 42.9 |
| NeMo Forced Aligner | 129.8 |
| WhisperX | 133.2 |

### Language Coverage Context

- 30 core languages: Chinese, English, Cantonese, Arabic, German, French,
  Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai,
  Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish,
  Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian,
  Macedonian.
- 22 Chinese dialects per official model card/repo docs.
- Official release highlights support for singing voice recognition as well.

## Paper Metadata

- Title: *Qwen3-ASR Technical Report*
- arXiv: https://arxiv.org/abs/2601.21337
- Submitted: **January 29, 2026**; revised version (v2): **January 30, 2026**

## Mac/MLX Implementation Landscape (2026-02-14)

- `Blaizzy/mlx-audio` includes a Qwen3-ASR path and remains a visible
  community MLX baseline.
- `ivan-digital/qwen3-asr-swift` is a native Swift+MLX ASR/TTS implementation
  with packaged quantized ASR models and published ASR latency claims.
  - Repo: https://github.com/ivan-digital/qwen3-asr-swift
  - Evidence: `Sources/Qwen3ASR/Qwen3ASR.swift` exposes `transcribe(...) -> String`
- `Flovflo/VoiceScribe` is a native macOS dictation app built on Swift+MLX
  Qwen3-ASR models.
  - Repo: https://github.com/Flovflo/VoiceScribe
  - Evidence: `Sources/VoiceScribeCore/ML/Models/Qwen3ASR.swift`
- `Jason-Queen/Qwen3-ASR-MLX-Server` is a server wrapper over `mlx-audio`, not a
  standalone model-core reimplementation.
  - Repo: https://github.com/Jason-Queen/Qwen3-ASR-MLX-Server
  - Evidence: imports `mlx_audio.stt.*` in `whisper_mlx_server.py`
- `predict-woo/qwen3-asr.cpp` is a notable non-MLX native implementation that
  includes a dedicated forced aligner in C++/GGML.
  - Repo: https://github.com/predict-woo/qwen3-asr.cpp
  - Evidence: `src/forced_aligner.cpp`, `src/forced_aligner.h`

## Forced Aligner and Timestamp Facts

- Official Qwen timestamping is a separate model:
  `Qwen/Qwen3-ForcedAligner-0.6B`.
- In official `qwen-asr`, `return_time_stamps=True` requires initializing a
  `forced_aligner`; otherwise it raises an explicit error.
- Official streaming path does **not** support timestamps in that stack.

## Swift Timestamp Status (Evidence-Based)

- In `ivan-digital/qwen3-asr-swift`, ASR docs and source currently show
  transcription support but no ASR timestamp/forced-aligner implementation in
  the public ASR module surface (`transcribe(...) -> String`).
- The repo roadmap lists ASR streaming as pending; no published native forced
  alignment API is present in its ASR modules/docs today.
- `Flovflo/VoiceScribe` focuses on dictation transcription flow; no dedicated
  forced-aligner/timestamp module is exposed in the public repo structure.

## Forced Aligner Direction Implication

- A native timestamp path is technically feasible (evidence: C++ `qwen3-asr.cpp`
  already ships a dedicated aligner implementation), but there is no widely
  adopted Apple-MLX-native forced-aligner reference implementation to reuse
  directly today.
- Practical takeaway for this repo:
  1. Keep optional `qwen-asr` backend for immediate user value.
  2. Build native MLX aligner behind strict quality gates (word timing quality
     and parity checks) before making it default.

## Practical MLX Optimization Guidance from Upstream Code

- `mlx-lm`'s `KVCache` uses in-place slice writes and stepped growth strategy;
  this is the proven baseline pattern for decode-time cache efficiency.
- Quantized model loading in MLX stacks generally applies module quantization
  before weight loading and persists quantization metadata in config.
- For this repo, measured wins continue to come from:
  tokenizer/model caching, preallocated KV paths, and validated quant profiles
  (`4-bit`, `group_size=64`) rather than speculative asynchronous decode logic.

## Source Audit: Mel + Streaming (2026-02-14)

This section records the exact primary sources used for current mel/streaming
architecture decisions.

### A) Official Qwen streaming semantics

- File: `qwen_asr/inference/qwen3_asr.py`
  - `init_streaming_state(...)` + `streaming_transcribe(...)` + `finish_streaming_transcribe(...)`.
  - Behavior: streaming re-feeds accumulated audio and uses rollback controls
    (`unfixed_chunk_num`, `unfixed_token_num`), with vLLM-only support.
- Source:
  - https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py

### B) Whisper feature extraction reference behavior

- File: `transformers/models/whisper/feature_extraction_whisper.py`
  - `_np_extract_fbank_features(...)` drops final STFT frame via `log_spec = log_spec[:, :-1]`.
  - This explains common +1 frame mismatches in naive custom mel paths.
- Source:
  - https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/feature_extraction_whisper.py

### C) Swift implementation landscape check

- Repo: `ivan-digital/qwen3-asr-swift`
  - Public ASR surface is offline `transcribe(...)`; no dedicated ASR streaming
    module found in `Sources/Qwen3ASR`.
- Source:
  - https://github.com/ivan-digital/qwen3-asr-swift/tree/main/Sources/Qwen3ASR

### D) MLX cache design references

- Repo: `ml-explore/mlx-lm`
  - Multiple model implementations rely on explicit cache offset tracking and
    `update_and_fetch` patterns for incremental decode.
- Source:
  - https://github.com/ml-explore/mlx-lm

Decision implication for this repo:
- Do not switch to custom mel by default until strict parity gates pass.
- Keep streaming marked experimental until a true resumable decode-cache path lands.

## External Audit Refresh (GitHub + Papers, 2026-02-14)

This pass re-checked upstream repos and paper artifacts to confirm roadmap priority.

### 1) Official streaming is still constrained upstream

- In official `qwen-asr`, streaming state is documented as **vLLM-only** and
  **no timestamps**:
  - https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py#L596-L599
- The streaming implementation explicitly re-feeds all accumulated audio each chunk:
  - https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py#L670-L673
- Timestamping in official stack is a separate forced aligner model and code path:
  - https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_forced_aligner.py#L309
  - https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_forced_aligner.py#L394

### 2) Swift ecosystem status on ASR timestamps/streaming

- `ivan-digital/qwen3-asr-swift` public ASR API currently exposes
  `transcribe(...) -> String` (no timestamp object surface in this API):
  - https://github.com/ivan-digital/qwen3-asr-swift/blob/main/Sources/Qwen3ASR/Qwen3ASR.swift#L52-L57
- Its README roadmap still lists **ASR streaming inference** as pending:
  - https://github.com/ivan-digital/qwen3-asr-swift/blob/main/README.md#roadmap
- Existing ASR speed claims in that repo are strong and useful as external target points:
  - https://github.com/ivan-digital/qwen3-asr-swift/blob/main/README.md#latency-m2-max-64-gb

### 3) Other GitHub landscape notes

- `Jason-Queen/Qwen3-ASR-MLX-Server` is service glue over `mlx-audio`, not a clean
  standalone model-core implementation:
  - https://github.com/Jason-Queen/Qwen3-ASR-MLX-Server/blob/main/whisper_mlx_server.py#L33-L34
- `ontypehq/mlx-swift-asr` is another native Swift effort with strong runtime focus
  (useful for implementation ideas and benchmark comparison):
  - https://github.com/ontypehq/mlx-swift-asr

### 4) Model artifact availability (HuggingFace)

- Official models:
  - https://huggingface.co/Qwen/Qwen3-ASR-1.7B
  - https://huggingface.co/Qwen/Qwen3-ASR-0.6B
  - https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B
- MLX-community quantized variants for both ASR and forced aligner already exist
  (4/5/6/8-bit variants), which reduces packaging risk for Mac-native workflows:
  - https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-4bit
  - https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit
  - https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-4bit
  - https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-8bit

### 5) Academic and MLX optimization references

- Qwen3-ASR technical report (primary paper source):
  - https://arxiv.org/abs/2601.21337
- MLX official optimization docs (compile/lazy evaluation/streams), relevant to
  stable hot-path design and benchmarking discipline:
  - https://ml-explore.github.io/mlx/build/html/usage/compile.html
  - https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html
  - https://ml-explore.github.io/mlx/build/html/usage/using_streams.html

### 6) MLX fused-attention and compile caveats (follow-up audit)

- MLX maintainer guidance indicates `mx.fast.scaled_dot_product_attention(...)`
  can regress if mask dtype does not match the attention tensor dtype.
  - Source: https://github.com/ml-explore/mlx/issues/1193
- MLX maintainer guidance also indicates `mx.compile` is not universally a speed
  win and can regress workloads with highly dynamic shapes.
  - Source: https://github.com/ml-explore/mlx/issues/1248

Implication for this repo:
- Keep additive attention masks dtype-aligned with active tensor dtype in both
  encoder and decoder paths.
- Prefer targeted compile usage only where shape stability is proven by
  benchmark, rather than enabling compile broadly in the decode loop.

### Practical Implication

- Near-term highest ROI remains: correctness parity, native aligner quality gates,
  quantized artifact quality, and decode-path cleanliness.
- Streaming should remain explicit experimental scope, not a production-critical
  investment lane until core offline quality and performance goals are saturated.

### Source snapshot commits used in this audit

- `QwenLM/Qwen3-ASR`: `c17a131`
- `ivan-digital/qwen3-asr-swift`: `7dd6029`

## Encoder Long-Context Optimization (2026-02-14)

We validated a hybrid encoder-window strategy for long audio:

- For smaller window counts, keep dense block-mask execution.
- For larger window counts, switch to segmented per-window execution, which is
  mathematically equivalent to block-diagonal masking but avoids dense `(L, L)` mask
  overhead.

Threshold decision:

- Use segmented execution when `num_windows >= 20`.

Benchmark artifact (synthetic, same weights/inputs, 8-layer encoder):

- `docs/benchmarks/2026-02-14-encoder-windowing-threshold.json`
- `docs/benchmarks/2026-02-14-encoder-windowing-threshold.md`

Observed:

- small context (`8 windows`): dense path is faster,
- crossover around `16-20 windows`,
- long context (`80 windows`): segmented path showed ~`4.17x` speedup with
  max-abs output diff on the order of `1e-6`.

See also:
- `docs/archive/ALGORITHMIC_MAXXING_2026-02-14.md` (archived) for paper-backed next-step algorithm candidates.

## Algorithmic Literature Refresh (2026-02-14)

Recent ASR speculative-decoding papers with practical relevance:

- SpecASR (arXiv:2507.18181): task-specialized speculative decoding for LLM-based ASR
  with reported `3.04x-3.79x` speedup over vanilla AR decode (paper claim).
- Token Map Drafting (arXiv:2507.21522): model-free speculative drafting for
  transformer ASR with reported `1.27x-1.37x` speedup in structured domains.
- Distil-Whisper (arXiv:2311.00430): high-quality distillation baseline for
  reduced-latency ASR deployments.

Implication for this repo:

- Highest-value next algorithmic path is still a strict-parity speculative decode
  experiment using `0.6B` as draft for `1.7B` with release-gate parity checks.
- Model-free/token-map drafting is promising for domain-constrained workloads,
  but not the default general-purpose path yet.
- Current local prototype status (implemented):
  - greedy text parity preserved,
  - tested short and 10s runs showed latency regressions, so it remains opt-in.

### Classic algorithm pass (old-paper gems applied)

For monotonic timestamp repair in forced alignment, we also reviewed classic
sequence algorithms and applied one directly:

- LIS complexity theory (Fredman 1975):
  - https://doi.org/10.1016/0012-365X(75)90103-X
- Fenwick tree (Fenwick 1994), enabling efficient prefix-best queries:
  - https://doi.org/10.1002/spe.4380240306

Applied change in this repo:
- Replaced O(n^2) LIS DP in `ForcedAlignTextProcessor.fix_timestamp(...)` with
  O(n log n) Fenwick-based LIS while preserving legacy tie semantics exactly.
- Added randomized parity tests against the old O(n^2) reference implementation
  so alignment behavior remains stable while complexity drops.
