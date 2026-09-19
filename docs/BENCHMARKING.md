# Benchmarking Protocol

Use this protocol for all runtime optimization work.

## Goal

Hold quality constant while improving runtime on Apple Silicon.

## Command

```bash
python scripts/benchmark_asr.py tests/fixtures/test_speech.wav \
  --model Qwen/Qwen3-ASR-0.6B \
  --dtype float16 \
  --warmup-runs 1 \
  --runs 5 \
  --json-output docs/benchmarks/latest.json
```

Speculative baseline-vs-draft benchmark:

```bash
python scripts/benchmark_speculative.py tests/fixtures/test_speech.wav \
  --model Qwen/Qwen3-ASR-1.7B \
  --draft-model Qwen/Qwen3-ASR-0.6B \
  --num-draft-tokens 4 \
  --runs 3 \
  --json-output docs/benchmarks/speculative-latest.json
```

Streaming benchmark:

```bash
python scripts/benchmark_streaming.py tests/fixtures/test_speech.wav \
  --model Qwen/Qwen3-ASR-0.6B \
  --chunk-size-sec 2.0 \
  --max-context-sec 30.0 \
  --runs 3 \
  --json-output docs/benchmarks/latest-streaming.json

# finalization_mode is accepted for compatibility; since 0.4.2 both modes decode
# the pending tail with the window re-decode
python scripts/benchmark_streaming.py tests/fixtures/test_speech.wav \
  --model Qwen/Qwen3-ASR-0.6B \
  --chunk-size-sec 2.0 \
  --max-context-sec 30.0 \
  --finalization-mode latency \
  --runs 3

# Speech-aware endpointing near chunk boundaries
python scripts/benchmark_streaming.py tests/fixtures/test_speech.wav \
  --model Qwen/Qwen3-ASR-0.6B \
  --chunk-size-sec 2.0 \
  --max-context-sec 30.0 \
  --endpointing-mode energy \
  --runs 3

# Streaming quality diagnostics (stability + rewrite + finalization delta)
python scripts/eval_streaming_metrics.py tests/fixtures/test_speech.wav \
  --model Qwen/Qwen3-ASR-0.6B \
  --chunk-size-sec 2.0 \
  --max-context-sec 30.0 \
  --endpointing-mode energy \
  --json-output docs/benchmarks/latest-streaming-quality.json
```

Latest streaming baseline artifact:
- `docs/benchmarks/2026-02-14-streaming-rolling-baseline.json`
- `docs/benchmarks/2026-02-14-streaming-rolling-baseline.md`

Streaming benchmark payload now includes `streaming_quality`:
- `partial_stability_mean`
- `rewrite_rate_mean`
- `finalization_delta_chars_mean` / `finalization_delta_chars_max`

Reference parity suite benchmark:

```bash
python scripts/eval_reference_parity_suite.py \
  --model Qwen/Qwen3-ASR-0.6B \
  --subsets test-clean,test-other \
  --samples-per-subset 5 \
  --include-long-mixes \
  --include-noise-variants \
  --noise-snrs-db 10,5 \
  --long-mixes 2 \
  --json-output docs/benchmarks/reference-parity-suite.json
```

The suite reports:
- `token_match_rate` (strict token-for-token parity),
- `text_match_rate` (Unicode-safe normalized text parity).

English quality benchmark lanes:

```bash
python scripts/eval_librispeech.py \
  --model Qwen/Qwen3-ASR-0.6B \
  --subset test-clean \
  --samples 100 \
  --sampling speaker_round_robin \
  --json-output docs/benchmarks/2026-02-15-librispeech-test-clean-100.json

python scripts/eval_librispeech.py \
  --model Qwen/Qwen3-ASR-0.6B \
  --subset test-other \
  --samples 100 \
  --sampling speaker_round_robin \
  --json-output docs/benchmarks/2026-02-15-librispeech-test-other-100.json

# Optional backend comparison on the same evaluated rows
python scripts/eval_librispeech_head2head.py \
  --mlx-json docs/benchmarks/2026-02-15-librispeech-test-other-100.json \
  --model Qwen/Qwen3-ASR-0.6B \
  --language English \
  --json-output docs/benchmarks/2026-02-15-quality-head2head-mlx-vs-pytorch-test-other100.json \
  --md-output docs/benchmarks/2026-02-15-quality-head2head-mlx-vs-pytorch-test-other100.md
```

Multilingual manifest parity benchmark:

```bash
python scripts/build_multilingual_manifest.py \
  --languages en_us,zh_cn,ja_jp,de_de,fr_fr,es_419,ru_ru,ar_eg,hi_in,ko_kr \
  --samples-per-language 10 \
  --output-manifest docs/benchmarks/fleurs-multilingual-manifest.jsonl

python scripts/eval_reference_parity_suite.py \
  --subsets '' \
  --samples-per-subset 1 \
  --manifest-jsonl docs/benchmarks/fleurs-multilingual-manifest.jsonl \
  --json-output docs/benchmarks/reference-parity-suite-multilingual.json

python scripts/analyze_reference_parity_mismatches.py \
  --input-json docs/benchmarks/reference-parity-suite-multilingual.json \
  --json-output docs/benchmarks/reference-parity-suite-multilingual-analysis.json \
  --md-output docs/benchmarks/reference-parity-suite-multilingual-analysis.md
```

Manifest quality head-to-head benchmark (MLX artifact vs PyTorch `qwen_asr`):

```bash
python scripts/eval_manifest_head2head.py \
  --mlx-json docs/benchmarks/2026-02-15-manifest-quality-longform10.json \
  --model Qwen/Qwen3-ASR-0.6B \
  --max-new-tokens 1024 \
  --progress-every 1 \
  --checkpoint-json /tmp/longform10-head2head-checkpoint.json \
  --checkpoint-every 1 \
  --json-output docs/benchmarks/2026-02-15-quality-head2head-mlx-vs-pytorch-longform10.json \
  --md-output docs/benchmarks/2026-02-15-quality-head2head-mlx-vs-pytorch-longform10.md
```

Real-world manifest lane (AMI + Earnings22 chunked, deterministic speaker-balanced sample):

```bash
# Expanded lane (recommended): n=200
python scripts/build_realworld_manifest.py \
  --sources ami_ihm_test,earnings22_chunked_test \
  --samples-per-source 100 \
  --seed 20260216 \
  --candidate-multiplier 10 \
  --min-unique-speakers 50 \
  --max-shards 40 \
  --output-manifest docs/benchmarks/2026-02-15-realworld-manifest-200.jsonl

python scripts/eval_manifest_quality.py \
  --manifest-jsonl docs/benchmarks/2026-02-15-realworld-manifest-200.jsonl \
  --model Qwen/Qwen3-ASR-0.6B \
  --dtype float16 \
  --json-output docs/benchmarks/2026-02-15-manifest-quality-realworld200-0p6b.json

python scripts/eval_manifest_head2head.py \
  --mlx-json docs/benchmarks/2026-02-15-manifest-quality-realworld200-0p6b.json \
  --model Qwen/Qwen3-ASR-0.6B \
  --max-new-tokens 1024 \
  --progress-every 5 \
  --checkpoint-json /tmp/realworld200-head2head-checkpoint.json \
  --checkpoint-every 10 \
  --json-output docs/benchmarks/2026-02-15-quality-head2head-mlx-vs-pytorch-realworld200.json \
  --md-output docs/benchmarks/2026-02-15-quality-head2head-mlx-vs-pytorch-realworld200.md

# Baseline smoke lane (fast): n=40
python scripts/build_realworld_manifest.py \
  --sources ami_ihm_test,earnings22_chunked_test \
  --samples-per-source 20 \
  --seed 20260215 \
  --candidate-multiplier 8 \
  --min-unique-speakers 10 \
  --max-shards 20 \
  --output-manifest docs/benchmarks/2026-02-15-realworld-manifest-40.jsonl
```

Latest expanded real-world artifacts:
- `docs/benchmarks/2026-02-15-realworld-manifest-200.jsonl`
- `docs/benchmarks/2026-02-15-manifest-quality-realworld200-0p6b.json`
- `docs/benchmarks/2026-02-15-manifest-quality-realworld200-0p6b.md`
- `docs/benchmarks/2026-02-15-quality-head2head-mlx-vs-pytorch-realworld200.json`
- `docs/benchmarks/2026-02-15-quality-head2head-mlx-vs-pytorch-realworld200.md`

Real-world non-synthetic long-form lane (Earnings22 full recordings):

```bash
python scripts/build_earnings22_longform_manifest.py \
  --samples 3 \
  --seed 20260216 \
  --min-duration-sec 850 \
  --max-duration-sec 1400 \
  --output-manifest docs/benchmarks/2026-02-15-earnings22-full-longform3-manifest.jsonl

python scripts/eval_manifest_quality.py \
  --manifest-jsonl docs/benchmarks/2026-02-15-earnings22-full-longform3-manifest.jsonl \
  --model Qwen/Qwen3-ASR-0.6B \
  --dtype float16 \
  --max-new-tokens 8192 \
  --json-output docs/benchmarks/2026-02-15-manifest-quality-earnings22-full-longform3-0p6b.json

python scripts/eval_manifest_head2head.py \
  --mlx-json docs/benchmarks/2026-02-15-manifest-quality-earnings22-full-longform3-0p6b.json \
  --model Qwen/Qwen3-ASR-0.6B \
  --max-new-tokens 8192 \
  --progress-every 1 \
  --checkpoint-json /tmp/earnings22-full-longform3-head2head-checkpoint.json \
  --checkpoint-every 1 \
  --json-output docs/benchmarks/2026-02-15-quality-head2head-mlx-vs-pytorch-earnings22-full-longform3.json \
  --md-output docs/benchmarks/2026-02-15-quality-head2head-mlx-vs-pytorch-earnings22-full-longform3.md
```

Latest multilingual smoke artifacts:
- `docs/benchmarks/2026-02-14-fleurs-multilingual-smoke-manifest.jsonl`
- `docs/benchmarks/2026-02-14-reference-parity-suite-multilingual-smoke.json`
- `docs/benchmarks/2026-02-14-reference-parity-suite-multilingual-smoke.md`
- `docs/benchmarks/2026-02-14-reference-parity-suite-multilingual-smoke-analysis.json`
- `docs/benchmarks/2026-02-14-reference-parity-suite-multilingual-smoke-analysis.md`

Latest multilingual expanded artifacts:
- `docs/benchmarks/2026-02-14-fleurs-multilingual-20-manifest.jsonl`
- `docs/benchmarks/2026-02-14-reference-parity-suite-multilingual-20.json`
- `docs/benchmarks/2026-02-14-reference-parity-suite-multilingual-20-analysis.json`
- `docs/benchmarks/2026-02-14-reference-parity-suite-multilingual-20-analysis.md`
- `docs/benchmarks/2026-02-14-fleurs-multilingual-100-manifest.jsonl`
- `docs/benchmarks/2026-02-14-reference-parity-suite-multilingual-100.json`
- `docs/benchmarks/2026-02-14-reference-parity-suite-multilingual-100-analysis.json`
- `docs/benchmarks/2026-02-14-reference-parity-suite-multilingual-100-analysis.md`

Quality and latency refresh for v0.4.0 (2026-09-07, Apple M4 Pro):
- `docs/benchmarks/2026-09-07-quality-matrix-refresh.md` (summary + commands)
- `docs/benchmarks/2026-09-07-librispeech-test-{clean,other}-100.json`
- `docs/benchmarks/2026-09-07-librispeech-test-{clean,other}-100-1p7b.json`
- `docs/benchmarks/2026-09-07-quant-matrix-test-{clean,other}-speaker100.{json,md}`
- `docs/benchmarks/2026-09-07-latency-{fp16,8bit-g64,4bit-g64,1p7b-fp16}-{short,10s}.json`
- `docs/benchmarks/2026-09-07-fleurs-multilingual-100-manifest.jsonl`
  (same 100 samples as the 2026-02-14 manifest, rebuilt with the same seed)
- `docs/benchmarks/2026-09-07-manifest-quality-multilingual100-{0p6b,1p7b}.json`
- `docs/benchmarks/2026-09-07-fleurs-longform-10x75-manifest.jsonl`
  (new deterministic long-form set derived from the manifest above)
- `docs/benchmarks/2026-09-07-manifest-quality-longform10-0p6b.json`

Previous quality-matrix refresh artifacts (2026-02-15):
- `docs/benchmarks/2026-02-15-librispeech-test-clean-100.json`
- `docs/benchmarks/2026-02-15-librispeech-test-other-100.json`
- `docs/benchmarks/2026-02-15-manifest-quality-multilingual100-0p6b-refresh.json`
- `docs/benchmarks/2026-02-15-manifest-quality-multilingual100-1p7b-refresh.json`
- `docs/benchmarks/2026-02-15-quality-matrix-refresh.json`
- `docs/benchmarks/2026-02-15-quality-matrix-refresh.md`

Latest smoke artifacts:
- `docs/benchmarks/2026-02-14-reference-parity-suite-smoke.json`
- `docs/benchmarks/2026-02-14-reference-parity-suite-smoke.md`
- `docs/benchmarks/2026-02-14-reference-parity-suite-smoke-v2.json`
- `docs/benchmarks/2026-02-14-reference-parity-suite-smoke-v2.md`
- `docs/benchmarks/2026-02-14-reference-parity-suite-english22.json`
- `docs/benchmarks/2026-02-14-reference-parity-suite-english22.md`

Mel parity evaluation:

```bash
# Optional dependency for HF reference lane:
# pip install transformers
python scripts/eval_mel_parity.py \
  --json-output docs/benchmarks/latest-mel-parity.json
```

Latest mel parity artifacts:
- `docs/benchmarks/2026-02-14-mel-parity.json`
- `docs/benchmarks/2026-02-14-mel-parity.md`

## Metrics

- `latency_sec.mean`
- `latency_sec.median`
- `rtf` (real-time factor = mean_latency / audio_duration)

Lower is better for all three.

## Method

1. Run benchmark on baseline commit.
2. Run benchmark on candidate commit with same machine/settings.
3. Compare JSON outputs.
4. If speed improves but quality gate fails, reject the optimization.

## Recommended Scenarios

- Short clip (latency sensitivity): `tests/fixtures/test_speech.wav`
- Long-form clip (stability/throughput): add a fixed long fixture and track same metrics.

## Reporting Template

Include this in PRs that claim performance gains:

```text
Machine: <chip + RAM + macOS>
Model: <id>, dtype=<dtype>
Before: mean=<x>s median=<y>s rtf=<z>
After:  mean=<x2>s median=<y2>s rtf=<z2>
Quality Gate: fast=<pass/fail>, release=<pass/fail or not run>
```

## Measurement history

The dated findings below are kept as a record of how each optimization was
validated at the time. Current numbers live in `docs/BENCHMARKS.md` and
`docs/benchmarks/2026-09-07-quality-matrix-refresh.md`; the February fp16
latencies are 2-3x higher than today's because fp16 inference was running in
float32 until v0.4.0, which also inflated the quantization speedup ratios.

## Local Finding (2026-02-14)

- Change: tokenizer instance caching across repeated `transcribe()` calls.
- Machine: Apple M4 Pro, macOS 26.2.
- Workload: `tests/fixtures/test_speech.wav`, model `Qwen/Qwen3-ASR-0.6B`, dtype `float16`, warmup=1, runs=3.
- Before: mean latency `1.7217s`, RTF `0.6796`.
- After: mean latency `0.5464s`, RTF `0.2157`.
- Delta: `-68.3%` mean latency, `-68.3%` RTF.

Raw JSON artifacts are tracked under `docs/benchmarks/`.

## Additional Local Finding (2026-02-14)

- Change set:
  - KV-cache preallocation (remove per-step concat growth cost),
  - direct grouped-query fused attention (remove explicit K/V head repetition).
- Machine: Apple M4 Pro, macOS 26.2.
- Model: `Qwen/Qwen3-ASR-0.6B`, dtype `float16`.
- Short clip (`tests/fixtures/test_speech.wav`): mean latency `0.5303s`, RTF `0.2093`.
- 10-second clip (`/tmp/test_10s.wav`): mean latency `0.9420s`, RTF `0.0942` (repeat run `0.9546s`, RTF `0.0955`).

## Quantized Comparison (2026-02-14)

Quantized runtime path is now validated end-to-end (local converted 4-bit model loads and transcribes).

- Machine: Apple M4 Pro, macOS 26.2.
- Model family: `Qwen/Qwen3-ASR-0.6B`.

Runtime:
- FP16 short fixture (`2.53s`): mean `0.4635s`, RTF `0.1830`.
- 4-bit (`g64`) short fixture (`2.53s`): mean `0.1275s`, RTF `0.0503`.
- 8-bit (`g64`) short fixture (`2.53s`): mean `0.1088s`, RTF `0.0429`.
- FP16 10s clip: mean `0.8341s`, RTF `0.0834`.
- 4-bit (`g64`) 10s clip: mean `0.1784s`, RTF `0.0178`.
- 8-bit (`g64`) 10s clip: mean `0.2682s`, RTF `0.0268`.

Quality (LibriSpeech `test-clean`, 100 utterances, deterministic `speaker_round_robin`):
- FP16: WER `0.022874`, CER `0.005865`, RTF `0.1270`.
- 8-bit (`g64`): WER `0.023306`, CER `0.005865`, RTF `0.0381`.
- 4-bit (`g64`): WER `0.027190`, CER `0.008798`, RTF `0.0315`.

Interpretation:
- 8-bit is near-fp16 on this sample (`+0.000432` absolute WER) with ~`3.19x` faster eval throughput.
- 4-bit is fastest for long clips (~`3.60x` eval throughput and `4.68x` long-latency speedup) but has measurable quality loss (`+0.004316` absolute WER).

## Quantization Matrix Sweep (2026-02-14)

A full sweep (`fp16`, `4bit-g64`, `4bit-g32`, `8bit-g64`) was run via:

```bash
python scripts/benchmark_quantization_matrix.py
```

Default evaluation lane now uses `test-clean`, `100` samples, and
`speaker_round_robin` sampling to reduce single-speaker/sample-order bias.

Artifacts:
- `docs/benchmarks/2026-02-14-quant-matrix.json`
- `docs/benchmarks/2026-02-14-quant-matrix.md`
- Refreshed run after WAV fast-path:
  - `docs/benchmarks/2026-02-14-quant-matrix-post-wavfast.json`
  - `docs/benchmarks/2026-02-14-quant-matrix-post-wavfast.md`
- Speaker-balanced refresh (`n=100`, `speaker_round_robin`):
  - `docs/benchmarks/2026-02-14-quant-matrix-speaker100.json`
  - `docs/benchmarks/2026-02-14-quant-matrix-speaker100.md`
- CI/manual workflow: `.github/workflows/quantization-matrix.yml`

Current recommended operating points for `Qwen/Qwen3-ASR-0.6B` on Apple Silicon:
- `4bit-g64` for speed-first workloads.
- `8bit-g64` for quality-sensitive quantized workloads.

Latest refreshed matrix highlights (`2026-02-14-quant-matrix-speaker100.md`):
- fp16 long 10s: mean `0.8341s`, RTF `0.0834`
- 4bit-g64 long 10s: mean `0.1784s`, RTF `0.0178`
- 4bit-g64 long speedup vs fp16: `4.68x`
- 8bit-g64 long speedup vs fp16: `3.11x`
- WER deltas should be interpreted with the larger speaker-balanced quality artifacts:
  - `docs/benchmarks/2026-02-14-librispeech-fp16-100-speaker-round-robin.json`
  - `docs/benchmarks/2026-02-14-librispeech-8bit-g64-100-speaker-round-robin.json`
  - `docs/benchmarks/2026-02-14-librispeech-4bit-g64-100-speaker-round-robin.json`

Harder-English quantization refresh (`test-other`, `n=100`, speaker-balanced):

```bash
python scripts/benchmark_quantization_matrix.py \
  --model Qwen/Qwen3-ASR-0.6B \
  --configs fp16,4:64,8:64 \
  --eval-subset test-other \
  --eval-samples 100 \
  --eval-sampling speaker_round_robin \
  --benchmark-runs 5 \
  --json-output docs/benchmarks/2026-02-15-quant-matrix-test-other-speaker100.json \
  --md-output docs/benchmarks/2026-02-15-quant-matrix-test-other-speaker100.md
```

Artifacts:
- `docs/benchmarks/2026-02-15-quant-matrix-test-other-speaker100.json`
- `docs/benchmarks/2026-02-15-quant-matrix-test-other-speaker100.md`

## KV Cache Write-Path Follow-up (2026-02-14)

`KVCache.update()` preallocated writes were compared:
- baseline: `mx.slice_update`
- candidate: in-place slice assignment

Artifacts:
- `docs/benchmarks/2026-02-14-kvcache-write-path.json`
- `docs/benchmarks/2026-02-14-kvcache-write-path.md`

Result summary:
- fp16 short/long and q4 short improved in mean latency.
- q4 long median stayed effectively unchanged; one outlier inflated mean.
- Decision: keep in-place write path (neutral-to-better overall).

## WAV Loader Fast-Path (2026-02-14)

Added a native WAV parser (PCM + IEEE float) with ffmpeg fallback for
unsupported/non-WAV formats. This removes ffmpeg process startup overhead for
common `.wav` inputs.

Artifacts:
- `docs/benchmarks/2026-02-14-wav-fastpath.json`
- `docs/benchmarks/2026-02-14-wav-fastpath.md`

Measured impact (same-session A/B):
- fp16 short: ~`-7.23%`
- fp16 long 10s: ~`+0.40%` (effectively neutral)
- 4bit-g64 short: ~`-24.78%`
- 4bit-g64 long 10s: ~`-9.35%`

Short-clip latency benefits are strongest for fast quantized profiles where
audio loading overhead is a larger fraction of total runtime.

## Rejected/Neutral Experiments (2026-02-14)

To avoid rediscovering low-signal paths, these were tested and not kept:

- Decode-loop `eval_interval` default changes (`0`, `4`):
  - No consistent net gain across fp16/q4 short+long scenarios.
  - Final decision: keep `eval_interval=1`.
- One-step-ahead async decode scheduling (mlx-lm style adaptation):
  - Regressed in end-to-end A/B on this repo's benchmark scenarios.
  - Final decision: reverted.
- `compute_features()` HF extractor fallback path:
  - Removed from runtime inference path.
  - Current implementation is native-only (`do_not_pad`, `max_length`, `longest`)
    with explicit validation and resample handling.
- Speculative decoding prototype (`1.7B` target + `0.6B` draft):
  - Kept parity (`text_match=true`) but regressed latency on tested short and 10s clips.
  - Status: experimental only; not enabled by default.
  - Artifacts:
    - `docs/benchmarks/2026-02-14-speculative-1p7b-vs-0p6b.json`
    - `docs/benchmarks/2026-02-14-speculative-1p7b-vs-0p6b-10s.json`
    - summary: `docs/benchmarks/2026-02-14-speculative-prototype.md`

## Forced Aligner Backend Smoke (2026-02-14)

Native MLX timestamp alignment was compared against `qwen_asr` on a short
deterministic fixture.

Artifacts:
- `docs/benchmarks/2026-02-14-aligner-backend-smoke.json`
- `docs/benchmarks/2026-02-14-aligner-backend-smoke.md`

Result snapshot (`tests/fixtures/test_speech.wav`, 5 warm runs):
- `mlx` mean: `0.0414s`
- `qwen_asr` mean: `0.1957s`
- Relative mean speed (`qwen_asr / mlx`): `4.73x`
- Both backends produced identical first-word timestamp spans in this smoke case.

Scope note:
- This is a single-sample sanity comparison, not yet a full multilingual
  timestamp quality parity verdict.

## Forced Aligner Parity Lane (2026-02-14)

A deterministic LibriSpeech subset parity run was added for native aligner
validation (`scripts/eval_aligner_parity.py`).

Artifacts:
- `docs/benchmarks/2026-02-14-aligner-parity-10.json`
- `docs/benchmarks/2026-02-14-aligner-parity-10.md`
- `docs/benchmarks/2026-02-14-aligner-parity-50.json`
- `docs/benchmarks/2026-02-14-aligner-parity-50.md`

Latest expanded snapshot (`test-clean`, 50 samples):
- text-match rate (`mlx` vs `qwen_asr` word sequence): `1.0000`
- timing MAE (all boundaries): `5.6909 ms`
- mean latency: `mlx=0.2113s`, `qwen_asr=0.5570s`
- relative speed (`qwen_asr / mlx`): `2.64x`

Gate integration:
- Optional release gate lane:
  - `RUN_ALIGNER_PARITY=1 ALIGNER_PARITY_SAMPLES=10`
  - thresholds: text-match `>= 1.0`, timing MAE `<= 60ms`.
