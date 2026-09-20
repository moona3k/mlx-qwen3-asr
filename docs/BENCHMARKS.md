# Benchmarks

Benchmark results for mlx-qwen3-asr on Apple Silicon. Every result has a
committed JSON artifact under `docs/benchmarks/` for reproducibility.

Two measurement generations are recorded here:

- **2026-09-07 (v0.4.0)**: English, multilingual, long-form, quantization and latency
  lanes, Apple M4 Pro (48 GB), macOS 26, MLX 0.30.6. Snapshot with commands:
  `docs/benchmarks/2026-09-07-quality-matrix-refresh.md`.
- **February 2026 (v0.2.x)**: real-world (AMI + Earnings22), aligner and mel
  parity, and the optimization studies further down.
  Those sections are labelled with their date. Their latencies predate the
  v0.4.0 float16 fix and are conservative by roughly 2-3x; their quality
  numbers still hold, since the dtype change produced identical hypotheses on
  the 100-sample LibriSpeech lane.

## Summary

| Metric | 0.6B fp16 | 0.6B 8-bit | 0.6B 4-bit | 1.7B fp16 |
|---|---|---|---|---|
| LibriSpeech test-clean WER | 2.33% | 2.33% | 2.59% | 1.94% |
| LibriSpeech test-other WER | 4.30% | 4.14% | 5.74% | 3.45% |
| Multilingual primary error (FLEURS 100) | 9.54% | — | — | 6.70% |
| Long-form primary error (FLEURS 10x80s) | 10.6% | — | — | — |
| Short clip latency (~2.5s) | 0.17s | 0.10s | 0.09s | 0.36s |
| 10s clip latency | 0.30s | 0.23s | 0.17s | 0.73s |
| Multilingual mean latency | 0.65s | — | — | 1.22s |
| Real-world200 WER (AMI+Earnings, Feb 2026) | 23.23% | — | — | — |
| MLX vs PyTorch primary error (multilingual-100) | 9.54% vs 10.34% | — | — | — |

### Hardware / MLX matrix

Every number in this file was produced on the machines below. Artifacts
written after 2026-09-19 carry a `runtime` block (`host_chip`, `memory_gb`,
`macos_version`, `python`, `mlx_version`, `git_commit`) from
`scripts/eval/provenance.py`, and the nightly lane uploads it with each run.

| Host | Memory | MLX | What was measured here |
|---|---:|---|---|
| Apple M4 Pro (maintainer laptop) | 48 GB | 0.30.6 (repo venv), 0.32.2 (temp venv) | everything |
| GitHub `macos-14` runner (M1, 7 GB) | 7 GB | latest release at run time | nightly LibriSpeech-100 + latency, PR fast gate |

Missing: any 8-16 GB Apple Silicon machine. To add a row, run
`scripts/eval_librispeech.py --samples 100 --sampling speaker_round_robin
--json-output ...` and `scripts/benchmark_asr.py tests/fixtures/test_speech.wav
--runs 5 --json-output ...`, commit both JSONs under `docs/benchmarks/` with the
host in the file name, and add the row.

---

## English Quality (LibriSpeech, 100 samples/subset)

| Model | Subset | WER | CER | Mean Latency | RTF |
|---|---|---:|---:|---:|---:|
| 0.6B | test-clean | 2.33% | 0.59% | 0.35s | 0.0393 |
| 0.6B | test-other | 4.30% | 2.11% | 0.40s | 0.0553 |
| 1.7B | test-clean | 1.94% | 0.57% | 0.77s | 0.0862 |
| 1.7B | test-other | 3.45% | 1.48% | 0.66s | 0.0914 |

Artifacts: `2026-09-07-librispeech-test-{clean,other}-100.json`,
`2026-09-07-librispeech-test-{clean,other}-100-1p7b.json`

---

## Latency

Median of 10 timed runs after 3 warm-ups, idle machine. The short clip is
`tests/fixtures/test_speech.wav` (2.5 s); the 10 s clip is that fixture tiled.

| Configuration | Short clip (~2.5s) | 10s clip | RTF (10s) | vs fp16 (10s) |
|---|---:|---:|---:|---:|
| 0.6B fp16 (baseline) | 0.17s | 0.30s | 0.029 | — |
| 0.6B 8-bit (g64) | 0.10s | 0.23s | 0.024 | 1.32x |
| 0.6B 4-bit (g64) | 0.09s | 0.17s | 0.018 | **1.71x** |
| 1.7B fp16 | 0.36s | 0.73s | 0.077 | 2.4x slower |

Quantization speedups are smaller than the February figures because fp16 no
longer runs in float32. Where absolute latency matters most, 4-bit is still the
fastest option; where quality matters, 8-bit matches fp16 output.

Artifacts: `2026-09-07-latency-{fp16,8bit-g64,4bit-g64,1p7b-fp16}-{short,10s}.json`

---

## Quantization Quality (0.6B, LibriSpeech test-clean)

100 speaker-balanced samples, round-robin sampling across speakers. 8-bit
produced the same hypothesis as fp16 on all 100 samples.

| Configuration | WER | CER | WER vs fp16 | Speed vs fp16 (10s clip) |
|---|---:|---:|---:|---:|
| fp16 (baseline) | 2.33% | 0.59% | — | — |
| 8-bit (g64) | 2.33% | 0.59% | +0.00pp | 1.32x |
| 4-bit (g64) | 2.59% | 0.86% | +0.26pp | 1.71x |

Artifact: `2026-09-07-quant-matrix-test-clean-speaker100.md`

---

### Published artifact recipe (2026-09-19)

The `moona3k/mlx-qwen3-asr-*` artifacts use `--encoder-bits 8` for the 4-bit
variants. On 0.6B the audio encoder carried most of the all-4-bit loss:

| 0.6B recipe | WER | CER | Size | Hyps differing from fp16 |
|---|---:|---:|---:|---:|
| fp16 | 2.33% | 0.59% | 1.8 GB | 0 |
| all 4-bit g64 | 2.63% | 0.93% | 430 MB | 28 |
| all 4-bit g32 | 2.55% | 0.79% | 477 MB | 22 |
| decoder 4-bit, encoder fp16 | 2.37% | 0.71% | 680 MB | 17 |
| **decoder 4-bit, encoder 8-bit** (published) | **2.37%** | **0.71%** | **517 MB** | 17 |
| decoder 4-bit, encoder 8-bit, embeddings 8-bit | 2.37% | 0.75% | 591 MB | 17 |
| all 8-bit g64 | 2.33% | 0.59% | 801 MB | 0 |

1.7B with the same 4-bit recipe: 1.73% WER (fp16 1.94%), 12/100 hypotheses
differ; 8-bit is identical to fp16 on 100/100.
Artifacts: `2026-09-19-quantized-artifacts-librispeech-test-clean-100-*.json`.

## Quantization Quality (0.6B, LibriSpeech test-other)

| Configuration | WER | CER | WER vs fp16 | Speed vs fp16 (10s clip) |
|---|---:|---:|---:|---:|
| fp16 (baseline) | 4.30% | 2.11% | — | — |
| 8-bit (g64) | 4.14% | 2.06% | -0.16pp | 1.32x |
| 4-bit (g64) | 5.74% | 2.71% | +1.43pp | 1.71x |

Artifact: `2026-09-07-quant-matrix-test-other-speaker100.md`

---

## Multilingual Quality (FLEURS, 10 languages x 10 samples)

Primary metric rule: CER for Chinese/Japanese/Korean; WER for all others. The
manifest is `2026-09-07-fleurs-multilingual-100-manifest.jsonl`, rebuilt with the
February seed and containing the same 100 samples.

### 0.6B (fp16)

| Language | Samples | WER | CER | Primary | Latency |
|---|---:|---:|---:|---:|---:|
| Arabic | 10 | 21.5% | 6.8% | 21.5% | 0.54s |
| Chinese | 10 | 91.7% | 5.0% | 5.0% | 0.36s |
| English | 10 | 4.6% | 1.6% | 4.6% | 0.32s |
| French | 10 | 17.3% | 9.2% | 17.3% | 0.50s |
| German | 10 | 8.0% | 4.7% | 8.0% | 0.65s |
| Hindi | 10 | 16.7% | 9.9% | 16.7% | 1.98s |
| Japanese | 10 | 89.7% | 9.3% | 9.3% | 0.51s |
| Korean | 10 | 17.2% | 6.7% | 6.7% | 0.46s |
| Russian | 10 | 8.8% | 3.4% | 8.8% | 0.62s |
| Spanish | 10 | 3.0% | 0.6% | 3.0% | 0.59s |
| **Aggregate** | **100** | **16.0%** | **5.4%** | **9.54%** | **0.65s** |

Artifact: `2026-09-07-manifest-quality-multilingual100-0p6b.json`

### 1.7B (fp16)

| Language | Samples | Primary | Latency |
|---|---:|---:|---:|
| Arabic | 10 | 16.0% | 1.09s |
| Chinese | 10 | 8.5% | 0.75s |
| English | 10 | 4.2% | 0.69s |
| French | 10 | 4.1% | 1.05s |
| German | 10 | 5.8% | 1.15s |
| Hindi | 10 | 17.7% | 3.12s |
| Japanese | 10 | 3.6% | 1.08s |
| Korean | 10 | 5.3% | 0.98s |
| Russian | 10 | 5.4% | 1.20s |
| Spanish | 10 | 0.7% | 1.09s |
| **Aggregate** | **100** | **6.70%** | **1.22s** |

Artifact: `2026-09-07-manifest-quality-multilingual100-1p7b.json`

### 0.6B vs 1.7B Comparison

| Language | 0.6B Primary | 1.7B Primary | Delta | Latency Ratio |
|---|---:|---:|---:|---:|
| Arabic | 21.5% | 16.0% | -5.5pp | 2.01x |
| Chinese | 5.0% | 8.5% | +3.5pp | 2.08x |
| English | 4.6% | 4.2% | -0.5pp | 2.18x |
| French | 17.3% | 4.1% | -13.2pp | 2.10x |
| German | 8.0% | 5.8% | -2.2pp | 1.76x |
| Hindi | 16.7% | 17.7% | +1.0pp | 1.58x |
| Japanese | 9.3% | 3.6% | -5.8pp | 2.11x |
| Korean | 6.7% | 5.3% | -1.4pp | 2.13x |
| Russian | 8.8% | 5.4% | -3.4pp | 1.95x |
| Spanish | 3.0% | 0.7% | -2.2pp | 1.87x |
| **Overall** | **9.54%** | **6.70%** | **-2.83pp** | **1.87x** |

Chinese moves the other way because the 1.7B spells out numbers in Chinese
characters (`二十九` instead of `29`) while the reference uses Arabic numerals;
both are correct.

**Takeaway:** 1.7B is the quality choice (30% relative improvement); 0.6B
is the speed choice (1.9x faster). The biggest 1.7B wins are on French, Japanese
and Arabic.

---

## Long-Form Quality (FLEURS concatenated, 78-90s per clip)

0.6B fp16, 10 clips (one per language) built deterministically from the
multilingual manifest; manifest `2026-09-07-fleurs-longform-10x75-manifest.jsonl`.

| Language | WER | CER | Primary | Latency |
|---|---:|---:|---:|---:|
| Arabic | 22.2% | 8.1% | 22.2% | 3.9s |
| Chinese | 47.6% | 3.2% | 3.2% | 2.0s |
| English | 5.7% | 1.9% | 5.7% | 2.7s |
| French | 19.7% | 11.2% | 19.7% | 3.5s |
| German | 6.1% | 3.4% | 6.1% | 3.4s |
| Hindi | 25.0% | 13.8% | 25.0% | 8.1s |
| Japanese | 89.5% | 10.6% | 10.6% | 2.8s |
| Korean | 10.3% | 3.7% | 3.7% | 3.9s |
| Russian | 11.6% | 4.2% | 11.6% | 4.1s |
| Spanish | 4.3% | 0.4% | 4.3% | 3.2s |
| **Aggregate** | **15.1%** | **6.0%** | **10.6%** | **3.8s** |

Quality is consistent with the short-clip lane: no truncation or chunking
artifacts on 80-second inputs.

Artifact: `2026-09-07-manifest-quality-longform10-0p6b.json`

---

## Real-World Quality (AMI + Earnings22 chunked, n=200, February 2026)

Deterministic mixed-condition lane with 100 AMI IHM meeting chunks and 100
Earnings22 chunked clips (16 speakers from AMI + 50 speakers from Earnings22).

| Metric | MLX (0.6B fp16) |
|---|---:|
| Primary error (WER) | 23.23% |
| WER | 23.23% |
| CER | 16.42% |
| Mean latency | 1.34s |

Artifacts:
- `2026-02-15-realworld-manifest-200.jsonl`
- `2026-02-15-manifest-quality-realworld200-0p6b.json`
- `2026-02-15-manifest-quality-realworld200-0p6b.md`

### Real-world long-form quality (Earnings22 full, n=3)

Deterministic non-synthetic full-recording lane from Earnings22 (`850s-1400s` filter; `~65 min` total audio).

| Metric | MLX (0.6B fp16) |
|---|---:|
| Primary error (WER) | 13.22% |
| WER | 13.22% |
| CER | 6.98% |
| Mean latency | 296.74s |
| Real-time factor | 0.228 |

Artifacts:
- `2026-02-15-earnings22-full-longform3-manifest.jsonl`
- `2026-02-15-manifest-quality-earnings22-full-longform3-0p6b.json`
- `2026-02-15-manifest-quality-earnings22-full-longform3-0p6b.md`

Head-to-head artifact:
- `2026-02-15-quality-head2head-mlx-vs-pytorch-earnings22-full-longform3.md` (in progress)

### MLX vs PyTorch Head-to-Head (Real-world manifest, n=200)

| Metric | MLX | PyTorch | Delta (MLX - Ref) |
|---|---:|---:|---:|
| Primary error | 23.23% | 23.04% | +0.19pp |
| WER | 23.23% | 23.04% | +0.19pp |
| CER | 16.42% | 16.31% | +0.10pp |
| Mean latency | 1.34s | 4.39s | 3.27x faster |

Artifact: `2026-02-15-quality-head2head-mlx-vs-pytorch-realworld200.md`

---

## MLX vs PyTorch Parity (0.6B, Multilingual-100)

Head-to-head on the same 100 FLEURS clips, same weights, greedy decode, run
2026-09-07 against `qwen-asr` (PyTorch reference). PyTorch latencies are CPU inference (`device_map="cpu"`, the reference stack's Apple Silicon path) and are not a like-for-like GPU comparison; they are reported because that is what a user gets from the official package on a Mac.

| Metric | MLX | PyTorch | Delta (MLX - Ref) |
|---|---:|---:|---:|
| WER | 16.00% | 16.69% | -0.70pp |
| CER | 5.43% | 5.64% | -0.21pp |
| Primary | 9.54% | 10.34% | **-0.81pp** |
| Mean latency | 0.65s | 12.91s | 19.8x faster |

MLX is slightly better on aggregate; the per-language deltas below are within
noise for 10 samples per language and show no systematic gap.

### Per-Language Breakdown

| Language | MLX Primary | PyTorch Primary | Delta |
|---|---:|---:|---:|
| Arabic | 21.5% | 24.0% | -2.5pp |
| Chinese | 5.0% | 5.5% | -0.6pp |
| English | 4.6% | 6.0% | -1.4pp |
| French | 17.3% | 15.5% | +1.8pp |
| German | 8.0% | 6.7% | +1.3pp |
| Hindi | 16.7% | 21.2% | -4.5pp |
| Japanese | 9.3% | 10.8% | -1.5pp |
| Korean | 6.7% | 6.5% | +0.2pp |
| Russian | 8.8% | 9.3% | -0.5pp |
| Spanish | 3.0% | 2.6% | +0.4pp |

Artifact: `2026-09-07-quality-head2head-mlx-vs-pytorch-multilingual100.md`

### MLX vs PyTorch Head-to-Head (LibriSpeech test-other, n=100)

| Metric | MLX | PyTorch | Delta (MLX - Ref) |
|---|---:|---:|---:|
| WER | 4.30% | 4.41% | -0.11pp |
| CER | 2.11% | 2.14% | -0.04pp |
| Mean latency | 0.40s | 2.76s | 7.0x faster |

Artifact: `2026-09-07-quality-head2head-mlx-vs-pytorch-test-other100.md`

### MLX vs PyTorch Head-to-Head (Long-form manifest, n=10)

10 concatenated FLEURS clips of 78-90 s, one per language.

| Metric | MLX | PyTorch | Delta (MLX - Ref) |
|---|---:|---:|---:|
| Primary error | 10.59% | 17.99% | -7.40pp |
| WER | 15.12% | 24.31% | -9.18pp |
| CER | 6.00% | 11.97% | -5.97pp |
| Mean latency | 3.75s | 26.76s | 7.1x faster |

The reference is fed each 80-second clip whole; MLX splits it at low-energy
points into 30 s chunks. The gap is mostly the reference degrading on long
inputs, not MLX gaining.

Artifact: `2026-09-07-quality-head2head-mlx-vs-pytorch-longform10.md`

### Token-Level Parity Analysis

Strict greedy parity on the same 100 multilingual clips: token match rate
66%, normalized text match rate 68%.

| Category | Count |
|---|---:|
| Exact match | 66 |
| Punctuation or tokenization | 2 |
| Numeric surface form | 5 |
| Content shift | 3 |
| Minor lexical shift | 24 |

Mismatches are dominated by lexical and numeric surface forms (`10,000` vs
`zehntausend`), which both score as errors or both as correct against the
reference transcript; they do not indicate a quality regression. Chinese,
Korean and Russian match token for token on 9 of 10 clips; Arabic, French
and Hindi diverge earliest.

Artifacts: `2026-09-07-reference-parity-suite-multilingual100.json`,
`2026-09-07-reference-parity-suite-multilingual100-analysis.md`

**2026-09-19 rerun (v0.4.1, after encoder tail padding, PR #21).** Same 100
clips; reference stack now `qwen-asr` on PyTorch 2.14 / transformers 4.57.6
(was 2.10): token match 64%, text match 67%. Six MLX outputs changed, all
attributable to #21; two reference outputs also changed with the stack
upgrade, and one flip is entirely due to that. Token flips on a tail-token
change are borderline fp16 decisions and net to roughly zero, so the encoder
was compared directly (MLX fp16 vs reference fp32, 20 clips): mean absolute
error on the last encoder token vs the reference fell from 0.0101 to 0.0046
(feature scale 0.016): lower on 15 clips, unchanged on 4 (no tail or an English
clip already at parity), higher on 1 (a Japanese clip, 0.0091 -> 0.0103). Overall error fell
from 0.00291 to 0.00275. #21 brings the encoder closer to the reference.

Caveat found during the 2026-09-19 aligner audit (below): the CPU `qwen-asr`
reference does not apply the encoder's `n_window_infer` attention window, so
the absolute errors on clips longer than 800 mel frames in this artifact are
dominated by the reference's missing mask, not by MLX. The before/after
direction stands (same reference on both sides); the long-clip magnitudes do
not measure MLX error.

Artifacts: `2026-09-19-reference-parity-suite-multilingual100.json`,
`2026-09-19-reference-parity-suite-multilingual100-analysis.md`,
`2026-09-19-encoder-parity-tail-padding.json`

---

### Long-Form Speed (February 2026, superseded by the head-to-head above)

| Metric | MLX | PyTorch | Ratio |
|---|---:|---:|---:|
| Mean latency (75-90s clips) | 11.55s | 48.39s | **4.19x faster** |

MLX on Apple Silicon is over 4x faster than PyTorch on the same machine for long-form inference.

Artifact: `2026-02-15-reference-parity-suite-longform10.md`

---

## Forced Aligner Parity (LibriSpeech, English)

MLX native aligner vs official `qwen-asr` PyTorch backend, 50 samples
(33 clips <= 8 s, 17 clips up to 20 s). Re-measured 2026-09-19 on two stacks.

| Metric | 2026-09-19, MLX 0.30.6 | 2026-09-19, MLX 0.32.2 | 2026-02-14 |
|---|---:|---:|---:|
| Text match rate | 100% | 100% | 100% |
| Timing MAE (all boundaries) | 5.57 ms | 5.57 ms | 5.69 ms |
| MLX mean latency | 0.12s | 0.15s | 0.21s |
| Official backend mean latency | 0.35s | 0.40s | 0.56s |
| Relative speed | **2.95x faster** | **2.75x faster** | 2.64x faster |

Aligner audio-encoder output vs the fp32 reference (new lane,
`scripts/eval_aligner_encoder_parity.py`): 0.078% mean / 0.089% max relative
error on all 50 clips, identical on MLX 0.30.6 and 0.32.2. The shipped CPU
reference does not apply the encoder's 800-frame attention window
(`_prepare_attention_mask` is defined but never called in `qwen-asr` 0.0.6);
compared as shipped, clips over 8 s show 5-13% error that disappears when the
reference's own mask is applied. The MLX encoder windows as trained.
Reported upstream as [QwenLM/Qwen3-ASR#213](https://github.com/QwenLM/Qwen3-ASR/issues/213).

Artifact: `2026-09-19-aligner-parity-50.md` (supersedes `2026-02-14-aligner-parity-50.md`)

---

## Mel Spectrogram Parity

MLX custom mel vs HuggingFace `WhisperFeatureExtractor(128)`.

| Metric | Value |
|---|---:|
| Max MAE | 2.83e-07 |
| Max absolute diff | 1.13e-04 |
| Frame length match | Exact |

Artifact: `2026-02-14-mel-parity.md`

---

## Optimization Impact

### Encoder Windowing (hybrid dense/segmented)

| Sequence Length | Windows | Dense (s) | Windowed (s) | Speedup |
|---:|---:|---:|---:|---:|
| 832 | 8 | 0.009 | 0.013 | 0.74x |
| 2,080 | 20 | 0.055 | 0.047 | 1.17x |
| 3,120 | 30 | 0.110 | 0.074 | 1.48x |
| 8,320 | 80 | 0.799 | 0.191 | **4.17x** |

Threshold: segmented execution kicks in at 20+ windows. Short audio uses dense masks (no regression), long audio gets up to 4.2x speedup.

Artifact: `2026-02-14-encoder-windowing-threshold.md`

### WAV Fast-Path Loader

| Scenario | Before | After | Improvement |
|---|---:|---:|---:|
| fp16 short | 0.545s | 0.506s | 7.2% faster |
| 4-bit short | 0.157s | 0.118s | **24.8% faster** |
| 4-bit 10s | 0.249s | 0.226s | 9.4% faster |

Biggest impact on quantized short clips where audio loading is a larger fraction of total time.

Artifact: `2026-02-14-wav-fastpath.md`

### Streaming Quality vs Offline (2026-09-19, v0.4.2)

`scripts/eval_streaming_manifest.py` on the maintained manifests, 2 s chunks,
30 s window, final text scored against the manifest references with the
`eval_manifest_quality` normalisation (`quality_vs_reference` in each artifact).
"Incremental KV" is the design shipped through v0.4.1; "window re-feed" is the
official recipe adopted in v0.4.2 (Decision 29).

| Lane | Offline primary | Incremental KV (fixed / energy) | Window re-feed (fixed / energy) | Re-feed RTF mean / p95 |
|---|---:|---:|---:|---:|
| Multilingual-100 (5-20 s clips) | 9.54% | 56.0% / 57.6% | **11.4% / 11.1%** | 0.083 / 0.135 |
| Long-form 10 x 75 s | 10.59% | 34.7% / 39.6% | **12.3% / 12.1%** | 0.178 / 0.254 |

The incremental design duplicated and dropped whole segments ("In the woman's
sitting group, failed to finish the." twice in one 13 s clip). The re-feed
design's remaining gap to offline is surface form (numerals, punctuation) and
occasional word choice; its `rewrite_rate` is ~0.65 because the last
`unfixed_token_num` tokens are regenerated every chunk, by design.

Artifacts: `2026-09-19-streaming-manifest-{multilingual100,longform10}.json`
and the `-incremental-kv` counterparts.

Prefix reuse (Decision 30, later on 2026-09-19): encoder output and decoder KV
for complete 8 s attention windows are cached across chunks; each chunk
encodes and prefills only what changed. Same machine, same day, cache off vs
on (`-no-prefix-reuse` vs `-prefix-reuse` artifacts):

| Lane | Primary off -> on (fixed / energy) | RTF mean off -> on | RTF p95 off -> on | Hypotheses changed |
|---|---:|---:|---:|---:|
| Multilingual-100 | 11.35 / 11.05 -> 11.32 / 11.02 | 0.090 -> 0.081 | 0.157 -> 0.134 | 3 / 200, each equal or better |
| Long-form 10 x 75 s | 12.33 / 12.13 -> 12.28 / 12.08 | 0.104 -> 0.083 | 0.136 -> 0.111 | 2 / 20, each equal or better |

Commit at silence (Decision 31), long-form lane, hard cut vs silence cut on
identical code (`-hard-cut` vs `-silence-cut` artifacts): primary fixed
12.28% -> 11.87%, energy 12.08% -> 12.23%; 8 rows better, 5 worse, 7 same;
zero adjacent duplicate words either way; the hard-cut boundary artifacts
(spurious sentence breaks and a hallucinated phrase at 30 s multiples) are
absent from the silence-cut hypotheses. Interleaved timing on 4 clips: RTF
0.0765 -> 0.0838.

Per 2 s chunk on a 30 s window the remaining time is ~55% greedy generation
(the rollback regenerates ~5 tokens plus the new ones), ~27% prefill of the
prompt tail and text prefix, ~13% encoder for the partial block. The 0.178 RTF
quoted above for the re-feed long-form lane came from an earlier run the same
day under unknown machine load; the cache-off rerun made immediately before
the cache-on run measured 0.104 on identical code, which is the baseline the
speedup is quoted against.

### Streaming (Rolling Decode, February 2026)

| Metric | Value |
|---|---:|
| Total latency (2.53s audio) | 1.13s |
| Per-chunk mean latency | 0.27s |
| Per-chunk p95 latency | 0.51s |
| Real-time factor | 0.45 |

Artifact: `2026-02-14-streaming-rolling-baseline.md`

### Speculative Decoding (Experimental)

0.6B draft → 1.7B target, fp16.

| Clip | Baseline | Speculative | Relative |
|---|---:|---:|---:|
| Short (~2.5s) | 1.45s | 2.72s | 0.53x (slower) |
| 10s | 2.68s | 4.90s | 0.55x (slower) |

Greedy parity verified, but currently slower due to draft audio encoder overhead. Kept as experimental opt-in.

Artifact: `2026-02-14-speculative-prototype.md`

---

## Artifact Index

All benchmark artifacts are committed under `docs/benchmarks/`. Key files:

| Artifact | Description |
|---|---|
| `2026-09-07-quality-matrix-refresh.md` | v0.4.0 quality + latency refresh (commands and tables) |
| `2026-09-07-quant-matrix-test-{clean,other}-speaker100.md` | v0.4.0 quantization quality |
| `2026-09-07-latency-*.json` | v0.4.0 idle-machine latency runs |
| `2026-09-07-quality-head2head-mlx-vs-pytorch-*.md` | v0.4.0 MLX vs PyTorch on multilingual-100, test-other, long-form |
| `2026-09-07-reference-parity-suite-multilingual100-analysis.md` | v0.4.0 token-level parity |
| `2026-09-19-reference-parity-suite-multilingual100-analysis.md` | v0.4.1 token-level parity rerun after encoder tail padding |
| `2026-09-19-encoder-parity-tail-padding.json` | v0.4.1 encoder-output error vs fp32 reference, before/after tail padding |
| `2026-09-19-aligner-parity-50.md` | Forced aligner audit on MLX 0.30.6 and 0.32.2: word timestamps + aligner encoder output; reference window-mask finding |
| `2026-09-19-streaming-manifest-*.json` | v0.4.2 streaming quality vs offline on maintained manifests (before/after re-feed); `-{no-,}prefix-reuse` pairs for Decision 30; `-{hard,silence}-cut` pair for Decision 31 |
| `2026-09-19-quantized-artifacts-*.json` | Quantized artifact evals (0.6B/1.7B x 4/8-bit) |
| `2026-02-14-quant-matrix-speaker100.md` | Quantization quality + latency matrix |
| `2026-02-15-quant-matrix-test-other-speaker100.md` | Quantization quality + latency on LibriSpeech test-other |
| `2026-02-15-manifest-quality-multilingual100-0p6b-refresh.json` | 0.6B multilingual quality |
| `2026-02-15-manifest-quality-multilingual100-1p7b-refresh.json` | 1.7B multilingual quality |
| `2026-02-15-quality-head2head-mlx-vs-pytorch-multilingual100.md` | MLX vs PyTorch parity |
| `2026-02-15-quality-head2head-mlx-vs-pytorch-test-other100.md` | MLX vs PyTorch parity on test-other |
| `2026-02-15-quality-head2head-mlx-vs-pytorch-longform10.md` | MLX vs PyTorch parity on long-form manifest |
| `2026-02-15-manifest-quality-longform10.md` | Long-form quality |
| `2026-02-15-reference-parity-suite-longform10.md` | Long-form speed (MLX vs PyTorch) |
| `2026-02-14-aligner-parity-50.md` | Forced aligner parity |
| `2026-02-14-mel-parity.md` | Mel spectrogram parity |
| `2026-02-14-encoder-windowing-threshold.md` | Encoder windowing optimization |
| `2026-02-14-wav-fastpath.md` | WAV fast-path optimization |
| `2026-02-14-speculative-prototype.md` | Speculative decoding prototype |
| `2026-02-14-streaming-rolling-baseline.md` | Streaming baseline |

90+ total JSON + markdown artifacts. See `docs/benchmarks/README.md` for the full chronological index.
