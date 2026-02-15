# Benchmarks Directory

Store benchmark JSON outputs here.

## Curation Policy

- Keep a small set of canonical, current artifacts in `main` (for release-gate and latest regressions).
- Archive one-off historical runs in GitHub Releases or tagged snapshots instead of accumulating every local run here.
- Prefer updating a canonical artifact (for example `latest.json` or current gate files) over adding new dated files unless the run is part of a documented decision.

Suggested naming:

- `baseline_<machine>_<model>_<dtype>.json`
- `pr<id>_<machine>_<model>_<dtype>.json`
- `latest.json` for quick local runs
- `nightly-latency.json` for scheduled runtime trend
- `nightly-librispeech.json` for scheduled quality trend

## Recorded Results

Tokenizer cache optimization benchmark (2026-02-14):

- Machine: Apple M4 Pro, macOS 26.2
- Model: `Qwen/Qwen3-ASR-0.6B`, dtype `float16`
- Audio: `tests/fixtures/test_speech.wav`
- Runs: warmup=1, measured=3

Artifacts:

- `docs/benchmarks/2026-02-14-tokenizer-cache-before.json`
- `docs/benchmarks/2026-02-14-tokenizer-cache-after.json`

Decode-path optimization benchmark (2026-02-14):

- Changes:
  - preallocated KV cache writes instead of repeated concatenation,
  - GQA attention path no longer explicitly repeats K/V heads before fused SDPA.
- Machine: Apple M4 Pro, macOS 26.2
- Model: `Qwen/Qwen3-ASR-0.6B`, dtype `float16`

Artifacts:

- `docs/benchmarks/2026-02-14-gqa-kvcache-short.json`
- `docs/benchmarks/2026-02-14-gqa-kvcache-10s.json`
- `docs/benchmarks/2026-02-14-gqa-kvcache-10s-repeat.json`

Quantized comparison artifacts (2026-02-14):

- Runtime:
  - `docs/benchmarks/2026-02-14-4bit-short.json`
  - `docs/benchmarks/2026-02-14-4bit-10s.json`
- Quality vs FP16 (LibriSpeech sample):
  - `docs/benchmarks/2026-02-14-librispeech-fp16-20.json`
  - `docs/benchmarks/2026-02-14-librispeech-4bit-20.json`

Speaker-balanced LibriSpeech quality artifacts (2026-02-14):

- `docs/benchmarks/2026-02-14-librispeech-fp16-100-speaker-round-robin.json`
- `docs/benchmarks/2026-02-14-librispeech-8bit-g64-100-speaker-round-robin.json`
- `docs/benchmarks/2026-02-14-librispeech-4bit-g64-100-speaker-round-robin.json`

Quantization matrix sweep artifacts (2026-02-14):

- `docs/benchmarks/2026-02-14-quant-matrix.json`
- `docs/benchmarks/2026-02-14-quant-matrix.md`
- `docs/benchmarks/2026-02-14-quant-matrix-post-wavfast.json`
- `docs/benchmarks/2026-02-14-quant-matrix-post-wavfast.md`
- `docs/benchmarks/2026-02-14-quant-matrix-speaker100.json`
- `docs/benchmarks/2026-02-14-quant-matrix-speaker100.md`
- CI/manual workflow output names:
  - `docs/benchmarks/ci-quant-matrix.json`
  - `docs/benchmarks/ci-quant-matrix.md`

KV cache write-path follow-up (2026-02-14):

- `docs/benchmarks/2026-02-14-kvcache-write-path.json`
- `docs/benchmarks/2026-02-14-kvcache-write-path.md`

WAV fast-path loader experiment (2026-02-14):

- `docs/benchmarks/2026-02-14-wav-fastpath.json`
- `docs/benchmarks/2026-02-14-wav-fastpath.md`

Forced aligner backend smoke comparison (2026-02-14):

- `docs/benchmarks/2026-02-14-aligner-backend-smoke.json`
- `docs/benchmarks/2026-02-14-aligner-backend-smoke.md`

Forced aligner parity lane run (2026-02-14):

- `docs/benchmarks/2026-02-14-aligner-parity-10.json`
- `docs/benchmarks/2026-02-14-aligner-parity-10.md`
- `docs/benchmarks/2026-02-14-aligner-parity-50.json`
- `docs/benchmarks/2026-02-14-aligner-parity-50.md`
- `docs/benchmarks/2026-02-14-aligner-parity-10-ja-ko-tokenizer.json`
- `docs/benchmarks/2026-02-14-aligner-parity-10-ja-ko-tokenizer.md`

Tokenizer loader path benchmark (2026-02-14):

- `docs/benchmarks/2026-02-14-tokenizer-loader-path-benchmark.json`
- `docs/benchmarks/2026-02-14-tokenizer-loader-path-benchmark.md`

Tokenizer resolved-local-path benchmark (2026-02-14):

- `docs/benchmarks/2026-02-14-tokenizer-local-path-benchmark.json`
- `docs/benchmarks/2026-02-14-tokenizer-local-path-benchmark.md`

Encoder windowing threshold benchmark (2026-02-14):

- `docs/benchmarks/2026-02-14-encoder-windowing-threshold.json`
- `docs/benchmarks/2026-02-14-encoder-windowing-threshold.md`

Post-change short-clip sanity benchmark (2026-02-14):

- `docs/benchmarks/2026-02-14-short-after-windowed-hybrid.json`

Speculative decoding prototype benchmarks (2026-02-14):

- `docs/benchmarks/2026-02-14-speculative-1p7b-vs-0p6b-smoke.json`
- `docs/benchmarks/2026-02-14-speculative-1p7b-vs-0p6b.json`
- `docs/benchmarks/2026-02-14-speculative-1p7b-vs-0p6b-10s.json`
- `docs/benchmarks/2026-02-14-speculative-prototype.md`

Reference parity suite smoke run (2026-02-14):

- `docs/benchmarks/2026-02-14-reference-parity-suite-smoke.json`
- `docs/benchmarks/2026-02-14-reference-parity-suite-smoke.md`
- `docs/benchmarks/2026-02-14-reference-parity-suite-smoke-v2.json`
- `docs/benchmarks/2026-02-14-reference-parity-suite-smoke-v2.md`
- `docs/benchmarks/2026-02-14-reference-parity-suite-english22.json`
- `docs/benchmarks/2026-02-14-reference-parity-suite-english22.md`
- `docs/benchmarks/2026-02-14-fleurs-multilingual-smoke-manifest.jsonl`
- `docs/benchmarks/2026-02-14-reference-parity-suite-multilingual-smoke.json`
- `docs/benchmarks/2026-02-14-reference-parity-suite-multilingual-smoke.md`
- `docs/benchmarks/2026-02-14-reference-parity-suite-multilingual-smoke-analysis.json`
- `docs/benchmarks/2026-02-14-reference-parity-suite-multilingual-smoke-analysis.md`
- `docs/benchmarks/2026-02-14-fleurs-multilingual-20-manifest.jsonl`
- `docs/benchmarks/2026-02-14-reference-parity-suite-multilingual-20.json`
- `docs/benchmarks/2026-02-14-reference-parity-suite-multilingual-20-analysis.json`
- `docs/benchmarks/2026-02-14-reference-parity-suite-multilingual-20-analysis.md`
- `docs/benchmarks/2026-02-14-fleurs-multilingual-100-manifest.jsonl`
- `docs/benchmarks/2026-02-14-reference-parity-suite-multilingual-100.json`
- `docs/benchmarks/2026-02-14-reference-parity-suite-multilingual-100-analysis.json`
- `docs/benchmarks/2026-02-14-reference-parity-suite-multilingual-100-analysis.md`

Long-form multilingual parity artifacts (2026-02-15):

- manifests:
  - `docs/benchmarks/2026-02-14-fleurs-longform-smoke2-manifest.jsonl`
  - `docs/benchmarks/2026-02-14-fleurs-longform-10x75-manifest.jsonl`
- smoke parity:
  - `docs/benchmarks/2026-02-14-reference-parity-suite-longform-smoke2.json`
  - `docs/benchmarks/2026-02-14-reference-parity-suite-longform-smoke2-1024.json`
- full 10-sample run (serial shards + aggregate):
  - `docs/benchmarks/2026-02-15-reference-parity-suite-longform10-shard00.json`
  - `docs/benchmarks/2026-02-15-reference-parity-suite-longform10-shard01.json`
  - `docs/benchmarks/2026-02-15-reference-parity-suite-longform10-shard02.json`
  - `docs/benchmarks/2026-02-15-reference-parity-suite-longform10-shard03.json`
  - `docs/benchmarks/2026-02-15-reference-parity-suite-longform10-shard04.json`
  - `docs/benchmarks/2026-02-15-reference-parity-suite-longform10.json`
  - `docs/benchmarks/2026-02-15-reference-parity-suite-longform10.md`

Release-gate quality-lane artifact (2026-02-15):

- `docs/benchmarks/2026-02-15-librispeech-release-gate-20.json`

Diarization quality artifacts (2026-02-15):

- `docs/benchmarks/2026-02-15-diarization-manifest-20.jsonl`
- `docs/benchmarks/2026-02-15-diarization-quality-20.json` (initial baseline)
- `docs/benchmarks/2026-02-15-diarization-quality-20.md` (summary + run history)
- `docs/benchmarks/2026-02-15-diarization-quality-20-v2.json` (regressed experiment; rejected)
- `docs/benchmarks/2026-02-15-diarization-quality-20-v3.json` (post-rollback verification)
- `docs/benchmarks/2026-02-15-diarization-quality-20-v4.json` (accepted quality+latency improvement)
- `docs/benchmarks/2026-02-15-diarization-quality-20-v5.json` (accepted quality+latency improvement)

Manifest quality artifacts (2026-02-15):

- `docs/benchmarks/2026-02-15-manifest-quality-longform10.json`
- `docs/benchmarks/2026-02-15-manifest-quality-longform10.md`
- `docs/benchmarks/2026-02-15-manifest-quality-multilingual100.json`
- `docs/benchmarks/2026-02-15-manifest-quality-multilingual100.md`
- `docs/benchmarks/2026-02-15-manifest-quality-multilingual100-post-langcanon.json`
- `docs/benchmarks/2026-02-15-manifest-quality-earnings22-full-longform3-0p6b.json`
- `docs/benchmarks/2026-02-15-manifest-quality-earnings22-full-longform3-0p6b.md`
- `docs/benchmarks/2026-02-15-manifest-quality-earnings22-full-longform1-0p6b.json`
- `docs/benchmarks/2026-02-15-manifest-quality-earnings22-full-longform1-0p6b.md`

Real-world full-length head-to-head artifacts (2026-02-15):

- `docs/benchmarks/2026-02-15-quality-head2head-mlx-vs-pytorch-earnings22-full-longform1.json`
- `docs/benchmarks/2026-02-15-quality-head2head-mlx-vs-pytorch-earnings22-full-longform1.md`
- `docs/benchmarks/2026-02-15-quality-head2head-mlx-vs-pytorch-earnings22-full-longform1-checkpoint.json`

Post-canonicalization multilingual triage artifacts (2026-02-15):

- `docs/benchmarks/2026-02-15-reference-parity-suite-multilingual100-post-langcanon.json`
- `docs/benchmarks/2026-02-15-reference-parity-suite-multilingual20-fp32.json`
- `docs/benchmarks/2026-02-15-multilingual-triage-checkpoint.json`
- `docs/benchmarks/2026-02-15-multilingual-triage-checkpoint.md`
- `docs/benchmarks/2026-02-15-logit-parity-probe-multilingual100-mismatches.json`
- `docs/benchmarks/2026-02-15-logit-parity-probe-multilingual100-mismatches.md`
- `docs/benchmarks/2026-02-15-logit-parity-probe-nonnear5-fp32.json`
- `docs/benchmarks/2026-02-15-logit-parity-probe-nonnear5-fp32.md`
- `docs/benchmarks/2026-02-15-nonnear5-stage-parity.json`
- `docs/benchmarks/2026-02-15-nonnear5-stage-parity.md`
- `docs/benchmarks/2026-02-15-cache-vs-recompute-nonnear5.json`
- `docs/benchmarks/2026-02-15-cache-vs-recompute-nonnear5.md`
- `docs/benchmarks/2026-02-15-logit-parity-probe-nonnear5-post-enc-clamp.json`
- `docs/benchmarks/2026-02-15-logit-parity-probe-nonnear5-post-enc-clamp.md`
- `docs/benchmarks/2026-02-15-logit-parity-probe-nonnear5-post-enc-clamp-compare.json`
- `docs/benchmarks/2026-02-15-audio-encoder-layerwise-parity-nonnear5.json`
- `docs/benchmarks/2026-02-15-audio-encoder-layerwise-parity-nonnear5.md`
- `docs/benchmarks/2026-02-15-logit-parity-probe-nonnear5-post-maskmin.json`
- `docs/benchmarks/2026-02-15-logit-parity-probe-nonnear5-post-maskmin.md`
- `docs/benchmarks/2026-02-15-logit-parity-probe-nonnear5-post-maskmin-compare.json`
- `docs/benchmarks/2026-02-15-nonnear-parity-matrix.json`
- `docs/benchmarks/2026-02-15-nonnear-parity-matrix.md`

Generate with:

```bash
python scripts/benchmark_asr.py tests/fixtures/test_speech.wav \
  --model Qwen/Qwen3-ASR-0.6B \
  --runs 5 \
  --json-output docs/benchmarks/latest.json
```
