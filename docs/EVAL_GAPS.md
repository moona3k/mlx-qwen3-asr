# Evaluation Coverage Matrix

This document tracks what is measured today versus what remains before we can
make broad "production-grade across languages/conditions" quality claims.

## Current Measured Coverage

- Token parity:
  - English mixed-condition lane (clean/other/long/noise): 22 samples.
  - Multilingual manifest lanes:
    - smoke: 4 samples (EN/CN/JA/DE),
    - medium: 20 samples (10 languages x 2),
    - expanded: 100 samples (10 languages x 10).
  - Long-form multilingual parity lane:
    - 10 synthetic concatenated clips (~75-90s each, 10 languages),
    - strict token/text parity currently `0.0` (all mismatch; early divergence).
- Aligner parity:
  - 50 LibriSpeech test-clean samples (MLX vs official qwen_asr backend).
- WER/CER lane:
  - LibriSpeech test-clean, 100 samples, speaker-balanced, 0.6B model
    (`fp16`, `4-bit`, `8-bit`).
  - LibriSpeech test-other, 100 samples, speaker-balanced, 0.6B model
    (`fp16`, `4-bit`, `8-bit`).
  - LibriSpeech test-clean + test-other, 100 samples each, 1.7B model
    (`fp16`).
  - FLEURS multilingual manifest-quality lane:
    - 100 short-form samples (10 languages x 10),
    - refreshed for both 0.6B and 1.7B (`fp16`),
    - Unicode-safe WER/CER + language-aware primary metric.
  - Long-form multilingual manifest-quality lane:
    - 10 synthetic concatenated clips (~75-90s each, 10 languages),
    - Unicode-safe WER/CER with language-aware primary metric
      (CER for zh/ja/ko; WER otherwise).
 - Real-world manifest-quality lane:
    - 200 real-world English clips (AMI IHM meetings + Earnings22 chunked),
    - deterministic speaker-balanced curation
      (100 clips/source, 16 AMI speakers + 50 Earnings22 speakers),
    - committed WER/CER + latency artifact.
  - Real-world non-synthetic long-form lane:
    - 3 full Earnings22 recordings (~65 min total; 15-23 min/clip),
    - deterministic family-aware sampling with committed manifest,
    - MLX quality artifact committed,
    - strict release profile now includes dedicated long-form quality gate
      (`RUN_REALWORLD_LONGFORM_EVAL=1` by default under strict profile),
    - interim MLX-vs-PyTorch full-length comparison committed for one recording
      (`...earnings22-full-longform1...`),
    - full `n=3` MLX-vs-PyTorch comparison remains pending.
- MLX-vs-PyTorch head-to-head:
  - Multilingual-100 direct comparison (MLX: 15.99% WER vs PyTorch: 16.69% WER).
  - LibriSpeech test-other direct comparison (MLX: 4.20% WER vs PyTorch: 4.41% WER).
  - Long-form manifest direct comparison (MLX: 16.71% WER vs PyTorch: 24.31% WER).
  - Real-world manifest direct comparison
    (MLX: 23.23% WER vs PyTorch: 23.04% WER, delta +0.19pp).
  - Versioned benchmark artifacts committed under `docs/benchmarks/`.
- Streaming diagnostics lane:
  - Per-session quality metrics exposed from runtime state:
    `partial_stability`, `rewrite_rate`, `finalization_delta_chars`.
  - KV-cache streaming with linear complexity (not O(n^2) re-transcription).
  - Tooling:
    - `scripts/eval_streaming_metrics.py` (single-run diagnostics probe),
    - `scripts/benchmark_streaming.py` (`streaming_quality` summary payload).
  - Strict release gate now includes a bounded streaming-quality check on
    fixture audio for `fixed` and `energy` endpointing modes.
- Release quality gate (all green):
  - ruff check, typed-core mypy, full pytest (441 tests), reference parity,
    LibriSpeech eval, manifest quality eval, benchmark ASR.
  - RTF=0.1526, latency_mean=0.3865s.

## Closed Gaps

1. **Non-English quality lane** — CLOSED. Multilingual quality lanes for both
   short-form (`n=100`) and long-form synthetic (`n=10`) across 10 languages
   with Unicode-safe WER/CER and language-aware primary metric (CER for
   zh/ja/ko, WER otherwise).

2. **Long-form quality lane** — CLOSED (synthetic). 10 synthetic concatenated
   clips (~75-90s each, 10 languages) with quality metrics. Real-world
   long-form remains a stretch goal.

3. **MLX-vs-PyTorch quality comparison** — CLOSED for current lanes.
   Multilingual-100, test-other, long-form synthetic, and real-world-200
   head-to-head artifacts are all committed. Real-world-200 remains near-parity
   (+0.19pp WER delta) with substantial MLX latency advantage.

4. **Streaming quality instrumentation** — CLOSED. Full instrumentation with
   `partial_stability`, `rewrite_rate`, `finalization_delta_chars`. KV-cache
   streaming shipped with linear complexity.

## Remaining Gaps (prioritized)

1. `P1` Real-world long-form MLX-vs-PyTorch head-to-head completion
   - Why: long-form MLX quality is now gated; we still need direct backend
     comparison evidence on the same full-recording lane.
   - Status: partial progress with committed one-recording head-to-head artifact;
     pending full `n=3` comparison artifact on the same manifest family.

2. `P2` Streaming quality dataset lane (committed artifacts)
   - Why: instrumentation is in place but no versioned benchmark dataset.
   - Status: strict release now gates fixture-level streaming quality; next step
     is multi-file versioned dataset artifacts.

## Follow-up Order

1. Commit real-world long-form MLX-vs-PyTorch comparison artifacts.
2. Commit streaming-quality versioned artifacts.
