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
    - MLX-vs-PyTorch full `n=3` long-form comparison artifact committed
      (`...earnings22-full-longform3...`, bounded reference chunking).
- MLX-vs-PyTorch head-to-head:
  - Multilingual-100 direct comparison (MLX: 16.00% WER vs PyTorch: 16.69% WER).
  - LibriSpeech test-other direct comparison (MLX: 4.20% WER vs PyTorch: 4.41% WER).
  - Long-form manifest direct comparison (MLX: 16.71% WER vs PyTorch: 24.31% WER).
  - Real-world manifest direct comparison
    (MLX: 23.23% WER vs PyTorch: 23.04% WER, delta +0.19pp).
  - Versioned benchmark artifacts committed under `docs/benchmarks/`.
- Streaming diagnostics lane:
  - Per-session quality metrics exposed from runtime state:
    `partial_stability`, `rewrite_rate`, `finalization_delta_chars`.
  - Windowed re-decode with text-prefix rollback (official recipe); window
    bounded by `max_context_sec`, so per-chunk cost is bounded, not linear.
  - Tooling:
    - `scripts/eval_streaming_metrics.py` (single-run diagnostics probe),
    - `scripts/benchmark_streaming.py` (`streaming_quality` summary payload).
  - Strict release gate now includes a bounded streaming-quality check on
    fixture audio for `fixed` and `energy` endpointing modes.
- Release quality gate (all green):
  - ruff check, typed-core mypy, full pytest, reference parity,
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
   `partial_stability`, `rewrite_rate`, `finalization_delta_chars`.

5. **Streaming quality dataset artifacts** — CLOSED (2026-09-19). The
   streaming-manifest lane ran on the maintained multilingual-100 and
   long-form-10 manifests and the final text was scored against the
   references. It exposed that the incremental KV-cache design scored 56%
   primary error vs 9.5% offline; streaming was rebuilt on the official
   re-feed recipe (Decision 29) and now scores 11.4% / 12.3%. Artifacts:
   `docs/benchmarks/2026-09-19-streaming-manifest-*.json` (with
   `-incremental-kv` before/after pairs).

## Remaining Gaps (prioritized)

1. `P2` Streaming residual gap and cost
   - Why: streaming trails offline by ~2pp on both lanes and re-encodes the
     window every chunk (RTF 0.18 on 75 s clips with a 30 s window).
   - Candidates: cache encoder output per 100-frame chunk across re-decodes;
     overlap windows at commit so boundary words are not cut.
2. Streaming lane in the release gate: closed 2026-09-19.
   `scripts/eval_streaming_manifest.py` scores `final_text` against manifest
   references itself (`quality_vs_reference`, schema v1.2) and the strict
   release gate runs it by default against the multilingual-100 manifest with
   a ceiling of offline primary error + 3pp. A test re-scores the committed
   pre-fix artifact and asserts the gate fails it (`docs/QUALITY_GATE.md`).

## Follow-up Order

1. Extend the strict streaming ceiling to the long-form lane
   (`2026-09-07-fleurs-longform-10x75-manifest.jsonl` against
   `2026-09-07-manifest-quality-longform10-0p6b.json`) once the P2 encoder
   caching / window-overlap work lands, so boundary regressions are gated too.
