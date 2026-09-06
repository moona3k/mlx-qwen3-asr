# Academic Best-Practice Review (2026-02-15)

Scope: verify current architecture against primary ASR/decoding literature and
identify only high-ROI follow-ups.

## Sources (primary)

- Qwen3-ASR technical report: https://arxiv.org/abs/2601.21337
- Whisper: https://arxiv.org/abs/2212.04356
- Emformer (streaming ASR): https://arxiv.org/abs/2010.10759
- Speculative decoding: https://arxiv.org/abs/2211.17192
- CTC segmentation for forced alignment: https://arxiv.org/abs/2007.09127
- SentencePiece: https://arxiv.org/abs/1808.06226
- BPE for subword units: https://arxiv.org/abs/1508.07909

## Best-Practice Cross-Check

| Area | Literature-backed best practice | Repo status | Assessment |
|---|---|---|---|
| Streaming decode | Reuse incremental state/KV cache; avoid full re-transcription per chunk (Emformer-style latency discipline). | `mlx_qwen3_asr/streaming.py` keeps `_cache` and `_next_position`, uses `prefill()` then `step()` across chunks. | Aligned |
| Streaming memory/latency control | Bound context growth and reset stale state when context is pruned. | `streaming.py` caps `audio_accum`, resets incremental state on trim. | Aligned |
| Streaming partial hypothesis stability | Keep a stable prefix and only allow tail revisions (common simultaneous ASR pattern; also reflected in Qwen streaming design). | `unfixed_chunk_num`/`unfixed_token_num`, stable/unstable split, tail refine mode. | Aligned |
| Forced alignment monotonicity | Enforce monotonic/consistent timestamp progression (CTC-segmentation motivation). | Native aligner applies timestamp post-fix + monotonic correction before word spans. | Aligned |
| Tokenizer compatibility | For pretrained models, keep tokenizer behavior bit-compatible with official vocab/merges + special tokens (BPE/SentencePiece practice). | Native tokenizer consumes `vocab.json`, `merges.txt`, `tokenizer_config.json`; special tokens preserved. | Aligned |
| Speculative decoding correctness | Verify draft tokens with target model to preserve exact greedy output (Leviathan et al.). | `generate_speculative()` verifies and trims caches on rejection; not default-on. | Aligned |
| Robust multilingual evaluation | Include multilingual and long-form eval lanes; avoid over-fitting to narrow English benchmarks (Whisper + Qwen3-ASR report context). | Manifest-driven multilingual and long-form parity/quality artifacts are committed under `docs/benchmarks/`. | Aligned |

## Gaps Worth Addressing Next

1. Streaming endpointing/VAD lane
- Why: low-latency ASR quality usually improves when chunking is speech-aware, not only fixed-size.
- Current: streaming path is fixed chunk size with rollback heuristics.
- Suggested: add optional VAD/endpointer gate and evaluate latency vs rollback rate.

2. Streaming-specific quality gate
- Why: offline WER/CER is insufficient for user-perceived streaming quality.
- Current: initial runtime metrics are now exposed (`partial_stability`,
  `rewrite_rate`, `finalization_delta_chars`) and wired into streaming benchmark output.
- Suggested: promote these metrics to a committed dataset-level gate lane.

3. Alignment confidence surface
- Why: forced alignment systems benefit from confidence diagnostics for downstream filtering.
- Current: timestamps returned without confidence.
- Suggested: expose per-word confidence proxy from timestamp logits/posteriors.

4. Optional tokenizer parity CI lane
- Why: protects against silent drift after tokenizer refactors.
- Current: native tests exist; manual HF parity spot-check passed.
- Suggested: optional CI job (when `transformers` installed) for encode/decode parity corpus.

## Conclusion

The current implementation is consistent with core literature-backed practices
for streaming cache reuse, forced alignment monotonicity, tokenizer compatibility,
and speculative-decoding correctness. The main remaining best-practice work is
in streaming product quality instrumentation (endpointing + streaming metrics),
not core model-path architecture.
