# Forced Aligner Parity and Encoder Parity (2026-09-19, 50 samples)

Audit of the native MLX forced aligner (`Qwen/Qwen3-ForcedAligner-0.6B`, fp16)
against the official `qwen-asr` PyTorch backend (CPU, fp32), on two MLX/torch
stacks. First aligner measurement since 2026-02-14; covers the newest MLX and
the post-#21 encoder tail padding. Host: Apple M4 Pro.

- Subset: LibriSpeech `test-clean`, first 50 deterministic samples
  (33 clips <= 8 s, 17 clips > 8 s, longest 20.1 s)
- Language: English
- Reference: `qwen-asr` 0.0.6, `transformers` 4.57.6, attention impl `sdpa`

## End-to-end word timestamps (`scripts/eval_aligner_parity.py`)

| Stack | Text match | MAE start (ms) | MAE end (ms) | MAE all (ms) | MLX latency (s) | qwen-asr latency (s) | Speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLX 0.30.6 / torch 2.10.0 | 50/50 | 4.09 | 7.04 | 5.57 | 0.118 | 0.348 | 2.95x |
| MLX 0.32.2 / torch 2.14.0 | 50/50 | 4.09 | 7.04 | 5.57 | 0.147 | 0.403 | 2.75x |
| 2026-02-14 baseline | 50/50 | 4.26 | 7.12 | 5.69 | 0.211 | 0.557 | 2.64x |

Timing error is identical to the millisecond on both stacks and slightly lower
than February. No word-level regression from #21 or from MLX 0.32.2.

Artifacts: `2026-09-19-aligner-parity-50-mlx0.30.6.json`,
`2026-09-19-aligner-parity-50-mlx0.32.2.json`.

## Aligner audio encoder output (`scripts/eval_aligner_encoder_parity.py`)

MLX `audio_tower` output vs the fp32 PyTorch `audio_tower`, per clip, over all
audio tokens. `relative_mae` is MAE divided by the mean |reference|.

| Stack | Reference attention | rel. MAE mean | rel. MAE max | last-token MAE max | max abs err |
|---|---|---:|---:|---:|---:|
| MLX 0.30.6 | windowed | 0.078% | 0.089% | 0.0011 | 0.018 |
| MLX 0.32.2 | windowed | 0.078% | 0.089% | 0.0011 | 0.007 |
| MLX 0.30.6 | as-shipped | 2.87% | 12.8% | 0.31 | 2.57 |
| MLX 0.32.2 | as-shipped | 2.87% | 12.8% | 0.31 | 2.57 |

Under the windowed reference every clip sits at 0.08% relative error, short
and long alike, on both MLX versions. That is fp16-vs-fp32 noise; the aligner
encoder is numerically correct.

### The reference does not window on CPU

`qwen-asr` 0.0.6 defines `Qwen3ASRAudioEncoder._prepare_attention_mask` (the
block-diagonal mask over `n_window_infer = 800` mel frames = 104 tokens that
the model was trained with) but never calls it. `cu_seqlens` is only consumed
by the flash-attention-2 kernel. On CPU (`sdpa` or `eager`) the shipped
reference attends across the whole clip.

Consequence, visible in the as-shipped rows: the 33 clips <= 8 s (one window)
match MLX at 0.078%; the 17 clips > 8 s diverge by 4.7% to 12.8%, growing with
duration (correlation 0.94). Applying the reference's own mask
(`--reference-attention windowed`, the default) removes the divergence
entirely, which fixes the cause: the MLX encoder windows as trained; the CPU
reference does not.

The same step is visible in `2026-09-19-encoder-parity-tail-padding.json`
(ASR 0.6B encoder): clips <= 800 frames show `mean_err` ~2.5e-5, clips > 800
frames show 0.0025 to 0.007, a 100x jump at exactly the window boundary. The
before/after comparison in that artifact is still valid (same reference both
sides) but its absolute long-clip numbers measure the reference's missing
mask, not MLX error. Word-timestamp parity survives the unmasked reference
because argmax over 80 ms timestamp classes is robust to it.

Artifacts: `2026-09-19-aligner-encoder-parity-50-mlx{0.30.6,0.32.2}-{windowed,as-shipped}.json`.

Reported upstream: https://github.com/QwenLM/Qwen3-ASR/issues/213. On the ASR
0.6B model the same clip shows 24.9% relative encoder divergence, and 4 of 22
LibriSpeech clips over 8 s change transcript (punctuation/casing) between the
shipped and windowed reference.

## Gate

`RUN_ALIGNER_PARITY=1` in `scripts/quality_gate.py` now runs both scripts:
text match 1.0 and timing MAE <= 60 ms as before, plus encoder
`relative_mae_max <= 0.005` and `last_token_mae_max <= 0.005` against the
windowed reference (5x headroom over the measured 0.00089 / 0.0011).

## Reproduce

```bash
# repo venv: MLX 0.30.6, torch 2.10, qwen-asr 0.0.6
python scripts/eval_aligner_parity.py --samples 50 \
  --json-output docs/benchmarks/2026-09-19-aligner-parity-50-mlx0.30.6.json
python scripts/eval_aligner_encoder_parity.py --samples 50 \
  --json-output docs/benchmarks/2026-09-19-aligner-encoder-parity-50-mlx0.30.6-windowed.json
python scripts/eval_aligner_encoder_parity.py --samples 50 --reference-attention as-shipped \
  --json-output docs/benchmarks/2026-09-19-aligner-encoder-parity-50-mlx0.30.6-as-shipped.json
# repeat from a venv with the newest mlx for the 0.32.2 rows
```
