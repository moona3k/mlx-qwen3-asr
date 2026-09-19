---
license: apache-2.0
base_model: Qwen/Qwen3-ASR-1.7B
library_name: mlx
pipeline_tag: automatic-speech-recognition
tags:
  - mlx
  - qwen3-asr
  - speech-recognition
  - quantized
  - 8-bit
language:
  - en
  - zh
  - ja
  - ko
  - de
  - fr
  - es
  - ru
  - ar
  - hi
---

# moona3k/mlx-qwen3-asr-1.7b-8bit

8-bit (group size 64) MLX quantization of [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
for [mlx-qwen3-asr](https://github.com/moona3k/mlx-qwen3-asr), a ground-up MLX
reimplementation of Qwen3-ASR for Apple Silicon. No PyTorch, no transformers,
no model conversion step on the user's side.

## Usage

```bash
pip install -U mlx-qwen3-asr
mlx-qwen3-asr audio.wav --model moona3k/mlx-qwen3-asr-1.7b-8bit
```

```python
import mlx_qwen3_asr as m

result = m.transcribe("audio.wav", model="moona3k/mlx-qwen3-asr-1.7b-8bit")
print(result.text)
```

Requires `mlx-qwen3-asr >= 0.4.1`. Word timestamps (`--timestamps`) and the
HTTP server (`mlx-qwen3-asr serve --model moona3k/mlx-qwen3-asr-1.7b-8bit`) work unchanged.

## What is quantized

- Every `Linear` and `Embedding` layer in the text decoder **and** the audio
  encoder is affine-quantized to 8 bits with group size 64 (`mlx.nn.quantize`).
- Remaining floating tensors (scales, biases, norms, conv stem) are stored in
  float16, so inference runs in float16 end to end.
- `lm_head` is tied to the token embedding in the source model and is not
  stored twice; the loader re-ties it.
- Download size: 2.0G (fp16 source: 4.4 GB).

## Quality

LibriSpeech test-clean, 100 speaker-balanced clips (`speaker_round_robin`),
greedy decoding, Apple M4 Pro, MLX 0.30.6:

| Model | WER | CER |
|---|---:|---:|
| `Qwen/Qwen3-ASR-1.7B` fp16 | 1.94% | 0.57% |
| **this artifact (8-bit g64)** | **1.94%** | **0.57%** |

Hypotheses: identical to fp16 on all 100 clips.

Latency envelope from the committed quantization matrix
(`docs/benchmarks/2026-09-07-quant-matrix-test-clean-speaker100.md`, 0.6B):
8-bit runs about 2.4x and 4-bit about 2.7x faster than fp16 on a 10 s clip.
Per-sample JSON for this artifact's evaluation is in the mlx-qwen3-asr
repository under `docs/benchmarks/2026-09-19-quantized-artifacts-*.json`.

## Reproduce

```bash
git clone https://github.com/moona3k/mlx-qwen3-asr && cd mlx-qwen3-asr  # commit 5479ea7
python scripts/convert.py --model Qwen/Qwen3-ASR-1.7B --quantize 8 --group-size 64 \
  --dtype float16 --output-dir Qwen3-ASR-1.7B-8bit-g64
python scripts/eval_librispeech.py --model Qwen3-ASR-1.7B-8bit-g64 --samples 100 --sampling speaker_round_robin
```

## License

Apache-2.0, following the source model.
