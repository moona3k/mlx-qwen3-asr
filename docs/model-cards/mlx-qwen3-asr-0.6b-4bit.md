---
license: apache-2.0
base_model: Qwen/Qwen3-ASR-0.6B
library_name: mlx
pipeline_tag: automatic-speech-recognition
tags:
  - mlx
  - qwen3-asr
  - speech-recognition
  - quantized
  - 4-bit
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

# moona3k/mlx-qwen3-asr-0.6b-4bit

4-bit (group size 64) MLX quantization of [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
for [mlx-qwen3-asr](https://github.com/moona3k/mlx-qwen3-asr), a ground-up MLX
reimplementation of Qwen3-ASR for Apple Silicon. No PyTorch, no transformers,
no model conversion step on the user's side.

## Usage

```bash
pip install -U mlx-qwen3-asr
mlx-qwen3-asr audio.wav --model moona3k/mlx-qwen3-asr-0.6b-4bit
```

```python
import mlx_qwen3_asr as m

result = m.transcribe("audio.wav", model="moona3k/mlx-qwen3-asr-0.6b-4bit")
print(result.text)
```

Requires `mlx-qwen3-asr >= 0.4.3`. Mixed encoder/decoder widths need the per-module loader introduced in 0.4.3; older versions fail to load this artifact with a shape error. Word timestamps (`--timestamps`) and the
HTTP server (`mlx-qwen3-asr serve --model moona3k/mlx-qwen3-asr-0.6b-4bit`) work unchanged.

## What is quantized

- Text decoder `Linear` and `Embedding` layers are affine-quantized to 4 bits;
  the audio encoder is quantized to **8 bits** (group size 64 for both). The
  encoder carries most of the 4-bit quality loss: on 0.6B, all-4-bit scored
  2.63% WER against 2.37% with an 8-bit encoder (fp16 2.33%), for 90 MB more.
- Remaining floating tensors (scales, biases, norms, conv stem) are stored in
  float16, so inference runs in float16 end to end.
- `lm_head` is tied to the token embedding in the source model and is not
  stored twice; the loader re-ties it.
- Download size: 517M (fp16 source: 1.8 GB).

## Quality

LibriSpeech test-clean, 100 speaker-balanced clips (`speaker_round_robin`),
greedy decoding, Apple M4 Pro, MLX 0.30.6:

| Model | WER | CER |
|---|---:|---:|
| `Qwen/Qwen3-ASR-0.6B` fp16 | 2.33% | 0.59% |
| **this artifact (4-bit g64)** | **2.37%** | **0.71%** |

Hypotheses: 17 of 100 hypotheses differ from fp16, mostly proper-noun spellings and British/American variants; WER is within 0.05pp of fp16.

Latency envelope from the committed quantization matrix
(`docs/benchmarks/2026-09-07-quant-matrix-test-clean-speaker100.md`, 0.6B):
8-bit runs about 2.4x and 4-bit about 2.7x faster than fp16 on a 10 s clip.
Per-sample JSON for this artifact's evaluation is committed in the
mlx-qwen3-asr repository as
`docs/benchmarks/2026-09-19-quantized-artifacts-librispeech-test-clean-100-0.6B_4bit.json`.

## Reproduce

```bash
git clone https://github.com/moona3k/mlx-qwen3-asr && cd mlx-qwen3-asr && git checkout v0.4.3
python scripts/convert.py --model Qwen/Qwen3-ASR-0.6B --quantize 4 --encoder-bits 8 \
  --group-size 64 --dtype float16 --output-dir Qwen3-ASR-0.6B-4bit-g64
python scripts/eval_librispeech.py --model Qwen3-ASR-0.6B-4bit-g64 --samples 100 --sampling speaker_round_robin
```

## License

Apache-2.0, following the source model.
