# Architecture: Qwen3-ASR

Deep dive into the Qwen3-ASR model architecture.

## High-Level Architecture

```
Audio (16kHz mono) → Mel Spectrogram (128 bins)
    → Conv2d stem (3 layers, stride 2 each → 8x downsample)
    → Sinusoidal position embeddings (NOT learned)
    → 24 Transformer encoder layers (bidirectional attention)
    → LayerNorm + GELU projection → audio features (2048-dim)

Text prompt: <system>...<user><audio_start><audio_pad>*N<audio_end>
    → Token embedding (151936 vocab)
    → Replace audio_pad positions with audio features
    → 28 Qwen3 decoder layers (MRoPE, SwiGLU, RMSNorm)
    → LM head → next token logits
```

## Audio Encoder

### Conv2d Stem (8x Downsample)

Three convolutional layers, each with stride 2:

| Layer | In Channels | Out Channels | Kernel | Stride | Padding |
|-------|------------|--------------|--------|--------|---------|
| conv2d1 | 1 | 480 | 3x3 | 2 | 1 |
| conv2d2 | 480 | 480 | 3x3 | 2 | 1 |
| conv2d3 | 480 | 480 | 3x3 | 2 | 1 |

Each followed by GELU activation.

**Input:** Mel spectrogram (batch, 1, n_frames, 128)
**After conv:** (batch, 480, n_frames/8, 128/8) = (batch, 480, time', 16)
**Reshape:** (batch, time', 480 x 16) = (batch, time', 7680)
**Linear projection:** (batch, time', 7680) -> (batch, time', 1024)

### Encoder Output Length Formula (Official)

Qwen3-ASR computes output lengths per 100-frame chunk (not by applying
3x stride-2 over the full sequence in one pass):

```text
input_lengths_leave = input_lengths % 100
feat_lengths = (input_lengths_leave - 1) // 2 + 1
output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
```

Implication: exact multiples of 100 frames produce 13 tokens per chunk
(e.g., 400 -> 52, 1000 -> 130).

### Sinusoidal Position Embeddings

Fixed (not learned) sinusoidal embeddings added after the conv stem:
- PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
- PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
- Maximum positions: 1500
- NOT stored in weight files -- computed at initialization

### Transformer Encoder Layers

24 layers, each with:
- **Pre-norm:** LayerNorm (with bias) -- NOT RMSNorm
- **Self-attention:** Bidirectional MHA, 16 heads, head_dim=64, WITH bias on projections
- **FFN:** Linear(1024->4096) + GELU + Linear(4096->1024), WITH bias
- **Residual connections** around attention and FFN

### Output Projection

After the transformer stack:
1. LayerNorm (ln_post)
2. Linear(1024->1024) + GELU (proj1)
3. Linear(1024->2048) (proj2)

Output: (batch, n_tokens, 2048) for 1.7B and 1024 for 0.6B.

## Text Decoder

### Interleaved MRoPE

Multi-dimensional Rotary Position Embedding -- the critical correctness component.

**Sections:** [24, 20, 20] for temporal, height, width dimensions
**Official interleaving rule:** build full temporal frequencies first, then
overwrite selected stride-3 indices for height and width.

**position_ids shape:** (batch, 3, seq_len) -- one row per spatial dimension
**Output cos/sin shape:** (batch, seq_len, head_dim=128)

### Q/K Norms

Qwen3 innovation: RMSNorm applied per-head on queries and keys before RoPE.
- q_norm: RMSNorm(head_dim=128)
- k_norm: RMSNorm(head_dim=128)

### SwiGLU MLP

For 1.7B:
```text
Linear(2048 -> 6144), SiLU gate, then Linear(6144 -> 2048)
```

For 0.6B:
```text
Linear(1024 -> 3072), SiLU gate, then Linear(3072 -> 1024)
```

### Decoder Layer Structure

Pre-norm with RMSNorm (NOT LayerNorm):
```
h = x + self_attn(rms_norm(x), cos, sin, mask, cache)
h = h + mlp(rms_norm(h))
```

## Audio-Text Fusion

Audio features are injected into the text embedding sequence:

1. Text prompt contains `<|audio_pad|>` placeholder tokens (token_id=151676)
2. Audio encoder produces features of shape (batch, n_audio_tokens, output_dim)
3. Text embeddings are computed for all tokens including placeholders
4. Placeholder positions are replaced with audio features
5. Combined sequence is processed by the text decoder

The replacement uses cumulative indexing:
- Find positions where input_ids == audio_token_id
- Map each position to the corresponding audio feature vector
- Use mx.where for efficient selection

## Model Configuration Comparison

### Audio Encoder

| Parameter | 1.7B | 0.6B |
|-----------|------|------|
| encoder_layers | 24 | 18 |
| encoder_attention_heads | 16 | 14 |
| encoder_ffn_dim | 4096 | 3584 |
| d_model | 1024 | 896 |
| head_dim | 64 | 64 |
| output_dim | 2048 | 1024 |
| n_window | 50 | 50 |
| n_window_infer | 800 | 800 |
| downsample_hidden_size | 480 | 480 |

### Text Decoder

| Parameter | 1.7B | 0.6B |
|-----------|------|------|
| vocab_size | 151936 | 151936 |
| hidden_size | 2048 | 1024 |
| intermediate_size | 6144 | 3072 |
| num_hidden_layers | 28 | 28 |
| num_attention_heads | 16 | 16 |
| num_key_value_heads | **8 (GQA)** | **8 (GQA)** |
| head_dim | 128 | 128 |
| max_position_embeddings | 65536 | 65536 |
| rope_theta | 1000000.0 | 1000000.0 |

**Key difference:** both use GQA (16/8); 1.7B is wider (`hidden_size=2048`, `intermediate_size=6144`) than 0.6B.

## Forced Aligner Architecture

The aligner (Qwen3-ForcedAligner-0.6B) is a separate model. Current
public HF config (`Qwen/Qwen3-ForcedAligner-0.6B`) specifies:

- **Audio encoder:** 24 layers, 16 heads, d_model=1024, output_dim=1024
- **Text decoder:** 28 layers, hidden_size=1024, GQA 16/8
- **Classification head:** Non-autoregressive, classify_num=5000 time bins
- **Time resolution:** timestamp_segment_time=80ms per classification unit
- **LIS correction:** Longest Increasing Subsequence for monotonic timestamps

## Weight Key Mapping

```
HuggingFace key                              -> MLX key
thinker.audio_tower.conv2d1.weight           -> audio_tower.conv2d1.weight (+ transpose)
thinker.audio_tower.conv2d1.bias             -> audio_tower.conv2d1.bias
thinker.audio_tower.layers.0.self_attn.*     -> audio_tower.layers.0.self_attn.*
thinker.model.layers.0.self_attn.q_proj.*    -> model.layers.0.self_attn.q_proj.*
thinker.model.embed_tokens.*                 -> model.embed_tokens.*
thinker.lm_head.*                            -> lm_head.*
```

All keys: strip `thinker.` prefix.
Conv2d weights: transpose (out, in, kH, kW) -> (out, kH, kW, in) via transpose(0, 2, 3, 1).

## Prompt Template

```
<|im_start|>system
{context}<|im_end|>
<|im_start|>user
<|audio_start|><|audio_pad|>...(N times)...<|audio_pad|><|audio_end|>
<|im_start|>assistant
```

The `context` string is injected as the system message content. It defaults to
empty (`""`), matching the official Qwen3-ASR implementation. Pass
space-separated domain terms (e.g., `"交易 停滞"`) to bias the decoder toward
specialized vocabulary.

Output format: `language {detected_language}<asr_text>{transcription text}`

## Key Constants

```python
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
NUM_MEL_BINS = 128
MAX_ASR_INPUT_SECONDS = 1200.0
MIN_ASR_INPUT_SECONDS = 0.5
REPETITION_THRESHOLD = 20
```
