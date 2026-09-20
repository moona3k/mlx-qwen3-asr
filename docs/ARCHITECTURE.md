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

**Tail chunk padding.** The official encoder runs `pad_sequence` over a
sample's chunks before the conv stem, then keeps only the first
`ceil(tail_frames / 8)` post-CNN tokens of the tail. A tail that follows at
least one full chunk is therefore convolved at width 100, not at its own
width; because the conv biases are non-zero and GELU follows each layer, the
two are not equivalent and the last valid tail token differs. `_encode_single`
mirrors this: it right-pads such tails to `chunk_size` and crops the output.
An input shorter than one chunk is convolved at its own width, matching
`pad_sequence` on a single element.

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

## Streaming Decode

`streaming.py` follows the official `qwen_asr` streaming recipe (Decision 29).
The model is only ever prompted the way it was trained: one system turn, one
user turn holding *all* the audio seen so far, one assistant turn.

```
chunk k arrives
  window  = audio_accum + chunk            (bounded by max_context_sec)
  prompt  = <system><user: audio(window)><assistant>
  prefix  = raw generated ids from step k-1, minus the last unfixed_token_num
            (backed off further if the cut lands inside a multi-byte character)
  prefill(prompt + prefix)  ->  greedy decode the remainder
  raw_ids = prefix + new ids                (kept for step k+1)
  text    = parse(decode(raw_ids))
```

- The first `unfixed_chunk_num` chunks of a window decode with no prefix.
- The prefix makes partial text stable by construction; only the trailing
  `unfixed_token_num` tokens can change, which is what `rewrite_rate` counts.
- When the next chunk would overflow `max_context_sec`, the window's text is
  committed to `committed_text` and a new window starts with that chunk.
  `state.text` is `committed_text` joined with the live window text. A word
  cut at the boundary can be duplicated or dropped once per window.
- Per-chunk cost is bounded by the window (encoder over <= 30 s plus a
  prefill of ~12.5 audio tokens/s and the prefix), not by session length.
- Prefix reuse (`reuse_window_prefix=True`, Decision 30): the encoder's
  attention windows are 800 mel frames (8 s, 104 tokens) and never attend
  across each other; the conv stem and position embeddings are per 100-frame
  chunk. So the encoder output of every complete 800-frame block is final,
  and because the decoder is causal, so are the KV entries for the prompt
  head and those blocks' audio tokens. `_WindowPrefixCache` keeps both; each
  chunk encodes only the trailing partial block and prefills the prompt tail
  plus the text prefix onto a `KVCache.fork()` of the cached KV. The log-mel
  clamps against the window's global max, so the cache is keyed on that max
  and rebuilt when a louder chunk raises it. A block counts as complete only
  when at least one frame follows it (the last STFT frame reaches 200 samples
  past the block). Two-stage prefill is bit-identical to the single pass;
  block-wise encoding differs by reduction order only (max 1.6e-6). Measured
  RTF 0.104 -> 0.083 on the long-form lane and 0.090 -> 0.081 on
  multilingual-100; generation (~55% of per-chunk time) is the floor.

The earlier design encoded each 2 s chunk alone and appended it to a live
decoder KV cache as a follow-up chat turn. It is linear in cost but the model
never saw that prompt shape in training: 56% primary error vs 9.5% offline on
the multilingual-100 lane (`docs/benchmarks/2026-09-19-streaming-manifest-*`).

## Quantization Layout

`nn.quantize` (affine, group size 64) is applied to every `Linear` and
`Embedding`. Each quantized module stores `weight` (packed `uint32`),
`scales` and `biases`; norms, conv stem and other floats stay in the
activation dtype (float16 by default).

- **Widths are per module.** The loader reads `bits` and `group_size` from
  each module's packed shapes (`bits = packed_cols * 32 / in_features`,
  `group_size = in_features / scale_cols`) and calls `nn.quantize` once per
  distinct pair; `quantization_config.json` is only the fallback. This is what
  lets an 8-bit audio encoder sit next to a 4-bit decoder.
- **Published recipe.** `--quantize 4 --encoder-bits 8`: on 0.6B the audio
  encoder carried most of the all-4-bit loss (2.63% -> 2.37% WER, fp16 2.33%).
  8-bit throughout is hypothesis-identical to fp16 on the 100-clip lane.
- **Tied `lm_head`.** Both source models tie `lm_head` to `embed_tokens`;
  artifacts drop `lm_head.*` and `_materialize_tied_lm_head_weights` recreates
  it at load, including the quantized `scales`/`biases`.
- **Activation dtype.** After loading, every floating parameter, quantized or
  not, is cast to the requested dtype. Community checkpoints store bf16
  floats; before this cast (0.4.1) they silently promoted the whole forward
  pass to float32.

## Key Constants

```python
SAMPLE_RATE = 16000          # audio.py
N_FFT = 400
HOP_LENGTH = 160
NUM_MEL_BINS = 128
MAX_CHUNK_SECONDS = 30.0     # chunking.py: long audio is split at low-energy points
MIN_CHUNK_SECONDS = 0.5
REPETITION_THRESHOLD = 20    # generate.py
AUTO_MAX_NEW_TOKENS_PER_SECOND = 12.0  # generate.py: per-chunk decode budget
```

There is no fixed input-length ceiling in the library; the server enforces
`--max-duration` (default 8 hours).
