# mlx-qwen3-asr

[![PyPI version](https://img.shields.io/pypi/v/mlx-qwen3-asr.svg)](https://pypi.org/project/mlx-qwen3-asr/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://pypi.org/project/mlx-qwen3-asr/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

Run [Qwen3-ASR](https://huggingface.co/collections/Qwen/qwen3-asr) — one of the strongest open-source speech recognition models — natively on Apple Silicon.

A ground-up reimplementation of the [official PyTorch model](https://github.com/QwenLM/Qwen3-ASR) using Apple's [MLX](https://github.com/ml-explore/mlx) framework. Same weights, benchmarked against official/reference outputs and ground-truth eval sets, optimized for Mac GPUs via Metal. No PyTorch dependency for core transcription.

## Why this exists

[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) is one of the strongest open-source ASR models available, with benchmark results exceeding Whisper-large-v3 across multiple languages and datasets. It supports 30 languages plus 22 Chinese dialects. But the official implementation is **PyTorch + NVIDIA CUDA** — it doesn't use Apple GPUs.

This project rewrites every layer for MLX so the same model runs natively on M1/M2/M3/M4 hardware. Not a wrapper — a full reimplementation with correct interleaved MRoPE, per-chunk windowed encoder attention, and all the architectural details that matter for output quality.

### What's included

- **Full encoder-decoder pipeline** — audio encoder (Conv2d stem + windowed transformer) and text decoder (Qwen3-style with interleaved MRoPE), reimplemented from scratch for MLX
- **Whisper-compatible mel frontend** — native log-mel spectrogram computation with cached filterbank and Hann window
- **Both model sizes** — 0.6B (fast, default) and 1.7B (higher accuracy)
- **Long audio support** — energy-based chunking up to 20 minutes per chunk, no 30-second feature truncation
- **Word-level timestamps** — native MLX forced aligner (default, 2.6x faster than PyTorch alternative) with O(n log n) LIS-based timestamp correction
- **Speaker diarization (optional)** — offline speaker-labeled outputs via `pyannote` integration (`--diarize`)
- **4-bit and 8-bit quantization** — up to 4.7x speedup with measured quality reporting on 100 speaker-balanced samples
- **Multiple output formats** — txt, json, srt, vtt, tsv
- **Session API** — explicit model/tokenizer ownership with no hidden global state
- **Speculative decoding** — experimental opt-in path (0.6B drafts for 1.7B target), parity-verified
- **Streaming** — KV-cache streaming with linear complexity, context trimming, and tail refinement
- **Native WAV fast-path** — custom binary WAV parser bypasses ffmpeg for PCM/float WAV files
- **462 tests** — every optimization is benchmark-gated with committed JSON artifacts
- **Minimal dependencies** — mlx, numpy, regex, huggingface-hub

## Requirements

- **Apple Silicon Mac** (M1/M2/M3/M4) — this is an MLX project, Metal GPU required
- **Python 3.10+**
- **ffmpeg** — required for non-WAV audio formats (mp3, m4a, flac, mp4, etc.). WAV files work without ffmpeg via the native fast-path loader
- **~1.2 GB memory** for 0.6B model (fp16), **~3.4 GB** for 1.7B

## Installation

Install from PyPI:

```bash
pip install mlx-qwen3-asr
```

For video and most non-WAV audio formats, install `ffmpeg` on your system:

```bash
brew install ffmpeg
```

Install with optional timestamp alignment extras (for Japanese/Korean tokenization parity):

```bash
pip install "mlx-qwen3-asr[aligner]"
```

Install with optional microphone capture support:

```bash
pip install "mlx-qwen3-asr[mic]"
```

Install with diarization extras:

```bash
pip install "mlx-qwen3-asr[diarize]"
```

Note: `--diarize` uses `pyannote` models. The default model is gated on
Hugging Face, so you usually need to accept model terms and set a token:

```bash
export PYANNOTE_AUTH_TOKEN=hf_...
```

Core ASR does not require any Hugging Face token.

For development:

```bash
git clone https://github.com/moona3k/mlx-qwen3-asr.git
cd mlx-qwen3-asr
pip install -e ".[dev]"
```

## Quick start

### Python API

```python
from mlx_qwen3_asr import transcribe

result = transcribe("audio.wav")
print(result.text)
print(result.language)
```

By default, `transcribe()` uses `Qwen/Qwen3-ASR-0.6B` for fast local usage on Mac. Use `Qwen/Qwen3-ASR-1.7B` when you want higher accuracy and can afford higher latency/memory.

With options:

```python
result = transcribe(
    "meeting.mp3",
    model="Qwen/Qwen3-ASR-1.7B",
    language="English",
    return_chunks=True,
    on_progress=lambda e: print(e["event"], e.get("progress", 0.0)),
    verbose=True,
)
print(result.text)
print(result.chunks)
```

### Session API (recommended for repeated calls)

The `Session` object owns model and tokenizer state explicitly — no hidden globals, no cache surprises:

```python
from mlx_qwen3_asr import Session

session = Session(model="Qwen/Qwen3-ASR-0.6B")

# Fast repeated transcription — model stays loaded
for audio_file in audio_files:
    result = session.transcribe(audio_file)
    print(result.text)
```

### Loading models explicitly

```python
from mlx_qwen3_asr import load_model, load_audio, transcribe

model, config = load_model("Qwen/Qwen3-ASR-0.6B")
audio = load_audio("speech.wav")
result = transcribe(audio, model=model)
```

### CLI

```bash
mlx-qwen3-asr audio.wav
```

Specify model, language, and output format:

```bash
mlx-qwen3-asr recording.mp3 --model Qwen/Qwen3-ASR-0.6B --language English -f srt -o output/
```

Word-level timestamps:

```bash
mlx-qwen3-asr audio.wav --timestamps
```

Speaker-labeled output (experimental, offline):

```bash
mlx-qwen3-asr meeting.wav --diarize --num-speakers 2 -f json
```

Multiple files with all output formats:

```bash
mlx-qwen3-asr *.wav -f all -o transcripts/ --verbose
```

Stdout/file behavior:

```bash
mlx-qwen3-asr audio.wav --stdout-only        # print only (no output file)
mlx-qwen3-asr audio.wav --quiet -o out/      # write files only (no stdout text)
```

Language discovery:

```bash
mlx-qwen3-asr --list-languages
```

Environment diagnostics (ffmpeg, optional diarization deps, token status):

```bash
mlx-qwen3-asr --doctor
```

Run `mlx-qwen3-asr --help` for the full list of options.

## Performance on Apple Silicon

Measured on Apple M4 Pro, macOS 26.2. All numbers from controlled runs with benchmark JSON artifacts committed to the repo. See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for the full breakdown.

### Latency (0.6B)

| Configuration | Short clip (~2.5s) | 10s clip | Real-time factor | vs fp16 |
|---|---|---|---|---|
| **fp16** (baseline) | 0.46s | 0.83s | 0.08x | — |
| **8-bit** (q8, group 64) | 0.11s | 0.27s | 0.03x | 3.11x faster |
| **4-bit** (q4, group 64) | 0.13s | 0.18s | 0.02x | **4.68x faster** |

### English quality refresh (LibriSpeech, 100 speaker-balanced samples per subset)

| Model | Subset | WER | CER | Mean eval latency | RTF |
|---|---|---|---|---|---|
| 0.6B | test-clean | 2.29% | 0.59% | 0.86s | 0.0957 |
| 0.6B | test-other | 4.20% | 2.09% | 0.71s | 0.0985 |
| 1.7B | test-clean | 1.99% | 0.61% | 2.43s | 0.2708 |
| 1.7B | test-other | 3.45% | 1.42% | 2.02s | 0.2814 |

Artifacts: `docs/benchmarks/2026-02-15-librispeech-test-clean-100.json`, `docs/benchmarks/2026-02-15-librispeech-test-other-100.json`, `docs/benchmarks/2026-02-15-librispeech-test-clean-100-1p7b.json`, `docs/benchmarks/2026-02-15-librispeech-test-other-100-1p7b.json`.

### Quantization quality (0.6B, LibriSpeech test-clean, 100 speaker-balanced samples)

| Configuration | WER | CER | Mean eval latency | vs fp16 Speed |
|---|---|---|---|---|
| fp16 | 2.29% | 0.59% | 1.09s | — |
| 8-bit (g64) | 2.33% | 0.59% | 0.34s | 3.11x |
| 4-bit (g64) | 2.72% | 0.88% | 0.30s | 4.68x |

8-bit is near-fp16 quality (+0.04pp WER). 4-bit trades +0.43pp WER for maximum speed.

On the harder `test-other` lane (`n=100`, speaker-balanced), 8-bit remains near-fp16
(-0.05pp WER) while 4-bit shows a larger quality tradeoff (+1.38pp WER). Speedups
remain strong (3.66x for 8-bit, 4.37x for 4-bit on the 10s benchmark clip).

Artifact: `docs/benchmarks/2026-02-15-quant-matrix-test-other-speaker100.md`.

### Multilingual quality (FLEURS, 10 languages x 10 samples)

| Model | Primary Error Rate | Mean Latency | Best Languages | Weakest |
|---|---|---|---|---|
| **0.6B** fp16 | 9.37% | 1.44s | Spanish 3.0%, Chinese 4.4%, English 4.6% | Hindi 16.7%, French 18.2%, Arabic 21.5% |
| **1.7B** fp16 | **6.70%** | 4.12s | Spanish 0.7%, Japanese 3.6%, French 4.1% | Chinese 8.5%, Arabic 16.5%, Hindi 17.4% |

The 1.7B delivers a 28% relative improvement, with the biggest gains on French (-14.1pp), Japanese (-4.9pp), and Arabic (-5.0pp). The 1.7B runs ~2.86x slower.

Artifacts: `docs/benchmarks/2026-02-15-manifest-quality-multilingual100-0p6b-refresh.json`, `docs/benchmarks/2026-02-15-manifest-quality-multilingual100-1p7b-refresh.json`.

### MLX vs PyTorch quality (0.6B, Multilingual-100)

| Metric | MLX | PyTorch | Delta |
|---|---:|---:|---:|
| Primary error rate | 9.54% | 10.34% | -0.81pp |
| WER | 16.00% | 16.69% | -0.70pp |
| CER | 5.43% | 5.64% | -0.21pp |

67% of samples produce identical text output. Remaining differences are minor lexical shifts, numeric surface forms (`10,000` vs `zehntausend`), or punctuation — not quality regressions.

On long-form audio (75-90s clips), **MLX is 4.19x faster** than PyTorch on the same machine.

On an expanded real-world mixed lane (AMI IHM meetings + Earnings22 chunked,
`n=200`), MLX remains near parity with PyTorch (**23.23%** vs **23.04%** WER,
`+0.19pp` delta) while staying **3.27x faster** on the same machine
(`1.34s` vs `4.39s` mean latency).

### Optimizations applied

- **Preallocated KV cache** with in-place slice writes and rollback-safe trimming
- **Direct grouped-query fused attention** via `mx.fast.scaled_dot_product_attention` (no explicit K/V head expansion)
- **Hybrid encoder windowing** — dense block-diagonal mask for short audio, segmented per-window execution for long contexts (up to 4.2x faster on long audio)
- **Cached mel filterbank and Hann window** — computed once, reused across calls
- **Native WAV fast-path** — custom binary parser bypasses ffmpeg process startup for PCM/float WAV files (up to 25% faster on quantized short clips)
- **Native in-repo BPE tokenizer** — no `transformers` dependency in runtime transcription path
- **Cached model and tokenizer instances** — repeated `transcribe()` calls skip reload overhead
- **4-bit / 8-bit quantization** — up to 4.7x speed gain with explicit per-profile quality reporting

Full benchmark report: `docs/BENCHMARKS.md`. Latest refresh snapshot: `docs/benchmarks/2026-02-15-quality-matrix-refresh.md`. All benchmark artifacts are committed under `docs/benchmarks/` for reproducibility.

## Model quality

Word error rates from the [Qwen3-ASR technical report](https://arxiv.org/abs/2601.21337) compared against current open-source and proprietary leaders (lower is better):

### English benchmarks

| Benchmark | GPT-4o-Transcribe | Parakeet-TDT-0.6B | Whisper-large-v3 | Qwen3-ASR-0.6B | **Qwen3-ASR-1.7B** |
|---|---:|---:|---:|---:|---:|
| LibriSpeech test-clean | **1.39** | 1.93 | 1.51 | 2.11 | 1.63 |
| LibriSpeech test-other | 3.75 | 3.59 | 3.97 | 4.55 | **3.38** |
| FLEURS-en | 2.40 | 4.85 | 4.08 | 4.39 | **3.35** |
| GigaSpeech | 25.50 | — | 9.76 | 8.88 | **8.45** |

### Chinese + multilingual benchmarks

| Benchmark | GPT-4o-Transcribe | Whisper-large-v3 | Qwen3-ASR-0.6B | **Qwen3-ASR-1.7B** |
|---|---:|---:|---:|---:|
| WenetSpeech test-net | 15.30 | 9.86 | 5.97 | **4.97** |
| AISHELL-2 test | 4.24 | 5.06 | 3.15 | **2.71** |
| FLEURS (12-lang avg) | — | 5.27 | 7.57 | **4.90** |
| CommonVoice | — | 10.77 | 12.75 | **9.18** |

### Robustness benchmarks

| Benchmark | GPT-4o-Transcribe | Whisper-large-v3 | Qwen3-ASR-0.6B | **Qwen3-ASR-1.7B** |
|---|---:|---:|---:|---:|
| Accented English | 28.56 | 21.30 | 16.62 | **16.07** |
| Extreme Noise | 36.11 | 63.17 | 17.88 | **16.17** |
| Elders & Kids (Mandarin) | 14.27 | 10.61 | 4.48 | **3.81** |

GPT-4o-Transcribe leads on clean English read speech (1.39 WER). Parakeet-TDT-0.6B is strong on English. But Qwen3-ASR dominates on Chinese, multilingual, noisy, and accented speech — and is the only open-source model competitive across all categories.

*Parakeet numbers from [model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3). All other numbers from the Qwen3-ASR paper. Robustness benchmarks are Qwen3-ASR internal test sets.*

### Correctness validation

This implementation is validated against the official PyTorch model via multiple parity gates:

- **MLX vs PyTorch head-to-head** — on the current multilingual-100 artifact, MLX shows lower aggregate primary error than PyTorch (9.54% vs 10.34%)
- **Token-level greedy parity** — current multilingual-100 parity artifact shows 67% exact text match and 64% exact token match across 10 languages; remaining diffs are mostly lexical/numeric surface-form differences
- **Expanded parity suite** — tested across LibriSpeech test-clean, test-other, synthetic long mixes, and noise variants (SNR 10dB, 5dB)
- **Long-form parity** — 10 multilingual clips (75-90s each) transcribed correctly with no chunking artifacts, 4.19x faster than PyTorch
- **Mel spectrogram parity** — custom MLX mel matches HuggingFace WhisperFeatureExtractor with MAE < 3e-7
- **Native aligner parity** — MLX forced aligner matches official `qwen-asr` backend with 100% text match rate, <6ms timing MAE, and 2.64x speed advantage on 50 LibriSpeech samples

## Model variants

| | Qwen3-ASR-0.6B (default) | Qwen3-ASR-1.7B |
|---|---|---|
| **Parameters** | 0.6B | 1.7B |
| **Audio encoder layers** | 18 | 24 |
| **Audio encoder dim** | 896 | 1024 |
| **Text decoder layers** | 28 | 28 |
| **Text hidden size** | 1024 | 2048 |
| **Text attention (Q/KV heads)** | GQA (16/8) | GQA (16/8) |
| **RoPE theta** | 1,000,000 | 1,000,000 |
| **HuggingFace** | `Qwen/Qwen3-ASR-0.6B` | `Qwen/Qwen3-ASR-1.7B` |

Both models use interleaved Multi-dimensional RoPE (MRoPE) with sections [24, 20, 20], 128-bin mel spectrograms, and the same tokenizer (vocabulary size 151,936).

```python
# Default: 0.6B (fast, ~1.2 GB memory)
result = transcribe("audio.wav")

# Accuracy-first: 1.7B (~3.4 GB memory)
result = transcribe("audio.wav", model="Qwen/Qwen3-ASR-1.7B")
```

## Timestamps

Word-level timestamps via forced alignment using a dedicated aligner model (`Qwen/Qwen3-ForcedAligner-0.6B`). This path is native MLX (no PyTorch backend bridge):

```bash
mlx-qwen3-asr audio.wav --timestamps
```

```python
result = transcribe("audio.wav", return_timestamps=True)
for segment in result.segments:
    print(f"{segment['start']:.2f}s - {segment['end']:.2f}s: {segment['text']}")
```

SRT/VTT outputs are grouped into subtitle-friendly phrase segments (not one word per cue).
When `-f srt` or `-f vtt` is requested in offline mode, timestamps are auto-enabled.

**Measured parity** (LibriSpeech test-clean, `n=50`):
| Metric | Value |
|---|---|
| Text match rate (MLX vs official) | 100% |
| Timing MAE (all word boundaries) | 5.69 ms |
| MLX aligner mean latency | 0.21s |
| Official backend mean latency | 0.56s |
| Relative speed | **2.64x faster** |

The aligner uses O(n log n) LIS-based timestamp correction (Fenwick tree) for monotonicity repair, validated against the legacy O(n^2) implementation via randomized parity tests.

For Japanese/Korean timestamp alignment, install the `[aligner]` extra so `nagisa`/`soynlp` tokenization matches the official path.

## Speaker diarization (optional)

Speaker attribution is available as an offline optional path powered by
`pyannote.audio`:

```python
result = transcribe("meeting.wav", diarize=True)
print(result.speaker_segments)
```

```bash
mlx-qwen3-asr meeting.wav --diarize -f json
```

Current status:
- The public API/CLI and output schema are stable.
- The diarization backend is `pyannote` (installed via `[diarize]` extra).
- Some pyannote models require Hugging Face token/terms acceptance. Configure
  `PYANNOTE_AUTH_TOKEN` (or `HF_TOKEN`) when needed.
- `--diarize` auto-enables timestamps and is not supported in `--streaming`/`--mic` mode.
- Migration note (2026-02-15): legacy diarization `window/hop` controls were
  removed (`diarization_window_sec`, `diarization_hop_sec`,
  `--diarization-window-sec`, `--diarization-hop-sec`). Speaker-count controls
  remain (`--num-speakers`, `--min-speakers`, `--max-speakers`).

### Diarization setup troubleshooting

1. Install optional diarization dependencies:
   ```bash
   pip install "mlx-qwen3-asr[diarize]"
   ```
2. Set a Hugging Face token if your selected diarization model is gated:
   ```bash
   export PYANNOTE_AUTH_TOKEN=hf_...
   ```
3. Run a quick smoke test:
   ```bash
   mlx-qwen3-asr meeting.wav --diarize -f json
   ```

Common errors and fixes:
- `requires optional dependency 'pyannote.audio'`: install `[diarize]` extra.
- `requires PyTorch via pyannote dependencies`: reinstall `[diarize]` extra in the active environment.
- `Failed to initialize pyannote pipeline ...`: accept model terms on Hugging Face and set `PYANNOTE_AUTH_TOKEN` (or `HF_TOKEN`).
- `--streaming does not support --diarize` / `--mic does not support --diarize`: use offline file transcription mode for diarization.

## Quantization

Convert and run a quantized model:

```bash
python scripts/convert.py \
  --model Qwen/Qwen3-ASR-0.6B \
  --quantize 4 --group-size 64 \
  --output-dir ./qwen3-asr-4bit

mlx-qwen3-asr audio.wav --model ./qwen3-asr-4bit
```

Recommended profiles:
- **Speed-first**: 4-bit, group_size=64 — 4.68x faster / +0.43 WER (`test-clean`), 4.37x faster / +1.38 WER (`test-other`)
- **Quality-first**: 8-bit, group_size=64 — 3.11x faster / +0.04 WER (`test-clean`), 3.66x faster / -0.05 WER (`test-other`)

Publish quantized models to HuggingFace:

```bash
HF_TOKEN=... python scripts/publish_quantized.py \
  --source-model Qwen/Qwen3-ASR-0.6B \
  --repo-id YOUR_USER/mlx-qwen3-asr-0.6b-4bit \
  --bits 4
```

## Output formats

```bash
mlx-qwen3-asr audio.wav -f txt           # plain text
mlx-qwen3-asr audio.wav -f srt -o out/   # SRT subtitles
mlx-qwen3-asr audio.wav -f json          # structured JSON
mlx-qwen3-asr audio.wav -f vtt -o out/   # WebVTT
mlx-qwen3-asr *.wav -f all -o out/       # all formats at once
```

Supported: `txt`, `json`, `srt`, `vtt`, `tsv`.
Subtitle formats (`srt`/`vtt`) require timestamp segments and are only supported in offline mode.

## Supported languages

Qwen3-ASR officially lists 30 core languages:

| | | | |
|---|---|---|---|
| Arabic | Cantonese | Chinese | Czech |
| Danish | Dutch | English | Filipino |
| Finnish | French | German | Greek |
| Hindi | Hungarian | Indonesian | Italian |
| Japanese | Korean | Macedonian | Malay |
| Persian | Polish | Portuguese | Romanian |
| Russian | Spanish | Swedish | Thai |
| Turkish | Vietnamese | | |

Plus 22 Chinese dialects (Sichuan, Shanghai, Cantonese, and others), for 52 total language/dialect variants.

Print CLI-accepted aliases/codes:

```bash
mlx-qwen3-asr --list-languages
```

## Experimental features

### Speculative decoding

Uses the 0.6B model as a draft to accelerate 1.7B inference. Currently parity-safe but slower on tested workloads due to draft audio encoder overhead:

```bash
mlx-qwen3-asr audio.wav \
  --model Qwen/Qwen3-ASR-1.7B \
  --draft-model Qwen/Qwen3-ASR-0.6B \
  --num-draft-tokens 4
```

```python
result = transcribe(
    "audio.wav",
    model="Qwen/Qwen3-ASR-1.7B",
    draft_model="Qwen/Qwen3-ASR-0.6B",
    num_draft_tokens=4,
)
```

Status: greedy parity verified, but 0.53-0.55x on short/10s clips. Not enabled by default until benchmark evidence shows net speed wins.

### Domain vocabulary context

When transcribing specialized audio — earnings calls, medical dictation, legal
proceedings — the model can confuse rare terms with more common homophones.
The `context` parameter lets you provide a hint: a string of domain-specific
words or phrases that gets injected into the system prompt, nudging the decoder
toward the correct vocabulary.

This matches the official Qwen3-ASR `context` API. The format is
space-separated terms:

```python
# Finance: avoids "e-bit-da" → "EBITDA", "FX" not "effects", etc.
result = transcribe("earnings-call.wav", context="EBITDA non-GAAP FX hedging")

# Medical
result = transcribe("consult.wav", context="metformin HbA1c nephropathy")

# Also works with streaming
state = init_streaming(context="EBITDA non-GAAP FX hedging")
```

```bash
mlx-qwen3-asr earnings-call.wav --context "EBITDA non-GAAP FX hedging"
```

For batch transcription, pass a list of per-audio context strings:

```python
results = transcribe_batch(
    [audio_en, audio_zh],
    context=["EBITDA non-GAAP", "交易 停滞"],
)
```

When omitted, the system prompt is empty (matching the official default) — no
domain bias is applied.

### Streaming

Rolling decode implementation for near-real-time transcription:

```python
from mlx_qwen3_asr.streaming import (
    init_streaming,
    feed_audio,
    finish_streaming,
    streaming_metrics,
)

state = init_streaming(chunk_size_sec=2.0, max_context_sec=30.0)
for chunk in audio_chunks:
    state = feed_audio(chunk, state)
    print(state.text)
state = finish_streaming(state)
print(streaming_metrics(state))
```

CLI:
```bash
mlx-qwen3-asr --streaming --stream-finalization-mode accuracy audio.wav
# Optional: speech-aware boundary selection near chunk edges
mlx-qwen3-asr --streaming --stream-endpointing-mode energy audio.wav
```

Live microphone transcription:

```bash
mlx-qwen3-asr --mic
mlx-qwen3-asr --mic --language Japanese
```

Optional microphone flags: `--mic-device`, `--mic-duration-sec`, `--mic-sample-rate`.

- Ingests small PCM chunks (default 2s)
- Incremental decoder KV-cache reuse across chunk turns (avoids O(n²) re-transcription)
- Bounded context window (default 30s) for stable memory/runtime
- Prefix rollback controls (`unfixed_chunk_num`, `unfixed_token_num`)
- `stable_text` is monotonic by design: corrections that would shorten already-stable
  prefix text are intentionally not applied to the stable prefix (favoring stability
  over maximal editability in partial output)
- Optional speech-aware endpointing (`endpointing_mode="energy"`) that selects
  low-energy boundaries near chunk edges
- Configurable finalization policy: `finalization_mode="accuracy"` (default) or `"latency"`
- Backward-compatible override: `enable_tail_refine=True|False`
- Input validation: handles int16 PCM normalization, non-1D arrays, empty input

## API reference

### `transcribe(audio, *, model, draft_model, context, language, return_timestamps, diarize, diarization_num_speakers, diarization_min_speakers, diarization_max_speakers, return_chunks, forced_aligner, dtype, max_new_tokens, num_draft_tokens, verbose, on_progress)`

Transcribe audio to text. Accepts a file path, numpy array, `mx.array`, or `(array, sample_rate)` tuple. Returns a `TranscriptionResult`.

Additional Python entry points:
- `transcribe_batch(audios, ...)` and `transcribe_batch_async(audios, ...)`
- `transcribe_async(audio, ...)`

### `Session(model, *, dtype, tokenizer_model)`

Explicit transcription session. Owns model and tokenizer state with no hidden globals.
- Offline: `session.transcribe(audio, ...)` with the same parameters as top-level `transcribe`.
- Async: `await session.transcribe_async(audio, ...)`.
- Streaming: `session.init_streaming(...)`, `session.feed_audio(pcm, state)`, `session.finish_streaming(state)`.
- Introspection: `session.model_info` (model id/path, dtype, vocab size, model-declared language codes).

### `streaming_metrics(state)`

Return streaming diagnostics for a session state:
- `partial_stability`
- `rewrite_rate`
- `finalization_delta_chars`

### `load_model(name_or_path, *, dtype)`

Load a Qwen3-ASR model and config from HuggingFace or local path. Returns `(model, config)`.

### `load_audio(path_or_url)`

Load and resample audio to mono 16 kHz. Returns an `mx.array`.

### `ForcedAligner(model_path, *, dtype, backend)`

Word-level forced aligner. Native backend: `mlx` (default).

### `TranscriptionResult`

Frozen dataclass:
- `text` (str) — transcribed text
- `language` (str) — detected or forced language (canonicalized names, e.g. `English`)
- `segments` (list[dict] | None) — word-level timestamps when requested: `[{"text": "hello", "start": 0.5, "end": 0.8}, ...]`
- `chunks` (list[dict] | None) — chunk-level transcript metadata when `return_chunks=True`
- `speaker_segments` (list[dict] | None) — speaker-attributed spans when `diarize=True`: `[{"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0, "text": "..."}, ...]`

## Quality gates

This project enforces parity with the official PyTorch implementation. No optimization lands without passing quality gates and committing benchmark artifacts.

```bash
# Unit tests (462 tests)
pytest -q

# Fast quality gate
python scripts/quality_gate.py --mode fast

# Release gate with token-level parity (downloads model weights)
RUN_REFERENCE_PARITY=1 python scripts/quality_gate.py --mode release

# Speaker-balanced WER evaluation (100 samples)
python scripts/eval_librispeech.py --subset test-clean --samples 100 --sampling speaker_round_robin

# Latency benchmark
python scripts/benchmark_asr.py tests/fixtures/test_speech.wav \
  --model Qwen/Qwen3-ASR-0.6B --runs 5 \
  --json-output docs/benchmarks/latest.json
```

Additional quality lanes available:
- **Aligner parity**: `RUN_ALIGNER_PARITY=1` — validates MLX aligner against official backend
- **Expanded parity suite**: `RUN_REFERENCE_PARITY_SUITE=1` — test-clean, test-other, long mixes, noise variants with Unicode-safe text comparison
- **Multilingual parity**: manifest-driven workflow via `scripts/build_multilingual_manifest.py` for cross-language validation
- **Streaming manifest quality**: `RUN_STREAMING_MANIFEST_QUALITY_EVAL=1` with `STREAMING_MANIFEST_QUALITY_EVAL_JSONL=...` — multi-file streaming stability/rewrite/finalization lane via `scripts/eval_streaming_manifest.py`
- **Real-world long-form quality**: `RUN_REALWORLD_LONGFORM_EVAL=1` on full-recording Earnings22 manifests
- **Diarization quality**: `RUN_DIARIZATION_QUALITY_EVAL=1` with `DIARIZATION_QUALITY_EVAL_JSONL=...` — DER/JER lane via `scripts/eval_diarization.py`

See `docs/QUALITY_GATE.md` for full documentation.
Evaluation coverage status and prioritized gaps are tracked in `docs/EVAL_GAPS.md`.

## Architecture overview

```
Audio (16kHz mono)
  → 128-bin log-mel spectrogram (native MLX, Whisper-compatible)
  → Conv2d stem (3 layers, stride 2 each → 8x downsample)
  → Sinusoidal position embeddings
  → Windowed transformer encoder (18 or 24 layers, hybrid dense/segmented attention)
  → LayerNorm + GELU projection → audio features

Chat-template prompt (context is optional domain vocabulary, empty by default):
  <|im_start|>system\n{context}<|im_end|>
  <|im_start|>user\n<|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>
  <|im_start|>assistant\n

  → Token embedding (151,936 vocab)
  → Replace audio_pad positions with encoded audio features
  → Qwen3 text decoder (28 layers, interleaved MRoPE, SwiGLU, RMSNorm)
  → Autoregressive decode with preallocated KV cache
  → Parse output: "language English<asr_text>transcribed text here"
```

Key architectural details:
- **Interleaved MRoPE** — sections [24, 20, 20] with stride-3 frequency assignment across temporal, height, and width dimensions. This is the detail other MLX ports get wrong (using standard RoPE or chunked assignment).
- **Audio encoder uses LayerNorm + bias** — different from the text decoder which uses RMSNorm without bias.
- **Q/K norms** — RMSNorm applied per-head on queries and keys before attention (Qwen3 innovation).

## Project structure

```
mlx_qwen3_asr/           # 7,602 lines of source
├── transcribe.py         # High-level pipeline + batch/async + diarization (739 lines)
├── cli.py                # CLI entry point and UX guardrails (664 lines)
├── streaming.py          # KV-cache streaming + context trimming (624 lines)
├── tokenizer.py          # Native BPE tokenizer + output parsing (607 lines)
├── diarization.py        # Optional pyannote integration + attribution helpers
├── audio.py              # Mel spectrogram + audio I/O (526 lines)
├── encoder.py            # Audio encoder (512 lines)
├── decoder.py            # Text decoder + KV cache (464 lines)
├── forced_aligner.py     # Forced alignment + LIS correction (439 lines)
├── model.py              # Top-level model + audio-text fusion (372 lines)
├── generate.py           # Autoregressive + speculative decode (350 lines)
├── load_models.py        # Model loading + caching (256 lines)
├── config.py             # Dataclass configs (228 lines)
├── session.py            # Session API (224 lines)
├── writers.py            # Output format writers (221 lines)
├── mrope.py              # Interleaved MRoPE (167 lines)
├── chunking.py           # Long audio splitting (104 lines)
├── attention.py          # Attention utilities (67 lines)
├── convert.py            # Weight remapping (67 lines)
├── eval_metrics.py       # WER/CER/BERTScore helpers (65 lines)
└── cache_utils.py        # KV cache utilities (57 lines)

tests/                    # 7,391 lines, 462 tests
scripts/                  # Benchmarks, evaluation, conversion, publishing
docs/                     # Architecture, decisions, benchmarks, roadmap
docs/benchmarks/          # 160+ committed artifacts for reproducibility
```

## Development

```bash
git clone https://github.com/moona3k/mlx-qwen3-asr.git
cd mlx-qwen3-asr
pip install -e ".[dev]"
pytest -q                 # 462 tests
```

## Acknowledgments

- [Qwen team](https://github.com/QwenLM) at Alibaba for the Qwen3-ASR model
- [Apple MLX team](https://github.com/ml-explore/mlx) for the MLX framework
- [mlx-whisper](https://github.com/ml-explore/mlx-examples) for architecture patterns and inspiration

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
