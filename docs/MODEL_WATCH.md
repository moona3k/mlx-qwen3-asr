# ASR Model Watch

Last verified: 2026-05-16.

This is a lightweight watchlist for ASR model movement that could affect the
future direction of `mlx-qwen3-asr`. It is not a roadmap to support every model
listed here. The project remains scoped to being the best local Apple Silicon
runtime for Qwen3-ASR unless a future Qwen ASR successor or a clearly superior
Apple Silicon opportunity justifies a deliberate port.

## Current Decision

Keep `mlx-qwen3-asr` focused on Qwen3-ASR.

The market has moved in three directions that matter:

1. Dedicated ASR accuracy is tightening around 2B-class models.
2. Long-form meeting transcripts increasingly include speaker, timestamp,
   hotword, and code-switching structure.
3. Streaming-first models are optimizing configurable latency rather than only
   offline WER.

Project response:

1. Benchmark competing systems as external runners.
2. Do not turn this package into a generic ASR model zoo.
3. Invest in the surfaces where Qwen3-ASR can remain differentiated locally:
   Apple Silicon performance, long-form stability, streaming polish, timestamp
   quality, server compatibility, and package reliability.

## Evaluation Criteria

Use these criteria when deciding whether a model belongs in the watchlist,
benchmark harness, or a future port investigation.

| Criterion | What to record | Why it matters here |
|---|---|---|
| License | Apache 2.0, MIT, CC-BY-4.0, custom, closed API | Determines whether the model can be used in commercial local-first apps and whether a port can be redistributed. |
| Apple Silicon feasibility | Existing MLX/Core ML/llama.cpp path, safetensors shape, custom kernels, memory floor | This repo's north star is local Apple Silicon, not CUDA-first serving. |
| Transcript structure | Word timestamps, segment timestamps, diarization, hotwords, language ID, code switching | The competitive bar is moving toward structured long-form transcripts. |
| Languages | Supported languages and whether language must be explicit | Qwen3-ASR's 30 languages plus 22 Chinese dialects are a core strength. |
| WER or quality | Open ASR leaderboard WER, model-card benchmark, domain benchmark | Quality claims need comparable datasets before they affect roadmap decisions. |
| Latency | RTFx, streaming delay, TTFT, chunk size, hardware | The repo should defend local responsiveness, not only offline accuracy. |
| Memory and size | Parameter count, quantization availability, expected RAM/VRAM | Apple Silicon practicality often depends more on memory than raw model quality. |

## Priority Tiers

### Tier 0: Core Project Surface

| Model | Why it matters | Verified facts | Action |
|---|---|---|---|
| Qwen3-ASR-0.6B / Qwen3-ASR-1.7B | This is the model family this repo implements. | Apache 2.0. 52 languages and dialects. Offline and streaming inference. Qwen3-ForcedAligner-0.6B supports timestamp prediction for up to 5 minutes in 11 languages. HF Open ASR leaderboard mean WER shown on the model cards: 6.42 for 0.6B, 5.76 for 1.7B. | Defend and polish. Keep this package Qwen-specific. |
| Future Qwen ASR successor | The only obvious reason to expand the core architecture scope. | No newer Qwen ASR successor was verified during this refresh. | Revisit immediately if Qwen releases a Qwen3.5/Qwen4 ASR, forced-aligner, or streaming successor. |

### Tier 1: Benchmark Immediately, Do Not Port Yet

| Model | Why it matters | License | Apple Silicon feasibility | Transcript structure | Languages | Reported quality/speed | Memory/size | Project action |
|---|---|---|---|---|---|---|---|---|
| Cohere Transcribe 03-2026 | Strong dedicated ASR accuracy pressure in the same broad size class. | Apache 2.0 | Transformers and vLLM support; model card lists `mlx-audio` ecosystem support. | No timestamps or speaker diarization; model card also flags no automatic language detection and inconsistent code-switching. | 14 languages. | Model card reports 5.42 average English Open ASR leaderboard WER as of 2026-03-26. | 2B parameters. | Add as an external benchmark runner. Do not port into this package unless local MLX performance is clearly compelling and scope is explicit. |
| IBM Granite Speech 4.1 2B | Compact speech-language model with ASR, speech translation, keyword biasing, and companion variants for structure/throughput. | Apache 2.0 | Transformers and vLLM support; safetensors BF16; quantizations exist on HF. No verified native MLX path during this refresh. | Base model focuses ASR/AST and keyword biasing. `granite-speech-4.1-2b-plus` adds speaker-attributed ASR and word-level timestamps; `nar` variant targets throughput. | Six stated supported ASR/AST languages: English, French, German, Spanish, Portuguese, Japanese; additional translation directions include Italian and Mandarin targets. | Model card reports HF Open ASR leaderboard WER 5.33 and RTFx 231.29. | 2B parameters, BF16. | Benchmark external runner. Watch plus/NAR variants for structured transcript and latency ideas. |
| NVIDIA Canary-Qwen-2.5B | Strong leaderboard competitor and relevant because it uses Qwen in a speech stack. | CC-BY-4.0 | NeMo/PyTorch first; CUDA-oriented. No verified MLX path during this refresh. | ASR-focused; no project-relevant diarization/timestamp claim verified in the model card. | English ASR focus in the evaluated card. | Model card reports HF Open ASR leaderboard WER 5.63 and RTFx 418. | 2.5B parameters. | Benchmark only if the harness can run NeMo cleanly. Not a port candidate unless architecture details become unusually reusable. |
| OpenAI `gpt-4o-transcribe` and `gpt-4o-transcribe-diarize` | Closed cloud quality and structured-transcript baseline. | Closed API | Not local; API only. | Diarize variant has built-in speaker diarization. | Vendor-managed. | Docs state improved WER/language recognition over original Whisper; no open weights or local latency. | N/A | Use as optional cloud baseline in benchmarks, never core dependency. |

### Tier 2: Watch for Directional Signals

| Model | Why it matters | License | Apple Silicon feasibility | Transcript structure | Languages | Reported quality/speed | Memory/size | Project action |
|---|---|---|---|---|---|---|---|---|
| Microsoft VibeVoice-ASR | Best directional signal for meeting-style transcription: who, when, what, hotwords, long context, and code switching in one model. | MIT | Transformers/vLLM upstream; `mlx-community/VibeVoice-ASR-4bit` exists via `mlx-audio`. | Joint ASR, diarization, timestamps, hotwords, 60-minute single-pass context. | 51 languages on HF; model card describes over 50 languages and code switching. | Evaluation images are not directly extractable from the model card; verify with local/domain benchmark before comparing WER. | 9B BF16 upstream; 4-bit MLX conversion exists. | Watch closely for product requirements. Not a core port target now due size and model-family drift. |
| Mistral Voxtral Mini 4B Realtime 2602 | Streaming-first open model with configurable transcription delay and an existing MLX conversion. | Apache 2.0 | `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit` exists and runs via `mlx-audio`; upstream is vLLM-first. | Realtime transcription; no diarization claim verified. Configurable delay from 80ms to 2.4s. | 13 languages. | Model card reports FLEURS average WER 8.72 at 480ms delay, 7.70 at 960ms, and 6.73 at 2400ms. HF Open ASR leaderboard mean WER shown on card: 7.68. | 4B BF16 upstream; MLX 4-bit conversion is about 3.13 GB on HF. | Benchmark for streaming UX only. Do not fold into this repo. |
| NVIDIA Nemotron Speech Streaming EN 0.6B | Clean streaming architecture reference for low-latency voice agents. | NVIDIA Open Model License | NeMo/PyTorch first; optimized for NVIDIA stack. | English text with punctuation/capitalization; hosted API examples include word time offsets, but local model card emphasis is streaming ASR. | English. | Model card reports WER 6.93 at 1.12s chunks, 7.07 at 0.56s, 7.67 at 0.16s, and 8.43 at 0.08s. | 600M parameters. | Watch for streaming architecture ideas; not a likely Apple Silicon port. |
| NVIDIA Parakeet-TDT 0.6B v3 | Strong compact multilingual transducer baseline with timestamps. | CC-BY-4.0 | NeMo/PyTorch first; no verified native MLX path. | Word-level and segment-level timestamps; long audio support. | 25 European languages. | Model card reports HF Open ASR leaderboard average WER 6.34. | 600M parameters; card says at least 2 GB RAM to load. | Benchmark if NeMo runner is cheap. Useful latency/timestamp baseline. |
| Kyutai STT 2.6B EN / MLX | Streaming STT with existing MLX release; useful for Apple Silicon streaming expectations. | CC-BY-4.0 | `kyutai/stt-2.6b-en-mlx` runs through `moshi_mlx`. | Streaming transcript; timestamps can be recovered from the text stream offset. | English for 2.6B; separate 1B EN/FR variant exists. | Model card describes 2.5s delay for 2.6B EN and robust long-audio use up to 2 hours. | About 2.6B parameters. | Watch as an Apple Silicon streaming baseline, not a Qwen runtime feature. |

## What This Means for `mlx-qwen3-asr`

### Keep

- Qwen-specific loader, model, tokenizer, streaming, forced-aligner, and server
  code.
- Native MLX implementation quality.
- Small, direct, dependency-conscious package shape.

### Add Next

1. `docs/MODEL_WATCH.md` stays as the source-of-truth watchlist.
2. Add benchmark runners only as optional external integrations. They should
   not add required dependencies to the package.
3. Standardize benchmark output so model comparisons record:
   - dataset and split,
   - normalization policy,
   - WER/CER,
   - RTF/RTFx,
   - peak memory,
   - device,
   - model revision,
   - timestamp/diarization availability.
4. Add a small refresh script or checklist before using this document for a
   product decision.

### Avoid

- Adding Cohere, Granite, Voxtral, VibeVoice, Parakeet, or Kyutai runtime
  support to the core package.
- Comparing vendor model-card WERs as if they were directly equivalent to local
  repo measurements.
- Pulling in NeMo, vLLM, Transformers, or PyTorch as required dependencies.
- Treating MLX-community conversions as proof that the architecture belongs in
  this repo.

## Benchmark Harness Shape

The benchmark harness should be model-agnostic while the runtime remains
Qwen-specific.

Recommended external runner boundary:

```text
benchmarks/asr_runners/
  qwen3_mlx.py          # this package
  whisper_cpp.py        # optional CLI runner
  cohere_transformers.py # optional
  granite_transformers.py # optional
  voxtral_mlx_audio.py  # optional
  vibevoice_mlx_audio.py # optional
  parakeet_nemo.py      # optional
  openai_transcribe.py  # optional cloud baseline
```

Each runner should return the same JSON schema:

```json
{
  "model": "string",
  "revision": "string",
  "runner": "string",
  "device": "string",
  "audio_id": "string",
  "duration_s": 0.0,
  "transcript": "string",
  "segments": [],
  "words": [],
  "speakers": [],
  "latency_s": 0.0,
  "rtf": 0.0,
  "peak_memory_gb": null
}
```

Core rule: optional competitors can live in benchmark tooling, but not in
`mlx_qwen3_asr/`.

## Refresh Checklist

Before making a roadmap decision from this document:

1. Re-check Hugging Face model cards and API metadata for last modified,
   license, language count, and quantization/MLX conversions.
2. Check the Open ASR leaderboard for current WER/RTFx.
3. Verify whether an MLX/Core ML/llama.cpp conversion exists and whether it is
   runnable on Apple Silicon without CUDA-only assumptions.
4. Run at least one local smoke for any model that could affect roadmap
   priority.
5. Separate model-card claims from locally measured results in any public docs.

## Source Snapshot

Primary sources checked on 2026-05-16:

- Qwen3-ASR-1.7B model card:
  https://huggingface.co/Qwen/Qwen3-ASR-1.7B
- Qwen3-ASR-0.6B model card:
  https://huggingface.co/Qwen/Qwen3-ASR-0.6B
- Cohere Transcribe 03-2026 model card:
  https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
- IBM Granite Speech 4.1 2B model card:
  https://huggingface.co/ibm-granite/granite-speech-4.1-2b
- Microsoft VibeVoice-ASR model card:
  https://huggingface.co/microsoft/VibeVoice-ASR
- Mistral Voxtral Mini 4B Realtime 2602 model card:
  https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602
- MLX-community Voxtral 4-bit conversion:
  https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit
- MLX-community VibeVoice 4-bit conversion:
  https://huggingface.co/mlx-community/VibeVoice-ASR-4bit
- NVIDIA Parakeet-TDT 0.6B v3 model card:
  https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- NVIDIA Canary-Qwen 2.5B model card:
  https://huggingface.co/nvidia/canary-qwen-2.5b
- NVIDIA Nemotron Speech Streaming EN 0.6B model card:
  https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b
- Kyutai STT 2.6B EN model card:
  https://huggingface.co/kyutai/stt-2.6b-en
- Kyutai STT 2.6B EN MLX model card:
  https://huggingface.co/kyutai/stt-2.6b-en-mlx
- OpenAI GPT-4o Transcribe model docs:
  https://developers.openai.com/api/docs/models/gpt-4o-transcribe
- OpenAI GPT-4o Transcribe Diarize model docs:
  https://developers.openai.com/api/docs/models/gpt-4o-transcribe-diarize
