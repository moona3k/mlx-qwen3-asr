# STT releases and standalone repo opportunities

Verified September 7, 2026 against primary announcements, model cards, papers,
and runtime documentation. No models downloaded or executed. This supplements
the May snapshot in `docs/MODEL_WATCH.md`; it does not change the roadmap.

## Recommendation

Investigate MOSS-Transcribe-Diarize first for a focused local meeting-transcript
package, and VibeVoice-ASR-Streaming second for live speaker attribution.
Granite Speech 5.0 is the compact English efficiency candidate. These are
different capability advances, not proven universal replacements for Qwen3-ASR.
All already have documented MLX support, so a new repository needs demonstrated
advantages in packaging, correctness, performance, or product integration.

## Candidates

### MOSS-Transcribe-Diarize 0.9B — July 9, 2026

Apache-2.0; upstream claims 50+ languages, up to 90 minutes in one pass,
speaker labels, timestamps, acoustic events, and hotword prompting. Its compact
size and combined outputs make it the most interesting offline product candidate.
These duration and quality claims remain untested locally.
[Official model card](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize).

Existing [Python MLX implementation](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/moss_transcribe_diarize/moss_transcribe_diarize.py)
and [Swift MLX implementation](https://github.com/Blaizzy/mlx-audio-swift/blob/main/Sources/MLXAudioSTT/Models/MossTranscribeDiarize/README.md)
mean this would be a focused runtime project, not a first port.

### VibeVoice-ASR-Streaming — September 3, 2026

Microsoft's MIT-licensed release adds streaming speaker attribution and hotwords
in ten languages. Names 1.5B and 7B refer to backbone scale; the Hugging Face
cards list approximately 3B and 9B total parameters.
[Release announcement](https://github.com/microsoft/VibeVoice),
[small model](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-1.5B),
[large model](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-7B).

Released checkpoints use about 2.9 seconds of audio plus 0.5 seconds lookahead,
target recordings up to eight minutes, and retain growing history. The reported
two-second expected attribution latency is not a measured Mac latency. Strong
speaker-attribution results are the authors' evaluation, not proof of universal
ASR superiority. [Technical report](https://arxiv.org/html/2609.02812v1).

The current [mlx-audio README](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/vibevoice_asr/README.md)
already documents both streaming checkpoints and live-audio state APIs. It lists
speaker/content output for streaming, versus speaker/timestamps/content offline.
Do not confuse this support with PR #925, which concerns VibeVoice TTS streaming.

### Granite Speech 5.0 TurboCTC — August 25, 2026

470M English encoder-only ASR. The standard model is Apache-2.0; the separate
`-nc` checkpoint is noncommercial. IBM reports 5.00% aggregate WER for the Apache
model on public English OpenASR tests, and over 12,600 RTFx using batched H200
inference. This suggests an efficiency advance, not laptop performance or a
direct comparison with older Qwen leaderboard numbers.
[IBM announcement](https://huggingface.co/blog/ibm-granite/granite-speech-5-0-470m-turboctc).

The [official model card](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc)
already documents mlx-audio 0.5.1 or later.

### Nemotron 3.5 ASR Streaming 0.6B — June 2026

Compact cached streaming with language detection and configurable chunking.
The advertised 40 locales comprise 19 transcription-ready, 13 broad-coverage,
and eight adaptation-ready locales; the last group requires fine-tuning.
The current weights card names OpenMDW-1.1. Chunk sizes are not end-to-end latency.
[NVIDIA model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b).

[MLX support](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/nemotron_asr/README.md)
already exists. The public [MacParakeet engine specification](https://github.com/moona3k/macparakeet/blob/main/spec/06-stt-engine.md)
also documents Nemotron 3.5 Beta and Cohere Transcribe engines; that is documented
integration status, not verification of the installed app.

## Decision gate before creating a repo

Use existing runtimes to compare representative audio against Qwen3-ASR and a
separate diarization baseline. Record exact model/runtime revisions, WER/CER,
speaker-attributed error, timestamp error, speaker continuity, peak memory,
and latency on the same Mac. Include overlap, silence, names, code-switching,
and long recordings. For VibeVoice, evaluate its supported duration first and
treat longer sessions as a separate experiment. Create a standalone package
only when this establishes a useful capability and a concrete runtime gap.
