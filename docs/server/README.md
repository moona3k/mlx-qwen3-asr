# Transcription Server

Turn any Apple Silicon Mac into a speech-to-text API endpoint.

```bash
pip install mlx-qwen3-asr[serve]
mlx-qwen3-asr serve --api-key $(openssl rand -hex 16)
```

That's it. Your Mac is now a transcription server.

## How it works

The server wraps the `mlx-qwen3-asr` library in a FastAPI HTTP service. Audio
goes in, text comes out. The Qwen3-ASR model stays loaded in memory across
requests — no per-request startup cost.

```
Client (any device)           Mac (server)
─────────────────            ──────────────
POST /transcribe  ─────────→  Queue job
                               │
GET /jobs/{id}    ─────────→  Return result
                               ↑
                          MLX inference
                          (Metal GPU)
```

Jobs process sequentially (one at a time, FIFO). Queue capped at 10 by default;
returns `503` when full.

## API at a glance

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | No | Server status, queue depth |
| `/transcribe` | POST | Yes | Submit audio → async job (poll for result) |
| `/jobs/{id}` | GET | Yes | Poll job status / get result |
| `/v1/audio/transcriptions` | POST | Yes | OpenAI-compatible (synchronous) |

### Submit audio

```bash
curl -X POST http://localhost:8765/transcribe \
  -H "Authorization: Bearer YOUR_KEY" \
  -F "audio=@recording.wav"
```

### Get result

```bash
curl http://localhost:8765/jobs/j_a1b2c3d4 \
  -H "Authorization: Bearer YOUR_KEY"
```

```json
{
  "job_id": "j_a1b2c3d4",
  "status": "completed",
  "result": {
    "text": "Your transcribed text here.",
    "language": "English"
  }
}
```

The `result` object mirrors the library's `TranscriptionResult` directly —
includes `segments` (word timestamps) and `chunks` (long-audio chunks) when
applicable.

### OpenAI-compatible endpoint

Existing OpenAI SDK code works with zero changes — just point at your Mac:

```python
from openai import OpenAI

client = OpenAI(api_key="YOUR_KEY", base_url="http://localhost:8765/v1")
result = client.audio.transcriptions.create(
    model="Qwen/Qwen3-ASR-0.6B",
    file=open("recording.wav", "rb"),
)
print(result.text)
```

Supports `response_format`: `json`, `text`, `verbose_json`, `srt`, `vtt`.
This endpoint is synchronous (blocks until done) — for long audio, use the
async `/transcribe` + polling flow instead.

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `8765` | Server port |
| `--host` | `0.0.0.0` | Bind address |
| `--api-key` | — (required) | API key(s), comma-separated |
| `--model` | `Qwen/Qwen3-ASR-0.6B` | Model to load |
| `--rate-limit` | `60` | Max submissions per minute per key |
| `--max-file-size` | `200` | Max upload size in MB |
| `--max-duration` | `1800` | Max audio duration in seconds |
| `--max-queue-depth` | `10` | Max queued jobs before 503 |
| `--job-ttl` | `3600` | Seconds to keep completed jobs |

API key can also be set via `MLX_ASR_API_KEY` environment variable.

## Prerequisites

- Apple Silicon Mac (M1+)
- `ffmpeg` for non-WAV formats (`brew install ffmpeg`). WAV uploads work without it.

## Internet-facing deployment

For internet exposure, put the server behind a reverse proxy for TLS:

```
Internet → Caddy/nginx (TLS) → mlx-qwen3-asr serve (localhost:8765)
```

Or use a tunnel:

```bash
# Cloudflare Tunnel
cloudflared tunnel --url http://localhost:8765

# Tailscale
tailscale serve --bg 8765
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for full deployment guide including launchd
service configuration and security checklist.

## Documentation

| Document | Contents |
|----------|----------|
| [API-SPEC.md](API-SPEC.md) | Full API specification — endpoints, schemas, auth, rate limiting, backpressure |
| [ADR-001-transcription-server.md](ADR-001-transcription-server.md) | Architecture decision record — design choices, rationale, alternatives rejected |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Deployment guide — reverse proxy, launchd, memory sizing, security checklist |
