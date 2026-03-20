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

## API at a glance

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server status (no auth) |
| `/transcribe` | POST | Submit audio for transcription |
| `/jobs/{id}` | GET | Poll job status / get result |
| `/jobs/{id}` | DELETE | Delete a job |

### Submit audio

```bash
# File upload
curl -X POST http://localhost:8765/transcribe \
  -H "Authorization: Bearer YOUR_KEY" \
  -F "audio=@recording.wav"

# URL reference
curl -X POST http://localhost:8765/transcribe \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://storage.example.com/audio.wav"}'
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
    "language": "en",
    "duration": 12.5
  }
}
```

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `8765` | Server port |
| `--host` | `0.0.0.0` | Bind address |
| `--api-key` | — (required) | API key(s), comma-separated |
| `--model` | `Qwen/Qwen3-ASR-0.6B` | Model to load |
| `--rate-limit` | `60` | Max requests per minute per key |
| `--max-file-size` | `500` | Max upload size in MB |
| `--job-ttl` | `3600` | Seconds to keep completed jobs |

API key can also be set via `MLX_ASR_API_KEY` environment variable.

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
service configuration.

## Documentation

| Document | Contents |
|----------|----------|
| [API-SPEC.md](API-SPEC.md) | Full API specification — endpoints, schemas, auth, rate limiting |
| [ADR-001-transcription-server.md](ADR-001-transcription-server.md) | Architecture decision record — design choices and rationale |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Deployment guide — reverse proxy, launchd, security checklist |
