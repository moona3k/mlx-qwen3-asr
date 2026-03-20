# Transcription Server — API Specification

**Version:** 1.0 (draft)
**Date:** 2026-03-20

## Overview

HTTP JSON API served by `mlx-qwen3-asr serve`. Turns any Apple Silicon Mac into
a speech-to-text endpoint.

**Base URL:** `http://<host>:<port>` (default port: `8765`)

## Authentication

All endpoints except `GET /health` require a Bearer token:

```
Authorization: Bearer <api-key>
```

API keys are configured at server startup via `--api-key` flag or
`MLX_ASR_API_KEY` environment variable. Multiple keys can be specified
(comma-separated).

Unauthenticated requests receive `401 Unauthorized`.
Invalid keys receive `403 Forbidden`.

## Rate Limiting

Per-key rate limiting. Default: 60 requests/minute. Configurable via
`--rate-limit`.

Exceeded limits return `429 Too Many Requests` with `Retry-After` header.

---

## Endpoints

### `GET /health`

Health check. No auth required.

**Response** `200 OK`:

```json
{
  "status": "ok",
  "model": "Qwen/Qwen3-ASR-0.6B",
  "dtype": "float16",
  "uptime_seconds": 3421,
  "active_jobs": 2
}
```

---

### `POST /transcribe`

Submit a transcription job. Returns immediately with a job ID.

**Content-Type:** `multipart/form-data` or `application/json`

#### Option A: File upload (multipart)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | file | yes | Audio file (wav, mp3, flac, m4a, ogg, etc.) |
| `language` | string | no | ISO 639-1 language code (e.g., `en`, `ja`). Auto-detect if omitted. |
| `timestamps` | bool | no | Include word-level timestamps. Default: `false`. |
| `context` | string | no | Custom system prompt for domain vocabulary biasing. |

#### Option B: URL reference (JSON)

```json
{
  "url": "https://storage.example.com/meeting.wav",
  "language": "en",
  "timestamps": true,
  "context": ""
}
```

**Response** `202 Accepted`:

```json
{
  "job_id": "j_a1b2c3d4",
  "status": "queued",
  "created_at": "2026-03-20T14:30:00Z"
}
```

**Error responses:**

| Code | Condition |
|------|-----------|
| `400` | No audio provided, unsupported format, or file exceeds size limit |
| `401` | Missing auth token |
| `403` | Invalid auth token |
| `429` | Rate limit exceeded |
| `413` | File too large (default max: 500 MB) |

---

### `GET /jobs/{job_id}`

Poll job status and retrieve results.

**Response** `200 OK` (processing):

```json
{
  "job_id": "j_a1b2c3d4",
  "status": "processing",
  "created_at": "2026-03-20T14:30:00Z",
  "progress": null
}
```

**Response** `200 OK` (completed):

```json
{
  "job_id": "j_a1b2c3d4",
  "status": "completed",
  "created_at": "2026-03-20T14:30:00Z",
  "completed_at": "2026-03-20T14:30:12Z",
  "result": {
    "text": "The full transcription text here.",
    "language": "en",
    "duration": 45.2,
    "segments": [
      {
        "start": 0.0,
        "end": 3.5,
        "text": "The full transcription"
      },
      {
        "start": 3.5,
        "end": 5.1,
        "text": "text here."
      }
    ],
    "words": [
      {
        "word": "The",
        "start": 0.0,
        "end": 0.3
      }
    ]
  }
}
```

`words` array is only present when `timestamps: true` was requested.

**Response** `200 OK` (failed):

```json
{
  "job_id": "j_a1b2c3d4",
  "status": "failed",
  "created_at": "2026-03-20T14:30:00Z",
  "error": "Audio file is corrupted or contains no speech."
}
```

**Error responses:**

| Code | Condition |
|------|-----------|
| `404` | Job ID not found (expired or never existed) |

---

### `DELETE /jobs/{job_id}`

Delete a job and its results. Useful for cleanup.

**Response** `204 No Content`

---

## Job lifecycle

```
queued → processing → completed
                    → failed
```

- Jobs are stored in-memory. Default TTL: 1 hour after completion.
- Server restart clears all jobs.
- Only one job processes at a time (sequential model inference). Queued jobs
  wait in FIFO order.

## CLI usage

```bash
# Start server
mlx-qwen3-asr serve --port 8765 --api-key mykey123

# With options
mlx-qwen3-asr serve \
  --port 8765 \
  --api-key mykey123 \
  --model Qwen/Qwen3-ASR-1.7B \
  --rate-limit 30 \
  --max-file-size 500 \
  --job-ttl 3600 \
  --host 0.0.0.0

# API key via environment variable
export MLX_ASR_API_KEY=mykey123
mlx-qwen3-asr serve
```

## Client examples

### cURL — file upload

```bash
curl -X POST http://localhost:8765/transcribe \
  -H "Authorization: Bearer mykey123" \
  -F "audio=@meeting.wav" \
  -F "language=en"
# → {"job_id": "j_a1b2c3d4", "status": "queued", ...}

curl http://localhost:8765/transcribe/jobs/j_a1b2c3d4 \
  -H "Authorization: Bearer mykey123"
# → {"job_id": "j_a1b2c3d4", "status": "completed", "result": {...}}
```

### cURL — URL reference

```bash
curl -X POST http://localhost:8765/transcribe \
  -H "Authorization: Bearer mykey123" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://storage.example.com/clip.wav"}'
```

### Python

```python
import requests

API = "http://192.168.1.42:8765"
KEY = "mykey123"
headers = {"Authorization": f"Bearer {KEY}"}

# Submit
with open("meeting.wav", "rb") as f:
    r = requests.post(f"{API}/transcribe", headers=headers, files={"audio": f})
job_id = r.json()["job_id"]

# Poll
import time
while True:
    r = requests.get(f"{API}/jobs/{job_id}", headers=headers)
    data = r.json()
    if data["status"] in ("completed", "failed"):
        break
    time.sleep(1)

print(data["result"]["text"])
```

## Configuration defaults

| Parameter | Default | CLI flag | Env var |
|-----------|---------|----------|---------|
| Port | `8765` | `--port` | `MLX_ASR_PORT` |
| Host | `0.0.0.0` | `--host` | `MLX_ASR_HOST` |
| API key(s) | — (required) | `--api-key` | `MLX_ASR_API_KEY` |
| Model | `Qwen/Qwen3-ASR-0.6B` | `--model` | — |
| Rate limit | `60` req/min | `--rate-limit` | — |
| Max file size | `500` MB | `--max-file-size` | — |
| Job TTL | `3600` seconds | `--job-ttl` | — |

## Future (not in v1)

- WebSocket streaming for real-time audio
- Callback/webhook on job completion
- Batch endpoint (multiple files in one request)
- TLS termination (use reverse proxy for now)
- Persistent job store (SQLite)
