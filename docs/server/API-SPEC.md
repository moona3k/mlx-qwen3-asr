# Transcription Server — API Specification

**Version:** 1.0 (draft, revised)
**Date:** 2026-03-20

## Overview

HTTP JSON API served by `mlx-qwen3-asr serve`. Turns any Apple Silicon Mac into
a speech-to-text endpoint.

**Base URL:** `http://<host>:<port>` (default port: `8765`)

## Prerequisites

- Apple Silicon Mac with `pip install mlx-qwen3-asr[serve]`
- `ffmpeg` installed on the host (`brew install ffmpeg`) — required for non-WAV
  audio formats (mp3, flac, m4a, ogg, etc.). WAV files work without ffmpeg.

## Authentication

All endpoints except `GET /health` require a Bearer token:

```
Authorization: Bearer <api-key>
```

API keys are configured at server startup via `--api-key` flag or
`MLX_ASR_API_KEY` environment variable. Multiple keys can be specified
(comma-separated). **The server refuses to start without at least one key.**

Unauthenticated requests receive `401 Unauthorized`.
Invalid keys receive `403 Forbidden`.

## Rate Limiting

Per-key rate limiting on submission endpoints. Default: 60 requests/minute.
Configurable via `--rate-limit`.

`GET /jobs/{id}` polling does **not** count against the rate limit, so clients
can poll freely without burning their submission budget.

Exceeded limits return `429 Too Many Requests` with `Retry-After` header.

## Backpressure

The job queue has a configurable max depth (default: 10). When the queue is full,
`POST /transcribe` returns `503 Service Unavailable` with a `Retry-After` header.

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
  "queued_jobs": 2,
  "processing_jobs": 1,
  "max_queue_depth": 10
}
```

---

### `POST /transcribe`

Submit a transcription job. Returns immediately with a job ID.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | file | yes | Audio file (wav, mp3, flac, m4a, ogg, etc.) |
| `language` | string | no | Language code (e.g., `en`, `ja`, `zh`). Auto-detect if omitted. |
| `timestamps` | string | no | `"true"` to include word-level timestamps. Default: `"false"`. Note: timestamps require the forced aligner model (~1.2 GB additional download on first use, ~1.5 GB additional RAM). |
| `context` | string | no | Custom system prompt for domain vocabulary biasing. |

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
| `400` | No audio file provided, or unsupported format |
| `401` | Missing auth token |
| `403` | Invalid auth token |
| `413` | File too large (default max: 200 MB) |
| `422` | Audio duration exceeds max allowed (default: 30 minutes) |
| `429` | Rate limit exceeded |
| `503` | Queue full — server is at capacity |

---

### `GET /jobs/{job_id}`

Poll job status and retrieve results. Does not count against rate limit.

**Response** `200 OK` (queued):

```json
{
  "job_id": "j_a1b2c3d4",
  "status": "queued",
  "created_at": "2026-03-20T14:30:00Z"
}
```

**Response** `200 OK` (processing):

```json
{
  "job_id": "j_a1b2c3d4",
  "status": "processing",
  "created_at": "2026-03-20T14:30:00Z"
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
    "language": "English",
    "segments": [
      {
        "text": "The",
        "start": 0.0,
        "end": 0.3
      },
      {
        "text": "full",
        "start": 0.3,
        "end": 0.6
      }
    ],
    "chunks": [
      {
        "text": "The full transcription text here.",
        "start": 0.0,
        "end": 5.1,
        "chunk_index": 0,
        "language": "English"
      }
    ]
  }
}
```

**Response schema notes:**

The `result` object directly mirrors the library's `TranscriptionResult` dataclass:

| Field | Type | Always present | Description |
|-------|------|---------------|-------------|
| `text` | string | yes | Full transcription text |
| `language` | string | yes | Detected or forced language. Returns the library's canonical form (e.g., `"English"`, `"Japanese"`), not ISO codes. |
| `segments` | array | only with `timestamps: true` | Word-level timestamps from forced aligner. Each item: `{text, start, end}`. |
| `chunks` | array | only for chunked audio | Chunk-level transcripts for long audio. Each item: `{text, start, end, chunk_index, language}`. |
| `speaker_segments` | array | only with diarization | Speaker-attributed spans (future). Each item: `{speaker, start, end, text}`. |

The server passes through the library output as-is — no schema translation layer.

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

### `POST /v1/audio/transcriptions`

OpenAI-compatible transcription endpoint. Drop-in replacement for the
OpenAI Audio API — existing clients just change the base URL.

**Key difference from `/transcribe`:** This endpoint is **synchronous** — it
blocks until transcription completes and returns the result directly. No job
queue, no polling. Best for short audio or when you need OpenAI SDK
compatibility.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | yes | Audio file (wav, mp3, flac, m4a, ogg, etc.) |
| `model` | string | no | Model identifier. Accepted for compatibility; server uses its loaded model. |
| `language` | string | no | Language code (e.g., `en`, `ja`, `zh`). Auto-detect if omitted. |
| `prompt` | string | no | Text to guide transcription (maps to `context` internally). |
| `response_format` | string | no | `json` (default), `text`, `verbose_json`, `srt`, `vtt` |
| `temperature` | float | no | Accepted for compatibility. Ignored (greedy decoding). |

**Response** (`response_format=json`, default) `200 OK`:

```json
{"text": "The transcribed text."}
```

**Response** (`response_format=text`) `200 OK`:

```
The transcribed text.
```

**Response** (`response_format=verbose_json`) `200 OK`:

```json
{
  "task": "transcribe",
  "language": "english",
  "duration": 5.1,
  "text": "The transcribed text.",
  "words": [
    {"word": "The", "start": 0.0, "end": 0.3},
    {"word": "transcribed", "start": 0.3, "end": 0.8},
    {"word": "text.", "start": 0.8, "end": 1.2}
  ]
}
```

**Response** (`response_format=srt`) `200 OK`:

```
1
00:00:00,000 --> 00:00:01,200
The transcribed text.
```

**Response** (`response_format=vtt`) `200 OK`:

```
WEBVTT

00:00:00.000 --> 00:00:01.200
The transcribed text.
```

**Notes:**

- `verbose_json`, `srt`, and `vtt` formats automatically enable word-level
  timestamps (forced aligner). First use downloads the aligner model (~1.2 GB).
- The `words` array in `verbose_json` maps OpenAI's `word` field from our
  word-level segments. Duration is estimated from the last timestamp.

**Error responses:**

| Code | Condition |
|------|-----------|
| `400` | Invalid `response_format` |
| `401` | Missing auth token |
| `403` | Invalid auth token |
| `413` | File too large |
| `429` | Rate limit exceeded |
| `500` | Transcription failed |

---

## Job lifecycle

```
queued → processing → completed
                    → failed
```

- Jobs are stored in-memory. Default TTL: 1 hour after completion/failure.
- Server restart clears all jobs.
- Only one job processes at a time (sequential model inference). Queued jobs
  wait in FIFO order.
- Max queue depth: 10 (configurable). Submissions beyond this return `503`.

## Temp file lifecycle

Uploaded audio is written to a `tempfile.NamedTemporaryFile` on disk (system
temp directory). The file is deleted in a `finally` block after transcription
completes or fails. Temp files for expired jobs are also cleaned up during
TTL expiry sweeps.

## CLI subcommand

The `serve` subcommand requires converting the existing CLI to a subcommand
architecture. The current CLI treats the first positional arg as an audio file
path. After the change:

```
mlx-qwen3-asr transcribe audio.wav    # explicit transcribe subcommand
mlx-qwen3-asr audio.wav               # legacy shorthand (positional arg without subcommand)
mlx-qwen3-asr serve [options]         # start server
```

Backward compatibility: bare `mlx-qwen3-asr audio.wav` continues to work.

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
  --max-file-size 200 \
  --max-duration 1800 \
  --max-queue-depth 10 \
  --job-ttl 3600 \
  --host 0.0.0.0

# API key via environment variable
export MLX_ASR_API_KEY=mykey123
mlx-qwen3-asr serve
```

## Client examples

### cURL

```bash
# Submit
curl -X POST http://localhost:8765/transcribe \
  -H "Authorization: Bearer mykey123" \
  -F "audio=@meeting.wav" \
  -F "language=en"
# → {"job_id": "j_a1b2c3d4", "status": "queued", ...}

# Poll
curl http://localhost:8765/jobs/j_a1b2c3d4 \
  -H "Authorization: Bearer mykey123"
# → {"job_id": "j_a1b2c3d4", "status": "completed", "result": {...}}
```

### Python

```python
import requests
import time

API = "http://192.168.1.42:8765"
KEY = "mykey123"
headers = {"Authorization": f"Bearer {KEY}"}

# Submit
with open("meeting.wav", "rb") as f:
    r = requests.post(f"{API}/transcribe", headers=headers, files={"audio": f})
job_id = r.json()["job_id"]

# Poll
while True:
    r = requests.get(f"{API}/jobs/{job_id}", headers=headers)
    data = r.json()
    if data["status"] in ("completed", "failed"):
        break
    time.sleep(1)

print(data["result"]["text"])
```

### OpenAI Python SDK

Existing OpenAI client code works with zero changes — just point at your Mac:

```python
from openai import OpenAI

client = OpenAI(
    api_key="mykey123",
    base_url="http://localhost:8765/v1",
)

result = client.audio.transcriptions.create(
    model="Qwen/Qwen3-ASR-0.6B",
    file=open("meeting.wav", "rb"),
)
print(result.text)

# With word timestamps
result = client.audio.transcriptions.create(
    model="Qwen/Qwen3-ASR-0.6B",
    file=open("meeting.wav", "rb"),
    response_format="verbose_json",
)
for word in result.words:
    print(f"{word['start']:.2f}s: {word['word']}")
```

## Configuration defaults

| Parameter | Default | CLI flag | Env var |
|-----------|---------|----------|---------|
| Port | `8765` | `--port` | `MLX_ASR_PORT` |
| Host | `0.0.0.0` | `--host` | `MLX_ASR_HOST` |
| API key(s) | — (required) | `--api-key` | `MLX_ASR_API_KEY` |
| Model | `Qwen/Qwen3-ASR-0.6B` | `--model` | — |
| Rate limit | `60` req/min | `--rate-limit` | — |
| Max file size | `200` MB | `--max-file-size` | — |
| Max audio duration | `1800` seconds (30 min) | `--max-duration` | — |
| Max queue depth | `10` | `--max-queue-depth` | — |
| Job TTL | `3600` seconds | `--job-ttl` | — |

## Future (not in v1)

- URL ingestion (requires SSRF mitigation: scheme allowlist, private-IP blocking, redirect limits)
- WebSocket streaming for real-time audio
- Job cancellation and `DELETE /jobs/{id}`
- Callback/webhook on job completion
- Batch endpoint (multiple files in one request)
- TLS termination (use reverse proxy for now)
- Persistent job store (SQLite)
