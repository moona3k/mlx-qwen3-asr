# ADR-001: Built-in Transcription Server

**Status:** Accepted
**Date:** 2026-03-20

## Context

Apple Silicon Macs are efficient inference machines for MLX workloads. Users want
to turn their Macs into transcription endpoints — accepting audio from any device
on the network (or internet) and returning text. Today `mlx-qwen3-asr` is a
library + CLI; there is no way to serve transcription over HTTP without writing
custom glue code.

Note: CLAUDE.md currently says "Not a server/API — library + CLI only." This
must be updated to reflect the new scope before merging.

## Decision

Ship a built-in HTTP transcription server inside the `mlx_qwen3_asr` package as
an optional feature, activated via `pip install mlx-qwen3-asr[serve]` and launched
with `mlx-qwen3-asr serve`.

### Design choices

| Question | Decision | Rationale |
|----------|----------|-----------|
| **Packaging** | In-package, optional deps | One repo, one install. Server is a natural extension of the CLI, not a separate product. Optional deps keep the core lean. |
| **Framework** | FastAPI + uvicorn + python-multipart | Async, lightweight, battle-tested. File upload handling is first-class. OpenAPI docs come free. `python-multipart` required for FastAPI file uploads. |
| **Tenancy** | Single-tenant | Your Macs, your API keys. No user management, no billing, no isolation concerns. |
| **Auth** | API key (Bearer token) | Simple, sufficient for single-tenant internet-facing use. Keys configured at startup. Server refuses to start without at least one key. |
| **Job model** | Async jobs (POST returns job ID, poll for result) | Uniform model for short and long audio. Avoids hanging HTTP connections on long files. |
| **Audio input** | File upload (multipart) only | Simple, universal, no SSRF risk. URL ingestion deferred to v2 (requires scheme allowlist, private-IP blocking, redirect limits, download timeout — too much complexity for v1). |
| **Rate limiting** | Per-key, requests/minute | Basic abuse protection for internet-facing deployment. `GET /jobs/{id}` polling does NOT count against the rate limit bucket. |
| **Model lifecycle** | Load on startup, keep in memory | Session object persists across requests. No per-request model loading. No hot-swap in v1. |
| **Job storage** | In-memory (dict) | Simple, no external deps. Jobs expire after configurable TTL. Acceptable for single-process server. |
| **Concurrency** | Sequential (FIFO) | One job at a time. Most Macs can't run parallel model inference anyway. Queue capped at configurable max depth (default 10); returns `503 Service Unavailable` when full. |
| **Temp files** | Write uploads to `tempfile.NamedTemporaryFile`, delete after transcription completes or fails | `load_audio_np()` requires a filesystem path. Temp files are cleaned up in a `finally` block and on job TTL expiry. |

### What this is NOT

- Not multi-tenant — no user accounts, no per-user quotas
- Not a distributed job queue — single process, in-memory state
- Not a WebSocket streaming server — async jobs only in v1
- Not a production deployment platform — no TLS termination, no process management (use a reverse proxy + systemd/launchd for that)

## Consequences

### Positive

- Any Mac becomes a transcription endpoint with one command
- Clients on any platform (phones, other machines, scripts) can use the ASR capability
- Aligns with "one-command setup" project philosophy
- No new repos or packages to maintain

### Negative

- Server deps (FastAPI, uvicorn, python-multipart) added to optional dependency surface
- In-memory job store means jobs are lost on restart
- Single-process model means throughput is limited to one transcription at a time per worker

### Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Scope creep toward "real" server product | Keep the server as a convenience feature. No multi-tenancy, no databases, no orchestration. |
| Memory pressure from large audio files | File size cap enforced at upload. Configurable max audio duration. Temp files cleaned up eagerly. |
| API key leakage | Keys are local config, not stored in code. Docs recommend reverse proxy + TLS for internet exposure. |
| Queue starvation from long files | Max queue depth with 503 backpressure. Max audio duration enforced at submission. |
| Temp file leaks on crash | Temp files written to system temp dir; OS cleans up on reboot. TTL expiry also cleans up. |

## Alternatives considered

1. **Separate package** (`mlx-qwen3-asr-server`) — cleaner separation but adds maintenance overhead and friction for users. Rejected.
2. **gRPC instead of HTTP** — better for streaming, worse for simplicity and tooling. HTTP is universal. Rejected for v1.
3. **Sync-only API** — simpler but blocks on long audio. Rejected in favor of uniform async job model.
4. **WebSocket streaming** — valuable for real-time use but adds significant complexity. Deferred to v2.
5. **URL ingestion** — convenient for cloud storage but introduces SSRF attack surface (private-IP access, metadata endpoints, localhost probing). Requires scheme allowlist, redirect limits, content-length enforcement, and download timeouts. Deferred to v2.
6. **DELETE /jobs/{id}** — useful for cleanup but semantics are unclear when a job is actively processing (cancel inference? kill ffmpeg subprocess? just hide result?). Deferred to v2 when cancellation can be designed properly.
