# Deployment Guide

How to run `mlx-qwen3-asr serve` as an internet-facing transcription server.

## Quick start (LAN)

```bash
pip install mlx-qwen3-asr[serve]
mlx-qwen3-asr serve --api-key $(openssl rand -hex 16)
```

Your Mac is now a transcription server on your local network.

## Internet-facing deployment

For internet exposure, put the server behind a reverse proxy that handles TLS
and connection management. The ASR server itself is HTTP-only by design.

### Architecture

```
Internet → Reverse Proxy (TLS) → mlx-qwen3-asr serve (localhost:8765)
           (nginx / caddy)
```

### Option A: Caddy (simplest)

```
# /etc/caddy/Caddyfile
transcribe.yourdomain.com {
    reverse_proxy localhost:8765
}
```

Caddy handles TLS certificates automatically via Let's Encrypt.

### Option B: nginx

```nginx
server {
    listen 443 ssl;
    server_name transcribe.yourdomain.com;

    ssl_certificate     /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    client_max_body_size 500M;

    location / {
        proxy_pass http://127.0.0.1:8765;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Option C: Tailscale / Cloudflare Tunnel

For private internet access without opening ports:

```bash
# Tailscale (machine-to-machine VPN)
tailscale serve --bg 8765

# Cloudflare Tunnel
cloudflared tunnel --url http://localhost:8765
```

Both options handle TLS and avoid exposing your Mac's IP directly.

## Running as a system service

### launchd (macOS native)

Create `~/Library/LaunchAgents/com.mlx-qwen3-asr.serve.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.mlx-qwen3-asr.serve</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/venv/bin/mlx-qwen3-asr</string>
        <string>serve</string>
        <string>--port</string>
        <string>8765</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>MLX_ASR_API_KEY</key>
        <string>your-api-key-here</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/mlx-qwen3-asr.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/mlx-qwen3-asr.err</string>
</dict>
</plist>
```

```bash
launchctl load ~/Library/LaunchAgents/com.mlx-qwen3-asr.serve.plist
```

## Operational notes

### Model pre-download

First run downloads the model from HuggingFace (~1.2 GB for 0.6B, ~3.4 GB for
1.7B). Pre-download to avoid startup delay:

```bash
python -c "from mlx_qwen3_asr import load_model; load_model()"
```

### Memory usage

| Model | Approximate RAM |
|-------|----------------|
| 0.6B fp16 | ~1.5 GB |
| 0.6B 4-bit | ~0.5 GB |
| 1.7B fp16 | ~3.5 GB |
| 1.7B 4-bit | ~1.2 GB |

Plus overhead for audio processing and job state. A Mac Mini with 16 GB RAM
handles the 0.6B model comfortably with room for the OS and other processes.

### Monitoring

- `GET /health` — check server status, model info, active job count
- Log output goes to stdout/stderr (capture via launchd or redirect)
- Watch for OOM: if the Mac starts swapping, the model is too large for
  available memory

### Security checklist

- [ ] TLS via reverse proxy (never expose HTTP to the internet)
- [ ] Strong API key (at least 32 hex characters)
- [ ] File size limit configured appropriately
- [ ] Rate limit configured appropriately
- [ ] Firewall rules if not using tunnel
