# MLX vs PyTorch Quality Head-to-Head (manifest-quality-v1, n=1)

- model: `Qwen/Qwen3-ASR-0.6B`
- samples: `1`
- reference_max_chunk_sec: `20.0`
- MLX primary: `0.0537`
- PyTorch primary: `0.0555`
- Delta primary (MLX-Ref): `-0.0018`

| System | Primary | WER | CER | Mean latency (s) |
|---|---:|---:|---:|---:|
| MLX | 0.0537 | 0.0537 | 0.0204 | 203.4708 |
| PyTorch ref | 0.0555 | 0.0555 | 0.0180 | 595.0302 |
