# Comparison: Qwen3-ASR Mac Implementations

Snapshot comparison for practitioners choosing a Qwen3-ASR path on Apple
Silicon. This is an implementation-focused view, not a model-quality ranking.

## Scope and Date

- Snapshot date: 2026-02-14
- Primary references: `docs/RESEARCH.md` and project benchmark artifacts under
  `docs/benchmarks/`

## Feature Matrix

| Feature | mlx-qwen3-asr | mlx-audio | qwen3-asr-swift | Official (PyTorch) |
|---|---|---|---|---|
| Installation UX | `pip install mlx-qwen3-asr` | `pip install mlx-audio` | source/build workflow | `pip install qwen-asr` (CUDA-focused runtime stack) |
| Qwen3-ASR focus | Single-model focused | Multi-model toolkit | Single-model focused | Single-model focused |
| Core backend | MLX (Metal) | MLX (Metal) | MLX (Metal) | PyTorch |
| Streaming status | Official re-feed recipe; within ~2pp of offline on multilingual-100 | Varies by revision | Streaming supported | Streaming supported (vLLM only) |
| Timestamp alignment | native MLX default + `qwen_asr` optional reference backend | Not a primary surfaced lane | No public forced-aligner API in cited snapshot | Official forced aligner |
| Quantized path | Published `moona3k/mlx-qwen3-asr-*` 4/8-bit artifacts, validated per release | Yes (`mlx-community`) | Varies | Varies |

## Measured Position of This Repo

From committed artifacts in this repo:

- Quantized long-clip speedup (`0.6B`, 10s lane, v0.4.0): `4bit-g64` at `1.71x` vs fp16; 8-bit matches fp16 output
  (`docs/benchmarks/2026-09-07-quality-matrix-refresh.md`)
- Published 4-bit artifacts keep the audio encoder at 8 bits: 0.6B 2.37% WER
  vs fp16 2.33%, 1.7B 1.73% vs 1.94% (`docs/benchmarks/2026-09-19-quantized-artifacts-*.json`)
- Streaming vs offline on the same 100 multilingual clips: 11.4% vs 9.5%
  primary error (`docs/benchmarks/2026-09-19-streaming-manifest-multilingual100.json`)
- Timestamp backend parity snapshot (`test-clean`, English, `n=50`):
  - text match rate: `1.0000`
  - timing MAE: `5.6909 ms`
  - relative speed (`qwen_asr / mlx`): `2.64x`
  (`docs/benchmarks/2026-02-14-aligner-parity-50.md`)

## Practical Trade-Offs

### mlx-qwen3-asr (this project)

- Best when you want a Python-first, pip-installable, Mac-native Qwen3-ASR path
  with explicit quality gates and benchmark artifacts.
- Keeps streaming/speculative lanes explicitly experimental to avoid over-claiming.

### mlx-audio

- Best when you need one package covering many speech models.
- Verify Qwen3-ASR behavior against the exact revision you deploy, because
  priorities are broader than single-model parity.

### qwen3-asr-swift

- Best when your product surface is Swift/macOS-native and you want app-level
  integration directly in Swift.
- Python library and packaging ergonomics are intentionally not its target.

### Official PyTorch stack

- Best reference for feature semantics and upstream correctness behavior.
- Usually the first source of truth for model updates and alignment behavior.

## Policy for This File

- Prefer implementation status facts over marketing language.
- When a comparison point is revision-sensitive, state it as such.
- Keep hard performance/quality claims tied to committed artifact files.
