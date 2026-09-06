# Python Readiness Checkpoint (2026-02-15)

This checkpoint captures the current Python implementation readiness for the
MLX-first Qwen3-ASR roadmap.

## Scope

- Validate release quality gate under strict profile.
- Confirm typed-core gate is active and green.
- Record residual gaps honestly before Swift-port handoff.

## Strict Gate Command

```bash
RUN_STRICT_RELEASE=1 \
MANIFEST_QUALITY_EVAL_JSONL=docs/benchmarks/2026-02-14-fleurs-multilingual-100-manifest.jsonl \
uv run python scripts/quality_gate.py --mode release
```

Run date: 2026-02-15

## Result

- Final status: `PASS`
- Total runtime: `674.10s`

Lane summary:

- `ruff check` passed (`0.03s`)
- typed-core `mypy` passed (`0.14s`)
- full `pytest` passed (`2.04s`)
- `tests/test_reference_parity.py` passed (`8.76s`)
- `scripts/eval_librispeech.py` passed (`21.35s`)
- `scripts/eval_manifest_quality.py` passed (`144.36s`)
- `scripts/eval_reference_parity_suite.py` passed (`493.98s`)
- `scripts/benchmark_asr.py` passed (`3.37s`)
  - note: `rtf=0.1526`, `latency_mean=0.3865s`

## Typed-Core Note

A typed-core blocker was fixed in `mlx_qwen3_asr/encoder.py` by explicitly
casting the fp16 clamp bound to `float` so `mx.clip` is type-compatible under
mypy.

## Readiness Assessment

The Python implementation is in a strong release-gated state for:

- deterministic parity checks,
- baseline English quality lane,
- multilingual manifest quality lane,
- performance regression guardrails,
- typed-core static checking.

## Residual Gaps (Intentional / Next)

These remain before calling Python "fully complete" for all quality claims:

- expand multilingual quality evaluation breadth (more languages/domains),
- increase hard-condition quality sets (noisy, overlap-heavy, long-form real audio),
- continue narrowing non-near token parity deltas where present,
- keep streaming marked experimental unless true incremental decoding is shipped.

## Swift Port Readiness

The current Python codebase is suitable as a correctness and benchmarking base
for Swift porting, with the caveat that parity/quality artifacts should continue
to be regenerated during the Swift implementation.
