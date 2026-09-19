#!/usr/bin/env python3
"""Benchmark/evaluate fp16 and quantized model variants in one run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import _repo_path  # noqa: F401
import numpy as np

from mlx_qwen3_asr import load_audio


@dataclass(frozen=True)
class QuantConfig:
    label: str
    bits: int | None
    group_size: int | None


def _run_json(cmd: list[str], cwd: Path) -> dict:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=True,
    )
    text = proc.stdout.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0:
        raise RuntimeError(f"No JSON payload found in output:\n{text}")
    payload = json.loads(text[start : end + 1])
    return payload


def _ensure_long_fixture(short_audio: Path, long_audio: Path, seconds: int) -> None:
    if long_audio.exists():
        return
    audio = np.array(load_audio(str(short_audio)))
    sr = 16000
    target = seconds * sr
    tiled = np.tile(audio, int(np.ceil(target / len(audio))))[:target]
    pcm = (np.clip(tiled, -1.0, 1.0) * 32767).astype(np.int16)
    long_audio.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(long_audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())


def _parse_configs(raw: str) -> list[QuantConfig]:
    items = [x.strip().lower() for x in raw.split(",") if x.strip()]
    out: list[QuantConfig] = []
    for item in items:
        if item == "fp16":
            out.append(QuantConfig(label="fp16", bits=None, group_size=None))
            continue
        if ":" not in item:
            raise ValueError(
                f"Invalid quant config '{item}'. Expected 'fp16' or '<bits>:<group_size>'."
            )
        bits_str, group_str = item.split(":", 1)
        try:
            bits = int(bits_str)
            group = int(group_str)
        except ValueError as exc:
            raise ValueError(
                f"Invalid quant config '{item}'. bits/group_size must be integers."
            ) from exc
        if bits not in {4, 8}:
            raise ValueError(
                f"Unsupported bit-width '{bits}' in config '{item}'. Expected one of: 4, 8."
            )
        if group <= 0:
            raise ValueError(
                f"Invalid group_size '{group}' in config '{item}'. Must be > 0."
            )
        out.append(QuantConfig(label=f"{bits}bit-g{group}", bits=bits, group_size=group))
    if not out:
        raise ValueError("No configs specified.")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run quantization benchmark/eval matrix.")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--configs", default="fp16,4:64,4:32,8:64")
    parser.add_argument("--short-audio", default="tests/fixtures/test_speech.wav")
    parser.add_argument("--long-seconds", type=int, default=10)
    parser.add_argument("--benchmark-runs", type=int, default=7)
    parser.add_argument("--eval-subset", default="test-clean")
    parser.add_argument("--eval-samples", type=int, default=100)
    parser.add_argument(
        "--eval-sampling",
        choices=["speaker_round_robin", "sequential"],
        default="speaker_round_robin",
    )
    parser.add_argument(
        "--work-dir",
        default=str(Path.home() / ".cache" / "mlx-qwen3-asr" / "quant-matrix"),
    )
    parser.add_argument("--json-output", default="docs/benchmarks/quantization-matrix.json")
    parser.add_argument("--md-output", default="docs/benchmarks/quantization-matrix.md")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    venv_py = root / ".venv" / "bin" / "python"
    py = str(venv_py) if venv_py.exists() else sys.executable
    convert_py = str(root / "scripts" / "convert.py")
    bench_py = str(root / "scripts" / "benchmark_asr.py")
    eval_py = str(root / "scripts" / "eval_librispeech.py")

    short_audio = (root / args.short_audio).resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    long_audio = work_dir / f"long_{args.long_seconds}s.wav"
    _ensure_long_fixture(short_audio, long_audio, seconds=args.long_seconds)

    configs = _parse_configs(args.configs)
    results: list[dict] = []
    started = time.perf_counter()

    for cfg in configs:
        if cfg.bits is None:
            model_ref = args.model
        else:
            out_dir = work_dir / f"{cfg.bits}bit-g{cfg.group_size}"
            if not out_dir.exists():
                subprocess.run(
                    [
                        py,
                        convert_py,
                        "--model",
                        args.model,
                        "--quantize",
                        str(cfg.bits),
                        "--group-size",
                        str(cfg.group_size),
                        "--output-dir",
                        str(out_dir),
                    ],
                    cwd=str(root),
                    check=True,
                )
            model_ref = str(out_dir)

        short_json = _run_json(
            [
                py,
                bench_py,
                str(short_audio),
                "--model",
                model_ref,
                "--dtype",
                "float16",
                "--warmup-runs",
                "1",
                "--runs",
                str(args.benchmark_runs),
            ],
            cwd=root,
        )
        long_json = _run_json(
            [
                py,
                bench_py,
                str(long_audio),
                "--model",
                model_ref,
                "--dtype",
                "float16",
                "--warmup-runs",
                "1",
                "--runs",
                str(args.benchmark_runs),
            ],
            cwd=root,
        )
        eval_json = _run_json(
            [
                py,
                eval_py,
                "--subset",
                args.eval_subset,
                "--samples",
                str(args.eval_samples),
                "--sampling",
                args.eval_sampling,
                "--model",
                model_ref,
                "--dtype",
                "float16",
                "--max-new-tokens",
                "256",
            ],
            cwd=root,
        )
        results.append(
            {
                "config": cfg.label,
                "model_ref": model_ref,
                "short": short_json,
                "long": long_json,
                "eval": eval_json,
            }
        )

    elapsed = time.perf_counter() - started
    payload = {
        "model": args.model,
        "configs": [c.label for c in configs],
        "benchmark_runs": args.benchmark_runs,
        "eval_subset": args.eval_subset,
        "eval_samples": args.eval_samples,
        "eval_sampling": args.eval_sampling,
        "long_seconds": args.long_seconds,
        "elapsed_sec": elapsed,
        "results": results,
    }

    out_json = (root / args.json_output).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fp16 = next((r for r in results if r["config"] == "fp16"), None)
    lines = [
        "# Quantization Matrix",
        "",
        f"- Source model: `{args.model}`",
        f"- Eval subset: `{args.eval_subset}`",
        f"- Eval samples: `{args.eval_samples}`",
        f"- Eval sampling: `{args.eval_sampling}`",
        f"- Benchmark runs: `{args.benchmark_runs}`",
        f"- Long clip length: `{args.long_seconds}s`",
        "",
        "| Config | Short Mean (s) | Short RTF | Long Mean (s) | Long RTF | WER | CER | Eval RTF |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        s = r["short"]["latency_sec"]["mean"]
        sr = r["short"]["rtf"]
        long_mean = r["long"]["latency_sec"]["mean"]
        long_rtf = r["long"]["rtf"]
        w = r["eval"]["wer"]
        c = r["eval"]["cer"]
        er = r["eval"]["rtf"]
        lines.append(
            f"| {r['config']} | {s:.4f} | {sr:.4f} | {long_mean:.4f} "
            f"| {long_rtf:.4f} | {w:.6f} | {c:.6f} | {er:.4f} |"
        )

    if fp16 is not None:
        fp16_long_rtf = fp16["long"]["rtf"]
        lines.append("")
        lines.append("## Relative to fp16")
        lines.append("")
        for r in results:
            if r["config"] == "fp16":
                continue
            speedup = fp16_long_rtf / max(r["long"]["rtf"], 1e-9)
            lines.append(
                f"- `{r['config']}` long-clip speedup vs fp16: `{speedup:.2f}x` "
                f"(WER delta `{r['eval']['wer'] - fp16['eval']['wer']:+.6f}`)"
            )

    out_md = (root / args.md_output).resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"\nWrote: {out_json}")
    print(f"Wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
