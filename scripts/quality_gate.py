#!/usr/bin/env python3
"""Quality gate runner for mlx-qwen3-asr.

Usage:
  python scripts/quality_gate.py --mode fast
  python scripts/quality_gate.py --mode release
  RUN_STRICT_RELEASE=1 MANIFEST_QUALITY_EVAL_JSONL=... python scripts/quality_gate.py --mode release
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

MYPY_TYPED_TARGETS = [
    "mlx_qwen3_asr/config.py",
    "mlx_qwen3_asr/chunking.py",
    "mlx_qwen3_asr/attention.py",
    "mlx_qwen3_asr/mrope.py",
    "mlx_qwen3_asr/encoder.py",
    "mlx_qwen3_asr/decoder.py",
    "mlx_qwen3_asr/model.py",
    "mlx_qwen3_asr/generate.py",
    "mlx_qwen3_asr/audio.py",
    "mlx_qwen3_asr/tokenizer.py",
    "mlx_qwen3_asr/load_models.py",
    "mlx_qwen3_asr/cache_utils.py",
    "mlx_qwen3_asr/writers.py",
    "mlx_qwen3_asr/transcribe.py",
    "mlx_qwen3_asr/session.py",
]


@dataclass
class StepResult:
    name: str
    cmd: str
    passed: bool
    duration_sec: float
    returncode: int
    note: str = ""


def _run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> StepResult:
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), check=False, env=env)
    elapsed = time.perf_counter() - started
    return StepResult(
        name=cmd[0],
        cmd=" ".join(shlex.quote(c) for c in cmd),
        passed=proc.returncode == 0,
        duration_sec=elapsed,
        returncode=proc.returncode,
    )


def _tracked_py_files(repo: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    tracked = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return [path for path in tracked if (repo / path).exists()]


def _module_available(python_bin: str, module_name: str, repo: Path) -> bool:
    probe = "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)"
    proc = subprocess.run(
        [python_bin, "-c", probe, module_name],
        cwd=str(repo),
        check=False,
    )
    return proc.returncode == 0


def _run_perf_benchmark_gate(
    *,
    repo: Path,
    python_bin: str,
    strict_release: bool,
) -> StepResult:
    audio_path = os.environ.get("PERF_BENCH_AUDIO")
    if not audio_path:
        default_audio = repo / "tests" / "fixtures" / "test_speech.wav"
        if default_audio.exists():
            audio_path = str(default_audio)
        else:
            return StepResult(
                name="perf-benchmark",
                cmd="scripts/benchmark_asr.py",
                passed=False,
                duration_sec=0.0,
                returncode=1,
                note="PERF_BENCH_AUDIO is required and no default fixture was found.",
            )

    fd, temp_json = tempfile.mkstemp(prefix="mlx_qwen3_asr_perf_", suffix=".json")
    os.close(fd)
    perf_json_output = os.environ.get("PERF_BENCH_JSON_OUTPUT", temp_json)

    cmd = [
        python_bin,
        str(repo / "scripts" / "benchmark_asr.py"),
        audio_path,
        "--model",
        os.environ.get("PERF_BENCH_MODEL", "Qwen/Qwen3-ASR-0.6B"),
        "--dtype",
        os.environ.get("PERF_BENCH_DTYPE", "float16"),
        "--warmup-runs",
        os.environ.get("PERF_BENCH_WARMUP_RUNS", "1"),
        "--runs",
        os.environ.get("PERF_BENCH_RUNS", "3"),
        "--max-new-tokens",
        os.environ.get("PERF_BENCH_MAX_NEW_TOKENS", "256"),
        "--json-output",
        perf_json_output,
    ]
    step = _run(cmd, repo)
    if not step.passed:
        return StepResult(
            name="perf-benchmark",
            cmd=step.cmd,
            passed=False,
            duration_sec=step.duration_sec,
            returncode=step.returncode,
        )

    try:
        payload = json.loads(Path(perf_json_output).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return StepResult(
            name="perf-benchmark",
            cmd=step.cmd,
            passed=False,
            duration_sec=step.duration_sec,
            returncode=1,
            note=f"Failed to parse perf benchmark JSON: {exc}",
        )

    rtf_threshold = float(
        os.environ.get(
            "PERF_BENCH_FAIL_RTF_ABOVE",
            "0.50" if strict_release else "1.00",
        )
    )
    latency_threshold_env = os.environ.get("PERF_BENCH_FAIL_LATENCY_MEAN_ABOVE")
    if strict_release and not latency_threshold_env:
        latency_threshold_env = "2.00"
    latency_threshold = float(latency_threshold_env) if latency_threshold_env else None

    rtf = float(payload.get("rtf", 0.0))
    latency_mean = float(payload.get("latency_sec", {}).get("mean", 0.0))
    failures: list[str] = []
    if rtf > rtf_threshold:
        failures.append(f"rtf={rtf:.4f} > threshold={rtf_threshold:.4f}")
    if latency_threshold is not None and latency_mean > latency_threshold:
        failures.append(
            f"latency_mean={latency_mean:.4f}s > threshold={latency_threshold:.4f}s"
        )

    return StepResult(
        name="perf-benchmark",
        cmd=step.cmd,
        passed=not failures,
        duration_sec=step.duration_sec,
        returncode=0 if not failures else 1,
        note=(
            f"rtf={rtf:.4f}, latency_mean={latency_mean:.4f}s"
            if not failures
            else "; ".join(failures)
        ),
    )


def _run_streaming_quality_gate(
    *,
    repo: Path,
    python_bin: str,
    strict_release: bool,
) -> StepResult:
    audio_path = os.environ.get("STREAMING_QUALITY_AUDIO")
    if not audio_path:
        default_audio = repo / "tests" / "fixtures" / "test_speech.wav"
        if default_audio.exists():
            audio_path = str(default_audio)
        else:
            return StepResult(
                name="streaming-quality",
                cmd="scripts/eval_streaming_metrics.py",
                passed=False,
                duration_sec=0.0,
                returncode=1,
                note=(
                    "STREAMING_QUALITY_AUDIO is required and no default fixture "
                    "was found."
                ),
            )

    mode_list = os.environ.get(
        "STREAMING_QUALITY_ENDPOINTING_MODES",
        "fixed,energy" if strict_release else "fixed",
    )
    endpointing_modes = [mode.strip() for mode in mode_list.split(",") if mode.strip()]
    if not endpointing_modes:
        return StepResult(
            name="streaming-quality",
            cmd="scripts/eval_streaming_metrics.py",
            passed=False,
            duration_sec=0.0,
            returncode=1,
            note="No endpointing modes configured for streaming quality gate.",
        )

    partial_stability_min = float(
        os.environ.get(
            "STREAMING_QUALITY_FAIL_PARTIAL_STABILITY_BELOW",
            "0.85" if strict_release else "0.0",
        )
    )
    rewrite_rate_max = float(
        os.environ.get(
            "STREAMING_QUALITY_FAIL_REWRITE_RATE_ABOVE",
            "0.30" if strict_release else "1.0",
        )
    )
    final_delta_max = int(
        os.environ.get(
            "STREAMING_QUALITY_FAIL_FINALIZATION_DELTA_CHARS_ABOVE",
            "32" if strict_release else "1000000",
        )
    )

    notes: list[str] = []
    total_duration = 0.0
    cmd_display = ""

    for mode in endpointing_modes:
        fd, temp_json = tempfile.mkstemp(
            prefix=f"mlx_qwen3_asr_streaming_{mode}_",
            suffix=".json",
        )
        os.close(fd)
        json_output = temp_json
        cmd = [
            python_bin,
            str(repo / "scripts" / "eval_streaming_metrics.py"),
            audio_path,
            "--model",
            os.environ.get("STREAMING_QUALITY_MODEL", "Qwen/Qwen3-ASR-0.6B"),
            "--dtype",
            os.environ.get("STREAMING_QUALITY_DTYPE", "float16"),
            "--chunk-size-sec",
            os.environ.get("STREAMING_QUALITY_CHUNK_SIZE_SEC", "2.0"),
            "--max-context-sec",
            os.environ.get("STREAMING_QUALITY_MAX_CONTEXT_SEC", "30.0"),
            "--endpointing-mode",
            mode,
            "--finalization-mode",
            os.environ.get("STREAMING_QUALITY_FINALIZATION_MODE", "accuracy"),
            "--unfixed-chunk-num",
            os.environ.get("STREAMING_QUALITY_UNFIXED_CHUNK_NUM", "2"),
            "--unfixed-token-num",
            os.environ.get("STREAMING_QUALITY_UNFIXED_TOKEN_NUM", "5"),
            "--json-output",
            json_output,
        ]
        step = _run(cmd, repo)
        total_duration += step.duration_sec
        cmd_display = step.cmd
        if not step.passed:
            return StepResult(
                name="streaming-quality",
                cmd=step.cmd,
                passed=False,
                duration_sec=total_duration,
                returncode=step.returncode,
                note=f"{mode}: eval_streaming_metrics failed",
            )

        try:
            payload = json.loads(Path(json_output).read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            return StepResult(
                name="streaming-quality",
                cmd=step.cmd,
                passed=False,
                duration_sec=total_duration,
                returncode=1,
                note=f"{mode}: failed to parse streaming metrics JSON: {exc}",
            )

        final_metrics = payload.get("final_metrics", {})
        partial_stability = float(final_metrics.get("partial_stability", 0.0))
        rewrite_rate = float(final_metrics.get("rewrite_rate", 1.0))
        final_delta = int(final_metrics.get("finalization_delta_chars", 0))

        failures: list[str] = []
        if partial_stability < partial_stability_min:
            failures.append(
                f"partial_stability={partial_stability:.4f} < {partial_stability_min:.4f}"
            )
        if rewrite_rate > rewrite_rate_max:
            failures.append(f"rewrite_rate={rewrite_rate:.4f} > {rewrite_rate_max:.4f}")
        if final_delta > final_delta_max:
            failures.append(f"finalization_delta_chars={final_delta} > {final_delta_max}")
        if failures:
            return StepResult(
                name="streaming-quality",
                cmd=step.cmd,
                passed=False,
                duration_sec=total_duration,
                returncode=1,
                note=f"{mode}: " + "; ".join(failures),
            )

        notes.append(
            f"{mode}: stability={partial_stability:.4f}, "
            f"rewrite_rate={rewrite_rate:.4f}, final_delta={final_delta}"
        )

    return StepResult(
        name="streaming-quality",
        cmd=cmd_display,
        passed=True,
        duration_sec=total_duration,
        returncode=0,
        note=" | ".join(notes),
    )


def _run_streaming_manifest_quality_gate(
    *,
    repo: Path,
    python_bin: str,
    strict_release: bool,
) -> StepResult:
    manifest_jsonl = os.environ.get("STREAMING_MANIFEST_QUALITY_EVAL_JSONL")
    if not manifest_jsonl:
        return StepResult(
            name="streaming-manifest-quality",
            cmd="scripts/eval_streaming_manifest.py --manifest-jsonl <path>",
            passed=False,
            duration_sec=0.0,
            returncode=1,
            note=(
                "RUN_STREAMING_MANIFEST_QUALITY_EVAL=1 requires "
                "STREAMING_MANIFEST_QUALITY_EVAL_JSONL"
            ),
        )

    manifest_path = Path(manifest_jsonl).expanduser().resolve()
    if not manifest_path.exists():
        return StepResult(
            name="streaming-manifest-quality",
            cmd="scripts/eval_streaming_manifest.py --manifest-jsonl <path>",
            passed=False,
            duration_sec=0.0,
            returncode=1,
            note=f"Manifest not found: {manifest_path}",
        )

    try:
        missing_audio = _missing_manifest_audio_paths(manifest_path)
    except RuntimeError as exc:
        return StepResult(
            name="streaming-manifest-quality",
            cmd="scripts/eval_streaming_manifest.py --manifest-jsonl <path>",
            passed=False,
            duration_sec=0.0,
            returncode=1,
            note=str(exc),
        )

    if missing_audio:
        return StepResult(
            name="streaming-manifest-quality",
            cmd="scripts/eval_streaming_manifest.py --manifest-jsonl <path>",
            passed=False,
            duration_sec=0.0,
            returncode=1,
            note=(
                "Streaming manifest references missing local audio files "
                f"(example: {missing_audio[0]}). Build/capture the dataset manifest first."
            ),
        )

    cmd = [
        python_bin,
        str(repo / "scripts" / "eval_streaming_manifest.py"),
        "--manifest-jsonl",
        str(manifest_path),
        "--model",
        os.environ.get("STREAMING_MANIFEST_QUALITY_EVAL_MODEL", "Qwen/Qwen3-ASR-0.6B"),
        "--dtype",
        os.environ.get("STREAMING_MANIFEST_QUALITY_EVAL_DTYPE", "float16"),
        "--endpointing-modes",
        os.environ.get(
            "STREAMING_MANIFEST_QUALITY_EVAL_ENDPOINTING_MODES",
            "fixed,energy",
        ),
        "--chunk-size-sec",
        os.environ.get("STREAMING_MANIFEST_QUALITY_EVAL_CHUNK_SIZE_SEC", "2.0"),
        "--max-context-sec",
        os.environ.get("STREAMING_MANIFEST_QUALITY_EVAL_MAX_CONTEXT_SEC", "30.0"),
        "--finalization-mode",
        os.environ.get("STREAMING_MANIFEST_QUALITY_EVAL_FINALIZATION_MODE", "accuracy"),
        "--unfixed-chunk-num",
        os.environ.get("STREAMING_MANIFEST_QUALITY_EVAL_UNFIXED_CHUNK_NUM", "2"),
        "--unfixed-token-num",
        os.environ.get("STREAMING_MANIFEST_QUALITY_EVAL_UNFIXED_TOKEN_NUM", "5"),
        "--fail-partial-stability-below",
        os.environ.get("STREAMING_MANIFEST_QUALITY_EVAL_FAIL_PARTIAL_STABILITY_BELOW", "0.85"),
        "--fail-rewrite-rate-above",
        os.environ.get("STREAMING_MANIFEST_QUALITY_EVAL_FAIL_REWRITE_RATE_ABOVE", "0.30"),
        "--fail-finalization-delta-chars-above",
        os.environ.get(
            "STREAMING_MANIFEST_QUALITY_EVAL_FAIL_FINALIZATION_DELTA_CHARS_ABOVE",
            "32",
        ),
    ]
    limit = os.environ.get("STREAMING_MANIFEST_QUALITY_EVAL_LIMIT")
    if limit:
        cmd.extend(["--limit", limit])
    json_output = os.environ.get("STREAMING_MANIFEST_QUALITY_EVAL_JSON_OUTPUT")
    if json_output:
        cmd.extend(["--json-output", json_output])
    return _run(cmd, repo)


def _missing_manifest_audio_paths(
    manifest_jsonl: Path,
    *,
    max_examples: int = 3,
) -> list[Path]:
    missing: list[Path] = []
    for line_no, raw in enumerate(
        manifest_jsonl.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        row = raw.strip()
        if not row:
            continue
        try:
            obj = json.loads(row)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid JSON in manifest row {line_no}: {manifest_jsonl}"
            ) from exc
        audio_path_value = obj.get("audio_path")
        if not audio_path_value:
            raise RuntimeError(
                f"Manifest row {line_no} missing audio_path: {manifest_jsonl}"
            )
        audio_path = Path(str(audio_path_value)).expanduser().resolve()
        if not audio_path.exists():
            missing.append(audio_path)
            if len(missing) >= max_examples:
                break
    return missing


def _run_realworld_longform_quality_gate(
    *,
    repo: Path,
    python_bin: str,
    strict_release: bool,
) -> StepResult:
    manifest_jsonl = os.environ.get("REALWORLD_LONGFORM_EVAL_JSONL")
    if not manifest_jsonl and strict_release:
        default_manifest = (
            repo / "docs" / "benchmarks" / "2026-02-15-earnings22-full-longform3-manifest.jsonl"
        )
        if default_manifest.exists():
            manifest_jsonl = str(default_manifest)

    if not manifest_jsonl:
        return StepResult(
            name="realworld-longform-quality",
            cmd="scripts/eval_manifest_quality.py --manifest-jsonl <path>",
            passed=False,
            duration_sec=0.0,
            returncode=1,
            note=(
                "RUN_REALWORLD_LONGFORM_EVAL=1 requires REALWORLD_LONGFORM_EVAL_JSONL."
            ),
        )

    manifest_path = Path(manifest_jsonl).expanduser().resolve()
    if not manifest_path.exists():
        return StepResult(
            name="realworld-longform-quality",
            cmd="scripts/eval_manifest_quality.py --manifest-jsonl <path>",
            passed=False,
            duration_sec=0.0,
            returncode=1,
            note=f"Manifest not found: {manifest_path}",
        )

    try:
        missing_audio = _missing_manifest_audio_paths(manifest_path)
    except RuntimeError as exc:
        return StepResult(
            name="realworld-longform-quality",
            cmd="scripts/eval_manifest_quality.py --manifest-jsonl <path>",
            passed=False,
            duration_sec=0.0,
            returncode=1,
            note=str(exc),
        )

    if missing_audio:
        first_missing = missing_audio[0]
        return StepResult(
            name="realworld-longform-quality",
            cmd="scripts/eval_manifest_quality.py --manifest-jsonl <path>",
            passed=False,
            duration_sec=0.0,
            returncode=1,
            note=(
                "Real-world long-form manifest references missing local audio files "
                f"(example: {first_missing}). Build/cache the dataset with: "
                "python scripts/build_earnings22_longform_manifest.py"
            ),
        )

    cmd = [
        python_bin,
        str(repo / "scripts" / "eval_manifest_quality.py"),
        "--manifest-jsonl",
        str(manifest_path),
        "--model",
        os.environ.get("REALWORLD_LONGFORM_EVAL_MODEL", "Qwen/Qwen3-ASR-0.6B"),
        "--dtype",
        os.environ.get("REALWORLD_LONGFORM_EVAL_DTYPE", "float16"),
        "--max-new-tokens",
        os.environ.get("REALWORLD_LONGFORM_EVAL_MAX_NEW_TOKENS", "1024"),
        "--fail-primary-above",
        os.environ.get(
            "REALWORLD_LONGFORM_EVAL_FAIL_PRIMARY_ABOVE",
            "0.20" if strict_release else "0.35",
        ),
    ]
    fail_wer = os.environ.get("REALWORLD_LONGFORM_EVAL_FAIL_WER_ABOVE")
    if fail_wer:
        cmd.extend(["--fail-wer-above", fail_wer])
    fail_cer = os.environ.get("REALWORLD_LONGFORM_EVAL_FAIL_CER_ABOVE")
    if fail_cer:
        cmd.extend(["--fail-cer-above", fail_cer])
    limit = os.environ.get("REALWORLD_LONGFORM_EVAL_LIMIT")
    if limit:
        cmd.extend(["--limit", limit])
    json_output = os.environ.get("REALWORLD_LONGFORM_EVAL_JSON_OUTPUT")
    if json_output:
        cmd.extend(["--json-output", json_output])
    return _run(cmd, repo)


def run_gate(mode: str, repo: Path, python_bin: str) -> tuple[list[StepResult], bool]:
    steps: list[StepResult] = []
    strict_release = mode == "release" and os.environ.get("RUN_STRICT_RELEASE", "0") == "1"

    tracked = _tracked_py_files(repo)
    if not tracked:
        steps.append(
            StepResult(
                name="ruff",
                cmd="",
                passed=False,
                duration_sec=0.0,
                returncode=1,
                note="No tracked Python files found.",
            )
        )
        return steps, False

    steps.append(_run([python_bin, "-m", "ruff", "check", *tracked], repo))
    steps.append(
        _run(
            [
                python_bin,
                "-m",
                "mypy",
                "--follow-imports=skip",
                "--ignore-missing-imports",
                *MYPY_TYPED_TARGETS,
            ],
            repo,
        )
    )
    steps.append(_run([python_bin, "-m", "pytest", "-q"], repo))

    if mode == "release":
        if not (repo / "tests" / "test_reference_parity.py").exists():
            steps.append(
                StepResult(
                    name="reference-parity",
                    cmd="pytest -q tests/test_reference_parity.py",
                    passed=False,
                    duration_sec=0.0,
                    returncode=1,
                    note="Missing tests/test_reference_parity.py",
                )
            )
        else:
            require_parity_deps = os.environ.get("REQUIRE_REFERENCE_PARITY_DEPS", "1") == "1"
            missing = [
                mod
                for mod in ("torch", "qwen_asr")
                if not _module_available(python_bin, mod, repo)
            ]
            if missing and require_parity_deps:
                steps.append(
                    StepResult(
                        name="reference-parity",
                        cmd=f"{python_bin} -m pytest -q tests/test_reference_parity.py",
                        passed=False,
                        duration_sec=0.0,
                        returncode=1,
                        note=(
                            "Missing required parity dependencies: "
                            f"{', '.join(missing)}. Install with: "
                            "pip install 'mlx-qwen3-asr[aligner]'"
                        ),
                    )
                )
            else:
                parity_env = dict(os.environ)
                parity_env["RUN_REFERENCE_PARITY"] = "1"
                steps.append(
                    _run(
                        [python_bin, "-m", "pytest", "-q", "tests/test_reference_parity.py"],
                        repo,
                        env=parity_env,
                    )
                )

        if os.environ.get("RUN_QUALITY_EVAL", "1") == "1":
            quality_eval_cmd = [
                python_bin,
                str(repo / "scripts" / "eval_librispeech.py"),
                "--model",
                os.environ.get("QUALITY_EVAL_MODEL", "Qwen/Qwen3-ASR-0.6B"),
                "--dtype",
                os.environ.get("QUALITY_EVAL_DTYPE", "float16"),
                "--subset",
                os.environ.get("QUALITY_EVAL_SUBSET", "test-clean"),
                "--samples",
                os.environ.get("QUALITY_EVAL_SAMPLES", "20"),
                "--sampling",
                os.environ.get("QUALITY_EVAL_SAMPLING", "speaker_round_robin"),
                "--max-new-tokens",
                os.environ.get("QUALITY_EVAL_MAX_NEW_TOKENS", "256"),
                "--fail-wer-above",
                os.environ.get("QUALITY_EVAL_FAIL_WER_ABOVE", "0.10"),
                "--fail-cer-above",
                os.environ.get("QUALITY_EVAL_FAIL_CER_ABOVE", "0.06"),
            ]
            quality_json = os.environ.get("QUALITY_EVAL_JSON_OUTPUT")
            if quality_json:
                quality_eval_cmd.extend(["--json-output", quality_json])
            steps.append(_run(quality_eval_cmd, repo))

        if os.environ.get("RUN_MANIFEST_QUALITY_EVAL", "1" if strict_release else "0") == "1":
            manifest_jsonl = os.environ.get("MANIFEST_QUALITY_EVAL_JSONL")
            if not manifest_jsonl:
                steps.append(
                    StepResult(
                        name="manifest-quality-eval",
                        cmd="scripts/eval_manifest_quality.py --manifest-jsonl <path>",
                        passed=False,
                        duration_sec=0.0,
                        returncode=1,
                        note="RUN_MANIFEST_QUALITY_EVAL=1 requires MANIFEST_QUALITY_EVAL_JSONL",
                    )
                )
            else:
                manifest_quality_cmd = [
                    python_bin,
                    str(repo / "scripts" / "eval_manifest_quality.py"),
                    "--manifest-jsonl",
                    manifest_jsonl,
                    "--model",
                    os.environ.get("MANIFEST_QUALITY_EVAL_MODEL", "Qwen/Qwen3-ASR-0.6B"),
                    "--dtype",
                    os.environ.get("MANIFEST_QUALITY_EVAL_DTYPE", "float16"),
                    "--max-new-tokens",
                    os.environ.get("MANIFEST_QUALITY_EVAL_MAX_NEW_TOKENS", "1024"),
                    "--fail-primary-above",
                    os.environ.get("MANIFEST_QUALITY_EVAL_FAIL_PRIMARY_ABOVE", "0.35"),
                ]
                fail_wer = os.environ.get("MANIFEST_QUALITY_EVAL_FAIL_WER_ABOVE")
                if fail_wer:
                    manifest_quality_cmd.extend(["--fail-wer-above", fail_wer])
                fail_cer = os.environ.get("MANIFEST_QUALITY_EVAL_FAIL_CER_ABOVE")
                if fail_cer:
                    manifest_quality_cmd.extend(["--fail-cer-above", fail_cer])
                limit = os.environ.get("MANIFEST_QUALITY_EVAL_LIMIT")
                if limit:
                    manifest_quality_cmd.extend(["--limit", limit])
                quality_json = os.environ.get("MANIFEST_QUALITY_EVAL_JSON_OUTPUT")
                if quality_json:
                    manifest_quality_cmd.extend(["--json-output", quality_json])
                steps.append(_run(manifest_quality_cmd, repo))

        if os.environ.get(
            "RUN_REALWORLD_LONGFORM_EVAL",
            "1" if strict_release else "0",
        ) == "1":
            steps.append(
                _run_realworld_longform_quality_gate(
                    repo=repo,
                    python_bin=python_bin,
                    strict_release=strict_release,
                )
            )

        if os.environ.get("RUN_DIARIZATION_QUALITY_EVAL", "0") == "1":
            diar_manifest_jsonl = os.environ.get("DIARIZATION_QUALITY_EVAL_JSONL")
            if not diar_manifest_jsonl:
                steps.append(
                    StepResult(
                        name="diarization-quality-eval",
                        cmd="scripts/eval_diarization.py --manifest-jsonl <path>",
                        passed=False,
                        duration_sec=0.0,
                        returncode=1,
                        note=(
                            "RUN_DIARIZATION_QUALITY_EVAL=1 requires "
                            "DIARIZATION_QUALITY_EVAL_JSONL"
                        ),
                    )
                )
            else:
                diar_quality_cmd = [
                    python_bin,
                    str(repo / "scripts" / "eval_diarization.py"),
                    "--manifest-jsonl",
                    diar_manifest_jsonl,
                    "--model",
                    os.environ.get("DIARIZATION_QUALITY_EVAL_MODEL", "Qwen/Qwen3-ASR-0.6B"),
                    "--dtype",
                    os.environ.get("DIARIZATION_QUALITY_EVAL_DTYPE", "float16"),
                    "--max-new-tokens",
                    os.environ.get("DIARIZATION_QUALITY_EVAL_MAX_NEW_TOKENS", "1024"),
                    "--min-speakers",
                    os.environ.get("DIARIZATION_QUALITY_EVAL_MIN_SPEAKERS", "1"),
                    "--max-speakers",
                    os.environ.get("DIARIZATION_QUALITY_EVAL_MAX_SPEAKERS", "8"),
                    "--frame-step-sec",
                    os.environ.get("DIARIZATION_QUALITY_EVAL_FRAME_STEP_SEC", "0.02"),
                    "--collar-sec",
                    os.environ.get("DIARIZATION_QUALITY_EVAL_COLLAR_SEC", "0.25"),
                ]
                num_speakers = os.environ.get("DIARIZATION_QUALITY_EVAL_NUM_SPEAKERS")
                if num_speakers:
                    diar_quality_cmd.extend(["--num-speakers", num_speakers])
                limit = os.environ.get("DIARIZATION_QUALITY_EVAL_LIMIT")
                if limit:
                    diar_quality_cmd.extend(["--limit", limit])
                fail_der = os.environ.get("DIARIZATION_QUALITY_EVAL_FAIL_DER_ABOVE")
                if fail_der:
                    diar_quality_cmd.extend(["--fail-der-above", fail_der])
                fail_jer = os.environ.get("DIARIZATION_QUALITY_EVAL_FAIL_JER_ABOVE")
                if fail_jer:
                    diar_quality_cmd.extend(["--fail-jer-above", fail_jer])
                if os.environ.get("DIARIZATION_QUALITY_EVAL_IGNORE_OVERLAP", "1") == "1":
                    diar_quality_cmd.append("--ignore-overlap")
                quality_json = os.environ.get("DIARIZATION_QUALITY_EVAL_JSON_OUTPUT")
                if quality_json:
                    diar_quality_cmd.extend(["--json-output", quality_json])
                steps.append(_run(diar_quality_cmd, repo))

        if os.environ.get("RUN_ALIGNER_PARITY") == "1":
            samples = os.environ.get("ALIGNER_PARITY_SAMPLES", "10")
            steps.append(
                _run(
                    [
                        python_bin,
                        str(repo / "scripts" / "eval_aligner_parity.py"),
                        "--subset",
                        "test-clean",
                        "--samples",
                        samples,
                        "--model",
                        "Qwen/Qwen3-ForcedAligner-0.6B",
                        "--fail-text-match-rate-below",
                        "1.0",
                        "--fail-timing-mae-ms-above",
                        "60.0",
                    ],
                    repo,
                )
            )

        if os.environ.get(
            "RUN_REFERENCE_PARITY_SUITE",
            "1" if strict_release else "0",
        ) == "1":
            subsets = os.environ.get(
                "REFERENCE_PARITY_SUITE_SUBSETS",
                "" if strict_release else "test-clean,test-other",
            )
            samples_per_subset = os.environ.get(
                "REFERENCE_PARITY_SUITE_SAMPLES_PER_SUBSET",
                "3",
            )
            max_new_tokens = os.environ.get("REFERENCE_PARITY_SUITE_MAX_NEW_TOKENS", "128")
            fail_match_rate = os.environ.get(
                "REFERENCE_PARITY_SUITE_FAIL_MATCH_RATE_BELOW",
                "0.58" if strict_release else "1.0",
            )
            fail_text_match_rate = os.environ.get(
                "REFERENCE_PARITY_SUITE_FAIL_TEXT_MATCH_RATE_BELOW",
                "0.61" if strict_release else None,
            )
            cmd = [
                python_bin,
                str(repo / "scripts" / "eval_reference_parity_suite.py"),
                "--model",
                os.environ.get("REFERENCE_PARITY_SUITE_MODEL", "Qwen/Qwen3-ASR-0.6B"),
                "--subsets",
                subsets,
                "--samples-per-subset",
                samples_per_subset,
                "--max-new-tokens",
                max_new_tokens,
                "--fail-match-rate-below",
                fail_match_rate,
            ]
            manifest_jsonl = os.environ.get("REFERENCE_PARITY_SUITE_MANIFEST_JSONL")
            if strict_release and not manifest_jsonl:
                default_manifest = (
                    repo
                    / "docs"
                    / "benchmarks"
                    / "2026-02-14-fleurs-multilingual-100-manifest.jsonl"
                )
                if default_manifest.exists():
                    manifest_jsonl = str(default_manifest)
            if manifest_jsonl:
                cmd.extend(["--manifest-jsonl", manifest_jsonl])
            if os.environ.get("REFERENCE_PARITY_SUITE_INCLUDE_LONG_MIXES", "1") == "1":
                cmd.append("--include-long-mixes")
                cmd.extend(
                    [
                        "--long-mixes",
                        os.environ.get("REFERENCE_PARITY_SUITE_LONG_MIXES", "2"),
                        "--long-mix-segments",
                        os.environ.get("REFERENCE_PARITY_SUITE_LONG_MIX_SEGMENTS", "4"),
                        "--long-mix-silence-sec",
                        os.environ.get("REFERENCE_PARITY_SUITE_LONG_MIX_SILENCE_SEC", "0.3"),
                    ]
                )
            if os.environ.get("REFERENCE_PARITY_SUITE_INCLUDE_NOISE_VARIANTS", "0") == "1":
                cmd.append("--include-noise-variants")
                cmd.extend(
                    [
                        "--noise-snrs-db",
                        os.environ.get("REFERENCE_PARITY_SUITE_NOISE_SNRS_DB", "10,5"),
                        "--noise-seed",
                        os.environ.get("REFERENCE_PARITY_SUITE_NOISE_SEED", "20260214"),
                    ]
                )
            if fail_text_match_rate:
                cmd.extend(["--fail-text-match-rate-below", fail_text_match_rate])
            json_output = os.environ.get("REFERENCE_PARITY_SUITE_JSON_OUTPUT")
            if json_output:
                cmd.extend(["--json-output", json_output])
            steps.append(_run(cmd, repo))

        if os.environ.get(
            "RUN_PERF_BENCHMARK",
            "1" if strict_release else "0",
        ) == "1":
            steps.append(
                _run_perf_benchmark_gate(
                    repo=repo,
                    python_bin=python_bin,
                    strict_release=strict_release,
                )
            )
        if os.environ.get(
            "RUN_STREAMING_QUALITY_EVAL",
            "1" if strict_release else "0",
        ) == "1":
            steps.append(
                _run_streaming_quality_gate(
                    repo=repo,
                    python_bin=python_bin,
                    strict_release=strict_release,
                )
            )
        if os.environ.get(
            "RUN_STREAMING_MANIFEST_QUALITY_EVAL",
            "0",
        ) == "1":
            steps.append(
                _run_streaming_manifest_quality_gate(
                    repo=repo,
                    python_bin=python_bin,
                    strict_release=strict_release,
                )
            )

    ok = all(step.passed for step in steps)
    return steps, ok


def _resolve_python(repo: Path) -> str:
    venv_py = repo / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repository quality gate checks.")
    parser.add_argument(
        "--mode",
        choices=["fast", "release"],
        default="fast",
        help="fast: lint+tests, release: fast + quality/parity/perf release lanes",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path to write gate result JSON.",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    python_bin = _resolve_python(repo)

    started = time.perf_counter()
    steps, ok = run_gate(args.mode, repo, python_bin)
    total = time.perf_counter() - started

    print(f"\nQuality Gate ({args.mode})")
    print("=" * 32)
    for step in steps:
        status = "PASS" if step.passed else "FAIL"
        print(f"[{status}] {step.cmd} ({step.duration_sec:.2f}s)")
        if step.note:
            print(f"       note: {step.note}")
    print(f"Total: {total:.2f}s")
    print(f"Result: {'PASS' if ok else 'FAIL'}")

    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": args.mode,
            "passed": ok,
            "total_duration_sec": total,
            "steps": [asdict(s) for s in steps],
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
