#!/usr/bin/env python3
"""Compare MLX manifest-quality artifact outputs vs qwen_asr reference."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from eval.metrics import edit_distance  # noqa: E402

from mlx_qwen3_asr.chunking import split_audio_into_chunks  # noqa: E402

_WS_RE = re.compile(r"\s+")


def _normalize_quality_text(text: str) -> str:
    s = unicodedata.normalize("NFKC", str(text or "")).casefold()
    out: list[str] = []
    for ch in s:
        if ch in {"’", "`"}:
            ch = "'"
        cat = unicodedata.category(ch)
        if cat and cat[0] in {"L", "N", "M"}:
            out.append(ch)
            continue
        if ch == "'":
            out.append(ch)
            continue
        if ch.isspace() or (cat and cat[0] in {"P", "S"}):
            out.append(" ")
    return _WS_RE.sub(" ", "".join(out)).strip()


def _wer_tokens(normalized: str) -> list[str]:
    if not normalized:
        return []
    if any(ch.isspace() for ch in normalized):
        return normalized.split()
    return list(normalized)


def _cer_tokens(normalized: str) -> list[str]:
    return list(normalized.replace(" ", ""))


def _is_char_primary_language(language: Optional[str]) -> bool:
    if not language:
        return False
    key = str(language).strip().lower().replace("-", "_").replace(" ", "_")
    char_primary = {
        "chinese",
        "japanese",
        "korean",
        "zh",
        "zh_cn",
        "cmn",
        "cmn_hans_cn",
        "ja",
        "ja_jp",
        "ko",
        "ko_kr",
    }
    if key in char_primary:
        return True
    prefixes = ("zh_", "cmn_", "ja_", "ko_")
    return key.startswith(prefixes)


def _select_language_arg(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None
    value = str(language).strip()
    if not value:
        return None
    lowered = value.lower()
    if lowered in {"unknown", "auto", "none"}:
        return None
    return value


def _error_rate(errors: int, total: int) -> float:
    return float(errors) / float(max(1, total))


def _read_audio(path: str) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "soundfile is required for manifest head-to-head evaluation. "
            "Install with: pip install 'mlx-qwen3-asr[eval]'"
        ) from exc

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1).astype(np.float32)
    else:
        audio = audio.astype(np.float32)
    return audio, int(sr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare MLX manifest-quality artifact outputs against qwen_asr "
            "on the same samples."
        )
    )
    parser.add_argument(
        "--mlx-json",
        required=True,
        help="Path to eval_manifest_quality JSON artifact.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--md-output", default=None)
    parser.add_argument(
        "--reference-max-chunk-sec",
        type=float,
        default=0.0,
        help=(
            "Optional max chunk size (seconds) for reference transcribe. "
            "When >0, long clips are split by energy before qwen_asr inference."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress every N samples to stderr (0 disables).",
    )
    parser.add_argument(
        "--checkpoint-json",
        default=None,
        help=(
            "Optional path for incremental checkpoint payloads written during "
            "evaluation."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Write checkpoint payload every N samples (requires --checkpoint-json).",
    )
    args = parser.parse_args()
    if args.progress_every < 0:
        raise ValueError(f"--progress-every must be >= 0, got: {args.progress_every}")
    if args.checkpoint_every <= 0:
        raise ValueError(f"--checkpoint-every must be > 0, got: {args.checkpoint_every}")
    if args.reference_max_chunk_sec < 0:
        raise ValueError(
            f"--reference-max-chunk-sec must be >= 0, got: {args.reference_max_chunk_sec}"
        )

    try:
        import qwen_asr
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "qwen_asr + torch are required for head-to-head reference evaluation."
        ) from exc

    src = Path(args.mlx_json).expanduser().resolve()
    mlx_payload = json.loads(src.read_text(encoding="utf-8"))
    rows = mlx_payload.get("rows", [])
    if not rows:
        raise RuntimeError(f"No rows in MLX artifact: {src}")

    ref = qwen_asr.Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="cpu",
        max_new_tokens=args.max_new_tokens,
    )

    mlx_wer_err = mlx_wer_den = 0
    mlx_cer_err = mlx_cer_den = 0
    mlx_primary_err = mlx_primary_den = 0

    ref_wer_err = ref_wer_den = 0
    ref_cer_err = ref_cer_den = 0
    ref_primary_err = ref_primary_den = 0

    ref_latencies: list[float] = []
    out_rows: list[dict] = []
    by_language: dict[str, dict] = {}
    run_t0 = time.perf_counter()
    checkpoint_path = (
        Path(args.checkpoint_json).expanduser().resolve()
        if args.checkpoint_json
        else None
    )
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(rows, start=1):
        lang = row.get("language")
        lang_key = str(lang or "unknown")
        char_primary = _is_char_primary_language(lang)
        primary_metric = "cer" if char_primary else "wer"
        audio_path = str(row["audio_path"])

        ref_norm = _normalize_quality_text(str(row.get("reference_raw", "")))

        audio, sr = _read_audio(audio_path)
        language_arg = _select_language_arg(lang)
        if args.reference_max_chunk_sec > 0:
            chunks = split_audio_into_chunks(
                audio,
                sr=sr,
                max_chunk_sec=float(args.reference_max_chunk_sec),
            )
            ref_text_parts: list[str] = []
            dt = 0.0
            for chunk_audio, _offset in chunks:
                t0 = time.perf_counter()
                out = ref.transcribe((chunk_audio, sr), language=language_arg)
                dt += time.perf_counter() - t0
                text = out[0].text if out else ""
                text = str(text).strip()
                if text:
                    ref_text_parts.append(text)
            ref_text = " ".join(ref_text_parts).strip()
        else:
            t0 = time.perf_counter()
            out = ref.transcribe((audio, sr), language=language_arg)
            dt = time.perf_counter() - t0
            ref_text = out[0].text if out else ""
        ref_latencies.append(dt)
        ref_hyp_norm = _normalize_quality_text(ref_text)

        ref_wer_tokens = _wer_tokens(ref_norm)
        ref_hyp_wer_tokens = _wer_tokens(ref_hyp_norm)
        ref_cer_tokens = _cer_tokens(ref_norm)
        ref_hyp_cer_tokens = _cer_tokens(ref_hyp_norm)

        r_wer_err = int(edit_distance(ref_wer_tokens, ref_hyp_wer_tokens))
        r_wer_den = len(ref_wer_tokens)
        r_cer_err = int(edit_distance(ref_cer_tokens, ref_hyp_cer_tokens))
        r_cer_den = len(ref_cer_tokens)
        r_primary_err = r_cer_err if char_primary else r_wer_err
        r_primary_den = r_cer_den if char_primary else r_wer_den

        ref_wer_err += r_wer_err
        ref_wer_den += r_wer_den
        ref_cer_err += r_cer_err
        ref_cer_den += r_cer_den
        ref_primary_err += r_primary_err
        ref_primary_den += r_primary_den

        m_wer_err = int(row.get("wer_errors", 0))
        m_wer_den = int(row.get("wer_denominator", 0))
        m_cer_err = int(row.get("cer_errors", 0))
        m_cer_den = int(row.get("cer_denominator", 0))
        m_primary_err = m_cer_err if char_primary else m_wer_err
        m_primary_den = m_cer_den if char_primary else m_wer_den

        mlx_wer_err += m_wer_err
        mlx_wer_den += m_wer_den
        mlx_cer_err += m_cer_err
        mlx_cer_den += m_cer_den
        mlx_primary_err += m_primary_err
        mlx_primary_den += m_primary_den

        lang_stats = by_language.setdefault(
            lang_key,
            {
                "samples": 0,
                "primary_metric": primary_metric,
                "mlx_primary_errors": 0,
                "mlx_primary_denominator": 0,
                "pytorch_primary_errors": 0,
                "pytorch_primary_denominator": 0,
                "mlx_latency_sec_mean": 0.0,
                "pytorch_latency_sec_mean": 0.0,
            },
        )
        lang_stats["samples"] += 1
        lang_stats["mlx_primary_errors"] += m_primary_err
        lang_stats["mlx_primary_denominator"] += m_primary_den
        lang_stats["pytorch_primary_errors"] += r_primary_err
        lang_stats["pytorch_primary_denominator"] += r_primary_den
        lang_stats["mlx_latency_sec_mean"] += float(row.get("latency_sec", 0.0))
        lang_stats["pytorch_latency_sec_mean"] += dt

        out_rows.append(
            {
                "index": i,
                "sample_id": row.get("sample_id"),
                "subset": row.get("subset"),
                "speaker_id": row.get("speaker_id"),
                "language": lang,
                "audio_path": audio_path,
                "primary_metric": primary_metric,
                "reference_normalized": ref_norm,
                "mlx_hypothesis_normalized": row.get("hypothesis_normalized"),
                "pytorch_hypothesis_normalized": ref_hyp_norm,
                "mlx_latency_sec": row.get("latency_sec"),
                "pytorch_latency_sec": dt,
                "mlx_primary_error_rate": _error_rate(m_primary_err, m_primary_den),
                "pytorch_primary_error_rate": _error_rate(r_primary_err, r_primary_den),
            }
        )

        if args.progress_every > 0 and (i % args.progress_every == 0 or i == len(rows)):
            elapsed = time.perf_counter() - run_t0
            per_sample = elapsed / float(max(1, i))
            eta = per_sample * float(len(rows) - i)
            print(
                (
                    f"[{i}/{len(rows)}] "
                    f"sample_id={row.get('sample_id')} "
                    f"lang={lang_key} "
                    f"ref_primary={_error_rate(r_primary_err, r_primary_den):.4f} "
                    f"latency={dt:.2f}s "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
                ),
                file=sys.stderr,
                flush=True,
            )

        if checkpoint_path is not None and (
            i % args.checkpoint_every == 0 or i == len(rows)
        ):
            checkpoint_payload = {
                "suite": "quality-head2head-manifest-v1",
                "status": "partial" if i < len(rows) else "complete",
                "source_mlx_artifact": str(src),
                "model": args.model,
                "processed_samples": i,
                "samples_total": len(rows),
                "systems_partial": {
                    "mlx": {
                        "wer": _error_rate(mlx_wer_err, mlx_wer_den),
                        "cer": _error_rate(mlx_cer_err, mlx_cer_den),
                        "primary_error_rate": _error_rate(mlx_primary_err, mlx_primary_den),
                        "mean_latency_sec": float(
                            np.mean(
                                [
                                    float(r.get("mlx_latency_sec", 0.0))
                                    for r in out_rows
                                ]
                            )
                        )
                        if out_rows
                        else 0.0,
                    },
                    "pytorch_ref": {
                        "wer": _error_rate(ref_wer_err, ref_wer_den),
                        "cer": _error_rate(ref_cer_err, ref_cer_den),
                        "primary_error_rate": _error_rate(ref_primary_err, ref_primary_den),
                        "mean_latency_sec": float(np.mean(ref_latencies))
                        if ref_latencies
                        else 0.0,
                    },
                },
                "delta_partial": {
                    "wer_mlx_minus_ref": _error_rate(mlx_wer_err, mlx_wer_den)
                    - _error_rate(ref_wer_err, ref_wer_den),
                    "cer_mlx_minus_ref": _error_rate(mlx_cer_err, mlx_cer_den)
                    - _error_rate(ref_cer_err, ref_cer_den),
                    "primary_mlx_minus_ref": _error_rate(mlx_primary_err, mlx_primary_den)
                    - _error_rate(ref_primary_err, ref_primary_den),
                },
                "elapsed_sec": time.perf_counter() - run_t0,
                "rows": out_rows,
            }
            checkpoint_path.write_text(
                json.dumps(checkpoint_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    for stats in by_language.values():
        n = max(1, int(stats["samples"]))
        stats["mlx_primary_error_rate"] = _error_rate(
            int(stats["mlx_primary_errors"]), int(stats["mlx_primary_denominator"])
        )
        stats["pytorch_primary_error_rate"] = _error_rate(
            int(stats["pytorch_primary_errors"]), int(stats["pytorch_primary_denominator"])
        )
        stats["delta_primary_mlx_minus_ref"] = (
            stats["mlx_primary_error_rate"] - stats["pytorch_primary_error_rate"]
        )
        stats["mlx_latency_sec_mean"] = float(stats["mlx_latency_sec_mean"]) / float(n)
        stats["pytorch_latency_sec_mean"] = float(stats["pytorch_latency_sec_mean"]) / float(n)

    payload = {
        "suite": "quality-head2head-manifest-v1",
        "source_mlx_artifact": str(src),
        "model": args.model,
        "reference_max_chunk_sec": float(args.reference_max_chunk_sec),
        "samples": len(rows),
        "systems": {
            "mlx": {
                "wer": _error_rate(mlx_wer_err, mlx_wer_den),
                "cer": _error_rate(mlx_cer_err, mlx_cer_den),
                "primary_error_rate": _error_rate(mlx_primary_err, mlx_primary_den),
                "mean_latency_sec": float(mlx_payload.get("latency_sec_mean", 0.0)),
            },
            "pytorch_ref": {
                "wer": _error_rate(ref_wer_err, ref_wer_den),
                "cer": _error_rate(ref_cer_err, ref_cer_den),
                "primary_error_rate": _error_rate(ref_primary_err, ref_primary_den),
                "mean_latency_sec": float(np.mean(ref_latencies)) if ref_latencies else 0.0,
            },
        },
        "delta": {
            "wer_mlx_minus_ref": _error_rate(mlx_wer_err, mlx_wer_den)
            - _error_rate(ref_wer_err, ref_wer_den),
            "cer_mlx_minus_ref": _error_rate(mlx_cer_err, mlx_cer_den)
            - _error_rate(ref_cer_err, ref_cer_den),
            "primary_mlx_minus_ref": _error_rate(mlx_primary_err, mlx_primary_den)
            - _error_rate(ref_primary_err, ref_primary_den),
        },
        "by_language": by_language,
        "rows": out_rows,
    }

    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.json_output:
        out_json = Path(args.json_output).expanduser().resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.md_output:
        out_md = Path(args.md_output).expanduser().resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        systems = payload["systems"]
        suite_name = mlx_payload.get("suite", "manifest")
        lines = [
            f"# MLX vs PyTorch Quality Head-to-Head ({suite_name}, n={len(rows)})",
            "",
            f"- model: `{args.model}`",
            f"- samples: `{len(rows)}`",
            f"- reference_max_chunk_sec: `{float(args.reference_max_chunk_sec):.1f}`",
            f"- MLX primary: `{systems['mlx']['primary_error_rate']:.4f}`",
            f"- PyTorch primary: `{systems['pytorch_ref']['primary_error_rate']:.4f}`",
            (
                "- Delta primary (MLX-Ref): "
                f"`{payload['delta']['primary_mlx_minus_ref']:+.4f}`"
            ),
            "",
            "| System | Primary | WER | CER | Mean latency (s) |",
            "|---|---:|---:|---:|---:|",
            (
                f"| MLX | {systems['mlx']['primary_error_rate']:.4f} "
                f"| {systems['mlx']['wer']:.4f} | {systems['mlx']['cer']:.4f} "
                f"| {systems['mlx']['mean_latency_sec']:.4f} |"
            ),
            (
                f"| PyTorch ref | {systems['pytorch_ref']['primary_error_rate']:.4f} "
                f"| {systems['pytorch_ref']['wer']:.4f} | {systems['pytorch_ref']['cer']:.4f} "
                f"| {systems['pytorch_ref']['mean_latency_sec']:.4f} |"
            ),
        ]
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
