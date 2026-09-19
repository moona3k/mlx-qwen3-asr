"""Text normalization and error-rate metrics for local eval scripts."""

from __future__ import annotations

import re
import unicodedata
from typing import Optional, Sequence

_NON_ALNUM_RE = re.compile(r"[^a-z0-9' ]+")
_SPACE_RE = re.compile(r"\s+")

_CHAR_PRIMARY_LANGUAGES = {
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
_CHAR_PRIMARY_PREFIXES = ("zh_", "cmn_", "ja_", "ko_")


def normalize_text(text: str) -> str:
    """Normalize text for stable ASR metric computation."""
    lowered = text.lower()
    cleaned = _NON_ALNUM_RE.sub(" ", lowered)
    return _SPACE_RE.sub(" ", cleaned).strip()


def normalize_quality_text(text: str) -> str:
    """Normalize text across scripts for stable multilingual metric computation.

    Keeps letters, numbers, marks and apostrophes; folds punctuation and
    symbols to whitespace. Shared by the manifest-quality and streaming
    manifest lanes so their error rates are directly comparable.
    """
    s = unicodedata.normalize("NFKC", str(text or "")).casefold()
    out: list[str] = []
    for ch in s:
        if ch in {"\u2019", "`"}:
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
    return _SPACE_RE.sub(" ", "".join(out)).strip()


def wer_tokens(normalized: str) -> list[str]:
    """Tokenize for WER; fall back to characters if no whitespace exists."""
    if not normalized:
        return []
    if any(ch.isspace() for ch in normalized):
        return normalized.split()
    return list(normalized)


def cer_tokens(normalized: str) -> list[str]:
    """Tokenize for CER (characters, whitespace removed)."""
    return list(normalized.replace(" ", ""))


def is_char_primary_language(language: Optional[str]) -> bool:
    """Return True when CER, not WER, is the primary metric for ``language``."""
    if not language:
        return False
    key = str(language).strip().lower().replace("-", "_").replace(" ", "_")
    if key in _CHAR_PRIMARY_LANGUAGES:
        return True
    return key.startswith(_CHAR_PRIMARY_PREFIXES)


def score_hypothesis(
    reference: str,
    hypothesis: str,
    language: Optional[str],
) -> dict[str, object]:
    """Score one hypothesis against its reference with the multilingual normalizer.

    Returns error counts and denominators for WER, CER and the language's
    primary metric so callers can aggregate at corpus level.
    """
    ref_norm = normalize_quality_text(reference)
    hyp_norm = normalize_quality_text(hypothesis)
    ref_wer = wer_tokens(ref_norm)
    hyp_wer = wer_tokens(hyp_norm)
    ref_cer = cer_tokens(ref_norm)
    hyp_cer = cer_tokens(hyp_norm)
    wer_err = int(edit_distance(ref_wer, hyp_wer))
    cer_err = int(edit_distance(ref_cer, hyp_cer))
    char_primary = is_char_primary_language(language)
    return {
        "reference_normalized": ref_norm,
        "hypothesis_normalized": hyp_norm,
        "wer_errors": wer_err,
        "wer_denominator": len(ref_wer),
        "cer_errors": cer_err,
        "cer_denominator": len(ref_cer),
        "primary_metric": "cer" if char_primary else "wer",
        "primary_errors": cer_err if char_primary else wer_err,
        "primary_denominator": len(ref_cer) if char_primary else len(ref_wer),
    }


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Compute Levenshtein distance between two token sequences."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference

    previous = list(range(len(hypothesis) + 1))
    for i, ref_tok in enumerate(reference, start=1):
        current = [i]
        for j, hyp_tok in enumerate(hypothesis, start=1):
            cost = 0 if ref_tok == hyp_tok else 1
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def compute_wer(reference_texts: Sequence[str], hypothesis_texts: Sequence[str]) -> float:
    """Compute corpus-level word error rate."""
    total_words = 0
    total_errors = 0
    for reference, hypothesis in zip(reference_texts, hypothesis_texts, strict=True):
        ref_tokens = normalize_text(reference).split()
        hyp_tokens = normalize_text(hypothesis).split()
        total_words += len(ref_tokens)
        total_errors += edit_distance(ref_tokens, hyp_tokens)
    return float(total_errors) / max(1, total_words)


def compute_cer(reference_texts: Sequence[str], hypothesis_texts: Sequence[str]) -> float:
    """Compute corpus-level character error rate."""
    total_chars = 0
    total_errors = 0
    for reference, hypothesis in zip(reference_texts, hypothesis_texts, strict=True):
        ref_chars = list(normalize_text(reference).replace(" ", ""))
        hyp_chars = list(normalize_text(hypothesis).replace(" ", ""))
        total_chars += len(ref_chars)
        total_errors += edit_distance(ref_chars, hyp_chars)
    return float(total_errors) / max(1, total_chars)
