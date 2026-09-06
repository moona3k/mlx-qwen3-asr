"""Text normalization and error-rate metrics for local eval scripts."""

from __future__ import annotations

import re
from typing import Sequence

_NON_ALNUM_RE = re.compile(r"[^a-z0-9' ]+")
_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize text for stable ASR metric computation."""
    lowered = text.lower()
    cleaned = _NON_ALNUM_RE.sub(" ", lowered)
    return _SPACE_RE.sub(" ", cleaned).strip()


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
