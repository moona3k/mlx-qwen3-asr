"""Forced alignment wrapper for word-level timestamps.

Native MLX forced aligner backend only.
"""

from __future__ import annotations

import unicodedata
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import numpy as np

from .audio import compute_features

DEFAULT_FORCED_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
ALIGNER_BACKEND_MLX = "mlx"


@dataclass(frozen=True)
class AlignedWord:
    """Word-level alignment item."""

    text: str
    start_time: float
    end_time: float


class ForcedAlignTextProcessor:
    """Text processor utilities for timestamp alignment input/output."""

    _nagisa: Any | None = None
    _ko_tokenizer: Any | None = None
    _ko_tokenizer_error: Exception | None = None

    @staticmethod
    def is_kept_char(ch: str) -> bool:
        if ch == "'":
            return True
        cat = unicodedata.category(ch)
        return cat.startswith("L") or cat.startswith("N")

    @classmethod
    def clean_token(cls, token: str) -> str:
        return "".join(ch for ch in token if cls.is_kept_char(ch))

    @staticmethod
    def is_cjk_char(ch: str) -> bool:
        code = ord(ch)
        return (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0x20000 <= code <= 0x2A6DF
            or 0x2A700 <= code <= 0x2B73F
            or 0x2B740 <= code <= 0x2B81F
            or 0x2B820 <= code <= 0x2CEAF
            or 0xF900 <= code <= 0xFAFF
        )

    @classmethod
    def split_segment_with_cjk(cls, seg: str) -> list[str]:
        tokens: list[str] = []
        buf: list[str] = []

        def flush() -> None:
            nonlocal buf
            if buf:
                tokens.append("".join(buf))
                buf = []

        for ch in seg:
            if cls.is_cjk_char(ch):
                flush()
                tokens.append(ch)
            else:
                buf.append(ch)

        flush()
        return tokens

    @classmethod
    def tokenize_space_lang(cls, text: str) -> list[str]:
        tokens: list[str] = []
        for seg in text.split():
            cleaned = cls.clean_token(seg)
            if cleaned:
                tokens.extend(cls.split_segment_with_cjk(cleaned))
        return tokens

    @classmethod
    def _tokenize_japanese(cls, text: str) -> list[str]:
        if cls._nagisa is None:
            try:
                import nagisa  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "Japanese tokenization requires optional dependency `nagisa`. "
                    "Install with: pip install \"mlx-qwen3-asr[aligner]\""
                ) from e
            cls._nagisa = nagisa

        words = cls._nagisa.tagging(text).words
        out: list[str] = []
        for w in words:
            cleaned = cls.clean_token(str(w))
            if cleaned:
                out.append(cleaned)
        return out

    @classmethod
    def _get_ko_tokenizer(cls) -> Any | None:
        if cls._ko_tokenizer is not None:
            return cls._ko_tokenizer
        if cls._ko_tokenizer_error is not None:
            raise RuntimeError(
                "Korean tokenizer backend failed to initialize."
            ) from cls._ko_tokenizer_error

        try:
            from soynlp.tokenizer import LTokenizer  # type: ignore
        except Exception as e:
            cls._ko_tokenizer_error = e
            raise RuntimeError(
                "Korean tokenization requires optional dependency `soynlp`. "
                "Install with: pip install \"mlx-qwen3-asr[aligner]\""
            ) from e

        scores: dict[str, float] = {}
        dict_path = Path(__file__).parent / "assets" / "korean_dict_jieba.dict"
        try:
            with dict_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = line.strip()
                    if not row:
                        continue
                    word = row.split()[0]
                    scores[word] = 1.0
        except OSError as e:
            cls._ko_tokenizer_error = e
            raise RuntimeError(
                "Missing Korean tokenizer dictionary asset "
                f"at '{dict_path}'. Reinstall mlx-qwen3-asr."
            ) from e

        try:
            cls._ko_tokenizer = LTokenizer(scores=scores)
        except Exception as e:
            cls._ko_tokenizer_error = e
            raise RuntimeError("Failed to initialize Korean tokenizer backend.") from e
        return cls._ko_tokenizer

    @classmethod
    def _tokenize_korean(cls, text: str) -> list[str]:
        tok = cls._get_ko_tokenizer()

        out: list[str] = []
        for w in tok.tokenize(text):
            cleaned = cls.clean_token(str(w))
            if cleaned:
                out.append(cleaned)
        return out

    @classmethod
    def tokenize_text(cls, text: str, language: str) -> list[str]:
        """Tokenize alignment text into word/character units.

        Matches official processing behavior for:
        - Japanese: `nagisa` tokenization
        - Korean: `soynlp` LTokenizer + official dict
        - Others: space/CJK tokenizer
        """
        lang = (language or "").strip().lower()
        if lang in {"japanese", "ja", "jp"}:
            return cls._tokenize_japanese(text)
        if lang in {"korean", "ko", "kr"}:
            return cls._tokenize_korean(text)
        return cls.tokenize_space_lang(text)

    @staticmethod
    def _lis_non_decreasing_indices(arr: list[float]) -> list[int]:
        """Return indices of one longest non-decreasing subsequence.

        Matches the legacy O(n^2) DP tie semantics exactly:
        - predecessor is the earliest index that yields the best length
        - end index is the earliest index that reaches global max length

        Uses coordinate compression + Fenwick tree.
        Time: O(n log n), Space: O(n).
        """
        n = len(arr)
        if n == 0:
            return []

        values = sorted(set(arr))
        m = len(values)

        # Fenwick tree nodes store (best_len, earliest_idx_for_that_len)
        bit: list[tuple[int, int]] = [(0, -1)] * (m + 1)
        parent = [-1] * n
        dp = [1] * n

        def _better(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
            """Compare LIS candidates by (max length, earliest index)."""
            if a[0] != b[0]:
                return a if a[0] > b[0] else b
            a_idx = a[1] if a[1] != -1 else n + 1
            b_idx = b[1] if b[1] != -1 else n + 1
            return a if a_idx <= b_idx else b

        def _query(pos: int) -> tuple[int, int]:
            best = (0, -1)
            while pos > 0:
                best = _better(best, bit[pos])
                pos -= pos & -pos
            return best

        def _update(pos: int, candidate: tuple[int, int]) -> None:
            while pos <= m:
                bit[pos] = _better(bit[pos], candidate)
                pos += pos & -pos

        max_len = 0
        end_idx = 0
        for i, x in enumerate(arr):
            rank = bisect_right(values, x)
            best_len, best_idx = _query(rank)
            dp[i] = best_len + 1
            parent[i] = best_idx
            _update(rank, (dp[i], i))
            if dp[i] > max_len:
                max_len = dp[i]
                end_idx = i

        out: list[int] = []
        cur = end_idx
        while cur != -1:
            out.append(cur)
            cur = parent[cur]
        out.reverse()
        return out

    @staticmethod
    def fix_timestamp(data: np.ndarray) -> list[int]:
        """Repair non-monotonic timestamp sequence via LIS-based correction."""
        arr = data.tolist()
        n = len(arr)
        if n == 0:
            return []
        lis_indices = ForcedAlignTextProcessor._lis_non_decreasing_indices(arr)

        is_normal = [False] * n
        for i in lis_indices:
            is_normal[i] = True

        out = arr.copy()
        i = 0
        while i < n:
            if is_normal[i]:
                i += 1
                continue

            j = i
            while j < n and not is_normal[j]:
                j += 1

            anomaly_count = j - i
            left_val = next((out[k] for k in range(i - 1, -1, -1) if is_normal[k]), None)
            right_val = next((out[k] for k in range(j, n) if is_normal[k]), None)

            if anomaly_count <= 2:
                for k in range(i, j):
                    if left_val is None:
                        out[k] = right_val
                    elif right_val is None:
                        out[k] = left_val
                    else:
                        out[k] = left_val if (k - (i - 1)) <= (j - k) else right_val
            else:
                if left_val is not None and right_val is not None:
                    step = (right_val - left_val) / (anomaly_count + 1)
                    for k in range(i, j):
                        out[k] = left_val + step * (k - i + 1)
                elif left_val is not None:
                    for k in range(i, j):
                        out[k] = left_val
                elif right_val is not None:
                    for k in range(i, j):
                        out[k] = right_val

            i = j

        return [int(v) for v in out]

    @classmethod
    def parse_timestamp_ms(cls, words: list[str], timestamp_ms: np.ndarray) -> list[AlignedWord]:
        fixed = cls.fix_timestamp(np.asarray(timestamp_ms))
        out: list[AlignedWord] = []
        for i, word in enumerate(words):
            start_i = i * 2
            end_i = start_i + 1
            if end_i >= len(fixed):
                break
            out.append(
                AlignedWord(
                    text=word,
                    start_time=round(float(fixed[start_i]) / 1000.0, 3),
                    end_time=round(float(fixed[end_i]) / 1000.0, 3),
                )
            )
        return out


class _MLXForcedAlignerBackend:
    """Native MLX forced aligner backend."""

    def __init__(self, model_path: str, dtype: mx.Dtype):
        from .load_models import _ModelHolder
        from .tokenizer import _TokenizerHolder

        self.dtype = dtype
        self.model, self.config = _ModelHolder.get(model_path, dtype=dtype)
        resolved_model_path = _ModelHolder.get_resolved_path(model_path, dtype=dtype)
        self.tokenizer = _TokenizerHolder.get(resolved_model_path)

        self.timestamp_token_id = self.config.timestamp_token_id
        self.timestamp_segment_time = self.config.timestamp_segment_time
        if self.config.classify_num is None:
            raise RuntimeError(
                "Model config missing classify_num required for forced alignment."
            )
        if self.timestamp_token_id is None or self.timestamp_segment_time is None:
            raise RuntimeError(
                "Model config missing timestamp_token_id/timestamp_segment_time."
            )

    @staticmethod
    def _build_prompt(words: list[str], n_audio_tokens: int) -> str:
        body = "<timestamp><timestamp>".join(words) + "<timestamp><timestamp>"
        return (
            "<|audio_start|>"
            + ("<|audio_pad|>" * n_audio_tokens)
            + "<|audio_end|>"
            + body
        )

    def align(self, audio: np.ndarray, text: str, language: str) -> list[AlignedWord]:
        if text.strip() == "":
            return []

        words = ForcedAlignTextProcessor.tokenize_text(text, language)
        if not words:
            return []

        mel, feature_lens = compute_features(audio.astype(np.float32))
        audio_features, _ = self.model.audio_tower(mel.astype(self.dtype), feature_lens)
        n_audio_tokens = int(audio_features.shape[1])

        prompt = self._build_prompt(words, n_audio_tokens)
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = mx.array([token_ids])

        seq_len = input_ids.shape[1]
        positions = mx.arange(seq_len)[None, :]
        position_ids = mx.stack([positions, positions, positions], axis=1)

        embeds = self.model.model.embed_tokens(input_ids)
        audio_mask = input_ids == self.model.audio_token_id
        embeds = self.model._inject_audio_features(
            embeds, audio_features.astype(embeds.dtype), audio_mask
        )
        hidden = self.model.model(
            inputs_embeds=embeds,
            position_ids=position_ids,
            attention_mask=None,
            cache=None,
        )
        logits = self.model.lm_head(hidden)

        pred_ids = np.array(mx.argmax(logits, axis=-1))[0]
        input_np = np.array(input_ids)[0]
        timestamp_ids = pred_ids[input_np == int(self.timestamp_token_id)]
        timestamp_ms = timestamp_ids.astype(np.float32) * float(self.timestamp_segment_time)
        return ForcedAlignTextProcessor.parse_timestamp_ms(words, timestamp_ms)


class ForcedAligner:
    """Word-level forced aligner.

    Args:
        model_path: HF repo ID or local path for the forced aligner model.
        dtype: MLX dtype for native backend paths.
        backend: Must be `mlx`.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_FORCED_ALIGNER_MODEL,
        dtype: mx.Dtype = mx.float16,
        backend: str = ALIGNER_BACKEND_MLX,
    ):
        self.model_path = model_path
        self.dtype = dtype
        self.backend = backend
        self._backend: Optional[Any] = None

    def _ensure_loaded(self) -> None:
        if self._backend is not None:
            return

        if self.backend != ALIGNER_BACKEND_MLX:
            raise RuntimeError(
                f"Unsupported aligner backend '{self.backend}'. "
                f"Expected: {ALIGNER_BACKEND_MLX}."
            )
        try:
            self._backend = _MLXForcedAlignerBackend(self.model_path, self.dtype)
            return
        except Exception as e:  # pragma: no cover - exercised via runtime integration
            raise RuntimeError(f"Failed to load MLX aligner backend: {e}") from e

    def align(
        self,
        audio: np.ndarray,
        text: str,
        language: str,
    ) -> list[AlignedWord]:
        """Align a transcript against audio and return word-level timestamps."""
        self._ensure_loaded()
        return self._backend.align(audio, text, language)  # type: ignore[union-attr]
