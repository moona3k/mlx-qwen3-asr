"""Tokenizer wrapper and prompt building for Qwen3-ASR."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import regex as re

from .cache_utils import LRUCache

_LANGUAGE_CANONICAL: dict[str, str] = {
    # English
    "en": "English",
    "en-us": "English",
    "en-gb": "English",
    "english": "English",
    # Chinese
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "cmn": "Chinese",
    "mandarin": "Chinese",
    "chinese": "Chinese",
    # Japanese
    "ja": "Japanese",
    "ja-jp": "Japanese",
    "japanese": "Japanese",
    # Korean
    "ko": "Korean",
    "ko-kr": "Korean",
    "korean": "Korean",
    # German
    "de": "German",
    "de-de": "German",
    "german": "German",
    # French
    "fr": "French",
    "fr-fr": "French",
    "french": "French",
    # Spanish
    "es": "Spanish",
    "es-es": "Spanish",
    "es-419": "Spanish",
    "spanish": "Spanish",
    # Russian
    "ru": "Russian",
    "ru-ru": "Russian",
    "russian": "Russian",
    # Arabic
    "ar": "Arabic",
    "ar-eg": "Arabic",
    "arabic": "Arabic",
    # Hindi
    "hi": "Hindi",
    "hi-in": "Hindi",
    "hindi": "Hindi",
    # Common additional languages
    "it": "Italian",
    "it-it": "Italian",
    "italian": "Italian",
    "pt": "Portuguese",
    "pt-br": "Portuguese",
    "pt-pt": "Portuguese",
    "portuguese": "Portuguese",
    "tr": "Turkish",
    "tr-tr": "Turkish",
    "turkish": "Turkish",
    "nl": "Dutch",
    "nl-nl": "Dutch",
    "dutch": "Dutch",
}


_KNOWN_LANGUAGE_NAMES = tuple(sorted(set(_LANGUAGE_CANONICAL.values())))

PRETOKENIZE_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def canonicalize_language(language: Optional[str]) -> Optional[str]:
    """Normalize common language aliases/codes to canonical prompt names."""
    if language is None:
        return None
    value = str(language).strip()
    if not value:
        return None
    key = value.casefold().replace("_", "-")
    return _LANGUAGE_CANONICAL.get(key, value)


def language_is_known(language: Optional[str]) -> bool:
    """Return True when a language maps through known aliases/codes."""
    if language is None:
        return False
    value = str(language).strip()
    if not value:
        return False
    key = value.casefold().replace("_", "-")
    return key in _LANGUAGE_CANONICAL


def known_language_names() -> tuple[str, ...]:
    """Return known canonical language names used for prompt forcing."""
    return _KNOWN_LANGUAGE_NAMES


def known_language_aliases() -> dict[str, tuple[str, ...]]:
    """Return canonical language names mapped to known aliases/codes."""
    aliases: dict[str, list[str]] = {}
    for alias, canonical in _LANGUAGE_CANONICAL.items():
        aliases.setdefault(canonical, []).append(alias)
    return {
        canonical: tuple(sorted(values))
        for canonical, values in sorted(aliases.items(), key=lambda item: item[0])
    }


def _bytes_to_unicode() -> dict[int, str]:
    """Build GPT2/Qwen byte-to-unicode reversible map."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs), strict=True))


def _get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    prev = word[0]
    for ch in word[1:]:
        pairs.add((prev, ch))
        prev = ch
    return pairs


def _resolve_tokenizer_dir(model_path: str) -> Path:
    path = Path(model_path)
    if path.exists() and (path / "vocab.json").exists() and (path / "merges.txt").exists():
        return path
    # Reuse model resolver so repo IDs map to local snapshot directories.
    from .load_models import _resolve_path

    return _resolve_path(model_path)


class _NativeQwenBPETokenizer:
    """Native byte-level BPE tokenizer compatible with Qwen2 tokenizer files."""

    def __init__(self, model_path: str):
        model_dir = _resolve_tokenizer_dir(model_path)
        self._encoder: dict[str, int] = json.loads((model_dir / "vocab.json").read_text("utf-8"))
        self._decoder: dict[int, str] = {v: k for k, v in self._encoder.items()}
        self._byte_encoder = _bytes_to_unicode()
        self._byte_decoder = {v: k for k, v in self._byte_encoder.items()}
        self._errors = "replace"
        self._pat = re.compile(PRETOKENIZE_REGEX)

        merges = (model_dir / "merges.txt").read_text("utf-8").splitlines()
        self._bpe_ranks: dict[tuple[str, str], int] = {}
        for i, line in enumerate(merges):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            self._bpe_ranks[(parts[0], parts[1])] = i
        self._bpe_cache: dict[str, str] = {}

        tok_cfg_path = model_dir / "tokenizer_config.json"
        tok_cfg = {}
        if tok_cfg_path.exists():
            tok_cfg = json.loads(tok_cfg_path.read_text("utf-8"))
        self._errors = str(tok_cfg.get("errors", "replace"))

        added = tok_cfg.get("added_tokens_decoder", {}) or {}
        self._added_tokens_encoder: dict[str, int] = {}
        self._added_tokens_decoder: dict[int, str] = {}
        self._special_added_token_ids: set[int] = set()
        for id_str, meta in added.items():
            try:
                tid = int(id_str)
            except (TypeError, ValueError):
                continue
            if not isinstance(meta, dict):
                continue
            content = str(meta.get("content", ""))
            if not content:
                continue
            self._added_tokens_encoder[content] = tid
            self._added_tokens_decoder[tid] = content
            if bool(meta.get("special", False)):
                self._special_added_token_ids.add(tid)

        eos_tok = tok_cfg.get("eos_token", "<|endoftext|>")
        unk_tok = tok_cfg.get("unk_token", eos_tok) or eos_tok
        self._unk_token_id = self._added_tokens_encoder.get(
            str(unk_tok),
            self._encoder.get(str(unk_tok), 151643),
        )

        # Longest-first added-token matching to keep special/added markers atomic.
        self._added_tokens = sorted(self._added_tokens_encoder, key=len, reverse=True)
        if self._added_tokens:
            escaped = "|".join(re.escape(t) for t in self._added_tokens)
            self._added_pat = re.compile(escaped)
        else:
            self._added_pat = None

    def _bpe(self, token: str) -> str:
        cached = self._bpe_cache.get(token)
        if cached is not None:
            return cached

        word = tuple(token)
        if not word:
            return ""
        pairs = _get_pairs(word)
        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda p: self._bpe_ranks.get(p, float("inf")))
            if bigram not in self._bpe_ranks:
                break
            first, second = bigram
            new_word: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)

        out = " ".join(word)
        self._bpe_cache[token] = out
        return out

    def _split_with_added_tokens(self, text: str) -> list[tuple[bool, str]]:
        if not text:
            return []
        if self._added_pat is None:
            return [(False, text)]
        parts: list[tuple[bool, str]] = []
        last = 0
        for m in self._added_pat.finditer(text):
            s, e = m.span()
            if s > last:
                parts.append((False, text[last:s]))
            parts.append((True, text[s:e]))
            last = e
        if last < len(text):
            parts.append((False, text[last:]))
        return parts

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text is None:
            return []
        s = str(text)
        ids: list[int] = []

        for is_added, segment in self._split_with_added_tokens(s):
            if not segment:
                continue
            if is_added:
                tid = self._added_tokens_encoder.get(segment)
                if tid is not None:
                    ids.append(tid)
                continue
            for token in re.findall(self._pat, segment):
                encoded = "".join(self._byte_encoder[b] for b in token.encode("utf-8"))
                for bpe_tok in self._bpe(encoded).split(" "):
                    ids.append(self._encoder.get(bpe_tok, self._unk_token_id))

        # Qwen2 tokenizer config for ASR does not auto-wrap BOS/EOS, keep this
        # behavior even when add_special_tokens=True for compatibility.
        _ = add_special_tokens
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        if ids is None:
            return ""

        pieces: list[str] = []
        byte_tokens: list[str] = []

        def flush_bytes() -> None:
            if not byte_tokens:
                return
            text = "".join(byte_tokens)
            data = bytearray(self._byte_decoder[c] for c in text)
            pieces.append(data.decode("utf-8", errors=self._errors))
            byte_tokens.clear()

        for tid in ids:
            idx = int(tid)
            added = self._added_tokens_decoder.get(idx)
            if added is not None:
                if skip_special_tokens and idx in self._special_added_token_ids:
                    continue
                flush_bytes()
                pieces.append(added)
                continue
            tok = self._decoder.get(idx)
            if tok is None:
                continue
            byte_tokens.append(tok)

        flush_bytes()
        return "".join(pieces)

    def get_vocab(self) -> dict[str, int]:
        out = dict(self._encoder)
        out.update(self._added_tokens_encoder)
        return out


class Tokenizer:
    """Thin wrapper around native Qwen-compatible byte-level BPE tokenizer.

    Handles prompt template construction with audio placeholder tokens.
    """

    # Fallback special token IDs (looked up from tokenizer at init)
    IM_START_ID = 151644           # <|im_start|>
    IM_END_ID = 151645             # <|im_end|>
    EOS_TOKEN_IDS = [151643, 151645]

    def __init__(self, model_path: str):
        self._tokenizer = _NativeQwenBPETokenizer(model_path)

        # Look up audio token IDs from the actual tokenizer vocab
        vocab = self._tokenizer.get_vocab()
        self.AUDIO_TOKEN_ID = vocab.get("<|audio_pad|>", 151676)
        self.AUDIO_START_TOKEN_ID = vocab.get("<|audio_start|>", 151669)
        self.AUDIO_END_TOKEN_ID = vocab.get("<|audio_end|>", 151670)
        self._system_prefix_tokens = self.encode("system\n")
        self._user_tokens = self.encode("user\n")
        self._assistant_tokens = self.encode("assistant\n")
        self._newline_id = self.encode("\n")[0]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def build_prompt_tokens(
        self,
        n_audio_tokens: int,
        language: Optional[str] = None,
        context: str = "",
    ) -> list[int]:
        """Build chat-template prompt with audio placeholder tokens.

        Prompt template (matching official Qwen3-ASR chat_template.json):
            <|im_start|>system
            {context}<|im_end|>
            <|im_start|>user
            <|audio_start|><|audio_pad|>...(N times)...<|audio_pad|><|audio_end|><|im_end|>
            <|im_start|>assistant

        If language is specified, the assistant prefix becomes:
            <|im_start|>assistant
            language {lang}<asr_text>

        Args:
            n_audio_tokens: Number of audio feature tokens to placeholder
            language: Optional language to force (e.g., "English", "Chinese")
            context: Domain-specific context string for the system prompt
                (e.g., space-separated terms like "交易 停滞" to bias
                transcription toward domain vocabulary). Empty string by
                default, matching the official Qwen3-ASR implementation.

        Returns:
            List of token IDs forming the complete prompt
        """
        # System message
        tokens = [self.IM_START_ID]
        tokens.extend(self._system_prefix_tokens)
        if context:
            tokens.extend(self.encode(context))
        tokens.append(self.IM_END_ID)
        tokens.append(self._newline_id)

        # User message with audio
        tokens.append(self.IM_START_ID)
        tokens.extend(self._user_tokens)
        tokens.append(self.AUDIO_START_TOKEN_ID)
        tokens.extend([self.AUDIO_TOKEN_ID] * n_audio_tokens)
        tokens.append(self.AUDIO_END_TOKEN_ID)
        tokens.append(self.IM_END_ID)
        tokens.append(self._newline_id)

        # Assistant prefix
        tokens.append(self.IM_START_ID)
        tokens.extend(self._assistant_tokens)

        # Optional language forcing
        if language:
            canon_lang = canonicalize_language(language)
            if canon_lang:
                tokens.extend(self.encode(f"language {canon_lang}<asr_text>"))

        return tokens

    def build_followup_prompt_tokens(
        self,
        n_audio_tokens: int,
        language: Optional[str] = None,
        context: str = "",
    ) -> list[int]:
        """Build a follow-up chat turn for incremental streaming.

        Template:
            <|im_end|>
            <|im_start|>user
            <|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>
            <|im_start|>assistant

        Args:
            n_audio_tokens: Number of audio feature tokens to placeholder.
            language: Optional language to force.
            context: Domain-specific context (currently unused in follow-up
                turns, accepted for API consistency).
        """
        tokens = [self.IM_END_ID, self._newline_id, self.IM_START_ID]
        tokens.extend(self._user_tokens)
        tokens.append(self.AUDIO_START_TOKEN_ID)
        tokens.extend([self.AUDIO_TOKEN_ID] * n_audio_tokens)
        tokens.append(self.AUDIO_END_TOKEN_ID)
        tokens.append(self.IM_END_ID)
        tokens.append(self._newline_id)
        tokens.append(self.IM_START_ID)
        tokens.extend(self._assistant_tokens)

        if language:
            canon_lang = canonicalize_language(language)
            if canon_lang:
                tokens.extend(self.encode(f"language {canon_lang}<asr_text>"))

        return tokens


class _TokenizerHolder:
    """Simple process-local cache for tokenizer instances."""

    _cache = LRUCache[str, Tokenizer](max_entries=8)

    @staticmethod
    def _canonical_key(model_path: str) -> str:
        p = Path(model_path)
        if p.exists():
            return str(p.resolve())
        return model_path

    @classmethod
    def get(cls, model_path: str) -> Tokenizer:
        key = cls._canonical_key(model_path)
        tok = cls._cache.get(key)
        if tok is None:
            tok = Tokenizer(model_path)
            cls._cache.put(key, tok)
        return tok

    @classmethod
    def set_cache_capacity(cls, max_entries: int) -> None:
        """Set tokenizer-holder LRU capacity for this process."""
        cls._cache.set_max_entries(max_entries)

    @classmethod
    def clear(cls) -> None:
        cls._cache.clear()


def _detect_and_fix_repetitions(text: str, threshold: int = 20) -> str:
    """Collapse pathological repetitions in decoder output."""

    def _fix_char_runs(s: str) -> str:
        out: list[str] = []
        i = 0
        n = len(s)
        while i < n:
            j = i + 1
            while j < n and s[j] == s[i]:
                j += 1
            run = j - i
            if run > threshold:
                out.append(s[i])
            else:
                out.append(s[i:j])
            i = j
        return "".join(out)

    def _fix_pattern_runs(s: str, max_pattern_len: int = 20) -> str:
        n = len(s)
        if n < threshold * 2:
            return s
        i = 0
        out: list[str] = []
        while i <= n - threshold * 2:
            found = False
            for k in range(1, max_pattern_len + 1):
                if i + (k * threshold) > n:
                    break
                p = s[i : i + k]
                ok = True
                for rep in range(1, threshold):
                    start = i + rep * k
                    if s[start : start + k] != p:
                        ok = False
                        break
                if not ok:
                    continue
                end = i + threshold * k
                while end + k <= n and s[end : end + k] == p:
                    end += k
                out.append(p)
                out.append(_fix_pattern_runs(s[end:], max_pattern_len=max_pattern_len))
                i = n
                found = True
                break
            if found:
                break
            out.append(s[i])
            i += 1
        if i < n:
            out.append(s[i:])
        return "".join(out)

    return _fix_pattern_runs(_fix_char_runs(text))


def _strip_trailing_special_tokens(s: str) -> str:
    out = s.rstrip()
    trailing_tokens = ("<|im_end|>", "<|endoftext|>")
    changed = True
    while changed:
        changed = False
        for token in trailing_tokens:
            if out.endswith(token):
                out = out[: -len(token)].rstrip()
                changed = True
    return out.strip()


def parse_asr_output(
    text: str,
    user_language: Optional[str] = None,
) -> tuple[str, str]:
    """Parse ASR model output into language and transcription text.

    Input format: "language English<asr_text>hello world"
    Output: ("English", "hello world")

    Args:
        text: Raw model output text
        user_language: Optional forced language. If provided, return this
            language and treat model output as plain transcription text.

    Returns:
        Tuple of (detected_language, transcription_text)
    """
    if text is None:
        return "unknown", ""
    s = str(text).strip()
    if not s:
        return "unknown", ""

    s = _detect_and_fix_repetitions(s)

    if user_language:
        forced = canonicalize_language(user_language) or user_language
        if "<asr_text>" in s:
            transcript = _strip_trailing_special_tokens(s.split("<asr_text>", 1)[1])
            return forced, transcript
        return forced, _strip_trailing_special_tokens(s)

    if "<asr_text>" in s:
        parts = s.split("<asr_text>", 1)
        lang_part = parts[0].strip()
        transcript = _strip_trailing_special_tokens(parts[1])

        # Extract language name from "language English"
        if lang_part.startswith("language "):
            lang = lang_part[len("language "):]
        else:
            lang = lang_part

        if "language none" in lang_part.lower() and not transcript:
            return "", ""

        return canonicalize_language(lang) or lang, transcript

    # Fallback: no asr_text marker
    return "unknown", _strip_trailing_special_tokens(s)
