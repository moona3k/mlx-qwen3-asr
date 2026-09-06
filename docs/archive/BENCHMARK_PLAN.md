# Benchmark Plan: WER Validation Against Paper Reference Numbers

## Objective

Validate that mlx-qwen3-asr produces quality equivalent to the official Qwen3-ASR
PyTorch implementation by comparing WER/CER on standard benchmarks against the
numbers published in the [Qwen3-ASR Technical Report (arXiv:2601.21337)](https://arxiv.org/abs/2601.21337)
and the [official GitHub repository](https://github.com/QwenLM/Qwen3-ASR).

No PyTorch reference run is needed — the paper **is** the reference.

---

## Current Validation Status

| What | Sample count | Result |
|------|-------------|--------|
| LibriSpeech test-clean (100 speaker-balanced) | 100 | WER 2.29% (fp16) |
| Reference parity English (token-exact vs PyTorch) | 22 | 73% token, 82% text |
| Reference parity multilingual (10 langs x 10) | 100 | 64% token, 67% text |
| Noise variants (SNR 10dB, clean+other) | 10 | 100%/80% text parity |
| Long-form audio WER | 0 | **not tested** |
| Non-English WER (standalone) | 0 | **not tested** |
| 1.7B model WER | 0 | **not tested** |

---

## Target Benchmarks

### Tier 1 — Must Have (Release Gate)

#### 1. LibriSpeech (English)

The universal ASR benchmark. Every paper reports this.

| Detail | Value |
|--------|-------|
| Dataset | [LibriSpeech ASR corpus](https://www.openslr.org/12) |
| HuggingFace | [`openslr/librispeech_asr`](https://huggingface.co/datasets/openslr/librispeech_asr) |
| Splits | `test-clean` (2,620 utterances, ~5.4 hrs), `test-other` (2,939 utterances, ~5.1 hrs) |
| Download | test-clean.tar.gz (346 MB), test-other.tar.gz (328 MB) |
| Metric | WER (word error rate), lower is better |
| Normalization | Lowercase, strip punctuation (standard LibriSpeech eval) |
| Existing infra | `scripts/eval_librispeech.py` — already supports full split eval |
| Run time estimate | ~1-2 hours per split on M4 Pro (0.6B fp16) |

**Paper reference numbers (WER %):**

| Split | Whisper-v3 | Qwen3-ASR-0.6B | Qwen3-ASR-1.7B | GPT-4o Transcribe |
|-------|-----------|----------------|----------------|-------------------|
| test-clean | 1.51 | 2.11 | 1.63 | 1.39 |
| test-other | 3.97 | 4.55 | 3.38 | 3.75 |

> Source: [Qwen3-ASR Technical Report, Table 3](https://arxiv.org/html/2601.21337v1)

**Pass criteria:** MLX WER within +0.3% absolute of the paper's 0.6B number.
- test-clean: WER <= 2.41%
- test-other: WER <= 4.85%

**Command:**
```bash
python scripts/eval_librispeech.py \
  --model Qwen/Qwen3-ASR-0.6B \
  --subset test-clean \
  --samples 2620 \
  --sampling sequential \
  --json-output docs/benchmarks/librispeech-test-clean-full.json

python scripts/eval_librispeech.py \
  --model Qwen/Qwen3-ASR-0.6B \
  --subset test-other \
  --samples 2939 \
  --sampling sequential \
  --json-output docs/benchmarks/librispeech-test-other-full.json
```

---

### Tier 2 — Should Have (Credibility)

#### 2. AISHELL-1 (Mandarin Chinese)

Standard Mandarin benchmark. Qwen3 is a Chinese-origin model — Chinese quality matters.

| Detail | Value |
|--------|-------|
| Dataset | [AISHELL-1](https://www.openslr.org/33/) |
| HuggingFace | [`AISHELL/AISHELL-1`](https://huggingface.co/datasets/AISHELL/AISHELL-1) |
| Test set | 7,176 utterances from 20 speakers |
| Download | ~15 GB (full dataset) |
| Metric | CER (character error rate — standard for Chinese ASR) |
| License | Free for academic use |

**Note:** The paper reports on AISHELL-**2** (not AISHELL-1). AISHELL-2 has restricted
access (requires institutional agreement). AISHELL-1 is freely available and widely
reported by other models, so we can cross-reference against community benchmarks
instead of the paper directly.

| Model | AISHELL-2 test (paper) |
|-------|----------------------|
| Whisper-v3 | 5.06 CER |
| Qwen3-ASR-0.6B | 3.15 CER |
| Qwen3-ASR-1.7B | 2.71 CER |

> Source: [Qwen3-ASR Technical Report, Table 3](https://arxiv.org/html/2601.21337v1)

**Pass criteria:** CER comparable to community AISHELL-1 reports for Qwen3-class models.

**Infra needed:** New eval script (modeled on `eval_librispeech.py`) that:
- Loads AISHELL-1 test split from HuggingFace
- Sets `language="Chinese"`
- Computes CER instead of WER

---

#### 3. FLEURS Multilingual (incl. Korean)

Validates multilingual claims. FLEURS is the benchmark used in the paper.

| Detail | Value |
|--------|-------|
| Dataset | [FLEURS](https://huggingface.co/datasets/google/fleurs) |
| HuggingFace | [`google/fleurs`](https://huggingface.co/datasets/google/fleurs) |
| Test split | ~350 sentences per language (up to 3 recordings each) |
| Total languages | 102 (we test a subset) |
| Metric | WER (most languages) or CER (CJK) |
| License | CC-BY 4.0 |

**Language selection (12 — matches paper's base FLEURS group):**

| Config | Language | Paper 0.6B WER | Paper 1.7B WER | Notes |
|--------|----------|---------------|---------------|-------|
| `en_us` | English | 4.39 | 3.35 | Baseline |
| `cmn_hans_cn` | Chinese | 2.88 | 2.41 | CER metric |
| `ko_kr` | Korean | — | — | User priority; in paper's 12-lang group |
| `ja_jp` | Japanese | — | — | CER metric; exercises nagisa tokenizer |
| `de_de` | German | — | — | European language check |
| `fr_fr` | French | — | — | European language check |
| `es_419` | Spanish | — | — | High-resource language |
| `ar_eg` | Arabic | — | — | RTL script |
| `ru_ru` | Russian | — | — | Cyrillic script |
| `hi_in` | Hindi | — | — | Indic script |

> Source: [Qwen3-ASR Technical Report, Table 3 and Table 5](https://arxiv.org/html/2601.21337v1).
> FLEURS individual per-language WER is reported for en and zh in Table 3.
> Aggregate 12-language average: Qwen3-0.6B = 7.57, Qwen3-1.7B = 4.90 ([Table 5](https://arxiv.org/html/2601.21337v1)).

**Paper aggregate reference (FLEURS 12 languages avg WER):**

| Model | Avg WER |
|-------|---------|
| Whisper-v3 | 5.27 |
| Qwen3-ASR-0.6B | 7.57 |
| Qwen3-ASR-1.7B | 4.90 |

**Pass criteria:** Per-language WER in reasonable range; aggregate within +1.0% of paper.

**Existing infra:** `scripts/build_multilingual_manifest.py` + `scripts/eval_reference_parity_suite.py`
already support FLEURS with any language config. Current runs measure token parity
(vs PyTorch reference), not standalone WER. Need to add WER-against-ground-truth mode.

**Current multilingual parity status (10 samples/lang, 100 total):**

| Language | Token match | Text match |
|----------|------------|------------|
| English | 80% | 90% |
| Korean | **90%** | **90%** |
| Russian | 90% | 90% |
| Chinese | 70% | 80% |
| Spanish | 70% | 70% |
| German | 60% | 60% |
| Japanese | 60% | 60% |
| Arabic | 40% | 50% |
| French | 40% | 40% |
| Hindi | 40% | 40% |

Korean parity is strong (90%). The gap languages (French, Hindi, Arabic) are
likely minor lexical/numeric surface form differences, not transcription failures.
WER against ground truth will confirm this.

---

### Tier 3 — Nice to Have (Differentiation)

#### 4. Earnings-22 (Long-form English)

Fills the long-audio validation gap. Real earnings calls, 5-20+ minutes each.

| Detail | Value |
|--------|-------|
| Dataset | [Earnings-22](https://huggingface.co/datasets/distil-whisper/earnings22) |
| HuggingFace | [`distil-whisper/earnings22`](https://huggingface.co/datasets/distil-whisper/earnings22) |
| Size | 125 files, ~119 hours total |
| Audio lengths | 5-45+ minutes per file |
| Metric | WER |
| License | Free for research |

**Why this matters:**
- Tests the chunking logic (energy-based splitting at boundaries)
- Tests encoder windowing at scale (segmented attention path)
- Tests chunk result merging (concatenation of multi-chunk transcriptions)
- Real-world noisy conditions (phone lines, accents, cross-talk)
- This is the #1 gap identified in the current validation

**Note:** The Qwen3-ASR paper does NOT report on Earnings-22 directly, but the
[HuggingFace Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
provides community reference numbers for many models on this dataset.

**Infra needed:** New eval script that handles long-form audio (>5 min per file).

---

#### 5. GigaSpeech (Diverse English)

Diverse real-world English (YouTube, podcasts, audiobooks).

| Detail | Value |
|--------|-------|
| Dataset | [GigaSpeech](https://huggingface.co/datasets/speechcolab/gigaspeech) |
| Test set | ~40 hours |
| Metric | WER |
| License | Apache 2.0 |

**Paper reference (WER %):**

| Model | GigaSpeech |
|-------|-----------|
| Whisper-v3 | 9.76 |
| Qwen3-ASR-0.6B | 8.88 |
| Qwen3-ASR-1.7B | 8.45 |
| GPT-4o | 25.50 |

> Source: [Qwen3-ASR Technical Report, Table 3](https://arxiv.org/html/2601.21337v1)

---

#### 6. Common Voice Korean (Korean-specific depth)

Adds Korean depth beyond FLEURS's ~350 sentences.

| Detail | Value |
|--------|-------|
| Dataset | [Common Voice](https://commonvoice.mozilla.org/en/datasets) |
| HuggingFace | [`mozilla-foundation/common_voice_17_0`](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) |
| Korean test set | Varies by release (typically hundreds to low thousands of clips) |
| Metric | CER (Korean) |
| License | CC-0 (public domain) |
| Access | Requires free Mozilla account to download |

**Paper reference (Common Voice 13 langs avg WER):**

| Model | CV avg |
|-------|--------|
| Whisper-v3 | 10.77 |
| Qwen3-ASR-0.6B | 12.75 |
| Qwen3-ASR-1.7B | 9.18 |

> Source: [Qwen3-ASR Technical Report, Table 5](https://arxiv.org/html/2601.21337v1)

---

## Forced Aligner Validation

The forced aligner has its own benchmark: Accumulated Average Shift (AAS) in milliseconds.

**Paper reference (AAS in ms, lower is better):**

| Language | NFA | WhisperX | Qwen3-FA |
|----------|-----|----------|----------|
| Chinese | 109.8 | — | 33.1 |
| English | 107.5 | 92.1 | 37.5 |
| French | 100.7 | 145.3 | 41.7 |
| German | 122.7 | 165.1 | 46.5 |
| Japanese | — | — | 42.4 |
| **Korean** | **—** | **—** | **37.2** |
| Russian | 200.7 | — | 40.2 |
| Spanish | 124.7 | 108.0 | 36.8 |
| **Average** | **129.8** | **133.2** | **42.9** |

> Source: [Qwen3-ASR Technical Report, Table 9](https://arxiv.org/html/2601.21337v1)

**Current status:** Native MLX aligner validated on 50 LibriSpeech samples with
5.7ms timing MAE vs PyTorch reference. Full language-specific AAS evaluation
not yet done.

---

## Noise & Robustness

The paper includes internal robustness benchmarks (Table 4) not reproducible
with public datasets, but noteworthy as targets:

| Condition | Qwen3-0.6B WER | Qwen3-1.7B WER |
|-----------|---------------|----------------|
| Dialog-Accented English | 16.62 | 16.07 |
| ExtremeNoise | 17.88 | 16.17 |
| TongueTwister | 4.06 | 2.44 |
| Elders & Kids (Mandarin) | 4.48 | 3.81 |

> Source: [Qwen3-ASR Technical Report, Table 4](https://arxiv.org/html/2601.21337v1)

These use proprietary internal datasets. For our validation, noise robustness is
partially covered by:
- Reference parity noise variants (SNR 10dB on LibriSpeech — 10 samples, parity verified)
- Earnings-22 naturally contains noisy real-world audio (phone lines, cross-talk)

---

## Streaming Validation

The paper reports streaming vs offline WER degradation:

| Model | Mode | LS-clean | LS-other | FLEURS-en | FLEURS-zh |
|-------|------|----------|----------|-----------|-----------|
| Qwen3-0.6B | Offline | 2.11 | 4.55 | 4.39 | 2.88 |
| Qwen3-0.6B | Streaming | 2.54 | 6.27 | 5.38 | 3.40 |

> Source: [Qwen3-ASR Technical Report, Table 8](https://arxiv.org/html/2601.21337v1)

**Expected degradation:** ~0.4-1.7% absolute WER increase in streaming mode.
Current streaming implementation is validated for stability but not WER-evaluated.

---

## Methodology

### WER Computation

Use [`jiwer`](https://github.com/jitsi/jiwer) (Python, backed by RapidFuzz C++):

```python
import jiwer

wer = jiwer.wer(reference, hypothesis)
cer = jiwer.cer(reference, hypothesis)
```

Standard normalization: lowercase, remove punctuation, collapse whitespace.
The project already has `scripts/eval/metrics.py` (`normalize_text`, `compute_wer`,
and `compute_cer`) — use these for consistency with existing benchmarks.

### Text Normalization Notes

- **English:** Lowercase, strip punctuation. Standard.
- **Chinese/Japanese:** CER (character-level), no word segmentation needed.
- **Korean:** CER is common; WER requires morphological segmentation.
  Use CER for consistency with Chinese/Japanese.
- **Other languages:** WER with language-appropriate normalization.

### Run Configuration

- **Decoding:** Greedy (temperature=0). Deterministic — one run is sufficient.
- **Model:** `Qwen/Qwen3-ASR-0.6B` (primary), `Qwen/Qwen3-ASR-1.7B` (stretch).
- **dtype:** `float16` (default operating point).
- **max_new_tokens:** 256 (short clips), 1024 (long-form).

---

## Implementation Priority

### Phase 1: Full LibriSpeech (highest value, infra exists)

1. Run `eval_librispeech.py` on full test-clean (2,620 samples)
2. Run `eval_librispeech.py` on full test-other (2,939 samples)
3. Document results, compare against paper's 2.11% / 4.55%
4. Estimated time: ~2-3 hours compute

### Phase 2: FLEURS WER (multilingual credibility)

1. Extend `eval_reference_parity_suite.py` or write new script to compute
   WER against FLEURS ground truth (not just token parity vs PyTorch)
2. Run on 10 languages, all ~350 test sentences per language
3. Include Korean (`ko_kr`) as a priority language
4. Compare per-language WER and aggregate against paper's numbers
5. Estimated time: ~4-6 hours compute

### Phase 3: AISHELL-1 (Chinese depth)

1. Write `eval_aishell.py` (modeled on `eval_librispeech.py`)
2. Run on full test set (7,176 utterances)
3. Compare CER against community benchmarks
4. Estimated time: ~3-4 hours compute

### Phase 4: Earnings-22 (long-form validation)

1. Write `eval_earnings22.py` for long-form audio handling
2. Start with 10-20 files (diverse lengths: 5min, 10min, 20min, 30min+)
3. Measure WER, also measure per-chunk behavior and merge quality
4. Compare against HF Open ASR Leaderboard community numbers
5. Estimated time: ~6-10 hours compute (long files)

### Phase 5: Stretch goals

- Common Voice Korean (deeper Korean coverage)
- GigaSpeech (diverse English)
- Streaming WER evaluation
- 1.7B model runs on all above

---

## Artifact Convention

All benchmark results follow the existing pattern:

```
docs/benchmarks/YYYY-MM-DD-{benchmark}-{variant}.json   # Raw data
docs/benchmarks/YYYY-MM-DD-{benchmark}-{variant}.md     # Human-readable summary
```

Example:
```
docs/benchmarks/2026-02-15-librispeech-test-clean-full.json
docs/benchmarks/2026-02-15-librispeech-test-clean-full.md
docs/benchmarks/2026-02-15-fleurs-wer-10lang.json
docs/benchmarks/2026-02-15-fleurs-wer-10lang.md
```

---

## Sources

- [Qwen3-ASR Technical Report (arXiv:2601.21337)](https://arxiv.org/abs/2601.21337)
- [Qwen3-ASR GitHub](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen3-ASR-0.6B on HuggingFace](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Qwen3-ASR-1.7B on HuggingFace](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [LibriSpeech ASR corpus](https://www.openslr.org/12)
- [AISHELL-1](https://www.openslr.org/33/)
- [FLEURS dataset](https://huggingface.co/datasets/google/fleurs)
- [Earnings-22](https://huggingface.co/datasets/distil-whisper/earnings22)
- [GigaSpeech](https://huggingface.co/datasets/speechcolab/gigaspeech)
- [Common Voice](https://commonvoice.mozilla.org/en/datasets)
- [jiwer (WER computation)](https://github.com/jitsi/jiwer)
- [HuggingFace Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
- [Qwen team announcement (X/Twitter)](https://x.com/Alibaba_Qwen/status/1965068737297707261)
