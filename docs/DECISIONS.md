# Technical Decisions

Key technical decisions made for mlx-qwen3-asr, with rationale.

## Decision 1: Python + MLX

**Choice:** Python with Apple's MLX framework
**Alternatives considered:** Rust, Swift, C++

**Rationale:**
- MLX has no Rust bindings -- the framework is Python and C++ only
- 95% of runtime is spent in Metal/C++ kernels inside MLX -- Python overhead is negligible
- Swift port already exists (qwen3-asr-swift) -- no need to duplicate
- Python ecosystem makes it easy to integrate with HuggingFace, numpy, etc.
- Fastest path to a working implementation

## Decision 2: Standalone Package (not part of mlx-audio)

**Choice:** Independent `mlx-qwen3-asr` package
**Alternative:** Contribute to mlx-audio

**Rationale:**
- mlx-audio has critical bugs for Qwen3-ASR:
  - Uses standard nn.RoPE instead of interleaved MRoPE
  - Issue #459: long-audio truncation
  - Historical config drift risk across revisions
- mlx-audio depends on bleeding-edge packages: `transformers==5.0.0rc3`, `mlx-lm==0.30.5`
- Qwen3-ASR deserves dedicated focus -- it's SOTA and complex enough to warrant its own package
- Standalone allows us to optimize specifically for this model without compromise

## Decision 3: Dual Timestamp Backends with Native-First Default (Superseded by Decision 20)

**Choice:** Use native MLX backend (`mlx`) as default timestamp path while
keeping `qwen_asr` as explicit official-reference option and `auto` fallback mode
**Alternative:** Keep `qwen_asr` as default backend

**Rationale:**
- Native MLX backend removes PyTorch runtime dependency from default timestamp use.
- `qwen_asr` remains the official reference implementation path and stays
  available as an explicit backend for conservative parity checks.
- Dual-backend design lets users choose:
  - native default (`mlx`),
  - official reference (`qwen_asr`),
  - pragmatic fallback (`auto`).
- This keeps native-first UX while preserving explicit reference fallback.

**Policy (important):**
- This is a transition state, not the end-state architecture.
- Project north star remains full native MLX for core + timestamps.
- Keep expanding native parity gates (timing quality + reliability +
  multilingual coverage + performance envelope) until `qwen_asr` is no longer
  needed in standard workflows.

## Decision 4: HuggingFace Tokenizer (Superseded by Decision 18)

**Choice:** Use `transformers.AutoTokenizer` (Qwen2TokenizerFast)
**Alternative:** Reimplement BPE tokenizer from scratch

**Rationale:**
- Qwen2TokenizerFast has 151936 tokens -- reimplementing is high effort, low value
- The tokenizer is well-tested and handles all edge cases
- Accept `transformers` as a dependency (needed for tokenizer only, not model inference)
- Tokenization is not a performance bottleneck

## Decision 5: ffmpeg for Audio Loading

**Choice:** Subprocess call to ffmpeg
**Alternative:** librosa, soundfile, torchaudio

**Rationale:**
- Same approach as mlx-whisper -- proven pattern
- Handles all audio formats (mp3, wav, flac, ogg, m4a, etc.)
- No additional Python dependencies (librosa pulls in many)
- ffmpeg is universally available (`brew install ffmpeg`)
- Consistent behavior across audio formats

## Decision 6: On-the-fly Weight Remapping

**Choice:** Remap HF weights at load time (strip `thinker.` prefix, transpose Conv2d)
**Alternative:** Require pre-converted weights

**Rationale:**
- Users can point directly at HuggingFace repo -- no separate conversion step
- Remapping is fast (< 1 second) compared to model download
- Reduces friction for new users
- Still support pre-converted local weights for advanced users

## Decision 7: Interleaved MRoPE (Custom Implementation)

**Choice:** Custom InterleavedMRoPE class instead of MLX's built-in nn.RoPE
**Alternative:** Use nn.RoPE with workarounds

**Rationale:**
- MLX's nn.RoPE doesn't support 3D interleaved frequency assignment
- The interleaving pattern (stride-3 across sections [24,20,20]) is specific to Qwen3
- Incorrect RoPE produces plausible but degraded transcription -- hard to debug
- This is the #1 bug in existing implementations (mlx-audio gets this wrong)
- Correctness is non-negotiable for the core position encoding

## Decision 8: Default Runtime Model = 0.6B

**Choice:** Default `transcribe()` and CLI to `Qwen/Qwen3-ASR-0.6B`
**Alternative:** Keep 1.7B as the default

**Rationale:**
- "One-line and it just works" is stronger with 0.6B on typical Mac hardware
- Lower memory footprint reduces first-run friction and OOM risk
- Better latency by default improves perceived product quality
- 1.7B remains a first-class opt-in for accuracy-focused workloads

## Decision 9: Native JA/KO Aligner Tokenization Mirrors Official Dependencies

**Choice:** For native MLX aligner path, require `nagisa` (JA) and `soynlp` + official
Korean dict (KO), and fail clearly when missing.
**Alternative:** Silent fallback to generic space/CJK tokenization.

**Rationale:**
- Timestamp quality gates require tokenizer behavior to match the official processor.
- Silent fallback hides quality degradation and weakens parity guarantees.
- Clear runtime errors are preferable to silent low-quality alignment in multilingual use.
- Vendoring the official Korean dictionary asset keeps behavior reproducible across machines.

## Decision 10: Prefer Direct `Qwen2Tokenizer` Loader over `AutoTokenizer` (Superseded by Decision 18)

**Choice:** In tokenizer loading, import and instantiate `Qwen2Tokenizer` directly when
available, with `AutoTokenizer` as fallback.
**Alternative:** Always use `AutoTokenizer.from_pretrained(...)`.

**Rationale:**
- `AutoTokenizer` dynamic import path pulls significantly more module graph at cold start.
- Direct `Qwen2Tokenizer` reduces process-level first-transcribe latency while preserving
  tokenization behavior and parity gates.
- Fallback path retains compatibility with older/variant transformer stacks.

## Decision 11: Resolve Local Model Snapshot Path Before Tokenizer Load

**Choice:** Pass resolved local model path (HF snapshot directory) to tokenizer loading
once the model is already resolved by `_ModelHolder`.
**Alternative:** Keep passing the original repo ID string to tokenizer loading.

**Rationale:**
- Repo-ID tokenizer loading may still perform Hub metadata checks in short-lived processes.
- Resolved local path removes that network overhead when weights/tokenizer files are already cached.
- This complements Decision 10 and further reduces first-transcribe latency without affecting quality.

## Decision 12: Streaming Uses Bounded Rolling Context Until True Incremental Decode Lands

**Choice:** Keep streaming as a rolling decode mode with a fixed max context window
(default 30s), explicitly marked experimental.
**Alternative:** Market current streaming as full incremental realtime ASR.

**Rationale:**
- Rolling context keeps per-chunk runtime bounded instead of growing with total session length.
- Prefix-rollback behavior remains useful for partial-output stability.
- This avoids over-claiming while preserving a usable API for live transcription workflows.
- A true production incremental mode still requires decoder cache lifecycle + chunk-level
  audio encoder state strategy; that remains on the roadmap.

## Decision 13: Add Explicit `Session` API While Keeping One-Liner Convenience

**Choice:** Introduce a first-class `Session` object that owns model/tokenizer state,
while preserving top-level `transcribe(...)` for simple usage.
**Alternative:** Keep only hidden process-global holders.

**Rationale:**
- Explicit state ownership is easier to reason about and test.
- Multiple model sessions can coexist in one process without implicit coupling.
- Keeps power-user workflows deterministic while preserving beginner ergonomics.

## Decision 14: Keep Streaming Experimental and Deprioritize Productionization

**Choice:** Keep streaming clearly labeled experimental and avoid major engineering
investment until core offline quality/speed gates are fully saturated.
**Alternative:** Spend near-term roadmap bandwidth on production-grade streaming.

**Rationale:**
- Qwen3-ASR's strongest practical value today is high-quality multilingual
  post-recording transcription.
- Current upstream streaming semantics are themselves constrained
  (vLLM-only and no timestamps), so production streaming is not a short path.
- Near-term engineering ROI is higher in correctness gates, native aligner quality,
  benchmark rigor, quantized packaging, and decode-path cleanliness.

## Decision 15: Adopt Upstream-Style ASR Output Repetition Cleanup

**Choice:** Apply repetition cleanup + edge-case parsing in `parse_asr_output(...)`,
aligned with official Qwen inference utility behavior.
**Alternative:** Keep minimal parser that only splits on `<asr_text>`.

**Rationale:**
- Repetition collapse directly reduces pathological decode tails in real-world audio.
- Handling `language None` and forced-language parse paths improves robustness.
- This is a low-risk, high-value inference-time quality safeguard.

## Decision 16: Harden Experimental Streaming Input Contracts and Remove Dead State

**Choice:** Validate streaming init params, normalize incoming PCM in `feed_audio(...)`,
and remove unused `previous_tokens` state.
**Alternative:** Keep permissive/unvalidated inputs and retain unused state fields.

**Rationale:**
- Upstream streaming path normalizes int16/shape behavior explicitly; matching that
  reduces avoidable runtime surprises.
- Strong input contracts (`chunk_size_sec/max_context_sec/sample_rate > 0`) fail fast
  with clear errors instead of silent undefined behavior.
- Removing dead state simplifies the module and makes future incremental refactors clearer.

## Decision 17: Keep Speculative Decoding as Opt-In Experimental Path

**Choice:** Implement speculative decoding behind explicit `draft_model` opt-in and
keep baseline greedy decode as default.
**Alternative:** Promote speculative decoding to default generation path immediately.

**Rationale:**
- Prototype achieves strict token-level parity in greedy mode.
- Current benchmark evidence on tested short/10s workloads shows latency regression
  (extra draft compute outweighs target savings).
- Keeping it opt-in preserves a clean default path while enabling continued
  experimentation on acceptance-rate and long-form workloads.

## Decision 18: Replace HF Tokenizer Runtime with Native In-Repo BPE

**Choice:** Implement and ship native byte-level BPE tokenizer (`vocab.json` + `merges.txt` +
`tokenizer_config.json`) in-repo for runtime encode/decode.
**Alternative:** Keep `transformers` tokenizer as runtime dependency.

**Rationale:**
- Removes heavy runtime dependency surface (`transformers`) from core ASR path.
- Reduces install and compatibility risk from upstream tokenizer API changes.
- Keeps tokenizer behavior deterministic and owned inside this repository.
- Maintains compatibility by preserving existing `Tokenizer` public interface and
  enforcing parity via unit/integration tests.

## Decision 19: Remove HF Feature-Extractor Fallback from Runtime `compute_features()`

**Choice:** Make `compute_features()` native-only for runtime inference and support
`do_not_pad`, `max_length`, and `longest` natively.
**Alternative:** Retain HF `WhisperFeatureExtractor` fallback for uncommon padding/sample-rate paths.

**Rationale:**
- Fully removes `transformers` from runtime transcription path.
- Simplifies feature extraction behavior and error handling.
- Keeps predictable performance characteristics and avoids hidden dependency drift.
- Optional research scripts can still evaluate parity vs HF reference implementation.

## Decision 20: Runtime Forced Aligner Backend is Native-Only

**Choice:** Keep runtime `ForcedAligner` backend fixed to native `mlx`;
use `qwen-asr` only in optional parity scripts.
**Alternative:** Continue dual runtime backends (`mlx` + `qwen_asr` + `auto`).

**Rationale:**
- Eliminates runtime PyTorch bridge complexity and cache churn.
- Simplifies CLI/API behavior and dependency expectations for end users.
- Preserves scientific comparability by keeping explicit reference-lane scripts.
- Aligns with project north-star of fully native MLX runtime paths.

## Decision 21: Diarization Is Optional and pyannote-First

**Choice:** Use pyannote.audio 4.x as the runtime diarization backend behind
the optional extra (`mlx-qwen3-asr[diarize]`), defaulting to
`pyannote/speaker-diarization-community-1`, while keeping the core ASR path
native MLX and torch-free.
**Alternatives:** (1) keep/expand the native heuristic diarization path, (2)
remove diarization support entirely.

**Rationale:**
- Delivers materially better diarization quality now.
- Tracks pyannote's current open-source pipeline/API rather than freezing a
  stale 3.x-era dependency stack.
- Concentrates engineering effort on the core ASR competency.
- Keeps default install lean and unchanged for users who do not need diarization.
- Preserves project identity where it matters most: the core ASR inference path.

## Decision 22: Remove Speculative Native Diarization Scaffolding

**Choice:** Delete speculative native diarization scaffolding (unused candidate
embedding backends, conversion scripts, and governance registries) until there
is a clear funded plan to pursue native model-based diarization.
**Alternatives:** keep scaffolding in-tree for a potential future roadmap.

**Rationale:**
- Reduces maintenance burden and cognitive overhead.
- Avoids dead-code drift and false signals about supported runtime paths.
- Keeps code and docs aligned with actual product direction.

## Decision 23: Match Official Qwen3-ASR `context` Prompt Contract

**Choice:** Expose `context` across public transcription and streaming APIs,
and render it as the entire system-message content with default `""`.
**Alternatives:** (1) keep the historical hardcoded helper prompt, (2) add a
local-only prompt feature that diverges from upstream batch/streaming semantics.

**Rationale:**
- The official Qwen3-ASR API exposes `context` for domain-vocabulary biasing in
  both offline and streaming flows.
- Defaulting to empty system content matches the shipped upstream chat template
  and avoids local prompt drift.
- Keeping one prompt contract across Python API, Session API, streaming, and
  CLI reduces surprise and keeps parity reasoning straightforward.

## Decision 24: Built-in HTTP Transcription Server as Optional Feature

**Choice:** Ship `server.py` inside `mlx_qwen3_asr` with FastAPI + uvicorn as
optional `[serve]` dependencies. Activated via `mlx-qwen3-asr serve`.
**Alternatives:** (1) separate package (`mlx-qwen3-asr-server`), (2) no server
at all — library + CLI only.

**Rationale:**
- Apple Silicon Macs are efficient inference machines; users want to turn them
  into transcription endpoints accessible from any device on the network or
  internet.
- In-package keeps it one repo, one install. Optional deps keep the core lean.
- Single-tenant, API-key auth, async job model (POST returns job ID, poll for
  result), sequential FIFO processing, in-memory job store with TTL.
- v1 scope deliberately narrow: file upload only (no URL ingestion — SSRF risk),
  no WebSocket streaming, no job cancellation, no multi-tenancy.
- Server response mirrors `TranscriptionResult` directly — no translation layer.

**Full spec:** `docs/server/` (ADR, API spec, deployment guide).

## Decision 25: Materialize Init-Time MLX Buffers; Serve Inference on One Thread

**Choice:** Every array built in an `__init__` and stored outside
`parameters()` is evaluated immediately (`SinusoidalPositionEmbedding._pe`,
`InterleavedMRoPE._inv_freq`). The HTTP server additionally constructs and runs
the model on one dedicated `ThreadPoolExecutor` worker.
**Alternative:** Server-only fix (load and infer on the same thread) with the
library left as is.

**Rationale:**
- MLX >= 0.31 binds an unevaluated graph to the thread that built it and
  refuses to evaluate it elsewhere (`There is no Stream(gpu, 1) in current
  thread`, issue #16). `mx.eval(model.parameters())` never reaches underscore
  attributes, so the model looked fully loaded but was not thread-portable.
- Fixing it at construction time makes every entry point safe, including
  `transcribe_async` and user threading. The server's single owner thread is
  kept as defence in depth and to keep inference serialized.
- `tests/test_model.py::TestThreadPortability` fails on MLX 0.32 without the
  fix; the repo venv's older MLX hides it, so CI on current MLX is the guard.

## Decision 26: Subtitle Cues by Display Width and Restored Punctuation

**Choice:** `group_subtitle_segments` measures cue width in display cells
(CJK counts double), applies the word cap only to space-delimited languages,
and re-attaches the transcript's punctuation to aligner segments so sentence
and clause boundaries can be used. Restoration happens only inside subtitle
grouping; the `segments` payload is unchanged.
**Alternative:** Content-sniffing the script per segment instead of using the
language label.

**Rationale:**
- The forced aligner emits one segment per CJK character and drops all
  punctuation, so a 10-word cap cut Chinese cues every 10 characters (issue
  #15) and no sentence rule ever fired.
- The transcript is fully punctuated; matching it character by character
  (case- and whitespace-insensitive, bail out on mismatch) is deterministic and
  cheap.
- The language label is already resolved and canonicalized upstream; one
  shared `tokenizer.join_text_parts` decides the delimiter everywhere so the
  three drifting alias sets that caused this and the Korean spacing bug are
  gone.

## Decision 27: Diarization Device Defaults to `auto` with Inference-Time CPU Fallback

**Choice:** `--diarize-device {auto,cpu,mps,cuda}` (PR #17, @ggshr9), default
`auto` (MPS, then CUDA, then CPU). If the pipeline *call* fails on an
accelerator, warn, remember the failed device for that pipeline identity, and
retry once on CPU.
**Alternative:** Default to `cpu` and make acceleration opt-in.

**Rationale:**
- Measured 26x faster diarization on MPS with byte-identical output; a CPU
  default would make nearly every Mac user pay the slow path.
- `Pipeline.to(device)` only moves tensors; MPS operator gaps raise when the
  pipeline runs. Guarding only the move (as the PR first did) could lose the
  transcription after ASR had finished. The retry makes `auto` safe.
- Torch stays behind the `[diarize]` extra; `cpu` never imports it.

## Decision 28: `TranscribeOptions` Is the Single Internal Contract

**Choice:** `transcribe`, `transcribe_batch` and `Session.transcribe` keep
explicitly typed signatures and build one `TranscribeOptions`; internals
receive that object; `transcribe_async`, `transcribe_batch_async` and
`Session.transcribe_async` forward `**kwargs`.
**Alternative:** Keep six explicit 15-keyword signatures plus dict/dataclass
conversion helpers.

**Rationale:**
- The six copies and two helpers that cancelled each other were ~200 lines of
  forwarding and a standing divergence risk (PR #17 hit it twice).
- A test enumerating `TranscribeOptions` fields against the three explicit
  signatures replaces hand-written call-site audits.
- Public keyword arguments are unchanged for every entry point.

## Decision 29: Streaming Re-Decodes the Audio Window with a Text-Prefix Rollback

**Choice:** On every chunk, re-encode the whole accumulated window (bounded by
`max_context_sec`) and decode it with the previously generated text, minus the
last `unfixed_token_num` tokens, forced into the prompt as a prefix; when the
window would overflow, commit its text and start a new window. This is the
official `qwen_asr` streaming recipe.
**Alternative (previous):** Encode each chunk alone and append it to a live
decoder KV cache as a follow-up chat turn (linear cost, no re-encoding).

**Rationale:**
- The model was never trained on follow-up-turn audio. On the maintained
  multilingual-100 lane the incremental design scored 56% primary error vs
  9.5% offline, with duplicated and dropped segments; on the long-form lane
  35-40% vs 10.6% (`docs/benchmarks/2026-09-19-streaming-manifest-*-incremental-kv.json`).
- The re-feed design scores 11.4% / 12.3% on the same lanes at RTF 0.08 /
  0.18, still well under real time on an M4 Pro.
- The prefix rollback makes partial text stable by construction, as upstream
  documents; the trade-off is that the last few tokens are regenerated every
  chunk, which the `rewrite_rate` metric now reports honestly (~0.65).
- Window commit at `max_context_sec` bounds per-chunk cost; a word cut at the
  boundary can be duplicated or dropped once per window. Re-using encoder
  output across chunks is a future optimisation, not a correctness need.

## Decision 30: Streaming Reuses Encoder Output and Decoder KV for Complete Attention Windows

**Choice:** Keep the Decision 29 re-decode, but cache what cannot change. The
encoder's attention windows (`n_window_infer` = 800 mel frames = 8 s = 104
tokens) never attend across each other and the conv stem and position
embeddings are per 100-frame chunk, so the encoder output of every complete
800-frame block is final. The decoder is causal, so the KV entries for the
prompt head and those blocks' audio tokens are final too. Each chunk encodes
only the uncached frames (new complete blocks + partial tail) in one pass,
prefills only the prompt tail plus the text prefix onto a `KVCache.fork()` of
the cached KV, and after generation trims that fork back to the block boundary
to become the new prefix KV. `reuse_window_prefix=False` restores the plain
re-decode for A/B runs.
**Alternative (rejected):** Cache encoder output only. It saves at most the
encoder's 18% share of per-chunk time; a first cut with separate block-encode
and KV-extension calls was 12% *slower* on 10-20 s clips because fixed
kernel-launch cost outweighed the saved compute.

**Rationale:**
- Per 2 s chunk on a 30 s window (M4 Pro, 0.6B fp16): mel 1%, encoder 18%,
  prefill 31%, generation 50%. Generation is the recipe's floor (the rollback
  regenerates ~5 tokens plus the new ones every chunk); only encoder and
  prefill are addressable, and only for the part of the window that repeats.
- Exactness: appending a prefill to a partially filled causal cache is
  bit-identical to one prefill (logits and KV, measured). Encoding whole
  attention windows separately differs from the batched pass by reduction
  order only (max 1.6e-6 on values of scale 1e-2). Mel frames of a complete
  block are bit-identical as audio grows, provided the window's global log-mel
  maximum is unchanged: the mel clamps every bin against it, so the cache is
  keyed on that maximum and rebuilt when a louder chunk raises it (about one
  chunk in eight on the long-form lane). A block counts as complete only when
  one frame follows it, because the last STFT frame reaches 200 samples past
  the block and would be re-padded otherwise.
- Measured on the maintained lanes, same machine, same day
  (`docs/benchmarks/2026-09-19-streaming-manifest-{multilingual100,longform10}-{no-,}prefix-reuse.json`):
  long-form 10 x 75 s RTF 0.104 -> 0.083 (p95 0.136 -> 0.111), primary
  12.33% -> 12.28% fixed; multilingual-100 RTF 0.090 -> 0.081 (p95
  0.157 -> 0.134), primary 11.35% -> 11.32%. Every changed hypothesis (2/20,
  3/200) scored equal or better. Both lanes stay under the strict offline +
  3pp ceiling.
- `KVCache.fork()` shares arrays in concatenating mode (immutable) and copies
  in preallocated mode, because MLX slice assignment is visible through every
  reference to the buffer.

## Decision 31: Streaming Commits Windows at a Pause, and New Windows Inherit the Language

**Choice:** When the live window must be committed, search its last 2 s for a
low-energy run of at least 120 ms (RMS at most half the window median), cut at
the run's centre, decode the truncated window once more for its final text
(rolling back the prefix by 10 tokens per carried second on top of the usual
`unfixed_token_num`), and carry the audio after the cut into the new window.
A new window is prompted with the language already detected for the stream;
a forced language always wins. Both default on (`commit_at_silence`,
`commit_lookback_sec`, `commit_min_silence_sec` on `init_streaming`).
**Alternatives (rejected):** text-level dedupe at the seam (fragments of a cut
word do not match the whole word, and legitimate repeats would be deleted);
overlapping audio without truncating the old window (duplicates the overlap
text); cutting at the single quietest 20 ms frame (measured: it landed in the
stop closure of "что" and duplicated the word on both sides).

**Rationale:**
- The official recipe never bounds the window, so the boundary is our
  problem alone. Hard cuts at 30 s multiples showed up in hypotheses as
  spurious sentence breaks ("for. Cost-saving", "EPC. Earlier", "que.") and
  once as a hallucinated phrase ("Very little. The middle of this world").
- Long-form lane, identical code, hard vs silence cut: fixed 12.28% -> 11.87%,
  energy 12.08% -> 12.23%. The energy rise is one English row where the model
  wrote three numbers as words instead of digits (3-4 word errors each); it is
  window-content variance, not a boundary effect. Every hard-cut boundary
  artifact is absent from the silence-cut output.
- The language carry fixed a real failure exposed by the shorter first chunk
  of a carried window: a Hindi stream re-detected as Indonesian and looped
  (59% error on that row, 25% with the carry). It also matches the official
  recipe, which detects language once per unbounded window.
- Cost is one extra window decode per commit: +9.6% RTF interleaved on 75-90 s
  clips (0.0765 -> 0.0838), still below the pre-Decision-30 0.104.
- Artifacts: `docs/benchmarks/2026-09-19-streaming-manifest-longform10-{hard,silence}-cut.json`.
