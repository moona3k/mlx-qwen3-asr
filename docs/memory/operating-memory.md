# Operating Memory

Primary memory entry point for agents.

This file is intentionally the front door: it contains protocol and current
compacted memory. Append-only history lives in `events/`.

## Protocol (Stable Rules)

### Two Tracks

1. `events/` is append-only and immutable.
2. `operating-memory.md` is compacted and mutable.

### Event Rules

1. Append events in `events/YYYY-MM.md` with IDs: `MEM-YYYY-MM-DD-NNN`.
2. Never rewrite prior event content; append follow-up corrections.
3. Minimum bar per event: `Decision`, `Reuse next time`, `Evidence`.

### Compaction Rules

1. Every compacted memory item must reference one or more event IDs.
2. Keep compacted guidance short, actionable, and current.
3. Promote repeated high-value patterns into `Distilled Learnings`.

### Promotion Rule

Promote to distilled learnings when:

1. A pattern appears at least twice, or
2. It prevented a costly miss/regression.

### Agent Update Flow

1. For non-trivial and meaningful work, append an event first.
2. Update compacted memory only if active guidance changed.
3. Keep references (`refs`) accurate.

## Current Operating Memory

## Active Defaults

1. For non-trivial and meaningful implementation work, append an immutable
   event first.
   - refs: `MEM-2026-02-16-002`, `MEM-2026-02-16-003`
2. Keep memory notes minimal but actionable:
   - include `Decision`, `Reuse next time`, `Evidence`.
   - refs: `MEM-2026-02-16-001`, `MEM-2026-02-16-002`,
     `MEM-2026-02-16-003`
3. Keep memory guidance lightweight and flexible to preserve contribution
   velocity.
   - refs: `MEM-2026-02-16-001`, `MEM-2026-02-16-002`,
     `MEM-2026-02-16-003`
4. Keep `CLAUDE.md` canonical and keep `AGENTS.md` as a thin compatibility
   delegator.
   - refs: `MEM-2026-02-16-004`
5. Test doubles carry the real signatures of what they replace; production
   code never introspects for test compatibility.
   - refs: `MEM-2026-09-06-005`
6. In the shared working tree, stage explicit paths; never `git commit -a`
   (it swept an unrelated uncommitted note into a fix commit).
   - refs: `MEM-2026-09-19-012`

## Distilled Learnings

1. Run the suite against the newest MLX release before merging anything that
   touches threading, dtype or the audio front end; the repo venv pins an
   older MLX that hides thread-affinity and `as_strided` behaviour changes.
   - refs: `MEM-2026-09-06-001`, `MEM-2026-09-06-003`

2. When changing process policy, update both `CLAUDE.md` and operating memory
   in the same commit to avoid drift.
   - refs: `MEM-2026-02-16-001`, `MEM-2026-02-16-002`,
     `MEM-2026-02-16-003`
3. For multi-agent compatibility, prefer one canonical guide plus one thin
   delegator file.
   - refs: `MEM-2026-02-16-004`
4. Any decode path that departs from how the model was prompted in training
   (streaming, speculative, chunking) needs a lane that scores its output
   against references before it ships. Stability and latency metrics alone
   passed a decoder that was 56% wrong. When building such a gate, prove it
   against the committed failing artifact in a test, not just against the
   current passing one.
   - refs: `MEM-2026-09-19-011`, `MEM-2026-09-19-013`
5. Judge numerics changes by the measurement closest to the change (encoder
   output vs the fp32 reference), not by downstream greedy token match, which
   flips on borderline fp16 decisions in both directions. Record the reference
   stack (torch/transformers versions) next to every parity number.
   - refs: `MEM-2026-09-19-010`
6. Quantization quality bar for publication: hypothesis-level diff against
   the committed fp16 rows, named baseline file, reproduce command pinned to a
   release tag. The audio encoder carries most of the 4-bit loss; quantize it
   at 8 bits.
   - refs: `MEM-2026-09-19-012`
7. Scripts import the checkout they live in (`_repo_path`); when measuring
   before/after from a worktree, print `mlx_qwen3_asr.__file__` first.
   - refs: `MEM-2026-09-19-009`
8. When a parity number looks wrong, bisect on input length and rerun in
   fp32 before blaming MLX. A clean step at a structural boundary (here 800
   mel frames) with no fp32 improvement means the two implementations differ
   in kind; then read the reference's attention path, not the MLX one. The
   reference can be the side that departs from training.
   - refs: `MEM-2026-09-19-014`
9. Profile before caching. The streaming item assumed the encoder dominated;
   it was 18% (prefill 31%, generation 50%). A cache that adds kernel launches
   (separate block encode + KV extension) was 12% slower on short clips even
   though it saved compute; folding the cached work into the calls that
   happen anyway (one encode, one prefill, trim after) made it a win on both
   lanes. Measure cache-off and cache-on back to back on the same machine.
   - refs: `MEM-2026-09-19-016`

## Open Risks

1. Memory updates are currently social-process enforced, not CI-enforced.
   - refs: `MEM-2026-02-16-001`, `MEM-2026-02-16-002`
2. Long-form streaming sits 1.3pp under the strict 3pp ceiling (12.28% vs
   13.59% after prefix reuse); the window-overlap item must not spend it.
   - refs: `MEM-2026-09-19-014`, `MEM-2026-09-19-016`; `docs/EVAL_GAPS.md`.
3. The shipped CPU `qwen-asr` reference does not window encoder attention
   (mask defined, never called). Every encoder-output number measured
   against it on clips over 8 s before 2026-09-19 overstates MLX error;
   `2026-09-19-encoder-parity-tail-padding.json` long-clip magnitudes are
   affected. Reported upstream as QwenLM/Qwen3-ASR#213 (open).
   - refs: `MEM-2026-09-19-014`
4. Publishing (PyPI, HuggingFace) runs from one maintainer machine by
   decision; the HF token lives in the gitignored `.secrets/hf_token`. A
   second maintainer needs their own token file, not a repo secret.
   - refs: `MEM-2026-09-19-012`, `MEM-2026-09-19-018`

## Handoff (2026-09-19)

Start with `docs/reviews/2026-09-19-day-review.html` for the concepts, then
the "Handoff" section of `docs/ROADMAP.md`. Items 1-5 closed the same day
(events 013-018, Decisions 30-31); item 6 needs an 8-16 GB machine. Before
changing streaming or encoder code, reproduce: `scripts/eval_librispeech.py
--samples 100`, both strict streaming lanes cache-off/on
(`eval_streaming_manifest.py --[no-]reuse-window-prefix`), and
`scripts/eval_aligner_encoder_parity.py --samples 50`.

## Revisit Triggers

1. If memory entries grow noisy or stale, tighten compaction and pruning rules.
   - refs: `MEM-2026-02-16-001`, `MEM-2026-02-16-002`
