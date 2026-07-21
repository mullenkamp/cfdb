# Critical design review brief — rechunkit/cfdb performance-round plan

I'm the author of this plan, reviewing my own design before implementation. Please act as an independent critical reviewer: stress-test the design, verify its claims against the actual source code, and probe for failure modes I did not think of. You may write and run throwaway experiments in `/tmp` where useful — measured evidence beats reasoning by inspection.

## Object under review

`plans/perf-round-plan-2026-07-21.md` (this repo). It plans performance fixes for four findings (P1–P4) from a completed code review.

## Context documents (read first)

- `plans/rechunkit-cfdb-review-2026-07-20.md` — the review report; §"Major performance findings" defines P1–P4 with measurements.
- `plans/perf-round-planning-notes.md` — carried-over scope/process notes.

## Code scope (same for all reviewers)

- `/home/mike/git/rechunkit` — the whole algorithm is `rechunkit/main.py` (~550 lines); also `docs/concepts/how-it-works.md`, `docs/concepts/optimization-internals.md`, `tests/`.
- `/home/mike/git/cfdb-repos/cfdb` — `cfdb/support_classes.py` (Rechunker, DatasetRechunker), `cfdb/indexers.py` (`slices_to_chunks_keys`, `index_combo_all`), `cfdb/utils.py`, `docs/concepts/rechunking-internals.md`, `cfdb/tests/`.
- `cfdb/tests/s3_config.toml` and any `*.ebooklet` files are git-ignored local test config, out of scope — don't open.

Both repos are at their committed states (rechunkit `db5ca6c`, cfdb `5f2dbbe`); the plan proposes changes on top of these.

## Hard constraints the design must not break

1. Every target chunk is yielded exactly once, in canonical C-order of the target chunk grid (cfdb's `DatasetRechunker` zip-synchronizes several per-variable generators and depends on identical ordering across them).
2. Every source read starts on a source-chunk boundary in absolute coordinates and spans at most one source chunk per dim (cfdb's source function fast path depends on this).
3. Data exactness — differential oracles against plain numpy must stay exact.
4. Public API backward compatibility — `calc_ideal_read_chunk_shape` and `calc_source_read_chunk_shape` gain only optional parameters.

## Deliverable

For each part below: a verdict — SOUND / UNSOUND (with the specific failure) / SOUND WITH CHANGES (with the specific change). **Begin each part by quoting the one or two plan sentences you are evaluating**, so the verdicts map unambiguously to plan text. Then, separately: a list of anything important the plan does NOT consider (interactions, edge cases, workloads where a "fix" makes things worse).

### Part A — R1 true-buffer budgeting (plan §R1)

Is `_buffer_bytes = prod(max(read+phase, target))·itemsize` actually the allocation `rechunker()` makes (main.py:488)? Is the "parse sel first, call `calc_source_read_chunk_shape` with identical args from both `rechunker()` and `_rechunk_plan`" restructuring sufficient to keep plan and buffer in agreement, and what happens if a caller passes `phase=None` where sel exists? Is returning `source_chunk_shape` as a documented floor when even one source chunk busts the budget the right contract?

### Part B — R2 ideal-shape clipping (plan §R2)

The claim: per-dim, when `lcm(src,tgt) ≥ cover = ceil(shape/src)·src`, using `cover` instead of the lcm is safe because that dim then has exactly one group, so cross-group alignment is moot. Verify this against `_rechunk_plan`'s ideal path (main.py:345-364), especially: the phase-shifted reads (`grp_start − phase`), the buffer extent argument (`max(ideal+phase, tgt) ≥ target_shape+phase`), whether the ideal-path trigger (`source_read == ideal`) stays consistent when both sides are clipped, and whether any read can now violate constraint 2. Also: is `shape=target_shape` (the sel window extent) the right clipping extent when a sel is present, or should it involve the phase?

### Part C — R3 batched single path (plan §R3)

The biggest new mechanism: consecutive single-path write chunks batched into `('multi', deduped_reads, writes, None)` groups; generator allocates one dedicated buffer per batched write chunk and scatters each source read into all overlapping buffers. Probe: canonical order and exactly-once under interleaved bulk/multi groups; the `n_batch` memory bound (is `max_mem − src_bytes − reserve` the right reserve accounting? is the transient decompressed source array inside `source()` counted anywhere?); whether `calc_n_reads_rechunker` stays truthful (it counts `len(read_chunks)`, now deduped); whether the scatter arithmetic (generalizing today's single-path `get_slice_min_max` intersection) is exactly specified; degenerate cases (n_batch=1, a batch spanning a bulk-group boundary, 1-dim arrays, target chunk larger than max_mem). Run a small prototype in /tmp if that's the fastest way to check the read-count claims (~3,720 for the ERA5-analog case in the plan).

### Part D — R4 pending-aware budget (plan §R4)

Is the `_pending_bytes` dominant-term formula a true upper bound on what the `pending` dict holds (main.py:500-544)? Does the shrink cascade terminate, and is "shrinking the group dim in lcm multiples is read-count-free" actually true in the constrained-bulk regime (groups still tile both grids)? Is the ≤2×-boundary-reads claim for sub-lcm shrinks right? Is the documented residual (irreducible when `src0 > t0` on wide arrays) correctly characterized, or is there a cheaper escape?

### Part E — R5 narrow candidate check (plan §R5)

`docs/concepts/optimization-internals.md` records a fuller candidate search that was implemented and rejected. The plan's version: 3 candidates, scored by running `_rechunk_plan` itself in counting mode, capped at ~20k target chunks. Does the "plan-as-scorer" refactor (a private `_read_chunk_shape` override) risk perturbing plan behavior for normal calls? Is `greedy(max_mem//2)` sufficient to fix the review's non-monotone case (`shape=(53,11)`, `src=(3,8)`, `tgt=(6,6)`, 512→54 vs 1024→68 reads)? Is the gate (1,500-config sweep: aggregate ≤ old, no case > +5%, violations strictly decrease) the right acceptance bar, and does the prior rejection's reasoning still argue for docs-only?

### Part F — C1 storage-space rechunk in cfdb (plan §C1)

The transform: `d = origin % chunk`, declare shape/sel shifted by `+d` so rechunkit's phase machinery aligns reads to the real storage grid; source functions shift back by a constant; claim: yielded slices are unchanged (sel-relative), so no consumer changes. Verify: the algebra against `indexers.slice_slice`/`slices_to_chunks_keys` (how origins enter today); rechunkit's sel validation bounds (main.py:336-343); the "yields unchanged" claim (worth an experiment: record yield sequences before/after a prototype of the transform on a prepended fixture); view composition (`_var._sel` is full-variable 0-based user space); `DatasetRechunker` ordering when per-variable `d` differs; empty variables; `calc_n_reads_rechunker` consistency. Also assess the plan's choice to keep the multi-chunk assembly loop as a fallback rather than deleting it.

### Part G — Cross-cutting

The Phase-0 benchmark plan (are the new count-based cases the right gates? anything unmeasured that should be?); the per-step checkpoint/pause protocol; the doc corrections (are the corrected max_mem/monotonicity statements accurate post-change?); interactions between steps (e.g. does R4's shrink interact badly with R2's clip or R3's reserve?); interactions with the just-shipped bug-round behavior (encoded-dtype fill consistency, legacy-int-fill warning, yield-lifetime contract); and the omissions hunt — what should this round fix or measure that the plan misses entirely?

## Notes

- Nonconforming inputs (odd shapes, tiny max_mem, 1-dim, phase at chunk-size−1) are fair game for experiments.
- If a part's design is sound but a stated number is wrong (read counts, memory bounds), say so explicitly — the numbers drive the pause-and-assess gates.
- Summarize the brief back before executing if that helps you structure the review.
