# Perf-round planning notes (carried over from the 2026-07-21 bug round)

Input material for planning the rechunkit/cfdb performance round (P1–P4 from the
2026-07-20 review). Read alongside `rechunkit-cfdb-review-2026-07-20.md` (the full
dual-blind review report, same directory) and the findings entry in
`envlib/OPEN_WORK.md`.

## Scope (from the review; deferred out of the bug round)

- **P1 — groupby-style read amplification (~100×, the big one)**: `calc_ideal_read_chunk_shape`
  takes per-dim LCMs without clipping to the array shape (ERA5-like case: ideal read 193 GiB vs
  0.4 GiB clipped), pushing full-spatial-field targets onto the constrained single path, which
  re-reads every overlapping source chunk per target chunk (~100× decompression measured at plan
  level; up to ~360× for esa-sst-shaped `groupby('time')`). Design pointers: clip the LCM per-dim
  (bounded by ceil(shape/src)*src-style limits); optionally an accumulate-into-N-target-buffers
  strategy for the single path.
- **P2 — canonical-order `pending` buffer unbounded by max_mem**: holds copies of one full
  target-chunk band; measured 30×–920× max_mem; scales with array width. Pointers: band-aware
  group ordering, or shrink read groups along dim0 when the band buffer would dwarf max_mem;
  at minimum document honestly.
- **P3 — post-prepend misaligned grids (~2^k× decompression)**: every rechunkit read spans 2
  storage chunks per misaligned dim; measured 3.46× decompress / 3.5× slower for a 1-element
  2-dim prepend. Pointers: small decompressed-chunk cache in cfdb's source function, OR (more
  elegant) pass the coordinate-origin phase through rechunkit's EXISTING sel/phase machinery —
  origins are just a second phase.
- **P4 — max_mem accounting + monotonicity**: buffer alloc is element-wise
  `max(read+phase, target)` (37% of avoidable cases 1.5–4× over budget); read counts are
  non-monotonic in max_mem (54→68 reads on a 2× budget; root cause per the blind review arm:
  the greedy grow in `calc_source_read_chunk_shape` picks target-grid-misaligned read shapes —
  candidate fix: align the grow to the target grid).
- **Deferred doc corrections that ride with the fixes** (deliberately left false in the bug
  round): rechunkit `how-it-works.md` monotonicity claim ("more memory always means fewer or
  equal reads" — measured false); `rechunker()` docstring + cfdb
  `docs/concepts/rechunking-internals.md` Memory Model table max_mem-bounds claims (violated by
  P2/P4). Fix the text alongside the behavior.

## Benchmark harness state + required prep

- Harness: `rechunkit/benchmarks/bench_rechunk.py` + `cfdb/benchmarks/bench_core.py`
  (median-of-5, seeded, JSON + `--compare`; rechunkit suite also captures tracemalloc peaks).
- **Baseline for the perf round = `benchmarks/results/after-bugfix-2026-07-21.json`** in each
  repo (bug-round deltas folded in; bug round showed zero regressions, encoded partial writes
  +10–20% faster from the encoded-space overlay).
- **Known harness gap — fix BEFORE capturing the perf baseline**: the rechunkit micro-cases run
  3–10 ms and same-tree variance is ±8–19% — too noisy to gate rechunkit-core changes, which is
  exactly what the perf round modifies. Scale configs up (bigger shapes and/or more N_RUNS)
  first, then re-capture.
- **Missing targeted cases to add**:
  - a NON-divisor groupby-amplification case for P1 (the existing `groupby_daily_fast` is
    divisor-aligned = the happy path; need a `(1, full, full)`-style target with non-divisor
    chunks, measuring reads/decompressions, not just time);
  - a wide-array pending-buffer memory case for P2 (peak-mem is already instrumented);
  - P3's case already exists: `rechunk_prepended` at ~376 vs ~1374 MB/s aligned — the ~3.7×
    win to harvest.

## Process notes

- Same working trees as the bug round (uncommitted until Mike's review); changelog entries
  extend the same Unreleased sections; ONE release cycle covers both rounds (Mike's ruling
  2026-07-21), with the `rechunkit>=` pin bump at release.
- Dual-blind plan review again per standing process. Check every reviewer section maps to a
  real plan item before counting its verdict (Gemini confabulated 3 of 7 sections in the
  bug-round review — round 2 with a focused brief was the remedy).
- At RELEASE (after both rounds): refresh the `cfdb` skill (`~/.claude/skills/cfdb/SKILL.md`)
  with the new indexing semantics, groupby anchoring, missing-value semantics, empty-coord
  raise (yield-lifetime note already added 2026-07-21); date-stamp the stale `0.9.4
  (unreleased)` changelog heading.
- Correctness safety net for perf changes: the bug round's differential oracles
  (~2,300 configs across both review arms) — rerun the t1/t4-style sweeps after any planning
  change to `_rechunk_plan`/`calc_source_read_chunk_shape`; canonical C-order yield behavior
  is load-bearing for `DatasetRechunker` zip-sync and must be preserved (or the zip redesigned).
