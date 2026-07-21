# Code Review: rechunkit + cfdb rechunking integration

**Date**: 2026-07-20
**Scope**: `~/git/rechunkit` (all of `rechunkit/main.py`, docs) and its use in cfdb (`support_classes.py` `Rechunker`/`DatasetRechunker`, `indexers.py`, `main.py` `iter_chunks`/`groupby`, `utils.py` period utilities and chunk-shape guessing, `dtypes.py` blank/fill interactions, `docs/concepts/rechunking-internals.md`).
**Method**: dual + blind + parallel per standing process — my own hands-on pass (differential oracle tests, memory probes, instrumented perf runs; scripts in `/tmp/reviewer_a/`) and an independent fresh-context Claude subagent with an adversarial brief on the same scope (scripts in `/tmp/reviewer_b/`), neither seeing the other's findings; then synthesis. **Full convergence** — every finding unique to one arm reproduced independently by the other during synthesis; zero disagreements. No code in either repo was modified.

Baselines before testing: rechunkit suite 48/48; cfdb suite 217/217 (S3/edataset tier excluded). The rechunkit installed in cfdb's venv (0.5.1) is byte-identical to the repo source, so cfdb-level tests exercised the reviewed code.

---

## Verdict

**The core logic is correct — and impressively so.** Across ~2,300 combined brute-force-oracle configurations (both arms; 1–3 dims; random + systematic shapes; aligned and phase-misaligned selections; ideal, constrained-bulk, and single memory paths), `rechunkit.rechunker()` produced exactly correct data, yielded every target chunk exactly once in canonical C-order, and `calc_n_reads_rechunker` matched actual source-call counts in every single config. The load-bearing invariant cfdb depends on — every read starts on a source-chunk boundary in absolute coordinates and spans at most one source chunk per dim — held in all ~12,600 observed reads across both arms. The cfdb bridge is exact under negative coordinate origins (post-prepend), views on origin-shifted variables, missing chunks, and mixed-dtype/mixed-storage-chunk `DatasetRechunker` zip-synchronization (the canonical-order contract it relies on is real). `rechunking-internals.md`'s origin/phase two-misalignment story checks out against measured behavior; an aligned rechunk decompresses each stored chunk exactly once (1.00× measured, ~1.4 GB/s locally).

The defects found are at the **edges**: missing-chunk fills on encoded dtypes, index bounds checking, groupby period semantics, view/aliasing contracts, and several documented memory/read-count claims that the code does not actually honor.

---

## Confirmed bugs (severity order, all reproduced)

### 1. Missing chunks read through the rechunker decode to wrong values (encoded dtypes)
`support_classes.py:213` — the encoded-path blank chunk is `_make_blank_chunk_array(False)`, which fills with **0 unconditionally** (`support_classes.py:624-638`) instead of `dtype.fillvalue`. `DTypeTranscoder.decode` maps only `value == fillvalue` to NaN, so encoded 0 then decodes to `0/factor + offset` — a plausible real number.

- **Explicit nonzero fillvalue** (the documented "user already knows dtype_encoded/offset/fillvalue" API): missing chunks → `.data` gives NaN, `rechunk()`/`iter_chunks(chunk_shape=…)`/`groupby()` give **50.0** (the offset) in the repro. Silent data fabrication.
- **Int-encoded dtypes** (`dtype('int32', min_value=.., max_value=..)` → `fillvalue=None`): `.data` gives 0, rechunker paths give **min_value − 1**. Two read paths of the same variable disagree.
- The auto-computed float encoding always sets `fillvalue=0`, so the default float path is safe (NaN both ways — verified).
- **Suspected propagation** (not run): netCDF export uses `iter_chunks(decoded=False)`, so exported files with missing chunks + nonzero fillvalue would carry encoded 0 ≠ `_FillValue`.

*Fix shape*: fill the encoded blank with `dtype.fillvalue` (0 only when fillvalue is None), and decide the int-encoded missing-value story deliberately (encoded ints currently have no representable missing value at all).

### 2. Index bounds holes in `indexers.py` — silent blank/phantom data
- `dv[n]` on a length-n variable returns `array([nan])` instead of raising — off-by-one at `indexers.py:169` (`>` should be `>=`).
- `dv[-1]` returns `array([nan])` (reads a phantom chunk before the origin) — negative indexing is neither supported nor rejected (`indexers.py:165-174`).
- Slice stops beyond the extent are accepted: `dv[35:60].data` on a length-40 variable returns a **shape-(25,) array** padded with phantom NaNs (`slice_slice`, `indexers.py:177-201`, validates nothing against `var_shape`). Note the rechunkit `sel` path *does* validate (`main.py:336-338`); only cfdb's direct-read path is affected.

All three are the same class: out-of-range access returns fabricated fill values instead of an error.

### 3. `DataVariableView.groupby` with irregular periods ('M'/'Y') on a sliced view returns wrong groups
`support_classes.py:1530` uses the **full** coordinate (`self._dataset[cn]`) to compute group boundaries, then `_groupby_period` applies them in **view-relative** index space. Measured: a March–June view returned groups sized 31/29/31/30 (Jan–Apr lengths at the view offset) instead of 31/30/31/30 (Mar–Jun). Dataset-level `ds.select(...).groupby(...)` resolves coords through `self[cn]` (view-relative) and is **correct** — verified; only the variable-level view path is broken.

### 4. groupby period fast path anchors at the first timestamp, not calendar boundaries
`period_to_chunk_size` (`utils.py:1053-1089`) checks only that `period/step` is an integer — never that `coord[0]` lies on a period boundary. With `step` set, `{'time': 'D'}` on an hourly coordinate starting 06:00/07:00 yields fixed 24-value windows each spanning **two calendar days**; the same call on a step-less coordinate takes the `compute_time_groups` fallback and produces true calendar days. Same spec, two semantics, chosen by an unrelated property; the docs (data-variables.md:154, CLAUDE.md "uniform groups") promise calendar alignment but nothing enforces the precondition. *Fix shape*: fall back (or offset the first group) when `coord[0]` truncated to the period ≠ `coord[0]`.

### 5. Data var created while its coordinate is empty gets `chunk_shape=(0,)` baked into metadata
`rechunkit.guess_chunk_shape((0,), …)` passes zero dims through; `utils.parse_var_inputs` calls it with the coord's current shape. Creating a data var with `chunk_shape=None` before appending coord data (the officially supported "create empty coords, append later" workflow — the named coord templates create empty) persists `chunk_shape=(0,)` → `ZeroDivisionError` on first write; the variable is permanently broken.

### 6. Yield aliasing — two related contracts nowhere documented
- **rechunkit direct-yields views into its internal buffer**; the pending-path yields copies. Holding yielded arrays across iterations silently corrupts an arbitrary, plan-dependent subset (measured: 80/256 at rechunkit level; 7/10 groups via public `var.groupby(...)`). Affects **unencoded** dtypes through `rechunk()`, `iter_chunks(chunk_shape=…)`, `groupby()`, `DatasetRechunker`; encoded dtypes are safe because `decode()` allocates. `list(var.groupby(...))` working or corrupting depends on the dtype's encoding — invisible to the user.
- **`iter_chunks` (storage-chunk mode) yields one shared blank array for every missing chunk** (`support_classes.py:696-714`, `1329-1350`): mutating a yielded missing-chunk array in place (a natural pattern) makes all later missing chunks yield the mutated values. Measured: a written 777.0 propagated.

*Fix shape*: either copy on yield (cost: one chunk-size memcpy per chunk) or document "consume/copy immediately" prominently in both repos.

---

## Major performance findings

### P1. Groupby-style targets with non-divisor chunks: ~100× read/decompression amplification
Two compounding causes (blind arm's find; mechanism independently confirmed):
- `calc_ideal_read_chunk_shape` (`main.py:144-148`) takes LCMs **without clipping to the array shape**. ERA5-like case — `shape=(720,721,1440)`, `src=(100,100,100)`, `groupby('time')` → `tgt=(1,721,1440)`: ideal read = `(100, 72100, 7200)` = **193 GiB**, vs **0.4 GiB** if clipped to the shape. The ideal path becomes unreachable at any sane `max_mem` even when a clipped ideal would easily fit; `calc_ideal_read_chunk_mem` (exposed to users for budgeting) inflates the same way.
- The constrained **single path** then assembles each full-spatial-field target chunk by re-reading every overlapping source chunk: ~96,000 reads vs 960 stored chunks = **100×** decompression (≈345 GB decompressed to iterate a 2.8 GB array). With divisor chunks amplification is exactly 1×.

Since `groupby('time')` (chunk size 1 → full spatial field per step) is a flagship cfdb iteration pattern, this is the single most impactful performance issue found. *Fix shape*: clip the LCM per-dim to `ceil(shape/src)*src`-style bounds, and/or an accumulate-into-N-target-buffers strategy on the constrained path.

### P2. The canonical-order `pending` buffer is unbounded by `max_mem`
`main.py:491-534`: when a read group spans multiple target-chunk rows, all not-yet-due chunks are held as **copies** until the row band completes — peak ≈ `(read_rows − tgt_rows) × full width × itemsize`, scaling with array width regardless of budget. Measured 30× `max_mem` (my arm) and **920×** (blind arm, wide-array construction), consuming the generator immediately in both cases. At SST/ERA5 scale this reaches GBs on a "128 MB" rechunk. Both `rechunker()`'s docstring and `rechunking-internals.md`'s memory-model table claim `max_mem` bounds the allocation — currently false.

### P3. Post-prepend misaligned storage grids: ~2^k× decompression (k = misaligned dims)
Coordinate origins from prepend shift the storage grid off the 0-based grid, so every rechunkit read spans 2 storage chunks per misaligned dim and the source function re-decompresses neighbours with no caching. Measured: a 1-element prepend on both dims of a 48 MB variable → **3.46×** decompress calls, **3.5×** slower (392 vs 1374 MB/s). A small decompressed-chunk cache in the source function (even 1–2 chunks) or passing the origin phase through to rechunkit's `sel` machinery would recover nearly all of it. (`.data` reads are unaffected — each chunk decompressed once either way.)

### P4. `max_mem` accounting holes in planning (rechunkit)
- Buffer allocation is `max(read_shape + phase, target_chunk)` **element-wise** (`main.py:479`); `calc_source_read_chunk_shape` budgets only `prod(read)`. In a 3,000-config sweep restricted to cases where the violation is avoidable, 37% allocated over budget, typically 1.5–4× (e.g. `src=(5,16)`, `tgt=(14,2)`, budget 640 B → buffer 1792 B). Phase adds up to ~2× more.
- Read counts are **not monotonic in `max_mem`** (docs claim "more memory always means fewer or equal reads"): `shape=(53,11) src=(3,8) tgt=(6,6)`: 512 B → 54 reads, 1024 B → **68**, 2048 B → 36. Root cause: the greedy grow heuristic (`main.py:222-239`) can pick target-grid-misaligned read shapes. 7/1500 randomized pairs violated monotonicity.

Neither breaks correctness (oracle clean throughout); both contradict documented contracts.

---

## Minor notes

- `guess_chunk_shape` rejects numpy integer shapes (`isinstance(v, int)` — hostile for a numpy-adjacent API; `operator.index` would fix), passes zero dims through (bug 5), and returns up to 1.5× `target_chunk_size` while the docstring says "only as large as target_chunk_size".
- `DatasetRechunker.calc_n_reads_rechunker` (`support_classes.py:323`): `total_writes = n_writes` (assignment, not `+=`) — defensible since writes are zip-synchronized, but reads are summed and the docstring says "for the batch"; reads as a typo.
- `Dataset.groupby(..., data_vars=[])` with an irregular period → IndexError at `main.py:371` (by inspection, not run).
- `written_chunks` set (constrained path) grows one tuple per target chunk — ~200 MB at 10⁶ chunks (analytic).
- Ideal-path "each source chunk read exactly once" holds only for phase=0; with a misaligned `sel`, boundary chunks are read once per adjacent group (≤2× per boundary, honestly counted by `calc_n_reads_rechunker`).
- Datetime encoding sizing calls `compute_int_and_offset(..., precision=1)` (`dtypes.py:503`), computing the range ×10 — can pick a wider int than needed (deliberate headroom?).

## Documentation corrections warranted

1. rechunkit `how-it-works.md:49-59` monotonicity claim — false (P4).
2. `rechunker()` docstring + cfdb `rechunking-internals.md:43,149-157` "max_mem bounds the allocation" — false (P2, P4).
3. cfdb data-variables.md:154 / CLAUDE.md groupby calendar-semantics claims — precondition stated, never enforced (bug 4).
4. Yield-lifetime contract (consume or copy immediately) — absent from both repos (bug 6).
5. `rechunking-internals.md` is otherwise accurate — the origin/phase story, fast-path guarantee, and multi-chunk assembly rationale all verified against behavior.

---

## Verification note

**Checked**: everything above labeled measured/confirmed was reproduced by running code — both arms independently for the convergent findings; each arm's unique findings were re-reproduced by the other during synthesis. Oracle coverage: ~2,300 configs across both arms, plus cfdb-level oracles with prepends/views/encoded dtypes; all `calc_n_reads` claims cross-checked against instrumented source-call counts. Test scripts: `/tmp/reviewer_a/t1–t8`, `/tmp/reviewer_b/` (7 harnesses).
**Not verified**: netCDF-export propagation of bug 1 (suspected, path read only); `Dataset.groupby(data_vars=[])` IndexError (inspection); `written_chunks` memory (analytic); geometry/String dtypes through the rechunker; EDataset/S3 rechunk paths; `map()`/multiprocessing paths. Gemini arm not run (would need Mike driving `agy` interactively — available on request; the Claude-side dual-blind was run instead).
**No code in either repo was modified.** Findings entered in `envlib/OPEN_WORK.md` Backlog (top item, 2026-07-20).
