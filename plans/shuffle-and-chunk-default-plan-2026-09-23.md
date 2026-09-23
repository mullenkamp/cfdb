<!-- Approved by Mike 2026-09-23 after review round cfdb-shuffle-plan-1b; implemented in the 0.10.0 working tree. This replaces the earlier unreviewed draft of the same name. -->

# Plan — byte-shuffle compression and a smaller default chunk size for cfdb (revised after review `cfdb-shuffle-plan-1b`)

## Core (read this; detail follows)

**What:** (1) add a byte-shuffle filter in front of cfdb's compressors, as new compression values
`zstd_shuffle` / `lz4_shuffle`, with `zstd_shuffle` the default for new files; (2) lower the default
chunk target for DATA VARIABLES to **2¹⁸ = 262 144 elements** (a byte target scaled by item size:
256 KiB for 1-byte types up to 2 MiB for 8-byte), coordinates unchanged at 2 MiB; (3) fix two
existing bugs the review found in the paths this change relies on: `Dataset.copy()` corrupts data
after a coordinate prepend, and `merge_into` masks the real error on an unreadable input.

**Mechanism:** the shuffle lives in the `Compressor` enum (cfdb-models), not a flag. Each variable
gets a compressor bound to the item size of the bytes it actually compresses, from one kind-gated
helper (String/Geometry/1-byte → no shuffle). The filter is the benchmarked bit-op byte-plane split.

**Top risks (all verified in review):** a wrong item size round-trips undetected, so tests check the
stored byte layout directly; `dtype.itemsize` is the DECODED width and String/Geometry numpy
placeholders report 16, so the helper must gate on kind and never use `dtype.itemsize`; old readers
fail loudly (verified on every read path), and must keep doing so.

## Context

Evidence: `benchmarks/compression/README.md` (findings 1–7), `benchmarks/RESULTS.md` ("Per-chunk
costs", "Remote (EDataset) reads", "Station (ts_ortho) data"). Shuffle + zstd-1 gives ~0.72× file
size on packed grids and station series, compresses faster everywhere and decompresses faster above
~4–12 K elements per chunk. Local reads are best at ~130–540 K elements per chunk. A flat byte target
cannot hit that for every item size: at 512 KiB rechunkit's guesser gave 8-byte types 27–35 K
elements (verified); at 2¹⁸ elements × itemsize every tested shape lands at 121–346 K.
Rechunking evidence (added 2026-09-23 after Mike asked for more investigation; to be written into
`benchmarks/RESULTS.md` as step 9): (a) rechunkit's `calc_n_reads_rechunker` on full-size shapes (SST
53 GB, forecast, stations) at the chunk shapes cfdb would guess for 256 KiB–4 MiB: whole-array
rechunks decompress the same bytes from 256 KiB to 2 MiB and 2× more at 4 MiB; subset queries
decompress far less with smaller chunks (SST point series 293× vs 2 927× at 2 MiB); 512 KiB and
1 MiB often guess the SAME shape. (b) Measured cfdb rechunking on d01 at those guesses, with a 4 MiB
memory budget to emulate larger-than-memory data: best at ~346 K elements for BOTH uint16 (512 KiB)
and uint32 (1 MiB); 4 MiB chunks 5–30× slower; ≤138 K elements up to 4–5× slower for pixel series;
with the default 512 MiB budget all chunkings are equal. Scripts:
`scratchpad/verify/rechunk_plans.py`, `rechunk_timing.py` (copy into `benchmarks/results/` in step 9).

Decisions (Mike, 2026-09-23): default `zstd_shuffle`; names `zstd_shuffle`/`lz4_shuffle`; coordinates
keep 2 MiB; the ~32 K-element floor is documentation only; fix the `copy()` prepend bug in this
change; add `format_version` now; chunk target **2¹⁸ elements × itemsize** (decided after the
rechunking evidence). Version assumed 0.10.0.
Review record: `benchmarks/results/review-cfdb-shuffle-plan-1.md` (to write, step 9).

## Steps

0. **cfdb-models** (`cfdb_models/data_models.py`): add `zstd_shuffle`, `lz4_shuffle` to `Compressor`;
   0.1.1 → 0.1.2. Release is Mike's; verify by PyPI presence.
1. **Tests first (step 7's list), then the fixtures:** before any code change, write two fixtures with
   the CURRENT release (grid and ts_ortho) and check them in.
2. **Item-size helper** (`support_classes.py` or `dtypes.py`): `shuffle_itemsize(dtype)` returns
   `np.dtype(dtype_encoded if not None else dtype_decoded).itemsize` only when `dtype.kind` is in
   `b i u f M`, else `None`; never reads `dtype.itemsize` (decoded width, mutated by
   `infer_itemsize`). Itemsize 1 behaves as no shuffle.
3. **Compressor** (`support_classes.py`, `class Compressor`): accept the four values; keep the
   bytes interface; shuffle variants split into planes with `(u >> 8*i).astype('u1')` and rejoin with
   shifts/ORs, porting `byte_planes`/`join_planes` from `benchmarks/compression/codecs.py`;
   `__reduce__` returns `(compression, compression_level, itemsize)`.
4. **Binding** (`support_classes.py:644-645`): build `self.dtype` first, then
   `self.compressor = dataset._compressor_for(shuffle_itemsize(self.dtype))`; `main.py:763` becomes a
   per-itemsize cache keeping the attribute name `_compressor` (`DatasetView` at `main.py:888` copies
   it; it is never read there).
5. **Defaults, errors, format version** (`utils.py:68-69`, `main.py:740-765, 1015-1041`,
   `edataset.py:83-89`): options gain the two values; default levels 1; `open_dataset` and
   `open_edataset` default to `'zstd_shuffle'` (`open_edataset` switches from hard-coded level 1 to
   `None` + the defaults table). Catch the msgspec `ValidationError` at `$.compression` on open and
   re-raise with "this file needs a newer cfdb / cfdb-models (compression '<value>')". Add
   `format_version: int = 1` to `SysMeta` (written by this version; future format changes bump it
   and readers refuse higher values with a readable message); it does not replace the enum.
6. **Chunk default** (`utils.py:28, 660, 732`): data variables target `2**18 * itemsize` bytes
   (`data_var_chunk_elements = 2**18`); coordinates keep `2**21` under a clearly named constant.
7. **Existing bugs:**
   - `Dataset.copy()` (`main.py:510-528`): the raw-byte fast path writes source chunk bytes under keys
     computed from origin-shifted source slices, while `Coord.like()` creates target coordinates at
     origin 0. After a prepend the copy reads back wrong (verified). Take the raw path only when every
     coordinate of the variable has origin 0 (keys then align); otherwise use the decoded
     `iter_chunks`/`set` path.
   - `merge_into` (`merge.py:66-72, 206`): initialise `opened = []` before `_open_inputs` so the
     original exception surfaces.
8. **Tests** (each seen FAILING against current code first, then passing; re-apply each defect after
   its fix and require red):
   - item-size helper, per dtype: packed int/float/datetime → encoded width (packed float64 → 2), raw
     types → own width, bool/uint8 → 1, String/Geometry → None;
   - **structural shuffle test**, per numeric dtype: write with `zstd_shuffle`, fetch the stored chunk,
     decompress with plain zstd, assert it equals the byte planes of the encoded array at the expected
     item size (catches a wrong or skipped shuffle, which round trips cannot);
   - round trips through `set`/read, `iter_chunks`, rechunked iteration, `groupby`, for the four
     compressions × the dtype list (NaN payloads compared via `tobytes()`), plus coordinates;
   - fixtures: the two current-release files read bit-identically afterwards; add two files written by
     the new version (shuffled, item sizes 1/2/4/8, grid and ts_ortho) so the binding cannot drift.
     Content: packed u16 and u32, raw float64, uint8, bool, raw and packed datetime64 with NaT, str,
     geometry, a prepended and an appended coordinate, missing chunks, attrs; read also through a
     `select()` view;
   - loudness: metadata recording `zstd_shuffle` decoded with an enum lacking it raises; the new
     message appears on open; `format_version` above the supported value is refused;
   - `copy()`: shuffled source, all four compressions, and a PREPENDED source (red today) — identical
     data and recorded compression;
   - `merge_into` with an unreadable input surfaces the real error (red today);
   - multiprocessing: `DataVariableView.map(..., n_workers=2)` on a shuffled variable returns correct
     decoded data;
   - chunk default: for item sizes 1/2/4/8 on grid, ts_ortho and forecast shapes, guessed elements lie
     within ~100 K–400 K (fails on BOTH sides), and coordinate guesses are unchanged;
   - EDataset push/pull of a shuffled dataset: credential-gated, marked.
9. **Docs:** rewrite (not just re-number) the rationale at `docs/concepts/chunking-storage.md:44`;
   update `docs/guide/data-variables.md`, `docs/guide/rechunking.md`, `docs/reference/dataset.md`,
   `docs/reference/data-variable.md`, `docs/changelog.md`, `README.md`, repo `CLAUDE.md`, and the cfdb
   skill: new values and default, the element-based target, ≥ ~32 K-element guidance, that
   `lz4_shuffle` buys size not decode speed, and that attaching to an existing remote uses the remote's
   compression. Cite benchmark files, don't restate numbers. Add a "Rechunking vs chunk size"
   section to `benchmarks/RESULTS.md` from the Context evidence, copying `rechunk_plans.py` and
   `rechunk_timing.py` into `benchmarks/results/rechunk-2026-09-23/`. Write the review record
   `benchmarks/results/review-cfdb-shuffle-plan-1.md`; replace the superseded repo draft
   `plans/shuffle-and-chunk-default-plan-2026-09-23.md` with this plan.
10. **Version and floors:** cfdb 0.9.7 → 0.10.0; `cfdb-models>=0.1.2` in the same commit, reason added
    to the pyproject floor comment block.
11. **After release (Mike):** refresh locks in the six downstream repos; ingest base-image rebuild;
    tick the `OPEN_WORK.md` items (shuffle, chunk default) and add the `copy()` fix to Done.

## Out of scope (backlog stays open)

`set()` per-call overhead; decode-buffer reuse; not storing all-missing chunks; cfdb-ingest's
chunking; ungrouped-dataset guidance in envlib-ingest-base; a `Compressor` interface returning arrays;
whether `combine()` should inherit the first input's compression when inputs differ.

## Verification

- `uv run pytest` (EDataset tests need the S3 config), including every new test seen red first and
  red again under its re-applied defect.
- End-to-end: `combine([d01_subset], new, compression='zstd_shuffle')`, then
  `benchmarks.compression.codec_bench new.cfdb` — stored size ~0.72× the zstd-1 original.
- Guessed chunk shapes printed for the review's shape set: 121–346 K elements.
- `uv run ruff check` on touched files: no NEW violations vs. the current baseline.
- Source left staged, not committed, for Mike's walkthrough.
