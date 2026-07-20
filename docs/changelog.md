# Changelog

## Unreleased

The bug-fix round from the 2026-07-20 rechunkit/cfdb dual-blind code review (correctness only; the companion performance round follows before release).

- Fixed: **missing chunks read through rechunker-based paths decoded to wrong values for encoded dtypes.** `rechunk()`, `iter_chunks(chunk_shape=...)`, `groupby()`, and netCDF export filled missing chunks with encoded 0 regardless of the dtype's reserved fillvalue — with an explicit nonzero fillvalue a missing chunk decoded to a plausible real value (the offset) instead of NaN, and int-packed dtypes returned `min_value - 1` through these paths while `.data` returned 0. Missing data now reads as the decoded blank (NaN/NaT/0) identically on every path
- Changed: **auto-packed integer dtypes now carry `fillvalue=0`** (encoded 0 was always structurally reserved: `offset = min_value - 1` puts every legitimate value at code ≥ 1). `decode()` maps the reserved code to decoded 0; `encode()` is unchanged, so a legitimate 0 (when `min_value <= 0`) still packs to `1 - min_value` and can never collide with the missing marker (in files or netCDF exports). The fillvalue now round-trips through stored dtype dicts (previously the reconstruction branch dropped it on reopen). **Legacy files** (created before this release; no fillvalue in the stored dtype) keep their old read behavior — a `UserWarning` fires when rechunker/encoded paths read such int-packed variables, since missing chunks there still read as `min_value - 1` (and a rechunk-copy of such a file materializes that value); recreate the variable to adopt the new semantics. `combine()` accepts the old/new fillvalue difference (output adopts the new-style dtype)
- Fixed: **partial chunk writes to int-packed variables with `min_value >= 2` raised** (`integer value(s) outside the encodable range`) — the blank cells of a partially-written chunk went through `encode()`. Writes for packed dtypes now read-modify-write in encoded space: only the user's values are encoded, untouched cells keep their exact stored codes (no decode→re-encode round trip), and unwritten cells remain honestly missing on disk instead of becoming written values. The same fix covers int-packed coordinates with a partial last chunk (`append`/`prepend` raised the same way)
- Fixed: **indexing bounds now follow numpy semantics.** Previously `var[n]` on a length-n variable and `var[-1]` silently returned blank fill data (a phantom chunk), and over-long slices returned phantom-padded arrays (`var[35:60]` on a length-40 variable → 25 values). Now: negative ints wrap (`var[-1]` = last element), out-of-range ints raise `IndexError`, slice bounds clamp to the variable extent exactly like numpy, and fully-out-of-range or descending slices raise `ValueError`. Consequences: `set()` through an over-long slice expects the clamped shape, and `.loc` with a scalar label beyond the coordinate end raises `IndexError` (previously returned a phantom blank)
- Fixed: **`groupby` with month/year periods on a sliced variable view returned wrong groups** — boundaries were computed from the full coordinate but applied view-relative (a Mar–Jun view got Jan–Apr month lengths). Groups are now computed from the view's own coordinate window. (Dataset-level `select().groupby()` was already correct)
- Fixed: **time-period groupby with an enforced `step` ignored calendar alignment** — `{'time': 'D'}` on hourly data starting 06:00 produced fixed 24-value windows each spanning two calendar days, while the same call on a step-less coordinate produced true calendar days. The fast path now additionally requires the first coordinate value to sit on the period boundary and otherwise falls back to calendar groups, so both paths always produce identical groups. Note the documented anchoring rules: single-unit periods are calendar-aligned; multi-count periods (`'7D'`, `'6h'`) anchor at the first coordinate value; `'W'` follows numpy's Thursday-anchored week epoch (use `'7D'` for first-timestamp-anchored weeks)
- Fixed: **creating a data variable with `chunk_shape=None` while a referenced coordinate was still empty** baked a zero chunk shape into the file, making the variable permanently unwritable (`ZeroDivisionError`). This now raises a `ValueError` naming the empty coordinate(s) — pass `chunk_shape` explicitly or fill the coordinate first (see the new guidance in the data-variables guide). Explicit `chunk_shape` values are now also validated (> 0) and accept numpy integers
- Fixed: netCDF export of int-packed variables crashed (`TypeError` computing `scale_factor` from the integer packing's absent factor); such variables now export with `scale_factor = 1.0`, `add_offset`, and `_FillValue = 0`, so external CF readers see missing chunks as masked and unpack written values correctly
- Fixed: **netCDF export of packed datetime variables wrote offset-shifted timestamps** (the packed values were written raw under `units: ... since 1970-01-01`, shifting exported times by the packing offset — decades off). Datetimes now always export as true epoch ticks
- Fixed: `compute_int_and_offset` width selection was off by one — at ranges exactly filling an integer width, the declared `max_value` was unencodable (packed ints raised; packed floats silently decoded the declared max to NaN). Auto-selected widths now reserve room for the fill code; dtypes at exact boundary ranges may select the next width up compared with earlier releases (new files only)
- Fixed: auto-packed datetime dtypes (and integer/float packings given numpy-scalar `min_value`/`max_value`) produced a numpy-typed `offset` that crashed dataset close (`msgspec` cannot encode `numpy.int64`) — packed-datetime variables created via `min_value`/`max_value` were never persistable before this fix
- Changed: `iter_chunks()` yields a fresh blank array for each missing chunk (previously one shared blank was yielded for all missing chunks, so mutating a yielded array in place poisoned later yields). The general yield-lifetime contract is now documented on all chunk generators: consume or copy each yielded array before advancing the generator, and treat yields as read-only
- Changed: `DatasetRechunker.calc_n_reads_rechunker` now sums write counts across variables (matching how reads were already counted), so both numbers measure physical I/O operations for the batch
- Fixed: `Variable.items(decoded=...)` ignored its `decoded` parameter

## 0.9.4 (2026-07-19)

- Fixed (follow-up, same guard): the float branch compares against bounds prepared as exactly-representable values in the data's own dtype (an unrepresentable bound shrinks one ULP toward the valid range) — a naive float32 comparison against `iinfo(uint32).max` promoted the bound to `float32(2**32)`, letting the top ~256-ULP band through to the undefined cast. Native-dtype comparison also keeps this hot per-chunk write path cheap: ~0.1 ms guard overhead per 500k-value chunk (vs ~1.6 ms for a float64-copy approach), with the fillvalue substitution pass skipped entirely for chunks containing no unencodable values; and the packed-**Integer** path now **raises** on values whose scaled representation exceeds the encoded width (previously wrapped modulo the width into plausible ints; ints have no reserved missing value, so mapping to fillvalue is not an option — fail loud). Values inside the width but outside the declared min/max remain the caller's policy, as for floats.
- Fixed: **packed-dtype `encode()` fabricated data for unencodable values.** Writing NaN (or inf, or a finite value outside the `min_value`/`max_value` encoding range, or NaT for packed datetimes) through an int-packed dtype hit numpy's undefined float→integer cast: instead of the documented "out-of-range becomes NaN", values wrapped into plausible in-range numbers (e.g. NaN → 214747.36 on a `precision=4, 0–100000` uint32 encoding; 60.0 → −5.536 on a `−5–55` uint16 one). Worse, the fabricated result differed between numpy's scalar and SIMD code paths, so small arrays could round-trip "correctly" while production-sized arrays corrupted — which is exactly how it evaded the test suite. `encode()` now maps every unencodable value (non-finite, NaT, out-of-range) to the reserved fillvalue before the cast, so they decode back as NaN/NaT as documented. Found by the envlib-ingest-base Phase-7 adversarial review (dual-blind: both reviewers hit the same function from different inputs). Any dataset previously written with NaN-holed arrays through a **uint32+** packing on a SIMD path should be checked for the fabricated constant (`(2147483648 / 10^precision) + offset`); uint16 packings happened to cast NaN→0 (the fillvalue) on x86 and round-tripped correctly by luck — the live esa-sst dataset was verified to be in that category.

## 0.9.3 (2026-07-14)

- Fixed: chained `.loc` access on a temporary variable — `ds[var].loc[...]` (read or write; on datasets, views, and EDatasets alike) crashed with `ReferenceError: weakly-referenced object no longer exists`; only the bound form (`v = ds[var]; v.loc[...]`) worked. The loc indexer is now created per access via a property and holds a strong reference to its variable for the duration of the expression; because it is no longer stored on the variable, no reference cycle is introduced and finalizer/collection timing is unchanged. Found by the envlib esa-sst ingest — the chained form is the natural notebook idiom (`cat.query(...)[0].open()[var].loc[...]`)
- Note for user code: `v.loc` now returns a fresh indexer object on each access (`v.loc is v.loc` is `False`), and assigning to `v.loc` raises `AttributeError`

## 0.9.2 (2026-07-13)

- Changed: `EDataset.push()` now returns ebooklet's `PushResult` (passthrough; requires ebooklet >= 0.10.0): `result.updated`, `result.failures` (failed keys → error strings; pending changes retained for retry), and `bool(result)` = fully-successful push. Previously it returned True/False/dict — note the old partial-failure dict was truthy, so `if ds.push():` misread partial failure as success; any failure is now falsy
- Updated dependency pins: `ebooklet>=0.10.0` (persistent journal, generational storage format 2, typed exceptions, PushResult, offline read mode — see ebooklet's changelog; format-1 remotes need the one-time re-push upgrade described there)

## 0.9.1 (2026-07-12)

- Fixed: `open_edataset` now supports `dataset_type='ts_ortho'` (it used to raise TypeError at create) and returns the class matching the dataset's STORED type for existing remotes (it always returned `EGrid`, even for ts_ortho remotes)
- Changed: in both `open_dataset` and `open_edataset`, the `dataset_type` parameter now applies only when CREATING — existing datasets always open with their stored type and the matching class (previously the class silently followed the parameter, so an existing ts_ortho file opened without the parameter got a Grid-classed object)
- Added public `dataset_type` property (`'grid'` or `'ts_ortho'`) on all dataset classes and views — consumers no longer need to read the private sys-metadata
- Fixed: the in-memory sys-metadata `dataset_type` was a plain string on freshly-created datasets but an enum on reopened ones (msgspec does not coerce on construction); it is now always the enum. On-disk format unchanged
- Fixed: creating a coordinate on a REOPENED dataset crashed with `TypeError: argument of type 'Type' is not iterable` (latent consequence of the string/enum inconsistency above)
- Updated dependency pins: `booklet>=0.12.6` (iterator-contract release; bump happened post-0.9.0 and was previously unchronicled), `ebooklet>=0.9.4` (5xx retry, 404-integrity re-check, and the flag='n' second-push data-loss fix — found by this release's live tier)

## 0.9.0

- Fixed: `EDataset.push()` mid-session used to publish a stale/empty SysMeta (no variables) and no attributes — `push()`/`changes()` now flush all in-memory state to the local file first, so a push at any point publishes exactly the current structure. Pushing remains explicit: nothing is ever pushed automatically on close
- Fixed: `open_edataset` with `'w'`/`'c'` and a fresh local file used to treat an EXISTING remote dataset as create-new, which would overwrite the remote's structure metadata on the next push — the create decision now also checks the remote, so a fresh local file attaches to the existing remote dataset
- Fixed: attributes split-brain — re-accessing a variable could create a second, independent in-memory attrs dict, and close() flushed the stale one last, silently losing newer edits; all attribute access now shares a single dataset-level source of truth
- Fixed: deleting a variable no longer lets its attributes resurrect at close; `attrs.clear()` now persists
- Added public `Dataset.sync()`: flush variable definitions and attributes to the local file without closing (never touches a remote)
- EDataset write sessions that change nothing no longer rewrite the metadata slot at open (previously every session pushed an unchanged metadata object)
- Updated dependency pins: `booklet>=0.12.4`, `ebooklet>=0.9.0`

## 0.6.0

- Added `DatasetRechunker` class for native multivariable rechunking
- Added `rechunker()` method to `DatasetBase` to expose the new multivariable `DatasetRechunker`
- Upgraded `iter_chunks` to use `DatasetRechunker` internally, ensuring optimized synchronization and shared memory budget across variables
- Fixed prepend and append bugs
- Fixed `copy()` dataset issue
- Updated dependencies

## 0.5.0

- Added Xarray backend for `cfdb` (read-only mapping)
- Added time aggregation capabilities (e.g. `groupby(period)`)
- Fixed Geometry insertion order
- Added comprehensive benchmarks

## 0.4.3

- Added `map()` method to `DataVariableView` for parallel chunk processing using multiprocessing
- Added pickling support (`__reduce__`) to `Geometry`, `String`, and `Compressor` classes
- Fixed `Dataset.prune()` to match updated booklet API (removed `reindex` parameter)

## 0.4.2

- Renamed `time` parameter to `iter_dim` in interpolation classes (`GridInterp`, `PointInterp`) and `DataVariableView.interp()` to generalize iteration over any non-spatial dimension
- Fixed API reference docs rendering by correcting mkdocstrings parser from `google` to `numpy`

## 0.3.5

- Updated EDataset with new `num_groups` parameter
- Uses cfdb-vars for variable definitions
- Added variable templates

## Previous Releases

See the [GitHub releases](https://github.com/mullenkamp/cfdb/releases) page for full version history.
