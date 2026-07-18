# Changelog

## 0.9.4 (unreleased)

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
