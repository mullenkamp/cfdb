# Changelog

## 0.9.1 (Latest)

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
