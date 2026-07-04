# Changelog

## 0.9.0 (Latest)

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
