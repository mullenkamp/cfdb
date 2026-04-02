# Changelog

## 0.6.0 (Latest)

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
