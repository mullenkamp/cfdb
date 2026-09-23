# Opening Datasets

All work in cfdb starts with `open_dataset()`, which returns a `Dataset` object backed by a [Booklet](https://github.com/mullenkamp/booklet) key-value store.

## Basic Usage

```python
import cfdb

ds = cfdb.open_dataset('data.cfdb', flag='r')
# ... use ds ...
ds.close()
```

Use a context manager to ensure proper cleanup:

```python
with cfdb.open_dataset('data.cfdb', flag='r') as ds:
    print(ds)
```

## Flags

The `flag` parameter controls how the file is opened:

| Flag | Meaning |
|------|---------|
| `'r'` | Open existing database for reading only (default) |
| `'w'` | Open existing database for reading and writing |
| `'c'` | Open for reading and writing, creating it if it doesn't exist |
| `'n'` | Always create a new empty database, open for reading and writing |

## Compression

All data in a cfdb file is compressed at the chunk level. Four options are available:

| `compression` | Description | Default Level |
|-----------|-------------|---------------|
| `zstd_shuffle` | Byte-shuffled zstd: the smallest files, fast both ways (default) | 1 |
| `zstd` | Plain zstd; readable by cfdb < 0.10 | 1 |
| `lz4_shuffle` | Byte-shuffled lz4: smaller than `lz4`, but decodes slower than it | 1 |
| `lz4` | Fastest decompression, largest files | 1 |

```python
# Default: zstd_shuffle, level 1
ds = cfdb.open_dataset('data.cfdb', flag='n')

# Plain zstd, e.g. for a file that users of cfdb < 0.10 must read
ds = cfdb.open_dataset('data.cfdb', flag='n', compression='zstd')

# lz4 for the fastest decompression
ds = cfdb.open_dataset('data.cfdb', flag='n', compression='lz4')
```

Compression settings are fixed at dataset creation, recorded in the file, and apply to all
variables. Opening an existing file (or attaching to an existing remote with `open_edataset`)
uses the recorded settings, whatever is passed. Files using a `*_shuffle` option need cfdb >= 0.10;
older versions refuse to open them. See [Chunking & Storage](../concepts/chunking-storage.md#compression)
for how the shuffle works and when plain `zstd` can be smaller, and
[Compression Benchmarks](../concepts/compression-benchmarks.md) for the measurements behind the defaults.

## Dataset Types

The `dataset_type` parameter selects the coordinate structure:

| Type | Description |
|------|-------------|
| `'grid'` | Standard N-dimensional grid (default). Each coordinate is 1-D with a unique axis. |
| `'ts_ortho'` | Time series with point geometries ([Orthogonal multidimensional array representation](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#_orthogonal_multidimensional_array_representation_of_time_series)). |
| `'ts_forecast'` | Forecasts at point geometries: `(point, forecast_reference_time, forecast_period)`. cfdb >= 0.9.6. |
| `'grid_forecast'` | Gridded forecasts: `(x, y, forecast_reference_time, forecast_period)`. cfdb >= 0.9.6. |

```python
ds = cfdb.open_dataset('stations.cfdb', flag='n', dataset_type='ts_ortho')
```

See [Dataset Types](dataset-types.md) for details.

## Booklet Kwargs

Additional keyword arguments are passed to `booklet.open()`. See the [Booklet documentation](https://github.com/mullenkamp/booklet) for available options.

## Closing and Error Handling

When an error occurs, cfdb will try to properly close the file and remove file locks. However, any unsaved changes will be lost. Using the context manager (`with` statement) is the safest approach.

!!! warning
    There may be edge cases where the file is not closed properly. Always use context managers for production code.
