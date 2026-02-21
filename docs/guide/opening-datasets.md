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

All data in a cfdb file is compressed at the chunk level. Two algorithms are available:

| Algorithm | Description | Default Level |
|-----------|-------------|---------------|
| `zstd` | Excellent compression ratio with good speed (default) | 1 |
| `lz4` | Fastest compression/decompression, lower ratio | 1 |

```python
# Use lz4 for speed-critical workflows
ds = cfdb.open_dataset('data.cfdb', flag='n', compression='lz4')

# Use higher zstd level for better compression
ds = cfdb.open_dataset('data.cfdb', flag='n', compression='zstd', compression_level=3)
```

Compression settings are fixed at dataset creation and apply to all variables.

## Dataset Types

The `dataset_type` parameter selects the coordinate structure:

| Type | Description |
|------|-------------|
| `'grid'` | Standard N-dimensional grid (default). Each coordinate is 1-D with a unique axis. |
| `'ts_ortho'` | Time series with point geometries ([Orthogonal multidimensional array representation](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#_orthogonal_multidimensional_array_representation_of_time_series)). |

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
