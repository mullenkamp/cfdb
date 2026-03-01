# DataVariable

Data variables store N-dimensional data referenced by coordinates. Unlike coordinates, they never hold full data in memory — all access goes through the chunk store.

## Class Hierarchy

```
Variable (base)
  └── DataVariableView (sliced view, supports read + write)
        └── DataVariable (full variable)
```

- `DataVariable` — the full variable, returned by `ds['var_name']` or creation methods
- `DataVariableView` — a sliced subset, returned by indexing

## Usage

```python
with cfdb.open_dataset(file_path) as ds:
    temp = ds['temperature']
    print(temp.shape)
    print(temp.coord_names)
```

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | str | Variable name |
| `shape` | tuple | Shape derived from coordinate sizes |
| `chunk_shape` | tuple | Chunk shape |
| `dtype` | DataType | cfdb data type |
| `coord_names` | tuple of str | Names of linked coordinates |
| `coords` | tuple | Coordinate objects for this variable |
| `ndims` | int | Number of dimensions |
| `attrs` | Attributes | Variable attributes |
| `units` | str or None | Physical units |
| `writable` | bool | Whether the dataset is writable |
| `data` | np.ndarray | Full array (reads all chunks — use with care) |
| `values` | np.ndarray | Alias for `data` |
| `loc` | LocationIndexer | Location-based indexer |

## Reading

### Indexing

```python
temp[0, 0]          # single value
temp[10:20, :]      # slice
temp[5, 0:100]      # mixed
```

### iter_chunks(include_data=False, decoded=True)

Iterate through chunks of the variable:

```python
# Just slices (for writing)
for chunk_slices in temp.iter_chunks():
    print(chunk_slices)

# Slices and data (for reading)
for chunk_slices, data in temp.iter_chunks(include_data=True):
    print(chunk_slices, data.shape)
```

### get_chunk(sel=None, missing_none=False)

Read data from one chunk.

### items(decoded=True)

Iterate through all individual positions yielding `(index, value)` tuples.

## Writing

### Indexing Assignment

```python
temp[:] = full_array
temp[0:10, :] = partial_data
temp[5, 100] = 42.0
```

### set(sel, data, decoded=True)

Set data at index positions. The `decoded` parameter controls whether input data is in decoded or encoded form.

## GroupBy

### groupby(coord_names, max_mem=2**27)

Group by one or more coordinates. Returns a generator of `(slices, data)` tuples:

```python
for slices, data in temp.groupby('latitude'):
    print(slices, data.shape)

for slices, data in temp.groupby(('latitude', 'time')):
    print(slices, data.shape)
```

## Parallel Map

### map(func, n_workers=None)

Apply a function to each chunk in parallel using multiprocessing. Yields `(target_chunk, result)` tuples as workers complete.

The user function receives `(target_chunk, data)` — the same values yielded by `iter_chunks(include_data=True)`. It must be a top-level picklable function (not a lambda or closure).

```python
def compute_mean(target_chunk, data):
    return data.mean()

with cfdb.open_dataset(file_path) as ds:
    means = [result for _, result in ds['temperature'].map(compute_mean, n_workers=4)]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `func` | callable | `func(target_chunk, data) -> result or None`. Return `None` to skip. |
| `n_workers` | int or None | Number of worker processes. Defaults to `os.cpu_count()`. |

Works on views too:

```python
with cfdb.open_dataset(file_path) as ds:
    view = ds['temperature'][0:50, :]
    for target_chunk, result in view.map(transform, n_workers=4):
        ...
```

## Interpolation

### interp(x=None, y=None, z=None, iter_dim=None, xy=None)

Create an interpolation object for spatial interpolation. Returns `GridInterp` for grid datasets or `PointInterp` for ts_ortho datasets. Requires geointerp and a CRS.

```python
gi = temp.interp()
for time_val, grid in gi.to_grid(grid_res=0.01):
    print(grid.shape)
```

See [GridInterp / PointInterp](interp.md) for details.

## Rechunking

### rechunker()

Return a `Rechunker` for on-the-fly rechunking:

```python
rechunker = temp.rechunker()
for slices, data in rechunker.rechunk((50, 50)):
    print(data.shape)
```

See [Rechunker](rechunker.md) for details.

## Other Methods

### load()

For EDataset: pre-fetch chunks from S3. No-op for local datasets.

### update_units(units)

Update the units value. Only on writable datasets.
