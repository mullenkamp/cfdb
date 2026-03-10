# Data Variables

Data variables store N-dimensional data referenced by coordinates. Unlike coordinates, data variables never hold full data in memory — data is always accessed chunk by chunk.

## Creating Data Variables

Data variables require existing coordinates:

```python
import cfdb
import numpy as np

with cfdb.open_dataset(file_path, flag='w') as ds:
    data_var = ds.create.data_var.generic(
        'temperature',
        ('latitude', 'time'),
        dtype='float32',
    )
```

### Template Methods

Like coordinates, common data variable types have template methods:

```python
with cfdb.open_dataset(file_path, flag='w') as ds:
    temp = ds.create.data_var.air_temperature(('latitude', 'longitude', 'time'))
```

Templates set standard names, dtypes, and attributes. Any parameter accepted by `generic()` can be overridden via `**kwargs`.

### Generic Creation Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Unique variable name |
| `coords` | tuple of str | Coordinate names defining the dimensions |
| `dtype` | str, np.dtype, or DataType | Data type |
| `chunk_shape` | tuple of int or None | Chunk shape (auto-estimated if None) |

### Creating from Existing

```python
with cfdb.open_dataset(file_path, flag='w') as ds:
    new_var = ds.create.data_var.like('temperature_copy', ds['temperature'])
```

## Writing Data

### Direct Assignment

The simplest way to write a full array:

```python
data = np.random.rand(200, 200).astype('float32')

with cfdb.open_dataset(file_path, flag='w') as ds:
    ds['temperature'][:] = data
```

Assignment uses numpy basic indexing (integers and slices):

```python
with cfdb.open_dataset(file_path, flag='w') as ds:
    temp = ds['temperature']
    temp[0:10, :] = data[0:10, :]      # slice assignment
    temp[5, 100] = 42.0                 # scalar assignment
```

!!! note
    Advanced indexing (fancy indexing with arrays) is currently not supported. It might be supported in the future.

### Chunk-Based Writing (Recommended)

For large datasets, iterate over chunk positions to control memory usage:

```python
with cfdb.open_dataset(file_path, flag='w') as ds:
    temp = ds['temperature']
    for chunk_slices in temp.iter_chunk_slices():
        temp[chunk_slices] = data[chunk_slices]
```

This is the recommended approach when your source data is larger than memory or comes in pieces.

## Reading Data

### Full Array

For small variables, read everything at once:

```python
with cfdb.open_dataset(file_path) as ds:
    all_data = ds['temperature'].values
```

### Chunk-Based Reading (Recommended)

For large datasets, iterate over chunks:

```python
with cfdb.open_dataset(file_path) as ds:
    temp = ds['temperature']
    for chunk_slices, chunk_data in temp.iter_chunks():
        print(chunk_slices, chunk_data.shape)
```

The `chunk_slices` tuple contains slices that can be used as numpy indexes.

You can also iterate with a different chunk shape by passing a dict of `{coord_name: int}`:

```python
with cfdb.open_dataset(file_path) as ds:
    temp = ds['temperature']
    for chunk_slices, chunk_data in temp.iter_chunks({'latitude': 50}):
        print(chunk_slices, chunk_data.shape)
```

For position-only iteration (no data loading), use `iter_chunk_slices()`:

```python
with cfdb.open_dataset(file_path) as ds:
    temp = ds['temperature']
    for chunk_slices in temp.iter_chunk_slices():
        print(chunk_slices)
```

## GroupBy

Group by one or more coordinate dimensions. This rechunks the data so each yielded array covers a single position along the grouped dimension(s) and the full extent of all other dimensions:

```python
with cfdb.open_dataset(file_path) as ds:
    for slices, data in ds['temperature'].groupby('latitude'):
        print(slices, data.shape)
```

Group by multiple coordinates:

```python
with cfdb.open_dataset(file_path) as ds:
    for slices, data in ds['temperature'].groupby(('latitude', 'time')):
        print(slices, data.shape)
```

### Time Period GroupBy

Pass a dict with period strings to group by time periods. This works on any datetime coordinate:

```python
with cfdb.open_dataset(file_path) as ds:
    temp = ds['temperature']

    # Daily groups (hourly data → 24 time steps per group)
    for slices, data in temp.groupby({'time': 'D'}):
        print(slices, data.shape)

    # Monthly groups (variable size — Jan=31, Feb=28/29, etc.)
    for slices, data in temp.groupby({'time': 'M'}):
        print(slices, data.shape)

    # Yearly groups
    for slices, data in temp.groupby({'time': 'Y'}):
        print(slices, data.shape)

    # Every 6 hours
    for slices, data in temp.groupby({'time': '6h'}):
        print(slices, data.shape)
```

**Supported period units:** `Y` (year), `M` (month), `W` (week), `D` (day), `h` (hour), `m` (minute), `s` (second), `ms`, `us`, `ns`. Prefix with a count for multiples, e.g. `'7D'`, `'3M'`, `'6h'`.

Dict values can also be integers (chunk sizes), which can be mixed with period strings on different coordinates:

```python
# Group by day on time, chunk size 50 on latitude
for slices, data in temp.groupby({'time': 'D', 'latitude': 50}):
    print(slices, data.shape)
```

**Performance:** When the period maps to a fixed number of time steps (e.g. daily on hourly data = 24 steps) and all groups are the same size, cfdb uses the efficient rechunker path. For irregular periods like monthly or yearly, it falls back to a slice-based iteration that reads each group separately.

The `max_mem` parameter controls the memory budget for the rechunking operation (default 128 MB).

## Parallel Map

The `map()` method applies a function to each chunk in parallel using multiprocessing. It yields `(target_chunk, result)` tuples as workers complete. The function receives exactly what `iter_chunks()` yields: a `target_chunk` tuple of slices and a `data` numpy array.

The function must be a top-level picklable function (not a lambda or closure).

By default (no `chunk_shape`), `map()` uses the efficient booklet.map path where workers decompress and compute directly. Pass a `chunk_shape` dict to use a pool-based approach with rechunked chunks instead.

### Transform and Write Back

```python
def scale_kelvin(target_chunk, data):
    return data + 273.15

with cfdb.open_dataset(file_path, flag='w') as ds:
    temp = ds['temperature']
    for target_chunk, result in temp.map(scale_kelvin, n_workers=4):
        temp[target_chunk] = result
```

### Aggregate

```python
def chunk_stats(target_chunk, data):
    return {'mean': float(data.mean()), 'std': float(data.std())}

with cfdb.open_dataset(file_path) as ds:
    stats = [result for _, result in ds['temperature'].map(chunk_stats)]
```

### Skip Chunks

Return `None` from your function to skip a chunk — it will not appear in the output:

```python
def only_positive_mean(target_chunk, data):
    m = data.mean()
    if m > 0:
        return m
    return None

with cfdb.open_dataset(file_path) as ds:
    positive_means = [result for _, result in ds['temperature'].map(only_positive_mean)]
```

### Map on a View

`map()` works on sliced views, processing only the selected chunks:

```python
with cfdb.open_dataset(file_path) as ds:
    view = ds['temperature'][0:50, :]
    for target_chunk, result in view.map(scale_kelvin, n_workers=4):
        print(target_chunk, result.shape)
```

## Dataset-Level Iteration

The methods above operate on a single data variable. The dataset also provides `iter_chunks`, `iter_chunk_slices`, `groupby`, and `map` that iterate over **multiple data variables in lockstep**. All data variables must share the same coordinates.

### iter_chunks

Yields `(target_chunk, var_data)` where `target_chunk` is a `{coord_name: slice}` dict and `var_data` is a `{var_name: ndarray}` dict:

```python
with cfdb.open_dataset(file_path) as ds:
    for target_chunk, var_data in ds.iter_chunks({'latitude': 25, 'longitude': 25}):
        temp_data = var_data['temperature']
        wind_data = var_data['wind_speed']
        print(target_chunk, temp_data.shape)
```

Use `data_vars` to limit which variables are included:

```python
for target_chunk, var_data in ds.iter_chunks({'latitude': 50}, data_vars=['temperature']):
    print(var_data.keys())  # {'temperature'}
```

### iter_chunk_slices

Position-only companion — no data loading:

```python
with cfdb.open_dataset(file_path) as ds:
    for chunk in ds.iter_chunk_slices({'latitude': 25, 'longitude': 25}):
        print(chunk)  # {'latitude': slice(0, 25), 'longitude': slice(0, 25)}
```

### groupby

Group by one or more coordinates across all data variables. Supports the same period string syntax as the variable-level `groupby`:

```python
with cfdb.open_dataset(file_path) as ds:
    # Group by single coordinate values
    for target_chunk, var_data in ds.groupby('latitude'):
        print(target_chunk, {k: v.shape for k, v in var_data.items()})

    # Group by time period
    for target_chunk, var_data in ds.groupby({'time': 'M'}, data_vars=['temperature']):
        print(target_chunk, {k: v.shape for k, v in var_data.items()})
```

### map

Apply a function to aligned chunks of multiple variables in parallel. The function receives `(target_chunk, var_data)` — same as `iter_chunks`:

```python
def sum_two_vars(target_chunk, var_data):
    return var_data['temperature'] + var_data['wind_speed']

with cfdb.open_dataset(file_path) as ds:
    for target_chunk, result in ds.map(sum_two_vars, {'latitude': 25}, n_workers=4):
        print(target_chunk, result.shape)
```

## Properties

```python
with cfdb.open_dataset(file_path) as ds:
    temp = ds['temperature']

    print(temp.name)          # variable name
    print(temp.shape)         # shape tuple
    print(temp.chunk_shape)   # chunk shape tuple
    print(temp.coord_names)   # coordinate names
    print(temp.dtype)         # cfdb DataType
    print(temp.ndims)         # number of dimensions
    print(temp.attrs)         # variable attributes
```

## Interpolation

Data variables support spatial interpolation via the `interp()` method. See [Interpolation](interpolation.md) for details.
