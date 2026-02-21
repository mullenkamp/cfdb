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

For large datasets, iterate over chunks to control memory usage:

```python
with cfdb.open_dataset(file_path, flag='w') as ds:
    temp = ds['temperature']
    for chunk_slices in temp.iter_chunks():
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
    for chunk_slices, chunk_data in temp.iter_chunks(include_data=True):
        print(chunk_slices, chunk_data.shape)
```

The `chunk_slices` tuple contains slices that can be used as numpy indexes.

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

The `max_mem` parameter controls the memory budget for the rechunking operation (default 128 MB).

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

## Grid Interpolation

Data variables support spatial interpolation via the `grid_interp()` method. See [Grid Interpolation](grid-interpolation.md) for details.
