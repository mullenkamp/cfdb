# Quick Start

This guide walks through a complete cfdb workflow: creating a dataset, adding coordinates and data variables, writing data, reading data back, and exporting.

## Create a Dataset

```python
import cfdb
import numpy as np

file_path = 'quickstart.cfdb'

ds = cfdb.open_dataset(file_path, flag='n')
```

The `flag='n'` creates a new empty file (replacing any existing one). Always close the dataset when done, or use a context manager:

```python
with cfdb.open_dataset(file_path, flag='n') as ds:
    # work with ds
    pass
```

## Create Coordinates

Coordinates must be created before data variables. Use template methods for common dimensions:

```python
with cfdb.open_dataset(file_path, flag='n') as ds:
    # Latitude with template method
    lat_data = np.linspace(0, 19.9, 200, dtype='float32')
    lat = ds.create.coord.lat(data=lat_data, chunk_shape=(20,))

    # Time with template method
    time_data = np.arange('2020-01-01', '2020-07-19', dtype='datetime64[D]')
    time = ds.create.coord.time(data=time_data, dtype_decoded=time_data.dtype)

    print(lat)
    print(time)
```

Coordinate data must be unique and sorted in ascending order. Once written, values cannot be changed — only appended or prepended.

## Create a Data Variable

Data variables are linked to one or more coordinates by name:

```python
with cfdb.open_dataset(file_path, flag='w') as ds:
    data_var = ds.create.data_var.generic(
        'temperature',
        ('latitude', 'time'),
        dtype='float32',
    )
    print(data_var)
```

## Write Data

The simplest way to write data:

```python
data = np.random.rand(200, 200).astype('float32') * 40

with cfdb.open_dataset(file_path, flag='w') as ds:
    ds['temperature'][:] = data
```

For large datasets, iterate over chunks:

```python
with cfdb.open_dataset(file_path, flag='w') as ds:
    temp = ds['temperature']
    for chunk_slices in temp.iter_chunks():
        temp[chunk_slices] = data[chunk_slices]
```

## Read Data

Read the full variable (only for small datasets):

```python
with cfdb.open_dataset(file_path) as ds:
    all_data = ds['temperature'].values
    print(all_data.shape)
```

For large datasets, iterate over chunks:

```python
with cfdb.open_dataset(file_path) as ds:
    temp = ds['temperature']
    for chunk_slices, chunk_data in temp.iter_chunks(include_data=True):
        print(chunk_slices, chunk_data.shape)
```

## Group By

Iterate by one coordinate dimension:

```python
with cfdb.open_dataset(file_path) as ds:
    for slices, data in ds['temperature'].groupby('latitude'):
        print(slices, data.shape)
```

## Attributes

Attach JSON-serializable metadata to variables or the dataset:

```python
with cfdb.open_dataset(file_path, flag='w') as ds:
    ds.attrs['title'] = 'Quick start example'
    ds['temperature'].attrs['units'] = 'degC'
```

## Export to NetCDF4

Requires `h5netcdf` (`pip install cfdb[netcdf4]`):

```python
with cfdb.open_dataset(file_path) as ds:
    ds.to_netcdf4('quickstart.nc')
```

## Next Steps

- [Opening Datasets](../guide/opening-datasets.md) — flags, compression, context managers
- [Coordinates](../guide/coordinates.md) — templates, append/prepend, datetime handling
- [Data Variables](../guide/data-variables.md) — writing, reading, groupby
- [Rechunking](../guide/rechunking.md) — flexible chunk access patterns
