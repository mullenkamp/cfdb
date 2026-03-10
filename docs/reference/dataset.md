# Dataset

The `Dataset` class is the main object for interacting with a cfdb file. It acts as a dictionary of variables (coordinates and data variables). Created via [`open_dataset()`](open-dataset.md).

## Usage

```python
import cfdb

with cfdb.open_dataset('data.cfdb', flag='r') as ds:
    print(ds)
    temp = ds['temperature']
```

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `file_path` | pathlib.Path | Path to the cfdb file |
| `writable` | bool | Whether the dataset is open for writing |
| `is_open` | bool | Whether the dataset is currently open |
| `compression` | str | Compression algorithm (`'zstd'` or `'lz4'`) |
| `compression_level` | int | Compression level |
| `crs` | pyproj.CRS or None | Coordinate reference system |
| `attrs` | Attributes | Dataset-level attributes |
| `create` | Creator | Variable creation interface (only when writable) |
| `var_names` | tuple of str | All variable names |
| `coord_names` | tuple of str | Coordinate variable names |
| `data_var_names` | tuple of str | Data variable names |
| `coords` | tuple | All Coordinate objects |
| `data_vars` | tuple | All DataVariable objects |
| `variables` | tuple | All Variable objects |

## Dict-Like Access

```python
ds['temperature']       # Get variable by name
'temperature' in ds     # Check if variable exists
len(ds)                 # Number of variables
for name in ds:         # Iterate variable names
    print(name)
del ds['temperature']   # Delete a variable (writable only)
```

## Methods

### get(var_name)

Get a variable by name. Returns a `Coordinate` or `DataVariable`.

### select(sel)

Filter the dataset by coordinate index positions. Returns a read-only `DatasetView`.

```python
view = ds.select({'latitude': slice(10, 20), 'time': slice(0, 30)})
```

### select_loc(sel)

Filter by coordinate values. Returns a read-only `DatasetView`.

```python
view = ds.select_loc({'latitude': slice(40.0, 50.0)})
```

### copy(file_path, include_data_vars=None, exclude_data_vars=None)

Copy the dataset to a new cfdb file. Returns the new `Dataset` (caller must close it).

### to_netcdf4(file_path, compression='gzip', include_data_vars=None, exclude_data_vars=None)

Export to netCDF4 format. Requires h5netcdf.

### iter_chunks(chunk_shape, data_vars=None, max_mem=2\*\*27)

Iterate over aligned chunks of multiple data variables. Always yields `(target_chunk, var_data)` where `target_chunk` is a dict of `{coord_name: slice}` and `var_data` is a dict of `{var_name: ndarray}`.

```python
for target_chunk, var_data in ds.iter_chunks({'latitude': 25, 'longitude': 25}):
    print(target_chunk, {k: v.shape for k, v in var_data.items()})
```

### iter_chunk_slices(chunk_shape, data_vars=None)

Iterate chunk position dicts without loading data. Yields `{coord_name: slice}` dicts.

```python
for chunk in ds.iter_chunk_slices({'latitude': 25, 'longitude': 25}):
    print(chunk)
```

### groupby(coord_names, data_vars=None, max_mem=2\*\*27)

Group by one or more coordinates across all data variables. Accepts a string, list of strings, or dict. Dict values can be `int` (chunk size) or `str` (time period like `'D'`, `'M'`, `'Y'`, `'6h'`).

```python
# Group by individual coordinate values
for target_chunk, var_data in ds.groupby('latitude'):
    print(target_chunk, {k: v.shape for k, v in var_data.items()})

# Group by time period
for target_chunk, var_data in ds.groupby({'time': 'M'}, data_vars=['temperature']):
    print(target_chunk, {k: v.shape for k, v in var_data.items()})
```

### map(func, chunk_shape, data_vars=None, max_mem=2\*\*27, n_workers=None)

Apply a function to aligned chunks in parallel. The function receives `(target_chunk, var_data)` — same as `iter_chunks`.

### close()

Close the database and flush metadata to disk.

### prune(timestamp=None, reindex=False)

Prune deleted data from the file. Returns the number of removed items.

## DatasetView

Returned by `select()` and `select_loc()`. Provides the same read interface as Dataset but is read-only and scoped to the selection.

::: cfdb.main.DatasetView
    options:
      show_root_heading: true
      show_source: false
      members:
        - get
        - var_names
        - coord_names
        - data_var_names
