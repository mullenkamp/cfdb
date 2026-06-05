## cfdb Library Reference

cfdb is a pure Python database for labeled multi-dimensional arrays following CF conventions. It is installed as a pip/uv package. Use this reference when writing code that imports or interacts with cfdb.

### Installation

```bash
pip install cfdb
# or
uv add cfdb
```

### Imports

```python
from cfdb import open_dataset, cfdb_to_netcdf4, dtypes
# For S3-backed datasets:
from cfdb import open_edataset
```

### Opening / Creating Datasets

```python
# Open existing (read-only)
ds = open_dataset('data.cfdb')

# Open existing (read-write)
ds = open_dataset('data.cfdb', flag='w')

# Create new (overwrites if exists)
ds = open_dataset('data.cfdb', flag='n')

# Create if doesn't exist, else open for read-write
ds = open_dataset('data.cfdb', flag='c')

# With compression options (defaults: zstd, level 1)
ds = open_dataset('data.cfdb', flag='n', compression='zstd', compression_level=1)
ds = open_dataset('data.cfdb', flag='n', compression='lz4')

# Dataset types: 'grid' (default) or 'ts_ortho' (time series with point geometries)
ds = open_dataset('data.cfdb', flag='n', dataset_type='ts_ortho')
```

Always use as a context manager:

```python
with open_dataset('data.cfdb', flag='n') as ds:
    # ... work with ds ...
    pass
# file is automatically closed
```

### Creating Coordinates

Coordinates must be created before data variables. Values must be unique and ascending. Use the named templates if available.

```python
import numpy as np

with open_dataset('data.cfdb', flag='n') as ds:
    # Named convenience methods (auto-assigns standard CF attrs):
    ds.create.coord.lat(data=lat_array, chunk_shape=(20,))
    ds.create.coord.lon(data=lon_array, chunk_shape=(20,))
    ds.create.coord.time(data=time_array, dtype=time_array.dtype)
    ds.create.coord.height()      # empty coord, append data later
    ds.create.coord.altitude()
    ds.create.coord.depth()

    # For ts_ortho datasets, use geometry coordinates:
    ds.create.coord.point()  # then append shapely Point objects

    # Generic method (full control):
    ds.create.coord.generic(
        name='my_coord',
        data=np.array([1.0, 2.0, 3.0]),  # optional
        dtype='float32',                   # str, np.dtype, or cfdb DataType
        chunk_shape=(100,),                # optional, auto-estimated if None
        step=True,                         # True=auto-detect, False=no step, or int/float
        axis='x',                          # 'x', 'y', 'z', 't', or None
    )

    # Create coordinate from existing one:
    ds.create.coord.like('new_name', existing_coord, copy_data=True)
```

**step parameter**: If coordinate values are regularly spaced (e.g. hourly), set `step=True` (auto-detect from data) or pass the explicit step value. This enforces regularity on future appends.

### Creating Data Variables

Data variables reference existing coordinates by name. Use the named templates if available.

```python
with open_dataset('data.cfdb', flag='w') as ds:
    # Generic method:
    dv = ds.create.data_var.generic(
        name='temperature',
        coords=('latitude', 'longitude', 'time'),  # must exist already
        dtype=dtypes.dtype('float32', precision=1, min_value=0, max_value=10000),
        chunk_shape=(20, 30, 10),  # optional
    )

    # Named convenience methods (auto-assigns standard CF attrs):
    dv = ds.create.data_var.air_temperature(('latitude', 'longitude', 'time'))

    # Create from existing:
    dv2 = ds.create.data_var.like('temp_copy', existing_data_var)
```

### Data Types (dtypes)

cfdb has its own type system for encoding/serialization. Create via `dtypes.dtype()`:

```python
from cfdb import dtypes

# Simple (no encoding, stored at full precision):
dt = dtypes.dtype('float32')
dt = dtypes.dtype('float64')
dt = dtypes.dtype('int32')
dt = dtypes.dtype('datetime64[D]')
dt = dtypes.dtype(np.dtype('float32'))

# Float with int encoding (smaller storage):
# precision=1 means 1 decimal place, min/max define the range
dt = dtypes.dtype('float32', precision=1, min_value=0, max_value=10000)

# Geometry types (require precision for WKT rounding):
dt = dtypes.dtype('point', precision=6)
dt = dtypes.dtype('polygon', precision=6)
dt = dtypes.dtype('linestring', precision=6)

# String type:
dt = dtypes.dtype('str')

# Pass an existing DataType through unchanged:
dt = dtypes.dtype(existing_dtype)
```

### Reading Data

```python
with open_dataset('data.cfdb') as ds:
    # Access variables by name:
    temp = ds['temperature']      # returns DataVariable
    lat = ds['latitude']          # returns Coordinate

    # Get all data as numpy array:
    data = temp.data              # loads full array into memory
    data = temp.values            # alias for .data
    arr = np.array(temp)          # numpy interop

    # Coordinate data is cached in memory:
    lat_values = lat.data         # numpy array

    # Variable metadata:
    temp.shape                    # tuple of ints
    temp.chunk_shape              # storage chunk shape
    temp.coord_names              # ('latitude', 'longitude', 'time')
    temp.dtype                    # cfdb DataType object
    temp.name                     # 'temperature'
    temp.ndims                    # number of dimensions

    # Dataset metadata:
    ds.var_names                  # all variable names
    ds.coord_names                # coordinate names only
    ds.data_var_names             # data variable names only
    ds.coords                     # tuple of Coordinate objects
    ds.data_vars                  # tuple of DataVariable objects
    ds.compression                # 'zstd' or 'lz4'
    ds.crs                        # pyproj.CRS or None
```

### Indexing and Selection

```python
with open_dataset('data.cfdb') as ds:
    temp = ds['temperature']

    # Index-based selection (returns a DataVariableView):
    view = temp[slice(0, 10), :, slice(2, 5)]
    view = temp[(slice(0, 10), slice(None), slice(2, 5))]
    data = view.data

    # Location-based selection via .loc (uses coordinate values):
    view = temp.loc[slice(0.4, 0.7), :, slice('1970-01-04', '1970-01-10')]

    # Dataset-level selection (returns DatasetView):
    ds_view = ds.select({'latitude': slice(1, 4), 'time': slice(2, 5)})
    ds_view = ds.select_loc({'latitude': slice(0.4, 0.7), 'time': slice('1970-01-04', '1970-01-10')})

    # Access variables from the view:
    temp_subset = ds_view['temperature']
```

### Writing Data

```python
with open_dataset('data.cfdb', flag='w') as ds:
    temp = ds['temperature']

    # Write all data at once:
    temp[:] = data_array

    # Write a slice:
    temp[slice(0, 10), :, slice(0, 5)] = partial_data

    # Explicit set method:
    temp.set((slice(0, 10), slice(None), slice(0, 5)), partial_data)

    # Append/prepend coordinate data:
    lat = ds['latitude']
    lat.append(new_lat_values)
    lat.prepend(earlier_lat_values)

    # Truncate coordinate (keep only values in [start, stop], inclusive):
    ds['time'].truncate(start='2023-02-01', stop='2023-02-28')
    ds['latitude'].truncate(start=-45.0)       # keep from -45 onward
    ds['latitude'].truncate(stop=0.0)          # keep up to 0
```

### Attributes

JSON-serializable key-value metadata on datasets and variables:

```python
with open_dataset('data.cfdb', flag='w') as ds:
    # Dataset-level attrs:
    ds.attrs['history'] = 'Created by script'
    ds.attrs.update({'source': 'model output', 'version': 2})

    # Variable-level attrs:
    temp = ds['temperature']
    temp.attrs['units'] = 'K'
    temp.attrs['long_name'] = 'Air Temperature'

    # Read attrs:
    units = temp.attrs['units']
    all_attrs = temp.attrs.data  # returns dict copy
```

### CRS (Coordinate Reference System)

```python
with open_dataset('data.cfdb', flag='w') as ds:
    # Grid dataset (separate x/y coords):
    ds.create.crs.from_user_input(4326, x_coord='longitude', y_coord='latitude')

    # ts_ortho dataset (geometry coord for xy):
    ds.create.crs.from_user_input(4326, xy_coord='point')

    # Access CRS:
    crs = ds.crs  # pyproj.CRS object or None
```

### Iterating Over Chunks

For large datasets, iterate rather than loading everything into memory:

```python
with open_dataset('data.cfdb') as ds:
    temp = ds['temperature']

    # Iterate with storage chunk shape:
    for chunk_slices, data in temp.iter_chunks():
        process(data)

    # Iterate with custom chunk shape:
    for chunk_slices, data in temp.iter_chunks({'latitude': 50, 'time': 5}):
        process(data)

    # Iterate chunk positions only (no data loading):
    for chunk_slices in temp.iter_chunk_slices():
        print(chunk_slices)

    # Dataset-level iteration over multiple aligned data vars:
    for target_chunk, var_data in ds.iter_chunks({'latitude': 50}, data_vars=['temperature']):
        # target_chunk: {coord_name: slice}
        # var_data: {var_name: ndarray}
        process(var_data['temperature'])
```

### GroupBy

Group data by coordinate values or time periods:

```python
with open_dataset('data.cfdb') as ds:
    temp = ds['temperature']

    # Group by each value along a coordinate (chunk_size=1):
    for slices, data in temp.groupby('latitude'):
        process(data)

    # Group by multiple coordinates:
    for slices, data in temp.groupby(['latitude', 'time']):
        process(data)

    # Group by time periods (only on datetime coords):
    for slices, data in temp.groupby({'time': 'D'}):    # daily
        process(data)
    for slices, data in temp.groupby({'time': 'M'}):    # monthly
        process(data)
    for slices, data in temp.groupby({'time': 'Y'}):    # yearly
        process(data)
    for slices, data in temp.groupby({'time': '7D'}):   # 7-day
        process(data)
    for slices, data in temp.groupby({'time': '6h'}):   # 6-hourly
        process(data)

    # Dataset-level groupby:
    for target_chunk, var_data in ds.groupby({'time': 'M'}, data_vars=['temperature']):
        monthly_temp = var_data['temperature']
```

### Rechunking

Change the chunk layout of a variable:

```python
with open_dataset('data.cfdb', flag='w') as ds:
    temp = ds['temperature']
    rechunker = temp.rechunker()

    # Estimate optimal chunk shape for a target size in bytes:
    new_shape = rechunker.guess_chunk_shape(target_chunk_size=2**20)

    # Rechunk and write to a new variable:
    new_var = ds.create.data_var.generic('temp_rechunked', temp.coord_names, dtype=temp.dtype, chunk_shape=new_shape)
    for write_chunk, data in rechunker.rechunk(new_shape):
        new_var.set(write_chunk, data)
```

### Parallel Map

Apply a function to chunks in parallel:

```python
def compute(target_chunk, var_data):
    return var_data['temperature'].mean()

with open_dataset('data.cfdb') as ds:
    # Dataset-level map:
    for target_chunk, result in ds.map(compute, {'latitude': 50}, data_vars=['temperature']):
        print(target_chunk, result)

    # Variable-level map:
    temp = ds['temperature']
    for chunk_slices, result in temp.map(my_func, chunk_shape={'latitude': 50}):
        print(result)
```

### Merging and Combining

Combine multiple dataset files together or merge data into an existing dataset:

```python
from cfdb import combine, merge_into

# Out-of-place combine (creates a new file with the coordinate union):
# overlap can be 'last' (default), 'first', or 'error'
new_ds = combine(['file1.cfdb', 'file2.cfdb'], 'combined.cfdb', overlap='last')

# In-place merge (destructively modifies the target file):
# Much faster for appending/prepending time steps (O(B) vs O(A+B)).
# By default, enforces strict spatial bounds (allow_expansion=True allows appending time).
merge_into(['new_data.cfdb'], 'existing_target.cfdb', allow_expansion=['time'], overlap='last')
```

### Copying and Converting

```python
with open_dataset('source.cfdb') as ds:
    # Copy to a new cfdb file:
    new_ds = ds.copy('dest.cfdb')
    new_ds.close()

    # Export to netCDF4 (requires h5netcdf):
    ds.to_netcdf4('output.nc')

# Standalone export function:
from cfdb import cfdb_to_netcdf4
cfdb_to_netcdf4('source.cfdb', 'output.nc', sel_loc={'time': slice('2020-01-01', '2020-12-31')})

# Importing netCDF4 (and other formats) into cfdb lives in the separate cfdb-ingest package:
# https://github.com/mullenkamp/cfdb-ingest
```

### Deleting Variables

```python
with open_dataset('data.cfdb', flag='w') as ds:
    del ds['temperature']   # delete a data variable
    del ds['altitude']      # delete a coordinate (only if no data vars reference it)
```

### Pruning

Remove deleted data from the file to reclaim space:

```python
with open_dataset('data.cfdb', flag='w') as ds:
    removed = ds.prune()  # returns count of removed items
```

### Interpolation

Requires the `geointerp` package and a CRS set on the dataset:

```python
with open_dataset('data.cfdb') as ds:
    temp = ds['temperature']
    interp_obj = temp.interp()  # auto-detects x/y from CRS axis metadata
    # For grid: returns GridInterp
    # For ts_ortho: returns PointInterp
```

### Key Rules

- Coordinate values must be **unique and ascending** (sorted).
- Coordinates support **append**, **prepend**, and **truncate** (remove values from start/end). Values cannot be modified in place.
- Data variables support `__setitem__` for writing data to any position.
- Always use context managers (`with`) to ensure proper cleanup.
- `DataVariable.data` loads the **entire array** into memory -- use `iter_chunks()` for large data.
- Thread-safe and multiprocessing-safe via locks.
