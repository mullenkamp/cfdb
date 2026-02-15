# cfdb

<p align="center">
    <em>CF conventions multi-dimensional array storage on top of Booklet</em>
</p>

[![build](https://github.com/mullenkamp/cfdb/workflows/Build/badge.svg)](https://github.com/mullenkamp/cfdb/actions)
[![codecov](https://codecov.io/gh/mullenkamp/cfdb/branch/master/graph/badge.svg)](https://codecov.io/gh/mullenkamp/cfdb)
[![PyPI version](https://badge.fury.io/py/cfdb.svg)](https://badge.fury.io/py/cfdb)

---

**Source Code**: <a href="https://github.com/mullenkamp/cfdb" target="_blank">https://github.com/mullenkamp/cfdb</a>

---
## Introduction
cfdb is a pure python database for managing labeled multi-dimensional arrays that mostly follows the [CF conventions](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html). It is an alternative to netcdf4 and [xarray](https://docs.xarray.dev/). It builds upon the [Booklet](https://github.com/mullenkamp/booklet) for the underlying local file storage and [EBooklet](https://github.com/mullenkamp/ebooklet) to sync and share on any S3 system. It has been designed to follow the programming style of opening a file, iteratively read data, iteratively write data, then closing the file.
It is thread-safe on reads and writes (using thread locks) and multiprocessing-safe (using file locks) including on the S3 remote (using object locking).

When an error occurs, cfdb will try to properly close the file and remove the file (object) locks. This will not sync any changes, so the user will lose any changes that were not synced. There will be circumstances that can occur that will not properly close the file, so care still needs to be made.


## Installation

Install via pip:

```
pip install cfdb
```

I'll probably put it on conda-forge once I feel appropriately motivated...

## Usage
### Opening a file/dataset
Usage starts off by opening the file (and closing the file when done):
```python
import cfdb
import numpy as np

file_path = '/path/to/file.cfdb'

ds = cfdb.open_dataset(file_path, flag='n')
# Do fancy stuff
ds.close()
```

By default, files will be open for read-only, so we need to specify that we want to write (in this case, 'n' is to open for write and replace the existing file with a new one). There are also some compression options, and those are described in the doc strings. Other kwargs from [Booklet](https://github.com/mullenkamp/booklet?tab=readme-ov-file#usage) can be passed to open_dataset.

The dataset can also be opened with the context manager like so:
```python
with cfdb.open_dataset(file_path, flag='n') as ds:
    print(ds)
```
This is generally encouraged as this will ensure that the file is closed properly and file locks are removed.

### Dataset types
There are currently two dataset types. Dataset types exist to structure the coordinates according to CF conventions. 

The default dataset type is "grid" which is the standard structure for coordinates. Each coordinate must be unique and increasing in ascending order (unless it's a string). Each coordinate represents a single axis (i.e. x, y, z, t). The z axis is currently optional.

The second optional dataset type is called [Orthogonal multidimensional array representation of time series](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#_orthogonal_multidimensional_array_representation_of_time_series). This is designed for time series data with sparse geometries (e.g. station time series data). The Geometry dtype must represent the xy axis. The time coordinate is the same as the "grid" time coordinate. The z axis is currently optional.

### Variables
In the [CF conventions](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#dimensions), variables are the objects that store data. These can be 1 dimensional or many dimensional. The dimensions are the labels of 1-D variables (like latitude or time). These 1-D variables are called coordinate variables (or coordinates) with the same name as their associated dimension. All variables that use these coordinates as their dimension labels are called data variables. The combination of multiple data variables with their coordinates in a single file is called a dataset.

#### Coordinates
Since all data variables must have coordinates, the coordinates must be created before data variables are created.

Coordinates in cfdb are more similar to the definition by the earlier [COARDS conventions](https://ferret.pmel.noaa.gov/Ferret/documentation/coards-netcdf-conventions) than the latter CF conventions. Coordinate values must be unique, sorted in ascending order (a partial consequence to np.sort), and cannot have null (or np.nan) values. The CF conventions do not have those limitations, but these limitations are good! Coordinates must also be only 1-D.

Coordinates can be created using the generic creation method, or templates can be used for some of the more common dimensions (like latitude, longitude, and time):
```python
lat_data = np.linspace(0, 19.9, 200, dtype='float32')

with cfdb.open_dataset(file_path, flag='n') as ds:
    lat_coord = ds.create.coord.lat(data=lat_data, chunk_shape=(20,))
    print(lat_coord)
```
When creating coordinates, the user can pass a np.ndarray as data and cfdb will figure out the rest (especially when using a creation template). Otherwise, a coordinate can be created without any data input and the data can be appended later:
```python
with cfdb.open_dataset(file_path, flag='n') as ds:
    lat_coord = ds.create.coord.lat(chunk_shape=(20,))
    lat_coord.append(lat_data)
    print(lat_coord.data)
```
Coordinate data can either be appended or prepended, but keep in mind the limitations described above! And once assigned, coordinate values cannot be changed. At some point, I'll implement the ability to shrink the size of coordinates, but for now they can only be expanded. As seen in the above example, the .data method will return the entire variable data as a single np.ndarray. Coordinates always hold the entire data in memory, while data variables never do. On disk, all data are stored as chunks, whether it's coordinates or data variables.

Let's add another coordinate for fun:
```python
time_data = np.linspace(0, 199, 200, dtype='datetime64[D]')

with cfdb.open_dataset(file_path, flag='w') as ds:
    time_coord = ds.create.coord.time(data=time_data, dtype_decoded=time_data.dtype, dtype_encoded='int32')
    print(time_coord)
```
A time variable works similarly to other numpy dtypes, but you can assign the precision of the datetime object within the brackets (shown as [D] for days). Look at the [numpy datetime reference page](https://numpy.org/doc/stable/reference/arrays.datetime.html#datetime-units) for all of the frequency codes. Do not use a frequency code finer than "ns". Encoding a datetime64 dtype to an int32 is possible down to the "m" (minute) resolution (with a max year of 6053), but all higher frequency codes should use int64.

#### Data Variables
Data variables are created in a similar way as coordinates except that you cannot pass data on creation and you must pass a tuple of the coordinate names to link the coordinates to the data variable:
```python
data_var_data = np.linspace(0, 3999.9, 40000, dtype='float64').reshape(200, 200)
name = 'data_var'
coords = ('latitude', 'time')
dtype_encoded = 'int32'
scale_factor = 0.1

with cfdb.open_dataset(file_path, flag='w') as ds:
    data_var = ds.create.data_var.generic(name, coords, data_var_data.dtype, dtype_encoded, scale_factor=scale_factor)
    data_var[:] = data_var_data
    data_var.attrs['test'] = ['test attributes']
    print(data_var)
```
There are data variable templates like the coordinates, but we will use the generic creation method for illustration. If no fillvalue or chunk_shape is passed, then cfdb figures them out for you.

Assigning data to data variables is different to coordinates. Data variables can only be expanded via the coordinates themselves. Assignment and selection is performed by the [basic numpy indexing](https://numpy.org/doc/stable/user/basics.indexing.html#basic-indexing), but not the [advanced indexing](https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing).

The example shown above is the simplest way of assigning data to a data variable, but it's not a preferred method when datasets are very large. The recommended way to write (and read) data is to iterate over the chunks:

```python
with cfdb.open_dataset(file_path, flag='w') as ds:
    data_var = ds[name]
    for chunk_slices in data_var.iter_chunks():
        data_var[chunk_slices] = data_var_data[chunk_slices]
```

This is a bit of a contrived example given that data_var_data is a single in-memory numpy array, but in many cases your data source will be much larger or in many pieces. The chunk_slices is a tuple of index slices that the data chunk covers. It is the same indexing that can be passed to a numpy ndarray.

Reading data uses the same "iter_chunks" method. This ensures that memory usage is kept to a minimum:

```python
with cfdb.open_dataset(file_path, flag='r') as ds:
    data_var = ds[name]
    for chunk_slices, data in data_var.iter_chunks(include_data=True):
        print(chunk_slices)
        print(data.shape)
```

There's a groupby method that works similarly to the iter_chunks method except that it requires one or more coordinate names (like pandas or xarray):

```python
with cfdb.open_dataset(file_path, flag='r') as ds:
    data_var = ds[name]
    for slices, data in data_var.groupby('latitude'):
        print(slices)
        print(data.shape)
```

#### Rechunking
All data for variables are stored as chunks of data. For example, the shape of your data may be 2000 x 2000, but the data are stored in 100 x 100 chunks. This is done for a variety of reasons including the ability to compress data. When a variable is created, either the user can define their own chunk shape or cfdb will determine the chunk shape automatically. 

The chunk shape defined in the variable might be good for some use cases but not others. The user might have specific use cases where they want a specific chunking; for example the groupby operation listed in the last example. In that example, the user wanted to iterate over each latitude but with all of the other coordinates (in this case the full time coordinate). A groupby operation is a common rechunking example, but the user might need chunks in many different shapes.

The [rechunkit package](https://github.com/mullenkamp/rechunkit) is used under the hood to rechunk the data in cfdb. It is exposed in cfdb via the "rechunker" method in a variable. The Rechunker class has several methods to help the user decide the chunk shape.

```python
new_chunk_shape = (41, 41)

with cfdb.open_dataset(file_path) as ds:
    data_var = ds[name]
    rechunker = data_var.rechunker()
    alt_chunk_shape = rechunker.guess_chunk_shape(2**8)
    n_chunks = rechunker.calc_n_chunks()
    print(n_chunks)
    n_reads, n_writes = rechunker.calc_n_reads_rechunker(new_chunk_shape)
    print(n_reads, n_writes)
    rechunk = rechunker.rechunk(new_chunk_shape)

    for slices, data in rechunk:
        print(slices)
        print(data.shape)
```

#### Grid Interpolation
Data variables support grid interpolation via the [geointerp](https://github.com/mullenkamp/geointerp) package. The dataset must have a CRS defined and the coordinates must have their axes set (x, y, and optionally z and t). Coordinate axes are auto-detected from the metadata, or can be passed explicitly.

All interpolation methods are generators that yield `(time_value, result)` tuples. When there is no time dimension, a single tuple is yielded with `time_value=None`. When a time dimension is present, the data is efficiently iterated using the rechunker/groupby.

##### Regridding to a new grid
```python
with cfdb.open_dataset(file_path) as ds:
    data_var = ds['temperature']
    for time_val, grid in data_var.grid_interp().to_grid(grid_res=0.01, to_crs=4326):
        print(time_val, grid.shape)
```

##### Sampling at point locations
```python
import numpy as np

target_points = np.array([
    [175.0, -41.0],
    [172.5, -43.5],
])

with cfdb.open_dataset(file_path) as ds:
    data_var = ds['temperature']
    for time_val, values in data_var.grid_interp().to_points(target_points, to_crs=4326):
        print(time_val, values)
```

##### Filling NaN values
```python
with cfdb.open_dataset(file_path) as ds:
    data_var = ds['temperature']
    for time_val, filled in data_var.grid_interp().interp_na(method='linear'):
        print(time_val, filled.shape)
```

##### Regridding vertical levels
For data on terrain-following coordinates where actual level heights vary at each grid point, `regrid_levels` interpolates onto fixed target levels. The `source_levels` parameter is the name of a data variable in the dataset that contains the actual level values (same shape as the data variable being interpolated).

```python
import numpy as np

target_levels = np.array([0, 50, 100, 200, 500])

with cfdb.open_dataset(file_path) as ds:
    data_var = ds['temperature']
    for time_val, regridded in data_var.grid_interp().regrid_levels(target_levels, source_levels='level_heights'):
        print(time_val, regridded.shape)
```

##### Explicit coordinate names
When axes are not set on the coordinates, pass the coordinate names explicitly:
```python
with cfdb.open_dataset(file_path) as ds:
    gi = ds['temperature'].grid_interp(x='longitude', y='latitude', time='time')
    for time_val, grid in gi.to_grid(grid_res=0.05):
        print(time_val, grid.shape)
```

#### Serializers
The datasets can be serialized to netcdf4 via the to_netcdf4 method. You must have the [h5netcdf package](https://h5netcdf.org/) installed for netcdf4. It can also be copied to another cfdb file.

```python
with open_dataset(file_path) as ds:
    new_ds = ds.copy(new_file_path)
    print(new_ds)
    new_ds.close()
    ds.to_netcdf4(nc_file_path)
```

## TODO - In no particular order
- Implement geospatial selections. 
    - Create three different methods on coordinates: nearest, inner, and outer. These will do the coordinate selection based on those three different options. 
- Implement units with [Pint](https://pint.readthedocs.io/en/stable/getting/overview.html) and uncertainties with [Uncertainties](https://pythonhosted.org/uncertainties/user_guide.html). Both of these packages are integrated, so I should implement them together.
- ~~Implement geospatial transformations. This kind of operation would heavily benefit from the efficient rechunking in cfdb. This will likely be it's own python package which will get integrated into cfdb.~~
- Remove dependency on h5netcdf and use h5py directly
- Implement more dataset types according to the CF conventions. Specifically, the Time Series structures. 
    - ~~The first one will be station geometries with uniform time called [Orthogonal multidimensional array representation](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#_orthogonal_multidimensional_array_representation_of_time_series).~~
    - The second will be station geometries with their own irregular time called [Indexed ragged array representation](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#_indexed_ragged_array_representation_of_time_series).

## Development

### Setup environment

We use [uv](https://docs.astral.sh/uv/) to manage the development environment and production build. 

## License

This project is licensed under the terms of the Apache Software License 2.0.
