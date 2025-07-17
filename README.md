# cfdb

<p align="center">
    <em>CF conventions multi-dimensional array storage on top of Booklet</em>
</p>

[![build](https://github.com/mullenkamp/cfdb/workflows/Build/badge.svg)](https://github.com/mullenkamp/cfdb/actions)
[![codecov](https://codecov.io/gh/mullenkamp/cfdb/branch/master/graph/badge.svg)](https://codecov.io/gh/mullenkamp/cfdb)
[![PyPI version](https://badge.fury.io/py/cfdb.svg)](https://badge.fury.io/py/cfdb)

---

**Source Code**: <a href="https://github.com/mullenkamp/cfdb" target="_blank">https://github.com/mullenkamp/cfbdb</a>

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

### Variables
In the [CF conventions](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#dimensions), variables are the objects that store data. These can be 1 dimensional or many dimensional. The dimensions are the labels of 1-D variables (like latitude or time). These 1-D variables are called coordinate variables (or coordinates) with the same name as their associated dimension. All variables that use these coordinates as their dimension labels are called data variables. The combination of multiple data variables with their coordinates in a single file is called a dataset.

#### Coordinates
Since all data variables must have coordinates, the coordinates must be created before data variables are created.

Coordinates in cfdb are more similar to the definition by the earlier [COARDS conventions](https://ferret.pmel.noaa.gov/Ferret/documentation/coards-netcdf-conventions) than the latter CF conventions. Coordinate values must be unique, sorted in ascending order (a partial consequence to np.sort), and cannot have null (or np.nan) values. The CF conventions do not have those limitations, but these limitations are good! Coordinates must also be only 1-D.

Coordinates can be created using the generic creation method, or templates can be used for some of the more common dimensions (like latitude, longitude, and time):
```python
lat_data = np.linspace(0, 19.9, 200, dtype='float32')

with cfdb.open_dataset(file_path, flag='n') as ds:
    lat_coord = ds.create.coord.latitude(data=lat_data, chunk_shape=(20,))
    print(lat_coord)
```
When creating coordinates, the user can pass a np.ndarray as data and cfdb will figure out the rest (especially when using a creation template). Otherwise, a coordinate can be created without any data input and the data can be appended later:
```python
with cfdb.open_dataset(file_path, flag='n') as ds:
    lat_coord = ds.create.coord.latitude(chunk_shape=(20,))
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
Since there are no data variable templates (yet), we need to use the generic creation method. If no fillvalue or chunk_shape is passed, then cfdb figures them out for you.

Assigning data to data variables is different to coordinates. Data variables can only be expanded via the coordinates themselves. Assignment and selection is performed by the [basic numpy indexing](https://numpy.org/doc/stable/user/basics.indexing.html#basic-indexing), but not the [advanced indexing](https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing).



## License

This project is licensed under the terms of the Apache Software License 2.0.
