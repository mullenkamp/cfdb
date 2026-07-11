# Coordinates

Coordinates define the labeled dimensions of a dataset. All data variables must reference existing coordinates, so coordinates must be created first.

## Constraints

Coordinates in cfdb follow rules closer to the [COARDS conventions](https://ferret.pmel.noaa.gov/Ferret/documentation/coards-netcdf-conventions) than CF:

- Values must be **unique**
- Values must be sorted in **ascending order** (except strings)
- Values **cannot contain null/NaN**
- Coordinates are always **1-D**
- Once written, values **cannot be changed in-place** — use append, prepend, or truncate

## Template Methods

cfdb provides template methods for common coordinate types. These set standard names, dtypes, and attributes automatically:

```python
with cfdb.open_dataset(file_path, flag='n') as ds:
    lat = ds.create.coord.lat(data=lat_data, chunk_shape=(20,))
    lon = ds.create.coord.lon(data=lon_data, chunk_shape=(36,))
    time = ds.create.coord.time(data=time_data)
```

Available template methods are generated from the [cfdb-vars](https://github.com/mullenkamp/cfdb-vars) package and include common variables like `lat`, `lon`, `time`, and others.

All template methods accept:

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | np.ndarray or None | Optional initial data |
| `step` | int, float, or bool | Enforce regularity (True=auto-detect, False=no step) |
| `**kwargs` | | Any parameter accepted by `generic()` |

## Generic Creation

For coordinates not covered by templates:

```python
with cfdb.open_dataset(file_path, flag='n') as ds:
    depth = ds.create.coord.generic(
        'depth',
        data=np.array([0, 10, 50, 100, 500], dtype='float32'),
        chunk_shape=(5,),
        axis='z',
    )
```

Parameters for `generic()`:

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Unique variable name |
| `data` | np.ndarray or None | Optional initial data |
| `dtype` | str, np.dtype, DataType, or None | Data type for encoding |
| `chunk_shape` | tuple of int or None | Chunk size (auto if None) |
| `step` | int, float, or bool | Enforce regular spacing |
| `axis` | str or None | Physical axis: `'x'`, `'y'`, `'z'`, `'t'` |

## Creating Without Data

Coordinates can be created empty and populated later:

```python
with cfdb.open_dataset(file_path, flag='n') as ds:
    lat = ds.create.coord.lat(chunk_shape=(20,))
    lat.append(np.linspace(0, 19.9, 200, dtype='float32'))
```

## Append and Prepend

Coordinates can only grow — values cannot be removed or modified.

```python
with cfdb.open_dataset(file_path, flag='w') as ds:
    coord = ds['latitude']

    # Add values to the end
    coord.append(np.array([20.0, 20.1, 20.2], dtype='float32'))

    # Add values to the beginning
    coord.prepend(np.array([-0.2, -0.1], dtype='float32'))

    # Remove values from start/end (keeps [start, stop] inclusive)
    coord.truncate(start=0.0, stop=19.0)

    # For datetime coordinates, strings are accepted
    ds['time'].truncate(start='2023-02-01', stop='2023-02-28')
```

New values from append/prepend must maintain uniqueness and ascending order. When a coordinate grows, the associated data variables expand automatically (filled with NaN/fillvalue). When a coordinate is truncated, orphaned data variable chunks are deleted.

## DateTime Coordinates

DateTime coordinates use numpy's `datetime64` type. Specify the precision in brackets:

```python
time_data = np.arange('2020-01-01', '2020-07-19', dtype='datetime64[D]')

with cfdb.open_dataset(file_path, flag='n') as ds:
    time = ds.create.coord.time(data=time_data)
```

See [numpy datetime reference](https://numpy.org/doc/stable/reference/arrays.datetime.html#datetime-units) for frequency codes. Do not use a frequency finer than `'ns'`.

!!! note
    Encoding `datetime64` to `int32` works down to minute (`'m'`) resolution (max year ~6053). Higher frequencies should use `int64`.

## Geometry Coordinates

For `ts_ortho` datasets, use geometry dtypes for spatial coordinates:

```python
import shapely

points = shapely.points([175.0, 172.5], [-41.0, -43.5])

with cfdb.open_dataset(file_path, flag='n', dataset_type='ts_ortho') as ds:
    stations = ds.create.coord.generic(
        'station',
        data=points,
        dtype='point',
        chunk_shape=(10,),
        axis='xy',
    )
```

Geometry types (`point`, `linestring`, `polygon`) require a `precision` parameter for WKT rounding.

## Step (Regular Spacing)

The `step` parameter enforces regular spacing on a coordinate:

- `True` — auto-detect the step from input data
- `False` — no step enforcement (default for `generic()`)
- An int or float — explicitly set the step value

When a step is set, `append()` and `prepend()` will validate that new data maintains regular spacing. If there is a gap between the existing data and the new data, cfdb will automatically fill in the missing intermediate values — as long as the gap is a valid multiple of the step. If the gap is not a valid multiple, a `ValueError` is raised.

```python
with cfdb.open_dataset(file_path, flag='n') as ds:
    time = ds.create.coord.time(
        data=np.array(['2023-01-01', '2023-01-02', '2023-01-03'], dtype='datetime64[D]'),
    )
    # Append non-adjacent data — Jan 4 and Jan 5 are auto-filled
    time.append(np.array(['2023-01-06', '2023-01-07'], dtype='datetime64[D]'))
    # time.data is now [Jan 1, Jan 2, Jan 3, Jan 4, Jan 5, Jan 6, Jan 7]
```

The auto-filled coordinate positions have empty (NaN/fillvalue) data variable values until explicitly written. This is useful because coordinates do not support an insert operation — append and prepend with auto-fill are the only way to extend a coordinate at non-adjacent positions.

## Accessing Coordinate Data

```python
with cfdb.open_dataset(file_path) as ds:
    lat = ds['latitude']

    # Full data array (held in memory)
    print(lat.data)

    # Properties
    print(lat.shape)
    print(lat.chunk_shape)
    print(lat.dtype)
    print(lat.axis)
    print(lat.step)
```

Coordinates always hold the entire data in memory (unlike data variables).

## Creating from Existing

Copy a coordinate's structure (and optionally data) to create a new one:

```python
with cfdb.open_dataset(file_path, flag='w') as ds:
    new_lat = ds.create.coord.like('new_latitude', ds['latitude'], copy_data=True)
```
