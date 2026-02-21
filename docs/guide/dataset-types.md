# Dataset Types

cfdb supports multiple coordinate structures through the `dataset_type` parameter.

## Grid

The default type (`dataset_type='grid'`) follows the standard CF conventions for N-dimensional data:

- Each coordinate is 1-D and represents a single axis (x, y, z, or t)
- Coordinate values must be unique and in ascending order
- The z axis is optional

This is the standard structure for gridded data like climate model output, satellite imagery, or any regular multi-dimensional array.

```python
import cfdb
import numpy as np

with cfdb.open_dataset('grid.cfdb', flag='n', dataset_type='grid') as ds:
    lat = ds.create.coord.lat(data=np.linspace(-90, 90, 180, dtype='float32'))
    lon = ds.create.coord.lon(data=np.linspace(-180, 180, 360, dtype='float32'))
    time = ds.create.coord.time(
        data=np.arange('2020-01-01', '2020-01-31', dtype='datetime64[D]')
    )

    temp = ds.create.data_var.generic(
        'temperature', ('latitude', 'longitude', 'time'), dtype='float32'
    )
```

## Time Series Orthogonal (ts_ortho)

The `ts_ortho` type implements the [Orthogonal multidimensional array representation of time series](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#_orthogonal_multidimensional_array_representation_of_time_series) from the CF conventions.

This is designed for time series data at sparse point locations (e.g., weather stations, monitoring sites):

- A geometry coordinate (Point dtype) represents the xy spatial axis
- A time coordinate is the same as the grid time coordinate
- The z axis is optional

```python
import cfdb
import numpy as np
import shapely

with cfdb.open_dataset('stations.cfdb', flag='n', dataset_type='ts_ortho') as ds:
    # Station locations as Point geometries
    points = shapely.points([175.0, 172.5, 174.8], [-41.0, -43.5, -36.9])
    stations = ds.create.coord.generic(
        'station', data=points, dtype='point', chunk_shape=(3,), axis='xy'
    )

    # Time coordinate
    time = ds.create.coord.time(
        data=np.arange('2020-01-01', '2021-01-01', dtype='datetime64[D]')
    )

    # Data variable indexed by station and time
    temp = ds.create.data_var.generic(
        'temperature', ('station', 'time'), dtype='float32'
    )

    # Set CRS
    ds.create.crs.from_user_input(4326, xy_coord='station')
```

## When to Use Each Type

| Use Case | Type |
|----------|------|
| Gridded climate/weather data | `grid` |
| Satellite imagery | `grid` |
| Regular multi-dimensional arrays | `grid` |
| Weather station time series | `ts_ortho` |
| Monitoring site observations | `ts_ortho` |
| Any sparse-point time series | `ts_ortho` |
