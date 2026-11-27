# Creator

The `Creator` class provides the interface for creating coordinates, data variables, and setting the CRS. It is accessed via `ds.create` on a writable dataset.

## Usage

<!-- invisible-code-block: python
import numpy as np
from cfdb.tests.docs_support import open_latlon_grid, open_station_ts
file_path = 'creator.cfdb'
lat_data = np.linspace(-90, 90, 181, dtype='float32')
lon_data = np.linspace(-180, 180, 361, dtype='float32')
-->

```python
import cfdb
import numpy as np

with cfdb.open_dataset(file_path, flag='n') as ds:
    # Create coordinates
    lat = ds.create.coord.lat(data=lat_data)
    lon = ds.create.coord.lon(data=lon_data)

    # Create data variables
    temp = ds.create.data_var.generic('temperature', ('latitude', 'longitude'), dtype='float32')

    # Set CRS
    ds.create.crs.from_user_input(4326, x_coord='longitude', y_coord='latitude')
```

## Sub-Objects

| Attribute | Class | Description |
|-----------|-------|-------------|
| `ds.create.coord` | Coord | Coordinate creation |
| `ds.create.data_var` | DataVar | Data variable creation |
| `ds.create.crs` | CRS | CRS assignment |

## Coord

### generic(name, data=None, dtype=None, chunk_shape=None, step=False, axis=None)

Create a coordinate with full control over all parameters.

<!-- invisible-code-block: python
ds = open_latlon_grid('creator_a.cfdb')
-->

```python
coord = ds.create.coord.generic(
    'depth',
    data=np.array([0, 10, 50, 100], dtype='float32'),
    chunk_shape=(4,),
    axis='z',
)
```

### like(name, coord, copy_data=False)

Create a coordinate from an existing one's parameters.

```python
new_lat = ds.create.coord.like('new_latitude', ds['latitude'], copy_data=True)
```

### Template Methods

Template methods set standard names, dtypes, and attributes automatically. They accept `data`, `step`, and any `generic()` kwargs.

| Method | Standard Name | Default Axis |
|--------|---------------|--------------|
| `lat` | latitude | y |
| `lon` | longitude | x |
| `time` | time | t |

Additional templates are generated from [cfdb-vars](https://github.com/mullenkamp/cfdb-vars). To see all available templates:

<!-- invisible-code-block: python
ds.close()
-->

```python
with cfdb.open_dataset(file_path, flag='n') as ds:
    print(dir(ds.create.coord))
```

Template signature:

```text
ds.create.coord.lat(data=None, step=True, **kwargs) -> Coordinate
```

## DataVar

### generic(name, coords, dtype, chunk_shape=None)

Create a data variable linked to existing coordinates.

<!-- invisible-code-block: python
ds = open_latlon_grid('creator_b.cfdb')
-->

```python
temp = ds.create.data_var.generic(
    'temperature',
    ('latitude', 'longitude'),
    dtype='float32',
    chunk_shape=(20, 36),
)
```

### like(name, data_var)

Create a data variable from an existing one's parameters.

```python
new_temp = ds.create.data_var.like('temp_copy', ds['temperature'])
```

### Template Methods

Data variable templates set standard names and dtypes. They accept `coords` and any `generic()` kwargs.

| Method | Standard Name |
|--------|---------------|
| Various | Generated from cfdb-vars |

Template signature:

```text
ds.create.data_var.air_temp(coords, **kwargs) -> DataVariable   # creates 'air_temperature'
```

To see all available templates:

<!-- invisible-code-block: python
ds.close()
-->

```python
with cfdb.open_dataset(file_path, flag='n') as ds:
    print(dir(ds.create.data_var))
```

## CRS

### from_user_input(crs, x_coord=None, y_coord=None, xy_coord=None)

Set the dataset's coordinate reference system and assign spatial axes.

For grid datasets with separate x/y coordinates:

<!-- invisible-code-block: python
ds = open_latlon_grid('creator_b.cfdb', flag='w')
-->

```python
ds.create.crs.from_user_input(4326, x_coord='longitude', y_coord='latitude')
```

For ts_ortho datasets with a geometry coordinate:

<!-- invisible-code-block: python
ds.close()
ds = open_station_ts('creator_c.cfdb')
-->

```python
ds.create.crs.from_user_input(4326, xy_coord='station')
```

<!-- invisible-code-block: python
ds.close()
-->

The `crs` parameter accepts anything `pyproj.CRS.from_user_input()` understands: EPSG codes (int), WKT strings, PROJ strings, etc.
