# Interpolation

Data variables support spatial interpolation via the [geointerp](https://github.com/mullenkamp/geointerp) package. This enables regridding, point sampling, NaN filling, vertical level regridding, and CRS transformations directly from a data variable.

The `interp()` method automatically selects the right interpolation backend based on the dataset type:

- **Grid datasets** use `GridInterp` (wraps `geointerp.GridInterpolator`)
- **ts_ortho datasets** use `PointInterp` (wraps `geointerp.PointInterpolator`)

## Prerequisites

- The dataset must have a CRS defined (`ds.create.crs.from_user_input(...)`)
- Coordinates must have their axes set (x, y / xy, and optionally z, t)

## Getting an Interpolation Object

```python
with cfdb.open_dataset(file_path) as ds:
    gi = ds['temperature'].interp()
```

Coordinate names are auto-detected from axis metadata. Pass them explicitly when axes are not set:

```python
# Grid datasets
gi = ds['temperature'].interp(x='longitude', y='latitude', iter_dim='time')

# ts_ortho datasets
pi = ds['temperature'].interp(xy='point', iter_dim='time')
```

## Dimension Iteration

All interpolation methods are **generators** that yield `(dim_value, result)` tuples:

- When there is no iteration dimension, a single tuple is yielded with `dim_value=None`
- When an iteration dimension is present, the data is iterated efficiently using groupby/rechunker

## Regridding to a New Grid

Interpolate onto a new regular grid:

```python
with cfdb.open_dataset(file_path) as ds:
    for time_val, grid in ds['temperature'].interp().to_grid(grid_res=0.01, to_crs=4326):
        print(time_val, grid.shape)
```

Grid dataset parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `grid_res` | float or tuple | Output grid resolution |
| `to_crs` | int, str, or None | Target CRS (default: dataset CRS) |
| `bbox` | tuple or None | Bounding box in target CRS |
| `order` | int | Spline order: 0=nearest, 1=linear, 3=cubic (default) |
| `extrapolation` | str | Mode for out-of-grid values |
| `fill_val` | float | Fill value for 'constant' extrapolation |
| `min_val` | float or None | Floor clamp value |

ts_ortho dataset parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `grid_res` | float | Output grid resolution |
| `to_crs` | int, str, or None | Target CRS (default: dataset CRS) |
| `bbox` | tuple or None | Bounding box in target CRS |
| `method` | str | `'nearest'`, `'linear'`, or `'cubic'` |
| `extrapolation` | str | `'constant'` or `'nearest'` |
| `fill_val` | float | Fill value for 'constant' extrapolation |
| `min_val` | float or None | Floor clamp value |

## Using a DataVariable as Target

Both `to_grid` and `to_points` accept a DataVariable from another open dataset as the first argument. The method derives the target parameters (coordinates, CRS, bbox) from the destination variable — it never writes to it:

```python
with cfdb.open_dataset(source_path) as src, cfdb.open_dataset(target_path) as dst:
    # Interpolate grid data onto another grid's coordinates
    for time_val, grid in src['temperature'].interp().to_grid(dst['temperature']):
        print(time_val, grid.shape)

    # Interpolate grid data onto point locations from a ts_ortho dataset
    for time_val, values in src['temperature'].interp().to_points(dst['temperature']):
        print(time_val, values.shape)
```

## CRS Transformations

Regrid data from one coordinate reference system to another using the `to_crs` parameter. For example, reproject from NZGD2000 (EPSG:2193) to WGS 84 (EPSG:4326):

```python
with cfdb.open_dataset(file_path) as ds:
    for time_val, grid in ds['temperature'].interp().to_grid(grid_res=0.01, to_crs=4326):
        print(time_val, grid.shape)
```

The source CRS is read from the dataset, and the output grid is produced in the target CRS. This works for both `to_grid` (regrid onto a new grid in the target CRS) and `to_points` (sample at points given in the target CRS).

## Sampling at Point Locations

Extract values at specific coordinates:

```python
import numpy as np

target_points = np.array([
    [175.0, -41.0],
    [172.5, -43.5],
])

with cfdb.open_dataset(file_path) as ds:
    for time_val, values in ds['temperature'].interp().to_points(target_points, to_crs=4326):
        print(time_val, values)
```

`target_points` can be a numpy array, a list of shapely Point geometries, or a DataVariable from a ts_ortho dataset.

Grid dataset parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `target_points` | np.ndarray, shapely Points, or DataVariable | Target point locations |
| `to_crs` | int, str, or None | CRS of target points |
| `order` | int | Spline order (0-5) |
| `min_val` | float or None | Floor clamp value |

ts_ortho dataset parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `target_points` | np.ndarray, shapely Points, or DataVariable | Target point locations |
| `to_crs` | int, str, or None | CRS of target points |
| `method` | str | `'nearest'`, `'linear'`, or `'cubic'` |
| `min_val` | float or None | Floor clamp value |

## Filling NaN Values

Interpolate to fill NaN gaps in the spatial data (grid datasets only):

```python
with cfdb.open_dataset(file_path) as ds:
    for time_val, filled in ds['temperature'].interp().interp_na(method='linear'):
        print(time_val, filled.shape)
```

Parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `method` | str | `'nearest'`, `'linear'`, or `'cubic'` |
| `min_val` | float or None | Floor clamp value |

## Regridding Vertical Levels

For data on terrain-following coordinates where actual level heights vary at each grid point, interpolate onto fixed target levels (grid datasets only):

```python
import numpy as np

target_levels = np.array([0, 50, 100, 200, 500])

with cfdb.open_dataset(file_path) as ds:
    for time_val, regridded in ds['temperature'].interp().regrid_levels(
        target_levels, source_levels='level_heights'
    ):
        print(time_val, regridded.shape)
```

The `source_levels` parameter must be the **name** of a data variable in the dataset that contains the actual level values (same shape as the data variable being interpolated).

Parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `target_levels` | array-like | Target level values (monotonically increasing) |
| `source_levels` | str | Name of a data variable with source level values |
| `axis` | int | Vertical axis index in the data (default: 0) |
| `method` | str | Currently only `'linear'` |
