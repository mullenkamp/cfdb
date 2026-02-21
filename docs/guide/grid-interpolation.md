# Grid Interpolation

Data variables support spatial interpolation via the [geointerp](https://github.com/mullenkamp/geointerp) package. This enables regridding, point sampling, NaN filling, and vertical level regridding directly from a data variable.

## Prerequisites

- The dataset must have a CRS defined (`ds.create.crs.from_user_input(...)`)
- Coordinates must have their axes set (x, y, and optionally z, t)

## Getting a GridInterp Object

```python
with cfdb.open_dataset(file_path) as ds:
    gi = ds['temperature'].grid_interp()
```

Coordinate names are auto-detected from axis metadata. Pass them explicitly when axes are not set:

```python
gi = ds['temperature'].grid_interp(x='longitude', y='latitude', time='time')
```

## Time Iteration

All interpolation methods are **generators** that yield `(time_value, result)` tuples:

- When there is no time dimension, a single tuple is yielded with `time_value=None`
- When a time dimension is present, the data is iterated efficiently using groupby/rechunker

## Regridding to a New Grid

Interpolate onto a new regular grid:

```python
with cfdb.open_dataset(file_path) as ds:
    for time_val, grid in ds['temperature'].grid_interp().to_grid(grid_res=0.01, to_crs=4326):
        print(time_val, grid.shape)
```

Parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `grid_res` | float or tuple | Output grid resolution |
| `to_crs` | int, str, or None | Target CRS (default: dataset CRS) |
| `bbox` | tuple or None | Bounding box in target CRS |
| `order` | int | Spline order: 0=nearest, 1=linear, 3=cubic (default) |
| `extrapolation` | str | Mode for out-of-grid values |
| `fill_val` | float | Fill value for 'constant' extrapolation |
| `min_val` | float or None | Floor clamp value |

## Sampling at Point Locations

Extract values at specific coordinates:

```python
import numpy as np

target_points = np.array([
    [175.0, -41.0],
    [172.5, -43.5],
])

with cfdb.open_dataset(file_path) as ds:
    for time_val, values in ds['temperature'].grid_interp().to_points(target_points, to_crs=4326):
        print(time_val, values)
```

Parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `target_points` | np.ndarray | Shape (M, 2) or (M, 3) array of (x, y[, z]) |
| `to_crs` | int, str, or None | CRS of target points |
| `order` | int | Spline order (0-5) |
| `min_val` | float or None | Floor clamp value |

## Filling NaN Values

Interpolate to fill NaN gaps in the spatial data:

```python
with cfdb.open_dataset(file_path) as ds:
    for time_val, filled in ds['temperature'].grid_interp().interp_na(method='linear'):
        print(time_val, filled.shape)
```

Parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `method` | str | `'nearest'`, `'linear'`, or `'cubic'` |
| `min_val` | float or None | Floor clamp value |

## Regridding Vertical Levels

For data on terrain-following coordinates where actual level heights vary at each grid point, interpolate onto fixed target levels:

```python
import numpy as np

target_levels = np.array([0, 50, 100, 200, 500])

with cfdb.open_dataset(file_path) as ds:
    for time_val, regridded in ds['temperature'].grid_interp().regrid_levels(
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
