# Dataset Types

cfdb supports multiple coordinate structures through the `dataset_type` parameter: `grid`, `ts_ortho`, and the two forecast types `ts_forecast` and `grid_forecast`.

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
    lat = ds.create.coord.lat(data=np.linspace(-90, 90, 181, dtype='float32'))
    lon = ds.create.coord.lon(data=np.linspace(-180, 180, 361, dtype='float32'))
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

## Forecast types (`ts_forecast`, `grid_forecast`)

A forecast archive is a sequence of *runs*: each run is issued at some time and predicts a series of
future steps. The two forecast types replace the single `time` axis with a pair:

- **`forecast_reference_time`** — when the run was issued ("init"). Carries CF `axis='T'`.
- **`forecast_period`** — how far ahead each value looks ("lead"). Carries **no** axis.

| type | dimensions |
|---|---|
| `ts_forecast` | `(point, forecast_reference_time, forecast_period)` |
| `grid_forecast` | `(x, y, forecast_reference_time, forecast_period)` |

### Why a lead axis rather than a valid-time axis

Indexing by valid time leaves the array almost empty — each run fills only a short diagonal band of
an axis spanning the whole record (~97 % NaN for a typical 72 h product). Indexing by *lead* is
dense: every run fills every lead slot. Valid time is `init + lead`, derived arithmetic, so nothing
is lost.

The cost is that reading *by valid time* becomes a diagonal gather across the two axes, which cfdb
does not do for you — a consumer wanting a valid-time series computes `init + lead` itself.

```python
import cfdb
import numpy as np
import shapely
from cfdb import dtypes

with cfdb.open_dataset('forecast.cfdb', flag='n', dataset_type='ts_forecast') as ds:
    ds.create.coord.point(
        data=np.array([shapely.Point(172.0, -43.5), shapely.Point(171.5, -43.0)], dtype='O')
    )
    # NOTE the explicit step -- see the warning below
    ds.create.coord.forecast_reference_time(
        data=np.array(['2024-01-01T00', '2024-01-01T03'], dtype='datetime64[m]'), step=180
    )
    lead = ds.create.coord.forecast_period(data=np.arange(1, 73, dtype='int32'), step=1)
    lead.attrs['units'] = 'h'      # REQUIRED -- not defaulted, see below

    ds.create.data_var.generic(
        'precipitation',
        ('point', 'forecast_reference_time', 'forecast_period'),
        dtype=dtypes.dtype('float32', precision=1, min_value=0, max_value=1000),
        chunk_shape=(2, 1, 72),
    )

    ds.create.crs.from_user_input(4326, xy_coord='point')
```

### Two things that will bite you

**`forecast_period` is a bare integer, `units` is REQUIRED, and it is deliberately not defaulted.**
cfdb has no timedelta dtype, so adding a lead to a `datetime64` evaluates in the *datetime's*
storage unit. Against the standard `datetime64[m]` time dtype, this silently adds **minutes**:

```python
lead = ds['forecast_period'].data
frt = ds['forecast_reference_time'].data

frt[-1] + lead.max()                                          # WRONG -- adds minutes
frt[-1] + np.timedelta64(int(lead.max()),
                         ds['forecast_period'].attrs['units'])  # correct
```

There is no default unit, on purpose: a 15-minute producer who simply forgot would otherwise be
silently labelled hourly — a valid-time range four times too long — and a downstream "units must be
declared" check could never fire, because the attribute would never be absent.

**Declare an explicit `step` on `forecast_reference_time`.** It is what makes recovery of a *missed*
run possible. With a step, skipping a run auto-fills the slot and a later `merge_into` writes into
it; without one, the merge raises `NotImplementedError: In-place coordinate insertions are
unsupported` and the run can never be back-filled. `step=True` is not enough — auto-detect infers
nothing from the single-value axis that the first-ever run creates, so the problem stays hidden
until the first missed run.

### Not supported

`.interp()` raises `NotImplementedError` on both forecast types: they have two non-spatial
dimensions, which the current interpolators do not handle.


## When to Use Each Type

| Use Case | Type |
|----------|------|
| Gridded climate/weather data | `grid` |
| Satellite imagery | `grid` |
| Regular multi-dimensional arrays | `grid` |
| Weather station time series | `ts_ortho` |
| Monitoring site observations | `ts_ortho` |
| Any sparse-point time series | `ts_ortho` |
| Weather forecasts at stations/points | `ts_forecast` |
| Gridded NWP forecast output | `grid_forecast` |
