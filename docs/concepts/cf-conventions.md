# CF Conventions

cfdb follows the [CF (Climate and Forecast) conventions](https://cfconventions.org/) for structuring labeled multi-dimensional arrays. This page explains how cfdb maps to CF concepts and where it differs.

## Core CF Concepts

### Variables

In CF, a **variable** stores data. Variables can be:

- **Coordinate variables** — 1-D variables that label a dimension (e.g., latitude, longitude, time)
- **Data variables** — N-dimensional variables that hold the actual data (e.g., temperature, pressure)

A **dataset** is the combination of coordinates and data variables in a single file.

### Dimensions

Dimensions are the axes of the data. Each coordinate variable shares its name with its dimension. For example, a `latitude` coordinate defines the `latitude` dimension.

### Attributes

Attributes are key-value metadata on variables or the dataset. CF defines standard attributes like `units`, `long_name`, `standard_name`, and `_FillValue`.

## cfdb's Coordinate Constraints

cfdb follows the earlier [COARDS conventions](https://ferret.pmel.noaa.gov/Ferret/documentation/coards-netcdf-conventions) more closely than full CF for coordinates:

| Rule | COARDS | CF | cfdb |
|------|--------|-----|------|
| Unique values | Required | Not required | **Required** |
| Ascending order | Required | Not required | **Required** |
| No NaN values | Required | Not required | **Required** |
| 1-D only | Required | Allows auxiliary | **Required** |

These constraints enable efficient binary search for location-based indexing and guarantee deterministic chunk key mapping.

## Axis Mapping

CF defines axis attributes for coordinates. cfdb supports:

| Axis | Description | Usage |
|------|-------------|-------|
| `x` | Easting / longitude | Required for CRS and grid interpolation |
| `y` | Northing / latitude | Required for CRS and grid interpolation |
| `z` | Vertical | Optional, used for level regridding |
| `t` | Time | Optional, used for time iteration in grid interpolation |
| `xy` | Combined spatial (geometry) | Used for `ts_ortho` datasets with Point coordinates |

Axes are set via the `axis` parameter during coordinate creation or with `coord.update_axis()`.

## Dataset Types

cfdb's dataset types map to CF structures:

| cfdb Type | CF Structure |
|-----------|-------------|
| `grid` | Standard CF dimensions/coordinates |
| `ts_ortho` | [Orthogonal multidimensional array representation of time series](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#_orthogonal_multidimensional_array_representation_of_time_series) |

## CF Attributes

When using template methods (e.g., `ds.create.coord.lat()`), standard CF attributes are set automatically from [cfdb-vars](https://github.com/mullenkamp/cfdb-vars):

- `standard_name`
- `long_name`
- `units`
- `axis`

For datetime coordinates, cfdb internally tracks the encoding but does not store some redundant attributes like `calendar` in the cfdb file. These are added when exporting to netCDF4.

## NetCDF4 Compatibility

When exporting to netCDF4 via `to_netcdf4()`, cfdb:

- Writes proper CF time units (`days since 1970-01-01`, etc.)
- Adds `calendar: proleptic_gregorian`
- Adds `scale_factor` and `add_offset` for encoded variables
- Adds `_FillValue` where applicable
- Preserves all user-set attributes
