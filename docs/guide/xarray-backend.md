# Xarray Backend

cfdb includes an [xarray backend engine](https://docs.xarray.dev/en/stable/internals/how-to-add-new-backend.html) that lets you open cfdb files directly with `xr.open_dataset()`. Data is loaded lazily — no array data is read until you access `.values` or perform a computation.

## Installation

Install cfdb with the xarray extra:

=== "pip"

    ```bash
    pip install cfdb[xarray]
    ```

=== "uv"

    ```bash
    uv add cfdb[xarray]
    ```

## Opening a File

After installing the extra, use `engine="cfdb"`:

```python
import xarray as xr

ds = xr.open_dataset("my_data.cfdb", engine="cfdb")
print(ds)
```

The backend maps cfdb coordinates to xarray coordinates and cfdb data variables to xarray data variables. Global attributes, variable attributes, units, and CRS (as `crs_wkt`) are all preserved.

You can also pass the backend class directly without relying on the entry point:

```python
from cfdb.xarray_backend import CfdbBackendEntrypoint

ds = xr.open_dataset("my_data.cfdb", engine=CfdbBackendEntrypoint)
```

## Lazy Loading

All data is loaded on demand. Slicing an xarray variable only reads the necessary cfdb chunks:

```python
ds = xr.open_dataset("my_data.cfdb", engine="cfdb")

# No data read yet
temp = ds["temperature"]

# Only reads chunks covering this slice
subset = temp.isel(latitude=slice(0, 10), time=0)
values = subset.values  # data read here
```

## Chunk Information

Each variable's encoding includes `preferred_chunks`, which reflects the cfdb storage chunk shape:

```python
ds = xr.open_dataset("my_data.cfdb", engine="cfdb")
print(ds["temperature"].encoding["preferred_chunks"])
# {'latitude': 50, 'longitude': 100, 'time': 10}
```

## Dropping Variables

Use `drop_variables` to skip specific variables when opening:

```python
ds = xr.open_dataset(
    "my_data.cfdb",
    engine="cfdb",
    drop_variables=["humidity"],
)
```

## Context Manager

The returned dataset supports context manager usage, which closes the underlying cfdb file:

```python
with xr.open_dataset("my_data.cfdb", engine="cfdb") as ds:
    values = ds["temperature"].values
# file is closed here
```

## Limitations

- **Read-only.** The xarray backend opens files in read mode. To create or modify cfdb files, use the native `cfdb.open_dataset()` API.
- **No dask integration.** The backend currently uses `threading.Lock` for thread safety. Dask chunked reading is not yet supported — use `cfdb`'s built-in `iter_chunks()` and `map()` for parallel chunk processing.
- **Geometry dtypes.** Point, LineString, and Polygon coordinates are exposed as object arrays of shapely geometries.
