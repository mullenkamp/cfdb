# Export & Copy

cfdb provides methods to export data to netCDF4 and copy datasets to new cfdb files.

## Export to NetCDF4

Requires `h5netcdf` (`pip install cfdb[netcdf4]`).

### From a Dataset Object

```python
with cfdb.open_dataset(file_path) as ds:
    ds.to_netcdf4('output.nc')
```

Options:

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_path` | str or Path | Output netCDF4 file path |
| `compression` | str | HDF5 compression (default: `'gzip'`) |
| `include_data_vars` | list of str or None | Only include these data variables |
| `exclude_data_vars` | list of str or None | Exclude these data variables |
| `**file_kwargs` | | Passed to `h5netcdf.File()` |

### Using the Module Function

For a one-step conversion with optional selection:

```python
import cfdb

cfdb.cfdb_to_netcdf4(
    'input.cfdb',
    'output.nc',
    sel_loc={'latitude': slice(40.0, 50.0)},
    include_data_vars=['temperature'],
)
```

Parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `cfdb_path` | str or Path | Source cfdb file |
| `nc_path` | str or Path | Target netCDF4 file |
| `compression` | str | HDF5 compression (default: `'gzip'`) |
| `sel` | dict or None | Selection by coordinate indexes |
| `sel_loc` | dict or None | Selection by coordinate values |
| `include_data_vars` | list of str or None | Only include these |
| `exclude_data_vars` | list of str or None | Exclude these |

## Copy to a New cfdb File

Copy a dataset (or a subset) to a new cfdb file:

```python
with cfdb.open_dataset(file_path) as ds:
    new_ds = ds.copy('copy.cfdb')
    print(new_ds)
    new_ds.close()
```

The copy preserves coordinates, data variables, attributes, and compression settings.

Options:

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_path` | str or Path | Target file path |
| `include_data_vars` | list of str or None | Only include these data variables |
| `exclude_data_vars` | list of str or None | Exclude these data variables |

## Importing from Other Formats

For importing data from netCDF4 and other formats into cfdb, use the separate [cfdb-ingest](https://github.com/mullenkamp/cfdb-ingest) package.

