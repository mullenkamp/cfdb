# Tools

## cfdb_to_netcdf4

::: cfdb.cfdb_to_netcdf4
    options:
      show_root_heading: true
      show_source: false
      show_root_full_path: false

## Dataset.copy

Copy a dataset to a new cfdb file. See [Dataset](dataset.md) for the full class reference.

```python
with cfdb.open_dataset('source.cfdb') as ds:
    new_ds = ds.copy('destination.cfdb')
    print(new_ds)
    new_ds.close()
```

Parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_path` | str or Path | Target file path |
| `include_data_vars` | list of str or None | Only include these data variables |
| `exclude_data_vars` | list of str or None | Exclude these data variables |

## Dataset.to_netcdf4

Export a dataset to netCDF4 format. Requires h5netcdf.

```python
with cfdb.open_dataset('source.cfdb') as ds:
    ds.to_netcdf4('output.nc', compression='gzip')
```

Parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_path` | str or Path | Output netCDF4 file path |
| `compression` | str | HDF5 compression (default: `'gzip'`) |
| `include_data_vars` | list of str or None | Only include these data variables |
| `exclude_data_vars` | list of str or None | Exclude these data variables |
| `**file_kwargs` | | Passed to `h5netcdf.File()` |

## Importing from Other Formats

For importing data from netCDF4 and other formats into cfdb, use the [cfdb-ingest](https://github.com/mullenkamp/cfdb-ingest) package.

