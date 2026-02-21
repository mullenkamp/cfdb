# cfdb

**CF conventions multi-dimensional array storage on top of Booklet**

[![build](https://github.com/mullenkamp/cfdb/workflows/Build/badge.svg)](https://github.com/mullenkamp/cfdb/actions)
[![PyPI version](https://badge.fury.io/py/cfdb.svg)](https://badge.fury.io/py/cfdb)

---

cfdb is a pure Python database for managing labeled multi-dimensional arrays following the [CF conventions](https://cfconventions.org/). It is an alternative to netCDF4/xarray, built on [Booklet](https://github.com/mullenkamp/booklet) for local file storage and [EBooklet](https://github.com/mullenkamp/ebooklet) for S3 sync.

## Key Features

- **CF conventions** — coordinates, data variables, and attributes following the CF standard
- **Chunk-based storage** — efficient compression with zstd or lz4, chunk-level read/write
- **Thread-safe and multiprocess-safe** — thread locks and file locks for concurrent access
- **Rechunking** — on-the-fly rechunking via [rechunkit](https://github.com/mullenkamp/rechunkit) for flexible data access patterns
- **Grid interpolation** — regridding, point sampling, NaN filling, and level regridding via [geointerp](https://github.com/mullenkamp/geointerp)
- **S3 remote sync** — `EDataset` links a local file with an S3 remote via [EBooklet](https://github.com/mullenkamp/ebooklet)
- **NetCDF4 export** — convert to/from netCDF4 with [h5netcdf](https://h5netcdf.org/)

## Quick Example

```python
import cfdb
import numpy as np

file_path = 'example.cfdb'

with cfdb.open_dataset(file_path, flag='n') as ds:
    # Create coordinates
    lat = ds.create.coord.lat(data=np.linspace(-90, 90, 180, dtype='float32'))
    lon = ds.create.coord.lon(data=np.linspace(-180, 180, 360, dtype='float32'))

    # Create a data variable
    temp = ds.create.data_var.generic(
        'temperature', ('latitude', 'longitude'), dtype='float32'
    )

    # Write data
    temp[:] = np.random.rand(180, 360).astype('float32') * 40 - 10

# Read it back
with cfdb.open_dataset(file_path) as ds:
    for chunk_slices, data in ds['temperature'].iter_chunks(include_data=True):
        print(chunk_slices, data.shape)
```

## Next Steps

- [Installation](getting-started/installation.md) — install cfdb and optional extras
- [Quick Start](getting-started/quickstart.md) — complete walkthrough of a typical workflow
- [User Guide](guide/opening-datasets.md) — detailed guides for every feature
- [API Reference](reference/index.md) — full function and class reference
