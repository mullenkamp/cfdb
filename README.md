# cfdb

<p align="center">
    <em>CF conventions multi-dimensional array storage on top of Booklet</em>
</p>

[![build](https://github.com/mullenkamp/cfdb/workflows/Build/badge.svg)](https://github.com/mullenkamp/cfdb/actions)
[![codecov](https://codecov.io/gh/mullenkamp/cfdb/branch/master/graph/badge.svg)](https://codecov.io/gh/mullenkamp/cfdb)
[![PyPI version](https://badge.fury.io/py/cfdb.svg)](https://badge.fury.io/py/cfdb)

---

**Documentation**: <a href="https://mullenkamp.github.io/cfdb/" target="_blank">https://mullenkamp.github.io/cfdb/</a>

**Source Code**: <a href="https://github.com/mullenkamp/cfdb" target="_blank">https://github.com/mullenkamp/cfdb</a>

---

cfdb is a pure Python database for managing labeled multi-dimensional arrays following the [CF conventions](https://cfconventions.org/). It is an alternative to netCDF4/xarray, built on [Booklet](https://github.com/mullenkamp/booklet) for local file storage and [EBooklet](https://github.com/mullenkamp/ebooklet) for S3 sync. Thread-safe and multiprocessing-safe via locks.

## Installation

```bash
pip install cfdb
```

## Quick Example

```python
import cfdb
import numpy as np

with cfdb.open_dataset('example.cfdb', flag='n') as ds:
    lat = ds.create.coord.lat(data=np.linspace(-90, 90, 180, dtype='float32'))
    lon = ds.create.coord.lon(data=np.linspace(-180, 180, 360, dtype='float32'))
    temp = ds.create.data_var.generic('temperature', ('latitude', 'longitude'), dtype='float32')
    temp[:] = np.random.rand(180, 360).astype('float32') * 40 - 10

with cfdb.open_dataset('example.cfdb') as ds:
    for slices, data in ds['temperature'].iter_chunks(include_data=True):
        print(slices, data.shape)
```

See the [full documentation](https://mullenkamp.github.io/cfdb/) for user guides, concepts, and API reference.

## License

This project is licensed under the terms of the Apache Software License 2.0.
