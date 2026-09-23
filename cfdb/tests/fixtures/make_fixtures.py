"""Build the file-format fixtures used by ``test_file_format.py``.

Run with the cfdb version whose files the fixtures should pin:

    uv run python cfdb/tests/fixtures/make_fixtures.py [--compression zstd_shuffle]

It writes ``v{version}_{compression}_grid.cfdb`` and ``..._ts_ortho.cfdb``, plus
``..._expected.npz`` holding every variable's decoded values AS READ BY THAT VERSION. A later
version must read the ``.cfdb`` files and reproduce the ``.npz`` values bit-for-bit
(``tobytes()`` equality, so NaN payloads and NaT count).

The content is chosen to exercise what varies the stored bytes: packed uint16 and uint32
floats, raw float64, uint8, bool, raw and packed datetime64 with NaT, str, geometry, a
prepended and an appended coordinate (nonzero origin), chunks never written, and attrs.
Keep it deterministic and small; never edit a committed fixture — add a new one.

Coordinates are appended/prepended AFTER the data is written: in cfdb 0.9.7 writing to a
multi-dimensional variable fails once a non-first coordinate has a nonzero origin (a separate
backlog bug; reads are correct), so data written first and then shifted is the portable order.
"""
import argparse
import pathlib

import numpy as np
import shapely

import cfdb
from cfdb import dtypes

HERE = pathlib.Path(__file__).parent


def build_grid(path, compression):
    with cfdb.open_dataset(path, flag='n', compression=compression, n_buckets=1009) as ds:
        t0 = np.datetime64('2024-01-01T00', 'h')
        ds.create.coord.time(data=t0 + np.arange(36), dtype='datetime64[h]', chunk_shape=(12,), step=True)
        ds.create.coord.generic('y', data=np.arange(5.0, 20.0), chunk_shape=(8,), step=1.0)
        ds.create.coord.generic('x', data=np.arange(30.0), chunk_shape=(10,), step=1.0)
        dims = ('time', 'y', 'x')
        cs = (12, 8, 10)
        rng = np.random.default_rng(20260923)
        shape = (36, 15, 30)
        smooth = (np.sin(np.linspace(0, 6, 36))[:, None, None] * 10
                  + np.linspace(0, 5, 15)[None, :, None] + np.linspace(0, 3, 30)[None, None, :])

        v = ds.create.data_var.generic('packed_u16', dims, dtypes.dtype('float32', 2, -20, 40), chunk_shape=cs)
        v[:24] = smooth[:24].astype('float32')          # time chunk 24-35 never written
        v = ds.create.data_var.generic('packed_u32', dims, dtypes.dtype('float32', 2, -1000, 50000), chunk_shape=cs)
        v[:] = (smooth * 100).astype('float32')
        v = ds.create.data_var.generic('raw_f64', dims, dtypes.dtype('float64'), chunk_shape=cs)
        a = smooth.copy()
        a[0, 0, :5] = np.nan
        v[:] = a
        v = ds.create.data_var.generic('raw_u8', dims, dtypes.dtype('uint8'), chunk_shape=cs)
        v[:] = rng.integers(0, 255, shape, dtype='uint8')
        v = ds.create.data_var.generic('flag_bool', dims, dtypes.dtype('bool'), chunk_shape=cs)
        v[:] = rng.random(shape) > 0.5
        when = np.datetime64('2020-01-01T00:00:00', 's') + (smooth * 3600).astype('int64').astype('timedelta64[s]')
        when[1, 1, :3] = np.datetime64('NaT')
        v = ds.create.data_var.generic('when_raw', dims, dtypes.dtype('datetime64[s]'), chunk_shape=cs)
        v[:] = when
        v = ds.create.data_var.generic('when_packed', dims, dtypes.dtype('datetime64[m]', None, '2019-01-01', '2021-01-01'), chunk_shape=cs)
        v[:] = when.astype('datetime64[m]')
        v = ds.create.data_var.generic('label', ('y',), dtypes.dtype('str'), chunk_shape=(8,))
        v[:] = np.array([f'row-{i}' for i in range(15)], dtype=object)
        ds['time'].append(t0 + np.arange(36, 48))      # appended coordinate: new cells never written
        ds['y'].prepend(np.arange(0.0, 5.0))           # prepended: origin -5, new cells never written
        ds.attrs['fixture'] = 'grid'
        ds['packed_u16'].attrs['units'] = 'degC'


def build_ts_ortho(path, compression):
    with cfdb.open_dataset(path, flag='n', dataset_type='ts_ortho', compression=compression, n_buckets=1009) as ds:
        ds.create.coord.point(data=np.array([shapely.Point(170.0 + i * 0.1, -43.0 - i * 0.05) for i in range(10)], dtype=object))
        t0 = np.datetime64('2024-01-01T00', 'h')
        ds.create.coord.time(data=t0 + np.arange(200), dtype='datetime64[h]', chunk_shape=(64,), step=True)
        v = ds.create.data_var.generic('flow', ('point', 'time'), dtypes.dtype('float32', 3, 0, 5000), chunk_shape=(1, 64))
        flow = (np.abs(np.sin(np.linspace(0, 20, 200)))[None, :] * np.arange(1, 11)[:, None] * 50).astype('float32')
        for i in range(0, 10, 2):                       # odd stations never written
            v[i, :] = flow[i]
        v = ds.create.data_var.generic('station_name', ('point',), dtypes.dtype('str'), chunk_shape=(10,))
        v[:] = np.array([f'station-{i}' for i in range(10)], dtype=object)
        v = ds.create.data_var.generic('ok', ('point', 'time'), dtypes.dtype('bool'), chunk_shape=(1, 64))
        v[:] = flow > 100
        ds.attrs['fixture'] = 'ts_ortho'


def expected(path):
    """Every coordinate's and data variable's decoded values, read through the public API."""
    out = {}
    with cfdb.open_dataset(path) as ds:
        for name in list(ds.coord_names) + list(ds.data_var_names):
            data = ds[name].data
            if data.dtype.kind in 'OT':  # geometry and str: a canonical text form, no pickling
                data = np.array([shapely.to_wkt(g, rounding_precision=6) if hasattr(g, 'geom_type') else str(g)
                                 for g in data.ravel()], dtype='U').reshape(data.shape)
            out[name] = np.asarray(data)
        out['__compression__'] = np.array(f'{ds.compression}-{ds.compression_level}')
        out['__attrs__'] = np.array(repr(sorted(ds.attrs.data.items())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--compression', default='zstd')
    a = ap.parse_args()
    stem = f'v{cfdb.__version__}_{a.compression}'
    for kind, builder in (('grid', build_grid), ('ts_ortho', build_ts_ortho)):
        path = HERE / f'{stem}_{kind}.cfdb'
        if path.exists():
            raise SystemExit(f'{path} exists; fixtures are never overwritten')
        builder(path, a.compression)
        np.savez_compressed(HERE / f'{stem}_{kind}_expected.npz', **expected(path))
        print(f'wrote {path.name} ({path.stat().st_size} bytes) and its expected values')


if __name__ == '__main__':
    main()
