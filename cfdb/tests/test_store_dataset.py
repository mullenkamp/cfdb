"""
cfdb.open_store_dataset: a dataset opened read-only on a caller-supplied store reads exactly what
open_dataset reads, for all four dataset types; bad stores fail clearly and are closed.

The stub store wraps a real booklet file and exposes ONLY the store contract (writable,
get_metadata, get, load_items, close, sync), recording every other attribute cfdb asks for.
"""
import numpy as np
import pytest
import shapely

import booklet
import cfdb
from cfdb import dtypes
from cfdb.main import Grid, GridForecast, TimeSeriesForecast, TimeSeriesOrtho

TIME = np.arange('2024-01-01T00', '2024-01-03T00', dtype='datetime64[h]')
FRT = np.array(['2024-01-01T00', '2024-01-01T06', '2024-01-01T12'], dtype='datetime64[m]')
LEAD = np.arange(1, 7, dtype='int32')
F32 = dtypes.dtype('float32', precision=2, min_value=-100, max_value=1000)
RNG = np.random.default_rng(11)


class StubStore:
    """The store contract over a real booklet file, and nothing more."""

    _CONTRACT = {'writable', 'get_metadata', 'get', 'load_items', 'close', 'sync'}

    def __init__(self, path, metadata=...):
        object.__setattr__(self, '_blt', booklet.open(str(path), 'r'))
        object.__setattr__(self, '_metadata', metadata)
        object.__setattr__(self, 'closed', 0)
        object.__setattr__(self, 'other', [])

    writable = False

    def get_metadata(self, include_timestamp=False):
        return self._blt.get_metadata() if self._metadata is ... else self._metadata

    def get(self, key, default=None):
        return self._blt.get(key, default)

    def load_items(self, keys=None):
        list(keys)
        return {}

    def close(self):
        object.__setattr__(self, 'closed', self.closed + 1)
        self._blt.close()

    def sync(self):
        pass

    def __getattr__(self, name):
        self.other.append(name)
        raise AttributeError(name)


def _data(shape):
    arr = np.round(RNG.uniform(0, 50, shape), 2).astype('float32')
    arr[RNG.uniform(size=shape) < 0.1] = np.nan
    return arr


def _build(path, dataset_type):
    with cfdb.open_dataset(str(path), flag='n', dataset_type=dataset_type) as ds:
        if dataset_type in ('grid', 'grid_forecast'):
            ds.create.coord.x(data=np.array([170.0, 170.5, 171.0], dtype='float32'), step=0.5)
            ds.create.coord.y(data=np.array([-44.0, -43.5], dtype='float32'), step=0.5)
        else:
            ds.create.coord.point(data=np.array([shapely.Point(172.0, -43.5), shapely.Point(171.5, -43.0)], dtype='O'))
        if dataset_type in ('grid', 'ts_ortho'):
            ds.create.coord.time(data=TIME, step=1)
        else:
            ds.create.coord.forecast_reference_time(data=FRT, step=360)
            ds.create.coord.forecast_period(data=LEAD, step=1)
        dims = {
            'grid': ('time', 'y', 'x'),
            'ts_ortho': ('point', 'time'),
            'ts_forecast': ('point', 'forecast_reference_time', 'forecast_period'),
            'grid_forecast': ('x', 'y', 'forecast_reference_time', 'forecast_period'),
        }[dataset_type]
        v = ds.create.data_var.generic('v', dims, dtype=F32)
        v[:] = _data(v.shape)
        v.attrs['units'] = 'mm'
        ds.attrs['title'] = dataset_type


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype == object:
        return a.shape == b.shape and all(x == y for x, y in zip(a.ravel(), b.ravel(), strict=True))
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind in 'fc')


@pytest.mark.parametrize('dataset_type,klass', [('grid', Grid), ('ts_ortho', TimeSeriesOrtho),
                                                ('ts_forecast', TimeSeriesForecast), ('grid_forecast', GridForecast)])
def test_reads_through_a_store_equal_open_dataset(tmp_path, dataset_type, klass):
    path = tmp_path / 'd.cfdb'
    _build(path, dataset_type)
    with cfdb.open_dataset(str(path)) as ref:
        expected = {n: ref[n].data for n in list(ref.coord_names) + list(ref.data_var_names)}
        expected_attrs = dict(ref.attrs)
    store = StubStore(path)
    with cfdb.open_store_dataset(store, name='stub') as ds:
        assert type(ds) is klass and ds.dataset_type == dataset_type
        assert dict(ds.attrs) == expected_attrs
        for n, data in expected.items():
            assert _same(ds[n].data, data), n
        if 'time' in ds.coord_names:
            key = tuple(slice(TIME[3], TIME[9]) if c == 'time' else slice(None) for c in ds['v'].coord_names)
            assert ds['v'].loc[key].data.shape[ds['v'].coord_names.index('time')] == 7
        assert 'stub' in repr(ds)
    assert store.closed == 1
    assert store.other == [], f'cfdb asked the store for more than the contract: {store.other}'


def test_a_store_without_metadata_fails_clearly_and_is_closed(tmp_path):
    path = tmp_path / 'd.cfdb'
    _build(path, 'grid')
    store = StubStore(path, metadata=None)
    with pytest.raises(ValueError, match='no cfdb dataset metadata'):
        cfdb.open_store_dataset(store)
    assert store.closed == 1


def test_an_unknown_dataset_type_fails_clearly_and_is_closed(tmp_path):
    path = tmp_path / 'd.cfdb'
    _build(path, 'grid')
    store = StubStore(path, metadata={'dataset_type': 'not_a_type'})
    with pytest.raises(TypeError, match='dataset_type must be one of'):
        cfdb.open_store_dataset(store)
    assert store.closed == 1


def test_a_writable_store_is_refused(tmp_path):
    path = tmp_path / 'd.cfdb'
    _build(path, 'grid')
    store = StubStore(path)
    object.__setattr__(store, 'writable', True)
    with pytest.raises(ValueError, match='read-only'):
        cfdb.open_store_dataset(store)
    assert store.closed == 1
