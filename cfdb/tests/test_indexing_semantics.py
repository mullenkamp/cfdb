"""
Regression tests for numpy indexing semantics (F2, 2026-07-21):
negative ints wrap, out-of-range ints raise IndexError, slice bounds clamp,
descending/fully-out-of-range slices raise loudly — applied only at raw-user-key
entry points so internal view (_sel) recomposition is untouched.
"""
import numpy as np
import pytest

import cfdb
from cfdb import dtypes

N = 40
CHUNK = (10,)


@pytest.fixture
def ds1d(tmp_path):
    path = tmp_path / 'a.cfdb'
    with cfdb.open_dataset(path, flag='n') as ds:
        ds.create.coord.generic(name='x', data=np.arange(0.0, N), chunk_shape=CHUNK, step=1.0)
        dv = ds.create.data_var.generic(name='v', coords=('x',), dtype=dtypes.dtype('float64'), chunk_shape=CHUNK)
        dv[:] = np.arange(float(N))
    with cfdb.open_dataset(path, flag='w') as ds:
        yield ds


@pytest.fixture
def ds2d_prepended(tmp_path):
    """2D var with negative origins on both dims (composition stress)."""
    path = tmp_path / 'b.cfdb'
    with cfdb.open_dataset(path, flag='n') as ds:
        ds.create.coord.generic(name='y', data=np.arange(0.0, 20), chunk_shape=(7,), step=1.0)
        ds.create.coord.generic(name='t', data=np.arange(0.0, 30), chunk_shape=(8,), step=1.0)
        dv = ds.create.data_var.generic(name='v', coords=('y', 't'), dtype=dtypes.dtype('float64'), chunk_shape=(7, 8))
        dv[:] = np.arange(600.0).reshape(20, 30)
        ds['y'].prepend(np.array([-2.0, -1.0]))
        ds['t'].prepend(np.array([-3.0, -2.0, -1.0]))
        dv = ds['v']
        dv[0:2, :] = np.full((2, 33), 1000.0)
        dv[2:, 0:3] = np.full((20, 3), 2000.0)
    with cfdb.open_dataset(path, flag='w') as ds:
        yield ds


def oracle_2d(ds):
    return ds['v'].data


###################################################
### Out-of-range and negative ints

def test_int_at_length_raises(ds1d):
    with pytest.raises(IndexError):
        ds1d['v'][N]
    with pytest.raises(IndexError):
        ds1d['v'][N + 5]


def test_negative_int_wraps(ds1d):
    assert ds1d['v'][-1].data[0] == N - 1.0
    assert ds1d['v'][-N].data[0] == 0.0
    with pytest.raises(IndexError):
        ds1d['v'][-N - 1]


def test_coordinate_negative_int_wraps(ds1d):
    assert ds1d['x'][-1].data[0] == N - 1.0
    with pytest.raises(IndexError):
        ds1d['x'][N]


###################################################
### Slice clamping

def test_overlong_slice_clamps_like_numpy(ds1d):
    oracle = np.arange(float(N))
    assert np.array_equal(ds1d['v'][35:60].data, oracle[35:60])
    assert ds1d['v'][35:60].shape == (5,)


def test_negative_slice_bounds_match_numpy(ds1d):
    oracle = np.arange(float(N))
    for key in (slice(-5, None), slice(None, -3), slice(-15, -5), slice(-100, 5)):
        got = ds1d['v'][key].data
        assert np.array_equal(got, oracle[key]), key


def test_fully_out_of_range_slice_raises(ds1d):
    with pytest.raises(ValueError):
        ds1d['v'][45:60]


def test_descending_slice_raises(ds1d):
    with pytest.raises(ValueError):
        ds1d['v'][5:2]


###################################################
### set() paths

def test_set_overlong_slice_expects_clamped_shape(ds1d):
    dv = ds1d['v']
    dv[35:60] = np.arange(5.0)          # clamped to 5 elements
    assert np.array_equal(dv[35:40].data, np.arange(5.0))
    with pytest.raises(ValueError):
        dv[35:60] = np.arange(25.0)     # no longer accepts the padded shape


def test_set_negative_index(ds1d):
    dv = ds1d['v']
    dv[-1] = np.array([777.0])
    assert dv[N - 1].data[0] == 777.0


###################################################
### Views: composition + the C2 canary

def test_view_reads_unaffected_by_normalization(ds1d):
    """C2 canary: every read of a mid-array view must work."""
    view = ds1d['v'][10:20]
    oracle = np.arange(float(N))[10:20]
    assert np.array_equal(view.data, oracle)
    for tc, d in view.iter_chunks():
        assert np.array_equal(d, oracle[tc])
    got = np.concatenate([d for _wc, d in view.rechunker().rechunk((4,))])
    assert np.array_equal(got, oracle)


def test_view_of_view_composition(ds1d):
    oracle = np.arange(float(N))
    vv = ds1d['v'][10:20][2:8]
    assert np.array_equal(vv.data, oracle[10:20][2:8])
    # negative + overlong keys are relative to the inner view
    assert np.array_equal(ds1d['v'][10:20][-3:].data, oracle[10:20][-3:])
    assert np.array_equal(ds1d['v'][10:20][5:99].data, oracle[10:20][5:99])
    with pytest.raises(IndexError):
        ds1d['v'][10:20][10]


def test_views_on_prepended_variable(ds2d_prepended):
    """Wrap/clamp must compose with negative origins and _sel."""
    oracle = oracle_2d(ds2d_prepended)      # shape (22, 33)
    dv = ds2d_prepended['v']
    assert np.array_equal(np.asarray(dv[-1].data).ravel(), oracle[-1])
    assert np.array_equal(dv[(-5, slice(None))].data.ravel(), oracle[-5])
    assert np.array_equal(dv[(slice(3, 18), slice(-10, None))].data, oracle[3:18, -10:])
    assert np.array_equal(dv[(slice(15, 99), slice(0, 5))].data, oracle[15:99, 0:5])
    view = dv[(slice(5, 15), slice(10, 30))]
    assert np.array_equal(view[(slice(-4, None), slice(15, 99))].data, oracle[5:15, 10:30][-4:, 15:99])
    with pytest.raises(IndexError):
        view[(10, 0)]


def test_set_negative_on_prepended_view(ds2d_prepended):
    dv = ds2d_prepended['v']
    view = dv[(slice(5, 15), slice(10, 30))]
    view[(-1, slice(0, 3))] = np.array([[9.0, 8.0, 7.0]])
    oracle = oracle_2d(ds2d_prepended)
    assert np.array_equal(oracle[14, 10:13], [9.0, 8.0, 7.0])


###################################################
### .loc behavior change: scalar label past the coordinate end now raises

def test_loc_scalar_past_end_raises(tmp_path):
    path = tmp_path / 'l.cfdb'
    with cfdb.open_dataset(path, flag='n') as ds:
        t = np.arange('2020-01-01', '2020-02-10', dtype='datetime64[D]')
        ds.create.coord.time(data=t, dtype='datetime64[D]', step=True)
        dv = ds.create.data_var.generic(name='v', coords=('time',), dtype=dtypes.dtype('float64'), chunk_shape=CHUNK)
        dv[:] = np.arange(float(len(t)))
        with pytest.raises(IndexError):
            dv.loc['2021-01-01']
        # in-range labels still work
        assert dv.loc['2020-01-05'].data[0] == 4.0


def test_select_out_of_range(ds1d):
    ds = ds1d
    # over-long slice clamps at dataset level too
    dsv = ds.select({'x': slice(35, 60)})
    assert dsv['v'].shape == (5,)
    with pytest.raises(IndexError):
        ds.select({'x': N})
    # negative select
    dsv2 = ds.select({'x': slice(-10, None)})
    assert np.array_equal(dsv2['v'].data, np.arange(float(N))[-10:])
