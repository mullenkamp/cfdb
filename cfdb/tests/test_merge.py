import tempfile
from pathlib import Path
import numpy as np
import pytest

import cfdb
from cfdb.merge import merge_into

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)

def test_merge_into_append(temp_dir):
    p_target = temp_dir / 'target.cfdb'
    p_new = temp_dir / 'new.cfdb'

    # Target: Time 0, 1
    with cfdb.open_dataset(p_target, 'n') as ds:
        ds.create.coord.generic('time', data=np.array(['2021-01-01', '2021-01-02'], dtype='datetime64[D]'))
        ds.create.coord.generic('lat', data=np.array([10.0, 20.0]))
        ds.create.data_var.generic('temp', ('time', 'lat'), dtype='float32')
        ds['temp'][:] = np.ones((2, 2), dtype='float32')

    # New: Time 2, 3
    with cfdb.open_dataset(p_new, 'n') as ds:
        ds.create.coord.generic('time', data=np.array(['2021-01-03', '2021-01-04'], dtype='datetime64[D]'))
        ds.create.coord.generic('lat', data=np.array([10.0, 20.0]))
        ds.create.data_var.generic('temp', ('time', 'lat'), dtype='float32')
        ds['temp'][:] = np.ones((2, 2), dtype='float32') * 2

    # Merge
    res = merge_into([p_new], p_target, allow_expansion=['time'])
    res.close()

    # Validate
    with cfdb.open_dataset(p_target, 'r') as ds:
        assert len(ds['time'].data) == 4
        assert ds['time'].data[-1] == np.datetime64('2021-01-04')
        data = ds['temp'][:].data
        assert data.shape == (4, 2)
        assert data[3, 0] == 2.0


def test_merge_into_prepend(temp_dir):
    p_target = temp_dir / 'target.cfdb'
    p_new = temp_dir / 'new.cfdb'

    # Target: Time 2, 3
    with cfdb.open_dataset(p_target, 'n') as ds:
        ds.create.coord.generic('time', data=np.array(['2021-01-03', '2021-01-04'], dtype='datetime64[D]'))
        ds.create.coord.generic('lat', data=np.array([10.0, 20.0]))
        ds.create.data_var.generic('temp', ('time', 'lat'), dtype='float32')
        ds['temp'][:] = np.ones((2, 2), dtype='float32')

    # New: Time 0, 1
    with cfdb.open_dataset(p_new, 'n') as ds:
        ds.create.coord.generic('time', data=np.array(['2021-01-01', '2021-01-02'], dtype='datetime64[D]'))
        ds.create.coord.generic('lat', data=np.array([10.0, 20.0]))
        ds.create.data_var.generic('temp', ('time', 'lat'), dtype='float32')
        ds['temp'][:] = np.ones((2, 2), dtype='float32') * 2

    # Merge
    res = merge_into([p_new], p_target, allow_expansion=['time'])
    res.close()

    # Validate
    with cfdb.open_dataset(p_target, 'r') as ds:
        assert len(ds['time'].data) == 4
        assert ds['time'].data[0] == np.datetime64('2021-01-01')
        data = ds['temp'][:].data
        assert data.shape == (4, 2)
        assert data[0, 0] == 2.0


def test_merge_into_strict_expansion(temp_dir):
    p_target = temp_dir / 'target.cfdb'
    p_new = temp_dir / 'new.cfdb'

    # Target: Lat 10, 20
    with cfdb.open_dataset(p_target, 'n') as ds:
        ds.create.coord.generic('time', data=np.array(['2021-01-01'], dtype='datetime64[D]'))
        ds.create.coord.generic('lat', data=np.array([10.0, 20.0]))

    # New: Lat 10, 20, 30
    with cfdb.open_dataset(p_new, 'n') as ds:
        ds.create.coord.generic('time', data=np.array(['2021-01-02'], dtype='datetime64[D]'))
        ds.create.coord.generic('lat', data=np.array([10.0, 20.0, 30.0]))

    # Merge should fail if allow_expansion=['time']
    with pytest.raises(ValueError, match="disallowed"):
        merge_into([p_new], p_target, allow_expansion=['time'])

    # Merge should succeed if allow_expansion=True
    res = merge_into([p_new], p_target, allow_expansion=True)
    res.close()
    
    with cfdb.open_dataset(p_target, 'r') as ds:
        assert len(ds['lat'].data) == 3


def test_merge_into_insert_fails(temp_dir):
    p_target = temp_dir / 'target.cfdb'
    p_new = temp_dir / 'new.cfdb'

    # Target: Time 0, 2
    with cfdb.open_dataset(p_target, 'n') as ds:
        ds.create.coord.generic('time', data=np.array(['2021-01-01', '2021-01-03'], dtype='datetime64[D]'))
        ds.create.coord.generic('lat', data=np.array([10.0]))

    # New: Time 1
    with cfdb.open_dataset(p_new, 'n') as ds:
        ds.create.coord.generic('time', data=np.array(['2021-01-02'], dtype='datetime64[D]'))
        ds.create.coord.generic('lat', data=np.array([10.0]))

    # Merge should fail due to insert
    with pytest.raises(NotImplementedError, match="insertions are unsupported"):
        merge_into([p_new], p_target, allow_expansion=True)


def test_merge_into_coord_mismatch(temp_dir):
    """Merging a variable with different coords than the target raises ValueError."""
    p_target = temp_dir / 'target.cfdb'
    p_new = temp_dir / 'new.cfdb'

    # Target: temp has coords (time, height, lat)
    with cfdb.open_dataset(p_target, 'n') as ds:
        ds.create.coord.generic('time', data=np.array(['2021-01-01', '2021-01-02'], dtype='datetime64[D]'))
        ds.create.coord.generic('height', data=np.array([0.0]))
        ds.create.coord.generic('lat', data=np.array([10.0, 20.0]))
        ds.create.data_var.generic('temp', ('time', 'height', 'lat'), dtype='float32')
        ds['temp'][:] = np.ones((2, 1, 2), dtype='float32')

    # New: temp has coords (time, lat) — missing height dimension
    with cfdb.open_dataset(p_new, 'n') as ds:
        ds.create.coord.generic('time', data=np.array(['2021-01-03', '2021-01-04'], dtype='datetime64[D]'))
        ds.create.coord.generic('lat', data=np.array([10.0, 20.0]))
        ds.create.data_var.generic('temp', ('time', 'lat'), dtype='float32')
        ds['temp'][:] = np.ones((2, 2), dtype='float32') * 2

    with pytest.raises(ValueError, match="has coords.*in source but.*in target"):
        merge_into([p_new], p_target, allow_expansion=['time'])


def _make_overlap_datasets(temp_dir):
    """Helper: target has time [1,2,3] val=1.0, source has time [2,3,4] val=9.0."""
    p_target = temp_dir / 'target.cfdb'
    p_new = temp_dir / 'new.cfdb'
    times_tgt = np.array(['2021-01-01', '2021-01-02', '2021-01-03'], dtype='datetime64[D]')
    times_new = np.array(['2021-01-02', '2021-01-03', '2021-01-04'], dtype='datetime64[D]')
    lat = np.array([10.0, 20.0])

    with cfdb.open_dataset(p_target, 'n') as ds:
        ds.create.coord.generic('time', data=times_tgt)
        ds.create.coord.generic('lat', data=lat)
        ds.create.data_var.generic('temp', ('time', 'lat'), dtype='float32')
        ds['temp'][:] = np.ones((3, 2), dtype='float32')

    with cfdb.open_dataset(p_new, 'n') as ds:
        ds.create.coord.generic('time', data=times_new)
        ds.create.coord.generic('lat', data=lat)
        ds.create.data_var.generic('temp', ('time', 'lat'), dtype='float32')
        ds['temp'][:] = np.full((3, 2), 9.0, dtype='float32')

    return p_target, p_new


def test_merge_into_overlap_last(temp_dir):
    """overlap='last': overlapping timesteps are overwritten by source."""
    p_target, p_new = _make_overlap_datasets(temp_dir)
    res = merge_into([p_new], p_target, allow_expansion=['time'], overlap='last')
    res.close()

    with cfdb.open_dataset(p_target) as ds:
        data = ds['temp'].data
        assert data.shape == (4, 2)
        assert data[0, 0] == 1.0   # t=Jan1 kept
        assert data[1, 0] == 9.0   # t=Jan2 overwritten
        assert data[2, 0] == 9.0   # t=Jan3 overwritten
        assert data[3, 0] == 9.0   # t=Jan4 new


def test_merge_into_overlap_first(temp_dir):
    """overlap='first': overlapping timesteps keep target data."""
    p_target = temp_dir / 'target.cfdb'
    p_new = temp_dir / 'new.cfdb'
    times = np.array(['2021-01-01', '2021-01-02', '2021-01-03'], dtype='datetime64[D]')
    lat = np.array([10.0, 20.0])

    with cfdb.open_dataset(p_target, 'n') as ds:
        ds.create.coord.generic('time', data=times)
        ds.create.coord.generic('lat', data=lat)
        ds.create.data_var.generic('temp', ('time', 'lat'), dtype='float32')
        ds['temp'][:] = np.ones((3, 2), dtype='float32')

    # Source has the same times but different values
    with cfdb.open_dataset(p_new, 'n') as ds:
        ds.create.coord.generic('time', data=times)
        ds.create.coord.generic('lat', data=lat)
        ds.create.data_var.generic('temp', ('time', 'lat'), dtype='float32')
        ds['temp'][:] = np.full((3, 2), 9.0, dtype='float32')

    res = merge_into([p_new], p_target, overlap='first')
    res.close()

    with cfdb.open_dataset(p_target) as ds:
        data = ds['temp'].data
        assert data.shape == (3, 2)
        # overlap='first' keeps target data, ignores source
        assert np.allclose(data, 1.0)


def test_merge_into_overlap_error(temp_dir):
    """overlap='error': raises on overlapping timesteps."""
    p_target, p_new = _make_overlap_datasets(temp_dir)
    with pytest.raises(ValueError, match="Overlap detected"):
        merge_into([p_new], p_target, allow_expansion=['time'], overlap='error')


def test_merge_into_new_variable(temp_dir):
    """Source introduces a variable not in the target."""
    p_target = temp_dir / 'target.cfdb'
    p_new = temp_dir / 'new.cfdb'
    times = np.array(['2021-01-01', '2021-01-02'], dtype='datetime64[D]')
    lat = np.array([10.0, 20.0])

    with cfdb.open_dataset(p_target, 'n') as ds:
        ds.create.coord.generic('time', data=times)
        ds.create.coord.generic('lat', data=lat)
        ds.create.data_var.generic('temp', ('time', 'lat'), dtype='float32')
        ds['temp'][:] = np.ones((2, 2), dtype='float32')

    with cfdb.open_dataset(p_new, 'n') as ds:
        ds.create.coord.generic('time', data=times)
        ds.create.coord.generic('lat', data=lat)
        ds.create.data_var.generic('temp', ('time', 'lat'), dtype='float32')
        ds['temp'][:] = np.ones((2, 2), dtype='float32')
        dv = ds.create.data_var.generic('pressure', ('time', 'lat'), dtype='float32')
        dv.attrs['units'] = 'Pa'
        dv[:] = np.full((2, 2), 1013.0, dtype='float32')

    res = merge_into([p_new], p_target, overlap='last')
    res.close()

    with cfdb.open_dataset(p_target) as ds:
        assert 'pressure' in ds.data_var_names
        assert ds['pressure'].data.shape == (2, 2)
        assert np.allclose(ds['pressure'].data, 1013.0)
        assert ds['pressure'].attrs['units'] == 'Pa'


def test_merge_into_roundtrip_persistence(temp_dir):
    """Data survives close/reopen after both append and prepend."""
    p_target = temp_dir / 'target.cfdb'
    p_append = temp_dir / 'append.cfdb'
    p_prepend = temp_dir / 'prepend.cfdb'
    lat = np.array([10.0, 20.0])

    with cfdb.open_dataset(p_target, 'n') as ds:
        ds.create.coord.generic('time', data=np.array(['2021-01-03', '2021-01-04'], dtype='datetime64[D]'))
        ds.create.coord.generic('lat', data=lat)
        ds.create.data_var.generic('temp', ('time', 'lat'), dtype='float32')
        ds['temp'][:] = np.full((2, 2), 5.0, dtype='float32')

    with cfdb.open_dataset(p_append, 'n') as ds:
        ds.create.coord.generic('time', data=np.array(['2021-01-05', '2021-01-06'], dtype='datetime64[D]'))
        ds.create.coord.generic('lat', data=lat)
        ds.create.data_var.generic('temp', ('time', 'lat'), dtype='float32')
        ds['temp'][:] = np.full((2, 2), 9.0, dtype='float32')

    with cfdb.open_dataset(p_prepend, 'n') as ds:
        ds.create.coord.generic('time', data=np.array(['2021-01-01', '2021-01-02'], dtype='datetime64[D]'))
        ds.create.coord.generic('lat', data=lat)
        ds.create.data_var.generic('temp', ('time', 'lat'), dtype='float32')
        ds['temp'][:] = np.full((2, 2), 1.0, dtype='float32')

    # Append
    res = merge_into([p_append], p_target, allow_expansion=['time'])
    res.close()

    # Prepend
    res = merge_into([p_prepend], p_target, allow_expansion=['time'])
    res.close()

    # Reopen and verify all data
    with cfdb.open_dataset(p_target) as ds:
        times = ds['time'].data
        assert len(times) == 6
        assert times[0] == np.datetime64('2021-01-01')
        assert times[-1] == np.datetime64('2021-01-06')
        data = ds['temp'].data
        assert data.shape == (6, 2)
        assert np.allclose(data[0], 1.0)   # prepended
        assert np.allclose(data[1], 1.0)   # prepended
        assert np.allclose(data[2], 5.0)   # original
        assert np.allclose(data[3], 5.0)   # original
        assert np.allclose(data[4], 9.0)   # appended
        assert np.allclose(data[5], 9.0)   # appended


def test_merge_into_prepend_and_append(temp_dir):
    """Source spans wider than target on both sides."""
    p_target = temp_dir / 'target.cfdb'
    p_new = temp_dir / 'new.cfdb'
    lat = np.array([10.0])

    with cfdb.open_dataset(p_target, 'n') as ds:
        ds.create.coord.generic('time', data=np.array(['2021-01-03', '2021-01-04'], dtype='datetime64[D]'))
        ds.create.coord.generic('lat', data=lat)
        ds.create.data_var.generic('temp', ('time', 'lat'), dtype='float32')
        ds['temp'][:] = np.full((2, 1), 5.0, dtype='float32')

    with cfdb.open_dataset(p_new, 'n') as ds:
        ds.create.coord.generic('time', data=np.array([
            '2021-01-01', '2021-01-02', '2021-01-05', '2021-01-06'
        ], dtype='datetime64[D]'))
        ds.create.coord.generic('lat', data=lat)
        ds.create.data_var.generic('temp', ('time', 'lat'), dtype='float32')
        ds['temp'][:] = np.full((4, 1), 9.0, dtype='float32')

    res = merge_into([p_new], p_target, allow_expansion=['time'])
    res.close()

    with cfdb.open_dataset(p_target) as ds:
        times = ds['time'].data
        assert len(times) == 6
        assert times[0] == np.datetime64('2021-01-01')
        assert times[-1] == np.datetime64('2021-01-06')
        data = ds['temp'].data
        assert data.shape == (6, 1)
        assert data[0, 0] == 9.0   # prepended
        assert data[1, 0] == 9.0   # prepended
        assert data[2, 0] == 5.0   # original
        assert data[3, 0] == 5.0   # original
        assert data[4, 0] == 9.0   # appended
        assert data[5, 0] == 9.0   # appended


def test_merge_into_exact_overlap_overwrite(temp_dir):
    """Source has identical times — overlap='last' fully replaces target data."""
    p_target = temp_dir / 'target.cfdb'
    p_new = temp_dir / 'new.cfdb'
    times = np.array(['2021-01-01', '2021-01-02'], dtype='datetime64[D]')
    lat = np.array([10.0, 20.0])

    with cfdb.open_dataset(p_target, 'n') as ds:
        ds.create.coord.generic('time', data=times)
        ds.create.coord.generic('lat', data=lat)
        ds.create.data_var.generic('temp', ('time', 'lat'), dtype='float32')
        ds['temp'][:] = np.ones((2, 2), dtype='float32')

    with cfdb.open_dataset(p_new, 'n') as ds:
        ds.create.coord.generic('time', data=times)
        ds.create.coord.generic('lat', data=lat)
        ds.create.data_var.generic('temp', ('time', 'lat'), dtype='float32')
        ds['temp'][:] = np.full((2, 2), 42.0, dtype='float32')

    res = merge_into([p_new], p_target, overlap='last')
    res.close()

    with cfdb.open_dataset(p_target) as ds:
        data = ds['temp'].data
        assert data.shape == (2, 2)
        assert np.allclose(data, 42.0)

