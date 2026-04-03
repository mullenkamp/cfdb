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

