import pytest
import numpy as np
from cfdb import open_dataset

def test_bool_behavior(tmp_path):
    fp = tmp_path / "test_bool.cfdb"

    # 1. New empty dataset
    with open_dataset(fp, flag='n') as ds:
        # ds is open, but empty (len(ds) == 0)
        assert ds.is_open is True
        assert bool(ds) is False
        assert len(ds) == 0
        
        # Add a coordinate
        ds.create.coord.lat(data=np.array([10, 20]), chunk_shape=(2,))
        
        # Now ds is open AND populated
        assert ds.is_open is True
        assert bool(ds) is True
        assert len(ds) == 1

    # 2. Re-open existing dataset
    with open_dataset(fp) as ds:
        assert ds.is_open is True
        assert bool(ds) is True
        assert len(ds) == 1

    # 3. Create another empty dataset
    fp2 = tmp_path / "test_bool_empty.cfdb"
    with open_dataset(fp2, flag='n') as ds:
        assert ds.is_open is True
        assert bool(ds) is False

def test_numpy_interop(tmp_path):
    fp = tmp_path / "test_numpy.cfdb"
    data = np.arange(10, dtype='float32')

    with open_dataset(fp, flag='n') as ds:
        ds.create.coord.lat(data=data, chunk_shape=(5,))
        coord = ds['latitude']
        
        # Test .values
        assert isinstance(coord.values, np.ndarray)
        assert np.allclose(coord.values, data)
        
        # Test np.array()
        arr = np.array(coord)
        assert isinstance(arr, np.ndarray)
        assert np.allclose(arr, data)
        
        # Test dtype
        arr_int = np.array(coord, dtype='int')
        assert arr_int.dtype == np.int64 or arr_int.dtype == np.int32
        assert np.allclose(arr_int, data.astype(int))

        # Test functions
        assert np.mean(coord) == np.mean(data)
