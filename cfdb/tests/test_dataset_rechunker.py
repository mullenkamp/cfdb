import os
import pathlib
import numpy as np
import pytest
import cfdb
from cfdb import dtypes
from cfdb.main import open_dataset
from cfdb.tests.test_dataset import populated_dataset

script_path = pathlib.Path(os.path.realpath(os.path.dirname(__file__)))

def test_dataset_rechunker_api(populated_dataset):
    """Test the basic DatasetRechunker API and synchronization."""
    with open_dataset(populated_dataset, 'w') as ds:
        # Add another variable to test synchronization
        coords = ('latitude', 'longitude', 'time')
        dv2 = ds.create.data_var.generic('pressure', coords, dtype='float32', chunk_shape=(20, 30, 10))
        dv2[:] = np.random.rand(*ds['temperature'].shape).astype('float32')
        
        data_vars = list(ds.data_var_names)
        dr = ds.rechunker(data_vars=data_vars)
        
        # Test planning methods
        chunk_shape = {'latitude': 50, 'longitude': 50}
        ideal_mem = dr.calc_ideal_read_chunk_mem(chunk_shape)
        assert ideal_mem > 0
        
        n_reads, n_writes = dr.calc_n_reads_rechunker(chunk_shape)
        assert n_reads > 0
        assert n_writes > 0
        
        # Test iteration
        count = 0
        for target_chunk, var_data in dr.rechunk(chunk_shape, max_mem=2**28):
            assert isinstance(target_chunk, dict)
            assert isinstance(var_data, dict)
            assert set(var_data.keys()) == set(data_vars)
            
            # Verify alignment
            for name in data_vars:
                dv = ds[name]
                sel = tuple(target_chunk[cn] for cn in dv.coord_names)
                # expected = dv[sel].data[0] # WRONG: data[0] slices first dim
                expected = dv[sel].data
                np.testing.assert_allclose(var_data[name], expected)
                
            count += 1
        assert count > 0

def test_dataset_rechunker_validation(populated_dataset):
    """Test validation logic in DatasetRechunker."""
    with open_dataset(populated_dataset) as ds:
        # Test unknown variable
        with pytest.raises(KeyError):
            ds.rechunker(data_vars=['nonexistent'])
            
        # Test empty list
        with pytest.raises(ValueError, match="No data variables specified"):
            ds.rechunker(data_vars=[])

def test_dataset_rechunker_with_selection(populated_dataset):
    """Test DatasetRechunker works correctly on a subset DatasetView."""
    with open_dataset(populated_dataset) as ds:
        # Create a DatasetView
        view = ds.select({'latitude': slice(10, 40), 'time': slice(2, 8)})
        
        # We need variables that share the same coords.
        # Assuming 'temperature' and 'pressure' exist and share coords in populated_dataset.
        data_vars = [name for name in view.data_var_names if view[name].coord_names == ('latitude', 'longitude', 'time')]
        if len(data_vars) < 2:
            pytest.skip("Test requires at least 2 data variables with the same coords")
            
        dr = view.rechunker(data_vars=data_vars)
        chunk_shape = {'latitude': 15, 'longitude': 20}
        
        # Test iteration on the view
        count = 0
        for target_chunk, var_data in dr.rechunk(chunk_shape):
            # Verify alignment against the view
            for name in data_vars:
                dv_view = view[name]
                sel = tuple(target_chunk[cn] for cn in dv_view.coord_names)
                expected = dv_view[sel].data
                np.testing.assert_allclose(var_data[name], expected)
            count += 1
        assert count > 0

def test_dataset_iter_chunks_uses_rechunker(populated_dataset):
    """Verify Dataset.iter_chunks uses the new rechunker path."""
    with open_dataset(populated_dataset) as ds:
        chunk_shape = {'latitude': 25}

        # This should now yield identical results to rechunker.rechunk
        iter_results = list(ds.iter_chunks(chunk_shape))
        rechunk_results = list(ds.rechunker().rechunk(chunk_shape))

        assert len(iter_results) == len(rechunk_results)
        for (t1, d1), (t2, d2) in zip(iter_results, rechunk_results):
            assert t1 == t2
            for name in d1:
                np.testing.assert_allclose(d1[name], d2[name])


def test_variable_rechunker_with_selection(populated_dataset):
    """Test variable-level rechunker with a view whose offset misaligns with chunk boundaries."""
    with open_dataset(populated_dataset) as ds:
        temp = ds['temperature']
        # Offset 10 misaligns with chunk_shape 20
        view = temp[slice(10, 40), :, slice(2, 8)]
        target_chunk_shape = (15, 20, 6)

        count = 0
        for slices, data in view.rechunker().rechunk(target_chunk_shape):
            expected = view[slices].data
            np.testing.assert_allclose(data, expected)
            count += 1
        assert count > 0


def test_rechunker_encoded_dtype(populated_dataset):
    """Test rechunker with encoded dtype (int-encoded float)."""
    with open_dataset(populated_dataset) as ds:
        temp = ds['temperature']
        assert temp.dtype.dtype_encoded is not None

        new_shape = (41, 41, 10)
        result = np.full(temp.shape, np.nan, dtype=temp.dtype.dtype_decoded)
        for slices, data in temp.rechunker().rechunk(new_shape):
            result[slices] = data

        np.testing.assert_allclose(result, temp.data)


def test_rechunker_encoded_dtype_with_selection(populated_dataset):
    """Test encoded dtype rechunker with a misaligned selection."""
    with open_dataset(populated_dataset) as ds:
        temp = ds['temperature']
        assert temp.dtype.dtype_encoded is not None

        view = temp[slice(10, 40), :, slice(2, 8)]
        target_chunk_shape = (15, 20, 6)

        count = 0
        for slices, data in view.rechunker().rechunk(target_chunk_shape):
            expected = view[slices].data
            np.testing.assert_allclose(data, expected)
            count += 1
        assert count > 0


def test_rechunker_after_prepend():
    """Test rechunker works correctly with non-zero coordinate origins from prepend."""
    fp = script_path.joinpath('test_rechunk_prepend.cfdb')

    main_lat = np.linspace(0, 4.9, 50, dtype='float32')
    prepend_lat = np.linspace(-5.0, -0.1, 50, dtype='float32')
    lon_data = np.linspace(0, 4.9, 50, dtype='float32')
    time_data = np.linspace(0, 9, 10, dtype='datetime64[D]')

    with open_dataset(fp, flag='n') as ds:
        ds.create.coord.lat(data=main_lat, chunk_shape=(20,))
        ds.create.coord.lon(data=lon_data, chunk_shape=(20,))
        ds.create.coord.time(data=time_data, dtype=time_data.dtype)
        dv = ds.create.data_var.generic('val', ('latitude', 'longitude', 'time'), dtype='float32', chunk_shape=(20, 20, 10))
        dv[:] = np.random.rand(50, 50, 10).astype('float32')

    with open_dataset(fp, flag='w') as ds:
        ds['latitude'].prepend(prepend_lat)
        # Write data into the prepended region
        ds['val'][slice(0, 50), :, :] = np.random.rand(50, 50, 10).astype('float32')

    with open_dataset(fp) as ds:
        val = ds['val']
        assert val.shape == (100, 50, 10)

        # Full rechunk
        new_shape = (30, 30, 10)
        result = np.full(val.shape, np.nan, dtype='float32')
        for slices, data in val.rechunker().rechunk(new_shape):
            result[slices] = data
        np.testing.assert_allclose(result, val.data)

        # Rechunk a view that crosses the prepend boundary
        view = val[slice(40, 70), :, :]
        result_view = np.full(view.shape, np.nan, dtype='float32')
        for slices, data in view.rechunker().rechunk((15, 25, 10)):
            result_view[slices] = data
        np.testing.assert_allclose(result_view, view.data)

    fp.unlink()


def test_rechunker_chunk_shape_larger_than_variable(populated_dataset):
    """Test rechunker when target chunk_shape exceeds variable dimensions."""
    with open_dataset(populated_dataset) as ds:
        temp = ds['temperature']
        # latitude has 100 elements, request 200
        big_shape = (200, 200, 20)
        chunks = list(temp.rechunker().rechunk(big_shape))
        # Should yield exactly one chunk covering everything
        assert len(chunks) == 1
        slices, data = chunks[0]
        np.testing.assert_allclose(data, temp.data)


def test_dataset_rechunker_single_variable(populated_dataset):
    """Test DatasetRechunker with a single variable."""
    with open_dataset(populated_dataset) as ds:
        chunk_shape = {'latitude': 25}
        dr = ds.rechunker(data_vars=['temperature'])

        count = 0
        for target_chunk, var_data in dr.rechunk(chunk_shape):
            assert list(var_data.keys()) == ['temperature']
            dv = ds['temperature']
            sel = tuple(target_chunk[cn] for cn in dv.coord_names)
            expected = dv[sel].data
            np.testing.assert_allclose(var_data['temperature'], expected)
            count += 1
        assert count > 0
