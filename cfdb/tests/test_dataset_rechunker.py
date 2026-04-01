import os
import numpy as np
import pytest
import cfdb
from cfdb.main import open_dataset
from cfdb.tests.test_dataset import populated_dataset

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
