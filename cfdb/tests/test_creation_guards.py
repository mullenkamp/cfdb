"""
Regression tests for F5 (empty-coord/zero chunk_shape creation guards), the F6
shared-blank copy, and the F7 tidies (2026-07-21).
"""
import numpy as np
import pytest

import cfdb
from cfdb import dtypes


###################################################
### F5: creation guards

def test_empty_coord_chunk_guess_raises(tmp_path):
    with cfdb.open_dataset(tmp_path / 'a.cfdb', flag='n') as ds:
        ds.create.coord.generic(name='x', dtype='int32', step=1)
        with pytest.raises(ValueError, match="'x' are empty"):
            ds.create.data_var.generic(name='v', coords=('x',), dtype=dtypes.dtype('float64'))


def test_empty_coord_with_explicit_chunk_shape_works(tmp_path):
    path = tmp_path / 'b.cfdb'
    with cfdb.open_dataset(path, flag='n') as ds:
        ds.create.coord.generic(name='x', dtype='int32', step=1)
        ds.create.data_var.generic(name='v', coords=('x',), dtype=dtypes.dtype('float64'), chunk_shape=(10,))
        ds['x'].append(np.arange(25, dtype='int32'))
        ds['v'][:] = np.arange(25.0)
    with cfdb.open_dataset(path) as ds:
        assert np.array_equal(ds['v'].data, np.arange(25.0))


def test_filled_coords_first_guesses_normally(tmp_path):
    with cfdb.open_dataset(tmp_path / 'c.cfdb', flag='n') as ds:
        ds.create.coord.generic(name='x', data=np.arange(0.0, 50), step=1.0)
        dv = ds.create.data_var.generic(name='v', coords=('x',), dtype=dtypes.dtype('float64'))
        assert all(c > 0 for c in dv.chunk_shape)
        dv[:] = np.arange(50.0)
        assert np.array_equal(dv.data, np.arange(50.0))


def test_explicit_zero_chunk_shape_raises(tmp_path):
    with cfdb.open_dataset(tmp_path / 'd.cfdb', flag='n') as ds:
        ds.create.coord.generic(name='x', data=np.arange(0.0, 20), step=1.0)
        with pytest.raises(ValueError, match='> 0'):
            ds.create.data_var.generic(name='v', coords=('x',), dtype=dtypes.dtype('float64'), chunk_shape=(0,))
        with pytest.raises(ValueError, match='> 0'):
            ds.create.coord.generic(name='y', dtype='int32', step=1, chunk_shape=(0,))


def test_numpy_int_chunk_shape_accepted(tmp_path):
    with cfdb.open_dataset(tmp_path / 'e.cfdb', flag='n') as ds:
        ds.create.coord.generic(name='x', data=np.arange(0.0, 20), step=1.0, chunk_shape=(np.int64(8),))
        dv = ds.create.data_var.generic(name='v', coords=('x',), dtype=dtypes.dtype('float64'),
                                        chunk_shape=(np.int32(8),))
        assert dv.chunk_shape == (8,)
        dv[:] = np.arange(20.0)
        assert np.array_equal(dv.data, np.arange(20.0))


###################################################
### F6: shared-blank copy

def test_mutating_yielded_missing_chunk_does_not_poison_later_yields(tmp_path):
    path = tmp_path / 'f.cfdb'
    with cfdb.open_dataset(path, flag='n') as ds:
        ds.create.coord.generic(name='x', data=np.arange(0.0, 40), chunk_shape=(10,), step=1.0)
        dv = ds.create.data_var.generic(name='v', coords=('x',), dtype=dtypes.dtype('float64'), chunk_shape=(10,))
        dv[0:10] = np.arange(10.0)         # chunks [10:40) missing
    with cfdb.open_dataset(path) as ds:
        mutated = False
        later_missing = []
        for _tc, d in ds['v'].iter_chunks():
            if np.isnan(d).all():
                if not mutated:
                    d[:] = 777.0           # consumer mutates the FIRST missing chunk
                    mutated = True
                else:
                    later_missing.append(d.copy())
        assert mutated and len(later_missing) == 2
        # the 777 mutation must NOT have propagated into later missing yields
        for d in later_missing:
            assert np.isnan(d).all()


###################################################
### F7: tidies

def test_dataset_rechunker_write_count_sums_across_vars(tmp_path):
    with cfdb.open_dataset(tmp_path / 'g.cfdb', flag='n') as ds:
        ds.create.coord.generic(name='x', data=np.arange(0.0, 40), chunk_shape=(10,), step=1.0)
        for name in ('a', 'b', 'c'):
            dv = ds.create.data_var.generic(name=name, coords=('x',), dtype=dtypes.dtype('float64'), chunk_shape=(10,))
            dv[:] = np.arange(40.0)
        n_reads, n_writes = ds.rechunker().calc_n_reads_rechunker({'x': 8})
        assert n_writes == 3 * 5           # 3 vars x ceil(40/8) target chunks each
        assert n_reads == 3 * 4            # 3 vars x 4 source chunks (ideal path)


def test_items_decoded_parameter_honored(tmp_path):
    with cfdb.open_dataset(tmp_path / 'h.cfdb', flag='n') as ds:
        ds.create.coord.generic(name='x', data=np.arange(0.0, 10), chunk_shape=(10,), step=1.0)
        dv = ds.create.data_var.generic(name='v', coords=('x',),
                                        dtype=dtypes.dtype('int32', min_value=5, max_value=100),
                                        chunk_shape=(10,))
        dv[:] = np.arange(5, 15, dtype='int32')
        decoded_vals = [v for _i, v in dv.items()]
        encoded_vals = [v for _i, v in dv.items(decoded=False)]
        assert decoded_vals[0] == 5
        assert encoded_vals[0] == 1        # 5 - offset(4) = 1
