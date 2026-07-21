"""Perf-round regression tests (C1: storage-space rechunk declaration).

The rechunker declares the source grid to rechunkit at the real storage-chunk
alignment (declared = user + origin-phase), so every read maps to exactly one
storage chunk after coordinate prepends. Fail-before: pre-C1 the same fixture
decompresses each stored chunk ~3.4x through the rechunker.
"""
import pathlib

import numpy as np
import pytest

import cfdb
import cfdb.indexers as indexers
import cfdb.support_classes as sc
from cfdb import dtypes

NY, NT = 41, 83
T0 = np.datetime64('2020-01-01T00', 'h')


@pytest.fixture()
def prepended_ds(tmp_path):
    p = tmp_path / 'prepended.cfdb'
    rng = np.random.default_rng(7)
    data = rng.normal(size=(NY, NT))
    with cfdb.open_dataset(p, flag='n') as ds:
        ds.create.coord.generic(name='y', data=np.arange(0.0, NY), chunk_shape=(7,), step=1.0)
        ds.create.coord.time(data=T0 + np.arange(NT), dtype='datetime64[h]', chunk_shape=(13,), step=True)
        v = ds.create.data_var.generic(name='v', coords=('y', 'time'),
                                       dtype=dtypes.dtype('float64'), chunk_shape=(7, 13))
        v[:] = data
        vi = ds.create.data_var.generic(name='vi', coords=('y', 'time'),
                                        dtype=dtypes.dtype('int32', min_value=0, max_value=60000),
                                        chunk_shape=(5, 17))
        vi[:] = (np.abs(data) * 100).astype('int32')
    with cfdb.open_dataset(p, flag='w') as ds:
        ds['y'].prepend(np.array([-1.0]))
        ds['time'].prepend(np.array([T0 - 1]))
    return p


def _nan_equal(a, b):
    return bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


def test_prepend_rechunk_single_decompress(prepended_ds, monkeypatch):
    # Post-prepend, every read must hit exactly ONE storage chunk: at
    # generous memory each stored chunk decompresses exactly once.
    # Pre-C1: ~3.4x (every read spans 2 storage chunks per misaligned dim).
    count = [0]
    orig = sc.Compressor._zstd_decompress

    def counting(self, data):
        count[0] += 1
        return orig(self, data)

    monkeypatch.setattr(sc.Compressor, '_zstd_decompress', counting)

    with cfdb.open_dataset(prepended_ds) as ds:
        v = ds['v']
        count[0] = 0
        for _wc, _arr in v.rechunker().rechunk((10, 20)):
            pass
        stored = 6 * 7  # written chunks (prepended row/col are blank)
        assert count[0] <= 1.15 * stored


def test_prepend_rechunk_exact_and_ordered(prepended_ds):
    # Exactness + canonical C-order through the storage-space declaration,
    # for a plain var, a sliced view, and an encoded var.
    with cfdb.open_dataset(prepended_ds) as ds:
        for name, view_sel in (('v', None), ('v', (slice(3, 30), slice(5, 70))), ('vi', None)):
            var = ds[name]
            if view_sel is not None:
                var = var[view_sel]
            ref = var.data
            out = np.full(var.shape, np.nan if name == 'v' else 0,
                          dtype=ref.dtype)
            last_start = None
            for wc, arr in var.rechunker().rechunk((10, 20), max_mem=2**14):
                start = tuple(s.start for s in wc)
                if last_start is not None:
                    assert start > last_start  # strict C-order of tuple starts
                last_start = start
                out[wc] = arr
            if name == 'v':
                assert _nan_equal(out, ref)
            else:
                assert np.array_equal(out, ref)


def test_prepend_count_truth(prepended_ds, monkeypatch):
    # calc_n_reads_rechunker must match the actual source-function calls
    # (one slices_to_chunks_keys invocation per read) on origin-shifted
    # variables and views.
    calls = [0]
    orig = indexers.slices_to_chunks_keys

    def counting(*a, **k):
        calls[0] += 1
        return orig(*a, **k)

    monkeypatch.setattr(indexers, 'slices_to_chunks_keys', counting)

    with cfdb.open_dataset(prepended_ds) as ds:
        for var in (ds['v'], ds['v'][3:30, 5:70]):
            planned, _w = var.rechunker().calc_n_reads_rechunker((10, 20), 2**14)
            calls[0] = 0
            for _wc, _arr in var.rechunker().rechunk((10, 20), max_mem=2**14):
                pass
            assert calls[0] == planned


def test_dataset_rechunker_zip_prepended(prepended_ds):
    # Two variables with DIFFERENT storage chunk shapes (7,13) and (5,17) on
    # the same prepended coords: per-variable declared spaces differ, but the
    # target grid is identical, so the zip stays synchronized and exact.
    with cfdb.open_dataset(prepended_ds) as ds:
        ref_v = ds['v'].data
        ref_vi = ds['vi'].data
        out_v = np.full(ref_v.shape, np.nan)
        out_vi = np.zeros(ref_vi.shape, dtype=ref_vi.dtype)
        for tc, var_data in ds.rechunker(['v', 'vi']).rechunk({'y': 10, 'time': 20}, max_mem=2**20):
            sel = tuple(tc[c] for c in ('y', 'time'))
            out_v[sel] = var_data['v']
            out_vi[sel] = var_data['vi']
        assert _nan_equal(out_v, ref_v)
        assert np.array_equal(out_vi, ref_vi)


def test_multi_chunk_assembly_fallback(prepended_ds, monkeypatch):
    # The storage-space declaration makes every normal read single-chunk, so
    # the assembly loop in the source function would otherwise rot untested.
    # Force it: a fake rechunkit.rechunker issues one read spanning multiple
    # storage chunks and yields the result straight through.
    captured = {}

    def fake_rechunker(source, shape, dtype, src_cs, tgt_cs, max_mem, sel=None, itemsize=None):
        # a declared-space window spanning >= 2 storage chunks per dim
        read = tuple(slice(s.start, min(s.start + 2 * c, s.stop)) for s, c in zip(sel, src_cs))
        data = source(read)
        captured['read'] = read
        captured['sel'] = sel
        yield tuple(slice(0, r.stop - r.start) for r in read), data

    monkeypatch.setattr(sc.rechunkit, 'rechunker', fake_rechunker)

    with cfdb.open_dataset(prepended_ds) as ds:
        v = ds['v']
        ref = v.data
        for _wc, arr in v.rechunker().rechunk((10, 20)):
            pass
        read, sel = captured['read'], captured['sel']
        # the read starts at sel.start, i.e. user index 0
        spans = tuple(r.stop - r.start for r in read)
        expected = ref[tuple(slice(0, sp) for sp in spans)]
        assert arr.shape == expected.shape
        assert _nan_equal(arr, expected)
