"""File-format, byte-shuffle compression and chunk-default tests (cfdb 0.10.0).

Written before the implementation and seen failing against 0.9.7 first. The shuffle checks look at
the STORED bytes, not only at round trips: a wrong but consistent item size round-trips perfectly
and only costs compression ratio, so a round trip alone cannot catch it.
"""
import math
import pathlib
import sys

import msgspec
import numpy as np
import pytest
import rechunkit
import shapely
import zstandard as zstd
import lz4.frame
from enum import Enum

import cfdb
from cfdb import dtypes, utils, support_classes as sc

FIXTURES = pathlib.Path(__file__).parent / 'fixtures'
COMPRESSIONS = ('zstd', 'lz4', 'zstd_shuffle', 'lz4_shuffle')

# (label, dtype factory, values factory, expected shuffle item size)
DTYPE_CASES = [
    ('uint8', lambda: dtypes.dtype('uint8'), lambda n: (np.arange(n) % 250).astype('uint8'), None),
    ('uint16', lambda: dtypes.dtype('uint16'), lambda n: (np.arange(n) * 3 % 60000).astype('uint16'), 2),
    ('int32', lambda: dtypes.dtype('int32'), lambda n: (np.arange(n) * 7 - 5000).astype('int32'), 4),
    ('int64', lambda: dtypes.dtype('int64'), lambda n: (np.arange(n) * 11 - 70000).astype('int64'), 8),
    ('float32', lambda: dtypes.dtype('float32'), lambda n: np.sin(np.arange(n) / 50).astype('float32'), 4),
    ('float64', lambda: dtypes.dtype('float64'), lambda n: np.sin(np.arange(n) / 50), 8),
    ('packed_f32_u16', lambda: dtypes.dtype('float32', 2, -10, 40),
     lambda n: (np.sin(np.arange(n) / 50) * 20 + 10).astype('float32'), 2),
    ('packed_f32_u32', lambda: dtypes.dtype('float32', 2, -1000, 50000),
     lambda n: (np.sin(np.arange(n) / 50) * 2000 + 3000).astype('float32'), 4),
    ('packed_f64_u16', lambda: dtypes.dtype('float64', 2, -1.0, 1.0), lambda n: np.round(np.sin(np.arange(n) / 50), 2), 2),
    ('datetime64_raw', lambda: dtypes.dtype('datetime64[s]'),
     lambda n: np.datetime64('2020-01-01', 's') + (np.arange(n) * 3600).astype('timedelta64[s]'), 8),
    ('datetime64_packed', lambda: dtypes.dtype('datetime64[m]', None, '2019-01-01', '2021-01-01'),
     lambda n: np.datetime64('2020-01-01', 'm') + (np.arange(n) * 60).astype('timedelta64[m]'), 4),
    ('bool', lambda: dtypes.dtype('bool'), lambda n: (np.arange(n) % 3) == 0, None),
]
NON_FIXED_CASES = [('str', lambda: dtypes.dtype('str')), ('point', lambda: dtypes.dtype('point', precision=5))]


def _grid(path, compression, dt, values, chunk=64):
    """1-D grid dataset with one variable 'v' holding ``values``."""
    n = len(values)
    ds = cfdb.open_dataset(path, flag='n', compression=compression, n_buckets=1009)
    ds.create.coord.generic('x', data=np.arange(float(n)), chunk_shape=(chunk,), step=1.0)
    v = ds.create.data_var.generic('v', ('x',), dt, chunk_shape=(chunk,))
    v[:] = values
    return ds


def _encoded_chunk(dt, values):
    """The encoded array cfdb hands its compressor for one full chunk."""
    enc = dt.dtype_encoded if dt.dtype_encoded is not None else dt.dtype_decoded
    return np.frombuffer(dt.dumps(values), dtype=enc)


def _planes_reference(arr, itemsize):
    """Byte-plane layout by an INDEPENDENT route (memory transpose, little-endian): plane i holds
    byte i of every value. The implementation uses shifts; the two agree on little-endian hosts."""
    return np.ascontiguousarray(arr.view('u1').reshape(-1, itemsize).T).tobytes()


# ---------------------------------------------------------------- fixtures written by older versions

@pytest.mark.parametrize('path', sorted(FIXTURES.glob('*.cfdb')), ids=lambda p: p.name)
def test_fixture_reads_bit_identically(path):
    exp = np.load(path.with_name(path.stem + '_expected.npz'))
    with cfdb.open_dataset(path) as ds:
        assert f'{ds.compression}-{ds.compression_level}' == str(exp['__compression__'])
        assert repr(sorted(ds.attrs.data.items())) == str(exp['__attrs__'])
        for name in exp.files:
            if name.startswith('__'):
                continue
            data = ds[name].data
            if data.dtype.kind in 'OT':
                data = np.array([shapely.to_wkt(g, rounding_precision=6) if hasattr(g, 'geom_type') else str(g)
                                 for g in data.ravel()], dtype='U').reshape(data.shape)
            ref = exp[name]
            assert data.dtype == ref.dtype and data.shape == ref.shape, name
            assert data.tobytes() == ref.tobytes(), name
            # and through a view
            if data.ndim >= 1 and data.shape[0] > 2:
                sub = ds.select({ds[name].coord_names[0]: slice(1, -1)})[name].data
                if sub.dtype.kind in 'OT':
                    continue
                assert sub.tobytes() == ref[1:-1].tobytes(), f'{name} via select()'


# ---------------------------------------------------------------- item-size binding

@pytest.mark.parametrize('label,make_dt,make_vals,itemsize', DTYPE_CASES, ids=[c[0] for c in DTYPE_CASES])
def test_shuffle_itemsize_is_the_encoded_width(label, make_dt, make_vals, itemsize):
    dt = make_dt()
    got = sc.shuffle_itemsize(dt)
    assert (got if got and got > 1 else None) == itemsize
    if itemsize is not None:  # and it is the width of the bytes actually compressed
        assert len(dt.dumps(make_vals(10))) == 10 * itemsize


@pytest.mark.parametrize('label,make_dt', NON_FIXED_CASES, ids=[c[0] for c in NON_FIXED_CASES])
def test_shuffle_itemsize_none_for_variable_length(label, make_dt):
    assert sc.shuffle_itemsize(make_dt()) is None


# ---------------------------------------------------------------- the stored bytes really are shuffled

@pytest.mark.parametrize('compression', ['zstd_shuffle', 'lz4_shuffle'])
@pytest.mark.parametrize('label,make_dt,make_vals,itemsize', DTYPE_CASES, ids=[c[0] for c in DTYPE_CASES])
def test_stored_chunk_layout(tmp_path, compression, label, make_dt, make_vals, itemsize):
    if sys.byteorder != 'little':
        pytest.skip('reference layout is written for little-endian hosts')
    dt = make_dt()
    values = make_vals(64)
    with _grid(tmp_path / 'f.cfdb', compression, dt, values) as ds:
        stored = ds._blt.get(utils.make_var_chunk_key('v', (0,)))
    raw = zstd.ZstdDecompressor().decompress(stored) if compression.startswith('zstd') else lz4.frame.decompress(stored)
    enc = _encoded_chunk(dt, values)
    if itemsize is None:
        assert raw == enc.tobytes()                       # 1-byte types are stored unshuffled
    else:
        assert raw == _planes_reference(enc, itemsize)   # planes of the ENCODED width
        assert raw != enc.tobytes()


@pytest.mark.parametrize('label,make_dt', NON_FIXED_CASES, ids=[c[0] for c in NON_FIXED_CASES])
def test_variable_length_stored_unshuffled(tmp_path, label, make_dt):
    dt = make_dt()
    values = (np.array([shapely.Point(170 + i / 10, -43.0) for i in range(64)], dtype=object) if label == 'point'
              else np.array([f's{i}' for i in range(64)], dtype=object))
    with _grid(tmp_path / 'f.cfdb', 'zstd_shuffle', dt, values) as ds:
        stored = ds._blt.get(utils.make_var_chunk_key('v', (0,)))
        assert zstd.ZstdDecompressor().decompress(stored) == dt.dumps(ds['v'][0:64].data)


# ---------------------------------------------------------------- round trips on every read path

@pytest.mark.parametrize('compression', COMPRESSIONS)
@pytest.mark.parametrize('label,make_dt,make_vals,itemsize', DTYPE_CASES, ids=[c[0] for c in DTYPE_CASES])
def test_round_trip_all_read_paths(tmp_path, compression, label, make_dt, make_vals, itemsize):
    dt = make_dt()
    values = make_vals(200)
    if values.dtype.kind == 'f':
        values = values.copy(); values[3] = np.nan
    if values.dtype.kind == 'M':
        values = values.copy(); values[3] = np.datetime64('NaT')
    with _grid(tmp_path / 'f.cfdb', compression, dt, values) as ds:
        v = ds['v']
        expect = dt.loads(dt.dumps(values), (200,))       # what any compression must give back
        assert v.data.tobytes() == expect.tobytes()
        got = np.concatenate([d for _, d in sorted(v.iter_chunks(), key=lambda t: t[0][0].start)])
        assert got.tobytes() == expect.tobytes()
        got = np.concatenate([np.array(d) for _, d in v.iter_chunks(chunk_shape={'x': 37})])
        assert got.tobytes() == expect.tobytes()


@pytest.mark.parametrize('compression', COMPRESSIONS)
def test_round_trip_multidim_groupby_and_coords(tmp_path, compression):
    t0 = np.datetime64('2024-01-01T00', 'h')
    with cfdb.open_dataset(tmp_path / 'f.cfdb', flag='n', compression=compression, n_buckets=1009) as ds:
        ds.create.coord.time(data=t0 + np.arange(96), dtype='datetime64[h]', chunk_shape=(24,), step=True)
        ds.create.coord.generic('y', data=np.arange(0.0, 20.0), chunk_shape=(8,), step=1.0)
        v = ds.create.data_var.generic('v', ('time', 'y'), dtypes.dtype('float32', 2, -50, 50), chunk_shape=(24, 8))
        data = (np.sin(np.arange(96 * 20) / 30).reshape(96, 20) * 30).astype('float32')
        v[:] = data
        ds['time'].append(t0 + np.arange(96, 120))
        ds['time'].prepend(t0 - np.arange(1, 25)[::-1])
    with cfdb.open_dataset(tmp_path / 'f.cfdb') as ds:
        assert ds.compression == compression
        assert np.array_equal(ds['time'].data, t0 + np.arange(-24, 120))
        full = ds['v'].data
        dt = ds['v'].dtype
        expect = dt.loads(dt.dumps(data), data.shape)      # the dtype's own encode/decode
        assert full[24:120].tobytes() == expect.tobytes()
        assert np.isnan(full[:24]).all() and np.isnan(full[120:]).all()
        daily = {sl[0].start: d.copy() for sl, d in ds['v'].groupby({'time': 'D'})}
        assert np.array_equal(np.concatenate([daily[k] for k in sorted(daily)]), full, equal_nan=True)


# ---------------------------------------------------------------- defaults, loud errors, format version

def test_new_datasets_default_to_zstd_shuffle(tmp_path):
    with cfdb.open_dataset(tmp_path / 'f.cfdb', flag='n') as ds:
        assert (ds.compression, ds.compression_level) == ('zstd_shuffle', 1)
        assert ds._sys_meta.format_version == 1
    import inspect
    from cfdb import edataset
    sig = inspect.signature(edataset.open_edataset)
    assert sig.parameters['compression'].default == 'zstd_shuffle'
    assert sig.parameters['compression_level'].default is None


def test_old_reader_fails_loudly_on_shuffled_file(tmp_path):
    """Simulate an older cfdb-models (enum without the shuffle values) decoding a new file's metadata."""
    class OldCompressor(Enum):
        zstd = 'zstd'
        lz4 = 'lz4'

    class OldSysMeta(msgspec.Struct):
        dataset_type: str
        compression: OldCompressor
        compression_level: int

    with cfdb.open_dataset(tmp_path / 'f.cfdb', flag='n', compression='zstd_shuffle') as ds:
        meta = ds._blt.get_metadata()
    with pytest.raises(msgspec.ValidationError, match='compression'):
        msgspec.convert(meta, OldSysMeta)


def test_unknown_compression_gives_upgrade_message(tmp_path):
    import booklet
    p = tmp_path / 'f.cfdb'
    with cfdb.open_dataset(p, flag='n', n_buckets=1009):
        pass
    with booklet.open(p, 'w') as b:
        m = b.get_metadata(); m['compression'] = 'zstd_from_the_future'; b.set_metadata(m)
    with pytest.raises(ValueError, match='newer'):
        cfdb.open_dataset(p)


def test_future_format_version_refused(tmp_path):
    import booklet
    p = tmp_path / 'f.cfdb'
    with cfdb.open_dataset(p, flag='n', n_buckets=1009):
        pass
    with booklet.open(p, 'w') as b:
        m = b.get_metadata(); m['format_version'] = 99; b.set_metadata(m)
    with pytest.raises(ValueError, match='newer'):
        cfdb.open_dataset(p)


# ---------------------------------------------------------------- copy, merge_into, multiprocessing

@pytest.mark.parametrize('compression', COMPRESSIONS)
@pytest.mark.parametrize('shift', ['none', 'append', 'prepend'])
def test_copy_preserves_data_and_compression(tmp_path, compression, shift):
    src, dst = tmp_path / 'src.cfdb', tmp_path / 'dst.cfdb'
    with cfdb.open_dataset(src, flag='n', compression=compression, n_buckets=1009) as ds:
        ds.create.coord.generic('x', data=np.arange(20.0, 40.0), chunk_shape=(10,), step=1.0)
        v = ds.create.data_var.generic('v', ('x',), dtypes.dtype('float32', 2, 0, 100), chunk_shape=(10,))
        if shift == 'prepend':
            ds['x'].prepend(np.arange(5.0, 20.0))
        elif shift == 'append':
            ds['x'].append(np.arange(40.0, 55.0))
        v = ds['v']
        v[:] = ds['x'].data.astype('float32')
        src_data = v.data.copy()
        ds.copy(dst)
    with cfdb.open_dataset(dst) as d2:
        assert d2.compression == compression
        assert d2['v'].data.tobytes() == src_data.tobytes()


def test_merge_into_surfaces_real_error(tmp_path):
    import booklet
    bad, tgt = tmp_path / 'bad.cfdb', tmp_path / 'tgt.cfdb'
    for p in (bad, tgt):
        with cfdb.open_dataset(p, flag='n', n_buckets=1009) as ds:
            ds.create.coord.generic('x', data=np.arange(10.0), chunk_shape=(10,), step=1.0)
    with booklet.open(bad, 'w') as b:
        m = b.get_metadata(); m['compression'] = 'zstd_from_the_future'; b.set_metadata(m)
    with pytest.raises(Exception) as err:
        cfdb.merge_into([bad], tgt)
    assert not isinstance(err.value, UnboundLocalError)


def _chunk_sum(target_chunk, data):
    return float(np.nansum(data))


def test_map_multiprocessing_on_shuffled_variable(tmp_path):
    values = (np.sin(np.arange(640) / 30) * 20 + 10).astype('float32')
    with _grid(tmp_path / 'f.cfdb', 'zstd_shuffle', dtypes.dtype('float32', 2, -10, 40), values) as ds:
        v = ds['v']
        expect = {sl[0].start: float(np.nansum(d)) for sl, d in v.iter_chunks()}
        got = {tc[0].start: r for tc, r in v.map(_chunk_sum, n_workers=2)}
        assert got.keys() == expect.keys()
        assert all(np.isclose(got[k], expect[k]) for k in expect)


# ---------------------------------------------------------------- chunk default

SHAPES = [(744, 1, 324, 277), (168, 1, 324, 277), (16650, 1000, 1600), (2000, 200000), (141, 183227),
          (500, 3000, 100), (29, 49, 14, 165, 221), (3600, 1800, 24), (1_000_000,)]


@pytest.mark.parametrize('itemsize', [1, 2, 4, 8])
@pytest.mark.parametrize('shape', SHAPES, ids=str)
def test_data_var_chunk_guess_in_element_band(shape, itemsize):
    target = utils.data_var_chunk_elements * itemsize
    n = math.prod(rechunkit.guess_chunk_shape(shape, itemsize, target))
    assert 100_000 <= n <= 400_000, (shape, itemsize, n)


@pytest.mark.parametrize('itemsize', [1, 2, 4, 8])
def test_created_data_var_uses_element_target(tmp_path, itemsize):
    dt = {1: 'uint8', 2: 'uint16', 4: 'uint32', 8: 'float64'}[itemsize]
    with cfdb.open_dataset(tmp_path / 'f.cfdb', flag='n', n_buckets=1009) as ds:
        ds.create.coord.generic('t', data=np.arange(2000.0), step=1.0)
        ds.create.coord.generic('s', data=np.arange(300.0), step=1.0)
        v = ds.create.data_var.generic('v', ('t', 's'), dtypes.dtype(dt))
        assert v.chunk_shape == rechunkit.guess_chunk_shape((2000, 300), itemsize, utils.data_var_chunk_elements * itemsize)


def test_coordinate_chunk_default_unchanged(tmp_path):
    with cfdb.open_dataset(tmp_path / 'f.cfdb', flag='n', n_buckets=1009) as ds:
        c = ds.create.coord.generic('t', data=np.arange(10.0), step=1.0)
        assert c.chunk_shape == rechunkit.guess_chunk_shape((1000000,), 8, 2**21)


# ---------------------------------------------------------------- code review cfdb-shuffle-code-1

@pytest.mark.skipif(not hasattr(np, 'float128') or np.dtype('float128').itemsize != 16, reason='no 16-byte float on this platform')
@pytest.mark.parametrize('compression', COMPRESSIONS)
def test_16_byte_values_round_trip_unshuffled(tmp_path, compression):
    """Worked in 0.9.7; the shuffle only handles 2/4/8-byte values, wider ones are stored as is."""
    values = np.arange(64, dtype='float128') / 3
    with _grid(tmp_path / 'f.cfdb', compression, dtypes.dtype('float128'), values) as ds:
        assert ds['v'].data.tobytes() == values.tobytes()
        stored = ds._blt.get(utils.make_var_chunk_key('v', (0,)))
    raw = zstd.ZstdDecompressor().decompress(stored) if compression.startswith('zstd') else lz4.frame.decompress(stored)
    assert raw == values.tobytes()


KIND_CASES = [('bool', 'bool', None), ('uint16', 'uint16', None), ('int16', 'int16', None), ('int32', 'int32', None),
              ('float32', 'float32', None), ('float64', 'float64', None), ('datetime64_raw', 'datetime64[s]', None),
              ('datetime64_packed_u2', 'datetime64[h]', ('2024-01-01', '2026-01-01'))]


@pytest.mark.parametrize('label,name,bounds', KIND_CASES, ids=[c[0] for c in KIND_CASES])
def test_created_data_var_uses_element_target_every_kind(tmp_path, label, name, bounds):
    dt = dtypes.dtype(name) if bounds is None else dtypes.dtype(name, None, *bounds)
    itemsize = sc.shuffle_itemsize(dt)
    with cfdb.open_dataset(tmp_path / 'f.cfdb', flag='n', n_buckets=1009) as ds:
        ds.create.coord.generic('t', data=np.arange(2000.0), step=1.0)
        ds.create.coord.generic('s', data=np.arange(300.0), step=1.0)
        v = ds.create.data_var.generic('v', ('t', 's'), dt)
        expect = rechunkit.guess_chunk_shape((2000, 300), itemsize, utils.data_var_chunk_elements * itemsize)
        assert v.chunk_shape == expect
        # and it differs from the variable-length (2 MiB) target, so a lost kind is visible
        assert expect != rechunkit.guess_chunk_shape((2000, 300), itemsize, utils.var_length_chunk_max) or itemsize == 8


def test_bad_compression_level_is_not_reported_as_newer_cfdb(tmp_path):
    import booklet
    p = tmp_path / 'f.cfdb'
    with cfdb.open_dataset(p, flag='n', n_buckets=1009):
        pass
    with booklet.open(p, 'w') as b:
        m = b.get_metadata(); m['compression_level'] = 'one'; b.set_metadata(m)
    with pytest.raises(Exception) as err:
        cfdb.open_dataset(p)
    assert 'newer' not in str(err.value)
    assert 'compression_level' in str(err.value)


def test_merge_into_pure_append_on_float_coordinate(tmp_path):
    tgt, src = tmp_path / 'tgt.cfdb', tmp_path / 'src.cfdb'
    for p, lo in ((tgt, 0.0), (src, 8.0)):
        with cfdb.open_dataset(p, flag='n', n_buckets=1009) as ds:
            ds.create.coord.generic('x', data=np.arange(lo, lo + 8.0), chunk_shape=(4,), step=1.0)
            v = ds.create.data_var.generic('v', ('x',), dtypes.dtype('uint32'), chunk_shape=(4,))
            v[:] = np.arange(lo, lo + 8.0).astype('uint32')
    cfdb.merge_into([src], tgt).close()
    with cfdb.open_dataset(tgt) as ds:
        assert ds['v'].data.tobytes() == np.arange(16, dtype='uint32').tobytes()


def test_open_inputs_closes_opened_files_when_a_later_one_fails(tmp_path):
    import booklet
    good, bad = tmp_path / 'good.cfdb', tmp_path / 'bad.cfdb'
    for p in (good, bad):
        with cfdb.open_dataset(p, flag='n', n_buckets=1009) as ds:
            ds.create.coord.generic('x', data=np.arange(10.0), chunk_shape=(10,), step=1.0)
    with booklet.open(bad, 'w') as b:
        m = b.get_metadata(); m['compression'] = 'zstd_from_the_future'; b.set_metadata(m)
    import importlib
    combine_mod = importlib.import_module('cfdb.combine')   # `from cfdb import combine` is the function
    with pytest.raises(ValueError) as err:
        combine_mod._open_inputs([good, bad])
    # err's traceback keeps the failed frame alive: a leaked handle would still be open here
    opened = [v for f in _frames(err.value.__traceback__) for v in f.f_locals.values()
              if isinstance(v, cfdb.main.Dataset)]
    assert opened and all(getattr(d._blt, '_file', None) is None or d._blt._file.closed for d in opened)


def _frames(tb):
    while tb is not None:
        yield tb.tb_frame
        tb = tb.tb_next
