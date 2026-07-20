"""
Regression tests for the encoded missing-chunk fill round (F1, 2026-07-21):
every read path must agree on what a MISSING chunk contains, in-session AND
after close/reopen, for every dtype encoding — plus the ride-along fixes in
the same family (encode width sizing, netCDF export of packed ints and
encoded datetimes, combine old/new interop, the legacy-dtype warning).
"""
import warnings

import numpy as np
import pytest

import cfdb
from cfdb import dtypes

###################################################
### Helpers

N = 30
CHUNK = (10,)


DTYPE_SPECS = {
    'plain_f8': dict(name='float64'),
    'float_auto': dict(name='float32', precision=2, min_value=-100, max_value=100),
    'float_explicit_fv': dict(name='float32', precision=1, dtype_encoded='uint16', offset=50.0, fillvalue=7),
    'int_min_neg': dict(name='int32', min_value=-10, max_value=1000),
    'int_min_one': dict(name='int32', min_value=1, max_value=1000),
    'int_min_five': dict(name='int32', min_value=5, max_value=1000),
    'datetime_enc': dict(name='datetime64[h]', min_value='2020-01-01', max_value='2021-01-01'),
}


def sample_values(spec_key):
    """Ten in-range values for the first chunk of each dtype spec."""
    if spec_key == 'plain_f8':
        return np.arange(10, dtype='float64') + 0.5
    if spec_key == 'float_auto':
        return np.linspace(-99.0, 99.0, 10).round(2).astype('float32')
    if spec_key == 'float_explicit_fv':
        return np.linspace(60.0, 90.0, 10).round(1).astype('float32')
    if spec_key == 'int_min_neg':
        return np.arange(-10, 0, dtype='int32')
    if spec_key == 'int_min_one':
        return np.arange(1, 11, dtype='int32')
    if spec_key == 'int_min_five':
        return np.arange(5, 15, dtype='int32')
    if spec_key == 'datetime_enc':
        return np.arange('2020-02-01T00', '2020-02-01T10', dtype='datetime64[h]')
    raise KeyError(spec_key)


def expected_missing(spec_key):
    """The decoded blank each dtype must show for missing data on EVERY path."""
    if spec_key.startswith('int'):
        return 0
    if spec_key == 'datetime_enc':
        return np.datetime64('NaT')
    return np.nan


def build(path, spec_key):
    """Dataset with one 1D var: first chunk written, chunks [10:30) missing."""
    with cfdb.open_dataset(path, flag='n') as ds:
        ds.create.coord.generic(name='x', data=np.arange(0.0, N), chunk_shape=CHUNK, step=1.0)
        dv = ds.create.data_var.generic(name='v', coords=('x',), dtype=dtypes.dtype(**DTYPE_SPECS[spec_key]),
                                        chunk_shape=CHUNK)
        dv[0:10] = sample_values(spec_key)


def read_all_paths(ds, spec_key):
    """Read v through every path; return dict of full-length arrays."""
    dv = ds['v']
    out = {'data': dv.data}

    rechunked = np.empty(N, dtype=dv.dtype.dtype_decoded)
    for wc, d in dv.rechunker().rechunk((15,)):
        rechunked[wc] = d
    out['rechunk'] = rechunked

    itered = np.empty(N, dtype=dv.dtype.dtype_decoded)
    for wc, d in dv.iter_chunks(chunk_shape={'x': 7}):
        itered[wc] = d
    out['iter_chunks'] = itered

    grouped = np.empty(N, dtype=dv.dtype.dtype_decoded)
    for wc, d in dv.groupby({'x': 12}):
        grouped[wc] = d
    out['groupby'] = grouped
    return out


def assert_missing_consistent(arrays, spec_key):
    missing = expected_missing(spec_key)
    written = sample_values(spec_key)
    for path_name, arr in arrays.items():
        if spec_key == 'float_auto':
            assert np.allclose(arr[:10].astype('float64'), written.astype('float64'), atol=0.006), path_name
        elif spec_key == 'float_explicit_fv':
            assert np.allclose(arr[:10].astype('float64'), written.astype('float64'), atol=0.06), path_name
        else:
            assert np.array_equal(arr[:10], written), path_name

        tail = arr[10:]
        if isinstance(missing, float) and np.isnan(missing):
            assert np.isnan(tail).all(), (path_name, tail[:3])
        elif isinstance(missing, np.datetime64) and np.isnat(missing):
            assert np.isnat(tail).all(), (path_name, tail[:3])
        else:
            assert (tail == missing).all(), (path_name, tail[:3])


###################################################
### Fill-consistency matrix

@pytest.mark.parametrize('spec_key', list(DTYPE_SPECS))
def test_missing_chunk_consistency_in_session(tmp_path, spec_key):
    path = tmp_path / 'f.cfdb'
    build(path, spec_key)
    with cfdb.open_dataset(path, flag='w') as ds:
        # same session that could have created it: re-create then read
        assert_missing_consistent(read_all_paths(ds, spec_key), spec_key)


@pytest.mark.parametrize('spec_key', list(DTYPE_SPECS))
def test_missing_chunk_consistency_after_reopen(tmp_path, spec_key):
    """The C1 regression: the fillvalue mapping must survive close/reopen
    (dtypes are reconstructed from the stored dict through the explicit branch)."""
    path = tmp_path / 'f.cfdb'
    build(path, spec_key)
    with cfdb.open_dataset(path) as ds:
        if spec_key.startswith('int'):
            assert ds['v'].dtype.fillvalue == 0
        assert_missing_consistent(read_all_paths(ds, spec_key), spec_key)


###################################################
### Legitimate zero must never alias the missing marker (Gemini R1 CRITICAL)

def test_legit_zero_min_neg_encodes_above_fill(tmp_path):
    dt = dtypes.dtype('int32', min_value=-10, max_value=1000)
    assert dt.fillvalue == 0
    encoded = dt.encode(np.array([0, -10, 1000], dtype='int32'))
    assert encoded[0] == 11          # 0 - offset = 0 - (-11) = 11, never the fill code
    assert (encoded > 0).all()
    assert np.array_equal(dt.decode(encoded), [0, -10, 1000])


def test_legit_zero_survives_write_read_everywhere(tmp_path):
    path = tmp_path / 'z.cfdb'
    with cfdb.open_dataset(path, flag='n') as ds:
        ds.create.coord.generic(name='x', data=np.arange(0.0, N), chunk_shape=CHUNK, step=1.0)
        dv = ds.create.data_var.generic(name='v', coords=('x',),
                                        dtype=dtypes.dtype('int32', min_value=-10, max_value=1000),
                                        chunk_shape=CHUNK)
        dv[0:10] = np.zeros(10, dtype='int32')       # legitimate zeros
    with cfdb.open_dataset(path) as ds:
        arrays = read_all_paths(ds, 'int_min_neg')
        for path_name, arr in arrays.items():
            assert (arr[:10] == 0).all(), path_name  # written zeros
            assert (arr[10:] == 0).all(), path_name  # missing also reads 0 (by ruling)


###################################################
### Encode/decode boundaries at full-width ranges (M4)

def test_full_width_range_boundaries_roundtrip():
    # 0..255 with the old sizing landed on uint8 and made 255 unencodable;
    # correct sizing must pick a width where declared min AND max round-trip.
    dt_int = dtypes.dtype('int32', min_value=0, max_value=255)
    vals = np.array([0, 1, 254, 255], dtype='int32')
    assert np.array_equal(dt_int.decode(dt_int.encode(vals)), vals)

    dt_f = dtypes.dtype('float32', precision=0, min_value=0, max_value=255)
    fvals = np.array([0.0, 255.0], dtype='float32')
    out = dt_f.decode(dt_f.encode(fvals))
    assert not np.isnan(out).any()
    assert np.allclose(out, fvals)


@pytest.mark.parametrize('min_value,max_value', [(0, 255), (0, 65535), (-5, 250), (1, 256)])
def test_declared_extremes_always_encodable(min_value, max_value):
    dt = dtypes.dtype('int64', min_value=min_value, max_value=max_value)
    vals = np.array([min_value, max_value], dtype='int64')
    assert np.array_equal(dt.decode(dt.encode(vals)), vals)


###################################################
### Partial writes (encoded-space overlay)

def test_partial_write_int_min_five(tmp_path):
    """Fail-before: blank cells of a partial chunk hit the encode range raise (min >= 2)."""
    path = tmp_path / 'p.cfdb'
    with cfdb.open_dataset(path, flag='n') as ds:
        ds.create.coord.generic(name='x', data=np.arange(0.0, N), chunk_shape=CHUNK, step=1.0)
        dv = ds.create.data_var.generic(name='v', coords=('x',),
                                        dtype=dtypes.dtype('int32', min_value=5, max_value=1000),
                                        chunk_shape=CHUNK)
        dv[3:7] = np.arange(50, 54, dtype='int32')    # partial chunk write
    with cfdb.open_dataset(path) as ds:
        d = ds['v'].data
        assert np.array_equal(d[3:7], np.arange(50, 54))
        assert (d[:3] == 0).all() and (d[7:] == 0).all()
        # unwritten cells must still be the encoded fill on disk (honestly missing)
        raw = ds['v']._get_encoded_chunk((slice(0, 10),))
        assert (raw[:3] == 0).all() and (raw[7:] == 0).all()
        assert (raw[3:7] > 0).all()


def test_partial_write_equivalence_float_datetime(tmp_path):
    """Encoded-space overlay must produce identical values to whole-array writes,
    including NaN/NaT handling."""
    path = tmp_path / 'e.cfdb'
    with cfdb.open_dataset(path, flag='n') as ds:
        ds.create.coord.generic(name='x', data=np.arange(0.0, N), chunk_shape=CHUNK, step=1.0)
        vf = ds.create.data_var.generic(name='vf', coords=('x',),
                                        dtype=dtypes.dtype('float32', precision=2, min_value=-100, max_value=100),
                                        chunk_shape=CHUNK)
        vals = np.linspace(-50, 50, 8).round(2).astype('float32')
        vals[3] = np.nan
        vf[4:12] = vals                                # spans two chunks, partial both
        vt = ds.create.data_var.generic(name='vt', coords=('x',),
                                        dtype=dtypes.dtype('datetime64[h]', min_value='2020-01-01', max_value='2021-01-01'),
                                        chunk_shape=CHUNK)
        tvals = np.arange('2020-03-01T00', '2020-03-01T08', dtype='datetime64[h]')
        tvals[2] = np.datetime64('NaT')
        vt[4:12] = tvals
    with cfdb.open_dataset(path) as ds:
        f = ds['vf'].data
        assert np.isnan(f[3 + 4])
        assert np.allclose(f[4:12][~np.isnan(vals)].astype('float64'),
                           vals[~np.isnan(vals)].astype('float64'), atol=0.006)
        assert np.isnan(f[:4]).all() and np.isnan(f[12:]).all()
        t = ds['vt'].data
        assert np.isnat(t[6])
        mask = ~np.isnat(tvals)
        assert np.array_equal(t[4:12][mask], tvals[mask])
        assert np.isnat(t[:4]).all() and np.isnat(t[12:]).all()


def test_int_encoded_coordinate_partial_tail_chunk(tmp_path):
    """Fail-before: an int-packed (min >= 2) coordinate with a partial last chunk hit
    the same encode raise in Coordinate._add_updated_data."""
    path = tmp_path / 'c.cfdb'
    with cfdb.open_dataset(path, flag='n') as ds:
        ds.create.coord.generic(name='xi', data=np.arange(5, 18, dtype='int32'),
                                dtype=dtypes.dtype('int32', min_value=5, max_value=1000),
                                chunk_shape=CHUNK, step=1)
        assert np.array_equal(ds['xi'].data, np.arange(5, 18))
        ds['xi'].append(np.arange(18, 22, dtype='int32'))
        assert np.array_equal(ds['xi'].data, np.arange(5, 22))
    with cfdb.open_dataset(path) as ds:
        assert np.array_equal(ds['xi'].data, np.arange(5, 22))


###################################################
### netCDF export (M1 + M2 + fill round-trip)

h5netcdf = pytest.importorskip('h5netcdf')


def test_export_int_encoded_succeeds_and_masks_missing(tmp_path):
    """Fail-before (M1): int-encoded export crashed on scale_factor = 1/None."""
    cf = tmp_path / 'x.cfdb'
    nc = tmp_path / 'x.nc'
    build(cf, 'int_min_neg')
    with cfdb.open_dataset(cf) as ds:
        ds.to_netcdf4(nc)
    with h5netcdf.File(nc, 'r') as h5:
        v = h5['v']
        assert v.attrs['_FillValue'] == 0
        assert v.attrs['scale_factor'] == 1.0
        raw = v[...]
        # packed values: written cells code >= 1, missing cells == fill (0)
        assert (raw[10:] == 0).all()
        assert (raw[:10] != 0).all()
        decoded = raw[:10] * v.attrs['scale_factor'] + v.attrs['add_offset']
        assert np.array_equal(decoded, sample_values('int_min_neg'))


def test_export_float_explicit_fillvalue_masks_missing(tmp_path):
    cf = tmp_path / 'x.cfdb'
    nc = tmp_path / 'x.nc'
    build(cf, 'float_explicit_fv')
    with cfdb.open_dataset(cf) as ds:
        ds.to_netcdf4(nc)
    with h5netcdf.File(nc, 'r') as h5:
        v = h5['v']
        assert v.attrs['_FillValue'] == 7
        raw = v[...]
        assert (raw[10:] == 7).all()      # missing exported as _FillValue
        assert (raw[:10] != 7).all()      # written values never collide with it


def test_export_encoded_datetime_ticks_match_units(tmp_path):
    """Fail-before (M2): encoded datetimes exported offset-shifted (~decades off)."""
    cf = tmp_path / 'x.cfdb'
    nc = tmp_path / 'x.nc'
    with cfdb.open_dataset(cf, flag='n') as ds:
        t = np.arange('2020-01-01T00', '2020-01-01T10', dtype='datetime64[h]')
        ds.create.coord.generic(name='time', data=t,
                                dtype=dtypes.dtype('datetime64[h]', min_value='2020-01-01', max_value='2021-01-01'),
                                chunk_shape=CHUNK, step=1)
        assert ds['time'].dtype.dtype_encoded is not None   # encoded coord under test
        dv = ds.create.data_var.generic(name='v', coords=('time',), dtype=dtypes.dtype('float64'),
                                        chunk_shape=CHUNK)
        dv[:] = np.arange(10.0)
        ds.to_netcdf4(nc)
    with h5netcdf.File(nc, 'r') as h5:
        time_var = h5['time']
        assert 'since 1970-01-01' in time_var.attrs['units']
        ticks = time_var[...]
        expected = np.arange('2020-01-01T00', '2020-01-01T10', dtype='datetime64[h]').astype('int64')
        assert np.array_equal(ticks, expected)


###################################################
### combine old/new interop (M3)

def _strip_fillvalue(path, var_names):
    """Turn a fresh file into a legacy-shaped one (stored dtype without fillvalue).

    On a reopened dataset the stored dtype is a struct; freshly created it is a dict.
    """
    with cfdb.open_dataset(path, flag='w') as ds:
        for name in var_names:
            stored = ds._sys_meta.variables[name].dtype
            if isinstance(stored, dict):
                stored.pop('fillvalue', None)
            else:
                stored.fillvalue = None


def test_combine_legacy_and_new_int_encoded(tmp_path):
    a = tmp_path / 'a.cfdb'
    b = tmp_path / 'b.cfdb'
    out = tmp_path / 'out.cfdb'
    for path, lo, hi in ((a, 0.0, 30.0), (b, 30.0, 60.0)):
        with cfdb.open_dataset(path, flag='n') as ds:
            ds.create.coord.generic(name='x', data=np.arange(lo, hi), chunk_shape=CHUNK, step=1.0)
            dv = ds.create.data_var.generic(name='v', coords=('x',),
                                            dtype=dtypes.dtype('int32', min_value=0, max_value=1000),
                                            chunk_shape=CHUNK)
            dv[:] = np.arange(30, dtype='int32')
    _strip_fillvalue(a, ['v'])                       # a becomes legacy-shaped

    new_ds = cfdb.combine([a, b], out)
    try:
        assert new_ds['v'].shape == (60,)
        # output adopts the new-style dict (fillvalue present)
        assert new_ds._sys_meta.variables['v'].dtype.get('fillvalue') == 0
        assert np.array_equal(new_ds['v'].data[30:], np.arange(30))
    finally:
        new_ds.close()


###################################################
### Legacy-dtype warning

def test_legacy_int_encoded_warns_on_rechunk(tmp_path):
    path = tmp_path / 'l.cfdb'
    build(path, 'int_min_five')
    _strip_fillvalue(path, ['v'])
    with cfdb.open_dataset(path) as ds:
        assert ds['v'].dtype.fillvalue is None       # legacy-shaped
        with pytest.warns(UserWarning, match='legacy int-encoded'):
            for _wc, _d in ds['v'].rechunker().rechunk((15,)):
                pass


def test_new_int_encoded_does_not_warn(tmp_path):
    path = tmp_path / 'n.cfdb'
    build(path, 'int_min_five')
    with cfdb.open_dataset(path) as ds:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            for _wc, _d in ds['v'].rechunker().rechunk((15,)):
                pass
