"""
Tests for the ts_forecast / grid_forecast dataset types.

Every failure mode these cover is SILENT in the pre-change code -- a wrong transpose, a
mislabelled CF axis, a fabricated standard_name, a lock that is never released. They are
written to pin the specific behaviour, not to exercise the happy path.
"""
import numpy as np
import pytest
import shapely

import cfdb
from cfdb import dtypes
from cfdb.main import Grid, GridForecast, TimeSeriesForecast, TimeSeriesOrtho

FRT = np.array(['2024-01-01T00', '2024-01-01T03', '2024-01-01T06'], dtype='datetime64[m]')
LEAD = np.arange(1, 5, dtype='int32')


def _ts_forecast(path, frt=FRT, step=180):
    """(point, forecast_reference_time, forecast_period)."""
    ds = cfdb.open_dataset(path, flag='n', dataset_type='ts_forecast')
    ds.create.coord.point(data=np.array([shapely.Point(172.0, -43.5),
                                         shapely.Point(171.5, -43.0)], dtype='O'))
    ds.create.coord.forecast_reference_time(data=frt, step=step)
    pc = ds.create.coord.forecast_period(data=LEAD, step=1)
    pc.attrs['units'] = 'h'
    ds.create.data_var.generic(
        'precipitation', ('point', 'forecast_reference_time', 'forecast_period'),
        dtype=dtypes.dtype('float32', precision=1, min_value=0, max_value=1000),
        chunk_shape=(2, 1, len(LEAD)))
    return ds


def _grid_forecast(path):
    """(x, y, forecast_reference_time, forecast_period)."""
    ds = cfdb.open_dataset(path, flag='n', dataset_type='grid_forecast')
    ds.create.coord.x(data=np.array([170.0, 170.5, 171.0], dtype='float32'), step=0.5)
    ds.create.coord.y(data=np.array([-44.0, -43.5], dtype='float32'), step=0.5)
    ds.create.coord.forecast_reference_time(data=FRT, step=180)
    pc = ds.create.coord.forecast_period(data=LEAD, step=1)
    pc.attrs['units'] = 'h'
    ds.create.data_var.generic(
        'precipitation', ('x', 'y', 'forecast_reference_time', 'forecast_period'),
        dtype=dtypes.dtype('float32', precision=1, min_value=0, max_value=1000),
        chunk_shape=(3, 2, 1, len(LEAD)))
    return ds


# --------------------------------------------------------------------------------------
# Round-trip: the stored type must pick the class on reopen, with no dataset_type passed.
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize('builder,dtype_name,klass', [
    (_ts_forecast, 'ts_forecast', TimeSeriesForecast),
    (_grid_forecast, 'grid_forecast', GridForecast),
])
def test_forecast_roundtrip_class_and_type(tmp_path, builder, dtype_name, klass):
    p = tmp_path / 'f.cfdb'
    with builder(str(p)) as ds:
        assert ds.dataset_type == dtype_name
        assert isinstance(ds, klass)
    with cfdb.open_dataset(str(p)) as ds:          # NOTE: no dataset_type= on reopen
        assert isinstance(ds, klass)
        assert ds.dataset_type == dtype_name


def test_forecast_types_are_distinct_classes():
    """A forecast type must not silently reuse the non-forecast class."""
    for a, b in [(TimeSeriesForecast, TimeSeriesOrtho), (GridForecast, Grid)]:
        assert a is not b and not issubclass(a, b)


# --------------------------------------------------------------------------------------
# D1 -- axis assignment. frt takes 'T'; forecast_period deliberately takes none.
# --------------------------------------------------------------------------------------

def test_axis_assignment_and_uniqueness(tmp_path):
    p = tmp_path / 'a.cfdb'
    with _ts_forecast(str(p)) as ds:
        assert ds['forecast_reference_time'].axis.value == 't'
        assert ds['forecast_period'].axis is None, \
            "forecast_period must carry NO axis -- CF defines only X/Y/Z/T"
        # a second 't' coordinate must still be refused
        with pytest.raises(ValueError, match='axis'):
            ds.create.coord.time(data=FRT, step=180)


def test_forecast_period_declares_units(tmp_path):
    """
    Load-bearing, not decoration: cfdb has no timedelta dtype, so lead is a bare integer and
    `frt[-1] + lead.max()` evaluates in the DATETIME's unit. Any consumer doing that
    arithmetic must read this attr; without it there is nothing to read.
    """
    p = tmp_path / 'u.cfdb'
    with _ts_forecast(str(p)) as ds:
        assert ds['forecast_period'].attrs['units'] == 'h'
        assert ds['forecast_reference_time'].attrs['standard_name'] == 'forecast_reference_time'


def test_forecast_period_units_is_NOT_defaulted(tmp_path):
    """
    Deliberate: a default would be silently wrong for any producer whose leads are not hourly,
    and it would make the downstream 'units must be declared' check unreachable, because the
    attribute would never be absent. Absent-by-default converts that into a loud refusal.
    """
    p = tmp_path / 'nd.cfdb'
    with cfdb.open_dataset(str(p), flag='n', dataset_type='ts_forecast') as ds:
        pc = ds.create.coord.forecast_period(data=LEAD, step=1)
        assert 'units' not in pc.attrs


def test_naive_lead_arithmetic_is_wrong_at_minute_resolution(tmp_path):
    """
    Pins the trap itself. Written at datetime64[m] ON PURPOSE -- at hour resolution the naive
    expression is accidentally correct and this test would pass while production broke.
    """
    p = tmp_path / 'n.cfdb'
    with _ts_forecast(str(p)) as ds:
        frt = ds['forecast_reference_time'].data
        lead = ds['forecast_period'].data
        unit = ds['forecast_period'].attrs['units']
    naive = frt[-1] + lead.max()                          # adds MINUTES
    correct = frt[-1] + np.timedelta64(int(lead.max()), unit)
    assert naive != correct
    assert naive == np.datetime64('2024-01-01T06:04', 'm')
    assert correct == np.datetime64('2024-01-01T10:00', 'm')


# --------------------------------------------------------------------------------------
# D4 -- interp must raise, not return a silently wrong GridInterp.
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize('builder', [_ts_forecast, _grid_forecast])
def test_interp_raises_on_forecast_types(tmp_path, builder):
    p = tmp_path / 'i.cfdb'
    with builder(str(p)) as ds:
        ds.create.crs.from_user_input(
            4326, **({'xy_coord': 'point'} if ds.dataset_type == 'ts_forecast'
                     else {'x_coord': 'x', 'y_coord': 'y'}))
        with pytest.raises(NotImplementedError, match='interp is not implemented'):
            ds['precipitation'].interp()


# --------------------------------------------------------------------------------------
# D5 -- the dtype guard must cover grid_forecast, not just grid.
# --------------------------------------------------------------------------------------

def test_grid_forecast_rejects_geometry_coord(tmp_path):
    p = tmp_path / 'g.cfdb'
    with cfdb.open_dataset(str(p), flag='n', dataset_type='grid_forecast') as ds:
        with pytest.raises(TypeError, match='Geometry dtype'):
            ds.create.coord.point(data=np.array([shapely.Point(172.0, -43.5)], dtype='O'))


def test_ts_forecast_rejects_independent_xy(tmp_path):
    """Inherited for free via the 'ts_' substring check -- asserted so a refactor cannot lose it."""
    p = tmp_path / 't.cfdb'
    with cfdb.open_dataset(str(p), flag='n', dataset_type='ts_forecast') as ds:
        with pytest.raises(TypeError, match='independent lat/y and lon/x'):
            ds.create.coord.lat(data=np.array([-43.5], dtype='float64'))


# --------------------------------------------------------------------------------------
# D5 -- featureType is NOT extended to ts_forecast (not a valid CF DSG).
# --------------------------------------------------------------------------------------

def test_ts_forecast_has_no_featuretype(tmp_path):
    p = tmp_path / 'ft.cfdb'
    with _ts_forecast(str(p)) as ds:
        assert 'featureType' not in ds.attrs
    with cfdb.open_dataset(str(tmp_path / 'o.cfdb'), flag='n', dataset_type='ts_ortho') as ds:
        assert ds.attrs['featureType'] == 'timeSeries'   # unchanged for ts_ortho


# --------------------------------------------------------------------------------------
# D9 -- netCDF export must not clobber an explicit standard_name.
# --------------------------------------------------------------------------------------

def test_netcdf_export_preserves_forecast_reference_time_standard_name(tmp_path):
    h5netcdf = pytest.importorskip('h5netcdf')
    pytest.importorskip('h5py')
    # grid_forecast, NOT ts_forecast: exporting any geometry coordinate hits a PRE-EXISTING
    # cfdb bug ("'Point' object has no attribute '_factor'", main.py:606) that ts_forecast
    # inherits from ts_ortho and that this change deliberately does not fix. Using the grid
    # variant keeps this test pointed at the standard_name behaviour it exists to pin.
    p, nc = tmp_path / 'e.cfdb', tmp_path / 'e.nc'
    with _grid_forecast(str(p)) as ds:
        ds.create.crs.from_user_input(4326, x_coord='x', y_coord='y')
        ds.to_netcdf4(str(nc))
    with h5netcdf.File(str(nc), 'r') as f:
        sn = f['forecast_reference_time'].attrs['standard_name']
        sn = sn.decode() if isinstance(sn, bytes) else str(sn)
        assert sn == 'forecast_reference_time', \
            "export clobbered the init axis with standard_name='time'"


# --------------------------------------------------------------------------------------
# D8 -- an explicit step is what makes recovery of a MISSED run possible.
# --------------------------------------------------------------------------------------

def test_missed_run_autofills_and_can_be_recovered(tmp_path):
    base, recov = tmp_path / 'b.cfdb', tmp_path / 'r.cfdb'
    early = np.array(['2024-01-01T00', '2024-01-01T03'], dtype='datetime64[m]')
    with _ts_forecast(str(base), frt=early, step=180) as ds:
        ds['forecast_reference_time'].append(
            np.array(['2024-01-01T09'], dtype='datetime64[m]'))       # 06:00 skipped
    with cfdb.open_dataset(str(base)) as ds:
        axis = ds['forecast_reference_time'].data
    assert np.datetime64('2024-01-01T06', 'm') in axis, 'step should auto-fill the missed init'

    with _ts_forecast(str(recov), frt=np.array(['2024-01-01T06'], dtype='datetime64[m]'),
                      step=180) as ds:
        ds['precipitation'][:, 0, :] = np.full((2, len(LEAD)), 9.0, dtype='f4')
    cfdb.merge_into([str(recov)], str(base), allow_expansion=['forecast_reference_time'])
    with cfdb.open_dataset(str(base)) as ds:
        i = list(ds['forecast_reference_time'].data).index(np.datetime64('2024-01-01T06', 'm'))
        assert np.nanmax(ds['precipitation'][:, i, :].data) == pytest.approx(9.0)


def test_step_true_infers_nothing_from_a_single_value(tmp_path):
    """
    The trap: auto-detect needs >=2 values, and the FIRST EVER forecast run creates a
    one-value axis. The dataset then looks fine until a run is missed, at which point
    recovery is impossible. This is why the plan requires an EXPLICIT step.
    """
    p = tmp_path / 's.cfdb'
    one = np.array(['2024-01-01T00'], dtype='datetime64[m]')
    with _ts_forecast(str(p), frt=one, step=True) as ds:
        assert ds['forecast_reference_time'].step is None
    with _ts_forecast(str(tmp_path / 's2.cfdb'), frt=one, step=180) as ds:
        assert ds['forecast_reference_time'].step == 180


# --------------------------------------------------------------------------------------
# combine / merge type agreement.
# --------------------------------------------------------------------------------------

def test_combine_same_forecast_type_ok_mixed_raises(tmp_path):
    a, b, c = tmp_path / 'c1.cfdb', tmp_path / 'c2.cfdb', tmp_path / 'c3.cfdb'
    _ts_forecast(str(a)).close()
    _ts_forecast(str(b)).close()
    cfdb.combine([str(a), str(b)], str(tmp_path / 'out.cfdb')).close()
    _grid_forecast(str(c)).close()
    with pytest.raises(ValueError, match='dataset_type'):
        cfdb.combine([str(a), str(c)], str(tmp_path / 'bad.cfdb'))


# --------------------------------------------------------------------------------------
# D7 -- an unknown dataset_type must raise WITHOUT leaking the open store.
# --------------------------------------------------------------------------------------

def test_unknown_dataset_type_raises_without_holding_the_file(tmp_path):
    """
    The pre-change code raised after booklet was already open and never closed it. The
    leaked handle holds an OS lock, so the symptom is not a warning -- the next write-open
    BLOCKS. Reopening for write here is the assertion.
    """
    p = tmp_path / 'bad.cfdb'
    with pytest.raises(TypeError, match='dataset_type must be one of'):
        cfdb.open_dataset(str(p), flag='n', dataset_type='not_a_real_type')
    with _ts_forecast(str(p)) as ds:            # would hang if the lock were still held
        assert ds.dataset_type == 'ts_forecast'


# --------------------------------------------------------------------------------------
# Regression tests for mutants that SURVIVED the ecan-theta-code-1 review.
# Both changes below were correct but unfalsifiable -- nothing in the suite covered them.
# --------------------------------------------------------------------------------------

def test_netcdf_export_preserves_standard_name_on_a_datetime_DATA_VARIABLE(tmp_path):
    """
    Survivor #1: reverting the data-variable `setdefault` back to an assignment passed the
    entire 314-test suite. The coordinate branches were covered; this third branch was not.
    """
    pytest.importorskip('h5netcdf')
    pytest.importorskip('h5py')
    h5netcdf = pytest.importorskip('h5netcdf')
    p, nc = tmp_path / 'dv.cfdb', tmp_path / 'dv.nc'
    with _grid_forecast(str(p)) as ds:
        ds.create.crs.from_user_input(4326, x_coord='x', y_coord='y')
        # NOTE the ENCODED dtype. An unencoded datetime takes the sibling branch, where the
        # standard_name line is commented out entirely -- a test built on that path passes no
        # matter what this branch does, which is exactly how the mutant survived the first time.
        dv = ds.create.data_var.generic(
            'run_started_at',
            ('forecast_reference_time',),
            dtype=dtypes.dtype('datetime64[m]', dtype_encoded='int32', offset=-36816481),
        )
        dv.attrs['standard_name'] = 'forecast_reference_time'
        dv[:] = FRT
        ds.to_netcdf4(str(nc))
    with h5netcdf.File(str(nc), 'r') as f:
        sn = f['run_started_at'].attrs['standard_name']
        sn = sn.decode() if isinstance(sn, bytes) else str(sn)
        assert sn == 'forecast_reference_time', \
            'export clobbered a datetime DATA VARIABLE with standard_name=time'


@pytest.mark.parametrize('dtype_name,klass_name', [
    ('grid', 'EGrid'),
    ('ts_ortho', 'ETimeSeriesOrtho'),
    ('ts_forecast', 'ETimeSeriesForecast'),
    ('grid_forecast', 'EGridForecast'),
])
def test_edataset_dispatch_maps_each_type_to_its_own_class(monkeypatch, tmp_path,
                                                           dtype_name, klass_name):
    """
    Survivor #2: routing ts_forecast to ETimeSeriesOrtho(..., 'ts_ortho') also passed the full
    suite -- no local test constructs an EDataset. That mutant would persist an S3-created
    forecast dataset with the WRONG stored dataset_type, which is not recoverable by reopening.

    ebooklet is stubbed so the dispatch can be exercised without S3; the assertion is on the
    class chosen and the type string handed to it, which is what gets written to the file.
    """
    from cfdb import edataset as ed

    captured = {}

    class _FakeBlt:
        writable = True
        def get_metadata(self):
            return None
        def close(self):
            pass

    def _fake_open_ebooklet(*args, **kwargs):
        return _FakeBlt()

    def _fake_init(self, fp, blt, create, compression, compression_level, dataset_type):
        captured['class'] = type(self).__name__
        captured['dataset_type'] = dataset_type

    monkeypatch.setattr(ed.ebooklet, 'open_ebooklet', _fake_open_ebooklet)
    monkeypatch.setattr(ed.EDataset, '__init__', _fake_init)

    ed.open_edataset(object(), str(tmp_path / 'e.cfdb'), flag='n', dataset_type=dtype_name)

    assert captured['class'] == klass_name
    assert captured['dataset_type'] == dtype_name, \
        'the type string passed to __init__ is what gets PERSISTED -- it must match'
