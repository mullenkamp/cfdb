"""
Regression tests for the groupby period round (F3+F4, 2026-07-21):
- irregular (M/Y) groups on a SLICED variable view are computed from the view's
  own coordinate window (F3)
- the fast path requires the first coordinate value on the period boundary,
  falling back to calendar groups otherwise (F4)
- 'W' anchors to epoch weeks (Thursdays) on both paths; '7D' anchors at coord[0]
- dataset- and variable-level groupby agree
"""
import numpy as np
import pytest

import cfdb
from cfdb import dtypes
from cfdb import support_classes


def build_daily(tmp_path, n_days=182, start='2020-01-01'):
    path = tmp_path / 'd.cfdb'
    t = np.arange(start, np.datetime64(start) + np.timedelta64(n_days, 'D'), dtype='datetime64[D]')
    with cfdb.open_dataset(path, flag='n') as ds:
        ds.create.coord.time(data=t, dtype='datetime64[D]', step=True)
        dv = ds.create.data_var.generic(name='v', coords=('time',), dtype=dtypes.dtype('float64'), chunk_shape=(50,))
        dv[:] = np.arange(float(n_days))
    return path


def build_hourly(tmp_path, start, n_hours, step=True):
    path = tmp_path / f'h_{step}.cfdb'
    t = np.arange(start, np.datetime64(start) + np.timedelta64(n_hours, 'h'), dtype='datetime64[h]')
    with cfdb.open_dataset(path, flag='n') as ds:
        ds.create.coord.time(data=t, dtype='datetime64[h]', step=step)
        dv = ds.create.data_var.generic(name='v', coords=('time',), dtype=dtypes.dtype('float64'), chunk_shape=(24,))
        dv[:] = np.arange(float(n_hours))
    return path


def groups_of(gen):
    return [(sl[0] if isinstance(sl, tuple) else sl, d.copy()) for sl, d in gen]


###################################################
### F3: irregular periods on views

def test_view_monthly_groups_match_calendar(tmp_path):
    path = build_daily(tmp_path)
    with cfdb.open_dataset(path) as ds:
        view = ds['v'][60:182]                     # 2020-03-01 .. 2020-06-30
        got = groups_of(view.groupby({'time': 'M'}))
        sizes = [d.size for _s, d in got]
        starts = [d[0] for _s, d in got]
        assert sizes == [31, 30, 31, 30]           # Mar Apr May Jun
        assert starts == [60.0, 91.0, 121.0, 152.0]
        # groups must equal slicing the full-var groups to the view window
        full = groups_of(ds['v'].groupby({'time': 'M'}))
        window = [d for _s, d in full if d[0] >= 60.0]
        assert all(np.array_equal(a[1], b) for a, b in zip(got, window))


def test_dataset_view_monthly_groups_still_correct(tmp_path):
    path = build_daily(tmp_path)
    with cfdb.open_dataset(path) as ds:
        dsv = ds.select({'time': slice(60, 182)})
        got = [(tc['time'], vd['v'].copy()) for tc, vd in dsv.groupby({'time': 'M'})]
        assert [d.size for _s, d in got] == [31, 30, 31, 30]
        assert [d[0] for _s, d in got] == [60.0, 91.0, 121.0, 152.0]


def test_variable_and_dataset_groupby_agree(tmp_path):
    path = build_daily(tmp_path)
    with cfdb.open_dataset(path) as ds:
        var_groups = [d for _s, d in groups_of(ds['v'].groupby({'time': 'M'}))]
        ds_groups = [vd['v'] for _tc, vd in ds.groupby({'time': 'M'})]
        assert len(var_groups) == len(ds_groups)
        assert all(np.array_equal(a, b) for a, b in zip(var_groups, ds_groups))


###################################################
### F4: anchor check

def test_midday_start_daily_matches_calendar(tmp_path):
    """06:00-start hourly data with step set: 'D' must give calendar days, not
    fixed 24h windows spanning two days each."""
    path = build_hourly(tmp_path, '2020-01-01T06', 72, step=True)
    with cfdb.open_dataset(path) as ds:
        got = groups_of(ds['v'].groupby({'time': 'D'}))
        sizes = [d.size for _s, d in got]
        assert sizes == [18, 24, 24, 6]            # calendar days
    # identical to the step=None fallback result
    path2 = build_hourly(tmp_path, '2020-01-01T06', 72, step=False)
    with cfdb.open_dataset(path2) as ds:
        fallback = groups_of(ds['v'].groupby({'time': 'D'}))
        assert [d.size for _s, d in fallback] == sizes


def test_midnight_start_keeps_fast_path(tmp_path, monkeypatch):
    """Boundary-anchored data keeps the rechunker fast path with unchanged slices."""
    path = build_hourly(tmp_path, '2020-01-01T00', 96, step=True)
    with cfdb.open_dataset(path) as ds:
        def boom(*args, **kwargs):
            raise AssertionError('irregular path must not run for anchored data')
        monkeypatch.setattr(support_classes.DataVariableView, '_groupby_period', boom)
        got = groups_of(ds['v'].groupby({'time': 'D'}))
        assert [(s.start, s.stop) for s, _d in got] == [(0, 24), (24, 48), (48, 72), (72, 96)]


def test_7d_anchors_at_first_value_both_paths(tmp_path):
    """'7D' anchors at coord[0]'s day on both the fast path and the fallback."""
    p1 = build_daily(tmp_path, n_days=30, start='2020-01-06')   # a Monday
    with cfdb.open_dataset(p1) as ds:
        sizes = [d.size for _s, d in groups_of(ds['v'].groupby({'time': '7D'}))]
        assert sizes == [7, 7, 7, 7, 2]


def test_weekly_pins_thursday_anchoring(tmp_path):
    """'W' truncation anchors to epoch weeks (Thursday boundaries) — consistent
    across paths but distinct from '7D'; pinned here on Monday-start data."""
    p1 = build_daily(tmp_path, n_days=30, start='2020-01-06')   # Monday
    with cfdb.open_dataset(p1) as ds:
        sizes = [d.size for _s, d in groups_of(ds['v'].groupby({'time': 'W'}))]
        assert sizes == [3, 7, 7, 7, 6]            # Mon..Wed, then Thu-anchored weeks


def test_hourly_6h_offset_start_consistent_across_paths(tmp_path):
    """Multi-count periods anchor at coord[0] (compute_time_groups semantics):
    '6h' from 03:00 gives 6-hour windows from 03:00 on BOTH paths — the fix's
    requirement is fast == fallback, and coord[0] sits on the 'h' unit boundary."""
    path = build_hourly(tmp_path, '2020-01-01T03', 24, step=True)
    with cfdb.open_dataset(path) as ds:
        fast = [d.size for _s, d in groups_of(ds['v'].groupby({'time': '6h'}))]
    path2 = build_hourly(tmp_path, '2020-01-01T03', 24, step=False)
    with cfdb.open_dataset(path2) as ds:
        fallback = [d.size for _s, d in groups_of(ds['v'].groupby({'time': '6h'}))]
    assert fast == fallback == [6, 6, 6, 6]


###################################################
### Guards

def test_dataset_groupby_empty_data_vars_raises(tmp_path):
    path = build_daily(tmp_path, n_days=40)
    with cfdb.open_dataset(path) as ds:
        with pytest.raises(ValueError, match='No data variables'):
            list(ds.groupby({'time': 'M'}, data_vars=[]))
