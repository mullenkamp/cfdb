"""
A stepped coordinate must REFUSE an append that does not land on its own grid.

The datetime branch of ``utils._generate_step_fill`` did not. It computed
``gap = int((end - start) / dt_step)`` -- truncating -- and then asserted
``np.isclose(gap, round(gap))`` on the already-truncated integer, which is true for every input.
The float branch checks before rounding and the integer branch uses modulo; only the datetime
branch, the one every time axis uses, was vacuous.

The consequence was silent: an off-grid value was accepted onto an axis that went on reporting
its declared step, so the axis said "uniform 6-hourly" while holding 7 h and 1 h gaps. Nothing
downstream re-derives an axis from its step, so nothing would ever have noticed.
"""
import numpy as np
import pytest

import cfdb


def _stepped_time_axis(path, step_minutes=360):
    with cfdb.open_dataset(path, flag='n') as ds:
        ds.create.coord.time(data=np.array(['2026-08-24T22:00'], dtype='datetime64[m]'), step=step_minutes)


def test_offgrid_datetime_append_is_refused(tmp_path):
    """+7 h onto a 6-hourly axis. Previously accepted."""
    path = str(tmp_path / 'a.cfdb')
    _stepped_time_axis(path)
    with cfdb.open_dataset(path, flag='w') as ds:
        with pytest.raises(ValueError, match='not a multiple of the step'):
            ds['time'].append(np.array(['2026-08-25T05:00'], dtype='datetime64[m]'))


def test_offgrid_append_does_not_corrupt_a_declared_step(tmp_path):
    """The observable damage: a non-uniform axis that still declares step=360."""
    path = str(tmp_path / 'b.cfdb')
    _stepped_time_axis(path)
    with cfdb.open_dataset(path, flag='w') as ds, pytest.raises(ValueError):
        ds['time'].append(np.array(['2026-08-25T05:00'], dtype='datetime64[m]'))
    with cfdb.open_dataset(path) as ds:
        values = np.asarray(ds['time'].data)
        gaps = np.diff(values).astype('timedelta64[m]').astype(int)
        assert len(set(gaps.tolist())) <= 1, f'axis is non-uniform but declares a step: gaps={gaps}'


def test_ongrid_datetime_append_still_autofills(tmp_path):
    """The behaviour that must survive the fix: a legitimate gap back-fills."""
    path = str(tmp_path / 'c.cfdb')
    _stepped_time_axis(path)
    with cfdb.open_dataset(path, flag='w') as ds:
        ds['time'].append(np.array(['2026-08-25T10:00'], dtype='datetime64[m]'))  # +12 h
    with cfdb.open_dataset(path) as ds:
        values = np.asarray(ds['time'].data)
        assert values.size == 3, values          # 22:00, 04:00 auto-filled, 10:00
        gaps = np.diff(values).astype('timedelta64[m]').astype(int)
        assert set(gaps.tolist()) == {360}, gaps


def test_offgrid_datetime_prepend_is_refused(tmp_path):
    """The prepend call site shares the branch."""
    path = str(tmp_path / 'd.cfdb')
    with cfdb.open_dataset(path, flag='n') as ds:
        ds.create.coord.time(data=np.array(['2026-08-25T10:00'], dtype='datetime64[m]'), step=360)
    with cfdb.open_dataset(path, flag='w') as ds:
        with pytest.raises(ValueError, match='not a multiple of the step'):
            ds['time'].prepend(np.array(['2026-08-25T03:00'], dtype='datetime64[m]'))  # -7 h


def test_minute_axis_offgrid_by_one_minute(tmp_path):
    """Truncation hid sub-step errors best at fine units: 359 min onto a 360 min step."""
    path = str(tmp_path / 'e.cfdb')
    _stepped_time_axis(path)
    with cfdb.open_dataset(path, flag='w') as ds:
        with pytest.raises(ValueError, match='not a multiple of the step'):
            ds['time'].append(np.array(['2026-08-25T03:59'], dtype='datetime64[m]'))
