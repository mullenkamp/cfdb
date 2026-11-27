"""
Export a small slice of every live envlib dataset with `to_netcdf4`, and check the netCDF file against cfdb's own
decoded values, read back with netCDF4 (default masking) and with xarray. Real data shows what synthetic tests
miss: cfdb 0.12.1's three export fixes (geometry as WKT, None attributes, packed valid ranges) were all found
this way (2026-11-27).

Each dataset's slice is 48 time steps from a window that holds data (the end, a month back, the middle, a
quarter in), and 4 positions on every other dimension (the middle, else the start). The checks: every
dimension's size; every variable equals cfdb's decoded values (to half a packing step for a packed variable,
since cfdb may unpack in float32 and the readers in float64), integers, strings, times and geometry (WKT)
exactly; the missing-value counts; every non-None attribute; the CRS variable; and that a packed variable's
stored values lie inside its stored valid range. A slice with no data anywhere is reported VACUOUS.

It reads the live public commons (no credentials) and needs envlib, netCDF4 and xarray besides cfdb, so run it
from an environment that has envlib, for example envlib-api's:

    cd ~/git/envlib-repos/envlib-api
    uv run --with netCDF4 --with xarray python ~/git/cfdb-repos/cfdb/plans/sweep_live_netcdf.py

To test an unreleased cfdb, overlay the checkout, and rebuild it each time (uv otherwise reuses a cached build):

    uv run --with netCDF4 --with xarray --with ~/git/cfdb-repos/cfdb --refresh-package cfdb \\
        python ~/git/cfdb-repos/cfdb/plans/sweep_live_netcdf.py [dataset_id ...]

Arguments are optional dataset ids, to sweep only those. Downloads go to a temporary directory and are removed
per dataset. Exits 1 if any dataset has a problem. It has failed, as it should, on two planted defects: valid
ranges in physical units, and `add_offset` one packing step out.
"""

import pathlib
import shutil
import sys
import tempfile
import time
import traceback

import cfdb
import envlib
import netCDF4
import numpy as np
import xarray as xr

ONLY = sys.argv[1:]  # optional dataset ids
OUT = pathlib.Path(tempfile.mkdtemp(prefix='cfdb-sweep-'))
assert tuple(int(p) for p in cfdb.__version__.split('.')[:3]) >= (0, 12, 1), f'cfdb {cfdb.__version__} lacks the export fixes'
TRANSFORMED = {'valid_min', 'valid_max', 'valid_range', 'add_offset', 'scale_factor', '_FillValue', 'dtype', 'axis',
               'units', 'calendar', 'standard_name', 'grid_mapping', 'geometry_encoding'}


def as_float(a):
    a = np.ma.filled(np.ma.asarray(a).astype('float64'), np.nan) if np.ma.isMaskedArray(a) else np.asarray(a, dtype='float64')
    return a


def compare(name, var, want, ncv, xv, problems):
    """want: cfdb decoded values; ncv: netCDF4 variable; xv: xarray DataArray."""
    kind = var.dtype.kind
    got = ncv[:]
    if kind == 'G':
        w = var.dtype.encode(want).tolist()
        if list(np.ravel(got)) != list(np.ravel(w)):
            problems.append(f'{name}: WKT differs')
        if list(np.ravel(xv.values)) != list(np.ravel(w)):
            problems.append(f'{name}: xarray WKT differs')
        return 'wkt'
    if kind in ('T', 'U', 'O', 'S'):
        if [str(s) for s in np.ravel(got)] != [str(s) for s in np.ravel(want)]:
            problems.append(f'{name}: strings differ')
        if [str(s) for s in np.ravel(xv.values)] != [str(s) for s in np.ravel(want)]:
            problems.append(f'{name}: xarray strings differ')
        return 'str'
    if kind == 'M':
        g = netCDF4.num2date(np.ma.filled(got, 0), ncv.units, ncv.calendar, only_use_cftime_datetimes=False,
                             only_use_python_datetimes=True)
        g = np.array(g, dtype='datetime64[s]')
        w = want.astype('datetime64[s]')
        ok = np.array_equal(np.where(np.isnat(w), np.datetime64(0, 's'), g), np.where(np.isnat(w), np.datetime64(0, 's'), w))
        if not ok:
            problems.append(f'{name}: times differ')
        if not np.array_equal(xv.values.astype('datetime64[s]'), w, equal_nan=True):
            problems.append(f'{name}: xarray times differ')
        return 'time'
    w = as_float(want)
    g = as_float(got)
    x = as_float(xv.values)
    n_nan_w, n_nan_g, n_nan_x = int(np.isnan(w).sum()), int(np.isnan(g).sum()), int(np.isnan(x).sum())
    if n_nan_g != n_nan_w:
        problems.append(f'{name}: netCDF4 missing {n_nan_g} vs cfdb {n_nan_w}')
    if n_nan_x != n_nan_w:
        problems.append(f'{name}: xarray missing {n_nan_x} vs cfdb {n_nan_w}')
    # A packed value is known to within its packing step; cfdb may unpack in float32 (cancellation near the
    # offset), netCDF4 and xarray in float64. So compare to half a step for a packed variable.
    step = ncv.getncattr('scale_factor') if 'scale_factor' in ncv.ncattrs() else 0.0
    tol = dict(rtol=1e-6, atol=max(1e-9, 0.5 * float(step)), equal_nan=True)
    if not np.allclose(g, w, **tol):
        problems.append(f'{name}: netCDF4 values differ (max {np.nanmax(np.abs(g - w))})')
    if not np.allclose(x, w, **tol):
        problems.append(f'{name}: xarray values differ (max {np.nanmax(np.abs(x - w))})')
    return f'num nan={n_nan_w}/{w.size}'


def check_attrs(where, want, got, problems):
    for k, v in want.items():
        if v is None or k in TRANSFORMED:
            continue
        if k not in got:
            problems.append(f'{where}: attr {k} missing')
            continue
        g = got[k]
        try:
            same = bool(np.all(np.asarray(g) == np.asarray(v))) if not isinstance(v, (dict, list)) else str(g) == str(v) or list(np.ravel(g)) == list(v)
        except Exception:
            same = str(g) == str(v)
        if not same:
            problems.append(f'{where}: attr {k} = {str(g)[:40]!r}, cfdb {str(v)[:40]!r}')


def slice_for(name, n):
    if name == 'time' or name.endswith('time'):
        return slice(max(0, n - 48), n)
    mid = n // 2
    return slice(max(0, mid - 2), min(n, mid + 2))


refs = envlib.Catalogue().query()
print(f'{len(refs)} datasets; cfdb {cfdb.__version__}\n')
summary = []
for i, ref in enumerate(refs):
    if ONLY and ref.dataset_id not in ONLY:
        continue
    work = OUT / f'd{i}'
    work.mkdir(parents=True, exist_ok=True)
    problems, notes = [], {}
    t0 = time.perf_counter()
    label = ref.dataset_id
    try:
        with ref.open(file_path=work / 'ds.cfdb') as ds:
            label = f"{ds.dataset_type} {ref.dataset_id} {ds.attrs.data.get('envlib_variable', '')}/{ds.attrs.data.get('envlib_owner', '')}"
            sel = {c: slice_for(c, ds[c].shape[0]) for c in ds.coord_names}
            time_names = [c for c in ds.coord_names if c == 'time' or c.endswith('time')]
            numeric = [n for n in ds.data_var_names if ds[n].dtype.kind in 'iuf']
            # a window that holds data: a slice of all-missing values would compare nothing
            view = None
            others = [c for c in ds.coord_names if c not in time_names]
            tries = [(where, back) for where in ('mid', 'start') for back in ((0, 1, 30, 'mid', 'quarter') if time_names else (0,))]
            for where, back in tries:
                for c in others:
                    n_c = ds[c].shape[0]
                    sel[c] = slice_for(c, n_c) if where == 'mid' else slice(0, min(n_c, 4))
                if time_names:
                    n_t = ds[time_names[0]].shape[0]
                    if back == 'mid':
                        end = n_t // 2 + 24
                    elif back == 'quarter':
                        end = n_t // 4 + 24
                    else:
                        end = n_t - back * 24 * 30
                    end = min(n_t, max(48, end))
                    sel[time_names[0]] = slice(end - 48, end)
                view = ds.select(sel)
                if not numeric or any(np.isfinite(np.asarray(view[n].data, dtype='float64')).any() for n in numeric):
                    break
            else:
                problems.append('VACUOUS: no window with data was found')
            notes['window'] = str(sel.get(time_names[0])) if time_names else 'no time'
            names = list(view.coord_names) + list(view.data_var_names)
            want = {n: view[n].data for n in names}
            var_attrs = {n: dict(view[n].attrs.data) for n in names}
            dtypes_ = {n: view[n] for n in names}
            shapes = {n: view[n].shape for n in names}
            g_attrs = dict(ds.attrs.data)
            has_crs = ds.crs is not None
            view.to_netcdf4(work / 'ds.nc')
        with netCDF4.Dataset(work / 'ds.nc') as nc, xr.open_dataset(work / 'ds.nc') as xd:
            check_attrs('global', g_attrs, {k: nc.getncattr(k) for k in nc.ncattrs()}, problems)
            if has_crs and 'spatial_ref' not in nc.variables and 'crs' not in nc.variables:
                problems.append('no grid-mapping variable')
            for n in names:
                if n not in nc.variables:
                    problems.append(f'{n}: not in the file')
                    continue
                ncv = nc[n]
                if ncv.shape != shapes[n]:
                    problems.append(f'{n}: shape {ncv.shape} vs {shapes[n]}')
                notes[n] = compare(n, dtypes_[n], want[n], ncv, xd[n], problems)
                check_attrs(n, var_attrs[n], {k: ncv.getncattr(k) for k in ncv.ncattrs()}, problems)
                a = {k: ncv.getncattr(k) for k in ncv.ncattrs()}
                if 'scale_factor' in a and ('valid_min' in a or 'valid_max' in a):
                    ncv.set_auto_maskandscale(False)
                    raw = np.asarray(ncv[:])
                    real = raw != a.get('_FillValue', np.nan)
                    lo, hi = a.get('valid_min', -np.inf), a.get('valid_max', np.inf)
                    if not np.all((raw[real] >= lo) & (raw[real] <= hi)):
                        problems.append(f'{n}: stored values outside the stored valid range')
    except Exception as err:
        problems.append(f'FAILED: {type(err).__name__}: {err}')
        traceback.print_exc(limit=-2)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    dt = time.perf_counter() - t0
    status = 'ok' if not problems else f'{len(problems)} problem(s)'
    print(f'{label}  [{dt:.1f} s] {status}')
    print('   vars:', ', '.join(f'{k}({v})' for k, v in notes.items()))
    for p in problems:
        print('   -', p)
    summary.append((label, problems))
shutil.rmtree(OUT, ignore_errors=True)
print(f'\n{sum(1 for _, p in summary if not p)} of {len(summary)} datasets with no problems')
sys.exit(1 if any(p for _, p in summary) else 0)
