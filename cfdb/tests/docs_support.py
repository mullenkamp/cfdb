"""Support for the docs example tests (Sybil), wired from the repo-root ``conftest.py``.

Every Python block of ``README.md`` and ``docs/**/*.md`` runs, page by page, top to bottom in one
namespace (Sybil). Run them with ``uv run pytest README.md docs``. Per page:

- the cwd and ``HOME`` are a fresh scratch directory;
- S3 examples reach a local moto server: the docs' placeholder endpoint ``https://s3.example.com``
  and placeholder keys are rewritten inside ``ebooklet.remote.S3Connection.__init__``;
- any non-loopback network connection fails the page (the docs run offline);
- afterwards, any cfdb dataset still open fails the page (an unclosed dataset makes the next open
  of its file wait forever on the lock).

Nothing here uses pytest fixtures: Sybil requests fixtures per example and never runs autouse ones,
so every per-page step lives in ``page_setup``/``page_teardown``, which Sybil calls once per page.
"""

import ctypes
import functools
import gc
import inspect
import ipaddress
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import types
import urllib.request
import warnings
from doctest import NORMALIZE_WHITESPACE

import boto3
import cfdb
import ebooklet
from ebooklet import S3Connection
import numpy as np
import shapely
from sybil import Sybil
from sybil.integration import pytest as sybil_pytest
from sybil.parsers.markdown import ClearNamespaceParser, PythonCodeBlockParser, SkipParser

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Per-example time limit (s). Healthy examples take well under a second; a page that leaks a
#: dataset lock would otherwise hang for good.
EXAMPLE_TIMEOUT = 30

#: Refuse a ``.data`` read that would decode more than this many bytes.
DATA_LIMIT_BYTES = 500e6

PLACEHOLDER_ENDPOINT = 'https://s3.example.com'
PLACEHOLDER_KEYS = ('...', 'YOUR_KEY', 'YOUR_SECRET', 'KEY', 'SECRET')
BUCKET = 'my-bucket'
_PUBLIC_READ = (
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":"*",'
    '"Action":["s3:GetObject","s3:HeadObject"],"Resource":"arn:aws:s3:::my-bucket/*"}]}'
)


class _State:
    def __init__(self):
        self.live = False
        self.moto_proc: subprocess.Popen | None = None
        self.moto_url: str | None = None
        self.pages_run = 0
        self.pages: dict = {}
        self.failed_pages: set = set()
        self.violations: list = []


STATE = _State()


# --- which files are docs ---------------------------------------------------------------------


def is_doc_file(path) -> bool:
    """README.md at the repo root, or any .md under docs/ (the one definition both conftest.py
    and the inventory test use)."""
    try:
        rel = pathlib.Path(path).resolve().relative_to(ROOT)
    except ValueError:
        return False
    return rel.suffix == '.md' and (rel == pathlib.Path('README.md') or rel.parts[0] == 'docs')


def doc_files() -> list[pathlib.Path]:
    return [ROOT / 'README.md', *sorted(p for p in (ROOT / 'docs').rglob('*.md') if is_doc_file(p))]


def make_sybil() -> Sybil:
    # No global NUMBER or ELLIPSIS: an example that needs one opts in with "# doctest: +ELLIPSIS".
    return Sybil(
        parsers=[
            PythonCodeBlockParser(doctest_optionflags=NORMALIZE_WHITESPACE),
            SkipParser(),
            ClearNamespaceParser(),
        ],
        patterns=['*.md'],
        path=str(ROOT),  # patterns are matched relative to this (default: this module's dir)
        setup=page_setup,
        teardown=page_teardown,
        name='docs',
    )


def is_docs_item(item) -> bool:
    return type(item).__name__ == 'SybilItem'


def page_key(item) -> str:
    return item.parent.nodeid


# --- session ----------------------------------------------------------------------------------


def session_finish(session):
    _stop_moto()


# --- moto -------------------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _die_with_parent():
    # Runs in the moto child before exec: the kernel sends it SIGTERM when pytest dies, even by
    # SIGTERM or SIGKILL, which skip pytest's own session-finish hook (Linux only).
    try:
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM)  # 1 = PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass


def _ensure_moto():
    if STATE.moto_proc is not None:
        return
    port = _free_port()
    proc = subprocess.Popen(  # noqa: S603 (a fixed command: this interpreter's moto server)
        [sys.executable, '-m', 'moto.server', '-H', '127.0.0.1', '-p', str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=_die_with_parent if sys.platform.startswith('linux') else None,  # noqa: PLW1509
    )
    url = f'http://127.0.0.1:{port}'
    deadline = time.monotonic() + 30
    while True:
        try:
            urllib.request.urlopen(f'{url}/moto-api/', timeout=1).close()  # noqa: S310 (loopback http)
            break
        except OSError:
            if proc.poll() is not None or time.monotonic() > deadline:
                proc.kill()
                msg = 'the moto server did not start'
                raise RuntimeError(msg) from None
            time.sleep(0.1)
    STATE.moto_proc, STATE.moto_url = proc, url


def _stop_moto():
    proc = STATE.moto_proc
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(5)
    STATE.moto_proc = None


def _reset_moto():
    req = urllib.request.Request(f'{STATE.moto_url}/moto-api/reset', method='POST')  # noqa: S310 (loopback http)
    urllib.request.urlopen(req, timeout=10).close()  # noqa: S310
    s3 = boto3.client(
        's3',
        endpoint_url=STATE.moto_url,
        aws_access_key_id='test',
        aws_secret_access_key='test',
        region_name='us-east-1',
    )
    s3.create_bucket(Bucket=BUCKET)
    # public read, so a published entry's data_url opens credential-free (as on a real public bucket)
    s3.put_bucket_policy(Bucket=BUCKET, Policy=_PUBLIC_READ)


_ORIG_S3_INIT = ebooklet.remote.S3Connection.__init__


def _install_s3_rewrite():
    """Rewrite the docs' placeholders to the moto server inside S3Connection itself.

    Patched on the class (not the ``ebooklet.S3Connection`` name), because ebooklet builds its
    own connection for URL-string and dict remotes (``ebooklet/remote.py`` ``check_remote_conn``).
    """
    cls = ebooklet.remote.S3Connection
    if getattr(cls.__init__, '_docs_rewrite', False):
        return
    orig = _ORIG_S3_INIT
    sig = inspect.signature(orig)

    @functools.wraps(orig)
    def rewritten_init(self, *args, **kwargs):
        bound = sig.bind(self, *args, **kwargs)
        a = bound.arguments
        for k in ('endpoint_url', 'db_url'):
            v = a.get(k)
            if isinstance(v, str) and v.startswith(PLACEHOLDER_ENDPOINT):
                a[k] = STATE.moto_url + v[len(PLACEHOLDER_ENDPOINT) :]
        for k in ('access_key_id', 'access_key'):
            if a.get(k) in PLACEHOLDER_KEYS:
                a[k] = 'test'
        with warnings.catch_warnings():
            # the moto URL is plain http by necessity; the docs' own URLs are https
            warnings.filterwarnings('ignore', message='db_url uses plain http', category=UserWarning)
            orig(*bound.args, **bound.kwargs)

    rewritten_init._docs_rewrite = True
    cls.__init__ = rewritten_init


def _remove_s3_rewrite():
    ebooklet.remote.S3Connection.__init__ = _ORIG_S3_INIT


# --- guards -----------------------------------------------------------------------------------

_orig_connect = socket.socket.connect
_orig_connect_ex = socket.socket.connect_ex
_orig_getaddrinfo = socket.getaddrinfo


def _is_loopback(host) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode()
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == 'localhost'


def _guarded_getaddrinfo(host, *args, **kwargs):
    # A name lookup fails before any connect, so a placeholder host the rewrite missed
    # (s3.example.com) would otherwise never reach the connect guard.
    if not _is_loopback(host):
        STATE.violations.append(f'name lookup of {host!r}')
        msg = f'docs offline run: name lookup of {host!r} refused'
        raise socket.gaierror(msg)
    return _orig_getaddrinfo(host, *args, **kwargs)


def _offline_check(sock, address):
    if sock.family not in (socket.AF_INET, socket.AF_INET6):
        return
    if not _is_loopback(address[0]):
        STATE.violations.append(str(address))
        msg = f'docs offline run: network connection to {address} refused (the docs examples run offline)'
        raise OSError(msg)


def _guarded_connect(self, address):
    _offline_check(self, address)
    return _orig_connect(self, address)


def _guarded_connect_ex(self, address):
    _offline_check(self, address)
    return _orig_connect_ex(self, address)


def _set_socket_guard(*, on: bool):
    socket.socket.connect = _guarded_connect if on else _orig_connect
    socket.socket.connect_ex = _guarded_connect_ex if on else _orig_connect_ex
    socket.getaddrinfo = _guarded_getaddrinfo if on else _orig_getaddrinfo


_orig_data = cfdb.support_classes.DataVariableView.data


def _guarded_data(self):
    n = int(np.prod(self.shape, dtype='int64'))
    try:
        itemsize = np.dtype(self.dtype.dtype).itemsize
    except Exception:  # an unusual dtype: assume the widest
        itemsize = 8
    if n * itemsize > DATA_LIMIT_BYTES:
        msg = (
            f'docs test guard: .data would decode {n:,} values (~{n * itemsize / 1e9:.1f} GB, shape '
            f'{self.shape}); select a smaller part first'
        )
        raise MemoryError(msg)
    return _orig_data.fget(self)


def _set_data_guard(*, on: bool):
    cfdb.support_classes.DataVariableView.data = property(_guarded_data) if on else _orig_data


# Sybil's traceback filter takes element [1] of the cut traceback, so a chained exception (raised
# inside an except clause) whose traceback is one frame turns the report into a pytest INTERNALERROR
# that hides the real error. Fall back to the uncut traceback.
_orig_traceback_filter = getattr(sybil_pytest.SybilItem, '_traceback_filter', None)


def _safe_traceback_filter(self, excinfo):
    try:
        return _orig_traceback_filter(self, excinfo)
    except IndexError:
        return excinfo.traceback


if _orig_traceback_filter is not None:
    sybil_pytest.SybilItem._traceback_filter = _safe_traceback_filter


# --- per page ---------------------------------------------------------------------------------


def page_setup(namespace):
    STATE.pages_run += 1
    scratch = pathlib.Path(tempfile.mkdtemp(prefix='docs-page-'))
    page = {
        'scratch': scratch,
        'cwd': os.getcwd(),
        'home': os.environ.get('HOME'),
        'module': f'_docs_page_{STATE.pages_run}',
        'violations': len(STATE.violations),
    }
    STATE.pages[id(namespace)] = page

    _ensure_moto()  # before the chdir/HOME change: the server outlives this page
    os.chdir(scratch)
    os.environ['HOME'] = str(scratch)

    _reset_moto()
    _install_s3_rewrite()
    _set_data_guard(on=True)
    _set_socket_guard(on=not STATE.live)

    namespace['docs_live'] = STATE.live
    # Functions defined on a page must pickle by reference for map(n_workers=...): give the page an
    # importable module whose attributes are the page's names.
    module = types.ModuleType(page['module'])

    def _getattr(name, _ns=namespace):
        try:
            return _ns[name]
        except KeyError:
            raise AttributeError(name) from None

    module.__getattr__ = _getattr
    sys.modules[page['module']] = module
    namespace['__name__'] = page['module']


def page_teardown(namespace):
    page = STATE.pages.pop(id(namespace))
    problems = []

    # The canary: the isolation really was in force for this page.
    if os.environ.get('HOME') != str(page['scratch']) or os.getcwd() != str(page['scratch']):
        problems.append('page isolation was not in force (HOME or the cwd was not the scratch dir)')

    gc.collect()
    # type(o), not isinstance(o): gc also holds weakref proxies, which raise once their referent dies.
    left_open = [o for o in gc.get_objects() if issubclass(type(o), cfdb.main.Dataset) and o.is_open]
    for ds in left_open:
        try:
            ds.close()
        except Exception as e:  # report, keep closing the rest
            problems.append(f'closing {ds.file_path} raised {type(e).__name__}: {e}')
    if left_open:
        problems.append(f'dataset(s) left open: {sorted(str(ds.file_path.name) for ds in left_open)}')

    new_violations = STATE.violations[page['violations'] :]
    if new_violations:
        problems.append(f'network connections attempted in an offline run: {new_violations}')

    _set_socket_guard(on=False)
    _set_data_guard(on=False)
    _remove_s3_rewrite()
    sys.modules.pop(page['module'], None)
    os.chdir(page['cwd'])
    if page['home'] is None:
        os.environ.pop('HOME', None)
    else:
        os.environ['HOME'] = page['home']
    shutil.rmtree(page['scratch'], ignore_errors=True)

    if problems:
        raise AssertionError('; '.join(problems))


# --- builders for invisible blocks ------------------------------------------------------------
# State a reader is assumed to already have: an existing file with the names a page uses. Values
# are seeded random numbers; no page shows them.

_COORD_CREATORS = {'latitude': 'lat', 'longitude': 'lon', 'time': 'time', 'x': 'x', 'y': 'y', 'height': 'height'}


def make_dataset(
    path,
    coords,
    variables,
    dims,
    *,
    chunk_shape=None,
    crs=None,
    dataset_type='grid',
    loc=10.0,
    scale=5.0,
    seed=0,
    units=None,
):
    """Create ``path`` with the given coordinates (name -> values, in creation order) and float32
    data variables on ``dims``. ``crs`` is an EPSG code (x/y taken from ``dims``) or None."""
    rng = np.random.default_rng(seed)
    with cfdb.open_dataset(path, flag='n', dataset_type=dataset_type) as ds:
        for name, values in coords.items():
            creator = getattr(ds.create.coord, _COORD_CREATORS.get(name, ''), None)
            values = np.asarray(values)
            if creator is None:
                ds.create.coord.generic(name, data=values, dtype=values.dtype)
            else:
                creator(data=values)
        shape = tuple(len(np.asarray(coords[d])) for d in dims)
        for name in variables:
            dv = ds.create.data_var.generic(name, dims, dtype='float32', chunk_shape=chunk_shape)
            if units:
                dv.attrs['units'] = units
            dv[:] = rng.normal(loc, scale, shape).astype('float32')
        if crs is not None:
            x = 'longitude' if 'longitude' in dims else 'x'
            y = 'latitude' if 'latitude' in dims else 'y'
            ds.create.crs.from_user_input(crs, x_coord=x, y_coord=y)
    return path


def make_global_grid(
    path, variables=('temperature',), dims=('latitude', 'longitude', 'time'), *, n_time=3, chunk_shape=None, crs=4326
):
    """A 1-degree global grid (181 x 360) with daily times from 2020-01-01."""
    coords = {
        'latitude': np.linspace(-90, 90, 181, dtype='float32'),
        'longitude': np.linspace(-180, 179, 360, dtype='float32'),
        'time': np.arange('2020-01-01', np.datetime64('2020-01-01') + n_time, dtype='datetime64[D]'),
    }
    coords = {d: coords[d] for d in dims}
    return make_dataset(path, coords, variables, dims, chunk_shape=chunk_shape, crs=crs)


def make_nz_grid(
    path,
    variables=('temperature',),
    dims=('latitude', 'longitude', 'time'),
    *,
    step=0.25,
    n_time=3,
    chunk_shape=None,
    crs=4326,
):
    """A small New Zealand lat/lon grid (-48..-34, 166..179) with daily times from 2020-01-01."""
    coords = {
        'latitude': np.arange(-48, -33.9, step, dtype='float32'),
        'longitude': np.arange(166, 179.1, step, dtype='float32'),
        'time': np.arange('2020-01-01', np.datetime64('2020-01-01') + n_time, dtype='datetime64[D]'),
    }
    coords = {d: coords[d] for d in dims}
    return make_dataset(path, coords, variables, dims, chunk_shape=chunk_shape, crs=crs, loc=15.0, scale=3.0)


# ---- builders for the guide pages: export, interpolation, rechunking, S3, xarray; reference: dtypes, edataset


def make_nz_grid_with_gaps(path, variables=('temperature',)):
    """make_nz_grid with a block of NaNs in every time step (for interp_na)."""
    make_nz_grid(path, variables)
    with cfdb.open_dataset(path, flag='w') as ds:
        for name in variables:
            ds[name][10:12, 10:12, :] = np.full((2, 2, 3), np.nan, dtype='float32')
    return path


def make_nztm_grid(path, variables=('temperature',)):
    """A 5 km NZGD2000 / NZTM (EPSG:2193) grid of 21 x 21 cells and two daily times."""
    coords = {
        'y': np.arange(5100000, 5200001, 5000, dtype='float64'),
        'x': np.arange(1500000, 1600001, 5000, dtype='float64'),
        'time': np.arange('2020-01-01', '2020-01-03', dtype='datetime64[D]'),
    }
    return make_dataset(path, coords, variables, ('y', 'x', 'time'), crs=2193, loc=15.0, scale=3.0)


def make_levels_grid(path):
    """A small NZ grid with a height axis: 'temperature' and the per-cell 'level_heights' (m)."""
    lat = np.arange(-46, -42.9, 0.5, dtype='float32')
    lon = np.arange(168, 171.1, 0.5, dtype='float32')
    height = np.arange(1, 7, dtype='float32')
    time = np.arange('2020-01-01', '2020-01-03', dtype='datetime64[D]')
    dims = ('height', 'latitude', 'longitude', 'time')
    shape = (len(height), len(lat), len(lon), len(time))
    rng = np.random.default_rng(0)
    with cfdb.open_dataset(path, flag='n') as ds:
        ds.create.coord.height(data=height)
        ds.create.coord.lat(data=lat)
        ds.create.coord.lon(data=lon)
        ds.create.coord.time(data=time)
        t = ds.create.data_var.generic('temperature', dims, dtype='float32')
        t[:] = (20 - 2 * np.arange(6)[:, None, None, None] + rng.normal(0, 1, shape)).astype('float32')
        h = ds.create.data_var.generic('level_heights', dims, dtype='float32')
        base = np.array([10, 60, 150, 300, 600, 1000], dtype='float32')[:, None, None, None]
        h[:] = (base + rng.uniform(0, 5, shape)).astype('float32')
        ds.create.crs.from_user_input(4326, x_coord='longitude', y_coord='latitude')
    return path


def make_stations(path, variables=('temperature',)):
    """A ts_ortho dataset: two NZ stations ('point') and three daily times, CRS 4326."""
    points = np.array([shapely.Point(172.5, -43.5), shapely.Point(174.8, -41.3)], dtype=object)
    time = np.arange('2020-01-01', '2020-01-04', dtype='datetime64[D]')
    rng = np.random.default_rng(0)
    with cfdb.open_dataset(path, flag='n', dataset_type='ts_ortho') as ds:
        ds.create.coord.point(data=points)
        ds.create.coord.time(data=time)
        for name in variables:
            dv = ds.create.data_var.generic(name, ('point', 'time'), dtype='float32')
            dv[:] = rng.normal(15, 3, (len(points), len(time))).astype('float32')
        ds.create.crs.from_user_input(4326, xy_coord='point')
    return path


def make_bucket(name):
    """Create a bucket on the docs' moto server (the docs' own bucket, 'my-bucket', always exists)."""
    s3 = boto3.client(
        's3',
        endpoint_url=STATE.moto_url,
        aws_access_key_id='test',
        aws_secret_access_key='test',
        region_name='us-east-1',
    )
    try:
        s3.create_bucket(Bucket=name)
    except s3.exceptions.BucketAlreadyOwnedByYou:
        pass
    return name


def make_remote(db_key, bucket='my-bucket', n_lat=20, n_lon=36):
    """Push a small 2-D (latitude, longitude) 'temperature' dataset to the moto remote at
    bucket/db_key, through a connection in the docs' placeholder form."""
    if bucket != BUCKET:
        make_bucket(bucket)
    conn = S3Connection(
        endpoint_url=PLACEHOLDER_ENDPOINT,
        access_key_id='...',
        access_key='...',
        bucket=bucket,
        db_key=db_key,
    )
    local = f'_builder_{bucket}_{db_key.replace("/", "_")}'
    with cfdb.open_edataset(conn, local, flag='n') as ds:
        ds.create.coord.lat(data=np.linspace(-45, -40, n_lat, dtype='float32'))
        ds.create.coord.lon(data=np.linspace(170, 177, n_lon, dtype='float32'))
        dv = ds.create.data_var.generic('temperature', ('latitude', 'longitude'), dtype='float32', chunk_shape=(10, 18))
        dv[:] = np.random.default_rng(0).normal(10, 5, (n_lat, n_lon)).astype('float32')
        ds.push()
    return db_key


def make_xr_example(path):
    """'temperature' and 'humidity' on (latitude 100, longitude 200, time 20), stored in
    (50, 100, 10) chunks, CRS 4326."""
    coords = {
        'latitude': np.linspace(-50, 49, 100, dtype='float32'),
        'longitude': np.linspace(0, 199, 200, dtype='float32'),
        'time': np.arange('2020-01-01', '2020-01-21', dtype='datetime64[D]'),
    }
    return make_dataset(
        path,
        coords,
        ('temperature', 'humidity'),
        ('latitude', 'longitude', 'time'),
        chunk_shape=(50, 100, 10),
        crs=4326,
    )


# ---- builders for the quickstart, data variables; reference: coordinate, creator, dataset, data-variable, tools


def make_dv_guide_coords(path='data_vars.cfdb'):
    """docs/guide/data-variables.md: coordinates only (latitude 200, longitude 100, 200 hourly
    times from 2020-01-01), so the page's 'temperature' on (latitude, time) matches its (200, 200)
    data."""
    coords = {
        'latitude': np.linspace(-49.75, 0, 200, dtype='float32'),
        'longitude': np.linspace(160, 184.75, 100, dtype='float32'),
        'time': np.arange('2020-01-01T00', '2020-01-09T08', dtype='datetime64[h]'),
    }
    return make_dataset(path, coords, (), ())


def make_multi_var(path='multi_var.cfdb'):
    """docs/guide/data-variables.md, dataset-level iteration: 'temperature' and 'wind_speed', both
    on (latitude, longitude, time): 100 x 100 x 48 hourly steps from 2020-01-01."""
    coords = {
        'latitude': np.linspace(-49.75, -25, 100, dtype='float32'),
        'longitude': np.linspace(160, 184.75, 100, dtype='float32'),
        'time': np.arange('2020-01-01T00', '2020-01-03T00', dtype='datetime64[h]'),
    }
    return make_dataset(path, coords, ('temperature', 'wind_speed'), ('latitude', 'longitude', 'time'))


def make_coord_ref(path='coord.cfdb'):
    """docs/reference/coordinate.md: latitude 0.0 .. 19.9 (step 0.1, 200 values, chunks of 20) and
    daily time 2023-01-01 .. 2023-03-31."""
    with cfdb.open_dataset(path, flag='n') as ds:
        ds.create.coord.lat(data=np.round(np.arange(200) * 0.1, 1).astype('float32'), chunk_shape=(20,))
        ds.create.coord.time(data=np.arange('2023-01-01', '2023-04-01', dtype='datetime64[D]'))
    return path


def open_latlon_grid(path, flag='n'):
    """docs/reference/creator.md: a writable dataset holding latitude (181) and longitude (361),
    returned OPEN (the page's fragments assume an open `ds`; the page closes it)."""
    ds = cfdb.open_dataset(path, flag=flag)
    if flag == 'n':
        ds.create.coord.lat(data=np.linspace(-90, 90, 181, dtype='float32'))
        ds.create.coord.lon(data=np.linspace(-180, 180, 361, dtype='float32'))
    return ds


def open_station_ts(path):
    """docs/reference/creator.md: a writable ts_ortho dataset with a point coordinate named
    'station' (two stations) and daily times, returned OPEN."""
    ds = cfdb.open_dataset(path, flag='n', dataset_type='ts_ortho')
    points = np.array([shapely.Point(172.5, -43.5), shapely.Point(171.0, -44.0)], dtype=object)
    ds.create.coord.generic('station', data=points, dtype=cfdb.dtypes.dtype('point', precision=5))
    ds.create.coord.time(data=np.arange('2023-01-01', '2023-01-05', dtype='datetime64[D]'))
    return ds


def make_time_grid(path='data.cfdb', variables=('temperature', 'pressure')):
    """A (time, latitude, longitude) grid: 60 daily times from 2023-01-01, 1-degree latitude
    (-90 .. 90) and 5-degree longitude (-180 .. 175), CRS 4326, chunks (10, 30, 36)."""
    coords = {
        'time': np.arange('2023-01-01', '2023-03-02', dtype='datetime64[D]'),
        'latitude': np.linspace(-90, 90, 181, dtype='float32'),
        'longitude': np.linspace(-180, 175, 72, dtype='float32'),
    }
    return make_dataset(
        path,
        coords,
        variables,
        ('time', 'latitude', 'longitude'),
        chunk_shape=(10, 30, 36),
        crs=4326,
        loc=285.0,
        scale=10.0,
    )


def make_dv_ref(path='dv_ref.cfdb'):
    """docs/reference/data-variable.md: 'temperature' on (latitude, longitude), 181 x 361 in
    (20, 36) chunks, and 'air_temperature' on (time, latitude, longitude) with 3 daily times."""
    coords = {
        'time': np.arange('2023-01-01', '2023-01-04', dtype='datetime64[D]'),
        'latitude': np.linspace(-90, 90, 181, dtype='float32'),
        'longitude': np.linspace(-180, 180, 361, dtype='float32'),
    }
    make_dataset(path, coords, ('temperature',), ('latitude', 'longitude'), chunk_shape=(20, 36))
    with cfdb.open_dataset(path, flag='w') as ds:
        dv = ds.create.data_var.generic('air_temperature', ('time', 'latitude', 'longitude'), dtype='float32')
        dv[:] = np.random.default_rng(2).normal(285, 10, dv.shape).astype('float32')
    return path


def make_interp_small(path='dvi.cfdb'):
    """docs/reference/data-variable.md, interp: a 1-degree 6 x 6 grid over New Zealand with 3 daily
    times, 'temperature' on (time, latitude, longitude), CRS 4326."""
    coords = {
        'time': np.arange('2023-01-01', '2023-01-04', dtype='datetime64[D]'),
        'latitude': np.arange(-46, -40, 1, dtype='float32'),
        'longitude': np.arange(168, 174, 1, dtype='float32'),
    }
    return make_dataset(
        path, coords, ('temperature',), ('time', 'latitude', 'longitude'), crs=4326, loc=15.0, scale=3.0
    )


# ---- builders for the guide pages: attributes, combining, coordinates, indexing


def make_attrs_file(path='attrs.cfdb'):
    """attributes.md: an existing file with a `temperature` variable."""
    return make_dataset(path, {'latitude': np.linspace(0, 1, 5)}, ['temperature'], ('latitude',))


def _region(path, lat, n_time=3, start='2020-01-01', variables=('temperature', 'humidity')):
    coords = {
        'latitude': lat,
        'longitude': np.arange(166, 179.1, 0.5, dtype='float32'),
        'time': np.arange(start, np.datetime64(start) + n_time, dtype='datetime64[D]'),
    }
    return make_dataset(path, coords, variables, ('latitude', 'longitude', 'time'), crs=4326)


def make_combine_inputs():
    """combining-datasets.md: two NZ regions (north/south, disjoint latitudes), two global grids
    covering different days, a target plus a newer day of data for merge_into, and a target that
    covers both regions for the overlap examples."""
    north = _region('region_north.cfdb', np.arange(-41, -33.9, 0.5, dtype='float32'))
    south = _region('region_south.cfdb', np.arange(-48, -41.4, 0.5, dtype='float32'))
    make_global_grid('full_globe_a.cfdb', ('temperature',))
    with cfdb.open_dataset('full_globe_a.cfdb') as a:
        lat, lon = a['latitude'].data, a['longitude'].data
    coords = {'latitude': lat, 'longitude': lon, 'time': np.arange('2020-01-04', '2020-01-07', dtype='datetime64[D]')}
    make_dataset('full_globe_b.cfdb', coords, ['temperature'], ('latitude', 'longitude', 'time'), crs=4326)
    _region('existing_target.cfdb', np.arange(-48, -33.9, 0.5, dtype='float32'), n_time=3)
    _region('new_weather_data.cfdb', np.arange(-48, -33.9, 0.5, dtype='float32'), n_time=1, start='2020-01-04')
    cfdb.combine([north, south], 'target.cfdb').close()
    return [north, south]


def make_lat_time(path='indexing.cfdb', n_lat=90, n_time=366):
    """indexing.md: `temperature` on (latitude, time); latitude 0.0, 1.0, ... (exact float labels
    such as 40.0, 45.0, 50.0), daily time from 2020-01-01."""
    coords = {
        'latitude': np.arange(n_lat, dtype='float32'),
        'time': np.arange('2020-01-01', np.datetime64('2020-01-01') + n_time, dtype='datetime64[D]'),
    }
    return make_dataset(path, coords, ['temperature'], ('latitude', 'time'))
