import io
import os
import warnings
import pytest
try:
    import tomllib as toml
except ImportError:
    import tomli as toml
import uuid6 as uuid
import numpy as np
import pathlib
import ebooklet
from cfdb import open_dataset, open_edataset, dtypes
from cfdb.data_models import PartialDataWarning

###################################################
### Parameters

script_path = pathlib.Path(os.path.realpath(os.path.dirname(__file__)))

file_path = script_path.joinpath('test_remote.cfdb')
name = 'air_temp'
coords = ('latitude', 'longitude', 'time')
chunk_shape = (20, 30, 10)
num_groups = 10

sel = (slice(1, 4), slice(None, None), slice(2, 5))
loc_sel = (slice(0.4, 0.7), slice(None, None), slice('1970-01-04', '1970-01-10'))

lat_data = np.linspace(0, 9.9, 100, dtype='float32')
lon_data = np.linspace(-5, 4.9, 100, dtype='float32')
time_data = np.linspace(0, 10, 10, dtype='datetime64[D]')

data_var_data = np.linspace(0, 9999.9, 100000, dtype='float32').reshape(100, 100, 10)

copy_file_path = script_path.joinpath('test_remote_copy.cfdb')
nc_file_path = script_path.joinpath('test_remote.nc')

## EDataset Connection
try:
    with io.open(script_path.joinpath('s3_config.toml'), "rb") as f:
        conn_config = toml.load(f)['connection_config']

    endpoint_url = conn_config['endpoint_url']
    access_key_id = conn_config['access_key_id']
    access_key = conn_config['access_key']
except:
    endpoint_url = os.environ.get('endpoint_url')
    access_key_id = os.environ.get('access_key_id')
    access_key = os.environ.get('access_key')

bucket = 'achelous'
db_key = uuid.uuid8().hex[-13:]
base_url = 'https://b2.tethys-ts.xyz/file/' + bucket + '/'
db_url = base_url + db_key


###################################################
### Helpers

def _clean_local():
    """Remove local file and remote_index to avoid UUID mismatch."""
    for suffix in ['', '.remote_index', '.changelog']:
        p = file_path.parent / (file_path.name + suffix)
        if p.exists():
            p.unlink()


###################################################
### Fixtures

@pytest.fixture(scope="module")
def remote_conn():
    if not access_key_id or not access_key:
        pytest.skip("S3 credentials not available")
    return ebooklet.S3Connection(access_key_id, access_key, db_key, bucket, endpoint_url=endpoint_url, db_url=db_url)


@pytest.fixture(scope="module", autouse=True)
def cleanup(request, remote_conn):
    """Cleanup test data on S3 and locally after all tests."""
    def remove_test_data():
        try:
            with remote_conn.open('w') as s3open:
                s3open.break_other_locks()
                s3open.delete_remote()
        except Exception:
            pass

        for fp in [file_path, copy_file_path, nc_file_path]:
            if fp.exists():
                fp.unlink()

        for suffix in ['.remote_index', '.changelog']:
            p = file_path.parent / (file_path.name + suffix)
            if p.exists():
                p.unlink()

    request.addfinalizer(remove_test_data)


@pytest.fixture(scope="module")
def pushed_dataset(remote_conn):
    """Create dataset, push to remote once. Shared by read-only tests."""
    _clean_local()

    with open_dataset(file_path, flag='n') as ds:
        ds.create.coord.lat(data=lat_data, chunk_shape=(20,))
        ds.create.coord.lon(data=lon_data, chunk_shape=(20,))
        ds.create.coord.time(data=time_data, dtype=time_data.dtype)

        data_dtype = dtypes.dtype(data_var_data.dtype, 1, 0, 10000)
        data_var = ds.create.data_var.generic(name, coords, data_dtype, chunk_shape=chunk_shape)
        data_var[:] = data_var_data

    with open_edataset(remote_conn, file_path, flag='w', num_groups=num_groups) as ds:
        changes = ds.changes()
        assert changes.push()

    _clean_local()

    return remote_conn


###################################################
### Tests


def test_edataset_push_and_pull(pushed_dataset):
    """Test basic push/pull round-trip with index and loc selection."""
    with open_edataset(pushed_dataset, file_path) as ds:
        data_var = ds[name]

        view1 = data_var[sel]
        assert np.allclose(view1.data, data_var_data[sel])

        view2 = data_var.loc[loc_sel]
        assert np.allclose(view2.data, data_var_data[(slice(4, 7), slice(None, None), slice(3, 9))])


def test_edataset_iter_chunks(pushed_dataset):
    """Test iter_chunks on an EDataset pulled from remote."""
    with open_edataset(pushed_dataset, file_path) as ds:
        dv = ds[name]
        result = np.full(dv.shape, np.nan, dtype=data_var_data.dtype)
        for chunk_slices, data in dv.iter_chunks():
            result[chunk_slices] = data

        assert np.allclose(result, data_var_data, atol=0.1)


def test_edataset_groupby(pushed_dataset):
    """Test groupby on an EDataset pulled from remote."""
    with open_edataset(pushed_dataset, file_path) as ds:
        dv = ds[name]
        result = np.full(dv.shape, np.nan, dtype=data_var_data.dtype)
        for slices, data in dv.groupby('time'):
            result[slices] = data

        assert np.allclose(result, data_var_data, atol=0.1)


def test_edataset_select(pushed_dataset):
    """Test dataset-level select and select_loc on EDataset."""
    with open_edataset(pushed_dataset, file_path) as ds:
        ds_view = ds.select({'latitude': slice(1, 4), 'time': slice(2, 5)})
        subset = ds_view[name]
        expected = data_var_data[1:4, :, 2:5]
        assert np.allclose(subset.data, expected, atol=0.1)

        ds_view_loc = ds.select_loc({'latitude': slice(0.4, 0.7)})
        subset_loc = ds_view_loc[name]
        expected_loc = data_var_data[4:7, :, :]
        assert np.allclose(subset_loc.data, expected_loc, atol=0.1)


def test_edataset_copy(pushed_dataset):
    """Test copying an EDataset to a local cfdb file."""
    try:
        with open_edataset(pushed_dataset, file_path) as ds:
            new_ds = ds.copy(copy_file_path)
            new_ds.close()

        with open_dataset(copy_file_path) as ds:
            assert name in ds.data_var_names
            assert set(ds.coord_names) == set(coords)
            result = ds[name].data
            assert np.allclose(result, data_var_data, atol=0.1)
    finally:
        if copy_file_path.exists():
            copy_file_path.unlink()


def test_edataset_to_netcdf4(pushed_dataset):
    """Test exporting an EDataset to netCDF4."""
    import h5netcdf

    try:
        with open_edataset(pushed_dataset, file_path) as ds:
            ds.to_netcdf4(nc_file_path)

        with h5netcdf.File(nc_file_path) as f:
            assert name in f.variables
            for coord in coords:
                assert coord in f.variables
    finally:
        if nc_file_path.exists():
            nc_file_path.unlink()


def test_edataset_write_and_push(pushed_dataset):
    """Test modifying data on an EDataset and pushing updates."""
    modified_slice = (slice(0, 5), slice(0, 5), slice(0, 5))
    new_data = np.zeros((5, 5, 5), dtype='float32')

    # Open for write — needs remote_index to exist so we can find the variable
    with open_edataset(pushed_dataset, file_path, flag='w') as ds:
        dv = ds[name]
        dv[modified_slice] = new_data
        changes = ds.changes()
        result = changes.push()
        assert result.updated is True and not result.failures

    _clean_local()

    with open_edataset(pushed_dataset, file_path) as ds:
        dv = ds[name]
        view = dv[modified_slice]
        assert np.allclose(view.data, new_data, atol=0.1)

    _clean_local()


def test_edataset_load_includes_coordinates(pushed_dataset):
    """load() must fetch coordinate chunks so open_dataset can read them."""
    _clean_local()

    with open_edataset(pushed_dataset, file_path) as ds:
        ds.load()

    with open_dataset(file_path, allow_partial=True) as ds:
        lat = ds['latitude'].data
        assert not np.any(np.isnan(lat)), "Latitude data is NaN after load()"
        assert np.allclose(lat, lat_data)

        lon = ds['longitude'].data
        assert not np.any(np.isnan(lon)), "Longitude data is NaN after load()"
        assert np.allclose(lon, lon_data)

        time = ds['time'].data
        assert not np.any(np.isnat(time)), "Time data is NaT after load()"
        assert np.array_equal(time, time_data)

    _clean_local()


def test_edataset_remote_flag_persisted(pushed_dataset):
    """open_dataset on a pulled edataset file should emit PartialDataWarning."""
    _clean_local()

    with open_edataset(pushed_dataset, file_path) as ds:
        pass

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        with open_dataset(file_path) as ds:
            pass
        partial_warnings = [x for x in w if issubclass(x.category, PartialDataWarning)]
        assert len(partial_warnings) == 1, f"Expected 1 PartialDataWarning, got {len(partial_warnings)}"

    _clean_local()


###################################################
### Publish-workflow footgun regressions (0.9.0)
# Separate remotes/local files so these don't disturb the shared pushed_dataset state above.

fg1_file_path = script_path.joinpath('test_remote_fg1.cfdb')
fg1_reader_path = script_path.joinpath('test_remote_fg1_reader.cfdb')
fg2_file_path = script_path.joinpath('test_remote_fg2.cfdb')
fg1_db_key = uuid.uuid8().hex[-13:]
fg2_db_key = uuid.uuid8().hex[-13:]

fg_lat_append = np.linspace(10.0, 10.9, 10, dtype='float32')


def _clean_fg_local():
    for base in [fg1_file_path, fg1_reader_path, fg2_file_path]:
        for suffix in ['', '.remote_index', '.changelog']:
            p = base.parent / (base.name + suffix)
            if p.exists():
                p.unlink()


def _make_fg_conn(key):
    if not access_key_id or not access_key:
        pytest.skip("S3 credentials not available")
    return ebooklet.S3Connection(access_key_id, access_key, key, bucket, endpoint_url=endpoint_url, db_url=base_url + key)


@pytest.fixture(scope="module")
def fg1_conn():
    return _make_fg_conn(fg1_db_key)


@pytest.fixture(scope="module")
def fg2_conn():
    return _make_fg_conn(fg2_db_key)


@pytest.fixture(scope="module", autouse=True)
def fg_cleanup(request):
    def remove_fg_data():
        for key in [fg1_db_key, fg2_db_key]:
            if not access_key_id or not access_key:
                continue
            try:
                conn = _make_fg_conn(key)
                with conn.open('w') as s3open:
                    s3open.break_other_locks()
                    s3open.delete_remote()
            except Exception:
                pass
        _clean_fg_local()

    request.addfinalizer(remove_fg_data)


def test_edataset_midsession_push(fg1_conn):
    """
    Footgun 1 regression: push() during the creation session must publish the
    complete structure metadata AND attrs (previously the remote got the empty
    SysMeta written at open, and no attrs at all).
    """
    _clean_fg_local()

    with open_edataset(fg1_conn, fg1_file_path, flag='n', num_groups=num_groups) as ds:
        ds.create.coord.lat(data=lat_data, chunk_shape=(20,))
        ds.attrs['project'] = 'footgun-test'
        ds['latitude'].attrs['note'] = 'mid-session'
        dv = ds.create.data_var.generic('temp', ('latitude',), dtypes.dtype('float32'), chunk_shape=(20,))
        dv[:] = lat_data

        assert ds.push()  # mid-session: the dataset is still open

        # Verify from an INDEPENDENT reader while the writer session is still open
        with open_edataset(fg1_conn, fg1_reader_path, flag='r') as ds2:
            assert 'latitude' in ds2.coord_names
            assert 'temp' in ds2.data_var_names
            assert ds2.attrs['project'] == 'footgun-test'
            assert ds2['latitude'].attrs['note'] == 'mid-session'
            assert np.allclose(ds2['temp'].data, lat_data)

        # The direct changes().push() path must be just as safe
        ds['latitude'].attrs['note2'] = 'via-changes'
        assert ds.changes().push()

    _clean_fg_local()

    with open_edataset(fg1_conn, fg1_file_path, flag='r') as ds:
        assert ds['latitude'].attrs['note2'] == 'via-changes'

    _clean_fg_local()


@pytest.fixture(scope="module")
def fg2_pushed(fg2_conn):
    """Create + push + close a small dataset for the attach regressions."""
    _clean_fg_local()

    with open_edataset(fg2_conn, fg2_file_path, flag='n', num_groups=num_groups) as ds:
        ds.create.coord.lat(data=lat_data, chunk_shape=(20,))
        dv = ds.create.data_var.generic('temp', ('latitude',), dtypes.dtype('float32'), chunk_shape=(20,))
        dv[:] = lat_data
        ds.attrs['origin'] = 'fg2'

    # num_groups must be re-passed here: the remote doesn't exist yet, so it can't be read from S3 metadata (same pattern as pushed_dataset above)
    with open_edataset(fg2_conn, fg2_file_path, flag='w', num_groups=num_groups) as ds:
        assert ds.push()

    _clean_fg_local()

    return fg2_conn


def test_edataset_fresh_local_w_attaches(fg2_pushed):
    """
    Footgun 2 regression: flag='w' with a fresh local file must ATTACH to the
    existing remote dataset (previously it silently created a new empty one,
    whose SysMeta would destroy the remote's structure metadata on push).
    """
    with open_edataset(fg2_pushed, fg2_file_path, flag='w') as ds:
        # structure must be loaded from the remote, not created empty
        assert 'latitude' in ds.coord_names
        assert 'temp' in ds.data_var_names
        assert ds.attrs['origin'] == 'fg2'

        # a real modification via the attached session must round-trip
        ds['latitude'].append(fg_lat_append)
        ds['temp'][100:110] = fg_lat_append
        assert ds.push()

    _clean_fg_local()

    with open_edataset(fg2_pushed, fg2_file_path, flag='r') as ds:
        assert ds['latitude'].shape == (110,)
        assert np.allclose(ds['latitude'].data[100:], fg_lat_append)
        assert np.allclose(ds['temp'].data[:100], lat_data)
        assert np.allclose(ds['temp'].data[100:], fg_lat_append)
        assert ds.attrs['origin'] == 'fg2'

    _clean_fg_local()


def test_edataset_fresh_local_c_attaches(fg2_pushed):
    """flag='c' with a fresh local file must attach to the existing remote too."""
    with open_edataset(fg2_pushed, fg2_file_path, flag='c') as ds:
        assert 'latitude' in ds.coord_names
        assert 'temp' in ds.data_var_names
        assert ds['latitude'].shape == (110,)

    _clean_fg_local()


def test_edataset_noop_session_pushes_nothing(fg2_pushed):
    """
    A write session that changes nothing must have nothing to push
    (previously every open rewrote the metadata slot, so every session
    pushed 'changes' even when idle).
    """
    with open_edataset(fg2_pushed, fg2_file_path, flag='w') as ds:
        assert ds.push().updated is False

    _clean_fg_local()


###################################################
### ts_ortho remotes (0.9.1)

ts1_file_path = script_path.joinpath('test_remote_ts1.cfdb')
ts1_reader_path = script_path.joinpath('test_remote_ts1_reader.cfdb')
ts1_db_key = uuid.uuid8().hex[-13:]


def _clean_ts1_local():
    for base in [ts1_file_path, ts1_reader_path]:
        for suffix in ['', '.remote_index', '.changelog']:
            p = base.parent / (base.name + suffix)
            if p.exists():
                p.unlink()


@pytest.fixture(scope="module")
def ts1_conn():
    return _make_fg_conn(ts1_db_key)


@pytest.fixture(scope="module", autouse=True)
def ts1_cleanup(request):
    def remove_ts1_data():
        if access_key_id and access_key:
            try:
                conn = _make_fg_conn(ts1_db_key)
                with conn.open('w') as s3open:
                    s3open.break_other_locks()
                    s3open.delete_remote()
            except Exception:
                pass
        _clean_ts1_local()

    request.addfinalizer(remove_ts1_data)


def test_edataset_ts_ortho_e2e(ts1_conn, fg2_pushed):
    """
    0.9.1 regression: open_edataset supports dataset_type='ts_ortho' at create
    (0.9.0 raised TypeError) and existing remotes come back with the class
    matching their STORED type (0.9.0 always returned EGrid).
    """
    import shapely

    _clean_ts1_local()

    geo_data = [shapely.Point(x, y) for x, y in zip(np.linspace(-5, 4.9, 20), np.linspace(0, 9.9, 20))]
    ts_values = np.linspace(0, 199.9, 200, dtype='float32').reshape(20, 10)

    with open_edataset(ts1_conn, ts1_file_path, flag='n', dataset_type='ts_ortho', num_groups=num_groups) as ds:
        assert type(ds).__name__ == 'ETimeSeriesOrtho'
        assert ds.dataset_type == 'ts_ortho'
        geo_coord = ds.create.coord.point()
        geo_coord.append(geo_data)
        ds.create.coord.time(data=time_data, dtype=time_data.dtype)
        dv = ds.create.data_var.generic('temp', ('point', 'time'), dtypes.dtype('float32'), chunk_shape=(10, 10))
        dv[:] = ts_values

    # num_groups must be re-passed: the remote doesn't exist yet (same pattern as fg2_pushed)
    with open_edataset(ts1_conn, ts1_file_path, flag='w', num_groups=num_groups) as ds:
        assert type(ds).__name__ == 'ETimeSeriesOrtho'
        assert ds.push()

    _clean_ts1_local()

    # Fresh-local reader: class + property from the STORED type, no param passed
    with open_edataset(ts1_conn, ts1_reader_path, flag='r') as ds:
        assert type(ds).__name__ == 'ETimeSeriesOrtho'
        assert ds.dataset_type == 'ts_ortho'
        assert 'point' in ds.coord_names
        assert np.allclose(ds['temp'].data, ts_values)

    # An existing GRID remote still comes back as EGrid
    with open_edataset(fg2_pushed, fg2_file_path, flag='r') as ds:
        assert type(ds).__name__ == 'EGrid'
        assert ds.dataset_type == 'grid'

    _clean_fg_local()
    _clean_ts1_local()
