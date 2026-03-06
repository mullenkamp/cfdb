import numpy as np
import pytest
import xarray as xr

from cfdb import open_dataset, dtypes
from cfdb.xarray_backend import CfdbBackendEntrypoint


@pytest.fixture
def grid_file(tmp_path):
    """Create a cfdb grid file with coords and data vars."""
    fp = tmp_path / "test_xr.cfdb"

    lat_data = np.linspace(0, 9.9, 20, dtype='float32')
    lon_data = np.linspace(-5, 4.9, 30, dtype='float32')
    time_data = np.linspace(0, 9, 10, dtype='datetime64[D]')

    data = np.random.default_rng(42).standard_normal((20, 30, 10)).astype('float32')

    with open_dataset(fp, flag='n') as ds:
        ds.create.coord.lat(data=lat_data, chunk_shape=(10,))
        ds.create.coord.lon(data=lon_data, chunk_shape=(15,))
        ds.create.coord.time(data=time_data, dtype=time_data.dtype)

        ds.create.crs.from_user_input(4326, 'longitude', 'latitude')

        data_dtype = dtypes.dtype('float32')
        dv = ds.create.data_var.generic('temperature', ('latitude', 'longitude', 'time'), data_dtype, chunk_shape=(10, 15, 5))
        dv[:] = data
        dv.attrs['long_name'] = 'Air Temperature'
        dv.attrs['units_attr'] = 'K'

        ds.attrs['history'] = 'test file'

    return fp, lat_data, lon_data, time_data, data


@pytest.fixture
def scaled_file(tmp_path):
    """Create a cfdb file with scaled integer encoding."""
    fp = tmp_path / "test_xr_scaled.cfdb"

    lat_data = np.linspace(0, 9, 10, dtype='float32')
    data = np.linspace(0, 100, 50, dtype='float32').reshape(10, 5)

    with open_dataset(fp, flag='n') as ds:
        ds.create.coord.lat(data=lat_data, chunk_shape=(10,))
        time_data = np.linspace(0, 4, 5, dtype='datetime64[D]')
        ds.create.coord.time(data=time_data, dtype=time_data.dtype)

        data_dtype = dtypes.dtype('float32', 1, 0, 10000)
        dv = ds.create.data_var.generic('pressure', ('latitude', 'time'), data_dtype, chunk_shape=(10, 5))
        dv[:] = data

    return fp, lat_data, time_data, data


def test_open_with_engine(grid_file):
    fp, lat_data, lon_data, time_data, data = grid_file

    ds = xr.open_dataset(fp, engine=CfdbBackendEntrypoint)

    assert set(ds.coords) == {'latitude', 'longitude', 'time'}
    assert set(ds.data_vars) == {'temperature'}

    np.testing.assert_allclose(ds['latitude'].values, lat_data, atol=1e-6)
    np.testing.assert_allclose(ds['longitude'].values, lon_data, atol=1e-6)
    np.testing.assert_array_equal(ds['time'].values, time_data)
    np.testing.assert_allclose(ds['temperature'].values, data, atol=1e-6)

    ds.close()


def test_lazy_loading(grid_file):
    fp, lat_data, lon_data, time_data, data = grid_file

    ds = xr.open_dataset(fp, engine=CfdbBackendEntrypoint)

    temp_var = ds['temperature']
    assert temp_var.shape == (20, 30, 10)
    assert temp_var.dims == ('latitude', 'longitude', 'time')

    subset = temp_var.isel(latitude=slice(0, 5))
    np.testing.assert_allclose(subset.values, data[:5, :, :])

    ds.close()


def test_global_attrs(grid_file):
    fp, *_ = grid_file

    ds = xr.open_dataset(fp, engine=CfdbBackendEntrypoint)

    assert ds.attrs['history'] == 'test file'
    assert 'crs_wkt' in ds.attrs

    ds.close()


def test_variable_attrs(grid_file):
    fp, *_ = grid_file

    ds = xr.open_dataset(fp, engine=CfdbBackendEntrypoint)

    assert ds['temperature'].attrs['long_name'] == 'Air Temperature'

    ds.close()


def test_drop_variables(grid_file):
    fp, lat_data, lon_data, time_data, data = grid_file

    ds = xr.open_dataset(fp, engine=CfdbBackendEntrypoint, drop_variables='temperature')
    assert 'temperature' not in ds
    assert 'latitude' in ds.coords
    ds.close()

    ds = xr.open_dataset(fp, engine=CfdbBackendEntrypoint, drop_variables=['longitude'])
    assert 'longitude' not in ds.coords
    assert 'temperature' in ds
    ds.close()


def test_preferred_chunks(grid_file):
    fp, *_ = grid_file

    ds = xr.open_dataset(fp, engine=CfdbBackendEntrypoint)

    temp_encoding = ds['temperature'].encoding
    assert temp_encoding['preferred_chunks'] == {
        'latitude': 10,
        'longitude': 15,
        'time': 5,
    }

    lat_encoding = ds['latitude'].encoding
    assert lat_encoding['preferred_chunks'] == {'latitude': 10}

    ds.close()


def test_guess_can_open():
    entry = CfdbBackendEntrypoint()
    assert entry.guess_can_open("test.cfdb") is True
    assert entry.guess_can_open("test.nc") is False
    assert entry.guess_can_open("test.cfdb.bak") is False
    assert entry.guess_can_open(123) is False


def test_scaled_dtype(scaled_file):
    fp, lat_data, time_data, data = scaled_file

    ds = xr.open_dataset(fp, engine=CfdbBackendEntrypoint)

    np.testing.assert_allclose(ds['pressure'].values, data, atol=0.1)

    ds.close()


def test_context_manager(grid_file):
    fp, *_ = grid_file

    with xr.open_dataset(fp, engine=CfdbBackendEntrypoint) as ds:
        assert 'temperature' in ds


def test_multiple_data_vars(tmp_path):
    fp = tmp_path / "test_multi.cfdb"

    lat = np.arange(5, dtype='float32')
    lon = np.arange(8, dtype='float32')

    temp = np.ones((5, 8), dtype='float32')
    wind = np.full((5, 8), 2.0, dtype='float32')

    with open_dataset(fp, flag='n') as ds:
        ds.create.coord.lat(data=lat, chunk_shape=(5,))
        ds.create.coord.lon(data=lon, chunk_shape=(8,))

        dt = dtypes.dtype('float32')
        dv1 = ds.create.data_var.generic('temp', ('latitude', 'longitude'), dt, chunk_shape=(5, 8))
        dv1[:] = temp
        dv2 = ds.create.data_var.generic('wind', ('latitude', 'longitude'), dt, chunk_shape=(5, 8))
        dv2[:] = wind

    ds = xr.open_dataset(fp, engine=CfdbBackendEntrypoint)
    assert set(ds.data_vars) == {'temp', 'wind'}
    np.testing.assert_allclose(ds['temp'].values, temp)
    np.testing.assert_allclose(ds['wind'].values, wind)
    ds.close()
