import os
import pytest
import numpy as np
import pathlib

from cfdb import open_dataset, dtypes

script_path = pathlib.Path(os.path.realpath(os.path.dirname(__file__)))

gi_file_2d = script_path.joinpath('test_gi_2d.blt')
gi_file_3d = script_path.joinpath('test_gi_3d.blt')
gi_file_no_crs = script_path.joinpath('test_gi_no_crs.blt')
gi_file_no_axis = script_path.joinpath('test_gi_no_axis.blt')
gi_file_levels = script_path.joinpath('test_gi_levels.blt')

# Test data
lat_data = np.linspace(-10, 10, 21, dtype='float32')
lon_data = np.linspace(170, 180, 11, dtype='float32')
time_data = np.array(['2020-01-01', '2020-01-02', '2020-01-03'], dtype='datetime64[D]')

# 2D data: lat x lon (linear gradient)
data_2d = np.outer(lat_data, lon_data).astype('float32')

# 3D data: time x lat x lon
data_3d = np.stack([data_2d * (i + 1) for i in range(len(time_data))]).astype('float32')


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    def remove_test_data():
        for fp in [gi_file_2d, gi_file_3d, gi_file_no_crs, gi_file_no_axis, gi_file_levels]:
            if fp.exists():
                fp.unlink()
    request.addfinalizer(remove_test_data)


def _create_2d_dataset():
    with open_dataset(gi_file_2d, flag='n') as ds:
        ds.create.coord.lat(data=lat_data, chunk_shape=(21,))
        ds.create.coord.lon(data=lon_data, chunk_shape=(11,))
        ds.create.crs.from_user_input(4326, 'longitude', 'latitude')

        data_dtype = dtypes.dtype('float32')
        dv = ds.create.data_var.generic('temperature', ('latitude', 'longitude'), data_dtype, chunk_shape=(21, 11))
        dv[:] = data_2d


def _create_3d_dataset():
    with open_dataset(gi_file_3d, flag='n') as ds:
        ds.create.coord.lat(data=lat_data, chunk_shape=(21,))
        ds.create.coord.lon(data=lon_data, chunk_shape=(11,))
        ds.create.coord.time(data=time_data, dtype=time_data.dtype)
        ds.create.crs.from_user_input(4326, 'longitude', 'latitude')

        data_dtype = dtypes.dtype('float32')
        dv = ds.create.data_var.generic('temperature', ('time', 'latitude', 'longitude'), data_dtype, chunk_shape=(3, 21, 11))
        dv[:] = data_3d


def test_axis_auto_detection():
    _create_3d_dataset()
    with open_dataset(gi_file_3d) as ds:
        dv = ds['temperature']
        gi = dv.grid_interp()
        assert gi._x_name == 'longitude'
        assert gi._y_name == 'latitude'
        assert gi._time_name == 'time'
        assert gi._z_name is None


def test_explicit_coord_names():
    with open_dataset(gi_file_3d) as ds:
        dv = ds['temperature']
        gi = dv.grid_interp(x='longitude', y='latitude', time='time')
        assert gi._x_name == 'longitude'
        assert gi._y_name == 'latitude'
        assert gi._time_name == 'time'


def test_missing_crs_error():
    with open_dataset(gi_file_no_crs, flag='n') as ds:
        ds.create.coord.lat(data=lat_data, chunk_shape=(21,))
        ds.create.coord.lon(data=lon_data, chunk_shape=(11,))
        data_dtype = dtypes.dtype('float32')
        dv = ds.create.data_var.generic('temperature', ('latitude', 'longitude'), data_dtype, chunk_shape=(21, 11))
        dv[:] = data_2d

    with open_dataset(gi_file_no_crs) as ds:
        dv = ds['temperature']
        with pytest.raises(ValueError, match="CRS"):
            dv.grid_interp(x='longitude', y='latitude')


def test_missing_xy_error():
    with open_dataset(gi_file_no_axis, flag='n') as ds:
        ds.create.coord.lat(data=lat_data, chunk_shape=(21,))
        ds.create.coord.lon(data=lon_data, chunk_shape=(11,))
        ds.create.crs.from_user_input(4326, 'longitude', 'latitude')
        data_dtype = dtypes.dtype('float32')
        dv = ds.create.data_var.generic('temperature', ('latitude', 'longitude'), data_dtype, chunk_shape=(21, 11))
        dv[:] = data_2d

    # Re-open and remove axis info to simulate missing axes
    with open_dataset(gi_file_no_axis, flag='w') as ds:
        ds['latitude'].update_axis(None)
        ds['longitude'].update_axis(None)

    with open_dataset(gi_file_no_axis) as ds:
        dv = ds['temperature']
        with pytest.raises(ValueError, match="x and y"):
            dv.grid_interp()


def test_to_grid_no_time():
    _create_2d_dataset()
    with open_dataset(gi_file_2d) as ds:
        dv = ds['temperature']
        gi = dv.grid_interp()
        results = list(gi.to_grid(grid_res=1.0, to_crs=4326))
        assert len(results) == 1
        time_val, grid = results[0]
        assert time_val is None
        assert isinstance(grid, np.ndarray)
        assert grid.ndim == 2
        assert grid.shape[0] > 0
        assert grid.shape[1] > 0


def test_to_grid_with_time():
    _create_3d_dataset()
    with open_dataset(gi_file_3d) as ds:
        dv = ds['temperature']
        gi = dv.grid_interp()
        result = gi.to_grid(grid_res=1.0, to_crs=4326)

        # Should be a generator
        results = list(result)
        assert len(results) == len(time_data)

        for time_val, grid in results:
            assert isinstance(grid, np.ndarray)
            assert grid.ndim == 2
            assert grid.shape[0] > 0
            assert grid.shape[1] > 0


def test_to_points_with_time():
    _create_3d_dataset()
    target_points = np.array([
        [175.0, 0.0],
        [176.0, 1.0],
        [177.0, -1.0],
    ])

    with open_dataset(gi_file_3d) as ds:
        dv = ds['temperature']
        gi = dv.grid_interp()
        result = gi.to_points(target_points, to_crs=4326)

        results = list(result)
        assert len(results) == len(time_data)

        for time_val, values in results:
            assert isinstance(values, np.ndarray)
            assert values.shape == (3,)


def test_to_points_no_time():
    _create_2d_dataset()
    target_points = np.array([
        [175.0, 0.0],
        [176.0, 1.0],
    ])

    with open_dataset(gi_file_2d) as ds:
        dv = ds['temperature']
        gi = dv.grid_interp()
        results = list(gi.to_points(target_points, to_crs=4326))

        assert len(results) == 1
        time_val, values = results[0]
        assert time_val is None
        assert isinstance(values, np.ndarray)
        assert values.shape == (2,)


def test_interp_na():
    _create_2d_dataset()

    # Insert NaN values into the data
    data_with_nan = data_2d.copy()
    data_with_nan[5, 5] = np.nan
    data_with_nan[10, 3] = np.nan

    with open_dataset(gi_file_2d, flag='w') as ds:
        dv = ds['temperature']
        dv[:] = data_with_nan

    with open_dataset(gi_file_2d) as ds:
        dv = ds['temperature']
        gi = dv.grid_interp()
        results = list(gi.interp_na(method='linear'))

        assert len(results) == 1
        time_val, result = results[0]
        assert time_val is None
        assert isinstance(result, np.ndarray)
        assert result.shape == data_2d.shape
        # NaN values should be filled
        assert not np.any(np.isnan(result))


def test_regrid_levels():
    n_levels = 5
    z_data = np.arange(n_levels, dtype='float32')

    with open_dataset(gi_file_levels, flag='n') as ds:
        z_coord = ds.create.coord.generic('level', data=z_data, dtype='float32', chunk_shape=(n_levels,), axis='z')
        ds.create.coord.lat(data=lat_data, chunk_shape=(21,))
        ds.create.coord.lon(data=lon_data, chunk_shape=(11,))
        ds.create.crs.from_user_input(4326, 'longitude', 'latitude')

        data_dtype = dtypes.dtype('float32')
        level_data = np.zeros((n_levels, len(lat_data), len(lon_data)), dtype='float32')
        for k in range(n_levels):
            level_data[k] = data_2d * (k + 1)

        dv = ds.create.data_var.generic('temperature', ('level', 'latitude', 'longitude'), data_dtype, chunk_shape=(n_levels, 21, 11))
        dv[:] = level_data

        # Source levels data variable: same shape, holds actual height values at each point
        source_level_values = np.zeros_like(level_data)
        for k in range(n_levels):
            source_level_values[k] = float(k) * 100.0  # 0, 100, 200, 300, 400

        levels_dv = ds.create.data_var.generic('level_heights', ('level', 'latitude', 'longitude'), data_dtype, chunk_shape=(n_levels, 21, 11))
        levels_dv[:] = source_level_values

    with open_dataset(gi_file_levels) as ds:
        dv = ds['temperature']
        gi = dv.grid_interp()

        target_levels = np.array([0.0, 200.0, 400.0])
        results = list(gi.regrid_levels(target_levels, source_levels='level_heights', axis=0))

        assert len(results) == 1
        time_val, result = results[0]
        assert time_val is None
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(target_levels)
        assert result.shape[1] == len(lat_data)
        assert result.shape[2] == len(lon_data)
