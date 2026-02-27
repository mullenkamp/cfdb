import os
import pytest
import numpy as np
import pathlib
import shapely

from cfdb import open_dataset, dtypes

script_path = pathlib.Path(os.path.realpath(os.path.dirname(__file__)))

gi_file_2d = script_path.joinpath('test_gi_2d.cfdb')
gi_file_3d = script_path.joinpath('test_gi_3d.cfdb')
gi_file_no_crs = script_path.joinpath('test_gi_no_crs.cfdb')
gi_file_no_axis = script_path.joinpath('test_gi_no_axis.cfdb')
gi_file_levels = script_path.joinpath('test_gi_levels.cfdb')
pi_file_2d = script_path.joinpath('test_pi_2d.cfdb')
pi_file_3d = script_path.joinpath('test_pi_3d.cfdb')
pi_file_no_crs = script_path.joinpath('test_pi_no_crs.cfdb')
target_grid_file = script_path.joinpath('test_target_grid.cfdb')
target_point_file = script_path.joinpath('test_target_point.cfdb')

# Test data for grid datasets
lat_data = np.linspace(-10, 10, 21, dtype='float32')
lon_data = np.linspace(170, 180, 11, dtype='float32')
time_data = np.array(['2020-01-01', '2020-01-02', '2020-01-03'], dtype='datetime64[D]')

# 2D data: lat x lon (linear gradient)
data_2d = np.outer(lat_data, lon_data).astype('float32')

# 3D data: time x lat x lon
data_3d = np.stack([data_2d * (i + 1) for i in range(len(time_data))]).astype('float32')

# Test data for ts_ortho datasets
n_points = 50
point_lons = np.linspace(170, 180, n_points, dtype='float64')
point_lats = np.linspace(-10, 10, n_points, dtype='float64')
point_geoms = [shapely.Point(x, y) for x, y in zip(point_lons, point_lats)]

# 1D point data (linear gradient based on lon)
point_data_1d = point_lons.astype('float32')

# 2D point data: time x points
point_data_2d = np.stack([point_data_1d * (i + 1) for i in range(len(time_data))]).astype('float32')


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    def remove_test_data():
        for fp in [gi_file_2d, gi_file_3d, gi_file_no_crs, gi_file_no_axis, gi_file_levels,
                   pi_file_2d, pi_file_3d, pi_file_no_crs, target_grid_file, target_point_file]:
            if fp.exists():
                fp.unlink()
    request.addfinalizer(remove_test_data)


##############################
### Grid dataset helpers


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


##############################
### ts_ortho dataset helpers


def _create_point_2d_dataset():
    with open_dataset(pi_file_2d, flag='n', dataset_type='ts_ortho') as ds:
        geo_coord = ds.create.coord.point()
        geo_coord.append(point_geoms)
        ds.create.crs.from_user_input(4326, xy_coord='point')

        data_dtype = dtypes.dtype('float32')
        dv = ds.create.data_var.generic('temperature', ('point',), data_dtype, chunk_shape=(n_points,))
        dv[:] = point_data_1d


def _create_point_3d_dataset():
    with open_dataset(pi_file_3d, flag='n', dataset_type='ts_ortho') as ds:
        geo_coord = ds.create.coord.point()
        geo_coord.append(point_geoms)
        ds.create.coord.time(data=time_data, dtype=time_data.dtype)
        ds.create.crs.from_user_input(4326, xy_coord='point')

        data_dtype = dtypes.dtype('float32')
        dv = ds.create.data_var.generic('temperature', ('time', 'point'), data_dtype, chunk_shape=(3, n_points))
        dv[:] = point_data_2d


##############################
### Grid interpolation tests (renamed from grid_interp)


def test_axis_auto_detection():
    _create_3d_dataset()
    with open_dataset(gi_file_3d) as ds:
        dv = ds['temperature']
        gi = dv.interp()
        assert gi._x_name == 'longitude'
        assert gi._y_name == 'latitude'
        assert gi._iter_dim_name == 'time'
        assert gi._z_name is None


def test_explicit_coord_names():
    with open_dataset(gi_file_3d) as ds:
        dv = ds['temperature']
        gi = dv.interp(x='longitude', y='latitude', iter_dim='time')
        assert gi._x_name == 'longitude'
        assert gi._y_name == 'latitude'
        assert gi._iter_dim_name == 'time'


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
            dv.interp(x='longitude', y='latitude')


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
            dv.interp()


def test_to_grid_no_time():
    _create_2d_dataset()
    with open_dataset(gi_file_2d) as ds:
        dv = ds['temperature']
        gi = dv.interp()
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
        gi = dv.interp()
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
        gi = dv.interp()
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
        gi = dv.interp()
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
        gi = dv.interp()
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
        gi = dv.interp()

        target_levels = np.array([0.0, 200.0, 400.0])
        results = list(gi.regrid_levels(target_levels, source_levels='level_heights', axis=0))

        assert len(results) == 1
        time_val, result = results[0]
        assert time_val is None
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(target_levels)
        assert result.shape[1] == len(lat_data)
        assert result.shape[2] == len(lon_data)


##############################
### ts_ortho interpolation tests


def test_ts_ortho_axis_detection():
    _create_point_2d_dataset()
    with open_dataset(pi_file_2d) as ds:
        dv = ds['temperature']
        pi = dv.interp()
        assert pi._xy_name == 'point'
        assert pi._iter_dim_name is None
        assert pi._z_name is None


def test_ts_ortho_to_grid():
    _create_point_2d_dataset()
    with open_dataset(pi_file_2d) as ds:
        dv = ds['temperature']
        pi = dv.interp()
        results = list(pi.to_grid(grid_res=1.0, to_crs=4326))

        assert len(results) == 1
        time_val, grid = results[0]
        assert time_val is None
        assert isinstance(grid, np.ndarray)
        assert grid.ndim == 2
        assert grid.shape[0] > 0
        assert grid.shape[1] > 0


def test_ts_ortho_to_points():
    _create_point_2d_dataset()
    target_pts = np.array([
        [175.0, 0.0],
        [176.0, 1.0],
        [177.0, -1.0],
    ])

    with open_dataset(pi_file_2d) as ds:
        dv = ds['temperature']
        pi = dv.interp()
        results = list(pi.to_points(target_pts, to_crs=4326))

        assert len(results) == 1
        time_val, values = results[0]
        assert time_val is None
        assert isinstance(values, np.ndarray)
        assert values.shape == (3,)


def test_ts_ortho_to_grid_with_time():
    _create_point_3d_dataset()
    with open_dataset(pi_file_3d) as ds:
        dv = ds['temperature']
        pi = dv.interp()
        results = list(pi.to_grid(grid_res=1.0, to_crs=4326))

        assert len(results) == len(time_data)

        for time_val, grid in results:
            assert isinstance(grid, np.ndarray)
            assert grid.ndim == 2
            assert grid.shape[0] > 0
            assert grid.shape[1] > 0


def test_ts_ortho_interp_na_error():
    _create_point_2d_dataset()
    with open_dataset(pi_file_2d) as ds:
        dv = ds['temperature']
        pi = dv.interp()
        with pytest.raises(NotImplementedError, match="interp_na"):
            list(pi.interp_na())


def test_ts_ortho_regrid_levels_error():
    _create_point_2d_dataset()
    with open_dataset(pi_file_2d) as ds:
        dv = ds['temperature']
        pi = dv.interp()
        with pytest.raises(NotImplementedError, match="regrid_levels"):
            list(pi.regrid_levels(target_levels=[0, 100], source_levels='foo'))


##############################
### Shapely Point input tests


def test_to_points_shapely_input():
    """Pass shapely Points as target_points, verify same results as numpy array input."""
    _create_2d_dataset()

    target_np = np.array([
        [175.0, 0.0],
        [176.0, 1.0],
    ])
    target_shapely = [shapely.Point(175.0, 0.0), shapely.Point(176.0, 1.0)]

    with open_dataset(gi_file_2d) as ds:
        dv = ds['temperature']

        results_np = list(dv.interp().to_points(target_np, to_crs=4326))
        results_shapely = list(dv.interp().to_points(target_shapely, to_crs=4326))

        assert len(results_np) == len(results_shapely)
        for (t1, v1), (t2, v2) in zip(results_np, results_shapely):
            assert t1 == t2
            np.testing.assert_array_almost_equal(v1, v2)


def test_to_points_shapely_input_ts_ortho():
    """Pass shapely Points as target_points for ts_ortho datasets."""
    _create_point_2d_dataset()

    target_np = np.array([
        [175.0, 0.0],
        [176.0, 1.0],
    ])
    target_shapely = [shapely.Point(175.0, 0.0), shapely.Point(176.0, 1.0)]

    with open_dataset(pi_file_2d) as ds:
        dv = ds['temperature']

        results_np = list(dv.interp().to_points(target_np, to_crs=4326))
        results_shapely = list(dv.interp().to_points(target_shapely, to_crs=4326))

        assert len(results_np) == len(results_shapely)
        for (t1, v1), (t2, v2) in zip(results_np, results_shapely):
            assert t1 == t2
            np.testing.assert_array_almost_equal(v1, v2)


##############################
### DataVariable as target input tests


def test_to_points_data_var_target():
    """Pass a ts_ortho DataVariable as target, verify output matches raw point input."""
    _create_2d_dataset()
    _create_point_2d_dataset()

    # Get target points from the ts_ortho dataset for direct comparison
    with open_dataset(pi_file_2d) as target_ds:
        target_geom = target_ds['point'].data
        raw_points = shapely.get_coordinates(target_geom)
        target_crs = target_ds.crs

    with open_dataset(gi_file_2d) as src_ds:
        dv = src_ds['temperature']

        # Using raw points
        results_raw = list(dv.interp().to_points(raw_points, to_crs=target_crs))

    with open_dataset(gi_file_2d) as src_ds, open_dataset(pi_file_2d) as target_ds:
        src_dv = src_ds['temperature']
        target_dv = target_ds['temperature']

        # Using DataVariable target
        results_dv = list(src_dv.interp().to_points(target_dv))

    assert len(results_raw) == len(results_dv)
    for (t1, v1), (t2, v2) in zip(results_raw, results_dv):
        assert t1 == t2
        np.testing.assert_array_almost_equal(v1, v2)


def test_to_grid_data_var_target():
    """Pass a grid DataVariable as target, verify output shape matches destination."""
    _create_3d_dataset()

    # Create a target grid dataset with different resolution
    with open_dataset(target_grid_file, flag='n') as ds:
        target_lat = np.linspace(-5, 5, 11, dtype='float32')
        target_lon = np.linspace(172, 178, 7, dtype='float32')
        ds.create.coord.lat(data=target_lat, chunk_shape=(11,))
        ds.create.coord.lon(data=target_lon, chunk_shape=(7,))
        ds.create.crs.from_user_input(4326, 'longitude', 'latitude')

        data_dtype = dtypes.dtype('float32')
        ds.create.data_var.generic('temperature', ('latitude', 'longitude'), data_dtype, chunk_shape=(11, 7))

    with open_dataset(gi_file_3d) as src_ds, open_dataset(target_grid_file) as target_ds:
        src_dv = src_ds['temperature']
        target_dv = target_ds['temperature']

        results = list(src_dv.interp().to_grid(target_dv))

        assert len(results) == len(time_data)
        for time_val, grid in results:
            assert isinstance(grid, np.ndarray)
            assert grid.ndim == 2
            assert grid.shape[0] > 0
            assert grid.shape[1] > 0


def test_to_grid_data_var_target_ts_ortho():
    """Pass a grid DataVariable as target to a ts_ortho source."""
    _create_point_3d_dataset()

    # Create target grid
    with open_dataset(target_grid_file, flag='n') as ds:
        target_lat = np.linspace(-5, 5, 11, dtype='float32')
        target_lon = np.linspace(172, 178, 7, dtype='float32')
        ds.create.coord.lat(data=target_lat, chunk_shape=(11,))
        ds.create.coord.lon(data=target_lon, chunk_shape=(7,))
        ds.create.crs.from_user_input(4326, 'longitude', 'latitude')

        data_dtype = dtypes.dtype('float32')
        ds.create.data_var.generic('temperature', ('latitude', 'longitude'), data_dtype, chunk_shape=(11, 7))

    with open_dataset(pi_file_3d) as src_ds, open_dataset(target_grid_file) as target_ds:
        src_dv = src_ds['temperature']
        target_dv = target_ds['temperature']

        results = list(src_dv.interp().to_grid(target_dv))

        assert len(results) == len(time_data)
        for time_val, grid in results:
            assert isinstance(grid, np.ndarray)
            assert grid.ndim == 2
            assert grid.shape[0] > 0
            assert grid.shape[1] > 0


def test_to_points_data_var_target_ts_ortho_source():
    """Pass a ts_ortho DataVariable as target to a ts_ortho source."""
    _create_point_2d_dataset()

    # Create a second ts_ortho dataset with different points as target
    target_geoms = [shapely.Point(175.0 + i * 0.5, -2.0 + i * 0.5) for i in range(5)]

    with open_dataset(target_point_file, flag='n', dataset_type='ts_ortho') as ds:
        geo_coord = ds.create.coord.point()
        geo_coord.append(target_geoms)
        ds.create.crs.from_user_input(4326, xy_coord='point')

        data_dtype = dtypes.dtype('float32')
        ds.create.data_var.generic('temperature', ('point',), data_dtype, chunk_shape=(5,))

    with open_dataset(pi_file_2d) as src_ds, open_dataset(target_point_file) as target_ds:
        src_dv = src_ds['temperature']
        target_dv = target_ds['temperature']

        results = list(src_dv.interp().to_points(target_dv))

        assert len(results) == 1
        time_val, values = results[0]
        assert time_val is None
        assert isinstance(values, np.ndarray)
        assert values.shape == (5,)
