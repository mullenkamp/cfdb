import os
import pytest
import numpy as np
import pathlib

from cfdb import open_dataset, combine, dtypes

script_path = pathlib.Path(os.path.realpath(os.path.dirname(__file__)))


def _make_time(start_day, n_days):
    """Create datetime64[D] array: start_day is days since epoch."""
    base = np.datetime64('1970-01-01', 'D')
    return np.array([base + np.timedelta64(i, 'D') for i in range(start_day, start_day + n_days)])


@pytest.fixture
def ds_paths():
    """Provide file paths and clean them up before and after tests."""
    paths = [
        script_path / 'combine_ds1.cfdb',
        script_path / 'combine_ds2.cfdb',
        script_path / 'combine_ds3.cfdb',
        script_path / 'combine_out.cfdb',
    ]
    for p in paths:
        if p.exists():
            p.unlink()
    yield paths
    for p in paths:
        if p.exists():
            p.unlink()


def _make_grid_dataset(path, lat_data, lon_data, time_data, var_data, var_name='temperature'):
    """Helper to create a grid dataset with lat, lon, time, and one data var."""
    data_dtype = dtypes.dtype('float32')
    n_time = time_data.shape[0]
    with open_dataset(path, 'n') as ds:
        ds.create.coord.lat(data=lat_data, chunk_shape=(20,))
        ds.create.coord.lon(data=lon_data, chunk_shape=(20,))
        ds.create.coord.time(data=time_data, dtype=time_data.dtype)
        dv = ds.create.data_var.generic(
            var_name, ('latitude', 'longitude', 'time'),
            data_dtype, chunk_shape=(20, 20, n_time)
        )
        dv[:] = var_data
        ds.attrs['source'] = str(path.name)


def test_combine_non_overlapping_time(ds_paths):
    """Combine two datasets with different time ranges."""
    p1, p2, _, out = ds_paths

    lat = np.linspace(0, 4.9, 50, dtype='float32')
    lon = np.linspace(-5, -0.1, 50, dtype='float32')
    t1 = _make_time(0, 5)
    t2 = _make_time(5, 5)

    d1 = np.ones((50, 50, 5), dtype='float32')
    d2 = np.ones((50, 50, 5), dtype='float32') * 2

    _make_grid_dataset(p1, lat, lon, t1, d1)
    _make_grid_dataset(p2, lat, lon, t2, d2)

    result = combine([p1, p2], out)

    assert result['time'].shape == (10,)
    assert np.allclose(result['temperature'].data[:, :, :5], d1)
    assert np.allclose(result['temperature'].data[:, :, 5:], d2)

    result.close()


def test_combine_non_overlapping_spatial(ds_paths):
    """Combine two datasets with different latitude ranges."""
    p1, p2, _, out = ds_paths

    lat1 = np.linspace(0, 4.9, 50, dtype='float32')
    lat2 = np.linspace(5, 9.9, 50, dtype='float32')
    lon = np.linspace(-5, -0.1, 50, dtype='float32')
    t = _make_time(0, 5)

    d1 = np.ones((50, 50, 5), dtype='float32')
    d2 = np.ones((50, 50, 5), dtype='float32') * 2

    _make_grid_dataset(p1, lat1, lon, t, d1)
    _make_grid_dataset(p2, lat2, lon, t, d2)

    result = combine([p1, p2], out)

    assert result['latitude'].shape == (100,)
    assert np.allclose(result['temperature'].data[:50, :, :], d1)
    assert np.allclose(result['temperature'].data[50:, :, :], d2)

    result.close()


def test_combine_overlap_last(ds_paths):
    """Overlapping time, overlap='last' -> last dataset wins."""
    p1, p2, _, out = ds_paths

    lat = np.linspace(0, 1.9, 20, dtype='float32')
    lon = np.linspace(0, 1.9, 20, dtype='float32')
    t1 = _make_time(0, 5)   # days 0-4
    t2 = _make_time(3, 5)   # days 3-7

    d1 = np.ones((20, 20, 5), dtype='float32')
    d2 = np.ones((20, 20, 5), dtype='float32') * 2

    _make_grid_dataset(p1, lat, lon, t1, d1)
    _make_grid_dataset(p2, lat, lon, t2, d2)

    result = combine([p1, p2], out, overlap='last')

    combined_time = _make_time(0, 8)
    assert result['time'].shape == (8,)
    assert np.all(result['time'].data == combined_time)

    data = result['temperature'].data
    assert np.allclose(data[:, :, :3], 1.0)  # days 0,1,2 from ds1
    assert np.allclose(data[:, :, 3:], 2.0)  # days 3-7 from ds2

    result.close()


def test_combine_overlap_error(ds_paths):
    """Overlapping data with overlap='error' should raise."""
    p1, p2, _, out = ds_paths

    lat = np.linspace(0, 1.9, 20, dtype='float32')
    lon = np.linspace(0, 1.9, 20, dtype='float32')
    t1 = _make_time(0, 5)
    t2 = _make_time(3, 5)

    d1 = np.ones((20, 20, 5), dtype='float32')
    d2 = np.ones((20, 20, 5), dtype='float32') * 2

    _make_grid_dataset(p1, lat, lon, t1, d1)
    _make_grid_dataset(p2, lat, lon, t2, d2)

    with pytest.raises(ValueError, match='Overlap'):
        combine([p1, p2], out, overlap='error')

    if out.exists():
        out.unlink()


def test_combine_incompatible_dtypes(ds_paths):
    """Datasets with different dtypes for the same coord should error."""
    p1, p2, _, out = ds_paths

    lat32 = np.linspace(0, 4.9, 50, dtype='float32')
    lat64 = np.linspace(5, 9.9, 50, dtype='float64')
    lon = np.linspace(-5, -0.1, 50, dtype='float32')
    t = _make_time(0, 5)

    _make_grid_dataset(p1, lat32, lon, t, np.ones((50, 50, 5), dtype='float32'))

    data_dtype = dtypes.dtype('float32')
    with open_dataset(p2, 'n') as ds:
        ds.create.coord.generic('latitude', data=lat64, chunk_shape=(20,), axis='y')
        ds.create.coord.lon(data=lon, chunk_shape=(20,))
        ds.create.coord.time(data=t, dtype=t.dtype)
        dv = ds.create.data_var.generic(
            'temperature', ('latitude', 'longitude', 'time'),
            data_dtype, chunk_shape=(20, 20, 5)
        )
        dv[:] = np.ones((50, 50, 5), dtype='float32')

    with pytest.raises(ValueError, match='incompatible dtypes'):
        combine([p1, p2], out)


def test_combine_different_dataset_types(ds_paths):
    """Combining grid and ts_ortho should error."""
    p1, p2, _, out = ds_paths
    import shapely

    lat = np.linspace(0, 4.9, 50, dtype='float32')
    lon = np.linspace(-5, -0.1, 50, dtype='float32')
    t = _make_time(0, 5)
    _make_grid_dataset(p1, lat, lon, t, np.ones((50, 50, 5), dtype='float32'))

    geo_data = [shapely.Point(x, y) for x, y in zip(lon, lat)]
    with open_dataset(p2, 'n', dataset_type='ts_ortho') as ds:
        ds.create.coord.point()
        ds['point'].append(geo_data)
        ds.create.coord.time(data=t, dtype=t.dtype)

    with pytest.raises(ValueError, match='dataset_type'):
        combine([p1, p2], out)


def test_combine_single_dataset(ds_paths):
    """Combining a single dataset produces an identical copy."""
    p1, _, _, out = ds_paths

    lat = np.linspace(0, 4.9, 50, dtype='float32')
    lon = np.linspace(-5, -0.1, 50, dtype='float32')
    t = _make_time(0, 5)
    d = np.linspace(0, 999, 50 * 50 * 5, dtype='float32').reshape(50, 50, 5)

    _make_grid_dataset(p1, lat, lon, t, d)

    result = combine([p1], out)

    assert np.allclose(result['latitude'].data, lat)
    assert np.allclose(result['longitude'].data, lon)
    assert np.all(result['time'].data == t)
    assert np.allclose(result['temperature'].data, d)

    result.close()


def test_combine_three_datasets(ds_paths):
    """Combine three non-overlapping time datasets."""
    p1, p2, p3, out = ds_paths

    lat = np.linspace(0, 1.9, 20, dtype='float32')
    lon = np.linspace(0, 1.9, 20, dtype='float32')
    t1 = _make_time(0, 3)
    t2 = _make_time(3, 3)
    t3 = _make_time(6, 3)

    d1 = np.ones((20, 20, 3), dtype='float32') * 1
    d2 = np.ones((20, 20, 3), dtype='float32') * 2
    d3 = np.ones((20, 20, 3), dtype='float32') * 3

    _make_grid_dataset(p1, lat, lon, t1, d1)
    _make_grid_dataset(p2, lat, lon, t2, d2)
    _make_grid_dataset(p3, lat, lon, t3, d3)

    result = combine([p1, p2, p3], out)

    assert result['time'].shape == (9,)
    data = result['temperature'].data
    assert np.allclose(data[:, :, :3], 1.0)
    assert np.allclose(data[:, :, 3:6], 2.0)
    assert np.allclose(data[:, :, 6:], 3.0)

    result.close()


def test_combine_with_open_datasets(ds_paths):
    """Pass open Dataset objects instead of file paths."""
    p1, p2, _, out = ds_paths

    lat = np.linspace(0, 1.9, 20, dtype='float32')
    lon = np.linspace(0, 1.9, 20, dtype='float32')
    t1 = _make_time(0, 5)
    t2 = _make_time(5, 5)

    d1 = np.ones((20, 20, 5), dtype='float32')
    d2 = np.ones((20, 20, 5), dtype='float32') * 2

    _make_grid_dataset(p1, lat, lon, t1, d1)
    _make_grid_dataset(p2, lat, lon, t2, d2)

    with open_dataset(p1) as ds1, open_dataset(p2) as ds2:
        result = combine([ds1, ds2], out)
        assert result['time'].shape == (10,)
        result.close()


def test_combine_include_data_vars(ds_paths):
    """Test include_data_vars filter."""
    p1, _, _, out = ds_paths

    lat = np.linspace(0, 1.9, 20, dtype='float32')
    lon = np.linspace(0, 1.9, 20, dtype='float32')
    t = _make_time(0, 5)
    d = np.ones((20, 20, 5), dtype='float32')

    data_dtype = dtypes.dtype('float32')
    with open_dataset(p1, 'n') as ds:
        ds.create.coord.lat(data=lat, chunk_shape=(20,))
        ds.create.coord.lon(data=lon, chunk_shape=(20,))
        ds.create.coord.time(data=t, dtype=t.dtype)
        dv1 = ds.create.data_var.generic('temp', ('latitude', 'longitude', 'time'), data_dtype, chunk_shape=(20, 20, 5))
        dv1[:] = d
        dv2 = ds.create.data_var.generic('wind', ('latitude', 'longitude', 'time'), data_dtype, chunk_shape=(20, 20, 5))
        dv2[:] = d * 2

    result = combine([p1], out, include_data_vars=['temp'])

    assert 'temp' in result.data_var_names
    assert 'wind' not in result.data_var_names

    result.close()


def test_combine_exclude_data_vars(ds_paths):
    """Test exclude_data_vars filter."""
    p1, _, _, out = ds_paths

    lat = np.linspace(0, 1.9, 20, dtype='float32')
    lon = np.linspace(0, 1.9, 20, dtype='float32')
    t = _make_time(0, 5)
    d = np.ones((20, 20, 5), dtype='float32')

    data_dtype = dtypes.dtype('float32')
    with open_dataset(p1, 'n') as ds:
        ds.create.coord.lat(data=lat, chunk_shape=(20,))
        ds.create.coord.lon(data=lon, chunk_shape=(20,))
        ds.create.coord.time(data=t, dtype=t.dtype)
        dv1 = ds.create.data_var.generic('temp', ('latitude', 'longitude', 'time'), data_dtype, chunk_shape=(20, 20, 5))
        dv1[:] = d
        dv2 = ds.create.data_var.generic('wind', ('latitude', 'longitude', 'time'), data_dtype, chunk_shape=(20, 20, 5))
        dv2[:] = d * 2

    result = combine([p1], out, exclude_data_vars=['wind'])

    assert 'temp' in result.data_var_names
    assert 'wind' not in result.data_var_names

    result.close()


def test_combine_with_sel(ds_paths):
    """Test sel parameter to subset during combine."""
    p1, _, _, out = ds_paths

    lat = np.linspace(0, 9.9, 100, dtype='float32')
    lon = np.linspace(-5, 4.9, 100, dtype='float32')
    t = _make_time(0, 10)
    d = np.linspace(0, 999, 100 * 100 * 10, dtype='float32').reshape(100, 100, 10)

    _make_grid_dataset(p1, lat, lon, t, d)

    result = combine(
        [p1], out,
        sel={'time': slice('1970-01-03', '1970-01-07')},
    )

    assert result['time'].shape[0] < 10
    assert result['time'].shape[0] > 0

    result.close()


def test_combine_crs_preserved(ds_paths):
    """Test that CRS is preserved in combined output."""
    p1, p2, _, out = ds_paths

    lat = np.linspace(0, 4.9, 50, dtype='float32')
    lon = np.linspace(-5, -0.1, 50, dtype='float32')
    t1 = _make_time(0, 5)
    t2 = _make_time(5, 5)

    d1 = np.ones((50, 50, 5), dtype='float32')
    d2 = np.ones((50, 50, 5), dtype='float32') * 2

    data_dtype = dtypes.dtype('float32')
    for p, t, d in [(p1, t1, d1), (p2, t2, d2)]:
        with open_dataset(p, 'n') as ds:
            ds.create.coord.lat(data=lat, chunk_shape=(20,))
            ds.create.coord.lon(data=lon, chunk_shape=(20,))
            ds.create.coord.time(data=t, dtype=t.dtype)
            ds.create.crs.from_user_input(4326, 'longitude', 'latitude')
            dv = ds.create.data_var.generic(
                'temperature', ('latitude', 'longitude', 'time'),
                data_dtype, chunk_shape=(20, 20, 5)
            )
            dv[:] = d

    result = combine([p1, p2], out)

    assert result.crs is not None
    assert result._sys_meta.crs is not None

    result.close()


def test_combine_empty_datasets_error():
    """Empty dataset list should error."""
    with pytest.raises(ValueError, match='empty'):
        combine([], 'out.cfdb')


def test_combine_invalid_overlap(ds_paths):
    """Invalid overlap value should error."""
    p1, _, _, out = ds_paths

    lat = np.linspace(0, 1.9, 20, dtype='float32')
    lon = np.linspace(0, 1.9, 20, dtype='float32')
    t = _make_time(0, 5)
    d = np.ones((20, 20, 5), dtype='float32')
    _make_grid_dataset(p1, lat, lon, t, d)

    with pytest.raises(ValueError, match='overlap'):
        combine([p1], out, overlap='invalid')


def test_combine_attrs_merged(ds_paths):
    """Test that global and variable attributes are merged."""
    p1, p2, _, out = ds_paths

    lat = np.linspace(0, 1.9, 20, dtype='float32')
    lon = np.linspace(0, 1.9, 20, dtype='float32')
    t1 = _make_time(0, 5)
    t2 = _make_time(5, 5)

    data_dtype = dtypes.dtype('float32')
    with open_dataset(p1, 'n') as ds:
        ds.create.coord.lat(data=lat, chunk_shape=(20,))
        ds.create.coord.lon(data=lon, chunk_shape=(20,))
        ds.create.coord.time(data=t1, dtype=t1.dtype)
        dv = ds.create.data_var.generic('temp', ('latitude', 'longitude', 'time'), data_dtype, chunk_shape=(20, 20, 5))
        dv[:] = np.ones((20, 20, 5), dtype='float32')
        dv.attrs['units'] = 'K'
        ds.attrs['history'] = 'first dataset'

    with open_dataset(p2, 'n') as ds:
        ds.create.coord.lat(data=lat, chunk_shape=(20,))
        ds.create.coord.lon(data=lon, chunk_shape=(20,))
        ds.create.coord.time(data=t2, dtype=t2.dtype)
        dv = ds.create.data_var.generic('temp', ('latitude', 'longitude', 'time'), data_dtype, chunk_shape=(20, 20, 5))
        dv[:] = np.ones((20, 20, 5), dtype='float32') * 2
        dv.attrs['units'] = 'K'
        ds.attrs['history'] = 'second dataset'

    result = combine([p1, p2], out)

    assert result.attrs['history'] == 'first dataset'
    assert result['temp'].attrs['units'] == 'K'

    result.close()
