import os
import pytest
import numpy as np
from time import time
import pathlib
import h5netcdf
import shapely

from cfdb import open_dataset, cfdb_to_netcdf4, netcdf4_to_cfdb, dtypes

###################################################
### Parameters

script_path = pathlib.Path(os.path.realpath(os.path.dirname(__file__)))

file_path = script_path.joinpath('test1.cfdb')
flag = 'n'

new_file_path = script_path.joinpath('test2.cfdb')
nc_file_path = script_path.joinpath('test1.nc')

name = 'air_temp'
coords = ('latitude', 'longitude', 'time')
# dtype_encoded = 'int32'
chunk_shape = (20, 30, 10)
# fillvalue = None
# scale_factor = 0.1
# add_offset = None
sel = (slice(1, 4), slice(None, None), slice(2, 5))
sel2 = (slice(1, 1), slice(None, None), slice(2, 5))
loc_sel = (slice(0.4, 0.7), slice(None, None), slice('1970-01-04', '1970-01-10'))
ds_sel = {'latitude': slice(1, 4), 'time': slice(2, 5)}
ds_loc_sel = {'latitude': slice(0.4, 0.7), 'time': slice('1970-01-04', '1970-01-10')}

lat_data = np.linspace(0, 9.9, 100, dtype='float32')
lon_data = np.linspace(-5, 4.9, 100, dtype='float32')
other_lat_data = np.linspace(-1, -0.1, 10, dtype='float32')
time_data = np.linspace(0, 10, 10, dtype='datetime64[D]')

geo_data = [shapely.Point(x, y) for x, y in zip(lon_data, lat_data)]

data_var_data = np.linspace(0, 9999.9, 100000, dtype='float32').reshape(100, 100, 10)

new_chunk_shape = (41, 41, 10)

@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    """Cleanup a test data."""
    def remove_test_data():
        if file_path.exists():
            file_path.unlink()
        if new_file_path.exists():
            new_file_path.unlink()
        if nc_file_path.exists():
            nc_file_path.unlink()

    request.addfinalizer(remove_test_data)


##############################
### Tests


def test_coord_creation_ts_ortho():
    with open_dataset(file_path, flag='n', dataset_type='ts_ortho') as ds:
        geo_coord = ds.create.coord.point()
        geo_coord.append(geo_data)

        _ = ds.create.crs.from_user_input(4326, xy_coord='point')

        time_coord = ds.create.coord.time(data=time_data, dtype=time_data.dtype)
        print(time_coord)

        assert np.all(time_coord.data == time_data)

        ds.attrs['history'] = 'Created some coords yo'


def test_coord_creation_grid():
    with open_dataset(file_path, flag='n') as ds:
        _ = ds.create.coord.height()
        del ds['height']
        _ = ds.create.coord.altitude()
        del ds['altitude']
        # _ = ds.create.coord.xy_from_crs(2193)
        # del ds['x']
        # del ds['y']

        lat_coord = ds.create.coord.lat(data=lat_data, chunk_shape=(20,))
        lat_coord.prepend(other_lat_data)
        del ds['latitude']
        lat_coord = ds.create.coord.lat(chunk_shape=(20,))
        lat_coord.append(lat_data)
        print(lat_coord)
        # print(lat_coord.data[6:12])

        assert np.allclose(lat_coord.data, lat_data)

        lon_coord = ds.create.coord.lon(data=lon_data, chunk_shape=(20,))
        print(lon_coord)
        # print(lat_coord.data[6:12])

        assert np.allclose(lon_coord.data, lon_data)

        time_coord = ds.create.coord.time(data=time_data, dtype=time_data.dtype)
        print(time_coord)

        assert np.all(time_coord.data == time_data)

        ds.attrs['history'] = 'Created some coords yo'


def test_crs():
    """

    """
    with open_dataset(file_path, flag='w') as ds:
        crs = ds.create.crs.from_user_input(4326, 'longitude', 'latitude')

        assert crs  == ds.crs


def test_data_var_creation():
    data_dtype = dtypes.dtype(data_var_data.dtype, 1, 0, 10000)

    with open_dataset(file_path, flag='w') as ds:
        data_var = ds.create.data_var.generic(name, coords, data_dtype, chunk_shape=chunk_shape)
        data_var[:] = data_var_data
        data_var.attrs['test'] = ['test1']
        print(data_var)
        assert np.allclose(data_var.data, data_var_data)


def test_select():
    with open_dataset(file_path) as ds:
        data_var = ds[name]
        view1 = data_var[sel]
        assert np.allclose(view1.data, data_var_data[sel])

        view2 = data_var.loc[loc_sel]
        assert np.allclose(view2.data, data_var_data[(slice(4, 7), slice(None, None), slice(3, 9))])

        view3 = ds.select(ds_sel)
        assert np.allclose(view3[name].data, data_var_data[sel])

        view4 = ds.select_loc(ds_loc_sel)
        assert np.allclose(view4[name].data, data_var_data[(slice(4, 7), slice(None, None), slice(3, 9))])

        try:
            view_fail = data_var[sel2]
        except ValueError:
            pass

def test_rechunker_assignment():
    data_dtype = dtypes.dtype(data_var_data.dtype, 1, 0, 10000)

    with open_dataset(file_path, flag='w') as ds:
        data_var = ds[name]

        data_var2 = ds.create.data_var.generic(name + '2', coords, data_dtype, chunk_shape=new_chunk_shape)

        rechunker = data_var.rechunker()
        rechunk = rechunker.rechunk(new_chunk_shape)

        for write_chunk, data in rechunk:
            data_var2.set(write_chunk, data)

        assert np.allclose(data_var2.data, data_var_data)

        del ds[name + '2']

        rechunker = data_var.rechunker()
        rechunk = rechunker.rechunk(new_chunk_shape)

        new_data = np.full(data_var_data.shape, np.nan, data_var_data.dtype)
        for write_chunk, data in rechunk:
            new_data[write_chunk] = data

        assert np.allclose(new_data, data_var_data)

        new_data = np.full(data_var_data.shape, np.nan, data_var_data.dtype)
        for write_chunk, data in data_var.groupby('latitude'):
            new_data[write_chunk] = data

        assert np.allclose(new_data, data_var_data)


def test_serialize():
    with open_dataset(file_path) as ds:
        new_ds = ds.copy(new_file_path)
        print(new_ds)
        new_ds.close()
        ds.to_netcdf4(nc_file_path)

    with open_dataset(new_file_path) as ds:
        print(ds)

    with h5netcdf.File(nc_file_path) as f:
        print(f)


def test_prune():
    with open_dataset(file_path, flag='w') as ds:
        removed = ds.prune()

    assert removed > 0


def test_cfdb_to_netcdf4():
    cfdb_to_netcdf4(file_path, nc_file_path, sel_loc=ds_loc_sel)

def test_netcdf4_to_cfdb():
    netcdf4_to_cfdb(nc_file_path, new_file_path)


##############################################
### Independent tests (use their own files)

ind_file_path = script_path.joinpath('test_ind.blt')

ind_lat_data = np.linspace(0, 9.9, 100, dtype='float32')
ind_lon_data = np.linspace(-5, 4.9, 100, dtype='float32')
ind_time_data = np.linspace(0, 9, 10, dtype='datetime64[D]')
ind_data_var_data = np.linspace(0, 9999.9, 100000, dtype='float32').reshape(100, 100, 10)


@pytest.fixture(scope="module")
def populated_dataset():
    """Create a populated grid dataset for independent tests."""
    data_dtype = dtypes.dtype(ind_data_var_data.dtype, 1, 0, 10000)

    with open_dataset(ind_file_path, flag='n') as ds:
        ds.create.coord.lat(data=ind_lat_data, chunk_shape=(20,))
        ds.create.coord.lon(data=ind_lon_data, chunk_shape=(20,))
        ds.create.coord.time(data=ind_time_data, dtype=ind_time_data.dtype)
        dv = ds.create.data_var.generic('temperature', ('latitude', 'longitude', 'time'), data_dtype, chunk_shape=(20, 30, 10))
        dv[:] = ind_data_var_data
        dv.attrs['units'] = 'K'
        ds.attrs['history'] = 'test dataset'

    yield ind_file_path

    if ind_file_path.exists():
        ind_file_path.unlink()


##############################
### Dataset repr and properties


def test_dataset_repr(populated_dataset):
    """Test that dataset __repr__ runs without error and contains key info."""
    with open_dataset(populated_dataset) as ds:
        rep = repr(ds)
        assert 'latitude' in rep
        assert 'longitude' in rep
        assert 'time' in rep
        assert 'temperature' in rep


def test_dataset_repr_closed():
    """Test repr on a closed dataset."""
    fp = script_path.joinpath('test_closed.blt')
    ds = open_dataset(fp, flag='n')
    ds.close()
    rep = repr(ds)
    assert 'closed' in rep
    fp.unlink()


def test_dataset_properties(populated_dataset):
    """Test var_names, coord_names, data_var_names, coords, data_vars, variables."""
    with open_dataset(populated_dataset) as ds:
        assert 'latitude' in ds.var_names
        assert 'longitude' in ds.var_names
        assert 'time' in ds.var_names
        assert 'temperature' in ds.var_names

        assert 'latitude' in ds.coord_names
        assert 'longitude' in ds.coord_names
        assert 'time' in ds.coord_names
        assert 'temperature' not in ds.coord_names

        assert 'temperature' in ds.data_var_names
        assert 'latitude' not in ds.data_var_names

        assert len(ds.coords) == 3
        assert len(ds.data_vars) == 1
        assert len(ds.variables) == 4


def test_dataset_container_protocol(populated_dataset):
    """Test __contains__, __iter__, __len__, __bool__."""
    with open_dataset(populated_dataset) as ds:
        assert 'latitude' in ds
        assert 'nonexistent' not in ds
        assert len(ds) == 4
        assert bool(ds) is True

        var_names = list(ds)
        assert len(var_names) == 4


def test_dataset_context_manager():
    """Test that context manager properly closes the file."""
    fp = script_path.joinpath('test_ctx.blt')
    with open_dataset(fp, flag='n') as ds:
        assert ds.is_open is True
    assert ds.is_open is False
    fp.unlink()


##############################
### Attributes


def test_attributes_read(populated_dataset):
    """Test reading attributes."""
    with open_dataset(populated_dataset) as ds:
        assert ds.attrs['history'] == 'test dataset'
        dv = ds['temperature']
        assert dv.attrs['units'] == 'K'


def test_attributes_write():
    """Test attribute set, get, update, pop, del, clear, contains, keys, values, items."""
    fp = script_path.joinpath('test_attrs.blt')
    with open_dataset(fp, flag='n') as ds:
        ds.create.coord.lat(data=ind_lat_data[:10], chunk_shape=(10,))

        # set/get
        ds.attrs['key1'] = 'value1'
        assert ds.attrs['key1'] == 'value1'
        assert ds.attrs.get('key1') == 'value1'
        assert ds.attrs.get('nonexistent') is None

        # contains
        assert 'key1' in ds.attrs

        # update
        ds.attrs.update({'key2': 42, 'key3': [1, 2, 3]})
        assert ds.attrs['key2'] == 42
        assert ds.attrs['key3'] == [1, 2, 3]

        # keys/values/items
        assert 'key1' in ds.attrs.keys()
        assert 'value1' in ds.attrs.values()
        items = dict(ds.attrs.items())
        assert items['key2'] == 42

        # pop
        val = ds.attrs.pop('key3')
        assert val == [1, 2, 3]
        assert 'key3' not in ds.attrs

        # del
        del ds.attrs['key2']
        assert 'key2' not in ds.attrs

        # clear
        ds.attrs['a'] = 1
        ds.attrs.clear()
        assert len(list(ds.attrs.keys())) == 0

    fp.unlink()


def test_attributes_read_only():
    """Test that writing to read-only attributes raises ValueError."""
    fp = script_path.joinpath('test_attrs_ro.blt')
    with open_dataset(fp, flag='n') as ds:
        ds.create.coord.lat(data=ind_lat_data[:10], chunk_shape=(10,))

    with open_dataset(fp) as ds:
        with pytest.raises(ValueError):
            ds.attrs['key'] = 'value'
        with pytest.raises(ValueError):
            ds.attrs.update({'key': 'value'})
        with pytest.raises(ValueError):
            ds.attrs.clear()
        with pytest.raises(ValueError):
            ds.attrs.pop('key')
        with pytest.raises(ValueError):
            del ds.attrs['key']

    fp.unlink()


def test_attributes_non_serializable():
    """Test that non-JSON-serializable values are rejected."""
    fp = script_path.joinpath('test_attrs_ns.blt')
    with open_dataset(fp, flag='n') as ds:
        ds.create.coord.lat(data=ind_lat_data[:10], chunk_shape=(10,))
        with pytest.raises(ValueError):
            ds.attrs['key'] = np.array([1, 2, 3])
    fp.unlink()


##############################
### Coordinate operations


def test_coordinate_repr(populated_dataset):
    """Test coordinate repr."""
    with open_dataset(populated_dataset) as ds:
        coord = ds['latitude']
        rep = repr(coord)
        assert 'latitude' in rep


def test_coordinate_properties(populated_dataset):
    """Test coordinate shape, chunk_shape, ndims, name."""
    with open_dataset(populated_dataset) as ds:
        lat = ds['latitude']
        assert lat.shape == (100,)
        assert lat.chunk_shape == (20,)
        assert lat.ndims == 1
        assert lat.name == 'latitude'


def test_coordinate_data_integrity(populated_dataset):
    """Test that coordinate data is correctly stored and retrieved."""
    with open_dataset(populated_dataset) as ds:
        lat = ds['latitude']
        assert np.allclose(lat.data, ind_lat_data)

        lon = ds['longitude']
        assert np.allclose(lon.data, ind_lon_data)

        t = ds['time']
        assert np.all(t.data == ind_time_data)


def test_coordinate_view(populated_dataset):
    """Test coordinate slicing returns CoordinateView."""
    with open_dataset(populated_dataset) as ds:
        lat = ds['latitude']
        view = lat[slice(5, 15)]
        assert view.shape == (10,)
        assert np.allclose(view.data, ind_lat_data[5:15])


def test_coordinate_append():
    """Test appending data to a coordinate."""
    fp = script_path.joinpath('test_append.blt')
    first = np.linspace(0, 4.9, 50, dtype='float32')
    second = np.linspace(5.0, 9.9, 50, dtype='float32')
    combined = np.concatenate([first, second])

    with open_dataset(fp, flag='n') as ds:
        coord = ds.create.coord.lat(data=first, chunk_shape=(20,))
        coord.append(second)
        assert coord.shape == (100,)
        assert np.allclose(coord.data, combined)

    fp.unlink()


def test_coordinate_prepend():
    """Test prepending data to a coordinate."""
    fp = script_path.joinpath('test_prepend.blt')
    main_data = np.linspace(0, 4.9, 50, dtype='float32')
    prepend_data = np.linspace(-5.0, -0.1, 50, dtype='float32')
    combined = np.concatenate([prepend_data, main_data])

    with open_dataset(fp, flag='n') as ds:
        coord = ds.create.coord.lat(data=main_data, chunk_shape=(20,))
        coord.prepend(prepend_data)
        assert coord.shape == (100,)
        assert np.allclose(coord.data, combined)

    fp.unlink()


##############################
### Data variable operations


def test_data_var_repr(populated_dataset):
    """Test data variable repr."""
    with open_dataset(populated_dataset) as ds:
        dv = ds['temperature']
        rep = repr(dv)
        assert 'temperature' in rep


def test_data_var_properties(populated_dataset):
    """Test data variable shape, chunk_shape, coord_names, ndims."""
    with open_dataset(populated_dataset) as ds:
        dv = ds['temperature']
        assert dv.shape == (100, 100, 10)
        assert dv.chunk_shape == (20, 30, 10)
        assert dv.coord_names == ('latitude', 'longitude', 'time')
        assert dv.ndims == 3


def test_data_var_data_integrity(populated_dataset):
    """Test that data variable data is correctly stored and retrieved."""
    with open_dataset(populated_dataset) as ds:
        dv = ds['temperature']
        assert np.allclose(dv.data, ind_data_var_data)


def test_data_var_iter_chunks_no_data(populated_dataset):
    """Test iterating chunks without data (just slices)."""
    with open_dataset(populated_dataset) as ds:
        dv = ds['temperature']
        chunks = list(dv.iter_chunks(include_data=False))
        assert len(chunks) > 0
        # Each chunk should be a tuple of slices
        for chunk in chunks:
            assert isinstance(chunk, tuple)
            assert len(chunk) == 3  # 3 dimensions
            for s in chunk:
                assert isinstance(s, slice)


def test_data_var_iter_chunks_with_data(populated_dataset):
    """Test iterating chunks with data and verifying correctness."""
    with open_dataset(populated_dataset) as ds:
        dv = ds['temperature']
        result = np.full(dv.shape, np.nan, dtype=ind_data_var_data.dtype)
        for chunk_slices, data in dv.iter_chunks(include_data=True):
            result[chunk_slices] = data

        assert np.allclose(result, ind_data_var_data)


def test_data_var_set_method():
    """Test DataVariable.set() directly (not via __setitem__)."""
    fp = script_path.joinpath('test_set.blt')
    small_data = np.array([1.0, 2.0, 3.0, 4.0], dtype='float32')
    lat = np.linspace(0, 1, 2, dtype='float32')
    lon = np.linspace(0, 1, 2, dtype='float32')

    with open_dataset(fp, flag='n') as ds:
        ds.create.coord.lat(data=lat, chunk_shape=(2,))
        ds.create.coord.lon(data=lon, chunk_shape=(2,))
        dv = ds.create.data_var.generic('temp', ('latitude', 'longitude'), dtypes.dtype('float32'), chunk_shape=(2, 2))
        dv.set((slice(None), slice(None)), small_data.reshape(2, 2))
        assert np.allclose(dv.data, small_data.reshape(2, 2))

    fp.unlink()


def test_data_var_get_chunk(populated_dataset):
    """Test get_chunk method."""
    with open_dataset(populated_dataset) as ds:
        dv = ds['temperature']
        chunk = dv.get_chunk()
        assert chunk is not None
        assert isinstance(chunk, np.ndarray)

        # missing_none for a chunk that exists
        chunk2 = dv.get_chunk(missing_none=True)
        assert chunk2 is not None


def test_data_var_view(populated_dataset):
    """Test data variable view via selection."""
    with open_dataset(populated_dataset) as ds:
        dv = ds['temperature']
        view = dv[slice(0, 5), slice(0, 10), slice(0, 3)]
        assert view.shape == (5, 10, 3)
        assert np.allclose(view.data, ind_data_var_data[0:5, 0:10, 0:3])


def test_data_var_len(populated_dataset):
    """Test __len__ returns product of shape."""
    with open_dataset(populated_dataset) as ds:
        dv = ds['temperature']
        assert len(dv) == 100 * 100 * 10


##############################
### Dataset selection


def test_select_returns_view(populated_dataset):
    """Test that select returns a DatasetView with correct structure."""
    with open_dataset(populated_dataset) as ds:
        view = ds.select({'latitude': slice(0, 10)})
        assert 'latitude' in view.var_names
        assert 'temperature' in view.var_names
        assert len(view.coord_names) == 3
        assert len(view.data_var_names) == 1
        assert view.is_open is True


def test_select_loc_returns_view(populated_dataset):
    """Test that select_loc returns a DatasetView with correct data."""
    with open_dataset(populated_dataset) as ds:
        view = ds.select_loc({'latitude': slice(0.4, 0.7)})
        dv = view['temperature']
        lat_view = view['latitude']
        assert lat_view.shape[0] < 100  # Should be a subset


def test_select_invalid_coord(populated_dataset):
    """Test that selecting on a nonexistent coordinate raises KeyError."""
    with open_dataset(populated_dataset) as ds:
        with pytest.raises(KeyError):
            ds.select({'nonexistent': slice(0, 5)})

        with pytest.raises(KeyError):
            ds.select_loc({'nonexistent': slice(0, 5)})


##############################
### Variable deletion


def test_delete_variable():
    """Test deleting a data variable and then a coordinate."""
    fp = script_path.joinpath('test_del.blt')
    lat = np.linspace(0, 4.9, 50, dtype='float32')
    lon = np.linspace(0, 4.9, 50, dtype='float32')

    with open_dataset(fp, flag='n') as ds:
        ds.create.coord.lat(data=lat, chunk_shape=(20,))
        ds.create.coord.lon(data=lon, chunk_shape=(20,))
        ds.create.data_var.generic('temp', ('latitude', 'longitude'), dtypes.dtype('float32'), chunk_shape=(20, 20))

        assert 'temp' in ds
        del ds['temp']
        assert 'temp' not in ds

        # Can now delete the coordinates since no data vars reference them
        del ds['latitude']
        assert 'latitude' not in ds

    fp.unlink()


def test_delete_coord_with_data_var_raises():
    """Test that deleting a coordinate while data vars reference it raises ValueError."""
    fp = script_path.joinpath('test_del2.blt')
    lat = np.linspace(0, 4.9, 50, dtype='float32')
    lon = np.linspace(0, 4.9, 50, dtype='float32')

    with open_dataset(fp, flag='n') as ds:
        ds.create.coord.lat(data=lat, chunk_shape=(20,))
        ds.create.coord.lon(data=lon, chunk_shape=(20,))
        ds.create.data_var.generic('temp', ('latitude', 'longitude'), dtypes.dtype('float32'), chunk_shape=(20, 20))

        with pytest.raises(ValueError, match='coordinate'):
            del ds['latitude']

    fp.unlink()


def test_delete_nonexistent_raises(populated_dataset):
    """Test that deleting a nonexistent variable raises KeyError."""
    with open_dataset(populated_dataset) as ds:
        with pytest.raises(KeyError):
            del ds['nonexistent']


##############################
### Compression options


def test_lz4_compression():
    """Test creating a dataset with lz4 compression."""
    fp = script_path.joinpath('test_lz4.blt')
    lat = np.linspace(0, 4.9, 50, dtype='float32')

    with open_dataset(fp, flag='n', compression='lz4') as ds:
        assert ds.compression == 'lz4'
        ds.create.coord.lat(data=lat, chunk_shape=(20,))
        assert np.allclose(ds['latitude'].data, lat)

    # Re-open and verify
    with open_dataset(fp) as ds:
        assert ds.compression == 'lz4'
        assert np.allclose(ds['latitude'].data, lat)

    fp.unlink()


def test_zstd_compression():
    """Test creating a dataset with zstd compression (default)."""
    fp = script_path.joinpath('test_zstd.blt')
    lat = np.linspace(0, 4.9, 50, dtype='float32')

    with open_dataset(fp, flag='n', compression='zstd') as ds:
        assert ds.compression == 'zstd'
        ds.create.coord.lat(data=lat, chunk_shape=(20,))

    with open_dataset(fp) as ds:
        assert ds.compression == 'zstd'
        assert np.allclose(ds['latitude'].data, lat)

    fp.unlink()


def test_invalid_compression():
    """Test that an invalid compression raises ValueError."""
    fp = script_path.joinpath('test_bad_comp.blt')
    with pytest.raises(ValueError):
        open_dataset(fp, flag='n', compression='invalid')
    if fp.exists():
        fp.unlink()


##############################
### Read-only safety


def test_read_only_no_write():
    """Test that writing to a read-only dataset raises ValueError."""
    fp = script_path.joinpath('test_ro.blt')
    lat = np.linspace(0, 4.9, 50, dtype='float32')
    lon = np.linspace(0, 4.9, 50, dtype='float32')

    with open_dataset(fp, flag='n') as ds:
        ds.create.coord.lat(data=lat, chunk_shape=(20,))
        ds.create.coord.lon(data=lon, chunk_shape=(20,))
        dv = ds.create.data_var.generic('temp', ('latitude', 'longitude'), dtypes.dtype('float32'), chunk_shape=(20, 20))
        dv[:] = np.zeros((50, 50), dtype='float32')

    with open_dataset(fp) as ds:
        dv = ds['temperature'] if 'temperature' in ds else ds['temp']
        with pytest.raises(ValueError):
            dv[:] = np.ones((50, 50), dtype='float32')
        coord = ds['latitude']
        with pytest.raises(ValueError):
            coord.append(np.array([5.0], dtype='float32'))
        with pytest.raises(ValueError):
            coord.prepend(np.array([-1.0], dtype='float32'))

    fp.unlink()


##############################
### Error handling


def test_get_nonexistent_variable(populated_dataset):
    """Test that getting a nonexistent variable raises ValueError."""
    with open_dataset(populated_dataset) as ds:
        with pytest.raises(ValueError):
            ds['nonexistent_var']


def test_get_with_non_string_key(populated_dataset):
    """Test that getting with non-string key raises TypeError."""
    with open_dataset(populated_dataset) as ds:
        with pytest.raises(TypeError):
            ds.get(123)


def test_duplicate_variable_name():
    """Test that creating a variable with an existing name raises ValueError."""
    fp = script_path.joinpath('test_dup.blt')
    with open_dataset(fp, flag='n') as ds:
        ds.create.coord.lat(data=ind_lat_data[:10], chunk_shape=(10,))
        with pytest.raises(ValueError, match='already contains'):
            ds.create.coord.lat(data=ind_lat_data[:10], chunk_shape=(10,))
    fp.unlink()


def test_duplicate_axis():
    """Test that creating a coordinate with a duplicate axis raises ValueError."""
    fp = script_path.joinpath('test_dup_axis.blt')
    with open_dataset(fp, flag='n') as ds:
        ds.create.coord.lat(data=ind_lat_data[:10], chunk_shape=(10,))
        with pytest.raises(ValueError, match='axis'):
            ds.create.coord.generic('other_lat', data=ind_lat_data[:10], chunk_shape=(10,), axis='y')
    fp.unlink()


##############################
### Multiple data variables


def test_multiple_data_vars():
    """Test creating and reading multiple data variables sharing coordinates."""
    fp = script_path.joinpath('test_multi_dv.blt')
    lat = np.linspace(0, 4.9, 50, dtype='float32')
    t = np.linspace(0, 4, 5, dtype='datetime64[D]')
    temp_data = np.random.randn(50, 5).astype('float32')
    wind_data = np.random.randn(50, 5).astype('float32')

    with open_dataset(fp, flag='n') as ds:
        ds.create.coord.lat(data=lat, chunk_shape=(20,))
        ds.create.coord.time(data=t, dtype=t.dtype)
        dv1 = ds.create.data_var.generic('temp', ('latitude', 'time'), dtypes.dtype('float32'), chunk_shape=(20, 5))
        dv1[:] = temp_data
        dv2 = ds.create.data_var.generic('wind', ('latitude', 'time'), dtypes.dtype('float32'), chunk_shape=(20, 5))
        dv2[:] = wind_data

    with open_dataset(fp) as ds:
        assert len(ds.data_var_names) == 2
        assert np.allclose(ds['temp'].data, temp_data)
        assert np.allclose(ds['wind'].data, wind_data)

    fp.unlink()


##############################
### Integer coordinate


def test_integer_coordinate():
    """Test creating a coordinate with integer dtype."""
    fp = script_path.joinpath('test_int_coord.blt')
    int_coord_data = np.arange(0, 100, dtype='int32')

    with open_dataset(fp, flag='n') as ds:
        coord = ds.create.coord.generic('station_id', data=int_coord_data, chunk_shape=(20,))
        assert np.all(coord.data == int_coord_data)

    with open_dataset(fp) as ds:
        assert np.all(ds['station_id'].data == int_coord_data)

    fp.unlink()


##############################
### Groupby


def test_groupby(populated_dataset):
    """Test groupby method produces correct data."""
    with open_dataset(populated_dataset) as ds:
        dv = ds['temperature']
        result = np.full(dv.shape, np.nan, dtype=ind_data_var_data.dtype)
        for slices, data in dv.groupby('latitude'):
            result[slices] = data

        assert np.allclose(result, ind_data_var_data)


def test_groupby_invalid_coord(populated_dataset):
    """Test groupby with invalid coordinate name raises ValueError."""
    with open_dataset(populated_dataset) as ds:
        dv = ds['temperature']
        with pytest.raises(ValueError, match='not a coord'):
            list(dv.groupby('nonexistent'))


##############################
### Units


def test_units():
    """Test units property and update_units method."""
    fp = script_path.joinpath('test_units.blt')
    lat = np.linspace(0, 4.9, 50, dtype='float32')

    with open_dataset(fp, flag='n') as ds:
        coord = ds.create.coord.lat(data=lat, chunk_shape=(20,))
        coord.update_units('degrees_north')
        assert coord.units == 'degrees_north'

        coord.update_units(None)
        assert coord.units is None

    fp.unlink()


def test_units_read_only():
    """Test that update_units on read-only dataset raises ValueError."""
    fp = script_path.joinpath('test_units_ro.blt')
    lat = np.linspace(0, 4.9, 50, dtype='float32')

    with open_dataset(fp, flag='n') as ds:
        ds.create.coord.lat(data=lat, chunk_shape=(20,))

    with open_dataset(fp) as ds:
        with pytest.raises(ValueError):
            ds['latitude'].update_units('degrees_north')

    fp.unlink()


##############################
### Reopen and persistence


def test_reopen_dataset():
    """Test that data persists after closing and reopening a dataset."""
    fp = script_path.joinpath('test_reopen.blt')
    lat = np.linspace(0, 4.9, 50, dtype='float32')
    lon = np.linspace(-5, -0.1, 50, dtype='float32')
    data = np.random.randn(50, 50).astype('float32')

    with open_dataset(fp, flag='n') as ds:
        ds.create.coord.lat(data=lat, chunk_shape=(20,))
        ds.create.coord.lon(data=lon, chunk_shape=(20,))
        dv = ds.create.data_var.generic('vals', ('latitude', 'longitude'), dtypes.dtype('float32'), chunk_shape=(20, 20))
        dv[:] = data
        ds.attrs['info'] = 'persist test'

    with open_dataset(fp) as ds:
        assert np.allclose(ds['latitude'].data, lat)
        assert np.allclose(ds['longitude'].data, lon)
        assert np.allclose(ds['vals'].data, data)
        assert ds.attrs['info'] == 'persist test'

    fp.unlink()


##############################
### Open with 'w' flag (modify existing)


def test_open_write_modify():
    """Test opening an existing dataset with 'w' flag and modifying data."""
    fp = script_path.joinpath('test_w_flag.blt')
    lat = np.linspace(0, 4.9, 50, dtype='float32')
    lon = np.linspace(0, 4.9, 50, dtype='float32')
    original = np.zeros((50, 50), dtype='float32')
    modified = np.ones((50, 50), dtype='float32')

    with open_dataset(fp, flag='n') as ds:
        ds.create.coord.lat(data=lat, chunk_shape=(20,))
        ds.create.coord.lon(data=lon, chunk_shape=(20,))
        dv = ds.create.data_var.generic('vals', ('latitude', 'longitude'), dtypes.dtype('float32'), chunk_shape=(20, 20))
        dv[:] = original

    with open_dataset(fp, flag='w') as ds:
        dv = ds['vals']
        dv[:] = modified

    with open_dataset(fp) as ds:
        assert np.allclose(ds['vals'].data, modified)

    fp.unlink()


##############################
### Empty dataset


def test_empty_dataset_repr():
    """Test repr on a dataset with no variables."""
    fp = script_path.joinpath('test_empty.blt')
    with open_dataset(fp, flag='n') as ds:
        rep = repr(ds)
        assert 'Coordinates' in rep
        assert 'Data Variables' in rep

    fp.unlink()


def test_empty_coord():
    """Test creating a coordinate without data."""
    fp = script_path.joinpath('test_empty_coord.blt')
    with open_dataset(fp, flag='n') as ds:
        coord = ds.create.coord.lat(chunk_shape=(20,))
        assert coord.shape == (0,)
        rep = repr(coord)
        assert 'latitude' in rep

    fp.unlink()


##############################
### Copy with filtering


def test_copy_include_data_vars():
    """Test copying a dataset with include_data_vars filter."""
    fp = script_path.joinpath('test_copy_src.blt')
    fp2 = script_path.joinpath('test_copy_dst.blt')
    lat = np.linspace(0, 4.9, 50, dtype='float32')
    t = np.linspace(0, 4, 5, dtype='datetime64[D]')

    with open_dataset(fp, flag='n') as ds:
        ds.create.coord.lat(data=lat, chunk_shape=(20,))
        ds.create.coord.time(data=t, dtype=t.dtype)
        dv1 = ds.create.data_var.generic('temp', ('latitude', 'time'), dtypes.dtype('float32'), chunk_shape=(20, 5))
        dv1[:] = np.ones((50, 5), dtype='float32')
        dv2 = ds.create.data_var.generic('wind', ('latitude', 'time'), dtypes.dtype('float32'), chunk_shape=(20, 5))
        dv2[:] = np.zeros((50, 5), dtype='float32')

    with open_dataset(fp) as ds:
        new_ds = ds.copy(fp2, include_data_vars=['temp'])
        assert 'temp' in new_ds.data_var_names
        assert 'wind' not in new_ds.data_var_names
        assert np.allclose(new_ds['temp'].data, np.ones((50, 5), dtype='float32'))
        new_ds.close()

    fp.unlink()
    fp2.unlink()
























