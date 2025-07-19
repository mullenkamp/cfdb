import copy
import io
import os
import pytest
try:
    import tomllib as toml
except ImportError:
    import tomli as toml
import uuid6 as uuid
import numpy as np
from time import time
import pathlib
from cfdb import open_dataset, open_edataset, cfdb_to_netcdf4, netcdf4_to_cfdb
import ebooklet
import h5netcdf

###################################################
### Parameters

script_path = pathlib.Path(os.path.realpath(os.path.dirname(__file__)))

file_path = script_path.joinpath('test1.blt')
flag = 'n'

new_file_path = script_path.joinpath('test2.blt')
nc_file_path = script_path.joinpath('test1.nc')

name = 'air_temp'
coords = ('latitude', 'time')
dtype_encoded = 'int32'
chunk_shape = (20, 30)
fillvalue = None
scale_factor = 0.1
add_offset = None
sel = (slice(1, 4), slice(2, 5))
sel2 = (slice(1, 1), slice(2, 5))
loc_sel = (slice(0.4, 0.7), slice('1970-01-06', '1970-01-15'))
ds_sel = {'latitude': slice(1, 4), 'time': slice(2, 5)}
ds_loc_sel = {'latitude': slice(0.4, 0.7), 'time': slice('1970-01-06', '1970-01-15')}

lat_data = np.linspace(0, 19.9, 200, dtype='float32')
other_lat_data = np.linspace(-10, -1, 10, dtype='float32')
time_data = np.linspace(0, 199, 200, dtype='datetime64[D]')

data_var_data = np.linspace(0, 3999.9, 40000, dtype='float32').reshape(200, 200)

new_chunk_shape = (41, 41)

## EDataset
try:
    with io.open(script_path.joinpath('s3_config.toml'), "rb") as f:
        conn_config = toml.load(f)['connection_config']

    endpoint_url = conn_config['endpoint_url']
    access_key_id = conn_config['access_key_id']
    access_key = conn_config['access_key']
except:
    endpoint_url = os.environ['endpoint_url']
    access_key_id = os.environ['access_key_id']
    access_key = os.environ['access_key']

bucket = 'achelous'
db_key = uuid.uuid8().hex[-13:]
# db_key = '7b120a3b4ec5d'
base_url = 'https://b2.tethys-ts.xyz/file/' + bucket + '/'
db_url = base_url +  db_key

remote_conn = ebooklet.S3Connection(access_key_id, access_key, db_key, bucket, endpoint_url=endpoint_url, db_url=db_url)



##############################
### Functions


# @pytest.fixture
# def get_logs(request):
#     yield

#     if request.node.rep_call.failed:
#         # Add code here to cleanup failure scenario
#         print("executing test failed")

#         with remote_conn.open('w') as s3open:
#             s3open.delete_remote()

#         file_path.unlink()
#         remote_index_path = file_path.parent.joinpath(file_path.name + '.remote_index')
#         remote_index_path.unlink()
#         new_file_path.unlink()
#         nc_file_path.unlink()


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    """Cleanup a test data."""
    def remove_test_data():
        with remote_conn.open('w') as s3open:
            s3open.delete_remote()

        file_path.unlink()
        remote_index_path = file_path.parent.joinpath(file_path.name + '.remote_index')
        remote_index_path.unlink()
        new_file_path.unlink()
        nc_file_path.unlink()
        # changelog_path = file_path.parent.joinpath(file_path.name + '.changelog')
        # if changelog_path.exists:
        #     changelog_path.unlink()

    request.addfinalizer(remove_test_data)


##############################
### Tests


def test_coord_creation():
    with open_dataset(file_path, flag='n') as ds:
        lat_coord = ds.create.coord.latitude(data=lat_data, chunk_shape=(20,))
        lat_coord.prepend(other_lat_data)
        del ds['latitude']
        lat_coord = ds.create.coord.latitude(chunk_shape=(20,))
        lat_coord.append(lat_data)
        print(lat_coord)
        # print(lat_coord.data[6:12])

        assert np.allclose(lat_coord.data, lat_data)

        time_coord = ds.create.coord.time(data=time_data, dtype_decoded=time_data.dtype, dtype_encoded='int32')
        print(time_coord)

        assert np.all(time_coord.data == time_data)

        ds.attrs['history'] = 'Created some coords yo'


def test_data_var_creation():
    with open_dataset(file_path, flag='w') as ds:
        data_var = ds.create.data_var.generic(name, coords, data_var_data.dtype, dtype_encoded, scale_factor=scale_factor, chunk_shape=(20, 20))
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
        assert np.allclose(view2.data, data_var_data[(slice(4, 7), slice(5, 14))])

        view3 = ds.select(ds_sel)
        assert np.allclose(view3[name].data, data_var_data[sel])

        view4 = ds.select_loc(ds_loc_sel)
        assert np.allclose(view4[name].data, data_var_data[(slice(4, 7), slice(5, 14))])

        try:
            view_fail = data_var[sel2]
        except ValueError:
            pass

def test_rechunker_assignment():
    with open_dataset(file_path, flag='w') as ds:
        data_var = ds[name]

        data_var2 = ds.create.data_var.generic(name + '2', coords, data_var_data.dtype, dtype_encoded, scale_factor=scale_factor, chunk_shape=new_chunk_shape)

        rechunker = data_var.rechunker()
        rechunk = rechunker.rechunk(new_chunk_shape, decoded=False)

        for write_chunk, data in rechunk:
            data_var2.set(write_chunk, data, encode=False)

        assert np.allclose(data_var2.data, data_var_data)

        del ds[name + '2']

        rechunker = data_var.rechunker()
        rechunk = rechunker.rechunk(new_chunk_shape, decoded=True)

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


def test_edataset():
    with open_edataset(remote_conn, file_path, flag='w') as ds:
        print(ds)
        changes = ds.changes()

        assert changes.push()

    file_path.unlink()

    with open_edataset(remote_conn, file_path) as ds:
        # print(ds)
        data_var = ds[name]

        view1 = data_var[sel]
        assert np.allclose(view1.data, data_var_data[sel])

        view2 = data_var.loc[loc_sel]
        assert np.allclose(view2.data, data_var_data[(slice(4, 7), slice(5, 14))])

    # with remote_conn.open('w') as s3open:
    #     s3open.delete_remote()

    # file_path.unlink()
    # remote_index_path = file_path.parent.joinpath(file_path.name + '.remote_index')
    # remote_index_path.unlink()


def test_cfdb_to_netcdf4():
    cfdb_to_netcdf4(file_path, nc_file_path, sel_loc=ds_loc_sel)

def test_netcdf4_to_cfdb():
    netcdf4_to_cfdb(nc_file_path, new_file_path)


# open_conn = remote_conn.open('w')
# open_conn.break_other_locks()
# open_conn.delete_remote()
# open_conn.close()


# f = ebooklet.open(remote_conn, file_path, 'n')

# f.set_metadata({'test': 'meta'})
# changes = f.changes()
# list(changes.iter_changes())
























