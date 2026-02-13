import io
import os
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

###################################################
### Parameters

script_path = pathlib.Path(os.path.realpath(os.path.dirname(__file__)))

file_path = script_path.joinpath('test_remote.blt')
name = 'air_temp'
coords = ('latitude', 'longitude', 'time')
chunk_shape = (20, 30, 10)

sel = (slice(1, 4), slice(None, None), slice(2, 5))
loc_sel = (slice(0.4, 0.7), slice(None, None), slice('1970-01-04', '1970-01-10'))

lat_data = np.linspace(0, 9.9, 100, dtype='float32')
lon_data = np.linspace(-5, 4.9, 100, dtype='float32')
time_data = np.linspace(0, 10, 10, dtype='datetime64[D]')

data_var_data = np.linspace(0, 9999.9, 100000, dtype='float32').reshape(100, 100, 10)

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
db_url = base_url +  db_key

remote_conn = ebooklet.S3Connection(access_key_id, access_key, db_key, bucket, endpoint_url=endpoint_url, db_url=db_url)


@pytest.fixture(scope="module", autouse=True)
def cleanup(request):
    """Cleanup test data on S3 and locally."""
    def remove_test_data():
        try:
            with remote_conn.open('w') as s3open:
                s3open.delete_remote()
        except Exception:
            pass

        if file_path.exists():
            file_path.unlink()
        
        remote_index_path = file_path.parent.joinpath(file_path.name + '.remote_index')
        if remote_index_path.exists():
            remote_index_path.unlink()

    request.addfinalizer(remove_test_data)


def test_edataset():
    # First create a local dataset with data to push
    with open_dataset(file_path, flag='n') as ds:
        ds.create.coord.lat(data=lat_data, chunk_shape=(20,))
        ds.create.coord.lon(data=lon_data, chunk_shape=(20,))
        ds.create.coord.time(data=time_data, dtype=time_data.dtype)
        
        data_dtype = dtypes.dtype(data_var_data.dtype, 1, 0, 10000)
        data_var = ds.create.data_var.generic(name, coords, data_dtype, chunk_shape=chunk_shape)
        data_var[:] = data_var_data

    # Now open as EDataset and push
    with open_edataset(remote_conn, file_path, flag='w') as ds:
        changes = ds.changes()
        assert changes.push()

    # Remove local file to force pull from remote
    file_path.unlink()

    # Re-open as EDataset and verify data
    with open_edataset(remote_conn, file_path) as ds:
        data_var = ds[name]

        # Verify basic selection
        view1 = data_var[sel]
        assert np.allclose(view1.data, data_var_data[sel])

        # Verify location-based selection
        # Note: loc_sel indices in test_dataset.py were (slice(4, 7), slice(None, None), slice(3, 9))
        # Based on lat_data (0 to 9.9, 100 steps -> 0.1 step), 0.4 to 0.7 is indeed index 4 to 7
        # Based on time_data (0 to 10 days), '1970-01-04' is day 3, '1970-01-10' is day 9
        view2 = data_var.loc[loc_sel]
        assert np.allclose(view2.data, data_var_data[(slice(4, 7), slice(None, None), slice(3, 9))])
