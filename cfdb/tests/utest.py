import copy
from typing import List, Optional, Sequence, Tuple, Iterator
import numpy as np
import itertools
from time import time
from math import ceil, floor, prod, gcd, lcm
from collections import Counter, deque
from itertools import count
import msgspec
import pathlib
import io
import os
import math
import ebooklet
try:
    import tomllib as toml
except ImportError:
    import tomli as toml

from cfdb import open_dataset, open_edataset, cfdb_to_netcdf4, netcdf4_to_cfdb
import h5netcdf
import rechunkit

###################################################
### Parameters

script_path = pathlib.Path(os.path.realpath(os.path.dirname(__file__)))

# source_shape = (30, 30)
source_shape = (30, 30, 30)

new_shape = (35, 31)

source_chunk_shape = (5, 2)
target_chunk_shape = (2, 5)
source_chunk_shape = (5, 2, 4)
target_chunk_shape = (2, 5, 3)
itemsize = 4
max_mem = 40 * itemsize
max_mem = 160 * itemsize

dtype = 'int32'

cache_path = pathlib.Path('/home/mike/data/cache/cfdb')

file_path = cache_path.joinpath('test1.blt')
flag = 'n'
compression='zstd'
kwargs = {}

name = 'coord1'
name = 'air_temp'

data = np.array([2, 3, 4, 5, 6, 7], dtype='uint32')
data = np.array([2, 3, 4, 5, 6, 7], dtype='int32')
new_data = np.arange(8, 12, dtype='int32')
old_data = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype='float32')
new_data = np.linspace(0.8, 1.2, 5, dtype='float32')
data = new_data
data = np.array([2, 3, 4, 5, 6, 7], dtype='str')
data_time = np.array([2, 3, 4, 5, 6, 7], dtype='datetime64[D]')

shape = None
chunk_shape = (4,)
fillvalue = None
dtype_decoded = None
dtype_encoded = None
scale_factor = None
add_offset = None

# data = None
shape = (6, )
chunk_shape = (40,)
fillvalue = None
dtype_decoded = 'int16'


cache_path = pathlib.Path('/home/mike/data/cache/cfdb')
file_path = cache_path.joinpath('test1.blt')
new_file_path = cache_path.joinpath('test2.blt')
nc_file_path = cache_path.joinpath('test1.nc')
flag = 'n'

name = 'air_temp'
coords = ('latitude', 'time')
dtype_decoded = 'float32'
dtype_encoded = 'int32'
chunk_shape = (20, 30)
fillvalue = None
scale_factor = 0.1
add_offset = None
sel = (slice(1, 4), slice(2, 5))
loc_sel = (slice(0.4, 0.7), slice('1970-01-06', '1970-01-15'))
ds_sel = {'latitude': slice(4, 14)}
ds_loc_sel = {'latitude': slice(0.4, 0.7)}

lat_data = np.linspace(0, 19.9, 200, dtype='float32')
time_data = np.linspace(0, 199, 200, dtype='datetime64[D]')

air_data = np.linspace(0, 3999.9, 40000, dtype='float32').reshape(200, 200)

era5_path = '/home/mike/data/ecmwf/reanalysis-era5-land/reanalysis-era5-land.total_precipitation.1950-01-01!1957-12-31.nc'

###################################################
### Functions


def find_nearest_idx(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx



d

###################################################
### Start

self = Dataset(file_path, flag=flag)
lat1 = self.create.coord.latitude()
lat1 = self.create.coord.latitude(data=data, chunk_shape=chunk_shape, step=True)
time1 = self.create.coord.time(shape=shape)
altitude1 = self.create.coord.altitude(shape=shape)
self1.close()



self1 = Dataset(file_path, flag=flag)
self = self1.create.coord.latitude(chunk_shape=(8,))
self = self1.create.coord.latitude(data=old_data, chunk_shape=(8,))
# self.data = old_data
self.append(old_data)
self.append(new_data)
self.prepend(old_data)

self = self1.create.coord.longitude(data=data)
self = self1.create.coord.time(data=data_time, dtype_decoded=data_time.dtype, dtype_encoded='int32')


self1 = open_dataset(file_path, flag=flag)
lat_coord = self1.create.coord.latitude(data=lat_data, chunk_shape=(20,))
time_coord = self1.create.coord.time(data=time_data, dtype_decoded=time_data.dtype, dtype_encoded='int32')

# sys_meta = self._sys_meta

self = self1.create.data_var.generic(name, coords, dtype_decoded, dtype_encoded, scale_factor=scale_factor, chunk_shape=(20, 20))

self[:] = air_data

lat_coord.append(new_data)
lat_coord.prepend(old_data)

view1 = self.loc[loc_sel]
view1.data

self.attrs['test_attr'] = ['test']

view2 = self1.select(ds_sel)


self1.close()


self1 = open_dataset(file_path)
self = self1[name]


new_ds = view2.copy(new_file_path)


for index, data in self.items():
    print(index)





##############################
### Tests


with open_dataset(file_path) as f:
    self = f[name]
    view1 = self.loc[loc_sel]
    print(view1.data)

f = open_dataset(file_path, flag=flag)
lat_coord = f.create.coord.latitude(data=lat_data, chunk_shape=(20,))

del f['latitude']

lat_coord = f.create.coord.latitude(chunk_shape=(20,))
lat_coord.append(lat_data)

time_coord = f.create.coord.time(data=time_data, dtype_decoded=time_data.dtype, dtype_encoded='int32')

air_temp_var = f.create.data_var.generic(name, coords, dtype_decoded, dtype_encoded, scale_factor=scale_factor, chunk_shape=(20, 20))

air_temp_var[:] = air_data

view1 = air_temp_var[sel]

assert np.allclose(view1.data, air_data[sel])

view2 = air_temp_var.loc[loc_sel]

assert np.allclose(view2.data, air_temp_var.data[(slice(4, 7), slice(5, 14))])
view2.data

del f[name]

view3 = f.sel(ds_sel)

new_chunk_shape = (40, 40)
air_temp2_var = f.create.data_var.generic(name + '2', coords, dtype_decoded, dtype_encoded, scale_factor=scale_factor, chunk_shape=new_chunk_shape)

rechunker = air_temp_var.rechunker()
rechunk = rechunker.rechunk(new_chunk_shape, decoded=False)

for write_chunk, data in rechunk:
    air_temp2_var.set(write_chunk, data, encode=False)

rechunk = rechunker.rechunk(new_chunk_shape)

for write_chunk, data in rechunk:
    air_temp2_var[write_chunk] = data


grp = air_temp_var.groupby(['latitude'])
for write_chunk, data in grp:
    print(write_chunk)
    print(data)


f.to_netcdf4(nc_file_path)

str_var = f.create.coord.generic('str_coord', dtype_decoded='U10', chunk_shape=(20,))

str_var.append(['b', 'c', 'yyyyyyy', 'hhh'])
str_var.prepend(['a'])


for chunk, data in air_temp_var:
    print(chunk)


f.close()




###################################################
### Chunker testing

source_shape = (31, 31)
shape = source_shape
# new_shape = (35, 31)

sel = (slice(3, 21), slice(11, 25))
sel = (slice(5, 20), slice(10, 24))

source_chunk_shape = (5, 2)
target_chunk_shape = (2, 5)
# source_chunk_shape = (5, 2, 4)
# target_chunk_shape = (2, 5, 3)
itemsize = 4
max_mem = 40 * itemsize
max_mem = 160 * itemsize

dtype = np.dtype('int32')

source = np.arange(1, prod(source_shape) + 1, dtype=dtype).reshape(source_shape)
source = source.__getitem__

# chunk_read_offset = tuple(s.start for s in sel)

# chunk_start = tuple(cs * (ss.start//cs) for cs, ss in zip(source_chunk_shape, sel))
# # chunk_start = tuple(ss.start for ss in self._sel)
# chunk_end = tuple(cs * ((ss.stop - 1)//cs + 1) for cs, ss in zip(source_chunk_shape, sel))

# shape = tuple(cs * ((ss.stop - ss.start - 1)//cs + 1) for cs, ss in zip(target_chunk_shape, sel))

# shape = tuple(ss.stop - ss.start for ss in sel)



# source_read_chunk_shape = out_chunks[0]
# inter_chunks = out_chunks[1]
# target_read_chunks = out_chunks[2]

source_read_chunk_shape = calc_source_read_chunk_shape(source_chunk_shape, target_chunk_shape, itemsize, max_mem)


n_reads_simple = calc_n_reads_simple(source_shape, source_chunk_shape, target_chunk_shape)

n_reads, n_writes = calc_n_reads_rechunker(source_shape, dtype, source_chunk_shape, target_chunk_shape, max_mem, sel)


target = np.zeros(source_shape, dtype=dtype)[sel]
for write_chunk, data in rechunker(source, shape, dtype, source_chunk_shape, target_chunk_shape, max_mem, sel):
    # print(write_chunk)
    target[write_chunk] = data

np.all(source(sel) == target)


target = np.zeros(source_shape, dtype=dtype)
for write_chunk, data in rechunker(source, shape, dtype, source_chunk_shape, target_chunk_shape, max_mem):
    # print(write_chunk)
    target[write_chunk] = data

np.all(source(()) == target)

# target = rechunker_example(source, source_chunk_shape, target_chunk_shape, max_mem)

### Testing
## Same shape to target
# inter0, inter_count0, tot_count0 = simple_rechunker(source, source_chunk_shape, inter_chunks)

# if not (inter0 == source).all():
#     raise ValueError()

# target1, target_count1, tot_count1 = simple_rechunker(inter0, inter_chunks, target_chunk_shape, target_chunk_shape)

# if not (target1 == source).all():
# ,, ,      raise ValueError()

# print(round(tot_count0/inter_count0, 2))
# print(round(tot_count1/target_count1, 2))
# print(round((tot_count1 + tot_count0)/(inter_count0 + target_count1), 2))

target2, target_count2, tot_count2 = simple_rechunker(source, source_chunk_shape, target_chunk_shape)

if not (target2 == source).all():
    raise ValueError()

print(round(tot_count2/target_count2, 2))
print(round(tot_count2/(tot_count1 + tot_count0), 2))


len(list(chunk_keys(shape, source_chunk_shape)))

s1 = set(chunk_keys(shape, source_chunk_shape))

## different shape to target
inter0, inter_count0, tot_count0 = rechunker(source, source_chunk_shape, source_read_chunk_shape, inter_chunks, new_shape)

new_shape_slice = tuple(slice(0, s) for s in new_shape)
if not (inter0 == source[new_shape_slice]).all():
    raise ValueError()

target1, target_count1, tot_count1 = rechunker(inter0, inter_chunks, target_chunk_shape, target_chunk_shape)

if not (target1 == source[new_shape_slice]).all():
    raise ValueError()

print(round(tot_count0/inter_count0, 2))
print(round(tot_count1/target_count1, 2))
print(round((tot_count1 + tot_count0)/(inter_count0 + target_count1), 2))



### Multi shape testing
results_dict = {}
for i in range(0, 11, 2):
    if i == 0:
        test_shape = shape
    else:
        test_shape = tuple(s * i for s in shape)

    print(test_shape)

    source1 = np.arange(1, np.prod(test_shape) + 1).reshape(test_shape)

    start1 = time()
    target2, target_count2, tot_count2 = rechunker(source1, source_chunk_shape, source_chunk_shape, target_chunk_shape)
    end1 = time()

    tot_time = end1 - start1
    n_points = np.prod(test_shape)

    results_dict[test_shape] = dict(n_points=n_points,
                                    n_chunks_target=target_count2,
                                    n_iter_source=tot_count2,
                                    source_target_ratio=round(tot_count2/target_count2, 2),
                                    time=round(tot_time, 4),
                                    n_pts_per_sec=round(n_points/tot_time)
                                    )







target2, target_count2, tot_count2 = rechunker(source, source_chunk_shape, source_chunk_shape, target_chunk_shape)


dtype = 'int32'
target_mem_arr1 = np.zeros(target_chunk_shape, dtype=dtype)
target = np.zeros(shape, dtype=dtype)

source_read_chunk_shape = calc_source_read_chunk_shape(source_chunk_shape, target_chunk_shape, itemsize, max_mem)

target_chunks = set(chunk_keys3(shape, target_chunk_shape))
target_chunks_iter = chunk_keys3(shape, target_chunk_shape)

for target_start_chunk in target_chunks_iter:
    if target_start_chunk in target_chunks:
        target_stop_chunk = tuple(min(ts + tc, sh) for ts, tc, sh in zip(target_start_chunk, target_chunk_shape, shape))
        source_start_chunk = tuple((ts//src) * src for ts, src in zip(target_start_chunk, source_read_chunk_shape))
        source_stop_chunk = tuple(min(ceil(te/src) * src, sh) for te, src, sh in zip(target_stop_chunk, source_read_chunk_shape, shape))


        for source_slice in chunk_keys4(source_stop_chunk, source_read_chunk_shape, source_start_chunk):
            d


### New one
source = np.arange(1, prod(source_shape) + 1).reshape(source_shape).astype(dtype)
# target = np.zeros(source_shape, dtype=dtype)

n_chunks_source = calc_n_chunks(source_shape, source_chunk_shape)
n_chunks_target = calc_n_chunks(source_shape, target_chunk_shape)

source_read_chunk_shape = calc_source_read_chunk_shape(source_chunk_shape, target_chunk_shape, itemsize, max_mem)

ideal_read_chunk_shape = calc_ideal_read_chunk_shape(source_chunk_shape, target_chunk_shape)
ideal_read_chunk_mem = calc_ideal_read_chunk_mem(ideal_read_chunk_shape, itemsize)

n_reads_simple = calc_n_reads_simple(source_shape, source_chunk_shape, target_chunk_shape)

n_reads, n_writes = calc_n_reads_rechunker(source_shape, source_chunk_shape, target_chunk_shape, itemsize, max_mem)

n_reads_ideal, _ = calc_n_reads_rechunker(source_shape, source_chunk_shape, target_chunk_shape, itemsize, ideal_read_chunk_mem)

target2 = rechunker(source, source_chunk_shape, target_chunk_shape, max_mem)

if not (target2 == source).all():
    raise ValueError()

target3 = rechunker(source, source_chunk_shape, target_chunk_shape, ideal_read_chunk_mem)

if not (target3 == source).all():
    raise ValueError()



#################################
### Other

encoding = dict(
    dtype_encoded='int32',
    dtype_decoded='float32',
    fillvalue=0,
    scale_factor=0.1
    )

encoding = msgspec.convert(encoding, Encoding)
variable = msgspec.convert({'shape': (10, 12), 'chunk_shape': (5, 4), 'coords': ('test1',), 'encoding': encoding}, Variable)

sys_meta = SysMeta(cfbooklet_type='Dataset', compression='zstd')
sys_meta = SysMeta(cfbooklet_type='Dataset', compression='zstd', variables={'test1': variable})

sys_meta = msgspec.convert(msgspec.to_builtins(sys_meta), SysMeta)

m1 = sys_meta




##############################
### h5netcdf

import xarray as xr
import h5netcdf

h5_path = pathlib.Path('/home/mike/data/cache/cfdb/h5netcdf.nc')


h5 = h5netcdf.File(h5_path, 'w')

h5.dimensions['lat'] = len(lat_data)

lat_data_enc = np.round(lat_data * 10).astype('int32')

lat_coord = h5.create_variable('lat', ('lat',), lat_data_enc.dtype, compression='gzip')
lat_coord[:] = lat_data_enc
lat_coord.attrs.update({'scale_factor': 0.0000001})

h5.dimensions['time'] = len(time_data)

time_data_enc = time_data.astype('int32')
time_coord = h5.create_variable('time', ('time',), time_data_enc.dtype, compression='gzip')
time_coord[:] = time_data_enc
time_coord.attrs.update({
    'long_name': 'time',
    'units': 'days since 1970-01-01 00:00:00',
    'standard_name': 'time',
    'calendar': 'proleptic_gregorian',
    'axis': 'T',
    })

air_data_enc = np.round(air_data * 10).astype('int32')
temp_var = h5.create_variable('air_temp', ('lat', 'time'), air_data_enc.dtype, compression='gzip')
temp_var[:] = air_data_enc
temp_var.attrs.update({
    'scale_factor': 0.1
    })

h5.close()


# fillvalue_dict = {'int8': -128, 'int16': -32768, 'int32': -2147483648, 'int64': -9223372036854775808, 'float32': np.nan, 'float64': np.nan, 'str': ''}

time_units_dict = {
    'M': 'months',
    'D': 'days',
    'h': 'hours',
    'm': 'minutes',
    's': 'seconds',
    'ms': 'milliseconds',
    'us': 'microseconds',
    'ns': 'nanoseconds',
    }

inv_time_units_dict = {value: key for key, value in time_units_dict.items()}

new_path = pathlib.Path('/home/mike/data/cache/cfdb/era5_test.cfdb')


h5 = h5netcdf.File(era5_path, 'r')

ds = open_dataset(new_path, 'n')

ds.close()


sel = (slice(24, 25), slice(50, 52), slice(50, 52))
h5_sel = (slice(24, 25), slice(78, 80), slice(50, 52))
loc_sel = (slice('1950-01-02T01', '1950-01-02T03'), slice(-42.30, -42.10), slice(171.30, 171.50))
ds_sel = {'longitude': 60, 'latitude': 70, 'time': slice(40, 100)}
ds_sel = {'time': slice(24, 25)}
ds_sel_loc = {'time': slice('1950-01-02T01', '1950-01-02T03')}

ds = open_dataset(new_path)

t2m = ds['t2m']
view1 = t2m[sel]

if np.allclose(view1.data, t2m._encoder.decode(h5_var[sel]), equal_nan=True):
    print('Booo!')

view2 = t2m.loc[loc_sel]

if np.allclose(view2.data, t2m._encoder.decode(h5_var[h5_sel]), equal_nan=True):
    print('Yay!')


ds_view = ds.select(ds_sel)
ds_view.to_netcdf4('/home/mike/data/cache/cfdb/nc_test.nc')

ds_view = ds.select_loc(ds_sel_loc)
ds_view.to_netcdf4('/home/mike/data/cache/cfdb/nc_test.nc')

ds.to_netcdf4('/home/mike/data/cache/cfdb/nc_test.nc')


lat = ds['latitude'].data
lon = ds['longitude'].data


h5_lat = h5['latitude'][:]
h5_lon = h5['longitude'][:]
h5_time = h5['time']


nc_path = era5_path
cfdb_path = new_path

netcdf4_to_cfdb(era5_path, new_path, sel=None, sel_loc=None)
netcdf4_to_cfdb(era5_path, new_path, sel=ds_sel, sel_loc=None)
netcdf4_to_cfdb(era5_path, new_path, sel=None, sel_loc=ds_sel_loc)
cfdb_to_netcdf4(new_path, '/home/mike/data/cache/cfdb/nc_test.nc')

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
# db_key = uuid.uuid8().hex[-13:]
db_key = new_path.name
base_url = 'https://b2.tethys-ts.xyz/file/' + bucket + '/'
db_url = base_url +  db_key

remote_conn = ebooklet.S3Connection(access_key_id, access_key, db_key, bucket, endpoint_url=endpoint_url, db_url=db_url)

ds = open_edataset(remote_conn, new_path, flag='w')
changes = ds.changes()
changes.push()
ds.close()


ds = open_edataset(remote_conn, new_path, flag='r')
view1 = ds.select_loc(ds_sel_loc)
t2m = ds['tp']
view2 = t2m.loc[loc_sel]


print(view1['tp'].data)

ds.load()







































