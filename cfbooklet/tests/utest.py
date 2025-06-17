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

###################################################
### Parameters

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

cache_path = pathlib.Path('/home/mike/data/cache/cfbooklet')

file_path = cache_path.joinpath('test1.blt')
flag = 'n'
compression='zstd'
kwargs = {}

name = 'coord1'

data = np.array([2, 3, 4, 5, 6, 7], dtype='uint32')
data = np.array([2, 3, 4, 5, 6, 7], dtype='int32')
data = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype='float32')
data = np.array([2, 3, 4, 5, 6, 7], dtype='str')
data = np.array([2, 3, 4, 5, 6, 7], dtype='datetime64[D]')

shape = None
chunk_shape = (4,)
fillvalue = None
dtype_decoded = None
dtype_encoded = None
scale_factor = None
add_offset = None

data = None
shape = (6, )
chunk_shape = (4,)
fillvalue = None
dtype_decoded = 'int16'

###################################################
### Functions

d

###################################################
### Start

self = Dataset(file_path, flag=flag)
lat1 = self.create.coord.latitude(shape=shape)
lat1 = self.create.coord.latitude(data=data, chunk_shape=chunk_shape, step=True)
time1 = self.create.coord.time(shape=shape)
altitude1 = self.create.coord.altitude(shape=shape)
self.close()



###################################################
### Chunker testing

out_chunks = rechunking_plan(shape, shape, source_chunk_shape, target_chunk_shape, itemsize, max_mem, True, False)

# out_chunks = rechunking_plan(shape, new_shape, source_chunk_shape, target_chunk_shape, itemsize, max_mem, True, True)

source = np.arange(1, prod(source_shape) + 1).reshape(source_shape).astype(dtype)
target = np.zeros(source_shape, dtype=dtype)

# source_read_chunk_shape = out_chunks[0]
# inter_chunks = out_chunks[1]
# target_read_chunks = out_chunks[2]

source_read_chunk_shape = calc_source_read_chunk_shape(source_chunk_shape, target_chunk_shape, itemsize, max_mem)

read_ratio = calc_read_ratio_source(source_read_chunk_shape, source_chunk_shape, target_chunk_shape)

n_reads_source = calc_n_reads_source(source_chunk_shape, target_chunk_shape)


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


















































