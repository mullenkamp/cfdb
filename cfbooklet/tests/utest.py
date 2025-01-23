import copy
import numpy as np


###################################################
### Parameters

shape = (410, 410, 410)

new_shape = (35, 31)

source_chunks = (5, 2, 5)
target_chunks = (2, 5, 2)
itemsize = 1
max_mem = 160

###################################################
### Functions


def find_first_source_read(slice1, source_chunks):
    """

    """
    return tuple(slice1[i].start//s * s for i, s in enumerate(source_chunks))


def cut_source_read_chunks(source_read_chunks, first_source_read, shape):
    """

    """
    source_read_chunks2 = []
    for i, s in enumerate(shape):
        diff = s - first_source_read[i]
        read_chunk = source_read_chunks[i]
        if diff < read_chunk:
            source_read_chunks2.append(diff)
        else:
            source_read_chunks2.append(read_chunk)

    return tuple(source_read_chunks2)


def chunk_source_slices(slice1, source_read_chunks, source_chunks, first_source_read, shape):
    """

    """
    source_read_chunks2 = cut_source_read_chunks(source_read_chunks, first_source_read, shape)

    # temp = []
    for source_slice in chunk_keys(source_read_chunks2, source_chunks):
        # print(source_slice)
        source_read_slice = []
        source_mem_slice = []
        inter_mem_slice = []
        for i, s in enumerate(source_slice):
            slice1_start = slice1[i].start
            slice1_stop = slice1[i].stop
            source_read_start = s.start + first_source_read[i]
            source_read_stop = s.stop + first_source_read[i]
            source_read_slice.append(slice(source_read_start, source_read_stop))

            if slice1_start > source_read_start:
                source_mem_start1 = slice1_start
            else:
                source_mem_start1 = source_read_start
            if slice1_stop > source_read_stop:
                source_mem_stop1 = source_read_stop
            else:
                source_mem_stop1 = slice1_stop

            source_mem_start = source_mem_start1 - source_read_start
            source_mem_stop = source_mem_stop1 - source_read_start
            source_mem_slice.append(slice(source_mem_start, source_mem_stop))

            inter_mem_start = source_mem_start1 - slice1_start
            inter_mem_stop = inter_mem_start + source_mem_stop1 - source_mem_start1
            inter_mem_slice.append(slice(inter_mem_start, inter_mem_stop))

        # temp.append((source_read_slice, source_mem_slice, inter_mem_slice))

        yield tuple(source_read_slice), tuple(source_mem_slice), tuple(inter_mem_slice)


def increase_read_chunks(write_slice, source_read, source_read_chunks):
    """

    """
    dim_bool = [s.stop > (source_read[i] + source_read_chunks[i]) for i, s in enumerate(write_slice)]
    while any(dim_bool):
        source_read_chunks = tuple(source_read_chunks[i] + source_chunks[i] if bool1 else source_read_chunks[i] for i, bool1 in enumerate(dim_bool))
        dim_bool = [s.stop > (source_read[i] + source_read_chunks[i]) for i, s in enumerate(write_slice)]

    return source_read_chunks


def rechunker(source, source_chunks, source_read_chunks, target_chunks, new_shape=None, dtype='int32'):
    """

    """
    shape = source.shape
    if isinstance(new_shape, tuple):
        bools1 = [new_shape[i] <= s for i, s in enumerate(shape)]
        if not all(bools1):
            raise ValueError('The new shape must be >= the source shape.')
        shape = new_shape

    target_mem_arr1 = np.zeros(target_chunks, dtype=dtype)
    target = np.zeros(shape, dtype=dtype)

    tot_count = 0
    target_count = 0
    for slice1 in chunk_keys(shape, target_chunks):
        # print(slice1)

        source_read_chunks1 = source_read_chunks
        first_source_read = find_first_source_read(slice1, source_chunks)

        source_read_chunks1 = increase_read_chunks(slice1, first_source_read, source_read_chunks1)

        iter_source_slices = chunk_source_slices(slice1, source_read_chunks1, source_chunks, first_source_read, shape)
        for source_read_slice, source_mem_slice, target_mem_slice in iter_source_slices:
            target_mem_arr1[target_mem_slice] = source[source_read_slice][source_mem_slice]
            tot_count += 1

        slice_shape = tuple(s.stop - s.start for s in slice1)
        if slice_shape == target_chunks:
            target[slice1] = target_mem_arr1
        else:
            new_mem_slice = tuple(slice(0, s) for s in slice_shape)
            target[slice1] = target_mem_arr1[new_mem_slice]

        target_count += 1

    return target, target_count, tot_count


###################################################
### Chunker testing


out_chunks = rechunking_plan(shape, shape, source_chunks, target_chunks, itemsize, max_mem, True, True)

# out_chunks = rechunking_plan(shape, new_shape, source_chunks, target_chunks, itemsize, max_mem, True, True)

source = np.arange(1, np.prod(shape) + 1).reshape(shape)

source_read_chunks = out_chunks[0]
inter_chunks = out_chunks[1]
target_read_chunks = out_chunks[2]





### Testing
## Same shape to target
inter0, inter_count0, tot_count0 = rechunker(source, source_chunks, source_read_chunks, inter_chunks)

if not (inter0 == source).all():
    raise ValueError()

target1, target_count1, tot_count1 = rechunker(inter0, inter_chunks, target_chunks, target_chunks)

if not (target1 == source).all():
    raise ValueError()

print(round(tot_count0/inter_count0, 2))
print(round(tot_count1/target_count1, 2))
print(round((tot_count1 + tot_count0)/(inter_count0 + target_count1), 2))

target2, target_count2, tot_count2 = rechunker(source, source_chunks, source_chunks, target_chunks)

if not (target2 == source).all():
    raise ValueError()

print(round(tot_count2/target_count2, 2))
print(round(tot_count2/(tot_count1 + tot_count0), 2))


## different shape to target
inter0, inter_count0, tot_count0 = rechunker(source, source_chunks, source_read_chunks, inter_chunks, new_shape)

new_shape_slice = tuple(slice(0, s) for s in new_shape)
if not (inter0 == source[new_shape_slice]).all():
    raise ValueError()

target1, target_count1, tot_count1 = rechunker(inter0, inter_chunks, target_chunks, target_chunks)

if not (target1 == source[new_shape_slice]).all():
    raise ValueError()

print(round(tot_count0/inter_count0, 2))
print(round(tot_count1/target_count1, 2))
print(round((tot_count1 + tot_count0)/(inter_count0 + target_count1), 2))












































































