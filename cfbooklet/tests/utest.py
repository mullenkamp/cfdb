import copy
from typing import List, Optional, Sequence, Tuple, Iterator
import numpy as np
import itertools
from time import time
from math import ceil, floor, prod, gcd, lcm
from collections import Counter, deque
from itertools import count

###################################################
### Parameters

source_shape = (30, 30)
source_shape = (30, 30, 30)

new_shape = (35, 31)

source_chunk_shape = (5, 2)
target_chunk_shape = (2, 5)
source_chunk_shape = (5, 2, 4)
target_chunk_shape = (2, 5, 3)
itemsize = 1
max_mem = 40
max_mem = 160

###################################################
### Functions


def chunk_keys(
    shape: Tuple[int, ...], chunk_shape: Tuple[int, ...]
) -> Iterator[Tuple[slice, ...]]:
    """Iterator over array indexing keys of the desired chunk sized.

    The union of all keys indexes every element of an array of shape ``shape``
    exactly once. Each array resulting from indexing is of shape ``chunks``,
    except possibly for the last arrays along each dimension (if ``chunks``
    do not even divide ``shape``).
    """
    ranges = [range(ceil(s / c)) for s, c in zip(shape, chunk_shape)]
    for indices in itertools.product(*ranges):
        yield tuple(
            slice(c * i, min(c * (i + 1), s)) for i, s, c in zip(indices, shape, chunk_shape)
        )


def chunk_keys2(
    shape: Tuple[int, ...], chunk_shape: Tuple[int, ...]
) -> Iterator[Tuple[slice, ...]]:
    """Iterator over array indexing keys of the desired chunk sized.

    The union of all keys indexes every element of an array of shape ``shape``
    exactly once. Each array resulting from indexing is of shape ``chunks``,
    except possibly for the last arrays along each dimension (if ``chunks``
    do not even divide ``shape``).
    """
    ranges = [range(ceil(s / c)) for s, c in zip(shape, chunk_shape)]
    dict1 = {}
    for indices in itertools.product(*ranges):
        t1 = tuple(
            slice(c * i, min(c * (i + 1), s)) for i, s, c in zip(indices, shape, chunk_shape)
        )
        chunk = tuple(s.start for s in t1)
        dict1[chunk] = t1

    return dict1


def chunk_keys3(
    shape: Tuple[int, ...], chunk_shape: Tuple[int, ...]
) -> Iterator[Tuple[int, ...]]:
    """Iterator over array indexing keys of the desired chunk sized.

    The union of all keys indexes every element of an array of shape ``shape``
    exactly once. Each array resulting from indexing is of shape ``chunks``,
    except possibly for the last arrays along each dimension (if ``chunks``
    do not even divide ``shape``).
    """
    ranges = [range(ceil(s / c)) for s, c in zip(shape, chunk_shape)]
    for indices in itertools.product(*ranges):
        yield tuple(c * i for i, c in zip(indices, chunk_shape))


def chunk_keys4(
    shape: Tuple[int, ...], chunk_shape: Tuple[int, ...], start_chunk: Tuple[int, ...]=None,
) -> Iterator[Tuple[slice, ...]]:
    """Iterator over array indexing keys of the desired chunk sized.

    The union of all keys indexes every element of an array of shape ``shape``
    exactly once. Each array resulting from indexing is of shape ``chunks``,
    except possibly for the last arrays along each dimension (if ``chunks``
    do not even divide ``shape``).
    """
    if not isinstance(start_chunk, tuple):
        start_chunk = tuple(0 for i in range(len(shape)))

    ranges = [range(ceil(sh / c)) for sh, c in zip(shape, chunk_shape)]
    for indices in itertools.product(*ranges):
        inside = True
        res = []
        for i, sh, c, sc in zip(indices, shape, chunk_shape, start_chunk):
            stop = min(c * (i + 1), sh)
            if stop <= sc:
                inside = False
                continue
            start = c * i
            if start < sc:
                start = sc
            res.append(slice(start, stop))
        if inside:
            yield tuple(res)


def chunk_range(
    chunk_start: Tuple[int, ...], chunk_end: Tuple[int, ...], chunk_step: Tuple[int, ...], include_partial_chunks=True, clip_ends=True,
) -> Iterator[Tuple[slice, ...]]:
    """Iterator over array indexing keys of the desired chunk sized.

    The union of all keys indexes every element of an array of shape ``shape``
    exactly once. Each array resulting from indexing is of shape ``chunks``,
    except possibly for the last arrays along each dimension (if ``chunks``
    do not even divide ``shape``).
    """
    if not isinstance(chunk_start, tuple):
        chunk_start = tuple(0 for i in range(len(chunk_end)))

    if include_partial_chunks:
        start_ranges = [cs * (sc//cs) for cs, sc in zip(chunk_step, chunk_start)]
    else:
        start_ranges = [cs * (((sc - 1)//cs) + 1) for cs, sc in zip(chunk_step, chunk_start)]

    ranges = [range(sr, ec, cs) for ec, cs, sr in zip(chunk_end, chunk_step, start_ranges)]
    # if include_partial_chunks:
    #     for indices in itertools.product(*ranges):
    #         yield tuple(slice(i, min(i + cs, ec)) for i, ec, cs, sc in zip(indices, end_chunk, chunk_step, start_chunk))
    # else:
    for indices in itertools.product(*ranges):
        # print(indices)
        inside = True
        res = []
        for i, ec, cs, sc in zip(indices, chunk_end, chunk_step, chunk_start):
            stop = i + cs
            if stop > ec:
                if clip_ends:
                    stop = ec
                inside = False

            start = i
            if start < sc:
                if clip_ends:
                    start = sc
                inside = False

            res.append(slice(start, stop))

        if inside or include_partial_chunks:
            yield tuple(res)


def calc_ideal_read_chunk_shape(source_chunk_shape, target_chunk_shape):
    """

    """
    return tuple(lcm(i, s) for i, s in zip(source_chunk_shape, target_chunk_shape))


def calc_source_read_chunk_shape(source_chunk_shape, target_chunk_shape, itemsize, max_mem):
    """

    """
    max_cells = int(max_mem / itemsize)
    source_len = len(source_chunk_shape)
    target_len = len(target_chunk_shape)

    if source_len != target_len:
        raise ValueError('The source_chunk_shape and target_chunk_shape do not have the same number of dims.')

    tot_source = prod(source_chunk_shape)
    if tot_source >= max_cells:
        return source_chunk_shape

    new_chunks = [lcm(i, s) for i, s in zip(source_chunk_shape, target_chunk_shape)]

    ## Max mem
    tot_target = prod(new_chunks)
    ideal_target = tot_target
    pos = 0
    while tot_target > max_cells:
        prod_chunk = new_chunks[pos]
        source_chunk = source_chunk_shape[pos]
        if prod_chunk > source_chunk:
            new_chunks[pos] = prod_chunk - source_chunk

        tot_target = prod(new_chunks)

        if tot_target == tot_source:
            return source_chunk_shape
        else:
            if pos + 1 == source_len:
                pos = 0
            else:
                pos += 1

    ## Min mem
    n_chunks_write = tuple(s//target_chunk_shape[i] for i, s in enumerate(new_chunks))
    for i in range(len(new_chunks)):
        while True:
            n_chunk_write = n_chunks_write[i]
            prod_chunk = new_chunks[i]
            source_chunk = source_chunk_shape[i]
            target_chunk = target_chunk_shape[i]
            new_chunk = prod_chunk - source_chunk
            if new_chunk//target_chunk == n_chunk_write:
                new_chunks[i] = new_chunk
            else:
                break

    ## Calc n chunks per read
    n_chunks_per_read = prod(tuple(nc//sc for nc, sc in zip(new_chunks, source_chunk_shape)))

    return tuple(new_chunks), n_chunks_per_read, ideal_target


def calc_n_chunks(shape, chunk_shape):
    """

    """
    chunk_start = tuple(0 for i in range(len(shape)))
    chunk_iter = chunk_range(chunk_start, shape, chunk_shape)

    counter = count()
    deque(zip(chunk_iter, counter), maxlen=0)

    return next(counter)


def calc_n_reads_target_simple(source_shape, source_chunk_shape, target_chunk_shape):
    """

    """
    chunk_start = tuple(0 for i in range(len(source_shape)))
    read_counter = count()

    for write_chunk in chunk_range(chunk_start, source_shape, target_chunk_shape):
        write_chunk_start = tuple(rc.start for rc in write_chunk)
        write_chunk_stop = tuple(rc.stop for rc in write_chunk)
        for chunk_slice in chunk_range(write_chunk_start, write_chunk_stop, source_chunk_shape):
            next(read_counter)

    return next(read_counter)


# def calc_n_reads_target_fancy(source_shape, source_chunk_shape, source_read_chunk_shape, target_chunk_shape):
#     """

#     """
#     chunk_start = tuple(0 for i in range(len(source_shape)))

#     all_write_chunks = set()
#     for write_chunk in chunk_range(chunk_start, source_shape, target_chunk_shape):
#         write_chunk_start = tuple(s.start for s in write_chunk)
#         all_write_chunks.add(write_chunk_start)

#     read_chunk_iter = chunk_range(chunk_start, source_shape, source_read_chunk_shape)
#     read_counter = count()
#     write_counter = count()
#     for read_chunk in read_chunk_iter:
#         read_chunk_start = tuple(rc.start for rc in read_chunk)
#         read_chunk_stop = tuple(rc.stop for rc in read_chunk)
#         write_chunks = list(chunk_range(read_chunk_start, read_chunk_stop, target_chunk_shape, False))
#         if write_chunks:
#             for chunk_slice in chunk_range(read_chunk_start, read_chunk_stop, source_chunk_shape):
#                 # start_chunk = tuple(s.start for s in chunk_slice)
#                 print(chunk_slice)
#                 next(read_counter)
#             for write_chunk in write_chunks:
#                 write_chunk_start = tuple(s.start for s in write_chunk)
#                 all_write_chunks.remove(write_chunk_start)
#                 # print(write_chunk)
#                 next(write_counter)

#     for write_chunk_start in all_write_chunks:
#         write_chunk_stop = tuple(wcs + tcsh for wcs, tcsh in zip(write_chunk_start, target_chunk_shape))
#         for chunk_slice in chunk_range(write_chunk_start, write_chunk_stop, source_chunk_shape):
#             next(read_counter)
#         next(write_counter)

#     return next(read_counter)


def calc_n_reads_target_fancy(source_shape, source_chunk_shape, target_chunk_shape, itemsize, max_mem):
    """

    """
    ## Calc the optimum read_chunk shape
    ## Calc n chunks per read
    ## Calc ideal read chunking shape
    source_read_chunk_shape, n_chunks_per_read, ideal_read_chunk_size = calc_source_read_chunk_shape(source_chunk_shape, target_chunk_shape, itemsize, max_mem)

    # ## Calc n chunks per read
    # n_chunks_per_read = prod(tuple(nc//sc for nc, sc in zip(source_read_chunk_shape, source_chunk_shape)))

    ## Calc ideal read chunking shape
    ideal_read_chunk_shape = calc_ideal_read_chunk_shape(source_chunk_shape, target_chunk_shape)

    ## Chunk_start
    chunk_start = tuple(0 for i in range(len(source_shape)))

    ## Counters
    read_counter = count()
    write_counter = count()

    ## If the read chunking is set to the ideal chunking case, then use the simple implementation. Otherwise, the more complicated one.
    if source_read_chunk_shape == ideal_read_chunk_shape:
        read_chunk_iter = chunk_range(chunk_start, source_shape, source_read_chunk_shape)
        for read_chunk_grp in read_chunk_iter:
            read_chunk_grp_start = tuple(s.start for s in read_chunk_grp)
            read_chunk_grp_stop = tuple(s.stop for s in read_chunk_grp)
            for read_chunk in chunk_range(read_chunk_grp_start, read_chunk_grp_stop, source_chunk_shape, True, True):
                # print(read_chunk)
                next(read_counter)
            for write_chunk1 in chunk_range(read_chunk_grp_start, read_chunk_grp_stop, target_chunk_shape, include_partial_chunks=True):
                next(write_counter)

    else:
        writen_chunks = set() # Need to keep track of the bulk writes

        read_chunk_iter = chunk_range(chunk_start, source_shape, target_chunk_shape)

        for write_chunk in read_chunk_iter:
            write_chunk_start = tuple(s.start for s in write_chunk)
            write_chunk_stop = tuple(s.stop for s in write_chunk)
            if write_chunk_start not in writen_chunks:
                read_chunk_start = tuple(rc * (wc//rc) for wc, rc in zip(write_chunk_start, source_chunk_shape))
                read_chunk_stop = tuple(min(max(rcs + rc, wc), sh) for rcs, rc, wc, sh in zip(read_chunk_start, source_read_chunk_shape, write_chunk_stop, source_shape))
                # read_chunk_stop = tuple(min(wc + rcg, sh) for wc, rcg, sh in zip(write_chunk_start, target_chunk_shape, source_shape))
                read_chunks = list(chunk_range(write_chunk_start, read_chunk_stop, source_chunk_shape, True, False))
                if len(read_chunks) <= n_chunks_per_read:
                    for read_chunk in read_chunks:
                        # print(read_chunk)
                        # print(len(read_chunks))
                        next(read_counter)

                    # write_chunk_stop = tuple(min(wc + rc, sh) for wc, rc, sh in zip(write_chunk_start, source_read_chunk_shape, source_shape))
                    for write_chunk1 in chunk_range(write_chunk_start, read_chunk_stop, target_chunk_shape, include_partial_chunks=False):
                        # print(write_chunk1)
                        write_chunk1_start = tuple(s.start for s in write_chunk1)
                        if write_chunk1_start not in writen_chunks:
                            writen_chunks.add(write_chunk1_start)
                            next(write_counter)
                else:
                    for read_chunk in chunk_range(write_chunk_start, write_chunk_stop, source_chunk_shape, True, False):
                        # print(read_chunk)
                        next(read_counter)
                    writen_chunks.add(write_chunk_start)
                    next(write_counter)

    return next(read_counter), next(write_counter)


# def calc_read_ratio_source(source_read_chunk_shape, source_chunk_shape, target_chunk_shape):
#     """

#     """
#     n_chunks_read = prod(tuple(s//source_chunk_shape[i] for i, s in enumerate(source_read_chunk_shape)))
#     n_chunks_bulk = prod(tuple(s//target_chunk_shape[i] for i, s in enumerate(source_read_chunk_shape)))

#     if n_chunks_bulk == 0:
#         return calc_n_chunks_source(source_chunk_shape, target_chunk_shape)
#     else:
#         return round(n_chunks_read/n_chunks_bulk, 3)





def find_first_source_read_chunk(slice1, source_chunk_shape):
    """

    """
    return tuple(slice1[i].start//s * s for i, s in enumerate(source_chunk_shape))


def cut_source_read_chunk_shape(source_read_chunk_shape, first_source_read_chunk, shape):
    """

    """
    source_read_chunk_shape2 = []
    for i, s in enumerate(shape):
        diff = s - first_source_read_chunk[i]
        read_chunk = source_read_chunk_shape[i]
        if diff < read_chunk:
            source_read_chunk_shape2.append(diff)
        else:
            source_read_chunk_shape2.append(read_chunk)

    return tuple(source_read_chunk_shape2)


def chunk_source_slices(slice1, source_read_chunk_shape, source_chunk_shape, first_source_read_chunk, shape):
    """

    """
    source_read_chunk_shape2 = cut_source_read_chunk_shape(source_read_chunk_shape, first_source_read_chunk, shape)

    # temp = []
    for source_slice in chunk_keys(source_read_chunk_shape2, source_chunk_shape):
        # print(source_slice)
        source_read_slice = []
        source_mem_slice = []
        inter_mem_slice = []
        for i, s in enumerate(source_slice):
            slice1_start = slice1[i].start
            slice1_stop = slice1[i].stop
            source_read_start = s.start + first_source_read_chunk[i]
            source_read_stop = s.stop + first_source_read_chunk[i]
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


def increase_read_chunks(write_slice, source_read, source_read_chunk_shape, source_chunk_shape):
    """

    """
    dim_bool = [s.stop > (source_read[i] + source_read_chunk_shape[i]) for i, s in enumerate(write_slice)]
    while any(dim_bool):
        source_read_chunk_shape = tuple(source_read_chunk_shape[i] + source_chunk_shape[i] if bool1 else source_read_chunk_shape[i] for i, bool1 in enumerate(dim_bool))
        dim_bool = [s.stop > (source_read[i] + source_read_chunk_shape[i]) for i, s in enumerate(write_slice)]

    return source_read_chunk_shape


def simple_rechunker(source, source_chunk_shape, target_chunk_shape, new_shape=None, dtype='int32'):
    """

    """
    shape = source.shape
    if isinstance(new_shape, tuple):
        bools1 = [new_shape[i] <= s for i, s in enumerate(shape)]
        if not all(bools1):
            raise ValueError('The new shape must be >= the source shape.')
        shape = new_shape

    target_mem_arr1 = np.zeros(target_chunk_shape, dtype=dtype)
    target = np.zeros(shape, dtype=dtype)

    tot_count = 0
    target_count = 0
    for slice1 in chunk_keys(shape, target_chunk_shape):
        # print(slice1)

        first_source_read_chunk = find_first_source_read_chunk(slice1, source_chunk_shape)

        source_read_chunk_shape1 = source_chunk_shape
        source_read_chunk_shape1 = increase_read_chunks(slice1, first_source_read_chunk, source_read_chunk_shape1)

        iter_source_slices = chunk_source_slices(slice1, source_read_chunk_shape1, source_chunk_shape, first_source_read_chunk, shape)
        for source_read_slice, source_mem_slice, target_mem_slice in iter_source_slices:
            target_mem_arr1[target_mem_slice] = source[source_read_slice][source_mem_slice]
            tot_count += 1

        slice_shape = tuple(s.stop - s.start for s in slice1)
        if slice_shape == target_chunk_shape:
            target[slice1] = target_mem_arr1
        else:
            new_mem_slice = tuple(slice(0, s) for s in slice_shape)
            target[slice1] = target_mem_arr1[new_mem_slice]

        target_count += 1

    return target, target_count, tot_count


###################################################
### Chunker testing

out_chunks = rechunking_plan(shape, shape, source_chunk_shape, target_chunk_shape, itemsize, max_mem, True, False)

# out_chunks = rechunking_plan(shape, new_shape, source_chunk_shape, target_chunk_shape, itemsize, max_mem, True, True)

source = np.arange(1, np.prod(shape) + 1).reshape(shape)

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
#     raise ValueError()

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


































































