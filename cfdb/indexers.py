#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 13:21:10 2023

@author: mike
"""
import numpy as np
import rechunkit

from . import utils
# import utils

# sup = np.testing.suppress_warnings()
# sup.filter(FutureWarning)

########################################################
### Parameters




########################################################
### Helper functions


def loc_index_numeric(key, coord_data):
    """

    """
    # if coord_data.dtype.kind == 'f':
    #     label_idx = np.nonzero(np.isclose(coord_data, key))[0][0]
    # else:
    #     label_idx = np.searchsorted(coord_data, key)

    label_idx = np.searchsorted(coord_data, key)

    return int(label_idx)


def loc_index_str(key, coord_data):
    """

    """
    if coord_data.dtype.kind == 'M':
        key = np.array(key, dtype=coord_data.dtype)
        label_idx = loc_index_numeric(key, coord_data)
    else:
        label_idx = np.nonzero(coord_data == key)[0][0]

    return int(label_idx)


def loc_index_slice(slice_obj, coord_data):
    """

    """
    start = slice_obj.start
    stop = slice_obj.stop

    if start is None:
        start_idx = None
    else:
        if isinstance(start, str):
            start_idx = loc_index_str(start, coord_data)
        else:
            start_idx = loc_index_numeric(start, coord_data)

    if stop is None:
        stop_idx = None
    else:
        if isinstance(stop, str):
            stop_idx = loc_index_str(stop, coord_data)
        else:
            stop_idx = loc_index_numeric(stop, coord_data)

    if (stop_idx is not None) and (start_idx is not None):
        if start_idx >= stop_idx:
            raise ValueError(f'start index at {start_idx} is equal to or greater than the stop index at {stop_idx}.')

    return slice(start_idx, stop_idx)


# def loc_index_array(values, dim_data):
#     """

#     """
#     values = np.asarray(values)

#     val_len = len(values)
#     if val_len == 0:
#         raise ValueError('The array is empty...')
#     elif val_len == 1:
#         index = loc_index_label(values[0], dim_data)

#     ## check if regular
#     index = loc_index_slice(slice(values[0], values[-1]), dim_data)

#     return index



# @sup
def loc_index_combo_one(key, coord_data):
    """

    """
    if isinstance(key, str):
        index_idx = loc_index_str(key, coord_data)

    elif isinstance(key, slice):
        index_idx = loc_index_slice(key, coord_data)

    elif key is None:
        index_idx = None

    else:
        index_idx = loc_index_numeric(key, coord_data)

    return index_idx


def loc_index_combo_all(key, coords):
    """

    """
    if isinstance(key, str):
        idx = loc_index_str(key, coords[0].data)
    elif isinstance(key, slice):
        idx = loc_index_slice(key, coords[0].data)
    elif key is None:
        idx = None
    elif isinstance(key, tuple):
        key_len = len(key)
        if key_len == 0:
            idx = None
        else:
            idx = tuple(loc_index_combo_one(key1, coords[pos].data) for pos, key1 in enumerate(key))

    else:
        idx = loc_index_numeric(key, coords[0].data)

    return idx

# def pos_to_keys(var_name, shape, pos):
#     """

#     """
#     ndims = len(shape)
#     if isinstance(pos, slice):
#         start = pos.start
#         stop = pos.stop
#         if start is None:
#             start = 0
#         if stop is None:


# def numpy_indexer_coord(key, coord_name, origin, data):
#     """

#     """
#     if isinstance(key, int):


def slice_int(key, coord_origins, var_shape, pos):
    """

    """
    if key > var_shape[pos]:
        raise ValueError('key is larger than the coord length.')

    slice1 = slice(key + coord_origins[pos], key + coord_origins[pos] + 1)

    return slice1


def slice_slice(key, coord_origins, var_shape, pos):
    """

    """
    start = key.start
    if isinstance(start, int):
        start = start + coord_origins[pos]
    else:
        start = coord_origins[pos]

    stop = key.stop
    if isinstance(stop, int):
        stop = stop + coord_origins[pos]
    else:
        stop = var_shape[pos] + coord_origins[pos]

    # slices = [slice(co, cs) for co, cs in zip(coord_origins, coord_sizes)]

    # TODO - Should I leave this test in here? Or should this be allowed?
    if start == stop:
        raise ValueError('The start and stop for the slice is the same, which will produce 0 output.')

    slice1 = slice(start, stop)

    return slice1


def slice_none(coord_origins, var_shape, pos):
    """

    """
    start = coord_origins[pos]
    stop = var_shape[pos] + coord_origins[pos]

    # slices = [slice(co, cs) for co, cs in zip(coord_origins, coord_sizes)]

    slice1 = slice(start, stop)

    return slice1


def index_combo_one(key, coord_origins, var_shape, pos):
    """

    """
    if isinstance(key, slice):
        slice1 = slice_slice(key, coord_origins, var_shape, pos)
    elif isinstance(key, int):
        slice1 = slice_int(key, coord_origins, var_shape, pos)
    elif key is None:
        slice1 = slice_none(coord_origins, var_shape, pos)
    else:
        raise TypeError('key must be an int, slice of ints, or None.')

    return slice1


def index_combo_all(key, coord_origins, var_shape):
    """

    """
    if isinstance(key, int):
        slices = [slice(co, cs) for co, cs in zip(coord_origins, var_shape)]
        slices[0] = slice_int(key, coord_origins, var_shape, 0)
    elif isinstance(key, slice):
        slices = [slice(co, cs) for co, cs in zip(coord_origins, var_shape)]
        slices[0] = slice_slice(key, coord_origins, var_shape, 0)
    elif key is None:
        slices = tuple(slice_none(coord_origins, var_shape, pos) for pos in range(0, len(var_shape)))
    elif isinstance(key, tuple):
        key_len = len(key)
        if key_len == 0:
            slices = tuple(slice_none(coord_origins, var_shape, pos) for pos in range(0, len(var_shape)))
        elif key_len != len(var_shape):
            raise ValueError('The tuple key must be the same length as the associated coordinates.')
        else:
            slices = tuple(index_combo_one(key1, coord_origins, var_shape, pos) for pos, key1 in enumerate(key))

    else:
        raise TypeError('key must be an int, slice of ints, or None.')

    return tuple(slices)


def determine_final_array_shape(key, coord_origins, var_shape):
    """

    """
    slices = index_combo_all(key, coord_origins, var_shape)
    new_shape = tuple(s.stop - s.start for s in slices)

    return new_shape


def slices_to_keys(slices, var_name, var_chunk_shape):
    """
    slices to keys
    """
    starts = tuple(s.start for s in slices)
    stops = tuple(s.stop for s in slices)
    chunk_iter2 = rechunkit.chunk_range(starts, stops, var_chunk_shape)
    for partial_chunk in chunk_iter2:
        starts_chunk = tuple((pc.start//cs) * cs for cs, pc in zip(var_chunk_shape, partial_chunk))
        new_key = utils.make_var_chunk_key(var_name, starts_chunk)

        yield new_key


def slices_to_chunks_keys(slices, var_name, var_chunk_shape, clip_ends=True):
    """
    slices from the output of index_combo_all.
    """
    starts = tuple(s.start for s in slices)
    stops = tuple(s.stop for s in slices)
    chunk_iter2 = rechunkit.chunk_range(starts, stops, var_chunk_shape, clip_ends=clip_ends)
    for partial_chunk in chunk_iter2:
        starts_chunk = tuple((pc.start//cs) * cs for cs, pc in zip(var_chunk_shape, partial_chunk))
        new_key = utils.make_var_chunk_key(var_name, starts_chunk)

        partial_chunk1 = tuple(slice(pc.start - start, pc.stop - start) for start, pc in zip(starts_chunk, partial_chunk))
        target_chunk = tuple(slice(s.start - start, s.stop - start) for start, s in zip(starts, partial_chunk))

        yield target_chunk, partial_chunk1, new_key



def check_sel_input_data(sel, input_data, coord_origins, shape):
    """

    """
    slices = index_combo_all(sel, coord_origins, shape)
    slices_shape = tuple(s.stop - s.start for s in slices)

    if input_data.shape != slices_shape:
        raise ValueError('The selection shape is not equal to the input data.')

    return slices



# def indexer_to_keys(key, var_name, var_chunk_shape, coord_origins, coord_sizes):
#     """

#     """
#     if isinstance(key, int):
#         new_pos = key + origin

#         new_key = utils.make_var_chunk_key(var_name, (new_pos,))

#         yield new_key

#     elif isinstance(key, slice):
#         start = key.start
#         if not isinstance(start, int):
#             start = origin

#         stop = key.stop
#         if not isinstance(stop, int):
#             stop = shape[0] + origin

#         chunk_iter = rechunkit.chunk_range((start,), (stop,), chunk_shape, clip_ends=False)
#         for chunk in chunk_iter:
#             new_key = utils.make_var_chunk_key(var_name, (chunk[0].start,))

#             yield new_key

#     elif key is None:
#          start = origin
#          stop = shape[0] + origin

#          chunk_iter = rechunkit.chunk_range((start,), (stop,), chunk_shape, clip_ends=False)
#          for chunk in chunk_iter:
#              new_key = utils.make_var_chunk_key(var_name, (chunk[0].start,))

#              yield new_key

#     # elif isinstance(key, (list, np.ndarray)):
#     #     key = np.asarray(key)

#     #     if key.dtype.kind == 'b':
#     #         if len(key) != shape[0]:
#     #             raise ValueError('If the input is a bool array, then it must be the same length as the coordinate.')
#     #     elif key.dtype.kind not in ('i', 'u'):
#     #         raise TypeError('If the input is an array, then it must be either a bool of the length of the coordinate or integers.')

#     #         return key
#     #     else:
#     #         idx = index_array(key, dim_data)

#     #         return idx
#     else:
#         raise TypeError('key must be an int, slice of ints, or None.')




#####################################################3
### Classes


class LocationIndexer:
    """

    """
    def __init__(self, variable):
        """

        """
        self.variable = variable


    def __getitem__(self, key):
        """

        """
        idx = loc_index_combo_all(key, self.variable.coords)

        return self.variable.get(idx)



    def __setitem__(self, key, data):
        """

        """
        idx = loc_index_combo_all(key, self.variable.coords)

        self.variable[idx] = data














































