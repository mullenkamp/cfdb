#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 13:21:10 2023

@author: mike
"""
import numpy as np

# from . import utils
import utils

import rechunker

sup = np.testing.suppress_warnings()
sup.filter(FutureWarning)

########################################################
### Parameters




########################################################
### Helper functions


def loc_index_slice(slice_obj, dim_data):
    """

    """
    start = slice_obj.start
    stop = slice_obj.stop

    ## use np.searchsorted because coordinates are sorted
    if start is None:
        start_idx = None
    else:
        try:
            # start_idx = np.nonzero(dim_data == start)[0][0]
            start_idx = np.searchsorted(dim_data, start)
        except TypeError:
            try:
                start_time = np.datetime64(start)
                start_idx = np.searchsorted(dim_data, start_time)
            except TypeError:
                raise ValueError(f'{start} not in coordinate.')

    ## stop_idx should include the stop label as per pandas
    if stop is None:
        stop_idx = None
    else:
        try:
            stop_idx = np.searchsorted(dim_data, start) + 1
        except TypeError:
            try:
                stop_time = np.datetime64(stop)
                stop_idx = np.searchsorted(dim_data, stop_time) + 1
            except TypeError:
                raise ValueError(f'{stop} not in coordinate.')

    if (stop_idx is not None) and (start_idx is not None):
        if start_idx > stop_idx:
            raise ValueError(f'start index at {start_idx} is after stop index at {stop_idx}.')

    return slice(start_idx, stop_idx)


def loc_index_label(label, dim_data):
    """

    """
    try:
        label_idx = np.searchsorted(dim_data, label)
    except TypeError:
        try:
            label_time = np.datetime64(label)
            label_idx = np.searchsorted(dim_data, label_time)
        except TypeError:
            raise ValueError(f'{label} not in coordinate.')

    return label_idx


def loc_index_array(values, dim_data):
    """

    """
    values = np.asarray(values)

    val_len = len(values)
    if val_len == 0:
        raise ValueError('The array is empty...')
    elif val_len == 1:
        index = loc_index_label(values[0], dim_data)

    ## check if regular
    index = loc_index_slice(slice(values[0], values[-1]), dim_data)

    return index


@sup
def loc_index_combo_one(key, dim_data):
    """

    """
    if isinstance(key, (int, float, str)):
        label_idx = loc_index_label(key, dim_data)

        return label_idx

    elif isinstance(key, slice):
        slice_idx = loc_index_slice(key, dim_data)

        return slice_idx

    elif key is None:
         return slice(None, None)

    elif isinstance(key, (list, np.ndarray)):
        key = np.asarray(key)

        if key.dtype.name == 'bool':
            if len(key) != len(dim_data):
                raise ValueError('If the input is a bool array, then it must be the same length as the coordinate.')

            return key
        else:
            idx = loc_index_array(key, dim_data)

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


# def numpy_indexer_coord(key, coord_name, origin_pos, data):
#     """

#     """
#     if isinstance(key, int):


def slice_int(key, coord_origin_poss, var_shape, pos):
    """

    """
    if key > var_shape[pos]:
        raise ValueError('key is larger than the coord length.')

    # slices = [slice(co, cs) for co, cs in zip(coord_origin_poss, coord_sizes)]

    slice1 = slice(key + coord_origin_poss[pos], key + coord_origin_poss[pos] + 1)

    return slice1


def slice_slice(key, coord_origin_poss, var_shape, pos):
    """

    """
    start = key.start
    if isinstance(start, int):
        start = start + coord_origin_poss[pos]
    else:
        start = coord_origin_poss[pos]

    stop = key.stop
    if isinstance(stop, int):
        stop = stop + coord_origin_poss[pos]
    else:
        stop = var_shape[pos] + coord_origin_poss[pos]

    # slices = [slice(co, cs) for co, cs in zip(coord_origin_poss, coord_sizes)]

    slice1 = slice(start, stop)

    return slice1


def slice_none(coord_origin_poss, var_shape, pos):
    """

    """
    start = coord_origin_poss[pos]
    stop = var_shape[pos] + coord_origin_poss[pos]

    # slices = [slice(co, cs) for co, cs in zip(coord_origin_poss, coord_sizes)]

    slice1 = slice(start, stop)

    return slice1


def index_combo_one(key, coord_origin_poss, var_shape, pos):
    """

    """
    if isinstance(key, int):
        slice1 = slice_int(key, coord_origin_poss, var_shape, pos)
    elif isinstance(key, slice):
        slice1 = slice_slice(key, coord_origin_poss, var_shape, pos)
    elif key is None:
        slice1 = slice_none(key, coord_origin_poss, var_shape, pos)
    else:
        raise TypeError('key must be an int, slice of ints, or None.')

    return slice1


def index_combo_all(key, coord_origin_poss, var_shape):
    """

    """
    if isinstance(key, int):
        slices = [slice(co, cs) for co, cs in zip(coord_origin_poss, var_shape)]
        slices[0] = slice_int(key, coord_origin_poss, var_shape, 0)
    elif isinstance(key, slice):
        slices = [slice(co, cs) for co, cs in zip(coord_origin_poss, var_shape)]
        slices[0] = slice_slice(key, coord_origin_poss, var_shape, 0)
    elif key is None:
        slices = tuple(slice_none(coord_origin_poss, var_shape, pos) for pos in range(0, len(var_shape)))
    elif isinstance(key, tuple):
        key_len = len(key)
        if key_len == 0:
            slices = tuple(slice_none(coord_origin_poss, var_shape, pos) for pos in range(0, len(var_shape)))
        elif key_len != len(var_shape):
            raise ValueError('The tuple key must be the same length as the associated coordinates.')
        else:
            slices = tuple(index_combo_one(key1, coord_origin_poss, var_shape, pos) for pos, key1 in enumerate(key))

    else:
        raise TypeError('key must be an int, slice of ints, or None.')

    return tuple(slices)


def determine_final_array_shape(key, coord_origin_poss, var_shape):
    """

    """
    slices = index_combo_all(key, coord_origin_poss, var_shape)
    new_shape = tuple(s.stop - s.start for s in slices)

    return new_shape


def slices_to_chunks_keys(slices, var_name, var_chunk_shape, clip_ends=True):
    """
    slices from the output of index_combo_all.
    """
    starts = tuple(s.start for s in slices)
    stops = tuple(s.stop for s in slices)
    # chunk_iter1 = rechunker.chunk_range(starts, stops, var_chunk_shape, clip_ends=False)
    chunk_iter2 = rechunker.chunk_range(starts, stops, var_chunk_shape, clip_ends=clip_ends)
    # for full_chunk, partial_chunk in zip(chunk_iter1, chunk_iter2):
    for partial_chunk in chunk_iter2:
        # starts_chunk = tuple(s.start for s in full_chunk)
        starts_chunk = tuple((pc.start//cs) * cs for cs, pc in zip(var_chunk_shape, partial_chunk))
        new_key = utils.make_var_chunk_key(var_name, starts_chunk)

        partial_chunk1 = tuple(slice(pc.start - start, pc.stop - start) for start, pc in zip(starts_chunk, partial_chunk))
        target_chunk = tuple(slice(s.start - start, s.stop - start) for start, s in zip(starts, partial_chunk))

        yield target_chunk, partial_chunk1, new_key





# def indexer_to_keys(key, var_name, var_chunk_shape, coord_origin_poss, coord_sizes):
#     """

#     """
#     if isinstance(key, int):
#         new_pos = key + origin_pos

#         new_key = utils.make_var_chunk_key(var_name, (new_pos,))

#         yield new_key

#     elif isinstance(key, slice):
#         start = key.start
#         if not isinstance(start, int):
#             start = origin_pos

#         stop = key.stop
#         if not isinstance(stop, int):
#             stop = shape[0] + origin_pos

#         chunk_iter = rechunker.chunk_range((start,), (stop,), chunk_shape, clip_ends=False)
#         for chunk in chunk_iter:
#             new_key = utils.make_var_chunk_key(var_name, (chunk[0].start,))

#             yield new_key

#     elif key is None:
#          start = origin_pos
#          stop = shape[0] + origin_pos
    
#          chunk_iter = rechunker.chunk_range((start,), (stop,), chunk_shape, clip_ends=False)
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
        if isinstance(key, (int, float, str, slice, list, np.ndarray)):
            index = index_combo_one(key, self.variable.data)

            return self.variable.encoding.decode(self.variable[index])

        elif isinstance(key, tuple):
            key_len = len(key)

            if key_len == 0:
                return self.variable.encoding.decode(self.variable[()])

            elif key_len > self.variable.ndim:
                raise ValueError('input must have <= ndims.')

            index = []
            for i, k in enumerate(key):
                index_i = index_combo_one(k, self.variable, i)
                index.append(index_i)

            return self.variable.encoding.decode(self.variable[tuple(index)])

        else:
            raise TypeError('You passed a strange object to index...')


    def __setitem__(self, key, value):
        """

        """
        if isinstance(key, (int, float, str, slice, list, np.ndarray)):
            index = index_combo_one(key, self.variable, 0)

            self.variable[index] = self.variable.encoding.encode(value)

        elif isinstance(key, tuple):
            key_len = len(key)

            if key_len == 0:
                self.variable[()] = self.variable.encoding.encode(value)

            elif key_len > self.variable.ndim:
                raise ValueError('input must have <= ndims.')

            index = []
            for i, k in enumerate(key):
                index_i = index_combo_one(k, self.variable, i)
                index.append(index_i)

            self.variable[tuple(index)] = self.variable.encoding.encode(value)

        else:
            raise TypeError('You passed a strange object to index...')













































