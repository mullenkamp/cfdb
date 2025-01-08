#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 13:21:10 2023

@author: mike
"""
import numpy as np

# from . import utils
import utils


sup = np.testing.suppress_warnings()
sup.filter(FutureWarning)

########################################################
### Parameters




########################################################
### Helper functions


def index_slice(slice_obj, dim_data):
    """

    """
    start = slice_obj.start
    stop = slice_obj.stop

    ## If the np.nonzero finds nothing, then it fails
    if start is None:
        start_idx = None
    else:
        try:
            start_idx = np.nonzero(dim_data == start)[0][0]
        except IndexError:
            try:
                start_time = np.datetime64(start)
                start_idx = np.nonzero(dim_data == start_time)[0][0]
            except IndexError:
                raise ValueError(f'{start} not in coordinate.')

    ## stop_idx should include the stop label as per pandas
    if stop is None:
        stop_idx = None
    else:
        try:
            stop_idx = np.nonzero(dim_data == stop)[0][0] + 1
        except IndexError:
            try:
                stop_time = np.datetime64(stop)
                stop_idx = np.nonzero(dim_data == stop_time)[0][0] + 1
            except IndexError:
                raise ValueError(f'{stop} not in coordinate.')

    if (stop_idx is not None) and (start_idx is not None):
        if start_idx > stop_idx:
            raise ValueError(f'start index at {start_idx} is after stop index at {stop_idx}.')

    return slice(start_idx, stop_idx)


def index_label(label, dim_data):
    """

    """
    try:
        label_idx = np.nonzero(dim_data == label)[0][0]
    except IndexError:
        try:
            label_time = np.datetime64(label)
            label_idx = np.nonzero(dim_data == label_time)[0][0]
        except IndexError:
            raise ValueError(f'{label} not in coordinate.')

    return label_idx


def index_array(values, dim_data):
    """

    """
    values = np.asarray(values)

    val_len = len(values)
    if val_len == 0:
        raise ValueError('The array is empty...')
    elif val_len == 1:
        index = index_label(values[0], dim_data)

    ## check if regular
    elif utils.is_regular_index(values):
        index = index_slice(slice(values[0], values[-1]), dim_data)

    # TODO I might need to do something more fancy here...
    else:
        index = values

    return index


@sup
def index_combo_one(key, variable, pos):
    """

    """
    if isinstance(key, (int, float, str)):
        dim_data = variable.file[variable.coords[pos]].data
        label_idx = index_label(key, dim_data)

        return label_idx

    elif isinstance(key, slice):
        dim_data = variable.file[variable.coords[pos]].data
        slice_idx = index_slice(key, dim_data)

        return slice_idx

    elif key is None:
         return slice(None, None)

    elif isinstance(key, (list, np.ndarray)):
        key = np.asarray(key)

        dim_data = variable.file[variable.coords[pos]].data

        if key.dtype.name == 'bool':
            if len(key) != len(dim_data):
                raise ValueError('If the input is a bool array, then it must be the same length as the coordinate.')

            return key
        else:
            idx = index_array(key, dim_data)

            return idx


def pos_to_keys(var_name, shape, pos):
    """

    """
    ndims = len(shape)
    if isinstance(pos, slice):
        start = pos.start
        stop = pos.stop
        if start is None:
            start = 0
        if stop is None:






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
            index = index_combo_one(key, self.variable, 0)

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
            raise ValueError('You passed a strange object to index...')


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
            raise ValueError('You passed a strange object to index...')













































