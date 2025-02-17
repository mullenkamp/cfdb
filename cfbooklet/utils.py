#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 19:52:08 2022

@author: mike
"""
import io
import pathlib
# import h5py
import os
import numpy as np
import msgspec
import re
# import xarray as xr
# from time import time
# from datetime import datetime
import cftime
import math
from typing import Set, Optional, Dict, Tuple, List, Union, Any
# import dateutil.parser as dparser
# import numcodecs
# import hdf5plugin

import data_models

########################################################
### Parmeters


CHUNK_BASE = 32*1024    # Multiplier by which chunks are adjusted
CHUNK_MIN = 32*1024      # Soft lower limit (32k)
CHUNK_MAX = 3*1024**2   # Hard upper limit (4M)

time_str_conversion = {'days': 'datetime64[D]',
                       'hours': 'datetime64[h]',
                       'minutes': 'datetime64[m]',
                       'seconds': 'datetime64[s]',
                       'milliseconds': 'datetime64[ms]',
                       'microseconds': 'datetime64[us]',
                       'nanoseconds': 'datetime64[ns]'}

# enc_fields = ('units', 'calendar', 'dtype', 'missing_value', '_FillValue', 'add_offset', 'scale_factor', 'dtype_decoded', 'dtype_encoded', 'compression')

fillvalue_dict = {'int8': -128, 'int16': -32768, 'int32': -2147483648, 'int64': -9223372036854775808, 'float32': np.nan, 'float64': np.nan, 'str': ''}

# ignore_attrs = ('DIMENSION_LIST', 'DIMENSION_LABELS', 'DIMENSION_SCALE', 'REFERENCE_LIST', 'CLASS', 'NAME', '_Netcdf4Coordinates', '_Netcdf4Dimid')

attrs_key_str = '_{var_name}.attrs'

var_regex = "^[a-zA-Z][a-z0-9_]*$"
var_pattern = re.compile(var_regex)

#########################################################
### Classes


# class ChunkIterator:
#     """
#     Class to iterate through list of chunks of a given dataset
#     """
#     def __init__(self, chunks, shape, source_sel=None):
#         self._shape = shape
#         rank = len(shape)

#         # if not dset.chunks:
#         #     # can only use with chunked datasets
#         #     raise TypeError("Chunked dataset required")

#         self._layout = chunks
#         if source_sel is None:
#             # select over entire dataset
#             slices = []
#             for dim in range(rank):
#                 slices.append(slice(0, self._shape[dim]))
#             self._sel = tuple(slices)
#         else:
#             if isinstance(source_sel, slice):
#                 self._sel = (source_sel,)
#             else:
#                 self._sel = source_sel
#         if len(self._sel) != rank:
#             raise ValueError("Invalid selection - selection region must have same rank as dataset")
#         self._chunk_index = []
#         for dim in range(rank):
#             s = self._sel[dim]
#             if s.start < 0 or s.stop > self._shape[dim] or s.stop <= s.start:
#                 raise ValueError("Invalid selection - selection region must be within dataset space")
#             index = s.start // self._layout[dim]
#             self._chunk_index.append(index)

#     def __iter__(self):
#         return self

#     def __next__(self):
#         rank = len(self._shape)
#         slices = []
#         if rank == 0 or self._chunk_index[0] * self._layout[0] >= self._sel[0].stop:
#             # ran past the last chunk, end iteration
#             raise StopIteration()

#         for dim in range(rank):
#             s = self._sel[dim]
#             start = self._chunk_index[dim] * self._layout[dim]
#             stop = (self._chunk_index[dim] + 1) * self._layout[dim]
#             # adjust the start if this is an edge chunk
#             if start < s.start:
#                 start = s.start
#             if stop > s.stop:
#                 stop = s.stop  # trim to end of the selection
#             s = slice(start, stop, 1)
#             slices.append(s)

#         # bump up the last index and carry forward if we run outside the selection
#         dim = rank - 1
#         while dim >= 0:
#             s = self._sel[dim]
#             self._chunk_index[dim] += 1

#             chunk_end = self._chunk_index[dim] * self._layout[dim]
#             if chunk_end < s.stop:
#                 # we still have room to extend along this dimensions
#                 return tuple(slices)

#             if dim > 0:
#                 # reset to the start and continue iterating with higher dimension
#                 self._chunk_index[dim] = 0
#             dim -= 1
#         return tuple(slices)


#########################################################
### Functions


def dataset_finalizer(blt_file, sys_meta):
    """

    """
    old_meta_data = blt_file.get_metadata()
    old_meta = msgspec.convert(old_meta_data, data_models.SysMeta)
    if old_meta != sys_meta:
        blt_file.set_metadata(msgspec.to_builtins(sys_meta))

    # if old_meta_data is not None:
    #     old_meta = msgspec.convert(old_meta_data, data_models.SysMeta)

    #     if old_meta != sys_meta:
    #         blt_file.set_metadata(msgspec.to_builtins(sys_meta))
    blt_file.close()


def attrs_finalizer(blt_file, attrs, var_name):
    """

    """
    if attrs:
        key = attrs_key_str.format(var_name=var_name)
        old_attrs = blt_file.get(key)
        if old_attrs != attrs:
            blt_file.set(key, attrs)


def compute_scale_and_offset(min_value: (int, float, np.number), max_value: (int, float, np.number), dtype: np.dtype):
    """
    Computes the scale (slope) and offset for a dataset using a min value, max value, and the required np.dtype. It leaves one value at the lower extreme to use for the nan fillvalue.
    These are the min values set asside for the fillvalue (up to 64 bits).
    int8:  -128
    int16: -32768
    int32: -2147483648
    int64: -9223372036854775808

    Unsigned integers are allowed and a value of 0 is set asside for the fillvalue.

    Parameters
    ----------
    min_value : int or float
        The min value of the dataset.
    max_value : int or float
        The max value of the dataset.
    dtype : np.dtype
        The data type that you want to shrink the data down to.

    Returns
    -------
    scale, offset as floats
    """
    bits = dtype.itemsize * 8
    data_range = max_value - min_value
    if bits < 64:
        target_range = 2**bits - 2
        target_min = -(2**(bits - 1) - 1)
        slope = data_range / target_range
    else:
        data_power = int(math.log10(data_range))
        target_range = 2**bits
        target_power = int(math.log10(target_range))
        target_min = -10**(target_power - 1)
        slope = 10**-(target_power - data_power)

    # Correction if the dtype is unsigned
    if dtype.kind == 'u':
        target_min = 1

    offset = min_value - (slope*target_min)

    return slope, offset


def check_var_name(var_name):
    """
    Function to test if the user-supplied var name is allowed.
    """
    if isinstance(var_name, str):
        if len(var_name) <= 256:
            if var_pattern.match(var_name):
                return True
    return False


def parse_var_inputs(name: str, data: np.ndarray | None = None, shape: Tuple[int] | None = None, chunk_shape: Tuple[int] | None = None, dtype_decoded: str | np.dtype | None = None, dtype_encoded: str | np.dtype | None = None, fillvalue: Union[int, float, str] = None, scale_factor: Union[float, int, None] = None, add_offset: Union[float, int, None] = None):
    """
    Function to process the inputs to a variable creation function.
    """
    ## Check var name
    if not check_var_name(name):
        raise ValueError(f'{name} is not a valid variable name.')

    ## Check data, shape, and dtype
    if isinstance(data, np.ndarray):
        shape = data.shape
        dtype_decoded = data.dtype
    else:
        if not isinstance(shape, tuple):
            raise ValueError('If data is not passed, then shape must be passed.')
        else:
            if not all([isinstance(i, (int, np.integer)) for i in shape]):
                raise ValueError('shape must be a tuple of ints.')
        if not isinstance(dtype_decoded, (str, np.dtype)):
            raise ValueError('If data is not passed, then dtype_decoded must be passed.')
        dtype_decoded = np.dtype(dtype_decoded)

    if not isinstance(dtype_encoded, (str, np.dtype)):
        dtype_encoded = dtype_decoded

    ## Chunk shape
    if isinstance(chunk_shape, tuple):
        if not all([isinstance(i, (int, np.integer)) for i in chunk_shape]):
            raise ValueError('chunk_shape must be a tuple of ints.')
    else:
        chunk_shape = guess_chunk(shape, shape, dtype_encoded)

    ## Fillvalue
    kind = dtype_encoded.kind
    if fillvalue is not None:
        fillvalue_dtype = np.dtype(type(fillvalue))

        if kind == 'u' and fillvalue_dtype.kind == 'i':
            if fillvalue < 0:
                raise ValueError('The dtype_encoded is an unsigned integer, but the fillvalue is < 0.')
            elif fillvalue_dtype.kind != kind:
                raise ValueError('The fillvalue dtype is not the same as the dtype_encoded dtype.')
    else:
        if kind == 'u':
            fillvalue = 0
        elif kind == 'f':
            fillvalue = np.nan
        elif kind == 'U':
            fillvalue = ''
        elif kind == 'i':
            fillvalue = fillvalue_dict[dtype_encoded.name]
        elif kind == 'M':
            fillvalue = np.datetime64('nat')
        else:
            raise TypeError('Unknown/unsupported data type.')

    ## Scale and offset
    if scale_factor is None and isinstance(add_offset, (int, float, np.number)):
        scale_factor = 1
    if isinstance(scale_factor, (int, float, np.number)) and add_offset is None:
        add_offset = 0

    if not isinstance(scale_factor, (int, float, np.number)) or isinstance(add_offset, (int, float, np.number)):
        raise ValueError('sclae_factor and add_offset must be either ints or floats.')







def encode_datetime(data, units=None, calendar='gregorian'):
    """

    """
    if units is None:
        output = data.astype('datetime64[s]').astype('int64')
    else:
        if '1970-01-01' in units:
            time_unit = units.split()[0]
            output = data.astype(time_str_conversion[time_unit]).astype('int64')
        else:
            output = cftime.date2num(data.astype('datetime64[s]').tolist(), units, calendar)

    return output


def decode_datetime(data, units=None, calendar='gregorian'):
    """

    """
    if units is None:
        output = data.astype('datetime64[s]')
    else:
        if '1970-01-01' in units:
            time_unit = units.split()[0]
            output = data.astype(time_str_conversion[time_unit])
        else:
            output = cftime.num2pydate(data, units, calendar).astype('datetime64[s]')

    return output


def encode_data(data, dtype_encoded, missing_value, add_offset=0, scale_factor=None, units=None, calendar=None, **kwargs) -> bytes:
    """

    """
    if 'datetime64' in data.dtype.name:
        data = encode_datetime(data, units, calendar)

    elif isinstance(scale_factor, (int, float, np.number)):
        # precision = int(np.abs(np.log10(val['scale_factor'])))
        data = np.round((data - add_offset)/scale_factor)

        if isinstance(missing_value, (int, np.number)):
            data[np.isnan(data)] = missing_value

    if (data.dtype.name != dtype_encoded) or (data.dtype.name == 'object'):
        data = data.astype(dtype_encoded)

    return data.tobytes()


def decode_data(data: bytes, dtype_encoded, dtype_decoded, missing_value, add_offset=0, scale_factor=None, units=None, calendar=None, **kwargs) -> np.ndarray:
    """

    """
    data = np.frombuffer(data, dtype=dtype_encoded)

    if isinstance(calendar, str):
        data = decode_datetime(data, units, calendar)

    elif isinstance(scale_factor, (int, float, np.number)):
        data = data.astype(dtype_decoded)

        if isinstance(missing_value, (int, np.number)):
            if isinstance(data, np.number):
                if data == missing_value:
                    data = np.nan
            else:
                data[data == missing_value] = np.nan

        data = (data * scale_factor) + add_offset

    elif (data.dtype.name != dtype_decoded) or (data.dtype.name == 'object'):
        data = data.astype(dtype_decoded)

    return data


# def get_encoding_data_from_attrs(attrs):
#     """

#     """
#     encoding = {}
#     for f, v in attrs.items():
#         if f in enc_fields:
#             if isinstance(v, bytes):
#                 encoding[f] = v.decode()
#             elif isinstance(v, np.ndarray):
#                 if len(v) == 1:
#                     encoding[f] = v[0]
#                 else:
#                     raise ValueError('encoding is an ndarray with len > 1.')
#             else:
#                 encoding[f] = v

#     return encoding


# def get_encoding_data_from_xr(data):
#     """

#     """
#     attrs = {f: v for f, v in data.attrs.items() if (f in enc_fields) and (f not in ignore_attrs)}
#     encoding = {f: v for f, v in data.encoding.items() if (f in enc_fields) and (f not in ignore_attrs)}

#     attrs.update(encoding)

#     return attrs


# def process_encoding(encoding, dtype):
#     """

#     """
#     if (dtype.name == 'object') or ('str' in dtype.name):
#         # encoding['dtype'] = h5py.string_dtype()
#         encoding['dtype'] = 'object'
#     elif ('datetime64' in dtype.name): # which means it's an xr.DataArray
#         encoding['dtype'] = 'int64'
#         encoding['calendar'] = 'gregorian'
#         encoding['units'] = 'seconds since 1970-01-01 00:00:00'
#         encoding['missing_value'] = missing_value_dict['int64']
#         encoding['_FillValue'] = encoding['missing_value']

#     elif 'calendar' in encoding: # Which means it's not an xr.DataArray
#         encoding['dtype'] = 'int64'
#         if 'units' not in encoding:
#             encoding['units'] = 'seconds since 1970-01-01 00:00:00'
#         encoding['missing_value'] = missing_value_dict['int64']
#         encoding['_FillValue'] = encoding['missing_value']

#     if 'dtype' not in encoding:
#         if np.issubdtype(dtype, np.floating):
#             # scale, offset = compute_scale_and_offset(min_value, max_value, n)
#             raise ValueError('float dtypes must have encoding data to encode to int.')
#         encoding['dtype'] = dtype.name
#     elif not isinstance(encoding['dtype'], str):
#         encoding['dtype'] = encoding['dtype'].name

#     if 'scale_factor' in encoding:
#         if not isinstance(encoding['scale_factor'], (int, float, np.number)):
#             raise TypeError('scale_factor must be an int or float.')

#         if not 'int' in encoding['dtype']:
#             raise ValueError('If scale_factor is assigned, then the dtype must be an integer.')
#         if 'add_offset' not in encoding:
#             encoding['add_offset'] = 0
#         elif not isinstance(encoding['add_offset'], (int, float, np.number)):
#             raise ValueError('add_offset must be a number.')

#     if 'int' in encoding['dtype']:
#         if ('_FillValue' in encoding) and ('missing_value' not in encoding):
#             encoding['missing_value'] = encoding['_FillValue']
#         if ('_FillValue' not in encoding) and ('missing_value' in encoding):
#             encoding['_FillValue'] = encoding['missing_value']

#         # if 'missing_value' not in encoding:
#         #     encoding['missing_value'] = missing_value_dict[encoding['dtype'].name]
#         #     encoding['_FillValue'] = encoding['missing_value']

#     return encoding


def assign_dtype_decoded(encoding):
    """

    """
    if encoding['dtype'] == 'object':
        encoding['dtype_decoded'] = encoding['dtype']
    elif ('calendar' in encoding) and ('units' in encoding):
        encoding['dtype_decoded'] = 'datetime64[s]'

    if 'scale_factor' in encoding:

        # if isinstance(encoding['scale_factor'], (int, np.integer)):
        #     encoding['dtype_decoded'] = np.dtype('float32')
        if np.dtype(encoding['dtype']).itemsize > 2:
            encoding['dtype_decoded'] = 'float64'
        else:
            encoding['dtype_decoded'] = 'float32'

    if 'dtype_decoded' not in encoding:
        encoding['dtype_decoded'] = encoding['dtype']

    return encoding


# def get_encodings(files, group=None):
#     """
#     I should add checking across the files for conflicts at some point.
#     """
#     # file_encs = {}
#     encs = {}
#     for i, file1 in enumerate(files):
#         file = open_file(file1, group)
#         # file_encs[i] = {}
#         if isinstance(file, xr.Dataset):
#             ds_list = list(file.variables)
#         else:
#             ds_list = list(file.keys())

#         for name in ds_list:
#             data = file[name]
#             if isinstance(data, xr.DataArray):
#                 encoding = get_encoding_data_from_xr(data)
#             else:
#                 encoding = get_encoding_data_from_attrs(data.attrs)
#             enc = process_encoding(encoding, data.dtype)
#             enc = assign_dtype_decoded(enc)
#             # file_encs[i].update({name: enc})

#             if name in encs:
#                 # TODO: equality check for existing enc dists
#                 new_enc = encs[name]
#                 shared_items = [True for k in new_enc if (k in enc) and (new_enc[k] == enc[k])]
#                 if not all(shared_items):
#                     raise ValueError(f'Not all files have the same encoding for {name}.')
#                 encs[name].update(enc)
#             else:
#                 encs[name] = enc

#         # for name, enc in encs.items():
#         #     enc = assign_dtype_decoded(enc)
#         #     encs[name] = enc

#         file.close()

#     return encs


# def get_attrs(files, group):
#     """

#     """
#     # file_attrs = {}
#     global_attrs = {}
#     attrs = {}
#     for i, file1 in enumerate(files):
#         with open_file(file1, group) as file:
#             global_attrs.update(dict(file.attrs))

#             if isinstance(file, xr.Dataset):
#                 vars_list = list(file.variables)
#             else:
#                 vars_list = [name for name in file]

#             for name in vars_list:
#                 attr = {f: v for f, v in file[name].attrs.items() if (f not in enc_fields) and (f not in ignore_attrs)}

#                 if name in attrs:
#                     attrs[name].update(attr)
#                 else:
#                     attrs[name] = attr

#     return attrs, global_attrs


# def is_scale(dataset):
#     """

#     """
#     check = h5py.h5ds.is_scale(dataset._id)

#     return check


# def is_regular_index(arr_index):
#     """

#     """
#     reg_bool = (np.sum(np.diff(arr_index)) == (len(arr_index) - 1)) or len(arr_index) == 1

#     return reg_bool


# def open_file(path, group=None):
#     """

#     """
#     if isinstance(path, (str, pathlib.Path, io.IOBase)):
#         try:
#             f = h5py.File(path, 'r')
#         except:
#             f = xr.open_dataset(path, cache=False)
#     elif isinstance(path, (h5py.File, xr.Dataset)):
#         f = path
#     elif isinstance(path, bytes):
#         try:
#             f = h5py.File(io.BytesIO(path), 'r')
#         except:
#             f = xr.open_dataset(io.BytesIO(path), cache=False)
#     else:
#         raise TypeError('path must be a str/pathlib path to an HDF5 file, an h5py.File, a bytes object of an HDF5 file, or an xarray Dataset.')

#     if isinstance(group, str) and not isinstance(f, xr.Dataset):
#         f = f[group]

#     return f


def extend_coords(files, encodings, group):
    """

    """
    coords_dict = {}

    for file1 in files:
        with open_file(file1, group) as file:
            if isinstance(file, xr.Dataset):
                ds_list = list(file.coords)
            else:
                ds_list = [ds_name for ds_name in file.keys() if is_scale(file[ds_name])]

            for ds_name in ds_list:
                ds = file[ds_name]

                if isinstance(file, xr.Dataset):
                    data = encode_data(ds.values, **encodings[ds_name])
                else:
                    if ds.dtype.name == 'object':
                        data = ds[:].astype(str).astype(h5py.string_dtype())
                    else:
                        data = ds[:]

                # Check for nan values in numeric types
                dtype = data.dtype
                if np.issubdtype(dtype, np.integer):
                    nan_value = missing_value_dict[dtype.name]
                    if nan_value in data:
                        raise ValueError(f'{ds_name} has nan values. Floats and integers coordinates cannot have nan values. Check the encoding values if the original values are floats.')

                if ds_name in coords_dict:
                    coords_dict[ds_name] = np.union1d(coords_dict[ds_name], data)
                else:
                    coords_dict[ds_name] = data

    return coords_dict


def index_variables(files, coords_dict, encodings, group):
    """

    """
    vars_dict = {}
    is_regular_dict = {}

    for i, file1 in enumerate(files):
        with open_file(file1, group) as file:
            # if i == 77:
            #     break

            if isinstance(file, xr.Dataset):
                ds_list = list(file.data_vars)
            else:
                ds_list = [ds_name for ds_name in file.keys() if not is_scale(file[ds_name])]

            _ = [is_regular_dict.update({ds_name: True}) for ds_name in ds_list if ds_name not in is_regular_dict]

            for ds_name in ds_list:
                ds = file[ds_name]

                var_enc = encodings[ds_name]

                dims = []
                global_index = {}
                local_index = {}
                remove_ds = False

                for dim in ds.dims:
                    if isinstance(ds, xr.DataArray):
                        dim_name = dim
                        dim_data = encode_data(ds[dim_name].values, **encodings[dim_name])
                    else:
                        dim_name = dim[0].name.split('/')[-1]
                        if dim[0].dtype.name == 'object':
                            dim_data = dim[0][:].astype(str).astype(h5py.string_dtype())
                        else:
                            dim_data = dim[0][:]

                    dims.append(dim_name)

                    # global_arr_index = np.searchsorted(coords_dict[dim_name], dim_data)
                    # local_arr_index = np.isin(dim_data, coords_dict[dim_name], assume_unique=True).nonzero()[0]
                    values, global_arr_index, local_arr_index = np.intersect1d(coords_dict[dim_name], dim_data, assume_unique=True, return_indices=True)

                    if len(global_arr_index) > 0:

                        global_index[dim_name] = global_arr_index
                        local_index[dim_name] = local_arr_index

                        if is_regular_dict[ds_name]:
                            if (not is_regular_index(global_arr_index)) or (not is_regular_index(local_arr_index)):
                                is_regular_dict[ds_name] = False
                    else:
                        remove_ds = True
                        break

                if remove_ds:
                    if ds_name in vars_dict:
                        if i in vars_dict[ds_name]['data']:
                            del vars_dict[ds_name]['data'][i]

                else:
                    dict1 = {'dims_order': tuple(i for i in range(len(dims))), 'global_index': global_index, 'local_index': local_index}

                    if ds_name in vars_dict:
                        if not np.in1d(vars_dict[ds_name]['dims'], dims).all():
                            raise ValueError('dims are not consistant between the same named dataset: ' + ds_name)
                        # if vars_dict[ds_name]['dtype'] != ds.dtype:
                        #     raise ValueError('dtypes are not consistant between the same named dataset: ' + ds_name)

                        dims_order = [vars_dict[ds_name]['dims'].index(dim) for dim in dims]
                        dict1['dims_order'] = tuple(dims_order)

                        vars_dict[ds_name]['data'][i] = dict1
                    else:
                        shape = tuple([coords_dict[dim_name].shape[0] for dim_name in dims])

                        if 'missing_value' in var_enc:
                            fillvalue = var_enc['missing_value']
                        else:
                            fillvalue = None

                        vars_dict[ds_name] = {'data': {i: dict1}, 'dims': tuple(dims), 'shape': shape, 'dtype': var_enc['dtype'], 'fillvalue': fillvalue, 'dtype_decoded': var_enc['dtype_decoded']}

    return vars_dict, is_regular_dict


def filter_coords(coords_dict, selection, encodings):
    """

    """
    for coord, sel in selection.items():
        if coord not in coords_dict:
            raise ValueError(coord + ' one of the coordinates.')

        coord_data = decode_data(coords_dict[coord], **encodings[coord])

        if isinstance(sel, slice):
            if 'datetime64' in coord_data.dtype.name:
                # if not isinstance(sel.start, (str, np.datetime64)):
                #     raise TypeError('Input for datetime selection should be either a datetime string or np.datetime64.')

                if sel.start is not None:
                    start = np.datetime64(sel.start, 's')
                else:
                    start = np.datetime64(coord_data[0] - 1, 's')

                if sel.stop is not None:
                    end = np.datetime64(sel.stop, 's')
                else:
                    end = np.datetime64(coord_data[-1] + 1, 's')

                bool_index = (start <= coord_data) & (coord_data < end)
            else:
                bool_index = (sel.start <= coord_data) & (coord_data < sel.stop)

        else:
            if isinstance(sel, (int, float)):
                sel = [sel]

            try:
                sel1 = np.array(sel)
            except:
                raise TypeError('selection input could not be coerced to an ndarray.')

            if sel1.dtype.name == 'bool':
                if sel1.shape[0] != coord_data.shape[0]:
                    raise ValueError('The boolean array does not have the same length as the coord array.')
                bool_index = sel1
            else:
                bool_index = np.in1d(coord_data, sel1)

        new_coord_data = encode_data(coord_data[bool_index], **encodings[coord])

        coords_dict[coord] = new_coord_data


def guess_chunk(shape, maxshape, dtype, chunk_max=2**21):
    """ Guess an appropriate chunk layout for a dataset, given its shape and
    the size of each element in bytes.  Will allocate chunks only as large
    as MAX_SIZE.  Chunks are generally close to some power-of-2 fraction of
    each axis, slightly favoring bigger values for the last index.
    Undocumented and subject to change without warning.
    """
    ndims = len(shape)

    if ndims > 0:

        # For unlimited dimensions we have to guess 1024
        shape1 = []
        for i, x in enumerate(maxshape):
            if x is None:
                if shape[i] > 1024:
                    shape1.append(shape[i])
                else:
                    shape1.append(1024)
            else:
                shape1.append(x)

        shape = tuple(shape1)

        # ndims = len(shape)
        # if ndims == 0:
        #     raise ValueError("Chunks not allowed for scalar datasets.")

        chunks = np.array(shape, dtype='=f8')
        if not np.all(np.isfinite(chunks)):
            raise ValueError("Illegal value in chunk tuple")

        # Determine the optimal chunk size in bytes using a PyTables expression.
        # This is kept as a float.
        typesize = np.dtype(dtype).itemsize
        # dset_size = np.prod(chunks)*typesize
        # target_size = CHUNK_BASE * (2**np.log10(dset_size/(1024.*1024)))

        # if target_size > CHUNK_MAX:
        #     target_size = CHUNK_MAX
        # elif target_size < CHUNK_MIN:
        #     target_size = CHUNK_MIN

        target_size = chunk_max

        idx = 0
        while True:
            # Repeatedly loop over the axes, dividing them by 2.  Stop when:
            # 1a. We're smaller than the target chunk size, OR
            # 1b. We're within 50% of the target chunk size, AND
            #  2. The chunk is smaller than the maximum chunk size

            chunk_bytes = math.prod(chunks)*typesize

            if (chunk_bytes < target_size or \
             abs(chunk_bytes - target_size)/target_size < 0.5):
                break

            if math.prod(chunks) == 1:
                break

            chunks[idx%ndims] = math.ceil(chunks[idx%ndims] / 2.0)
            idx += 1

        return tuple(int(x) for x in chunks)
    else:
        return None


def guess_chunk_time(shape, maxshape, dtype, time_index, chunk_max=3*2**20):
    """ Guess an appropriate chunk layout for a dataset, given its shape and
    the size of each element in bytes.  Will allocate chunks only as large
    as MAX_SIZE.  Chunks are generally close to some power-of-2 fraction of
    each axis, slightly favoring bigger values for the last index.
    Undocumented and subject to change without warning.
    """
    ndims = len(shape)

    if ndims > 0:

        # For unlimited dimensions we have to guess 1024
        shape1 = []
        for i, x in enumerate(maxshape):
            if x is None:
                if shape[i] > 1024:
                    shape1.append(shape[i])
                else:
                    shape1.append(1024)
            else:
                shape1.append(x)

        shape = tuple(shape1)

        chunks = np.array(shape, dtype='=f8')
        if not np.all(np.isfinite(chunks)):
            raise ValueError("Illegal value in chunk tuple")

        # Determine the optimal chunk size in bytes using a PyTables expression.
        # This is kept as a float.
        typesize = np.dtype(dtype).itemsize

        target_size = chunk_max

        while True:
            # Repeatedly loop over the axes, dividing them by 2.  Stop when:
            # 1a. We're smaller than the target chunk size, OR
            # 1b. We're within 50% of the target chunk size, AND
            #  2. The chunk is smaller than the maximum chunk size

            chunk_bytes = product(chunks)*typesize

            if (chunk_bytes < target_size or \
             abs(chunk_bytes - target_size)/target_size < 0.5):
                break

            if chunks[time_index] == 1:
                break

            chunks[time_index] = np.ceil(chunks[time_index] / 2.0)

        return tuple(int(x) for x in chunks)
    else:
        return None


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
            [1, 4, 7],
            [1, 5, 6],
            [1, 5, 7],
            [2, 4, 6],
            [2, 4, 7],
            [2, 5, 6],
            [2, 5, 7],
            [3, 4, 6],
            [3, 4, 7],
            [3, 5, 6],
            [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]

    return out


def get_compressor(name: str = None):
    """

    """
    if name is None:
        compressor = {}
    elif name.lower() == 'none':
        compressor = {}
    elif name.lower() == 'gzip':
        compressor = {'compression': name}
    elif name.lower() == 'lzf':
        compressor = {'compression': name}
    elif name.lower() == 'zstd':
        compressor = hdf5plugin.Zstd(1)
    elif name.lower() == 'lz4':
        compressor = hdf5plugin.LZ4()
    else:
        raise ValueError('name must be one of gzip, lzf, zstd, lz4, or None.')

    return compressor


def fill_ds_by_chunks(ds, files, ds_vars, var_name, group, encodings):
    """

    """
    dims = ds_vars['dims']
    if ds_vars['fillvalue'] is None:
        fillvalue = -99
    else:
        fillvalue = ds_vars['fillvalue']

    for chunk in ds.iter_chunks():
        chunk_size1 = tuple(c.stop - c.start for c in chunk)
        chunk_arr = np.full(chunk_size1, fill_value=fillvalue, dtype=ds_vars['dtype'], order='C')
        for i_file, data in ds_vars['data'].items():
            # if i_file == 9:
            #     break
            g_bool_index = [(chunk[i].start <= data['global_index'][dim]) & (data['global_index'][dim] < chunk[i].stop) for i, dim in enumerate(dims)]
            bool1 = all([a.any() for a in g_bool_index])
            if bool1:
                l_slices = {}
                for i, dim in enumerate(dims):
                    w = g_bool_index[i]
                    l_index = data['local_index'][dim][w]
                    if is_regular_index(l_index):
                        l_slices[dim] = slice(l_index[0], l_index[-1] + 1, None)
                    else:
                        l_slices[dim] = l_index

                if tuple(range(len(dims))) == data['dims_order']:
                    transpose_order = None
                else:
                    transpose_order = tuple(data['dims_order'].index(i) for i in range(len(data['dims_order'])))

                with open_file(files[i_file], group) as f:
                    if isinstance(f, xr.Dataset):
                        l_data = encode_data(f[var_name][tuple(l_slices.values())].values, **encodings[var_name])
                    else:
                        l_data = f[var_name][tuple(l_slices.values())]

                    if transpose_order is not None:
                        l_data = l_data.transpose(transpose_order)

                    g_chunk_index = []
                    for i, dim in enumerate(dims):
                        s1 = data['global_index'][dim][g_bool_index[i]] - chunk[i].start
                        if is_regular_index(s1):
                            s1 = slice(s1[0], s1[-1] + 1, None)
                        g_chunk_index.append(s1)
                    chunk_arr[tuple(g_chunk_index)] = l_data

        ## Save chunk to new dataset
        ds[chunk] = chunk_arr


def fill_ds_by_files(ds, files, ds_vars, var_name, group, encodings):
    """
    Currently the implementation is simple. It loads one entire input file into the ds. It would be nice to chunk the file before loading to handle very large input files.
    """
    dims = ds_vars['dims']
    dtype = ds_vars['dtype']

    for i_file, data in ds_vars['data'].items():
        dims_order = data['dims_order']
        g_index_start = tuple(data['global_index'][dim][0] for dim in dims)

        if tuple(range(len(dims))) == data['dims_order']:
            transpose_order = None
        else:
            transpose_order = tuple(dims_order.index(i) for i in range(len(dims_order)))
            g_index_start = tuple(g_index_start[i] for i in dims_order)

        file_shape = tuple(len(arr) for dim, arr in data['local_index'].items())
        chunk_size = guess_chunk(file_shape, file_shape, dtype, 2**27)
        chunk_iter = ChunkIterator(chunk_size, file_shape)

        with open_file(files[i_file], group) as f:
            for chunk in chunk_iter:
                # g_chunk_slices = []
                # l_slices = []
                # for dim in dims:
                #     g_index = data['global_index'][dim]
                #     g_chunk_slices.append(slice(g_index[0], g_index[-1] + 1, None))

                #     l_index = data['local_index'][dim]
                #     l_slices.append(slice(l_index[0], l_index[-1] + 1, None))

                if isinstance(f, xr.Dataset):
                    l_data = encode_data(f[var_name][chunk].values, **encodings[var_name])
                else:
                    l_data = f[var_name][chunk]

                if transpose_order is not None:
                    l_data = l_data.transpose(transpose_order)

                g_chunk_slices = tuple(slice(g_index_start[i] + s.start, g_index_start[i] + s.stop, 1) for i, s in enumerate(chunk))

                ds[g_chunk_slices] = l_data


def get_dtype_shape(data=None, dtype=None, shape=None):
    """

    """
    if data is None:
        if (shape is None) or (dtype is None):
            raise ValueError('shape and dtype must be passed or data must be passed.')
        if not isinstance(dtype, str):
            dtype = dtype.name
    else:
        shape = data.shape
        dtype = data.dtype.name

    return dtype, shape


















































































