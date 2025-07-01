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
import copy
# import xarray as xr
# from time import time
# from datetime import datetime
import cftime
import math
from typing import Set, Optional, Dict, Tuple, List, Union, Any
# import zstandard as zstd
# import lz4

# import dateutil.parser as dparser
# import numcodecs
# import hdf5plugin

import data_models, rechunker, indexers

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

var_chunk_key_str = '{var_name}!{dims}'

attrs_key_str = '_{var_name}.attrs'

name_indent = 4
value_indent = 20
var_name_regex = "^[a-zA-Z][a-zA-Z0-9_]*$"
var_name_pattern = re.compile(var_name_regex)

default_params = {'lon': {'name': 'longitude', 'dtype_encoded': 'int32', 'fillvalue': -2147483648, 'scale_factor': 0.0000001, 'dtype_decoded': 'float32'},
                 'lat': {'name': 'latitude', 'dtype_encoded': 'int32', 'fillvalue': -2147483648, 'scale_factor': 0.0000001, 'dtype_decoded': 'float32'},
                 'height': {'name': 'height', 'dtype_encoded': 'int32', 'fillvalue': -2147483648, 'scale_factor': 0.001, 'dtype_decoded': 'float32'},
                 'altitude': {'name': 'altitude', 'dtype_encoded': 'int32', 'fillvalue': -2147483648, 'scale_factor': 0.001, 'dtype_decoded': 'float32'},
                 'time': {'name': 'time', 'dtype_encoded': 'int64', 'dtype_decoded': 'datetime64[s]'},
                 'modified_date': {'name': 'modified_date', 'dtype_encoded': 'datetime64[us]', 'dtype_decoded': 'datetime64[us]'},
                 'band': {'name': 'band', 'dtype_decoded': 'uint8', 'dtype_encoded': 'uint8', 'fillvalue': 0},
                 # 'chunk_day': {'dtype_encoded': 'int32'},
                 # 'chunk_date': {'fillvalue': -99999999, 'units': "days since 1970-01-01 00:00:00"},
                 'censor_code': {'name': 'censor_code', 'dtype_decoded': 'uint8', 'dtype_encoded': 'uint8', 'fillvalue': 0},
                 # 'bore_top_of_screen': {'dtype_encoded': 'int16', 'fillvalue': 9999, 'scale_factor': 0.1},
                 # 'bore_bottom_of_screen': {'dtype_encoded': 'int16', 'fillvalue': 9999, 'scale_factor': 0.1},
                 # 'bore_depth': {'dtype_encoded': 'int16', 'fillvalue': -9999, 'scale_factor': 0.1},
                 # 'reference_level': {'dtype_encoded': 'int16', 'fillvalue': -9999, 'scale_factor': 1},
                 }

# base_attrs = {'station_id': {'cf_role': "timeseries_id", 'description': 'The unique ID associated with the geometry for a single result.'},
#               'lat': {'standard_name': "latitude", 'units': "degrees_north"},
#               'lon': {'standard_name': "longitude", 'units': "degrees_east"},
#               'altitude': {'standard_name': 'surface_altitude', 'long_name': 'height above the geoid to the lower boundary of the atmosphere', 'units': 'm'},
#               'geometry': {'long_name': 'The hexadecimal encoding of the Well-Known Binary (WKB) geometry', 'crs_EPSG': 4326},
#               'station_geometry': {'long_name': 'The hexadecimal encoding of the Well-Known Binary (WKB) station geometry', 'crs_EPSG': 4326},
#               'height': {'standard_name': 'height', 'long_name': 'vertical distance above the surface', 'units': 'm', 'positive': 'up'},
#               'time': {'standard_name': 'time', 'long_name': 'start_time'}, 'name': {'long_name': 'station name'},
#               'ref': {'long_name': 'station reference id given by the owner'}, 'modified_date': {'long_name': 'last modified date'},
#               'band': {'long_name': 'band number'},
#               'chunk_date': {'long_name': 'chunking date'},
#               'chunk_day': {'long_name': 'chunking day', 'description': 'The chunk day is the number of days after 1970-01-01. Can be negative for days before 1970-01-01 with a minimum of -106751, which is 1677-09-22 (minimum possible date). The maximum value is 106751.'},
#               'chunk_hash': {'long_name': 'chunk hash', 'description': 'The unique hash of the results parameter for comparison purposes.'},
#               'chunk_id': {'long_name': 'chunk id', 'description': 'The unique id of the results chunk associated with the specific station.'},
#               'censor_code': {'long_name': 'data censor code', 'standard_name': 'status_flag', 'flag_values': '0 1 2 3 4 5', 'flag_meanings': 'greater_than less_than not_censored non-detect present_but_not_quantified unknown'},
#               'bore_top_of_screen': {'long_name': 'bore top of screen', 'description': 'The depth to the top of the screen from the reference level.', 'units': 'm', 'positive': 'down'},
#               'bore_bottom_of_screen': {'long_name': 'bore bottom of screen', 'description': 'The depth to the bottom of the screen from the reference level.', 'units': 'm', 'positive': 'down'},
#               'bore_depth': {'long_name': 'bore depth', 'description': 'The depth of the bore from the reference level.', 'units': 'm', 'positive': 'down'},
#               'alt_name': {'long_name': 'Alternative name', 'description': 'The alternative name for the station'},
#               'reference_level': {'long_name': 'The bore reference level', 'description': 'The bore reference level for measurements.', 'units': 'mm', 'positive': 'up'}
#               }

default_attrs = dict(
    lat={
        'long_name': 'latitude',
        'units': 'degrees_north',
        'standard_name': 'latitude',
        'axis': 'Y',
        },
    lon={
        'long_name': 'longitude',
        'units': 'degrees_east',
        'standard_name': 'longitude',
        'axis': 'X',
        },
    height={
        'long_name': 'height',
        'units': 'm',
        'standard_name': 'height',
        'positive': 'up',
        'axis': 'Z',
        },
    altitude={
        'long_name': 'altitude',
        'units': 'm',
        'standard_name': 'altitude',
        'positive': 'up',
        'axis': 'Z',
        },
    time={
        'long_name': 'time',
        'units': 'seconds since 1970-01-01 00:00:00',
        'standard_name': 'time',
        'calendar': 'proleptic_gregorian',
        'axis': 'T',
        },
    )

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


# def make_compressor(


def min_max_dates_per_bit_len(n_bits):
    """

    """
    n_bits_options = (16, 32, 64)
    if n_bits not in n_bits_options:
        raise ValueError(f'n_bits must be one of {n_bits_options}')

    freq_codes = ('D', 'h', 'm', 's')
    res_dict = {}
    for code in freq_codes:
        int_len = int(2**n_bits*.5)
        min_date = np.datetime64(-int_len + 1, code).astype(str)
        max_date = np.datetime64(int_len - 1, code).astype(str)
        res_dict[code] = (min_date, max_date)

    return res_dict


def dataset_finalizer(blt_file, sys_meta):
    """

    """
    old_meta_data = blt_file.get_metadata()
    if old_meta_data is not None:
        old_meta = msgspec.convert(old_meta_data, data_models.SysMeta)
        if old_meta != sys_meta:
            blt_file.set_metadata(msgspec.to_builtins(sys_meta))
    else:
        blt_file.set_metadata(msgspec.to_builtins(sys_meta))

    blt_file.close()


def attrs_finalizer(blt_file, attrs, var_name, writeable):
    """

    """
    if attrs and writeable:
        key = attrs_key_str.format(var_name=var_name)
        old_attrs = blt_file.get(key)
        if old_attrs is not None:
            old_attrs = msgspec.json.decode(old_attrs)
            if old_attrs != attrs:
                blt_file.set(key, msgspec.json.encode(attrs))
        else:
            blt_file.set(key, msgspec.json.encode(attrs))


def compute_scale_and_offset(min_value: Union[int, float, np.number], max_value: Union[int, float, np.number], dtype: Union[np.dtype, str]):
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
    if isinstance(dtype, str):
        dtype = np.dtype(dtype)
    bits = dtype.itemsize * 8
    data_range = max_value - min_value
    target_range = 2**bits - 2
    slope = data_range / target_range

    if bits < 64:
        target_min = -(2**(bits - 1) - 1)
    else:
        target_min = -(2**(bits - 1) - 1000)
    # if bits < 64:
    #     target_range = 2**bits - 2
    #     target_min = -(2**(bits - 1) - 1)
    #     slope = data_range / target_range
    # else:
    #     data_power = int(math.log10(data_range))
    #     target_range = 2**bits
    #     target_power = int(math.log10(target_range))
    #     target_min = -10**(target_power - 1)
    #     slope = 10**-(target_power - data_power)

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
            if var_name_pattern.match(var_name):
                return True
    return False


def coord_data_step_check(data: np.ndarray, dtype_decoded: np.dtype, step: int | float | bool = False):
    """

    """
    # diff = np.diff(data)
    if isinstance(step, bool):
        diff = np.diff(data)
        if dtype_decoded == 'f':
            step = float(np.round(diff[0], 5))
            if not np.allclose(step, diff):
                raise ValueError('step is set to True, but the data does not seem to be regular.')
            # data = np.linspace(data[0], data[-1], len(diff) + 1, dtype=dtype_decoded)
        else:
            step = int(diff[0])

            if not np.all(np.equal(step, diff)):
                raise ValueError('step is set to True, but the data does not seem to be regular.')
    elif isinstance(step, (float, np.floating)):
        if step <= 0:
            raise ValueError('step must be greater than 0.')
        # if not np.allclose(step, diff):
        #     raise ValueError('step does not seem to be the interval of the data.')
        step = float(round(step, 5))
        # num = round((data[-1] - data[0])/step, 5)
        # if not num.is_integer():
        #     raise ValueError('The step is not a multiple of the difference between the first and last values of the data.')

        # data = np.linspace(data[0], data[-1], int(num) + 1, dtype=dtype_decoded)
    elif isinstance(step, (int, np.integer)):
        if step <= 0:
            raise ValueError('step must be greater than 0.')
        # if not np.all(np.equal(step, diff)):
        #     raise ValueError('step is set to True, but the data does not seem to be regular.')
        step = int(step)

        # data = np.linspace(data[0], data[-1], int(num) + 1, dtype=dtype_decoded)
    else:
        raise TypeError('step must be a bool, int, or float. The int or float must be greater than 0.')

    num = round((data[-1] - data[0])/step, 5)
    if not num.is_integer():
        raise ValueError('The step is not a multiple of the difference between the first and last values of the data.')

    return step, int(num)


def init_coord_data_checks(data: np.ndarray, step: int | float | bool, dtype_decoded, shape):
    """

    """
    # dtype_decoded = data.dtype
    # shape = data.shape

    if len(shape) > 1:
        raise ValueError('Coordinates must be 1D.')

    if len(np.unique(data)) < shape[0]:
        raise ValueError('The data for coords must be unique.')

    if dtype_decoded.kind in ('f', 'u', 'i', 'M'):
        data.sort()
        if step:
            step, num = coord_data_step_check(data, dtype_decoded, step)
            # data = np.linspace(data[0], data[-1], num + 1, dtype=dtype_decoded)
        else:
            step = None
    else:
        step = None

    return step


def append_coord_data_checks(new_data: np.ndarray, source_data: np.ndarray, source_dtype_decoded: np.dtype = None, source_step: int | float | None = None):
    """

    """
    # new_shape = new_data.shape
    new_dtype_decoded = new_data.dtype

    if source_dtype_decoded != new_dtype_decoded:
        raise TypeError('The data dtype does not match the originally assigned dtype.')

    # print(source_data)

    if source_data.size > 0:
        if source_dtype_decoded.kind != 'U':
            last = source_data[-1]

            if not np.all(last < new_data):
                raise ValueError('Appending requires that all values are greater than the existing values.')

            new_data.sort()
            if source_step:
                new_step, new_num = coord_data_step_check(new_data, source_dtype_decoded, source_step)

                new_data = np.linspace(source_data[0], new_data[-1], len(source_data) + new_num + 1, dtype=source_dtype_decoded)
            else:
                new_data = np.append(source_data, new_data)

        else:
            s1 = set(source_data)
            s1.update(set(new_data))
            if len(s1) != (len(source_data) + len(new_data)):
                raise ValueError('The data for coords must be unique.')

            new_data = np.append(source_data, new_data)

    else:
        _ = init_coord_data_checks(new_data, source_step, source_dtype_decoded, new_data.shape)

    return new_data


def prepend_coord_data_checks(new_data: np.ndarray, source_data: np.ndarray, source_dtype_decoded: np.dtype = None, source_step: int | float | None = None):
    """

    """
    # new_shape = new_data.shape
    new_dtype_decoded = new_data.dtype

    if source_dtype_decoded != new_dtype_decoded:
        raise TypeError('The data dtype does not match the originally assigned dtype.')

    if source_data.size > 0:
        if source_dtype_decoded.kind != 'U':
            first = source_data[0]

            if not np.all(first > new_data):
                raise ValueError('Prepending requires that all values are less than the existing values.')

            new_data.sort()
            if source_step:
                new_step, new_num = coord_data_step_check(new_data, source_dtype_decoded, source_step)

                new_data = np.linspace(new_step[0], source_data[-1], len(source_data) + new_num + 1, dtype=source_dtype_decoded)
            else:
                new_data = np.append(new_data, source_data)
        else:
            s1 = set(source_data)
            s1.update(set(new_data))
            if len(s1) != (len(source_data) + len(new_data)):
                raise ValueError('The data for coords must be unique.')

            new_data = np.append(new_data, source_data)

    else:
        new_data, new_step = init_coord_data_checks(new_data, source_step, source_dtype_decoded, new_data.shape)

    return new_data


# def coord_data_checks(data: np.ndarray, step: int | float | bool = False, source_data: np.ndarray = None, source_dtype_decoded: np.dtype = None, source_step: int | float | None = None):
#     """

#     """
#     shape = data.shape
#     dtype_decoded = data.dtype

#     if dtype_decoded is not None:
#         if source_dtype_decoded != dtype_decoded:
#             raise TypeError('The data dtype does not match the originally assigned dtype.')

#     if source_data:
#         s1 = set(source_data)
#         s1.update(set(data))
#         if len(s1) != (len(source_data) + len(data)):
#             raise ValueError('The data for coords must be unique.')
#     elif len(np.unique(data)) < shape[0]:
#         raise ValueError('The data for coords must be unique.')

#     if dtype_decoded.kind in ('f', 'u', 'i', 'M'):
#         data.sort()
#         if step:
#             diff = np.diff(data)
#             if isinstance(step, bool):
#                 if dtype_decoded == 'f':
#                     step = float(np.round(diff[0], 5))
#                     if not np.allclose(step, diff):
#                         raise ValueError('step is set to True, but the data does not seem to be regular.')
#                     # data = np.linspace(data[0], data[-1], len(diff) + 1, dtype=dtype_decoded)
#                 else:
#                     step = int(diff[0])

#                     if not np.all(np.equal(step, diff)):
#                         raise ValueError('step is set to True, but the data does not seem to be regular.')
#             elif isinstance(step, (float, np.floating)):
#                 if step <= 0:
#                     raise ValueError('step must be greater than 0.')
#                 # if not np.allclose(step, diff):
#                 #     raise ValueError('step does not seem to be the interval of the data.')
#                 step = float(round(step, 5))
#                 num = round((data[-1] - data[0])/step, 5)
#                 if not num.is_integer():
#                     raise ValueError('The step is not a multiple of the dirrefernce between the first and last values of the data.')

#                 data = np.linspace(data[0], data[-1], int(num) + 1, dtype=dtype_decoded)
#             elif isinstance(step, (int, np.integer)):
#                 if step <= 0:
#                     raise ValueError('step must be greater than 0.')
#                 # if not np.all(np.equal(step, diff)):
#                 #     raise ValueError('step is set to True, but the data does not seem to be regular.')
#                 step = int(step)
#                 num = round((data[-1] - data[0])/step, 5)
#                 if not num.is_integer():
#                     raise ValueError('The step is not a multiple of the dirrefernce between the first and last values of the data.')

#                 data = np.linspace(data[0], data[-1], int(num) + 1, dtype=dtype_decoded)
#             else:
#                 raise TypeError('step must be a bool, int, or float. The int or float must be greater than 0.')
#         else:
#             step = None
#     else:
#         step = None

#     return data, step


def parse_dtypes(dtype_decoded, dtype_encoded):
    """

    """
    dtype_decoded = np.dtype(dtype_decoded)

    # if dtype_decoded.kind == 'M':
    #     dtype_encoded = np.dtype('int64')

    if isinstance(dtype_encoded, str):
        dtype_encoded = np.dtype(dtype_encoded)

    elif not isinstance(dtype_encoded, np.dtype):
        dtype_encoded = dtype_decoded

    return dtype_decoded, dtype_encoded


def parse_dtype_names(dtype_decoded, dtype_encoded):
    """

    """
    if dtype_encoded.kind == 'U':
        dtype_decoded_name = dtype_decoded.descr[0][1]
        dtype_encoded_name = dtype_encoded.descr[0][1]
    else:
        dtype_decoded_name = dtype_decoded.name
        dtype_encoded_name = dtype_encoded.name

    return dtype_decoded_name, dtype_encoded_name


def parse_fillvalue(fillvalue, dtype_encoded):
    """

    """
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
            fillvalue = None
        elif kind == 'U':
            fillvalue = ''
        elif kind == 'i':
            fillvalue = fillvalue_dict[dtype_encoded.name]
        elif kind == 'M':
            fillvalue = None
        else:
            raise TypeError('Unknown/unsupported data type.')

    return fillvalue


def parse_scale_offset(scale_factor, add_offset, dtype_decoded):
    """

    """
    ## Scale and offset
    if scale_factor is None and isinstance(add_offset, (int, float, np.number)):
        scale_factor = 1
    # if isinstance(scale_factor, (int, float, np.number)) and add_offset is None:
    #     add_offset = 0

    if isinstance(scale_factor, (int, float, np.number)) and dtype_decoded.kind != 'f':
        raise ValueError('scale_factor and add_offset only apply to floats.')

    return scale_factor, add_offset


def parse_coord_inputs(name: str, data: np.ndarray | None = None, chunk_shape: Tuple[int] | None = None, dtype_decoded: str | np.dtype | None = None, dtype_encoded: str | np.dtype | None = None, fillvalue: Union[int, float, str] = None, scale_factor: Union[float, int, None] = None, add_offset: Union[float, int, None] = None, step: int | float | bool = False):
    """

    """
    ## Check var name
    if not check_var_name(name):
        raise ValueError(f'{name} is not a valid variable name.')

    ## Check data, shape, dtype, and step
    if isinstance(data, np.ndarray):
        dtype_decoded = data.dtype

        step = init_coord_data_checks(data, step, dtype_decoded, data.shape)

        # if dtype_decoded.kind == 'M':
        #     dtype_encoded = dtype_decoded

        ## dtype encoding
        dtype_decoded, dtype_encoded = parse_dtypes(dtype_decoded, dtype_encoded)

    else:
        ## dtype encoding
        dtype_decoded, dtype_encoded = parse_dtypes(dtype_decoded, dtype_encoded)

        if dtype_decoded.kind in ('u', 'i') and isinstance(step, (float, np.floating)):
            if not step.is_integer():
                raise ValueError('If the dtype_decoded is an integer, then step must be an integer.')
            else:
                step = int(step)
        elif isinstance(step, bool):
            if step:
                raise TypeError('If data is not passed, then step cannot be set to True')
            else:
                step = None
        elif isinstance(step, np.floating):
            step = float(round(step, 5))
        else:
            raise TypeError('step must be a bool, int, or float. The int or float must be greater than 0.')

    ## Guess the chunk_shape from the dtype
    if isinstance(chunk_shape, tuple):
        if not all([isinstance(c, int) for c in chunk_shape]):
            raise TypeError('chunk_shape must be a tuple of ints.')
    elif chunk_shape is None:
        chunk_shape = guess_chunk_shape((1000000,), dtype_encoded, 2**20)
    else:
        raise TypeError('chunk_shape must be either a tuple of ints or None.')

    ## fillvalue
    fillvalue = parse_fillvalue(fillvalue, dtype_encoded)

    ## Scale and offset
    scale_factor, add_offset = parse_scale_offset(scale_factor, add_offset, dtype_decoded)

    ## Save metadata
    dtype_decoded_name, dtype_encoded_name = parse_dtype_names(dtype_decoded, dtype_encoded)

    # enc = data_models.Encoding(dtype_encoded=dtype_encoded_name, dtype_decoded=dtype_decoded_name, fillvalue=fillvalue, scale_factor=scale_factor, add_offset=add_offset)

    var = data_models.CoordinateVariable(shape=(0,), chunk_shape=chunk_shape, origin=0, step=step, dtype_encoded=dtype_encoded_name, dtype_decoded=dtype_decoded_name, fillvalue=fillvalue, scale_factor=scale_factor, add_offset=add_offset)

    return name, var


def parse_var_inputs(sys_meta: data_models.SysMeta, name: str, coords: Tuple[str,...], dtype_decoded: str | np.dtype, dtype_encoded: str | np.dtype | None = None, chunk_shape: Tuple[int] | None = None, fillvalue: Union[int, float, str] = None, scale_factor: Union[float, int, None] = None, add_offset: Union[float, int, None] = None):
    """
    Function to process the inputs to a variable creation function.
    """
    ## Check var name
    if not check_var_name(name):
        raise ValueError(f'{name} is not a valid variable name.')

    if name in sys_meta.variables:
        raise ValueError(f"Dataset already contains the variable {name}.")

    ## Check shape and dtype
    if len(coords) == 0:
        raise ValueError('coords must have at least one value.')

    shape = []
    for coord_name in coords:
        if not isinstance(coord_name, str):
            raise TypeError('coords must contain strings of the coordinate names.')
        if coord_name not in sys_meta.variables:
            raise ValueError(f'{coord_name} not in the list of coordinates.')
        else:
            coord = sys_meta.variables[coord_name]
            shape.append(coord.shape[0])

    ## dtypes
    dtype_decoded, dtype_encoded = parse_dtypes(dtype_decoded, dtype_encoded)

    ## Guess the chunk_shape from the dtype
    if isinstance(chunk_shape, tuple):
        if not all([isinstance(c, int) for c in chunk_shape]):
            raise TypeError('chunk_shape must be a tuple of ints.')
    elif chunk_shape is None:
        chunk_shape = guess_chunk_shape(shape, dtype_encoded, 2**21)
    else:
        raise TypeError('chunk_shape must be either a tuple of ints or None.')

    ## fillvalue
    fillvalue = parse_fillvalue(fillvalue, dtype_encoded)

    ## Scale and offset
    scale_factor, add_offset = parse_scale_offset(scale_factor, add_offset, dtype_decoded)

    ## Save metadata
    dtype_decoded_name, dtype_encoded_name = parse_dtype_names(dtype_decoded, dtype_encoded)

    # enc = data_models.Encoding(dtype_encoded=dtype_encoded_name, dtype_decoded=dtype_decoded_name, fillvalue=fillvalue, scale_factor=scale_factor, add_offset=add_offset)

    var = data_models.DataVariable(coords=tuple(coords), chunk_shape=chunk_shape, dtype_encoded=dtype_encoded_name, dtype_decoded=dtype_decoded_name, fillvalue=fillvalue, scale_factor=scale_factor, add_offset=add_offset)

    return name, var


# def encode_datetime(data, units=None, calendar='gregorian'):
#     """

#     """
#     if units is None:
#         output = data.astype('datetime64[s]').astype('int64')
#     else:
#         if '1970-01-01' in units:
#             time_unit = units.split()[0]
#             output = data.astype(time_str_conversion[time_unit]).astype('int64')
#         else:
#             output = cftime.date2num(data.astype('datetime64[s]').tolist(), units, calendar)

#     return output


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


# def encode_data(data, dtype_encoded, fillvalue, add_offset, scale_factor, compressor) -> bytes:
#     """

#     """
#     if 'datetime64' in data.dtype.name:
#         data = data.astype('int64')

#     elif isinstance(scale_factor, (int, float, np.number)):
#         # precision = int(np.abs(np.log10(val['scale_factor'])))
#         data = np.round((data - add_offset)/scale_factor)

#     if isinstance(fillvalue, (int, np.number)):
#         data[np.isnan(data)] = fillvalue

#     # if (data.dtype.name != dtype_encoded) or (data.dtype.name == 'object'):
#     #     data = data.astype(dtype_encoded)
#     # print(data)
#     data = data.astype(dtype_encoded)

#     return compressor.compress(data.tobytes())


# def decode_data(data: bytes, dtype_encoded, dtype_decoded, missing_value, add_offset=0, scale_factor=None, units=None, calendar=None, **kwargs) -> np.ndarray:
#     """

#     """
#     data = np.frombuffer(data, dtype=dtype_encoded)

#     if isinstance(calendar, str):
#         data = decode_datetime(data, units, calendar)

#     elif isinstance(scale_factor, (int, float, np.number)):
#         data = data.astype(dtype_decoded)

#         if isinstance(missing_value, (int, np.number)):
#             if isinstance(data, np.number):
#                 if data == missing_value:
#                     data = np.nan
#             else:
#                 data[data == missing_value] = np.nan

#         data = (data * scale_factor) + add_offset

#     elif (data.dtype.name != dtype_decoded) or (data.dtype.name == 'object'):
#         data = data.astype(dtype_decoded)

#     return data


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


# def assign_dtype_decoded(encoding):
#     """

#     """
#     if encoding['dtype'] == 'object':
#         encoding['dtype_decoded'] = encoding['dtype']
#     elif ('calendar' in encoding) and ('units' in encoding):
#         encoding['dtype_decoded'] = 'datetime64[s]'

#     if 'scale_factor' in encoding:

#         # if isinstance(encoding['scale_factor'], (int, np.integer)):
#         #     encoding['dtype_decoded'] = np.dtype('float32')
#         if np.dtype(encoding['dtype']).itemsize > 2:
#             encoding['dtype_decoded'] = 'float64'
#         else:
#             encoding['dtype_decoded'] = 'float32'

#     if 'dtype_decoded' not in encoding:
#         encoding['dtype_decoded'] = encoding['dtype']

#     return encoding


def make_var_chunk_key(var_name, chunk_start):
    """

    """
    dims = '.'.join(map(str, chunk_start))
    var_chunk_key = var_chunk_key_str.format(var_name=var_name, dims=dims)

    return var_chunk_key


def check_sel_input_data(sel, input_data, coord_origins, shape):
    """

    """
    slices = indexers.index_combo_all(sel, coord_origins, shape)
    slices_shape = tuple(s.stop - s.start for s in slices)

    if input_data.shape != slices_shape:
        raise ValueError('The selection shape is not equal to the input data.')

    return slices


# def write_chunk(blt_file, var_name, chunk_start_pos, data_chunk_bytes):
#     """

#     """
#     dims = '.'.join(map(str, chunk_start_pos))
#     var_chunk_key = var_chunk_key_str.format(var_name=var_name, dims=dims)

#     # var_name, dims = var_chunk_key.split('!')
#     # chunk_start_pos = tuple(map(int, dims.split('.')))

#     blt_file[var_chunk_key] = data_chunk_bytes


# def write_init_data(blt_file, var_name, var_meta, data, compressor):
#     """

#     """
#     dtype_decoded = np.dtype(var_meta.encoding.dtype_decoded)
#     fillvalue = dtype_decoded.type(var_meta.encoding.fillvalue)
#     dtype_encoded = np.dtype(var_meta.encoding.dtype_encoded)
#     add_offset = var_meta.encoding.add_offset
#     scale_factor = var_meta.encoding.scale_factor

#     mem_arr1 = np.full(var_meta.chunk_shape, fill_value=fillvalue, dtype=dtype_encoded)

#     chunk_iter = rechunker.chunk_range(var_meta.origin, var_meta.shape, var_meta.chunk_shape, clip_ends=True)
#     for chunk in chunk_iter:
#         # print(chunk)
#         mem_arr2 = mem_arr1.copy()
#         mem_chunk = tuple(slice(0, s.stop - s.start) for s in chunk)
#         mem_arr2[mem_chunk] = data[chunk]

#         chunk_start_pos = tuple(s.start for s in chunk)
#         # print(mem_arr2)
#         data_chunk_bytes = encode_data(mem_arr2, dtype_encoded, fillvalue_encoded, add_offset, scale_factor, compressor)

#         write_chunk(blt_file, var_name, chunk_start_pos, data_chunk_bytes)


# def coord_init(name, shape, chunk_shape, enc, sys_meta, step):
#     """

#     """
#     ## Update sys_meta
#     if name in sys_meta.variables:
#         raise ValueError(f'Dataset already contains the variable {name}.')

#     var = data_models.Variable(shape=shape, chunk_shape=chunk_shape, origin=(0,), coords=(name,), is_coord=True, encoding=enc, step=step)

#     sys_meta.variables[name] = var

#     # if data is not None:
#     #     write_init_data(blt_file, name, var, data, compressor)


def check_coords(coords, shape, sys_meta):
    """

    """
    # exist_coords = set(sys_meta.variables.keys())
    # new_coords = set(coords)
    # diff_coords = new_coords.difference(exist_coords)

    # if diff_coords:
    #     raise ValueError(f'{diff_coords} does not exist. Create the coord(s) before creating the data variable.')

    if len(coords) != len(shape):
        raise ValueError(f'The coords length ({len(coords)}) != the shape length ({len(shape)})')

    for coord, size in zip(coords, shape):
        if coord not in sys_meta.variables:
            raise ValueError(f'{coord} does not exist. Create the coord before creating the data variable.')

        exist_coord = sys_meta.variables[coord]

        if not exist_coord.is_coord:
            raise TypeError(f'{coord} must be a coord. This is a data variable.')

        if size != exist_coord.shape[0]:
            raise ValueError(f'The {coord} shape length ({size}) != existing coord length ({exist_coord.shape[0]})')



# def data_var_init(name, coords, shape, chunk_shape, enc, sys_meta):
#     """

#     """
#     ## Update sys_meta
#     if name in sys_meta.variables:
#         raise ValueError(f'Dataset already contains the variable {name}.')

#     var = data_models.Variable(shape=shape, chunk_shape=chunk_shape, origin=0, coords=coords, is_coord=False, encoding=enc)

#     sys_meta.variables[name] = var


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


def guess_chunk_shape(shape: Tuple[int, ...], dtype: np.dtype, target_chunk_size: int = 2**21) -> Tuple[int, ...]:
    """
    Guess an appropriate chunk layout for a dataset, given its shape and
    the size of each element in bytes.  Will allocate chunks only as large
    as target_chunk_size.  Chunks are generally close to some power-of-2 fraction of
    each axis, slightly favoring bigger values for the last index.
    """
    ndims = len(shape)

    if ndims > 0:

        if not all(isinstance(v, int) for v in shape):
            raise TypeError('All values in the shape must be ints.')

        chunks = np.array(shape, dtype='=f8')
        if not np.all(np.isfinite(chunks)):
            raise ValueError("Illegal value in chunk tuple")

        dtype = np.dtype(dtype)
        typesize = dtype.itemsize

        idx = 0
        while True:
            chunk_bytes = math.prod(chunks)*typesize

            if (chunk_bytes < target_chunk_size or \
             abs(chunk_bytes - target_chunk_size)/target_chunk_size < 0.5):
                break

            if math.prod(chunks) == 1:
                break

            chunks[idx%ndims] = math.ceil(chunks[idx%ndims] / 2.0)
            idx += 1

        return tuple(int(x) for x in chunks)
    else:
        return None


# def guess_chunk_hdf5(shape, maxshape, dtype, chunk_max=2**21):
#     """ Guess an appropriate chunk layout for a dataset, given its shape and
#     the size of each element in bytes.  Will allocate chunks only as large
#     as MAX_SIZE.  Chunks are generally close to some power-of-2 fraction of
#     each axis, slightly favoring bigger values for the last index.
#     Undocumented and subject to change without warning.
#     """
#     ndims = len(shape)

#     if ndims > 0:

#         # For unlimited dimensions we have to guess 1024
#         shape1 = []
#         for i, x in enumerate(maxshape):
#             if x is None:
#                 if shape[i] > 1024:
#                     shape1.append(shape[i])
#                 else:
#                     shape1.append(1024)
#             else:
#                 shape1.append(x)

#         shape = tuple(shape1)

#         # ndims = len(shape)
#         # if ndims == 0:
#         #     raise ValueError("Chunks not allowed for scalar datasets.")

#         chunks = np.array(shape, dtype='=f8')
#         if not np.all(np.isfinite(chunks)):
#             raise ValueError("Illegal value in chunk tuple")

#         # Determine the optimal chunk size in bytes using a PyTables expression.
#         # This is kept as a float.
#         typesize = np.dtype(dtype).itemsize
#         # dset_size = np.prod(chunks)*typesize
#         # target_size = CHUNK_BASE * (2**np.log10(dset_size/(1024.*1024)))

#         # if target_size > CHUNK_MAX:
#         #     target_size = CHUNK_MAX
#         # elif target_size < CHUNK_MIN:
#         #     target_size = CHUNK_MIN

#         target_size = chunk_max

#         idx = 0
#         while True:
#             # Repeatedly loop over the axes, dividing them by 2.  Stop when:
#             # 1a. We're smaller than the target chunk size, OR
#             # 1b. We're within 50% of the target chunk size, AND
#             #  2. The chunk is smaller than the maximum chunk size

#             chunk_bytes = math.prod(chunks)*typesize

#             if (chunk_bytes < target_size or \
#              abs(chunk_bytes - target_size)/target_size < 0.5):
#                 break

#             if math.prod(chunks) == 1:
#                 break

#             chunks[idx%ndims] = math.ceil(chunks[idx%ndims] / 2.0)
#             idx += 1

#         return tuple(int(x) for x in chunks)
#     else:
#         return None


# def guess_chunk_time(shape, maxshape, dtype, time_index, chunk_max=3*2**20):
#     """ Guess an appropriate chunk layout for a dataset, given its shape and
#     the size of each element in bytes.  Will allocate chunks only as large
#     as MAX_SIZE.  Chunks are generally close to some power-of-2 fraction of
#     each axis, slightly favoring bigger values for the last index.
#     Undocumented and subject to change without warning.
#     """
#     ndims = len(shape)

#     if ndims > 0:

#         # For unlimited dimensions we have to guess 1024
#         shape1 = []
#         for i, x in enumerate(maxshape):
#             if x is None:
#                 if shape[i] > 1024:
#                     shape1.append(shape[i])
#                 else:
#                     shape1.append(1024)
#             else:
#                 shape1.append(x)

#         shape = tuple(shape1)

#         chunks = np.array(shape, dtype='=f8')
#         if not np.all(np.isfinite(chunks)):
#             raise ValueError("Illegal value in chunk tuple")

#         # Determine the optimal chunk size in bytes using a PyTables expression.
#         # This is kept as a float.
#         typesize = np.dtype(dtype).itemsize

#         target_size = chunk_max

#         while True:
#             # Repeatedly loop over the axes, dividing them by 2.  Stop when:
#             # 1a. We're smaller than the target chunk size, OR
#             # 1b. We're within 50% of the target chunk size, AND
#             #  2. The chunk is smaller than the maximum chunk size

#             chunk_bytes = math.prod(chunks)*typesize

#             if (chunk_bytes < target_size or \
#              abs(chunk_bytes - target_size)/target_size < 0.5):
#                 break

#             if chunks[time_index] == 1:
#                 break

#             chunks[time_index] = np.ceil(chunks[time_index] / 2.0)

#         return tuple(int(x) for x in chunks)
#     else:
#         return None


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


def is_var_name(name):
    """

    """
    res = var_name_pattern.search(name)
    if res:
        return True
    else:
        return False


def format_value(value):
    """

    """
    if isinstance(value, (int, np.integer)):
        return str(value)
    elif isinstance(value, (float, np.floating)):
        return f'{value:.2f}'
    else:
        return value


def append_summary(summary, summ_dict):
    """

    """
    for key, value in summ_dict.items():
        spacing = value_indent - len(key)
        if spacing < 1:
            spacing = 1

        summary += f"""\n{key}""" + """ """ * spacing + value

    return summary


def data_variable_summary(ds):
    """

    """
    if ds:
        summ_dict = {'name': ds.name, 'dims order': '(' + ', '.join(ds.coord_names) + ')', 'shape': str(ds.shape), 'chunk size': str(ds.chunk_shape)}

        type1 = type(ds)

        summary = f"""<cfdb.{type1.__name__}>"""

        summary = append_summary(summary, summ_dict)

        summary += """\nCoordinates:"""

        for coord in ds.coords:
            coord_name = coord.name
            dtype_name = coord.dtype_decoded
            dim_len = coord.shape[0]
            first_value = format_value(coord.data[0])
            spacing = value_indent - name_indent - len(coord_name)
            if spacing < 1:
                spacing = 1
            dim_str = f"""\n    {coord_name}""" + """ """ * spacing
            dim_str += f"""({dim_len}) {dtype_name} {first_value} ..."""
            summary += dim_str

        attrs_summary = make_attrs_repr(ds.attrs, name_indent, value_indent, 'Attributes')
        summary += """\n""" + attrs_summary

    else:
        summary = """Dataset is closed"""

    return summary


def coordinate_summary(ds):
    """

    """
    if ds:
        name = ds.name
        # dim_len = ds.ndims
        # dtype_name = ds.dtype.name
        # dtype_decoded = ds.encoding['dtype_decoded']

        first_value = format_value(ds.data[0])
        last_value = format_value(ds.data[-1])

        # summ_dict = {'name': name, 'dtype encoded': dtype_name, 'dtype decoded': dtype_decoded, 'chunk size': str(ds.chunks), 'dim length': str(dim_len), 'values': f"""{first_value} ... {last_value}"""}
        summ_dict = {'name': name, 'shape': str(ds.shape), 'chunk shape': str(ds.chunk_shape), 'values': f"""{first_value} ... {last_value}"""}

        type1 = type(ds)

        summary = f"""<cfdb.{type1.__name__}>"""

        summary = append_summary(summary, summ_dict)

        attrs_summary = make_attrs_repr(ds.attrs, name_indent, value_indent, 'Attributes')
        summary += """\n""" + attrs_summary
    else:
        summary = """Dataset is closed"""

    return summary


def make_attrs_repr(attrs, name_indent, value_indent, header):
    summary = f"""{header}:"""
    for key, value in attrs.items():
        spacing = value_indent - name_indent - len(key)
        if spacing < 1:
            spacing = 1
        line_str = f"""\n    {key}""" + """ """ * spacing + f"""{value}"""
        summary += line_str

    return summary


# def create_h5py_data_variable(file, name: str, dims: (str, tuple, list), shape: (tuple, list), encoding: dict, data=None, **kwargs):
#     """

#     """
#     dtype = encoding['dtype']

#     ## Check if dims already exist and if the dim lengths match
#     if isinstance(dims, str):
#         dims = [dims]

#     for i, dim in enumerate(dims):
#         if dim not in file:
#             raise ValueError(f'{dim} not in File')

#         dim_len = file._file[dim].shape[0]
#         if dim_len != shape[i]:
#             raise ValueError(f'{dim} does not have the same length as the input data/shape dim.')

#     ## Make chunks
#     if 'chunks' not in kwargs:
#         if 'maxshape' in kwargs:
#             maxshape = kwargs['maxshape']
#         else:
#             maxshape = shape
#         kwargs.setdefault('chunks', utils.guess_chunk(shape, maxshape, dtype))

#     ## Create variable
#     if data is None:
#         ds = file._file.create_dataset(name, shape, dtype=dtype, track_order=True, **kwargs)
#     else:
#         ## Encode data before creating variable
#         data = utils.encode_data(data, **encoding)

#         ds = file._file.create_dataset(name, dtype=dtype, data=data, track_order=True, **kwargs)

#     for i, dim in enumerate(dims):
#         ds.dims[i].attach_scale(file._file[dim])
#         ds.dims[i].label = dim

#     return ds


# def create_h5py_coordinate(file, name: str, data, shape: (tuple, list), encoding: dict, **kwargs):
#     """

#     """
#     if len(shape) != 1:
#         raise ValueError('The shape of a coordinate must be 1-D.')

#     dtype = encoding['dtype']

#     ## Make chunks
#     if 'chunks' not in kwargs:
#         if 'maxshape' in kwargs:
#             maxshape = kwargs['maxshape']
#         else:
#             maxshape = shape
#         kwargs.setdefault('chunks', utils.guess_chunk(shape, maxshape, dtype))

#     ## Encode data before creating variable/coordinate
#     # print(encoding)
#     data = utils.encode_data(data, **encoding)

#     # print(data)
#     # print(dtype)

#     ## Make Variable
#     ds = file._file.create_dataset(name, dtype=dtype, data=data, track_order=True, **kwargs)

#     ds.make_scale(name)
#     ds.dims[0].label = name

#     return ds


# def copy_data_variable(to_file, from_variable, name, include_data=True, include_attrs=True, **kwargs):
#     """

#     """
#     other1 = from_variable._dataset
#     for k in ('chunks', 'compression',
#               'compression_opts', 'scaleoffset', 'shuffle', 'fletcher32',
#               'fillvalue'):
#         kwargs.setdefault(k, getattr(other1, k))

#     if 'compression' in other1.attrs:
#         compression = other1.attrs['compression']
#         kwargs.update(**utils.get_compressor(compression))
#     else:
#         compression = kwargs['compression']

#     # TODO: more elegant way to pass these (dcpl to create_variable?)
#     dcpl = other1.id.get_create_plist()
#     kwargs.setdefault('track_times', dcpl.get_obj_track_times())
#     # kwargs.setdefault('track_order', dcpl.get_attr_creation_order() > 0)

#     # Special case: the maxshape property always exists, but if we pass it
#     # to create_variable, the new variable will automatically get chunked
#     # layout. So we copy it only if it is different from shape.
#     if other1.maxshape != other1.shape:
#         kwargs.setdefault('maxshape', other1.maxshape)

#     encoding = from_variable.encoding._encoding.copy()
#     shape = from_variable.shape

#     ds0 = create_h5py_data_variable(to_file, name, tuple(dim.label for dim in other1.dims), shape, encoding, **kwargs)

#     if include_data:
#         # Directly copy chunks using write_direct_chunk
#         for chunk in ds0.iter_chunks():
#             chunk_starts = tuple(c.start for c in chunk)
#             filter_mask, data = other1.id.read_direct_chunk(chunk_starts)
#             ds0.id.write_direct_chunk(chunk_starts, data, filter_mask)

#     ds = DataVariable(ds0, to_file, encoding)
#     if include_attrs:
#         ds.attrs.update(from_variable.attrs)

#     return ds


# def copy_coordinate(to_file, from_coordinate, name, include_attrs=True, **kwargs):
#     """

#     """
#     other1 = from_coordinate._dataset
#     for k in ('chunks', 'compression',
#               'compression_opts', 'scaleoffset', 'shuffle', 'fletcher32',
#               'fillvalue'):
#         kwargs.setdefault(k, getattr(other1, k))

#     if 'compression' in other1.attrs:
#         compression = other1.attrs['compression']
#         kwargs.update(**utils.get_compressor(compression))
#     else:
#         compression = kwargs['compression']

#     # TODO: more elegant way to pass these (dcpl to create_variable?)
#     dcpl = other1.id.get_create_plist()
#     kwargs.setdefault('track_times', dcpl.get_obj_track_times())
#     # kwargs.setdefault('track_order', dcpl.get_attr_creation_order() > 0)

#     # Special case: the maxshape property always exists, but if we pass it
#     # to create_variable, the new variable will automatically get chunked
#     # layout. So we copy it only if it is different from shape.
#     if other1.maxshape != other1.shape:
#         kwargs.setdefault('maxshape', other1.maxshape)

#     encoding = from_coordinate.encoding._encoding.copy()
#     shape = from_coordinate.shape

#     ds0 = create_h5py_coordinate(to_file, name, from_coordinate.data, shape, encoding, **kwargs)

#     ds = Coordinate(ds0, to_file, encoding)
#     if include_attrs:
#         ds.attrs.update(from_coordinate.attrs)

#     return ds


def prepare_encodings_for_variables(dtype_encoded, dtype_decoded, scale_factor, add_offset, fillvalue, units, calendar):
    """

    """
    encoding = {'dtype': dtype_encoded, 'dtype_encoded': dtype_encoded, 'missing_value': fillvalue, '_FillValue': fillvalue, 'add_offset': add_offset, 'scale_factor': scale_factor, 'units': units, 'calendar': calendar}
    for key, value in copy.deepcopy(encoding).items():
        if value is None:
            del encoding[key]

    if 'datetime64' in dtype_decoded:
        if 'units' not in encoding:
            encoding['units'] = 'seconds since 1970-01-01'
        if 'calendar' not in encoding:
            encoding['calendar'] = 'gregorian'
        encoding['dtype'] = 'int64'

    return encoding


def file_summary(ds):
    """

    """
    if ds:
        file_path = ds.file_path
        if file_path.exists() and file_path.is_file():
            file_size = file_path.stat().st_size*0.000001
            file_size_str = """{file_size:.1f} MB""".format(file_size=file_size)
        else:
            file_size_str = """NA"""

        summ_dict = {'file name': file_path.name, 'file size': file_size_str, 'writable': str(ds.writable)}

        summary = """<cfdb.Dataset>"""

        summary = append_summary(summary, summ_dict)

        summary += """\nCoordinates:"""

        for var in ds.coords:
            dim_name = var.name
            dtype_name = var.dtype_decoded
            dim_len = var.shape[0]
            first_value = format_value(var.data[0])
            last_value = format_value(var.data[-1])
            spacing = value_indent - name_indent - len(dim_name)
            if spacing < 1:
                spacing = 1
            dim_str = f"""\n    {dim_name}""" + """ """ * spacing
            dim_str += f"""({dim_len}) {dtype_name} {first_value} ... {last_value}"""
            summary += dim_str

        summary += """\nData Variables:"""

        for dv in ds.data_vars:
            dv_name = dv.name
            dtype_name = dv.dtype_decoded
            # shape = dv.shape
            dims = ', '.join(dv.coord_names)
            # first_value = format_value(dv[tuple(0 for i in range(len(shape)))])
            spacing = value_indent - name_indent - len(dv_name)
            if spacing < 1:
                spacing = 1
            ds_str = f"""\n    {dv_name}""" + """ """ * spacing
            ds_str += f"""({dims}) {dtype_name}"""
            summary += ds_str

        attrs_summary = make_attrs_repr(ds.attrs, name_indent, value_indent, 'Attributes')
        summary += """\n""" + attrs_summary
    else:
        summary = """File is closed"""

    return summary















































































