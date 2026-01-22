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
from copy import deepcopy
# import xarray as xr
# from time import time
# from datetime import datetime
# import cftime
import math
import rechunkit
from typing import Set, Optional, Dict, Tuple, List, Union, Any
# import zstandard as zstd
# import lz4
import booklet

# import dateutil.parser as dparser
# import numcodecs
# import hdf5plugin

from . import data_models, dtypes
# import data_models, dtypes

########################################################
### Parmeters


# CHUNK_BASE = 32*1024    # Multiplier by which chunks are adjusted
# CHUNK_MIN = 32*1024      # Soft lower limit (32k)
chunk_max = 2**21   # Hard upper limit (4M)

time_str_conversion = {'days': 'datetime64[D]',
                       'hours': 'datetime64[h]',
                       'minutes': 'datetime64[m]',
                       'seconds': 'datetime64[s]',
                       'milliseconds': 'datetime64[ms]',
                       'microseconds': 'datetime64[us]',
                       'nanoseconds': 'datetime64[ns]'}

# enc_fields = ('units', 'calendar', 'dtype', 'missing_value', '_FillValue', 'add_offset', 'scale_factor', 'dtype_decoded', 'dtype_encoded', 'compression')

fillvalue_dict = {
    'int8': -128,
    'int16': -32768,
    'int32': -2147483648,
    'int64': -9223372036854775808
    }

var_chunk_key_str = '{var_name}!{dims}'

attrs_key_str = '_{var_name}.attrs'

name_indent = 4
value_indent = 20
var_name_regex = "^[a-zA-Z][a-zA-Z0-9_]*$"
var_name_pattern = re.compile(var_name_regex)

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

compression_options = ('zstd', 'lz4')
default_compression_levels = {'zstd': 1, 'lz4': 1}
default_n_buckets = 144013

time_dtype_params = {
    'datetime64[M]': {'dtype_encoded': 'int16', 'name': 'datetime64[M]', 'offset': -841},
    'datetime64[D]': {'dtype_encoded': 'int32', 'name': 'datetime64[D]', 'offset': -25568},
    'datetime64[h]': {'dtype_encoded': 'int32', 'name': 'datetime64[h]', 'offset': -613609},
    'datetime64[m]': {'dtype_encoded': 'int32', 'name': 'datetime64[m]', 'offset': -36816481},
    'datetime64[s]': {'dtype_encoded': 'int64', 'name': 'datetime64[s]', 'offset': -2208988801},
    'datetime64[ms]': {'dtype_encoded': 'int64', 'name': 'datetime64[ms]', 'offset': -2208988800001},
    'datetime64[us]': {'dtype_encoded': 'int64', 'name': 'datetime64[us]', 'offset': -2208988800000001},
    'datetime64[ns]': {'dtype_encoded': 'int64', 'name': 'datetime64[ns]', 'offset': -631152000000000001},
    }

default_dtype_params = {
    'lon': {'precision': 6, 'name': 'float64', 'offset': -180.000001, 'dtype_encoded': 'int32'},
    'lat': {'precision': 6, 'name': 'float64', 'offset': -90.000001, 'dtype_encoded': 'int32'},
    'height': {'dtype_encoded': 'int32', 'offset': -1, 'precision': 3, 'name': 'float64'},
    'altitude': {'dtype_encoded': 'int32', 'offset': -11000.001, 'precision': 3, 'name': 'float64'},
    'time': time_dtype_params['datetime64[m]'],
    'modified_date': {'dtype_encoded': 'int64', 'name': 'datetime64[us]', 'offset': 1756684800000000},
    'band': {'name': 'uint8'},
    'censor_code': {'name': 'uint8'},
    'x': {'precision': 1, 'name': 'float32'},
    'y': {'precision': 1, 'name': 'float32'},
                 # 'bore_top_of_screen': {'dtype_encoded': 'int16', 'fillvalue': 9999, 'scale_factor': 0.1},
                 # 'bore_bottom_of_screen': {'dtype_encoded': 'int16', 'fillvalue': 9999, 'scale_factor': 0.1},
                 # 'bore_depth': {'dtype_encoded': 'int16', 'fillvalue': -9999, 'scale_factor': 0.1},
                 # 'reference_level': {'dtype_encoded': 'int16', 'fillvalue': -9999, 'scale_factor': 1},
                 }

default_var_params = {
    'lon': {'name': 'longitude', 'axis': 'x'},
    'lat': {'name': 'latitude', 'axis': 'y'},
    'height': {'name': 'height', 'axis': 'z'},
    'altitude': {'name': 'altitude', 'axis': 'z'},
    'time': {'name': 'time', 'axis': 't'},
    'modified_date': {'name': 'modified_date'},
    'band': {'name': 'band'},
    'censor_code': {'name': 'censor_code'},
    'x': {'name': 'x', 'axis': 'x'},
    'y': {'name': 'y', 'axis': 'y'},
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
        # 'units': 'seconds since 1970-01-01 00:00:00',
        'standard_name': 'time',
        # 'calendar': 'proleptic_gregorian',
        'axis': 'T',
        },
    censor_code = {
        'long_name': 'data censor code',
        'standard_name': 'status_flag',
        'flag_values': '0 1 2 3 4 5',
        'flag_meanings': 'greater_than less_than not_censored non-detect present_but_not_quantified unknown'
        },
    band = {
        'long_name': 'band number',
        },
    y={
        'long_name': 'y coordinate of projection',
        'units': 'metres',
        'standard_name': 'projection_y_coordinate',
        'axis': 'Y',
        },
    x={
        'long_name': 'x coordinate of projection',
        'units': 'metres',
        'standard_name': 'projection_x_coordinate',
        'axis': 'X',
        },
    )


crs_name_dict = {
    'Lat': 'latitude',
    'Lon': 'longitude',
    'N': 'y',
    'E': 'x',
    }

#########################################################
### Classes



#########################################################
### Functions


def filter_var_names(ds, include_data_vars, exclude_data_vars):
    """

    """
    if include_data_vars is not None:
        if isinstance(include_data_vars, str):
            include_data_vars = [include_data_vars]
        data_var_names = set(include_data_vars)
    elif exclude_data_vars is not None:
        if isinstance(exclude_data_vars, str):
            exclude_data_vars = [exclude_data_vars]
        data_var_names_all = set(ds.data_var_names)
        data_var_names = data_var_names_all.difference(set(exclude_data_vars))
    else:
        data_var_names = set(ds.data_var_names)

    coord_names = set()
    for data_var_name in data_var_names:
        data_var = ds[data_var_name]
        coord_names.update(data_var.coord_names)

    return data_var_names, coord_names


def parse_cf_time_units(dtype_decoded):
    """

    """
    np_time_str = dtype_decoded.str.split('[')[1].split(']')[0]
    time_name = time_units_dict[np_time_str]
    datetime = np.datetime64('1970-01-01', np_time_str)
    units = f'{time_name} since {datetime}'

    return units


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


def coord_data_step_check(data: np.ndarray, dtype: dtypes.DataType, step: int | float | bool = False):
    """

    """
    diff = np.diff(data)
    if isinstance(step, bool):
        # diff = np.diff(data)
        if dtype.kind == 'f':
            step = float(np.round(diff[0], 5))
            if not np.allclose(step, diff):
                # raise ValueError('step is set to True, but the data does not seem to be regular.')
                step = None
            # data = np.linspace(data[0], data[-1], len(diff) + 1, dtype=dtype_decoded)
        elif dtype.kind in ('u', 'i', 'M'):
            if dtype.kind == 'M':
                step = int(diff[0].astype(int))
            else:
                step = int(diff[0])

            if not np.all(np.equal(step, diff)):
                # raise ValueError('step is set to True, but the data does not seem to be regular.')
                step = None
        else:
            step = None
    elif isinstance(step, (float, np.floating)):
        step = float(round(step, 5))
        if step <= 0:
            raise ValueError('step must be greater than 0.')
        if not np.allclose(step, diff):
            raise ValueError('step does not seem to be the interval of the data.')
        # num = round((data[-1] - data[0])/step, 5)
        # if not num.is_integer():
        #     raise ValueError('The step is not a multiple of the difference between the first and last values of the data.')

        # data = np.linspace(data[0], data[-1], int(num) + 1, dtype=dtype_decoded)
    elif isinstance(step, (int, np.integer)):
        step = int(step)

        if step <= 0:
            raise ValueError('step must be greater than 0.')
        if not np.all(np.equal(step, diff)):
            raise ValueError('step does not seem to be the interval of the data.')

        # data = np.linspace(data[0], data[-1], int(num) + 1, dtype=dtype_decoded)
    elif step is not None:
        raise TypeError('step must be a bool, int, or float. The int or float must be greater than 0.')

    # num = round((data[-1] - data[0])/step, 5)
    # if not num.is_integer():
    #     raise ValueError('The step is not a multiple of the difference between the first and last values of the data.')

    return step


def init_coord_data_checks(data: np.ndarray, step: int | float | bool, dtype, shape):
    """

    """
    # dtype_decoded = data.dtype
    # shape = data.shape

    if len(shape) > 1:
        raise ValueError('Coordinates must be 1D.')

    if len(np.unique(data)) < shape[0]:
        raise ValueError('The data for coords must be unique.')

    if dtype.kind in ('f', 'u', 'i', 'M'):
        data.sort()
        if step:
            step = coord_data_step_check(data, dtype, step)
            # data = np.linspace(data[0], data[-1], num + 1, dtype=dtype_decoded)
        else:
            step = None
    else:
        step = None

    return step


def append_coord_data_checks(new_data: np.ndarray, source_data: np.ndarray, source_dtype: dtypes.DataType = None, source_step: int | float | None = None):
    """

    """
    # new_shape = new_data.shape
    # new_dtype_decoded = new_data.dtype
    new_data = np.asarray(new_data, dtype=source_dtype.dtype_decoded)

    # if source_dtype_decoded != new_dtype_decoded:
    #     raise TypeError('The data dtype does not match the originally assigned dtype.')

    # print(source_data)

    if source_data.size > 0:
        if source_dtype.kind != 'U':
            last = source_data[-1]

            if not np.all(last < new_data):
                raise ValueError('Appending requires that all values are greater than the existing values.')

            new_data.sort()
            if source_step:
                _ = coord_data_step_check(new_data, source_dtype, source_step)

                new_data = np.linspace(source_data[0], new_data[-1], len(source_data) + len(new_data), dtype=source_dtype.dtype_decoded)
            else:
                new_data = np.append(source_data, new_data)

        else:
            s1 = set(source_data)
            s1.update(set(new_data))
            if len(s1) != (len(source_data) + len(new_data)):
                raise ValueError('The data for coords must be unique.')

            new_data = np.append(source_data, new_data)

    else:
        _ = init_coord_data_checks(new_data, source_step, source_dtype, new_data.shape)

    return new_data


def prepend_coord_data_checks(new_data: np.ndarray, source_data: np.ndarray, source_dtype: dtypes.DataType = None, source_step: int | float | None = None):
    """

    """
    # new_shape = new_data.shape
    # new_dtype_decoded = new_data.dtype
    new_data = np.asarray(new_data, dtype=source_dtype.dtype_decoded)

    # if source_dtype_decoded != new_dtype_decoded:
    #     raise TypeError('The data dtype does not match the originally assigned dtype.')

    if source_data.size > 0:
        if source_dtype.kind != 'U':
            first = source_data[0]

            if not np.all(first > new_data):
                raise ValueError('Prepending requires that all values are less than the existing values.')

            new_data.sort()
            if source_step:
                _ = coord_data_step_check(new_data, source_dtype, source_step)

                new_data = np.linspace(new_data[0], source_data[-1], len(source_data) + len(new_data), dtype=source_dtype.dtype_decoded)
            else:
                new_data = np.append(new_data, source_data)
        else:
            s1 = set(source_data)
            s1.update(set(new_data))
            if len(s1) != (len(source_data) + len(new_data)):
                raise ValueError('The data for coords must be unique.')

            new_data = np.append(new_data, source_data)

    else:
        _ = init_coord_data_checks(new_data, source_step, source_dtype, new_data.shape)

    return new_data


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
        dtype_decoded_name = dtype_decoded.str
        dtype_encoded_name = dtype_encoded.str
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


def parse_coord_inputs(name: str, data: np.ndarray | None = None, chunk_shape: Tuple[int] | None = None, dtype: str | np.dtype | dtypes.DataType | None = None, step: int | float | bool = False, axis: str=None):
    """

    """
    ## Check var name
    if not check_var_name(name):
        raise ValueError(f'{name} is not a valid variable name.')

    ## Parse dtype
    if isinstance(dtype, (str, np.dtype)):
        dtype = dtypes.dtype(dtype)

    ## Check data, shape, dtype, and step
    if isinstance(data, np.ndarray):
        if not isinstance(dtype, dtypes.DataType):
            np_dtype = data.dtype
            dtype = dtypes.dtype(np_dtype)

        step = init_coord_data_checks(data, step, dtype, data.shape)

        # if dtype_decoded.kind == 'M':
        #     dtype_encoded = dtype_decoded

        ## dtype encoding
        # dtype_decoded, dtype_encoded = parse_dtypes(dtype_decoded, dtype_encoded)

    else:
        if dtype is None:
            raise TypeError('dtype must not be None.')

        ## dtype encoding
        # dtype_decoded, dtype_encoded = parse_dtypes(dtype_decoded, dtype_encoded)

        if dtype.kind in ('u', 'i'):
            if isinstance(step, (float, np.floating)):
                if step.is_integer():
                    step = int(step)
                else:
                    raise ValueError('If the dtype is an integer, then step must be an integer.')

        elif isinstance(step, bool):
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
        if dtype.dtype_encoded is None:
            itemsize = dtype.itemsize
            if itemsize is None:
                if dtype.name == 'str':
                    itemsize = 12
                else:
                    itemsize = 60
        else:
            itemsize = dtype.dtype_encoded.itemsize

        chunk_shape = rechunkit.guess_chunk_shape((1000000,), itemsize, chunk_max)
    else:
        raise TypeError('chunk_shape must be either a tuple of ints or None.')

    if isinstance(axis, str):
        axis0 = data_models.Axis(axis)
    else:
        axis0 = None

    dtype_dict = dtype.to_dict()

    var = data_models.CoordinateVariable(shape=(0,), chunk_shape=chunk_shape, origin=0, step=step, dtype=dtype_dict, axis=axis0)

    return name, var


def parse_var_inputs(sys_meta: data_models.SysMeta, name: str, coords: Tuple[str,...], dtype: str | np.dtype | dtypes.DataType, chunk_shape: Tuple[int] | None = None):
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
    # dtype_decoded, dtype_encoded = parse_dtypes(dtype_decoded, dtype_encoded)
    if isinstance(dtype, (str, np.dtype)):
        dtype = dtypes.dtype(dtype)
    elif not isinstance(dtype, dtypes.DataType):
        raise TypeError('dtype must be either a str, np.dtype, or cfdb.dtype.')

    ## Guess the chunk_shape from the dtype
    if isinstance(chunk_shape, tuple):
        if not all([isinstance(c, int) for c in chunk_shape]):
            raise TypeError('chunk_shape must be a tuple of ints.')
    elif chunk_shape is None:
        if dtype.dtype_encoded is None:
            itemsize = dtype.itemsize
            if itemsize is None:
                if dtype.name == 'str':
                    itemsize = 12
                else:
                    itemsize = 60
        else:
            itemsize = dtype.dtype_encoded.itemsize
        chunk_shape = rechunkit.guess_chunk_shape(shape, itemsize, chunk_max)
    else:
        raise TypeError('chunk_shape must be either a tuple of ints or None.')

    var = data_models.DataVariable(coords=tuple(coords), chunk_shape=chunk_shape, dtype=dtype.to_dict())

    return name, var


def make_var_chunk_key(var_name, chunk_start):
    """

    """
    dims = '.'.join(map(str, chunk_start))
    var_chunk_key = var_chunk_key_str.format(var_name=var_name, dims=dims)

    return var_chunk_key


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
    type1 = type(ds)

    if ds:
        summ_dict = {'name': ds.name, 'dtype': ds.dtype.name, 'dims order': '(' + ', '.join(ds.coord_names) + ')', 'shape': str(ds.shape), 'chunk shape': str(ds.chunk_shape)}

        summary = f"""<cfdb.{type1.__name__}>"""

        summary = append_summary(summary, summ_dict)

        summary += """\nCoordinates:"""

        for coord in ds.coords:
            coord_name = coord.name
            dtype_name = coord.dtype.name
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
        summary = f"""<cfdb.{type1.__name__} is closed>"""

    return summary


def coordinate_summary(ds):
    """

    """
    type1 = type(ds)

    if ds:
        name = ds.name
        # dim_len = ds.ndims
        # dtype_name = ds.dtype.name
        # dtype_decoded = ds.encoding['dtype_decoded']
        data = ds.data
        if len(data) > 0:
            first_value = format_value(ds.data[0])
            last_value = format_value(ds.data[-1])
        else:
            first_value = ''
            last_value = ''

        # summ_dict = {'name': name, 'dtype encoded': dtype_name, 'dtype decoded': dtype_decoded, 'chunk size': str(ds.chunks), 'dim length': str(dim_len), 'values': f"""{first_value} ... {last_value}"""}
        summ_dict = {'name': name, 'dtype': ds.dtype.name, 'shape': str(ds.shape), 'chunk shape': str(ds.chunk_shape), 'values': f"""{first_value} ... {last_value}"""}

        summary = f"""<cfdb.{type1.__name__}>"""

        summary = append_summary(summary, summ_dict)

        attrs_summary = make_attrs_repr(ds.attrs, name_indent, value_indent, 'Attributes')
        summary += """\n""" + attrs_summary
    else:
        summary = f"""<cfdb.{type1.__name__} is closed>"""

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


def file_summary(ds):
    """

    """
    type1 = type(ds)

    if ds:
        file_path = ds.file_path
        if file_path.exists() and file_path.is_file():
            file_size = file_path.stat().st_size*0.000001
            file_size_str = """{file_size:.1f} MB""".format(file_size=file_size)
        else:
            file_size_str = """NA"""

        summ_dict = {'file name': file_path.name, 'file size': file_size_str, 'writable': str(ds.writable)}

        summary = f"""<cfdb.{type1.__name__}>"""

        summary = append_summary(summary, summ_dict)

        summary += """\nCoordinates:"""

        for var in ds.coords:
            dim_name = var.name
            dtype_name = var.dtype.name
            dim_len = var.shape[0]
            # print(var.data)
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
            dtype_name = dv.dtype.name
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
        summary = f"""<cfdb.{type1.__name__} is closed>"""

    return summary


def get_dtype_params(name, kwargs={}):
    """

    """
    params = deepcopy(default_dtype_params[name])
    params.update(kwargs)

    return name, params

def get_var_params(name, kwargs={}):
    """

    """
    if 'dtype' in kwargs:
        dtype = dtypes.dtype(kwargs.pop('dtype'))
    else:
        dtype = dtypes.dtype(**default_dtype_params[name])

    var_params = deepcopy(default_var_params[name])
    var_params.update(kwargs)

    var_name = var_params.pop('name')

    attrs = deepcopy(default_attrs[name])

    return var_name, var_params, dtype, attrs








































































