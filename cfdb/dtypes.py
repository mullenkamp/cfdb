#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 11:04:14 2025

@author: mike
"""
import numpy as np
from typing import Set, Optional, Dict, Tuple, List, Union, Any
import msgspec
import shapely

sup = np.testing.suppress_warnings()
sup.filter(RuntimeWarning)


###########################################
### Transcoders


class Transcoder:
    """
    Convert a dtype to some kind of integer dtype. Currently, the source dtype can be either a float,
    """
    def __init__(self, dtype_decoded: str, dtype_encoded: str, precision: int, offset: int=0):
        """

        """
        self.dtype_decoded = np.dtype(dtype_decoded)
        self.dtype_encoded = np.dtype(dtype_encoded)
        self.factor = 10**precision
        self.offset = offset

    @sup
    def encode(self, data_decoded: np.ndarray):
        """

        """
        if self.factor != 1:
            data_encoded = (np.round(data_decoded * self.factor) + self.offset).astype(self.dtype_encoded)
        else:
            data_encoded = (data_decoded + self.offset).astype(self.dtype_encoded)

        return data_encoded

    def decode(self, data_encoded: np.ndarray):
        """

        """
        data_decoded = data_encoded.astype(self.dtype_decoded)

        ## Datetime exception...
        if self.dtype_decoded.kind == 'M':
            data_decoded[data_decoded == np.array(0, dtype=self.dtype_decoded)] = np.datetime64('nat')
        elif self.dtype_decoded.kind == 'f':
            data_decoded[data_decoded == 0] = np.nan

        if self.factor != 1:
            data_decoded = (data_decoded - self.offset)/self.factor
        else:
            data_decoded = data_decoded - self.offset

        return data_decoded

    def from_bytes(self, data_bytes: bytes, chunk_shape: tuple=None):
        """

        """
        data = np.frombuffer(data_bytes, dtype=self.dtype_encoded).reshape(chunk_shape)

        return data

    def to_bytes(self, data_encoded: np.ndarray):
        """

        """
        return data_encoded.tobytes()

    def to_dict(self):
        """

        """
        dict1 = {
            'dtype_encoded': self.dtype_encoded.name,
            'offset': self.offset,
            }
        return dict1



############################################
### DTypes


class DataType:
    def __repr__(self):
        """

        """
        return self.name

    def to_dict(self):
        """

        """
        dict1 = {'name': self.name}
        if self.precision is not None:
            dict1['precision'] = self.precision
        if hasattr(self, 'crs'):
            dict1['crs'] = self.crs.to_string()
        if self.transcoder is not None:
            dict1.update(self.transcoder.to_dict())

        return dict1



class FixedLen(DataType):
    def __init__(self, np_dtype):
        """

        """
        self.kind = np_dtype.kind
        self.itemsize = np_dtype.itemsize
        self.np_dtype = np_dtype
        self.precision = None
        self.transcoder = None
        self.name = np_dtype.name

    def dumps(self, data: np.ndarray):
        """

        """
        return data.tobytes()

    def loads(self, data_bytes: bytes, chunk_shape: tuple=None):
        """

        """
        data = np.frombuffer(bytearray(data_bytes), dtype=self.np_dtype).reshape(chunk_shape)

        return data


class Geometry(DataType):
    """

    """
    def __init__(self, precision: int=None):
        """

        """
        self.kind = 'G'
        self.np_dtype = np.dtypes.ObjectDType()
        self.precision = precision
        self.transcoder = None
        self.itemsize = None
        # self.crs = pyproj.CRS.from_user_input(crs)

    def dumps(self, data: np.ndarray):
        """

        """
        return msgspec.msgpack.encode(shapely.to_wkt(data, rounding_precision=self.precision).tolist())

    def loads(self, data_bytes: bytes, chunk_shape: tuple=None):
        """

        """
        data = np.asarray(shapely.from_wkt(msgspec.msgpack.decode(data_bytes)), dtype=self.np_dtype).reshape(chunk_shape)

        return data


class Point(Geometry):
    name = 'Point'


class LineString(Geometry):
    name = 'LineString'


class Polygon(Geometry):
    name = 'Polygon'


# class Integer(FixedLen):
#     """

#     """

class Bool(FixedLen):
    """

    """


class DTypeTranscoder(DataType):
    def __init__(self, np_dtype, precision: int=None, transcoder: Transcoder=None):
        """

        """
        self.kind = np_dtype.kind
        self.itemsize = np_dtype.itemsize
        self.np_dtype = np_dtype
        self.transcoder = transcoder
        self.precision  = precision
        self.name = np_dtype.name

    def dumps(self, data: np.ndarray):
        """

        """
        if self.transcoder is None:
            if isinstance(self.precision, int):
                return data.round(self.precision).tobytes()
            else:
                return data.tobytes()
        else:
            data_encoded = self.transcoder.encode(data)
            return self.transcoder.to_bytes(data_encoded)

    def loads(self, data_bytes: bytes, chunk_shape: tuple=None):
        """

        """
        if self.transcoder is not None:
            data_encoded = self.transcoder.from_bytes(data_bytes, chunk_shape)
            data_decoded = self.transcoder.decode(data_encoded)
        else:
            data_decoded = np.frombuffer(bytearray(data_bytes), dtype=self.np_dtype).reshape(chunk_shape)

        return data_decoded


class Float(DTypeTranscoder):
    """

    """

class Integer(DTypeTranscoder):
    """

    """


class DateTime(DTypeTranscoder):
    """

    """


class String(DataType):
    """

    """
    def __init__(self):
        """

        """
        self.kind = 'T'
        self.np_dtype = np.dtypes.StringDType(na_object=None)
        self.precision = None
        self.transcoder = None
        self.itemsize = None
        self.name = self.np_dtype.name

    def dumps(self, data: np.ndarray):
        """

        """
        return msgspec.msgpack.encode(data.tolist())

    def loads(self, data_bytes: bytes, chunk_shape: tuple=None):
        """

        """
        data = np.asarray(msgspec.msgpack.decode(data_bytes), dtype=self.np_dtype).reshape(chunk_shape)

        return data

    def __repr__(self):
        """

        """
        return self.name


class Categorical:
    """
    This class and dtype should be similar to the pandas categorical dtype. Preferably, all string arrays should be cat dtypes. In the CF conventions, this is equivelant to `flags <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#flags>`_. The CF conventions of assigning the attrs flag_values and flag_meanings should be used for compatability.
    As in the CF conventions, two python lists can be used (one int in increasing order from 0 as the index, and the other as the string values). The string values would have no sorted order. They would be assigned the int index as they are assigned.
    This class should replace the fixed-length numpy unicode class for data variables.
    At the moment, I don't want to implement this until I've got the rest of the package implemented.
    """
    # TODO

###################################################
### Functions


def compute_int_and_offset(min_value: Union[int, float, np.number], max_value: Union[int, float, np.number], precision: int):
    """
    Computes the integer byte size and offset for a float using a min value, max value, and the precision. A value of 0 is set asside for the fillvalue.
    Parameters
    ----------
    min_value : int or float
        The min value of the dataset.
    max_value : int or float
        The max value of the dataset.
    precision : int
        The precision of the float data. This should be an int that you would pass to round.

    Returns
    -------
    int itemsize, offset
    """
    if max_value < min_value:
        raise ValueError('max_value must be larger than min_value')

    factor = 10**precision
    max_value_int = int(round(max_value * factor))
    min_value_int = int(round(min_value * factor))
    data_range = max_value_int - min_value_int + 1

    ## Determine offset
    offset = 1 - min_value_int

    ## Determine int byte size
    int_byte_size = None
    for i in (1, 2, 4, 8):
        max_int = 256**i
        if data_range < max_int:
            int_byte_size = i
            break

    return int_byte_size, offset


def parse_np_dtypes(np_dtype, precision: int=None, min_value: float | int=None, max_value: float | int=None, dtype_encoded: str=None, offset: int=None):
    """

    """
    np_name = np_dtype.name.lower()
    if np_name == 'bool':
        dtype1 = Bool(np_dtype)
    elif 'int' in np_name:
        if dtype_encoded is not None and offset is not None:
            transcoder = Transcoder(np_dtype.name, dtype_encoded, 0, offset)
        elif isinstance(min_value, (int, np.integer)) and isinstance(max_value, (int, np.integer)):
            int_byte_size, offset = compute_int_and_offset(min_value, max_value, 0)
            transcoder = Transcoder(np_dtype.name, f'u{int_byte_size}', 0, offset)
        else:
            transcoder = None

        dtype1 = Integer(np_dtype, None, transcoder)
    elif 'datetime' in np_name:
        if dtype_encoded is not None and offset is not None:
            transcoder = Transcoder(np_dtype.name, dtype_encoded, 0, offset)
        elif isinstance(min_value, (str, np.datetime64)) and isinstance(max_value, (str, np.datetime64)):
            min_value_t = np.array(min_value, dtype=np_dtype)
            max_value_t = np.array(max_value, dtype=np_dtype)
            int_byte_size, offset = compute_int_and_offset(min_value_t.astype(int), max_value_t.astype(int), 0)
            transcoder = Transcoder(np_dtype.name, f'u{int_byte_size}', 0, offset)
        else:
            transcoder = None

        # print(datetime_to_int)
        dtype1 = DateTime(np_dtype, None, transcoder)
    elif 'float' in np_name:
        if dtype_encoded is not None and offset is not None:
            transcoder = Transcoder(np_dtype.name, dtype_encoded, precision, offset)
        elif isinstance(min_value, (int, float, np.number)) and isinstance(max_value, (int, float, np.number)) and isinstance(precision, int):
            int_byte_size, offset = compute_int_and_offset(min_value, max_value, precision)
            transcoder = Transcoder(np_dtype.name, f'u{int_byte_size}', precision, offset)
        else:
            transcoder = None

        dtype1 = Float(np_dtype, precision, transcoder)

    elif 'str' in np_name:
        dtype1 = String()
    else:
        raise NotImplementedError(f'The dtype {np_name} is not implemented.')

    return dtype1


def dtype(name: str | np.dtype, precision: int=None, min_value: float | int | str | np.datetime64=None, max_value: float | int | str | np.datetime64=None, dtype_encoded: str=None, offset: int=None):
    """
    Function to initialise a cfdb DataType. Data Types in cfdb not only describe the data type that the user's data is in, but also how the data is serialised to bytes.

    Parameters
    ----------
    name: str or np.dtype
        The name of the data type. It can either be a string name or a np.dtype. Geometry data types do not exist in numpy, so they must be a string.
    precision: int or None
        The precision of the data. Only applies to Geometry and float objects. This is essentially the value that you'd pass to the round function.
    min_value: int, float, str, np.dtaetime64, or None
        The minimum possible value of the data. Along with the max_value and precision, this helps to shrink the data when serialising to bytes. Only applies to floats and DateTime dtypes.
    max_value: int, float, str, np.dtaetime64, or None
        The maximum possible value of the data. See min_value for description.

    Returns
    -------
    cfdb.DataType
    """
    if isinstance(name, str):
        name1 = name.lower()
        if name1 in ('point', 'line', 'linestring', 'polygon'):
            if name1 == 'point':
                dtype1 = Point(precision)
            elif name1 in ('line', 'linestring'):
                dtype1 = LineString(precision)
            else:
                dtype1 = Polygon(precision)

        elif 'str' in name1:
            dtype1 = String()
        else:
            np_dtype = np.dtype(name)
            dtype1 = parse_np_dtypes(np_dtype, precision, min_value, max_value, dtype_encoded, offset)

    elif isinstance(name, np.dtype):
        dtype1 = parse_np_dtypes(name, precision, min_value, max_value, dtype_encoded, offset)
    else:
        raise TypeError('name must be either a string or a np.dtype.')

    return dtype1






















































































