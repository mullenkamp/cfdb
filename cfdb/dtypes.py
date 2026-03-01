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


# class Transcoder:
#     """
#     Convert a dtype to some kind of integer dtype. Currently, the source dtype can be either a float,
#     """
#     def __init__(self, dtype_decoded: str, dtype_encoded: str, precision: int | float=0, offset: int=0, fillvalue: int=0):
#         """

#         """
#         self.dtype_decoded = np.dtype(dtype_decoded)
#         self.dtype_encoded = np.dtype(dtype_encoded)
#         self.precision = precision
#         self.factor = 10**precision
#         self.offset = offset
#         self.fillvalue = fillvalue

#     @sup
#     def encode(self, data_decoded: np.ndarray):
#         """

#         """
#         if self.factor != 1:
#             data_encoded = (np.round((data_decoded - self.offset)  * self.factor)).astype(self.dtype_encoded)
#         else:
#             data_encoded = (data_decoded - self.offset).astype(self.dtype_encoded)

#         return data_encoded

#     def decode(self, data_encoded: np.ndarray):
#         """

#         """
#         data_decoded = data_encoded.astype(self.dtype_decoded)

#         ## Datetime exception...
#         if self.dtype_decoded.kind == 'M':
#             data_decoded[data_decoded == np.array(0, dtype=self.dtype_decoded)] = np.datetime64('nat')
#         elif self.dtype_decoded.kind == 'f':
#             data_decoded[data_decoded == self.fillvalue] = np.nan

#         if self.factor != 1:
#             data_decoded = (data_decoded / self.factor)  + self.offset
#         else:
#             data_decoded = data_decoded + self.offset

#         return data_decoded

#     def from_bytes(self, data_bytes: bytes, chunk_shape: tuple=None):
#         """

#         """
#         data = np.frombuffer(data_bytes, dtype=self.dtype_encoded).reshape(chunk_shape)

#         return data

#     def to_bytes(self, data_encoded: np.ndarray):
#         """

#         """
#         return data_encoded.tobytes()

#     def to_dict(self):
#         """

#         """
#         dict1 = {
#             'dtype_encoded': self.dtype_encoded.name,
#             'offset': self.offset,
#             }
#         return dict1


############################################
### DTypes


class DataType:
    name: str = None
    kind: str = None
    itemsize: int = None
    dtype_decoded: np.dtype = None
    dtype_encoded: np.dtype = None
    precision: int | float = None
    fillvalue: int = None
    offset: int | float = None

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
        if self.dtype_encoded is not None:
            dict1['dtype_encoded'] = self.dtype_encoded.name
        if self.fillvalue is not None:
            dict1['fillvalue'] = self.fillvalue
        if self.offset is not None:
            dict1['offset'] = self.offset

        return dict1


class FixedLen(DataType):
    def __init__(self, dtype_decoded):
        """

        """
        self.kind = dtype_decoded.kind
        self.itemsize = dtype_decoded.itemsize
        self.dtype_decoded = dtype_decoded
        self.precision = None
        self.name = dtype_decoded.name

    def dumps(self, data: np.ndarray):
        """

        """
        return data.tobytes()

    def loads(self, data_bytes: bytes, chunk_shape: tuple=None):
        """

        """
        data = np.frombuffer(bytearray(data_bytes), dtype=self.dtype_decoded).reshape(chunk_shape)

        return data


class Geometry(DataType):
    """

    """
    def __init__(self, precision: int):
        """

        """
        self.kind = 'G'
        self.dtype_decoded = np.dtypes.ObjectDType()
        self.dtype_encoded = np.dtypes.StringDType(na_object=None)
        self.precision = int(precision)
        self.itemsize = None
        self._decoder = msgspec.msgpack.Decoder()
        self._encoder = msgspec.msgpack.Encoder()

    def __reduce__(self):
        return (self.__class__, (self.precision,))

    def encode(self, data: np.ndarray) -> np.ndarray:
        """
        From shapely geometries to WKT strings.
        """
        strings = shapely.to_wkt(data, rounding_precision=self.precision).astype(self.dtype_encoded)

        return strings


    def to_bytes(self, data: np.ndarray) -> bytes:
        """

        """
        return self._encoder.encode(data.tolist())


    def dumps(self, data: np.ndarray) -> bytes:
        """

        """
        return self.to_bytes(self.encode(data))


    def from_bytes(self, data_bytes: bytes, chunk_shape: tuple=None) -> np.ndarray:
        """

        """
        return np.asarray(self._decoder.decode(data_bytes), dtype=self.dtype_encoded).reshape(chunk_shape)


    def decode(self, data: np.ndarray) -> np.ndarray:
        """
        From WKT strings to shapely geometries.
        """
        data = shapely.from_wkt(data.astype(self.dtype_decoded))

        return data


    def loads(self, data_bytes: bytes, chunk_shape: tuple=None):
        """

        """
        data = self.decode(self.from_bytes(data_bytes, chunk_shape))

        return data


class Point(Geometry):
    name = 'Point'


class LineString(Geometry):
    name = 'LineString'


class Polygon(Geometry):
    name = 'Polygon'


class Bool(FixedLen):
    """

    """


class DTypeTranscoder(DataType):
    def __init__(self, dtype_decoded: np.dtype, dtype_encoded: np.dtype=None, precision: int | float=None, offset: int=None, fillvalue: int=None):
        """

        """
        self.name = dtype_decoded.name
        self.kind = dtype_decoded.kind
        self.itemsize = dtype_decoded.itemsize
        self.dtype_decoded = dtype_decoded
        self.dtype_encoded = dtype_encoded
        self.precision = precision
        if self.precision is None:
            self._factor = None
        else:
            self._factor = 10**precision
        self.offset = offset
        self.fillvalue = fillvalue

    @sup
    def encode(self, data_decoded: np.ndarray):
        """

        """
        if self._factor is None:
            data_encoded = (data_decoded - self.offset).astype(self.dtype_encoded)
        else:
            data_encoded = (np.round((data_decoded - self.offset) * self._factor)).astype(self.dtype_encoded)

        return data_encoded

    def decode(self, data_encoded: np.ndarray):
        """

        """
        data_decoded = data_encoded.astype(self.dtype_decoded)

        ## Datetime exception...
        if self.dtype_decoded.kind == 'M':
            data_decoded[data_decoded == np.array(0, dtype=self.dtype_decoded)] = np.datetime64('nat')
        elif self.dtype_decoded.kind == 'f':
            data_decoded[data_decoded == self.fillvalue] = np.nan

        if self._factor is None:
            data_decoded = data_decoded + self.offset
        else:
            data_decoded = (data_decoded / self._factor)  + self.offset

        return data_decoded

    def from_bytes(self, data_bytes: bytes, chunk_shape: tuple=None):
        """

        """
        data = np.frombuffer(bytearray(data_bytes), dtype=self.dtype_encoded).reshape(chunk_shape)

        return data

    def to_bytes(self, data_encoded: np.ndarray):
        """

        """
        return data_encoded.tobytes()


    def dumps(self, data: np.ndarray):
        """

        """
        if self.dtype_encoded is None:
            if self._factor is None:
                return data.tobytes()
            else:
                return data.round(self.precision).tobytes()

        else:
            data_encoded = self.encode(data)
            return self.to_bytes(data_encoded)

    def loads(self, data_bytes: bytes, chunk_shape: tuple=None):
        """

        """
        if self.dtype_encoded is None:
            data_decoded = np.frombuffer(bytearray(data_bytes), dtype=self.dtype_decoded).reshape(chunk_shape)
        else:
            data_encoded = self.from_bytes(data_bytes, chunk_shape)
            data_decoded = self.decode(data_encoded)

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
        self.name = 'str'
        self.kind = 'T'
        self.dtype_decoded = np.dtypes.StringDType(na_object=None)
        self.precision = None
        self.itemsize = None
        self._decoder = msgspec.msgpack.Decoder()
        self._encoder = msgspec.msgpack.Encoder()

    def __reduce__(self):
        return (String, ())

    def dumps(self, data: np.ndarray):
        """

        """
        return self._encoder.encode(data.tolist())


    def loads(self, data_bytes: bytes, chunk_shape: tuple=None):
        """

        """
        data = np.asarray(self._decoder.decode(data_bytes), dtype=self.dtype_decoded).reshape(chunk_shape)

        return data


# TODO
# class Categorical(DataType):
#     """
#     This class and dtype should be similar to the pandas categorical dtype. Preferably, all string arrays should be cat dtypes. In the CF conventions, this is equivelant to `flags <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#flags>`_. The CF conventions of assigning the attrs flag_values and flag_meanings should be used for compatability.
#     As in the CF conventions, two python lists can be used (one int in increasing order from 0 as the index, and the other as the string values). The string values would have no sorted order. They would be assigned the int index as they are assigned.
#     This class should replace the fixed-length numpy unicode class for data variables.
#     At the moment, I don't want to implement this until I've got the rest of the package implemented.
#     """

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
        The number of decimals of precision of the float data. This should be an int that you would pass to round.

    Returns
    -------
    int itemsize, offset
    """
    if max_value < min_value:
        raise ValueError('max_value must be larger than min_value')

    factor = 10**precision
    # factor = 1/precision
    # max_value_int = int(round(max_value * factor))
    # min_value_int = int(round(min_value * factor))
    max_value_int = max_value * factor
    min_value_int = min_value * factor
    data_range = max_value_int - min_value_int + 1

    ## Determine offset
    offset = min_value - 1

    ## Determine int byte size
    int_byte_size = None
    for i in (1, 2, 4, 8):
        max_int = 256**i
        if data_range <= max_int:
            int_byte_size = i
            break

    if int_byte_size is None:
        raise ValueError('8 bytes is not enough!')

    return int_byte_size, offset


def parse_np_dtypes(dtype_decoded: np.dtype, precision: int=None, min_value: float | int=None, max_value: float | int=None, dtype_encoded: str=None, offset: float | int=None, fillvalue: int=None):
    """

    """
    np_name = dtype_decoded.name.lower()
    if np_name == 'bool':
        dtype1 = Bool(dtype_decoded)
    elif 'int' in np_name:
        if dtype_encoded is not None and offset is not None:
            dtype_encoded = np.dtype(dtype_encoded)
            dtype1 = Integer(dtype_decoded, dtype_encoded, None, offset, None)
        elif isinstance(min_value, (int, np.integer)) and isinstance(max_value, (int, np.integer)):
            int_byte_size, offset = compute_int_and_offset(min_value, max_value, 0)
            dtype_encoded = np.dtype(f'u{int_byte_size}')
            dtype1 = Integer(dtype_decoded, dtype_encoded, None, offset, None)
        else:
            dtype1 = Integer(dtype_decoded)

    elif 'datetime' in np_name:
        if dtype_encoded is not None and offset is not None:
            dtype_encoded = np.dtype(dtype_encoded)
            dtype1 = DateTime(dtype_decoded, dtype_encoded, None, offset, None)
        elif isinstance(min_value, (str, np.datetime64)) and isinstance(max_value, (str, np.datetime64)):
            min_value_t = np.array(min_value, dtype=dtype_decoded)
            max_value_t = np.array(max_value, dtype=dtype_decoded)
            int_byte_size, offset = compute_int_and_offset(min_value_t.astype(int), max_value_t.astype(int), 1)
            dtype_encoded = np.dtype(f'u{int_byte_size}')
            dtype1 = DateTime(dtype_decoded, dtype_encoded, None, offset, None)
        else:
            dtype1 = DateTime(dtype_decoded)

    elif 'float' in np_name:
        if dtype_encoded is not None and offset is not None and fillvalue is not None and precision is not None:
            dtype_encoded = np.dtype(dtype_encoded)
            dtype1 = Float(dtype_decoded, dtype_encoded, precision, offset, fillvalue)
        elif isinstance(min_value, (int, float, np.number)) and isinstance(max_value, (int, float, np.number)) and isinstance(precision, int):
            int_byte_size, offset = compute_int_and_offset(min_value, max_value, precision)
            dtype_encoded = np.dtype(f'u{int_byte_size}')
            dtype1 = Float(dtype_decoded, dtype_encoded, precision, offset, 0)
        else:
            dtype1 = Float(dtype_decoded, None, precision)

    elif 'str' in np_name:
        dtype1 = String()
    else:
        raise NotImplementedError(f'The dtype {np_name} is not implemented.')

    return dtype1


def dtype(name: str | np.dtype | DataType, precision: int=None, min_value: float | int | str | np.datetime64=None, max_value: float | int | str | np.datetime64=None, dtype_encoded: str=None, offset: float | int=None, fillvalue: int=None):
    """
    Function to initialise a cfdb DataType. Data Types in cfdb not only describe the data type that the user's data is in, but also how the data is serialised (and encoded) to bytes.

    Parameters
    ----------
    name: str, np.dtype, or DataType
        The name of the data type. It can either be a string name, a np.dtype, or a DataType. If name is a string, then it must correspond to a numpy dtype for the decoding except for geometry dtypes. Geometry data types do not exist in numpy, so name must be a string of 'point', 'line', 'linestring', or 'polygon'.
    precision: int or None
        The number of decimals of precision of the data. Only applies to DateTime and float objects. This is essentially the value that you'd pass to the round function/method. This must be passed for geometry dtypes.
    min_value: int, float, str, np.datetime64, or None
        The minimum possible value of the data. Along with the max_value and precision, this helps to shrink the data when serialising to bytes. Only applies to floats and DateTime dtypes and will only be used to determine the dtype encoding.
    max_value: int, float, str, np.dtaetime64, or None
        The maximum possible value of the data. See min_value for description.
    dtype_encoded: str or None
        The np.dtype str name to be used in the encoding. Only applies to floats and DateTime dtypes and the offset and fillvalue must also be passed.
    offset: int, float, or None
        The offset when used for encoding floats and DateTime dtypes.
    fillvalue: int or None
        The fillvalue when used for encoding floats and DateTime dtypes.

    Notes
    -----
    When the decoded dtype is a float or DateTime dtype, the data can be encoded to a smaller integer. To determine the appropriate encoding, the precision, min_value and max_value must be passed. If they are not passed, no interger encoding will be used. If the user already knows the resulting dtype_encoded, offset, and fillvalue, then these must be passed (instead of the other three mentioned above) or no integer encoding will be used.

    Geometry data are encoded and decoded via WKT (and converted to/from bytes via msgpack), and currently the precision parameter is required.

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
            dtype_decoded = np.dtype(name)
            dtype1 = parse_np_dtypes(dtype_decoded, precision, min_value, max_value, dtype_encoded, offset, fillvalue)

    elif isinstance(name, np.dtype):
        dtype1 = parse_np_dtypes(name, precision, min_value, max_value, dtype_encoded, offset, fillvalue)
    elif isinstance(name, DataType):
        dtype1 = name
    else:
        raise TypeError('name must be either a string, a np.dtype, or a cfdb.DataType.')

    return dtype1






















































































