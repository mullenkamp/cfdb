#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 09:23:18 2025

@author: mike
"""
import msgspec
import enum
from typing import Set, Optional, Dict, Tuple, List, Union, Any
# import numpy as np

# import utils

####################################################
### Parameters





###################################################
### Models


class Type(enum.Enum):
    """

    """
    Dataset = 'Dataset'


class Compressor(enum.Enum):
    """

    """
    zstd = 'zstd'
    lz4 = 'lz4'


# class Encoding(msgspec.Struct):
#     """

#     """
#     dtype_encoded: str
#     dtype_decoded: str
#     fillvalue: Union[int, None] = None
#     # fillvalue_decoded: Union[int, None]
#     scale_factor: Union[float, int, None] = None
#     add_offset: Union[float, int, None] = None
#     # units: Union[str, None] = None
#     # calendar: Union[str, None] = None

#     # def encode(self, values):
#     #     return utils.encode_data(np.asarray(values), **self._encoding)

#     # def decode(self, bytes_data):
#     #     return utils.decode_data(bytes_data, **self._encoding)


class DataVariable(msgspec.Struct, tag='data_var'):
    """

    """
    chunk_shape: Tuple[int, ...]
    coords: Tuple[str, ...]
    dtype_encoded: str
    dtype_decoded: str
    fillvalue: Union[int, None] = None
    scale_factor: Union[float, int, None] = None
    add_offset: Union[float, int, None] = None


class CoordinateVariable(msgspec.Struct, tag='coord'):
    """

    """
    shape: Tuple[int, ...]
    chunk_shape: Tuple[int, ...]
    dtype_encoded: str
    dtype_decoded: str
    fillvalue: Union[int, None] = None
    scale_factor: Union[float, int, None] = None
    add_offset: Union[float, int, None] = None
    origin_pos: Union[int, None] = None
    step: Union[float, int, None] = None
    auto_increment: bool = False


class SysMeta(msgspec.Struct):
    """

    """
    object_type: Type
    compression: Compressor
    compression_level: int
    variables: Dict[str, Union[DataVariable, CoordinateVariable]] = {}

    # def __post_init__(self):


























































































