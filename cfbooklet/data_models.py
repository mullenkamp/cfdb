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


class CFBookletTypes(enum.Enum):
    """

    """
    Dataset = 'Dataset'


class Compression(enum.Enum):
    """

    """
    zstd = 'zstd'
    lz4 = 'lz4'


class Encoding(msgspec.Struct):
    """

    """
    dtype_encoded: str
    dtype_decoded: str
    fillvalue: Union[int, float, str]
    scale_factor: Union[float, int, None] = None
    add_offset: Union[float, int, None] = None
    units: Union[str, None] = None
    calendar: Union[str, None] = None

    # def encode(self, values):
    #     return utils.encode_data(np.asarray(values), **self._encoding)

    # def decode(self, bytes_data):
    #     return utils.decode_data(bytes_data, **self._encoding)


class Variable(msgspec.Struct):
    """

    """
    shape: Tuple[int, ...]
    chunk_shape: Tuple[int, ...]
    coords: Tuple[str, ...]
    encoding: Encoding


class SysMeta(msgspec.Struct):
    """

    """
    cfbooklet_type: CFBookletTypes
    compression: Compression
    variables: Dict[str, Variable] = {}

    # def __post_init__(self):


























































































