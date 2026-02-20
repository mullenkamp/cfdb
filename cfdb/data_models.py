#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 09:23:18 2025

@author: mike
"""
import msgspec
from typing import Dict, Tuple, Union

from cfdb_models.data_models import Type, Compressor, Axis, DataType


###################################################
### Models


class DataVariable(msgspec.Struct, tag='data_var'):
    """

    """
    chunk_shape: Tuple[int, ...]
    coords: Tuple[str, ...]
    dtype: DataType
    units: Union[str, None] = None


class CoordinateVariable(msgspec.Struct, tag='coord'):
    """

    """
    shape: Tuple[int, ...]
    chunk_shape: Tuple[int, ...]
    dtype: DataType
    origin: Union[int, None] = None
    step: Union[float, int, None] = None
    auto_increment: bool = False
    axis: Union[Axis, None] = None
    units: Union[str, None] = None


class SysMeta(msgspec.Struct):
    """

    """
    dataset_type: Type
    compression: Compressor
    compression_level: int
    variables: Dict[str, Union[DataVariable, CoordinateVariable]] = {}
    crs: Union[str, None] = None

    # def __post_init__(self):


























































































