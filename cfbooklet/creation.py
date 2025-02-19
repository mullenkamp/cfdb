#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 17:08:09 2025

@author: mike
"""
import numpy as np
from typing import Set, Optional, Dict, Tuple, List, Union, Any
import msgspec

import utils, data_models, rechunker, support_classes as sc

#################################################


class Coord:
    """

    """
    def __init__(self, blt_file, sys_meta, finalizers, var_cache):
        """

        """
        self._blt = blt_file
        self._sys_meta = sys_meta
        self._finalizers = finalizers
        self._var_cache = var_cache


    def generic(self, name: str, data: np.ndarray | None = None, shape: Tuple[int] | None = None, chunk_shape: Tuple[int] | None = None, dtype_decoded: str | np.dtype | None = None, dtype_encoded: str | np.dtype | None = None, fillvalue: Union[int, float, str] = None, scale_factor: Union[float, int, None] = None, add_offset: Union[float, int, None] = None):
        """
        The generic method to create a coordinate.
        """
        name, data, shape, chunk_shape, enc = utils.parse_var_inputs(name, data, shape, chunk_shape, dtype_decoded, dtype_encoded, fillvalue, scale_factor, add_offset)

        ## Update sys_meta
        if name in self._sys_meta.variables:
            raise ValueError(f'Dataset already contains the variable {name}.')

        var = data_models.Variable(shape=shape, chunk_shape=chunk_shape, start_chunk_pos=(0,), coords=(name,), encoding=enc)

        self._sys_meta.variables[name] = var

        if data is not None:
            utils.write_init_data(self._blt, name, var, data)

        ## Init Coordinate
        coord = sc.Coordinate(self._blt, name, self._sys_meta, self._finalizers)
        self._var_cache[name] = coord

        return coord


    def latitude(self, name: str, data: np.ndarray | None = None, shape: Tuple[int] | None = None, chunk_shape: Tuple[int] | None = None, **kwargs):
        """

        """
        encodings = utils.default_encodings['lat']
        encodings.update(kwargs)

        name, data, shape, chunk_shape, enc = utils.parse_var_inputs(name, data, shape, chunk_shape, **encodings)

        ## check sys_meta
        if name in self.sys_meta.variables:
            raise ValueError(f'Dataset already contains the variable {name}.')

        var = data_models.Variable(shape=shape, chunk_shape=chunk_shape, start_chunk_pos=(0,), coords=(name,), encoding=enc)

        self._sys_meta.variables[name] = var

        if data is not None:
            utils.write_init_data(self._blt, name, var, data)

        ## Init Coordinate
        coord = sc.Coordinate(self._blt, name, self._sys_meta, self._finalizers)
        self._var_cache[name] = coord

        return coord












class DataVar:
    """

    """
    def __init__(self, blt_file, sys_meta):
        """

        """









class Creator:
    """

    """
    def __init__(self, blt_file, sys_meta):
        """

        """



























































































