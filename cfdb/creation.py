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
    def __init__(self, blt_file, sys_meta, finalizers, var_cache, compressor):
        """

        """
        self._blt = blt_file
        self._sys_meta = sys_meta
        self._finalizers = finalizers
        self._var_cache = var_cache
        self._compressor = compressor


    def generic(self, name: str, data: np.ndarray | None = None, dtype_decoded: str | np.dtype | None = None, dtype_encoded: str | np.dtype | None = None, chunk_shape: Tuple[int] | None = None, fillvalue: Union[int, float, str] = None, scale_factor: Union[float, int, None] = None, add_offset: Union[float, int, None] = None, step: int | float | bool=False):
        """
        The generic method to create a coordinate.
        """
        if name in self._sys_meta.variables:
            raise ValueError(f"Dataset already contains the variable {name}.")

        # print(params)

        name, var = utils.parse_coord_inputs(name, data, chunk_shape, dtype_decoded, dtype_encoded, fillvalue, scale_factor, add_offset, step=step)

        ## Var init process
        self._sys_meta.variables[name] = var

        ## Init Coordinate
        coord = sc.Coordinate(name, self._blt, self._sys_meta, self._compressor, self._finalizers)
        # coord.attrs.update(utils.default_attrs['lat'])

        ## Add data if it has been passed
        if isinstance(data, np.ndarray):
            coord.append(data)

        self._var_cache[name] = coord

        return coord


    def latitude(self, data: np.ndarray | None = None, step: int | float | bool=False, **kwargs):
        """

        """
        params = utils.default_params['lat']
        params.update(kwargs)

        if params['name'] in self._sys_meta.variables:
            raise ValueError(f"Dataset already contains the variable {params['name']}.")

        # print(params)

        name, var = utils.parse_coord_inputs(data=data, step=step, **params)

        ## Var init process
        self._sys_meta.variables[name] = var

        ## Init Coordinate
        coord = sc.Coordinate(name, self._blt, self._sys_meta, self._compressor, self._finalizers)
        coord.attrs.update(utils.default_attrs['lat'])

        ## Add data if it has been passed
        if isinstance(data, np.ndarray):
            coord.append(data)

        self._var_cache[name] = coord

        return coord


    def longitude(self, data: np.ndarray | None = None, step: int | float | bool=False, **kwargs):
        """

        """
        params = utils.default_params['lon']
        params.update(kwargs)

        if params['name'] in self._sys_meta.variables:
            raise ValueError(f"Dataset already contains the variable {params['name']}.")

        name, var = utils.parse_coord_inputs(data=data, step=step, **params)

        ## Var init process
        self._sys_meta.variables[name] = var

        ## Init Coordinate
        coord = sc.Coordinate(name, self._blt, self._sys_meta, self._compressor, self._finalizers)
        coord.attrs.update(utils.default_attrs['lon'])

        ## Add data if it has been passed
        if isinstance(data, np.ndarray):
            coord.append(data)

        self._var_cache[name] = coord

        return coord


    def time(self, data: np.ndarray | None = None, step: int | float | bool=False, **kwargs):
        """

        """
        params = utils.default_params['time']
        params.update(kwargs)

        if params['name'] in self._sys_meta.variables:
            raise ValueError(f"Dataset already contains the variable {params['name']}.")

        name, var = utils.parse_coord_inputs(data=data, step=step, **params)

        ## Var init process
        self._sys_meta.variables[name] = var

        ## Init Coordinate
        coord = sc.Coordinate(name, self._blt, self._sys_meta, self._compressor, self._finalizers)
        coord.attrs.update(utils.default_attrs['time'])

        ## Add data if it has been passed
        if isinstance(data, np.ndarray):
            coord.append(data)

        self._var_cache[name] = coord

        return coord


    def height(self, data: np.ndarray | None = None, step: int | float | bool=False, **kwargs):
        """

        """
        params = utils.default_params['height']
        params.update(kwargs)

        if params['name'] in self._sys_meta.variables:
            raise ValueError(f"Dataset already contains the variable {params['name']}.")

        name, var = utils.parse_coord_inputs(data=data, step=step, **params)

        ## Var init process
        self._sys_meta.variables[name] = var

        ## Init Coordinate
        coord = sc.Coordinate(name, self._blt, self._sys_meta, self._compressor, self._finalizers)
        coord.attrs.update(utils.default_attrs['height'])

        ## Add data if it has been passed
        if isinstance(data, np.ndarray):
            coord.append(data)

        self._var_cache[name] = coord

        return coord


    def altitude(self, data: np.ndarray | None = None, step: int | float | bool=False, **kwargs):
        """

        """
        params = utils.default_params['altitude']
        params.update(kwargs)

        if params['name'] in self._sys_meta.variables:
            raise ValueError(f"Dataset already contains the variable {params['name']}.")

        name, var = utils.parse_coord_inputs(data=data, step=step, **params)

        ## Var init process
        self._sys_meta.variables[name] = var

        ## Init Coordinate
        coord = sc.Coordinate(name, self._blt, self._sys_meta, self._compressor, self._finalizers)
        coord.attrs.update(utils.default_attrs['altitude'])

        ## Add data if it has been passed
        if isinstance(data, np.ndarray):
            coord.append(data)

        self._var_cache[name] = coord

        return coord


class DataVar:
    """

    """
    def __init__(self, blt_file, sys_meta, finalizers, var_cache, compressor):
        """

        """
        self._blt = blt_file
        self._sys_meta = sys_meta
        self._finalizers = finalizers
        self._var_cache = var_cache
        self._compressor = compressor


    def generic(self, name: str, coords: Tuple[str], dtype_decoded: str | np.dtype, dtype_encoded: str | np.dtype | None = None, chunk_shape: Tuple[int] | None = None, fillvalue: Union[int, float, str] = None, scale_factor: Union[float, int, None] = None, add_offset: Union[float, int, None] = None):
        """
        The generic method to create a Data Variable.
        """
        ## Check base inputs
        name, var = utils.parse_var_inputs(self._sys_meta, name, coords, chunk_shape, dtype_decoded, dtype_encoded, fillvalue, scale_factor, add_offset)

        ## Var init process
        self._sys_meta.variables[name] = var

        ## Init Data var
        data_var = sc.DataVariable(name, self._blt, self._sys_meta, self._compressor, self._finalizers)

        self._var_cache[name] = data_var

        return data_var









class Creator:
    """

    """
    def __init__(self, blt_file, sys_meta, finalizers, var_cache, compressor):
        """

        """
        self.coord = Coord(blt_file, sys_meta, finalizers, var_cache, compressor)
        self.data_var = DataVar(blt_file, sys_meta, finalizers, var_cache, compressor)




























































































