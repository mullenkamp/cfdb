#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 17:08:09 2025

@author: mike
"""
import numpy as np
from typing import Set, Optional, Dict, Tuple, List, Union, Any

from . import utils, data_models, support_classes as sc
# import utils, data_models, rechunkit, support_classes as sc

#################################################


class Coord:
    """

    """
    def __init__(self, dataset):
        """

        """
        self._dataset = dataset
        # self._sys_meta = sys_meta
        # self._finalizers = finalizers
        # self._var_cache = var_cache
        # self._compressor = compressor


    def generic(self, name: str, data: np.ndarray | None = None, dtype_decoded: str | np.dtype | None = None, dtype_encoded: str | np.dtype | None = None, chunk_shape: Tuple[int] | None = None, fillvalue: Union[int, float, str] = None, scale_factor: Union[float, int, None] = None, add_offset: Union[float, int, None] = None, step: int | float | bool=False):
        """
        The generic method to create a coordinate.
        """
        if name in self._dataset._sys_meta.variables:
            raise ValueError(f"Dataset already contains the variable {name}.")

        # print(params)

        name, var = utils.parse_coord_inputs(name, data, chunk_shape, dtype_decoded, dtype_encoded, fillvalue, scale_factor, add_offset, step=step)

        ## Var init process
        self._dataset._sys_meta.variables[name] = var

        ## Init Coordinate
        coord = sc.Coordinate(name, self._dataset)
        # coord.attrs.update(utils.default_attrs['lat'])

        ## Add data if it has been passed
        if isinstance(data, np.ndarray):
            coord.append(data)

        self._dataset._var_cache[name] = coord

        ## Add attributes to datetime vars
        if coord.dtype_decoded.kind == 'M':
            coord.attrs['units'] = utils.parse_cf_time_units(coord.dtype_decoded)
            coord.attrs['calendar'] = 'proleptic_gregorian'

        return coord


    def like(self, name: str, coord: Union[sc.Coordinate, sc.CoordinateView], copy_data=False):
        """

        """
        if copy_data:
            data = coord.data
        else:
            data = None

        new_coord = self.generic(name, data, dtype_decoded=coord.dtype_decoded, dtype_encoded=coord.dtype_encoded, chunk_shape=coord.chunk_shape, fillvalue=coord.fillvalue, scale_factor=coord.scale_factor, add_offset=coord.add_offset, step=coord.step)

        return new_coord



    def latitude(self, data: np.ndarray | None = None, step: int | float | bool=False, **kwargs):
        """

        """
        name, params = utils.get_var_params('lat', kwargs)

        # print(params)

        coord = self.generic(name, data, step=step, **params)
        coord.attrs.update(utils.default_attrs['lat'])

        return coord


    def longitude(self, data: np.ndarray | None = None, step: int | float | bool=False, **kwargs):
        """

        """
        name, params = utils.get_var_params('lon', kwargs)

        # print(params)

        coord = self.generic(name, data, step=step, **params)
        coord.attrs.update(utils.default_attrs['lon'])

        return coord


    def time(self, data: np.ndarray | None = None, step: int | float | bool=False, **kwargs):
        """

        """
        name, params = utils.get_var_params('time', kwargs)

        # print(params)

        coord = self.generic(name, data, step=step, **params)
        coord.attrs.update(utils.default_attrs['time'])

        return coord


    def height(self, data: np.ndarray | None = None, step: int | float | bool=False, **kwargs):
        """

        """
        name, params = utils.get_var_params('height', kwargs)

        # print(params)

        coord = self.generic(name, data, step=step, **params)
        coord.attrs.update(utils.default_attrs['height'])

        return coord


    def altitude(self, data: np.ndarray | None = None, step: int | float | bool=False, **kwargs):
        """

        """
        name, params = utils.get_var_params('altitude', kwargs)

        # print(params)

        coord = self.generic(name, data, step=step, **params)
        coord.attrs.update(utils.default_attrs['altitude'])

        return coord


class DataVar:
    """

    """
    def __init__(self, dataset):
        """

        """
        self._dataset = dataset
        # self._sys_meta = sys_meta
        # self._finalizers = finalizers
        # self._var_cache = var_cache
        # self._compressor = compressor


    def generic(self, name: str, coords: Tuple[str], dtype_decoded: str | np.dtype, dtype_encoded: str | np.dtype | None = None, chunk_shape: Tuple[int] | None = None, fillvalue: Union[int, float, str] = None, scale_factor: Union[float, int, None] = None, add_offset: Union[float, int, None] = None):
        """
        The generic method to create a Data Variable.
        """
        ## Check base inputs
        name, var = utils.parse_var_inputs(self._dataset._sys_meta, name, coords, dtype_decoded, dtype_encoded, chunk_shape, fillvalue, scale_factor, add_offset)

        ## Var init process
        self._dataset._sys_meta.variables[name] = var

        ## Init Data var
        data_var = sc.DataVariable(name, self._dataset)

        self._dataset._var_cache[name] = data_var

        ## Add attributes to datetime vars
        if data_var.dtype_decoded.kind == 'M':
            data_var.attrs['units'] = utils.parse_cf_time_units(data_var.dtype_decoded)
            data_var.attrs['calendar'] = 'proleptic_gregorian'

        return data_var


    def like(self, name: str, data_var: Union[sc.DataVariable, sc.DataVariableView]):
        """

        """
        new_data_var = self.generic(name, data_var.coord_names, dtype_decoded=data_var.dtype_decoded, dtype_encoded=data_var.dtype_encoded, chunk_shape=data_var.chunk_shape, fillvalue=data_var.fillvalue, scale_factor=data_var.scale_factor, add_offset=data_var.add_offset)

        return new_data_var









class Creator:
    """

    """
    def __init__(self, dataset):
        """

        """
        self.coord = Coord(dataset)
        self.data_var = DataVar(dataset)




























































































