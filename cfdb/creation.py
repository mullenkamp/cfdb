#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 17:08:09 2025

@author: mike
"""
import numpy as np
from typing import Set, Optional, Dict, Tuple, List, Union, AnyStr
import pyproj

from . import utils, dtypes, data_models, support_classes as sc
# import utils, dtypes, data_models, support_classes as sc

#################################################


def make_inner_coord_method(var_name):
    def method(self, data: np.ndarray | None = None, step: int | float | bool=True, **kwargs):
        name, var_params, attrs = utils.get_var_params(var_name, kwargs)

        if isinstance(data, np.ndarray):
            if 'M8' in data.dtype.str:
                var_params['dtype'] = data.dtype

        coord = self.generic(name, data, step=step, **var_params)
        coord.attrs.update(attrs)

        return coord

    return method


def create_coord_methods(var_names):
    """
    Function to create the default variables in a creation class via a decorator.
    """
    def decorator(cls):
        for var_name in var_names:

            doc = f"""
            Create a {var_name} coordinate. The standard dtype and attributes will be assigned. See the generic method for all of the available parameters.

            Parameters
            ----------
            data: np.ndarray or None
                Optionally provide coordintate data as an np.ndarray.
            step: int, float, or bool
                If the coordinate data is regular (hourly for example), then assign a step to ensure the coordinate will always stay regular. False will not set a step, while True will attempt to figure out the step from the input data (if passed).
            kwargs
                Any kwargs that can be passed to the generic coordinate method.

            Returns
            -------
            Coordinate
            """
            method = make_inner_coord_method(var_name)
    
            method.__name__ = var_name
            method.__doc__ = doc
            setattr(cls, var_name, method)
    
        return cls

    return decorator


def make_inner_data_var_method(var_name):
    def method(self, coords: Tuple[str], **kwargs):
        name, var_params, attrs = utils.get_var_params(var_name, kwargs)

        data_var = self.generic(name, coords, **var_params)
        data_var.attrs.update(attrs)

        return data_var

    return method


def create_data_var_methods(var_names):
    """
    Function to create the default variables in a creation class via a decorator.
    """
    def decorator(cls):
        for var_name in var_names:

            doc = f"""
            Create a {var_name} data variable. The standard dtype and attributes will be assigned. See the generic method for all of the available parameters.

            Parameters
            ----------
            coords: tuple of str
                The coordinate names in the order of the dimensions. The coordinate must already exist.
            kwargs
                Any kwargs that can be passed to the generic data variable method.

            Returns
            -------
            Data Variable
            """
            method = make_inner_data_var_method(var_name)
    
            method.__name__ = var_name
            method.__doc__ = doc
            setattr(cls, var_name, method)
    
        return cls

    return decorator


class CRS:
    """

    """
    def __init__(self, dataset):
        """

        """
        self._dataset = dataset


    def from_user_input(self, crs: str | int | pyproj.CRS, x_coord: str=None, y_coord: str=None, xy_coord: str=None):
        """

        """
        ## Check coords
        coord_names = self._dataset.coord_names
        if isinstance(xy_coord, str):
            if xy_coord not in coord_names:
                raise ValueError(f'{xy_coord} not in coords: {coord_names}')
            coord = self._dataset[xy_coord]
            if coord.dtype.kind != 'G':
                raise TypeError(f'{xy_coord} must be a Geometry dtype.')

            self._dataset._sys_meta.variables[xy_coord].axis = data_models.Axis('xy')
        else:
            if x_coord not in coord_names:
                raise ValueError(f'{x_coord} not in coords: {coord_names}')
            if y_coord not in coord_names:
                raise ValueError(f'{y_coord} not in coords: {coord_names}')
            self._dataset._sys_meta.variables[x_coord].axis = data_models.Axis('x')
            self._dataset._sys_meta.variables[y_coord].axis = data_models.Axis('y')

        ## Parse crs
        crs0 = pyproj.CRS.from_user_input(crs)

        ## Update the metadata
        self._dataset._sys_meta.crs = crs0.to_string()

        self._dataset.crs = crs0 # Probably needs to change in the future...

        return crs0


@create_coord_methods(var_names=('time', 'lat', 'lon', 'height', 'altitude', 'x', 'y', 'point', 'line', 'polygon'))
class Coord:
    """

    """
    def __init__(self, dataset):
        """

        """
        self._dataset = dataset


    def generic(self, name: str, data: np.ndarray | None = None, dtype: str | np.dtype | dtypes.DataType | None = None, chunk_shape: Tuple[int] | None = None, step: int | float | bool=False, axis: str=None):
        """
        The generic method to create a coordinate.

        Parameters
        ----------
        name: str
            The name of the coordinate. It must be unique and follow the `CF conventions for variables names <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#_naming_conventions>`_.
        data: np.ndarray or None
            Data to be added after creation. The length and dtype of the data will override other parameters.
        dtype: str, np.dtype, Dtype, or None
            The name of the data type. It can either be a string name, a np.dtype, or a DataType. If name is a string, then it must correspond to a numpy dtype for the decoding except for geometry dtypes.
        chunk_shape: tuple of ints or None
            The chunk shape that the data will be stored as. If None, then it will be estimated. The estimated chunk shape will be optimally estimated to make it efficient to rechunk later.
        step: int, float, or bool
            If the coordinate data is regular (hourly for example), then assign a step to ensure the coordinate will always stay regular. False will not set a step, while True will attempt to figure out the step from the input data (if passed).
        axis: str or None
            The physical axis representation of the coordinate. I.e. x, y, z, t. There cannot be duplicate axes in coordinates.

        Returns
        -------
        cfdb.Coordinate
        """
        if isinstance(axis, str):
            axis = axis.lower()
            axis1 = data_models.Axis(axis)

        for var_name, var in self._dataset._sys_meta.variables.items():
            if name == var_name:
                raise ValueError(f"Dataset already contains the variable {name}.")
            if isinstance(axis, str):
                if var.axis == axis1:
                    raise ValueError(f"axis {axis} already exists.")

        # print(params)

        name, var = utils.parse_coord_inputs(self._dataset._sys_meta.dataset_type, name, data, chunk_shape, dtype, step=step, axis=axis)

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
        # if coord.dtype_decoded.kind == 'M':
        #     coord.attrs['units'] = utils.parse_cf_time_units(coord.dtype_decoded)
        #     coord.attrs['calendar'] = 'proleptic_gregorian'

        return coord


    def like(self, name: str, coord: Union[sc.Coordinate, sc.CoordinateView], copy_data=False):
        """
        Create a Coordinate based on the parameters of another Coordinate. A new unique name must be passed.
        """
        if copy_data:
            data = coord.data
        else:
            data = None

        new_coord = self.generic(name, data, dtype=coord.dtype, chunk_shape=coord.chunk_shape, step=coord.step, axis=coord.axis)

        return new_coord


    # def xy_from_crs(self, crs: Union[str, int, pyproj.CRS], x_coord: str=None, y_coord: str=None, x_data: np.ndarray | None = None, y_data: np.ndarray | None = None, **kwargs):
    #     """

    #     """
    #     ## Parse crs
    #     crs0 = pyproj.CRS.from_user_input(crs)

    #     coord_dict = {}
    #     for axis in crs0.axis_info:
    #         abbrev = axis.abbrev
    #         # units = axis.unit_name
    #         direction = axis.direction
    #         if direction == 'north':
    #             if not isinstance(y_coord, str):
    #                 y_coord = utils.crs_name_dict[abbrev]
    #             if abbrev == 'Lat':
    #                 name, var_params, dtype, attrs = utils.get_var_params('latitude')
    #             else:
    #                 name, var_params, dtype, attrs = utils.get_var_params('y')
    #             coord_dict[name] = (y_coord, var_params, dtype, attrs)
    #         elif direction == 'east':
    #             if not isinstance(x_coord, str):
    #                 x_coord = utils.crs_name_dict[abbrev]
    #             if abbrev == 'Lon':
    #                 name, var_params, dtype, attrs = utils.get_var_params('longitude')
    #             else:
    #                 name, var_params, dtype, attrs = utils.get_var_params('x')
    #             coord_dict[name] = (x_coord, var_params, dtype, attrs)

    #     ## Create coordinates
    #     for name in coord_dict:
    #         coord_name, var_params, dtype, attrs = coord_dict[name]
    #         if name == 'latitude':
    #             _ = self.generic(coord_name, y_data, dtype=dtype, **kwargs)
    #         elif name == 'longitude':
    #             _ = self.generic(coord_name, x_data, dtype=dtype, **kwargs)

    #     ## Update the metadata for crs
    #     self._dataset._sys_meta.crs = crs0.to_string()
    #     self._dataset._sys_meta.variables[x_coord].axis = data_models.Axis('x')
    #     self._dataset._sys_meta.variables[y_coord].axis = data_models.Axis('y')

    #     self._dataset.crs = crs0 # Probably needs to change in the future...

    #     return crs0


@create_data_var_methods(var_names=('precip', 'air_temp', 'wind_speed', 'wind_direction', 'relative_humidity', 'dew_temp', 'soil_temp', 'lwe_soil_moisture'))
class DataVar:
    """

    """
    def __init__(self, dataset):
        """

        """
        self._dataset = dataset


    def generic(self, name: str, coords: Tuple[str], dtype: str | np.dtype | dtypes.DataType, chunk_shape: Tuple[int] | None = None):
        """
        The generic method to create a Data Variable.

        Parameters
        ----------
        name: str
            The name of the coordinate. It must be unique and follow the `CF conventions for variables names <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#_naming_conventions>`_.
        coords: tuple of str
            The coordinate names in the order of the dimensions. The coordinate must already exist.
        dtype: str, np.dtype, Dtype, or None
            The name of the data type. It can either be a string name, a np.dtype, or a DataType. If name is a string, then it must correspond to a numpy dtype for the decoding except for geometry dtypes.
        chunk_shape: tuple of ints or None
            The chunk shape that the data will be stored as. If None, then it will be estimated. The estimated chunk shape will be optimally estimated to make it efficient to rechunk later.

        Returns
        -------
        cfdb.DataVariable
        """
        ## Check base inputs
        name, var = utils.parse_var_inputs(self._dataset._sys_meta, name, coords, dtype, chunk_shape)

        ## Var init process
        self._dataset._sys_meta.variables[name] = var

        ## Init Data var
        data_var = sc.DataVariable(name, self._dataset)

        self._dataset._var_cache[name] = data_var

        ## Add attributes to datetime vars
        # if data_var.dtype_decoded.kind == 'M':
        #     data_var.attrs['units'] = utils.parse_cf_time_units(data_var.dtype_decoded)
        #     data_var.attrs['calendar'] = 'proleptic_gregorian'

        return data_var


    def like(self, name: str, data_var: Union[sc.DataVariable, sc.DataVariableView]):
        """
        Create a Data Variable based on the parameters of another Data Variable. A new unique name must be passed.
        """
        new_data_var = self.generic(name, data_var.coord_names, dtype=data_var.dtype, chunk_shape=data_var.chunk_shape)

        return new_data_var


class Creator:
    """

    """
    def __init__(self, dataset):
        """

        """
        self.coord = Coord(dataset)
        self.data_var = DataVar(dataset)
        self.crs = CRS(dataset)




























































































