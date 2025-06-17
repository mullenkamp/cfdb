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


    def generic(self, name: str, data: np.ndarray | None = None, shape: Tuple[int] | None = None, chunk_shape: Tuple[int] | None = None, dtype_decoded: str | np.dtype | None = None, dtype_encoded: str | np.dtype | None = None, fillvalue: Union[int, float, str] = None, scale_factor: Union[float, int, None] = None, add_offset: Union[float, int, None] = None, step: int | float | bool=False):
        """
        The generic method to create a coordinate.
        """
        name, data, shape, chunk_shape, enc = utils.parse_var_inputs(name, data, shape, chunk_shape, dtype_decoded, dtype_encoded, fillvalue, scale_factor, add_offset, step)

        ## Var init processes
        utils.coord_init(name, data, shape, chunk_shape, enc, self._sys_meta, self._blt, self._compressor, step)

        ## Init Coordinate
        coord = sc.Coordinate(self._blt, name, self._sys_meta, self._finalizers)
        self._var_cache[name] = coord

        return coord


    def latitude(self, name: str | None=None, data: np.ndarray | None = None, shape: Tuple[int] | None = None, chunk_shape: Tuple[int] | None = None, step: int | float | bool=False, **kwargs):
        """

        """
        encodings = utils.default_encodings['lat']
        encodings.update(kwargs)

        if not isinstance(name, str):
            name = 'latitude'

        name, data, shape, chunk_shape, enc, step = utils.parse_var_inputs(True, name, data, shape, chunk_shape, step=step, **encodings)

        ## Var init processes
        utils.coord_init(name, data, shape, chunk_shape, enc, self._sys_meta, self._blt, self._compressor, step)

        ## Init Coordinate
        coord = sc.Coordinate(self._blt, name, self._sys_meta, self._finalizers)
        coord.attrs.update(utils.default_attrs['lat'])

        self._var_cache[name] = coord

        return coord


    def longitude(self, name: str | None=None, data: np.ndarray | None = None, shape: Tuple[int] | None = None, chunk_shape: Tuple[int] | None = None, step: int | float | bool=False, **kwargs):
        """

        """
        encodings = utils.default_encodings['lon']
        encodings.update(kwargs)

        if not isinstance(name, str):
            name = 'longitude'

        name, data, shape, chunk_shape, enc, step = utils.parse_var_inputs(True, name, data, shape, chunk_shape, step=step, **encodings)

        ## Var init processes
        utils.coord_init(name, data, shape, chunk_shape, enc, self._sys_meta, self._blt, self._compressor, step)

        ## Init Coordinate
        coord = sc.Coordinate(self._blt, name, self._sys_meta, self._finalizers)
        coord.attrs.update(utils.default_attrs['lon'])

        self._var_cache[name] = coord

        return coord

    def time(self, name: str | None=None, data: np.ndarray | None = None, shape: Tuple[int] | None = None, chunk_shape: Tuple[int] | None = None, step: int | float | bool=False, **kwargs):
        """

        """
        encodings = utils.default_encodings['time']
        encodings.update(kwargs)

        if not isinstance(name, str):
            name = 'time'

        name, data, shape, chunk_shape, enc, step = utils.parse_var_inputs(True, name, data, shape, chunk_shape, step=step, **encodings)

        ## Var init processes
        utils.coord_init(name, data, shape, chunk_shape, enc, self._sys_meta, self._blt, self._compressor, step)

        ## Init Coordinate
        coord = sc.Coordinate(self._blt, name, self._sys_meta, self._finalizers)
        coord.attrs.update(utils.default_attrs['time'])

        self._var_cache[name] = coord

        return coord

    def height(self, name: str | None=None, data: np.ndarray | None = None, shape: Tuple[int] | None = None, chunk_shape: Tuple[int] | None = None, step: int | float | bool=False, **kwargs):
        """

        """
        encodings = utils.default_encodings['height']
        encodings.update(kwargs)

        if not isinstance(name, str):
            name = 'height'

        name, data, shape, chunk_shape, enc, step = utils.parse_var_inputs(True, name, data, shape, chunk_shape, step=step, **encodings)

        ## Var init processes
        utils.coord_init(name, data, shape, chunk_shape, enc, self._sys_meta, self._blt, self._compressor, step)

        ## Init Coordinate
        coord = sc.Coordinate(self._blt, name, self._sys_meta, self._finalizers)
        coord.attrs.update(utils.default_attrs['height'])

        self._var_cache[name] = coord

        return coord

    def altitude(self, name: str | None=None, data: np.ndarray | None = None, shape: Tuple[int] | None = None, chunk_shape: Tuple[int] | None = None, step: int | float | bool=False, **kwargs):
        """

        """
        encodings = utils.default_encodings['altitude']
        encodings.update(kwargs)

        if not isinstance(name, str):
            name = 'altitude'

        name, data, shape, chunk_shape, enc, step = utils.parse_var_inputs(True, name, data, shape, chunk_shape, step=step, **encodings)

        ## Var init processes
        utils.coord_init(name, data, shape, chunk_shape, enc, self._sys_meta, self._blt, self._compressor, step)

        ## Init Coordinate
        coord = sc.Coordinate(self._blt, name, self._sys_meta, self._finalizers)
        coord.attrs.update(utils.default_attrs['altitude'])

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


    def generic(self, name: str, coords: Tuple[str], shape: Tuple[int], chunk_shape: Tuple[int] | None = None, dtype_decoded: str | np.dtype | None = None, dtype_encoded: str | np.dtype | None = None, fillvalue: Union[int, float, str] = None, scale_factor: Union[float, int, None] = None, add_offset: Union[float, int, None] = None):
        """
        The generic method to create a coordinate.
        """
        ## Check base inputs
        name, data, shape, chunk_shape, enc = utils.parse_var_inputs(name, None, shape, chunk_shape, dtype_decoded, dtype_encoded, fillvalue, scale_factor, add_offset)

        ## Check coords
        utils.check_coords(coords, shape, self._sys_meta)

        ## Var init processes
        utils.data_var_init(name, coords, shape, chunk_shape, enc, self._sys_meta)

        ## Init Coordinate
        data_var = sc.DataVar(self._blt, name, self._sys_meta, self._finalizers)
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




























































































