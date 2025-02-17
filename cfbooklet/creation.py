#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 17:08:09 2025

@author: mike
"""
import numpy as np
from typing import Set, Optional, Dict, Tuple, List, Union, Any

import utils

#################################################


class Coord:
    """

    """
    def __init__(self, blt_file, sys_meta):
        """

        """


    def generic(self, name: str, data: np.ndarray | None = None, shape: Tuple[int] | None = None, chunk_shape: Tuple[int] | None = None, dtype_decoded: str | np.dtype | None = None, dtype_encoded: str | np.dtype | None = None, fillvalue: Union[int, float, str] = None, scale_factor: Union[float, int, None] = None, add_offset: Union[float, int, None] = None):
        """
        The generic method to create a coordinate.
        """
        ## Check var name
        if not utils.check_var_name(name):
            raise ValueError(f'{name} is not a valid variable name.')

        ## Check data, shape, and dtype
        if isinstance(data, np.ndarray):
            shape = data.shape
            dtype_decoded = data.dtype
        else:
            if not isinstance(shape, tuple):
                raise ValueError('If data is not passed, then shape must be passed.')
            else:
                if not all([isinstance(i, (int, np.integer)) for i in shape]):
                    raise ValueError('shape must be a tuple of ints.')
            if not isinstance(dtype_decoded, (str, np.dtype)):
                raise ValueError('If data is not passed, then dtype_decoded must be passed.')
            dtype_decoded = np.dtype(dtype_decoded)

        if not isinstance(dtype_encoded, (str, np.dtype)):
            dtype_encoded = dtype_decoded

        ## Fillvalue
        kind = dtype_encoded.kind
        if fillvalue is not None:
            fillvalue_dtype = np.dtype(type(fillvalue))

            if kind == 'u' and fillvalue_dtype.kind == 'i':
                if fillvalue < 0:
                    raise ValueError('The dtype_encoded is an unsigned integer, but the fillvalue is < 0.')
                elif fillvalue_dtype.kind != kind:
                    raise ValueError('The fillvalue dtype is not the same as the dtype_encoded dtype.')
        else:
            if kind == 'u':
                fillvalue = 0
            elif kind == 'f':
                fillvalue = np.nan
            elif kind == 'U':
                fillvalue = ''
            elif kind == 'i':
                fillvalue = utils.fillvalue_dict[dtype_encoded.name]
            elif kind == 'M':
                fillvalue = np.datetime64('nat')
            else:
                raise TypeError('Unknown/unsupported data type.')

        ## Scale and offset
        if scale_factor is None and isinstance(add_offset, (int, float, np.number)):
            scale_factor = 1
        if isinstance(scale_factor, (int, float, np.number)) and add_offset is None:
            add_offset = 0

        if not isinstance(scale_factor, (int, float, np.number)) or isinstance(add_offset, (int, float, np.number)):
            raise ValueError('sclae_factor and add_offset must be either ints or floats.')







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



























































































