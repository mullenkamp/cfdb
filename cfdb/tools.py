#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 09:04:49 2025

@author: mike
"""
from typing import List, Union
import pathlib

try:
    import h5netcdf
    import_h5netcdf = True
except ImportError:
    import_h5netcdf = False

from . import main
# import main

#########################################
### Functions


def cfdb_to_netcdf4(cfdb_path: Union[str, pathlib.Path], nc_path: Union[str, pathlib.Path], compression: str='gzip', sel: dict=None, sel_loc: dict=None, include_data_vars: List[str]=None, exclude_data_vars: List[str]=None, **kwargs):
    """
    Simple function to convert a cfdb to a netcdf4. Selection options are also available. The h5netcdf package must be installed to write netcdf4 files.

    Parameters
    ----------
    cfdb_path: str or pathlib.Path
        The source path of the cfdb to be converted.
    nc_path: str or pathlib.Path
        The target path for the netcdf4 file.
    sel: dict
        Selection by coordinate indexes.
    sel_loc: dict
        Selection by coordinate values.
    max_mem: int
        The max memory in bytes if required when coordinates are in decending order (and must be resorted in ascending order).
    kwargs
        Any kwargs that can be passed to the h5netcdf.File function.

    Returns
    -------
    None
    """
    if not import_h5netcdf:
        raise ImportError('h5netcdf must be installed to save files to netcdf4.')

    if (sel is not None) and (sel_loc is not None):
        raise ValueError('Only one of sel or sel_loc can be passed, not both.')

    with main.open_dataset(cfdb_path) as ds:
        if isinstance(sel, dict):
            ds_view = ds.select(sel)
        elif isinstance(sel_loc, dict):
            ds_view = ds.select_loc(sel_loc)
        else:
            ds_view = ds

        ds_view.to_netcdf4(nc_path, compression=compression, include_data_vars=include_data_vars, exclude_data_vars=exclude_data_vars, **kwargs)
