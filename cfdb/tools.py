#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 09:04:49 2025

@author: mike
"""
import numpy as np
import rechunkit
import copy
from typing import List, Union
import pathlib

try:
    import h5netcdf
    import_h5netcdf = True
except ImportError:
    import_h5netcdf = False

from . import utils, main, indexers, support_classes as sc
# import utils, main, indexers, support_classes as sc

##########################################
### Parameters

inv_time_units_dict = {value: key for key, value in utils.time_units_dict.items()}



#########################################
### Functions


class H5DataVarReader:
    """

    """
    def __init__(self, h5_data_var, inverted_coords, shape):
        """

        """
        self.is_inverted = any(inverted_coords)
        self.data_var = h5_data_var
        self.inverted_coords = inverted_coords
        self.shape = shape

    def get(self, slices):
        """

        """
        if self.is_inverted:
            source_slices = tuple(slice(s - cs.stop, s - cs.start) if inverted else cs for inverted, cs, s in zip(self.inverted_coords, slices, self.shape))
            data = np.flip(self.data_var[source_slices], np.nonzero(self.inverted_coords)[0])
        else:
            data = self.data_var[slices]

        return data


def filter_var_names_h5(h5, include_data_vars, exclude_data_vars):
    """

    """
    coord_names_all = set(h5.dims)
    data_var_names_all = set(h5.variables).difference(coord_names_all)

    if include_data_vars is not None:
        if isinstance(include_data_vars, str):
            include_data_vars = [include_data_vars]
        data_var_names = set(include_data_vars)
        if not data_var_names.isubset(data_var_names_all):
            raise ValueError(f'{data_var_names} is not a subset of {data_var_names_all}')
    elif exclude_data_vars is not None:
        if isinstance(exclude_data_vars, str):
            exclude_data_vars = [exclude_data_vars]
        data_var_names = data_var_names_all.difference(set(exclude_data_vars))
    else:
        data_var_names = data_var_names_all

    coord_names = set()
    for data_var_name in data_var_names:
        data_var = h5[data_var_name]
        coord_names.update(data_var.dimensions)

    return data_var_names, coord_names


def parse_attrs(attrs):
    """

    """
    input_params = {}
    for attr, value in copy.deepcopy(attrs).items():
        if attr == 'scale_factor':
            input_params['scale_factor'] = float(attrs.pop(attr))
        elif attr == 'add_offset':
            input_params['add_offset'] = float(attrs.pop(attr))
        elif attr == '_FillValue':
            if value is not None:
                input_params['fillvalue'] = int(attrs.pop(attr))
        elif attr == 'missing_value':
            del attrs['missing_value']
        elif isinstance(value, np.bytes_):
            attrs[attr] = str(value.astype(str))
        elif isinstance(value, np.floating):
            attrs[attr] = float(value)
        elif isinstance(value, np.integer):
            attrs[attr] = int(value)
        elif isinstance(value, np.str_):
            attrs[attr] = str(value)

    return attrs, input_params


def parse_cf_dates(units, dtype_encoded):
    """

    """
    if ' since ' in units:
        freq, start_date = units.split(' since ')
        freq_code = inv_time_units_dict[freq]
        origin_date = np.datetime64(start_date, freq_code)
        unix_date = np.datetime64('1970-01-01', freq_code)
        # origin_diff = (unix_date - origin_date).astype(dtype_encoded)
        units = f'{freq} since {str(unix_date)}'
        if freq_code not in ('M', 'D', 'h', 'm'):
            dtype_encoded = np.dtype('int64')
        dtype_decoded = origin_date.dtype
    else:
        dtype_decoded = dtype_encoded
        origin_date = None

    return units, dtype_decoded, dtype_encoded, origin_date


def netcdf4_to_cfdb(nc_path: Union[str, pathlib.Path], cfdb_path: Union[str, pathlib.Path], sel: dict=None, sel_loc: dict=None, include_data_vars: List[str]=None, exclude_data_vars: List[str]=None, max_mem: int=2**27, **kwargs):
    """
    Simple function to convert a netcdf4 to a cfdb. Selection options are also available. The h5netcdf package must be installed to read netcdf4 files.

    Parameters
    ----------
    nc_path: str or pathlib.Path
        The source netcdf4 file to be converted.
    cfdb_path: str or pathlib.Path
        The target path for the cfdb.
    sel: dict
        Selection by coordinate indexes.
    sel_loc: dict
        Selection by coordinate values.
    max_mem: int
        The max memory in bytes if required when coordinates are in decending order (and must be resorted in ascending order).
    kwargs
        Any kwargs that can be passed to the cfdb.open_dataset function.

    Returns
    -------
    None
    """
    if not import_h5netcdf:
        raise ImportError('h5netcdf must be installed to save files to netcdf4.')

    if (sel is not None) and (sel_loc is not None):
        raise ValueError('Only one of sel or sel_loc can be passed, not both.')

    ## Get the coordinates data
    inverted_coords = []
    # coords_data = {}
    sel_dict = {}
    with main.open_dataset(cfdb_path, 'n', **kwargs) as ds:
        with h5netcdf.File(nc_path, 'r') as h5:
            dims = tuple(h5.dims)

            ## Check the selection inputs
            if isinstance(sel, dict):
                for key in sel:
                    if key not in dims:
                        raise ValueError(f'{key} is not a dimension in the dataset.')
            elif isinstance(sel_loc, dict):
                for key in sel_loc:
                    if key not in dims:
                        raise ValueError(f'{key} is not a dimension in the dataset.')

            data_var_names, coord_names = filter_var_names_h5(h5, include_data_vars, exclude_data_vars)

            for dim in coord_names:
                h5_coord = h5[dim]
                dtype_encoded = h5_coord.dtype
                attrs = dict(h5_coord.attrs)
                attrs, input_params = parse_attrs(attrs)

                if 'scale_factor' in input_params:
                    dtype_decoded = np.dtype('float64')
                elif 'units' in attrs:
                    units, dtype_decoded, dtype_encoded, origin_date = parse_cf_dates(attrs['units'], dtype_encoded)
                    attrs['units'] = units
                else:
                    dtype_decoded = dtype_encoded

                input_params['dtype_decoded'] = dtype_decoded
                input_params['dtype_encoded'] = dtype_encoded

                # chunk_start = (0,)
                shape = h5_coord.shape
                chunk_shape = h5_coord.chunks
                if chunk_shape is None:
                    chunk_shape = rechunkit.guess_chunk_shape(shape, dtype_encoded)

                input_params['chunk_shape'] = chunk_shape

                data = h5_coord[()]
                h5_coord_diff = np.diff(data)
                if h5_coord_diff[0] > 0:
                    order_check = np.all(h5_coord_diff > 0)
                    inverted = False
                else:
                    order_check = np.all(h5_coord_diff < 0)
                    inverted = True

                inverted_coords.append(inverted)

                if not order_check:
                    raise ValueError('Either the coordinate values are not increasing/decreasing or they are not unique.')

                data = h5_coord[()]

                if inverted:
                    data.sort()

                ## Decode data if necessary
                if dtype_decoded.kind == 'M':
                    data = data + origin_date
                elif 'scale_factor' in input_params:
                    if 'add_offset' in input_params:
                        add_offset = input_params['add_offset']
                    else:
                        add_offset = None
                    if 'fillvalue' in input_params:
                        fillvalue = input_params['fillvalue']
                    else:
                        fillvalue = None
                    encoding = sc.Encoding(chunk_shape, dtype_decoded, dtype_encoded, fillvalue, input_params['scale_factor'], add_offset, None)

                    data = encoding.decode(data)

                ## Selection
                if isinstance(sel, dict):
                    if dim in sel:
                        slices = indexers.index_combo_one(sel[dim], (0,), shape, 0)
                        data = data[slices]
                    else:
                        slices = indexers.slice_none((0,), shape, 0)

                elif isinstance(sel_loc, dict):
                    if dim in sel_loc:
                        idx = indexers.loc_index_combo_one(sel_loc[dim], data)
                        slices = indexers.index_combo_one(idx, (0,), shape, 0)
                        data = data[slices]
                    else:
                        slices = indexers.slice_none((0,), shape, 0)
                else:
                    slices = indexers.slice_none((0,), shape, 0)

                sel_dict[dim] = slices

                ## Create coord
                coord = ds.create.coord.generic(dim, data=data, **input_params)
                coord.attrs.update(attrs)

                # coords_data[dim] = {'data': data, 'attrs': attrs, 'input_params': input_params}

            ## Data Vars
            inverted_coords = tuple(inverted_coords)
            # is_inverted = any(inverted_coords)

            for var_name in data_var_names:
                h5_var = h5[var_name]
                dtype_encoded = h5_var.dtype
                attrs = dict(h5_var.attrs)
                attrs, input_params = parse_attrs(attrs)

                if 'scale_factor' in input_params:
                    dtype_decoded = np.dtype('float64')
                elif 'units' in attrs:
                    units, dtype_decoded, dtype_encoded, origin_date = parse_cf_dates(attrs['units'], dtype_encoded)
                    attrs['units'] = units
                else:
                    dtype_decoded = dtype_encoded

                var_sel = tuple(sel_dict[dim] for dim in h5_var.dimensions)

                # chunk_start = tuple(s.start for s in var_sel)
                # shape = tuple(s.stop - s.start for s in var_sel)
                # chunk_start = tuple(0 for i in range(len(h5_var.shape)))
                shape = h5_var.shape
                chunk_shape = h5_var.chunks
                if chunk_shape is None:
                    chunk_shape = rechunkit.guess_chunk_shape(shape, dtype_encoded)

                data_var = ds.create.data_var.generic(var_name, h5_var.dimensions, dtype_decoded=dtype_decoded, dtype_encoded=dtype_encoded, chunk_shape=chunk_shape, **input_params)
                data_var.attrs.update(attrs)

                h5_reader = H5DataVarReader(h5_var, inverted_coords, shape)

                chunks_iter = rechunkit.rechunker(h5_reader.get, shape, dtype_encoded, chunk_shape, chunk_shape, max_mem, var_sel)
                for chunk_slices, encoded_data in chunks_iter:
                    if not np.all(encoded_data == data_var.fillvalue):
                        data_var.set(chunk_slices, encoded_data, False)


                # chunks_iter = rechunkit.chunk_range(chunk_start, shape, chunk_shape)
                # for chunk_slices in chunks_iter:
                #     if is_inverted:
                #         source_slices = tuple(slice(s - cs.stop, s - cs.start) if inverted else cs for inverted, cs, s in zip(inverted_coords, chunk_slices, shape))
                #         data = np.flip(h5_var[source_slices], np.nonzero(inverted_coords)[0])
                #     else:
                #         data = h5_var[chunk_slices]
                #     if not np.all(data == data_var.fillvalue):
                #         # data_var.set_chunk(chunk_slices, data, False)
                #         data_var.set(chunk_slices, data, False)

            ds.attrs.update(dict(h5.attrs))


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


































































