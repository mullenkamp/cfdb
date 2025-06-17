"""
Created on 2022-09-30.

@author: Mike K
"""
import h5py
import io
import os
import numpy as np
import xarray as xr
# from time import time
# import numcodecs
import hdf5plugin
from typing import Union, List
import pathlib
import copy

from . import utils
# import utils

##############################################
### Parameters



##############################################
### Functions


###################################################
### Class


class Combine(object):
    """
    Class to load and combine one or more HDF5 data files (or xarray datasets) with optional filters. The class will then export the combined data to an HDF5 file, file object, or xr.Dataset.

    Parameters
    ----------
    data : str, pathlib.Path, io.BytesIO, xr.Dataset, or list of str, pathlib.Path, io.BytesIO, bytes, or xr.Dataset
        The input data need to be a path to HDF5 file(s), BytesIO objects, bytes objects, or xr.Datasets (or some combo of those).
    group : str or None
        The group or group path within the hdf5 file(s) to the datasets.

    Returns
    -------
    H5 instance
    """
    def __init__(self, data: Union[List[Union[str, pathlib.Path, io.BytesIO, xr.Dataset]], Union[str, pathlib.Path, io.BytesIO, xr.Dataset]], group=None):
        """
        Class to load and combine one or more HDF5 data files (or xarray datasets) with optional filters. The class will then export the combined data to an HDF5 file, file object, or xr.Dataset.

        Parameters
        ----------
        data : str, pathlib.Path, io.BytesIO, xr.Dataset, or list of str, pathlib.Path, io.BytesIO, bytes, or xr.Dataset
            The input data need to be a path to HDF5 file(s), BytesIO objects, bytes objects, or xr.Datasets (or some combo of those).
        group : str or None
            The group or group path within the hdf5 file(s) to the datasets.

        Returns
        -------
        H5 instance
        """
        ## Read paths input into the appropriate file objects
        if isinstance(data, list):
            data1 = data
        else:
            data1 = [data]

        ## Get encodings
        encodings = utils.get_encodings(data1, group)

        ## Get attrs
        attrs, global_attrs = utils.get_attrs(data1, group)

        ## Get the extended coords
        coords_dict = utils.extend_coords(data1, encodings, group)

        ## Add the variables as datasets
        vars_dict, is_regular_dict = utils.index_variables(data1, coords_dict, encodings, group)

        ## Assign attributes
        self._files = data1
        self._group = group
        self._coords_dict = coords_dict
        self._data_vars_dict = vars_dict
        self._attrs = attrs
        self._global_attrs = global_attrs
        self._encodings = encodings
        self._is_regular_dict = is_regular_dict


    def _build_empty_ds(self):
        """

        """
        if self._data_vars_dict:

            ## get all of the coords associated with the existing data vars
            all_data_coords = set()
            for ds in self._data_vars_dict:
                for dim in self._data_vars_dict[ds]['dims']:
                    all_data_coords.add(dim)

            ## Create empty xr.Dataset
            data_vars = {}
            for k, v in self._data_vars_dict.items():
                if 'datetime' in v['dtype_decoded']:
                    data_vars[k] = (v['dims'], np.empty(v['shape'], dtype=np.dtype('datetime64[ns]')))
                else:
                    data_vars[k] = (v['dims'], np.empty(v['shape'], dtype=v['dtype_decoded']))

            coords = {}
            for k, v in self._coords_dict.items():
                if k in all_data_coords:
                    coords[k] = utils.decode_data(v, **self._encodings[k])

            xr_ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=self._global_attrs)

            for ds_name, attr in self._attrs.items():
                if ds_name in xr_ds:
                    xr_ds[ds_name].attrs = attr
            for ds_name, enc in self._encodings.items():
                if ds_name in xr_ds:
                    xr_ds[ds_name].encoding = enc

        else:
            xr_ds = xr.Dataset()

        return xr_ds


    def __repr__(self):
        """

        """
        xr_ds = self._build_empty_ds()

        return xr_ds.__repr__()


    def sel(self, selection: dict=None, include_coords: list=None, exclude_coords: list=None, include_data_vars: list=None, exclude_data_vars: list=None):
        """
        Filter the data by a selection, include, and exclude. Returns a new H5 instance. The selection parameter is very similar to xarry's .sel method.

        Parameters
        ----------
        selection : dict
            This filter requires a dict of coordinates using three optional types of filter values. These include slice instances (the best and preferred option), a list/np.ndarray of coordinate values, or a bool np.ndarray of the coordinate data length.
        include_coords : list
            A list of coordinates to include in the output. Only data variables with included coordinates will be included in the output.
        exclude_dims : list
            A list of coordinates to exclude from the output. Only data variables with coordinates that have not been excluded will be included in the output.
        include_data_vars : list
            A list of data variables to include in the output. Only coordinates that have data variables will be included in the output.
        exclude_data_vars : list
            A list of data variables to exclude from the output. Only coordinates that have data variables will be included in the output.

        Returns
        -------
        H5 instance
        """
        c = self.copy()
        if selection is not None:
            utils.filter_coords(c._coords_dict, selection, self._encodings)
            vars_dict, is_regular_dict = utils.index_variables(self._files, c._coords_dict, c._encodings, self._group)

            c._data_vars_dict = vars_dict
            c._is_regular_dict = is_regular_dict

        if include_coords is not None:
            coords_rem_list = []
            for k in list(c._coords_dict.keys()):
                if k not in include_coords:
                    _ = c._coords_dict.pop(k)
                    coords_rem_list.append(k)

            if coords_rem_list:
                for k in list(c._data_vars_dict.keys()):
                    for coord in coords_rem_list:
                        if coord in c._data_vars_dict[k]['dims']:
                            c._data_vars_dict.pop(k)
                            break

        if exclude_coords is not None:
            coords_rem_list = []
            for k in list(c._coords_dict.keys()):
                if k in exclude_coords:
                    _ = c._coords_dict.pop(k)
                    coords_rem_list.append(k)

            if coords_rem_list:
                for k in list(c._data_vars_dict.keys()):
                    for coord in coords_rem_list:
                        if coord in c._data_vars_dict[k]['dims']:
                            c._data_vars_dict.pop(k)
                            break

        if include_data_vars is not None:
            c._data_vars_dict = {k: v for k, v in c._data_vars_dict.items() if k in include_data_vars}

            include_dims = set()
            for k, v in c._data_vars_dict.items():
                include_dims.update(set(v['dims']))

            for k in list(c._coords_dict.keys()):
                if k not in include_dims:
                    _ = c._coords_dict.pop(k)

        if exclude_data_vars is not None:
            c._data_vars_dict = {k: v for k, v in c._data_vars_dict.items() if k not in exclude_data_vars}

            include_dims = set()
            for k, v in c._data_vars_dict.items():
                include_dims.update(set(v['dims']))

            for k in list(c._coords_dict.keys()):
                if k not in include_dims:
                    _ = c._coords_dict.pop(k)

        return c


    def copy(self):
        """
        Deep copy an Combine instance.
        """
        c = copy.deepcopy(self)

        return c


    def coords(self):
        """
        A Summary of the coordinates.
        """
        coords_summ = {}
        for k, v in self._coords_dict.items():
            encs = copy.deepcopy(self._encodings[k])
            coords_summ[k] = {'shape': v.shape}
            coords_summ[k].update(encs)

        return coords_summ


    def data_vars(self):
        """
        A summary of the data variables.
        """
        vars_summ = {}
        for k, v in self._data_vars_dict.items():
            encs = copy.deepcopy(self._encodings[k])
            vars_summ[k] = {k1: v1 for k1, v1 in v.items() if k1 in ['dims', 'shape']}
            vars_summ[k].update(encs)

        return vars_summ


    def variables(self):
        """
        A summary of all coordinates and data variables.
        """
        coords_summ = self.dims()
        vars_summ = self.datasets()

        coords_summ.update(vars_summ)

        return coords_summ


    def to_hdf5(self, output: Union[str, pathlib.Path, io.BytesIO], group=None, chunks=None, unlimited_dims=None, compression='lzf', libver='earliest'):
        """
        Method to output the filtered data to an HDF5 file or file object.

        Parameters
        ----------
        output : str, pathlib.Path, or io.BytesIO
            The output path of the new combined hdf5 file.
        group : str or None
            The group or group path within the hdf5 file to save the datasets.
        chunks : dict of tuples
            The chunks per dataset. Must be a dictionary of dataset names with tuple values of appropriate dimensions. A value of None will perform auto-chunking.
        unlimited_dims : str, list of str, or None
            The dimensions/dimensions that should be assigned as "unlimited" in the hdf5 file.
        compression : str or None
            The compression used for the chunks in the hdf5 files. Must be one of gzip, lzf, zstd, lz4, or None. gzip is compatible with any hdf5 installation (not only h5py), so this should be used if interoperability across platforms is important. lzf is compatible with any h5py installation, so if only python users will need to access these files then this is a better option than gzip. zstd and lz4 require the hdf5plugin python package, but zstd is the best compression option if users have access to the hdf5plugin package. None has no compression and is generally not recommended except in niche situations.
        libver : The hdf5 library version according to h5py. This is for advanced users only. https://docs.h5py.org/en/stable/high/file.html#version-bounding.

        Returns
        -------
        None
        """
        ## Check if there's anything to save
        if self._coords_dict:

            ## Set up initial parameters
            if isinstance(unlimited_dims, str):
                unlimited_dims = [unlimited_dims]
            else:
                unlimited_dims = []

            compressor = utils.get_compressor(compression)

            ## Create new file
            with h5py.File(output, 'w', libver=libver, rdcc_nbytes=3*1024*1024, track_order=True) as nf:

                if isinstance(group, str):
                    nf1 = nf.create_group(group, track_order=True)
                else:
                    nf1 = nf

                ## Add the coords as datasets
                # dim_id = 0
                for coord, arr in self._coords_dict.items():
                    # if coord == 'time':
                    #     break
                    shape = arr.shape
                    dtype = self._encodings[coord]['dtype']

                    maxshape = tuple([s if s not in unlimited_dims else None for s in shape])

                    chunks1 = utils.guess_chunk(shape, maxshape, dtype)

                    if isinstance(chunks, dict):
                        if coord in chunks:
                            chunks1 = chunks[coord]

                    if dtype == 'object':
                        coord_dtype = h5py.string_dtype()
                    else:
                        coord_dtype = dtype

                    ds = nf1.create_dataset(coord, shape, chunks=chunks1, maxshape=maxshape, dtype=coord_dtype, track_order=True, **compressor)

                    ds[:] = arr

                    ds.make_scale(coord)
                    ds.dims[0].label = coord

                    # ds.attrs['_Netcdf4Dimid'] = dim_id
                    # dim_id += 1
                    # ds.attrs['DIMENSION_LABELS'] = coord

                ## Add the variables as datasets
                vars_dict = self._data_vars_dict

                for var_name in vars_dict:
                    shape = vars_dict[var_name]['shape']
                    dims = vars_dict[var_name]['dims']
                    # nc_coords = np.zeros(len(dims), dtype='int32')
                    # nc_labels = np.zeros(len(dims), dtype='object')
                    maxshape = tuple([s if dims[i] not in unlimited_dims else None for i, s in enumerate(shape)])

                    chunks1 = utils.guess_chunk(shape, maxshape, vars_dict[var_name]['dtype'])

                    if isinstance(chunks, dict):
                        if var_name in chunks:
                            chunks1 = chunks[var_name]

                    if len(shape) == 0:
                        chunks1 = None
                        compressor1 = {}
                        vars_dict[var_name]['fillvalue'] = None
                        maxshape = None
                    else:
                        compressor1 = compressor

                    if vars_dict[var_name]['dtype'] == 'object':
                        ds_dtype = h5py.string_dtype()
                    else:
                        ds_dtype = vars_dict[var_name]['dtype']

                    ds = nf1.create_dataset(var_name, shape, chunks=chunks1, maxshape=maxshape, dtype=ds_dtype, fillvalue=vars_dict[var_name]['fillvalue'], track_order=True, **compressor1)

                    for i, dim in enumerate(dims):
                        ds.dims[i].attach_scale(nf1[dim])
                        ds.dims[i].label = dim
                        # dim_id = nf1[dim].attrs['_Netcdf4Dimid']
                        # nc_coords[i] = dim_id
                        # nc_labels[i] = dim

                    # ds.attrs['_Netcdf4Coordinates'] = nc_coords
                    # ds.attrs['_Netcdf4Dimid'] = 4
                    # ds.attrs['DIMENSION_LABELS'] = nc_labels

                    ds_vars = vars_dict[var_name]

                    n_files = len(ds_vars['data'])
                    mean_ds_file_size = utils.product(shape)/n_files

                    # Load data by file if no chunks are assigned
                    if ds.chunks is None:
                        for i in ds_vars['data']:
                            with utils.open_file(self._files[i], self._group) as file:
                                ds_old = file[var_name]

                                if isinstance(ds_old, xr.DataArray):
                                    ds[()] = utils.encode_data(ds_old.values, **self._encodings[var_name])
                                else:
                                    ds[()] = ds_old[()]
                    else:
                        # If files are big and regular fill by file
                        if self._is_regular_dict[var_name] and (mean_ds_file_size > (3 * utils.product(ds.chunks))):
                            utils.fill_ds_by_files(ds, self._files, ds_vars, var_name, self._group, self._encodings)
                        # Otherwise fill by chunk
                        else:
                            utils.fill_ds_by_chunks(ds, self._files, ds_vars, var_name, self._group, self._encodings)

                ## Assign attrs and encodings
                for ds_name, attr in self._attrs.items():
                    if ds_name in nf1:
                        nf1[ds_name].attrs.update(attr)

                for ds_name, encs in self._encodings.items():
                    if ds_name in nf1:
                        for f, enc in encs.items():
                            nf1[ds_name].attrs.update({f: enc})

                # nf1.attrs['_NCProperties'] = b'version=2,hdf5=1.12.2,h5py=3.7.0'
                nf1.attrs.update(self._global_attrs)
                # nf1.attrs.update({'unlimited_dims': ''})

            if isinstance(output, io.BytesIO):
                output.seek(0)

        else:
            print('No data to save')


    def to_xarray(self, **kwargs):
        """
        Save an HDF5 file to an io.BytesIO object which is then opened by xr.open_dataset using the h5netcdf engine.

        Parameters
        ----------
        kwargs
            Any kwargs that can be passed to open_dataset EXCEPT engine.

        Returns
        -------
        xr.Dataset
        """
        if self._coords_dict:
            b1 = io.BytesIO()

            self.to_hdf5(b1)

            xr_ds = xr.open_dataset(b1, engine='h5netcdf', **kwargs)
        else:
            xr_ds = xr.Dataset()

        return xr_ds


## Backwards compatibility
H5 = Combine

################################################
### Convenience functions


def xr_to_hdf5(data: Union[List[xr.Dataset], xr.Dataset], output: Union[str, pathlib.Path, io.BytesIO], group=None, chunks=None, unlimited_dims=None, compression='lzf'):
    """
    Convenience function to take one or more xr.Datasets and output the data to an HDF5 file or file object.

    Parameters
    ----------
    data : xr.Dataset, or list of xr.Dataset
        The input data as xr.Datasets.
    output : str, pathlib.Path, or io.BytesIO
        The output path of the new combined hdf5 file.
    group : str or None
        The group or group path within the hdf5 file to save the datasets.
    chunks : dict of tuples
        The chunks per dataset. Must be a dictionary of dataset names with tuple values of appropriate dimensions. A value of None will perform auto-chunking.
    unlimited_dims : str, list of str, or None
        The dimensions/dimensions that should be assigned as "unlimited" in the hdf5 file.
    compression : str
        The compression used for the chunks in the hdf5 files. Must be one of gzip, lzf, zstd, or None. gzip is compatible with any hdf5 installation (not only h5py), so this should be used if interoperability across platforms is important. lzf is compatible with any h5py installation, so if only python users will need to access these files then this is a better option than gzip. zstd requires the hdf5plugin python package, but is the best compression option if users have access to the hdf5plugin package. None has no compression and is generally not recommended except in niche situations.

    Returns
    -------
    None
    """
    Combine(data).to_hdf5(output, group, chunks, unlimited_dims, compression)














######################################
### Testing
