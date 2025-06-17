#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:25:06 2025

@author: mike
"""
import re
import numpy as np
import booklet
from typing import Union
import pathlib
import msgspec
import weakref
import io
import copy
import zstandard as zstd

# from . import utils, indexers
import utils, indexers, data_models, support_classes as sc, creation

############################################
### Parameters

compression_options = ('zstd', 'lz4')
default_n_buckets = 144013


############################################
### Functions




############################################
### Classes


class Dataset:
    """

    """
    def __init__(self, file_path: Union[str, pathlib.Path], flag: str = "r", compression='zstd', **kwargs):
        """
        Compression can be either zstd, lz4, or None. But there's no point in using None.
        """
        if 'n_buckets' not in kwargs:
            kwargs['n_buckets'] = default_n_buckets

        fp = pathlib.Path(file_path)
        fp_exists = fp.exists()
        self._blt = booklet.open(file_path, flag, key_serializer='str', **kwargs)
        self.writable = self._blt.writable

        ## Set/Get system metadata
        if not fp_exists or flag in ('n', 'c'):
            # Checks
            if compression.lower() not in compression_options:
                raise ValueError(f'compression must be one of {compression_options}.')

            self._meta = data_models.SysMeta(cfbooklet_type='Dataset', compression=compression, variables={})
            self._blt.set_metadata(msgspec.to_builtins(self._meta))

        else:
            self._meta = msgspec.convert(self._blt.get_metadata(), data_models.SysMeta)

        self.compression = self._meta.compression
        self._compressor = zstd.ZstdCompressor(level=1)
        self._finalizers = []
        self._finalizers.append(weakref.finalize(self, utils.dataset_finalizer, self._blt, self._meta))

        self.attrs = sc.Attributes(self._blt, '_', self._finalizers)

        self._var_cache = weakref.WeakValueDictionary()

        if self.writable:
            self.create = creation.Creator(self._blt, self._meta, self._finalizers, self._var_cache, self._compressor)


    # @property
    # def attrs(self):
    #     """
    #     Attributes of the dataset.
    #     """
    #     return Attributes(self._blt, '_')

    @property
    def variables(self):
        """
        Return a tuple of all the variables (coord and data variables).
        """
        return tuple(self._meta.variables.keys())

    @property
    def coords(self):
        """
        Return a tuple of all the coordinates.
        """
        return tuple(k for k, v in self._meta.variables.items() if v.is_coord)

    @property
    def data_vars(self):
        """
        Return a tuple of all the data variables.
        """
        return tuple(k for k, v in self._meta.variables.items() if not v.is_coord)


    # def __bool__(self):
    #     """

    #     """
    #     return self._file.__bool__()

    def __iter__(self):
        for key in self._meta.variables:
            yield key

    def __len__(self):
        return len(self._meta.variables)

    def __contains__(self, key):
        return key in self._meta.variables

    def get(self, var_name):
        """

        """
        if not isinstance(var_name, str):
            raise TypeError('var_name must be a string.')

        if var_name not in self._var_cache:
            raise ValueError(f'The Variable {var_name} does not exist.')

        return self._var_cache[var_name]


    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        if isinstance(value, sc.Variable):
            setattr(self, key, value)
        else:
            raise TypeError('Assigned value must be a Variable or Coordinate object.')

    def __delitem__(self, key):
        try:
            if key not in self:
                raise KeyError(key)

            # Check if the object to delete is a coordinate
            # And if it is, check that no variables are attached to it
            if isinstance(self[key], sc.Coordinate):
                for var_name in self.data_vars:
                    if key in self[var_name].coords:
                        raise ValueError(f'{key} is a coordinate of {var_name}. You must delete all variables associated with a coordinate before you can delete the coordinate.')

            # TODO: Needs to be done...
            del self._file[key]
            # delattr(self, key)
        except Exception as err:
            raise err

    def __enter__(self):
        return self

    def __exit__(self, *args):
        # self._file.__exit__()
        self.close()

    def close(self):
        """

        """
        for finalizer in reversed(self._finalizers):
            finalizer()

        # self._file.close()
        # if self.lock_fileno is not None:
        #     fcntl.flock(self.lock_fileno, fcntl.LOCK_UN)
        #     os.close(self.lock_fileno)

    # def sync(self):
    #     """

    #     """
    #     old_meta = msgspec.convert(self._blt.get_metadata(), data_models.SysMeta)
    #     if old_meta != self._meta:
    #         self._blt.set_metadata(msgspec.to_builtins(self._meta))
    #     self._blt.sync()


    def __repr__(self):
        """

        """
        return utils.file_summary(self)


    def intersect(self, coords: dict=None, include_dims: list=None, exclude_dims: list=None, include_variables: list=None, exclude_variables: list=None, **file_kwargs):
        """
        This method should create a "view" on the existing object by default and otpionally create a new file.
        """
        ## Check for coordinate names in input
        dims = np.asarray(self.coords)

        if coords is not None:
            keys = tuple(coords.keys())
            for key in keys:
                if key not in dims:
                    raise KeyError(f'{key} is not in the coordinates.')

        if include_dims is not None:
            include_dims_check = np.isin(include_dims, dims)
            if not include_dims_check.all():
                no_dims = ', '.join(include_dims[np.where(include_dims_check)[0].tolist()])
                raise KeyError(f'{no_dims} are not in dims.')

        if exclude_dims is not None:
            exclude_dims_check = np.isin(exclude_dims, dims)
            if not exclude_dims_check.all():
                no_dims = ', '.join(exclude_dims[np.where(exclude_dims_check)[0].tolist()])
                raise KeyError(f'{no_dims} are not in dims.')

        ## Check if variables exist
        variables = np.array(self.data_vars)

        if include_variables is not None:
            include_variables_check = np.isin(include_variables, variables)
            if not include_variables_check.all():
                no_variables = ', '.join(include_variables[np.where(include_variables_check)[0].tolist()])
                raise KeyError(f'{no_variables} are not in variables.')

        if exclude_variables is not None:
            exclude_variables_check = np.isin(exclude_variables, variables)
            if not exclude_variables_check.all():
                no_variables = ', '.join(exclude_variables[np.where(exclude_variables_check)[0].tolist()])
                raise KeyError(f'{no_variables} are not in variables.')

        ## Filter dims
        if include_dims is not None:
            dims = dims[np.isin(dims, include_dims)]
        if exclude_dims is not None:
            dims = dims[~np.isin(dims, exclude_dims)]

        ## Filter variables
        if include_variables is not None:
            variables = variables[np.isin(variables, include_variables)]
        if exclude_variables is not None:
            variables = variables[~np.isin(variables, exclude_variables)]

        for ds_name in copy.deepcopy(variables):
            ds = self[ds_name]
            ds_dims = np.asarray(ds.coords)
            dims_check = np.isin(ds_dims, dims).all()
            if not dims_check:
                variables = np.delete(variables, np.where(variables == ds_name)[0])

        ## Create file
        file_kwargs['mode'] = 'w'
        new_file = File(**file_kwargs)

        ## Iterate through the coordinates
        for dim_name in dims:
            old_dim = self[dim_name]

            if coords is not None:
                if dim_name in coords:
                    data = old_dim.loc[coords[dim_name]]
                else:
                    data = old_dim.data
            else:
                data = old_dim.data

            new_dim = new_file.create_coordinate(dim_name, data, encoding=old_dim.encoding._encoding)
            new_dim.attrs.update(old_dim.attrs)

        ## Iterate through the old variables
        # TODO: Make the variable copy when doing a selection more RAM efficient
        for ds_name in variables:
            old_ds = self[ds_name]

            if coords is not None:
                ds_dims = old_ds.coords

                ds_sel = []
                for dim in ds_dims:
                    if dim in keys:
                        ds_sel.append(coords[dim])
                    else:
                        ds_sel.append(None)

                data = old_ds.loc[tuple(ds_sel)]
                new_ds = new_file.create_data_variable(ds_name, old_ds.coords, data=data, encoding=old_ds.encoding._encoding)
                new_ds.attrs.update(old_ds.attrs)
            else:
                new_ds = old_ds.copy(new_file)

        ## Add global attrs
        # new_file.attrs.update(self.attrs)

        return new_file


    # def to_pandas(self):
    #     """
    #     Convert the entire file into a pandas DataFrame.
    #     """
    #     if not import_pandas:
    #         raise ImportError('pandas could not be imported.')

    #     # TODO: This feels wrong...but it works...
    #     result = None
    #     for var_name in self.data_vars:
    #         if result is None:
    #             result = self[var_name].to_pandas().to_frame()
    #         else:
    #             result = result.join(self[var_name].to_pandas().to_frame(), how='outer')

    #     self.close()

    #     return result


    # def to_xarray(self, **kwargs):
    #     """
    #     Closes the file and opens it in xarray.

    #     Parameters
    #     ----------
    #     kwargs
    #         Any kwargs that can be passed to xr.open_dataset.

    #     Returns
    #     -------
    #     xr.Dataset
    #     """
    #     if not import_xarray:
    #         raise ImportError('xarray could not be imported.')

    #     filename = pathlib.Path(self.filename)

    #     if filename.is_file():
    #         self.close()
    #     else:
    #         temp_file = tempfile.NamedTemporaryFile()
    #         filename = temp_file.name
    #         self.to_file(filename)
    #         self.close()

    #     x1 = xr.open_dataset(filename, **kwargs)

    #     return x1


    def to_netcdf4(self, name: Union[str, pathlib.Path, io.BytesIO], compression: str='gzip', **file_kwargs):
        """
        Like copy, but must be a file path and will not be returned.
        """
        file = self.copy(name, compression, **file_kwargs)
        file.close()


    # def copy(self, name: Union[str, pathlib.Path, io.BytesIO]=None, **file_kwargs):
    #     """
    #     Copy a file object. kwargs can be any parameter for File.
    #     """
    #     # kwargs.setdefault('mode', 'w')
    #     file = File(name, mode='w', compression=compression, **file_kwargs)

    #     ## Create coordinates
    #     for dim_name in self.coords:
    #         dim = self[dim_name]
    #         _ = copy_coordinate(file, dim, dim_name)

    #     ## Create variables
    #     for ds_name in self.data_vars:
    #         ds = self[ds_name]
    #         _ = copy_data_variable(file, ds, ds_name)

    #     return file


    # def create_coordinate(self, name, data, dtype_encoded=None, dtype_decoded=None, scale_factor=None, add_offset=None, fillvalue=None, units=None, calendar=None, **kwargs):
    #     """

    #     """
    #     if 'compression' not in kwargs:
    #         compression = self.compression
    #         compressor = utils.get_compressor(compression)
    #         kwargs.update({**compressor})
    #     else:
    #         compression = kwargs['compression']

    #     data = np.asarray(data)

    #     dtype_decoded, shape = utils.get_dtype_shape(data, dtype=dtype_decoded, shape=None)

    #     if dtype_encoded is None:
    #         dtype_encoded = dtype_decoded

    #     encoding = prepare_encodings_for_variables(dtype_encoded, dtype_decoded, scale_factor, add_offset, fillvalue, units, calendar)

    #     coordinate = create_h5py_coordinate(self, name, data, shape, encoding, **kwargs)
    #     dim = Coordinate(coordinate, self, encoding)
    #     dim.encoding['compression'] = str(compression)

    #     return dim


    # def create_data_variable(self, name: str, dims: (str, tuple, list), shape: (tuple, list)=None, data=None, dtype_encoded=None, dtype_decoded=None, scale_factor=None, add_offset=None, fillvalue=None, units=None, calendar=None, **kwargs):
    #     """
    #     Add auto_encode option to determine the scale and offset automatically from the desired dtype? No, but provide the tool to allow the user to do it beforehand if they want.
    #     """
    #     if 'compression' not in kwargs:
    #         compression = self.compression
    #         compressor = utils.get_compressor(compression)
    #         kwargs.update({**compressor})
    #     else:
    #         compression = kwargs['compression']

    #     if data is not None:
    #         data = np.asarray(data)

    #     dtype_decoded, shape = utils.get_dtype_shape(data, dtype_decoded, shape)

    #     if dtype_encoded is None:
    #         dtype_encoded = dtype_decoded

    #     encoding = prepare_encodings_for_variables(dtype_encoded, dtype_decoded, scale_factor, add_offset, fillvalue, units, calendar)

    #     ds0 = create_h5py_data_variable(self, name, dims, shape, encoding, data, **kwargs)
    #     ds = DataVariable(ds0, self, encoding)
    #     ds.encoding['compression'] = str(compression)

    #     return ds


    # def create_data_variable_like(self, from_data_var: DataVariable, name: str, include_data: bool=False, include_attrs: bool=False, **kwargs):
    #     """ Create a variable similar to `other`.

    #     name
    #         Name of the variable (absolute or relative).  Provide None to make
    #         an anonymous variable.
    #     from_variable
    #         The variable which the new variable should mimic. All properties, such
    #         as shape, dtype, chunking, ... will be taken from it, but no data
    #         or attributes are being copied.

    #     Any variable keywords (see create_variable) may be provided, including
    #     shape and dtype, in which case the provided values take precedence over
    #     those from `other`.
    #     """
    #     ds = copy_data_variable(self, from_data_var, name, include_data, include_attrs, **kwargs)

    #     return ds


















































































































