#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:25:06 2025

@author: mike
"""
import numpy as np
import booklet
from typing import Union, List
import pathlib
import msgspec
import weakref
from copy import deepcopy
import pyproj
import math

try:
    import h5netcdf
    import_h5netcdf = True
except ImportError:
    import_h5netcdf = False

try:
    import ebooklet
    import_ebooklet = True
except ImportError:
    import_ebooklet = False

from . import utils, indexers, data_models, creation, support_classes as sc
# import utils, indexers, data_models, creation, dtypes, support_classes as sc


############################################
### Parameters




############################################
### Functions




############################################
### Classes


class DatasetBase:

    # def __bool__(self):
    #     """

    #     """
    #     return self._file.__bool__()

    def __iter__(self):
        for key in self.var_names:
            yield key

    def __len__(self):
        return len(self.var_names)

    def __contains__(self, key):
        return key in self.var_names


    def __getitem__(self, key):
        return self.get(key)


    def __delitem__(self, key):
        if key not in self:
            raise KeyError(key)

        if not self.writable:
            raise ValueError('Dataset is not writable.')

        # Check if the object to delete is a coordinate
        # And if it is, check that no variables are attached to it
        if isinstance(self[key], sc.Coordinate):
            for var_name, var in self._sys_meta.variables.items():
                if isinstance(var, data_models.DataVariable):
                    if key in var.coords:
                        raise ValueError(f'{key} is a coordinate of {var_name}. You must delete all variables associated with a coordinate before you can delete the coordinate.')

        # Delete all chunks from file
        var = self[key]
        coord_origins = var.get_coord_origins()

        slices = indexers.index_combo_all(None, coord_origins, var.shape)
        for target_chunk, source_chunk, blt_key in indexers.slices_to_chunks_keys(slices, var.name, var.chunk_shape):
            try:
                del self._blt[blt_key]
            except KeyError:
                pass

        # Delete the attrs key
        try:
            del self._blt[sc.attrs_key.format(var_name=key)]
        except KeyError:
            pass

        # Delete in cache
        try:
            del self._var_cache[key]
        except KeyError:
            pass

        # Delete the instance in the sys meta
        del self._sys_meta.variables[key]


    # def sync(self):
    #     """

    #     """
    #     old_meta = msgspec.convert(self._blt.get_metadata(), data_models.SysMeta)
    #     if old_meta != self._meta:
    #         self._blt.set_metadata(msgspec.to_builtins(self._meta))
    #     self._blt.sync()

    def __bool__(self):
        return self.is_open


    def __repr__(self):
        """

        """
        return utils.file_summary(self)

    # def update_crs(self, crs: str | int | pyproj.CRS, x_coord: str, y_coord: str):
    #     """

    #     """
    #     if self.writable:
    #         self.create

    @property
    def coords(self):
        """
        Return a tuple of coords.
        """
        return tuple(self[coord_name] for coord_name in self.coord_names)

    @property
    def data_vars(self):
        """
        Return a tuple of data variables.
        """
        return tuple(self[var_name] for var_name in self.data_var_names)

    @property
    def variables(self):
        """
        Return a tuple of variables.
        """
        return tuple(self[var_name] for var_name in self.var_names)


    def select(self, sel: dict):
        """
        Filter the dataset variables by a selection of the coordinate positions.
        """
        ## Checks on input
        coord_names = self.coord_names
        for key in sel:
            if key not in coord_names:
                raise KeyError(f'The coordinate {key} does not exist in the dataset.')

        ## Create selections per coord
        _sel = {}
        for coord_name in coord_names:
            coord = self[coord_name]
            if coord_name in sel:
                slices = indexers.index_combo_all(sel[coord_name], coord.get_coord_origins(), coord.shape)
            else:
                slices = indexers.index_combo_all(None, coord.get_coord_origins(), coord.shape)
            _sel[coord_name] = slices

        ## Create selections for data vars
        data_var_names = self.data_var_names
        for data_var_name in data_var_names:
            data_var = self[data_var_name]
            data_var_sel = tuple(_sel[coord_name][0] for coord_name in data_var.coord_names)
            _sel[data_var_name] = data_var_sel

        ## Init DatasetView
        return DatasetView(self, _sel)


    def select_loc(self, sel: dict):
        """
        Filter the dataset variables by a selection of the coordinate locations/values.
        """
        ## Checks on input
        coord_names = self.coord_names
        for key in sel:
            if key not in coord_names:
                raise KeyError(f'The coordinate {key} does not exist in the dataset.')

        ## Create selections per coord
        _sel = {}
        for coord_name in coord_names:
            coord = self[coord_name]
            if coord_name in sel:
                slices = indexers.index_combo_all(indexers.loc_index_combo_all(sel[coord_name], (coord,)), coord.get_coord_origins(), coord.shape)
            else:
                slices = indexers.index_combo_all(None, coord.get_coord_origins(), coord.shape)
            _sel[coord_name] = slices

        ## Create selections for data vars
        data_var_names = self.data_var_names
        for data_var_name in data_var_names:
            data_var = self[data_var_name]
            data_var_sel = tuple(_sel[coord_name][0] for coord_name in data_var.coord_names)
            _sel[data_var_name] = data_var_sel

        ## Init DatasetView
        return DatasetView(self, _sel)


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


    def copy(self, file_path: Union[str, pathlib.Path], include_data_vars: List[str]=None, exclude_data_vars: List[str]=None):
        """
        Copy a dataset to a new cfdb file.
        """
        kwargs = dict(n_buckets=self._blt._n_buckets, buffer_size=self._blt._write_buffer_size)

        new_ds = open_dataset(file_path, 'n', compression=self.compression, compression_level=self.compression_level, **kwargs)

        data_var_names, coord_names = utils.filter_var_names(self, include_data_vars, exclude_data_vars)

        for coord_name in coord_names:
            coord = self[coord_name]
            new_coord = new_ds.create.coord.like(coord_name, coord, True)
            new_coord.attrs.update(coord.attrs.data)

        for data_var_name in data_var_names:
            data_var = self[data_var_name]
            new_data_var = new_ds.create.data_var.like(data_var_name, data_var)
            new_data_var.attrs.update(data_var.attrs.data)

            ## Write data
            data_var.load()
            coord_origins = data_var.get_coord_origins()
            slices = indexers.index_combo_all(data_var._sel, coord_origins, data_var.shape)

            for target_chunk, source_chunk, blt_key in indexers.slices_to_chunks_keys(slices, data_var.name, data_var.chunk_shape):
                ts, b1 = self._blt.get_timestamp(blt_key, True, False)
                if b1 is not None:
                    target_shape = tuple(tc.stop - tc.start for tc in target_chunk)
                    source_shape = tuple(sc.stop - sc.start for sc in source_chunk)

                    new_key = utils.make_var_chunk_key(data_var_name, [tc.start for tc in target_chunk])

                    if math.prod(target_shape) == math.prod(source_shape):
                        new_data_var._blt.set(new_key, b1, ts, False)
                    else:
                        data = self.dtype.loads(self.compressor.decompress(b1), self.chunk_shape)
                        data_b =  data_var.compressor.compress(data_var.dtype.dumps(data[source_chunk]))
                        new_data_var._blt.set(new_key, data_b, ts, False)

            # for write_chunk, data in data_var.iter_chunks():
            #     new_data_var.set(write_chunk, data)

        new_ds.attrs.update(self.attrs.data)

        return new_ds


    def to_netcdf4(self, file_path: Union[str, pathlib.Path], compression: str='gzip', include_data_vars: List[str]=None, exclude_data_vars: List[str]=None, **file_kwargs):
        """
        Save a dataset to a netcdf4 file using h5netcdf.
        """
        if not import_h5netcdf:
            raise ImportError('h5netcdf must be installed to save files to netcdf4.')

        data_var_names, coord_names = utils.filter_var_names(self, include_data_vars, exclude_data_vars)

        with h5netcdf.File(file_path, 'w', **file_kwargs) as h5:
            # dims/coords
            for coord_name in coord_names:
                coord = self[coord_name]
                h5.dimensions[coord_name] = coord.shape[0]
                coord_len = coord.shape[0]
                chunk_len = coord.chunk_shape[0]
                if chunk_len > coord_len:
                    chunk_shape = (coord_len,)
                else:
                    chunk_shape = (chunk_len,)

                attrs = deepcopy(coord.attrs.data)
                fillvalue = coord.dtype.fillvalue
                if coord.dtype.dtype_encoded is None:
                    if coord.dtype.kind in ('G', 'T'):
                        dtype_encoded = np.dtypes.StringDType(na_object=None)
                    elif coord.dtype.kind == 'u':
                        dtype_encoded = coord.dtype.dtype_decoded
                    elif coord.dtype.kind == 'i':
                        dtype_encoded = coord.dtype.dtype_decoded
                        if fillvalue is not None:
                            fillvalue = utils.fillvalue_dict[dtype_encoded.name]
                    elif coord.dtype.kind == 'M':
                        units = utils.parse_cf_time_units(coord.dtype.dtype_decoded)
                        attrs['units'] = units
                        attrs['calendar'] = "proleptic_gregorian"
                        attrs['standard_name'] = 'time'
                        dtype_encoded = np.dtypes.Int64DType()
                    else:
                        dtype_encoded = coord.dtype.dtype_decoded
                else:
                    if coord.dtype.kind == 'M':
                        units = utils.parse_cf_time_units(coord.dtype.dtype_decoded)
                        attrs['units'] = units
                        attrs['calendar'] = "proleptic_gregorian"
                        attrs['standard_name'] = 'time'
                        dtype_encoded = np.dtypes.Int64DType()
                    else:
                        dtype_encoded = coord.dtype.dtype_encoded
                        attrs['add_offset'] = coord.dtype.offset
                        attrs['scale_factor'] = 1/coord.dtype._factor

                if fillvalue is not None:
                    attrs['_FillValue'] = fillvalue

                attrs.update({'dtype': dtype_encoded.name})

                h5_coord = h5.create_variable(coord_name, (coord_name,), dtype_encoded, compression=compression, chunks=chunk_shape, fillvalue=fillvalue)

                try:
                    h5_coord.attrs.update(attrs)
                except Exception as err:
                    print(attrs)
                    raise err

                for write_chunk, data in coord.iter_chunks(True, decoded=False):
                    h5_coord[write_chunk] = data.astype(dtype_encoded)

            # Data vars
            for data_var_name in data_var_names:
                data_var = self[data_var_name]
                chunk_shape = []
                for s, cs in zip(data_var.shape, data_var.chunk_shape):
                    if cs > s:
                        chunk_shape.append(s)
                    else:
                        chunk_shape.append(cs)

                attrs = deepcopy(data_var.attrs.data)

                fillvalue = data_var.dtype.fillvalue
                if data_var.dtype.dtype_encoded is None:
                    if data_var.dtype.kind in ('G', 'T'):
                        dtype_encoded = np.dtypes.StringDType(na_object=None)
                    elif data_var.dtype.kind == 'u':
                        dtype_encoded = data_var.dtype.dtype_decoded
                    elif data_var.dtype.kind == 'i':
                        dtype_encoded = data_var.dtype.dtype_decoded
                        if fillvalue is not None:
                            fillvalue = utils.fillvalue_dict[dtype_encoded.name]
                    elif data_var.dtype.kind == 'M':
                        units = utils.parse_cf_time_units(data_var.dtype.dtype_decoded)
                        attrs['units'] = units
                        attrs['calendar'] = "proleptic_gregorian"
                        # attrs['standard_name'] = 'time'
                        dtype_encoded = np.dtypes.Int64DType()
                    else:
                        dtype_encoded = coord.dtype.dtype_decoded
                else:
                    if data_var.dtype.kind == 'M':
                        units = utils.parse_cf_time_units(data_var.dtype.dtype_decoded)
                        attrs['units'] = units
                        attrs['calendar'] = "proleptic_gregorian"
                        attrs['standard_name'] = 'time'
                        dtype_encoded = np.dtypes.Int64DType()
                    else:
                        dtype_encoded = data_var.dtype.dtype_encoded
                        attrs['add_offset'] = data_var.dtype.offset
                        attrs['scale_factor'] = 1/data_var.dtype._factor

                if fillvalue is not None:
                    attrs['_FillValue'] = fillvalue

                attrs.update({'dtype': dtype_encoded.name})

                h5_data_var = h5.create_variable(data_var_name, data_var.coord_names, dtype_encoded, compression=compression, chunks=tuple(chunk_shape), fillvalue=fillvalue)

                h5_data_var.attrs.update(attrs)

                for write_chunk, data in data_var.iter_chunks(True, decoded=False):
                    h5_data_var[write_chunk] = data.astype(dtype_encoded)

            # Add global attrs
            h5.attrs.update(self.attrs.data)


class Dataset(DatasetBase):
    """

    """
    def __init__(self, file_path, open_blt, create, compression, compression_level, dataset_type):
        """
        Compression can be either zstd, lz4, or None. But there's no point in using None.
        """
        self._blt = open_blt
        self.writable = self._blt.writable
        self.file_path = pathlib.Path(file_path)
        self.is_open = True

        if hasattr(self._blt, 'load_items'):
            self._has_load_items = True
        else:
            self._has_load_items = False

        ## Set/Get system metadata
        if create:
            # Checks
            compression = compression.lower()
            if compression not in utils.compression_options:
                raise ValueError(f'compression must be one of {utils.compression_options}.')
            if compression_level is None:
                compression_level = utils.default_compression_levels[compression]
            elif not isinstance(compression_level, int):
                raise ValueError('compression_level must be either None or an int.')

            self._sys_meta = data_models.SysMeta(dataset_type=dataset_type, compression=data_models.Compressor(compression), compression_level=compression_level, variables={})
            self._blt.set_metadata(msgspec.to_builtins(self._sys_meta))

        else:
            self._sys_meta = msgspec.convert(self._blt.get_metadata(), data_models.SysMeta)

        self.compression = self._sys_meta.compression.value
        self.compression_level = self._sys_meta.compression_level
        self._compressor = sc.Compressor(self.compression, self.compression_level)

        self._finalizers = [weakref.finalize(self, utils.dataset_finalizer, self._blt, self._sys_meta)]

        self.attrs = sc.Attributes(self._blt, '_', self.writable, self._finalizers)
        if create and dataset_type == 'ts_ortho':
            self.attrs['featureType'] = 'timeSeries'

        if self._sys_meta.crs is None:
            self.crs = None
        else:
            self.crs = pyproj.CRS.from_user_input(self._sys_meta.crs)

        self._var_cache = weakref.WeakValueDictionary()

        if self.writable:
            self.create = creation.Creator(self)


    def get(self, var_name):
        """
        Get a variable contained within the dataset.
        """
        if not isinstance(var_name, str):
            raise TypeError('var_name must be a string.')

        if var_name not in self:
            raise ValueError(f'The Variable {var_name} does not exist.')

        # if self._sel is not None:
        #     if var_name not in self._sel:
        #         raise ValueError(f'The Variable {var_name} does not exist in view.')

        if var_name not in self._var_cache:
            var_meta = self._sys_meta.variables[var_name]
            if isinstance(var_meta, data_models.DataVariable):
                var = sc.DataVariable(var_name, self)
            else:
                var = sc.Coordinate(var_name, self)
            self._var_cache[var_name] = var

        return self._var_cache[var_name]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """
        Close the database.
        """
        # self.sync()
        for finalizer in reversed(self._finalizers):
            finalizer()
        self.is_open = False


    @property
    def var_names(self):
        """
        Return a tuple of all the variables names (coord and data variables).
        """
        return tuple(self._sys_meta.variables.keys())

    @property
    def coord_names(self):
        """
        Return a tuple of all the coordinate names.
        """
        return tuple(k for k, v in self._sys_meta.variables.items() if isinstance(v, data_models.CoordinateVariable))


    @property
    def data_var_names(self):
        """
        Return a tuple of all the data variable names.
        """
        return tuple(k for k, v in self._sys_meta.variables.items() if isinstance(v, data_models.DataVariable))


    def prune(self, timestamp=None, reindex=False):
        """
        Prunes deleted data from the file. Returns the number of removed items. The method can also prune remove keys/values older than the timestamp. The user can also reindex the booklet file. False does no reindexing, True increases the n_buckets to a preassigned value, or an int of the n_buckets. True can only be used if the default n_buckets were used at original initialisation.
        """
        return self._blt.prune(timestamp, reindex)

    # def sync(self):
    #     """

    #     """
    #     self._blt.sync()




class DatasetView(DatasetBase):
    """

    """
    def __init__(self, dataset, sel):
        """

        """
        self._dataset = dataset
        self._sel = sel
        self._blt = dataset._blt
        self._has_load_items = dataset._has_load_items
        self.writable = False
        self.file_path = dataset.file_path
        self._sys_meta = dataset._sys_meta
        self._compressor = dataset._compressor
        self.compression = dataset.compression
        self.compression_level = dataset.compression_level
        self.attrs = dataset.attrs
        self._var_cache = dataset._var_cache


    def get(self, var_name):
        """
        Get a variable contained within the dataset.
        """
        if self._sel is not None:
            if var_name not in self._sel:
                raise ValueError(f'The Variable {var_name} does not exist in view.')

        return self._dataset.get(var_name)[self._sel[var_name]]


    @property
    def is_open(self):
        return self._dataset.is_open

    @property
    def var_names(self):
        """
        Return a tuple of all the variables names (coord and data variables).
        """
        return tuple(self._sel.keys())

    @property
    def coord_names(self):
        """
        Return a tuple of all the coordinate names.
        """
        return tuple(k for k, v in self._sys_meta.variables.items() if isinstance(v, data_models.CoordinateVariable) if k in self._sel)

    @property
    def data_var_names(self):
        """
        Return a tuple of all the data variable names.
        """
        return tuple(k for k, v in self._sys_meta.variables.items() if isinstance(v, data_models.DataVariable) if k in self._sel)

    # @property
    # def coords(self):
    #     return tuple(self[coord_name][self._sel[coord_name]] for coord_name in self.coord_names if coord_name in self._sel)

    # @property
    # def data_vars(self):
    #     return tuple(self[var_name][self._sel[var_name]] for var_name in self.data_var_names if var_name in self._sel)

    # @property
    # def variables(self):
    #     return tuple(self[var_name][self._sel[var_name]] for var_name in self.var_names if var_name in self._sel)



class EDataset(Dataset):
    """

    """
    def changes(self):
        """
        Return a Change object of the changes that have occurred during this session.
        """
        return self._blt.changes()

    def delete_remote(self):
        """
        Completely delete the remote dataset, but keep the local dataset.
        """
        self._blt.delete_remote()

    def copy_remote(self, remote_conn: ebooklet.S3Connection):
        """
        Copy the entire remote dataset to another remote location. The new location must be empty.
        """
        self._blt.copy_remote(remote_conn)


class Grid(Dataset):
    """

    """

class EGrid(EDataset):
    """

    """

class TimeSeriesOrtho(Dataset):
    """

    """

class ETimeSeriesOrtho(Dataset):
    """

    """

#######################################################
### Open functions


def open_dataset(file_path: Union[str, pathlib.Path],
                 flag: str = "r",
                 dataset_type: str='grid',
                 compression: str='zstd',
                 compression_level: int=None,
                 **kwargs):
    """
    Open a cfdb dataset. This uses the python package booklet for managing data in a single file.

    Parameters
    ----------
    file_path: str or pathlib.Path
        It must be a path to a local file location. If you want to use a tempfile, then use the name from the NamedTemporaryFile initialized class.
    flag: str
        Flag associated with how the file is opened according to the dbm style. See below for details.
    dataset_type: str
        The dataset type to be opened. The current options are "grid" and "ts_ortho". The default is "grid". See below for a description.
    compression: str
        The compression algorithm used for compressing all data. Must be either zstd or lz4. The option zstd has a really good combo of compression ratio to speed, while lz4 has a stronger emphasis on speed. Default is zstd.
    compression_level: int or None
        The compression level used by the compression algorithm. Setting this to None will use the deafults, which is currently 1 for both compression options.
    kwargs
        Any kwargs that can be passed to booklet.open.

    Returns
    -------
    cfdb.Dataset

    The optional *flag* argument can be:
    +---------+-------------------------------------------+
    | Value   | Meaning                                   |
    +=========+===========================================+
    | ``'r'`` | Open existing database for reading only   |
    |         | (default)                                 |
    +---------+-------------------------------------------+
    | ``'w'`` | Open existing database for reading and    |
    |         | writing                                   |
    +---------+-------------------------------------------+
    | ``'c'`` | Open database for reading and writing,    |
    |         | creating it if it doesn't exist           |
    +---------+-------------------------------------------+
    | ``'n'`` | Always create a new, empty database, open |
    |         | for reading and writing                   |
    +---------+-------------------------------------------+

    The dataset_type argument determines the dimension/coordinate data structure:
        grid: The standard CF conventions dimensions/coordinates. Each coordinate must be unique and increasing in ascending order. Each coordinate represents a single axis (i.e. x, y, z, t). The z axis is currently optional.
        ts_ortho: The a special time series coordinate structure representing the `Orthogonal multidimensional array representation of time series <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#_orthogonal_multidimensional_array_representation_of_time_series>`_. The Geometry dtype must represent the xy axis. The time coordinate is the same as the "grid" time coordinate. The z axis is currently optional.
    """
    if 'n_buckets' not in kwargs:
        kwargs['n_buckets'] = utils.default_n_buckets

    fp = pathlib.Path(file_path)
    fp_exists = fp.exists()
    open_blt = booklet.open(file_path, flag, key_serializer='str', **kwargs)

    if not fp_exists or flag == 'n':
        create = True
    else:
        create = False

    if dataset_type.lower() == 'grid':
        return Grid(fp, open_blt, create, compression, compression_level, 'grid')
    elif dataset_type.lower() == 'ts_ortho':
        return TimeSeriesOrtho(fp, open_blt, create, compression, compression_level, 'ts_ortho')
    else:
        raise TypeError('The only option for the dataset type is "grid".')


def open_edataset(remote_conn: Union[ebooklet.S3Connection, str, dict],
                  file_path: Union[str, pathlib.Path],
                  flag: str = "r",
                  dataset_type: str='grid',
                  compression: str='zstd',
                  compression_level: int=1,
                  **kwargs):
    """
    Open a cfdb that is linked with a remote S3 database.

    Parameters
    -----------
    remote_conn : S3Connection, str, or dict
        The object to connect to a remote. It can be an S3Connection object, an http url string, or a dict with the parameters for initializing an S3Connection object.

    file_path : str or pathlib.Path
        It must be a path to a local file location. If you want to use a tempfile, then use the name from the NamedTemporaryFile initialized class.

    flag : str
        Flag associated with how the file is opened according to the dbm style. See below for details.
    dataset_type: str
        The dataset type to be opened. The current options are "grid" and "ts_ortho". The default is "grid". See below for a description.
    compression: str
        The compression algorithm used for compressing all data. Must be either zstd or lz4. The option zstd has a really good combo of compression ratio to speed, while lz4 has a stronger emphasis on speed (and is lightning fast). Default is zstd.
    compression_level: int or None
        The compression level used by the compression algorithm. Setting this to None will d=used the deafults, which is 1 for both compression options.
    kwargs
        Any kwargs that can be passed to ebooklet.open.

    Returns
    -------
    cfdb.EDataset

    The optional *flag* argument can be:
    +---------+-------------------------------------------+
    | Value   | Meaning                                   |
    +=========+===========================================+
    | ``'r'`` | Open existing database for reading only   |
    |         | (default)                                 |
    +---------+-------------------------------------------+
    | ``'w'`` | Open existing database for reading and    |
    |         | writing                                   |
    +---------+-------------------------------------------+
    | ``'c'`` | Open database for reading and writing,    |
    |         | creating it if it doesn't exist           |
    +---------+-------------------------------------------+
    | ``'n'`` | Always create a new, empty database, open |
    |         | for reading and writing                   |
    +---------+-------------------------------------------+

    The dataset_type argument determines the dimension/coordinate data structure:
        grid: The standard CF conventions dimensions/coordinates. Each coordinate must be unique and increasing in ascending order. Each coordinate represents a single axis (i.e. x, y, z, t). The z axis is currently optional.
        ts_ortho: The a special time series coordinate structure representing the `Orthogonal multidimensional array representation of time series <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#_orthogonal_multidimensional_array_representation_of_time_series>`_. The Geometry dtype must represent the xy axis. The time coordinate is the same as the "grid" time coordinate. The z axis is currently optional.
    """
    if not import_ebooklet:
        raise ImportError('EBooklet must be installed to open EDatasets.')

    if 'n_buckets' not in kwargs:
        kwargs['n_buckets'] = utils.default_n_buckets

    fp = pathlib.Path(file_path)
    fp_exists = fp.exists()
    open_blt = ebooklet.open(remote_conn, file_path, flag, **kwargs)

    if (not fp_exists or flag == 'n') and open_blt.writable:
        create = True
    else:
        create = False

    if dataset_type.lower() == 'grid':
        return EGrid(fp, open_blt, create, compression, compression_level, 'grid')
    else:
        raise TypeError('The only option for the dataset type is "grid".')




































































































