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

# from . import utils, indexers
import utils, indexers, data_models

############################################
### Parameters

compression_options = ('zstd', 'lz4', None)
default_n_buckets = 144013
name_indent = 4
value_indent = 20
var_name_regex = "^[a-zA-Z][a-zA-Z0-9_]*$"
var_name_pattern = re.compile(var_name_regex)

############################################
### Functions

def is_var_name(name):
    """

    """
    res = var_name_pattern.search(name)
    if res:
        return True
    else:
        return False


def open_variable(ds, var_name):
    """

    """
    if var_name not in ds.sys_meta['variable']:
        raise KeyError(f'{var_name} does not exist as a variable.')

    var_meta = ds.sys_meta['variable'][var_name]

    if var_meta['is_coord']:
        var = Coordinate(ds, var_name)
    else:
        var = DataVariable(ds, var_name)

    return var


def format_value(value):
    """

    """
    if isinstance(value, (int, np.integer)):
        return str(value)
    elif isinstance(value, (float, np.floating)):
        return f'{value:.2f}'
    else:
        return value


def append_summary(summary, summ_dict):
    """

    """
    for key, value in summ_dict.items():
        spacing = value_indent - len(key)
        if spacing < 1:
            spacing = 1

        summary += f"""\n{key}""" + """ """ * spacing + value

    return summary


def data_variable_summary(ds):
    """

    """
    if ds:
        summ_dict = {'name': ds.name, 'dims order': '(' + ', '.join(ds.coords) + ')', 'chunk size': str(ds.chunks)}

        summary = """<cfbooklet.DataVariable>"""

        summary = append_summary(summary, summ_dict)

        summary += """\nCoordinates:"""

        for dim_name in ds.coords:
            dim = ds.file[dim_name]
            dtype_name = dim.encoding['dtype_decoded']
            dim_len = dim.shape[0]
            first_value = format_value(dim[0])
            spacing = value_indent - name_indent - len(dim_name)
            if spacing < 1:
                spacing = 1
            dim_str = f"""\n    {dim_name}""" + """ """ * spacing
            dim_str += f"""({dim_len}) {dtype_name} {first_value} ..."""
            summary += dim_str

        attrs_summary = make_attrs_repr(ds.attrs, name_indent, value_indent, 'Attributes')
        summary += """\n""" + attrs_summary

    else:
        summary = """DataVariable is closed"""

    return summary


def coordinate_summary(ds):
    """

    """
    if ds:
        name = ds.name
        dim_len = ds.shape[0]
        # dtype_name = ds.dtype.name
        # dtype_decoded = ds.encoding['dtype_decoded']

        first_value = format_value(ds.data[0])
        last_value = format_value(ds.data[-1])

        # summ_dict = {'name': name, 'dtype encoded': dtype_name, 'dtype decoded': dtype_decoded, 'chunk size': str(ds.chunks), 'dim length': str(dim_len), 'values': f"""{first_value} ... {last_value}"""}
        summ_dict = {'name': name, 'chunk size': str(ds.chunks), 'dim length': str(dim_len), 'values': f"""{first_value} ... {last_value}"""}

        summary = """<cfbooklet.Coordinate>"""

        summary = append_summary(summary, summ_dict)

        attrs_summary = make_attrs_repr(ds.attrs, name_indent, value_indent, 'Attributes')
        summary += """\n""" + attrs_summary
    else:
        summary = """Coordinate is closed"""

    return summary


def make_attrs_repr(attrs, name_indent, value_indent, header):
    summary = f"""{header}:"""
    for key, value in attrs.items():
        spacing = value_indent - name_indent - len(key)
        if spacing < 1:
            spacing = 1
        line_str = f"""\n    {key}""" + """ """ * spacing + f"""{value}"""
        summary += line_str

    return summary


def create_h5py_data_variable(file, name: str, dims: (str, tuple, list), shape: (tuple, list), encoding: dict, data=None, **kwargs):
    """

    """
    dtype = encoding['dtype']

    ## Check if dims already exist and if the dim lengths match
    if isinstance(dims, str):
        dims = [dims]

    for i, dim in enumerate(dims):
        if dim not in file:
            raise ValueError(f'{dim} not in File')

        dim_len = file._file[dim].shape[0]
        if dim_len != shape[i]:
            raise ValueError(f'{dim} does not have the same length as the input data/shape dim.')

    ## Make chunks
    if 'chunks' not in kwargs:
        if 'maxshape' in kwargs:
            maxshape = kwargs['maxshape']
        else:
            maxshape = shape
        kwargs.setdefault('chunks', utils.guess_chunk(shape, maxshape, dtype))

    ## Create variable
    if data is None:
        ds = file._file.create_dataset(name, shape, dtype=dtype, track_order=True, **kwargs)
    else:
        ## Encode data before creating variable
        data = utils.encode_data(data, **encoding)

        ds = file._file.create_dataset(name, dtype=dtype, data=data, track_order=True, **kwargs)

    for i, dim in enumerate(dims):
        ds.dims[i].attach_scale(file._file[dim])
        ds.dims[i].label = dim

    return ds


def create_h5py_coordinate(file, name: str, data, shape: (tuple, list), encoding: dict, **kwargs):
    """

    """
    if len(shape) != 1:
        raise ValueError('The shape of a coordinate must be 1-D.')

    dtype = encoding['dtype']

    ## Make chunks
    if 'chunks' not in kwargs:
        if 'maxshape' in kwargs:
            maxshape = kwargs['maxshape']
        else:
            maxshape = shape
        kwargs.setdefault('chunks', utils.guess_chunk(shape, maxshape, dtype))

    ## Encode data before creating variable/coordinate
    # print(encoding)
    data = utils.encode_data(data, **encoding)

    # print(data)
    # print(dtype)

    ## Make Variable
    ds = file._file.create_dataset(name, dtype=dtype, data=data, track_order=True, **kwargs)

    ds.make_scale(name)
    ds.dims[0].label = name

    return ds


def copy_data_variable(to_file, from_variable, name, include_data=True, include_attrs=True, **kwargs):
    """

    """
    other1 = from_variable._dataset
    for k in ('chunks', 'compression',
              'compression_opts', 'scaleoffset', 'shuffle', 'fletcher32',
              'fillvalue'):
        kwargs.setdefault(k, getattr(other1, k))

    if 'compression' in other1.attrs:
        compression = other1.attrs['compression']
        kwargs.update(**utils.get_compressor(compression))
    else:
        compression = kwargs['compression']

    # TODO: more elegant way to pass these (dcpl to create_variable?)
    dcpl = other1.id.get_create_plist()
    kwargs.setdefault('track_times', dcpl.get_obj_track_times())
    # kwargs.setdefault('track_order', dcpl.get_attr_creation_order() > 0)

    # Special case: the maxshape property always exists, but if we pass it
    # to create_variable, the new variable will automatically get chunked
    # layout. So we copy it only if it is different from shape.
    if other1.maxshape != other1.shape:
        kwargs.setdefault('maxshape', other1.maxshape)

    encoding = from_variable.encoding._encoding.copy()
    shape = from_variable.shape

    ds0 = create_h5py_data_variable(to_file, name, tuple(dim.label for dim in other1.dims), shape, encoding, **kwargs)

    if include_data:
        # Directly copy chunks using write_direct_chunk
        for chunk in ds0.iter_chunks():
            chunk_starts = tuple(c.start for c in chunk)
            filter_mask, data = other1.id.read_direct_chunk(chunk_starts)
            ds0.id.write_direct_chunk(chunk_starts, data, filter_mask)

    ds = DataVariable(ds0, to_file, encoding)
    if include_attrs:
        ds.attrs.update(from_variable.attrs)

    return ds


def copy_coordinate(to_file, from_coordinate, name, include_attrs=True, **kwargs):
    """

    """
    other1 = from_coordinate._dataset
    for k in ('chunks', 'compression',
              'compression_opts', 'scaleoffset', 'shuffle', 'fletcher32',
              'fillvalue'):
        kwargs.setdefault(k, getattr(other1, k))

    if 'compression' in other1.attrs:
        compression = other1.attrs['compression']
        kwargs.update(**utils.get_compressor(compression))
    else:
        compression = kwargs['compression']

    # TODO: more elegant way to pass these (dcpl to create_variable?)
    dcpl = other1.id.get_create_plist()
    kwargs.setdefault('track_times', dcpl.get_obj_track_times())
    # kwargs.setdefault('track_order', dcpl.get_attr_creation_order() > 0)

    # Special case: the maxshape property always exists, but if we pass it
    # to create_variable, the new variable will automatically get chunked
    # layout. So we copy it only if it is different from shape.
    if other1.maxshape != other1.shape:
        kwargs.setdefault('maxshape', other1.maxshape)

    encoding = from_coordinate.encoding._encoding.copy()
    shape = from_coordinate.shape

    ds0 = create_h5py_coordinate(to_file, name, from_coordinate.data, shape, encoding, **kwargs)

    ds = Coordinate(ds0, to_file, encoding)
    if include_attrs:
        ds.attrs.update(from_coordinate.attrs)

    return ds


def prepare_encodings_for_variables(dtype_encoded, dtype_decoded, scale_factor, add_offset, fillvalue, units, calendar):
    """

    """
    encoding = {'dtype': dtype_encoded, 'dtype_encoded': dtype_encoded, 'missing_value': fillvalue, '_FillValue': fillvalue, 'add_offset': add_offset, 'scale_factor': scale_factor, 'units': units, 'calendar': calendar}
    for key, value in copy.deepcopy(encoding).items():
        if value is None:
            del encoding[key]

    if 'datetime64' in dtype_decoded:
        if 'units' not in encoding:
            encoding['units'] = 'seconds since 1970-01-01'
        if 'calendar' not in encoding:
            encoding['calendar'] = 'gregorian'
        encoding['dtype'] = 'int64'

    return encoding


def file_summary(file):
    """

    """
    if file:
        file_path = pathlib.Path(file.filename)
        if file_path.exists() and file_path.is_file():
            file_size = file_path.stat().st_size*0.000001
            file_size_str = """{file_size:.1f} MB""".format(file_size=file_size)
        else:
            file_size_str = """NA"""

        summ_dict = {'file name': file_path.name, 'file size': file_size_str, 'writable': str(file.writable)}

        summary = """<hdf5tools.File>"""

        summary = append_summary(summary, summ_dict)

        summary += """\nCoordinates:"""

        for dim_name in file.coords:
            dim = file[dim_name]
            dtype_name = dim.encoding['dtype_decoded']
            dim_len = dim.shape[0]
            first_value = format_value(dim[0])
            spacing = value_indent - name_indent - len(dim_name)
            if spacing < 1:
                spacing = 1
            dim_str = f"""\n    {dim_name}""" + """ """ * spacing
            dim_str += f"""({dim_len}) {dtype_name} {first_value} ..."""
            summary += dim_str

        summary += """\nData Variables:"""

        for ds_name in file.data_vars:
            ds = file[ds_name]
            dtype_name = ds.encoding['dtype_decoded']
            shape = ds.shape
            dims = ', '.join(ds.coords)
            first_value = format_value(ds[tuple(0 for i in range(len(shape)))])
            spacing = value_indent - name_indent - len(ds_name)
            if spacing < 1:
                spacing = 1
            ds_str = f"""\n    {ds_name}""" + """ """ * spacing
            ds_str += f"""({dims}) {dtype_name} {first_value} ..."""
            summary += ds_str

        attrs_summary = make_attrs_repr(file.attrs, name_indent, value_indent, 'Attributes')
        summary += """\n""" + attrs_summary
    else:
        summary = """File is closed"""

    return summary

###################################################
### Classes


class Attributes:
    """

    """
    def __init__(self, blt_file, var_name, finalizers):
        """

        """
        key = f'_{var_name}.attrs'
        self.data = blt_file.get(key)
        if self.data is None:
            self.data = {}

        self._blt = blt_file
        self._var_name = var_name
        finalizers.append(weakref.finalize(self, utils.attrs_finalizer, self._blt, self.data, var_name))

    def set(self, key, value):
        """

        """
        # TODO input checks
        self.data[key] = value

    def __setitem__(self, key, value):
        """

        """
        self.set(key, value)

    def get(self, key):
        """

        """
        value = self.data.get(key)

        return value

    def __getitem__(self, key):
        """

        """
        value = self.get(key)

        return value

    def clear(self):
        self.data.clear()

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    # def pop(self, key, default=None):
    #     return self.data.pop(key, default)

    # def update(self, other=()):
    #     self.data.update(other)
    #     self._finalizer.detach()
    #     self._finalizer = utils.sys_meta_finalizer(self.data, self._blt)

    def __delitem__(self, key):
        del self.data[key]

    def __contains__(self, key):
        return key in self.data

    def __iter__(self):
        return self.keys()

    # def sync(self):
    #     utils.attrs_finalizer(self._blt, self.data, self._var_name)

    # def close(self):
    #     self._finalizer()

    def __repr__(self):
        return self.data.__repr__()



# class Attributes:
#     """

#     """
#     def __init__(self, attrs: h5py.AttributeManager):
#         self._attrs = attrs

#     def get(self, key, default=None):
#         return self._attrs.get(key, default)

#     def __getitem__(self, key):
#         return self._attrs[key]

#     def __setitem__(self, key, value):
#         self._attrs[key] = value

#     def clear(self):
#         self._attrs.clear()

#     def keys(self):
#         for key in self._attrs.keys():
#             if key not in utils.ignore_attrs:
#                 yield key

#     def values(self):
#         for key, value in self._attrs.items():
#             if key not in utils.ignore_attrs:
#                 yield value

#     def items(self):
#         for key, value in self._attrs.items():
#             if key not in utils.ignore_attrs:
#                 yield key, value

#     def pop(self, key, default=None):
#         return self._attrs.pop(key, default)

#     def update(self, other=()):
#         self._attrs.update(other)

#     def create(self, key, data, shape=None, dtype=None):
#         self._attrs.create(key, data, shape, dtype)

#     def modify(self, key, value):
#         self._attrs.modify(key, value)

#     def __delitem__(self, key):
#         del self._attrs[key]

#     def __contains__(self, key):
#         return key in self._attrs

#     def __iter__(self):
#         return self._attrs.__iter__()

#     def __repr__(self):
#         return make_attrs_repr(self, name_indent, value_indent, 'Attributes')


class Encoding:
    """

    """
    def __init__(self, var_encoding):
        self._encoding = msgspec.to_builtins(var_encoding)
        for key, val in var_encoding.items():
            setattr(self, key, val)

    # def get(self, key, default=None):
    #     return self._encoding.get(key, default)

    # def __getitem__(self, key):
    #     return self._encoding[key]

    # def __setitem__(self, key, value):
    #     if key in utils.enc_fields:
    #         self._encoding[key] = value
    #         if self._writable:
    #             self._attrs[key] = value
    #     else:
    #         raise ValueError(f'key must be one of {utils.enc_fields}.')

    # def clear(self):
    #     keys = list(self._encoding.keys())
    #     self._encoding.clear()
    #     if self._writable:
    #         for key in keys:
    #             del self._attrs[key]

    # def keys(self):
    #     return self._encoding.keys()

    # def values(self):
    #     return self._encoding.values()

    # def items(self):
    #     return self._encoding.items()

    # def pop(self, key, default=None):
    #     if self._writable:
    #         if key in self._attrs:
    #             del self._attrs[key]
    #     return self._encoding.pop(key, default)

    # def update(self, other=()):
    #     key_values = {**other}
    #     for key, value in key_values.items():
    #         if key in utils.enc_fields:
    #             self._encoding[key] = value
    #             if self._writable:
    #                 self._attrs[key] = value

    # def __delitem__(self, key):
    #     del self._encoding[key]
    #     if self._writable:
    #         del self._attrs[key]

    # def __contains__(self, key):
    #     return key in self._encoding

    # def __iter__(self):
    #     return self._encoding.__iter__()

    # def __repr__(self):
    #     return make_attrs_repr(self, name_indent, value_indent, 'Encodings')

    def encode(self, values):
        return utils.encode_data(np.asarray(values), **self._encoding)

    def decode(self, bytes_data):
        # results = utils.decode_data(values, **self._encoding)

        # if results.ndim == 0:
        #     return results[()]
        # else:
        #     return results

        return utils.decode_data(bytes_data, **self._encoding)


class Variable:
    """

    """
    def __init__(self, blt_file, var_name, sys_meta, finalizers):
        """

        """
        self._sys_meta = sys_meta
        self._var
        self._blt = blt_file
        self.name = var_name
        self.attrs = Attributes(self._blt, var_name, finalizers)
        self.encoding = Encoding(getattr(getattr(self._sys_meta, 'variables')[self.name], 'encoding'))
        self.loc = indexers.LocationIndexer(self)

    @property
    def _get_var_sys_meta(self):
        """

        """
        return getattr(self._sys_meta, 'variables')[self.name]

    @property
    def shape(self):
        return getattr(self._get_var_sys_meta, 'shape')

    @property
    def coords(self):
        return getattr(self._get_var_sys_meta, 'coords')

    @property
    def ndim(self):
        return len(getattr(self._get_var_sys_meta, 'coords'))

    # @property
    # def dtype(self):
    #     # TODO: should be in the encoding data
    #     return np.dtype(getattr(self._get_var_sys_meta, 'dtype_decoded'))

    @property
    def chunk_shape(self):
        return getattr(self._get_var_sys_meta, 'chunk_shape')

    # @property
    # def size(self):
    #     return self._dataset.size

    # @property
    # def nbytes(self):
    #     return self._dataset.nbytes

    # @property
    # def maxshape(self):
    #     return self._dataset.maxshape

    # @property
    # def fillvalue(self):
    #     return getattr(self._get_var_sys_meta, 'fillvalue')

    # def reshape(self, new_shape, axis=None):
    #     """ Reshape the dataset, or the specified axis.

    #     The dataset must be stored in chunked format; it can be resized up to
    #     the "maximum shape" (keyword maxshape) specified at creation time.
    #     The rank of the dataset cannot be changed.

    #     "shape" should be a shape tuple, or if an axis is specified, an integer.

    #     BEWARE: This functions differently than the NumPy resize() method!
    #     The data is not "reshuffled" to fit in the new shape; each axis is
    #     grown or shrunk independently.  The coordinates of existing data are
    #     fixed.
    #     """
    #     self._dataset.resize(new_shape, axis)

    def rechunker(self, target_chunk_shape, max_mem):
        """
        Generator to rechunk the variable into a new chunk shape.
        """
        pass


    def __getitem__(self, key):
        return self.encoding.decode(self._dataset[key])

    def __setitem__(self, key, value):
        self._dataset[key] = self.encoding.encode(value)

    def iter_chunks(self, sel=None):
        return self._dataset.iter_chunks(sel)

    # def __bool__(self):
    #     return self._dataset.__bool__()

    # def len(self):
    #     return self._dataset.len()

    def sel(self, selection: dict, **file_kwargs):
        """
        Return a Selection object
        """
        dims = np.array(self.coords)

        ## Checks
        if selection is not None:
            keys = tuple(selection.keys())
            for key in keys:
                if key not in dims:
                    raise KeyError(f'{key} is not in the coordinates.')

        ## Create file
        file_kwargs['mode'] = 'w'
        new_file = File(**file_kwargs)

        ## Iterate through the coordinates
        for dim_name in dims:
            old_dim = self.file[dim_name]

            if selection is not None:
                if dim_name in selection:
                    data = old_dim.loc[selection[dim_name]]
                else:
                    data = old_dim.data
            else:
                data = old_dim.data

            new_dim = new_file.create_coordinate(dim_name, data, encoding=old_dim.encoding._encoding)
            new_dim.attrs.update(old_dim.attrs)

        ## Iterate through the old variable
        # TODO: Make the variable copy when doing a selection more RAM efficient

        ds_sel = []
        for dim in dims:
            if dim in keys:
                ds_sel.append(selection[dim])
            else:
                ds_sel.append(None)

        # print(ds_sel)

        data = self.loc[tuple(ds_sel)]
        new_ds = new_file.create_data_variable(self.name, self.coords, data=data, encoding=self.encoding._encoding)
        new_ds.attrs.update(self.attrs)

        return new_ds


class Coordinate(Variable):
    """

    """
    @property
    def data(self):
        return self[()]

    def prepend(self, data=None, length=None):
        """
        Prepend data to the start of the coordinate. The extra length will be added to the associated data variables with the fillvalue. One of data or length must be passed. A negative length can be passed to shrink the coordinate from the start.
        """

    def append(self, data=None, length=None):
        """
        Append data to the end of the coordinate. The extra length will be added to the associated data variables with the fillvalue. One of data or length must be passed. A negative length can be passed to shrink the coordinate from the end.
        """


    def copy(self, to_file=None, name: str=None, include_attrs=True, **kwargs):
        """
        Copy a Coordinate object.
        """
        if (to_file is None) and (name is None):
            raise ValueError('If to_file is None, then a name must be passed and it must be different from the original.')

        if to_file is None:
            to_file = self.file

        if name is None:
            name = self.name

        ds = copy_coordinate(to_file, self, name, include_attrs=include_attrs, **kwargs)

        return ds

    def __repr__(self):
        """

        """
        return coordinate_summary(self)


    def to_pandas(self):
        """

        """
        if not import_pandas:
            raise ImportError('pandas could not be imported.')

        return pd.Index(self.data, name=self.name)


    # def to_xarray(self):
    #     """

    #     """


class DataVariable(Variable):
    """

    """
    def to_pandas(self):
        """

        """
        if not import_pandas:
            raise ImportError('pandas could not be imported.')

        indexes = []
        for dim in self.coords:
            coord = self.file[dim]
            indexes.append(coord.data)

        pd_index = pd.MultiIndex.from_product(indexes, names=self.coords)

        series = pd.Series(self[()].flatten(), index=pd_index)
        series.name = self.name

        return series


    def to_xarray(self, **kwargs):
        """

        """
        if not import_xarray:
            raise ImportError('xarray could not be imported.')

        da = xr.DataArray(data=self[()], coords=[self.file[dim].data for dim in self.coords], dims=self.coords, name=self.name, attrs=self.attrs)

        return da


    def copy(self, to_file=None, name: str=None, include_data=True, include_attrs=True, **kwargs):
        """
        Copy a DataVariable object.
        """
        if (to_file is None) and (name is None):
            raise ValueError('If to_file is None, then a name must be passed and it must be different from the original.')

        if to_file is None:
            to_file = self.file

        if name is None:
            name = self.name

        ds = copy_data_variable(to_file, self, name, include_data=include_data, include_attrs=include_attrs, **kwargs)

        return ds


    def __repr__(self):
        """

        """
        return data_variable_summary(self)


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
        self._finalizers = []
        self._finalizers.append(weakref.finalize(self, utils.dataset_finalizer, self._blt, self._meta))

        self.attrs = Attributes(self._blt, '_', self._finalizers)

        self._cache = {}


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
        return tuple(k for k, v in self._meta.variables.items() if v['is_coord'])

    @property
    def data_vars(self):
        """
        Return a tuple of all the data variables.
        """
        return tuple(k for k, v in self._meta.variables.items() if not v['is_coord'])


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
        if isinstance(var_name, str):
            pass

        else:
            raise TypeError('var_name must be a string.')


    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        if isinstance(value, Variable):
            setattr(self, key, value)
        else:
            raise TypeError('Assigned value must be a Variable or Coordinate object.')

    def __delitem__(self, key):
        try:
            if key not in self:
                raise KeyError(key)

            # Check if the object to delete is a coordinate
            # And if it is, check that no variables are attached to it
            if isinstance(self[key], Coordinate):
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
        return file_summary(self)


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


    def copy(self, name: Union[str, pathlib.Path, io.BytesIO]=None, **file_kwargs):
        """
        Copy a file object. kwargs can be any parameter for File.
        """
        # kwargs.setdefault('mode', 'w')
        file = File(name, mode='w', compression=compression, **file_kwargs)

        ## Create coordinates
        for dim_name in self.coords:
            dim = self[dim_name]
            _ = copy_coordinate(file, dim, dim_name)

        ## Create variables
        for ds_name in self.data_vars:
            ds = self[ds_name]
            _ = copy_data_variable(file, ds, ds_name)

        return file


    def create_coordinate(self, name, data, dtype_encoded=None, dtype_decoded=None, scale_factor=None, add_offset=None, fillvalue=None, units=None, calendar=None, **kwargs):
        """

        """
        if 'compression' not in kwargs:
            compression = self.compression
            compressor = utils.get_compressor(compression)
            kwargs.update({**compressor})
        else:
            compression = kwargs['compression']

        data = np.asarray(data)

        dtype_decoded, shape = utils.get_dtype_shape(data, dtype=dtype_decoded, shape=None)

        if dtype_encoded is None:
            dtype_encoded = dtype_decoded

        encoding = prepare_encodings_for_variables(dtype_encoded, dtype_decoded, scale_factor, add_offset, fillvalue, units, calendar)

        coordinate = create_h5py_coordinate(self, name, data, shape, encoding, **kwargs)
        dim = Coordinate(coordinate, self, encoding)
        dim.encoding['compression'] = str(compression)

        return dim


    def create_data_variable(self, name: str, dims: (str, tuple, list), shape: (tuple, list)=None, data=None, dtype_encoded=None, dtype_decoded=None, scale_factor=None, add_offset=None, fillvalue=None, units=None, calendar=None, **kwargs):
        """
        Add auto_encode option to determine the scale and offset automatically from the desired dtype? No, but provide the tool to allow the user to do it beforehand if they want.
        """
        if 'compression' not in kwargs:
            compression = self.compression
            compressor = utils.get_compressor(compression)
            kwargs.update({**compressor})
        else:
            compression = kwargs['compression']

        if data is not None:
            data = np.asarray(data)

        dtype_decoded, shape = utils.get_dtype_shape(data, dtype_decoded, shape)

        if dtype_encoded is None:
            dtype_encoded = dtype_decoded

        encoding = prepare_encodings_for_variables(dtype_encoded, dtype_decoded, scale_factor, add_offset, fillvalue, units, calendar)

        ds0 = create_h5py_data_variable(self, name, dims, shape, encoding, data, **kwargs)
        ds = DataVariable(ds0, self, encoding)
        ds.encoding['compression'] = str(compression)

        return ds


    def create_data_variable_like(self, from_data_var: DataVariable, name: str, include_data: bool=False, include_attrs: bool=False, **kwargs):
        """ Create a variable similar to `other`.

        name
            Name of the variable (absolute or relative).  Provide None to make
            an anonymous variable.
        from_variable
            The variable which the new variable should mimic. All properties, such
            as shape, dtype, chunking, ... will be taken from it, but no data
            or attributes are being copied.

        Any variable keywords (see create_variable) may be provided, including
        shape and dtype, in which case the provided values take precedence over
        those from `other`.
        """
        ds = copy_data_variable(self, from_data_var, name, include_data, include_attrs, **kwargs)

        return ds


















































































































