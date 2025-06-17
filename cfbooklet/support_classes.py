#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:05:23 2025

@author: mike
"""
import numpy as np
import weakref
import msgspec

import utils, indexers

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

    def pop(self, key, default=None):
        return self.data.pop(key, default)

    def update(self, other=()):
        self.data.update(other)

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


class Encoding:
    """

    """
    def __init__(self, var_encoding):
        # self._encoding = msgspec.to_builtins(var_encoding)
        self._encoding = var_encoding
        for key, val in self._encoding.items():
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
        self._blt = blt_file
        self.name = var_name
        self.attrs = Attributes(self._blt, var_name, finalizers)
        self.encoding = msgspec.to_builtins(self._sys_meta.variables[self.name].encoding)
        self._encoding = Encoding(self.encoding)
        self.loc = indexers.LocationIndexer(self)
        self._finalizers = finalizers

        ## Assign all the encodings - should I do this?
        # for name, val in self._encoding_dict.items():
        #     setattr(self, name, val)

    # @property
    # def attrs(self):
    #     return Attributes(self._blt, self.name, self._finalizers)

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
        return self._encoding.decode(self._dataset[key])

    def __setitem__(self, key, value):
        self._dataset[key] = self._encoding.encode(value)

    def iter_chunks(self, sel=None):
        return self._dataset.iter_chunks(sel)

    # def __bool__(self):
    #     return self._dataset.__bool__()

    # def len(self):
    #     return self._dataset.len()

    # def sel(self, selection: dict, **file_kwargs):
    #     """
    #     Return a Selection object
    #     """
    #     dims = np.array(self.coords)

    #     ## Checks
    #     if selection is not None:
    #         keys = tuple(selection.keys())
    #         for key in keys:
    #             if key not in dims:
    #                 raise KeyError(f'{key} is not in the coordinates.')

    #     ## Create file
    #     file_kwargs['mode'] = 'w'
    #     new_file = File(**file_kwargs)

    #     ## Iterate through the coordinates
    #     for dim_name in dims:
    #         old_dim = self.file[dim_name]

    #         if selection is not None:
    #             if dim_name in selection:
    #                 data = old_dim.loc[selection[dim_name]]
    #             else:
    #                 data = old_dim.data
    #         else:
    #             data = old_dim.data

    #         new_dim = new_file.create_coordinate(dim_name, data, encoding=old_dim.encoding._encoding)
    #         new_dim.attrs.update(old_dim.attrs)

    #     ## Iterate through the old variable
    #     # TODO: Make the variable copy when doing a selection more RAM efficient

    #     ds_sel = []
    #     for dim in dims:
    #         if dim in keys:
    #             ds_sel.append(selection[dim])
    #         else:
    #             ds_sel.append(None)

    #     # print(ds_sel)

    #     data = self.loc[tuple(ds_sel)]
    #     new_ds = new_file.create_data_variable(self.name, self.coords, data=data, encoding=self.encoding._encoding)
    #     new_ds.attrs.update(self.attrs)

    #     return new_ds


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

    @property
    def step(self):
        return getattr(self._get_var_sys_meta, 'step')


    # def copy(self, to_file=None, name: str=None, include_attrs=True, **kwargs):
    #     """
    #     Copy a Coordinate object.
    #     """
    #     if (to_file is None) and (name is None):
    #         raise ValueError('If to_file is None, then a name must be passed and it must be different from the original.')

    #     if to_file is None:
    #         to_file = self.file

    #     if name is None:
    #         name = self.name

    #     ds = copy_coordinate(to_file, self, name, include_attrs=include_attrs, **kwargs)

    #     return ds

    def __repr__(self):
        """

        """
        return utils.coordinate_summary(self)


    # def to_pandas(self):
    #     """

    #     """
    #     if not import_pandas:
    #         raise ImportError('pandas could not be imported.')

    #     return pd.Index(self.data, name=self.name)


    # def to_xarray(self):
    #     """

    #     """


class DataVariable(Variable):
    """

    """
    # def to_pandas(self):
    #     """

    #     """
    #     if not import_pandas:
    #         raise ImportError('pandas could not be imported.')

    #     indexes = []
    #     for dim in self.coords:
    #         coord = self.file[dim]
    #         indexes.append(coord.data)

    #     pd_index = pd.MultiIndex.from_product(indexes, names=self.coords)

    #     series = pd.Series(self[()].flatten(), index=pd_index)
    #     series.name = self.name

    #     return series


    # def to_xarray(self, **kwargs):
    #     """

    #     """
    #     if not import_xarray:
    #         raise ImportError('xarray could not be imported.')

    #     da = xr.DataArray(data=self[()], coords=[self.file[dim].data for dim in self.coords], dims=self.coords, name=self.name, attrs=self.attrs)

    #     return da


    # def copy(self, to_file=None, name: str=None, include_data=True, include_attrs=True, **kwargs):
    #     """
    #     Copy a DataVariable object.
    #     """
    #     if (to_file is None) and (name is None):
    #         raise ValueError('If to_file is None, then a name must be passed and it must be different from the original.')

    #     if to_file is None:
    #         to_file = self.file

    #     if name is None:
    #         name = self.name

    #     ds = copy_data_variable(to_file, self, name, include_data=include_data, include_attrs=include_attrs, **kwargs)

    #     return ds


    def __repr__(self):
        """

        """
        return utils.data_variable_summary(self)







































































































