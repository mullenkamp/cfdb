#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:05:23 2025

@author: mike
"""
import numpy as np
import weakref
import msgspec
import lz4.frame
import zstandard as zstd

import utils, indexers

###################################################
### Classes


class Categorical:
    """
    This class and dtype should be similar to the pandas categorical dtype. Preferably, all string arrays should be cat dtypes. In the CF conventions, this is equivelant to `flags <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#flags>`_. The CF conventions of assigning the attrs flag_values and flag_meanings should be used for compatability.
    As in the CF conventions, two python lists can be used (one int in increasing order from 0 as the index, and the other as the string values). The string values would have no sorted order. They would be assigned the int index as they are assigned.
    This class should replace the fixed-length numpy unicode class.
    At the moment, I don't want to implement this until I've got the rest of the package implemented.
    """
    # TODO


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


class Compressor:
    """

    """
    def __init__(self, compression, compression_level):
        """

        """
        self.compression = compression
        self.compression_level = compression_level

        if compression == 'lz4':
            self.compress = self._lz4_compress
            self.decompress = self._lz4_decompress
        elif compression == 'zstd':
            self._cctx = zstd.ZstdCompressor(level=compression_level)
            self._dctx = zstd.ZstdDecompressor()
            self.compress = self._zstd_compress
            self.decompress = self._zstd_decompress
        else:
            raise ValueError('compression must be either lz4 or zstd')

    def _lz4_compress(self, data: bytes):
        """

        """
        return lz4.frame.compress(data, compression_level=self.compression_level)

    def _lz4_decompress(self, data: bytes):
        """

        """
        return lz4.frame.decompress(data)

    def _zstd_compress(self, data: bytes):
        """

        """
        return self._cctx.compress(data)

    def _zstd_decompress(self, data: bytes):
        """

        """
        return self._dctx.decompress(data)


class Encoding:
    """

    """
    def __init__(self, chunk_shape, dtype_decoded, dtype_encoded, fillvalue, scale_factor, add_offset, compressor):
        # self._encoding = msgspec.to_builtins(var_encoding)
        # self._encoding = var_encoding
        self.compressor = compressor
        self.chunk_shape = chunk_shape
        self.dtype_decoded = dtype_decoded
        self.dtype_encoded = dtype_encoded
        self.fillvalue = fillvalue
        self.scale_factor = scale_factor
        self.add_offset = add_offset
        # for key, val in self._encoding.items():
        #     setattr(self, key, val)

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

    def encode(self, data: np.ndarray):
        if data.dtype != self.dtype_decoded:
            raise TypeError('The data dtype does not match the assigned dtype_decoded.')

        if self.dtype_encoded != self.dtype_decoded:

            # if data.dtype.kind == 'M':
            #     data = data.astype(self.dtype_encoded)

            if isinstance(self.add_offset, (int, float)):
                data = data - self.add_offset

            if isinstance(self.scale_factor, (int, float)):
                # precision = int(np.abs(np.log10(val['scale_factor'])))
                data = data/self.scale_factor

            if isinstance(self.fillvalue, int) and (self.dtype_decoded.kind == 'f'):
                data[np.isnan(data)] = self.fillvalue

            data = data.astype(self.dtype_encoded)

        return self.compressor.compress(data.tobytes())


    def decode(self, data: bytes):
        data = np.frombuffer(self.compressor.decompress(data), dtype=self.dtype_encoded).reshape(self.chunk_shape)

        if self.dtype_encoded != self.dtype_decoded:
            data = data.astype(self.dtype_decoded)

            if isinstance(self.fillvalue, int) and (self.dtype_decoded.kind == 'f'):
                data[np.isclose(data, self.fillvalue)] = np.nan

            if isinstance(self.scale_factor, (int, float)):
                data = data * self.scale_factor

            if isinstance(self.add_offset, (int, float)):
                data = data + self.add_offset

        return data


class Variable:
    """

    """
    def __init__(self, var_name, blt_file, sys_meta, compressor, finalizers):
        """

        """
        self._var_meta = sys_meta.variables[var_name]
        self._blt = blt_file
        self.name = var_name
        self.attrs = Attributes(self._blt, var_name, finalizers)
        # self.encoding = msgspec.to_builtins(self._sys_meta.variables[self.name].encoding)
        self.chunk_shape = self._var_meta.chunk_shape
        self.dtype_decoded = self._var_meta.dtype_decoded
        self.dtype_encoded = self._var_meta.dtype_encoded
        self.fillvalue = self._var_meta.fillvalue
        self.scale_factor = self._var_meta.scale_factor
        self.add_offset = self._var_meta.add_offset
        self.coords = self._var_meta.coords
        self.ndim = len(self.coords)

        self._encoding = Encoding(self.chunk_shape, self.dtype_decoded, self.dtype_encoded, self.fillvalue, self.scale_factor, self.add_offset, compressor)
        self.loc = indexers.LocationIndexer(self)
        self._finalizers = finalizers

        ## Assign all the encodings - should I do this?
        # for name, val in self._encoding_dict.items():
        #     setattr(self, name, val)

    # @property
    # def attrs(self):
    #     return Attributes(self._blt, self.name, self._finalizers)

    # @property
    # def _get_var_sys_meta(self):
    #     """

    #     """
    #     return getattr(self._sys_meta, 'variables')[self.name]

    @property
    def shape(self):
        return getattr(self._var_meta, 'shape')

    # @property
    # def coords(self):
    #     return getattr(self._get_var_sys_meta, 'coords')

    # @property
    # def ndim(self):
    #     return len(getattr(self._get_var_sys_meta, 'coords'))

    # @property
    # def dtype(self):
    #     # TODO: should be in the encoding data
    #     return np.dtype(getattr(self._get_var_sys_meta, 'dtype_decoded'))

    # @property
    # def chunk_shape(self):
    #     return getattr(self._get_var_sys_meta, 'chunk_shape')

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

    def prepend(self, data):
        """
        Prepend data to the start of the coordinate. The extra length will be added to the associated data variables with the fillvalue. One of data or length must be passed. A negative length can be passed to shrink the coordinate from the start.
        """


    def append(self, data):
        """
        Append data to the end of the coordinate. The extra length will be added to the associated data variables with the fillvalue. One of data or length must be passed. A negative length can be passed to shrink the coordinate from the end.
        """
        new_data, start_write_pos = utils.append_coord_data_checks(data, self.data, self.dtype_decoded, self.step)



    def resize(self, start, end):
        """
        Resize a coordinate. If step is an int or float, then resizing can add or truncate the length. If step is None, then the coordinate can only have the length truncated.
        If the coordinate length is reduced, then all data variables associated with the coordinate will have their data truncated.
        """


    @property
    def step(self):
        return getattr(self._var_meta, 'step')

    @property
    def auto_increment(self):
        return getattr(self._var_meta, 'auto_increment')



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







































































































