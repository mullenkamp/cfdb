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
import math

import utils, indexers, rechunker

###################################################
### Parameters

attrs_key = '_{var_name}.attrs'

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
    def __init__(self, blt_file, var_name, writable, finalizers):
        """

        """
        key = attrs_key.format(var_name=var_name)
        data = blt_file.get(key)
        if data is None:
            self.data = {}
        else:
            self.data = msgspec.json.decode(data)

        self._blt = blt_file
        self._var_name = var_name
        finalizers.append(weakref.finalize(self, utils.attrs_finalizer, self._blt, self.data, var_name, writable))
        self.writable = writable

    def set(self, key, value):
        """

        """
        # TODO input checks
        if self.writable:
            self.data[key] = value
        else:
            raise ValueError('Dataset is not writable.')

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
        if self.writable:
            self.data.update(other)
        else:
            raise ValueError('Dataset is not writable.')

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


    def to_bytes(self, encoded_array: np.ndarray) -> bytes:
        """
        from encoded array to bytes
        """
        return self.compressor.compress(encoded_array.tobytes())

    def from_bytes(self, data: bytes, count=-1, offset=0) -> np.ndarray:
        """
        from bytes to encoded array. The count and offset are from the np.frombuffer function.
        """
        b1 = self.compressor.decompress(data)
        encoded_array = np.frombuffer(b1, dtype=self.dtype_encoded, count=count, offset=offset).reshape(self.chunk_shape)

        return encoded_array


    def encode(self, array: np.ndarray):
        if array.dtype != self.dtype_decoded:
            raise TypeError('The data dtype does not match the assigned dtype_decoded.')

        if self.dtype_encoded != self.dtype_decoded:

            # if data.dtype.kind == 'M':
            #     data = data.astype(self.dtype_encoded)

            if isinstance(self.add_offset, (int, float)):
                array = array - self.add_offset

            if isinstance(self.scale_factor, (int, float)):
                # precision = int(np.abs(np.log10(val['scale_factor'])))
                array = array/self.scale_factor

            if isinstance(self.fillvalue, int) and (self.dtype_decoded.kind == 'f'):
                array[np.isnan(array)] = self.fillvalue

            array = array.astype(self.dtype_encoded)

        return array


    def decode(self, array: np.ndarray):
        """
        encoded array into decode array
        """
        if self.dtype_encoded != self.dtype_decoded:
            array = array.astype(self.dtype_decoded)

            if isinstance(self.fillvalue, int) and (self.dtype_decoded.kind == 'f'):
                array[np.isclose(array, self.fillvalue)] = np.nan

            if isinstance(self.scale_factor, (int, float)):
                array = array * self.scale_factor

            if isinstance(self.add_offset, (int, float)):
                array = array + self.add_offset

        return array


class Variable:
    """

    """
    def __init__(self, var_name, dataset, slices=None):
        """

        """
        self._dataset = dataset
        self._sys_meta = dataset._sys_meta
        self._var_meta = dataset._sys_meta.variables[var_name]
        self._blt = dataset._blt
        self.name = var_name
        self.attrs = Attributes(self._blt, var_name, dataset.writable, dataset._finalizers)
        # self.encoding = msgspec.to_builtins(self._sys_meta.variables[self.name].encoding)
        self.chunk_shape = self._var_meta.chunk_shape
        # self.origin = self._var_meta.origin
        self.dtype_decoded = np.dtype(self._var_meta.dtype_decoded)
        self.dtype_encoded = np.dtype(self._var_meta.dtype_encoded)
        self.fillvalue = self._var_meta.fillvalue
        self.scale_factor = self._var_meta.scale_factor
        self.add_offset = self._var_meta.add_offset
        if hasattr(self._var_meta, 'coords'):
            self.coord_names = self._var_meta.coords
            self.ndims = len(self.coord_names)
        else:
            self.coord_names = (var_name,)
            self.ndims = 1

        # if sel is None:
        #     self._sel = tuple(slice(None, None) for i in range(self.ndims))
        # else:
        #     self._sel = sel

        self._sel = slices

        self._encoder = Encoding(self.chunk_shape, self.dtype_decoded, self.dtype_encoded, self.fillvalue, self.scale_factor, self.add_offset, dataset._compressor)
        self.loc = indexers.LocationIndexer(self)
        self._finalizers = dataset._finalizers
        self.writable = dataset.writable

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

    # @property
    # def coords(self):
    #     return getattr(self._var_meta, 'coords')

    # @property
    # def ndim(self):
    #     return len(getattr(self._var_meta, 'coords'))

    # @property
    # def chunk_shape(self):
    #     return getattr(self._var_meta, 'chunk_shape')

    # @property
    # def dtype_decoded(self):
    #     return np.dtype(getattr(self._var_meta, 'dtype_decoded'))

    # @property
    # def dtype_encoded(self):
    #     return np.dtype(getattr(self._var_meta, 'dtype_encoded'))

    # @property
    # def scale_factor(self):
    #     return getattr(self._var_meta, 'scale_factor')

    # @property
    # def add_offset(self):
    #     return getattr(self._var_meta, 'add_offset')

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
    #     return getattr(self._var_meta, 'fillvalue')

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


    def _make_blank_decoded_array(self, sel, coord_origins):
        """

        """
        new_shape = indexers.determine_final_array_shape(sel, coord_origins, self.shape)

        if self.dtype_decoded.kind == 'f':
            fillvalue = np.nan
        else:
            fillvalue = self.fillvalue

        return np.full(new_shape, fillvalue, self.dtype_decoded)


    def _make_blank_encoded_array(self, sel, coord_origins):
        """

        """
        new_shape = indexers.determine_final_array_shape(sel, coord_origins, self.shape)

        if self.dtype_encoded.kind == 'f':
            fillvalue = np.nan
        else:
            fillvalue = self.fillvalue

        return np.full(new_shape, fillvalue, self.dtype_encoded)


    def rechunk(self, chunk_shape, max_mem=2**27, decoded=False):
        """
        Generator to rechunk the variable into a new chunk shape.
        """
        func = lambda sel: self.get_chunk(sel, decoded=False)

        rechunker1 = rechunker.rechunker(func, self.shape, self.dtype_encoded, self.chunk_shape, chunk_shape, max_mem)

        if decoded:
            for slices, encoded_data in rechunker1:
                yield slices, self._encoder.decode(encoded_data)
        else:
            for slices, encoded_data in rechunker1:
                yield slices, encoded_data


    def __getitem__(self, sel):
        return self.get(sel)


    # def __setitem__(self, sel, data):
    #     """

    #     """
    #     coord_origins = self.get_coord_origins()

    #     blank = self._make_blank_encoded_array(None, coord_origins)

    #     slices = utils.check_sel_input_data(sel, data, coord_origins, self.shape)
    #     for target_chunk, source_chunk, blt_key in indexers.slices_to_chunks_keys(slices, self.name, self.chunk_shape):
    #         b1 = self._blt.get(blt_key)
    #         if b1 is None:
    #             # blank_slices = tuple(slice(0, sc.stop - sc.start) for sc in source_chunk)
    #             new_data = blank.copy()
    #         else:
    #             new_data = self._encoder.from_bytes(b1)

    #         new_data[source_chunk] = self._encoder.encode(data[target_chunk])
    #         self._blt.set(blt_key, self._encoder.to_bytes(new_data))


    def iter_chunks(self):
        """

        """
        coord_origins = self.get_coord_origins()

        blank = self._make_blank_decoded_array(self._sel, coord_origins)

        slices = indexers.index_combo_all(self._sel, coord_origins, self.shape)
        for target_chunk, source_chunk, blt_key in indexers.slices_to_chunks_keys(slices, self.name, self.chunk_shape):
            # print(target_chunk, source_chunk, blt_key)
            b1 = self._blt.get(blt_key)
            if b1 is None:
                blank_slices = tuple(slice(0, sc.stop - sc.start) for sc in source_chunk)
                yield target_chunk, blank[blank_slices]
            else:
                data = self._encoder.decode(self._encoder.from_bytes(b1))
                yield target_chunk, data[source_chunk]


    def get_chunk(self, sel=None, decoded=True, missing_none=False):
        """

        """
        if sel is None:
            sel = self._sel
        coord_origins = self.get_coord_origins()
        slices = indexers.index_combo_all(sel, coord_origins, self.shape)
        starts_chunk = tuple((pc.start//cs) * cs for cs, pc in zip(self.chunk_shape, slices))
        blt_key = utils.make_var_chunk_key(self.name, starts_chunk)
        b1 = self._blt.get(blt_key)
        if missing_none and b1 is None:
            return None
        elif b1 is None:
            if decoded:
                return self._make_blank_decoded_array(sel, coord_origins)
            else:
                return self._make_blank_encoded_array(sel, coord_origins)
        else:
            encoded_data = self._encoder.from_bytes(b1)
            if decoded:
                return self._encoder.decode(encoded_data)
            else:
                return encoded_data


    def get_coord_origins(self):
        """

        """
        if hasattr(self, 'coords'):
            coord_origins = tuple(self._sys_meta.variables[coord].origin for coord in self.coord_names)
        else:
            coord_origins = (self.origin,)

        return coord_origins


    @property
    def coords(self):
        return tuple(self._dataset[coord_name] for coord_name in self.coord_names)


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
        if not hasattr(self, '_data'):
            coord_origins = self.get_coord_origins()

            target = self._make_blank_decoded_array(self._sel, coord_origins)

            for target_chunk, data in self.iter_chunks():
                target[target_chunk] = data

            self._data = target

        return self._data


    def get(self, sel):
        """

        """
        coord_origins = self.get_coord_origins()

        slices = indexers.index_combo_all(sel, coord_origins, self.shape)

        if self._sel is not None:
            slices = tuple(slice(s.start, s.stop) if ss.start is None else slice(ss.start + s.start, ss.start + s.stop) for ss, s in zip(self._sel, slices))

        return CoordinateView(self.name, self._dataset, slices)


    def _add_updated_data(self, chunk_start, chunk_stop, new_origin, updated_data):
        """

        """
        chunk_len = self.chunk_shape[0]

        mem_arr1 = np.full(self.chunk_shape, fill_value=self.fillvalue, dtype=self.dtype_encoded)

        # print(chunk_start)

        chunk_iter = rechunker.chunk_range(chunk_start, chunk_stop, self.chunk_shape, clip_ends=True)
        for chunk in chunk_iter:
            chunk = chunk[0] # Because coords are always 1D
            # print(chunk)

            chunk_start_pos = chunk.start
            chunk_stop_pos = chunk.stop

            chunk_origin = (chunk_start_pos//chunk_len) * chunk_len
            mem_chunk_start_pos = chunk_start_pos - chunk_origin
            mem_chunk_stop_pos = chunk_stop_pos - chunk_origin
            mem_chunk_slice = slice(mem_chunk_start_pos, mem_chunk_stop_pos)

            coord_start_pos = chunk_start_pos - new_origin
            coord_stop_pos = chunk_stop_pos - new_origin
            coord_chunk_slice = slice(coord_start_pos, coord_stop_pos)

            # print(updated_data[coord_chunk_slice])

            mem_arr2 = mem_arr1.copy()
            mem_arr2[mem_chunk_slice] = self._encoder.encode(updated_data[coord_chunk_slice])

            key = utils.make_var_chunk_key(self.name, (chunk_origin,))
            # print(key)

            self._blt.set(key, self._encoder.to_bytes(mem_arr2))

        self._data = updated_data


    def prepend(self, data):
        """
        Prepend data to the start of the coordinate. The extra length will be added to the associated data variables with the fillvalue. One of data or length must be passed. A negative length can be passed to shrink the coordinate from the start.
        """
        if not self.writable:
            raise ValueError('Dataset is not writable.')

        updated_data = utils.prepend_coord_data_checks(data, self.data, self.dtype_decoded, self.step)

        data_diff = updated_data.size - self.data.size

        new_origin = self.origin - data_diff
        chunk_stop = (updated_data.size + new_origin,)

        chunk_start = (new_origin,)

        self._add_updated_data(chunk_start, chunk_stop, new_origin, updated_data)

        self._var_meta.origin = new_origin
        self._var_meta.shape = updated_data.shape


    def append(self, data):
        """
        Append data to the end of the coordinate. The extra length will be added to the associated data variables with the fillvalue. One of data or length must be passed. A negative length can be passed to shrink the coordinate from the end.
        """
        if not self.writable:
            raise ValueError('Dataset is not writable.')

        updated_data = utils.append_coord_data_checks(data, self.data, self.dtype_decoded, self.step)

        shape = (updated_data.size,)

        chunk_start = (self.origin,)
        chunk_stop = shape

        self._add_updated_data(chunk_start, chunk_stop, self.origin, updated_data)

        self._var_meta.shape = shape


    # def resize(self, start=None, end=None):
    #     """
    #     Resize a coordinate. If step is an int or float, then resizing can add or truncate the length. If step is None, then the coordinate can only have the length truncated.
    #     If the coordinate length is reduced, then all data variables associated with the coordinate will have their data truncated.
    #     """
    #     if end is not None:
    #         idx = indexers.loc_index_combo_one(end, self.data)
    #         if self.step is not None:
    #             pass
    #         else:
    #             updated_data =


    @property
    def step(self):
        return getattr(self._var_meta, 'step')

    @property
    def auto_increment(self):
        return getattr(self._var_meta, 'auto_increment')

    @property
    def origin(self):
        return getattr(self._var_meta, 'origin')

    @property
    def shape(self):
        return getattr(self._var_meta, 'shape')



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


class CoordinateView(Coordinate):
    """

    """
    @property
    def shape(self):
        return tuple(s.stop - s.start for s in self._sel)

    @property
    def coords(self):
        return tuple(self._dataset[coord_name][self._sel[i]] for i, coord_name in enumerate(self.coord_names))

    def prepend(self, data):
        """
        Not implemented for views.
        """
        raise NotImplementedError()

    def append(self, data):
        """
        Not implemented for views.
        """
        raise NotImplementedError()

    def get_chunk(self, sel=None, decoded=True, missing_none=False):
        """

        """
        raise NotImplementedError()



class DataVariable(Variable):
    """

    """
    @property
    def data(self):
        coord_origins = self.get_coord_origins()

        target = self._make_blank_decoded_array(self._sel, coord_origins)

        for target_chunk, data in self.iter_chunks():
            target[target_chunk] = data

        return target


    def get(self, sel):
        """

        """
        coord_origins = self.get_coord_origins()

        slices = indexers.index_combo_all(sel, coord_origins, self.shape)

        if self._sel is not None:
            slices = tuple(slice(s.start, s.stop) if ss.start is None else slice(ss.start + s.start, ss.start + s.stop) for ss, s in zip(self._sel, slices))

        return DataVariableView(self.name, self._dataset, slices)


    def __setitem__(self, sel, data):
        """

        """
        if not self.writable:
            raise ValueError('Dataset is not writable.')

        coord_origins = self.get_coord_origins()

        chunk_blank = np.full(self.chunk_shape, self.fillvalue, self.dtype_encoded)

        slices = utils.check_sel_input_data(sel, data, coord_origins, self.shape)

        if self._sel is not None:
            slices = tuple(slice(s.start, s.stop) if ss.start is None else slice(ss.start + s.start, ss.start + s.stop) for ss, s in zip(self._sel, slices))

        for target_chunk, source_chunk, blt_key in indexers.slices_to_chunks_keys(slices, self.name, self.chunk_shape):
            b1 = self._blt.get(blt_key)
            if b1 is None:
                new_data = chunk_blank.copy()
            else:
                new_data = self._encoder.from_bytes(b1)

            new_data[source_chunk] = self._encoder.encode(data[target_chunk])
            self._blt.set(blt_key, self._encoder.to_bytes(new_data))

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


    # @property
    # def coords(self):
    #     return getattr(self._var_meta, 'coords')

    @property
    def shape(self):
        return tuple(self._sys_meta.variables[coord_name].shape[0] for coord_name in self.coord_names)

    # @property
    # def coords(self):
    #     return tuple(self._dataset[coord_name] for coord_name in self.coord_names)




class DataVariableView(DataVariable):
    """

    """
    @property
    def shape(self):
        return tuple(s.stop - s.start for s in self._sel)

    @property
    def coords(self):
        return tuple(self._dataset[coord_name][self._sel[i]] for i, coord_name in enumerate(self.coord_names))

    def get_chunk(self, sel=None, decoded=True, missing_none=False):
        """

        """
        raise NotImplementedError()



































































































