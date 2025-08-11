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
from typing import Set, Optional, Dict, Tuple, List, Union, Any, Iterable
from copy import deepcopy
import itertools
import rechunkit
import pyproj

from . import utils, indexers, dtypes
# import utils, indexers, dtypes

###################################################
### Parameters

attrs_key = '_{var_name}.attrs'

###################################################
### Classes


class Rechunker:
    """

    """
    def __init__(self, var):
        """

        """
        self._var = var


    def guess_chunk_shape(self, target_chunk_size: int):
        """
        Guess an appropriate chunk layout for a dataset, given its shape and
        the size of each element in bytes.  Will allocate chunks only as large
        as target_chunk_size. Chunks will be assigned to the highest composite number within the target_chunk_size. Using composite numbers will benefit the rehunking process as there is a very high likelihood that the least common multiple of two composite numbers will be significantly lower than the product of those two numbers.

        Parameters
        ----------
        target_chunk_size: int
            The maximum size per chunk in bytes.

        Returns
        -------
        tuple of ints
            shape of the chunk
        """
        chunk_shape = rechunkit.guess_chunk_shape(self._var.shape, self._var.dtype_encoded, target_chunk_size)

        return chunk_shape

    def calc_ideal_read_chunk_shape(self, target_chunk_shape: Tuple[int, ...]):
        """
        Calculates the minimum ideal read chunk shape between a source and target.
        """
        return rechunkit.calc_ideal_read_chunk_shape(self._var.chunk_shape, target_chunk_shape)

    def calc_ideal_read_chunk_mem(self, target_chunk_shape: Tuple[int, ...]):
        """
        Calculates the minimum ideal read chunk memory between a source and target.
        """
        ideal_read_chunk_shape = rechunkit.calc_ideal_read_chunk_shape(self._var.chunk_shape, target_chunk_shape)
        return rechunkit.calc_ideal_read_chunk_mem(ideal_read_chunk_shape, self._var.dtype_encoded.itemsize)

    def calc_source_read_chunk_shape(self, target_chunk_shape: Tuple[int, ...], max_mem: int):
        """
        Calculates the optimum read chunk shape given a maximum amount of available memory.

        Parameters
        ----------
        target_chunk_shape: tuple of int
            The target chunk shape
        max_mem: int
            The max allocated memory to perform the chunking operation in bytes.

        Returns
        -------
        optimal chunk shape: tuple of ints
        """
        return rechunkit.calc_source_read_chunk_shape(self._var.chunk_shape, target_chunk_shape, self._var.dtype_encoded.itemsize, max_mem)

    def calc_n_chunks(self):
        """
        Calculate the total number of chunks in the existing variable.
        """
        return rechunkit.calc_n_chunks(self._var.shape, self._var.chunk_shape)

    def calc_n_reads_rechunker(self, target_chunk_shape: Tuple[int, ...], max_mem: int=2**27):
        """
        Calculate the total number of reads and writes using the rechunker.

        Parameters
        ----------
        target_chunk_shape: tuple of ints
            The chunk_shape of the target.
        max_mem: int
            The max allocated memory to perform the chunking operation in bytes. This will only be as large as necessary for an optimum size chunk for the rechunking.

        Returns
        -------
        tuple
            of n_reads, n_writes
        """
        return rechunkit.calc_n_reads_rechunker(self._var.shape, self._var.dtype_encoded, self._var.chunk_shape, target_chunk_shape, max_mem, self._var._sel)


    def rechunk(self, target_chunk_shape, max_mem: int=2**27):
        """
        This method takes a target chunk_shape and max memory size and returns a generator that converts to the new target chunk shape. It optimises the rechunking by using an in-memory numpy ndarray with a size defined by the max_mem.

        Parameters
        ----------
        target_chunk_shape: tuple of ints
            The chunk_shape of the target.
        max_mem: int
            The max allocated memory to perform the chunking operation in bytes. This will only be as large as necessary for an optimum size chunk for the rechunking.

        Returns
        -------
        Generator
            tuple of the target slices to the np.ndarray of data
        """
        self._var.load()

        func = lambda sel: self._var.get_chunk(sel)

        np_dtype = self._var.dtype.np_dtype

        if self._var.dtype.itemsize is None:
            itemsize = self._var.estimate_itemsize()

            if itemsize is None:
                raise ValueError('Variable is empty, so no rechunking is possible.')
        else:
            itemsize = self._var.dtype.itemsize

        if self.dtype.transcoder is None:
            rechunkit1 = rechunkit.rechunker(func, self._var.shape, np_dtype, itemsize, self._var.chunk_shape, target_chunk_shape, max_mem, self._var._sel)

            for slices, data_decoded in rechunkit1:
                yield slices, data_decoded
        else:
            rechunkit1 = rechunkit.rechunker(func, self._var.shape, self.dtype.transcoder.dest_dtype, self.dtype.transcoder.dest_dtype.itemsize, self._var.chunk_shape, target_chunk_shape, max_mem, self._var._sel)

            for slices, data_decoded in rechunkit1:
                yield slices, data_decoded


class Attributes:
    """

    """
    def __init__(self, blt_file, var_name, writable, finalizers):
        """

        """
        key = attrs_key.format(var_name=var_name)
        data = blt_file.get(key)
        if data is None:
            self._data = {}
        else:
            self._data = msgspec.json.decode(data)

        self._blt = blt_file
        # self._var_name = var_name
        finalizers.append(weakref.finalize(self, utils.attrs_finalizer, self._blt, self._data, var_name, writable))
        self.writable = writable

    @property
    def data(self):
        """

        """
        return deepcopy(self._data)

    def set(self, key, value):
        """

        """
        if self.writable:
            try:
                msgspec.json.encode(value)
            except:
                raise ValueError('The value passed is not json serializable.')
            self._data[key] = value
        else:
            raise ValueError('Dataset is not writable.')

    def __setitem__(self, key, value):
        """

        """
        self.set(key, value)

    def get(self, key):
        """

        """
        value = deepcopy(self._data.get(key))

        return value

    def __getitem__(self, key):
        """

        """
        value = self.get(key)

        return value

    def clear(self):
        if self.writable:
            self._data.clear()
        else:
            raise ValueError('Dataset is not writable.')

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def pop(self, key, default=None):
        if self.writable:
            return self._data.pop(key, default)
        else:
            raise ValueError('Dataset is not writable.')

    def update(self, other=()):
        if self.writable:
            try:
                msgspec.json.encode(other)
            except:
                raise ValueError('The values passed are not json serializable.')
            self._data.update(other)
        else:
            raise ValueError('Dataset is not writable.')

    def __delitem__(self, key):
        if self.writable:
            del self._data[key]
        else:
            raise ValueError('Dataset is not writable.')

    def __contains__(self, key):
        return key in self._data

    def __iter__(self):
        return self.keys()

    # def sync(self):
    #     utils.attrs_finalizer(self._blt, self.data, self._var_name)

    # def close(self):
    #     self._finalizer()

    def __repr__(self):
        return self._data.__repr__()


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
            self._cctx = zstd.ZstdCompressor(level=self.compression_level)
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


# TODO - I'll probably need something more specific for cfdb than pyproj CRS directly
# class CRS:
#     """

#     """
#     def __init__(self, crs: pyproj.CRS):
#         """

#         """




class Variable:
    """

    """
    def __init__(self, var_name, dataset, sel=None):
        """

        """
        self._dataset = dataset
        self._sys_meta = dataset._sys_meta
        self._var_meta = dataset._sys_meta.variables[var_name]
        self._blt = dataset._blt
        self._has_load_items = dataset._has_load_items
        self.name = var_name
        self.attrs = Attributes(self._blt, var_name, dataset.writable, dataset._finalizers)
        # self.encoding = msgspec.to_builtins(self._sys_meta.variables[self.name].encoding)
        self.chunk_shape = self._var_meta.chunk_shape
        # self.origin = self._var_meta.origin
        # self.dtype_decoded = np.dtype(self._var_meta.dtype_decoded)
        # self.dtype_encoded = np.dtype(self._var_meta.dtype_encoded)
        # self.fillvalue = self._var_meta.fillvalue
        # self.scale_factor = self._var_meta.scale_factor
        # self.add_offset = self._var_meta.add_offset
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

        self._sel = sel

        # self._encoder = Encoding(self.chunk_shape, self.dtype_decoded, self.dtype_encoded, self.fillvalue, self.scale_factor, self.add_offset, dataset._compressor)
        self.compressor = dataset._compressor
        self.dtype = dtypes.dtype(**self._var_meta.dtype)
        self.loc = indexers.LocationIndexer(self)
        self._finalizers = dataset._finalizers
        self.writable = dataset.writable

        ## Assign all the encodings - should I do this?
        # for name, val in self._encoding_dict.items():
        #     setattr(self, name, val)

    @property
    def is_open(self):
        return self._dataset.is_open

    def __bool__(self):
        return self.is_open


    def _make_blank_sel_array(self, sel, coord_origins, decoded=True):
        """

        """
        new_shape = indexers.determine_final_array_shape(sel, coord_origins, self.shape)

        if self.dtype_decoded.kind == 'f' and decoded:
            fillvalue = np.nan
        else:
            fillvalue = self.fillvalue

        if decoded:
            return np.full(new_shape, fillvalue, self.dtype_decoded)
        else:
            return np.full(new_shape, fillvalue, self.dtype_encoded)


    def _make_blank_chunk_array(self, decoded=True):
        """

        """
        if self.dtype_decoded.kind == 'f' and decoded:
            fillvalue = np.nan
        else:
            fillvalue = self.fillvalue

        if decoded:
            return np.full(self.chunk_shape, fillvalue, self.dtype_decoded)
        else:
            return np.full(self.chunk_shape, fillvalue, self.dtype_encoded)


    def rechunker(self):
        """
        Initialize a Rechunker class to assist in rechunking the variable.
        """
        return Rechunker(self)

    def estimate_itemsize(self):
        """

        """
        if self.dtype.itemsize is None:
            coord_origins = self.get_coord_origins()
            slices = indexers.index_combo_all(None, coord_origins, self.shape)
            starts_chunk = tuple((pc.start//cs) * cs for cs, pc in zip(self.chunk_shape, slices))
            blt_key = utils.make_var_chunk_key(self.name, starts_chunk)
            b1 = self._blt.get(blt_key)
            if b1 is not None:
                itemsize = len(b1)/math.prod(self._var.chunk_shape)
                self.dtype.itemsize = itemsize

        return self.dtype.itemsize


    def __getitem__(self, sel):
        return self.get(sel)


    # def __delitem__(self, sel):
    #     """
    #     Should I implement this as a way to "delete" data? It wouldn't actually delete rather. It would instead set those values to the fillvalue/nan. I should probably delete chunks if the values become nan.
    #     """
        # TODO


    def iter_chunks(self, include_data=True, decoded=True):
        """
        Iterate through the chunks of the variable and return numpy arrays associated with the index slices (Optional). This should be the main way for users to get large amounts of data from a variable. The "ends" of the data will be clipped to the shape of the variable (i.e. not all chunks will be the chunk_shape).

        Parameters
        ----------
        decoded: bool
            Should the data be decoded?
        include_data: bool
            Should the data be included in the output?

        Returns
        -------
        Generator
            tuple of slices of the indexes, numpy array of the data
        """
        self.load()

        coord_origins = self.get_coord_origins()

        blank = self._make_blank_chunk_array(decoded)

        slices = indexers.index_combo_all(self._sel, coord_origins, self.shape)

        if include_data:
            for target_chunk, source_chunk, blt_key in indexers.slices_to_chunks_keys(slices, self.name, self.chunk_shape):
                # print(target_chunk, source_chunk, blt_key)
                b1 = self._blt.get(blt_key)
                if b1 is None:
                    blank_slices = tuple(slice(0, sc.stop - sc.start) for sc in source_chunk)
                    yield target_chunk, blank[blank_slices]
                else:
                    if decoded:
                        data = self._encoder.decode(self._encoder.from_bytes(b1))
                    else:
                        data = self._encoder.from_bytes(b1)

                    yield target_chunk, data[source_chunk]
        else:
            starts = tuple(s.start for s in slices)
            stops = tuple(s.stop for s in slices)
            chunk_iter2 = rechunkit.chunk_range(starts, stops, self.chunk_shape)
            for partial_chunk in chunk_iter2:
                target_chunk = tuple(slice(s.start - start, s.stop - start) for start, s in zip(starts, partial_chunk))

                yield target_chunk

    def __iter__(self):
        return self.iter_chunks()


    def items(self, decoded=True):
        """
        Iterate through all indexes.
        """
        for target_chunk, data in self.iter_chunks(decoded=True):
            data_starts = tuple(s.start for s in target_chunk)
            for index in itertools.product(*(range(s.start, s.stop) for s in target_chunk)):
                data_index = tuple(i - ds for i, ds in zip(index, data_starts))

                yield index, data[data_index]


    def get_chunk(self, sel=None, missing_none=False):
        """
        Get data from one chunk. The method will return the first chunk parsed from sel.

        Parameters
        ----------
        sel: tuple of slices, ints
            The selection based on index positions.
        missing_none: bool
            If chunk is missing, should the method return None or a blank array (filled with the fillvalue)?

        Returns
        -------
        np.ndarray
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
            return self._make_blank_chunk_array()
        else:
            return self.dtype.loads(b1)


    def get_coord_origins(self):
        """
        Get the coordinate origins for the variable.
        """
        if hasattr(self, 'coords'):
            coord_origins = tuple(self._sys_meta.variables[coord].origin for coord in self.coord_names)
        else:
            coord_origins = (self.origin,)

        return coord_origins


    @property
    def coords(self):
        if self._sel is None:
            return tuple(self._dataset[coord_name] for coord_name in self.coord_names)
        else:
            return tuple(self._dataset[coord_name][self._sel[i]] for i, coord_name in enumerate(self.coord_names))


    def __len__(self):
        return math.prod(self.shape)

    def load(self):
        """
        This method only applies if the dataset has been open as an EDataset.
        Load the chunks from the remote into the local file based on the selection. If not selection has been made, then it will load in all the chunks.
        """
        if self._has_load_items:
            coord_origins = self.get_coord_origins()
            slices = indexers.index_combo_all(self._sel, coord_origins, self.shape)
            # keys = list(indexers.slices_to_keys(slices, self.name, self.chunk_shape))
            # print(keys)
            # failures = self._blt.load_items(keys)
            failures = self._blt.load_items(indexers.slices_to_keys(slices, self.name, self.chunk_shape))
            # self._blt.sync()
            if failures:
                raise Exception(failures)


class CoordinateView(Variable):
    """

    """
    @property
    def data(self):
        if not hasattr(self, '_data'):
            coord_origins = self.get_coord_origins()

            target = self._make_blank_sel_array(self._sel, coord_origins)

            for target_chunk, data in self.iter_chunks():
                target[target_chunk] = data

            self._data = target

        return self._data

    @property
    def dim(self):
        """

        """
        return self._var_meta.dim


    def get(self, sel):
        """
        Get a CoordinateView based on the index position(s).
        The parameter sel can be an int, slice, or some combo within a tuple. For example, a tuple of slices (of the index positions).

        Parameters
        ----------
        sel: int, slice, tuple of ints or slices
            It can be an int, slice, or a tuple of ints or slices. Numpy advanced indexing is not implemented.

        Returns
        -------
        cfdb.CoordinateView
        """
        coord_origins = self.get_coord_origins()

        slices = indexers.index_combo_all(sel, coord_origins, self.shape)

        if self._sel is not None:
            slices = tuple(slice(s.start, s.stop) if ss.start is None else slice(ss.start + s.start, ss.start + s.stop) for ss, s in zip(self._sel, slices))

        return CoordinateView(self.name, self._dataset, slices)


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
        return tuple(s.stop - s.start for s in self._sel)




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


class Coordinate(CoordinateView):
    """

    """
    @property
    def shape(self):
        return getattr(self._var_meta, 'shape')


    def _add_updated_data(self, chunk_start, chunk_stop, new_origin, updated_data):
        """

        """
        chunk_len = self.chunk_shape[0]

        mem_arr1 = np.full(self.chunk_shape, fill_value=self.fillvalue, dtype=self.dtype_encoded)

        # print(chunk_start)

        chunk_iter = rechunkit.chunk_range(chunk_start, chunk_stop, self.chunk_shape, clip_ends=True)
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
        Prepend data to the start of the coordinate. The extra length will be added to the associated data variables with the fillvalue.
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
        Append data to the end of the coordinate. The extra length will be added to the associated data variables with the fillvalue.
        """
        if not self.writable:
            raise ValueError('Dataset is not writable.')

        updated_data = utils.append_coord_data_checks(data, self.data, self.dtype_decoded, self.step)

        shape = (updated_data.size,)

        chunk_start = (self.origin,)
        chunk_stop = shape

        self._add_updated_data(chunk_start, chunk_stop, self.origin, updated_data)

        self._var_meta.shape = shape



class DataVariableView(Variable):
    """

    """
    @property
    def data(self):
        coord_origins = self.get_coord_origins()

        target = self._make_blank_sel_array(self._sel, coord_origins)

        for target_chunk, data in self.iter_chunks():
            target[target_chunk] = data

        return target


    def get(self, sel):
        """
        Get a DataVariableView based on the index position(s).
        The parameter sel can be an int, slice, or some combo within a tuple. For example, a tuple of slices (of the index positions).

        Parameters
        ----------
        sel: int, slice, tuple of ints or slices
            It can be an int, slice, or a tuple of ints or slices. Numpy advanced indexing is not implemented.

        Returns
        -------
        cfdb.DataVariableView
        """
        coord_origins = self.get_coord_origins()

        slices = indexers.index_combo_all(sel, coord_origins, self.shape)

        if self._sel is not None:
            slices = tuple(slice(s.start, s.stop) if ss.start is None else slice(ss.start + s.start, ss.start + s.stop) for ss, s in zip(self._sel, slices))

        return DataVariableView(self.name, self._dataset, slices)


    def set(self, sel, data, encode=True):
        """
        Set data based on index positions.
        """
        if not self.writable:
            raise ValueError('Dataset is not writable.')

        coord_origins = self.get_coord_origins()

        chunk_blank = self._make_blank_chunk_array(False)

        slices = indexers.check_sel_input_data(sel, data, coord_origins, self.shape)

        if self._sel is not None:
            slices = tuple(slice(s.start, s.stop) if ss.start is None else slice(ss.start + s.start, ss.start + s.stop) for ss, s in zip(self._sel, slices))

        for target_chunk, source_chunk, blt_key in indexers.slices_to_chunks_keys(slices, self.name, self.chunk_shape):
            b1 = self._blt.get(blt_key)
            if b1 is None:
                new_data = chunk_blank.copy()
            else:
                new_data = self._encoder.from_bytes(b1)

            if encode:
                new_data[source_chunk] = self._encoder.encode(data[target_chunk])
            else:
                new_data[source_chunk] = data[target_chunk]
            self._blt.set(blt_key, self._encoder.to_bytes(new_data))


    # def set_chunk(self, sel, data, encode=True):
    #     """
    #     Set the first chunk associated with the selection.
    #     """
    #     if not self.writable:
    #         raise ValueError('Dataset is not writable.')

    #     if sel is None:
    #         sel = self._sel
    #     coord_origins = self.get_coord_origins()
    #     slices = indexers.index_combo_all(sel, coord_origins, self.shape)
    #     starts_chunk = tuple((pc.start//cs) * cs for cs, pc in zip(self.chunk_shape, slices))
    #     chunk_stop = tuple(min(cs, s - sc) for cs, sc, s in zip(self.chunk_shape, starts_chunk, self.shape))
    #     if data.shape != chunk_stop:
    #         raise ValueError(f'The shape of this chunk should be {chunk_stop}, but the data passed is {data.shape}')

    #     blt_key = utils.make_var_chunk_key(self.name, starts_chunk)
    #     if encode:
    #         self._blt.set(blt_key, self._encoder.to_bytes(self._encoder.encode(data)))
    #     else:
    #         self._blt.set(blt_key, self._encoder.to_bytes(data))


    def __setitem__(self, sel, data):
        """

        """
        self.set(sel, data)


    def groupby(self, coord_names: Union[str, Iterable], max_mem: int=2**27, decoded=True):
        """
        This method takes one or more coord names to group by and returns a generator. This generator will return chunks of data according to these groupings with the associated tuple of slices. The more max_mem provided, the more efficient the chunking.
        This is effectively the rechunking method where each coord name supplied is set to 1 and all other coords are set to their full their full length.

        Parameters
        ----------
        coord_names: str or Iterable
            The coord names to group by.
        max_mem: int
            The max allocated memory to perform the chunking operation in bytes. This will only be as large as necessary for an optimum size chunk for the rechunking.

        Returns
        -------
        Generator
            tuple of the target slices to the np.ndarray of data
        """
        self.load()

        var_coord_names = self.coord_names
        if isinstance(coord_names, str):
            coord_names = (coord_names,)
        else:
            coord_names = tuple(coord_names)

        # checks
        for coord_name in coord_names:
            if coord_name not in var_coord_names:
                raise ValueError(f'{coord_name} is not a coord of this variable.')

        # Build target chunk shape
        target_chunk_shape = []
        for coord in self.coords:
            coord_name = coord.name
            if coord_name in coord_names:
                target_chunk_shape.append(1)
            else:
                target_chunk_shape.append(coord.shape[0])

        # Do the chunking
        func = lambda sel: self.get_chunk(sel, decoded=False)

        rechunkit1 = rechunkit.rechunker(func, self.shape, self.dtype_encoded, self.chunk_shape, tuple(target_chunk_shape), max_mem, self._sel)

        if decoded:
            for slices, encoded_data in rechunkit1:
                yield slices, self._encoder.decode(encoded_data)
        else:
            for slices, encoded_data in rechunkit1:
                yield slices, encoded_data


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
        return tuple(s.stop - s.start for s in self._sel)

    # @property
    # def coords(self):
    #     return tuple(self._dataset[coord_name][self._sel[i]] for i, coord_name in enumerate(self.coord_names))




class DataVariable(DataVariableView):
    """

    """
    @property
    def shape(self):
        return tuple(self._sys_meta.variables[coord_name].shape[0] for coord_name in self.coord_names)




































































































