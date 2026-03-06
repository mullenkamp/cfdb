"""
Xarray backend engine for cfdb files.

Enables opening cfdb files via xarray:

    import xarray as xr
    ds = xr.open_dataset("file.cfdb", engine="cfdb")
"""
import os
import numpy as np
import threading

from xarray.backends.common import BackendEntrypoint, BackendArray
from xarray.core import indexing
import xarray as xr

from cfdb.main import open_dataset
from cfdb import data_models


class CfdbBackendArray(BackendArray):
    """Lazy array wrapper that reads chunks from a cfdb variable on demand."""

    def __init__(self, cfdb_variable, dtype, shape, lock):
        self._var = cfdb_variable
        self.dtype = dtype
        self.shape = shape
        self.lock = lock

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key):
        with self.lock:
            view = self._var[key]
            return np.asarray(view.data)


class CfdbDataStore:
    """Wraps an open cfdb Dataset, exposing variables/attrs for xarray."""

    def __init__(self, filename, lock=None):
        if lock is None:
            lock = threading.Lock()
        self.lock = lock
        self._ds = open_dataset(filename, flag='r')

    def get_variables(self):
        variables = {}
        ds = self._ds

        for name in ds.coord_names:
            cfdb_var = ds[name]
            np_dtype = cfdb_var.dtype.dtype_decoded
            shape = cfdb_var.shape

            backend_array = CfdbBackendArray(cfdb_var, np_dtype, shape, self.lock)
            data = indexing.LazilyIndexedArray(backend_array)

            attrs = dict(cfdb_var.attrs.data)
            if cfdb_var.units is not None:
                attrs.setdefault('units', cfdb_var.units)
            if cfdb_var.axis is not None:
                attrs['axis'] = cfdb_var.axis.value.upper()

            encoding = {
                'preferred_chunks': {name: cfdb_var.chunk_shape[0]},
            }

            variables[name] = xr.Variable((name,), data, attrs=attrs, encoding=encoding)

        for name in ds.data_var_names:
            cfdb_var = ds[name]
            np_dtype = cfdb_var.dtype.dtype_decoded
            shape = cfdb_var.shape
            dims = cfdb_var.coord_names

            backend_array = CfdbBackendArray(cfdb_var, np_dtype, shape, self.lock)
            data = indexing.LazilyIndexedArray(backend_array)

            attrs = dict(cfdb_var.attrs.data)
            if cfdb_var.units is not None:
                attrs.setdefault('units', cfdb_var.units)

            encoding = {
                'preferred_chunks': {
                    dim: cs for dim, cs in zip(dims, cfdb_var.chunk_shape)
                },
            }

            variables[name] = xr.Variable(dims, data, attrs=attrs, encoding=encoding)

        return variables

    def get_attrs(self):
        attrs = dict(self._ds.attrs.data)
        if self._ds.crs is not None:
            attrs['crs_wkt'] = self._ds.crs.to_wkt()
        return attrs

    def close(self):
        self._ds.close()


class CfdbBackendEntrypoint(BackendEntrypoint):
    """Xarray backend for cfdb files."""

    description = "Open cfdb files using the cfdb engine"
    open_dataset_parameters = ("filename_or_obj", "drop_variables", "lock")

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        lock=None,
    ):
        store = CfdbDataStore(filename_or_obj, lock=lock)

        all_vars = store.get_variables()
        global_attrs = store.get_attrs()

        coord_names = set(store._ds.coord_names)

        if drop_variables is None:
            drop_variables = set()
        elif isinstance(drop_variables, str):
            drop_variables = {drop_variables}
        else:
            drop_variables = set(drop_variables)

        data_vars = {}
        coords = {}
        for vname, var in all_vars.items():
            if vname in drop_variables:
                continue
            if vname in coord_names:
                coords[vname] = var
            else:
                data_vars[vname] = var

        ds = xr.Dataset(data_vars, coords=coords, attrs=global_attrs)
        ds.set_close(store.close)

        return ds

    def guess_can_open(self, filename_or_obj):
        try:
            _, ext = os.path.splitext(filename_or_obj)
        except TypeError:
            return False
        return ext in ('.cfdb',)
