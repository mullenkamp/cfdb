# Implementation Plan for API Improvements

This document outlines the steps to improve the `cfdb` public API, ordered from simplest to most complex.

## 1. Standardize `__bool__` Behavior (Completed)

Currently, `DatasetBase.__bool__` returns `self.is_open`. Python containers typically use `__bool__` to indicate if the container is not empty.

**Target File:** `cfdb/main.py`

**Steps:**
1.  Modify `DatasetBase.__bool__` to return `len(self) > 0`.
2.  Ensure `is_open` remains accessible as a property for checking file status.

## 2. Complete Docstrings (Completed)

Several public methods and classes lack documentation.

**Target Files:** `cfdb/main.py`, `cfdb/support_classes.py`

**Steps:**
1.  **`cfdb/main.py`**:
    *   Add docstrings to `Dataset.__init__`, `Dataset.__repr__`.
    *   Add docstrings to `DatasetView` class and its methods.
2.  **`cfdb/support_classes.py`**:
    *   Add docstrings to `Variable`, `Coordinate`, and `DataVariable` classes and methods where missing (e.g., `__init__`, `__repr__`).

## 3. Add Numpy Interoperability (`__array__`) (Completed)

Allow `Variable`, `Coordinate`, and `DataVariable` instances to be passed directly to numpy functions (e.g., `np.array(ds['var'])`, `np.mean(ds['var'])`).

**Target File:** `cfdb/support_classes.py`

**Steps:**
1.  Implement `__array__(self, dtype=None)` method in the `Variable` class (or `DataVariableView`/`CoordinateView` if more appropriate).
    *   The method should return `self.data` (which returns a numpy array).
    *   Respect the optional `dtype` argument if possible.
2.  Add a `values` property to `Variable` as an alias for `data` to align with pandas/xarray conventions.

## 4. Implement Lazy `to_xarray()` (Read-Only Backend)

Enable `cfdb` to serve as a lazy-loading backend for `xarray`. The `to_xarray()` method should return an `xarray.Dataset` where data is loaded from disk only when accessed (sliced or computed), preventing full RAM dumps.

**Target Files:** `cfdb/support_classes.py` (or new `cfdb/xarray_utils.py`), `cfdb/main.py`

**Steps:**
1.  **Define `CFDBBackendArray` Wrapper**:
    *   Create a class (e.g., `CFDBBackendArray`) that wraps a `cfdb.Variable`.
    *   Implement `__getitem__`: It must accept slicing keys, delegate to the underlying `cfdb.Variable` (which already supports slicing), and return a **numpy array**.
    *   Implement `shape` and `dtype` properties.
    *   *Note:* This wrapper allows `xarray` to see the `cfdb` variable as a "duck array" that it can index lazily.

2.  **Implement `to_xarray()` in `Dataset`**:
    *   Iterate over all coordinates and data variables in the `cfdb.Dataset`.
    *   For each variable:
        *   Instantiate `CFDBBackendArray(variable)`.
        *   Create an `xarray.Variable` using the wrapper, passing the correct dimensions and attributes.
    *   Construct and return an `xarray.Dataset` composed of these variables.
    *   Ensure the implementation requires `xarray` but handles `ImportError` if not installed.

3.  **Optional: Register as Xarray Backend (Future)**:
    *   Once the wrapper is working, this logic could easily be extended to register `cfdb` as a formal `xarray` engine (e.g., `xr.open_dataset('file.cfdb', engine='cfdb')`), but `ds.to_xarray()` is the primary goal here.
