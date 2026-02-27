#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interpolation wrapper for cfdb DataVariableView.

Integrates geointerp's GridInterpolator and PointInterpolator so users can
perform regridding, point sampling, NaN filling, and level regridding
directly from a data variable.

GridInterp is used for 'grid' dataset types (regular grids).
PointInterp is used for 'ts_ortho' dataset types (scattered point data).
"""
import numpy as np
import shapely
from geointerp import GridInterpolator, PointInterpolator

from . import data_models


###################################################
### Helpers


def _coerce_target_points(target_points):
    """
    Accept target_points as an ndarray of coordinates or an array/list
    of shapely Point geometries. Returns an ndarray of shape (M, 2) or (M, 3).
    """
    target_points = np.asarray(target_points)
    if target_points.dtype.kind == 'O':  # shapely geometry objects
        target_points = shapely.get_coordinates(target_points)
    return target_points


def _extract_grid_params(data_var):
    """
    Derive grid_res, bbox, to_crs from a grid DataVariable/DataVariableView.

    Returns
    -------
    dict with keys: grid_res, bbox, to_crs
    """
    dataset = data_var._dataset

    if dataset.crs is None:
        raise ValueError("Destination DataVariable's dataset must have a CRS defined.")

    # Find x and y coordinates from axis metadata
    x_name = None
    y_name = None
    for coord_name in data_var.coord_names:
        var_meta = dataset._sys_meta.variables[coord_name]
        if isinstance(var_meta, data_models.CoordinateVariable) and var_meta.axis is not None:
            if var_meta.axis.value == 'x':
                x_name = coord_name
            elif var_meta.axis.value == 'y':
                y_name = coord_name

    if x_name is None or y_name is None:
        raise ValueError("Destination DataVariable must have x and y coordinates with axis metadata.")

    x_data = dataset[x_name].data.ravel()
    y_data = dataset[y_name].data.ravel()

    grid_res = float(abs(x_data[1] - x_data[0]))
    bbox = (float(x_data.min()), float(x_data.max()), float(y_data.min()), float(y_data.max()))
    to_crs = dataset.crs

    return {'grid_res': grid_res, 'bbox': bbox, 'to_crs': to_crs}


def _extract_point_params(data_var):
    """
    Derive target_points, to_crs from a ts_ortho DataVariable/DataVariableView.

    Returns
    -------
    dict with keys: target_points, to_crs
    """
    dataset = data_var._dataset

    if dataset.crs is None:
        raise ValueError("Destination DataVariable's dataset must have a CRS defined.")

    # Find xy geometry coordinate from axis metadata
    xy_name = None
    for coord_name in data_var.coord_names:
        var_meta = dataset._sys_meta.variables[coord_name]
        if isinstance(var_meta, data_models.CoordinateVariable) and var_meta.axis is not None:
            if var_meta.axis.value == 'xy':
                xy_name = coord_name

    if xy_name is None:
        raise ValueError("Destination DataVariable must have an xy geometry coordinate.")

    geom_data = dataset[xy_name].data
    target_points = shapely.get_coordinates(geom_data)
    to_crs = dataset.crs

    return {'target_points': target_points, 'to_crs': to_crs}


###################################################
### GridInterp


class GridInterp:
    """
    Wrapper around geointerp.GridInterpolator that auto-detects CRS and
    spatial/temporal coordinate names from dataset axis metadata, handles
    dimension reordering, and iterates over time steps when present.

    All interpolation methods are generators that yield (time_value, result)
    tuples. When there is no time dimension, a single tuple is yielded with
    time_value=None.

    Parameters
    ----------
    data_var : DataVariableView or DataVariable
        The data variable to interpolate.
    x : str or None
        Name of the x coordinate. Auto-detected from axis metadata if None.
    y : str or None
        Name of the y coordinate. Auto-detected from axis metadata if None.
    z : str or None
        Name of the z coordinate. Auto-detected from axis metadata if None.
    time : str or None
        Name of the time coordinate. Auto-detected from axis metadata if None.

    Raises
    ------
    ValueError
        If the dataset has no CRS defined or x/y coordinates cannot be determined.
    """

    def __init__(self, data_var, x=None, y=None, z=None, time=None):
        self._data_var = data_var
        self._dataset = data_var._dataset

        self._resolve_axes(x, y, z, time)

        if self._dataset.crs is None:
            raise ValueError("Dataset must have a CRS defined.")
        if self._x_name is None or self._y_name is None:
            raise ValueError("Could not determine x and y coordinates. Pass them explicitly.")

        self._compute_spatial_transpose()

    def _resolve_axes(self, x, y, z, time):
        detected = {'x': x, 'y': y, 'z': z, 't': time}

        for coord_name in self._data_var.coord_names:
            var_meta = self._dataset._sys_meta.variables[coord_name]
            if isinstance(var_meta, data_models.CoordinateVariable) and var_meta.axis is not None:
                axis_key = var_meta.axis.value
                if axis_key in detected and detected[axis_key] is None:
                    detected[axis_key] = coord_name

        self._x_name = detected['x']
        self._y_name = detected['y']
        self._z_name = detected['z']
        self._time_name = detected['t']

    def _compute_spatial_transpose(self):
        spatial_coord_names = [c for c in self._data_var.coord_names if c != self._time_name]

        if self._z_name is not None:
            expected_order = [self._z_name, self._y_name, self._x_name]
        else:
            expected_order = [self._y_name, self._x_name]

        if spatial_coord_names == expected_order:
            self._spatial_axes = None
        else:
            self._spatial_axes = tuple(spatial_coord_names.index(name) for name in expected_order)

        if self._time_name is not None:
            self._time_dim_idx = list(self._data_var.coord_names).index(self._time_name)
        else:
            self._time_dim_idx = None

    def _get_source_coords(self):
        x_arr = self._dataset[self._x_name].data.ravel()
        y_arr = self._dataset[self._y_name].data.ravel()

        if self._z_name is not None:
            z_arr = self._dataset[self._z_name].data.ravel()
            return (x_arr, y_arr, z_arr)

        return (x_arr, y_arr)

    def _prepare_spatial(self, data):
        data = data.squeeze(axis=self._time_dim_idx)
        if self._spatial_axes is not None:
            data = np.transpose(data, self._spatial_axes)
        return data

    def _iter_time_slices(self, extra_data_var=None):
        if self._time_name is None:
            data = self._data_var.data
            if self._spatial_axes is not None:
                data = np.transpose(data, self._spatial_axes)
            if extra_data_var is not None:
                extra_data = extra_data_var.data
                if self._spatial_axes is not None:
                    extra_data = np.transpose(extra_data, self._spatial_axes)
                yield (None, data, extra_data)
            else:
                yield (None, data)
        else:
            time_data = self._dataset[self._time_name].data

            if extra_data_var is not None:
                for (slices, spatial_data), (_, extra_spatial) in zip(
                    self._data_var.groupby(self._time_name),
                    extra_data_var.groupby(self._time_name),
                ):
                    t_idx = slices[self._time_dim_idx].start
                    yield (time_data[t_idx], self._prepare_spatial(spatial_data), self._prepare_spatial(extra_spatial))
            else:
                for slices, spatial_data in self._data_var.groupby(self._time_name):
                    t_idx = slices[self._time_dim_idx].start
                    yield (time_data[t_idx], self._prepare_spatial(spatial_data))

    def to_grid(self, grid_res=None, to_crs=None, bbox=None, order=3, extrapolation='constant', fill_val=np.nan, min_val=None):
        """
        Regrid the data variable onto a new regular grid.

        The first argument can be a DataVariableView from a grid dataset,
        in which case grid_res, bbox, and to_crs are derived from it.

        Parameters
        ----------
        grid_res : float, tuple of float, or DataVariableView
            Output grid resolution, or a destination DataVariable to
            derive target grid parameters from.
        to_crs : int, str, or None
            Target CRS for the output grid (e.g. 4326). Defaults to the
            dataset's CRS. Ignored if grid_res is a DataVariable.
        bbox : tuple of float or None
            Bounding box in to_crs coordinates. Ignored if grid_res is a
            DataVariable.
        order : int
            Spline interpolation order (0-5). 0=nearest, 1=linear, 3=cubic.
        extrapolation : str
            Mode for values outside grid: 'constant', 'nearest', 'reflect',
            'mirror', or 'wrap'.
        fill_val : float
            Fill value for 'constant' extrapolation.
        min_val : float or None
            Floor value; results below this are clamped.

        Yields
        ------
        tuple of (time_value, np.ndarray)
            time_value is None when there is no time dimension. The array
            is the regridded 2D or 3D spatial data.
        """
        from .support_classes import DataVariableView

        if isinstance(grid_res, DataVariableView):
            params = _extract_grid_params(grid_res)
            grid_res = params['grid_res']
            to_crs = params['to_crs']
            bbox = params['bbox']

        source_coords = self._get_source_coords()
        gi = GridInterpolator(from_crs=self._dataset.crs)
        interp_func = gi.to_grid(source_coords, grid_res, to_crs=to_crs, bbox=bbox, order=order, extrapolation=extrapolation, fill_val=fill_val, min_val=min_val)

        for time_val, data in self._iter_time_slices():
            yield (time_val, interp_func(data))

    def to_points(self, target_points, to_crs=None, order=3, min_val=None):
        """
        Sample the data variable at specific target point locations.

        Parameters
        ----------
        target_points : np.ndarray, list of shapely Points, or DataVariableView
            Array of shape (M, 2) or (M, 3) with target point locations,
            a list/array of shapely Point geometries, or a DataVariableView
            from a ts_ortho dataset to derive target points from.
        to_crs : int, str, or None
            CRS of target_points if different from the dataset's CRS.
            Ignored if target_points is a DataVariable.
        order : int
            Spline interpolation order (0-5). 0=nearest, 1=linear, 3=cubic.
        min_val : float or None
            Floor value; results below this are clamped.

        Yields
        ------
        tuple of (time_value, np.ndarray)
            time_value is None when there is no time dimension. The array
            is a 1D array of shape (M,) with interpolated values at each
            target point.
        """
        from .support_classes import DataVariableView

        if isinstance(target_points, DataVariableView):
            params = _extract_point_params(target_points)
            target_points = params['target_points']
            to_crs = params['to_crs']
        else:
            target_points = _coerce_target_points(target_points)

        source_coords = self._get_source_coords()
        gi = GridInterpolator(from_crs=self._dataset.crs)
        interp_func = gi.to_points(source_coords, target_points, to_crs=to_crs, order=order, min_val=min_val)

        for time_val, data in self._iter_time_slices():
            yield (time_val, interp_func(data))

    def interp_na(self, method='linear', min_val=None):
        """
        Fill NaN values in the data variable via spatial interpolation.

        Parameters
        ----------
        method : str
            Interpolation method: 'nearest', 'linear', or 'cubic'.
        min_val : float or None
            Floor value; results below this are clamped.

        Yields
        ------
        tuple of (time_value, np.ndarray)
            time_value is None when there is no time dimension. The array
            has the same shape as the source spatial data with NaN values
            filled.
        """
        source_coords = self._get_source_coords()
        gi = GridInterpolator(from_crs=self._dataset.crs)
        fill_func = gi.interp_na(source_coords, method=method, min_val=min_val)

        for time_val, data in self._iter_time_slices():
            yield (time_val, fill_func(data))

    def regrid_levels(self, target_levels, source_levels, axis=0, method='linear'):
        """
        Regrid data from variable vertical levels onto fixed target levels.

        This is useful for data on terrain-following or sigma coordinates
        where the actual level heights vary at each grid point.

        Parameters
        ----------
        target_levels : array-like of float
            Target level values to interpolate onto (must be monotonically
            increasing).
        source_levels : str
            Name of a data variable in the dataset that contains the source
            level values. Must have the same shape as this data variable.
        axis : int
            The axis in the data that corresponds to the vertical/level
            dimension.
        method : str
            Interpolation method. Currently only 'linear' is supported.

        Yields
        ------
        tuple of (time_value, np.ndarray)
            time_value is None when there is no time dimension. The array
            has the vertical axis replaced by the target levels.

        Raises
        ------
        TypeError
            If source_levels is not a string.
        ValueError
            If source_levels does not exist as a data variable in the dataset.
        """
        if not isinstance(source_levels, str):
            raise TypeError("source_levels must be a string (name of a data variable in the dataset).")
        if source_levels not in self._dataset.data_var_names:
            raise ValueError(f"'{source_levels}' is not a data variable in the dataset.")

        levels_var = self._dataset[source_levels]
        gi = GridInterpolator(from_crs=self._dataset.crs)
        regrid_func = gi.regrid_levels(np.asarray(target_levels, dtype=float), axis=axis, method=method)

        for time_val, data, levels_data in self._iter_time_slices(extra_data_var=levels_var):
            yield (time_val, regrid_func(data, levels_data))


###################################################
### PointInterp


class PointInterp:
    """
    Wrapper around geointerp.PointInterpolator that auto-detects CRS and
    spatial/temporal coordinate names from dataset axis metadata for
    ts_ortho (scattered point) datasets.

    All interpolation methods are generators that yield (time_value, result)
    tuples. When there is no time dimension, a single tuple is yielded with
    time_value=None.

    Parameters
    ----------
    data_var : DataVariableView or DataVariable
        The data variable to interpolate.
    xy : str or None
        Name of the xy geometry coordinate. Auto-detected from axis metadata if None.
    z : str or None
        Name of the z coordinate. Auto-detected from axis metadata if None.
    time : str or None
        Name of the time coordinate. Auto-detected from axis metadata if None.

    Raises
    ------
    ValueError
        If the dataset has no CRS defined or xy coordinate cannot be determined.
    """

    def __init__(self, data_var, xy=None, z=None, time=None):
        self._data_var = data_var
        self._dataset = data_var._dataset

        self._resolve_axes(xy, z, time)

        if self._dataset.crs is None:
            raise ValueError("Dataset must have a CRS defined.")
        if self._xy_name is None:
            raise ValueError("Could not determine xy geometry coordinate. Pass it explicitly.")

    def _resolve_axes(self, xy, z, time):
        detected = {'xy': xy, 'z': z, 't': time}

        for coord_name in self._data_var.coord_names:
            var_meta = self._dataset._sys_meta.variables[coord_name]
            if isinstance(var_meta, data_models.CoordinateVariable) and var_meta.axis is not None:
                axis_key = var_meta.axis.value
                if axis_key in detected and detected[axis_key] is None:
                    detected[axis_key] = coord_name

        self._xy_name = detected['xy']
        self._z_name = detected['z']
        self._time_name = detected['t']

    def _get_source_points(self):
        """Extract (x, y) coordinates from shapely geometry objects."""
        geom_data = self._dataset[self._xy_name].data
        return shapely.get_coordinates(geom_data)

    def _get_coord_order(self):
        """Get the ordering of non-time coordinates in the data variable."""
        spatial_coord_names = [c for c in self._data_var.coord_names if c != self._time_name]
        return spatial_coord_names

    def _iter_time_slices(self):
        """
        Iterate over time steps, yielding (time_value, data) tuples.

        For the no-z case, data is 1D (n_points,) per time step.
        With z, data is 2D (z, n_points) or (n_points, z) depending on
        coordinate ordering — always transposed to (z, n_points).
        """
        spatial_coords = self._get_coord_order()

        # Determine if we need to transpose z and point dims
        if self._z_name is not None:
            expected_order = [self._z_name, self._xy_name]
            needs_transpose = spatial_coords != expected_order
        else:
            needs_transpose = False

        if self._time_name is None:
            data = self._data_var.data
            if needs_transpose:
                # Find the axes to transpose
                axes = tuple(spatial_coords.index(name) for name in [self._z_name, self._xy_name])
                data = np.transpose(data, axes)
            yield (None, data)
        else:
            time_data = self._dataset[self._time_name].data
            time_dim_idx = list(self._data_var.coord_names).index(self._time_name)

            for slices, spatial_data in self._data_var.groupby(self._time_name):
                t_idx = slices[time_dim_idx].start
                data = spatial_data.squeeze(axis=time_dim_idx)
                if needs_transpose:
                    axes = tuple(spatial_coords.index(name) for name in [self._z_name, self._xy_name])
                    data = np.transpose(data, axes)
                yield (time_data[t_idx], data)

    def to_grid(self, grid_res=None, to_crs=None, bbox=None, method='linear', extrapolation='constant', fill_val=np.nan, min_val=None):
        """
        Interpolate scattered point data onto a regular grid.

        The first argument can be a DataVariableView from a grid dataset,
        in which case grid_res, bbox, and to_crs are derived from it.

        Parameters
        ----------
        grid_res : float, or DataVariableView
            Output grid resolution, or a destination DataVariable.
        to_crs : int, str, or None
            Target CRS for the output grid.
        bbox : tuple of float or None
            Bounding box in to_crs coordinates.
        method : str
            'nearest', 'linear', or 'cubic'.
        extrapolation : str
            'constant' or 'nearest'.
        fill_val : float
            Fill value for 'constant' extrapolation.
        min_val : float or None
            Floor value.

        Yields
        ------
        tuple of (time_value, np.ndarray)
            time_value is None when there is no time dimension.
        """
        from .support_classes import DataVariableView

        if isinstance(grid_res, DataVariableView):
            params = _extract_grid_params(grid_res)
            grid_res = params['grid_res']
            to_crs = params['to_crs']
            bbox = params['bbox']

        source_points = self._get_source_points()
        pi = PointInterpolator(from_crs=self._dataset.crs)

        if self._z_name is None:
            interp_func = pi.to_grid(source_points, grid_res, to_crs=to_crs, bbox=bbox, method=method, extrapolation=extrapolation, fill_val=fill_val, min_val=min_val)

            for time_val, data in self._iter_time_slices():
                yield (time_val, interp_func(data))
        else:
            z_data = self._dataset[self._z_name].data.ravel()
            interp_func = pi.to_grid(source_points, grid_res, to_crs=to_crs, bbox=bbox, method=method, extrapolation=extrapolation, fill_val=fill_val, min_val=min_val)

            for time_val, data in self._iter_time_slices():
                # data shape: (nz, n_points)
                grids = []
                for zi in range(len(z_data)):
                    grids.append(interp_func(data[zi]))
                yield (time_val, np.stack(grids))

    def to_points(self, target_points, to_crs=None, method='linear', min_val=None):
        """
        Interpolate scattered point data at target point locations.

        Parameters
        ----------
        target_points : np.ndarray, list of shapely Points, or DataVariableView
            Target point locations, shapely Points, or a DataVariableView
            from a ts_ortho dataset.
        to_crs : int, str, or None
            CRS of target_points.
        method : str
            'nearest', 'linear', or 'cubic'.
        min_val : float or None
            Floor value.

        Yields
        ------
        tuple of (time_value, np.ndarray)
            time_value is None when there is no time dimension.
        """
        from .support_classes import DataVariableView

        if isinstance(target_points, DataVariableView):
            params = _extract_point_params(target_points)
            target_points = params['target_points']
            to_crs = params['to_crs']
        else:
            target_points = _coerce_target_points(target_points)

        source_points = self._get_source_points()
        pi = PointInterpolator(from_crs=self._dataset.crs)

        if self._z_name is None:
            interp_func = pi.to_points(source_points, target_points, to_crs=to_crs, method=method, min_val=min_val)

            for time_val, data in self._iter_time_slices():
                yield (time_val, interp_func(data))
        else:
            z_data = self._dataset[self._z_name].data.ravel()
            interp_func = pi.to_points(source_points, target_points, to_crs=to_crs, method=method, min_val=min_val)

            for time_val, data in self._iter_time_slices():
                # data shape: (nz, n_points)
                results = []
                for zi in range(len(z_data)):
                    results.append(interp_func(data[zi]))
                yield (time_val, np.stack(results))

    def interp_na(self, **kwargs):
        """Not supported for point data."""
        raise NotImplementedError("interp_na is not supported for ts_ortho (point) datasets. It is only available for grid datasets.")

    def regrid_levels(self, **kwargs):
        """Not supported for point data."""
        raise NotImplementedError("regrid_levels is not supported for ts_ortho (point) datasets. It is only available for grid datasets.")
