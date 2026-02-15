#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid interpolation wrapper for cfdb DataVariableView.

Integrates geointerp's GridInterpolator so users can perform regridding,
point sampling, NaN filling, and level regridding directly from a data variable.
"""
import numpy as np
from geointerp import GridInterpolator

from . import data_models


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

    def to_grid(self, grid_res, to_crs=None, bbox=None, order=3, extrapolation='constant', fill_val=np.nan, min_val=None):
        """
        Regrid the data variable onto a new regular grid.

        Parameters
        ----------
        grid_res : float or tuple of float
            Output grid resolution. A single float applies to all spatial
            dimensions; a tuple gives per-dimension resolution in (x, y) or
            (x, y, z) order.
        to_crs : int, str, or None
            Target CRS for the output grid (e.g. 4326). Defaults to the
            dataset's CRS.
        bbox : tuple of float or None
            Bounding box in to_crs coordinates. 2D: (x_min, x_max, y_min,
            y_max). 3D: (x_min, x_max, y_min, y_max, z_min, z_max).
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
        target_points : np.ndarray
            Array of shape (M, 2) or (M, 3) with target point locations in
            (x, y) or (x, y, z) order, in to_crs coordinates.
        to_crs : int, str, or None
            CRS of target_points if different from the dataset's CRS.
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
