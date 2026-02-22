#!/usr/bin/env python3
"""
Combine multiple cfdb datasets into a single output file.
"""
import itertools
import pathlib

import numpy as np
import pyproj

from cfdb import dtypes as dtypes_mod
from cfdb.main import Dataset, DatasetView, open_dataset


def _open_inputs(datasets):
    """
    Open input datasets. Returns (list of datasets, list of ones we opened).
    """
    opened = []
    ds_list = []
    for ds in datasets:
        if isinstance(ds, str | pathlib.Path):
            d = open_dataset(ds, "r")
            opened.append(d)
            ds_list.append(d)
        elif isinstance(ds, Dataset | DatasetView):
            ds_list.append(ds)
        else:
            msg = f"datasets must contain file paths or Dataset objects, got {type(ds)}"
            raise TypeError(msg)
    return ds_list, opened


def _validate_dataset_types(ds_list):
    """
    Validate all datasets have the same dataset_type.
    """
    first_type = ds_list[0]._sys_meta.dataset_type
    for i, ds in enumerate(ds_list[1:], 1):
        if ds._sys_meta.dataset_type != first_type:
            msg = (
                f"All datasets must have the same dataset_type. "
                f"Dataset 0 has {first_type.value}, dataset {i} has {ds._sys_meta.dataset_type.value}."
            )
            raise ValueError(msg)
    return first_type


def _validate_crs(ds_list):
    """
    Validate all datasets with a CRS agree.
    """
    crs_values = []
    for ds in ds_list:
        crs_values.append(ds._sys_meta.crs)

    non_none = [c for c in crs_values if c is not None]
    if non_none:
        first = non_none[0]
        for _i, c in enumerate(crs_values):
            if c is not None and c != first:
                msg = "CRS mismatch between datasets."
                raise ValueError(msg)
        return first
    return None


def _compute_combined_coords(ds_list):
    """
    Compute the union of coordinates across all datasets.
    Returns a list of dicts with coord info.
    """
    # Collect all coord names in order (preserving first-seen order)
    coord_names_ordered = []
    seen = set()
    for ds in ds_list:
        for name in ds.coord_names:
            if name not in seen:
                coord_names_ordered.append(name)
                seen.add(name)

    coord_infos = []
    for coord_name in coord_names_ordered:
        # Gather info from all datasets that have this coord
        first_dtype_dict = None
        first_axis = None
        first_chunk_shape = None
        first_step = None
        all_steps_agree = True
        all_data = []

        for ds in ds_list:
            if coord_name not in ds.coord_names:
                continue
            coord = ds[coord_name]
            dtype_dict = coord.dtype.to_dict()
            axis = coord._var_meta.axis

            if first_dtype_dict is None:
                first_dtype_dict = dtype_dict
                first_axis = axis
                first_chunk_shape = coord.chunk_shape
                first_step = coord._var_meta.step
            else:
                if dtype_dict != first_dtype_dict:
                    msg = (
                        f"Coordinate {coord_name} has incompatible dtypes across datasets: "
                        f"{first_dtype_dict} vs {dtype_dict}"
                    )
                    raise ValueError(msg)
                if axis != first_axis:
                    msg = (
                        f"Coordinate {coord_name} has incompatible axis across datasets: "
                        f"{first_axis} vs {axis}"
                    )
                    raise ValueError(msg)
                if coord._var_meta.step != first_step:
                    all_steps_agree = False

            all_data.append(coord.data)

        # Compute sorted union
        dtype_obj = dtypes_mod.dtype(**first_dtype_dict)
        if dtype_obj.kind == "G":
            # Geometry: WKT-based set union, preserve order
            wkt_set = {}
            for data in all_data:
                encoded = dtype_obj.encode(data)
                for wkt, geo in zip(encoded, data, strict=False):
                    if wkt not in wkt_set:
                        wkt_set[wkt] = geo
            combined_data = np.array(list(wkt_set.values()), dtype=dtype_obj.dtype_decoded)
        else:
            # Numeric/datetime/string: sorted union
            combined = all_data[0]
            for d in all_data[1:]:
                combined = np.union1d(combined, d)
            combined_data = combined

        # Determine step
        step = first_step
        if not all_steps_agree:
            step = None
        elif step is not None and len(combined_data) > 1 and dtype_obj.kind in ("f", "u", "i", "M"):
            # Verify regularity is preserved in the union
            diff = np.diff(combined_data)
            if dtype_obj.kind == "f":
                if not np.allclose(float(step), diff):
                    step = None
            else:
                int_step = int(step)
                if not np.all(np.equal(int_step, diff.astype(int) if dtype_obj.kind == "M" else diff)):
                    step = None

        coord_infos.append({
            "name": coord_name,
            "data": combined_data,
            "dtype_dict": first_dtype_dict,
            "axis": first_axis,
            "chunk_shape": first_chunk_shape,
            "step": step,
        })

    return coord_infos


def _collect_data_vars(ds_list, include_data_vars, exclude_data_vars):
    """
    Collect data variable info across all datasets, applying include/exclude filters.
    """
    # Gather all data var names (preserving first-seen order)
    all_names_ordered = []
    seen = set()
    for ds in ds_list:
        for name in ds.data_var_names:
            if name not in seen:
                all_names_ordered.append(name)
                seen.add(name)

    # Apply include/exclude
    if include_data_vars is not None:
        if isinstance(include_data_vars, str):
            include_data_vars = [include_data_vars]
        all_names_ordered = [n for n in all_names_ordered if n in set(include_data_vars)]
    elif exclude_data_vars is not None:
        if isinstance(exclude_data_vars, str):
            exclude_data_vars = [exclude_data_vars]
        exclude_set = set(exclude_data_vars)
        all_names_ordered = [n for n in all_names_ordered if n not in exclude_set]

    var_infos = []
    for var_name in all_names_ordered:
        first_coords = None
        first_dtype_dict = None
        first_chunk_shape = None

        for ds in ds_list:
            if var_name not in ds.data_var_names:
                continue
            var = ds[var_name]
            coords_tuple = var.coord_names
            dtype_dict = var.dtype.to_dict()

            if first_coords is None:
                first_coords = coords_tuple
                first_dtype_dict = dtype_dict
                first_chunk_shape = var.chunk_shape
            else:
                if coords_tuple != first_coords:
                    msg = (
                        f"Data variable {var_name} has incompatible coords across datasets: "
                        f"{first_coords} vs {coords_tuple}"
                    )
                    raise ValueError(msg)
                if dtype_dict != first_dtype_dict:
                    msg = (
                        f"Data variable {var_name} has incompatible dtypes across datasets: "
                        f"{first_dtype_dict} vs {dtype_dict}"
                    )
                    raise ValueError(msg)

        var_infos.append({
            "name": var_name,
            "coords": first_coords,
            "dtype_dict": first_dtype_dict,
            "chunk_shape": first_chunk_shape,
        })

    return var_infos


def _write_data(ds_list, output_ds, var_infos, coord_infos, overlap):
    """
    Write data from source datasets into the output dataset.
    Uses iter_chunks which correctly handles DatasetView selections.
    """
    # Build coord lookup: name -> combined data array
    coord_data_map = {ci["name"]: ci["data"] for ci in coord_infos}

    for var_info in var_infos:
        var_name = var_info["name"]
        out_var = output_ds[var_name]
        coord_names = var_info["coords"]

        for ds in ds_list:
            if var_name not in ds.data_var_names:
                continue

            src_var = ds[var_name]

            # Precompute index mapping for each dimension
            # Maps source coord position (within the view) to output coord position
            index_maps = []
            for coord_name in coord_names:
                src_coord = ds[coord_name]
                src_data = src_coord.data  # Respects view selection
                out_data = coord_data_map[coord_name]

                if src_coord.dtype.kind == "G":
                    ci_match = next(
                        c for c in coord_infos if c["name"] == coord_name
                    )
                    out_dtype = dtypes_mod.dtype(**ci_match["dtype_dict"])
                    out_encoded = out_dtype.encode(out_data)
                    src_encoded = out_dtype.encode(src_data)
                    wkt_to_idx = {wkt: i for i, wkt in enumerate(out_encoded)}
                    mapping = np.array([wkt_to_idx[wkt] for wkt in src_encoded], dtype=np.intp)
                elif src_coord.dtype.kind == "f":
                    mapping = np.array([
                        int(np.nonzero(np.isclose(out_data, v))[0][0]) for v in src_data
                    ], dtype=np.intp)
                else:
                    mapping = np.searchsorted(out_data, src_data).astype(np.intp)

                index_maps.append(mapping)

            # Transfer data chunk by chunk using iter_chunks
            # iter_chunks handles selections/views transparently
            for target_chunk, data in src_var.iter_chunks(include_data=True):
                if data is None:
                    continue

                # target_chunk is relative to the view start (0-based)
                # Map source chunk positions to output positions
                out_slices = []
                for dim_i in range(len(coord_names)):
                    tc = target_chunk[dim_i]
                    src_indices = np.arange(tc.start, tc.stop)

                    # Clip to valid range of the mapping
                    valid_mask = src_indices < len(index_maps[dim_i])
                    if not np.all(valid_mask):
                        src_indices = src_indices[valid_mask]

                    if len(src_indices) == 0:
                        out_slices.append(None)
                        continue

                    out_indices = index_maps[dim_i][src_indices]
                    out_slices.append((int(out_indices[0]), int(out_indices[-1]) + 1, out_indices))

                if any(s is None for s in out_slices):
                    continue

                # Check if all dimensions map contiguously
                all_contiguous = all(
                    len(s[2]) <= 1 or np.all(np.diff(s[2]) == 1) for s in out_slices
                )

                if all_contiguous:
                    out_sel = tuple(slice(s[0], s[1]) for s in out_slices)

                    if overlap == "first":
                        existing = out_var.get_chunk(out_sel, missing_none=True)
                        if existing is not None:
                            continue
                    elif overlap == "error":
                        existing = out_var.get_chunk(out_sel, missing_none=True)
                        if existing is not None:
                            msg = (
                                f"Overlap detected for variable {var_name} at {out_sel}. "
                                f'Use overlap="last" or overlap="first" to handle overlaps.'
                            )
                            raise ValueError(msg)

                    out_var.set(out_sel, data)
                else:
                    _write_noncontiguous(out_var, data, out_slices, overlap, var_name)


def _write_noncontiguous(out_var, src_data, out_slices, overlap, var_name):
    """
    Handle non-contiguous index mapping by writing element by element.
    """
    ranges = [range(len(s[2])) for s in out_slices]
    for idx in itertools.product(*ranges):
        out_idx = tuple(int(out_slices[dim_i][2][idx[dim_i]]) for dim_i in range(len(out_slices)))

        if overlap == "first":
            existing = out_var.get_chunk(out_idx, missing_none=True)
            if existing is not None:
                continue
        elif overlap == "error":
            existing = out_var.get_chunk(out_idx, missing_none=True)
            if existing is not None:
                msg = f"Overlap detected for variable {var_name} at {out_idx}."
                raise ValueError(msg)

        out_var.set(out_idx, src_data[idx])


def combine(
    datasets: list,
    output_path,
    sel: dict | None = None,
    overlap: str = "last",
    compression: str | None = None,
    compression_level: int | None = None,
    include_data_vars: list | None = None,
    exclude_data_vars: list | None = None,
):
    """
    Combine multiple cfdb datasets into a single output file.

    Parameters
    ----------
    datasets : list
        List of file paths (str/Path) or open Dataset objects.
    output_path : str or Path
        Path for the new combined cfdb file.
    sel : dict or None
        Location-based selection applied to each input dataset before combining.
        Works like Dataset.select_loc(). Keys are coordinate names, values are
        slices or values for location-based indexing.
    overlap : str
        How to handle data variables when there are overlapping coordinate values:
        - 'last': last dataset wins (default, most performant)
        - 'first': first dataset wins (skip if data already written)
        - 'error': raise ValueError on overlap
    compression : str or None
        Compression algorithm ('zstd' or 'lz4'). Inherited from first dataset if None.
    compression_level : int or None
        Compression level. Inherited from first dataset if None.
    include_data_vars : list or None
        Only include these data variables.
    exclude_data_vars : list or None
        Exclude these data variables.

    Returns
    -------
    Dataset
        The output dataset (open for reading and writing).
    """
    if overlap not in ("last", "first", "error"):
        msg = f"overlap must be 'last', 'first', or 'error', got '{overlap}'"
        raise ValueError(msg)

    if len(datasets) == 0:
        msg = "datasets must not be empty."
        raise ValueError(msg)

    ## Phase 1: Open & Validate
    ds_list, opened = _open_inputs(datasets)

    try:
        dataset_type = _validate_dataset_types(ds_list)
        crs_str = _validate_crs(ds_list)

        ## Phase 1b: Apply location selection
        if sel is not None:
            views = []
            for ds in ds_list:
                views.append(ds.select_loc(sel))
            ds_list = views

        ## Determine compression
        if compression is None:
            compression = ds_list[0].compression
            compression_level = ds_list[0].compression_level

        ## Phase 2: Compute combined coordinates
        coord_infos = _compute_combined_coords(ds_list)

        ## Phase 3: Validate data variables
        var_infos = _collect_data_vars(ds_list, include_data_vars, exclude_data_vars)

        ## Phase 4: Create output & write data
        output_ds = open_dataset(
            output_path, "n",
            dataset_type=dataset_type.value,
            compression=compression,
            compression_level=compression_level,
        )

        # Create coordinates
        for ci in coord_infos:
            axis_str = ci["axis"].value if ci["axis"] is not None else None
            output_ds.create.coord.generic(
                ci["name"],
                data=ci["data"],
                dtype=dtypes_mod.dtype(**ci["dtype_dict"]),
                chunk_shape=ci["chunk_shape"],
                step=ci["step"] if ci["step"] is not None else False,
                axis=axis_str,
            )

        # Create data variables
        for vi in var_infos:
            output_ds.create.data_var.generic(
                vi["name"],
                vi["coords"],
                dtype=dtypes_mod.dtype(**vi["dtype_dict"]),
                chunk_shape=vi["chunk_shape"],
            )

        # Write data
        _write_data(ds_list, output_ds, var_infos, coord_infos, overlap)

        ## Phase 5: Merge metadata
        # CRS
        if crs_str is not None:
            output_ds._sys_meta.crs = crs_str
            output_ds.crs = pyproj.CRS.from_user_input(crs_str)

            # Set axis metadata on output coords to match source
            for ci in coord_infos:
                if ci["axis"] is not None:
                    output_ds._sys_meta.variables[ci["name"]].axis = ci["axis"]

        # Global attributes: first-wins merge
        merged_attrs = {}
        for ds in reversed(ds_list):
            merged_attrs.update(ds.attrs.data)
        output_ds.attrs.update(merged_attrs)

        # Coordinate/variable attributes: from first dataset defining each
        for ci in coord_infos:
            for ds in ds_list:
                if ci["name"] in ds.coord_names:
                    output_ds[ci["name"]].attrs.update(ds[ci["name"]].attrs.data)
                    break

        for vi in var_infos:
            for ds in ds_list:
                if vi["name"] in ds.data_var_names:
                    output_ds[vi["name"]].attrs.update(ds[vi["name"]].attrs.data)
                    break

        return output_ds

    finally:
        for ds in opened:
            ds.close()
