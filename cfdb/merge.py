import itertools
import pathlib
import numpy as np

from cfdb import dtypes as dtypes_mod
from cfdb.main import Dataset, DatasetView, open_dataset
from cfdb.combine import _open_inputs, _validate_dataset_types, _validate_crs, _collect_data_vars, _write_data

def merge_into(
    datasets: list,
    target_path: str | pathlib.Path,
    allow_expansion: bool | list = True,
    overlap: str = "last",
):
    """
    Destructively merge multiple cfdb datasets into an existing target cfdb file.

    Unlike `combine`, `merge_into` modifies the target dataset in-place.
    This provides O(B) performance for appending/prepending time steps compared to
    O(A+B) for `combine`. However, it strictly enforces that incoming coordinates
    are either exactly matching, strictly appending, or strictly prepending.
    In-place insertions into the middle of a dataset are not supported and will raise an error.

    Parameters
    ----------
    datasets : list
        List of file paths (str/Path) or open Dataset objects to merge.
    target_path : str or Path
        Path to the existing target cfdb file. This file will be modified in-place.
    allow_expansion : bool or list
        If False, no coordinates can expand (e.g. extending spatial bounds).
        If True, any coordinate can expand if new values are prepended or appended.
        If a list of strings (e.g. `['time']`), only the specified coordinates can expand.
    overlap : str
        How to handle data variables when there are overlapping coordinate values:
        - 'last': last dataset wins (default, overwrites existing target data)
        - 'first': first dataset wins (keeps existing target data, ignores new data)
        - 'error': raise ValueError on overlap

    Returns
    -------
    Dataset
        The modified target dataset (open for reading and writing).
    """
    if overlap not in ("last", "first", "error"):
        msg = f"overlap must be 'last', 'first', or 'error', got '{overlap}'"
        raise ValueError(msg)

    if len(datasets) == 0:
        msg = "datasets must not be empty."
        raise ValueError(msg)

    target_path = pathlib.Path(target_path)
    if not target_path.exists():
        raise FileNotFoundError(f"target_path does not exist: {target_path}")

    # Phase 1: Open Target and Inputs
    target_ds = open_dataset(target_path, "w")
    try:
        ds_list, opened = _open_inputs(datasets)
        
        all_ds = [target_ds] + ds_list
        _validate_dataset_types(all_ds)
        _validate_crs(all_ds)

        # Phase 2 & 3: Generic Coordinate Constraints & Extend Target Coordinates
        incoming_coords = set()
        for ds in ds_list:
            incoming_coords.update(ds.coord_names)
            
        for coord_name in incoming_coords:
            if coord_name not in target_ds.coord_names:
                src_coord = next(ds[coord_name] for ds in ds_list if coord_name in ds.coord_names)
                if allow_expansion is False or (isinstance(allow_expansion, list) and coord_name not in allow_expansion):
                    raise ValueError(f"Expansion for coordinate '{coord_name}' is disallowed.")
                target_ds.create.coord.like(coord_name, src_coord, copy_data=True)
                continue

            tgt_coord = target_ds[coord_name]
            tgt_data = tgt_coord.data
            if len(tgt_data) == 0:
                continue
                
            dtype_obj = tgt_coord.dtype
            kind = dtype_obj.kind
            
            all_incoming_data = []
            for ds in ds_list:
                if coord_name in ds.coord_names:
                    all_incoming_data.append(ds[coord_name].data)
                    
            if not all_incoming_data:
                continue
                
            if kind == "G":
                tgt_encoded = set(dtype_obj.encode(tgt_data))
                new_geos = {}
                for d in all_incoming_data:
                    for wkt, geo in zip(dtype_obj.encode(d), d):
                        if wkt not in tgt_encoded and wkt not in new_geos:
                            new_geos[wkt] = geo
                            
                if new_geos:
                    if allow_expansion is False or (isinstance(allow_expansion, list) and coord_name not in allow_expansion):
                        raise ValueError(f"Expansion for coordinate '{coord_name}' is disallowed.")
                    append_vals = np.array(list(new_geos.values()), dtype=dtype_obj.dtype_decoded)
                    target_ds[coord_name].append(append_vals)
            else:
                combined_incoming = all_incoming_data[0]
                for d in all_incoming_data[1:]:
                    combined_incoming = np.union1d(combined_incoming, d)
                    
                target_min, target_max = tgt_data[0], tgt_data[-1]
                
                prepend_mask = combined_incoming < target_min
                append_mask = combined_incoming > target_max
                middle_mask = (combined_incoming >= target_min) & (combined_incoming <= target_max)
                
                prepend_vals = combined_incoming[prepend_mask]
                append_vals = combined_incoming[append_mask]
                middle_vals = combined_incoming[middle_mask]
                
                if kind == "f":
                    # For floats, np.isin can be brittle, use isclose instead
                    insert_mask = np.array([not np.any(np.isclose(v, tgt_data)) for v in middle_vals])
                else:
                    insert_mask = np.isin(middle_vals, tgt_data, invert=True)
                    
                insert_vals = middle_vals[insert_mask]
                
                if len(insert_vals) > 0:
                    raise NotImplementedError(f"In-place coordinate insertions are unsupported. Values: {insert_vals} for coordinate '{coord_name}'")
                    
                if len(prepend_vals) > 0 or len(append_vals) > 0:
                    if allow_expansion is False or (isinstance(allow_expansion, list) and coord_name not in allow_expansion):
                        raise ValueError(f"Expansion for coordinate '{coord_name}' is disallowed.")
                        
                if len(prepend_vals) > 0:
                    target_ds[coord_name].prepend(prepend_vals)
                if len(append_vals) > 0:
                    target_ds[coord_name].append(append_vals)
                    
        # Phase 4: Ingest Data Variables
        var_infos = _collect_data_vars(ds_list, None, None)
        for vi in var_infos:
            var_name = vi["name"]
            if var_name not in target_ds.data_var_names:
                target_ds.create.data_var.generic(
                    var_name,
                    vi["coords"],
                    dtype=dtypes_mod.dtype(**vi["dtype_dict"]),
                    chunk_shape=vi["chunk_shape"],
                )
                for ds in ds_list:
                    if var_name in ds.data_var_names:
                        target_ds[var_name].attrs.update(ds[var_name].attrs.data)
                        break
        
        coord_infos = []
        for coord_name in target_ds.coord_names:
            coord = target_ds[coord_name]
            coord_infos.append({
                "name": coord_name,
                "data": coord.data,
                "dtype_dict": coord.dtype.to_dict(),
                "axis": coord._var_meta.axis,
                "chunk_shape": coord.chunk_shape,
                "step": coord._var_meta.step,
            })
            
        _write_data(ds_list, target_ds, var_infos, coord_infos, overlap)

        # Update global attributes
        merged_attrs = {}
        for ds in reversed(ds_list):
            merged_attrs.update(ds.attrs.data)
        target_ds.attrs.update(merged_attrs)

        return target_ds
    except Exception:
        # If an error occurred, closing the target dataset is still important
        target_ds.close()
        raise
    finally:
        for ds in opened:
            ds.close()
