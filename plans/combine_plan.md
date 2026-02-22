# Plan: `combine` Function

## Context

cfdb currently has no way to merge multiple dataset files together. There is a legacy `combine.py` (h5py/xarray-based) in `cfdb/legacy/`, but nothing for the current Booklet-backed architecture. Users need the ability to combine datasets that share the same type (e.g., grid or ts_ortho) — for instance merging datasets covering different time periods or spatial regions into one file.

## Approach

Create a new `cfdb/combine.py` module with a `combine()` function, export it from `__init__.py`, and add tests.

## Function Signature

```python
def combine(
    datasets: list,           # List of file paths (str/Path) or open Dataset objects
    output_path,              # Path for the new combined cfdb file
    sel: dict = None,         # Location-based selection applied to each input dataset
    overlap: str = 'last',    # 'last' (default), 'first', or 'error'
    compression: str = None,  # Inherit from first dataset if None
    compression_level: int = None,
    include_data_vars: list = None,
    exclude_data_vars: list = None,
) -> Dataset:
```

The `sel` parameter works like `Dataset.select_loc()` — a dict mapping coordinate names to value-based slices or arrays. For example:

```python
combine(
    datasets=[ds1, ds2, ds3],
    output_path='combined.cfdb',
    sel={'time': slice('2020-01-01', '2020-12-31'), 'latitude': slice(-45, 45)},
)
```

This filters each input dataset before combining, so only the selected region ends up in the output.

## Algorithm (5 phases)

### Phase 1: Open & Validate
- Accept file paths or open `Dataset` objects; open any paths with `open_dataset(path, 'r')`
- Validate all datasets have the same `dataset_type` (from `_sys_meta.dataset_type`)
- Track which datasets we opened (to close them at the end)

### Phase 1b: Apply Location Selection
- If `sel` is provided, apply `select_loc(sel)` to each input dataset (using existing `Dataset.select_loc()` from `main.py:188-213`), producing `DatasetView` objects that only expose the selected region
- The rest of the algorithm operates on these views transparently — coordinate `.data` and `iter_chunks()` already respect the selection

### Phase 2: Compute Combined Coordinates
- Collect all unique coordinate names across inputs
- For each coordinate name, validate dtype compatibility (compare `dtype.to_dict()`)
- Validate axis compatibility (must agree across datasets)
- Compute sorted union of coordinate values using `np.union1d` (for numeric/datetime/string) or WKT-based set union (for geometry types)
- Determine `step`: keep if all agree and union preserves regularity, else `None`
- Determine `chunk_shape`: use from first dataset that defines the coordinate

### Phase 3: Validate Data Variables
- Collect all unique data variable names (filtered by `include_data_vars`/`exclude_data_vars` using `utils.filter_var_names` pattern)
- For each data var, validate: same `coords` tuple (dimension names + order) and same `dtype.to_dict()` across all datasets that contain it
- Use `chunk_shape` from the first dataset that defines the variable

### Phase 4: Create Output & Write Data
1. Create output dataset via `open_dataset(output_path, 'n', ...)`
2. Create coordinates using `create.coord.generic()` with the combined data
3. Create empty data variables using `create.data_var.generic()`
4. For each data variable, for each input dataset:
   - Precompute index mapping: `np.searchsorted(output_coord_data, source_coord_data)` for each dimension
   - Iterate source chunks via the low-level Booklet key iteration (following `Dataset.copy()` pattern from `main.py:290-308`)
   - Map each source chunk's position to the output coordinate space
   - If indices are contiguous, write as a slice; otherwise use element-wise assignment
   - Handle overlap per the `overlap` parameter

### Phase 5: Merge Metadata & Cleanup
- CRS: all must agree or error
- Global attributes: first-wins merge
- Coordinate/variable attributes: from first dataset defining each
- Close any datasets we opened; return the output dataset

## Overlap Handling
- `'last'`: Write all datasets in order; last write wins (most performant — no reads needed)
- `'first'`: Check if output chunk already has data before writing; skip if so
- `'error'`: Check for existing data; raise `ValueError` if overlap detected

## Files to Create/Modify

| File | Action |
|------|--------|
| `cfdb/combine.py` | **New** — all combine logic |
| `cfdb/__init__.py` | Add `from cfdb.combine import combine` |
| `cfdb/tests/test_combine.py` | **New** — test cases |

### Key existing code to reuse
- `Dataset.copy()` pattern (`main.py:270-315`) — chunk-level data transfer between Booklet files
- `utils.filter_var_names()` (`utils.py:123-144`) — include/exclude data var filtering
- `utils.make_var_chunk_key()` (`utils.py:653-660`) — chunk key construction
- `create.coord.generic()` / `create.data_var.generic()` (`creation.py`) — variable creation
- `indexers.slices_to_chunks_keys()` (`indexers.py:285-299`) — chunk iteration

## Test Cases
- Non-overlapping time ranges (simple concat)
- Non-overlapping spatial ranges
- Overlapping coords with all three overlap modes
- Incompatible dtypes → error
- Incompatible dimensions → error
- Different dataset_types → error
- Single dataset (identity)
- Three or more datasets
- CRS validation
- Location selection (`sel`) to subset during combine

## Verification
```bash
uv run pytest cfdb/tests/test_combine.py -v
```
