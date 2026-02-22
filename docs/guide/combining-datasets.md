# Combining Datasets

cfdb can merge multiple datasets into a single output file using the `combine` function. This is useful for joining datasets that cover different time periods, spatial regions, or that were split across files.

## Basic Usage

```python
import cfdb

result = cfdb.combine(
    ['region_north.cfdb', 'region_south.cfdb'],
    'combined.cfdb',
)
print(result)
result.close()
```

The function accepts file paths or open `Dataset` objects, computes the union of all coordinates, validates compatibility, and writes all data into a new file.

## Combining Time Periods

Merge datasets covering different date ranges:

```python
result = cfdb.combine(
    ['data_2020.cfdb', 'data_2021.cfdb', 'data_2022.cfdb'],
    'data_2020_2022.cfdb',
)
result.close()
```

Coordinates are merged as a sorted union — non-overlapping ranges are concatenated, overlapping values are deduplicated.

## Combining Spatial Regions

```python
result = cfdb.combine(
    ['tile_a.cfdb', 'tile_b.cfdb'],
    'merged_tiles.cfdb',
)
result.close()
```

This works for any coordinate type, including latitude/longitude and geometry coordinates.

## Handling Overlaps

When datasets share coordinate values, the `overlap` parameter controls what happens to the data variables in the overlapping region:

```python
# Last dataset wins (default, most performant — no reads needed)
cfdb.combine(datasets, 'out.cfdb', overlap='last')

# First dataset wins (skips writing if data already exists)
cfdb.combine(datasets, 'out.cfdb', overlap='first')

# Raise an error if any overlap is detected
cfdb.combine(datasets, 'out.cfdb', overlap='error')
```

!!! note
    Overlap handling applies to **data variables**, not coordinates. Coordinate values are always merged as a sorted union regardless of the overlap setting.

## Subsetting with `sel`

Apply a location-based selection to each input dataset before combining. This filters the data so only the selected region ends up in the output:

```python
result = cfdb.combine(
    ['full_globe_a.cfdb', 'full_globe_b.cfdb'],
    'europe_combined.cfdb',
    sel={
        'latitude': slice(35.0, 72.0),
        'longitude': slice(-25.0, 45.0),
    },
)
result.close()
```

The `sel` parameter works like `Dataset.select_loc()` — keys are coordinate names, values are slices or arrays for location-based indexing.

## Filtering Data Variables

Include or exclude specific data variables:

```python
# Only include temperature
result = cfdb.combine(
    datasets, 'out.cfdb',
    include_data_vars=['temperature'],
)

# Include everything except humidity
result = cfdb.combine(
    datasets, 'out.cfdb',
    exclude_data_vars=['humidity'],
)
```

## Compression

By default, compression settings are inherited from the first input dataset. Override them explicitly:

```python
result = cfdb.combine(
    datasets, 'out.cfdb',
    compression='zstd',
    compression_level=1,
)
```

## Using Open Datasets

You can pass already-open `Dataset` objects instead of file paths:

```python
with cfdb.open_dataset('a.cfdb') as ds_a:
    with cfdb.open_dataset('b.cfdb') as ds_b:
        result = cfdb.combine([ds_a, ds_b], 'combined.cfdb')
        result.close()
```

Datasets passed as file paths are opened and closed automatically.

## Requirements

All input datasets must:

- Have the **same `dataset_type`** (e.g., all `grid` or all `ts_ortho`)
- Have **compatible coordinate dtypes** for any shared coordinate names
- Have **compatible data variable dtypes and dimensions** for any shared variable names
- Have **matching CRS** if any datasets define one
