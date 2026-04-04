# Combining and Merging Datasets

cfdb provides two distinct functions for combining multiple datasets together:
- `combine`: An **out-of-place** merge that creates a brand new dataset from the union of all inputs.
- `merge_into`: An **in-place** merge that destructively modifies an existing target dataset.

## `combine` (Out-of-Place)

The `combine` function takes multiple input datasets, computes the union of their coordinates, and writes all data into a new output file. This is useful for joining datasets that cover different spatial regions or creating a clean dataset from many smaller files.

```python
import cfdb

# Creates a new 'combined.cfdb' file
result = cfdb.combine(
    ['region_north.cfdb', 'region_south.cfdb'],
    'combined.cfdb',
)
print(result)
result.close()
```

The function accepts file paths or open `Dataset` objects.

### Subsetting with `sel`

You can apply a location-based selection to each input dataset before combining. This filters the data so only the selected region ends up in the output:

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

### Compression and Variables

By default, compression settings are inherited from the first input dataset, but they can be overridden. You can also explicitly filter which data variables to include:

```python
result = cfdb.combine(
    datasets, 'out.cfdb',
    compression='zstd',
    compression_level=1,
    include_data_vars=['temperature'],
    # exclude_data_vars=['humidity'],
)
```

---

## `merge_into` (In-Place)

For large, continuously updated databases (e.g., adding yesterday's weather data to a 100GB climate cache), recreating the entire file with `combine` is prohibitively slow and requires double the disk space.

The `merge_into` function solves this by destructively writing new data directly into an existing dataset file. Because of the way `cfdb` stores coordinate metadata, **appending or prepending data along a coordinate (like `time`) is extremely fast (O(new_data))**, completely avoiding the need to rewrite the existing chunks.

```python
import cfdb

# Destructively modifies 'existing_target.cfdb' in-place
result = cfdb.merge_into(
    ['new_weather_data.cfdb'],
    'existing_target.cfdb',
    allow_expansion=['time']
)
result.close()
```

### Expansion Constraints

To protect against accidental database corruption and massive performance penalties, `merge_into` enforces strict rules about coordinate expansion:

1. **Insertions are blocked**: You cannot insert new coordinate values *into the middle* of an existing dataset's coordinate range. You can only strictly append (after the max value) or prepend (before the min value).
2. **Expansion guardrails**: The `allow_expansion` parameter controls which coordinates are allowed to grow. 
    - `allow_expansion=True`: Any coordinate can grow via append/prepend.
    - `allow_expansion=['time']`: **Recommended.** Only the `time` coordinate is allowed to grow. If the incoming dataset has a slightly different spatial bounding box (e.g., an extra latitude row), `merge_into` will raise an error and abort instead of silently attempting an expensive in-place spatial expansion.
    - `allow_expansion=False`: No coordinates can grow; incoming data must perfectly match or be a subset of the target's existing coordinates.

---

## Handling Overlaps

Both `combine` and `merge_into` support an `overlap` parameter that controls what happens when multiple datasets contain data for the exact same coordinate values:

```python
# Last dataset wins (default, most performant — overwrites existing data)
cfdb.combine(datasets, 'out.cfdb', overlap='last')
cfdb.merge_into(datasets, 'target.cfdb', overlap='last')

# First dataset wins (skips writing if data already exists in the target)
cfdb.combine(datasets, 'out.cfdb', overlap='first')

# Raise an error if any overlap is detected
cfdb.combine(datasets, 'out.cfdb', overlap='error')
```

!!! note
    Overlap handling applies to **data variables**, not coordinates. Coordinate values are always merged as a sorted union regardless of the overlap setting.

## Requirements

For both functions, all input datasets must:
- Have the **same `dataset_type`** (e.g., all `grid` or all `ts_ortho`)
- Have **compatible coordinate dtypes** for any shared coordinate names
- Have **compatible data variable dtypes and dimensions** for any shared variable names
- Have **matching CRS** if any datasets define one
