# Indexing & Selection

cfdb supports basic numpy-style indexing and location-based selection.

## Index-Based Selection

Use integers and slices to select by position, just like numpy basic indexing:

```python
with cfdb.open_dataset(file_path) as ds:
    temp = ds['temperature']

    # Single element
    val = temp[0, 0]

    # Slice
    subset = temp[10:20, :]

    # Mixed
    row = temp[5, 0:100]
```

Bounds behavior follows numpy semantics:

```python
with cfdb.open_dataset(file_path) as ds:
    temp = ds['temperature']       # shape (40, 100)

    temp[-1]                       # negative ints wrap: the last row
    temp[35:60, :]                 # slice bounds clamp: rows 35-39 (5 rows)
    temp[40]                       # IndexError: out of bounds
    temp[45:60, :]                 # ValueError: empty selection
```

Out-of-range integer indexes raise `IndexError`; slices clamp to the variable extent
exactly like numpy arrays. Selections that would be empty (fully out-of-range or
descending slices) raise a `ValueError` — cfdb has no zero-size selections. A `.loc`
scalar label beyond the coordinate's last value also raises `IndexError`.

!!! note
    Advanced indexing (fancy indexing with arrays or boolean masks) is currently not supported.

## Location-Based Selection (.loc)

Use `.loc[]` to select by coordinate values instead of positions:

```python
with cfdb.open_dataset(file_path) as ds:
    temp = ds['temperature']

    # Select by coordinate value
    val = temp.loc[45.0, '2020-06-15']

    # Slice by coordinate values
    subset = temp.loc[40.0:50.0, '2020-01-01':'2020-06-30']
```

`.loc` works with any coordinate data type, including datetime:

```python
with cfdb.open_dataset(file_path) as ds:
    coord = ds['latitude']
    lat_subset = coord.loc[40.0:50.0]
```

## Dataset-Level Selection

### select()

Filter the entire dataset by coordinate index positions, returning a `DatasetView`:

```python
with cfdb.open_dataset(file_path) as ds:
    view = ds.select({'latitude': slice(10, 20), 'time': slice(0, 30)})
    print(view)

    # Access variables through the view
    for chunk_slices, data in view['temperature'].iter_chunks():
        print(data.shape)
```

### select_loc()

Filter by coordinate values:

```python
with cfdb.open_dataset(file_path) as ds:
    view = ds.select_loc({
        'latitude': slice(40.0, 50.0),
        'time': slice('2020-01-01', '2020-03-01'),
    })
```

## DatasetView

Both `select()` and `select_loc()` return a `DatasetView` — a read-only view over a subset of the dataset. It supports the same interface as `Dataset` for reading:

```python
with cfdb.open_dataset(file_path) as ds:
    view = ds.select({'latitude': slice(0, 10)})

    # Iterate variables
    for name in view:
        print(name)

    # Access coordinates and data
    print(view.coord_names)
    print(view.data_var_names)
    temp = view['temperature']

    # CRS is accessible on views
    print(view.crs)
```

### Chaining selections

Selections can be chained — call `select()` or `select_loc()` on a `DatasetView` to narrow it further:

```python
with cfdb.open_dataset(file_path) as ds:
    # Position-based chaining: second selection is relative to the first
    view = ds.select({'latitude': slice(20, 60)})
    narrower = view.select({'latitude': slice(5, 15)})  # positions 25-35 in the full dataset

    # Location-based chaining: narrows by coordinate values within the view
    view = ds.select({'latitude': slice(0, 50)})
    narrower = view.select_loc({'latitude': slice(40.0, 45.0)})
```

!!! warning
    `DatasetView` is read-only. Writing through a view is not supported.
