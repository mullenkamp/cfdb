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
    for chunk_slices, data in view['temperature'].iter_chunks(include_data=True):
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
```

!!! warning
    `DatasetView` is read-only. Writing through a view is not supported.
