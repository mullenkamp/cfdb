# Coordinate

Coordinate variables define the labeled dimensions of a dataset. They hold all data in memory and support append/prepend operations.

## Class Hierarchy

```
Variable (base)
  └── CoordinateView (sliced view, read-only)
        └── Coordinate (full coordinate, writable)
```

- `Coordinate` — the full coordinate with append/prepend, returned by `ds['coord_name']` or creation methods
- `CoordinateView` — a sliced subset, returned by indexing a coordinate

## Usage

```python
with cfdb.open_dataset(file_path) as ds:
    lat = ds['latitude']
    print(lat.data)        # full array (in memory)
    print(lat.shape)       # (200,)
    print(lat.chunk_shape) # (20,)
```

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | str | Variable name |
| `data` | np.ndarray | Full coordinate data (always in memory) |
| `values` | np.ndarray | Alias for `data` |
| `shape` | tuple | Shape of the coordinate |
| `chunk_shape` | tuple | Chunk shape |
| `dtype` | DataType | cfdb data type |
| `coord_names` | tuple of str | `(name,)` — the coordinate's own name |
| `ndims` | int | Always 1 |
| `attrs` | Attributes | Variable attributes |
| `axis` | Axis or None | Physical axis (x, y, z, t, xy) |
| `step` | number or None | Regular spacing step |
| `origin` | int | Starting position in global index space |
| `auto_increment` | bool or None | Whether auto-increment is enabled |
| `units` | str or None | Physical units |
| `writable` | bool | Whether the dataset is writable |
| `loc` | LocationIndexer | Location-based indexer |

## Methods

### get(sel)

Get a `CoordinateView` by index position. `sel` can be an int, slice, or tuple.

```python
subset = lat[10:20]      # CoordinateView
subset = lat.get(slice(10, 20))  # equivalent
```

### append(data)

Append data to the end of the coordinate. Data must maintain uniqueness and ascending order.

```python
lat.append(np.array([20.0, 20.1], dtype='float32'))
```

### prepend(data)

Prepend data to the beginning of the coordinate.

```python
lat.prepend(np.array([-0.2, -0.1], dtype='float32'))
```

### iter_chunks(include_data=False, decoded=True)

Iterate through chunks. Returns `(slices,)` or `(slices, data)` tuples.

### rechunker()

Return a `Rechunker` for this variable.

### load()

For EDataset: pre-fetch chunks from S3. No-op for local datasets.

### update_step(step)

Update the step value. Only on writable datasets.

### update_axis(axis)

Update the axis value. Only on writable datasets.

### update_units(units)

Update the units value. Only on writable datasets.

### get_chunk(sel=None, missing_none=False)

Get data from one chunk.

### items(decoded=True)

Iterate through all individual index positions yielding `(index, value)` tuples.
