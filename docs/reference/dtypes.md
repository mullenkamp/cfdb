# Data Types

## dtype() Factory

::: cfdb.dtypes.dtype
    options:
      show_root_heading: true
      show_source: false
      show_root_full_path: false

## Type Classes

All types inherit from `DataType` and provide `dumps()`/`loads()` for serialization.

### Float

Floating-point data with optional integer encoding:

```python
cfdb.dtypes.dtype('float32')                                    # no encoding
cfdb.dtypes.dtype('float64', precision=2)                       # rounds to 2 decimals
cfdb.dtypes.dtype('float64', precision=2, min_value=-50, max_value=100)  # integer encoding
```

### Integer

Integer data with optional smaller encoding:

```python
cfdb.dtypes.dtype('int32')
cfdb.dtypes.dtype('int64', min_value=0, max_value=1000)
```

### DateTime

Numpy datetime64 data:

```python
cfdb.dtypes.dtype('datetime64[D]')
cfdb.dtypes.dtype('datetime64[h]')
```

### Bool

Boolean data:

```python
cfdb.dtypes.dtype('bool')
```

### String

Variable-length strings (msgpack serialized):

```python
cfdb.dtypes.dtype('str')
```

### Point / LineString / Polygon

Geometry types using shapely and WKT:

```python
cfdb.dtypes.dtype('point', precision=6)
cfdb.dtypes.dtype('linestring', precision=4)
cfdb.dtypes.dtype('polygon', precision=4)
```

## DataType Base Class

All types share these attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | str | Type name |
| `kind` | str | Kind code (f=float, i=int, M=datetime, T=string, G=geometry, u=unsigned, b=bool) |
| `itemsize` | int or None | Bytes per element (None for variable-length) |
| `dtype_decoded` | np.dtype | Decoded (in-memory) numpy dtype |
| `dtype_encoded` | np.dtype or None | Encoded (on-disk) numpy dtype |
| `precision` | int or None | Decimal precision or WKT rounding |
| `fillvalue` | int or None | Fill value for encoded data |
| `offset` | number or None | Offset for encoding |

## Helper Function

### compute_scale_and_offset

::: cfdb.compute_scale_and_offset
    options:
      show_root_heading: true
      show_source: false
      show_root_full_path: false
