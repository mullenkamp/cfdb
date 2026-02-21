# Data Types

cfdb uses a custom type system that handles both the in-memory representation (decoded) and the on-disk serialization (encoded) of data. Nearly all use numpy dtypes.

## The dtype() Factory

Create a `DataType` using `cfdb.dtypes.dtype()`:

```python
import cfdb

# Simple float — no encoding, stored as-is
dt = cfdb.dtypes.dtype('float32')

# Float with precision and integer encoding
dt = cfdb.dtypes.dtype('float64', precision=2, min_value=-50.0, max_value=100.0)

# Datetime
dt = cfdb.dtypes.dtype('datetime64[D]')

# Geometry
dt = cfdb.dtypes.dtype('point', precision=6)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str, np.dtype, or DataType | Type name or existing dtype |
| `precision` | int or None | Decimal precision (floats) or WKT rounding (geometry) |
| `min_value` | number or None | Minimum value (for integer encoding) |
| `max_value` | number or None | Maximum value (for integer encoding) |
| `dtype_encoded` | str or None | Explicit encoded dtype name |
| `offset` | number or None | Offset for encoding |
| `fillvalue` | int or None | Fill value for encoding |

## Type Classes

### Float

Stores floating-point data. Can optionally encode to a smaller integer for better compression:

```python
# No encoding — stored as float32 bytes
dt = cfdb.dtypes.dtype('float32')

# With precision only — rounds to 2 decimals but stays float
dt = cfdb.dtypes.dtype('float64', precision=2)

# With integer encoding — compresses float64 to uint2 using scale+offset
dt = cfdb.dtypes.dtype('float64', precision=2, min_value=-50.0, max_value=100.0)
```

When `min_value`, `max_value`, and `precision` are provided, cfdb computes the smallest unsigned integer type and offset needed to represent the range.

### Integer

Stores integer data. Can optionally encode to a smaller integer:

```python
dt = cfdb.dtypes.dtype('int32')
dt = cfdb.dtypes.dtype('int64', min_value=0, max_value=1000)
```

### DateTime

Stores numpy `datetime64` data:

```python
# Days precision
dt = cfdb.dtypes.dtype('datetime64[D]')

# Hourly precision
dt = cfdb.dtypes.dtype('datetime64[h]')

# Nanosecond precision (uses int64 encoding)
dt = cfdb.dtypes.dtype('datetime64[ns]')
```

See the [numpy datetime reference](https://numpy.org/doc/stable/reference/arrays.datetime.html#datetime-units) for frequency codes. Do not use frequencies finer than `'ns'`.

### Bool

Stores boolean data:

```python
dt = cfdb.dtypes.dtype('bool')
```

### String

Stores variable-length strings via msgpack serialization:

```python
dt = cfdb.dtypes.dtype('str')
```

String coordinates do not need to be in ascending order.

### Geometry Types

Geometry data uses [shapely](https://shapely.readthedocs.io/) objects, serialized to WKT strings and stored via msgpack. The `precision` parameter controls WKT rounding:

```python
# Point geometries with 6 decimal places
dt = cfdb.dtypes.dtype('point', precision=6)

# LineString
dt = cfdb.dtypes.dtype('linestring', precision=4)

# Polygon
dt = cfdb.dtypes.dtype('polygon', precision=4)
```

## Encoding and Decoding

Each dtype handles two transformations:

1. **encode/decode** — convert between the decoded numpy dtype and the encoded dtype (e.g., float64 → uint2 using scale and offset)
2. **dumps/loads** — convert between numpy arrays and raw bytes for storage

When no encoding is configured, `dumps`/`loads` use `tobytes()`/`frombuffer()` directly.

## Using with compute_scale_and_offset

The helper function `cfdb.compute_scale_and_offset` can pre-calculate encoding parameters:

```python
from cfdb import compute_scale_and_offset

# Determine scale and offset for a data range
scale, offset = compute_scale_and_offset(min_val=-50.0, max_val=100.0, precision=2)
```
