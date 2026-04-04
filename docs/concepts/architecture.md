# Architecture

This page explains the internal design of cfdb for users who want to understand how data is stored and managed.

## System Overview

```
open_dataset() / open_edataset()
        │
        ▼
    Dataset / EDataset
        │
        ├── SysMeta (system metadata in Booklet metadata field)
        │     ├── dataset_type, compression, crs
        │     └── variables: dict of CoordinateVariable / DataVariable
        │
        ├── Creator (coord, data_var, crs sub-objects)
        │
        ├── Attributes (JSON dict stored as Booklet key)
        │
        ├── DatasetRechunker (synchronized multi-variable rechunkit wrapper)
        │
        └── Variable objects (Coordinate / DataVariable)
              ├── DataType (encoding/decoding)
              ├── Compressor (zstd/lz4)
              └── Rechunker (single-variable rechunkit wrapper)
```

## Public API

### Datasets

```mermaid
classDiagram
    direction TB

    class cfdb {
        <<module>>
        open_dataset(file_path, flag, dataset_type, compression, compression_level) Dataset
        open_edataset(remote_conn, file_path, flag, dataset_type, compression, ...) EDataset
        cfdb_to_netcdf4(cfdb_path, nc_path, compression, sel, sel_loc, ...)
        combine(cfdb_paths, output_path, ...)
        merge_into(source_path, dest_path, ...)
        guess_chunk_shape(shape, itemsize, target_size) tuple
        compute_scale_and_offset(min_value, max_value, precision) tuple
    }

    class dtypes {
        <<module>>
        dtype(name, precision, min_value, max_value, ...) DataType
    }

    class tools {
        <<module>>
        netcdf4_to_cfdb(nc_path, cfdb_path, sel, sel_loc, max_mem, ...)
        cfdb_to_netcdf4(cfdb_path, nc_path, compression, sel, sel_loc, ...)
    }

    class Dataset {
        +file_path : Path
        +compression : str
        +compression_level : int
        +writable : bool
        +is_open : bool
        +var_names : tuple
        +coord_names : tuple
        +data_var_names : tuple
        +coords : tuple~Coordinate~
        +data_vars : tuple~DataVariable~
        +attrs : Attributes
        +crs : pyproj.CRS
        +create : Creator
        +get(var_name) Coordinate | DataVariable
        +select(sel) DatasetView
        +select_loc(sel) DatasetView
        +iter_chunks(chunk_shape, data_vars, max_mem, include_data) Generator
        +groupby(coord_names, data_vars, max_mem) Generator
        +map(func, chunk_shape, data_vars, max_mem, n_workers) Generator
        +rechunker(data_vars) DatasetRechunker
        +copy(file_path, include_data_vars, exclude_data_vars) Dataset
        +to_netcdf4(file_path, compression, include_data_vars, exclude_data_vars)
        +prune(timestamp) int
        +close()
    }

    class EDataset {
        +changes() Change
        +delete_remote()
        +copy_remote(remote_conn)
    }

    class DatasetView {
        +var_names : tuple
        +coord_names : tuple
        +data_var_names : tuple
        +coords : tuple
        +data_vars : tuple
        +attrs : Attributes
        +crs : pyproj.CRS
        +get(var_name) CoordinateView | DataVariableView
        +select(sel) DatasetView
        +select_loc(sel) DatasetView
    }

    class Creator {
        +coord : Coord
        +data_var : DataVar
        +crs : CRS
    }

    class Coord {
        +generic(name, data, dtype, chunk_shape, step, axis) Coordinate
        +like(name, coord, copy_data) Coordinate
        +lat(data, step, ...) Coordinate
        +lon(data, step, ...) Coordinate
        +time(data, step, ...) Coordinate
        +height(...) Coordinate
        +altitude(...) Coordinate
        +depth(...) Coordinate
        +point(...) Coordinate
        +x(data, step, ...) Coordinate
        +y(data, step, ...) Coordinate
        +z(data, step, ...) Coordinate
    }

    class DataVar {
        +generic(name, coords, dtype, chunk_shape) DataVariable
        +like(name, data_var) DataVariable
        +air_temperature(coords) DataVariable
        +precipitation(coords) DataVariable
    }

    class CRS {
        +from_user_input(crs, x_coord, y_coord, xy_coord) pyproj.CRS
    }

    cfdb --> Dataset : open_dataset()
    cfdb --> EDataset : open_edataset()
    EDataset --|> Dataset
    Dataset --> DatasetView : select() / select_loc()
    Dataset *-- Creator : create
    Creator *-- Coord : coord
    Creator *-- DataVar : data_var
    Creator *-- CRS : crs
```

### Variables

```mermaid
classDiagram
    direction TB

    class Variable {
        <<abstract>>
        +name : str
        +shape : tuple
        +chunk_shape : tuple
        +dtype : DataType
        +ndims : int
        +coord_names : tuple
        +attrs : Attributes
        +writable : bool
        +is_open : bool
        +units : str
        +loc : LocationIndexer
        +data : ndarray
        +values : ndarray
    }

    class CoordinateView {
        +iter_chunks(include_data, decoded) Generator
        +items(decoded) Generator
        +get_chunk(sel, missing_none) ndarray
    }

    class Coordinate {
        +step : number
        +origin : int
        +axis : str
        +auto_increment : bool
        +append(data)
        +prepend(data)
        +truncate(start, stop)
        +update_step(step)
        +update_axis(axis)
        +get_coord_origins() tuple
        +load()
    }

    class DataVariableView {
        +coords : tuple~Coordinate~
        +set(sel, data, decoded)
        +iter_chunks(chunk_shape, max_mem, decoded, include_data) Generator
        +items(decoded) Generator
        +get_chunk(sel, missing_none) ndarray
        +groupby(coord_names, max_mem) Generator
        +map(func, chunk_shape, max_mem, n_workers) Generator
        +interp(x, y, z, xy) GridInterp | PointInterp
    }

    class DataVariable {
        +rechunker() Rechunker
        +load()
    }

    class Attributes {
        +data : dict
        +writable : bool
        +get(key) value
        +set(key, value)
        +keys() Iterator
        +values() Iterator
        +items() Iterator
        +pop(key, default) value
        +update(other)
        +clear()
    }

    class Rechunker {
        +guess_chunk_shape(target_size) tuple
        +calc_ideal_read_chunk_shape(target_shape) tuple
        +calc_source_read_chunk_shape(target_shape, max_mem) tuple
        +calc_n_chunks() int
        +calc_n_reads_rechunker(target_shape, max_mem) tuple
        +rechunk(target_shape, max_mem) Generator
    }

    class DatasetRechunker {
        +calc_ideal_read_chunk_mem(chunk_shape) int
        +calc_n_reads_rechunker(chunk_shape, max_mem) tuple
        +rechunk(chunk_shape, max_mem) Generator
    }

    class LocationIndexer {
        +__getitem__(sel) View
    }

    Variable <|-- CoordinateView
    CoordinateView <|-- Coordinate
    Variable <|-- DataVariableView
    DataVariableView <|-- DataVariable
    Variable *-- LocationIndexer : loc
    Variable *-- Attributes : attrs
    DataVariable --> Rechunker : rechunker()
```

### Data Types

```mermaid
classDiagram
    direction TB

    class DataType {
        <<abstract>>
        +name : str
        +kind : str
        +itemsize : int
        +dtype_decoded : numpy.dtype
        +dtype_encoded : numpy.dtype
        +precision : int
        +fillvalue : int
        +offset : number
        +to_dict() dict
    }

    class Float {
        +encode(data) ndarray
        +decode(data) ndarray
        +dumps(data) bytes
        +loads(data_bytes, chunk_shape) ndarray
    }
    class Integer {
        +encode(data) ndarray
        +decode(data) ndarray
        +dumps(data) bytes
        +loads(data_bytes, chunk_shape) ndarray
    }
    class DateTime {
        +encode(data) ndarray
        +decode(data) ndarray
        +dumps(data) bytes
        +loads(data_bytes, chunk_shape) ndarray
    }
    class Bool {
        +dumps(data) bytes
        +loads(data_bytes, chunk_shape) ndarray
    }
    class String {
        +dumps(data) bytes
        +loads(data_bytes) ndarray
    }
    class Geometry {
        <<abstract>>
        +encode(data) list
        +decode(data) ndarray
        +dumps(data) bytes
        +loads(data_bytes) ndarray
    }
    class Point
    class LineString
    class Polygon

    DataType <|-- Float
    DataType <|-- Integer
    DataType <|-- DateTime
    DataType <|-- Bool
    DataType <|-- String
    DataType <|-- Geometry
    Geometry <|-- Point
    Geometry <|-- LineString
    Geometry <|-- Polygon
```

## Booklet Storage

cfdb uses [Booklet](https://github.com/mullenkamp/booklet) as a key-value store. Booklet is a persistent dict-like database stored in a single file, with support for thread locks and file locks.

All data in a cfdb file lives in one Booklet file:

- **System metadata** — stored in Booklet's metadata field (a single JSON blob)
- **Data chunks** — stored as Booklet key-value pairs
- **Attributes** — stored as separate Booklet keys (`_{var_name}.attrs`)

## Metadata Lifecycle

1. On open, `SysMeta` is deserialized from the Booklet metadata via `msgspec.convert()`
2. During the session, `SysMeta` is modified in memory (adding variables, changing shapes, etc.)
3. On close, a `weakref.finalize` callback serializes `SysMeta` back to Booklet metadata

This means metadata changes are batched and written on close, not on every operation.

## Chunk Storage

Data chunks are stored with keys formatted as:

```
{var_name}!{dim0_start},{dim1_start},...
```

For example, a 2-D variable `temperature` with chunk starting at position (100, 200) would have the key `temperature!100,200`.

This key format is generated by `utils.make_var_chunk_key()`.

## Variable Hierarchy

```
Variable (base)
├── CoordinateView → Coordinate
│     - Holds all data in memory
│     - Supports append/prepend
│     - .data returns full array
│
└── DataVariableView → DataVariable
      - Never holds full data in memory
      - Supports __setitem__ for writing
      - .data reads all chunks (expensive)
```

The `View` variants represent subsets created by indexing or `select()`.

## Thread and Multiprocess Safety

- **Thread safety**: Booklet uses thread locks for concurrent read/write access
- **Multiprocess safety**: File locks prevent corruption from multiple processes
- **S3 safety** (EDataset): Object locking on the remote ensures consistency

## Error Handling

When an error occurs, cfdb attempts to:

1. Close the Booklet file properly
2. Remove file/object locks

Changes that were not synced are lost. The `weakref.finalize` mechanism ensures cleanup runs even on unexpected exits.

## Dependencies

| Package | Role |
|---------|------|
| booklet | Local key-value file storage |
| ebooklet | S3 remote sync (optional) |
| numpy | Array operations |
| msgspec | Fast serialization (metadata, strings, geometry) |
| zstandard | Zstd compression |
| lz4 | LZ4 compression |
| rechunkit | Rechunking algorithms |
| shapely | Geometry types (WKT conversion) |
| pyproj | CRS handling |
| cfdb-models | Shared data model types |
| cfdb-vars | Variable definitions and templates |
| geointerp | Grid interpolation and CRS transformation |
| h5netcdf | NetCDF4 I/O (optional) |
| xarray | Xarray backend integration (optional) |
