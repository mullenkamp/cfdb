# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

cfdb is a pure Python database for managing labeled multi-dimensional arrays following [CF conventions](https://cfconventions.org/). It is an alternative to netcdf4/xarray, built on [Booklet](https://github.com/mullenkamp/booklet) for local file storage and [EBooklet](https://github.com/mullenkamp/ebooklet) for S3 sync. Thread-safe and multiprocessing-safe via locks.

## Development Setup

Uses [uv](https://docs.astral.sh/uv/) for environment management. Python >=3.10.

```bash
uv sync --dev          # Install all dependencies including dev group
```

## Commands

```bash
# Run all tests (requires S3 credentials for EDataset tests)
uv run pytest

# Run a single test file
uv run pytest cfdb/tests/test_dtypes.py
uv run pytest cfdb/tests/test_dataset.py

# Run a single test
uv run pytest cfdb/tests/test_dataset.py::test_coord_creation_grid

# Lint
uv run ruff check .
uv run black --check .
```

S3 credentials for EDataset tests come from either `cfdb/tests/s3_config.toml` or environment variables (`endpoint_url`, `access_key_id`, `access_key`).

## Architecture

### Core Layers

**Entry points** (`main.py`): `open_dataset()` and `open_edataset()` are the public API. They return `Dataset` (local) or `EDataset` (S3-backed) objects. Datasets use Booklet as a key-value store with string keys for chunk data.

**Data models** (`data_models.py`): msgspec Structs defining metadata schemas — `SysMeta` (top-level), `CoordinateVariable`, `DataVariable`, `DataType`. System metadata is stored in the Booklet file's metadata field. Variables are tagged unions (`tag='data_var'` / `tag='coord'`).

**Support classes** (`support_classes.py`): Runtime variable objects exposed to users. Hierarchy:
- `Variable` → `CoordinateView` → `Coordinate` (coordinates hold data in memory, support append/prepend)
- `Variable` → `DataVariableView` → `DataVariable` (data vars never hold full data in memory, support `__setitem__`)
- `Attributes` — JSON-serializable attrs stored as a separate Booklet key (`_{var_name}.attrs`)
- `Compressor` — wraps zstd or lz4 compression
- `Rechunker` — wraps the rechunkit package for chunk shape conversion

**Data types** (`dtypes.py`): Custom type system for serialization. `DataType` base class with subclasses: `Float`, `Integer`, `DateTime`, `Bool`, `String`, `Point`, `LineString`, `Polygon`. Each dtype handles encode/decode (scaling, offset) and dumps/loads (bytes serialization). Geometry types use WKT via shapely + msgpack.

**Variable creation** (`creation.py`): `Creator` exposes `coord`, `data_var`, and `crs` sub-objects. Template methods for common variables (lat, lon, time, etc.) are dynamically generated via decorators `@create_coord_methods` and `@create_data_var_methods`.

**Indexing** (`indexers.py`): Handles index-based and location-based selection. Converts user selections to chunk keys for Booklet lookups. `LocationIndexer` supports `.loc[]` syntax.

**Conversion tools** (`tools.py`): `netcdf4_to_cfdb()` and `cfdb_to_netcdf4()` for format conversion. Requires h5netcdf.

**Legacy code** (`core.py`): Old h5py-based implementation, not part of the current API. `combine.py` is also legacy (uses h5py/xarray directly).

### Key Patterns

- **Chunk storage**: Data chunks are stored in Booklet with keys formatted as `{var_name}!{dim_starts}` (see `utils.make_var_chunk_key`).
- **Metadata lifecycle**: `SysMeta` is deserialized from Booklet metadata on open and serialized back via `weakref.finalize` on close.
- **Dataset types**: `grid` (standard N-D) and `ts_ortho` (time series with point geometries). Controlled by `data_models.Type` enum.
- **Compression**: All data compressed with zstd (default) or lz4, configured at dataset level.
- **Coordinates are immutable once written** — values cannot be changed, only appended/prepended. Must be unique and ascending.

### Dependencies

Core: booklet, rechunkit, numpy, zstandard, msgspec, lz4, shapely, pyproj
Optional: h5netcdf + cftime (netcdf4 support), ebooklet (S3 support)
