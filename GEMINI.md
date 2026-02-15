# Gemini Context for cfdb

This file provides context for the Gemini AI agent to understand and work with the `cfdb` project.

## Project Overview

**cfdb** (CF conventions multi-dimensional array database) is a pure Python library for managing labeled multi-dimensional arrays, adhering largely to [CF conventions](https://cfconventions.org/). It serves as an alternative to `netcdf4` and `xarray`, leveraging [Booklet](https://github.com/mullenkamp/booklet) for local file storage and [EBooklet](https://github.com/mullenkamp/ebooklet) for S3-backed synchronization.

### Key Features
*   **Storage:** Uses `Booklet` as a key-value store (string keys for chunks).
*   **Concurrency:** Thread-safe and multiprocessing-safe using file/object locks.
*   **Data Structure:**
    *   **Datasets:** Collections of variables. Two types: `grid` (default) and `orthogonal_multidimensional_array_representation_of_time_series`.
    *   **Coordinates:** 1D, unique, sorted, non-null arrays defining dimensions (e.g., lat, lon, time). Immutable values (append/prepend only).
    *   **Data Variables:** N-dimensional arrays linked to coordinates. Data never held fully in memory; accessed via chunks.
*   **Compression:** Supports `zstd` and `lz4`.

## Architecture & Core Modules

*   **Entry Points** (`cfdb/main.py`): `open_dataset` (local) and `open_edataset` (S3).
*   **Data Models** (`cfdb/data_models.py`): `msgspec` Structs for metadata (`SysMeta`, `CoordinateVariable`, `DataVariable`).
*   **Support Classes** (`cfdb/support_classes.py`): Runtime objects (`Variable`, `Coordinate`, `DataVariable`, `Rechunker`).
*   **Data Types** (`cfdb/dtypes.py`): Custom type system (`Float`, `Integer`, `DateTime`, `Geometry` w/ WKT).
*   **Creation** (`cfdb/creation.py`): Factory patterns for creating coords and variables.
*   **Indexing** (`cfdb/indexers.py`): Handling selection logic (`.loc[]`, integer indexing).
*   **Legacy code** (`core.py`): Old h5py-based implementation, not part of the current API. `combine.py` is also legacy (uses h5py/xarray directly).

## Build & Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and [hatchling](https://hatch.pypa.io/latest/) as the build backend.

### Prerequisites
*   Python >= 3.10
*   `uv` installed

### Key Commands

**Setup & Installation**
```bash
uv sync --dev          # Install dependencies (including dev group)
```

**Testing**
```bash
uv run pytest          # Run all tests
uv run pytest cfdb/tests/test_dataset.py  # Run specific test file
```
*Note: S3 tests require credentials in `cfdb/tests/s3_config.toml` or env vars.*

**Linting & Formatting**
```bash
uv run ruff check .    # Run linter
uv run black --check . # Check formatting
```

**Documentation**
```bash
mkdocs build           # Build documentation
mkdocs serve           # Serve documentation locally
```

## Conventions

*   **Style:** Follows `black` formatting and `ruff` linting rules.
*   **Testing:** Uses `pytest`.
*   **Code Structure:**
    *   Prefer context managers (`with cfdb.open_dataset(...)`) for safe file handling.
    *   Iterate over chunks (`iter_chunks`) for large data operations to minimize memory usage.
    *   Coordinates must be created before data variables.
