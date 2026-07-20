# Gemini Context for cfdb

This file provides context for the Gemini AI agent to understand and work with the `cfdb` project.

## Project Overview

**cfdb** (CF conventions multi-dimensional array database) is a pure Python library for managing labeled multi-dimensional arrays, adhering largely to [CF conventions](https://cfconventions.org/). It serves as an alternative to `netcdf4` and `xarray`, leveraging [Booklet](https://github.com/mullenkamp/booklet) for local file storage and [EBooklet](https://github.com/mullenkamp/ebooklet) for S3-backed synchronization.

### Key Features
*   **Storage:** Uses `Booklet` as a key-value store (string keys for chunks).
*   **Concurrency:** Thread-safe and multiprocessing-safe using file/object locks.
*   **Data Structure:**
    *   **Datasets:** Collections of variables. Two types: `grid` (default) and `orthogonal_multidimensional_array_representation_of_time_series`.
    *   **Coordinates:** 1D, unique, sorted, non-null arrays defining dimensions (e.g., lat, lon, time). Values cannot be changed in-place; supports `append()`, `prepend()`, and `truncate(start, stop)` to grow or shrink. There is no insert operation. When a coordinate has an enforced `step` and new data is appended/prepended with a gap, cfdb auto-fills the missing intermediate values (if the gap is a valid multiple of the step). Data variables at filled positions are empty until written.
    *   **Data Variables:** N-dimensional arrays linked to coordinates. Data never held fully in memory; accessed via chunks.
*   **Rechunking:** Native multivariable rechunking for synchronized, high-performance block iteration.
*   **Compression:** Supports `zstd` and `lz4`.

## Architecture & Core Modules

*   **Entry Points** (`cfdb/main.py`): `open_dataset` (local) and `open_edataset` (S3).
*   **Data Models** (`cfdb/data_models.py`): `msgspec` Structs for metadata (`SysMeta`, `CoordinateVariable`, `DataVariable`).
*   **Support Classes** (`cfdb/support_classes.py`): Runtime objects (`Variable`, `Coordinate`, `DataVariable`, `Rechunker`, `DatasetRechunker`).
*   **Data Types** (`cfdb/dtypes.py`): Custom type system (`Float`, `Integer`, `DateTime`, `Geometry` w/ WKT).
*   **Creation** (`cfdb/creation.py`): Factory patterns for creating coords and variables.
*   **Indexing** (`cfdb/indexers.py`): Handling selection logic (`.loc[]`, integer indexing).
*   **Legacy code** (`core.py`): Old h5py-based implementation, not part of the current API.

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

## Implementation Learnings & Constraints

*   **Rechunkit Determinism:** The underlying `rechunkit` engine is deterministic. Multivariable rechunking works by zipping synchronized generators. This requires all involved variables to share identical coordinate names and shapes. Storage chunking can differ.
*   **Rechunker Offsets & Storage Chunks:** When `rechunkit` operates on a `DatasetView` (a subset with a `sel` offset), it requests data slices that are shifted by that offset. This means `rechunkit`'s read requests can and often do span multiple underlying *storage chunks*. The source read function passed to `rechunkit.rechunker()` **cannot** simply wrap `get_chunk()` (which truncates to a single storage chunk boundary). It must be capable of fetching, assembling, and returning a precisely sliced array that potentially crosses multiple storage chunk boundaries to satisfy `rechunkit`'s exact requested slice.
*   **Memory Management:** `DatasetRechunker` implements a shared memory budget. It divides the user-provided `max_mem` by the number of variables to ensure the total buffer footprint stays within limits.
*   **Indexing vs Slicing:** When verifying data, remember that `cfdb` indexers preserve dimensions. Slicing a `DataVariableView` result (e.g., `.data[0]`) will collapse the first dimension, which can cause shape mismatches in tests.
*   **HDF5/S3 State:** Metadata like `shape` and `dtype` should be captured while file handles are open. Accessing properties on closed underlying handles will trigger errors.
*   **Reference cycles and `weakref.finalize`:** `Dataset` uses `weakref.finalize` for cleanup (flushing metadata, releasing file locks). Any class STORED as an attribute on `Dataset` or `Variable` that holds a reference back to it **must** use `weakref.proxy(dataset)` — not a direct reference. A strong back-reference creates a cycle that prevents the finalizer from running on Python 3.12+, causing file locks to persist and silent hangs during GC/teardown. This applies to `Creator`/`Coord`/`DataVar`/`CRS` (in `creation.py`). The sanctioned alternative for expression-scoped helpers: create them per access via a property and give them a STRONG reference — safe precisely because they are never stored, so no cycle exists. `LocationIndexer` (in `indexers.py`) uses this pattern; its strong ref keeps chained temporaries (`ds[var].loc[...]`) alive mid-expression — do NOT convert it back to a weakref (that was the 0.9.2 chained-`.loc` ReferenceError bug).
*   **Missing-value fill for packed dtypes (2026-07-21 round):** auto-packed Integer dtypes carry `fillvalue=0`; encoded 0 is structurally reserved (offset = min−1 puts every legitimate value, including a legit 0 when min ≤ 0, at code ≥ 1). `decode()` maps the reserved code to decoded 0; `encode()` has NO zero-mapping — adding one would alias legitimate 0 with the missing marker and corrupt netCDF exports. The fillvalue must pass through `parse_np_dtypes`' explicit Integer branch or it evaporates on file reopen.
*   **Encoded-space write overlay:** `DataVariableView.set()` and `Coordinate._add_updated_data` overlay packed (f/i/u/M) chunks in ENCODED space, encoding only the user's slice. Reverting to whole-chunk decode→re-encode breaks min≥2 packed-int partial writes (encode range raise on blank cells) and destroys on-disk missingness.
*   **User-key normalization layering:** numpy indexing semantics (negative wrap, IndexError, slice clamping) live in `indexers.normalize_user_key`, applied ONLY at raw-user-key entry points (`_get`, `set`, `select`/`select_loc`). Never clamp inside `slice_slice`/`index_combo_all`: internal `_sel` tuples are full-variable coordinates evaluated against view shapes — clamping them empty-selects every offset view.
*   **Groupby fast path == fallback:** both groupby implementations resolve through `utils.resolve_groupby_spec`; the rechunker fast path additionally requires `coord[0]` on the period-unit boundary. The fast path is an optimization only — it must always produce the identical groups the calendar fallback would.
*   **Yield lifetime:** rechunker-based generators may yield views into a reused internal buffer — consume or copy before advancing, treat as read-only. `iter_chunks` missing-chunk blanks are copied at yield; keep those copies.
