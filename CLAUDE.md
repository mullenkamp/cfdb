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

**Data models** (`data_models.py`): msgspec Structs defining metadata schemas — `SysMeta` (top-level), `CoordinateVariable`, `DataVariable`. Shared types (`Type`, `Compressor`, `Axis`, `DataType`) are imported from `cfdb_models.data_models`. System metadata is stored in the Booklet file's metadata field. Variables are tagged unions (`tag='data_var'` / `tag='coord'`).

**Support classes** (`support_classes.py`): Runtime variable objects exposed to users. Hierarchy:
- `Variable` → `CoordinateView` → `Coordinate` (coordinates hold data in memory, support append/prepend)
- `Variable` → `DataVariableView` → `DataVariable` (data vars never hold full data in memory, support `__setitem__`)
- `Attributes` — JSON-serializable attrs stored as a separate Booklet key (`_{var_name}.attrs`)
- `Compressor` — wraps zstd or lz4, optionally with a byte-shuffle filter (`zstd_shuffle`/`lz4_shuffle`). Bound PER VARIABLE to the stored value width via `shuffle_itemsize(dtype)` (the ENCODED width for packed dtypes; None for String/Geometry, which must be gated by kind — their numpy placeholder dtypes report itemsize 16). Never bind from `dtype.itemsize`, which is the decoded width. `Dataset._compressor_for(itemsize)` caches one per width
- `Rechunker` — wraps the rechunkit package for chunk shape conversion. Declares the array to rechunkit in a STORAGE-SPACE-aligned declared space (`declared = user + origin % chunk`), so coordinate origins from prepend/append are folded into rechunkit's phase machinery and every read maps to exactly one storage chunk. The multi-chunk assembly in the source function is a defensive fallback (kept + tested).
- `DatasetRechunker` — synchronized multi-variable rechunking. Zips per-variable `Rechunker` generators, relying on rechunkit's deterministic iteration order.

**Data types** (`dtypes.py`): Custom type system for serialization. `DataType` base class with subclasses: `Float`, `Integer`, `DateTime`, `Bool`, `String`, `Point`, `LineString`, `Polygon`. Each dtype handles encode/decode (scaling, offset) and dumps/loads (bytes serialization). Geometry types use WKT via shapely + msgpack.

**Variable creation** (`creation.py`): `Creator` exposes `coord`, `data_var`, and `crs` sub-objects. Template methods for common variables (lat, lon, time, etc.) are dynamically generated via decorators `@create_coord_methods` and `@create_data_var_methods`.

**Indexing** (`indexers.py`): Handles index-based and location-based selection. Converts user selections to chunk keys for Booklet lookups. `LocationIndexer` supports `.loc[]` syntax.

**Conversion tools** (`tools.py`): `cfdb_to_netcdf4()` exports a cfdb to netCDF4 (requires h5netcdf). Importing netCDF4/other formats *into* cfdb lives in the separate [cfdb-ingest](https://github.com/mullenkamp/cfdb-ingest) package, not here.

**Legacy code** (`core.py`): Old h5py-based implementation, not part of the current API.

### Key Patterns

- **Chunk storage**: Data chunks are stored in Booklet with keys formatted as `{var_name}!{dim_starts}` (see `utils.make_var_chunk_key`).
- **Metadata lifecycle**: `SysMeta` is deserialized from Booklet metadata on open and serialized back via `weakref.finalize` on close.
- **Dataset types**: `grid` (standard N-D), `ts_ortho` (time series with point geometries), and — from 0.9.6 — `ts_forecast` / `grid_forecast`, which swap the `time` axis for `(forecast_reference_time, forecast_period)`. Controlled by the `data_models.Type` enum, which lives in the **separate `cfdb-models` distribution** (so a new type needs a cfdb-models release, and `cfdb`'s floor on it bumped in the same commit). ⚠️ **Nearly every `dataset_type` check is exact string equality** — a new type inherits nothing and silently skips guards; `utils.parse_coord_inputs`'s `'ts_' in dataset_type` is the sole substring test. `.interp()` raises `NotImplementedError` for both forecast types (two non-spatial dims).
- **Compression**: one setting per dataset, recorded in `SysMeta.compression` (the `Compressor` enum from cfdb-models): `zstd_shuffle` (default since 0.10), `zstd`, `lz4_shuffle`, `lz4`. The shuffle lives IN the enum, not in a separate flag, so an older cfdb refuses a shuffled file loudly (`Invalid enum value`) instead of silently ignoring an unknown field and reading shuffled bytes as plain ones — keep it that way. `SysMeta.format_version` (1 since 0.10; 0 = older file) is checked on open for future format changes. Tests pin the stored byte layout and old/new files: `cfdb/tests/test_file_format.py`, fixtures in `cfdb/tests/fixtures/` (never edit a committed fixture; add a new one with `make_fixtures.py`).
- **Default chunk size**: data variables target 2¹⁸ ELEMENTS (`utils.data_var_chunk_elements`, a byte target of 2¹⁸ × stored item size); coordinates and str/geometry variables keep 2 MiB (`utils.coord_chunk_max`, `utils.var_length_chunk_max`). Evidence in `benchmarks/RESULTS.md`.
- **Coordinate mutability** — values cannot be changed in-place. Coordinates support `append()`, `prepend()`, and `truncate(start, stop)` (removes values outside [start, stop] inclusive, along with orphaned data variable chunks). Must be unique and ascending. There is no insert operation.
- **Step auto-fill on append/prepend** — when a coordinate has an enforced `step` and new data is appended/prepended with a gap (not adjacent to existing data), cfdb automatically generates the missing intermediate coordinate values if the gap is a valid multiple of the step. Data variable positions at the filled coordinates are empty (NaN/fillvalue) until written. If the gap is not a valid multiple, a `ValueError` is raised.
- **Rechunking and coordinate origins**: the `Rechunker` declares the source grid to rechunkit at the real storage-chunk alignment — `d = origin % chunk` per dim; `declared = user + d`; `decl_shape = full_shape + d`; `sel_decl = sel + d`; the source function converts back with a constant shift that is always a multiple of `chunk_shape`. Every read then maps to exactly ONE storage chunk (post-prepend decompress amplification 3.4x → 1.0x). Do NOT pass `None` as sel when `d != 0` (materialize the full-range sel — otherwise rechunkit drops the phase), and do NOT delete the multi-chunk assembly loop in the source functions — it is the tested defensive fallback. `calc_n_reads_rechunker` applies the identical transform so predictions match instrumented reads. Yields are selection-relative and therefore unchanged by the transform (pinned by tests). See `docs/concepts/rechunking-internals.md`.
- **rechunkit memory contract (perf round)**: `max_mem` is an honest total (bulk buffer + pending copies + batch buffers) with documented floors and the wide-array residual; `Rechunker`/`iter_chunks`/`groupby`/`map`/`DatasetRechunker` all default to 2**29. The `calc_ideal_read_chunk_*` methods pass `shape=` so reported numbers are extent-clipped — keep that threading when touching budgeting APIs.
- **GroupBy with time periods**: `groupby()` on both `DataVariableView` and `Dataset` accepts a dict with period strings (e.g. `{'time': 'D'}`, `{'time': 'M'}`). Both call sites resolve the spec through ONE shared helper, `utils.resolve_groupby_spec` (which also view-restricts coordinate data — variable views compute groups from their own coord window). The rechunker fast path requires BOTH a regular step-divisible period AND `coord[0]` sitting on the period-unit boundary; otherwise it falls back to calendar-correct slice-based iteration (`compute_time_groups`) — the two paths must always produce identical groups (fast is an optimization, never a semantics change). Anchoring rules: single-unit periods are calendar-aligned; multi-count periods (`'7D'`, `'6h'`) anchor at `coord[0]`; `'W'` anchors on numpy's Thursday week epoch.
- **Missing-value fill for packed dtypes**: auto-packed Integer dtypes carry `fillvalue=0` (encoded 0 is structurally reserved — `offset = min_value − 1` puts legit values at code ≥ 1, INCLUDING legit 0 when min ≤ 0, which packs to `1−min`). `decode()` maps the reserved code to decoded 0; `encode()` has NO zero-mapping — do not add one, it would alias legit 0 with the missing marker and corrupt netCDF exports (`_FillValue=0` would mask real zeros). Missing chunks must read identically (NaN/NaT/0) through `.data`, `rechunk`, `iter_chunks(chunk_shape=…)`, and `groupby`. The fillvalue must survive reopen: `parse_np_dtypes`' explicit Integer branch passes it through (stored dicts re-enter there). Legacy files (no stored fillvalue) keep old semantics and trigger a `UserWarning` on rechunker/encoded reads.
- **Encoded-space write overlay**: `DataVariableView.set()` and `Coordinate._add_updated_data` read-modify-write packed (f/i/u/M-transcoder) chunks in ENCODED space — only the user's slice passes through `encode()`. Do not "simplify" back to decoding the whole chunk and re-encoding: blank cells would hit the encode out-of-range raise (min≥2 packed ints), unwritten cells would stop being honestly missing on disk, and untouched cells would take a lossy-ish decode→re-encode round trip.
- **Numpy indexing semantics + layering**: raw user keys (negative wrap, IndexError on out-of-range ints, slice clamping) are normalized by `indexers.normalize_user_key` at the USER ENTRY POINTS only (`_get`, `check_sel_input_data`/`set`, `select`/`select_loc`). NEVER move clamping into `slice_slice`/`index_combo_all` — internal recomposed `_sel` tuples are full-variable 0-based coordinates evaluated against VIEW shapes, and clamping them empty-selects every view not starting at 0.
- **Yield lifetime**: rechunker-based generators may yield views into rechunkit's reused internal buffer — the documented contract (all chunk-generator docstrings + `docs/concepts/rechunking-internals.md`) is consume-or-copy before advancing, yields are read-only. `iter_chunks` copies its missing-chunk blanks at yield so consumer mutation can't poison later yields — keep those `.copy()` calls.
- **Reference cycles and `weakref.finalize`**: `Dataset` uses `weakref.finalize` for cleanup. Any class STORED as an attribute on `Dataset`/`Variable` that holds a reference back to it **must** use `weakref.proxy(dataset)` — not a direct reference. A strong back-reference creates a cycle that prevents the finalizer from running on Python 3.12+, causing file locks to persist and hangs. This applies to `Creator`/`Coord`/`DataVar`/`CRS` (in `creation.py`). The sanctioned alternative for expression-scoped helpers: create them per access via a property and give them a STRONG reference — safe precisely because they are never stored, so no cycle exists. `LocationIndexer` (in `indexers.py`) uses this pattern; its strong ref is what keeps chained temporaries (`ds[var].loc[...]`) alive mid-expression — do NOT convert it back to a weakref (that was the 0.9.2 chained-`.loc` ReferenceError bug).

### Dependencies

Core: booklet, cfdb-models, rechunkit, numpy, zstandard, msgspec, lz4, shapely, pyproj
Optional: h5netcdf (netcdf4 export support), ebooklet (S3 support)
