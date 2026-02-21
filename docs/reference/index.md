# API Reference

This section provides detailed API documentation for cfdb's public interface.

## Entry Points

| Function | Description |
|----------|-------------|
| [`open_dataset`](open-dataset.md) | Open a local cfdb dataset |
| [`open_edataset`](open-edataset.md) | Open an S3-backed cfdb dataset |

## Core Classes

| Class | Description |
|-------|-------------|
| [`Dataset`](dataset.md) | Main dataset class (dict-like access to variables) |
| [`EDataset`](edataset.md) | S3-backed dataset (extends Dataset) |
| [`Coordinate`](coordinate.md) | Coordinate variable (holds data in memory) |
| [`DataVariable`](data-variable.md) | Data variable (chunk-based access) |

## Creation

| Class | Description |
|-------|-------------|
| [`Creator`](creator.md) | Variable and CRS creation interface (`ds.create`) |

## Data Types

| Function/Class | Description |
|----------------|-------------|
| [`dtypes.dtype()`](dtypes.md) | Factory function for creating data types |

## Interpolation

| Class | Description |
|-------|-------------|
| [`GridInterp`](grid-interp.md) | Grid interpolation wrapper |

## Rechunking

| Class | Description |
|-------|-------------|
| [`Rechunker`](rechunker.md) | On-the-fly rechunking interface |

## Tools

| Function | Description |
|----------|-------------|
| [`cfdb_to_netcdf4`](tools.md) | Convert cfdb to netCDF4 |

## Module Structure

```
cfdb/
├── __init__.py          # Public API: open_dataset, open_edataset, dtypes, etc.
├── main.py              # Dataset, DatasetView, open_dataset()
├── edataset.py          # EDataset, open_edataset()
├── support_classes.py   # Variable, Coordinate, DataVariable, Rechunker, etc.
├── creation.py          # Creator, Coord, DataVar, CRS
├── dtypes.py            # DataType hierarchy, dtype() factory
├── grid_interp.py       # GridInterp wrapper
├── data_models.py       # msgspec Structs for metadata
├── indexers.py           # Index and location-based selection
├── tools.py             # NetCDF4 conversion functions
└── utils.py             # Internal utilities
```
