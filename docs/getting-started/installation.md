# Installation

## Requirements

- Python >= 3.10

## Install from PyPI

=== "pip"

    ```bash
    pip install cfdb
    ```

=== "uv"

    ```bash
    uv add cfdb
    ```

## Optional Extras

cfdb has optional dependency groups for additional functionality:

| Extra | Packages | Purpose |
|-------|----------|---------|
| `netcdf4` | h5netcdf, cftime | NetCDF4 import/export |
| `ebooklet` | ebooklet | S3 remote sync (EDataset) |

Install extras as needed:

=== "pip"

    ```bash
    pip install cfdb[netcdf4]
    pip install cfdb[ebooklet]
    pip install cfdb[netcdf4,ebooklet]
    ```

=== "uv"

    ```bash
    uv add cfdb[netcdf4]
    uv add cfdb[ebooklet]
    uv add cfdb[netcdf4,ebooklet]
    ```

## Development Setup

cfdb uses [uv](https://docs.astral.sh/uv/) for environment management.

```bash
git clone https://github.com/mullenkamp/cfdb.git
cd cfdb
uv sync --dev
```

Run tests:

```bash
uv run pytest
```

!!! note
    EDataset tests require S3 credentials configured either in `cfdb/tests/s3_config.toml` or via environment variables (`endpoint_url`, `access_key_id`, `access_key`).
