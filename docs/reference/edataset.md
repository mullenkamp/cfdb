# EDataset

The `EDataset` class extends `Dataset` with S3 remote sync capabilities via [EBooklet](https://github.com/mullenkamp/ebooklet). Created via [`open_edataset()`](open-edataset.md).

## Usage

```python
import cfdb
from ebooklet import S3Connection

remote = S3Connection(
    endpoint_url='https://s3.example.com',
    access_key_id='KEY',
    access_key='SECRET',
    bucket='bucket',
    db_key='data.cfdb',
)

with cfdb.open_edataset(remote, 'local.cfdb') as ds:
    print(ds)
```

## Inherited Interface

EDataset inherits all properties and methods from [Dataset](dataset.md), including:

- Dict-like variable access (`ds['var_name']`, `del ds['var_name']`, etc.)
- `select()` / `select_loc()`
- `copy()` / `to_netcdf4()`
- `attrs`, `coords`, `data_vars`, etc.

## Additional Methods

### push(force_push=False)

Push local changes to the remote. Safe to call mid-session. Returns an ebooklet
`PushResult`: `updated` (the remote changed), `failures` (failed keys/groups →
error strings; the pending changes are kept for a retry), and `bool(result)`,
which is True only for a fully successful push that changed the remote. Use
`force_push=True` to retry after a partial failure.

```python
result = ds.push()
```

### changes()

Return a Change object describing modifications made during this session.

```python
changes = ds.changes()
```

### delete_remote()

Completely delete the remote dataset, keeping the local file.

```python
ds.delete_remote()
```

### copy_remote(remote_conn)

Copy the entire remote dataset to another S3 location. The target must be empty.

```python
ds.copy_remote(new_s3_connection)
```

## Data Loading

When using EDataset, chunks are loaded from S3 on demand. Call `load()` on a variable to pre-fetch:

```python
temp = ds['temperature']
temp.load()  # downloads required chunks from S3
```
