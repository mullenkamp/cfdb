# S3 Remote (EDataset)

cfdb supports syncing datasets with S3-compatible object storage via [EBooklet](https://github.com/mullenkamp/ebooklet). The `EDataset` class extends `Dataset` with remote sync capabilities.

## Installation

```bash
pip install cfdb[ebooklet]
```

## Opening an EDataset

```python
import cfdb
from ebooklet import S3Connection

remote_conn = S3Connection(
    endpoint_url='https://s3.example.com',
    access_key_id='YOUR_KEY',
    access_key='YOUR_SECRET',
    bucket='my-bucket',
    object_key='datasets/example.cfdb',
)

with cfdb.open_edataset(remote_conn, 'local_cache.cfdb', flag='r') as ds:
    print(ds)
```

### S3Connection

The `remote_conn` parameter accepts:

| Type | Description |
|------|-------------|
| `S3Connection` | Fully configured connection object |
| `str` | HTTP URL for the remote |
| `dict` | Parameters for `S3Connection()` |

### Parameters

`open_edataset()` accepts the same parameters as `open_dataset()` plus:

| Parameter | Type | Description |
|-----------|------|-------------|
| `remote_conn` | S3Connection, str, or dict | Remote connection |
| `num_groups` | int or None | S3 object groups (required for flag='n') |

## Reading Remote Data

When reading from an EDataset, chunks are loaded from S3 on demand. Use `load()` on a variable to pre-fetch chunks:

```python
with cfdb.open_edataset(remote_conn, 'local.cfdb') as ds:
    temp = ds['temperature']
    temp.load()  # fetch all chunks from S3

    for slices, data in temp.iter_chunks(include_data=True):
        print(data.shape)
```

Selections also trigger loading only the required chunks:

```python
with cfdb.open_edataset(remote_conn, 'local.cfdb') as ds:
    temp = ds['temperature']
    subset = temp[0:10, :]
    subset.load()  # loads only the chunks needed
```

## Writing and Syncing

Write data locally, then changes are synced on close:

```python
with cfdb.open_edataset(remote_conn, 'local.cfdb', flag='w') as ds:
    ds['temperature'][0:10, :] = new_data
```

### Tracking Changes

Check what has changed during the current session:

```python
with cfdb.open_edataset(remote_conn, 'local.cfdb', flag='w') as ds:
    ds['temperature'][0, 0] = 42.0
    changes = ds.changes()
    print(changes)
```

## Remote Management

### Delete Remote

Remove the remote dataset while keeping the local file:

```python
with cfdb.open_edataset(remote_conn, 'local.cfdb', flag='w') as ds:
    ds.delete_remote()
```

### Copy Remote

Copy the entire remote dataset to another S3 location:

```python
new_remote = S3Connection(
    endpoint_url='https://s3.example.com',
    access_key_id='YOUR_KEY',
    access_key='YOUR_SECRET',
    bucket='backup-bucket',
    object_key='datasets/copy.cfdb',
)

with cfdb.open_edataset(remote_conn, 'local.cfdb') as ds:
    ds.copy_remote(new_remote)
```

## Thread and Multiprocess Safety

EDataset inherits the same safety properties as Dataset — thread locks for concurrent reads/writes and file locks for multiprocessing. The S3 remote uses object locking for consistency.
