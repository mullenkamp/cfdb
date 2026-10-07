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
    db_key='datasets/example.cfdb',
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
| `group_bytes` | int, None, or omitted | How chunks are stored as S3 objects (ebooklet ≥ 0.11). An int: grouped - chunks new to the remote are packed, in the order they were written, into group objects of up to that many bytes. `None`: ungrouped, one object per chunk. Omitted: an existing remote keeps its mode and the `group_bytes` it records (whatever its last push packed with); a new dataset is grouped at 32 MiB. Another int packs new chunks to it from then on and replaces the recorded value. See [Chunk sizes for remote datasets](#chunk-sizes-for-remote-datasets). |

## Reading Remote Data

When reading from an EDataset, chunks are loaded from S3 on demand. Use `load()` on a variable to pre-fetch chunks:

```python
with cfdb.open_edataset(remote_conn, 'local.cfdb') as ds:
    temp = ds['temperature']
    temp.load()  # fetch all chunks from S3

    for slices, data in temp.iter_chunks():
        print(data.shape)
```

Selections also trigger loading only the required chunks:

```python
with cfdb.open_edataset(remote_conn, 'local.cfdb') as ds:
    temp = ds['temperature']
    subset = temp[0:10, :]
    subset.load()  # loads only the chunks needed
```

## Writing and Pushing

Writes only modify the local file. Nothing is ever uploaded automatically —
publishing to the remote is always an explicit `push()`:

```python
with cfdb.open_edataset(remote_conn, 'local.cfdb', flag='w') as ds:
    ds['temperature'][0:10, :] = new_data
    ds.push()
```

`push()` can be called at any point in the session — including in the same
session that creates the dataset — and publishes the current state of the
dataset (variables, data, and attributes). Closing without pushing simply
leaves the changes local; a later session can push them.

### Tracking Changes

Check what has changed during the current session:

```python
with cfdb.open_edataset(remote_conn, 'local.cfdb', flag='w') as ds:
    ds['temperature'][0, 0] = 42.0
    changes = ds.changes()
    print(changes)
```

### Attaching to an Existing Remote

The local file is just a cache/working copy: opening with `'w'` or `'c'` and a
fresh local file path attaches to the existing remote dataset (its structure
is pulled on demand). A new dataset is only created when one exists neither
locally nor remotely (or with `flag='n'`, which always creates new).

### Dataset Types

Both dataset types work as remotes: pass `dataset_type='ts_ortho'` when
*creating* a station-time-series remote (as of 0.9.1 — earlier versions raised).
Existing remotes always open with their stored type and the matching class;
check it via the `dataset_type` property:

```python
with cfdb.open_edataset(remote, 'stations.cfdb') as ds:
    print(ds.dataset_type)   # 'grid', 'ts_ortho', 'ts_forecast' or 'grid_forecast'
```

## Chunk sizes for remote datasets

cfdb's default chunk size (about 2¹⁸ elements; see
[Chunking & Storage](../concepts/chunking-storage.md#automatic-chunk-shape)) is tuned for local reads.
A remote adds a per-request cost of roughly 0.15–0.4 s. That usually outweighs the local
differences between chunk sizes, so pick the chunk shape for a remote with requests in mind.

### Grouped or ungrouped

| | Grouped (`group_bytes=<int>`, the default) | Ungrouped (`group_bytes=None`) |
|---|---|---|
| Storage | chunks packed in write order into objects of up to `group_bytes` | one object per chunk |
| Requests per read | one per group touched | one per chunk touched |
| An append uploads | the new chunks, plus at most the partly filled last group | only the new chunks |
| An update of existing chunks re-uploads | each group holding a changed chunk | only the changed chunks |
| Suits | most datasets, including ones that grow by appending | datasets pushed very often in small increments (each push would re-upload the last group) |

Grouping follows **write order**: chunks written together share groups. A dataset built and extended
time band by time band keeps each band in its own groups, so an append touches only new groups, and
a read of one band's chunks needs few requests. (Before ebooklet 0.11 groups were assigned by hashing
the chunk key, which scattered every append over nearly all groups; such remotes must be moved to the
current format - see ebooklet's changelog.)

### Ungrouped remotes: make chunks at least ~1.5–2 MB compressed

Every chunk is a separate request, and requests are expensive. In measurements over a home connection
in New Zealand, a remote served 15–22 requests per second on 10 threads regardless of object size.
Below about 1.5–2 MB per object, reads are limited by the number of requests rather than by
bandwidth. So for an ungrouped remote:

- **Size chunks at or above ~1.5–2 MB compressed.** That is usually larger than cfdb's default, so
  pass `chunk_shape` explicitly when creating the variables.
- **Shape them for the most common query**, which for a growing dataset is usually "the latest
  data". The ECMWF forecast layout (one run × all lead times × one level × the full grid per chunk)
  answers "latest run" in a few requests.
- **Don't leave empty chunks behind.** A chunk that holds only missing values still costs a request.
  Write only where there is data, rather than whole rows padded with missing values.

For station data (`ts_ortho`), the choice is between one station per chunk and many stations per
chunk, and it depends on which query matters most. Measured on an hourly streamflow dataset
(141 stations, 21 years), with times projected from the real compressed chunk sizes:

| Chunk shape | Latest 7 days, all stations | One station, full history | Whole dataset | Objects uploaded per hourly update |
|---|---|---|---|---|
| (1 station, 25 000 hours) | 141 requests, 6.6 s | 7 requests, 0.3 s | 801 requests, 39 s | 141 |
| (141 stations, 2 190 hours) | 1 request, 0.1 s | 84 requests, 6.5 s | 84 requests, 6.5 s | 1 |

Putting every station into quarter-year chunks makes the latest-data read and each update a single
request, but turns one station's history into a scan (0.3 s becomes 6.5 s). Keep one station per
chunk when single-station histories are the main read; use multi-station chunks when "latest data,
all stations" is. The (141, 2 190) shape is ~310 K elements, inside cfdb's best range for local reads.

```python
# Growing station dataset on an ungrouped remote, read mostly as "latest data, all stations":
# all stations x one quarter (2 190 hours) per chunk
with cfdb.open_edataset(remote_conn, 'flow.cfdb', flag='n', dataset_type='ts_ortho', group_bytes=None) as ds:
    ...  # create the point and time coordinates
    ds.create.data_var.generic('streamflow', ('point', 'time'), dtype=flow_dtype,
                               chunk_shape=(n_stations, 2190))
```

### Grouped remotes: chunk shape decides the bytes

Grouping bundles chunks into fewer requests, so smaller chunks cost little extra. The cost that
remains is bytes downloaded: a read fetches whole chunks, and one request per group spans from the
first to the last chunk needed in that group. On a (hash-grouped, pre-0.11) SST archive with chunks
120 days deep and 134 MB groups, reading one day downloaded all 120 days, and one point's 45-year
series downloaded 24× the data it needed. Shape the chunks for the queries you expect. Write-order
groups of the default 32 MiB bound that span to one group; a smaller `group_bytes` lowers it further
at the cost of more objects (the default has not yet been benchmarked against alternatives).

The measurements behind this section are in the repository's `benchmarks/RESULTS.md`, sections
"Remote (EDataset) reads vs chunk size" and "Station (ts_ortho) data".

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
    db_key='datasets/copy.cfdb',
)

with cfdb.open_edataset(remote_conn, 'local.cfdb') as ds:
    ds.copy_remote(new_remote)
```

## Thread and Multiprocess Safety

EDataset inherits the same safety properties as Dataset — thread locks for concurrent reads/writes and file locks for multiprocessing. The S3 remote uses object locking for consistency.
