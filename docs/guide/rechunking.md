# Rechunking

All variable data in cfdb is stored as chunks. The chunk shape chosen at creation may be optimal for some access patterns but not others. cfdb provides two ways to iterate with a different chunk layout — without modifying the stored data.
The rechunking functionality is provided by the [rechunkit](https://github.com/mullenkamp/rechunkit) package.

## Why Rechunking Matters

Consider a dataset with shape `(2000, 2000)` stored in `(100, 100)` chunks. Reading a single row requires touching 20 chunks. If your workload reads full rows, a `(1, 2000)` chunk shape would be ideal — but that same shape is terrible for column reads.

Rechunking solves this by reading the existing chunks and yielding data in a new chunk layout, using an intermediate in-memory buffer to minimize reads.

## Using iter_chunks with chunk_shape

The simplest way to rechunk is to pass a `chunk_shape` dict to `iter_chunks()`. Coordinates not listed use their full length:

```python
with cfdb.open_dataset(file_path) as ds:
    temp = ds['temperature']

    # Iterate with 50-element latitude chunks (other coords use full length)
    for slices, data in temp.iter_chunks({'latitude': 50}):
        print(slices, data.shape)

    # Position-only (no data loading)
    for slices in temp.iter_chunk_slices({'latitude': 50}):
        print(slices)
```

This also works at the dataset level:

```python
with cfdb.open_dataset(file_path) as ds:
    for target_chunk, var_data in ds.iter_chunks({'latitude': 50}):
        print(target_chunk, {k: v.shape for k, v in var_data.items()})
```

## Using the Rechunker Directly

For advanced use cases — estimating rechunk costs, guessing chunk shapes, or fine-tuning the memory buffer — use the `Rechunker` class directly:

```python
with cfdb.open_dataset(file_path) as ds:
    temp = ds['temperature']
    rechunker = temp.rechunker()
```

## Guessing a Chunk Shape

Estimate a chunk shape that fits within a target byte budget:

```python
rechunker = temp.rechunker()
chunk_shape = rechunker.guess_chunk_shape(target_chunk_size=2**8)  # 256 bytes
print(chunk_shape)
```

The algorithm uses composite numbers to maximize compatibility when rechunking between different layouts. This is performed automatically when creating a variable when the chunk_shape is not defined.

## Rechunking Data

The `rechunk()` method returns a generator yielding `(slices, data)` tuples in the new chunk layout:

```python
new_chunk_shape = (41, 41)

with cfdb.open_dataset(file_path) as ds:
    rechunker = ds['temperature'].rechunker()

    for slices, data in rechunker.rechunk(new_chunk_shape):
        print(slices, data.shape)
```

### Memory Control

The `max_mem` parameter (default 128 MB) controls the intermediate buffer size:

```python
for slices, data in rechunker.rechunk(new_chunk_shape, max_mem=2**27):
    print(data.shape)
```

Larger buffers reduce the number of read operations but use more memory. The rechunker will only use an amount of memory up to the ideal chunk shape. See the [rechunkit](https://github.com/mullenkamp/rechunkit) package for more details.

## Planning Rechunk Operations

Estimate the cost of a rechunk before doing it:

```python
rechunker = temp.rechunker()

# Number of existing chunks
n_chunks = rechunker.calc_n_chunks()

# Number of reads and writes for a rechunk operation
n_reads, n_writes = rechunker.calc_n_reads_rechunker(new_chunk_shape)

# Ideal read chunk shape (LCM-based)
ideal_shape = rechunker.calc_ideal_read_chunk_shape(new_chunk_shape)

# Memory required for the ideal read chunk
ideal_mem = rechunker.calc_ideal_read_chunk_mem(new_chunk_shape)

# Optimal read shape given a memory budget
optimal_shape = rechunker.calc_source_read_chunk_shape(new_chunk_shape, max_mem=2**27)
```

## Rechunker Methods Summary

| Method | Description |
|--------|-------------|
| `guess_chunk_shape(target_chunk_size)` | Estimate chunk shape for a byte budget |
| `rechunk(target_chunk_shape, max_mem)` | Generator yielding rechunked data |
| `calc_n_chunks()` | Count existing chunks |
| `calc_n_reads_rechunker(target, max_mem)` | Estimate read/write count |
| `calc_ideal_read_chunk_shape(target)` | Minimum ideal read chunk shape |
| `calc_ideal_read_chunk_mem(target)` | Memory for ideal read chunk |
| `calc_source_read_chunk_shape(target, max_mem)` | Optimal read shape for memory budget |

## Relationship to GroupBy and iter_chunks

The `groupby()` method is sugar over `iter_chunks()` — it sets chunk size 1 along the grouped dimensions and uses the full length for all others. Similarly, `iter_chunks(chunk_shape)` delegates to the rechunker internally when a custom chunk shape is provided. The `Rechunker` class is useful when you need the planning methods (`calc_n_reads_rechunker`, `guess_chunk_shape`, etc.) or direct control over the rechunk operation. See [Data Variables — GroupBy](data-variables.md#groupby).
