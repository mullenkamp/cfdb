# Chunking & Storage

cfdb stores all data — both coordinates and data variables — as compressed chunks. This page explains how chunking works and how to choose good chunk shapes.

## What is a Chunk?

A chunk is a fixed-size rectangular block of the full array. For example, a variable with shape `(1000, 2000)` and chunk shape `(100, 200)` is stored as 100 separate chunks (10 along the first axis, 10 along the second).

Each chunk is independently compressed and stored as a single Booklet key-value entry.

## Chunk Key Format

Chunk keys follow the pattern:

```
{var_name}!{dim0_start},{dim1_start},...
```

For example, chunk `(200, 400)` of variable `temperature` is stored with key `temperature!200,400`.

## Compression

Every chunk is compressed before storage. The algorithm is set at dataset creation:

| Algorithm | Library | Characteristics |
|-----------|---------|----------------|
| `zstd` | zstandard | Best compression ratio at reasonable speed (default) |
| `lz4` | lz4 | Fastest compression/decompression |

Compression level defaults to 1 for both algorithms. Higher levels improve ratio but slow down writes.

## Automatic Chunk Shape

When `chunk_shape=None` is passed during variable creation, cfdb uses `rechunkit.guess_chunk_shape()` to estimate an appropriate chunk shape based on:

- The variable's total shape
- The dtype's element size
- A target chunk byte size

The algorithm prefers **composite numbers** for chunk dimensions. This is important because rechunking between two chunk shapes is most efficient when the least common multiple (LCM) of corresponding dimensions is small — and composite numbers tend to have lower LCMs than primes. 

The trade off is that a larger chunk would have a higher compression ratio, but a larger chunk would slow downs reads do to having to decompress a large amount of data for a small slicing request.

The default chunk byte size is a maximum of ~2 MB. Both compression algorithms used in cfdb tend to max out the compression ratio between 1-2 MB of raw data. A chunk byte size greater than 2 MB would not significantly improve the compression and would slow down reads. If anything, the user should reduce the default chunk byte size rather than increase it.

## Choosing Chunk Shapes

The optimal chunk shape depends on your access pattern:

| Access Pattern | Ideal Chunk Shape |
|---------------|-------------------|
| Read full rows | `(1, N)` — thin along rows, wide along columns |
| Read full columns | `(N, 1)` — wide along rows, thin along columns |
| Read spatial blocks | `(M, M)` — square chunks |
| Time series at one point | `(1, 1, T)` — thin spatially, long temporally |
| Spatial snapshot at one time | `(Y, X, 1)` — wide spatially, thin temporally |

In practice, the auto-estimated chunk shape is a reasonable starting point. Use the [Rechunker](../guide/rechunking.md) when you need a different access pattern.

## Coordinate Chunk Storage

Coordinates are also stored as chunks, but they always hold the full data **in memory**. This is because coordinate data is typically small (1-D arrays) and needed frequently for index lookups.

## Data Variable Chunk Storage

Data variables **never** hold full data in memory. Every read goes through the chunk store. This keeps memory usage predictable even for very large datasets.

## Chunk Alignment and Origins

Coordinates can have a non-zero **origin** when data is prepended. The origin tracks the starting position of the coordinate in the global index space. This allows prepending data without rewriting existing chunks.

For example, if a coordinate originally starts at index 0 and you prepend 100 values, the origin becomes -100 and existing chunks keep their original keys.
