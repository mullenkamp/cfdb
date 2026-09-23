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

| `compression` | Library | Characteristics |
|-----------|---------|----------------|
| `zstd_shuffle` | zstandard | **Default.** Byte-shuffled zstd: the smallest files and faster than plain zstd both ways for chunks above a few thousand elements |
| `zstd` | zstandard | Plain zstd; readable by cfdb < 0.10 |
| `lz4_shuffle` | lz4 | Byte-shuffled lz4: smaller than `lz4`, but decodes slower than it |
| `lz4` | lz4 | Fastest decompression, largest files |

The `*_shuffle` values split each chunk's values into byte planes before compressing (all low
bytes, then all high bytes, ...). Packed values keep smooth, compressible high bytes and noisy low
bytes; separating them lets the codec compress each well. The split uses the width of the values as
stored (e.g. 2 bytes for a float packed to uint16); only 2-, 4- and 8-byte values are shuffled —
1-byte, bool, 16-byte, string and geometry variables are stored unshuffled. The size win is largest
for packed values; unpacked full-precision `float64` gains little and, where many values repeat
exactly, can come out up to ~13 % larger shuffled (while still reading and writing faster), so use
`compression='zstd'` for such data if size matters most. Measurements behind these choices:
`benchmarks/compression/README.md` and the review record `benchmarks/results/review-cfdb-shuffle-code-1.md`.

Compression level defaults to 1 for every option; with the shuffle in front, higher zstd levels gain
~1.5 % in size for ~1.3–1.4× slower writes. The compression is recorded in the file and used for
every later read and write; files using a `*_shuffle` value need cfdb >= 0.10.

## Automatic Chunk Shape

When `chunk_shape=None` is passed during variable creation, cfdb uses `rechunkit.guess_chunk_shape()` to estimate an appropriate chunk shape based on:

- The variable's total shape
- The dtype's element size
- A target chunk byte size

The algorithm prefers **composite numbers** for chunk dimensions. This is important because rechunking between two chunk shapes is most efficient when the least common multiple (LCM) of corresponding dimensions is small — and composite numbers tend to have lower LCMs than primes. 

For data variables the default target is **2¹⁸ elements per chunk** (passed to rechunkit as a byte
target of 2¹⁸ × the stored item size: 512 KiB for packed uint16, 1 MiB for 4-byte, 2 MiB for 8-byte
values; the guess may exceed it by up to 1.5×). Coordinates and string/geometry variables use a
2 MiB byte target.

Why elements rather than bytes: what a chunk costs depends mostly on how many values it holds.
Measured on real data (`benchmarks/RESULTS.md`):

- **Compression ratio** is essentially flat from multi-MB chunks down to a few thousand elements, so
  large chunks buy almost no extra compression.
- **Reads** are cheapest per value at roughly 10⁵–5·10⁵ elements per chunk. Much smaller chunks pay
  a fixed cost of ~10 µs per chunk (lookup, decode, loop); multi-MB chunks are slower per value
  because every chunk decodes into freshly allocated memory.
- **Rechunking** (`iter_chunks(chunk_shape=...)`, `groupby`) on data larger than its memory budget
  is fastest at ~3·10⁵ elements per chunk and several times slower with multi-MB chunks.
- **Small selections** (one point's time series, one time step) decompress whole chunks, so smaller
  chunks read less.

A single byte target cannot keep every item size in that range: at 512 KiB, 8-byte values got only
~30 K elements per chunk. If you choose chunk shapes yourself, keep them at **≥ ~32 K elements**
(below that, reads and writes get markedly slower per value) and avoid multi-MB chunks unless your
workload is whole-array scans.

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
