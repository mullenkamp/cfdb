# Rechunking Internals

This page explains how cfdb's rechunking system works under the hood. It is aimed at contributors and advanced users who want to understand the design, debug performance, or extend the rechunking code. For usage instructions, see the [Rechunking guide](../guide/rechunking.md).

## Overview

Rechunking converts data from one chunk layout to another without modifying the stored data. The core problem is straightforward: read chunks in the old layout and yield them in the new one. The complexity comes from doing this efficiently under a memory budget, handling coordinate origins, supporting encoded dtypes, and synchronizing across multiple variables.

The rechunking pipeline has three layers:

1. **rechunkit** (external library) — plans which source chunks to read, manages an intermediate buffer, and yields data in the target layout
2. **Rechunker** (cfdb) — bridges cfdb's storage layer and rechunkit by providing a source function that reads arbitrary slices from Booklet chunks
3. **DatasetRechunker** (cfdb) — coordinates multiple `Rechunker` instances to yield synchronized chunks across variables

## The Source Function

When a user calls `var.rechunker().rechunk(target_shape, max_mem)`, cfdb constructs a **source function** and passes it to `rechunkit.rechunker()`. This source function is the critical bridge between cfdb's chunk storage and rechunkit's generic rechunking algorithm.

rechunkit calls the source function with a tuple of slices (absolute positions in the full array) and expects back a numpy array of exactly that shape. rechunkit decides *which* slices to request based on the source chunk shape, target chunk shape, and memory budget. cfdb's job is to fulfill those requests from Booklet storage.

### Single-Chunk Fast Path

Most rechunkit read requests are aligned to the source chunk boundaries and fit within a single storage chunk. In this common case, the source function takes a fast path:

1. Map the requested slices to a Booklet chunk key via `slices_to_chunks_keys()`
2. Read the compressed bytes from Booklet
3. Decompress and deserialize into a numpy array
4. Slice out the requested portion and return it directly

No intermediate allocation is needed — the sliced array from the deserialized chunk is returned as-is.

### Multi-Chunk Assembly

When a read request spans multiple storage chunks, the source function must assemble data from several chunks into a single output array. This happens when rechunkit's read buffer is larger than one source chunk, or — critically — when a **selection offset** misaligns the read requests with the storage chunk grid.

In this case, the source function:

1. Allocates an output array sized to the requested slice (not the full variable)
2. Iterates through all overlapping storage chunks via `slices_to_chunks_keys()`
3. Reads each chunk from Booklet, deserializes, and copies the relevant portion into the output array
4. Returns the assembled array

The output array size is controlled by rechunkit's memory planning, so this stays within the user's `max_mem` budget.

## Selections and Chunk Alignment

Understanding how selections interact with chunk boundaries is important context for why the source function is designed the way it is.

### How rechunkit Handles Selections

When a user rechunks a view (e.g., `var[10:40, :].rechunker().rechunk(...)`), cfdb passes both the full array shape and the selection to rechunkit:

```python
rechunkit.rechunker(
    source_func,
    shape=full_shape,             # Shape of the entire stored array
    ...
    source_chunk_shape=(20, 30),  # Storage chunk layout
    target_chunk_shape=(15, 20),  # Desired output layout
    sel=(slice(10, 40), ...),     # The view's selection
    ...
)
```

rechunkit computes the **phase** — the misalignment between the selection start and the source chunk grid (`sel.start % source_chunk_shape`). It then shifts its read plan backward by this phase so that every read request, after being shifted to absolute positions, lands on a source chunk boundary. This guarantees that a source function backed by chunk-aligned storage can serve each read without needing to assemble across chunk boundaries — even when the selection offset doesn't fall on a chunk boundary.

### Why Multi-Chunk Assembly Is Still Needed

rechunkit's selection alignment solves the offset problem in **0-based index space**. However, cfdb has a second source of misalignment that rechunkit can't address: **coordinate origins**.

When data is prepended to a coordinate, its origin shifts to a negative value (see [Chunking & Storage](chunking-storage.md#chunk-alignment-and-origins)). Storage chunk keys are floor-aligned to `chunk_shape` in storage coordinate space, which creates a mismatch between the 0-based index space that rechunkit operates in and the actual storage chunk grid.

For example, with `origin = -50` and `chunk_shape = 20`:

```
Storage chunks (storage space):  [-60───-40)[-40───-20)[-20────0)[0─────20)
                                      ↑
                                  origin = -50 (mid-chunk)

User indices (0-based):          [0──────10)[10─────30)[30─────50)[50────70)
                                  ↑
                                  index 0 = storage position -50
```

User index 0 maps to storage position -50, which is at offset 10 within the storage chunk `[-60, -40)`. A rechunkit read at `[0, 20)` (aligned in 0-based space) maps to storage `[-50, -30)`, spanning two storage chunks. rechunkit can't fix this because it doesn't know about coordinate origins — it only sees a 0-based array of the declared shape.

This is why cfdb's source function uses `slices_to_chunks_keys()` to handle arbitrary multi-chunk reads, with a single-chunk fast path for the common case. When origins are zero (no prepend/append), rechunkit's alignment guarantee means every read maps to exactly one storage chunk and the fast path is always taken. When origins are non-zero, the multi-chunk assembly handles the storage-level misalignment transparently.

## Encoded vs Decoded Paths

cfdb's type system supports **encoded dtypes** where data is stored in a compact form (e.g., float values scaled and stored as integers). The rechunker has two separate code paths for this:

### Decoded Path

Used when `dtype.dtype_encoded is None` (no encoding). The source function deserializes chunks with `dtype.loads()`, which returns the final user-facing numpy array. rechunkit operates directly on decoded data.

```
Booklet bytes → decompress → dtype.loads() → decoded array
                                                    ↓
                                              rechunkit buffer
                                                    ↓
                                              yield to user
```

### Encoded Path

Used when `dtype.dtype_encoded is not None`. The source function deserializes chunks with `dtype.from_bytes()`, which returns the **encoded** (compact) array. rechunkit operates on the smaller encoded representation, which reduces buffer memory. After rechunkit yields each target chunk, cfdb decodes it before yielding to the user.

```
Booklet bytes → decompress → dtype.from_bytes() → encoded array
                                                        ↓
                                                  rechunkit buffer
                                                  (smaller, int16 etc.)
                                                        ↓
                                                  dtype.decode()
                                                        ↓
                                                  yield decoded to user
```

The encoded path matters for performance: if a float32 variable is stored as int16 with a scale factor, the rechunkit buffer is half the size, and the `max_mem` budget goes twice as far.

## Multi-Variable Synchronization

`DatasetRechunker` coordinates rechunking across multiple variables that share the same coordinates. This enables computations like `speed = sqrt(u**2 + v**2)` without loading full arrays.

### How Synchronization Works

The design relies on rechunkit's **deterministic iteration order**. Given identical `shape`, `source_chunk_shape`, `target_chunk_shape`, and `max_mem`, rechunkit always yields chunks in the same order. `DatasetRechunker` exploits this:

1. Creates one `Rechunker` generator per variable
2. Divides the total `max_mem` equally across variables
3. Zips the generators together — each step yields the same target chunk position for all variables

```python
var_gens = [ds[name].rechunker().rechunk(shape, per_var_mem) for name in data_vars]

for items in zip(*var_gens):
    write_slices = items[0][0]   # Same for all variables
    var_data = {name: item[1] for name, item in zip(data_vars, items)}
    yield target_chunk_dict, var_data
```

### Requirements

All variables in a `DatasetRechunker` must share identical coordinates and shapes. Storage chunk shapes can differ — each variable's `Rechunker` handles its own storage layout independently.

## Yield Lifetime Contract

Arrays yielded by the rechunking generators (`Rechunker.rechunk`, `iter_chunks` with a
`chunk_shape`, `groupby`, `DatasetRechunker.rechunk`, and the dataset-level wrappers)
may be **views into rechunkit's internal buffer**, which is reused as iteration
advances. Two rules for consumers:

1. **Consume or `.copy()` each yielded array before advancing the generator.**
   Collecting yields (e.g. `list(var.groupby(...))`) leaves earlier arrays pointing
   at overwritten buffer memory — silently wrong data. Whether a given chunk is a
   view or a copy depends on plan internals (and on whether the dtype is encoded),
   so never rely on it.
2. **Treat yielded arrays as read-only.** Storage-chunk iteration yields slices of
   freshly deserialized chunks, but mutating yields in place is not a supported way
   to write data — use `set()`/`__setitem__`.

The supported pattern processes each chunk inside the loop body (writing it to a
variable, aggregating it, or copying it).

## Memory Model

Data flows through several stages during rechunking. Understanding where memory is allocated helps with tuning `max_mem`:

| Stage | What lives in memory | Size |
|-------|---------------------|------|
| Source function call | One deserialized storage chunk (or assembled multi-chunk array) | Up to source read chunk size |
| rechunkit buffer | Intermediate array for converting layouts | Controlled by `max_mem` |
| Yielded chunk | Target chunk array passed to user | One target chunk |

The user's `max_mem` controls the rechunkit buffer, which is the largest allocation. The source function's temporary arrays are bounded by rechunkit's read plan — it never requests more data than fits in the buffer.

For `DatasetRechunker`, the effective per-variable budget is `max_mem / n_variables`. With many variables, this can become small enough to force extra read passes. Monitor with `calc_n_reads_rechunker()`.

## Chunk Key Mapping

The source function maps requested slices to Booklet keys through `indexers.slices_to_chunks_keys()`:

```
Requested slices: (slice(100, 150), slice(200, 300))
Chunk shape:      (100, 100)

Step 1 — Find overlapping chunks via rechunkit.chunk_range():
  → (slice(100, 150), slice(200, 300))  [one chunk in this case]

Step 2 — Compute chunk origin (align to grid):
  → starts_chunk = (100, 200)

Step 3 — Build Booklet key:
  → "temperature!100,200"

Step 4 — Compute slice within the chunk:
  → source_chunk = (slice(0, 50), slice(0, 100))

Step 5 — Compute position in the output array:
  → target_chunk = (slice(0, 50), slice(0, 100))
```

When a request spans multiple chunks, steps 2-5 repeat for each overlapping chunk, and the results are assembled into the output array.

## Design Decisions

**Why not just use `get_chunk()` as the source function?**

`get_chunk()` reads from exactly one storage chunk — it finds which chunk a selection starts in, clips to that chunk's boundary, and returns. rechunkit guarantees that read requests are aligned to `source_chunk_shape` in 0-based index space, which would make `get_chunk()` sufficient for datasets with zero coordinate origins. However, when coordinates have been prepended (non-zero origins), the storage chunk grid is offset from the 0-based grid. A 0-based-aligned read can still span two storage chunks in this case. The multi-chunk source function handles this transparently, with a fast path that avoids extra allocations when the read does map to a single chunk.

**Why pass the full shape to rechunkit instead of the view shape?**

rechunkit needs the full shape to correctly plan reads relative to the storage chunk grid. The selection is passed separately so rechunkit can compute the offset. If we passed the view shape, rechunkit would assume chunks start at position 0 within the view, which would not match the actual storage layout.

**Why divide max_mem equally across variables?**

This is the simplest approach that guarantees the total memory stays within budget. A more sophisticated approach could weight by dtype size, but the equal split is predictable and avoids edge cases. Users who need fine control can use individual `Rechunker` instances.
