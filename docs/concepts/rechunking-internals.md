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

When a read request spans multiple storage chunks, the source function assembles data from several chunks into a single output array:

1. Allocates an output array sized to the requested slice (not the full variable)
2. Iterates through all overlapping storage chunks via `slices_to_chunks_keys()`
3. Reads each chunk from Booklet, deserializes, and copies the relevant portion into the output array
4. Returns the assembled array

Since the **storage-space declaration** (next section) every normal read maps to exactly one storage chunk, so this assembly loop is a **defensive fallback** — kept (and deliberately exercised by a test) so a future invariant drift degrades gracefully instead of corrupting reads. The output array size follows rechunkit's read plan.

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

### Coordinate Origins: The Storage-Space Declaration

rechunkit's selection alignment solves the offset problem for **selection starts**. cfdb has a second source of misalignment: **coordinate origins**. When data is prepended to a coordinate, its origin shifts negative (see [Chunking & Storage](chunking-storage.md#chunk-alignment-and-origins)), and storage chunk keys are floor-aligned to `chunk_shape` in storage coordinate space — so the storage grid sits off the 0-based index grid.

cfdb resolves this by folding the origin into rechunkit's *existing* phase machinery: the `Rechunker` declares the array to rechunkit in a **declared space** shifted so that storage-chunk boundaries land on multiples of `chunk_shape`:

```
d          = origin % chunk_shape          # per dim, 0 <= d < chunk (Python mod)
declared   = user index + d                # what rechunkit sees
storage    = declared - shift              # shift = d - origin, a multiple of chunk
decl_shape = full_shape + d
sel_decl   = (user sel, or the full range) + d
```

With `origin = -50` and `chunk_shape = 20`: `d = 10`, so user index 0 becomes declared index 10 — exactly the offset of storage position −50 within its chunk `[-60, -40)`. The selection's phase against the declared grid is then the true misalignment against the **storage** grid, and rechunkit's phase-shifted reads land on real storage-chunk boundaries: **every read maps to exactly one storage chunk**, for any origins. The source function converts declared reads to storage space by subtracting the constant `shift` and feeds `slices_to_chunks_keys()` directly.

Two properties make this transparent to everything downstream: `shift` is always a multiple of `chunk_shape` (key alignment is preserved), and rechunkit's yields are selection-relative (the shifted selection contains the same elements, so consumers — `iter_chunks`, `groupby`, `DatasetRechunker` — see byte-identical output; verified against the pre-change implementation). For aligned variables (`d == 0` in every dim) the declaration degenerates to the plain 0-based call.

Before this declaration existed, every post-prepend read spanned 2 storage chunks per misaligned dim and re-decompressed the neighbours (~3.4x measured on a 2-dim prepend); it is now exactly 1.0x.

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
| Source function call | One deserialized storage chunk | One storage chunk |
| rechunkit bulk buffer | Intermediate array for converting layouts | Counted against `max_mem` |
| Canonical-order pending | Copies of not-yet-due target chunks | Counted against `max_mem` |
| Batched-path buffers | One dedicated array per batched target chunk | Counted against `max_mem` |
| Yielded chunk | Target chunk array passed to user | One target chunk |

`max_mem` is an honest **total** over the bulk buffer, the canonical-order pending copies, and the batched-single-path buffers, with two documented exceptions:

- **Irreducible floors** — you cannot read less than one storage chunk or materialize less than one target chunk. The bulk-path floor is per-dim `max(source_chunk, target_chunk)`; the batched-path floor is one target chunk + one source chunk. When a floor exceeds `max_mem`, the allocation is the floor.
- **Wide-array pending residual** — when a single source chunk spans multiple target-chunk rows on a wide array (tall source chunks rechunked to much flatter targets), one target-row band `(ceil(src0/tgt0) − 1) × row_width × itemsize` is physically unavoidable without multiplying reads; cfdb holds the band. Peak memory for that operation is ~one band regardless of `max_mem`.

For `DatasetRechunker`, the effective per-variable budget is `max_mem / n_variables`. With many variables, this can become small enough to force extra read passes. Monitor with `calc_n_reads_rechunker()`.

For `map()` with multiprocessing: `max_mem` is **per call** — N workers peak at ~N × the budget. The batched single path keeps each worker close to its budget, which removes the slack the old over-allocation accidentally provided in some shapes; size `max_mem` accordingly.

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
