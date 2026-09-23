# cfdb Benchmark Results

Comparison of cfdb vs zarr (v3.1.5) for chunked N-dimensional array storage.
Both use zstd level 1 compression and raw float32 data.

## Results: cfdb vs zarr

### Small (100x100x49, chunks 20x20x49)

| Operation | cfdb | zarr | cfdb speedup |
|---|---|---|---|
| write_full | 10.7ms | 19.3ms | **1.8x** |
| write_append | 2.5ms | 10.8ms | **4.3x** |
| read_full | 1.6ms | 5.8ms | **3.6x** |
| read_slice_aligned | 364us | 1.1ms | **3.0x** |
| read_slice_unaligned | 488us | 1.6ms | **3.3x** |
| iterate_chunks | 1.8ms | 10.5ms | **5.8x** |
| iterate_rechunked | 1.8ms | 35.3ms | **19.6x** |
| groupby_7day | 2.3ms | 35.8ms | **15.6x** |
| file_size | 2.5MB | 1.7MB | 0.7x |

### Medium (500x500x98, chunks 50x50x14)

| Operation | cfdb | zarr | cfdb speedup |
|---|---|---|---|
| write_full | 144ms | 270ms | **1.9x** |
| write_append | 18ms | 34ms | **1.9x** |
| read_full | 110ms | 163ms | **1.5x** |
| read_slice_aligned | 424us | 1.3ms | **3.1x** |
| read_slice_unaligned | 678us | 1.5ms | **2.2x** |
| iterate_chunks | 62ms | 265ms | **4.3x** |
| iterate_rechunked | 72ms | 268ms | **3.7x** |
| groupby_7day | 84ms | 277ms | **3.3x** |
| file_size | 84.7MB | 83.8MB | ~equal |

### Large (1000x1000x364, chunks 100x100x28)

| Operation | cfdb | zarr | cfdb speedup |
|---|---|---|---|
| write_full | 2.23s | 1.07s | 0.5x |
| write_append | 159ms | 71ms | 0.4x |
| read_full | 1.52s | 933ms | 0.6x |
| read_slice_aligned | 1.4ms | 2.6ms | **1.9x** |
| read_slice_unaligned | 3.4ms | 2.6ms | 0.8x |
| iterate_chunks | 868ms | 1.29s | **1.5x** |
| iterate_rechunked | 988ms | 1.39s | **1.4x** |
| groupby_7day | 1.22s | 1.38s | **1.1x** |
| file_size | 1.22GB | 1.22GB | equal |

### Key Observations

- cfdb is **faster for all read/iterate/groupby operations** across every tier
- The advantage is largest at small/medium scale (3-20x) where zarr's per-chunk filesystem overhead dominates
- At large scale, cfdb still wins on iteration/groupby (1.1-1.5x) but zarr wins on bulk writes and full reads — zarr's directory-per-chunk layout allows the OS to write/read many small files efficiently
- File sizes are equal when both use raw float32

## Investigation: Rechunking vs Raw Reads

### The Question

When iterating with a rechunked shape (e.g. full spatial, 7 time steps on data stored in (100,100,28) chunks), why was zarr initially appearing faster than cfdb despite re-reading chunks multiple times?

### OS Page Cache Effect

Zarr stores each chunk as a separate file. When `temp[:, :, k:k+7]` reads a chunk file, the OS kernel caches it in RAM. Subsequent reads of the same file (needed because 7-day windows overlap within 28-day storage chunks) are served from memory, not disk.

For the large tier:
- **zarr**: 6400 chunk file reads, but ~5100 are page-cache hits → effectively free
- **cfdb rechunker** (with sufficient max_mem): 1300 reads, each chunk once

### max_mem Impact

The rechunker needs a buffer large enough to hold the ideal read chunk shape. With insufficient memory, it must make multiple passes, re-reading chunks:

| max_mem | Source read shape | Chunk reads | Rechunked time |
|---|---|---|---|
| 128MB (default) | (700, 600, 150) | 6400 | ~10s |
| 512MB | (900, 800, 180) | 6400 | ~4.5s |
| 1GB | (1000, 1000, 210) | 1300 | ~1.7s |
| 2GB | (1000, 1000, 210) | 1300 | ~1.2s |

At 1GB+, the rechunker achieves the ideal read shape and reads each chunk exactly once.

### Encoded Dtype Overhead

cfdb supports int-encoded float32 (e.g. precision=1, range 0-10000), reducing file size by ~25%. But the decode step (`astype(uint32→float32)` + scale/offset) adds cost, especially on large output arrays:

| Metric | Raw float32 | Encoded uint32 |
|---|---|---|
| File size (large) | 1.22GB | 929MB |
| Storage iteration | 825ms | 707ms |
| Rechunked iteration | 892ms | 1.09s |
| Decode cost on 53 output chunks | 0 | ~0.3s |

The file size savings come with a ~0.2-0.3s decode penalty at large scale.

## Component Profile (Large Tier)

### Raw float32

| Component | Time |
|---|---|
| Booklet lookup | 465ms |
| zstd decompress | 1.09s |
| loads (deserialize) | 408ms |
| rechunkit buffer mgmt | 111ms |

### Encoded uint32 (precision=1)

| Component | Time |
|---|---|
| Booklet lookup | 348ms |
| zstd decompress | 861ms |
| loads (deserialize+decode) | 514ms |
| decode (small storage chunks) | 119ms |
| decode (large output chunks) | 1.46s |
| rechunkit buffer mgmt | 107ms |

Run the profiler: `python -m benchmarks.profile_cfdb --tier large [--encoded]`

## Per-chunk costs vs chunk size, real data (2026-09-23, revised after review round `cfdb-profile-1`)

`profile_cfdb.py --source PATH --var NAME` rewrites one real variable at a ladder of chunk shapes
(halving the largest axis, down to ≥ 1000 elements; the smallest that fits here is 2040) and times
each step of the read and write paths per chunk. Source: the 557 MB WRF d01 dataset, zstd-1, warm
page cache, single thread, min of 3. Full tables (per component, per real and per stored element,
both write patterns, per-file close): `results/profile-ladder-2026-09-23/{var}.{txt,json}`. The review
record, with each finding's verdict: `results/review-cfdb-profile-1.md`.

**Read (`iter_chunks`), ns per element** — per REAL element / per STORED element (stored includes
the blank edge padding of shapes that don't divide the 324 × 277 grid):

| chunk elements | stored/real (u16) | air_temperature (u16) | surface_pressure (u32) | ivt (f32) |
|---|---|---|---|---|
| 2 153 952 (stored shape) | 1.00 | 3.02 / 3.02 | 5.19 / 4.82 | 4.46 / 4.13 |
| 1 076 976 | 1.00 | 2.18 / 2.18 | 4.17 / 3.87 | 3.24 / 3.00 |
| 536 544 | 1.50 | 1.84 / 1.23 | 3.40 / 2.11 | 2.27 / 1.41 |
| 268 272 | 1.50 | 1.71 / 1.15 | 3.38 / 2.10 | 2.23 / 1.38 |
| 134 136 | 1.25 | 1.58 / 1.27 | 3.24 / 2.42 | 2.22 / 1.65 |
| 66 240 | 1.38 | 1.89 / 1.36 | 3.55 / 2.38 | 2.56 / 1.72 |
| 32 640 | 1.23 | 1.88 / 1.53 | 3.67 / 2.77 | 2.49 / 1.88 |
| 16 320 | 1.16 | 2.16 / 1.86 | 4.05 / 3.24 | 2.62 / 2.10 |
| 8 160 | 1.10 | 3.25 / 2.96 | 4.60 / 3.90 | 3.10 / 2.63 |
| 4 080 | 1.10 | 4.42 / 4.04 | 5.98 / 5.06 | 4.02 / 3.41 |
| 2 040 | 1.06 | 6.68 / 6.28 | 8.47 / 7.39 | 5.82 / 5.07 |

**Write, ns per element** — block writes (one `set()` per 4.3 MB source block) / one `set()` per
chunk (the loop the cfdb skill recommends), both per REAL element; per-stored-element figures are in
the logs:

| chunk elements | air_temperature (u16) | surface_pressure (u32) | ivt (f32) |
|---|---|---|---|
| 2 153 952 (stored shape) | 4.56 / 4.51 | 10.72 / 10.34 | 5.32 / 5.30 |
| 1 076 976 | 4.61 / 4.35 | 10.83 / 9.88 | 5.68 / 4.93 |
| 536 544 | 4.78 / 4.48 | 10.60 / 10.37 | 5.46 / 5.37 |
| 268 272 | 5.30 / 5.15 | 10.72 / 10.64 | 5.46 / 5.39 |
| 134 136 | 5.30 / 5.28 | 11.13 / 11.23 | 5.53 / 5.39 |
| 66 240 | 6.23 / 6.84 | 11.77 / 12.68 | 5.71 / 5.89 |
| 32 640 | 5.92 / 7.10 | 12.74 / 14.15 | 6.00 / 6.68 |
| 16 320 | 6.96 / 9.37 | 14.11 / 16.76 | 6.53 / 8.11 |
| 8 160 | 9.35 / 14.95 | 16.78 / 21.80 | 7.61 / 10.65 |
| 4 080 | 11.02 / 21.46 | 17.16 / 29.22 | 9.56 / 16.12 |
| 2 040 | 14.32 / 34.93 | 20.66 / 44.05 | 13.05 / 26.22 |

- **Reads are best from ~130 K to ~540 K elements** and within ~20 % of that from ~540 K down to
  ~30 K (per real element). They rise 18–37 % by 16 K and 2.6–4.2× by 2 K. The stored shape (4.3 MB
  u16, 8.6 MB u32/f32) is 1.6–2.0× the best. Per stored element the best is 270–540 K.
- **The large-chunk read penalty is mostly memory allocation, not size.** Every chunk decodes into
  fresh multi-MB buffers, which page-fault under the default glibc allocator: a decode with a reused
  buffer costs the same per element up to 2.15 M elements; with fresh buffers 2.3× more. Allocator
  tuning (`MALLOC_TRIM_THRESHOLD_`, `MALLOC_MMAP_THRESHOLD_`, `MALLOC_TOP_PAD_`) made stored-shape
  reads 15–25 % faster here; a ~1.2–1.5× penalty remains, mostly decompression. The component
  split at ≥ 1 M elements is not reliable for the same reason.
- **Block writes change little down to ~130 K elements**, then rise: 1.2–1.5× by 16 K, 1.9–3.1× by
  2 K. The largest chunks look cheapest per real element only because they have no padding on this
  grid; per stored element the best is 270–540 K (see logs).
- **One `set()` per chunk costs a fixed ~20–40 µs extra per call**, so that pattern diverges from
  block writes below ~64 K elements: 1.2–1.35× by 16 K, 2–2.5× by 2 K. Most of it is `set()`'s own
  per-call overhead — each call builds 8 cfdb objects because `get_coord_origins` tests
  `hasattr(self, 'coords')` on a property — so it is a fixable cfdb cost, not a property of chunk size.
- **Per-chunk cost at 2 K elements.** Read 10–15 µs: key generation ~2.5, Booklet get ~1.8,
  decompress 5–7, from_bytes ~0.8, decode ~3.4 (packed only). Block write 23–37 µs: key generation
  ~2.5, Booklet get of the missing key ~1.9, encode ~1.4 (paid once per `set()` call, shared across
  chunks), blank copy + slice assignment ~2.5, compress 8–22, Booklet set 5–7. zstd's marginal cost
  per element at 2–4 K is ~1.8× its rate at 66 K (decompress), so its small-chunk cost is not just a
  fixed overhead per call.
- **Per file:** Booklet sync 0.2–4 ms plus the file close, 25–50 ms for these 30–64 MB files and
  ~3 ms for a 6 MB one (filesystem- and state-dependent); the same at every chunk size.

**Self-check.** Components are timed step for step as the pipelines run them, and their sum is
compared with the full pipeline. Below 1 M elements they agree within −7 % to +8 %. The check is
coarse: mutation tests in review put its detection floor at ~5–10 µs/chunk, and an earlier version
of this profiler passed it while two ~7 µs errors cancelled. After the fix, re-applying that error
moves the 2–8 K write residual from +1.5 µs to −5 to −6 µs (−7 to −21 %), so it is detected there;
at 16 K and above it is not.

**Not measured here:** cold (disk) reads, since dropping the page cache needs root; codecs other
than zstd-1. Remote (EDataset) reads and station (ts_ortho) data are covered in the two sections
below.

## Remote (EDataset) reads vs chunk size (2026-09-23)

Measured over HTTP on two public datasets from one home connection in NZ (B2 via
`b2.tethys-ts.xyz` / `b2.envlib.xyz`), at one time of day. Scripts: `results/remote-2026-09-23/`.

**How ebooklet fetches.** An ungrouped remote stores one object per chunk: one GET per chunk. A
grouped remote (`num_groups`) hashes chunks into group objects and issues one ranged GET per group
touched, spanning from the first to the last needed chunk in that group; GETs run on 10 threads.

**Request cost.** ~0.4 s per GET serially for objects not already cached at the CDN (a repeatedly
fetched object answered in 0.155 s). With 10 threads: 15–17 GETs/s whatever the size, and 31 MB/s
aggregate for 2.7 MB objects but only 3.9 MB/s for 256 KB requests. Below ~1.5–2 MB per GET,
remote reads are request-bound.

**Ungrouped** (`ecmwf-forecasts/ifs/nz_0p25.cfdb`, 3.6–7.2 MB raw chunks, ~2.5 MB compressed): a
full-variable read of 15 chunks (37–39 MB) took 2.4–4.1 s from an empty cache (9–15 MB/s), against
0.08–0.11 s from the local cache. Every chunk is a request, so smaller chunks multiply the request
count directly: on an ungrouped remote, chunks below ~2 MB compressed lose throughput steeply.

**Grouped** (`cci-sst-l4-v3-nz`, 149 groups, chunks (120, 60, 60) = 864 KB raw, ~330 KB compressed).
ebooklet's fetch plan, computed exactly from the remote index for `temperature`:

| selection | chunks | GETs | needed MB | downloaded MB |
|---|---|---|---|---|
| one day, full grid | 459 | 143 | 150 | 150 (1.0×) |
| one point, all 45 years | 139 | 96 | 67 | 1 617 (**24×**) |
| 300 × 300-cell region, one year | 100 | 75 | 47 | 74 (1.6×) |
| full grid, one year | 1 836 | 149 | 607 | 607 (1.0×) |
| whole variable | 63 801 | 149 | 19 966 | 19 969 (1.0×) |

The region read, timed from an empty cache: 78 GETs (75 planned + 3 other), 74.0 MB downloaded as
planned, in 5.4 s (13.7 MB/s); 0.07 s from the local cache.

- **Grouping caps the request count at `num_groups`**, so on a grouped remote smaller chunks do not
  multiply requests for large reads. For small reads, GETs ≈ chunks touched, up to that cap.
- **What costs on a grouped remote is bytes, and chunk SHAPE decides the bytes.** Reading one day
  downloads 150 MB because each chunk is 120 days deep (120× read amplification); a point series
  touches whole 60 × 60 spatial chunks. Chunk size mostly matters through this amplification.
- **Sparse selections over-fetch, by design.** A GET spans every chunk packed between the first and
  last needed chunk in its group, so a single-point time series downloads 24× what it needs. ebooklet
  chose one span per group deliberately over several smaller ranged reads: it is simpler and makes
  fewer, larger HTTP requests, which suits the per-request costs above. It is independent of chunk size.
- **Every remote read is ~50–80× slower than the same read from the local cache**, which dwarfs the
  µs-level local differences between chunk sizes.

**Not measured:** other networks or times of day; datasets rewritten at other chunk sizes (neither
remote can be rewritten without write access), so the chunk-size effect is inferred from the
request and byte costs above. For station data it is projected from real compressed sizes in the
next section.

## Station (ts_ortho) data: local and remote (2026-09-23)

Source: ECan hourly streamflow, public and ungrouped (`ecan-env-monitoring/envlib/ecan-env/ecan-streamflow`):
141 stations × 183 227 hours (2005–2026), packed uint32, stored as (1 station × 25 000 hours) =
100 KB raw, 16–41 KB compressed. The whole dataset was downloaded (32.7 MB compressed) to profile it
locally. Logs: `results/profile-ladder-2026-09-23/streamflow_{per_station,multi_station}.txt`
(first 43 stations; ladders start at (1, 24 576) and (43, 24 576) and write in blocks of 49 152 hours
that every rung divides, so no write is a read-modify-write). Remote projection script:
`results/remote-2026-09-23/ts_ortho_shapes.py`.

**Local: the same curve as the grids.** Read, ns per real element (multi-station chunks): 2.31 at
264–528 K elements, 2.40 at 132 K, 2.64 at 33 K, 2.88 at 16 K, 3.5 at 8 K, 6.9 at 2 K, 11.0 at 1 K.
Block writes: 6.6–7.1 at 132–528 K, 9.0 at 33 K, 14.3 at 2 K. Per-station chunks give the same
per-element costs at the same chunk size (24.6 K elements: read 2.75, write 8.8). Chunk size, not
station layout, sets the local cost.

**Remote (ungrouped): chunk shape decides everything.** Measured cold, with the current layout: latest
7 days for all stations 6.4 s (141 GETs for 2.3 MB), one station's full history 1.1 s (8 GETs), the
whole dataset 40.1 s (1 072 GETs, 0.82 MB/s). Projected for other shapes from the real compressed
sizes, at the measured 22 GETs/s and 13 MB/s:

| chunk shape | median KB compressed | latest 7 days, all stations | one station, full history | whole dataset | objects re-uploaded per hourly update |
|---|---|---|---|---|---|
| (1, 25 000), current | 41 | 141 GETs, 6.6 s | 7 GETs, 0.3 s | 801 GETs, 38.9 s | 141 |
| (10, 8 760) | 129 | 15 GETs, 0.9 s | 17 GETs, 0.9 s | 264 GETs, 14.6 s | 15 |
| (141, 8 760) | 1 739 | 1 GET, 0.2 s | 21 GETs, 3.6 s | 21 GETs, 3.6 s | 1 |
| (141, 2 190) | 454 | 1 GET, 0.1 s | 84 GETs, 6.5 s | 84 GETs, 6.5 s | 1 |
| (141, 720) | 152 | 1 GET, 0.1 s | 255 GETs, 14.3 s | 255 GETs, 14.3 s | 1 |

- The projection matches the measured current layout (6.6 vs 6.4 s; 38.9 vs 40.1 s) except for
  small GET counts, where one round of request latency (~0.4 s) sets a floor the rate model misses
  (one station: 0.3 s projected, 1.1 s measured).
- **Multi-station chunks win for growing telemetry**: the latest-data query becomes one GET, and each
  update re-uploads one object instead of one per station. The cost is single-station history, which
  becomes a scan. (141, 2 190) — one quarter of all stations, ~310 K elements, in the local sweet spot
  — turns "latest, all stations" from 6.6 s into 0.1 s and the whole dataset from 39 s into 6.5 s, at
  6.5 s instead of 0.3 s for one station's full history.
- **25 % of the stored chunks are empty.** 271 of the 1 072 stored streamflow chunks hold only missing
  values (22 bytes each), most likely from writing whole station rows including years before a station
  reported. On an ungrouped remote each is still a GET: a quarter of the requests in a full read, and
  36 of the 80 in a 10-station history read, fetch nothing. The projection counts only non-empty
  chunks, so it is slightly optimistic for the current layout.

## Rechunking vs chunk size (2026-09-23)

Everything above reads or writes each stored chunk once. This section covers reading a variable
back in a DIFFERENT shape (`iter_chunks(chunk_shape=...)`, `groupby`, point and snapshot
selections), for the chunk shape cfdb would guess at byte targets of 256 KiB to 4 MiB. Files:
`results/rechunk-2026-09-23/` (`rechunk_plans.{py,txt}`, `rechunk_timing.{py,txt}`).

**Read plans on full-size shapes** (no data needed: rechunkit's `calc_n_reads_rechunker` at the
default 512 MiB budget; each read decompresses one whole chunk). Read amplification = bytes
decompressed ÷ bytes wanted:

| query | 256 KiB | 512 KiB / 1 MiB | 2 MiB | 4 MiB |
|---|---|---|---|---|
| SST (16650, 1000, 1600) u16: one point, 45 years | 293× | 732× | 2 927× | 2 906× |
| SST: whole array → daily grids | 3.2× | 3.2× | 3.2× | **6.2×** |
| SST: whole array → pixel series | 2.1× | 2.1× | 5.7× | 5.7× |
| forecast (29, 49, 14, 165, 221) u16: latest run | 5.0× | 6.5× | 7.5× | **22.4×** |
| stations (141, 183 227) u32: one station, full history | 6.3× | 6.4× | 25.4× | 25.4× |

rechunkit's guesser snaps dimensions to composite numbers, so 512 KiB and 1 MiB often give the same
shape (they do for all three datasets above).

**Measured on d01** (`air_temperature` uint16 and `surface_pressure` uint32, 168 × 324 × 277,
rewritten at each guessed shape; seconds, min of 3). The 4 MiB memory budget makes the 30–60 MB
variables 7–15× larger than memory, emulating the regime a large dataset is in by default:

| byte target | u16 elements | u16 pixel series (4 MiB budget) | u16 hourly grids (4 MiB) | u32 elements | u32 pixel series (4 MiB) | u32 hourly grids (4 MiB) |
|---|---|---|---|---|---|---|
| 256 KiB | 138 K | 0.159 | 0.063 | 55 K | 0.299 | 0.187 |
| 512 KiB | 346 K | **0.036** | **0.056** | 138 K | 0.257 | 0.156 |
| 1 MiB | 346 K | 0.034 | 0.055 | 346 K | **0.059** | 0.190 |
| 2 MiB | 864 K | 0.129 | 0.108 | 346 K | 0.059 | 0.189 |
| 4 MiB | 1.99 M | 0.980 | 1.021 | 864 K | 0.874 | 1.173 |

- **Rechunking under memory pressure is best at ~3.5·10⁵ elements per chunk for BOTH item sizes**:
  that is 512 KiB for uint16 but 1 MiB for uint32, which is why cfdb 0.10's default is an element
  target (2¹⁸ elements × item size) rather than a byte target.
- Large chunks are 5–30× slower (4 MiB); too-small chunks are also up to 4–5× slower for pixel
  series (55–138 K elements).
- With the default 512 MiB budget every chunking rechunks at about the same speed (the variable fits
  in memory), and point or single-time-step selections favour smaller chunks but take milliseconds.
- Caveat: the memory-pressure runs emulate large data 7–15× oversubscribed; the real SST case at the
  default budget is ~100×. The read plans above cover that scale for bytes read, not timing.

The tier mode above had been broken since `3c2b3b4` removed `iter_chunk_slices()`; it now uses
`iter_chunks(include_data=False)`.

## Bug Fix: groupby Period Fast Path

### Issue

`groupby({'time': '7D'})` on 365 daily time steps fell back to the slow `_groupby_period` path because the last group had 1 day instead of 7. The strict uniformity check (`len(set(group_sizes)) == 1`) rejected the rechunker fast path.

**Impact**: 6400 chunk reads instead of 1300 — a **5x read amplification**.

### Fix

Removed the uniformity check for regular periods (D, h, W, 7D, 6h, etc.). The rechunker already handles remainder chunks. Irregular periods (M, Y) still use the slice-based path.

**Before**: groupby 7D on large tier = **4.5s**
**After**: groupby 7D on large tier = **1.2s**
