# Compression Benchmarks

These measurements are why cfdb 0.10 compresses with `zstd_shuffle` by default and sizes data
variable chunks at about 2¹⁸ elements. [Chunking & Storage](chunking-storage.md) covers how both
work; this page shows the evidence behind them.

**In short:**

- Shuffling the bytes of each value before compressing makes files **~0.7× the size** of plain
  zstd on packed grids and station series (0.83× on raw float32), and makes zstd **faster in
  both directions** at the chunk sizes cfdb uses.
- Plain `lz4` is still the fastest decoder, but its files are **1.2–1.4× larger** than zstd's.
  `lz4_shuffle` shrinks them to 0.76–0.95× zstd, but decodes more slowly than plain `lz4`.
- The compression ratio barely changes from multi-MB chunks down to a few thousand elements, so
  large chunks buy almost no compression. Speed does depend on chunk size, and it peaks near the
  2¹⁸-element default.
- Other options were measured and not adopted. Some gave smaller files but slower decoding, and
  some needed a new dependency. None beat `zstd_shuffle` on size, compress speed and decompress
  speed at once. (Other compressors, compression levels and filters are in
  [What else was tried](#what-else-was-tried).)

## Datasets and method

| Dataset | Content | Stored chunks | Encoded size |
|---|---|---|---|
| WRF grid | 33 variables of 12 km regional climate model output: 27 packed to uint16, 3 to uint32, 2 uint8, 1 raw float32 | (24, 1, 324, 277) = 4.3 MB | 1.1 GB |
| ERA5 3 km grid | 4 raw float32 variables | whole array, 2.5 MB | 10 MB |
| Hourly streamflow | a `ts_ortho` dataset: 141 stations, packed uint32 | one station × 25 000 hours = 100 KB | 107 MB |

Each compressor is given the *encoded* chunk exactly as cfdb hands it over (packed integers, not the
decoded floats), and every chunk must decompress **bit-identically**. Speeds are single-threaded
throughput in uncompressed MB/s on one machine (AMD Ryzen 9 7900), taking the best of three runs.
Absolute speeds will differ on other hardware; the ratios between options are the useful part.
The figures are drawn from the committed result files by
`benchmarks/compression/plot_docs.py`.

## Size and speed at the stored chunk size

![Whole-file size against decompress and compress speed for the four cfdb options and three alternatives](../assets/benchmarks/codec-tradeoff-light.svg#only-light)
![Whole-file size against decompress and compress speed for the four cfdb options and three alternatives](../assets/benchmarks/codec-tradeoff-dark.svg#only-dark)

The WRF grid, re-compressed at its stored 4.3 MB chunks:

| `compression` | Whole file | vs `zstd` | Compress | Decompress |
|---|---|---|---|---|
| `zstd` | 565 MB | 1.00 | 740 MB/s | 1.8 GB/s |
| **`zstd_shuffle`** (default) | **401 MB** | **0.71** | **1300 MB/s** | **3.1 GB/s** |
| `lz4` | 746 MB | 1.32 | 1190 MB/s | 7.1 GB/s |
| `lz4_shuffle` | 467 MB | 0.83 | 1740 MB/s | 4.2 GB/s |

Speeds are pooled over the 23 variables that compress less than 5:1 as stored. Near-constant fields,
such as sea ice, compress thousands to one, and would otherwise dominate the averages.

**Why the shuffle helps.** A value packed into a uint16 has a smooth, predictable high byte and a
noisy low byte, and the two alternate through memory. zstd has to work through that mixture, and
gets only 1.1–1.4:1 on smooth fields. The shuffle stores all the high bytes of a chunk first, then
all the low bytes. zstd then compresses the smooth half very well and passes quickly over the
noisy half, so the files are smaller and zstd runs faster. The split and join are a few vectorized
numpy operations per chunk and need no extra dependency.

## Across datasets

![File size relative to zstd for zstd_shuffle, lz4 and lz4_shuffle on three datasets](../assets/benchmarks/datasets-light.svg#only-light)
![File size relative to zstd for zstd_shuffle, lz4 and lz4_shuffle on three datasets](../assets/benchmarks/datasets-dark.svg#only-dark)

The saving carries over to station data: the streamflow series shrinks by the same 28 %, and
compression is twice as fast. At that dataset's small 25 000-element chunks, decompression is only
1.1× faster than plain zstd. On raw float32 the saving is smaller (0.83×), because the low bytes of
full-precision floats are mostly noise. The shuffle helps lz4 least on floats: `lz4_shuffle` is
0.95× zstd there, against 0.83× on the WRF grid and 0.76× on the stations.

## Per variable

![zstd_shuffle size divided by zstd size for each WRF variable, coloured by stored type](../assets/benchmarks/per-variable-light.svg#only-light)
![zstd_shuffle size divided by zstd size for each WRF variable, coloured by stored type](../assets/benchmarks/per-variable-dark.svg#only-dark)

Almost every variable shrinks, most to 0.55–0.8× of their zstd size. Three cases are worth knowing:

- **1-byte values** (`land_use_modis`, `land_sea_mask`, stored as uint8) are unchanged, because a
  single byte has nothing to shuffle. cfdb stores 1-byte, bool, string and geometry variables
  unshuffled.
- **`moisture_flux` is 4 % larger** shuffled (11.7 → 12.2 MB), the only WRF variable that grows.
  In the ERA5 set, `pwat` (raw float32 with many exactly repeated values) is 21 % larger, while
  that file as a whole still shrinks to 0.83×. Unpacked floats with repeated values are the case
  where plain `zstd` can win.
- **The largest ratios are on near-constant fields** (`sea_ice`, `snow_depth`), which take up
  kilobytes either way, so they barely change the file size. The sizes printed next to the bars
  show how much each variable actually contributes.

**Unpacked float64 was tested separately.** Full-precision float64 with many exactly repeated
values came out up to 13 % larger shuffled (0.90–1.13× across the tests), though still faster to
read and write. If file size matters most for such data, create the dataset with
`compression='zstd'`. Datetimes and float32 values widened to float64 shrink a lot (0.03–0.80×).
These tests are recorded in `benchmarks/results/review-cfdb-shuffle-code-1.md`.

## Chunk size

![Whole-file size, decompress speed and compress speed against elements per chunk for five pipelines](../assets/benchmarks/block-size-light.svg#only-light)
![Whole-file size, decompress speed and compress speed against elements per chunk for five pipelines](../assets/benchmarks/block-size-dark.svg#only-dark)

The same WRF variables were re-compressed at block sizes from the stored 2.15 M elements down to
120. Each speed point is a single timed call, so the curves are a little noisier than the table
above.

- **Size barely changes from 2 M down to ~4 K elements** (`zstd_shuffle`: 400 → 407 MB), then
  grows. Large chunks buy almost nothing in compression.
- **Speed is highest between ~65 K and ~540 K elements** for every option, and cfdb's default
  (shaded) sits in that range. Every call to the compressor has a fixed overhead of a few
  microseconds, which dominates below a few thousand elements.
- **The shuffle's speed advantage needs chunks of more than a few thousand elements.** At ~4 K
  elements and below, `zstd_shuffle` decompresses slower than plain `zstd`; at ~1 K and below it
  also compresses slower. The size advantage holds at every block size. On the streamflow data, the
  decompression crossover is higher, between 6 K and 12 K elements.

This is one reason cfdb recommends at least ~32 K elements per chunk. What chunk size does to cfdb's
full read path (key lookups, decoding, rechunking, remote requests) is measured separately in
`benchmarks/RESULTS.md` and summarized in [Chunking & Storage](chunking-storage.md#automatic-chunk-shape).

## What else was tried

Other options were measured on the WRF grid before choosing the default. The pcodec, blosc2, zfp and
fpzip rows come from an earlier exploratory run (recorded in the compression README); sizes are
relative to the 565 MB `zstd` file above.

| Option | Result | Why not adopted |
|---|---|---|
| zstd level 3 (with the shuffle) | 1.5 % smaller | 1.3–1.4× slower compression; level 1 stays the default |
| shuffle + a row-difference (y-delta) filter, then zstd | 0.61× zstd | decodes at less than half the speed of `zstd_shuffle`; only works for grids |
| blosc1 (C shuffle + zstd) | 0.76× zstd, decodes a bit faster | a new dependency; 7 % larger than `zstd_shuffle` at large chunks |
| pcodec | 0.55× zstd | ~5× slower to compress, ~2.4× slower to decompress; numeric types only |
| blosc2 (shuffle + bytedelta + zstd) | 0.63× zstd | a large dependency; decodes slower than `zstd_shuffle` |
| zfp (lossless), fpzip | 0.88× zstd or worse | much slower in both directions |
| compressing the decoded floats instead of packed integers | larger on every variable tested | keep the integer packing |

Reviewers also tried other zstd settings (match lengths, window sizes, long-distance matching,
negative levels), per-byte-plane levels and bit-level shuffling. None beat `zstd_shuffle` on size,
compress speed and decompress speed at once.

## Caveats

- One machine, one set of library versions. zstd 1.5.6 and 1.5.7 differ by ~2 % at level 1.
- The WRF grid dominates the evidence. The ERA5 grid is small (10 MB), and the streamflow set is a
  single variable.
- `lz4_shuffle` was measured after the benchmark review, so no reviewer has checked those numbers.
- Bool, datetime and int64 variables round-trip correctly in the tests, but their speed was not
  benchmarked.

## Reproducing

The tools run on any cfdb file. From a clone of the repository:

```bash
# compare compressors on a dataset
uv run python -m benchmarks.compression.codec_bench PATH.cfdb --output-json out.json

# compression ratio and speed vs block size
uv run python -m benchmarks.compression.chunk_size_sweep PATH.cfdb --output-json sweep.json

# redraw the figures on this page
uv run --with matplotlib python -m benchmarks.compression.plot_docs
```

Full tables, per-variable results and the review records are in
[`benchmarks/compression/README.md`](https://github.com/mullenkamp/cfdb/blob/main/benchmarks/compression/README.md).
