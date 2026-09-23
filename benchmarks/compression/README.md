# Chunk compression benchmarks

> **Outcome:** implemented in cfdb 0.10 as the `zstd_shuffle` compression (the new default) and
> `lz4_shuffle`; see `docs/changelog.md` and `plans/shuffle-and-chunk-default-plan-2026-09-23.md`.

Tools for measuring how alternative chunk codecs and chunk sizes behave on **real cfdb
datasets**, plus the results that motivated them. Assessment only — nothing here changes what
cfdb writes.

Every codec is fed the *encoded* chunk array exactly as cfdb hands it to its compressor (packed
uint16/uint32, raw float32, ...), and every chunk passes `gate.RoundTripGate`: **bit-identical**
(`tobytes()` equality, so NaN payloads count), input not mutated by compress, C-contiguous output,
and the previous chunk's output re-checked so a decompressor that reuses a buffer is caught.
`test_gate.py` holds one mutant per criterion (`uv run pytest benchmarks/compression/test_gate.py`).
Throughput is raw (uncompressed) MB/s, single thread, min of `--reps` after an untimed warm-up.
Set `OMP_NUM_THREADS=1` when optional OpenMP-built codecs are installed.

## Run on your own dataset

```bash
# codec comparison (default: zstd-1, shuffle+zstd-1, shuffle+ydelta+zstd-1)
uv run python -m benchmarks.compression.codec_bench PATH.cfdb --output-json out.json
uv run python -m benchmarks.compression.codec_bench PATH.cfdb --codecs all --max-bytes-per-var 50e6

# compression ratio vs block size, per variable, with a whole-file projection
uv run python -m benchmarks.compression.chunk_size_sweep PATH.cfdb --output-json sweep.json
uv run python -m benchmarks.compression.chunk_size_sweep PATH.cfdb --shrink-axis 0   # shrink ONE axis: shape vs size

# decompress / compress / whole-file tables (markdown, identical columns) from a sweep JSON
uv run python -m benchmarks.compression.sweep_tables sweep.json --elements 2153952,268272,66240,4080
```

The core codecs need only cfdb's own dependencies. Optional codecs register themselves when
importable — run with `uv run --with pcodec --with blosc --with blosc2 --with zfpy ...`:

**Table 1 — Optional codecs** (registered only when the package imports)

| codec | package | note |
|---|---|---|
| `blosc1 shuffle zstd-1`, `blosc1 bitshuffle zstd-1` | `blosc` | c-blosc1, zero-dependency wheel |
| `blosc2 shuffle zstd-1`, `blosc2 shuffle+bytedelta zstd-1`, `blosc2 shuffle lz4` | `blosc2` | 22 MB wheel + httpx/pydantic/rich/numexpr |
| `pcodec L4`, `pcodec L8` | `pcodec` | numeric-only (no bool/datetime/str), 1.5 MB, numpy-only |
| `zfp lossless` | `zfpy` | lossless mode; built for lossy floats |

`codecs.py` is the registry; add a `(compress, decompress)` pair there to test something new.
`chunks.py` streams stored chunks through the private Booklet handle (`ds._blt`) in numeric
chunk order, so the actual stored size is available next to the array; it never holds more than
one chunk. String/geometry variables are skipped with a notice.

## Findings (2026-09-22, revised after review round `cfdb-compression-1`)

Datasets: `wrf_v50_12km_d01` (557 MB, 33 vars, chunks (24,1,324,277) = 4.3 MB; 27 packed uint16,
3 packed uint32, 2 uint8, 1 raw float32) and `gabriele_3km_era5` (4 vars, raw float32, whole-array
chunks of 2.5 MB — thin support, 10 MB of data). Ryzen 9 7900, single thread. Full tables in
`results/`; the review record with every finding's verdict is `results/review-cfdb-compression-1.md`.

**1. The lever is a byte-shuffle filter in front of zstd, not a different entropy coder.**
Packed uint16 interleaves a near-random low byte with a smooth high byte; zstd-1 gets only
1.1–1.4x on the smooth fields. Splitting the bytes into planes first (`codecs.byte_planes`, ~10
lines of numpy, no new dependency) fixes that. **At the 4.3 MB chunks of this dataset** it wins
on all three axes:

**Table 2 — Codec comparison at 4.3 MB chunks** (d01, whole file, single thread)

Uncompressed, the data is **~1.1 GB as the codec sees it** (1116 MB of packed uint16/uint32/uint8
and raw float32), or ~1.9 GB once decoded to float32 in memory. Compression ratio = 1116 MB ÷ file.

| pipeline | file | vs current | compress | decompress |
|---|---|---|---|---|
| zstd-1 (current) | 556 MB | 1.00 | 740 MB/s | 1.8 GB/s |
| **shuffle + zstd-1** | **401 MB** | **0.72** | **1325 MB/s** | **3.1 GB/s** |
| shuffle + y-delta + zstd-1 | 343 MB | 0.62 | 990 MB/s | 1.3 GB/s |
| blosc1 shuffle + zstd-1 (c-blosc1, zero-dep wheel) | 427 MB | 0.77 | 1010 MB/s | 3.4 GB/s |
| lz4-1 (cfdb's other option) | 746 MB | 1.34 | 1190 MB/s | **7.1 GB/s** |
| shuffle + lz4-1 | 466 MB | 0.84 | **1745 MB/s** | 4.2 GB/s |
| pcodec L8 | 313 MB | 0.54 | 250 MB/s | 1.3 GB/s |
| blosc2 shuffle+bytedelta zstd-1 | 358 MB | 0.62 | 1030 MB/s | 2.0 GB/s |
| zstd-3 / zstd-9 (no filter) | 513 / 470 MB | 0.92 / 0.85 | 316 / 97 MB/s | 1.5 GB/s |
| zfp lossless / fpzip | 496 MB / worse | 0.84 | 170 / 140 MB/s | 0.4 / 0.15 GB/s |

**The speed win is chunk-size conditional** (review finding): the plane split/join has a fixed
per-call cost of a few microseconds, so below ~4 K elements the numpy shuffle *decompresses*
slower than plain zstd-1 and below ~2 K it *compresses* slower; the size win holds throughout.
Tables 3–5 show it.

**lz4.** Plain lz4-1 decodes fastest of everything at every block size but produces the largest
file (1.34x the current one). The shuffle helps it a lot on size (746 → 466 MB) and compress speed,
but at lz4's decode speed the plane join becomes the bottleneck, so decode drops from 7.1 to
4.2 GB/s (Table 2). Compared with shuffle+zstd-1, shuffle+lz4-1 gives a 16 % larger file for 35 %
faster compression and 37 % faster decompression. It's the choice when decode speed matters more
than size. At small blocks plain lz4-1 stays the fastest decoder (2.2 GB/s at 4 K elements), while
shuffle+lz4-1's decode falls to about plain zstd-1's (1.23 vs 1.05 GB/s) and its compression stays
2.4x faster (1.23 vs 0.51 GB/s).

**Tables 3–5 — the five pipelines vs block size.** d01, one `chunk_size_sweep` run
(`results/2026-09-22_wrf_v50_12km_d01/chunk_size_sweep_all.json`), pooled over all 33 variables.
Same columns and row order in all three; generated by `sweep_tables.py`. Column = elements per
block (×2 bytes for most variables); 2.15 M is the stored 4.3 MB chunk. Speeds here are one
timed call per block, so they differ by a few percent from Table 2's min-of-3.

**Table 3 — Decompress speed vs block size** (MB/s)

| pipeline | 2.15 M | 268 K | 66 K | 16 K | 8 K | 4 K | 2 K | 960 | 240 | 120 |
|---|---|---|---|---|---|---|---|---|---|---|
| zstd-1 | 2385 | 2426 | 2153 | 1814 | 1326 | 1050 | 756 | 455 | 153 | 79 |
| shuffle+zstd-1 | 2913 | 3426 | 2802 | 1931 | 1433 | 971 | 594 | 326 | 102 | 53 |
| blosc1 shuffle zstd-1 | 3669 | 3665 | 3269 | 2686 | 2024 | 1424 | 909 | 535 | 176 | 95 |
| lz4-1 | 6336 | 7770 | 6582 | 3731 | 3078 | 2178 | 1357 | 739 | 210 | 108 |
| shuffle+lz4-1 | 4317 | 4577 | 3950 | 2582 | 1946 | 1231 | 709 | 361 | 104 | 53 |

**Table 4 — Compress speed vs block size** (MB/s)

| pipeline | 2.15 M | 268 K | 66 K | 16 K | 8 K | 4 K | 2 K | 960 | 240 | 120 |
|---|---|---|---|---|---|---|---|---|---|---|
| zstd-1 | 920 | 969 | 819 | 808 | 556 | 511 | 440 | 329 | 145 | 88 |
| shuffle+zstd-1 | 1259 | 1571 | 1194 | 1036 | 866 | 667 | 465 | 290 | 103 | 58 |
| blosc1 shuffle zstd-1 | 1107 | 1192 | 1161 | 1114 | 913 | 703 | 494 | 313 | 111 | 66 |
| lz4-1 | 1314 | 1622 | 1549 | 1147 | 1208 | 1187 | 1138 | 982 | 488 | 299 |
| shuffle+lz4-1 | 1632 | 2302 | 2340 | 1905 | 1626 | 1233 | 849 | 519 | 175 | 93 |

**Table 5 — Whole-file size vs block size** (MB; every variable at that block size, from 1116 MB uncompressed)

| pipeline | 2.15 M | 268 K | 66 K | 16 K | 8 K | 4 K | 2 K | 960 | 240 | 120 |
|---|---|---|---|---|---|---|---|---|---|---|
| zstd-1 | 565 | 539 | 526 | 524 | 518 | 517 | 529 | 549 | 620 | 676 |
| shuffle+zstd-1 | 400 | 398 | 404 | 405 | 407 | 407 | 415 | 431 | 481 | 524 |
| blosc1 shuffle zstd-1 | 427 | 420 | 409 | 406 | 409 | 410 | 420 | 443 | 530 | 647 |
| lz4-1 | 747 | 718 | 700 | 693 | 710 | 725 | 748 | 777 | 847 | 916 |
| shuffle+lz4-1 | 467 | 451 | 452 | 465 | 465 | 469 | 475 | 485 | 547 | 620 |

Below the crossover a C-level shuffle is the better shuffled pipeline: `blosc1` (python-blosc,
c-blosc1, no dependencies) decodes faster than shuffle+zstd-1 at every block size, and faster
than shuffle+lz4-1 from ~16 K elements down (1.42 vs 1.23 GB/s at 4 K); above that shuffle+lz4-1
wins. It matches the numpy shuffle+zstd-1 on size from ~66 K down to ~2 K elements. It loses 7 % at 4.3 MB
because its internal blocking drops long-range matches (`sea_surface_temp` 1.9 vs 11.4), and
10 % at 240 elements. Numpy shuffle for grids, a C shuffle for a small-chunk world.

Same shape on the raw-float32 dataset: shuffle+zstd-1 = 0.83x, faster both ways. One variable
there (`pwat`, many exact repeats) prefers plain zstd (3.0 vs 2.5); the y-delta recovers it (2.9).

**2. zstd level 1 stays optimal.** With the shuffle in front, zstd-3 is 1.5 % smaller for 1.3–1.4x
slower compression (d01: 394.6 vs 400.6 MB at 935 vs 1325 MB/s). Level −1 behind the shuffle is
3.4 % larger for 12 % faster compression and is the fastest numpy variant at 4 K elements
(reviewer measurement). The formulation of the shuffle matters more than the level: the `numpy
transpose` form is ~3x slower to undo than the bit-op form in `codecs.py`.

**3. The delta's gain is in the LOW byte plane**, so there is no cheap partial version — and
storing the low plane raw is catastrophic on redundant variables (RH 20.9x → 2.0x; reviewer
measurement): the low plane only *looks* random on the genuinely hard fields. Delta along the
contiguous (x) axis forces a scalar `cumsum` on decode; along y it is row-minus-row and
vectorises. Along time it buys nothing on hourly data. The numpy y-delta matches blosc2's
`bytedelta` on **size** (343 vs 358 MB) but decodes 1.5x slower (1.3 vs 2.0 GB/s); it is
shape-aware and its decode cost grows with the y extent — a separate, grid-only decision.

**4. Keep the integer packing.** Every codec run on the *decoded* float32 (pcodec detects
"multiples of 0.01") lost to the packed uint16 pipeline on every variable tested (5 in the
exploration round + `relative_humidity` by a reviewer: packed+pcodec 0.067 B/elem vs
float32+pcodec 0.088).

**5. Ratio vs chunk size (the surprising one).** For the smooth packed fields the ratio is flat
from 4.3 MB down to ~2–4 K elements, then erodes (Table 5 above).

Three arms reproduced the flat curve by independent routes (random-offset tiles, flat memory
slabs, 1-D slicing). Caveats: (i) highly redundant variables (constants, masks, terrain) ARE
strongly size-dependent (snow_depth: 25 000x at 4 MB, 84x at 2 KB, 7.6x at 160 B) but are ~1 % of
the bytes; (ii) zstd-1's *rise* at small blocks (1.31 → 1.53 on air_temperature) is a parameter
effect, not a data property — zstd's level-1 table lowers `min_match` from 7 to 5 for small
sources, and `min_match=5` on the full chunk gives 1.52; (iii) the default ladder halves y and x
before time, so its small-block rows still hold 12–24 timesteps and the projection is ~5 %
optimistic at 4 K for time-redundant variables — `sea_surface_temp` is 11x with 24 timesteps and
1.9x with 1 at the same spatial extent (`--shrink-axis 0`), because its redundancy lies along
time; (iv) Booklet's fixed per-chunk overhead (key + index slot, ~40 B) is not included:
negligible at 8 KB, tens of percent at 240 B. Throughput, not ratio, is the small-chunk cost: every call pays a
roughly fixed cost (warm, air_temperature: ~1.5–2 µs zstd-1, ~2.5 µs shuffle+zstd-1, ~0.7 µs lz4-1,
of which only ~0.3 µs is Python), against ~0.55 ns/byte of real work for zstd-1 decode. That fixed
cost is ~10–30 % of the time at 16 K elements, ~25–55 % at 4 K, and dominant at ~1 K and below. cfdb's
own per-chunk costs (Booklet lookup, key formatting, the `iter_chunks` loop, dtype decode) come on
top; they are measured in `benchmarks/RESULTS.md`, section "Per-chunk costs vs chunk size".

**6. Nothing found that beats shuffle+zstd-1 on all three axes.** Three reviewers searched
(zstd `min_match`/`hash_log`/`window_log`/strategies/LDM/negative levels, per-plane level
schemes, dictionaries, lz4/lz4hc after the shuffle, bitshuffle, blosc1, blosc2, pcodec, zfp) and
all returned a negative result with numbers. One lead for a sub-1 KB-chunk world: per-variable
zstd dictionaries (+22 % ratio and +23 % compress speed on RH at 960 B blocks), at the cost of a
trained, versioned dictionary per variable and chunks that are no longer self-contained.

**Implementation notes.** Production `from_bytes` wraps the decompressed bytes in a `bytearray`
copy that the baseline here omits (~10 % of its decode time); a shuffle behind the current
bytes-returning `Compressor` interface would need one extra copy unless the interface returns
arrays. The d01 file was written with libzstd 1.5.7; the repo env has 1.5.6, whose level 1 is
~2 % looser, so `zstd-1` re-runs read 565 MB against 556 MB stored (the subset used for review
was written by this env and matches exactly).

**7. Station (ts_ortho) data: the size win carries over; at its small stored chunks the speed win
mostly doesn't.** ECan hourly streamflow (public, 141 stations, packed uint32, stored as one station
× 25 000 hours = 100 KB raw per chunk; `results/2026-09-23_ecan_streamflow/`):

| pipeline | file | vs stored | compress | decompress |
|---|---|---|---|---|
| zstd-1 (current) | 32.7 MB | 1.00 | 935 MB/s | 2.9 GB/s |
| **shuffle + zstd-1** | **23.5 MB** | **0.72** | **1837 MB/s** | **3.3 GB/s** |
| shuffle + zstd-3 | 22.9 MB | 0.70 | 1417 MB/s | 3.3 GB/s |
| blosc1 shuffle + zstd-1 | 23.9 MB | 0.73 | 1967 MB/s | 6.8 GB/s |
| lz4-1 | 46.9 MB | 1.43 | 1927 MB/s | 3.7 GB/s |
| shuffle + lz4-1 | 25.0 MB | 0.76 | 2926 MB/s | 2.2 GB/s |

The shuffle saves the same 28 % as on the grids and doubles compression speed; decompression is only
1.1× faster at these 25 K-element chunks, and the sweep puts the decode crossover between ~12 K and ~6 K
elements (grids: ~4 K). shuffle+lz4-1 decodes slower than plain lz4-1 at every size here and slower than
zstd-1 at the stored size, so on small chunks it is not the fast option. `shuffle+ydelta+zstd-1`
equals plain shuffle because the second-to-last axis (stations) has extent 1 per chunk, so the
filter correctly applies no delta. The sweep's whole-file projection (26.4 MB for shuffle+zstd-1)
sits ~12 % above `codec_bench`'s actual 23.5 MB because it samples only the first 20 MB of chunks;
use `codec_bench` for sizes and the sweep for the shape of the curve.

**Caveats.** Not measured: bool/datetime/int64 variables (the filters round-trip them; speed
unmeasured). `shuffle+lz4-1` was measured after review round `cfdb-compression-1` and has not been
reviewed.

## Results layout

- `results/review-cfdb-compression-1.md` — the review round: 17 findings with verdicts, the exploration results, before/after of the tools.
- `results/<date>_<dataset>/codec_bench.{txt,json}`, `chunk_size_sweep.{txt,json}` — output of the two tools (regenerated with the reviewed tools; `blosc1` included). `*_lz4.{txt,json}` (d01): lz4-1 and shuffle+lz4-1 against the zstd pair; `chunk_size_sweep_all.{txt,json}` (d01): all five pipelines in one run, the source of Tables 3–5. `2026-09-23_ecan_streamflow/`: the ts_ortho run in finding 7.
- `results/2026-09-22_wrf_v50_12km_d01/exploration/` — the original scratchpad round (30+ codecs
  including blosc2/pcodec/zfp/fpzip/bitshuffle, delta-axis experiments, float32-direct test,
  per-variable size sweep); `REPORT.md` there is the narrative (superseded by the Findings above
  where they disagree), `bench.py` the codec zoo. Those
  scripts expect a `chunks/*.npy` extraction (`extract_chunks.py`) and the optional packages.
