# Perf-round benchmark results — 2026-07-21

Comparison of `perf-baseline-2026-07-21.json` (pre-round, commit 5f2dbbe, on
rechunkit 0.5.1) vs `after-perf-2026-07-21.json` (cfdb C1/C2 changes + the
full patched rechunkit via `uv run --with ~/git/rechunkit --refresh-package
rechunkit`). Median of 5 runs per case on a seeded ~51 MB dataset;
`planned_reads`/`stored_chunks` metrics are deterministic.

Regenerate with:

    uv run --with /home/mike/git/rechunkit --refresh-package rechunkit \
        python benchmarks/bench_core.py --compare \
        benchmarks/results/perf-baseline-2026-07-21.json \
        benchmarks/results/after-perf-2026-07-21.json

## Headline

| Case | Before | After |
|---|---|---|
| groupby_amplification (non-divisor daily groups, 4 MB budget) | 27.3x planned reads, 58 MB/s | **2.4x planned reads, 556 MB/s (9.6x faster)** |
| rechunk_prepended (post-prepend origins) | 400 MB/s | **642 MB/s (+61%)**; decompress amplification 3.44x -> **1.00x** (storage-space declaration — every read hits exactly one storage chunk) |
| everything else | — | +2–8% faster across the board (rechunkit planning improvements riding along); no case slower |

The remaining gap between rechunk_prepended (642 MB/s) and rechunk_aligned
(1470 MB/s) is the phase-boundary re-read term at read-group edges plus the
larger phase-padded buffer — not neighbour re-decompression, which is fully
eliminated.

## Full comparison

```
case                       metric               base          new    delta
read_full_plain            median_s          0.05532      0.05137    -7.1%
read_full_plain            mb_per_s            882.6        950.5    +7.7%
write_full_plain           median_s          0.09879       0.0933    -5.6%
write_full_plain           mb_per_s            494.2        523.4    +5.9%
write_partial_v            median_s           0.2763       0.2653    -4.0%
write_partial_v            blocks_per_s        723.8        753.8    +4.2%
write_partial_vf           median_s          0.08094      0.07847    -3.1%
write_partial_vf           blocks_per_s         2471         2549    +3.1%
write_partial_vi           median_s          0.07722      0.07591    -1.7%
write_partial_vi           blocks_per_s         2590         2635    +1.7%
rechunk_aligned            median_s          0.03544      0.03322    -6.3%
rechunk_aligned            mb_per_s             1378         1470    +6.7%
iter_chunks_storage        median_s          0.03228      0.03035    -6.0%
iter_chunks_storage        mb_per_s             1512         1609    +6.4%
groupby_daily_fast         median_s          0.03617      0.03405    -5.9%
groupby_daily_fast         mb_per_s             1350         1434    +6.3%
groupby_amplification      planned_reads         2730          240   -91.2%
groupby_amplification      stored_chunks          100          100    +0.0%
groupby_amplification      amplification         27.3          2.4   -91.2%
groupby_amplification      max_mem_mb              4            4    +0.0%
groupby_amplification      median_s           0.8459      0.08778   -89.6%
groupby_amplification      mb_per_s            57.72        556.3  +863.7%
many_small_ops_100         median_s           0.1796       0.1737    -3.3%
many_small_ops_100         ops_per_s            1670         1727    +3.4%
rechunk_prepended          median_s           0.1222      0.07612   -37.7%
rechunk_prepended          mb_per_s            399.7        641.5   +60.5%
```
