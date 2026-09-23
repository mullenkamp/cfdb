# Review round `cfdb-compression-1` — 2026-09-22

Code + methods review of `benchmarks/compression/` and its conclusions. Three blind, parallel
arms on the same brief and scope (repo + a 104 MB, 19-variable subset of d01): Fable 5.1 (effort
high, 23 min), Sonnet 5 (effort max, 26 min), Gemini 3.1 Pro (effort high, 6 min). All three
executed the tools. Brief and raw reports: `~/git/ai-review-harness/results/cfdb-compression-1/`.

Verdicts: **verified** = reproduced by me from the recorded JSON/logs or by running it;
**refuted** = checked and wrong; **accepted** = not independently checked, with the reason.

## Findings and verdicts

| # | Finding | Arms | Verdict | Action |
|---|---|---|---|---|
| 1 | Conclusion 1 ("better on all three axes") holds only above ~4 K elements; below that the numpy shuffle is slower than zstd-1 both ways (per-call cost of the plane split/join) | F, S | **verified** from the d01 sweep JSON: decompress crosses at ~4 K elements (1088 vs 981 MB/s), compress at ~2 K | README reworded; crossover table added |
| 2 | Round-trip gate escapes: (a) a compressor that mutates its input in place passes (sweep blocks are writable copies); (b) a decompressor returning a view into a reused buffer passes (each check is immediate); (c) non-contiguous / Fortran output passes | F (a), S+G (b), G (c) | **verified**: all three pass a replica of the old gate and are refused by the new one (`test_gate.py`) | `gate.py`: snapshot before compress, C-contiguity, previous-output re-check; 12 mutant/control tests |
| 3 | Gate does refuse a flipped byte, a truncated plane, wrong dtype; `tobytes()` is the right comparison — `array_equal(equal_nan=True)` would pass a NaN whose payload bits changed | F, S | **verified** (`test_gate.py::test_nan_payload_*`) | kept |
| 4 | Conclusion 2's numbers wrong: shuffle+zstd-3 is 1.5 % smaller at 1.34× slower, not "0.1 % for 2–3×" (I had conflated the ydelta comparison and the unshuffled zstd-3 speed) | F | **verified** from `codec_bench.json` | README corrected |
| 5 | "blosc2 shuffle+bytedelta is matched by numpy" is true for size only; numpy ydelta decodes 1.5× slower (1300 vs 2006 MB/s) | F | **verified** from my own logs | README corrected |
| 6 | Conclusion 4 (ratio flat to ~4 K elements) survives independent routes: Fable (public `iter_chunks`, transpose shuffle, random-offset tiles / flat slabs / k-timestep tiles), Gemini (own slicing, 1-D and N-D), Sonnet (own sweep) | F, S, G | **verified** by three arms, all executed | kept |
| 7 | Caveat to 6: the default ladder halves y/x before time, so small-block rows keep 12–24 timesteps; the projection is ~5 % optimistic at 4 K for time-redundant variables (`sea_surface_temp`: 11.4 at 24 steps, 1.9 at 1) | F | **verified** (`--shrink-axis 0` vs `2`) | README caveat |
| 8 | Caveat to 6: zstd-1's *rise* at small blocks (1.31 → 1.53) is a level-1 parameter effect — the level table lowers `min_match` from 7 to 5 for small sources | F | **verified**: `from_level(1, min_match=5)` gives 1.517 on the full chunk; `from_level(1, source_size=8 KB)` reports `min_match=5` | README note |
| 9 | `iter_var_chunks` sorted Booklet keys as strings (`0, 120, 144, 24, 48, ...`), so the byte cap sampled a scattered set of chunks, at every rung. Top-rung projections still matched the uncapped `codec_bench` (1.97 vs 1.98; 2.78 vs 2.78) | S (also noted by F) | **verified** by re-sorting the real keys | numeric chunk order in `chunks.py`; the cap now keeps the first chunks as documented. Sonnet's suggested fix (zero-pad the on-disk keys) rejected — that is cfdb's file format |
| 10 | Whole-file projection mixes physical block sizes when variables have different chunk shapes (column = ladder step index; short ladders clamp to their last rung). Harmless on both datasets (one chunk shape each) | G, F, S | **verified** (Gemini's cited example was wrong for d01 — all 33 vars share one ladder — but the design flaw is real) | projection now on a common element grid, nearest rung per variable, flagged when > 1.5× off |
| 11 | Timing is fair; tool is 0–28 % pessimistic vs hot `timeit`, most on the fastest paths (first-call allocation); min-of-2 is thin in narrow (`--vars`) runs | F, S, G | **verified** by two arms' `timeit` runs; consistent with my own earlier pipeline timings | untimed warm-up per codec; `--reps` default 3 |
| 12 | Unmodelled: production `from_bytes` wraps bytes in a `bytearray` copy (~10 % of the baseline's decode time, flatters the current codec); a shuffle behind the bytes-returning `Compressor` needs an extra copy unless the interface returns arrays | F | **accepted** — read `dtypes.py:342`, not timed | README note; implementation consideration |
| 13 | HARD subset: only `vorticity` changes membership under the shuffle; a fixed baseline-selected set is the correct design for a comparison (Fable), vs "misleading, group by dtype" (Gemini, reasoning only) | F vs G | **verified** Fable's count from the JSON; Gemini's severity refuted for this data | table leads with ALL; HARD membership printed with its definition |
| 14 | `iter_var_chunks` yields exactly the compressor's bytes: prefix unambiguous, every write path stores full `chunk_shape`, 38/38 chunks re-encoded byte-identical | F, S | **accepted** — both executed and agree; not re-run | — |
| 15 | String/geometry variables would crash `frombuffer` rather than skip | S | **verified** by reading `dtypes.Geometry` (`StringDType`/object) | skipped with a notice |
| 16 | Filters round-trip int8–int64, uint64, f64, bool, datetime64+NaT, big-endian, Fortran/strided, zero-size; `delta_axis` agrees by construction; row-loop inverse ~5× faster than `np.cumsum(axis=…)` | F, S, G | **accepted** — three arms executed, agree | — |
| 17 | libzstd 1.5.6/1.5.7 caveat does not apply to the subset (written by this env); gabriele (10 MB, 4 vars) is thin support | F | **verified** | README wording |

## Open exploration — nothing beats `shuffle+zstd-1` on all three axes

All three arms returned a negative result with numbers. What they tried: zstd `min_match`,
`hash_log`, `window_log`, strategies (dfast/greedy/btultra2), long-distance matching, negative
levels, per-plane level schemes, storing the low plane raw, per-variable dictionaries, lz4/lz4hc
after the shuffle, bitshuffle, blosc1, blosc2 (shuffle, bytedelta), pcodec L4/L8, zfp. Useful
results, all measured with the tool on the subset:

- **blosc1 (c-blosc1, zero-dependency wheel) is the small-chunk winner** (Fable; verified with
  `chunk_size_sweep` on 6 vars): decompress 1.6–2× faster than the numpy shuffle below ~4 K
  elements at comparable size, fastest of everything at every block size, but 8–12 % larger at
  4.3 MB because its internal blocking loses long-range matches. Numpy shuffle for grids, a C
  shuffle for a small-chunk world.
- zstd level −1 behind the shuffle: 3.4 % larger, 12 % faster compress; the fastest numpy variant
  at 4 K elements (Fable, accepted).
- blosc2's shuffle is *worse* on whole-file ratio than the numpy one (2.48 vs 2.60) — blocking
  before zstd (Gemini, Fable, Sonnet agree; also in my round-1 numbers).
- "Store the low byte plane raw" is catastrophic on redundant variables (RH 20.9× → 2.0×): the
  low plane only *looks* random on the hard variables (Sonnet, accepted). Consistent with the
  round-2 finding that the delta's gain lives in the low plane.
- **Per-variable zstd dictionaries** help exactly where the shuffle is weakest — sub-1 KB blocks:
  RH at 960 B 10.6 → 12.9 (+22 %) and +23 % compress speed; wind_direction +4 % / +100 %
  (Sonnet, accepted — not reproduced). Architectural cost: a trained, versioned dictionary per
  variable, and chunks stop being self-contained. Backlog lead, not a recommendation.
- TurboPFor / streamvbyte / ALP have no maintained PyPI wheels (Sonnet, accepted).

## What changed in the tools (before → after)

| | before | after |
|---|---|---|
| gate | inline `tobytes()` equality after decompress | `gate.RoundTripGate`: snapshot before compress, dtype/shape, C-contiguous, bytes, previous output re-checked; `test_gate.py` (7 mutants + NaN mutant + 3 controls) |
| chunk order | string sort of Booklet keys | numeric chunk-start order |
| timing | min-of-2, no warm-up | untimed warm-up per codec, min-of-3 |
| dtypes | crash on string/geometry vars | skipped with a notice |
| projection | by ladder step index | common element grid, nearest rung, mismatch flagged |
| table | HARD first | ALL first; HARD membership + definition printed |

Conclusions 1–5 stand with the corrections above; none reversed.
