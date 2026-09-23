# Review round `cfdb-profile-1` — 2026-09-23

Code + methods review of the real-data ladder mode added to `benchmarks/profile_cfdb.py` and of the
conclusions first written to `RESULTS.md`. Two blind, parallel arms on the same brief and scope
(cfdb repo + the 104 MB, 19-variable d01 subset): Gemini 3.1 Pro (effort high, ~6 min) and Fable 5.1
(effort high, ~21 min). Both executed the profiler and wrote independent experiments. Brief and raw
reports: `~/git/ai-review-harness/results/cfdb-profile-1/`.

Verdicts: **verified** = reproduced by me; **refuted** = checked and wrong; **accepted** = not
independently checked, with the reason.

| # | Finding | Arms | Verdict | Action |
|---|---|---|---|---|
| 1 | `encode` was timed per target chunk, but `DataVariableView.set` encodes the whole input once per call. At 2 K-element chunks this overcounted by ~6 µs/chunk; "encode ~7.5–8 µs/chunk, as large as compression" was an artefact | G, F | **verified**: one 2.15 M block 2.76 ms vs the same data as 1 122 pieces 9.69 ms (+6.2 µs/piece) | encode timed per source block |
| 2 | The write self-check was blind by cancellation: the encode overcount (+~7 µs) cancelled uncovered per-chunk work (−~5 µs), so the residual read ≈ 0 | G, F | **verified**: residual re-derived with encode per block is 5–7 µs/chunk for packed vars, matching ivt's; uncovered steps timed directly: key generator 2.7 + missing-key `get` 1.9 + strided-assign extra 0.3 µs | those steps added as components (`keys`, `get_missing`, strided `assemble`) |
| 3 | One `set()` per target chunk — the loop the cfdb skill recommends — costs 2–2.5× a block write; each `set()` builds 8 cfdb objects because `get_coord_origins` tests `hasattr(self, 'coords')` on a property (`support_classes.py`) | F | **verified**: 69.8 vs 28.0 µs/chunk (u16), 49.5 vs 23.6 (f32) at 2 040 elements; 17 952 `__init__` calls for 2 244 sets | added as a pipeline (`write_per_chunk`). The `hasattr` cost is a cfdb change — **not made**, backlog |
| 4 | Padding: per-real-element accounting charges a shape for its fit to this grid; report per stored element too. RESULTS.md's table had omitted the three highest-padding rungs | G, F | **verified** (omission was unintentional but real) | both columns, every rung |
| 5 | The large-chunk penalty is fresh-buffer page faults / allocator behaviour, not the CPU cache (Gemini said cache) | F (G: cache) | **verified F, refuted G's mechanism**: a decode-like op with a reused buffer stays at 0.16 ns/el to 2.15 M elements; fresh buffers 0.39 (2.3×); cache effects only at ~50 MB. glibc tuning (`MALLOC_TRIM/MMAP/TOP_PAD`) made stored-shape reads 15–25 % faster here (Fable: ~40 % on its machine) | component split flagged unreliable ≥ 1 M elements; penalty stated as default-allocator behaviour |
| 6 | The two-point "fixed cost" split booked zstd's small-input inefficiency as fixed; "codec ≈ a third of the read fixed cost" unsupported | F | **verified** from the JSON: zstd decompress local slope 1.6 ns/el at 2–4 K vs 0.8 at 66 K | split removed; per-chunk costs reported directly |
| 7 | The residual is a weak check (~5–10 µs/chunk floor, blind to cancellation); RESULTS.md's "writes within ~5 %" was wrong — ivt write residual +22–28 % at 2–8 K, reads +16–19 % at 2–4 K | F | **verified** from the JSON | residual shown in µs and %, floor stated. **Re-mutated**: with the fix, the 2–8 K write residual is +1.5–2.4 µs; re-applying the per-chunk encode defect moves it to −5 to −6 µs (−7 to −21 %) — detected at ≤ 8 K, not at 16 K |
| 8 | File close after a fresh write is Booklet sync (1–4 ms) + file close (fs-dependent), not "the OS writing the file out"; the reported 17–50 ms did not reproduce | F | **attribution verified, magnitude partly refuted**: an isolated 6.4 MB write gave sync 2.1 ms + close 1–13 ms, with `os.sync` afterwards still 14–29 ms; but the full reruns (30–64 MB files) measured 24–51 ms again, so the close grows with file size | sync and close timed separately; reported per file with that caveat |
| 9 | Rewritten files match the source (values, dtype metadata, compression) | G, F | **verified** bit-exact (Gemini had used `np.allclose`) | — |
| 10 | Tier-mode fix is equivalent for its use | G, F | **verified** by both arms' runs; accepted | — |
| 11 | The leading 89-row slice makes the last time chunk 71 % real for u32/f32 at the top two rungs (~8 % stored-element bias) | F | **accepted** — derived from shapes, not re-run | noted |

## What the corrected conclusions are
See `RESULTS.md`, section "Per-chunk costs vs chunk size, real data", rewritten from the rerun.
