# Review round `cfdb-shuffle-code-1` — 2026-09-23

Code review of the cfdb 0.10.0 / cfdb-models 0.1.2 working tree (byte-shuffle compression,
element-based chunk default, `format_version`, `copy()` and `merge_into` fixes). Three blind,
parallel arms on the same brief; scope: the `cfdb` and `cfdb-models` repos (S3 credentials moved
out for the round, restored and checksum-verified after). Arms: Sonnet 5 (effort max, 22 turns),
Gemini 3.1 Pro (effort high, 336 turns), GLM-5.3 via OpenRouter (effort high; hit its 90-minute limit
before writing a report — its findings were recovered from its activity log). All three built the
environment and ran the suite; Sonnet and GLM also ran the 0.9.7 code side by side. Brief and raw
output: `~/git/ai-review-harness/results/cfdb-shuffle-code-1/`.

Verdicts: **verified** = reproduced by the author; **refuted** = checked and wrong; **accepted** =
not independently re-run, with the reason.

| # | Finding | Arms | Verdict | Change |
|---|---|---|---|---|
| 1 | 16-byte values (`float128`) crash under `zstd_shuffle` (`data type '<u16' not understood`) — a regression: they worked in 0.9.7 and still work with `zstd` | G | **verified** | shuffle only 2/4/8-byte values (`SHUFFLE_ITEMSIZES`); others stored as is |
| 2 | Surviving mutants: dropping `M`, `i` or `b` from the chunk kind check leaves every test green (no test created those kinds without a chunk shape) | S, GLM | **verified** (466 passed with the `M` mutant) | test creating every kind without a chunk shape |
| 3 | The upgrade message's `'$.compression'` also matches `'$.compression_level'`, so a malformed level was reported as "written by a newer cfdb" | GLM | **verified** | match the exact path `` `$.compression` `` |
| 4 | Mutating `DatasetView._compressor_for` survives | GLM | **verified: the method was unreachable** (views build variables from the base dataset) | removed |
| 5 | The "shuffle decodes faster" claim had no real-data evidence for raw 8-byte types | S | **measured**: datetime64 and float32-widened float64 gain a lot (0.03–0.80× size, faster both ways); full-precision float64 is faster but 0.90–1.13× the size (larger at 262 K elements) | caveat in changelog and chunking docs; default unchanged |
| 6 | `_open_inputs` leaves already-opened inputs open (files locked) when a later one fails | S | **verified** (read; pre-existing) | close them on failure; test |
| 7 | `merge_into` crashes on any pure append along a float coordinate (`IndexError`: an empty float mask) | GLM | **verified** (pre-existing: same crash in 0.9.7, any compression) | `dtype=bool`; test |
| 8 | `get_chunk` returns data that differs from the full array | GLM | **refuted**: `get_chunk` returns the first chunk of a selection by design; identical in 0.9.7, `zstd` and `zstd_shuffle` | none |
| 9 | Default chunks for station/forecast data span many stations | G | **refuted as a regression**: the new default spans FEWER stations than the old (60 vs 200; 6 vs 24) | none |
| 10 | Big-endian hosts pay an extra byte-swap copy in `unshuffle_bytes` | G | accepted (reasoning; no big-endian hosts in use) | none |
| 11 | Sound, by execution: host-independent layout; nothing reads the unbound dataset compressor; a writable reopen by 0.10 leaves 0.9.7 files readable by 0.9.7; the upgrade message fails safe; every chunk path uses the bound compressor; `copy()` correct for 2-D double prepends, ts_ortho and views of shifted datasets with no extra memory | S, G, GLM | accepted (several arms executed each) | none |

## Implementation check after the fixes

- 15 new tests, each seen failing for its intended reason before the fix (one first failed because
  of a bug in the test itself — fixed and re-run).
- Full suite: 495 passed (incl. S3-backed EDataset tests).
- Re-mutation: the 11 earlier mutants plus 7 new ones (`M`/`i`/`b` dropped from the kind check,
  shuffle any item size > 1, prefix-matching upgrade message, float mask without `dtype=bool`,
  `_open_inputs` not closing on failure) — all 18 turned tests red; every file restored checksum-identical.
- `ruff --select F` on every touched file: identical counts to HEAD.
