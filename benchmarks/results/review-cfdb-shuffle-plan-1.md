# Review round `cfdb-shuffle-plan-1(b)` — 2026-09-23

Plan review of "byte-shuffle compression and a smaller default chunk size" before implementation.
Three blind, parallel arms on the same brief, which carried a frozen copy of the plan; scope: the
`cfdb` and `cfdb-models` repos. Arms: Fable 5.1 (effort high), Sonnet 5 (effort max), Gemini 3.1 Pro
(effort high). Fable and Sonnet each prototyped the plan and ran the full suite (311 passed at
baseline and with the prototype). Brief, frozen plan and raw reports:
`~/git/ai-review-harness/results/cfdb-shuffle-plan-1b/`.

Round `cfdb-shuffle-plan-1` was stopped a few minutes in: a git-ignored file holding S3 credentials
(`cfdb/tests/s3_config.toml`) was readable in the containers through the read-only `/staging` mount,
which `--exclude` does not filter. No arm output from any round contains the file's name outside
the staging manifest or either secret value. The file was moved out of the repo for `-1b` and
restored afterwards (checksum verified). Harness backlog item filed.

Verdicts: **verified** = reproduced by the author; **accepted** = not independently re-run, with
the reason.

| # | Finding | Arms | Verdict | Plan change |
|---|---|---|---|---|
| 1 | A single byte target (512 KiB) gives 8-byte types 27–35 K elements per chunk and one 4-byte grid 52 K, below the measured best range; rechunkit's composite snapping undershoots the target by up to 2.5× | F | **verified** (guessed shapes for 8 real shapes) | target 2¹⁸ ELEMENTS × item size; a follow-up rechunking study (`benchmarks/RESULTS.md`, "Rechunking vs chunk size") confirmed ~3·10⁵ elements is best for both 2- and 4-byte values |
| 2 | A wrong but consistent item size round-trips perfectly; only the ratio suffers (2–4 %); random data defeats size tests | F, S, G | **verified** | structural test: stored chunk = byte planes of the encoded array; second fixture set written by the new version |
| 3 | `dtype.itemsize` is the DECODED width (packed float64 reports 8, stores 2); String/Geometry numpy placeholders report itemsize 16 | S, F | **verified** | one kind-gated helper, `shuffle_itemsize`, unit-tested per dtype |
| 4 | `Dataset.copy()` silently corrupts data after a coordinate prepend (pre-existing, independent of compression) | S | **verified** (copy reads back `[0 0 0 0 0 5 6 …]`) | fixed in this change; test red on 0.9.7 |
| 5 | `merge_into` hides the real error behind `UnboundLocalError` when an input cannot be opened | F | **verified** | fixed in this change |
| 6 | Older cfdb refuses a `zstd_shuffle` file at open on every read path; a separate flag would be silently ignored and then dropped on the next write | all | **verified** (ran) | keep the enum; add an upgrade message and `format_version` |
| 7 | `copy()` has no compression argument | F | **verified** | end-to-end check via `combine(compression=...)` |
| 8 | Attaching to an existing remote ignores the `compression` argument | all | **verified** (asked for lz4-9 on a public remote, got zstd-1) | docstring |
| 9 | Pickling must be tested through a real `map()`, not an attribute round trip | S | accepted (the arm ran it) | test changed |
| 10 | No existing test encodes a guessed chunk shape; every compressor holder is fine; `copy()` is the only raw-byte path | all | accepted (two arms ran the suite with prototypes) | none needed |
| 11 | Fixtures must cover grid and ts_ortho, packed/raw types, datetimes with NaT, str, geometry, shifted coordinates, missing chunks, attrs | F, S | accepted | fixture content |
| 12 | `lz4_shuffle` does not beat plain lz4 on decode; the chunking docs' "compression maxes out at 1–2 MB" contradicts the evidence; ruff is far from clean | S, F | accepted | docs; lint: no new rule families — counts rose only in style rules the codebase already breaks throughout (`Q000` single quotes, `EM101`/`EM102` exception messages, one `E501`); no `F` (correctness) rules added |

Found during implementation, not by the review: in 0.9.7, writing to a multi-dimensional variable
fails with a shape error once a non-first coordinate has a nonzero origin (reads are correct).
Loud, not silent; out of scope; filed in the backlog. The fixture generator orders its writes
around it. Matrix (coordinates `a` 12 values, `y` 15 values; `v[:] = data` after one prepend):

| variable dims | prepended | variable created | result |
|---|---|---|---|
| `('a', 'y')` | `a` (first dim) | before or after the prepend | ok, reads back identical |
| `('a', 'y')` | `y` (second dim) | before or after | `ValueError: cannot reshape array of size 240 into shape (12,25)` |
| `('y', 'a')` | `a` (second dim) | before or after | `ValueError: cannot reshape array of size 255 into shape (15,22)` |
| `('y', 'a')` | `y` (first dim) | before or after | ok, reads back identical |

Data written BEFORE a second-dimension prepend reads back correctly afterwards (`v.data` and
`v[:, 5:]` both match).

## Implementation check (2026-09-23)

- New tests: 153, each seen failing against 0.9.7 for its intended reason before the change (plus 2
  more once the 0.10.0 fixtures were added).
- Full suite after the change: 480 passed (incl. the S3-backed EDataset tests and both fixture sets).
- Re-mutation: 11 defects re-applied one at a time (shuffle on compress only, decoded-width item
  size, no kind gate, unbound compressor, `__reduce__` without item size, old `copy()` fast path,
  old `merge_into`, flat 512 KiB target, no format-version check, no upgrade message, no
  format-version stamp): all 11 turned tests red; every file restored checksum-identical.
- End to end: the 19-variable d01 subset rewritten through `combine(compression='zstd_shuffle')`
  stored 74.5 MB vs 102.8 MB (0.72×, matching the benchmarks), all variables byte-identical.
- Fixtures: files written by 0.9.7 (zstd) and 0.10.0 (zstd_shuffle) from identical content record
  identical values for every array.
