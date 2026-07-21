# rechunkit + cfdb performance round (P1–P4)

## Context

The 2026-07-20 dual-blind review verified rechunkit's core algorithm correct (~2,300 oracle configs) and identified four performance findings, deferred out of the bug round (shipped, committed: rechunkit `db5ca6c`, cfdb `5f2dbbe`). This round fixes P1–P4 plus the doc claims they falsify. Inputs: `cfdb/plans/perf-round-planning-notes.md`, `cfdb/plans/rechunkit-cfdb-review-2026-07-20.md`, review findings in `envlib/OPEN_WORK.md`.

**Mike's rulings (2026-07-21, perf round):**
1. **P1b batched single path — INCLUDE** (the round's payoff: flagship groupby amplification ~90× → ~4× at default memory, incl. esa-sst-shaped data and multi-variable DatasetRechunker).
2. **P4b — narrow candidate check** (3 candidates, plan-as-scorer), despite the documented prior rejection of the full candidate search; new evidence (>20% pathologies) justifies the revisit.
3. **P2 — enforce the budget fully**: shrink read groups below the LCM when needed (≤2× outer-dim boundary reads beats 30×–920× memory violations).
4. **cfdb `Rechunker.rechunk`/`calc_n_reads_rechunker` default max_mem 2**27 → 2**29** (consistent with iter_chunks/groupby/map/DatasetRechunker).
5. **Benchmark checkpoint + pause-and-assess at every major step**: after each of R2, R3, R4, R5, C1 — run that step's benchmark/count cases, compare measured wins against the plan's predictions, then PAUSE for Mike's assessment (keep / adjust / revert) before starting the next step. The added complexity of each step must earn its measured improvement; R5 (candidate check) and R4's below-LCM shrinks are the most likely candidates for a revert-if-marginal call.

No version bumps (one release cycle covers bug + perf rounds; `rechunkit>=` pin at release). Changelog entries extend the same Unreleased sections. **The dual-blind plan review is COMPLETE** (see Review trail below) — both arms converged; amendments folded in; implementation proceeds with fail-before + benchmark gates and the per-step pause-and-assess checkpoints.

**Design-phase verification already done**: a fresh-context Plan agent designed against the live code; I re-verified its load-bearing claims: rechunkit yields are sel-relative (`_rechunk_plan` computes in `[0, target_shape)`; `rechunker()` yields unchanged) so P3's transform needs no yield shift-back; the prior candidate-search rejection is real (`optimization-internals.md:21-56`); both trees committed clean.

---

## Phase 0 — Benchmark harness prep (BEFORE any perf edit)

The bug-round harness gap: rechunkit micro-cases run 3–10 ms with ±8–19% same-tree variance — too noisy to gate rechunkit-core changes. On the committed (pre-perf) trees:

- **Scale up** `rechunkit/benchmarks/bench_rechunk.py` configs (bigger shapes and/or more N_RUNS) until same-tree variance is ≲5% on timing cases.
- **Add deterministic-count cases** (counts, not wall time — these are the real perf gate):
  - `groupby_amplification`: non-divisor `(1, full, full)` target over a 3-D array (scaled ERA5 analog, e.g. `shape=(72,73,144)`, `src=(10,10,10)`), instrumented source-function call count + wall time. This is P1's fail-before case.
  - `pending_wide`: the review's wide-array construction; tracemalloc peak vs max_mem ratio. P2's fail-before case.
  - `mixed_plan_peak` (**both review arms**): a tuned config forcing a MIXED bulk+multi plan, tracemalloc peak vs max_mem — `pending_wide` is pure-bulk and `groupby_amplification` pure-multi, so neither covers the one place two steps' memory stacks (the R3 `pending_bound` term's gate).
  - `plan_only_small_constrained` (**Gemini Part G**): planning wall time on a small constrained case UNDER `SCORE_LIMIT` — R5's scoring passes and R4's cascade add planning work exactly where the existing `plan_only_large` (above the limit, scoring skipped) doesn't measure; guards millisecond-scale regressions on cheap queries.
- cfdb `benchmarks/bench_core.py`: keep `rechunk_prepended` (P3's case, 376 vs 1374 MB/s aligned — the ~3.7× harvest); add a non-divisor `groupby({'time':'D'})`-style amplification case with a decompress counter.
- **Recapture baselines** on the committed trees with the new harness → `benchmarks/results/perf-baseline-<date>.json` in each repo (the `after-bugfix-2026-07-21.json` files predate the harness changes and are superseded).

## rechunkit steps (all in `rechunkit/main.py`; they stack in this order)

### R1 — P4a foundation: true-buffer budgeting + param threading

- New module-private single source of truth, used by BOTH the planner and `rechunker()`'s allocation:
  `_buffer_bytes(read_shape, target_chunk_shape, phase, itemsize) = prod(max(r+p, t) per dim) * itemsize`
- Backward-compatible signatures: `calc_ideal_read_chunk_shape(src, tgt, shape=None)`, `calc_source_read_chunk_shape(src, tgt, itemsize, max_mem, shape=None, phase=None)` (phase defaults to zeros — documented approximation for bare public calls; `sel` is known everywhere that matters).
- Inside `calc_source_read_chunk_shape`: early floor (`_buffer_bytes(src,...) >= max_mem` → return src); ideal check, shrink loop, and grow steps all switch from `prod(read)` to `_buffer_bytes`.
- **Floor contract stated honestly (blind-review A, measured)**: the floor's *allocation* is `_buffer_bytes(src) = prod(max(src+phase, tgt))·itemsize` — it includes the full target chunk and can exceed max_mem (flagship groupby: 396 MiB on a 128 MiB budget, 3.1×). Docs must distinguish the **bulk floor** (`_buffer_bytes(src)`) from the **multi floor** (`tgt_bytes`, R3) and note R3 supersedes the bulk floor for single-path cases. cfdb's public `Rechunker.calc_source_read_chunk_shape` doesn't pass sel → docstring notes the standalone number is a lower bound for misaligned views.
- **Consistency requirement (correctness-critical)**: `rechunker()` (main.py:475) and `_rechunk_plan` (main.py:328) currently compute the read shape independently; both must parse `sel` → `target_shape`, `phase` FIRST and call `calc_source_read_chunk_shape` with identical arguments — divergence breaks bulk-path buffer indexing.
- Fail-before test: reviewer's case `src=(5,16)`, `tgt=(14,2)`, budget 640 B (currently allocates 1792 B); plus ~200-config randomized sweep asserting `_buffer_bytes(result) <= max_mem` except documented-floor cases.

### R2 — P1a: clip the ideal read shape to the array extent

Per dim (shape = the extent the group grid tiles = `target_shape` from the plan; the variable/view shape from cfdb's public methods):
```
cover_d = ceil(shape_d / src_d) * src_d      # smallest source-aligned extent covering the dim
ideal_d = lcm(src_d, tgt_d) if lcm < cover_d else cover_d
```
Correct because cross-group lcm alignment only matters where a dim has ≥2 groups; `cover_d` yields exactly one group in that dim, stays a multiple of `src_d` (so `k_factors = ideal // src` stays integral). Phase interaction verified: buffer dim `max(ideal_d + phase_d, t_d) ≥ target_shape_d + phase_d` covers the backward-extended reads; reads remain individually source-aligned (the cfdb invariant). `_rechunk_plan`/`rechunker()` pass `shape=target_shape` to both calc functions (the ideal-path trigger `source_read == ideal` must compare like with like). `calc_n_reads_rechunker` flows through automatically. ERA5 effect: ideal 193 GiB → 480 MB; `k_factors` (1,721,72) → (1,8,15). Blind review verified with a 1,200-config differential oracle (0 failures on exactness/exactly-once/canonical-order/alignment/read-count truthfulness) and confirmed phase must NOT enter the cover formula (reads are clipped to `grp_stop = target_shape`; adding phase would over-allocate). **Checkpoint expectation**: the R2 win appears at max_mem 2**29 (ideal path, 1×); at 2**27 the flagship stays constrained until R3 — gate R2's benchmark at 2**29.

### R3 — P1b: batched single path ('multi' groups)

When write chunks fall to the single path, batch consecutive ones (they're already in C-order) and dedup their reads:
- Plan side (`_rechunk_plan` constrained branch): `n_batch = max(1, (max_mem − src_bytes − reserve − pending_bound) // tgt_bytes)` where `reserve` = bulk buffer bytes if a bulk read shape is feasible under `_buffer_bytes`, else 0 (if infeasible, skip the bulk condition entirely — every write chunk batches), and **`pending_bound` = the R4 `_pending_bytes` bound for the plan's bulk groups (0 for pure-multi plans)**. **BOTH review arms independently found the original formula unsound for mixed bulk+multi plans**: pending copies from bulk groups (and multi buffers parked in `pending`) coexist with batch buffers, so without the joint term peak memory can reach `max_mem + bulk_pending`. Floor documented: when `n_batch` clamps to 1, peak = `reserve + pending + src_bytes + tgt_bytes` — irreducible, same class as today's single-path overhead. Accumulate `batch_writes` + `batch_reads` (dict keyed by read-start tuple = dedup, insertion-ordered); flush as `('multi', reads, writes, None)` when full, before any bulk group, and at loop end. `written_chunks` dedup untouched.
- Generator side (`rechunker()`): allocate `mem_arr1` **lazily** on first 'bulk' group (pure-multi plans never allocate it — also closes the latent hole where ERA5 allocates a 415 MB buffer it barely touches). 'multi' handling: one dedicated `np.empty(wc_extent)` per batch write chunk (total ≤ n_batch × tgt_bytes by construction); each read calls `source()` ONCE and scatters into every overlapping batch buffer (today's single-path intersection arithmetic, generalized); yield through the existing `_canon_idx`/pending machinery — dedicated buffers need no pending `.copy()`.
- ERA5 quantification: ~3,720 reads (3.9×) at 2**27 vs 86,400 (90×) today; ~2,280 (2.4×) at 2**29÷2 vars — **blind review reproduced all three numbers exactly by prototype** (and 960 = 1× at 2**29). Memory stays ≤ budget with the joint `pending_bound` term.
- **R3 is load-bearing for MEMORY, not just reads (blind-review G)**: today's flagship groupby allocates a 396 MiB bulk buffer on a 128 MiB budget (pre-existing 3.1× violation R1's floor cannot fix — the minimum src-aligned read still forces it); only R3's per-target-chunk buffers (31 × 3.96 MiB ≈ 123 MiB) meet budget.
- Tests: exactly-once + strict canonical-order (Counter over yields, adversarial matrix: 1–3 dims, phase≠0, tuned max_mem forcing mixed bulk/single/multi plans); source-read invariant (wrap `source`, assert per-dim start alignment and span ≤ src); differential oracle vs numpy reference **including an encoded-dtype config through the cfdb multi path** (blind-review N.C. 6); fail-before count test (scaled ERA5 analog: source calls ≤ 5× stored chunks tight-mem, == stored chunks generous-mem). **Checkpoint extra**: a wall-time spot-check on the scatter loop (`O(n_batch × reads)` intersections — not visible in count gates).

### R4 — P2: pending-aware planning budget

- Conservative `_pending_bytes(read_shape)` upper bound (dominant term: `(rows0 − 1) × row-band chunks × tgt_bytes` when >1 group exists in a faster dim); enforce `_buffer_bytes + _pending_bytes ≤ max_mem` on both the ideal candidate and the greedy result.
- Enforcement cascade on the outermost violating dim: shrink in **lcm multiples first (read-count-free — groups still tile both grids)**, then in src-chunk steps (≤2× boundary reads in that dim, per ruling 3), floor at one src chunk.
- Residual documented honestly: when `src0 > t0` on a wide array and even `R0 = src0` violates, pending is physically irreducible — doc caveat with the formula `(⌈src0/t0⌉−1) × row_width × itemsize`. **Framing per blind review D: the enforceable contract is "budget honored down to the src-chunk floor + documented residual"** — ruling 3's "enforce fully" cannot beat the src0 floor (measured: 1.7× residual at the floor on the P2 repro where `lcm0 = src0`); the R4 checkpoint is judged against this bar, not an unachievable one. (Blind review measured the `_pending_bytes` bound as tight: 1,791 vs bound 1,800.)
- Fail-before test: reviewer's wide-array construction, immediate consumption, tracemalloc peak < 3× max_mem (before: 30×–920×); read-count guard ≤ 2.2× pre-change on the same config, **plus an explicit sub-lcm-shrink case pinning the ≤2× boundary-read constant** (asserted, not yet proven — blind-review N.C. 9).

### R5 — P4b: narrow candidate check (sweep-gated, lands last)

- Constrained branch only: candidates `{greedy(max_mem), aligned_snap(max_mem), greedy(max_mem//2)}` (aligned_snap: snap factors down to multiples of `q_d = ideal_d // src_d`, regrow under `_buffer_bytes`); **all candidates filtered by `_buffer_bytes` against the real max_mem** (R5 depends on R1 — pre-R1, `greedy(mm//2)` can report fitting while its true buffer busts). Score by running `_rechunk_plan` itself in counting mode — **the override MUST be a parameter threaded through `_rechunk_plan`, never a module-global monkeypatch** (cfdb is documented thread-/mp-safe; a global patch during scoring would corrupt a concurrent normal plan — blind-review E). Pick min reads, ties → aligned. Skip scoring above `SCORE_LIMIT ≈ 20,000` target chunks (greedy kept unless aligned is within the volume guard). Planning cost: ≤3 plan iterations, only for constrained cases under the limit.
- **Gate (one-off script, not CI)**: ~1,500-config randomized sweep — aggregate reads new ≤ old, no single case > +5%, monotonicity-violation count strictly decreases. **Pin the exact config generator + memory ladder in the gate script** — the violation rate is distribution-dependent (review baseline 7/1500; blind arm measured ~9% on its own 2-D generator). CI keeps the fixed regression case `shape=(53,11)`, `src=(3,8)`, `tgt=(6,6)` **with `itemsize=8` — load-bearing and mandatory** (blind review measured the case is already monotone at itemsize 1–2, so the test is vacuous without the pin; it also confirmed `greedy(mm//2)` restores `reads(1024)=54 ≤ 54` at itemsize 8): `reads(1024) ≤ reads(512)`.
- Append a "2026-07 revisit" subsection to `docs/concepts/optimization-internals.md` (history stays truthful: what changed vs the rejected search and why).

### R6 — rechunkit docs + tests + suite

Doc corrections riding with the behavior: `how-it-works.md:63-70` monotonicity claim → "generally decreases; small increases possible; no guarantee"; `rechunker()` docstring max_mem claim (main.py:453) → accurate contract (bounds read buffer + pending copies + batch buffers; floor + wide-array residual caveats). Full suite (51) + oracle sweep green after EACH of R1–R5, not just at the end (each step alone changes plan geometry).

## cfdb steps

### C0 — local rechunkit overlay

Bug-round convention: run cfdb tests via `uv run --with /home/mike/git/rechunkit --refresh-package rechunkit ...` (NO editable install, no pyproject/lock edits; a stale-cache feature-assert first — e.g. `hasattr` on a new rechunkit symbol).

### C1 — P3: storage-space rechunk (independent of R1–R5; works on rechunkit 0.5.1 sel contract)

In `Rechunker.rechunk` + `Rechunker.calc_n_reads_rechunker` (`support_classes.py:146-260, 124-143`), per dim:
```
d = origin % chunk            # 0 ≤ d < chunk (Python mod handles negative origins)
S = d − origin                # storage→declared shift, a multiple of chunk
decl_shape = full_shape + d;  sel_decl = (sel_user or full) shifted by +d
```
- Fast path: all `d == 0` → legacy args, zero-diff for aligned variables. Empty-variable guard before the transform.
- Sel contract satisfied (`start ≥ 0`, `stop ≤ decl_shape`, step 1); phase becomes the selection's misalignment against the STORAGE grid, so every rechunkit read maps to exactly one storage chunk.
- Source functions (`read_decoded`/`read_encoded`): replace `index_combo_all` with the direct `−S` shift → `slices_to_chunks_keys` unchanged; the single-chunk fast branch now always fires; KEEP the multi-chunk assembly loop as a defensive fallback. Optional tidy: factor the two near-identical closures into one helper.
- **Yields unchanged** (sel-relative slices select the same elements) — no consumer changes in iter_chunks/groupby/DatasetRechunker/Dataset.rechunk; pin with a byte-identical yield-sequence test (plain, sliced view, and 2-var DatasetRechunker with different storage chunk shapes/origins). `calc_n_reads_rechunker` gets the identical transform so predictions match instrumented counts. DatasetRechunker zip-sync safe: per-var `d` differs but the target chunk grid is identical.
- **Both review arms verified the algebra independently** — the blind arm end-to-end (real `rechunker` + real `slices_to_chunks_keys` on simulated prepended storage): yields byte-identical, `d=0` zero-diff, every transformed read exactly one storage chunk, amplification 2.56×/5.4× → 1.0× (1-dim worst case 1.14×, inside the 1.15× bar). Two implementation musts from it: **materialize `sel_decl` explicitly when `sel_user is None` and `d≠0`** (passing None would drop the phase); and **the assembly fallback becomes dead code post-transform — add a test that deliberately forces it** (or an assert that the fast branch fired) so it can't silently rot.
- Fallback if a hole emerges in implementation (b): the 1–2-entry decompressed-chunk cache keyed by `blt_key` (~15 lines) — do NOT implement both.
- Fail-before test: 2-dim 1-element-prepend fixture, decompress counter, full `rechunk()`: count ≤ 1.15× stored chunks (before: 3.46×). Genuine-cfdb prepend fixture + view + DatasetRechunker paths were NOT covered by the review prototype — the plan's yield-stability tests close that.

### C2 — defaults + threading

- `Rechunker.rechunk` + `calc_n_reads_rechunker` default max_mem 2**27 → 2**29 (ruling 4; changelog note).
- `Rechunker.calc_ideal_read_chunk_shape/mem`, `calc_source_read_chunk_shape` pass `shape=self._var.shape` (user-facing budgeting numbers deflate from 193 GiB-class fiction to truth; signatures unchanged). **The clip must reach ALL public budgeting consumers** — including `DatasetRechunker.calc_ideal_read_chunk_mem` (which sums per-var `Rechunker` calls) — or user-facing numbers disagree with the planner (blind-review N.C. 4).
- `warn_legacy_int_fill` call sites and encoded-path decode flow untouched — re-run the bug round's fill-consistency matrix to prove it.

### C3 — cfdb docs + verification

`docs/concepts/rechunking-internals.md`: line 43 + Memory Model (166-178) max_mem claims → corrected contract **distinguishing the bulk floor (`_buffer_bytes(src)`) from the multi floor (`tgt_bytes`)**; source-function section + "Selections and Chunk Alignment" → storage-space declaration, single-chunk reads, assembly as fallback; **a `map()`/multiprocessing note: max_mem is per-call, N workers = N× peak — R3 tightens each worker toward the budget, removing slack the old over-allocation accidentally provided** (blind-review N.C. 5). `Rechunker` docstrings same correction. Guide pages: note the groupby/full-field iteration pattern is now efficient at defaults.

## Verification

- **Fail-before**: every perf fix lands with a deterministic count/memory test proven failing on the committed pre-perf tree (git worktree), per stack convention. Counts and tracemalloc peaks, not wall time.
- **Correctness safety net after every rechunkit-core step**: differential oracle sweeps (t1/t4-style, few hundred configs: 1–3 dims, phase-shifted sels, all three-plus-new path types) — data exactness, exactly-once, canonical C-order. cfdb-level oracles re-run incl. prepends/views/encoded dtypes + the bug round's fill-consistency matrix (62-test suite stays green).
- **Full suites**: rechunkit 51 + new; cfdb 279 + new (local; live tier if creds available — no S3-path changes expected); envlib 144 smoke via `--with` overlays of BOTH patched repos.
- **Benchmark gate**: Phase-0 baseline vs post-fix on identical harness; headline table must show the four wins (groupby amplification, pending peak-mem, prepended-rechunk throughput, buffer-budget compliance) and no >10% regression elsewhere (outside the now-tightened noise band); table goes in the round record.
- **P4b sweep gate** (R5) as specified.
- ruff/black on touched files; both mkdocs strict builds.

## Sequencing

1. Phase 0 (harness + baselines, committed trees).
2. R1 → R2 → R3 → R4 → R5 (each: implement → oracle sweep → suite → **benchmark checkpoint → pause for Mike's keep/adjust/revert assessment**, per ruling 5) → R6.
3. C0 → C1 (may start in parallel with R-steps; works against 0.5.1; own benchmark checkpoint + pause) → C2 → C3.
4. Changelog Unreleased entries both repos; `envlib/OPEN_WORK.md` annotation; source left staged for Mike's review per commit-cadence rules.
5. Release ritual afterward (both rounds together): version bumps, `rechunkit>=` pin, cfdb skill refresh, date-stamp stale `0.9.4 (unreleased)` heading.

## Review trail (dual-blind plan review, 2026-07-21 — COMPLETE)

**Gemini arm** (Mike-driven agy, neutral brief `cfdb/plans/perf-plan-review-brief-gemini.md`): A, B, D, E, F SOUND; C SOUND WITH CHANGES (the n_batch pending term); G (delivered on follow-up) SOUND WITH ADDITIONS (planning-overhead benchmark gate). Confabulation check passed — all quotes map to real plan text; ran its own prototypes (confirmed 1× reads on the flagship at generous mem, traced the 53×11 non-monotone case, verified the C1 negative-modulo algebra).

**Blind Claude arm** (fresh-context Opus 4.8, adversarial brief, prototypes in /tmp/plan_reviewer/exp1–10): B, F SOUND (B via a 1,200-config zero-failure oracle; F via byte-identical yield diffs + amplification collapse measured); A, C, E SOUND WITH CHANGES; D SOUND with framing correction. Reproduced the plan's ERA5 read counts exactly (3,720/2,280/960). Its 9-item NOT CONSIDERED list is folded into the sections above.

**Convergence**: both arms independently found the identical mixed bulk+multi accounting hole (the round's one CRITICAL-class amendment) and independently endorsed every core design. **Zero cross-arm disagreements** — Gemini's Part A/D "SOUND" vs the blind arm's amendments on the same parts are depth differences (doc-contract wording, floor physics), not contradictions; the blind arm's versions are adopted because they carry measurements.

**Amendments folded in from the synthesis** (all reflected in the step sections above): R3 `pending_bound` term + n_batch=1 floor + mixed-plan tracemalloc benchmark; R1/R6 floor contract split (bulk vs multi floor; 396 MiB measurement); R4 "budget to the src floor + documented residual" framing (reconciles ruling 3 with the physics) + sub-lcm read-constant test; R5 itemsize=8 pin (test was vacuous without it), threaded-parameter override (thread/mp safety), pinned sweep generator; R2 checkpoint gated at 2**29; C1 explicit `sel_decl` materialization + fallback-forcing test; C2 clip into `DatasetRechunker.calc_ideal_read_chunk_mem`; Phase-0 `mixed_plan_peak` + `plan_only_small_constrained` cases; encoded-dtype config in the R3 oracle; R3 scatter wall-time spot-check.

**First implementation action**: sync `cfdb/plans/perf-round-plan-2026-07-21.md` (the git-tracked copy both reviewers read) with these amendments so the repo record matches the approved plan.

## Out of scope

- `written_chunks` set memory at 10⁶+ chunks (minor note; revisit only if P1b's plan changes make it worse).
- Datetime `precision=1` sizing headroom (minor note — possibly deliberate).
- Geometry/String dtypes through the rechunker; EDataset/S3 rechunk paths; `map()` multiprocessing.
- The decompressed-chunk cache (P3 option a) — fallback only, not alongside the transform.

---

## Implementation record (2026-07-21, all steps complete)

Executed R1→R5, C0→C3 with per-step benchmark checkpoints and Mike's keep/proceed ruling at each (R2, R3, R4, R5, C1). Headline results vs the perf baselines:

| Metric | Before | After |
|---|---|---|
| cfdb groupby amplification (non-divisor daily, 4 MB budget) | 27.3× reads, 58 MB/s | 2.4× reads, 552 MB/s (9.6× faster) |
| rechunkit groupby analog (1 MB / 512 KB budget) | 9× / 9× | 1× / 2.12× |
| Post-prepend rechunk decompress amplification | 3.44× | 1.00× (throughput +59%) |
| Constrained-path benchmark throughput | 852 M cells/s | 1.56 G cells/s (+79%, identical peak memory) |
| Reducible pending peak vs budget | 2.54× | 1.41× |
| Monotonicity violations (pinned 1,500-config sweep) | 203 (pristine) | 104, worst 1.67× |
| Sweep total reads | 879,229 (pristine) | 860,745 (−2.1%) |

**Deviations from the reviewed plan, discovered during implementation** (each measured before adoption):
1. **R1 floor takes memory-free growth**: when even one source chunk busts the budget, reads grow through target-forced buffer dims at zero allocation cost (the old aspect-scaling behavior was allocation-equivalent but read-worse). One pre-existing edge-case test updated to the true-buffer contract.
2. **R4's "shrink in LCM multiples first" stage is vacuous**: post-R2-clip, read shapes never exceed the per-dim LCM (`ideal = min(lcm, cover)`), so src-chunk steps are the only shrink granularity. The blind review's "free shrink" property described shapes that cannot occur.
3. **The pending cascade only shrinks dims whose shrink STRICTLY reduces buffer+pending** — blind shrinking under a constant outer band destroyed reads (63→723 class) while saving nothing.
4. **Bulk affordability gate at n_batch ≥ 2**: a mixed plan whose reservations leave a batch quota of 1 cannot dedup and is strictly worse than all-multi at the same budget; the regime flip sits outside the candidate check's reach (it varies bulk on/off at a fixed shape), so it is gated structurally. Plus `_bulk_possible` so all-multi plans never pay reserve/pending for a bulk buffer that cannot form.
5. **R5 grew to 5 candidates** (+ enforced-ideal, + src) and the enforced-ideal path now routes through scoring instead of returning unexamined; `rechunker()` threads its computed read shape into `_rechunk_plan` (consistency by construction; no double scoring).
6. **C1 is a single universal transform path** (no d==0 legacy branch): for aligned variables it degenerates to identical rechunkit arguments — proven by the byte-identical yield digests (94-line comparison across plain/view/encoded/DatasetRechunker paths).
7. **Two Phase-0 memory cases (pending_wide, mixed_plan_peak) turned out to be in the IRREDUCIBLE regime** (src0 > tgt0 at one-chunk reads) — kept as documented-residual records; `pending_reducible` was added as the case where R4's cascade actually fires.
8. **Known cost, accepted at the R5 checkpoint**: plan-only calls on constrained plans under 20,000 target chunks pay up to ~6.5× planning time (45→294 ms canary, `plan_only_small_constrained` — Gemini's Part-G case doing its job); invisible on data paths; scoring changes ~9% of constrained plan cells, median −19% reads when it does, never worse.

**Verification**: rechunkit 65/65 (+13 new tests, fail-before proven per step); differential oracles 2,100+ configs across seeds (exactness, exactly-once, canonical C-order, read alignment, count-truth — all hold at every step); cfdb 298/298 incl. live S3 tier (5 new tests); envlib 144/144 incl. live tier on dual overlays; final benchmarks captured as `benchmarks/results/after-perf-2026-07-21.json` in both repos, with human-readable before/after reports at `benchmarks/results/perf-round-2026-07-21.md` in each; ruff F/E9 clean on introduced code; both mkdocs strict builds pass. All changes uncommitted, awaiting Mike's source review; one release cycle covers both rounds.
