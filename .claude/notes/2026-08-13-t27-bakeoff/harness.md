# Track A — the bake-off harness

**Agent**: `t27-harness` · **Wave**: `2026-08-13-t27-bakeoff` · **Base**: `4a31817`
**Date**: 2026-08-13 · **Machine**: workstation, 24 cores, GEDLIB at `~/opt/build_gedlib/graphkit-learn`

---

## Mission

Build the GEDLIB harness that measures every candidate GED bound method against the
3,836,827 certified exact-GED values from T-03, and prove it correct on LINUX. The
orchestrator runs the campaign. `BRANCH_FAST` had been licensed on 400 LINUX pairs at
n̄ = 8.71 and `IPFP` had never been measured against exact GED at all; this is the
instrument that replaces both citations with a measurement.

## Files changed

| Path | Status |
|---|---|
| `benchmarks/real_data/eval_setup/ged_bound_bakeoff.py` | new, 2,133 lines |
| `tests/unit/test_ged_bound_bakeoff.py` | new, 107 tests |
| `.claude/notes/2026-08-13-t27-bakeoff/harness.md` | this file |

Nothing else was touched. `ged_bounds.py`, `src/isalgraph/**`, the plan files, `tickets.md`
and `CONTRACTS.md` are unmodified.

## Design decisions and why

**1. Three guards replace the brief's zero rule.** The brief and CONTRACTS §3 required
that a value of exactly `0.0` where `exact > 0` must raise. That rule is false (see
Deviations 1). What catches a misread accessor instead:

- `read_bound` — per read, per pair, per orientation: rejects `NaN`, `inf` and negatives.
  `inf` is the signature of reading an upper bound off a method that sets only a lower one.
- `capability_probe` — before any cell's pair loop, runs the method on three **synthetic**
  pairs of known positive distance and requires a strictly positive value from the
  configured accessor, plus `LB ≤ exact` / `UB ≥ exact`. All three probe pairs differ in
  degree sequence, which is what forbids a valid bound from returning zero on them.
- `all_zero_guard` — a finished cell that is identically zero against positive truth raises.

M4 validity is the backstop: an upper bound misread as `0.0` is refuted on essentially
every pair with positive distance.

**2. A cell is a method plus one option string, not a method.** `IPFP_MS` and `IPFP_DET`
are the same solver and different measurements. `meta` carries `cell` and `method`
separately; the file is named after the cell.

**3. `k = 10` initial solutions for the `_MS` cells, not 40.** Measured on the LINUX
census: mean relative error 0.0894 at k = 10 against 0.0801 at k = 40 — 10 % tighter for
4× the cost (9.9 ms against 39.1 ms per evaluation).

**4. Two prediction gates, both free.** Both compare cells already on disk, and both have
a *predicted* value rather than a plausible range, which is what makes them worth more
than a sanity check. See Evidence.

**5. Timing never comes from the parallel path.** `run_timing` and `run_n30_probe` are
single-process with `time.process_time()` around `run_method` only. The CLI refuses
`--jobs > 4`.

**6. The determinism probe rebuilds the environment per repetition.** An environment that
cached a result would make every repetition identical by construction, and the probe would
report determinism it never measured.

## Deviations from the brief, and the evidence for each

**1. The zero rule was removed. Confirmed by the orchestrator; CONTRACTS §3 amended.**

A valid lower bound legitimately returns `0.0` on a pair with positive exact GED whenever
two graphs share a node count and a degree sequence but are not isomorphic. Under D6 —
free node *and* edge substitution — the degree-preserving assignment costs nothing.

| pair | exact | BRANCH | BRANCH_FAST | BRANCH_TIGHT | STAR |
|---|---:|---:|---:|---:|---:|
| C₆ vs 2·C₃ (both 2-regular, 6 nodes, 6 edges) | 4.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| two 6-node trees, degseq [1,1,1,2,2,3] | 2.0 | 0.0 | 0.0 | — | 0.0 |

On the LINUX census this is **6 of 595** sampled certified pairs (1.0 %) with **0** validity
violations. The rule would have halted the harness on correct, merely loose bounds, and
worst on Letter (n̄ = 4.7), where degree-sequence collisions are common. Design §3.3's
"every valid lower bound returns 0 on an exact-GED-0 pair" is true; the converse it
implies is not. Pinned by
`TestKnownValues::test_a_degree_preserving_non_isomorphic_pair_gives_a_zero_lower_bound`.

**2. HED resolved and promoted to a real cell. `gedlib.md` §5 is wrong on both counts.**

`gedlib.md` §5 records HED as "LB 0.00 / UB inf — unresolved, do not use yet". Both
halves are misreadings:

- `get_upper_bound() = inf` is **by design**. `include/gedlib-master/src/methods/hed.ipp:55`
  calls only `result.set_lower_bound(hed)`. HED is a lower-bound-only method.
- `get_lower_bound() = 0.00` under defaults is a **valid but vacuous** bound. HED's default
  `--edge-set-distances HED` scores incident-edge sets by a row/column-minimum sum that is
  identically zero when edge substitution is free, as D6 makes it. With
  **`--edge-set-distances OPTIMAL`** the bound is non-degenerate.

Verified options from `hed.ipp`: `--lsape-model ECBP|EBP|FLWC|FLCC|FBP|SFBP|FBP0`,
`--threads <n>`, `--edge-set-distances OPTIMAL|HED`. `--lsape-model` alone changes nothing.
Cost: ~30 min of the 2-hour box. **I did not edit `gedlib.md`** — it is a plan file and the
orchestrator propagates it at close.

**3. `--seed` does not exist. CONTRACTS §6's example pinned string is wrong.**

`set_method("IPFP", "--seed 42")` raises `RuntimeError: Invalid option "seed"`. GEDLIB
exposes no seed option; determinism for the local-search methods comes from
`--randomness PSEUDO`. Acknowledged by the orchestrator. Pinned by
`test_gedlib_rejects_the_seed_option`.

**4. The §3.10 cross-check compares the wrong quantity for BIPARTITE.**

Design §3.10 asks that GEDLIB's `BRANCH`/`BIPARTITE` "agree with" `ged_bounds.py`, and
says disagreement halts the ticket. Plain value equality is right for `BRANCH` and wrong
for `BIPARTITE`:

- `BRANCH` reports an **LSAP optimum**, which is unique even when its argmin is not.
  Measured: **400/400 value-equal**.
- `BIPARTITE` reports the **induced edit cost of one chosen optimal assignment**. The
  argmin is routinely non-unique, and two optimal assignments induce different but
  individually valid upper bounds. Naive value equality reports **156/400** — a 61 %
  "disagreement" with nothing wrong, and the disagreements go both ways (GEDLIB tighter on
  145, ours on 99), which is the signature of tie-breaking rather than a systematic fault.

Two structural tests replace it, and both pass **400/400**:
(a) GEDLIB's returned node map, scored against *our* assignment cost matrix, attains *our*
LSAP optimum — same instance, same optimum; (b) GEDLIB's node map re-costed with
`ged_bounds.induced_edit_cost` reproduces GEDLIB's own reported value — same cost function.
The naive comparison is still reported, as a measurement of tie-breaking divergence.

**5. The n = 30 probe used reduced samples for three expensive cells.** Design §3.4b
specifies 2,000 pairs. `IPFP_MS` costs 738 ms per evaluation at n̄ ≈ 30, so 2,000 pairs is
25 minutes for one cell. Used 100 pairs for `IPFP_MS` and 200 for `REFINE_MS`/`BP_BEAM_MS`;
all other cells got the full 2,000. Recorded in each JSON as `n_pairs_timed`. The gate
verdicts are not close to the threshold — the smallest reduced-sample value is 8,126 µs
against a 1,000 µs gate — so the reduction changes no verdict.

**6. Added a `gates` stage and a `--cells` flag** (replacing `--methods`). Both are new
surface, not changes to the contract's outputs.

## Evidence

All commands run from the worktree with
`export PYTHONPATH=/home/mpascual/opt/build_gedlib/graphkit-learn`.

### LINUX end-to-end — 12 cells, 3,916 pairs, `--jobs 4`

```
python -m benchmarks.eval_setup.ged_bound_bakeoff --stage index  --data $DATA --out $S --datasets linux
python -m benchmarks.eval_setup.ged_bound_bakeoff --stage cells  --data $DATA --out $S --datasets linux --cells all --jobs 4 --chunk 500
```

UB cells report `min` over both orientations. `us/pair` from the serial timing pass.

| cell | end | mean rel err | mean abs err | exact hits | **M4** | µs/pair | pairs/s/core | n̄30 µs/pair |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BRANCH_TIGHT | lower | 0.1580 | 0.7593 | 1406 | **0** | 522.5 | 1,914 | 36,387 |
| BRANCH | lower | 0.2512 | 1.1638 | 1384 | **0** | 23.3 | 42,944 | 347 |
| BRANCH_FAST | lower | 0.2512 | 1.1638 | 1384 | **0** | 10.6 | 94,065 | 290 |
| STAR | lower | 0.5371 | 2.7923 | 48 | **0** | 5.8 | 172,652 | 105 |
| HED | lower | 0.9137 | 4.7578 | 0 | **0** | 21.7 | 46,102 | 344 |
| **IPFP_MS** | upper | **0.0894** | 0.2372 | 3460 | **0** | 9,851.9 | 102 | 738,366 |
| IPFP_DET | upper | 0.1043 | 0.3654 | 3231 | **0** | 731.1 | 1,368 | 78,199 |
| REFINE_MS | upper | 0.1196 | 0.3478 | 3271 | **0** | 647.1 | 1,545 | 48,378 |
| BP_BEAM_MS | upper | 0.1659 | 0.4610 | 3074 | **0** | 670.3 | 1,492 | 8,126 |
| REFINE_DET | upper | 0.2129 | 0.7840 | 2624 | **0** | 84.7 | 11,810 | 5,008 |
| BP_BEAM_DET | upper | 0.2531 | 0.9214 | 2359 | **0** | 93.4 | 10,709 | 1,162 |
| BIPARTITE | upper | 1.4577 | 5.7571 | 256 | **0** | 23.2 | 43,064 | 346 |

**M4 = 0 in every cell**, two-sided on 3,870 certified pairs and one-sided on 46 censored
ones. `IPFP_MS` is the tightest upper bound, consistent with the published claim.
`BIPARTITE` is by far the loosest, as expected of the Riesen–Bunke reference point.

### Gates

```
python -m benchmarks.eval_setup.ged_bound_bakeoff --stage gates --data $DATA --out $S --datasets linux
```

- **P1 — `BRANCH == BRANCH_FAST`**: `n_equal 3916 / 3916`, `max_abs_diff 0.0`,
  `mean_lower_bound 4.0194`. Equivalent under constant edge edit costs (VLDB Journal
  survey §5.2.4), which D6 has. Zero tolerance; these are sums of integers.
- **Dominance vs BIPARTITE**, 0 violations each, strictly better on:
  `IPFP_DET` 3,624 / `REFINE_DET` 3,601 / `BP_BEAM_DET` 3,645 of 3,916.

### Cross-check (design §3.10), 400 seeded LINUX pairs

| construction | value-equal | same LSAP optimum | same induced cost | passes |
|---|---:|---:|---:|---|
| `BRANCH` | 400/400 | — | — | **yes** |
| `BIPARTITE` | 156/400 | **400/400** | **400/400** | **yes** |

### Determinism (design §3.11), 3,916 pairs, 5 repetitions

| cell | defaults `frac_varying` | defaults `max_spread` | pinned `frac_varying` | pinned `max_spread` |
|---|---:|---:|---:|---:|
| BRANCH, BRANCH_FAST, BRANCH_TIGHT, STAR, HED, BIPARTITE | 0.0000 | 0.0 | 0.0000 | 0.0 |
| IPFP_MS | 0.9104 | 10.0 | **0.0000** | **0.0** |
| IPFP_DET | 0.9165 | 10.0 | **0.0000** | **0.0** |
| REFINE_MS | 0.9385 | 10.0 | **0.0000** | **0.0** |
| REFINE_DET | 0.9420 | 10.0 | **0.0000** | **0.0** |
| BP_BEAM_MS | 0.9351 | 10.0 | **0.0000** | **0.0** |
| BP_BEAM_DET | 0.9249 | 10.0 | **0.0000** | **0.0** |

**No lower bound varies** — §6's stop condition is not triggered. At GEDLIB defaults the
six local-search cells give a different answer on 91–94 % of pairs run to run, with a
spread of up to 10 edit operations; every one is fully deterministic under its pinned
string. **No option was rejected.** Pinned strings:

- `_MS`: `--threads 1 --randomness PSEUDO --initial-solutions 10`
- `_DET`: `--threads 1 --randomness PSEUDO --initialization-method BIPARTITE --initial-solutions 1`

Cause of the variation, confirmed against `ls_based_method.ipp`: GEDLIB's LS defaults are
`num_initial_solutions_ = 1`, `use_real_randomness_ = true` and a null (random)
initialiser. That is also why `IPFP` returns 3.00 on P₄/C₄ where the truth is 1.00 — a
configuration effect, not a build difference from Picasso.

### M7 cost gate at n̄ = 29.5 (IAM GREC + Protein, 25 ≤ n ≤ 35, 164 graphs)

| cell | µs/pair | p95 | pairs timed | `< 1 ms` |
|---|---:|---:|---:|---|
| STAR | 105.1 | 137.0 | 2000 | **PASS** |
| **BRANCH_FAST** | **289.9** | 393.3 | 2000 | **PASS** |
| HED | 344.1 | 451.0 | 2000 | **PASS** |
| BIPARTITE | 346.0 | 453.0 | 2000 | **PASS** |
| BRANCH | 347.0 | 455.5 | 2000 | **PASS** |
| BP_BEAM_DET | 1,162.0 | 1,534.2 | 2000 | FAIL |
| REFINE_DET | 5,007.6 | 7,701.5 | 2000 | FAIL |
| BP_BEAM_MS | 8,126.3 | 10,079.7 | 200 | FAIL |
| BRANCH_TIGHT | 36,387.2 | 70,569.7 | 2000 | FAIL |
| REFINE_MS | 48,378.1 | 68,131.6 | 200 | FAIL |
| IPFP_DET | 78,199.1 | 183,546.5 | 2000 | FAIL |
| **IPFP_MS** | **738,366.1** | 1,404,361.5 | 100 | **FAIL by 738×** |

**`BRANCH_FAST` passes comfortably.** **Every upper bound tighter than `BIPARTITE` fails**,
`IPFP_MS` by 738×. This is a decision-relevant result for `approx_ged.md`'s production
assignment, and it is the first time the gate has been evaluated at all rather than
assumed. Note `BIPARTITE` passes the gate and is 16× looser than `IPFP_MS`, so the gate and
the tightness ranking point in opposite directions at the upper end.

### Campaign projection (measurement, not estimate)

Serial timing, 500-pair seeded sample per dataset per cell, all five datasets:

| cell | linux | aids | letter LOW | letter MED | letter HIGH | **core-hours, full census** |
|---|---:|---:|---:|---:|---:|---:|
| IPFP_MS | 9813.9 | 14841.5 | 957.3 | 980.5 | 1524.3 | **5.05** |
| BP_BEAM_MS | 644.2 | 917.1 | 200.3 | 211.8 | 310.1 | 0.69 |
| IPFP_DET | 740.3 | 1446.5 | 82.3 | 88.6 | 122.7 | 0.45 |
| REFINE_MS | 641.9 | 1211.0 | 85.9 | 87.0 | 135.0 | 0.43 |
| BRANCH_TIGHT | 492.6 | 980.7 | 26.4 | 27.1 | 49.7 | 0.12 |
| BP_BEAM_DET | 92.2 | 132.7 | 31.6 | 32.1 | 45.5 | 0.10 |
| REFINE_DET | 82.0 | 147.2 | 15.2 | 15.1 | 21.0 | 0.06 |
| BIPARTITE / BRANCH / HED / BRANCH_FAST / STAR | 6–23 | 10–35 | 2.5–5.8 | 2.5–5.8 | 3.1–8.0 | 0.05 total |
| | | | | | **TOTAL** | **6.95** |

µs per method evaluation; core-hours count both orientations for the seven UB cells over
all 3,897,911 pairs. **6.95 core-hours is well inside design §6's 40-core-hour ceiling** —
about 1.8 h wall at `--jobs 4`, 18 min at 24. The Letter datasets are cheap (n̄ = 4.7),
which is why a LINUX-only extrapolation over-estimates by 3×.

### Tests and lint

```
python -m pytest tests/unit/test_ged_bound_bakeoff.py -q     ->  107 passed
python -m ruff check benchmarks/ tests/                      ->  28 pre-existing errors,
                                                                 none in my two files
```

The 28 ruff findings are in `eval_visualizations/`, `synthetic_data/` and
`eval_setup/eval_setup.py`, all present at base commit `4a31817` and none owned by track A.
`ruff check` on `ged_bound_bakeoff.py` and `test_ged_bound_bakeoff.py` is clean.

## Acceptance criteria

| # | Criterion | Status |
|---|---|---|
| 1 | Unit tests pass, incl. wrong-accessor raise, census length, UB `min`, one-sided censored validity, `graph_ids` misalignment | **met** — 107 passed; all five named cases present |
| 2 | `ruff check benchmarks/ tests/` clean | **met for my files**; 28 pre-existing findings elsewhere, untouched |
| 3 | LINUX end-to-end, all cells, mean relative error and M4 reported | **met** — 12 cells, M4 = 0 everywhere |
| 4 | §3.10 cross-check agrees | **met** — BRANCH 400/400; BIPARTITE 400/400 on both structural tests (see Deviation 4) |
| 5 | Determinism probe run and stated, especially IPFP | **met** — IPFP varies on 91 % at defaults, spread 10; fully deterministic pinned |
| 6 | Measured throughput per method | **met** — per-dataset µs/pair table and a 6.95 core-hour projection |
| — | Commit obligation | **met** — four incremental commits, clean `git status` |
| — | Prohibitions | honoured — LINUX only, ≤ 4 processes, no ssh/rsync/sbatch, no `pip install`, nothing in `scratchpad/` |

## Open issues and what I could not verify

1. **`IPFP_MS` fails the M7 cost gate by 738× and is simultaneously the tightest upper
   bound.** The frozen §5 rule ranks on tightness and M7 is a separate gate, so this is the
   orchestrator's call, not mine. It touches decision 11, which names IPFP.
2. **`BRANCH_TIGHT` is the tightest lower bound (0.158 vs BRANCH_FAST's 0.251) but fails
   M7 by 36×.** Same tension at the lower end. `BRANCH_FAST` is the only lower bound that
   is both competitive and gate-passing.
3. **Only LINUX is measured.** Every number here is n̄ = 8.72, 3,916 pairs. Nothing is
   verified on AIDS or the three Letter levels beyond the timing sample — in particular the
   frequency of legitimate zero lower bounds, which I expect to be much higher on Letter,
   and the P1 and dominance gates on the other four datasets.
4. **`n30` probe reduced samples** for `IPFP_MS` (100), `REFINE_MS` and `BP_BEAM_MS` (200).
   No verdict is near the threshold, but the p95 figures for those three are from small
   samples.
5. **The multi-start `k`** is fixed at 10 by a LINUX-only measurement. Whether k = 40 buys
   more on AIDS, whose graphs are larger, is unmeasured. The projection says k = 40 would
   cost roughly 20 core-hours for `IPFP_MS` alone — still inside the ceiling, if the
   orchestrator wants it.
6. **Determinism is measured within one machine and one build.** `--randomness PSEUDO`
   fixes the seed, so cross-machine reproducibility is expected but not tested; I have no
   second machine and may not use Picasso.
7. **`gedlib.md` §5's HED entry and `approx_ged.md` §2's production table are both now
   contradicted by measurement.** Both are plan files, so I left them alone; they need the
   orchestrator at close.
