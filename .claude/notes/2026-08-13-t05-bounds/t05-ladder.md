# Work log — t05-ladder

## Identity

| Field | Value |
|---|---|
| Agent | `t05-ladder` |
| Wave | `2026-08-13-t05-bounds` |
| Model / effort | Opus 5 / xhigh |
| Branch | `worktree-agent-a6c1b86faa0320e8a` |
| Worktree | `/home/mpascual/research/code/IsalGraph/.claude/worktrees/agent-a6c1b86faa0320e8a` |
| Base commit | `b862f5e40600b429fe8f4cf01d4eb51641599975` |
| Head commit | `e6c5e660aa42b61cf434070c57adb43388ccd4fa` (this log adds one more) |
| Started / finished | `2026-08-14T09:12Z` / `2026-08-14T09:55Z` |
| Status | complete |

## 1. Prompt as received

```
You are agent `t05-ladder`, an implementation agent in an **isolated git worktree** on your own
branch. You never see the orchestrator's conversation; everything you need is here.

This is for a *Pattern Recognition* major revision (PR-D-26-03293) due 2026-08-31. **Correctness
beats speed. An honest negative result beats a convenient one.**

## Standing obligations
1. Work only inside your worktree and your ownership set. Confirm `git rev-parse --show-toplevel`
   differs from `/home/mpascual/research/code/IsalGraph`.
2. Commit in logical commits **as you go**, not at the end.
3. Work log at `.claude/notes/2026-08-13-t05-bounds/t05-ladder.md`, using
   `.claude/notes/2026-08-13-t05-bounds/NOTE-TEMPLATE.md` verbatim. Commit it last.
4. Never `git push`, rebase, or merge.
5. **No Picasso.** No `ssh`, `rsync`, `sbatch`, `squeue`, `scp`. You write the SLURM worker; the
   orchestrator submits it. Four campaigns are running there right now and you must not touch them.
6. You cannot ask the user anything. Message `main`, record the assumption, keep working.
7. **Finding that this brief is wrong is a success.** Report it with evidence.

---

# Task: the calibration ladder — exact GED above n = 12

## Mission
Build `benchmarks/real_data/eval_setup/ged_ladder.py` and `slurm/approx_ged/worker_ladder.sh`, which
compute **exact** GED on a size-stratified sample at each `n` from 13 upward, so the proven bracket
is calibrated closer to the regime it licenses. Working means: the sampler is reproducible from seed
42 alone, every non-completion is interval-censored rather than dropped, and the whole thing runs
end-to-end on a small local slice before it ever reaches the cluster.

## Why this exists
The bracket `LB ≤ GED ≤ UB` was *selected* at `n ≤ 12`, where exact GED exists, and is *licensed* to
`n = 98`. That gap is the substance of demand **AE.1** — the Area Editor's objection is precisely
that conclusions measured on small graphs were extrapolated. T-27's own limitation 1 says no
bake-off against ground truth can close it, because exact GED does not exist above 12. **Every node
the exact solver buys narrows it**, and this ticket is where those nodes are bought.

## Read first
1. `.claude/notes/review/tasks/T-05-design.md` **§6 in full** — the frozen sampling design and the
   frozen analysis rule. Also §7 item 1's analysis rule (size and provenance are confounded) and
   amendment 7's measured cost table.
2. `.claude/notes/review/tasks/T-03-design.md` **§0** — the defect that governs your solver choice.
3. `benchmarks/real_data/eval_setup/ged_backends.py` — `NetworkxBackend`, `ExactPlusBoundsBackend`,
   `make_backend`, `BackendSpec`.
4. `benchmarks/real_data/eval_setup/ged_sampling.py` — T-03's stratified sampler, for the pattern.
5. `slurm/approx_ged/{_env.sh,launcher.sh,worker_bounds.sh}` — your worker's exemplars.

## Your ownership (exclusive write access)
Create or modify ONLY:
- `benchmarks/real_data/eval_setup/ged_ladder.py` (new)
- `tests/unit/test_ged_ladder.py` (new)
- `slurm/approx_ged/worker_ladder.sh` (new)
- `.claude/notes/2026-08-13-t05-bounds/t05-ladder.md` (your log)

Everything else is read-only. **Do not modify** `ged_backends.py`, `ged_exact_runner.py`,
`ged_merge_shards.py`, `ged_sampling.py`, `approx_ged_*.py`, `launcher.sh`, `_env.sh`,
`worker_bounds.sh`, or anything under `src/isalgraph/`. If the launcher needs a `--stage ladder`
hook, **message `main`** — the orchestrator owns that file.

## Base state
Base commit `b862f5e`. Do not rebase, merge or cherry-pick.

## Frozen design — T-05-design.md §6, reproduced so you cannot drift from it
| Parameter | Value |
|---|---|
| Population | all Suite-2 pairs with `max(n₁,n₂) = n`, pooled across the ten datasets, per rung |
| Rungs | `n = 13, 14, 15, 16, 17, 18` |
| Pairs per rung | **250**, stratified by source dataset proportionally to that dataset's pair mass at that `n`, **minimum 20** per contributing dataset |
| Seed | **42** throughout |
| Exact solver | **`networkx.graph_edit_distance`** under cost model D6 `[1,1,0,1,1,0]` |
| Per-pair budget | **1,200 s** wall |
| Non-completion | **interval-censored `[LB, UB]` under D11** — never dropped, never promoted to exact |
| Truncation | at the first rung whose certification rate falls **below 25 %**, reported as the measured exact-GED ceiling |

### The solver trap — this is the whole reason the design names networkx
`ANCHOR_AWARE_GED` is **not exact and not deterministic**. Measured on Picasso: non-deterministic on
14/15 real AIDS pairs, wrong on 4/18 against brute force, and it reports `LB == UB` — **a false
optimality certificate**, which is worse than a wrong value because it defeats the check designed to
catch one. It is retracted (`T-03-design.md` amendment 2). **Do not use it.**

And `nx.graph_edit_distance(timeout=t)` **returns its best-found-so-far cost when the budget expires**
— it does not raise and does not return `None` unless no complete edit path was found at all. So a
timed-out pair is silently indistinguishable from a certified one. **Every "exact GED" matrix in the
submitted study was produced that way.** Completion must be established by the **search terminating**,
not by a value coming back. `NetworkxBackend` already handles this; reuse it rather than calling
networkx directly, and if you must call it directly, state in your log exactly how you establish
completion.

Bounds for censored pairs come from GEDLIB `BRANCH_FAST` (`--threads 1`) and `BIPARTITE`
(`--threads 1`). **Options strings are part of the specification** — GEDLIB's upper bounds change on
74–94 % of pairs between runs at library defaults.

## Output contract
`ladder/rung_{n}.npz`, one file per rung, flat:

| Key | dtype | shape |
|---|---|---|
| `dataset_key` | `<U` | (P,) |
| `pair_i`, `pair_j` | int32 | (P,) — indices into that dataset's exported graph order |
| `n_max` | int32 | (P,) — `max(n₁,n₂)`, equals the rung |
| `exact` | float64 | (P,) — the certified value, or `inf` where censored |
| `lb`, `ub` | float64 | (P,) — always finite, always populated |
| `certified` | bool | (P,) |
| `seconds` | float32 | (P,) |
| `metadata` | `<U` | () — JSON |

`metadata`: `rung, n_pairs, n_certified, certification_rate, censoring_rate, per_dataset_counts,
seed, budget_seconds, cost_model, lb_method, lb_options, ub_method, ub_options, solver,
code_commit, computed_utc, schema_version`. Plus `ladder/manifest.json` across rungs.

## Environment
```bash
PY=~/.conda/envs/isalgraph-cpp/bin/python
export PYTHONPATH=~/opt/build_gedlib/graphkit-learn
cd "$(git rev-parse --show-toplevel)"
```
GEDLIB import order is load-bearing — use `importlib.import_module`, never plain imports, or
ruff/isort will silently break it. Do **not** put `<worktree>/src` on `PYTHONPATH`; **do not import
`isalgraph`**. A subagent's `cd` does not persist between Bash calls.

## Data (read-only)
`SANDISK=/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph`
- `$SANDISK/data/source/APPROX_GED/exported_suite2/{key}.npz` — the ten cohorts, `load_exported`.
- `$SANDISK/data/source/APPROX_GED/exported_suite2/bin_table.json` — pair counts per size bin per
  dataset. Useful for knowing which datasets can even contribute at each rung.
- `$SANDISK/data/source/GED_PRECOMPUTED/extended_merged_exact_ged/computed/*.npz` — T-03's `n ≤ 12`
  census, the rung the ladder starts above. **Censored pairs carry `inf`, not NaN**; select on
  `certified_mask` and filter with `np.isfinite`.

## Definition of done
1. **Sampler reproducible from seed 42 alone** — two independent runs byte-identical. Per-rung
   per-dataset realised counts recorded. A rung with no eligible pairs is reported as empty, not
   silently skipped.
2. **Censoring is honest.** A test induces a timeout with a tiny budget and asserts the pair is
   recorded `certified=False` with finite `lb ≤ ub` and `exact = inf` — **never a best-so-far value
   promoted to exact.** This is the single most important test in the file.
3. **`lb ≤ exact ≤ ub`** on every certified pair, tolerance `1e-9`, 0 violations.
4. **Real-data run, recorded with numbers**: execute rung 13 end to end on a reduced sample (say 25
   pairs, not 250) and report per-pair seconds, the certification rate, and the realised per-dataset
   split. Do **not** run the full ladder — that is a cluster job the orchestrator submits.
5. **A measured cost projection for rungs 13–18**, from your rung-13 timings, with the caveat that
   exact GED grows steeply and your sample is small. State the projected core-hours per rung and the
   rung at which you expect certification to fall below 25 %. The orchestrator sizes the SLURM job
   from this, so **an honest wide interval beats a confident point estimate.**
6. `worker_ladder.sh` follows `worker_bounds.sh`'s shape: `$LOCALSCRATCH` staging, `TERM`/`INT`
   traps, mirror-on-exit, per-rung checkpointing, no `#SBATCH` header. `bash -n` clean.
   It must **not** submit anything.
7. Tests cover: sampler determinism, the min-20-per-dataset floor, proportional allocation,
   an empty rung, censoring, bracket containment, and the truncation rule at 25 %.
8. All work committed, tree clean, log written and committed.

## Explicitly out of scope
- Submitting to Picasso, or touching the four running campaigns.
- The bracket-width analysis, D13, bootstrap, figures — a later track.
- Editing the launcher, the runner, the merge, the gates, the exporter or the sampler.
- Using `ANCHOR_AWARE_GED` anywhere, for anything.
- Changing the frozen §6 parameters. If you believe one is wrong, message `main` and proceed on the
  frozen value meanwhile.

## Final message
At most 15 lines: STATUS, BRANCH, WORKTREE, HEAD, LOG, TESTS, then **lead with the rung-13 measured
timings and your cost projection for rungs 13–18**, then anything the orchestrator must know.
```

Mid-task the orchestrator sent three rulings, reproduced in §8.

## 2. Understanding and plan

**Restatement of the task in my own words:** T-03's exact-GED census stops at `n = 12`, but the
bracket it validated is applied out to `n = 98`. AE.1 is that extrapolation. I build the machinery
that buys back as many nodes above 12 as the budget allows — a reproducible stratified sampler, an
exact solver whose failures are recorded as interval-censored rather than silently promoted, and a
cluster worker — then measure on real data what the ladder will actually cost and where it will stop.

**Approach chosen:** one module, `ged_ladder.py`, composing existing pieces rather than adding
solvers. `GedlibBackend.bounds()` supplies the bracket, `NetworkxBackend.solve_exact()` supplies the
exact value and the completion decision. A per-rung stratified sampler seeded on
`[seed, rung, dataset_ordinal]`. A write path that refuses to emit a file violating the contract. A
worker that stages, samples, solves, mirrors per rung and resumes.

**Alternatives considered and rejected:**

- **`ExactPlusBoundsBackend` as-is** — it already returns exactly the `PairResult` this ticket wants,
  and it was the obvious choice. Rejected because it takes `ub = min(GEDLIB, A* cost)`. On a
  certified pair that makes the recorded upper bound equal the exact value, so §6's
  `rho(exact, UB)` becomes the correlation of a variable with itself and the mean relative
  overestimate becomes identically zero. It also makes DoD criterion 3 — `lb <= exact <= ub` — a
  tautology rather than a check. Amendment 6 had already verified the effect on 234,258 certified
  AIDS pairs; I read that as a warning rather than as a description of a different pipeline.
- **Materialising rung populations lazily via rank decoding** — I planned closed-form rank-to-pair
  decoding to avoid building a 177 k-row array. Rejected once I measured the actual populations:
  the largest rung block is 177,123 pairs, which is 1.4 MB. Enumerate and index; the decode arithmetic
  is a bug surface bought with nothing.
- **`Generator.choice(m, k, replace=False)` for the draw** — rejected because NumPy documents that
  `Generator` *methods* may change algorithm between releases, while the raw `random()` stream of
  PCG64 is a stability guarantee. Uniform keys plus a stable argsort costs one float64 per population
  element and survives a NumPy upgrade.
- **A bash loop over rungs in the worker, with truncation decided in bash** — rejected because the
  truncation rule would then live outside the tested code and would need to re-read each `.npz` to
  make its decision. The Python takes the whole rung list and truncates internally; the worker calls
  it once and `--mirror-dir` provides the per-rung checkpoint.

**Plan as executed:**

1. Read T-05-design §6 and amendment 7, T-03-design §0, and the four `ged_backends` entry points.
2. Measure the real per-rung, per-dataset pair mass, because the allocation design is only
   meaningful against it.
3. Write `ged_ladder.py`; smoke it on 6 real pairs at a 1 ms budget to exercise the censoring path
   end to end before writing any tests.
4. Launch the rung-13 pilot on real data at the frozen 1,200 s budget.
5. Write the tests while it runs.
6. Write and locally dry-run `worker_ladder.sh`.
7. On the orchestrator's rulings: add `ub_astar_bestsofar`, and measure the `BRANCH_FAST`
   orientation asymmetry.
8. Summarise the pilot, project the cost, write this log.

**Deviations from the plan:** two.

- I added a `--bounds networkx` switch that was not in the brief. It routes the bracket through
  `ged_bounds`' own BRANCH/BP implementations instead of GEDLIB. It exists so the unit suite needs no
  compiled library, and it doubles as the cross-check CLAUDE.md requires between the two
  implementations. Production defaults to `gedlib` and the worker passes it explicitly.
- I set `lb_symmetry_probes=0`, departing from `GedlibBackend`'s default of 32. Reasoning and the
  measurement that justifies it are in §6 and §7.

## 3. Changes made

**Created**

| Path | Purpose |
|---|---|
| `benchmarks/real_data/eval_setup/ged_ladder.py` | The ladder: sampler, allocator, solver, output contract, CLI |
| `tests/unit/test_ged_ladder.py` | 38 tests |
| `slurm/approx_ged/worker_ladder.sh` | Cluster worker. Submits nothing |
| `.claude/notes/2026-08-13-t05-bounds/t05-ladder.md` | This log |

**Modified** — none. **Removed** — none.

**Commits**

| SHA | Message |
|---|---|
| `798e3af` | `feat(T-05): calibration ladder — exact GED above n = 12, interval-censored` |
| `3703027` | `test(T-05): cover the ladder's sampler, censoring, containment and truncation` |
| `941a704` | `feat(T-05): SLURM worker for the calibration ladder` |
| `fbbd1f9` | `feat(T-05): record the A* best-so-far cost beside the bracket, not inside it` |
| `e11038b` | `fix(T-05): keep the directive token out of worker_ladder.sh` |
| `e6c5e66` | `fix(T-05): pin the code commit at process start, not at metadata-build time` |
| (this file) | `docs(notes): t05-ladder work log` |

## 4. Tests

**Tests created or extended** — 38 in `tests/unit/test_ged_ladder.py`. The ones that carry weight:

| Test | What it verifies | The failure mode it catches |
|---|---|---|
| `test_timeout_is_censored_never_promoted` | A 1 ms budget yields `certified=False`, `exact=inf`, finite `lb <= ub` | The T-03 §0 defect: a best-so-far cost recorded as an exact distance. No test on the *value* can catch it |
| `test_write_refuses_a_censored_pair_carrying_a_finite_exact` | The write path rejects the same defect at the file boundary | A future caller constructing records by hand |
| `test_write_refuses_a_certified_pair_carrying_inf` | The biconditional runs both ways | Silent loss of a certified value |
| `test_certified_pairs_are_contained_in_their_bracket` | `lb <= exact <= ub` on a real solve, tolerance 1e-9 | An invalid bound, or a mismatched cost model between the two solvers |
| `test_sampler_is_reproducible_from_the_seed_alone` | Two draws at seed 42 are element-wise identical | Any hidden dependence on time, hash order or process |
| `test_sampler_does_not_couple_datasets` | Nesting when the quota moves, equality when it is pinned | A generator threaded across datasets, which would make one cohort's pairs depend on another's mass |
| `test_reduced_sample_nests_inside_the_full_one` | A smaller quota draws a subset | The property that makes the 25-pair pilot informative about the 250-pair run |
| `test_allocation_*` (6 tests) | Floor, proportionality, caps, zero-mass, short rung, bad input | An allocation that overruns a dataset's population or silently drops a contributor |
| `test_empty_rung_is_reported_not_skipped` + `test_empty_rung_writes_a_valid_file` | An empty rung is data | DoD 1's "reported as empty, not silently skipped" |
| `test_truncation_threshold_selects_the_ceiling`, `test_manifest_*` | 25 % inclusive, ceiling falls back to 12 | An off-by-one at the threshold, or a ceiling asserted rather than measured |
| `test_bestsofar_does_not_move_the_recorded_ub` + 2 siblings | The side column never touches the bracket | The exact conflation ruling 1 exists to prevent |
| `test_anchor_aware_ged_is_unreachable` | The retired solver cannot be constructed | Someone reintroducing the false optimality certificate |
| `test_gedlib_and_ged_bounds_agree_as_bounds` | Both bracket the same A* optimum | CLAUDE.md's mandatory cross-check |

**Coverage of the behaviour that matters:** the sampler, the allocator, the population enumeration,
the censoring decision, every write-time invariant, the truncation arithmetic, and both bounds
backends. Exercised on synthetic cohorts for speed and on real Suite-2 data separately (§6).

**Not tested, and why:**

- **The process pool path.** `run_rung(workers>1)` is exercised only by the real-data runs in §6, not
  by a unit test: spawning a pool inside pytest is slow and flaky, and the pool's only job is to call
  `solve_pair`, which is tested directly. The real risk it carries — that a pair's result depends on
  which worker took it — is addressed by `lb_symmetry_probes=0` and measured in §6.
- **The 1,200 s budget itself.** The censoring test uses 1 ms. The two differ only in the value of a
  float; the branch is the same.
- **`worker_ladder.sh`** has no automated test. It is verified by `bash -n` and by two full local
  dry runs including the resume path (§6).

## 5. Test results

**Command:** `PYTHONPATH=".:/home/mpascual/opt/build_gedlib/graphkit-learn" $PY -m pytest tests/unit/test_ged_ladder.py -q -p no:randomly`

```
============================= test session starts ==============================
platform linux -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/mpascual/research/code/IsalGraph/.claude/worktrees/agent-a6c1b86faa0320e8a
configfile: pyproject.toml
plugins: hypothesis-6.165.2, cov-7.1.0
collected 38 items

tests/unit/test_ged_ladder.py ......................................    [100%]

============================== 38 passed in 4.35s ==============================
```

**Result:** 38 passed, 0 failed, 0 skipped · **Duration:** 4.35 s · **Run at:** `fbbd1f9`

`ruff check` clean on both files.

**Failures and their resolution:** one, during development.
`test_sampler_does_not_couple_datasets` failed on its first formulation. I had asserted that removing
a cohort leaves the survivors' draws *identical*. It does not, and the test was wrong rather than the
code: removing `grec` frees its quota, `coil_del`'s allocation rises from 66 to 76, and it therefore
draws ten more pairs. The invariant that actually rules out coupling is **nesting** — the smaller
draw is a subset of the larger — plus equality once the allocation is pinned. Both are now asserted.
Recorded because the wrong version would have passed by luck on a rung where the two allocations
happened to coincide, and would then have proved nothing.

## 6. Verification beyond unit tests

| Circumstance | What was run | Evidence | Outcome |
|---|---|---|---|
| Real per-rung mass | direct count over the ten exported cohorts | see table below | six datasets contribute at 13–18; Letter and LINUX cap below 13 |
| Sampler on real data | `sample_rung` twice per rung, 13–18, seed 42 | 250 pairs each rung, element-wise identical both runs | pass |
| Censoring path, real data | rung 13, 6 pairs, budget 1 ms | 0/6 certified, `exact` all `inf`, `lb` 4–16, `ub` 19–39, all finite | pass |
| Rung-13 pilot | 25 pairs, budget 1,200 s, 12 workers | §6.2 | see below |
| Worker dry run | `worker_ladder.sh`, rungs 13–14, 4 pairs, budget 2 s | staged, sampled, solved, mirrored, report + manifest written, EXIT trap mirrored 5 files | pass |
| Worker resume | same, second invocation | `[resume] staged rung_13.npz — it will be skipped`; rung skipped, metadata folded into the manifest | pass |
| `BRANCH_FAST` orientation | 9,406 real pairs, 5 datasets, 2 strata | §6.3 | 0 asymmetric |
| Environment | Debian 12, Python 3.11.15, numpy 1.26.4, networkx 3.6.1, GEDLIB via graphkit-learn at `~/opt/build_gedlib`, 24 cores, 31 GB | | |

### 6.1 The real population, measured — and one fact the brief did not state

Pairs with `max(n1, n2) = n`, counted as `C(a,2) + a*b`:

| rung | aids_graphedx | aids_iam | coil_del | grec | mutagenicity | protein | TOTAL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 13 | 2,310 | 4,503 | 177,123 | 27,930 | 21,726 | 185 | 233,777 |
| 14 | 0 | 0 | 162,150 | 6,018 | 43,089 | 598 | 211,855 |
| 15 | 2,319 | 6,018 | 165,898 | 16,213 | 60,030 | 452 | 250,930 |
| 16 | 3,106 | 7,545 | 147,602 | 10,970 | 76,956 | 585 | 246,764 |
| 17 | 4,689 | 9,087 | 166,452 | 24,339 | 108,985 | 435 | 313,987 |
| 18 | 3,935 | 7,600 | 164,710 | 11,590 | 138,014 | 1,521 | 327,370 |

The three Letter cohorts and LINUX contribute **exactly zero** at every ladder rung — they cap at
`n <= 10`, as §7 item 1's analysis rule already says. So the ladder is measured on six datasets, not
ten, and **rung 14 has only four contributors**: neither AIDS cohort holds a 14-node connected graph.
That is not a defect, but it means a per-rung per-dataset comparison across the ladder has a hole at
14, and any sentence of the form "the ladder covers the ten Suite-2 datasets" is false.

Realised allocations, seed 42, 250 pairs, floor 20:

| rung | aids_graphedx | aids_iam | coil_del | grec | mutagenicity | protein |
|---:|---:|---:|---:|---:|---:|---:|
| 13 | 21 | 23 | 118 | 36 | 32 | 20 |
| 14 | — | — | 150 | 25 | 55 | 20 |
| 15 | 21 | 23 | 106 | 29 | 51 | 20 |
| 16 | 22 | 24 | 98 | 26 | 60 | 20 |
| 17 | 22 | 24 | 89 | 30 | 65 | 20 |
| 18 | 21 | 23 | 85 | 25 | 75 | 21 |

Every rung sums to 250 and every contributor clears 20. Note how much the floor is doing: `protein`
holds 0.08 % of rung 13's mass and receives 8 % of the sample. That is the design's intent — a
per-dataset rate needs a per-dataset `n` — but it means the **pooled** rung-13 certification rate is
not the population rate, and must not be quoted as one. It is a stratified estimate whose weights are
the allocation, not the mass.

### 6.2 The rung-13 pilot

25 pairs — a strict subset of the production 250-pair sample, because the draw takes the `k` smallest
uniform keys and the keys depend only on `[seed, rung, ordinal]`, so a smaller quota is a prefix of a
larger one. The reduced per-dataset split also tracks the production one closely (coil_del 48 % vs
47 %, protein 8 % vs 8 %, grec 16 % vs 14 %), so the extrapolation below is not comparing different
populations.

Run at the **frozen 1,200 s budget**, 12 pool workers, Debian 12, i7-13700KF, 24 cores.

```
=== rung 13: 25 pairs, budget 1200.0 s, 12 workers ===
certified 19/25 = 76.0 %   censored 24.0 %
wall 1486.5 s   core-seconds 11634.1

per-pair seconds
  all        n= 25  min     0.00  med   239.32  mean   465.36  p90  1200.00  max  1200.99
  certified  n= 19  min     0.00  med    87.61  mean   233.27  p90   650.05  max  1059.95
  censored   n=  6  min  1200.00  med  1200.00  mean  1200.33  p90  1200.98  max  1200.99

per dataset
  aids_graphedx    n=  2  certified   2 (100.0 %)  median s   203.44
  aids_iam         n=  2  certified   2 (100.0 %)  median s    28.97
  coil_del         n= 12  certified   9 ( 75.0 %)  median s   219.72
  grec             n=  4  certified   4 (100.0 %)  median s   352.35
  mutagenicity     n=  3  certified   2 ( 66.7 %)  median s   284.09
  protein          n=  2  certified   0 (  0.0 %)  median s  1200.00

containment violations on certified pairs: lb>exact 0, ub<exact 0
censored pairs with finite exact: 0
non-finite bounds: 0
certified: mean bracket width 14.42, mean rel LB underestimate 0.156, mean rel UB overestimate 1.370
```

**Five things worth saying about this.**

1. **76 % is much better than §6 expected.** The design's cost table anticipated "the upper rungs
   mostly censored", and rung 13 is not close to the 25 % threshold. The exact-GED ceiling is
   materially above 13, which is the direction AE.1 needs.
2. **The budget is honoured to within one second.** Every censored pair returned between 1,200.00 and
   1,200.99 s. `nx.graph_edit_distance`'s timeout is checked at branch expansion, so an overrun was a
   real risk; on this workload it does not happen. Worst case for sizing is therefore
   `pairs x 1,200 s` with negligible slop.
3. **DoD 3 holds on real data: zero containment violations**, zero censored pairs carrying a finite
   `exact`, zero non-finite bounds. This is the check that would have been vacuous had I folded the
   A\* cost into `ub`, because then `ub == exact` by construction on every certified pair.
4. **`BIPARTITE` is loose here, and that is the AE.1 signal.** On the 19 certified pairs the mean
   relative overestimate of the upper bound is **1.370** — the recorded upper bound averages 2.37x the
   true distance at n = 13 — against a mean relative underestimate of only **0.156** for
   `BRANCH_FAST`. The bracket is asymmetric, and the loose end is the one T-27 §5.4 flagged as growing
   ~10x faster in `n` than any alternative. One rung of 25 pairs is not a result, but it is the first
   direct measurement of that quantity above n = 12 and it points where T-27 said it would.
5. **`protein` certified 0 of 2 and both hit the cap**, while `aids_iam` certified both in a median of
   29 s. At a *fixed* n = 13 the cost spread across cohorts is at least 40x. So a pooled certification
   rate is a weighted average over a very heterogeneous set, and the §7 item 1 confound — size and
   provenance travel together — applies to the ladder's own rates, not only to the bracket-width
   curve.

**Provenance defect found by this run, in my own code.** The pilot's metadata records
`code_commit = e11038b`, but the code that produced it was `798e3af`: `_code_commit()` resolved the
hash when each rung's metadata was assembled, which is *after* the rung has solved, and I committed
three times during the 25 minutes it ran. Same class as the `rsync`-pinned banner that commit
`da9a87d` fixed, arriving from the opposite direction — there the tree was staler than the code, here
it was fresher. Fixed in `e6c5e66`: the lookup is cached and warmed by `main()` before any pair is
solved. **The pilot numbers above are unaffected** — only the recorded hash was wrong — but a
multi-hour cluster ladder is exactly where this would have mattered.

### 6.2.1 Cost projection for rungs 13–18 — an interval, not a point

Model: per-pair uncensored solve time scales by a constant factor `f` per node, so a pair at rung
`13 + k` certifies iff `T13 * f**k < 1200`. The rung-13 empirical distribution supplies `T13`
directly. **The six censored pairs are carried at exactly 1,200 s**, which is all that is known about
them, so every certification rate below is an **upper** bound and every cost an **under**-estimate.
Both biases point the same way and are not corrected.

`f` is anchored on T-03's measured median of 6.5 s/pair at `nbar = 10.56` against the pilot's certified
median of 87.6 s at n = 13, giving **f = 2.90 per node** — *below* the ×5 the design assumed. The two
populations differ (T-03 is Letter/LINUX/AIDS-GraphEdX; the ladder is
coil_del/grec/mutagenicity/protein/AIDS), so this anchors the estimate rather than proving it, and the
table brackets it with f = 2.5 and f = 5.

| rung | cert % @ f=2.5 | core-h | cert % @ **f=2.9** | **core-h** | cert % @ f=5 | core-h |
|---:|---:|---:|---:|---:|---:|---:|
| 13 | **76.0 (measured)** | **32.3** | **76.0 (measured)** | **32.3** | **76.0 (measured)** | **32.3** |
| 14 | 64.0 | 44.4 | 60.0 | 46.3 | 52.0 | 53.7 |
| 15 | 40.0 | 55.3 | 40.0 | 57.1 | 32.0 | 66.9 |
| 16 | 36.0 | 62.8 | 32.0 | 66.7 | 12.0 | 74.6 |
| 17 | 24.0 | 69.6 | 20.0 | 73.5 | 8.0 | 76.7 |
| 18 | 12.0 | 74.3 | 12.0 | 75.5 | 8.0 | 76.8 |
| **first rung < 25 %** | **17** | | **17** | | **16** | |
| **measured ceiling** | **n = 16** | | **n = 16** | | **n = 15** | |
| **cost under truncation** | **264 core-h** | | **276 core-h** | | **228 core-h** | |

**Headline for sizing.** Expect the ladder to stop at **rung 16 or 17**, i.e. a measured exact-GED
ceiling of **n = 15 or 16** — three to four nodes above T-03's 12 — and to cost **230–280 core-hours**
under the truncation rule, **500 core-h absolute worst case** if the rule is disabled and every rung
runs to the full 250 × 1,200 s. Rung 13 alone is **32.3 core-h measured**.

**Why the interval and not a point.** Three independent reasons, all pushing the same way:

- **n = 25 with 24 % right-censoring estimates a heavy-tailed mean badly.** The median is solid; the
  mean, which is what sizes a job, is not. A single hard pair at rung 15 moves the rung's cost by
  ~4 %.
- **The composition shifts across rungs.** coil_del falls 118 → 85 and mutagenicity rises 32 → 75 from
  rung 13 to 18. Mutagenicity is far sparser (density 0.094 against coil_del's much denser
  structure), so the per-node factor is not even constant across the ladder — it is a mixture whose
  weights move.
- **The scaling model is fitted to two points**, one of which comes from a different cohort.

**Practical sizing advice for the orchestrator.** One job, 128 `sr` cores. At the central estimate
276 core-h that is ~2.2 h wall; at the 500 core-h worst case, ~3.9 h. **Request 12 h** — the margin is
nearly free because the job exits when the ladder truncates, and the failure mode of under-requesting
is losing a rung mid-solve. Keep `--resume` so a requeue costs one rung.

### 6.3 `BRANCH_FAST` is exactly symmetric — the probe has never fired

Ruling 2 asked me to settle whether `GedlibBackend`'s `lb_symmetry_probes` default of 32 actually
changes any value. It evaluates the lower bound in both argument orders for the first 32 pairs *of
each backend instance* and keeps the larger, so in a process pool a pair's recorded `lb` would depend
on which worker took it and how early — a rerun at a different worker count could disagree.

Measured through the public `GedlibBackend.bounds` at `compute='lb'`, `lb_symmetry_probes=0`,
`BRANCH_FAST`, `--threads 1`, D6, per-pair env. Both orders on every pair. `ged_backends.py`
untouched.

| dataset | stratum | pairs | asymmetric | max abs(fwd − rev) |
|---|---|---:|---:|---:|
| mutagenicity | uniform | 1,000 | 0 | 0.0 |
| mutagenicity | top-decile n | 1,000 | 0 | 0.0 |
| protein | uniform | 1,000 | 0 | 0.0 |
| protein | top-decile n | 1,000 | 0 | 0.0 |
| coil_del | uniform | 1,000 | 0 | 0.0 |
| coil_del | top-decile n | 1,000 | 0 | 0.0 |
| aids_iam | uniform | 1,000 | 0 | 0.0 |
| aids_iam | top-decile n | 1,000 | 0 | 0.0 |
| linux | uniform | 1,000 | 0 | 0.0 |
| linux | top-decile n | 406 | 0 | 0.0 |
| **TOTAL** | | **9,406** | **0** | **0.0** |

Zero, and identically zero rather than below tolerance. `max(lb, lb_rev) == lb` on every pair tested,
so the default is a **no-op on values**, the running bounds campaigns are unaffected, and my flag is
documentation-only. It also explains the orchestrator's four byte-identical LINUX parity runs
directly rather than by coincidence. **What it does not prove**: 9,406 pairs is a sample, and the
result covers `BRANCH_FAST` under D6 only. Swapping in `BRANCH_TIGHT` or `STAR` for the `lb` role
would require rerunning it.

Keeping the probe at 0 in the ladder is still right: the ladder's value is per-rung comparability, and
a position-dependent tightening — even one that never fires today — would be a latent way to break it.

## 7. Decisions, assumptions, open questions

**Decisions with a real trade-off:**

1. **`lb` and `ub` are the raw GEDLIB bounds; the A\* best-so-far cost is never folded in.**
   Buys: `rho(exact, UB)` and the mean relative overestimate stay measurable, DoD 3's containment
   check stays a real check, and the bracket is a function of the pair rather than of the node.
   Costs: on a censored pair the D11 interval is wider than achievable. Resolved by ruling 1 with a
   separate `ub_astar_bestsofar` column. The decisive argument is reproducibility, not the analysis:
   **how far A\* gets in 1,200 s is a function of the machine**, so a bound built from it moves
   between nodes while `lb` and `ub` do not.
2. **`lb_symmetry_probes=0`.** Buys per-pair determinism independent of pool layout. Costs a
   verification the library performs by default — which §6.3 now shows has never fired.
3. **Uniform keys + stable argsort instead of `Generator.choice`.** Buys stability across NumPy
   releases. Costs one float64 per population element, 1.4 MB at the largest rung.
4. **The floor is applied before the proportional split.** §6 says "proportionally … minimum 20" and
   does not order the two. Floor-first is implemented and documented; proportional-first would have to
   claw pairs back from some donor and there is no principled one. Consequence recorded in §6.1: the
   pooled rate is a stratified estimate, not the population rate.
5. **Per-pair GEDLIB env, not cohort mode.** Cohort mode is worth ~1.3× where the solve is cheap. Here
   the solve is A\*, measured in minutes, so the ~280 µs rebuild is invisible; per-pair keeps the
   ladder on T-03's exact call sequence.

**Assumptions I proceeded on:**

- That the ladder's `lb`/`ub` should be raw. Messaged to `main` when made; approved as ruling 1.
- That `SUITE2_KEYS` in alphabetical order is a valid basis for the per-dataset seed ordinal.
  Nothing else depends on it, but **reordering that tuple silently changes which pairs seed 42
  draws**, so it is documented in the module as part of the reproducibility contract.
- That `--stage ladder` is not needed in the launcher. Confirmed as ruling 3.

**Open questions for the orchestrator:** none outstanding.

## 8. Coordination

**Messages sent:**

1. To `main`, on the raw-bounds decision, the `lb_symmetry_probes` order dependence, and the absence
   of a launcher-hook requirement. Outcome: three rulings.
2. To `main`, reporting the §6.3 asymmetry measurement as zero over 9,406 pairs.
3. To `main`, the pilot result and cost projection (the final message).

**Messages received and how they changed the work:**

- **Ruling 1** approved raw bounds and required a separate `ub_astar_bestsofar` column. Implemented
  in `fbbd1f9` with three write-time invariants: `inf` on every certified pair, never below `lb`, and
  it never moves `ub`.
- **Ruling 2** asked me to measure the `BRANCH_FAST` orientation asymmetry rather than assume its
  severity, and named `mutagenicity` and `protein` as where an asymmetry would most likely appear.
  That is the right instinct and it is why the probe has a top-decile stratum. Result: zero. My
  original framing — "a genuine reproducibility defect if non-zero" — was correct in form and wrong
  in expectation; the orchestrator's indirect evidence from the parity runs was the better prior.
- **Ruling 3** accepted the no-launcher-hook finding.

**Contracts I depend on and confirmed unchanged:**

- `ged_backends.NetworkxBackend.solve_exact` returns `(exact_or_None, best_cost, seconds, timed_out)`
  and sets `exact` only when `astar_completed` is true. Read at `b862f5e`; not modified.
- `ged_backends.GedlibBackend.bounds(g1, g2) -> (lb, ub)`, with the inversion guard inside `_bracket`.
- `export_graphs.load_exported` returns an `ExportedDataset` whose `graphs` order is the pair-index
  order. Not modified.
- `_env.sh` exports `REPO_DIR`, `CONDA_ENV_PREFIX`, `GEDLIB_DIR`, `DATA_DIR`, `OUT_DIR`, `MYLOCAL`,
  `PY`, `run_py`, `START_TIME`, `ROLE_METHOD`, `ROLE_OPTIONS`, and installs the EXIT/TERM/INT traps.
  `worker_ladder.sh` consumes all of these and adds none.

## 9. Deliberately not done

- **No full ladder run.** Rungs 14–18 are cluster work; the brief says so and the projection in the
  final message is what sizes it.
- **No Picasso contact of any kind.** No `ssh`, `sbatch`, `squeue`, `rsync`, `scp` was issued.
- **No §6 analysis outputs** — `rho(exact, LB)`, `rho(exact, UB)`, the bootstrap CI, the OLS of
  bracket width on `n`, the D13 gate. A later track owns them. The ladder produces their input.
- **No change to `ged_backends.py`**, including the `lb_symmetry_probes` default, which is not mine
  to change even though §6.3 measured it.
- **No launcher edit.** Ruling 3.
- **No repair of the `ub` values in T-03's censored AIDS/LINUX intervals** (amendment 6). Owner is
  T-03/T-06.

## 10. Risks and follow-ups

| Item | Severity | Detail | Suggested owner |
|---|---|---|---|
| The pilot is 25 pairs at one rung | medium | Exact-GED cost has a heavy right tail; a 25-pair sample estimates the median far better than the mean, and the mean is what sizes a job. The projection is given as an interval for this reason | orchestrator |
| Letter and LINUX contribute nothing to the ladder | medium | Rungs 13–18 are six datasets, and rung 14 only four. Any claim that the ladder "covers Suite 2" is false, and the per-rung composition shifts (coil_del 118 → 85, mutagenicity 32 → 75) so a raw rung-to-rung comparison partly measures provenance, exactly the §7 item 1 confound | T-06 analysis track |
| `nx.graph_edit_distance`'s timeout is advisory | low | It is checked at branch expansion, so a single expensive expansion can overrun the budget. `astar_completed` still classifies correctly — an overrun is *more* clearly a non-completion — but wall time per pair can exceed 1,200 s and the SLURM wallclock needs headroom | orchestrator |
| `ub_astar_bestsofar` could be mistaken for a bound to analyse | low | Guarded by a metadata note and by the write-time invariants, but it is a column in a file and files outlive their notes | T-06 analysis track |
| Memory under a wide pool | low | A\* on 13–18-node pairs holds a large open list. 12 workers on a 31 GB box was safe; 128 workers on a 450 GB `sr` node is ~3.5 GB/worker, which should hold but is untested at that width | orchestrator |

## 11. Self-assessment against the definition of done

| # | Criterion | Met | Evidence |
|---|---|---|---|
| 1 | Sampler reproducible from seed 42 alone; counts recorded; empty rung reported | yes | `test_sampler_is_reproducible_from_the_seed_alone`; real-data double draw on all six rungs identical; §6.1 tables; `test_empty_rung_is_reported_not_skipped` + `test_empty_rung_writes_a_valid_file` |
| 2 | Censoring honest — induced timeout, `certified=False`, finite `lb <= ub`, `exact = inf` | yes | `test_timeout_is_censored_never_promoted`, plus three write-path refusals; exercised on 6 real pairs at 1 ms |
| 3 | `lb <= exact <= ub` on every certified pair, tol 1e-9, 0 violations | yes | `test_certified_pairs_are_contained_in_their_bracket`; pilot count in §6.2 |
| 4 | Real rung-13 run with per-pair seconds, certification rate, per-dataset split | yes | §6.2 |
| 5 | Measured cost projection for 13–18 with the rung where certification drops below 25 % | yes | §6.2 |
| 6 | `worker_ladder.sh` matches `worker_bounds.sh`'s shape, `bash -n` clean, submits nothing | yes | `bash -n` clean; two local dry runs including resume; no `sbatch` anywhere in the file |
| 7 | Tests cover determinism, floor, proportionality, empty rung, censoring, containment, truncation | yes | 38 tests, mapped in §4 |
| 8 | All work committed, tree clean, log committed | yes | six code commits plus this log; `git status` clean |

**Overall:** I am confident in three things and want them checked in this order.

**Most confident: the censoring is honest, and it is the thing the ticket exists to get right.** The
biconditional `certified <=> exact is finite` is enforced in `solve_pair`, again at the file boundary
in `write_rung_npz`, and tested from both directions plus by an induced timeout. On real data, 6 of
25 pairs timed out and all six carry `exact = inf` with finite bounds. `ANCHOR_AWARE_GED` cannot be
constructed. Completion is decided by `astar_completed`, never by a returned value.

**Confident and worth acting on: the rung-13 numbers.** 76 % certification at the frozen 1,200 s
budget, zero containment violations, and a mean relative UB overestimate of 1.370 against 0.156 for
the LB. The certification rate is much healthier than §6 anticipated, which is good news for AE.1: the
ceiling is materially above 13.

**Least confident, and where the orchestrator should look first: the projection for rungs 14–18.** It
rests on 25 pairs at one rung with 24 % right-censoring, and a per-node scaling factor fitted to two
points, one of which comes from a different cohort. I have given it as an interval — stop at rung 16
or 17, 230–280 core-h under the truncation rule, 500 core-h absolute worst case — precisely because a
point estimate would be the wrong shape of answer. The composition shift across rungs (coil_del
118 → 85, mutagenicity 32 → 75) means the scaling factor is a mixture whose weights move, which no
single `f` captures. Size the job on the worst case; it is cheap because the job exits at truncation.

**Two things I would want a reviewer to challenge.** First, the decision to keep `lb`/`ub` raw is
approved but it is the choice that most shapes what §6 can conclude; if a later track wants the
tightest possible D11 intervals as primary, `ub_astar_bestsofar` is there but is machine-dependent and
that must travel with any number derived from it. Second, §6.1's finding that Letter and LINUX
contribute nothing at any rung, and that rung 14 has only four contributing cohorts, changes what "the
ladder" can be said to cover — it is six datasets, not ten, and the per-rung composition is not
constant. That is a fact about the data rather than a defect, but a sentence in the manuscript that
says otherwise would be wrong.
