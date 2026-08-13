# Wave 2026-08-12-exact-ged — orchestrator work log

**Ticket**: T-03, exact GED on Picasso. **Date**: 2026-08-12. **Base**: `29886f8` (wave 1),
`2cf495a` (wave 2). **Branch**: `integration/2026-08-12-exact-ged`.

Design and its amendments: `.claude/notes/review/tasks/T-03-design.md` — **read its Changelog**.
Frozen interfaces: `CONTRACTS.md` in this directory.

---

## 1. What was built

| Agent | Owned | Verdict | Head |
|---|---|---|---|
| `task-export` | `export_graphs.py` + 40 tests | **ACCEPT** | `8dcbc28` |
| `task-gedlib-gates` | `ged_backends.py`, `ged_gates.py` + 95 tests | **ACCEPT** | `ae19cf8` |
| `task-runner` | pair index, sampler, runner, merge + 170 tests | **FIXUP** → fixed | `5ab2d99` |
| `task-gates-v2` (wave 2) | ANCHOR retirement | **salvaged** — died on a session limit, work uncommitted | — |
| `task-slurm` (wave 2) | SLURM scripts | **REDONE by orchestrator** — died on a session limit | — |

**691 unit tests pass**, up from a 386 baseline. Ruff clean.

Two wave-2 agents died mid-task when the account hit a session limit. `task-gates-v2` had finished
`ged_backends.py` and left it uncommitted; I salvaged it, re-anchored the seven tests that asserted
the retired contract, and finished the gate-0 wiring. `task-slurm` had barely started; I wrote the
SLURM layer myself.

---

## 2. Parallelisation — the design and why

**A SLURM job array over contiguous upper-triangle index ranges, `--constraint=sr` (128 c / 450 GB),
64 cores per task, task count derived at launch from a measured per-pair rate.**

Three facts forced it.

1. **SCBI's two-hour floor is binding.** A job must run ≥ 2 h or it costs the scheduler more to place
   than to run, and this account has already been written to about a 12,600-task campaign of
   minute-long jobs. The launcher derives the task count from `--sec-per-pair` and **exits 3** rather
   than submitting a short task. This is why LINUX does not run alone — 18 minutes on 8 cores — and
   why the small suite bundles four datasets into one job.
2. **Per-pair wall time is a *reported* quantity** for the D12 censoring analysis, so the node family
   must be pinned or the timing measures the scheduler. `sr` has the largest homogeneous idle pool
   (~45 nodes measured today).
3. **The per-pair rate was unknown** until measured, and the plan's 12–20 s/pair came from a solver
   we have now retired.

Rejected: **one fat `bc` node** — a single point of failure for a 1,000-core-hour run, and only ~8
idle. **One task per pair or per small block** — violates the floor by three orders of magnitude.
**Chunking by graph row `i`** — rows hold `n − i − 1` pairs each, so tasks would be badly unbalanced;
contiguous *linear* upper-triangle ranges are equal by construction and split evenly with the
remainder on the low tasks, never a ragged tail.

**Dry run at 15 s/pair**: small suite 4 cores / 2.26 h · stage 1 37 cores / 2.49 h · stage 2 **7
tasks × 64 cores / 2.75 h per task** · 448 concurrent cores against a 9,000-core QOS cap. Every task
clears the floor.

---

## 3. Numbers that came out different from the plan

| Quantity | Plan | Measured 2026-08-12 |
|---|---|---|
| fscratch file count | 399.7k / 400k, "EXCEEDED" | **221.0k** — the emergency had passed |
| GEDLIB build tree | 50–90k files | 55,410, of which 52,185 in `include/` |
| `export_graphs.py` | listed as surviving | **did not exist**; written here |
| local data root | `…/research/isalgraph/data/source` | **moved** to `…/ISAL/completed/isalgraph/…` |
| queue time | "the real risk, unbudgeted" | **zero** — `--test-only` reports an immediate start on `sr014` |
| `ANCHOR_AWARE_GED` | exact (decision 11) | **refuted** — see §4 |
| GraphEdX published AIDS GED | the gate-0 oracle | **approximate upper bound** — see §4 |
| LINUX cost | 2.17 s/pair, 2.4 core-h | **2.708 s/pair, 2.95 core-h** (measured, all 3,916 pairs) |
| AIDS cost | 12–20 s/pair | median **6.5 s**, mean **10.1 s** among completions, 27 % over 30 s |

Two further defects found while reading code, neither in any audit:

- **`ged_computer.py::compute_ged_pair` records timed-out searches as exact.**
  `nx.graph_edit_distance(timeout=t)` returns its best-so-far cost; the function returns it unless
  it is `None`. Every "exact GED" in the submitted study may be an uncertified upper bound.
- **`graphedx_loader.py::_strip_node_attributes`** adds nodes as `range(n)` then adds edges with
  *original* labels. Inert today (all 911 AIDS and 89 LINUX graphs verified `0..n-1`), silent if it
  ever fires. Flagged, not fixed — outside this ticket.

---

## 4. The two findings that changed the plan

### `ANCHOR_AWARE_GED` is not an exact solver

| Evidence | Result |
|---|---|
| Same real AIDS pair, 6 fresh `GEDEnvGXL` instances | e.g. `[10, 6, 6, 6, 6, 4]`; brute force = **2** |
| Non-determinism across 15 real AIDS pairs | **14 / 15** |
| Wrong vs exhaustive brute force, 18 small pairs | **4 / 18**, always over, never under |
| `networkx` A* on the same oracle | **0 / 18 wrong** |
| `--threads 1`, `--map-root-to-root`, `--search-method DFS` | none restore it |

It is a randomised upper-bound heuristic that reports `LB == UB` — a **false optimality certificate**,
worse than a wrong number because it defeats the check meant to catch one. The 2026-08-11 verification
behind decision 11 was **one trivial pair** (P₄ vs C₄), which it passes by luck.

**PI-authorised replacement**: exact = `networkx` A* run to completion; GEDLIB keeps `BRANCH_FAST`
(LB) and `IPFP` (UB). Certification is decided by **whether A\* completed**, never by a solver's
self-report. `ANCHOR_AWARE_GED` and `HED` are guarded by name and asserted absent from every
`set_method` call — **no core-hour is spent on either**, per the PI's direction.

### GraphEdX's published AIDS GED matrix is approximate

208 certified pairs: **150 ours-lower, 58 equal, 0 ours-higher**, mean Δ = −1.58, max −8.
Independently re-verified on 9 fresh pairs: **5 proved suboptimal, 0 in the falsifying direction** —
GraphEdX publishes **11** for train pair (76, 211) while A* found an **achievable** path of cost **6**
in 0.50 s. GED is a minimum and we exhibited a cheaper achievable path.

The one-sidedness is what separates "their reference is approximate" from "our solver is buggy": a
buggy solver errs in both directions. Gate 0 is therefore re-anchored on brute-force enumeration, and
the GraphEdX comparison is demoted from a gate to a **reported finding**.

> **This upgrades the recompute from a cost-model unification to a correctness fix.** The submitted
> LINUX ρ = 0.433 and AIDS ρ = 0.349 were computed against an approximate reference *and* are
> within-split figures. That belongs in the R3.5a/R3.5b response as a result, not a caveat.

---

## 5. Gate outcomes

| Gate | Where | Status |
|---|---|---|
| **1** bracket validity | local, `networkx` backend, 24 pairs | **PASS**, 0 violations |
| **2** archived 400-pair LINUX replay | local, 25 pairs | **PASS** |
| **3** exact-solver agreement + benchmark | local, 24 pairs | **PASS**, 0 disagreements |
| **probe, 1, 2, 3 on GEDLIB** | Picasso job **1967801** | see below |

### Picasso gates run, job 1967801 (sr008, AMD EPYC 7H12, 4 cores)

**GEDLIB works on the compute node**: `P4/C4 BRANCH_FAST lb=1.0`, and `isalgraph` correctly absent.
**Gate 3 runs against real GEDLIB** — 500 cross-dataset pairs, ~2.2 s/pair, environment built once
with per-pair `restart_env`. Three failures, each with a distinct and useful cause:

1. **Gate 1 — the invariant-1 guard fired**:
   `BRANCH_FAST.get_lower_bound returned 0.00 for a pair whose distance cannot be zero`.
   Two readings and they must be separated before production:
   - the documented accessor trap is real for this pair, or
   - **the guard is a false positive on a genuinely isomorphic duplicate.** AIDS and IAM Letter both
     contain isomorphic duplicates, for which `LB = 0` is correct. The guard's "cannot be zero" test
     must therefore be an isomorphism check, not a node/edge-count comparison.

   The distinction matters: reading 1 invalidates GEDLIB as a bound source; reading 2 is a defect in
   our own guard. **Diagnose by taking the offending pair and testing `nx.is_isomorphic`.**
2. **Gate 2 — `No module named 'torch'`.** It replays the archived LINUX sample through
   `validate_ged_bounds.load_pairs`, which reads GraphEdX `.pt` files. Torch is deliberately absent
   from the cluster, which is the entire reason the datasets arrive pre-serialized. Gate 2 must
   either load LINUX from the CONTRACT A export or run on the workstation. It **passed locally**.
3. **Gate 0** — same structural cause; runs on the workstation by design.

> **The split is now clear and should be written into the plan**: gates that read GraphEdX's `.pt`
> ground truth (0 and 2) are **workstation** gates; gates that exercise GEDLIB (probe, 1, 3) are
> **cluster** gates. No single machine can run all four, and the earlier `--gate all` design assumed
> one could.

⚠ **The job header prints `Git commit: d6a9f4b`, which is stale.** The rsync excludes `.git`, so the
cluster's git metadata predates the code it is running. The *code* is current; the *provenance line*
is wrong, and a provenance line that lies is worse than none. Fix by exporting the local SHA through
`--export` rather than reading git on the far side.
| **0** GraphEdX report | local | **runs**; the finding is established (§4), the packaged report is unfinished |
| **4** structural | in `ged_merge_shards` | **PASS** on the real LINUX matrix |

**Gate 2's history is worth keeping**: it first failed on 30/100 pairs whose replay UB sat *below* the
archive, then was diagnosed — the archive predates the `min`-symmetrisation and holds
forward-orientation-only bounds. It matches forward 100/100. Not a disagreement; the measured gain
from invariant 3.

---

## 6. Production status

**LINUX is complete and verified** — the local end-to-end smoke computed all 3,916 pairs:

```
gate 4 passed: 3911 certified, 5 censored, 0 certified zeros, max asymmetry 0
symmetric True   diag0 True   GED 1..16, mean 5.20
certified 3911/3916 (99.9%)   censoring rate 0.13% at a 60 s timeout
2.708 s/pair mean, 60.0 s max, 2.95 core-h
```

That is one of the five Suite-1 datasets delivered, in CONTRACT D form with all ten keys, consumable
unchanged by `eval_correlation.py`, `method_comparator.py`, `dataset_filter.py` and `validator.py`.

**Data staged and verified on Picasso**: five `.npz` plus `manifest.json`, **55 KB across 6 files**,
replacing a 6,767-file source tree. All sha256 match, all counts match, **5,350 graphs /
3,897,911 pairs**. fscratch 221.0k → 222.3k files, entirely the repo sync.

**Timeout is 60 s**, the submission's `ged_computer` default, unchanged per the plan and recorded here
explicitly. Censoring is reported per stratum, never pooled (D12).

---

## 6b. Production launched 2026-08-12 16:00 UTC — Suite 1, all five datasets

| Job | Content | Resources | Projected |
|---|---|---|---|
| **1972177** | Letter LOW/MED/HIGH + LINUX, all 3,602,615 pairs | 4 c | ~2.3 h |
| **1972178** | AIDS stage 1 — **22,051 pairs over 769/769 graphs** | 50 c | ~2.5 h |
| **1972179** | AIDS stage 2 census, 295,296 pairs, array `0-9%8` | 10 × 64 c | ~2.6 h/task |

Sized from **20 s/pair**, the conservative reading of the real-AIDS measurement (median 6.5 s, mean
10.1 s among completions, 27 % over 30 s) at a 60 s timeout. Every task clears the two-hour floor.

**Stage 1 on the real cohort**: 22,051 pairs — core 16,110, halo +5,860, top-up +81 — covering
**769/769 graphs**, with **35 of 90 strata non-empty**. The synthetic dry run gave 90/90; real AIDS
is sparse and its densities concentrate, so most size × density cells are genuinely empty in the
*population*. The sampler reports them empty rather than topping them up, which is the specified
behaviour. The requirement that binds — every graph represented — holds exactly.

### Why production is not gated on the GEDLIB gates

**GEDLIB is not in the production path.** The workers run `--backend networkx`: A* for the exact
value and our own `ged_bounds.py` for the bracket. GEDLIB's exact role was retired by amendment 2,
and its bound role belongs to T-05's calibration ladder, not to computing these matrices. Blocking
production on gates that validate an unused component would have been the wrong dependency, so the
`afterok` chain was cut and job **1972154** continues as a standalone T-05 diagnostic.

What does gate production, and its evidence:

| Check | Evidence |
|---|---|
| A* is exact | brute-force enumeration, **18/18**; local gate 3, **0 disagreements** |
| bracket validity | local gate 1 **PASS**, non-vacuous; gate 2 **PASS**; archived 400-pair gate 2, **0 violations**, 35 unit tests |
| structural | gate 4 **PASS** on the complete real LINUX matrix |

### Two defects the cluster gate run exposed, both ours

1. **The zero-guard rejected a valid lower bound.** `BRANCH_FAST` returning `0.00` is the *trivial*
   bound — always valid, merely uninformative — and my own probe had already recorded it on real
   pairs with true GED 2 and 6. The plan's `0 < v < inf` rule exists to catch an accessor
   *mismatch*, which can only manifest on the **upper** bound. Fixed; gate 1 then passed. The rate of
   zero lower bounds is now counted as the bound-quality statistic it actually is.
2. **Gate 1 "PASSED on 0 pairs" — vacuously.** With a bounds-only GEDLIB backend there is no `exact`
   to bracket, so it evaluated nothing and reported success. **A gate that can pass on an empty set
   is not a gate.** It needs `ExactPlusBoundsBackend`, and it needs a minimum-pair assertion.
   Open — see §7.

---

## 7. What is not done

1. **Gate 1 must be made non-vacuous.** It passed on **0 pairs** because a bounds-only backend
   supplies no `exact`. Give it `ExactPlusBoundsBackend` and assert a minimum pair count, so an
   empty evaluation can never read as a pass. **This is the highest-priority follow-up**: it is the
   one defect that could let a future run start on an unvalidated bracket.
2. **The gate probe fails on 1 pair.** Its report was still on the node's `$LOCALSCRATCH` when the
   job was running; read `gates/gateprobe.json` from the mirrored output and diagnose.
3. **The brute-force oracle gate is specified and only partly implemented.** Its *content* is
   established — my own enumeration put `networkx` A* at 18/18 — but it is not yet a repeatable gate.
4. **The gate set needs splitting in the plan**, not just in the worker: gates 0 and 2 read GraphEdX
   `.pt` ground truth and are **workstation** gates; probe, 1 and 3 exercise GEDLIB and are
   **cluster** gates. No single machine runs all four, and `--gate all` assumed one could.
5. **Stage 1's `K/q/f` were not recomputed** from the final measured rate. Defaults 180/10/30 stand
   and produced 22,051 pairs ≈ 122 core-hours at 20 s/pair, close enough to the ~100 core-hour
   budget that re-deriving them would have changed the pre-registered design for no gain.
6. **The job header prints a stale git SHA** (`d6a9f4b`), because rsync excludes `.git`. The code is
   current; the provenance line is not, and a provenance line that lies is worse than none. Export
   the local SHA through `--export` instead of reading git on the far side.
7. **The census supersession decision is still pending**, by design: stage 2 replaces stage 1 only if
   it lands before the T-20 freeze, decided on the calendar and never on the values.

---

## 8. Decomposition retrospective

The wave-1 split into export / backend+gates / index+sampler+runner held: **no merge conflict**, and
the only integration failure was a test asserting `ged_backends` was *absent*, which was true on the
runner's branch and false after merge. That is the cost of the disjointness rule, and it is cheap.

Two contract defects were found by the agents, not by me, and both were real: the merge CLI could not
write CONTRACT D because four of its keys exist in no shard, and CONTRACTS §7's `0 < v < inf`
contradicted §5's allowance for isomorphic zeros. A third — the unspecified `searchsorted` side —
would have silently emptied the top density quintile on the real cohort. **Freezing contracts in
advance did not prevent defects; it made them visible early and cheap.**

The wave-2 failure mode was new: both agents died on an account-level session limit, not on the task.
The lesson is that agent work must be **committed incrementally**, not at the end — `task-gates-v2`
had finished a substantial module and left it uncommitted, and it survived only because a worktree
persists after the agent does not.

---

## 9. COMPLETE — all five Suite-1 datasets, 2026-08-13

Twelve SLURM jobs, all `COMPLETED`. **~2,081 core-hours, ~6.5 h wall, zero requeues, zero failures.**

| Dataset | graphs | pairs | certified | censored | core-h |
|---|---:|---:|---:|---:|---:|
| IAM Letter LOW | 1,180 | 695,610 | 695,610 (100 %) | 0 | 1.2 |
| IAM Letter MED | 1,253 | 784,378 | 784,378 (100 %) | 0 | 1.5 |
| IAM Letter HIGH | 2,059 | 2,118,711 | 2,118,711 (100 %) | 0 | 11.5 |
| LINUX | 89 | 3,916 | 3,870 (98.83 %) | 46 | 5.8 |
| AIDS | 769 | 295,296 | 234,258 (79.33 %) | 61,038 | 2,060.6 |
| **Total** | **5,350** | **3,897,911** | **3,836,827 (98.43 %)** | **61,084 (1.57 %)** | **≈ 2,081** |

Every matrix symmetric, diagonal zero, gate 4 passed. **The pair total reproduces the locked cohort
exactly.** AIDS is 99 % of the cost, as the plan predicted; the total came in 26 % above the plan's
upper estimate of 1,650 core-h, because `sr` is ~2× slower per core than the machines the plan's
per-pair figures were measured on.

**Stage 1 and stage 2 agree on their 22,051-pair overlap** — the merge asserts no conflicting value
on any pair index present in more than one shard, and it passed. That was the designed verification
of decision 21's two-stage structure.

### Censoring is a reported result, not a footnote

| Dataset | censoring at 60 s |
|---|---:|
| Letter LOW / MED / HIGH | **0 %** |
| LINUX | 1.17 % |
| AIDS | **20.67 %** |

⚠ **The censoring rate is hardware-dependent and this must be stated wherever it is printed.** The
same LINUX cohort censored **5 / 3,916 (0.13 %)** on the workstation and **46 / 3,916 (1.17 %)** on
`sr` — a 9× difference from nothing but a slower core against a fixed 60 s wall. A censoring rate is
therefore a property of *(cohort, timeout, machine)*, never of the cohort alone.

### Deliverable

`GED_PRECOMPUTED/extended_merged_exact_ged/` — 38 MB, 9 files:
`computed/{5}.npz` (ours), `reference/{aids,linux}_graphedx.npz` (GraphEdX, within-split),
`manifest.json`, `PROVENANCE.md`. Mirrored at `results/exact_ged/` and in
`execs/isalgraph/exact_ged` on Picasso.

### The finding that came out of building it

Testing the plan's premise rather than inheriting it **retracted amendment 3** — see the design
note's changelog. GraphEdX's matrix uses **unit node costs, the same as D6**, so the earlier
"their reference is approximate" conclusion was the arithmetic of a wrong cost model. Like-for-like
over the full overlap: **0 pairs where ours exceeds theirs**, agreement on all but 2 of 131,148.

That error was mine twice over: I drew the conclusion from a gate configured off the plan's premise,
then "independently verified" it with a script that inherited the same premise. **An independent
check that reuses the original assumption is not independent**, and the tell was available all along
— 77,739 supposedly-equal AIDS pairs had differing node counts, which is impossible if one side
charges for node operations and the other does not.
