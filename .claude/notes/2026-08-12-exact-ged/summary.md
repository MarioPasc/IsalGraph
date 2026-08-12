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
| **1, 2, 3 on GEDLIB** | Picasso job **1967743** | submitted, running |
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

## 7. What is not done

1. **Gates 1–3 have not yet returned from Picasso** (job 1967743). Production is gated on them.
2. **The brute-force oracle gate is specified and only partly implemented.** Its *content* is
   established — my own enumeration put `networkx` A* at 18/18 — but it is not yet a repeatable gate.
3. **Letter LOW/MED/HIGH and AIDS have not been computed.** The scripts, sizing and gates are in
   place; they were not submitted because gate 1–3 results on GEDLIB must land first.
4. **Stage 1's `K/q/f` have not been recomputed** from the final measured rate. Defaults 180/10/30
   stand; the recomputation holds the ~100 core-hour budget and the `K : q : f` ratios.

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
