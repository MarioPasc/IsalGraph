# T-03 — Exact GED on Picasso: frozen design

**Written 2026-08-12, before any production run.** Owner: T-03. Serves R3.5a, R3.5b, R1.3a, AE.1, E2/F2.

This file exists so that the sampling design, the supersession rule and the parallelisation choice are
on record **before** any number is produced. Nothing here may be changed after the first production
job is submitted without a dated changelog entry at the bottom.

Inputs: [exact_ged](../plan/exact_ged.md) · [gedlib](../plan/gedlib.md) · [data](../plan/data.md) §1 ·
[statistics](../plan/statistics.md) D6/D11/D12/§8 · [decisions](../plan/decisions.md) 3, 11, 12, 21.

---

## 0. State measured on 2026-08-12, and where it differs from the plan

Every one of these was read live, not assumed. **Four differ from what the plan predicted.**

| Item | Plan | Measured 2026-08-12 | Consequence |
|---|---|---|---|
| fscratch file count | 399.7k / 400k hard — "EXCEEDED" | **221.0k / 250k soft, 400k hard** | 179k headroom. The quota emergency the plan designs around **has passed**. T-23's prune is still worth doing but blocks nothing |
| GEDLIB build tree | 50–90k files | **55,410**, of which **52,185** in `gklearn/gedlib/include` | the prune returns ~52k files; `ext/` is already absent |
| GEDLIB working | verified 2026-08-11 | **re-verified today** — `ANCHOR_AWARE_GED` on P₄ vs C₄ = 1.00 / 1.00 | no build needed; the quota risk of building is **not incurred** |
| `export_graphs.py` | "already does this"; listed as surviving in data.md §6 | **does not exist anywhere on this machine** | must be written. data.md §6's surviving-scripts list is wrong on this entry |
| local data root | `…/research/isalgraph/data/source` | **`…/research/ISAL/completed/isalgraph/data/source`** | the tree was moved. `eval_setup.py:75 DEFAULT_SOURCE_DIR` is stale and points at nothing; `validate_ged_bounds.py` already has the correct path |
| Picasso CPU families | — | `sd` 52c/187G · **`sr` 128c/450G, ~45 idle** · `bc` 256c/700G, ~8 idle · `bl` 128c/1900G | pin `sr`: largest homogeneous idle pool |
| queue state | "queue time is the real risk" | 64 idle CPU nodes, 47 pending jobs cluster-wide | queue time is **not** currently the binding risk |

### A defect found while reading the code, not in any audit

`ged_computer.py::compute_ged_pair` passes `timeout=` to `nx.graph_edit_distance` and returns
`float(ged) if ged is not None else inf`. **`nx.graph_edit_distance(timeout=t)` returns its
best-found-so-far cost when the budget expires — it does not raise and does not return `None` unless
*no* complete edit path was found at all.** So a pair that timed out mid-search is recorded as an
exact value, indistinguishable from a certified one. Every "exact GED" matrix in the submitted study
was produced this way.

This is a further, independent argument for GEDLIB as primary: `ANCHOR_AWARE_GED` returns **both**
bounds, and `LB == UB` is a *certificate* of optimality. Under D11 a pair without that certificate is
interval-censored `[LB, UB]` rather than silently promoted to exact. **The new pipeline records the
certificate per pair.** Logged here because it changes what the phrase "exact GED" meant in the
submission, and R3.5a's ladder must not inherit the ambiguity.

---

## 1. Cost model — non-negotiable

| Use | `edit_cost_constant` | Order |
|---|---|---|
| **Production** (D6) | `[1, 1, 0, 1, 1, 0]` | `[node_ins, node_del, node_rel, edge_ins, edge_del, edge_rel]` |
| **Gate 0 only** (GraphEdX's own model) | `[0, 0, 0, 1, 1, 0]` | idem |

Gate 0 under the production model produces a guaranteed mismatch that reads exactly like a solver
bug. The two constants are named `UNIT_COSTS` and `GRAPHEDX_COSTS` and already exist in
`ged_bounds.py::EditCosts`; the GEDLIB path reuses the same objects via `.as_gedlib_constant()`, so
the two implementations cannot drift apart.

---

## 2. Cohort — locked, reproduce or stop

Filter `min_nodes=2, require_connected=True, n_max=12`, splits merged (decision 3).

| Dataset | graphs | pairs |
|---|---:|---:|
| IAM Letter LOW | 1,180 | 695,610 |
| IAM Letter MED | 1,253 | 784,378 |
| IAM Letter HIGH | 2,059 | 2,118,711 |
| LINUX | 89 | 3,916 |
| AIDS (GraphEdX) | 769 | 295,296 |
| **Total** | **5,350** | **3,897,911** |

The export step asserts these counts and **exits non-zero** on any mismatch. It does not adjust the
filter. A mismatch stops the ticket and is reported.

---

## 3. Parallelisation — the choice and why

Three facts decide it.

1. **SCBI's two-hour floor is binding.** Manuel, soporte@scbi.uma.es, 2026-08-07: a job must run
   ≥ 2 h or it costs the scheduler more to place than to run. This account has already been written
   to about it. Any design producing short tasks is out, however convenient.
2. **Per-pair wall time is a reported quantity** (D12 censoring analysis), so the node family must be
   pinned or the timing measures the scheduler rather than the solver.
3. **The per-pair rate is unknown until measured.** The 12–20 s/pair figure in exact_ged.md §2 is
   `networkx` A* at n ≈ 10.6 (data.md §4). GEDLIB `ANCHOR_AWARE_GED` is specialised C++ and
   exact_ged.md §1 requires benchmarking it first. If it is materially faster the whole cost model of
   this ticket changes.

**Therefore: a job array over contiguous upper-triangle index ranges, on `--constraint=sr`
(128 cores, 450 GB, AMD EPYC 7H12 @ 2.6 GHz, ~45 idle), 64 cores per task, with the task count
derived at launch from the measured per-pair rate so that every task runs ≥ 2 h by construction.**

```
n_tasks = clamp( floor( total_pairs * sec_per_pair / (cores_per_task * target_seconds) ), 1, ∞ )
```
with `target_seconds = 9000` (2.5 h, giving margin over the 2 h floor). The launcher refuses to
submit if the projection puts a task under 2 h; it reduces `n_tasks` instead.

Rejected alternatives, with reasons:

| Option | Rejected because |
|---|---|
| One fat `bc` node (256 cores) + `ProcessPoolExecutor` | a single point of failure for a 1,000+ core-hour run; a node loss costs everything since the last checkpoint. Only ~8 bc nodes idle |
| One task per pair, or per small block | violates the 2 h floor by three orders of magnitude; this is precisely the 12,600-task pattern SCBI complained about |
| Chunking by graph row `i` | rows have wildly unequal cost (`n − i − 1` pairs each), so tasks would be badly unbalanced. Contiguous **linear** upper-triangle index ranges are equal-sized by construction |

**Chunking.** Pairs are indexed by the linear upper-triangle index
`k(i,j) = i·n − i(i+1)/2 + (j − i − 1)`, `0 ≤ k < C(n,2)`, inverted in closed form. Task `t` of `T`
owns `[t·C/T, (t+1)·C/T)` split **evenly** with the remainder spread over the low tasks — never a
fixed block with a ragged tail, since the short remainder task is the thing the floor forbids. Ranges
are contiguous, so a chunk is resumable from its own checkpoint and shards merge deterministically by
index.

**Within a task**, a `ProcessPoolExecutor` of `SLURM_CPUS_PER_TASK` workers consumes the chunk.
Each worker holds one `GEDEnvGXL` built once per process — GEDLIB env construction is not free and
must not be per-pair.

**Checkpointing.** Every 2,000 pairs per task, written to `$LOCALSCRATCH` and mirrored to the shard
directory. A requeue loses at most ~2,000 pairs ≈ minutes.

**`$LOCALSCRATCH`.** Used, per the skill's default. This workload writes **few large files**, not
thousands of small ones, so the mandatory-staging rationale does not apply — but the per-task shard
plus checkpoints still belong on the node, with a whole-tree mirror back on exit and `TERM`/`INT`
traps so a wallclock kill does not lose the shard.

**Merge.** A dependent `afterany` job concatenates shards into one `.npz` per dataset, asserts the
matrix is symmetric and complete, and **deletes the shards**. Output stays at ~30 files total for the
whole GED programme (exact_ged.md §5.1).

---

## 4. Stage 1 — the sampling design, frozen

Stage 1 is the **pre-declared reported analysis** (decision 21). Its design is fixed here, before it
runs, so that the choice between a stage-1 and a stage-2 ρ cannot be made after seeing either.

**Population**: the 769 AIDS graphs surviving the Suite-1 filter. **Seed 42** throughout.

Stage 1 must satisfy two requirements that pull in opposite directions. It must span **all 769
graphs** — because exact_ged.md §3's whole argument is that effective sample size is governed by the
number of graphs, and a stage 1 built on a subset of graphs would concede the opposite. And it must
support the **D2 graph-level cluster bootstrap**, which resamples graphs and recomputes ρ over the
*induced* pair submatrix — which has holes unless the induced submatrix is complete.

**Resolution: a complete core block plus a stratified halo.**

| Component | Definition | Pairs |
|---|---|---:|
| **Core** | **Simple random sample** of `K = 180` graphs, seed 42; **all** `C(180,2)` pairs computed | 16,110 |
| **Halo** | For each of the 589 non-core graphs, `q = 10` partners drawn uniformly from all 769 | ≤ 5,890 |
| **Top-up** | Any non-empty pair-stratum holding `< f = 30` sampled pairs is filled to `f` by uniform draw within the cell | ~1,000–2,500 |
| **Total** | | **≈ 22,500–24,500** |

At 16 s/pair that is **100–109 core-hours** — the ~100 core-h stage-1 budget. `K`, `q` and `f` are
recomputed once from the *measured* rate before stage 1 is submitted, holding the core-hour budget
fixed rather than the pair count; the recomputation is logged and the ratios `K : q : f` are held.

**The core is a simple random sample, not a stratified one.** A stratified core would estimate ρ over
the stratified-subsample population rather than over the 769, and would need design weights to get
back. A simple random sample of graphs makes the core-block ρ **exactly unbiased** for the population
ρ and makes the D2 bootstrap exact on the core. Stratification is therefore applied only to the halo
and top-up, whose job is coverage, not estimation.

**Pair strata** (from statistics.md §8, adapted to pair level, AIDS-internal):
- size cell = unordered pair of node-count bins over `{2–5, 6–9, 10–12}` → 6 cells;
- density cell = unordered pair of AIDS-internal density quintiles → 15 cells;
- stratum = their cross product; empty cells are not topped up and are reported as empty.

### Analysis rule, frozen

- **Primary stage-1 ρ = Spearman over the core block only**, with the D2 graph-level cluster
  bootstrap (2,000 replicates, percentile CI, seed 42) resampling the `K` core graphs. Complete
  induced submatrix, no holes, no weights.
- Halo and top-up carry **coverage**, not the point estimate: per-stratum ρ (exploratory per §8),
  the demonstration that all 769 graphs and all non-empty strata are represented, and the
  per-stratum censoring rate (D12).
- A secondary ρ over **all** sampled pairs is printed beside the core estimate, labelled as such.

### Supersession rule, frozen

> If the stage-2 census completes **and passes gates 1 and 4** before the T-20 text freeze, the
> reported AIDS ρ is the census value and stage 1 becomes a methodological consistency check. If it
> has not completed and been verified by the freeze, the reported AIDS ρ is the stage-1 **core-block**
> value.
>
> **The decision is made on the calendar, not on the values.** No comparison of the two ρ values may
> precede the freeze check. Both are printed in the response letter whichever is primary, and the
> letter states that this rule was fixed on 2026-08-12, before either stage ran.

Stage 2 **seeds its checkpoint from stage 1's results**, so shared pairs are computed once. Identical
values on the shared pairs is itself an internal consistency check and is asserted at merge.

---

## 5. Validation gates

All four run before any production pair is computed. Each returns a pass/fail exit code and writes a
JSON record.

| Gate | What | Cost model | Pass condition |
|---|---|---|---|
| **0** | ~500 within-split AIDS pairs recomputed and compared to GraphEdX's published matrix | **`[0,0,0,1,1,0]`** | exact agreement |
| **1** | `LB ≤ exact ≤ UB` on every gate pair, GEDLIB `BRANCH_FAST` / `ANCHOR_AWARE_GED` / `IPFP` | `[1,1,0,1,1,0]` | 0 violations |
| **2** | Replay of the archived 400-pair LINUX sample (`gate2-linux-400-seed42.json`) through GEDLIB, compared to `ged_bounds.py` | `[1,1,0,1,1,0]` | agreement, or a disagreement diagnosed and attributed to one implementation |
| **3** | `ANCHOR_AWARE_GED` vs `networkx` A* on a shared cross-dataset sample | `[1,1,0,1,1,0]` | exact agreement |

**Gate 3 doubles as the required benchmark** (exact_ged.md §1): it times both solvers on the same
pairs, stratified by `n`, and its output is what sizes stages 1 and 2. If `ANCHOR_AWARE_GED` is
materially faster than `networkx`, that is the measurement which may raise the exact-GED ceiling above
n = 12, and it is reported whether or not we act on it in this revision.

**Gate 0 caveat, stated in advance.** Agreement under GraphEdX's pseudometric model validates *the
solver*, not our cost model — D6 justifies the cost model separately. And if GraphEdX's published
values are themselves approximate rather than exact, gate 0 can fail for a reason that is not our
bug; the gate therefore records the *signed* discrepancy distribution, not just a boolean, so that
"ours is systematically ≤ theirs" (they are upper bounds) is distinguishable from noise.

**Gate 4, structural, asserted at merge on every matrix**: symmetric to machine precision; diagonal
zero; every off-diagonal entry either `0 < v < inf` or explicitly flagged censored with its
`[LB, UB]`. Non-computable pairs are **interval-censored, never dropped** (D11).

**Upper-bound direction dependence.** `BIPARTITE`, `IPFP`, `REFINE`, `BP_BEAM` build an edit path
from a directed assignment and are not symmetric. Every UB is computed in **both orientations and
minimised**; the result is asserted symmetric before it is written. `ANCHOR_AWARE_GED` and the
`BRANCH*` lower bounds need no such treatment, and symmetry is asserted for them too rather than
assumed.

**Every GEDLIB read is asserted `0 < value < inf`.** An upper-bound method returns
`get_lower_bound() = 0.00` and `HED` returns `inf`; neither raises. `HED` is not used.

---

## 6. Execution order

| # | Job | Content | Cores | Est. wall | Gate |
|---|---|---|---:|---|---|
| 0 | local | export 5 datasets to one file each, assert §2 counts, rsync to fscratch, verify checksums | — | min | counts |
| 1 | `ged-gates` | gates 0–3 + the solver benchmark | 4 | ~2.5–3.5 h | **blocks everything** |
| 2 | `ged-small` | Letter LOW/MED/HIGH + LINUX production, all pairs | 8 | ~2–3 h | end-to-end proof on real data |
| 3 | `ged-aids-s1` | AIDS stage 1, ~23k pairs | 64 | ~2.5 h | reported analysis |
| 4 | `ged-aids-s2` | AIDS stage 2 census, array, seeded from stage 1 | 64 × T | ~2.5 h/task | supersedes per §4 |
| 5 | `ged-merge` | `afterany` merge, gate 4, shard deletion, rsync to `execs/` | 2 | min | structural |

Jobs 3 and 4 are submitted **together** (decision 21). Job 2 is the "small real Picasso job that
completes end to end before AIDS is submitted" the ticket requires — it is LINUX plus the three
Letter sets rather than LINUX alone, because LINUX alone on 8 cores is 18 minutes and would violate
the 2 h floor.

**Output** → `/mnt/home/users/tic_163_uma/mpascual/execs/isalgraph/exact_ged` (home quota, 13.7k/35k
files — ample). **Input** → `fscratch/datasets/isalgraph`. Serialized datasets are **one file each**;
the IAM Letter GXL tree (6,767 files) is never transferred.

---

## 7. Acceptance

1. Cohort counts reproduce §2 exactly.
2. Gate 0 passes under `[0,0,0,1,1,0]` before production starts.
3. Gates 1 and 3 pass; GEDLIB agrees with `ged_bounds.py` on the archived 400-pair sample, or the
   disagreement is diagnosed and attributed.
4. Every matrix symmetric; every value `0 < v < inf` or explicitly censored.
5. Stage 1 landed and analysable independently of stage 2.
6. fscratch file count no higher than 221.0k ± 100 after the run.
7. Results in `execs/isalgraph/exact_ged` and mirrored locally.

---

## Changelog

- **2026-08-12** — written, before any production job. No entries yet.
