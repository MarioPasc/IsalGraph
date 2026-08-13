# T-05 — Bounded GED over Suite 2 on Picasso: frozen design

**Written 2026-08-13, before any production run.** Owner: T-05.
Serves **AE.1** (size-scaling evidence), **R3.5b** (per-dataset primary correlations),
**R3.7a** (the `n` ceiling *with its cause*), and supplies **F1** (the D13 bracket family, 10 tests)
and **F2**'s `B1a`/`B3a` rows (80 tests) with their reference distance.

This file exists so that the method specification, the output contract, the sampling design of the
calibration ladder and the parallelisation choice are on record **before** any number is produced.
Nothing here may be changed after the first production job is submitted without a dated changelog
entry at the bottom.

Inputs: [approx_ged](../plan/approx_ged.md) · [gedlib](../plan/gedlib.md) · [data](../plan/data.md) §1 ·
[statistics](../plan/statistics.md) D6/D11/D12/D13 · [preregistration](../plan/preregistration.md) §3–§4 ·
[decisions](../plan/decisions.md) 3, 4, 11, 12, 21, 25, 27 and §6 ·
**T-27 report** `results/reports/T-27-ged-bound-bakeoff/REPORT.md` · `T-03-design.md`.

Base commit: `34e3ade822ce82424b6fb4d12045b678d56ad798`.

---

## 0. State measured on 2026-08-13, and where it differs from the plan

Every row was read live today, not assumed. **Six differ from what the plan or the board predicted.**

| Item | Plan / board said | Measured 2026-08-13 | Consequence |
|---|---|---|---|
| fscratch file count | 221.0k / 250k soft (T-03, 08-12) | **224.3k / 250k soft, 400k hard**; space 0.47/1.40 TB | 25.7k headroom to soft, 175.7k to hard. This design adds **≈ 40 files**. Not a constraint |
| Picasso `sr` idle | "~45 idle" (T-03) | **42 idle × 128 c = 5,376 idle cores**; 47 pending / 47 running cluster-wide | queue time is not the binding risk; the **2 h floor** is |
| Account's own jobs | — | **3 IsalSR jobs running on `sr`** (40 cores, ~1.5 d left) | do not saturate `sr`; this ticket needs ≤ 2 nodes |
| GEDLIB on Picasso | verified 2026-08-11 | **re-verified today**: `BRANCH_FAST` LB 1.00, `BIPARTITE` UB 1.00, `BP_BEAM --initialization-method BIPARTITE --initial-solutions 1` UB 1.00 on P₄ vs C₄ (true GED 1). numpy 2.4.6, nx 3.6.1 | no build needed; the quota risk of building is **not incurred** |
| Suite-2 exporter | board implies the data is exportable | **does not exist.** `export_graphs.py:52` hardcodes `FILTER_N_MAX = 12` and a 5-key registry. The Suite-2 registry lives in `cohort_audit.py:66-92` + `iam_gxl_loader.py:97`, which **audit** but do not export | Track A must be written |
| IAM source root | T-27 repro line says `$D/source/IAM_Database/extracted` | **that path does not exist.** The tree is `$D/source/APPROX_GED/datasets/IAM_Database/extracted` (35,604 files, 331 MB); GraphEdX/Letter Suite-1 sources are under `$D/source/GED_PRECOMPUTED/datasets` | T-27's §9 repro line is stale; record it. Both roots must be passed |
| `GedlibBackend` options | T-27: "a method name without its options string is no longer a valid specification" | `ged_backends.py:777` emits **only** `--threads {n}`; defaults are `lb="BRANCH_FAST"`, `ub="IPFP"` | the backend **cannot express the selected specification**. Track B must parameterise it |

### Two premises of the board row that this design corrects

1. **"≈ 0.57 core-h for all 21.7 M pairs" is not the cost of this ticket.** That figure is
   `BRANCH_FAST`/`IPFP` at "~100 µs/pair", a number `approx_ged.md` §2 carried from before T-27
   measured anything. T-27 §5 measured **285 µs/pair** for `BRANCH_FAST` and **345 µs/pair** for
   `BIPARTITE` at n̄ = 29.51, and `BIPARTITE` must run in **both orientations**. The naive Suite-2
   projection is **1.7 + 4.2 = 5.9 core-h** on the workstation, and T-27 limitation 3 states these
   are *lower bounds* because the gate was probed at n̄ = 29.5 while Suite 2 reaches n = 98.
   **Nothing is sized from this paragraph** — §5 sizes from a measured rate.

2. **`IPFP` is not the upper bound.** The board row already says so; it is restated here because
   `approx_ged.md` §2's *Production assignment* table still prints `IPFP`, and that table is what a
   careless reader copies. The selected pair is `BRANCH_FAST` / `BIPARTITE`, **with options**.

---

## 1. Method specification — frozen, options included

Cost model **D6**, `edit_cost_constant = [1, 1, 0, 1, 1, 0]`
(`[node_ins, node_del, node_rel, edge_ins, edge_del, edge_rel]`), `CONSTANT` edit cost, every run.

| Role | Method | Options string, verbatim | Accessor | Basis |
|---|---|---|---|---|
| **Lower bound — primary** | `BRANCH_FAST` | `--threads 1` | `get_lower_bound()` | T-27: wins 5/5; **provably equal to `BRANCH` under constant edge costs** and measured identical on all 3,836,827 certified pairs |
| **Upper bound — primary** | `BIPARTITE` | `--threads 1` | `get_upper_bound()` | T-27: wins 5/5 **by elimination** under the frozen M7 gate |
| **Upper bound — sensitivity** | `BP_BEAM` | `--threads 1 --randomness PSEUDO --initialization-method BIPARTITE --initial-solutions 1` | `get_upper_bound()` | T-27 §5.2: misses the gate by **17 %**, and trips D13 on **0 of 5** datasets where `BIPARTITE` trips it on 2 |
| **Upper bound — tightness reference**, §1.1 subsample only | `IPFP` | `--threads 1 --randomness PSEUDO --initial-solutions 10` | `get_upper_bound()` | T-27 §3.2: **tightest of seven**, 13× tighter than `BIPARTITE`. Infeasible on all 21.7 M (×808 the gate, 9,742 core-h); affordable on a size-stratified subsample |

### 1.1 The `IPFP_MS` subsample — design frozen before it runs

**PI decision 2026-08-13**: the sensitivity arm covers `BP_BEAM_DET` on **all** of Suite 2 and
`IPFP_MS` on a **size-stratified subsample**. The subsample exists to answer one question — *how much
tightness does the frozen gate cost, as a function of `n`, in the regime AE.1 disputes* — and its
design is fixed here so the answer cannot be shaped by the result.

| Parameter | Value |
|---|---|
| Stratum | bin of `max(n₁, n₂)`, edges `[2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 99)` → **14 bins** |
| Draw | **uniform within bin**, pooled across the ten datasets, without replacement, **seed 42** |
| Size | `min(2,000, bin population)` per non-empty bin → **≤ 28,000 pairs** |
| Companions | `BRANCH_FAST`, `BIPARTITE`, `BP_BEAM_DET` are **read off the full matrices** at the same pair indices — no recomputation, and the four methods are compared on identical pairs |

> **This is a size-stratified sample, not a random sample of Suite-2 pairs.** It deliberately
> over-weights large `n` — a proportional sample would be ~78 % Letter and COIL-DEL below n = 25 and
> would answer nothing about n = 98. **Every figure from it is reported per bin and never pooled into
> a cohort-level mean.** Stated here so the constraint travels with the number.

Projected cost, scaling T-27's 808 ms/pair at n̄ = 29.51 by `(n/29.51)³` over the bin midpoints, two
orientations, doubled for Picasso: **≈ 93 core-h**. Sized from measurement at launch like every other
role (§5).

**Why the sensitivity arm is not optional.** T-27 §5.4 and decision 26 both hand T-05 two measured
costs of the frozen gate, and both bite hardest exactly where this ticket operates:

- **D13 fires on 2 of 5 Suite-1 datasets under `BIPARTITE`** and removes **8 F2 rows** per firing
  dataset (7 × B1a + 1 × B3a, preregistration §3). Suite 2 has ten datasets and reaches n = 98. If
  D13 fires broadly, `N_actual` collapses and the confirmatory design is gutted. Running the arm
  that fires on none is the only way to establish whether an uninformative bracket is a property of
  *the data* or of *the gate*.
- **`BIPARTITE`'s relative error grows ~10× faster in `n`** than any alternative (+0.294/node on
  AIDS against `IPFP_MS`'s +0.029). AE.1 is precisely the objection that small-graph conclusions
  were licensed to n = 98. The frozen gate selected the bound that degrades fastest in that
  direction; reporting it alone, unqualified, is indefensible.

Both arms are computed and reported. **`BIPARTITE` remains primary** — the PI ruling of 2026-08-13
(T-27 §5.4) stands and is not reopened here. `BP_BEAM_DET` is a **disclosed sensitivity arm**, never
substituted for the primary after the fact.

### Non-determinism

GEDLIB's `LSBasedMethod` defaults to one random start under `REAL` randomness and its upper bounds
change on **91.5–93.6 %** of pairs between runs (T-27 §4.2). Every options string above is pinned;
T-27 measured **0.0000** variation on all 60 pinned cells. Any run whose options string is not
recorded verbatim in the output metadata is invalid.

### Reads that fail silently

`get_lower_bound()` on an upper-bound method returns **0.00** and raises nothing. Guards, all
mandatory:

- **Per read**: reject non-finite; reject `< 0`. Reject `== 0` **unless the pair can attainably have
  distance 0** (`ged_backends.py:402 zero_distance_is_attainable`). Zeros are legitimate and common —
  Suite 1 alone has **306,768 certified off-diagonal pairs with exact GED = 0**, 15.5 % of Letter LOW.
  A blanket `0 < v < inf` assertion would abort the run on correct values.
- **Per campaign, at init**: an accessor probe on P₄ vs C₄ (true GED 1) asserting the method returns
  1.00 through the accessor it is being read with. This is the check that catches a wrong accessor.
- **Per merged matrix**: the fraction of exact zeros off-diagonal is recorded, and a matrix that is
  **≥ 99 % zero off-diagonal** aborts the merge. That is the shape of the silent-zero failure.

---

## 2. Cohort — locked, reproduce or stop

Suite 2, [data](data.md) §1, T-01-certified 2026-08-13. Filter `min_nodes = 2`,
`require_connected = True`, **no `n_max`**, splits merged (decision 3), IAM datasets enumerated by
**split index, not directory** (decision 27).

| # | Key | Source | graphs | pairs |
|---|---|---|---:|---:|
| 1 | `iam_letter_low` | IAM `Letter/LOW` | 1,180 | 695,610 |
| 2 | `iam_letter_med` | IAM `Letter/MED` | 1,253 | 784,378 |
| 3 | `iam_letter_high` | IAM `Letter/HIGH` | 2,059 | 2,118,711 |
| 4 | `linux` | GraphEdX | 89 | 3,916 |
| 5 | `aids_graphedx` | GraphEdX, **no `n_max`** | **819** | 334,971 |
| 6 | `grec` | IAM `GREC/data` | 650 | 210,925 |
| 7 | `aids_iam` | IAM `AIDS/data` | 1,811 | 1,638,955 |
| 8 | `coil_del` | IAM `COIL-DEL/data` | **3,900** | 7,603,050 |
| 9 | `mutagenicity` | IAM `Mutagenicity/data` | 4,040 | 8,158,780 |
| 10 | `protein` | IAM `Protein/data` | 569 | 161,596 |
| | **Total** | | **16,370** | **21,710,892** |

The exporter asserts every count and **exits non-zero** on any mismatch. It does not adjust the
filter. A mismatch stops the ticket and is reported.

> **`aids_graphedx` (819) is a different cohort from Suite 1's `aids` (769).** Suite 1 applies
> `n_max = 12` and drops 51 graphs at n̄ = 18.2. The keys are deliberately distinct so the two
> `.npz` files can never be confused by a loader.

---

## 3. Output contract — frozen

### 3.1 Layout

```
$SANDISK/data/source/APPROX_GED/
  LB/{key}.npz                    10 files — BRANCH_FAST                    (primary)
  UB/{key}.npz                    10 files — BIPARTITE                      (primary)
  UB_SENSITIVITY/{key}.npz        10 files — BP_BEAM_DET                    (disclosed arm, full)
  UB_TIGHT/subsample.npz            1 file — IPFP_MS on the §1.1 subsample  (disclosed arm, sampled)
  manifest.json                     1 file — provenance for all 31
  PROVENANCE.md                     1 file
  ladder/                           calibration ladder (§6)
```

`UB_TIGHT/` holds **one** flat file, not ten: the subsample is pooled across datasets by construction,
so a dense per-dataset matrix would be 99.9 % missing. Its keys are
`dataset_key, pair_i, pair_j, n_max, bin_index, value, value_fwd, value_rev, seconds, metadata` —
the T-27 cell layout, which the analysis already reads.

**File-count discipline.** One file per dataset per role, dense matrices, `savez_compressed`.
Thirty-one data files for 21.7 M pairs × 3 full methods plus a sampled fourth. Shards live on
`$LOCALSCRATCH`, are merged on the
node, and are **deleted** after the merge asserts the structural gate — the T-03 pattern
(`T-03-design.md` §3). The IAM GXL tree (35,604 files) is **never transferred to Picasso**.

### 3.2 Schema — byte-identical key set to the exact-GED files

Every file carries exactly the ten keys of
`GED_PRECOMPUTED/extended_merged_exact_ged/computed/*.npz`, so **one loader reads both**:

| Key | dtype | shape | Contents in a bound file |
|---|---|---|---|
| `ged_matrix` | float64 | (N, N) | **the value this directory is about** — LB in `LB/`, `BIPARTITE` UB in `UB/`, `BP_BEAM_DET` UB in `UB_SENSITIVITY/` |
| `lb_matrix` | float64 | (N, N) | `BRANCH_FAST` lower bound — **the same array in all three files** |
| `ub_matrix` | float64 | (N, N) | `BIPARTITE` upper bound — **the same array in all three files** |
| `certified_mask` | bool | (N, N) | `lb_matrix == ub_matrix` — the pairs where GED is exact for free (approx_ged §4). Diagonal `True` |
| `seconds_matrix` | float32 | (N, N) | wall time **for this file's own method**, both orientations summed for an upper bound |
| `node_counts` | int32 | (N,) | |
| `edge_counts` | int32 | (N,) | |
| `graph_ids` | `<U` | (N,) | `{key}_{split}_{id}` |
| `labels` | `<U` | (N,) | **class label where the dataset has one** (Letter 15, GREC 22, Mutagenicity 2, Protein 6, AIDS 2, COIL-DEL 100); `''` where it does not. Suite-1 files leave this empty; populating it here is free and T-06/T-18 need it |
| `metadata` | `<U` | () | JSON: `dataset, role, method, options_string, accessor, cost_model, n_graphs, n_pairs, filter, splits_merged, n_zero_offdiag, n_certified, seconds_total, gedlib_commit, code_commit, exported_utc, schema_version, picasso_jobid` |

`ged_matrix` carries **the bound, never an interpolation.** approx_ged §4 forbids a midpoint and
this schema offers nowhere to hide one: the three matrices are three measurements, and
`certified_mask` says exactly where two of them coincide.

**Rationale for the redundancy.** `lb_matrix`/`ub_matrix` repeat across the three files so that the
bracket, and therefore `(UB − LB)/UB` and the certification rate, travel with **any** single file.
Compressed cost is a few hundred MB total; the alternative is a loader that must open three files to
answer one question.

### 3.3 Suite-2 graph export (input side)

`$SANDISK/data/source/APPROX_GED/exported_suite2/{key}.npz`, mirrored to
`picasso:/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/isalgraph/suite2/`, in the schema
`export_graphs.py:310 save_exported` already defines — `graph_ids, n_nodes, n_edges, edge_offsets,
edges (CSR, 2×M int32), splits, labels, metadata`. **10 files + 1 manifest.**

---

## 4. Validation gates — all four run before any production pair is written

| Gate | What | Pass condition | Why it exists |
|---|---|---|---|
| **G1 — cohort** | exporter reproduces §2 | all 10 counts exact | T-01's numbers are the only source a printed number may come from |
| **G2 — T-27 reproduction** | on `iam_letter_{low,med,high}` and `linux`, whose Suite-2 cohort is **identical** to Suite 1, the new pipeline's `BRANCH_FAST` and `BIPARTITE` values must equal T-27's `data/cells/{ds}__{CELL}.npz` `value` array | **exact equality on all 3,602,615 pairs**, both cells | the strongest gate available: an independent end-to-end check of loader, cost model, options string, symmetrisation and pair ordering against a census already on record |
| **G3 — bracket validity** | `lb_matrix ≤ ub_matrix` on every Suite-2 pair; and `lb ≤ exact ≤ ub` against T-03's certified values on the four G2 datasets and on the 769-graph `aids` overlap | **0 violations**, compared at `1e-9` | `LB ≤ GED ≤ UB` is the only thing the large-`n` argument rests on |
| **G4 — structural** | every matrix symmetric to machine precision; diagonal zero; every entry finite and `≥ 0`; off-diagonal zero fraction recorded and `< 0.99`; `certified_mask` diagonal `True` | pass or `MergeError` | decision §6.2: an upper-bound matrix filled in one orientation **is not a distance matrix** |

**Symmetrisation.** `BIPARTITE`, `BP_BEAM`, `IPFP` and `REFINE` build an edit path from a *directed*
assignment and are not symmetric. Every upper bound is computed in **both orientations and
minimised** (`ged_backends.py:950-955` already does this), and the result is asserted symmetric
before it is written. `BRANCH_FAST` is symmetric; symmetry is **asserted** for it on a probe rather
than assumed.

**Gate 2 of T-03 is not re-run.** T-27 discharged the two-sided `ged_bounds.py` cross-check on the
archived 400-pair LINUX sample: `BRANCH` 400/400 value-equal, `BIPARTITE` 156/400 value-equal but
passing, GEDLIB's node map attaining our LSAP optimum 400/400. Re-running it would re-measure a
closed question. **G2 above supersedes it at 9,000× the sample size.**

---

## 5. Parallelisation — the choice and why

Four facts decide it, and they point somewhere different from T-03.

1. **The compute is small.** Naive projection from T-27 §5, doubled for Picasso cores being ~2×
   slower than the workstation the figures came from: `BRANCH_FAST` **3.4 core-h**, `BIPARTITE`
   (×2 orientations) **8.4 core-h**, `BP_BEAM_DET` (×2) **28 core-h**, `IPFP_MS` on the §1.1
   subsample **93 core-h** — **≈ 133 core-h**, plus the §6 ladder at ≈ 300 core-h realistic /
   500 worst case. **≈ 430–630 core-h for the whole ticket.** T-03 spent 2,081.
2. **SCBI's two-hour floor is binding, and it is the *only* binding constraint.** Manuel,
   soporte@scbi.uma.es, 2026-08-07. Forty core-hours on one 128-core node is **19 minutes** — three
   orders of magnitude the wrong side of the floor. The design problem here is the opposite of
   T-03's: not how to split the work, but how to keep from splitting it.
3. **The projection is uncertain by several-fold in both directions.** Per-pair cost scales roughly
   as `max(n₁,n₂)³`; T-27's rate was probed at n̄ = 29.51 while Suite-2 datasets run from n̄ = 4.07
   (Letter LOW) to n̄ = 31.68 (Protein) with a tail to n = 98, and Jensen's inequality makes any
   `n̄`-based projection an underestimate on the right-skewed sets. **Nothing is sized from the
   projection.**
4. **Per-pair wall time is a reported quantity** (D12, and T-27 §5 extended to Suite 2), so the node
   family must be pinned or the timing measures the scheduler.

**Therefore: four concurrent single-node jobs, one per role, on `--constraint=sr`, plus the §6 ladder
job, with the core count per job derived at launch from a rate measured on the real Suite-2 data on
a `sr` compute node, such that every job runs ≥ 2 h by construction.**

| Job | Role | Projected core-h | Cores (sized at launch) |
|---|---|---:|---|
| `aged-lb` | `BRANCH_FAST`, all Suite 2 | 3.4 | 1 |
| `aged-ub` | `BIPARTITE`, all Suite 2 | 8.4 | 2–3 |
| `aged-ubs` | `BP_BEAM_DET`, all Suite 2 | 28 | 8–10 |
| `aged-ubt` | `IPFP_MS`, §1.1 subsample | 93 | 28–32 |
| `aged-ladder` | exact A* ladder (§6) | 300–500 | 128 |

Five jobs on at most two `sr` nodes' worth of cores, against 42 idle. The account's three running
IsalSR jobs are untouched.

```
cores = clamp( floor( measured_core_seconds / TARGET_SECONDS ), 1, 128 ),  TARGET_SECONDS = 10800 (3 h)
```

The launcher **refuses to submit** if the projection puts a job under `FLOOR_SECONDS = 7200`; it
reduces `cores` instead, exactly as `slurm/exact_ged/launcher.sh:95-183` already does. If a role's
total work cannot fill 2 h even on one core, **it is merged into the adjacent role's job rather than
submitted short.**

**The rate is measured, not assumed.** A `probe` stage runs first *inside* the same job, on a seeded
stratified sample of 3,000 pairs spanning every dataset and every `n` decile, single-process
`time.process_time()`, and writes `probe.json`. The worker then sizes its own internal worker pool
and its dataset order from that measurement. This keeps the measurement on the hardware that does
the work and avoids a separate short probe job.

Rejected alternatives:

| Option | Rejected because |
|---|---|
| A job array over datasets | Letter LOW is ~90 core-seconds. Nine of ten tasks would be minutes long — the 12,600-task pattern SCBI wrote to this account about |
| A job array over pair-index chunks (T-03's shape) | correct for 2,081 core-h, absurd for 40. Chunking exists in `ged_pair_index.py` and is **retained for resumability inside one task**, not for fan-out |
| One job for all roles | a failure in the 93-core-h `IPFP_MS` arm would take the 3.4-core-h primary lower bound down with it. **The primary deliverable must not depend on the arms** |
| Run it all on the workstation | the two primary roles are ~12 core-h, ~35 min wall at 22 jobs, and need no queue, no rsync, no quota. **Rejected on instruction** — the user directed Picasso; and the arms plus §6's ladder are ~530 core-h, which the workstation cannot absorb. Recorded because it is the honest comparison for the primary pair |

**Within a job**, a `fork`-context pool of `SLURM_CPUS_PER_TASK` workers consumes contiguous
upper-triangle index ranges. Each worker holds **one `GEDEnvGXL` per process per dataset**, built
once — GEDLIB env construction is not free and must never be per-pair. `OMP/MKL/OPENBLAS_NUM_THREADS=1`
and `--threads 1` in every options string: threading is inside our pool, never inside GEDLIB.

**Checkpointing** every 2,000 pairs to `$LOCALSCRATCH`, one file overwritten in place
(`ged_exact_runner.py:584 _write_npz_atomic`), fingerprinted against the owned pair set, with
`TERM`/`INT` traps so a wallclock kill does not lose the shard. A requeue loses minutes.

**Merge** on the same node at the end of the job: shards → dense matrices → G4 → shard deletion →
`cp -a` mirror to `$OUT_DIR`. Output lands in
`/mnt/home/users/tic_163_uma/mpascual/execs/isalgraph/approx_ged` (home quota, 13.8k/35k files).

---

## 6. The calibration ladder — sampling design, frozen

`approx_ged.md` §3.1 item 1 and AE.1 both require it: the bracket is *selected* at `n ≤ 12` and
*licensed* to `n = 98`, and T-27 §8 limitation 1 states plainly that no bake-off against ground
truth can close that gap because exact GED does not exist above 12. **Every node the exact solver
buys narrows it.**

**Population**: all Suite-2 pairs with `max(n₁, n₂) = n`, pooled across the ten datasets, for each
rung `n`. **Seed 42.**

| Parameter | Value | Rationale |
|---|---|---|
| Rungs | `n = 13, 14, 15, 16, 17, 18` | n = 12 is T-03's ceiling; the ladder starts one node above it |
| Pairs per rung | **250**, stratified by source dataset proportionally to that dataset's pair mass at that `n`, minimum 20 per contributing dataset | 250 supports a graph-level bootstrap CI on ρ at each rung without dominating cost |
| Exact solver | **`networkx.graph_edit_distance`**, D6 costs | T-03 amendment 2: `ANCHOR_AWARE_GED` is a randomised heuristic that reports a **false optimality certificate**. It is not used |
| Per-pair budget | **1,200 s** wall, enforced by the pool | bounds worst-case cost exactly |
| Non-completion | **interval-censored `[LB, UB]` under D11**, never dropped, never promoted to exact | `nx.graph_edit_distance(timeout=t)` returns best-so-far and does **not** signal incompleteness (`T-03-design.md` §0). Completion is established by the search terminating, not by a returned value |

**Worst-case cost**: 6 × 250 × 1,200 s = 1,800,000 s = **500 core-h**. Realistic cost, extrapolating
T-03's measured median 6.5 s/pair at n̄ = 10.56 with the ×5-per-node scaling: **≈ 300 core-h**, with
the upper rungs mostly censored. One job, 128 cores, ~2.5–4 h wall. Clears the floor.

**Analysis rule, frozen before the run:**

- Per rung, over the **certified** pairs only: ρ(exact, LB), ρ(exact, UB), mean relative
  underestimate of LB, mean relative overestimate of UB, certification rate `LB = UB`, and
  ρ(Lev, exact) − ρ(Lev, LB) with a graph-level bootstrap CI (D2/D7, 2,000 replicates, seed 42).
- The **censoring rate is reported per rung** (D12) and the ladder is truncated at the first rung
  whose certification rate falls below **25 %**, which is reported as the measured exact-GED ceiling
  rather than asserted.
- **Regression, not assumption of transfer** (§3.1 item 2): OLS of mean relative bracket width on
  `n` over rungs 3–`n_top` (pooling T-27's `n ≤ 12` measurements with the ladder), extrapolated to
  the Suite-2 range **with its prediction interval**, and reported as an extrapolation.
- **The D13 gate is not evaluated on the ladder.** D13 is a per-Suite-2-dataset test (family F1) run
  by T-06 on the full matrices. The ladder informs it; it does not pre-empt it.

**Pre-declared, and this is the point of writing it down now:** if the ladder shows the ρ-gap
growing with `n` beyond the D13 threshold, **the large-`n` extension is reported descriptively** and
the exact-GED results become primary (approx_ged §3, decision rule). That is a legitimate outcome
and it is on record before the measurement.

---

## 7. Analysis deliverables

Computable once the matrices exist; no extra cluster time.

1. **`(UB − LB)/UB` versus `n`, across all of Suite 2** — approx_ged §3.1 item 3, *"the single
   measurement that answers AE.1 most directly"*, and T-27 §5.4 raised it to **T-05's most important
   measurement** because `BIPARTITE`'s error grows ~10× faster in `n` than any alternative.
   Reported per dataset and pooled, primary arm and sensitivity arm side by side, with the
   separation between *"IsalGraph degrades at scale"* and *"our reference degrades at scale"* stated
   explicitly.
2. **Certification rate per dataset and per size stratum** — approx_ged §4 forbids promising a rate
   before T-05 measures it. T-27 measured 1.2–40.2 % for `BIPARTITE` at `n ≤ 12`; this extends it.
3. **Bracket width by size and density stratum** (statistics §8 strata).
4. **Per-dataset cost table** extending T-27 §5 to Suite 2 at real `n`, which is the measurement
   that tests T-27 limitation 3.
5. **Primary-vs-sensitivity bracket comparison**: per dataset, `ρ(Lev, UB_BIPARTITE)` against
   `ρ(Lev, UB_BP_BEAM_DET)` on the same graph-level resamples, so the cost of the frozen gate is
   quantified on Suite 2 rather than inferred from Suite 1.

---

## 8. Acceptance criteria

Each is checkable by a named command or artifact.

1. **Cohort.** `export_graphs_suite2.py --verify-only` reproduces all ten §2 rows exactly and exits 0.
2. **G2.** `BRANCH_FAST` and `BIPARTITE` values on `iam_letter_{low,med,high}` and `linux` equal
   T-27's `data/cells/{ds}__{CELL}.npz` `value` arrays **element-wise, 3,602,615 pairs, both cells**.
3. **G3.** Zero bracket violations over all 21,710,892 Suite-2 pairs, and zero
   `lb ≤ exact ≤ ub` violations over T-03's certified pairs on the overlapping cohorts.
4. **G4.** Every one of the 30 dense output matrices symmetric, zero-diagonal, finite, `≥ 0`,
   off-diagonal-zero fraction < 0.99.
5. **Schema.** All 30 dense files load with the exact-GED loader and carry the ten keys of §3.2 with
   the stated dtypes; `manifest.json` records the options string for every role.
5b. **Subsample.** `UB_TIGHT/subsample.npz` holds ≤ 28,000 pairs, reproducible from seed 42 by
   re-running the sampler, and every one of its pairs is present in all three dense roles.
6. **Ladder.** Rungs 13–`n_top` landed with per-rung certification and censoring rates; the measured
   exact-GED ceiling reported.
7. **Analysis.** Deliverables §7.1–§7.5 written to
   `results/reports/T-05-bounded-ged/`.
8. **Quota.** fscratch file count no higher than 224.3k + 100 after the run.
9. **Suite.** `pytest tests/ -q` at or above the reference state (726 passed / 271 skipped with the
   engine).

---

## 9. Stop-and-ask conditions

I halt and escalate rather than proceed if:

- **G2 fails.** A mismatch against T-27 means the loader, cost model, options string, symmetrisation
  or pair ordering differs from the run that selected the methods. Nothing downstream is meaningful
  until it is diagnosed.
- **G3 finds any bracket violation.** `LB ≤ GED ≤ UB` is the whole argument.
- **The measured Suite-2 rate exceeds the projection by more than 10×**, i.e. > 400 core-h for the
  three roles. That would mean T-27 limitation 3 is severe and the sensitivity arm needs rescoping.
- **The exporter cannot reproduce a §2 count.** T-01 is closed and its numbers are locked.
- **The ladder's certification rate at n = 13 is below 50 %**, which would mean the exact ceiling is
  n = 12 exactly and the ladder buys nothing — a reportable negative result, not a failure.

---

## Changelog

- **2026-08-13** — written, before any production job.
