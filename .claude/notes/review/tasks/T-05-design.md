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
| **G3 — bracket validity** | `lb_matrix ≤ ub_matrix` on every Suite-2 pair; and `lb ≤ exact ≤ ub` against T-03's certified values on the four G2 datasets and on the AIDS overlap **joined on `graph_ids`** (below) | **0 violations**, compared at `1e-9` | `LB ≤ GED ≤ UB` is the only thing the large-`n` argument rests on |
| **G4 — structural** | every matrix symmetric to machine precision; diagonal zero; every entry finite and `≥ 0`; off-diagonal zero fraction recorded and `< 0.99`; `certified_mask` diagonal `True` | pass or `MergeError` | decision §6.2: an upper-bound matrix filled in one orientation **is not a distance matrix** |

**Symmetrisation.** `BIPARTITE`, `BP_BEAM`, `IPFP` and `REFINE` build an edit path from a *directed*
assignment and are not symmetric. Every upper bound is computed in **both orientations and
minimised** (`ged_backends.py:950-955` already does this), and the result is asserted symmetric
before it is written. `BRANCH_FAST` is symmetric; symmetry is **asserted** for it on a probe rather
than assumed.

> **G3's AIDS arm joins on `graph_ids`, never positionally** (amendment 2). Suite-2
> `aids_graphedx` has **819** graphs and Suite-1 `aids` has **769**; a positional comparison would
> silently compare unrelated graphs. Measured: the 769 are a **strict subset** of the 819 — overlap
> exactly 769 — which is structural, since Suite 1 *is* Suite 2 plus `n_max = 12`. So the arm is
> **not skipped**: the id join recovers the full **295,296-pair** `lb ≤ exact ≤ ub` check, the
> largest such arm in the cohort outside Letter.

> **Censored pairs are marked with `inf`, not `NaN`** (amendment 2). T-03's `ged_matrix` holds 92
> non-finite entries on `linux` (2 × 46 censored) and **zero** NaNs. A filter written as
> `np.isnan(...)` passes all 92 straight through, and `inf <= x` evaluates False while raising
> nothing. **Select on `certified_mask` first, and filter with `np.isfinite`.**

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

   > ### ⚠ Analysis rule, frozen 2026-08-13 before any production pair — **size and provenance are
   > confounded, and the pooled curve alone cannot answer AE.1**
   >
   > Measured by `wave-t05-export` from the completed bin table: **bins 0–2 are ~90 % Letter and
   > bins 8–13 are 50–97 % Mutagenicity + COIL-DEL** (bin 13 `[80,99)` is **97.1 % Mutagenicity**
   > alone). Density moves with provenance across the same range — 0.607 on Letter HIGH against
   > **0.094** on Mutagenicity. So a single curve fitted across all fourteen bins fits **a dataset
   > transition and a density transition as faithfully as a size one**, and no sampling design can
   > remove this: it is a property of which real datasets contain large connected graphs.
   >
   > **Therefore the primary measurement is the within-dataset slope, not the pooled curve.**
   >
   > - **Primary**: fit `(UB − LB)/UB` on `max(n₁,n₂)` **within each dataset separately**, with the
   >   D2 graph-level bootstrap. Four datasets span enough `n` to carry an unconfounded slope on
   >   their own — **Mutagenicity** (n̄ 28.53, to 98), **COIL-DEL** (21.54, to 77), **AIDS-IAM**
   >   (14.02, to 85) and **Protein** (31.68, to 96). Letter and LINUX cap at n ≤ 10 and constrain
   >   only the small-`n` end.
   > - **Secondary**: within (dataset × density stratum) where the cell is populated
   >   ([statistics](../plan/statistics.md) §8 strata), which separates the size effect from the
   >   density effect that travels with it.
   > - **Pooled**: reported as a descriptive overlay only, **carrying the confound in its caption**,
   >   never as the estimate a conclusion rests on.
   >
   > This is the same per-dataset-primary / pooled-demoted structure D5 already imposes for R3.5b,
   > applied to the size-scaling curve. Freezing it now is what stops the pooled curve being
   > preferred later because it happens to look cleaner.
   >
   > **The 97.1 % figure travels with any bin-13 number.** `bin_table.json` carries the dominance
   > share per bin precisely so that a top-of-range claim cannot be quoted without it.
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
9. **Suite.** `pytest tests/unit/ -q` at or above the **measured** pre-wave baseline. CLAUDE.md's
   "726 passed / 271 skipped" is stale — the suite has grown since. Measured by the orchestrator on
   a clean main checkout at `885d98d`, before any agent's work was merged:

   > | `PYTHONPATH` | failed | passed | skipped |
   > |---|---:|---:|---:|
   > | without GEDLIB | **8** | **864** | **44** |
   > | **with** `~/opt/build_gedlib/graphkit-learn` | **8** | **907** | **1** |

   **The 43-test gap is GEDLIB availability, not a regression.** Those tests skip when the in-place
   GEDLIB build is off the path. A branch measured with GEDLIB and compared against a baseline
   measured without it appears to have gained 43 tests and lost 43 skips out of nowhere; **the
   comparison must hold `PYTHONPATH` fixed.** This bit once already in this wave — two tracks
   reported skip counts of 44 and 1 for the same suite.

   **All 8 failures pre-date this wave** and are `tests/unit/test_export_graphs.py`'s real-data
   tests, red because of the path defect in amendment 1 finding 3. The merge criterion is therefore
   *no new failure and no lost pass*, not "green" — and any claim that this wave broke the suite must
   be checked against this table first.

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

- **2026-08-14, amendment 8 — `BRANCH_FAST` is symmetric in practice, so
  `lb_symmetry_probes` has never fired. A suspected reproducibility defect, measured and closed.**

  `GedlibBackend` evaluates the lower bound in **both** orientations for the first
  `lb_symmetry_probes = 32` pairs of each backend instance and keeps `max(lb, lb_rev)`. In a
  process pool that makes a pair's recorded `lb` depend on **which worker took it and how early**,
  so a rerun at a different `--workers` count would assign the both-orientation treatment to a
  different set of pairs. Raised by `t05-ladder` while three campaigns were already running.

  **Measured, `BRANCH_FAST` / `--threads 1` / D6, both argument orders on every pair:**

  | | pairs | asymmetric | max \|fwd − rev\| |
  |---|---:|---:|---:|
  | mutagenicity, coil_del, protein, aids_iam, linux — uniform **and** top-decile `n` strata | **9,406** | **0** | **0.0** |

  Two strata per dataset so an asymmetry appearing only at large `n` could not hide; the top-decile
  pool for Mutagenicity and Protein reaches the top of the Suite-2 size range. **Identically equal,
  not equal within tolerance.**

  **Independently corroborated end-to-end**: the LINUX census reproduces T-27 byte-for-byte at
  `--workers 4`, `--workers 1`, in per-pair mode and in cohort mode. Under the asymmetry hypothesis
  those four runs give the both-orientation treatment to different pairs, so **identical sha256
  across all four is only explicable if the two orientations always agree.** Two independent lines
  of evidence, one direct and one end-to-end.

  **So `max(lb, lb_rev) == lb` always, the probe is a no-op on values, and the running campaigns are
  unaffected.** Record it as a verification that has never fired, not as a tightening.

  > **Scope, and it is a real limit.** 9,406 pairs is a sample, not the 21.7 M census, and it covers
  > **`BRANCH_FAST` under D6 only**. The symmetry follows from a constant-edge-cost assignment
  > problem; **it does not transfer to `BRANCH_TIGHT` or `STAR`**, which are different algorithms. A
  > later ticket swapping the `lb` role must re-measure before assuming it.

- **2026-08-14, amendment 7 — the measured cost table. This supersedes §5's projections; do not
  quote §5's core-hour figures.** Every rate below is from a Picasso `sr` core (AMD EPYC 7H12) in
  **cohort** env mode, anchored on IAM Protein's 161,596 pairs, which all four roles ran.

  | Role | Method | Protein, measured | Suite-2 total | §5 said | miss |
  |---|---|---:|---:|---:|---:|
  | `lb` | `BRANCH_FAST` | **14.0 ms/pair** | ~47 core-h | 3.4 | 14× |
  | `ub` | `BIPARTITE` ×2 | — | ~113 core-h | 8.4 | 13× |
  | `ubs` | `BP_BEAM_DET` ×2 | **163.7 ms/pair** | ~381 core-h | 28 | 14× |
  | `ubt` | `IPFP_MS` ×2, 28,000-pair subsample | *ratio estimate* | **~886 core-h** | 93 | **9.5×** |
  | | | | **≈ 1,590 core-h** | 133 + ladder | |

  **Every method costs ~13–14× more at real Suite-2 sizes than T-27's §5 gate probe implied, and the
  factor is remarkably consistent across three independently measured methods.** That is T-27
  limitation 3 — *"the cost gate is evaluated at n̄ = 29.5 … the Suite-2 projections in §5 are lower
  bounds on true cost"* — quantified. The probe was **160 graphs with `25 ≤ n ≤ 35`**; Suite 2 runs
  to `n = 98`. **A future ticket should read that limitation as "×13", not as a caveat.**

  **The measured method ratios also drift from T-27's**, in the same direction:

  | Ratio | T-27 §5 | measured on Protein |
  |---|---:|---:|
  | `ubs` / `lb` | 8.15× | **11.7×** |

  > ⚠ **`ubt`'s 886 core-h is a RATIO estimate, not a measurement, and is the weakest number in this
  > ticket.** It chains the measured `ubs` rate through T-27's `IPFP_MS`/`BP_BEAM_DET` ratio of
  > **696×**, and T-27's `IPFP_MS` figure came from the same narrow band that just proved a ×13
  > underestimate for `BRANCH_FAST`. The caveat is written into
  > `datasets/isalgraph/suite2/probe_measured.json` itself, not only here, so it cannot be read as
  > measured. Sized at 128 cores / 6.92 h projected into a **24 h** wallclock — 3.5× headroom — with
  > checkpointing every 2,000 pairs, so a 3× miss costs wall time, not work.

  **Total ≈ 1,590 core-h against §5's 430–630.** Under T-03's 2,081 and under the ~5,000 threshold
  at which this ticket escalates, so the campaign proceeded; recorded because the arm scope was
  approved on the smaller figure.

- **2026-08-14, amendment 5 — ⚠ AMENDMENT 4'S DIAGNOSIS IS WITHDRAWN. The cost is real; the cause
  I gave for it was wrong.** Amendment 4 attributed the ~33× gap against T-27's 285 µs/pair to
  per-pair GEDLIB environment rebuild, and I asked the PI to authorise a fix on that basis.
  `t05-cohort-env` challenged it, and I settled it by measuring on Picasso rather than adjudicating
  between two local runs — the machine was the confound.

  **Picasso, IAM Protein, 2,000 seeded pairs, one process, same pairs through both paths:**

  | | µs/pair | share |
  |---|---:|---:|
  | per-pair total | 9,502 | |
  | ├ env setup (`restart_env` + 2 × `add_nx_graph` + `init` + `set_method`) | 2,077 | **22 %** |
  | └ **`BRANCH_FAST` solve** | **7,424** | **78 %** |
  | cohort mode | 7,366 | |
  | **speed-up** | | **1.29×** |

  Reproduced independently by the agent across three roles and three datasets: the saving is a flat
  per-pair constant **equal to the bare rebuild** — 276.4 µs measured directly on Protein against
  278–287 µs observed, 185.1 µs on Mutagenicity against 183–190 µs, agreeing within 3 % across roles
  whose totals span 6×. So the gain is largest where the solve is cheapest: `lb` 1.79–5.77×,
  `ub` 1.36–2.09×, `ubs` **1.08–1.33×**.

  **The real cause of the gap** is the one T-27 stated and I under-weighted. Its cost probe was
  **160 graphs with `25 ≤ n ≤ 35`**, while Protein's real pair population runs to `n = 96` with its
  mass in the `[30,50)` bins. T-27 limitation 3: *"the Suite-2 projections in §5 are lower bounds on
  true cost."* I quoted that sentence in this note and still built a projection that ignored it.

  **A second hypothesis was also wrong** and is recorded so neither is repeated: the agent proposed
  that the gap was `--compute both` invoking `IPFP`. The cancelled job's own metadata says
  `compute: lb`, `method: BRANCH_FAST`, `ub_matrix` all `+inf` — `IPFP` was never invoked.

  **What survives.** Amendment 4's *measured* numbers stand: 18.58 ms/pair on Protein, the per-bin
  curve, and **70 core-h for `lb`** summed against the bin table. Cohort mode is kept — parity-exact
  on two machines, ~1.3× on the expensive datasets, one less per-pair failure surface — but it is
  **opt-in** (`--env-mode`, default `per-pair`) so T-03 is untouched, and it does **not** materially
  change the campaign's cost. **The campaign is ~810 core-h, not ~25.**

- **2026-08-14, amendment 6 — `IPFP` at T-03's default options is irreproducible, and it reaches
  the published D11 intervals.** `--ub-options "--threads 1"` leaves `IPFP` on GEDLIB's
  `--randomness REAL` default. Measured over five fresh runs per string:

  | Population | `--threads 1` | frozen `--randomness PSEUDO --initial-solutions 10` |
  |---|---|---|
  | LINUX, all 3,916 pairs | **74.2 %** of values change, max spread 10 | **0.0 %**, 5/5 bit-identical |
  | AIDS, 400 censored pairs | **82.0 %** change, max spread 6 | **0.0 %**, 5/5 bit-identical |

  Consistent with T-27 §4.2, which measured GEDLIB's `LSBasedMethod` upper bounds changing on
  **91.5–93.6 %** of pairs at library defaults. The spurious `P₄`/`C₄` accessor-probe failures
  (10/40 per-pair, 5/40 cohort) are the same defect surfacing at init, so **any `--compute both`
  campaign at T-03's defaults can abort on a probe that is not actually broken.**

  **The blast radius is bounded, and orchestrator-verified rather than argued.**
  `ExactPlusBoundsBackend` takes `ub = min(IPFP, A* cost)`, so on a certified pair the recorded upper
  bound *is* the exact value. Checked directly against T-03's census: `ub_matrix == ged_matrix` on
  **all 234,258 certified AIDS pairs and all 3,870 certified LINUX pairs**. Every censored pair
  carries a non-finite `ged` with finite bounds. **So the exposure is exactly the D11 interval upper
  ends — 61,038 AIDS and 46 LINUX — and nothing else.** A rerun today reproduces T-03's recorded `ub`
  on ~47 % (AIDS) and ~67 % (LINUX) of them; the lower ends (`BRANCH_FAST`) were 5/5 identical.

  Not fixed here: T-03's file is a closed ticket's artifact. **Owner: T-03 / T-06.** The repair is
  61,084 pairs under the frozen `PSEUDO` string, not 21.7 M — hours, not a campaign.

- **2026-08-13, amendment 4 — §5's cost model was wrong by ~20×, and the cause is a conformance
  failure against §5's own text. Measured on Picasso, not projected.**
  ⚠ **Its diagnosis is superseded by amendment 5; its measurements stand.**

  Job `1990832` (`aged-lb`, 1 core, `sr008`) ran the real campaign far enough to measure two datasets
  before being cancelled. **IAM Protein: 161,596 pairs in 3,002 s on one core = 18.58 ms/pair** at
  n̄ = 31.68. T-27 §5 measured `BRANCH_FAST` at **285 µs/pair** at n̄ = 29.51. That is a **~33× gap at
  the same graph size**, and it is not cluster slowness.

  **Cause.** `GedlibBackend.bounds()` calls `_fresh_env()` → `env.restart_env()` **per pair**
  (`ged_backends.py:~1006-1014`) and re-adds both graphs, so all 21,710,892 pairs rebuild the GEDLIB
  environment. T-27's bake-off builds **one env per dataset** and calls `run_method(i, j)`, which is
  where its 285 µs comes from. Invisible in T-03, where a pair cost ~6.5 s of exact A* and the
  rebuild was noise; at ~100 µs of actual solving it *is* the cost.

  > **§5 already required the opposite** — *"Each worker holds one `GEDEnvGXL` built once per process
  > — GEDLIB env construction is not free and must not be per-pair."* The requirement was written and
  > never verified against the implementation, which inherited T-03's per-pair pattern. **Stating a
  > performance requirement in a design note is not the same as checking it.**

  **The measured cost curve.** Per-bin mean seconds from Protein's own `seconds_matrix`
  (161,596 real timings, binned on `max(n₁,n₂)` with the §1.1 edges):

  | bin | `[2,4)` | `[10,12)` | `[20,25)` | `[30,40)` | `[50,60)` | `[80,99)` |
  |---|---:|---:|---:|---:|---:|---:|
  | ms/pair | 1.49 | 3.19 | 8.50 | 15.74 | 27.39 | 40.53 |

  **Log–log slope 1.12**, where `BRANCH_FAST` is `O(n²Δ² + n³)`. The near-linearity is the signature
  of the defect: a ~1.5 ms/pair fixed cost dominates the solver at every size in this cohort.

  **Four estimates of the same quantity, and why they differ** — `lb` over all 21,710,892 pairs:

  | Estimate | Value | Why it is wrong or right |
  |---|---:|---|
  | Board / `approx_ged.md` | 0.57 core-h | predates T-27; ~100 µs/pair was never measured |
  | §5 projection | 3.4 core-h | T-27's rate, which assumes cohort-mode env |
  | naive `n̄³` scaling | 46 core-h | ignores Jensen and the fixed cost |
  | worker's flat probe | 96 core-h | **mean rate × 21.7 M** — the probe is equal-per-bin and over-weights large `n` ~22×, exactly as `wave-t05-export` warned |
  | **measured, summed per bin** | **70 core-h** | Protein's measured per-bin curve against the real bin table |

  Extrapolating by T-27's method ratios, the four roles as-implemented cost **~810+ core-hours**
  against **~25** with env reuse. **PI decision 2026-08-13: fix env reuse before the campaign runs.**
  Parity is checkable rather than hoped-for — T-27 ran cohort-mode and the per-pair runner reproduces
  it byte-identically, so the two modes are *already known to agree* on 3,916 LINUX pairs.

  **Two smaller consequences.** The worker's probe print projects a flat mean × total pairs, a method
  already established as wrong by ~22× on this cohort; it is advisory only (the launcher sizes per
  bin) but should not compute a number that alarming a way we know is invalid. And the Picasso
  checkout is populated by `rsync`, so `git rev-parse` names whatever was last *pulled* there — the
  banner announced `d6a9f4b` while running code eleven commits ahead. `ISALGRAPH_CODE_COMMIT` now
  takes precedence; **provenance that names the wrong commit is worse than none, because it looks
  checkable.**

- **2026-08-13, amendment 3 — two negative results from wave `2026-08-13-t05-bounds`, both
  correcting something this project had written down. Orchestrator re-measured both.**

  **(a) The lazy `zero_ok` change buys nothing, and CONTRACTS §6.1's stated reason for it — which
  I wrote — is factually wrong.** §6.1 asserted that the eager guard "reaches `nx.is_isomorphic`
  whenever `n₁ == n₂ and m₁ == m₂` — … a VF2 call on ~30-node graphs for COIL-DEL and
  Mutagenicity, 21.7 M times". Measured by `wave-t05-runner`: the `(n, m)` precheck short-circuits
  **before** VF2 on **99.5 % of COIL-DEL and 99.4 % of Mutagenicity pairs** (25/5,000 and 32/5,000
  get past it), and where it *is* entered often — Letter LOW at 23.9 % — the graphs are `n ≤ 7`.
  The whole guard costs 0.58–1.03 µs/pair against a GEDLIB solve of 865–938 µs/pair, **0.1 % of the
  work**, so no reordering of it can yield more than ~1.001×. Measured speed-ups: COIL-DEL 0.998×,
  Mutagenicity 1.004×, AIDS-IAM 1.005×.

  > **The change stays** — it is behaviour-identical and strictly not-more-work — but it is **not a
  > performance improvement**, no job is sized on it, and §6.1's rationale must not be repeated
  > anywhere. My error was assuming equal `(n, m)` is common at `n ≈ 30`; the joint distribution is
  > wide there and it is rare. It is common only where the graphs are tiny and VF2 is trivial.

  **(b) The upper bound's orientation asymmetry is not what `decisions.md` §6 records, and the rate
  falls with graph size.** §6 states "tighter on **33.2 %** of pairs, mean gain **1.15** edit
  operations", from **our own BP implementation on 400 LINUX pairs at n̄ = 8.71**. Measured with the
  **production** method (`BIPARTITE`, `--threads 1`), re-verified by the orchestrator:

  | Population | asymmetric | mean \|fwd−rev\| (all pairs) | (asymmetric only) | max |
  |---|---:|---:|---:|---:|
  | LINUX, **all 3,916** pairs | **22.8 %** | 0.737 | 3.24 | 12 |
  | Mutagenicity, 4,000 pairs, `n ≤ 98` | **11.2 %** | 0.335 | 3.00 | 18 |

  **The rate roughly halves from n̄ = 8.71 to n̄ = 28.5 while the per-asymmetric-pair magnitude does
  not** (3.24 → 3.00). The two figures differ in *implementation* and *population* at once, so
  neither difference alone explains the gap with §6 — what is solid is the **direction in `n` under
  one fixed method**, which is new information §6 does not contain. **§6's number must be rewritten
  from the `ubt` subsample, which spans `n = 2…98` across all ten datasets, not patched.** Retire
  the 400-pair BP figure from any cohort-level claim.

  Reassuring, and worth keeping: reverse-tighter 12.5 % against forward-tighter 10.2 % on LINUX is
  near-balanced, which is what an undirected cohort should show. **Pair order carries no
  information, so the symmetrisation is doing its job and nothing upstream is mis-ordered.**

  **(c) `--compute lb` / `--compute ub` give 1.81× / 1.28×, not 2× each.** `both` is one lower-bound
  solve plus **two** upper-bound solves (both orientations), so dropping the upper end removes 2 of
  3 solves and dropping the lower end removes 1 of 3, with fixed per-pair overhead flattening both
  below their solve-count ideal. Measured on all 3,916 LINUX pairs: `both` 129.6, `lb` 71.5, `ub`
  101.1 µs/pair. **Budget the `ub`/`ubs` campaigns at ~78 % of a two-sided run.**

- **2026-08-13** — written, before any production job.

- **2026-08-13, amendment 1 — three defects in the wave contracts, found by `wave-t05-export` during
  recon and verified by the orchestrator before any production pair was computed.** Full text in
  `.claude/notes/2026-08-13-t05-bounds/CONTRACTS.md` §2.1 and §5.

  1. **`graph_ids` is the loader's native id.** CONTRACTS §2 originally specified
     `{key}_{split}_{sourceid}`. Measured in `extended_merged_exact_ged/computed/*.npz`: Letter ids
     are bare filename stems (`IP1_0000`, `AP1_0001`) and only the GraphEdX ids
     (`linux_train_0000`) match that pattern. Applying it literally would have broken the
     element-wise reproduction of the three Letter `graph_ids` arrays — which is the check that
     proves the graph order, and therefore every pair index, is right. **The contract was wrong.**
  2. **No class count is asserted.** The counts in §2 (Letter 15, GREC 22, Mutagenicity 2,
     Protein 6, AIDS 2, COIL-DEL 100) are **raw** dataset counts — re-verified correct as raw
     figures against the `.cxl` indices — not counts that survive `require_connected`. Realised
     counts and the sorted class list are measured outputs in the manifest.
  3. **The subsample is two files**, `UB_TIGHT/subsample_pairs.npz` (the sampler's pair list) and
     `UB_TIGHT/subsample.npz` (the campaign's result), so the run cannot overwrite its own input.

  > ⚠ **Two findings this amendment carries beyond T-05.**
  >
  > **Labels do not survive the filter intact.** `Letter LOW retains 9 of its 15 classes` and
  > `GREC 17 of its 22`; `LINUX` and `AIDS (GraphEdX)` carry **no class label at all**. Any
  > manuscript sentence of the form "Letter, 15 classes" or "GREC, 22 classes" is **false of the
  > filtered cohort**. This is the labels counterpart of the size-biased connectivity discard in
  > `decisions.md` §7 and it belongs to **T-18 and T-06**.
  >
  > **A frozen reproduction script cannot load GraphEdX from today's tree.**
  > `export_graphs.py:430` and `cohort_audit.py:254` both resolve `<source>/GED_PRECOMPUTED/<NAME>`;
  > the real path is `<source>/GED_PRECOMPUTED/datasets/<NAME>`, and because IAM now sits under
  > `APPROX_GED/datasets/IAM_Database/extracted`, **no single `--source` makes either module resolve
  > both roots**. So decision 22's tracked `cohort_audit.py` — the script whose whole purpose is that
  > "what it measures becomes the table" — **cannot re-derive the LINUX and AIDS-GraphEdX rows on the
  > current tree without a path fix.** Neither file is patched here; both are frozen T-01/T-03
  > artifacts. **This must be propagated at close.**
