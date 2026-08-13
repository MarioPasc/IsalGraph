# `slurm/approx_ged` — the T-05 approximate-GED bound campaigns on Picasso

Four bound campaigns over the **21,710,892** Suite-2 pairs, plus a cross-fill and the
independent gates. Modelled on `slurm/exact_ged/` (T-03), from which the launcher's
`_clean_job_id`, `submit`, `assert_dependency_took` and the `$LOCALSCRATCH` staging in
`_env.sh` are taken essentially unchanged.

**Contract**: `.claude/notes/2026-08-13-t05-bounds/CONTRACTS.md`.
**Design**: `.claude/notes/review/tasks/T-05-design.md` §4 (gates) and §5 (parallelisation).

---

## The one thing that makes this different from T-03

T-03 spent 2,081 core-hours and the design problem was **how to split the work**. This
programme is **~133 core-hours**, and on one 128-core node that is under an hour — three
orders of magnitude the wrong side of SCBI's two-hour floor. The design problem here is
the opposite: **how to keep from splitting it**.

So: **one single-node job per role.** Not a job array over datasets (Letter LOW is ~90
core-seconds; nine of ten tasks would be minutes long — the 12,600-task pattern Manuel at
`soporte@scbi.uma.es` wrote to *this account* about on 2026-08-07). Not an array over pair
chunks (correct for 2,081 core-h, absurd for 133). `ged_pair_index.py`'s chunking is
retained for **resumability inside one task**, not for fan-out.

Two consequences you will meet in the code:

| | T-03 (`slurm/exact_ged`) | T-05 (here) |
|---|---|---|
| Job shape | job arrays over pair chunks | one single-node job per role |
| Core formula | `ceil(core_seconds / TARGET)` | **`floor(core_seconds / TARGET)`** |

The rounding direction is **load-bearing**. `ceil` makes the projected wall **≤**
`TARGET_SECONDS` and therefore able to land under `FLOOR_SECONDS`; `floor` makes it **≥**
`TARGET_SECONDS`. T-03 could use `ceil` because its work never approached the floor. Here
the floor is the only binding constraint.

---

## Files

| File | What |
|---|---|
| `launcher.sh` | the only human entry point. Every `#SBATCH` flag lives here |
| `_env.sh` | sourced by every worker: interpreter, `PYTHONPATH`, `$LOCALSCRATCH`, traps, and the **frozen role table** |
| `worker_bounds.sh` | roles `lb` / `ub` / `ubs` over the ten datasets |
| `worker_subsample.sh` | role `ubt` over the CONTRACTS §5 subsample |
| `worker_crossfill.sh` | CONTRACTS §4.2 cross-fill, then gates G4 and `lb-consistency` |

Workers carry **no `#SBATCH` header**. One launcher dispatches four differently-sized jobs;
four headers would drift apart. This is the convention `slurm/exact_ged` established.

The four roles and their **verbatim** options strings live in one place only —
`_env.sh`'s `ROLE_METHOD` / `ROLE_OPTIONS` tables. They never cross `--export`. The
options string is part of the method name: GEDLIB's upper bounds change on 91.5–93.6 % of
pairs between runs at library defaults (T-27 §4.2), so a run whose metadata does not
record the string verbatim is invalid at the gate.

---

## Submission order and the dependency chain

```
                aged-lb   ┐
                aged-ub   ├─ afterok ─→ aged-crossfill ─→ (gates G4, lb-consistency)
                aged-ubs  ┘
                aged-ubt  ─ independent, no dependant
```

`aged-ubt` is deliberately **outside** the chain. T-05-design §5: *"a failure in the
93-core-h `IPFP_MS` arm would take the 3.4-core-h primary lower bound down with it. The
primary deliverable must not depend on the arms."*

The cross-fill waits on **`afterok`**, not `afterany`. `certified_mask` is a *proof*
(CONTRACTS §4.1) — the derived statement that a proven lower bound of `k` and an exhibited
edit path of cost `k` together prove GED = `k`. Cross-filling from a partial campaign
would not leave a hole in it; it would write a **false** entry, indistinguishable
downstream from a correct one. `worker_crossfill.sh` re-checks all 30 role files and exits
4 if any is missing, so the guard survives a manual `--stage merge`.

### Recommended sequence

```bash
# 0. Sizing report only. Submits NOTHING, and normally exits 3 (see below).
bash slurm/approx_ged/launcher.sh --stage probe

# 1. Look at every sbatch line before any of them runs.
bash slurm/approx_ged/launcher.sh --dry-run --stage all

# 2. The cheapest role first, alone. It is the primary deliverable.
bash slurm/approx_ged/launcher.sh --stage lb

#    --- HUMAN CHECK 1 (below) ---

# 3. Feed the measured rate back and submit the rest.
bash slurm/approx_ged/launcher.sh --stage all \
    --probe-json <OUT_DIR>/run_report_<jobid>.json \
    --bins       <the sampler's per-bin pair table>
```

### `--stage probe` exits 3, and that is the correct behaviour

T-05-design §5 says a separate probe job would itself violate the floor. The launcher does
not assert that — it computes it, through the same `assert_clears_floor` that guards every
production job, and refuses. At the projected rates the standalone probe is ~2,411
core-seconds against a 7,200-second floor.

The probe that actually runs is the one **inside** each production job, on the hardware
that does the work, before that job's production pass. It writes its measured
seconds-per-pair into the job's `run_report_<jobid>.json`, which is what you pass back as
`--probe-json`.

---

## What a human must check between stages

**HUMAN CHECK 1 — after the first role lands, before submitting the rest.**

1. `run_report_<jobid>.json`, keys `projected_wall_seconds` vs `realised_wall_seconds`.
   T-27 limitation 3 says the projections are **lower** bounds on true cost. If realised
   is more than ~3× projected, re-size before submitting `ubs` and `ubt` — the `--time`
   headroom is 3–4×, not 10×.
2. `cleared_floor` must be `true`. If it is `false`, the next submission of that role
   needs fewer cores or `--group`.
3. `probe_stratified`. If `false`, the worker fell back to a contiguous first chunk of one
   dataset because the sampler's stratified probe list was absent. **That rate is biased
   low** — contiguous upper-triangle indices over a size-ordered export oversample the
   small-`n` corner, and per-pair cost scales ~`max(n₁,n₂)³`. Do not feed a
   non-stratified probe to `--probe-json` as if it were the design's probe.
4. `sizing_evidence` — `binned` beats `flat-measured` beats `projected`. If it says
   `projected` you sized from the contract's `n̄`-based numbers, which for a cohort running
   from `n̄ = 4.07` to `n̄ = 31.68` with a tail to `n = 98` is wrong by a large factor, and
   Jensen's inequality makes it wrong in the **under**-estimating direction.

**HUMAN CHECK 2 — after cross-fill.**

5. `worker_crossfill.sh` runs only **G4-verify** and **`lb-consistency`**. G2 and G3 need
   T-27's recorded cells and T-03's exact census, which live on the workstation, not on
   the cluster. **Run them there before quoting a single number from this campaign.** The
   split is the same one `slurm/exact_ged/worker_gates.sh` makes.
6. Read the gate JSON, do not just read the exit code. Every record names the tolerance it
   used and why.

**HUMAN CHECK 3 — resource accounting, cluster-wide.**

7. Total cores across the wave at the projected rates: **1 + 2 + 9 + 31 = 43**, well under
   the two `sr` nodes' worth this wave is allowed and against 42 idle `sr` nodes at design
   time. If a re-size pushes the total past ~256, stop and reconsider grouping.

---

## Sizing, in detail

```
core_seconds  = Σ_bins  pairs_in_bin × measured_rate_in_bin       ← best
              | total_pairs × measured_flat_rate                  ← acceptable
              | total_pairs × CONTRACTS §3 projection             ← logs a WARNING
cores         = clamp( floor(core_seconds / 10800), 1, 128 )
wall          = core_seconds / cores           and wall ≥ 7200, or exit 3
```

The per-bin path needs two files whose `bin_edges` **must agree**, or the launcher ignores
the binned path rather than multiplying counts by rates from a different binning:

```jsonc
// --bins   (from the sampler)
{"bin_edges": [2,4,6,8,10,12,15,20,25,30,40,50,60,80,99],
 "totals":    [ ... 14 pair counts ... ],
 "datasets":  {"linux": [...14...], ...}}      // "totals" optional; summed from these

// --probe-json  (from a completed job's run_report, or hand-assembled)
{"bin_edges": [ ...same 14 edges... ],
 "per_bin_seconds_per_pair": {"lb": {"0": 1.2e-5, ...}, "ub": {...}},
 "seconds_per_pair": {"lb": 5.6e-4, ...}}      // the flat fallback
```

### Resolved values at the CONTRACTS §3 projected rates

| Job | Role | Pairs | core-s | Cores | Projected wall | `--time` | Worker |
|---|---|---:|---:|---:|---:|---|---|
| `aged-lb` | `lb` | 21,710,892 | 12,240 | **1** | 3.40 h | `0-12:00:00` | `worker_bounds.sh` |
| `aged-ub` | `ub` | 21,710,892 | 30,240 | **2** | 4.20 h | `0-12:00:00` | `worker_bounds.sh` |
| `aged-ubs` | `ubs` | 21,710,892 | 100,800 | **9** | 3.11 h | `0-12:00:00` | `worker_bounds.sh` |
| `aged-ubt` | `ubt` | 28,000 | 334,800 | **31** | 3.00 h | `1-00:00:00` | `worker_subsample.sh` |
| `aged-crossfill` | — | — | — | 4 | — | `0-04:00:00` | `worker_crossfill.sh` |

All on `--constraint=sr`, `--mem=64G`, `--account=tic_163_uma`. `sr` is pinned because
per-pair wall time is a **reported** quantity (D12, and T-27 §5 extended to Suite 2); a
mixed Intel/AMD pool would make the timing a measurement of the scheduler.

Every wall above clears the 7,200 s floor **by construction**, not by luck: `floor`
division guarantees `wall ≥ TARGET_SECONDS = 10800` whenever `core_seconds ≥ 10800`, and
`assert_clears_floor` refuses the submission when it does not.

### When a role cannot fill two hours

Do not submit it short. Merge it into an adjacent job:

```bash
bash slurm/approx_ged/launcher.sh --stage all --group lb,ub
# -> aged-lb-ub: 3 cores, projected wall 3.93 h    (core-seconds add)
```

Grouped roles run **sequentially inside one job** and share one job id, so the cross-fill's
dependency list de-duplicates them. The roles list crosses `--export` **colon**-separated
(`ROLES=lb:ub`) because `--export` splits on every comma.

---

## Traps this code is written around

- **`GROUPS` is a bash builtin array** holding the user's group ids. Assigning to it fails
  (`rc=1`), which under `set -e` kills the launcher during flag parsing — or, depending on
  context, silently leaves it empty so `--group` is ignored and four jobs are submitted
  where two were asked for. The variable here is `ROLE_GROUPS`. Measured on bash 5.2.15.
  Same trap: `SECONDS`, `PIPESTATUS`, `RANDOM`, `LINENO`, `FUNCNAME`.
- **Picasso's Lua `sbatch` wrapper emits ANSI codes and a multi-line warning**, so
  `--parsable` does not return just the id. `_clean_job_id` takes the **last line first**,
  then strips — a line-wise `sed` leaves a multi-line "id" and the guard then fires *after*
  submission, leaving an untracked job on the cluster.
- **A bad job id in `--dependency` is ACCEPTED** and recorded as `Dependency=(null)`; the
  downstream job then starts immediately against partial input. `assert_dependency_took`
  checks and cancels.
- **`--export` splits on every comma.** No value it carries contains one. The GEDLIB
  options strings — which contain spaces and would be the obvious thing to ship — never
  cross it at all; the workers read them from `_env.sh`'s frozen table.
- **Picasso exports a shared `PYTHONPYCACHEPREFIX`**, so tasks of one user on a node write
  the same `.pyc` paths. Symptom: an intermittent `ModuleNotFoundError` on a module that is
  plainly present. `_env.sh` gives each task its own.
- **`PYTHONPATH` is the repo ROOT, never `${REPO_DIR}/src`.** `isalgraph` is not imported
  by this ticket at all (CONTRACTS §9); T-05 does not touch the encoder and must not
  acquire a dependency on the C++ engine.
- **`conda` is not on `PATH` on compute nodes.** The interpreter is invoked by absolute
  prefix path. No `module load` anywhere.
- **Shards are deleted by the merge, after its own structural gate passes** (CONTRACTS
  §6.2, §7). `worker_bounds.sh` passes `--delete-shards` and adds no `rm` of its own; a
  gate failure raises `MergeError`, `set -e` aborts, and the shards survive for
  diagnosis. `worker_subsample.sh` passes no such flag because its merger has none — its
  shards live on `$LOCALSCRATCH`, outside the mirrored `out/` tree, so SLURM wipes them
  when the job ends and they never touch the fscratch file quota.
- **The `ubt` role uses a different merger**, `approx_ged_subsample_merge.py`, not
  `ged_merge_shards.py`. CONTRACTS §7's merge writes a dense `(N, N)` matrix and cannot
  express `UB_TIGHT/subsample.npz`: the subsample is pooled across all ten datasets
  (CONTRACTS §5), so `--n-graphs` is meaningless and no key names one cohort. Widening the
  dense merger for a 28,000-row special case would put T-03's closed, load-bearing dense
  path at risk. Its CLI is `--shards --pair-list --out --role --method --options`.
- **`UB_TIGHT/subsample_pairs.npz` and `UB_TIGHT/subsample.npz` are two different files.**
  The first is the sampler's pair list, written ahead of the run and reproducible from seed
  42; the second is this campaign's result. `worker_subsample.sh` asserts they do not
  resolve to the same path, because pointing `--out` at the pair list would destroy the
  only auditable record of which pairs were drawn.

---

## What is not here

- **The calibration ladder** (exact GED above `n = 12`, T-05-design §6, ~300–500 core-h) —
  a later wave. Nothing in this directory computes it.
- **Any analysis, figure, correlation or bootstrap.** This directory produces the bound
  files and checks them; it draws nothing.
- **G2 and G3 on the cluster.** They need references that live on the workstation.
