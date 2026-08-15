# T-05 session handoff — paste this into a fresh Claude Code session

You are the orchestrator finishing IsalGraph ticket **T-05** (bounded GED over Suite 2).
Repo `/home/mpascual/research/code/IsalGraph`, branch `main`, HEAD `9db9b2a`, tree clean,
suite green (`1273 passed / 1 skipped`).

Load the `review-ticket` skill. **You own every Picasso interaction; subagents never touch it.**

## Read first
- `.claude/notes/review/tasks/T-05-design.md` — the frozen design **and amendments 1–11**.
  Amendments 5, 7 and 11 record things I got wrong and withdrew; do not re-derive them.
- `.claude/notes/2026-08-13-t05-bounds/CONTRACTS.md` — §3 roles/options, §4 output schema, §5 subsample.
- `.claude/notes/2026-08-13-t05-bounds/summary.md` — wave summary.

## State: the compute is ~80 % done and running

**Already complete and mirrored on Picasso** (`$E` = `/mnt/home/users/tic_163_uma/mpascual/execs/isalgraph/approx_ged`):
- `$E/{LB,UB,UB_SENSITIVITY}/` — **8 of 10 datasets each** (all but `coil_del`, `mutagenicity`)
- `$E/UB_TIGHT/subsample.npz` — 28,000-pair `IPFP_MS` arm, **done** (job 1993160, 300.8 core-h)
- `$E/ladder/rung_{13..18}.npz` — **done** (job 1993507). Ceiling **n = 17** (rung 18 = 20.8 % < 25 % rule)

**Running now** — 60 array tasks, `--workers 1`, ~3 h each, each with an `afterok` merge:

| role | dataset | array | tasks | merge |
|---|---|---|---:|---|
| lb | coil_del | 2005378 | 3 | 2005379 |
| lb | mutagenicity | 2005380 | 2 | 2005381 |
| ub | coil_del | 2005385 | 7 | 2005386 |
| ub | mutagenicity | 2005387 | 5 | 2005388 |
| ubs | coil_del | 2005389 | 24 | 2005390 |
| ubs | mutagenicity | 2005391 | 19 | 2005392 |

Check: `ssh picasso "sacct -j 2005379,2005381,2005386,2005388,2005390,2005392 -X -n -P -o JobID,State"`
Shards: `$E/shards/{role}_{key}/` (deleted by the merge on success).

## Your remaining steps
1. **Wait for all six merges** → `$E/{LB,UB,UB_SENSITIVITY}` each hold 10 files.
   Monitor on *change/failure only*; a heartbeat every cycle drowns the signal.
2. **Run `bash slurm/approx_ged/finalize_local.sh`** (local). It pulls, refuses to proceed on
   <30 matrices, cross-fills the bracket into all three role files, and runs the gates.
3. **Verify the gates yourself, do not trust its exit code.** G2 is the strong one: on
   `iam_letter_{low,med,high}` + `linux` (Suite-2 cohort identical to Suite 1) the values must equal
   T-27's census **element-wise, 3,602,615 pairs**. My pre-run checksums of the `value` float64 array:
   `BRANCH_FAST` sum 15740 sha256[:16] `e95b44c7edad1369` · `BIPARTITE` 42936 / `2528fd19b98accb0`
   · `BP_BEAM_DET` 23984 / `ba116a0290986360` (LINUX, 3,916 pairs).
4. **Publish** to `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/APPROX_GED/`
   → `LB/ UB/ UB_SENSITIVITY/ UB_TIGHT/ ladder/ gates/ exported_suite2/` + `manifest.json` + `PROVENANCE.md`.
5. **`Skill(review-close)`** — it owns the board strike and propagation. Do not hand-write those.

## Traps that already cost time
- **The process pool is negative-scaling.** 1 worker 36 core-s, 4 → 212, 15 → 928, 32 → 5,260 on
  identical work. `--workers 1` is the measured optimum. Never raise it.
- Build runner method flags as a **bash array**; `$(echo --lb-options "$OPTS")` word-splits and dies.
- Picasso's checkout is rsynced, so `git rev-parse` lies — pass `ISALGRAPH_CODE_COMMIT=$(git rev-parse HEAD)`.
- T-03's censored pairs carry **`inf`, not NaN**; filter `np.isfinite`, select on `certified_mask`.
- Stale failed logs live in `$E/logs/failed_2005370/` — do not read them as live errors.

## Open, needs the PI (do not decide alone)
**T-03's `ub_matrix` is irreproducible.** Its default `--ub-options "--threads 1"` leaves `IPFP` on
`--randomness REAL`: 74–82 % of values change between runs. Exposure is bounded and verified —
`ub_matrix == ged_matrix` on all 234,258 certified AIDS + 3,870 certified LINUX pairs — so it is
**exactly the 61,084 D11 censored interval upper ends**. Repair is hours under the frozen `PSEUDO`
string, but it is a closed ticket's file. Needs an owner.

## Findings `review-close` must propagate
Letter LOW retains **9 of 15** classes / GREC **17 of 22** after the connectivity filter (LINUX and
AIDS-GraphEdX carry none) · size and provenance are **confounded** across the size bins, so the AE.1
curve is **within-dataset primary**, pooled descriptive only · `decisions.md` §6's 33.2 % orientation
figure describes graphs ~3× larger than it was measured on, and the two upper bounds move in
**opposite** directions in `n` · CLAUDE.md's "assert `0 < value < inf` on every read" is wrong per
pair (GED is legitimately 0 for isomorphic graphs, 15.5 % of Letter LOW).
