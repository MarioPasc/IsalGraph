# T-03 — exact GED on Picasso

Design and rationale: `.claude/notes/review/tasks/T-03-design.md`. **Read its Changelog first** —
amendments 2 and 3 changed the solver and the gate, and this directory implements the amended plan.

## Submit order

```bash
bash slurm/exact_ged/launcher.sh --dry-run --stage all --sec-per-pair 15   # always first
bash slurm/exact_ged/launcher.sh --stage gates                            # blocks everything
# read the gate report, take the real rate from gate 3, then:
bash slurm/exact_ged/launcher.sh --stage small --sec-per-pair <measured>
bash slurm/exact_ged/launcher.sh --stage aids  --sec-per-pair <measured>
bash slurm/exact_ged/launcher.sh --stage merge
```

`--stage all` chains everything with `afterok`, which is right once the rate is known. Until then,
run `gates` alone and read it — **`--sec-per-pair` is required and the launcher refuses to guess.**

| Job | Content | Cores | Wall |
|---|---|---:|---|
| `ged-gates` | gates 0–3 + the solver benchmark | 4 | ~2–4 h |
| `ged-small` | Letter LOW/MED/HIGH + LINUX, all pairs, merged per dataset | derived | ~2–3 h |
| `ged-aids-s1` | stage 1, 22,106 pairs over **all 769 graphs** | derived | ~2.5 h |
| `ged-aids-s2` | stage 2 census, array over upper-triangle chunks | 64 × T | ~2.5 h/task |
| `ged-merge` | merge, gate 4, shard deletion | 2 | minutes |

## What the design forces, and why

- **No task runs under two hours.** SCBI asked for this directly (2026-08-07); this account has
  already been written to about short jobs. The launcher derives the task count from
  `--sec-per-pair` and **exits 3** rather than submitting a short task. LINUX alone would be
  18 minutes, which is why the small suite bundles four datasets into one job.
- **`--constraint=sr`** (128 c / 450 GB). Per-pair wall time is a *reported* quantity for the D12
  censoring analysis, so a mixed Intel/AMD pool would make the timing a measurement of the scheduler.
- **CPU only.** No `--gres`, no `--constraint=dgx`.
- **`PYTHONPATH="${REPO_DIR}:${GEDLIB_DIR}"` — repo root, never `${REPO_DIR}/src`.** `isalgraph` is
  not installed on the cluster and T-03 must not import it; the gate worker asserts it is absent.
- **Exact = `networkx` A\*; GEDLIB is bounds-only.** `ANCHOR_AWARE_GED` is retired (amendment 2) and
  guarded by name. Certification is decided by whether A* completed, never by a solver's self-report.

## Monitoring — two Picasso quirks

```bash
ssh picasso 'squeue'                     # -u is REJECTED by Picasso's wrapper
ssh picasso 'sacct -j <id> -X -n -P -o JobID,State,Elapsed,NodeList'
ssh picasso 'sacct -j <id> -n -P -o JobID,MaxRSS | grep .batch'   # NO -X: it blanks MaxRSS
```

`sacct -X` records memory on the `.batch` step, not the allocation, so `-X` returns an empty
`MaxRSS` — a table of blank cells and no error, which is the worst outcome when the table exists to
size `--mem`.

## Re-running only the failed array tasks

```bash
FAILED=$(ssh picasso "sacct -j <ARRAY_ID> -n -X -o JobID,State \
  | awk '\$2!=\"COMPLETED\"{split(\$1,a,\"_\"); print a[2]}' | paste -sd, -")
# then resubmit with --array="${FAILED}%8"
```

Resume is free: each task checkpoints every 2,000 pairs to a single file overwritten in place, and
the runner skips completed pairs on restart. A requeue loses minutes, not hours.

## Stage 1 is not a matrix, and that is deliberate

Stage 1 computes 22,106 of 295,296 pairs, so a full 769×769 matrix would be 92.5 % empty — a shape
that invites every consumer to read missing as censored. It stays a CONTRACT C shard carrying pair
indices, values, certification flags and per-pair wall times. That is also the format `--seed-from`
consumes, so stage 2 reuses stage 1's work without conversion, and the merge asserts the two agree on
every shared pair.
