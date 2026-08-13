# The Picasso loop — orchestrator only

Subagents never run anything here. They may load `picasso-sbatch` and *write* scripts; every
command below is yours. Measurements are from the T-03 campaign (2026-08-12/13) and are
**live state** — re-read them, never quote them.

```
repo    /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalGraph
env     /mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/isalgraph
data    /mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/isalgraph
out     /mnt/home/users/tic_163_uma/mpascual/execs/isalgraph/<ticket>
account tic_163_uma          QOS: short(2h) / medium_uma(3d) / long_uma(7d), cpu=9000/user
```

## 0. Read the live state before sizing anything

```bash
ssh picasso 'quota; sacct -j <prev> -X -n -P -o State,Elapsed'
ssh picasso 'sinfo -p cpu_partition -o "%5c %9m %20f" -h | sort | uniq -c | sort -rn'
ssh picasso 'scontrol show config | grep -E "MaxArraySize|MaxJobCount"'
```

**`squeue -u` is rejected** by Picasso's wrapper (`[ERROR]: Invalid option -u`) — the wrapper
already scopes to you. Use bare `squeue`, or `sacct`.

CPU families measured 2026-08-12: `sd` 52c/187G intel · `sr` 128c/450G amd (largest idle
pool, ~45 nodes) · `bc` 256c/700G amd · `bl` 128c/1900G amd.

**Pin the family with `--constraint` whenever wall time is a reported quantity.** A mixed
Intel/AMD pool turns any timing into a measurement of the scheduler.

## 1. Size from a measured rate

Never from the plan's estimate. Measured across three datasets in T-03, **`sr` cores are
~2× slower** than the workstation the plan's per-pair figures came from — consistently
enough to be a usable correction, but only because it was measured.

```
n_tasks = floor(total_units × sec_per_unit / (cores_per_task × target_seconds))
```

with `target_seconds` ≈ 9000 (2.5 h). **Refuse to submit if the projection puts any task
under 2 h** — reduce the task count instead.

> **SCBI's two-hour floor is binding and this account has been written to about it**
> (Manuel, soporte@scbi.uma.es, 2026-08-07) after a 12,600-task campaign of minute-long
> jobs. Grouping is makespan-neutral: `N` units of duration `T` under throttle `K` finish at
> `N·T/K` either way. Contact them before a first array over 1000 tasks.

## 2. Local smoke, then transfer, then three stages

```bash
# 1. real data, complete small dataset, end to end -- not "it started"
$PY -m benchmarks... --backend <real> ...

# 2. transfer; .git is excluded, so the far side's git metadata is STALE
rsync -a --delete --exclude='.git' --exclude='__pycache__' --exclude='.claude/worktrees' \
      --exclude='.hypothesis' --exclude='.ruff_cache' --exclude='*.pyc' \
      ./ picasso:$REPO/

# 3. validate, one task, campaign
ssh picasso 'sbatch --test-only ... worker.sh'     # unsatisfiable requests, in one second
ssh picasso 'bash launcher.sh --stage <one>'       # ONE real task
ssh picasso 'bash launcher.sh --stage all ...'     # the campaign
```

⚠ Because `.git` is excluded, `git rev-parse` on the cluster reports a **stale** commit. If
a worker prints a provenance line, export the local SHA through `--export` instead.

## 3. Eight traps that fail silently

| # | Trap |
|---|---|
| 1 | **`--parsable` does not return just the id.** The Lua wrapper prepends ANSI codes and a multi-line warning. A line-wise `sed` leaves a multi-line "id" and the guard fires *after* submission, leaving an untracked job. Take `tail -n 1` **first**, then strip, then assert `^[0-9]+$`. |
| 2 | **`--export` splits on every comma**, so a comma inside a *value* truncates it and the tail becomes a junk variable. Nothing errors. Ship lists colon-separated. |
| 3 | **`#SBATCH` after the first uncommented line is ignored**, silently, with defaults. |
| 4 | **A dependency built from a bad id is accepted**, recorded `Dependency=(null)`, and the downstream job starts immediately against partial input. Assert with `scontrol show job`. |
| 5 | **`sacct -X` reports an empty `MaxRSS`** — memory lives on the `.batch` step. Drop `-X` for memory, keep it for states. |
| 6 | **SLURM copies the batch script to `/var/spool/slurmd/job<id>/`**, so `$(dirname "${BASH_SOURCE[0]}")` finds nothing. Source shared fragments from `$REPO_DIR`. If the fragment sets `set -euo pipefail`, failing to source it disables that too — set it in each worker as well. |
| 7 | **SIGTERM bypasses the EXIT trap.** SLURM waits 30 s before SIGKILL; without `trap 'exit 143' TERM` the copy-back never runs and output dies on the node. |
| 8 | **`PYTHONPYCACHEPREFIX` is shared per user per node**, so concurrent tasks race on `.pyc` paths — an intermittent `ModuleNotFoundError` on a module that is plainly present. Give each task its own. |

## 4. `PYTHONPATH` — repo root, never `src`

```bash
export PYTHONPATH="${REPO_DIR}:${EXTRA_CHECKOUT}"     # NOT ${REPO_DIR}/src
```

`isalgraph` is **not installed** in the cluster conda env. A ticket that does not need the
encoder must not import it, and a src-first path is the documented way this project silently
loads pure Python where it thinks it has the C++ engine. If a ticket *does* need the engine,
build it on the cluster — the `.so` lives in site-packages and **does not rsync** — with
`-march=x86-64-v3`, never `-march=native` (`sr` and `bl` lack AVX-512 and produce SIGILL).

## 5. Monitor for terminal states, not for progress

```bash
Monitor(command: <poll sacct, emit on COMPLETED|FAILED|TIMEOUT|CANCELLED|NODE_FAIL, break when none active>,
        persistent: true)
```

**A filter that greps only for progress is silent through a crashloop, and silence is
indistinguishable from health.** Emit on every terminal state. Escalate immediately on
`ModuleNotFoundError`, `FileNotFoundError`, `oom-kill`, or a task exiting under a minute —
a 300-task array failing identically is 300 wasted allocations and a day of queue time.

## 6. Output hygiene

`fscratch` limits **file count** (~250k soft / 400k hard), not just space. Per-task shards
are fine; **merge them and delete the shards** as the final step. Serialize inputs to one
file per dataset before transferring — the IAM Letter GXL tree is 6,767 files and became
6 files totalling 55 KB.

Write per-task output to `$LOCALSCRATCH` and mirror the **whole tree** back on exit — never
a list of expected files, because everything you named comes back and nothing looks wrong
until the analysis needs the one you forgot. Copy to a per-task temp name in the destination
and `mv -f`; `cp` truncates before writing, so concurrent tasks can leave a short file.

Verify counts and shapes before declaring done, then mirror to `execs/` and off-cluster.
