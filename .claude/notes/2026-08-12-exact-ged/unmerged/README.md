# Unmerged agent work, preserved 2026-08-13

Work that a wave-2 agent produced but never committed, kept because it contains design the
shipped version does not. **None of it ran.** Do not execute anything here.

## `launcher-task-slurm-draft.sh` — 629 lines

`task-slurm` was writing this when it died on an account session limit, having reported only
"writing the launcher first". It was left uncommitted in its worktree; the shipped
`slurm/exact_ged/launcher.sh` (237 lines) was written from scratch by the orchestrator
instead, and is the one that ran the T-03 campaign.

**Status**: never executed, never syntax-checked against the real CLIs, and written against
an earlier contract — it points `DATA_DIR` at `fscratch/datasets/isalgraph/exact_ged` and
adds a `WORK_DIR`, neither of which the shipped layout uses. Treat it as a design sketch.

### Four ideas in it that the shipped launcher lacks

Worth lifting if the SLURM layer is revisited — each is a real improvement, not a stylistic
difference:

1. **Per-stage QOS.** `medium_uma` (3 d) for the compute stages, `short` (2 h) for the merge
   barrier. The shipped version leaves QOS to the scheduler's default on every job. Asking
   for `short` on a two-minute merge is the courteous request and may schedule faster.
2. **`--stage merge --after <jobid>`.** A merge that can be attached to an already-running
   array after the fact. The shipped version only chains a merge submitted in the same
   invocation, which is why the T-03 merge had to be run by hand on the login node once the
   census finished.
3. **`aids1` and `aids2` as separate stages.** The shipped launcher submits both together
   under `--stage aids`, so re-running only stage 2 means editing the script.
4. **Colon-separated key and count lists** (`SMALL_KEYS`, `SMALL_NGRAPHS`), which is the
   correct handling of the `--export` comma-splitting trap for list-valued variables. The
   shipped version sidesteps the issue by hard-coding the dataset loop inside
   `worker_small.sh` instead — workable, but less general.

### What the shipped version has that this lacks

It ran. It also carries the `_env.sh` fix for SLURM's spool-directory copy, the `SIGTERM`
trap, the `$LOCALSCRATCH` whole-tree copy-back, and the sub-2 h refusal — all of which were
found by running the thing, not by writing it.

## The other stranded worktree

`agent-a448612b12841ddf9` (`task-gates-v2`) also died with uncommitted changes to
`ged_backends.py` and `ged_gates.py`. Those were **salvaged in full** into the integration
branch and then extended, so nothing from that worktree is preserved here — verified by
diffing: the only worktree-only lines were the pre-edit versions of lines that were
subsequently changed.
