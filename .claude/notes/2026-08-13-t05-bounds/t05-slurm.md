# Work log — t05-slurm

## Identity

| Field | Value |
|---|---|
| Agent | `wave-t05-slurm` |
| Wave | `2026-08-13-t05-bounds` |
| Model / effort | `claude-opus-5` / `xhigh` |
| Branch | `worktree-agent-a60da166efbbe5eac` |
| Worktree | `/home/mpascual/research/code/IsalGraph/.claude/worktrees/agent-a60da166efbbe5eac` |
| Base commit | `885d98d8e6b37dfeb98c4df741510fc28d4a8615` |
| Head commit | `<final sha>` |
| Started / finished | `2026-08-13T00:00:00Z` / `<ISO timestamp>` |
| Status | in progress |

Worktree confirmed distinct from `/home/mpascual/research/code/IsalGraph` before any edit
(`git rev-parse --show-toplevel` → the worktree path above).

## 1. Prompt as received

```
You are agent `wave-t05-slurm`, an implementation agent working inside an **isolated git worktree**
on a branch of your own, in parallel with two peers who own different files. You never see the
orchestrator's conversation; everything you need is in this prompt and in the repository.

This work is for a *Pattern Recognition* major revision (PR-D-26-03293) due 2026-08-31, read by
reviewers who checked every number last round. **Correctness beats speed. An honest negative result
beats a convenient one.**

You write the gates that decide whether other agents' output is trustworthy. **You are the
independent verifier**: code the gate against the written contract, not against what the runner's
implementation happens to do. If the two disagree, surfacing that is the whole value of the split.

## Standing obligations
1. Work only inside your worktree. Every file you create or edit must lie inside your declared
   ownership set. Everything else is read-only reference. Confirm at the start that
   `git rev-parse --show-toplevel` differs from `/home/mpascual/research/code/IsalGraph`; if it does
   not, stop and message `main`.
2. Commit your work in logical commits **as you go**, not at the end. Sessions die; uncommitted work
   cannot be merged, because the orchestrator merges your branch, not your working tree.
3. Maintain your work log at `.claude/notes/2026-08-13-t05-bounds/t05-slurm.md` from your first
   action to your last, using the template committed at
   `.claude/notes/2026-08-13-t05-bounds/NOTE-TEMPLATE.md`, and commit it as your final commit.
4. Never run `git push`, never rebase or merge, never touch a peer's branch or worktree.
5. **You have no access to Picasso, and this is absolute even though you are writing SLURM scripts.**
   No `ssh`, `rsync`, `sbatch` (not even `--test-only`), `squeue`, `scancel`, `scp`. You write and
   syntax-check scripts; the orchestrator submits them. A script you cannot test on the cluster must
   be written so that its failure modes are obvious on reading.
6. You cannot ask the user anything. On an ambiguity, message `main` with a specific question, record
   the assumption you are proceeding on in your log, and keep working. Do not block.
7. Never change a frozen contract yourself. Propose it to `main`. **Finding that your brief is wrong
   is a success** — report it with evidence.
8. Report failure honestly. "This does not work and here is why" beats a plausible-looking
   implementation that was never exercised.
9. Plan before editing and write the plan into your log. Implement in small verified steps. Write
   tests as you go. Run the suite before your final commit and record the real output, failures
   included.

---

# Task: the Picasso launcher/worker pair for T-05, and the independent validation gates

## Mission
Write `slurm/approx_ged/` — a launcher plus workers that run the four T-05 bound campaigns on
Picasso, modelled on the working `slurm/exact_ged/` pair, sizing every job from a **measured** rate so
that each clears SCBI's two-hour floor by construction. Write
`benchmarks/real_data/eval_setup/approx_ged_gates.py` — the independent gates G2, G3 and G4-verify
that decide whether the campaign output is trustworthy. Working means: `bash -n` clean, a `--dry-run`
that prints every `sbatch` line it *would* issue, and gates that pass on real recorded data and fail
on a deliberately perturbed copy of it.

## Why this exists
T-05 computes a proven GED bracket over **21,710,892** Suite-2 pairs and hands the result to the
paper's entire large-`n` argument (demand AE.1). Two things can go wrong silently. **The cluster
side**: GEDLIB reads through the wrong accessor return `0.00` and raise nothing, and an upper-bound
matrix filled in one orientation is not a distance matrix. **The scheduling side**: this workload is
~130 core-hours against T-03's 2,081, so the binding constraint is not compute but SCBI's two-hour
minimum — Manuel at soporte@scbi.uma.es wrote to *this account* after a 12,600-task campaign of
minute-long jobs. A design that produces short tasks is out, however convenient.

## Repository orientation
- Repository root: your worktree (`git rev-parse --show-toplevel`).
- **Read first, in this order**:
  1. `.claude/notes/2026-08-13-t05-bounds/CONTRACTS.md` — §3 roles, §4 output schema, §5 subsample,
     §6 runner CLI, §7 merge CLI, §8 Picasso environment. **This is your specification.**
  2. `.claude/notes/review/tasks/T-05-design.md` §4 (the four gates and what each is for) and §5
     (the parallelisation argument and the rejected alternatives).
  3. `slurm/exact_ged/` **in full** — `README.md`, `_env.sh`, `launcher.sh`, `worker_gates.sh`,
     `worker_small.sh`, `worker_aids.sh`, `worker_merge.sh`. This is T-03's working pair and your
     exemplar. Note especially `launcher.sh`'s `cores_for_single_task()` (~:95), `tasks_for_array()`
     (~:103), `FLOOR_SECONDS=7200`, the hard `exit 3` when a projection falls under the floor (~:183),
     and `_clean_job_id()` (~:66), which works around Picasso's ANSI-emitting Lua `sbatch` wrapper.
  4. `.claude/notes/review/tasks/T-03-design.md` §3 and §6 — why the array shape was chosen there,
     and why it is **wrong here** (T-05-design §5 explains the difference).
- **Invoke the `picasso-sbatch` skill before writing any SLURM script.** It is the authority on
  partitions, constraints, node families and wallclocks; values written anywhere else go stale.
- Conventions: `CLAUDE.md` is loaded. Additionally: NumPy-style docstrings, full type annotations,
  `logging` never `print` in library code, Python 3.11.

## Your ownership (exclusive write access)
Create or modify ONLY:
- `slurm/approx_ged/` — everything under it (`README.md`, `_env.sh`, `launcher.sh`,
  `worker_bounds.sh`, `worker_subsample.sh`, and any further worker you justify)
- `benchmarks/real_data/eval_setup/approx_ged_gates.py`
- `tests/unit/test_approx_ged_gates.py`
- `.claude/notes/2026-08-13-t05-bounds/t05-slurm.md` (your log)

Everything else is read-only, including **`slurm/exact_ged/`, which you copy from and must not
modify**, and every file your peers own.

## Base state
- Base commit: `885d98d8e6b37dfeb98c4df741510fc28d4a8615`.
- Your peers branch from the same commit. Do not rebase, merge or cherry-pick.

## Frozen contracts
From `CONTRACTS.md`; code against them exactly.

- **The four roles and their jobs** (CONTRACTS §3, design §5). Projected core-hours, to be replaced
  at launch by measurement:

  | Job | Role | Method + options | Scope | Projected core-h |
  |---|---|---|---|---:|
  | `aged-lb`  | `lb`  | `BRANCH_FAST`, `--threads 1` | all 21,710,892 pairs | 3.4 |
  | `aged-ub`  | `ub`  | `BIPARTITE`, `--threads 1` | all 21,710,892 pairs | 8.4 |
  | `aged-ubs` | `ubs` | `BP_BEAM`, `--threads 1 --randomness PSEUDO --initialization-method BIPARTITE --initial-solutions 1` | all 21,710,892 pairs | 28 |
  | `aged-ubt` | `ubt` | `IPFP`, `--threads 1 --randomness PSEUDO --initial-solutions 10` | the §5 subsample (≤ 28,000 pairs) | 93 |

- **Runner CLI** (CONTRACTS §6) — what your workers invoke:
  `--input --out --backend gedlib --cost-model unit --lb-method --lb-options --ub-method --ub-options
  --compute {lb,ub,both} --role --chunk-index --n-chunks --pair-list --workers --checkpoint-every
  --checkpoint`.
  **Merge CLI** (CONTRACTS §7): `--shards --key --n-graphs --out --ged-from {exact,lb,ub} --role
  --seconds-role --delete-shards`.
  Peer `wave-t05-runner` is implementing these to the same contract; **code against the contract, not
  against its branch, which you cannot see.**
- **Picasso environment, verbatim** — CONTRACTS §8. `ACCOUNT=tic_163_uma`, `CONSTRAINT=sr`
  (128 c / 450 GB), conda by **absolute prefix path** (conda is not in `PATH` on the compute nodes),
  `PYTHONPATH=$REPO_DIR:$GEDLIB_DIR`, no `module load`, `OMP/MKL/OPENBLAS_NUM_THREADS=1`, a per-task
  `PYTHONPYCACHEPREFIX`. Input `…/fscratch/datasets/isalgraph/suite2`, output
  `…/execs/isalgraph/approx_ged`.
- **`FLOOR_SECONDS = 7200`, `TARGET_SECONDS = 10800`.** A job projected under the floor is **not
  submitted short**: reduce cores, or merge the role into an adjacent job. Refusing to submit is the
  correct behaviour and must be an explicit non-zero exit, as `slurm/exact_ged/launcher.sh:183` does.

## The sizing problem — read this before designing the launcher
The naive projection is ~133 core-hours for all four roles. On one 128-core node that is under an
hour, three orders of magnitude the wrong side of the floor. **The design problem here is the
opposite of T-03's: not how to split the work, but how to keep from splitting it.** Consequences:

- **One single-node job per role**, not a job array over datasets (Letter LOW is ~90 core-seconds;
  nine of ten array tasks would be minutes long) and not an array over pair chunks (correct for
  2,081 core-h, absurd for 130). `ged_pair_index.py`'s chunking is retained **for resumability inside
  one task**, not for fan-out.
- **Cores are computed, not fixed**:
  `cores = clamp(floor(measured_core_seconds / TARGET_SECONDS), 1, 128)`.
- **The rate is measured on the compute node, not assumed.** A `probe` stage runs *inside* the same
  job before production, on a seeded stratified sample of 3,000 pairs spanning every dataset and every
  `n` decile, single-process `time.process_time()`, writing `probe.json`. A separate probe job would
  itself violate the floor. If the measured rate implies a wall under the floor, the worker logs it
  loudly and continues — the *launcher* is where a submission is refused.
- Per-pair cost scales roughly as `max(n₁,n₂)³` and T-27's rate was probed at `n̄ = 29.5` while Suite 2
  reaches `n = 98`, so the projection is a **lower** bound on true cost. Wallclocks need real
  headroom: `12:00:00` for the three full-Suite roles, `24:00:00` for `aged-ubt`.

Also: the account has **three IsalSR jobs already running on `sr`** and 42 `sr` nodes were idle at
design time. Do not request more than two nodes' worth of cores across the wave.

## The gates — `approx_ged_gates.py`
A CLI with `--gate {G2,G3,G4,lb-consistency,all}`, each writing a JSON record and returning a
pass/fail exit code.

- **G2 — reproduction against T-27.** For `iam_letter_low`, `iam_letter_med`, `iam_letter_high` and
  `linux`, whose Suite-2 cohort is **identical** to Suite 1, the campaign's `BRANCH_FAST` and
  `BIPARTITE` values must equal T-27's `data/cells/{ds}__{CELL}.npz` `value` arrays **element-wise on
  all 3,602,615 pairs**. T-27's arrays are flat, in canonical `numpy.triu_indices(N, k=1)` order; the
  campaign output is a dense symmetric matrix — take its upper triangle in the same order. This is
  the strongest gate available: one comparison covers loader, cost model, options string,
  symmetrisation and pair ordering against a census already on record.
- **G3 — bracket validity.** `lb_matrix ≤ ub_matrix` on every Suite-2 pair, at tolerance `1e-9`; and
  `lb ≤ exact ≤ ub` against T-03's certified values in
  `extended_merged_exact_ged/computed/{iam_letter_*,linux}.npz` (`certified_mask` selects). Report
  violations with their pair indices, not just a count.
- **G4-verify — structural, on the written files.** Symmetric to machine precision; diagonal zero;
  every entry finite and `>= 0`; off-diagonal exact-zero fraction recorded and `< 0.99`;
  `certified_mask` diagonal `True`; all ten CONTRACTS §4 keys present with the stated dtypes. This
  duplicates the merge's own gate **on purpose** — an independent reader of the finished file.
- **`lb-consistency`** — re-run `BRANCH_FAST` in-process on a seeded 5,000-pair sample per dataset and
  compare to `LB/{key}.npz`. The cross-check that the three separate role campaigns saw the same
  lower bound, at negligible cost.

**Tolerance matters and has burned this project twice.** GED under D6 is integer-valued and stored as
float; T-03 recorded two successively tighter guesses (`1e-9`, then `1e-6`) both reporting storage
noise as disagreement against a *third-party* file. Against **our own** output, exact equality is the
right expectation for G2 and `1e-9` for inequality comparisons — state which you use and why in every
gate's JSON record.

## Environment bootstrap
```bash
PY=~/.conda/envs/isalgraph-cpp/bin/python
export PYTHONPATH=~/opt/build_gedlib/graphkit-learn
cd "$(git rev-parse --show-toplevel)"
```
Do not put `<worktree>/src` on `PYTHONPATH`, and **do not import `isalgraph`**. A subagent's `cd`
does not persist between Bash calls, so prefix every command with
`cd "<your absolute worktree path>" && …`.

## Verification commands
```bash
$PY -m pytest tests/unit/test_approx_ged_gates.py -q
$PY -m pytest tests/unit/ -q                       # before your final commit
$PY -m ruff check benchmarks/ tests/
bash -n slurm/approx_ged/*.sh                      # every script
shellcheck slurm/approx_ged/*.sh || true           # if available; record what it says
```

## Data and shared resources
Read-only, under `SANDISK=/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph`:
- `$SANDISK/results/reports/T-27-ged-bound-bakeoff/data/cells/*.npz` — T-27's 60 recorded cells, key
  `value`. Your G2 reference.
- `$SANDISK/results/reports/T-27-ged-bound-bakeoff/data/index/*.npz` — pair indices and exact values.
- `$SANDISK/data/source/GED_PRECOMPUTED/extended_merged_exact_ged/computed/*.npz` — T-03's exact
  census and **the schema your G4-verify checks against**. Your G3 reference.
- `$SANDISK/data/exported/linux.npz` — an exported cohort in the input schema.
- **No Picasso.** Not even `sbatch --test-only`. The orchestrator submits.

## Definition of done
1. `launcher.sh --dry-run` prints every `sbatch` invocation it would issue, with resolved cores,
   wallclock, constraint, dependencies and job names, and issues **none**.
   `--stage {probe,lb,ub,ubs,ubt,merge,all}`.
2. The launcher **refuses with a non-zero exit** when a projected per-job wall falls below
   `FLOOR_SECONDS`, and a test or dry-run demonstrates the refusal with a small synthetic rate.
3. `bash -n` clean on every script; every `#SBATCH` flag lives on the launcher's command line, not in
   a worker header, matching `slurm/exact_ged`'s split.
4. Workers stage to `$LOCALSCRATCH`, install `TERM`/`INT` traps, mirror the tree back on exit, and
   delete shards only **after** the merge's structural gate passes.
5. **G2 verified in both directions on real recorded data**: build a contract-shaped
   `LB/linux.npz`-like file from T-27's `linux__BRANCH_FAST.npz` values, assert the gate passes; then
   perturb a single entry and assert it fails with that pair's index named. Same for G3 and G4-verify
   (an all-zero matrix, an asymmetric matrix, a non-zero diagonal, a missing key).
6. Every gate writes a JSON record naming the tolerance it used and why.
7. `README.md` states the submission order, the dependency chain, and what a human must check between
   stages.
8. All work committed on your branch; working tree clean. Work log written and committed.

## Explicitly out of scope
- **Submitting, transferring, or connecting to Picasso in any way.**
- The calibration ladder (exact GED above n = 12) and its worker — a later wave.
- Any analysis, figure, correlation, bootstrap or D13 evaluation.
- Editing `slurm/exact_ged/`, the runner, the merge, or the exporter.
- Re-running T-03's gate 2 (`ged_bounds.py` cross-check). T-27 discharged it; T-05-design §4 says so.

## Work log — mandatory
Maintain `.claude/notes/2026-08-13-t05-bounds/t05-slurm.md` using the template at
`.claude/notes/2026-08-13-t05-bounds/NOTE-TEMPLATE.md` **verbatim** (read it first). Write it
continuously. Commit it last with `docs(notes): t05-slurm work log`.

## Peers in this wave
- `main` — the orchestrator. Message it for ambiguities, contract defects, blockers, or anything
  needing a decision outside your ownership. **It owns every Picasso interaction and will submit your
  scripts itself.**
- `wave-t05-export` — exports the ten Suite-2 datasets and emits the subsample pair list. Owns
  `export_graphs_suite2.py`, `approx_ged_sampling.py` and their tests. Its output schema (CONTRACTS
  §2) is what your workers point `--input` at.
- `wave-t05-runner` — implements the runner and merge CLIs you invoke. Owns `ged_backends.py`,
  `ged_exact_runner.py`, `ged_merge_shards.py`, `approx_ged_crossfill.py` and their tests. **Code
  your gates against CONTRACTS, not against its implementation** — you are the independent check on
  it. Message it only about the CLI surface, and message `main` if the contract itself looks wrong.

## Final message format
At most 15 lines: STATUS, BRANCH, WORKTREE, HEAD, LOG, TESTS (counts + command), then three bullets
on what you built, anything the orchestrator must know, and anything unfinished. Include the resolved
core counts and wallclocks your launcher would use at the projected rates.
```

## 2. Understanding and plan

**Restatement of the task in my own words:** Build the Picasso submission machinery for T-05's four
GED-bound campaigns — a launcher that sizes each single-node job from a rate so every job clears
SCBI's 7,200 s floor, and workers that stage on `$LOCALSCRATCH` and run the runner/merge CLIs a peer
is writing to the same contract. Separately, write the independent validation gates that read the
finished campaign files and decide whether they can be trusted, coding those gates against
`CONTRACTS.md` rather than against the peer's implementation.

**Approach chosen:** Copy the structural conventions of `slurm/exact_ged/` (launcher owns every
`#SBATCH` flag, workers source a shared `_env.sh`, `$LOCALSCRATCH` staging with `EXIT`/`TERM`/`INT`
traps, `_clean_job_id` for Picasso's ANSI-emitting Lua wrapper) and change only what T-05's sizing
argument requires: single-node jobs per role instead of arrays, `floor` instead of `ceil` in the core
formula, and a `--group` mechanism that makes "merge the role into an adjacent job" executable rather
than a README sentence. For the gates, read only `numpy` and the two recorded references
(T-27 cells, T-03 computed), and go directly to `gklearn.gedlib` for `lb-consistency` rather than
through the peer's `GedlibBackend`, because routing the independent check through the code it checks
would make it circular.

**Alternatives considered and rejected:**
- A job array over datasets or pair chunks — rejected by T-05-design §5 for this workload
  (Letter LOW is ~90 core-seconds; nine of ten tasks would be minutes long). Retained the chunking
  flags for in-task resumability only.
- `ceil` in `cores_for_single_task`, as `slurm/exact_ged/launcher.sh:95` uses — rejected. `ceil`
  makes the projected wall **≤** `TARGET_SECONDS`, which can land under `FLOOR_SECONDS`; `floor`
  makes it **≥** `TARGET_SECONDS`. T-03 could use `ceil` because its work was 2,081 core-h and the
  ceiling never bound; here the floor is the only binding constraint, so the rounding direction is
  load-bearing. Documented in the launcher and the README.
- Reusing `ged_gates.py`'s `GateResult`/`environment_record` — rejected. `ged_gates.py` imports
  `ged_backends`, which peer `wave-t05-runner` is actively editing; importing it would couple my
  gate module to a file under concurrent modification and would make the module fail to import when
  GEDLIB is absent. Duplicated ~30 lines of record scaffolding instead.
- Driving `lb-consistency` through `GedlibBackend` — rejected as circular. The gate exists to check
  the campaign; running it through the campaign's own backend checks only determinism.
- A single `--sec-per-pair` flag as T-03 uses — rejected. The four roles differ by a factor of
  ~20,000 in per-pair cost (`5.6e-4` s/pair for `lb`, `~12` s/pair for `ubt`); one flag cannot size
  four jobs.

**Plan as executed:**
1. Confirm worktree, read `CONTRACTS.md`, `T-05-design.md` §3–§5, all of `slurm/exact_ged/`, invoke
   the `picasso-sbatch` skill.
2. Measure the real reference data (T-27 cells, T-27 index, T-03 computed) to learn the actual keys,
   dtypes, pair ordering and NaN policy before writing a line of gate code.
3. Write `slurm/approx_ged/_env.sh`, `launcher.sh`, `worker_bounds.sh`, `worker_subsample.sh`,
   `worker_crossfill.sh`, `README.md`. `bash -n` + `shellcheck` each. Commit.
4. Write `approx_ged_gates.py` (G2, G3, G4, `lb-consistency`). Commit.
5. Write `tests/unit/test_approx_ged_gates.py` against the **real** recorded data, both directions
   (pass on true data, fail on a perturbed copy naming the pair index). Commit.
6. Run `bash -n`, `shellcheck`, `ruff`, the new test file, and the whole `tests/unit/` suite. Record
   verbatim output. Commit the log.

**Deviations from the plan:** recorded in §7 as they arose.

## 3. Changes made

*(filled in as work proceeds)*

## 4. Tests

*(filled in as work proceeds)*

## 5. Test results

*(filled in as work proceeds)*

## 6. Verification beyond unit tests

*(filled in as work proceeds)*

## 7. Decisions, assumptions, open questions

*(filled in as work proceeds)*

## 8. Coordination

*(filled in as work proceeds)*

## 9. Deliberately not done

*(filled in as work proceeds)*

## 10. Risks and follow-ups

*(filled in as work proceeds)*

## 11. Self-assessment against the definition of done

*(filled in as work proceeds)*
