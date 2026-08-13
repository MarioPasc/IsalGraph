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
| Head commit | `922c53f` + this log commit |
| Started / finished | `2026-08-13` / `2026-08-13` |
| Status | complete |

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

**Created**
| Path | Purpose |
|---|---|
| `slurm/approx_ged/launcher.sh` | the only human entry point; every `#SBATCH` flag, the sizing arithmetic, the floor refusal |
| `slurm/approx_ged/_env.sh` | sourced by every worker: interpreter, `PYTHONPATH`, `$LOCALSCRATCH`, traps, and the frozen role table |
| `slurm/approx_ged/worker_bounds.sh` | roles `lb`/`ub`/`ubs` over the ten datasets, with the in-job probe |
| `slurm/approx_ged/worker_subsample.sh` | role `ubt` over the CONTRACTS §5 subsample |
| `slurm/approx_ged/worker_crossfill.sh` | CONTRACTS §4.2 cross-fill, then gates G4 and `lb-consistency` |
| `slurm/approx_ged/README.md` | submission order, dependency chain, the three human checks |
| `benchmarks/real_data/eval_setup/approx_ged_gates.py` | gates G2, G3, G4-verify, `lb-consistency` |
| `tests/unit/test_approx_ged_gates.py` | 40 tests, most against real recorded data |
| `.claude/notes/2026-08-13-t05-bounds/t05-slurm.md` | this log |

**Modified / Removed**: none. Nothing outside my ownership set was touched.

**Commits**
| SHA | Message |
|---|---|
| `8051c23` | `docs(notes): t05-slurm plan before implementation` |
| `270e6e3` | `feat(T-05): Picasso launcher/worker pair for the four bound campaigns` |
| `baa7482` | `feat(T-05): independent validation gates G2, G3, G4-verify and lb-consistency` |
| `3cd71b7` | `test(T-05): gates verified in both directions on real recorded data` |
| `922c53f` | `fix(T-05): apply orchestrator rulings 1 and 4, and the inf correction` |
| *(final)* | `docs(notes): t05-slurm work log` |

`git diff --name-only 885d98d8..HEAD` returns exactly the eight paths above.

## 4. Tests

**Tests created** — 40 in `tests/unit/test_approx_ged_gates.py`.

| Test group | What it verifies | The failure mode it catches |
|---|---|---|
| `test_g2_passes_on_t27_recorded_values` | a campaign built from T-27's real `BRANCH_FAST`/`BIPARTITE` values passes, 2 × 3,916 comparisons | a gate that rejects correct data |
| `test_g2_fails_on_a_single_perturbed_entry` | one changed bound in 3,916 is caught and pair `[3, 7]` is named | a gate that cannot fail |
| `test_g2_reports_graph_order_before_comparing_values` | a swapped cohort reports "graph order differs" with `n_compared == 0` | 3.6 M bounds compared against the wrong pairs (orchestrator amendment 1) |
| `test_g2_cannot_pass_without_a_graph_order_reference` | an unevaluable precondition fails, never passes vacuously | a silently unchecked precondition |
| `test_g2_full_coverage_is_3_602_615_pairs` | the four G2 datasets cover exactly the documented count | a partial run reported as the full gate |
| `test_g3_passes_on_the_real_bracket` | `BF ≤ BIPARTITE`, both bracket T-03's 3,870 certified exact values | — |
| `test_g3_catches_an_inverted_bracket` / `..._upper_bound_below_exact` | `lb > ub` and `ub < exact` are caught with pair indices | the only claim the large-`n` argument rests on |
| `test_g3_joins_a_superset_cohort_on_graph_ids` | a campaign cohort strictly containing the reference is joined, and the exact arm runs on the induced submatrix (3,916 pairs, 3,870 certified) | throwing away the largest `lb ≤ exact ≤ ub` arm above Letter (ruling 4) |
| `test_g3_skips_only_when_there_is_no_overlap` | zero graphs in common is the only case that skips | a positional comparison of unrelated graphs |
| `test_g3_selects_on_isfinite_not_isnan` | asserts the real census carries 92 `+inf` and 0 NaN | an `isnan` guard passing 92 infinities through, `inf <= x` being False silently |
| `test_g3_reports_an_inf_bearing_reference_as_censored_not_violated` | an `inf` inside `certified_mask` is counted and excluded | a censoring reported as a bracket violation |
| `test_g4_*` (10 tests) | all-zero, asymmetric, non-zero diagonal, missing key, wrong dtype, false `certified` diagonal, self-reported mask, empty options string, missing file | each of the silent-corruption modes CONTRACTS §4/§7 names |
| `test_g4_accepts_an_all_empty_labels_column` | `labels` is checked for presence and dtype only | a class-count assertion failing on LINUX/AIDS-GraphEdX (orchestrator amendment) |
| `test_legitimate_zeros_are_accepted` | 15.5 % zeros pass; all-zero fails | the CLAUDE.md per-pair `0 < v` rule rejecting correct Letter data |
| `test_lb_consistency_*` (3 tests) | GEDLIB end-to-end reproduces T-27's census on 400 sampled LINUX pairs; a perturbed LB is caught; the draw is seed-reproducible | the three role campaigns silently disagreeing on the lower bound |
| `test_launcher_*` (7 tests) | dry-run issues nothing, resolves 1/2/9/31 cores, exits 3 under the floor, `probe` refuses, `--group` merges, warns on flat projection, rejects a bad stage | a short job reaching SCBI's queue |
| `test_workers_carry_no_sbatch_header` | the launcher/worker split holds | four headers drifting apart |
| `test_launcher_does_not_use_the_bash_builtin_GROUPS` | regression on the bug found below | `--group` silently ignored |

**Coverage of what matters:** every gate is exercised in both directions — passing on real
recorded data and failing on a deliberately perturbed copy of that same data, with the perturbed
entry's pair index asserted in the JSON record. The launcher is exercised through `subprocess`
against its real code path, not a re-implementation.

**Not tested, and why:**
- **Anything on Picasso.** No SSH, no `sbatch`, not even `--test-only` (standing obligation 5).
  The workers' SLURM-side behaviour — `$LOCALSCRATCH` staging, the `TERM` trap firing on a
  wallclock kill, the copy-back — is verified only by `bash -n` and by reading. This is the
  largest untested surface and the orchestrator should treat the first submission as the test.
- **The runner and merge CLIs themselves.** Owned by `wave-t05-runner`; I code against CONTRACTS.
- **`shellcheck`** — not installed on this machine. Recorded rather than skipped silently.

## 5. Test results

**Command:** `PYTHONPATH=~/opt/build_gedlib/graphkit-learn $PY -m pytest tests/unit/test_approx_ged_gates.py -q -p no:randomly`

```
collected 43 items
tests/unit/test_approx_ged_gates.py .................................... [ 83%]
.......                                                                  [100%]
============================== 43 passed in 8.74s ==============================
```

**Command:** `PYTHONPATH=~/opt/build_gedlib/graphkit-learn $PY -m pytest tests/unit/ -q`

```
================== 8 failed, 950 passed, 1 skipped in 25.34s ===================
FAILED tests/unit/test_export_graphs.py::test_real_export_reproduces_the_locked_cohort[iam_letter_low]
FAILED tests/unit/test_export_graphs.py::test_real_export_reproduces_the_locked_cohort[iam_letter_med]
FAILED tests/unit/test_export_graphs.py::test_real_export_reproduces_the_locked_cohort[iam_letter_high]
FAILED tests/unit/test_export_graphs.py::test_real_export_reproduces_the_locked_cohort[linux]
FAILED tests/unit/test_export_graphs.py::test_real_export_reproduces_the_locked_cohort[aids]
FAILED tests/unit/test_export_graphs.py::test_real_export_all_five_totals
FAILED tests/unit/test_export_graphs.py::test_real_export_is_deterministic
FAILED tests/unit/test_real_aids_retains_within_split_structure
E   FileNotFoundError: GraphEdX dataset not found:
    /media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/GED_PRECOMPUTED/AIDS
```

**Result:** 950 passed, 8 failed, 1 skipped · **Run at:** `922c53f`.

**Failures and their resolution:** all eight are **environmental and pre-existing**, not caused by
this work. `GED_PRECOMPUTED/` on this machine contains only `datasets/` and
`extended_merged_exact_ged/`; the `AIDS/` source tree the exporter needs is absent (verified with
`test -d`). They live in `test_export_graphs.py`, which tests `export_graphs.py` — a file I did
not modify and which nothing I wrote imports, so a causal link is impossible. **I did not fix
them**: `export_graphs.py` is frozen (CONTRACTS §9) and the missing tree is not mine to restore.

**Lint:** `$PY -m ruff check benchmarks/real_data/eval_setup/approx_ged_gates.py
tests/unit/test_approx_ged_gates.py` → `All checks passed!`. `ruff check benchmarks/` reports 28
errors, **all pre-existing** in `eval_visualizations/` and `synthetic_data/`; confirmed identical
count with my changes stashed.

**`bash -n`:** clean on all five scripts. **`shellcheck`:** not installed (`command -v shellcheck`
→ not found), so the shell scripts have had syntax checking but no static analysis.

## 6. Verification beyond unit tests

| Circumstance | What was run | Evidence | Outcome |
|---|---|---|---|
| Real data — G2 | gates against T-27's 60 recorded cells | LINUX 2 × 3,916 comparisons, 0 disagreements; all four G2 datasets 2 × 3,602,615 | pass |
| Real data — G3 | against T-03's exact census | LINUX 3,870 certified of 3,916 (46 censored), 0 bracket violations | pass |
| Real data — `lb-consistency` | **GEDLIB end-to-end**, real solver | LINUX cohort rebuilt from the CSR export, `BRANCH_FAST` under D6 `[1,1,0,1,1,0]`, 400 sampled pairs, 0 mismatches against T-27 | pass |
| Reference-schema recon | measured before writing any gate | T-27 cells carry `value`/`value_fwd`/`value_rev`/**`meta`** (not `metadata`); flat, `triu_indices(N,1)`-ordered — verified `array_equal` against the index files' `pair_i`/`pair_j` | informed the code |
| **NaN discovery** | T-03 computed files | `linux` and `aids` report `max|A - A.T| = nan`: `ged_matrix` carries NaN on censored pairs | G3 now selects on `certified_mask` **and** `isfinite` |
| **Zero-fraction discovery** | T-03 `iam_letter_low` | 215,968 exactly-zero off-diagonal entries of 1,391,220 (15.5 %) | corrected the `0 < v` rule; see §7 |
| Launcher, dry-run | `--stage all`, `--stage probe`, `--group lb,ub`, `--rate-lb 0.00001`, `--bins`+`--probe-json` | 5 `[DRY-RUN] sbatch` lines and 0 submissions; probe and low-rate paths both `exit 3`; grouping yields `aged-lb-ub` at 3 cores; binned sizing yields `evidence=*:binned` | all as designed |
| Environment | Debian 6.1.0-52, bash 5.2.15, Python 3.11.15, numpy per env, GEDLIB via `~/opt/build_gedlib/graphkit-learn` (importable, `GEDEnvGXL` present) | | |
| **Not exercised** | Picasso | prohibited by standing obligation 5 | — |

## 7. Decisions, assumptions, open questions

**Decisions with a real trade-off:**

- **`floor` rather than `ceil` in `cores_for_single_task`.** Costs a little parallelism (`ubt`
  gets 31 cores where `ceil` would give 32); buys the guarantee that the projected wall is
  **≥ `TARGET_SECONDS`** and therefore never under the floor. `slurm/exact_ged` uses `ceil`
  correctly for its own workload, where 2,081 core-hours never approached the floor.
- **Deriving the per-pair rate from the contract's core-hours instead of writing it down.**
  A written-out `11.957142857` × 28,000 = 334,799.99996, which floors to **30** cores instead of
  31. Deriving costs one `python3` call per role and keeps the provenance visible.
- **`--group` as a real mechanism rather than a README sentence.** T-05-design §5 says a role
  that cannot fill two hours "is merged into the adjacent role's job". Making that executable
  cost ~15 lines (a role→group resolver, a colon-separated `ROLES`, a loop in the worker) and
  turns the launcher's refusal into something a human can act on in one flag.
- **`lb-consistency` calls GEDLIB directly, not `GedlibBackend`.** Costs ~30 lines of duplicated
  env setup; buys non-circularity. Running the independent check through the campaign's own
  backend would verify determinism where the intent is to verify correctness.
- **Duplicating ~30 lines of record scaffolding rather than importing `ged_gates.py`.**
  `ged_gates.py` imports `ged_backends`, which `wave-t05-runner` is actively editing; importing
  it would couple my gate to a file under concurrent modification and make the module unusable
  without GEDLIB installed.
- **G4 treats `seconds_matrix` asymmetry as recorded-not-fatal**, unlike the three value
  matrices. The decision §6.2 concern — "an upper-bound matrix filled in one orientation is not a
  distance matrix" — is about values. A timing matrix is diagnostic.

**Assumptions I proceeded on** (all messaged to `main`):

- **A stratified probe pair list will exist at `$DATA_DIR/probe_pairs.npz`.** If it does not, the
  worker falls back to a contiguous first chunk of `grec` and logs that the rate is biased low.
  What breaks if wrong: the first job's measured rate under-sizes the rest of the wave. Chosen
  because blocking on an input nobody in this wave owns would stall the whole track.
- ~~The merge CLI will grow a flat-output mode for `ubt`.~~ **Resolved by ruling 1**: a separate
  `approx_ged_subsample_merge.py` owned by `wave-t05-runner`. My invention is removed; the
  residual risk is that the module does not exist when `aged-ubt` reaches its merge step.
- **`aged-ubt`'s `--input` is a directory**, not a file, since the subsample spans ten datasets.
  Still an assumption about the runner's CLI; unresolved.

**Open questions for the orchestrator:** the four numbered items in my message — the merge-CLI
gap, the probe list's owner, the `0 < v` rule correction, and T-05-design §4's `aids` overlap
phrasing. I proceeded on the assumptions above rather than blocking.

## 8. Coordination

**Messages sent:** one to `main`, covering (1) the merge-CLI gap for `ubt`, (2) the unowned
stratified probe list, (3) the correction to CLAUDE.md's `0 < v < inf` rule with the 215,968-zero
evidence, (4) T-05-design §4's `aids`-overlap phrasing, (5) the `GROUPS` bash-builtin bug, (6)
T-03's NaN-on-censored-pairs, (7) confirmation both amendments landed plus the bin-table schema
for `wave-t05-export`, (8) the resolved sizing, (9) unowned working-tree changes, (10) the suite
result. No reply required to finish; all four questions have a recorded fallback.

**Messages received and how they changed the work — second message, four rulings.** All applied
in `922c53f`:

- **Ruling 1 (my finding upheld, resolved differently than I assumed).** The `ubt` merge is a
  **separate entry point**, `approx_ged_subsample_merge.py` with CLI
  `--shards --pair-list --out --role --method --options`, not a flat mode on the dense merger —
  T-03's dense path is closed and load-bearing and is not widened for a 28,000-row case.
  `worker_subsample.sh` repointed; my `--pair-list`-in-place-of-`--n-graphs` invention dropped.
  Its shards now carry no `--delete-shards` because that merger has no such flag; they live
  outside the mirrored `out/` tree on `$LOCALSCRATCH` and are wiped with the job.
- **Ruling 2 (upheld).** `wave-t05-export` will emit `probe_pairs.npz` and the bin table in the
  schema I specified. The loud-warning fallback stays, deliberately.
- **Ruling 3 (upheld on substance).** My read guard is correct. CLAUDE.md itself is the user's
  file and is not being edited on my report; the correction is surfaced with my evidence.
- **Ruling 4 — OVERRIDDEN, in my favour, and I was wrong.** I had G3 *skip* the exact arm on a
  cohort mismatch. Suite-1 `aids` (769) is a **strict subset** of Suite-2 `aids_graphedx` (819),
  because Suite 1 is Suite 2 plus `n_max = 12`, so skipping discards the largest
  `lb ≤ exact ≤ ub` arm available above Letter. G3 now **joins on `graph_ids`** and runs on the
  induced submatrix; the join reduces to the positional comparison when the cohorts are
  identical, so it is one code path. My instinct that no *positional* comparison is valid was
  right — the id join is the fix, not the skip.
- **Correction to my FYI 6, and I was wrong on the mechanism.** T-03's `ged_matrix` carries
  **`+inf`** on censored pairs, not NaN: `linux` 92 non-finite, all `+inf`, `n_nan = 0`; `aids`
  122,076, all `+inf`. The NaN I reported came from `inf - inf` inside my own symmetry
  difference. My guards already used `np.isfinite` throughout, so no defect shipped, but the
  regression tests were missing and are now added — including one where an `inf` sits *inside*
  `certified_mask`.
- **Fixed while applying ruling 4:** the violation reporting would have double-filtered
  (`lower[selected]` on an array already reduced by `selected`), naming the wrong pairs. Caught
  by restructuring to filter once. On a gate whose entire job is naming pairs, that would have
  been worse than not reporting them.

**First message — two CONTRACTS amendments and two requests.** All four implemented:
- *Amendment 1* — `graph_ids` is the loader's native id, so **no gate validates the form of an
  id**. `_graph_ids_match` checks identity against every available reference (T-27 index and
  T-03 computed) and runs as a G2 **precondition** that short-circuits before any value
  comparison. `test_g2_reports_graph_order_before_comparing_values` asserts `n_compared == 0`.
- *Amendment 3* — `worker_subsample.sh` reads `subsample_pairs.npz` and writes `subsample.npz`,
  and asserts via `readlink -f` that the two never resolve to the same path.
- *labels* — G4 checks presence and dtype only, with a test that an all-empty column passes.
- *Request 1* — binned sizing, with a loud fallback and an `evidence` tag per role.
- *Request 2* — `projected_wall_seconds` vs `realised_wall_seconds` in every job's report.

**Contracts I depend on and confirmed unchanged:** §3 roles and verbatim options strings, §4 ten
keys and dtypes, §4.1 `certified_mask` as derived proof, §4.2 cross-fill, §6 runner CLI, §8
Picasso environment, `FLOOR_SECONDS = 7200`.

**Noted:** `CONTRACTS.md:8` records base commit `34e3ade822...`; the wave base is `885d98d8...`.
Informational only — no contract content depends on it.

## 9. Deliberately not done

- **The calibration ladder** (T-05-design §6, exact GED above `n = 12`, ~300–500 core-h) — out of
  scope, a later wave. No worker here computes it and `--stage` has no rung for it.
- **Any analysis, figure, correlation, bootstrap or D13 evaluation** — out of scope.
- **T-03's gate 2** (`ged_bounds.py` two-sided cross-check) — T-27 discharged it and
  T-05-design §4 says so; G2 supersedes it at 9,000× the sample size.
- **G1 (cohort)** — belongs to `wave-t05-export`, which owns the exporter.
- **Editing `slurm/exact_ged/`** — read from extensively, modified not at all.
- **Fixing the 8 pre-existing `test_export_graphs.py` failures** — environmental, and
  `export_graphs.py` is frozen.
- **Restoring the unowned working-tree deletions under `.claude/skills/`** — not mine; reported
  rather than touched.

## 10. Risks and follow-ups

| Item | Severity | Detail | Suggested owner |
|---|---|---|---|
| `ubt` merger does not exist yet | **high** | Ruling 1 assigns `approx_ged_subsample_merge.py` to `wave-t05-runner`. `worker_subsample.sh` now invokes it with the ruled CLI, but I have never seen it run; if the module is absent or its flags differ, `aged-ubt` fails *after* spending its 93 core-hours | `wave-t05-runner` |
| No Picasso-side execution anywhere | **high** | `$LOCALSCRATCH` staging, the `TERM` trap, and the copy-back are verified by `bash -n` and by reading only. Treat the first `--stage lb` as the integration test and read its log before submitting the rest | orchestrator |
| Probe list absent → biased-low rate | medium | The fallback is loud but still under-sizes. `probe_stratified: false` in the run report is the tell | `wave-t05-export` |
| Runner CLI drift | medium | I coded against CONTRACTS §6 without sight of the implementation. The flags most likely to differ: `--compute`, `--role`, and `--input` accepting a directory for `ubt` | orchestrator to reconcile at merge |
| Projections are lower bounds | medium | T-27 limitation 3. Wallclocks carry 3–4× headroom, which is deliberate but not unlimited; if realised ≫ projected the `ubs`/`ubt` submissions need re-sizing | human check 1 in the README |
| `shellcheck` never run | low | Not installed here. Worth one pass before the first submission | orchestrator |
| G2 needs ~700 MB transient RAM on `iam_letter_high` | low | The full-coverage test builds 2,059² matrices; marked `slow` | — |

## 11. Self-assessment against the definition of done

| # | Criterion | Met | Evidence |
|---|---|---|---|
| 1 | `--dry-run` prints every `sbatch` line, issues none; `--stage {probe,lb,ub,ubs,ubt,merge,all}` | yes | `test_launcher_dry_run_issues_no_sbatch` — 5 `[DRY-RUN] sbatch` lines, resolved cores/wallclock/constraint/dependency/job-name; all seven stages accepted, `test_launcher_rejects_an_unknown_stage` |
| 2 | Refuses with non-zero exit under `FLOOR_SECONDS`, demonstrated | yes | `test_launcher_refuses_a_job_under_the_two_hour_floor` (exit 3, `--rate-lb 0.00001`) and `test_launcher_probe_stage_refuses_and_submits_nothing` |
| 3 | `bash -n` clean; every `#SBATCH` on the launcher's line | yes | `bash -n` on all five; `test_workers_carry_no_sbatch_header` |
| 4 | `$LOCALSCRATCH`, `TERM`/`INT` traps, whole-tree mirror, shards deleted only after the gate | partial | Present and reviewed in `_env.sh`/workers, and shard deletion is delegated to the merge's own post-gate `--delete-shards` (CONTRACTS §6.2) with no `rm` of my own. **Never executed on a cluster** — see risk 2 |
| 5 | G2/G3/G4 verified in both directions on real recorded data | yes | 20 real-data gate tests; each perturbation test asserts the named pair index. `lb-consistency` additionally runs GEDLIB end-to-end |
| 6 | Every gate writes a JSON record naming its tolerance | yes | `test_every_gate_record_names_its_tolerance_and_why` asserts non-empty `tolerance` and a rationale > 80 chars on G2/G3/G4 |
| 7 | `README.md` states submission order, dependency chain, human checks | yes | `slurm/approx_ged/README.md` — three numbered human checks, the `afterok` rationale, the traps section |
| 8 | All work committed, tree clean, log committed | yes | five commits; `git diff --name-only base..HEAD` is exactly my ownership set |

**Overall.** I am confident about the gates: they run against the real T-27 and T-03 records
rather than fixtures, they fail on perturbed copies of that same data with the offending pair
named, and `lb-consistency` closes the loop through the real GEDLIB solver — reproducing T-27's
recorded `BRANCH_FAST` on 400 LINUX pairs at exact equality. I am confident about the launcher's
arithmetic, which is exercised through its real code path and which refused every case it should.

I am **not** confident about anything that only runs on Picasso. The workers have never executed;
their `$LOCALSCRATCH` staging, signal traps and copy-back are verified by reading and by `bash -n`
alone, and `shellcheck` was unavailable. **Scrutinise first**: whether
`approx_ged_subsample_merge.py` exists with the ruled CLI when `aged-ubt` reaches its merge — that
failure costs 93 core-hours *after* the compute is done. Second: whether `wave-t05-runner`'s
runner CLI matches the contract I coded against, particularly `--compute`, `--role`, and `--input`
accepting a directory for the subsample role.

Two of my own findings turned out to be partly wrong, and both are worth the orchestrator's
attention as evidence about how much to trust the rest. I reported T-03's censored pairs as NaN
when they are `+inf` — my guards were already right (`np.isfinite`) but my *explanation* was not,
and I had drawn it from a `nan` that my own `inf - inf` symmetry difference produced. And I had
G3 skip the AIDS overlap as an unsafe positional comparison when the correct move was an id join
on a structural subset. In both cases the code was safe and the reasoning was not, which is the
failure mode to watch for in what I have written.
