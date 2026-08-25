# T-06 exhaustive re-run — handoff

Written 2026-08-25. **Read this before touching anything under `T06_exhaustive/`.**

You are picking up mid-flight. A Picasso campaign is running that recomputes the
IsalGraph arm of T-06 with the *exhaustive* canonical string instead of the pruned one,
plus a greedy ablation. Figures and tables are already rebuilt and are waiting for that
data. **Your job is to land the results and regenerate — not to redesign anything.**

---

## 1. Read these first, in this order

| file | why |
|---|---|
| `.claude/notes/review/tasks/T-06-EXPERIMENTS-EXPLAINED.md` | what the two experiments measure, end to end. Start here if you do not know what Claim A and Claim B are |
| `.claude/notes/review/tasks/T-06-POSITIONING.md` | **what may and may not be claimed.** §5 is why this campaign exists; §7 is the decision table |
| `.claude/notes/review/tasks/T-06-FRAMING.md` §6 | the red lines. Do not cross them, do not soften them |
| `.claude/notes/review/tasks/T-06-article-notes.md` §10 | the "NOT claimable" list |
| `.claude/notes/review/tasks/T-06-FILES.md` | where every T-06 artifact lives |
| `.claude/notes/2026-08-25-t06-exhaustive/log.md` | the campaign agent's own work log |
| `.claude/CLAUDE.md` | engine rules, Picasso rules, reference test state |

Measurement scripts, all runnable, all committed:

- `.claude/notes/review/tasks/t06_bit_convention.py` — four bit conventions and what each decides
- `.claude/notes/review/tasks/t06_pruned_vs_exhaustive.py` — the length gap this campaign closes
- `.claude/notes/review/tasks/t06_exhaustive_ceiling.py` — how far exhaustive reaches on the C++ engine

---

## 2. What is running right now

```
2102929  encode      COMPLETED   30/30 cells, 0 failures
2106062  distances   COMPLETED
2106063  f2-shards   RUNNING     (afterok 2106062)
2106064  f2-merge    PENDING     (afterok 2106063)
```

Check it:

```bash
ssh picasso 'sacct -j 2102929,2106062,2106063,2106064 -X -n -P -o JobID,State,Elapsed'
ssh picasso 'ls ~/execs/isalgraph/logs/'
```

Output tree: `/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/isalgraph/T06_exhaustive/`
with `encodings/`, `distances/`, `families/`, `gates/`.

**Everything compute-heavy runs on Picasso. Nothing runs locally.** That is a standing
instruction from the PI, not a preference.

### If a stage failed

Re-run the launcher; completed cells are skipped:

```bash
ssh picasso 'cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalGraph && \
  bash slurm/t06_exhaustive/pipeline_launcher.sh --test-only'
```

Do **not** pass `AFTER_ENCODE` for a job that has already completed — Slurm rejects a
dependency on a job that has aged out of the active queue with `Job dependency problem`.

---

## 3. The scoped task

**Three steps. Nothing else.**

### Step 1 — copy the results back

When `2106064` (f2-merge) reaches COMPLETED, rsync `T06_exhaustive/` from Picasso to
`/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/T06_exhaustive/`.

A partial local run of the same campaign already sits there (11/15 cells, stopped
deliberately). **The Picasso results are authoritative — overwrite it.**

### Step 2 — regenerate figures and tables

```bash
R=/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/results/reports/T-06-full-recompute
E=<the new T06_exhaustive>/encodings
PY=~/.conda/envs/isalgraph-cpp/bin/python

$PY -m benchmarks.real_data.eval_size_profile.figures --profile $R/data/size_profile.json --out-dir $R/figures
$PY -m benchmarks.real_data.eval_t06_figures.fig_ic     --encodings $E --strata $R/data/claim_a_strata.json --out-dir $R/figures
$PY -m benchmarks.real_data.eval_t06_figures.tables     --strata $R/data/claim_a_strata.json --profile $R/data/size_profile.json --encodings $E --out-dir $R/tables
```

The figure code is **data-driven**: series, legend groups, the small-multiples grid and
every table row derive from whichever representations the archive holds. Adding
`isalgraph_exhaustive` and `isalgraph_greedy` needs **no code edit**. If you find
yourself editing a figure module to make a new arm appear, stop — that is a bug in the
module, not a task.

### Step 3 — report the headline numbers

The three the PI is waiting for:

- median entropy bits at **n = 20** and **n = 40** for the exhaustive arm, against the
  pruned arm's **136** and **349** and nauty-sparse6's **144** and **336**;
- whether the exhaustive arm now beats nauty-sparse6 at n = 40 (it needs to close 349 → below 336);
- the greedy ablation's ψ, to sit beside the pruned/exhaustive **0.0 %**.

---

## 4. Scope — what NOT to do

- **Do not redesign the figures.** They were iterated with the PI over several rounds and
  are settled: one panel plus a compression-ratio inset for the IC figure, a 3×2
  per-competitor grid inside the bracket regime for the ρ figure, family-grouped legends.
- **Do not add competitors, drop competitors, or change a bit convention.** Every one of
  those was decided and written up. `T-06-POSITIONING.md` §7 is the decision table.
- **Do not touch `data/source/T06/` or `results/reports/T-06-full-recompute/data/`.** That
  is the pre-registered record and it must stay byte-identical. Only `figures/` and
  `tables/` under the report are regenerated.
- **Do not regenerate T-04a's frozen grid.** `grid.py` builds it from
  `available_backends()`, which went 10 → 12 when the new arms were registered.
  Regenerating changes a pre-registered artifact's shape. It may be worth doing
  deliberately; it is not a side effect you take on your own.
- **Do not re-run anything locally.**
- **Do not assert `N_actual == 79` for the new arm.** It is `101 − 5·3 − c`, and `c` is the
  completion term the campaign exists to move. The internal `discrepancy == 0` check stays.

---

## 5. Acceptance criteria

1. `T06_exhaustive/` on the Sandisk matches Picasso; `data/source/T06/` unchanged.
2. Four figures and four tables regenerate without a code edit, and the tables compile
   (`pdflatex` over a stub preamble with `booktabs`, `rotating`, `amsmath`).
3. Every figure legend shows `isalgraph_exhaustive` and `isalgraph_greedy` without one
   having been named in any figure module.
4. `$PY -m pytest tests/ -q` ≥ **2,618 passed / 321 skipped**. A drop needs an explanation.
5. `ruff check` and `mypy --strict` clean on `src/isalgraph/`.
6. The three headline numbers in §3 are reported with their provenance.
7. Nothing under §4 was done.

---

## 6. Traps already paid for — do not rediscover these

1. **`PYTHONPATH` must be `$REPO` only, never `$REPO/src`.** The extension installs into
   site-packages; a src-first path shadows it, `engine()` silently returns `python`, and
   the campaign measures the pure-Python reference ~100× slower with a completely
   different censoring rate at the same nominal budget. The `picasso-sbatch` skill's
   template gets this wrong for this project.
2. **Every data root must be threaded explicitly.** `t06_cohort.DEFAULT_COHORT_ROOT` is a
   `/media/...` workstation path. Unset on the cluster, it fails with a confusing "cohort
   export not found". Array 2102923 died this way, 6/6 units per task.
3. **The distance worker requires `gates/gate_T06_reproduction.json`.** It refuses to
   compute any production matrix until T-04a's table is reproduced at max |Δ| = 0.0000.
   Array 2102963 died this way. The gate is staged now.
4. **Nothing GED-related lives on Picasso by default.** T-03/T-05 computed there but the
   results were pulled back and cleaned. `eval/ged_matrices`, `APPROX_GED` (LB/UB) and the
   competitor `distances/` were all staged for this run. Exclude `datasets/` when staging
   `APPROX_GED` — it is 35,604 files of GXL input used to *compute* GED, not to read it,
   and fscratch's limit is a **file count**.
5. **Registering a backend can break a test that enumerates them.** Eight test files sweep
   `available_backends()`. Run the full suite, not a targeted file.
6. **`pkill` is blocked** by this project's permission settings; `kill` by PID is not.

---

## 7. Known defect, not yet fixed

The distance workers log `could not read git HEAD for /home/mpascual/research/code/IsalGraph`
— `benchmarks/eval_distance/schema.py` reads provenance from a hard-coded workstation
path, so **every artifact this campaign writes carries an empty commit field**. Harmless
to the numbers, wrong for a citable artifact. Worth a small fix (take the repo root from
the output tree or an argument) before these become the published matrices. It was left
alone deliberately rather than patched mid-flight.

---

## 8. State of the repo

Clean at `aeea524`. Everything in §1 and §3 is committed. The figure package lives at
`benchmarks/real_data/eval_t06_figures/` — `design.py` is the single design source and no
other module may define a colour, a font size or a display name.
