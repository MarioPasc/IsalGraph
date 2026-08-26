# T-06 exhaustive re-run — handoff

Written 2026-08-25. **Read this before touching anything under `T06_exhaustive/`.**

~~You are picking up mid-flight.~~ A Picasso campaign ~~is running~~ recomputed the
IsalGraph arm of T-06 with the *exhaustive* canonical string instead of the pruned one,
plus a greedy ablation. Figures and tables are already rebuilt and are waiting for that
data. **Your job is to land the results and regenerate — not to redesign anything.**

---

## 0. ✅ EXECUTED AND CLOSED — 2026-08-26

**The three steps in §3 are done and the campaign is complete.** Everything below this
section is the brief as written on 2026-08-25 and is preserved unchanged except where a
statement was measured to be wrong, in which case it is **struck and corrected in place**.
Read §0 for what happened; read the rest for why it was set up that way.

### The question this campaign existed to answer

> **Does the exhaustive arm beat nauty-sparse6 at `n = 40`? — NO.**
> **342.4 bits against 336.0.** It needed below 336 and closed 6.3 of the 12.7-bit gap.

| median entropy bits | `n = 20` | `n = 40` |
|---|---|---|
| `isalgraph_exhaustive` (hybrid) | **114.1** | **342.4** |
| `isalgraph_pruned` | 136.3 | 348.7 |
| `sparse6_nauty` | 144.0 | **336.0** |
| `isalgraph_greedy` | 130.0 | 355.0 |

The pruned and nauty-sparse6 columns reproduce §3's 136 / 349 and 144 / 336 exactly, which
is what certifies the staging. At `n = 20` the arm wins outright by 20.8 %.

**Why it stops working, and it is not the encoding.** The D14 cascade substitutes the
pruned string when the 30 s budget expires, and that rate rises monotonically: 0 % at
`n ≤ 12`, 4.5 % at 13–20, 41.2 % at 21–30, 60.4 % at 31–40, 98.4 % at 41–60. **At the
`n = 40` stratum itself it is 96.8 %** — three of 93 graphs carry a true `w*_G`. Split by
completion, the gain is flat at **12.5–17.0 % in every band and highest at `n = 31–40`**.
`T-06-POSITIONING.md` §5's projection to ≈ 310 extrapolated an `n ≤ 26` measurement; the
encoding delivers, the clock does not.

### The confirmatory family

```
family_F2.json:  N_actual=79  closed_form=79  discrepancy=+0   (101 - 5*3 - 7)
                 BH over N_actual : 76 rejected at q=0.05      (T-06: 75 of 79)
                 120 a1 cells · 300 rho rows · 25 mrm fits
```

Same cardinality as T-06 and the same seven `c` cells. **`c` is unmovable by this
campaign** — see the correction in §4. What moved is the evidence inside the family.

### The greedy ablation

Non-invariance **50.7 / 68.8 / 81.2 / 91.0 / 84.7 %** at `n = 5–9`, against **0 %** for
both canonical arms, and rising with `n`. On bits it is a wash against pruned (shorter on
5,821 of 21,720, longer on 3,506) and loses cleanly against `w*_G` (longer on 8,094,
shorter on 1,999 — every one of those a fallback row). **`psi` proper was never measured
for it**; the rate above is a different statistic and `PSI_MEASURED` carries `--`.

Two invariants verified on all 18,461 completed graphs, **0 violations** either way:
neither greedy-min nor pruned is ever shorter than `w*_G`; `w*_G` is strictly shorter than
pruned on 7,316 (39.6 %).

### Where everything landed

`T-06-FILES.md` § "Addendum — the `T06_exhaustive` campaign" is the pointer index: the
185-cell Sandisk tree, the 165-cell Picasso tree and why they differ, the three partial
sets in `families/`, and the eight report artifacts. The full work log with every
derivation and every failure is `.claude/notes/2026-08-25-t06-exhaustive/log.md`.

### What is still open

1. **`fig1`–`fig3` do not carry the new arms and cannot** without recomputing
   `size_profile.json` (~45 min / 24 cores → Picasso, new output path). See criterion 3.
2. **The decision.** At 30 s the `n = 40` sentence is not available. One re-encode at a
   larger budget is the only route to it, and the 17 % gain sitting in the 39.6 % of
   `n = 31–40` graphs that do finish is the argument for it.
3. **`src_commit = unknown`** on every cell — §7, narrower than it was written.

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

~~The figure code is **data-driven**: series, legend groups, the small-multiples grid and
every table row derive from whichever representations the archive holds. Adding
`isalgraph_exhaustive` and `isalgraph_greedy` needs **no code edit**. If you find
yourself editing a figure module to make a new arm appear, stop — that is a bug in the
module, not a task.~~

> ## ⚠ CORRECTED 2026-08-26 — this is WRONG, and the failure is silent
>
> The figure *modules* are data-driven. The **registry they filter through is not**.
> `design.REPRESENTATIONS` is a hand-written literal of 11 entries and
> `design.present()` drops unregistered keys **deliberately** — *"a figure must never
> invent a style for a backend the registry does not know, because that is how a colour
> starts drifting between figures."* Every consumer goes through it: `fig_ic` via
> `present()`, `eval_size_profile.figures` via `design.ORDER` / `design.BY_KEY`,
> `tables` by iterating `REPRESENTATIONS`.
>
> Followed literally, this paragraph produces **four figures and four tables that
> regenerate successfully with both new arms missing, and no error anywhere.**
>
> Registering the arms in `design.py` is the fix and is *not* what the last sentence
> warns against: `design.py` is the single design source, and `fig_ic.py`, `tables.py`
> and `figures.py` were not edited. `tables.py` additionally needs the four measured
> per-key constants (`PSI_MEASURED`, `COMPLETION_FLOOR`, `EXECUTABLE`,
> `PAYLOAD_PER_BYTE`) — all `.get()`-guarded, so a missing entry renders a blank or a
> `?` rather than raising.
>
> **The colour and display name for a new arm exist nowhere and must be chosen.** See
> `.claude/notes/2026-08-25-t06-exhaustive/log.md` § "The handoff was wrong that no code
> edit was needed" for what was picked and what is worth a second look.

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
- **Do not assert `N_actual == 79` for the new arm.** It is `101 − 5·3 − c`, ~~and `c` is the
  completion term the campaign exists to move~~. The internal `discrepancy == 0` check stays.

  > **Measured 2026-08-26: `c` is structurally unmovable by this campaign, and
  > `N_actual` came back 79.** `c` counts *comparator* cells failing completion. All
  > seven are `agm_cam` (6) and `min_dfs` (1) — competitor encodings **reused verbatim**
  > from `T06/encodings/`, which this campaign does not recompute. The one arm whose
  > completion it does change is the reference arm, and the reference arm is **exempt
  > from `c`** (`preregistration.md` §5.1 consequence 2, D14 governs the reference arm);
  > the merge logs the exemption four times, for `aids_iam` 0.9630, `coil_del` 0.5021,
  > `mutagenicity` 0.7802 and `protein` 0.3638.
  >
  > So the instruction stands but its reason is inverted: **79 is a consequence, not an
  > assumption to avoid**. Final family — `101 - 5*3 - 7`, `discrepancy = 0`, 120 A1
  > cells, 300 ρ rows, and **76 BH rejections against T-06's 75**. What moved is the
  > evidence inside the family, not its cardinality.
  >
  > Two defects had to be cleared to get there, both recorded in
  > `.claude/notes/2026-08-25-t06-exhaustive/log.md`: a `completion_rates.json` generated
  > from a 30-cell tree gave `N_actual = 86` with `discrepancy = 0` (consistent and
  > wrong), and staging `isalgraph_pruned` / `isalgraph_canonical` onto Picasso — they
  > are needed for `fig4`, not for F2 — raised `FamilyError: c names an undeclared
  > representation`. **Picasso's F2 tree is 165 cells / 11 representations; the Sandisk
  > tree is 185 / 13 for the figures. Criterion 1's "matches Picasso" must be read that
  > way.**

---

## 5. Acceptance criteria

1. `T06_exhaustive/` on the Sandisk matches Picasso; `data/source/T06/` unchanged.
2. ~~Four figures and four tables regenerate without a code edit~~, and the tables compile
   (`pdflatex` over a stub preamble with `booktabs`, `rotating`, `amsmath`).
   **Corrected 2026-08-26:** the tables compile (0 errors) and all eight artifacts
   regenerate, but *"without a code edit"* is unachievable — see the correction under
   §3 Step 2. Two files changed, both registries: `design.py` and `tables.py`.
3. ~~Every figure legend shows `isalgraph_exhaustive` and `isalgraph_greedy` without one
   having been named in any figure module.~~
   **Corrected 2026-08-26: satisfiable for `fig4` only, and this is not a defect.**
   `fig1_rho_vs_size`, `fig2_rho_by_representation` and `fig3_absolute_scale` are built
   from the frozen `results/.../data/size_profile.json`, which holds **7 representations**
   and neither new arm. Putting the new arms in them means re-running `size_profile.py`
   (`--encodings`, `--ged-root`, `--approx-root`; ~45 min on 24 cores, bootstrap-dominated)
   to a **new** output path — a Picasso job, outside the three steps, and forbidden as an
   in-place rewrite by §4. `fig4_information_content` and three of the four tables carry
   both arms; `tab_representation_headtohead` correctly omits them because it is the
   *comparator* table and both arms are `is_ours`.
4. `$PY -m pytest tests/ -q` ≥ **2,618 passed / 321 skipped**. A drop needs an explanation.
5. `ruff check` and `mypy --strict` clean on `src/isalgraph/`.
6. The three headline numbers in §3 are reported with their provenance.
7. Nothing under §4 was done.

> **Environment, 2026-08-26.** Neither §3 nor §5 says so, but the workstation had
> **no `isalgraph-cpp` conda env** and the **Sandisk was unmounted**. Both are
> prerequisites for every criterion above and both had to be restored first. Check
> `isalgraph.engine() == "cpp"` and `ls /media/mpascual/Sandisk2TB` before starting.

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
7. **Trap 4's staging covers `distances/` and NOT `encodings/`, and F2 needs both.**
   Added 2026-08-26, after it cost 6 h 50 m. Picasso's `T06_exhaustive/encodings/` held
   only the 30 cells the encode array wrote; the competitor and `isalgraph_pruned` cells
   live on the Sandisk as symlinks into `T06/` and were never pushed up. Claim B reads
   `distances/` and was fine; **Claim A reads `encodings/` and had one arm to compare
   against itself**, so all 15 shards wrote `a1_cells: []`, reported `ok=3 fail=0`, and
   the merge died in the A2 post-hoc with `MultiplicityError: need >= 2 named methods,
   got 1`. Stage with `rsync -aL` so the symlinks resolve — 185 cells, 8.7 MB — and
   confirm `find encodings -name '*.npz' | wc -l` is 185, not 30, before submitting F2.
   `f2_worker.sh` guarded the distance side explicitly and had **no equivalent guard on
   `${ENC}`**. One is now in place — it counts *comparators* (representations that are
   not `isalgraph_*`) and is FATAL below one. Note the count that matters is comparators,
   not representations: run 1's tree held two representations and still emitted nothing,
   because A1 emits one cell per comparator and no IsalGraph arm is one.

---

## 7. Known defect, not yet fixed

The distance workers log `could not read git HEAD for /home/mpascual/research/code/IsalGraph`
— `benchmarks/eval_distance/schema.py` reads provenance from a hard-coded workstation
path, so ~~**every artifact this campaign writes carries an empty commit field**~~. Harmless
to the numbers, wrong for a citable artifact. Worth a small fix (take the repo root from
the output tree or an argument) before these become the published matrices. It was left
alone deliberately rather than patched mid-flight.

> **Measured 2026-08-26 — the defect is real but one field wide, not all of them.**
> Read back from a written cell (`suite2/protein__isalgraph_exhaustive`, encodings and
> distances alike):
>
> | field | value |
> |---|---|
> | `code_commit` | `d6a9f4b1033d3d8c6757f5b2ae95d1b77f532bd2` — **correct** |
> | `src_commit` | `unknown` — this is the hard-coded-path defect |
> | `isalgraph_engine` | `cpp` |
> | `isalgraph_build_hash` | `298fc1188bf1b051` |
> | `encode_budget_s` | 30.0 (encodings) |
>
> So the matrices **are** attributable to a commit and are citable today; what is lost is
> `src_commit`. The fix is still worth taking, at lower priority than the wording implied.
> The engine trap (§6.1) was avoided cleanly — no cell reports `python`.

---

## 8. State of the repo

Clean at `aeea524`. Everything in §1 and §3 is committed. The figure package lives at
`benchmarks/real_data/eval_t06_figures/` — `design.py` is the single design source and no
other module may define a colour, a font size or a display name.
