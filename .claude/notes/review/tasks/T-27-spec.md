# T-27 — GED bound bake-off: select both ends by measurement

**Status**: OPEN, specified 2026-08-13 by T-02. **Not yet started.**
**Owner**: unassigned — hand to an agent via `/review-ticket T-27`.
**Priority**: **P0. Gates T-05.** No production distance matrix may be computed before this closes.
**Depends**: T-03 (done — supplies the ground truth) · **Blocks**: T-05, and therefore T-06.
**Estimate**: 1–2 days. Compute ≈ 5–20 core-hours.
**Serves**: AE.1, R3.5b, R3.7a · **Decision**: [decisions](../plan/decisions.md) 26
**Read first**: [approx_ged](../plan/approx_ged.md) §2–§3 · [gedlib](../plan/gedlib.md) (**all of it —
two traps fail silently**) · [exact_ged](../plan/exact_ged.md) §4 · [preregistration](../plan/preregistration.md) §6

---

## 1. Why this ticket exists

The revision's entire large-`n` argument rests on a **proven bracket** `LB ≤ GED ≤ UB`. Two methods
were named as primary, and neither choice is currently defensible at the standard R3 applied last
round — they checked thirteen of thirteen checkable claims.

| End | Named primary | What actually supports it |
|---|---|---|
| **Lower** | `BRANCH_FAST` | literature dominance (`BED ≥ LED`, `BED ≥ HED`) **plus 400 LINUX pairs at n̄ = 8.71** — licensing a regime that runs to `n = 98` |
| **Upper** | `IPFP` | **nothing measured.** [approx_ged](../plan/approx_ged.md) §2 states it in its own words: *"that is now a prediction to test, not a recorded fact"*. The only measured UB is our own BP implementation, at **+135 % on LINUX** |

The LB case is the *same* generalisation failure that produced six retired figures in T-25 — measured
on one dataset, printed as a cohort property, all six wrong in the flattering direction — one review
round after being caught doing it. A reviewer asking "why IPFP?" currently gets a citation.

**The reason to fix it now rather than argue it:** the ground truth exists.

---

## 2. The ground truth — T-03 delivered the complete Suite-1 exact GED

**This is new since the plan was written and it is what makes T-27 cheap.**

| | |
|---|---|
| Pairs | **3,897,911** — the complete Suite-1 census, all five datasets, all pairs |
| Certified exact | **98.43 %** (`LB == UB` from the solver) |
| Interval-censored | 1.57 % — **excluded from the tightness comparison**, reported per dataset |
| Cost model | D6 unit: `[1, 1, 0, 1, 1, 0]` |
| Location | `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/GED_PRECOMPUTED/extended_merged_exact_ged/` — `computed/`, `reference/`, `manifest.json`, `PROVENANCE.md` |
| Mirror | `results/exact_ged/` in-repo, and on Picasso |
| Caveat carried from T-03 | the **exact solver changed** — `ANCHOR_AWARE_GED` is non-deterministic and **not exact**. Read `.claude/notes/review/tasks/T-03-design.md` before assuming otherwise |

Per-dataset pair counts: Letter LOW 695,610 · Letter MED 784,378 · Letter HIGH 2,118,711 ·
LINUX 3,916 · AIDS (GraphEdX, `n ≤ 12`) 295,296.

At GEDLIB's measured ~100 µs/pair, one method over the full census is **≈ 0.9 core-hours**. The whole
eight-method grid is **≈ 7 core-hours**. There is no reason to select by argument.

---

## 3. The grid — every proven method, per dataset

| End | Methods | Accessor | Reference |
|---|---|---|---|
| **Lower** | `BRANCH`, `BRANCH_FAST`, `BRANCH_TIGHT`, `STAR` | `get_lower_bound()` | Blumenthal & Gamper, *IEEE TKDE* 30(3):503–516, 2018 (first three); Zeng et al., *VLDB* 2009 (STAR) |
| **Upper** | `IPFP`, `REFINE`, `BIPARTITE`, `BP_BEAM` | `get_upper_bound()` | Bougleux et al. 2017; Zeng et al. 2009 / GEDLIB; Riesen & Bunke, *IVC* 27(7):950–959, 2009; Neuhaus & Riesen |

**`HED` is excluded and that is a finding to report, not an omission.** Fischer et al.,
***Pattern Recognition*** 48(2):331–343, 2015 — the venue-fit citation — returns
`get_lower_bound() = 0.00` and `get_upper_bound() = inf` under default options
([gedlib](../plan/gedlib.md) §5). **Spend at most 2 hours** trying to configure it; if it does not
yield a finite bound, record the attempt and the options tried, and move on. Do not let it block.

**5 datasets × 8 methods = 40 cells.** Every cell is attempted and reported, failures included.

### Sampling, pre-declared before the run

Run the **full census** (3,897,911 certified pairs per method). **Fallback, fixed now**: if measured
throughput projects past **40 core-hours** for the grid, drop to a stratified sample of **100,000
pairs per dataset**, seed 42, stratified by `max(n₁, n₂)`. Declare which was used in every table.
Do not decide this after seeing results.

---

## 4. What is measured per cell

| # | Quantity | Definition |
|---|---|---|
| **M1** | Relative error | UB: `(UB − exact)/exact`; LB: `(exact − LB)/exact`. Mean, median, IQR, 95th pct |
| **M2** | Absolute error | same, in edit operations — `exact = 0` pairs make M1 undefined; count and exclude them, and **report the count** |
| **M3** | Certification rate | fraction with `bound == exact` — where the bound is exact for free |
| **M4** | Validity | fraction violating `LB ≤ exact` or `UB ≥ exact`. **Any violation is a bug**, in GEDLIB or in the harness. Halt and report — see §8 |
| **M5** | ρ(bound, exact) | Spearman, per dataset, with graph-level bootstrap CI (D2, 2,000 replicates, seed 42) |
| **M6** | ρ(Lev, bound) | the quantity the paper actually reports, against ρ(Lev, exact) as the anchor |
| **M7** | Cost | µs/pair, `time.process_time()`, single thread, reported at n̄ per dataset |
| **M8** | Symmetry | fraction where `d(G,H) ≠ d(H,G)`, and the mean gain from `min`. **Every UB method is expected to fail this** — see §7 |

**M6 is the one that decides the paper.** M1 measures the bound; M6 measures whether swapping the
bound changes the reported correlation. A method can be looser on M1 and still preferable if
`ρ(Lev, bound)` tracks `ρ(Lev, exact)` more closely, because that is the quantity the manuscript
prints.

---

## 5. Selection rule — frozen 2026-08-13, before the run

> **Per end**, the primary method is the one minimising **mean relative error (M1)** on that
> dataset, subject to **M4 = 0** (no bound violated), **M3 well-defined**, and **M7 < 1 ms/pair at
> n̄ = 30**.
>
> **A single global primary is declared only if the same method wins on ≥ 4 of the 5 datasets.**
> Otherwise the primary is declared **per dataset**, and the paper says so — heterogeneity here is a
> result about GED approximation, not a defect in this study.
>
> **Ties (within 2 % relative) break on M7 (cost), then on M6 (agreement with ρ(Lev, exact)).**
> Ties **never** break on which method flatters IsalGraph's ρ.

This mirrors [competitors](../plan/competitors.md) §3.4's structure deliberately: selection criteria
that are blind to the outcome we would prefer.

---

## 6. Deliverables — the plot / p-value / literature triad

**One per end (lower, upper). Both go into the manuscript's §3.3 or supplementary, and into T-14.**

### 6.1 The plot

Built through **`isalgraph.viz`** — palettes, IEEE sizes, `save_figure`. **Do not hand-roll
matplotlib in a figure script** (project rule; `benchmarks/plotting_styles.py` re-exports from
`isalgraph.viz.style` so the published palette cannot drift). A new view module is fine; a bare
`import matplotlib` in a script is not.

Two panels per end, one figure:

- **(a) Relative error vs `n`** — x = `max(n₁, n₂)` from 2 to 12, y = mean relative error, one line
  per method, ribbon = IQR, faceted or coloured by dataset. **This is the panel that answers whether
  the choice transfers across `n`**, which is the whole question the 400-LINUX-pair measurement
  could not answer.
- **(b) Per-dataset ranking** — forest plot of mean relative error with bootstrap CI, methods on the
  y-axis, grouped by dataset. If the ranking is consistent, the figure shows it at a glance; if it is
  not, that is the result.

### 6.2 The p-values

**These are a selection procedure, not a hypothesis test, and they are explicitly OUTSIDE the
confirmatory family** ([preregistration](../plan/preregistration.md) §6). Report them as such, in
those words, so no reviewer reads them as an unregistered claim.

- **Within each end**, pairwise **Wilcoxon signed-rank** on paired per-pair relative error
  (the same pairs, so the pairing is exact), **Holm-corrected** within the end — 6 comparisons for
  4 methods.
- **Across datasets**, **Friedman + Wilcoxon–Holm + critical-difference diagram** over the 5 datasets
  (Demšar, *JMLR* 7:1–30, 2006). ⚠ **State the `N = 5` caveat in the caption** — [statistics](../plan/statistics.md)
  §4 locks that Friedman at `N = 5` separates almost nothing, which is why the exact regime gets no
  CD diagram in the main analysis. Here it is a *descriptive* aid to a selection, not a result. If it
  separates nothing, say so.
- Effect sizes lead, p-values support (D10). Report matched-pairs rank-biserial correlation beside
  every Wilcoxon.

### 6.3 The literature table

One row per method: **citation (authors, title, venue, volume:pages, year, DOI)**, complexity,
proof status (proven bound vs heuristic), what the source paper claims about tightness, and
**whether our measurement agrees with that claim**. The last column is the valuable one — a
disagreement with a published tightness ordering, measured over 3.9 M pairs, is publishable in its
own right and is exactly the kind of thing a *Pattern Recognition* reviewer rewards.

Include `HED` with its status, since it is the **Pattern Recognition**–venue citation and its
presence serves EiC.b.

### 6.4 The closing paragraph

~150 words tying the three together: what the literature predicted, what we measured, whether they
agree, and which method is primary at each end and why. This is the paragraph T-20 lifts.

---

## 7. Traps — every one of these fails silently

Read [gedlib](../plan/gedlib.md) in full. The four that will cost you a day each:

1. **Wrong accessor returns garbage, not an error.** `get_lower_bound()` on an upper-bound method
   returns **0.00**. `HED` returns `inf`. Neither raises. **Assert `0 < value < inf` on every single
   read** — a whole matrix can fill with zeros and look like a result.
2. **Import order.** `gklearn.gedlib.libraries_import` must load *before* `gedlibpy_gxl`, or
   `libdoublefann.so.2` is missing. **isort/ruff will reorder plain imports and break it** — use
   `importlib.import_module`, which formatters cannot touch.
3. **Upper bounds are not symmetric.** Every GEDLIB UB builds its edit path from a *directed*
   assignment. A matrix filled in one orientation **is not a distance matrix**. Measured on our own
   BP: tighter on 33.2 % of pairs, mean gain 1.15 edit operations. Fill both triangles and take
   `min`, or assert and fail loudly. The LB is symmetric and needs no treatment. **M8 measures this
   per method — do not assume, measure.**
4. **`fscratch` is a FILE-COUNT quota**, 250k soft / 400k hard, and a GEDLIB build tree is 50–90k
   files. Delete the build tree once the `.so` exists. The failure surfaces as
   `shutil.Error: [Errno 122]` mid-`copytree`, not as a compile error.

**Cross-check, do not skip**: `benchmarks/real_data/eval_setup/ged_bounds.py` implements BP and
BRANCH-FAST directly and is tracked with 35 unit tests. GEDLIB and it **must agree** on the same
pairs. Disagreement is a bug in one of them and we need to know which — that is what gate 2 exists
for ([exact_ged](../plan/exact_ged.md) §4).

---

## 8. Acceptance criteria

1. All 40 cells attempted; failures reported with the reason, not omitted.
2. **M4 = 0 across every cell.** Any bound violating `LB ≤ exact ≤ UB` **halts the ticket** — it is
   either a GEDLIB bug or a harness bug, and both are more important than the selection.
3. The §5 rule is applied **as written**, and the ticket states which branch it took (global primary
   vs per-dataset) and on what margin.
4. Both figures render through `isalgraph.viz` and are reproducible from a tracked script under
   `benchmarks/real_data/eval_setup/`. **Nothing in `scratchpad/`** — that is what lost fifteen
   measurement scripts from this project.
5. Unit tests for the harness, in `tests/unit/`, following `tests/unit/test_ged_bounds.py`.
6. The literature table's citations are **verified**, not recalled — DOI or venue page checked.
7. Article notes and a response-letter fragment emitted, per `review-close`.
8. `.claude/notes/review/plan/approx_ged.md` §2's production-assignment table is updated **in place**
   with the selected methods and the measurement that selected them.

## 9. Stop and ask

- **M4 > 0 anywhere** — a violated proven bound.
- **The ranking is inconsistent across datasets** (no method wins ≥ 4 of 5). The rule says declare
  per dataset; confirm before writing that into the paper, because it changes what §4 can claim.
- **`BRANCH_FAST` loses.** It is named in [decisions](../plan/decisions.md) 11 and in
  [approx_ged](../plan/approx_ged.md) §2 as primary, and changing it touches a signed decision.
- **ρ(Lev, bound) inverts the M1 ranking** — the tightest bound giving the *worst* agreement with
  ρ(Lev, exact). That is a real possibility and a genuinely interesting result; it needs a human
  before it becomes a selection.
- Compute projecting past **40 core-hours** — take the §3 fallback, then report.

## 10. Picasso

GEDLIB is installed and verified at
`/mnt/home/users/tic_163_uma/mpascual/fscratch/build_gedlib/graphkit-learn`
(export `PYTHONPATH` to the checkout — the build is in-place).

**The agent writes the launcher/worker pair via the `picasso-sbatch` skill and does not submit it.**
No `ssh`, no `rsync`, no `sbatch`. The orchestrator owns every cluster interaction. Respect the
**2-hour job floor** — group the 40 cells rather than submitting 40 short tasks; SCBI wrote to this
account about exactly that after a 12,600-task campaign of minute-long jobs.
