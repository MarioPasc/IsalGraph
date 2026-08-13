# T-27 design — GED bound bake-off

**Written** 2026-08-13 by the orchestrator, **before any measurement run**.
**Spec**: [`T-27-spec.md`](T-27-spec.md) — this note refines it against measured state; where the two
differ, the difference is recorded in §1 with the evidence.
**Depends**: T-03 (done) · **Blocks**: T-05, T-06.
**Report deliverable**: `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/results/reports/T-27-ged-bound-bakeoff/`

---

## 1. Measured state — nine values differ from what the spec assumed

Every row below was read from the artifacts on 2026-08-13, not from the plan.

| # | Spec / plan said | Measured | Consequence |
|---|---|---|---|
| 1 | Ground truth is `ANCHOR_AWARE_GED` (spec §2) | **`networkx.graph_edit_distance`** — `metadata.ged_method == "networkx"` in all five `computed/*.npz` | None for correctness (T-03 already retracted `ANCHOR_AWARE_GED`); the report must not attribute the ground truth to GEDLIB, or the bake-off becomes GEDLIB-vs-GEDLIB |
| 2 | Censoring 1.57 %, "reported per dataset" | **AIDS 20.67 %** (61,038/295,296); LINUX 1.17 %; Letter ×3 **0 %**. The 1.57 % is a census-weighted average dominated by uncensored Letter | AIDS tightness is conditioned on the *solved* 79.3 %, and censoring is **not** missing-at-random — the solver timed out at 60 s on the harder pairs. Must be stated wherever an AIDS mean is printed. §3.5 recovers one-sided validity from the censored fifth |
| 3 | (not anticipated) | **306,768 certified pairs have exact GED = 0 off-diagonal** — Letter LOW 15.52 %, MED 14.04 %, HIGH 4.19 %; AIDS and LINUX 0 % | M1 is undefined on 8.0 % of the census, concentrated in one dataset family. Excluded from M1 (spec §4 M2 already requires this); **also inflates M3**, because every valid LB is exact for free at GED = 0 — §3.3 |
| 4 | GEDLIB "installed and verified on Picasso" (spec §10) | Also **not installed locally**; built locally 2026-08-13 from `jajupmochi/graphkit-learn` at `~/opt/build_gedlib/graphkit-learn` | §2 — the campaign runs locally, not on Picasso |
| 5 | ≈ 100 µs/pair (spec §2) | gedlib.md §5's own smoke numbers are **0.09–0.89 ms** on P₄/C₄ (4 nodes). 100 µs is the floor, not the rate | Budget re-derived in §2 from a measured rate, not the plan's estimate |
| 6 | "5 datasets" for the ≥ 4-of-5 rule (spec §5) | Three of the five are **IAM Letter LOW / MED / HIGH** — one 15-class letter corpus at three distortion levels, sharing graph identities and generator | The "5 datasets" are **3 independent corpora**. §3.2 keeps the frozen rule and adds a corpus-collapsed companion view |
| 7 | M7 gate: "< 1 ms/pair at n̄ = 30" (spec §5) | Suite 1 is `n ≤ 12` by construction; **no n̄ = 30 pair exists in the ground truth**. Dataset n̄: Letter LOW 4.7, MED 4.7, HIGH 4.7, LINUX 8.7, AIDS 10.3 | The frozen cost gate is **unevaluable on the bake-off corpus**. §3.4 adds a separate timing probe at n̄ ≈ 30 drawn from IAM GREC / Protein via the tracked `iam_gxl_loader.py` |
| 8 | Bootstrap: "graph-level, 2,000 replicates" | statistics.md §5's frozen tier table puts **all five Suite-1 datasets in tier 1 or 2** → 2,000 replicates, **all** induced pairs, no subsampling | Confirms the census bootstrap is in budget; no D15 subsample applies here |
| 9 | `results/reports/` is the report home | **Empty** — T-27 is the first report | Naming convention set here: `T-<id>-<slug>/` with `REPORT.md`, `figures/`, `data/` |

**Unchanged and confirmed**: pair census 3,897,911; certified 3,836,827 (98.43 %); cost model
`[1,1,0,1,1,0]`; per-dataset counts; Levenshtein matrices present for all five datasets in three
encoder variants (`exhaustive`, `greedy`, `greedy_single`), so M6 is computable without new encoding.

**Derived census used throughout**

| Dataset | Graphs | Pairs | Certified | Censored | Exact = 0 | **M1-eligible** | n̄ |
|---|---:|---:|---:|---:|---:|---:|---:|
| LINUX | 89 | 3,916 | 3,870 | 46 | 0 | **3,870** | 8.7 |
| AIDS (GraphEdX) | 769 | 295,296 | 234,258 | 61,038 | 0 | **234,258** | 10.3 |
| IAM Letter LOW | 1,180 | 695,610 | 695,610 | 0 | 107,984 | **587,626** | 4.7 |
| IAM Letter MED | 1,253 | 784,378 | 784,378 | 0 | 110,116 | **674,262** | 4.7 |
| IAM Letter HIGH | 2,059 | 2,118,711 | 2,118,711 | 0 | 88,668 | **2,030,043** | 4.7 |
| **Total** | 5,350 | **3,897,911** | **3,836,827** | 61,084 | 306,768 | **3,530,059** | — |

---

## 2. Where it runs — locally, not on Picasso

**Decision: the whole campaign runs on the workstation.** The spec assumed Picasso because that is
where GEDLIB was verified; the measured state removes the reason.

| | Picasso (spec §10) | **Workstation (chosen)** |
|---|---|---|
| GEDLIB | installed, verified 2026-08-11 | **built 2026-08-13**, same source, same recipe |
| Cores | queue-dependent | **24, idle** |
| Job floor | **2 h**, must group 40 cells | none |
| Data motion | rsync 38 MB ground truth + 45 MB results each way | **already local** |
| Queue latency | hours to days | zero |
| Quota | fscratch file-count limit, 50–90k-file build tree | `/home` 504 GB free, no file quota |

Budget from a **measured** rate rather than the plan's estimate (§1 row 5). Work is
12 passes over the census — 4 LB methods × 1 orientation + 4 UB methods × 2 orientations (§3.6) —
so 46.8 M method evaluations. At the gedlib.md §5 envelope of 0.09–0.89 ms this is **1.2–11.6
core-hours**, i.e. **3–30 minutes wall on 24 cores**. Well under the spec's 40-core-hour fallback
trigger, so §3 of the spec keeps the **full census** and the stratified-100k fallback is not taken.

**Rejected alternatives**: (a) Picasso — costs a queue wait and a 2-hour job floor to buy nothing;
(b) local subsample — the census is affordable, and subsampling would forfeit the one thing that
makes T-27 answer the generalisation objection at all.

**Consequence to honour**: M7 timings are workstation timings. They are internally consistent across
methods, which is what a *selection* needs, and the report states the machine.

---

## 3. Frozen before the run

Everything in this section is fixed now and committed before the first measurement.
**The spec's §5 selection rule is applied verbatim and is not modified.** §3.1–§3.5 add
*companions* — pre-declared secondary views reported beside the frozen rule, never in place of it.

### 3.1 M1 companion — mean absolute error, pre-declared

The frozen rule ranks on **mean relative error**. On this corpus that denominator is a small
integer: Letter's graphs average 4.7 nodes, so most exact GEDs are 1–4, and 8.0 % of the census is
exact = 0 and dropped outright. A relative-error mean over such a corpus is dominated by the
smallest-GED pairs and is not comparable across datasets with different GED scales.

**Frozen**: the primary ranking is the spec's mean relative error. **Mean absolute error (M2) is
computed and ranked in parallel, and the report states for every dataset whether the two rankings
agree.** A disagreement is reported as a finding about the metric, not used to override the rule.

### 3.2 The "≥ 4 of 5 datasets" branch — frozen rule kept, companion added

Letter LOW/MED/HIGH are one corpus at three distortion levels (§1 row 6), so the five-dataset vote
is really 3 + 1 + 1 and a Letter-favouring method starts with three votes of five.

**Frozen**: the spec's rule is evaluated **as written on the five datasets**, and the ticket reports
which branch it took and on what margin (spec acceptance criterion 3). **Additionally**, a
corpus-collapsed companion is reported: three units (Letter, LINUX, AIDS), Letter's vote taken as the
majority of its three levels, global primary iff a method wins **all three**. If the two views
disagree, that disagreement is escalated to the human before anything is written into the paper.

### 3.3 M3 certification rate — reported twice

Every valid lower bound returns 0 on an exact-GED-0 pair, so those 306,768 pairs certify for free and
inflate M3 by up to 15.5 % on Letter LOW. **Frozen**: M3 is reported **both** over all certified
pairs **and** over `exact > 0` only, with the second as the headline. Upper bounds get the same
treatment for symmetry of presentation.

### 3.4 M7 cost gate — measured on a separate probe

The frozen gate is "< 1 ms/pair at n̄ = 30" and no such pair exists in Suite 1 (§1 row 7).

**Frozen**: (a) µs/pair is reported per dataset at that dataset's own n̄, from a **serial** pass on a
seeded 2,000-pair sample (seed 42) with `time.process_time()` around `run_method` only — never from
the 24-way parallel pass, whose timings are contended and meaningless; (b) the gate itself is
evaluated on a **separate probe**: 2,000 pairs drawn seed 42 from IAM GREC and Protein graphs with
`25 ≤ n ≤ 35`, loaded through the tracked `iam_gxl_loader.py`. A method fails the gate only on
probe evidence. If the probe cannot be built, the gate is reported **unevaluated**, not passed.

### 3.5 Censored pairs give a one-sided validity test — used, not discarded

For a censored pair the exact solver still returns an interval `[lb_s, ub_s]` containing the true
GED. A candidate lower bound is **refuted** iff `LB > ub_s`; a candidate upper bound is **refuted**
iff `UB < lb_s`. **Frozen**: M4 is evaluated on **all 3,897,911 pairs** — two-sided on the certified
3,836,827, one-sided on the censored 61,084 — while M1/M2/M3/M5/M6 use certified pairs only. This
buys 61,084 extra validity checks at no extra compute, and the AIDS fifth is exactly where a bound
is most likely to break.

### 3.6 Symmetry (M8) — both orientations, always

Every GEDLIB upper bound builds its edit path from a directed assignment (gedlib.md §7). **Frozen**:
UB methods are evaluated in **both** orientations and M8 reports the disagreement fraction and the
mean gain from `min`. The reported UB value for M1/M2/M3/M5/M6 is the **`min` of the two
orientations**, which is the value a production distance matrix would carry. LB methods are run in
one orientation, and symmetry is spot-checked on a seeded 10,000-pair sample per dataset; if an LB
is asymmetric anywhere, that is a defect and halts the ticket like an M4 violation.

### 3.7 Bootstrap — D2 verbatim

Graph-level cluster bootstrap, **2,000 replicates**, percentile CI, **seed 42**, all induced pairs
(statistics.md §5 tier table; no Suite-1 subsampling). Method comparisons within a dataset use the
**same** resample for both methods (D7). Effect sizes lead, p-values support (D10).

### 3.8 Significance — selection procedure, stated as such

Wilcoxon signed-rank on paired per-pair error, **Holm-corrected within each end** (6 comparisons for
4 methods), matched-pairs rank-biserial correlation beside every p-value. Friedman + CD diagram over
the five datasets with the `N = 5` caveat **and** the §3.2 non-independence caveat in the caption.
Reported in the words "this is a selection procedure, not a hypothesis test, and it is outside the
confirmatory family" (preregistration §6 already excludes it).

### 3.9 HED — 2-hour box, then stop

Attempt `HED` with explicit method options; if it does not yield a finite bound within 2 hours of
effort, record the options tried and report it as a finding (spec §3). It is the *Pattern
Recognition*-venue citation and its status serves EiC.b either way.

### 3.10 Cross-check against our own implementation

`benchmarks/real_data/eval_setup/ged_bounds.py` (BRANCH LB + Riesen–Bunke UB + exact A*, 35 tests)
is run on a seeded 400-pair LINUX sample and must agree with GEDLIB's `BRANCH`/`BIPARTITE` on the
same pairs. Disagreement is a bug in one of them and halts the ticket.

### 3.11 Determinism — pinned options, variance reported (user decision, 2026-08-13)

**Live evidence forcing this.** On the local build, `IPFP` returns **UB = 3.00** on P₄ vs C₄ where
the true GED is **1.00** and gedlib.md §5 recorded **1.00** on Picasso. IPFP, REFINE and BP_BEAM are
local-search methods with randomised multi-start, so the reported value depends on initialisation.
IPFP is the *signed primary upper bound* (decision 11), and on the simplest possible input it
overestimates by 200 % and disagrees with the record.

Reproducibility is **not** a criterion in the frozen §5 rule. Adding it after seeing tightness
results would be outcome-dependent, so the treatment is fixed here, before the run:

1. **Variance probe, first.** Every one of the 8 methods is run **5 independent repetitions** at
   GEDLIB defaults on a seeded 5,000-pair sample per dataset (seed 42). Reported per method: the
   fraction of pairs whose value varies across repetitions, and the maximum spread in edit
   operations. Lower bounds are expected to be deterministic; variation in an LB is a defect and
   halts the ticket.
2. **Then pin.** Method options are pinned explicitly — single-threaded, fixed pseudo-random seed,
   and the initialisation count discovered from GEDLIB and **reported**, not left implicit. The
   probe is repeated under the pinned options and must show **zero** variation. A method that still
   varies is reported as irreducibly non-deterministic, and that status is printed beside every one
   of its numbers.
3. **Method identity in the paper is `method + options string`.** The literature table carries the
   exact options column, so "IPFP" in the manuscript is never ambiguous.
4. **Selection still ranks purely on the frozen §5 criteria.** Determinism is *reported*, not used
   to disqualify — that would be the outcome-dependent move this section exists to prevent. If a
   non-deterministic method wins, that becomes a stop-and-ask (§6).

### 3.12 Scope boundary — T-27 stays at `n ≤ 12` (user decision, 2026-08-13)

A large-`n` dominance arm was considered and **declined**. Between two valid lower bounds the larger
is strictly tighter and between two valid upper bounds the smaller is strictly tighter, so the
methods *could* have been ranked above `n = 12` without any ground truth, out to `n = 98`.

**Decision: they are not.** T-27's selection rests on the 3,530,059 exact-anchored M1-eligible pairs
at `n ≤ 12`, and nothing else.

**The obligation this creates.** T-27 narrows the generalisation gap from *400 LINUX pairs at
n̄ = 8.71* to *3.5 M pairs across five datasets at n̄ = 4.7–10.3*. It does not close it: the licensed
regime still runs to `n = 98`. **The report states this as a limitation in its own words**, and the
bracket-width `(UB − LB)/UB` versus `n` measurement — approx_ged §3.1 item 3, "the single
measurement that answers AE.1 most directly" — remains **T-05's**, not T-27's. `review-close`
carries that forward so T-05 reads it.

---

## 4. Decomposition — three tracks, disjoint ownership

Contracts in `.claude/notes/2026-08-13-t27-bakeoff/CONTRACTS.md`, written by the orchestrator and
committed before the wave starts.

| Track | Deliverable | Owns (may create/edit) |
|---|---|---|
| **A — harness** | GEDLIB grid runner producing one `.npz` per (dataset × method) to the contract | `benchmarks/real_data/eval_setup/ged_bound_bakeoff.py`, `tests/unit/test_ged_bound_bakeoff.py` |
| **B — analysis + figures** | M1–M8 aggregation, bootstrap, Wilcoxon/Friedman, both figures through `isalgraph.viz` | `benchmarks/real_data/eval_setup/ged_bakeoff_analysis.py`, `src/isalgraph/viz/bound_bakeoff_view.py`, `tests/unit/test_ged_bakeoff_analysis.py`, `tests/viz/test_bound_bakeoff_view.py` |
| **C — literature** | Verified citation table for all 9 methods: authors, title, venue, volume:pages, year, DOI, complexity, proof status, claimed tightness | `.claude/notes/review/tasks/T-27-literature.md` |

B develops against a synthetic fixture built from the contract, so it does not wait on A.
**The orchestrator runs the campaign** — no agent executes the grid, and no agent touches Picasso,
`scratchpad/`, the board, or any plan file.

Worktrees are safe here: nothing in T-27 imports `isalgraph.core._native`, reports an IsalGraph
timing, or touches `src/isalgraph/core/`. Track B imports `isalgraph.viz` only, which is engine-independent.

---

## 5. Acceptance criteria

Each is checked by the orchestrator, by re-running the named command in a clean main checkout.

1. **All 40 cells attempted**, plus HED as a 41st; every failure reported with its reason.
   *Check*: `data/cells/` holds 40 `.npz` files; `REPORT.md` §grid has 41 rows.
2. **M4 = 0 across every cell**, two-sided on certified pairs and one-sided on censored ones (§3.5).
   *Check*: `data/validity.json` reports `violations == 0` for 41 cells over 3,897,911 pairs.
   **Any violation halts the ticket.**
3. **The spec §5 rule applied verbatim**, with the branch taken and its margin stated, plus the §3.1
   absolute-error and §3.2 corpus-collapsed companions and whether they agree.
4. **Both figures render through `isalgraph.viz`** from a tracked script; `python -m pytest
   tests/viz/test_bound_bakeoff_view.py -q` passes and no script imports `matplotlib` directly.
5. **Unit tests** for harness and analysis in `tests/unit/`, following `tests/unit/test_ged_bounds.py`.
   *Check*: full suite ≥ 726 passed with the engine (the CLAUDE.md reference state), plus the new tests.
6. **Citations verified**, DOI or venue page checked, not recalled.
7. **Cross-check passes** (§3.10): GEDLIB `BRANCH`/`BIPARTITE` agree with `ged_bounds.py` on 400 seeded
   LINUX pairs.
8. **Report written** to `results/reports/T-27-ged-bound-bakeoff/` with `REPORT.md`, `figures/`,
   `data/`, all plots embedded and all tables reproducible from `data/`.
9. **`approx_ged.md` §2's production-assignment table updated in place** with the selected methods and
   the measurement that selected them (handed to `review-close`).
10. **Nothing in `scratchpad/`.**

## 6. Stop and ask

- **M4 > 0 anywhere** — a violated proven bound.
- **No method wins ≥ 4 of 5** (per-dataset primary), **or** §3.1/§3.2's companions contradict the
  frozen rule.
- **`BRANCH_FAST` loses** — it is named in decision 11 and touches a signed decision.
- **ρ(Lev, bound) inverts the M1 ranking** — the tightest bound giving the worst agreement with
  ρ(Lev, exact).
- **An LB is asymmetric** (§3.6), or the §3.10 cross-check disagrees.
- **An LB varies across repetitions** (§3.11 step 1), or a method that wins its end is still
  non-deterministic under pinned options (§3.11 step 4).
- Compute projecting past 40 core-hours — take the spec §3 fallback and report.

---

## 7. Live evidence already in hand, before the wave

Recorded here so it is dated ahead of any result.

| Observation | Evidence |
|---|---|
| Local GEDLIB build works, 21 methods, capability matrix reproduces gedlib.md §5 | smoke on P₄/C₄, unit costs |
| **`IPFP` UB = 3.00 where truth = 1.00**, against 1.00 on Picasso | same smoke — drives §3.11 |
| Methods are **10–50 µs/pair** at n = 4 locally, against 0.09–0.89 ms recorded on Picasso | `time.process_time()` around `run_method`; the campaign budget in §2 is conservative |
| `HED` returns LB 0.00 / UB `inf`, reproducing the Picasso finding exactly | §3.9's 2-hour box applies |
