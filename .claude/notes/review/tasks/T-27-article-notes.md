# T-27 — article notes

**Closed 2026-08-13** · Serves **AE.1, R3.5b, R3.7a** · Closes decision 26, upholds decision 11
**Source of every number**: `results/reports/T-27-ged-bound-bakeoff/` — `REPORT.md`, `data/`, `figures/`

Ordered by consequence. Items 1–4 change what the paper may claim; 5–8 are reporting obligations.

---

## 1. The lower bound now rests on a theorem, not on a sample — **T-20, §3.3**

> The survey (Blumenthal, Boria, Gamper, Bougleux & Brun, *VLDB Journal*, `10.1007/s00778-019-00544-1`)
> §5.2.4 proves **BRANCH and BRANCH-FAST are equivalent for constant edge edit costs**. Our cost
> model has constant edge edit costs, so the general `BRANCH ≥ BRANCH-FAST` collapses to equality
> here. **Measured identical on all 3,836,827 certified pairs across five datasets, maximum absolute
> difference 0.0.**

**Why this matters more than a tightness number.** R3's objection to the previous round was that
conclusions measured on small samples were licensed to a much wider regime. For this choice the
objection dissolves: the equivalence is *proven for all n* under our cost model, so the selection of
`BRANCH_FAST` over `BRANCH` is a cost decision, not an empirical extrapolation. Replaces the old
justification, which was 400 LINUX pairs at n̄ = 8.71.

**Provenance**: `data/analysis/validity.json` → `proven_orderings`; harness gate, all 5 datasets.

## 2. GEDLIB's upper bounds are not reproducible at library defaults — **T-20 §3.3, T-21**

| Configuration | Pairs whose value changes between runs |
|---|---|
| GEDLIB defaults | **91.5 – 93.6 %**, spreads up to **10 edit operations** |
| Pinned `--threads 1 --randomness PSEUDO --initial-solutions 10` | **0.0000 — every cell, every dataset** |

`LSBasedMethod`, the base class of IPFP, REFINE and BP-BEAM, defaults `--initialization-method` to
`RANDOM` and `--randomness` to `REAL`. This is why the same method returned an upper bound of 3.00
on P₄ vs C₄ (true GED 1.00) on one machine and 1.00 on another.

**Consequence for the manuscript**: **a method name is not a specification.** Every GED number from
a local-search upper bound must carry its options string. This is a reproducibility point in our
favour if we state it, and a defect a reviewer can find if we do not.

**Provenance**: `data/determinism/*.json`, 60 probes, 5 repetitions on seeded 5,000-pair samples.

## 3. The upper bound is selected under a cost constraint, and that has two measured costs — **T-20, T-14**

`BIPARTITE` is the **loosest of seven** upper bounds measured (mean relative error **1.095** against
`IPFP_MS` 0.084, a factor of 13). It is primary **only** because the pre-declared cost gate
(< 1 ms/pair at n̄ = 30) excludes every tighter method. **This must be stated plainly; it is not a
finding that IPFP is bad.**

| | LINUX | AIDS | L-LOW | L-MED | L-HIGH |
|---|---:|---:|---:|---:|---:|
| ρ(Lev, UB) − ρ(Lev, exact), `BIPARTITE` | −0.078 | −0.048 | **−0.219** | **−0.177** | −0.013 |
| same, `BP_BEAM_DET` | −0.030 | +0.037 | −0.004 | +0.001 | +0.005 |

**D13 fires on 2 of 5** under `BIPARTITE` (CI excludes 0 and |gap| > 0.05) and on **0 of 5** under
`BP_BEAM_DET`, which misses the cost gate by 17 %. `BRANCH_FAST` fires on none.

**Second cost — the error compounds with size.** OLS slope of mean relative error on `max(n₁,n₂)`,
per node, on AIDS (n = 4→12): `BIPARTITE` **+0.294** (0.00 → **2.19**), against `IPFP_MS` +0.029,
`BP_BEAM_MS` +0.055, `BRANCH_FAST` +0.036. **The selected upper bound is the one whose error grows
fastest in exactly the direction AE.1 concerns.**

**PI decision 2026-08-13**: frozen gate stands as primary; tighter methods reported as a disclosed
sensitivity arm with their costs. **Disclose the gate and the sensitivity arm together** — reporting
the selection without its cost is the version a reviewer will catch.

**Provenance**: `data/analysis/bootstrap.json` (`d_rho_lev`), `metrics.json` (`error_vs_n`).

## 4. Certification is measured, and the reported bracket certifies at the low end — **T-20 §3.3**

Fraction of certified pairs where the bound equals exact GED:

| | LINUX | AIDS | L-LOW | L-MED | L-HIGH |
|---|---:|---:|---:|---:|---:|
| `IPFP_MS` | 0.894 | 0.729 | 0.973 | 0.967 | 0.939 |
| `BRANCH_FAST` | 0.358 | 0.202 | 0.858 | 0.851 | 0.661 |
| **`BIPARTITE`** | **0.066** | **0.012** | 0.383 | 0.387 | 0.402 |

This **discharges `approx_ged.md` §4's standing instruction** not to promise a certification rate
before it was measured per dataset. The earlier LINUX figure of 1.5 % (our own BP) and the retired
9.8–11.3 % are both superseded.

## 5. Every proven relation in the literature holds on our data — **T-14, R3.5b**

| Prediction | Kind | Verdict |
|---|---|---|
| `BRANCH` = `BRANCH_FAST` under constant edge costs | proven | **CONFIRMED**, exact equality, 3.8 M pairs |
| `BRANCH ≥ HED` | proven | **CONFIRMED**, 0 violations |
| every UB ≥ exact; every LB ≤ exact | proven | **CONFIRMED**, 0 violations in 46,774,932 evaluations |
| `REFINE`/`BP_BEAM` ≤ `BIPARTITE` under a BIPARTITE initialiser | proven | **CONFIRMED**, 0 violations |
| `IPFP` is the tightest upper bound | empirical | **CONFIRMED** |
| `BIPARTITE` is the loosest | empirical | **CONFIRMED**, by 6.7× |
| `BRANCH_TIGHT` tighter than `BRANCH` | empirical | **CONFIRMED**, 0.103 vs 0.166 |

**No published claim is contradicted.** Worth one sentence in the response: our measurements agree
with the reference implementation's authors at a scale their own evaluation did not reach.

## 6. HED — the *Pattern Recognition* citation now carries a number — **T-20 related work, EiC.b**

Fischer, Suen, Frinken, Riesen & Bunke, ***Pattern Recognition*** 48(2):331–343, 2015. Previously
recorded in the plan as unusable. It is **lower-bound-only by design** and its default is *vacuous*
under our cost model because edge substitution is free; under `--edge-set-distances OPTIMAL` it is a
valid bound on every pair. Loosest in the grid (0.899), **confirming the published `BED ≥ HED`
dominance** over 3.8 M pairs.

Also: `IPFP` is **Pattern Recognition Letters** 87:38–46, 2017 — a second *PR*-family citation, so
EiC.b no longer rests on HED alone.

## 7. Bibliography corrections — **T-20, T-21**

Four defects in `approx_ged.md` §5, all verified against DOI: IPFP had no venue/volume/pages;
BP_BEAM was attributed to "Neuhaus & Riesen" but is **Riesen, Fischer & Bunke**, ANNPR 2014, LNAI
8774:117–128; "Zeng et al., *VLDB* 2009" is **PVLDB 2(1):25–36**; REFINE is Zeng's, not GEDLIB's.
**The survey was omitted entirely** though every complexity figure and both dominance claims come
from it. Full table with per-field provenance: `T-27-literature.md`.

## 8. Reproduction parameters — **T-21**

Cost model D6 `[1,1,0,1,1,0]`. Exact anchors from **`networkx.graph_edit_distance`** (T-03), 60 s
per-pair budget, `certified_mask` marks completed searches. Bounds from **GEDLIB** via
`jajupmochi/graphkit-learn`, in-place build. Comparisons at **tolerance 1e-9**. Bootstrap:
graph-level cluster resampling, **2,000 replicates, percentile CI, seed 42**, all induced pairs.
Levenshtein from the **exhaustive** (canonical) encoder variant. Hardware: 24-core Debian 12
workstation; timing measured **single-process** with `time.process_time()`. Campaign ≈ 7 core-hours.

---

## What is NOT claimable from this ticket

1. **That the selection transfers to `n = 98`.** It was made at **`n ≤ 12`**, the ceiling where exact
   GED exists. The gap is **narrowed** — 400 LINUX pairs at n̄ = 8.71 → 3.5 M pairs across five
   datasets at n̄ = 4.7–10.3 — **not closed**. Item 3's slope is a reason to expect it *not* to
   transfer for `BIPARTITE`.
2. **That `BIPARTITE` is a good upper bound.** It is the loosest measured. Do not let the sentence
   "selected by measurement" imply "selected for tightness" — it was selected under a cost
   constraint, and §5 of the report says which.
3. **AIDS tightness figures as unconditional.** **20.67 %** of AIDS pairs are interval-censored and
   the censored pairs are the ones the exact solver could not finish in 60 s — systematically the
   harder ones. Every AIDS number here is conditioned on the solved 79.3 % and is optimistic by an
   unknown amount. **This dependence must travel in the same sentence as the number.**
4. **The cost figures as machine-independent.** µs/pair are 24-core Debian workstation timings at
   n̄ = 29.5 on IAM GREC and Protein. The Picasso figures in `gedlib.md` §5 are ~20× slower for the
   same work. Suite-2 projections use **21,710,892** pairs (T-01's re-derived cohort, decision 27)
   and are **lower bounds** on true cost, since Suite 2 reaches `n = 98` where every method is slower.
5. **Any p-value from this ticket as a hypothesis test.** It is a **selection procedure, explicitly
   outside the pre-registered confirmatory family** (preregistration §6). Wilcoxon over 2,030,043
   dyadically dependent pairs returns p ≈ 0 for differences of no practical size. Effect sizes and
   graph-level bootstrap CIs carry the inference (D10).
6. **A critical-difference diagram as evidence of separation.** `N = 5`, and the five are not
   independent — Letter LOW/MED/HIGH are one 15-class corpus at three distortion levels, so the vote
   is really 3 + 1 + 1. Reported descriptively, with the corpus-collapsed companion beside it.
7. **A certification *rate* without its dataset.** It ranges from 1.2 % to 40.2 % for `BIPARTITE`
   across five datasets — a single cohort-wide figure would repeat exactly the generalisation error
   that retired six figures in T-25.
8. **`STAR` as a proven bound under labels.** Its validity needs **uniform** edit costs (Zeng et al.,
   Lemma 4.2), satisfied here only because our graphs are effectively unlabeled. If
   [labels](../plan/labels.md) Tier 2 is ever promoted, this must be re-derived.
