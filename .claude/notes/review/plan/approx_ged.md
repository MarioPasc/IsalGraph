# Approximate GED — Suite 2, the proven bracket

**Owner**: T-05 · **Serves**: AE.1, R3.7a, R3.5b
**Status**: LOCKED (decision 11). **Cost: ≈ 0.57 core-hours for all 21.7 M pairs** (~~1.05 core-h /
40 M~~ — T-01 re-derived the cohort 2026-08-13, [data](data.md) §1.3).

Related: [gedlib](gedlib.md) · [exact_ged](exact_ged.md) (the calibration anchor) ·
[data](data.md) · [statistics](statistics.md) D13

---

## 1. The principle — two regimes, reported separately and never mixed

| Regime | Reference | Role |
|---|---|---|
| **`n ≤ 12`** (Suite 1) | **exact GED** — `ANCHOR_AWARE_GED` | ground truth + **calibration anchor** |
| **`n > 12`** (Suite 2) | **proven bracket** — `BRANCH_FAST` (lower), `IPFP` (upper) | size-scaling evidence |

### Why proof status matters

If we report a *heuristic* estimate, a reviewer can ask how far it is from the truth and we have no
answer. If we report a **proven bracket** `LB ≤ GED ≤ UB`, the true value is contained by
construction and the only open question is the bracket's width, which we measure.

The asymmetry drives the choice:

- **Upper bounds are structurally easy.** Any method returning a *valid edit path* yields a proven
  upper bound — the path's cost is achievable and GED is the minimum over all paths.
- **Lower bounds need a theorem.** Each requires a proof that no edit path can be cheaper. Only the
  published families qualify.

**Every GED number above n = 12 comes from a method with a published proof, not from a heuristic
that merely performs well.**

### The size story, in three separate statements

1. **IsalGraph encoding has no ~12-node ceiling** — 122 µs at n = 20, 3.9 ms at n̄ = 32, no timeout
   to n = 96. The locked cohort carries it to **n = 98**.
2. **Exact GED does** — 36.9 s/pair at n = 12, ×5 per node. No public benchmark supplies exact GED
   beyond this; GraphEdX stops there for the same reason. **A constraint on the field, not on this
   work.**
3. **Above n = 12 the reference is a proven bracket**, calibrated on the regime below and quoted
   alongside every number.

---

## 2. Method assignment

### Proven lower bound

| Role | Method | Reference | Complexity | Status |
|---|---|---|---|---|
| **Primary** | **`BRANCH_FAST`** | Blumenthal & Gamper, *IEEE TKDE* 30(3):503–516, 2018 | `O(n²Δ² + n³)` | **proven LB; pseudo-metric** |
| Tighter | `BRANCH` | same | `O(n²Δ³ + n³)` | proven LB, strictly ≥ BRANCH_FAST |
| Anytime | `BRANCH_TIGHT` | same | iterative | proven LB at every iteration |
| Venue-fit alternative | `HED` | Fischer et al., ***Pattern Recognition*** 48(2):331–343, 2015 | `O(n²)` | proven LB — **accessor broken, see [gedlib](gedlib.md) §5** |
| Legacy reference | `STAR` | Zeng et al., *VLDB* 2009 | `O(n³)` | proven LB |

Three reasons `BRANCH_FAST` is primary:

1. **Tightest family.** The literature ordering is `BED ≥ LED` and `BED ≥ HED` — branch edit distance
   dominates the linear and Hausdorff bounds. Tighter LB → narrower bracket → stronger claim.
2. **Measured, and re-measured 2026-08-12 on a reproducible artifact**: on 400 LINUX pairs under unit
   costs, ρ(exact, LB) = **0.859** with **−26.3 %** bias, against ρ(exact, UB) = **0.522** with
   **+135.2 %** bias. The lower bound is the better proxy on our data — *not* the intuitive
   expectation, which is why we measured it.
   ⚠ **The earlier figures (0.966 / 0.840, −11 % / +78 %) do not reproduce and must not be quoted.**
   All six miss in the flattering direction, consistent with having been measured on IAM Letter and
   printed as a cohort property. **Re-derive per dataset in §3.1's ladder and print each with its
   population.** Full result: [exact_ged](exact_ged.md) §4 gate 2.
3. **It is a pseudo-metric** on a graph collection (same paper). Corollary 2.13 claims the IsalGraph
   distance is a metric; validating it against a reference with metric structure is coherent in a way
   validating against an arbitrary heuristic is not.

### Proven upper bound

| Role | Method | Reference | Status |
|---|---|---|---|
| **Primary** | **`IPFP`** | Bougleux et al., 2017 | proven UB (returns a valid edit path) |
| Refinement | `REFINE` | Zeng et al. 2009 / GEDLIB | proven UB; local search on the assignment |
| **Reference point** | **`BIPARTITE`** | Riesen & Bunke, *IVC* 27(7):950–959, 2009 | proven UB; the baseline every reader knows |
| Alternative | `BP_BEAM` | Neuhaus & Riesen | proven UB |

`BIPARTITE` is reported because it is the comparator every reader knows, **not** because it is good:
our own implementation of it overestimates by **+135 % on LINUX** (measured 2026-08-12, symmetrised;
+165 % in a single orientation) and it is the loosest member of the family. `IPFP` and `REFINE`
handle node and edge assignment simultaneously rather than sequentially, and should do much better —
**but that is now a prediction to test, not a recorded fact**, since the figure it was based on did
not reproduce.

> **Select the primary UB by measured tightness on the calibration set**, criterion fixed in advance:
> the method minimising **mean relative overestimate against exact GED**, subject to costing under
> **1 ms/pair at n̄ = 30**.

> ## ⚠ SCOPED OUT 2026-08-13 (T-02) — both ends are now selected by T-27, not by this file
>
> Two premises above are weaker than they read, and T-03 removed the reason to tolerate either:
>
> | Premise | Evidence behind it |
> |---|---|
> | `IPFP` is the primary **upper** bound | **none.** §2 says so itself: *"that is now a prediction to test, not a recorded fact"*. The only measured UB is our own BP at **+135 %** |
> | `BRANCH_FAST` is the primary **lower** bound | literature dominance (`BED ≥ LED`, `BED ≥ HED`) **plus 400 LINUX pairs at n̄ = 8.71**, licensing a regime that runs to `n = 98` |
>
> The LB case is structurally the same generalisation that produced the six retired figures — measured
> on one dataset, printed as a cohort property — one round after being caught doing it.
>
> **T-03 removed the excuse.** The complete Suite-1 exact GED now exists: **3,897,911 pairs, 98.43 %
> certified exact**, at
> `…/data/source/GED_PRECOMPUTED/extended_merged_exact_ged/`. GEDLIB runs at ~100 µs/pair, so the
> **full** proven-method grid against that ground truth costs **≈ 5 core-hours**.
>
> **New ticket T-27** runs it: LB ∈ {BRANCH, BRANCH_FAST, BRANCH_TIGHT, STAR}, UB ∈ {IPFP, REFINE,
> BIPARTITE, BP_BEAM}, per dataset, selecting each end by the rule stated above (and its
> underestimate mirror for the LB). Spec: `.claude/notes/review/tasks/T-27-spec.md`.
> **T-27 closes before T-05 computes any production matrix.** Until it does, `BRANCH_FAST` and `IPFP`
> are *defaults*, not selections, and no draft may justify either by measurement.

### Production assignment

| Computation | Suite | Method | Accessor |
|---|---|---|---|
| Exact GED — primary reference and calibration anchor | 1 | **`ANCHOR_AWARE_GED`** | both; assert `LB == UB` |
| Exact GED — independent cross-check | 1, sample | `networkx.graph_edit_distance` | — |
| **Proven lower bound** | 1 (calibration) + 2 (all) | **`BRANCH_FAST`** | `get_lower_bound()` |
| LB — tightening, if the bracket is too wide | 2 | `BRANCH_TIGHT` | `get_lower_bound()` |
| **Proven upper bound** | 1 (calibration) + 2 (all) | **`IPFP`** | `get_upper_bound()` |
| UB — literature reference point | 1 + 2 | `BIPARTITE` | `get_upper_bound()` |
| UB — refinement arm, if `IPFP` is loose | 2 | `REFINE` | `get_upper_bound()` |

**Cost**: `BRANCH_FAST` / `IPFP` run at ~100 µs/pair at n̄ = 30, so all **21,710,892** Suite-2 pairs cost
**≈ 0.57 core-hours**. **No pair subsampling is needed anywhere in Suite 2.**
(~~40 M pairs, ≈ 1.05 core-hours~~ — superseded by T-01's cohort re-derivation, decision 27.)

---

## 3. Calibration — the gate on everything above n = 12

Correlating Levenshtein against an *approximate* GED mixes two effects: how well Levenshtein tracks
true GED, and how well the approximation tracks true GED. Bipartite GED's error is known to grow with
graph size, so a declining ρ at large `n` would be **uninterpretable**. Mandatory protocol:

1. **On `n ≤ 12`, where exact GED exists**, report all three: ρ(Lev, GED_exact), ρ(Lev, GED_approx),
   ρ(GED_approx, GED_exact), plus the mean relative overestimate.
2. **State the calibration in the paper.** If ρ(GED_approx, GED_exact) is high and
   ρ(Lev, GED_approx) ≈ ρ(Lev, GED_exact) on the same pairs, the approximation is a validated
   stand-in. If not, we report the exact-GED result and say the extension is not supportable — **a
   legitimate outcome.**
3. **Above `n = 12`**, report ρ(Lev, GED_approx) with the calibration quoted alongside every number.

**Pre-declared decision rule** (fixed now, not after seeing the number):

> If the 95 % CI for `ρ(Lev, exact) − ρ(Lev, approx)` excludes 0 **and** the point estimate exceeds
> **0.05** in absolute value, the approximation is not a validated stand-in. The exact-GED results
> become primary and the large-`n` extension is reported as **descriptive only**.

### 3.1 The calibration must reach the regime it licenses

The gate is computed where exact GED exists (`n ≤ 12`), and the bounds were validated on **`n = 3–9`
only**, while the licensed regime runs to **`n = 98`**. Bracket *validity* is not at risk —
`LB ≤ GED ≤ UB` is proven at every `n`. **Tightness** is, and tightness is what the argument rests on.

Three additions, all cheap:

1. **A size-stratified exact ladder.** Run `ANCHOR_AWARE_GED` on a stratified sample at each `n` from
   3 up to the feasible ceiling, with a fixed per-pair budget and interval censoring (D11) above it.
   **Every node the exact solver buys widens the calibration and directly strengthens AE.1.**
2. **Regress, do not assume transfer.** Fit relative bracket width `(UB − LB)/UB` and the ρ-gap on
   `n` over the ladder; report the extrapolation to the Suite-2 range **with its uncertainty**.
3. **Report `(UB − LB)/UB` as a function of `n` across all of Suite 2.** Needs no exact GED,
   computable on all 21.7 M pairs, and is **the single measurement that answers AE.1 most directly**:
   it separates "IsalGraph degrades at scale" from "our reference degrades at scale".

---

## 4. How the bracket is reported — no interpolation

Above `n = 12` we hold `LB ≤ GED ≤ UB` and do **not** know where in the bracket the true value lies.
**Do not report a midpoint or any other interpolation** — it would be an unjustified assumption
sitting under every downstream number.

Instead, **correlate Levenshtein against the lower and upper bounds separately and report both ρ.**

**D13 — the agreement rule, pre-declared.** Per dataset, bootstrap `ρ(Lev, LB) − ρ(Lev, UB)` on the
**same** graph-level resamples (D7). The bracket is **uninformative** at that dataset if the 95 % CI
excludes 0 **and** the point estimate exceeds **0.05** in absolute value.

> **PROMOTED 2026-08-13 (T-02, decision 25).** D13 is now **family F1 of the confirmatory design** —
> 10 tests, one per Suite-2 dataset, BH-FDR at q = 0.05, prior to and separate from the primary
> family. The claim it registers is *the conclusion is invariant to where inside the proven bracket
> the true value lies*, which is the scientific content of reporting a bracket at all and is stronger
> stated as a pre-registered result than as a footnote. It sits in its own family because its outcome
> **removes rows from** the primary family (8 per uninformative dataset), and a test cannot set the
> cardinality of the family containing it. Full structure:
> [preregistration](../plan/preregistration.md) §3.
>
> This is also why `ρ(Lev, UB)` is **not** given its own per-dataset rows in the primary family: it
> would be a near-duplicate of `ρ(Lev, LB)` on the same pairs. The upper bound keeps every reporting
> obligation in this section and carries its confirmatory weight through F1.

| Example | ρ(Lev, LB) | ρ(Lev, UB) | difference, 95 % CI | verdict |
|---|---:|---:|---|---|
| concordant | 0.61 | 0.58 | 0.03, [−0.01, 0.07] | report ρ ≈ 0.6; conclusion is **robust to the bracket's interior** |
| uninformative | 0.55 | 0.31 | 0.24, [0.19, 0.29] | report "ρ lies between 0.31 and 0.55" **descriptively**; exclude from the confirmatory family |

Without a pre-declared threshold, "agree" gets decided after seeing the numbers — on the rule
governing **every result above n = 12**.

**Also report**, per size and density stratum:

- **bracket width** `UB − LB`, absolute and relative;
- **certification rate** — the fraction with `LB = UB`, where GED is exact for free. ⚠ **Measured
  1.5 % on LINUX** with our BP, not the 9.8–11.3 % previously recorded ([exact_ged](exact_ged.md)
  §4). `IPFP` should do materially better, but **do not promise a rate in the text before T-05
  measures it per dataset** — the difference between 1.5 % and 10 % is the difference between "exact
  for free on a tenth of pairs" and "essentially never".
- **symmetry**: every upper-bound method GEDLIB offers builds its edit path from a *directed*
  assignment and is **not symmetric**. Fill both triangles and take the `min`, or assert symmetry and
  fail loudly. Measured on our own implementation: tighter on 33.2 % of pairs, mean gain 1.15 edit
  operations. The lower bound is symmetric and needs no treatment.

A disagreement between the two ends is itself an informative and publishable outcome.

---

## 5. References to cite

Blumenthal & Gamper, *IEEE TKDE* 30(3):503–516, 2018 (BRANCH / BRANCH-FAST — our LB) ·
Bougleux et al., 2017 (IPFP — our UB) ·
Riesen & Bunke, *Image and Vision Computing* 27(7):950–959, 2009 (BIPARTITE — the reference point) ·
Fischer et al., ***Pattern Recognition*** 48(2):331–343, 2015 (HED — venue fit for EiC.b) ·
Zeng et al., *VLDB* 2009 (STAR — **already in our bibliography**, and commenting on it individually
fixes the `methodology.tex:803` citation group) ·
Blumenthal et al., GbRPR 2019 (GEDLIB itself) ·
Jain et al., NeurIPS 2024 (GraphEdX, already cited).

Slot accounting in [compliance](compliance.md).
