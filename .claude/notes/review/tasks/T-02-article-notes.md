# T-02 — article notes

**For**: T-20 (manuscript), T-14 (letter), T-06 (execution), T-21 (reproducibility)
**Ordered by consequence.** Source: [preregistration](../plan/preregistration.md),
[statistics](../plan/statistics.md) §5 and §9, `T-02-design.md`.

---

## A. Changes what the paper may claim

### A1 — The multiplicity family is pre-registered, and the paper must say so
**Owner**: T-20 → §3.2 (statistical protocol) · **Lands**: new paragraph, ~90 words

The paper states the confirmatory family, **its cardinality**, and that it was frozen **before any
p-value was computed** (2026-08-13). Three fixed-sequence families, BH-FDR at q = 0.05 within each:
`F0` calibration gate (5), `F1` bracket gate (10), `F2` primary (182) — **`N_max = 197`**. Cite
Benjamini & Hochberg, *JRSS-B* 57(1):289–300, 1995 and Dmitrienko, Tamhane & Bretz, *Multiple Testing
Problems in Pharmaceutical Statistics*, CRC Press, 2009, ch. 5 (gatekeeping).

This is the difference between answering R3.5c and appearing to. A stated cardinality is checkable;
"we corrected for multiple comparisons" is not.

### A2 — The exact regime gets no omnibus, and the reason is printed
**Owner**: T-20 → §4 · **Lands**: one sentence in the results preamble

Friedman at `N = 5` datasets yields a critical difference wide enough to separate almost nothing. The
omnibus and CD diagram are reported **for the ten-dataset approximate regime only**; the exact regime
is descriptive — per-dataset ρ with graph-level bootstrap CIs and D7 paired differences. **Stating
the reason converts a missing figure into a methodological choice.** Demšar, *JMLR* 7:1–30, 2006.

### A3 — The upper bound's role is invariance, not a second correlation
**Owner**: T-20 → §3.2 and §4.3 · **Lands**: one paragraph

Both `ρ(Lev, LB)` and `ρ(Lev, UB)` are reported per dataset with no interpolation, but the
**registered** claim is their bootstrap difference: *the conclusion is invariant to where inside the
proven bracket the true value lies*. Reason to print: the two correlations are computed on the same
pairs from two bounds on the same quantity, so treating them as independent evidence would overstate
what the bracket shows. This framing is stronger than reporting both ρ separately, and it is the one
a reviewer can check.

---

## B. Reporting obligations

### B1 — D15's resampling effort, per dataset
**Owner**: T-20 → every table caption in §4 · **Lands**: caption text + one supplementary table

Every table states its **replicate count, permutation count and subsample size**. The frozen tier
assignment ([statistics](../plan/statistics.md) §5) is a supplementary table. Tier 3 applies to
**COIL-DEL** (1,000 replicates / 1,999 permutations / 2 × 10⁶ induced pairs = **7.72 %**, seed 42) and
**Mutagenicity** (same, **24.51 %**). Everything else runs all pairs.

⚠ **These percentages are a property of the pair counts in [data](../plan/data.md) §1, whose Suite-2
half is unverified pending T-01.** If T-01's re-derivation moves a pair count across a tier boundary,
the tier moves and this note changes.

### B2 — The subsample is validated at the production ratio
**Owner**: T-20 → supplementary · **Lands**: one table, ~60 words

Validating a 7.72 % subsample by drawing 94.4 % of a smaller dataset would demonstrate nothing. The
comparison runs at the **matched fractions** — 163,564 (7.72 %) and 519,296 (24.51 %) pairs of IAM
Letter HIGH against its all-pairs protocol — **plus a structure-matched arm on Mutagenicity itself**,
because Letter HIGH's `n̄ = 4.58` against COIL-DEL's 21.5 makes a ratio match on Letter HIGH a test of
the estimator rather than of the application. Pre-declared revision threshold: **10 % relative
difference in CI half-width**.

### B3 — Reduction of the family is disclosed, not silent
**Owner**: T-20 → §3.2 · **Lands**: two sentences + one supplementary column

`N_actual = 182 − 15k − 8d` is printed alongside `N_max`, with the exclusion list, and every
confirmatory table carries a **BH-over-`N_max` sensitivity column**. State that
[competitors](../plan/competitors.md) §3.4's exclusion rule is **blind to correlation with GED** —
that is what makes the reduction legitimate rather than a smaller denominator chosen after the fact.

---

## C. Reproduction parameters

| Parameter | Value |
|---|---|
| Bootstrap | graph-level cluster, percentile CI, **seed 42** |
| Replicates | 2,000 (tiers 1–2) / 1,000 (tier 3) |
| Mantel permutations | 9,999 / 4,999 / 1,999 by tier; joint row-and-column permutation of graph labels |
| Within-replicate subsample | tier 3 only: 2 × 10⁶ induced pairs, seed 42 |
| FDR | Benjamini–Hochberg, **q = 0.05**, within each of F0/F1/F2 |
| Post-hoc | Wilcoxon signed-rank with **Holm**, nested under the omnibus, **not** counted in BH |
| Freeze date | **2026-08-13**, before any p-value existed |
| Family cardinality | `N_max = 197` (5 + 10 + 182) |
| Compute | ≈ 40–80 core-hours, ~1 h on 64 cores |

---

## D. What is NOT claimable

1. **Do not write that both bracket ends were tested confirmatorily.** They were not.
   `ρ(Lev, UB)` is reported everywhere and registered only through F1's difference test.
2. **Do not quote ρ(exact, LB) = 0.859 or ρ(exact, UB) = 0.522 as cohort properties.** They are
   **400 LINUX pairs at n̄ = 8.71**. The population travels with the number in the same sentence, and
   **T-27 supersedes both.**
3. **Do not justify `IPFP` or `BRANCH_FAST` by measurement** until T-27 closes. `IPFP` has never been
   measured against exact GED; the plan says so in its own words. Until then they are *defaults*.
4. **Do not print any Suite-2 pair count, graph count, `n_max`, density or discard ratio** until
   T-01's `cohort_audit.py` re-derives it. Precedent: T-25 re-derived six figures from a lost script
   and none reproduced, all six in the flattering direction.
5. **Do not present the labels analysis as part of the confirmatory family.** S-d is open; if Tier 2
   is promoted it becomes its own pre-declared family with its own q = 0.05.
6. **`N_max = 197` is a ceiling, not the number that will be printed.** `N_actual` is what BH runs
   over; both appear, and the printed FDR threshold follows `N_actual`.
7. **The tier boundaries are not a general recommendation.** They are a budget decision for this
   cohort at these pair counts, and the paper says so.
