# T-06 — response-letter fragment

Written 2026-08-24 at close. Owner: **T-14** (letter), with manuscript edits owned by **T-20**.

**This fragment reports a largely negative recompute.** The register that follows is deliberate: a
reviewer who asked whether the correlation is real is owed a direct answer, and the answer is mostly
*no*. Attempting to soften it is the one thing that would make the revision fail, because every number
below is recomputable from the archived artifacts in under an hour.

---

## Fragment A — the recompute itself (AE.1, R3.5b: "recompute under one cost model")

> We have recomputed every reported quantity under a single edit-cost model (node and edge
> insertion/deletion = 1, substitutions free), on frozen cohorts of **16,370** Suite-2 graphs
> (**21,710,892** pairs) and **5,350** Suite-1 graphs (**3,897,911** pairs). All exact graph edit
> distances below 12 nodes are recomputed with A\*; above that we report the proven BRANCH-FAST /
> IPFP bracket as two series, never interpolated.
>
> The analysis follows a pre-registration frozen before any distance matrix existed: three families in
> fixed sequence (F0 calibration, 5 tests; F1 bracket, 10; F2 primary, 182), Benjamini–Hochberg at
> q = 0.05 within each, and the family cardinality defined by enumeration with a closed form printed
> as a check. **`N_actual = 79`; the enumeration and the closed form agreed exactly, and all 79 cells
> carry a p-value.**

## Fragment B — the calibration gate fired, and the large-`n` extension is now descriptive (AE.1)

> The pre-registered calibration gate (F0) compares ρ(Levenshtein, exact GED) against
> ρ(Levenshtein, approximate GED) on the five Suite-1 datasets. **It fires on four of five.** Under
> the pre-declared majority branch, the exact-GED results become primary and the 81 approximate-regime
> cells are reported **descriptively only**.
>
> The bracket gate (F1) independently returns **`d = 7 of 10`**: on seven Suite-2 datasets the
> conclusion is not invariant to where inside the proven bracket the truth lies.
>
> **We report this as the outcome it is.** `approx_ged.md` §3 anticipated it as a legitimate result,
> and it is the honest reading of a bracket that is too wide at these sizes to conclude from.

## Fragment C — 🔴 the central negative (R3.5, the size-confound question)

> **We report directly that the reported correlation is substantially a size effect.**
>
> Against the trivial baseline `|n_i − n_j|` — count the nodes and subtract, with no representation
> at all — the canonical string is **significantly worse on 17 of 25 records (68 %)**, including
> **four of five Suite-1 datasets measured against exact graph edit distance**, where no
> bracket argument applies. One record is undetermined; seven favour the string.
>
> The mechanism is visible when the size channel is removed by construction. Restricting to pairs of
> equal node count — where `|n_i − n_j|` is identically zero and the baseline is undefined — the
> correlation falls from **ρ = 1.00 at n = 3 to 0.26 at n = 12**, and averages **0.135** over
> n = 13–30. Above n ≈ 40 **no representation in our comparison, ours or any competitor, is reliably
> distinguishable from ρ = 0.**
>
> We therefore withdraw the claim that the encoding's edit distance approximates graph edit distance
> at the sizes Suite 2 covers.

**Attach immediately, in the same paragraph — this is the contribution the negative buys:**

> This is in part a property of the benchmarks. **Node-count difference alone attains ρ = 0.71–0.997
> against ground-truth GED on these datasets, exceeding 0.96 on seven of ten**, reaching 0.9971 on
> COIL-DEL. Correlation with GED on this data measures size agreement more than structural fidelity —
> for every representation, ours included. We report the within-`n` decomposition because it is the
> only view in which the two can be separated, and we recommend it for any graph-distance surrogate.

## Fragment D — the limitation, stated as a condition rather than a threshold (R3)

> On IAM Letter — the same generator at three distortion levels, so source, labelling and construction
> are fixed and only the graphs differ — mean node count rises 4.07 → 4.58 while mean edge count rises
> 3.07 → 4.56. **The trivial size baseline stays flat at ρ ≈ 0.92 while the canonical string falls
> from 0.93 to 0.67**, and in the regression the structural coefficient's share falls from dominant
> (0.63×) through parity (1.10×) to minority (2.78×), crossing at the middle distortion level.
>
> **The representation tracks edit distance where there is little structure to track.**

## Fragment E — what survives, scoped (AE.1, R2)

> Three results survive the recompute.
>
> **1. Completeness at scale.** The instruction string is a complete graph invariant, and we verify
> it on **24,764,422 pairs with zero collisions** — 3,424,764 of them certified non-isomorphic by
> *exact* GED. On Suite 2 certification is `LB > 0`, which certifies non-isomorphism but not its
> converse, so pairs the bound could not separate lie outside that half of the test.
>
> **2. Compactness among canonical codes, above n ≈ 20.** Against gSpan minimum-DFS — itself a
> canonical code, and the comparator we consider most important — the instruction string is shorter on
> **112 of 112 size strata above n = 20, median +214.8 bits, with no losses and no ties.** The
> advantage **grows** with graph size, from 20.4 % of strata at n ≤ 5 to 45.6 % above n = 41. **Pooled
> across all sizes the comparison is not favourable**, and we state the size scope wherever the claim
> appears. Edge-list serialisations (sparse6, nauty-sparse6) beat it at scale, which is expected on
> sparse cohorts: they encode an edge list and exploit a property of the data that an instruction
> string does not.
>
> **3. Predictable cost.** Encoding cost is governed by the automorphism group, not by graph size:
> at a 300 s budget, censoring is **0 % for all 3,703 Mutagenicity graphs with |Aut| ≤ 10⁴, 21.9 % at
> 10⁴–10⁸, and 100 % above 10⁸.** A user can compute |Aut| in milliseconds and know in advance whether
> the method applies.
>
> **No single representation leads on both axes.** The most compact serialisation admits no metric
> satisfying the distance axioms; the best-correlating representation admits no bit count. Among those
> measurable on both, ours is more compact than min-DFS and nauty-graph6 and is **dominated on both
> axes by nauty-sparse6.**

## Fragment F — execution integrity (R3.5c, T-21)

> The analysis was pre-registered before any distance matrix existed, and we report it unchanged,
> including its negative results. During execution we found and corrected eight defects in our own
> analysis code. **Every one fell in the descriptive layer; the registered family — its cardinality,
> its 79 cells and its Benjamini–Hochberg columns — did not move.**
>
> We note this was in part a consequence of a conservative reading taken at a point the
> pre-registration left undefined: had we read it the other way, three cells affected by a
> resampling defect would have been inside the confirmatory family. **The registered family was
> insulated, not unreachable**, and we record the three undefined terms we encountered with the
> resolutions taken and their dates.
>
> All resampling is at the **graph** level, never the pair level. An unplanned check confirms
> determinism: 14 model fits computed twice by independent processes hours apart at the same seed are
> byte-identical, and three artifact types re-derived by an independent path agree to 1.1 × 10⁻¹⁶.

---

## Provenance table — one row per claim

| claim | number | artifact |
|---|---|---|
| cohort sizes | 16,370 / 21,710,892 / 5,350 / 3,897,911 | `encodings/manifest.json`, `ladder.json` |
| `N_actual`, discrepancy | 79 / 0 | `families/family_F2.json` |
| BH rejections | 75 of 79 (35 for / 34 against) | `families/family_F2.json` |
| F0 fires | 4 of 5 | `families/family_F0.json` |
| F1 `d` | 7 of 10 | `families/family_F1.json` |
| below size null | 17 of 25, all significant | `families/rho_table.json` |
| Suite-1 exact excess | −0.4597 … +0.0139 | `families/rho_table.json` |
| within-`n` collapse | 1.0000 → 0.2608 → 0.135 | `size_profile.json` |
| benchmark size-domination | ρ 0.71–0.997 | `families/rho_table.json` |
| Letter control | 0.93 → 0.67 vs flat 0.92 | `size_profile.json`, `families/rho_table.json` |
| zero collisions | 24,764,422 pairs | `ladder.json`, `ladder_suite1.json` |
| min-DFS compactness | 112/112, +214.8 bits | `claim_a_strata.json` |
| Claim A by size band | 20.4 % → 45.6 % | `claim_a_strata.json` |
| censoring by \|Aut\| | 0 % / 21.9 % / 100 % | `censoring.json` |
| dominance / no leader | 8 bound-pairs | `t06_dominance.py`, `collinearity.json` |
| determinism | 14 fits byte-identical | `families/f2_partials*/` |

All under `results/reports/T-06-full-recompute/`. Engine `cpp`, build `298fc1188bf1b051`, seed 42,
300 s encode budget.

---

## ⚠ Do not lift into the letter

`mutagenicity`'s β_lev = +0.5229 (**retracted** — tier-3 subsampling; its own bootstrap gives ≈ 0.10);
any coefficient from `aids_iam` or `coil_del` (**not identifiable**, VIF 18.1 / 16.2); a dose–response
for the LB/UB straddle (**retracted** — identity artefact); the straddle's weight transfer with a
p-value (7/8, p = 0.070 — only R² falling 8/8 at p = 0.0078 is supported); `43 s/graph` or `≈ 520×`
(**unprovenanced**); *"clears the size baseline on 5 of 5 Suite-2 datasets"* (**true under UB only —
the verdict inverts on 7 of 10**). Full list: `T-06-article-notes.md` §10.
