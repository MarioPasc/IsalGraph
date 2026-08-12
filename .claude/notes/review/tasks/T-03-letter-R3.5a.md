# Response-letter fragment — R3.5a, the pair-accounting ladder

**Draft from T-03, 2026-08-12.** Numbers marked ⟨…⟩ are pending the AIDS and Letter runs; every
other figure is measured and traceable to `data.md` §1 or to this ticket's work log. **Do not print
a ⟨…⟩ value.**

Style contract: academic "we", active voice, no significance inflation, one term per concept,
quantify everything.

---

## Draft

> The reviewer asks how many pairs enter each analysis and how many are lost on the way. The
> submission did not report this, and we now give the full ladder per dataset. We also correct two
> defects in the ground truth that the accounting exposed.
>
> **The cohort.** We apply one filter to every dataset — at least two nodes, connected, at most
> twelve nodes — and merge the train, validation and test splits, since graph edit distance is
> symmetric and carries no train/test semantics. This retains 5,350 graphs and 3,897,911 pairs.
>
> | Dataset | raw | connected & 2 ≤ n ≤ 12 | retention | pairs |
> |---|---:|---:|---:|---:|
> | IAM Letter LOW | 2,250 | 1,180 | 52.4 % | 695,610 |
> | IAM Letter MED | 2,250 | 1,253 | 55.7 % | 784,378 |
> | IAM Letter HIGH | 2,250 | 2,059 | 91.5 % | 2,118,711 |
> | LINUX | 89 | 89 | 100 % | 3,916 |
> | AIDS | 911 | 769 | 84.4 % | 295,296 |
> | **Total** | | **5,350** | | **3,897,911** |
>
> The discards separate cleanly: Letter loses 1,069, 993 and 183 graphs to disconnection and 1, 4 and
> 8 to the two-node floor; AIDS loses 91 to disconnection and 51 to the twelve-node ceiling. We note
> that the AIDS raw count is 911 rather than the 819 implied by the submission, since the
> distributed `graphs.json` is already connectivity-filtered and 819 is the connected count.
>
> **Two corrections to the ground truth.** Recomputing every distance surfaced two problems with the
> reference used in the submission, and we report both because each moves the published
> correlations.
>
> First, the graph edit distances distributed with GraphEdX cover only pairs **within** a split.
> LINUX carries 1,685 of 3,916 pairs (43.0 %) and AIDS 181,909 of the pairs on its raw set (43.9 %).
> The submitted ρ = 0.433 for LINUX and ρ = 0.349 for AIDS are therefore within-split figures, which
> the source does not state and which we did not previously recognise.
>
> Second, those distributed values are not exact. Under GraphEdX's own cost model we recomputed 208
> AIDS pairs with an exhaustive A* search and found 150 below the published value, 58 equal, and none
> above, with a mean difference of 1.58 edit operations and a maximum of 8. Since graph edit distance
> is a minimum and an A* search returns an achievable edit path, a value below the published one is a
> proof that the published one is not optimal. For the AIDS training pair (76, 211) the distributed
> matrix gives 11 while we exhibit a path of cost 6. We conclude that the distributed matrix is an
> approximate upper bound rather than exact graph edit distance. The strictly one-sided discrepancy
> is what identifies the reference rather than our solver as the source: a faulty solver errs in both
> directions.
>
> **What we compute instead.** We recompute every distance ourselves under a single cost model — unit
> cost for node and edge insertion and deletion, free substitution — for the reasons given in our
> response to comment R3.5b. We obtain exact values with an A* search run to completion, and we
> record for each pair whether the search completed. A pair whose search reaches its 300-second
> budget without completing is **interval-censored** at the proven bracket [LB, UB], with the lower
> bound from BRANCH-FAST and the upper bound from IPFP, both from GEDLIB; it is never dropped and
> never recorded as exact. This matters because the submitted pipeline recorded the best value found
> at timeout as though it were exact, which is the defect that produces silently non-optimal entries.
>
> We report the censoring rate per stratum rather than pooled, since censoring correlates with graph
> size and symmetry. For LINUX, 3,911 of 3,916 pairs (99.87 %) are exact and 5 are censored.
> ⟨Letter and AIDS rows pending.⟩
>
> **The ladder.** For each dataset we now report `raw → connected → within the size ceiling →
> distance available → distance > 0 → Levenshtein > 0 → analysed`. ⟨Table pending the full run.⟩
>
> Two consequences follow, and we state them plainly. Our correlations will not reproduce the
> published GraphEdX figures, because the pair set grows by 2.3× on LINUX and 2.25× on AIDS and the
> cost model changes. And the AIDS and LINUX values in the submission were computed against an
> approximate reference, so their replacements are not directly comparable with them. We give both
> and explain the difference rather than quietly substituting one for the other.

---

## Provenance

| Claim | Source |
|---|---|
| cohort counts, retention, drop breakdown | `export_graphs.py`, run 2026-08-12; reproduces `data.md` §1 exactly |
| AIDS raw 911, not 819 | `data.md` §1, audit I-02; re-verified on the source tree |
| within-split coverage 43.0 % / 43.9 % | `exact_ged.md` §2 |
| 208 pairs, 150/58/0, mean 1.58, max 8 | gate 0, `task-gedlib-gates` log |
| pair (76, 211): published 11, achievable 6 | orchestrator's independent verification, this ticket |
| LINUX 3,911 exact / 5 censored | measured, full LINUX run, this ticket |
| timeout recorded as exact | `ged_computer.py::compute_ged_pair`, read 2026-08-12 |
| 2.3× and 2.25× pair growth | `exact_ged.md` §6, `data.md` §5 — **not** 1.62×, which mixes populations |

⚠ The draft says a **300-second** budget because that is the plan's stated censoring budget for the
reported analysis. **The runs so far used 60 s**, the submission's `ged_computer` default. Reconcile
before printing: either re-run at 300 s or print 60 s. The LINUX censoring rate of 0.13 % is a 60 s
figure.

---

## ⚠ 2026-08-13 — THIS FRAGMENT IS SUPERSEDED AND MUST BE REWRITTEN

The paragraph beginning *"Second, those distributed values are not exact"* asserts a claim that
has since been **retracted**. See `T-03-design.md` amendment 4.

GraphEdX's published AIDS matrix uses **unit node costs — the same model as D6** — not the zero
node cost the plan assumes. Measured 4/4 against unit, 0/4 against zero, with the published value
exceeding the zero-node value by exactly `|n1 - n2|` in every case. The "150 below, 58 equal, 0
above" figure was the arithmetic of comparing under the wrong cost model, not evidence of
non-optimality.

**What the rewritten fragment should say instead**, measured over the full overlap:

- our values and GraphEdX's agree on **105,270 of 105,272** finite AIDS overlap pairs and
  **1,665 of 1,665** finite LINUX overlap pairs;
- **zero** pairs where ours exceeds theirs — the direction that would falsify our solver;
- **two** AIDS pairs where ours is lower by 2, both certified, so those two published entries are
  provably non-optimal. Two in 131,148 is a rounding error and should be reported as such, not
  built into an argument.

**The recompute's justification stands on its original two grounds**, both untouched:
GraphEdX publishes GED for **within-split pairs only** (131,148 of 295,296 AIDS pairs = 44.4 %;
1,685 of 3,916 LINUX = 43.0 %), and IAM Letter ships **no GED matrix at all**, so one cost model
across the cohort is unobtainable from the distributions as shipped.

Do not reuse the retracted paragraph. The provenance document generated with the data
(`GED_PRECOMPUTED/extended_merged_exact_ged/PROVENANCE.md`) carries the corrected version.
