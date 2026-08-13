# Response-letter fragment — R3.5c (and the statistical half of R3.5a)

**Emitted by**: T-02, 2026-08-13 · **Assembled by**: T-14 · **Register**: academic "we", active voice
**Status**: draft. Numbers marked ⚠ are unverified until the named ticket closes.

---

## R3.5c — pair dependence, the bootstrap, and the resampling unit

> *"The analysis treats graph pairs as independent observations … the bootstrap procedure is not
> described … resampling should be at the graph level."* (`mail.txt:106`)

**Draft response.**

We accept this. The submitted analysis resampled graph *pairs*, and pairs sharing a graph are not
independent: `d(G₁,G₂)` and `d(G₁,G₃)` both depend on `G₁`. The asymptotic Spearman test we reported
compounded the error, since its null distribution is derived for independent observations and our
pair counts overstate the information the data carry. We have replaced the procedure rather than
adjusted it.

All uncertainty now comes from a **graph-level cluster bootstrap**: graphs are resampled with
replacement and the correlation is recomputed over the induced pair submatrix, 2,000 replicates,
percentile intervals, seed 42. All significance comes from a **Mantel permutation test** with joint
row-and-column permutation of graph labels. Comparisons between two representations on one dataset
use the *same* resample for both correlations and report the percentile interval of the difference.
We did not use Hotelling–Williams or Steiger, whose textbook application to dependent correlations
assumes independent observations and would have repeated the reviewer's objection in a less visible
form.

The consequence is a loss of apparent precision, and we report it as such. The effective sample size
is governed by the number of graphs, not pairs: LINUX contributes **89 graphs**, not 3,916
independent observations. Several intervals widen and some statements that were significant under the
submitted analysis no longer are. We regard that as the correct result.

We also pre-registered the confirmatory family. Before computing any p-value we enumerated every
confirmatory comparison and fixed its cardinality at **197 tests**, arranged as three families tested
in fixed sequence — a calibration gate (5), a bracket-agreement gate (10), and the primary
comparisons (182) — with Benjamini–Hochberg control at q = 0.05 applied within each. The enumeration,
its date, and the rule by which the family may shrink are stated in the paper and in the
supplementary material. Every table reports its resampling unit, replicate count, permutation count,
interval method, seed and subsample size.

*(~290 words. Trim target under page pressure: the Hotelling–Williams sentence and the final
sentence of paragraph 3.)*

---

## R3.5a — the statistical half (exclusions and removals)

T-02 owns the **definition** of the pair-accounting ladder; T-06 fills it and T-03's fragment
(`T-03-letter-R3.5a.md`) covers the exact-GED half. Frozen ladder, per dataset:

```
raw → connected → GED-available → GED > 0 → Lev > 0 → analysed
```

with the connectivity-retention percentage as a printed column. Two rules travel with it:
non-computable exact GED is **interval-censored `[LB, UB]`, not missing** (D11), and encoding-censored
graphs are **analysed with a greedy-min fallback plus a complete-case sensitivity arm**, never dropped
(D14) — because the graphs that fail canonicalisation are exactly those with large automorphism
groups, and dropping them would remove the hardest cases from a claim about scalability.

---

## Provenance

| Claim | Source artifact |
|---|---|
| Graph-level bootstrap, 2,000 replicates, seed 42 | [statistics](../plan/statistics.md) D2 |
| Mantel, 9,999 permutations | [statistics](../plan/statistics.md) D3; `correlation_metrics.py::mantel_test` |
| Same-resample difference CI, not Steiger | [statistics](../plan/statistics.md) §4 (D7) |
| LINUX = 89 graphs | [data](../plan/data.md) §1 — Suite-1 row, **verified** (`export_graphs.py` asserts it) |
| `N_max = 197`; 5 + 10 + 182 | [preregistration](../plan/preregistration.md) §1, §4.2 |
| Freeze date 2026-08-13 | [preregistration](../plan/preregistration.md) §8 changelog |
| Ladder definition | [statistics](../plan/statistics.md) §10 |
| ⚠ Any Suite-2 count quoted here | **unverified pending T-01** — do not add one to this fragment |
