# Verified numbers — wave 2026-08-30-article-wl

**Every figure below was recomputed from the primary artifacts by the orchestrator on 2026-08-30**,
not transcribed from a report or from `prose.md`. Where this file disagrees with `prose.md`'s claim
register, **this file wins** and the disagreement is recorded.

Sources, all under
`/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/results/reports/T-06-full-recompute/`:

- `wl/data/t28_probe_point_estimates.json` — per (cell, reference, representation, view) ρ
- `wl/data/t28_bootstrap_verdicts.json` — paired graph-level bootstrap, 14 cells, both views
- `wl/data/t28_signtest_equal_n.json` — per-stratum sign tests inside equal-`n` strata
- `ged/data/editpath_results_full.json` — the edit-path admissibility campaign

> ## 🔴 CORRECTED 2026-08-30 — an authoritative `size_null_rho` field exists; do not derive it
>
> An earlier version of this file derived the size null as `ρ_arm − excess_arm`, mixing the
> bootstrap's **intersected** pair set with the probe file's **own-pairs** ρ. **That was wrong for
> `aids` and `linux`.** `t28_probe_point_estimates.json` carries a per-row **`size_null_rho`** field,
> and `ged/data/rho_table.json` carries `size_null.point`. **Read those. Never derive it.**
>
> The two conventions coincide wherever every arm completes — the three IAM Letter cells — and
> diverge where `agm_cam` and `min_dfs` do not. This is the denominator problem in its fourth
> costume today, and it shipped into the manuscript before being caught.

Convention used throughout the tables below: **point estimates over each cell's own pairs**, read
from `size_null_rho` (WL and spectral references) and `size_null.point` (`exact`/`lb`/`ub`). The
paired bootstrap is a **different estimand on the intersected pair set** and is used only where an
interval or a significance verdict is quoted. **Never put the two in one table row.**

---

## 1. Claim-register audit — three claims checked, two defective

| Claim | Verdict |
|---|---|
| **C14** | 🔴 **Defective.** Two medians, two denominators, one sentence |
| **C15** | ✅ **Sound. Write it as frozen** |
| **C17** | 🔴 **Defective.** Counts do not reproduce from any primary file |

### C14 — the size null drop

| ρ(\|n_i − n_j\|, d_ref) | denominator | value |
|---|---|---:|
| exact GED | the 5 Suite-1 datasets | **0.9139** |
| WL kernel | **the same 5 datasets** | **0.4283** |
| WL kernel | all 14 cells | 0.5160 |
| WL kernel | 11 distinct datasets (`wl/REPORT.md` §3.1) | 0.5700 |

**Authoritative** per-cell Suite-1 values, `size_null_rho`, view `all_pairs`, arm
`isalgraph_pruned` — with the exact-GED counterpart from `rho_table.json`:

| dataset | exact ρ | exact null | excess | WL ρ | WL null | excess |
|---|---:|---:|---:|---:|---:|---:|
| AIDS | 0.3266 | 0.7863 | −0.4597 | 0.3154 | **0.2127** | +0.1027 |
| LINUX | 0.4850 | 0.7097 | −0.2247 | 0.4855 | **0.1585** | +0.3270 |
| IAM Letter LOW | 0.9278 | 0.9139 | +0.0139 | 0.7128 | 0.5696 | +0.1432 |
| IAM Letter MED | 0.8833 | 0.9146 | −0.0313 | 0.7109 | 0.5160 | +0.1950 |
| IAM Letter HIGH | 0.6660 | 0.9195 | −0.2536 | 0.5959 | 0.4283 | +0.1676 |

C14's first half is explicitly Suite-1, so the matched pairing is **0.914 → 0.428**. The frozen
wording's 0.516 is a 14-cell median. **Never write "0.914 to 0.516" unqualified.**

🔴 **C14 is wrong a second time: it says the correlation falls "on three of the five".** At full
precision it falls on **four of five** — AIDS −0.011123, IAM Letter LOW −0.215053, MED −0.172369,
HIGH −0.070100 — and rises on LINUX by **+0.000507**, which is flat. **Write "four of the five".**
The correction strengthens the claim it sits in, which is why it must be made.

### C15 — the nauty comparison (verified, use as frozen)

From `t28_signtest_equal_n.json`, arm `isalgraph_pruned`, strata deduplicated by (dataset, `n`):

| reference | competitor | band | strata | higher | lower | median Δρ | p |
|---|---|---|---:|---:|---:|---:|---:|
| `wl` | `nauty_graph6` | n ≤ 20 | 94 | **58** | **31** | +0.0460 | **0.005545** |
| `wl` | `sparse6_nauty` | n ≤ 20 | 94 | **58** | **31** | +0.0489 | **0.005545** |
| `wl` | `nauty_graph6` | n > 20 | 110 | 53 | 57 | −0.0210 | 0.775 |
| `wl` | `sparse6_nauty` | n > 20 | 110 | **40** | **70** | −0.0493 | **0.005447** |
| `wl` | `min_dfs` | n ≤ 20 | 94 | 28 | 60 | −0.0415 | 0.000847 |
| `wl` | `min_dfs` | n > 20 | 110 | 42 | 59 | −0.0341 | 0.111 |
| `wl` | `agm_cam` | n ≤ 20 | 63 | 21 | 36 | −0.0175 | 0.0627 |
| `exact` | `nauty_graph6` | n ≤ 20 | 23 | 15 | 5 | +0.0572 | 0.0414 |
| `exact` | `sparse6_nauty` | n ≤ 20 | 23 | 15 | 5 | +0.0524 | 0.0414 |
| `exact` | `min_dfs` | n ≤ 20 | 23 | 2 | 18 | −0.0790 | 0.000403 |
| `exact` | `agm_cam` | n ≤ 20 | 23 | 1 | 19 | −0.0690 | 4.01e−05 |

Both nauty arms are **58/31 at n ≤ 20**, which is why C15 says *"p = 0.0055 against each"*.

**This table also settles a red line**: under **exact GED** at `n ≤ 20` the arm leads both nauty arms
15/5 at p = 0.041, so the `n ≤ 20` lead holds under exact GED *and* WL. The reversal above `n = 20`
is a size effect, not a reference effect.

**One scope caveat.** C15's closing *"outranked by the minimum DFS code under every reference"* is
unambiguous on the dataset-level bootstrap (3 W / 2 T / 9 L). By this per-stratum test the `n > 20`
WL cell is 42/59 at **p = 0.11** — losing on the point estimate, not significant. Keep that sentence
on the bootstrap, or scope it. Do not blur the two estimands.

### C17 — the family split (rewritten on measured values)

Reference `wl`, view `all_pairs`, the 14 bootstrap cells:

| arm | cells with positive excess | mean excess |
|---|---:|---:|
| `isalgraph_pruned` | 12 / 14 | **+0.1250** |
| `min_dfs` | 12 / 14 — **the same twelve** | **+0.1494** |
| `nauty_graph6` | 5 / 14 | −0.0406 |
| `sparse6_nauty` | 2 / 14 | −0.0928 |
| `agm_cam` | **disputed — see below** | |

Both canonical codes fail on the same two cells: `2/coil_del` and `2/protein`.

**C17 as frozen is wrong twice.** It says the serialisations clear on *"one and zero"* (measured:
five and two) and that AGM CAM clears twelve *"alike"* (measured: neither 12 nor reproducible). Its
*"+0.148 against +0.125"* appears in **no primary file** — only in `prose.md` and the wave contract.

**AGM CAM gets a clause, not a count.** Two independent recomputations give 9/14 (mean −0.0432) and
14/14 (mean +0.0609). It refuses above `n = 12` by its own scope guard and completes on **6.15 %** of
Protein, so its ρ is conditioned on the graphs symmetric enough to finish, and pairing it with an
all-pairs null versus a restricted null gives different, individually defensible answers. State the
scope limitation instead.

**IsalGraph is not distinctive here**: min-DFS clears the same twelve cells with a *larger* mean
excess. The sentence must not imply otherwise.

---

## 2. The WL paired bootstrap (from `wl/REPORT.md` §3.4, cross-checked)

14 of 15 cells; `suite2/mutagenicity` timed out at 10 h and is **absent**. Four of the fourteen are
Suite-1/Suite-2 duplicates, so distinct datasets = **10**. **Say "cells", never "datasets".**

| reference | `agm_cam` | `min_dfs` | `nauty_graph6` | `sparse6_nauty` | clears its null |
|---|---|---|---|---|---:|
| GED exact | 1W 0T 4L | 0W 0T 5L | 3W 1T 1L | 3W 2T 0L | 1 / 5 |
| GED lower bound | 0W 0T 9L | 1W 0T 8L | 4W 0T 5L | 7W 2T 0L | **0 / 9** |
| GED upper bound | 1W 5T 3L | 1W 2T 6L | 2W 2T 5L | 2W 2T 5L | 5 / 9 |
| **WL kernel** | **8W 4T 2L** | 3W 2T **9L** | **12W 0T 2L** | **14W 0T 0L** | **12 / 14** |
| spectral (norm L) | 0W 0T 14L | 1W 0T 13L | 7W 0T 7L | 9W 5T 0L | **0 / 14** |
| spectral (comb L) | 4W 4T 6L | 3W 2T 9L | 12W 1T 1L | 14W 0T 0L | 2 / 14 |
| spectral (adjacency) | 0W 0T 14L | 1W 0T 13L | 3W 2T 9L | 6W 2T 6L | **0 / 14** |
| spectral ESD | 4W 3T 7L | 1W 2T 11L | 9W 5T 0L | 12W 2T 0L | **0 / 14** |

Under `equal_n`, `agm_cam` collapses from 8W 4T 2L to **2W 5T 7L**, so that result is
**view-dependent and must be quoted with its view**. The nauty result holds in both views.
`min_dfs` is nine losses either way.

Suite-1 WL excesses with intervals: `aids` +0.1121 [+0.0593, +0.1668] · `iam_letter_high` +0.1676
[+0.1434, +0.1911] · `iam_letter_low` +0.1432 [+0.1209, +0.1664] · `iam_letter_med` +0.1950
[+0.1715, +0.2192] · `linux` +0.3189 [+0.1699, +0.4454]. **5 of 5 positive.**

### The anti-reference-shopping evidence

`spectral_esd` has the **lowest size null of all eight references** (median 0.303, min −0.061) and
the encoding tracks it **worst**, clearing the null on **0 of 14**. The least size-dominated
reference is the one we do worst against. This is the strongest available answer to a charge that
the WL reference was chosen for being friendly, and it should be reported for that reason.

---

## 3. Edit-path admissibility (`ged/data/editpath_results_full.json`)

Engine `cpp`, seed 20260830, 6 replicates, 119,580 pairs × replicates, endpoints rejected: **0** for
either arm on all five datasets.

| | IsalGraph (pruned) | min-DFS |
|---|---:|---:|
| admissible intermediates | **92.0331 %** | **52.2646 %** |
| 95 % CI | [91.9091, 92.1570] | [52.0008, 52.5369] |
| intermediates | 532,315 | 246,220 |
| whole paths clean | **80.5265 %** | **38.4690 %** |
| mean path length | 5.381 | 2.988 |

**Rejection reasons — this is the mechanism, and it is exact.**

- IsalGraph: `{self_loop: 42,409}`. **One reason, nothing else.** Every inadmissible string decodes
  and stays connected.
- min-DFS: `{forward_index_not_next: 88,049, backward_source_not_rightmost_vertex: 13,835,
  forward_source_off_rightmost_path: 10,533, repeated_edge: 4,575, backward_target_not_ancestor:
  542}`. Five reasons, all ordering constraints, **zero disconnection**.

Per dataset — IsalGraph: `iam_letter_low` 100.0, `iam_letter_med` 98.0, `aids` 93.983, `linux`
91.508, `iam_letter_high` 80.968. min-DFS: `aids` 50.468, `linux` 52.683, `iam_letter_high` 53.454,
`iam_letter_med` 55.186, `iam_letter_low` 57.207.

### The steelman, and it should be reported

`min_dfs_renumbered`: **61.5734 %** [61.2916, 61.8453], whole paths clean 47.8466 %. Renumbering to
give the ordering constraints their best case lifts min-DFS from 52.26 % to 61.57 %, **and it still
loses by more than thirty points.** Reporting it pre-empts the objection that the measurement
punishes an incidental numbering convention rather than a property of the code space.

### 🔴 Do not quote the two `_permissive` arms as a rival convention

`isalgraph_permissive` = 100.0 % [100.0, 100.0]; `min_dfs_permissive` = 89.4509 % [89.2833, 89.6163].
Side by side these look like a protocol under which the gap is 100 against 89 rather than 92 against
52. **They relax different constraints for each arm**: for IsalGraph permissive drops the self-loop
rejection, its only one, so it measures decodability; for min-DFS it drops the ordering constraints
and leaves only `disconnected` (25,974). Not like-for-like, and pairing them is the denominator
error again.

The one true use: **100.0 % with a degenerate interval is the measured form of the structural claim**
that the interpreter is total on `Σ*`, so no edit can leave the language. State it as a measurement
of decodability, never as an admissibility rate comparable to 89.45.

### One provenance to confirm before it ships

The draft attributes **39.02 %** to *"the diagonal-preferring path returned by the standard
implementation"*. In the artifact that is `pooled/min_dfs/replicate_0`, **labelled a replicate**. It
is anomalous only for min-DFS — replicates 1–5 run 52.08–52.66, while IsalGraph's `replicate_0` is
91.38 against its own 92.03 — which is consistent with `replicate_0` being the deterministic
diagonal path, and 52.26 − 39.02 = 13.24 matches *"13 points below"*. **But the file does not say
so.** Confirm against the harness, or drop the attribution and keep the arithmetic.

---

## 4. Bibliography

`wilson2008spectra` — **verified 2026-08-30 against the Crossref API** for
`10.1016/j.patcog.2008.03.011`: Wilson, Richard C. and Zhu, Ping, *A study of graph spectra for
comparing graphs and trees*, **Pattern Recognition** 41(9):2833–2841, 2008. Every field returned
exactly as recorded. Owned by `sec4-design` in `review1/article/refs_added.bib`.

`shervashidze2011wl` already exists in the article's `refs_added.bib`. The article reads
**both** bibliographies: `\bibliography{cas-refs,refs_added}` at `backmatter.tex:76`.
