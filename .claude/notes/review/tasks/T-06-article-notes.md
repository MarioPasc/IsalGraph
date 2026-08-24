# T-06 — article notes

Written 2026-08-24 at close. **Ordered by consequence**: items that change what the paper may claim
come first, reporting obligations after, and the "not claimable" list last — that section prevents
more damage than the rest of the file.

Every number names the artifact that produced it. All artifacts are archived at
`results/reports/T-06-full-recompute/` on the Sandisk2TB volume.

---

## 1. 🔴 Claim B must be re-scoped. It fails against exact ground truth. — **T-20 §Results, T-14**

> **On 17 of 25 records (68 %), node-count difference alone predicts graph edit distance
> *significantly* better than the canonical string. Every one of the 17 is significant at the
> graph-level bootstrap; one record is undetermined; the remaining 7 favour the string.**

**Including 4 of 5 Suite-1 datasets against *exact* GED**, where no bracket argument applies:

| dataset | ρ(Lev, exact) | size null | excess |
|---|---:|---:|---:|
| `iam_letter_low` | 0.9278 | 0.9139 | **+0.0139** clears |
| `iam_letter_med` | 0.8833 | 0.9146 | −0.0313 |
| `iam_letter_high` | 0.6660 | 0.9195 | −0.2536 |
| `linux` | 0.4850 | 0.7097 | −0.2247 |
| `aids` | 0.3266 | 0.7863 | **−0.4597** |

`rho_table.json`; 3,731,532 Suite-1 pairs. **The trend strengthened as coverage completed** (60 % at
10 records → 68 % at 25), rather than regressing toward the mean.

**Manuscript consequence:** any sentence quoting a pooled ρ as evidence of *structural* fidelity must
be rewritten. The pooled ρ ≈ 0.93 on sparse IAM is largely the size channel — holding `n` fixed the
same arm scores **0.2608 at n = 12** and **0.135** averaged over n 13–30 (`size_profile.json`).

---

## 2. The benchmark itself is size-dominated — a methodological contribution — **T-20 §Evaluation**

> **Node-count difference alone attains ρ = 0.71–0.997 against ground-truth GED on these benchmarks,
> exceeding 0.96 on seven of ten Suite-2 datasets. On `coil_del` it reaches 0.9971.**

This is a property of the **data**, not of the approximation: exact GED is itself ≈ 0.92
size-dominated on IAM Letter. **Correlation with GED on this data therefore measures size agreement
more than structural fidelity — for every representation, ours included.**

It is the motivation for the within-`n` decomposition (item 3), which stops looking like a defensive
slice and becomes the correct instrument. Source: `rho_table.json`, `size_profile.json`.

---

## 3. The within-`n` decomposition — the correct instrument, and a field-level result — **T-20 §Evaluation**

Within a stratum `n_i = n_j`, the size null `|n_i − n_j|` is **identically zero**, so its rank
correlation is undefined and there is nothing to subtract. **Raw ρ inside a stratum is the structural
signal with the size channel removed by construction rather than by adjustment.**

> **Above `n ≈ 40`, not one of the seven representations — IsalGraph or any competitor — is reliably
> distinguishable from ρ = 0.**

That is a statement about **the approach**, not about IsalGraph, measured on 21.7 M pairs.
`isalgraph_pruned` falls from **ρ = 1.0000 at n = 3 to 0.2608 at n = 12**, every point BH-significant,
so the decay is not a power artefact. Source: `size_profile.json`, 2,355 strata / 1,553 with a
defined ρ / 582 aggregated points.

**Not a budget artefact — measured.** Removing every censored-touching pair *lowers* ρ at both bounds
and both size restrictions (LB all-`n` −0.0305, LB n>40 −0.0354, UB all-`n` −0.0170, UB n>40 −0.0725):
`size_profile_censoring_confound.json`. Report all three quantities, never the Δ alone — censored
pairs do correlate worse in isolation (0.3273 against 0.6095 at n > 40); they simply do not explain
the collapse.

---

## 4. The IAM Letter control — the limitation, stated as a condition — **T-20 §Limitations**

LOW / MED / HIGH are the **same generator at three distortion levels**: source, labelling and
construction fixed, only the graphs differ. Node count barely moves (mean 4.07 → 4.58, unchanged
spread); **mean edge count rises 3.07 → 4.56 (+49 %)**. The family adds *structure*, not *size*.

| dataset | ρ(Lev, exact) | size null | β_lev | β_size | ratio |
|---|---:|---:|---:|---:|---:|
| LOW | 0.9278 | 0.9139 | +0.5624 | +0.3537 | **0.63×** |
| MED | 0.8833 | 0.9146 | +0.4610 | +0.5086 | **1.10×** |
| HIGH | 0.6660 | 0.9195 | +0.2696 | +0.7507 | **2.78×** |

> **Holding the generator fixed and adding structural distortion, the trivial size baseline stays flat
> at ρ ≈ 0.92 while the canonical string falls from 0.93 to 0.67. β_lev halves, β_size doubles, and
> both instruments cross at MED.**

**This is the honest limitation and it is far better than "degrades on harder data"** — it names what
gets harder, shows the baseline unaffected, and puts the break-even **between LOW and MED distortion**
rather than at a coincidental size. Three rows, checkable at a glance.

---

## 5. Claim A — a real advantage, and the scope is mandatory — **T-20 §Results**

> **Against `min_dfs` — which `competitors.md` §2 calls "the single most important comparator", and
> which is itself a canonical code — IsalGraph is shorter on 112 of 112 size strata above `n = 20`,
> median +214.8 bits, zero losses and zero ties.**

**The advantage grows with size**, opposite in direction to Claim B:

| `n` | 1–5 | 6–10 | 11–20 | 21–40 | **41+** |
|---|---|---|---|---|---|
| shorter than competitor | 20.4 % | 16.3 % | 18.9 % | 30.0 % | **45.6 %** |
| median gap | −1.2 bits | +0.5 | +5.8 | +58.6 | **+242.1** |

`claim_a_strata.json`, 1,578 strata, 7 comparators, IUT `p = max` over both bit conventions.

> ⚠ **The scope is not decorative.** Pooled over all sizes the significant A1 results run **10 against
> / 9 for**. **Never write "IsalGraph produces shorter encodings" unqualified** — *"above n ≈ 20"*
> goes in the same sentence, every time.

**Frozen wording for the general claim:** *"IsalGraph is the most compact of the canonical-code
representations. Edge-list serialisations beat it at scale."* Say **edge-list**, not
"sparsity-exploiting": it names the mechanism, and conceding it is what makes the `min_dfs` result read
as fair.

---

## 6. Completeness at cohort scale — the one unscoped positive — **T-20 §Results, opening**

> **Zero encoding collisions across 24,764,422 GED-positive pairs.** Suite 1 against **exact** GED, so
> `GED > 0` certifies non-isomorphism: **3,424,764 certified pairs, zero collisions.** Suite 2 at
> `LB > 0`: **21,339,658 further pairs, zero collisions.**

It survives D14: the 101 censored Mutagenicity graphs carry a greedy-min fallback that is **not**
canonical and therefore outside the theorem — and they collide with nothing either.

**It is a count, not an estimate — there is no interval to argue with**, which is why it leads.
Source: `ladder.json`, `ladder_suite1.json`.

*Caveat, one clause:* on Suite 2 the certification is `LB > 0`; `LB = 0` does not certify isomorphism,
so pairs the bound could not separate are outside that half. Suite 1 has no such gap. **Say it.**

---

## 7. Encoder cost is governed by |Aut|, not by n — predictive — **T-20 §Complexity**

> **D14 censoring at the frozen 300 s budget is 0 % for all 3,703 Mutagenicity graphs with
> `|Aut| ≤ 10⁴`, 21.85 % at 10⁴–10⁸, and 100 % (35 of 35) above 10⁸.**

Nearly a step function in **symmetry**, not size — mechanistically right, since the canonical search
space is governed by the automorphism group and `n` was only ever a proxy.

**This is predictive**: a user computes `|Aut|` in milliseconds and knows in advance whether the method
will encode their graphs. Report the censoring rate **by stratum**, never as the diluted dataset figure
(2.50 % of Mutagenicity) or the cohort figure (0.62 %) — both hide the structure. Source:
`censoring.json`.

---

## 8. No representation leads on both axes — **T-20 §Discussion**

> **The most compact serialisation (`sparse6`) admits no metric satisfying the distance axioms — it is
> k-excluded at F3 = 1/50. The best-correlating representation (`wl_subtree`) admits no bit count —
> `BitCountUndefined`. Neither axis-leader is evaluable on the other axis.**

Among those measurable on both: IsalGraph is decisively more compact than `min_dfs` and
`nauty_graph6`; its correlation against them is **bracket-dependent** — indistinguishable under LB,
weaker under UB. **It is dominated on both axes by `sparse6_nauty`, which is both more compact and
better correlated.** *Say the last clause* — omitting the one representation that dominates us is the
most checkable dishonesty available in this paper. Source: `.claude/notes/review/tasks/t06_dominance.py`.

---

## 9. Reproducibility — **T-21**

> **An unplanned determinism check: 14 MRM fits computed twice by independent processes, in separate
> output trees, started hours apart, at seed 42, are byte-identical — β₁, the full standardised β
> vector, R² and the permutation p-value, to the last stored digit.**

Not designed to succeed, which is why it is worth quoting. Plus three artifact types re-derived from
raw inputs by an independent path, **max discrepancy 1.1 × 10⁻¹⁶**.

### Reproduction parameters — a number without these is not reportable

| | |
|---|---|
| engine | `cpp`, build hash **`298fc1188bf1b051`**, gcc 12.2.0, `-march=x86-64-v3` |
| seed | **42** throughout |
| encode budget | **300 s per graph**, enforced by a killed subprocess (D14) |
| cost model | node ins/del = 1, edge ins/del = 1, substitutions free (D6) |
| exact GED | `networkx` A*, `n ≤ 12` ceiling |
| bracket | BRANCH-FAST (LB) / IPFP (UB), GEDLIB via `graphkit-learn` |
| resampling | **graph-level** cluster bootstrap; D15 tiers 1/2/3 = (2000, 9999, —) / (2000, 4999, —) / (1000, 1999, 2×10⁶) |
| multiplicity | BH-FDR at q = 0.05 within each family; FCR-adjusted intervals for the gates |
| hardware | 24-core workstation; **no timing from these runs is publishable** (concurrent shards) |

---

## 10. 🔴 What is NOT claimable

**Read this before quoting anything from T-06.**

| ❌ Do not write | Why |
|---|---|
| *"IsalGraph clears the size baseline on 5 of 5 Suite-2 datasets"* | True under **UB** and false under **LB** — the verdict **inverts on 7 of 10**. The most damaging available sentence, and trivially checkable. The honest word is **undetermined** |
| *"ρ ≈ 0.93 demonstrates structural fidelity"* | Mostly the **size channel**. This paper supplies the instrument that refutes it (item 3) |
| *"competitive with the best representations"* on Claim B | Best on **none** of the records. Not a scoping — a contradiction |
| *"most compact among representations admitting a metric"* | **False.** True in **0 of 122** strata; `sparse6_nauty` beats it at every size above 20 |
| *"IsalGraph computes everywhere, unlike the competitors"* | **Eight representations also complete on 100 % of every cell.** Only `agm_cam` (6.15 % floor) and `min_dfs` (0.948) are worse |
| *"IsalGraph produces shorter encodings"* unqualified | Pooled over all sizes, A1 rejections run **10 against / 9 for**. The *"above n ≈ 20"* scope is load-bearing |
| *"N of M cells are significant"* as evidence of success | The 75 BH rejections split **35 for / 34 against**. A rejection is against `H₀: Δ = 0` and can mean *significantly worse* |
| Any β₁ without β_size beside it | **410 of 1,329** strata are discordant between bit conventions; a coefficient without its competitor inverts in meaning |
| `mutagenicity`'s β_lev = +0.5229, or *"Levenshtein dominates on the largest dataset"* | **RETRACTED.** Its own bootstrap puts β_lev in **[0.092, 0.103]** — a fivefold disagreement caused by tier-3 subsampling (`statistics.md` §5). Which predictor dominates reverses |
| Any coefficient from `aids_iam` or `coil_del` | **Not identifiable** — VIF 18.1 and 16.2, `r(Lev,\|Δn\|)` = 0.96 and 0.94. High R² and a small p-value are compatible with an arbitrary split |
| A dose–response for the LB/UB straddle | **RETRACTED** — an artefact of `drop = ratio_lb − ratio_ub` correlating X with X − U. Permutation null gives +0.90 for pure noise; observed +0.95, **p = 0.29** |
| The LB→UB weight transfer with a p-value | **7 of 8, p = 0.070 — not significant.** Descriptive only. **Only R² falling 8/8 (p = 0.0078) is supported** |
| `43 s/graph`, `≈ 520×`, `≥ 6.8 core-hours` | **Retracted as unprovenanced** — the run that produced them left no artifact |
| Any F0/F1/F2 result restated more favourably than it came out | The confirmatory layer is pre-registered; its value is that it is reported unchanged |

### Measured a mechanism for, never an incidence of

- **The LB/UB straddle direction.** LB is +0.06 *more* size-dominated than exact, UB −0.18 *less*, and
  the weight transfers accordingly on 7 of 8 bound-pairs. **The direction is real and predicted the β₁
  finding before measurement; the magnitude is not quantified** and must not be presented as a law.
- **Encoder cost by |Aut|** (item 7) is measured on **Mutagenicity only** — the only dataset that
  censors. The step function is not established cohort-wide.

### Properties of the measurement setup, not of the object

- The **2.50 % / 0.62 % censoring rates** are properties of the **300 s budget** and the hardware. The
  `|Aut|` stratum is the transferable form.
- The **`n ≤ 12` exact-GED ceiling** is a property of `networkx` A* and the per-pair budget, not of GED.
- **`aids` and `linux` lose ~56 % of pairs at the GED-available rung**, and the survivors are a
  **size-biased subsample, not a random one** — any ρ on those two is computed on the small-graph half.
