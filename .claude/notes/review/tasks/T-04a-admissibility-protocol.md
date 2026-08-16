# T-04a annex — metric admissibility: frozen protocol

**Status: FROZEN by the commit that adds this file, before any of E1–E4 runs.**
**Parent**: [T-04a design note](T-04a-design.md). **Feeds**: T-17 (the AE.3 table), T-20, T-06.
**Serves**: AE.3, R1.2b's *uniqueness* axis, R3.6b.

---

## 0. The question, stated precisely

The T-04a grid established *which* representations admit a distance under `competitors.md` §3.4.
This annex establishes **why**, in a form a reviewer can check, and separates two failure modes that
are routinely conflated — including in the request that prompted this annex.

For a representation `R` and a distance `d`, write `d_R(G,H) = d(R(G), R(H))`. The object we care
about is `d_R` **on the quotient space of isomorphism classes**, because that is the space graph
edit distance lives on. There are exactly three outcomes:

| | `d_R(G, π(G)) = 0` for every relabelling π? | `d_R(G,H) = 0 ⇒ G ≅ H`? | `d_R` on iso-classes |
|---|---|---|---|
| **I. not invariant** | **no** | — | **not a well-defined function**: its value depends on a choice of labelling |
| **II. invariant, not injective** | yes | **no** | a **pseudometric** |
| **III. invariant and injective** | yes | yes | a **metric** |

**Class I and class II are opposite defects.** A class-I representation *over*-discriminates: it
separates graphs that are the same. A class-II representation *under*-discriminates: it identifies
graphs that are different. An argument that excludes one does not touch the other, and treating them
as one criterion would exclude the wrong methods.

---

## 1. Pre-declared decisions — frozen, and none of them may be revisited after seeing a number

**D-A1. Class I is excluded from the running comparison.** Already `competitors.md` §3.4 and already
executed by the T-04a grid: `adjacency`, `graph6`, `sparse6` have no admissible distance. This annex
supplies the evidence, not the decision.

**D-A2. Class II is RETAINED, declared, and scope-limited. It is NOT excluded.** Three reasons, all
of which predate any measurement:

1. `competitors.md` §3.3's F2 criterion already pre-commits, in its own words: *"a violation.
   Comparing a metric against a non-metric is **legitimate but must be declared**."* Excluding on
   pseudometricity would overturn a frozen criterion.
2. `wl_subtree` — the only class-II member in the pool — **beats IsalGraph on LINUX and AIDS**.
   Removing a comparator that outperforms us, on a criterion the protocol said would be handled by
   declaration, after seeing that it outperforms us, is the outcome-dependent selection decision 24
   exists to prevent.
3. It is the most-cited baseline in the field (Shervashidze et al., *JMLR* 12:2539–2561, 2011). AE.3
   asked for a side-by-side of *existing* representations; dropping the standard one invites the
   query it was meant to answer.

**What class II *does* cost**, and this is the scope limit: a class-II representation is **barred
from any claim of the form `d = 0` certifies isomorphism**, its measured collision rate is printed
beside every result it appears in, and it is excluded from Claim A by the existing rule that a
feature vector has no bit count.

**D-A3. The exclusion evidence is a characterisation, not a significance test, wherever one is
available.** For the `n²` family, invariance is *decidable*: T-04 proved by exhaustive enumeration
over every connected graph on `n = 2…6` that the strict upper triangle is relabelling-invariant
**iff** the graph is complete. A p-value against a null nobody believes is weaker than a proof, and
substituting one for the other would be a downgrade dressed as rigour. Statistics enter where the
quantity is genuinely a population parameter: **cohort composition**, **collision rates**, and
**between-representation comparisons**.

**D-A4. Intervals are exact, and the resampling unit is the graph.** Proportions get **Clopper–
Pearson** exact binomial intervals (Clopper & Pearson, *Biometrika* 26(4):404–413, 1934), because
every count of interest sits at or near a boundary (0 or 1) where the Wald interval is invalid and
can leave the unit interval. Continuous statistics get a **graph-level percentile bootstrap**, 2,000
resamples, seed 42, per D2 — ρ moved 0.07 between two 200-graph draws, so the effective sample size
is governed by graphs, not by pairs. Where zero events are observed in `N` trials, the one-sided
95 % upper bound is reported by the **rule of three** (`3/N`), not as "0".

**D-A5. Between-representation comparisons are paired and Holm-corrected.** Every representation
sees **the same graphs and the same relabellings** (D7), so comparisons are paired by construction.
Use the **Wilcoxon signed-rank** test on the per-graph statistic with **Holm** correction across the
pairwise family, and report the **matched-pairs rank-biserial correlation** as the effect size. This
family is **exploratory** and is explicitly *not* added to the frozen confirmatory family of
`preregistration.md` — it changes neither `N_max` nor `N_actual` (decision 23's precedent for
`size_null`).

**D-A6. E4's outcome is publishable in either direction, declared now.** E4 asks whether a
*non-invariant* representation significantly outperforms IsalGraph on ρ. **If it does**, that is the
headline justification for §3.4 being F5-blind and is reported as such. **If it does not**, the
F5-blindness argument loses its empirical support and is reported as resting on principle alone.
Both are stated in the paper.

---

## 2. E1 — invariance and the separation ratio

**Population**: the frozen `S200` draw (identical to the grid's, re-derived from
`(ALL_DATASETS, 200, 42)`) for the pooled result, plus a per-dataset seed-42 200-graph draw for
per-dataset rates.

**Per representation `R`, per graph `G`**: 50 independent relabellings via `fixtures.shuffled_copy`,
seed 42 — never `nx.relabel_nodes(copy=True)`, which preserves insertion order (finding 13). Record
the **self-distance** `s_i = d_R(G, π_i(G))` under `R`'s primary distance from the grid, or under
`levenshtein` where the grid admitted none.

**Reported**:

- **Invariance rate** = fraction of `(G, π)` with `s = 0`, with a Clopper–Pearson 95 % CI.
- **Separation ratio** `ψ_R = E[d_R(G, π(G))] / E[d_R(G, H) | G ≇ H]`, per representation per
  dataset, with a graph-level bootstrap CI. **`ψ = 0` ⇔ invariant. `ψ ≈ 1` means the distance
  between two relabellings of one graph is as large as between two different graphs — the
  representation is measuring node ordering, not structure.** This is the statistic the ticket board
  names, made real.
- The **exhaustive complement**: re-verify over every connected graph on `n = 2…6` (1/2/6/21/112 =
  142 graphs, OEIS A001349) and extend to `n = 7` if it costs under ten minutes, that the invariant
  set for the `n²` family is exactly the complete graphs. **This is the load-bearing evidence**; the
  sampled rate is the cohort statement layered on top of it.

## 3. E2 — completeness, i.e. is it a metric or a pseudometric

**Population**: per-dataset seed-42 200-graph draws, all `C(200,2) = 19,900` pairs, all ten datasets.

**Per invariant representation**: enumerate every pair with `d_R = 0` and settle `G ≅ H` with an
**exact** isomorphism test (`networkx` VF2; a `pynauty` certificate may be used as a pre-filter but
never as the verdict). A pair with `d_R = 0` and `G ≇ H` is a **collision**.

**Reported**: collision rate with an exact Clopper–Pearson CI per representation per dataset, and
the rule-of-three upper bound where zero collisions are observed. **Also the converse**: pairs with
`G ≅ H` but `d_R > 0`, which must be 0 for an invariant representation — a nonzero count is a defect
in our code, not a property of the method, and is escalated rather than reported.

**Proof by exhibition, no statistics**: `K₃,₃` versus the triangular prism — both 3-regular on 6
nodes, non-isomorphic, indistinguishable to 1-WL. Assert every class-III backend separates them and
that `wl_subtree` does not. One fixture settles the qualitative claim; the rates quantify how often
it bites on real data.

## 4. E3 — the metric axioms, exhaustively rather than by sampling

The grid checks F2 on 5,000 random triples. Here: **all** triples over the 142 connected graphs on
`n ≤ 6` (`C(142,3) = 470,660`), plus all `C(200,3)` triples on `S200` for any distance cheap enough.

Identity, symmetry, and `d(a,b) ≤ d(a,c) + d(c,b)`. **Levenshtein is provably a metric on strings
and the WL kernel distance is a pseudometric induced by a PSD kernel, so the expected violation count
is exactly zero. This is a correctness check on our implementation, not a discovery** — a nonzero
count is a bug and is escalated. Zero violations are reported with the rule-of-three bound.

## 5. E4 — quantify the trap that F5-blindness exists to prevent

**The sharpest form of the argument, and the reason this annex is worth running.**
`competitors.md` §3's header already asserts it: *the raw adjacency matrix scores ρ = 0.75–0.87
against exact GED while failing F3.* E4 measures it.

For every representation, **including the three excluded ones**, report jointly:

| ρ(d_R, exact GED) | ψ_R (separation ratio) | reading |
|---|---|---|
| high | ≈ 0 | a good, well-defined graph distance |
| **high** | **high** | **the trap**: correlates with GED *and* assigns large distances between identical graphs |
| low | ≈ 0 | well-defined and weak |

**The test**: is `ρ_adjacency > ρ_isalgraph_pruned` under the paired graph-level bootstrap (D7, same
resamples), per Suite-1 dataset? Report the paired difference, its 95 % CI, and Holm-corrected
p-values across the five datasets.

**If the difference is positive and its CI excludes 0**, the statement the paper can make is exact
and strong: *a representation whose distance is not a well-defined function on isomorphism classes
significantly outperforms every admissible representation on correlation with GED — which is why
correlation with GED cannot be the criterion by which a distance is selected.* That sentence is the
justification for decision 24, and it is currently asserted rather than measured.

---

## 6. Deliverable

Self-contained, on the external drive, mirroring the convention T-05's `APPROX_GED/` established:

```
/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/experiments/metric-admissibility/
  PROTOCOL.md      a copy of this file, as run
  README.md        what it concludes, and how to re-run it
  results/*.json   one file per experiment, schema in each
  figures/*.pdf    via isalgraph.viz -- never hand-rolled matplotlib
  code/            a snapshot of the modules that produced it
  manifest.json    sha256 and shape of every file
```

The runnable code lives in the repository at `src/isalgraph/competitors/admissibility/` so that it
is tracked, tested, and passes `ruff` and `mypy --strict` like everything else in `src/`; the folder
above carries the snapshot, the results and the report.

## 7. Stop and ask

- Any class-III representation shows a **nonzero collision rate** — that would contradict the
  complete-invariant theorem and is a defect in our code or in the theorem, not a result.
- Any admissible distance violates the triangle inequality — Levenshtein cannot, so it is a bug.
- E4's difference is **negative** on a majority of datasets, i.e. the excluded representations are
  *worse* on ρ. D-A6 already fixes what is reported; the escalation is that the F5-blindness argument
  must then be rewritten before T-20 drafts around it.

## 8. Changelog

| Date | Change | Anything already computed? |
|---|---|---|
| 2026-08-16 | Initial freeze, D-A1…D-A6 and E1–E4 fixed before any run | no |
