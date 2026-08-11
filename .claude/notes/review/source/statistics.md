# Statistical protocol — DECIDED

**Status**: v2.0, 2026-08-11. **These are decisions, not options.** Fixed before T-06 so nothing is
computed twice. Deviations require a changelog entry.

Answers R3.5a, R3.5b, R3.5c and AE.4.

---

## 1. The story, in one paragraph

The paper makes two claims and they need different statistics because **their units of analysis
differ**.

> **Claim A (information content).** IsalGraph encodes a graph in fewer bits than competing
> reversible serialisations. Unit = **one graph**. Observations are independent. Ordinary paired
> non-parametric statistics apply.
>
> **Claim B (metric locality).** Levenshtein distance on IsalGraph strings tracks graph edit
> distance better than competing representations' distances do. Unit = **one graph pair**.
> Observations are **dyadically dependent** — `d(G₁,G₂)` and `d(G₁,G₃)` share `G₁`. Ordinary
> statistics do **not** apply, and the submitted version used them.

Everything below follows from that distinction. Stating it explicitly in the paper is itself part
of the answer to R3.5c.

---

## 2. Decisions

| # | Decision | Replaces | Driver |
|---|---|---|---|
| **D1** | **Spearman ρ is the primary association measure**; Kendall τ-b reported as a tie-robustness check | — | continuity: reviewers quote ρ = 0.433 / 0.349 |
| **D2** | **All uncertainty comes from a graph-level cluster bootstrap**, 2,000 replicates, percentile CI, seed 42 | pair-level `bootstrap_correlation` | **R3.5c** |
| **D3** | **All significance comes from the Mantel permutation test**, 9,999 permutations of graph labels | asymptotic Spearman test on pair counts | **R3.5c** |
| **D4** | **MRM is a confirmatory analysis**: `GED ~ Lev + \|Δn\| + \|Δdensity\|`, permutation inference | nothing — new | pre-empts the size-confound attack |
| **D5** | **Per-dataset results are primary; pooled results are secondary** and never a headline | pooled OLS β in `conclusion.tex:38–41` | **R3.5b** |
| **D6** | **One GED cost model across all datasets**: node ins/del = 1, edge ins/del = 1, substitutions free | IAM uniform vs GraphEdX topology-only | **R3.5b** |
| **D7** | **Method comparison within a dataset** = bootstrap CI on the difference of ρ | — | dependence |
| **D8** | **Method comparison across datasets** = Friedman + Wilcoxon–Holm + critical-difference diagram | — | Demšar (2006), the venue standard |
| **D9** | **Multiplicity** = Benjamini–Hochberg FDR at q = 0.05 over a pre-declared confirmatory family | nothing | scale of the design |
| **D10** | **Effect sizes with CIs lead. p-values are supporting detail.** | significance-as-effect-size at `conclusion.tex:37` | **R3.6b** |
| **D11** | **Non-computable exact GED is interval-censored `[LB, UB]`, not missing** | silent exclusion | **R3.5a** |
| **D12** | **Censoring and timeout rates are reported per stratum, never pooled** | nothing | censoring is symmetry-correlated (`data.md` H8) |

### D6 explained — the cost model, and why it is one decision for *both* experiments

GED is defined only relative to a **cost function**: what it costs to delete a node, insert an edge,
substitute a label. The submission uses **two different ones**:

- IAM Letter: node ins/del = 1, edge ins/del = 1, node substitution = 0;
- LINUX and AIDS (from GraphEdX): **node operations cost 0**, edge operations = 1.

The same pair of graphs gets **different GED values** depending on which dataset it came from, and
Table 3 pools both onto one axis. That is R3.5b.

**Decision: unit cost — node insert/delete = 1, edge insert/delete = 1, substitutions free.**
Four justifications, in the order they should appear in the response letter:

1. **It keeps GED a metric.** With zero node-operation cost, inserting an isolated vertex is free,
   so two non-isomorphic graphs can sit at distance 0 and GED is only a *pseudo*metric. The paper's
   Corollary 2.13 asserts that the IsalGraph graph distance **is** an isomorphism-invariant metric.
   Validating a metric against a pseudometric reference is incoherent. This argument is formal,
   short, and a reviewer cannot dispute it.
2. **It is dimensionally commensurate with Levenshtein.** Levenshtein counts unit edit operations on
   a string. The like-for-like reference is a GED that counts unit edit operations on a graph. A
   zero-node-cost GED silently reweights the comparison along exactly the size axis we stratify by.
3. **It is the community convention.** Riesen & Bunke's IAM benchmark (SSPR 2008) uses it, eight of
   our ten datasets are IAM, and it is GEDLIB's default — so BP, BRANCH and BRANCH-FAST are all
   specified against it in their source papers.
4. **It restores a single operation alphabet.** See below.

### The same decision fixes the message-length experiment

The information-content experiment defines
`B_GED(G) = (N − 1 + M) + 2M⌈log₂ N⌉` bits (`computational_experiments.tex:162–176`). The term
`N − 1 + M` is a **count of edit operations** — node insertions plus edge insertions. So the
message-length reference and the GED reference are already built on the same operation set; the
submission just never says so, and calls the result "standard" without support (R3.6a).

**Unify them explicitly:**

> One edit-operation alphabet — {node insert/delete, edge insert/delete}, unit cost — underlies
> both experiments. **Claim B** measures the *number* of operations separating two graphs (GED) and
> asks whether Levenshtein tracks it. **Claim A** measures the *bits* needed to transmit the
> operations that build one graph from empty, and asks whether IsalGraph is shorter.

This is worth doing for its own sake, not just to satisfy a reviewer. It turns two loosely related
experiments into two measurements of one object, replaces the unsupported word "standard" with a
derivation, and answers R3.6a and R3.5b with a single paragraph.

Published GraphEdX values will no longer match ours — expected, and stated in the text.

---

## 3. Claim A — information content

Unit = graph. No dependence problem.

| Question | Procedure |
|---|---|
| Is IsalGraph shorter than competitor X on dataset D? | **Wilcoxon signed-rank** on paired per-graph bit counts |
| By how much? | **median per-graph difference + bootstrap CI**; matched-pairs rank-biserial correlation |
| "shorter for 98.8 %–99.6 % of graphs" (`results.tex:11`, currently **no CI**) | proportion + **Clopper–Pearson** binomial CI |
| Which method wins overall? | **Friedman + Wilcoxon–Holm + CD diagram** over datasets (D8) |

Never report a mean bit count without dispersion: length distributions are right-skewed
(Mutagenicity median n = 27, max n = 417).

---

## 4. Claim B — metric locality

Unit = graph pair. Dyadically dependent.

**Point estimate** — Spearman ρ per (dataset × representation × GED reference), Kendall τ-b beside it.

**Uncertainty (D2)** — resample **graphs** with replacement, recompute ρ over the induced pair
submatrix, 2,000 replicates, percentile CI.

> Expect intervals to widen substantially. Effective sample size is governed by the number of
> **graphs**, not pairs: LINUX has **89 graphs**, not 3,916 independent observations. Some
> currently "significant" statements will weaken. **That is the correct outcome and we report it.**

**Significance (D3)** — Mantel test, 9,999 joint row/column permutations. `mantel_test` already
exists in `correlation_metrics.py` and has never been reported (E10).

*Known critique*: Mantel has been criticised for inflated type-I error under autocorrelation
(Guillot & Rousset, 2013). Our defence is D10 — the inference is carried by bootstrap CIs; Mantel
p-values accompany them.

**Comparing two representations on one dataset (D7)** — resample graphs, recompute **both**
correlations on the **same** resample, take the difference, percentile CI. Significant iff the CI
excludes 0.

> Explicitly **not** Hotelling–Williams or Steiger. Those are the textbook tools for dependent
> correlations sharing a variable, but they assume independent observations — which is exactly the
> error R3.5c identified. Using them would repeat it in a more sophisticated form.

**Across datasets (D8)** — Friedman omnibus on per-dataset ranks, pairwise Wilcoxon signed-rank with
Holm correction, presented as a critical-difference diagram. Demšar, *JMLR* 7:1–30, 2006. Two
separate diagrams — one for the exact-GED regime, one for the approximate — because mixing
references in one omnibus is indefensible.

---

## 5. D4 — the confound nobody asked about

Both Levenshtein distance and GED grow with graph size. A reviewer can ask whether the reported
correlation is structural agreement or merely size agreement. **We must have the answer before
they ask.**

**MRM (multiple regression on distance matrices), permutation inference:**

```
GED_ij  ~  β₁·Lev_ij  +  β₂·|n_i − n_j|  +  β₃·|density_i − density_j|
```

Report the standardised partial coefficient β₁ with a permutation CI. Also report the simple
**partial Mantel** of Lev and GED controlling for `|n_i − n_j|`, which is the same idea in the form
reviewers will recognise.

**Interpretation, fixed in advance:**
- β₁ remains large → the association is structural. Claim B stands as stated.
- β₁ collapses → the correlation was largely size agreement, and **Claim B must be restated**.

**Run this in the first week.** It can refute the paper's central result, and we need time to
absorb that if it does.

---

## 6. Calibration — the gate on everything above n = 12

`data.md` §5: exact GED is unobtainable above ≈ 12 nodes, so the extension datasets use bounds.
That is only legitimate if the bounds are calibrated **where exact GED exists**.

| Quantity | Procedure |
|---|---|
| ρ(exact, LB), ρ(exact, UB) | Mantel + graph-level bootstrap CI |
| **Agreement**, not just correlation | **Bland–Altman**: bias and 95 % limits of agreement |
| Bound validity | violation count — **measured: 0/400 both bounds** |
| Certification rate | fraction with LB = UB — **measured: 9.8–11.3 %** |
| **The gate** | ρ(Lev, exact) − ρ(Lev, approx) **on the same pairs**, bootstrap CI |

**Pre-declared decision rule** (fix now, not after seeing the number):

> If the 95 % CI for `ρ(Lev, exact) − ρ(Lev, approx)` excludes 0 **and** the point estimate exceeds
> 0.05 in absolute value, the approximation is not a validated stand-in. In that case the exact-GED
> results are primary and the large-n extension is reported as **descriptive only**, with the
> discrepancy stated in the text.

**Already measured** (`data.md` §5): ρ(exact, BRANCH-FAST lower) = **0.966**;
ρ(exact, BP upper) = **0.840**. **The lower bound is the better reference** and is designated
primary. Both are reported as a bracket.

---

## 7. Stratification

**Terminology**: "arity" belongs to hypergraphs (IsalHG). For simple graphs the variables are:

| Variable | Definition | Bins |
|---|---|---|
| **Node count** | `n` | 3–5, 6–9, 10–12, 13–20, 21–40, > 40 |
| **Density** | `2m / (n(n−1))` | quintiles, pooled across datasets |
| **Mean degree** | `2m / n` | quartiles |
| **Symmetry** — *new, driven by `data.md` H8* | orbit count / \|Aut(G)\| from nauty | quartiles |

The symmetry variable is the discovery of §H8: canonicalisation cost tracks structural symmetry,
not size or density (Protein n = 96 → 1.1 s; Mutagenicity n = 98 → > 5 min, at the same density).
nauty is already being vendored as a competitor backend, so the orbit count is free. **No reviewer
asked for this**, and it converts the scalability limitation from an apology into a characterisation.

Procedure: within-stratum ρ with graph-level bootstrap CI; pool across datasets so strata contain
structurally comparable graphs regardless of provenance. Formal monotone-trend testing via
**Jonckheere–Terpstra** only if a trend is claimed. With ~5 strata, correlating stratum-level ρ
against stratum density is **descriptive** — labelled as such.

**Stratified analyses are exploratory** and are excluded from the D9 confirmatory family.

**This is where the AIDS question is settled.** If ρ recovers on sparse AIDS strata, the density
explanation at `conclusion.tex:30–36` is supported; if not, that passage is wrong and gets rewritten.

---

## 8. Mandatory reporting

The manuscript's entire description of its bootstrap is one parenthesis (`results.tex:175–176`:
*"bootstrap 95 % CIs overlap substantially"*). Every item below appears in the revision:

- resampling **unit** (graph), replicate count (2,000), CI method (percentile), seed (42);
- permutation count (9,999) and what is permuted (graph labels, jointly on rows and columns);
- **the pair-accounting ladder, per dataset**:
  `raw → connected → GED-available → GED > 0 → Lev > 0 → analysed`
  with the **connectivity-retention** column (`data.md` §2.2 measures 51.4 %–100 %, never reported);
- **which numbers are exact GED and which are bounds**, on every table row;
- **encoding timeout rate per stratum** (D12), with the timeout value used;
- software and library versions, including GEDLIB.

---

## 9. What we drop, and why

| Dropped | Reason |
|---|---|
| Asymptotic Spearman test on pair counts (`computational_experiments.tex:208–209`) | the defect R3.5c identified |
| Pair-level bootstrap (`correlation_metrics.py::bootstrap_correlation`) | wrong resampling unit — replaced, not supplemented |
| Pooled OLS β as a headline (`conclusion.tex:38–41`) | R3.5b |
| Hotelling–Williams / Steiger | assume independence (§4) |
| Bonferroni | too conservative at this family size |
| Significance as a stand-in for effect size (`conclusion.tex:37`) | R3.6b |

---

## 10. Confirmatory / exploratory split — DECIDED (author, 2026-08-11)

**Confirmatory family** (BH-FDR at q = 0.05 applies to exactly these):

| Claim | Comparison | Unit |
|---|---|---|
| A | IsalGraph (pruned canonical) vs **each** competitor serialisation, **per dataset**, on bits per graph | graph |
| A | Friedman omnibus + Wilcoxon–Holm across datasets | dataset |
| B | ρ(Lev-on-IsalGraph, GED) vs ρ(competitor distance, GED), **per dataset** | graph pair |
| B | Friedman omnibus + Wilcoxon–Holm across datasets, exact and approximate regimes **separately** | dataset |
| B | MRM partial coefficient β₁ for Lev, controlling for \|Δn\| and \|Δdensity\| (D4) | graph pair |
| Cal. | ρ(Lev, exact) − ρ(Lev, approx) on shared pairs — the §6 gate | graph pair |

**Exploratory** (reported with CIs, labelled as such, **excluded** from FDR):
all stratified analyses (node count, density, degree, symmetry); per-stratum timeout and censoring
rates; the pruned-vs-exhaustive encoding comparison; encode-time regressions.

S2 (cost model) is resolved by D6 above.

---

## 11. Change log

| Date | Ver | Change |
|---|---|---|
| 2026-08-11 | v1.0 | First draft, presented as options |
| 2026-08-11 | v2.0 | **Converted to decisions D1–D12.** Bootstrap replicates 10,000 → 2,000 (67 M pairs). Spearman kept primary, Kendall demoted to robustness check. MRM promoted to confirmatory. Symmetry added as a stratification variable (`data.md` H8). Cost model fixed. Open questions reduced to two |
