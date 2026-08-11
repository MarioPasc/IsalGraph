# Statistical protocol — DECIDED

**Status**: **v2.1, 2026-08-11.** **These are decisions, not options.** Fixed before T-06 so nothing
is computed twice. Deviations require a changelog entry.

Answers R3.5a, R3.5b, R3.5c and AE.4.

> **v2.1** adds **D13–D15**, closing `gap-audit.md` MF2–MF5 and MF16: the resampling budget was
> under-estimated by 10²–10³× (§4.1), the calibration did not reach the regime it licenses (§6.1),
> the large-`n` decision rule had no threshold (§6.2), and encoding-censored graphs had no analysis
> rule (§6.3).

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
| **D13** | **The large-`n` bracket-agreement rule has a pre-declared threshold** (§6.2) | "if they agree" — undefined | `gap-audit.md` MF3 |
| **D14** | **Encoding-censored graphs are analysed, not dropped**: greedy-min fallback arm + a complete-case sensitivity arm (§6.3) | silent exclusion | `gap-audit.md` MF4 |
| **D15** | **Resampling effort is scaled per dataset under a written compute budget** (§4.1) | 2,000 × 9,999 everywhere, unbudgeted | `gap-audit.md` MF5 |

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
Holm correction, presented as a critical-difference diagram. Demšar, *JMLR* 7:1–30, 2006. The exact
and approximate regimes are never mixed in one omnibus.

> **Amended 2026-08-11 (`gap-audit.md` MF16).** The exact-GED regime has **five** datasets. The
> Friedman statistic is conservative at `N = 5` and the critical difference is wide enough to
> separate almost nothing, so a CD diagram there would be an underpowered figure dressed as a result.
> **Locked: the omnibus and the CD diagram are reported for the ten-dataset approximate regime only.
> The exact regime is reported descriptively** — per-dataset ρ with graph-level bootstrap CIs and the
> D7 paired differences — **and the reason is stated in the text.**

`jonckheere_terpstra` and `holm_bonferroni` already exist in
`benchmarks/real_data/eval_correlation/correlation_metrics.py` alongside `mantel_test` and
`bootstrap_correlation`; only the resampling **unit** has to change, not the machinery.

### 4.1 D15 — resampling effort and its budget

> **In plain terms.** D2 says: to get a confidence interval on ρ, resample the *graphs* 2,000 times
> and recompute ρ from scratch each time. On COIL-DEL one recomputation touches **25.9 M pairs**, so
> 2,000 of them is 5×10¹⁰ operations — and we do that once per (dataset × competitor × bracket end)
> cell, of which there are about 120. The plan budgeted 4–8 core-hours for all of it. It is closer to
> 40–80.
>
> **The fix is arithmetic, not statistical.** For the two giant datasets, use 1,000 replicates
> instead of 2,000 and, *inside* each replicate, compute ρ on a random 2 M of the induced pairs
> rather than all 25.9 M. **The resampling unit is still the graph** — that is the entire point of
> R3.5c and it does not change. We are only thinning the arithmetic within a replicate, and we
> validate on IAM Letter HIGH that the thinning does not move the CI.

`plan.md` §2 and `data.md` §7 budget the bootstrap and Mantel work at 4–10 core-hours. The arithmetic
implied by D2 (2,000 replicates) and D3 (9,999 permutations) is two to three orders of magnitude
larger:

| Dataset | graphs | pairs | 2,000 × pairs | 9,999 × pairs |
|---|---:|---:|---:|---:|
| COIL-DEL | 7,200 | 25,916,400 | 5.2 × 10¹⁰ | 2.6 × 10¹¹ |
| Mutagenicity | 4,040 | 8,158,780 | 1.6 × 10¹⁰ | 8.2 × 10¹⁰ |
| IAM Letter HIGH | 2,059 | 2,118,711 | 4.2 × 10⁹ | 2.1 × 10¹⁰ |

and that is **one cell**. The §10 confirmatory family is IsalGraph versus each competitor, per
dataset, over two bracket ends — roughly **120 cells** for Claim B, each also carrying a D7
difference bootstrap on the same resamples. Spearman requires **re-ranking inside every replicate**,
so the per-replicate cost is `O(p log p)`; ranks computed once on the full matrix cannot be reused.

**Locked policy.** Effort is a function of dataset size, fixed in advance and reported:

| Pairs in the dataset | Bootstrap replicates | Mantel permutations | Within-replicate pairs |
|---|---:|---:|---|
| ≤ 10⁶ | 2,000 | 9,999 | all |
| 10⁶ – 5 × 10⁶ | 2,000 | 4,999 | all |
| > 5 × 10⁶ | **1,000** | **1,999** | **uniform subsample of 2 × 10⁶ induced pairs, seed 42** |

Three rules that keep this honest:

1. **The resampling unit is unchanged.** Graphs are always resampled with replacement; the
   subsampling is applied to the *induced pairs within a replicate* and never to the graph list.
   D2's answer to R3.5c is untouched.
2. **The subsample is validated, not assumed.** On IAM Letter HIGH (2.1 M pairs) the full and
   subsampled protocols are both run and the CIs compared. If they differ materially the tier is
   revised, and either way the comparison is reported.
3. **Every table states its replicate count, permutation count and subsample size.** A CI computed
   from 1,000 replicates is not silently presented beside one from 2,000.

**Budget under this policy**: ≈ 40–80 core-hours, ~1 h on 64 cores. Fifty times the current estimate
and still negligible beside T-03 — the point is that it is now *written down* instead of discovered
in week three.

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

### 6.1 The calibration must reach into the regime it licenses

**The defect** (`gap-audit.md` MF2). The gate above is computed where exact GED exists — `n ≤ 12` —
and the bounds in `data.md` §5 were validated on **`n = 3–9` only**. The regime being licensed runs
to **`n = 98`**, with Suite-2 means from 11.45 to 31.68. So the calibration sample tops out three to
ten times below the population it certifies, and the plan's own warning applies to it:

> Bipartite GED's error is known to grow with graph size, so a declining ρ at large n would be
> uninterpretable. (`plan.md` §3.3)

Bracket **validity** is not at risk — `LB ≤ GED ≤ UB` is proven at every `n`. **Tightness** is, and
tightness is what the argument rests on. Three additions, all cheap:

1. **A size-stratified exact ladder.** Run `ANCHOR_AWARE_GED` on a stratified sample at each `n` from
   3 up to the feasible ceiling, with a fixed per-pair time budget and interval censoring (D11)
   above it. `plan.md` open question 16 already proposes benchmarking `ANCHOR_AWARE_GED` against
   `networkx` A*; this makes it **part of the design** rather than an opportunistic extra. Every node
   the exact solver buys widens the calibration and directly strengthens AE.1.
2. **Regress, do not assume transfer.** Fit relative bracket width `(UB − LB)/UB` and the ρ-gap
   `ρ(Lev, exact) − ρ(Lev, approx)` on `n` over the ladder, and report the extrapolation to the
   Suite-2 range **with its uncertainty**.
3. **Report `(UB − LB)/UB` as a function of `n` across all of Suite 2.** This needs no exact GED, is
   computable on all 40 M pairs, and is the strongest available evidence that the reference does not
   degrade with size. It is also the single measurement that answers AE.1 most directly: it separates
   "IsalGraph degrades at scale" from "our reference degrades at scale".

### 6.2 D13 — the large-`n` decision rule, pre-declared

> **In plain terms.** Above `n = 12` we do not know the true GED — only that it lies somewhere
> between our lower and upper bounds. So we compute the correlation **twice**: once treating GED as
> the lower bound, once as the upper bound. If both give roughly the same ρ, it does not matter where
> the truth sits inside the bracket and the conclusion holds. If they give very different ρ, the
> bracket is too wide to conclude anything at that size.
>
> The rule below just says how different is *too* different, and fixes it before we look.
>
> | Example | ρ(Lev, LB) | ρ(Lev, UB) | difference, 95 % CI | verdict |
> |---|---:|---:|---|---|
> | concordant | 0.61 | 0.58 | 0.03, [−0.01, 0.07] | report ρ ≈ 0.6; state that the conclusion is **robust to the bracket's interior** |
> | uninformative | 0.55 | 0.31 | 0.24, [0.19, 0.29] | report "ρ lies between 0.31 and 0.55", **descriptively**, and exclude the dataset from the confirmatory family |
>
> Without a pre-declared threshold, "agree" gets decided after seeing the numbers — on the rule that
> governs **every result above n = 12**.

`plan.md` §7.3 correlates Levenshtein against both bracket ends and concludes:

> If ρ(Lev, LB) and ρ(Lev, UB) **agree**, the conclusion is robust to wherever the true GED lies in
> the bracket … If they **disagree**, the bracket is too wide to support a claim at that size.

"Agree" was undefined. Since this rule governs **every number above `n = 12`**, choosing the
threshold after seeing the estimates is exactly what §6's pre-declaration exists to prevent.

> **Locked, symmetric with the §6 gate.** Per dataset, bootstrap the difference
> `ρ(Lev, LB) − ρ(Lev, UB)` on the **same** graph-level resamples (D7). The bracket is declared
> **uninformative** at that dataset if the 95 % CI excludes 0 **and** the point estimate exceeds
> **0.05** in absolute value. Otherwise the two ends are reported as concordant and the conclusion is
> stated as robust to the bracket's interior.
>
> Where the bracket is uninformative, the dataset's ρ is reported **as an interval, descriptively**,
> and is **excluded from the D9 confirmatory family** — it is not quietly reported as a point
> estimate.

### 6.3 D14 — encoding-censored graphs are analysed, not dropped

> **In plain terms.** A few large, highly symmetric graphs will not finish canonicalisation inside
> the 300 s timeout, so they end up with **no canonical string**. The obvious move is to drop them.
> The problem is *which* graphs get dropped: `data.md` §4.4 shows the failures are exactly the ones
> with a huge automorphism group (`|Aut| > 20,000`), so dropping them removes the hardest cases and
> the paper then reports "IsalGraph handles n̄ ≈ 30" on a sample the hard cases were quietly deleted
> from. That is the same silent selection bias as the connectivity discard, which `plan.md` open
> question 15 already treats carefully.
>
> So instead: give those graphs their **greedy-min** string, which always completes (25.7 ms at
> `n = 96`), flag every affected pair, and **also** report the analysis with them removed. Then:
>
> - the two numbers agree → censoring did not matter, and we have shown it rather than assumed it;
> - the two numbers differ → **that difference is the measurement of the bias**, and it gets reported
>   as such.
>
> "Primary arm" = the number that goes in the results table. "Sensitivity arm" = the number reported
> beside it that tests whether the first one was an artefact.

`data.md` §4.3 measured it: pruned canonicalisation on Mutagenicity graph 3703 (`n = 98`) **did not
finish in four minutes**, and §4.4 identifies `|Aut(G)| > 20,000` as the mechanism. With the locked
300 s timeout (`plan.md` open question 13), **some Suite-2 graphs will have no canonical string.**

D12 requires reporting the censoring rate. It never said what the analysis *does* with a censored
graph. The default — drop it — deletes that graph and every pair containing it, and it deletes
preferentially the **high-`|Aut|` graphs**, which is precisely the population the scalability claim is
about. This is the same selection-bias structure that `plan.md` open question 15 flags for the
connectivity discard, and it was unflagged.

> **Locked.** A graph whose canonical encoding is censored is **not** removed from the corpus.
>
> - **Primary arm** — the censored graph enters with its **greedy-min** string, which is always
>   available (`data.md` §4.3: 25.7 ms at `n = 96`), and every affected pair is **flagged** in the
>   output.
> - **Sensitivity arm** — complete-case analysis over uncensored graphs only. Both ρ values are
>   reported; a material gap between them *is* the selection-bias measurement.
> - **Reporting** — censoring rate **per symmetry stratum** (D12), and the retained-versus-censored
>   `n̄`, density and orbit-count comparison, in the same form as the connectivity-discard table that
>   open question 15 requires.
>
> Rationale: the greedy-min substitution is a *stated degradation of the representation*, not a
> missing observation, and it is exactly the fallback a practitioner would use. Reporting both arms
> converts an exclusion into a characterisation — the same move `data.md` H8 makes for the
> scalability limit.

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
| **L** | **ρ(Lev, GED_topo) − ρ(Lev, GED_lab), per labeled dataset** — the label-blindness cost (`labels.md` B3) | graph pair |

**The family must be enumerated and counted before any p-value is computed.** With 6 competitor
representations, 10 datasets and 2 bracket ends, Claim B alone contributes ~120 comparisons; BH-FDR
at `q = 0.05` behaves very differently over 20 tests than over 200, and the count is not something to
discover afterwards. **Write the explicit list into the pre-registration section of T-02, with its
cardinality, and freeze it before T-06 runs.**

**Exploratory** (reported with CIs, labelled as such, **excluded** from FDR):
all stratified analyses (node count, density, degree, symmetry); per-stratum timeout and censoring
rates; the pruned-vs-exhaustive encoding comparison; encode-time regressions; the D14 complete-case
sensitivity arm; the `labels.md` §5 dataset-level regression (`N = 10`); the per-dataset GEDLIB
cost-model sensitivity arms.

**Also excluded**, by D13: any dataset whose bracket is declared uninformative at §6.2's threshold.
Its ρ is reported as a descriptive interval and contributes no test.

S2 (cost model) is resolved by D6 above.

---

## 11. Change log

| Date | Ver | Change |
|---|---|---|
| 2026-08-11 | v1.0 | First draft, presented as options |
| 2026-08-11 | v2.0 | **Converted to decisions D1–D12.** Bootstrap replicates 10,000 → 2,000 (67 M pairs). Spearman kept primary, Kendall demoted to robustness check. MRM promoted to confirmatory. Symmetry added as a stratification variable (`data.md` H8). Cost model fixed. Open questions reduced to two |
| 2026-08-11 | **v2.1** | **D13–D15 added**, closing `gap-audit.md` MF2–MF5 and MF16. §4.1 — resampling effort tiered by dataset size, with the 10²–10³× compute under-estimate corrected and written down. §6.1 — size-stratified exact calibration ladder, because the calibration regime (`n ≤ 12`, bounds validated at `n = 3–9`) did not reach the inference regime (`n ≤ 98`). §6.2 — the large-`n` bracket-agreement rule gains a pre-declared threshold. §6.3 — encoding-censored graphs are analysed with a greedy-min fallback plus a complete-case sensitivity arm, instead of being dropped and silently biasing against high-\|Aut\| graphs. §4 — Friedman/CD restricted to the ten-dataset approximate regime; five datasets is underpowered. §10 — confirmatory family must be enumerated and counted; label rows added |
