# Statistical protocol — DECIDED

**Owner**: T-02 (lock and pre-registration), T-06 (execution) · **Serves**: R3.5a, R3.5b, R3.5c, AE.4c
**Status**: LOCKED. **These are decisions, not options.** Fixed before T-06 so nothing is computed
twice. Deviations require a changelog entry.
**T-02 CLOSED 2026-08-13** — the confirmatory family is enumerated and frozen in
[preregistration](preregistration.md) (`N_max = 197`). See §12 RESULT.

Related: [preregistration](preregistration.md) (**the frozen family — authoritative**) ·
[exact_ged](exact_ged.md) · [approx_ged](approx_ged.md) · [data](data.md) ·
[labels](labels.md) · [corrections](corrections.md)

---

## 1. The story, in one paragraph

The paper makes two claims and they need different statistics because **their units of analysis
differ**.

> **Claim A (information content).** IsalGraph encodes a graph in fewer bits than competing
> reversible serialisations. Unit = **one graph**. Observations are independent. Ordinary paired
> non-parametric statistics apply.
>
> **Claim B (metric locality).** Levenshtein distance on IsalGraph strings tracks graph edit distance
> better than competing representations' distances do. Unit = **one graph pair**. Observations are
> **dyadically dependent** — `d(G₁,G₂)` and `d(G₁,G₃)` share `G₁`. Ordinary statistics do **not**
> apply, and the submitted version used them.

Everything below follows from that distinction. Stating it explicitly in the paper is itself part of
the answer to R3.5c.

---

## 2. The decisions

| # | Decision | Replaces | Driver |
|---|---|---|---|
| **D1** | **Spearman ρ is the primary association measure**; Kendall τ-b as a tie-robustness check | — | continuity: reviewers quote ρ = 0.433 / 0.349 |
| **D2** | **All uncertainty comes from a graph-level cluster bootstrap**, 2,000 replicates, percentile CI, seed 42 | pair-level `bootstrap_correlation` | **R3.5c** |
| **D3** | **All significance comes from the Mantel permutation test**, 9,999 permutations of graph labels | asymptotic Spearman test on pair counts | **R3.5c** |
| **D4** | **MRM is a confirmatory analysis**: `GED ~ Lev + \|Δn\| + \|Δdensity\|`, permutation inference | nothing — new | pre-empts the size-confound attack |
| **D5** | **Per-dataset results are primary; pooled results are secondary** and never a headline | pooled OLS β at `conclusion.tex:38–41` | **R3.5b** |
| **D6** | **One GED cost model across all datasets**: node ins/del = 1, edge ins/del = 1, substitutions free | IAM uniform vs GraphEdX topology-only | **R3.5b** |
| **D7** | **Method comparison within a dataset** = bootstrap CI on the difference of ρ | — | dependence |
| **D8** | **Method comparison across datasets** = Friedman + Wilcoxon–Holm + critical-difference diagram | — | Demšar, *JMLR* 7:1–30, 2006 |
| **D9** | **Multiplicity** = Benjamini–Hochberg FDR at q = 0.05 over a **pre-declared** confirmatory family | nothing | scale of the design |
| **D10** | **Effect sizes with CIs lead. p-values are supporting detail.** | significance-as-effect-size at `conclusion.tex:37` | **R3.6b** |
| **D11** | **Non-computable exact GED is interval-censored `[LB, UB]`, not missing** | silent exclusion | **R3.5a** |
| **D12** | **Censoring and timeout rates are reported per stratum, never pooled** | nothing | censoring is symmetry-correlated |
| **D13** | **The large-`n` bracket-agreement rule has a pre-declared threshold** ([approx_ged](approx_ged.md) §4) | "if they agree" — undefined | audit MF3 |
| **D14** | **Encoding-censored graphs are analysed, not dropped**: greedy-min fallback + complete-case sensitivity arm | silent exclusion | audit MF4 |
| **D15** | **Resampling effort is scaled per dataset under a written compute budget** (§5) | 2,000 × 9,999 everywhere, unbudgeted | audit MF5 |

### D6 explained — one cost model, for *both* experiments

> ## ➕ ADDED 2026-08-13 (T-27) — D6 carries an unstated precondition that two methods depend on
>
> D6 is stated as a choice about comparability. It is also a **validity precondition** for part of
> the bound machinery, and that was nowhere written down:
>
> - **`STAR` is only a proven lower bound under *uniform* edit costs** (Zeng et al., PVLDB
>   2(1):25–36, Lemma 4.2). We satisfy this **only because our graphs are effectively unlabeled** —
>   node and edge attributes are constant dummies, and IAM Letter's `labels` array is a graph-level
>   class label, not a node label. **If a labelled variant is ever run, `STAR`'s validity is not
>   guaranteed and must be re-derived**, not assumed. This bears directly on [labels](labels.md)
>   Tier 2: promoting labels changes which GED bounds remain proven.
> - **`BRANCH` and `BRANCH_FAST` are *equivalent*, not merely ordered, under constant edge edit
>   costs** (survey §5.2.4) — which D6 has. T-27 measured them identical on all 3,836,827 certified
>   pairs. So D6 does not merely make the two comparable; it **collapses them into one method**, and
>   the choice between them is a cost decision, not an empirical one.
> - **A zero lower bound is legitimate under D6.** Free node *and* edge substitution means any
>   degree-preserving assignment costs nothing, so two non-isomorphic graphs with identical degree
>   sequences get `LB = 0`. Verified: C₆ vs two disjoint triangles has exact GED **4.0** and all four
>   LB methods return **0.00**, all valid. **Do not treat `LB == 0` as evidence of a defect** — it is
>   ~1 % of certified LINUX pairs and far more on Letter, where n̄ = 4.7 makes degree collisions common.

> ## ⚠ CORRECTED 2026-08-13 — the premise below is wrong; the decision survives
>
> This paragraph said LINUX and AIDS charge **zero for node operations**. **They do not.** T-03
> measured GraphEdX's published AIDS values against both models: **unit-node 4/4, zero-node 0/4**,
> with the published value exceeding the zero-node value by exactly `|n₁ − n₂|` every time.
> GraphEdX ships **unit node costs — the same model as D6**.
>
> **So the cost-model heterogeneity described below may never have existed**, and the sentence
> "the same pair of graphs gets different GED values depending on which dataset it came from"
> **must not be printed until re-verified**. Anyone writing the R3.5b response should check the
> IAM Letter side too, rather than inherit it from here.
>
> **D6 itself stands**, on justification 1, which is an argument about zero node cost *in general*
> and does not depend on what GraphEdX shipped. Justifications 2–4 also stand. What weakens is
> only the *rhetorical* framing that the submission mixed two models.
>
> The remaining, measured reasons to recompute: GraphEdX publishes GED for **within-split pairs
> only** (44.4 % of AIDS pairs, 43.0 % of LINUX), and **IAM Letter ships no GED matrix at all**, so
> a single cost model across the cohort is unobtainable from the distributions as shipped.
> Record: `.claude/notes/review/tasks/T-03-design.md` amendment 4.

~~The submission uses **two**: IAM Letter charges node ins/del = 1, edge ins/del = 1; LINUX and AIDS
(from GraphEdX) charge **zero for node operations**. The same pair of graphs gets different GED values
depending on which dataset it came from, and Table 3 pools both onto one axis. **That is R3.5b.**~~

**Decision: unit cost — node insert/delete = 1, edge insert/delete = 1, substitutions free.** Four
justifications, in the order they should appear in the response letter:

1. **It keeps GED a metric.** With zero node cost, inserting an isolated vertex is free, so two
   non-isomorphic graphs can sit at distance 0 and GED is only a *pseudo*metric. Corollary 2.13
   asserts the IsalGraph graph distance **is** an isomorphism-invariant metric. Validating a metric
   against a pseudometric reference is incoherent. **Formal, short, and not disputable.**
2. **It is dimensionally commensurate with Levenshtein**, which counts unit edit operations on a
   string. A zero-node-cost GED silently reweights the comparison along exactly the size axis we
   stratify by.
3. **It is the community convention** — Riesen & Bunke's IAM benchmark uses it, eight of our ten
   datasets are IAM, and it is GEDLIB's default, so BP, BRANCH and BRANCH-FAST are all specified
   against it in their source papers.
4. **It restores a single operation alphabet**, which also fixes the message-length experiment:
   `B_GED(G) = (N − 1 + M) + 2M⌈log₂ N⌉` already counts node plus edge insertions. The submission
   never says so and calls the result "standard" without support (R3.6a).

> **One edit-operation alphabet** — {node insert/delete, edge insert/delete}, unit cost — underlies
> both experiments. **Claim B** measures the *number* of operations separating two graphs and asks
> whether Levenshtein tracks it. **Claim A** measures the *bits* needed to transmit the operations
> that build one graph from empty, and asks whether IsalGraph is shorter.

⚠ **Corollary 2.13 is D6's lead justification, so auditing it is a prerequisite for the argument
that justifies T-03.** Owner **T-22**, see [corrections](corrections.md).

---

## 3. Claim A — information content

Unit = graph. No dependence problem.

| Question | Procedure |
|---|---|
| Is IsalGraph shorter than competitor X on dataset D? | **Wilcoxon signed-rank** on paired per-graph bit counts |
| By how much? | **median per-graph difference + bootstrap CI**; matched-pairs rank-biserial correlation |
| "shorter for 98.8–99.6 % of graphs" (`results.tex:11`, currently **no CI**) | proportion + **Clopper–Pearson** binomial CI |
| Which method wins overall? | **Friedman + Wilcoxon–Holm + CD diagram** over datasets (D8) |

Never report a mean bit count without dispersion: length distributions are right-skewed
(Mutagenicity median `n = 27`, **max `n = 98`** — the retained-set maximum; 417 is a raw-set value
and the 417-node graph is disconnected). At max 98 the skew is **3.6× the median**, not 15.4× — still
right-skewed, but do not lean on the tail as hard as the retired number allowed.

---

## 4. Claim B — metric locality

Unit = graph pair. Dyadically dependent.

**Point estimate** — Spearman ρ per (dataset × representation × GED reference), Kendall τ-b beside it.

**Uncertainty (D2)** — resample **graphs** with replacement, recompute ρ over the induced pair
submatrix, 2,000 replicates, percentile CI.

> Expect intervals to widen substantially. Effective sample size is governed by the number of
> **graphs**, not pairs: LINUX has **89 graphs**, not 3,916 independent observations. Some currently
> "significant" statements will weaken. **That is the correct outcome and we report it.**
>
> This is also why [exact_ged](exact_ged.md) §3 runs T-03 in two stages: if D2 is right, the AIDS
> all-pairs census buys coverage, not precision.

**Significance (D3)** — Mantel test, 9,999 joint row/column permutations. `mantel_test` already
exists in `correlation_metrics.py` and has never been reported (E10).
*Known critique*: Mantel has been criticised for inflated type-I error under autocorrelation
(Guillot & Rousset, 2013). Our defence is D10 — inference is carried by bootstrap CIs; Mantel
p-values accompany them.

**Two representations on one dataset (D7)** — resample graphs, recompute **both** correlations on the
**same** resample, take the difference, percentile CI. Significant iff the CI excludes 0.

> Explicitly **not** Hotelling–Williams or Steiger. Those are the textbook tools for dependent
> correlations sharing a variable, but they assume independent observations — exactly the error
> R3.5c identified. Using them would repeat it in a more sophisticated form.

**Across datasets (D8)** — Friedman omnibus on per-dataset ranks, pairwise Wilcoxon signed-rank with
Holm correction, presented as a critical-difference diagram. **The exact and approximate regimes are
never mixed in one omnibus.**

> **The exact regime has five datasets.** Friedman is conservative at `N = 5` and the critical
> difference is wide enough to separate almost nothing, so a CD diagram there would be an
> underpowered figure dressed as a result. **Locked: the omnibus and CD diagram are reported for the
> ten-dataset approximate regime only. The exact regime is reported descriptively** — per-dataset ρ
> with graph-level bootstrap CIs and D7 paired differences — **and the reason is stated in the text.**

`jonckheere_terpstra` and `holm_bonferroni` already exist alongside `mantel_test` and
`bootstrap_correlation` in `benchmarks/real_data/eval_correlation/correlation_metrics.py`; only the
resampling **unit** changes, not the machinery.

---

## 5. D15 — resampling effort and its budget

> **In plain terms.** D2 says: resample the *graphs* 2,000 times and recompute ρ from scratch each
> time. On COIL-DEL one recomputation touches **25.9 M pairs**, so 2,000 of them is 5 × 10¹⁰
> operations — once per (dataset × competitor × bracket end) cell, of which there are ~120. The
> original budget was 4–8 core-hours. It is closer to **40–80**.

Spearman requires **re-ranking inside every replicate**, so per-replicate cost is `O(p log p)`; ranks
computed once on the full matrix cannot be reused.

**Locked policy** — effort is a function of dataset size, fixed in advance and reported:

| Pairs in the dataset | Bootstrap replicates | Mantel permutations | Within-replicate pairs |
|---|---:|---:|---|
| ≤ 10⁶ | 2,000 | 9,999 | all |
| 10⁶ – 5 × 10⁶ | 2,000 | 4,999 | all |
| > 5 × 10⁶ | **1,000** | **1,999** | **uniform subsample of 2 × 10⁶ induced pairs, seed 42** |

### Frozen tier assignment — T-02, 2026-08-13

Applying the table to the locked pair counts in [data](data.md) §1. **This is the assignment T-06
runs; it is not recomputed at execution time.**

| Dataset | Suite-2 pairs | Tier | Replicates | Permutations | Within-replicate pairs |
|---|---:|---|---:|---:|---|
| LINUX | 3,916 | 1 | 2,000 | 9,999 | all |
| Protein | 161,596 | 1 | 2,000 | 9,999 | all |
| GREC | 210,925 | 1 | 2,000 | 9,999 | all |
| AIDS (GraphEdX) | 334,971 | 1 | 2,000 | 9,999 | all |
| IAM Letter LOW | 695,610 | 1 | 2,000 | 9,999 | all |
| IAM Letter MED | 784,378 | 1 | 2,000 | 9,999 | all |
| AIDS (IAM) | 1,638,955 | 2 | 2,000 | 4,999 | all |
| IAM Letter HIGH | 2,118,711 | 2 | 2,000 | 4,999 | all |
| **COIL-DEL** | **7,603,050** | **3** | 1,000 | 1,999 | **2 × 10⁶ (26.31 %)**, seed 42 |
| **Mutagenicity** | 8,158,780 | **3** | 1,000 | 1,999 | **2 × 10⁶ (24.51 %)**, seed 42 |

> **Updated 2026-08-13 (T-01).** COIL-DEL is **7,603,050** pairs, not 25,916,400 — the previous
> figure enumerated 7,200 files where the split index defines 3,900 ([data](data.md) §1.3). It stays
> in tier 3, and its subsample ratio rises from 7.72 % to **26.31 %**, so **both tier-3 datasets now
> sit in a narrow 24–27 % band** and one matched ratio validates both.
> ~~COIL-DEL 25,916,400 pairs, subsample 7.72 %~~

In the **exact** regime (Suite 1, `n ≤ 12`) only IAM Letter HIGH reaches tier 2; the other four
datasets are tier 1. No Suite-1 dataset is subsampled.

> ## 🔴 DEFECT FOUND 2026-08-24 (T-06) — under tier 3, an MRM point estimate falls OUTSIDE its own interval
>
> **Measured, first time tier 3 has ever been run.** Of 37 MRM fits, exactly **4** have a β₁ point
> estimate outside its own 95 % bootstrap interval, and they are precisely the two tier-3 datasets ×
> two bounds:
>
> | fit | β₁ | its own 95 % CI |
> |---|---:|---|
> | `mutagenicity`@lb | **+0.5229** | **[+0.0919, +0.1028]** |
> | `mutagenicity`@ub | +0.7303 | [+0.4284, +0.4805] |
> | `coil_del`@lb | +0.2494 | [+0.0646, +0.0726] |
> | `coil_del`@ub | +1.4892 | [+1.5122, +1.5618] |
>
> Separation by tier is exact — **tier 1: 28 consistent / 0 not; tier 2: 5 / 0; tier 3: 0 / 4.**
>
> **Cause.** Tier 3 is the only tier with a within-replicate pair budget. The **point estimate is
> fitted on every pair** (7.6 M / 8.16 M); **every replicate is fitted on a 2 × 10⁶ subsample.** They
> do not estimate the same quantity, so the interval cannot be expected to cover the point.
> `PercentileInterval.point` is documented as *"the full-sample estimate, never the bootstrap mean"* —
> correct in principle, and under tier-3 subsampling it **guarantees** the mismatch.
>
> **Rule 1 below is therefore necessary but not sufficient.** It keeps the *resampling unit* honest;
> it does not keep the point estimate and its interval on the same estimand.
>
> **Consequence for T-06:** the four fits are excluded from every D4 count. `mutagenicity`'s
> β_lev = +0.5229 was briefly reported as "Levenshtein dominates on the largest dataset" and is
> **retracted** — its own bootstrap puts β_lev near 0.10, reversing which predictor dominates.
> **The confirmatory family is unaffected**: B3e is Suite-1 only and no Suite-1 dataset is tier 3.
>
> **Owner: unassigned.** Either compute the point on the same subsample the replicates use, or label
> the interval as not covering it. **IsalSR and IsalHG inherit this** — it lay dormant only because
> no tier-3 dataset had been run before 2026-08-24.

Three rules that keep this honest:

1. **The resampling unit is unchanged.** Graphs are always resampled with replacement; subsampling
   applies to the *induced pairs within a replicate*, never to the graph list. D2's answer to R3.5c
   is untouched.
2. **The subsample is validated at the production *ratio*, not the production *count*.**

   > ## ⚠ CORRECTED 2026-08-13 (T-02) — the rule below was near-vacuous as written
   >
   > IAM Letter HIGH holds **2,118,711** pairs. Drawing the tier-3 subsample of 2 × 10⁶ from it is a
   > **94.4 % sample**, so the two protocols agree by construction and the comparison measures
   > nothing. The ratios the subsample actually runs at in production are **26.31 %** (COIL-DEL) and
   > **24.51 %** (Mutagenicity) — nowhere near what was being validated.
   >
   > **Replacement rule, frozen:**
   > - **Ratio-matched arm.** On IAM Letter HIGH, run the all-pairs protocol against the subsampled
   >   protocol at **519,296 pairs (24.51 %)** — the more aggressive of the two production fractions,
   >   which bounds the other — and compare the CIs. Reported either way.
   > - **Structure-matched arm.** Letter HIGH has `n̄ = 4.58`; COIL-DEL and Mutagenicity have
   >   `n̄ = 21.5` and `28.5`. A ratio matched on a structurally unlike dataset validates the
   >   *estimator*, not the *application*. So additionally run **one representative cell on
   >   Mutagenicity itself** at all pairs versus its 24.51 % subsample, at the tier-3 replicate count.
   > - **Branch, pre-declared**: if either arm's CI half-width differs by more than **10 % relative**,
   >   the tier boundary is revised upward and the revision is recorded in
   >   [preregistration](preregistration.md) §8. Either way the comparison is reported.

   ~~On IAM Letter HIGH (2.1 M pairs) both protocols run and the CIs are compared. If they differ
   materially the tier is revised; either way the comparison is reported.~~
3. **Every table states its replicate count, permutation count and subsample size.** A CI from 1,000
   replicates is not silently presented beside one from 2,000.

**Budget: ≈ 40–80 core-hours, ~1 h on 64 cores.** Fifty times the original estimate and still
negligible beside T-03 — the point is that it is now written down instead of discovered in week three.

---

## 6. D4 — the confound nobody asked about

Both Levenshtein and GED grow with graph size. A reviewer can ask whether the reported correlation is
structural agreement or merely size agreement. **We must have the answer before they ask.**

```
GED_ij  ~  β₁·Lev_ij  +  β₂·|n_i − n_j|  +  β₃·|density_i − density_j|
```

Report the standardised partial coefficient β₁ with a permutation CI, plus the simple **partial
Mantel** of Lev and GED controlling for `|n_i − n_j|` — the same idea in the form reviewers recognise.

**Interpretation, fixed in advance:**
- β₁ remains large → the association is structural. Claim B stands as stated.
- β₁ collapses → the correlation was largely size agreement, and **Claim B must be restated**.

**Run this in the first week.** It can refute the paper's central result and we need time to absorb
that if it does.

> D4 is self-labelled "asked for by nobody" **and is promoted to confirmatory**, so it joins D9's
> multiplicity family and can produce a headline. Cheap, but the promotion is scope the label does
> not cover — keep it visible.

---

## 7. D14 — encoding-censored graphs

> **In plain terms.** A few large, highly symmetric graphs will not finish canonicalisation inside
> the 300 s timeout. The obvious move is to drop them. The problem is *which* graphs get dropped:
> the failures are exactly the ones with a huge automorphism group (`|Aut| > 20,000`), so dropping
> them removes the hardest cases and the paper then reports "IsalGraph handles n̄ ≈ 30" on a sample
> the hard cases were quietly deleted from — the same silent selection bias as the connectivity
> discard.

**Locked.** A graph whose canonical encoding is censored is **not** removed from the corpus.

- **Primary arm** — the censored graph enters with its **greedy-min** string, which is always
  available (25.7 ms at `n = 96`), and every affected pair is **flagged** in the output.
- **Sensitivity arm** — complete-case analysis over uncensored graphs only. Both ρ values are
  reported; a material gap between them **is** the selection-bias measurement.
- **Reporting** — censoring rate **per symmetry stratum** (D12), plus the retained-versus-censored
  `n̄`, density and orbit-count comparison, in the same form as the connectivity-discard table.

The greedy-min substitution is a *stated degradation of the representation*, not a missing
observation, and it is exactly the fallback a practitioner would use. Reporting both arms converts an
exclusion into a characterisation.

> ## ⚠ QUALIFIED 2026-08-15 (T-05) — **"a few" understates it, and the 300 s timeout cannot be
> enforced the obvious way.** D14 itself survives intact and is *more* necessary than when written.
>
> D14's premise was set when every canonicalised graph in this project was `n ≤ 12`. T-05 measured
> `canonical_string` through the C++ engine on **10 uniformly random graphs per dataset, seed 42**,
> each in its own process killed at a **15 s** budget:
>
> | dataset | `n` range | canonical killed | canonical median | greedy-min killed | greedy-min median |
> |---|---|---:|---:|---:|---:|
> | `protein` | 11–48 | **5/10** | 0.899 s | 0/10 | 2.6 ms |
> | `coil_del` | 7–35 | **5/10** | 0.089 s | 0/10 | 3.1 ms |
> | `mutagenicity` | 10–37 | 1/10 | 0.317 s | 0/10 | 3.9 ms |
> | `grec` | 8–17 | 0/10 | 0.2 ms | 0/10 | 0.2 ms |
> | `aids_iam` / `aids_graphedx` | 7–12 | 0/10 | 0.2 ms | 0/10 | 0.2 ms |
>
> **15 s is not 300 s and these rates must not be extrapolated to it.** What they do establish: at
> Suite-2 sizes censoring is a **bulk property of two or three datasets**, not a marginal tail. So
> D14's greedy-min primary arm and its complete-case sensitivity arm will carry real weight, and the
> **censoring-rate table is a headline result of T-06, not a footnote**.
>
> **The cliff is not a node count.** COIL-DEL censors 5/10 by `n = 35` while Mutagenicity censors
> 1/10 by `n = 37`, and GREC is clean to `n = 17` with a 10.7 ms maximum. That is the direction §8
> already argues — canonicalisation cost tracks **structural symmetry**, not size or density — and it
> is why §8's symmetry stratum exists. Ten graphs per dataset cannot resolve the mechanism, only the
> fact that the ordering is not by `n`.
>
> ### 🔴 A Python signal-based timeout does NOT interrupt the C++ engine
>
> CPython runs signal handlers only *between bytecode instructions*, so `SIGALRM` stays queued for
> the entire duration of a native call. A first attempt using `signal.setitimer` **hung for 25
> minutes on a single graph** with the budget silently not applying. The table above was produced
> with a **killed subprocess**, which does work. Anyone implementing D14 against the C++ engine must
> do the same — and this failure presents as a hang, not as an error.
>
> **What survives**: all of D14. The fallback is sound and now measured at Suite-2 sizes rather than
> at `n ≤ 12` — greedy-min ran 0.2–3.9 ms with **0 kills** across all six datasets, four to five
> orders of magnitude under exhaustive canonicalisation.
>
> Data: `.claude/notes/review/tasks/T-05-canonicalisation-probe.json`. **Owner: T-06.**

---

## 8. Stratification

"Arity" belongs to hypergraphs (IsalHG). For simple graphs the variables are:

| Variable | Definition | Bins |
|---|---|---|
| **Node count** | `n` | 3–5, 6–9, 10–12, 13–20, 21–40, > 40 |
| **Density** | `2m / (n(n−1))` | quintiles, pooled across datasets |
| **Mean degree** | `2m / n` | quartiles |
| **Symmetry** — *new* | orbit count / `\|Aut(G)\|` from nauty | quartiles |

The symmetry variable comes from the finding that canonicalisation cost tracks structural symmetry,
not size or density (Protein `n = 96` → 1.1 s; Mutagenicity `n = 98` → > 5 min, at the same density).
nauty is already vendored as a competitor backend, so the orbit count is free. **No reviewer asked
for this**, and it converts the scalability limitation from an apology into a characterisation.

Procedure: within-stratum ρ with graph-level bootstrap CI; pool across datasets so strata contain
structurally comparable graphs regardless of provenance. Formal monotone-trend testing via
**Jonckheere–Terpstra** only if a trend is claimed. With ~5 strata, correlating stratum-level ρ
against stratum density is **descriptive** — labelled as such.

**Stratified analyses are exploratory** and excluded from the D9 confirmatory family.

> **This is where the AIDS question is settled.** R1.3 attributes the AIDS degradation to label loss;
> the rebuttal stands (the GraphEdX GED is itself topology-only, so both sides of the correlation are
> label-blind). But we also test the authors' *own* density claim: **stratify AIDS pairs by density
> and report ρ within strata.** If ρ recovers on sparse strata, `conclusion.tex:30–36` is supported;
> if not, **that passage is wrong and gets rewritten.** Run it early.

---

## 9. Confirmatory / exploratory split

**Confirmatory family** — **enumerated and FROZEN 2026-08-13 in
[preregistration](preregistration.md). That file is authoritative; the sketch below is its index.**

`N_max = 197`, in three fixed-sequence families (Dmitrienko, Tamhane & Bretz 2009), BH-FDR at
q = 0.05 **within** each:

| Family | Content | Tests |
|---|---|---:|
| **F0** | calibration gate — ρ(Lev, exact) − ρ(Lev, approx), per Suite-1 dataset | 5 |
| **F1** | bracket gate (**D13, promoted to confirmatory**) — ρ(Lev, LB) − ρ(Lev, UB), per Suite-2 dataset | 10 |
| **F2** | primary — A1 (60) · A2 (1) · B1e (35) · B1a (70) · B2 (1) · B3e (5) · B3a (10) | 182 |

> ## ⚠ CORRECTED 2026-08-24 (T-06) — the formula below omits TWO terms and under-counts `N_actual`
>
> The struck line is missing the `+ k·d` overlap term and the `c` term, both added to
> [preregistration](preregistration.md) §5/§5.1 on 2026-08-16. **Both omissions shrink the BH
> denominator, which lowers the correction on every surviving test** — the anti-conservative
> direction, and the one a reviewer pushes hardest on.
>
> **Correct form, and `preregistration.md` §5 is the authority:**
>
> ```
> N_actual(F2) = 182 − 15·k − 8·d + k·d − c        ordinary branch
> N_actual(F2) = 101 −  5·k          − c        when F0's majority branch fires (§5.3)
> ```
>
> **`N_actual` is defined by ENUMERATION**; the closed form is a printed check and the enumeration
> wins on disagreement. Cite `eval_stats/family.py:_closed_form` rather than restating the formula —
> **the code has carried the correct form throughout; five prose restatements did not**, and this was
> the last live one (T-06 design note §8.1).
>
> **As run by T-06, 2026-08-24:** F0's majority branch fired (4 of 5), so `k = 3`, `d` **not applied**,
> `c = 7`, **`N_actual = 79`**, enumeration and closed form agreeing at discrepancy 0.

~~`N_actual(F2) = 182 − 15k − 8d`, with `k` set by T-04a's F5-blind exclusion rule and `d` by F1.~~ BH is
computed over `N_actual`; `N_max`, the exclusion list and a BH-over-`N_max` sensitivity column are all
printed.

> ## ⚠ CORRECTED 2026-08-13 (T-02) — two rows above contradicted §4 and D13
>
> **1. The exact regime gets no omnibus.** §4 locks: *"the omnibus and CD diagram are reported for the
> ten-dataset approximate regime only. The exact regime is reported descriptively"*, because Friedman
> at `N = 5` separates almost nothing. The struck row below said "exact and approximate regimes
> **separately**", which would have put an underpowered omnibus into the confirmatory family. §4 wins;
> **F2 carries one omnibus per claim, on the ten-dataset approximate regime.**
>
> **2. The calibration gate and D13 are gates, not family members.** Both decide which downstream
> tests are admissible. Leaving them inside the family makes its cardinality a function of a test
> inside it. They are now **F0** and **F1**, prior and separate.
>
> **3. The L row is out.** S-d is open until 2026-08-18 and [labels](labels.md) Tier 2 is "logged, not
> written up". A conditional row makes the cardinality indeterminate today. If Tier 2 is promoted it
> enters as its **own** pre-declared family — see [preregistration](preregistration.md) §6.
>
> **4. ρ(Lev, UB) does not enter F2.** It is a near-duplicate of ρ(Lev, LB) on the same pairs, and BH
> behaves worst on near-duplicates. **The upper bound is reported in full** under
> [approx_ged](approx_ged.md) §4; its confirmatory role is F1. Reasoning:
> [preregistration](preregistration.md) §4.3.

~~| Claim | Comparison | Unit |~~
~~| A | IsalGraph vs **each** competitor serialisation, **per dataset**, on bits per graph | graph |~~
~~| A | Friedman omnibus + Wilcoxon–Holm across datasets | dataset |~~
~~| B | ρ(Lev-on-IsalGraph, GED) vs ρ(competitor distance, GED), **per dataset** | graph pair |~~
~~| B | Friedman omnibus + Wilcoxon–Holm across datasets, exact and approximate regimes **separately** | dataset |~~
~~| B | MRM partial coefficient β₁ (D4) | graph pair |~~
~~| Cal. | ρ(Lev, exact) − ρ(Lev, approx) on shared pairs — the calibration gate | graph pair |~~
~~| L | ρ(Lev, GED_topo) − ρ(Lev, GED_lab) per labeled dataset — [labels](labels.md) Tier 2 only | graph pair |~~

**Exploratory** — reported with CIs, labelled as such, **excluded** from FDR: all stratified analyses;
per-stratum timeout and censoring rates; the pruned-vs-exhaustive encoding comparison; encode-time
regressions; D14's complete-case arm; the dataset-level regression (`N = 10`); the per-dataset GEDLIB
cost-model sensitivity arms. **Also excluded** by D13: any dataset whose bracket is uninformative.

---

## 10. Mandatory reporting

The manuscript's entire description of its bootstrap is one parenthesis (`results.tex:175–176`).
Every item below appears in the revision:

- resampling **unit** (graph), replicate count, CI method (percentile), seed (42);
- permutation count and what is permuted (graph labels, jointly on rows and columns);
- **the pair-accounting ladder, per dataset**:
  `raw → connected → GED-available → GED > 0 → Lev > 0 → analysed`, with the
  **connectivity-retention** column ([data](data.md) measures 51.4 %–100 %, never reported);
- **which numbers are exact GED and which are bounds**, on every table row;
- **encoding timeout rate per stratum** (D12), with the timeout value used;
- software and library versions, including GEDLIB.

---

## 11. What we drop, and why

| Dropped | Reason |
|---|---|
| Asymptotic Spearman test on pair counts (`computational_experiments.tex:208–209`) | the defect R3.5c identified |
| Pair-level bootstrap (`correlation_metrics.py::bootstrap_correlation`) | wrong resampling unit — replaced, not supplemented |
| Pooled OLS β as a headline (`conclusion.tex:38–41`) | R3.5b |
| Hotelling–Williams / Steiger | assume independence |
| Bonferroni | too conservative at this family size |
| Significance as a stand-in for effect size (`conclusion.tex:37`) | R3.6b |

---

## 12. RESULT — T-02, closed 2026-08-13

**Deliverable**: [preregistration](preregistration.md), frozen before any p-value exists.

| Outcome | Value |
|---|---|
| Confirmatory family, total | **`N_max = 197`** across three fixed-sequence families |
| F0 calibration gate | 5 tests (one per Suite-1 dataset) |
| F1 bracket gate — **D13 promoted to confirmatory** | 10 tests (one per Suite-2 dataset) |
| F2 primary | 182 tests: A1 60 · A2 1 · B1e 35 · B1a 70 · B2 1 · B3e 5 · B3a 10 |
| Reduction rule | `N_actual(F2) = 182 − 15k − 8d`; BH over `N_actual`, `N_max` sensitivity printed |
| D15 tiers | assigned per dataset from the locked pair counts, §5 |
| Comparator sets | Claim A **6** serialisations; Claim B **7** (the six + WL kernel) |

**Pre-declared rules and which branch they take** — none has fired yet; all three determinations
(`k`, `d`, the primary bound at each end) are pre-declared parameters resolved by pre-declared rules,
recorded in [preregistration](preregistration.md) §7.

**Four corrections this ticket made**, all propagated in place: the exact-regime omnibus contradiction
(§9 vs §4), the two gates misfiled inside the family they gate, the conditional labels row, and
D15's near-vacuous subsample validation (§5 rule 2). Detail:
`.claude/notes/review/tasks/T-02-design.md`.

**Standing request answered.** §9 required "the explicit list … with its cardinality … frozen before
T-06 runs" and [decisions](decisions.md) §5 recorded it as the outstanding item under *Confirmatory
vs exploratory*. Both are now discharged.

**Debt carried, and it is not T-02's.** `k` needs **T-04a**; `d` needs T-06's own F1 run; the primary
bound at each end needs **T-27**. Article notes: `.claude/notes/review/tasks/T-02-article-notes.md`.
