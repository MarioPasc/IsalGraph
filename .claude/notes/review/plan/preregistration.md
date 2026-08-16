# Pre-registration — the frozen confirmatory family

**Owner**: T-02 (freeze), T-06 (execute) · **Serves**: R3.5a, R3.5c, AE.4c
**Status**: **FROZEN 2026-08-13.** Cardinality fixed before any p-value is computed.
**Rule**: a test not enumerated here is **exploratory** and is excluded from FDR, whatever it measures.

Related: [statistics](statistics.md) (the protocol D1–D15) · [approx_ged](approx_ged.md) §3–§4 (the two
gates) · [competitors](competitors.md) §3.4 (the exclusion rule) · [data](data.md) (the cohort) ·
[labels](labels.md) (the family that is deliberately *not* here)

---

## 0. Why this file exists separately

[statistics](statistics.md) §9 said: *"The family must be enumerated and counted before any p-value is
computed … write the explicit list into T-02's pre-registration section, with its cardinality, and
freeze it before T-06 runs."* It never was. This is that list.

BH-FDR at q = 0.05 behaves very differently over 20 tests than over 200. Discovering the count
afterwards makes the threshold a function of how much analysis happened to get run, which is the
multiplicity defect in a more sophisticated form than the one R3.5c already found.

---

## 1. Structure — three families in fixed sequence, not one flat list

Two of the protocol's pre-declared rules are **gates**: their outcome decides which downstream tests
are admissible. Putting a gate inside the family it gates makes the family's cardinality a function
of a test inside it, which is circular. They are therefore separated into a **fixed-sequence
gatekeeping design** (Dmitrienko, Tamhane & Bretz, *Multiple Testing Problems in Pharmaceutical
Statistics*, CRC Press, 2009, ch. 5), with BH-FDR (Benjamini & Hochberg, *JRSS-B* 57(1):289–300, 1995)
applied **within** each family at q = 0.05.

| Family | Content | Tests | Role |
|---|---|---:|---|
| **F0 — calibration gate** | [approx_ged](approx_ged.md) §3's rule, per Suite-1 dataset | **5** | decides whether the approximate regime may be **confirmatory at all** |
| **F1 — bracket gate (D13)** | [approx_ged](approx_ged.md) §4's rule, per Suite-2 dataset | **10** | decides **which** Suite-2 datasets enter F2 |
| **F2 — primary** | Claims A and B | **182** | the paper's headline comparisons |
| | | **197** | |

Each family is tested at q = 0.05 in its own right. F2 is entered only for the datasets F0 and F1
admit. **The sequence is fixed now and does not depend on any outcome.**

---

## 2. F0 — the calibration gate (5 tests)

Per Suite-1 dataset, on the pairs where both quantities exist:

```
ρ(Lev, GED_exact) − ρ(Lev, GED_approx)      graph-level bootstrap (D2), same resamples (D7)
```

| # | Dataset | Pairs (Suite 1) |
|---|---|---:|
| F0.1 | IAM Letter LOW | 695,610 |
| F0.2 | IAM Letter MED | 784,378 |
| F0.3 | IAM Letter HIGH | 2,118,711 |
| F0.4 | LINUX | 3,916 |
| F0.5 | AIDS (GraphEdX), `n ≤ 12` | 295,296 |

**Pre-declared branch** ([approx_ged](approx_ged.md) §3): the approximation is **not** a validated
stand-in at a dataset if its BH-adjusted CI excludes 0 **and** `|point estimate| > 0.05`. If it fails
on a **majority (≥ 3) of the five**, the exact-GED results become primary, F1 and F2's
approximate-regime rows are reported **descriptively only**, and N_actual drops accordingly.

---

## 3. F1 — the bracket gate, D13 promoted (10 tests)

Per Suite-2 dataset, on the same graph-level resamples (D7):

```
ρ(Lev, GED_LB) − ρ(Lev, GED_UB)
```

**D13 is confirmatory, not merely a reporting rule** (decision 25). The claim it registers is *the
conclusion is invariant to where inside the proven bracket the true value lies* — which is the
scientific content of reporting a bracket rather than a point estimate, and it deserves to be a
pre-registered claim rather than a footnote.

| # | Dataset | Pairs | # | Dataset | Pairs |
|---|---|---:|---|---|---:|
| F1.1 | IAM Letter LOW | 695,610 | F1.6 | GREC | 210,925 |
| F1.2 | IAM Letter MED | 784,378 | F1.7 | AIDS (IAM) | 1,638,955 |
| F1.3 | IAM Letter HIGH | 2,118,711 | F1.8 | COIL-DEL | **7,603,050** |
| F1.4 | LINUX | 3,916 | F1.9 | Mutagenicity | 8,158,780 |
| F1.5 | AIDS (GraphEdX), no `n_max` | 334,971 | F1.10 | Protein | 161,596 |

**Suite-2 total: 21,710,892 pairs over 16,370 graphs**, re-derived by T-01 on 2026-08-13. COIL-DEL was
recorded as 25,916,400; that figure enumerated 7,200 files where the split index defines 3,900
([data](data.md) §1.3). **The family cardinality is unaffected** — F1 has one test per dataset and the
dataset count did not change.

**Pre-declared branch**: a dataset's bracket is **uninformative** if its BH-adjusted CI excludes 0
**and** `|point estimate| > 0.05`. Its ρ is then reported as an interval, descriptively, and its
**8 F2 rows are removed** (7 × B1a + 1 × B3a).

---

## 4. F2 — the primary family (182 tests)

### 4.1 Comparator sets, fixed

| Set | Members | Size |
|---|---|---:|
| **Claim A serialisations** | graph6, sparse6, nauty-canonical graph6, adjacency matrix, AGM canonical code, gSpan minimum DFS code | **6** |
| **Claim B comparator distances** | the six above **+ WL subtree kernel distance** | **7** |

WL enters Claim B and not Claim A: it is **not reversible** and emits a feature vector, so it has no
bit count to compare, but it does yield a distance and [competitors](competitors.md) §6 puts it in
experiment (b). IsalGraph is the *reference arm* in every row and is never a comparator against
itself.

### 4.2 The enumeration

| Row | Test | Enumeration | Tests |
|---|---|---|---:|
| **A1** | Wilcoxon signed-rank, IsalGraph vs competitor, bits per graph, per dataset | 6 × 10 | **60** |
| **A2** | Friedman omnibus on bits, across the 10 datasets | 1 | **1** |
| **B1e** | bootstrap CI on ρ(Lev, exact) − ρ(comparator distance, exact), per Suite-1 dataset | 7 × 5 | **35** |
| **B1a** | bootstrap CI on ρ(Lev, LB) − ρ(comparator distance, LB), per Suite-2 dataset | 7 × 10 | **70** |
| **B2** | Friedman omnibus on ρ, **approximate regime only** | 1 | **1** |
| **B3e** | MRM standardised β₁ (D4), permutation inference, per Suite-1 dataset | 5 | **5** |
| **B3a** | MRM standardised β₁ (D4), permutation inference, per Suite-2 dataset | 10 | **10** |
| | | **N_max** | **182** |

**The Wilcoxon–Holm post-hoc under A2 and B2 is not counted in BH.** Holm already controls the FWER
within each post-hoc set; nesting it inside BH would correct twice. The omnibus is the BH-family
member; the post-hoc is reported under Holm and labelled as such (D8).

### 4.3 Why ρ(Lev, UB) is not a fourth B row

`ρ(Lev, LB)` and `ρ(Lev, UB)` are computed on the **same pairs** from two bounds on the **same
quantity**, so they are near-duplicates by construction. BH assumes independence or PRDS and behaves
worst on families of near-duplicates: adding 70 highly correlated tests inflates N without adding
evidence, and makes every genuinely independent test in the family harder to detect.

**The upper bound is not demoted — it is reported in full**, per
[approx_ged](approx_ged.md) §4: both ρ values printed per dataset, no interpolation, bracket width
`(UB − LB)/UB` per size and density stratum, certification rate, and symmetrisation. Its
**confirmatory** role is F1, where the invariance claim is registered directly. The primary reference
is the one already signed in [decisions](decisions.md) §5: exact GED for Suite 1, `BRANCH_FAST` for
Suite 2, **subject to T-27's per-dataset re-selection**.

---

## 5. N_max, N_actual, and the reduction rule

**BH is computed over `N_actual`. `N_max`, the exclusion list and a BH-over-`N_max` sensitivity column
are all printed** (decision 24). The sensitivity column is a re-threshold of stored p-values and costs
nothing, and it removes the only objection a reviewer can raise to reducing the denominator.

**`N_actual(F2)` is defined as the cardinality of the admissible cell set, enumerated in code**, with
the closed form below printed beside it as a check. **Where the two disagree the enumeration wins and
the discrepancy is reported** — coefficient arithmetic over three interacting reduction terms is
exactly where a silent double-count hides.

```
N_actual(F2) = 182 − 15·k − 8·d − c            (applied in that order; see §5.2)
```

| Symbol | Meaning | Range |
|---|---|---|
| `k` | representations excluded by [competitors](competitors.md) §3.4 — **no** candidate distance passes F1 at 100 %, F2, F3 and F4 | 0–7 |
| `d` | Suite-2 datasets whose bracket F1 declares uninformative | 0–10 |
| **`c`** | **individual F2 cells removed for non-computability**, counted *after* `k` and `d` — added 2026-08-16, see §5.1 | 0–115 |

### 5.1 `c` — the suite-restricted case, added 2026-08-16 (T-06)

**The hole this fills.** §5 as originally frozen had a term for a representation with *no admissible
distance* (`k`, −15, keeps Claim A) and a sentence for one that *cannot be computed at all* (−10
more, "recorded separately"). It had **no case for a representation computable on some datasets and
not others**. [competitors/README](competitors/README.md) **finding 6** raised it — *"§5's reduction
rule has no case for a representation computable on one suite and not the other"* — and assigned it
to "T-02's owner", i.e. nobody currently active.

T-04 then measured it. From **finding 5**, `agm_cam`'s per-dataset **failure** rates across Suite 2:

| dataset | Letter ×3 | LINUX | AIDS-GraphEdX | GREC | AIDS-IAM | COIL-DEL | Protein | Mutagenicity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fails | 0 % | 0 % | 0.4 % (Suite 1) | **24 %** | **18 %** | **46 %** | **90 %** | **98 %** |

> **`c` counts individual F2 cells, not representations.** A cell is `(row, representation,
> dataset)` for `row ∈ {A1, B1e, B1a}`. A cell is removed iff its representation **fails the
> computability criterion on that cell's dataset**.
>
> **Criterion, F5-blind**: a representation is computable on a dataset **iff it produces an encoding
> for ≥ 99 % of that dataset's graphs** within the frozen per-graph budget of **300 s**
> ([statistics](statistics.md) D14, enforced by a **killed subprocess**).

**Why per-cell and not per-representation.** A per-representation gate would delete `agm_cam`'s
rows on Letter ×3 and LINUX, where it completes at **100 %**. Those tests are computable and
informative, and removing them shrinks the BH denominator on evidence we actually hold. **Shrinking
`N_actual` further than the data forces is the anti-conservative direction** — it lowers the BH
burden on every surviving test — and it is the reduction a reviewer pushes hardest on. Per-cell
charging is the conservative reading and it is the one that matches what was measured.

**Three consequences of counting cells rather than representations:**

1. **`wl_subtree` has no A1 row** (no bit count — it raises `BitCountUndefined`), so a WL failure
   costs **1** cell per dataset, not 2. `k`'s range is likewise 0–7, not 0–6: Claim B has seven
   comparators.
2. **The IsalGraph reference arm is never charged to `c`.** It is not a comparator, and **D14
   governs it instead**: a censored graph is retained with its greedy-min string and flagged, never
   dropped. Its censoring rate is a *reported result*, not an exclusion. The `fallback_used` rate is
   printed per dataset beside every IsalGraph row.
3. **`A2`, `B2`, `B3e` and `B3a` are never charged to `c`.** The two omnibuses are single tests over
   whatever cells survive, and the MRM rows use the IsalGraph arm alone.

### 5.2 Precedence — the three terms are applied in order, and never double-count

```
1. k  removes a representation's 15 Claim-B cells   (5 B1e + 10 B1a) entirely
2. d  removes 8 cells per uninformative dataset     (7 B1a + 1 B3a)
3. c  removes, from what REMAINS, each (row, representation, dataset) cell
      whose representation fails the >= 99 % criterion on that dataset
```

A cell already removed by `k` or by `d` is **not** counted again in `c`. This is why `N_actual` is
defined by enumeration: with `k`, `d` and `c` interacting on a shared cell set, the closed form is a
check, not a definition.

**Budget provenance — a measured completion rate from another ticket is not an `c` determination.**
T-04 and T-04a measure encodability at each **backend's own configured budget** (`agm_cam` raises
`AGMBudgetExceeded`; `min_dfs` raises `MinDfsBudgetExceeded` at `max_projections = 50,000`;
`isalgraph_pruned` raises `CanonicalizationTimeoutError` at a **2 s** wall clock). Those are useful
as shape and are **not** the 300 s criterion above. `c` is determined by T-06's own encoding
campaign, under this criterion, and by nothing else.

**Recorded, not silent.** Every representation with any cell removed by `c` still enters the **AE.3
comparison table** (T-17) with its **measured per-dataset completion rate** printed and the reason
stated — the same disposition §3.4 gives a `k`-excluded representation. The full removed-cell list
is printed with the family.

> **Note for [competitors/README](competitors/README.md) finding 6**: that finding states the cost as
> **−10** (B1a only). §5.1 charges the Suite-2 **A1** cells as well, because a bit count needs an
> encoding and a representation that did not encode has none. **§5.1 is authoritative; finding 6 is
> incomplete on this point.**

**Why 15 and not 25.** T-04a's criteria F1–F4 are properties of a **distance**, not of an encoding. A
representation with no admissible distance loses its Claim B rows (5 × B1e + 10 × B1a = 15) and
**keeps its Claim A rows**, because a bit count needs no distance. A representation that cannot be
computed at all — `pynauty` failing to build, gSpan's minimum DFS code proving unreachable
([competitors](competitors.md) §2) — loses its Claim A rows too, at 10 each, and that case is
recorded separately.

**Why the reduction carries no bias.** T-04a's selection rule is **F5-blind by construction**: ties
break on cost (F6), *never* on correlation with GED. The exclusion is therefore independent of the
hypotheses in F2. F1's reduction is outcome-dependent by design, which is precisely why F1 is a
separate, prior family rather than a subset of F2.

---

## 6. What is deliberately NOT in the family

| Excluded | Why |
|---|---|
| **Friedman omnibus / CD diagram on the exact regime** | [statistics](statistics.md) §4 locks this: at `N = 5` the critical difference separates almost nothing, and an underpowered figure dressed as a result is worse than no figure. The exact regime is descriptive: per-dataset ρ with graph-level bootstrap CIs and D7 paired differences, **with the reason stated in the text** |
| **ρ(Lev, UB) per-dataset comparisons** | §4.3 — near-duplicates; the UB's confirmatory role is F1 |
| **Labels L1–L3** ([labels](labels.md) Tier 2) | **S-d is open (due 2026-08-18)** and Tier 2 is "logged, not written up". If Tier 2 is ever promoted it enters as its **own** pre-declared family with its own q = 0.05, appended here with a dated changelog entry — it does not silently enlarge F2 |
| All stratified analyses; per-stratum timeout and censoring rates | [statistics](statistics.md) §8 — exploratory by decision |
| D14's complete-case sensitivity arm | it is a sensitivity arm; the primary arm carries the claim |
| Pruned-vs-exhaustive encoding comparison; encode-time regressions | descriptive measurements, not hypotheses |
| Dataset-level regression (`N = 10`) | descriptive at that N, and labelled so |
| Per-dataset GEDLIB cost-model sensitivity arms | sensitivity, by construction |
| **T-27's method bake-off** | it is a **selection** procedure, not a hypothesis test. Its output is which method is primary; it makes no claim requiring FDR |

---

## 7. Changelog rule

The family is frozen. Any change to §2–§5 after 2026-08-13 requires a dated entry in §8 stating
**what changed, why, and whether any p-value had already been computed under the previous version.**
A change made after seeing results is disclosed as such in the paper.

Two determinations are known to be outstanding and are **not** deviations — they are pre-declared
parameters resolved by pre-declared rules:

| Outstanding | Resolved by | Due |
|---|---|---|
| `k` — representations excluded | **T-04a**, rule in [competitors](competitors.md) §3.4 | before T-06 |
| `d` — datasets with an uninformative bracket | **F1**, rule in §3 above | during T-06 |
| **`c` — F2 cells removed for non-computability** | **T-06's encoding campaign**, rule in §5.1, precedence in §5.2 | during T-06, before any F2 p-value |
| The primary bound at each end | **T-27**, rule in [approx_ged](approx_ged.md) §2 | before T-06 |

## 8. Changelog

| Date | Change | p-values already computed? |
|---|---|---|
| 2026-08-13 | Initial freeze, N_max = 197 across three families | none |
| **2026-08-16** | **§5 reduction rule gains a fourth term, and §5.2 gives the three terms an explicit precedence.** `N_actual(F2) = 182 − 15k − 8d − c`, with `c` defined in the new §5.1 and `N_actual` **defined by enumeration** with the closed form as a printed check. **Why**: the rule as frozen had no case for a representation computable on some datasets and not others — [competitors/README](competitors/README.md) **finding 6**, assigned to "T-02's owner" and unowned since. Without it T-06 would either print a column conditioned on tractability or reduce the denominator by an unwritten rule. **`N_max` is unchanged at 182 / 197** — this adds a reduction term, not a test. The criterion is a per-dataset completion rate, never a correlation, so the reduction stays F5-blind (decision 24) | **none.** T-06 had not begun computing; no distance matrix, no ρ and no p-value existed under either version |
| **2026-08-16** (same day, superseding the entry above **before any computation**) | **`s` (per-representation, −20) is REPLACED by `c` (per-cell).** Raised by the T-04a session against the first draft and **verified against the sources**. Three defects: (i) **the arithmetic over-charged.** `agm_cam` completes at **100 %** on Letter ×3 and LINUX (finding 5), so a per-representation gate deleted ~10 cells it does deliver — and under-counting `N_actual` is the **anti-conservative** direction, lowering the BH burden on every surviving test. (ii) **the citation was wrong** — the hole is finding **6**; finding **4** is the sparse6 `m/n` inversion. (iii) `k` and `s` were both ranged 0–6 although Claim B has **seven** comparators. Also added: the precedence rule preventing `k`/`d`/`c` double-counting, the exemption of the IsalGraph arm (governed by D14, not by `c`), and the statement that another ticket's completion rate at its **own** budget is not a `c` determination | **none.** No cell of F2 had been computed under either version |
