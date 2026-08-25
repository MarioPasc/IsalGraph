# T-06 — the two experiments, end to end

Written 2026-08-25 for the PI. Companion to `T-06-FRAMING.md` (what may be claimed)
and `T-06-FILES.md` (where everything lives). This file answers a different question:
**what was actually measured, on what, and what does each number mean.**

Nothing here is new science. Every figure is traceable to
`results/reports/T-06-full-recompute/`; where I recomputed something to check it, I say so.

---

## 0. The shape of the campaign

Two claims, measured on the same frozen cohorts, with the same encoder.

| | Claim A | Claim B |
|---|---|---|
| question | is the instruction string **short**? | does its **edit distance** track graph edit distance? |
| unit | one graph | one *pair* of graphs |
| response | bits | Spearman ρ |
| comparator | 6 serialisations + `sparse6_nauty` | 4 metric-admissible representations + a trivial baseline |
| verdict direction with `n` | **improves** | **collapses** |

The last row is the ticket's most interesting fact and the reason the two must be
presented together: they move in opposite directions, so they cannot share a cause.

### The cohorts

| | graphs | pairs | GED reference |
|---|---:|---:|---|
| **Suite 1** | 5,350 | 3,897,911 | **exact**, `networkx` A\*, ceiling `n ≤ 12` |
| **Suite 2** | 16,370 | 21,710,892 | **bracket**: BRANCH-FAST (LB) / IPFP (UB), GEDLIB |

Cost model D6: node insert/delete = 1, edge insert/delete = 1, **substitutions free**.
Node and edge labels are therefore *not* used — both sides of every correlation are
topology-only, which is the direct answer to R1.3b.

Engine `cpp`, build `298fc1188bf1b051`, seed 42, 300 s per-graph encode budget.

### The three GED references, and why `exact` is not a third bound

`exact` is ground truth. `lb` and `ub` are two ends of a **proven** bracket on the same
pairs, never averaged and never interpolated. The bracket turned out wide enough at these
sizes to **flip verdicts**, which the pre-registration anticipated and F1 measured
(`d = 7 of 10`). Every Claim B verdict in the paper therefore carries both ends.

The mechanism was measured on the three IAM Letter datasets, whose Suite-1 and Suite-2
cohorts are *identical* (same graph ids, same pairs), so the three references are
comparable on the same data:

| dataset | ρ(\|Δn\|, exact) | ρ(\|Δn\|, lb) | ρ(\|Δn\|, ub) |
|---|---:|---:|---:|
| `iam_letter_low` | 0.9139 | 0.9804 | 0.7482 |
| `iam_letter_med` | 0.9146 | 0.9740 | 0.7363 |
| `iam_letter_high` | 0.9195 | 0.9224 | 0.7080 |

**BRANCH-FAST's lower bound is *more* size-dominated than truth; IPFP's upper bound is
*less*.** They straddle. An arm carrying structural information must therefore look worse
against LB and better against UB, and that is exactly the inversion observed on 7 of 10
datasets. This is a finding about the bounds, useful to anyone bracketing GED.

---

## 1. Claim A — information content

### 1.1 What is counted

`competitors/bits.py` is the only module that produces a bit count, and it emits
**two conventions for every representation, always together**:

| convention | definition | what it answers |
|---|---|---|
| **entropy bound** | `L · log₂|Σ|` for a string; `n(n−1)/2` for a raw bit vector | like-for-like *encoding efficiency* |
| **realised bytes** | the serialised length **as the format defines it** | what a practitioner stores |

Concretely:

| representation | entropy bits | realised bits |
|---|---|---|
| IsalGraph | `L · log₂ 9 = 3.170 L` | `8 L` — one ASCII char per instruction |
| adjacency / AGM CAM | `T = n(n−1)/2` | `8·⌈T/8⌉` — **the payload packed into bytes** |
| graph6 / nauty-graph6 | `6·len(wire)` | `8·len(wire)` |
| sparse6 / nauty-sparse6 | `6·len(wire) − 6` (the `:` prefix is framing) | `8·len(wire)` |
| gSpan min-DFS | `m · 2·bitlen(n−1)` | `8·len(text)`, **flagged `inflated`** |
| WL subtree, size null | — | — → `BitCountUndefined` |

Two rules the module exists to enforce, both worth knowing:
**measure the wire, never a closed form** (graph6's `1 + ⌈n(n−1)/12⌉` is wrong above
`n = 62` and Suite 2 reaches 98); and **never `len(text)·8`** for a bit vector, which
would inflate the adjacency matrix eightfold and hand us a baseline we beat for free.

### 1.2 How the two conventions are combined

Claim A is a **conjunction** — *fewer bits under both conventions* — so the
pre-registration resolves it with an **intersection–union test**:

    p = max(p_entropy, p_realised)

and a stratum is a win only when the median gap is positive under both. The IUT is the
valid level-α test for a conjunction, is conservative for BH, and removes the need to
name a primary convention — which was the F-5 hazard, not a way of managing it.

### 1.3 The unit of analysis

Per **node-count stratum**: a Wilcoxon signed-rank on paired bit counts within
`(dataset, comparator, n)`, minimum 8 graphs. 1,578 strata; 2,815 graphs sit in strata
too thin to test and are reported as such. Stratifying on `n` is what makes the
comparison fair — encoding length is dominated by size, so a pooled test would mostly
measure which cohort each representation happened to cover.

### 1.4 The result

**364 win / 630 tie / 584 loss** over 1,578 strata. Pooled, Claim A is **net-negative**.

The scope is the whole story:

| `n` | 1–5 | 6–10 | 11–20 | 21–40 | **41+** |
|---|---:|---:|---:|---:|---:|
| strata | 217 | 350 | 289 | 420 | 302 |
| IsalGraph shorter | 23.0 % | 14.0 % | 15.9 % | 25.0 % | **37.7 %** |
| median gap (bits) | −0.7 | +1.5 | +5.8 | +9.9 | **+164.7** |

Per comparator, above `n = 20`:

| comparator | strata | win % | median gap | max `n` |
|---|---:|---:|---:|---:|
| **gSpan min-DFS** | 112 | **100.0** | **+215** | 96 |
| nauty-graph6 | 122 | 33.6 | +131 | 98 |
| graph6 | 122 | 33.6 | +131 | 98 |
| adjacency | 122 | 20.5 | +123 | 98 |
| **nauty-sparse6** | 122 | **0.0** | **−46** | 98 |
| sparse6 | 122 | 0.0 | −61 | 98 |
| AGM CAM | 0 | — | — | **12** |

`graph6` and `nauty-graph6` carry **identical** counts by construction: graph6 writes the
full upper triangle at fixed width, so its length is a function of `n` alone, and
canonicalising permutes the bits without changing how many there are. Verified
elementwise.

**What is true:** *IsalGraph is the most compact of the canonical codes, and beats gSpan
min-DFS on 112 of 112 strata above `n = 20` with no losses and no ties.*
**What is false:** *"most compact among representations admitting a metric"* — that holds
in **0 of 122** strata. `nauty-sparse6` blocks it at every size above 20.

### 1.5 The realised-bytes convention is not comparable across formats

I recomputed the whole per-stratum IUT from the raw `.npz` cells. **The frozen arm reproduces
the published 364 / 630 / 584 exactly**, and the per-band and per-comparator tables with it, so
the pipeline is validated before anything is varied. Then, measured over both cohorts, how many
payload bits each stored byte actually carries:

| representation | payload bits per stored byte | overhead |
|---|---:|---:|
| adjacency | 7.50 of 8 — **the payload is packed** | 1.07× |
| graph6, nauty-graph6, AGM CAM | 6.00 of 8 — the format's published ASCII cost | 1.33× |
| sparse6 / nauty-sparse6 | 5.45 / 5.50 of 8 | 1.47× / 1.45× |
| **IsalGraph** | **3.17 of 8** — one ASCII character per instruction | **2.52×** |
| gSpan min-DFS | 1.83 of 8 — **already flagged `inflated`** | 4.37× |

**The realised-bytes column measures how wasteful each format's rendering happens to be, not how
well it encodes.** IsalGraph has no standardised wire format; eight bits per symbol is the default
of writing the string out as text, which is exactly what `bits.py` refuses for the adjacency
matrix in its own docstring. min-DFS carries an `inflated` flag for this; IsalGraph does not.

Four conventions, same 1,578 strata (`t06_bit_convention.py`):

| convention | win | tie | loss | 41+ win % | vs nauty-graph6, n>20 | strictly shortest among admissible, n>20 |
|---|---:|---:|---:|---:|---:|---:|
| **frozen** (as published) | 364 | 630 | **584** | 37.7 | 33.6 % | **0 of 112** |
| ours packed, `8⌈L/2⌉` | 772 | 292 | 514 | 55.6 | 63.9 % | **27 of 112** |
| all packed, `8⌈b/8⌉` | 754 | 276 | 548 | 59.9 | 63.9 % | **27 of 112** |
| **entropy bound alone** | **808** | 159 | 611 | 60.3 | 63.9 % | **27 of 112** |

**The three even-handed rules agree with each other and disagree with the frozen one.** They give
the same 27-of-112, the same 63.9 % against nauty-graph6 and the same 24.6 % against nauty-sparse6;
they differ materially only against `adjacency` (47.5 % vs 63.1 %). So the choice among them does
not drive the conclusion — which is itself the robustness statement worth reporting.

**Decision taken (PI, 2026-08-25):** report the **entropy bound** as the primary descriptive
convention. It is the most favourable of the four *and* it requires no new definition at all —
`competitors.md` §5 locked *"report both conventions for every method"* before any bit count
existed, so the marginal is the half of the locked pair the conjunction absorbs, not a post-hoc
substitute. The pre-registered IUT is reported unchanged beside it. Both columns appear in
`tab_representation_headtohead.tex`, and the overhead table above is emitted as
`tab_bit_overhead.tex` so the reason is on the page rather than in a rebuttal.

## 2. Claim B — correlation with graph edit distance

### 2.1 What is correlated

For each pair `(G, H)` in a dataset: `Levenshtein(w(G), w(H))` against the GED reference,
**Spearman ρ**, over every pair the reference is defined on. `levenshtein` is the primary
distance for six of the seven measurable representations; `wl_subtree` uses `kernel`.

Each representation's primary distance was **selected by measurement in T-04a**, under a
rule fixed in advance and deliberately **blind to ρ**: the cheapest candidate passing
F1 (well-defined at 100 %), F2 (metric axioms), F3 (isomorphism-invariant) and F4
(non-degenerate), tie-broken by cost, **never by correlation**. Selecting on ρ would be
selecting the baseline that makes IsalGraph look best.

That rule excluded three representations outright: `adjacency`, `graph6` and `sparse6`
each fail F3 at **1/50** relabellings — their distance changes when you relabel the graph
and nothing else. **They carry no Claim B column at all.** T-04a's E4 annex shows why
that matters: `adjacency` out-correlates IsalGraph against exact GED on 3 of 5 datasets
*despite* failing F3, because both it and GED are size-dominated. A rule with sight of
F5 would have picked it.

### 2.2 The two views, and which one is the instrument

**`all_pairs`** — every pair in the dataset. This is the confirmatory view and it is what
the head-to-head table reports.

**`equal_n`** — pairs where `n_i = n_j`. Inside such a stratum `|n_i − n_j|` is
**identically zero**, so the trivial size baseline is *undefined* and there is nothing to
subtract. **Raw ρ inside a stratum is the structural signal with the size channel removed
by construction rather than by adjustment.** This is descriptive, and it is the correct
instrument.

Why it is needed: the benchmark itself is size-dominated. `ρ(|n_i − n_j|, GED)` —
the trivial baseline against ground truth, with no representation involved —
runs **0.71–0.997**, exceeding 0.96 on seven of ten Suite-2 datasets and reaching
**0.9971** on COIL-DEL. On IAM Letter it predicts *exact* GED at ρ ≈ 0.92. So correlation
with GED on this data measures **size agreement more than structural fidelity, for every
representation, ours included.**

### 2.3 So: LB, UB, or the bracket midpoint?

**Never a midpoint, and never an interpolation.** LB and UB are reported as two separate
series and two separate verdicts, on identical pairs. Your recollection that IsalGraph
correlates well against UB is right, and here is the exact shape of it:

| reference | records | below its own size null | clears it |
|---|---:|---:|---:|
| `exact` (Suite 1) | 5 | **4** | 1 |
| `lb` (Suite 2) | 10 | **10** | 0 |
| `ub` (Suite 2) | 10 | 3 | **7** |

**On the same pairs, the verdict inverts on 7 of 10 datasets.** That is why
*"IsalGraph clears the size baseline on the Suite-2 datasets"* is on the red-line list:
it is true under UB alone and false under LB, and a reviewer recomputing it finds the
omission. The honest word for the Suite-2 half is **undetermined**.

The Suite-1 half is not undetermined. Against **exact** GED, with no bound and no
interpolation, the trivial baseline beats the representation on 4 of 5 datasets:

| dataset | ρ(Lev, exact) | size null | excess |
|---|---:|---:|---:|
| `iam_letter_low` | 0.9278 | 0.9139 | **+0.0139** clears |
| `iam_letter_med` | 0.8833 | 0.9146 | −0.0313 |
| `iam_letter_high` | 0.6660 | 0.9195 | −0.2536 |
| `linux` | 0.4850 | 0.7097 | −0.2247 |
| `aids` | 0.3266 | 0.7863 | **−0.4597** |

Over all 25 records: **below the null on 17, every one of them significantly so**;
1 undetermined; 7 favour the string.

### 2.4 The head-to-head, `all_pairs`

**0 win / 1 tie / 24 loss** over 25 records, from a *paired* graph-level bootstrap on
identical pairs and identical resamples — never from two overlapping marginal intervals.
IsalGraph is the best representation on **none** of the records.

### 2.5 The head-to-head, within equal `n` — and this is where it changes

Sign test over per-stratum Δρ. Strata within a dataset are disjoint graph sets, so the
test is valid, and it weights every stratum equally regardless of pair count. **I
recomputed all of these from `size_profile.json` and they reproduce `REPORT.md` exactly.**

| comparator | `n ≤ 20`, exact | `n > 20`, LB | `n > 20`, UB |
|---|---|---|---|
| gSpan min-DFS | loss, p = 4e−4 | tie, p = 0.16 | loss, p = 0.0027 |
| AGM CAM | loss, p = 4e−5 | out of scope | out of scope |
| nauty-graph6 | **WIN, p = 0.041** | tie, p = 1 | loss, p = 0.028 |
| nauty-sparse6 | **WIN, p = 0.041** | loss, p = 0.0029 | loss, p = 2e−6 |
| WL subtree | **WIN, p = 0.012** | loss, p = 3.7e−4 | loss, p = 7.3e−7 |

**Inside equal-`n` strata against exact ground truth, the instruction string beats
nauty-graph6, nauty-sparse6 and the WL kernel, and loses to the two other canonical
codes.** Above `n = 20` it loses to everything under UB and ties two under LB.

This is not a contradiction of §2.4. The pooled view mixes the size channel, in which
representations that agree on size win; the within-`n` view removes it. The paper's own
argument is that within-`n` is the correct instrument, so this row is available — **but
only with `n ≤ 20` in the same sentence**, and `exact` there means `n ≤ 12`, the A\*
ceiling. See `T-06-POSITIONING.md` §2.

### 2.6 The collapse, and what it is not

`isalgraph_pruned` within-`n`: **ρ = 1.0000 at n = 3 → 0.2608 at n = 12 → 0.135 averaged
over n 13–30**. Above `n ≈ 40` **no representation in the comparison — ours or any
competitor — is reliably distinguishable from ρ = 0.** That is a statement about the
approach, measured on 21.7 M pairs, and it is publishable in its own right.

**It is not a budget artefact.** Removing every pair touching a censored graph *lowers*
ρ at both bounds and both size restrictions (LB all-`n` −0.0305, LB n>40 −0.0354, UB
all-`n` −0.0170, UB n>40 −0.0725). Report all three quantities, never the Δ alone:
censored-touching pairs do correlate worse in isolation (0.3273 against 0.6095 at
n > 40), they simply do not explain the collapse.

### 2.7 The model that controls for size instead of stratifying it away — D4

`GED ~ β₁·Lev + β₂·|Δn| + β₃·|Δdensity|`, all standardised.

> **β₁ is significant and positive on 19 of 19 usable fits, and the size coefficient
> exceeds it on 17 of 19, by 1.1×–5.6×.**

So the defensible sentence carries both halves: *Levenshtein contributes significant
incremental information beyond size and density, but node-count difference does most of
the work.* Six fits are excluded and the reasons are not interchangeable — `aids_iam` and
`coil_del` are **collinear** (VIF 18.1 / 16.2, r(Lev,|Δn|) = 0.96 / 0.94, so the split
between the two predictors is arbitrary), and `coil_del` and `mutagenicity` have a **point
estimate outside their own bootstrap interval** because tier-3 resampling fits the point
on all pairs and each replicate on a 2 M subsample.

A significant β₁ on a dataset whose within-`n` ρ is noise is **not** a contradiction:
§2.2 asks whether the distance tracks GED at a fixed size; the MRM asks whether it adds
anything *given* size, across all sizes. On these cohorts GED is itself heavily
size-driven, so a representation can carry real information about the size-driven part
and none about the residual.

### 2.8 The cleanest control in the ticket

IAM Letter LOW / MED / HIGH are the **same generator at three distortion levels** —
source, labelling and construction fixed, only the graphs differ. Mean node count rises
4.07 → 4.58; mean **edge** count rises 3.07 → 4.56 (+49 %). The family adds *structure*,
not *size*.

| dataset | ρ(Lev, exact) | size null | β_lev | β_size | ratio |
|---|---:|---:|---:|---:|---:|
| LOW | **0.9278** | 0.9139 | +0.5624 | +0.3537 | **0.63×** |
| MED | 0.8833 | 0.9146 | +0.4610 | +0.5086 | **1.10×** |
| HIGH | **0.6660** | 0.9195 | +0.2696 | +0.7507 | **2.78×** |

**The trivial baseline stays flat at ρ ≈ 0.92 while the string falls 0.93 → 0.67.** Two
independent instruments cross at the same dataset. This is the limitation stated as a
*condition* — the representation tracks edit distance where there is little structure to
track — and it is far better than "degrades on harder data".

---

## 3. What neither experiment measures

**Zero encoding collisions over 24,764,422 GED-positive pairs.** Suite 1 at exact
`GED > 0`, which certifies non-isomorphism: 3,424,764 certified pairs, zero collisions.
Suite 2 at `LB > 0`: 21,339,658 further pairs, zero collisions. It is a **count**, not an
estimate, so there is no interval to argue with. Caveat, one clause: on Suite 2 `LB = 0`
does not certify isomorphism, so pairs the bound could not separate lie outside that half.

**Encoding cost is governed by |Aut(G)|, not by `n`.** At the 300 s budget, censoring is
**0 %** for all 3,703 Mutagenicity graphs with |Aut| ≤ 10⁴, **21.85 %** at 10⁴–10⁸ and
**100 %** (35 of 35) above 10⁸. Nearly a step function in symmetry. This is *predictive*:
a user computes |Aut| in milliseconds and knows in advance whether the method applies.
Measured on Mutagenicity only — the only dataset that censors.

**Coverage is not a differentiator.** Eight representations complete on 100 % of every
cell. Only `agm_cam` (6.15 % floor on Protein) and `min_dfs` (0.9478) are worse.

---

## 4. Two things I measured while checking this file

Both are new, both are small, both are favourable, and both need your decision before
they go anywhere. Detail in `T-06-POSITIONING.md`.

**4.0 The `SUITE1_ONLY` guard at `n = 12` is stale.** It was set at a 2 s budget on the
pure-Python path. Measured on the C++ engine at 60 s, the exhaustive canonical completes on
**100 % of sampled graphs through `n = 20`** (median 9 ms, max 33 s) and 75–100 % through `n = 26`
at a 20 s budget. Full table and the recommendation in `T-06-POSITIONING.md` §5.

**4.1 The main arm is a length-suboptimal canonical form.** `isalgraph_pruned` restricts
the V/v branch to candidates sharing the maximum structural triplet. Its own docstring
says it "may produce longer strings"; on Suite 1, where both arms exist, it is:

| | n ≤ 8 | n = 11 | n = 12 | pooled n ≤ 12 |
|---|---|---|---|---|
| longer than the exhaustive form | 0–13 % of graphs | **50.2 %** | **63.9 %** | 11.1 % |
| median excess | 0 | +1 symbol (+5.6 %) | +1 symbol (+5.0 %) | +0.14 symbols |

**Never shorter, on any of 5,350 graphs** — which is what the theory says. So the
measured compactness advantage is a *lower bound* on what `w*_G` achieves. Above the A\* ceiling
the gap **widens**: 8.8 % at `n = 13`, 12.0 % at `n = 18`, 12–22 % at `n = 23`–26.

**4.2 The realised-bytes convention is not applied even-handedly** (§1.5). Under a
nibble-packed realised convention for the IsalGraph arm only — `8·⌈L/2⌉`, two symbols to
a byte, the same "pack the payload into bytes" rule `bits.py` already applies to the
adjacency matrix — the per-stratum IUT returns **772 win / 292 tie / 514 loss** instead of
364 / 630 / 584, the win rate rises at every size band, and *"strictly shorter than every
metric-admissible competitor above n = 20"* goes from **0 of 112** to **27 of 112**.

---

## 5. Where the code is

| artifact | code |
|---|---|
| the three ρ figures | `benchmarks/real_data/eval_size_profile/figures.py` |
| **the information-content figure** | `benchmarks/real_data/eval_t06_figures/fig_ic.py` |
| **the comparison tables**, including the wide summary | `benchmarks/real_data/eval_t06_figures/tables.py` |
| the exhaustive-canonical ceiling | `.claude/notes/review/tasks/t06_exhaustive_ceiling.py` |
| the bit-convention arms | `.claude/notes/review/tasks/t06_bit_convention.py` |
| pruned vs exhaustive length | `.claude/notes/review/tasks/t06_pruned_vs_exhaustive.py` |
| **colours, type sizes, names, taxonomy** | `benchmarks/real_data/eval_t06_figures/design.py` |
| archive readers, aggregation | `benchmarks/real_data/eval_t06_figures/data.py` |

`design.py` is the single design source. `eval_size_profile/figures.py` now imports from
it and no longer defines its own palette, display names, draw order or type sizes.
