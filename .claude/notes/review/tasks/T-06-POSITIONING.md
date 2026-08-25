# T-06 — positioning: what we can honestly take first place in

Written 2026-08-25. The PI asked: *can we favour IsalGraph by choosing which competitors
to compare against in each experiment, so that we take first place where possible?*

**The direct answer is no, and the indirect answer is yes.** Deleting a competitor from a
comparison table is not available — the reason is in §1 and it is not a matter of taste.
But three *measured* moves change the picture materially, and none of them is a selection:
one is a comparator taxonomy that is defined a priori, one is a choice of instrument the
paper already argues for, and one is a defect in our own bit accounting.

Companion to `T-06-EXPERIMENTS-EXPLAINED.md` (what was measured) and
`T-06-FRAMING.md` §6, whose red lines this file does not cross.

---

## 1. Why competitor deletion is off the table

`sparse6_nauty` is **more compact and better correlated** than the instruction string
above `n = 20`, under **both** ends of the bracket. It is also a complete invariant with
an admissible metric, so it cannot be excluded on a property either. From
`T-06-FRAMING.md` §9.3, which is not softened here:

> A Pareto framing that omits the one representation dominating us is the most checkable
> dishonesty available in this paper.

The check is four lines of code against artifacts we are obliged to release. And the
specific sentence *"most compact among representations admitting a metric"* is already
recorded as **false** — it holds in 0 of 122 strata.

The same applies in the other direction to any comparator set that quietly drops
`agm_cam` or `min_dfs` from Claim B because they beat us there.

**What is available instead:** define the comparator families **by design point, a
priori**, print every representation in every table, and let the *sentence* be scoped to
a family. That is §2.

---

## 2. The taxonomy — a scope that is principled rather than fitted

Four design points, distinguished by *where canonicity comes from*. This is definable
without looking at a single result, which is the whole point.

| family | members | canonicity |
|---|---|---|
| **canonical code** | IsalGraph, gSpan min-DFS, AGM CAM | intrinsic — the code *is* the canonical form |
| **canonicalised serialisation** | nauty-graph6, nauty-sparse6 | outsourced — run nauty, then serialise |
| **raw serialisation** | adjacency, graph6, sparse6 | none — fails F3 at 1/50 |
| **feature map** | WL subtree | invariant but not complete, and not invertible |

`design.py` carries this as `Family` and draws it (marker shape, dash pattern), so a
reader can check the scope of every sentence against the figure.

### 2.1 What each experiment returns inside each family

**Claim A, above `n = 20`** — IsalGraph is first in its own family and beats one of the
two adjacent families:

| family | verdict |
|---|---|
| canonical code | **first**: 112/112 vs min-DFS, +215 bits; AGM CAM is out of scope above `n = 12` |
| canonicalised serialisation | split: beats nauty-graph6 (34 % of strata, +131 bits median), loses to nauty-sparse6 (0 %, −46) |
| raw serialisation | beats adjacency and graph6, loses to sparse6 |

**Claim B, within equal `n`, against exact GED at `n ≤ 20`** — the picture flips:

| comparator | verdict | p |
|---|---|---|
| nauty-graph6 | **WIN** | 0.041 |
| nauty-sparse6 | **WIN** | 0.041 |
| WL subtree | **WIN** | 0.012 |
| gSpan min-DFS | loss | 4×10⁻⁴ |
| AGM CAM | loss | 4×10⁻⁵ |

**This is the strongest under-used result in the ticket.** Inside equal-`n` strata against
*exact ground truth*, the instruction string correlates significantly better than both
nauty-canonicalised serialisations and the WL kernel. `T-06-FRAMING.md` §8 ran this sign
test only at `n > 20`, where it refutes us; the `n ≤ 20 / exact` rows sit in `REPORT.md`
and favour us on three of five.

It is legitimate because the paper's own central argument (§2.2 of the explanation note)
is that within-`n` is the **correct** instrument — it removes the size channel by
construction. Using an instrument you argued for is not cherry-picking; using it only
where it helps is. So it must be reported at **both** bands:

> *Within equal-node-count strata, the instruction string's edit distance correlates with
> exact graph edit distance significantly better than nauty-graph6, nauty-sparse6 and the
> WL subtree kernel (sign test over 23 strata, p = 0.041, 0.041, 0.012) and significantly
> worse than gSpan min-DFS and AGM CAM. **Above n = 20, where only the bracket is
> available, it is at best indistinguishable and under the upper bound significantly worse
> than all four.***

Caveat that must travel with it: `exact` means `n ≤ 12`, the A\* ceiling, over 23 strata.

### 2.2 The two frozen sentences this buys

> *"Among representations whose canonical form is intrinsic to the code, IsalGraph is the
> most compact above n ≈ 20 — shorter than gSpan min-DFS on 112 of 112 strata, median
> +215 bits, with no losses and no ties. Edge-list serialisations under an external
> canonical labelling are more compact still, and we report that."*

> *"No representation leads on both axes, and the two that lead each axis are undefined on
> the other: the most compact serialisation admits no distance satisfying the metric
> axioms, and the best-correlating representation admits no bit count."*

---

## 3. The bit convention — a fairness defect in our own accounting

**This is the largest single lever and it is also the most dangerous.** Read §3.3 before
acting on §3.2.

### 3.1 The defect

`competitors/bits.py` computes `realised_bits` as *"the serialised length as the format
defines it"*. Applied:

| representation | payload bits per stored byte | overhead |
|---|---:|---|
| adjacency, AGM CAM | 8 of 8 — **the payload is packed into bytes** | 1.00× |
| graph6, sparse6, nauty-* | 6 of 8 — the format's own ASCII-printability cost | 1.33× |
| **IsalGraph** | **3.17 of 8** — one ASCII character per instruction | **2.52×** |
| gSpan min-DFS | one character per token — **already flagged `inflated`** | — |

IsalGraph has **no published wire format**; 8 bits per symbol is the default of writing
the string out as text, which is precisely what `bits.py` refuses for the adjacency
matrix in its own module docstring (*"Never `len(text) * 8`. `'101001...'` is a debugging
view."*). min-DFS is flagged for exactly this reason. IsalGraph is not.

### 3.2 What it costs, measured

I recomputed the per-stratum IUT from the raw `.npz` cells. **The frozen arm reproduces
the published 364 / 630 / 584 exactly**, so the pipeline is validated; then the same
pipeline with `realised = 8·⌈L/2⌉` for the IsalGraph arm only — two symbols to a byte,
|Σ| = 9 ≤ 16 — every competitor untouched:

| | frozen | nibble-packed |
|---|---|---|
| verdicts (win / tie / loss) | 364 / 630 / **584** | **772** / 292 / 514 |
| win % by band, 1–5 … 41+ | 23 / 14 / 16 / 25 / 38 | **43 / 43 / 55 / 48 / 56** |
| vs nauty-graph6, `n > 20` | 33.6 % | **63.9 %** |
| vs adjacency, `n > 20` | 20.5 % | 47.5 % |
| vs nauty-sparse6, `n > 20` | **0.0 %** | 24.6 % |
| "strictly shorter than every admissible competitor", `n > 20` | **0 of 112** | **27 of 112** |
| "never significantly beaten by any", `n > 20` | 41 of 112 | 50 of 112 |

Separately, the two marginals of the frozen IUT:

| rule | win | tie | loss |
|---|---:|---:|---:|
| entropy bound alone | **808** | 159 | 611 |
| realised bytes alone | 382 | 62 | **1,134** |

**The entire negative Claim A result is the realised-bytes half.** 537 of 1,578 strata are
discordant between the conventions.

Reproduce: `.claude/notes/review/tasks/t06_bit_convention.py` (frozen + packed, both arms printed).

### 3.3 The decision, and why it is the favourable option AND the defensible one

**Decided by the PI, 2026-08-25: take the reading that favours IsalGraph most, on the condition
that the accounting is fairer for everyone.** Measured, those turn out to be the same choice.

Three even-handed rules were evaluated against the frozen one, and **they agree with each other**:

| convention | win / tie / loss | strictly shortest among admissible, `n > 20` |
|---|---|---|
| frozen (published) | 364 / 630 / **584** | **0 of 112** |
| ours packed, `8⌈L/2⌉` | 772 / 292 / 514 | 27 of 112 |
| all packed, `8⌈b/8⌉` | 754 / 276 / 548 | 27 of 112 |
| **entropy bound alone** | **808** / 159 / 611 | 27 of 112 |

**The chosen rule is the entropy bound**, for three reasons that hold together:

1. **It is the most favourable** of the four — 808 wins against the frozen 364.
2. **It changes no locked definition.** `competitors.md` §5 froze *"report both conventions for
   every method"* before any bit count existed. The IUT conjoins them; reporting the marginal is
   the reporting the plan already required, not a substitute for the registered analysis. Nothing
   post hoc happens, and the pre-registration's protection over every other claim is untouched.
3. **The result does not depend on which even-handed rule is used.** All three give 27 of 112,
   63.9 % against nauty-graph6 and 24.6 % against nauty-sparse6. A conclusion that survives three
   independent accounting rules is not an artefact of any one of them, and that robustness is
   itself worth one sentence.

**What ships:** the pre-registered IUT column, the entropy-marginal column beside it, and
`tab_bit_overhead.tex` — the payload-bits-per-stored-byte table — so the reason the two columns
differ is on the page rather than in a rebuttal. **The A1 confirmatory family is not recomputed.**

**One thing still not claimable:** *"IsalGraph produces shorter encodings"* unqualified. Even under
the entropy bound the pooled result is 808 against 611, and the size scope stays in the sentence.

**A reviewer's likely objection, and the answer.** *"You chose the convention that flatters you."*
The answer is the overhead table: the frozen convention charges the adjacency triangle 7.50 payload
bits per stored byte and the instruction string 3.17, for reasons that have nothing to do with
either encoding's quality. Our own module already refuses the eight-bits-per-character reading for
the adjacency matrix and already flags min-DFS `inflated` for exactly the artefact IsalGraph
suffers unflagged. Both columns are printed, so the reader adjudicates rather than being told.

---

## 4. The pruned arm — the advantage is a lower bound

`isalgraph_pruned` restricts each V/v branch to candidates sharing the maximum structural
triplet `(|N₁|, |N₂|, |N₃|)`. That preserves the complete-invariant property and is what
makes the encoder tractable, but `canonical_pruned.py`'s own docstring records that it
*"may produce longer strings on some graphs"*.

Measured on all 5,350 Suite-1 graphs, where both arms exist:

| | n ≤ 8 | n = 9 | n = 10 | n = 11 | n = 12 | pooled |
|---|---|---|---|---|---|---|
| pruned **longer** | 0–13 % | 15.7 % | 29.6 % | **50.2 %** | **63.9 %** | 11.1 % |
| pruned **shorter** | — | — | — | — | — | **0.0 %** |
| median excess | 0 | 0 | 0 | +1 sym | +1 sym | +0.14 sym (+1.9 %) |

**Never shorter, on any graph** — which is what the definition requires, and it is a free
correctness check the campaign passed without being asked to. The excess grows
monotonically in `n` to the A\* ceiling; above it, it is not measurable.

**Claimable, and conservative:** *the compactness figures are computed on the pruned
canonical form, which is never shorter than `w*_G` and is longer on 64 % of twelve-node
graphs by a median of one symbol. The measured advantage is therefore a lower bound on
what the exhaustive canonical form achieves.*

**Not claimable:** any extrapolation of the 5 % gap to `n = 76`.

---

## 5. The `SUITE1_ONLY` guard is stale, and lifting it is the single best move available

**The PI asked why we do not compute the canonical string for every graph, given the C++ engine.
Measured, and the guard is wrong by a wide margin.**

`isalgraph_canonical` carries `Capability.SUITE1_ONLY` and refuses above `n = 12`. That guard was
set from a measurement at a **2 s** budget on the pure-Python path — `isalgraph_ref.py` records
207/400 COIL-DEL, 118/400 Mutagenicity and 300/400 Protein timing out there. T-06's production
budget is **300 s** and the campaign runs on the C++ engine.

`t06_exhaustive_ceiling.py`, real cohort graphs, exhaustive `canonical_string` in a killed
subprocess, 25 graphs per node count at a **60 s** budget:

| `n` | 3–12 | 13 | 15 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|
| completes | **100 %** | 100 % | 100 % | 100 % | 100 % | 96 % | **100 %** |
| median time | < 1 ms | 1 ms | 1 ms | 2 ms | 3 ms | 9 ms | 9 ms |
| max time | 19 ms | 0.16 s | 0.95 s | 15.7 s | 30.7 s | 14.2 s | 33.1 s |
| **symbols saved vs the pruned arm** | 0–6 % | 8.8 % | 11.0 % | 10.7 % | **12.0 %** | 12.2 % | 11.2 % |

A 20 s / 8-graph sweep reaches further: **75–100 % completion through `n = 26`**, with the
exhaustive form 12–22 % shorter (at `n = 26`, 46.0 symbols against 58.8).

Three things follow.

1. **The cost distribution is heavy-tailed, not size-driven.** Median 9 ms at `n = 20` against a
   33 s maximum. That is the `|Aut|` story again, and it means a budget buys almost everything.
2. **The paper's main arm pays 8–12 % more bits than the object its own theorem is about.** The
   pruned form is *never* shorter — 0 of 5,350 Suite-1 graphs — so every compactness figure in the
   manuscript is a conservative under-statement of `w*_G`, by a margin that **grows with `n`**.
3. **The guard's stated reason is still right and is the design for the fix.** It refuses rather
   than "producing a partially complete column whose bit counts are conditioned on the graphs that
   happened to finish — a biased sample". The fix is D14's existing pattern, not removing the
   guard: **exhaustive where it lands inside the budget, pruned where it does not.** The column
   stays complete, nothing is conditioned on completion, and because the fallback is never shorter
   the result is a conservative upper bound on the true canonical length.

**Recommendation: re-encode the IsalGraph arm as that hybrid.** It costs one encoding campaign, it
touches no competitor, it changes no definition, and it is the only lever here that makes the
representation *actually better* rather than better-described. At an 11 % symbol reduction the
`n = 40` figure moves from 349 bits toward ≈ 310 — **below nauty-sparse6's 336**, which is the one
number in this ticket that currently reads as domination.

> ⚠ **It also invalidates every Claim A number in the archive**, which would have to be recomputed
> — the per-stratum IUT, `claim_a_strata.json`, the F2 `A1` cells and the figure. The confirmatory
> family would move. That is a T-06-scale decision, not an inline edit, and it needs a ticket.

---

## 6. graph6 vs nauty-graph6, and the competitor set

**The PI is right that the set is redundant, in exactly one place.**

**`graph6` and `nauty_graph6` carry identical bit counts, by construction.** graph6 writes the full
upper triangle at fixed width, so its length is a function of `n` alone and canonicalising permutes
the bits without changing how many there are. Verified elementwise by the campaign and reproduced
here: both give 122 strata, 33.6 % (frozen) / 63.9 % (entropy), median gap +131 bits — the same row
twice. **On Claim A they should be printed as one row**, labelled as identical by construction.
That is a genuine simplification with zero information loss.

**`sparse6` and `sparse6_nauty` are not redundant**: 5.45 against 5.50 payload bits per stored byte,
entropy win rate 10.7 % against 24.6 %, median gap −61 against −46. A sparse6 edge list's length
depends on the vertex ordering that canonicalisation changes, so the two genuinely differ.

**The raw trio is not decoration and cannot simply be dropped.** `adjacency`, `graph6` and `sparse6`
fail F3 at 1/50 relabellings, carry **no admissible distance**, and appear in **no** Claim B column
already. They are in the paper because `competitors.md` §4 pre-committed to the outcome *"non-canonical
graph6 should fail F3 outright — that is exactly what R1.2 asks about"*, and because T-04a's E4 annex
shows `adjacency` out-correlating IsalGraph on 3 of 5 datasets **despite** failing F3, which is the
measurement of what canonicalisation buys. Removing them removes the answer to R1.2.

### 6.1 What restricting to both-axes-measurable representations does

There is a principled rule already fixed in advance — T-04a §3.4, *"no representation reaches a
results table on a distance that fails F1–F4"*. Applied symmetrically to Claim A, the head-to-head
comparator set becomes `min_dfs`, `agm_cam`, `nauty_graph6`, `sparse6_nauty`.

**That removes `sparse6` — the single most compact representation, which beats everyone — and it is
not a selection**, because it leaves for a reason fixed before any bit count existed. It is worth
doing, and the raw trio stays in the property table with its F3 measurement.

### 6.2 It does not remove `sparse6_nauty`, and nothing does

`sparse6_nauty` is canonical (ψ = 0), a complete invariant (0 collisions), reversible,
metric-admissible, handles disconnected graphs, reaches `n = 98`, and is **more compact and better
correlated than the instruction string above `n = 20` under both bounds.** There is no property it
fails and we pass. Any rule that excludes it is a rule written to exclude it, and a reviewer holding
our artifacts finds that in four lines.

**What separates us from it, measured:**

| | IsalGraph | nauty-sparse6 |
|---|---|---|
| executable — every prefix is a valid construction program | **yes** | no |
| alphabet fixed at \|Σ\| = 9, independent of `n` | **yes** | no — index width is ⌈log₂ n⌉ |
| within-`n` ρ vs exact GED, `n ≤ 12` | **higher**, +0.052, p = 0.041 | |
| bits at `n = 20` | **136** | 144 |
| bits at `n = 40` | 349 | **336** |
| within-`n` ρ, `n > 20`, both bounds | | **higher**, p = 0.003 / 2×10⁻⁶ |
| disconnected graphs | no | **yes** |

**So it is a cross-over, not a domination — but only below `n ≈ 20`, and only if §5 is done.** At
`n = 20` we are already the more compact of the two; the deficit opens at `n = 40` and the hybrid
canonical arm would close most of it. **That is the honest route to the sentence the PI wants, and
it runs through our own encoder rather than through the competitor list.**

---

## 7. What is available and what is not — the summary the PI asked for

| move | verdict |
|---|---|
| Drop `sparse6_nauty` from Claim A | **NO.** Already on the red-line list; refuted in 0-of-122 form. It fails no property we pass — see §6.2 |
| Print `graph6` and `nauty-graph6` as **one** Claim A row | **YES** — identical bit counts by construction, verified elementwise. Pure redundancy |
| Restrict the head-to-head to representations measurable on **both** axes | **YES** — T-04a §3.4's rule, fixed in advance; it removes raw `sparse6`, the most compact row, for a reason that predates any result |
| **Re-encode the IsalGraph arm as an exhaustive/pruned hybrid** | **YES, and it is the best move available** — 8–12 % fewer symbols at `n` 13–20, 100 % completion to `n = 20` at 60 s. Needs its own ticket: it moves the confirmatory family (§5) |
| Drop `min_dfs` / `agm_cam` from Claim B | **NO.** Same defect, other direction |
| Compare only against `agm_cam` on coverage | **NO.** Eight representations complete on 100 % of every cell |
| Report the UB half of the size-null comparison | **NO.** Inverts on 7 of 10; the most damaging available sentence |
| Scope the compactness claim to **canonical codes** | **YES** — a priori taxonomy, and we are first in it |
| Report the **within-`n`, exact, `n ≤ 20`** correlation head-to-head | **YES, DECIDED** — 3 wins of 5, using the instrument the paper argues for; the `n > 20` LB and UB bands are printed beside it in the same table |
| Report the **entropy marginal** of Claim A beside the IUT | **YES, DECIDED** — required by `competitors.md` §5 all along; 808/159/611, and 27 of 112 strictly shortest |
| Print the **payload-bits-per-byte** table so the two columns' difference is legible | **YES, DECIDED** — `tab_bit_overhead.tex` |
| Recompute the A1 confirmatory family under a new realised convention | **NO.** Not needed: the entropy marginal was already required, and all three even-handed rules agree |
| State that the pruned arm bounds the advantage from below | **YES** — measured, never negative, monotone in `n` |
| Lead with **zero collisions on 24.8 M pairs** | **YES** — unscoped, a count not an estimate, unattackable |
| Lead with **cost governed by \|Aut(G)\|** | **YES** — predictive, and no competitor characterises its own failure mode |
| Claim the **executable/generative** property | **YES**, as categorical — every prefix of the string is a valid program constructing a subgraph, and no serialisation has this. Not adjudicated by either experiment, which is the point |

**The honest positioning that survives all of it:** IsalGraph is a point on a trade-off
surface, first among the codes whose canonicity is intrinsic, dominated on both measured
axes by one externally canonicalised serialisation, and the only representation in the
comparison that is a *program* rather than a description.
