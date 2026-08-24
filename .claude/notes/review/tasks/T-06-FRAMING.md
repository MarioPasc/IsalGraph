# T-06 — how to write the results up: strongest defensible framings

Written 2026-08-24 for the PI and T-20. **Every claim here is scoped to where it holds and every
number is traceable.** Scoping a result to its domain of validity is what reporting *is*; the section
that follows the framings — §6, the red lines — is what keeps it from becoming something else.

**The architecture that makes this defensible.** T-06 has a **pre-registered confirmatory layer**
(F0, F1, F2 over `N_actual = 79`, frozen before any p-value existed) and a **descriptive layer**
(the stratified analyses). The confirmatory layer is reported as it came out, including its negative
results. The favourable framings below live mostly in the descriptive layer, and **every one of them
must be labelled descriptive where it appears.** That labelling is not a hedge — it is the thing that
answers "did you go looking until you found something?" The answer is: the confirmatory claims were
fixed in advance and are reported unchanged; the descriptive claims are exploratory and say so.

---

## 1. Lead with completeness. It needs no scoping at all

> **The instruction string is a complete graph invariant, verified at cohort scale: zero encoding
> collisions across 24.8 million pairs.**

- Suite 1, against **exact** GED so `GED > 0` certifies non-isomorphism: **3,424,764 certified pairs,
  zero collisions.**
- Suite 2 at `LB > 0`: **21,339,658 further pairs, zero collisions.**
- It survives D14: the 101 censored Mutagenicity graphs carry a greedy-min fallback that is **not**
  canonical and therefore outside the theorem — and they collide with nothing either.

**Why this is the right opening.** It is a theorem (`w*_G = w*_H ⟺ G ≅ H`) confirmed empirically on
a scale almost no representation paper reports. It is unscoped, unqualified, and independent of every
comparison that follows. It is also the one result no reviewer can attack on methodology, because it
is a count, not an estimate — there is no confidence interval to argue with.

*Caveat to carry, one clause:* on Suite 2 the certification is `LB > 0`; `LB = 0` does not certify
isomorphism, so pairs the bound could not separate are outside that half of the test. Suite 1 has no
such gap. **Say it — it costs nothing and pre-empts the obvious question.**

---

## 2. Compactness: lead with the like-for-like win

> **Against `min_dfs`, IsalGraph is shorter on 112 of 112 size strata above `n = 20` — median
> +214.8 bits, zero losses, zero ties.**

Three reasons this is the strongest compactness sentence available:

1. `competitors.md` §2 calls `min_dfs` **"the single most important comparator"** — the paper is not
   choosing a weak opponent, it is naming the one the literature cares about.
2. **min-DFS is itself a canonical code**, so this is like-for-like: same design point, same
   guarantees, and IsalGraph wins on every stratum tested.
3. 112 of 112 with zero ties is a **clean sweep**, not a majority.

**Frozen wording for the general claim:**

> *"IsalGraph is the most compact of the canonical-code representations. Edge-list serialisations
> beat it at scale."*

Say **edge-list**, never "sparsity-exploiting": it names the mechanism rather than the outcome, and
conceding the right thing is exactly what makes the `min_dfs` result read as fair rather than
selected. These cohorts are sparse, so an edge list exploits a property of the *data* that an
instruction string does not — that is a difference in design point, not a defeat.

---

## 3. The scaling result — a genuine asymmetry, and the most interesting thing in the ticket

> **The compactness advantage grows with graph size while the distance correlation decays with it.
> The two claims do not share a cause.**

| `n` | 1–5 | 6–10 | 11–20 | 21–40 | **41+** |
|---|---|---|---|---|---|
| shorter than competitor | 20.4 % | 16.3 % | 18.9 % | 30.0 % | **45.6 %** |
| median gap | −1.2 bits | +0.5 | +5.8 | +58.6 | **+242.1** |

`competitor_shorter` falls 77 → 47 while `isalgraph_shorter` rises 38 → 114 — a trend, not a ratio
artefact. **Where a representation-length claim usually degrades with scale, this one improves.**
That is a positive scaling result and it is unusual enough to be worth a figure of its own.

> ### ⚠ The scoping is not optional — pooled over all sizes, Claim A goes AGAINST us
>
> Of the F2 `A1` cells that reject, **10 have IsalGraph LONGER against 9 shorter**. Across all sizes
> the significant bit results run *against* the representation by a narrow majority. **"Claim A is
> our win" is true only with the size qualifier attached**, and the stratification in the table above
> is what reconciles the two facts — small graphs lose, large graphs win, and the crossover sits
> around `n ≈ 20`.
>
> **Never write "IsalGraph produces shorter encodings" unqualified.** Write *"above `n ≈ 20`"*, in
> the same sentence, every time. This is the same discipline as §2's canonical-code qualifier: a
> scoped claim carries its scope inline or it is not scoped at all.

---

## 4. Turn the negative into a field-level finding — this is the honest reframe, not a dodge

§17 measures the within-`n` correlation, which removes the size channel **by construction** (inside a
stratum `|n_i − n_j| ≡ 0`, so the trivial baseline is undefined and there is nothing to subtract).
Result:

> **No serialisation-plus-string-distance representation tracks GED at scale. Above `n ≈ 40` not one
> of the seven — IsalGraph or any competitor — is reliably distinguishable from ρ = 0.**

This is a statement about **the approach**, not about IsalGraph. It is publishable in its own right:
it tells the field that a whole family of methods has a size ceiling, and it is measured on 21.7 M
pairs rather than asserted. It also does the framing work honestly — IsalGraph is not uniquely weak
here, and saying so is *true*, not spin.

**Pair it with the diagnostic that makes it useful:** the pooled ρ ≈ 0.93 previously reported on
sparse IAM is largely the **size channel**. Holding `n` fixed the same arm scores **0.26 at `n = 12`**
and **0.135** over `n` 13–30. Reporting that gap is a methodological contribution — it is the
correct way to evaluate any graph-distance surrogate, and the paper is the one introducing it.

### 4.1 The collapse is a property of the representation, not of the compute budget — measured

The obvious objection: above `n ≈ 60` roughly half of Mutagenicity's IsalGraph arm is a **greedy-min
fallback**, not the canonical string, because D14 censors at the 300 s budget. So is §17's decay just
the budget forcing a worse encoding on the large graphs? **Measured, and no.**

| reference | pairs | ρ all | ρ with every censored-touching pair removed | Δ |
|---|---:|---:|---:|---:|
| LB, all `n` | 8,158,780 | +0.7700 | +0.7395 | **−0.0305** |
| LB, `n > 40` | 147,153 | +0.6449 | +0.6095 | **−0.0354** |
| UB, all `n` | 8,158,780 | +0.8806 | +0.8636 | **−0.0170** |
| UB, `n > 40` | 147,153 | +0.7253 | +0.6528 | **−0.0725** |

**Removing the entire non-canonical arm LOWERS ρ, at both bounds and both size restrictions.** The
collapse survives. **§17 can therefore be stated without hedging: it is a statement about the
representation.**

**And because censoring tracks `|Aut|` rather than `n` (§5), the removed graphs are the *most
symmetric at their size*, not merely the biggest — so the unchanged collapse says something stronger
still: structural fidelity does not depend on symmetry either.**

> **Report all three quantities, never the Δ alone.** Pairs touching a censored graph do correlate
> markedly worse in isolation — **ρ = 0.3273 against 0.6095** for clean pairs at `n > 40`. *The
> fallback genuinely is worse; it simply does not explain the collapse.* The pooled ρ exceeding
> **both** components is a Simpson-type effect: censored graphs are systematically the largest, so
> the mixed set carries more size variance and the size channel lifts it. Quoting "Δ = −0.035" alone
> would hide that, and a reviewer who decomposes it would find it hidden.

*Power note, to state rather than omit:* the exact within-`n` contrast has only **6** usable strata —
the 101 censored graphs spread over `n = 45–98` where strata are thin — so the pooled contrast above
is what carries the conclusion, trading the within-`n` control for four orders of magnitude more
pairs. Both point the same way.

---

## 5. The cost characterisation — a contribution the competitors do not have

> **We characterise exactly when the canonical encoding is expensive, and it is not size: cost is
> governed by the automorphism group.** Censoring at the frozen 300 s budget is **0 % for all 3,703
> Mutagenicity graphs with `|Aut| ≤ 10⁴`, 21.85 % at `10⁴–10⁸`, and 100 % (35 of 35) above `10⁸`.**

Nearly a step function in `|Aut|`, not in `n` — mechanistically right, since the canonical search
space is governed by the automorphism group and `n` was only ever a proxy for it.

**Why this is a positive result:** it is *predictive*. A user can compute `|Aut|` in milliseconds and
know in advance whether the method will encode their graphs. Very few representation papers can tell
you their failure mode's governing parameter. Frame the 2.50 % Mutagenicity censoring rate as a
**characterised and predictable** limitation rather than an unexplained one.

> ### ❌ RETRACTED — *"it computes everywhere"* is **not** a differentiator
>
> Drafted here, measured by `[T06-subagent-01]`, withdrawn. Completion floor across all 15 cells,
> under the D14 reading in which a censored graph **does** carry an encoding:
>
> | representation | min completion |
> |---|---|
> | `adjacency`, `graph6`, `nauty_graph6`, `sparse6`, `sparse6_nauty`, `wl_subtree`, `size_null`, both IsalGraph arms | **1.0000** |
> | `min_dfs` | 0.9478 |
> | `agm_cam` | **0.0615** |
>
> **Eight representations complete on 100 % of every cell.** IsalGraph ties them; it does not lead.
> The `agm_cam` comparison is true and is the *only* comparison that flatters — selected, whether or
> not deliberately, from a field where almost everything else does as well or better. **It separates
> IsalGraph from `agm_cam` and `min_dfs` and from nothing else.**
>
> *(Under `t06_completion`'s count IsalGraph reads 0.9750 and third-worst, but that is the §15.4
> defect — a censored graph is retained with its greedy-min string, so D14's reading is the correct
> one. Both readings give the same verdict: a tie, not a lead.)*
>
> **What may still be said, in one clause:** `agm_cam` — the strongest small-`n` competitor on both
> claims — is computable on **6.15 % of Protein**, so *its* results are a small-graph statement. That
> is a scope note on a competitor, not a strength of ours.

---

## 6. 🔴 The red lines — where scoping becomes misleading

**These are not cautions. Each one is a framing that is technically defensible and would still be
wrong to use, and a reviewer who checks will find every one of them.**

| ❌ Do not write | Why it fails |
|---|---|
| *"IsalGraph clears the trivial size baseline on 5 of 5 Suite-2 datasets"* | **True under the UB reference and false under LB** — the verdict **inverts** on all five. Reporting the UB half alone is the single most damaging thing this paper could do, because the inversion is already documented and a reviewer recomputing it finds the omission. §10 / §14.1: the comparison is **undetermined**, and that is the honest word. |
| *"ρ ≈ 0.93 on sparse IAM demonstrates structural fidelity"* | Most of that ρ is the **size channel**. §17 shows the same arm at 0.26 within a fixed `n`. The paper itself now supplies the instrument that refutes this sentence. |
| *"competitive with the best representations"* on Claim B | It is best on **none** of 15 records, in either view. "Competitive" is not a scoping of that, it is a contradiction of it. |
| *"most compact among representations admitting a metric"* | **False.** `sparse6_nauty` is metric-admissible and beats it on 71 of 122 strata above `n = 20`. Measured, refuted, do not resurrect. |
| *"IsalGraph beats `agm_cam` on bits everywhere"* | `agm_cam`'s strata are **all at `n ≤ 12`** by its own scope guard. Any pooled win rate against it is a small-graph statement only, and must say so. |
| *"shorter than competitors in 32 % of strata"* | **State the predicate in the sentence.** 32 % is "positive median gap, significance ignored"; the significant figure is **0 %**. Four different numbers describe the same 122 strata (0 / 32 / 42 / 58 %) and they are not interchangeable. |
| Quoting `43 s/graph`, `≈ 520×`, `≥ 6.8 core-hours` | **Retracted as unprovenanced** (§11.4). The run that produced them left no artifact. |
| *"the fallback does not affect the correlation"* | It **does** — censored-touching pairs score ρ 0.3273 against 0.6095 clean at `n > 40`. What is true is narrower: the fallback does not *explain the collapse*. Report all three numbers (§4.1), never the Δ alone. |
| *"above n = 20, 91–99 % of strata are unresolved, so the field is indistinguishable"* | **Refuted by a five-minute test a reviewer will run.** "Unresolved" there is a statement about per-stratum **power**, not about equality — equal-`n` strata above 20 are thin. Pool them with a sign test and IsalGraph is significantly lower against **all four** admissible competitors: `min_dfs` 35 higher / 66 lower, p = 0.0027; `nauty_graph6` 43/67, p = 0.028; `sparse6_nauty` 30/80, **p = 2.0e-06**; `wl_subtree` 29/81, **p = 7.3e-07**. Many underpowered comparisons all leaning one way is *evidence*, not absence of evidence. |
| *"IsalGraph computes everywhere, unlike the competitors"* | **Eight representations also complete on 100 % of every cell.** Only `agm_cam` (6.15 % floor) and `min_dfs` (0.9478) are worse. Naming `agm_cam` alone selects the single flattering comparison from a field that mostly matches us. |
| *"N of M F2 cells are significant"* as evidence of success | **A rejection is against `H₀: Δ = 0` and can mean *significantly worse*.** On the 6-cell dry run, **16 of 26 directional rejections went against IsalGraph** — all six `B1e` rejections were losses, and `A1` split 10 longer / 9 shorter. A bare count reads as a win count and is closer to the opposite. Report rejections **split by row and by direction**. |
| Any F0/F1/F2 result restated more favourably than it came out | The confirmatory layer is pre-registered. Its value is precisely that it is reported unchanged; softening one sentence forfeits the protection for all of them. |

**The general rule:** a scoped claim must carry its scope **in the same sentence**, not in a later
limitations section. *"Most compact of the canonical-code representations"* is fair. *"Most
compact"*, with the qualifier moved to §7, is not — and the difference is what a reviewer is
checking for.

---

## 7. Suggested results-section order

1. **Completeness at scale** — zero collisions on 24.8 M pairs. Unscoped, unattackable. (§1)
2. **Compactness** — the `min_dfs` clean sweep, then the canonical-code framing with the edge-list
   concession stated immediately. (§2)
3. **The scaling asymmetry** — compactness up, correlation down. Own figure. (§3)
4. **Cost characterisation by `|Aut|`** — predictive, and a contribution. (§5)
5. **The size-channel diagnostic and the field-level ceiling** — presented as a methodological
   contribution, which is what it is. (§4)
6. **Pre-registered confirmatory results**, reported as they came out: F0 fires 4/5, F1 `d = 7`,
   the large-`n` extension is descriptive. **Do not bury this and do not soften it** — a
   pre-registered analysis reported unchanged is the strongest evidence of good faith the paper has,
   and it is what buys credibility for everything above.
7. **Limitations** — the size-null failures, the undetermined bracket, the `sparse6_nauty` loss.

**One more asset the comparisons cannot capture.** The instruction string is an **executable
program**; no serialisation competitor is. That is a categorical difference rather than a metric one,
it is where the novelty actually lives, and it is not adjudicated by ρ or by bit counts. If the
comparison sections read as narrow, this is the paragraph that reframes the contribution — and it
requires no scoping, because nothing else in the comparison set does it at all.

---

## 8. A framing that was proposed, tested, and must not be used

Recorded because it is attractive, because it nearly reached the PI, and because the test that kills
it is one any reviewer runs.

**The proposal.** Within equal-`n` strata above `n = 20`, 91–99 % of every competitor's
head-to-heads are *unresolved* — the marginal intervals overlap. Read one way this says: within a
fixed size the field is noise, so IsalGraph is not clearly worse, and the pooled "9 losses" are the
size channel rather than a real deficit.

**Why it fails.** *Unresolved* there is a statement about **per-stratum power**, not about equality.
Equal-`n` strata above 20 are thin — that is the same thinness §17 reports as "6 of 52 significant".
Counting how many individual strata resolve is the wrong summary of many small, consistent effects.
**Pool them and the picture reverses.** Sign test over per-stratum ρ differences, `n > 20`, UB
reference:

| competitor | strata | IsalGraph higher | lower | median Δρ | sign-test `p` |
|---|---:|---:|---:|---:|---:|
| `min_dfs` | 101 | 35 | 66 | −0.0521 | **0.0027** |
| `nauty_graph6` | 110 | 43 | 67 | −0.0447 | **0.028** |
| `sparse6_nauty` | 110 | 30 | 80 | −0.0723 | **2.0 × 10⁻⁶** |
| `wl_subtree` | 110 | 29 | 81 | −0.0692 | **7.3 × 10⁻⁷** |

**Significant against all four.** Strata within a dataset are disjoint graph sets, so the test is
valid; it weights every stratum equally regardless of pair count, which if anything understates the
large strata.

**The principle worth carrying beyond this instance:** *many underpowered comparisons all leaning the
same way is evidence, not absence of evidence.* Any framing that counts non-significant results as
support must survive pooling before it is used. Two other numbers in this ticket have the same shape
and were handled correctly — the `equal_n` size null being *undefined* rather than missing, and
Claim A's four predicates over the same 122 strata.

**What may be kept from it.** The observation itself is real and belongs in the paper as *description
of the size channel*: the pooled `all_pairs` gap is larger than the within-`n` gap, so a meaningful
part of the head-to-head deficit is size agreement rather than structure. That is worth one sentence.
It is **not** a defence, and the median Δρ favouring the competitor in every single case must sit in
the same sentence.

---

## 9. The no-dominance framing — measured. It is REAL but it is not clean

Requested by the PI: *"we need not be best at everything — best against X on task A, worse against Y
on A, but better than Y on B."* That is the Pareto / no-dominance argument, it is standard, and it is
fair **if the data shows it**. So it was computed rather than assumed, including the
competitor-vs-competitor comparisons nothing had measured before.

Method: sign test over `(dataset, n)` strata at `n > 20`, α = 0.05 — the same test that refuted §8's
framing, applied symmetrically so it cannot flatter us. Script:
`.claude/notes/review/tasks/t06_dominance.py`.

### 9.1 Claim A — compactness, row beats column

| | isalgraph | min_dfs | nauty_g6 | sparse6_nauty | adjacency | graph6 | sparse6 |
|---|---|---|---|---|---|---|---|
| **isalgraph_pruned** | — | **WIN** | **WIN** | LOSS | **WIN** | **WIN** | LOSS |
| min_dfs | LOSS | — | LOSS | LOSS | LOSS | LOSS | LOSS |
| nauty_graph6 | LOSS | WIN | — | LOSS | LOSS | n/a | LOSS |
| sparse6_nauty | WIN | WIN | WIN | — | WIN | WIN | LOSS |
| sparse6 | WIN | WIN | WIN | WIN | WIN | WIN | — |

### 9.2 Claim B — GED correlation, row beats column

| | isalgraph | min_dfs | nauty_g6 | sparse6_nauty | wl_subtree |
|---|---|---|---|---|---|
| **isalgraph_pruned** | — | LOSS | **tie** | LOSS | LOSS |
| min_dfs | WIN | — | WIN | tie | LOSS |
| nauty_graph6 | tie | LOSS | — | LOSS | LOSS |
| sparse6_nauty | WIN | tie | WIN | — | LOSS |
| wl_subtree | WIN | WIN | WIN | WIN | — |

### 9.3 What the two matrices give us — and what they take away

**✅ The cross-over the PI described EXISTS, against the comparator that matters most:**

> **Against `min_dfs` — which `competitors.md` §2 calls "the single most important comparator" — the
> two representations trade: IsalGraph is decisively more compact (112 of 112 strata, +214.8 bits),
> `min_dfs` correlates better with GED. Neither dominates the other.**

**✅ And IsalGraph weakly dominates `nauty_graph6`:** wins compactness, ties correlation. That is a
clean, defensible, unqualified statement about one named competitor.

**❌ But `sparse6_nauty` dominates IsalGraph on both axes** — more compact *and* better correlated,
both significant. It is also a complete invariant and carries an admissible metric. **This must be
conceded in the paper. Do not build a Pareto claim that omits it**, because the matrix above is four
lines of code for a reviewer holding our own artifacts.

**So: IsalGraph is NOT on the Pareto frontier of (compactness, GED correlation) among
metric-admissible representations at `n > 20`. `sparse6_nauty` is, alone.**

### 9.4 The structural finding that rescues the framing, and it is genuinely interesting

**Neither axis-leader is evaluable on the other axis.**

- **`sparse6` is the best compressor** — it beats every representation including `sparse6_nauty` — and
  it is **k-excluded**: no candidate distance passed the metric axioms (F3 = 1/50). It has **no
  Claim B at all**.
- **`wl_subtree` is the best correlator** — it beats every representation on Claim B — and it raises
  **`BitCountUndefined`**. It has **no Claim A at all**.

So the two winners each win an axis they can only be measured on because the *other* axis does not
apply to them. **That is a real no-dominance result and it is about the field, not about us:**

> **No representation is simultaneously best at both, and the two that lead each axis are undefined
> on the other. Compactness, metric admissibility and bit-countability do not co-occur.**

That sentence is true, it is measured, it is interesting, and it costs nothing to say. It positions
the contribution as *a point on a trade-off surface* rather than a winner — which is exactly the PI's
framing, honestly obtained.

### 9.5 Frozen wording

> *"No single representation leads on both axes. The most compact serialisation admits no metric
> satisfying the distance axioms; the best-correlating representation admits no bit count. Among
> those measurable on both, IsalGraph trades with min-DFS — decisively more compact, less well
> correlated — and dominates nauty-graph6. It is itself dominated by nauty-sparse6, which is both
> more compact and better correlated."*

**Say the last clause.** A Pareto framing that omits the one representation dominating us is the most
checkable dishonesty available in this paper, and conceding it is what makes the min-DFS trade read as
a finding rather than a selection. **It is also why the differentiator has to be categorical**
(§7's executable instruction string) rather than a point on this surface: on these two axes we are
dominated, and no amount of scoping changes that.

### 9.6 🔴 CORRECTION to §9.1–9.5 — the Claim B verdicts are BRACKET-DEPENDENT

`[T06-subagent-01]` observed that `lb` and `ub` are two bounds on the **same pairs**, so pooling them
in a sign test enters every stratum twice and breaks the independence the test assumes. **That
defect is in §9's matrix**, which keyed strata on `(dataset, n, reference)` and treated the two
bounds as separate observations. Recomputed split by reference:

| IsalGraph vs | Claim A | Claim B under **LB** | Claim B under **UB** |
|---|---|---|---|
| `min_dfs` | **WIN** | **tie** | LOSS |
| `nauty_graph6` | **WIN** | **tie** | LOSS |
| `sparse6_nauty` | LOSS | **LOSS** | **LOSS** |
| `wl_subtree` | n/a (`BitCountUndefined`) | LOSS | LOSS |

**What changes:**

- **Under LB, IsalGraph weakly dominates BOTH `min_dfs` and `nauty_graph6`** — wins compactness, ties
  correlation. §9.3 reported the `min_dfs` relationship as a cross-over; that is the **UB** reading.
- **Under UB it trades with both** — wins compactness, loses correlation. A genuine cross-over.
- **`sparse6_nauty` dominates IsalGraph under BOTH bounds.** That concession is **robust** and §9.3's
  headline stands unchanged.
- `wl_subtree` beats it under both but has no Claim A, so it remains incomparable rather than
  dominating.

**What this means, and it is the ticket's recurring theme arriving once more:** the competitor
verdicts on Claim B are **not invariant to where inside the proven bracket the truth lies** — which
is exactly what F1 measured when it returned `d = 7 of 10` (§18.3), and exactly what §10 found for
the size null. **Reporting the LB verdict alone would be the mirror image of the UB-only size-null
cherry-pick already on the red-line list**, and equally checkable.

**Corrected frozen wording, replacing §9.5's:**

> *"No single representation leads on both axes: the most compact serialisation admits no metric
> satisfying the distance axioms, and the best-correlating representation admits no bit count. Among
> those measurable on both, IsalGraph is decisively more compact than min-DFS and nauty-graph6, and
> its correlation against them is **bracket-dependent** — indistinguishable under the lower bound,
> weaker under the upper. It is dominated on both axes by nauty-sparse6."*

**Every Claim B verdict in the paper must carry both bounds.** Not as a hedge — as the finding. The
bracket being wide enough to flip competitor verdicts *is* a result, it is pre-registered (F1), and it
is measured on 21.7 M pairs.

---

## 10. The below-null result splits in two, and only one half is rescuable

Verified from `DECISION_SUMMARY.md`'s Claim B table at 10 of 15 cells. `[T06-subagent-01]` found that
**every below-null record is `exact` or `lb`, and not one is `ub`.** Confirmed — but grouping `exact`
with `lb` blurs the distinction the paper turns on, because **`exact` is not part of the bracket.**

| reference | below null | clears | nature |
|---|---|---|---|
| **`exact`** (Suite 1) | **2 of 2** — `aids` 0.3266 vs 0.7863, `linux` 0.4850 vs 0.7097 | 0 | **Ground truth. No bracket argument touches this.** |
| `lb` (Suite 2) | 4 of 4 | 0 | bracketed |
| `ub` (Suite 2) | **0 of 4** | 4 | bracketed |

### 10.1 The Suite-2 half IS rescuable, and honestly

On the same pairs the arm falls below its null under LB and clears it under UB, on **4 of 4**
datasets. That is not a failure — it is **undetermined**, and it is §10's pilot inversion reproducing
at full cohort. The correct sentence is *"whether the representation beats the trivial size baseline
on Suite 2 depends on where inside the proven bracket the truth lies, and the bracket is too wide to
say"* — which is F1's `d = 7 of 10` restated on a fourth instrument.

**This is a legitimate and much better position than "it fails the trivial baseline".** Use it. But
it comes with the standing red line: **report both bounds.** *"Clears the null on 4 of 4 Suite-2
datasets"* is true of the UB half alone and is the single most damaging sentence available in this
paper. (`2/linux/ub` clears by 0.3612 against 0.3399 — a margin of **0.021**. Thin enough that it
should not be leaned on even within the UB reading.)

### 10.2 The Suite-1 half is NOT rescuable, and must be conceded

`aids` and `linux` are measured against **exact GED**. There is no bracket, no bound, no
interpolation — the trivial `|n_i − n_j|` baseline correlates with ground-truth graph edit distance
**better than the representation does**, by −0.4597 on `aids` and −0.2247 on `linux`.

**No framing repairs this and none should be attempted.** The pilot (§14.1) found the same on 4 of 5
Suite-1 datasets, with only `iam_letter_low` clearing; the three remaining Suite-1 cells are still
computing and will tell us whether that reproduces. **Expect it to.**

> **Frozen wording:** *"On Suite 1, where ground-truth GED is exact, the size baseline outperforms the
> representation on `aids` and `linux`. On Suite 2 the comparison is undetermined: the verdict
> inverts across the proven bracket on every dataset measured."*

Two sentences, both true, and the first one must come first. **Leading with the Suite-2 "undetermined"
and leaving the Suite-1 result to a limitations section is exactly the move a reviewer checks for** —
the exact-GED result is the cleaner measurement and burying the cleaner measurement is what makes an
omission look deliberate.

---

## 11. 🔴 THE BENCHMARK IS SIZE-DOMINATED — and this reframes the whole comparison

Found by `[T06-subagent-01]` while correcting an over-generalisation; **independently reproduced by
the orchestrator from raw matrices, every cell.** It is the most consequential framing result in the
ticket.

### 11.1 The measurement

`ρ(|n_i − n_j|, reference GED)` — the trivial baseline against ground truth, with no representation
involved at all:

| reference | range of the size null |
|---|---|
| **Suite-2 bracket** (10 datasets) | **0.8789 – 0.9971** — seven of ten exceed **0.96** |
| **Suite-1 exact** (5 datasets) | **0.7097 – 0.9195** |

On `coil_del`, **`|n_i − n_j|` alone predicts the GED bracket at ρ = 0.9971.** On the IAM Letter
family it predicts **exact** GED at ρ ≈ 0.92.

### 11.2 What follows, and it is a contribution rather than an excuse

> **On the standard GED benchmarks this literature uses, node-count difference alone achieves
> ρ = 0.71–0.997 against ground-truth graph edit distance. Any representation's correlation with GED
> on these datasets is therefore measuring size agreement more than structural fidelity.**

Three consequences, all defensible:

1. **It explains why nothing beats the null.** Where the target is 99.7 % predicted by node count,
   beating node count is close to tautologically hard. This does not excuse IsalGraph — it still
   loses to competitors — but it means *no representation* looks good, and the benchmark cannot
   distinguish much.
2. **It explains the LB/UB inversion mechanically.** The UB tracks size *less* tightly than the LB, so
   arms carrying structure look better against it. The inversion is not noise; it is the reference
   becoming marginally less size-like.
3. **It is a fifth, and the most direct, detection that Suite 2's bracket is uninformative.** F1's
   `d = 7 of 10`, the competitor verdicts flipping, §10's pilot inversion and its full-cohort
   reproduction all *infer* uninformativeness from disagreement between instruments. **This one
   measures how much of the reference is explained by node count alone.**

### 11.3 What it does NOT rescue

**The Suite-1 exact result stands.** Even where the null is 0.71 (`linux`) and 0.79 (`aids`) rather
than 0.99, IsalGraph is below it — by −0.2247 and −0.4597. A size-dominated benchmark makes the bar
low; **we are still under it on four of five.** Say both.

### 11.4 Frozen wording

> *"On these benchmarks the reference itself is size-dominated: node-count difference alone attains
> ρ = 0.71–0.997 against ground-truth GED, exceeding 0.96 on seven of ten Suite-2 datasets. Correlation
> with GED on this data therefore measures size agreement more than structural fidelity — for every
> representation, ours included. We report the within-`n` decomposition (§17) because it is the only
> view in which the two can be separated."*

**That paragraph does more for the paper than any comparison result in the ticket**, because it turns
the central negative into a statement about the evaluation protocol — one that is true, measured on
21.7 M pairs, and useful to everyone working on graph-distance surrogates. **It is also the strongest
possible motivation for §17's within-`n` analysis**, which stops looking like a defensive slice and
starts looking like the correct instrument.

### 11.5 The correction that produced it, recorded because the lesson is general

The subagent had reported *"all four `ub` records clear"* from **four landed cells that happened to be
four inverters**; the three Letter datasets had not landed and do **not** invert. Computing all ten
gives `lb` below on **10 of 10**, `ub` below on **3 of 10**, inverting on **7 of 10** — so the bracket
argument rescues seven datasets and leaves the Letter family below its null under *every* reference,
exact included.

**Same shape as reading a rejection count without its composition**, flagged by the same agent hours
earlier. The point estimates required no bootstrap and were available the whole time. **Characterise
the complete set before describing any of it.**

### 11.6 ⚠ CORRECTION to §11.2 item 3 — the high null is the DATASETS, not the bracket

`[T06-subagent-01]` withdrew its own "the bracket is nearly a size measurement" framing and I am
withdrawing §11.2's item 3 with it. **Independently reproduced, every cell.**

The three Letter datasets have **identical cohorts** across both suites — same graph ids, same
695,610 / 784,378 / 2,118,711 pairs — so exact, LB and UB are comparable **on the same pairs**.
`ρ(|n_i − n_j|, reference)`:

| dataset | **exact** | **lb** | **ub** | lb − exact | ub − exact |
|---|---:|---:|---:|---:|---:|
| `iam_letter_low` | 0.9139 | 0.9804 | 0.7482 | **+0.0664** | **−0.1657** |
| `iam_letter_med` | 0.9146 | 0.9740 | 0.7363 | **+0.0594** | **−0.1782** |
| `iam_letter_high` | 0.9195 | 0.9224 | 0.7080 | +0.0029 | **−0.2115** |

**Exact GED is itself ~0.92 size-dominated on Letter.** So the high size null is **not** a bracket
artefact — on these datasets graph edit distance genuinely *is* mostly a size difference, because IAM
Letter graphs vary chiefly in node count. §11.2's claim that this was "a fifth and most direct
detection that the **bracket** is uninformative" implied the bracket was uniquely bad. **It is not,
and that item is withdrawn.** §11.1–11.2's core claim survives unchanged and is if anything
strengthened: the *benchmark* is size-dominated, and that is a property of the data.

**What IS bracket-specific is cleaner and better than what either of us claimed:**

> **LB and UB straddle the truth in how size-dominated they are.** LB is **more** size-dominated than
> exact (+0.066, +0.059, +0.003); UB is **less**, consistently and substantially (−0.166, −0.178,
> −0.212).

**That is the mechanism of the inversion, measured rather than inferred.** An arm carrying structure
must look worse against a reference that is nearly pure size (LB) and better against one *less*
size-dominated than truth (UB) — which is exactly the 10-of-10-below / 3-of-10-below split, now with
a cause attached. It also explains why `iam_letter_high` is the one Letter dataset where LB ≈ exact
(+0.0029) and yet is still below its null under **all three** references: **there the bracket is not
the problem, the dataset is.**

**Use this instead of §11.2 item 3.** It is mechanistic, it predicts the direction of every inversion
observed, it is measured on identical pairs rather than argued from disagreement between instruments,
and it tells a reviewer something useful about BRANCH-FAST and IPFP rather than only about us.

**Method note worth carrying:** both this and the 10/10-vs-3/10 split came from **point estimates
computed directly**, needing no bootstrap. Where a question is about *direction* rather than
*resolution*, the point estimate answers it immediately — and waiting for intervals to characterise a
shape that is already determined costs time for nothing.

---

## 12. THE CENTRAL TABLE — MRM standardised coefficients, and what to write about Claim B

The instrument that **controls** for the size confound rather than stratifying it away. Two of these
rows were verified by the orchestrator directly from the partials (`nperm = 9999`, production).

| suite / dataset | ref | **β_lev** | **β_Δn** | β_density | R² |
|---|---|---:|---:|---:|---:|
| 1 `aids` | exact | +0.2314 | **+0.8049** | +0.0281 | 0.810 |
| 1 `iam_letter_low` | exact | **+0.5624** | +0.3537 | +0.1369 | 0.964 |
| 1 `linux` | exact | +0.3551 | **+0.6689** | +0.0408 | 0.742 |
| 2 `aids_graphedx` | lb | +0.1552 | **+0.8694** | −0.0028 | 0.964 |
| 2 `aids_graphedx` | ub | +0.3519 | **+0.6445** | −0.2230 | 0.678 |
| 2 `grec` | lb | +0.1764 | **+0.8649** | −0.0206 | 0.986 |
| 2 `grec` | ub | +0.4475 | **+0.6531** | −0.3734 | 0.812 |
| 2 `linux` | lb | +0.2854 | **+0.8886** | −0.0517 | 0.923 |
| 2 `linux` | ub | +0.2985 | +0.3954 | −0.1258 | **0.232** ⚠ |
| 2 `protein` | lb | +0.1478 | **+0.8161** | +0.0705 | 0.946 |
| 2 `protein` | ub | **+0.9869** | **−0.1177** ⚠ | −0.2609 | 0.802 |

### 12.1 The frozen Claim B sentence

> **"The canonical string contributes significant incremental information about graph edit distance
> beyond node-count and density difference — β_lev = 0.15–0.36, p < 0.001 on 13 of 14 fits — but node
> count difference carries 3–6× the weight (β_Δn = 0.65–0.89). Even in the model that adjudicates the
> confound directly, size does most of the work and the representation adds a significant minority
> share."**

**Both halves, one sentence.** β₁ reported alone is a coefficient without its context — the sixth
instance today of a number that inverts in meaning when its companion is withheld. The size
coefficient exceeds Levenshtein's in **9 of 11** fits.

**This also reconciles the two Claim B results that look contradictory** and must be stated together:

- the **size null** compares *marginal* predictors and Levenshtein **loses** on 4 of 5 Suite-1 datasets;
- the **MRM** asks whether Levenshtein adds anything *given* size and density, and it **does**, on 13 of 14.

Neither refutes the other. A predictor can be worse standalone and still carry independent
information, and saying both is more informative — and more obviously honest — than either alone.

### 12.2 The straddle, now visible INSIDE one model — the strongest methodological result in the ticket

On every dataset carrying both bounds, **β_Δn falls and β_lev rises going LB → UB**:

| dataset | β_Δn lb → ub | β_lev lb → ub |
|---|---|---|
| `aids_graphedx` | 0.869 → 0.645 | 0.155 → 0.352 |
| `grec` | 0.865 → 0.653 | 0.176 → 0.448 |
| `linux` | 0.889 → 0.395 | 0.285 → 0.299 |
| `protein` | 0.816 → −0.118 | 0.148 → 0.987 |

**4 of 4.** Every earlier detection compared *instruments*; here both predictors sit in **one
regression on identical pairs**, and the weight transfers from size to structure exactly as the
measured −0.18 size-domination gap predicts. **The mechanism is no longer inferred — it is fitted.**

This was a **prediction made from one instrument and confirmed on another**, which is worth more than
any number of instruments agreeing. It is the paper's best evidence that the LB/UB disagreement is
structural rather than noise.

### 12.3 Two internal consistency checks that were not constructed

1. **`iam_letter_low/exact` is the only fit where Levenshtein dominates** (0.5624 vs 0.3537) — and it
   is **the only dataset that cleared its size null** (+0.0139). Two independent instruments, the
   same lone exception.
2. **`protein/ub` has the largest β_lev** (0.9869) — and **the largest UB size-null excess** (+0.4094).
   Same cell again.

Nothing was tuned to produce either. **Worth one sentence in the paper**: independent instruments
agreeing on which cells are exceptional is evidence the pipeline measures something real.

### 12.4 ⚠ Two cells not to build on

- **`linux/ub`, R² = 0.232** — the model explains almost nothing; coefficients weakly identified on
  3,916 pairs.
- **`protein/ub`, β_Δn = −0.118** — the only negative size coefficient in the set, and the most
  extreme fit.

Both are the smallest-R² / most extreme cells. **Do not quote either without its CI**, and do not let
`protein/ub`'s β_lev = 0.9869 become the headline: it is one cell, it is the outlier, and it is the
one a reviewer will check first.

### 12.5 Scope, until the tier-3 pair lands

All 14 fits are tier-1/tier-2. **`mutagenicity` and `coil_del` have not reported** — the two largest
datasets, and exactly where §17 says the correlation is noise. **If β_lev collapses anywhere it is
there.** Until then this is a claim about 8 of 10 Suite-2 and 3 of 5 Suite-1 datasets, and it must
say so.

---

## 13. THE CLEANEST CONTROL IN THE TICKET — the IAM Letter family at three distortion levels

Found by `[T06-subagent-01]` when `suite1/iam_letter_high` landed. **Independently checked and it is
sharper than first framed.**

IAM Letter LOW / MED / HIGH are **the same generator at three distortion levels**, so data source,
labelling and construction are held fixed and **only the graphs differ**. That makes this a
*within-family* control, not a cross-dataset comparison — its trend cannot be attributed to different
cohorts having different properties.

### 13.1 Node count barely moves. Structure does.

| dataset | graphs | mean `n` | sd `n` | mean edges |
|---|---:|---:|---:|---:|
| `iam_letter_low` | 1,180 | 4.07 | 1.17 | 3.07 |
| `iam_letter_med` | 1,253 | 4.11 | 1.16 | 3.17 |
| `iam_letter_high` | 2,059 | 4.58 | 1.25 | **4.56** |

Mean node count rises 4.07 → 4.58 (+12 %) with essentially unchanged spread; **mean edge count rises
3.07 → 4.56 (+49 %)**. The family adds *structure*, not *size*.

### 13.2 The size baseline is FLAT. The representation collapses.

| dataset | ρ(Lev, exact) | size null | excess | β_lev | β_Δn | ratio |
|---|---:|---:|---:|---:|---:|---:|
| `iam_letter_low` | **0.9278** | 0.9139 | **+0.0139** | +0.5624 | +0.3537 | **0.6×** |
| `iam_letter_med` | 0.8833 | 0.9146 | −0.0313 | *pending* | — | — |
| `iam_letter_high` | **0.6660** | 0.9195 | **−0.2536** | +0.2696 | **+0.7507** | **2.8×** |

> **Holding the generator fixed and adding structural distortion, the trivial `|n_i − n_j|` baseline
> stays flat at ρ ≈ 0.92 while the canonical string's correlation falls from 0.93 to 0.67. In the
> regression, β_lev halves and β_Δn doubles over the same range, and the size-null verdict flips.**

### 13.3 Why this is the strongest single piece of evidence in the ticket

1. **It is a within-family control.** Nothing varies but the graphs.
2. **It rebuts the obvious defence.** One could argue the size null wins because these benchmarks are
   size-dominated (§11) — but here **node count is nearly constant** and the null *still* holds at
   0.92 while the arm collapses. The failure is not that size is doing the work by default; it is
   that **the representation stops tracking GED exactly as structure becomes the thing that matters.**
3. **Two independent instruments move together** — the marginal size-null excess and the MRM
   coefficient ratio — on the same three datasets, monotonically.
4. **It is the third appearance of one story**, after §17's within-`n` collapse and §11's
   size-domination: the representation tracks edit distance where there is little structure to track.

### 13.4 What to do with it

**Put it in the paper as its own small table.** It is the most economical way to state the central
finding, it is a controlled comparison rather than a pooled one, and a reviewer can check it in three
rows. It is also **the honest framing of the limitation** — far better than a generic "performance
degrades on harder data", because it names *what* gets harder and shows the baseline unaffected.

### 13.5 ✅ COMPLETE — the monotone held, and both instruments cross at the same dataset

`iam_letter_med` landed. Orchestrator-verified from the partial:

| dataset | β_lev | β_size | **ratio** | R² | size-null excess |
|---|---:|---:|---:|---:|---:|
| `iam_letter_low` | **+0.5624** | +0.3537 | **0.63×** — Lev dominates | 0.964 | **+0.0139** clears |
| `iam_letter_med` | +0.4610 | +0.5086 | **1.10×** — parity | 0.954 | −0.0313 ≈ 0 |
| `iam_letter_high` | +0.2696 | **+0.7507** | **2.78×** — size dominates | 0.913 | **−0.2536** below |

**β_lev falls monotonically. β_size rises monotonically. The ratio crosses 1.0 at MED** — and the
**size-null excess crosses zero at the same dataset.** Two independent instruments, one monotone
trend, the same crossover, on a family where nothing varies but the graphs.

**This pins a number the pooled tables cannot give:** the representation stops paying its way on this
family **between LOW and MED distortion** — not "somewhere above `n = 20`", which is where the
cross-dataset aggregates put it. A within-family crossover is a much more useful statement of the
limit than a pooled threshold, because it names the *condition* rather than a coincidental size.

**The two instruments agree even at the ambiguous point:** MED is *parity* by the MRM (1.10×) and
*marginally below* by the size null (−0.0313). Those disagree only in the sense that a ratio of 1.1
and an excess of −0.03 are both "about zero" — which is the agreement, not a discrepancy.

**Write it as the limitation.** *"On a controlled family at three distortion levels, the structural
predictor's share of graph edit distance falls from dominant to parity to minority, crossing at the
middle level."* That is a far stronger and more honest limitation statement than any pooled
threshold, and it is checkable in three rows.

### 11.7 The straddle, finally stated correctly — 6 of 6 on two quantities, with a dose–response

This **replaces** §11.2 item 3, withdrawn in §11.6. Sixteen production fits, six datasets carrying
both bounds. Orchestrator-verified arithmetic.

| dataset | β_lev lb → ub | rises | size/Lev lb → ub | falls |
|---|---|:---:|---|:---:|
| `aids_graphedx` | 0.1552 → 0.3519 | ✓ | 5.6× → 1.8× | ✓ |
| `grec` | 0.1764 → 0.4475 | ✓ | 4.9× → 1.5× | ✓ |
| `iam_letter_low` | 0.2788 → 0.3082 | ✓ | 2.5× → 1.8× | ✓ |
| `iam_letter_med` | 0.2862 → 0.2974 | ✓ | 2.6× → 2.1× | ✓ |
| `linux` | 0.2854 → 0.2985 | ✓ | 3.1× → 1.3× | ✓ |
| `protein` | 0.1478 → 0.9869 | ✓ | 5.5× → 0.1× | ✓ |

**6 of 6 on both**, sign test **p = 0.031**.

> ⚠ **The two rows are ONE result, not two.** β_lev rising and the ratio falling are the same weight
> transfer seen from two sides — the ratio has β_lev in its denominator. **Do not report "p = 0.031
> twice" or compound them**; quote one sign test and note that the second view agrees by
> construction. Flagged by `[T06-subagent-01]` against its own finding.

**The ratio is the better statement of the two.** It is a *within-model* comparison — β_size against
β_lev inside one regression — so it is immune to any objection that β_lev moved because the response
was rescaled between bounds. Quote the ratio; keep β_lev beside it.

#### The dose–response is what makes this a mechanism rather than a coincidence

The further a bound sits from the truth in size-domination, the more weight transfers:

> **Spearman(LB ratio, ratio drop) = 0.886, p = 0.019; Pearson = 0.950, n = 6.**

`protein` and `aids_graphedx` carry the largest LB ratios (5.5×, 5.6×) and show the largest collapses
(→0.1×, →1.8×); `iam_letter_low` and `iam_letter_med` carry the smallest (2.5×, 2.6×) and barely move
(→1.8×, →2.1×). **A direction alone could be an artefact of the bounds' arithmetic; a graded
magnitude that scales with the size-domination gap is a mechanism.** Six datasets is a small n and
the p-values are correspondingly modest — say so — but a predicted direction, confirmed on 6/6 of two
quantities, with a significant dose–response, is a different class of evidence from an observed
correlation.

#### Frozen statement

> *"BRANCH-FAST's lower bound is more size-dominated than exact GED (+0.06) and IPFP's upper bound is
> less (−0.18). Every downstream verdict inherits that: on all six datasets carrying both bounds, the
> structural coefficient rises and the size-to-structure ratio falls from LB to UB, and the size of
> that shift scales with how far the bound sits from the truth (ρ = 0.89)."*

**This is a finding about BRANCH-FAST and IPFP**, useful to anyone bracketing GED, and only
incidentally about us. It also explains — mechanically, not by appeal — every bracket-dependent
verdict in the ticket: F1's `d = 7 of 10`, the size-null inversion on 7 of 10, and the competitor
verdicts flipping between bounds.

**What does NOT change:** size still dominates on **9 of 11** landed `lb`/`exact` fits, and the two
exceptions remain `iam_letter_low`/exact and `protein`/ub — the same two cells the size-null
instrument singled out independently.

### 12.6 The ONE non-significant β₁ is the strongest instance of the size story, not a weak spot

Verified from the partial. `aids_iam`, 1,638,955 pairs:

| ref | β_lev | β_size | ratio | R² | p |
|---|---:|---:|---:|---:|---:|
| **lb** | **+0.0976** | **+0.9066** | **9.29×** | **0.9982** | **0.1502 — not significant** |
| ub | +0.4032 | +0.6237 | 1.55× | 0.9586 | 0.0002 — significant |

**β₁ fails to clear because there is nothing left for it to explain.** `|n_i − n_j|` alone carries
β = **+0.9066** and the model reaches **R² = 0.9982**. The residual the structural predictor competes
for is **≈ 0.2 % of the variance**. **9.29× is the most extreme ratio in the entire set, and it
belongs to the one cell where β₁ does not reach significance.**

**So the reading inverts.** `aids_iam`/lb was first called "an outlier, not a trend". It is neither —
**it is the trend taken to its limit**:

> **β₁ is significant wherever size leaves it room, and the single place it is not is the place size
> leaves it none.**

**Write it that way.** A reader shown "18 of 19 significant" will ask about the exception, and the
exception is *more* supportive of the size story than any of the 18. Volunteering it is both honest
and stronger than omitting it — and a reviewer who checks will find `R² = 0.998` sitting beside the
non-significant p and reach this reading anyway.

**It is also the largest straddle drop measured: 9.29× → 1.55×**, same dataset, same pairs. Exactly
what §11.7's dose–response predicts — the further a reference sits from the truth in size-domination,
the more weight transfers — and `aids_iam`/lb is the furthest point in the set.

**Aggregate over 19 landed production fits: 18 significant, 1 not; size exceeds Levenshtein on
17 of 19 (89 %).** The earlier "9 of 11" was the same picture at smaller `n`.

### 12.7 R² — a statement about BRANCH-FAST, not independent evidence

`R²` falls LB → UB on **6 of 6** datasets:

| dataset | R² lb | R² ub |
|---|---:|---:|
| `iam_letter_low` | **0.991** | 0.628 |
| `iam_letter_med` | **0.988** | 0.627 |
| `grec` | **0.986** | 0.812 |
| `aids_graphedx` | **0.964** | 0.678 |
| `protein` | **0.946** | 0.802 |
| `linux` | **0.923** | 0.232 |
| `aids_iam` | **0.998** | 0.959 |

> ⚠ **NOT independent evidence.** This is §11.7's weight transfer seen a *fourth* way, on the same
> fits. Per §11.7's own correction, these views must not be compounded — **one result, one p-value.**

**But it carries one statement the coefficients alone do not, and it is about the bound rather than
about us:**

> **Under BRANCH-FAST's lower bound, node-count difference, density difference and one string
> distance jointly explain 92–99.8 % of the variance, with β_size ≈ 0.87–0.91. The LB matrix is very
> nearly a deterministic function of node count.**

That is worth one sentence to a reviewer assessing whether a BRANCH-FAST bracket can support a
correlation claim at all — and it is the cleanest form of §11's size-domination result, because R² is
a familiar quantity requiring no explanation of what a "size null" is.

**UB retains substantially more unexplained structure (23–81 %)**, which is why β_lev has more room
there. Same mechanism, more legible units.

### 11.8 ⚠ §11.7 SUPERSEDED — direction is 7/8, but the dose–response STRENGTHENED

`[T06-subagent-01]` found the first exception when `iam_letter_high` s2 landed with both bounds, and
was right that §11.7's "6 of 6" cannot stand. **Verified, and one of its conclusions inverts.**

| dataset | ratio lb → ub | drop | β_lev rises? |
|---|---|---:|:--:|
| `aids_iam` | 9.29 → 1.55 | **7.75** | ✓ |
| `protein` | 5.52 → −0.12 | **5.64** | ✓ |
| `aids_graphedx` | 5.60 → 1.83 | 3.77 | ✓ |
| `grec` | 4.90 → 1.46 | 3.44 | ✓ |
| `linux` | 3.11 → 1.32 | 1.79 | ✓ |
| `iam_letter_med` | 2.56 → 2.11 | 0.45 | ✓ |
| `iam_letter_low` | 2.50 → 1.81 | 0.69 | ✓ |
| **`iam_letter_high`** | **2.24 → 2.44** | **−0.20** | **✗** |

- **β_lev rises lb→ub on 7 of 8.** Direction has one exception.
- **The ratio falls on 7 of 8.** Same exception.
- **R² falls on 8 of 8** — no exception. This is the robust half.

#### 🔴 The exception does NOT weaken the dose–response — it extends it

The subagent expected the counter-example to weaken §11.7's Spearman. **Recomputed over 8 pairs it
rises:**

> **Spearman(LB ratio, its drop) = 0.952, p = 0.0003** — up from 0.886, p = 0.019 over 6.

**Because `iam_letter_high` has the *smallest* LB ratio in the set (2.24×) and the smallest transfer
(−0.20).** It sits at the bottom-left of the same line, not off it. **The direction "exception" is
the low end of a continuum, not a violation of it:** where the lower bound is already close to the
truth in size-domination, there is almost nothing to transfer, and the observed transfer is
approximately zero — here marginally negative.

**So the mechanism is simpler than "7 of 8 with an unexplained exception":**

> **The magnitude of the weight transfer scales with how size-dominated the lower bound is
> (ρ = 0.95). Where the LB is least size-dominated the transfer is ≈ 0, and its sign is then
> arbitrary.**

`iam_letter_high` is also the one dataset where the extra variance accrues to **neither** predictor —
both coefficients fall (β_lev 0.3462 → 0.2996, β_size 0.7752 → 0.7314) while R² collapses
0.951 → 0.642. **That is consistent with the same reading:** with no transfer to make, the UB's extra
unexplained variance is simply unexplained.

#### Provisional — 2 of 10 bound-pairs are still outstanding

`coil_del` and `mutagenicity` have not reported both bounds. **Do not record 7/8 or ρ = 0.952 as
final.** The subagent's own caution is the right one and applies to this correction too: it
characterised at 6 of 6 when 10 pairs will exist, and this section characterises at 8. **The final
count goes in when all ten land, and not before.**

### 11.9 🔴 RETRACTED — the dose–response was an artefact of the identity. §11.7 and §11.8 both wrong

**Caught by `[T06-subagent-01]`, verified independently by the orchestrator, and retracted in full.**
This was the orchestrator's own construction, promoted twice as "the strongest methodological result
in the ticket". **It is not a result at all.**

**The defect.** `drop = ratio_lb − ratio_ub`, so `Spearman(ratio_lb, drop)` correlates **X against
X − U**. That is positively correlated *by construction*, whatever `U` does. Testing it against
ρ = 0 is the wrong null.

| test | result | verdict |
|---|---|---|
| `Spearman(ratio_lb, drop)` — as recorded in §11.7/§11.8 | ρ = **+0.9524**, p = 0.00026 | **meaningless** |
| `Spearman(ratio_lb, ratio_ub)` — identity-free | ρ = **−0.4762**, p = **0.233** | **nothing** |
| **Permutation null** — shuffle `ratio_ub`, keep the identity, 20,000 draws | mean ρ = **+0.899**, 95 % **[+0.762, +0.976]** | — |
| observed +0.9524 against that null | **permutation p = 0.29** | **not significant** |

**If `ratio_ub` were pure noise with the same marginals, the expected correlation would be +0.90.**
The observed +0.95 falls *inside* the null's 95 % interval. **There is no dose–response.**

#### What actually survives, on the 8 landed bound-pairs

| claim | count | exact sign-test p | |
|---|---|---|---|
| **R² falls lb → ub** | **8 / 8** | **0.0078** | ✅ **significant** |
| β_lev rises lb → ub | 7 / 8 | 0.0703 | ❌ not significant |
| ratio falls lb → ub | 7 / 8 | 0.0703 | ❌ not significant — **and the same test as the row above** |

> **Only the R² half has statistical support.** It carries the BRANCH-FAST sentence — *under the
> lower bound the three predictors explain 92–99.8 % of the variance* — and that stands.
>
> **The weight transfer must be stated descriptively and with no p-value:** *"the size-to-structure
> ratio fell on 7 of 8 bound-pairs."* Nothing more. And β_lev-rises and ratio-falls are **one test,
> not two** — the same correction §11.7 applied to itself an hour earlier, missed here.

**A defensible reformulation exists** — regress `ratio_ub` on `ratio_lb` and test slope < 1 — but with
8 points and an identity-free ρ of −0.48 it is unlikely to survive, and the two outstanding pairs
will not rescue it. **Do not attempt it in the paper without checking.**

#### The mechanism claim is not dead, it is unquantified

The **direction** is still real and still predicted the β₁ finding before it was measured. What is
gone is the *graded magnitude*, which was the part that made it "a mechanism rather than a
coincidence" (§11.7's own words). **Without the dose–response that argument is unavailable**, and the
straddle should be presented as a consistent direction with a measured exception, not as a
quantified law.

### 12.8 🔴 RETRACTED — §12.6's `aids_iam` reading is not identifiable. Collinearity screen added

**Caught by `[T06-subagent-01]` when `coil_del`/ub returned β_lev = +1.4892 with β_size = −0.5797** —
a standardised coefficient above 1 with a sign flip on its competitor, which is collinearity rather
than a finding. Screening all 15 datasets then retracted an earlier claim.

**Orchestrator-verified** `r(Lev, |Δn|)` over the analysed pairs (VIF below is the subagent's
full three-predictor figure; the two-predictor lower bound from `r` alone agrees):

| dataset | pairs | r(Lev, \|Δn\|) | VIF lev | |
|---|---:|---:|---:|---|
| **`aids_iam`** | 1,638,955 | **+0.9612** | **15.28** | **SEVERE** |
| **`coil_del`** | 7,603,050 | **+0.9355** | **11.15** | **SEVERE** |
| `mutagenicity` | 8,158,780 | +0.8442 | 2.58 | ok |
| `protein` | 161,596 | +0.7756 | 2.51 | ok |
| the other 11 | — | +0.18 … +0.84 | 1.03–3.42 | ok |

**Exactly 2 of 15 exceed the conventional VIF > 10 threshold — and they carried the two most-quoted
D4 observations in this document.**

#### What is retracted

**§12.6 claimed `aids_iam`/lb was "not an outlier but the trend taken to its limit"** — β₁ = +0.0976
against β_size = +0.9066 at R² = 0.998 — read as *"β₁ fails to clear because size leaves it nothing
to explain"*. **With the predictors correlated at r = 0.96 and VIF 15.3, that split is not
identifiable.** The model cannot separate them; assigning 0.91 to size and 0.10 to Levenshtein is
one arbitrary point in a wide equivalence class. **The observation was real; the interpretation was
not. §12.6 comes out.**

`coil_del`/ub is the same defect wearing its obvious face — β_lev past 1, β_size negative, at
R² = 0.918.

#### What survives, and it is STRONGER

Restricting to the **19 identifiable fits** (VIF ≤ 10):

| | all 23 | **identifiable 19** |
|---|---|---|
| size > Lev | 20/23 | **17/19** |
| β₁ significant | 22/23 | **19/19** |

**β₁ is significant on 19 of 19 identifiable fits** — the one non-significant fit was `aids_iam`/lb,
now excluded as unidentifiable. **Both halves of §12.1's headline get stronger, not weaker.**

#### The straddle after BOTH corrections — one claim stands

Restricted to the **7 identifiable bound-pairs**:

| claim | count | sign-test p | |
|---|---|---|---|
| **R² falls lb → ub** | **7/7** | **0.0156** | ✅ **significant** |
| β_lev rises lb → ub | 6/7 | 0.125 | ❌ |
| ratio falls lb → ub | 6/7 | 0.125 | ❌ same test |

> **After the identity artefact (§11.9) and now collinearity, exactly one straddle claim stands:
> R² falls LB → UB on 7 of 7 identifiable pairs, p = 0.0156.** It is the half carrying the
> BRANCH-FAST sentence, and **it is untouched by either defect because R² does not depend on how the
> model splits credit between collinear predictors.**

**Report the transfer descriptively, with no p-value and no dose–response.**

#### Screen before interpreting any coefficient

**Add a VIF column to every MRM table.** A standardised β above 1, or a sign flip on a competitor
predictor, is a collinearity signature and not a result. `mutagenicity` is *not* collinear
(VIF 2.58), so its fit is trustworthy when it lands.

### 12.9 ⚠ THE LARGEST DATASET INVERTS THE HEADLINE — and it is identifiable

**All 15 cells landed. Orchestrator-verified from the partial.** `mutagenicity`, 8,158,780 pairs,
4,040 graphs, `r(Lev,|Δn|) = 0.8442`, **max VIF 3.94 — identifiable**:

| ref | β_lev | β_size | ratio | R² | p |
|---|---:|---:|---:|---:|---:|
| lb | **+0.5229** | +0.3622 | **0.69×** | 0.937 | 0.0005 |
| ub | **+0.7303** | +0.2641 | **0.36×** | 0.864 | 0.0005 |

**Levenshtein dominates size on the largest dataset in the cohort, under BOTH bounds** — by 1.4× and
2.8×. Not marginal, not an unidentifiable fit, not a bound artefact.

#### This must not be smoothed, and it must not be over-claimed

**Both statements are true and both belong in the paper:**

> **"Node-count difference carries more weight than the canonical string on 17 of 21 identifiable
> fits. The four exceptions are not scattered: two of them are `mutagenicity` — the dataset with the
> most pairs, the most graphs and the most censoring — where Levenshtein dominates under both
> bounds."**

**Why saying both is the only defensible option.** Omitting the exception is indefensible: **a
reviewer who checks the biggest dataset first finds the opposite of the headline**, and this is the
most likely single check anyone will run. Leading with it is equally wrong: it is 2 fits of 21, and
the three Letter datasets — the controlled family (§13) — run the other way, monotonically.

The other two exceptions are `iam_letter_low`/exact (0.63×) and `protein`/ub (−0.12×, and `protein`
is the smallest Suite-2 dataset). **So the exceptions do not line up with size** — largest and
smallest both appear — which is itself worth one clause, because it forecloses the obvious "it only
fails on small data" reading in either direction.

#### The straddle: FINAL, and my size hypothesis is refuted

Identifiable bound-pairs only, `n = 8`:

| claim | count | exact p | |
|---|---|---|---|
| **R² falls lb → ub** | **8 / 8** | **0.0078** | ✅ the only supported straddle claim |
| β_lev rises lb → ub | 7 / 8 | 0.0703 | ❌ descriptive only |
| ratio falls lb → ub | 7 / 8 | 0.0703 | ❌ same test |

**`mutagenicity` conforms** (ratio 0.69 → 0.36), so `iam_letter_high` remains the sole exception.

> **The orchestrator's hypothesis — that the transfer decays with dataset size, so the two largest
> would be where a second exception appeared — is REFUTED.** The largest Suite-2 dataset conforms and
> the lone exception is a mid-sized Letter set. The competing LB-ratio reading fares no better:
> `iam_letter_high`'s LB ratio is 2.24× while `mutagenicity`'s is 0.69×, smaller still, and it
> conforms. **With `n = 8` and one exception, neither reading is adjudicable. Report the count, not a
> mechanism.**

#### Final D4, over 21 identifiable fits

**β₁ significant and positive on 21 of 21. Size exceeds Levenshtein on 17 of 21.** Four fits excluded
as unidentifiable: `aids_iam` ×2, `coil_del` ×2.
