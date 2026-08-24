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

**Also usable:** IsalGraph produces an encoding for **100 % of both cohorts** (with the D14 fallback
on 101 graphs), while `agm_cam` — the strongest small-`n` competitor — manages **6.15 % of Protein**
and **6.29 % of Mutagenicity** because it is refused above `n = 12`. Computability across the cohort
is a real property and it is worth one sentence.

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
