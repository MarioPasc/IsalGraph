# AGM canonical adjacency-matrix code (CAM)

**Verdict**: **RUN, RESTRICTED TO SUITE 1.** No usable implementation exists — we must write it,
and we did (§1). It is a genuine canonical form with the **second-best GED separation in the pool**.
But the exact code is a lex-leader minimisation and **our branch and bound stops closing at
`n ≈ 14`**: exact on 5/5 graphs up to GREC, **3/5 at AIDS (IAM)**, **0/5 from COIL-DEL upward**.

> **This is the folder's hardest constraint and it must reach the plan before T-06 is scheduled.**
> [competitors](../competitors.md) §2 budgets AGM at "1 d, derive from nauty labelling". **Both
> halves are wrong**: it cannot be derived from nauty's labelling (§2.3), and the cost is not the
> implementation but the fact that the thing being implemented is intractable above `n = 14`.

**Role**: [competitors](../competitors.md) §2 row 5 · serves **R1.2a** (named by R1 **by name**),
**AE.4a**, **AE.3**
**Evidence**: measured on this workstation, 2026-08-13,
`scratchpad/competitors/{agm_cam.py,probe,sweep,scale,stability}.py`.
Cross-refs: [adjacency-matrix](adjacency-matrix.md), [nauty](nauty.md), [README](README.md) §2.

---

## 1. Reproducibility — the blocking question

**There is no package.** AGM is Inokuchi, Washio & Motoda, *An Apriori-Based Algorithm for Mining
Frequent Substructures from Graph Data*, **PKDD 2000, LNCS 1910:13–23**, doi:10.1007/3-540-45372-5_2;
extended as *Complete Mining of Frequent Patterns from Graphs: Mining Graph Data*, **Machine
Learning 50:321–354, 2003**, doi:10.1023/A:1021726221443. Neither ships code. No maintained Python
or C++ package exposes "the AGM canonical code of one graph" — the canonical form exists inside
frequent-subgraph miners, applied to *mined patterns*, never as a standalone graph encoder.

**So we implement it.** `scratchpad/competitors/agm_cam.py`, ~120 lines, branch and bound.

### Validation — it is correct where it terminates

| Check | Result |
|---|---|
| vs brute force over **all `n!` permutations** | **327 graphs, 0 mismatches**: every isomorphism class on `n ≤ 6` (2, 4, 11, 34, 156 — including disconnected), plus 120 random graphs at `n = 7, 8` |
| reversibility, `code → graph` | isomorphic on all 327 |
| F3, 40 graphs × 25 genuine relabellings | **40 / 40 invariant** |
| 200 relabellings of the running example | **1 distinct code** |

### The definition we implemented, and why the convention needs stating

Following Jiang, Coenen & Zito, *A survey of frequent subgraph mining algorithms*,
**Knowledge Engineering Review 28(1):75–105, 2013**, doi:10.1017/S0269888912000331, §3.1:

> "Given an adjacency matrix `M` of a graph `g`, an encoding of `M` can be obtained by the sequence
> obtained from concatenating the lower (or upper) triangular entries of `M`, including entries on
> the diagonal. Since different permutations of the set of vertexes correspond to different
> adjacency matrices, the canonical (CAM) form of `g` is defined as the maximal (or minimal)
> encoding."

**AGM takes the minimum; FFSM (Huan, Wang & Prins, ICDM 2003) takes the maximum.** They are mirror
images and neither is more canonical than the other, so **the convention must be stated in the
paper or the numbers are unreproducible.** We use AGM's minimum, on the strict lower triangle read
row by row — which for an unlabelled simple graph is the same bit sequence as the strict upper
triangle read column-wise, i.e. **identical to graph6's payload and to the adjacency-matrix row**.

That reading order is not cosmetic: the first `k(k−1)/2` bits depend only on the first `k` vertices
of the permutation, and **that prefix property is the only reason branch and bound is possible at
all**.

---

## 2. What the representation looks like

`G` = 4-cycle `(0,1,2,3)` + triangle `(3,4,5)`, `n = 6`, `m = 7`, `|Aut(G)| = 4`.

```
adjacency (incident labelling)  '101001000100111'
AGM CAM (lex-min labelling)     '000001110011110'      <-- same length, different permutation
AGM CAM (relabelled G)          '000001110011110'      <-- INVARIANT
AGM CAM (H = G minus 0-3)       '000001011111000'      <-- 5 of 15 bits move for ONE edit
search nodes expanded            47
```

| Property | AGM CAM | IsalGraph pruned |
|---|---|---|
| Reversible | **yes** (code + `n`) | yes, up to isomorphism |
| Isomorphism-invariant | **yes** (40/40) | **yes** (40/40) |
| Complete invariant | **yes** | yes, within a directedness class |
| Alphabet | `{0,1}` | `\|Σ\| = 9` |
| Length | **`n(n−1)/2` exactly** | data-dependent |
| **Handles disconnected** | **yes** — AGM was designed for it | **no** |
| Handles isolated vertices | **yes** | n/a |
| Computable at `n = 98` | **no** (§2.2) | yes, `pruned` only |

### 2.1 Why AGM belongs in the pool even though it is a bit string

Jiang et al. classify every frequent-subgraph miner's representation into exactly two families:
**CAM** (AGM, FSG, FFSM) and **M-DFSC** (gSpan). R1.2 named one of each. Covering both families
with a measured row is what turns "we cite AGM and gSpan" into "we compared against the two
canonical-representation families in the mining literature", which is what AE.3 asks for.

### 2.2 The ceiling — measured, and it is low

Branch and bound with a **300,000 search-node budget**, median of 5 synthetic graphs per Suite-2
profile:

| Profile | `n` | `m` | exact | median search nodes | median wall-clock |
|---|---:|---:|:---:|---:|---:|
| Letter LOW / MED | 4 | 3 | **5/5** | 17 | 0.03 ms |
| Letter HIGH | 5 | 5 | **5/5** | 36 | 0.05 ms |
| LINUX | 9 | 8 | **5/5** | 1,440 | 3.4 ms |
| GREC | 11 | 12 | **5/5** | 6,606 | 20 ms |
| AIDS (GraphEdX) | 11 | 11 | **5/5** | 13,362 | 38 ms |
| **AIDS (IAM)** | 14 | 15 | **3/5** | 259,033 | **1.02 s** |
| **COIL-DEL** | 22 | 54 | **0/5** | budget hit | 2.4 s |
| **Mutagenicity** | 29 | 30 | **0/5** | budget hit | 3.3 s |
| **Protein** | 32 | 61 | **0/5** | budget hit | 4.0 s |
| ceiling sweep, `n ≥ 20` | 20–98 | — | **0/3 at every profile** | budget hit | — |

### 2.2b The real cohort — the ceiling is at Suite 1's edge, and it is dataset-shaped

Synthetic `G(n,m)` was pessimistic in one direction and optimistic in another. Measured on the
actual graphs:

| Dataset | `n̄` | `n_max` | graphs | budget | **exact** | median ms/graph |
|---|---:|---:|---:|---:|---:|---:|
| Letter LOW / MED / HIGH | 4.1–4.6 | 7–9 | 4,492 | 200k | **100 %** | 0.04–0.07 |
| LINUX | 8.71 | 10 | 89 | 200k | **100 %** | 8.0 |
| AIDS (Suite 1) | 10.56 | 12 | 769 | 200k | **99.6 %** (3 fail) | 54.4 |
| **GREC** (Suite 2) | 11.54 | 24 | 400 sampled | 100k | **76 %** (96 fail) | **173** |

> **All of Suite 1 is computable; Suite 2 is not.** AIDS at `n ≤ 12` loses 3 graphs of 769. GREC,
> whose *mean* is only one node larger but whose tail reaches 24, loses **24 %** — and at 173 ms per
> graph, raising the budget is not free: 16,370 Suite-2 graphs at even the GREC rate is ~47
> core-minutes for a column that would still be a quarter empty.
>
> The failure is driven by the **tail**, not the mean, which is why `n̄` is the wrong statistic to
> plan with. It also means the ceiling cannot be stated as a single `n`: it depends on how
> symmetric the individual graph is.

Raising the budget does not rescue it: search nodes grow ~20× from `n = 11` to `n = 14`, so
reaching `n = 32` needs something like `10¹⁰` nodes per graph, against 16,370 graphs.

**Both failure directions are present**, which rules out a cheap fix: COIL-DEL (dense) and
Mutagenicity (sparse, `m/n = 1.03`) both fail. Sparse graphs are *worse* for the minimum-code
objective because near-empty prefixes tie constantly and prefix pruning never bites.

**Complexity context, stated without over-claiming.** Computing a lex-leader — the lexicographically
least representative of an orbit under a permutation group — is **NP-hard** (Crawford, Ginsberg,
Luks & Roy, *Symmetry-breaking predicates for search problems*, **KR 1996**, 148–159; the framework
is Babai & Luks, *Canonical labeling of graphs*, **STOC 1983**, 171–183). AGM's canonical code is a
lex-leader of the adjacency bit string under `S_n`. **We do not claim the graph-restricted case
inherits that bound** — no such reduction is given here — only that the general problem is hard and
that our measured behaviour is consistent with it. Report the measurement; cite the complexity as
context.

### 2.3 nauty cannot supply the AGM labelling — the plan's premise is wrong

[competitors](../competitors.md) §2 budgets AGM at "1 d, derive from nauty labelling". **It does not
follow.** nauty produces *a* canonical labelling — the one its own refinement and automorphism
pruning arrive at — not the labelling that minimises AGM's code. Measured on the running example:

```
nauty canonical labelling -> graph6 payload   ' E@ro'  ->  bits 001110010011100
AGM lex-min labelling                                  ->  bits 000001110011110
```

Different labellings, different bit strings, both canonical. nauty's automorphism group **can** be
used to prune the AGM search (orbit-based pruning), and that is a real optimisation worth
implementing — but it changes the constant, not the asymptotics, and it does not produce the answer
directly.

---

## 3. Which distance does it accept?

| Candidate | F1 | Note |
|---|---|---|
| Hamming | 100 % of equal-`n` pairs | length is `n(n−1)/2`, a function of `n` |
| **padded Hamming** | **100 %** | principled, same argument as [adjacency-matrix](adjacency-matrix.md) §3 |
| Levenshtein | 100 % | defined |

Measured, 120 one-edit pairs (unit GED = 1) vs 120 random same-`n` pairs, `n ∈ [6,12]`:

| | median Lev, GED = 1 | median Lev, random | **separation** | median Ham, GED = 1 | median Ham, random | median length |
|---|---:|---:|---:|---:|---:|---:|
| AGM CAM | 4.5 | 9.0 | **0.50** | 5.0 | 12.0 | 36 |

> **Separation 0.50 — second best in the pool, ahead of IsalGraph's 0.73 and nauty's 0.83.**
> A one-edit pair costs half what an unrelated pair costs.

**On the real cohort, against certified exact GED** (200-graph sample, Levenshtein):

| | Letter LOW | Letter MED | Letter HIGH | LINUX | AIDS |
|---|---:|---:|---:|---:|---:|
| **AGM CAM** | 0.911 | **0.920** | **0.892** | **0.798** | *(3/769 encode failures)* |
| IsalGraph pruned | **0.925** | 0.916 | 0.683 | 0.474 | — |
| min-DFS code | **0.972** | **0.965** | 0.842 | 0.653 | — |
| *(size null)* | *0.899* | *0.909* | *0.926* | *0.713* | *0.799* |

> **AGM beats IsalGraph on Letter HIGH (+0.209) and LINUX (+0.324), ties on Letter MED and loses
> Letter LOW by 0.014.** The lex-min canonical form tracks GED **better** than IsalGraph's canonical
> string on three of four datasets where it is computable, and it is the only representation besides
> min-DFS to come close to the size null on LINUX.
>
> This sharpens [competitors](../competitors.md) §4 outcome 2. "Canonical does not imply stable" is
> true of **nauty** (0.677 / 0.663 / 0.639 / 0.538) and much **less** true of AGM (0.911 / 0.920 /
> 0.892 / 0.798). The two canonical forms differ by ~0.25 in ρ on the same graphs with the same
> distance. **They are not interchangeable and the paper must not collapse them into one row** —
> which is also why §4's outcome 2 should name nauty specifically rather than "canonical forms".

Primary distance by §3.4's rule: **padded Hamming** (cheapest passing F1 at 100 %, F2, F3, F4).

---

## 4. Fit for the information-content / message-length experiment (Claim A)

**Yes in principle, restricted in practice.** The bit count is `n(n−1)/2` — **identical to the raw
adjacency matrix, by construction**, because canonicalisation permutes bits without changing their
number.

> **Consequence: AGM contributes nothing new to Claim A.** Its Claim A column is the adjacency
> column. Printing them as two rows with identical numbers invites a reviewer to assume an error.
> Print one `n²` row with a footnote naming its four members ([README](README.md) §2), and let AGM
> earn its place on Claim B and on the AE.3 properties table, where it does have something to say.

Where AGM *is* computable (`n ≤ 12`, i.e. all of **Suite 1**), its Claim A row is exact and free.
Above that it is not computable at all, so the cell is `undefined`, not `approximate`.

---

## 5. Scope alignment and IsalGraph's advantage

**Aligned — R1 named AGM by name**, and [demands](../demands.md) R1.2a makes **T-08** the owner
because "the ask is *discussion*". This file's measurements are what upgrade that discussion from
assertion to evidence, and they are worth having even if the empirical row is cut.

**IsalGraph's advantage over AGM, audited:**

| Axis (R1.2b) | Winner | Measurement |
|---|---|---|
| Uniqueness | **tie** | both 40/40 invariant, both complete invariants |
| Expressiveness | **AGM** | handles disconnected graphs and isolated vertices; IsalGraph raises `DisconnectedGraphError` |
| **Computational efficiency** | **IsalGraph, decisively** | AGM: 1.0 s at `n = 14` and **no exact answer at all from `n ≈ 20`**. IsalGraph `pruned`: 1.02 ms at `n = 32`, ~1000× faster and it terminates |
| **Scalability** | **IsalGraph** | AGM's ceiling is `n ≈ 14`; AE.1 requires `n = 98` |
| Downstream learning | **not evaluated** | R1.2b's fifth axis |
| Bits | **AGM** below `n ≈ 14`, tie with adjacency by construction | see §4 |
| Edit-distance compatibility | **AGM** (0.50 vs 0.73) | §3 |

> **The foundational advantage is real and it is exactly one thing: tractability.** AGM defines a
> canonical form with better GED behaviour than ours and cannot compute it at the scale the Area
> Editor asked us to reach. IsalGraph defines a canonical form that is computable in milliseconds
> at `n = 98`. **That is the sentence the paper should write**, and it is far stronger than the
> unscoped "no existing method is simultaneously compact, reversible, structure-preserving and
> canonicalisable" at `introduction.tex:33` that R1.2 and R3.1 both object to — because it is
> measured, bounded, and it concedes the axes where AGM wins.

---

## 6. Summary

| # | Question | Answer |
|---|---|---|
| 1 | Reproducible? | **No package exists.** We wrote it (~120 lines) and **validated it against brute force on 327 graphs, 0 mismatches** |
| 2 | Representation | lex-min adjacency bit string, `n(n−1)/2` bits, **50/50 invariant on real graphs**, complete invariant, handles disconnected |
| 3 | Distance | **padded Hamming**. Real ρ **0.80–0.92 — beats IsalGraph on 3 of 4 datasets**, by up to +0.324 |
| 4 | Claim A? | **Yes but redundant** — identical to the adjacency row by construction; and **not computable on Suite 2** |
| 5 | Scope | **In, restricted to Suite 1.** R1 named it; the family coverage (CAM vs M-DFSC) needs it |
| — | IsalGraph advantage | **Tractability only, and it is decisive.** AGM wins expressiveness and GED tracking; it cannot be computed on 24 % of GREC |

---

## 7. For the integration agent

**Recommended policy, and the reason to fix it before T-06 rather than during it:**

1. **AGM runs on Suite 1 only.** Measured on the real cohort: **100 %** exact on all three Letter
   sets and LINUX, **99.6 %** on AIDS (3 of 769 fail at a 200k-node budget), and **76 %** on GREC
   (96 of 400 fail at 100k, 173 ms/graph). It is excluded from Suite 2 with that sentence printed.
   A stated ceiling is a result; a silent one is a defect. **The 3 AIDS failures must also be
   printed**, not dropped — they are why AGM has no ρ column on AIDS.
2. **Do not substitute a heuristic labelling above the ceiling.** An incumbent from the greedy
   initialisation is *not* canonical, would fail F3, and would put a non-invariant code into a table
   whose column header says canonical. That is precisely the error graph6 is in the pool to expose.
3. Under [preregistration](../preregistration.md) §5, an AGM restricted to Suite 1 does not set
   `k = 1` — it keeps its 5 B1e rows and loses its 10 B1a rows. **That case is not in the current
   reduction rule**, which only has "no admissible distance" (`−15`) and "not computable at all"
   (`−15 −10`). A representation computable on one suite and not the other needs its own line.
   **Raise this with T-02's owner** — the family cardinality `N_max = 182` depends on it.
4. Implementation notes: keep the node budget and **raise `AGMBudgetExceeded` rather than returning
   the incumbent**. Add orbit pruning from `pynauty.autgrp` if the ceiling needs to move — it will
   help, but it will not reach `n = 32`.
5. Fix and document the convention (**minimum**, strict lower triangle row-wise) in one place, and
   assert bit-for-bit agreement with the [adjacency-matrix](adjacency-matrix.md) reading order on a
   handful of graphs.
