# Adjacency matrix — the `n²` reference point

**Verdict**: **RUN — as the reference point, not as a rival.** It is the one representation every
reviewer already has in mind, it is the floor for the `n²` family, and it is free.

**Role**: [competitors](../competitors.md) §2 row 4 · serves **AE.4a**, **AE.3**, **R3.6a**
**Evidence**: measured on this workstation, 2026-08-13.
Cross-refs: [graph6](graph6.md) (the same bits, packed), [agm](agm.md) (the same bits, canonically
permuted), [README](README.md) §2.

---

## 1. Reproducibility — is anything blocking?

**Nothing.** It is `nx.to_numpy_array`, or nine lines of pure Python for the upper triangle. There
is no package to install, no version to pin, and no failure mode. Effort: **trivial**, exactly as
[competitors](../competitors.md) §2 estimates.

---

## 2. What the representation looks like

`G` = 4-cycle `(0,1,2,3)` + triangle `(3,4,5)`, `n = 6`, `m = 7`.

```
strict upper triangle, column-wise:  '101001000100111'      15 bits
same for H = G minus edge (0,3):     '100001000100111'
```

The convention matters and must be fixed once: **strict upper triangle read column-wise**,
`a(0,1) a(0,2) a(1,2) a(0,3) …`. This is byte-identical to graph6's payload and to AGM's CAM code
reading order, which is what makes [README](README.md) §2's family argument work. Choosing
row-major instead would break the correspondence for no benefit.

| Test | Result |
|---|---|
| 200 relabellings of the running example | **122 distinct bit strings** |
| F3 sweep: 40 graphs × 25 relabellings | **0 / 40 invariant** |

| Property | adjacency | IsalGraph pruned |
|---|---|---|
| Reversible | **yes**, exactly | yes, up to isomorphism |
| Isomorphism-invariant | **no** (0/40) | **yes** (40/40) |
| Complete invariant | no | yes, within a directedness class |
| Alphabet | `{0,1}` — the only binary member of the pool | `\|Σ\| = 9` |
| Length | **`n(n−1)/2` exactly**; a function of `n` alone | data-dependent |
| Handles disconnected | **yes** | no |
| Handles isolated vertices | **yes** | n/a |
| Encode cost | `Θ(n²)`, microseconds | 0.01–1.02 ms (pruned) |

---

## 3. Which distance does it accept?

This is the **only** representation in the pool for which the padded-Hamming convention in
[competitors](../competitors.md) §3.2 is natural rather than improvised: the code is a positional
bit vector indexed by vertex pairs, so embedding two graphs in a common `max(n₁,n₂)` frame is
well defined and its justification — that the padding coincides with the node insertions D6
charges — actually holds.

| Candidate | F1 | Note |
|---|---|---|
| Hamming | 100 % of equal-`n` pairs, 0 % otherwise | length is `n(n−1)/2` |
| **padded Hamming** | **100 %** | **the only pool member where this is principled** |
| Levenshtein | 100 % | defined but semantically odd on a positional vector |

Measured, 120 one-edit pairs (unit GED = 1, edited copy randomly relabelled) vs 120 random same-`n`
pairs:

| | median Lev, GED = 1 | median Lev, random | **separation** | median Ham, GED = 1 | median Ham, random |
|---|---:|---:|---:|---:|---:|
| adjacency | 12.0 | 13.0 | **0.92** | 17.0 | 18.0 |

Separation **0.92** on synthetic `G(n,m)`: a one-edit pair is nearly indistinguishable from an
unrelated one. **Fails F3, takes no primary distance, excluded from the running Claim B comparison**
and reported in the supplementary grid with the reason printed.

### 3.1 On the real cohort it is *not* a pushover, and understanding why matters

Spearman ρ of Levenshtein against T-03's certified exact GED, 200-graph sample per dataset:

| | Letter LOW | Letter MED | Letter HIGH | LINUX | AIDS |
|---|---:|---:|---:|---:|---:|
| **adjacency** | **0.873** | **0.850** | **0.839** | **0.754** | **0.787** |
| IsalGraph pruned | 0.925 | 0.916 | 0.683 | 0.474 | 0.255 |
| graph6 (same bits, packed) | 0.691 | 0.681 | 0.670 | 0.507 | 0.456 |
| nauty→graph6 (same bits, canonical) | 0.677 | 0.663 | 0.639 | 0.538 | 0.460 |

**The raw adjacency matrix — invariant on 0–6 of 50 graphs — beats IsalGraph on Letter HIGH, LINUX
and AIDS.** That result cannot be printed without an explanation, so we chased it down.

**Hypothesis 1, refuted: it is the corpus's incidental vertex order.** IAM and GraphEdX graphs carry
a consistent labelling (drawing order, node id), so "arbitrary" might be correlated across graphs.
Test: relabel every graph independently at random and recompute. Canonical representations are the
control and must not move.

| ρ after independent random relabelling | Letter LOW | Letter HIGH | LINUX | AIDS |
|---|---:|---:|---:|---:|
| adjacency | 0.873 → 0.826 (**−0.046**) | 0.844 → 0.811 (−0.034) | 0.754 → 0.702 (−0.052) | 0.800 → 0.770 (−0.029) |
| sparse6 | 0.748 → 0.652 (−0.095) | 0.673 → 0.647 (−0.026) | 0.559 → 0.342 (**−0.217**) | 0.487 → 0.237 (**−0.250**) |
| graph6 | 0.691 → 0.676 (−0.016) | 0.691 → 0.676 (−0.015) | 0.507 → 0.377 (−0.130) | 0.474 → 0.387 (−0.088) |
| nauty→graph6 *(control)* | **0.0000** | **0.0000** | **0.0000** | **0.0000** |
| min-DFS *(control)* | **0.0000** | **0.0000** | **0.0000** | **0.0000** |

The controls move by exactly zero, so the harness is sound. **sparse6 loses half its signal**
(−0.217 on LINUX, −0.250 on AIDS) — that part *was* the corpus labelling. **The adjacency matrix
barely moves.** Hypothesis refuted.

**Hypothesis 2, confirmed: it is graph size.** Levenshtein between two bit strings of length
`n₁(n₁−1)/2` and `n₂(n₂−1)/2` is bounded below by their length difference, a monotone function of
`|n₁ − n₂|`; and under D6, `GED ≥ |n₁ − n₂|`. So the adjacency matrix's distance is largely a
**proxy for node-count difference**.

| | Letter LOW | Letter MED | Letter HIGH | LINUX | AIDS |
|---|---:|---:|---:|---:|---:|
| **NULL: `ρ(\|n₁−n₂\|, GED)`** | **0.899** | **0.909** | **0.926** | **0.713** | **0.799** |
| adjacency | 0.873 | 0.850 | 0.839 | 0.754 | 0.787 |
| adjacency, **equal-`n` pairs only** | **0.565** | **0.429** | **0.424** | **0.300** | **0.243** |

Restricted to equal-`n` pairs the adjacency matrix collapses to **0.24–0.57**, while the canonical
representations hold 0.63–1.00 ([README](README.md) §4.2). **That is the honest comparison, and it
is the one the paper should lead with.**

> **Two things follow, and the second is the more important.**
>
> 1. The adjacency matrix's apparent strength is a size proxy, not structure. It still **fails F3**
>    and still takes no primary distance under §3.4 — the rule was right.
> 2. **The size null beats IsalGraph on four of five datasets** ([README](README.md) §5, finding 1).
>    That is not a fact about the adjacency matrix; it is a fact about how every ρ in this paper
>    must be reported. It was found here only because this row looked wrong.

---

## 4. Fit for the information-content / message-length experiment (Claim A)

**Yes, and it is the natural denominator.** [competitors](../competitors.md) §5 already assigns it
the entropy bound `n(n−1)/2` — no alphabet argument needed, since the symbols *are* bits.

| Convention | Value |
|---|---|
| Entropy bound | **`n(n−1)/2`** |
| Realised bytes | `8 · ⌈n(n−1)/16⌉` if packed two-per-byte… — **do not** invent a packing. Use `⌈n(n−1)/16⌉` bytes, i.e. the triangle packed 8 bits to a byte, and say so |

Measured on the **real cohort**, median entropy-bound bits over all retained graphs (Suite 2:
400-graph sample), and the percentage of graphs on which IsalGraph pruned is strictly shorter:

| Dataset | `n̄` | **adjacency** | graph6 | sparse6 | min-DFS | IsalGraph pruned | **% Isal < adjacency** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Letter LOW | 4.07 | **6.0** | 12.0 | 24.0 | 12.0 | 12.7 | **0.0 %** |
| Letter MED | 4.11 | **6.0** | 12.0 | 24.0 | 12.0 | 12.7 | **0.0 %** |
| Letter HIGH | 4.58 | **10.0** | 18.0 | 36.0 | 24.0 | 25.4 | **0.0 %** |
| LINUX | 8.71 | **36.0** | 42.0 | 60.0 | 64.0 | 41.2 | 15.7 % |
| AIDS | 10.56 | **55.0** | 66.0 | 72.0 | 88.0 | 57.1 | 29.9 % |
| GREC | 11.54 | 55.0 | 66.0 | 78.0 | 96.0 | 72.9 | 23.0 % |

> **The adjacency matrix beats IsalGraph on every Suite-1 dataset, and on the three Letter sets
> IsalGraph is shorter on 0.0 % of graphs — not one graph out of 4,492.** This is the single most
> load-bearing number in the folder and it must reach the paper.
>
> It is not a defeat: `n(n−1)/2` is minimal *when the graph is small*, and Suite 1 is exactly that
> regime (`n̄` 4.07–10.56). The submitted manuscript compared only against the author-defined
> `B_GED` construction, which IsalGraph beats on **100 %** of GREC graphs — but which the adjacency
> matrix also beats everywhere. **R3.6a's "narrow the claim accordingly" applies to us at least as
> much as the reviewer knew.**
>
> The claim that survives measurement: *IsalGraph is shorter than every other **string**
> serialisation — graph6, sparse6, the minimum DFS code — and than the explicit-construction
> reference model; the raw adjacency matrix is shorter at Suite-1 sizes, and IsalGraph's `m`-scaling
> overtakes its `n²` growth only for large sparse graphs.* Whether that crossover is reached inside
> Suite 2 is open until the AIDS-IAM / COIL-DEL / Mutagenicity / Protein rows land.

---

## 5. Scope alignment and IsalGraph's advantage

**Aligned as the reference point.** R1.2 asks how IsalGraph "differs conceptually from existing
representations"; the adjacency matrix is the representation the answer has to start from, and
`introduction.tex:11–17` already cites it. Its inclusion costs nothing and its omission would be
conspicuous.

**IsalGraph's advantage**: canonicity (40/40 vs 0/40), completeness as an invariant, and `m`-scaling
instead of `n²`-scaling — real, and worth exactly the three profiles where it wins.

**What the adjacency matrix has that IsalGraph does not**: exactness, universality (disconnected,
isolated, directed, weighted, labelled — all without ceremony), a length that is a closed form, a
decode with no search, and **fewer bits on 7 of 10 profiles**.

---

## 6. Summary

| # | Question | Answer |
|---|---|---|
| 1 | Reproducible? | **Yes, trivially.** No dependency at all |
| 2 | Representation | `n(n−1)/2` positional bits; **not** isomorphism-invariant (0–6 / 50 on real graphs) |
| 3 | Distance | **padded Hamming** is principled here and only here. ρ = **0.75–0.87** on real data — but that is a **size proxy**, and it collapses to **0.24–0.57** on equal-`n` pairs |
| 4 | Claim A? | **Yes — the denominator.** Beats IsalGraph on **every Suite-1 dataset**; IsalGraph is shorter on **0.0 %** of Letter graphs |
| 5 | Scope | **In, as the reference point.** Omitting it would be conspicuous — and it turned out to be the row that exposed the size null |
| — | IsalGraph advantage | **Canonicity yes** (0–6/50 vs 50/50, and 0.63–0.98 vs 0.24–0.57 on equal-`n` pairs); **compactness no at Suite-1 sizes** |

---

## 7. For the integration agent

- Fix the reading order to **strict upper triangle, column-wise**, and assert it against
  `nx.to_graph6_bytes` on a few graphs — the two must agree bit for bit, and that assertion is what
  keeps [README](README.md) §2's family claim true in code rather than in prose.
- Report `n(n−1)/2` for the entropy bound. **Do not** report the realised byte count as
  `len(str) * 8` — the string `'101001…'` is a debugging view, not a serialisation, and counting it
  as 8 bits per character would inflate the adjacency matrix by 8× and hand us a baseline we beat
  for free. This is the single easiest way to produce a wrong Claim A table.
- The padded-Hamming frame must be built from the **canonical** labelling when the pair comes from
  a canonical backend, and from the incident labelling otherwise. Mixing the two silently compares
  different things.
