# Weisfeiler–Lehman subtree kernel

**Verdict**: **RUN — Claim B only.** It is not a serialisation, has no bit count, and is **not a
complete invariant**: `K_{3,3}` and the triangular prism get **distance exactly 0.0000** while every
other member of the pool separates them. That one 6-node witness is the folder's cleanest evidence
for R1.2's uniqueness axis.

**Role**: [competitors](../competitors.md) §2 row 7 · serves **AE.4a**, **AE.3**, **R1.2b**
**Evidence**: measured on this workstation, 2026-08-13.
Cross-refs: [README](README.md) §4, [statistics](../statistics.md) §4.

---

## 1. Reproducibility — is anything blocking?

**Nothing, twice over.**

| Route | Status |
|---|---|
| `grakel` 0.1.8 | **already installed in the `isalgraph-cpp` env**; no new dependency |
| 40-line reimplementation | written, and **cross-checked exactly** against `grakel` |

```python
from grakel import Graph, WeisfeilerLehman, VertexHistogram
K = WeisfeilerLehman(n_iter=3, base_graph_kernel=VertexHistogram,
                     normalize=False).fit_transform(graphs)
```

> **One trap, and it changes every number.** `grakel`'s `n_iter = k` runs **`k` rounds counting the
> base histogram as round 0**, i.e. it refines `k − 1` times. Our own implementation's `h` counts
> refinements. Measured on the running example, `grakel(n_iter=3)` and `ours(h=2)` agree to
> **5.830952 exactly**, while `ours(h=3)` gives 6.928203. **Fix `h` once, in one place, and state it
> in the paper caption.** The manuscript's existing WL numbers (E10) must be checked against
> whichever convention produced them before they are re-quoted.

Two independent implementations agreeing to machine precision is worth the 40 lines: it makes the WL
row auditable without a third-party version pin.

Cite: Shervashidze, Schweitzer, van Leeuwen, Mehlhorn & Borgwardt, *Weisfeiler-Lehman graph
kernels*, **JMLR 12:2539–2561, 2011**. The manuscript already cites `weisfeiler1968reduction` at
`introduction.tex:27`; **the kernel paper is a different citation and is missing.**

---

## 2. What the representation looks like

It is **not a string and not a serialisation**. It is a sparse count vector over WL colours: the
multiset of refined neighbourhood signatures, accumulated over `h` rounds.

`G` = 4-cycle `(0,1,2,3)` + triangle `(3,4,5)`, `n = 6`, `m = 7`, `h = 3` → **10 non-zero features**.
`H` = `G` minus edge `(0,3)` → **13 non-zero features**. The dimension is data-dependent and the
coordinates are only comparable across graphs because the signatures are built from uncompressed
parent labels.

### 2.1 The completeness witness — run it, print it, put it in the paper

```
K_{3,3}  and  C₃ × K₂ (triangular prism):  both connected, both 3-regular on 6 vertices, NOT isomorphic

  WL h = 1, 2, 3, 5     identical feature vector      kernel distance  0.0000   <-- FAILS
  nauty -> graph6       'Es\o'          vs 'E{Sw'                               separates
  AGM CAM               '000111111011100' vs '001101110111100'                  separates
  min-DFS code          '0-1 1-2 2-3 3-0 3-4 4-1 4-5 5-0 5-2'
                     vs '0-1 1-2 2-0 2-3 3-4 4-0 4-5 5-1 5-3'                   separates
  IsalGraph canonical   'VVVpvvPpCpCPCnC' vs 'VVVpvpvPCnCNNCnC'                 separates
```

1-WL cannot distinguish regular graphs of the same degree and order — the colouring is constant
after round 1, so refinement never starts. **No number of rounds fixes it**, which is why `h = 5`
is in the table.

### 2.2 The incompleteness fires on the real cohort, not just on constructed examples

Measured over the certified-exact pairs of the 200-graph sample, comparing the fraction of pairs at
**kernel distance 0** against the fraction that are genuinely isomorphic (`GED = 0`):

| Dataset | pairs | frac `d_WL = 0` | frac `GED = 0` | **false zeros** |
|---|---:|---:|---:|---:|
| Letter LOW | 19,900 | 0.1438 | 0.1438 | **0** |
| Letter MED | 19,900 | 0.1530 | 0.1530 | **0** |
| Letter HIGH | 19,900 | 0.0453 | 0.0453 | **0** |
| LINUX | 3,870 | 0.00026 | 0.0000 | **≈ 1 pair** |
| AIDS | 15,686 | 0.00038 | 0.0000 | **≈ 6 pairs** |

> **On the three Letter sets WL's zero-set matches the isomorphic set exactly** — the pseudometric
> behaves as a metric there, and saying so is the honest framing. On LINUX and AIDS it does not:
> around one and six pairs respectively receive distance 0 at strictly positive GED. Small, but not
> zero, and **it is a real-data witness rather than a constructed one**. Report both halves.

> This is the concrete answer to R1.2's *"does the proposed representation provide benefits in terms
> of uniqueness, expressiveness…"*. `introduction.tex:27` cites the WL test as the expressivity
> yardstick for MPNNs; this example shows the yardstick failing on a 6-node graph that IsalGraph
> encodes distinctly. It costs one small figure and it is the strongest single artifact in the pool.

| Property | WL subtree kernel | IsalGraph pruned |
|---|---|---|
| Reversible | **no** | yes, up to isomorphism |
| Isomorphism-invariant | **yes** (40/40) | **yes** (40/40) |
| **Complete invariant** | **no** — witness above | **yes**, within a directedness class |
| Output | sparse count vector, data-dependent dimension | string, `\|Σ\| = 9` |
| Bit count | **none** — see §4 | `L log₂ 9` |
| Handles disconnected | **yes** | no |
| Encode cost | `Θ(h·m)`, microseconds | 0.01–1.02 ms (pruned) |

---

## 3. Which distance does it accept?

**Exactly one, and it is the only member of the pool that does not take an edit distance.**

```
d(G, H) = sqrt( K(G,G) + K(H,H) − 2 K(G,H) )
```

the RKHS distance induced by the linear kernel on the WL feature vectors.

| F-criterion | Result |
|---|---|
| **F1** well-defined | **100 %** — defined on every pair regardless of `n` |
| **F2** metric axioms | **pseudometric only.** Symmetry and triangle inequality hold (it is a Euclidean distance in feature space); **identity of indiscernibles fails** — `d(K_{3,3}, prism) = 0` with the graphs non-isomorphic. **This must be declared**, per [competitors](../competitors.md) §3.3 F2 |
| **F3** invariance | **passes**, 40/40 |
| **F4** non-degenerate | passes on random graphs; **degenerate on regular ones** |
| **F5** tracks GED | **measured**, see below |
| **F6** affordable | microseconds |

ρ against certified exact GED, 200-graph sample per dataset:

| | Letter LOW | Letter MED | Letter HIGH | LINUX | AIDS |
|---|---:|---:|---:|---:|---:|
| **WL, `h = 2`** | 0.895 | 0.869 | 0.580 | **0.573** | **0.459** |
| WL, `h = 3` | 0.821 | 0.810 | 0.494 | 0.501 | 0.412 |
| IsalGraph pruned | 0.925 | 0.916 | 0.683 | 0.474 | 0.255 |
| *(size null)* | *0.899* | *0.909* | *0.926* | *0.713* | *0.799* |

> **WL beats IsalGraph on LINUX (+0.099) and AIDS (+0.204)** — the two datasets where the
> manuscript's ρ is already weakest, and R3.6b already says so. A *non-reversible, incomplete*
> feature vector tracks GED better than the canonical string on exactly the datasets the paper
> concedes. That is a result and it goes in the table.
>
> **More rounds are worse, uniformly.** `h = 3` is below `h = 2` on all five datasets. Fix `h = 2`,
> state it, and do not tune it — tuning `h` on ρ would be selecting a baseline on the outcome, the
> same error [competitors](../competitors.md) §3.4 forbids for our own distances.

> **F2 is the row that matters.** IsalGraph's Levenshtein distance on canonical strings is a **true
> metric** on isomorphism classes precisely because the canonical string is a complete invariant.
> The WL kernel distance is a **pseudometric**. Comparing a metric against a pseudometric is
> legitimate and we do it — but §3.3 requires it be declared, and the `K_{3,3}` witness is what
> makes the declaration concrete rather than a footnote.

Normalisation is a live choice: `normalize=True` divides by `sqrt(K(x,x)K(y,y))`, which removes the
graph-size signal that GED depends on. **Use `normalize=False`** and say so — a normalised kernel
would be correlated against a size-sensitive GED and would look worse for reasons that have nothing
to do with WL.

---

## 4. Fit for the information-content / message-length experiment (Claim A)

**No, and this is already settled.** [preregistration](../preregistration.md) §4.1 fixes the Claim A
set at **6** serialisations and excludes WL because "it is not reversible and emits a feature
vector, so it has no bit count to compare". That is correct and this file changes nothing about it.

A feature-vector "bit cost" could be manufactured — dimension × counter width — but it would measure
our choice of container, not the encoding, and it would be indefensible next to a reversible format.
**Leave the Claim A cell empty and print the reason.**

---

## 5. Scope alignment and IsalGraph's advantage

**Aligned, with one caveat worth stating in the plan.** WL is not a graph *canonicalisation* method
and R1.2's request was specifically about canonicalisation methods (AGM, gSpan). WL enters through
**AE.4a**'s "choice of benchmark models" and because [competitors](../competitors.md) §6 puts it in
experiment (b) — and because E10 says the machinery already exists and has never been reported.
Marginal cost ≈ 0.

**IsalGraph's advantage over WL is foundational and is the clearest in the folder:**

| Axis (R1.2b) | Winner | Measurement |
|---|---|---|
| **Uniqueness** | **IsalGraph** | complete invariant vs **provably incomplete**, `K_{3,3}` / prism witness |
| **Expressiveness** | **IsalGraph** | reversible; WL cannot reconstruct a graph from its colour multiset at all |
| Computational efficiency | **WL** | `Θ(h·m)`, microseconds |
| Scalability | **WL** | no ceiling |
| **Downstream learning** | **WL** | the only pool member with an established learning record; IsalGraph reads **"not evaluated"** and R1.2b requires that word |
| Metric | **IsalGraph** | metric vs **pseudometric** |

> The honest framing: WL is the reference point for *how far a cheap, incomplete invariant gets
> you*. ~~If `ρ(WL distance, GED)` comes out above IsalGraph's on some dataset…~~ **It does — on
> LINUX (0.573 vs 0.474) and AIDS (0.459 vs 0.255).** We print it. What bounds it is that WL cannot
> serialise, cannot reverse, and cannot separate `K_{3,3}` from a prism — so it answers a different
> question, and the comparison table's job is to make that visible rather than to win the row.

---

## 6. Summary

| # | Question | Answer |
|---|---|---|
| 1 | Reproducible? | **Yes, twice.** `grakel` 0.1.8 already installed, plus a 40-line reimplementation agreeing exactly. **Watch the `n_iter` vs `h` off-by-one** |
| 2 | Representation | sparse colour-count vector; invariant (50/50 on real graphs) but **not complete** — `d(K_{3,3}, prism) = 0`, and ~6 false zeros on real AIDS |
| 3 | Distance | RKHS kernel distance only. **F1 100 %, F2 pseudometric — must be declared**, F3 pass. Real ρ **0.46–0.90 at `h = 2`; beats IsalGraph on LINUX and AIDS** |
| 4 | Claim A? | **No.** Not reversible, no bit count. Already excluded by preregistration §4.1 |
| 5 | Scope | **In, Claim B only.** Enters via AE.4a, not via R1.2's canonicalisation ask |
| — | IsalGraph advantage | **Yes, foundational**: completeness, reversibility, true metric. WL wins efficiency, scalability, downstream record |

---

## 7. For the integration agent

- WL implements `GEDBackend`-like semantics, not `ReprBackend` — it has `distance(a, b)` but no
  `encode(G) -> str` and no `bit_length(G)`. **Give it its own protocol or make `bit_length` raise**;
  returning a fabricated number is the failure mode to avoid.
- Fit the kernel on the **whole dataset at once**. `fit_transform` on a subset produces a different
  colour vocabulary and therefore different distances — a per-batch fit would make the distance
  matrix depend on batching order, which is a silent-corruption bug of the same family as the
  `get_lower_bound()` trap in CLAUDE.md.
- `normalize=False`, `n_iter` fixed and recorded, and assert the two implementations agree on a
  fixture — that assertion is cheap and it pins the off-by-one forever.
- Add the `K_{3,3}` / prism pair as a **unit test fixture**: WL distance 0, every other backend
  non-zero. It is a two-line regression test that would catch a broken canonical backend instantly.
