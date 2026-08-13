# sparse6 — McKay's edge-list serialisation for sparse graphs

**Verdict**: **RUN.** Reproducible in one line; fails isomorphism invariance on **0 of 40** graphs,
so it takes no primary distance — but it is **IsalGraph's only genuine rival on bits**, because it
is the one competitor whose length also scales with `m` rather than `n²`.

**Role**: [competitors](../competitors.md) §2 row 2 · serves **AE.4a**, **R3.6a**
**Evidence**: measured on this workstation, 2026-08-13, `scratchpad/competitors/{probe,sweep,scale,stability}.py`.
Cross-refs: [graph6](graph6.md) (same family, `n²` scaling), [README](README.md) §3.

> ⚠ **This file overturns pre-committed outcome 3 in [competitors](../competitors.md) §4.**
> That section expects sparse6 to beat IsalGraph *on sparse graphs*. Measured, the ordering is
> **inverted**: IsalGraph wins on the sparse profiles and loses on the dense ones. §4 must be
> restated before it is printed. See §4 below.

---

## 1. Reproducibility — is anything blocking?

**Nothing.** Same package, same call shape as graph6.

```python
import networkx as nx
nx.to_sparse6_bytes(G, header=False)     # b':EaWIzR\n'
nx.from_sparse6_bytes(b":EaWIzR")
```

Measured **0.028–0.157 ms per graph** over the full cohort profile range — the **cheapest**
backend in the pool at every size, cheaper than graph6 above `n = 9` because it never materialises
the `n²` triangle.

Spec: <https://users.cecs.anu.edu.au/~bdm/data/formats.txt>. Cite McKay & Piperno,
*Practical graph isomorphism, II*, **J. Symb. Comput. 60:94–112, 2014**, doi:10.1016/j.jsc.2013.09.003.

---

## 2. What the representation looks like

`G` = 4-cycle `(0,1,2,3)` + triangle `(3,4,5)`, `n = 6`, `m = 7`, `|Aut(G)| = 4`.

```
sparse6(G)                     ':EaWIzR'                  7 bytes
sparse6(G relabelled)          ':EgHeALN'                 <-- CHANGED
sparse6(H = G minus edge 0-3)  ':EaYms'                   6 bytes -- note the length change
```

**Structure.** `':'` + `N(n)` + a bit stream of `(b_i, x_i)` groups, where `b_i` is one flag bit and
`x_i` is a `k = ⌈log₂ n⌉`-bit vertex index. Decoding walks a cursor `v` upward, incrementing it on
`b_i = 1`, and emits `{x_i, v}`. Byte length `≈ 1 + ⌈m(k+1)/6⌉`.

**Two consequences that matter downstream and are easy to miss:**

1. **The length depends on `m`, so it varies between two graphs of the same `n`.** graph6's does
   not. Plain Hamming is therefore undefined on far more pairs than for graph6 — measured
   **defined on 30.8 %** of one-edit pairs, against 100 % for graph6. This is the concrete case
   [competitors](../competitors.md) §3 was written to catch.
2. **The encoding is sensitive to the *ordering* of the labels, not only to the labels.** The
   cursor walk means a permutation that preserves the edge set can still change `m(k+1)` bits of
   payload. Hence the 123-distinct-code result below.

| Test | Result |
|---|---|
| 200 relabellings of the running example | **123 distinct sparse6 strings** |
| F3 sweep: 40 graphs × 25 relabellings | **0 / 40 invariant**; up to **26 distinct codes** |

| Property | sparse6 | IsalGraph pruned |
|---|---|---|
| Reversible | **yes**, exactly | yes, up to isomorphism |
| Isomorphism-invariant | **no** (0/40) | **yes** (40/40) |
| Complete invariant | no | yes, within a directedness class |
| Fixed finite alphabet | yes, 64 printable ASCII | yes, `\|Σ\| = 9` |
| Length | `≈ 1 + ⌈m(k+1)/6⌉`, **varies with `m`** | data-dependent |
| Scaling | `Θ(m log n)` | `Θ(L)`, `L` grows with `m` |
| Handles disconnected | **yes** | no |
| Handles isolated vertices | **yes** | n/a |
| Encode cost | 0.028–0.157 ms — **cheapest in the pool** | 0.01–1.02 ms (pruned) |

---

## 3. Which distance does it accept?

| Candidate | F1 (well-defined) | Verdict |
|---|---|---|
| Hamming | **30.8 %** of pairs | fails F1 outright |
| padded Hamming | n/a | undefined — sparse6 is not a positional bit vector, so there is no frame to pad into. **This is why the padding convention cannot be applied uniformly** |
| Levenshtein | 100 % | defined, but fails F3 |

Measured over 120 one-edit pairs (unit GED = 1, edited copy randomly relabelled) against 120
random same-`n` pairs, `n ∈ [6,12]`:

| | median Lev, GED = 1 | median Lev, random | **separation** |
|---|---:|---:|---:|
| sparse6 | 11.0 | 12.5 | **0.88** |

A one-edit pair costs 88 % of what an unrelated pair costs. **No usable GED signal.** By
[competitors](../competitors.md) §3.4 sparse6 takes no primary distance and is excluded from the
running Claim B comparison, contributing `k = 1` to the `N_actual` reduction
([preregistration](../preregistration.md) §5) — it loses its 15 Claim B rows and keeps its 10
Claim A rows.

> **A canonicalised sparse6 is available and is the honest fix**, exactly as for graph6: relabel by
> nauty first, then serialise. It is one extra line given the [nauty](nauty.md) backend. **Do it** —
> it costs nothing and it removes the objection that we compared a canonical method against a
> non-canonical one on the compactness axis. Report both rows.

---

## 4. Fit for the information-content / message-length experiment (Claim A)

**Yes — and this is where sparse6 earns its place.** Entropy-bound bits, median of 5 synthetic
graphs per Suite-2 profile, `n` and `m` set to T-01's measured per-dataset means:

| Profile | `n` | `m` | `m/n` | sparse6 | graph6 | adjacency | **IsalGraph pruned** | winner |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Letter LOW | 4 | 3 | 0.75 | 24 | 12 | **6** | 13 | adjacency |
| Letter HIGH | 5 | 5 | 1.00 | 36 | 18 | **10** | 25 | adjacency |
| LINUX | 9 | 8 | 0.89 | 66 | 42 | **36** | 38 | adjacency |
| AIDS (GraphEdX) | 11 | 11 | 1.00 | 78 | 66 | **55** | 60 | adjacency |
| GREC | 11 | 12 | 1.09 | 84 | 66 | **55** | 70 | adjacency |
| AIDS (IAM) | 14 | 15 | 1.07 | 102 | 102 | 91 | **82** | **IsalGraph** |
| COIL-DEL | 22 | 54 | 2.45 | 348 | 240 | **231** | 418 | adjacency |
| **Mutagenicity** | 29 | 30 | 1.03 | 222 | 414 | 406 | **181** | **IsalGraph** |
| Protein | 32 | 61 | 1.91 | **396** | 504 | 496 | 533 | **sparse6** |

> **The ordering is governed by density, not size.** Both sparse6 and IsalGraph scale with `m`;
> adjacency, graph6, nauty→graph6 and AGM all scale with `n²`. So:
>
> - below `n ≈ 14` the `n²` formats win outright — `n(n−1)/2` is simply small;
> - above it, the `m`-scaling formats win, and **which of the two** depends on `m/n`;
> - **IsalGraph beats sparse6 at `m/n ≈ 1` (Mutagenicity: 181 vs 222) and loses at `m/n ≈ 2`
>   (Protein: 533 vs 396; COIL-DEL: 418 vs 348).**
>
> [competitors](../competitors.md) §4 outcome 3 predicts the opposite and must be rewritten. The
> defensible pre-commitment is: *IsalGraph's bit cost is governed by `m`, so it wins against the
> `n²` serialisations once `m ≪ n²/2`, and against sparse6 only near `m ≈ n`.* Stated that way it
> survives whichever direction the real cohort falls, and it is a sharper claim than "compact".

**Conventions.** Entropy bound `6 · len(sparse6)`; realised `8 · len(sparse6)`. The `':'` prefix is
one byte of framing, not payload — **count it in the realised figure and exclude it from the
entropy bound**, and say which you did.

---

## 5. Scope alignment and IsalGraph's advantage

**Aligned and load-bearing.** sparse6 is the strongest possible answer to R3.6a: a published,
implemented, reversible serialisation designed for exactly the regime IsalGraph claims — sparse
graphs. If we omitted it, the compactness claim would rest on beating formats that were never
trying to be compact.

**IsalGraph's advantage over sparse6:**

| Axis | Verdict |
|---|---|
| Uniqueness | **IsalGraph.** 40/40 vs 0/40 invariant; sparse6 is not canonical and its authors never claimed it was |
| Expressiveness | **sparse6.** Handles disconnected graphs, isolated vertices, loops and multi-edges; IsalGraph raises `DisconnectedGraphError` |
| Efficiency | **sparse6**, by 1–2 orders of magnitude (0.028–0.157 ms vs 0.01–1.02 ms), and it has no failure mode |
| Scalability | **sparse6.** `Θ(m log n)` with no search |
| Downstream learning | **not evaluated** — R1.2b's fifth axis, reported as such |
| Bits | **density-dependent**, table above |
| Edit-distance compatibility | **IsalGraph** (0.73 vs 0.88), though both are poor; see [gspan-mdfsc](gspan-mdfsc.md) for the one that is good |

**The honest one-line claim**: IsalGraph is *canonical* and *edit-distance-comparable*; sparse6 is
neither, and is faster and more compact on dense graphs. That is a real contribution and it is not
the same claim as "shortest".

---

## 6. Summary

| # | Question | Answer |
|---|---|---|
| 1 | Reproducible? | **Yes, trivially.** `networkx`, both directions, 0.028–0.157 ms — cheapest in the pool |
| 2 | Representation | `m`-scaling edge stream; **not** isomorphism-invariant (0/40, 123 codes from 200 relabellings) |
| 3 | Distance | Hamming defined on **30.8 %** of pairs (fails F1); Levenshtein defined but **separation 0.88** |
| 4 | Claim A? | **Yes, and it is the key row.** Beats IsalGraph at `m/n ≈ 2`, loses at `m/n ≈ 1` |
| 5 | Scope | **In.** The compactness rival R3.6a is really asking for |
| — | IsalGraph advantage | **Yes on canonicity and on GED tracking; no on cost, robustness, or bits at high density** |

---

## 7. For the integration agent

- **Strip the trailing newline** from `to_sparse6_bytes`, and decide once whether the `':'` counts.
  Both are 8 realised bits per graph and both will otherwise drift between scripts.
- Register **two** backends: `sparse6` and `sparse6-nauty` (canonical relabelling first). The
  second costs one line and removes a reviewer objection.
- **Do not attempt padded Hamming on sparse6.** There is no positional frame. Emit `undefined` and
  print it in the supplementary grid — a failed cell is a result
  ([competitors](../competitors.md) §3.2).
- `networkx` emits sparse6 with `k = ⌈log₂ n⌉`; for `n` a power of two the spec has an off-by-one
  special case. Suite 2 contains graphs at `n = 16, 32, 64` — **assert round-trip equality on
  every encode** rather than trusting the length formula.
