# graph6 — McKay's 6-bit-packed adjacency serialisation

**Verdict**: **RUN — as the negative control.** Reproducible in one line; fails isomorphism
invariance on **0 of 40** graphs. Its failure is the finding, not a defect in the experiment.

**Role**: [competitors](../competitors.md) §2 row 1 · serves **AE.4a**, **AE.3**, **R3.6a**
**Evidence**: measured on this workstation, 2026-08-13, `scratchpad/competitors/{probe,sweep,scale,stability}.py`.
Cross-refs: [nauty](nauty.md) (the canonical variant), [adjacency-matrix](adjacency-matrix.md)
(the same bits unpacked), [README](README.md) §2 (the four-member family).

---

## 1. Reproducibility — is anything blocking?

**Nothing.** `networkx` ≥ 2.0 implements both directions; it is already a dependency of
`isalgraph.adapters`.

```python
import networkx as nx
nx.to_graph6_bytes(G, header=False)      # encode
nx.from_graph6_bytes(b"ElCW")            # decode -- exact, not up to isomorphism
```

Measured: **0.035–0.338 ms per graph** across the whole cohort profile range (n = 4 to 32).
No compilation, no external binary, no cluster step. `networkx` 3.6.1 verified.

> **The format is a specification, not a package**: Brendan McKay,
> <https://users.cecs.anu.edu.au/~bdm/data/formats.txt>. Cite the `nauty`/Traces paper for it —
> McKay & Piperno, *Practical graph isomorphism, II*, **J. Symb. Comput. 60:94–112, 2014**,
> doi:10.1016/j.jsc.2013.09.003. There is no separate graph6 paper to cite.

---

## 2. What the representation looks like

Running example throughout this folder:
**`G` = a 4-cycle `(0,1,2,3)` sharing node 3 with a triangle `(3,4,5)`. `n = 6`, `m = 7`,
`|Aut(G)| = 4`.**

```
graph6(G)                     'ElCW'                     4 bytes
graph6(G relabelled)          'ElCW' … or 121 other values -- see below
graph6(H = G minus edge 0-3)  'EhCW'
```

**Structure.** `N(n)` (one byte, `n + 63`, for `n ≤ 62`) followed by `R(x)`, where `x` is the
strict upper triangle of the adjacency matrix read **column-wise** —
`a(0,1) a(0,2) a(1,2) a(0,3) a(1,3) a(2,3) …` — packed 6 bits to a printable ASCII byte at
offset 63. Length is exactly `1 + ⌈n(n−1)/12⌉` bytes for `n ≤ 62`.

### The property that decides everything

**graph6 serialises the labelling it is handed.** It performs no canonicalisation.

| Test | Result |
|---|---|
| 200 relabellings of the running example (`\|Aut\| = 4`, so ≤ 180 distinct labellings exist) | **122 distinct graph6 strings** |
| F3 sweep, synthetic: 40 graphs × 25 genuine relabellings each | **0 / 40 invariant**; up to **26 distinct codes** from 26 copies of one graph |
| **F3 on the real cohort**: 50 graphs × 20 relabellings, per dataset | Letter LOW **4/50** · MED **2/50** · HIGH **6/50** · LINUX **0/50** · AIDS **0/50** |

> The handful of Letter successes are **not** partial invariance — they are tiny graphs (`n̄ ≈ 4`)
> with large automorphism groups, where 20 draws can miss every distinguishable labelling. Report
> the LINUX/AIDS rows (0/50) as the representative result and the Letter rows as the reason F3
> must be run on graphs large enough to have distinguishable labellings.

> A relabelling produced by `nx.relabel_nodes(copy=True)` alone **preserves insertion order** and
> makes order-dependent formats look invariant. Every measurement here rebuilds the copy with a
> fresh insertion order. The single-relabelling coincidence in §2's first table is exactly that
> trap: one draw out of ~180 happened to collide. **Do not report a single-relabelling check.**

| Property | graph6 | IsalGraph pruned |
|---|---|---|
| Reversible | **yes**, exactly | yes, up to isomorphism |
| Isomorphism-invariant | **no** (0/40) | **yes** (40/40) |
| Complete invariant | no | yes, within a directedness class |
| Fixed finite alphabet | yes, 64 printable ASCII | yes, `\|Σ\| = 9` |
| Length | deterministic, `1 + ⌈n(n−1)/12⌉` | data-dependent |
| Handles disconnected | **yes** | no (`DisconnectedGraphError`) |
| Handles isolated vertices | **yes** | n/a |
| Encode cost | `Θ(n²)`, 0.03–0.34 ms measured | 0.01–1.02 ms measured (pruned) |

---

## 3. Which distance does it accept?

| Candidate | Verdict | Why |
|---|---|---|
| Hamming | **defined only for equal `n`** | length is a function of `n` alone, so it is defined on 100 % of equal-`n` pairs and 0 % otherwise. Suite 2 spans `n` = 2…98, so equal-`n` pairs are a minority |
| padded Hamming | defined for all pairs | embed both triangles in a common `max(n₁,n₂)` frame — but see below |
| Levenshtein | defined for all pairs | measured |

**All three fail F3 before F1 is even reached.** Measured over 120 one-edit pairs (exact unit
GED = 1, edited copy randomly relabelled) against 120 random same-`n` pairs:

| | median Levenshtein, GED = 1 | median Levenshtein, random pair | **separation** |
|---|---:|---:|---:|
| graph6 | 6.0 | 6.0 | **1.00** |

A separation of **1.00** means a one-edit pair is indistinguishable from an unrelated pair.

**On the real cohort**, ρ of Levenshtein against certified exact GED, and the same ρ after every
graph is independently relabelled at random:

| | Letter LOW | Letter MED | Letter HIGH | LINUX | AIDS |
|---|---:|---:|---:|---:|---:|
| graph6, all pairs | 0.691 | 0.681 | 0.670 | 0.507 | 0.456 |
| graph6, **after random relabelling** | 0.676 | — | 0.676 | **0.377** | **0.387** |
| graph6, **equal-`n` pairs only** | **0.539** | **0.430** | **0.447** | **0.286** | **0.171** |
| *(size null `\|n₁−n₂\|`)* | *0.899* | *0.909* | *0.926* | *0.713* | *0.799* |

> graph6 never beats the size null, and on equal-`n` pairs — where the size channel is constant —
> it drops to **0.17–0.54**. Part of what remains is the corpus's own labelling convention: random
> relabelling costs a further **0.13 on LINUX** and **0.09 on AIDS**. Nothing left is structure.

**graph6 under any string distance carries no usable GED signal.** By §3.4's rule it takes no
primary distance and is excluded from the running comparison, entering the AE.3 table on its
qualitative row.

---

## 4. Fit for the information-content / message-length experiment (Claim A)

**Yes, and it is the cleanest member of the Claim A set.** Claim A needs a bit count, not a
distance, so F3's failure does not disqualify it ([preregistration](../preregistration.md) §5:
a representation with no admissible distance keeps its Claim A rows).

| Convention | Value |
|---|---|
| Entropy bound | `6 · (1 + ⌈n(n−1)/12⌉)` bits — the header plus the packed payload |
| Payload proper | `n(n−1)/2` bits, identical to the raw adjacency matrix |
| Realised bytes | `8 · (1 + ⌈n(n−1)/12⌉)` |

Measured entropy-bound bits, median of 5 synthetic graphs per Suite-2 profile:

| Profile | `n` | `m` | graph6 | adjacency | IsalGraph pruned |
|---|---:|---:|---:|---:|---:|
| Letter LOW | 4 | 3 | 12 | 6 | 13 |
| LINUX | 9 | 8 | 42 | 36 | 38 |
| AIDS (IAM) | 14 | 15 | 102 | 91 | **82** |
| COIL-DEL | 22 | 54 | **240** | **231** | 418 |
| Mutagenicity | 29 | 30 | 414 | 406 | **181** |
| Protein | 32 | 61 | 504 | 496 | 533 |

> **This retires R3.6a properly.** graph6 is a *published, implemented, reversible* serialisation,
> which is exactly what R3 asked to see beside the author-defined "GED standard construction". The
> comparison is now against something standard in fact rather than in adjective.

---

## 5. Scope alignment and IsalGraph's advantage

**Aligned, with its role stated.** graph6 is not a canonicalisation method and nobody claims it is;
including it as if it were a rival would be a strawman. Its job in the pool is to **isolate the
variable**: graph6 and nauty→graph6 are byte-identical in format and differ only in *which
labelling* is serialised. Putting them side by side answers R1.2's uniqueness axis by
subtraction — separation 1.00 versus 0.83 — instead of by assertion.

**IsalGraph's advantage over graph6 is foundational and measured**: canonicity (40/40 vs 0/40),
completeness as an invariant, and a bit count that grows with `m` rather than `n²`, which flips the
compactness ordering at low density (Mutagenicity 181 vs 414 bits) and against it at high density
(COIL-DEL 418 vs 240).

**What graph6 has that IsalGraph does not**: exact reversibility to the labelled graph, disconnected
and edgeless graphs, a deterministic length, and a decode that is `O(n²)` with no search.

---

## 6. Summary

| # | Question | Answer |
|---|---|---|
| 1 | Reproducible? | **Yes, trivially.** `networkx`, both directions, 0.03–0.34 ms |
| 2 | Representation | 6-bit-packed upper triangle; **not** isomorphism-invariant (0/40) |
| 3 | Distance | Hamming (equal-`n` only), padded Hamming, Levenshtein — **all separation ≈ 1.00, no signal** |
| 4 | Claim A? | **Yes.** `6(1 + ⌈n(n−1)/12⌉)` bits; wins below `n ≈ 14`, loses on sparse graphs above it |
| 5 | Scope | **In, as the control.** Isolates canonicity from serialisation format |
| — | IsalGraph advantage | **Yes**: canonical, complete, `m`-scaling instead of `n²`-scaling |

---

## 7. For the integration agent

- `ReprBackend.encode` → `nx.to_graph6_bytes(G, header=False).decode().strip()`. **Strip the
  trailing newline**; `to_graph6_bytes` appends one and it silently costs 8 realised bits per graph.
- ~~Normalise node labels first (`nx.convert_node_labels_to_integers(G, ordering="sorted")`)~~
  **CORRECTED 2026-08-15 by T-04: that call does not pin the labelling.** It renames node *values*
  and leaves insertion order alone, and `to_graph6_bytes` re-derives its labelling from insertion
  order. Measured: it disagrees with a genuine sorted rebuild on **290 of 300** scrambled graphs,
  and it made `graph6` and `sparse6` serialise *different* labellings. **Rebuild the graph** with
  `add_nodes_from(sorted(g.nodes()))` then `add_edges_from(g.edges())`. Normalise first so the
  backend is deterministic **on a given input labelling**. That is determinism, not invariance.
- `n > 62` uses the 4-byte `N(n)` form. Suite 2 tops out at 98, so **this branch is live** —
  test it. `networkx` handles it; the closed-form length above does not.
- Record the payload length `n(n−1)/2` separately from the byte length. §4's two conventions need
  both and they are not recoverable from each other after the fact.

---

## RESULT — T-04a, 2026-08-23. Outcome 1 confirmed, and generalised

**[competitors](../competitors.md) §4's outcome 1 — "non-canonical graph6 should fail F3 outright" —
is confirmed**, and it generalises to the whole non-canonical family: `adjacency`, `graph6` and
`sparse6` each fail F3 at **1/50** on the frozen `S200` draw and each take **no primary distance**.
`hamming` additionally fails F1 at **0.035**. graph6 is therefore the negative control the plan
designed it to be, and it earns that role on measurement rather than on argument.

**E1 quantifies it**: ψ, the median distance between a graph and a relabelled copy of itself, is
**0.32 – 1.003** for graph6 across eleven draws, peaking on LINUX at **1.003 [0.953, 1.054]**. All
eleven intervals exclude 0. The seven canonical representations are at **0.0000** throughout. A ψ
above 1 means the median self-distance under relabelling exceeds one edit — the representation
disagrees with itself by more than the unit it is supposed to measure in.

**Exhaustively**, the invariant set of the n² family — which graph6 belongs to — is **exactly
`{K_n}`**: over all 995 connected graphs to `n = 7` under full `n!` enumeration (**1,866,256**
distinct labelled graphs, OEIS A001187), no other graph has a relabelling-invariant serialisation.
There is no subfamily on which graph6 could be rescued. **Inherits: T-17, T-20.**
