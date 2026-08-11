# Competitor representations — backends and metric feasibility

**Status**: v1.0, 2026-08-11. Owner: **T-04** (backends) and **T-04a** (metric feasibility).
Companion to `plan.md` §4, which holds the vendoring architecture and the placement of competitors
into the three experiments.

Serves **R1.1** (a comparator in a similar problem setting), **R1.2** (AGM, gSpan, and the five
axes), **AE.3** (the side-by-side comparison the Area Editor endorsed) and **AE.4a** (choice of
benchmark models).

---

## 1. Why this file exists

`plan.md` §4.2 assigned a distance to each competitor by inspection: Hamming for the fixed-format
serializations, Levenshtein for the variable-length ones. The 2026-08-11 audit found that **plain
Hamming is undefined for most pairs** (`gap-audit.md` MF6) — graph6 encodes `n` in its header and
packs `n(n−1)/2` bits, so graphs with different node counts produce strings of different length. Node
counts run 2–12 in Suite 1 and 2–98 in Suite 2; equal-`n` pairs are a small minority.

The first fix proposed a padding convention. **That was a unilateral choice and is withdrawn.** The
distance assignment is not something to decide by argument when it can be measured — and it must not
be decided *after* seeing which assignment flatters IsalGraph.

**Locked instead: T-04a measures which metrics are well-defined, computable and meaningful for each
representation, on a fixed sample, before any production distance matrix is computed.**

---

## 2. T-04a — the metric feasibility experiment

### 2.1 Sample

Fixed and small: **200 graphs stratified by node count** across the ten locked datasets
(`data.md` §0), drawn with seed 42, plus their `C(200,2) = 19,900` pairs. Strata: `n ∈ [2,5]`,
`[6,9]`, `[10,12]`, `[13,20]`, `[21,40]`, `> 40`, so the unequal-`n` case is guaranteed to dominate
exactly as it does in production.

### 2.2 Grid

Every (representation × distance) cell is attempted, including the ones expected to fail. A cell that
fails is a **result**, and one that a reviewer would otherwise ask about.

| Representation | Produces | Candidate distances |
|---|---|---|
| adjacency matrix (row-major, node order as given) | fixed-length bit string per `n` | Hamming, padded Hamming |
| **graph6** | printable ASCII, length `≈ n(n−1)/12` | Hamming, padded Hamming, Levenshtein |
| **sparse6** | printable ASCII, length `≈ m log n / 6` | Hamming, padded Hamming, Levenshtein |
| **nauty**-canonical relabelling → graph6 | as graph6, canonical | Hamming, padded Hamming, Levenshtein |
| bliss / Traces → graph6 | as above | as above |
| AGM canonical code | variable-length string | Levenshtein |
| **gSpan minimum DFS code** | variable-length string | Levenshtein |
| **IsalGraph** pruned canonical | variable-length string | Levenshtein |
| WL subtree kernel | feature vector | kernel distance (already computed, E10) |

`padded Hamming` = embed both adjacency matrices in a common `max(n₁, n₂)` frame and compare upper
triangles. It is a *candidate*, not the answer — its justification is that the padding coincides with
the node-insertion operation D6 charges, and §2.3 tests whether that justification survives contact
with data.

### 2.3 What is measured per cell

| # | Criterion | Test | Fails if |
|---|---|---|---|
| **F1** | **Well-defined** | fraction of the 19,900 pairs for which the distance is computable at all | any pair is undefined — the cell is reported as *undefined for X % of pairs* and cannot be a primary metric |
| **F2** | **Metric axioms** | identity, symmetry, and triangle inequality sampled over 5,000 random triples | a violation. IsalGraph's Cor. 2.13 claims metricity; comparing against a non-metric competitor distance is legitimate but must be **declared** |
| **F3** | **Isomorphism-invariant** | 50 graphs × 20 random relabellings each; distance to self must be 0 | non-zero — the representation is order-dependent and the comparison measures node ordering, not structure |
| **F4** | **Non-degenerate** | fraction of distinct pairs at distance 0, and the coefficient of variation | mass at 0, or near-constant. A distance that cannot separate is not a baseline |
| **F5** | **Tracks GED at all** | Spearman ρ against exact GED on the `n ≤ 12` subsample | ρ ≈ 0 with a *well-defined, invariant* metric is a genuine finding, not a failure — see §3 |
| **F6** | **Affordable** | µs/pair at `n̄ = 30` | > 1 ms/pair — the same ceiling `plan.md` §7.3 applies to the GED upper bound |

### 2.4 Selection rule, fixed in advance

> For each representation, the **primary** distance is the cheapest one that passes **F1 at 100 %**,
> **F2**, **F3** and **F4**. If more than one qualifies, the tie is broken by **F6**, not by **F5** —
> selecting on correlation with GED would be selecting the baseline that makes IsalGraph look best.
>
> If **no** distance passes for a representation, that representation enters the **AE.3 comparison
> table** on its qualitative properties and is **excluded from the running comparison**, with the
> reason printed.

Every attempted cell is reported in supplementary regardless of outcome, so the selection is
auditable rather than asserted.

---

## 3. Two outcomes that are results, not failures

**Non-canonical graph6 should correlate poorly.** graph6 without canonical relabelling is
order-dependent, so it should fail **F3** outright. That is exactly the point R1.2 asks about — it
isolates *why* a representation must be canonical before an edit distance on it means anything — and
it is the cleanest available demonstration that IsalGraph's canonicalisation is doing work.

**Canonical graph6 under Hamming may also correlate poorly**, for a different reason: nauty's
canonical form is chosen to be a unique representative, not a *stable* one, so two similar graphs can
receive very different labellings and their bit strings need not be close. If that holds, it
separates the two properties the paper actually claims — canonical **and** edit-distance-compatible —
and shows they are independent. **Pre-committed: report it either way.**

**And the converse must be pre-committed too.** If sparse6 beats IsalGraph on bits for sparse graphs
— the expected outcome, since sparse6 is a bit-packed format with no reversible-edit-distance
property — the contribution is stated as *canonical and edit-distance-compatible*, not *shortest*
(`plan.md` §4.2, MF7). Deciding that after seeing the numbers is not available to us.

---

## 4. Bit accounting for Claim A

Separate from distances, and equally in need of a stated convention.

`B_Isal(w) = L log₂ 9` (`computational_experiments.tex:157–160`) is an **entropy bound** on the
symbol stream. graph6 and sparse6 emit printable ASCII with **6 payload bits per byte**. Comparing an
entropy bound against a wire format flatters us, and R3 — who checked thirteen of thirteen checkable
claims — would find it.

**Locked: report both conventions for every method.**

| Convention | Definition |
|---|---|
| **Entropy bound** | `L log₂ |Σ|` for string representations; `n(n−1)/2` for the adjacency matrix |
| **Realised bytes** | the actual serialized length as the format defines it |

State which is primary and why. The entropy bound is the like-for-like comparison of *encoding
efficiency*; the realised byte count is what a practitioner stores. They answer different questions
and the paper should say so rather than quietly picking one.

---

## 5. Risk carried from `plan.md` §4.2

**gSpan's minimum DFS code may not be exposed.** `LasseRegin/gSpan` is a frequent-subgraph miner; the
minimum DFS code of a single graph is an internal sub-component. **Verify on day 1 of T-04.** If it is
not reachable, extract or reimplement within the same 2–3 day budget; if that slips, gSpan is
**discussed** in the related-work section and the running comparator set drops to nauty-graph6,
sparse6 and AGM. R1.2 is answered by citation and by the AE.3 table either way — only the empirical
row is lost.

---

## 6. Acceptance criteria

1. The full (representation × distance) grid is attempted and reported, failures included.
2. Every primary distance is selected by the §2.4 rule, **before** any production matrix is computed.
3. No representation reaches a results table on a distance that fails F1, F2, F3 or F4.
4. Both bit conventions reported for every method in Claim A.
5. The two pre-committed outcomes in §3 are stated in the paper regardless of which way they fall.

---

## 7. Change log

| Date | Ver | Change |
|---|---|---|
| 2026-08-11 | v1.0 | Created at author request, replacing the unilateral padded-Hamming decision in `plan.md` §4.2 with a measured selection (T-04a). Absorbs `gap-audit.md` MF6 and MF7 |
