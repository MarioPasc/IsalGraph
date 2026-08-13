# Competitors — representations, backends and metric selection

**Owner**: T-04 (backends), **T-04a** (metric feasibility), T-17 (the comparison table)
**Serves**: **AE.4a** (requirement modal, the real owner), AE.3, R1.1, R1.2a/b, R3.6a's expensive branch

Related: [tickets](tickets.md) · [statistics](statistics.md) · [demands](demands.md) ·
[schedule](schedule.md) (what gets cut first)

> ⚠ **Scouted 2026-08-13. The evidence behind this file is now [`competitors/`](competitors/README.md)
> — one file per competitor, every backend installed and run locally.** Its §4 lists twelve findings
> with owners. **Four sections of this file are superseded**: §2's AGM effort estimate and its
> "derive from nauty labelling" premise (both wrong — [agm](competitors/agm.md) §2.3), §2's gSpan
> vendoring plan (three repositories tested, all three rejected —
> [gspan-mdfsc](competitors/gspan-mdfsc.md) §1), §4's pre-committed outcome 3 (**inverted** —
> [sparse6](competitors/sparse6.md) §4), and §2's bliss/Traces counter-case (**expired** —
> [nauty](competitors/nauty.md) §8). §3, §3.4 and §5 are unaffected and remain authoritative.

> **Modal note that decides the scope.** R1.1's "would compare" is a **suggestion** and cannot carry
> six backends alone. It does not have to: **AE.4a** — "the choice of benchmark models" (`mail.txt:66`),
> weighted by the Area Editor's priority statement at `:67` — is **requirement-modal** and is the real
> owner. Cutting the competitors would leave a requirement the Area Editor singled out with no owner.

---

## 1. Architecture — follow IsalHG's `iso_backends`

`IsalHG/src/isalhg/iso_backends/` is the model: an ABC (`base.py`), a **lazy registry** keyed by name
(`registry.py`, `_LAZY_MODULES` so optional deps import only on request), a `subprocess_base.py` for
external binaries, and `BackendUnavailableError` on failure. Same idiom as IsalGraph's existing
`core/backends.py` (`BackendError`, never degrade silently).

**`src/isalgraph/competitors/`**, with **two** protocols — IsalHG's `IsoBackend` answers a different
question (fingerprint / are-isomorphic) than we need:

| Protocol | Methods | Implementations |
|---|---|---|
| `ReprBackend` | `encode(G) -> str\|bytes`, `bit_length(G) -> int`, `distance(a, b) -> float` | graph6, sparse6, nauty-canonical graph6, AGM code, **gSpan min-DFS code**, IsalGraph |
| `GEDBackend` | `ged(G, H) -> float`, `kind: 'exact'\|'upper'\|'lower'` | GEDLIB methods — see [gedlib](gedlib.md) |

Reuse IsalHG's `IsoBackend` shape for **nauty**: we need canonical relabelling anyway, to make the
graph6 comparison *fair* rather than a strawman.

---

## 2. The competitor set

| Competitor | Reversible | Canonical | String | Distance | Effort |
|---|---|---|---|---|---|
| graph6 | yes | only if relabelled | yes | **T-04a decides** | hours (`nx.to_graph6_bytes`) |
| **sparse6** | yes | only if relabelled | yes | **T-04a decides** | hours |
| **nauty** canonical labelling | yes | **yes** | via graph6 | **T-04a decides** | 1 d (`pynauty`) |
| adjacency matrix | yes | no | no | **T-04a decides** | trivial |
| AGM canonical code | yes | yes | yes | Levenshtein | 1 d, derive from nauty labelling |
| **gSpan minimum DFS code** | yes | **yes** | **yes** | **Levenshtein** | **2–3 d**, vendor `LasseRegin/gSpan` |
| WL subtree kernel | no | — | no | kernel | already computed (E10) |
| ~~bliss / Traces~~ | yes | yes | via graph6 | — | **CUT** — see below |

**gSpan's minimum DFS code is the single most important comparator**: canonical, a string,
edit-distance-comparable, named by R1, same problem setting.

> **bliss / Traces are cut (1.0 d, decision S-g).** They are absent from the `ReprBackend` set,
> functionally duplicate nauty — all three emit a canonical labelling serialised to graph6, differing
> in *speed*, not representation — produce no table row, and were requested by nobody. nauty alone
> discharges the "we need canonical relabelling anyway" rationale.
> *Counter-case, stated fairly*: they are cheap insurance if `pynauty` fails to build on Picasso,
> which would otherwise take the graph6 **and** AGM rows down with it.
> *Bonus*: cutting them also releases a bibliography slot ([compliance](compliance.md)).

> **Next component cut, if T-04 slips**: the **AGM `ReprBackend`** (1 d). R1.2a's owner is T-08
> because "the ask is *discussion*", and AE.3 is satisfied by a **qualitative** properties row in
> T-17. Not recommended — AGM is named by R1 by name and a measured row is more defensible than an
> asserted one — but it is the correct next candidate.

**Risk**: `LasseRegin/gSpan` is a *frequent-subgraph miner*; the minimum DFS code of one graph is an
internal sub-component and may not be exposed. **Verify on day 1 of T-04.** If unreachable, extract
or reimplement within the same 2–3 day budget. If that slips, gSpan is **discussed** in related work
and the running set drops to nauty-graph6 + sparse6 + AGM. R1.2 is answered by citation and by the
AE.3 table either way — only the empirical row is lost.

---

## 3. T-04a — the distance is measured, not asserted

**Why this exists.** The first plan assigned a distance to each competitor by inspection: Hamming for
fixed-format serialisations, Levenshtein for variable-length ones. **Plain Hamming is undefined for
most pairs** — graph6 encodes `n` in its header and packs `n(n−1)/2` bits, so graphs with different
node counts give strings of different length. Node counts run 2–12 in Suite 1 and 2–98 in Suite 2, so
equal-`n` pairs are a small minority. "Hamming on graph6 correlates poorly" would have recorded an
artefact of undefinedness as a finding — inside the comparison the Area Editor explicitly endorsed.

A padding convention was proposed and **withdrawn**: that was decided by argument when it can be
decided by measurement, and a distance chosen after seeing which choice flatters IsalGraph is not a
distance we can defend.

### 3.1 Sample

**200 graphs stratified by node count** across the ten locked datasets, seed 42, plus their
`C(200,2) = 19,900` pairs. Strata: `n ∈ [2,5]`, `[6,9]`, `[10,12]`, `[13,20]`, `[21,40]`, `> 40`, so
the unequal-`n` case dominates exactly as it does in production.

### 3.2 Grid — every cell attempted, including the ones expected to fail

A cell that fails is a **result**, and one a reviewer would otherwise ask about.

| Representation | Produces | Candidate distances |
|---|---|---|
| adjacency matrix | fixed-length bit string per `n` | Hamming, padded Hamming |
| graph6 | ASCII, length ≈ `n(n−1)/12` | Hamming, padded Hamming, Levenshtein |
| sparse6 | ASCII, length ≈ `m log n / 6` | Hamming, padded Hamming, Levenshtein |
| nauty-canonical → graph6 | as graph6, canonical | Hamming, padded Hamming, Levenshtein |
| AGM canonical code | variable-length string | Levenshtein |
| gSpan minimum DFS code | variable-length string | Levenshtein |
| IsalGraph pruned canonical | variable-length string | Levenshtein |
| WL subtree kernel | feature vector | kernel distance |

`padded Hamming` = embed both adjacency matrices in a common `max(n₁, n₂)` frame and compare upper
triangles. A **candidate**, not the answer — its justification is that the padding coincides with the
node insertion D6 charges, and §3.3 tests whether that survives contact with data.

### 3.3 What is measured per cell

| # | Criterion | Test | Fails if |
|---|---|---|---|
| **F1** | Well-defined | fraction of 19,900 pairs where the distance is computable | any pair undefined → reported as *undefined for X %*, cannot be primary |
| **F2** | Metric axioms | identity, symmetry, triangle inequality over 5,000 random triples | a violation. Comparing a metric against a non-metric is legitimate but must be **declared** |
| **F3** | Isomorphism-invariant | 50 graphs × 20 relabellings; distance to self must be 0 | non-zero → the comparison measures node ordering, not structure |
| **F4** | Non-degenerate | mass at distance 0; coefficient of variation | mass at 0, or near-constant |
| **F5** | Tracks GED | Spearman ρ against exact GED on the `n ≤ 12` subsample | ρ ≈ 0 with a well-defined invariant metric is a **finding, not a failure** |
| **F6** | Affordable | µs/pair at n̄ = 30 | > 1 ms/pair |

### 3.4 Selection rule, fixed in advance

> For each representation the **primary** distance is the cheapest that passes **F1 at 100 %**,
> **F2**, **F3** and **F4**. Ties are broken by **F6**, **never by F5** — selecting on correlation
> with GED would be selecting the baseline that makes IsalGraph look best.
>
> If **no** distance passes, that representation enters the **AE.3 comparison table** on its
> qualitative properties and is **excluded from the running comparison**, with the reason printed.

Every attempted cell is reported in supplementary regardless of outcome, so the selection is
auditable rather than asserted. **T-04a must close before any production distance matrix is
computed** — it gates T-06.

---

## 4. Three outcomes pre-committed as publishable

Deciding these after seeing the numbers is not available to us.

1. **Non-canonical graph6 should fail F3 outright.** That is exactly what R1.2 asks about — it
   isolates *why* a representation must be canonical before an edit distance on it means anything.
2. **Canonical graph6 under Hamming may also correlate poorly**, for a different reason: nauty's
   canonical form is a *unique* representative, not a *stable* one, so two similar graphs can receive
   very different labellings. If that holds, it separates the two properties the paper claims —
   canonical **and** edit-distance-compatible — and shows they are independent.
3. **If sparse6 beats IsalGraph on bits for sparse graphs** — the expected outcome, since sparse6 is
   a bit-packed format with no edit-distance property, and sparse graphs are exactly where IsalGraph
   claims compactness — the contribution is stated as *canonical **and** edit-distance-compatible*,
   not *shortest*.

---

## 5. Bit accounting for Claim A

`B_Isal(w) = L log₂ 9` is an **entropy bound** on the symbol stream. graph6 and sparse6 emit
printable ASCII with **6 payload bits per byte**. Comparing an entropy bound against a wire format
flatters us, and R3 — who checked thirteen of thirteen checkable claims — would find it.

**Locked: report both conventions for every method.**

| Convention | Definition |
|---|---|
| **Entropy bound** | `L log₂ \|Σ\|` for string representations; `n(n−1)/2` for the adjacency matrix |
| **Realised bytes** | the actual serialized length as the format defines it |

State which is primary and why. The entropy bound is the like-for-like comparison of *encoding
efficiency*; the realised byte count is what a practitioner stores. They answer different questions
and the paper should say so rather than quietly picking one.

---

## 6. Where each competitor lands in the paper

| Experiment | Gains | Retires |
|---|---|---|
| **(a) Message length** — §3.2.3 / Tab. 2 / Fig. 1 | bit cost for graph6, sparse6, nauty-graph6, adjacency, AGM, min-DFS | **R3.6a** — we stop calling our own model "standard" and put real serializations beside it |
| **(b) GED proxy** — §3.2.5 / Tab. 3 / Fig. 3 | ρ for Levenshtein-on-min-DFS, distance-on-nauty-graph6, WL | **R1.1** (proxy half) |
| **(c) Runtime** — §4.2 / Fig. 2 | encode-time curves for min-DFS and nauty | **R1.1** + the per-graph/per-pair category error |
| **(d) [28]/[29] delta** | **conceptual table only — no experiment** | R3.1a, AE.3, R3.7b |

Building an experiment for (d) would be a category error: it asks what we borrowed from our own prior
work, which is answered by reading the sources ([corrections](corrections.md), T-07).

---

## 7. Acceptance criteria

1. The full (representation × distance) grid is attempted and reported, failures included.
2. Every primary distance is selected by §3.4's rule **before** any production matrix is computed.
3. No representation reaches a results table on a distance that fails F1, F2, F3 or F4.
4. Both bit conventions reported for every method in Claim A.
5. The three pre-committed outcomes in §4 are stated in the paper regardless of which way they fall.
