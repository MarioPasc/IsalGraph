# Competitors — representations, backends and metric selection

**Owner**: T-04 (backends), **T-04a** (metric feasibility), T-17 (the comparison table)
**Serves**: **AE.4a** (requirement modal, the real owner), AE.3, R1.1, R1.2a/b, R3.6a's expensive branch

Related: [tickets](tickets.md) · [statistics](statistics.md) · [demands](demands.md) ·
[schedule](schedule.md) (what gets cut first)

> ⚠ **Scouted 2026-08-13 on the real cohort. The evidence behind this file is now
> [`competitors/`](competitors/README.md)** — one file per competitor, every backend installed and
> run against Suite 1's certified exact GED and Suite 2's IAM GXL. Its §5 lists fifteen findings
> with owners. **Superseded here**: §2's AGM effort estimate and its "derive from nauty labelling"
> premise (both wrong — [agm](competitors/agm.md) §2.3), §2's gSpan vendoring plan (three
> repositories tested, all three rejected — [gspan-mdfsc](competitors/gspan-mdfsc.md) §1), §4's
> pre-committed outcome 3 (**inverted** — [sparse6](competitors/sparse6.md) §4), and §2's
> bliss/Traces counter-case (**expired** — [nauty](competitors/nauty.md) §8).
>
> **§3.4's selection rule survives and is vindicated**: the raw adjacency matrix scores ρ = 0.75–0.87
> against exact GED while failing F3, which is exactly the trap an F5-blind, F3-gated rule prevents.
> **§4 needs a fourth pre-committed outcome**: `ρ(|n₁−n₂|, GED)` is 0.71–0.93 and IsalGraph clears
> it on two of five datasets. §5 is unaffected.

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

> ## ⚠ RESULT 2026-08-16 (T-04a). §3.4's rule as written is incomplete and was repaired before use.
>
> **A candidate distance must read the representation.** §3.2's table is the candidate set, and
> `size_null` appears in none of its rows — but the rule as *implemented* ranged over every
> registered metric. Measured on the frozen draw, `size_null` (`|n₁−n₂|`) passes F1 at 100 %, F2, F3
> at **50/50** and F4 on **every** backend, and is **10.9× cheaper** than `levenshtein`, so "the
> cheapest that passes F1–F4" would have named *count the nodes and subtract* the primary distance of
> all eleven representations. `levenshtein_char` is a second instance, **3.4× cheaper** than
> `levenshtein` for `isalgraph_pruned`.
>
> The rule now ranges over metrics with `consumes ∈ {symbols, frame, features}`. Every cell is still
> measured and printed; ineligibility is recorded, not hidden. **`k = 3`**: `adjacency`, `graph6` and
> `sparse6` have no admissible distance, each failing F3 at **1/50**.
>
> Also measured: **F1 and encodability are different axes and §3.3 conflated them.** A distance being
> undefined on a pair is a property of the distance; a representation failing to encode a graph is a
> property of the representation, and `preregistration.md` §5 charges them differently. They are now
> reported separately as F0 and F1.

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


---

## 8. RESULT — T-04, closed 2026-08-15

**Built, not run.** T-04 ships the machinery; T-04a runs the grid, T-06 the production matrices,
T-17 the AE.3 table. Full API: `src/isalgraph/competitors/README.md`.

| Delivered | |
|---|---|
| Backends | **11 registered**, 10 available (`size_null` filtered as a baseline), 0 unavailable |
| Metrics | 6 — `levenshtein`, `levenshtein_char`, `hamming`, `padded_hamming`, `kernel`, `size_null` |
| Tests | **383**, in six files; full suite **2,106 passed / 321 skipped**; ruff + `mypy --strict` clean |
| Size | 26 files, **+9,510 / −20** |

### Did the pre-declared rules fire as written?

| Rule | Fired |
|---|---|
| §3.4's selection rule is **F5-blind by construction** | **yes, structurally.** `grid.py` computes F1–F4 and F6; `f5.py` is the only entry point that can reach a GED loader, and a test asserts `grid.py`'s **import closure** never does. Decision 24 is defensible on the import graph, not on prose |
| §3.2 "every cell attempted; a failure is a result" | **yes.** The 20-graph dry run emits all 66 cells. `padded_hamming × sparse6` is **undefined** and prints as such — the cell §3.2 was written to catch |
| §5 "both bit conventions for every method" | **yes**, and a defect was found doing it: `realised_bits` for `adjacency`/`agm_cam` was **halved** (`8·⌈T/16⌉` for `8·⌈T/8⌉`). Found independently by two tracks |
| §4 outcome 1, "non-canonical graph6 should fail F3" | **confirmed, and explained.** It fails on exactly the non-complete graphs — see [competitors/README](competitors/README.md) §3 |
| §4 outcome 2, "canonical graph6 may also correlate poorly" | **confirmed.** nauty→graph6 is the weakest of the three canonical serialisations, 0.42–0.68 all-pairs |
| §4 outcome 3, "sparse6 beats IsalGraph on sparse graphs" | **inverted**, as finding 4 already recorded; it resolves on `m/n`, not size |
| §4's missing **fourth** pre-committed outcome, the size null | **now measured and it is the headline.** See §1 of [T-04 article notes](../tasks/T-04-article-notes.md) — IsalGraph clears it on one of five, and on none by a margin that survives resampling |

### Standing requests this file made of T-04, answered

- *"Verify on day 1 whether `LasseRegin/gSpan` exposes the minimum DFS code"* (§2 Risk) — **answered
  by the scout and confirmed here: vendor nothing.** Three repositories tested, three rejected. The
  min-DFS code is ours, with V1/V2/V3 oracles; distinct codes at `n = 2…6` are **1/2/6/21/112**
  (OEIS A001349, zero collisions), and the `kaviniitm` gate ships as an acceptance test any future
  third-party candidate must pass, **K2 first**.
- *"T-04a decides the distance"* (§2 table) — **the machinery is shipped and unrun.** `grid.py`
  proves it runs end to end on a 20-graph dry run; T-04a runs the 200-graph stratified sample.
- *"`GEDBackend`"* (§1) — **deliberately not built.** GEDLIB lives in
  `benchmarks/real_data/eval_setup/ged_backends.py` (T-27); a second one would fork the cost model.

### Artifacts

| File | From |
|---|---|
| `.claude/notes/2026-08-14-t04-competitors/corrected_rho_table.json` | `reproduce --mode table` — **quote this, not §4.1/§4.2** |
| `.../repro_artefacts.json` | `reproduce --mode artefacts` — 40/40 Suite-1 cells at delta `0.00e+00` |
| `.../smoke_picasso_suite{1,2}.json` | the loginexa run; `pynauty` from-source gate |
| `.../agm_ceiling_B.json` | AGM's ceiling, `agm.md` §2.2b |
| `.../{summary,VERIFICATION,WAVE0-FINDINGS,PICASSO,CONTRACTS}.md` | the wave record |

**Article notes**: [T-04-article-notes.md](../tasks/T-04-article-notes.md) — read its
*"What is NOT claimable"* section before quoting anything from this folder.

---

## 9. RESULT — T-04a, closed 2026-08-23

**Run, not built.** All **66 cells** measured on one frozen pooled draw; the E1–E4 admissibility
annex run on top of it. No cluster: the whole campaign is a workstation job.

**The sample `S200`**, frozen by the design note before any agent started and reproducible from
`(ALL_DATASETS, 200, 42)` alone: 200 graphs, six node-count strata `[33, 33, 33, 33, 34, 34]`,
`n` 2–83, suite split **51 / 149**. LINUX draws zero graphs — 89 graphs at `n ∈ [4,10]` against
thousands in the strata it falls in. That is a property of stratifying on node count, and it is
reported rather than repaired.

### 9.1 The selection table — §3.4's rule, as it fired

| representation | primary distance | why |
|---|---|---|
| `adjacency` | **none admissible** | F3 = 1/50; `hamming` also F1 = 0.032 |
| `graph6` | **none admissible** | F3 = 1/50; `hamming` also F1 = 0.035 |
| `sparse6` | **none admissible** | F3 = 1/50; `padded_hamming` F1 = **0.0** — no positional frame |
| `nauty_graph6` | `levenshtein` | F6 tie-break over `padded_hamming`, **68×** cheaper |
| `agm_cam` | `levenshtein` | F6 tie-break over `padded_hamming`, **8.6×** cheaper |
| `sparse6_nauty` | `levenshtein` | sole candidate passing F1–F4 |
| `min_dfs` | `levenshtein` | sole candidate passing F1–F4 |
| `isalgraph_pruned` | `levenshtein` | sole candidate passing F1–F4 |
| `isalgraph_canonical` | `levenshtein` | sole candidate passing F1–F4 |
| `wl_subtree` | `kernel` | sole candidate passing F1–F4 |
| `size_null` | none | baseline; no candidate distance exists for it |

**`k = 3`** over preregistration §4.1's seven-member Claim-B set (`graph6`, `sparse6`,
`nauty_graph6`, `adjacency`, `agm_cam`, `min_dfs`, `wl_subtree`). **`padded_hamming` is primary for
nothing**, which retires §3's provisional column and the same claim in
[nauty](competitors/nauty.md) and [agm](competitors/agm.md), all three corrected in place.

**The rule fired as written on the metrics it was allowed to range over** — but only after the
repair recorded in the §3 RESULT block above. As shipped it ranged over `size_null` and
`levenshtein_char` and would have named *count the nodes and subtract* the primary distance of all
eleven representations.

### 9.2 F0 — encodability, **at each backend's own budget**

`timeout_s = 2.0`, `max_projections = 50,000`, `search_nodes = 200,000`. **Not** D14's 300 s. A
different budget gives a different F0, so the budget travels with the number.

| representation | F0 overall | breakdown |
|---|---:|---|
| `agm_cam` | 98/200 | 101 `SuiteScopeError` (scope), 1 `AGMBudgetExceeded` (budget) |
| `isalgraph_canonical` | 99/200 | 101 `SuiteScopeError` — a capability refusal, not a timeout |
| `isalgraph_pruned` | 191/200 | 9 `CanonicalizationTimeoutError` |
| `min_dfs` | 192/200 | 8 `MinDfsBudgetExceeded` |
| the other seven | 200/200 | — |

All four are **1.00 on Suite 1**; every exclusion is a Suite-2 graph. **`SuiteScopeError` must never
be summed with a budget error** — the first is a decision, the second an outcome, and
`preregistration.md` §5 charges them differently.

### 9.3 F5 — computed after selection was written to disk, never fed back

`paired_null_ci.json` is the authority for every interval: paired graph-level bootstrap, 2,000
resamples, seed 42, both arms on the identical pair set and the identical resamples.

| | IsalGraph − size null, all-pairs |
|---|---|
| **Suite 1, exact GED** | Letter LOW **+0.026 [+0.008, +0.046]** · MED −0.044 · HIGH −0.220 · LINUX −0.239 · AIDS −0.528 |
| **Suite 2 vs LB** (`BRANCH_FAST`) | GREC −0.214 · AIDS-IAM −0.248 · COIL-DEL −0.082 · Mutagenicity −0.289 · Protein −0.233 |
| **Suite 2 vs UB** (`BIPARTITE`) | GREC +0.122 · AIDS-IAM +0.027 · COIL-DEL +0.197 · Mutagenicity +0.196 · Protein +0.383 |

**All 15 differences exclude zero.** IsalGraph clears the size null on **1 of 5** Suite-1 datasets
and on **6 of 15** records overall, and is the best representation on **none** of the 15
(`min_dfs` 4, `agm_cam` 3, `wl_subtree` 3, `sparse6_nauty` 3, `nauty_graph6` 2).

**The verdict flips with the end of the bracket on 5 of 5 Suite-2 datasets**, and the mechanism is
measured: `ρ(|n₁−n₂|, LB) = 0.960–0.998`, so the lower bound very nearly *is* the size null and no
representation can beat it. That is degenerate by construction, **not a defect in `BRANCH_FAST`**,
which is a proven bound T-27 selected on measurement. `ρ(|n₁−n₂|, UB) = 0.460–0.754`; on Suite 1,
where truth exists, `ρ(|n₁−n₂|, exact) = 0.713–0.920` — **between** the two arms. This is the
empirical case for [approx_ged](approx_ged.md) §4's no-interpolation rule.

**Supersession, as §3.7 fixed it before the outcome was known**: F5 reproduces
`corrected_rho_table.json` at **max |delta| = 0.0000**, so §3.7's tie-break never had to fire.

### 9.4 The E1–E4 annex

| | measured |
|---|---|
| **E1 — ψ**, relabelling sensitivity under `levenshtein` | **0.0000** on all eleven draws for the seven canonical representations (77/77 rows). Excluded three: `adjacency` 0.07–0.74, `graph6` 0.32–**1.003** [0.953, 1.054] on LINUX, `sparse6` 0.54–**1.148** [1.111, 1.187] on AIDS. All 33 intervals exclude 0. Under `padded_hamming`, pooled `adjacency` ψ = 0.988 vs 0.072 — **14×**, so the padding convention makes it worse, not better |
| **E1 — exhaustive** | the invariant set of the n² family is **exactly `{K_n}`** over all 995 connected graphs to `n = 7`, full `n!` enumeration, **1,866,256** distinct labelled graphs (OEIS A001187) |
| **E2 — completeness** | six complete invariants: **0 collisions**, and the zero set is **≡** the VF2-certified isomorphic set; ≤ 2.0–3.4 × 10⁻⁵ by rule of three. `wl_subtree`: **45 / 183,016 = 2.46 × 10⁻⁴** [1.79, 3.29] × 10⁻⁴. On LINUX and AIDS, which hold no duplicate graphs, **every zero WL emits is a false isomorphism certificate** — 1/1 and 11/11, AIDS CI [0.715, 1.000]. K₃,₃ vs the triangular prism: WL exactly 0.0, all six canonical separate them |
| **E3 — axioms** | **0 violations in 9,881,851 checks** over all **467,180** triples of the 142 connected graphs on `n ≤ 6`, `worst_excess = 0.0` exactly. A correctness check on our implementation, **not a discovery** |
| **E4 — the trap** | `adjacency` beats IsalGraph **significantly on 3 of 5** all-pairs (AIDS **+0.500** [+0.443, +0.558], LINUX **+0.262**, Letter HIGH **+0.123**, all Holm p = 0.005) and on **0 of 5** equal-`n`, losing significantly on three. `ρ(d_adjacency, \|n₁−n₂\|) = 0.83–0.93` |

**E4 is why §3.4's rule is F5-blind.** A representation that fails F3 at 1/50 — that is, one whose
distance changes when you relabel the graph and nothing else — nonetheless out-correlates IsalGraph
against exact GED on three of five datasets, because both it and GED are dominated by size. A
selection rule with sight of F5 would have picked it.

### 9.5 §4's three pre-committed outcomes, answered

1. **"Non-canonical graph6 should fail F3 outright."** **Confirmed, and it generalises**: all three
   non-canonical serialisations fail at **1/50**, and E1 quantifies the failure continuously —
   ψ up to **1.148**, an edit distance larger than the strings' own scale.
2. **"Canonical graph6 under Hamming may also correlate poorly ... it separates the two properties."**
   **Confirmed in the stronger form.** The separation is not between two grades of correlation but
   between ψ = 0 and ψ > 0: `nauty_graph6`, `agm_cam` and `sparse6_nauty` are exactly relabelling-
   invariant, and their correlations then range 0.46–0.99 across records. Canonical **and**
   edit-distance-compatible are independent properties, and E2 supplies the second half — six
   complete invariants at 0 collisions against WL's 45.
3. **"If sparse6 beats IsalGraph on bits."** Settled by T-04, unchanged here. What T-04a adds is that
   sparse6 buys those bits by having **no admissible distance at all**.

### 9.6 Artifacts

| Location | |
|---|---|
| `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/experiments/metric-admissibility/` | **canonical, self-contained** — `PROTOCOL.md`, `README.md`, `results/` (7 JSON), `code/` (13 files), `manifest.json` with sha256 over 22 files |
| `/media/mpascual/Sandisk2TB/research/isalgraph/T-04a/` | the run directory, with the `.log` files; the JSONs are byte-identical to the deliverable |

`paired_null_ci.json` is the authority for every interval quoted anywhere.

### Standing requests this file made of T-04a, answered

- *"T-04a decides the distance"* (§2 table, §8) — **decided**, §9.1. `levenshtein` for six,
  `kernel` for `wl_subtree`, none for three.
- *"T-04a must close before any production distance matrix is computed — it gates T-06"* (§3) —
  **released**. T-06 has the table.
- *"The three pre-committed outcomes in §4 are stated in the paper regardless of which way they
  fall"* (§7 criterion 5) — answered in §9.5 and carried to the article notes.
- *"No representation reaches a results table on a distance that fails F1, F2, F3 or F4"*
  (§7 criterion 3) — enforced: `adjacency`, `graph6` and `sparse6` carry **no** distance column.
  E4 is the measurement of what criterion 3 buys.

**Article notes**: [T-04a-article-notes.md](../tasks/T-04a-article-notes.md) — read its
*"What is NOT claimable"* section before quoting anything from this run.
