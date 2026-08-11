# Revision plan — PR-D-26-03293 (IsalGraph)

**Status**: v0.4, 2026-08-11. Competitor architecture, GED strategy and dataset extension settled;
feasibility measured.
**Decision**: Major Revision, Pattern Recognition. **Due 2026-08-31 (20 days remaining).**
**Mode**: full recompute. Lock data → lock methods → lock statistics → recompute → write.
Page budget is **not** a constraint during drafting; trim at the end.

**Companion documents** — read these before executing any ticket:
- **`data.md`** — measured dataset inventory, exact/approximate GED cost, encoding cost, compute
  budget. All numbers measured 2026-08-11, not quoted.
- **`statistics.md`** — the proposed statistical protocol (Mantel, graph-level bootstrap, partial
  Mantel/MRM, Friedman/CD, multiplicity).

---

## 0. Author decisions taken

| # | Decision | Section |
|---|---|---|
| 1 | Re-execute everything with the C++ engine | §2 |
| 2 | Competitors enter the experiments, vendored **as backends in the IsalHG `iso_backends` style** | §4 |
| 3 | **Merge all splits.** GED is symmetric and carries no train/test semantics | §3.2 |
| 4 | **Exact GED for `n <= 12`; approximate GED above it, on larger real datasets** — replaces the controlled-edit cohort | §3.3 |
| 5 | **Decline the sequential-model experiment**; defer to future work, downgrade LM claims to conjecture | §6 |
| 6 | Render an S2G/G2S example figure as in IsalSR / IsalHG | T-09 |
| 7 | Ignore the page budget while drafting | T-15 |
| 8 | gSpan vendored from `github.com/LasseRegin/gSpan` | §4.2 |
| 9 | IsalChem source at `github.com/icai-uma/IsalChem`; paper unavailable | T-07 |
| 10 | **[28] is and will remain arXiv-only** | §5.3 |

---

## 1. What the data audit found (2026-08-11, measured)

### F1 — The current benchmarks contain no large graphs. The `n <= 12` filter is nearly vacuous.

| Dataset | raw N | **max n** | median n | connected | survives `n<=12` |
|---|---|---|---|---|---|
| IAM Letter LOW | 2,250 | **8** | 5 | 1,180 | **2,250 (100%)** |
| IAM Letter MED | 2,250 | **9** | 5 | 1,253 | **2,250 (100%)** |
| IAM Letter HIGH | 2,250 | **9** | 5 | 2,059 | **2,250 (100%)** |
| LINUX | 89 | **10** | 9 | 89 (100%) | **89 (100%)** |
| AIDS | 911 | **20** | 11 | 819 | 769 |

`n_max: 12` removes **zero** graphs from IAM and LINUX. What removes IAM graphs is the
**connectivity** requirement (2,250 → 1,180 on LOW, 47.6% lost) — stated in the manuscript, never
quantified. This is why §3.3 must bring in new datasets rather than unfilter the old ones.

### F2 — GraphEdX ships GED only *within* train/val/test splits.

| Dataset | splits | within-split pairs | `n_valid_ged_pairs` | all pairs | coverage |
|---|---|---|---|---|---|
| LINUX | 53 / 17 / 19 | 1,378 + 136 + 171 = **1,685** | **1,685** | 3,916 | 43.0% |
| AIDS | 546 / 182 / 183 | 148,785 + 16,471 + 16,653 = **181,909** | **181,909** | 414,505 | 43.9% |

Exact match on both. **This corrects `verified-discrepancies.md` E2**, which attributes the LINUX
3,916 → 1,685 drop to the `GED > 0` / `Lev > 0` filter. It is missing ground truth, not filtering.
The published LINUX ρ = 0.433 and AIDS ρ = 0.349 are within-split figures, undisclosed.

### F3 — Encoding cost has collapsed. GED is the only cost centre.

Real benchmark graphs, `engine() == 'cpp'`, single thread, `process_time`:

| n | exact GED / pair (nx A*) | pruned-canonical encode / graph | ratio |
|---:|---:|---:|---:|
| 5 | 4.0 ms | 6 µs | 6.7 × 10² |
| 9 | 336 ms | 16 µs | 2.1 × 10⁴ |
| 11 | 7.48 s | 21 µs | 3.6 × 10⁵ |
| 12 | **36.9 s** | 27 µs | 1.4 × 10⁶ |
| 20 | *(intractable)* | **122 µs** | — |

Exact GED grows ≈ **5× per added node** near n = 12; encoding ≈ 1.15× per node.
`codebase-pointers.md`'s "the canonical encoder is the bottleneck" was true under pure Python and
is **no longer true**.

---

## 2. Re-run cost with the C++ engine

| Stage | Core-hours | On 64 cores |
|---|---|---|
| All encoding, 4 algorithms, 5 datasets | < 0.01 | seconds |
| Levenshtein, 3.9 M pairs | 1–2 | ~2 min |
| WL kernel (not accelerated) | 2–4 | ~5 min |
| IAM exact GED from scratch, n ≤ 9 | ~13 | ~12 min |
| Graph-level bootstrap + Mantel | 4–8 | ~10 min |
| **Reproduce the submitted study** | **~20–27** | **< 1 h** |

The `2-00:00:00` limit at `config.yaml:42` was sized for pure Python and is now a ~50×
over-provision. **All new compute is GED.**

---

## 3. Data and GED strategy

### 3.1 Principle

Two regimes, reported separately and never mixed:

| Regime | Reference | Role |
|---|---|---|
| **n ≤ 12** | **exact GED**, computed by us, one cost model | ground truth + **calibration anchor** |
| **n > 12** | **approximate GED** (Riesen–Bunke bipartite), calibrated on the regime above | size-scaling evidence |

### 3.2 Splits are merged (decision 3)

Each dataset becomes a single pool. GED is a symmetric function of two graphs with no train/test
semantics; the correlation study is a global measurement. Consequently:

**Recompute *all* GED ourselves, for every dataset, under one stated cost model.** Do not mix our
values with GraphEdX's.

- retires **R3.5b** outright — the heterogeneous-cost objection disappears rather than being
  caveated;
- retires **F2** — no split-shaped holes;
- gives LINUX 3,916 pairs (from 1,685, **2.3×**) and AIDS 295,296 (from 131,148, **2.25×**), on the
  two datasets where ρ is weakest;
- **keeps GraphEdX as a validation arm**: recompute ~500 within-split AIDS pairs and assert exact
  agreement. If they disagree, the cost models differ and everything downstream is suspect. **This
  check runs first and gates T-03.**

| GED job | Pairs | Core-hours | On 64 cores |
|---|---|---|---|
| IAM LOW + MED + HIGH (n ≤ 9) | ~3.6 M | ~13 | 12 min |
| LINUX all-pairs (n ≤ 10) | 3,916 | ~2.4 | 3 min |
| AIDS all-pairs (n ≤ 12) | 295,296 | **1,000–1,600** | **16–26 h** |
| **Total** | | **~1,020–1,620** | **~17–26 h** |

One `cpu` job, 64–128 cores, `1-00:00:00`, checkpointing (`ged_computer.py` already does).
Write it with the **`picasso-sbatch`** skill.

### 3.3 Approximate GED above n = 12 (decision 4) — and the calibration that makes it rigorous

**I agree this beats the controlled-edit cohort**: constructed pairs live in a `k`-ball around a
base graph and are not distributed like real pairs, which is exactly the objection R3 would raise.
Real datasets with a calibrated approximation is the stronger design.

**But it is confounded unless calibrated, and the confound is fatal if missed.** Correlating
Levenshtein against an *approximate* GED at large n mixes two effects: how well Levenshtein tracks
true GED, and how well the approximation tracks true GED. Bipartite GED's error is known to grow
with graph size, so a declining ρ at large n would be uninterpretable. Mandatory protocol:

1. **On n ≤ 12, where exact GED exists**, report all three:
   ρ(Lev, GED_exact), ρ(Lev, GED_approx), ρ(GED_approx, GED_exact),
   plus the mean relative overestimate of the approximation.
2. **State the calibration in the paper.** If ρ(GED_approx, GED_exact) is high and
   ρ(Lev, GED_approx) ≈ ρ(Lev, GED_exact) on the same pairs, the approximation is a validated
   stand-in and the extension to n > 12 is defensible. If not, we report the exact-GED result and
   say the extension is not supportable — that is a legitimate outcome.
3. **Above n = 12**, report ρ(Lev, GED_approx) with the calibration quoted alongside every number.
4. **Bracket it.** Riesen–Bunke bipartite is a proven **upper bound**. Pair it with a lower bound
   (degree-sequence / edge-count / bipartite lower bound). Where **upper = lower, GED is exact** —
   this yields free certified values even at large n, and the fraction of such pairs is itself a
   reportable quantity.

**Implementation: write Riesen–Bunke BP ourselves, do not vendor.** It is ~150 lines — build the
`(n+m)×(n+m)` cost matrix with substitution / deletion / insertion blocks, solve with
`scipy.optimize.linear_sum_assignment`, evaluate the induced edit path. Citable directly, no install
risk. GMatch4py is unmaintained Cython and graphkit-learn needs GEDLIB compiled; both are worse
trades than a day of our own code. Add `nx.optimize_graph_edit_distance` as a free second upper
bound and, optionally, the already-cited **GraphEdX** neural predictions as a third comparator.

References to cite: Riesen & Bunke, *Image and Vision Computing* 27(7), 2009 (BP-GED — the standard);
Fischer et al., ***Pattern Recognition*** 48(2), 2015 (Hausdorff GED — venue fit for EiC.b);
Blumenthal & Gamper, *Pattern Recognition Letters*, 2020 (exact GED); Jain et al., NeurIPS 2024
(already cited).

### 3.4 New datasets — where I diverge from the suggestion

**`cs.cornell.edu/~arb/data/` is Austin Benson's higher-order repository** — hypergraphs, simplicial
complexes, temporal networks. That is **IsalHG's** domain, not IsalGraph's, and a simple-graph paper
drawing from it would look like a dataset grab.

**The natural extension is the IAM Graph Database itself** (Riesen & Bunke, SSPR 2008) — we already
use IAM Letter, it is *the* pattern-recognition GED benchmark, and it ships published edit costs.
Proposed cohort, spanning n̄ ≈ 5 → 40:

| Dataset | Source | approx. n̄ | Why |
|---|---|---|---|
| **GREC** | IAM | ~11.5 | same family, symbol recognition, standard GED benchmark |
| **Mutagenicity** | IAM | ~30 | molecular, sparse, 3× the current ceiling |
| **Protein** | IAM | ~33 | different domain, moderate density |
| **MUTAG** | TUDataset | ~17.9 | the most-cited small molecular benchmark |
| **IMDB-BINARY** | TUDataset | ~19.8 | **dense social graphs** — stress-tests the density claim |
| **PROTEINS** | TUDataset | ~39.1 | graph-learning community's standard |

**Must be verified before committing (T-01b)** — these are literature figures, not measurements:

- **connectivity retention** — G2S requires connected input; PROTEINS and NCI1 carry disconnected
  instances. Report the retention rate per dataset, as we now must for IAM (F1).
- **density and encoding cost** — the `n^{4.9}` fit came from ER `p = 0.35`. IMDB/COLLAB are dense;
  encoding cost there must be **measured**, not extrapolated. If COLLAB (n̄ ≈ 74, dense) is
  expensive even in C++, that is itself a finding and belongs in the limitations.
- **pair subsampling** — BP-GED is `O(n³)` per pair; cheap at n = 40, but all-pairs over 1,000+
  graphs is millions of pairs. Subsample with a stated, seeded protocol.

This cohort answers **AE.1** with real data at real sizes, and it lets us keep the three separate
statements of §3.5.

### 3.5 What we say about size

1. **IsalGraph encoding has no ~12-node ceiling** — measured, 122 µs at n = 20 on real graphs;
   extended to n̄ ≈ 40 by the new cohort and to n = 100+ synthetically.
2. **Exact GED does** — measured, 36.9 s/pair at n = 12, ×5 per node. No public benchmark supplies
   exact GED beyond this; GraphEdX stops there for the same reason. A constraint on the field.
3. **Above n = 12 the reference is an approximation**, with its agreement to exact GED calibrated
   on the regime below and quoted alongside every number.

---

## 4. Competitors — architecture and placement

### 4.1 Vendoring: follow IsalHG's `iso_backends`

`IsalHG/src/isalhg/iso_backends/` is the model: an ABC (`base.py`), a **lazy registry** keyed by
name (`registry.py`, with `_LAZY_MODULES` so optional deps import only on request), a
`subprocess_base.py` for external binaries, and `BackendUnavailableError` on failure. This is the
same idiom as IsalGraph's existing `core/backends.py` (`BackendError`, never degrade silently).

**Proposed: `src/isalgraph/competitors/`** with **two** protocols, because IsalHG's `IsoBackend`
answers a different question (fingerprint / are-isomorphic) than we need here:

| Protocol | Methods | Implementations |
|---|---|---|
| `ReprBackend` | `encode(G) -> str\|bytes`, `bit_length(G) -> int`, `distance(a, b) -> float` | graph6, sparse6, nauty-canonical graph6, AGM code, **gSpan min-DFS code**, IsalGraph |
| `GEDBackend` | `ged(G, H) -> float`, `kind: 'exact'\|'upper'\|'lower'` | networkx A* (exact), **Riesen–Bunke BP** (upper), degree/edge lower bound, GraphEdX (reference) |

Reuse IsalHG's `IsoBackend` shape verbatim for **nauty / bliss / Traces** — we need canonical
relabelling anyway, to make the graph6 comparison *fair* rather than a strawman.

### 4.2 Competitor set

| Competitor | Reversible | Canonical | String | Distance | Effort |
|---|---|---|---|---|---|
| graph6 / sparse6 | yes | only if relabelled | yes | Hamming | hours (`nx.to_graph6_bytes`) |
| **nauty** canonical labelling | yes | **yes** | via graph6 | Hamming | 1 d (`pynauty`, IsalHG has the pattern) |
| bliss / Traces | yes | yes | via graph6 | Hamming | 0.5 d each, reuse IsalHG |
| adjacency matrix | yes | no | no | Hamming | trivial |
| AGM canonical code | yes | yes | yes | Levenshtein | 1 d, derive from nauty labelling |
| **gSpan minimum DFS code** | yes | **yes** | **yes** | **Levenshtein** | **2–3 d**, vendor `LasseRegin/gSpan` |
| WL subtree kernel | no | — | no | kernel | already computed |

**gSpan's minimum DFS code remains the single most important comparator**: canonical, a string,
edit-distance-comparable, named by R1, same problem setting.

**Risk on the vendored gSpan**: `LasseRegin/gSpan` is a *frequent-subgraph miner*. We need the
**minimum DFS code of one graph**, which is an internal sub-component and may not be exposed. If it
is not, extract or reimplement it — budget the same 2–3 days and verify on day 1 of T-04.

### 4.3 Where each experiment gains columns

| Experiment | Gains | Retires |
|---|---|---|
| **(a) Message length**, §3.2.3 / Table 2 / Fig 1 | bit cost for graph6, nauty-graph6, adjacency, AGM, min-DFS | **R3.6a** — we stop calling our own model "standard" and put real serializations beside it |
| **(b) GED proxy**, §3.2.5 / Table 3 / Fig 3 | ρ for Levenshtein-on-min-DFS, Hamming-on-nauty-graph6, WL | **R1.1** (proxy half) |
| **(c) Runtime**, §4.2 / Fig 2 | encode-time curves for min-DFS and nauty | **R1.1** + **D16** — the per-graph/per-pair category error |
| **(d) [28] / [29] delta** | **conceptual table only — no experiment** | R3.1 / AE.3 / R3.7b |

Building an experiment for (d) would be a category error: it asks what we borrowed from our own
prior work, which is answered by reading the sources.

*Stated in advance*: Hamming on non-canonical graph6 should correlate **poorly**, because bit
position is not edit-aligned. That is an informative result isolating why canonical **and**
edit-distance-compatible is the contribution. Report it either way.

---

## 5. Prior-work and bibliography

### 5.1 [29] IsalChem — code only

Source at `github.com/icai-uma/IsalChem`; the paper is unavailable. The code is sufficient for the
**architecture** rows of the inherited/modified/new table (CDLL, two-pointer VM, alphabet,
incremental construction, normalisation). It is **not** sufficient for R3's claim that [29] contains
an LSTM experiment — but if the repo carries an LSTM training script, that half of
`verified-discrepancies.md` **D19** is confirmed. Check on day 1 of T-07.

### 5.2 [28] — the preprint

PDF in-repo at `docs/references/2512_10429v2.pdf`. Read it to confirm the Transformer-classification
claim (D19's other half) and to write the delta table.

### 5.3 [28] is permanently arXiv-only (decision 10)

EiC.a's "substitute arXiv citations with their peer-reviewed versions" **cannot be satisfied** for
[28]. Response: state it plainly in one sentence. Then reduce the *visible* arXiv footprint —
**strip the `note = {arXiv:...}` fields from the five entries that already name ICLR / NeurIPS
venues** (`kipf2017gcn`, `velickovic2018gat`, `xu2019powerful`, `fey2019pyg`, `jain2024graphedx`).
That takes the rendered arXiv count from **6 to 1**.

### 5.4 Bibliography budget

43 cited → 55 ceiling = **12 slots**.

| Purpose | Slots |
|---|---|
| AGM, gSpan, nauty/Traces, bliss, graph6, Babai | 5–6 |
| GED approximation: Riesen–Bunke 2009, Fischer 2015 (*Pattern Recognition*), Blumenthal 2020 | 3 |
| New datasets: IAM Graph Database, TUDataset | 2 |
| Recent (2025–26) pattern-recognition work — **weakest current position: nothing third-party after 2024** | 2–3 |

Over budget by ~2; drop the weakest additions or retire a dead citation. Also fix the one
uncommented group `\cite{garey1979,Zeng:2009}` (`methodology.tex:803`). The four-way group at
`introduction.tex:31` is already individually commented — **do not "fix" it**.

---

## 6. Sequential model — declined (decision 5)

R3.2's modal is the softest in the report ("*would substantially strengthen*", against "should
provide" / "should be narrowed" / "should be described"), so it is framed as an enhancement. A
credible sequence-model study is a paper, not a subsection.

**The decline is only defensible if the claims come down with it.** Non-negotiable:

| Location | Required change |
|---|---|
| `main.tex:122–126` (abstract) | "language-model-compatible … **with direct applications in** graph similarity search, graph generation, graph-conditioned LM" → format compatibility as a **property**; applications as conjecture ("may enable") or dropped |
| `introduction.tex:35–37` | keep as motivation, explicitly not a result |
| `conclusion.tex:76` | already hedged ("can be consumed", "may enable") — leave |
| `conclusion.tex:88–95` | **expand**: name the Transformer/LSTM study as the designated next step, citing [28] and [29] as templates |
| Limitations | **add R3.7a in substance**: no sequential model and no downstream pattern-recognition task is evaluated |

**Residual risk**: R3 may hold the line in round 2. Mitigation is §4 — we chose the comparison the
Area Editor endorsed (AE.3) over the experiment one reviewer suggested, and the letter should frame
it as exactly that exchange.

---

## 7. Ticket board

| ID | Ticket | Depends | Days | Pri |
|---|---|---|---|---|
| **T-01** | **Data lock**: size/density/connectivity audit tables; drop the vacuous `n_max`; merge splits; define cohorts | — | 1–2 | **P0** |
| **T-01b** | **New-dataset audit**: fetch GREC, Mutagenicity, Protein, MUTAG, IMDB-B, PROTEINS; measure n, density, **connectivity retention**, encoding cost | T-01 | 2–4 | **P0** |
| **T-02** | **Statistics lock**: §8; graph-level bootstrap; Mantel; pair-accounting ladder | T-01 | 2–4 | **P0** |
| **T-03** | **Exact-GED job on Picasso** — full spec in §7.1 | T-01 | 3–8 | **P0 — long pole** |
| **T-04** | **Competitor backends**: `src/isalgraph/competitors/` in the IsalHG idiom; graph6, nauty, bliss/Traces, AGM, **gSpan min-DFS** | — | 3–8 | **P0** |
| **T-05** | **Approximate GED**: implement Riesen–Bunke BP + lower bound; **calibration arm at n ≤ 12**; apply to the new cohort | T-01b, T-03 | 5–10 | **P0** |
| **T-06** | **Full recompute**: all experiments, C++ engine, new cohorts, competitor columns, new statistics | T-02..T-05 | 10–14 | **P0** |
| **T-07** | **Read [28] PDF + [29] source**; inherited/modified/new table; resolve D19 | — | 1–4 | **P0** |
| **T-08** | **Related work section** + bibliography to ≤55 (§5.4) | T-07 | 4–10 | P1 |
| **T-09** | **S2G/G2S example figure** via `isalgraph.viz`, matching IsalSR / IsalHG | — | 1 | P1 |
| **T-10** | **Canonical search-space schematic** — renderer exists (`viz/search_tree.py`) | — | 0.5 | P1 |
| **T-11** | **Manuscript errors** (§9) | — | 2 | P1 |
| **T-12** | **Claim scoping** (§10) | T-06 | 2 | P1 |
| **T-13** | **Complexity section**: `P(M)` recomputation, four costed operations, three-way separation | — | 2 | P1 |
| **T-14** | **Response letter** | all | 3 | **P0** |
| **T-15** | **Page trim to 35** + supplementary + AI declaration | all | 2 | **P0** |

| **T-16** | **`wl_pruned_canonical` C++ variant** — 1-WL colour refinement replacing the structural triplet as the candidate key. Spec in §7.2 | — | 3–4 | P1 |

**Critical path**: T-01 → T-01b → T-03/T-05 → T-06 → T-14. T-04, T-07 and T-16 run in parallel off it.

### 7.1 T-03 — recover **all** exact GED (author decision, 2026-08-11)

**Scope**: every pair of every connected graph in the five original datasets. No subsampling, no
split structure, no reliance on GraphEdX's within-split coverage.

| Dataset | connected | **all pairs** | ~s/pair | core-hours |
|---|---:|---:|---:|---:|
| Letter LOW | 1,180 | 695,610 | 0.004 | 0.8 |
| Letter MED | 1,253 | 784,378 | 0.004 | 0.9 |
| Letter HIGH | 2,059 | 2,118,711 | 0.008 | 4.7 |
| LINUX | 89 | 3,916 | 2.17 | 2.4 |
| **AIDS (GraphEdX)** | 769 | **295,296** | 12–20 | **985–1,640** |
| **Total** | | **3.90 M** | | **≈ 1,000–1,650** |

**16–26 h on 64 cores.** Counts are pre-reconciliation (open question 3).

**Configuration** — fixed:
- cost model: **unit node + unit edge, substitutions free** (`statistics.md` D6)
- **GED timeout: unchanged from the submission** (author decision). Record it explicitly and report
  the censoring rate **per stratum** — censoring is symmetry-correlated, never pool it
- non-computable pairs are **interval-censored `[LB, UB]`**, not dropped (`statistics.md` D11)
- checkpoint every 5,000 pairs (`ged_computer.py` already does)
- write the SLURM script with the **`picasso-sbatch`** skill; `cpu` constraint, 64–128 cores,
  `1-00:00:00`, 128 GB

**Gate — run this first, before the main job**: recompute ~500 *within-split* AIDS pairs under
GraphEdX's own topology-only cost model and assert exact agreement with the published matrix. If
they disagree, our solver or our cost model is wrong and everything downstream is invalid. Only
after this passes does the unit-cost production run start.

**Expected consequence**: LINUX ρ = 0.433 and AIDS ρ = 0.349 will both change — the pair sets grow
2.3× and 2.25×, and the cost model changes. Every downstream number must be re-derived.

### 7.2 T-16 — the `wl_pruned_canonical` variant

**Motivation, measured** (`data.md` §4.4): canonicalisation cost is governed by \|Aut(G)\|, and the
incumbent structural triplet `(|N₁|,|N₂|,|N₃|)` partitions the hard Mutagenicity graph into only
**28 classes of 98 nodes**, where 1-WL reaches **66**. WL is **2.4× finer** on exactly the
instances that fail.

**Design** — mirrors `IsalSR/src/isalsr/core/native/src/{wl.cpp, canonical.cpp}` and
`IsalHG/.../greedy_min_wl_pruned.py`:

- 1-WL colour refinement to stability; FNV-1a 64-bit hashing, byte-stable, `PYTHONHASHSEED`-independent
- candidate key becomes the WL colour; **keep the entire argmin-colour class**, never a representative
- **no raw node id is read** — this is the admissibility argument, verbatim from
  `greedy_min_wl_pruned.py` lines 10–15
- C++ only, in `src/isalgraph/core/native/`, with a Python reference for the differential suite
- **cyclic-graph WL, not IsalSR's DAG subtree hash** — IsalSR's `compute_subtree_hashes` is a
  bottom-up BFS over a topological order and does not transfer to graphs with cycles

**Acceptance criteria**:
1. byte-identical canonical strings to the incumbent on all 3,079 differential graphs — it is a
   *pruning*, so the output must not change;
2. measurable reduction in the timeout rate on Mutagenicity and Protein;
3. the residual timeout rate reported per symmetry stratum — **WL will not rescue large-\|Aut\|
   graphs and the paper must say so.**

**Do not attempt automorphism pruning.** Individualisation-refinement with automorphism detection is
what nauty/bliss/Traces do and is the actual fix; re-implementing it is a project, not a revision.
State it as future work and cite nauty, which is already being vendored as a competitor (§4.2).

---

## 8. Statistics and stratification lock

| Item | Current | Locked | Driver |
|---|---|---|---|
| Correlation | Spearman ρ, asymptotic p, pair-level | ρ + **graph-level bootstrap CI** + **Mantel permutation test** | R3.5c |
| Bootstrap unit | pairs | **graphs** — resample graphs, recompute over induced pairs | R3.5c |
| Reported effect | pooled OLS β as headline | **per-dataset primary**; pooled demoted | R3.5b |
| Pair accounting | one unreconciled number | **raw → connected → GED-available → GED>0 → Lev>0 → analysed**, per dataset | R3.5a, E2, F2 |
| Exclusions | unjustified | justified per stage; `Lev=0 & GED>0` counted and **reported explicitly** | R3.5a |
| Stratification | none | by **node count** and **true density**, within and across datasets | AE.1, R1.3, E1 |
| Dataset properties | graphs, pairs, `m̄` | **+ `n̄`, + density, + connectivity retention** | E1, F1 |
| GED cost model | mixed across datasets | **one model, recomputed throughout** | R3.5b |
| GED reference | exact only, undisclosed gaps | **exact ≤ 12 / calibrated approximate > 12** | AE.1, R3.7 |

### The AIDS question, settled with data

R1.3 attributes the AIDS degradation to label loss. The rebuttal stands: **the GraphEdX GED is
itself topology-only**, so both sides of the correlation are label-blind and a label-loss mechanism
cannot explain that number. But we now also test the authors' *own* density claim:

- report true density per dataset (**currently uncomputable from the paper** — E1);
- **stratify AIDS pairs by density and report ρ within strata**, on 295,296 pairs instead of 131,148.

**This can refute `conclusion.tex:30–36`.** If ρ does not recover on sparse AIDS strata, the density
explanation is wrong and must be rewritten. Run it early.

---

## 9. Manuscript errors (all accepted)

| ID | Defect | Fix |
|---|---|---|
| R3.4a / D5 | Alg. 2 `C`/`c` guards **and** duplicate checks reversed vs Table 1 | rewrite `methodology.tex:321–336` to match `graph_to_string.py:208–238` |
| R3.4c / D1 | `n^{9.0}` at `conclusion.tex:50` has no source; `:50` vs `:68` disagree | all exponents re-derived in T-06 |
| R3.4c / D2 | `n^{4.9}` called "super-polynomial" | three-way separation, T-13 |
| R3.7e / D20 | "breaks permutation equivariance" | → **invariance**. `M → P M Pᵀ` *is* equivariance |
| E1 | density never computed; no node count reported | T-01 |
| E2 / **F2** | 473,147-pair gap; LINUX 3,916 vs 1,685 | **cause: within-split GED coverage.** Fixed by T-03 |
| E3 | fits declared `n = 3–20`, greedy data to 50 | re-derived, T-06 |
| E4 | a fourth node range (`n = 3–11`) | cross-referenced |
| E5 | abstract self-contradiction (`:106` vs `:114`) | §10 B1 |
| E6 | "labels present in all five datasets" — **false for LINUX** | corrected |
| E7 | algorithms float to pp. 33–35 | T-15 |
| E8 | draft self-correction printed in Example 2.3 | delete; `[0,2,1]` is right |
| E11 | generative-AI declaration commented out | restore — Elsevier compliance |

---

## 10. Claim scoping (all accepted)

- **B1** — scope G2S: undirected **connected**; directed **root reaching all nodes**. State the
  asymmetry: S2G total, G2S partial. `main.tex:106–108`, `introduction.tex:33`, `:45–46`,
  `conclusion.tex:74`.
- **B2** — directedness: flag is **external metadata**; **restate Theorem 2.12 within a fixed
  directedness class**; move the "deterministic given `w` and the flag" hypothesis from the proof
  into the statement. Use the **exact witness** (one undirected edge and one directed arc both
  canonicalise to `"V"`); never quote a collision rate without its enumeration window.
- **B3** — "GED **standard** construction" → "explicit-construction reference model"; §4.3(a)
  supplies the real baselines.
- **B4** — propagate the results section's conditional framing to abstract and conclusion. Numbers
  re-derived in T-06.
- **B5** — limitations: `n` ceiling **with its cause** (§3.5); exponential worst case; no sequential
  or downstream task (§6).
- **B6** — unify the four-properties claim; attach it to the §4 table; soften "no existing method".

---

## 11. Open questions

Consolidated. Data-side detail in `data.md` §9, statistics-side in `statistics.md` §10.

### Resolved by author decision or measurement

| # | Item | Resolution |
|---|---|---|
| 1 | Exact-GED scope | **All-pairs, recover everything** (author, 2026-08-11). ~1,000–1,650 core-h, 16–26 h on 64 cores. Applies to the **five original datasets only** — exact GED is unobtainable on the extension cohort |
| 2 | One cost model | **Unit node + unit edge** (`statistics.md` D6). Published GraphEdX values will no longer match ours; stated in the text |
| 4 | Primary large-n reference | **BRANCH-FAST**, ρ(exact, LB) = 0.966 vs ρ(exact, UB) = 0.840 |
| 5 | Refine the upper bound | **GEDLIB supplies IPFP / REFINE / BRANCH_TIGHT** — use them instead of our plain BP |
| 6 | Calibration gate | Rule fixed in `statistics.md` §6 |
| 8 | MRM / partial Mantel | **Confirmatory** (`statistics.md` D4). Run in week 1 |
| 9 | Kendall τ-b | Spearman primary, τ-b as robustness check (`statistics.md` D1) |
| 10 | Exhaustive canonical > n = 12 | Measured — fails on 55 % of Protein graphs; **report the pruned/exhaustive gap as a result** |
| 11 | Bounds implementation | **GEDLIB** (recognised, builds on Picasso — `data.md` §7.5); our `ged_bounds.py` retained as a cross-check |

### Still open

| # | Question | Recommendation |
|---|---|---|
| ~~3~~ | ~~Reconcile Letter counts~~ — **closed**. The pipeline filter is `min_nodes = 2`, `require_connected = True`; applying it reproduces the manuscript **exactly** (1,180 / 1,253 / 2,059 / 89 / 769, 3,897,911 pairs, m̄ to two decimals). See `data.md` §0 | — |
| ~~7~~ | ~~Cohort / GREC~~ — **closed**. Add Mutagenicity, Protein, COIL-DEL, AIDS-IAM **and GREC**; drop COIL-RAG, Fingerprint, Web. GREC's 59.1 % retention is misleading: its discard is **size-unbiased** (11.59 vs 11.45 nodes), the cleanest in the cohort (`data.md` §2.2.1) | — |
| **15** | **The connectivity discard is size-biased on the datasets we added for scaling** — Mutagenicity discards graphs 1.9× larger than it keeps, AIDS-IAM 2.3× (`data.md` §2.2.1). Any "n̄ ≈ 30" claim is on a subsample with the large graphs preferentially removed | report retained **and** discarded `n̄`/`n_max`; state the precondition as a scope limitation with its measured cost |
| **16** | **Benchmark GEDLIB `ANCHOR_AWARE_GED` (exact) against `networkx` A*** — it may push the exact-GED ceiling above n = 12 and enlarge the calibration regime | add as a T-03 pre-step |
| **12** | Which comparisons are **confirmatory** vs exploratory? (`statistics.md` S1) | confirmatory = IsalGraph-pruned vs each competitor, per dataset, on both claims |
| **13** | **G2S timeout**: keep at a generous value (300 s) rather than removing it — removal **hangs on real graphs** (`data.md` §4.3) | keep; record per-graph time; report the rate per stratum |
| **14** | **Symmetry stratification** (orbit count via nauty) — new, unasked-for, and it explains the cost better than n or density | adopt |

---

## 12. Change log

| Date | Ver | Change |
|---|---|---|
| 2026-08-11 | v0.1 | Initial grouping and disposition |
| 2026-08-11 | v0.2 | Data audit F1–F3. Ordered recompute. Sequential model declined. Competitors into three experiments. E2's cause corrected |
| 2026-08-11 | v0.3 | Splits merged; all GED recomputed under one cost model. Controlled-edit cohort **dropped** in favour of exact ≤12 + calibrated approximate >12. Competitor backends in the IsalHG `iso_backends` idiom. New-dataset cohort proposed (IAM family + TUDataset, not Benson). [28] permanently arXiv-only |
| 2026-08-11 | v0.4 | **`data.md`** and **`statistics.md`** added, all figures measured. Cohort resolved to the IAM Graph Database alone (TUDataset unnecessary — Mutagenicity reaches n = 417). Key measured results: pruned canonical encodes n̄ = 32 in 3.9 ms with no timeout to n = 96; exact GED is 36.9 s/pair at n = 12; the whole approximate-GED extension costs 1.24 core-hours; **BRANCH-FAST lower bound tracks exact GED better than the BP upper bound** (ρ 0.966 vs 0.840). Open questions restructured by decision deadline |
