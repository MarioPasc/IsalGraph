# T-04 — article notes

**Closed 2026-08-15.** Ordered by consequence: what changes the paper's claims first, reporting
obligations after, then what is **not** claimable.

Every number below names its provenance. `MEASURED` = produced by this ticket from `src/`.
`REPRODUCED` = this ticket re-derived a scout number and it held. `SUPERSEDED` = a plan-file number
this ticket replaced.

**Reproduction parameters, and they travel with every number here**: `isalgraph-cpp`, Python
3.11.15, `networkx` 3.6.1, `numpy` 1.26.4, `rapidfuzz` 3.14.5, `pynauty` 2.8.8.1, `grakel` 0.1.10
(its `__version__` string reads `0.1.8` and is stale), `scipy` 1.17.1, engine `cpp`
(`build_hash 298fc1188bf1b051`). Seed **42** throughout. GED reference: T-03's certified exact
matrices under the D6 unit cost model `[1,1,0,1,1,0]`, certified-exact pairs only. Budgets: AGM
200,000 search nodes (Suite 1) / 100,000 (Suite 2), min-DFS `max_projections = 50,000`, IsalGraph
2.0 s wall clock. Picasso figures: login node `picasso3`, gcc 12.2.0, **pure-Python engine**.

---

## 1. The size-null claim is not robust to the draw — **T-20**, results §res-correlation

**This is the item that changes what the paper may assert, and it is negative.**

`competitors/README` §4.1 says IsalGraph clears the size null on **two** of five Suite-1 datasets.
On a single-draw recomputation under the frozen conventions it clears **one**:

| Dataset | null ρ | IsalGraph pruned ρ | margin | §4.1 said |
|---|---:|---:|---:|---|
| Letter LOW | 0.8991 | **0.9253** | **+0.026** | +0.026 ✔ |
| Letter MED | 0.9190 | 0.8750 | **−0.044** | +0.007 ✘ |
| Letter HIGH | 0.9174 | 0.6969 | −0.221 | −0.243 |
| LINUX | 0.7134 | 0.4742 | −0.239 | −0.239 |
| AIDS | 0.7844 | 0.2561 | −0.528 | −0.544 |

`MEASURED`, `corrected_rho_table.json`, all-pairs view.

**The defensible statement is not "one of five" either.** The margin that vanished is `+0.007`, and
finding 14 records ρ moving by up to **0.07** between two independent 200-graph draws of the same
dataset. A `+0.007` margin is an order of magnitude below the noise. **What the paper can say:
IsalGraph clears the trivial size baseline on Letter LOW by 0.026, and on no dataset by a margin
that survives resampling.** Anything stronger requires the CIs.

> **Precondition, now binding: graph-level bootstrap CIs on every printed ρ.** `competitors/README`
> §7 already lists them as open and finding 14 predicts they will be wide. Until they exist, no ρ
> ordering between two representations separated by less than ~0.07 is reportable. Owner: **T-02**
> (`statistics.md` D2), executed by **T-06**.

## 2. The comparison the paper should lead with is equal-`n`, and IsalGraph leads a column there

Restricting to `n₁ = n₂` (22–26 % of pairs) removes the size channel, so the null is constant and
every column is pure structure. `MEASURED`, `corrected_rho_table.json`, equal-`n` view:

| | Letter LOW | Letter MED | Letter HIGH | LINUX | AIDS |
|---|---:|---:|---:|---:|---:|
| adjacency | 0.6067 | 0.5209 | 0.3654 | 0.2362 | 0.1571 |
| graph6 | 0.5389 | 0.4482 | 0.4070 | 0.2862 | 0.1250 |
| sparse6 | 0.5588 | 0.4635 | 0.4377 | 0.2553 | 0.1537 |
| nauty→graph6 | 0.9737 | 0.9520 | 0.6803 | 0.2605 | 0.1375 |
| AGM CAM | 0.9944 | 0.9896 | 0.7597 | 0.6826 | — |
| min-DFS | 0.9956 | 0.9268 | 0.7739 | **0.5404** | 0.3993 |
| WL subtree | 0.9797 | 0.9503 | 0.6964 | 0.6713 | **0.4332** |
| IsalGraph pruned | 0.9806 | 0.9105 | 0.6166 | 0.3972 | 0.2187 |
| **IsalGraph canonical** | **0.9987** | 0.9073 | 0.6953 | 0.3492 | 0.1859 |

- **The canonical / non-canonical split on Letter LOW is 0.54–0.61 against 0.97–1.00 — a gap of
  0.42–0.46**, invisible in the all-pairs view because the size channel floats everyone. **That
  subtraction is R1.2's uniqueness answer** and it is better than anything in the current draft.
- **`isalgraph_canonical` is best in column on Letter LOW (0.9987).** `SUPERSEDED`: §4.2's
  "min-DFS still wins every column" is false on this draw — WL also takes AIDS.
- Subject to §1: on four of five columns the leader's margin is inside the resampling envelope.

## 3. Claim A resolves on `m/n`, and the reference point is stronger than the draft admits — **T-20**

`REPRODUCED` from `src/`, all ten datasets, Suite-1 rows exact (full cohort):

- IsalGraph is **never** the most compact representation on Suite 1; the adjacency matrix wins every
  dataset and IsalGraph is shorter on **0.0 %** of the three Letter sets.
- **Mutagenicity is where IsalGraph wins outright** — 147.4 bits against adjacency 300.0, graph6
  306.0, sparse6 168.0, min-DFS 250.0 and `B_GED` 310.0. Large (`n̄ = 27.9`) *and* sparse
  (`m/n = 1.03`), which is the regime the `m`-scaling argument predicts and the one AE.1 asked the
  paper to reach.
- **`m/n`, not `n`, is the variable.** Protein (`n̄ = 31.9`, `m/n = 1.94`) is larger and IsalGraph
  only ties; COIL-DEL (`n̄ = 21.3`, `m/n = 2.51`) is smaller and it loses badly.

**The claim that survives, for R3.6a**: *IsalGraph is shorter than every other string serialisation
and than the explicit-construction reference model; the raw adjacency matrix is shorter at these
sizes, and the crossover is at `n ≈ 14` and low density.*

**Two corrections to the numbers themselves.** The `sparse6` column of §4.3 is **6 bits high on
every row** — it counted the `':'`, which the frozen convention treats as framing, not payload; use
Letter LOW **18.0**, Protein **384.0**, Mutagenicity **162.0**. And the **five Suite-2 rows are
draw-dependent**: they are medians over a 400-graph sample whose source is gone (§7).

## 4. WL: the convention was ours, and `h = 2` is now justified without touching ρ — **T-06**, **T-20**

`MEASURED`. **grakel has no off-by-one**: `n_iter = k ≡ h = k`, from
`grakel/kernels/weisfeiler_lehman.py:109` and confirmed by arithmetic (`K(G,G) = 62 = 36 + 26` at
`n_iter = 1`). `grakel(n_iter=2) = ours(h=2) = 5.830951894845301`; `grakel(n_iter=3) =
7.211102550927978`. Two independent implementations agree to `1e-9` on all five datasets.

The apparent off-by-one lived in `scratch/backends.py::wl_features`, which compresses colours **per
graph, per round**, making rounds ≥ 2 cross-graph incomparable. Consequences:

- **`h = 2` means `n_iter = 2`.** `benchmarks/real_data/eval_setup/wl_kernel_computer.py` defaults
  to `n_iter = 5`, which is **`h = 5`** — three refinement rounds past the selected one, not two.
- §4.1's WL row moves: Letter LOW **0.895 → 0.7792**, MED **0.869 → 0.7746**, HIGH → 0.5674,
  LINUX → 0.5665, AIDS → **0.4714**.
- **`h = 2` is selected on cost, F5-blind.** `h = 2` vs `h = 5` over 60 graphs / 1,770 pairs:
  dimension grows **4.9× / 17.8× / 24.1×** (Letter LOW / LINUX / AIDS), 11.2 % / 16.6 % / 13.6 % of
  pair orderings flip — and **`frac(d = 0)` is identical at both**, so three extra refinements
  separate **zero** additional pairs. That justifies `h = 2` without any appeal to ρ, which is
  exactly the form `competitors.md` §3.4 requires.
- **Citation gap**: the manuscript cites `weisfeiler1968reduction`; the *kernel* paper —
  Shervashidze, Schweitzer, van Leeuwen, Mehlhorn & Borgwardt, **JMLR 12:2539–2561, 2011** — is a
  different reference and is missing. Owner: **T-14** / bibliography.

## 5. F3 for a non-canonical format counts the complete graphs — **T-20**, the F3 caption

`MEASURED`, and it is a **theorem, not a sample**. Over every connected graph on `n = 2…6`, exactly
**5** are invariant under **every** relabelling, and **all 5 are complete**: the strict upper
triangle is relabelling-invariant iff the adjacency matrix is constant off-diagonal. Exhaustive
enumeration over all `n!` relabellings returns the same counts as the 20-draw harness.

So `adjacency`/`graph6`/`sparse6` scoring `0–9 / 50` is **the number of complete graphs in the
draw** — Letter (many `K₂`/`K₃`) scores 4–9/50, LINUX, AIDS and GREC score **0/50**. Corroborated
independently on Picasso: GREC gives 0/50 for all three. **Say this in the caption**; it converts an
incidental-looking number into a statement about the cohort.

## 6. The pool's internal structure, for AE.3 — **T-17**

`MEASURED`. The four `n²` members emit **the same bit sequence** (strict upper triangle,
**column-wise**), asserted in code across 7 fixtures, 300 random graphs and all eight boundary sizes
including graph6's 4-byte `N(n)` branch. So they share one Claim A number, `n(n−1)/2` — **print one
`n²` row with a footnote, not four identical columns.**

**`agm.md` §2.3's worked example is wrong**: `'E@ro'` unpacks to `000001110011110`, which **equals
AGM's code** on the running example; the printed `001110010011100` is neither string. The
conclusion — nauty cannot supply the AGM labelling — survives on a better artefact: **nauty/AGM
agreement is 38/60, 32/60, 16/60, 12/60, 1/60, 0/60 at `n = 5…10`.**

Ceilings, all `REPRODUCED` and all reportable: AGM **100 %** through LINUX, **99.6 %** on Suite-1
AIDS (3 named failures), **76.00 %** on GREC; min-DFS **24/400** on Mutagenicity and **0** on the
other four Suite-2 datasets; `isalgraph_canonical` unusable on Suite 2.

## 7. Reproducibility obligations — **T-21**

- **Suite 2 is no longer reproducible from source on this workstation.** `real_suite2.py` sampled
  400 graphs from `IAM_Database/extracted`, which is **absent**; the cohort was recovered as
  exported `.npz` from Picasso and enumerates in a different order. Coarse statistics survive
  (adjacency, graph6: 10/10 exact), finer ones do not (min-DFS Protein **620.0 vs 615.0**; AGM
  AIDS-IAM **80.25 % vs 82 %** — a sample difference, since GREC reproduces exactly with the same
  code). **Any Suite-2 number must be quoted with its draw, or requoted from a recorded sample.**
- **grakel cannot run on Picasso**: it is numpy-1 code and that environment carries numpy 2.4.6 for
  T-05. A source rebuild does not help (`ComplexWarning` moved to `numpy.exceptions`). Cluster WL
  numbers must come from our own implementation or a `numpy < 2` environment. Owner: **T-06**.
- **`pynauty` builds from source under gcc 12.2.0 on Picasso** and gives byte-identical output to
  the workstation, inversion guard included. Stop-condition 2 closed; `k` in
  `N_actual = 182 − 15k − 8d` is not at risk from the nauty family.
- **Fig. 2 must be language-matched.** Measured on Picasso with **both arms in Python**, GREC:
  `min_dfs` **1.03 ms/graph** against `isalgraph_pruned` **17.6 ms/graph**. Timing a pure-Python
  competitor against the C++ engine reproduces R1.1's own complaint inside our answer to it; every
  smoke header records `isalgraph_engine` so a timing cannot be quoted without it.

---

## What is NOT claimable

- **Not "IsalGraph clears the size null on two of five datasets."** Superseded (§1). Not "one of
  five" as a stable fact either — the honest form names Letter LOW and the resampling envelope.
- **Not "min-DFS wins every column" (equal-`n`).** False on the single-draw table (§2).
- **Not any ρ ordering with a margin below ~0.07** until the graph-level bootstrap CIs exist (§1).
- **Not §4.1 or §4.2 of `competitors/README` as printed.** Three-draw composites; superseded by
  `corrected_rho_table.json`.
- **Not "grakel's `n_iter` is off by one."** Refuted (§4). And do not quote any WL ρ produced by
  `scratch/backends.py::wl_features` — that implementation is not cross-graph comparable.
- **Not "an inverted `canon_label` passes F3", and not `nx.is_isomorphic` as a guard against it.**
  Both refuted: the inversion fails F3 loudly, and the guard is vacuous because any bijective
  relabelling is isomorphic by construction.
- **Not the `sparse6` Claim A column as printed in §4.3** — 6 bits high on every row.
- **Not the five Suite-2 Claim A rows as exactly reproducible** (§7). The five Suite-1 rows are.
- **Not a Suite-2 Claim-A or Claim-B row for `agm_cam` or `isalgraph_canonical` at all.** They are
  `SUITE1_ONLY`; their ceilings are measurable there and their *columns* are refused, because a
  column built from whichever graphs finished is conditioned on tractability.
- **Not a bit count for `wl_subtree` or `size_null`.** Both raise `BitCountUndefined`. WL's Claim A
  cell is empty with the reason printed, and `VectorBackend` has no `bits()` so fabricating one is
  unreachable.
- **Not the `min_dfs` `realised_bits` figure unlabelled.** It is `8·len(character rendering)`,
  flagged `inflated=True`, and its entropy bound `m·2⌈log₂ n⌉` is a **fixed-width upper bound** — a
  reviewer can say so, and the defence is that the same convention is applied to `B_GED`'s
  `2M⌈log₂ N⌉`.
- **Not a T-04 ρ as a selection input.** F5 is descriptive here. `grid.py` cannot import a GED
  loader and a test asserts it; T-04a selects on F1–F4 with F6 as tiebreak.
