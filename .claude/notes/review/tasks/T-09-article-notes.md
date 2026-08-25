# T-09 article notes — explanatory figures

**Closed 2026-08-25.** Ordered by consequence: items that change what the paper may claim
first, reporting obligations after, then what is *not* claimable.

Artifacts: `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/results/reports/T-09-explanatory-figures/figures/`
— five figures, each as `.pdf` (manuscript) and `.png` (docs site).
Regenerate with `$PY -c "from isalgraph.viz.figures import render_all; render_all(<dir>, formats=('pdf','png'), paper_only=True)"`.

---

## 1. A manuscript statement is wrong, and the figure R3.7c asked for exposes it

**Owner: T-11** (manuscript defects) or **T-20** (§2 rewrite). **Lands: `methodology.tex:462`,
Remark 2.7 (`\label{rem:search-space}`).**

Remark 2.7 is the prose R3.7c points at — the demand index records that "prose already
states the reviewer's exact sentence, only the figure is missing". It reads:

> Only the identity of the uninserted neighbour chosen at each `V`/`v` step contributes to
> the search space.

Definition 2.6, three lines above, defines `w*_G` as "the shortest string encodable by any
execution of GTS **from any starting node**", and `core/canonical.py` searches start nodes
as well as neighbour choices. The schematic therefore draws a **forest with one root per
start node**, which contradicts the word "Only" in the remark it illustrates. A reviewer who
asked for this figure will read the two together.

**Measured on the running example** (`n = 6`, `|Aut(G)| = 1`), greedy `G2S` from each of the
six start nodes:

| start node | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| `\|G2S(G, v)\|` | **9** | 10 | 9 | 11 | 10 | 10 |
| attains `w*_G` | **yes** | no | no | no | no | no |

Six distinct strings. The start node is not a free choice; it is the outer loop of the
search.

**Proposed replacement, one clause, no page cost:**

> Two things are searched over: the starting node, and the identity of the uninserted
> neighbour chosen at each `V`/`v` step. The priority order `V ≻ v ≻ C ≻ c` and the
> minimum-displacement pair ordering `P(M)` (Definition 2.5) are intrinsic to the algorithm
> definition and are *not* branched over. The `C`/`c` steps are fully deterministic given
> the pointer positions and the output graph constructed so far.

**What survives**: the remark's real content — that the pair ordering and the priority
cascade do not branch — is correct and is what R3.7c is about. Only the scope of "Only"
is wrong.

---

## 2. The pruned canonical form is not reachable by any greedy run

**Owner: T-20**, §2.3 (pruned canonicalisation) or the caption of the pruned figure pair.

On the running example the two canonical forms are **different strings of the same length**:

| form | string | length |
|---|---|---|
| exhaustive (`canonical_string`) | `VVVnvPCPV` | 9 |
| pruned (`pruned_canonical_string`) | `VVpvvPVnC` | 9 |

Remark 2.11 (`methodology.tex:614`) already permits this — the two are different canonical
forms, not two spellings of one — so this is confirmation, not a correction.

The new fact is that **`VVpvvPVnC` is emitted by no greedy `G2S` run from any start node**
(all six enumerated; see §1's table for the exhaustive form's). It is reachable only through
the neighbour-choice branch. That is direct, small-scale evidence that the branch in
Definition 2.6 is load-bearing rather than a formality, and it is worth one sentence beside
the search-space schematic.

---

## 3. What the figures are, and what each answers

**Owner: T-20** (placement, captions), **T-15** (page budget), **T-26** (page arithmetic).

| File | Answers | Printed size |
|---|---|---|
| `canonical_search_tree.pdf` | **R3.7c**, requested | 7.0 × 3.4 in |
| `fig_worked_example_s2g_canonical.pdf` | author preference | 7.0 × 2.84 in |
| `fig_worked_example_g2s_canonical.pdf` | author preference | 7.0 × 2.84 in |
| `fig_worked_example_s2g_pruned.pdf` | author preference | 7.0 × 2.84 in |
| `fig_worked_example_g2s_pruned.pdf` | author preference | 7.0 × 2.84 in |

All at IEEE text width, `pdf.fonttype = 42` (TrueType embedded, Elsevier requirement),
`savefig.dpi = 300`.

**One running example serves all five**, so a reader carries one graph through §2:

```
n = 6,  m = 6,  E = {(0,1), (0,2), (0,3), (1,3), (2,4), (3,5)}
degrees 3,2,2,3,1,1        |Aut(G)| = 1
a triangle {0,1,3}, a path 0–2–4, a pendant 3–5
```

It was chosen by enumerating every connected graph on 5–6 nodes with `n−1` to `n+2` edges
against five criteria (design note §2). **Two of them are unsatisfiable together at `n = 5`,
and that is what fixes `n = 6`:** the greedy encoder must attain the canonical string from
some start node (or the two panels cannot share one string), and the graph must be
asymmetric (or the search tree is degenerate). At `n = 5` the only graphs satisfying the
first are `C₅` (`|Aut| = 10`) and `K₂,₃` (`|Aut| = 12`).

### The reading convention, which the caption must state

Both worked-example panels have six columns, indexed by the **encoder's symbol groups**
(`V │ V │ V │ nv │ PC │ PV` for the exhaustive form). No step is omitted from either. S2G
consumes one symbol per step and G2S emits a whole group per pass of its outer loop, so
indexing each by its own unit would give them different column counts and nothing to
compare.

Ghosting is a conservation argument and the caption should say so in one line:

| | instruction strip | graph |
|---|---|---|
| **S2G** | starts solid, drains | starts ghosted, fills |
| **G2S** | starts ghosted, fills | starts solid, drains |

Each panel shows one representation emptying into the other, and the two run in opposite
directions. That is the round trip, visible rather than asserted.

---

## 4. The G2S panel shows the encoder searching, not a replay

**Owner: T-21** (implementation and reproducibility), one sentence; **T-20** if the caption
needs it.

`GraphToString.run_with_trace` does not trace the encoder: it takes the finished string and
**replays** it through `StringToGraph`. A G2S figure built from it is the S2G figure with
its mask flipped — same states, drawn twice.

The panels are therefore built from `isalgraph.viz.encoder_trace`, which re-runs the
encoder's outer loop with the rejected displacement pairs recorded.
`core/graph_to_string.py` is **not modified** — it is the reference the C++ differential
suite compares against — so this is a mirror, and it is pinned to the frozen encoder by
test.

**Validation, measured:** the mirror's emitted string is byte-identical to
`GraphToString.run`'s on **134,609 `(graph, start)` pairs** — exhaustively over every
connected graph on 4–6 nodes within the edge budget, from every start node, plus random
graphs to `n = 14` and directed graphs exercising the `c` branch — with **zero mismatches**.
The suite keeps the cheap half of that sweep (`tests/viz/test_encoder_trace.py`, 12 tests,
9.3 s).

What each G2S column's caption reports, and what no replay could:

| step | pairs rejected | winning displacement | cascade level |
|---|---|---|---|
| 1 | 0 | `(+0,+0)` | `V` |
| 2 | 0 | `(+0,+0)` | `V` |
| 3 | 0 | `(+0,+0)` | `V` |
| 4 | 2 | `(+0,+1)` | `v` |
| 5 | 3 | `(−1,+0)` | `C` |
| 6 | 3 | `(−1,+0)` | `V` |

---

## 5. A rendering defect that affected every figure using the CDLL ring

**Owner: T-21**, only if the reproducibility section lists figure provenance. Otherwise
record-only.

`cdll_view.draw_cdll_ring` drew the π and σ pointer arrows with **only their heads
visible**. The tail offset was expressed in axis units (`radius + 0.35`) and the head shrink
in points (`node_radius * 72`), so the visible body length depended on the rendered figure
size and went negative at column scale. Both endpoints are now computed in data coordinates
with no shrink.

This affects any figure that called `draw_cdll_ring`, which includes the committed
`docs/figures/` set and the benchmark figure scripts that import it. Those have been
regenerated. **Nothing in the submitted PDF is affected** — the submitted
`fig_algorithm_overview.pdf` predates this module.

---

## 6. The graphical abstract was NOT regenerated, and why

**Owner: T-24** (submission package).

The board row asked that both figures "regenerate the graphical abstract". T-09 produced the
panels it would be composed from and stopped there, deliberately:

**`graphical_abtract.pdf` panel (b) prints numbers T-06 retired** — `Wins: 99.6 %`,
`β = 0.537`, `R² = 0.947`, and a `14,108×` speedup. T-06 withdrew Claim B at scale (below its
own size null on 17 of 25 records) and moved the ρ figures. Regenerating panel (a) while
leaving panel (b) would produce an artifact that is half-current and wholly unusable, and
would make the stale half look freshly checked.

**T-24 owns this**, together with the misspelt filename (`abtract`). What T-09 hands over:
the two panels for (a), and the fact that (b) needs T-06's numbers before it can be redrawn.

---

## 7. Reproduction parameters

| Parameter | Value |
|---|---|
| Environment | `~/.conda/envs/isalgraph-cpp`, Python 3.11.15 |
| Engine | `isalgraph.engine() == "cpp"`, build hash `298fc1188bf1` |
| matplotlib | 3.11.1, `Agg` |
| Fonts in the PDFs | **DejaVu Serif / DejaVu Sans Mono**, CID TrueType, embedded and subset; no Type 3, so the Elsevier requirement is met. `BASE_RCPARAMS` asks for Times New Roman first and falls back because Times is not installed here — **a machine with Times renders these figures differently from the same code**. Every figure in the revision is generated on this workstation, so the set is internally consistent; **T-21** should state the font stack if it lists figure provenance |
| Randomness | **none.** Explicit edge list, explicit start node, explicit pinned node coordinates. No seed, because no draw |
| Layout | `worked_example.RUNNING_EXAMPLE_POSITIONS`, pinned; NetworkX is not required |
| rcParams | `style.BASE_RCPARAMS` via `apply_ieee_style()` |
| Suite at close | **2,583 passed / 321 skipped**, 9 min 18 s |

---

## 8. What is NOT claimable from this ticket

- **Not that the pruned and exhaustive canonical forms always differ.** T-09 measured one
  graph. Remark 2.11 states the general position; T-09 provides an instance, not an
  incidence rate.
- **Not that greedy `G2S` usually fails to attain `w*_G`.** T-09 measured six start nodes on
  one graph. The design-note sweep found 1,175 graphs on 5–6 nodes where greedy *does*
  attain it from some node, which says the opposite is common — but neither number is a
  rate over any population the paper reports on.
- **Not any statement about figure page cost.** The printed sizes in §3 are figure
  dimensions, not typeset page fractions. **T-26** owns that arithmetic, and the inventory
  in `manuscript.md` §2 priced this as **one** 0.75-page figure when it is now **four**
  panels plus the schematic.
- **Not that the G2S panel shows the canonical search.** It shows **one execution** — the
  greedy one for the exhaustive form, the pruned canonicalisation's chosen one for the
  pruned form. The *search* over executions is the search-tree schematic's job, and that
  figure is truncated at depth 3.
- **Not a claim about `Remark 2.7` from the figure alone.** §1's correction rests on reading
  Definition 2.6 and `core/canonical.py`, with the six-start-node table as illustration. The
  figure makes the gap visible; it does not prove it.
