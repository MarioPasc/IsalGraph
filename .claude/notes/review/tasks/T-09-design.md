# T-09 design — explanatory figures

**Ticket**: T-09, board row 197 of `.claude/notes/review/plan/tickets.md`.
**Deliverables**: the canonical search-space schematic (**R3.7c, requested**) and the S2G/G2S
worked example (**author preference, first page cut**); both regenerate the graphical abstract.
**Depends**: nothing. Not on the critical path.
**Base commit**: `27bdfcb` (main, clean apart from an untracked `T-06-FILES.md`).
**Read**: [manuscript](../plan/manuscript.md) §2–§3, [decisions](../plan/decisions.md) S-g,
[demands](../plan/demands.md) R3.7c.

## 0. Agreed with the PI, 2026-08-25, before any figure was rendered

| Question | Decision |
|---|---|
| What the G2S panel shows | **The encoder's own iterations, with a validated pair-loop mirror.** Not the replay. §3 |
| Panel geometry | **Filmstrip wrapped into blocks of five columns**, each column `(CDLL ring, instruction strip, graph)`, 7.0 in wide. §4 R6 |
| Running example | **Switch to the shared example** in §2 — one graph for all three figures |

---

## 1. State measured now

Everything below was measured on 2026-08-25 in this checkout, not taken from the plan.

| Fact | Plan said | Measured | Consequence |
|---|---|---|---|
| Search-tree renderer | "renderer exists: `viz/search_tree.py::canonical_search_tree_figure`" | **exists, 743 lines, parameterised** (`max_depth`, `max_roots`, `figsize`, `show_graph_inset`), tested against `canonical_string` on six families | R3.7c needs **polish and re-pointing**, not a new renderer |
| S2G/G2S worked example | "NEW" | **`figures.py::build_s2g_steps_figure` / `build_g2s_steps_figure` already exist** and are committed to `docs/figures/` | the ticket is a rebuild, not a green field |
| The figure the manuscript lost | — | `fig_algorithm_overview.pdf` is **commented out at `methodology.tex:379–382`** with the note `%% Figure commented out to meet the 35-page limit`. Its caption describes a **row-per-step** layout | "first page cut" is literal; the figure exists in the source, disabled |
| Engine | — | `isalgraph.engine() == 'cpp'`, build hash `298fc1188bf1`, matplotlib 3.11.1, env `isalgraph-cpp` | no worktree; **T-09 imports `isalgraph`, so it runs in place** (§3 of `review-ticket`) |
| Results root | "results folder" | `…/isalgraph/results/reports/T-0x-<slug>/{figures,data}/` is the established convention (T-05, T-06, T-27) | T-09 writes `reports/T-09-explanatory-figures/figures/` |
| Graphical abstract | "regenerate" | `graphical_abtract.pdf` panel **(b) carries numbers T-06 retired** — `Wins: 99.6 %`, `β = 0.537`, `R² = 0.947`, `14,108×` | **T-09 regenerates panel (a) only.** Panel (b) is stale science and belongs to T-20/T-24, and is recorded here so it is not silently reprinted |

### Two defects the existing figures carry

1. **The two committed step figures are of two different examples.** `build_s2g_steps_figure`
   decodes `WORKED_EXAMPLE_STRING = "VNVnVCPvNC"`; `build_g2s_steps_figure` encodes
   `WORKED_EXAMPLE_EDGES`, a 7-node graph, and emits `VVpvvpvPCPV`. Nothing connects them, so the
   pair cannot illustrate the round trip, which is the one property that makes the two algorithms
   worth showing together.
2. **The instruction strip marks past and future but not *present*.**
   `instruction_view.draw_instruction_strip` splits cells on `i < current_idx` /
   `i >= current_idx` (`instruction_view.py:109`); no cell is distinguished as *the instruction
   being executed*. In a worked example that is the single most important cell.

A third, which is a documented decision rather than a defect: `_grey_masks`
(`composite.py:49`) ghosts with `GRAYED_FACE = "#DDDDDD"` at `alpha = 0.25` — a **filled grey**.
The sibling projects ghost with a **white fill and a dashed grey outline**
(`IsalSR/viz/backends/matplotlib_dag.py:215–238`: `GHOST_FACE "#ffffff"`,
`GHOST_EDGE_COLOR "#b9bec7"`, `linestyle (0, (2.6, 2.0))`, `GHOST_TEXT_COLOR "#8b939e"`), which is
what reads as "whitened" in print and what this ticket adopts.

---

## 2. The running example — chosen by enumeration, not by taste

A single graph serves all three figures, so the reader carries one example through §2. It was
picked by sweeping every connected graph on 5–6 nodes with `n` or `n+1` edges (1,175 candidates
satisfied the operation-coverage and greedy-attainment filters) against five criteria:

| # | Criterion | Why |
|---|---|---|
| C1 | `|w*_G| <= 10` | ten states fit one figure |
| C2 | `w*_G` contains a primary insert, a secondary insert, a connect and a move | all four operation classes appear |
| C3 | greedy `G2S(G, v₀) == w*_G` for some `v₀` | the G2S panel emits exactly the string the S2G panel consumes |
| C4 | `|Aut(G)| == 1` | the search tree branches genuinely; on a vertex-transitive graph it is degenerate |
| C5 | 5–6 nodes | legible at IEEE column scale |

**No 5-node graph satisfies C3 and C4 together.** At `n = 5` the only graphs whose greedy encoding
attains the canonical string are `C₅` (`|Aut| = 10`) and `K₂,₃` (`|Aut| = 12`) — and on both, every
start node yields the same string, so the search tree has nothing to show. That measurement is what
forces `n = 6`.

### The example

```
n = 6,  m = 6,  E = {(0,1), (0,2), (0,3), (1,3), (2,4), (3,5)}
degrees  0:3  1:2  2:2  3:3  4:1  5:1        |Aut(G)| = 1
```

A triangle `{0,1,3}`, a path `0–2–4`, a pendant `3–5`.

| Quantity | Value |
|---|---|
| `canonical_string(G)` | **`VVVnvPCPV`**, length 9 |
| greedy `G2S` from node **0** | **`VVVnvPCPV`** — attains `w*_G` |
| greedy `G2S` from nodes 1–5 | `VVpvpvPVPC` (10), `VVpvvnvPC` (9), `VVVpvnvNNpC` (11), `VpvnvvnvPC` (10), `VpvvPVnCNV` (10) — all six differ |
| `S2G(w*_G)` | 6 nodes, 6 edges, **isomorphic to `G`** (verified with `nx.is_isomorphic`) |
| S2G snapshots | 10 (one per symbol, plus the initial state) |
| G2S encoder iterations | **6**, emitting the groups `V │ V │ V │ nv │ PC │ PV` |
| `pruned_canonical_string(G)` | `VVpvvPVnC` — **length 9, different string** |

The last row is expected, not a defect: Remark 2.11 (`methodology.tex:614`) already states that the
pruned and exhaustive forms are different canonical forms. It is recorded because the caption must
say `w*_G` of the **exhaustive** definition, and because a reader who runs
`pruned_canonical_string` on the figure's graph will get a different string.

---

## 3. What each figure shows, and the honesty constraint on the G2S panel

`viz/README.md` is explicit, and it is the constraint that shapes this ticket:

> A `"g2s"` step figure shows an interpreter executing the finished string, not the encoder
> searching. […] tentative pointer positions it walks and abandons, displacement pairs it rejects
> because no operation applies, and the `V ≻ v ≻ C ≻ c` cascade it runs at each pair. None of that
> is in the trace.

`GraphToString.run_with_trace` **replays** the emitted string through `StringToGraph`. So a G2S
figure built from it shows the same state sequence as the S2G figure with the grey mask inverted.
Two panels of the same states are one panel printed twice.

`GraphToString.run(v₀, trace=True)` is different: it appends a snapshot **at the top of each outer
iteration** (`graph_to_string.py:145–153`), so it records the encoder's real iteration boundaries —
6 for this example, against the replay's 10. What it does not record is which displacement pairs
were tested and rejected inside an iteration.

**Design**: a read-only *mirror* of the pair loop in `viz/`, which replays
`generate_pairs_sorted_by_sum` from a recorded iteration state and records, per pair, which of
`V ≻ v ≻ C ≻ c` applied and why the others did not. `core/graph_to_string.py` is **frozen**
(`CLAUDE.md`) and is not touched. The mirror is made non-drifting the same way `search_tree.py` is:
a test asserts that the operation the mirror selects at every iteration equals the symbol group the
real encoder emitted, over a family of graphs. A mirror that disagrees fails the suite rather than
producing a wrong figure.

### The three figures

**F1 — S2G worked example.** Ten states, one per symbol of `w*_G`. Per step: the CDLL ring with
π/σ, the instruction strip with the executing cell marked, and the graph. The graph starts fully
whitened and fills in; the element created by the current step carries the accent halo.

**F2 — G2S worked example.** Six encoder iterations. Per iteration: the CDLL ring with π/σ and the
tentative positions, the displacement pairs tested in `P(M)` order with the rejected ones struck,
the operation the cascade selected, and the emission group appended to the string. The graph starts
solid and whitens as structure is captured.

**F3 — canonical search-space schematic (R3.7c).** `canonical_search_tree_figure` re-pointed at the
running example, with the path through start node 0 highlighted as `w*_G`.

**F4 — graphical abstract panel (a).** Composed from F1 and F2's primitives. Panel (b) is out of
scope and its staleness is recorded in §1.

---

## 4. Frozen before any figure is rendered

| # | Rule | Rationale |
|---|---|---|
| R1 | The running example is `n = 6`, `E = {(0,1),(0,2),(0,3),(1,3),(2,4),(3,5)}`, start node `0`, `w*_G = VVVnvPCPV`. It is a module constant, not a literal in a figure builder | one example across three figures; changing it later changes all three together or not at all |
| R2 | Every figure is defined by an explicit edge list and an explicit start node. **No random draw, no seed** | the committed artifacts are byte-reproducible across machines |
| R3 | The G2S panel is built from the **encoder's own iteration snapshots**, never from the replay trace | §3; the replay does not show the encoder deciding |
| R4 | The pair-loop mirror is validated against the frozen encoder by a test that compares the selected operation per iteration. A mismatch is a test failure, not a figure warning | `search_tree.py`'s precedent; a schematic that has drifted from its algorithm is worse than none |
| R5 | Ghost = white fill + dashed grey outline + grey glyph. Accent = halo on the element created by the current step. Dim = present but already consumed. **Three states, not two** | the sibling idiom; "whitened" is what the print reader sees |
| R6 | Figure widths are `IEEE_TEXT_WIDTH_INCHES` (7.0 in) for F1–F3. Heights are declared in a frozen layout dataclass in true printed inches | `IsalSR/viz/algorithm_trace.py:72`; kills the implicit scaling factor |
| R7 | Every figure is written as **both** `.pdf` (vector, for the manuscript) and `.png` (for the docs site), at `savefig.dpi = 300`, `pdf.fonttype = 42` | Elsevier requires embedded TrueType; the manuscript takes the PDF |
| R8 | Output goes to `…/results/reports/T-09-explanatory-figures/figures/`; `docs/figures/` stays the default of `python -m isalgraph.viz` | the results archive is the deliverable; the repo copy stays regenerable |
| R9 | No number that T-06 retired is reprinted in any T-09 artifact | the graphical abstract's panel (b) is the specific hazard |

---

## 5. Acceptance criteria

Each is checkable by the named command.

| # | Criterion | Proof |
|---|---|---|
| A1 | The running example's facts reproduce exactly: `canonical_string == "VVVnvPCPV"`, greedy from 0 equals it, `S2G(w*)` is isomorphic to `G`, `\|Aut\| = 1` | a test in `tests/viz/`, run with `$PY -m pytest tests/viz -q` |
| A2 | The pair-loop mirror selects the same operation as the frozen encoder on every iteration, over ≥ 20 graphs | the R4 test |
| A3 | F1, F2, F3 render to PDF and PNG without exception, at 7.0 in width | `$PY -m isalgraph.viz <dir> --formats pdf,png` writes 2 files per figure |
| A4 | `import isalgraph.viz` still succeeds with matplotlib blocked | `tests/viz/test_import_without_matplotlib.py` |
| A5 | `ruff check` and `mypy --strict` clean on `src/isalgraph/` | the two commands |
| A6 | Full suite at or above the reference state **2,550 passed / 321 skipped** | `$PY -m pytest tests/ -q` |
| A7 | The artifacts exist under `…/reports/T-09-explanatory-figures/figures/` | `ls` |
| A8 | Every figure is visually inspected at print size before the ticket closes | recorded in the work log |

---

## 6. Stop and ask

- The pair-loop mirror disagrees with the frozen encoder on any graph → **stop**. That is either a
  bug in the mirror or a fact about the encoder, and the second one is not a figure problem.
- Any change would require touching `core/graph_to_string.py`, `core/canonical.py` or another
  frozen reference file → **stop**. Re-proving C++ parity is not in a figure ticket.
- The full suite drops below the reference state → **stop and diagnose before continuing**.

---

## 7. Findings carried out of this ticket

**Remark 2.7 (`methodology.tex:462`, `\label{rem:search-space}`) is the prose R3.7c points at, and
it excludes half of its own search space.** It reads:

> Only the identity of the uninserted neighbour chosen at each `V`/`v` step contributes to the
> search space.

But Definition 2.6, three lines above it, defines `w*_G` as "the shortest string encodable by any
execution of GTS **from any starting node**", and `canonical.py` searches over start nodes as well
as over neighbour choices. The schematic R3.7c asks for therefore shows a **forest with one root
per start node**, which contradicts the word "Only" in the remark it illustrates.

This is measurable on the running example: the six start nodes give six different strings of
lengths 9, 10, 9, 11, 10, 10, and only node 0 attains the minimum. The start node is not a free
choice; it is the outer loop of the search.

**Proposed fix** — one clause, no page cost: *"Two things are searched over: the starting node, and
the identity of the uninserted neighbour chosen at each `V`/`v` step. The priority order […] and the
minimum-displacement pair ordering […] are intrinsic to the algorithm definition and are not
branched over."* **Owner: T-11** (manuscript defects) or **T-20** (§2 rewrite); T-09 emits the note
and the figure that makes the gap visible.
