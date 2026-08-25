# `isalgraph.viz`

Figure toolkit for the IsalGraph paper. Every figure in the revision is
built from these primitives, and the same primitives render an instruction
string, the CDLL that executes it, and the graph it builds, in one visual
language.

## Hello world

```python
from isalgraph.core.string_to_graph import StringToGraph
from isalgraph.viz.composite import single_card_figure

graph, trace = StringToGraph("VNVnVCPvNC", directed_graph=False).run_with_trace()
fig = single_card_figure(trace, full_graph=graph, title="S2G final state")
fig.savefig("card.png", dpi=300)
```

Regenerate every committed figure with `python -m isalgraph.viz`; the
builders live in `figures.py` and write to `docs/figures/`.

## Public API

### Traces (`isalgraph.core.trace`, stdlib-only)

| Name | Purpose |
|---|---|
| `StepSnapshot` | VM state after one instruction: ring order, both pointers, active node/edge masks, `created_edge`, partial string. |
| `AlgorithmTrace` | `direction` (`"s2g"`/`"g2s"`), `directed`, `final_graph`, `snapshots`. Envelope tag `isalgraph.trace.v1`. |
| `dump_trace` / `load_trace` | JSON round-trip. |
| `graph_edges` / `graph_to_dict` | Edge enumeration and envelope serialisation for `SparseGraph`. |
| `cdll_forward_order` | Ring payloads in forward circular order, as graph node ids. |

Produced by `StringToGraph.run_with_trace()` and
`GraphToString.run_with_trace(initial_node)`. Neither changes the existing
`run()` signatures or the instruction semantics.

### What a `"g2s"` trace does and does not show

`GraphToString.run_with_trace` takes the string from the frozen `run()` and then
**replays it** through `StringToGraph.run_with_trace`; it does not instrument the
encoder. The replay is exact — the encoder allocates output nodes and grows the
CDLL with the same calls the interpreter makes, so the replay reconstructs the
encoder's output graph identically, not merely up to isomorphism — and it is the
only way to obtain one state per instruction, because the encoder emits a whole
displacement group (`"NNN"` + `"V"`) per iteration and jumps its pointer straight
to the tentative slot.

The consequence matters when reading a figure. **A `"g2s"` step figure shows an
interpreter executing the finished string, not the encoder searching.** The
encoder's search is strictly richer: tentative pointer positions it walks and
abandons, displacement pairs it rejects because no operation applies, and the
`V ≻ v ≻ C ≻ c` cascade it runs at each pair. None of that is in the trace. For
the decision structure, use `canonical_search_tree_figure` instead — that is
exactly what it draws.

Two further decisions worth knowing:

- **Snapshots store graph node ids, already resolved.** CDLL indices are
  not graph node ids (`core/README.md`, invariant 1). Resolving at
  emission means the view layer never calls `cdll.get_value` and cannot
  conflate the two index spaces.
- **`created_edge` is recorded by the emitter.** A `C`/`c` between
  already-adjacent nodes is a genuine no-op, so attributing edges by
  counting `V`/`C` tokens desynchronises on the first such instruction.

### Views

| Function | Draws |
|---|---|
| `cdll_view.draw_cdll_ring(ax, order, primary_idx, secondary_idx, ...)` | The ring, legacy positional signature preserved. |
| `cdll_view.draw_cdll_ring_for_snapshot(ax, snapshot, ...)` | The ring for a `StepSnapshot`. |
| `instruction_view.draw_instruction_strip(ax, instructions, *, current_idx, direction, ...)` | The instruction strip. |
| `graph_view.draw_graph(ax, graph, *, backend, ...)` | The graph, via the backend registry. |
| `nx_view.draw_nx_graph(G, ax, ...)` | A `networkx.Graph`, absorbed from the benchmark scripts. |
| `alignment_view.levenshtein_alignment` / `draw_alignment` | String alignment for the distance figures. |

### Composites

`composite.draw_column`, `single_card_figure`, `steps_figure`,
`roundtrip_figure`; `search_tree.canonical_search_tree_figure`; and the
two manuscript panels, `worked_example.s2g_worked_example_figure` /
`g2s_worked_example_figure`.

Palettes are built once per figure from the final graph, and the layout is
pinned by threading each column's returned layout into the next, so nodes
neither recolour nor move between frames.

### Style

`style` holds the palettes, `BASE_RCPARAMS` (`apply_ieee_style()`), IEEE
widths, `get_figure_size`, `render_colored_string` and `save_figure`.
`benchmarks/plotting_styles.py` re-exports from here, so there is one
source of truth and the paper's existing figures cannot drift in colour.

## Backend contract

A backend implements `GraphVizBackend`:

```python
def draw(self, graph, ax, *, node_colors, edge_colors,
         grayed_nodes=frozenset(), grayed_edges=frozenset(),
         layout=None) -> dict[NodeId, Position]: ...
```

Three clauses, all covered by tests in `tests/viz/test_rendering.py`:

1. **A backend never creates a figure.** It paints on the supplied `Axes`.
2. **Layout in, layout out.** Given `layout`, it must reuse those exact
   coordinates; given `None`, it computes one and returns it. This is what
   pins node positions across the columns of a step figure.
3. **Grey masks are caller-decided.** Which elements are ghosts depends on
   direction, and that is the composer's call, not the backend's.

`is_available()` is a classmethod declaring whether the backend's
third-party library imports. `available_backends()` filters on it, so it
never lists a backend that cannot draw — the upstream IsalHG version
detects the library only at draw time and reports backends that then fail.

| Backend | Requires | Notes |
|---|---|---|
| `matplotlib` | matplotlib | **Default.** Circles and lines; no drawing library needed. |
| `networkx` | networkx | `draw_networkx_*`; richer layouts, curved edges. |
| `igraph` | igraph | Kamada-Kawai / Sugiyama layout, painted by the matplotlib backend. |

Importing `isalgraph.viz` succeeds with none of these installed: every
third-party import sits inside a function body and `Axes` is referenced
only under `TYPE_CHECKING`. `tests/viz/test_import_without_matplotlib.py`
enforces both, by blocking the imports with a meta-path finder and by
walking the AST of every module.

## Colour semantics

Nine symbols in four case-pairs. The fact a reader most needs is *which
pointer acted*, so two orthogonal channels carry it.

**Hue encodes the operation.** `INSTRUCTION_PALETTE` is unchanged from
`benchmarks/plotting_styles.INSTRUCTION_COLORS`, so figures already in the
paper keep their colours: movement blue-cyan, insertion green, connection
red-rose, `W` grey. Within a case-pair the lowercase variant is a lightness
shift of its uppercase partner, so the pair reads as one family.

**A stroke accent encodes the pointer.** Each instruction cell is outlined
in `POINTER_PALETTE[0]` (primary, `#EE6677`) or `POINTER_PALETTE[1]`
(secondary, `#4477AA`) — the same two colours as the CDLL ring arrows. The
correspondence is mechanical: the red arrow, the red-outlined `V` cell and
the π label are one pointer, verifiable at print size without a legend.

Hue alone could not carry the pointer channel without recolouring an
already-published palette, which is why the accent is a second channel
rather than a replacement.

Two further rules:

- **Categorical palettes exclude grey.** `GRAYED_FACE` means "not yet
  built"; `tab20`/`tab20c` contain greys, so an active node coloured from
  them can look ghosted. `build_node_palette` and `build_edge_palette` draw
  from `PAUL_TOL_MUTED` and fall back to a continuous colormap.
- **The ring colours nodes by pointer, the graph by identity.** Deliberate:
  in the ring, pointer position is the information; in the graph panel,
  node identity is. The pointer channel stays consistent across both.

## Grey-mask inversion

`direction` inverts what counts as built:

- `"s2g"` — the algorithm builds the graph from the string. Built elements
  are solid, the rest are ghosts of the target; the panel starts fully grey
  and ends fully coloured. The strip runs the other way, greying out as
  instructions are spent.
- `"g2s"` — the algorithm consumes the graph and emits the string. Encoded
  elements are ghosts, what remains solid is what it has yet to capture;
  the panel starts solid and ends grey, and the strip fills in.

For a standalone card the progress mask conveys nothing, so
`single_card_figure(..., color_whole_string=True)` (the default) colours
every cell.

## The two worked-example panels

`worked_example` builds the S2G and G2S panels the manuscript lost --
`fig_algorithm_overview.pdf` is still in the article source, commented
out at `methodology.tex:379` with the note *"Figure commented out to
meet the 35-page limit"*. Four decisions carry them.

**One index for both.** S2G consumes one symbol per step; G2S emits a
whole *group* per pass of its outer loop. Indexed by their own units the
two panels would have different column counts and nothing to compare, so
both are indexed by the **encoder's group boundaries**. For the running
example that is `V | V | V | nv | PC | PV` -- six columns, no step
omitted from either.

**One renderer.** `draw_columns` is the only function that lays anything
out. Both builders reduce their trace to a tuple of `ExampleColumn`, so
the panels are identical in geometry by construction rather than by two
builders being kept in step.

**Ghosting is a conservation argument.** Ink moves between the strip and
the graph in the direction the algorithm runs:

| | strip | graph |
|---|---|---|
| **S2G** | starts solid, drains | starts ghosted, fills |
| **G2S** | starts ghosted, fills | starts solid, drains |

Making both panels fill in -- which the first draft did -- is what makes
the two figures look like the same algorithm drawn twice. A ghost is a
white disc with a dashed grey outline (the `IsalSR` idiom), so it
recedes by carrying no ink rather than grey ink; the element the step
moved between representations carries an accent halo in both panels.

**The G2S panel is built from the encoder, not from a replay.** See the
next section.

## `encoder_trace`: what the G2S panel is drawn from

`GraphToString.run_with_trace` replays the finished string, so a panel
built from it is the S2G panel with its mask flipped. `encoder_trace`
re-runs the encoder loop with the rejected displacement pairs recorded,
in input-graph node ids.

`core/graph_to_string.py` is frozen, so this is a mirror, and a mirror's
hazard is drift. `tests/viz/test_encoder_trace.py` pins it: the mirror's
string must equal `GraphToString`'s byte for byte, exhaustively on small
graphs and on random larger and directed ones. Out of band it was checked
on **134,609 `(graph, start)` pairs to n = 14, with zero mismatches**.

`trace_execution(graph, target)` recovers the execution behind a *given*
string by branching only on the uninserted-neighbour choice at `V`/`v` --
the only freedom an execution has (Remark 2.7). That is what lets the
**pruned** canonical string be drawn with the same encoder-side detail as
the greedy one: no greedy run emits it.

## Canonical search tree

`search_tree.canonical_search_tree_figure(graph)` answers Reviewer 3's
request for a schematic of the search space. Read against
`core/canonical.py::_step`:

- **Branches:** the starting node (`canonical_string` searches from every
  node reaching all others), and the uninserted-neighbour choice at each
  `V`/`v` step (`_step` recurses over the whole candidate set).
- **Fixed, never branching:** displacement ordering — `_step` walks pairs
  in increasing `|a| + |b|` and commits to the first admitting any
  operation, every arm ending in `return` — and the priority
  `V ≻ v ≻ C ≻ c`.

The enumerator replays from the root rather than backtracking with undo, so
it needs nothing private from `core`. `tests/viz/test_search_tree.py`
checks it reproduces `canonical_string` on six graph families, so the
schematic cannot drift from the algorithm it depicts.
