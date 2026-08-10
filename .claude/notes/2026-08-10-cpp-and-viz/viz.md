# Work log — viz

## Identity

| Field | Value |
|---|---|
| Agent | `wave-viz` |
| Wave | `2026-08-10-cpp-and-viz` |
| Model / effort | Opus 4.7 (claude-opus-5) / xhigh |
| Branch | `wave/viz` |
| Worktree | `/home/mpascual/research/code/IsalGraph/.claude/worktrees/agent-a6973cd64d76ae718` |
| Base commit | `2f393a1` |
| Head commit | `80459e9` (+ this log commit) |
| Started / finished | 2026-08-10 · single session |
| Status | complete |

## 1. Prompt as received

```
You are the agent `wave-viz` in a two-agent parallel wave. Your orchestrator is `main`
(reach it with SendMessage({to: "main", message: "..."})). You work in your own git worktree on
your own branch. Read this entire brief before touching anything.

# 0. Who you are

You are a scientific-visualization engineer building a reusable figure toolkit for a paper under
revision at Pattern Recognition (deadline 2026-08-31).

The product is a **library**, not a set of scripts. Every figure in the revision must be buildable
from `isalgraph.viz` primitives, and the same primitives must render an IsalGraph instruction
string, the CDLL that executes it, and the graph it builds, in one coherent visual language.

Standing rules:
- **`isalgraph.core` stays dependency-free.** Anything you add under `core/` is stdlib-only.
- **Every third-party import in `viz/` lives inside a function body**, never at module scope, so
  `import isalgraph.viz` succeeds without matplotlib installed. This is a hard rule, not a
  preference.
- **A backend never creates a figure.** It paints on a caller-supplied `Axes` and **returns the
  layout it used**, so callers can pin node positions across panels.
- **Do not change algorithm semantics.** You add trace emission; you do not touch how instructions
  execute.
- Never claim a figure renders unless you rendered it and looked at the file.
- You cannot ask the user anything. If you need a decision, message `main` and continue on a
  recorded assumption.

# 1. Wave facts

| Field | Value |
|---|---|
| `WAVE_ID` | `2026-08-10-cpp-and-viz` |
| `BASE_SHA` | `2f393a1` |
| Your branch | create `wave/viz` in your worktree |
| Main checkout (read-only for you) | `/home/mpascual/research/code/IsalGraph` |

# 2. Mission

Build `src/isalgraph/viz/`, modelled closely on the sibling repo IsalHG's `viz` package
(`/home/mpascual/research/code/IsalHG/src/isalhg/viz/` — read all of it), adapted from hypergraphs
to plain simple graphs.

IsalGraph's drawing code is currently scattered through `benchmarks/real_data/eval_visualizations/`
as one-off figure scripts. The revision needs figures that a reviewer asked for explicitly — among
them a schematic of the canonical search space — and they must be built from shared primitives, not
another one-off.

## 2.1 What IsalGraph objects look like

Read `src/isalgraph/core/README.md` in full first. In brief:

- `SparseGraph` — contiguous integer node ids, adjacency sets, directed or undirected.
- `CircularDoublyLinkedList` — array-backed CDLL. **CDLL indices are NOT graph node indices**; the
  graph node at a CDLL index is `cdll.get_value(idx)`.
- **Exactly two pointers**: primary and secondary. They are CDLL indices.
- Alphabet `Σ = {N, n, P, p, V, v, C, c, W}` — 9 symbols, uppercase acts on the primary pointer,
  lowercase on the secondary. `N/P` = move next/prev, `V/v` = new node + edge, `C/c` = edge between
  the two pointed nodes, `W` = no-op.

## 2.2 Deliverable A — the trace schema (`src/isalgraph/core/trace.py`, new, stdlib-only)

Model it on `/home/mpascual/research/code/IsalHG/src/isalhg/core/trace.py`. Keep its three
load-bearing design decisions:

1. A snapshot stores **id masks over one final structure**, never a copy of the structure.
2. The full graph is serialised exactly once, in the envelope.
3. The envelope carries a version tag: `"schema": "isalgraph.trace.v1"`.

Required shape (frozen dataclasses, `to_json`/`from_json`, `dump_trace`/`load_trace`):

```python
@dataclass(frozen=True)
class StepSnapshot:
    step_idx: int                       # 0 = initial state; i = after instruction i-1
    instruction: str | None             # one char of Σ, or None at step 0
    cdll_node_order: tuple[int, ...]    # GRAPH node ids in forward circular order from the head
    primary_node: int                   # GRAPH node id under the primary pointer
    secondary_node: int                 # GRAPH node id under the secondary pointer
    active_nodes: tuple[int, ...]       # sorted
    active_edges: tuple[tuple[int, int], ...]   # sorted, normalised (u<=v) when undirected
    created_edge: tuple[int, int] | None        # the edge this step created, if any
    partial_string: str

@dataclass(frozen=True)
class AlgorithmTrace:
    direction: str                      # "s2g" | "g2s"; validate in __post_init__
    directed: bool
    final_graph: dict                   # {"n_nodes": int, "edges": [[u, v], ...], "directed": bool}
    snapshots: tuple[StepSnapshot, ...] # length == len(instructions) + 1
```

Two IsalGraph-specific requirements:

- **Snapshots store graph node ids, already resolved from CDLL indices.** The view layer must never
  call `cdll.get_value`. This is invariant 1 of `core/README.md` and the single easiest way to
  corrupt a figure silently.
- **`created_edge` is recorded by the emitter, not re-derived by the view.** IsalHG re-derives
  edge attribution in its view layer by counting `V`/`C` tokens, and its own docstring admits the
  counter desynchronises if an edge is added twice. In IsalGraph, `C`/`c` between already-adjacent
  nodes is a genuine no-op, so re-deriving *would* desynchronise. Record it at the source.

## 2.3 Deliverable B — trace emission in the two converters

`StringToGraph` and `GraphToString` already accept `trace=True` and return deep-copied snapshot
tuples. That is expensive and unstructured.

**Add new methods; do not change the existing `run()` signatures or return types**, and do not touch
`_execute_instruction` semantics or the greedy search. Your peer `wave-cpp-engine` is porting these
exact algorithms to C++ and its parity target is the current behaviour.

```python
StringToGraph.run_with_trace(self) -> tuple[SparseGraph, AlgorithmTrace]
GraphToString.run_with_trace(self, initial_node: int) -> tuple[str, AlgorithmTrace]
```

Add a test asserting `run()` and `run_with_trace()` produce the same graph / same string, so the
frozen behaviour is pinned.

## 2.4 Deliverable C — `src/isalgraph/viz/`

Port the IsalHG architecture. The recon below is accurate; trust it and read the source to confirm.

| File | Port status |
|---|---|
| `base.py` | 1:1. `Position` alias + `GraphVizBackend` ABC. Contract: backend never creates a figure, paints on a supplied `Axes`, takes `layout` in and **returns the layout used**, receives `grayed_nodes`/`grayed_edges` as caller-decided sets. Import `Axes` under `if TYPE_CHECKING:` and alias to `Any` at runtime. |
| `style.py` | Port structure, extend the palette (see below). |
| `layout.py` | `cdll_ring_positions` ports verbatim (stdlib `math` only). `compact_primal_layout` simplifies: no clique expansion needed, so it becomes `nx.spring_layout` on the graph itself, keeping the normalise-to-`fit_fraction` step and the stray-component strip for disconnected graphs. |
| `registry.py` | 1:1, **plus fix an upstream defect**: IsalHG's `available_backends()` reports backends whose third-party library is missing, because detection happens at draw time. Add an `is_available()` classmethod to the ABC and have `available_backends()` call it. |
| `trace_io.py` | Same ~70-line shape; `graph_from_edgelist` replaces `hypergraph_from_hif`. |
| `instruction_view.py` | Highest-value port. Adapt for 9 symbols instead of 5. |
| `cdll_view.py` | Near 1:1 and simpler with two pointers. |
| `graph_view.py` | Replaces `hypergraph_view.py`. Keep the ~45-line dispatcher unchanged in shape. |
| `composite.py` | Structure ports: `GridSpec` + `subgridspec`, `_ROW_RATIOS`, palettes built once per figure from the final graph, layout pinned by threading the returned layout into the next column, `_sample_indices` for even step sampling, and the S2G/G2S grey-mask inversion. |
| `backends/matplotlib_backend.py` | **New, and make it the default.** A plain graph needs no third-party drawing library: nodes as `Circle`, edges as `Line2D`, arrowheads for directed. ~40 lines. This makes the whole module work on matplotlib alone. |
| `backends/networkx_backend.py` | Optional; `nx.draw_networkx_*` with `ax=`. |
| `backends/igraph_backend.py` | Optional; for Kamada-Kawai / Sugiyama layouts. |
| IsalHG's three hypergraph backends | **Drop.** Two of them exist only to work around HyperNetX/HyperGraphX bugs. |
| `cohort_panel.py` | Port if time allows; retune the hardcoded `1.2` xlim and drop `axis_margin` to ~0.12 (it was sized for HyperGraphX bezier hulls; straight edges need far less). |

### Colour semantics — the one real design decision

IsalHG's alphabet has 5 symbols and no case distinction. IsalGraph has 9 in 4 case-pairs, and the
information a reader most needs is **which pointer acted**. Use a two-axis scheme:

- **Hue = operation**: movement (`N`/`P`, `n`/`p`), insertion (`V`/`v`), connection (`C`/`c`),
  no-op (`W`, neutral grey).
- **Case = pointer**: uppercase/primary and lowercase/secondary must be visually paired but
  distinguishable (e.g. saturation or lightness).
- **Tie the pointer distinction to `POINTER_PALETTE[0]` and `[1]`** so that primary is the same
  colour on the instruction strip, on the CDLL ring arrow, and in the graph view. IsalHG's
  `style.py` docstring states this mechanical-correspondence principle for hyperedges; it is the
  reason its figures read at print size, and it is worth more here than any other styling choice.

Start from `benchmarks/plotting_styles.py` in this repo, which already holds Paul Tol palettes,
IEEE column widths (`IEEE_COLUMN_WIDTH_INCHES = 3.39`, `IEEE_TEXT_WIDTH_INCHES = 7.0`),
`PLOT_SETTINGS` rcParams, `get_figure_size`, `render_colored_string`, `bootstrap_ci`,
`save_figure`, `save_latex_table` — the paper's existing visual identity. Preserve it; the revision
must not look like a different paper. Keep `pdf.fonttype = 42` / `ps.fonttype = 42`.

Fix one upstream bug while porting `save_figure`: IsalHG uses `Path.with_suffix`, which truncates at
a dot in the stem (`run_1.5` → `run_1.pdf`). Use `bp.parent / f"{bp.name}.{fmt}"`.

### Existing primitives to absorb

`benchmarks/real_data/eval_visualizations/` already contains working code — absorb it, then leave
thin re-export shims behind so existing figure scripts keep importing:

- `cdll_drawing.py` — `draw_cdll_ring`, `_draw_pointer_arrow`, instruction legend
- `graph_drawing.py` — `draw_graph`
- `string_alignment.py` — `levenshtein_alignment`, `draw_alignment`
- `benchmarks/plotting_styles.py` — the style system above

## 2.5 Deliverable D — the reviewer's figure

Reviewer 3 asked, verbatim:

> Section 2.3 could also benefit from a small schematic illustrating the canonical search space:
> different starting nodes and alternative uninserted-neighbour choices form the search branches,
> whereas displacement ordering and the priority (V ≻ v ≻ C ≻ c) remain fixed.

Provide a `canonical_search_tree_figure(...)` in `viz/` that draws exactly this: a tree whose roots
are the candidate starting nodes, whose branches are the alternative uninserted-neighbour choices at
`V`/`v` steps, and which visually marks displacement ordering and the `V ≻ v ≻ C ≻ c` priority as
**fixed, non-branching** constraints. Read `src/isalgraph/core/canonical.py::_step` to get the
branching structure right — note that the search commits to the first displacement pair `(a,b)` that
admits any operation and branches only over the candidate set. Render it for a small worked example
(6–8 nodes) and commit the PNG under `docs/figures/`.

# 3. Ownership — you may create or edit ONLY these paths

```
src/isalgraph/viz/**                                          (new)
src/isalgraph/core/trace.py                                   (new)
src/isalgraph/core/string_to_graph.py                         (ADD run_with_trace only)
src/isalgraph/core/graph_to_string.py                         (ADD run_with_trace only)
tests/viz/**                                                  (new)
tests/unit/test_trace.py                                      (new)
benchmarks/plotting_styles.py
benchmarks/real_data/eval_visualizations/cdll_drawing.py
benchmarks/real_data/eval_visualizations/graph_drawing.py
benchmarks/real_data/eval_visualizations/string_alignment.py
docs/figures/**                                               (new)
.claude/notes/2026-08-10-cpp-and-viz/viz.md                   (your log)
```

**Everything else is read-only to you.** Explicitly, do NOT edit: `pyproject.toml` (`main` has
already added `networkx` to the `viz` extra for you), `src/isalgraph/__init__.py`,
`src/isalgraph/core/__init__.py`, `src/isalgraph/errors.py` (it already has `VizError`,
`VizBackendNotFoundError`, `VizBackendUnavailableError` waiting for you — use them, do not add
more), `src/isalgraph/core/{canonical,canonical_pruned,cdll,sparse_graph}.py`,
`src/isalgraph/core/backends.py`, `src/isalgraph/core/native/**`, `CMakeLists.txt`,
`experiments/**`, any other file under `benchmarks/`.

`main` owns the two `__init__.py` files and will wire `isalgraph.viz` into the public API at merge
time. Do not do it yourself.

Your peer `wave-cpp-engine` is adding a C++ backend under `src/isalgraph/core/native/` and a
dispatcher `src/isalgraph/core/backends.py`. Your code must keep working when the C++ engine is
absent — which is the state of your worktree. Do not import `backends` or `_native`.

# 4. Environment — verbatim

```bash
WT=$(git rev-parse --show-toplevel)
PY=~/.conda/envs/isalgraph/bin/python
cd "$WT"
export PYTHONPATH="$WT/src"        # REQUIRED — see below
```

The `isalgraph` conda env has an editable install pointing at the **main checkout**
(`/home/mpascual/research/code/IsalGraph/src`) via a plain `.pth` file. Without `PYTHONPATH` your
tests would import the main tree's code and never see your new modules. Verify before doing
anything else:

```bash
$PY -c "import isalgraph; print(isalgraph.__file__)"   # MUST print a path inside your worktree
```

Do **not** run `pip install`; it would repoint the shared environment and break your peer.

Installed: Python 3.11.15, matplotlib 3.11.1, networkx 3.6.1, igraph 1.0.0, numpy 1.26.4,
scipy 1.17.1, pandas 3.0.5, pytest 9.1.1, hypothesis, ruff 0.16.2, mypy 2.3.0. No `xgi`,
`hypernetx`, or `hypergraphx` — you do not need them. 24 cores; leave headroom for your peer.

```bash
$PY -m pytest tests/ -q
$PY -m ruff check src/ tests/ benchmarks/
$PY -m mypy src/isalgraph/
```

**Baseline on `BASE_SHA`: 450 passed, 271 skipped, 0 failed.** Do not reduce it.

Testing convention, copied from IsalHG because it survives matplotlib version bumps: `matplotlib.use("Agg")`
at module scope before importing pyplot; `pytest.importorskip` for optional backends; assert that
rendered files exist and exceed a byte-size floor (~1000 B) rather than hashing images; always
`plt.close(fig)`. Put the real assertions on the pure functions — palette hex validity, layout
geometry, registry errors, trace round-trips.

# 5. Definition of done

1. `import isalgraph.viz` succeeds in an interpreter with matplotlib uninstalled. Prove it, e.g.
   by blocking the import with a `sys.modules` sentinel or a stub finder in a test.
2. `isalgraph.core.trace` is stdlib-only; a trace round-trips through `dump_trace`/`load_trace`
   byte-identically. Assert no third-party import appears at module scope in `core/`.
3. `run()` and `run_with_trace()` agree on the graph/string they produce, for both converters, over
   a property test.
4. `len(trace.snapshots) == len(instructions) + 1`, and replaying the snapshots' `active_nodes` /
   `active_edges` reconstructs the final graph.
5. A working `matplotlib` backend as default, plus at least one optional backend, both satisfying
   the `layout`-in/`layout`-out contract. Verify pinning works: two panels drawn with a threaded
   layout place every node at identical coordinates.
6. Renders, committed under `docs/figures/` as PNG, and inspected by you:
   - a single "card" (CDLL ring + instruction strip + graph) for one worked example;
   - a multi-column step figure for a S2G trace and for a G2S trace, with the grey masks inverted
     between them;
   - the canonical-search-tree schematic of §2.5.
7. Shims left in `benchmarks/real_data/eval_visualizations/` so existing figure scripts still
   import; confirm by importing each previously-working module.
8. `ruff check` clean on `src/`, `tests/`, `benchmarks/`; `mypy --strict` clean on
   `src/isalgraph/`; full suite green.
9. A `src/isalgraph/viz/README.md` documenting the public API, the backend contract, the colour
   semantics, and the 5-line hello-world.

# 6. Mandatory obligations

- **Commit everything before you finish.** An uncommitted worktree does not exist to a merge.
- **Write your log to `.claude/notes/2026-08-10-cpp-and-viz/viz.md` inside your worktree and commit
  it** as your final commit (`docs(notes): viz work log`). Use the template at
  `/home/mpascual/.claude/skills/parallel-agents/assets/note-template.md` — read it and follow every
  section. Write it as you go, not from memory at the end.
- Your log's file list must match `git diff --name-only 2f393a1..HEAD` exactly. `main` checks.
- Conventional commits (`feat:`, `fix:`, `test:`, `docs:`). No `Co-authored-by` trailers.
- Do not push, rebase, merge, or touch your peer's branch.

# 7. Peers

| Name | Mission | Owns |
|---|---|---|
| `main` | Orchestrator (this session). Merges, owns both `__init__.py`, `errors.py`, `pyproject.toml`, `experiments/`, docs. | integration |
| `wave-cpp-engine` | nanobind C++ engine with parity to the frozen Python reference | `CMakeLists.txt`, `pyproject.toml`, `src/isalgraph/core/native/**`, `src/isalgraph/core/backends.py`, `src/isalgraph/core/algorithms/*.py`, `tests/native/**`, `docs/engineering/**` |

Message `main` when: you need a contract changed; you want to alter an existing signature in the two
converters; a needed dependency is missing; or you are about to exceed your ownership boundary.

# 8. Final message to main

End with a compact report: the public API you settled on, which files ported 1:1 vs. needed
redesign, the figures you rendered and where they are, what you could not do and why, your branch
name and head SHA, and the single thing `main` should scrutinise first.
```

## 2. Understanding and plan

**Restatement of the task in my own words:** Build a reusable figure library
`isalgraph.viz`, modelled on IsalHG's viz package but adapted from hypergraphs
to simple graphs, backed by a new stdlib-only trace schema in
`isalgraph.core.trace` and new `run_with_trace` methods on both converters that
do not disturb the frozen algorithms. Deliver four rendered figures, including
a schematic of the canonical search space that Reviewer 3 asked for.

**Approach chosen:** Port the IsalHG module structure file-for-file where the
abstraction transfers (base/registry/layout/composite/trace_io), redesign where
hypergraph assumptions do not (graph_view, matplotlib backend, colour
semantics), and absorb the three existing benchmark drawing modules by moving
their implementations into `viz` and leaving re-export shims.

**Alternatives considered and rejected:**
- Emitting the G2S trace from inside `GraphToString.run()`'s loop — rejected:
  it would mean editing the loop that `wave-cpp-engine` is porting for parity,
  and the encoder emits whole displacement groups per iteration so it cannot
  produce one snapshot per instruction anyway.
- Re-deriving `created_edge` in the view by counting `V`/`C` tokens (IsalHG's
  approach) — rejected: `C`/`c` over an already-adjacent pair is a genuine
  no-op in IsalGraph, so the counter desynchronises. Recorded at the emitter.
- Importing `canonical._step`, `_walk`, `_undo_edge`, `_undo_node` for the
  search-tree enumerator — rejected: those are private and `canonical.py` is
  read-only to me. Used replay-from-root instead, and pinned it with a test
  against `canonical_string`.
- Recolouring `INSTRUCTION_COLORS` so hue could encode the pointer — rejected:
  the palette is already published in paper figures. See §7.
- Keeping `tab20`/`tab20c` for the node/edge palettes (the IsalHG choice) —
  rejected after inspecting a render: both contain greys that collide with
  `GRAYED_FACE`, so an active node looked ghosted.

**Plan as executed:**
1. Read the whole IsalHG viz package and `core/trace.py`; read IsalGraph's
   `SparseGraph`, `CDLL`, both converters, `canonical._step`, and the existing
   benchmark drawing code.
2. Write `core/trace.py` (stdlib-only), add `run_with_trace` to both converters,
   validate against `run()` on randomised inputs before building anything on it.
3. Write the viz foundation (base, style, layout, registry), then the three
   backends, then the three views and `composite`.
4. Absorb `cdll_drawing` / `graph_drawing` / `string_alignment`; leave shims;
   single-source the palettes in `benchmarks/plotting_styles.py`.
5. Write `search_tree.py` for Reviewer 3's figure; write `figures.py` +
   `__main__.py` so the committed PNGs are reproducible.
6. Render, inspect each PNG, fix the defects found, re-render.
7. Tests, ruff, mypy, full suite.

**Deviations from the plan:** Three.
- The brief's literal colour instruction could not be followed without
  recolouring published figures; I used two orthogonal channels instead (§7).
- `cohort_panel.py` was listed as "port if time allows"; I did not port it (§9).
- `benchmarks/plotting_styles.save_figure` turned out **not** to have the
  `with_suffix` bug (it already used f-string concatenation). The bug was
  IsalHG's, and the fix landed in my `viz/style.py` port instead.

## 3. Changes made

**Created**

| Path | Purpose |
|---|---|
| `src/isalgraph/core/trace.py` | Stdlib-only trace schema: `StepSnapshot`, `AlgorithmTrace`, JSON I/O, edge helpers, `cdll_forward_order`. |
| `src/isalgraph/viz/__init__.py` | Package surface: ABC, `Position`, registry accessors. |
| `src/isalgraph/viz/__main__.py` | `python -m isalgraph.viz` regenerates `docs/figures/`. |
| `src/isalgraph/viz/base.py` | `GraphVizBackend` ABC, `Position`, `is_available()` hook. |
| `src/isalgraph/viz/style.py` | Palettes, rcParams, IEEE sizes, `save_figure`, `render_colored_string`. |
| `src/isalgraph/viz/layout.py` | `cdll_ring_positions`, `compact_graph_layout`. |
| `src/isalgraph/viz/registry.py` | Lazy name-keyed backend registry with availability filtering. |
| `src/isalgraph/viz/trace_io.py` | `graph_from_edgelist`, `load_trace_for_viz`. |
| `src/isalgraph/viz/instruction_view.py` | Instruction strip with the two colour channels + legend. |
| `src/isalgraph/viz/cdll_view.py` | CDLL ring: legacy signature preserved + snapshot entry point. |
| `src/isalgraph/viz/graph_view.py` | Backend dispatcher. |
| `src/isalgraph/viz/nx_view.py` | Absorbed `draw_graph` (networkx flavour), renamed `draw_nx_graph`. |
| `src/isalgraph/viz/alignment_view.py` | Absorbed `levenshtein_alignment`, `draw_alignment`. |
| `src/isalgraph/viz/composite.py` | `draw_column`, card/steps/roundtrip figures, grey-mask inversion. |
| `src/isalgraph/viz/search_tree.py` | Reviewer 3's canonical-search-space schematic + enumerator. |
| `src/isalgraph/viz/figures.py` | Reproducible builders for the committed figures. |
| `src/isalgraph/viz/backends/__init__.py` | Namespace. |
| `src/isalgraph/viz/backends/matplotlib_backend.py` | Default backend; Circle + Line2D + arrowheads. |
| `src/isalgraph/viz/backends/networkx_backend.py` | Optional `draw_networkx_*` backend. |
| `src/isalgraph/viz/backends/igraph_backend.py` | Optional igraph-layout backend, matplotlib painting. |
| `src/isalgraph/viz/README.md` | Public API, backend contract, colour semantics, hello-world. |
| `tests/unit/test_trace.py` | Trace schema + both emitters (22 tests). |
| `tests/viz/test_import_without_matplotlib.py` | Dependency-free import enforcement (22 tests). |
| `tests/viz/test_style_and_layout.py` | Palettes, layout geometry, registry (32 tests). |
| `tests/viz/test_rendering.py` | Views, backend contract, composites (20 tests). |
| `tests/viz/test_search_tree.py` | Search-tree enumerator vs `canonical_string` (14 tests). |
| `docs/figures/isalgraph_card_s2g.png` | Single card. |
| `docs/figures/isalgraph_steps_s2g.png` | S2G multi-column step figure. |
| `docs/figures/isalgraph_steps_g2s.png` | G2S step figure, grey mask inverted. |
| `docs/figures/canonical_search_tree.png` | Reviewer 3's schematic. |

**Modified**

| Path | Change | Reason |
|---|---|---|
| `src/isalgraph/core/string_to_graph.py` | Added `run_with_trace`, `_snapshot`, `_created_edge_for`; trace imports. | Deliverable B. `run()` and `_execute_instruction` untouched. |
| `src/isalgraph/core/graph_to_string.py` | Added `run_with_trace`; two imports. | Deliverable B. `run()` and the greedy search untouched. |
| `benchmarks/plotting_styles.py` | Palettes and IEEE widths now re-export from `isalgraph.viz.style`; fixed `height_ratio: float = None` annotation. | Single source of truth so paper and library figures cannot drift in colour. |
| `benchmarks/real_data/eval_visualizations/cdll_drawing.py` | Replaced with a re-export shim. | Implementation absorbed into `viz.cdll_view`. |
| `benchmarks/real_data/eval_visualizations/graph_drawing.py` | Replaced with a re-export shim. | Implementation absorbed into `viz.nx_view`. |
| `benchmarks/real_data/eval_visualizations/string_alignment.py` | Replaced with a re-export shim. | Implementation absorbed into `viz.alignment_view`. |

**Removed**

None.

**Commits**

| SHA | Message |
|---|---|
| `951311c` | `feat(core): add stdlib-only trace schema and run_with_trace emitters` |
| `a368fdf` | `feat(viz): add isalgraph.viz figure toolkit with matplotlib default backend` |
| `80459e9` | `docs(viz): state that g2s traces replay the string rather than instrument the encoder` |
| (this) | `docs(notes): viz work log` |

`git diff --name-only 2f393a1..HEAD` returns exactly the 37 paths: the 36 listed
above (6 modified + 30 created) plus this log file.

## 4. Tests

**Tests created or extended**

| Test | File | What it verifies | Why it matters |
|---|---|---|---|
| `test_core_has_no_module_scope_third_party_imports` | `tests/unit/test_trace.py` | AST-walks every `core/*.py`; no non-stdlib module-scope import | The `core` dependency-free rule is otherwise unenforced |
| `test_trace_round_trips_through_json` | `tests/unit/test_trace.py` | Hypothesis: `from_json(to_json(t)) == t` | Schema drift between the two directions |
| `test_dump_and_load_are_byte_identical` | `tests/unit/test_trace.py` | Re-dumping a loaded trace gives identical bytes | Non-deterministic serialisation |
| `test_s2g_run_and_run_with_trace_agree` | `tests/unit/test_trace.py` | Hypothesis over 120 strings × directedness | Pins the frozen `run()` behaviour |
| `test_s2g_snapshot_count_and_replay` | `tests/unit/test_trace.py` | `len == n+1`; masks replay to the final graph; monotone growth | Deliverable A/B core contract |
| `test_created_edges_partition_the_final_edge_set` | `tests/unit/test_trace.py` | Each edge attributed exactly once, to exactly one step | The desync the brief warned about |
| `test_noop_connect_records_no_created_edge` | `tests/unit/test_trace.py` | `VnC`: the `C` over an adjacent pair creates nothing | The precise case token-counting gets wrong |
| `test_g2s_run_and_run_with_trace_agree` | `tests/unit/test_trace.py` | Same string; replay isomorphic to input | Deliverable B for the encoder |
| `test_g2s_replay_reproduces_the_encoder_output_graph_exactly` | `tests/unit/test_trace.py` | Replay edge set == `_output_graph` edge set | Backs the exactness claim in the docstring |
| `test_viz_imports_without_any_drawing_library` | `tests/viz/test_import_without_matplotlib.py` | Meta-path finder blocks 6 libraries; all 13 viz modules import | Definition of done #1 |
| `test_available_backends_is_empty_without_matplotlib` | same | No backend claims availability when the library is gone | The upstream defect `is_available()` fixes |
| `test_no_module_scope_drawing_imports` | same | AST-walks every `viz/*.py` | Catches a regression the runtime test could miss |
| `test_threading_the_layout_pins_every_node` | `tests/viz/test_rendering.py` | Two panels, threaded layout, identical coordinates per node | Definition of done #5 |
| `test_backend_never_creates_a_figure` | same | `plt.get_fignums()` unchanged across a draw | Backend contract clause 1 |
| `test_grayed_elements_are_caller_decided` | same | Grey mask does not perturb the returned layout | Backend contract clause 3 |
| `test_grey_masks_invert_between_the_two_directions` | same | S2G starts ghosted/ends solid; G2S the reverse | Definition of done #6 |
| `test_save_figure_keeps_dots_in_the_stem` | same | `run_1.5` → `run_1.5.png` | The `with_suffix` bug |
| `test_enumerator_agrees_with_canonical_string` | `tests/viz/test_search_tree.py` | Replay enumeration == `canonical_string` on 6 graph families | The schematic cannot drift from the algorithm |
| `test_branch_points_are_only_starting_nodes_and_vv_choices` | same | No fan-out on `C`/`c` or single-candidate `V`/`v` | The figure's central claim |
| `test_exactly_one_canonical_path_is_marked` | same | One marked node per depth | A real bug I found by looking at the render |
| `test_node_palette_is_total_and_grey_free` | `tests/viz/test_style_and_layout.py` | No palette entry equals `GRAYED_FACE` | The ghost/active collision |
| `test_case_selects_the_pointer_accent` | same | Uppercase → `POINTER_PALETTE[0]`, lowercase → `[1]` | The pointer channel |

**Coverage of the behaviour that matters:** Both converters are exercised by
Hypothesis over random strings and both directedness settings, including the
degenerate empty string and single-node graphs. All three backends are tested
against the layout-in/layout-out contract, with `importorskip` for the optional
two. The search-tree enumerator is checked against the real algorithm on paths,
cycles, a star, and the 7-node worked example.

**Not tested, and why:**
- Pixel content of the rendered PNGs. Deliberate: image hashes break on every
  freetype/matplotlib release. I substituted a byte-size floor plus manual
  visual inspection of all four figures (§6).
- `nx_view.draw_nx_graph` and `alignment_view` are absorbed verbatim and are
  covered only by the import shims, not by rendering tests. They were untested
  before this change too, so this is not a regression, but it is a gap.
- The `igraph` backend's non-default layouts (`sugiyama`, `fr`) — only `kk` is
  exercised.
- `_dim_grayed` in the networkx backend inspects `ax.collections` by length
  matching; if networkx changes how many collections it adds, ghosting could
  silently stop applying. The test checks the layout contract, not the alpha.

## 5. Test results

**Command:** `PYTHONPATH=$PWD/src MPLBACKEND=Agg ~/.conda/envs/isalgraph/bin/python -m pytest tests/ -q`

```
======================= 560 passed, 271 skipped in 6.44s =======================
```

New tests alone (`tests/viz tests/unit/test_trace.py`): `110 passed in 2.04s`.

**Result:** 560 passed, 0 failed, 271 skipped · **Duration:** 6.44 s · **Run at:** `a368fdf`

Baseline was 450 passed / 271 skipped / 0 failed, so +110 tests, no regressions
and no change in the skip count.

`mypy --strict src/isalgraph/`: `Success: no issues found in 41 source files`.

`ruff check src/ tests/ benchmarks/`: 80 errors, **all pre-existing at
`BASE_SHA`** and all in `benchmarks/` files outside my ownership. Verified by
running the same command on the main checkout, which reports the same 80. My
files contribute zero: `ruff check src/ tests/` is clean. See §7.

**Failures and their resolution:**

1. `test_noop_connect_records_no_created_edge` failed on first run, asserting
   `created_edge is None` where the code produced `(0, 0)`. **The test was
   wrong, not the code.** `V` moves neither pointer, so after `VC` both
   pointers still rest on node 0 and the `C` legitimately creates a self-loop.
   Rewrote the test to use `VnC` (walk the secondary onto node 1 first, so the
   `C` targets an already-adjacent pair) and added a second test documenting
   the self-loop case. Both pass.
2. 11 mypy errors and 4 ruff errors on first lint. Causes: `RcParams` is typed
   over ~300 literal keys and rejects `dict[str, Any]` (fixed with a cast);
   `min(key=...)` returning `int | None` (fixed by extracting a typed helper);
   `np.ndarray` without type args in the absorbed `nx_view` (annotations
   loosened, since numpy is `Any` at runtime there); `Line2D` in a
   `list[Patch]`; `plt.Circle` is not an explicit re-export. All fixed; both
   tools now clean on the code I own.
3. During the mypy fix the formatter hook silently dropped a function-local
   `from matplotlib.patches import Circle`, producing an `F821`. Caught by the
   next ruff run and re-added.

## 6. Verification beyond unit tests

| Circumstance | What was run | Evidence | Outcome |
|---|---|---|---|
| Randomised differential check of the trace emitters (before building anything on them) | Ad-hoc script: 400 random strings × 2 directedness for S2G; 200 graphs for G2S | S2G: graph equality, `len(snapshots)==n+1`, monotone masks, `created_edge` partition all hold. G2S: 170 graphs passed (30 skipped as disconnected) | pass |
| Exactness of the G2S replay | 200 random graphs, comparing replayed edge set to `GraphToString._output_graph` | 175 comparable graphs, exact edge-set equality in every case | pass — backs the docstring claim |
| Visual inspection of every committed figure | Opened all four PNGs and read them | Found and fixed 5 real defects: `Circle` rendered as ellipses under unequal aspect; 7 root subtrees too dense; canonical-path highlight indistinguishable from the `V`-branch red; suptitle colliding with the axes title; **two sibling branches both flagged canonical** (prefix ties) | pass after fixes |
| Grey/active collision | Inspected `isalgraph_steps_s2g.png` at step 10 | Node 4 rendered in a `tab20c` grey, visually identical to a ghosted node | fixed: palettes now exclude grey; re-rendered and re-inspected |
| Downstream import compatibility | Imported all 25 modules under `eval_visualizations` that previously imported from the four absorbed files | 25/25 importable | pass |
| Palette parity | Asserted `INSTRUCTION_COLORS` equals a frozen literal copy kept in `plotting_styles.py`, plus `PAUL_TOL_*`, IEEE widths, container types | byte-identical | pass |
| Dependency-free import | Meta-path finder blocking matplotlib, networkx, igraph, numpy, scipy, pandas | all 13 viz modules import; `available_backends() == ()` | pass |
| Figure reproducibility | `python -m isalgraph.viz` run 3× | 4 figures each time, no seeds involved (fixed string + fixed edge list) | pass |
| Environment | Debian 12, Python 3.11.15, matplotlib 3.11.1, networkx 3.6.1, igraph 1.0.0, numpy 1.26.4, pytest 9.1.1, ruff 0.16.2, mypy 2.3.0 | | |

## 7. Decisions, assumptions, open questions

**Decisions with a real trade-off:**

- **G2S trace by replay rather than in-loop emission.** Costs one extra S2G pass
  over the emitted string (negligible) and means the G2S snapshots are
  *S2G-semantics* states. Buys: `run()` and the greedy search are untouched, so
  `wave-cpp-engine`'s parity target is safe, and the trace has one state per
  instruction, which the encoder itself cannot produce because it emits whole
  displacement groups per iteration. Justified by the exactness check in §6.

  The replay is exact rather than approximate for a structural reason worth
  recording: `_emit_primary_moves(a)` appends exactly `|a|` characters while
  `_move_pointer(ptr, a)` walks exactly `|a|` slots, so the interpreter's
  pointer trajectory over the emitted run coincides with the encoder's jump to
  the tentative slot. Node ids agree because both allocate contiguously in
  insertion order. Verified on 175 graphs by edge-set equality, not isomorphism.

  **The cost this imposes on a reader, and how it is mitigated.** A `"g2s"`
  step figure shows an interpreter executing the finished string, *not* the
  encoder searching. The encoder's real search is strictly richer — tentative
  pointer positions it walks and abandons, displacement pairs rejected because
  no operation applies, the `V ≻ v ≻ C ≻ c` cascade at each pair — and none of
  it appears in the trace. Someone reading a G2S figure could easily believe
  they are watching the search. For a paper figure that ambiguity is
  unacceptable, so it is stated explicitly in three places: a `.. warning::`
  block in `GraphToString.run_with_trace`'s docstring, a dedicated subsection
  of `viz/README.md`, and here. Readers who want the decision structure are
  pointed at `canonical_search_tree_figure`, which is precisely what that
  figure draws. (Added in `80459e9` at `main`'s request; the original docstring
  described the replay mechanism but did not warn about the misreading.)

- **Colour: two orthogonal channels, not case-as-hue.** The brief asked that
  case encode the pointer via `POINTER_PALETTE[0]/[1]`. That is not compatible
  with preserving the published palette: `INSTRUCTION_COLORS["N"]` is `#4477AA`,
  which *is* the secondary pointer's colour, so hue cannot carry both channels
  without recolouring figures already in the paper. I kept hue = operation
  (palette byte-identical) and added a stroke accent on every instruction cell
  = acting pointer, in the same two colours as the ring arrows. Costs: an extra
  visual channel a reader must learn. Buys: mechanical correspondence *and* an
  unchanged paper identity. Messaged to `main`.

- **Ring nodes coloured by pointer, graph nodes by identity.** Inconsistent on
  its face, but each panel encodes what matters in it. The pointer channel is
  consistent across both. Documented in the README.

- **Search-tree enumeration by replay, not backtracking.** Costs asymptotically
  more work; irrelevant at 6-8 nodes. Buys: no dependency on `canonical.py`'s
  private undo helpers, which I may not edit and which could change.

- **`max_roots=3` default on the schematic.** The true search roots at all 7
  nodes; showing all 7 produced an unreadable wall. The canonical root is always
  retained. The figure is a schematic, not an exhaustive enumeration, and the
  caption says "starting node" is a branch axis.

**Assumptions I proceeded on:**

- CDLL slot 0 is always live in both converters, so anchoring the ring order
  there gives a stable rotation across frames. True because neither converter
  calls `remove` (only `canonical.py` does, during backtracking). If a future
  converter removes CDLL nodes, `cdll_forward_order`'s default anchor breaks;
  it takes an explicit `anchor` argument for that reason.
- `docs/figures/` is mine to create and no other agent writes there.

**Open questions for the orchestrator:**

- ~~The 80 pre-existing ruff errors in `benchmarks/` block definition-of-done #8~~
  **Resolved.** `main` withdrew the criterion: the errors predate the wave, sit
  in files assigned to nobody, and `main` will add the rules to the existing
  `benchmarks/**/*.py` per-file-ignores at integration, matching the E402/ANN
  convention already in `pyproject.toml`. Instructed not to fix them; I did not.
- `main` owns the two `__init__.py` files; `isalgraph.viz` is not yet wired into
  the public API. `import isalgraph.viz` works regardless.

## 8. Coordination

**Messages sent:** One to `main` at the end, covering the G2S replay decision
(with its exactness evidence), the colour-semantics deviation and why the
literal instruction was not followable, and the pre-existing ruff situation.
No reply required to proceed; all three were recorded decisions, not blockers.

**Messages received and how they changed the work:** One from `main`, a RETURN
verdict on a single missing obligation — this log file, which `main` found the
notes directory empty of. Three substantive effects:

1. Wrote and committed this log. That was the whole verdict; the work itself was
   independently re-verified by `main` and accepted.
2. `main` accepted the G2S replay decision but required that the
   replay-vs-instrumentation distinction be stated plainly in the
   `run_with_trace` docstring and in `viz/README.md`, on the grounds that a
   reader of a G2S figure could otherwise believe they are seeing the encoder's
   search states. The original docstring described the replay mechanism but did
   not warn against that misreading, so this was a genuine gap rather than
   something already covered. Fixed in `80459e9`; recorded in §7.
3. `main` withdrew the "ruff clean on `benchmarks/`" criterion as its own
   error and instructed me not to fix the 80 pre-existing errors. Recorded in
   §7 and §11.

`main` also confirmed the colour deviation was an improvement on the brief,
having not checked the published palette when writing it.

**Contracts I depend on and confirmed unchanged:**
- `isalgraph.errors` already exports `VizError`, `VizBackendNotFoundError`,
  `VizBackendUnavailableError`. I used the latter two and added none.
- I did not import `isalgraph.core.backends` or `_native`, so nothing here
  depends on `wave-cpp-engine`'s branch.
- `SparseGraph`, `CircularDoublyLinkedList`, `canonical.py` read-only and
  unmodified.

## 9. Deliberately not done

- **`cohort_panel.py`** — the brief marked it "port if time allows". Skipped:
  it is a cohort-grid convenience wrapper with no consumer in this repo, and
  porting it would have meant guessing at the cohort data shape. The retuning
  notes in the brief (xlim `1.2`, `axis_margin` → 0.12) are recorded here for
  whoever picks it up; I did apply the `axis_margin = 0.12` value to the
  matplotlib backend, where straight edges genuinely need less slack than
  HyperGraphX bezier hulls.
- **Wiring `isalgraph.viz` into `src/isalgraph/__init__.py`** — `main` owns it.
- **Fixing the 80 pre-existing ruff errors** — outside my ownership.
- **PDF/SVG output of the committed figures** — only PNG was required; the
  builders accept a `formats` tuple, so `python -m isalgraph.viz --formats pdf,png`
  produces them on demand.
- **Rendering tests for `nx_view` / `alignment_view`** — absorbed verbatim and
  previously untested; noted as a gap in §4 rather than silently covered.

## 10. Risks and follow-ups

| Item | Severity | Detail | Suggested owner |
|---|---|---|---|
| `_dim_grayed` length-matches `ax.collections` | medium | If networkx changes how many collections `draw_networkx_*` adds, ghosting silently stops applying in the networkx backend. The matplotlib default backend is unaffected. | next wave |
| Pre-existing ruff errors in `benchmarks/` | low | 80 errors, all at baseline, all outside my ownership. Blocks a literal reading of DoD #8. | orchestrator |
| `nx_view` / `alignment_view` lack rendering tests | low | Absorbed verbatim; no regression vs. baseline, but now they are library code and should be covered. | next wave |
| Search-tree schematic shows 3 of 7 roots | low | A reader could infer the search roots at only 3 nodes. Caption and README both say otherwise, but a reviewer might still ask. | user, at figure-caption time |
| `benchmarks/plotting_styles.py` now imports `isalgraph.viz.style` | low | Adds an import edge from `benchmarks` to the library. Correct direction per the layering rules, but it means the benchmark scripts now need `isalgraph` importable — which they already did. | — |

## 11. Self-assessment against the definition of done

| # | Criterion | Met | Evidence |
|---|---|---|---|
| 1 | `import isalgraph.viz` with matplotlib uninstalled | yes | `test_viz_imports_without_any_drawing_library` + `test_blocking_finder_actually_blocks` (guards the guard) + AST test over all 13 modules |
| 2 | `core.trace` stdlib-only; byte-identical round-trip; no third-party module-scope import in `core/` | yes | `test_dump_and_load_are_byte_identical`, `test_core_has_no_module_scope_third_party_imports` (parametrised over every `core/*.py`) |
| 3 | `run()` and `run_with_trace()` agree, both converters, property test | yes | `test_s2g_run_and_run_with_trace_agree` (Hypothesis, 120 examples × 2), `test_g2s_run_and_run_with_trace_agree` (60 examples) |
| 4 | `len(snapshots) == n+1`; masks reconstruct the final graph | yes | `test_s2g_snapshot_count_and_replay` |
| 5 | matplotlib default + ≥1 optional backend, layout contract, pinning verified | yes | 3 backends; `test_threading_the_layout_pins_every_node` asserts exact per-node coordinate equality |
| 6 | Card, S2G steps, G2S steps with inverted masks, search tree — rendered and inspected | yes | 4 PNGs in `docs/figures/`; all four opened and read; 5 defects found and fixed that way; `test_grey_masks_invert_between_the_two_directions` |
| 7 | Shims so existing scripts import | yes | 25/25 `eval_visualizations` modules import; palette parity asserted |
| 8 | ruff clean on `src/`/`tests/`; mypy strict; suite green | yes | mypy clean (41 files); suite 560/0/271; ruff clean on `src/` and `tests/`. The `benchmarks/` half of this criterion was **withdrawn by `main`** as its own error: the 80 errors predate the wave (80 at `2f393a1`, 80 here — zero introduced) and `main` will handle them via per-file-ignores at integration |
| 9 | `viz/README.md` with API, contract, colour semantics, hello-world | yes | `src/isalgraph/viz/README.md` |

**Overall:** I am confident in the trace layer and the backend contract — both
are pinned by property tests and by a differential check against the frozen
`run()` on 600+ randomised inputs, and the exactness claim I put in the G2S
docstring is empirically backed rather than asserted. I am confident the figures
render, because I looked at every one and the first drafts of two of them were
wrong in ways only inspection catches. With criterion #8's `benchmarks/` clause
withdrawn by `main`, every criterion is met.

**The single thing `main` should scrutinise first** is the colour-semantics
deviation in §7: I did not follow the brief literally, because following it
would have recoloured a palette already published in the paper's figures. If
`main` disagrees, the change is localised to `INSTRUCTION_PALETTE` and
`pointer_accent` in `src/isalgraph/viz/style.py` and the `accent_lw` stroke in
`instruction_view.py`. Second priority: the `_mark_optimal` tie-breaking in
`search_tree.py`, where several sibling branches genuinely realise the same
canonical string (a real property of the labelling-independent encoding) and I
chose to display only one.
