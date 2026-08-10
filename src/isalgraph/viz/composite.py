"""Composite figure builders.

The atomic unit is a three-row column showing one snapshot's
``(CDLL ring, instruction strip, graph)``. Higher-level figures stack
columns horizontally (:func:`steps_figure`) or stack two direction blocks
vertically (:func:`roundtrip_figure`).

Two rules keep multi-panel figures readable:

* **Palettes are built once per figure, from the final graph**, never
  per column, so a node keeps its colour across every frame.
* **The layout is pinned by threading it forward**: each column receives
  the layout the previous column returned, so nodes never jump.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isalgraph.core.sparse_graph import SparseGraph
from isalgraph.core.trace import AlgorithmTrace, Edge, StepSnapshot, graph_edges
from isalgraph.types import NodeId
from isalgraph.viz.base import Position
from isalgraph.viz.cdll_view import draw_cdll_ring_for_snapshot
from isalgraph.viz.graph_view import draw_graph
from isalgraph.viz.instruction_view import draw_instruction_strip
from isalgraph.viz.registry import DEFAULT_BACKEND
from isalgraph.viz.style import build_edge_palette, build_node_palette
from isalgraph.viz.trace_io import graph_from_edgelist

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.gridspec import SubplotSpec
else:
    Figure = Any
    SubplotSpec = Any


#: Relative heights of the (ring, strip, graph) rows within a column.
_ROW_RATIOS: tuple[float, float, float] = (1.9, 1.0, 1.6)


def _column_axes(fig: Figure, spec: SubplotSpec) -> tuple[Any, Any, Any]:
    """Carve a three-row sub-gridspec out of *spec* and return its axes."""
    sub = spec.subgridspec(3, 1, height_ratios=list(_ROW_RATIOS), hspace=0.05)
    return fig.add_subplot(sub[0]), fig.add_subplot(sub[1]), fig.add_subplot(sub[2])


def _grey_masks(
    snapshot: StepSnapshot,
    full_graph: SparseGraph,
    direction: str,
) -> tuple[frozenset[NodeId], frozenset[Edge]]:
    """Return the ``(grayed_nodes, grayed_edges)`` pair for one column.

    The two directions invert the mask:

    * ``"s2g"`` -- the algorithm *builds* the graph from the string, so
      elements already built are in colour and the rest are ghosts of the
      target. The panel starts fully grey and ends fully coloured.
    * ``"g2s"`` -- the algorithm *consumes* the graph and emits the
      string, so elements already encoded are ghosts and what remains in
      colour is what it has yet to capture. The panel starts fully
      coloured and ends fully grey.
    """
    active_nodes = set(snapshot.active_nodes)
    active_edges = set(snapshot.active_edges)
    if direction == "g2s":
        return frozenset(active_nodes), frozenset(active_edges)
    all_nodes = range(full_graph.node_count())
    return (
        frozenset(v for v in all_nodes if v not in active_nodes),
        frozenset(e for e in graph_edges(full_graph) if e not in active_edges),
    )


def draw_column(
    fig: Figure,
    spec: SubplotSpec,
    snapshot: StepSnapshot,
    full_graph: SparseGraph,
    instructions: str,
    *,
    backend: str = DEFAULT_BACKEND,
    node_palette: dict[NodeId, str],
    edge_palette: dict[Edge, str],
    graph_layout: dict[NodeId, Position] | None,
    column_title: str | None = None,
    direction: str = "s2g",
    column_inches: float | None = None,
    strip_current_idx: int | None = None,
) -> dict[NodeId, Position]:
    """Draw one snapshot column into *fig* at *spec*.

    Args:
        fig: Target figure.
        spec: The ``SubplotSpec`` this column occupies.
        snapshot: The step to render.
        full_graph: The final graph, drawn in every column with a
            direction-dependent grey mask.
        instructions: The complete instruction string.
        backend: Graph drawing backend name.
        node_palette: Figure-wide node colours.
        edge_palette: Figure-wide edge colours.
        graph_layout: Pinned layout from the previous column, or ``None``.
        column_title: Title placed above the CDLL ring.
        direction: ``"s2g"`` or ``"g2s"``; see :func:`_grey_masks`.
        column_inches: Physical column width, for strip font sizing.
        strip_current_idx: Override the strip's progress index. Under
            ``"s2g"`` the strip greys out as instructions are spent, so a
            standalone final-state card would show an entirely grey
            string; passing ``0`` there colours every cell.

    Returns:
        The layout used, to thread into the next column.
    """
    ax_cdll, ax_strip, ax_graph = _column_axes(fig, spec)

    if column_title is not None:
        ax_cdll.set_title(column_title, fontsize=8, pad=3)

    draw_cdll_ring_for_snapshot(ax_cdll, snapshot)
    draw_instruction_strip(
        ax_strip,
        instructions,
        current_idx=snapshot.step_idx if strip_current_idx is None else strip_current_idx,
        axis_width_inches=column_inches,
        direction=direction,
    )

    grayed_nodes, grayed_edges = _grey_masks(snapshot, full_graph, direction)
    return draw_graph(
        ax_graph,
        full_graph,
        backend=backend,
        node_colors=node_palette,
        edge_colors=edge_palette,
        grayed_nodes=grayed_nodes,
        grayed_edges=grayed_edges,
        layout=graph_layout,
    )


def _sample_indices(n: int, k: int) -> list[int]:
    """Return up to *k* evenly spaced indices in ``[0, n)``, endpoints included."""
    if n <= 0:
        return []
    if n <= k or k <= 1:
        return list(range(n))
    step = (n - 1) / (k - 1)
    return sorted({round(i * step) for i in range(k)})


def _palettes(
    full_graph: SparseGraph,
) -> tuple[dict[NodeId, str], dict[Edge, str]]:
    """Build the figure-wide node and edge palettes from the final graph."""
    return (
        build_node_palette(full_graph.node_count()),
        build_edge_palette(graph_edges(full_graph)),
    )


def single_card_figure(
    trace: AlgorithmTrace,
    *,
    step_idx: int = -1,
    backend: str = DEFAULT_BACKEND,
    title: str | None = None,
    full_graph: SparseGraph | None = None,
    figsize: tuple[float, float] = (3.2, 6.0),
    color_whole_string: bool = True,
) -> Figure:
    """Build a one-column card: CDLL ring, instruction strip, graph.

    Args:
        trace: The trace to draw from.
        step_idx: Index into ``trace.snapshots``; ``-1`` is the final state.
        backend: Graph drawing backend name.
        title: Title above the ring.
        full_graph: Final graph; rebuilt from the trace envelope if absent.
        figsize: Figure size in inches.
        color_whole_string: Render every instruction cell in colour rather
            than applying the direction's progress mask. A card shows one
            state next to the whole string, so the progress mask has
            nothing to convey and would only grey the string out.

    Returns:
        The created figure. The caller owns it and must close it.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    graph = full_graph if full_graph is not None else graph_from_edgelist(trace.final_graph)
    node_palette, edge_palette = _palettes(graph)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 1, figure=fig)
    draw_column(
        fig,
        gs[0, 0],
        trace.snapshots[step_idx],
        graph,
        trace.instruction_string,
        backend=backend,
        node_palette=node_palette,
        edge_palette=edge_palette,
        graph_layout=None,
        column_title=title,
        direction=trace.direction,
        column_inches=figsize[0] * 0.9,
        strip_current_idx=_full_color_index(trace.direction, trace.instruction_string)
        if color_whole_string
        else None,
    )
    return fig


def _full_color_index(direction: str, instructions: str) -> int:
    """Return the ``current_idx`` that colours every cell for *direction*."""
    return len(instructions) if direction == "g2s" else 0


def steps_figure(
    trace: AlgorithmTrace,
    *,
    backend: str = DEFAULT_BACKEND,
    n_columns: int = 7,
    full_graph: SparseGraph | None = None,
    overall_title: str | None = None,
    column_inches: float = 2.4,
    column_height: float = 6.0,
) -> Figure:
    """Build a ``3 x n_columns`` step figure sampling *trace* evenly.

    Args:
        trace: The trace to draw.
        backend: Graph drawing backend name.
        n_columns: Maximum number of columns; all snapshots are used when
            the trace is shorter.
        full_graph: Final graph; rebuilt from the trace envelope if absent.
        overall_title: Figure suptitle.
        column_inches: Width per column, in inches.
        column_height: Figure height, in inches.

    Returns:
        The created figure. The caller owns it and must close it.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    graph = full_graph if full_graph is not None else graph_from_edgelist(trace.final_graph)
    node_palette, edge_palette = _palettes(graph)

    indices = _sample_indices(len(trace.snapshots), n_columns)
    ncols = max(len(indices), 1)
    fig = plt.figure(figsize=(column_inches * ncols, column_height))
    gs = GridSpec(1, ncols, figure=fig, wspace=0.10)

    pinned: dict[NodeId, Position] | None = None
    for col, idx in enumerate(indices):
        snap = trace.snapshots[idx]
        pinned = draw_column(
            fig,
            gs[0, col],
            snap,
            graph,
            trace.instruction_string,
            backend=backend,
            node_palette=node_palette,
            edge_palette=edge_palette,
            graph_layout=pinned,
            column_title=f"Step {snap.step_idx}",
            direction=trace.direction,
            column_inches=column_inches,
        )
    if overall_title is not None:
        fig.suptitle(overall_title, fontsize=10)
    return fig


def roundtrip_figure(
    g2s_trace: AlgorithmTrace,
    s2g_trace: AlgorithmTrace,
    *,
    backend: str = DEFAULT_BACKEND,
    n_columns: int = 7,
    full_graph: SparseGraph | None = None,
    overall_title: str | None = None,
    column_inches: float = 2.4,
) -> Figure:
    """Build the two-block round-trip collage.

    The top block is the ``"g2s"`` trace (the algorithm producing the
    string) and the bottom block the ``"s2g"`` trace (consuming it). The
    same column indices are sampled from both so the blocks line up, and
    the grey masks run in opposite directions between them.

    Args:
        g2s_trace: Encoding trace, drawn on top.
        s2g_trace: Decoding trace, drawn below.
        backend: Graph drawing backend name.
        n_columns: Maximum number of columns.
        full_graph: Final graph; rebuilt from *s2g_trace* if absent.
        overall_title: Figure suptitle.
        column_inches: Width per column, in inches.

    Returns:
        The created figure. The caller owns it and must close it.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    graph = full_graph if full_graph is not None else graph_from_edgelist(s2g_trace.final_graph)
    node_palette, edge_palette = _palettes(graph)

    n = min(len(g2s_trace.snapshots), len(s2g_trace.snapshots))
    indices = _sample_indices(n, n_columns)
    ncols = max(len(indices), 1)
    fig = plt.figure(figsize=(column_inches * ncols, 12.0))
    gs = GridSpec(2, ncols, figure=fig, wspace=0.10, hspace=0.15)

    pinned_top: dict[NodeId, Position] | None = None
    pinned_bottom: dict[NodeId, Position] | None = None
    for col, idx in enumerate(indices):
        for row, (tr, pinned) in enumerate(((g2s_trace, pinned_top), (s2g_trace, pinned_bottom))):
            snap = tr.snapshots[idx]
            used = draw_column(
                fig,
                gs[row, col],
                snap,
                graph,
                tr.instruction_string,
                backend=backend,
                node_palette=node_palette,
                edge_palette=edge_palette,
                graph_layout=pinned,
                column_title=f"Step {snap.step_idx}",
                direction=tr.direction,
                column_inches=column_inches,
            )
            if row == 0:
                pinned_top = used
            else:
                pinned_bottom = used

    if overall_title is not None:
        fig.suptitle(overall_title, fontsize=10)
    return fig


__all__ = [
    "draw_column",
    "roundtrip_figure",
    "single_card_figure",
    "steps_figure",
]
