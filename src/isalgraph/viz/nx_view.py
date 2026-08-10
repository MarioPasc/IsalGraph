# ruff: noqa: N803
"""NetworkX-flavoured graph drawing, absorbed from the benchmark scripts.

The published figure scripts draw ``networkx.Graph`` objects rather than
:class:`~isalgraph.core.sparse_graph.SparseGraph`, with a spring layout
and an optional canonical string rendered underneath. That entry point is
preserved here verbatim in behaviour so those scripts keep working; the
library-facing equivalent, which takes a ``SparseGraph`` and dispatches
through the backend registry, is
:func:`isalgraph.viz.graph_view.draw_graph`.

The original lived at
``benchmarks/real_data/eval_visualizations/graph_drawing.py`` and imported
matplotlib, networkx and numpy at module scope. Here those imports move
into the function body so ``import isalgraph.viz`` stays dependency-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isalgraph.viz.style import PAUL_TOL_MUTED, render_colored_string

if TYPE_CHECKING:
    import networkx as nx
    import numpy as np
    from matplotlib.axes import Axes
else:
    nx = Any
    np = Any
    Axes = Any

# Default node color from Paul Tol muted palette (indigo)
_DEFAULT_NODE_COLOR = PAUL_TOL_MUTED[1]


def draw_nx_graph(
    G: nx.Graph,
    ax: Axes,
    *,
    title: str | None = None,
    canonical_string: str | None = None,
    highlight_edges: list[tuple[int, int]] | None = None,
    highlight_color: str = "#EE6677",
    node_color: str | None = None,
    edge_color: str = "0.5",
    node_size: int = 200,
    layout_seed: int = 42,
    show_labels: bool = True,
    pos: dict[Any, Any] | None = None,
) -> dict[Any, Any]:
    """Draw a graph on the given axes with publication styling.

    Args:
        G: NetworkX graph to draw.
        ax: Matplotlib axes.
        title: Optional title above the graph.
        canonical_string: Optional IsalGraph string to render below.
        highlight_edges: Edges to draw with highlight styling.
        highlight_color: Color for highlighted edges.
        node_color: Node fill color (default: Paul Tol indigo).
        edge_color: Normal edge color.
        node_size: Node circle size in points^2.
        layout_seed: Seed for spring_layout reproducibility.
        show_labels: Whether to show node ID labels.
        pos: Pre-computed layout positions (overrides spring_layout).

    Returns:
        Position dict for layout reuse across multiple calls.
    """
    import networkx as nx  # noqa: PLC0415

    nc = node_color or _DEFAULT_NODE_COLOR

    if pos is None:
        pos = nx.spring_layout(G, seed=layout_seed)

    # Draw normal edges
    normal_edges = list(G.edges())
    highlight_set = set()
    if highlight_edges:
        highlight_set = {(min(u, v), max(u, v)) for u, v in highlight_edges}
        normal_edges = [(u, v) for u, v in G.edges() if (min(u, v), max(u, v)) not in highlight_set]

    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, ax=ax, edge_color=edge_color, width=0.8)

    if highlight_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=highlight_edges,
            ax=ax,
            edge_color=highlight_color,
            width=1.5,
            style="dashed",
        )

    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=nc, node_size=node_size, edgecolors="0.3", linewidths=0.5
    )

    if show_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color="white")

    if title:
        ax.set_title(title, fontsize=9)

    if canonical_string:
        # Render colored string below the graph
        x_center = sum(p[0] for p in pos.values()) / max(len(pos), 1)
        y_min = min(p[1] for p in pos.values())
        render_colored_string(
            ax, canonical_string, x=x_center, y=y_min - 0.25, fontsize=7, mono=True
        )

    # Pad axes limits so node circles are never clipped.
    # Node positions define data range, but circles extend beyond in points.
    # A 15% margin on the data span is sufficient for typical node sizes.
    if pos:
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 0.1)
        pad = 0.20 * span
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)

    ax.set_aspect("equal")
    ax.axis("off")

    return pos
