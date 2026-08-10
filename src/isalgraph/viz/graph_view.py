"""Graph drawing dispatch.

Thin wrapper over :func:`isalgraph.viz.registry.get_backend`: resolves a
backend name to an instance and forwards the draw call, so callers never
touch the registry directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isalgraph.core.sparse_graph import SparseGraph
from isalgraph.core.trace import Edge
from isalgraph.types import NodeId
from isalgraph.viz.base import Position
from isalgraph.viz.registry import DEFAULT_BACKEND, get_backend

if TYPE_CHECKING:
    from matplotlib.axes import Axes
else:
    Axes = Any


def draw_graph(
    ax: Axes,
    graph: SparseGraph,
    *,
    backend: str = DEFAULT_BACKEND,
    node_colors: dict[NodeId, str] | None = None,
    edge_colors: dict[Edge, str] | None = None,
    grayed_nodes: frozenset[NodeId] = frozenset(),
    grayed_edges: frozenset[Edge] = frozenset(),
    layout: dict[NodeId, Position] | None = None,
) -> dict[NodeId, Position]:
    """Draw *graph* on *ax* via the named backend and return the layout used.

    Args:
        ax: Target matplotlib axes.
        graph: The graph to draw.
        backend: Registry key; defaults to ``"matplotlib"``.
        node_colors: Per-node colour. Empty mapping means backend defaults.
        edge_colors: Per-edge colour keyed by normalised ``(u, v)``.
        grayed_nodes: Nodes to ghost.
        grayed_edges: Edges to ghost.
        layout: Pinned coordinates; computed by the backend when ``None``.

    Returns:
        The layout actually used, for pinning into the next panel.
    """
    return get_backend(backend).draw(
        graph,
        ax,
        node_colors=node_colors or {},
        edge_colors=edge_colors or {},
        grayed_nodes=grayed_nodes,
        grayed_edges=grayed_edges,
        layout=layout,
    )


__all__ = ["draw_graph"]
