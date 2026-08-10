"""Optional backend built on ``networkx.draw_networkx_*``.

Worth having alongside the default because NetworkX offers layout
algorithms the raw backend does not (Kamada-Kawai, shell, spectral) and
handles curved multi-edges. It obeys the same layout-in / layout-out
contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isalgraph.core.sparse_graph import SparseGraph
from isalgraph.core.trace import Edge, graph_edges
from isalgraph.types import NodeId
from isalgraph.viz.base import GraphVizBackend, Position
from isalgraph.viz.layout import compact_graph_layout
from isalgraph.viz.registry import register_backend
from isalgraph.viz.style import (
    ACTIVE_ALPHA,
    DEFAULT_NODE_COLOR,
    GRAYED_ALPHA,
    GRAYED_EDGE,
    GRAYED_FACE,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
else:
    Axes = Any


class NetworkXBackend(GraphVizBackend):
    """Draw a :class:`SparseGraph` via NetworkX's matplotlib helpers."""

    def __init__(self, *, node_size: int = 180, show_labels: bool = True) -> None:
        self._node_size = node_size
        self._show_labels = show_labels

    @property
    def name(self) -> str:
        return "networkx"

    @classmethod
    def is_available(cls) -> bool:
        """Report whether both networkx and matplotlib import."""
        try:
            import matplotlib  # noqa: F401
            import networkx  # noqa: F401
        except ImportError:
            return False
        return True

    def draw(
        self,
        graph: SparseGraph,
        ax: Axes,
        *,
        node_colors: dict[NodeId, str],
        edge_colors: dict[Edge, str],
        grayed_nodes: frozenset[NodeId] = frozenset(),
        grayed_edges: frozenset[Edge] = frozenset(),
        layout: dict[NodeId, Position] | None = None,
    ) -> dict[NodeId, Position]:
        """Draw *graph* on *ax*; see :meth:`GraphVizBackend.draw`."""
        import networkx as nx

        g: Any = nx.DiGraph() if graph.directed() else nx.Graph()
        g.add_nodes_from(range(graph.node_count()))
        edges = [e for e in graph_edges(graph) if e[0] != e[1]]
        g.add_edges_from(edges)

        pos = layout if layout is not None else compact_graph_layout(graph)
        nodes = list(range(graph.node_count()))

        nx.draw_networkx_nodes(
            g,
            pos,
            ax=ax,
            nodelist=nodes,
            node_size=self._node_size,
            node_color=[
                GRAYED_FACE if v in grayed_nodes else node_colors.get(v, DEFAULT_NODE_COLOR)
                for v in nodes
            ],
            edgecolors="0.25",
            linewidths=0.8,
            alpha=None,
        )
        if edges:
            nx.draw_networkx_edges(
                g,
                pos,
                ax=ax,
                edgelist=edges,
                edge_color=[
                    GRAYED_EDGE if e in grayed_edges else edge_colors.get(e, "0.5") for e in edges
                ],
                width=1.2,
                alpha=ACTIVE_ALPHA,
                arrows=graph.directed(),
                node_size=self._node_size,
            )
        if self._show_labels:
            nx.draw_networkx_labels(
                g,
                pos,
                ax=ax,
                labels={v: str(v) for v in nodes},
                font_size=7,
            )

        # NetworkX has no per-element alpha in these calls, so ghosting is
        # applied afterwards by dimming the collections it just added.
        _dim_grayed(ax, nodes, grayed_nodes, edges, grayed_edges)

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return dict(pos)


def _dim_grayed(
    ax: Axes,
    nodes: list[NodeId],
    grayed_nodes: frozenset[NodeId],
    edges: list[Edge],
    grayed_edges: frozenset[Edge],
) -> None:
    """Apply per-element alpha to the node and edge collections on *ax*."""
    from matplotlib.collections import LineCollection, PathCollection

    for coll in ax.collections:
        if isinstance(coll, PathCollection):
            offsets = coll.get_offsets()
            if len(offsets) == len(nodes):  # type: ignore[arg-type]
                coll.set_alpha([GRAYED_ALPHA if v in grayed_nodes else ACTIVE_ALPHA for v in nodes])
        elif isinstance(coll, LineCollection) and len(coll.get_segments()) == len(edges):
            coll.set_alpha([GRAYED_ALPHA if e in grayed_edges else ACTIVE_ALPHA for e in edges])


register_backend("networkx", NetworkXBackend)

__all__ = ["NetworkXBackend"]
