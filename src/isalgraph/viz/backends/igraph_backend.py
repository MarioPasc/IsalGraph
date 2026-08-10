"""Optional backend using igraph for layout, matplotlib for painting.

igraph is worth carrying for its layout catalogue -- Kamada-Kawai,
Reingold-Tilford, and the Sugiyama layered layout, which is the natural
choice for the directed encodings. Painting is delegated to
:class:`~isalgraph.viz.backends.matplotlib_backend.MatplotlibBackend`, so
the visual language stays identical across backends and only the node
coordinates change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isalgraph.core.sparse_graph import SparseGraph
from isalgraph.core.trace import Edge, graph_edges
from isalgraph.types import NodeId
from isalgraph.viz.backends.matplotlib_backend import MatplotlibBackend
from isalgraph.viz.base import GraphVizBackend, Position
from isalgraph.viz.registry import register_backend

if TYPE_CHECKING:
    from matplotlib.axes import Axes
else:
    Axes = Any


class IGraphBackend(GraphVizBackend):
    """Lay out with igraph, paint with the matplotlib backend.

    Args:
        layout_name: Any layout accepted by ``igraph.Graph.layout``, for
            example ``"kk"`` (Kamada-Kawai), ``"fr"`` (Fruchterman-
            Reingold) or ``"sugiyama"``.
        fit_fraction: Fraction of the ``[-1, 1]^2`` canvas the drawing is
            rescaled to occupy, matching
            :func:`~isalgraph.viz.layout.compact_graph_layout`.
    """

    def __init__(self, *, layout_name: str = "kk", fit_fraction: float = 0.78) -> None:
        self._layout_name = layout_name
        self._fit_fraction = fit_fraction
        self._painter = MatplotlibBackend()

    @property
    def name(self) -> str:
        return "igraph"

    @classmethod
    def is_available(cls) -> bool:
        """Report whether both igraph and matplotlib import."""
        try:
            import igraph  # noqa: F401
            import matplotlib  # noqa: F401
        except ImportError:
            return False
        return True

    def _igraph_layout(self, graph: SparseGraph) -> dict[NodeId, Position]:
        """Compute node coordinates with igraph, normalised to the canvas."""
        import igraph as ig

        n = graph.node_count()
        edges = [(u, v) for u, v in graph_edges(graph) if u != v]
        g = ig.Graph(n=n, edges=edges, directed=graph.directed())
        coords = g.layout(self._layout_name)

        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        if not xs:
            return {}
        cx, cy = (max(xs) + min(xs)) / 2.0, (max(ys) + min(ys)) / 2.0
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
        scale = 2.0 * self._fit_fraction
        # igraph's y axis points down; flip it so the drawing matches the
        # orientation produced by the NetworkX and matplotlib backends.
        return {v: (scale * (xs[v] - cx) / span, -scale * (ys[v] - cy) / span) for v in range(n)}

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
        pos = layout if layout is not None else self._igraph_layout(graph)
        return self._painter.draw(
            graph,
            ax,
            node_colors=node_colors,
            edge_colors=edge_colors,
            grayed_nodes=grayed_nodes,
            grayed_edges=grayed_edges,
            layout=pos,
        )


register_backend("igraph", IGraphBackend)

__all__ = ["IGraphBackend"]
