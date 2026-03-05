"""igraph <-> SparseGraph adapter."""

from __future__ import annotations

try:
    import igraph as ig
except ImportError as exc:
    raise ImportError(
        "igraph is required for this adapter. Install it with: pip install isalgraph[igraph]"
    ) from exc

from isalgraph.adapters.base import GraphAdapter
from isalgraph.core.sparse_graph import SparseGraph


class IGraphAdapter(GraphAdapter[ig.Graph]):
    """Bridge between igraph graphs and IsalGraph SparseGraph."""

    def from_external(self, graph: ig.Graph, *, directed: bool) -> SparseGraph:
        """Convert an igraph graph to a SparseGraph.

        igraph nodes are already integer-indexed (0..N-1), so no
        relabeling is needed.
        """
        n = graph.vcount()
        sg = SparseGraph(max_nodes=n, directed_graph=directed)
        for _ in range(n):
            sg.add_node()
        for e in graph.es:
            sg.add_edge(e.source, e.target)
        return sg

    def to_external(self, sparse_graph: SparseGraph) -> ig.Graph:
        """Convert a SparseGraph to an igraph graph."""
        n = sparse_graph.node_count()
        g = ig.Graph(n=n, directed=sparse_graph.directed())
        edges: list[tuple[int, int]] = []
        for u in range(n):
            for v in sparse_graph.neighbors(u):
                if sparse_graph.directed() or u < v:
                    edges.append((u, v))
        g.add_edges(edges)
        return g
