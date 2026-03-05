"""NetworkX <-> SparseGraph adapter."""

from __future__ import annotations

from typing import TypeAlias

try:
    import networkx as nx
except ImportError as exc:
    raise ImportError(
        "NetworkX is required for this adapter. Install it with: pip install isalgraph[networkx]"
    ) from exc

from isalgraph.adapters.base import GraphAdapter
from isalgraph.core.sparse_graph import SparseGraph

NxGraph: TypeAlias = nx.Graph | nx.DiGraph


class NetworkXAdapter(GraphAdapter[NxGraph]):
    """Bridge between NetworkX graphs and IsalGraph SparseGraph.

    Node mapping: external NetworkX node labels are mapped to contiguous
    integer IDs (sorted order). The mapping is stored on the adapter
    instance after each ``from_external`` call for potential recovery.

    Edge attributes and node attributes are **stripped** during
    ``from_external`` and not restored during ``to_external``.
    """

    def __init__(self) -> None:
        self._label_to_id: dict[object, int] = {}
        self._id_to_label: dict[int, object] = {}

    def from_external(self, graph: NxGraph, *, directed: bool) -> SparseGraph:
        """Convert a NetworkX graph to a SparseGraph.

        Args:
            graph: A ``nx.Graph`` or ``nx.DiGraph``.
            directed: Whether the SparseGraph should be directed.
                For consistency, this should match the type of *graph*.
        """
        node_list = sorted(graph.nodes())
        label_to_id = {label: i for i, label in enumerate(node_list)}

        sg = SparseGraph(max_nodes=len(node_list), directed_graph=directed)
        for _ in node_list:
            sg.add_node()
        for u, v in graph.edges():
            sg.add_edge(label_to_id[u], label_to_id[v])

        self._label_to_id = label_to_id
        self._id_to_label = {v: k for k, v in label_to_id.items()}
        return sg

    def to_external(self, sparse_graph: SparseGraph) -> NxGraph:
        """Convert a SparseGraph to a NetworkX graph."""
        g: NxGraph = nx.DiGraph() if sparse_graph.directed() else nx.Graph()
        for i in range(sparse_graph.node_count()):
            g.add_node(i)
        for u in range(sparse_graph.node_count()):
            for v in sparse_graph.neighbors(u):
                if sparse_graph.directed() or u < v:
                    g.add_edge(u, v)
        return g
