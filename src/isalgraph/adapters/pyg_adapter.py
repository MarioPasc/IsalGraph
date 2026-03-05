"""PyTorch Geometric <-> SparseGraph adapter."""

from __future__ import annotations

try:
    import torch
    from torch_geometric.data import Data
except ImportError as exc:
    raise ImportError(
        "PyTorch and PyTorch Geometric are required for this adapter. "
        "Install them with: pip install isalgraph[pyg]"
    ) from exc

from isalgraph.adapters.base import GraphAdapter
from isalgraph.core.sparse_graph import SparseGraph


class PyGAdapter(GraphAdapter[Data]):
    """Bridge between PyTorch Geometric Data objects and IsalGraph SparseGraph.

    PyG stores edges as a ``(2, E)`` tensor ``edge_index``.  This adapter
    converts between that format and SparseGraph's adjacency-set representation.

    Node features and edge features are **not preserved** during conversion.
    """

    def from_external(self, graph: Data, *, directed: bool) -> SparseGraph:
        """Convert a PyG Data object to a SparseGraph."""
        if graph.num_nodes is None:
            raise ValueError("PyG Data object has no num_nodes")

        n: int = int(graph.num_nodes)
        sg = SparseGraph(max_nodes=n, directed_graph=directed)
        for _ in range(n):
            sg.add_node()

        if graph.edge_index is not None:
            edge_index = graph.edge_index
            for i in range(edge_index.size(1)):
                src = int(edge_index[0, i])
                tgt = int(edge_index[1, i])
                sg.add_edge(src, tgt)
        return sg

    def to_external(self, sparse_graph: SparseGraph) -> Data:
        """Convert a SparseGraph to a PyG Data object."""
        n = sparse_graph.node_count()
        sources: list[int] = []
        targets: list[int] = []

        for u in range(n):
            for v in sparse_graph.neighbors(u):
                if sparse_graph.directed() or u < v:
                    sources.append(u)
                    targets.append(v)
                    if not sparse_graph.directed():
                        sources.append(v)
                        targets.append(u)

        if sources:
            edge_index = torch.tensor([sources, targets], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return Data(
            num_nodes=n,
            edge_index=edge_index,
        )
