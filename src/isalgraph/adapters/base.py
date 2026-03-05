"""Abstract adapter interface (ABC) for graph library bridges.

Follows the Bridge pattern (Gamma et al., 1994): a single abstract
interface that concrete adapters (NetworkX, igraph, PyG) implement to
translate between external graph objects and IsalGraph's SparseGraph.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from isalgraph.core.graph_to_string import GraphToString
from isalgraph.core.sparse_graph import SparseGraph
from isalgraph.core.string_to_graph import StringToGraph

T = TypeVar("T")


class GraphAdapter(ABC, Generic[T]):
    """Abstract bridge between external graph libraries and IsalGraph core."""

    @abstractmethod
    def from_external(self, graph: T, *, directed: bool) -> SparseGraph:
        """Convert an external graph object to a SparseGraph."""
        ...

    @abstractmethod
    def to_external(self, sparse_graph: SparseGraph) -> T:
        """Convert a SparseGraph to an external graph object."""
        ...

    def to_isalgraph_string(self, graph: T, *, directed: bool, initial_node: int = 0) -> str:
        """Convert an external graph to its IsalGraph instruction string."""
        sg = self.from_external(graph, directed=directed)
        gts = GraphToString(sg)
        string, _ = gts.run(initial_node)
        return string

    def from_isalgraph_string(self, string: str, *, directed: bool) -> T:
        """Convert an IsalGraph instruction string to an external graph."""
        stg = StringToGraph(string, directed_graph=directed)
        sg, _ = stg.run()
        return self.to_external(sg)
