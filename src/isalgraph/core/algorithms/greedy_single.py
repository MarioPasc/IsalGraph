"""Greedy-single G2S algorithm.

Runs the greedy GraphToString from a single starting node. The starting
node can be specified explicitly or chosen uniformly at random.

This is the fastest G2S algorithm (single execution) but produces
strings that depend on the starting node choice. Useful as a baseline
or when encoding speed is critical.
"""

from __future__ import annotations

from isalgraph.core.algorithms.base import G2SAlgorithm
from isalgraph.core.graph_to_string import GraphToString
from isalgraph.core.sparse_graph import SparseGraph


class GreedySingleG2S(G2SAlgorithm):
    """Greedy-single: greedy encoding from one starting node.

    Args:
        start_node: The starting node index. If None, defaults to 0.

    Time complexity: O(T_greedy) for a single greedy execution.
    """

    def __init__(self, start_node: int | None = None) -> None:
        self._start_node = start_node

    def encode(self, graph: SparseGraph) -> str:
        """Encode graph using greedy algorithm from a single starting node.

        Args:
            graph: The SparseGraph to encode.

        Returns:
            The greedy string from the specified starting node.

        Raises:
            ValueError: If the starting node is out of range or
                cannot reach all other nodes.
        """
        n = graph.node_count()
        if n == 0:
            return ""
        if n == 1 and graph.logical_edge_count() == 0:
            return ""

        v0 = self._start_node if self._start_node is not None else 0
        if v0 < 0 or v0 >= n:
            raise ValueError(f"start_node={v0} out of range [0, {n})")

        gts = GraphToString(graph)
        s, _ = gts.run(initial_node=v0)
        return s

    @property
    def name(self) -> str:
        return "greedy_single"

    def __repr__(self) -> str:
        return f"GreedySingleG2S(start_node={self._start_node})"
