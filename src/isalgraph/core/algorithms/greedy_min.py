"""Greedy-min G2S algorithm.

Runs the greedy GraphToString from every starting node, then selects
the lexicographically smallest among the shortest strings produced.
This is the default IsalGraph encoding algorithm.

Compared to ExhaustiveG2S, greedy-min is much faster (polynomial) but
may not produce the true canonical string because the greedy neighbor
choice at V/v steps is not isomorphism-equivariant.
"""

from __future__ import annotations

from isalgraph.core.algorithms.base import G2SAlgorithm
from isalgraph.core.graph_to_string import GraphToString
from isalgraph.core.sparse_graph import SparseGraph


class GreedyMinG2S(G2SAlgorithm):
    """Greedy-min: greedy encoding from all starting nodes, pick lexmin shortest.

    For a graph with N nodes, runs N greedy G2S executions (one per
    starting node) and returns the shortest string. Ties in length are
    broken lexicographically.

    Time complexity: O(N * T_greedy) where T_greedy is the per-node
    greedy execution cost.
    """

    def encode(self, graph: SparseGraph) -> str:
        """Encode graph using greedy-min algorithm.

        Args:
            graph: The SparseGraph to encode.

        Returns:
            The lexmin shortest greedy string across all starting nodes.

        Raises:
            ValueError: If no starting node can reach all other nodes.
        """
        n = graph.node_count()
        if n == 0:
            return ""
        if n == 1 and graph.logical_edge_count() == 0:
            return ""

        results: list[tuple[int, str]] = []
        for v in range(n):
            try:
                gts = GraphToString(graph)
                s, _ = gts.run(initial_node=v)
                results.append((len(s), s))
            except (ValueError, RuntimeError):
                continue

        if not results:
            raise ValueError(
                "No starting node can reach all other nodes. "
                "For undirected graphs, the graph must be connected."
            )

        # Sort by (length, string) to get lexmin among shortest
        results.sort()
        return results[0][1]

    @property
    def name(self) -> str:
        return "greedy_min"
