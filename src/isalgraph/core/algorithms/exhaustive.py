"""Exhaustive (canonical) G2S algorithm.

Uses backtracking search over all valid neighbor choices at V/v branch
points, across all starting nodes. Produces the true canonical string
w*_G = lexmin among shortest strings over all starting nodes and all
execution paths.

This is a **complete graph invariant**: w*_G = w*_H iff G ~ H.

The implementation delegates to the existing ``canonical_string()``
function in ``isalgraph.core.canonical`` which contains the optimized
backtracking search with in-place mutation and undo.
"""

from __future__ import annotations

from isalgraph.core.algorithms.base import G2SAlgorithm
from isalgraph.core.canonical import canonical_string
from isalgraph.core.sparse_graph import SparseGraph


class ExhaustiveG2S(G2SAlgorithm):
    """Exhaustive: backtracking search for the true canonical string.

    Explores all possible neighbor orderings at V/v branch points via
    depth-first backtracking. The result is a complete graph invariant.

    Time complexity: exponential in the worst case (factorial branching
    at each V/v step), but pruning via the greedy pair ordering and
    length bound keeps it practical for small graphs (N <= ~15).
    """

    def encode(self, graph: SparseGraph) -> str:
        """Encode graph using exhaustive canonical search.

        Args:
            graph: The SparseGraph to encode.

        Returns:
            The canonical string w*_G.

        Raises:
            ValueError: If no starting node can reach all other nodes.
        """
        return canonical_string(graph)

    @property
    def name(self) -> str:
        return "exhaustive"
