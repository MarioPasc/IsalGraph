"""Pruned exhaustive (canonical) G2S algorithm.

Uses the same backtracking search as the exhaustive algorithm but with
structural triplet pruning at V/v branch points. Only candidates sharing
the maximum lexicographic triplet (|N_1|, |N_2|, |N_3|) are explored,
dramatically reducing the branching factor while preserving the complete
invariant property.

The implementation delegates to ``pruned_canonical_string()`` in
``isalgraph.core.canonical_pruned``.
"""

from __future__ import annotations

from isalgraph.core.algorithms.base import G2SAlgorithm
from isalgraph.core.canonical_pruned import pruned_canonical_string
from isalgraph.core.sparse_graph import SparseGraph


class PrunedExhaustiveG2S(G2SAlgorithm):
    """Pruned exhaustive: backtracking with structural triplet pruning.

    At each V/v branch point, only candidates with the maximum
    lexicographic triplet (|N_1|, |N_2|, |N_3|) are explored.
    The result is a complete graph invariant identical to the
    exhaustive canonical string.

    Time complexity: same worst case as exhaustive, but typically
    much faster on heterogeneous graphs where triplet pruning
    reduces the branching factor to 1.
    """

    def encode(self, graph: SparseGraph) -> str:
        """Encode graph using pruned exhaustive canonical search.

        Args:
            graph: The SparseGraph to encode.

        Returns:
            The canonical string w*_G.

        Raises:
            ValueError: If no starting node can reach all other nodes.
        """
        return pruned_canonical_string(graph)

    @property
    def name(self) -> str:
        return "pruned_exhaustive"
