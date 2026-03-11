"""Abstract base class for G2S (Graph-to-String) algorithms.

All G2S algorithms share the same contract: given a SparseGraph, produce
an IsalGraph instruction string. They differ in how they select starting
nodes and whether they explore multiple neighbor orderings.

The hierarchy:
    G2SAlgorithm (ABC)
    ├── GreedyMinG2S     -- greedy over all starting nodes, pick lexmin shortest
    ├── ExhaustiveG2S    -- backtracking search (complete invariant)
    └── GreedySingleG2S  -- greedy from a single (random) starting node
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from isalgraph.core.sparse_graph import SparseGraph


class G2SAlgorithm(ABC):
    """Abstract Graph-to-String algorithm.

    Subclasses must implement ``encode()`` which converts a SparseGraph
    into an IsalGraph instruction string.
    """

    @abstractmethod
    def encode(self, graph: SparseGraph) -> str:
        """Convert a graph to an IsalGraph instruction string.

        Args:
            graph: The SparseGraph to encode. Must be connected (undirected)
                or have all nodes reachable from at least one node (directed).

        Returns:
            An IsalGraph instruction string over the alphabet
            {N, n, P, p, V, v, C, c, W}.

        Raises:
            ValueError: If the graph cannot be encoded (e.g., disconnected).
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this algorithm (e.g., 'greedy_min')."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
