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
from collections.abc import Callable
from typing import TypeVar

from isalgraph.core.sparse_graph import SparseGraph
from isalgraph.errors import EncodingError

T = TypeVar("T")

# ----------------------------------------------------------------------
# WAVE SHIM -- delete at integration.
#
# Introduced for wave 2026-08-10-cpp-and-viz on the instruction of `main`
# ("Keep your shim in algorithms/*.py, but make it unconditional and
# self-documenting").  The dispatch layer raises the isalgraph.errors classes,
# but errors.py does not yet give them their builtin mixins, so
# DisconnectedGraphError is not a ValueError.  Six existing tests pin the
# ValueError contract on these algorithm classes, among them
# tests/unit/test_algorithms.py::TestExhaustiveG2S::test_disconnected_raises
# and ::test_directed_unreachable_raises.
#
# TO REMOVE: once errors.py declares
# `class DisconnectedGraphError(EncodingError, ValueError)`, delete this
# function and every `_as_legacy_value_error(...)` call, then re-run
# `pytest tests/unit/test_algorithms.py` to confirm it was a no-op.
# ----------------------------------------------------------------------


def _as_legacy_value_error(call: Callable[[], T]) -> T:
    """Run *call*, re-raising any encoding failure as a plain ``ValueError``.

    Args:
        call: Zero-argument callable performing the encode.

    Returns:
        Whatever *call* returns.

    Raises:
        ValueError: If *call* raises :class:`~isalgraph.errors.EncodingError`.
    """
    try:
        return call()
    except EncodingError as exc:
        raise ValueError(str(exc)) from exc


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
