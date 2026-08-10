"""Exhaustive (canonical) G2S algorithm.

Uses backtracking search over all valid neighbor choices at V/v branch
points, across all starting nodes. Produces the true canonical string
w*_G = lexmin among shortest strings over all starting nodes and all
execution paths.

This is a **complete graph invariant**: w*_G = w*_H iff G ~ H.

The implementation delegates to ``isalgraph.core.backends.canonical_string``,
which dispatches to the C++ engine when it is available and to the Python
reference in ``isalgraph.core.canonical`` otherwise.  Both produce
byte-identical strings; see ``tests/native/test_native_differential.py``.
"""

from __future__ import annotations

from isalgraph.core.algorithms.base import G2SAlgorithm, _as_legacy_value_error
from isalgraph.core.backends import canonical_string
from isalgraph.core.sparse_graph import SparseGraph


class ExhaustiveG2S(G2SAlgorithm):
    """Exhaustive: backtracking search for the true canonical string.

    Explores all possible neighbor orderings at V/v branch points via
    depth-first backtracking. The result is a complete graph invariant.

    Time complexity: exponential in the worst case (factorial branching
    at each V/v step), but pruning via the greedy pair ordering and
    length bound keeps it practical for small graphs (N <= ~15).

    Args:
        backend: ``"cpp"``, ``"python"``, or ``None`` to follow the active
            engine (``ISALGRAPH_ENGINE``, else the compiled-in default).
        timeout_s: Wall-clock budget per encode, in seconds. Supported by the
            ``cpp`` backend only.
        threads: Worker threads over the starting-node loop. The default of 1
            is the only value safe to assume inside a SLURM cgroup.
    """

    def __init__(
        self,
        backend: str | None = None,
        *,
        timeout_s: float | None = None,
        threads: int = 1,
    ) -> None:
        self._backend = backend
        self._timeout_s = timeout_s
        self._threads = threads

    def encode(self, graph: SparseGraph) -> str:
        """Encode graph using exhaustive canonical search.

        Args:
            graph: The SparseGraph to encode.

        Returns:
            The canonical string w*_G.

        Raises:
            ValueError: If no starting node can reach all other nodes.
        """
        return _as_legacy_value_error(
            lambda: canonical_string(
                graph,
                timeout_s=self._timeout_s,
                threads=self._threads,
                backend=self._backend,
            )
        )

    @property
    def name(self) -> str:
        return "exhaustive"
