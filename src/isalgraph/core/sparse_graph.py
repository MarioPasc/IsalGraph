"""Adjacency-set sparse graph representation.

Migration of the original ``sparsegraph.py`` with the following bug fix:

* **B1**: ``_edge_count`` was initialized to 1 instead of 0.

Additionally, ``is_isomorphic`` no longer uses ``print()`` for diagnostics —
it returns ``bool`` only.  Callers who need diagnostics should use the
NetworkX adapter's ``nx.is_isomorphic`` for cross-validation.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class SparseGraph:
    """High-performance adjacency-set graph with contiguous integer node IDs.

    Optimized for O(1) average edge insertion and membership testing.
    Supports both directed and undirected semantics.

    Args:
        max_nodes: Upper bound on node count (pre-allocates storage).
        directed_graph: Whether edges are directed.
    """

    __slots__ = (
        "_adjacency",
        "_max_nodes",
        "_node_count",
        "_edge_count",
        "_directed_graph",
    )

    def __init__(self, max_nodes: int, directed_graph: bool) -> None:
        self._max_nodes: int = max_nodes
        self._directed_graph: bool = directed_graph
        self._adjacency: list[set[int]] = [set() for _ in range(max_nodes)]
        self._node_count: int = 0
        # BUG FIX B1: was 1 in the original code.
        self._edge_count: int = 0

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def directed(self) -> bool:
        """Return whether the graph is directed."""
        return self._directed_graph

    def node_count(self) -> int:
        """Return the current number of nodes."""
        return self._node_count

    def edge_count(self) -> int:
        """Return the raw stored edge count.

        For undirected graphs each logical edge is counted twice
        (once per direction).  Use ``logical_edge_count()`` for the
        user-facing count.
        """
        return self._edge_count

    def logical_edge_count(self) -> int:
        """Return the number of logical edges.

        For directed graphs this equals ``edge_count()``.
        For undirected graphs this equals ``edge_count() // 2``.
        """
        if self._directed_graph:
            return self._edge_count
        return self._edge_count // 2

    def max_nodes(self) -> int:
        """Return the pre-allocated maximum node capacity."""
        return self._max_nodes

    def neighbors(self, node: int) -> set[int]:
        """Return the adjacency set of *node* (read-only view intended)."""
        if node < 0 or node >= self._node_count:
            raise IndexError(f"Invalid node ID: {node}")
        return self._adjacency[node]

    def has_edge(self, source: int, target: int) -> bool:
        """Return whether the directed edge *source* -> *target* exists."""
        if source < 0 or source >= self._node_count:
            raise IndexError(f"Invalid source node ID: {source}")
        if target < 0 or target >= self._node_count:
            raise IndexError(f"Invalid target node ID: {target}")
        return target in self._adjacency[source]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(self) -> int:
        """Add a new node and return its integer ID.

        Raises:
            RuntimeError: If the graph has reached ``max_nodes``.
        """
        if self._node_count >= self._max_nodes:
            raise RuntimeError(f"Maximum number of nodes reached: {self._max_nodes}")
        node_id: int = self._node_count
        self._node_count += 1
        return node_id

    def add_edge(self, source: int, target: int) -> None:
        """Add edge *source* -> *target*.

        For undirected graphs, the reverse edge *target* -> *source* is
        also inserted and the stored edge count is incremented by 2.

        Raises:
            IndexError: If either node ID is out of range.
        """
        if source < 0 or source >= self._node_count:
            raise IndexError(f"Invalid source node ID: {source}")
        if target < 0 or target >= self._node_count:
            raise IndexError(f"Invalid target node ID: {target}")

        if target not in self._adjacency[source]:
            self._adjacency[source].add(target)
            self._edge_count += 1

            if not self._directed_graph:
                self._adjacency[target].add(source)
                self._edge_count += 1

    # ------------------------------------------------------------------
    # Isomorphism (backtracking)
    # ------------------------------------------------------------------

    def is_isomorphic(self, other: SparseGraph) -> bool:
        """Test structural isomorphism with *other* via backtracking.

        This is a simple implementation for testing/debugging.  For
        production use, prefer the NetworkX adapter's ``nx.is_isomorphic``.
        """
        if not isinstance(other, SparseGraph):
            return False
        if self._directed_graph != other._directed_graph:
            return False
        if self._node_count != other._node_count:
            return False

        n = self._node_count
        if n == 0:
            return True

        self_deg = [len(self._adjacency[u]) for u in range(n)]
        other_deg = [len(other._adjacency[u]) for u in range(n)]

        if sorted(self_deg) != sorted(other_deg):
            return False

        # Order by degree descending for early pruning.
        self_order = sorted(range(n), key=lambda u: self_deg[u], reverse=True)
        other_order = sorted(range(n), key=lambda u: other_deg[u], reverse=True)

        mapping: dict[int, int] = {}
        used: set[int] = set()

        def _backtrack(i: int) -> bool:
            if i == n:
                return True
            u = self_order[i]
            for v in other_order:
                if v in used:
                    continue
                if self_deg[u] != other_deg[v]:
                    continue
                # Check consistency with already-mapped neighbours.
                ok = True
                for u2, v2 in mapping.items():
                    if (u2 in self._adjacency[u]) != (v2 in other._adjacency[v]):
                        ok = False
                        break
                    if self._directed_graph and (
                        (u in self._adjacency[u2]) != (v in other._adjacency[v2])
                    ):
                        ok = False
                        break
                if not ok:
                    continue
                mapping[u] = v
                used.add(v)
                if _backtrack(i + 1):
                    return True
                del mapping[u]
                used.remove(v)
            return False

        return _backtrack(0)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nodes={self._node_count}, "
            f"edges={self.logical_edge_count()}, "
            f"directed={self._directed_graph})"
        )
