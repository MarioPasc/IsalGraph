"""Unit tests for canonical string computation (Phase 3)."""

from __future__ import annotations

import pytest

from isalgraph.core.canonical import (
    canonical_string,
    graph_distance,
    levenshtein,
)
from isalgraph.core.sparse_graph import SparseGraph

# ======================================================================
# Levenshtein distance
# ======================================================================


class TestLevenshtein:
    def test_identical(self) -> None:
        assert levenshtein("abc", "abc") == 0

    def test_empty(self) -> None:
        assert levenshtein("", "") == 0
        assert levenshtein("abc", "") == 3
        assert levenshtein("", "abc") == 3

    def test_insertion(self) -> None:
        assert levenshtein("ac", "abc") == 1

    def test_deletion(self) -> None:
        assert levenshtein("abc", "ac") == 1

    def test_substitution(self) -> None:
        assert levenshtein("abc", "axc") == 1

    def test_mixed(self) -> None:
        assert levenshtein("kitten", "sitting") == 3


# ======================================================================
# Canonical string basics
# ======================================================================


class TestCanonicalStringBasics:
    def test_empty_graph(self) -> None:
        g = SparseGraph(0, directed_graph=False)
        assert canonical_string(g) == ""

    def test_single_node(self) -> None:
        g = SparseGraph(1, directed_graph=False)
        g.add_node()
        assert canonical_string(g) == ""

    def test_two_nodes_undirected(self) -> None:
        g = SparseGraph(2, directed_graph=False)
        g.add_node()
        g.add_node()
        g.add_edge(0, 1)
        w = canonical_string(g)
        assert w == "V"

    def test_path_3(self) -> None:
        """Path 0-1-2: starting from node 1 (center) gives shortest string."""
        g = SparseGraph(3, directed_graph=False)
        for _ in range(3):
            g.add_node()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        w = canonical_string(g)
        # From center node: VV (2 chars). From endpoints: VNV (3 chars).
        assert len(w) == 2


# ======================================================================
# Canonical string invariance (isomorphic graphs have same canonical string)
# ======================================================================


class TestCanonicalInvariance:
    def test_triangle_relabeling(self) -> None:
        """Triangle 0-1-2 vs triangle 0-2-1 (relabeled) must have same canonical string."""
        g1 = SparseGraph(3, directed_graph=False)
        for _ in range(3):
            g1.add_node()
        g1.add_edge(0, 1)
        g1.add_edge(1, 2)
        g1.add_edge(0, 2)

        g2 = SparseGraph(3, directed_graph=False)
        for _ in range(3):
            g2.add_node()
        g2.add_edge(0, 2)
        g2.add_edge(2, 1)
        g2.add_edge(0, 1)

        assert canonical_string(g1) == canonical_string(g2)

    def test_path_4_relabeling(self) -> None:
        """Path 0-1-2-3 vs path 3-2-1-0 (reversed) must have same canonical string."""
        g1 = SparseGraph(4, directed_graph=False)
        for _ in range(4):
            g1.add_node()
        g1.add_edge(0, 1)
        g1.add_edge(1, 2)
        g1.add_edge(2, 3)

        # Relabeled: 0->3, 1->2, 2->1, 3->0
        g2 = SparseGraph(4, directed_graph=False)
        for _ in range(4):
            g2.add_node()
        g2.add_edge(3, 2)
        g2.add_edge(2, 1)
        g2.add_edge(1, 0)

        assert canonical_string(g1) == canonical_string(g2)

    def test_star_relabeling(self) -> None:
        """Star with center=0 vs center=3 must have same canonical string."""
        g1 = SparseGraph(5, directed_graph=False)
        for _ in range(5):
            g1.add_node()
        for i in range(1, 5):
            g1.add_edge(0, i)

        # Relabeled: center is node 3
        g2 = SparseGraph(5, directed_graph=False)
        for _ in range(5):
            g2.add_node()
        for i in [0, 1, 2, 4]:
            g2.add_edge(3, i)

        assert canonical_string(g1) == canonical_string(g2)


# ======================================================================
# Non-isomorphic graphs have different canonical strings
# ======================================================================


class TestCanonicalDiscrimination:
    def test_path_vs_triangle(self) -> None:
        """Path 0-1-2 vs triangle 0-1-2-0 must have different canonical strings."""
        path = SparseGraph(3, directed_graph=False)
        for _ in range(3):
            path.add_node()
        path.add_edge(0, 1)
        path.add_edge(1, 2)

        triangle = SparseGraph(3, directed_graph=False)
        for _ in range(3):
            triangle.add_node()
        triangle.add_edge(0, 1)
        triangle.add_edge(1, 2)
        triangle.add_edge(0, 2)

        assert canonical_string(path) != canonical_string(triangle)

    def test_star_vs_path_4(self) -> None:
        """Star-4 (center + 3 leaves) vs path of 4 nodes."""
        star = SparseGraph(4, directed_graph=False)
        for _ in range(4):
            star.add_node()
        star.add_edge(0, 1)
        star.add_edge(0, 2)
        star.add_edge(0, 3)

        path = SparseGraph(4, directed_graph=False)
        for _ in range(4):
            path.add_node()
        path.add_edge(0, 1)
        path.add_edge(1, 2)
        path.add_edge(2, 3)

        assert canonical_string(star) != canonical_string(path)


# ======================================================================
# Graph distance
# ======================================================================


class TestGraphDistance:
    def test_isomorphic_distance_zero(self) -> None:
        """Isomorphic graphs have distance 0."""
        g1 = SparseGraph(3, directed_graph=False)
        for _ in range(3):
            g1.add_node()
        g1.add_edge(0, 1)
        g1.add_edge(1, 2)
        g1.add_edge(0, 2)

        g2 = SparseGraph(3, directed_graph=False)
        for _ in range(3):
            g2.add_node()
        g2.add_edge(0, 2)
        g2.add_edge(2, 1)
        g2.add_edge(0, 1)

        assert graph_distance(g1, g2) == 0

    def test_non_isomorphic_positive_distance(self) -> None:
        """Non-isomorphic graphs have positive distance."""
        path = SparseGraph(3, directed_graph=False)
        for _ in range(3):
            path.add_node()
        path.add_edge(0, 1)
        path.add_edge(1, 2)

        triangle = SparseGraph(3, directed_graph=False)
        for _ in range(3):
            triangle.add_node()
        triangle.add_edge(0, 1)
        triangle.add_edge(1, 2)
        triangle.add_edge(0, 2)

        assert graph_distance(path, triangle) > 0

    def test_disconnected_raises(self) -> None:
        """Disconnected graph should raise ValueError from GraphToString."""
        g = SparseGraph(3, directed_graph=False)
        for _ in range(3):
            g.add_node()
        g.add_edge(0, 1)
        # Node 2 is disconnected

        with pytest.raises(ValueError, match="reach all"):
            canonical_string(g)


# ======================================================================
# Directed canonical string (exercises the 'c' branch, lines 329-345)
# ======================================================================


class TestCanonicalDirected:
    """Test canonical string on directed graphs.

    These tests exercise the 'c' instruction branch in _step (lines 329-345)
    and the _is_reachable returning True for n<=1 (line 124).
    """

    def test_single_node_directed(self) -> None:
        """Single node directed graph: canonical string is empty.

        This exercises _is_reachable returning True for n<=1 (line 124)
        via the n==0 path in canonical_string, but let's be explicit.
        """
        g = SparseGraph(1, directed_graph=True)
        g.add_node()
        assert canonical_string(g) == ""

    def test_two_node_directed(self) -> None:
        """Directed edge 0->1: canonical string should be 'V'."""
        g = SparseGraph(2, directed_graph=True)
        g.add_node()
        g.add_node()
        g.add_edge(0, 1)
        w = canonical_string(g)
        assert w == "V"

    def test_directed_triangle_canonical(self) -> None:
        """Directed triangle 0->1->2->0: exercises the c instruction.

        The cycle 0->1->2->0 requires a back-edge (c instruction) since
        the last edge 2->0 goes from a later node to the initial node,
        requiring a secondary->primary edge direction.
        """
        g = SparseGraph(3, directed_graph=True)
        for _ in range(3):
            g.add_node()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 0)
        w = canonical_string(g)
        # Must contain c or C for the back-edge
        assert "c" in w or "C" in w
        # Verify round-trip
        from isalgraph.core.string_to_graph import StringToGraph

        stg = StringToGraph(w, directed_graph=True)
        g2, _ = stg.run()
        assert g.is_isomorphic(g2)

    def test_directed_canonical_invariance(self) -> None:
        """Two isomorphic directed graphs have same canonical string.

        g1: 0->1, 0->2, 1->2
        g2: 0->2, 0->1, 2->1 (relabeling: 1<->2)
        """
        g1 = SparseGraph(3, directed_graph=True)
        for _ in range(3):
            g1.add_node()
        g1.add_edge(0, 1)
        g1.add_edge(0, 2)
        g1.add_edge(1, 2)

        g2 = SparseGraph(3, directed_graph=True)
        for _ in range(3):
            g2.add_node()
        g2.add_edge(0, 2)
        g2.add_edge(0, 1)
        g2.add_edge(2, 1)

        assert canonical_string(g1) == canonical_string(g2)

    def test_directed_with_bidirectional_edges(self) -> None:
        """Directed graph with edges in both directions.

        0->1, 1->0, 0->2: exercises c branch for the 1->0 back-edge.
        """
        g = SparseGraph(3, directed_graph=True)
        for _ in range(3):
            g.add_node()
        g.add_edge(0, 1)
        g.add_edge(1, 0)
        g.add_edge(0, 2)
        w = canonical_string(g)
        # Verify round-trip
        from isalgraph.core.string_to_graph import StringToGraph

        stg = StringToGraph(w, directed_graph=True)
        g2, _ = stg.run()
        assert g.is_isomorphic(g2)


# ======================================================================
# Direct _is_reachable tests (line 124: n <= 1 early return)
# ======================================================================


class TestIsReachable:
    """Direct tests for _is_reachable to hit the n<=1 early return (line 124)."""

    def test_empty_graph_reachable(self) -> None:
        """Empty graph (0 nodes): _is_reachable returns True for n<=1."""
        from isalgraph.core.canonical import _is_reachable  # noqa: PLC2701

        g = SparseGraph(0, directed_graph=False)
        assert _is_reachable(g, 0) is True

    def test_single_node_reachable(self) -> None:
        """Single node: _is_reachable returns True for n<=1."""
        from isalgraph.core.canonical import _is_reachable  # noqa: PLC2701

        g = SparseGraph(1, directed_graph=False)
        g.add_node()
        assert _is_reachable(g, 0) is True
