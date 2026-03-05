"""Unit tests for SparseGraph."""

from __future__ import annotations

import pytest

from isalgraph.core.sparse_graph import SparseGraph


class TestSparseGraphBasics:
    """Basic node/edge operations."""

    def test_empty_graph(self) -> None:
        g = SparseGraph(10, directed_graph=False)
        assert g.node_count() == 0
        assert g.edge_count() == 0
        assert g.logical_edge_count() == 0

    def test_edge_count_init_is_zero(self) -> None:
        """Regression: original code initialized _edge_count to 1 (B1)."""
        g = SparseGraph(5, directed_graph=True)
        assert g.edge_count() == 0
        g2 = SparseGraph(5, directed_graph=False)
        assert g2.edge_count() == 0

    def test_add_node(self) -> None:
        g = SparseGraph(5, directed_graph=False)
        assert g.add_node() == 0
        assert g.add_node() == 1
        assert g.node_count() == 2

    def test_add_node_overflow(self) -> None:
        g = SparseGraph(2, directed_graph=False)
        g.add_node()
        g.add_node()
        with pytest.raises(RuntimeError, match="Maximum"):
            g.add_node()

    def test_directed_edge(self) -> None:
        g = SparseGraph(3, directed_graph=True)
        g.add_node()
        g.add_node()
        g.add_edge(0, 1)
        assert g.has_edge(0, 1)
        assert not g.has_edge(1, 0)
        assert g.edge_count() == 1
        assert g.logical_edge_count() == 1

    def test_undirected_edge(self) -> None:
        g = SparseGraph(3, directed_graph=False)
        g.add_node()
        g.add_node()
        g.add_edge(0, 1)
        assert g.has_edge(0, 1)
        assert g.has_edge(1, 0)
        # Raw edge count = 2 (both directions stored).
        assert g.edge_count() == 2
        assert g.logical_edge_count() == 1

    def test_neighbors(self) -> None:
        g = SparseGraph(5, directed_graph=False)
        for _ in range(3):
            g.add_node()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        assert g.neighbors(0) == {1, 2}
        assert g.neighbors(1) == {0}

    def test_invalid_edge_raises(self) -> None:
        g = SparseGraph(5, directed_graph=False)
        g.add_node()
        with pytest.raises(IndexError):
            g.add_edge(0, 1)  # node 1 doesn't exist

    def test_invalid_neighbor_raises(self) -> None:
        g = SparseGraph(5, directed_graph=False)
        with pytest.raises(IndexError):
            g.neighbors(0)  # no nodes yet


class TestSparseGraphIsomorphism:
    """Backtracking isomorphism checker."""

    def test_identical_graphs(self) -> None:
        g1 = SparseGraph(3, directed_graph=False)
        g2 = SparseGraph(3, directed_graph=False)
        for g in (g1, g2):
            g.add_node()
            g.add_node()
            g.add_node()
            g.add_edge(0, 1)
            g.add_edge(1, 2)
        assert g1.is_isomorphic(g2)

    def test_different_node_count(self) -> None:
        g1 = SparseGraph(3, directed_graph=False)
        g2 = SparseGraph(3, directed_graph=False)
        g1.add_node()
        g1.add_node()
        g2.add_node()
        assert not g1.is_isomorphic(g2)

    def test_relabeled_triangle(self) -> None:
        """Triangle 0-1-2-0 vs triangle with permuted labels."""
        g1 = SparseGraph(3, directed_graph=False)
        g2 = SparseGraph(3, directed_graph=False)
        for g in (g1, g2):
            for _ in range(3):
                g.add_node()
        # g1: 0-1, 1-2, 2-0
        g1.add_edge(0, 1)
        g1.add_edge(1, 2)
        g1.add_edge(2, 0)
        # g2: 0-2, 2-1, 1-0  (same structure, relabeled)
        g2.add_edge(0, 2)
        g2.add_edge(2, 1)
        g2.add_edge(1, 0)
        assert g1.is_isomorphic(g2)

    def test_directed_not_isomorphic(self) -> None:
        """Directed: 0->1 vs 1->0 are not isomorphic (different structure)."""
        g1 = SparseGraph(3, directed_graph=True)
        g2 = SparseGraph(3, directed_graph=True)
        for g in (g1, g2):
            g.add_node()
            g.add_node()
            g.add_node()
        g1.add_edge(0, 1)
        g1.add_edge(1, 2)
        g2.add_edge(1, 0)
        g2.add_edge(2, 1)
        # These ARE isomorphic: both are paths of length 2.
        assert g1.is_isomorphic(g2)

    def test_empty_graphs_isomorphic(self) -> None:
        g1 = SparseGraph(0, directed_graph=False)
        g2 = SparseGraph(0, directed_graph=False)
        assert g1.is_isomorphic(g2)

    def test_single_node_isomorphic(self) -> None:
        g1 = SparseGraph(1, directed_graph=False)
        g2 = SparseGraph(1, directed_graph=False)
        g1.add_node()
        g2.add_node()
        assert g1.is_isomorphic(g2)


class TestSparseGraphHasEdgeErrors:
    """has_edge with invalid node IDs."""

    def test_has_edge_invalid_source(self) -> None:
        g = SparseGraph(3, directed_graph=False)
        g.add_node()
        g.add_node()
        with pytest.raises(IndexError, match="source"):
            g.has_edge(5, 0)

    def test_has_edge_negative_source(self) -> None:
        g = SparseGraph(3, directed_graph=False)
        g.add_node()
        with pytest.raises(IndexError, match="source"):
            g.has_edge(-1, 0)

    def test_has_edge_invalid_target(self) -> None:
        g = SparseGraph(3, directed_graph=False)
        g.add_node()
        g.add_node()
        with pytest.raises(IndexError, match="target"):
            g.has_edge(0, 5)

    def test_has_edge_negative_target(self) -> None:
        g = SparseGraph(3, directed_graph=False)
        g.add_node()
        with pytest.raises(IndexError, match="target"):
            g.has_edge(0, -1)


class TestSparseGraphAddEdgeErrors:
    """add_edge with invalid node IDs."""

    def test_add_edge_invalid_source(self) -> None:
        g = SparseGraph(3, directed_graph=False)
        g.add_node()
        with pytest.raises(IndexError, match="source"):
            g.add_edge(5, 0)

    def test_add_edge_negative_source(self) -> None:
        g = SparseGraph(3, directed_graph=False)
        g.add_node()
        with pytest.raises(IndexError, match="source"):
            g.add_edge(-1, 0)


class TestSparseGraphIsomorphismEdgeCases:
    """Edge cases for the backtracking isomorphism checker."""

    def test_not_isomorphic_with_non_sparsegraph(self) -> None:
        """is_isomorphic returns False when other is not a SparseGraph."""
        g = SparseGraph(3, directed_graph=False)
        g.add_node()
        assert not g.is_isomorphic("not a graph")  # type: ignore[arg-type]

    def test_not_isomorphic_different_directed_flag(self) -> None:
        """is_isomorphic returns False when directed flags differ."""
        g1 = SparseGraph(2, directed_graph=True)
        g2 = SparseGraph(2, directed_graph=False)
        g1.add_node()
        g2.add_node()
        assert not g1.is_isomorphic(g2)

    def test_not_isomorphic_different_degree_sequence(self) -> None:
        """is_isomorphic returns False when degree sequences differ."""
        # g1: star graph (center degree 3)
        g1 = SparseGraph(4, directed_graph=False)
        for _ in range(4):
            g1.add_node()
        g1.add_edge(0, 1)
        g1.add_edge(0, 2)
        g1.add_edge(0, 3)

        # g2: path graph (max degree 2)
        g2 = SparseGraph(4, directed_graph=False)
        for _ in range(4):
            g2.add_node()
        g2.add_edge(0, 1)
        g2.add_edge(1, 2)
        g2.add_edge(2, 3)

        assert not g1.is_isomorphic(g2)


class TestSparseGraphRepr:
    """String representation."""

    def test_repr_undirected(self) -> None:
        g = SparseGraph(5, directed_graph=False)
        g.add_node()
        g.add_node()
        g.add_edge(0, 1)
        r = repr(g)
        assert "SparseGraph" in r
        assert "nodes=2" in r
        assert "edges=1" in r
        assert "directed=False" in r

    def test_repr_directed(self) -> None:
        g = SparseGraph(5, directed_graph=True)
        g.add_node()
        r = repr(g)
        assert "directed=True" in r
