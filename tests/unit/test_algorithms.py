"""Unit tests for G2S algorithm implementations.

Covers:
- GreedyMinG2S: greedy encoding from all starting nodes
- GreedySingleG2S: greedy encoding from a single starting node
- ExhaustiveG2S: canonical exhaustive search
- G2SAlgorithm: abstract base class contract
"""

from __future__ import annotations

import pytest

from isalgraph.core.algorithms import (
    DEFAULT_ALGORITHM,
    ExhaustiveG2S,
    G2SAlgorithm,
    GreedyMinG2S,
    GreedySingleG2S,
    PrunedExhaustiveG2S,
)
from isalgraph.core.sparse_graph import SparseGraph
from isalgraph.core.string_to_graph import StringToGraph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph(n: int, edges: list[tuple[int, int]], directed: bool = False) -> SparseGraph:
    g = SparseGraph(max_nodes=n, directed_graph=directed)
    for _ in range(n):
        g.add_node()
    for u, v in edges:
        g.add_edge(u, v)
    return g


def _roundtrip(string: str, directed: bool = False) -> SparseGraph:
    """S2G: string -> graph."""
    stg = StringToGraph(string, directed_graph=directed)
    g, _ = stg.run()
    return g


# ---------------------------------------------------------------------------
# Test: G2SAlgorithm base class
# ---------------------------------------------------------------------------


class TestG2SAlgorithmBase:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            G2SAlgorithm()  # type: ignore[abstract]

    def test_all_subclasses_implement_encode(self) -> None:
        for cls in [GreedyMinG2S, ExhaustiveG2S, GreedySingleG2S, PrunedExhaustiveG2S]:
            instance = cls() if cls != GreedySingleG2S else cls(start_node=0)
            assert hasattr(instance, "encode")
            assert callable(instance.encode)

    def test_all_subclasses_implement_name(self) -> None:
        for cls in [GreedyMinG2S, ExhaustiveG2S, GreedySingleG2S, PrunedExhaustiveG2S]:
            instance = cls() if cls != GreedySingleG2S else cls(start_node=0)
            assert isinstance(instance.name, str)
            assert len(instance.name) > 0

    def test_default_algorithm_is_greedy_min(self) -> None:
        assert DEFAULT_ALGORITHM is GreedyMinG2S


# ---------------------------------------------------------------------------
# Test: GreedyMinG2S
# ---------------------------------------------------------------------------


class TestGreedyMinG2S:
    def test_name(self) -> None:
        algo = GreedyMinG2S()
        assert algo.name == "greedy_min"

    def test_repr(self) -> None:
        algo = GreedyMinG2S()
        assert repr(algo) == "GreedyMinG2S()"

    def test_single_node(self) -> None:
        g = _make_graph(1, [])
        algo = GreedyMinG2S()
        assert algo.encode(g) == ""

    def test_empty_graph(self) -> None:
        g = SparseGraph(0, directed_graph=False)
        algo = GreedyMinG2S()
        assert algo.encode(g) == ""

    def test_two_node_edge(self) -> None:
        g = _make_graph(2, [(0, 1)])
        algo = GreedyMinG2S()
        result = algo.encode(g)
        assert len(result) > 0
        # Round-trip: S2G should give isomorphic graph
        g2 = _roundtrip(result)
        assert g2.node_count() == 2
        assert g2.logical_edge_count() == 1

    def test_triangle_undirected(self, triangle_undirected: SparseGraph) -> None:
        algo = GreedyMinG2S()
        result = algo.encode(triangle_undirected)
        g2 = _roundtrip(result)
        assert g2.node_count() == 3
        assert g2.logical_edge_count() == 3

    def test_triangle_directed(self, triangle_directed: SparseGraph) -> None:
        algo = GreedyMinG2S()
        result = algo.encode(triangle_directed)
        g2 = _roundtrip(result, directed=True)
        assert g2.node_count() == 3
        assert g2.logical_edge_count() == 3

    def test_path_undirected(self, path_3_undirected: SparseGraph) -> None:
        algo = GreedyMinG2S()
        result = algo.encode(path_3_undirected)
        g2 = _roundtrip(result)
        assert g2.node_count() == 3
        assert g2.logical_edge_count() == 2

    def test_star_graph(self) -> None:
        # Star with center=0 and leaves 1,2,3,4
        g = _make_graph(5, [(0, 1), (0, 2), (0, 3), (0, 4)])
        algo = GreedyMinG2S()
        result = algo.encode(g)
        g2 = _roundtrip(result)
        assert g2.node_count() == 5
        assert g2.logical_edge_count() == 4

    def test_selects_lexmin_among_shortest(self) -> None:
        """For symmetric graphs, different starting nodes may give same-length
        strings. GreedyMin should pick the lexicographically smallest."""
        g = _make_graph(3, [(0, 1), (1, 2), (2, 0)])  # triangle
        algo = GreedyMinG2S()
        result = algo.encode(g)
        # Run greedy from each node independently and verify
        from isalgraph.core.graph_to_string import GraphToString

        all_results: list[tuple[int, str]] = []
        for v in range(3):
            gts = GraphToString(g)
            s, _ = gts.run(initial_node=v)
            all_results.append((len(s), s))
        all_results.sort()
        assert result == all_results[0][1]

    def test_disconnected_raises(self) -> None:
        """Disconnected undirected graph: no starting node can reach all."""
        g = _make_graph(4, [(0, 1)])  # nodes 2,3 isolated
        algo = GreedyMinG2S()
        with pytest.raises(ValueError, match="No starting node"):
            algo.encode(g)

    def test_directed_unreachable_raises(self) -> None:
        """Directed graph where no single node can reach all others."""
        g = _make_graph(3, [(0, 1), (2, 1)], directed=True)
        algo = GreedyMinG2S()
        with pytest.raises(ValueError, match="No starting node"):
            algo.encode(g)

    def test_complete_graph(self) -> None:
        edges = [(i, j) for i in range(5) for j in range(5) if i != j]
        g = _make_graph(5, edges)
        algo = GreedyMinG2S()
        result = algo.encode(g)
        g2 = _roundtrip(result)
        assert g2.node_count() == 5
        # K5 undirected has 10 edges
        assert g2.logical_edge_count() == 10


# ---------------------------------------------------------------------------
# Test: GreedySingleG2S
# ---------------------------------------------------------------------------


class TestGreedySingleG2S:
    def test_name(self) -> None:
        algo = GreedySingleG2S()
        assert algo.name == "greedy_single"

    def test_repr_default(self) -> None:
        algo = GreedySingleG2S()
        assert repr(algo) == "GreedySingleG2S(start_node=None)"

    def test_repr_with_node(self) -> None:
        algo = GreedySingleG2S(start_node=3)
        assert repr(algo) == "GreedySingleG2S(start_node=3)"

    def test_single_node(self) -> None:
        g = _make_graph(1, [])
        algo = GreedySingleG2S()
        assert algo.encode(g) == ""

    def test_empty_graph(self) -> None:
        g = SparseGraph(0, directed_graph=False)
        algo = GreedySingleG2S()
        assert algo.encode(g) == ""

    def test_default_start_node_is_zero(self) -> None:
        g = _make_graph(3, [(0, 1), (1, 2), (2, 0)])
        algo = GreedySingleG2S()
        result = algo.encode(g)
        # Compare with explicit start_node=0
        algo0 = GreedySingleG2S(start_node=0)
        result0 = algo0.encode(g)
        assert result == result0

    def test_explicit_start_node(self) -> None:
        g = _make_graph(3, [(0, 1), (1, 2), (2, 0)])
        algo = GreedySingleG2S(start_node=1)
        result = algo.encode(g)
        g2 = _roundtrip(result)
        assert g2.node_count() == 3
        assert g2.logical_edge_count() == 3

    def test_different_start_nodes_may_differ(self) -> None:
        """Path graph: starting from endpoint vs middle gives different strings."""
        g = _make_graph(4, [(0, 1), (1, 2), (2, 3)])
        algo0 = GreedySingleG2S(start_node=0)
        algo1 = GreedySingleG2S(start_node=1)
        r0 = algo0.encode(g)
        r1 = algo1.encode(g)
        # Both valid round-trips
        for r in [r0, r1]:
            g2 = _roundtrip(r)
            assert g2.node_count() == 4
            assert g2.logical_edge_count() == 3

    def test_start_node_out_of_range_raises(self) -> None:
        g = _make_graph(3, [(0, 1), (1, 2)])
        algo = GreedySingleG2S(start_node=5)
        with pytest.raises(ValueError, match="out of range"):
            algo.encode(g)

    def test_negative_start_node_raises(self) -> None:
        g = _make_graph(3, [(0, 1), (1, 2)])
        algo = GreedySingleG2S(start_node=-1)
        with pytest.raises(ValueError, match="out of range"):
            algo.encode(g)

    def test_triangle_roundtrip(self, triangle_undirected: SparseGraph) -> None:
        algo = GreedySingleG2S(start_node=0)
        result = algo.encode(triangle_undirected)
        g2 = _roundtrip(result)
        assert g2.node_count() == 3
        assert g2.logical_edge_count() == 3

    def test_directed_roundtrip(self, triangle_directed: SparseGraph) -> None:
        algo = GreedySingleG2S(start_node=0)
        result = algo.encode(triangle_directed)
        g2 = _roundtrip(result, directed=True)
        assert g2.node_count() == 3
        assert g2.logical_edge_count() == 3


# ---------------------------------------------------------------------------
# Test: ExhaustiveG2S
# ---------------------------------------------------------------------------


class TestExhaustiveG2S:
    def test_name(self) -> None:
        algo = ExhaustiveG2S()
        assert algo.name == "exhaustive"

    def test_repr(self) -> None:
        algo = ExhaustiveG2S()
        assert repr(algo) == "ExhaustiveG2S()"

    def test_single_node(self) -> None:
        g = _make_graph(1, [])
        algo = ExhaustiveG2S()
        result = algo.encode(g)
        assert result == ""

    def test_triangle(self, triangle_undirected: SparseGraph) -> None:
        algo = ExhaustiveG2S()
        result = algo.encode(triangle_undirected)
        g2 = _roundtrip(result)
        assert g2.node_count() == 3
        assert g2.logical_edge_count() == 3

    def test_canonical_invariance(self) -> None:
        """Isomorphic graphs must produce the same canonical string."""
        # Two different labelings of the same triangle
        g1 = _make_graph(3, [(0, 1), (1, 2), (2, 0)])
        g2 = _make_graph(3, [(0, 2), (2, 1), (1, 0)])
        algo = ExhaustiveG2S()
        assert algo.encode(g1) == algo.encode(g2)

    def test_non_isomorphic_differ(self) -> None:
        """Non-isomorphic graphs must produce different strings."""
        triangle = _make_graph(3, [(0, 1), (1, 2), (2, 0)])
        path = _make_graph(3, [(0, 1), (1, 2)])
        algo = ExhaustiveG2S()
        assert algo.encode(triangle) != algo.encode(path)

    def test_path_roundtrip(self, path_3_undirected: SparseGraph) -> None:
        algo = ExhaustiveG2S()
        result = algo.encode(path_3_undirected)
        g2 = _roundtrip(result)
        assert g2.node_count() == 3
        assert g2.logical_edge_count() == 2


# ---------------------------------------------------------------------------
# Test: PrunedExhaustiveG2S (supplement)
# ---------------------------------------------------------------------------


class TestPrunedExhaustiveG2SAlgorithm:
    def test_name(self) -> None:
        algo = PrunedExhaustiveG2S()
        assert algo.name == "pruned_exhaustive"

    def test_repr(self) -> None:
        algo = PrunedExhaustiveG2S()
        assert repr(algo) == "PrunedExhaustiveG2S()"

    def test_encode_triangle(self, triangle_undirected: SparseGraph) -> None:
        algo = PrunedExhaustiveG2S()
        result = algo.encode(triangle_undirected)
        # Must equal exhaustive canonical
        exhaustive = ExhaustiveG2S()
        assert result == exhaustive.encode(triangle_undirected)


# ---------------------------------------------------------------------------
# Test: StringToGraph constructor edge cases (coverage for lines 56, 59)
# ---------------------------------------------------------------------------


class TestStringToGraphConstructorEdgeCases:
    def test_both_directed_args_raises(self) -> None:
        with pytest.raises(TypeError, match="Cannot specify both"):
            StringToGraph("V", directed_graph=True, directed=False)

    def test_neither_directed_arg_raises(self) -> None:
        with pytest.raises(TypeError, match="requires"):
            StringToGraph("V")
