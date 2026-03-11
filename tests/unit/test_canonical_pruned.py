"""Unit tests for pruned canonical string computation."""

from __future__ import annotations

import time

import pytest

from isalgraph.core.canonical import canonical_string
from isalgraph.core.canonical_pruned import (
    compute_structural_triplets,
    pruned_canonical_string,
    pruned_graph_distance,
)
from isalgraph.core.sparse_graph import SparseGraph
from isalgraph.core.string_to_graph import StringToGraph

# ======================================================================
# Helper: build graphs
# ======================================================================


def _make_graph(
    n: int,
    edges: list[tuple[int, int]],
    *,
    directed: bool = False,
) -> SparseGraph:
    """Helper to build a SparseGraph."""
    g = SparseGraph(n, directed_graph=directed)
    for _ in range(n):
        g.add_node()
    for u, v in edges:
        g.add_edge(u, v)
    return g


def _make_path(n: int) -> SparseGraph:
    """Linear path 0-1-..-(n-1)."""
    return _make_graph(n, [(i, i + 1) for i in range(n - 1)])


def _make_cycle(n: int) -> SparseGraph:
    """Cycle of n nodes."""
    edges = [(i, (i + 1) % n) for i in range(n)]
    return _make_graph(n, edges)


def _make_star(n_leaves: int) -> SparseGraph:
    """Star with center=0 and n_leaves leaves."""
    n = n_leaves + 1
    return _make_graph(n, [(0, i) for i in range(1, n)])


def _make_complete(n: int) -> SparseGraph:
    """Complete graph K_n."""
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return _make_graph(n, edges)


def _make_petersen() -> SparseGraph:
    """Petersen graph (10 nodes, 15 edges)."""
    outer = [(i, (i + 1) % 5) for i in range(5)]
    inner = [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
    spokes = [(i, i + 5) for i in range(5)]
    return _make_graph(10, outer + inner + spokes)


# ======================================================================
# Section A: Triplet computation correctness
# ======================================================================


class TestTripletComputation:
    def test_single_node(self) -> None:
        g = _make_graph(1, [])
        triplets = compute_structural_triplets(g)
        assert triplets == [(0, 0, 0)]

    def test_edge(self) -> None:
        """Edge 0-1: both nodes have triplet (1, 0, 0)."""
        g = _make_graph(2, [(0, 1)])
        triplets = compute_structural_triplets(g)
        assert triplets[0] == (1, 0, 0)
        assert triplets[1] == (1, 0, 0)

    def test_path_3(self) -> None:
        """Path 0-1-2: center has (2,0,0), endpoints have (1,1,0)."""
        g = _make_path(3)
        triplets = compute_structural_triplets(g)
        assert triplets[1] == (2, 0, 0)  # center
        assert triplets[0] == (1, 1, 0)  # endpoint
        assert triplets[2] == (1, 1, 0)  # endpoint

    def test_triangle(self) -> None:
        """Triangle: all nodes have triplet (2, 0, 0)."""
        g = _make_graph(3, [(0, 1), (1, 2), (0, 2)])
        triplets = compute_structural_triplets(g)
        assert all(t == (2, 0, 0) for t in triplets)

    def test_star_3(self) -> None:
        """Star K_{1,3}: center (3,0,0), leaves (1,2,0)."""
        g = _make_star(3)
        triplets = compute_structural_triplets(g)
        assert triplets[0] == (3, 0, 0)  # center
        for i in range(1, 4):
            assert triplets[i] == (1, 2, 0)  # leaves

    def test_path_5_distance_3(self) -> None:
        """Path 0-1-2-3-4: node 0 has triplet (1,1,1)."""
        g = _make_path(5)
        triplets = compute_structural_triplets(g)
        assert triplets[0] == (1, 1, 1)  # endpoint: d1=1, d2=2, d3=3
        assert triplets[4] == (1, 1, 1)  # other endpoint
        assert triplets[2] == (2, 2, 0)  # center node

    def test_cycle_6(self) -> None:
        """Cycle C_6: all nodes have triplet (2, 2, 2)."""
        g = _make_cycle(6)
        triplets = compute_structural_triplets(g)
        # C_6: each node has 2 neighbors at d=1, 2 at d=2, 1 at d=3
        assert all(t == (2, 2, 1) for t in triplets)

    def test_petersen_graph(self) -> None:
        """Petersen graph: all nodes have triplet (3, 6, 0) — diameter 2."""
        g = _make_petersen()
        triplets = compute_structural_triplets(g)
        # Petersen is vertex-transitive, 3-regular, diameter 2
        assert all(t == (3, 6, 0) for t in triplets)

    def test_triplet_relabeling_invariance(self) -> None:
        """Triplet multiset is invariant under relabeling."""
        # Star with center=0
        g1 = _make_star(3)
        t1 = compute_structural_triplets(g1)

        # Star with center=2
        g2 = _make_graph(4, [(2, 0), (2, 1), (2, 3)])
        t2 = compute_structural_triplets(g2)

        assert sorted(t1) == sorted(t2)

    def test_directed_triplet(self) -> None:
        """Directed path 0->1->2: triplets account for directed neighbors."""
        g = _make_graph(3, [(0, 1), (1, 2)], directed=True)
        triplets = compute_structural_triplets(g)
        # In directed graph, neighbors(0)={1}, neighbors(1)={2}, neighbors(2)={}
        assert triplets[0] == (1, 1, 0)
        assert triplets[1] == (1, 0, 0)
        assert triplets[2] == (0, 0, 0)


# ======================================================================
# Section B: Pruned is valid complete invariant (CRITICAL)
# ======================================================================


def _assert_valid_pruned(g: SparseGraph) -> str:
    """Assert pruned_canonical_string produces a valid encoding that round-trips."""
    w = pruned_canonical_string(g)
    stg = StringToGraph(w, directed_graph=g.directed())
    g2, _ = stg.run()
    assert g.node_count() == g2.node_count(), (
        f"Node count mismatch: {g.node_count()} vs {g2.node_count()}"
    )
    assert g.logical_edge_count() == g2.logical_edge_count(), (
        f"Edge count mismatch: {g.logical_edge_count()} vs {g2.logical_edge_count()}"
    )
    assert g.is_isomorphic(g2), f"Round-trip failed for pruned string '{w}'"
    return w


class TestPrunedCompleteInvariant:
    """The pruned canonical string is a valid complete graph invariant.

    Note: the pruned string may differ from the exhaustive canonical string
    because the pruning restricts which candidates are explored at V/v
    branch points. Both are complete invariants but define different
    canonical forms. The pruned version may produce longer strings on
    heterogeneous graphs.
    """

    def test_path_3(self) -> None:
        _assert_valid_pruned(_make_path(3))

    def test_path_4(self) -> None:
        _assert_valid_pruned(_make_path(4))

    def test_path_5(self) -> None:
        _assert_valid_pruned(_make_path(5))

    def test_triangle(self) -> None:
        _assert_valid_pruned(_make_graph(3, [(0, 1), (1, 2), (0, 2)]))

    def test_cycle_4(self) -> None:
        _assert_valid_pruned(_make_cycle(4))

    def test_cycle_5(self) -> None:
        _assert_valid_pruned(_make_cycle(5))

    def test_star_3(self) -> None:
        _assert_valid_pruned(_make_star(3))

    def test_star_4(self) -> None:
        _assert_valid_pruned(_make_star(4))

    def test_star_5(self) -> None:
        _assert_valid_pruned(_make_star(5))

    def test_k4(self) -> None:
        _assert_valid_pruned(_make_complete(4))

    def test_k5(self) -> None:
        _assert_valid_pruned(_make_complete(5))

    def test_diamond(self) -> None:
        """K_4 minus one edge."""
        _assert_valid_pruned(_make_graph(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]))

    def test_house(self) -> None:
        """House graph: square + triangle on top."""
        _assert_valid_pruned(_make_graph(5, [(0, 1), (1, 2), (2, 3), (3, 0), (2, 4), (3, 4)]))

    def test_bull(self) -> None:
        """Bull graph: triangle 0-1-2 with pendant edges 0-3 and 1-4."""
        _assert_valid_pruned(_make_graph(5, [(0, 1), (1, 2), (0, 2), (0, 3), (1, 4)]))

    def test_petersen(self) -> None:
        _assert_valid_pruned(_make_petersen())

    def test_directed_triangle(self) -> None:
        _assert_valid_pruned(_make_graph(3, [(0, 1), (1, 2), (2, 0)], directed=True))

    def test_directed_diamond(self) -> None:
        _assert_valid_pruned(_make_graph(4, [(0, 1), (0, 2), (1, 3), (2, 3)], directed=True))

    def test_directed_cycle_4(self) -> None:
        _assert_valid_pruned(_make_graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)], directed=True))

    def test_self_loop(self) -> None:
        _assert_valid_pruned(_make_graph(2, [(0, 1), (0, 0)]))

    def test_empty_graph(self) -> None:
        g = SparseGraph(0, directed_graph=False)
        assert pruned_canonical_string(g) == ""

    def test_single_node(self) -> None:
        assert pruned_canonical_string(_make_graph(1, [])) == ""

    def test_two_nodes(self) -> None:
        _assert_valid_pruned(_make_graph(2, [(0, 1)]))

    def test_vertex_transitive_matches_exhaustive(self) -> None:
        """On vertex-transitive graphs (all nodes have same triplet), pruning
        has no effect and the result matches the exhaustive canonical string."""
        for g in [
            _make_graph(3, [(0, 1), (1, 2), (0, 2)]),  # triangle
            _make_cycle(5),
            _make_complete(4),
            _make_petersen(),
        ]:
            assert pruned_canonical_string(g) == canonical_string(g)


# ======================================================================
# Section C: Invariance under relabeling
# ======================================================================


class TestPrunedInvariance:
    def test_triangle_relabeling(self) -> None:
        g1 = _make_graph(3, [(0, 1), (1, 2), (0, 2)])
        g2 = _make_graph(3, [(0, 2), (2, 1), (0, 1)])
        assert pruned_canonical_string(g1) == pruned_canonical_string(g2)

    def test_path_4_relabeling(self) -> None:
        """Path 0-1-2-3 vs reversed 3-2-1-0."""
        g1 = _make_path(4)
        g2 = _make_graph(4, [(3, 2), (2, 1), (1, 0)])
        assert pruned_canonical_string(g1) == pruned_canonical_string(g2)

    def test_star_relabeling(self) -> None:
        """Star center=0 vs center=3."""
        g1 = _make_star(4)
        g2 = _make_graph(5, [(3, 0), (3, 1), (3, 2), (3, 4)])
        assert pruned_canonical_string(g1) == pruned_canonical_string(g2)

    def test_petersen_relabeling(self) -> None:
        """Petersen with a different labeling."""
        g1 = _make_petersen()
        # Relabel: swap nodes 0 and 5
        outer = [(i, (i + 1) % 5) for i in range(5)]
        inner = [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
        spokes = [(i, i + 5) for i in range(5)]
        all_edges = outer + inner + spokes
        # Apply permutation: 0<->5
        perm = {0: 5, 5: 0}
        relabeled = [(perm.get(u, u), perm.get(v, v)) for u, v in all_edges]
        g2 = _make_graph(10, relabeled)
        assert pruned_canonical_string(g1) == pruned_canonical_string(g2)


# ======================================================================
# Section D: Non-isomorphic discrimination
# ======================================================================


class TestPrunedDiscrimination:
    def test_path_vs_triangle(self) -> None:
        path = _make_path(3)
        triangle = _make_graph(3, [(0, 1), (1, 2), (0, 2)])
        assert pruned_canonical_string(path) != pruned_canonical_string(triangle)

    def test_star_vs_path_4(self) -> None:
        star = _make_star(3)
        path = _make_path(4)
        assert pruned_canonical_string(star) != pruned_canonical_string(path)

    def test_cycle_4_vs_star_3(self) -> None:
        cycle = _make_cycle(4)
        star = _make_star(3)
        assert pruned_canonical_string(cycle) != pruned_canonical_string(star)


# ======================================================================
# Section E: Round-trip correctness
# ======================================================================


class TestPrunedRoundTrip:
    def test_triangle_roundtrip(self) -> None:
        g = _make_graph(3, [(0, 1), (1, 2), (0, 2)])
        w = pruned_canonical_string(g)
        stg = StringToGraph(w, directed_graph=False)
        g2, _ = stg.run()
        assert g.is_isomorphic(g2)

    def test_star_roundtrip(self) -> None:
        g = _make_star(4)
        w = pruned_canonical_string(g)
        stg = StringToGraph(w, directed_graph=False)
        g2, _ = stg.run()
        assert g.is_isomorphic(g2)

    def test_petersen_roundtrip(self) -> None:
        g = _make_petersen()
        w = pruned_canonical_string(g)
        stg = StringToGraph(w, directed_graph=False)
        g2, _ = stg.run()
        assert g.is_isomorphic(g2)


# ======================================================================
# Section F: Algorithm class interface
# ======================================================================


class TestPrunedExhaustiveAlgorithm:
    def test_name(self) -> None:
        from isalgraph.core.algorithms.pruned_exhaustive import PrunedExhaustiveG2S

        alg = PrunedExhaustiveG2S()
        assert alg.name == "pruned_exhaustive"

    def test_encode_matches(self) -> None:
        from isalgraph.core.algorithms.pruned_exhaustive import PrunedExhaustiveG2S

        alg = PrunedExhaustiveG2S()
        g = _make_graph(3, [(0, 1), (1, 2), (0, 2)])
        assert alg.encode(g) == pruned_canonical_string(g)

    def test_repr(self) -> None:
        from isalgraph.core.algorithms.pruned_exhaustive import PrunedExhaustiveG2S

        alg = PrunedExhaustiveG2S()
        assert "PrunedExhaustiveG2S" in repr(alg)


# ======================================================================
# Section G: Graph distance
# ======================================================================


class TestPrunedGraphDistance:
    def test_isomorphic_distance_zero(self) -> None:
        g1 = _make_graph(3, [(0, 1), (1, 2), (0, 2)])
        g2 = _make_graph(3, [(0, 2), (2, 1), (0, 1)])
        assert pruned_graph_distance(g1, g2) == 0

    def test_non_isomorphic_positive(self) -> None:
        path = _make_path(3)
        triangle = _make_graph(3, [(0, 1), (1, 2), (0, 2)])
        assert pruned_graph_distance(path, triangle) > 0


# ======================================================================
# Section H: Performance sanity check
# ======================================================================


class TestPrunedPerformance:
    def test_tree_faster_than_exhaustive(self) -> None:
        """For a tree (7 nodes), pruned should be very fast (m=1 at most steps)."""
        # Binary tree: 0->{1,2}, 1->{3,4}, 2->{5,6}
        g = _make_graph(
            7,
            [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)],
        )

        t0 = time.perf_counter()
        w_pruned = pruned_canonical_string(g)
        t_pruned = time.perf_counter() - t0

        t0 = time.perf_counter()
        canonical_string(g)
        t_exhaustive = time.perf_counter() - t0

        # Pruned must produce a valid encoding (round-trips)
        _assert_valid_pruned(g)
        # Pruned should be at most as slow (typically faster)
        # We give generous margin to avoid flaky tests
        assert t_pruned <= t_exhaustive * 3 + 0.01

    def test_complete_graph_same_time(self) -> None:
        """For K_5 (symmetric), pruning may not help. Both should agree."""
        g = _make_complete(5)

        w_pruned = pruned_canonical_string(g)
        w_exhaustive = canonical_string(g)
        assert w_pruned == w_exhaustive

    def test_disconnected_raises(self) -> None:
        """Disconnected graph should raise ValueError."""
        g = SparseGraph(3, directed_graph=False)
        for _ in range(3):
            g.add_node()
        g.add_edge(0, 1)
        # Node 2 is disconnected

        with pytest.raises(ValueError, match="reach all"):
            pruned_canonical_string(g)
