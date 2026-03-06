"""Unit tests for GraphToString."""

from __future__ import annotations

import pytest

from isalgraph.core.graph_to_string import GraphToString, generate_pairs_sorted_by_sum
from isalgraph.core.sparse_graph import SparseGraph


class TestGeneratePairs:
    """Tests for the pair generation / sorting function."""

    def test_sort_by_absolute_sum(self) -> None:
        """Regression: original sorted by a+b instead of |a|+|b| (B2)."""
        pairs = generate_pairs_sorted_by_sum(2)
        costs = [abs(a) + abs(b) for a, b in pairs]
        assert costs == sorted(costs), "Pairs must be sorted by |a|+|b|"

    def test_first_pair_is_0_0(self) -> None:
        pairs = generate_pairs_sorted_by_sum(1)
        assert pairs[0] == (0, 0)

    def test_pair_count(self) -> None:
        """m=2 -> range [-2, 2] = 5 values per axis -> 25 pairs."""
        pairs = generate_pairs_sorted_by_sum(2)
        assert len(pairs) == 25

    def test_invalid_m(self) -> None:
        with pytest.raises(ValueError):
            generate_pairs_sorted_by_sum(0)
        with pytest.raises(ValueError):
            generate_pairs_sorted_by_sum(-1)


class TestGraphToStringBasics:
    """Basic GraphToString correctness."""

    def test_single_node(self) -> None:
        """A single-node graph produces an empty string."""
        g = SparseGraph(1, directed_graph=False)
        g.add_node()
        gts = GraphToString(g)
        s, _ = gts.run(0)
        assert s == ""

    def test_two_node_undirected(self) -> None:
        """Path 0-1 should produce 'V'."""
        g = SparseGraph(2, directed_graph=False)
        g.add_node()
        g.add_node()
        g.add_edge(0, 1)
        gts = GraphToString(g)
        s, _ = gts.run(0)
        assert "V" in s or "v" in s

    def test_invalid_initial_node(self) -> None:
        g = SparseGraph(2, directed_graph=False)
        g.add_node()
        gts = GraphToString(g)
        with pytest.raises(ValueError):
            gts.run(5)

    def test_trace_nonempty(self) -> None:
        g = SparseGraph(2, directed_graph=False)
        g.add_node()
        g.add_node()
        g.add_edge(0, 1)
        gts = GraphToString(g)
        _, trace = gts.run(0, trace=True)
        assert len(trace) >= 2  # at least initial + final


class TestGeneratePairsOrdering:
    """Detailed ordering verification for generate_pairs_sorted_by_sum."""

    def test_pair_ordering_m3(self) -> None:
        """For m=3, verify: first pair is (0,0), cost groups are monotone,
        and within each cost group pairs are sorted by (|a|, |b|) then (a, b)."""
        pairs = generate_pairs_sorted_by_sum(3)
        # First pair must be (0, 0) with cost 0
        assert pairs[0] == (0, 0)

        # Group by cost and verify monotone cost ordering
        prev_cost = -1
        for a, b in pairs:
            cost = abs(a) + abs(b)
            assert cost >= prev_cost, f"Cost decreased: {prev_cost} -> {cost} at ({a}, {b})"
            prev_cost = cost

        # Within each cost group, verify deterministic tiebreaker: (|a|, (a, b))
        from itertools import groupby

        for cost, group in groupby(pairs, key=lambda p: abs(p[0]) + abs(p[1])):
            group_list = list(group)
            expected = sorted(group_list, key=lambda p: (abs(p[0]), p))
            assert group_list == expected, (
                f"Within cost={cost}, ordering mismatch: {group_list} != {expected}"
            )

        # Total count: range [-3, 3] = 7 values per axis -> 49 pairs
        assert len(pairs) == 49


class TestGraphToStringPointerUpdate:
    """Verify that pointers are properly updated after operations (B4)."""

    def test_path_3_produces_valid_string(self) -> None:
        """0-1-2 path should produce a valid string with N/P moves."""
        g = SparseGraph(3, directed_graph=False)
        for _ in range(3):
            g.add_node()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        gts = GraphToString(g)
        s, _ = gts.run(0)
        # String should contain exactly 2 V/v instructions.
        v_count = s.count("V") + s.count("v")
        assert v_count == 2


class TestGraphToStringEdgesOnly:
    """Verify that edge-only insertion (C/c) works after all nodes are added (B3)."""

    def test_triangle_needs_C(self, triangle_undirected: SparseGraph) -> None:
        """Triangle 0-1-2-0 needs 2 V/v (for nodes 1,2) and 1 C/c (for edge 1-2 or equiv)."""
        gts = GraphToString(triangle_undirected)
        s, _ = gts.run(0)
        v_count = s.count("V") + s.count("v")
        c_count = s.count("C") + s.count("c")
        assert v_count == 2
        assert c_count == 1


class TestGraphToStringReachability:
    """Verify reachability checking (lines 332-333)."""

    def test_directed_unreachable_raises(self) -> None:
        """Directed graph where not all nodes reachable from initial_node."""
        # 0 -> 1, 2 -> 0 (node 2 not reachable from 0)
        g = SparseGraph(3, directed_graph=True)
        g.add_node()
        g.add_node()
        g.add_node()
        g.add_edge(0, 1)
        g.add_edge(2, 0)
        gts = GraphToString(g)
        with pytest.raises(ValueError, match="Unreachable nodes"):
            gts.run(0)

    def test_undirected_disconnected_raises(self) -> None:
        """Undirected disconnected graph raises."""
        g = SparseGraph(3, directed_graph=False)
        g.add_node()
        g.add_node()
        g.add_node()
        g.add_edge(0, 1)
        # node 2 is disconnected
        gts = GraphToString(g)
        with pytest.raises(ValueError, match="Unreachable nodes"):
            gts.run(0)

    def test_negative_initial_node(self) -> None:
        g = SparseGraph(2, directed_graph=False)
        g.add_node()
        gts = GraphToString(g)
        with pytest.raises(ValueError, match="out of range"):
            gts.run(-1)
