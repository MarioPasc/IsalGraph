"""Unit tests for the cohort audit (T-01).

The audit replaces four measurement scripts that no longer exist. What these tests
protect is the arithmetic a reviewer will read:

* the **two density conventions** -- the cohort table has always reported the mean
  of per-graph densities, not the density of the mean graph, and the two differ by
  around 10 % on this cohort;
* the **discarded side**, per discard reason, which no previous script emitted;
* the **size-bias ratio**, the figure ``data.md`` section 3 quotes as 1.92x / 2.27x /
  1.58x;
* the **Suite-1 lock**, which must fail loudly rather than quietly re-baseline.
"""

from __future__ import annotations

import networkx as nx
import pytest

from benchmarks.eval_setup.cohort_audit import (
    NO_N_MAX,
    SUITE1_EXPECTED,
    SUITE1_EXPECTED_TOTAL_PAIRS,
    SUITE1_KEYS,
    SUITE1_N_MAX,
    SUITE2_KEYS,
    CohortRow,
    GraphStats,
    _reason_bucket,
    audit,
    check_suite1,
    to_markdown,
)

pytestmark = pytest.mark.unit


def _path(n: int) -> nx.Graph:
    """Connected path on ``n`` nodes: ``n - 1`` edges."""
    return nx.path_graph(n)


def _disconnected(n: int) -> nx.Graph:
    """``n`` isolated nodes, no edges."""
    g = nx.Graph()
    g.add_nodes_from(range(n))
    return g


class TestGraphStats:
    def test_empty_is_all_zero(self) -> None:
        assert GraphStats.from_graphs([]) == GraphStats()

    def test_hand_computed_summary(self) -> None:
        # Triangle (n=3, m=3, density 1.0) and path on 3 (n=3, m=2, density 2/3).
        stats = GraphStats.from_graphs([nx.complete_graph(3), _path(3)])

        assert stats.count == 2
        assert stats.n_mean == pytest.approx(3.0)
        assert stats.n_median == pytest.approx(3.0)
        assert stats.n_min == 3
        assert stats.n_max == 3
        assert stats.m_mean == pytest.approx(2.5)
        assert stats.density_mean == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)

    def test_the_two_density_conventions_differ(self) -> None:
        """Averaging per-graph density is not the density of the average graph."""
        graphs = [nx.complete_graph(3), _path(10)]
        stats = GraphStats.from_graphs(graphs)

        # per-graph: (1.0 + 9/45) / 2 = 0.6
        assert stats.density_mean == pytest.approx(0.6)
        # aggregate: n_mean 6.5, m_mean 6.0 -> 2*6/(6.5*5.5)
        assert stats.density_aggregate == pytest.approx(2 * 6.0 / (6.5 * 5.5))
        assert stats.density_mean != pytest.approx(stats.density_aggregate)

    def test_single_node_graph_excluded_from_density_only(self) -> None:
        """Density is undefined at n < 2; the node count still counts."""
        stats = GraphStats.from_graphs([_disconnected(1), nx.complete_graph(3)])

        assert stats.count == 2
        assert stats.n_min == 1
        assert stats.density_mean == pytest.approx(1.0)


class TestReasonBucket:
    @pytest.mark.parametrize(
        ("reason", "expected"),
        [
            ("trivial (1 nodes < min_nodes=2)", "trivial"),
            ("too_large (18 nodes > n_max=12)", "too_large"),
            ("disconnected", "disconnected"),
            ("something else", "other"),
        ],
    )
    def test_categories(self, reason: str, expected: str) -> None:
        assert _reason_bucket(reason) == expected


class TestAudit:
    def test_splits_retained_and_discarded_by_reason(self) -> None:
        graphs = [
            _path(3),  # kept
            _path(5),  # kept
            _disconnected(1),  # trivial
            _disconnected(4),  # disconnected
            _path(20),  # too large under SUITE1_N_MAX
        ]
        ids = [f"g{i}" for i in range(len(graphs))]

        row = audit("t", "suite1", "test", graphs, ids, SUITE1_N_MAX)

        assert (row.n_raw, row.n_kept) == (5, 2)
        assert row.n_pairs == 1  # C(2, 2)
        assert row.keep_pct == pytest.approx(40.0)
        assert set(row.discarded_by_reason) == {"trivial", "disconnected", "too_large"}
        assert row.discarded_by_reason["too_large"].n_max == 20

    def test_size_bias_is_discarded_over_retained_mean(self) -> None:
        graphs = [_path(4), _path(4), _disconnected(8)]
        row = audit("t", "suite2", "test", graphs, ["a", "b", "c"], NO_N_MAX)

        assert row.retained.n_mean == pytest.approx(4.0)
        assert row.discarded.n_mean == pytest.approx(8.0)
        assert row.size_bias == pytest.approx(2.0)

    def test_no_n_max_keeps_a_graph_suite_one_would_drop(self) -> None:
        graphs = [_path(50)]
        suite1 = audit("t", "suite1", "test", graphs, ["a"], SUITE1_N_MAX)
        suite2 = audit("t", "suite2", "test", graphs, ["a"], NO_N_MAX)

        assert suite1.n_kept == 0
        assert suite2.n_kept == 1
        assert suite2.retained.n_max == 50

    def test_nothing_discarded_gives_zero_bias_not_a_crash(self) -> None:
        row = audit("t", "suite2", "test", [_path(3)], ["a"], NO_N_MAX)

        assert row.discarded.count == 0
        assert row.size_bias == pytest.approx(0.0)

    def test_pair_count_is_exactly_n_choose_two(self) -> None:
        graphs = [_path(3) for _ in range(10)]
        row = audit("t", "suite2", "test", graphs, [str(i) for i in range(10)], NO_N_MAX)

        assert row.n_pairs == 45


class TestSuiteDefinitions:
    def test_suite2_extends_suite1(self) -> None:
        assert SUITE2_KEYS[: len(SUITE1_KEYS)] == SUITE1_KEYS
        assert len(SUITE1_KEYS) == 5
        assert len(SUITE2_KEYS) == 10

    def test_locked_totals_are_internally_consistent(self) -> None:
        assert sum(p for _, p in SUITE1_EXPECTED.values()) == SUITE1_EXPECTED_TOTAL_PAIRS
        assert set(SUITE1_EXPECTED) == set(SUITE1_KEYS)


def _locked_rows() -> list[CohortRow]:
    return [
        CohortRow(
            key=key,
            suite="suite1",
            source="test",
            n_raw=kept,
            n_kept=kept,
            keep_pct=100.0,
            n_pairs=pairs,
        )
        for key, (kept, pairs) in SUITE1_EXPECTED.items()
    ]


class TestSuiteOneLock:
    def test_locked_cohort_passes(self) -> None:
        assert check_suite1(_locked_rows()) == []

    def test_one_graph_fewer_is_caught(self) -> None:
        rows = _locked_rows()
        rows[0].n_kept -= 1

        problems = check_suite1(rows)

        assert any("kept" in p for p in problems)

    def test_pair_count_drift_is_caught_independently(self) -> None:
        """The pair count is checked on its own, not derived from n_kept."""
        rows = _locked_rows()
        rows[-1].n_pairs += 1

        problems = check_suite1(rows)

        assert any("pairs" in p for p in problems)
        assert any("total pairs" in p for p in problems)


def test_markdown_carries_both_sides_and_a_total() -> None:
    rows = [audit("d", "suite2", "test", [_path(3), _disconnected(9)], ["a", "b"], NO_N_MAX)]

    table = to_markdown(rows, "T")

    assert "### T" in table
    assert "disc. n̄" in table
    assert "**Total**" in table
