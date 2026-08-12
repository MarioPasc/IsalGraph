"""Unit tests for ged_bounds -- the independent GED bracket.

These tests are validation gate 2's guarantee that the cross-check
implementation still exists and still holds its bounds. The gate previously
named a script that was never committed; a defect the test suite could not
catch because there was no test.

The invariants tested are the ones the revision leans on: the lower bound
never exceeds the true distance, the upper bound never falls below it, both
collapse to zero on identical graphs, and the bracket reproduces the GEDLIB
smoke test verified on Picasso.
"""

from __future__ import annotations

import networkx as nx
import pytest

from benchmarks.eval_setup.ged_bounds import (
    GRAPHEDX_COSTS,
    UNIT_COSTS,
    EditCosts,
    GEDBoundsError,
    bipartite_upper_bound,
    branch_lower_bound,
    exact_ged,
    ged_bracket,
    induced_edit_cost,
)


def _connected_gnp(n: int, p: float, seed: int) -> nx.Graph | None:
    """Return a connected G(n, p) sample, or None if it is disconnected."""
    g = nx.gnp_random_graph(n, p, seed=seed)
    return g if g.number_of_nodes() >= 2 and nx.is_connected(g) else None


# Small connected pairs, deterministic, cheap enough for exact A*.
_PAIRS = [
    (a, b)
    for a, b in (
        (_connected_gnp(n1, 0.45, s), _connected_gnp(n2, 0.55, s + 5000))
        for s, (n1, n2) in enumerate(
            [(3, 4), (4, 4), (4, 5), (5, 5), (5, 6), (6, 6), (3, 6), (6, 4), (5, 3), (6, 5)]
        )
    )
    if a is not None and b is not None
]


class TestCostModel:
    """The cost model must round-trip to GEDLIB's argument order."""

    def test_unit_costs_are_d6(self) -> None:
        """UNIT_COSTS is statistics.md D6, GEDLIB [1, 1, 0, 1, 1, 0]."""
        assert UNIT_COSTS.as_gedlib_constant() == [1.0, 1.0, 0.0, 1.0, 1.0, 0.0]

    def test_graphedx_costs_charge_nothing_for_nodes(self) -> None:
        """The T-03 agreement gate runs under [0, 0, 0, 1, 1, 0], not D6."""
        assert GRAPHEDX_COSTS.as_gedlib_constant() == [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]


class TestKnownValues:
    """Values verified against GEDLIB on Picasso, 2026-08-11."""

    def test_path4_vs_cycle4(self) -> None:
        """P4 vs C4 is one edge insertion: GEDLIB reported 1.00 for all three."""
        p4, c4 = nx.path_graph(4), nx.cycle_graph(4)
        lb, ub = ged_bracket(p4, c4)
        assert exact_ged(p4, c4) == pytest.approx(1.0)
        assert lb == pytest.approx(1.0)
        assert ub == pytest.approx(1.0)

    def test_disjoint_node_counts_pay_node_operations(self) -> None:
        """Under D6 a missing node costs 1 plus its incident edges."""
        g1, g2 = nx.path_graph(3), nx.path_graph(4)
        assert exact_ged(g1, g2) == pytest.approx(2.0)

    def test_graphedx_model_makes_node_operations_free(self) -> None:
        """Under GraphEdX costs the same pair pays only for the edge."""
        g1, g2 = nx.path_graph(3), nx.path_graph(4)
        assert exact_ged(g1, g2, GRAPHEDX_COSTS) == pytest.approx(1.0)


class TestBracketValidity:
    """LB <= exact <= UB is the gate; a violation is a cost-model mismatch."""

    @pytest.mark.parametrize("g1,g2", _PAIRS)
    def test_bracket_contains_exact(self, g1: nx.Graph, g2: nx.Graph) -> None:
        """The bracket must contain the exact distance on every pair."""
        lb, ub = ged_bracket(g1, g2)
        exact = exact_ged(g1, g2)
        assert lb <= exact + 1e-9
        assert ub >= exact - 1e-9

    @pytest.mark.parametrize("g1,g2", _PAIRS)
    def test_lower_never_exceeds_upper(self, g1: nx.Graph, g2: nx.Graph) -> None:
        """The two constructions must not cross."""
        lb, ub = ged_bracket(g1, g2)
        assert lb <= ub + 1e-9

    @pytest.mark.parametrize(
        "graph",
        [nx.path_graph(5), nx.cycle_graph(6), nx.complete_graph(5), nx.star_graph(6)],
    )
    def test_identity_is_zero(self, graph: nx.Graph) -> None:
        """A graph against itself is at distance zero, certified."""
        lb, ub = ged_bracket(graph, graph.copy())
        assert lb == pytest.approx(0.0)
        assert ub == pytest.approx(0.0)

    @pytest.mark.parametrize("g1,g2", _PAIRS)
    def test_symmetry(self, g1: nx.Graph, g2: nx.Graph) -> None:
        """Both bounds must be symmetric as exposed, or the matrix is not a distance."""
        assert branch_lower_bound(g1, g2) == pytest.approx(branch_lower_bound(g2, g1))
        assert bipartite_upper_bound(g1, g2) == pytest.approx(bipartite_upper_bound(g2, g1))

    def test_raw_bipartite_heuristic_is_asymmetric(self) -> None:
        """The underlying heuristic is direction-dependent; symmetrise=True hides it.

        This is why the public bound takes the minimum of both orientations.
        GEDLIB's BIPARTITE and IPFP have the same property, so a production
        matrix filled in one orientation would not be symmetric.
        """
        found = False
        for seed in range(60):
            g1 = _connected_gnp(6, 0.45, seed)
            g2 = _connected_gnp(5, 0.55, seed + 500)
            if g1 is None or g2 is None:
                continue
            forward = bipartite_upper_bound(g1, g2, symmetrise=False)
            backward = bipartite_upper_bound(g2, g1, symmetrise=False)
            if forward != pytest.approx(backward):
                found = True
                # The symmetrised bound is the tighter of the two, and valid.
                assert bipartite_upper_bound(g1, g2) == pytest.approx(min(forward, backward))
                assert min(forward, backward) >= exact_ged(g1, g2) - 1e-9
                break
        assert found, "expected at least one asymmetric pair in the sweep"

    def test_relabelling_invariance(self) -> None:
        """Node identifiers must not change either bound."""
        g = nx.gnp_random_graph(7, 0.5, seed=11)
        h = nx.relabel_nodes(g, {n: 100 - n for n in g.nodes()})
        other = nx.cycle_graph(6)
        assert branch_lower_bound(g, other) == pytest.approx(branch_lower_bound(h, other))
        assert bipartite_upper_bound(g, other) == pytest.approx(bipartite_upper_bound(h, other))


class TestInducedEditCost:
    """The upper bound is only provable because the induced cost is exact."""

    def test_empty_mapping_deletes_and_inserts_everything(self) -> None:
        """Mapping nothing pays for every node and edge on both sides."""
        g1, g2 = nx.path_graph(3), nx.path_graph(2)
        mapping = {n: None for n in g1.nodes()}
        # 3 node deletions + 2 edge deletions + 2 node insertions + 1 edge insertion
        assert induced_edit_cost(g1, g2, mapping) == pytest.approx(8.0)

    def test_identity_mapping_costs_nothing(self) -> None:
        """A perfect mapping between identical graphs is free."""
        g = nx.cycle_graph(5)
        assert induced_edit_cost(g, g.copy(), {n: n for n in g.nodes()}) == pytest.approx(0.0)

    def test_non_injective_mapping_is_rejected(self) -> None:
        """Two nodes may not map to the same target."""
        g1, g2 = nx.path_graph(3), nx.path_graph(3)
        with pytest.raises(GEDBoundsError, match="injective"):
            induced_edit_cost(g1, g2, {0: 0, 1: 0, 2: 2})

    @pytest.mark.parametrize("g1,g2", _PAIRS)
    def test_induced_cost_is_an_upper_bound(self, g1: nx.Graph, g2: nx.Graph) -> None:
        """Any mapping's induced cost is achievable, hence >= GED."""
        nodes2 = list(g2.nodes())
        mapping = {u: (nodes2[i] if i < len(nodes2) else None) for i, u in enumerate(g1.nodes())}
        assert induced_edit_cost(g1, g2, mapping) >= exact_ged(g1, g2) - 1e-9


class TestCustomCosts:
    """The bounds must track the cost model rather than assume unit costs."""

    def test_expensive_edges_raise_both_bounds(self) -> None:
        """Scaling edge costs scales the edge contribution to the bracket."""
        g1, g2 = nx.path_graph(4), nx.cycle_graph(4)
        cheap = ged_bracket(g1, g2, UNIT_COSTS)
        dear = ged_bracket(g1, g2, EditCosts(edge_ins=10.0, edge_del=10.0))
        assert dear[0] > cheap[0]
        assert dear[1] > cheap[1]

    @pytest.mark.parametrize("g1,g2", _PAIRS[:5])
    def test_bracket_holds_under_graphedx_costs(self, g1: nx.Graph, g2: nx.Graph) -> None:
        """The gate's cost model must not break the bracket."""
        lb, ub = ged_bracket(g1, g2, GRAPHEDX_COSTS)
        exact = exact_ged(g1, g2, GRAPHEDX_COSTS)
        assert lb <= exact + 1e-9 <= ub + 2e-9
