"""Integration tests for NetworkX adapter.

Cross-validates IsalGraph round-trip correctness using NetworkX's
``nx.is_isomorphic`` (VF2 algorithm), which serves as an independent
oracle for graph isomorphism.
"""

from __future__ import annotations

import pytest

nx = pytest.importorskip("networkx")

from isalgraph.adapters.networkx_adapter import NetworkXAdapter
from isalgraph.core.graph_to_string import GraphToString
from isalgraph.core.string_to_graph import StringToGraph


@pytest.fixture()
def adapter() -> NetworkXAdapter:
    return NetworkXAdapter()


# ======================================================================
# Adapter conversion tests
# ======================================================================


class TestNetworkXAdapterConversion:
    """Test from_external / to_external correctness."""

    def test_undirected_roundtrip(self, adapter: NetworkXAdapter) -> None:
        g = nx.path_graph(5)
        sg = adapter.from_external(g, directed=False)
        assert sg.node_count() == 5
        assert sg.logical_edge_count() == 4

        g2 = adapter.to_external(sg)
        assert nx.is_isomorphic(g, g2)

    def test_directed_roundtrip(self, adapter: NetworkXAdapter) -> None:
        g = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
        sg = adapter.from_external(g, directed=True)
        assert sg.node_count() == 3
        assert sg.logical_edge_count() == 3

        g2 = adapter.to_external(sg)
        assert nx.is_isomorphic(g, g2)

    def test_complete_graph(self, adapter: NetworkXAdapter) -> None:
        g = nx.complete_graph(5)
        sg = adapter.from_external(g, directed=False)
        assert sg.node_count() == 5
        assert sg.logical_edge_count() == 10  # C(5,2) = 10

        g2 = adapter.to_external(sg)
        assert nx.is_isomorphic(g, g2)

    def test_single_node(self, adapter: NetworkXAdapter) -> None:
        g = nx.Graph()
        g.add_node(0)
        sg = adapter.from_external(g, directed=False)
        assert sg.node_count() == 1
        assert sg.logical_edge_count() == 0

    def test_string_roundtrip(self, adapter: NetworkXAdapter) -> None:
        """to_isalgraph_string -> from_isalgraph_string round-trip."""
        g = nx.cycle_graph(4)
        s = adapter.to_isalgraph_string(g, directed=False)
        g2 = adapter.from_isalgraph_string(s, directed=False)
        assert nx.is_isomorphic(g, g2)

    def test_string_roundtrip_with_algorithm(self, adapter: NetworkXAdapter) -> None:
        """to_isalgraph_string with explicit algorithm parameter."""
        from isalgraph.core.algorithms import GreedyMinG2S

        g = nx.cycle_graph(5)
        algo = GreedyMinG2S()
        s = adapter.to_isalgraph_string(g, directed=False, algorithm=algo)
        g2 = adapter.from_isalgraph_string(s, directed=False)
        assert nx.is_isomorphic(g, g2)


# ======================================================================
# Cross-validation: NetworkX isomorphism vs IsalGraph round-trip
# ======================================================================


def _nx_cross_validate(nxg: nx.Graph | nx.DiGraph, directed: bool) -> None:
    """Full round-trip cross-validated with nx.is_isomorphic."""
    adapter = NetworkXAdapter()
    sg1 = adapter.from_external(nxg, directed=directed)
    gts = GraphToString(sg1)
    w, _ = gts.run(0)

    stg = StringToGraph(w, directed_graph=directed)
    sg2, _ = stg.run()

    nxg2 = adapter.to_external(sg2)

    assert nx.is_isomorphic(nxg, nxg2), (
        f"NX cross-validation failed.\n"
        f"  Original: {nxg.number_of_nodes()} nodes, {nxg.number_of_edges()} edges\n"
        f"  Recovered: {nxg2.number_of_nodes()} nodes, {nxg2.number_of_edges()} edges\n"
        f"  String: {w!r}"
    )


class TestCrossValidation:
    """Cross-validate with diverse NetworkX graph families."""

    def test_path_graph(self) -> None:
        for n in range(2, 8):
            _nx_cross_validate(nx.path_graph(n), directed=False)

    def test_cycle_graph(self) -> None:
        for n in range(3, 8):
            _nx_cross_validate(nx.cycle_graph(n), directed=False)

    def test_complete_graph(self) -> None:
        for n in range(2, 7):
            _nx_cross_validate(nx.complete_graph(n), directed=False)

    def test_star_graph(self) -> None:
        for n in range(2, 8):
            _nx_cross_validate(nx.star_graph(n), directed=False)

    def test_wheel_graph(self) -> None:
        for n in range(4, 8):
            _nx_cross_validate(nx.wheel_graph(n), directed=False)

    def test_grid_graph(self) -> None:
        g = nx.grid_2d_graph(3, 3)
        _nx_cross_validate(g, directed=False)

    def test_petersen_graph(self) -> None:
        _nx_cross_validate(nx.petersen_graph(), directed=False)

    def test_complete_bipartite(self) -> None:
        _nx_cross_validate(nx.complete_bipartite_graph(3, 3), directed=False)

    def test_tree(self) -> None:
        g = nx.random_labeled_tree(10, seed=42)
        _nx_cross_validate(g, directed=False)

    def test_barabasi_albert(self) -> None:
        g = nx.barabasi_albert_graph(15, 2, seed=42)
        _nx_cross_validate(g, directed=False)

    def test_watts_strogatz(self) -> None:
        g = nx.watts_strogatz_graph(12, 4, 0.3, seed=42)
        _nx_cross_validate(g, directed=False)

    def test_erdos_renyi(self) -> None:
        g = nx.erdos_renyi_graph(10, 0.3, seed=42)
        # Extract largest connected component.
        largest_cc = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest_cc).copy()
        if g.number_of_nodes() >= 2:
            g = nx.convert_node_labels_to_integers(g)
            _nx_cross_validate(g, directed=False)

    @pytest.mark.parametrize("seed", range(20))
    def test_random_gnp_undirected(self, seed: int) -> None:
        g = nx.gnp_random_graph(8, 0.4, seed=seed)
        # Extract largest connected component.
        if g.number_of_nodes() == 0:
            return
        largest_cc = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest_cc).copy()
        if g.number_of_nodes() >= 2:
            g = nx.convert_node_labels_to_integers(g)
            _nx_cross_validate(g, directed=False)

    @pytest.mark.parametrize("seed", range(20))
    def test_random_gnp_directed(self, seed: int) -> None:
        g = nx.gnp_random_graph(8, 0.3, seed=seed, directed=True)
        # For directed graphs, extract nodes reachable from node 0.
        if g.number_of_nodes() == 0:
            return
        # Use node 0 as start; extract its reachable subgraph.
        reachable = nx.descendants(g, 0) | {0}
        g = g.subgraph(reachable).copy()
        if g.number_of_nodes() >= 2:
            g = nx.convert_node_labels_to_integers(g)
            _nx_cross_validate(g, directed=True)


# ======================================================================
# All starting nodes test
# ======================================================================


def test_all_starting_nodes_produce_isomorphic_output() -> None:
    """G2S from any starting node should produce a graph isomorphic to the input."""
    g = nx.petersen_graph()
    adapter = NetworkXAdapter()
    sg = adapter.from_external(g, directed=False)

    for start_node in range(sg.node_count()):
        gts = GraphToString(sg)
        w, _ = gts.run(start_node)
        stg = StringToGraph(w, directed_graph=False)
        sg2, _ = stg.run()
        nxg2 = adapter.to_external(sg2)
        assert nx.is_isomorphic(g, nxg2), f"Failed for start_node={start_node}, string={w!r}"
