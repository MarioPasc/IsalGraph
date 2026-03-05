"""Integration tests for igraph adapter."""

from __future__ import annotations

import pytest

ig = pytest.importorskip("igraph")

from isalgraph.adapters.igraph_adapter import IGraphAdapter
from isalgraph.core.graph_to_string import GraphToString
from isalgraph.core.string_to_graph import StringToGraph


@pytest.fixture()
def adapter() -> IGraphAdapter:
    return IGraphAdapter()


class TestIGraphAdapterConversion:
    """Test from_external / to_external."""

    def test_undirected_path(self, adapter: IGraphAdapter) -> None:
        g = ig.Graph.Ring(5, circular=False)
        sg = adapter.from_external(g, directed=False)
        assert sg.node_count() == 5
        assert sg.logical_edge_count() == 4

        g2 = adapter.to_external(sg)
        assert g.isomorphic(g2)

    def test_directed_cycle(self, adapter: IGraphAdapter) -> None:
        g = ig.Graph.Ring(5, directed=True)
        sg = adapter.from_external(g, directed=True)
        assert sg.node_count() == 5
        assert sg.logical_edge_count() == 5

        g2 = adapter.to_external(sg)
        assert g.isomorphic(g2)

    def test_complete_graph(self, adapter: IGraphAdapter) -> None:
        g = ig.Graph.Full(5)
        sg = adapter.from_external(g, directed=False)
        assert sg.node_count() == 5
        assert sg.logical_edge_count() == 10

        g2 = adapter.to_external(sg)
        assert g.isomorphic(g2)


class TestIGraphRoundTrip:
    """Cross-validate round-trip via igraph isomorphism."""

    def test_petersen(self, adapter: IGraphAdapter) -> None:
        g = ig.Graph.Famous("Petersen")
        sg = adapter.from_external(g, directed=False)
        gts = GraphToString(sg)
        w, _ = gts.run(0)

        stg = StringToGraph(w, directed_graph=False)
        sg2, _ = stg.run()
        g2 = adapter.to_external(sg2)

        assert g.isomorphic(g2)

    @pytest.mark.parametrize("n", range(3, 8))
    def test_cycle(self, adapter: IGraphAdapter, n: int) -> None:
        g = ig.Graph.Ring(n)
        sg = adapter.from_external(g, directed=False)
        gts = GraphToString(sg)
        w, _ = gts.run(0)

        stg = StringToGraph(w, directed_graph=False)
        sg2, _ = stg.run()
        g2 = adapter.to_external(sg2)

        assert g.isomorphic(g2)
