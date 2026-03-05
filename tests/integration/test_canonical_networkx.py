"""Cross-validate canonical string invariance using NetworkX.

Generates random graphs, randomly relabels them to produce isomorphic
copies, and verifies that canonical_string() produces identical output.
"""

from __future__ import annotations

import random

import pytest

nx = pytest.importorskip("networkx")

from isalgraph.adapters.networkx_adapter import NetworkXAdapter
from isalgraph.core.canonical import canonical_string


def _make_relabeled_pair(
    g: nx.Graph,
    seed: int,
) -> tuple[nx.Graph, nx.Graph]:
    """Return (g, g') where g' is a random relabeling of g."""
    rng = random.Random(seed)
    nodes = list(g.nodes())
    perm = list(nodes)
    rng.shuffle(perm)
    mapping = dict(zip(nodes, perm))
    g2 = nx.relabel_nodes(g, mapping)
    return g, g2


class TestCanonicalInvarianceNx:
    """Canonical string invariance with random relabelings."""

    @pytest.mark.parametrize("seed", range(10))
    def test_random_tree(self, seed: int) -> None:
        g = nx.random_labeled_tree(8, seed=seed)
        g1, g2 = _make_relabeled_pair(g, seed + 1000)
        adapter = NetworkXAdapter()
        sg1 = adapter.from_external(g1, directed=False)
        sg2 = adapter.from_external(g2, directed=False)
        assert canonical_string(sg1) == canonical_string(sg2)

    @pytest.mark.parametrize("n", range(3, 8))
    def test_cycle(self, n: int) -> None:
        g = nx.cycle_graph(n)
        g1, g2 = _make_relabeled_pair(g, n * 7)
        adapter = NetworkXAdapter()
        sg1 = adapter.from_external(g1, directed=False)
        sg2 = adapter.from_external(g2, directed=False)
        assert canonical_string(sg1) == canonical_string(sg2)

    @pytest.mark.parametrize("n", range(2, 7))
    def test_complete(self, n: int) -> None:
        g = nx.complete_graph(n)
        g1, g2 = _make_relabeled_pair(g, n * 13)
        adapter = NetworkXAdapter()
        sg1 = adapter.from_external(g1, directed=False)
        sg2 = adapter.from_external(g2, directed=False)
        assert canonical_string(sg1) == canonical_string(sg2)

    def test_petersen(self) -> None:
        g = nx.petersen_graph()
        g1, g2 = _make_relabeled_pair(g, 42)
        adapter = NetworkXAdapter()
        sg1 = adapter.from_external(g1, directed=False)
        sg2 = adapter.from_external(g2, directed=False)
        assert canonical_string(sg1) == canonical_string(sg2)


class TestCanonicalDiscriminationNx:
    """Non-isomorphic graphs must have different canonical strings."""

    def test_path_vs_cycle_5(self) -> None:
        adapter = NetworkXAdapter()
        path = adapter.from_external(nx.path_graph(5), directed=False)
        cycle = adapter.from_external(nx.cycle_graph(5), directed=False)
        assert canonical_string(path) != canonical_string(cycle)

    def test_star_vs_path_5(self) -> None:
        adapter = NetworkXAdapter()
        star = adapter.from_external(nx.star_graph(4), directed=False)
        path = adapter.from_external(nx.path_graph(5), directed=False)
        assert canonical_string(star) != canonical_string(path)

    def test_k4_vs_cycle_4(self) -> None:
        adapter = NetworkXAdapter()
        k4 = adapter.from_external(nx.complete_graph(4), directed=False)
        c4 = adapter.from_external(nx.cycle_graph(4), directed=False)
        assert canonical_string(k4) != canonical_string(c4)
