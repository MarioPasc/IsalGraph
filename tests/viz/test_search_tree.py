"""Tests for the canonical-search-tree schematic.

The load-bearing test is :func:`test_enumerator_agrees_with_canonical_string`.
The figure re-implements the search by replay rather than backtracking,
so without that check the schematic could silently drift from the
algorithm it claims to depict.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from isalgraph.core.canonical import canonical_string  # noqa: E402
from isalgraph.core.sparse_graph import SparseGraph  # noqa: E402
from isalgraph.viz.figures import build_example_graph  # noqa: E402
from isalgraph.viz.search_tree import (  # noqa: E402
    _canonical_from,
    canonical_search_tree_figure,
    enumerate_search_tree,
)
from isalgraph.viz.style import save_figure  # noqa: E402


def _path_graph(n: int) -> SparseGraph:
    g = SparseGraph(n, False)
    for _ in range(n):
        g.add_node()
    for i in range(n - 1):
        g.add_edge(i, i + 1)
    return g


def _cycle(n: int) -> SparseGraph:
    g = _path_graph(n)
    g.add_edge(n - 1, 0)
    return g


def _star(n: int) -> SparseGraph:
    g = SparseGraph(n, False)
    for _ in range(n):
        g.add_node()
    for i in range(1, n):
        g.add_edge(0, i)
    return g


GRAPHS = {
    "path4": _path_graph(4),
    "path6": _path_graph(6),
    "cycle5": _cycle(5),
    "cycle6": _cycle(6),
    "star5": _star(5),
    "example7": build_example_graph(),
}


@pytest.mark.parametrize("name", sorted(GRAPHS), ids=sorted(GRAPHS))
def test_enumerator_agrees_with_canonical_string(name: str) -> None:
    """Replay enumeration must reproduce ``canonical.canonical_string``.

    ``_canonical_from`` walks every V/v choice by replaying from the root;
    ``canonical_string`` walks the same space by backtracking with undo.
    Taking the ``(len, lex)`` minimum over starting nodes must give the
    same answer, or the schematic depicts a different algorithm.
    """
    graph = GRAPHS[name]
    best = min(
        (_canonical_from(graph, v) for v in range(graph.node_count())),
        key=lambda w: (len(w), w),
    )
    assert best == canonical_string(graph)


def test_branch_points_are_only_starting_nodes_and_vv_choices() -> None:
    """No node may fan out on a C/c step, nor on a single-candidate V/v."""
    tree = enumerate_search_tree(build_example_graph(), max_depth=3)
    for node in tree.nodes:
        if len(node.children) > 1:
            for child in tree.children_of(node.index):
                assert child.step is not None
                assert child.step.op in ("V", "v"), (
                    f"fan-out on a {child.step.op} step; only V/v may branch"
                )
                assert child.step.n_candidates > 1


def test_sibling_count_matches_the_candidate_set() -> None:
    """Each fan-out width must equal the recorded candidate-set size."""
    tree = enumerate_search_tree(build_example_graph(), max_depth=2)
    for node in tree.nodes:
        children = tree.children_of(node.index)
        if len(children) > 1:
            assert len({c.step.n_candidates for c in children if c.step}) == 1
            assert len(children) == children[0].step.n_candidates  # type: ignore[union-attr]


def test_roots_are_the_viable_starting_nodes() -> None:
    graph = build_example_graph()
    tree = enumerate_search_tree(graph, max_depth=1)
    assert len(tree.roots) == graph.node_count()  # connected: every node is viable
    assert {tree.nodes[r].start_node for r in tree.roots} == set(range(graph.node_count()))


def test_max_roots_caps_the_fan_and_keeps_the_canonical_root() -> None:
    graph = build_example_graph()
    tree = enumerate_search_tree(graph, max_depth=2, max_roots=3)
    assert len(tree.roots) == 3
    assert any(node.optimal for node in tree.nodes), "canonical root was dropped"


def test_exactly_one_canonical_path_is_marked() -> None:
    """Ties must not light up sibling branches; only one path is marked."""
    tree = enumerate_search_tree(build_example_graph(), max_depth=3, max_roots=3)
    marked = [n for n in tree.nodes if n.optimal]
    assert marked, "no canonical path marked"
    # One marked node per depth level: a path, not a subtree.
    depths = [n.depth for n in marked]
    assert len(depths) == len(set(depths))
    for node in marked:
        assert tree.canonical.startswith(node.prefix)


def test_depth_budget_is_respected() -> None:
    for depth in (1, 2, 3):
        tree = enumerate_search_tree(build_example_graph(), max_depth=depth, max_roots=2)
        assert max(n.depth for n in tree.nodes) <= depth


def test_disconnected_graph_is_rejected() -> None:
    graph = SparseGraph(4, False)
    for _ in range(4):
        graph.add_node()
    graph.add_edge(0, 1)
    with pytest.raises(ValueError, match="reaches every other node"):
        enumerate_search_tree(graph)


def test_search_tree_figure_renders(tmp_path: Path) -> None:
    fig = canonical_search_tree_figure(build_example_graph(), max_depth=3, max_roots=3)
    written = save_figure(fig, tmp_path / "tree", formats=("png",))
    for path in written:
        assert path.exists() and path.stat().st_size > 1000
    plt.close(fig)
