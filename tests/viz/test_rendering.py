"""Rendering tests.

Convention, copied from IsalHG because it survives matplotlib version
bumps: force the Agg backend before importing pyplot, skip optional
backends via ``importorskip``, assert that a written file exists and
exceeds a byte floor rather than hashing pixels, and always close the
figure. Image hashes break on every freetype and matplotlib release; a
size floor catches the failure that actually matters, an empty canvas.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from isalgraph.core.sparse_graph import SparseGraph  # noqa: E402
from isalgraph.core.string_to_graph import StringToGraph  # noqa: E402
from isalgraph.core.trace import graph_edges  # noqa: E402
from isalgraph.viz.cdll_view import draw_cdll_ring, draw_cdll_ring_for_snapshot  # noqa: E402
from isalgraph.viz.composite import (  # noqa: E402
    roundtrip_figure,
    single_card_figure,
    steps_figure,
)
from isalgraph.viz.graph_view import draw_graph  # noqa: E402
from isalgraph.viz.instruction_view import (  # noqa: E402
    draw_instruction_strip,
    instruction_legend_handles,
)
from isalgraph.viz.registry import available_backends  # noqa: E402
from isalgraph.viz.style import build_edge_palette, build_node_palette, save_figure  # noqa: E402

MIN_BYTES = 1000
EXAMPLE = "VNVnVCPvNC"


@pytest.fixture
def example() -> tuple[SparseGraph, object]:
    """Return the worked-example graph and its S2G trace."""
    graph, trace = StringToGraph(EXAMPLE, directed_graph=False).run_with_trace()
    return graph, trace


def _assert_written(paths: list[Path]) -> None:
    for path in paths:
        assert path.exists(), path
        assert path.stat().st_size > MIN_BYTES, f"{path} is {path.stat().st_size} bytes"


# ---------------------------------------------------------------------------
# Individual views
# ---------------------------------------------------------------------------


def test_instruction_strip_renders(tmp_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 1))
    draw_instruction_strip(ax, EXAMPLE, current_idx=4, axis_width_inches=4.0)
    _assert_written(save_figure(fig, tmp_path / "strip", formats=("png",)))
    plt.close(fig)


def test_empty_instruction_strip_does_not_raise() -> None:
    fig, ax = plt.subplots()
    draw_instruction_strip(ax, "", current_idx=0)
    plt.close(fig)


def test_instruction_legend_has_one_handle_per_operation_plus_pointers() -> None:
    assert len(instruction_legend_handles(include_pointers=False)) == 4
    assert len(instruction_legend_handles(include_pointers=True)) == 6


def test_cdll_ring_renders_from_a_snapshot(tmp_path: Path, example: tuple) -> None:
    _, trace = example
    fig, ax = plt.subplots(figsize=(3, 3))
    draw_cdll_ring_for_snapshot(ax, trace.snapshots[-1])
    _assert_written(save_figure(fig, tmp_path / "ring", formats=("png",)))
    plt.close(fig)


def test_legacy_cdll_ring_signature_still_works(tmp_path: Path) -> None:
    """The pre-existing figure scripts call this positional form."""
    fig, ax = plt.subplots(figsize=(3, 3))
    draw_cdll_ring(ax, [0, 1, 2, 3], 1, 3, new_node_payload=3, show_legend=True)
    _assert_written(save_figure(fig, tmp_path / "legacy_ring", formats=("png",)))
    plt.close(fig)


def test_empty_cdll_ring_does_not_raise() -> None:
    fig, ax = plt.subplots()
    draw_cdll_ring(ax, [], 0, 0)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Backend contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["matplotlib", "networkx", "igraph"])
def test_backend_returns_a_total_layout(backend: str, example: tuple) -> None:
    """Each backend must return coordinates for every node it drew."""
    if backend not in available_backends():
        pytest.skip(f"{backend} backend unavailable")
    graph, _ = example
    fig, ax = plt.subplots()
    used = draw_graph(
        ax,
        graph,
        backend=backend,
        node_colors=build_node_palette(graph.node_count()),
        edge_colors=build_edge_palette(graph_edges(graph)),
    )
    assert set(used) == set(range(graph.node_count()))
    plt.close(fig)


@pytest.mark.parametrize("backend", ["matplotlib", "networkx", "igraph"])
def test_threading_the_layout_pins_every_node(backend: str, example: tuple) -> None:
    """Two panels drawn with a threaded layout must agree to the last bit."""
    if backend not in available_backends():
        pytest.skip(f"{backend} backend unavailable")
    graph, _ = example
    palette = build_node_palette(graph.node_count())
    edges = build_edge_palette(graph_edges(graph))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    first = draw_graph(ax1, graph, backend=backend, node_colors=palette, edge_colors=edges)
    second = draw_graph(
        ax2, graph, backend=backend, node_colors=palette, edge_colors=edges, layout=first
    )
    assert set(first) == set(second)
    for node in first:
        assert tuple(first[node]) == tuple(second[node]), f"node {node} moved"
    plt.close(fig)


def test_backend_never_creates_a_figure(example: tuple) -> None:
    """Drawing must add nothing to pyplot's figure registry."""
    graph, _ = example
    fig, ax = plt.subplots()
    before = set(plt.get_fignums())
    draw_graph(ax, graph, node_colors={}, edge_colors={})
    assert set(plt.get_fignums()) == before
    plt.close(fig)


def test_grayed_elements_are_caller_decided(example: tuple) -> None:
    """Passing a grey mask must not change the layout the backend returns."""
    graph, _ = example
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plain = draw_graph(ax1, graph, node_colors={}, edge_colors={})
    ghosted = draw_graph(
        ax2,
        graph,
        node_colors={},
        edge_colors={},
        grayed_nodes=frozenset({0, 1}),
        grayed_edges=frozenset(graph_edges(graph)[:1]),
        layout=plain,
    )
    assert plain == ghosted
    plt.close(fig)


def test_directed_graph_renders(tmp_path: Path) -> None:
    graph, _ = StringToGraph("VNVCv", directed_graph=True).run()
    fig, ax = plt.subplots()
    draw_graph(ax, graph, node_colors=build_node_palette(graph.node_count()), edge_colors={})
    _assert_written(save_figure(fig, tmp_path / "directed", formats=("png",)))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Composite figures
# ---------------------------------------------------------------------------


def test_single_card_renders(tmp_path: Path, example: tuple) -> None:
    graph, trace = example
    fig = single_card_figure(trace, full_graph=graph, title="final")
    _assert_written(save_figure(fig, tmp_path / "card", formats=("png",)))
    plt.close(fig)


def test_steps_figure_renders(tmp_path: Path, example: tuple) -> None:
    graph, trace = example
    fig = steps_figure(trace, full_graph=graph, n_columns=5)
    _assert_written(save_figure(fig, tmp_path / "steps", formats=("png",)))
    plt.close(fig)


def test_roundtrip_figure_renders(tmp_path: Path) -> None:
    from isalgraph.core.graph_to_string import GraphToString

    graph, s2g = StringToGraph(EXAMPLE, directed_graph=False).run_with_trace()
    _, g2s = GraphToString(graph).run_with_trace(0)
    fig = roundtrip_figure(g2s, s2g, n_columns=4)
    _assert_written(save_figure(fig, tmp_path / "roundtrip", formats=("png",)))
    plt.close(fig)


def test_grey_masks_invert_between_the_two_directions(example: tuple) -> None:
    """S2G starts fully ghosted and ends solid; G2S does the reverse."""
    from isalgraph.viz.composite import _grey_masks

    graph, s2g = example
    n_nodes = graph.node_count()

    first_s2g, _ = _grey_masks(s2g.snapshots[0], graph, "s2g")
    last_s2g, last_edges = _grey_masks(s2g.snapshots[-1], graph, "s2g")
    assert len(first_s2g) == n_nodes - 1  # only the seed node is built
    assert last_s2g == frozenset() and last_edges == frozenset()

    first_g2s, _ = _grey_masks(s2g.snapshots[0], graph, "g2s")
    last_g2s, _ = _grey_masks(s2g.snapshots[-1], graph, "g2s")
    assert len(first_g2s) == 1
    assert len(last_g2s) == n_nodes


def test_save_figure_keeps_dots_in_the_stem(tmp_path: Path) -> None:
    """``Path.with_suffix`` would truncate ``run_1.5`` to ``run_1.pdf``."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    written = save_figure(fig, tmp_path / "run_1.5", formats=("png",))
    assert written[0].name == "run_1.5.png"
    _assert_written(written)
    plt.close(fig)
