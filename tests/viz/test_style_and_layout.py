"""Tests for the pure parts of the viz layer: palettes, layout, registry.

These carry the real assertions. Rendering tests can only check that a
file appeared; palette validity and layout geometry are checkable facts.
"""

from __future__ import annotations

import math
import re

import pytest

from isalgraph.core.sparse_graph import SparseGraph
from isalgraph.core.trace import graph_edges
from isalgraph.errors import VizBackendNotFoundError
from isalgraph.viz import layout as layout_mod
from isalgraph.viz import registry as registry_mod
from isalgraph.viz import style

HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _ring(n: int) -> SparseGraph:
    """Return an undirected cycle on *n* nodes."""
    g = SparseGraph(n, False)
    for _ in range(n):
        g.add_node()
    for i in range(n):
        g.add_edge(i, (i + 1) % n)
    return g


# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "palette",
    [
        style.INSTRUCTION_PALETTE,
        style.PAUL_TOL_BRIGHT,
        style.PAUL_TOL_HIGH_CONTRAST,
    ],
)
def test_palette_values_are_valid_hex(palette: dict[str, str]) -> None:
    assert all(HEX.match(v) for v in palette.values()), palette


def test_muted_palette_is_valid_hex() -> None:
    assert all(HEX.match(v) for v in style.PAUL_TOL_MUTED)


def test_instruction_palette_covers_the_whole_alphabet() -> None:
    assert set(style.INSTRUCTION_PALETTE) == set("NnPpVvCcW")
    assert set(style.INSTRUCTION_POINTER) == set("NnPpVvCcW")
    assert set(style.INSTRUCTION_OPERATION) == set("NnPpVvCcW")


def test_case_selects_the_pointer_accent() -> None:
    """Uppercase acts on the primary pointer, lowercase on the secondary."""
    for upper, lower in (("N", "n"), ("P", "p"), ("V", "v"), ("C", "c")):
        assert style.pointer_accent(upper) == style.POINTER_PALETTE[0]
        assert style.pointer_accent(lower) == style.POINTER_PALETTE[1]
    assert style.pointer_accent("W") == style.GRAYED_EDGE
    assert style.pointer_accent(None) == style.GRAYED_EDGE


def test_pointer_palette_holds_two_distinct_colors() -> None:
    assert len(style.POINTER_PALETTE) == 2
    assert style.POINTER_PALETTE[0] != style.POINTER_PALETTE[1]


def test_instruction_pairs_share_an_operation_class() -> None:
    """Hue encodes the operation, so a case-pair must share its class."""
    for upper, lower in (("N", "n"), ("P", "p"), ("V", "v"), ("C", "c")):
        assert style.INSTRUCTION_OPERATION[upper] == style.INSTRUCTION_OPERATION[lower]


@pytest.mark.parametrize("n", [0, 1, 2, 5, 9, 12, 30])
def test_node_palette_is_total_and_grey_free(n: int) -> None:
    """Every node gets a colour, and none of them is the ghost grey."""
    palette = style.build_node_palette(n)
    assert set(palette) == set(range(n))
    assert all(HEX.match(v) for v in palette.values())
    assert style.GRAYED_FACE.upper() not in {v.upper() for v in palette.values()}


def test_edge_palette_is_keyed_by_normalised_edges() -> None:
    graph = _ring(5)
    edges = graph_edges(graph)
    palette = style.build_edge_palette(edges)
    assert set(palette) == set(edges)
    assert all(HEX.match(v) for v in palette.values())


def test_color_for_instruction_falls_back_to_grey() -> None:
    assert style.color_for_instruction("V") == style.INSTRUCTION_PALETTE["V"]
    assert style.color_for_instruction(None) == style.GRAYED_FACE
    assert style.color_for_instruction("?") == style.GRAYED_FACE


def test_get_figure_size_matches_ieee_widths() -> None:
    assert style.get_figure_size("single")[0] == pytest.approx(3.39)
    assert style.get_figure_size("double")[0] == pytest.approx(7.0)
    with pytest.raises(ValueError, match="single"):
        style.get_figure_size("triple")


def test_rcparams_pin_type42_fonts() -> None:
    """Publishers reject Type-3 glyphs; both must stay at 42."""
    assert style.BASE_RCPARAMS["pdf.fonttype"] == 42
    assert style.BASE_RCPARAMS["ps.fonttype"] == 42


# ---------------------------------------------------------------------------
# Layout geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 5, 8])
def test_ring_positions_lie_on_the_circle_and_are_evenly_spaced(n: int) -> None:
    order = tuple(range(n))
    pos = layout_mod.cdll_ring_positions(order, radius=2.0)
    assert set(pos) == set(order)
    for x, y in pos.values():
        assert math.hypot(x, y) == pytest.approx(2.0)

    angles = [math.atan2(y, x) % (2 * math.pi) for x, y in (pos[v] for v in order)]
    gaps = [(angles[i] - angles[(i + 1) % n]) % (2 * math.pi) for i in range(n)]
    assert all(g == pytest.approx(2 * math.pi / n) for g in gaps)


def test_ring_starts_at_twelve_oclock_and_runs_clockwise() -> None:
    pos = layout_mod.cdll_ring_positions((0, 1, 2, 3), radius=1.0)
    assert pos[0] == pytest.approx((0.0, 1.0), abs=1e-9)
    # Clockwise from the top means the next slot sits at +x.
    assert pos[1][0] == pytest.approx(1.0, abs=1e-9)


def test_ring_handles_degenerate_sizes() -> None:
    assert layout_mod.cdll_ring_positions(()) == {}
    assert layout_mod.cdll_ring_positions((7,)) == {7: (0.0, 0.0)}


def test_compact_layout_covers_every_node_and_fits_the_canvas() -> None:
    pytest.importorskip("networkx")
    graph = _ring(7)
    pos = layout_mod.compact_graph_layout(graph, fit_fraction=0.78)
    assert set(pos) == set(range(7))
    assert all(abs(x) <= 1.2 and abs(y) <= 1.2 for x, y in pos.values())


def test_compact_layout_parks_disconnected_components_on_a_strip() -> None:
    pytest.importorskip("networkx")
    graph = SparseGraph(6, False)
    for _ in range(6):
        graph.add_node()
    for u, v in ((0, 1), (1, 2), (2, 0)):
        graph.add_edge(u, v)
    pos = layout_mod.compact_graph_layout(graph, margin=0.18)
    assert set(pos) == set(range(6))
    strays = [pos[v][0] for v in (3, 4, 5)]
    assert all(x == pytest.approx(1.18) for x in strays)


def test_compact_layout_of_an_edgeless_graph_falls_back_to_a_ring() -> None:
    pytest.importorskip("networkx")
    graph = SparseGraph(4, False)
    for _ in range(4):
        graph.add_node()
    pos = layout_mod.compact_graph_layout(graph)
    assert set(pos) == set(range(4))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_unknown_backend_raises() -> None:
    with pytest.raises(VizBackendNotFoundError, match="not registered"):
        registry_mod.get_backend("no-such-backend")


def test_matplotlib_is_the_default_and_is_available() -> None:
    pytest.importorskip("matplotlib")
    assert registry_mod.DEFAULT_BACKEND == "matplotlib"
    assert "matplotlib" in registry_mod.available_backends()


def test_available_is_a_subset_of_registered() -> None:
    assert set(registry_mod.available_backends()) <= set(registry_mod.registered_backends())


def test_registered_backends_declare_availability() -> None:
    """Every backend must answer ``is_available`` without drawing anything."""
    for name in registry_mod.registered_backends():
        backend = registry_mod.get_backend(name, require_available=False)
        assert isinstance(type(backend).is_available(), bool)
        assert backend.name == name
