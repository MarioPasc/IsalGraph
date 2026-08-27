"""Tests for the two worked-example figures and the running example.

Two kinds of check live here.

**The running example's facts.** The figure builders quote
``VVVnvPCPV`` and ``VVpvvPVnC`` as module constants rather than
recomputing them, so a change to the encoder would otherwise show up as a
quietly different figure instead of a failure. Every claim the figures
make about the example -- the two canonical strings, the round trip, the
start node whose greedy encode attains the exhaustive form, the pinned
relabelling the layouts use -- is asserted here.

**The layout contract.** The panels are only comparable if they are
identical in geometry, so the test asserts they share a figure size and a
column count rather than trusting two builders to stay in step.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from isalgraph.core.canonical import canonical_string  # noqa: E402
from isalgraph.core.canonical_pruned import pruned_canonical_string  # noqa: E402
from isalgraph.core.graph_to_string import GraphToString  # noqa: E402
from isalgraph.core.trace import graph_edges  # noqa: E402
from isalgraph.viz.encoder_trace import trace_execution  # noqa: E402
from isalgraph.viz.figures import (  # noqa: E402
    DECODED_TO_INPUT,
    RUNNING_EXAMPLE_CANONICAL,
    RUNNING_EXAMPLE_EDGES,
    RUNNING_EXAMPLE_N_NODES,
    RUNNING_EXAMPLE_PRUNED,
    RUNNING_EXAMPLE_START,
    build_example_graph,
    build_g2s_worked_example,
    build_s2g_worked_example,
    build_search_tree_figure,
    decoded_positions,
)
from isalgraph.viz.style import save_figure  # noqa: E402
from isalgraph.viz.worked_example import (  # noqa: E402
    RUNNING_EXAMPLE_POSITIONS,
    WorkedExampleError,
    decode_trace,
    g2s_columns,
    group_spans,
    s2g_columns,
)

_BOTH_STRINGS = (RUNNING_EXAMPLE_CANONICAL, RUNNING_EXAMPLE_PRUNED)


# ---------------------------------------------------------------------------
# The running example
# ---------------------------------------------------------------------------


def test_canonical_strings_are_what_the_figures_claim() -> None:
    """Both pinned strings still come out of the two canonicalisations."""
    graph = build_example_graph()
    assert canonical_string(graph) == RUNNING_EXAMPLE_CANONICAL
    assert pruned_canonical_string(graph) == RUNNING_EXAMPLE_PRUNED


def test_pruned_and_exhaustive_forms_differ_at_equal_length() -> None:
    """The two canonical forms are different strings of the same length.

    Remark 2.11 permits exactly this, and the figure set exists partly to
    show it. A test asserting only that both exist would pass if they
    silently converged and the pruned figure became a duplicate.
    """
    assert RUNNING_EXAMPLE_CANONICAL != RUNNING_EXAMPLE_PRUNED
    assert len(RUNNING_EXAMPLE_CANONICAL) == len(RUNNING_EXAMPLE_PRUNED)


def test_greedy_from_the_declared_start_attains_the_canonical_string() -> None:
    """Node 0's greedy encode is the exhaustive canonical string.

    This is what lets the G2S and S2G panels of the canonical pair share
    one string, and it is the property that forced ``n = 6``.
    """
    graph = build_example_graph()
    emitted, _ = GraphToString(graph).run(RUNNING_EXAMPLE_START)
    assert emitted == RUNNING_EXAMPLE_CANONICAL


def test_no_greedy_run_reaches_the_pruned_string() -> None:
    """The pruned form needs the neighbour-choice branch, not just a start node.

    If some greedy run did emit it, the pruned G2S panel would be a
    greedy trace and the target-directed reconstruction would be
    unnecessary machinery.
    """
    graph = build_example_graph()
    emitted = {GraphToString(graph).run(v)[0] for v in range(RUNNING_EXAMPLE_N_NODES)}
    assert RUNNING_EXAMPLE_PRUNED not in emitted


@pytest.mark.parametrize("instructions", _BOTH_STRINGS)
def test_round_trip_and_pinned_relabelling(instructions: str) -> None:
    """``S2G`` of each string rebuilds the example under the pinned map.

    The layouts draw the decoded graph at the input graph's coordinates,
    so the relabelling in :data:`DECODED_TO_INPUT` is load-bearing: if it
    were wrong the figure would still render, with the edges joining the
    wrong discs.
    """
    decoded, _ = decode_trace(instructions)
    mapping = DECODED_TO_INPUT[instructions]
    assert decoded.node_count() == RUNNING_EXAMPLE_N_NODES
    relabelled = {tuple(sorted((mapping[u], mapping[v]))) for u, v in graph_edges(decoded)}
    assert relabelled == {tuple(sorted(e)) for e in RUNNING_EXAMPLE_EDGES}


@pytest.mark.parametrize("instructions", _BOTH_STRINGS)
def test_decoded_positions_cover_every_node(instructions: str) -> None:
    """Every decoded node has a pinned coordinate."""
    positions = decoded_positions(instructions)
    assert set(positions) == set(range(RUNNING_EXAMPLE_N_NODES))
    assert set(positions.values()) == set(RUNNING_EXAMPLE_POSITIONS.values())


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------


def test_group_spans_partition_the_string() -> None:
    """Spans tile ``[0, len)`` with no gap and no overlap."""
    groups = ("V", "V", "nv", "PC")
    spans = group_spans(groups)
    assert spans[0][0] == 0
    assert spans[-1][1] == sum(len(g) for g in groups)
    for (_, hi), (lo, _) in zip(spans, spans[1:], strict=False):
        assert hi == lo


@pytest.mark.parametrize("instructions", _BOTH_STRINGS)
def test_both_panels_produce_the_same_columns(instructions: str) -> None:
    """S2G and G2S columns line up one to one.

    The whole point of indexing both panels by the encoder's group
    boundaries is that column *k* of one is comparable with column *k* of
    the other. If the counts or the titles diverge, they are not.
    """
    encoder = trace_execution(build_example_graph(), instructions)
    decoded, trace = decode_trace(instructions)
    s2g = s2g_columns(trace, encoder.groups)
    g2s = g2s_columns(encoder)
    assert len(s2g) == len(g2s) == len(encoder)
    assert [c.title for c in s2g] == [c.title for c in g2s]
    assert [c.span for c in s2g] == [c.span for c in g2s]
    assert [c.consumed for c in s2g] == [c.consumed for c in g2s]


@pytest.mark.parametrize("instructions", _BOTH_STRINGS)
def test_the_two_panels_drain_in_opposite_directions(instructions: str) -> None:
    """Ink is conserved, and each panel moves it the way its algorithm does.

    S2G ends with the graph complete and the strip spent; G2S ends with
    the strip complete and the graph spent. Getting this backwards is not
    a cosmetic error -- it makes the encoder panel show a decoder, which
    is what the first draft of these figures did.
    """
    encoder = trace_execution(build_example_graph(), instructions)
    _, trace = decode_trace(instructions)

    s2g = s2g_columns(trace, encoder.groups)
    assert s2g[0].strip_solid_side == "suffix"
    assert len(s2g[-1].present_nodes) == RUNNING_EXAMPLE_N_NODES
    assert len(s2g[-1].present_edges) == len(RUNNING_EXAMPLE_EDGES)
    for earlier, later in zip(s2g, s2g[1:], strict=False):
        assert earlier.present_nodes <= later.present_nodes
        assert earlier.present_edges <= later.present_edges

    g2s = g2s_columns(encoder)
    assert g2s[0].strip_solid_side == "prefix"
    assert g2s[-1].present_nodes == frozenset()
    assert g2s[-1].present_edges == frozenset()
    for earlier, later in zip(g2s, g2s[1:], strict=False):
        assert later.present_nodes <= earlier.present_nodes
        assert later.present_edges <= earlier.present_edges

    for columns in (s2g, g2s):
        assert columns[-1].consumed == len(instructions)


def test_columns_reject_groups_that_do_not_span_the_string() -> None:
    """A group list that stops short is a caller error, not a short figure."""
    _, trace = decode_trace(RUNNING_EXAMPLE_CANONICAL)
    with pytest.raises(WorkedExampleError, match="groups cover"):
        s2g_columns(trace, ("V", "V"))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("instructions", _BOTH_STRINGS)
def test_the_two_panels_share_a_geometry(instructions: str) -> None:
    """Both panels render at the same size, which is the layout contract."""
    s2g = build_s2g_worked_example(instructions)
    g2s = build_g2s_worked_example(instructions)
    try:
        assert s2g.get_size_inches().tolist() == g2s.get_size_inches().tolist()
        assert s2g.get_figwidth() == pytest.approx(7.0)
    finally:
        plt.close(s2g)
        plt.close(g2s)


@pytest.mark.parametrize("instructions", _BOTH_STRINGS)
def test_panels_write_a_non_trivial_pdf(instructions: str, tmp_path: object) -> None:
    """Both panels survive a vector write at the size the manuscript takes."""
    from pathlib import Path

    out = Path(str(tmp_path))
    for name, builder in (("s2g", build_s2g_worked_example), ("g2s", build_g2s_worked_example)):
        fig = builder(instructions)
        try:
            written = save_figure(fig, out / f"{name}_{len(instructions)}", formats=("pdf",))
        finally:
            plt.close(fig)
        assert written[0].stat().st_size > 4_000, written[0]


def test_search_tree_draws_every_start_node() -> None:
    """The schematic shows one subtree per node, not a truncated sample.

    The width of that forest is half of what R3.7c asked to see; capping
    it would make the figure's width an artefact of the drawing.

    The width assertion is not incidental. Point sizes inside a figure are
    absolute, so a figure rendered wider than the text block it is placed
    in has every label scaled down by the ratio: at the previous 7.0 in
    this schematic's 5.5 pt labels reached a 4.72 in text block at 3.7 pt.
    Rendering at the placement width is what makes the declared sizes the
    printed ones.
    """
    from isalgraph.viz.style import PATREC_TEXT_WIDTH_INCHES

    fig = build_search_tree_figure()
    try:
        assert fig.get_figwidth() == pytest.approx(PATREC_TEXT_WIDTH_INCHES)
    finally:
        plt.close(fig)

    from isalgraph.viz.search_tree import enumerate_search_tree

    tree = enumerate_search_tree(build_example_graph(), max_depth=3, max_roots=None)
    assert len(tree.roots) == RUNNING_EXAMPLE_N_NODES
