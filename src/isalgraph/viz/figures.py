"""Reproducible builders for the figures committed under ``docs/figures/``.

Run as ``python -m isalgraph.viz`` to regenerate every figure in place.
Each builder is a plain function returning a ``Figure``, so the same
worked examples can be reused from a notebook or a paper build script.

The worked example is fixed and seed-free: it is defined by an explicit
instruction string, an explicit edge list and an explicit node layout,
not by a random draw, so the committed figures are byte-reproducible
across machines.

One running example, three figures
----------------------------------
:data:`RUNNING_EXAMPLE_EDGES` serves the S2G panel, the G2S panel and the
canonical-search-space schematic, so a reader carries one graph through
the whole of §2. It was chosen by enumerating every connected graph on
5--6 nodes against five criteria; the sweep and the criteria are recorded
in ``.claude/notes/review/tasks/T-09-design.md`` §2. Two of them conflict
at ``n = 5`` and that conflict is what fixes ``n = 6``:

* the greedy encoder must attain the canonical string from some starting
  node, or the two panels cannot share one string; and
* the graph must be asymmetric, or the search tree the schematic draws is
  degenerate.

At ``n = 5`` the only graphs satisfying the first are :math:`C_5` and
:math:`K_{2,3}`, whose automorphism groups have order 10 and 12.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from isalgraph.core.graph_to_string import GraphToString
from isalgraph.core.sparse_graph import SparseGraph
from isalgraph.core.string_to_graph import StringToGraph
from isalgraph.core.trace import AlgorithmTrace
from isalgraph.types import NodeId
from isalgraph.viz.base import Position
from isalgraph.viz.composite import single_card_figure, steps_figure
from isalgraph.viz.encoder_trace import EncoderTrace, trace_encoder, trace_execution
from isalgraph.viz.search_tree import canonical_search_tree_figure
from isalgraph.viz.search_walkthrough import (
    canonical_search_walkthrough_figure,
    pruned_search_walkthrough_figure,
)
from isalgraph.viz.style import PATREC_TEXT_WIDTH_INCHES, apply_ieee_style, save_figure
from isalgraph.viz.worked_example import (
    RUNNING_EXAMPLE_POSITIONS,
    decode_trace,
    g2s_worked_example_figure,
    s2g_worked_example_figure,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure
else:
    Figure = Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The running example
# ---------------------------------------------------------------------------

#: Edges of the running example: a triangle ``{0,1,3}``, a path ``0-2-4``
#: and a pendant ``3-5``. ``|Aut(G)| = 1``.
RUNNING_EXAMPLE_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 3),
    (2, 4),
    (3, 5),
)

#: Node count of the running example.
RUNNING_EXAMPLE_N_NODES: int = 6

#: The starting node whose *greedy* encoding attains the canonical string.
#: The other five give strings of length 10, 9, 11, 10 and 10, so this is
#: the one place the start-node search actually pays.
RUNNING_EXAMPLE_START: NodeId = 0

#: ``canonical_string(RUNNING_EXAMPLE)``. Pinned here and checked by test
#: rather than recomputed at figure-build time, so a change to the
#: encoder shows up as a test failure and not as a quietly different
#: figure.
RUNNING_EXAMPLE_CANONICAL: str = "VVVnvPCPV"

#: ``pruned_canonical_string(RUNNING_EXAMPLE)``. Same length as the
#: exhaustive form and a **different string**, which Remark 2.11 already
#: allows: the two are different canonical forms, not two spellings of
#: one. The figures are generated for both so the difference is visible
#: rather than asserted.
RUNNING_EXAMPLE_PRUNED: str = "VVpvvPVnC"

#: ``StringToGraph`` allocates its own node ids, so a decoded graph is the
#: running example relabelled. Applying the map to the decoded graph gives
#: the input edge set back; a test asserts it for both strings. Drawing
#: the panels through it means the same structure occupies the same place
#: on the page in every figure, which is what makes the round trip legible.
DECODED_TO_INPUT: dict[str, dict[NodeId, NodeId]] = {
    RUNNING_EXAMPLE_CANONICAL: {0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5: 4},
    RUNNING_EXAMPLE_PRUNED: {0: 2, 1: 0, 2: 4, 3: 3, 4: 1, 5: 5},
}


def decoded_positions(instructions: str) -> dict[NodeId, Position]:
    """Return the pinned layout for the graph ``S2G(instructions)`` builds.

    Args:
        instructions: One of the two canonical strings of the running
            example.

    Returns:
        Coordinates keyed by *decoded* node id.

    Raises:
        KeyError: If *instructions* has no pinned relabelling.
    """
    return {
        decoded: RUNNING_EXAMPLE_POSITIONS[original]
        for decoded, original in DECODED_TO_INPUT[instructions].items()
    }


# ---------------------------------------------------------------------------
# Legacy worked example, retained for the older step and card figures
# ---------------------------------------------------------------------------

#: Instruction string for the card figure. Exercises all four operation
#: classes; unrelated to the running example, and deliberately left alone
#: so ``isalgraph_card_s2g`` does not move.
WORKED_EXAMPLE_STRING: str = "VNVnVCPvNC"

#: Retained under its original name; now an alias of the running example.
WORKED_EXAMPLE_EDGES: tuple[tuple[int, int], ...] = RUNNING_EXAMPLE_EDGES
WORKED_EXAMPLE_N_NODES: int = RUNNING_EXAMPLE_N_NODES


def build_example_graph(
    n_nodes: int = RUNNING_EXAMPLE_N_NODES,
    edges: tuple[tuple[int, int], ...] = RUNNING_EXAMPLE_EDGES,
    *,
    directed: bool = False,
) -> SparseGraph:
    """Build the fixed worked-example graph.

    Args:
        n_nodes: Node count.
        edges: Edge list.
        directed: Whether to build a directed graph.

    Returns:
        The constructed graph.
    """
    graph = SparseGraph(n_nodes, directed)
    for _ in range(n_nodes):
        graph.add_node()
    for u, v in edges:
        graph.add_edge(u, v)
    return graph


def s2g_example_trace(
    instructions: str = RUNNING_EXAMPLE_CANONICAL,
) -> tuple[SparseGraph, AlgorithmTrace]:
    """Return the graph and ``"s2g"`` trace for the worked instruction string."""
    return StringToGraph(instructions, directed_graph=False).run_with_trace()


def g2s_example_trace(graph: SparseGraph | None = None) -> tuple[str, AlgorithmTrace]:
    """Return the encoding and ``"g2s"`` replay trace for the worked example.

    Kept for the older ``isalgraph_steps_g2s`` figure. The worked-example
    G2S panel does **not** use this: the trace it returns is a replay of
    the finished string, not the encoder running. See
    :func:`g2s_example_encoder_trace`.
    """
    g = graph if graph is not None else build_example_graph()
    return GraphToString(g).run_with_trace(RUNNING_EXAMPLE_START)


def g2s_example_encoder_trace(graph: SparseGraph | None = None) -> EncoderTrace:
    """Return the instrumented encoder trace for the worked example."""
    g = graph if graph is not None else build_example_graph()
    return trace_encoder(g, RUNNING_EXAMPLE_START)


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------


def build_card_figure() -> Figure:
    """Build the single-card figure: CDLL ring, instruction strip, graph."""
    graph, trace = s2g_example_trace(WORKED_EXAMPLE_STRING)
    return single_card_figure(
        trace,
        title=f"S2G final state\n$w$ = {WORKED_EXAMPLE_STRING}",
        full_graph=graph,
    )


def build_s2g_steps_figure() -> Figure:
    """Build the multi-column S2G step figure."""
    graph, trace = s2g_example_trace(WORKED_EXAMPLE_STRING)
    return steps_figure(
        trace,
        full_graph=graph,
        n_columns=6,
        overall_title=(
            "String to graph: the graph fills in as instructions are consumed "
            f"($w$ = {WORKED_EXAMPLE_STRING})"
        ),
    )


def build_g2s_steps_figure() -> Figure:
    """Build the multi-column G2S step figure, with the grey mask inverted."""
    graph = build_example_graph()
    encoded, trace = g2s_example_trace(graph)
    return steps_figure(
        trace,
        n_columns=6,
        overall_title=(
            f"Graph to string: the graph greys out as structure is encoded ($w$ = {encoded})"
        ),
    )


def build_s2g_worked_example(instructions: str = RUNNING_EXAMPLE_CANONICAL) -> Figure:
    """Build the S2G worked-example panel.

    Columns are the encoder's symbol groups rather than single symbols,
    so this panel and :func:`build_g2s_worked_example` have the same
    columns, the same milestones and the same layout.

    Args:
        instructions: The canonical string to decode. One of
            :data:`RUNNING_EXAMPLE_CANONICAL` or
            :data:`RUNNING_EXAMPLE_PRUNED`.

    Returns:
        The created figure. The caller owns it and must close it.
    """
    graph, trace = decode_trace(instructions)
    groups = trace_execution(build_example_graph(), instructions).groups
    return s2g_worked_example_figure(
        trace,
        graph,
        groups=groups,
        positions=decoded_positions(instructions),
    )


def build_g2s_worked_example(instructions: str = RUNNING_EXAMPLE_CANONICAL) -> Figure:
    """Build the G2S worked-example panel.

    Args:
        instructions: The canonical string the depicted execution emits.
            For :data:`RUNNING_EXAMPLE_CANONICAL` this is the greedy
            encode from node 0; for :data:`RUNNING_EXAMPLE_PRUNED` it is
            the execution the pruned canonicalisation selects, which no
            greedy run reaches.

    Returns:
        The created figure. The caller owns it and must close it.
    """
    return g2s_worked_example_figure(
        trace_execution(build_example_graph(), instructions),
        positions=RUNNING_EXAMPLE_POSITIONS,
    )


def build_s2g_worked_example_pruned() -> Figure:
    """Build the S2G panel for the pruned canonical string."""
    return build_s2g_worked_example(RUNNING_EXAMPLE_PRUNED)


def build_g2s_worked_example_pruned() -> Figure:
    """Build the G2S panel for the pruned canonical string."""
    return build_g2s_worked_example(RUNNING_EXAMPLE_PRUNED)


def build_search_tree_figure() -> Figure:
    """Build the canonical-search-space schematic requested by Reviewer 3.

    Every starting node is drawn. Truncating the root set would show a
    forest whose width is an artefact of the drawing rather than of the
    search, and that width -- one subtree per node -- is half of what the
    schematic exists to show.
    """
    return canonical_search_tree_figure(
        build_example_graph(),
        max_depth=3,
        max_roots=None,
        figsize=(PATREC_TEXT_WIDTH_INCHES, 3.0),
        # The inset sits in the band under the tree, to the right of the
        # legend, so it gives the reader the graph the six subtrees are
        # rooted in without taking width from the leaf row.
        show_graph_inset=True,
        inset_positions=RUNNING_EXAMPLE_POSITIONS,
    )


def build_canonical_search_walkthrough() -> Figure:
    """Build the merged search-space / worked-example figure, exhaustive form.

    Landscape, for a rotated float: panel (a)'s depth axis is horizontal,
    which is what gives each of the seven steps enough width to label.
    """
    return canonical_search_walkthrough_figure(
        build_example_graph(),
        start_node=RUNNING_EXAMPLE_START,
        positions=RUNNING_EXAMPLE_POSITIONS,
    )


def build_pruned_search_walkthrough() -> Figure:
    """Build the same figure for the pruned canonicalisation.

    No starting node is passed. The pruned canonical string is generally
    emitted by no greedy run, so the builder recovers the execution from
    the string and finds the starting node itself.
    """
    return pruned_search_walkthrough_figure(
        build_example_graph(),
        positions=RUNNING_EXAMPLE_POSITIONS,
    )


#: Output basename -> builder.
FIGURE_BUILDERS: dict[str, Any] = {
    "isalgraph_card_s2g": build_card_figure,
    "isalgraph_steps_s2g": build_s2g_steps_figure,
    "isalgraph_steps_g2s": build_g2s_steps_figure,
    "fig_worked_example_s2g_canonical": build_s2g_worked_example,
    "fig_worked_example_g2s_canonical": build_g2s_worked_example,
    "fig_worked_example_s2g_pruned": build_s2g_worked_example_pruned,
    "fig_worked_example_g2s_pruned": build_g2s_worked_example_pruned,
    "canonical_search_tree": build_search_tree_figure,
    "fig_canonical_search_walkthrough": build_canonical_search_walkthrough,
    "fig_pruned_search_walkthrough": build_pruned_search_walkthrough,
}

#: The subset that goes into the manuscript. ``render_all(..., paper_only=True)``
#: renders exactly these, under the names the ``.tex`` will reference.
PAPER_FIGURES: tuple[str, ...] = (
    "fig_worked_example_s2g_canonical",
    "fig_worked_example_g2s_canonical",
    "fig_worked_example_s2g_pruned",
    "fig_worked_example_g2s_pruned",
    "canonical_search_tree",
    "fig_canonical_search_walkthrough",
    "fig_pruned_search_walkthrough",
)


def render_all(
    output_dir: str | Path,
    *,
    formats: tuple[str, ...] = ("png",),
    paper_only: bool = False,
) -> list[Path]:
    """Render every figure in :data:`FIGURE_BUILDERS` into *output_dir*.

    Args:
        output_dir: Destination directory; created if absent.
        formats: Formats to emit per figure.
        paper_only: Render only :data:`PAPER_FIGURES`.

    Returns:
        Every path written.
    """
    import matplotlib.pyplot as plt

    apply_ieee_style()
    out_dir = Path(output_dir)
    written: list[Path] = []
    names = PAPER_FIGURES if paper_only else tuple(FIGURE_BUILDERS)
    for name in names:
        fig = FIGURE_BUILDERS[name]()
        written.extend(save_figure(fig, out_dir / name, formats=formats))
        plt.close(fig)
        logger.info("rendered %s", name)
    return written


__all__ = [
    "DECODED_TO_INPUT",
    "FIGURE_BUILDERS",
    "PAPER_FIGURES",
    "RUNNING_EXAMPLE_CANONICAL",
    "RUNNING_EXAMPLE_PRUNED",
    "RUNNING_EXAMPLE_EDGES",
    "RUNNING_EXAMPLE_N_NODES",
    "RUNNING_EXAMPLE_START",
    "WORKED_EXAMPLE_EDGES",
    "WORKED_EXAMPLE_STRING",
    "build_card_figure",
    "build_example_graph",
    "build_g2s_steps_figure",
    "build_g2s_worked_example",
    "build_g2s_worked_example_pruned",
    "build_s2g_steps_figure",
    "build_s2g_worked_example",
    "build_s2g_worked_example_pruned",
    "build_canonical_search_walkthrough",
    "build_pruned_search_walkthrough",
    "build_search_tree_figure",
    "decoded_positions",
    "g2s_example_encoder_trace",
    "g2s_example_trace",
    "render_all",
    "s2g_example_trace",
]
