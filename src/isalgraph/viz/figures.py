"""Reproducible builders for the figures committed under ``docs/figures/``.

Run as ``python -m isalgraph.viz`` to regenerate every figure in place.
Each builder is a plain function returning a ``Figure``, so the same
worked examples can be reused from a notebook or a paper build script.

The worked example is fixed and seed-free: it is defined by an explicit
instruction string and an explicit edge list, not by a random draw, so
the committed PNGs are byte-reproducible across machines.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from isalgraph.core.graph_to_string import GraphToString
from isalgraph.core.sparse_graph import SparseGraph
from isalgraph.core.string_to_graph import StringToGraph
from isalgraph.core.trace import AlgorithmTrace
from isalgraph.viz.composite import single_card_figure, steps_figure
from isalgraph.viz.search_tree import canonical_search_tree_figure
from isalgraph.viz.style import apply_ieee_style, save_figure

if TYPE_CHECKING:
    from matplotlib.figure import Figure
else:
    Figure = Any

logger = logging.getLogger(__name__)

#: Worked example for the card and step figures. Exercises all four
#: operation classes: insertion via both pointers, a connection, and
#: movement of both pointers.
WORKED_EXAMPLE_STRING: str = "VNVnVCPvNC"

#: Worked example for the search-tree schematic: seven nodes, a cycle
#: plus two pendants, so several starting nodes are viable and the
#: candidate sets at the early V steps have size > 1.
WORKED_EXAMPLE_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (1, 4),
    (3, 5),
    (4, 6),
)
WORKED_EXAMPLE_N_NODES: int = 7


def build_example_graph(
    n_nodes: int = WORKED_EXAMPLE_N_NODES,
    edges: tuple[tuple[int, int], ...] = WORKED_EXAMPLE_EDGES,
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
    instructions: str = WORKED_EXAMPLE_STRING,
) -> tuple[SparseGraph, AlgorithmTrace]:
    """Return the graph and ``"s2g"`` trace for the worked instruction string."""
    return StringToGraph(instructions, directed_graph=False).run_with_trace()


def g2s_example_trace(graph: SparseGraph | None = None) -> tuple[str, AlgorithmTrace]:
    """Return the encoding and ``"g2s"`` trace for the worked example graph."""
    g = graph if graph is not None else build_example_graph()
    return GraphToString(g).run_with_trace(0)


def build_card_figure() -> Figure:
    """Build the single-card figure: CDLL ring, instruction strip, graph."""
    graph, trace = s2g_example_trace()
    return single_card_figure(
        trace,
        title=f"S2G final state\n$w$ = {WORKED_EXAMPLE_STRING}",
        full_graph=graph,
    )


def build_s2g_steps_figure() -> Figure:
    """Build the multi-column S2G step figure."""
    graph, trace = s2g_example_trace()
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


def build_search_tree_figure() -> Figure:
    """Build the canonical-search-space schematic requested by Reviewer 3."""
    return canonical_search_tree_figure(build_example_graph(), max_depth=3)


#: Output basename -> builder.
FIGURE_BUILDERS: dict[str, Any] = {
    "isalgraph_card_s2g": build_card_figure,
    "isalgraph_steps_s2g": build_s2g_steps_figure,
    "isalgraph_steps_g2s": build_g2s_steps_figure,
    "canonical_search_tree": build_search_tree_figure,
}


def render_all(
    output_dir: str | Path,
    *,
    formats: tuple[str, ...] = ("png",),
) -> list[Path]:
    """Render every figure in :data:`FIGURE_BUILDERS` into *output_dir*.

    Args:
        output_dir: Destination directory; created if absent.
        formats: Formats to emit per figure.

    Returns:
        Every path written.
    """
    import matplotlib.pyplot as plt

    apply_ieee_style()
    out_dir = Path(output_dir)
    written: list[Path] = []
    for name, builder in FIGURE_BUILDERS.items():
        fig = builder()
        written.extend(save_figure(fig, out_dir / name, formats=formats))
        plt.close(fig)
        logger.info("rendered %s", name)
    return written


__all__ = [
    "FIGURE_BUILDERS",
    "WORKED_EXAMPLE_EDGES",
    "WORKED_EXAMPLE_STRING",
    "build_card_figure",
    "build_example_graph",
    "build_g2s_steps_figure",
    "build_s2g_steps_figure",
    "build_search_tree_figure",
    "g2s_example_trace",
    "render_all",
    "s2g_example_trace",
]
