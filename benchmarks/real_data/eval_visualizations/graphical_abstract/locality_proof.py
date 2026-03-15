"""Locality proof figure for the graphical abstract.

Shows that a single edge edit (GED=1) corresponds to a small
Levenshtein distance in the canonical string, using the house graph
as the running example:
  - G₀ = house graph (5 nodes, 6 edges)
  - G₋ = house − edge(2,3) = cycle₅, GED=1, Lev=1
  - G₊ = house + edge(0,3),         GED=1, Lev=4
"""

from __future__ import annotations

import argparse
import logging
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

from benchmarks.plotting_styles import (
    INSTRUCTION_COLORS,
    PAUL_TOL_BRIGHT,
    PAUL_TOL_MUTED,
    apply_ieee_style,
    save_figure,
)
from isalgraph.adapters.networkx_adapter import NetworkXAdapter
from isalgraph.core.graph_to_string import GraphToString

logger = logging.getLogger(__name__)

_NODE_COLOR = PAUL_TOL_MUTED[1]  # "#332288" indigo
_HIGHLIGHT_COLOR = PAUL_TOL_BRIGHT["red"]  # "#EE6677"
_GHOST_COLOR = "0.70"
_FORMATS = ("pdf", "svg", "png")


# =========================================================================
# Helpers
# =========================================================================


def _greedy_min(G: nx.Graph) -> str:
    """Compute greedy-min string (shortest, then lexmin across all starts)."""
    adapter = NetworkXAdapter()
    sg = adapter.from_external(G, directed=False)
    best: str | None = None
    for v0 in range(G.number_of_nodes()):
        g2s = GraphToString(sg)
        w, _ = g2s.run(initial_node=v0)
        if best is None or len(w) < len(best) or (len(w) == len(best) and w < best):
            best = w
    assert best is not None
    return best


def _draw_graph_cell(
    ax: plt.Axes,
    G: nx.Graph,
    pos: dict,
    *,
    highlight_edges: list[tuple[int, int]] | None = None,
    ghost_edges: list[tuple[int, int]] | None = None,
    node_size: int = 200,
) -> None:
    """Draw a graph with optional highlighted/ghost edges.

    Args:
        ax: Target axes.
        G: Graph to draw.
        pos: Layout positions.
        highlight_edges: Edges to draw in red (newly added).
        ghost_edges: Edges to draw as dashed gray (removed).
        node_size: Node marker size.
    """
    # Draw ghost (removed) edges first
    if ghost_edges:
        nx.draw_networkx_edges(
            G if not ghost_edges else nx.Graph(list(G.edges()) + ghost_edges),
            pos,
            edgelist=ghost_edges,
            ax=ax,
            edge_color=_GHOST_COLOR,
            width=1.2,
            style="dashed",
        )

    # Normal edges
    normal_edges = list(G.edges())
    if highlight_edges:
        hl_set = {(min(u, v), max(u, v)) for u, v in highlight_edges}
        normal_edges = [(u, v) for u, v in G.edges() if (min(u, v), max(u, v)) not in hl_set]

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=normal_edges,
        ax=ax,
        edge_color="0.5",
        width=0.8,
    )

    # Highlighted (added) edges
    if highlight_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=highlight_edges,
            ax=ax,
            edge_color=_HIGHLIGHT_COLOR,
            width=1.8,
            style="solid",
        )

    # Nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=_NODE_COLOR,
        node_size=node_size,
        edgecolors="0.3",
        linewidths=0.5,
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6, font_color="white")

    # Padding
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 0.1)
    pad = 0.25 * span
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_heatmap_string(
    ax: plt.Axes,
    string: str,
    *,
    fontsize: float = 7,
    cell_w: float = 0.9,
    cell_h: float = 0.7,
) -> None:
    """Draw IsalGraph string as horizontal colored boxes.

    Args:
        ax: Target axes.
        string: Instruction string.
        fontsize: Character font size.
        cell_w: Width of each cell in data coords.
        cell_h: Height of each cell in data coords.
    """
    n = len(string)
    for i, ch in enumerate(string):
        color = INSTRUCTION_COLORS.get(ch, "#000000")
        rect = mpatches.FancyBboxPatch(
            (i, 0),
            cell_w * 0.90,
            cell_h,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor="0.35",
            linewidth=0.3,
        )
        ax.add_patch(rect)
        ax.text(
            i + cell_w * 0.45,
            cell_h / 2,
            ch,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontfamily="monospace",
            fontweight="bold",
            color="white",
        )

    ax.set_xlim(-0.2, n + 0.2)
    ax.set_ylim(-0.15, cell_h + 0.15)
    ax.set_aspect("auto")
    ax.axis("off")


# =========================================================================
# Main figure
# =========================================================================


def generate_locality_figure(output_dir: str) -> str:
    """Generate the locality proof figure.

    Layout: three columns (G₋, G₀, G₊) with graphs on top, strings below,
    and bidirectional arrows with GED/Lev labels between columns.

    Args:
        output_dir: Output directory.

    Returns:
        Base output path (without extension).
    """
    from matplotlib.gridspec import GridSpec

    # ---- Build graphs ----
    G0 = nx.house_graph()
    w0 = _greedy_min(G0)

    # G₋ = house − edge(2,3) = cycle₅
    G_minus = G0.copy()
    G_minus.remove_edge(2, 3)
    w_minus = _greedy_min(G_minus)

    # G₊ = house + edge(0,3)
    G_plus = G0.copy()
    G_plus.add_edge(0, 3)
    w_plus = _greedy_min(G_plus)

    import Levenshtein

    lev_minus = Levenshtein.distance(w0, w_minus)
    lev_plus = Levenshtein.distance(w0, w_plus)
    logger.info("G0: w=%r", w0)
    logger.info("G-: w=%r, GED=1, Lev=%d", w_minus, lev_minus)
    logger.info("G+: w=%r, GED=1, Lev=%d", w_plus, lev_plus)

    # ---- Fixed layout (consistent across all three graphs) ----
    # Use the union of all edges for layout stability
    G_union = nx.Graph()
    G_union.add_nodes_from(range(max(G_plus.number_of_nodes(), G0.number_of_nodes())))
    for G in [G0, G_minus, G_plus]:
        G_union.add_edges_from(G.edges())
    pos = nx.spring_layout(G_union, seed=42)

    # ---- Figure layout ----
    fig = plt.figure(figsize=(5.5, 2.2))
    gs = GridSpec(
        2,
        3,
        figure=fig,
        height_ratios=[3.0, 0.8],
        width_ratios=[1, 1, 1],
        hspace=0.15,
        wspace=0.25,
        left=0.03,
        right=0.97,
        top=0.88,
        bottom=0.04,
    )

    # Graphs (top row)
    ax_g_minus = fig.add_subplot(gs[0, 0])
    ax_g0 = fig.add_subplot(gs[0, 1])
    ax_g_plus = fig.add_subplot(gs[0, 2])

    # Strings (bottom row)
    ax_w_minus = fig.add_subplot(gs[1, 0])
    ax_w0 = fig.add_subplot(gs[1, 1])
    ax_w_plus = fig.add_subplot(gs[1, 2])

    # ---- Draw graphs ----
    # G₋: show removed edge as dashed
    _draw_graph_cell(
        ax_g_minus,
        G_minus,
        pos,
        ghost_edges=[(2, 3)],
    )
    ax_g_minus.set_title(
        r"$G_{-}$ (edge removed)",
        fontsize=7,
        pad=4,
        color="0.3",
    )

    # G₀: center, no highlights
    _draw_graph_cell(ax_g0, G0, pos)
    ax_g0.set_title(
        r"$G_0$ (house graph)",
        fontsize=7,
        pad=4,
        fontweight="bold",
        color="0.2",
    )

    # G₊: show added edge highlighted
    _draw_graph_cell(
        ax_g_plus,
        G_plus,
        pos,
        highlight_edges=[(0, 3)],
    )
    ax_g_plus.set_title(
        r"$G_{+}$ (edge added)",
        fontsize=7,
        pad=4,
        color="0.3",
    )

    # ---- Draw strings ----
    _draw_heatmap_string(ax_w_minus, w_minus, fontsize=6)
    _draw_heatmap_string(ax_w0, w0, fontsize=6)
    _draw_heatmap_string(ax_w_plus, w_plus, fontsize=6)

    # ---- Arrows with distance labels ----
    fig.canvas.draw()

    # Arrow between G₋ and G₀
    fig.text(
        0.355,
        0.50,
        r"$\longleftrightarrow$",
        ha="center",
        va="center",
        fontsize=16,
        color="0.4",
        transform=fig.transFigure,
    )
    fig.text(
        0.355,
        0.42,
        f"GED = 1\nLev = {lev_minus}",
        ha="center",
        va="top",
        fontsize=6.5,
        color="0.35",
        transform=fig.transFigure,
        linespacing=1.3,
    )

    # Arrow between G₀ and G₊
    fig.text(
        0.655,
        0.50,
        r"$\longleftrightarrow$",
        ha="center",
        va="center",
        fontsize=16,
        color="0.4",
        transform=fig.transFigure,
    )
    fig.text(
        0.655,
        0.42,
        f"GED = 1\nLev = {lev_plus}",
        ha="center",
        va="top",
        fontsize=6.5,
        color="0.35",
        transform=fig.transFigure,
        linespacing=1.3,
    )

    # ---- Save ----
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "locality_proof")
    save_figure(fig, path, formats=_FORMATS)
    plt.close(fig)
    logger.info("Locality proof saved: %s", path)
    return path


# =========================================================================
# CLI
# =========================================================================


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate locality proof figure for graphical abstract."
    )
    parser.add_argument(
        "--output-dir",
        default="paper_figures/graphical_abstract",
        help="Output directory.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    apply_ieee_style()
    generate_locality_figure(args.output_dir)


if __name__ == "__main__":
    main()
