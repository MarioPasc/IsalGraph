"""Panel A: Bijective string <-> CDLL <-> graph encoding concept.

Generates the left panel of the graphical abstract showing the IsalGraph
bijective encoding through the CDLL auxiliary structure:
  - Top: Colored instruction string w*
  - Center: Simplified CDLL ring with pointer labels
  - Bottom: House graph G
  - Bidirectional arrows connecting all three elements.

Uses a single-axes approach to avoid cross-axes arrow rendering issues.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from benchmarks.plotting_styles import (
    INSTRUCTION_COLORS,
    PAUL_TOL_MUTED,
    apply_ieee_style,
    save_figure,
)
from isalgraph.adapters.networkx_adapter import NetworkXAdapter
from isalgraph.core.graph_to_string import GraphToString
from isalgraph.core.string_to_graph import StringToGraph

logger = logging.getLogger(__name__)

_NODE_COLOR = PAUL_TOL_MUTED[1]  # "#332288" indigo
_PRIMARY_LABEL_COLOR = "#EE6677"
_SECONDARY_LABEL_COLOR = "#4477AA"
_ARROW_COLOR = "0.45"


# =========================================================================
# Drawing helpers (all in data coordinates on a single axes)
# =========================================================================


def _draw_instruction_boxes(
    ax: plt.Axes,
    string: str,
    center_x: float,
    center_y: float,
    *,
    cell_w: float = 0.08,
    cell_h: float = 0.025,
) -> None:
    """Draw horizontal row of colored instruction boxes."""
    n = len(string)
    total_w = n * cell_w
    # Offset so the string block + w* label together are centered
    label_space = 0.05  # space for w* label to the left
    x_start = center_x - total_w / 2 + label_space / 2

    for i, ch in enumerate(string):
        color = INSTRUCTION_COLORS.get(ch, "#000000")
        x = x_start + i * cell_w
        rect = mpatches.FancyBboxPatch(
            (x, center_y - cell_h / 2),
            cell_w * 0.90,
            cell_h,
            boxstyle="round,pad=0.003",
            facecolor=color,
            edgecolor="0.35",
            linewidth=0.3,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(
            x + cell_w * 0.45,
            center_y,
            ch,
            ha="center",
            va="center",
            fontsize=6.5,
            fontfamily="monospace",
            fontweight="bold",
            color="white",
            zorder=4,
        )

    # w* label to the left of the boxes
    ax.text(
        x_start - 0.02,
        center_y,
        r"$w^*$",
        ha="right",
        va="center",
        fontsize=7,
        color="0.3",
    )


def _draw_cdll_ring(
    ax: plt.Axes,
    cdll_order: Sequence[int],
    primary_payload: int,
    secondary_payload: int,
    center_x: float,
    center_y: float,
    *,
    radius: float = 0.09,
    node_r: float = 0.018,
) -> None:
    """Draw simplified CDLL ring at given center position."""
    n = len(cdll_order)
    if n == 0:
        return

    # Dashed backbone circle
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(
        center_x + radius * np.cos(theta),
        center_y + radius * np.sin(theta),
        color="0.75",
        linewidth=0.5,
        linestyle="--",
        zorder=1,
    )

    angles = [np.pi / 2 - 2 * np.pi * i / n for i in range(n)]
    for i, payload in enumerate(cdll_order):
        cx = center_x + radius * np.cos(angles[i])
        cy = center_y + radius * np.sin(angles[i])

        circle = plt.Circle(
            (cx, cy),
            node_r,
            facecolor=_NODE_COLOR,
            edgecolor="0.2",
            linewidth=0.4,
            zorder=3,
        )
        ax.add_patch(circle)
        ax.text(
            cx,
            cy,
            str(payload),
            ha="center",
            va="center",
            fontsize=4.5,
            fontweight="bold",
            color="white",
            zorder=4,
        )

        # Pointer labels outside ring
        label_r = radius + node_r + 0.015
        lx = center_x + label_r * np.cos(angles[i])
        ly = center_y + label_r * np.sin(angles[i])

        if payload == primary_payload and payload == secondary_payload:
            ax.text(
                lx - 0.012,
                ly,
                r"$\pi$",
                ha="center",
                va="center",
                fontsize=5.5,
                fontweight="bold",
                color=_PRIMARY_LABEL_COLOR,
                zorder=4,
            )
            ax.text(
                lx + 0.012,
                ly,
                r"$\sigma$",
                ha="center",
                va="center",
                fontsize=5.5,
                fontweight="bold",
                color=_SECONDARY_LABEL_COLOR,
                zorder=4,
            )
        elif payload == primary_payload:
            ax.text(
                lx,
                ly,
                r"$\pi$",
                ha="center",
                va="center",
                fontsize=5.5,
                fontweight="bold",
                color=_PRIMARY_LABEL_COLOR,
                zorder=4,
            )
        elif payload == secondary_payload:
            ax.text(
                lx,
                ly,
                r"$\sigma$",
                ha="center",
                va="center",
                fontsize=5.5,
                fontweight="bold",
                color=_SECONDARY_LABEL_COLOR,
                zorder=4,
            )

    # Label below
    ax.text(
        center_x,
        center_y - radius - node_r - 0.025,
        "CDLL",
        ha="center",
        va="top",
        fontsize=5.5,
        fontstyle="italic",
        color="0.45",
    )


def _draw_graph(
    ax: plt.Axes,
    G: nx.Graph,
    center_x: float,
    center_y: float,
    *,
    scale: float = 0.10,
    layout_seed: int = 42,
    node_size: int = 90,
) -> None:
    """Draw graph at given center position using spring layout."""
    pos_raw = nx.spring_layout(G, seed=layout_seed)

    # Rescale and offset to desired center
    xs = np.array([p[0] for p in pos_raw.values()])
    ys = np.array([p[1] for p in pos_raw.values()])
    cx, cy = xs.mean(), ys.mean()
    pos = {
        n: (center_x + (p[0] - cx) * scale, center_y + (p[1] - cy) * scale)
        for n, p in pos_raw.items()
    }

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="0.5", width=0.6)
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=_NODE_COLOR,
        node_size=node_size,
        edgecolors="0.3",
        linewidths=0.4,
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=4.5, font_color="white")

    # G label
    graph_xs = [p[0] for p in pos.values()]
    ax.text(
        min(graph_xs) - 0.035,
        center_y,
        r"$G$",
        ha="right",
        va="center",
        fontsize=8,
        color="0.3",
    )


def _draw_biarrow(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    label: str | None = None,
    label_side: str = "right",
) -> None:
    """Draw bidirectional arrow in data coordinates."""
    arrow = mpatches.FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="<->",
        color=_ARROW_COLOR,
        linewidth=0.8,
        mutation_scale=7,
        shrinkA=2,
        shrinkB=2,
        zorder=2,
    )
    ax.add_patch(arrow)

    if label:
        mx = (x0 + x1) / 2
        my = (y0 + y1) / 2
        offset = 0.025 if label_side == "right" else -0.025
        ax.text(
            mx + offset,
            my,
            label,
            ha="center",
            va="center",
            fontsize=5,
            color="0.45",
            fontstyle="italic",
        )


# =========================================================================
# Main panel generation
# =========================================================================


def draw_panel_a(fig: plt.Figure, gs: object) -> None:
    """Draw Panel A (encoding concept) into the given figure/gridspec region.

    Uses a single axes to avoid cross-axes arrow rendering issues.
    Layout: string (top=0.92) -> CDLL (mid=0.55) -> graph (bot=0.20).

    Args:
        fig: Matplotlib figure.
        gs: GridSpec or SubplotSpec to use for axes placement.
    """
    ax = fig.add_subplot(gs)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("auto")
    ax.axis("off")

    # Invisible anchors at corners to prevent bbox_inches='tight' cropping
    ax.text(0.0, 0.0, "", fontsize=1)
    ax.text(1.0, 1.0, "", fontsize=1)

    # Compute G2S string and S2G trace
    G = nx.house_graph()
    adapter = NetworkXAdapter()
    sg = adapter.from_external(G, directed=False)
    g2s = GraphToString(sg)
    w, _ = g2s.run(initial_node=0)
    logger.info("House graph G2S string: %r (len=%d)", w, len(w))

    # Run S2G to get final CDLL state
    s2g = StringToGraph(w, False)
    result_graph, trace = s2g.run(trace=True)
    final_graph, final_cdll, final_pri, final_sec, _ = trace[-1]

    # Verify round-trip
    G_result = adapter.to_external(result_graph)
    if not nx.is_isomorphic(G, G_result):
        logger.warning("Round-trip verification FAILED!")

    # Extract CDLL order
    cdll_order: list[int] = []
    node_idx = final_pri
    for _ in range(final_cdll.size()):
        cdll_order.append(final_cdll.get_value(node_idx))
        node_idx = final_cdll.next_node(node_idx)
    pri_payload = final_cdll.get_value(final_pri)
    sec_payload = final_cdll.get_value(final_sec)

    # Layout positions (in [0,1] x [0,1] axes coordinates)
    cx = 0.50
    y_string = 0.92
    y_cdll = 0.57
    y_graph = 0.17

    # Draw three zones — use cell_w=0.06 to fit 10 chars comfortably
    _draw_instruction_boxes(ax, w, cx, y_string, cell_w=0.06, cell_h=0.022)
    _draw_cdll_ring(
        ax,
        cdll_order,
        pri_payload,
        sec_payload,
        cx,
        y_cdll,
        radius=0.08,
        node_r=0.016,
    )
    _draw_graph(ax, G, cx, y_graph, scale=0.11)

    # Bidirectional arrows
    # String <-> CDLL: single vertical
    _draw_biarrow(ax, cx, y_string - 0.035, cx, y_cdll + 0.115)

    # CDLL <-> Graph: two diagonal arrows with S2G / G2S labels
    gap = 0.04
    _draw_biarrow(
        ax,
        cx - gap,
        y_cdll - 0.13,
        cx - gap,
        y_graph + 0.15,
        label="S2G",
        label_side="left",
    )
    _draw_biarrow(
        ax,
        cx + gap,
        y_cdll - 0.13,
        cx + gap,
        y_graph + 0.15,
        label="G2S",
        label_side="right",
    )


def generate_panel_a(output_dir: str) -> str:
    """Generate Panel A as a standalone figure.

    Args:
        output_dir: Directory to save output files.

    Returns:
        Base path of saved figure (without extension).
    """
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(2.0, 1.77))
    gs = GridSpec(1, 1, figure=fig, left=0.02, right=0.98, top=0.98, bottom=0.02)
    draw_panel_a(fig, gs[0])

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "panel_a_encoding")
    save_figure(fig, path, formats=("pdf", "svg", "png"))
    plt.close(fig)
    logger.info("Panel A saved: %s", path)
    return path


# =========================================================================
# Individual element exports (for Inkscape workflow)
# =========================================================================

_FORMATS = ("pdf", "svg", "png")


def _get_house_data() -> tuple:
    """Compute house graph data needed for individual exports."""
    G = nx.house_graph()
    adapter = NetworkXAdapter()
    sg = adapter.from_external(G, directed=False)
    g2s = GraphToString(sg)
    w, _ = g2s.run(initial_node=0)

    s2g = StringToGraph(w, False)
    _, trace = s2g.run(trace=True)
    final_cdll = trace[-1][1]
    final_pri = trace[-1][2]
    final_sec = trace[-1][3]

    cdll_order: list[int] = []
    node_idx = final_pri
    for _ in range(final_cdll.size()):
        cdll_order.append(final_cdll.get_value(node_idx))
        node_idx = final_cdll.next_node(node_idx)
    pri_payload = final_cdll.get_value(final_pri)
    sec_payload = final_cdll.get_value(final_sec)
    return G, w, cdll_order, pri_payload, sec_payload


def generate_string_standalone(output_dir: str) -> str:
    """Generate instruction string as a standalone figure."""
    _, w, _, _, _ = _get_house_data()

    fig, ax = plt.subplots(figsize=(3.0, 0.6))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.1, 0.1)
    ax.set_aspect("auto")
    ax.axis("off")
    _draw_instruction_boxes(ax, w, 0.50, 0.0, cell_w=0.08, cell_h=0.06)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "a_instruction_string")
    save_figure(fig, path, formats=_FORMATS)
    plt.close(fig)
    logger.info("Instruction string saved: %s", path)
    return path


def generate_cdll_standalone(output_dir: str) -> str:
    """Generate CDLL ring as a standalone figure."""
    _, _, cdll_order, pri_payload, sec_payload = _get_house_data()

    fig, ax = plt.subplots(figsize=(1.8, 1.8))
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    _draw_cdll_ring(
        ax,
        cdll_order,
        pri_payload,
        sec_payload,
        0.5,
        0.5,
        radius=0.3,
        node_r=0.06,
    )

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "a_cdll_ring")
    save_figure(fig, path, formats=_FORMATS)
    plt.close(fig)
    logger.info("CDLL ring saved: %s", path)
    return path


def generate_graph_standalone(output_dir: str) -> str:
    """Generate house graph as a standalone figure."""
    G, _, _, _, _ = _get_house_data()

    fig, ax = plt.subplots(figsize=(1.8, 1.8))
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    _draw_graph(ax, G, 0.5, 0.5, scale=0.35, node_size=250)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "a_house_graph")
    save_figure(fig, path, formats=_FORMATS)
    plt.close(fig)
    logger.info("House graph saved: %s", path)
    return path


def generate_all_standalone(output_dir: str) -> list[str]:
    """Generate all Panel A elements as individual files."""
    return [
        generate_string_standalone(output_dir),
        generate_cdll_standalone(output_dir),
        generate_graph_standalone(output_dir),
    ]


# =========================================================================
# CLI
# =========================================================================


def main() -> None:
    """CLI entry point for Panel A generation."""
    parser = argparse.ArgumentParser(
        description="Generate Panel A (encoding concept) for graphical abstract."
    )
    parser.add_argument(
        "--output-dir",
        default="paper_figures/graphical_abstract",
        help="Output directory.",
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="Also generate individual element files.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    apply_ieee_style()
    generate_panel_a(args.output_dir)
    if args.individual:
        generate_all_standalone(args.output_dir)


if __name__ == "__main__":
    main()
