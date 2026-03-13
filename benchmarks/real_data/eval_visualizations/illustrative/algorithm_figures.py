# ruff: noqa: N803, N806
"""Algorithm step-by-step illustration figures for the Methods section.

Generates three didactic figures showing StringToGraph and GraphToString
algorithms operating on a house graph example:
  - fig_s2g_walkthrough: Step-by-step S2G execution
  - fig_g2s_walkthrough: Step-by-step G2S execution
  - fig_algorithm_overview: Compact 2-column summary with round-trip
"""

from __future__ import annotations

import argparse
import logging
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from benchmarks.eval_visualizations.cdll_drawing import (
    draw_cdll_ring,
    get_legend_handles,
)
from benchmarks.eval_visualizations.graph_drawing import draw_graph
from benchmarks.plotting_styles import (
    INSTRUCTION_COLORS,
    apply_ieee_style,
    save_figure,
)
from isalgraph.adapters.networkx_adapter import NetworkXAdapter
from isalgraph.core.graph_to_string import GraphToString
from isalgraph.core.string_to_graph import StringToGraph

logger = logging.getLogger(__name__)

# =============================================================================
# CDLL state extraction
# =============================================================================


def _extract_cdll_order(cdll: object, start_node: int) -> list[int]:
    """Extract CDLL traversal order as list of payloads starting from a node."""
    order: list[int] = []
    node = start_node
    for _ in range(cdll.size()):  # type: ignore[union-attr]
        order.append(cdll.get_value(node))  # type: ignore[union-attr]
        node = cdll.next_node(node)  # type: ignore[union-attr]
    return order


def _sparse_to_nx(sg: object) -> nx.Graph:
    """Convert SparseGraph to NetworkX for drawing."""
    adapter = NetworkXAdapter()
    return adapter.to_external(sg)  # type: ignore[arg-type]


# =============================================================================
# Annotation helpers
# =============================================================================

_INSTR_DESCRIPTIONS: dict[str, str] = {
    "V": "New node + edge from primary",
    "v": "New node + edge from secondary",
    "C": "Edge: primary to secondary",
    "c": "Edge: secondary to primary",
    "N": "Move primary next",
    "P": "Move primary prev",
    "n": "Move secondary next",
    "p": "Move secondary prev",
    "W": "No-op",
}


def _annotate_step(prev_graph: object, curr_graph: object, instruction: str) -> str:
    """Generate brief annotation for a step."""
    desc = _INSTR_DESCRIPTIONS.get(instruction, "")
    prev_n = prev_graph.node_count() if prev_graph is not None else 0  # type: ignore[union-attr]
    curr_n = curr_graph.node_count()  # type: ignore[union-attr]
    prev_e = prev_graph.logical_edge_count() if prev_graph is not None else 0  # type: ignore[union-attr]
    curr_e = curr_graph.logical_edge_count()  # type: ignore[union-attr]

    parts = [desc]
    if curr_n > prev_n:
        parts.append(f"node {curr_n - 1}")
    if curr_e > prev_e:
        parts.append("+edge")
    return ", ".join(parts)


def _pick_n(trace_list: list, n: int) -> list[tuple[int, object]]:
    """Select *n* evenly-spaced snapshots (always includes first and last)."""
    total = len(trace_list)
    if total <= n:
        return [(i, t) for i, t in enumerate(trace_list)]
    indices = [round(i * (total - 1) / (n - 1)) for i in range(n)]
    return [(idx, trace_list[idx]) for idx in indices]


def _annotate_colored_prefix(
    ax: plt.Axes,
    prefix: str,
    *,
    fontsize: float = 5.5,
    y_offset: float = -0.10,
) -> None:
    """Show colored instruction prefix below an axes."""
    if not prefix:
        return
    n = len(prefix)
    char_spacing = min(0.055, 0.8 / max(n, 1))
    x_start = 0.5 - (n - 1) * char_spacing / 2
    for i, ch in enumerate(prefix):
        color = INSTRUCTION_COLORS.get(ch, "#000000")
        ax.text(
            x_start + i * char_spacing,
            y_offset,
            ch,
            transform=ax.transAxes,
            fontsize=fontsize,
            ha="center",
            fontfamily="monospace",
            fontweight="bold",
            color=color,
            clip_on=False,
        )


# =============================================================================
# Instruction heatmap rendering
# =============================================================================


def _render_instruction_heatmap(
    ax: plt.Axes,
    full_string: str,
    current_idx: int,
    *,
    cell_height: float = 0.6,
    cell_width: float = 0.5,
) -> None:
    """Render instruction string as a vertical column of colored cells.

    Each instruction is a colored rectangle stacked vertically (top to bottom).
    Completed instructions have full color; remaining are dimmed.

    Args:
        ax: Matplotlib axes.
        full_string: Complete instruction string.
        current_idx: Index of last completed instruction (-1 for none).
        cell_height: Height of each cell.
        cell_width: Width of each cell.
    """
    import matplotlib.patches as mpatches

    n = len(full_string)
    if n == 0:
        ax.axis("off")
        return

    for i, ch in enumerate(full_string):
        color = INSTRUCTION_COLORS.get(ch, "#000000")
        completed = i <= current_idx
        alpha = 1.0 if completed else 0.15
        y = (n - 1 - i) * cell_height  # Top-to-bottom

        rect = mpatches.FancyBboxPatch(
            (-cell_width / 2, y),
            cell_width,
            cell_height * 0.85,
            boxstyle="round,pad=0.02",
            facecolor=color,
            alpha=alpha,
            edgecolor="0.4" if completed else "0.8",
            linewidth=0.4,
        )
        ax.add_patch(rect)
        ax.text(
            0,
            y + cell_height * 0.425,
            ch,
            ha="center",
            va="center",
            fontsize=7,
            fontfamily="monospace",
            fontweight="bold" if i == current_idx else "normal",
            color="white" if completed else "0.6",
        )

    ax.set_xlim(-cell_width, cell_width)
    ax.set_ylim(-cell_height * 0.2, n * cell_height + cell_height * 0.1)
    ax.set_aspect("auto")
    ax.axis("off")


def _render_instruction_heatmap_horizontal(
    ax: plt.Axes,
    full_string: str,
    current_idx: int,
    *,
    cell_width: float = 0.6,
    cell_height: float = 0.5,
) -> None:
    """Render instruction string as a horizontal row of colored cells.

    Same visual style as the vertical heatmap but arranged left-to-right.
    Completed instructions have full color; remaining are dimmed (whitish).

    Args:
        ax: Matplotlib axes.
        full_string: Complete instruction string.
        current_idx: Index of last completed instruction (-1 for none).
        cell_width: Width of each cell.
        cell_height: Height of each cell.
    """
    import matplotlib.patches as mpatches

    n = len(full_string)
    if n == 0:
        ax.axis("off")
        return

    for i, ch in enumerate(full_string):
        color = INSTRUCTION_COLORS.get(ch, "#000000")
        completed = i <= current_idx
        alpha = 1.0 if completed else 0.15
        x = i * cell_width

        rect = mpatches.FancyBboxPatch(
            (x, -cell_height / 2),
            cell_width * 0.85,
            cell_height,
            boxstyle="round,pad=0.02",
            facecolor=color,
            alpha=alpha,
            edgecolor="0.4" if completed else "0.8",
            linewidth=0.4,
        )
        ax.add_patch(rect)
        ax.text(
            x + cell_width * 0.425,
            0,
            ch,
            ha="center",
            va="center",
            fontsize=6,
            fontfamily="monospace",
            fontweight="bold" if i == current_idx else "normal",
            color="white" if completed else "0.6",
        )

    ax.set_xlim(-cell_width * 0.2, n * cell_width + cell_width * 0.1)
    ax.set_ylim(-cell_height, cell_height)
    ax.set_aspect("auto")
    ax.axis("off")


def _get_instruction_legend_handles() -> list:
    """Return legend handles for the instruction color alphabet."""
    import matplotlib.patches as mpatches

    # Group by semantic category for clarity
    entries = ["V", "v", "C", "c", "N", "P", "n", "p"]
    return [mpatches.Patch(facecolor=INSTRUCTION_COLORS[ch], label=ch) for ch in entries]


# =============================================================================
# Figure 1: S2G Walkthrough
# =============================================================================


def generate_s2g_walkthrough(
    w: str,
    output_dir: str,
) -> str:
    """Generate step-by-step S2G walkthrough figure.

    Args:
        w: IsalGraph instruction string.
        output_dir: Output directory.

    Returns:
        Path to saved figure (without extension).
    """
    s2g = StringToGraph(w, False)
    _, trace = s2g.run(trace=True)

    n_steps = len(trace)
    row_height = 1.6
    fig_height = min(n_steps * row_height + 1.0, 18.0)

    fig, axes = plt.subplots(
        n_steps,
        3,
        figsize=(7.0, fig_height),
        gridspec_kw={"width_ratios": [0.8, 2.5, 3.5]},
    )
    if n_steps == 1:
        axes = axes[np.newaxis, :]

    # Compute fixed layout from final graph for consistency across steps
    final_graph_nx = _sparse_to_nx(trace[-1][0])
    fixed_pos = nx.spring_layout(final_graph_nx, seed=42)

    _GRAPH_NODE_SIZE = 150

    for step_idx, (graph, cdll, pri, sec, prefix) in enumerate(trace):
        ax_instr = axes[step_idx, 0]
        ax_cdll = axes[step_idx, 1]
        ax_graph = axes[step_idx, 2]

        # Column 1: Instruction heatmap
        current_char_idx = len(prefix) - 1
        _render_instruction_heatmap(ax_instr, w, current_char_idx)
        step_label = f"Step {step_idx}" if step_idx > 0 else "Init"
        ax_instr.set_ylabel(
            step_label,
            fontsize=7,
            rotation=0,
            labelpad=25,
            va="center",
        )

        # Column 2: CDLL ring
        cdll_order = _extract_cdll_order(cdll, pri)
        sec_payload = cdll.get_value(sec)
        pri_idx_in_order = 0
        sec_idx_in_order = cdll_order.index(sec_payload) if sec_payload in cdll_order else 0

        new_payload = None
        if step_idx > 0 and graph.node_count() > trace[step_idx - 1][0].node_count():
            new_payload = graph.node_count() - 1

        draw_cdll_ring(
            ax_cdll,
            cdll_order,
            pri_idx_in_order,
            sec_idx_in_order,
            new_node_payload=new_payload,
            radius=0.8,
            node_radius=0.18,
        )

        # Column 3: Graph with extra margin
        G_nx = _sparse_to_nx(graph)
        if G_nx.number_of_nodes() > 0:
            highlight = []
            if step_idx > 0:
                prev_graph = trace[step_idx - 1][0]
                prev_n = prev_graph.node_count()
                for u in range(graph.node_count()):
                    for v in graph.neighbors(u):
                        if u < v and (u >= prev_n or v >= prev_n or not prev_graph.has_edge(u, v)):
                            highlight.append((u, v))

            draw_graph(
                G_nx,
                ax_graph,
                highlight_edges=highlight or None,
                node_size=_GRAPH_NODE_SIZE,
                pos=fixed_pos,
            )
        else:
            ax_graph.axis("off")

        # Annotation
        if step_idx > 0 and prefix:
            instr = prefix[-1]
            prev_g = trace[step_idx - 1][0]
            note = _annotate_step(prev_g, graph, instr)
            ax_graph.set_title(note, fontsize=6, pad=2, style="italic")

    # Column headers on top row
    axes[0, 0].set_title("Instr.", fontsize=7, fontweight="bold")
    axes[0, 1].set_title("CDLL", fontsize=7, fontweight="bold")
    axes[0, 2].set_title("Graph", fontsize=7, fontweight="bold")

    # Shared legend at bottom: π/σ + new node
    fig.legend(
        handles=get_legend_handles(include_new_node=True),
        loc="lower center",
        ncol=3,
        fontsize=7,
        framealpha=0.8,
        bbox_to_anchor=(0.5, -0.01),
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    path = os.path.join(output_dir, "fig_s2g_walkthrough")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("S2G walkthrough saved: %s", path)
    return path


# =============================================================================
# Figure 2: G2S Walkthrough
# =============================================================================


def _draw_graph_with_ghosts(
    G_target: nx.Graph,
    inserted_nodes: set[int],
    inserted_edges: set[tuple[int, int]],
    ax: plt.Axes,
    pos: dict,
    *,
    node_size: int = 200,
) -> None:
    """Draw target graph with inserted parts solid and remaining dashed.

    Args:
        G_target: Full target graph.
        inserted_nodes: Nodes inserted so far.
        inserted_edges: Edges inserted so far (canonical order).
        ax: Matplotlib axes.
        pos: Pre-computed layout.
        node_size: Node size (consistent with S2G graphs).
    """
    ghost_edges = [
        (u, v) for u, v in G_target.edges() if (min(u, v), max(u, v)) not in inserted_edges
    ]
    nx.draw_networkx_edges(
        G_target,
        pos,
        edgelist=ghost_edges,
        ax=ax,
        edge_color="0.85",
        width=0.6,
        style="dashed",
    )

    real_edges = [(u, v) for u, v in G_target.edges() if (min(u, v), max(u, v)) in inserted_edges]
    nx.draw_networkx_edges(
        G_target,
        pos,
        edgelist=real_edges,
        ax=ax,
        edge_color="0.4",
        width=1.0,
    )

    ghost_nodes = [n for n in G_target.nodes() if n not in inserted_nodes]
    if ghost_nodes:
        nx.draw_networkx_nodes(
            G_target,
            pos,
            nodelist=ghost_nodes,
            ax=ax,
            node_color="white",
            node_size=node_size,
            edgecolors="0.7",
            linewidths=0.8,
        )
        nx.draw_networkx_labels(
            G_target,
            pos,
            labels={n: str(n) for n in ghost_nodes},
            ax=ax,
            font_size=7,
            font_color="0.7",
        )

    real_nodes = [n for n in G_target.nodes() if n in inserted_nodes]
    if real_nodes:
        nx.draw_networkx_nodes(
            G_target,
            pos,
            nodelist=real_nodes,
            ax=ax,
            node_color="#4477AA",
            node_size=node_size,
            edgecolors="0.3",
            linewidths=0.5,
        )
        nx.draw_networkx_labels(
            G_target,
            pos,
            labels={n: str(n) for n in real_nodes},
            ax=ax,
            font_size=7,
            font_color="white",
        )

    # Pad axes limits so node circles are never clipped
    if pos:
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 0.1)
        pad = 0.20 * span
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)

    ax.set_aspect("equal")
    ax.axis("off")


def generate_g2s_walkthrough(
    G_target: nx.Graph,
    initial_node: int,
    output_dir: str,
) -> str:
    """Generate step-by-step G2S walkthrough figure.

    Args:
        G_target: Target NetworkX graph.
        initial_node: Starting node for G2S.
        output_dir: Output directory.

    Returns:
        Path to saved figure (without extension).
    """
    adapter = NetworkXAdapter()
    sg = adapter.from_external(G_target, directed=False)

    g2s = GraphToString(sg)
    w, trace = g2s.run(initial_node=initial_node, trace=True)

    n_steps = len(trace)
    row_height = 1.6
    fig_height = min(n_steps * row_height + 1.0, 18.0)

    fig, axes = plt.subplots(
        n_steps,
        3,
        figsize=(7.0, fig_height),
        gridspec_kw={"width_ratios": [0.8, 2.5, 3.5]},
    )
    if n_steps == 1:
        axes = axes[np.newaxis, :]

    pos = nx.spring_layout(G_target, seed=42)

    _GRAPH_NODE_SIZE = 150

    for step_idx, (out_graph, cdll, pri, sec, prefix) in enumerate(trace):
        ax_instr = axes[step_idx, 0]
        ax_cdll = axes[step_idx, 1]
        ax_graph = axes[step_idx, 2]

        # Column 1: Instruction heatmap
        current_char_idx = len(prefix) - 1 if prefix else -1
        _render_instruction_heatmap(ax_instr, w, current_char_idx)
        step_label = f"Step {step_idx}" if step_idx > 0 else "Init"
        ax_instr.set_ylabel(
            step_label,
            fontsize=7,
            rotation=0,
            labelpad=25,
            va="center",
        )

        # Column 2: CDLL ring
        cdll_order = _extract_cdll_order(cdll, pri)
        pri_idx = 0
        sec_payload = cdll.get_value(sec)
        sec_idx = cdll_order.index(sec_payload) if sec_payload in cdll_order else 0

        new_payload = None
        if step_idx > 0 and out_graph.node_count() > trace[step_idx - 1][0].node_count():
            new_payload = out_graph.node_count() - 1

        draw_cdll_ring(
            ax_cdll,
            cdll_order,
            pri_idx,
            sec_idx,
            new_node_payload=new_payload,
            radius=0.8,
            node_radius=0.18,
        )

        # Column 3: Target graph with ghosts
        inserted_nodes: set[int] = set(range(out_graph.node_count()))
        inserted_edges: set[tuple[int, int]] = set()
        for u in range(out_graph.node_count()):
            for v in out_graph.neighbors(u):
                if u < v:
                    inserted_edges.add((u, v))

        _draw_graph_with_ghosts(
            G_target,
            inserted_nodes,
            inserted_edges,
            ax_graph,
            pos,
            node_size=_GRAPH_NODE_SIZE,
        )

        # Annotation
        if step_idx > 0 and prefix:
            prev_prefix = trace[step_idx - 1][4]
            new_chars = prefix[len(prev_prefix) :]
            structural = [c for c in new_chars if c in "VvCc"]
            if structural:
                ax_graph.set_title(
                    f"{new_chars} ({_INSTR_DESCRIPTIONS.get(structural[0], '')})",
                    fontsize=6,
                    pad=2,
                    style="italic",
                )

    # Column headers on top row
    axes[0, 0].set_title("Instr.", fontsize=7, fontweight="bold")
    axes[0, 1].set_title("CDLL", fontsize=7, fontweight="bold")
    axes[0, 2].set_title("Graph", fontsize=7, fontweight="bold")

    # Shared legend at bottom: instruction colors
    fig.legend(
        handles=_get_instruction_legend_handles(),
        loc="lower center",
        ncol=4,
        fontsize=6,
        framealpha=0.8,
        bbox_to_anchor=(0.5, -0.01),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    path = os.path.join(output_dir, "fig_g2s_walkthrough")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("G2S walkthrough saved: %s", path)
    return path


# =============================================================================
# Figure 3: Algorithm Overview (compact 2-column)
# =============================================================================


def _add_group_boxes(
    fig: plt.Figure,
    axes: np.ndarray,
    *,
    s2g_cols: tuple[int, ...] = (0, 1, 2),
    g2s_cols: tuple[int, ...] = (3, 4, 5),
) -> None:
    """Add grouped column boxes and vertical divider to the overview figure.

    Draws semi-transparent background rectangles spanning the S2G and G2S
    column groups, with group titles and sub-column labels, plus a vertical
    separator line between the two groups.

    Args:
        fig: The figure.
        axes: 2D array of axes (n_rows x 6).
        s2g_cols: Column indices for S2G group.
        g2s_cols: Column indices for G2S group.
    """
    import matplotlib.patches as mpatches
    from matplotlib.transforms import Bbox

    renderer = fig.canvas.get_renderer()
    fig.draw(renderer)

    def _axes_group_bbox(cols: tuple[int, ...]) -> Bbox:
        """Compute bounding box in figure coords spanning given columns."""
        bboxes = []
        n_rows = axes.shape[0]
        for r in range(n_rows):
            for c in cols:
                bb = axes[r, c].get_tightbbox(renderer)
                if bb is not None:
                    bboxes.append(bb.transformed(fig.transFigure.inverted()))
        return Bbox.union(bboxes)

    pad = 0.008  # Padding around group box

    # S2G group box
    s2g_bb = _axes_group_bbox(s2g_cols)
    s2g_rect = mpatches.FancyBboxPatch(
        (s2g_bb.x0 - pad, s2g_bb.y0 - pad),
        s2g_bb.width + 2 * pad,
        s2g_bb.height + 2 * pad,
        boxstyle="round,pad=0.005",
        facecolor="#EE6677",
        alpha=0.06,
        edgecolor="#EE6677",
        linewidth=1.0,
        transform=fig.transFigure,
        zorder=0,
    )
    fig.patches.append(s2g_rect)

    # G2S group box
    g2s_bb = _axes_group_bbox(g2s_cols)
    g2s_rect = mpatches.FancyBboxPatch(
        (g2s_bb.x0 - pad, g2s_bb.y0 - pad),
        g2s_bb.width + 2 * pad,
        g2s_bb.height + 2 * pad,
        boxstyle="round,pad=0.005",
        facecolor="#4477AA",
        alpha=0.06,
        edgecolor="#4477AA",
        linewidth=1.0,
        transform=fig.transFigure,
        zorder=0,
    )
    fig.patches.append(g2s_rect)

    # Group titles above boxes
    fig.text(
        s2g_bb.x0 + s2g_bb.width / 2,
        s2g_bb.y1 + pad + 0.015,
        "String-to-Graph (S2G)",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color="#CC3355",
        transform=fig.transFigure,
    )
    fig.text(
        g2s_bb.x0 + g2s_bb.width / 2,
        g2s_bb.y1 + pad + 0.015,
        "Graph-to-String (G2S)",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color="#335588",
        transform=fig.transFigure,
    )

    # Vertical divider between the two groups
    mid_x = (s2g_bb.x1 + g2s_bb.x0) / 2
    fig.add_artist(
        plt.Line2D(
            [mid_x, mid_x],
            [min(s2g_bb.y0, g2s_bb.y0) - pad, max(s2g_bb.y1, g2s_bb.y1) + pad],
            transform=fig.transFigure,
            color="0.5",
            linewidth=0.8,
            linestyle="--",
            zorder=1,
        )
    )


def _generate_overview_grid(
    s2g_trace: list,
    g2s_trace: list,
    w: str,
    w_g2s: str,
    G_target: nx.Graph,
    snaps_s2g: list[tuple[int, object]],
    snaps_g2s: list[tuple[int, object]],
    output_path: str,
) -> str:
    """Shared implementation for overview grids (3-snapshot and full).

    Args:
        s2g_trace: Full S2G trace (for layout computation).
        g2s_trace: Full G2S trace (unused directly, kept for symmetry).
        w: S2G instruction string.
        w_g2s: G2S instruction string.
        G_target: Target NetworkX graph.
        snaps_s2g: Selected S2G snapshots as (step_idx, trace_entry).
        snaps_g2s: Selected G2S snapshots as (step_idx, trace_entry).
        output_path: Full output path (without extension).

    Returns:
        Path to saved figure (without extension).
    """
    n_rows = max(len(snaps_s2g), len(snaps_g2s))
    _GRAPH_NODE_SIZE = 150

    row_height = 1.8 if n_rows <= 4 else 1.5
    fig_height = min(n_rows * row_height + 0.8, 22.0)

    fig, axes = plt.subplots(
        n_rows,
        6,
        figsize=(10.0, fig_height),
        gridspec_kw={
            "width_ratios": [1.8, 0.6, 3.5, 1.8, 0.6, 3.5],
            "wspace": 0.30,
        },
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    pos_target = nx.spring_layout(G_target, seed=42)
    final_s2g_nx = _sparse_to_nx(s2g_trace[-1][0])
    pos_s2g = nx.spring_layout(final_s2g_nx, seed=42)

    # S2G snapshots (left 3 columns)
    for row, (_step_idx, (graph, cdll, pri, sec, prefix)) in enumerate(
        snaps_s2g,
    ):
        ax_cdll = axes[row, 0]
        ax_instr = axes[row, 1]
        ax_graph = axes[row, 2]

        cdll_order = _extract_cdll_order(cdll, pri)
        pri_idx = 0
        sec_payload = cdll.get_value(sec)
        sec_idx = cdll_order.index(sec_payload) if sec_payload in cdll_order else 0

        draw_cdll_ring(
            ax_cdll,
            cdll_order,
            pri_idx,
            sec_idx,
            radius=0.7,
            node_radius=0.15,
        )
        _render_instruction_heatmap(
            ax_instr,
            w,
            len(prefix) - 1,
            cell_height=0.5,
            cell_width=0.4,
        )

        G_nx = _sparse_to_nx(graph)
        if G_nx.number_of_nodes() > 0:
            draw_graph(
                G_nx,
                ax_graph,
                node_size=_GRAPH_NODE_SIZE,
                pos=pos_s2g,
            )
        else:
            ax_graph.axis("off")

    # Hide unused S2G rows (if G2S has more steps)
    for row in range(len(snaps_s2g), n_rows):
        for c in range(3):
            axes[row, c].axis("off")

    # G2S snapshots (right 3 columns)
    for row, (_step_idx, (out_graph, cdll, pri, sec, prefix)) in enumerate(
        snaps_g2s,
    ):
        ax_cdll = axes[row, 3]
        ax_instr = axes[row, 4]
        ax_graph = axes[row, 5]

        cdll_order = _extract_cdll_order(cdll, pri)
        pri_idx = 0
        sec_payload = cdll.get_value(sec)
        sec_idx = cdll_order.index(sec_payload) if sec_payload in cdll_order else 0

        draw_cdll_ring(
            ax_cdll,
            cdll_order,
            pri_idx,
            sec_idx,
            radius=0.7,
            node_radius=0.15,
        )
        current_char_idx = len(prefix) - 1 if prefix else -1
        _render_instruction_heatmap(
            ax_instr,
            w_g2s,
            current_char_idx,
            cell_height=0.5,
            cell_width=0.4,
        )

        inserted_nodes = set(range(out_graph.node_count()))
        inserted_edges: set[tuple[int, int]] = set()
        for u in range(out_graph.node_count()):
            for v in out_graph.neighbors(u):
                if u < v:
                    inserted_edges.add((u, v))

        _draw_graph_with_ghosts(
            G_target,
            inserted_nodes,
            inserted_edges,
            ax_graph,
            pos_target,
            node_size=_GRAPH_NODE_SIZE,
        )

    # Hide unused G2S rows
    for row in range(len(snaps_g2s), n_rows):
        for c in range(3, 6):
            axes[row, c].axis("off")

    # Layout, then add decorations
    # Reserve just enough bottom space for the single-line legend
    legend_height_inches = 0.3
    legend_margin = legend_height_inches / fig_height
    plt.tight_layout(rect=[0, legend_margin, 1, 0.92])

    # Sub-column headers — use ax titles (same pad ensures alignment)
    for col, label in [
        (0, "CDLL"),
        (1, "Instr."),
        (2, "Graph"),
        (3, "CDLL"),
        (4, "Instr."),
        (5, "Graph"),
    ]:
        axes[0, col].set_title(label, fontsize=7, fontweight="bold", pad=8)

    # Group boxes and vertical divider
    _add_group_boxes(fig, axes)

    # Single legend: π/σ only — anchor just below the lowest axes row
    fig.legend(
        handles=get_legend_handles(include_new_node=False),
        loc="lower center",
        ncol=2,
        fontsize=6,
        framealpha=0.8,
        bbox_to_anchor=(0.5, 0.08),
    )

    save_figure(fig, output_path)
    plt.close(fig)
    return output_path


# =============================================================================
# Figure 3b: Algorithm Overview — Horizontal layout (5 snapshots)
# =============================================================================


def _add_group_boxes_horizontal(
    fig: plt.Figure,
    s2g_axes: list[list[plt.Axes]],
    g2s_axes: list[list[plt.Axes]],
) -> None:
    """Add grouped boxes and labels for the horizontal overview layout.

    Args:
        fig: The figure.
        s2g_axes: 2-element list of [cdll_axes_row, graph_axes_row].
        g2s_axes: 2-element list of [cdll_axes_row, graph_axes_row].
    """
    import matplotlib.patches as mpatches
    from matplotlib.transforms import Bbox

    renderer = fig.canvas.get_renderer()
    fig.draw(renderer)

    pad = 0.008

    def _group_bbox(axes_rows: list[list[plt.Axes]]) -> Bbox:
        bboxes = []
        for row in axes_rows:
            for ax in row:
                bb = ax.get_tightbbox(renderer)
                if bb is not None:
                    bboxes.append(bb.transformed(fig.transFigure.inverted()))
        return Bbox.union(bboxes)

    s2g_bb = _group_bbox(s2g_axes)
    fig.patches.append(
        mpatches.FancyBboxPatch(
            (s2g_bb.x0 - pad, s2g_bb.y0 - pad),
            s2g_bb.width + 2 * pad,
            s2g_bb.height + 2 * pad,
            boxstyle="round,pad=0.005",
            facecolor="#EE6677",
            alpha=0.06,
            edgecolor="#EE6677",
            linewidth=1.0,
            transform=fig.transFigure,
            zorder=0,
        )
    )

    g2s_bb = _group_bbox(g2s_axes)
    fig.patches.append(
        mpatches.FancyBboxPatch(
            (g2s_bb.x0 - pad, g2s_bb.y0 - pad),
            g2s_bb.width + 2 * pad,
            g2s_bb.height + 2 * pad,
            boxstyle="round,pad=0.005",
            facecolor="#4477AA",
            alpha=0.06,
            edgecolor="#4477AA",
            linewidth=1.0,
            transform=fig.transFigure,
            zorder=0,
        )
    )

    # Group titles (rotated, to the left of boxes)
    fig.text(
        s2g_bb.x0 - pad - 0.025,
        s2g_bb.y0 + s2g_bb.height / 2,
        "String-to-Graph (S2G)",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="#CC3355",
        transform=fig.transFigure,
        rotation=90,
    )
    fig.text(
        g2s_bb.x0 - pad - 0.025,
        g2s_bb.y0 + g2s_bb.height / 2,
        "Graph-to-String (G2S)",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="#335588",
        transform=fig.transFigure,
        rotation=90,
    )

    # Horizontal divider between the two groups
    mid_y = (s2g_bb.y0 + g2s_bb.y1) / 2
    fig.add_artist(
        plt.Line2D(
            [max(s2g_bb.x0, g2s_bb.x0) - pad, min(s2g_bb.x1, g2s_bb.x1) + pad],
            [mid_y, mid_y],
            transform=fig.transFigure,
            color="0.5",
            linewidth=0.8,
            linestyle="--",
            zorder=1,
        )
    )


def _generate_overview_grid_horizontal(
    s2g_trace: list,
    g2s_trace: list,
    w: str,
    w_g2s: str,
    G_target: nx.Graph,
    snaps_s2g: list[tuple[int, object]],
    snaps_g2s: list[tuple[int, object]],
    output_path: str,
) -> str:
    """Horizontal overview: steps progress left-to-right, S2G above G2S.

    Layout per group (3 rows x n_cols):
        Row 0: CDLL rings
        Row 1: Instruction heatmap (horizontal colored blocks)
        Row 2: Graphs

    Args:
        s2g_trace: Full S2G trace (for layout computation).
        g2s_trace: Full G2S trace (unused directly, kept for symmetry).
        w: S2G instruction string.
        w_g2s: G2S instruction string.
        G_target: Target NetworkX graph.
        snaps_s2g: Selected S2G snapshots as (step_idx, trace_entry).
        snaps_g2s: Selected G2S snapshots as (step_idx, trace_entry).
        output_path: Full output path (without extension).

    Returns:
        Path to saved figure (without extension).
    """
    from matplotlib.gridspec import GridSpec

    n_cols = max(len(snaps_s2g), len(snaps_g2s))
    _GRAPH_NODE_SIZE = 150

    fig = plt.figure(figsize=(2.0 * n_cols + 0.5, 7.5))

    outer_gs = GridSpec(
        2,
        1,
        figure=fig,
        hspace=0.35,
        top=0.92,
        bottom=0.06,
        left=0.08,
        right=0.97,
    )

    # 3 rows per group: CDLL, Instruction strip, Graph
    gs_s2g = outer_gs[0].subgridspec(
        3, n_cols, height_ratios=[1.0, 0.30, 1.6], hspace=0.15, wspace=0.20
    )
    gs_g2s = outer_gs[1].subgridspec(
        3, n_cols, height_ratios=[1.0, 0.30, 1.6], hspace=0.15, wspace=0.20
    )

    s2g_cdll_axes = [fig.add_subplot(gs_s2g[0, c]) for c in range(n_cols)]
    s2g_instr_axes = [fig.add_subplot(gs_s2g[1, c]) for c in range(n_cols)]
    s2g_graph_axes = [fig.add_subplot(gs_s2g[2, c]) for c in range(n_cols)]
    g2s_cdll_axes = [fig.add_subplot(gs_g2s[0, c]) for c in range(n_cols)]
    g2s_instr_axes = [fig.add_subplot(gs_g2s[1, c]) for c in range(n_cols)]
    g2s_graph_axes = [fig.add_subplot(gs_g2s[2, c]) for c in range(n_cols)]

    # Compute fixed layouts from final states
    pos_target = nx.spring_layout(G_target, seed=42)
    final_s2g_nx = _sparse_to_nx(s2g_trace[-1][0])
    pos_s2g = nx.spring_layout(final_s2g_nx, seed=42)

    # ---- S2G snapshots ----
    for col, (step_idx, (graph, cdll, pri, sec, prefix)) in enumerate(snaps_s2g):
        # Row 0: CDLL ring
        cdll_order = _extract_cdll_order(cdll, pri)
        sec_payload = cdll.get_value(sec)
        sec_idx = cdll_order.index(sec_payload) if sec_payload in cdll_order else 0
        draw_cdll_ring(
            s2g_cdll_axes[col],
            cdll_order,
            0,
            sec_idx,
            radius=0.7,
            node_radius=0.15,
        )
        label = "Init" if step_idx == 0 else f"Step {step_idx}"
        s2g_cdll_axes[col].set_title(label, fontsize=7, fontweight="bold", pad=3)

        # Row 1: Horizontal instruction heatmap
        current_char_idx = len(prefix) - 1
        _render_instruction_heatmap_horizontal(s2g_instr_axes[col], w, current_char_idx)

        # Row 2: Graph
        G_nx = _sparse_to_nx(graph)
        if G_nx.number_of_nodes() > 0:
            draw_graph(G_nx, s2g_graph_axes[col], node_size=_GRAPH_NODE_SIZE, pos=pos_s2g)
        else:
            s2g_graph_axes[col].axis("off")

    # ---- G2S snapshots ----
    for col, (step_idx, (out_graph, cdll, pri, sec, prefix)) in enumerate(snaps_g2s):
        # Row 0: CDLL ring
        cdll_order = _extract_cdll_order(cdll, pri)
        sec_payload = cdll.get_value(sec)
        sec_idx = cdll_order.index(sec_payload) if sec_payload in cdll_order else 0
        draw_cdll_ring(
            g2s_cdll_axes[col],
            cdll_order,
            0,
            sec_idx,
            radius=0.7,
            node_radius=0.15,
        )
        label = "Init" if step_idx == 0 else f"Step {step_idx}"
        g2s_cdll_axes[col].set_title(label, fontsize=7, fontweight="bold", pad=3)

        # Row 1: Horizontal instruction heatmap
        current_char_idx = len(prefix) - 1 if prefix else -1
        _render_instruction_heatmap_horizontal(g2s_instr_axes[col], w_g2s, current_char_idx)

        # Row 2: Graph with ghosts
        inserted_nodes: set[int] = set(range(out_graph.node_count()))
        inserted_edges: set[tuple[int, int]] = set()
        for u in range(out_graph.node_count()):
            for v in out_graph.neighbors(u):
                if u < v:
                    inserted_edges.add((u, v))

        _draw_graph_with_ghosts(
            G_target,
            inserted_nodes,
            inserted_edges,
            g2s_graph_axes[col],
            pos_target,
            node_size=_GRAPH_NODE_SIZE,
        )

    # Hide unused columns
    for col in range(len(snaps_s2g), n_cols):
        s2g_cdll_axes[col].axis("off")
        s2g_instr_axes[col].axis("off")
        s2g_graph_axes[col].axis("off")
    for col in range(len(snaps_g2s), n_cols):
        g2s_cdll_axes[col].axis("off")
        g2s_instr_axes[col].axis("off")
        g2s_graph_axes[col].axis("off")

    # Group boxes and divider
    _add_group_boxes_horizontal(
        fig,
        [s2g_cdll_axes, s2g_instr_axes, s2g_graph_axes],
        [g2s_cdll_axes, g2s_instr_axes, g2s_graph_axes],
    )

    # Single legend at bottom
    fig.legend(
        handles=get_legend_handles(include_new_node=False),
        loc="lower center",
        ncol=2,
        fontsize=6,
        framealpha=0.8,
        bbox_to_anchor=(0.5, 0.0),
    )

    save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def generate_algorithm_overview(
    w: str,
    G_target: nx.Graph,
    initial_node: int,
    output_dir: str,
) -> str:
    """Generate horizontal algorithm overview (5 snapshots each, left-to-right).

    Args:
        w: IsalGraph instruction string.
        G_target: Target NetworkX graph.
        initial_node: Starting node for G2S.
        output_dir: Output directory.

    Returns:
        Path to saved figure (without extension).
    """
    s2g = StringToGraph(w, False)
    _, s2g_trace = s2g.run(trace=True)

    adapter = NetworkXAdapter()
    sg = adapter.from_external(G_target, directed=False)
    g2s = GraphToString(sg)
    w_g2s, g2s_trace = g2s.run(initial_node=initial_node, trace=True)

    path = os.path.join(output_dir, "fig_algorithm_overview")
    result = _generate_overview_grid_horizontal(
        s2g_trace,
        g2s_trace,
        w,
        w_g2s,
        G_target,
        _pick_n(s2g_trace, 5),
        _pick_n(g2s_trace, 5),
        path,
    )
    logger.info("Algorithm overview saved: %s", result)
    return result


def generate_algorithm_overview_full(
    w: str,
    G_target: nx.Graph,
    initial_node: int,
    output_dir: str,
) -> str:
    """Generate full-step 2-column algorithm overview (all steps).

    Same layout as the compact overview, but every trace snapshot is shown.

    Args:
        w: IsalGraph instruction string.
        G_target: Target NetworkX graph.
        initial_node: Starting node for G2S.
        output_dir: Output directory.

    Returns:
        Path to saved figure (without extension).
    """
    s2g = StringToGraph(w, False)
    _, s2g_trace = s2g.run(trace=True)

    adapter = NetworkXAdapter()
    sg = adapter.from_external(G_target, directed=False)
    g2s = GraphToString(sg)
    w_g2s, g2s_trace = g2s.run(initial_node=initial_node, trace=True)

    all_s2g = [(i, t) for i, t in enumerate(s2g_trace)]
    all_g2s = [(i, t) for i, t in enumerate(g2s_trace)]

    path = os.path.join(output_dir, "fig_algorithm_overview_full")
    result = _generate_overview_grid(
        s2g_trace,
        g2s_trace,
        w,
        w_g2s,
        G_target,
        all_s2g,
        all_g2s,
        path,
    )
    logger.info("Algorithm overview (full) saved: %s", result)
    return result


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """CLI entry point for algorithm illustration figures."""
    parser = argparse.ArgumentParser(
        description="Generate algorithm step-by-step illustration figures."
    )
    parser.add_argument(
        "--output-dir",
        default="paper_figures/algorithm_illustration",
        help="Output directory for figures.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    apply_ieee_style()
    os.makedirs(args.output_dir, exist_ok=True)

    # Example: house graph (5 nodes, 6 edges — square + triangle)
    G = nx.house_graph()
    adapter = NetworkXAdapter()
    sg = adapter.from_external(G, directed=False)

    # Get verified G2S string
    g2s = GraphToString(sg)
    w, _ = g2s.run(initial_node=0)
    logger.info("House graph G2S string: %r (len=%d)", w, len(w))

    # Generate all 4 figures
    generate_s2g_walkthrough(w, args.output_dir)
    generate_g2s_walkthrough(G, initial_node=0, output_dir=args.output_dir)
    generate_algorithm_overview(w, G, initial_node=0, output_dir=args.output_dir)
    generate_algorithm_overview_full(w, G, initial_node=0, output_dir=args.output_dir)

    logger.info("All algorithm figures generated in %s", args.output_dir)


if __name__ == "__main__":
    main()
