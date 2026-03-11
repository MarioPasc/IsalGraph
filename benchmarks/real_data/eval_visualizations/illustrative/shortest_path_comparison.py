"""Shortest-path comparison: GED path vs. Levenshtein path between two graphs.

Generates a publication figure showing two panels:
  - Top panel (blue): Shortest path in GED space (graphs with edit highlights)
  - Bottom panel (red): Shortest path in Levenshtein space (decoded graphs + strings)

This illustrates a core property of IsalGraph: the string-space shortest path
induces a different sequence of intermediate structures than the graph-space
shortest path, even when the total distances are identical.

The figure is designed for IEEE/Elsevier D1-journal format (Information Systems),
matching the visual language of ``topology_and_complexity.py``.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import ConnectionPatch
from matplotlib.transforms import Bbox

from benchmarks.plotting_styles import INSTRUCTION_COLORS, PAUL_TOL_BRIGHT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CLR_ADDED = PAUL_TOL_BRIGHT["green"]  # added edge
_CLR_DELETED = PAUL_TOL_BRIGHT["red"]  # deleted edge
_CLR_UNCHANGED = "#4C5B7A"  # muted blue-grey — default edges
_CLR_NODE_FILL = "#E8EBF0"  # light blue-grey fill
_CLR_NODE_EDGE = "#4C5B7A"  # node border
_CLR_ADDED_NODE_FILL = "#C8E6C9"  # light green for added nodes
_CLR_ADDED_NODE_EDGE = PAUL_TOL_BRIGHT["green"]  # green border

_CLR_PANEL_GED = PAUL_TOL_BRIGHT["blue"]  # panel background / accent
_CLR_PANEL_LEV = PAUL_TOL_BRIGHT["red"]  # panel background / accent


# ---------------------------------------------------------------------------
# Graph pair construction
# ---------------------------------------------------------------------------


def build_example_pair() -> tuple[nx.Graph, nx.Graph]:
    """Build two graphs A, B with GED = Levenshtein = 5.

    G_A (5 nodes, 7 edges) and G_B (6 nodes, 7 edges) differ by 5 operations:
      GED path:  del_edge(0,3), del_edge(0,4), add_node(5),
                 add_edge(1,5), add_edge(3,5)
      Levenshtein: ins 'p', sub V->v, sub V->p, sub P->v, sub C->v

    Returns:
        (G_A, G_B) pair of NetworkX graphs.
    """
    G_A = nx.Graph()
    G_A.add_nodes_from(range(5))
    G_A.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 4), (3, 4)])

    G_B = nx.Graph()
    G_B.add_nodes_from(range(6))
    G_B.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 5), (2, 4), (3, 4), (3, 5)])

    return G_A, G_B


def compute_ged_path(
    G_A: nx.Graph,
    G_B: nx.Graph,
) -> list[tuple[nx.Graph, list[tuple[str, Any]]]]:
    """Compute the GED shortest path as a sequence of intermediate graphs.

    Edits are reordered for visual clarity:
      edge deletions -> node deletions -> node additions -> edge additions

    Returns:
        List of (graph, cumulative_edits) tuples.
    """
    for node_path, edge_path, cost in nx.optimize_edit_paths(
        G_A,
        G_B,
        node_subst_cost=lambda n1, n2: 0,
        node_del_cost=lambda n: 1,
        node_ins_cost=lambda n: 1,
        edge_subst_cost=lambda e1, e2: 0,
        edge_del_cost=lambda e: 1,
        edge_ins_cost=lambda e: 1,
    ):
        break

    # Extract raw edits
    raw_edits: list[tuple[str, Any]] = []
    node_map = {nb: na for na, nb in node_path if na is not None and nb is not None}

    for ea, eb in edge_path:
        if ea is None and eb is not None:
            u = node_map.get(eb[0], eb[0])
            v = node_map.get(eb[1], eb[1])
            raw_edits.append(("add_edge", (u, v)))
        elif eb is None and ea is not None:
            raw_edits.append(("del_edge", ea))

    for na, nb in node_path:
        if na is None and nb is not None:
            raw_edits.append(("add_node", nb))
        elif nb is None and na is not None:
            raw_edits.append(("del_node", na))

    # Reorder: edge_del -> node_del -> node_add -> edge_add
    priority = {"del_edge": 0, "del_node": 1, "add_node": 2, "add_edge": 3}
    edits = sorted(raw_edits, key=lambda x: priority.get(x[0], 99))

    path: list[tuple[nx.Graph, list[tuple[str, Any]]]] = [(G_A.copy(), [])]
    G_curr = G_A.copy()
    cumulative: list[tuple[str, Any]] = []

    for op_type, target in edits:
        if op_type == "add_edge":
            G_curr.add_edge(*target)
        elif op_type == "del_edge":
            G_curr.remove_edge(*target)
        elif op_type == "add_node":
            G_curr.add_node(target)
        elif op_type == "del_node":
            G_curr.remove_node(target)
        cumulative.append((op_type, target))
        path.append((G_curr.copy(), list(cumulative)))

    return path


def compute_levenshtein_path(
    w_A: str,
    w_B: str,
) -> list[tuple[str, str | None]]:
    """Compute the Levenshtein shortest path as intermediate strings.

    Returns:
        List of (intermediate_string, edit_description) tuples.
    """
    from benchmarks.eval_visualizations.string_alignment import levenshtein_alignment

    alignment = levenshtein_alignment(w_A, w_B)

    current = list(w_A)
    path: list[tuple[str, str | None]] = [("".join(current), None)]
    s_idx = 0

    for op, cs, ct in alignment:
        if op == "substitute":
            desc = f"sub '{cs}'\u2192'{ct}'"
            current[s_idx] = ct
            path.append(("".join(current), desc))
            s_idx += 1
        elif op == "delete":
            desc = f"del '{cs}'"
            current.pop(s_idx)
            path.append(("".join(current), desc))
        elif op == "insert":
            desc = f"ins '{ct}'"
            current.insert(s_idx, ct)
            path.append(("".join(current), desc))
            s_idx += 1
        elif op == "match":
            s_idx += 1

    return path


def _compute_edge_diff(
    G_prev: nx.Graph,
    G_curr: nx.Graph,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Compute edges added and removed between two graphs.

    Returns:
        (added_edges, removed_edges) as sets of sorted (u, v) tuples.
    """
    prev_edges = {tuple(sorted(e)) for e in G_prev.edges()}
    curr_edges = {tuple(sorted(e)) for e in G_curr.edges()}
    return curr_edges - prev_edges, prev_edges - curr_edges


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _get_shared_layout(graphs: list[nx.Graph], seed: int = 42) -> dict[int, np.ndarray]:
    """Compute a single layout shared across all graphs."""
    G_union = nx.Graph()
    for G in graphs:
        G_union.add_nodes_from(G.nodes())
        G_union.add_edges_from(G.edges())
    return nx.spring_layout(G_union, seed=seed, k=2.2, iterations=100)


def _draw_graph(
    ax: plt.Axes,
    G: nx.Graph,
    pos: dict[int, np.ndarray],
    *,
    added_edges: set[tuple[int, int]] | None = None,
    removed_edges: set[tuple[int, int]] | None = None,
    added_nodes: set[int] | None = None,
    node_size: int = 400,
    edge_width_default: float = 1.4,
    edge_alpha_default: float = 0.5,
    highlight_width: float = 3.5,
    font_size: int = 11,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Draw a graph with optional edge-diff and node highlighting."""
    if added_edges is None:
        added_edges = set()
    if removed_edges is None:
        removed_edges = set()
    if added_nodes is None:
        added_nodes = set()

    def _norm(e: tuple[int, int]) -> tuple[int, int]:
        return (min(e), max(e))

    added_norm = {_norm(e) for e in added_edges}
    removed_norm = {_norm(e) for e in removed_edges}

    # Unchanged edges
    unchanged = [e for e in G.edges() if _norm(e) not in added_norm]
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=unchanged,
        ax=ax,
        edge_color=_CLR_UNCHANGED,
        width=edge_width_default,
        alpha=edge_alpha_default,
    )

    # Added edges (thick green)
    added_in_graph = [e for e in G.edges() if _norm(e) in added_norm]
    if added_in_graph:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=added_in_graph,
            ax=ax,
            edge_color=_CLR_ADDED,
            width=highlight_width,
            alpha=1.0,
            style="solid",
        )

    # Ghost edges for removed (dashed red, no cross marker)
    for e in removed_norm:
        u, v = e
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=_CLR_DELETED,
                linewidth=2.8,
                linestyle=(0, (4, 3)),
                alpha=0.7,
                zorder=1,
            )

    # Nodes — separate added from regular
    regular_nodes = [n for n in G.nodes() if n not in added_nodes]
    new_nodes = [n for n in G.nodes() if n in added_nodes]

    if regular_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=regular_nodes,
            ax=ax,
            node_color=_CLR_NODE_FILL,
            node_size=node_size,
            edgecolors=_CLR_NODE_EDGE,
            linewidths=1.8,
        )
    if new_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=new_nodes,
            ax=ax,
            node_color=_CLR_ADDED_NODE_FILL,
            node_size=node_size,
            edgecolors=_CLR_ADDED_NODE_EDGE,
            linewidths=2.5,
        )

    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        font_size=font_size,
        font_weight="bold",
        font_color="#2C3E50",
    )
    ax.set_aspect("equal")
    ax.axis("off")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def _render_horizontal_heatmap(
    ax: plt.Axes,
    string: str,
    *,
    fontsize: int = 8,
) -> None:
    """Render instruction string as a horizontal row of colored cells.

    Matches the style from ``topology_and_complexity.py``: integer x-coords,
    aspect='auto', compact height.  Fontsize reduced for 6-column layout.
    """
    n = len(string)
    if n == 0:
        ax.axis("off")
        return

    for i, ch in enumerate(string):
        color = INSTRUCTION_COLORS.get(ch, "#000000")
        rect = mpatches.FancyBboxPatch(
            (i + 0.05, 0.075),
            0.9,
            0.85,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor="0.4",
            linewidth=0.3,
        )
        ax.add_patch(rect)
        ax.text(
            i + 0.5,
            0.5,
            ch,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontfamily="monospace",
            fontweight="bold",
            color="white",
        )

    ax.set_xlim(-0.1, n + 0.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect("auto")
    ax.axis("off")


def _add_arrow_between_axes(
    fig: plt.Figure,
    ax_from: plt.Axes,
    ax_to: plt.Axes,
    label: str,
    color: str,
    *,
    fontsize: int = 10,
) -> None:
    """Draw a horizontal arrow in the gap between two axes, at mid-height."""
    bbox_from = ax_from.get_position()
    bbox_to = ax_to.get_position()

    # Horizontal arrow at vertical centre of the allocated axes space
    y_mid = (bbox_from.y0 + bbox_from.y1) / 2
    x_start = bbox_from.x1 + 0.004
    x_end = bbox_to.x0 - 0.004

    con = ConnectionPatch(
        xyA=(x_start, y_mid),
        xyB=(x_end, y_mid),
        coordsA="figure fraction",
        coordsB="figure fraction",
        arrowstyle="-|>",
        color=color,
        linewidth=2.0,
        mutation_scale=15,
    )
    fig.add_artist(con)

    fig.text(
        (x_start + x_end) / 2,
        y_mid + 0.018,
        label,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color=color,
        fontweight="bold",
        fontstyle="italic",
        bbox=dict(
            boxstyle="round,pad=0.10",
            facecolor="white",
            edgecolor="none",
            alpha=0.85,
        ),
    )


def _add_background_panels(
    fig: plt.Figure,
    ged_axes: list[plt.Axes],
    lev_axes: list[plt.Axes],
    str_axes: list[plt.Axes],
) -> None:
    """Add semi-transparent background panels behind GED and Lev sections.

    Blue panel behind GED row, red panel behind Lev + string rows.
    Matches the visual language from ``topology_and_complexity.py``.
    """
    renderer = fig.canvas.get_renderer()
    fig.draw(renderer)

    pad = 0.00

    def _union_bbox(axes: list[plt.Axes]) -> Bbox | None:
        bboxes = []
        for ax in axes:
            bb = ax.get_tightbbox(renderer)
            if bb is not None:
                bboxes.append(bb.transformed(fig.transFigure.inverted()))
        return Bbox.union(bboxes) if bboxes else None

    panels = [
        ("ged", ged_axes, [], _CLR_PANEL_GED),
        ("lev", lev_axes, str_axes, _CLR_PANEL_LEV),
    ]

    for _name, graph_axes, extra_axes, color in panels:
        bbox = _union_bbox(graph_axes + extra_axes)
        if bbox is None:
            continue

        rect = mpatches.FancyBboxPatch(
            (bbox.x0 - pad, bbox.y0 - pad),
            bbox.width + 2 * pad,
            bbox.height + 2 * pad,
            boxstyle="round,pad=0.008",
            facecolor=color,
            alpha=0.06,
            edgecolor=color,
            linewidth=1.0,
            transform=fig.transFigure,
            zorder=0,
        )
        fig.patches.append(rect)


# ---------------------------------------------------------------------------
# Main figure generator
# ---------------------------------------------------------------------------


def generate_shortest_path_comparison(
    output_dir: str,
    *,
    seed: int = 42,
) -> str:
    """Generate the shortest-path comparison figure (d_GED = d_Lev = 5).

    Layout (nested GridSpec, 6 columns):
      Top panel (blue bg):   GED path graphs  (G_A -> G_1 -> ... -> G_B)
      Bottom panel (red bg): Lev path graphs + instruction strings

    Args:
        output_dir: Directory to save the figure.
        seed: Random seed for layout.

    Returns:
        Path to the saved figure.
    """
    from benchmarks.plotting_styles import apply_ieee_style, save_figure

    apply_ieee_style()

    G_A, G_B = build_example_pair()

    # Compute canonical strings
    from isalgraph.adapters.networkx_adapter import NetworkXAdapter
    from isalgraph.core.canonical import canonical_string
    from isalgraph.core.string_to_graph import StringToGraph

    adapter = NetworkXAdapter()
    sg_A = adapter.from_external(G_A, directed=False)
    sg_B = adapter.from_external(G_B, directed=False)
    w_A = canonical_string(sg_A)
    w_B = canonical_string(sg_B)

    logger.info("w_A = %s, w_B = %s, GED = 5, Lev = 5", w_A, w_B)

    # Compute paths
    ged_path = compute_ged_path(G_A, G_B)
    lev_path = compute_levenshtein_path(w_A, w_B)

    # Decode Levenshtein strings to graphs
    lev_graphs: list[nx.Graph] = []
    for w, _ in lev_path:
        s2g = StringToGraph(w, directed=False)
        sg, _ = s2g.run()
        lev_graphs.append(adapter.to_external(sg))

    # Use the original G_A / G_B for the Lev endpoints so they look
    # identical to the GED row endpoints (S2G round-trip preserves
    # isomorphism but not node labelling).
    lev_graphs[0] = G_A.copy()
    lev_graphs[-1] = G_B.copy()

    # Compute edge diffs for Lev-decoded graphs
    lev_edge_diffs: list[tuple[set[tuple[int, int]], set[tuple[int, int]]]] = [
        (set(), set()),
    ]
    for i in range(1, len(lev_graphs)):
        added, removed = _compute_edge_diff(lev_graphs[i - 1], lev_graphs[i])
        lev_edge_diffs.append((added, removed))

    # Shared layout across ALL graphs
    all_graphs = [g for g, _ in ged_path] + lev_graphs
    pos = _get_shared_layout(all_graphs, seed=seed)

    # Uniform data limits so every subplot occupies the same space
    all_xy = np.array(list(pos.values()))
    data_pad = 0.15
    xlim = (float(all_xy[:, 0].min() - data_pad), float(all_xy[:, 0].max() + data_pad))
    ylim = (float(all_xy[:, 1].min() - data_pad), float(all_xy[:, 1].max() + data_pad))

    n_cols = max(len(ged_path), len(lev_path))

    # --- Create figure with nested GridSpec ---
    fig_width = 18.0  # Wide landscape for 6 columns
    fig_height = 6.0  # Compact height
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Two main sections: GED panel and Lev panel
    gs_main = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[1, 1.45],
        hspace=0.05,
        left=0.05,
        right=0.98,
        top=0.92,
        bottom=0.04,
    )

    # GED section: single row of graphs
    gs_ged = GridSpecFromSubplotSpec(
        1,
        n_cols,
        subplot_spec=gs_main[0],
        wspace=0.40,
    )

    # Lev section: graph row + string row (tight vertical spacing)
    gs_lev = GridSpecFromSubplotSpec(
        2,
        n_cols,
        subplot_spec=gs_main[1],
        height_ratios=[5, 0.6],
        hspace=-0.07,
        wspace=0.40,
    )

    # === GED PATH GRAPHS ===
    ged_axes = []
    for i in range(len(ged_path)):
        ax = fig.add_subplot(gs_ged[0, i])
        ged_axes.append(ax)

        added_e, removed_e, added_n = set(), set(), set()
        if i > 0:
            last_edit = ged_path[i][1][-1]
            op_type, target = last_edit
            if op_type == "add_edge":
                added_e = {target}
            elif op_type == "del_edge":
                removed_e = {target}
            elif op_type == "add_node":
                added_n = {target}

        _draw_graph(
            ax,
            ged_path[i][0],
            pos,
            added_edges=added_e,
            removed_edges=removed_e,
            added_nodes=added_n,
            xlim=xlim,
            ylim=ylim,
        )

        if i == 0:
            label = r"$G_A$"
        elif i == len(ged_path) - 1:
            label = r"$G_B$"
        else:
            label = f"$G_{{{i}}}$"
        ax.set_title(label, fontsize=13, fontweight="bold", pad=6)

    # GED section title
    ged_bbox = gs_main[0].get_position(fig)
    fig.text(
        0.015,
        (ged_bbox.y0 + ged_bbox.y1) / 2,
        "(a) GED path",
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=_CLR_PANEL_GED,
        rotation=90,
    )

    # Arrows between GED columns
    for i in range(len(ged_path) - 1):
        edit = ged_path[i + 1][1][-1]
        op_type, target = edit
        if op_type == "add_edge":
            label = f"+e({target[0]},{target[1]})"
            color = _CLR_ADDED
        elif op_type == "del_edge":
            label = f"\u2212e({target[0]},{target[1]})"
            color = _CLR_DELETED
        elif op_type == "add_node":
            label = f"+n({target})"
            color = _CLR_ADDED
        elif op_type == "del_node":
            label = f"\u2212n({target})"
            color = _CLR_DELETED
        else:
            label, color = "", "black"
        _add_arrow_between_axes(fig, ged_axes[i], ged_axes[i + 1], label, color)

    # === LEV PATH GRAPHS (with edge-diff highlighting) ===
    lev_g_axes = []
    for i in range(len(lev_path)):
        ax = fig.add_subplot(gs_lev[0, i])
        lev_g_axes.append(ax)

        added_e, removed_e = lev_edge_diffs[i]
        _draw_graph(
            ax,
            lev_graphs[i],
            pos,
            added_edges=added_e,
            removed_edges=removed_e,
            highlight_width=2.5,
            xlim=xlim,
            ylim=ylim,
        )

        if i == 0:
            label = r"$\mathrm{S2G}(w_A^*)$"
        elif i == len(lev_path) - 1:
            label = r"$\mathrm{S2G}(w_B^*)$"
        else:
            label = r"$\mathrm{S2G}(w_{" + str(i) + r"})$"
        ax.set_title(label, fontsize=12, fontweight="bold", pad=6)

    # Lev section title
    lev_bbox = gs_main[1].get_position(fig)
    fig.text(
        0.015,
        (lev_bbox.y0 + lev_bbox.y1) / 2,
        "(b) Lev path",
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=_CLR_PANEL_LEV,
        rotation=90,
    )

    # Arrows between Lev graph columns (compact labels)
    for i in range(len(lev_path) - 1):
        desc = lev_path[i + 1][1] or ""
        if "ins" in desc:
            ch = desc.split("'")[1] if "'" in desc else "?"
            label = f"ins '{ch}'"
            color = PAUL_TOL_BRIGHT["blue"]
        elif "del" in desc:
            ch = desc.split("'")[1] if "'" in desc else "?"
            label = f"del '{ch}'"
            color = PAUL_TOL_BRIGHT["yellow"]
        elif "sub" in desc:
            # Extract chars: sub 'X'->'Y' -> X->Y
            parts = desc.replace("sub ", "").replace("'", "").split("\u2192")
            if len(parts) == 2:
                label = f"{parts[0]}\u2192{parts[1]}"
            else:
                label = desc
            color = _CLR_DELETED
        else:
            label, color = "", "black"
        _add_arrow_between_axes(fig, lev_g_axes[i], lev_g_axes[i + 1], label, color)

    # === LEV PATH STRINGS (horizontal heatmaps, tight below graphs) ===
    str_axes = []
    for i in range(len(lev_path)):
        ax = fig.add_subplot(gs_lev[1, i])
        str_axes.append(ax)
        _render_horizontal_heatmap(ax, lev_path[i][0])

    # === Background panels ===
    _add_background_panels(fig, ged_axes, lev_g_axes, str_axes)

    # === Title ===
    fig.suptitle(
        r"Shortest paths: Graph Edit Distance vs. Levenshtein distance"
        r" ($d_{\mathrm{GED}} = d_{\mathrm{Lev}} = 5$)",
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )

    # === Legend (bottom center) ===
    legend_elements = [
        plt.Line2D([0], [0], color=_CLR_ADDED, linewidth=3.0, label="Added edge"),
        plt.Line2D(
            [0],
            [0],
            color=_CLR_DELETED,
            linewidth=2.5,
            linestyle="--",
            label="Removed edge",
        ),
        plt.Line2D(
            [0],
            [0],
            color=_CLR_UNCHANGED,
            linewidth=1.2,
            alpha=0.5,
            label="Unchanged edge",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=_CLR_ADDED_NODE_FILL,
            markeredgecolor=_CLR_ADDED_NODE_EDGE,
            markersize=8,
            markeredgewidth=2,
            label="Added node",
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, -0.025),
    )

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "fig_shortest_path_comparison")
    saved = save_figure(fig, out_path)
    plt.close(fig)
    logger.info("Saved: %s", saved)

    # Generate LaTeX caption
    n_A = G_A.number_of_nodes()
    n_B = G_B.number_of_nodes()
    m_A = G_A.number_of_edges()
    m_B = G_B.number_of_edges()
    n_steps = len(ged_path) - 1

    ged_ops = [op for op, _ in ged_path[-1][1]]
    n_edge_del = ged_ops.count("del_edge")
    n_edge_add = ged_ops.count("add_edge")
    n_node_add = ged_ops.count("add_node")
    n_node_del = ged_ops.count("del_node")

    lev_ops = []
    for _, desc in lev_path[1:]:
        if desc and "ins" in desc:
            lev_ops.append("insertion")
        elif desc and "del" in desc:
            lev_ops.append("deletion")
        elif desc and "sub" in desc:
            lev_ops.append("substitution")
    n_ins = lev_ops.count("insertion")
    n_del = lev_ops.count("deletion")
    n_sub = lev_ops.count("substitution")

    ged_parts = []
    if n_edge_del:
        ged_parts.append(f"{n_edge_del} edge deletion{'s' if n_edge_del > 1 else ''}")
    if n_node_del:
        ged_parts.append(f"{n_node_del} node deletion{'s' if n_node_del > 1 else ''}")
    if n_node_add:
        ged_parts.append(f"{n_node_add} node insertion{'s' if n_node_add > 1 else ''}")
    if n_edge_add:
        ged_parts.append(f"{n_edge_add} edge insertion{'s' if n_edge_add > 1 else ''}")

    lev_parts = []
    if n_ins:
        lev_parts.append(f"{n_ins} insertion{'s' if n_ins > 1 else ''}")
    if n_del:
        lev_parts.append(f"{n_del} deletion{'s' if n_del > 1 else ''}")
    if n_sub:
        lev_parts.append(f"{n_sub} substitution{'s' if n_sub > 1 else ''}")

    caption = (
        r"\caption{Shortest paths in graph edit distance (GED) space vs.\ "
        r"Levenshtein distance space between two graphs "
        f"$G_A$ (${n_A}$ nodes, ${m_A}$ edges) and "
        f"$G_B$ (${n_B}$ nodes, ${m_B}$ edges) with "
        f"$d_{{\\mathrm{{GED}}}} = d_{{\\mathrm{{Lev}}}} = {n_steps}$. "
        r"\textbf{(a)}~The GED shortest path applies "
        f"{', '.join(ged_parts)}"
        r", producing a smooth structural transition. "
        r"\textbf{(b)}~The Levenshtein shortest path between the canonical "
        f"strings $w_A^* = \\texttt{{{w_A}}}$ and "
        f"$w_B^* = \\texttt{{{w_B}}}$ applies "
        f"{', '.join(lev_parts)}"
        r". Each intermediate string $w_i$ is decoded via $\mathrm{S2G}$ "
        r"to reveal its induced graph structure. "
        r"Notably, $G_i \ncong \mathrm{S2G}(w_i)$ for all intermediate "
        r"steps: the two paths traverse fundamentally different regions "
        r"of graph space despite starting and ending at the same pair "
        r"of graphs.}"
    )

    caption_path = os.path.join(output_dir, "fig_shortest_path_comparison_caption.tex")
    with open(caption_path, "w") as f:
        f.write(caption + "\n")
    logger.info("Caption: %s", caption_path)

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate shortest-path comparison figure (GED vs Levenshtein).",
    )
    parser.add_argument(
        "--output-dir",
        default="paper_figures/shortest_path",
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for graph layout.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    generate_shortest_path_comparison(args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
