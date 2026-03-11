# ruff: noqa: N803, N806, E402
"""Publication figures: graph-space topology and empirical complexity.

Figure 1 -- Neighbourhood grid:
    For a base graph G₀, display its GED-1 neighbours (edge edits) and its
    Levenshtein-close neighbours side by side, showing that the two notions
    of closeness largely agree.

Figure 2 -- Distance field:
    Conceptual 2D map of graph space around G₀, with GED on one axis and
    Levenshtein on the other.  Graph thumbnails placed at their (GED, Lev)
    coordinates illustrate how the two metrics define different but
    correlated "directions" in graph space.

Figure 3 -- Empirical complexity:
    Measure CPU time of greedy single-start, greedy-min, exhaustive
    canonical encoding, and exact GED on random connected graphs of
    increasing size.  X = n (node count), Y = time (log scale).
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import time
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.gridspec import GridSpec

from benchmarks.plotting_styles import (
    INSTRUCTION_COLORS,
    PAUL_TOL_BRIGHT,
    apply_ieee_style,
    get_figure_size,
    render_colored_string,
    save_figure,
)
from isalgraph.adapters.networkx_adapter import NetworkXAdapter
from isalgraph.core.canonical import canonical_string as compute_canonical
from isalgraph.core.graph_to_string import GraphToString
from isalgraph.core.string_to_graph import StringToGraph

logger = logging.getLogger(__name__)

_ADAPTER = NetworkXAdapter()
_ALPHABET = list("NnPpVvCcW")


# ============================================================================
# Shared helpers
# ============================================================================


def _levenshtein(s1: str, s2: str) -> int:
    """Levenshtein edit distance (pure-Python, fine for short strings)."""
    n, m = len(s1), len(s2)
    if n < m:
        return _levenshtein(s2, s1)
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(n):
        curr = [i + 1]
        for j in range(m):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (s1[i] != s2[j])))
        prev = curr
    return prev[m]


def _nx_to_sg(G: nx.Graph):  # type: ignore[type-arg]
    return _ADAPTER.from_external(G, directed=False)


def _greedy_min(G: nx.Graph) -> str:
    """Greedy-min encoding: shortest then lexmin across all starting nodes."""
    sg = _nx_to_sg(G)
    best: str | None = None
    for v in range(sg.node_count()):
        try:
            gts = GraphToString(sg)
            s, _ = gts.run(initial_node=v)
        except (ValueError, RuntimeError):
            continue
        if best is None or len(s) < len(best) or (len(s) == len(best) and s < best):
            best = s
    assert best is not None
    return best


def _decode(w: str) -> nx.Graph | None:
    """Decode an IsalGraph string to a NetworkX graph; None on failure."""
    try:
        s2g = StringToGraph(w, directed_graph=False)
        sg, _ = s2g.run()
        G = _ADAPTER.to_external(sg)
        if G.number_of_nodes() == 0:
            return None
        return G
    except Exception:  # noqa: BLE001
        return None


def _ged(G1: nx.Graph, G2: nx.Graph) -> int:
    """Exact GED between two small unlabelled graphs."""
    return int(
        nx.graph_edit_distance(
            G1,
            G2,
            node_subst_cost=lambda _a, _b: 0,
            edge_subst_cost=lambda _a, _b: 0,
        )
    )


# ============================================================================
# Figure 1 — Neighbourhood topology
# ============================================================================


def _ged1_neighbors(
    G0: nx.Graph,
    w0: str,
) -> list[dict[str, Any]]:
    """All non-isomorphic GED-1 neighbours of G0 (edge add / remove)."""
    results: list[dict[str, Any]] = []
    seen: list[nx.Graph] = []

    def _try(G: nx.Graph, edit_type: str, edge: tuple[int, int]) -> None:
        for G_prev in seen:
            if nx.is_isomorphic(G, G_prev):
                return
        if nx.is_isomorphic(G, G0):
            return
        seen.append(G)
        w = _greedy_min(G)
        results.append(
            {
                "graph": G,
                "string": w,
                "lev": _levenshtein(w0, w),
                "edit_type": edit_type,
                "edge": edge,
            }
        )

    # Edge deletions
    for u, v in list(G0.edges()):
        G = G0.copy()
        G.remove_edge(u, v)
        if nx.is_connected(G):
            _try(G, "del", (u, v))

    # Edge additions
    for u, v in nx.non_edges(G0):
        G = G0.copy()
        G.add_edge(u, v)
        _try(G, "add", (u, v))

    results.sort(key=lambda d: (d["lev"], d["string"]))
    return results


def _lev1_neighbors(
    G0: nx.Graph,
    w0: str,
    *,
    max_results: int = 8,
    exclude: list[nx.Graph] | None = None,
) -> list[dict[str, Any]]:
    """Find non-isomorphic Lev-1 neighbours by direct string perturbation.

    Every IsalGraph string over Sigma={N,n,P,p,V,v,C,c,W} decodes to a
    valid graph (the instruction set is total), so single-character edits
    always produce decodable strings.  This function enumerates all
    substitutions, deletions, and insertions of w0, decodes each, and
    returns the non-isomorphic connected neighbours sorted by GED.
    """
    exclude_list: list[nx.Graph] = [G0] + (exclude or [])
    seen: list[nx.Graph] = list(exclude_list)
    candidates: list[dict[str, Any]] = []
    m = len(w0)

    perturbations: list[str] = []
    # Substitutions
    for i in range(m):
        for c in _ALPHABET:
            if c != w0[i]:
                perturbations.append(w0[:i] + c + w0[i + 1 :])
    # Deletions
    for i in range(m):
        perturbations.append(w0[:i] + w0[i + 1 :])
    # Insertions
    for i in range(m + 1):
        for c in _ALPHABET:
            perturbations.append(w0[:i] + c + w0[i:])

    for wp in perturbations:
        Gp = _decode(wp)
        if Gp is None or not nx.is_connected(Gp):
            continue
        # Deduplicate by isomorphism
        if any(nx.is_isomorphic(Gp, prev) for prev in seen):
            continue
        seen.append(Gp)
        candidates.append({"graph": Gp, "string": wp, "lev": 1})

    # Compute exact GED for candidates
    for d in candidates:
        d["ged"] = _ged(G0, d["graph"])

    candidates.sort(key=lambda d: (d["ged"], d["string"]))
    return candidates[:max_results]


def _draw_cell(
    ax: plt.Axes,
    G: nx.Graph,
    pos: dict,
    w: str,
    *,
    title: str = "",
    ghost_edge: tuple[int, int] | None = None,
    highlight_edge: tuple[int, int] | None = None,
    node_size: int = 150,
) -> None:
    """Draw one graph cell with string annotation."""
    from benchmarks.eval_visualizations.graph_drawing import _DEFAULT_NODE_COLOR

    # Normal edges
    all_edges = list(G.edges())
    hl_set: set[tuple[int, int]] = set()
    if highlight_edge is not None:
        hl_set = {(min(highlight_edge), max(highlight_edge))}

    normal = [e for e in all_edges if (min(e), max(e)) not in hl_set]
    nx.draw_networkx_edges(G, pos, edgelist=normal, edge_color="0.5", width=0.8, ax=ax)

    # Highlighted (added) edge in red
    if highlight_edge is not None:
        hl = [e for e in all_edges if (min(e), max(e)) in hl_set]
        if hl:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=hl,
                edge_color=PAUL_TOL_BRIGHT["red"],
                width=1.5,
                ax=ax,
            )

    # Ghost (removed) edge as dashed red
    if ghost_edge is not None:
        u, v = ghost_edge
        if u in pos and v in pos:
            ax.plot(
                [pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                "--",
                color=PAUL_TOL_BRIGHT["red"],
                linewidth=1.0,
                zorder=0,
                alpha=0.6,
            )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=_DEFAULT_NODE_COLOR,
        node_size=node_size,
        edgecolors="0.3",
        linewidths=0.5,
        ax=ax,
    )
    nx.draw_networkx_labels(G, pos, font_size=6, font_color="white", ax=ax)

    if title:
        ax.set_title(title, fontsize=7, pad=3)

    # Axis limits (same padding as draw_graph)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 0.1)
    pad = 0.25 * span
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    y_lo = min(ys) - pad
    y_hi = max(ys) + pad
    # Extra bottom space for the string
    ax.set_ylim(y_lo - 0.35 * span, y_hi)

    # Render coloured string below graph
    x_center = np.mean(xs)
    render_colored_string(ax, w, x=x_center, y=min(ys) - 0.22 * span, fontsize=5, mono=True)

    ax.set_aspect("equal")
    ax.axis("off")


# ---- Enhanced neighbourhood figure helpers --------------------------------


def _render_horizontal_heatmap(
    ax: plt.Axes,
    string: str,
    *,
    fontsize: int = 6,
) -> None:
    """Render instruction string as a horizontal row of colored cells.

    Adapts the vertical heatmap from ``algorithm_figures.py`` to horizontal
    layout.  Each instruction character becomes a colored rectangle arranged
    left-to-right.

    Args:
        ax: Matplotlib axes for the heatmap.
        string: IsalGraph instruction string.
        fontsize: Font size for instruction characters.
    """
    import matplotlib.patches as mpatches  # noqa: PLC0415

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


def _align_to_reference(
    G_ref: nx.Graph,
    G_nbr: nx.Graph,
) -> dict[int, int] | None:
    """Find node mapping from *G_nbr* to *G_ref* maximising edge overlap.

    Uses brute-force permutation search over ``|V|!`` mappings, which is
    feasible for the small graphs used in this figure (5 nodes → 120
    permutations).

    Args:
        G_ref: Reference graph (G₀).
        G_nbr: Neighbour graph to align.

    Returns:
        Mapping ``{nbr_node: ref_node}`` or ``None`` if the graphs have
        different numbers of nodes.
    """
    ref_nodes = sorted(G_ref.nodes())
    nbr_nodes = sorted(G_nbr.nodes())

    if len(ref_nodes) != len(nbr_nodes):
        return None

    ref_edges = {(min(u, v), max(u, v)) for u, v in G_ref.edges()}

    best_mapping: dict[int, int] | None = None
    best_overlap = -1

    for perm in itertools.permutations(ref_nodes):
        mapping = dict(zip(nbr_nodes, perm))
        overlap = sum(
            1
            for u, v in G_nbr.edges()
            if (min(mapping[u], mapping[v]), max(mapping[u], mapping[v])) in ref_edges
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_mapping = mapping

    return best_mapping


def _draw_graph_cell(
    ax: plt.Axes,
    G: nx.Graph,
    pos: dict,
    *,
    ref_edges: set[tuple[int, int]] | None = None,
    title: str = "",
    node_size: int = 150,
) -> None:
    """Draw a graph cell with optional missing-edge overlay.

    Args:
        ax: Matplotlib axes.
        G: Graph to draw.
        pos: Node positions (typically G₀'s spring layout).
        ref_edges: Edge set from G₀ (as ``(min, max)`` tuples).  Edges
            present here but absent in *G* are drawn as dashed red lines
            with ``alpha=0.4``.
        title: Optional title above the graph.
        node_size: Node marker size.
    """
    from benchmarks.eval_visualizations.graph_drawing import _DEFAULT_NODE_COLOR  # noqa: PLC0415

    # Normal edges
    nx.draw_networkx_edges(G, pos, edgelist=list(G.edges()), edge_color="0.5", width=0.8, ax=ax)

    # Missing edges from reference (dashed red, alpha=0.4)
    if ref_edges is not None:
        graph_edges = {(min(u, v), max(u, v)) for u, v in G.edges()}
        for u, v in ref_edges - graph_edges:
            if u in pos and v in pos:
                ax.plot(
                    [pos[u][0], pos[v][0]],
                    [pos[u][1], pos[v][1]],
                    "--",
                    color=PAUL_TOL_BRIGHT["red"],
                    linewidth=1.0,
                    alpha=0.4,
                    zorder=0,
                )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=_DEFAULT_NODE_COLOR,
        node_size=node_size,
        edgecolors="0.3",
        linewidths=0.5,
        ax=ax,
    )
    nx.draw_networkx_labels(G, pos, font_size=6, font_color="white", ax=ax)

    if title:
        ax.set_title(title, fontsize=7, pad=3)

    # Axis limits with uniform padding
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 0.1)
    pad = 0.25 * span
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.axis("off")


def _add_neighborhood_panels(
    fig: plt.Figure,
    group_axes: dict[str, list[plt.Axes]],
) -> None:
    """Add semi-transparent background panels and group titles.

    Draws three rounded rectangles (G₀, GED-1, Lev-1) with titles above,
    following the same visual language as ``_add_group_boxes`` in
    ``algorithm_figures.py``.

    Args:
        fig: The figure.
        group_axes: Mapping ``{"g0": [...], "ged1": [...], "lev1": [...]}``
            where each value is the list of axes belonging to that panel.
    """
    import matplotlib.patches as mpatches  # noqa: PLC0415
    from matplotlib.transforms import Bbox  # noqa: PLC0415

    renderer = fig.canvas.get_renderer()
    fig.draw(renderer)

    panel_conf = {
        "g0": {
            "color": "#444444",
            "label": "$G_0$",
        },
        "ged1": {
            "color": PAUL_TOL_BRIGHT["blue"],
            "label": "1-GED Neighbours",
        },
        "lev1": {
            "color": PAUL_TOL_BRIGHT["red"],
            "label": "1-Lev Neighbours",
        },
    }

    pad = 0.008

    # First pass: compute bounding boxes for all groups
    group_bboxes: dict[str, Bbox] = {}
    for key, axes in group_axes.items():
        bboxes = []
        for ax in axes:
            bb = ax.get_tightbbox(renderer)
            if bb is not None:
                bboxes.append(bb.transformed(fig.transFigure.inverted()))
        if bboxes:
            group_bboxes[key] = Bbox.union(bboxes)

    # Make G₀ panel span the full vertical extent of GED + Lev panels,
    # and clip its right edge so it doesn't overlap with the neighbor panels.
    if "g0" in group_bboxes and ("ged1" in group_bboxes or "lev1" in group_bboxes):
        other_bboxes = [group_bboxes[k] for k in ("ged1", "lev1") if k in group_bboxes]
        full_y0 = min(bb.y0 for bb in other_bboxes)
        full_y1 = max(bb.y1 for bb in other_bboxes)
        g0_bb = group_bboxes["g0"]
        # Clip right edge: stop before the leftmost neighbor panel starts
        right_clip = min(bb.x0 for bb in other_bboxes) - 4 * pad
        group_bboxes["g0"] = Bbox.from_extents(
            g0_bb.x0, full_y0, min(g0_bb.x1, right_clip), full_y1
        )

    # Second pass: draw panels and titles
    for key, union in group_bboxes.items():
        conf = panel_conf[key]

        rect = mpatches.FancyBboxPatch(
            (union.x0 - pad, union.y0 - pad),
            union.width + 2 * pad,
            union.height + 2 * pad,
            boxstyle="round,pad=0.005",
            facecolor=conf["color"],
            alpha=0.06,
            edgecolor=conf["color"],
            linewidth=1.0,
            transform=fig.transFigure,
            zorder=0,
        )
        fig.patches.append(rect)

        # Title above panel
        fig.text(
            union.x0 + union.width / 2,
            union.y1 + pad + 0.01,
            conf["label"],
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=conf["color"],
            transform=fig.transFigure,
        )


def generate_neighborhood_figure(output_dir: str) -> str:
    """Generate the GED-1 vs Levenshtein-1 neighbourhood grid figure.

    Layout: G₀ in column 0 spanning both rows, GED-1 neighbours in row (a),
    Lev-1 neighbours in row (b).  Each graph cell has a horizontal instruction
    heatmap below it.  Three background panels group G₀, GED-1, and Lev-1
    sections.  Missing edges (present in G₀ but absent in the neighbour) are
    drawn as dashed red lines with alpha=0.4.
    """
    apply_ieee_style()
    os.makedirs(output_dir, exist_ok=True)

    # --- Base graph: house graph (5 nodes, 6 edges) -----------------------
    G0 = nx.house_graph()
    w0 = _greedy_min(G0)
    pos0 = nx.spring_layout(G0, seed=42)
    ref_edges = {(min(u, v), max(u, v)) for u, v in G0.edges()}
    logger.info("Base graph: house, w0=%r (len=%d)", w0, len(w0))

    # --- GED-1 neighbours --------------------------------------------------
    ged1 = _ged1_neighbors(G0, w0)
    logger.info("GED-1 neighbours: %d unique", len(ged1))

    # --- Lev-1 neighbours (direct perturbation) ----------------------------
    lev1_all = _lev1_neighbors(G0, w0, max_results=30)
    # Filter to same node count so we can align positions to G₀'s layout
    n0 = G0.number_of_nodes()
    lev1_same = [d for d in lev1_all if d["graph"].number_of_nodes() == n0]
    logger.info("Lev-1 neighbours (same size): %d / %d", len(lev1_same), len(lev1_all))

    # Select up to n_show from each
    n_show = 4
    sel_ged1 = ged1[:n_show]

    # Pick one Lev-1 neighbour per distinct GED value for visual diversity
    _seen_ged: set[int] = set()
    sel_lev: list[dict[str, Any]] = []
    for d in lev1_same:
        if d["ged"] not in _seen_ged:
            _seen_ged.add(d["ged"])
            sel_lev.append(d)
        if len(sel_lev) >= n_show:
            break
    for d in lev1_same:
        if d not in sel_lev:
            sel_lev.append(d)
        if len(sel_lev) >= n_show:
            break

    # Align Lev-1 neighbours to G₀'s node ordering
    for d in sel_lev:
        mapping = _align_to_reference(G0, d["graph"])
        if mapping is not None:
            d["aligned_graph"] = nx.relabel_nodes(d["graph"], mapping)
        else:
            d["aligned_graph"] = d["graph"]

    # --- Figure layout: 5 rows × 5 columns --------------------------------
    # Rows 0,3: graph cells | Rows 1,4: heatmap cells | Row 2: spacer
    w_fig, _ = get_figure_size("double")
    fig = plt.figure(figsize=(w_fig, 5.0))

    gs = GridSpec(
        5,
        5,
        figure=fig,
        height_ratios=[4, 0.7, 0.6, 4, 0.7],
        width_ratios=[0.75, 1, 1, 1, 1],
        wspace=0.12,
        hspace=0.12,
        left=0.03,
        right=0.98,
        top=0.90,
        bottom=0.04,
    )

    # --- G₀ (column 0, spanning all rows) ----------------------------------
    ax_g0 = fig.add_subplot(gs[0:4, 0])
    _draw_graph_cell(ax_g0, G0, pos0, node_size=200)

    ax_g0_hm = fig.add_subplot(gs[4, 0])
    _render_horizontal_heatmap(ax_g0_hm, w0)

    # Track axes for background panels
    g0_axes: list[plt.Axes] = [ax_g0, ax_g0_hm]
    ged1_axes: list[plt.Axes] = []
    lev1_axes: list[plt.Axes] = []

    # --- GED-1 neighbours (rows 0-1, columns 1-4) -------------------------
    for i, d in enumerate(sel_ged1):
        ax = fig.add_subplot(gs[0, i + 1])
        _draw_graph_cell(
            ax,
            d["graph"],
            pos0,
            ref_edges=ref_edges,
            title=f"Lev = {d['lev']}",
        )
        ged1_axes.append(ax)

        ax_hm = fig.add_subplot(gs[1, i + 1])
        _render_horizontal_heatmap(ax_hm, d["string"])
        ged1_axes.append(ax_hm)

    # --- Lev-1 neighbours (rows 3-4, columns 1-4) -------------------------
    for i, d in enumerate(sel_lev):
        ax = fig.add_subplot(gs[3, i + 1])
        G_draw = d["aligned_graph"]
        _draw_graph_cell(
            ax,
            G_draw,
            pos0,
            ref_edges=ref_edges,
            title=f"GED = {d['ged']}",
        )
        lev1_axes.append(ax)

        ax_hm = fig.add_subplot(gs[4, i + 1])
        _render_horizontal_heatmap(ax_hm, d["string"])
        lev1_axes.append(ax_hm)

    # --- Background panels -------------------------------------------------
    fig.canvas.draw()
    _add_neighborhood_panels(fig, {"g0": g0_axes, "ged1": ged1_axes, "lev1": lev1_axes})

    path = os.path.join(output_dir, "fig_neighborhood_topology")
    save_figure(fig, path)
    plt.close(fig)

    # --- Auto-generate caption ---
    n_ged1 = len(sel_ged1)
    n_lev1 = len(sel_lev)
    ged1_levs = [d["lev"] for d in sel_ged1]
    lev1_geds = [d["ged"] for d in sel_lev]

    caption = (
        f"Neighbourhood topology of the house graph $G_0$ ({G0.number_of_nodes()} nodes, "
        f"{G0.number_of_edges()} edges) under two distance metrics. "
        f"Centre column: base graph $G_0$ with its canonical IsalGraph encoding (colour-coded "
        f"by instruction type). "
        f"Top rows: {n_ged1} representative 1-GED neighbours (single edge edit), "
        f"with Levenshtein distances "
        f"$\\mathrm{{Lev}} \\in [{min(ged1_levs)},\\, {max(ged1_levs)}]$ "
        f"to the encoding of $G_0$. "
        f"Bottom rows: {n_lev1} representative 1-Levenshtein neighbours "
        f"(single character substitution, insertion, or deletion in the instruction string), "
        f"with GED values "
        f"$\\mathrm{{GED}} \\in [{min(lev1_geds)},\\, {max(lev1_geds)}]$. "
        f"Dashed red edges indicate structural differences from $G_0$. "
        f"Horizontal heatmaps below each graph render the IsalGraph instruction string "
        f"with per-character colouring (alphabet $\\Sigma = \\{{N,n,P,p,V,v,C,c,W\\}}$). "
        f"The asymmetry between 1-GED and 1-Levenshtein neighbourhoods illustrates that "
        f"graph-space proximity does not imply string-space proximity, and vice versa."
    )

    caption_path = os.path.join(output_dir, "fig_neighborhood_topology_caption.txt")
    with open(caption_path, "w", encoding="utf-8") as f:
        f.write(caption + "\n")
    logger.info("Caption saved: %s", caption_path)

    logger.info("Neighbourhood figure saved: %s", path)
    return path


# ============================================================================
# Figure 2 — Distance field (GED × Levenshtein landscape)
# ============================================================================


def _draw_thumbnail(
    fig: plt.Figure,
    ax_main: plt.Axes,
    G: nx.Graph,
    w: str,
    x_data: float,
    y_data: float,
    *,
    border_color: str = "0.5",
    border_width: float = 1.0,
    size_inches: float = 0.75,
    node_size: int = 60,
    string_fontsize: int = 4,
) -> plt.Axes:
    """Draw a graph thumbnail as an inset axes at a (GED, Lev) position."""
    from benchmarks.eval_visualizations.graph_drawing import _DEFAULT_NODE_COLOR

    # Convert data coordinates → figure-fraction coordinates
    display_pt = ax_main.transData.transform((x_data, y_data))
    fig_pt = fig.transFigure.inverted().transform(display_pt)

    fw, fh = fig.get_size_inches()
    w_frac = size_inches / fw
    h_frac = (size_inches * 1.25) / fh  # extra height for string

    ax = fig.add_axes(
        [fig_pt[0] - w_frac / 2, fig_pt[1] - h_frac / 2, w_frac, h_frac],
        zorder=10,
    )

    pos = nx.spring_layout(G, seed=42)

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color="0.5", width=0.6, ax=ax)
    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=_DEFAULT_NODE_COLOR,
        node_size=node_size,
        edgecolors="0.3",
        linewidths=0.4,
        ax=ax,
    )
    nx.draw_networkx_labels(G, pos, font_size=4, font_color="white", ax=ax)

    # Axis limits
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 0.1)
    pad = 0.25 * span
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - 0.35 * span - pad, max(ys) + pad)

    # Coloured string below graph
    x_center = float(np.mean(xs))
    render_colored_string(
        ax,
        w,
        x=x_center,
        y=min(ys) - 0.22 * span,
        fontsize=string_fontsize,
        mono=True,
    )

    ax.set_aspect("equal")
    ax.axis("off")

    # Add coloured border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(border_color)
        spine.set_linewidth(border_width)

    return ax


def generate_distance_field_figure(output_dir: str) -> str:
    """Generate the graph-space distance field figure.

    Places G₀ at the origin of a 2D coordinate system where
    x = GED distance and y = Levenshtein distance.  Graph thumbnails
    are drawn at their (GED, Lev) coordinates, illustrating how the
    two metrics define correlated but distinct "directions" in the
    space of graphs.
    """
    apply_ieee_style()
    os.makedirs(output_dir, exist_ok=True)

    G0 = nx.house_graph()
    w0 = _greedy_min(G0)
    logger.info("Distance field: G0=house, w0=%r", w0)

    # --- Collect GED-1 neighbours -----------------------------------------
    ged1 = _ged1_neighbors(G0, w0)
    # Add GED field
    for d in ged1:
        d["ged"] = 1
        d["source"] = "ged1"

    # --- Collect Lev-1 neighbours (direct perturbation) --------------------
    lev1 = _lev1_neighbors(G0, w0, max_results=30)
    for d in lev1:
        d["source"] = "lev1"

    # --- Merge and tag overlaps -------------------------------------------
    all_nb: list[dict[str, Any]] = []
    for d in ged1:
        tag = "both" if d["lev"] <= 2 else "ged1"
        all_nb.append(
            {
                "graph": d["graph"],
                "string": d["string"],
                "ged": 1,
                "lev": d["lev"],
                "source": tag,
            }
        )
    for d in lev1:
        tag = "both" if d["ged"] == 1 else "lev1"
        all_nb.append(
            {
                "graph": d["graph"],
                "string": d["string"],
                "ged": d["ged"],
                "lev": d["lev"],
                "source": tag,
            }
        )

    # --- Select diverse subset for display --------------------------------
    # Prefer one graph per unique (ged, lev) cell; limit to ~10 total
    cells: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for d in all_nb:
        key = (d["ged"], d["lev"])
        cells.setdefault(key, []).append(d)

    selected: list[dict[str, Any]] = []
    for key in sorted(cells):
        # Prefer "both" > "ged1" > "lev1" for diversity
        bucket = sorted(cells[key], key=lambda d: {"both": 0, "ged1": 1, "lev1": 2}[d["source"]])
        selected.extend(bucket[:2])  # at most 2 per cell

    # Cap total
    if len(selected) > 12:
        # Keep at least one from each source category
        by_src: dict[str, list[dict[str, Any]]] = {}
        for d in selected:
            by_src.setdefault(d["source"], []).append(d)
        kept: list[dict[str, Any]] = []
        for src in ["both", "ged1", "lev1"]:
            kept.extend(by_src.get(src, [])[:4])
        selected = kept[:12]

    logger.info("Distance field: %d graphs selected", len(selected))

    # --- Compute layout positions -----------------------------------------
    max_ged = max((d["ged"] for d in selected), default=1)
    max_lev = max((d["lev"] for d in selected), default=1)

    # Handle overlapping cells: jitter horizontally
    cell_counts: dict[tuple[int, int], int] = {}
    for d in selected:
        key = (d["ged"], d["lev"])
        idx = cell_counts.get(key, 0)
        cell_counts[key] = idx + 1
        d["_jitter"] = (idx - 0.15) * 0.35  # small horizontal offset

    # --- Create figure ----------------------------------------------------
    w_fig, _ = get_figure_size("double")
    fig_h = w_fig * 0.72
    fig = plt.figure(figsize=(w_fig, fig_h))

    # Main coordinate axes (background)
    main_ax = fig.add_axes([0.10, 0.10, 0.85, 0.85])
    main_ax.set_xlim(-0.7, max_ged + 1.0)
    main_ax.set_ylim(-0.7, max_lev + 1.0)
    main_ax.set_xlabel(r"Graph Edit Distance to $G_0$", fontsize=9)
    main_ax.set_ylabel(r"Levenshtein distance to $w_0$", fontsize=9)
    main_ax.set_xticks(range(max_ged + 1))
    main_ax.set_yticks(range(max_lev + 1))
    main_ax.tick_params(labelsize=7)
    main_ax.grid(True, which="major", linewidth=0.3, alpha=0.3, linestyle=":")

    # Shaded regions for GED-1 ball and Lev-close ball
    import matplotlib.patches as mpatches  # noqa: PLC0415

    ged_band = mpatches.FancyBboxPatch(
        (0.55, -0.5),
        0.9,
        max_lev + 1.2,
        boxstyle="round,pad=0.05",
        facecolor=PAUL_TOL_BRIGHT["blue"],
        alpha=0.06,
        edgecolor="none",
    )
    main_ax.add_patch(ged_band)
    lev_band = mpatches.FancyBboxPatch(
        (-0.5, 0.4),
        max_ged + 1.2,
        1.2,
        boxstyle="round,pad=0.05",
        facecolor=PAUL_TOL_BRIGHT["red"],
        alpha=0.06,
        edgecolor="none",
    )
    main_ax.add_patch(lev_band)

    # Band labels
    main_ax.text(
        1.0,
        max_lev + 0.6,
        "GED-1 ball",
        fontsize=7,
        ha="center",
        color=PAUL_TOL_BRIGHT["blue"],
        fontstyle="italic",
    )
    main_ax.text(
        max_ged + 0.4,
        0.8,
        "Lev-1\nball",
        fontsize=7,
        ha="center",
        color=PAUL_TOL_BRIGHT["red"],
        fontstyle="italic",
    )

    # Draw connecting lines from origin to each graph (in data coords)
    for d in selected:
        x_pt = d["ged"] + d.get("_jitter", 0)
        y_pt = d["lev"]
        main_ax.plot(
            [0, x_pt],
            [0, y_pt],
            "-",
            color="0.80",
            linewidth=0.5,
            zorder=1,
        )

    # Force a draw so transData is calibrated for thumbnail placement
    fig.canvas.draw()

    # --- Category colours for borders -------------------------------------
    _BORDER_COLORS = {
        "ged1": PAUL_TOL_BRIGHT["blue"],
        "lev1": PAUL_TOL_BRIGHT["red"],
        "both": PAUL_TOL_BRIGHT["green"],
    }

    # --- Draw G0 thumbnail at origin (larger) -----------------------------
    _draw_thumbnail(
        fig,
        main_ax,
        G0,
        w0,
        0.0,
        0.0,
        border_color=PAUL_TOL_BRIGHT["yellow"],
        border_width=2.0,
        size_inches=0.90,
        node_size=80,
        string_fontsize=5,
    )
    # Label G0
    main_ax.text(
        0.0,
        -0.55,
        "$G_0$",
        fontsize=8,
        ha="center",
        fontweight="bold",
    )

    # --- Draw neighbour thumbnails ----------------------------------------
    for d in selected:
        x_pt = d["ged"] + d.get("_jitter", 0)
        y_pt = d["lev"]
        bc = _BORDER_COLORS[d["source"]]
        _draw_thumbnail(
            fig,
            main_ax,
            d["graph"],
            d["string"],
            x_pt,
            y_pt,
            border_color=bc,
            border_width=1.5,
            size_inches=0.70,
            node_size=50,
            string_fontsize=4,
        )

    # --- Legend for border colours ----------------------------------------
    legend_handles = [
        mpatches.Patch(
            facecolor="none", edgecolor=PAUL_TOL_BRIGHT["blue"], linewidth=1.5, label="GED-1 only"
        ),
        mpatches.Patch(
            facecolor="none",
            edgecolor=PAUL_TOL_BRIGHT["red"],
            linewidth=1.5,
            label="Lev-1 only",
        ),
        mpatches.Patch(
            facecolor="none", edgecolor=PAUL_TOL_BRIGHT["green"], linewidth=1.5, label="Both"
        ),
        mpatches.Patch(
            facecolor="none",
            edgecolor=PAUL_TOL_BRIGHT["yellow"],
            linewidth=1.5,
            label="$G_0$ (base)",
        ),
    ]
    main_ax.legend(
        handles=legend_handles,
        fontsize=6,
        loc="upper left",
        framealpha=0.9,
        ncol=2,
    )

    path = os.path.join(output_dir, "fig_distance_field")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("Distance field figure saved: %s", path)
    return path


# ============================================================================
# Figure 3 — Empirical complexity
# ============================================================================


def _random_connected(n: int, p: float, rng: np.random.Generator) -> nx.Graph:
    """Generate a random connected G(n,p) graph."""
    for _ in range(500):
        seed = int(rng.integers(0, 2**31))
        G = nx.gnp_random_graph(n, p, seed=seed)
        if nx.is_connected(G):
            return G
    # Fallback: complete graph
    return nx.complete_graph(n)


def _time_it(func, *args, n_reps: int = 3) -> float:
    """Return median CPU time (seconds) of func(*args) over n_reps calls."""
    times = []
    for _ in range(n_reps):
        t0 = time.process_time()
        func(*args)
        t1 = time.process_time()
        times.append(t1 - t0)
    return float(np.median(times))


def generate_complexity_figure(
    output_dir: str,
    *,
    seed: int = 42,
    n_graphs: int = 8,
    n_range: tuple[int, int] = (4, 13),
    p: float = 0.35,
    exhaustive_timeout: float = 60.0,
) -> str:
    """Generate the empirical complexity figure.

    Args:
        output_dir: Output directory.
        seed: Random seed.
        n_graphs: Graphs per node-count level.
        n_range: (min_n, max_n_exclusive) for graph sizes.
        p: Edge probability for G(n,p).
        exhaustive_timeout: Skip exhaustive if a single probe exceeds this.
    """
    apply_ieee_style()
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    sizes = list(range(n_range[0], n_range[1]))
    data: dict[str, dict[int, list[float]]] = {
        "greedy_single": {},
        "greedy_min": {},
        "pruned_exhaustive": {},
        "exhaustive": {},
        "exact_ged": {},
    }
    for key in data:
        for n in sizes:
            data[key][n] = []

    for n in sizes:
        logger.info("Timing n=%d ...", n)
        graphs = [_random_connected(n, p, rng) for _ in range(n_graphs)]

        for G in graphs:
            sg = _nx_to_sg(G)

            # Greedy single-start (node 0)
            def _gs(sg_=sg):  # noqa: E731
                GraphToString(sg_).run(initial_node=0)

            data["greedy_single"][n].append(_time_it(_gs, n_reps=5))

            # Greedy-min (all starts)
            def _gm(G_=G):  # noqa: E731
                _greedy_min(G_)

            data["greedy_min"][n].append(_time_it(_gm, n_reps=3))

            # Pruned exhaustive (with timeout probe)
            from isalgraph.core.canonical_pruned import pruned_canonical_string  # noqa: PLC0415

            t_probe_pe = time.process_time()
            import contextlib as _ctx_pe  # noqa: PLC0415

            with _ctx_pe.suppress(Exception):
                pruned_canonical_string(sg)
            t_probe_pe = time.process_time() - t_probe_pe

            if t_probe_pe < exhaustive_timeout:
                data["pruned_exhaustive"][n].append(
                    _time_it(
                        lambda sg_=sg: pruned_canonical_string(sg_),
                        n_reps=max(1, min(3, int(5.0 / max(t_probe_pe, 1e-6)))),
                    )
                )

            # Exhaustive canonical (with timeout probe)
            t_probe = time.process_time()
            import contextlib  # noqa: PLC0415

            with contextlib.suppress(Exception):
                compute_canonical(sg)
            t_probe = time.process_time() - t_probe

            if t_probe < exhaustive_timeout:
                data["exhaustive"][n].append(
                    _time_it(
                        lambda sg_=sg: compute_canonical(sg_),
                        n_reps=max(1, min(3, int(5.0 / max(t_probe, 1e-6)))),
                    )
                )

        # Exact GED: time a random pair
        G1, G2 = graphs[0], graphs[min(1, len(graphs) - 1)]
        for _ in range(min(n_graphs, 5)):
            data["exact_ged"][n].append(_time_it(lambda g1=G1, g2=G2: _ged(g1, g2), n_reps=3))

    # --- Plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=get_figure_size("double", height_ratio=0.55))

    _COLORS = {
        "greedy_single": PAUL_TOL_BRIGHT["cyan"],
        "greedy_min": PAUL_TOL_BRIGHT["red"],
        "pruned_exhaustive": PAUL_TOL_BRIGHT["blue"],
        "exhaustive": PAUL_TOL_BRIGHT["purple"],
        "exact_ged": PAUL_TOL_BRIGHT["grey"],
    }
    _MARKERS = {
        "greedy_single": "^",
        "greedy_min": "D",
        "pruned_exhaustive": "d",
        "exhaustive": "s",
        "exact_ged": "o",
    }
    _LABELS = {
        "greedy_single": "Greedy (single start)",
        "greedy_min": r"Greedy-min ($n$ starts)",
        "pruned_exhaustive": "Pruned canonical",
        "exhaustive": "Exhaustive canonical",
        "exact_ged": "Exact GED (one pair)",
    }

    for key in ["exact_ged", "exhaustive", "pruned_exhaustive", "greedy_min", "greedy_single"]:
        xs, ys_med, ys_lo, ys_hi = [], [], [], []
        for n in sizes:
            vals = data[key][n]
            if not vals:
                continue
            med = np.median(vals)
            q25 = np.percentile(vals, 25)
            q75 = np.percentile(vals, 75)
            xs.append(n)
            ys_med.append(med)
            ys_lo.append(med - q25)
            ys_hi.append(q75 - med)
        if not xs:
            continue
        ax.errorbar(
            xs,
            ys_med,
            yerr=[ys_lo, ys_hi],
            marker=_MARKERS[key],
            color=_COLORS[key],
            linewidth=1.2,
            markersize=5,
            capsize=2,
            capthick=0.8,
            label=_LABELS[key],
            zorder=3,
        )

    # Fit power-law reference lines for greedy methods
    for key, ls in [("greedy_single", ":"), ("greedy_min", "--")]:
        xs_fit, ys_fit = [], []
        for n in sizes:
            vals = data[key][n]
            if vals:
                xs_fit.append(n)
                ys_fit.append(np.median(vals))
        if len(xs_fit) >= 3:
            log_x = np.log(xs_fit)
            log_y = np.log(ys_fit)
            alpha, log_c = np.polyfit(log_x, log_y, 1)
            x_ref = np.linspace(sizes[0], sizes[-1], 50)
            y_ref = np.exp(log_c) * x_ref**alpha
            ax.plot(
                x_ref,
                y_ref,
                ls,
                color=_COLORS[key],
                alpha=0.5,
                linewidth=0.8,
                label=f"$\\alpha = {alpha:.1f}$  (fit)",
            )

    ax.set_yscale("log")
    ax.set_xlabel("Number of nodes ($n$)", fontsize=9)
    ax.set_ylabel("CPU time (s)", fontsize=9)
    ax.set_xticks(sizes)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9, ncol=2)
    ax.tick_params(labelsize=7)
    ax.grid(True, which="major", linewidth=0.4, alpha=0.4, linestyle=":")

    path = os.path.join(output_dir, "fig_empirical_complexity")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("Complexity figure saved: %s", path)
    return path


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate topology and complexity publication figures.",
    )
    parser.add_argument(
        "--output-dir",
        default="paper_figures/topology_and_complexity",
        help="Output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for complexity figure.",
    )
    parser.add_argument(
        "--n-graphs",
        type=int,
        default=8,
        help="Graphs per node-count level.",
    )
    parser.add_argument(
        "--n-min",
        type=int,
        default=4,
        help="Minimum node count.",
    )
    parser.add_argument(
        "--n-max",
        type=int,
        default=13,
        help="Maximum node count (exclusive).",
    )
    parser.add_argument(
        "--edge-prob",
        type=float,
        default=0.35,
        help="Edge probability for G(n,p).",
    )
    parser.add_argument(
        "--exhaustive-timeout",
        type=float,
        default=60.0,
        help="Skip exhaustive canonical if probe exceeds this (seconds).",
    )
    parser.add_argument(
        "--only",
        choices=["neighborhood", "distance_field", "complexity"],
        default=None,
        help="Generate only one figure.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    apply_ieee_style()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.only is None or args.only == "neighborhood":
        generate_neighborhood_figure(args.output_dir)

    if args.only is None or args.only == "distance_field":
        generate_distance_field_figure(args.output_dir)

    if args.only is None or args.only == "complexity":
        generate_complexity_figure(
            args.output_dir,
            seed=args.seed,
            n_graphs=args.n_graphs,
            n_range=(args.n_min, args.n_max),
            p=args.edge_prob,
            exhaustive_timeout=args.exhaustive_timeout,
        )

    logger.info("All figures generated in %s", args.output_dir)


if __name__ == "__main__":
    main()
