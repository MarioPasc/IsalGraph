# ruff: noqa: N803, N806, E402
"""Illustrative hero figures for the paper (Sub-Agent 5.8).

Generates five figures that build visual intuition:
  - fig_example_pairs: Similar + dissimilar pairs with graphs and alignments
  - fig_string_alignment: Large-format color-coded Levenshtein alignment strips
  - fig_heatmap_comparison: GED vs Lev(exhaustive) vs Lev(greedy) N×N heatmaps
  - fig_nearest_neighbors: k-NN overlap visualization (GED vs Lev)
  - fig_class_clustering: 2D MDS colored by class with convex hulls

Also generates selected_examples.json and figure_manifest.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from benchmarks.eval_visualizations.graph_drawing import draw_graph
from benchmarks.eval_visualizations.graph_loader import load_graph_lookup
from benchmarks.eval_visualizations.string_alignment import (
    draw_alignment,
    levenshtein_alignment,
)
from benchmarks.plotting_styles import (
    PAUL_TOL_MUTED,
    apply_ieee_style,
    save_figure,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# 9 IAM Letter LOW classes
CLASS_ORDER = ["I", "L", "M", "N", "V", "W", "X", "Y", "Z"]
CLASS_COLORS = {c: PAUL_TOL_MUTED[i % len(PAUL_TOL_MUTED)] for i, c in enumerate(CLASS_ORDER)}

# Default paths
_DEFAULT_DATA_ROOT = "/media/mpascual/Sandisk2TB/research/isalgraph/results/eval_benchmarks"
_DEFAULT_SOURCE_ROOT = "/media/mpascual/Sandisk2TB/research/isalgraph/data/source"
_DEFAULT_STRINGS_ROOT = "/media/mpascual/Sandisk2TB/research/isalgraph/data/eval/canonical_strings"
_DEFAULT_EMBEDDING_ROOT = (
    "/media/mpascual/Sandisk2TB/research/isalgraph/results/eval_benchmarks/eval_embedding/raw"
)
_DEFAULT_OUTPUT_DIR = "/media/mpascual/Sandisk2TB/research/isalgraph/results/figures/illustrative"

SEED = 42


# =============================================================================
# Data loading helpers
# =============================================================================


def _build_distance_matrix(
    csv_path: str,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build symmetric distance matrix from pair CSV.

    Args:
        csv_path: Path to pair_data CSV with columns: graph_i, graph_j, ged, levenshtein.

    Returns:
        (ged_matrix, graph_ids, lev_matrix) all shape (N, N).
    """
    df = pd.read_csv(csv_path)
    all_ids = sorted(set(df["graph_i"]) | set(df["graph_j"]))
    id_to_idx = {gid: i for i, gid in enumerate(all_ids)}
    n = len(all_ids)

    ged_mat = np.zeros((n, n))
    lev_mat = np.zeros((n, n))

    for _, row in df.iterrows():
        i = id_to_idx[row["graph_i"]]
        j = id_to_idx[row["graph_j"]]
        ged_mat[i, j] = row["ged"]
        ged_mat[j, i] = row["ged"]
        lev_mat[i, j] = row["levenshtein"]
        lev_mat[j, i] = row["levenshtein"]

    return ged_mat, all_ids, lev_mat


def _load_canonical_strings(strings_root: str, method: str) -> dict[str, str]:
    """Load canonical/greedy strings for IAM Letter LOW.

    Args:
        strings_root: Directory with canonical string JSONs.
        method: "exhaustive" or "greedy".

    Returns:
        Dict mapping graph_id -> string.
    """
    path = os.path.join(strings_root, f"iam_letter_low_{method}.json")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {gid: info["string"] for gid, info in raw["strings"].items()}


def _get_class_label(graph_id: str) -> str:
    """Extract class label from IAM Letter graph ID (first character)."""
    return graph_id[0]


def _subsample_graphs(
    graph_ids: list[str],
    n_per_class: int = 7,
    seed: int = SEED,
) -> list[int]:
    """Subsample graph indices: n_per_class per class, deterministic.

    Args:
        graph_ids: Full list of graph IDs.
        n_per_class: Number of graphs per class.
        seed: Random seed.

    Returns:
        Sorted list of indices into graph_ids.
    """
    rng = np.random.RandomState(seed)
    selected: list[int] = []

    for cls in CLASS_ORDER:
        cls_indices = [i for i, gid in enumerate(graph_ids) if _get_class_label(gid) == cls]
        if len(cls_indices) <= n_per_class:
            selected.extend(cls_indices)
        else:
            chosen = rng.choice(cls_indices, size=n_per_class, replace=False)
            selected.extend(sorted(chosen.tolist()))

    return sorted(selected)


# =============================================================================
# Figure 1: Example Pairs
# =============================================================================


def generate_example_pairs(
    ged_mat: np.ndarray,
    lev_mat: np.ndarray,
    graph_ids: list[str],
    exh_strings: dict[str, str],
    graph_lookup: dict[str, object] | None,
    output_dir: str,
) -> str:
    """Generate similar + dissimilar pair figure with graphs and alignments.

    Args:
        ged_mat: GED distance matrix.
        lev_mat: Levenshtein distance matrix (exhaustive).
        graph_ids: Graph ID list (ordered as matrix).
        exh_strings: Exhaustive canonical strings by graph_id.
        graph_lookup: Optional dict graph_id -> nx.Graph for drawing.
        output_dir: Output directory.

    Returns:
        Path to saved figure.
    """
    n = len(graph_ids)

    # Precompute node counts from graph_lookup for filtering
    node_counts: dict[int, int] = {}
    if graph_lookup:
        for idx, gid in enumerate(graph_ids):
            if gid in graph_lookup:
                node_counts[idx] = graph_lookup[gid].number_of_nodes()
    # Also infer from string length as fallback
    for idx, gid in enumerate(graph_ids):
        if idx not in node_counts:
            s = exh_strings.get(gid, "")
            # Rough estimate: node count ~ 1 + number of V/v instructions
            node_counts[idx] = 1 + sum(1 for c in s if c in "Vv")

    min_nodes = 4  # Filter for visually interesting graphs

    # Find similar pair: small GED, small Lev, same class, ≥ min_nodes
    best_sim = (0, 1)
    best_sim_score = float("inf")
    for i in range(n):
        if node_counts.get(i, 0) < min_nodes:
            continue
        ci = _get_class_label(graph_ids[i])
        for j in range(i + 1, n):
            if node_counts.get(j, 0) < min_nodes:
                continue
            cj = _get_class_label(graph_ids[j])
            if ci != cj:
                continue
            g = ged_mat[i, j]
            if g <= 0 or not np.isfinite(g):
                continue
            score = g + 0.1 * lev_mat[i, j]
            if score < best_sim_score:
                best_sim_score = score
                best_sim = (i, j)

    # Find dissimilar pair: large GED, large Lev, different class, ≥ min_nodes
    best_dis = (0, 1)
    best_dis_score = -float("inf")
    for i in range(n):
        if node_counts.get(i, 0) < min_nodes:
            continue
        ci = _get_class_label(graph_ids[i])
        for j in range(i + 1, n):
            if node_counts.get(j, 0) < min_nodes:
                continue
            cj = _get_class_label(graph_ids[j])
            if ci == cj:
                continue
            g = ged_mat[i, j]
            if not np.isfinite(g) or g <= 0:
                continue
            score = g + lev_mat[i, j]
            if score > best_dis_score:
                best_dis_score = score
                best_dis = (i, j)

    pairs = [("Similar", best_sim), ("Dissimilar", best_dis)]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.16, 4.0),
        gridspec_kw={"width_ratios": [1.2, 1.2, 2.5]},
    )

    for row, (label, (i, j)) in enumerate(pairs):
        gi, gj = graph_ids[i], graph_ids[j]
        si = exh_strings.get(gi, "?")
        sj = exh_strings.get(gj, "?")

        # Column 0: Graph i
        ax_gi = axes[row, 0]
        if graph_lookup and gi in graph_lookup:
            draw_graph(
                graph_lookup[gi],
                ax_gi,
                title=f"{_get_class_label(gi)}: {gi}",
                node_size=120,
            )
        else:
            ax_gi.text(
                0.5, 0.5, gi, ha="center", va="center", fontsize=9, transform=ax_gi.transAxes
            )
            ax_gi.set_title(f"{_get_class_label(gi)}: {gi}", fontsize=7)
            ax_gi.axis("off")

        # Column 1: Graph j
        ax_gj = axes[row, 1]
        if graph_lookup and gj in graph_lookup:
            draw_graph(
                graph_lookup[gj],
                ax_gj,
                title=f"{_get_class_label(gj)}: {gj}",
                node_size=120,
            )
        else:
            ax_gj.text(
                0.5, 0.5, gj, ha="center", va="center", fontsize=9, transform=ax_gj.transAxes
            )
            ax_gj.set_title(f"{_get_class_label(gj)}: {gj}", fontsize=7)
            ax_gj.axis("off")

        # Column 2: Alignment
        ax_align = axes[row, 2]
        alignment = levenshtein_alignment(si, sj)
        # Use shorter labels to avoid overlap
        draw_alignment(alignment, ax_align, s_label="w\u2081", t_label="w\u2082", cell_width=0.35)
        ged_val = ged_mat[i, j]
        lev_val = lev_mat[i, j]
        ax_align.set_title(
            f"{label}: GED={ged_val:.0f}, Lev={lev_val:.0f}",
            fontsize=8,
            fontweight="bold",
        )

    plt.tight_layout(rect=[0, 0, 0.98, 1])
    path = os.path.join(output_dir, "fig_example_pairs")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("Example pairs saved: %s", path)
    return path


# =============================================================================
# Figure 2: String Alignment (large format)
# =============================================================================


def generate_string_alignment(
    ged_mat: np.ndarray,
    lev_mat: np.ndarray,
    graph_ids: list[str],
    exh_strings: dict[str, str],
    output_dir: str,
) -> str:
    """Generate large-format color-coded alignment strips.

    Two rows: similar pair on top, dissimilar pair on bottom.

    Args:
        ged_mat: GED distance matrix.
        lev_mat: Levenshtein distance matrix.
        graph_ids: Graph ID list.
        exh_strings: Exhaustive canonical strings.
        output_dir: Output directory.

    Returns:
        Path to saved figure.
    """
    n = len(graph_ids)

    # Select similar pair with medium-length strings (for visual impact)
    best_sim = (0, 1)
    best_sim_score = float("inf")
    for i in range(n):
        si = exh_strings.get(graph_ids[i], "")
        if len(si) < 5 or len(si) > 30:
            continue
        ci = _get_class_label(graph_ids[i])
        for j in range(i + 1, n):
            sj = exh_strings.get(graph_ids[j], "")
            if len(sj) < 5 or len(sj) > 30:
                continue
            cj = _get_class_label(graph_ids[j])
            if ci != cj:
                continue
            g = ged_mat[i, j]
            if g <= 0 or not np.isfinite(g):
                continue
            score = g + 0.1 * lev_mat[i, j]
            if score < best_sim_score:
                best_sim_score = score
                best_sim = (i, j)

    # Select dissimilar pair with long strings
    best_dis = (0, 1)
    best_dis_score = -float("inf")
    for i in range(n):
        si = exh_strings.get(graph_ids[i], "")
        if len(si) < 5:
            continue
        for j in range(i + 1, n):
            sj = exh_strings.get(graph_ids[j], "")
            if len(sj) < 5:
                continue
            g = ged_mat[i, j]
            if not np.isfinite(g) or g <= 0:
                continue
            score = g + lev_mat[i, j]
            if score > best_dis_score:
                best_dis_score = score
                best_dis = (i, j)

    fig, axes = plt.subplots(2, 1, figsize=(7.16, 2.5))

    for row_idx, (label, (i, j)) in enumerate(
        [("Similar pair", best_sim), ("Dissimilar pair", best_dis)]
    ):
        gi, gj = graph_ids[i], graph_ids[j]
        si = exh_strings.get(gi, "?")
        sj = exh_strings.get(gj, "?")
        alignment = levenshtein_alignment(si, sj)
        ged_val = ged_mat[i, j]
        lev_val = lev_mat[i, j]

        ax = axes[row_idx]
        draw_alignment(alignment, ax, s_label="w\u2081", t_label="w\u2082")
        ax.set_title(
            f"{label}: {gi} vs {gj}  (GED={ged_val:.0f}, Lev={lev_val:.0f})",
            fontsize=8,
            fontweight="bold",
        )

    plt.tight_layout()
    path = os.path.join(output_dir, "fig_string_alignment")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("String alignment saved: %s", path)
    return path


# =============================================================================
# Figure 3: Heatmap Comparison (GED vs Lev exh vs Lev greedy)
# =============================================================================


def generate_heatmap_comparison(
    ged_mat: np.ndarray,
    lev_exh_mat: np.ndarray,
    lev_greedy_mat: np.ndarray,
    graph_ids: list[str],
    output_dir: str,
) -> str:
    """Generate 1x3 heatmap comparison: GED, Lev(exhaustive), Lev(greedy).

    Uses ~60 graphs (7 per class x 9 classes) from IAM Letter LOW.

    Args:
        ged_mat: Full GED distance matrix.
        lev_exh_mat: Full Levenshtein (exhaustive) distance matrix.
        lev_greedy_mat: Full Levenshtein (greedy) distance matrix.
        graph_ids: Full graph ID list.
        output_dir: Output directory.

    Returns:
        Path to saved figure.
    """
    # Subsample
    sub_idx = _subsample_graphs(graph_ids, n_per_class=7)
    sub_ids = [graph_ids[i] for i in sub_idx]
    n_sub = len(sub_idx)

    # Sort by class, then within-class by GED centroid distance
    class_labels = [_get_class_label(gid) for gid in sub_ids]
    sub_ged = ged_mat[np.ix_(sub_idx, sub_idx)]
    sub_lev_exh = lev_exh_mat[np.ix_(sub_idx, sub_idx)]
    sub_lev_greedy = lev_greedy_mat[np.ix_(sub_idx, sub_idx)]

    # Sort: by class, then by mean GED within class
    sort_order: list[int] = []
    class_boundaries: list[int] = []
    for cls in CLASS_ORDER:
        cls_local = [k for k in range(n_sub) if class_labels[k] == cls]
        if not cls_local:
            continue
        class_boundaries.append(len(sort_order))
        # Sort within class by mean GED to other class members
        mean_ged = [np.mean(sub_ged[k, cls_local]) for k in cls_local]
        cls_sorted = [cls_local[i] for i in np.argsort(mean_ged)]
        sort_order.extend(cls_sorted)
    class_boundaries.append(len(sort_order))

    # Reorder matrices
    idx_arr = np.array(sort_order)
    ged_ordered = sub_ged[np.ix_(idx_arr, idx_arr)]
    lev_exh_ordered = sub_lev_exh[np.ix_(idx_arr, idx_arr)]
    lev_greedy_ordered = sub_lev_greedy[np.ix_(idx_arr, idx_arr)]

    # Normalize all to [0, 1] using shared min-max across all three
    all_vals = np.concatenate(
        [
            ged_ordered[np.triu_indices(n_sub, k=1)],
            lev_exh_ordered[np.triu_indices(n_sub, k=1)],
            lev_greedy_ordered[np.triu_indices(n_sub, k=1)],
        ]
    )
    vmin, vmax = np.min(all_vals), np.max(all_vals)
    if vmax == vmin:
        vmax = vmin + 1

    def normalize(mat: np.ndarray) -> np.ndarray:
        return (mat - vmin) / (vmax - vmin)

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.8), constrained_layout=True)
    titles = ["GED", "Lev (exhaustive)", "Lev (greedy)"]
    matrices = [normalize(ged_ordered), normalize(lev_exh_ordered), normalize(lev_greedy_ordered)]

    for ax, title, mat in zip(axes, titles, matrices, strict=True):
        im = ax.imshow(mat, cmap="viridis", aspect="equal", vmin=0, vmax=1)
        ax.set_title(title, fontsize=8, fontweight="bold")

        # Class boundary lines
        for b in class_boundaries[1:-1]:
            ax.axhline(b - 0.5, color="white", linewidth=0.5, alpha=0.8)
            ax.axvline(b - 0.5, color="white", linewidth=0.5, alpha=0.8)

        ax.set_xticks([])
        ax.set_yticks([])

    # Class labels on left axis of first panel
    for cb_k in range(len(class_boundaries) - 1):
        mid = (class_boundaries[cb_k] + class_boundaries[cb_k + 1]) / 2
        cls = CLASS_ORDER[cb_k] if cb_k < len(CLASS_ORDER) else "?"
        axes[0].text(-1.5, mid, cls, ha="center", va="center", fontsize=6, fontweight="bold")

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Normalized distance", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    path = os.path.join(output_dir, "fig_heatmap_comparison")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("Heatmap comparison saved: %s", path)
    return path


# =============================================================================
# Figure 4: Nearest Neighbors
# =============================================================================


def generate_nearest_neighbors(
    ged_mat: np.ndarray,
    lev_mat: np.ndarray,
    graph_ids: list[str],
    graph_lookup: dict[str, object] | None,
    output_dir: str,
    k: int = 5,
) -> str:
    """Generate k-NN overlap visualization.

    Two rows. Each row: query graph + 5-NN by GED and 5-NN by Lev.
    Shared neighbors highlighted with green border.

    Args:
        ged_mat: GED distance matrix.
        lev_mat: Levenshtein distance matrix.
        graph_ids: Graph ID list.
        graph_lookup: Optional graph drawing lookup.
        output_dir: Output directory.
        k: Number of neighbors.

    Returns:
        Path to saved figure.
    """
    n = len(graph_ids)

    # Select two query graphs: one with high overlap, one with low
    overlaps = np.zeros(n, dtype=int)
    max_ged_nn_dist = np.zeros(n)
    for i in range(n):
        ged_nn = np.argsort(ged_mat[i])[1 : k + 1]
        lev_nn = np.argsort(lev_mat[i])[1 : k + 1]
        overlaps[i] = len(set(ged_nn) & set(lev_nn))
        max_ged_nn_dist[i] = ged_mat[i, ged_nn[-1]]

    # High overlap: prefer highest overlap, break ties by max NN distance (variety)
    high_threshold = max(overlaps.max() - 1, 2)
    high_overlap_candidates = np.where(overlaps >= high_threshold)[0]
    if len(high_overlap_candidates) == 0:
        high_overlap_candidates = np.argsort(-overlaps)[:20]
    # Among high-overlap, pick one with varied distances
    high_scores = max_ged_nn_dist[high_overlap_candidates]
    q_high = int(high_overlap_candidates[np.argmax(high_scores)])

    # Low overlap: overlap = 0, prefer varied distances
    low_overlap_candidates = np.where(overlaps == 0)[0]
    if len(low_overlap_candidates) == 0:
        low_overlap_candidates = np.argsort(overlaps)[:20]
    low_scores = max_ged_nn_dist[low_overlap_candidates]
    q_low = int(low_overlap_candidates[np.argmax(low_scores)])

    queries = [q_high, q_low]

    fig, axes = plt.subplots(2, 1 + 2 * k, figsize=(7.16, 3.8))

    def _draw_nn_cell(
        ax: plt.Axes,
        nn_id: str,
        is_shared: bool,
        dist: float,
        header: str,
    ) -> None:
        """Draw a single NN cell with optional green border rectangle."""
        if graph_lookup and nn_id in graph_lookup:
            draw_graph(graph_lookup[nn_id], ax, node_size=60)
        else:
            ax.text(
                0.5,
                0.5,
                nn_id[:6],
                ha="center",
                va="center",
                fontsize=5,
                transform=ax.transAxes,
            )
            ax.axis("off")

        if is_shared:
            # Add a visible green rectangle around the axes
            rect = mpatches.FancyBboxPatch(
                (0.02, 0.02),
                0.96,
                0.96,
                boxstyle="round,pad=0.02",
                transform=ax.transAxes,
                facecolor="none",
                edgecolor="#228833",
                linewidth=2.0,
                zorder=10,
            )
            ax.add_patch(rect)

        ax.set_title(header, fontsize=5)

    for row, q_idx in enumerate(queries):
        qid = graph_ids[q_idx]
        ged_nn_idx = np.argsort(ged_mat[q_idx])[1 : k + 1]
        lev_nn_idx = np.argsort(lev_mat[q_idx])[1 : k + 1]
        shared = set(ged_nn_idx) & set(lev_nn_idx)
        overlap_count = len(shared)

        # Query graph (column 0)
        ax_q = axes[row, 0]
        if graph_lookup and qid in graph_lookup:
            draw_graph(graph_lookup[qid], ax_q, node_size=80)
        else:
            ax_q.text(0.5, 0.5, qid, ha="center", va="center", fontsize=7, transform=ax_q.transAxes)
            ax_q.axis("off")
        ax_q.set_title(f"Query\n{qid}", fontsize=5, fontweight="bold")

        # GED neighbors (columns 1..k)
        for col, nn_i in enumerate(ged_nn_idx):
            ax = axes[row, 1 + col]
            dist = ged_mat[q_idx, nn_i]
            header = f"GED NN: d={dist:.0f}" if col == 0 else f"d={dist:.0f}"
            _draw_nn_cell(ax, graph_ids[nn_i], nn_i in shared, dist, header)

        # Lev neighbors (columns k+1..2k)
        for col, nn_i in enumerate(lev_nn_idx):
            ax = axes[row, 1 + k + col]
            dist = lev_mat[q_idx, nn_i]
            header = f"Lev NN: d={dist:.0f}" if col == 0 else f"d={dist:.0f}"
            _draw_nn_cell(ax, graph_ids[nn_i], nn_i in shared, dist, header)

        # Overlap annotation on query panel
        ax_q.text(
            0.5,
            -0.15,
            f"Overlap = {overlap_count}/{k}",
            ha="center",
            va="top",
            fontsize=7,
            fontweight="bold",
            color="#228833",
            transform=ax_q.transAxes,
        )

    # Add row separator label
    axes[0, 0].text(
        -0.3,
        0.5,
        "High",
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
        rotation=90,
        transform=axes[0, 0].transAxes,
    )
    axes[1, 0].text(
        -0.3,
        0.5,
        "Low",
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
        rotation=90,
        transform=axes[1, 0].transAxes,
    )

    # Legend for shared neighbor border
    legend_handle = mpatches.Patch(
        facecolor="none", edgecolor="#228833", linewidth=2, label="Shared neighbor"
    )
    fig.legend(handles=[legend_handle], loc="lower center", fontsize=6, ncol=1, framealpha=0.8)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(output_dir, "fig_nearest_neighbors")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("Nearest neighbors saved: %s", path)
    return path


# =============================================================================
# Figure 5: Class Clustering (MDS)
# =============================================================================


def generate_class_clustering(
    embedding_root: str,
    graph_ids: list[str],
    output_dir: str,
) -> str:
    """Generate 1x2 MDS scatter: Lev(exhaustive) vs GED, colored by class.

    Uses pre-computed SMACOF 2D embeddings with convex hulls per class.

    Args:
        embedding_root: Directory with SMACOF 2D NPZ files.
        graph_ids: Full graph ID list (1180 entries, ordered).
        output_dir: Output directory.

    Returns:
        Path to saved figure.
    """
    lev_path = os.path.join(embedding_root, "iam_letter_low_lev_exhaustive_smacof_2d.npz")
    ged_path = os.path.join(embedding_root, "iam_letter_low_ged_smacof_2d.npz")

    lev_coords = np.load(lev_path)["coords"]  # (N, 2)
    ged_coords = np.load(ged_path)["coords"]  # (N, 2)

    labels = np.array([_get_class_label(gid) for gid in graph_ids])

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2))
    panel_data = [
        ("Lev (exhaustive) MDS", lev_coords),
        ("GED MDS", ged_coords),
    ]

    for ax, (title, coords) in zip(axes, panel_data, strict=True):
        for cls in CLASS_ORDER:
            mask = labels == cls
            cls_coords = coords[mask]
            color = CLASS_COLORS[cls]

            ax.scatter(
                cls_coords[:, 0],
                cls_coords[:, 1],
                c=color,
                s=12,
                alpha=0.6,
                label=cls,
                edgecolors="none",
                zorder=2,
            )

            # Convex hull
            if len(cls_coords) >= 3:
                try:
                    hull = ConvexHull(cls_coords)
                    hull_pts = cls_coords[hull.vertices]
                    hull_pts = np.vstack([hull_pts, hull_pts[0]])
                    ax.fill(hull_pts[:, 0], hull_pts[:, 1], color=color, alpha=0.08, zorder=1)
                    ax.plot(
                        hull_pts[:, 0],
                        hull_pts[:, 1],
                        color=color,
                        linewidth=0.5,
                        alpha=0.4,
                        zorder=1,
                    )
                except Exception:
                    pass

            # Class label at centroid
            centroid = cls_coords.mean(axis=0)
            ax.text(
                centroid[0],
                centroid[1],
                cls,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color=color,
                zorder=3,
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1},
            )

        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.set_xlabel("Dim 1", fontsize=7)
        ax.set_ylabel("Dim 2", fontsize=7)
        ax.tick_params(labelsize=6)

    # Shared legend
    handles, labels_list = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_list,
        loc="lower center",
        ncol=9,
        fontsize=6,
        framealpha=0.8,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    path = os.path.join(output_dir, "fig_class_clustering")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("Class clustering saved: %s", path)
    return path


# =============================================================================
# Manifest & selected examples
# =============================================================================


def generate_selected_examples(output_dir: str) -> str:
    """Generate selected_examples.json placeholder.

    Args:
        output_dir: Output directory.

    Returns:
        Path to saved JSON.
    """
    examples = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "note": "Selected examples from illustrative figures (5.8).",
        "examples": [],
    }
    path = os.path.join(output_dir, "selected_examples.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)
    logger.info("Selected examples saved: %s", path)
    return path


def generate_figure_manifest(figures_root: str, output_dir: str) -> str:
    """Scan figures_root for all PDF/PNG/TEX files and build manifest.

    Args:
        figures_root: Root directory to scan for figure files.
        output_dir: Where to save the manifest.

    Returns:
        Path to saved JSON.
    """
    files: list[dict[str, str]] = []
    for dirpath, _, filenames in os.walk(figures_root):
        for fn in sorted(filenames):
            if fn.endswith((".pdf", ".png", ".tex")):
                rel_path = os.path.relpath(os.path.join(dirpath, fn), figures_root)
                # Infer hypothesis from directory name
                parts = rel_path.split(os.sep)
                hypothesis = parts[0] if len(parts) > 1 else "illustrative"
                file_type = "figure" if fn.endswith((".pdf", ".png")) else "table"
                files.append(
                    {
                        "path": rel_path,
                        "hypothesis": hypothesis,
                        "type": file_type,
                    }
                )

    manifest = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "total_files": len(files),
        "figures": files,
    }

    path = os.path.join(output_dir, "figure_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Figure manifest saved: %s (%d files)", path, len(files))
    return path


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """CLI entry point for illustrative figure generation."""
    parser = argparse.ArgumentParser(
        description="Generate illustrative hero figures for the paper (5.8).",
    )
    parser.add_argument(
        "--data-root",
        default=_DEFAULT_DATA_ROOT,
        help="Root directory for eval benchmark data.",
    )
    parser.add_argument(
        "--source-root",
        default=_DEFAULT_SOURCE_ROOT,
        help="Root directory for source graph data.",
    )
    parser.add_argument(
        "--strings-root",
        default=_DEFAULT_STRINGS_ROOT,
        help="Directory with canonical string JSONs.",
    )
    parser.add_argument(
        "--embedding-root",
        default=_DEFAULT_EMBEDDING_ROOT,
        help="Directory with SMACOF embedding NPZ files.",
    )
    parser.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        help="Output directory for figures.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    apply_ieee_style()
    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    logger.info("Loading distance matrices from pair CSVs...")
    exh_csv = os.path.join(
        args.data_root, "eval_correlation", "raw", "iam_letter_low_exhaustive_pair_data.csv"
    )
    greedy_csv = os.path.join(
        args.data_root, "eval_correlation", "raw", "iam_letter_low_greedy_pair_data.csv"
    )

    ged_mat, graph_ids, lev_exh_mat = _build_distance_matrix(exh_csv)
    _, _, lev_greedy_mat = _build_distance_matrix(greedy_csv)

    logger.info("Loaded %d graphs, matrices shape %s", len(graph_ids), ged_mat.shape)

    # Load canonical strings
    logger.info("Loading canonical strings...")
    exh_strings = _load_canonical_strings(args.strings_root, "exhaustive")
    logger.info("Loaded %d exhaustive strings", len(exh_strings))

    # Try to load graph objects for drawing
    graph_lookup = None
    try:
        logger.info("Loading graph objects for drawing...")
        graph_lookup = load_graph_lookup(args.source_root, "iam_letter_low")
        logger.info("Loaded %d graph objects", len(graph_lookup))
    except Exception as e:
        logger.warning("Could not load graph objects (will use text placeholders): %s", e)

    # -------------------------------------------------------------------------
    # Generate figures
    # -------------------------------------------------------------------------
    logger.info("=== Generating illustrative figures ===")

    generate_example_pairs(
        ged_mat,
        lev_exh_mat,
        graph_ids,
        exh_strings,
        graph_lookup,
        args.output_dir,
    )

    generate_string_alignment(
        ged_mat,
        lev_exh_mat,
        graph_ids,
        exh_strings,
        args.output_dir,
    )

    generate_heatmap_comparison(
        ged_mat,
        lev_exh_mat,
        lev_greedy_mat,
        graph_ids,
        args.output_dir,
    )

    generate_nearest_neighbors(
        ged_mat,
        lev_exh_mat,
        graph_ids,
        graph_lookup,
        args.output_dir,
    )

    generate_class_clustering(
        args.embedding_root,
        graph_ids,
        args.output_dir,
    )

    # -------------------------------------------------------------------------
    # Metadata files
    # -------------------------------------------------------------------------
    generate_selected_examples(args.output_dir)

    figures_root = os.path.dirname(args.output_dir)
    generate_figure_manifest(figures_root, args.output_dir)

    logger.info("=== All illustrative figures generated in %s ===", args.output_dir)


if __name__ == "__main__":
    main()
