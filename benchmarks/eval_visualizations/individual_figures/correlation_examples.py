# ruff: noqa: N803, N806
"""Individual example figures for core correlation hypotheses (H1.1-H1.3).

Generates:
  - H1.1: 3 example pairs (concordant similar, dissimilar, discordant)
  - H1.2: Same letter class at 3 distortion levels
  - H1.3: Sparse vs dense pair comparison
"""

from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from benchmarks.eval_visualizations.example_selector import (
    SelectedExample,
    save_selection_log,
    select_concordant_similar,
    select_discordant,
    select_dissimilar,
)
from benchmarks.eval_visualizations.graph_drawing import draw_graph
from benchmarks.eval_visualizations.graph_loader import load_graph_lookup
from benchmarks.eval_visualizations.result_loader import (
    DATASET_DISPLAY,
    AllResults,
)
from benchmarks.eval_visualizations.string_alignment import (
    draw_alignment,
    levenshtein_alignment,
)
from benchmarks.plotting_styles import save_figure

logger = logging.getLogger(__name__)

_DISTORTION_LABELS = {
    "iam_letter_low": "LOW",
    "iam_letter_med": "MED",
    "iam_letter_high": "HIGH",
}
_IAM_LEVELS = ["iam_letter_low", "iam_letter_med", "iam_letter_high"]


# =====================================================================
# Helpers
# =====================================================================


def _get_string(
    results: AllResults,
    dataset: str,
    method: str,
    graph_id: str,
) -> str | None:
    """Get canonical string for a graph by ID."""
    strings = results.canonical_strings.get((dataset, method), {})
    return strings.get(graph_id)


def _draw_pair_with_alignment(
    fig: plt.Figure,
    gs_graphs: GridSpec,
    gs_align: GridSpec,
    graph_lookup: dict,
    results: AllResults,
    dataset: str,
    i: int,
    j: int,
    *,
    method: str = "exhaustive",
    row_label: str | None = None,
) -> None:
    """Draw one example row: two graphs + alignment strip below."""
    arts = results.datasets[dataset]
    gid_i, gid_j = arts.graph_ids[i], arts.graph_ids[j]
    G_i = graph_lookup.get(gid_i)
    G_j = graph_lookup.get(gid_j)
    s_i = _get_string(results, dataset, method, gid_i)
    s_j = _get_string(results, dataset, method, gid_j)

    ged_val = float(arts.ged_matrix[i, j])
    lev_matrix = results.levenshtein_matrices.get((dataset, method))
    lev_val = float(lev_matrix[i, j]) if lev_matrix is not None else 0

    # Graph panels (split the graph row into 2 columns)
    gs_sub = gs_graphs.subgridspec(1, 2, wspace=0.2)
    ax_gi = fig.add_subplot(gs_sub[0, 0])
    ax_gj = fig.add_subplot(gs_sub[0, 1])

    if G_i is not None:
        draw_graph(G_i, ax_gi, title=gid_i, node_size=150)
    else:
        ax_gi.text(0.5, 0.5, gid_i, ha="center", va="center", transform=ax_gi.transAxes, fontsize=7)
        ax_gi.axis("off")

    if G_j is not None:
        draw_graph(G_j, ax_gj, title=gid_j, node_size=150)
    else:
        ax_gj.text(0.5, 0.5, gid_j, ha="center", va="center", transform=ax_gj.transAxes, fontsize=7)
        ax_gj.axis("off")

    # Alignment strip
    ax_align = fig.add_subplot(gs_align)
    if s_i and s_j:
        alignment = levenshtein_alignment(s_i, s_j)
        draw_alignment(alignment, ax_align, s_label="$w_i$", t_label="$w_j$")

    ax_align.set_title(f"GED={ged_val:.0f}, Lev={lev_val:.0f}", fontsize=7, pad=2)

    if row_label:
        ax_gi.annotate(
            row_label,
            xy=(-0.3, 0.5),
            xycoords="axes fraction",
            fontsize=7,
            ha="right",
            va="center",
            rotation=90,
            fontweight="bold",
        )


# =====================================================================
# H1.1 -- Individual concordant/discordant examples
# =====================================================================


def generate_h1_1_individual(
    results: AllResults,
    source_root: str,
    output_dir: str,
    dataset: str = "iam_letter_low",
) -> str:
    """Generate H1.1 individual figure: 3 example pairs with alignments."""
    if dataset not in results.datasets:
        logger.warning("Dataset %s not available", dataset)
        return ""

    arts = results.datasets[dataset]
    ged = arts.ged_matrix
    lev_matrix = results.levenshtein_matrices.get((dataset, "exhaustive"))
    if lev_matrix is None:
        logger.warning("No exhaustive Levenshtein matrix for %s", dataset)
        return ""

    graph_lookup = load_graph_lookup(source_root, dataset)
    if not graph_lookup:
        logger.warning("Could not load graphs for %s", dataset)
        return ""

    n = min(ged.shape[0], lev_matrix.shape[0])
    ged_sub = ged[:n, :n]
    lev_sub = lev_matrix[:n, :n]
    labels = arts.labels[:n] if arts.labels is not None else None

    # Select examples
    i1, j1 = select_concordant_similar(ged_sub, lev_sub, arts.node_counts[:n], labels)
    i2, j2 = select_dissimilar(ged_sub, lev_sub, arts.node_counts[:n], labels)
    i3, j3 = select_discordant(ged_sub, lev_sub)

    selections = [
        SelectedExample(
            hypothesis="H1.1",
            role="concordant_similar",
            dataset=dataset,
            indices=(i1, j1),
            graph_ids=(arts.graph_ids[i1], arts.graph_ids[j1]),
            criterion="min_lev_at_ged=1",
            ged=float(ged_sub[i1, j1]),
            lev=float(lev_sub[i1, j1]),
        ),
        SelectedExample(
            hypothesis="H1.1",
            role="concordant_dissimilar",
            dataset=dataset,
            indices=(i2, j2),
            graph_ids=(arts.graph_ids[i2], arts.graph_ids[j2]),
            criterion="high_ged_high_lev",
            ged=float(ged_sub[i2, j2]),
            lev=float(lev_sub[i2, j2]),
        ),
        SelectedExample(
            hypothesis="H1.1",
            role="discordant",
            dataset=dataset,
            indices=(i3, j3),
            graph_ids=(arts.graph_ids[i3], arts.graph_ids[j3]),
            criterion="max_rank_disagreement",
            ged=float(ged_sub[i3, j3]),
            lev=float(lev_sub[i3, j3]),
        ),
    ]
    save_selection_log(selections, os.path.join(output_dir, "h1_1_selections.json"))

    # Build figure: 3 groups of (graph row + alignment row)
    fig = plt.figure(figsize=(7.0, 7.5))
    gs = GridSpec(
        6,
        2,
        figure=fig,
        height_ratios=[2, 0.8, 2, 0.8, 2, 0.8],
        hspace=0.5,
        wspace=0.3,
    )

    row_labels = ["Similar\nconcordant", "Dissimilar\nconcordant", "Discordant"]
    pairs = [(i1, j1), (i2, j2), (i3, j3)]

    for group_idx, ((i, j), label) in enumerate(zip(pairs, row_labels, strict=True)):
        graph_row = group_idx * 2
        align_row = group_idx * 2 + 1
        _draw_pair_with_alignment(
            fig,
            gs[graph_row, :],
            gs[align_row, :],
            graph_lookup,
            results,
            dataset,
            i,
            j,
            row_label=label,
        )

    fig.suptitle(f"Example pairs -- {DATASET_DISPLAY[dataset]}", fontsize=9, y=0.98)

    path = os.path.join(output_dir, "individual_concordant_pairs")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("H1.1 individual figure saved: %s", path)
    return path


# =====================================================================
# H1.2 -- Same class at different distortion levels
# =====================================================================


def _find_same_class_pair(
    arts: object,
    lev_matrix: np.ndarray,
    target_label: str = "A",
    target_ged: float = 1.0,
) -> tuple[int, int] | None:
    """Find a same-class pair with GED close to target_ged.

    Uses vectorized search over all class pairs for robustness.
    """
    if arts.labels is None:
        return None

    n = min(arts.ged_matrix.shape[0], lev_matrix.shape[0])
    class_mask = np.array([str(lbl) == target_label for lbl in arts.labels[:n]])
    class_indices = np.where(class_mask)[0]

    if len(class_indices) < 2:
        return None

    # Vectorized: extract all same-class pairs
    ci = class_indices
    ged_sub = arts.ged_matrix[np.ix_(ci, ci)]
    triu_r, triu_c = np.triu_indices(len(ci), k=1)
    ged_vals = ged_sub[triu_r, triu_c].astype(float)

    valid = np.isfinite(ged_vals) & (ged_vals >= 0)
    if not valid.any():
        return None

    # Prefer GED > 0 pairs first; fall back to GED >= 0
    positive = valid & (ged_vals > 0)
    if positive.any():
        diffs = np.abs(ged_vals - target_ged)
        diffs[~positive] = np.inf
    else:
        diffs = np.abs(ged_vals - target_ged)
        diffs[~valid] = np.inf

    best_idx = int(np.argmin(diffs))
    return int(ci[triu_r[best_idx]]), int(ci[triu_c[best_idx]])


def _find_shared_class(results: AllResults) -> str:
    """Find a class label present in all 3 IAM Letter distortion levels.

    Returns the class with the most total graphs across all levels.
    Falls back to "M" if no common class is found.
    """
    class_sets = []
    for ds in _IAM_LEVELS:
        if ds not in results.datasets or results.datasets[ds].labels is None:
            continue
        class_sets.append(set(str(lbl) for lbl in results.datasets[ds].labels))

    if len(class_sets) < 2:
        return "M"

    shared = class_sets[0]
    for s in class_sets[1:]:
        shared &= s

    if not shared:
        return "M"

    # Pick the class with the most total graphs across all levels
    best_label = "M"
    best_count = 0
    for lbl in sorted(shared):
        total = 0
        for ds in _IAM_LEVELS:
            if ds in results.datasets and results.datasets[ds].labels is not None:
                total += sum(1 for x in results.datasets[ds].labels if str(x) == lbl)
        if total > best_count:
            best_count = total
            best_label = lbl
    return best_label


def generate_h1_2_individual(
    results: AllResults,
    source_root: str,
    output_dir: str,
) -> str:
    """Generate H1.2 individual figure: same class at 3 distortion levels."""
    target_class = _find_shared_class(results)

    fig = plt.figure(figsize=(7.0, 5.0))
    gs = GridSpec(
        2,
        3,
        figure=fig,
        height_ratios=[2, 0.8],
        hspace=0.4,
        wspace=0.3,
    )

    selections: list[SelectedExample] = []

    for col_idx, ds in enumerate(_IAM_LEVELS):
        if ds not in results.datasets:
            continue

        arts = results.datasets[ds]
        lev_matrix = results.levenshtein_matrices.get((ds, "exhaustive"))
        if lev_matrix is None:
            continue

        graph_lookup = load_graph_lookup(source_root, ds)
        pair = _find_same_class_pair(arts, lev_matrix, target_label=target_class)
        if pair is None:
            continue

        i, j = pair
        gid_i, gid_j = arts.graph_ids[i], arts.graph_ids[j]

        selections.append(
            SelectedExample(
                hypothesis="H1.2",
                role=f"distortion_{_DISTORTION_LABELS[ds]}",
                dataset=ds,
                indices=(i, j),
                graph_ids=(gid_i, gid_j),
                criterion=f"same_class_{target_class}_ged_near_1",
                ged=float(arts.ged_matrix[i, j]),
                lev=float(lev_matrix[i, j]) if lev_matrix is not None else None,
            )
        )

        # Draw pair in column
        # Top: two graphs side by side in a sub-gridspec
        gs_top = gs[0, col_idx].subgridspec(1, 2, wspace=0.2)
        ax_gi = fig.add_subplot(gs_top[0, 0])
        ax_gj = fig.add_subplot(gs_top[0, 1])

        G_i = graph_lookup.get(gid_i)
        G_j = graph_lookup.get(gid_j)

        if G_i is not None:
            draw_graph(G_i, ax_gi, node_size=120)
        else:
            ax_gi.text(
                0.5, 0.5, gid_i, ha="center", va="center", transform=ax_gi.transAxes, fontsize=6
            )
            ax_gi.axis("off")

        if G_j is not None:
            draw_graph(G_j, ax_gj, node_size=120)
        else:
            ax_gj.text(
                0.5, 0.5, gid_j, ha="center", va="center", transform=ax_gj.transAxes, fontsize=6
            )
            ax_gj.axis("off")

        ax_gi.set_title(_DISTORTION_LABELS[ds], fontsize=8, fontweight="bold")

        # Bottom: alignment
        ax_align = fig.add_subplot(gs[1, col_idx])
        s_i = _get_string(results, ds, "exhaustive", gid_i)
        s_j = _get_string(results, ds, "exhaustive", gid_j)
        if s_i and s_j:
            alignment = levenshtein_alignment(s_i, s_j)
            draw_alignment(alignment, ax_align, s_label="$w_i$", t_label="$w_j$", cell_width=0.3)

        ged_val = float(arts.ged_matrix[i, j])
        lev_val = float(lev_matrix[i, j])
        ax_align.set_title(f"GED={ged_val:.0f}, Lev={lev_val:.0f}", fontsize=6, pad=2)

    save_selection_log(selections, os.path.join(output_dir, "h1_2_selections.json"))

    fig.suptitle(f'Same class ("{target_class}") across distortion levels', fontsize=9, y=0.98)

    path = os.path.join(output_dir, "individual_distortion_examples")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("H1.2 individual figure saved: %s", path)
    return path


# =====================================================================
# H1.3 -- Sparse vs dense pair
# =====================================================================


def _select_by_density(
    arts: object,
    lev_matrix: np.ndarray,
    *,
    high_density: bool,
    min_nodes: int = 4,
) -> tuple[int, int] | None:
    """Select a pair of graphs at the density extreme.

    Args:
        min_nodes: Minimum node count per graph (avoids trivial 2-node examples).
    """
    n = min(arts.ged_matrix.shape[0], lev_matrix.shape[0])
    nc = arts.node_counts[:n].astype(float)
    ec = arts.edge_counts[:n].astype(float)

    # Density = 2E / (V*(V-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        density = 2 * ec / (nc * (nc - 1))
    density = np.where(np.isfinite(density), density, 0)

    # Filter by minimum node count
    size_ok = nc >= min_nodes
    density = np.where(size_ok, density, -1 if high_density else 2)

    # Sort by density
    order = np.argsort(density)
    if high_density:
        order = order[::-1]

    # Find a pair among top-density graphs with GED > 0
    candidates = order[: min(50, len(order))]
    for a_idx in range(len(candidates)):
        for b_idx in range(a_idx + 1, len(candidates)):
            i, j = int(candidates[a_idx]), int(candidates[b_idx])
            ged_val = float(arts.ged_matrix[i, j])
            lev_val = float(lev_matrix[i, j])
            if np.isfinite(ged_val) and ged_val > 0 and lev_val > 0:
                return i, j
    return None


def generate_h1_3_individual(
    results: AllResults,
    source_root: str,
    output_dir: str,
) -> str:
    """Generate H1.3 individual figure: sparse vs dense pair comparison."""
    fig = plt.figure(figsize=(7.0, 5.0))
    gs = GridSpec(
        4,
        2,
        figure=fig,
        height_ratios=[2, 0.8, 2, 0.8],
        hspace=0.5,
        wspace=0.3,
    )

    selections: list[SelectedExample] = []
    configs = [
        ("linux", False, "Sparse (LINUX)"),
        ("aids", True, "Dense (AIDS)"),
    ]

    for row_group, (ds, high_density, label) in enumerate(configs):
        if ds not in results.datasets:
            continue

        arts = results.datasets[ds]
        lev_matrix = results.levenshtein_matrices.get((ds, "exhaustive"))
        if lev_matrix is None:
            continue

        graph_lookup = load_graph_lookup(source_root, ds)
        pair = _select_by_density(arts, lev_matrix, high_density=high_density)
        if pair is None:
            continue

        i, j = pair
        gid_i, gid_j = arts.graph_ids[i], arts.graph_ids[j]

        selections.append(
            SelectedExample(
                hypothesis="H1.3",
                role="dense" if high_density else "sparse",
                dataset=ds,
                indices=(i, j),
                graph_ids=(gid_i, gid_j),
                criterion="density_extreme",
                ged=float(arts.ged_matrix[i, j]),
                lev=float(lev_matrix[i, j]),
            )
        )

        graph_row = row_group * 2
        align_row = row_group * 2 + 1

        _draw_pair_with_alignment(
            fig,
            gs[graph_row, :],
            gs[align_row, :],
            graph_lookup,
            results,
            ds,
            i,
            j,
            row_label=label,
        )

    save_selection_log(selections, os.path.join(output_dir, "h1_3_selections.json"))

    path = os.path.join(output_dir, "individual_sparse_vs_dense")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("H1.3 individual figure saved: %s", path)
    return path


# =====================================================================
# Orchestration
# =====================================================================


def generate_all_core_individual(
    results: AllResults,
    source_root: str,
    output_root: str,
) -> None:
    """Generate all individual example figures for H1.1-H1.3."""
    # H1.1
    h1_1_dir = os.path.join(output_root, "H1_1_global_correlation")
    os.makedirs(h1_1_dir, exist_ok=True)
    generate_h1_1_individual(results, source_root, h1_1_dir)

    # H1.2
    h1_2_dir = os.path.join(output_root, "H1_2_monotone_degradation")
    os.makedirs(h1_2_dir, exist_ok=True)
    generate_h1_2_individual(results, source_root, h1_2_dir)

    # H1.3
    h1_3_dir = os.path.join(output_root, "H1_3_density_stratification")
    os.makedirs(h1_3_dir, exist_ok=True)
    generate_h1_3_individual(results, source_root, h1_3_dir)
