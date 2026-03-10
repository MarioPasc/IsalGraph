# ruff: noqa: N803, N806
"""Individual example figures for derived hypotheses (H1.4-H1.5).

Generates:
  - H1.4: Same-class vs different-class pair examples
  - H1.5: Method divergence examples (exhaustive != greedy)
"""

from __future__ import annotations

import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from benchmarks.eval_visualizations.example_selector import (
    SelectedExample,
    save_selection_log,
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


# =====================================================================
# H1.4 -- Same-class vs different-class examples
# =====================================================================


def _select_same_class_pair(
    arts: object,
    lev_matrix: np.ndarray,
) -> tuple[int, int] | None:
    """Select a same-class pair with small Levenshtein distance."""
    if arts.labels is None:
        return None

    n = min(arts.ged_matrix.shape[0], lev_matrix.shape[0])
    labels = arts.labels[:n]

    best_pair = None
    best_lev = float("inf")

    for i in range(min(n, 200)):
        for j in range(i + 1, min(n, 200)):
            if str(labels[i]) != str(labels[j]):
                continue
            lev_val = float(lev_matrix[i, j])
            ged_val = float(arts.ged_matrix[i, j])
            if (
                np.isfinite(lev_val)
                and np.isfinite(ged_val)
                and lev_val > 0
                and ged_val > 0
                and lev_val < best_lev
            ):
                best_lev = lev_val
                best_pair = (i, j)

    return best_pair


def _select_diff_class_pair(
    arts: object,
    lev_matrix: np.ndarray,
) -> tuple[int, int] | None:
    """Select a different-class pair with large Levenshtein distance."""
    if arts.labels is None:
        return None

    n = min(arts.ged_matrix.shape[0], lev_matrix.shape[0])
    labels = arts.labels[:n]

    best_pair = None
    best_lev = -1.0

    for i in range(min(n, 200)):
        for j in range(i + 1, min(n, 200)):
            if str(labels[i]) == str(labels[j]):
                continue
            lev_val = float(lev_matrix[i, j])
            ged_val = float(arts.ged_matrix[i, j])
            if np.isfinite(lev_val) and np.isfinite(ged_val) and lev_val > best_lev:
                best_lev = lev_val
                best_pair = (i, j)

    return best_pair


def generate_h1_4_individual(
    results: AllResults,
    source_root: str,
    output_dir: str,
    dataset: str = "iam_letter_low",
) -> str:
    """Generate H1.4 individual figure: same-class vs different-class."""
    if dataset not in results.datasets:
        logger.warning("Dataset %s not available", dataset)
        return ""

    arts = results.datasets[dataset]
    lev_matrix = results.levenshtein_matrices.get((dataset, "exhaustive"))
    if lev_matrix is None:
        logger.warning("No exhaustive Levenshtein matrix for %s", dataset)
        return ""

    graph_lookup = load_graph_lookup(source_root, dataset)
    if not graph_lookup:
        logger.warning("Could not load graphs for %s", dataset)
        return ""

    same_pair = _select_same_class_pair(arts, lev_matrix)
    diff_pair = _select_diff_class_pair(arts, lev_matrix)

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

    for row_group, (pair, label) in enumerate(
        [
            (same_pair, "Same class"),
            (diff_pair, "Different class"),
        ]
    ):
        if pair is None:
            continue

        i, j = pair
        gid_i, gid_j = arts.graph_ids[i], arts.graph_ids[j]

        # Class labels
        label_i = str(arts.labels[i]) if arts.labels is not None else "?"
        label_j = str(arts.labels[j]) if arts.labels is not None else "?"

        selections.append(
            SelectedExample(
                hypothesis="H1.4",
                role="same_class" if row_group == 0 else "diff_class",
                dataset=dataset,
                indices=(i, j),
                graph_ids=(gid_i, gid_j),
                criterion=f"class_{label_i}_vs_{label_j}",
                ged=float(arts.ged_matrix[i, j]),
                lev=float(lev_matrix[i, j]),
            )
        )

        graph_row = row_group * 2
        align_row = row_group * 2 + 1

        # Graph panels
        ax_gi = fig.add_subplot(gs[graph_row, 0])
        ax_gj = fig.add_subplot(gs[graph_row, 1])

        G_i = graph_lookup.get(gid_i)
        G_j = graph_lookup.get(gid_j)

        title_i = f"{gid_i} (class {label_i})"
        title_j = f"{gid_j} (class {label_j})"

        if G_i is not None:
            draw_graph(G_i, ax_gi, title=title_i, node_size=150)
        else:
            ax_gi.text(
                0.5, 0.5, gid_i, ha="center", va="center", transform=ax_gi.transAxes, fontsize=7
            )
            ax_gi.axis("off")

        if G_j is not None:
            draw_graph(G_j, ax_gj, title=title_j, node_size=150)
        else:
            ax_gj.text(
                0.5, 0.5, gid_j, ha="center", va="center", transform=ax_gj.transAxes, fontsize=7
            )
            ax_gj.axis("off")

        # Row label
        ax_gi.annotate(
            label,
            xy=(-0.3, 0.5),
            xycoords="axes fraction",
            fontsize=7,
            ha="right",
            va="center",
            rotation=90,
            fontweight="bold",
        )

        # Alignment
        ax_align = fig.add_subplot(gs[align_row, :])
        s_i = results.canonical_strings.get((dataset, "exhaustive"), {}).get(gid_i)
        s_j = results.canonical_strings.get((dataset, "exhaustive"), {}).get(gid_j)

        if s_i and s_j:
            alignment = levenshtein_alignment(s_i, s_j)
            draw_alignment(alignment, ax_align, s_label="$w_i$", t_label="$w_j$")

        ged_val = float(arts.ged_matrix[i, j])
        lev_val = float(lev_matrix[i, j])
        ax_align.set_title(f"GED={ged_val:.0f}, Lev={lev_val:.0f}", fontsize=7, pad=2)

    save_selection_log(selections, os.path.join(output_dir, "h1_4_selections.json"))

    fig.suptitle(
        f"Class discrimination -- {DATASET_DISPLAY[dataset]}",
        fontsize=9,
        y=0.98,
    )

    path = os.path.join(output_dir, "individual_class_pair_examples")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("H1.4 individual figure saved: %s", path)
    return path


# =====================================================================
# H1.5 -- Method divergence examples
# =====================================================================


def generate_h1_5_individual(
    results: AllResults,
    source_root: str,
    method_comparison_dir: str,
    output_dir: str,
) -> str:
    """Generate H1.5 individual figure: method divergence examples.

    Shows one graph where exhaustive != greedy and one where they agree.
    """
    # Find a dataset with method comparison data
    dataset = None
    mc_data = None
    for ds in ["linux", "aids", "iam_letter_low"]:
        mc_path = os.path.join(method_comparison_dir, f"{ds}_comparison.json")
        if os.path.isfile(mc_path):
            with open(mc_path, encoding="utf-8") as f:
                mc_data = json.load(f)
            dataset = ds
            break

    if mc_data is None or dataset is None:
        logger.warning("No method comparison data found")
        return ""

    if dataset not in results.datasets:
        logger.warning("Dataset %s not available in results", dataset)
        return ""

    graph_lookup = load_graph_lookup(source_root, dataset)
    per_graph = mc_data.get("per_graph", [])

    # Find one divergent and one agreeing example
    divergent = None
    agreeing = None
    for entry in per_graph:
        if not entry.get("strings_identical", True) and divergent is None:
            divergent = entry
        if entry.get("strings_identical", False) and agreeing is None:
            agreeing = entry
        if divergent and agreeing:
            break

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

    for row_group, (entry, label) in enumerate(
        [
            (divergent, "Different strings"),
            (agreeing, "Identical strings"),
        ]
    ):
        if entry is None:
            continue

        gid = entry["graph_id"]
        s_exh = entry["exhaustive_string"]
        s_gre = entry["greedy_string"]
        lev_between = entry.get("levenshtein_between_methods", 0)

        G = graph_lookup.get(gid)

        graph_row = row_group * 2
        align_row = row_group * 2 + 1

        # Draw graph once spanning left column
        ax_g = fig.add_subplot(gs[graph_row, 0])
        if G is not None:
            draw_graph(G, ax_g, title=gid, node_size=150)
        else:
            ax_g.text(0.5, 0.5, gid, ha="center", va="center", transform=ax_g.transAxes, fontsize=7)
            ax_g.axis("off")

        ax_g.annotate(
            label,
            xy=(-0.3, 0.5),
            xycoords="axes fraction",
            fontsize=7,
            ha="right",
            va="center",
            rotation=90,
            fontweight="bold",
        )

        # String comparison in right column
        ax_str = fig.add_subplot(gs[graph_row, 1])
        ax_str.text(
            0.05,
            0.7,
            f"Canonical: {s_exh}",
            transform=ax_str.transAxes,
            fontsize=6,
            fontfamily="monospace",
        )
        ax_str.text(
            0.05,
            0.3,
            f"Greedy:    {s_gre}",
            transform=ax_str.transAxes,
            fontsize=6,
            fontfamily="monospace",
        )
        ax_str.set_title(f"Lev between methods = {lev_between}", fontsize=7)
        ax_str.axis("off")

        # Alignment between the two strings
        ax_align = fig.add_subplot(gs[align_row, :])
        if s_exh and s_gre:
            alignment = levenshtein_alignment(s_exh, s_gre)
            draw_alignment(
                alignment,
                ax_align,
                s_label="Canonical",
                t_label="Greedy",
            )

        selections.append(
            SelectedExample(
                hypothesis="H1.5",
                role="divergent" if row_group == 0 else "agreeing",
                dataset=dataset,
                indices=(),
                graph_ids=(gid,),
                criterion=f"lev_between={lev_between}",
            )
        )

    save_selection_log(selections, os.path.join(output_dir, "h1_5_selections.json"))

    fig.suptitle(
        f"Method comparison -- {DATASET_DISPLAY[dataset]}",
        fontsize=9,
        y=0.98,
    )

    path = os.path.join(output_dir, "individual_method_divergence")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("H1.5 individual figure saved: %s", path)
    return path


# =====================================================================
# Orchestration
# =====================================================================


def generate_all_derived_individual(
    results: AllResults,
    source_root: str,
    method_comparison_dir: str,
    output_root: str,
) -> None:
    """Generate all individual example figures for H1.4-H1.5."""
    # H1.4
    h1_4_dir = os.path.join(output_root, "H1_4_class_discrimination")
    os.makedirs(h1_4_dir, exist_ok=True)
    generate_h1_4_individual(results, source_root, h1_4_dir)

    # H1.5
    h1_5_dir = os.path.join(output_root, "H1_5_exhaustive_vs_greedy")
    os.makedirs(h1_5_dir, exist_ok=True)
    generate_h1_5_individual(results, source_root, method_comparison_dir, h1_5_dir)
