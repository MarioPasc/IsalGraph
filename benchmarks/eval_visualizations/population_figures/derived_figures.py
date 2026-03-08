# ruff: noqa: N803, N806
"""Population figures and tables for derived hypotheses (H1.4-H1.5).

Generates:
  - H1.4: Split-violin within-class vs between-class Levenshtein
  - H1.5: Exhaustive vs greedy method comparison (grouped bar + scatter)
  - Companion LaTeX tables for each
"""

from __future__ import annotations

import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarks.eval_visualizations.result_loader import (
    ALL_DATASETS,
    DATASET_DISPLAY,
    AllResults,
)
from benchmarks.eval_visualizations.table_generator import (
    generate_dual_table,
)
from benchmarks.plotting_styles import (
    PAUL_TOL_MUTED,
    save_figure,
)

logger = logging.getLogger(__name__)

_IAM_LEVELS = ["iam_letter_low", "iam_letter_med", "iam_letter_high"]
_DISTORTION_LABELS = {
    "iam_letter_low": "LOW",
    "iam_letter_med": "MED",
    "iam_letter_high": "HIGH",
}

_WITHIN_COLOR = PAUL_TOL_MUTED[3]  # green
_BETWEEN_COLOR = PAUL_TOL_MUTED[0]  # rose
_EXHAUSTIVE_COLOR = PAUL_TOL_MUTED[1]  # indigo
_GREEDY_COLOR = PAUL_TOL_MUTED[2]  # sand


def _load_full_stats(stats_dir: str, dataset: str, method: str) -> dict | None:
    """Load full correlation stats JSON for a dataset-method pair."""
    path = os.path.join(stats_dir, f"{dataset}_{method}_correlation_stats.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_cross_analysis(stats_dir: str) -> dict | None:
    """Load cross-dataset analysis JSON."""
    path = os.path.join(stats_dir, "cross_dataset_analysis.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _get_within_between_distributions(
    results: AllResults,
    dataset: str,
    method: str = "exhaustive",
    *,
    max_pairs: int = 50_000,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute within-class and between-class Levenshtein distributions.

    Returns:
        (within_lev, between_lev) arrays, or None if labels unavailable.
    """
    if dataset not in results.datasets:
        return None

    arts = results.datasets[dataset]
    if arts.labels is None:
        return None

    lev_matrix = results.levenshtein_matrices.get((dataset, method))
    if lev_matrix is None:
        return None

    n = min(len(arts.labels), lev_matrix.shape[0])
    labels = arts.labels[:n]
    triu_i, triu_j = np.triu_indices(n, k=1)
    lev_vals = lev_matrix[triu_i, triu_j].astype(float)
    same_class = labels[triu_i] == labels[triu_j]
    valid = np.isfinite(lev_vals)

    within = lev_vals[valid & same_class]
    between = lev_vals[valid & ~same_class]

    # Subsample for violin plot performance
    rng = np.random.default_rng(42)
    if len(within) > max_pairs:
        within = rng.choice(within, max_pairs, replace=False)
    if len(between) > max_pairs:
        between = rng.choice(between, max_pairs, replace=False)

    return within, between


# =====================================================================
# H1.4 -- Within-class vs Between-class Discrimination
# =====================================================================


def _draw_split_violin(
    ax: plt.Axes,
    data_left: np.ndarray,
    data_right: np.ndarray,
    position: float,
    *,
    color_left: str = _WITHIN_COLOR,
    color_right: str = _BETWEEN_COLOR,
) -> None:
    """Draw a split violin at the given position."""
    # Left half (within)
    parts_l = ax.violinplot(
        [data_left],
        positions=[position],
        showmeans=False,
        showextrema=False,
    )
    for body in parts_l["bodies"]:
        verts = body.get_paths()[0].vertices
        verts[:, 0] = np.clip(verts[:, 0], -np.inf, position)
        body.set_facecolor(color_left)
        body.set_alpha(0.7)

    # Right half (between)
    parts_r = ax.violinplot(
        [data_right],
        positions=[position],
        showmeans=False,
        showextrema=False,
    )
    for body in parts_r["bodies"]:
        verts = body.get_paths()[0].vertices
        verts[:, 0] = np.clip(verts[:, 0], position, np.inf)
        body.set_facecolor(color_right)
        body.set_alpha(0.7)

    # Median lines
    for data, offset in [(data_left, -0.05), (data_right, 0.05)]:
        median = np.median(data)
        ax.plot(
            [position + offset - 0.03, position + offset + 0.03],
            [median, median],
            "-",
            color="0.2",
            lw=1.0,
        )


def generate_h1_4_population(
    results: AllResults,
    stats_dir: str,
    output_dir: str,
) -> str:
    """Generate H1.4 population figure: split-violin within vs between."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.0), sharey=True)

    for col_idx, ds in enumerate(_IAM_LEVELS):
        ax = axes[col_idx]

        dist = _get_within_between_distributions(results, ds)
        if dist is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(_DISTORTION_LABELS[ds], fontsize=8)
            continue

        within, between = dist
        _draw_split_violin(ax, within, between, position=1.0)

        # Annotate Cohen's d from stats
        stats = _load_full_stats(stats_dir, ds, "exhaustive")
        if stats and "class_analysis" in stats:
            ca = stats["class_analysis"]
            d = ca.get("cohens_d", 0)
            p_text = ca.get("effect_interpretation", "")
            ax.text(
                0.95,
                0.95,
                f"$d$={d:.2f}\n({p_text})",
                transform=ax.transAxes,
                fontsize=6,
                va="top",
                ha="right",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1},
            )

        ax.set_title(_DISTORTION_LABELS[ds], fontsize=8, fontweight="bold")
        ax.set_xticks([1.0])
        ax.set_xticklabels([""])
        if col_idx == 0:
            ax.set_ylabel("Levenshtein distance", fontsize=7)
        ax.tick_params(labelsize=6)

    # Shared legend
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=_WITHIN_COLOR, alpha=0.7, label="Within-class"),
        Patch(facecolor=_BETWEEN_COLOR, alpha=0.7, label="Between-class"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        fontsize=7,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    path = os.path.join(output_dir, "population_within_vs_between")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("H1.4 population figure saved: %s", path)
    return path


def generate_h1_4_table(stats_dir: str, output_dir: str) -> None:
    """Generate H1.4 class discrimination table."""
    rows: list[dict] = []
    for ds in _IAM_LEVELS:
        stats = _load_full_stats(stats_dir, ds, "exhaustive")
        if stats is None or "class_analysis" not in stats:
            rows.append(
                {
                    "Distortion": _DISTORTION_LABELS[ds],
                    "Within ($\\mu \\pm \\sigma$)": "---",
                    "Between ($\\mu \\pm \\sigma$)": "---",
                    "Cohen's $d$": "---",
                    "Effect": "---",
                }
            )
            continue

        ca = stats["class_analysis"]
        rows.append(
            {
                "Distortion": _DISTORTION_LABELS[ds],
                "Within ($\\mu \\pm \\sigma$)": (
                    f"{ca.get('mean_within_lev', 0):.2f} $\\pm$ {ca.get('std_within_lev', 0):.2f}"
                ),
                "Between ($\\mu \\pm \\sigma$)": (
                    f"{ca.get('mean_between_lev', 0):.2f} $\\pm$ {ca.get('std_between_lev', 0):.2f}"
                ),
                "Cohen's $d$": f"{ca.get('cohens_d', 0):.3f}",
                "Effect": ca.get("effect_interpretation", "---"),
            }
        )

    df = pd.DataFrame(rows)
    generate_dual_table(
        df,
        output_dir,
        "table_class_discrimination",
        caption=(
            "Within-class vs between-class Levenshtein distance"
            " for IAM Letter datasets (canonical encoding)."
        ),
        label="tab:class-discrimination",
        highlight_cols={"Cohen's $d$"},
    )
    logger.info("H1.4 table saved to %s", output_dir)


# =====================================================================
# H1.5 -- Exhaustive vs Greedy Comparison
# =====================================================================


def generate_h1_5_population(
    stats_dir: str,
    output_dir: str,
) -> str:
    """Generate H1.5 population figure: grouped bar + rho scatter."""
    fig, (ax_bar, ax_scatter) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    cross = _load_cross_analysis(stats_dir)
    if cross is None or "h5_method_comparison" not in cross:
        logger.warning("No method comparison data available")
        fig.text(0.5, 0.5, "No data", ha="center", va="center")
        path = os.path.join(output_dir, "population_method_comparison")
        save_figure(fig, path)
        plt.close(fig)
        return path

    h5 = cross["h5_method_comparison"]

    # Collect data
    datasets = []
    rho_exh, rho_gre = [], []
    ci_exh_lo, ci_exh_hi = [], []
    ci_gre_lo, ci_gre_hi = [], []

    for ds in ALL_DATASETS:
        if ds not in h5:
            continue
        entry = h5[ds]
        datasets.append(ds)
        rho_exh.append(entry["rho_exhaustive"])
        rho_gre.append(entry["rho_greedy"])

        ci_e = entry.get("ci_exhaustive", [entry["rho_exhaustive"]] * 2)
        ci_g = entry.get("ci_greedy", [entry["rho_greedy"]] * 2)
        ci_exh_lo.append(entry["rho_exhaustive"] - ci_e[0])
        ci_exh_hi.append(ci_e[1] - entry["rho_exhaustive"])
        ci_gre_lo.append(entry["rho_greedy"] - ci_g[0])
        ci_gre_hi.append(ci_g[1] - entry["rho_greedy"])

    display_names = [DATASET_DISPLAY[ds] for ds in datasets]
    y_pos = np.arange(len(datasets))
    bar_height = 0.35

    # Panel (a): Grouped bar
    ax_bar.barh(
        y_pos - bar_height / 2,
        rho_exh,
        height=bar_height,
        xerr=[ci_exh_lo, ci_exh_hi],
        color=_EXHAUSTIVE_COLOR,
        edgecolor="0.3",
        linewidth=0.5,
        capsize=2,
        error_kw={"linewidth": 0.8},
        label="Canonical",
    )
    ax_bar.barh(
        y_pos + bar_height / 2,
        rho_gre,
        height=bar_height,
        xerr=[ci_gre_lo, ci_gre_hi],
        color=_GREEDY_COLOR,
        edgecolor="0.3",
        linewidth=0.5,
        capsize=2,
        error_kw={"linewidth": 0.8},
        label="Greedy-min",
    )
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(display_names, fontsize=6)
    ax_bar.set_xlabel("Spearman $\\rho$", fontsize=7)
    ax_bar.set_title("(a) Method comparison", fontsize=8)
    ax_bar.legend(fontsize=6, loc="lower right")
    ax_bar.tick_params(labelsize=5)
    ax_bar.set_xlim(0, 1.05)
    ax_bar.invert_yaxis()

    # Panel (b): Scatter rho_exh vs rho_gre
    for idx, ds in enumerate(datasets):
        color = PAUL_TOL_MUTED[idx % len(PAUL_TOL_MUTED)]
        ax_scatter.scatter(
            rho_exh[idx],
            rho_gre[idx],
            color=color,
            s=40,
            zorder=5,
            label=DATASET_DISPLAY[ds],
            edgecolors="0.3",
            linewidth=0.5,
        )

    # Identity line
    lo = min(min(rho_exh), min(rho_gre)) - 0.05
    hi = max(max(rho_exh), max(rho_gre)) + 0.05
    ax_scatter.plot([lo, hi], [lo, hi], "--", color="0.6", lw=0.8)
    ax_scatter.set_xlabel("$\\rho$ canonical", fontsize=7)
    ax_scatter.set_ylabel("$\\rho$ greedy-min", fontsize=7)
    ax_scatter.set_title("(b) $\\rho$ comparison", fontsize=8)
    ax_scatter.legend(fontsize=5, loc="upper left")
    ax_scatter.tick_params(labelsize=5)
    ax_scatter.set_aspect("equal")

    fig.tight_layout()
    path = os.path.join(output_dir, "population_method_comparison")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("H1.5 population figure saved: %s", path)
    return path


def generate_h1_5_table(stats_dir: str, output_dir: str) -> None:
    """Generate H1.5 method comparison table."""
    cross = _load_cross_analysis(stats_dir)
    if cross is None or "h5_method_comparison" not in cross:
        logger.warning("No method comparison data for table")
        return

    h5 = cross["h5_method_comparison"]
    rows: list[dict] = []

    for ds in ALL_DATASETS:
        if ds not in h5:
            continue
        entry = h5[ds]
        ci_e = entry.get("ci_exhaustive", [0, 0])
        ci_g = entry.get("ci_greedy", [0, 0])

        rows.append(
            {
                "Dataset": DATASET_DISPLAY[ds],
                "$\\rho$ canonical": (
                    f"{entry['rho_exhaustive']:.3f} [{ci_e[0]:.3f}, {ci_e[1]:.3f}]"
                ),
                "$\\rho$ greedy": (f"{entry['rho_greedy']:.3f} [{ci_g[0]:.3f}, {ci_g[1]:.3f}]"),
                "$\\Delta\\rho$": f"{entry.get('delta_rho', 0):+.3f}",
            }
        )

    # Load agreement % from method comparison JSONs
    mc_dir_alt = stats_dir.replace(
        "results/eval_benchmarks/eval_correlation/stats",
        "data/eval/method_comparison",
    )
    for row in rows:
        ds_key = next(
            (k for k, v in DATASET_DISPLAY.items() if v == row["Dataset"]),
            None,
        )
        if ds_key is None:
            row["Agreement %"] = "---"
            continue
        mc_path = os.path.join(mc_dir_alt, f"{ds_key}_comparison.json")
        if not os.path.isfile(mc_path):
            row["Agreement %"] = "---"
            continue
        with open(mc_path, encoding="utf-8") as f:
            mc_data = json.load(f)
        agg = mc_data.get("aggregate", {})
        row["Agreement %"] = f"{agg.get('pct_identical_strings', 0):.1f}"

    df = pd.DataFrame(rows)
    generate_dual_table(
        df,
        output_dir,
        "table_method_comparison",
        caption=(
            "Exhaustive (canonical) vs greedy-min encoding:"
            " Spearman correlation with GED across datasets."
        ),
        label="tab:method-comparison",
        highlight_cols={"$\\rho$ canonical", "$\\rho$ greedy"},
    )
    logger.info("H1.5 table saved to %s", output_dir)


# =====================================================================
# Orchestration
# =====================================================================


def generate_all_derived_population(
    results: AllResults,
    stats_dir: str,
    output_root: str,
) -> None:
    """Generate all population figures and tables for H1.4-H1.5."""
    # H1.4
    h1_4_dir = os.path.join(output_root, "H1_4_class_discrimination")
    os.makedirs(h1_4_dir, exist_ok=True)
    generate_h1_4_population(results, stats_dir, h1_4_dir)
    generate_h1_4_table(stats_dir, h1_4_dir)

    # H1.5
    h1_5_dir = os.path.join(output_root, "H1_5_exhaustive_vs_greedy")
    os.makedirs(h1_5_dir, exist_ok=True)
    generate_h1_5_population(stats_dir, h1_5_dir)
    generate_h1_5_table(stats_dir, h1_5_dir)
