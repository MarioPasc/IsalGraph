# ruff: noqa: N803, N806
"""Population figures and tables for core correlation hypotheses (H1.1-H1.3).

Generates:
  - H1.1: 2x3 overview panel (hexbin + bar + P@k) + summary table
  - H1.2: Monotone degradation trend plot + table
  - H1.3: Density stratification plot + table
"""

from __future__ import annotations

import json
import logging
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy import stats as sp_stats

from benchmarks.eval_visualizations.result_loader import (
    ALL_DATASETS,
    DATASET_DISPLAY,
    AllResults,
)
from benchmarks.eval_visualizations.table_generator import (
    format_significance,
    generate_dual_table,
)
from benchmarks.plotting_styles import (
    PAUL_TOL_MUTED,
    get_figure_size,
    save_figure,
)

logger = logging.getLogger(__name__)

# Datasets for hexbin scatter panels in H1.1
_HEXBIN_DATASETS = ["iam_letter_low", "linux", "aids"]

# IAM distortion levels (ordered)
_IAM_LEVELS = ["iam_letter_low", "iam_letter_med", "iam_letter_high"]
_DISTORTION_LABELS = {
    "iam_letter_low": "LOW",
    "iam_letter_med": "MED",
    "iam_letter_high": "HIGH",
}

# Colors
_OLS_COLOR = "#EE6677"
_EXHAUSTIVE_COLOR = PAUL_TOL_MUTED[1]  # indigo
_GREEDY_COLOR = PAUL_TOL_MUTED[2]  # sand

# Per-dataset colors (consistent across all bar charts)
_DATASET_COLORS = {ds: PAUL_TOL_MUTED[i] for i, ds in enumerate(ALL_DATASETS)}


# =====================================================================
# Data helpers
# =====================================================================


def _load_full_stats(stats_dir: str, dataset: str, method: str) -> dict | None:
    """Load full correlation stats JSON for a dataset-method pair."""
    path = os.path.join(stats_dir, f"{dataset}_{method}_correlation_stats.json")
    if not os.path.isfile(path):
        logger.warning("Stats file not found: %s", path)
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


def _get_distance_vectors(
    results: AllResults,
    dataset: str,
    method: str = "exhaustive",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract paired GED and Levenshtein distance vectors."""
    if dataset not in results.datasets:
        return None
    ged = results.datasets[dataset].ged_matrix
    lev_matrix = results.levenshtein_matrices.get((dataset, method))
    if lev_matrix is None:
        return None

    n = min(ged.shape[0], lev_matrix.shape[0])
    triu_i, triu_j = np.triu_indices(n, k=1)
    ged_vec = ged[triu_i, triu_j].astype(float)
    lev_vec = lev_matrix[triu_i, triu_j].astype(float)

    valid = np.isfinite(ged_vec) & np.isfinite(lev_vec) & (ged_vec > 0) & (lev_vec > 0)
    if not valid.any():
        return None
    return ged_vec[valid], lev_vec[valid]


def _density_bin_midpoint(bin_str: str) -> float:
    """Extract midpoint from a density bin string like '[0.286, 0.500]'."""
    nums = re.findall(r"[\d.]+", bin_str)
    if len(nums) == 2:
        return (float(nums[0]) + float(nums[1])) / 2
    return 0.0


# =====================================================================
# H1.1 -- Global Positive Correlation
# =====================================================================


def generate_h1_1_population(
    results: AllResults,
    stats_dir: str,
    output_dir: str,
) -> str:
    """Generate H1.1 population figure: 2x3 overview panel.

    Top row: hexbin scatter for 3 representative datasets.
    Bottom row: bar charts for rho, r, and P@k across all 5 datasets.
    """
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.5))
    panel_labels = "abcdef"

    # --- Top row: hexbin panels ---
    for col, dataset in enumerate(_HEXBIN_DATASETS):
        ax = axes[0, col]
        vecs = _get_distance_vectors(results, dataset, "exhaustive")
        if vecs is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.set_title(f"({panel_labels[col]}) {DATASET_DISPLAY[dataset]}", fontsize=8)
            continue

        ged_vec, lev_vec = vecs

        ax.hexbin(
            ged_vec,
            lev_vec,
            gridsize=30,
            cmap="viridis",
            norm=LogNorm(),
            mincnt=1,
            linewidths=0.1,
            edgecolors="0.7",
        )

        # Identity line
        lo = min(ged_vec.min(), lev_vec.min())
        hi = max(ged_vec.max(), lev_vec.max())
        ax.plot([lo, hi], [lo, hi], "--", color="0.6", lw=0.8, zorder=7)

        # OLS regression line
        slope, intercept, *_ = sp_stats.linregress(ged_vec, lev_vec)
        x_fit = np.array([lo, hi])
        ax.plot(x_fit, slope * x_fit + intercept, "-", color=_OLS_COLOR, lw=1.2, zorder=7)

        # Annotation
        stats = _load_full_stats(stats_dir, dataset, "exhaustive")
        if stats:
            rho = stats["spearman"]["statistic"]
            r = stats["pearson"]["statistic"]
            n_pairs = stats["n_valid_pairs"]
            sig = format_significance(stats["spearman"].get("p_value", 0))
            ax.text(
                0.05,
                0.95,
                f"$\\rho$={rho:.2f} {sig}\n$r$={r:.2f}\n$n$={n_pairs:,}",
                transform=ax.transAxes,
                fontsize=6,
                va="top",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1},
                zorder=10,
            )

        ax.set_title(f"({panel_labels[col]}) {DATASET_DISPLAY[dataset]}", fontsize=8)
        ax.set_xlabel("GED", fontsize=7)
        if col == 0:
            ax.set_ylabel("Levenshtein", fontsize=7)
        ax.tick_params(labelsize=5)

    # --- Bottom row: bar charts ---
    rho_vals, rho_errs_lo, rho_errs_hi = [], [], []
    r_vals, r_errs_lo, r_errs_hi = [], [], []
    pak_data: dict[int, list[float]] = {5: [], 10: [], 20: []}

    for ds in ALL_DATASETS:
        stats = _load_full_stats(stats_dir, ds, "exhaustive")
        if stats is None:
            rho_vals.append(0)
            rho_errs_lo.append(0)
            rho_errs_hi.append(0)
            r_vals.append(0)
            r_errs_lo.append(0)
            r_errs_hi.append(0)
            for k in pak_data:
                pak_data[k].append(0)
            continue

        sp = stats["spearman"]
        rho_vals.append(sp["statistic"])
        rho_errs_lo.append(sp["statistic"] - sp.get("ci_lower", sp["statistic"]))
        rho_errs_hi.append(sp.get("ci_upper", sp["statistic"]) - sp["statistic"])

        pe = stats["pearson"]
        r_vals.append(pe["statistic"])
        r_errs_lo.append(pe["statistic"] - pe.get("ci_lower", pe["statistic"]))
        r_errs_hi.append(pe.get("ci_upper", pe["statistic"]) - pe["statistic"])

        pak = stats.get("precision_at_k", {})
        for k in pak_data:
            entry = pak.get(str(k), {})
            pak_data[k].append(entry.get("precision", 0))

    display_names = [DATASET_DISPLAY[ds] for ds in ALL_DATASETS]
    y_pos = np.arange(len(ALL_DATASETS))
    colors = [_DATASET_COLORS[ds] for ds in ALL_DATASETS]

    # Panel (d): Spearman rho
    ax_d = axes[1, 0]
    ax_d.barh(
        y_pos,
        rho_vals,
        xerr=[rho_errs_lo, rho_errs_hi],
        color=colors,
        edgecolor="0.3",
        linewidth=0.5,
        capsize=2,
        error_kw={"linewidth": 0.8},
    )
    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(display_names, fontsize=6)
    ax_d.set_xlabel("Spearman $\\rho$", fontsize=7)
    ax_d.set_title(f"({panel_labels[3]}) Spearman $\\rho$", fontsize=8)
    ax_d.tick_params(labelsize=5)
    ax_d.set_xlim(0, 1.05)
    ax_d.invert_yaxis()

    # Panel (e): Pearson r
    ax_e = axes[1, 1]
    ax_e.barh(
        y_pos,
        r_vals,
        xerr=[r_errs_lo, r_errs_hi],
        color=colors,
        edgecolor="0.3",
        linewidth=0.5,
        capsize=2,
        error_kw={"linewidth": 0.8},
    )
    ax_e.set_yticks(y_pos)
    ax_e.set_yticklabels(display_names, fontsize=6)
    ax_e.set_xlabel("Pearson $r$", fontsize=7)
    ax_e.set_title(f"({panel_labels[4]}) Pearson $r$", fontsize=8)
    ax_e.tick_params(labelsize=5)
    ax_e.set_xlim(0, 1.05)
    ax_e.invert_yaxis()

    # Panel (f): Grouped bar P@k
    ax_f = axes[1, 2]
    bar_width = 0.25
    k_values = [5, 10, 20]
    k_colors = [PAUL_TOL_MUTED[3], PAUL_TOL_MUTED[4], PAUL_TOL_MUTED[6]]
    for idx, k in enumerate(k_values):
        offset = (idx - 1) * bar_width
        ax_f.barh(
            y_pos + offset,
            pak_data[k],
            height=bar_width,
            label=f"P@{k}",
            color=k_colors[idx],
            edgecolor="0.3",
            linewidth=0.5,
        )
    ax_f.set_yticks(y_pos)
    ax_f.set_yticklabels(display_names, fontsize=6)
    ax_f.set_xlabel("Precision", fontsize=7)
    ax_f.set_title(f"({panel_labels[5]}) Precision@$k$", fontsize=8)
    ax_f.legend(fontsize=5, loc="lower right")
    ax_f.tick_params(labelsize=5)
    ax_f.invert_yaxis()

    fig.tight_layout()
    path = os.path.join(output_dir, "population_correlation_overview")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("H1.1 population figure saved: %s", path)
    return path


def generate_h1_1_table(stats_dir: str, output_dir: str) -> None:
    """Generate H1.1 correlation summary table."""
    rows: list[dict] = []
    for ds in ALL_DATASETS:
        stats = _load_full_stats(stats_dir, ds, "exhaustive")
        if stats is None:
            rows.append(
                {
                    "Dataset": DATASET_DISPLAY[ds],
                    "N": "---",
                    "Pairs": "---",
                    "Spearman $\\rho$": "---",
                    "Pearson $r$": "---",
                    "Kendall $\\tau$": "---",
                    "CCC": "---",
                    "P@10": "---",
                    "Mantel $p$": "---",
                }
            )
            continue

        sp = stats["spearman"]
        pe = stats["pearson"]
        ke = stats["kendall"]
        mantel = stats.get("mantel", {})
        pak10 = stats.get("precision_at_k", {}).get("10", {})

        sig_rho = format_significance(sp.get("p_value", 1.0))
        sig_r = format_significance(pe.get("p_value", 1.0))

        rows.append(
            {
                "Dataset": DATASET_DISPLAY[ds],
                "N": f"{stats['n_graphs']:,}",
                "Pairs": f"{stats['n_valid_pairs']:,}",
                "Spearman $\\rho$": (
                    f"{sp['statistic']:.3f}"
                    f" [{sp.get('ci_lower', 0):.3f}, {sp.get('ci_upper', 0):.3f}]"
                    f" {sig_rho}"
                ),
                "Pearson $r$": (
                    f"{pe['statistic']:.3f}"
                    f" [{pe.get('ci_lower', 0):.3f}, {pe.get('ci_upper', 0):.3f}]"
                    f" {sig_r}"
                ),
                "Kendall $\\tau$": f"{ke['statistic']:.3f}",
                "CCC": f"{stats.get('lins_ccc_raw', 0):.3f}",
                "P@10": f"{pak10.get('precision', 0):.3f}",
                "Mantel $p$": format_significance(mantel.get("p_value", 1.0)),
            }
        )

    df = pd.DataFrame(rows)
    generate_dual_table(
        df,
        output_dir,
        "table_correlation_summary",
        caption=(
            "Correlation between GED and IsalGraph Levenshtein distance"
            " (canonical encoding) across all datasets."
        ),
        label="tab:correlation-summary",
        highlight_cols={"Spearman $\\rho$", "Pearson $r$", "CCC"},
    )
    logger.info("H1.1 table saved to %s", output_dir)


# =====================================================================
# H1.2 -- Monotone Degradation
# =====================================================================


def generate_h1_2_population(stats_dir: str, output_dir: str) -> str:
    """Generate H1.2 population figure: rho vs distortion with CI bands."""
    fig, ax = plt.subplots(figsize=get_figure_size("single"))

    x_positions = np.arange(len(_IAM_LEVELS))
    x_labels = [_DISTORTION_LABELS[ds] for ds in _IAM_LEVELS]

    for method, color, ls, label in [
        ("exhaustive", _EXHAUSTIVE_COLOR, "-", "Canonical"),
        ("greedy", _GREEDY_COLOR, "--", "Greedy-min"),
    ]:
        rhos, ci_lo, ci_hi = [], [], []
        for ds in _IAM_LEVELS:
            stats = _load_full_stats(stats_dir, ds, method)
            if stats is None:
                rhos.append(np.nan)
                ci_lo.append(np.nan)
                ci_hi.append(np.nan)
                continue
            sp = stats["spearman"]
            rhos.append(sp["statistic"])
            ci_lo.append(sp.get("ci_lower", sp["statistic"]))
            ci_hi.append(sp.get("ci_upper", sp["statistic"]))

        rhos_arr = np.array(rhos)
        ci_lo_arr = np.array(ci_lo)
        ci_hi_arr = np.array(ci_hi)

        ax.plot(
            x_positions,
            rhos_arr,
            ls,
            color=color,
            marker="o",
            markersize=5,
            label=label,
            lw=1.5,
        )
        ax.fill_between(x_positions, ci_lo_arr, ci_hi_arr, alpha=0.2, color=color)

    # Annotate monotonicity
    cross = _load_cross_analysis(stats_dir)
    if cross and "h2_monotone_degradation" in cross:
        h2 = cross["h2_monotone_degradation"]
        mono = h2.get("is_monotone_decreasing", False)
        mono_text = "Monotone decreasing" if mono else "Non-monotone"
        ax.text(
            0.95,
            0.05,
            mono_text,
            transform=ax.transAxes,
            fontsize=7,
            ha="right",
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.5", "pad": 2},
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_xlabel("Distortion level", fontsize=8)
    ax.set_ylabel("Spearman $\\rho$", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(labelsize=7)

    fig.tight_layout()
    path = os.path.join(output_dir, "population_distortion_trend")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("H1.2 population figure saved: %s", path)
    return path


def generate_h1_2_table(stats_dir: str, output_dir: str) -> None:
    """Generate H1.2 monotone degradation table."""
    rows: list[dict] = []
    for ds in _IAM_LEVELS:
        stats_exh = _load_full_stats(stats_dir, ds, "exhaustive")
        stats_gre = _load_full_stats(stats_dir, ds, "greedy")

        if stats_exh is None or stats_gre is None:
            rows.append(
                {
                    "Distortion": _DISTORTION_LABELS[ds],
                    "$\\rho$ canonical": "---",
                    "$\\rho$ greedy": "---",
                    "$\\Delta\\rho$": "---",
                }
            )
            continue

        rho_e = stats_exh["spearman"]["statistic"]
        rho_g = stats_gre["spearman"]["statistic"]
        ci_e = (
            stats_exh["spearman"].get("ci_lower", 0),
            stats_exh["spearman"].get("ci_upper", 0),
        )
        ci_g = (
            stats_gre["spearman"].get("ci_lower", 0),
            stats_gre["spearman"].get("ci_upper", 0),
        )

        rows.append(
            {
                "Distortion": _DISTORTION_LABELS[ds],
                "$\\rho$ canonical": f"{rho_e:.3f} [{ci_e[0]:.3f}, {ci_e[1]:.3f}]",
                "$\\rho$ greedy": f"{rho_g:.3f} [{ci_g[0]:.3f}, {ci_g[1]:.3f}]",
                "$\\Delta\\rho$": f"{rho_e - rho_g:+.3f}",
            }
        )

    df = pd.DataFrame(rows)
    generate_dual_table(
        df,
        output_dir,
        "table_monotone_degradation",
        caption=("Spearman correlation vs distortion level for IAM Letter datasets."),
        label="tab:monotone-degradation",
        highlight_cols={"$\\rho$ canonical", "$\\rho$ greedy"},
    )
    logger.info("H1.2 table saved to %s", output_dir)


# =====================================================================
# H1.3 -- Density Stratification
# =====================================================================


def generate_h1_3_population(stats_dir: str, output_dir: str) -> str:
    """Generate H1.3 density stratification figure.

    Each dataset is a line connecting rho values at its density bin midpoints.
    """
    fig, ax = plt.subplots(figsize=get_figure_size("single"))

    for ds_idx, ds in enumerate(ALL_DATASETS):
        stats = _load_full_stats(stats_dir, ds, "exhaustive")
        if stats is None or "density_stratified" not in stats:
            continue

        bins = stats["density_stratified"]
        midpoints = [_density_bin_midpoint(b["density_bin"]) for b in bins]
        rhos = [b["spearman_rho"] for b in bins]
        color = PAUL_TOL_MUTED[ds_idx % len(PAUL_TOL_MUTED)]
        ax.plot(
            midpoints,
            rhos,
            "o-",
            color=color,
            label=DATASET_DISPLAY[ds],
            markersize=4,
            lw=1.2,
        )

    ax.set_xlabel("Edge density", fontsize=8)
    ax.set_ylabel("Spearman $\\rho$", fontsize=8)
    ax.legend(fontsize=6, loc="best")
    ax.tick_params(labelsize=7)

    fig.tight_layout()
    path = os.path.join(output_dir, "population_density_vs_rho")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("H1.3 population figure saved: %s", path)
    return path


def generate_h1_3_table(stats_dir: str, output_dir: str) -> None:
    """Generate H1.3 density stratification table."""
    rows: list[dict] = []
    for ds in ALL_DATASETS:
        stats = _load_full_stats(stats_dir, ds, "exhaustive")
        if stats is None or "density_stratified" not in stats:
            continue

        for b in stats["density_stratified"]:
            rows.append(
                {
                    "Dataset": DATASET_DISPLAY[ds],
                    "Density bin": b["density_bin"],
                    "N pairs": f"{b['n_pairs']:,}",
                    "Spearman $\\rho$": f"{b['spearman_rho']:.3f}",
                }
            )

    df = pd.DataFrame(rows)
    generate_dual_table(
        df,
        output_dir,
        "table_density_stratification",
        caption="Spearman correlation stratified by edge density.",
        label="tab:density-stratification",
        highlight_cols={"Spearman $\\rho$"},
    )
    logger.info("H1.3 table saved to %s", output_dir)


# =====================================================================
# Orchestration
# =====================================================================


def generate_all_core_population(
    results: AllResults,
    stats_dir: str,
    output_root: str,
) -> None:
    """Generate all population figures and tables for H1.1-H1.3.

    Args:
        results: Loaded evaluation results.
        stats_dir: Path to correlation stats directory.
        output_root: Root output directory (subdirs created per hypothesis).
    """
    # H1.1
    h1_1_dir = os.path.join(output_root, "H1_1_global_correlation")
    os.makedirs(h1_1_dir, exist_ok=True)
    generate_h1_1_population(results, stats_dir, h1_1_dir)
    generate_h1_1_table(stats_dir, h1_1_dir)

    # H1.2
    h1_2_dir = os.path.join(output_root, "H1_2_monotone_degradation")
    os.makedirs(h1_2_dir, exist_ok=True)
    generate_h1_2_population(stats_dir, h1_2_dir)
    generate_h1_2_table(stats_dir, h1_2_dir)

    # H1.3
    h1_3_dir = os.path.join(output_root, "H1_3_density_stratification")
    os.makedirs(h1_3_dir, exist_ok=True)
    generate_h1_3_population(stats_dir, h1_3_dir)
    generate_h1_3_table(stats_dir, h1_3_dir)
