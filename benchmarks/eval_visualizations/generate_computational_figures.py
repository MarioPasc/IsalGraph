# ruff: noqa: N803, N806, E402
"""Generate computational advantage figures (C3.1, C3.2, C3.3).

Reads timing data from eval_computational outputs and produces:
  - C3.1: Speedup factor bar chart + individual timing trace + table
  - C3.2: Crossover curves (log-log) + zoomed individual + table
  - C3.3: Amortized pipeline stacked bars + pie chart + table

Usage:
    python -m benchmarks.eval_visualizations.generate_computational_figures \
        --comp-dir results/eval_benchmarks/eval_computational \
        --output-dir results/figures/computational
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from benchmarks.eval_visualizations.table_generator import generate_dual_table
from benchmarks.plotting_styles import (
    PAUL_TOL_BRIGHT,
    PAUL_TOL_MUTED,
    PLOT_SETTINGS,
    apply_ieee_style,
    get_figure_size,
    save_figure,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

ALL_DATASETS = ["iam_letter_low", "iam_letter_med", "iam_letter_high", "linux", "aids"]
DATASET_DISPLAY: dict[str, str] = {
    "iam_letter_low": "IAM LOW",
    "iam_letter_med": "IAM MED",
    "iam_letter_high": "IAM HIGH",
    "linux": "LINUX",
    "aids": "AIDS",
}

# Colors for methods
GED_COLOR = PAUL_TOL_BRIGHT["red"]
ENCODE_COLOR = PAUL_TOL_BRIGHT["blue"]
LEV_COLOR = PAUL_TOL_BRIGHT["green"]
GREEDY_COLOR = PAUL_TOL_BRIGHT["cyan"]
EXHAUSTIVE_COLOR = PAUL_TOL_BRIGHT["purple"]
TOTAL_ISAL_COLOR = PAUL_TOL_MUTED[1]  # indigo

DEFAULT_COMP_DIR = (
    "/media/mpascual/Sandisk2TB/research/isalgraph/results/eval_benchmarks/eval_computational"
)
DEFAULT_OUTPUT_DIR = "/media/mpascual/Sandisk2TB/research/isalgraph/results/figures/computational"


# =============================================================================
# Data loading
# =============================================================================


def _load_timing_stats(comp_dir: str) -> dict[str, dict]:
    """Load per-dataset timing stats JSONs."""
    stats: dict[str, dict] = {}
    stats_dir = os.path.join(comp_dir, "stats")
    for ds in ALL_DATASETS:
        path = os.path.join(stats_dir, f"{ds}_timing_stats.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                stats[ds] = json.load(f)
        else:
            logger.warning("Timing stats not found: %s", path)
    return stats


def _load_encoding_csv(comp_dir: str, dataset: str) -> pd.DataFrame | None:
    """Load encoding times CSV."""
    path = os.path.join(comp_dir, "raw", f"{dataset}_encoding_times.csv")
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def _load_ged_csv(comp_dir: str, dataset: str) -> pd.DataFrame | None:
    """Load GED times CSV."""
    path = os.path.join(comp_dir, "raw", f"{dataset}_ged_times.csv")
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def _load_lev_csv(comp_dir: str, dataset: str) -> pd.DataFrame | None:
    """Load Levenshtein times CSV."""
    path = os.path.join(comp_dir, "raw", f"{dataset}_levenshtein_times.csv")
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def _load_amortized_csv(comp_dir: str, dataset: str) -> pd.DataFrame | None:
    """Load amortized comparison CSV."""
    path = os.path.join(comp_dir, "raw", f"{dataset}_amortized_comparison.csv")
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


# =============================================================================
# C3.1 — Speedup Factor
# =============================================================================


def generate_c31_population(
    all_stats: dict[str, dict],
    output_dir: str,
) -> None:
    """C3.1 Population: Horizontal bar chart of speedup per dataset (log x-axis).

    Two bars per dataset: exhaustive and greedy pipeline speedup.
    Vertical line at speedup=1 (breakeven).
    """
    fig_dir = os.path.join(output_dir, "C3_1_speedup")
    os.makedirs(fig_dir, exist_ok=True)

    # Compute per-dataset speedup from median GED vs median IsalGraph total
    datasets_with_data = []
    speedup_exh = []
    speedup_greedy = []

    for ds in ALL_DATASETS:
        if ds not in all_stats:
            continue
        st = all_stats[ds]

        ged_med = st["ged_median_s"]
        enc_med = st["encoding_median_s"]
        lev_med = st["levenshtein_median_s"]

        # Exhaustive pipeline: 2*encoding + levenshtein (pair comparison)
        t_exh = 2 * enc_med + lev_med
        sp_exh = ged_med / t_exh if t_exh > 0 else float("nan")

        # Greedy is not directly stored; approximate from encoding CSV
        # For now, use the greedy encoding time from the encoding CSV
        sp_greedy = sp_exh  # placeholder, will refine below

        datasets_with_data.append(ds)
        speedup_exh.append(sp_exh)
        speedup_greedy.append(sp_greedy)

    # Refine greedy speedup from encoding CSVs
    for i, ds in enumerate(datasets_with_data):
        enc_df = _load_encoding_csv(DEFAULT_COMP_DIR, ds)
        if enc_df is not None and "greedy_time_median_s" in enc_df.columns:
            greedy_enc_med = enc_df["greedy_time_median_s"].median()
            lev_med = all_stats[ds]["levenshtein_median_s"]
            ged_med = all_stats[ds]["ged_median_s"]
            t_greedy = 2 * greedy_enc_med + lev_med
            speedup_greedy[i] = ged_med / t_greedy if t_greedy > 0 else float("nan")

    # Plot
    fig, ax = plt.subplots(figsize=get_figure_size("single", height_ratio=0.9))

    y_pos = np.arange(len(datasets_with_data))
    bar_h = 0.35

    bars_exh = ax.barh(
        y_pos - bar_h / 2,
        speedup_exh,
        bar_h,
        color=EXHAUSTIVE_COLOR,
        alpha=0.85,
        label="Exhaustive",
        edgecolor="white",
        linewidth=0.5,
    )
    bars_greedy = ax.barh(
        y_pos + bar_h / 2,
        speedup_greedy,
        bar_h,
        color=GREEDY_COLOR,
        alpha=0.85,
        label="Greedy",
        edgecolor="white",
        linewidth=0.5,
    )

    # Breakeven line
    ax.axvline(x=1, color="0.3", linestyle="--", linewidth=0.8, zorder=0)
    ax.text(1.05, len(datasets_with_data) - 0.3, "breakeven", fontsize=7, color="0.4", va="top")

    # Annotate bars with speedup values
    for bars in [bars_exh, bars_greedy]:
        for bar in bars:
            w = bar.get_width()
            if np.isfinite(w) and w > 0:
                ax.text(
                    w * 1.05,
                    bar.get_y() + bar.get_height() / 2,
                    f"{w:.1f}x",
                    va="center",
                    fontsize=7,
                )

    # Geometric mean annotation
    geo_mean_exh = np.exp(np.nanmean(np.log([s for s in speedup_exh if s > 0])))
    geo_mean_greedy = np.exp(np.nanmean(np.log([s for s in speedup_greedy if s > 0])))
    ax.text(
        0.95,
        0.02,
        f"Geo. mean: Exh={geo_mean_exh:.1f}x, Greedy={geo_mean_greedy:.1f}x",
        transform=ax.transAxes,
        fontsize=7,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.5},
    )

    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([DATASET_DISPLAY[ds] for ds in datasets_with_data])
    ax.set_xlabel("Speedup factor (GED / IsalGraph)")
    ax.set_title("C3.1: Computational Speedup", fontsize=PLOT_SETTINGS["axes_titlesize"])
    ax.legend(fontsize=PLOT_SETTINGS["legend_fontsize"], loc="lower right")
    ax.invert_yaxis()

    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, "population_speedup_bar"))
    plt.close(fig)
    logger.info("C3.1 population figure saved.")


def generate_c31_individual(
    all_stats: dict[str, dict],
    comp_dir: str,
    output_dir: str,
) -> None:
    """C3.1 Individual: Timeline comparing GED vs IsalGraph for one LINUX pair."""
    fig_dir = os.path.join(output_dir, "C3_1_speedup")
    os.makedirs(fig_dir, exist_ok=True)

    # Pick one representative pair from LINUX
    ds = "linux"
    ged_df = _load_ged_csv(comp_dir, ds)
    enc_df = _load_encoding_csv(comp_dir, ds)
    lev_df = _load_lev_csv(comp_dir, ds)

    if ged_df is None or enc_df is None or lev_df is None:
        logger.warning("Cannot generate C3.1 individual: missing LINUX CSVs.")
        return

    # Pick a pair with moderate speedup (5-15x) for readable visualization
    # Avoid extreme pairs where the IsalGraph bar would be invisible
    ged_df_sorted = ged_df.sort_values("ged_time_median_s", ascending=True)
    # Target the median GED time pair for visual balance
    pair = ged_df_sorted.iloc[len(ged_df_sorted) // 2]
    gi, gj = pair["graph_i"], pair["graph_j"]

    ged_time = pair["ged_time_median_s"]

    # Get encoding times for the two graphs
    enc_i_row = enc_df[enc_df["graph_id"] == gi]
    enc_j_row = enc_df[enc_df["graph_id"] == gj]

    enc_time_i = enc_i_row["exhaustive_time_median_s"].values[0] if len(enc_i_row) > 0 else 0
    enc_time_j = enc_j_row["exhaustive_time_median_s"].values[0] if len(enc_j_row) > 0 else 0

    # Get Levenshtein time for the pair
    lev_row = lev_df[(lev_df["graph_i"] == gi) & (lev_df["graph_j"] == gj)]
    if len(lev_row) == 0:
        lev_row = lev_df[(lev_df["graph_i"] == gj) & (lev_df["graph_j"] == gi)]
    lev_time = lev_row["c_ext_time_median_s"].values[0] if len(lev_row) > 0 else 0

    total_isal = enc_time_i + enc_time_j + lev_time
    speedup = ged_time / total_isal if total_isal > 0 else 0

    fig, ax = plt.subplots(figsize=get_figure_size("single", height_ratio=0.7))

    # GED bar (top)
    ax.barh(1, ged_time * 1000, 0.5, color=GED_COLOR, alpha=0.85, label="GED (A*)")

    # IsalGraph stacked bar (bottom)
    ax.barh(0, enc_time_i * 1000, 0.5, color=ENCODE_COLOR, alpha=0.85, label="Encode $G_i$")
    ax.barh(
        0,
        enc_time_j * 1000,
        0.5,
        left=enc_time_i * 1000,
        color=PAUL_TOL_BRIGHT["yellow"],
        alpha=0.85,
        label="Encode $G_j$",
    )
    ax.barh(
        0,
        lev_time * 1000,
        0.5,
        left=(enc_time_i + enc_time_j) * 1000,
        color=LEV_COLOR,
        alpha=0.85,
        label="Levenshtein",
    )

    # Speedup annotation
    ax.text(
        max(ged_time, total_isal) * 1000 * 1.02,
        0.5,
        f"Speedup = {speedup:.1f}x",
        va="center",
        fontsize=PLOT_SETTINGS["annotation_fontsize"],
        fontweight="bold",
    )

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["IsalGraph", "GED"])
    ax.set_xlabel("Time (ms)")
    ax.set_title(
        f"C3.1: {gi} vs {gj}",
        fontsize=PLOT_SETTINGS["axes_titlesize"],
    )
    ax.legend(
        fontsize=7,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.18),
        columnspacing=0.8,
        handletextpad=0.4,
    )

    fig.subplots_adjust(bottom=0.3)
    save_figure(fig, os.path.join(fig_dir, "individual_timing_trace"))
    plt.close(fig)
    logger.info("C3.1 individual figure saved.")


def generate_c31_table(
    all_stats: dict[str, dict],
    comp_dir: str,
    output_dir: str,
) -> None:
    """C3.1 Table: Dataset, T_GED, T_encode, T_Lev, T_total, Speedup."""
    fig_dir = os.path.join(output_dir, "C3_1_speedup")
    os.makedirs(fig_dir, exist_ok=True)

    rows = []
    for ds in ALL_DATASETS:
        if ds not in all_stats:
            continue
        st = all_stats[ds]
        ged_med = st["ged_median_s"] * 1000  # ms
        enc_med = st["encoding_median_s"] * 1000
        lev_med = st["levenshtein_median_s"] * 1000
        t_total = 2 * enc_med + lev_med
        speedup = ged_med / t_total if t_total > 0 else float("nan")

        rows.append(
            {
                "Dataset": DATASET_DISPLAY[ds],
                r"$T_{\text{GED}}$ (ms)": f"{ged_med:.2f}",
                r"$T_{\text{encode}}$ (ms)": f"{enc_med:.4f}",
                r"$T_{\text{Lev}}$ (ms)": f"{lev_med:.6f}",
                r"$T_{\text{total}}$ (ms)": f"{t_total:.4f}",
                "Speedup": f"{speedup:.1f}x",
            }
        )

    df = pd.DataFrame(rows)
    generate_dual_table(
        df,
        fig_dir,
        "table_speedup",
        caption="Computational speedup: GED vs IsalGraph pipeline (median times).",
        label="tab:speedup",
        highlight_cols={"Speedup"},
    )
    logger.info("C3.1 table saved.")


# =============================================================================
# C3.2 — Crossover Point
# =============================================================================


def generate_c32_population(
    all_stats: dict[str, dict],
    comp_dir: str,
    output_dir: str,
) -> None:
    """C3.2 Population: Log-log plot of time vs n with crossover markers.

    Three curves: GED, IsalGraph(exhaustive), IsalGraph(greedy).
    Regression lines. Crossover annotations.
    """
    fig_dir = os.path.join(output_dir, "C3_2_crossover")
    os.makedirs(fig_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=get_figure_size("double", height_ratio=0.55))

    # Aggregate timing data from all datasets
    all_ged_n = []
    all_ged_t = []
    all_exh_n = []
    all_exh_t = []
    all_greedy_n = []
    all_greedy_t = []

    for ds in ALL_DATASETS:
        enc_df = _load_encoding_csv(comp_dir, ds)
        ged_df = _load_ged_csv(comp_dir, ds)

        if enc_df is None or ged_df is None:
            continue

        # Encoding times per graph (n -> time)
        for _, row in enc_df.iterrows():
            n = row["n_nodes"]
            exh_t = row["exhaustive_time_median_s"]
            greedy_t = row["greedy_time_median_s"]
            all_exh_n.append(n)
            all_exh_t.append(exh_t)
            all_greedy_n.append(n)
            all_greedy_t.append(greedy_t)

        # GED times per pair (max_n -> time)
        for _, row in ged_df.iterrows():
            all_ged_n.append(row["max_n"])
            all_ged_t.append(row["ged_time_median_s"])

    all_ged_n = np.array(all_ged_n)
    all_ged_t = np.array(all_ged_t)
    all_exh_n = np.array(all_exh_n)
    all_exh_t = np.array(all_exh_t)
    all_greedy_n = np.array(all_greedy_n)
    all_greedy_t = np.array(all_greedy_t)

    # Filter valid points
    ged_mask = (all_ged_t > 0) & np.isfinite(all_ged_t)
    exh_mask = (all_exh_t > 0) & np.isfinite(all_exh_t)
    greedy_mask = (all_greedy_t > 0) & np.isfinite(all_greedy_t)

    # Scatter raw data
    ax.scatter(
        all_ged_n[ged_mask],
        all_ged_t[ged_mask],
        c=GED_COLOR,
        alpha=0.15,
        s=8,
        zorder=2,
        label="_nolegend_",
    )
    ax.scatter(
        all_exh_n[exh_mask],
        all_exh_t[exh_mask],
        c=EXHAUSTIVE_COLOR,
        alpha=0.15,
        s=8,
        zorder=2,
        label="_nolegend_",
    )
    ax.scatter(
        all_greedy_n[greedy_mask],
        all_greedy_t[greedy_mask],
        c=GREEDY_COLOR,
        alpha=0.15,
        s=8,
        zorder=2,
        label="_nolegend_",
    )

    # Compute binned medians for cleaner trend lines
    for data_n, data_t, color, label, marker in [
        (all_ged_n[ged_mask], all_ged_t[ged_mask], GED_COLOR, "GED (A*)", "o"),
        (all_exh_n[exh_mask], all_exh_t[exh_mask], EXHAUSTIVE_COLOR, "Exhaustive enc.", "s"),
        (all_greedy_n[greedy_mask], all_greedy_t[greedy_mask], GREEDY_COLOR, "Greedy enc.", "^"),
    ]:
        ns = data_n.astype(int)
        un = sorted(set(ns))
        medians_n = []
        medians_t = []
        q25_t = []
        q75_t = []
        for n_val in un:
            mask_n = ns == n_val
            if mask_n.sum() >= 3:
                medians_n.append(n_val)
                vals = data_t[mask_n]
                medians_t.append(np.median(vals))
                q25_t.append(np.percentile(vals, 25))
                q75_t.append(np.percentile(vals, 75))

        if len(medians_n) > 1:
            medians_n = np.array(medians_n)
            medians_t = np.array(medians_t)
            q25_t = np.array(q25_t)
            q75_t = np.array(q75_t)

            ax.plot(
                medians_n,
                medians_t,
                color=color,
                linewidth=PLOT_SETTINGS["line_width_thick"],
                marker=marker,
                markersize=5,
                label=label,
                zorder=4,
            )
            ax.fill_between(
                medians_n,
                q25_t,
                q75_t,
                color=color,
                alpha=0.15,
                zorder=1,
            )

    # Fit and plot regression lines (power law: t = c * n^alpha)
    n_fit = np.linspace(2, 14, 100)
    for data_n, data_t, mask, color, _scaling_key in [
        (all_ged_n, all_ged_t, ged_mask, GED_COLOR, "ged"),
        (all_exh_n, all_exh_t, exh_mask, EXHAUSTIVE_COLOR, "exh"),
        (all_greedy_n, all_greedy_t, greedy_mask, GREEDY_COLOR, "greedy"),
    ]:
        log_n = np.log(data_n[mask])
        log_t = np.log(data_t[mask])
        valid = np.isfinite(log_n) & np.isfinite(log_t)
        if valid.sum() > 2:
            slope, intercept, r_value, p_value, _ = sp_stats.linregress(log_n[valid], log_t[valid])
            fit_t = np.exp(intercept) * n_fit**slope
            ax.plot(
                n_fit,
                fit_t,
                color=color,
                linestyle="--",
                linewidth=0.8,
                alpha=0.6,
                zorder=3,
            )

    # Reference complexity lines
    n_ref = np.linspace(3, 13, 50)
    # Normalize: pick a baseline value near median of GED data
    t_base = np.median(all_ged_t[ged_mask & (all_ged_n.astype(int) == 6)])
    n_base = 6.0

    for exp_val, line_label, ls in [
        (2, r"$O(n^2)$", ":"),
        (3, r"$O(n^3)$", "-."),
    ]:
        t_ref = t_base * (n_ref / n_base) ** exp_val
        ax.plot(n_ref, t_ref, color="0.6", linestyle=ls, linewidth=0.6, alpha=0.5, label=line_label)

    # Crossover annotations
    for ds in ALL_DATASETS:
        if ds not in all_stats:
            continue
        xover_n = all_stats[ds]["crossover"]["crossover_n"]
        if xover_n is not None:
            ax.axvline(
                x=xover_n,
                color="0.5",
                linestyle=":",
                linewidth=0.5,
                alpha=0.4,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of nodes (n)")
    ax.set_ylabel("Time per operation (s)")
    ax.set_title(
        "C3.2: Scaling — GED vs IsalGraph Encoding", fontsize=PLOT_SETTINGS["axes_titlesize"]
    )
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xticks([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    ax.set_xticklabels(["3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])

    ax.legend(fontsize=PLOT_SETTINGS["legend_fontsize"], loc="upper left", ncol=2)

    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, "population_crossover_curves"))
    plt.close(fig)
    logger.info("C3.2 population figure saved.")


def generate_c32_individual(
    all_stats: dict[str, dict],
    comp_dir: str,
    output_dir: str,
) -> None:
    """C3.2 Individual: Zoomed crossover region for one dataset."""
    fig_dir = os.path.join(output_dir, "C3_2_crossover")
    os.makedirs(fig_dir, exist_ok=True)

    # Use AIDS (clearest crossover with all bins populated)
    ds = "aids"
    if ds not in all_stats:
        logger.warning("Cannot generate C3.2 individual: missing AIDS stats.")
        return

    st = all_stats[ds]
    bins = st["crossover"]["bins"]

    bin_centers = []
    ged_times = []
    isal_times = []

    for b in bins:
        if b["t_ged_s"] is not None and b["t_isalgraph_s"] is not None:
            center = (b["bin_lo"] + b["bin_hi"]) / 2
            bin_centers.append(center)
            ged_times.append(b["t_ged_s"])
            isal_times.append(b["t_isalgraph_s"])

    if not bin_centers:
        logger.warning("No valid crossover bins for AIDS.")
        return

    fig, ax = plt.subplots(figsize=get_figure_size("single", height_ratio=0.8))

    ax.plot(
        bin_centers,
        np.array(ged_times) * 1000,
        color=GED_COLOR,
        marker="o",
        linewidth=PLOT_SETTINGS["line_width_thick"],
        markersize=6,
        label="GED",
    )
    ax.plot(
        bin_centers,
        np.array(isal_times) * 1000,
        color=TOTAL_ISAL_COLOR,
        marker="s",
        linewidth=PLOT_SETTINGS["line_width_thick"],
        markersize=6,
        label="IsalGraph",
    )

    # Shade regions
    bc = np.array(bin_centers)
    gt = np.array(ged_times) * 1000
    it = np.array(isal_times) * 1000

    ged_faster = gt < it
    isal_faster = gt >= it

    if np.any(isal_faster):
        ax.fill_between(
            bc,
            gt,
            it,
            where=isal_faster,
            color=ENCODE_COLOR,
            alpha=0.1,
            label="IsalGraph faster",
        )
    if np.any(ged_faster):
        ax.fill_between(
            bc,
            gt,
            it,
            where=ged_faster,
            color=GED_COLOR,
            alpha=0.1,
            label="GED faster",
        )

    # Annotate speedup at each bin
    for c, g, isal in zip(bin_centers, gt, it, strict=True):
        sp = g / isal if isal > 0 else 0
        ax.annotate(
            f"{sp:.1f}x",
            (c, min(g, isal)),
            textcoords="offset points",
            xytext=(0, -12),
            fontsize=7,
            ha="center",
            color="0.3",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Number of nodes (n)")
    ax.set_ylabel("Time per pair (ms)")
    ax.set_title(
        f"C3.2: Crossover — {DATASET_DISPLAY[ds]}", fontsize=PLOT_SETTINGS["axes_titlesize"]
    )
    ax.legend(fontsize=PLOT_SETTINGS["legend_fontsize"], loc="upper left")

    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, "individual_crossover_example"))
    plt.close(fig)
    logger.info("C3.2 individual figure saved.")


def generate_c32_table(
    all_stats: dict[str, dict],
    output_dir: str,
) -> None:
    """C3.2 Table: Method, scaling model, exponent, R², crossover n, speedup at n=12."""
    fig_dir = os.path.join(output_dir, "C3_2_crossover")
    os.makedirs(fig_dir, exist_ok=True)

    rows = []
    for ds in ALL_DATASETS:
        if ds not in all_stats:
            continue
        sc = all_stats[ds]["scaling"]
        xover_n = all_stats[ds]["crossover"]["crossover_n"]

        # GED
        ged_s = sc["ged_vs_max_n"]
        rows.append(
            {
                "Dataset": DATASET_DISPLAY[ds],
                "Method": "GED",
                r"$\alpha$": f"{ged_s['alpha']:.2f}",
                r"$R^2$": f"{ged_s['r_squared']:.3f}",
                r"$n^*$": str(xover_n) if xover_n else "---",
            }
        )

        # Exhaustive
        exh_s = sc["exhaustive_encoding_vs_n"]
        rows.append(
            {
                "Dataset": DATASET_DISPLAY[ds],
                "Method": "Exh. enc.",
                r"$\alpha$": f"{exh_s['alpha']:.2f}",
                r"$R^2$": f"{exh_s['r_squared']:.3f}",
                r"$n^*$": "",
            }
        )

        # Greedy
        gr_s = sc["greedy_encoding_vs_n"]
        rows.append(
            {
                "Dataset": DATASET_DISPLAY[ds],
                "Method": "Greedy enc.",
                r"$\alpha$": f"{gr_s['alpha']:.2f}",
                r"$R^2$": f"{gr_s['r_squared']:.3f}",
                r"$n^*$": "",
            }
        )

    df = pd.DataFrame(rows)
    generate_dual_table(
        df,
        fig_dir,
        "table_crossover",
        caption="Scaling exponents and crossover points.",
        label="tab:crossover",
        highlight_cols={r"$R^2$"},
    )
    logger.info("C3.2 table saved.")


# =============================================================================
# C3.3 — Amortized Pipeline
# =============================================================================


def generate_c33_population(
    all_stats: dict[str, dict],
    comp_dir: str,
    output_dir: str,
) -> None:
    """C3.3 Population: Stacked bar per dataset — encoding + Lev vs GED total."""
    fig_dir = os.path.join(output_dir, "C3_3_amortized_pipeline")
    os.makedirs(fig_dir, exist_ok=True)

    # Use amortized CSVs — take n_graphs=100 row for each dataset
    ds_labels = []
    enc_times = []
    lev_times = []
    ged_times = []
    speedups = []

    for ds in ALL_DATASETS:
        amort_df = _load_amortized_csv(comp_dir, ds)
        if amort_df is None:
            continue

        # Pick the row closest to n_graphs=100
        target_n = 100
        if target_n in amort_df["n_graphs"].values:
            row = amort_df[amort_df["n_graphs"] == target_n].iloc[0]
        else:
            row = amort_df.iloc[min(2, len(amort_df) - 1)]

        ds_labels.append(DATASET_DISPLAY[ds])
        enc_times.append(row["total_encoding_time_s"])
        lev_times.append(row["total_levenshtein_time_s"])
        ged_times.append(row["total_ged_time_s"])
        speedups.append(row["speedup"])

    if not ds_labels:
        logger.warning("No amortized data found.")
        return

    fig, ax = plt.subplots(figsize=get_figure_size("double", height_ratio=0.55))

    x = np.arange(len(ds_labels))
    bar_w = 0.35

    enc_arr = np.array(enc_times)
    lev_arr = np.array(lev_times)
    ged_arr = np.array(ged_times)
    isal_arr = enc_arr + lev_arr  # total IsalGraph

    # IsalGraph total bar (encoding dominates >99%)
    ax.bar(
        x - bar_w / 2,
        isal_arr,
        bar_w,
        color=ENCODE_COLOR,
        alpha=0.85,
        label="IsalGraph (encode + Lev)",
        edgecolor="white",
        linewidth=0.5,
    )

    # GED total
    ax.bar(
        x + bar_w / 2,
        ged_arr,
        bar_w,
        color=GED_COLOR,
        alpha=0.85,
        label="GED total",
        edgecolor="white",
        linewidth=0.5,
    )

    # Annotate speedups
    for i, sp in enumerate(speedups):
        max_h = max(isal_arr[i], ged_arr[i])
        ax.text(
            x[i],
            max_h * 1.5,
            f"{sp:.0f}x",
            ha="center",
            fontsize=PLOT_SETTINGS["annotation_fontsize"],
            fontweight="bold",
            color="0.2",
        )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(ds_labels)
    ax.set_ylabel("Total time (s)")
    ax.set_title(
        "C3.3: Amortized Pipeline Comparison (N=100 graphs)",
        fontsize=PLOT_SETTINGS["axes_titlesize"],
    )
    ax.legend(fontsize=PLOT_SETTINGS["legend_fontsize"], loc="upper left")
    # Note: encoding fraction is >99.9% in all datasets; Levenshtein is negligible
    ax.text(
        0.98,
        0.02,
        "Encoding >99.9% of IsalGraph time",
        transform=ax.transAxes,
        fontsize=7,
        ha="right",
        va="bottom",
        style="italic",
        color="0.4",
    )

    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, "population_amortized_comparison"))
    plt.close(fig)
    logger.info("C3.3 population figure saved.")


def generate_c33_individual(
    all_stats: dict[str, dict],
    comp_dir: str,
    output_dir: str,
) -> None:
    """C3.3 Individual: Pie chart decomposing IsalGraph time for largest dataset."""
    fig_dir = os.path.join(output_dir, "C3_3_amortized_pipeline")
    os.makedirs(fig_dir, exist_ok=True)

    # Use AIDS as largest dataset (most graphs with n up to 12)
    ds = "aids"
    amort_df = _load_amortized_csv(comp_dir, ds)
    if amort_df is None:
        logger.warning("Cannot generate C3.3 individual: missing AIDS amortized CSV.")
        return

    # Take the largest n_graphs row
    row = amort_df.iloc[-1]
    enc_frac = row["encoding_fraction"]
    lev_frac = 1.0 - enc_frac
    total_isal = row["total_isalgraph_time_s"]
    total_ged = row["total_ged_time_s"]

    # Since encoding dominates (~99.9%), use a horizontal stacked bar instead of
    # pie charts, which renders the relative sizes more clearly.
    fig, ax = plt.subplots(figsize=get_figure_size("single", height_ratio=0.45))

    n_graphs = int(row["n_graphs"])
    enc_time = total_isal * enc_frac
    lev_time_val = total_isal * lev_frac

    # IsalGraph: stacked bar
    ax.barh(
        0,
        enc_time,
        0.5,
        color=ENCODE_COLOR,
        alpha=0.85,
        label="Encoding",
    )
    ax.barh(
        0,
        lev_time_val,
        0.5,
        left=enc_time,
        color=LEV_COLOR,
        alpha=0.85,
        label=f"Levenshtein ({lev_frac * 100:.2f}%)",
    )
    # GED bar
    ax.barh(1, total_ged, 0.5, color=GED_COLOR, alpha=0.85, label="GED")

    speedup_val = total_ged / total_isal if total_isal > 0 else 0
    ax.text(
        total_ged * 1.02,
        0.5,
        f"Speedup = {speedup_val:.0f}x",
        va="center",
        fontsize=PLOT_SETTINGS["annotation_fontsize"],
        fontweight="bold",
    )

    ax.set_yticks([0, 1])
    ax.set_yticklabels([f"IsalGraph\n({total_isal:.1f}s)", f"GED\n({total_ged:.0f}s)"])
    ax.set_xlabel("Total time (s)")
    ax.set_title(
        f"C3.3: Time Budget — {DATASET_DISPLAY[ds]} (N={n_graphs})",
        fontsize=PLOT_SETTINGS["axes_titlesize"],
    )
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xscale("log")

    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, "individual_pipeline_breakdown"))
    plt.close(fig)
    logger.info("C3.3 individual figure saved.")


def generate_c33_table(
    all_stats: dict[str, dict],
    comp_dir: str,
    output_dir: str,
) -> None:
    """C3.3 Table: Dataset, N, Pairs, T_encode, T_Lev, T_IsalGraph, T_GED, Speedup."""
    fig_dir = os.path.join(output_dir, "C3_3_amortized_pipeline")
    os.makedirs(fig_dir, exist_ok=True)

    rows = []
    for ds in ALL_DATASETS:
        amort_df = _load_amortized_csv(comp_dir, ds)
        if amort_df is None:
            continue

        # Pick n=100 row
        target_n = 100
        if target_n in amort_df["n_graphs"].values:
            row = amort_df[amort_df["n_graphs"] == target_n].iloc[0]
        else:
            row = amort_df.iloc[min(2, len(amort_df) - 1)]

        rows.append(
            {
                "Dataset": DATASET_DISPLAY[ds],
                "N": int(row["n_graphs"]),
                "Pairs": int(row["n_pairs"]),
                r"$T_{\text{enc}}$ (s)": f"{row['total_encoding_time_s']:.2f}",
                r"$T_{\text{Lev}}$ (s)": f"{row['total_levenshtein_time_s']:.4f}",
                r"$T_{\text{Isal}}$ (s)": f"{row['total_isalgraph_time_s']:.2f}",
                r"$T_{\text{GED}}$ (s)": f"{row['total_ged_time_s']:.0f}",
                "Speedup": f"{row['speedup']:.0f}x",
            }
        )

    df = pd.DataFrame(rows)
    generate_dual_table(
        df,
        fig_dir,
        "table_amortized",
        caption="Amortized pipeline comparison (N=100 graphs).",
        label="tab:amortized",
        highlight_cols={"Speedup"},
    )
    logger.info("C3.3 table saved.")


# =============================================================================
# Main
# =============================================================================


def generate_all(comp_dir: str, output_dir: str) -> None:
    """Generate all computational advantage figures and tables."""
    all_stats = _load_timing_stats(comp_dir)
    if not all_stats:
        logger.error("No timing stats found in %s/stats/", comp_dir)
        return

    # C3.1 — Speedup
    generate_c31_population(all_stats, output_dir)
    generate_c31_individual(all_stats, comp_dir, output_dir)
    generate_c31_table(all_stats, comp_dir, output_dir)

    # C3.2 — Crossover
    generate_c32_population(all_stats, comp_dir, output_dir)
    generate_c32_individual(all_stats, comp_dir, output_dir)
    generate_c32_table(all_stats, output_dir)

    # C3.3 — Amortized Pipeline
    generate_c33_population(all_stats, comp_dir, output_dir)
    generate_c33_individual(all_stats, comp_dir, output_dir)
    generate_c33_table(all_stats, comp_dir, output_dir)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate computational advantage figures (C3.1, C3.2, C3.3).",
    )
    parser.add_argument(
        "--comp-dir",
        default=DEFAULT_COMP_DIR,
        help="Directory with eval_computational outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for figures and tables.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    apply_ieee_style()

    generate_all(args.comp_dir, args.output_dir)
    logger.info("All computational figures saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
