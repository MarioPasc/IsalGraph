# ruff: noqa: E402
"""Empirical complexity of IsalGraph encoding methods on random graphs.

Generates a semilogy plot of encoding time T(n) vs number of nodes n
for three methods: Canonical, Greedy-Min, and Greedy-rnd(v₀).
Data from Erdős–Rényi G(n,p) and Barabási–Albert random graph families.
Demonstrates polynomial time complexity T ~ O(n^α) via OLS on log-log.

Usage:
    python -m benchmarks.eval_visualizations.fig_empirical_complexity \
        --data-dir /media/mpascual/Sandisk2TB/research/isalgraph/results/eval_benchmarks/eval_encoding/raw \
        --output-dir /media/mpascual/Sandisk2TB/research/isalgraph/results/figures/complexity
"""

from __future__ import annotations

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from benchmarks.plotting_styles import (
    PAUL_TOL_BRIGHT,
    PLOT_SETTINGS,
    apply_ieee_style,
    get_figure_size,
    save_figure,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_FAMILIES = {"ba_m1", "ba_m2", "gnp_03", "gnp_05"}

METHOD_COLORS = {
    "canonical": PAUL_TOL_BRIGHT["blue"],
    "greedy_min": PAUL_TOL_BRIGHT["red"],
    "greedy_single": PAUL_TOL_BRIGHT["green"],
}

METHOD_LABELS = {
    "canonical": "Canonical",
    "greedy_min": "Greedy-Min",
    "greedy_single": r"Greedy-rnd($v_0$)",
}

METHOD_MARKERS = {
    "canonical": "o",
    "greedy_min": "s",
    "greedy_single": "^",
}

FAMILY_DISPLAY = {
    "ba_m1": "BA(m=1)",
    "ba_m2": "BA(m=2)",
    "gnp_03": "G(n,0.3)",
    "gnp_05": "G(n,0.5)",
}

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def _load_greedy_single_times(data_dir: str) -> pd.DataFrame:
    """Load per-start-node greedy times, aggregate to per-instance median.

    Each row in the CSV is one G2S run from one starting node.
    We take the median across start nodes per (family, n_nodes, instance)
    to represent the typical time for one greedy encoding.
    """
    path = os.path.join(data_dir, "synthetic_greedy_times.csv")
    df = pd.read_csv(path)
    df = df[df["family"].isin(RANDOM_FAMILIES)]

    # Median across start nodes → one time per graph instance
    agg = df.groupby(["family", "n_nodes", "instance"])["greedy_time_s"].median().reset_index()
    return agg


def _load_canonical_greedy_min_times(data_dir: str) -> pd.DataFrame:
    """Load canonical and greedy-min times from the canonical CSV.

    Rows with inf or missing times (timeouts) are filtered out.
    """
    path = os.path.join(data_dir, "synthetic_canonical_times.csv")
    df = pd.read_csv(path)
    df = df[df["family"].isin(RANDOM_FAMILIES)]
    df = df.replace([np.inf, -np.inf], np.nan)
    # Keep only rows where canonical completed successfully
    df = df[df["canonical_time_s"].notna() & (df["canonical_time_s"] > 0)]
    return df


def _aggregate_per_n(
    times: pd.Series,
    n_nodes: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute median and IQR per node count.

    Returns:
        (ns, medians, q25, q75) arrays, filtered to n with >= 2 data points.
    """
    df = pd.DataFrame({"n": n_nodes.values, "t": times.values})
    grouped = df.groupby("n")["t"]
    medians = grouped.median()
    q25 = grouped.quantile(0.25)
    q75 = grouped.quantile(0.75)
    counts = grouped.count()

    mask = counts >= 2
    ns = medians.index[mask].values.astype(float)
    return ns, medians[mask].values, q25[mask].values, q75[mask].values


def _fit_polynomial(
    ns: np.ndarray,
    ts: np.ndarray,
) -> tuple[float, float, float]:
    """Fit T = c · n^α via OLS on log-log data.

    Returns:
        (alpha, c, r_squared).
    """
    valid = (ns > 0) & (ts > 0)
    log_n = np.log(ns[valid])
    log_t = np.log(ts[valid])
    result = sp_stats.linregress(log_n, log_t)
    return result.slope, np.exp(result.intercept), result.rvalue**2


# ---------------------------------------------------------------------------
# Figure Generation
# ---------------------------------------------------------------------------


def generate_empirical_complexity(
    data_dir: str,
    output_dir: str,
) -> str:
    """Generate the empirical complexity figure.

    Args:
        data_dir: Directory containing raw CSV timing data.
        output_dir: Directory to save output figures.

    Returns:
        Base path of saved figure.
    """
    apply_ieee_style()

    # --- Load data ---
    gs_df = _load_greedy_single_times(data_dir)
    can_df = _load_canonical_greedy_min_times(data_dir)

    # --- Aggregate per n ---
    gs_ns, gs_med, gs_q25, gs_q75 = _aggregate_per_n(gs_df["greedy_time_s"], gs_df["n_nodes"])
    gm_ns, gm_med, gm_q25, gm_q75 = _aggregate_per_n(can_df["greedy_min_time_s"], can_df["n_nodes"])
    can_ns, can_med, can_q25, can_q75 = _aggregate_per_n(
        can_df["canonical_time_s"], can_df["n_nodes"]
    )

    # --- Polynomial fits ---
    gs_alpha, gs_c, gs_r2 = _fit_polynomial(gs_ns, gs_med)
    gm_alpha, gm_c, gm_r2 = _fit_polynomial(gm_ns, gm_med)
    can_alpha, can_c, can_r2 = _fit_polynomial(can_ns, can_med)

    logger.info(
        "Fitted exponents: Greedy-rnd α=%.2f (R²=%.3f), "
        "Greedy-Min α=%.2f (R²=%.3f), Canonical α=%.2f (R²=%.3f)",
        gs_alpha,
        gs_r2,
        gm_alpha,
        gm_r2,
        can_alpha,
        can_r2,
    )

    # --- Plot ---
    fig, ax = plt.subplots(figsize=get_figure_size("single", height_ratio=0.9))

    methods_data = [
        ("greedy_single", gs_ns, gs_med, gs_q25, gs_q75, gs_alpha, gs_c, gs_r2),
        ("greedy_min", gm_ns, gm_med, gm_q25, gm_q75, gm_alpha, gm_c, gm_r2),
        ("canonical", can_ns, can_med, can_q25, can_q75, can_alpha, can_c, can_r2),
    ]

    for method, ns, med, q25, q75, alpha, c, r2 in methods_data:
        color = METHOD_COLORS[method]
        label = METHOD_LABELS[method]
        marker = METHOD_MARKERS[method]

        # Legend label includes scaling law
        legend_label = f"{label}: $T \\sim n^{{{alpha:.1f}}}$ ($R^2$={r2:.3f})"

        # Data points with IQR error bars
        ax.errorbar(
            ns,
            med,
            yerr=[med - q25, q75 - med],
            fmt=marker,
            color=color,
            label=legend_label,
            markersize=4,
            capsize=1.5,
            capthick=0.5,
            elinewidth=0.5,
            markeredgewidth=0.4,
            zorder=3,
        )

        # Fitted polynomial curve
        n_fit = np.linspace(max(ns.min(), 3), ns.max(), 200)
        t_fit = c * n_fit**alpha
        ax.plot(n_fit, t_fit, "--", color=color, linewidth=0.8, alpha=0.6, zorder=2)

    ax.set_yscale("log")
    ax.set_xlabel(r"Number of nodes $n$")
    ax.set_ylabel("Encoding time (s)")

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # X-axis ticks up to 50
    ax.set_xlim(None, 52)
    ax.set_xticks(np.arange(0, 51, 10))

    # Legend below the plot, outside the axes
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=1,
        fontsize=PLOT_SETTINGS["annotation_fontsize"],
        handletextpad=0.4,
        borderpad=0.4,
        frameon=False,
    )

    fig.tight_layout()

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    base_path = os.path.join(output_dir, "fig_empirical_complexity")
    saved = save_figure(fig, base_path)
    plt.close(fig)

    for p in saved:
        logger.info("Saved: %s", p)

    # --- Save caption ---
    caption_lines = [
        "Empirical time complexity of IsalGraph encoding methods on random graphs "
        "(Barab\\'asi--Albert $m \\in \\{1,2\\}$ and Erd\\H{o}s--R\\'enyi $p \\in \\{0.3, 0.5\\}$). "
        "Horizontal axis: number of nodes $n$; vertical axis: encoding time in seconds (log scale). "
        "Markers show the median across graph instances; error bars denote the interquartile range. "
        "Dashed lines are polynomial fits $T = c \\cdot n^{\\alpha}$ via OLS on log--log data. ",
    ]
    # Caption method display names (LaTeX-safe, no matplotlib math)
    _caption_names = {
        "canonical": "Canonical",
        "greedy_min": "Greedy-Min",
        "greedy_single": "Greedy-rnd($v_0$)",
    }
    for method, _ns, _med, _q25, _q75, alpha, _c, r2 in methods_data:
        caption_lines.append(
            f"{_caption_names[method]}: $\\alpha = {alpha:.1f}$, $R^2 = {r2:.3f}$. "
        )
    caption_lines.append(
        "Greedy methods exhibit polynomial scaling ($\\alpha \\approx 3$--$5$), "
        "while the canonical method scales super-polynomially ($\\alpha \\approx 9$) "
        "on random graphs and becomes infeasible beyond $n \\approx 12$."
    )
    caption_text = "".join(caption_lines)

    caption_path = base_path + "_caption.txt"
    with open(caption_path, "w", encoding="utf-8") as f:
        f.write(caption_text)
    logger.info("Caption saved: %s", caption_path)

    return base_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate empirical complexity figure for IsalGraph encoding."
    )
    parser.add_argument(
        "--data-dir",
        default="/media/mpascual/Sandisk2TB/research/isalgraph/results/eval_benchmarks/eval_encoding/raw",
        help="Directory containing raw CSV timing data.",
    )
    parser.add_argument(
        "--output-dir",
        default="/media/mpascual/Sandisk2TB/research/isalgraph/results/figures/complexity",
        help="Output directory for figures.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    generate_empirical_complexity(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
