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
    "canonical_pruned": PAUL_TOL_BRIGHT["cyan"],
    "greedy_min": PAUL_TOL_BRIGHT["red"],
    "greedy_single": PAUL_TOL_BRIGHT["green"],
    "ged": PAUL_TOL_BRIGHT["purple"],
}

METHOD_LABELS = {
    "canonical": "Canonical",
    "canonical_pruned": "Canonical (Pruned)",
    "greedy_min": "Greedy-Min",
    "greedy_single": r"Greedy-rnd($v_0$)",
    "ged": "GED (per pair)",
}

METHOD_MARKERS = {
    "canonical": "o",
    "canonical_pruned": "D",
    "greedy_min": "s",
    "greedy_single": "^",
    "ged": "X",
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


def _load_ged_times(comp_dir: str) -> pd.DataFrame:
    """Load GED per-pair computation times from all datasets.

    Reads ``{dataset}_ged_times.csv`` files and returns a DataFrame
    with columns ``max_n`` and ``ged_time_median_s``.
    """
    frames = []
    if not os.path.isdir(comp_dir):
        return pd.DataFrame()
    for fname in sorted(os.listdir(comp_dir)):
        if fname.endswith("_ged_times.csv"):
            path = os.path.join(comp_dir, fname)
            df = pd.read_csv(path)
            if "max_n" in df.columns and "ged_time_median_s" in df.columns:
                frames.append(df[["max_n", "ged_time_median_s"]])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


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
    comp_dir: str = "",
) -> str:
    """Generate the empirical complexity figure.

    Args:
        data_dir: Directory containing raw CSV timing data.
        output_dir: Directory to save output figures.
        comp_dir: Directory with computational evaluation outputs
            (``raw/{dataset}_ged_times.csv``). If provided, GED per-pair
            computation time is overlaid.

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

    # Pruned exhaustive (may be absent in older CSV files)
    pe_ns = pe_med = pe_q25 = pe_q75 = None
    if "pruned_exhaustive_time_s" in can_df.columns:
        pe_df = can_df[
            can_df["pruned_exhaustive_time_s"].notna() & (can_df["pruned_exhaustive_time_s"] > 0)
        ]
        if len(pe_df) > 0:
            pe_ns, pe_med, pe_q25, pe_q75 = _aggregate_per_n(
                pe_df["pruned_exhaustive_time_s"], pe_df["n_nodes"]
            )

    # --- Polynomial fits ---
    gs_alpha, gs_c, gs_r2 = _fit_polynomial(gs_ns, gs_med)
    gm_alpha, gm_c, gm_r2 = _fit_polynomial(gm_ns, gm_med)
    can_alpha, can_c, can_r2 = _fit_polynomial(can_ns, can_med)

    pe_alpha = pe_c = pe_r2 = None
    if pe_ns is not None and len(pe_ns) >= 2:
        pe_alpha, pe_c, pe_r2 = _fit_polynomial(pe_ns, pe_med)

    log_parts = [
        f"Greedy-rnd α={gs_alpha:.2f} (R²={gs_r2:.3f})",
        f"Greedy-Min α={gm_alpha:.2f} (R²={gm_r2:.3f})",
        f"Canonical α={can_alpha:.2f} (R²={can_r2:.3f})",
    ]
    if pe_alpha is not None:
        log_parts.append(f"Canonical (Pruned) α={pe_alpha:.2f} (R²={pe_r2:.3f})")
    logger.info("Fitted exponents: %s", ", ".join(log_parts))

    # --- Plot ---
    fig, ax = plt.subplots(figsize=get_figure_size("single", height_ratio=0.9))

    methods_data = [
        ("greedy_single", gs_ns, gs_med, gs_q25, gs_q75, gs_alpha, gs_c, gs_r2),
        ("greedy_min", gm_ns, gm_med, gm_q25, gm_q75, gm_alpha, gm_c, gm_r2),
    ]
    if pe_alpha is not None:
        methods_data.append(
            ("canonical_pruned", pe_ns, pe_med, pe_q25, pe_q75, pe_alpha, pe_c, pe_r2),
        )
    # Keep canonical data for caption but don't plot it
    all_methods_data = methods_data + [
        ("canonical", can_ns, can_med, can_q25, can_q75, can_alpha, can_c, can_r2),
    ]

    for method, ns, med, q25, q75, alpha, c, r2 in methods_data:
        color = METHOD_COLORS[method]
        label = METHOD_LABELS[method]
        marker = METHOD_MARKERS[method]

        # Data points with IQR error bars
        ax.errorbar(
            ns,
            med,
            yerr=[med - q25, q75 - med],
            fmt=marker,
            color=color,
            label=label,
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

    # --- GED per-pair computation time (from real datasets) ---
    ged_alpha = ged_c = ged_r2 = None
    ged_comp_raw = os.path.join(comp_dir, "raw") if comp_dir else ""
    ged_df = _load_ged_times(ged_comp_raw) if ged_comp_raw else pd.DataFrame()
    if not ged_df.empty:
        ged_ns, ged_med, ged_q25, ged_q75 = _aggregate_per_n(
            ged_df["ged_time_median_s"],
            ged_df["max_n"],
        )
        if len(ged_ns) >= 2:
            ged_alpha, ged_c, ged_r2 = _fit_polynomial(ged_ns, ged_med)
            ax.errorbar(
                ged_ns,
                ged_med,
                yerr=[ged_med - ged_q25, ged_q75 - ged_med],
                fmt=METHOD_MARKERS["ged"],
                color=METHOD_COLORS["ged"],
                label=METHOD_LABELS["ged"],
                markersize=4,
                capsize=1.5,
                capthick=0.5,
                elinewidth=0.5,
                markeredgewidth=0.4,
                zorder=3,
            )
            n_fit = np.linspace(max(ged_ns.min(), 3), ged_ns.max(), 200)
            t_fit = ged_c * n_fit**ged_alpha
            ax.plot(
                n_fit,
                t_fit,
                "--",
                color=METHOD_COLORS["ged"],
                linewidth=0.8,
                alpha=0.6,
                zorder=2,
            )
            logger.info(
                "GED (per pair) α=%.2f (R²=%.3f)",
                ged_alpha,
                ged_r2,
            )

    ax.set_yscale("log")
    ax.set_xlabel(r"Number of nodes $n$")
    ax.set_ylabel("Time (s)")

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # X-axis ticks up to 50
    ax.set_xlim(None, 52)
    ax.set_xticks(np.arange(0, 51, 10))

    # Legend inside the plot, lower right
    ax.legend(
        loc="lower right",
        fontsize=PLOT_SETTINGS["annotation_fontsize"],
        handletextpad=0.4,
        borderpad=0.4,
        framealpha=0.85,
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
        "canonical_pruned": "Canonical (Pruned)",
        "greedy_min": "Greedy-Min",
        "greedy_single": "Greedy-rnd($v_0$)",
    }
    for method, _ns, _med, _q25, _q75, alpha, _c, r2 in all_methods_data:
        caption_lines.append(
            f"{_caption_names[method]}: $T \\sim n^{{{alpha:.1f}}}$, $R^2 = {r2:.3f}$. "
        )
    if ged_alpha is not None:
        _caption_names["ged"] = "GED (per pair)"
        caption_lines.append(
            f"GED (per pair): $T \\sim n^{{{ged_alpha:.1f}}}$, $R^2 = {ged_r2:.3f}$. "
        )
    caption_lines.append(
        "Greedy methods exhibit polynomial scaling ($\\alpha \\approx 3$--$5$), "
        "while the canonical method scales super-polynomially ($\\alpha \\approx 9$) "
        "on random graphs and becomes infeasible beyond $n \\approx 12$. "
    )
    if ged_alpha is not None:
        caption_lines.append(
            "GED computation time (per graph pair, from real datasets) is overlaid "
            "for reference: it grows as $T \\sim n^{"
            f"{ged_alpha:.1f}"
            "}$, vastly exceeding all IsalGraph encoding methods. "
            "Note that encoding times are per-graph (synthetic random graphs), "
            "while GED is per-pair (real benchmark datasets); the full IsalGraph "
            "pipeline for one pair ($2 \\times$ encode $+$ Levenshtein) remains "
            "well below GED."
        )
    caption_text = "".join(caption_lines)

    caption_path = base_path + "_caption.txt"
    with open(caption_path, "w", encoding="utf-8") as f:
        f.write(caption_text)
    logger.info("Caption saved: %s", caption_path)

    return base_path


# ---------------------------------------------------------------------------
# Combined figure: complexity + compression ratio (shared x-axis)
# ---------------------------------------------------------------------------


def generate_combined_complexity_ratio(
    encoding_dir: str,
    comp_dir: str,
    msg_raw_dir: str,
    output_dir: str,
) -> str:
    """Generate fig_complexity_ratio_combined.pdf — 1x2 horizontal layout.

    Left panel (a): empirical complexity (encoding time + GED).
    Right panel (b): compression ratio heatmap with colorbar below.
    Independent x-axes (different data ranges).

    Returns:
        Base path of saved figure.
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    apply_ieee_style()
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load encoding data ------------------------------------------------
    gs_df = _load_greedy_single_times(encoding_dir)
    can_df = _load_canonical_greedy_min_times(encoding_dir)

    gs_ns, gs_med, gs_q25, gs_q75 = _aggregate_per_n(gs_df["greedy_time_s"], gs_df["n_nodes"])
    gm_ns, gm_med, gm_q25, gm_q75 = _aggregate_per_n(
        can_df["greedy_min_time_s"],
        can_df["n_nodes"],
    )

    pe_ns = pe_med = pe_q25 = pe_q75 = None
    pe_alpha = pe_c = pe_r2 = None
    if "pruned_exhaustive_time_s" in can_df.columns:
        pe_df = can_df[
            can_df["pruned_exhaustive_time_s"].notna() & (can_df["pruned_exhaustive_time_s"] > 0)
        ]
        if len(pe_df) > 0:
            pe_ns, pe_med, pe_q25, pe_q75 = _aggregate_per_n(
                pe_df["pruned_exhaustive_time_s"],
                pe_df["n_nodes"],
            )
            pe_alpha, pe_c, pe_r2 = _fit_polynomial(pe_ns, pe_med)

    gs_alpha, gs_c, gs_r2 = _fit_polynomial(gs_ns, gs_med)
    gm_alpha, gm_c, gm_r2 = _fit_polynomial(gm_ns, gm_med)
    can_alpha, can_c, can_r2 = _fit_polynomial(
        *_aggregate_per_n(can_df["canonical_time_s"], can_df["n_nodes"])[:2],
    )

    methods_data = [
        ("greedy_single", gs_ns, gs_med, gs_q25, gs_q75, gs_alpha, gs_c, gs_r2),
        ("greedy_min", gm_ns, gm_med, gm_q25, gm_q75, gm_alpha, gm_c, gm_r2),
    ]
    if pe_alpha is not None:
        methods_data.append(
            ("canonical_pruned", pe_ns, pe_med, pe_q25, pe_q75, pe_alpha, pe_c, pe_r2),
        )

    # GED times
    ged_alpha = ged_c = ged_r2 = None
    ged_comp_raw = os.path.join(comp_dir, "raw") if comp_dir else ""
    ged_df = _load_ged_times(ged_comp_raw) if ged_comp_raw else pd.DataFrame()
    ged_data = None
    if not ged_df.empty:
        ged_ns, ged_med, ged_q25, ged_q75 = _aggregate_per_n(
            ged_df["ged_time_median_s"],
            ged_df["max_n"],
        )
        if len(ged_ns) >= 2:
            ged_alpha, ged_c, ged_r2 = _fit_polynomial(ged_ns, ged_med)
            ged_data = (ged_ns, ged_med, ged_q25, ged_q75)

    # ---- Load ratio data ---------------------------------------------------
    from benchmarks.eval_visualizations.fig_message_length import _load_message_length_csvs

    ratio_df = _load_message_length_csvs(msg_raw_dir)
    ratio_col = "ratio_uniform_standard"
    ratio_methods = ["pruned_exhaustive", "greedy", "greedy_single"]
    ratio_df = ratio_df[ratio_df["method"].isin(ratio_methods)]
    ratio_df = ratio_df[np.isfinite(ratio_df[ratio_col]) & (ratio_df[ratio_col] > 0)]

    # ---- Create figure: 1x2 horizontal layout -----------------------------
    fig = plt.figure(figsize=(7.16, 3.2))
    outer_gs = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1, 1],
        wspace=0.40,
        left=0.07,
        right=0.97,
        top=0.91,
        bottom=0.16,
    )

    # Left panel (a): complexity — single axes
    ax_left = fig.add_subplot(outer_gs[0, 0])

    # Right panel (b): heatmap + colorbar — nested 2-row GridSpec
    inner_gs = GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=outer_gs[0, 1],
        height_ratios=[1, 0.05],
        hspace=0.30,
    )
    ax_right = fig.add_subplot(inner_gs[0, 0])
    cbar_ax = fig.add_subplot(inner_gs[1, 0])

    # ---- Left panel: encoding times ----------------------------------------
    for method, ns, med, q25, q75, alpha, c, _r2 in methods_data:
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]
        ax_left.errorbar(
            ns,
            med,
            yerr=[med - q25, q75 - med],
            fmt=marker,
            color=color,
            label=METHOD_LABELS[method],
            markersize=4,
            capsize=1.5,
            capthick=0.5,
            elinewidth=0.5,
            markeredgewidth=0.4,
            zorder=3,
        )
        n_fit = np.linspace(max(ns.min(), 3), ns.max(), 200)
        ax_left.plot(n_fit, c * n_fit**alpha, "--", color=color, lw=0.8, alpha=0.6, zorder=2)

    if ged_data is not None:
        gns, gmed, gq25, gq75 = ged_data
        ax_left.errorbar(
            gns,
            gmed,
            yerr=[gmed - gq25, gq75 - gmed],
            fmt=METHOD_MARKERS["ged"],
            color=METHOD_COLORS["ged"],
            label=METHOD_LABELS["ged"],
            markersize=4,
            capsize=1.5,
            capthick=0.5,
            elinewidth=0.5,
            markeredgewidth=0.4,
            zorder=3,
        )
        n_fit = np.linspace(max(gns.min(), 3), gns.max(), 200)
        ax_left.plot(
            n_fit,
            ged_c * n_fit**ged_alpha,
            "--",
            color=METHOD_COLORS["ged"],
            lw=0.8,
            alpha=0.6,
            zorder=2,
        )

    ax_left.set_yscale("log")
    ax_left.set_xlabel(r"Number of nodes $n$", fontsize=7)
    ax_left.set_ylabel("Time (s)", fontsize=7)
    ax_left.set_title("(a) Empirical time complexity", fontsize=8, loc="left")
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_left.tick_params(labelsize=6)
    ax_left.set_xlim(None, 52)
    ax_left.set_xticks(np.arange(0, 51, 10))
    ax_left.legend(
        loc="lower right",
        fontsize=PLOT_SETTINGS["annotation_fontsize"],
        handletextpad=0.4,
        borderpad=0.4,
        framealpha=0.85,
    )

    # ---- Right panel: compression ratio heatmap ----------------------------
    if not ratio_df.empty:
        x_ratio = ratio_df["n_nodes"].values.astype(float)
        y_ratio = ratio_df[ratio_col].values.astype(float)

        y_lo = min(float(y_ratio.min()), 0.5)
        y_hi = max(float(y_ratio.max()), 3.5)
        x_lo, x_hi = float(x_ratio.min()), float(x_ratio.max())

        n_xbins = min(50, int(x_hi - x_lo + 1))
        n_ybins = 50
        xbins = np.linspace(x_lo - 0.5, x_hi + 0.5, n_xbins + 1)
        ybins = np.linspace(y_lo, y_hi, n_ybins + 1)

        H, xedges, yedges = np.histogram2d(x_ratio, y_ratio, bins=[xbins, ybins])
        H_log = np.log1p(H)

        im = ax_right.pcolormesh(
            xedges,
            yedges,
            H_log.T,
            cmap="viridis",
            vmin=0,
            vmax=H_log.max(),
            rasterized=True,
        )
        ax_right.axhline(y=1.0, color="white", lw=1.0, ls="--", alpha=0.8, zorder=2)

        # OLS trend
        mask = np.isfinite(x_ratio) & np.isfinite(y_ratio)
        if mask.sum() > 2:
            slope, intercept = np.polyfit(x_ratio[mask], y_ratio[mask], 1)
            x_line = np.linspace(x_lo, x_hi, 200)
            ax_right.plot(
                x_line,
                slope * x_line + intercept,
                "-",
                color="red",
                lw=1.2,
                alpha=0.9,
                zorder=3,
            )
            crossover_n = (1.0 - intercept) / slope if abs(slope) > 1e-12 else None
            text_parts = [rf"slope $= {slope:.4f}$"]
            if crossover_n is not None and crossover_n < x_lo:
                text_parts.append(r"ratio $> 1\ \forall\ N$")
            elif crossover_n is not None and x_lo <= crossover_n <= x_hi:
                text_parts.append(rf"$N^* = {crossover_n:.1f}$")
            ax_right.text(
                0.95,
                0.95,
                "\n".join(text_parts),
                transform=ax_right.transAxes,
                fontsize=6,
                va="top",
                ha="right",
                bbox={"boxstyle": "round,pad=0.3", "fc": "wheat", "alpha": 0.85},
                zorder=5,
            )

        cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("Count (log scale)", fontsize=6)
        cbar.ax.tick_params(labelsize=5)

    ax_right.set_xlabel("Number of nodes $N$", fontsize=7)
    ax_right.set_ylabel("Ratio (GED / IsalGraph)", fontsize=7)
    ax_right.set_title("(b) Information content ratio", fontsize=8, loc="left")
    ax_right.tick_params(labelsize=6)

    # ---- Save --------------------------------------------------------------
    base_path = os.path.join(output_dir, "fig_complexity_ratio_combined")
    save_figure(fig, base_path)
    plt.close(fig)

    # ---- Caption -----------------------------------------------------------
    caption = (
        "Combined view of encoding time complexity and information content. "
        "(a) Empirical time complexity of IsalGraph encoding methods on random graphs "
        "(Barab\\'asi--Albert and Erd\\H{o}s--R\\'enyi), with GED per-pair "
        "computation time from real benchmark datasets overlaid for reference. "
        "Vertical axis: time in seconds (log scale). "
        "Dashed lines: polynomial fits $T = c \\cdot n^{\\alpha}$. "
    )
    for method, _ns, _med, _q25, _q75, alpha, _c, r2 in methods_data:
        lbl = METHOD_LABELS.get(method, method)
        caption += f"{lbl}: $T \\sim n^{{{alpha:.1f}}}$. "
    if ged_alpha is not None:
        caption += f"GED (per pair): $T \\sim n^{{{ged_alpha:.1f}}}$. "
    caption += (
        "(b) Compression ratio (GED standard encoding / IsalGraph uniform encoding) "
        "vs.\\ number of nodes, aggregated across all datasets and methods "
        "(2D histogram, log-scaled counts). "
        "White dashed line: break-even (ratio $= 1$). "
        "Red line: OLS trend. "
        "Together, these panels show that IsalGraph's polynomial encoding time "
        "yields consistently shorter representations than the GED construction "
        "model, while being orders of magnitude faster to compute."
    )
    caption_path = base_path + "_caption.txt"
    with open(caption_path, "w", encoding="utf-8") as f:
        f.write(caption + "\n")
    logger.info("Caption saved: %s", caption_path)

    logger.info("Combined figure saved: %s", base_path)
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
        "--comp-dir",
        default="",
        help="Directory with computational evaluation outputs (for GED times).",
    )
    parser.add_argument(
        "--output-dir",
        default="/media/mpascual/Sandisk2TB/research/isalgraph/results/figures/complexity",
        help="Output directory for figures.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    generate_empirical_complexity(args.data_dir, args.output_dir, args.comp_dir)


if __name__ == "__main__":
    main()
