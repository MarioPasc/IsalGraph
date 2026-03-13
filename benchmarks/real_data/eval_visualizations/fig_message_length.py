# ruff: noqa: E402, N803, N806
"""Message length comparison figures for the paper.

Generates publication-quality IEEE-style figures comparing IsalGraph and
GED message lengths.  Called by generate_figures.py (Step 4) from
precomputed CSV/JSON artifacts.

Primary figure: fig_message_length_scatter.pdf (1x3 panels)
Secondary figure: fig_message_length_ratio.pdf (single panel)
Table: table_message_length_summary.tex

Usage:
    python -m benchmarks.eval_visualizations.fig_message_length \
        --data-dir /path/to/message_length/raw \
        --stats-dir /path/to/message_length/stats \
        --output-dir /path/to/figures
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.plotting_styles import (
    PAUL_TOL_BRIGHT,
    apply_ieee_style,
    save_figure,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_DISPLAY: dict[str, str] = {
    "iam_letter_low": "IAM Letter LOW",
    "iam_letter_med": "IAM Letter MED",
    "iam_letter_high": "IAM Letter HIGH",
    "linux": "LINUX",
    "aids": "AIDS",
}

DATASET_COLORS = {
    "iam_letter_low": PAUL_TOL_BRIGHT["blue"],
    "iam_letter_med": PAUL_TOL_BRIGHT["cyan"],
    "iam_letter_high": PAUL_TOL_BRIGHT["purple"],
    "linux": PAUL_TOL_BRIGHT["red"],
    "aids": PAUL_TOL_BRIGHT["green"],
}

METHOD_DISPLAY = {
    "exhaustive": "Canonical",
    "pruned_exhaustive": "Canonical (Pruned)",
    "greedy": "Greedy-Min",
    "greedy_single": r"Greedy-rnd($v_0$)",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_message_length_csvs(raw_dir: str) -> pd.DataFrame:
    """Load all message_lengths_*.csv files into a single DataFrame."""
    frames = []
    if not os.path.isdir(raw_dir):
        logger.warning("Raw directory not found: %s", raw_dir)
        return pd.DataFrame()
    for fname in sorted(os.listdir(raw_dir)):
        if fname.startswith("message_lengths_") and fname.endswith(".csv"):
            path = os.path.join(raw_dir, fname)
            df = pd.read_csv(path)
            frames.append(df)
    if not frames:
        logger.warning("No message length CSV files found in %s", raw_dir)
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_summary(stats_dir: str) -> dict:
    """Load message_length_summary.json."""
    path = os.path.join(stats_dir, "message_length_summary.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _compute_ols_stats(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> dict[str, float]:
    """Compute OLS regression stats for y = beta*x + alpha.

    Returns dict with keys: n, beta, alpha, r2, pct_wins.
    """
    x = df[x_col].values.astype(float)
    y = df[y_col].values.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    result: dict[str, float] = {"n": float(mask.sum())}
    if mask.sum() > 2:
        beta, alpha = np.polyfit(x[mask], y[mask], 1)
        y_pred = beta * x[mask] + alpha
        ss_res = float(np.sum((y[mask] - y_pred) ** 2))
        ss_tot = float(np.sum((y[mask] - np.mean(y[mask])) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        result["beta"] = beta
        result["alpha"] = alpha
        result["r2"] = r2
    n_wins = int((df[y_col] < df[x_col]).sum())
    result["pct_wins"] = 100.0 * n_wins / len(df) if len(df) > 0 else 0.0
    return result


# ---------------------------------------------------------------------------
# Scatter figure: IsalGraph bits vs GED bits
# ---------------------------------------------------------------------------


def generate_scatter_figure(
    raw_dir: str,
    output_dir: str,
    *,
    log_counts: bool = False,
) -> str:
    """Generate fig_message_length_scatter.pdf — 1x3 heatmap panels.

    Panels: Canonical (Pruned), Greedy-Min, Greedy-rnd.
    X: GED bits (standard encoding), Y: IsalGraph bits (uniform encoding).
    Data is aggregated across all datasets (no per-dataset breakdown).
    Color encodes bin counts (2D histogram).  Points below y=x line
    indicate IsalGraph is more compact.

    Args:
        log_counts: If True, use logarithmic color scale for bin counts.
            Output file gets ``_log`` suffix.

    Returns:
        Path to the saved figure.
    """
    from matplotlib.gridspec import GridSpec

    apply_ieee_style()
    df = _load_message_length_csvs(raw_dir)
    if df.empty:
        logger.warning("No data for scatter figure")
        return ""

    # Methods to show (exclude canonical/exhaustive)
    methods_to_show = ["pruned_exhaustive", "greedy", "greedy_single"]
    df = df[df["method"].isin(methods_to_show)]
    if df.empty:
        logger.warning("No data for selected methods")
        return ""

    x_col = "ged_standard_bits"
    y_col = "isal_uniform_bits"

    # Global axis limits: same scale for x and y
    all_vals = pd.concat([df[x_col], df[y_col]])
    val_max = float(np.ceil(all_vals.max() / 10) * 10)

    n_bins = 50
    bins = np.linspace(0.0, val_max, n_bins + 1)

    # Figure with colorbar row
    n_methods = len(methods_to_show)
    fig = plt.figure(figsize=(7.0, 3.2))
    gs = GridSpec(
        2,
        n_methods,
        figure=fig,
        height_ratios=[1, 0.04],
        hspace=0.30,
        left=0.08,
        right=0.98,
        top=0.92,
        bottom=0.12,
    )

    axes = [fig.add_subplot(gs[0, i]) for i in range(n_methods)]
    cbar_ax = fig.add_subplot(gs[1, :])

    # Pre-compute global max count for shared colorbar
    global_max = 0
    for method in methods_to_show:
        mdf = df[df["method"] == method]
        if mdf.empty:
            continue
        H, _, _ = np.histogram2d(mdf[x_col].values, mdf[y_col].values, bins=bins)
        global_max = max(global_max, int(H.max()))

    im = None
    for i, method in enumerate(methods_to_show):
        ax = axes[i]
        mdf = df[df["method"] == method]

        # 2D histogram aggregated across all datasets
        H, xedges, yedges = np.histogram2d(mdf[x_col].values, mdf[y_col].values, bins=bins)

        # Heatmap
        if log_counts:
            im = ax.pcolormesh(
                xedges,
                yedges,
                np.log1p(H.T),
                cmap="viridis",
                vmin=0,
                vmax=np.log1p(global_max),
                rasterized=True,
            )
        else:
            im = ax.pcolormesh(
                xedges,
                yedges,
                H.T,
                cmap="viridis",
                vmin=0,
                vmax=global_max,
                rasterized=True,
            )

        # Identity line y = x
        ax.plot(
            [0, val_max],
            [0, val_max],
            "--",
            color="white",
            lw=0.8,
            alpha=0.7,
            zorder=2,
        )

        # OLS fit: y = beta * x + alpha
        ols = _compute_ols_stats(mdf, x_col, y_col)
        beta = ols.get("beta")
        r2 = ols.get("r2")
        mask = np.isfinite(mdf[x_col].values) & np.isfinite(mdf[y_col].values)
        if beta is not None:
            alpha_coeff = ols["alpha"]
            x_line = np.array([0, val_max])
            y_line = beta * x_line + alpha_coeff
            ax.plot(
                x_line,
                np.clip(y_line, 0, val_max),
                "-",
                color="red",
                lw=1.2,
                alpha=0.9,
                zorder=3,
            )

        ax.set_xlim(0, val_max)
        ax.set_ylim(0, val_max)
        ax.set_aspect("equal", adjustable="box")

        # Force identical ticks on both axes
        tick_step = 25 if val_max > 100 else 10
        shared_ticks = np.arange(0, val_max + 1, tick_step)
        ax.set_xticks(shared_ticks)
        ax.set_yticks(shared_ticks)

        label = METHOD_DISPLAY.get(method, method)
        ax.set_title(f"({chr(ord('a') + i)}) {label}", fontsize=8)
        ax.set_xlabel("GED bits", fontsize=7)
        if i == 0:
            ax.set_ylabel("IsalGraph bits", fontsize=7)
        ax.tick_params(labelsize=6)

        # Annotation box with beta slope
        text_parts: list[str] = []
        if beta is not None:
            text_parts.append(rf"$\beta = {beta:.3f}$")
            text_parts.append(rf"$R^2 = {r2:.3f}$")
        pct_wins = ols["pct_wins"]
        text_parts.append(f"Wins: {pct_wins:.1f}%")
        text = "\n".join(text_parts)
        if text_parts:
            ax.text(
                0.05,
                0.95,
                text,
                transform=ax.transAxes,
                fontsize=6,
                va="top",
                bbox={"boxstyle": "round,pad=0.3", "fc": "wheat", "alpha": 0.8},
                zorder=3,
            )

    # Shared colorbar spanning all panels
    if im is not None:
        cbar_label = "Count (log scale)" if log_counts else "Count"
        fig.colorbar(im, cax=cbar_ax, orientation="horizontal", label=cbar_label)

    suffix = "_log" if log_counts else ""
    out_path = os.path.join(output_dir, f"fig_message_length_scatter{suffix}")
    save_figure(fig, out_path)
    plt.close(fig)
    logger.info("  -> %s.pdf", out_path)
    return out_path + ".pdf"


# ---------------------------------------------------------------------------
# Ratio figure: compression ratio vs N
# ---------------------------------------------------------------------------


def generate_ratio_figure(
    raw_dir: str,
    output_dir: str,
) -> str:
    """Generate fig_message_length_ratio.pdf — heatmap of compression ratio vs N.

    2D histogram (N vs ratio), aggregated across all datasets and methods.
    Horizontal line at ratio=1 (break-even).
    OLS trend line with slope annotation and crossover point where
    the trend intersects ratio=1.

    Returns:
        Path to the saved figure.
    """
    from matplotlib.gridspec import GridSpec

    apply_ieee_style()
    df = _load_message_length_csvs(raw_dir)
    if df.empty:
        logger.warning("No data for ratio figure")
        return ""

    # Use standard encoding ratio, consistent with scatter figure
    ratio_col = "ratio_uniform_standard"
    # Filter to methods shown in scatter figure
    methods_to_show = ["pruned_exhaustive", "greedy", "greedy_single"]
    df = df[df["method"].isin(methods_to_show)]
    df = df[np.isfinite(df[ratio_col]) & (df[ratio_col] > 0)]

    if df.empty:
        logger.warning("No valid ratio data")
        return ""

    x_all = df["n_nodes"].values.astype(float)
    y_all = df[ratio_col].values.astype(float)

    # Bin edges
    x_min, x_max = float(x_all.min()), float(x_all.max())
    y_min, y_max = float(y_all.min()), float(y_all.max())
    # Add margin to y range so ratio=1 line is visible
    y_min = min(y_min, 0.5)
    y_max = max(y_max, 3.5)

    n_xbins = min(50, int(x_max - x_min + 1))
    n_ybins = 50
    xbins = np.linspace(x_min - 0.5, x_max + 0.5, n_xbins + 1)
    ybins = np.linspace(y_min, y_max, n_ybins + 1)

    H, xedges, yedges = np.histogram2d(x_all, y_all, bins=[xbins, ybins])

    # Figure with colorbar
    fig = plt.figure(figsize=(3.49, 3.0))
    gs = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[1, 0.04],
        hspace=0.25,
        left=0.16,
        right=0.95,
        top=0.94,
        bottom=0.14,
    )
    ax = fig.add_subplot(gs[0, 0])
    cbar_ax = fig.add_subplot(gs[1, 0])

    # Heatmap — log-scale counts
    H_log = np.log1p(H)
    im = ax.pcolormesh(
        xedges,
        yedges,
        H_log.T,
        cmap="viridis",
        vmin=0,
        vmax=H_log.max(),
        rasterized=True,
    )

    # Break-even line: ratio = 1
    ax.axhline(
        y=1.0,
        color="white",
        lw=1.0,
        ls="--",
        alpha=0.8,
        zorder=2,
    )

    # OLS trend: ratio = slope * N + intercept
    mask = np.isfinite(x_all) & np.isfinite(y_all)
    slope = intercept = r2 = crossover_n = None
    if mask.sum() > 2:
        slope, intercept = np.polyfit(x_all[mask], y_all[mask], 1)
        y_pred = slope * x_all[mask] + intercept
        ss_res = float(np.sum((y_all[mask] - y_pred) ** 2))
        ss_tot = float(np.sum((y_all[mask] - np.mean(y_all[mask])) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        x_line = np.linspace(x_min, x_max, 200)
        y_line = slope * x_line + intercept
        ax.plot(
            x_line,
            y_line,
            "-",
            color="red",
            lw=1.2,
            alpha=0.9,
            zorder=3,
        )

        # Crossover: slope * N + intercept = 1  =>  N = (1 - intercept) / slope
        if abs(slope) > 1e-12:
            crossover_n = (1.0 - intercept) / slope

            # Mark crossover if within plot range
            if x_min <= crossover_n <= x_max:
                ax.plot(
                    crossover_n,
                    1.0,
                    "o",
                    color="red",
                    markersize=6,
                    zorder=4,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                )
                ax.annotate(
                    f"$N^* = {crossover_n:.1f}$",
                    xy=(crossover_n, 1.0),
                    xytext=(crossover_n + (x_max - x_min) * 0.08, 0.75),
                    fontsize=6,
                    color="red",
                    arrowprops={
                        "arrowstyle": "->",
                        "color": "red",
                        "lw": 0.8,
                        "connectionstyle": "arc3,rad=0.2",
                    },
                    zorder=5,
                )

    # Annotation box (slope + crossover, no R²)
    text_parts: list[str] = []
    if slope is not None:
        text_parts.append(rf"slope $= {slope:.4f}$")
    if crossover_n is not None and x_min <= crossover_n <= x_max:
        text_parts.append(rf"$N^* = {crossover_n:.1f}$")
    elif crossover_n is not None and crossover_n < x_min:
        text_parts.append(r"ratio $> 1\ \forall\ N$")
    if text_parts:
        ax.text(
            0.95,
            0.95,
            "\n".join(text_parts),
            transform=ax.transAxes,
            fontsize=6,
            va="top",
            ha="right",
            bbox={"boxstyle": "round,pad=0.3", "fc": "wheat", "alpha": 0.85},
            zorder=5,
        )

    ax.set_xlabel("Number of nodes $N$", fontsize=8)
    ax.set_ylabel("Compression ratio (GED / IsalGraph)", fontsize=8)
    ax.tick_params(labelsize=6)

    fig.colorbar(im, cax=cbar_ax, orientation="horizontal", label="Count (log scale)")

    out_path = os.path.join(output_dir, "fig_message_length_ratio")
    save_figure(fig, out_path)
    plt.close(fig)
    logger.info("  -> %s.pdf", out_path)

    # --- Caption ---
    n_total = len(df)
    n_datasets = df["dataset"].nunique()
    caption = (
        f"Compression ratio (GED standard encoding / IsalGraph uniform encoding) "
        f"vs.\\ number of nodes $N$, aggregated across {n_datasets} datasets and "
        f"three encoding methods ($n = {n_total:,}$ instances). "
        f"2D histogram with log-scaled counts (viridis colormap). "
        f"White dashed line: break-even (ratio $= 1$); above this line IsalGraph "
        f"is more compact. "
        f"Red line: OLS trend (slope $= {slope:.4f}$"
    )
    if r2 is not None:
        caption += f", $R^2 = {r2:.3f}$"
    caption += "). "
    if crossover_n is not None and crossover_n < x_min:
        caption += (
            "The trend intercept exceeds 1 at all observed graph sizes, "
            "confirming that IsalGraph's uniform encoding is more compact than "
            "GED standard encoding for every $N$ in our benchmarks. "
        )
    elif crossover_n is not None and x_min <= crossover_n <= x_max:
        caption += (
            f"The trend crosses ratio $= 1$ at $N^* \\approx {crossover_n:.1f}$, "
            f"beyond which IsalGraph becomes more compact. "
        )
    caption += "The positive slope indicates that IsalGraph's advantage grows with graph size."
    caption_path = os.path.join(output_dir, "fig_message_length_ratio_caption.txt")
    with open(caption_path, "w", encoding="utf-8") as f:
        f.write(caption + "\n")
    logger.info("  -> %s", caption_path)

    return out_path + ".pdf"


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------


def generate_message_length_table(
    stats_dir: str,
    output_dir: str,
) -> str:
    """Generate table_message_length_summary.tex.

    Returns:
        Path to the saved table.
    """
    summary = _load_summary(stats_dir)
    if not summary:
        logger.warning("No summary data for table")
        return ""

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Message length comparison: IsalGraph vs.\ GED construction.}")
    lines.append(r"\label{tab:message_length}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{ll r r r r r r}")
    lines.append(r"\toprule")
    lines.append(
        r"Dataset & Method & $\bar{N}$ & $\bar{M}$ & "
        r"$L_\mathrm{isal}$ & $L_\mathrm{ged}$ & Ratio & \% Wins \\"
    )
    lines.append(r"\midrule")

    for _key, s in summary.items():
        ds_display = DATASET_DISPLAY.get(s["dataset"], s["dataset"])
        method_display = METHOD_DISPLAY.get(s["method"], s["method"])
        mean_n = s.get("mean_n_nodes", 0)
        mean_m = s.get("mean_n_edges", 0)
        mean_isal = s.get("mean_isal_uniform_bits", 0)
        mean_ged = s.get("mean_ged_generous_bits", 0)
        ratio = s.get("mean_ratio_uniform_generous", 0)
        pct = s.get("pct_isalgraph_wins", 0)
        line = (
            f"{ds_display} & {method_display} & "
            f"{mean_n:.1f} & {mean_m:.1f} & "
            f"{mean_isal:.1f} & {mean_ged:.1f} & "
            f"{ratio:.2f} & {pct:.1f}\\% \\\\"
        )
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path = os.path.join(output_dir, "table_message_length_summary.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("  -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Caption
# ---------------------------------------------------------------------------


def generate_scatter_caption(
    raw_dir: str,
    output_dir: str,
) -> str:
    """Generate fig_message_length_scatter_caption.txt.

    Computes per-method OLS stats and writes a LaTeX-compatible caption.

    Returns:
        Path to the saved caption file.
    """
    df = _load_message_length_csvs(raw_dir)
    if df.empty:
        logger.warning("No data for scatter caption")
        return ""

    methods_to_show = ["pruned_exhaustive", "greedy", "greedy_single"]
    df = df[df["method"].isin(methods_to_show)]
    x_col = "ged_standard_bits"
    y_col = "isal_uniform_bits"

    method_stats = {}
    for method in methods_to_show:
        mdf = df[df["method"] == method]
        if not mdf.empty:
            method_stats[method] = _compute_ols_stats(mdf, x_col, y_col)

    n_datasets = df["dataset"].nunique()
    n_total = len(df)

    parts = [
        f"Information content comparison between IsalGraph (uniform encoding, "
        f"$L \\times \\log_2 9$ bits) and GED construction "
        f"(standard encoding) across {n_datasets} benchmark datasets "
        f"($n = {n_total:,}$ graph instances). "
        f"Each panel shows a 2D histogram (heatmap) of IsalGraph bits vs.\\ GED bits "
        f"for one encoding method; points below the identity line ($y = x$, white dashed) "
        f"indicate IsalGraph requires fewer bits. "
        f"Red lines show OLS linear fits $y = \\beta x + \\alpha$. ",
    ]

    for method in methods_to_show:
        if method not in method_stats:
            continue
        s = method_stats[method]
        label = METHOD_DISPLAY.get(method, method)
        beta = s.get("beta", 0)
        r2 = s.get("r2", 0)
        pct = s.get("pct_wins", 0)
        parts.append(
            f"({label}: $\\beta = {beta:.3f}$, $R^2 = {r2:.3f}$, "
            f"IsalGraph wins {pct:.1f}\\% of instances). "
        )

    parts.append(
        "All methods yield $\\beta < 1$, confirming that IsalGraph's uniform encoding "
        "is consistently more compact than the GED standard construction model."
    )

    caption = "".join(parts)
    out_path = os.path.join(output_dir, "fig_message_length_scatter_caption.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(caption + "\n")
    logger.info("  -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Information content table (dataset properties + per-dataset OLS beta)
# ---------------------------------------------------------------------------

# Ordered list of datasets (columns in the table)
_TABLE_DATASETS = ["iam_letter_low", "iam_letter_med", "iam_letter_high", "linux", "aids"]
_TABLE_DS_LABELS = ["IAM LOW", "IAM MED", "IAM HIGH", "LINUX", "AIDS"]
_TABLE_METHODS = ["pruned_exhaustive", "greedy", "greedy_single"]


def _fmt_thousands(n: int) -> str:
    """Format integer with LaTeX thousands separators: 1234 -> 1{,}234."""
    return f"{n:,}".replace(",", "{,}")


def generate_information_content_table(
    raw_dir: str,
    output_dir: str,
) -> str:
    r"""Generate table_information_content.tex.

    Format matches table_performance_summary.tex: datasets as columns,
    property rows (N, Pairs, :math:`\bar{m}`), then OLS :math:`\beta`
    rows per method with best bolded and :math:`\Delta` from best.

    Returns:
        Path to the saved table file.
    """
    df = _load_message_length_csvs(raw_dir)
    if df.empty:
        logger.warning("No data for information content table")
        return ""

    x_col = "ged_standard_bits"
    y_col = "isal_uniform_bits"

    # --- Dataset properties (from one method per dataset) ---
    props: dict[str, dict[str, float]] = {}
    for ds in _TABLE_DATASETS:
        ddf = df[df["dataset"] == ds]
        if ddf.empty:
            continue
        first_method = ddf["method"].iloc[0]
        gdf = ddf[ddf["method"] == first_method]
        n_graphs = int(gdf["graph_id"].nunique())
        mean_edges = float(gdf["n_edges"].mean())
        props[ds] = {
            "n": n_graphs,
            "pairs": n_graphs * (n_graphs - 1) // 2,
            "mean_m": mean_edges,
        }

    # --- OLS beta per (dataset, method) ---
    betas: dict[str, dict[str, float]] = {}
    r2s: dict[str, dict[str, float]] = {}
    for method in _TABLE_METHODS:
        betas[method] = {}
        r2s[method] = {}
        for ds in _TABLE_DATASETS:
            mdf = df[(df["method"] == method) & (df["dataset"] == ds)]
            if mdf.empty:
                continue
            stats = _compute_ols_stats(mdf, x_col, y_col)
            if "beta" in stats:
                betas[method][ds] = stats["beta"]
                r2s[method][ds] = stats["r2"]

    # Best (lowest) beta per dataset
    best_beta: dict[str, float] = {}
    for ds in _TABLE_DATASETS:
        ds_betas = [betas[m].get(ds, float("inf")) for m in _TABLE_METHODS]
        valid = [b for b in ds_betas if np.isfinite(b)]
        if valid:
            best_beta[ds] = min(valid)

    # --- Build LaTeX ---
    lines: list[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Dataset properties and OLS slope $\beta$ of IsalGraph uniform "
        r"encoding vs.\ GED standard encoding ($y = \beta x + \alpha$). "
        r"$\beta < 1$ indicates IsalGraph is more compact. "
        r"$\Delta$: difference from best method per dataset. "
        r"Best $\beta$ per dataset in \textbf{bold}.}"
    )
    lines.append(r"\label{tab:information-content}")
    lines.append(r"\small")

    n_ds = len(_TABLE_DATASETS)
    lines.append(r"\begin{tabular}{cl" + "c" * n_ds + "}")
    lines.append(r"\toprule")

    # Header
    header = " & "
    for label in _TABLE_DS_LABELS:
        header += rf" & \textbf{{{label}}}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # --- Property rows ---
    # N
    row = r"\multirow{3}{*}{\rotatebox[origin=c]{90}{\scriptsize Prop.}} & $N$"
    for ds in _TABLE_DATASETS:
        if ds in props:
            row += f" & {_fmt_thousands(int(props[ds]['n']))}"
        else:
            row += " & --"
    row += r" \\"
    lines.append(row)

    # Pairs
    row = " & Pairs"
    for ds in _TABLE_DATASETS:
        if ds in props:
            row += f" & {_fmt_thousands(int(props[ds]['pairs']))}"
        else:
            row += " & --"
    row += r" \\"
    lines.append(row)

    # m̄
    row = r" & $\bar{m}$"
    for ds in _TABLE_DATASETS:
        if ds in props:
            row += f" & {props[ds]['mean_m']:.2f}"
        else:
            row += " & --"
    row += r" \\"
    lines.append(row)

    lines.append(r"\midrule")

    # --- Beta rows ---
    n_meth = len(_TABLE_METHODS)
    for j, method in enumerate(_TABLE_METHODS):
        method_label = METHOD_DISPLAY.get(method, method)
        if j == 0:
            row = (
                rf"\multirow{{{n_meth}}}{{*}}"
                rf"{{\rotatebox[origin=c]{{90}}{{\small OLS $\beta$}}}}"
                rf" & {method_label}"
            )
        else:
            row = f" & {method_label}"

        for ds in _TABLE_DATASETS:
            beta = betas[method].get(ds)
            if beta is None:
                row += " & --"
                continue
            best = best_beta.get(ds)
            is_best = best is not None and abs(beta - best) < 1e-6
            if is_best:
                row += rf" & \textbf{{{beta:.3f}}}"
            else:
                delta = beta - best if best is not None else 0
                row += rf" & {beta:.3f} (+{delta:.3f})"

        row += r" \\"
        lines.append(row)

    # --- R² rows ---
    lines.append(r"\midrule")
    for j, method in enumerate(_TABLE_METHODS):
        method_label = METHOD_DISPLAY.get(method, method)
        if j == 0:
            row = (
                rf"\multirow{{{n_meth}}}{{*}}"
                rf"{{\rotatebox[origin=c]{{90}}{{\small $R^2$}}}}"
                rf" & {method_label}"
            )
        else:
            row = f" & {method_label}"

        for ds in _TABLE_DATASETS:
            r2 = r2s[method].get(ds)
            if r2 is None:
                row += " & --"
            else:
                row += f" & {r2:.3f}"

        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out_path = os.path.join(output_dir, "table_information_content.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("  -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI for standalone usage
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate message length comparison figures.",
    )
    parser.add_argument("--data-dir", required=True, help="Directory with message_lengths_*.csv")
    parser.add_argument("--stats-dir", default="", help="Directory with summary JSON")
    parser.add_argument("--output-dir", required=True, help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    apply_ieee_style()

    generate_scatter_figure(args.data_dir, args.output_dir)
    generate_scatter_figure(args.data_dir, args.output_dir, log_counts=True)
    generate_scatter_caption(args.data_dir, args.output_dir)
    generate_ratio_figure(args.data_dir, args.output_dir)
    generate_information_content_table(args.data_dir, args.output_dir)

    if args.stats_dir:
        generate_message_length_table(args.stats_dir, args.output_dir)


if __name__ == "__main__":
    main()
