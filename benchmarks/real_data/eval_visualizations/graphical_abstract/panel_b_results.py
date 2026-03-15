"""Panel B: Three main results in priority order.

Generates the right panel of the graphical abstract:
  B1 (top, largest): Message length comparison scatter/heatmap
  B2 (bottom-left): Computational speedup line chart
  B3 (bottom-right): GED proxy quality formula block
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy import stats as sp_stats

from benchmarks.plotting_styles import (
    PAUL_TOL_BRIGHT,
    apply_ieee_style,
    save_figure,
)

logger = logging.getLogger(__name__)

# Method of interest for the graphical abstract
_TARGET_METHOD = "pruned_exhaustive"
_SPEEDUP_METHOD = "greedy_single"  # Greedy-rnd(v_0)

# Verified fallback values (from run 20260312_180741_30cb1b7)
_FALLBACK_MSG_BETA = 0.537
_FALLBACK_MSG_R2 = 0.947
_FALLBACK_MSG_WINS = 99.6

# Aggregated geometric-mean speedups from composite_method_tradeoff_v2 caption.
# These are cross-dataset geo-means, not per-dataset values.
_FALLBACK_SPEEDUP_NODES = [3, 5, 7, 9, 11]
_FALLBACK_SPEEDUP_VALUES = [48.0, 120.0, 500.0, 3000.0, 14108.0]

_FALLBACK_RHO = 0.691
_FALLBACK_BETA_GED = 0.80


# =========================================================================
# Data loading
# =========================================================================


def _load_message_length_data(
    raw_dir: str,
    method: str = _TARGET_METHOD,
) -> pd.DataFrame | None:
    """Load message length CSVs for a given method.

    Args:
        raw_dir: Path to message_length/raw/ directory.
        method: Encoding method to filter.

    Returns:
        DataFrame with columns isal_uniform_bits, ged_standard_bits, or None.
    """
    pattern = os.path.join(raw_dir, "message_lengths_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        logger.warning("No message length CSVs found in %s", raw_dir)
        return None

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception:
            logger.warning("Failed to read %s", f)

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    filtered = combined[combined["method"] == method].copy()
    if filtered.empty:
        logger.warning("No data for method=%s, using all methods", method)
        filtered = combined.copy()

    return filtered


def _compute_ols(
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """Compute OLS regression statistics.

    Args:
        x: Independent variable (GED bits).
        y: Dependent variable (IsalGraph bits).

    Returns:
        Dict with keys: beta, intercept, r2, pct_wins, n.
    """
    slope, intercept, r, _p, _se = sp_stats.linregress(x, y)
    wins = float((y < x).sum()) / len(y) * 100
    return {
        "beta": slope,
        "intercept": intercept,
        "r2": r**2,
        "pct_wins": wins,
        "n": len(x),
    }


def _load_speedup_data(
    comp_dir: str,
) -> tuple[list[int], list[float]] | None:
    """Load binned geometric-mean speedups for Greedy-rnd from timing data.

    Attempts to replicate the binning from composite_method_tradeoff_v2.
    Falls back to None if data structure doesn't match expectations.

    Args:
        comp_dir: Path to computational/ directory (with raw/ and stats/).

    Returns:
        (node_bins, speedup_values) or None if data unavailable.
    """
    stats_dir = os.path.join(comp_dir, "stats")
    summary_path = os.path.join(stats_dir, "summary.json")
    if not os.path.isfile(summary_path):
        logger.warning("Computational summary not found: %s", summary_path)
        return None

    try:
        with open(summary_path) as f:
            data = json.load(f)
    except Exception:
        logger.warning("Failed to read computational summary")
        return None

    # Extract per-dataset per-bin speedups and compute geometric mean
    per_dataset = data.get("per_dataset", {})
    if not per_dataset:
        return None

    # Collect speedups per bin label across datasets
    bin_speedups: dict[str, list[float]] = {}
    for ds_data in per_dataset.values():
        bins = ds_data.get("crossover", {}).get("bins", [])
        for b in bins:
            sp = b.get("speedup")
            label = b.get("bin", b.get("bin_label", ""))
            if sp is not None and sp > 0:
                bin_speedups.setdefault(label, []).append(sp)

    if not bin_speedups:
        return None

    # Sort bins by their lower bound and compute geometric mean
    sorted_bins = sorted(
        bin_speedups.items(),
        key=lambda kv: int(kv[0].split("-")[0]) if "-" in kv[0] else 0,
    )
    nodes = []
    values = []
    for label, speeds in sorted_bins:
        parts = label.split("-")
        if len(parts) == 2:
            mid = (int(parts[0]) + int(parts[1])) / 2
        else:
            mid = float(parts[0])
        nodes.append(int(mid))
        geo_mean = float(np.exp(np.mean(np.log(speeds))))
        values.append(geo_mean)

    if len(nodes) < 2:
        return None

    logger.info("Loaded speedup from summary: %s", list(zip(nodes, values)))
    return nodes, values


# =========================================================================
# Sub-panel drawing functions
# =========================================================================


def _draw_b1_message_length(
    ax: plt.Axes,
    run_dir: str | None = None,
    *,
    stats_outside: bool = False,
) -> None:
    """Draw B1: Message length comparison scatter/heatmap.

    Args:
        ax: Target axes.
        run_dir: Run directory root (contains message_length/raw/).
        stats_outside: If True, place stats text below the plot instead of inside.
    """
    df = None
    if run_dir:
        raw_dir = os.path.join(run_dir, "message_length", "raw")
        df = _load_message_length_data(raw_dir, method=_TARGET_METHOD)

    if df is not None and len(df) > 10:
        x = df["ged_standard_bits"].values
        y = df["isal_uniform_bits"].values
        stats = _compute_ols(x, y)
        beta = stats["beta"]
        r2 = stats["r2"]
        wins = stats["pct_wins"]
        logger.info(
            "Message length OLS (%s): beta=%.4f, R2=%.4f, Wins=%.1f%%",
            _TARGET_METHOD,
            beta,
            r2,
            wins,
        )

        # Shared axis range: same limits and ticks for x and y
        axis_max = max(x.max(), y.max()) * 1.05
        axis_min = 0
        n_bins = 50

        # 2D density heatmap with log-scale coloring (matching paper figure)
        h, xedges, yedges = np.histogram2d(
            x,
            y,
            bins=n_bins,
            range=[[axis_min, axis_max], [axis_min, axis_max]],
        )
        # Log1p transform for color scale; show all bins including 0
        h_log = np.log1p(h.T)

        ax.pcolormesh(
            xedges,
            yedges,
            h_log,
            cmap="viridis",
            rasterized=True,
        )

        # Parity line y=x (dashed white for visibility on viridis)
        ax.plot(
            [axis_min, axis_max],
            [axis_min, axis_max],
            "--",
            color="white",
            linewidth=0.8,
            alpha=0.85,
            zorder=3,
        )

        # OLS regression line
        x_fit = np.array([axis_min, axis_max])
        y_fit = beta * x_fit + stats["intercept"]
        ax.plot(
            x_fit,
            y_fit,
            "-",
            color=PAUL_TOL_BRIGHT["red"],
            linewidth=1.0,
            zorder=4,
        )

        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)

        # Same ticks on both axes so x=y diagonal is visually meaningful
        tick_step = 50
        shared_ticks = np.arange(0, axis_max + 1, tick_step)
        ax.set_xticks(shared_ticks)
        ax.set_yticks(shared_ticks)
    else:
        # Schematic fallback
        beta = _FALLBACK_MSG_BETA
        r2 = _FALLBACK_MSG_R2
        wins = _FALLBACK_MSG_WINS
        logger.info("Using fallback message length values")

        ax.plot([0, 200], [0, 200], "--", color="0.6", linewidth=0.7)
        ax.plot(
            [0, 200],
            [0, 200 * beta],
            "-",
            color=PAUL_TOL_BRIGHT["red"],
            linewidth=1.0,
        )
        ax.fill_between(
            [0, 200],
            [0, 200],
            [0, 200 * beta],
            alpha=0.10,
            color=PAUL_TOL_BRIGHT["green"],
        )
        ax.text(
            140,
            85,
            "IsalGraph\nmore compact",
            fontsize=5,
            ha="center",
            va="center",
            color=PAUL_TOL_BRIGHT["green"],
            fontstyle="italic",
        )
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 200)

    # Stats annotation
    stats_text = (
        rf"$\beta = {beta:.3f}$" + "\n" + rf"$R^2 = {r2:.3f}$" + "\n" + rf"Wins: ${wins:.1f}\%$"
    )
    if stats_outside:
        # Place below the plot, outside axes
        ax.text(
            0.98,
            -0.22,
            stats_text,
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="right",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.9,
                edgecolor="0.7",
                linewidth=0.5,
            ),
        )
    else:
        # Place inside the upper-left corner (for composite view)
        ax.text(
            0.04,
            0.96,
            stats_text,
            transform=ax.transAxes,
            fontsize=6,
            va="top",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.85,
                edgecolor="0.7",
                linewidth=0.5,
            ),
        )

    ax.set_xlabel("GED bits", fontsize=7, labelpad=2)
    ax.set_ylabel("IsalGraph bits", fontsize=7, labelpad=2)
    ax.tick_params(labelsize=5.5)
    ax.set_aspect("equal")


def _draw_b2_speedup(
    ax: plt.Axes,
    run_dir: str | None = None,
) -> None:
    """Draw B2: Computational speedup line chart.

    Args:
        ax: Target axes.
        run_dir: Run directory root (contains computational/).
    """
    # Always use fallback: the summary.json stores per-dataset speedups,
    # not the cross-dataset geometric means shown in composite_method_tradeoff_v2.
    # Replicating the full binning pipeline is out of scope for the abstract.
    nodes = list(_FALLBACK_SPEEDUP_NODES)
    speedup = list(_FALLBACK_SPEEDUP_VALUES)
    logger.info("Using caption speedup values for Greedy-rnd(v0)")

    color = PAUL_TOL_BRIGHT["green"]  # "#228833" for Greedy-rnd

    ax.semilogy(
        nodes,
        speedup,
        "-",
        color=color,
        linewidth=1.2,
        marker="^",
        markersize=4,
        markeredgecolor="0.3",
        markeredgewidth=0.4,
        zorder=3,
    )

    # Fill area between line and y=1
    ax.fill_between(
        nodes,
        1,
        speedup,
        alpha=0.10,
        color=color,
    )

    # Breakeven line
    ax.axhline(
        1,
        color="0.6",
        linewidth=0.5,
        linestyle="--",
    )

    # Annotate extreme values
    min_idx = 0
    max_idx = len(speedup) - 1
    ax.annotate(
        f"{speedup[min_idx]:.0f}" + r"$\times$",
        xy=(nodes[min_idx], speedup[min_idx]),
        xytext=(8, -2),
        textcoords="offset points",
        fontsize=5,
        ha="left",
        va="center",
        color=color,
        fontweight="bold",
    )
    ax.annotate(
        f"{speedup[max_idx]:,.0f}" + r"$\times$",
        xy=(nodes[max_idx], speedup[max_idx]),
        xytext=(-10, -8),
        textcoords="offset points",
        fontsize=5,
        ha="right",
        va="top",
        color=color,
        fontweight="bold",
    )

    ax.set_xlabel("Nodes $n$", fontsize=7, labelpad=2)
    ax.set_ylabel("Speedup", fontsize=7, labelpad=2)
    ax.tick_params(labelsize=5.5)

    if len(nodes) > 1:
        ax.set_xticks(nodes)
        ax.set_xticklabels([str(n) for n in nodes], fontsize=5)

    # Extend y-axis to fit annotation at top
    ax.set_ylim(top=max(speedup) * 3.0)

    # Title
    ax.set_title(
        r"Greedy-rnd($v_0$) speedup",
        fontsize=6.5,
        pad=3,
    )

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_b3_ged_proxy(
    ax: plt.Axes,
    rho: float = _FALLBACK_RHO,
    beta_ged: float = _FALLBACK_BETA_GED,
) -> None:
    """Draw B3: GED proxy quality formula block.

    Args:
        ax: Target axes.
        rho: Spearman correlation value.
        beta_ged: OLS slope value.
    """
    ax.axis("off")

    # Title
    ax.text(
        0.5,
        0.92,
        "GED proxy quality",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.5,
        fontweight="bold",
        color="0.3",
    )

    # Main formula
    ax.text(
        0.5,
        0.62,
        r"$d_{\mathrm{Lev}}(w^*_A, w^*_B) \approx d_{\mathrm{GED}}(G_A, G_B)$",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.5,
    )

    # Separator line (use plot in axes coords since axhline rejects transform)
    ax.plot(
        [0.1, 0.9],
        [0.47, 0.47],
        color="0.7",
        linewidth=0.5,
        transform=ax.transAxes,
        clip_on=False,
    )

    # Statistics
    stats_text = rf"Spearman $\rho = {rho:.3f}$" + "\n" + rf"OLS slope $\beta = {beta_ged:.2f}$"
    ax.text(
        0.5,
        0.25,
        stats_text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.5,
        color="0.4",
        linespacing=1.5,
    )

    # Light border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.8")
        spine.set_linewidth(0.5)


# =========================================================================
# Main panel generation
# =========================================================================


def draw_panel_b(
    fig: plt.Figure,
    gs: object,
    run_dir: str | None = None,
) -> None:
    """Draw Panel B (results) into the given figure/gridspec region.

    Args:
        fig: Matplotlib figure.
        gs: GridSpec or SubplotSpec to use for axes placement.
        run_dir: Run directory root for loading data.
    """
    inner = GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=gs,
        height_ratios=[1.1, 1.0],
        width_ratios=[1.1, 0.9],
        hspace=0.65,
        wspace=0.40,
    )

    ax_msg = fig.add_subplot(inner[0, :])  # B1: full width
    ax_speed = fig.add_subplot(inner[1, 0])  # B2: bottom-left
    ax_corr = fig.add_subplot(inner[1, 1])  # B3: bottom-right

    _draw_b1_message_length(ax_msg, run_dir=run_dir)
    _draw_b2_speedup(ax_speed, run_dir=run_dir)
    _draw_b3_ged_proxy(ax_corr)


def generate_panel_b(
    output_dir: str,
    run_dir: str | None = None,
) -> str:
    """Generate Panel B as a standalone figure.

    Args:
        output_dir: Directory to save output files.
        run_dir: Run directory root for loading data.

    Returns:
        Base path of saved figure (without extension).
    """
    fig = plt.figure(figsize=(2.8, 1.77))
    gs = GridSpec(1, 1, figure=fig, left=0.12, right=0.97, top=0.96, bottom=0.08)
    draw_panel_b(fig, gs[0], run_dir=run_dir)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "panel_b_results")
    save_figure(fig, path, formats=("pdf", "svg", "png"))
    plt.close(fig)
    logger.info("Panel B saved: %s", path)
    return path


# =========================================================================
# Individual sub-panel exports (for Inkscape workflow)
# =========================================================================

_FORMATS = ("pdf", "svg", "png")


def generate_b1_standalone(
    output_dir: str,
    run_dir: str | None = None,
) -> str:
    """Generate B1 (message length scatter) as a standalone figure.

    Stats box is placed outside the plot area for cleaner heatmap.
    """
    fig, ax = plt.subplots(figsize=(3.0, 2.8))
    _draw_b1_message_length(ax, run_dir=run_dir, stats_outside=True)
    fig.subplots_adjust(bottom=0.22, left=0.16, right=0.96, top=0.96)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "b1_message_length")
    save_figure(fig, path, formats=_FORMATS)
    plt.close(fig)
    logger.info("B1 (message length) saved: %s", path)
    return path


def generate_b2_standalone(
    output_dir: str,
    run_dir: str | None = None,
) -> str:
    """Generate B2 (speedup chart) as a standalone figure."""
    fig, ax = plt.subplots(figsize=(2.5, 1.8))
    _draw_b2_speedup(ax, run_dir=run_dir)
    fig.subplots_adjust(bottom=0.18, left=0.18, right=0.95, top=0.90)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "b2_speedup")
    save_figure(fig, path, formats=_FORMATS)
    plt.close(fig)
    logger.info("B2 (speedup) saved: %s", path)
    return path


def generate_b3_standalone(output_dir: str) -> str:
    """Generate B3 (GED proxy formula) as a standalone figure."""
    fig, ax = plt.subplots(figsize=(2.2, 1.4))
    _draw_b3_ged_proxy(ax)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "b3_ged_proxy")
    save_figure(fig, path, formats=_FORMATS)
    plt.close(fig)
    logger.info("B3 (GED proxy) saved: %s", path)
    return path


def generate_all_standalone(
    output_dir: str,
    run_dir: str | None = None,
) -> list[str]:
    """Generate all three sub-panels as individual files."""
    paths = [
        generate_b1_standalone(output_dir, run_dir=run_dir),
        generate_b2_standalone(output_dir, run_dir=run_dir),
        generate_b3_standalone(output_dir),
    ]
    return paths


# =========================================================================
# CLI
# =========================================================================


def main() -> None:
    """CLI entry point for Panel B generation."""
    parser = argparse.ArgumentParser(
        description="Generate Panel B (results) for graphical abstract."
    )
    parser.add_argument(
        "--output-dir",
        default="paper_figures/graphical_abstract",
        help="Output directory.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Pipeline run directory (contains message_length/, computational/).",
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="Also generate individual sub-panel files.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    apply_ieee_style()
    generate_panel_b(args.output_dir, run_dir=args.run_dir)
    if args.individual:
        generate_all_standalone(args.output_dir, run_dir=args.run_dir)


if __name__ == "__main__":
    main()
