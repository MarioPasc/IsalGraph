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


# ---------------------------------------------------------------------------
# Scatter figure: IsalGraph bits vs GED bits
# ---------------------------------------------------------------------------


def generate_scatter_figure(
    raw_dir: str,
    output_dir: str,
) -> str:
    """Generate fig_message_length_scatter.pdf — 1xK panels.

    Panels: one per available method.
    X: GED bits (generous), Y: IsalGraph bits (uniform).
    Points below y=x mean IsalGraph is more compact.

    Returns:
        Path to the saved figure.
    """
    apply_ieee_style()
    df = _load_message_length_csvs(raw_dir)
    if df.empty:
        logger.warning("No data for scatter figure")
        return ""

    methods = sorted(df["method"].unique())
    n_panels = len(methods)
    if n_panels == 0:
        return ""

    fig, axes = plt.subplots(1, n_panels, figsize=(7.0, 3.0), squeeze=False)
    axes = axes[0]

    for ax_idx, method in enumerate(methods):
        ax = axes[ax_idx]
        mdf = df[df["method"] == method]

        for dataset, color in DATASET_COLORS.items():
            ddf = mdf[mdf["dataset"] == dataset]
            if ddf.empty:
                continue
            ax.scatter(
                ddf["ged_generous_bits"],
                ddf["isal_uniform_bits"],
                c=color,
                s=12,
                alpha=0.6,
                label=DATASET_DISPLAY.get(dataset, dataset),
                edgecolors="none",
            )

        # Identity line y=x
        all_vals = pd.concat([mdf["ged_generous_bits"], mdf["isal_uniform_bits"]])
        if not all_vals.empty:
            hi = all_vals.max() * 1.05
            ax.plot([0, hi], [0, hi], "--", color="gray", lw=0.8, alpha=0.7, zorder=0)
            ax.set_xlim(0, hi)
            ax.set_ylim(0, hi)

        # Annotation box
        valid = mdf[np.isfinite(mdf["ratio_uniform_generous"])]
        if not valid.empty:
            mean_ratio = valid["ratio_uniform_generous"].mean()
            n_wins = (valid["isal_uniform_bits"] < valid["ged_generous_bits"]).sum()
            pct_wins = 100.0 * n_wins / len(valid)
            text = f"Mean ratio: {mean_ratio:.2f}\nIsalGraph wins: {pct_wins:.0f}%"
            ax.text(
                0.05,
                0.95,
                text,
                transform=ax.transAxes,
                fontsize=6,
                va="top",
                bbox={"boxstyle": "round,pad=0.3", "fc": "wheat", "alpha": 0.8},
            )

        label = METHOD_DISPLAY.get(method, method)
        ax.set_title(f"({chr(ord('a') + ax_idx)}) {label}", fontsize=8)
        ax.set_xlabel("GED bits (generous)", fontsize=7)
        if ax_idx == 0:
            ax.set_ylabel("IsalGraph bits (uniform)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_aspect("equal", adjustable="box")

    if n_panels > 0:
        axes[-1].legend(fontsize=5, loc="lower right", framealpha=0.8)

    fig.tight_layout()
    out_path = os.path.join(output_dir, "fig_message_length_scatter")
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
    """Generate fig_message_length_ratio.pdf — single panel.

    X: number of nodes N, Y: compression ratio (GED/IsalGraph).
    Horizontal line at ratio=1 (break-even).

    Returns:
        Path to the saved figure.
    """
    apply_ieee_style()
    df = _load_message_length_csvs(raw_dir)
    if df.empty:
        logger.warning("No data for ratio figure")
        return ""

    fig, ax = plt.subplots(1, 1, figsize=(3.39, 2.55))

    all_x, all_y = [], []

    for dataset, color in DATASET_COLORS.items():
        ddf = df[(df["dataset"] == dataset) & np.isfinite(df["ratio_uniform_generous"])]
        if ddf.empty:
            continue
        x = ddf["n_nodes"].values
        y = ddf["ratio_uniform_generous"].values
        all_x.extend(x)
        all_y.extend(y)
        ax.scatter(
            x,
            y,
            c=color,
            s=10,
            alpha=0.5,
            label=DATASET_DISPLAY.get(dataset, dataset),
            edgecolors="none",
        )

    ax.axhline(y=1.0, color="gray", lw=0.8, ls="--", alpha=0.7, zorder=0)

    # Linear trend
    if len(all_x) > 2:
        x_arr = np.array(all_x, dtype=float)
        y_arr = np.array(all_y, dtype=float)
        mask = np.isfinite(y_arr)
        if mask.sum() > 2:
            coeffs = np.polyfit(x_arr[mask], y_arr[mask], 1)
            x_line = np.linspace(x_arr[mask].min(), x_arr[mask].max(), 100)
            y_line = np.polyval(coeffs, x_line)
            ax.plot(x_line, y_line, "-", color="black", lw=1.0, alpha=0.7, label="Trend")

    ax.set_xlabel("Number of nodes $N$", fontsize=8)
    ax.set_ylabel("Compression ratio (GED / IsalGraph)", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5, loc="best", framealpha=0.8)

    fig.tight_layout()
    out_path = os.path.join(output_dir, "fig_message_length_ratio")
    save_figure(fig, out_path)
    plt.close(fig)
    logger.info("  -> %s.pdf", out_path)
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
    generate_ratio_figure(args.data_dir, args.output_dir)

    if args.stats_dir:
        generate_message_length_table(args.stats_dir, args.output_dir)


if __name__ == "__main__":
    main()
