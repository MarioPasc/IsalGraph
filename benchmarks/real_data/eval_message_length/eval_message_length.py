# ruff: noqa: N803, N806
"""CLI orchestrator for message length analysis.

Computes IsalGraph vs GED message lengths for all (dataset, method) pairs
using precomputed canonical strings and graph metadata from Step 1.
Demonstrates that IsalGraph strings are dramatically more compact than GED
operation sequences for encoding graphs.

Usage:
    python -m benchmarks.eval_message_length.eval_message_length \
        --data-root data/eval \
        --output-dir results/eval_message_length \
        --datasets all --methods auto \
        --csv --plot --table
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_DATA_ROOT = "/media/mpascual/Sandisk2TB/research/isalgraph/data/eval"
DEFAULT_OUTPUT_DIR = "/media/mpascual/Sandisk2TB/research/isalgraph/results/eval_message_length"

ALL_DATASETS = ["iam_letter_low", "iam_letter_med", "iam_letter_high", "linux", "aids"]
DEFAULT_METHODS = ["exhaustive", "pruned_exhaustive", "greedy", "greedy_single"]

DATASET_DISPLAY: dict[str, str] = {
    "iam_letter_low": "IAM Letter LOW",
    "iam_letter_med": "IAM Letter MED",
    "iam_letter_high": "IAM Letter HIGH",
    "linux": "LINUX",
    "aids": "AIDS",
}

GED_SCHEMES = ("generous", "standard", "full")
ISAL_SCHEMES = ("uniform", "entropy")


# =============================================================================
# Data loading
# =============================================================================


def _load_canonical_strings(data_root: str, dataset: str, method: str) -> dict[str, dict] | None:
    """Load canonical strings JSON for a (dataset, method) pair.

    Returns:
        Dict mapping graph_id -> {"string": str, "length": int, ...},
        or None if file not found.
    """
    path = os.path.join(data_root, "canonical_strings", f"{dataset}_{method}.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("strings", {})


def _load_graph_metadata(data_root: str, dataset: str) -> dict | None:
    """Load graph metadata JSON for a dataset.

    Returns:
        Dict with keys: graph_ids, node_counts, edge_counts, n_graphs.
        Or None if file not found.
    """
    path = os.path.join(data_root, "graph_metadata", f"{dataset}.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _detect_methods(data_root: str, dataset: str) -> list[str]:
    """Auto-detect available methods for a dataset."""
    cs_dir = os.path.join(data_root, "canonical_strings")
    if not os.path.isdir(cs_dir):
        return []
    found = []
    for method in DEFAULT_METHODS:
        path = os.path.join(cs_dir, f"{dataset}_{method}.json")
        if os.path.isfile(path):
            found.append(method)
    return found


# =============================================================================
# Per-graph message length computation
# =============================================================================


def compute_message_lengths_for_dataset(
    data_root: str,
    dataset: str,
    method: str,
) -> list[dict]:
    """Compute message lengths for all graphs in a (dataset, method) pair.

    Loads canonical strings and graph metadata, joins on graph_id, and
    computes IsalGraph and GED message lengths for each graph.

    Returns:
        List of dicts with per-graph message length data.
    """
    from benchmarks.eval_message_length.message_length_computer import (
        combinatorial_lower_bound,
        compression_ratio,
        ged_construction_message_length,
        isalgraph_message_length_entropy,
        isalgraph_message_length_uniform,
    )

    strings_dict = _load_canonical_strings(data_root, dataset, method)
    if strings_dict is None:
        logger.warning("No canonical strings for %s/%s", dataset, method)
        return []

    metadata = _load_graph_metadata(data_root, dataset)
    if metadata is None:
        logger.warning("No graph metadata for %s", dataset)
        return []

    # Build lookup: graph_id -> (n_nodes, n_edges)
    graph_ids = metadata.get("graph_ids", [])
    node_counts = metadata.get("node_counts", [])
    edge_counts = metadata.get("edge_counts", [])
    meta_lookup: dict[str, tuple[int, int]] = {}
    for gid, nc, ec in zip(graph_ids, node_counts, edge_counts, strict=False):
        meta_lookup[gid] = (nc, ec)

    rows: list[dict] = []
    skipped = 0

    for graph_id, entry in strings_dict.items():
        string = entry.get("string")
        if string is None:
            skipped += 1
            continue

        if graph_id not in meta_lookup:
            skipped += 1
            continue

        n_nodes, n_edges = meta_lookup[graph_id]
        string_length = len(string)

        if n_nodes <= 1:
            skipped += 1
            continue

        # IsalGraph message lengths
        isal_uniform = isalgraph_message_length_uniform(string_length)
        isal_entropy = isalgraph_message_length_entropy(string)

        # GED message lengths (all 3 schemes)
        ged_generous = ged_construction_message_length(n_nodes, n_edges, "generous")
        ged_standard = ged_construction_message_length(n_nodes, n_edges, "standard")
        ged_full = ged_construction_message_length(n_nodes, n_edges, "full")

        # Combinatorial lower bound
        comb_lb = combinatorial_lower_bound(n_nodes, n_edges)

        # Compression ratios
        ratio_uniform_generous = compression_ratio(isal_uniform, ged_generous)
        ratio_uniform_standard = compression_ratio(isal_uniform, ged_standard)
        ratio_entropy_generous = compression_ratio(isal_entropy, ged_generous)

        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "graph_id": graph_id,
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "string_length": string_length,
                "isal_uniform_bits": isal_uniform,
                "isal_entropy_bits": isal_entropy,
                "ged_generous_bits": ged_generous,
                "ged_standard_bits": ged_standard,
                "ged_full_bits": ged_full,
                "combinatorial_lb_bits": comb_lb,
                "ratio_uniform_generous": ratio_uniform_generous,
                "ratio_uniform_standard": ratio_uniform_standard,
                "ratio_entropy_generous": ratio_entropy_generous,
            }
        )

    if skipped > 0:
        logger.info("  %s/%s: %d graphs computed, %d skipped", dataset, method, len(rows), skipped)
    else:
        logger.info("  %s/%s: %d graphs computed", dataset, method, len(rows))

    return rows


# =============================================================================
# Full pipeline
# =============================================================================


def compute_all_message_lengths(
    data_root: str,
    datasets: list[str],
    methods: list[str],
    output_dir: str,
    *,
    save_csv: bool = True,
    save_plot: bool = True,
    save_table: bool = True,
) -> None:
    """Compute message lengths for all (dataset, method) pairs.

    Saves:
        - CSV per (dataset, method): message_lengths_{dataset}_{method}.csv
        - Summary JSON: message_length_summary.json
        - Figures (if --plot): scatter + ratio plots
        - LaTeX tables (if --table)
    """
    import csv

    os.makedirs(output_dir, exist_ok=True)
    raw_dir = os.path.join(output_dir, "raw")
    stats_dir = os.path.join(output_dir, "stats")
    figures_dir = os.path.join(output_dir, "figures")
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    all_rows: list[dict] = []
    summary: dict[str, dict] = {}

    for dataset in datasets:
        ds_methods = _detect_methods(data_root, dataset) if methods == ["auto"] else methods

        for method in ds_methods:
            logger.info("Processing %s / %s ...", dataset, method)
            rows = compute_message_lengths_for_dataset(data_root, dataset, method)

            if not rows:
                continue

            all_rows.extend(rows)

            # Save per-(dataset, method) CSV
            if save_csv:
                csv_path = os.path.join(raw_dir, f"message_lengths_{dataset}_{method}.csv")
                fieldnames = list(rows[0].keys())
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                logger.info("  Saved: %s (%d rows)", csv_path, len(rows))

            # Compute summary statistics
            key = f"{dataset}_{method}"
            n_graphs = len(rows)
            isal_bits = [r["isal_uniform_bits"] for r in rows]
            ged_bits = [r["ged_generous_bits"] for r in rows]
            ratios = [
                r["ratio_uniform_generous"]
                for r in rows
                if math.isfinite(r["ratio_uniform_generous"])
            ]
            n_isal_wins = sum(1 for r in rows if r["isal_uniform_bits"] < r["ged_generous_bits"])

            summary[key] = {
                "dataset": dataset,
                "method": method,
                "n_graphs": n_graphs,
                "mean_n_nodes": _mean([r["n_nodes"] for r in rows]),
                "mean_n_edges": _mean([r["n_edges"] for r in rows]),
                "mean_string_length": _mean([r["string_length"] for r in rows]),
                "mean_isal_uniform_bits": _mean(isal_bits),
                "mean_isal_entropy_bits": _mean([r["isal_entropy_bits"] for r in rows]),
                "mean_ged_generous_bits": _mean(ged_bits),
                "mean_ged_standard_bits": _mean([r["ged_standard_bits"] for r in rows]),
                "mean_ged_full_bits": _mean([r["ged_full_bits"] for r in rows]),
                "mean_ratio_uniform_generous": _mean(ratios) if ratios else float("nan"),
                "pct_isalgraph_wins": 100.0 * n_isal_wins / n_graphs if n_graphs > 0 else 0.0,
                "n_isal_wins": n_isal_wins,
            }

    # Save summary JSON
    summary_path = os.path.join(stats_dir, "message_length_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_safe)
    logger.info("Summary: %s", summary_path)

    # Print summary table
    _print_summary(summary)

    # Generate figures
    if save_plot and all_rows:
        os.makedirs(figures_dir, exist_ok=True)
        _generate_figures(all_rows, summary, figures_dir)

    # Generate LaTeX tables
    if save_table and summary:
        os.makedirs(tables_dir, exist_ok=True)
        _generate_tables(summary, tables_dir)


# =============================================================================
# Summary helpers
# =============================================================================


def _mean(values: list) -> float:
    """Compute mean of a list, returning NaN for empty lists."""
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _json_safe(obj: object) -> object:
    """JSON serialization fallback for NaN/Inf."""
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return "Inf" if obj > 0 else "-Inf"
    return str(obj)


def _print_summary(summary: dict[str, dict]) -> None:
    """Print a formatted summary table to the logger."""
    logger.info("")
    logger.info("=" * 90)
    logger.info("MESSAGE LENGTH SUMMARY")
    logger.info("=" * 90)
    header = (
        f"{'Dataset':<20} {'Method':<18} {'N':>5} "
        f"{'IsalBits':>10} {'GEDBits':>10} {'Ratio':>7} {'%Wins':>7}"
    )
    logger.info(header)
    logger.info("-" * 90)

    for _key, s in summary.items():
        line = (
            f"{s['dataset']:<20} {s['method']:<18} {s['n_graphs']:>5} "
            f"{s['mean_isal_uniform_bits']:>10.1f} {s['mean_ged_generous_bits']:>10.1f} "
            f"{s['mean_ratio_uniform_generous']:>7.2f} {s['pct_isalgraph_wins']:>6.1f}%"
        )
        logger.info(line)
    logger.info("=" * 90)


# =============================================================================
# Figure generation
# =============================================================================


def _generate_figures(
    all_rows: list[dict],
    summary: dict[str, dict],
    figures_dir: str,
) -> None:
    """Generate publication-quality figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from benchmarks.plotting_styles import (
        PAUL_TOL_BRIGHT,
        PLOT_SETTINGS,
        apply_ieee_style,
    )

    apply_ieee_style()

    # --- Primary figure: scatter (1x3 panels) ---
    _generate_scatter_figure(all_rows, figures_dir, plt, PAUL_TOL_BRIGHT, PLOT_SETTINGS)

    # --- Secondary figure: ratio vs N ---
    _generate_ratio_figure(all_rows, figures_dir, plt, PAUL_TOL_BRIGHT, PLOT_SETTINGS)

    logger.info("Figures saved to %s", figures_dir)


def _generate_scatter_figure(
    all_rows: list[dict],
    figures_dir: str,
    plt: object,
    colors: dict,
    settings: dict,
) -> None:
    """fig_message_length_scatter.pdf — 1x3 panels.

    Panels: (a) Exhaustive/Canonical, (b) Greedy-Min, (c) Greedy-rnd v_0
    X: GED bits (generous), Y: IsalGraph bits (uniform)
    """
    method_groups = {
        "exhaustive": "Canonical",
        "pruned_exhaustive": "Canonical (Pruned)",
        "greedy": "Greedy-Min",
        "greedy_single": r"Greedy-rnd($v_0$)",
    }

    # Filter to methods with data
    available_methods = sorted({r["method"] for r in all_rows})
    plot_methods = [m for m in method_groups if m in available_methods]

    if not plot_methods:
        logger.warning("No methods available for scatter plot")
        return

    n_panels = len(plot_methods)
    fig, axes = plt.subplots(1, n_panels, figsize=(7.0, 3.0), squeeze=False)
    axes = axes[0]

    dataset_colors = {
        "iam_letter_low": colors["blue"],
        "iam_letter_med": colors["cyan"],
        "iam_letter_high": colors["purple"],
        "linux": colors["red"],
        "aids": colors["green"],
    }

    for ax_idx, method in enumerate(plot_methods):
        ax = axes[ax_idx]
        method_rows = [r for r in all_rows if r["method"] == method]

        # Scatter by dataset
        for dataset, color in dataset_colors.items():
            ds_rows = [r for r in method_rows if r["dataset"] == dataset]
            if not ds_rows:
                continue
            x = [r["ged_generous_bits"] for r in ds_rows]
            y = [r["isal_uniform_bits"] for r in ds_rows]
            label = DATASET_DISPLAY.get(dataset, dataset)
            ax.scatter(x, y, c=color, s=12, alpha=0.6, label=label, edgecolors="none")

        # Identity line
        all_vals = [r["ged_generous_bits"] for r in method_rows] + [
            r["isal_uniform_bits"] for r in method_rows
        ]
        if all_vals:
            lo, hi = 0, max(all_vals) * 1.05
            ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=0.8, alpha=0.7, zorder=0)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

        # Annotation box
        ratios = [
            r["ratio_uniform_generous"]
            for r in method_rows
            if math.isfinite(r["ratio_uniform_generous"])
        ]
        n_wins = sum(1 for r in method_rows if r["isal_uniform_bits"] < r["ged_generous_bits"])
        n_total = len(method_rows)
        if ratios and n_total > 0:
            mean_ratio = sum(ratios) / len(ratios)
            pct_wins = 100.0 * n_wins / n_total
            text = f"Mean ratio: {mean_ratio:.2f}\nIsalGraph wins: {pct_wins:.0f}%"
            ax.text(
                0.05,
                0.95,
                text,
                transform=ax.transAxes,
                fontsize=6,
                verticalalignment="top",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.8},
            )

        label_text = method_groups.get(method, method)
        panel_letter = chr(ord("a") + ax_idx)
        ax.set_title(f"({panel_letter}) {label_text}", fontsize=8)
        ax.set_xlabel("GED bits (generous)", fontsize=7)
        if ax_idx == 0:
            ax.set_ylabel("IsalGraph bits (uniform)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_aspect("equal", adjustable="box")

    # Legend on last panel
    if n_panels > 0:
        axes[-1].legend(fontsize=5, loc="lower right", framealpha=0.8)

    fig.tight_layout()
    from benchmarks.plotting_styles import save_figure

    save_figure(fig, os.path.join(figures_dir, "fig_message_length_scatter"))
    plt.close(fig)
    logger.info("  -> fig_message_length_scatter.pdf")


def _generate_ratio_figure(
    all_rows: list[dict],
    figures_dir: str,
    plt: object,
    colors: dict,
    settings: dict,
) -> None:
    """fig_message_length_ratio.pdf — ratio vs N scatter + trend.

    X: number of nodes N, Y: compression ratio (GED/IsalGraph)
    """
    fig, ax = plt.subplots(1, 1, figsize=(3.39, 2.55))

    dataset_colors = {
        "iam_letter_low": colors["blue"],
        "iam_letter_med": colors["cyan"],
        "iam_letter_high": colors["purple"],
        "linux": colors["red"],
        "aids": colors["green"],
    }

    all_x: list[float] = []
    all_y: list[float] = []

    for dataset, color in dataset_colors.items():
        ds_rows = [
            r
            for r in all_rows
            if r["dataset"] == dataset and math.isfinite(r["ratio_uniform_generous"])
        ]
        if not ds_rows:
            continue
        x = [r["n_nodes"] for r in ds_rows]
        y = [r["ratio_uniform_generous"] for r in ds_rows]
        all_x.extend(x)
        all_y.extend(y)
        label = DATASET_DISPLAY.get(dataset, dataset)
        ax.scatter(x, y, c=color, s=10, alpha=0.5, label=label, edgecolors="none")

    # Horizontal line at ratio = 1
    ax.axhline(y=1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.7, zorder=0)

    # Linear regression trend
    if len(all_x) > 2:
        import numpy as np

        x_arr = np.array(all_x, dtype=float)
        y_arr = np.array(all_y, dtype=float)
        mask = np.isfinite(y_arr)
        if mask.sum() > 2:
            coeffs = np.polyfit(x_arr[mask], y_arr[mask], 1)
            x_line = np.linspace(x_arr[mask].min(), x_arr[mask].max(), 100)
            y_line = np.polyval(coeffs, x_line)
            ax.plot(x_line, y_line, "-", color="black", linewidth=1.0, alpha=0.7, label="Trend")

    ax.set_xlabel("Number of nodes $N$", fontsize=8)
    ax.set_ylabel("Compression ratio (GED / IsalGraph)", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5, loc="best", framealpha=0.8)

    fig.tight_layout()
    from benchmarks.plotting_styles import save_figure

    save_figure(fig, os.path.join(figures_dir, "fig_message_length_ratio"))
    plt.close(fig)
    logger.info("  -> fig_message_length_ratio.pdf")


# =============================================================================
# LaTeX table generation
# =============================================================================


def _generate_tables(summary: dict[str, dict], tables_dir: str) -> None:
    """Generate table_message_length_summary.tex."""
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
        method_display = _method_display(s["method"])
        line = (
            f"{ds_display} & {method_display} & "
            f"{s['mean_n_nodes']:.1f} & {s['mean_n_edges']:.1f} & "
            f"{s['mean_isal_uniform_bits']:.1f} & {s['mean_ged_generous_bits']:.1f} & "
            f"{s['mean_ratio_uniform_generous']:.2f} & "
            f"{s['pct_isalgraph_wins']:.1f}\\% \\\\"
        )
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table_path = os.path.join(tables_dir, "table_message_length_summary.tex")
    with open(table_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("  Table: %s", table_path)


def _method_display(method: str) -> str:
    """Human-readable method name for LaTeX."""
    mapping = {
        "exhaustive": "Canonical",
        "pruned_exhaustive": "Canonical (Pruned)",
        "greedy": "Greedy-Min",
        "greedy_single": r"Greedy-rnd($v_0$)",
    }
    return mapping.get(method, method)


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Message length analysis: IsalGraph vs GED encoding efficiency.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="Root directory for eval artifacts (canonical_strings/, graph_metadata/).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for results.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Comma-separated dataset names, or 'all'.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="auto",
        help="Comma-separated method names, or 'auto' to detect available.",
    )
    parser.add_argument("--csv", action="store_true", help="Save per-graph CSV files.")
    parser.add_argument("--plot", action="store_true", help="Generate figures.")
    parser.add_argument("--table", action="store_true", help="Generate LaTeX tables.")

    args = parser.parse_args()

    # Parse datasets
    if args.datasets == "all":
        datasets = ALL_DATASETS
    else:
        datasets = [d.strip() for d in args.datasets.split(",")]

    # Parse methods
    methods = ["auto"] if args.methods == "auto" else [m.strip() for m in args.methods.split(",")]

    logger.info("Message length analysis")
    logger.info("  Data root:  %s", args.data_root)
    logger.info("  Output dir: %s", args.output_dir)
    logger.info("  Datasets:   %s", datasets)
    logger.info("  Methods:    %s", methods)

    compute_all_message_lengths(
        data_root=args.data_root,
        datasets=datasets,
        methods=methods,
        output_dir=args.output_dir,
        save_csv=args.csv,
        save_plot=args.plot,
        save_table=args.table,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
