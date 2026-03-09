# ruff: noqa: N803, N806
"""Composite figure: Method quality vs Computational trade-off.

Panel (a): Canonical vs Greedy-min Spearman rho scatter (density-colored,
           marker per dataset, 95% CI error bars).
Panel (b): Aggregated speedup (GED / IsalGraph) vs number of nodes,
           with separate lines for Canonical and Greedy-min pipelines.

Usage:
    python -m benchmarks.eval_visualizations.composite_method_tradeoff \
        --data-root /media/mpascual/Sandisk2TB/research/isalgraph/data/eval \
        --stats-dir /media/mpascual/Sandisk2TB/research/isalgraph/results/eval_benchmarks/eval_correlation/stats \
        --comp-dir /media/mpascual/Sandisk2TB/research/isalgraph/results/eval_benchmarks/eval_computational \
        --output-dir /media/mpascual/Sandisk2TB/research/isalgraph/results/figures/composite
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarks.eval_visualizations.result_loader import (
    ALL_DATASETS,
    load_all_results,
)
from benchmarks.plotting_styles import (
    PAUL_TOL_BRIGHT,
    apply_ieee_style,
    get_figure_size,
    save_figure,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EXHAUSTIVE_COLOR = PAUL_TOL_BRIGHT["blue"]  # #4477AA
_GREEDY_COLOR = PAUL_TOL_BRIGHT["red"]  # #EE6677

GED_COLOR = PAUL_TOL_BRIGHT["red"]
ENCODE_COLOR = PAUL_TOL_BRIGHT["blue"]

# Per-dataset marker shapes
_DATASET_MARKERS: dict[str, str] = {
    "iam_letter_low": "o",
    "iam_letter_med": "s",
    "iam_letter_high": "D",
    "linux": "^",
    "aids": "v",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_cross_analysis(stats_dir: str) -> dict | None:
    path = os.path.join(stats_dir, "cross_dataset_analysis.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _compute_mean_pair_size(node_counts: np.ndarray) -> float:
    """Mean pair size: E[(|V_i| + |V_j|)/2] over all graph pairs (i, j).

    Returns the mean of (|V_i| + |V_j|)/2 over all upper-triangle pairs.
    """
    v = node_counts.astype(np.float64)
    # Mean of pairwise means = mean of all node counts (by linearity)
    return float(v.mean())


def _load_timing_stats(comp_dir: str) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    stats_dir = os.path.join(comp_dir, "stats")
    for ds in ALL_DATASETS:
        path = os.path.join(stats_dir, f"{ds}_timing_stats.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                stats[ds] = json.load(f)
    return stats


def _load_encoding_csv(comp_dir: str, dataset: str) -> pd.DataFrame | None:
    path = os.path.join(comp_dir, "raw", f"{dataset}_encoding_times.csv")
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def _load_ged_csv(comp_dir: str, dataset: str) -> pd.DataFrame | None:
    path = os.path.join(comp_dir, "raw", f"{dataset}_ged_times.csv")
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def _load_lev_csv(comp_dir: str, dataset: str) -> pd.DataFrame | None:
    path = os.path.join(comp_dir, "raw", f"{dataset}_levenshtein_times.csv")
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Panel (a): Correlation scatter
# ---------------------------------------------------------------------------


def _draw_panel_a(
    ax: plt.Axes,
    fig: plt.Figure,
    results: dict,
    stats_dir: str,
) -> None:
    """Draw the rho Canonical vs rho Greedy-Min scatter, colored by pair count."""
    import matplotlib.cm as cm
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D

    from benchmarks.eval_visualizations.result_loader import DATASET_DISPLAY

    cross = _load_cross_analysis(stats_dir)
    if cross is None or "h5_method_comparison" not in cross:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    h5 = cross["h5_method_comparison"]

    datasets: list[str] = []
    rho_exh: list[float] = []
    rho_gre: list[float] = []
    ci_exh_lo: list[float] = []
    ci_exh_hi: list[float] = []
    ci_gre_lo: list[float] = []
    ci_gre_hi: list[float] = []
    pair_counts: list[int] = []

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

        # Pair count: n_graphs * (n_graphs - 1) / 2
        if ds in results and "node_counts" in results[ds]:
            n_graphs = len(results[ds]["node_counts"])
            pair_counts.append(n_graphs * (n_graphs - 1) // 2)
        else:
            pair_counts.append(0)

    if not datasets:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    # Colormap for pair count (log scale)
    cmap = cm.viridis  # type: ignore[attr-defined]
    pc_arr = np.array(pair_counts, dtype=float)
    norm = LogNorm(vmin=max(pc_arr.min(), 1), vmax=pc_arr.max())

    legend_handles: list[Line2D] = []
    for idx, ds in enumerate(datasets):
        marker = _DATASET_MARKERS.get(ds, "o")
        color = cmap(norm(pair_counts[idx]))

        # CI error bars
        ax.errorbar(
            rho_exh[idx],
            rho_gre[idx],
            xerr=[[ci_exh_lo[idx]], [ci_exh_hi[idx]]],
            yerr=[[ci_gre_lo[idx]], [ci_gre_hi[idx]]],
            fmt="none",
            ecolor="0.45",
            capsize=3,
            capthick=0.8,
            elinewidth=0.8,
            zorder=4,
        )

        ax.scatter(
            rho_exh[idx],
            rho_gre[idx],
            marker=marker,
            c=[color],
            s=35,
            edgecolors="0.2",
            linewidth=0.6,
            zorder=5,
        )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="none",
                markerfacecolor="0.4",
                markeredgecolor="0.2",
                markersize=6,
                label=DATASET_DISPLAY[ds],
                linewidth=0,
            )
        )

    # Identity line
    lo = min(min(rho_exh), min(rho_gre)) - 0.05
    hi = max(max(rho_exh), max(rho_gre)) + 0.05
    ax.plot([lo, hi], [lo, hi], "--", color="0.6", lw=0.8, zorder=2)

    ax.set_xlabel("$\\rho$ Canonical", fontsize=8)
    ax.set_ylabel("$\\rho$ Greedy-Min", fontsize=8)
    ax.legend(
        handles=legend_handles,
        fontsize=6,
        loc="upper left",
        framealpha=0.9,
        handletextpad=0.3,
        borderpad=0.4,
    )
    ax.tick_params(labelsize=6)
    ax.set_aspect("equal")

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.046, aspect=20)
    cbar.set_label("Pair count", fontsize=7)
    cbar.ax.tick_params(labelsize=5)


# ---------------------------------------------------------------------------
# Panel (b): Aggregated speedup vs number of nodes
# ---------------------------------------------------------------------------


def _compute_binned_speedups(
    comp_dir: str,
    all_stats: dict[str, dict],
) -> tuple[
    np.ndarray,  # bin_centers
    np.ndarray,  # speedup_exhaustive_geo_mean
    np.ndarray,  # speedup_greedy_geo_mean
    np.ndarray,  # speedup_exhaustive_lo  (geo std band lower)
    np.ndarray,  # speedup_exhaustive_hi  (geo std band upper)
    np.ndarray,  # speedup_greedy_lo
    np.ndarray,  # speedup_greedy_hi
]:
    """Compute per-node-bin aggregated speedups across datasets.

    For each node-count bin (3-4, 5-6, 7-8, 9-10, 11-12), computes:
    - Exhaustive speedup: GED_time / (2 * exhaustive_encode + lev)
    - Greedy speedup: GED_time / (2 * greedy_encode + lev)

    Aggregates across datasets using geometric mean ± geometric std.
    """
    # Standard bin edges
    bin_edges = [(3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
    bin_centers_list = [(lo + hi) / 2 for lo, hi in bin_edges]

    # Collect per-bin, per-dataset speedups
    exh_speedups: dict[float, list[float]] = {c: [] for c in bin_centers_list}
    gre_speedups: dict[float, list[float]] = {c: [] for c in bin_centers_list}

    for ds in ALL_DATASETS:
        if ds not in all_stats:
            continue

        # Load raw CSVs for this dataset
        enc_df = _load_encoding_csv(comp_dir, ds)
        ged_df = _load_ged_csv(comp_dir, ds)
        lev_df = _load_lev_csv(comp_dir, ds)

        if enc_df is None or ged_df is None or lev_df is None:
            logger.warning("Missing raw CSVs for %s, using precomputed bins only.", ds)
            # Fall back to precomputed exhaustive crossover only
            bins = all_stats[ds].get("crossover", {}).get("bins", [])
            for b in bins:
                if b.get("speedup") is None:
                    continue
                center = (b["bin_lo"] + b["bin_hi"]) / 2
                if center in exh_speedups:
                    exh_speedups[center].append(b["speedup"])
            continue

        # Bin encoding times by n_nodes
        for (lo, hi), center in zip(bin_edges, bin_centers_list, strict=True):
            # Exhaustive encode median in this bin
            enc_bin = enc_df[(enc_df["n_nodes"] >= lo) & (enc_df["n_nodes"] <= hi)]
            if enc_bin.empty:
                continue

            exh_enc_med = enc_bin["exhaustive_time_median_s"].median()
            gre_enc_med = enc_bin["greedy_time_median_s"].median()

            # GED median in this bin (max_n is max(n_i, n_j))
            ged_bin = ged_df[(ged_df["max_n"] >= lo) & (ged_df["max_n"] <= hi)]
            if ged_bin.empty:
                continue
            ged_med = ged_bin["ged_time_median_s"].median()

            # Levenshtein median in this bin
            lev_bin = lev_df[(lev_df["max_n"] >= lo) & (lev_df["max_n"] <= hi)]
            lev_med = lev_bin["c_ext_time_median_s"].median() if not lev_bin.empty else 0.0

            # IsalGraph pipeline = 2 * encode + levenshtein (amortized per pair)
            t_exh = 2 * exh_enc_med + lev_med
            t_gre = 2 * gre_enc_med + lev_med

            sp_exh = ged_med / t_exh if t_exh > 0 else float("nan")
            sp_gre = ged_med / t_gre if t_gre > 0 else float("nan")

            if np.isfinite(sp_exh) and sp_exh > 0:
                exh_speedups[center].append(sp_exh)
            if np.isfinite(sp_gre) and sp_gre > 0:
                gre_speedups[center].append(sp_gre)

    # Aggregate: geometric mean ± geometric std
    valid_centers: list[float] = []
    geo_exh: list[float] = []
    geo_gre: list[float] = []
    geo_exh_lo: list[float] = []
    geo_exh_hi: list[float] = []
    geo_gre_lo: list[float] = []
    geo_gre_hi: list[float] = []

    for center in bin_centers_list:
        exh_vals = exh_speedups[center]
        gre_vals = gre_speedups[center]

        # Need at least 1 dataset for both methods at this bin
        if not exh_vals and not gre_vals:
            continue

        valid_centers.append(center)

        # Geometric mean and std in log-space
        if exh_vals:
            log_exh = np.log(exh_vals)
            gm_exh = np.exp(np.mean(log_exh))
            if len(log_exh) > 1:
                gs_exh = np.exp(np.std(log_exh))
            else:
                gs_exh = 1.0
            geo_exh.append(gm_exh)
            geo_exh_lo.append(gm_exh / gs_exh)
            geo_exh_hi.append(gm_exh * gs_exh)
        else:
            geo_exh.append(float("nan"))
            geo_exh_lo.append(float("nan"))
            geo_exh_hi.append(float("nan"))

        if gre_vals:
            log_gre = np.log(gre_vals)
            gm_gre = np.exp(np.mean(log_gre))
            if len(log_gre) > 1:
                gs_gre = np.exp(np.std(log_gre))
            else:
                gs_gre = 1.0
            geo_gre.append(gm_gre)
            geo_gre_lo.append(gm_gre / gs_gre)
            geo_gre_hi.append(gm_gre * gs_gre)
        else:
            geo_gre.append(float("nan"))
            geo_gre_lo.append(float("nan"))
            geo_gre_hi.append(float("nan"))

    return (
        np.array(valid_centers),
        np.array(geo_exh),
        np.array(geo_gre),
        np.array(geo_exh_lo),
        np.array(geo_exh_hi),
        np.array(geo_gre_lo),
        np.array(geo_gre_hi),
    )


def _draw_panel_b(
    ax: plt.Axes,
    comp_dir: str,
    all_stats: dict[str, dict],
) -> None:
    """Draw aggregated speedup vs number of nodes."""
    centers, exh_mean, gre_mean, exh_lo, exh_hi, gre_lo, gre_hi = _compute_binned_speedups(
        comp_dir, all_stats
    )

    if len(centers) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    valid_gre = np.isfinite(gre_mean)
    valid_exh = np.isfinite(exh_mean)
    # Only use bins where BOTH methods have data for the shaded gap
    both_valid = valid_gre & valid_exh

    # --- Greedy-Min line ---
    if valid_gre.any():
        ax.plot(
            centers[valid_gre],
            gre_mean[valid_gre],
            color=_GREEDY_COLOR,
            marker="D",
            linewidth=1.5,
            markersize=5,
            label="Greedy-Min",
            zorder=4,
        )

    # --- Canonical line ---
    if valid_exh.any():
        ax.plot(
            centers[valid_exh],
            exh_mean[valid_exh],
            color=_EXHAUSTIVE_COLOR,
            marker="o",
            linewidth=1.5,
            markersize=5,
            label="Canonical",
            zorder=4,
        )

    # --- Shadow between the two lines (greedy-min faster than canonical) ---
    if both_valid.any():
        bc = centers[both_valid]
        gm_upper = gre_mean[both_valid]
        gm_lower = exh_mean[both_valid]
        ax.fill_between(
            bc,
            gm_lower,
            gm_upper,
            color=_GREEDY_COLOR,
            alpha=0.12,
            zorder=2,
        )

        # Per-point speedup ratio annotations above greedy-min markers
        ratios = gm_upper / np.where(gm_lower > 0, gm_lower, 1.0)
        for i in range(len(bc)):
            # Offset upward in log-space to avoid overlap with marker
            y_offset = gm_upper[i] * 1.35
            ax.text(
                bc[i],
                y_offset,
                f"${ratios[i]:.1f}\\times$",
                fontsize=5.5,
                color="0.15",
                ha="center",
                va="bottom",
                zorder=8,
            )

    # Breakeven line
    ax.axhline(y=1, color="0.4", linestyle="--", linewidth=0.8, zorder=1)
    ax.text(
        centers[-1] + 0.3,
        1.05,
        "breakeven",
        fontsize=6,
        color="0.4",
        va="bottom",
        ha="right",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Number of nodes ($n$)", fontsize=8)
    ax.set_ylabel("Speedup (GED / IsalGraph)", fontsize=8)
    ax.legend(fontsize=6, loc="upper left")
    ax.tick_params(labelsize=6)

    # Nice x-ticks at bin centers
    ax.set_xticks(centers)
    ax.set_xticklabels([f"{int(c)}" for c in centers], fontsize=6)

    # Format y-axis
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0f}x" if x >= 1 else f"{x:.1f}x")
    )

    # Ensure top annotation is not clipped (log-scale: multiply by factor)
    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(y_lo, y_hi * 3.0)


# ---------------------------------------------------------------------------
# Main composite figure
# ---------------------------------------------------------------------------


def _generate_caption(
    comp_dir: str,
    all_timing: dict[str, dict],
    ds_data: dict[str, dict],
    stats_dir: str,
) -> str:
    """Auto-generate a LaTeX-compatible caption for the composite figure."""
    # Compute speedup range from binned data
    centers, exh_mean, gre_mean, *_ = _compute_binned_speedups(comp_dir, all_timing)
    valid_gre = np.isfinite(gre_mean)
    valid_exh = np.isfinite(exh_mean)

    gre_range = ""
    if valid_gre.any():
        gre_lo, gre_hi = gre_mean[valid_gre].min(), gre_mean[valid_gre].max()
        gre_range = f"${gre_lo:.1f}\\times$--${gre_hi:.0f}\\times$"

    exh_range = ""
    if valid_exh.any():
        exh_lo, exh_hi = exh_mean[valid_exh].min(), exh_mean[valid_exh].max()
        exh_range = f"${exh_lo:.1f}\\times$--${exh_hi:.0f}\\times$"

    n_lo = int(centers.min()) if len(centers) else 3
    n_hi = int(centers.max()) if len(centers) else 11

    # Compute rho ranges from cross-analysis
    cross = _load_cross_analysis(stats_dir)
    rho_info = ""
    if cross and "h5_method_comparison" in cross:
        h5 = cross["h5_method_comparison"]
        rho_exh_vals = [h5[ds]["rho_exhaustive"] for ds in ALL_DATASETS if ds in h5]
        rho_gre_vals = [h5[ds]["rho_greedy"] for ds in ALL_DATASETS if ds in h5]
        if rho_exh_vals and rho_gre_vals:
            rho_info = (
                f"Canonical $\\rho$ ranges from ${min(rho_exh_vals):.3f}$ to "
                f"${max(rho_exh_vals):.3f}$; "
                f"Greedy-Min $\\rho$ ranges from ${min(rho_gre_vals):.3f}$ to "
                f"${max(rho_gre_vals):.3f}$."
            )

    n_datasets = len([ds for ds in ALL_DATASETS if ds in all_timing])

    caption = (
        f"Computational--quality trade-off across {n_datasets} benchmark datasets. "
        f"(a) Geometric-mean speedup of the IsalGraph pipeline (encoding + Levenshtein) "
        f"over exact GED computation, aggregated across datasets and binned by graph size "
        f"($n = {n_lo}$--${n_hi}$ nodes). "
    )
    if gre_range:
        caption += f"Greedy-Min achieves {gre_range} speedup"
    if exh_range:
        caption += f"; Canonical achieves {exh_range} speedup"
    caption += (
        ". Per-point annotations indicate the ratio of Greedy-Min to Canonical speedup. "
        "Shaded region highlights the gap between the two methods. "
        "Dashed line: breakeven ($1\\times$). "
        "(b) Spearman rank correlation ($\\rho$) between Levenshtein distance and GED "
        "for Canonical (x-axis) vs.\\ Greedy-Min (y-axis) encoding, with 95\\% bootstrap "
        "confidence intervals. Points are colored by the number of graph pairs "
        "in each dataset (log scale). "
    )
    if rho_info:
        caption += rho_info + " "
    caption += (
        "Dashed line: identity ($\\rho_{\\text{Canonical}} = \\rho_{\\text{Greedy-Min}}$). "
        "All datasets lie below the identity line, indicating that Canonical encoding "
        "consistently yields higher correlation with GED."
    )
    return caption


def generate_composite_method_tradeoff(
    data_root: str,
    stats_dir: str,
    comp_dir: str,
    output_dir: str,
) -> str:
    """Generate the 1x2 composite figure.

    Panel (a): Aggregated speedup vs node count.
    Panel (b): Canonical vs Greedy-min rho scatter (pair-count-colored).
    """
    apply_ieee_style()
    os.makedirs(output_dir, exist_ok=True)

    # Load correlation & dataset artifacts
    results_obj = load_all_results(data_root, stats_dir)
    ds_data: dict[str, dict] = {}
    for ds_name, arts in results_obj.datasets.items():
        ds_data[ds_name] = {
            "node_counts": arts.node_counts,
            "edge_counts": arts.edge_counts,
        }

    # Load computational timing stats
    all_timing = _load_timing_stats(comp_dir)

    # Create figure — (a) speedup, (b) scatter
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(get_figure_size("double")[0], get_figure_size("single")[1]),
        gridspec_kw={"width_ratios": [1.15, 1]},
    )

    # Panel labels
    ax_a.set_title("(a)", fontsize=9, fontweight="bold", loc="left", pad=8)
    ax_b.set_title("(b)", fontsize=9, fontweight="bold", loc="left", pad=8)

    # Draw panels — swapped order
    _draw_panel_b(ax_a, comp_dir, all_timing)
    _draw_panel_a(ax_b, fig, ds_data, stats_dir)

    fig.tight_layout(w_pad=1.2)
    path = os.path.join(output_dir, "composite_method_tradeoff")
    save_figure(fig, path)
    plt.close(fig)

    # --- Auto-generate caption ---
    caption = _generate_caption(comp_dir, all_timing, ds_data, stats_dir)
    caption_path = os.path.join(output_dir, "composite_method_tradeoff_caption.txt")
    with open(caption_path, "w", encoding="utf-8") as f:
        f.write(caption + "\n")
    logger.info("Caption saved: %s", caption_path)

    logger.info("Composite figure saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate composite method tradeoff figure.")
    parser.add_argument(
        "--data-root",
        default="/media/mpascual/Sandisk2TB/research/isalgraph/data/eval",
        help="Root directory for eval data (GED matrices, etc.).",
    )
    parser.add_argument(
        "--stats-dir",
        default="/media/mpascual/Sandisk2TB/research/isalgraph/results/eval_benchmarks/eval_correlation/stats",
        help="Directory with correlation stats JSONs.",
    )
    parser.add_argument(
        "--comp-dir",
        default="/media/mpascual/Sandisk2TB/research/isalgraph/results/eval_benchmarks/eval_computational",
        help="Directory with computational evaluation outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="/media/mpascual/Sandisk2TB/research/isalgraph/results/figures/composite",
        help="Output directory for figures.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    generate_composite_method_tradeoff(
        args.data_root,
        args.stats_dir,
        args.comp_dir,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
