# ruff: noqa: N803, N806
"""Central 2x5 correlation heatmap grid — the paper's main evaluation figure.

Two rows (greedy-min, canonical) x five datasets.
Each cell is a hexbin density plot with KDE contour overlay of GED vs
Levenshtein distance, with identity line, OLS regression line, and
correlation annotations.  Shared colorbar spans all columns at bottom.
"""

from __future__ import annotations

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy import stats as sp_stats

from benchmarks.eval_visualizations.result_loader import (
    ALL_DATASETS,
    DATASET_DISPLAY,
    AllResults,
    load_all_results,
)
from benchmarks.eval_visualizations.table_generator import (
    format_significance,
    generate_dual_table,
)
from benchmarks.plotting_styles import apply_ieee_style, save_figure

logger = logging.getLogger(__name__)

# Row definitions
METHODS = ["greedy_min", "pruned_exhaustive", "exhaustive"]
METHOD_DISPLAY: dict[str, str] = {
    "greedy_min": r"$\mathbf{Greedy\text{-}min}$",
    "pruned_exhaustive": r"$\mathbf{Canonical\text{-}Pruned}$",
    "exhaustive": "Canonical",
}

# OLS line color (Paul Tol red)
_OLS_COLOR = "#EE6677"


# =============================================================================
# Data preparation
# =============================================================================


def _get_distance_vectors(
    results: AllResults,
    dataset: str,
    method: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract paired GED and Levenshtein distance vectors for a cell.

    Args:
        results: Loaded results.
        dataset: Dataset name.
        method: One of "greedy_single", "greedy_min", "exhaustive".

    Returns:
        (ged_vec, lev_vec) of valid pairs, or None if data unavailable.
    """
    if dataset not in results.datasets:
        return None

    arts = results.datasets[dataset]
    ged = arts.ged_matrix

    # Get Levenshtein matrix
    if method == "greedy_min":
        lev_matrix = results.levenshtein_matrices.get((dataset, "greedy"))
    elif method == "exhaustive":
        lev_matrix = results.levenshtein_matrices.get((dataset, "exhaustive"))
    elif method == "greedy_single":
        lev_matrix = results.levenshtein_matrices.get((dataset, "greedy_single"))
    else:
        return None
    if lev_matrix is None:
        return None

    # Extract upper triangle, filter valid pairs
    n = min(ged.shape[0], lev_matrix.shape[0])
    triu_i, triu_j = np.triu_indices(n, k=1)
    ged_vec = ged[triu_i, triu_j].astype(float)
    lev_vec = lev_matrix[triu_i, triu_j].astype(float)

    valid = np.isfinite(ged_vec) & np.isfinite(lev_vec) & (ged_vec > 0) & (lev_vec > 0)
    if not valid.any():
        return None

    return ged_vec[valid], lev_vec[valid]


# =============================================================================
# Correlation statistics
# =============================================================================


def _compute_cell_stats(
    ged_vec: np.ndarray,
    lev_vec: np.ndarray,
) -> dict:
    """Compute correlation statistics for a single cell."""
    rho, rho_p = sp_stats.spearmanr(ged_vec, lev_vec)
    r, r_p = sp_stats.pearsonr(ged_vec, lev_vec)

    # OLS regression
    slope, intercept, r_val, p_val, stderr = sp_stats.linregress(ged_vec, lev_vec)

    # Lin's CCC
    ccc = _lins_ccc(ged_vec, lev_vec)

    return {
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "pearson_r": float(r),
        "pearson_p": float(r_p),
        "ols_slope": float(slope),
        "ols_intercept": float(intercept),
        "ols_r2": float(r_val**2),
        "lins_ccc": float(ccc),
        "n_pairs": len(ged_vec),
    }


def _lins_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Lin's Concordance Correlation Coefficient."""
    mx, my = x.mean(), y.mean()
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    r = float(np.corrcoef(x, y)[0, 1])
    denominator = sx**2 + sy**2 + (mx - my) ** 2
    if denominator == 0:
        return 0.0
    return 2 * r * sx * sy / denominator


# =============================================================================
# Figure generation
# =============================================================================


def _add_kde_contours(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_levels: int = 3,
    color: str = "black",
) -> None:
    """Overlay KDE density contours on a hexbin plot.

    Uses Gaussian KDE evaluated on a regular grid, with contour levels
    at quantile thresholds of the density field.

    Args:
        ax: Matplotlib axes (already has hexbin drawn).
        x: x-coordinates.
        y: y-coordinates.
        n_levels: Number of contour levels.
        color: Contour line color.
    """
    from scipy.stats import gaussian_kde  # noqa: F811

    # Subsample for KDE performance (>50K pairs → subsample to 10K)
    n = len(x)
    if n > 10_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, 10_000, replace=False)
        x_kde, y_kde = x[idx], y[idx]
    else:
        x_kde, y_kde = x, y

    try:
        kde = gaussian_kde(np.vstack([x_kde, y_kde]))
    except np.linalg.LinAlgError:
        return  # Singular matrix — skip contours

    # Evaluate on grid
    x_grid = np.linspace(x.min(), x.max(), 80)
    y_grid = np.linspace(y.min(), y.max(), 80)
    xx, yy = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    zz = kde(positions).reshape(xx.shape)

    # Contour levels at high-density quantiles only (50%, 75%, 90%)
    sorted_z = np.sort(zz.ravel())
    quantiles = np.linspace(0.5, 0.9, n_levels)
    levels = [sorted_z[int(q * len(sorted_z))] for q in quantiles]
    levels = sorted(set(levels))  # Deduplicate

    if len(levels) < 2:
        return

    ax.contour(
        xx,
        yy,
        zz,
        levels=levels,
        colors=color,
        linewidths=0.4,
        alpha=0.5,
        zorder=6,
    )


def generate_heatmap_grid(
    results: AllResults,
    output_dir: str,
    *,
    znorm: bool = False,
) -> str:
    """Generate the 2x5 hexbin+KDE heatmap grid figure.

    Each cell shows hexbin density with KDE contour overlay.
    Shared horizontal colorbar spans all columns at figure bottom.

    Args:
        results: Loaded evaluation results.
        output_dir: Output directory.
        znorm: If True, z-normalize both axes per cell.

    Returns:
        Path to saved figure (without extension).
    """
    suffix = "znorm" if znorm else "raw"

    n_rows = len(METHODS)
    n_cols = len(ALL_DATASETS)

    # Use GridSpec: data rows + narrow colorbar row at bottom
    fig = plt.figure(figsize=(7.0, 2.8 * n_rows + 0.4))
    gs = fig.add_gridspec(
        n_rows + 1,
        n_cols,
        height_ratios=[*([1] * n_rows), 0.05],
        hspace=0.35,
        wspace=0.30,
        left=0.10,
        right=0.95,
        top=0.92,
        bottom=0.08,
    )

    axes = np.empty((n_rows, n_cols), dtype=object)
    for r in range(n_rows):
        for c in range(n_cols):
            axes[r, c] = fig.add_subplot(gs[r, c])

    # Colorbar axes spanning all columns at bottom
    cbar_ax = fig.add_subplot(gs[n_rows, :])

    # Track hexbin for shared colorbar
    last_hb = None

    for row_idx, method in enumerate(METHODS):
        for col_idx, dataset in enumerate(ALL_DATASETS):
            ax = axes[row_idx, col_idx]

            vecs = _get_distance_vectors(results, dataset, method)
            if vecs is None:
                ax.text(
                    0.5,
                    0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    fontsize=8,
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            ged_vec, lev_vec = vecs

            if znorm:
                ged_vec = (ged_vec - ged_vec.mean()) / max(ged_vec.std(), 1e-10)
                lev_vec = (lev_vec - lev_vec.mean()) / max(lev_vec.std(), 1e-10)

            # Hexbin density plot
            hb = ax.hexbin(
                ged_vec,
                lev_vec,
                gridsize=30,
                cmap="viridis",
                norm=LogNorm(),
                mincnt=1,
                linewidths=0.1,
                edgecolors="0.7",
            )
            last_hb = hb

            # KDE contour overlay
            _add_kde_contours(ax, ged_vec, lev_vec)

            # Identity line
            lo = min(ged_vec.min(), lev_vec.min())
            hi = max(ged_vec.max(), lev_vec.max())
            ax.plot(
                [lo, hi],
                [lo, hi],
                "--",
                color="0.6",
                lw=0.8,
                zorder=7,
            )

            # OLS regression line
            cell_stats = _compute_cell_stats(ged_vec, lev_vec)
            x_fit = np.array([lo, hi])
            y_fit = cell_stats["ols_slope"] * x_fit + cell_stats["ols_intercept"]
            ax.plot(
                x_fit,
                y_fit,
                "-",
                color=_OLS_COLOR,
                lw=1.2,
                zorder=7,
            )

            # Correlation annotation
            rho = cell_stats["spearman_rho"]
            r = cell_stats["pearson_r"]
            ax.text(
                0.05,
                0.95,
                f"$\\rho$={rho:.2f}\n$r$={r:.2f}",
                transform=ax.transAxes,
                fontsize=6,
                va="top",
                ha="left",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.8,
                    "edgecolor": "none",
                    "pad": 1,
                },
                zorder=10,
            )

            # Axis labels (bottom row only for x)
            if row_idx == n_rows - 1:
                xlabel = "GED" if not znorm else "GED (z)"
                ax.set_xlabel(xlabel, fontsize=7)
            if col_idx == 0:
                ylabel = "Lev." if not znorm else "Lev. (z)"
                ax.set_ylabel(ylabel, fontsize=7)

            ax.tick_params(labelsize=5)

            # Column title (top row only)
            if row_idx == 0:
                ax.set_title(DATASET_DISPLAY[dataset], fontsize=8)

        # Row label
        axes[row_idx, 0].annotate(
            METHOD_DISPLAY[method],
            xy=(-0.55, 0.5),
            xycoords="axes fraction",
            fontsize=7,
            ha="right",
            va="center",
            rotation=90,
        )

    # Shared horizontal colorbar at bottom
    if last_hb is not None:
        fig.colorbar(
            last_hb,
            cax=cbar_ax,
            orientation="horizontal",
            label="Count",
        )
        cbar_ax.tick_params(labelsize=5)
    else:
        cbar_ax.set_visible(False)

    path = os.path.join(output_dir, f"fig_central_heatmap_grid_{suffix}")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("Heatmap grid (%s) saved: %s", suffix, path)
    return path


# =============================================================================
# Aggregated size-stratified correlation (1x2)
# =============================================================================


def _collect_aggregated_pairs(
    results: AllResults,
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Collect GED, Lev, and mean pair size across all datasets for one method.

    For each valid pair (i,j), the size proxy is the mean node count
    (|V_i| + |V_j|) / 2.

    Returns:
        (ged_all, lev_all, size_all) concatenated across datasets, or None.
    """
    ged_list: list[np.ndarray] = []
    lev_list: list[np.ndarray] = []
    size_list: list[np.ndarray] = []

    for dataset in ALL_DATASETS:
        vecs = _get_distance_vectors(results, dataset, method)
        if vecs is None:
            continue

        arts = results.datasets.get(dataset)
        if arts is None:
            continue

        node_counts = arts.node_counts.astype(float)
        ged_mat = arts.ged_matrix
        if method == "greedy_min":
            lev_mat = results.levenshtein_matrices.get((dataset, "greedy"))
        elif method == "greedy_single":
            lev_mat = results.levenshtein_matrices.get((dataset, "greedy_single"))
        else:
            lev_mat = results.levenshtein_matrices.get((dataset, "exhaustive"))
        if lev_mat is None:
            continue

        n = min(ged_mat.shape[0], lev_mat.shape[0], len(node_counts))
        triu_i, triu_j = np.triu_indices(n, k=1)
        ged_flat = ged_mat[triu_i, triu_j].astype(float)
        lev_flat = lev_mat[triu_i, triu_j].astype(float)

        # Mean pair size (node count)
        pair_size = (node_counts[triu_i] + node_counts[triu_j]) / 2.0

        valid = np.isfinite(ged_flat) & np.isfinite(lev_flat) & (ged_flat > 0) & (lev_flat > 0)
        ged_list.append(ged_flat[valid])
        lev_list.append(lev_flat[valid])
        size_list.append(pair_size[valid])

    if not ged_list:
        return None

    return np.concatenate(ged_list), np.concatenate(lev_list), np.concatenate(size_list)


def _compute_agreement_stats(
    ged: np.ndarray,
    lev: np.ndarray,
) -> dict:
    """Compute a rich set of agreement statistics between GED and Lev vectors.

    Metrics:
        rho:  Spearman rank correlation (rank preservation).
        r:    Pearson linear correlation.
        ccc:  Lin's Concordance Correlation Coefficient (agreement with
              the identity line — the gold standard for method comparison;
              Lin, 1989, Biometrics 45(1):255-268).
        r2:   OLS R² (proportion of variance explained by linear fit).
        beta: OLS slope (scaling factor; beta=1 means same scale).
        mae:  Mean Absolute Error from identity (mean |Lev - GED|).
        n:    Number of valid pairs.
    """
    cell = _compute_cell_stats(ged, lev)
    mae = float(np.mean(np.abs(lev - ged)))
    return {
        "rho": cell["spearman_rho"],
        "r": cell["pearson_r"],
        "ccc": cell["lins_ccc"],
        "r2": cell["ols_r2"],
        "beta": cell["ols_slope"],
        "mae": mae,
        "n": cell["n_pairs"],
    }


def generate_aggregated_density_heatmap(
    results: AllResults,
    output_dir: str,
    *,
    cbar_pad: float = 0.22,
) -> str:
    """Generate 1x3 aggregated heatmap of pair counts.

    Panels: (a) Canonical, (b) Greedy-min, (c) Greedy-rnd(v₀).
    Each integer-aligned cell (GED=i, Lev=j) is colored by the count of
    graph pairs falling into that cell.  Uses square bins aligned to the
    integer lattice — both GED and Levenshtein are discrete, so hexbins
    cause Moiré aliasing artifacts on integer grids.

    Statistics annotated per panel:
        rho (Spearman rank correlation), beta (OLS slope).
    """
    from matplotlib.gridspec import GridSpec

    # --- Panel definitions: (method_key, panel_label) ---
    panel_defs = [
        ("exhaustive", "(a) Canonical"),
        ("greedy_min", r"(b) Greedy-min"),
        ("greedy_single", r"(c) Greedy-rnd($v_0$)"),
    ]

    # --- First pass: collect data, compute shared limits -----------------
    panel_data: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray, dict] | None] = []
    global_max = 0.0

    for method, label in panel_defs:
        data = _collect_aggregated_pairs(results, method)
        if data is None:
            panel_data.append(None)
            continue
        ged_all, lev_all, size_all = data
        stats = _compute_agreement_stats(ged_all, lev_all)
        cell = _compute_cell_stats(ged_all, lev_all)
        stats["intercept"] = cell["ols_intercept"]

        p999 = max(np.percentile(ged_all, 99.9), np.percentile(lev_all, 99.9))
        global_max = max(global_max, p999)

        panel_data.append((method, label, ged_all, lev_all, size_all, stats))

    n_panels = len(panel_defs)

    # Integer-aligned bins: edges at -0.5, 0.5, ..., ceil(max)+0.5
    bin_max = int(np.ceil(global_max)) + 1
    bin_edges = np.arange(-0.5, bin_max + 0.5, 1.0)
    shared_lim = (0.5, bin_max - 0.5)  # Start at 1 (GED=0, Lev=0 pairs are excluded)
    shared_ticks = np.arange(1, bin_max, max(1, bin_max // 8))

    # Precompute shared count normalization across all panels
    max_count = 1
    for pdata in panel_data:
        if pdata is None:
            continue
        _, _, g, l, _, _ = pdata
        counts = np.histogram2d(g, l, bins=[bin_edges, bin_edges])[0]
        max_count = max(max_count, int(counts.max()))
    count_norm = LogNorm(vmin=1, vmax=max_count)

    # --- Figure layout: data row + thin colorbar row --------------------
    fig = plt.figure(figsize=(10.0, 3.8))
    gs = GridSpec(
        2,
        n_panels,
        figure=fig,
        height_ratios=[1, 0.03],
        hspace=cbar_pad,
        wspace=0.06,
        left=0.06,
        right=0.97,
        bottom=0.10,
        top=0.90,
    )
    axes_list = [fig.add_subplot(gs[0, i]) for i in range(n_panels)]
    cbar_ax = fig.add_subplot(gs[1, :])

    stat_bbox = {
        "boxstyle": "round,pad=0.35",
        "facecolor": "white",
        "edgecolor": "0.4",
        "linewidth": 0.6,
        "alpha": 0.92,
    }

    last_mesh = None

    for col_idx, (ax, pdata) in enumerate(zip(axes_list, panel_data, strict=True)):
        if pdata is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue

        _method, label, ged_all, lev_all, size_all, stats = pdata

        # 2D histogram: count of pairs per integer cell
        counts_2d = np.histogram2d(
            ged_all,
            lev_all,
            bins=[bin_edges, bin_edges],
        )[0].T  # (n_lev_bins, n_ged_bins)
        counts_filled = counts_2d.copy()
        counts_filled[counts_filled < 1] = 0.5

        mesh = ax.pcolormesh(
            bin_edges,
            bin_edges,
            counts_filled,
            cmap="viridis",
            norm=count_norm,
            rasterized=True,
        )
        last_mesh = mesh

        # Identity line
        ax.plot(
            [shared_lim[0], shared_lim[1]],
            [shared_lim[0], shared_lim[1]],
            "--",
            color="0.3",
            lw=0.8,
            zorder=7,
        )

        # OLS regression line
        x_fit = np.array([0, shared_lim[1]])
        y_fit = stats["beta"] * x_fit + stats["intercept"]
        ax.plot(x_fit, y_fit, "-", color=_OLS_COLOR, lw=1.2, zorder=7)

        # Annotation: rho + beta (bottom-right corner)
        ann = f"$\\rho = {stats['rho']:.3f}$\n$\\beta = {stats['beta']:.2f}$"
        ax.text(
            0.96,
            0.04,
            ann,
            transform=ax.transAxes,
            fontsize=7,
            va="bottom",
            ha="right",
            bbox=stat_bbox,
            zorder=10,
            linespacing=1.3,
        )

        ax.set_xlim(shared_lim)
        ax.set_ylim(shared_lim)
        ax.set_xticks(shared_ticks)
        ax.set_yticks(shared_ticks)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(label, fontsize=9, fontweight="bold", loc="left")
        ax.set_xlabel("GED", fontsize=8)
        ax.tick_params(labelsize=6)

        # Only first panel gets y-label
        if col_idx == 0:
            ax.set_ylabel("Levenshtein distance", fontsize=8)
        else:
            ax.tick_params(labelleft=False)

    # Horizontal colorbar spanning all columns
    if last_mesh is not None:
        cb = fig.colorbar(last_mesh, cax=cbar_ax, orientation="horizontal")
        cb.set_label("Count (number of graph pairs)", fontsize=7)
        cb.ax.tick_params(labelsize=6)
    else:
        cbar_ax.set_visible(False)

    path = os.path.join(output_dir, "fig_aggregated_density_correlation")
    save_figure(fig, path)
    plt.close(fig)

    # --- Caption with actual computed values ----------------------------
    stats_by_label: dict[str, dict] = {}
    for pdata in panel_data:
        if pdata is not None:
            stats_by_label[pdata[0]] = pdata[5]

    s_canon = stats_by_label.get("exhaustive", {})
    s_greedy = stats_by_label.get("greedy_min", {})
    s_single = stats_by_label.get("greedy_single", {})

    caption_path = os.path.join(output_dir, "fig_aggregated_density_correlation_caption.txt")
    caption = (
        "Aggregated correlation between graph edit distance (GED) and Levenshtein "
        "distance across all five benchmark datasets. Each cell at integer "
        "coordinates $(i, j)$ shows the count of graph pairs with "
        "$\\text{GED} = i$ and $\\text{Lev} = j$ (log scale; "
        "light = few pairs, dark = many pairs); white cells contain no observed "
        "pairs. "
        "Dashed grey line: identity ($\\text{Lev} = \\text{GED}$). "
        "Solid red line: ordinary least-squares (OLS) regression. "
        "(a) Canonical encoding "
        f"($n = {s_canon.get('n', 0):,}$ pairs, "
        f"$\\rho = {s_canon.get('rho', 0):.3f}$, "
        f"$\\beta = {s_canon.get('beta', 0):.2f}$). "
        "(b) Greedy-min encoding "
        f"($n = {s_greedy.get('n', 0):,}$ pairs, "
        f"$\\rho = {s_greedy.get('rho', 0):.3f}$, "
        f"$\\beta = {s_greedy.get('beta', 0):.2f}$). "
        "(c) Greedy-rnd($v_0$) encoding "
        f"($n = {s_single.get('n', 0):,}$ pairs, "
        f"$\\rho = {s_single.get('rho', 0):.3f}$, "
        f"$\\beta = {s_single.get('beta', 0):.2f}$). "
        "Reported statistics: "
        "$\\rho$ denotes Spearman's rank correlation coefficient, measuring "
        "monotonic association between the two distance measures. "
        "$\\beta$ denotes the OLS regression slope; $\\beta = 1$ would indicate "
        "that Levenshtein and GED operate on the same scale, while $\\beta < 1$ "
        "indicates that Levenshtein distances grow more slowly than GED."
    )
    with open(caption_path, "w", encoding="utf-8") as f:
        f.write(caption + "\n")

    logger.info("Aggregated density heatmap saved: %s", path)
    return path


# =============================================================================
# Table generation
# =============================================================================


def generate_companion_table(
    results: AllResults,
    output_dir: str,
) -> None:
    """Generate companion correlation table (15 rows: 3 methods x 5 datasets)."""
    rows: list[dict] = []

    for method in METHODS:
        # Try to load pre-computed stats for exhaustive/greedy
        for dataset in ALL_DATASETS:
            vecs = _get_distance_vectors(results, dataset, method)
            if vecs is None:
                rows.append(
                    {
                        "Dataset": DATASET_DISPLAY[dataset],
                        "Method": METHOD_DISPLAY[method],
                        "Spearman rho": "---",
                        "Pearson r": "---",
                        "Lin CCC": "---",
                        "OLS slope": "---",
                        "n_pairs": "---",
                    }
                )
                continue

            ged_vec, lev_vec = vecs
            cell = _compute_cell_stats(ged_vec, lev_vec)

            sig_rho = format_significance(cell["spearman_p"])
            sig_r = format_significance(cell["pearson_p"])

            rows.append(
                {
                    "Dataset": DATASET_DISPLAY[dataset],
                    "Method": METHOD_DISPLAY[method],
                    "Spearman rho": f"{cell['spearman_rho']:.3f} {sig_rho}",
                    "Pearson r": f"{cell['pearson_r']:.3f} {sig_r}",
                    "Lin CCC": f"{cell['lins_ccc']:.3f}",
                    "OLS slope": f"{cell['ols_slope']:.3f}",
                    "n_pairs": f"{cell['n_pairs']:,}",
                }
            )

    df = pd.DataFrame(rows)
    generate_dual_table(
        df,
        output_dir,
        "table_central_heatmap",
        caption=(
            "Correlation between GED and IsalGraph Levenshtein distance"
            " across encoding methods and datasets."
        ),
        label="tab:central-heatmap",
        highlight_cols={"Spearman rho", "Pearson r", "Lin CCC"},
    )
    logger.info("Companion table saved to %s", output_dir)


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """CLI entry point for central heatmap grid."""
    parser = argparse.ArgumentParser(description="Generate 3x5 central correlation heatmap grid.")
    parser.add_argument(
        "--data-root",
        required=True,
        help="Root directory for Agent 0 outputs (ged_matrices/, levenshtein_matrices/, etc.).",
    )
    parser.add_argument(
        "--correlation-dir",
        default=None,
        help="Directory with correlation stats JSONs from Agent 2.",
    )
    parser.add_argument(
        "--output-dir",
        default="paper_figures/central_heatmap",
        help="Output directory.",
    )
    parser.add_argument("--plot", action="store_true", help="Generate figures.")
    parser.add_argument("--table", action="store_true", help="Generate tables.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    apply_ieee_style()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    results = load_all_results(args.data_root, args.correlation_dir)

    if args.plot:
        generate_heatmap_grid(results, args.output_dir, znorm=False)
        generate_heatmap_grid(results, args.output_dir, znorm=True)
        generate_aggregated_density_heatmap(results, args.output_dir)

    if args.table:
        generate_companion_table(results, args.output_dir)

    if not args.plot and not args.table:
        logger.warning("No output requested. Use --plot and/or --table.")


if __name__ == "__main__":
    main()
