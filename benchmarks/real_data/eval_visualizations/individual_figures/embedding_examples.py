# ruff: noqa: N803, N806
"""Individual example figures for embedding hypotheses H2.1-H2.3.

Generates:
  - H2.1: 2D MDS scatter of IAM LOW (Lev vs GED), colored by class
  - H2.2: Procrustes overlay of Lev and GED MDS for IAM LOW
  - H2.3: Best vs worst Shepard hexbin diagrams
"""

from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import procrustes as scipy_procrustes
from scipy.stats import binned_statistic_2d  # noqa: E402
from sklearn.isotonic import IsotonicRegression

from benchmarks.eval_visualizations.embedding_loader import (
    EmbeddingData,
    load_smacof_coords,
)
from benchmarks.eval_visualizations.result_loader import (
    ALL_DATASETS,
    DATASET_DISPLAY,
    AllResults,
)
from benchmarks.plotting_styles import PAUL_TOL_MUTED, save_figure

logger = logging.getLogger(__name__)


# =====================================================================
# H2.1 — 2D MDS scatter colored by class
# =====================================================================

# IAM Letter LOW has 9 classes — map each to a Paul Tol muted color
_CLASS_PALETTE = PAUL_TOL_MUTED[:9]


def generate_h2_1_individual(
    results: AllResults,
    emb: EmbeddingData,
    embedding_dir: str,
    output_dir: str,
    dataset: str = "iam_letter_low",
    method: str = "exhaustive",
) -> str:
    """Generate H2.1 individual figure: 2D MDS scatter for IAM LOW.

    Left panel: Levenshtein-based MDS, colored by class label.
    Right panel: GED-based MDS, colored by class label.
    """
    arts = results.datasets.get(dataset)
    if arts is None or arts.labels is None:
        logger.warning("Dataset %s not available or has no labels", dataset)
        return ""

    # Load 2D SMACOF coordinates
    coords_lev = load_smacof_coords(embedding_dir, dataset, f"lev_{method}", 2)
    coords_ged = load_smacof_coords(embedding_dir, dataset, "ged", 2)

    if coords_lev is None or coords_ged is None:
        logger.warning("Missing SMACOF 2D coordinates for %s", dataset)
        return ""

    labels = np.array([str(lbl) for lbl in arts.labels])
    unique_labels = sorted(set(labels))
    label_to_color = {
        lbl: _CLASS_PALETTE[i % len(_CLASS_PALETTE)] for i, lbl in enumerate(unique_labels)
    }

    fig, (ax_lev, ax_ged) = plt.subplots(1, 2, figsize=(7.0, 3.2))

    # Shuffle plot order for fair z-order across classes
    rng = np.random.default_rng(42)
    plot_order = rng.permutation(len(labels))

    for ax, coords, title in [
        (ax_lev, coords_lev, f"Levenshtein ({method})"),
        (ax_ged, coords_ged, "GED"),
    ]:
        # Small jitter to reveal density (integer Lev → tight MDS clusters)
        jitter_scale = 0.015 * (coords.max(axis=0) - coords.min(axis=0))
        jittered = coords + rng.normal(0, jitter_scale, coords.shape)

        # Plot shuffled for z-order fairness
        for lbl in unique_labels:
            mask = labels[plot_order] == lbl
            idx = plot_order[mask]
            ax.scatter(
                jittered[idx, 0],
                jittered[idx, 1],
                c=label_to_color[lbl],
                label=lbl,
                s=3,
                alpha=0.3,
                edgecolors="none",
                rasterized=True,
            )
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("MDS dim 1", fontsize=7)
        ax.set_ylabel("MDS dim 2", fontsize=7)
        ax.tick_params(labelsize=6)

    # Shared legend
    handles, lbls = ax_lev.get_legend_handles_labels()
    fig.legend(
        handles,
        lbls,
        loc="lower center",
        ncol=min(len(unique_labels), 9),
        fontsize=6,
        markerscale=1.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        f"2D MDS embedding — {DATASET_DISPLAY[dataset]}",
        fontsize=9,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    path = os.path.join(output_dir, "individual_mds_scatter")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("H2.1 individual figure saved: %s", path)
    return path


# =====================================================================
# H2.2 — Procrustes overlay
# =====================================================================


def generate_h2_2_individual(
    results: AllResults,
    emb: EmbeddingData,
    embedding_dir: str,
    output_dir: str,
    dataset: str = "iam_letter_low",
    method: str = "exhaustive",
) -> str:
    """Generate H2.2 individual figure: Procrustes overlay of Lev and GED MDS.

    After Procrustes alignment, superimposes both 2D configurations.
    Lines connect corresponding points, colored by residual magnitude.
    """
    arts = results.datasets.get(dataset)
    if arts is None:
        logger.warning("Dataset %s not available", dataset)
        return ""

    coords_lev = load_smacof_coords(embedding_dir, dataset, f"lev_{method}", 2)
    coords_ged = load_smacof_coords(embedding_dir, dataset, "ged", 2)

    if coords_lev is None or coords_ged is None:
        logger.warning("Missing SMACOF 2D coordinates for %s", dataset)
        return ""

    # Procrustes alignment: both configs standardized (centered, unit Frobenius norm)
    std_lev, aligned_ged, disparity = scipy_procrustes(coords_lev, coords_ged)

    # Residuals per point
    residuals = np.sqrt(np.sum((std_lev - aligned_ged) ** 2, axis=1))
    res_norm = residuals / (residuals.max() + 1e-12)

    # Small jitter to reveal density (integer MDS → tight clusters)
    rng = np.random.default_rng(42)
    jitter_scale = 0.012 * (std_lev.max(axis=0) - std_lev.min(axis=0))
    std_lev_j = std_lev + rng.normal(0, jitter_scale, std_lev.shape)
    aligned_ged_j = aligned_ged + rng.normal(0, jitter_scale, aligned_ged.shape)

    fig, ax = plt.subplots(figsize=(3.39, 3.39))

    # Connecting lines colored by residual (low=green, high=red)
    for i in range(len(std_lev)):
        color_val = plt.cm.RdYlGn_r(res_norm[i])
        ax.plot(
            [std_lev_j[i, 0], aligned_ged_j[i, 0]],
            [std_lev_j[i, 1], aligned_ged_j[i, 1]],
            color=color_val,
            linewidth=0.3,
            alpha=0.3,
            zorder=1,
        )

    # Lev points (circles, blue)
    ax.scatter(
        std_lev_j[:, 0],
        std_lev_j[:, 1],
        c=PAUL_TOL_MUTED[4],
        marker="o",
        s=4,
        alpha=0.5,
        label="Levenshtein",
        edgecolors="none",
        zorder=2,
        rasterized=True,
    )

    # GED points (triangles, red)
    ax.scatter(
        aligned_ged_j[:, 0],
        aligned_ged_j[:, 1],
        c=PAUL_TOL_MUTED[0],
        marker="^",
        s=4,
        alpha=0.5,
        label="GED (aligned)",
        edgecolors="none",
        zorder=2,
        rasterized=True,
    )

    # Colorbar for residuals
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(0, residuals.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Residual", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    # Report Procrustes m²
    proc_entry = emb.stats.get(dataset, None)
    m2_str = ""
    if proc_entry:
        proc = proc_entry.procrustes.get((method, 2))
        if proc:
            m2_str = f"$m^2 = {proc.m_squared:.4f}$"

    ax.set_title(
        f"Procrustes overlay — {DATASET_DISPLAY[dataset]}\n{m2_str}",
        fontsize=8,
    )
    ax.set_xlabel("Aligned dim 1", fontsize=7)
    ax.set_ylabel("Aligned dim 2", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, loc="upper left", markerscale=1.5)

    fig.tight_layout()
    path = os.path.join(output_dir, "individual_procrustes_overlay")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("H2.2 individual figure saved: %s", path)
    return path


# =====================================================================
# H2.3 — Shepard hexbin diagrams
# =====================================================================


def _compute_embedded_distances(coords: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances from embedding coordinates.

    Returns upper-triangle vector (same ordering as scipy.spatial.distance.pdist).
    """
    from scipy.spatial.distance import pdist

    return pdist(coords, metric="euclidean")


def _get_original_distances_upper(matrix: np.ndarray) -> np.ndarray:
    """Extract upper-triangle of a symmetric distance matrix as a flat vector."""
    n = matrix.shape[0]
    idx = np.triu_indices(n, k=1)
    return matrix[idx].astype(float)


def _find_best_worst_shepard(
    emb: EmbeddingData,
    method: str = "exhaustive",
    dim: int = 2,
) -> tuple[str | None, str | None]:
    """Find dataset with best and worst Shepard lev_r_squared."""
    best_ds, worst_ds = None, None
    best_r2, worst_r2 = -1.0, 2.0

    for ds in ALL_DATASETS:
        ds_stats = emb.stats.get(ds)
        if ds_stats is None:
            continue
        shep = ds_stats.shepard.get((method, dim))
        if shep is None:
            continue

        if shep.lev_r_squared > best_r2:
            best_r2 = shep.lev_r_squared
            best_ds = ds
        if shep.lev_r_squared < worst_r2:
            worst_r2 = shep.lev_r_squared
            worst_ds = ds

    return best_ds, worst_ds


def _compute_pair_sizes(node_counts: np.ndarray, n: int) -> np.ndarray:
    """Compute mean pair size (|V_i|+|V_j|)/2 for upper-triangle pairs."""
    v = node_counts[:n].astype(np.float64)
    idx_i, idx_j = np.triu_indices(n, k=1)
    return (v[idx_i] + v[idx_j]) / 2.0


def _collect_shepard_row_data(
    results: AllResults,
    emb: EmbeddingData,
    embedding_dir: str,
    ds: str,
    method: str,
    dim: int,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float] | None]:
    """Collect original/embedded distance vectors, pair sizes, and R² for one dataset row.

    Returns:
        List of (source_label, orig_valid, emb_valid, pair_sizes_valid, r2) or None per column.
    """
    arts = results.datasets.get(ds)
    if arts is None:
        return [None, None]

    lev_matrix = results.levenshtein_matrices.get((ds, method))
    ged_matrix = arts.ged_matrix
    coords_lev = load_smacof_coords(embedding_dir, ds, f"lev_{method}", dim)
    coords_ged = load_smacof_coords(embedding_dir, ds, "ged", dim)

    row_data: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float] | None] = []
    for source_label, orig_matrix, coords in [
        ("Levenshtein", lev_matrix, coords_lev),
        ("GED", ged_matrix, coords_ged),
    ]:
        if orig_matrix is None or coords is None:
            row_data.append(None)
            continue

        n = min(orig_matrix.shape[0], coords.shape[0])
        orig_sub = orig_matrix[:n, :n]
        orig_vec = _get_original_distances_upper(orig_sub)
        emb_vec = _compute_embedded_distances(coords[:n])
        pair_sizes = _compute_pair_sizes(arts.node_counts, n)

        valid = np.isfinite(orig_vec) & np.isfinite(emb_vec) & (orig_vec > 0)
        orig_valid = orig_vec[valid]
        emb_valid = emb_vec[valid]
        sizes_valid = pair_sizes[valid]

        if len(orig_valid) < 10:
            row_data.append(None)
            continue

        # Get R²
        r2 = 0.0
        ds_stats = emb.stats.get(ds)
        if ds_stats:
            shep = ds_stats.shepard.get((method, dim))
            if shep:
                r2 = shep.lev_r_squared if source_label == "Levenshtein" else shep.ged_r_squared

        row_data.append((source_label, orig_valid, emb_valid, sizes_valid, r2))

    return row_data


def generate_h2_3_individual(
    results: AllResults,
    emb: EmbeddingData,
    embedding_dir: str,
    output_dir: str,
    method: str = "exhaustive",
    dim: int = 2,
) -> str:
    """Generate H2.3 individual figure: best vs worst Shepard diagrams.

    2x2 grid: rows = best/worst dataset, columns = Lev/GED.
    Each cell uses integer-aligned square bins colored by mean pair size.
    Shared colorbar row at bottom. Panels labelled (a) and (b).
    """
    from matplotlib.gridspec import GridSpec

    best_ds, worst_ds = _find_best_worst_shepard(emb, method, dim)
    if best_ds is None or worst_ds is None:
        logger.warning("Cannot determine best/worst Shepard datasets")
        return ""

    # Pre-collect data to compute shared axis limits per row
    rows_info: list[tuple[str, str, list]] = []  # (ds, panel_label, row_data)
    for ds, panel_label in [(best_ds, "(a)"), (worst_ds, "(b)")]:
        row_data = _collect_shepard_row_data(results, emb, embedding_dir, ds, method, dim)
        rows_info.append((ds, panel_label, row_data))

    # Compute global min/max for colorbar (mean pair size across all cells)
    global_size_min, global_size_max = np.inf, -np.inf
    for _, _, row_data in rows_info:
        for cell in row_data:
            if cell is not None:
                _, _, _, sizes_valid, _ = cell
                global_size_min = min(global_size_min, sizes_valid.min())
                global_size_max = max(global_size_max, sizes_valid.max())

    # Build figure: 2 data rows + 1 colorbar row
    fig = plt.figure(figsize=(5.0, 4.8))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.04], hspace=0.12, wspace=0.10)

    axes = np.empty((2, 2), dtype=object)
    for r in range(2):
        for c in range(2):
            axes[r, c] = fig.add_subplot(gs[r, c])

    # R² textbox style
    r2_bbox = {
        "boxstyle": "round,pad=0.35",
        "facecolor": "white",
        "edgecolor": "0.4",
        "linewidth": 0.6,
        "alpha": 0.92,
    }

    last_mappable = None  # for colorbar

    for row_idx, (ds, panel_label, row_data) in enumerate(rows_info):
        # Compute shared axis limits for this row
        all_orig: list[np.ndarray] = []
        all_emb: list[np.ndarray] = []
        for cell in row_data:
            if cell is not None:
                _, ov, ev, _, _ = cell
                all_orig.append(ov)
                all_emb.append(ev)

        if all_orig:
            x_max = max(np.percentile(o, 99.5) for o in all_orig)
            y_max = max(np.percentile(e, 99.5) for e in all_emb)
        else:
            x_max, y_max = 1.0, 1.0

        for col_idx, cell in enumerate(row_data):
            ax = axes[row_idx, col_idx]

            if cell is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
                continue

            source_label, orig_valid, emb_valid, sizes_valid, r2 = cell

            # Integer-aligned bins on x (original distances are integers).
            # y bins use same width (1.0) for visual squares.
            x_bin_max = int(np.ceil(x_max)) + 1
            y_bin_max = int(np.ceil(y_max)) + 1
            x_edges = np.arange(-0.5, x_bin_max + 0.5, 1.0)
            y_edges = np.arange(-0.5, y_bin_max + 0.5, 1.0)

            # Bin by mean pair size
            stat_result = binned_statistic_2d(
                orig_valid,
                emb_valid,
                sizes_valid,
                statistic="mean",
                bins=[x_edges, y_edges],
            )
            grid = stat_result.statistic.T  # shape: (ny, nx)

            # Mask empty cells
            masked = np.ma.masked_invalid(grid)

            # Plot
            im = ax.pcolormesh(
                x_edges,
                y_edges,
                masked,
                cmap="viridis",
                vmin=global_size_min,
                vmax=global_size_max,
                rasterized=True,
            )
            last_mappable = im

            # Isotonic regression line
            iso = IsotonicRegression(increasing=True)
            sort_idx = np.argsort(orig_valid)
            orig_sorted = orig_valid[sort_idx]
            emb_sorted = emb_valid[sort_idx]
            iso_pred = iso.fit_transform(orig_sorted, emb_sorted)
            ax.plot(
                orig_sorted, iso_pred, color=PAUL_TOL_MUTED[0], linewidth=1.2, label="Isotonic fit"
            )

            # Enforce shared limits within row
            ax.set_xlim(-0.5, x_max * 1.03)
            ax.set_ylim(-0.5, y_max * 1.03)

            # Axis labels
            if row_idx == 1:
                ax.set_xlabel("Original distance", fontsize=7)
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)
            if col_idx == 0:
                ax.set_ylabel("Embedded distance", fontsize=7)
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
            ax.tick_params(labelsize=6)

            # R² annotation
            ax.text(
                0.05,
                0.93,
                f"$R^2 = {r2:.3f}$",
                transform=ax.transAxes,
                fontsize=7,
                va="top",
                ha="left",
                bbox=r2_bbox,
            )

            # Column titles on first row only
            if row_idx == 0:
                ax.set_title(source_label, fontsize=9)

        # Panel label
        axes[row_idx, 0].text(
            -0.22,
            0.5,
            f"{panel_label} {DATASET_DISPLAY[ds]}",
            transform=axes[row_idx, 0].transAxes,
            fontsize=8,
            fontweight="bold",
            va="center",
            ha="right",
            rotation=90,
        )

    # Shared horizontal colorbar in its own row
    if last_mappable is not None:
        cbar_ax = fig.add_subplot(gs[2, :])
        cbar = fig.colorbar(last_mappable, cax=cbar_ax, orientation="horizontal")
        cbar.set_label(
            r"Mean pair size  $\bar{n} = (|V_i|+|V_j|)/2$",
            fontsize=7,
        )
        cbar.ax.tick_params(labelsize=6)

    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.10, top=0.92)

    path = os.path.join(output_dir, "individual_shepard_diagrams")
    save_figure(fig, path)
    plt.close(fig)

    # --- Auto-generate caption ---
    best_display = DATASET_DISPLAY.get(best_ds, best_ds)
    worst_display = DATASET_DISPLAY.get(worst_ds, worst_ds)

    # Collect R² values for caption
    r2_values: dict[str, dict[str, float]] = {}
    for ds_name, _, row_data in rows_info:
        r2_values[ds_name] = {}
        for cell in row_data:
            if cell is not None:
                src, _, _, _, r2_val = cell
                r2_values[ds_name][src] = r2_val

    r2_best_lev = r2_values.get(best_ds, {}).get("Levenshtein", 0.0)
    r2_best_ged = r2_values.get(best_ds, {}).get("GED", 0.0)
    r2_worst_lev = r2_values.get(worst_ds, {}).get("Levenshtein", 0.0)
    r2_worst_ged = r2_values.get(worst_ds, {}).get("GED", 0.0)

    caption = (
        f"Shepard diagrams for {dim}D SMACOF (MDS) embeddings of the best- and "
        f"worst-performing datasets. Each cell is an integer-aligned square bin "
        f"colored by the mean graph-pair size $\\bar{{n}} = (|V_i|+|V_j|)/2$ "
        f"(viridis scale; white = no data). Solid line: isotonic regression fit. "
        f"(a) {best_display} (highest Levenshtein $R^2$): "
        f"Levenshtein $R^2 = {r2_best_lev:.3f}$, GED $R^2 = {r2_best_ged:.3f}$. "
        f"(b) {worst_display} (lowest Levenshtein $R^2$): "
        f"Levenshtein $R^2 = {r2_worst_lev:.3f}$, GED $R^2 = {r2_worst_ged:.3f}$. "
        f"Left column: Levenshtein-based embedding; right column: GED-based embedding. "
        f"$R^2$ is computed between the original pairwise distances and the isotonic "
        f"regression of embedded Euclidean distances, measuring how faithfully the "
        f"embedding preserves the distance structure."
    )
    caption_path = os.path.join(output_dir, "individual_shepard_diagrams_caption.txt")
    with open(caption_path, "w", encoding="utf-8") as f:
        f.write(caption + "\n")
    logger.info("Caption saved: %s", caption_path)

    logger.info("H2.3 individual figure saved: %s", path)
    return path


# =====================================================================
# Orchestration
# =====================================================================


def generate_all_embedding_individual(
    results: AllResults,
    emb: EmbeddingData,
    embedding_dir: str,
    output_root: str,
) -> None:
    """Generate all individual example figures for H2.1-H2.3."""
    # H2.1
    h2_1_dir = os.path.join(output_root, "H2_1_low_distortion")
    os.makedirs(h2_1_dir, exist_ok=True)
    generate_h2_1_individual(results, emb, embedding_dir, h2_1_dir)

    # H2.2
    h2_2_dir = os.path.join(output_root, "H2_2_geometric_agreement")
    os.makedirs(h2_2_dir, exist_ok=True)
    generate_h2_2_individual(results, emb, embedding_dir, h2_2_dir)

    # H2.3
    h2_3_dir = os.path.join(output_root, "H2_3_shepard_fidelity")
    os.makedirs(h2_3_dir, exist_ok=True)
    generate_h2_3_individual(results, emb, embedding_dir, h2_3_dir)
