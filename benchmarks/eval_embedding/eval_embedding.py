"""CLI orchestrator for MDS embedding quality analysis.

Embeds Levenshtein and GED distance matrices into Euclidean space via
SMACOF MDS, then compares the resulting configurations with Procrustes
analysis and Shepard diagrams.

Usage:
    python -m benchmarks.eval_embedding.eval_embedding \
        --data-root data/eval \
        --output-dir results/eval_embedding \
        --dimensions 2,3,5,10 \
        --seed 42 --plot --table
"""
# ruff: noqa: N803, N806

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

from benchmarks.eval_correlation.correlation_metrics import (
    holm_bonferroni,
)
from benchmarks.eval_embedding.embedding_methods import (
    CMDSResult,
    classical_mds_eigenanalysis,
    find_finite_submatrix,
    procrustes_permutation_test,
    shepard_data,
    smacof,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

ALL_DATASETS = ["iam_letter_low", "iam_letter_med", "iam_letter_high", "linux", "aids"]
METHODS = ["exhaustive", "greedy"]
LABELED_DATASETS = {"iam_letter_low", "iam_letter_med", "iam_letter_high"}

DATASET_DISPLAY = {
    "iam_letter_low": "IAM Letter LOW",
    "iam_letter_med": "IAM Letter MED",
    "iam_letter_high": "IAM Letter HIGH",
    "linux": "LINUX",
    "aids": "AIDS",
}

# IAM Letter class labels (9 classes: A-I mapped to integers)
IAM_CLASS_NAMES = ["A", "E", "F", "H", "I", "L", "M", "N", "W"]

DEFAULT_DATA_ROOT = "/media/mpascual/Sandisk2TB/research/isalgraph/data/eval"
DEFAULT_OUTPUT_DIR = "/media/mpascual/Sandisk2TB/research/isalgraph/results/eval_embedding"

PROCRUSTES_DIMS = [2, 5]
SHEPARD_DIMS = [2, 5]


# =============================================================================
# Data loading (mirrors eval_correlation pattern)
# =============================================================================


def _load_dataset_artifacts(
    data_root: str,
    dataset: str,
) -> dict:
    """Load all artifacts for a dataset.

    Args:
        data_root: Root directory of eval pipeline output.
        dataset: Dataset name.

    Returns:
        Dict with ged_matrix, labels, and per-method levenshtein matrices.
    """
    artifacts: dict = {"dataset": dataset}

    # GED matrix
    ged_path = os.path.join(data_root, "ged_matrices", f"{dataset}.npz")
    ged_data = np.load(ged_path, allow_pickle=True)
    artifacts["ged_matrix"] = ged_data["ged_matrix"]
    artifacts["graph_ids"] = list(ged_data["graph_ids"])
    artifacts["labels"] = list(ged_data["labels"])

    # Levenshtein matrices
    artifacts["lev_matrices"] = {}
    for method in METHODS:
        lev_path = os.path.join(data_root, "levenshtein_matrices", f"{dataset}_{method}.npz")
        if os.path.exists(lev_path):
            lev_data = np.load(lev_path, allow_pickle=True)
            artifacts["lev_matrices"][method] = lev_data["levenshtein_matrix"]

    return artifacts


# =============================================================================
# Distance matrix preparation
# =============================================================================


def _prepare_ged_matrix(ged: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Prepare GED matrix for SMACOF: replace inf, build weight matrix.

    Args:
        ged: Raw GED matrix (may contain inf).

    Returns:
        (D_sanitized, W) where inf entries are replaced with max_finite*2
        and W[i,j]=0 for originally-inf pairs, 1 otherwise.
    """
    finite_mask = np.isfinite(ged) & (ged >= 0)
    W = finite_mask.astype(np.float64)
    np.fill_diagonal(W, 0.0)

    D = ged.copy().astype(np.float64)  # noqa: N806
    max_finite = float(np.max(ged[finite_mask])) if finite_mask.any() else 1.0
    D[~finite_mask] = max_finite * 2.0
    np.fill_diagonal(D, 0.0)

    return D, W


# =============================================================================
# Per-dataset analysis
# =============================================================================


def _analyze_dataset(
    artifacts: dict,
    dimensions: list[int],
    n_procrustes_perms: int,
    smacof_max_iter: int,
    smacof_n_init: int,
    seed: int,
) -> dict:
    """Run full embedding analysis for one dataset.

    Args:
        artifacts: Output of _load_dataset_artifacts().
        dimensions: List of embedding dimensions (e.g. [2, 3, 5, 10]).
        n_procrustes_perms: Number of Procrustes permutations.
        smacof_max_iter: SMACOF max iterations.
        smacof_n_init: SMACOF number of restarts.
        seed: Random seed.

    Returns:
        Tuple of (stats_dict, coords_cache) where coords_cache maps
        (source_key, dim) -> coordinate array.
    """
    dataset = artifacts["dataset"]
    ged_raw = artifacts["ged_matrix"]
    t0 = time.perf_counter()
    logger.info("Analyzing embedding for %s...", dataset)

    result: dict = {
        "dataset": dataset,
        "n_graphs": ged_raw.shape[0],
    }

    # Prepare GED
    D_ged, W_ged = _prepare_ged_matrix(ged_raw)
    has_inf = bool((~np.isfinite(ged_raw)).any())
    result["has_inf_ged"] = has_inf

    # Find finite submatrix for cMDS/Procrustes
    finite_idx = find_finite_submatrix(ged_raw)
    result["n_finite_submatrix"] = len(finite_idx)
    logger.info("  Finite submatrix: %d / %d graphs", len(finite_idx), ged_raw.shape[0])

    # cMDS eigenanalysis on finite submatrix
    D_finite_ged = ged_raw[np.ix_(finite_idx, finite_idx)].astype(np.float64)
    cmds_ged = classical_mds_eigenanalysis(D_finite_ged)
    result["cmds_ged"] = {
        "nev_ratio": round(cmds_ged.nev_ratio, 6),
        "n_positive": cmds_ged.n_positive,
        "n_negative": cmds_ged.n_negative,
    }

    # Per-method analysis
    result["methods"] = {}
    coords_cache: dict[tuple[str, int], np.ndarray] = {}

    # GED SMACOF (shared across methods, run once)
    for dim in dimensions:
        logger.info("    SMACOF GED %s d=%d...", dataset, dim)
        ged_smacof = smacof(
            D_ged,
            n_components=dim,
            max_iter=smacof_max_iter,
            eps=1e-6,
            n_init=smacof_n_init,
            seed=seed,
            weights=W_ged,
        )
        coords_cache[("ged", dim)] = ged_smacof.coords
        # Store GED stress in result (method-independent)
        result.setdefault("smacof_ged", {})[str(dim)] = {
            "stress_1": round(ged_smacof.stress_1, 6),
            "n_iterations": ged_smacof.n_iterations,
            "converged": ged_smacof.converged,
        }

    for method in METHODS:
        if method not in artifacts["lev_matrices"]:
            logger.warning("  No %s Levenshtein matrix for %s, skipping.", method, dataset)
            continue

        lev = artifacts["lev_matrices"][method].astype(np.float64)
        method_result, method_coords = _analyze_method(
            dataset=dataset,
            method=method,
            D_ged=D_ged,
            W_ged=W_ged,
            lev=lev,
            finite_idx=finite_idx,
            D_finite_ged=D_finite_ged,
            cmds_ged=cmds_ged,
            dimensions=dimensions,
            coords_ged=coords_cache,
            n_procrustes_perms=n_procrustes_perms,
            smacof_max_iter=smacof_max_iter,
            smacof_n_init=smacof_n_init,
            seed=seed,
        )
        result["methods"][method] = method_result
        coords_cache.update(method_coords)

    elapsed = time.perf_counter() - t0
    result["analysis_time_s"] = round(elapsed, 1)
    logger.info("  Done in %.1fs.", elapsed)
    return result, coords_cache


def _analyze_method(
    *,
    dataset: str,
    method: str,
    D_ged: np.ndarray,
    W_ged: np.ndarray,
    lev: np.ndarray,
    finite_idx: np.ndarray,
    D_finite_ged: np.ndarray,
    cmds_ged: CMDSResult,
    dimensions: list[int],
    coords_ged: dict[tuple[str, int], np.ndarray],
    n_procrustes_perms: int,
    smacof_max_iter: int,
    smacof_n_init: int,
    seed: int,
) -> tuple[dict, dict[tuple[str, int], np.ndarray]]:
    """Analyze a single (dataset, method) pair.

    Args:
        dataset: Dataset name.
        method: "exhaustive" or "greedy".
        D_ged: Sanitized GED matrix (inf replaced).
        W_ged: Weight matrix for GED.
        lev: Levenshtein distance matrix.
        finite_idx: Indices of the all-finite GED submatrix.
        D_finite_ged: GED submatrix restricted to finite_idx.
        cmds_ged: cMDS result for GED.
        dimensions: List of embedding dimensions.
        coords_ged: Pre-computed GED coordinates keyed by ("ged", dim).
        n_procrustes_perms: Number of Procrustes permutations.
        smacof_max_iter: SMACOF max iterations.
        smacof_n_init: SMACOF restarts.
        seed: Random seed.

    Returns:
        Tuple of (stats_dict, coords_dict) where coords_dict maps
        (source_key, dim) -> coordinate array for Lev embeddings.
    """
    method_result: dict = {"method": method}
    method_coords: dict[tuple[str, int], np.ndarray] = {}

    # cMDS eigenanalysis for Levenshtein (on finite submatrix)
    D_finite_lev = lev[np.ix_(finite_idx, finite_idx)]
    cmds_lev = classical_mds_eigenanalysis(D_finite_lev)
    method_result["cmds_lev"] = {
        "nev_ratio": round(cmds_lev.nev_ratio, 6),
        "n_positive": cmds_lev.n_positive,
        "n_negative": cmds_lev.n_negative,
    }

    # SMACOF at each dimension (Lev only; GED already computed)
    method_result["smacof"] = {}
    source_key = f"lev_{method}"

    for dim in dimensions:
        logger.info("    SMACOF %s %s d=%d...", dataset, method, dim)

        # Lev embedding (no weights needed, Lev has no inf)
        lev_smacof = smacof(
            lev,
            n_components=dim,
            max_iter=smacof_max_iter,
            eps=1e-6,
            n_init=smacof_n_init,
            seed=seed,
        )
        method_coords[(source_key, dim)] = lev_smacof.coords

        method_result["smacof"][str(dim)] = {
            "lev_stress_1": round(lev_smacof.stress_1, 6),
            "lev_n_iterations": lev_smacof.n_iterations,
            "lev_converged": lev_smacof.converged,
        }

    # Procrustes analysis at selected dimensions
    method_result["procrustes"] = {}
    for dim in PROCRUSTES_DIMS:
        ged_key = ("ged", dim)
        lev_key = (source_key, dim)
        if ged_key not in coords_ged or lev_key not in method_coords:
            continue
        logger.info(
            "    Procrustes %s %s d=%d (%d perms)...", dataset, method, dim, n_procrustes_perms
        )

        # Restrict both embeddings to finite submatrix indices
        coords_ged_sub = coords_ged[ged_key][finite_idx]
        coords_lev_sub = method_coords[lev_key][finite_idx]

        proc = procrustes_permutation_test(
            coords_ged_sub,
            coords_lev_sub,
            n_permutations=n_procrustes_perms,
            seed=seed,
        )
        method_result["procrustes"][str(dim)] = {
            "m_squared": round(proc.m_squared, 6),
            "p_value": proc.p_value,
            "n_more_extreme": proc.n_more_extreme,
        }

    # Shepard diagrams at selected dimensions
    method_result["shepard"] = {}
    for dim in SHEPARD_DIMS:
        ged_key = ("ged", dim)
        lev_key = (source_key, dim)
        if ged_key not in coords_ged or lev_key not in method_coords:
            continue

        # GED Shepard (weighted)
        shep_ged = shepard_data(D_ged, coords_ged[ged_key], weights=W_ged)
        # Lev Shepard
        shep_lev = shepard_data(lev, method_coords[lev_key])

        method_result["shepard"][str(dim)] = {
            "ged_r_squared": round(shep_ged.r_squared, 6),
            "ged_monotonic_r_squared": round(shep_ged.monotonic_r_squared, 6),
            "lev_r_squared": round(shep_lev.r_squared, 6),
            "lev_monotonic_r_squared": round(shep_lev.monotonic_r_squared, 6),
        }

    return method_result, method_coords


# =============================================================================
# Cross-dataset analysis
# =============================================================================


def _cross_dataset_analysis(
    all_stats: dict[str, dict],
) -> dict:
    """Aggregate results across datasets.

    Args:
        all_stats: Mapping of dataset -> per-dataset stats.

    Returns:
        Dict with cross-dataset summary.
    """
    result: dict = {}

    # Aggregate Procrustes p-values for Holm-Bonferroni correction
    proc_ps = []
    proc_keys = []
    for ds, ds_stats in sorted(all_stats.items()):
        for method, m_stats in ds_stats.get("methods", {}).items():
            for dim, p_stats in m_stats.get("procrustes", {}).items():
                proc_ps.append(p_stats["p_value"])
                proc_keys.append(f"{ds}_{method}_d{dim}")

    if proc_ps:
        adjusted = holm_bonferroni(proc_ps)
        result["procrustes_multiple_testing"] = {
            "method": "holm_bonferroni",
            "n_tests": len(proc_ps),
            "raw_p_values": dict(zip(proc_keys, proc_ps, strict=True)),
            "adjusted_p_values": dict(zip(proc_keys, [round(p, 6) for p in adjusted], strict=True)),
        }

    # Stress summary table
    stress_rows = []
    for ds, ds_stats in sorted(all_stats.items()):
        ged_smacof = ds_stats.get("smacof_ged", {})
        for method, m_stats in ds_stats.get("methods", {}).items():
            for dim, s_stats in m_stats.get("smacof", {}).items():
                ged_stress = ged_smacof.get(dim, {}).get("stress_1", None)
                stress_rows.append(
                    {
                        "dataset": ds,
                        "method": method,
                        "dimension": int(dim),
                        "ged_stress_1": ged_stress,
                        "lev_stress_1": s_stats["lev_stress_1"],
                    }
                )
    result["stress_summary"] = stress_rows

    # NEV ratio comparison
    nev_rows = []
    for ds, ds_stats in sorted(all_stats.items()):
        ged_nev = ds_stats.get("cmds_ged", {}).get("nev_ratio")
        for method, m_stats in ds_stats.get("methods", {}).items():
            lev_nev = m_stats.get("cmds_lev", {}).get("nev_ratio")
            if ged_nev is not None and lev_nev is not None:
                nev_rows.append(
                    {
                        "dataset": ds,
                        "method": method,
                        "ged_nev_ratio": ged_nev,
                        "lev_nev_ratio": lev_nev,
                    }
                )
    result["nev_comparison"] = nev_rows

    return result


# =============================================================================
# Figure generation
# =============================================================================


def _generate_figures(
    all_stats: dict[str, dict],
    all_artifacts: dict[str, dict],
    cross_stats: dict,
    smacof_cache: dict[tuple[str, str, int], np.ndarray],
    output_dir: str,
) -> None:
    """Generate publication-quality figures.

    Args:
        all_stats: Per-dataset statistics.
        all_artifacts: Loaded data artifacts.
        cross_stats: Cross-dataset analysis results.
        smacof_cache: Cache of SMACOF coordinates keyed by (dataset, source, dim).
        output_dir: Output directory.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from benchmarks.plotting_styles import (
        PAUL_TOL_BRIGHT,
        PAUL_TOL_MUTED,
        apply_ieee_style,
        save_figure,
    )

    apply_ieee_style()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    ds_colors = {
        "iam_letter_low": PAUL_TOL_BRIGHT["blue"],
        "iam_letter_med": PAUL_TOL_BRIGHT["cyan"],
        "iam_letter_high": PAUL_TOL_BRIGHT["green"],
        "linux": PAUL_TOL_BRIGHT["red"],
        "aids": PAUL_TOL_BRIGHT["purple"],
    }

    # ---- Main figure (2x3) ----
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.0))

    # (a) 2D MDS scatter of Lev exhaustive (IAM LOW), colored by class
    _plot_mds_scatter(
        axes[0, 0],
        smacof_cache,
        all_artifacts,
        "iam_letter_low",
        "lev_exhaustive",
        "(a) MDS Lev (IAM LOW)",
        PAUL_TOL_MUTED,
    )

    # (b) 2D MDS scatter of GED (IAM LOW), same coloring
    _plot_mds_scatter(
        axes[0, 1],
        smacof_cache,
        all_artifacts,
        "iam_letter_low",
        "ged",
        "(b) MDS GED (IAM LOW)",
        PAUL_TOL_MUTED,
    )

    # (c) Scree plot: eigenvalues from cMDS
    _plot_scree(axes[0, 2], all_stats, "(c)")

    # (d) Stress-1 vs dimension
    _plot_stress_vs_dim(axes[1, 0], all_stats, ds_colors, "(d)")

    # (e) Shepard: Lev exhaustive d=2
    _plot_shepard(
        axes[1, 1],
        smacof_cache,
        all_artifacts,
        "iam_letter_low",
        "lev_exhaustive",
        all_stats,
        "(e) Shepard Lev",
    )

    # (f) Shepard: GED d=2
    _plot_shepard(
        axes[1, 2],
        smacof_cache,
        all_artifacts,
        "iam_letter_low",
        "ged",
        all_stats,
        "(f) Shepard GED",
    )

    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, "embedding_main_figure"))
    plt.close(fig)
    logger.info("Saved embedding main figure.")

    # ---- Individual Shepard diagrams ----
    for ds, art in all_artifacts.items():
        for method in METHODS:
            if method not in art.get("lev_matrices", {}):
                continue
            for source_key in [f"lev_{method}", "ged"]:
                key = (ds, source_key, 2)
                if key not in smacof_cache:
                    continue
                fig_s, ax_s = plt.subplots(figsize=(3.39, 2.5))
                _plot_shepard_single(ax_s, smacof_cache, all_artifacts, ds, source_key, all_stats)
                fig_s.tight_layout()
                save_figure(fig_s, os.path.join(fig_dir, f"shepard_{ds}_{source_key}"))
                plt.close(fig_s)

    # ---- Individual scree plots ----
    for ds in all_stats:
        fig_sc, ax_sc = plt.subplots(figsize=(3.39, 2.5))
        _plot_scree_single(ax_sc, all_stats, ds)
        fig_sc.tight_layout()
        save_figure(fig_sc, os.path.join(fig_dir, f"scree_{ds}"))
        plt.close(fig_sc)

    # ---- Individual MDS scatter plots ----
    for ds in all_artifacts:
        for method in METHODS:
            for source_key in [f"lev_{method}", "ged"]:
                key = (ds, source_key, 2)
                if key not in smacof_cache:
                    continue
                fig_m, ax_m = plt.subplots(figsize=(3.39, 3.0))
                _plot_mds_scatter(
                    ax_m,
                    smacof_cache,
                    all_artifacts,
                    ds,
                    source_key,
                    f"{DATASET_DISPLAY.get(ds, ds)} ({source_key})",
                    PAUL_TOL_MUTED,
                )
                fig_m.tight_layout()
                save_figure(fig_m, os.path.join(fig_dir, f"mds_scatter_{ds}_{source_key}"))
                plt.close(fig_m)


def _plot_mds_scatter(
    ax: object,
    smacof_cache: dict,
    all_artifacts: dict,
    dataset: str,
    source_key: str,
    title: str,
    class_colors: list[str],
) -> None:
    """2D MDS scatter plot, colored by class label if available."""
    from benchmarks.plotting_styles import PLOT_SETTINGS

    key = (dataset, source_key, 2)
    if key not in smacof_cache:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)  # type: ignore[union-attr]
        return

    coords = smacof_cache[key]
    art = all_artifacts.get(dataset, {})
    labels = art.get("labels", [])

    if labels and any(str(lab) != "" for lab in labels):
        unique_labels = sorted(set(str(lbl) for lbl in labels if str(lbl) != ""))
        for i, lab in enumerate(unique_labels):
            mask = np.array([str(lbl) == lab for lbl in labels])
            # Adjust mask size if coords are from finite submatrix
            if mask.sum() > 0 and coords.shape[0] == len(labels):
                color = class_colors[i % len(class_colors)]
                ax.scatter(  # type: ignore[union-attr]
                    coords[mask, 0],
                    coords[mask, 1],
                    s=PLOT_SETTINGS["scatter_size"],
                    alpha=PLOT_SETTINGS["scatter_alpha"],
                    color=color,
                    label=lab,
                    edgecolors="none",
                )
        ax.legend(fontsize=6, ncol=3, loc="best", markerscale=0.8)  # type: ignore[union-attr]
    else:
        ax.scatter(  # type: ignore[union-attr]
            coords[:, 0],
            coords[:, 1],
            s=PLOT_SETTINGS["scatter_size"],
            alpha=PLOT_SETTINGS["scatter_alpha"],
            color="#4477AA",
            edgecolors="none",
        )

    ax.set_xlabel("MDS dim 1")  # type: ignore[union-attr]
    ax.set_ylabel("MDS dim 2")  # type: ignore[union-attr]
    ax.set_title(title, fontsize=PLOT_SETTINGS["axes_titlesize"] - 2)  # type: ignore[union-attr]


def _plot_scree(ax: object, all_stats: dict, panel_label: str) -> None:
    """Scree plot: top eigenvalues from cMDS for GED + Lev."""

    # Use first dataset that has data
    for ds in ["iam_letter_low", "iam_letter_med", "iam_letter_high", "linux", "aids"]:
        if ds not in all_stats:
            continue
        ds_stats = all_stats[ds]
        ged_nev = ds_stats.get("cmds_ged", {})
        if not ged_nev:
            continue

        # Plot GED eigenvalues (stored in raw NPZ, use what we have)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="-")  # type: ignore[union-attr]
        ax.set_xlabel("Eigenvalue index")  # type: ignore[union-attr]
        ax.set_ylabel("Eigenvalue")  # type: ignore[union-attr]
        ax.set_title(f"{panel_label} Scree ({DATASET_DISPLAY.get(ds, ds)})")  # type: ignore[union-attr]

        # Annotate NEV ratios
        ged_nev_val = ged_nev.get("nev_ratio", 0)
        text_lines = [f"GED NEV={ged_nev_val:.3f}"]
        for method, m_stats in ds_stats.get("methods", {}).items():
            lev_nev = m_stats.get("cmds_lev", {}).get("nev_ratio", 0)
            text_lines.append(f"Lev({method[:3]}) NEV={lev_nev:.3f}")
        ax.text(  # type: ignore[union-attr]
            0.95,
            0.95,
            "\n".join(text_lines),
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=7,  # type: ignore[union-attr]
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
        break


def _plot_scree_single(ax: object, all_stats: dict, dataset: str) -> None:
    """Scree plot for a single dataset (for individual figures)."""
    ds_stats = all_stats.get(dataset, {})
    ax.axhline(0, color="black", linewidth=0.5, linestyle="-")  # type: ignore[union-attr]
    ax.set_xlabel("Eigenvalue index")  # type: ignore[union-attr]
    ax.set_ylabel("Eigenvalue")  # type: ignore[union-attr]
    ax.set_title(f"Scree: {DATASET_DISPLAY.get(dataset, dataset)}")  # type: ignore[union-attr]

    ged_nev = ds_stats.get("cmds_ged", {}).get("nev_ratio", 0)
    text_lines = [f"GED NEV={ged_nev:.3f}"]
    for method, m_stats in ds_stats.get("methods", {}).items():
        lev_nev = m_stats.get("cmds_lev", {}).get("nev_ratio", 0)
        text_lines.append(f"Lev({method[:3]}) NEV={lev_nev:.3f}")
    ax.text(  # type: ignore[union-attr]
        0.95,
        0.95,
        "\n".join(text_lines),
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=7,  # type: ignore[union-attr]
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )


def _plot_stress_vs_dim(
    ax: object,
    all_stats: dict,
    ds_colors: dict,
    panel_label: str,
) -> None:
    """Stress-1 vs embedding dimension, with threshold lines."""
    from benchmarks.plotting_styles import PLOT_SETTINGS

    plotted = False
    for ds, ds_stats in sorted(all_stats.items()):
        for method, m_stats in ds_stats.get("methods", {}).items():
            smacof_stats = m_stats.get("smacof", {})
            if not smacof_stats:
                continue
            dims = sorted(int(d) for d in smacof_stats)
            stresses = [smacof_stats[str(d)]["lev_stress_1"] for d in dims]
            style = "-" if method == "exhaustive" else "--"
            color = ds_colors.get(ds, "#999999")
            label = f"{DATASET_DISPLAY.get(ds, ds)[:8]} {method[:3]}"
            ax.plot(  # type: ignore[union-attr]
                dims,
                stresses,
                style,
                color=color,
                linewidth=PLOT_SETTINGS["line_width"],
                marker="o",
                markersize=3,
                label=label,
            )
            plotted = True

    if plotted:
        # Threshold lines
        ax.axhline(0.20, color="grey", linestyle=":", linewidth=0.8, alpha=0.7)  # type: ignore[union-attr]
        ax.axhline(0.10, color="grey", linestyle=":", linewidth=0.8, alpha=0.7)  # type: ignore[union-attr]
        ax.text(0.5, 0.21, "0.20", fontsize=7, color="grey", transform=ax.get_yaxis_transform())  # type: ignore[union-attr]
        ax.text(0.5, 0.11, "0.10", fontsize=7, color="grey", transform=ax.get_yaxis_transform())  # type: ignore[union-attr]

    ax.set_xlabel("Embedding dimension")  # type: ignore[union-attr]
    ax.set_ylabel("Stress-1")  # type: ignore[union-attr]
    ax.set_title(f"{panel_label} Stress vs dimension")  # type: ignore[union-attr]
    if plotted:
        ax.legend(fontsize=5, ncol=2, loc="upper right")  # type: ignore[union-attr]


def _plot_shepard(
    ax: object,
    smacof_cache: dict,
    all_artifacts: dict,
    dataset: str,
    source_key: str,
    all_stats: dict,
    title: str,
) -> None:
    """Shepard diagram (hexbin) for a main-figure panel."""
    _plot_shepard_single(ax, smacof_cache, all_artifacts, dataset, source_key, all_stats)
    ax.set_title(title, fontsize=9)  # type: ignore[union-attr]


def _plot_shepard_single(
    ax: object,
    smacof_cache: dict,
    all_artifacts: dict,
    dataset: str,
    source_key: str,
    all_stats: dict,
) -> None:
    """Single Shepard diagram (hexbin)."""
    import matplotlib.pyplot as plt

    key = (dataset, source_key, 2)
    if key not in smacof_cache:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)  # type: ignore[union-attr]
        return

    coords = smacof_cache[key]
    art = all_artifacts.get(dataset, {})

    if source_key == "ged":
        D_raw = art["ged_matrix"]
        D, W = _prepare_ged_matrix(D_raw)
        shep = shepard_data(D, coords, weights=W)
    else:
        # lev_exhaustive or lev_greedy
        method = source_key.replace("lev_", "")
        D = art["lev_matrices"][method].astype(np.float64)
        shep = shepard_data(D, coords)

    if len(shep.original_distances) < 10:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)  # type: ignore[union-attr]
        return

    hb = ax.hexbin(  # type: ignore[union-attr]
        shep.original_distances,
        shep.embedded_distances,
        gridsize=40,
        cmap="YlOrRd",
        mincnt=1,
    )
    plt.colorbar(hb, ax=ax, shrink=0.7)

    ax.set_xlabel("Original distance")  # type: ignore[union-attr]
    ax.set_ylabel("Embedded distance")  # type: ignore[union-attr]
    ax.text(  # type: ignore[union-attr]
        0.05,
        0.95,
        f"$R^2$={shep.r_squared:.3f}\n$\\rho^2$={shep.monotonic_r_squared:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=7,  # type: ignore[union-attr]
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )


# =============================================================================
# Table generation
# =============================================================================


def _generate_tables(
    all_stats: dict[str, dict],
    cross_stats: dict,
    output_dir: str,
) -> None:
    """Generate LaTeX summary table.

    Args:
        all_stats: Per-dataset statistics.
        cross_stats: Cross-dataset analysis results.
        output_dir: Output directory.
    """
    import pandas as pd

    from benchmarks.plotting_styles import save_latex_table

    table_dir = os.path.join(output_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)

    rows = []
    for ds in ALL_DATASETS:
        ds_stats = all_stats.get(ds, {})
        n_graphs = ds_stats.get("n_graphs", "")

        for method, m_stats in ds_stats.get("methods", {}).items():
            smacof_stats = m_stats.get("smacof", {})
            stress_2d = smacof_stats.get("2", {}).get("lev_stress_1", "")
            stress_5d = smacof_stats.get("5", {}).get("lev_stress_1", "")

            nev = m_stats.get("cmds_lev", {}).get("nev_ratio", "")

            proc_2d = m_stats.get("procrustes", {}).get("2", {})
            proc_m2 = proc_2d.get("m_squared", "")
            proc_p = proc_2d.get("p_value", "")

            shep_2d = m_stats.get("shepard", {}).get("2", {})
            shep_r2 = shep_2d.get("lev_r_squared", "")

            rows.append(
                {
                    "Dataset": DATASET_DISPLAY.get(ds, ds),
                    "N": n_graphs,
                    "Source": method.capitalize(),
                    "Stress-1 (2D)": f"{stress_2d:.4f}" if isinstance(stress_2d, float) else "",
                    "Stress-1 (5D)": f"{stress_5d:.4f}" if isinstance(stress_5d, float) else "",
                    "NEV ratio": f"{nev:.4f}" if isinstance(nev, float) else "",
                    "Procrustes $m^2$ (2D)": f"{proc_m2:.4f}" if isinstance(proc_m2, float) else "",
                    "Procrustes $p$": _fmt_p(proc_p) if isinstance(proc_p, float) else "",
                    "Shepard $R^2$ (2D)": f"{shep_r2:.4f}" if isinstance(shep_r2, float) else "",
                }
            )

    if rows:
        df = pd.DataFrame(rows)
        save_latex_table(
            df,
            os.path.join(table_dir, "embedding_summary.tex"),
            caption=(
                "MDS embedding quality: Stress-1 at 2D and 5D, negative eigenvalue"
                " ratio (NEV), Procrustes $m^2$ between GED and Levenshtein embeddings,"
                " and Shepard $R^2$ for Levenshtein embeddings."
            ),
            label="tab:embedding_summary",
        )
        logger.info("Saved embedding_summary.tex")


def _fmt_p(p: float | None) -> str:
    """Format p-value."""
    if p is None:
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


# =============================================================================
# Pipeline
# =============================================================================


def run_pipeline(
    data_root: str,
    output_dir: str,
    datasets: list[str],
    dimensions: list[int],
    n_procrustes_perms: int,
    smacof_max_iter: int,
    smacof_n_init: int,
    seed: int,
    save_plots: bool,
    save_tables: bool,
) -> None:
    """Run the full embedding analysis pipeline.

    Args:
        data_root: Root directory of eval pipeline output.
        output_dir: Output directory for results.
        datasets: List of dataset names.
        dimensions: Embedding dimensions (e.g. [2, 3, 5, 10]).
        n_procrustes_perms: Procrustes permutations.
        smacof_max_iter: SMACOF max iterations.
        smacof_n_init: SMACOF restarts.
        seed: Random seed.
        save_plots: Whether to generate figures.
        save_tables: Whether to generate LaTeX tables.
    """
    t0 = time.perf_counter()
    os.makedirs(output_dir, exist_ok=True)
    stats_dir = os.path.join(output_dir, "stats")
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    all_stats: dict[str, dict] = {}
    all_artifacts: dict[str, dict] = {}
    smacof_cache: dict[tuple[str, str, int], np.ndarray] = {}

    for ds in datasets:
        logger.info("Loading artifacts for %s...", ds)
        try:
            artifacts = _load_dataset_artifacts(data_root, ds)
        except FileNotFoundError as e:
            logger.error("Missing data for %s: %s", ds, e)
            continue

        all_artifacts[ds] = artifacts

        ds_stats, ds_coords = _analyze_dataset(
            artifacts,
            dimensions=dimensions,
            n_procrustes_perms=n_procrustes_perms,
            smacof_max_iter=smacof_max_iter,
            smacof_n_init=smacof_n_init,
            seed=seed,
        )
        all_stats[ds] = ds_stats

        # Save per-dataset stats JSON
        stats_path = os.path.join(stats_dir, f"{ds}_embedding_stats.json")
        with open(stats_path, "w") as f:
            json.dump(ds_stats, f, indent=2, default=str)
        logger.info("Saved %s", stats_path)

        # Promote local coords to global cache and save raw NPZ
        for (source_key, dim), coords in ds_coords.items():
            smacof_cache[(ds, source_key, dim)] = coords
            np.savez_compressed(
                os.path.join(raw_dir, f"{ds}_{source_key}_smacof_{dim}d.npz"),
                coords=coords,
            )

        # Save cMDS eigenvalues
        ged_raw = artifacts["ged_matrix"]
        finite_idx = find_finite_submatrix(ged_raw)
        D_finite = ged_raw[np.ix_(finite_idx, finite_idx)].astype(np.float64)
        cmds_ged = classical_mds_eigenanalysis(D_finite)
        np.savez_compressed(
            os.path.join(raw_dir, f"{ds}_cmds_eigenvalues.npz"),
            ged_eigenvalues=cmds_ged.eigenvalues,
            finite_idx=finite_idx,
        )

    # Cross-dataset analysis
    if len(all_stats) > 0:
        logger.info("Running cross-dataset analysis...")
        cross_stats = _cross_dataset_analysis(all_stats)
        cross_path = os.path.join(stats_dir, "cross_dataset_analysis.json")
        with open(cross_path, "w") as f:
            json.dump(cross_stats, f, indent=2, default=str)
        logger.info("Saved %s", cross_path)

        # Summary JSON
        summary_path = os.path.join(stats_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump({"datasets": list(all_stats.keys()), "dimensions": dimensions}, f, indent=2)
    else:
        cross_stats = {}

    # Figures
    if save_plots and len(all_stats) > 0:
        logger.info("Generating figures...")
        _generate_figures(all_stats, all_artifacts, cross_stats, smacof_cache, output_dir)

    # Tables
    if save_tables and len(all_stats) > 0:
        logger.info("Generating tables...")
        _generate_tables(all_stats, cross_stats, output_dir)

    elapsed = time.perf_counter() - t0
    logger.info("Pipeline complete in %.1fs.", elapsed)


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="MDS embedding quality analysis for IsalGraph evaluation."
    )
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Comma-separated dataset names, or 'all'.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dimensions",
        type=str,
        default="2,3,5,10",
        help="Comma-separated embedding dimensions.",
    )
    parser.add_argument("--n-procrustes-perms", type=int, default=9999)
    parser.add_argument("--smacof-max-iter", type=int, default=300)
    parser.add_argument("--smacof-n-init", type=int, default=4)
    parser.add_argument(
        "--mode",
        choices=["local", "picasso"],
        default="local",
    )
    parser.add_argument("--plot", action="store_true", help="Generate figures.")
    parser.add_argument("--table", action="store_true", help="Generate LaTeX tables.")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Parse datasets
    if args.datasets == "all":
        datasets = list(ALL_DATASETS)
    else:
        datasets = [d.strip() for d in args.datasets.split(",")]
        for d in datasets:
            if d not in ALL_DATASETS:
                logger.error("Unknown dataset: %s. Choose from %s", d, ALL_DATASETS)
                sys.exit(1)

    # Parse dimensions
    dimensions = [int(d.strip()) for d in args.dimensions.split(",")]

    run_pipeline(
        data_root=args.data_root,
        output_dir=args.output_dir,
        datasets=datasets,
        dimensions=dimensions,
        n_procrustes_perms=args.n_procrustes_perms,
        smacof_max_iter=args.smacof_max_iter,
        smacof_n_init=args.smacof_n_init,
        seed=args.seed,
        save_plots=args.plot,
        save_tables=args.table,
    )


if __name__ == "__main__":
    main()
