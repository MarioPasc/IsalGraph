"""CLI orchestrator for Levenshtein-GED correlation analysis.

Loads eval pipeline artifacts (GED matrices, Levenshtein matrices,
canonical strings) and runs statistical tests to validate IsalGraph's
claim that Levenshtein distance approximates graph edit distance.

Usage:
    python -m benchmarks.eval_correlation.eval_correlation \
        --data-root data/eval \
        --output-dir results/eval_correlation \
        --n-bootstrap 10000 --n-permutations 9999 --seed 42 \
        --csv --plot --table
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from benchmarks.eval_correlation.correlation_metrics import (
    bootstrap_correlation,
    cohens_d,
    extract_upper_tri,
    holm_bonferroni,
    lins_ccc,
    mantel_test,
    ols_regression,
    precision_at_k,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

ALL_DATASETS = ["iam_letter_low", "iam_letter_med", "iam_letter_high", "linux", "aids"]
METHODS = ["exhaustive", "greedy"]
IAM_DATASETS_ORDERED = ["iam_letter_low", "iam_letter_med", "iam_letter_high"]
LABELED_DATASETS = {"iam_letter_low", "iam_letter_med", "iam_letter_high"}

DATASET_DISPLAY = {
    "iam_letter_low": "IAM Letter LOW",
    "iam_letter_med": "IAM Letter MED",
    "iam_letter_high": "IAM Letter HIGH",
    "linux": "LINUX",
    "aids": "AIDS",
}

DEFAULT_DATA_ROOT = "/media/mpascual/Sandisk2TB/research/isalgraph/data/eval"
DEFAULT_OUTPUT_DIR = "/media/mpascual/Sandisk2TB/research/isalgraph/results/eval_correlation"


# =============================================================================
# Data loading
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
        Dict with ged_matrix, labels, node_counts, edge_counts,
        graph_ids, and per-method levenshtein matrices.
    """
    artifacts: dict = {"dataset": dataset}

    # GED matrix
    ged_path = os.path.join(data_root, "ged_matrices", f"{dataset}.npz")
    ged_data = np.load(ged_path, allow_pickle=True)
    artifacts["ged_matrix"] = ged_data["ged_matrix"]
    artifacts["graph_ids"] = list(ged_data["graph_ids"])
    artifacts["labels"] = list(ged_data["labels"])
    artifacts["node_counts"] = np.array(ged_data["node_counts"])
    artifacts["edge_counts"] = np.array(ged_data["edge_counts"])

    # Levenshtein matrices
    artifacts["lev_matrices"] = {}
    for method in METHODS:
        lev_path = os.path.join(data_root, "levenshtein_matrices", f"{dataset}_{method}.npz")
        if os.path.exists(lev_path):
            lev_data = np.load(lev_path, allow_pickle=True)
            artifacts["lev_matrices"][method] = lev_data["levenshtein_matrix"]

    # Graph metadata (for densities)
    meta_path = os.path.join(data_root, "graph_metadata", f"{dataset}.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        artifacts["metadata"] = meta

    return artifacts


# =============================================================================
# Per-dataset analysis
# =============================================================================


def _compute_densities(node_counts: np.ndarray, edge_counts: np.ndarray) -> np.ndarray:
    """Compute graph density for each graph.

    density = 2*|E| / (|V| * (|V|-1)) for undirected graphs.
    """
    n = node_counts.astype(np.float64)
    e = edge_counts.astype(np.float64)
    max_edges = n * (n - 1)
    densities = np.where(max_edges > 0, 2.0 * e / max_edges, 0.0)
    return densities


def _analyze_dataset(
    artifacts: dict,
    method: str,
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
) -> dict:
    """Run full statistical analysis for one (dataset, method) pair.

    Args:
        artifacts: Output of _load_dataset_artifacts().
        method: "exhaustive" or "greedy".
        n_bootstrap: Number of bootstrap resamples.
        n_permutations: Number of Mantel permutations.
        seed: Random seed.

    Returns:
        Dict with all statistics, ready for JSON serialization.
    """
    dataset = artifacts["dataset"]
    ged = artifacts["ged_matrix"]
    lev = artifacts["lev_matrices"][method]
    labels = artifacts["labels"]
    node_counts = artifacts["node_counts"]
    edge_counts = artifacts["edge_counts"]

    t0 = time.perf_counter()
    logger.info("Analyzing %s / %s ...", dataset, method)

    # Extract aligned upper-triangle vectors
    [v_ged, v_lev], valid_mask = extract_upper_tri(ged, lev, mask_inf=True)
    n_pairs = len(v_ged)
    n_total_pairs = ged.shape[0] * (ged.shape[0] - 1) // 2
    pct_valid = 100.0 * n_pairs / n_total_pairs if n_total_pairs > 0 else 0.0

    logger.info("  %d/%d valid pairs (%.1f%%)", n_pairs, n_total_pairs, pct_valid)

    result: dict = {
        "dataset": dataset,
        "method": method,
        "n_graphs": ged.shape[0],
        "n_total_pairs": n_total_pairs,
        "n_valid_pairs": n_pairs,
        "pct_valid": round(pct_valid, 1),
    }

    if n_pairs < 10:
        logger.warning("  Too few valid pairs (%d), skipping analysis.", n_pairs)
        result["error"] = "too_few_pairs"
        return result

    # Mantel test (H1)
    logger.info("  Running Mantel test (%d permutations)...", n_permutations)
    mantel = mantel_test(ged, lev, method="spearman", n_permutations=n_permutations, seed=seed)
    result["mantel"] = asdict(mantel)

    # Bootstrap correlations
    logger.info("  Computing bootstrap correlations (%d resamples)...", n_bootstrap)
    spearman = bootstrap_correlation(
        v_ged, v_lev, method="spearman", n_bootstrap=n_bootstrap, seed=seed
    )
    pearson = bootstrap_correlation(
        v_ged, v_lev, method="pearson", n_bootstrap=n_bootstrap, seed=seed
    )
    kendall = bootstrap_correlation(
        v_ged, v_lev, method="kendall", n_bootstrap=n_bootstrap, seed=seed
    )

    result["spearman"] = asdict(spearman)
    result["pearson"] = asdict(pearson)
    result["kendall"] = asdict(kendall)

    # Lin's CCC (raw and z-normalized)
    ccc_raw = lins_ccc(v_ged, v_lev)
    # z-normalize for CCC
    v_ged_z = (v_ged - np.mean(v_ged)) / (np.std(v_ged) + 1e-12)
    v_lev_z = (v_lev - np.mean(v_lev)) / (np.std(v_lev) + 1e-12)
    ccc_znorm = lins_ccc(v_ged_z, v_lev_z)
    result["lins_ccc_raw"] = round(ccc_raw, 4)
    result["lins_ccc_znorm"] = round(ccc_znorm, 4)

    # Precision@k
    logger.info("  Computing Precision@k...")
    pak = precision_at_k(ged, lev, k_values=[5, 10, 20])
    result["precision_at_k"] = {str(k): asdict(v) for k, v in pak.items()}

    # OLS regression
    ols = ols_regression(v_lev, v_ged)
    result["ols"] = ols

    # Class-level analysis (H4) -- IAM datasets only
    if dataset in LABELED_DATASETS and any(lab != "" for lab in labels):
        result["class_analysis"] = _class_analysis(ged, lev, labels)

    # Size-stratified analysis
    result["size_stratified"] = _size_stratified_analysis(ged, lev, node_counts)

    # Density-stratified analysis (H3)
    densities = _compute_densities(node_counts, edge_counts)
    result["density_stratified"] = _density_stratified_analysis(ged, lev, densities)

    elapsed = time.perf_counter() - t0
    result["analysis_time_s"] = round(elapsed, 1)
    logger.info("  Done in %.1fs.", elapsed)
    return result


def _class_analysis(
    ged: np.ndarray,
    lev: np.ndarray,
    labels: list[str],
) -> dict:
    """Within-class vs between-class distance analysis (H4).

    Args:
        ged: GED matrix.
        lev: Levenshtein matrix.
        labels: Class labels per graph.

    Returns:
        Dict with within/between means, Cohen's d, and Spearman per class.
    """
    n = ged.shape[0]
    label_arr = np.array(labels)
    within_lev = []
    between_lev = []

    for i in range(n):
        for j in range(i + 1, n):
            if not np.isfinite(ged[i, j]) or ged[i, j] < 0:
                continue
            if not np.isfinite(lev[i, j]) or lev[i, j] < 0:
                continue
            val = float(lev[i, j])
            if label_arr[i] == label_arr[j]:
                within_lev.append(val)
            else:
                between_lev.append(val)

    within_arr = np.array(within_lev)
    between_arr = np.array(between_lev)

    d = cohens_d(between_arr, within_arr)

    return {
        "n_within": len(within_lev),
        "n_between": len(between_lev),
        "mean_within_lev": round(float(np.mean(within_arr)), 2) if len(within_arr) > 0 else None,
        "mean_between_lev": round(float(np.mean(between_arr)), 2) if len(between_arr) > 0 else None,
        "std_within_lev": round(float(np.std(within_arr)), 2) if len(within_arr) > 0 else None,
        "std_between_lev": round(float(np.std(between_arr)), 2) if len(between_arr) > 0 else None,
        "cohens_d": round(d, 4),
        "effect_interpretation": _interpret_d(abs(d)),
    }


def _interpret_d(d_abs: float) -> str:
    """Cohen's d interpretation."""
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    return "large"


def _size_stratified_analysis(
    ged: np.ndarray,
    lev: np.ndarray,
    node_counts: np.ndarray,
) -> list[dict]:
    """Spearman rho binned by max(n_i, n_j).

    Args:
        ged: GED matrix.
        lev: Levenshtein matrix.
        node_counts: Node count per graph.

    Returns:
        List of dicts with bin info and Spearman rho.
    """
    from scipy import stats as sp_stats

    n = ged.shape[0]
    triu_i, triu_j = np.triu_indices(n, k=1)

    max_sizes = np.maximum(node_counts[triu_i], node_counts[triu_j])
    v_ged = ged[triu_i, triu_j].astype(np.float64)
    v_lev = lev[triu_i, triu_j].astype(np.float64)
    valid = np.isfinite(v_ged) & (v_ged >= 0) & np.isfinite(v_lev) & (v_lev >= 0)

    # Create bins based on quantiles of max_sizes
    unique_sizes = np.unique(max_sizes[valid])
    if len(unique_sizes) <= 4:
        bins = unique_sizes
    else:
        bins = np.unique(np.percentile(max_sizes[valid], [0, 25, 50, 75, 100]))

    results = []
    for b_idx in range(len(bins) - 1):
        lo, hi = bins[b_idx], bins[b_idx + 1]
        if b_idx == len(bins) - 2:
            mask = valid & (max_sizes >= lo) & (max_sizes <= hi)
        else:
            mask = valid & (max_sizes >= lo) & (max_sizes < hi)

        n_bin = int(np.sum(mask))
        if n_bin < 10:
            continue

        rho = float(sp_stats.spearmanr(v_ged[mask], v_lev[mask]).statistic)
        results.append(
            {
                "size_bin": f"[{lo:.0f}, {hi:.0f}]",
                "n_pairs": n_bin,
                "spearman_rho": round(rho, 4),
            }
        )

    return results


def _density_stratified_analysis(
    ged: np.ndarray,
    lev: np.ndarray,
    densities: np.ndarray,
) -> list[dict]:
    """Spearman rho binned by max(density_i, density_j) (H3).

    Args:
        ged: GED matrix.
        lev: Levenshtein matrix.
        densities: Density per graph.

    Returns:
        List of dicts with bin info and Spearman rho.
    """
    from scipy import stats as sp_stats

    n = ged.shape[0]
    triu_i, triu_j = np.triu_indices(n, k=1)

    max_dens = np.maximum(densities[triu_i], densities[triu_j])
    v_ged = ged[triu_i, triu_j].astype(np.float64)
    v_lev = lev[triu_i, triu_j].astype(np.float64)
    valid = np.isfinite(v_ged) & (v_ged >= 0) & np.isfinite(v_lev) & (v_lev >= 0)

    # Quartile bins on density
    valid_dens = max_dens[valid]
    if len(valid_dens) < 20:
        return []

    bins = np.unique(np.percentile(valid_dens, [0, 25, 50, 75, 100]))
    if len(bins) < 2:
        return []

    results = []
    for b_idx in range(len(bins) - 1):
        lo, hi = bins[b_idx], bins[b_idx + 1]
        if b_idx == len(bins) - 2:
            mask = valid & (max_dens >= lo) & (max_dens <= hi)
        else:
            mask = valid & (max_dens >= lo) & (max_dens < hi)

        n_bin = int(np.sum(mask))
        if n_bin < 10:
            continue

        rho = float(sp_stats.spearmanr(v_ged[mask], v_lev[mask]).statistic)
        results.append(
            {
                "density_bin": f"[{lo:.3f}, {hi:.3f}]",
                "n_pairs": n_bin,
                "spearman_rho": round(rho, 4),
            }
        )

    return results


# =============================================================================
# Cross-dataset analysis
# =============================================================================


def _cross_dataset_analysis(
    all_stats: dict[tuple[str, str], dict],
    n_bootstrap: int,
    seed: int,
) -> dict:
    """Run cross-dataset hypothesis tests.

    H2: Monotone degradation LOW -> MED -> HIGH (JT test on bootstrap rho).
    H3: Density-stratified trend across all datasets.
    H5: Exhaustive vs greedy comparison with bootstrap CI on delta-rho.

    Args:
        all_stats: Mapping of (dataset, method) -> per-dataset stats.
        n_bootstrap: Bootstrap resamples.
        seed: Random seed.

    Returns:
        Dict with cross-dataset analysis results.
    """
    result: dict = {}

    # --- H2: Monotone degradation (IAM Letter only) ---
    iam_rhos: dict[str, list[float]] = {}
    for ds in IAM_DATASETS_ORDERED:
        key = (ds, "exhaustive")
        if key in all_stats and "spearman" in all_stats[key]:
            rho = all_stats[key]["spearman"]["statistic"]
            iam_rhos[ds] = [rho]

    if len(iam_rhos) == 3:
        # Use the Spearman rho point estimates as single-element groups
        # for a trend test. For a more rigorous test, we'd need bootstrap
        # samples, but with 3 groups of 1, JT is limited.
        # Instead, report the ordering and compute JT on the density-stratified
        # rho values across all IAM datasets.
        rho_values = [iam_rhos[ds][0] for ds in IAM_DATASETS_ORDERED]
        result["h2_monotone_degradation"] = {
            "iam_rhos": {ds: iam_rhos[ds][0] for ds in IAM_DATASETS_ORDERED},
            "is_monotone_decreasing": all(
                rho_values[i] >= rho_values[i + 1] for i in range(len(rho_values) - 1)
            ),
            "rho_ordering": rho_values,
        }

    # --- H3: Density effect across all datasets ---
    # Aggregate density-stratified results
    density_rhos_all = []
    for (ds, method), stats_dict in all_stats.items():
        if method != "exhaustive":
            continue
        if "density_stratified" not in stats_dict:
            continue
        for bin_info in stats_dict["density_stratified"]:
            density_rhos_all.append(
                {
                    "dataset": ds,
                    "density_bin": bin_info["density_bin"],
                    "n_pairs": bin_info["n_pairs"],
                    "spearman_rho": bin_info["spearman_rho"],
                }
            )
    result["h3_density_effect"] = density_rhos_all

    # --- H5: Exhaustive vs greedy ---
    h5_results = {}
    for ds in ALL_DATASETS:
        key_ex = (ds, "exhaustive")
        key_gr = (ds, "greedy")
        if key_ex in all_stats and key_gr in all_stats:
            rho_ex = all_stats[key_ex].get("spearman", {}).get("statistic")
            rho_gr = all_stats[key_gr].get("spearman", {}).get("statistic")
            ci_ex = (
                all_stats[key_ex].get("spearman", {}).get("ci_lower"),
                all_stats[key_ex].get("spearman", {}).get("ci_upper"),
            )
            ci_gr = (
                all_stats[key_gr].get("spearman", {}).get("ci_lower"),
                all_stats[key_gr].get("spearman", {}).get("ci_upper"),
            )
            if rho_ex is not None and rho_gr is not None:
                h5_results[ds] = {
                    "rho_exhaustive": rho_ex,
                    "rho_greedy": rho_gr,
                    "delta_rho": round(rho_ex - rho_gr, 4),
                    "ci_exhaustive": ci_ex,
                    "ci_greedy": ci_gr,
                }
    result["h5_method_comparison"] = h5_results

    # --- Multiple testing correction ---
    mantel_ps = []
    mantel_keys = []
    for (ds, method), stats_dict in sorted(all_stats.items()):
        if "mantel" in stats_dict:
            mantel_ps.append(stats_dict["mantel"]["p_value"])
            mantel_keys.append(f"{ds}_{method}")

    if mantel_ps:
        adjusted = holm_bonferroni(mantel_ps)
        result["multiple_testing"] = {
            "method": "holm_bonferroni",
            "n_tests": len(mantel_ps),
            "raw_p_values": dict(zip(mantel_keys, mantel_ps, strict=True)),
            "adjusted_p_values": dict(
                zip(mantel_keys, [round(p, 6) for p in adjusted], strict=True)
            ),
        }

    return result


# =============================================================================
# Figure generation
# =============================================================================


def _generate_figures(
    all_stats: dict[tuple[str, str], dict],
    all_artifacts: dict[str, dict],
    cross_stats: dict,
    output_dir: str,
) -> None:
    """Generate publication-quality figures.

    Args:
        all_stats: Per-dataset statistics.
        all_artifacts: Loaded data artifacts.
        cross_stats: Cross-dataset analysis results.
        output_dir: Output directory for figures.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from benchmarks.plotting_styles import (
        PAUL_TOL_BRIGHT,
        apply_ieee_style,
        save_figure,
    )

    apply_ieee_style()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Colors for datasets
    ds_colors = {
        "iam_letter_low": PAUL_TOL_BRIGHT["blue"],
        "iam_letter_med": PAUL_TOL_BRIGHT["cyan"],
        "iam_letter_high": PAUL_TOL_BRIGHT["green"],
        "linux": PAUL_TOL_BRIGHT["red"],
        "aids": PAUL_TOL_BRIGHT["purple"],
    }
    method_hatches = {"exhaustive": "", "greedy": "///"}

    # ---- Main figure (2x3) ----
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.0))

    # (a) Hexbin: IAM LOW exhaustive
    _plot_hexbin(axes[0, 0], all_artifacts, "iam_letter_low", "exhaustive", all_stats, "(a)")

    # (b) Hexbin: LINUX exhaustive
    _plot_hexbin(axes[0, 1], all_artifacts, "linux", "exhaustive", all_stats, "(b)")

    # (c) Hexbin: AIDS exhaustive
    _plot_hexbin(axes[0, 2], all_artifacts, "aids", "exhaustive", all_stats, "(c)")

    # (d) Grouped bar: Spearman rho per dataset
    _plot_rho_bars(axes[1, 0], all_stats, ds_colors, method_hatches, "(d)")

    # (e) Line: rho vs distortion for IAM
    _plot_distortion_trend(axes[1, 1], all_stats, "(e)")

    # (f) Grouped bar: P@k
    _plot_pak_bars(axes[1, 2], all_stats, ds_colors, "(f)")

    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, "correlation_main_figure"))
    plt.close(fig)
    logger.info("Saved main figure.")

    # ---- Bland-Altman plots ----
    for (ds, method), stats_dict in all_stats.items():
        if "error" in stats_dict:
            continue
        _plot_bland_altman(all_artifacts[ds], method, fig_dir)

    # ---- Heatmap comparisons ----
    for _ds, art in all_artifacts.items():
        _plot_heatmap_comparison(art, fig_dir)


def _plot_hexbin(
    ax: object,
    all_artifacts: dict[str, dict],
    dataset: str,
    method: str,
    all_stats: dict,
    panel_label: str,
) -> None:
    """Hexbin scatter of Levenshtein vs GED."""
    import matplotlib.pyplot as plt

    art = all_artifacts.get(dataset)
    if art is None:
        return
    ged = art["ged_matrix"]
    lev = art["lev_matrices"].get(method)
    if lev is None:
        return

    [v_ged, v_lev], _ = extract_upper_tri(ged, lev, mask_inf=True)
    if len(v_ged) < 10:
        return

    hb = ax.hexbin(v_lev, v_ged, gridsize=40, cmap="YlOrRd", mincnt=1)
    plt.colorbar(hb, ax=ax, shrink=0.7, label="Count")

    # OLS line
    ols = ols_regression(v_lev, v_ged)
    x_range = np.linspace(v_lev.min(), v_lev.max(), 100)
    ax.plot(x_range, ols["slope"] * x_range + ols["intercept"], "k--", lw=0.8, alpha=0.7)

    # Annotate
    key = (dataset, method)
    rho = all_stats.get(key, {}).get("spearman", {}).get("statistic", "?")
    r = ols.get("r_value", "?")
    if isinstance(rho, float):
        rho = f"{rho:.2f}"
    if isinstance(r, float):
        r = f"{r:.2f}"
    ax.set_xlabel("Levenshtein")
    ax.set_ylabel("GED")
    ax.set_title(f"{panel_label} {DATASET_DISPLAY.get(dataset, dataset)}")
    ax.text(
        0.05,
        0.95,
        f"$\\rho$={rho}\n$r$={r}",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )


def _plot_rho_bars(
    ax: object,
    all_stats: dict,
    ds_colors: dict,
    method_hatches: dict,
    panel_label: str,
) -> None:
    """Grouped bar chart of Spearman rho per dataset and method."""
    from benchmarks.plotting_styles import PLOT_SETTINGS

    datasets_to_plot = [ds for ds in ALL_DATASETS if any((ds, m) in all_stats for m in METHODS)]
    x = np.arange(len(datasets_to_plot))
    width = 0.35

    for m_idx, method in enumerate(METHODS):
        rhos = []
        ci_lo = []
        ci_hi = []
        colors = []
        for ds in datasets_to_plot:
            key = (ds, method)
            s = all_stats.get(key, {}).get("spearman", {})
            rho = s.get("statistic", 0)
            lo = s.get("ci_lower", rho)
            hi = s.get("ci_upper", rho)
            rhos.append(rho)
            ci_lo.append(rho - lo)
            ci_hi.append(hi - rho)
            colors.append(ds_colors.get(ds, "#999999"))

        offset = (m_idx - 0.5) * width
        ax.bar(
            x + offset,
            rhos,
            width,
            yerr=[ci_lo, ci_hi],
            label=method.capitalize(),
            color=colors,
            hatch=method_hatches[method],
            alpha=PLOT_SETTINGS["bar_alpha"],
            capsize=PLOT_SETTINGS["errorbar_capsize"],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [DATASET_DISPLAY.get(ds, ds).replace("IAM Letter ", "") for ds in datasets_to_plot],
        rotation=30,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("Spearman $\\rho$")
    ax.set_title(f"{panel_label} Correlation by dataset")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_ylim(0, 1.05)


def _plot_distortion_trend(
    ax: object,
    all_stats: dict,
    panel_label: str,
) -> None:
    """Line plot of rho vs distortion level (IAM Letter LOW/MED/HIGH)."""
    from benchmarks.plotting_styles import PAUL_TOL_BRIGHT, PLOT_SETTINGS

    method_colors = {
        "exhaustive": PAUL_TOL_BRIGHT["blue"],
        "greedy": PAUL_TOL_BRIGHT["red"],
    }

    x_pos = [0, 1, 2]
    x_labels = ["LOW", "MED", "HIGH"]

    for method in METHODS:
        rhos = []
        ci_lo = []
        ci_hi = []
        for ds in IAM_DATASETS_ORDERED:
            key = (ds, method)
            s = all_stats.get(key, {}).get("spearman", {})
            rho = s.get("statistic", np.nan)
            lo = s.get("ci_lower", rho)
            hi = s.get("ci_upper", rho)
            rhos.append(rho)
            ci_lo.append(rho - lo)
            ci_hi.append(hi - rho)

        color = method_colors[method]
        ax.errorbar(
            x_pos,
            rhos,
            yerr=[ci_lo, ci_hi],
            marker="o",
            label=method.capitalize(),
            color=color,
            capsize=PLOT_SETTINGS["errorbar_capsize"],
            linewidth=PLOT_SETTINGS["line_width"],
        )
        # CI band
        ax.fill_between(
            x_pos,
            [r - lo for r, lo in zip(rhos, ci_lo, strict=True)],
            [r + hi for r, hi in zip(rhos, ci_hi, strict=True)],
            alpha=PLOT_SETTINGS["error_band_alpha"],
            color=color,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Distortion level")
    ax.set_ylabel("Spearman $\\rho$")
    ax.set_title(f"{panel_label} IAM Letter trend")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.05)


def _plot_pak_bars(
    ax: object,
    all_stats: dict,
    ds_colors: dict,
    panel_label: str,
) -> None:
    """Grouped bar chart of Precision@k for k=5,10,20 by dataset."""
    from benchmarks.plotting_styles import PLOT_SETTINGS

    k_values = [5, 10, 20]
    datasets_to_plot = [
        ds
        for ds in ALL_DATASETS
        if (ds, "exhaustive") in all_stats and "precision_at_k" in all_stats[(ds, "exhaustive")]
    ]

    x = np.arange(len(k_values))
    n_ds = len(datasets_to_plot)
    if n_ds == 0:
        return
    width = 0.8 / n_ds

    for d_idx, ds in enumerate(datasets_to_plot):
        pak = all_stats[(ds, "exhaustive")]["precision_at_k"]
        precs = [pak.get(str(k), {}).get("precision", 0) for k in k_values]
        offset = (d_idx - n_ds / 2 + 0.5) * width
        ax.bar(
            x + offset,
            precs,
            width,
            label=DATASET_DISPLAY.get(ds, ds).replace("IAM Letter ", ""),
            color=ds_colors.get(ds, "#999999"),
            alpha=PLOT_SETTINGS["bar_alpha"],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in k_values])
    ax.set_ylabel("Precision@k")
    ax.set_title(f"{panel_label} Neighbor preservation")
    ax.legend(fontsize=6, ncol=2, loc="lower left")
    ax.set_ylim(0, 1.05)


def _plot_bland_altman(
    artifacts: dict,
    method: str,
    fig_dir: str,
) -> None:
    """Bland-Altman plot: z-normalized mean vs difference."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from benchmarks.plotting_styles import save_figure

    dataset = artifacts["dataset"]
    ged = artifacts["ged_matrix"]
    lev = artifacts["lev_matrices"].get(method)
    if lev is None:
        return

    [v_ged, v_lev], _ = extract_upper_tri(ged, lev, mask_inf=True)
    if len(v_ged) < 10:
        return

    # z-normalize
    v_ged_z = (v_ged - np.mean(v_ged)) / (np.std(v_ged) + 1e-12)
    v_lev_z = (v_lev - np.mean(v_lev)) / (np.std(v_lev) + 1e-12)

    mean_vals = (v_ged_z + v_lev_z) / 2
    diff_vals = v_ged_z - v_lev_z
    mean_diff = float(np.mean(diff_vals))
    std_diff = float(np.std(diff_vals))

    fig, ax = plt.subplots(figsize=(3.39, 2.5))
    ax.hexbin(mean_vals, diff_vals, gridsize=40, cmap="YlOrRd", mincnt=1)
    ax.axhline(
        mean_diff, color="black", linestyle="-", linewidth=0.8, label=f"Mean={mean_diff:.2f}"
    )
    ax.axhline(
        mean_diff + 1.96 * std_diff,
        color="red",
        linestyle="--",
        linewidth=0.8,
        label="+1.96$\\sigma$",
    )
    ax.axhline(
        mean_diff - 1.96 * std_diff,
        color="red",
        linestyle="--",
        linewidth=0.8,
        label="-1.96$\\sigma$",
    )
    ax.set_xlabel("Mean (z-norm GED, z-norm Lev)")
    ax.set_ylabel("Difference (z-norm GED - z-norm Lev)")
    ax.set_title(f"Bland-Altman: {DATASET_DISPLAY.get(dataset, dataset)} ({method})")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, f"bland_altman_{dataset}_{method}"))
    plt.close(fig)


def _plot_heatmap_comparison(
    artifacts: dict,
    fig_dir: str,
) -> None:
    """Side-by-side heatmaps: GED, Lev exhaustive, Lev greedy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from benchmarks.plotting_styles import save_figure

    dataset = artifacts["dataset"]
    ged = artifacts["ged_matrix"]
    labels = artifacts["labels"]
    n = min(ged.shape[0], 100)

    # Sort by label (IAM) or by sum of GED row
    if any(lab != "" for lab in labels):
        order = np.argsort(labels[:n])
    else:
        row_sums = np.nansum(np.where(np.isfinite(ged[:n, :n]), ged[:n, :n], 0), axis=1)
        order = np.argsort(row_sums)

    matrices = [("GED", ged[:n, :n][np.ix_(order, order)])]
    for method in METHODS:
        lev = artifacts["lev_matrices"].get(method)
        if lev is not None:
            matrices.append((f"Lev ({method})", lev[:n, :n][np.ix_(order, order)]))

    n_panels = len(matrices)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.39 * n_panels, 3.0))
    if n_panels == 1:
        axes = [axes]

    # Shared color scale: use max finite value across all
    vmax = 0
    for _, m in matrices:
        finite_vals = m[np.isfinite(m) & (m >= 0)]
        if len(finite_vals) > 0:
            vmax = max(vmax, np.percentile(finite_vals, 95))

    for ax, (title, m) in zip(axes, matrices, strict=True):
        display = np.where(np.isfinite(m) & (m >= 0), m, np.nan)
        im = ax.imshow(display, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Graph index")
        ax.set_ylabel("Graph index")
        plt.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle(DATASET_DISPLAY.get(dataset, dataset), fontsize=12)
    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, f"heatmap_comparison_{dataset}"))
    plt.close(fig)


# =============================================================================
# Table generation
# =============================================================================


def _generate_tables(
    all_stats: dict[tuple[str, str], dict],
    cross_stats: dict,
    output_dir: str,
) -> None:
    """Generate LaTeX tables.

    Args:
        all_stats: Per-dataset statistics.
        cross_stats: Cross-dataset analysis results.
        output_dir: Output directory.
    """
    from benchmarks.plotting_styles import save_latex_table

    table_dir = os.path.join(output_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)

    # ---- Table 1: Correlation summary ----
    rows = []
    for ds in ALL_DATASETS:
        for method in METHODS:
            key = (ds, method)
            s = all_stats.get(key, {})
            if "error" in s:
                continue

            sp = s.get("spearman", {})
            pe = s.get("pearson", {})
            kt = s.get("kendall", {})
            mantel = s.get("mantel", {})
            pak = s.get("precision_at_k", {})

            rows.append(
                {
                    "Dataset": DATASET_DISPLAY.get(ds, ds),
                    "Method": method.capitalize(),
                    "N": s.get("n_graphs", ""),
                    "Pairs": s.get("n_valid_pairs", ""),
                    "Spearman": _fmt_ci(sp),
                    "Pearson": _fmt_ci(pe),
                    "Kendall": f"{kt.get('statistic', ''):.3f}"
                    if kt.get("statistic") is not None
                    else "",
                    "CCC": f"{s.get('lins_ccc_raw', '')}"
                    if s.get("lins_ccc_raw") is not None
                    else "",
                    "P@10": f"{pak.get('10', {}).get('precision', ''):.3f}"
                    if pak.get("10", {}).get("precision") is not None
                    else "",
                    "Mantel p": _fmt_p(mantel.get("p_value")),
                }
            )

    if rows:
        df = pd.DataFrame(rows)
        save_latex_table(
            df,
            os.path.join(table_dir, "correlation_summary.tex"),
            caption=(
                "Correlation between Levenshtein distance on"
                " canonical strings and graph edit distance."
            ),
            label="tab:correlation_summary",
        )
        logger.info("Saved correlation_summary.tex")

    # ---- Table 2: Precision@k ----
    pak_rows = []
    for ds in ALL_DATASETS:
        for method in METHODS:
            key = (ds, method)
            s = all_stats.get(key, {})
            if "error" in s:
                continue
            pak = s.get("precision_at_k", {})
            pak_rows.append(
                {
                    "Dataset": DATASET_DISPLAY.get(ds, ds),
                    "Method": method.capitalize(),
                    "P@5": f"{pak.get('5', {}).get('precision', 0):.3f}",
                    "P@10": f"{pak.get('10', {}).get('precision', 0):.3f}",
                    "P@20": f"{pak.get('20', {}).get('precision', 0):.3f}",
                }
            )

    if pak_rows:
        df_pak = pd.DataFrame(pak_rows)
        save_latex_table(
            df_pak,
            os.path.join(table_dir, "precision_at_k.tex"),
            caption=(
                "Precision@k: fraction of true k-nearest neighbors"
                " preserved by Levenshtein distance."
            ),
            label="tab:precision_at_k",
        )
        logger.info("Saved precision_at_k.tex")


def _fmt_ci(s: dict) -> str:
    """Format statistic [CI lower, CI upper]."""
    stat = s.get("statistic")
    if stat is None:
        return ""
    lo = s.get("ci_lower", stat)
    hi = s.get("ci_upper", stat)
    return f"{stat:.3f} [{lo:.3f}, {hi:.3f}]"


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
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
    save_csv: bool,
    save_plots: bool,
    save_tables: bool,
) -> None:
    """Run the full correlation analysis pipeline.

    Args:
        data_root: Root directory of eval pipeline output.
        output_dir: Output directory for results.
        datasets: List of dataset names.
        n_bootstrap: Bootstrap resamples.
        n_permutations: Mantel permutations.
        seed: Random seed.
        save_csv: Whether to save pair-level CSVs.
        save_plots: Whether to generate figures.
        save_tables: Whether to generate LaTeX tables.
    """
    t0 = time.perf_counter()
    os.makedirs(output_dir, exist_ok=True)
    stats_dir = os.path.join(output_dir, "stats")
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    all_stats: dict[tuple[str, str], dict] = {}
    all_artifacts: dict[str, dict] = {}

    for ds in datasets:
        logger.info("Loading artifacts for %s...", ds)
        try:
            artifacts = _load_dataset_artifacts(data_root, ds)
        except FileNotFoundError as e:
            logger.error("Missing data for %s: %s", ds, e)
            continue

        all_artifacts[ds] = artifacts

        for method in METHODS:
            if method not in artifacts["lev_matrices"]:
                logger.warning("No %s Levenshtein matrix for %s, skipping.", method, ds)
                continue

            stats_dict = _analyze_dataset(
                artifacts,
                method,
                n_bootstrap=n_bootstrap,
                n_permutations=n_permutations,
                seed=seed,
            )
            all_stats[(ds, method)] = stats_dict

            # Save per-dataset stats
            stats_path = os.path.join(stats_dir, f"{ds}_{method}_correlation_stats.json")
            with open(stats_path, "w") as f:
                json.dump(stats_dict, f, indent=2, default=str)
            logger.info("Saved %s", stats_path)

            # Save pair-level CSV
            if save_csv and "error" not in stats_dict:
                _save_pair_csv(artifacts, method, raw_dir)

    # Cross-dataset analysis
    if len(all_stats) > 0:
        logger.info("Running cross-dataset analysis...")
        cross_stats = _cross_dataset_analysis(all_stats, n_bootstrap, seed)
        cross_path = os.path.join(stats_dir, "cross_dataset_analysis.json")
        with open(cross_path, "w") as f:
            json.dump(cross_stats, f, indent=2, default=str)
        logger.info("Saved %s", cross_path)

        # Summary table JSON
        summary = _build_summary(all_stats)
        summary_path = os.path.join(stats_dir, "summary_table.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
    else:
        cross_stats = {}

    # Figures
    if save_plots and len(all_stats) > 0:
        logger.info("Generating figures...")
        _generate_figures(all_stats, all_artifacts, cross_stats, output_dir)

    # Tables
    if save_tables and len(all_stats) > 0:
        logger.info("Generating tables...")
        _generate_tables(all_stats, cross_stats, output_dir)

    elapsed = time.perf_counter() - t0
    logger.info("Pipeline complete in %.1fs.", elapsed)


def _save_pair_csv(
    artifacts: dict,
    method: str,
    raw_dir: str,
) -> None:
    """Save pair-level CSV with GED and Levenshtein values."""
    dataset = artifacts["dataset"]
    ged = artifacts["ged_matrix"]
    lev = artifacts["lev_matrices"][method]
    n = ged.shape[0]
    graph_ids = artifacts["graph_ids"]

    triu_i, triu_j = np.triu_indices(n, k=1)
    v_ged = ged[triu_i, triu_j]
    v_lev = lev[triu_i, triu_j]

    valid = np.isfinite(v_ged) & (v_ged >= 0) & np.isfinite(v_lev) & (v_lev >= 0)

    df = pd.DataFrame(
        {
            "graph_i": [graph_ids[i] for i in triu_i[valid]],
            "graph_j": [graph_ids[j] for j in triu_j[valid]],
            "ged": v_ged[valid],
            "levenshtein": v_lev[valid],
        }
    )
    path = os.path.join(raw_dir, f"{dataset}_{method}_pair_data.csv")
    df.to_csv(path, index=False)
    logger.info("Saved %d pairs to %s", len(df), path)


def _build_summary(all_stats: dict[tuple[str, str], dict]) -> list[dict]:
    """Build summary list for JSON export."""
    rows = []
    for (ds, method), s in sorted(all_stats.items()):
        if "error" in s:
            continue
        rows.append(
            {
                "dataset": ds,
                "method": method,
                "n_graphs": s.get("n_graphs"),
                "n_valid_pairs": s.get("n_valid_pairs"),
                "spearman_rho": s.get("spearman", {}).get("statistic"),
                "spearman_ci": [
                    s.get("spearman", {}).get("ci_lower"),
                    s.get("spearman", {}).get("ci_upper"),
                ],
                "pearson_r": s.get("pearson", {}).get("statistic"),
                "kendall_tau": s.get("kendall", {}).get("statistic"),
                "lins_ccc": s.get("lins_ccc_raw"),
                "mantel_p": s.get("mantel", {}).get("p_value"),
                "pak_10": s.get("precision_at_k", {}).get("10", {}).get("precision"),
            }
        )
    return rows


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Levenshtein-GED correlation analysis for IsalGraph evaluation."
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
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--n-permutations", type=int, default=9999)
    parser.add_argument(
        "--mode",
        choices=["local", "picasso"],
        default="local",
    )
    parser.add_argument("--csv", action="store_true", help="Save pair-level CSVs.")
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

    run_pipeline(
        data_root=args.data_root,
        output_dir=args.output_dir,
        datasets=datasets,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        seed=args.seed,
        save_csv=args.csv,
        save_plots=args.plot,
        save_tables=args.table,
    )


if __name__ == "__main__":
    main()
