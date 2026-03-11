"""Generalized method comparison across all algorithm pairs.

Computes per-method statistics (Lev vs GED, Lev vs WL) and pairwise
comparisons between all available algorithms' Levenshtein matrices.
"""

from __future__ import annotations

import itertools
import json
import logging
import os

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def _upper_triangle_vectors(
    *matrices: np.ndarray,
    mask_inf: bool = True,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Extract upper-triangle vectors from matrices, optionally masking inf.

    Args:
        matrices: One or more square matrices of the same shape.
        mask_inf: If True, exclude entries where any matrix has inf or -1.

    Returns:
        (vectors, mask) where vectors[i] is the masked upper-triangle of matrices[i].
    """
    n = matrices[0].shape[0]
    triu_idx = np.triu_indices(n, k=1)
    vectors = [m[triu_idx].astype(np.float64) for m in matrices]

    if mask_inf:
        valid = np.ones(len(vectors[0]), dtype=bool)
        for v in vectors:
            valid &= np.isfinite(v) & (v >= 0)
        vectors = [v[valid] for v in vectors]
        return vectors, valid

    return vectors, np.ones(len(vectors[0]), dtype=bool)


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> dict:
    """Compute Spearman and Pearson with error handling.

    Args:
        x: First array.
        y: Second array.

    Returns:
        Dict with spearman_r, pearson_r (None if insufficient data).
    """
    result: dict = {
        "spearman_r": None,
        "spearman_p": None,
        "pearson_r": None,
        "pearson_p": None,
        "n_pairs": len(x),
    }
    if len(x) < 3:
        return result

    try:
        sr, sp = stats.spearmanr(x, y)
        result["spearman_r"] = round(float(sr), 4)
        result["spearman_p"] = float(sp)
    except Exception:
        pass

    try:
        pr, pp = stats.pearsonr(x, y)
        result["pearson_r"] = round(float(pr), 4)
        result["pearson_p"] = float(pp)
    except Exception:
        pass

    return result


def compare_all_methods(
    method_lev_matrices: dict[str, np.ndarray],
    ged_matrix: np.ndarray,
    wl_distance_matrix: np.ndarray | None,
    graph_ids: list[str],
    dataset_name: str,
) -> dict:
    """Compare all available methods: per-method and pairwise.

    For each method:
      - Lev vs GED correlation
      - Lev vs WL correlation (if WL available)
    For each pair of methods:
      - Matrix-level differences and correlations
    Also: WL vs GED correlation (once, algorithm-independent).

    Args:
        method_lev_matrices: Mapping of method name -> Levenshtein matrix.
        ged_matrix: GED matrix.
        wl_distance_matrix: WL kernel distance matrix (None if not computed).
        graph_ids: Graph identifiers.
        dataset_name: Dataset name.

    Returns:
        Comparison dict ready for JSON serialization.
    """
    method_names = sorted(method_lev_matrices.keys())
    logger.info("Comparing methods: %s for %s", method_names, dataset_name)

    result: dict = {
        "dataset": dataset_name,
        "n_graphs": len(graph_ids),
        "methods": method_names,
    }

    # ---- Per-method: Lev vs GED ----
    per_method: dict[str, dict] = {}
    for method in method_names:
        lev = method_lev_matrices[method]
        method_stats: dict = {}

        # Lev vs GED
        [v_lev, v_ged], _ = _upper_triangle_vectors(lev, ged_matrix)
        method_stats["lev_vs_ged"] = _safe_correlation(v_lev, v_ged)

        # Lev vs WL
        if wl_distance_matrix is not None:
            [v_lev_w, v_wl], _ = _upper_triangle_vectors(lev, wl_distance_matrix)
            method_stats["lev_vs_wl"] = _safe_correlation(v_lev_w, v_wl)

        per_method[method] = method_stats

    result["per_method"] = per_method

    # ---- WL vs GED (algorithm-independent, once per dataset) ----
    if wl_distance_matrix is not None:
        [v_wl, v_ged], _ = _upper_triangle_vectors(wl_distance_matrix, ged_matrix)
        result["wl_vs_ged"] = _safe_correlation(v_wl, v_ged)

    # ---- Pairwise method comparisons ----
    pairwise: dict[str, dict] = {}
    for m1, m2 in itertools.combinations(method_names, 2):
        lev1 = method_lev_matrices[m1]
        lev2 = method_lev_matrices[m2]
        pair_key = f"{m1}_vs_{m2}"

        [v1, v2], _ = _upper_triangle_vectors(lev1, lev2)
        n_pairs = len(v1)

        pair_stats: dict = {"n_pairs": n_pairs}
        if n_pairs > 0:
            diff = np.abs(v1 - v2)
            pair_stats["max_abs_diff"] = int(np.max(diff))
            pair_stats["mean_abs_diff"] = round(float(np.mean(diff)), 2)
            pair_stats["frac_identical_entries"] = round(float(np.mean(diff == 0)), 4)
            pair_stats.update(_safe_correlation(v1, v2))

        pairwise[pair_key] = pair_stats

    result["pairwise"] = pairwise

    return result


def save_method_comparison(
    comparison: dict,
    output_path: str,
) -> None:
    """Save method comparison to JSON.

    Args:
        comparison: Comparison dict from compare_all_methods().
        output_path: Output JSON path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info("Saved method comparison to %s", output_path)
