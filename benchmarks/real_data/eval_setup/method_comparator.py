"""Exhaustive vs greedy-min method comparison.

Aggregates per-graph comparison data and computes matrix-level
correlation statistics between the two Levenshtein matrices and GED.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
from scipy import stats

from benchmarks.eval_setup.canonical_computer import CanonicalResult

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


def compare_methods(
    canonical_results: list[CanonicalResult],
    lev_matrix_exhaustive: np.ndarray,
    lev_matrix_greedy: np.ndarray,
    ged_matrix: np.ndarray,
    graph_ids: list[str],
    dataset_name: str,
) -> dict:
    """Full comparison: per-graph and matrix-level.

    Args:
        canonical_results: Per-graph dual canonical results.
        lev_matrix_exhaustive: Levenshtein matrix from exhaustive strings.
        lev_matrix_greedy: Levenshtein matrix from greedy strings.
        ged_matrix: GED matrix.
        graph_ids: Graph identifiers.
        dataset_name: Dataset name.

    Returns:
        Comparison dict ready for JSON serialization.
    """
    # ---- Per-graph comparison ----
    per_graph: list[dict] = []
    n_identical_strings = 0
    n_identical_lengths = 0
    length_gaps: list[int] = []
    lev_between_methods: list[int] = []
    speedups: list[float] = []
    total_exhaustive_time = 0.0
    total_greedy_time = 0.0

    for r in canonical_results:
        entry = {
            "graph_id": r.graph_id,
            "exhaustive_string": r.exhaustive_string,
            "greedy_string": r.greedy_string,
            "exhaustive_length": r.exhaustive_length,
            "greedy_length": r.greedy_length,
            "strings_identical": r.strings_identical,
            "length_gap": r.length_gap,
            "levenshtein_between_methods": r.levenshtein_between_methods,
            "exhaustive_time_s": r.exhaustive_time_s,
            "greedy_time_s": r.greedy_time_s,
            "speedup": r.speedup,
            "n_nodes": r.n_nodes,
            "n_edges": r.n_edges,
            "density": r.density,
        }
        per_graph.append(entry)

        if r.strings_identical is True:
            n_identical_strings += 1
        if (
            r.exhaustive_length >= 0
            and r.greedy_length >= 0
            and r.exhaustive_length == r.greedy_length
        ):
            n_identical_lengths += 1
        if r.length_gap is not None:
            length_gaps.append(r.length_gap)
        if r.levenshtein_between_methods is not None:
            lev_between_methods.append(r.levenshtein_between_methods)
        if r.speedup is not None and r.speedup > 0:
            speedups.append(r.speedup)
        total_exhaustive_time += r.exhaustive_time_s
        total_greedy_time += r.greedy_time_s

    n_graphs = len(canonical_results)

    # ---- Aggregate ----
    aggregate = {
        "n_identical_strings": n_identical_strings,
        "pct_identical_strings": round(100.0 * n_identical_strings / n_graphs, 1)
        if n_graphs > 0
        else 0.0,
        "n_identical_lengths": n_identical_lengths,
        "pct_identical_lengths": round(100.0 * n_identical_lengths / n_graphs, 1)
        if n_graphs > 0
        else 0.0,
        "mean_length_gap": round(float(np.mean(length_gaps)), 1) if length_gaps else None,
        "max_length_gap": int(max(length_gaps)) if length_gaps else None,
        "mean_levenshtein_between_methods": round(float(np.mean(lev_between_methods)), 1)
        if lev_between_methods
        else None,
        "max_levenshtein_between_methods": int(max(lev_between_methods))
        if lev_between_methods
        else None,
        "mean_speedup": round(float(np.mean(speedups)), 1) if speedups else None,
        "median_speedup": round(float(np.median(speedups)), 1) if speedups else None,
        "total_exhaustive_time_s": round(total_exhaustive_time, 2),
        "total_greedy_time_s": round(total_greedy_time, 2),
    }

    # ---- Matrix-level comparison ----
    matrix_comparison = _matrix_level_comparison(
        lev_matrix_exhaustive, lev_matrix_greedy, ged_matrix
    )

    return {
        "dataset": dataset_name,
        "n_graphs": n_graphs,
        "per_graph": per_graph,
        "aggregate": aggregate,
        "matrix_comparison": matrix_comparison,
    }


def _matrix_level_comparison(
    lev_exhaust: np.ndarray,
    lev_greedy: np.ndarray,
    ged: np.ndarray,
) -> dict:
    """Compare upper triangles of Levenshtein matrices against GED.

    Args:
        lev_exhaust: Exhaustive Levenshtein matrix.
        lev_greedy: Greedy Levenshtein matrix.
        ged: GED matrix.

    Returns:
        Dict with correlation statistics.
    """
    [v_exhaust, v_greedy, v_ged], valid_mask = _upper_triangle_vectors(lev_exhaust, lev_greedy, ged)

    n_pairs = len(v_ged)

    result: dict = {
        "n_pairs": n_pairs,
        "exhaustive_vs_ged": _safe_correlation(v_exhaust, v_ged),
        "greedy_vs_ged": _safe_correlation(v_greedy, v_ged),
    }

    # Comparison between the two Levenshtein matrices
    if n_pairs > 0:
        diff = np.abs(v_exhaust - v_greedy)
        result["exhaustive_vs_greedy"] = {
            "max_abs_diff": int(np.max(diff)),
            "mean_abs_diff": round(float(np.mean(diff)), 2),
            "frac_identical_entries": round(float(np.mean(diff == 0)), 4),
        }
        result["exhaustive_vs_greedy"].update(_safe_correlation(v_exhaust, v_greedy))
    else:
        result["exhaustive_vs_greedy"] = {}

    return result


def save_method_comparison(
    comparison: dict,
    output_path: str,
) -> None:
    """Save method comparison to JSON.

    Args:
        comparison: Comparison dict from compare_methods().
        output_path: Output JSON path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info("Saved method comparison to %s", output_path)
