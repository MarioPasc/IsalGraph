"""All-pairs Levenshtein distance matrix computation.

Computes pairwise Levenshtein distances for a list of canonical strings,
using the C extension (python-Levenshtein) when available, with cross-
validation against the pure-Python implementation.
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np

logger = logging.getLogger(__name__)


def _get_levenshtein_func(use_c_extension: bool = True):
    """Get the best available Levenshtein distance function.

    Returns:
        (func, name) tuple.
    """
    if use_c_extension:
        try:
            import Levenshtein as LevExt

            return LevExt.distance, "c_extension"
        except ImportError:
            logger.info("python-Levenshtein not available, using pure Python")

    from isalgraph.core.canonical import levenshtein

    return levenshtein, "pure_python"


def cross_validate_levenshtein(
    strings: list[str],
    n_check: int = 50,
) -> dict:
    """Cross-validate C extension against pure-Python implementation.

    Args:
        strings: List of strings.
        n_check: Number of strings to check (uses first n_check).

    Returns:
        Dict with validation results.
    """
    try:
        import Levenshtein as LevExt
    except ImportError:
        return {"status": "skipped", "reason": "python-Levenshtein not installed"}

    from isalgraph.core.canonical import levenshtein as lev_pure

    subset = strings[: min(n_check, len(strings))]
    n = len(subset)
    mismatches = 0
    total_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            d_c = LevExt.distance(subset[i], subset[j])
            d_py = lev_pure(subset[i], subset[j])
            if d_c != d_py:
                mismatches += 1
                logger.warning(
                    "Levenshtein mismatch at (%d, %d): C=%d, Python=%d",
                    i,
                    j,
                    d_c,
                    d_py,
                )
            total_pairs += 1

    return {
        "status": "passed" if mismatches == 0 else "failed",
        "n_strings": n,
        "n_pairs_checked": total_pairs,
        "n_mismatches": mismatches,
    }


def compute_levenshtein_matrix(
    strings: list[str | None],
    graph_ids: list[str],
    method: str,
    use_c_extension: bool = True,
) -> np.ndarray:
    """Compute all-pairs Levenshtein distance matrix.

    Args:
        strings: List of canonical strings (None for missing/timeout graphs).
        graph_ids: Graph identifiers (for logging).
        method: 'exhaustive' or 'greedy' (for logging).
        use_c_extension: Whether to prefer the C extension.

    Returns:
        Symmetric int32 [N, N] matrix. -1 for pairs involving None strings.
    """
    lev_func, func_name = _get_levenshtein_func(use_c_extension)
    n = len(strings)
    total_pairs = n * (n - 1) // 2

    logger.info(
        "Computing %s Levenshtein matrix: %d strings, %d pairs, using %s",
        method,
        n,
        total_pairs,
        func_name,
    )

    matrix = np.full((n, n), -1, dtype=np.int32)
    np.fill_diagonal(matrix, 0)

    t0 = time.perf_counter()
    computed = 0

    for i in range(n):
        if strings[i] is None:
            continue
        for j in range(i + 1, n):
            if strings[j] is None:
                continue
            d = lev_func(strings[i], strings[j])
            matrix[i, j] = d
            matrix[j, i] = d
            computed += 1

    elapsed = time.perf_counter() - t0
    logger.info(
        "Levenshtein %s: %d pairs in %.2fs (%.0f pairs/s)",
        method,
        computed,
        elapsed,
        computed / elapsed if elapsed > 0 else 0,
    )
    return matrix


def save_levenshtein_matrix(
    matrix: np.ndarray,
    graph_ids: list[str],
    method: str,
    output_path: str,
) -> None:
    """Save Levenshtein matrix in .npz format.

    Args:
        matrix: Pairwise Levenshtein distance matrix [N, N].
        graph_ids: Graph identifiers.
        method: 'exhaustive' or 'greedy'.
        output_path: Output .npz path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        levenshtein_matrix=matrix,
        graph_ids=np.array(graph_ids, dtype=str),
        method=np.array(method),
    )
    logger.info("Saved Levenshtein matrix (%s) to %s", method, output_path)
