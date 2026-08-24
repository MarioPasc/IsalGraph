"""Readers for a dense distance matrix, for the statistics track.

Two things the statistics track needs on every matrix and should not
re-derive: the equal-``n`` mask, and the strict upper triangle as flat arrays
that can be indexed back to ``(i, j)``.

**The equal-``n`` mask is computed, never materialised.**  CONTRACTS §4 is
explicit: it is ``node_counts[:, None] == node_counts[None, :]`` and it is not
a file.  A stored mask is a mask that can go stale against the cohort it was
derived from, and at 16,370 graphs it is a 268 MB array to reconstruct a
32 kB vector.
"""

from __future__ import annotations

import numpy as np


def encodable_mask(status: np.ndarray, length: np.ndarray) -> np.ndarray:
    """Return the ``(G,)`` mask that is true where a representation encoded a graph.

    The rule is CONTRACTS §3.2's: ``status == "error"`` or ``length < 0`` means
    no usable encoding.  **A censored row is encodable** -- D14 retains it with
    its greedy-min fallback string, so it has an encoding and enters the pair
    set.  An empty string is not by itself a fault: a one-node graph
    legitimately encodes to zero symbols.

    This is the single source of truth for the rule.  ``distance_runner``
    marks invalid rows with it and ``size_null`` restricts the baseline with
    it, so the arm and its null are defined on exactly the same pairs.  Two
    copies of this predicate that drifted apart would produce an arm and a
    null measured on different pair sets, which is not detectable in the
    output.

    Args:
        status: ``(G,)`` of ``ok`` | ``censored`` | ``fallback`` | ``error``.
        length: ``(G,)`` symbol counts, ``-1`` when not encoded.

    Returns:
        ``bool (G,)``, true where the graph has a usable encoding.
    """
    invalid = (np.asarray(status) == "error") | (np.asarray(length, dtype=np.int64) < 0)
    return ~np.asarray(invalid, dtype=bool)


def pair_mask(row_mask: np.ndarray) -> np.ndarray:
    """Lift a ``(G,)`` row mask to the ``(G, G)`` pairs where both ends hold.

    Args:
        row_mask: ``(G,)`` boolean, typically from :func:`encodable_mask`.

    Returns:
        ``bool (G, G)``.
    """
    rows = np.asarray(row_mask, dtype=bool)
    return np.asarray(rows[:, None] & rows[None, :], dtype=bool)


def equal_n_mask(node_counts: np.ndarray) -> np.ndarray:
    """Return the ``(G, G)`` mask that is true where two graphs share ``n``.

    Args:
        node_counts: ``(G,)`` integer node counts.

    Returns:
        ``bool (G, G)``.  Never write this to disk.
    """
    counts = np.asarray(node_counts)
    return np.asarray(counts[:, None] == counts[None, :], dtype=bool)


def upper_triangle(
    matrix: np.ndarray, *, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the strict upper triangle as ``(values, i_idx, j_idx)``.

    Strict, so the zero diagonal never enters a correlation and inflates it.

    Args:
        matrix: ``(G, G)``.
        mask: optional ``(G, G)`` selector intersected with the triangle --
            typically ``defined_mask``, or ``defined_mask & equal_n_mask``.

    Returns:
        Three parallel arrays: the selected values, and their row and column
        indices, so a caller can join against another matrix or report which
        pair produced an outlier.

    Raises:
        ValueError: when *matrix* is not square, or *mask* does not match it.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix has shape {matrix.shape}, expected square")
    rows, cols = np.triu_indices(matrix.shape[0], k=1)
    if mask is not None:
        if mask.shape != matrix.shape:
            raise ValueError(f"mask {mask.shape} does not match matrix {matrix.shape}")
        keep = np.asarray(mask, dtype=bool)[rows, cols]
        rows, cols = rows[keep], cols[keep]
    return matrix[rows, cols], rows, cols


def paired_upper_triangle(
    left: np.ndarray, right: np.ndarray, *, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract the same strict upper triangle from two aligned matrices.

    The join a correlation needs: one distance matrix against one GED matrix,
    on identical ``(i, j)`` positions, with a single mask applied to both.

    Args:
        left: ``(G, G)``, e.g. a representation's distances.
        right: ``(G, G)``, e.g. the GED matrix.
        mask: optional selector, typically both files' ``defined_mask`` and
            the GED ``certified_mask`` intersected.

    Returns:
        ``(left_values, right_values, i_idx, j_idx)``.

    Raises:
        ValueError: when the two matrices differ in shape.
    """
    if left.shape != right.shape:
        raise ValueError(f"matrices differ in shape: {left.shape} vs {right.shape}")
    left_values, rows, cols = upper_triangle(left, mask=mask)
    return left_values, right[rows, cols], rows, cols


__all__ = [
    "encodable_mask",
    "equal_n_mask",
    "pair_mask",
    "paired_upper_triangle",
    "upper_triangle",
]
