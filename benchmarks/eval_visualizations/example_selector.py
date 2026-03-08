"""Deterministic selection of illustrative graph pairs.

Selects graph pairs from distance matrices for visualization,
with all selections logged to JSON for reproducibility.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SelectedExample:
    """Record of a selected example for traceability."""

    hypothesis: str
    role: str
    dataset: str
    indices: tuple[int, ...]
    graph_ids: tuple[str, ...]
    criterion: str
    ged: float | None = None
    lev: float | None = None


def select_concordant_similar(
    ged: np.ndarray,
    lev: np.ndarray,
    node_counts: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    target_ged: float = 1.0,
    min_nodes: int = 3,
    max_nodes: int = 10,
) -> tuple[int, int]:
    """Select a pair with small GED and small Levenshtein (concordant).

    Prefers same-class pairs if labels are available.

    Args:
        ged: GED distance matrix.
        lev: Levenshtein distance matrix.
        node_counts: Number of nodes per graph.
        labels: Optional class labels.
        target_ged: Target GED value to match.
        min_nodes: Minimum node count filter.
        max_nodes: Maximum node count filter.

    Returns:
        (i, j) indices of the selected pair.
    """
    n = ged.shape[0]
    best_pair = (0, 1)
    best_score = float("inf")

    for i in range(n):
        if not (min_nodes <= node_counts[i] <= max_nodes):
            continue
        for j in range(i + 1, n):
            if not (min_nodes <= node_counts[j] <= max_nodes):
                continue
            g = ged[i, j]
            if not np.isfinite(g) or g <= 0:
                continue

            score = abs(g - target_ged) + 0.1 * lev[i, j]
            # Prefer same class
            if labels is not None and labels[i] != labels[j]:
                score += 10.0

            if score < best_score:
                best_score = score
                best_pair = (i, j)

    return best_pair


def select_dissimilar(
    ged: np.ndarray,
    lev: np.ndarray,
    node_counts: np.ndarray,
    labels: np.ndarray | None = None,
) -> tuple[int, int]:
    """Select a pair with large GED and large Levenshtein.

    Args:
        ged: GED distance matrix.
        lev: Levenshtein distance matrix.
        node_counts: Number of nodes per graph.
        labels: Optional class labels.

    Returns:
        (i, j) indices of the selected pair.
    """
    n = ged.shape[0]
    best_pair = (0, 1)
    best_score = -float("inf")

    for i in range(n):
        for j in range(i + 1, n):
            g = ged[i, j]
            if not np.isfinite(g) or g <= 0:
                continue
            score = g + lev[i, j]
            # Prefer different class
            if labels is not None and labels[i] == labels[j]:
                score -= 10.0
            if score > best_score:
                best_score = score
                best_pair = (i, j)

    return best_pair


def select_discordant(
    ged: np.ndarray,
    lev: np.ndarray,
) -> tuple[int, int]:
    """Select a pair with maximum rank disagreement between GED and Levenshtein.

    Finds (i, j) maximizing |rank_GED(i,j) - rank_Lev(i,j)|.

    Args:
        ged: GED distance matrix.
        lev: Levenshtein distance matrix.

    Returns:
        (i, j) indices of the selected pair.
    """
    n = ged.shape[0]
    # Extract upper triangle pairs
    triu_i, triu_j = np.triu_indices(n, k=1)
    ged_vec = ged[triu_i, triu_j]
    lev_vec = lev[triu_i, triu_j].astype(float)

    # Filter valid pairs
    valid = np.isfinite(ged_vec) & (ged_vec > 0) & np.isfinite(lev_vec) & (lev_vec > 0)
    if not valid.any():
        return (0, 1)

    valid_indices = np.where(valid)[0]
    ged_valid = ged_vec[valid_indices]
    lev_valid = lev_vec[valid_indices]

    # Rank within valid set
    ged_ranks = np.argsort(np.argsort(ged_valid)).astype(float)
    lev_ranks = np.argsort(np.argsort(lev_valid)).astype(float)

    rank_diff = np.abs(ged_ranks - lev_ranks)
    best_local = int(np.argmax(rank_diff))
    best_global = valid_indices[best_local]

    return (int(triu_i[best_global]), int(triu_j[best_global]))


def save_selection_log(
    selections: list[SelectedExample],
    output_path: str,
) -> None:
    """Save selection log to JSON for reproducibility.

    Args:
        selections: List of selected examples.
        output_path: Output JSON file path.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    records = [asdict(s) for s in selections]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=str)
    logger.info("Selection log saved: %s (%d examples)", output_path, len(records))
