# ruff: noqa: N803, N806
"""Unified loader for all Agent 0-4 evaluation outputs.

Loads JSON, CSV, and NPZ files into typed dataclasses with graceful
degradation when agent outputs are missing.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

ALL_DATASETS = ["iam_letter_low", "iam_letter_med", "iam_letter_high", "linux", "aids"]
METHODS = ["exhaustive", "greedy"]
LABELED_DATASETS = {"iam_letter_low", "iam_letter_med", "iam_letter_high"}
DATASET_DISPLAY: dict[str, str] = {
    "iam_letter_low": "IAM Letter LOW",
    "iam_letter_med": "IAM Letter MED",
    "iam_letter_high": "IAM Letter HIGH",
    "linux": "LINUX",
    "aids": "AIDS",
}

# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class DatasetArtifacts:
    """Core artifacts for a single dataset from Agent 0."""

    dataset: str
    ged_matrix: np.ndarray
    node_counts: np.ndarray
    edge_counts: np.ndarray
    graph_ids: list[str]
    labels: np.ndarray | None = None


@dataclass
class CorrelationStats:
    """Correlation statistics from Agent 2."""

    dataset: str
    method: str
    spearman: dict = field(default_factory=dict)
    pearson: dict = field(default_factory=dict)
    kendall: dict = field(default_factory=dict)
    lins_ccc_raw: float = 0.0
    lins_ccc_znorm: float = 0.0
    ols: dict = field(default_factory=dict)
    precision_at_k: dict = field(default_factory=dict)
    n_valid_pairs: int = 0


@dataclass
class AllResults:
    """Container for all evaluation results."""

    datasets: dict[str, DatasetArtifacts] = field(default_factory=dict)
    correlation: dict[tuple[str, str], CorrelationStats] = field(default_factory=dict)
    levenshtein_matrices: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    canonical_strings: dict[tuple[str, str], dict[str, str]] = field(default_factory=dict)


# =============================================================================
# Loaders
# =============================================================================


def load_all_results(
    data_root: str,
    correlation_dir: str | None = None,
) -> AllResults:
    """Load all available evaluation results.

    Args:
        data_root: Root directory for Agent 0 outputs (ged_matrices/, levenshtein_matrices/, etc.).
        correlation_dir: Directory containing correlation stats JSONs from Agent 2.

    Returns:
        AllResults with graceful None for missing data.
    """
    results = AllResults()

    for dataset in ALL_DATASETS:
        arts = _load_dataset_artifacts(data_root, dataset)
        if arts is not None:
            results.datasets[dataset] = arts

        for method in METHODS:
            lev = _load_levenshtein_matrix(data_root, dataset, method)
            if lev is not None:
                results.levenshtein_matrices[(dataset, method)] = lev

            cs = _load_canonical_strings(data_root, dataset, method)
            if cs is not None:
                results.canonical_strings[(dataset, method)] = cs

            if correlation_dir is not None:
                stats = _load_correlation_stats(correlation_dir, dataset, method)
                if stats is not None:
                    results.correlation[(dataset, method)] = stats

    n_ds = len(results.datasets)
    n_lev = len(results.levenshtein_matrices)
    n_corr = len(results.correlation)
    logger.info(
        "Loaded: %d datasets, %d Levenshtein matrices, %d correlation stats",
        n_ds,
        n_lev,
        n_corr,
    )
    return results


def _load_dataset_artifacts(
    data_root: str,
    dataset: str,
) -> DatasetArtifacts | None:
    """Load GED matrix and metadata for a dataset."""
    ged_path = os.path.join(data_root, "ged_matrices", f"{dataset}.npz")
    if not os.path.isfile(ged_path):
        logger.debug("GED matrix not found: %s", ged_path)
        return None

    data = np.load(ged_path, allow_pickle=True)
    graph_ids = [str(gid) for gid in data["graph_ids"]]
    labels = data.get("labels", None)

    return DatasetArtifacts(
        dataset=dataset,
        ged_matrix=data["ged_matrix"],
        node_counts=data["node_counts"],
        edge_counts=data["edge_counts"],
        graph_ids=graph_ids,
        labels=labels,
    )


def _load_levenshtein_matrix(
    data_root: str,
    dataset: str,
    method: str,
) -> np.ndarray | None:
    """Load a precomputed Levenshtein distance matrix."""
    path = os.path.join(data_root, "levenshtein_matrices", f"{dataset}_{method}.npz")
    if not os.path.isfile(path):
        logger.debug("Levenshtein matrix not found: %s", path)
        return None

    data = np.load(path, allow_pickle=True)
    return data["levenshtein_matrix"]


def _load_canonical_strings(
    data_root: str,
    dataset: str,
    method: str,
) -> dict[str, str] | None:
    """Load canonical/greedy strings for a dataset.

    Returns:
        Dict mapping graph_id -> string, or None if file not found.
    """
    path = os.path.join(data_root, "canonical_strings", f"{dataset}_{method}.json")
    if not os.path.isfile(path):
        logger.debug("Canonical strings not found: %s", path)
        return None

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    strings_dict = raw.get("strings", {})
    return {gid: info["string"] for gid, info in strings_dict.items()}


def _load_correlation_stats(
    correlation_dir: str,
    dataset: str,
    method: str,
) -> CorrelationStats | None:
    """Load correlation statistics JSON from Agent 2."""
    path = os.path.join(correlation_dir, f"{dataset}_{method}_correlation_stats.json")
    if not os.path.isfile(path):
        logger.debug("Correlation stats not found: %s", path)
        return None

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    return CorrelationStats(
        dataset=dataset,
        method=method,
        spearman=raw.get("spearman", {}),
        pearson=raw.get("pearson", {}),
        kendall=raw.get("kendall", {}),
        lins_ccc_raw=raw.get("lins_ccc_raw", 0.0),
        lins_ccc_znorm=raw.get("lins_ccc_znorm", 0.0),
        ols=raw.get("ols", {}),
        precision_at_k=raw.get("precision_at_k", {}),
        n_valid_pairs=raw.get("n_valid_pairs", 0),
    )
