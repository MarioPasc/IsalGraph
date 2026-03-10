# ruff: noqa: N803, N806
"""Loader for Agent 2 embedding evaluation outputs.

Loads embedding stats JSONs, SMACOF coordinate NPZs, and eigenvalue NPZs
into typed dataclasses for visualization.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

import numpy as np

from benchmarks.eval_visualizations.result_loader import ALL_DATASETS, METHODS

logger = logging.getLogger(__name__)

# Dimensions used in the embedding evaluation
DIMENSIONS = [2, 3, 5, 10]
# Dimensions used for Procrustes / Shepard (subset)
PROCRUSTES_DIMS = [2, 5]


@dataclass
class SmacofEntry:
    """SMACOF result for one distance source at one dimension."""

    stress_1: float
    n_iterations: int
    converged: bool


@dataclass
class ProcrustesEntry:
    """Procrustes test result at one dimension."""

    m_squared: float
    p_value: float
    n_more_extreme: int


@dataclass
class ShepardEntry:
    """Shepard diagram statistics at one dimension."""

    ged_r_squared: float
    ged_monotonic_r_squared: float
    lev_r_squared: float
    lev_monotonic_r_squared: float


@dataclass
class DatasetEmbeddingStats:
    """All embedding statistics for one dataset."""

    dataset: str
    n_graphs: int
    n_finite: int
    has_inf_ged: bool

    # CMDS negative eigenvalue ratios
    ged_nev_ratio: float
    lev_nev_ratio: dict[str, float] = field(default_factory=dict)  # method -> ratio

    # SMACOF stress per dimension
    ged_smacof: dict[int, SmacofEntry] = field(default_factory=dict)  # dim -> entry
    lev_smacof: dict[tuple[str, int], SmacofEntry] = field(
        default_factory=dict
    )  # (method, dim) -> entry

    # Procrustes per method × dim
    procrustes: dict[tuple[str, int], ProcrustesEntry] = field(default_factory=dict)

    # Shepard per method × dim
    shepard: dict[tuple[str, int], ShepardEntry] = field(default_factory=dict)


@dataclass
class EmbeddingData:
    """Container for all embedding evaluation data."""

    stats: dict[str, DatasetEmbeddingStats] = field(default_factory=dict)
    cross_analysis: dict = field(default_factory=dict)


def load_embedding_data(embedding_dir: str) -> EmbeddingData:
    """Load all embedding evaluation outputs.

    Args:
        embedding_dir: Root of eval_embedding output (contains stats/, raw/).

    Returns:
        EmbeddingData with graceful degradation for missing files.
    """
    result = EmbeddingData()
    stats_dir = os.path.join(embedding_dir, "stats")

    for dataset in ALL_DATASETS:
        ds_stats = _load_dataset_embedding_stats(stats_dir, dataset)
        if ds_stats is not None:
            result.stats[dataset] = ds_stats

    # Cross-dataset analysis
    cross_path = os.path.join(stats_dir, "cross_dataset_analysis.json")
    if os.path.isfile(cross_path):
        with open(cross_path, encoding="utf-8") as f:
            result.cross_analysis = json.load(f)

    logger.info("Loaded embedding stats for %d datasets", len(result.stats))
    return result


def load_smacof_coords(
    embedding_dir: str,
    dataset: str,
    distance_type: str,
    dim: int,
) -> np.ndarray | None:
    """Load SMACOF coordinate array.

    Args:
        embedding_dir: Root of eval_embedding output.
        dataset: Dataset name.
        distance_type: One of "ged", "lev_exhaustive", "lev_greedy".
        dim: Embedding dimension (2, 3, 5, or 10).

    Returns:
        Array of shape (n_graphs, dim) or None.
    """
    path = os.path.join(embedding_dir, "raw", f"{dataset}_{distance_type}_smacof_{dim}d.npz")
    if not os.path.isfile(path):
        logger.debug("SMACOF coords not found: %s", path)
        return None

    data = np.load(path)
    return data["coords"]


def _load_dataset_embedding_stats(
    stats_dir: str,
    dataset: str,
) -> DatasetEmbeddingStats | None:
    """Load embedding stats JSON for one dataset."""
    path = os.path.join(stats_dir, f"{dataset}_embedding_stats.json")
    if not os.path.isfile(path):
        logger.debug("Embedding stats not found: %s", path)
        return None

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    ds = DatasetEmbeddingStats(
        dataset=dataset,
        n_graphs=raw["n_graphs"],
        n_finite=raw["n_finite_submatrix"],
        has_inf_ged=raw["has_inf_ged"],
        ged_nev_ratio=raw["cmds_ged"]["nev_ratio"],
    )

    # GED SMACOF
    for dim_str, info in raw.get("smacof_ged", {}).items():
        dim = int(dim_str)
        ds.ged_smacof[dim] = SmacofEntry(
            stress_1=info["stress_1"],
            n_iterations=info["n_iterations"],
            converged=info["converged"],
        )

    # Per-method data
    for method in METHODS:
        mdata = raw.get("methods", {}).get(method)
        if mdata is None:
            continue

        # LEV NEV ratio
        cmds_lev = mdata.get("cmds_lev", {})
        ds.lev_nev_ratio[method] = cmds_lev.get("nev_ratio", 0.0)

        # LEV SMACOF
        for dim_str, info in mdata.get("smacof", {}).items():
            dim = int(dim_str)
            ds.lev_smacof[(method, dim)] = SmacofEntry(
                stress_1=info["lev_stress_1"],
                n_iterations=info["lev_n_iterations"],
                converged=info["lev_converged"],
            )

        # Procrustes
        for dim_str, info in mdata.get("procrustes", {}).items():
            dim = int(dim_str)
            ds.procrustes[(method, dim)] = ProcrustesEntry(
                m_squared=info["m_squared"],
                p_value=info["p_value"],
                n_more_extreme=info["n_more_extreme"],
            )

        # Shepard
        for dim_str, info in mdata.get("shepard", {}).items():
            dim = int(dim_str)
            ds.shepard[(method, dim)] = ShepardEntry(
                ged_r_squared=info["ged_r_squared"],
                ged_monotonic_r_squared=info["ged_monotonic_r_squared"],
                lev_r_squared=info["lev_r_squared"],
                lev_monotonic_r_squared=info["lev_monotonic_r_squared"],
            )

    return ds
