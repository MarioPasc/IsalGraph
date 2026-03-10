"""Dataset filtering by node count and connectivity.

Ensures all graphs are within the exhaustive canonical search feasibility
range and are connected (required by GraphToString / canonical_string).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result of filtering a dataset."""

    n_raw: int
    n_kept: int
    n_dropped_size: int
    n_dropped_disconnected: int
    n_dropped_trivial: int
    kept_indices: list[int]
    dropped_indices: list[int]
    dropped_reasons: dict[int, str] = field(default_factory=dict)
    dropped_graph_ids: list[str] = field(default_factory=list)
    dropped_node_counts: list[int] = field(default_factory=list)
    node_count_histogram_raw: dict[int, int] = field(default_factory=dict)
    node_count_histogram_kept: dict[int, int] = field(default_factory=dict)


def filter_graphs(
    graphs: list[nx.Graph],
    graph_ids: list[str],
    n_max: int,
    require_connected: bool = True,
    min_nodes: int = 2,
) -> FilterResult:
    """Filter graphs by node count, connectivity, and minimum size.

    Args:
        graphs: List of NetworkX graphs.
        graph_ids: Corresponding graph identifiers.
        n_max: Maximum allowed node count (inclusive).
        require_connected: If True, drop disconnected graphs.
        min_nodes: Minimum node count (inclusive). Graphs below this
            produce trivial canonical strings.

    Returns:
        FilterResult with indices and statistics.
    """
    kept_idx: list[int] = []
    dropped_idx: list[int] = []
    dropped_reasons: dict[int, str] = {}
    dropped_ids: list[str] = []
    dropped_counts: list[int] = []
    n_dropped_size = 0
    n_dropped_disconnected = 0
    n_dropped_trivial = 0

    raw_counts: list[int] = []
    kept_counts: list[int] = []

    for i, (g, gid) in enumerate(zip(graphs, graph_ids, strict=True)):
        n = g.number_of_nodes()
        raw_counts.append(n)

        if n < min_nodes:
            dropped_idx.append(i)
            dropped_reasons[i] = f"trivial ({n} nodes < min_nodes={min_nodes})"
            dropped_ids.append(gid)
            dropped_counts.append(n)
            n_dropped_trivial += 1
            logger.debug("Dropped %s: %d nodes < min_nodes=%d", gid, n, min_nodes)
            continue

        if n > n_max:
            dropped_idx.append(i)
            dropped_reasons[i] = f"too_large ({n} nodes > n_max={n_max})"
            dropped_ids.append(gid)
            dropped_counts.append(n)
            n_dropped_size += 1
            logger.debug("Dropped %s: %d nodes > n_max=%d", gid, n, n_max)
            continue

        if require_connected and not nx.is_connected(g):
            dropped_idx.append(i)
            dropped_reasons[i] = "disconnected"
            dropped_ids.append(gid)
            dropped_counts.append(n)
            n_dropped_disconnected += 1
            logger.debug("Dropped %s: disconnected (%d nodes)", gid, n)
            continue

        kept_idx.append(i)
        kept_counts.append(n)

    result = FilterResult(
        n_raw=len(graphs),
        n_kept=len(kept_idx),
        n_dropped_size=n_dropped_size,
        n_dropped_disconnected=n_dropped_disconnected,
        n_dropped_trivial=n_dropped_trivial,
        kept_indices=kept_idx,
        dropped_indices=dropped_idx,
        dropped_reasons=dropped_reasons,
        dropped_graph_ids=dropped_ids,
        dropped_node_counts=dropped_counts,
        node_count_histogram_raw=dict(sorted(Counter(raw_counts).items())),
        node_count_histogram_kept=dict(sorted(Counter(kept_counts).items())),
    )

    logger.info(
        "Filter %d -> %d graphs (dropped: %d size, %d disconnected, %d trivial)",
        result.n_raw,
        result.n_kept,
        n_dropped_size,
        n_dropped_disconnected,
        n_dropped_trivial,
    )
    return result


def extract_ged_submatrix(
    ged_matrix: np.ndarray,
    kept_indices: list[int],
) -> np.ndarray:
    """Extract submatrix for kept graphs.

    Args:
        ged_matrix: Full GED matrix [N, N].
        kept_indices: Indices of kept graphs.

    Returns:
        Submatrix [K, K] where K = len(kept_indices).
    """
    idx = np.array(kept_indices)
    return ged_matrix[np.ix_(idx, idx)]


def build_filtering_report(
    results: dict[str, FilterResult],
    n_max: int,
) -> dict:
    """Build the filtering_report.json content.

    Args:
        results: Mapping of dataset name to FilterResult.
        n_max: The node-count threshold used.

    Returns:
        Dict ready for JSON serialization.
    """
    report: dict = {"n_max_filter": n_max, "datasets": {}}
    for name, fr in results.items():
        max_raw = max(fr.node_count_histogram_raw.keys()) if fr.node_count_histogram_raw else 0
        max_kept = max(fr.node_count_histogram_kept.keys()) if fr.node_count_histogram_kept else 0
        report["datasets"][name] = {
            "n_raw": fr.n_raw,
            "n_kept": fr.n_kept,
            "n_dropped": fr.n_raw - fr.n_kept,
            "n_dropped_size": fr.n_dropped_size,
            "n_dropped_disconnected": fr.n_dropped_disconnected,
            "n_dropped_trivial": fr.n_dropped_trivial,
            "pct_kept": round(100.0 * fr.n_kept / fr.n_raw, 1) if fr.n_raw > 0 else 0.0,
            "max_nodes_raw": max_raw,
            "max_nodes_kept": max_kept,
            "dropped_graph_ids": fr.dropped_graph_ids,
            "dropped_node_counts": sorted(set(fr.dropped_node_counts)),
            "node_count_histogram_raw": fr.node_count_histogram_raw,
            "node_count_histogram_kept": fr.node_count_histogram_kept,
        }
    return report
