"""All-pairs exact GED computation for IAM Letter datasets.

Uses NetworkX's A* graph edit distance with uniform topology-only costs.
Supports parallelism via ProcessPoolExecutor and checkpointing.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

# Uniform topology-only cost function (from spec Section 1.4)
_NODE_SUBST_COST = lambda n1, n2: 0  # noqa: E731
_NODE_DEL_COST = lambda n: 1  # noqa: E731
_NODE_INS_COST = lambda n: 1  # noqa: E731
_EDGE_SUBST_COST = lambda e1, e2: 0  # noqa: E731
_EDGE_DEL_COST = lambda e: 1  # noqa: E731
_EDGE_INS_COST = lambda e: 1  # noqa: E731


def compute_ged_pair(
    g1: nx.Graph,
    g2: nx.Graph,
    timeout: float = 60.0,
) -> float:
    """Compute exact GED for one pair with timeout.

    Args:
        g1: First graph.
        g2: Second graph.
        timeout: Maximum seconds for the computation.

    Returns:
        Graph edit distance (float). inf on timeout.
    """
    try:
        ged = nx.graph_edit_distance(
            g1,
            g2,
            node_subst_cost=_NODE_SUBST_COST,
            node_del_cost=_NODE_DEL_COST,
            node_ins_cost=_NODE_INS_COST,
            edge_subst_cost=_EDGE_SUBST_COST,
            edge_del_cost=_EDGE_DEL_COST,
            edge_ins_cost=_EDGE_INS_COST,
            timeout=timeout,
        )
        return float(ged) if ged is not None else float("inf")
    except Exception:
        logger.exception("GED computation failed")
        return float("inf")


def _ged_worker(args: tuple) -> tuple[int, int, float]:
    """Worker for ProcessPoolExecutor.

    Args:
        args: (i, j, g1, g2, timeout) tuple.

    Returns:
        (i, j, ged) tuple.
    """
    i, j, g1, g2, timeout = args
    ged = compute_ged_pair(g1, g2, timeout)
    return (i, j, ged)


def compute_all_pairs_ged(
    graphs: list[nx.Graph],
    graph_ids: list[str],
    n_workers: int = 1,
    checkpoint_path: str | None = None,
    checkpoint_interval: int = 10_000,
    timeout_per_pair: float = 60.0,
) -> np.ndarray:
    """Compute all-pairs GED matrix with checkpointing and parallelism.

    Only the upper triangle is computed (GED is symmetric).

    Args:
        graphs: List of NetworkX graphs.
        graph_ids: Graph identifiers (for logging).
        n_workers: Number of parallel workers.
        checkpoint_path: Path to save/load checkpoint .npz file.
        checkpoint_interval: Save checkpoint every N pairs.
        timeout_per_pair: Per-pair timeout in seconds.

    Returns:
        Symmetric GED matrix [N, N] with float64 dtype.
    """
    n = len(graphs)
    total_pairs = n * (n - 1) // 2

    # Load checkpoint if available
    ged_matrix = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(ged_matrix, 0.0)
    completed: set[tuple[int, int]] = set()

    if checkpoint_path and os.path.exists(checkpoint_path):
        data = np.load(checkpoint_path, allow_pickle=True)
        if "ged_matrix" in data and data["ged_matrix"].shape == (n, n):
            ged_matrix = data["ged_matrix"].astype(np.float64)
            for i in range(n):
                for j in range(i + 1, n):
                    if np.isfinite(ged_matrix[i, j]):
                        completed.add((i, j))
            logger.info("Resumed from checkpoint: %d/%d pairs", len(completed), total_pairs)

    # Build work items for remaining pairs
    work_items: list[tuple[int, int, nx.Graph, nx.Graph, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in completed:
                work_items.append((i, j, graphs[i], graphs[j], timeout_per_pair))

    if not work_items:
        logger.info("All %d pairs already computed", total_pairs)
        return ged_matrix

    logger.info(
        "Computing GED: %d remaining pairs (of %d total), %d workers",
        len(work_items),
        total_pairs,
        n_workers,
    )
    t0 = time.perf_counter()
    done_count = len(completed)

    def _save_checkpoint() -> None:
        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            np.savez_compressed(checkpoint_path, ged_matrix=ged_matrix)
            logger.info("Checkpoint saved: %d/%d pairs", done_count, total_pairs)

    if n_workers <= 1:
        # Sequential
        for idx, (i, j, g1, g2, timeout) in enumerate(work_items):
            ged = compute_ged_pair(g1, g2, timeout)
            ged_matrix[i, j] = ged
            ged_matrix[j, i] = ged
            done_count += 1

            if done_count % checkpoint_interval == 0:
                _save_checkpoint()
            if done_count % max(1, total_pairs // 20) == 0:
                elapsed = time.perf_counter() - t0
                rate = (idx + 1) / elapsed
                remaining = (len(work_items) - idx - 1) / rate if rate > 0 else 0
                logger.info(
                    "  %d/%d pairs (%.1f%%), %.1f pairs/s, ~%.0fs remaining",
                    done_count,
                    total_pairs,
                    100.0 * done_count / total_pairs,
                    rate,
                    remaining,
                )
    else:
        # Parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_ged_worker, item): item[:2] for item in work_items}

            for fut in as_completed(futures):
                try:
                    i, j, ged = fut.result()
                    ged_matrix[i, j] = ged
                    ged_matrix[j, i] = ged
                except Exception:
                    i, j = futures[fut]
                    logger.exception("Worker failed for pair (%d, %d)", i, j)

                done_count += 1
                if done_count % checkpoint_interval == 0:
                    _save_checkpoint()
                if done_count % max(1, total_pairs // 20) == 0:
                    elapsed = time.perf_counter() - t0
                    logger.info(
                        "  %d/%d pairs (%.1f%%)",
                        done_count,
                        total_pairs,
                        100.0 * done_count / total_pairs,
                    )

    # Final checkpoint
    _save_checkpoint()

    elapsed = time.perf_counter() - t0
    n_finite = np.isfinite(ged_matrix[np.triu_indices(n, k=1)]).sum()
    logger.info(
        "GED computation complete: %d pairs in %.1fs (%.1f pairs/s), %d finite",
        total_pairs,
        elapsed,
        total_pairs / elapsed if elapsed > 0 else 0,
        n_finite,
    )
    return ged_matrix


def save_ged_matrix(
    ged_matrix: np.ndarray,
    graph_ids: list[str],
    labels: list[str],
    node_counts: list[int],
    edge_counts: list[int],
    output_path: str,
    dataset_name: str,
    ged_method: str,
    ged_cost_function: str,
    source: str,
    n_max_filter: int,
    n_dropped: int,
) -> None:
    """Save GED matrix in the standardized .npz format.

    Args:
        ged_matrix: Pairwise GED matrix [N, N].
        graph_ids: Graph identifiers.
        labels: Class labels.
        node_counts: Node count per graph.
        edge_counts: Edge count per graph.
        output_path: Path for the .npz file.
        dataset_name: Dataset name for metadata.
        ged_method: e.g. 'exact_a_star' or 'precomputed'.
        ged_cost_function: e.g. 'uniform_topology_only'.
        source: e.g. 'networkx' or 'pyg_bai2019'.
        n_max_filter: Node-count filter threshold.
        n_dropped: Number of graphs dropped.
    """
    n = len(graph_ids)
    n_valid = int(np.isfinite(ged_matrix[np.triu_indices(n, k=1)]).sum())

    metadata = json.dumps(
        {
            "dataset": dataset_name,
            "ged_method": ged_method,
            "ged_cost_function": ged_cost_function,
            "source": source,
            "n_graphs": n,
            "n_valid_pairs": n_valid,
            "n_max_filter": n_max_filter,
            "n_dropped": n_dropped,
        }
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        ged_matrix=ged_matrix,
        node_counts=np.array(node_counts, dtype=np.int32),
        edge_counts=np.array(edge_counts, dtype=np.int32),
        graph_ids=np.array(graph_ids, dtype=str),
        labels=np.array(labels, dtype=str),
        metadata=np.array(metadata),
    )
    logger.info("Saved GED matrix to %s (%d graphs, %d valid pairs)", output_path, n, n_valid)
