# ruff: noqa: E402
"""Weisfeiler-Lehman subtree kernel distance computation.

Computes WL kernel matrix and derives a kernel-induced distance matrix.
The WL kernel operates on graphs (not strings), so one distance matrix
is computed per dataset (algorithm-independent).

Uses grakel for the WL kernel computation. Includes a numpy 2.x
compatibility shim for grakel 0.1.10 (np.ComplexWarning removed).
"""

from __future__ import annotations

import json
import logging
import os
import time

import numpy as np

logger = logging.getLogger(__name__)


def _apply_grakel_numpy_shim() -> None:
    """Fix grakel numpy 2.x compatibility.

    grakel 0.1.10 imports numpy.ComplexWarning which was removed in
    numpy 2.0. This shim restores the attribute as DeprecationWarning.
    """
    if not hasattr(np, "ComplexWarning"):
        np.ComplexWarning = DeprecationWarning  # type: ignore[attr-defined]


def nx_graphs_to_grakel(graphs: list) -> list:
    """Convert NetworkX graphs to grakel format.

    grakel expects each graph as a tuple (adjacency_dict, node_labels_dict).
    Since IsalGraph uses unlabeled nodes, we assign a uniform label '0' to all.

    Args:
        graphs: List of NetworkX graphs.

    Returns:
        List of (adjacency_dict, node_labels_dict) tuples for grakel.
    """
    grakel_graphs = []
    for g in graphs:
        nodes = list(g.nodes())
        adj_dict = {}
        for u in nodes:
            adj_dict[u] = list(g.neighbors(u))
        node_labels = {u: "0" for u in nodes}
        grakel_graphs.append((adj_dict, node_labels))
    return grakel_graphs


def compute_wl_kernel_matrix(
    graphs: list,
    n_iter: int = 5,
) -> np.ndarray:
    """Compute the WL subtree kernel matrix.

    Args:
        graphs: List of grakel-format graphs.
        n_iter: Number of WL iterations.

    Returns:
        Kernel matrix K of shape (N, N), not normalized.
    """
    _apply_grakel_numpy_shim()
    from grakel import WeisfeilerLehman
    from grakel.kernels import VertexHistogram

    wl = WeisfeilerLehman(
        n_iter=n_iter,
        base_graph_kernel=VertexHistogram,
        normalize=False,
    )
    kernel = wl.fit_transform(graphs)
    return np.array(kernel, dtype=np.float64)


def kernel_to_distance(kernel_matrix: np.ndarray) -> np.ndarray:
    """Convert a PSD kernel matrix to a distance matrix.

    Uses the identity: d(i,j) = sqrt(K(i,i) + K(j,j) - 2*K(i,j))
    with clamping to avoid negative values from numerical noise.

    Args:
        kernel_matrix: Positive semi-definite kernel matrix (N, N).

    Returns:
        Distance matrix of shape (N, N).
    """
    diag = np.diag(kernel_matrix)
    # d^2(i,j) = K(i,i) + K(j,j) - 2*K(i,j)
    d_sq = diag[:, None] + diag[None, :] - 2.0 * kernel_matrix
    # Clamp numerical noise
    d_sq = np.maximum(d_sq, 0.0)
    return np.sqrt(d_sq)


def compute_wl_kernel_distance(
    nx_graphs: list,
    n_iter: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute WL kernel distance matrix from NetworkX graphs.

    Args:
        nx_graphs: List of NetworkX graphs.
        n_iter: Number of WL iterations.

    Returns:
        (distance_matrix, kernel_matrix) tuple.
    """
    logger.info("Converting %d graphs to grakel format...", len(nx_graphs))
    grakel_graphs = nx_graphs_to_grakel(nx_graphs)

    logger.info("Computing WL kernel matrix (n_iter=%d)...", n_iter)
    t0 = time.perf_counter()
    k_matrix = compute_wl_kernel_matrix(grakel_graphs, n_iter=n_iter)
    elapsed = time.perf_counter() - t0
    logger.info("WL kernel computed in %.2fs", elapsed)

    logger.info("Converting kernel to distance matrix...")
    d_matrix = kernel_to_distance(k_matrix)

    return d_matrix, k_matrix


def save_wl_kernel_matrix(
    distance_matrix: np.ndarray,
    kernel_matrix: np.ndarray,
    graph_ids: list[str],
    n_iter: int,
    output_path: str,
) -> None:
    """Save WL kernel and distance matrices.

    Args:
        distance_matrix: WL distance matrix (N, N).
        kernel_matrix: WL kernel matrix (N, N).
        graph_ids: Graph identifiers.
        n_iter: Number of WL iterations used.
        output_path: Output .npz path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    metadata = {
        "n_iter": n_iter,
        "n_graphs": len(graph_ids),
        "kernel_type": "weisfeiler_lehman_subtree",
        "base_kernel": "vertex_histogram",
        "normalized": False,
    }

    np.savez_compressed(
        output_path,
        distance_matrix=distance_matrix,
        kernel_matrix=kernel_matrix,
        graph_ids=np.array(graph_ids, dtype=str),
        metadata=np.array(json.dumps(metadata)),
    )
    logger.info("Saved WL kernel data to %s (%d graphs)", output_path, len(graph_ids))


def compute_wl_effective_dimensionality(kernel_matrix: np.ndarray) -> dict:
    """Compute diagnostic statistics for the WL kernel matrix.

    Args:
        kernel_matrix: WL kernel matrix (N, N).

    Returns:
        Dict with effective_rank, n_distinct_distances, distance_variance.
    """
    dist = kernel_to_distance(kernel_matrix)

    # Effective rank: number of eigenvalues > 1% of max eigenvalue
    eigenvalues = np.linalg.eigvalsh(kernel_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]
    max_eig = eigenvalues[0] if len(eigenvalues) > 0 else 0.0
    threshold = 0.01 * max_eig if max_eig > 0 else 0.0
    effective_rank = int(np.sum(eigenvalues > threshold))

    # Distance statistics (upper triangle only)
    n = dist.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    d_upper = dist[triu_idx]

    n_distinct = len(np.unique(np.round(d_upper, decimals=8)))
    d_var = float(np.var(d_upper))

    return {
        "effective_rank": effective_rank,
        "n_distinct_distances": n_distinct,
        "distance_variance": round(d_var, 6),
        "max_eigenvalue": round(float(max_eig), 6),
        "min_positive_eigenvalue": round(float(eigenvalues[eigenvalues > threshold][-1]), 6)
        if np.any(eigenvalues > threshold)
        else 0.0,
    }
