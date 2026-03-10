"""GraphEdX dataset loader for LINUX and AIDS.

Loads graphs and precomputed GED matrices from GraphEdX .pt files
(Jain et al., NeurIPS 2024). The .pt files contain NetworkX Graph
objects directly and GED results as upper-triangle tuple lists.

Reference:
    Jain, E., Roy, I., Meher, S., Chakrabarti, S., & De, A. (2024).
    Graph Edit Distance with General Costs Using Neural Set Divergence.
    NeurIPS 2024. arXiv:2409.17687.

Original data:
    Bai, Y. et al. (2019). SimGNN. WSDM 2019.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


@dataclass
class GraphEdXDatasetResult:
    """Result of loading a GraphEdX dataset."""

    graphs: list[nx.Graph] = field(default_factory=list)
    graph_ids: list[str] = field(default_factory=list)
    ged_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    n_graphs: int = 0
    dataset_name: str = ""
    split_sizes: dict[str, int] = field(default_factory=dict)
    n_valid_ged_pairs: int = 0


def _load_split_results(result_path: str, n: int) -> np.ndarray:
    """Decode GraphEdX result tuples into a GED matrix.

    GraphEdX stores within-split GED as a list of (ged, ged, cost)
    tuples in upper-triangle order including diagonal, with length
    n*(n+1)/2.

    Args:
        result_path: Path to {split}_result.pt file.
        n: Number of graphs in the split.

    Returns:
        GED matrix [n, n] with diagonal = 0.
    """
    import torch

    results = torch.load(result_path, weights_only=False)
    expected_len = n * (n + 1) // 2
    if len(results) != expected_len:
        raise ValueError(
            f"Expected {expected_len} result entries for {n} graphs, got {len(results)}"
        )

    ged_matrix = np.zeros((n, n), dtype=np.float64)
    idx = 0
    for i in range(n):
        for j in range(i, n):
            raw_ged = float(results[idx][0])
            # Round near-integer values (float precision artifacts like 4.999999)
            rounded = round(raw_ged)
            if abs(raw_ged - rounded) < 0.01:
                raw_ged = float(rounded)
            ged_matrix[i, j] = raw_ged
            ged_matrix[j, i] = raw_ged
            idx += 1

    return ged_matrix


def _strip_node_attributes(g: nx.Graph) -> nx.Graph:
    """Return a copy of g with all node/edge attributes removed.

    GraphEdX graphs may carry residual attributes (e.g., feature
    tensors). We strip everything for topology-only analysis.
    """
    clean = nx.Graph()
    clean.add_nodes_from(range(g.number_of_nodes()))
    for u, v in g.edges():
        if u != v:
            clean.add_edge(u, v)
    return clean


def load_graphedx_dataset(
    name: str,
    source_dir: str,
) -> GraphEdXDatasetResult:
    """Load a GraphEdX dataset (LINUX or AIDS).

    Merges train/val/test splits. Within-split GED pairs are populated;
    cross-split pairs are set to inf.

    Args:
        name: Dataset directory name ('LINUX' or 'AIDS').
        source_dir: Path to the directory containing the .pt files.

    Returns:
        GraphEdXDatasetResult with graphs, IDs, and merged GED matrix.
    """
    import torch

    dataset_dir = os.path.join(source_dir, name)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"GraphEdX dataset not found: {dataset_dir}")

    logger.info("Loading GraphEdX '%s' from %s ...", name, dataset_dir)

    result = GraphEdXDatasetResult(dataset_name=name)
    name_lower = name.lower()

    # Load all splits
    split_offsets: dict[str, tuple[int, int]] = {}  # split -> (start_idx, end_idx)
    split_ged_matrices: dict[str, np.ndarray] = {}

    offset = 0
    for split in SPLITS:
        graphs_path = os.path.join(dataset_dir, f"{split}_graphs.pt")
        result_path = os.path.join(dataset_dir, f"{split}_result.pt")

        if not os.path.isfile(graphs_path):
            logger.warning("Split file not found: %s", graphs_path)
            continue

        nx_graphs = torch.load(graphs_path, weights_only=False)
        n_split = len(nx_graphs)

        for idx, g in enumerate(nx_graphs):
            clean_g = _strip_node_attributes(g)
            result.graphs.append(clean_g)
            result.graph_ids.append(f"{name_lower}_{split}_{idx:04d}")

        split_offsets[split] = (offset, offset + n_split)
        result.split_sizes[split] = n_split

        # Load within-split GED
        if os.path.isfile(result_path):
            split_ged = _load_split_results(result_path, n_split)
            split_ged_matrices[split] = split_ged

        offset += n_split
        logger.info("  %s: %d graphs", split, n_split)

    result.n_graphs = len(result.graphs)

    # Build merged GED matrix (cross-split = inf)
    n_total = result.n_graphs
    merged_ged = np.full((n_total, n_total), np.inf, dtype=np.float64)

    n_valid = 0
    for split, (start, end) in split_offsets.items():
        if split in split_ged_matrices:
            split_ged = split_ged_matrices[split]
            merged_ged[start:end, start:end] = split_ged
            n_split = end - start
            # Count valid pairs (upper triangle, excluding diagonal)
            n_valid += n_split * (n_split - 1) // 2

    result.ged_matrix = merged_ged
    result.n_valid_ged_pairs = n_valid

    logger.info(
        "Loaded %s: %d graphs total, %d valid GED pairs (within-split only)",
        name,
        result.n_graphs,
        result.n_valid_ged_pairs,
    )
    return result
