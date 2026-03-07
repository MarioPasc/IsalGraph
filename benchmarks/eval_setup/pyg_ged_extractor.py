"""PyTorch Geometric GEDDataset loader and GED matrix extraction.

Loads LINUX and ALKANE datasets from PyG, converts graphs to NetworkX
(topology only), and extracts precomputed GED matrices.

Reference:
    Bai et al. (2019). SimGNN: A Neural Network Approach to Fast
    Graph Similarity Computation. WSDM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PyGDatasetResult:
    """Result of loading a PyG GED dataset."""

    graphs: list[nx.Graph] = field(default_factory=list)
    graph_ids: list[str] = field(default_factory=list)
    ged_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    n_graphs: int = 0
    dataset_name: str = ""
    train_size: int = 0
    test_size: int = 0


def _pyg_data_to_nx(data, n_nodes: int) -> nx.Graph:
    """Convert a PyG Data object to NetworkX graph (topology only).

    Args:
        data: PyG Data object with edge_index.
        n_nodes: Number of nodes.

    Returns:
        Undirected NetworkX graph with integer node labels.
    """
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))

    edge_index = data.edge_index
    if edge_index is not None and edge_index.size(1) > 0:
        for i in range(edge_index.size(1)):
            src = int(edge_index[0, i])
            tgt = int(edge_index[1, i])
            if src != tgt:
                g.add_edge(src, tgt)

    return g


def load_pyg_ged_dataset(
    name: str,
    root: str,
) -> PyGDatasetResult:
    """Load a PyG GEDDataset and extract graphs + precomputed GED.

    Args:
        name: Dataset name ('LINUX' or 'ALKANE').
        root: Download/cache root directory.

    Returns:
        PyGDatasetResult with graphs, IDs, and raw GED matrix.

    Raises:
        ImportError: If torch or torch_geometric is not installed.
    """
    try:
        from torch_geometric.datasets import GEDDataset
    except ImportError as exc:
        raise ImportError(
            "PyTorch and PyTorch Geometric are required. Install with: pip install isalgraph[pyg]"
        ) from exc

    logger.info("Loading PyG GEDDataset '%s' from %s ...", name, root)

    # Load train and test splits
    train_dataset = GEDDataset(root=root, name=name, train=True)
    test_dataset = GEDDataset(root=root, name=name, train=False)

    result = PyGDatasetResult(
        dataset_name=name,
        train_size=len(train_dataset),
        test_size=len(test_dataset),
    )

    # Convert all graphs to NetworkX (train first, then test)
    for idx, data in enumerate(train_dataset):
        n = int(data.num_nodes) if data.num_nodes is not None else 0
        g = _pyg_data_to_nx(data, n)
        result.graphs.append(g)
        result.graph_ids.append(f"train_{idx:04d}")

    for idx, data in enumerate(test_dataset):
        n = int(data.num_nodes) if data.num_nodes is not None else 0
        g = _pyg_data_to_nx(data, n)
        result.graphs.append(g)
        result.graph_ids.append(f"test_{idx:04d}")

    result.n_graphs = len(result.graphs)

    # Extract precomputed GED matrix
    # PyG stores GED as a [N, N] tensor on the train dataset object
    ged_tensor = train_dataset.ged
    if ged_tensor is not None:
        ged_np = ged_tensor.numpy().astype(np.float64)
        # Replace 0 on off-diagonal with inf only where truly missing
        # PyG uses 0 for unavailable pairs in some versions
        # Actually, PyG GEDDataset uses actual GED values; 0 means identical graphs
        # inf/nan entries are genuinely unavailable
        ged_np = np.where(np.isnan(ged_np), np.inf, ged_np)
        result.ged_matrix = ged_np
        logger.info(
            "GED matrix shape: %s, finite entries: %d/%d",
            ged_np.shape,
            np.isfinite(ged_np).sum(),
            ged_np.size,
        )
    else:
        logger.warning("No precomputed GED matrix found for %s", name)
        result.ged_matrix = np.full((result.n_graphs, result.n_graphs), np.inf)

    logger.info(
        "Loaded %s: %d graphs (%d train + %d test)",
        name,
        result.n_graphs,
        result.train_size,
        result.test_size,
    )
    return result
