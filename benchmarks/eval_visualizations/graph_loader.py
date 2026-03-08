# ruff: noqa: N803
"""Utility for loading original NetworkX graph objects by dataset and graph ID."""

from __future__ import annotations

import logging
import os

import networkx as nx

logger = logging.getLogger(__name__)


def load_graph_lookup(
    source_root: str,
    dataset: str,
) -> dict[str, nx.Graph]:
    """Load original graphs and return a graph_id to nx.Graph mapping.

    Args:
        source_root: Root directory containing source data (Letter/, LINUX/, AIDS/).
        dataset: Dataset name (e.g. "iam_letter_low", "linux", "aids").

    Returns:
        Dictionary mapping graph_id strings to NetworkX Graph objects.
    """
    if dataset.startswith("iam_letter_"):
        level = dataset.split("_")[-1].upper()
        letter_dir = os.path.join(source_root, "Letter")
        if not os.path.isdir(letter_dir):
            logger.warning("IAM Letter directory not found: %s", letter_dir)
            return {}
        from benchmarks.eval_setup.iam_letter_loader import load_iam_letter

        result = load_iam_letter(letter_dir, level)
        return dict(zip(result.graph_ids, result.graphs, strict=True))

    if dataset in ("linux", "aids"):
        from benchmarks.eval_setup.graphedx_loader import load_graphedx_dataset

        result = load_graphedx_dataset(dataset.upper(), source_root)
        return dict(zip(result.graph_ids, result.graphs, strict=True))

    logger.warning("Unknown dataset for graph loading: %s", dataset)
    return {}
