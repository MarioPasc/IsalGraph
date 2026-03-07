"""Validate round-trip isomorphism using original source graphs.

For each graph G in the source data:
1. Load the original NetworkX graph from source files.
2. Load the canonical/greedy string from eval data.
3. Reconstruct G' = S2G(string) and convert to NetworkX.
4. Assert nx.is_isomorphic(G, G').

This test requires --source-dir pointing to the original datasets.
"""

from __future__ import annotations

import os

import pytest

nx = pytest.importorskip("networkx")

from isalgraph.adapters.networkx_adapter import NetworkXAdapter
from isalgraph.core.string_to_graph import StringToGraph
from tests.eval_validation.conftest import DATASETS, METHODS, EvalData

pytestmark = pytest.mark.eval_validation

# Sample size for isomorphism checks (VF2 is fast for small graphs)
ISOMORPHISM_SAMPLE_SIZE = 200


def _load_source_graphs(dataset: str, source_dir: str, graph_ids: list[str]) -> dict[str, nx.Graph]:
    """Load original graphs from source data."""
    if dataset.startswith("iam_letter_"):
        return _load_iam_letter_graphs(dataset, source_dir, graph_ids)
    if dataset in ("linux", "aids"):
        return _load_graphedx_graphs(dataset, source_dir, graph_ids)
    return {}


def _load_iam_letter_graphs(
    dataset: str, source_dir: str, graph_ids: list[str]
) -> dict[str, nx.Graph]:
    """Load IAM Letter graphs from GXL files."""
    from benchmarks.eval_setup.iam_letter_loader import parse_gxl

    level = dataset.replace("iam_letter_", "").upper()
    level_dir = os.path.join(source_dir, "Letter", level)
    if not os.path.isdir(level_dir):
        return {}

    result = {}
    for gid in graph_ids:
        gxl_path = os.path.join(level_dir, f"{gid}.gxl")
        if os.path.isfile(gxl_path):
            result[gid] = parse_gxl(gxl_path)
    return result


def _load_graphedx_graphs(
    dataset: str, source_dir: str, graph_ids: list[str]
) -> dict[str, nx.Graph]:
    """Load GraphEdX graphs from .pt files."""
    import torch

    name = dataset.upper()
    dataset_dir = os.path.join(source_dir, name)
    if not os.path.isdir(dataset_dir):
        return {}

    # Load all splits and build ID -> graph mapping
    all_graphs: dict[str, nx.Graph] = {}
    for split in ("train", "val", "test"):
        graphs_path = os.path.join(dataset_dir, f"{split}_graphs.pt")
        if not os.path.isfile(graphs_path):
            continue
        nx_graphs = torch.load(graphs_path, weights_only=False)
        for idx, g in enumerate(nx_graphs):
            gid = f"{dataset}_{split}_{idx:04d}"
            # Strip attributes for topology-only comparison
            clean = nx.Graph()
            clean.add_nodes_from(range(g.number_of_nodes()))
            for u, v in g.edges():
                if u != v:
                    clean.add_edge(u, v)
            all_graphs[gid] = clean

    return {gid: all_graphs[gid] for gid in graph_ids if gid in all_graphs}


class TestRoundTripIsomorphism:
    """S2G(string) must produce a graph isomorphic to the original."""

    @pytest.mark.parametrize("dataset", DATASETS)
    @pytest.mark.parametrize("method", METHODS)
    def test_isomorphism(
        self, eval_data: EvalData, source_dir: str | None, dataset: str, method: str
    ):
        if source_dir is None:
            pytest.skip("--source-dir not provided")

        key = (dataset, method)
        if key not in eval_data.canonical:
            pytest.skip(f"No {method} strings for {dataset}")

        ds = eval_data.canonical[key]
        sample_ids = ds.graph_ids[:ISOMORPHISM_SAMPLE_SIZE]

        # Load original graphs
        originals = _load_source_graphs(dataset, source_dir, sample_ids)
        if not originals:
            pytest.skip(f"Could not load source graphs for {dataset}")

        adapter = NetworkXAdapter()
        failures = []
        n_tested = 0

        for gid in sample_ids:
            if gid not in originals:
                continue
            original = originals[gid]
            string = ds.strings[gid]

            # Reconstruct
            converter = StringToGraph(string, directed_graph=False)
            sg, _ = converter.run()
            reconstructed = adapter.to_external(sg)

            if not nx.is_isomorphic(original, reconstructed):
                failures.append(
                    f"{gid}: NOT isomorphic "
                    f"(original: {original.number_of_nodes()}n/{original.number_of_edges()}e, "
                    f"reconstructed: {reconstructed.number_of_nodes()}n/"
                    f"{reconstructed.number_of_edges()}e, "
                    f"string='{string[:30]}...')"
                )
            n_tested += 1

        assert not failures, (
            f"{dataset}/{method}: {len(failures)}/{n_tested} "
            f"round-trip isomorphism failures:\n" + "\n".join(failures[:10])
        )
