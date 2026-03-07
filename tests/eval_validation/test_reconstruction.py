"""Validate that S2G(string) reconstructs graphs with correct node/edge counts.

For each generated string, StringToGraph must produce a graph whose
node_count and edge_count match the metadata from the original graph.
"""

from __future__ import annotations

import pytest

from isalgraph.core.string_to_graph import StringToGraph
from tests.eval_validation.conftest import DATASETS, METHODS, EvalData

pytestmark = pytest.mark.eval_validation


def _s2g(string: str):
    """Convert IsalGraph string to SparseGraph."""
    converter = StringToGraph(string, directed_graph=False)
    sg, _ = converter.run()
    return sg


class TestNodeCountPreservation:
    """S2G(string) must produce a graph with the correct number of nodes."""

    @pytest.mark.parametrize("dataset", DATASETS)
    @pytest.mark.parametrize("method", METHODS)
    def test_node_count(self, eval_data: EvalData, dataset: str, method: str):
        key = (dataset, method)
        if key not in eval_data.canonical:
            pytest.skip(f"No {method} strings for {dataset}")

        ds = eval_data.canonical[key]
        mismatches = []
        for gid, string in ds.strings.items():
            expected_nodes = ds.node_counts.get(gid)
            if expected_nodes is None:
                continue
            sg = _s2g(string)
            actual_nodes = sg.node_count()
            if actual_nodes != expected_nodes:
                mismatches.append(
                    f"{gid}: expected {expected_nodes} nodes, got {actual_nodes} "
                    f"(string='{string[:30]}...')"
                )

        assert not mismatches, (
            f"{dataset}/{method}: {len(mismatches)} node count mismatches:\n"
            + "\n".join(mismatches[:10])
        )


class TestEdgeCountPreservation:
    """S2G(string) must produce a graph with the correct number of edges."""

    @pytest.mark.parametrize("dataset", DATASETS)
    @pytest.mark.parametrize("method", METHODS)
    def test_edge_count(self, eval_data: EvalData, dataset: str, method: str):
        key = (dataset, method)
        if key not in eval_data.canonical:
            pytest.skip(f"No {method} strings for {dataset}")

        ds = eval_data.canonical[key]
        mismatches = []
        for gid, string in ds.strings.items():
            expected_edges = ds.edge_counts.get(gid)
            if expected_edges is None:
                continue
            sg = _s2g(string)
            actual_edges = sg.logical_edge_count()
            if actual_edges != expected_edges:
                mismatches.append(
                    f"{gid}: expected {expected_edges} edges, got {actual_edges} "
                    f"(string='{string[:30]}...')"
                )

        assert not mismatches, (
            f"{dataset}/{method}: {len(mismatches)} edge count mismatches:\n"
            + "\n".join(mismatches[:10])
        )


class TestNoSelfLoops:
    """S2G(string) must never produce self-loops."""

    @pytest.mark.parametrize("dataset", DATASETS)
    @pytest.mark.parametrize("method", METHODS)
    def test_no_self_loops(self, eval_data: EvalData, dataset: str, method: str):
        key = (dataset, method)
        if key not in eval_data.canonical:
            pytest.skip(f"No {method} strings for {dataset}")

        ds = eval_data.canonical[key]
        violations = []
        for gid, string in ds.strings.items():
            sg = _s2g(string)
            for node in range(sg.node_count()):
                if sg.has_edge(node, node):
                    violations.append(f"{gid}: self-loop on node {node}")
                    break

        assert not violations, (
            f"{dataset}/{method}: {len(violations)} graphs with self-loops:\n"
            + "\n".join(violations[:10])
        )
