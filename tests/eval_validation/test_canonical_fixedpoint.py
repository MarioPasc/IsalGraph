"""Test the canonical fixed-point property on real eval data.

The canonical string is a complete graph invariant. Therefore:

    canonical_string(S2G(w*_G)) == w*_G

This is the strongest possible validation: if this fails, either the
canonical algorithm or the string-to-graph reconstruction is broken.

For greedy strings (NOT invariant), we test a weaker property:
S2G(w'_G) produces a valid graph and G2S from the best starting node
produces a string of the same length.
"""

from __future__ import annotations

import pytest

from isalgraph.core.canonical import canonical_string
from isalgraph.core.graph_to_string import GraphToString
from isalgraph.core.string_to_graph import StringToGraph
from tests.eval_validation.conftest import DATASETS, EvalData

pytestmark = pytest.mark.eval_validation

# Sample size per dataset for exhaustive fixed-point (canonical_string is expensive)
FIXEDPOINT_SAMPLE_SIZE = 100


def _s2g(string: str):
    """Convert IsalGraph string to SparseGraph."""
    converter = StringToGraph(string, directed_graph=False)
    sg, _ = converter.run()
    return sg


class TestExhaustiveFixedPoint:
    """canonical_string(S2G(w*)) must equal w* for all graphs."""

    @pytest.mark.parametrize("dataset", DATASETS)
    def test_fixedpoint(self, eval_data: EvalData, dataset: str):
        key = (dataset, "exhaustive")
        if key not in eval_data.canonical:
            pytest.skip(f"No exhaustive strings for {dataset}")

        ds = eval_data.canonical[key]
        ids = ds.graph_ids[:FIXEDPOINT_SAMPLE_SIZE]
        n_tested = 0
        failures = []

        for gid in ids:
            w_star = ds.strings[gid]
            sg = _s2g(w_star)
            w_recomputed = canonical_string(sg)

            if w_recomputed != w_star:
                failures.append(
                    f"{gid}: expected '{w_star}' (len={len(w_star)}), "
                    f"got '{w_recomputed}' (len={len(w_recomputed)})"
                )
            n_tested += 1

        assert not failures, (
            f"{dataset}: {len(failures)}/{n_tested} fixed-point violations:\n"
            + "\n".join(failures[:10])
        )


class TestGreedyReconstructionValid:
    """S2G(w'_G) must produce a graph that G2S can encode from every starting node.

    NOTE: The greedy-min length is NOT an isomorphism invariant. For isomorphic
    graphs with different node labelings, _find_new_neighbor iterates set[int]
    whose order depends on integer hash values, causing different tie-breaking
    choices and potentially different string lengths. Therefore we do NOT test
    that recomputed greedy length == stored greedy length. We only verify that
    G2S succeeds for all starting nodes and produces valid strings.
    """

    @pytest.mark.parametrize("dataset", DATASETS)
    def test_greedy_reconstruction_valid(self, eval_data: EvalData, dataset: str):
        key = (dataset, "greedy")
        if key not in eval_data.canonical:
            pytest.skip(f"No greedy strings for {dataset}")

        ds = eval_data.canonical[key]
        ids = ds.graph_ids[:FIXEDPOINT_SAMPLE_SIZE]
        failures = []

        for gid in ids:
            w_greedy = ds.strings[gid]
            sg = _s2g(w_greedy)
            n = sg.node_count()

            for v in range(n):
                try:
                    gts = GraphToString(sg)
                    s, _ = gts.run(initial_node=v)
                    if len(s) < n - 1:
                        failures.append(f"{gid} start={v}: len={len(s)} < n-1={n - 1}")
                        break
                except (ValueError, RuntimeError) as e:
                    failures.append(f"{gid} start={v}: {e}")
                    break

        assert not failures, (
            f"{dataset}: {len(failures)} greedy reconstruction failures:\n"
            + "\n".join(failures[:10])
        )
