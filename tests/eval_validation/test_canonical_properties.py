"""Validate canonical string algebraic properties on real eval data.

Tests:
1. Exhaustive length <= greedy length (minimality).
2. GED=0 pairs have identical exhaustive strings (completeness).
3. Identical exhaustive strings imply identical graph structure (soundness).
"""

from __future__ import annotations

import numpy as np
import pytest

from isalgraph.core.string_to_graph import StringToGraph
from tests.eval_validation.conftest import DATASETS, EvalData

pytestmark = pytest.mark.eval_validation


def _s2g(string: str):
    """Convert IsalGraph string to SparseGraph."""
    converter = StringToGraph(string, directed_graph=False)
    sg, _ = converter.run()
    return sg


class TestExhaustiveMinimality:
    """Exhaustive canonical string must be <= greedy-min string in length."""

    @pytest.mark.parametrize("dataset", DATASETS)
    def test_exhaustive_leq_greedy(self, eval_data: EvalData, dataset: str):
        exh_key = (dataset, "exhaustive")
        gre_key = (dataset, "greedy")
        if exh_key not in eval_data.canonical or gre_key not in eval_data.canonical:
            pytest.skip(f"Missing exhaustive or greedy strings for {dataset}")

        exh = eval_data.canonical[exh_key]
        gre = eval_data.canonical[gre_key]
        violations = []

        for gid in exh.graph_ids:
            if gid not in gre.strings:
                continue
            e_len = exh.lengths[gid]
            g_len = gre.lengths[gid]
            if e_len > g_len:
                violations.append(f"{gid}: exhaustive({e_len}) > greedy({g_len})")

        assert not violations, (
            f"{dataset}: {len(violations)} graphs where exhaustive > greedy:\n"
            + "\n".join(violations[:10])
        )


class TestExhaustiveCompleteness:
    """If GED(G, H) = 0 then w*_G must equal w*_H (complete invariant)."""

    @pytest.mark.parametrize("dataset", DATASETS)
    def test_ged_zero_implies_same_string(self, eval_data: EvalData, dataset: str):
        if dataset not in eval_data.ged_matrices:
            pytest.skip(f"No GED matrix for {dataset}")
        exh_key = (dataset, "exhaustive")
        if exh_key not in eval_data.canonical:
            pytest.skip(f"No exhaustive strings for {dataset}")

        ged = eval_data.ged_matrices[dataset]
        ged_ids = eval_data.ged_graph_ids[dataset]
        exh = eval_data.canonical[exh_key]

        n = len(ged_ids)
        violations = []
        n_zero_pairs = 0

        for i in range(n):
            for j in range(i + 1, n):
                if not np.isfinite(ged[i, j]) or ged[i, j] != 0.0:
                    continue
                n_zero_pairs += 1
                gid_i = ged_ids[i]
                gid_j = ged_ids[j]
                if gid_i not in exh.strings or gid_j not in exh.strings:
                    continue
                w_i = exh.strings[gid_i]
                w_j = exh.strings[gid_j]
                if w_i != w_j:
                    violations.append(
                        f"({gid_i}, {gid_j}): GED=0 but w*='{w_i[:20]}...' vs '{w_j[:20]}...'"
                    )

        assert not violations, (
            f"{dataset}: {len(violations)}/{n_zero_pairs} GED=0 pairs "
            f"with different exhaustive strings:\n" + "\n".join(violations[:10])
        )


class TestExhaustiveSoundness:
    """Identical exhaustive strings must encode isomorphic graphs.

    We verify this by checking that S2G(w*) produces the same node
    and edge counts for graphs sharing the same canonical string.
    (Full isomorphism is guaranteed by the canonical string definition,
    but node/edge count is a fast sanity check.)
    """

    @pytest.mark.parametrize("dataset", DATASETS)
    def test_same_string_same_structure(self, eval_data: EvalData, dataset: str):
        exh_key = (dataset, "exhaustive")
        if exh_key not in eval_data.canonical:
            pytest.skip(f"No exhaustive strings for {dataset}")

        exh = eval_data.canonical[exh_key]

        # Group by canonical string
        string_groups: dict[str, list[str]] = {}
        for gid, string in exh.strings.items():
            string_groups.setdefault(string, []).append(gid)

        violations = []
        for string, gids in string_groups.items():
            if len(gids) < 2:
                continue
            # All graphs with same canonical string must have same node/edge counts
            nc = {gid: exh.node_counts.get(gid) for gid in gids}
            ec = {gid: exh.edge_counts.get(gid) for gid in gids}
            unique_nc = set(v for v in nc.values() if v is not None)
            unique_ec = set(v for v in ec.values() if v is not None)
            if len(unique_nc) > 1 or len(unique_ec) > 1:
                violations.append(
                    f"w*='{string[:20]}...': {len(gids)} graphs, "
                    f"node_counts={unique_nc}, edge_counts={unique_ec}"
                )

        assert not violations, (
            f"{dataset}: {len(violations)} canonical strings map to graphs with "
            f"different structures:\n" + "\n".join(violations[:10])
        )
