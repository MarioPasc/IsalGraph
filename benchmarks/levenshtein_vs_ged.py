"""Levenshtein distance between canonical strings vs graph edit distance.

Tests the preprint's Section 2.3 "Topological Properties" claim:

    "A minimal change in the adjacency matrix corresponds to a small
     modification in the representing string, as measured by the
     Levenshtein distance between strings."

This benchmark empirically validates two aspects:

1. **Locality**: Controlled edge-edit experiments where we start from a base
   graph, perturb it by adding/removing k edges, and verify that the
   Levenshtein distance between canonical strings scales with k.

2. **Correlation with GED**: For pairs of non-isomorphic graphs, we compute
   both the Levenshtein distance between canonical strings and the exact
   graph edit distance (GED), then measure Spearman and Pearson correlation.

Since both canonical string computation (exhaustive backtracking) and exact
GED (NP-hard in general) are expensive, this benchmark uses small graphs
(N <= 8) to remain tractable.

References:
    - Lopez-Rubio (2025). arXiv:2512.10429v2, Section 2.3.
    - Zeng et al. (2009). "Comparing Stars: On Approximating Graph Edit
      Distance." PVLDB.

Usage:
    python benchmarks/levenshtein_vs_ged.py --seed 42
    python benchmarks/levenshtein_vs_ged.py --seed 42 --max-nodes 7 --output-dir /tmp
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import networkx as nx
from scipy import stats

from isalgraph.adapters.networkx_adapter import NetworkXAdapter
from isalgraph.core.canonical import canonical_string, levenshtein

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = "/media/mpascual/Sandisk2TB/research/isalgraph"
DEFAULT_SEED = 42
DEFAULT_MAX_NODES = 7  # Keep small: both canonical and GED are expensive
DEFAULT_NUM_PAIRS = 200

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DistancePairRecord:
    """One data point: two graphs and their distances."""

    pair_id: int
    experiment: str  # "edge_edit", "family_pair", "random_pair"
    family: str  # graph family description
    n1: int
    n2: int
    m1: int
    m2: int
    edit_k: int  # number of edge edits (-1 if not applicable)
    levenshtein_dist: int
    ged: float  # graph edit distance (may be float from NX)
    canonical_1: str
    canonical_2: str
    time_s: float


@dataclass
class BenchmarkSummary:
    """Aggregate results."""

    total_pairs: int = 0
    records: list[dict[str, Any]] = field(default_factory=list)
    total_time_s: float = 0.0
    errors: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Graph edit distance
# ---------------------------------------------------------------------------


def _exact_ged(g1: nx.Graph, g2: nx.Graph, timeout: float = 30.0) -> float:
    """Compute exact graph edit distance using NetworkX.

    For unlabeled graphs with same node count, this finds the optimal
    node bijection minimizing edge mismatches. All edit operations
    (node/edge insertion, deletion, substitution) have unit cost.

    Falls back to the best upper bound from optimize_graph_edit_distance
    if the exact computation is too slow.

    Args:
        g1: First graph.
        g2: Second graph.
        timeout: Maximum seconds before falling back to approximation.

    Returns:
        Graph edit distance (exact or best upper bound).
    """
    # For small graphs, exact is fast enough
    n = max(g1.number_of_nodes(), g2.number_of_nodes())
    if n <= 8:
        try:
            return float(nx.graph_edit_distance(g1, g2, timeout=timeout))
        except nx.NetworkXError:
            pass

    # Fallback: iterative upper bound (anytime algorithm)
    best = float("inf")
    for ub in nx.optimize_graph_edit_distance(g1, g2):
        best = ub
    return float(best)


# ---------------------------------------------------------------------------
# Experiment 1: Controlled edge edits
# ---------------------------------------------------------------------------


def _generate_edge_edit_pairs(
    rng: random.Random, max_nodes: int, num_base_graphs: int = 15
) -> list[tuple[str, nx.Graph, nx.Graph, int]]:
    """Generate (base_graph, perturbed_graph, k) tuples.

    For each base graph, perturb by adding/removing k=1,2,3,4 edges.

    Returns list of (family, g_base, g_perturbed, k).
    """
    pairs: list[tuple[str, nx.Graph, nx.Graph, int]] = []

    def _perturb(g: nx.Graph, k: int, rng: random.Random) -> nx.Graph | None:
        """Add/remove k edges randomly. Returns None if impossible."""
        g2 = g.copy()
        n = g2.number_of_nodes()
        nodes = list(g2.nodes())
        existing = set(g2.edges())

        for _ in range(k):
            if rng.random() < 0.5 and len(existing) > n - 1:
                # Remove a random edge (keep connected)
                edge_list = list(g2.edges())
                rng.shuffle(edge_list)
                removed = False
                for u, v in edge_list:
                    g2.remove_edge(u, v)
                    if nx.is_connected(g2):
                        existing.discard((u, v))
                        existing.discard((v, u))
                        removed = True
                        break
                    g2.add_edge(u, v)
                if not removed:
                    return None
            else:
                # Add a random edge
                non_edges = [
                    (u, v)
                    for u in nodes
                    for v in nodes
                    if u < v and (u, v) not in existing and (v, u) not in existing
                ]
                if not non_edges:
                    return None
                u, v = rng.choice(non_edges)
                g2.add_edge(u, v)
                existing.add((u, v))

        if not nx.is_connected(g2):
            return None
        return g2

    # Generate diverse base graphs
    base_graphs: list[tuple[str, nx.Graph]] = []
    for n in range(4, max_nodes + 1):
        # Tree
        seed = rng.randint(0, 2**31)
        base_graphs.append(("tree", nx.random_labeled_tree(n, seed=seed)))
        # Cycle
        base_graphs.append(("cycle", nx.cycle_graph(n)))
        # GNP
        for _attempt in range(10):
            seed = rng.randint(0, 2**31)
            g = nx.gnp_random_graph(n, 0.4, seed=seed)
            if nx.is_connected(g):
                base_graphs.append(("gnp", g))
                break

    # Limit to num_base_graphs
    if len(base_graphs) > num_base_graphs:
        base_graphs = rng.sample(base_graphs, num_base_graphs)

    # Perturb each base graph by k=1,2,3,4 edges
    for family, base_g in base_graphs:
        for k in range(1, 5):
            g_perturbed = _perturb(base_g, k, rng)
            if g_perturbed is not None and not nx.is_isomorphic(base_g, g_perturbed):
                pairs.append((family, base_g, g_perturbed, k))

    return pairs


# ---------------------------------------------------------------------------
# Experiment 2: Family pairs (different graph structures, same N)
# ---------------------------------------------------------------------------


def _generate_family_pairs(max_nodes: int) -> list[tuple[str, nx.Graph, nx.Graph, int]]:
    """Generate non-isomorphic graph pairs from different families.

    Returns list of (description, g1, g2, -1).
    """
    pairs: list[tuple[str, nx.Graph, nx.Graph, int]] = []

    for n in range(4, max_nodes + 1):
        # Path vs Cycle
        pairs.append(("path_vs_cycle", nx.path_graph(n), nx.cycle_graph(n), -1))
        # Star vs Path
        pairs.append(("star_vs_path", nx.star_graph(n - 1), nx.path_graph(n), -1))
        # Complete vs Cycle
        if n >= 4:
            pairs.append(("complete_vs_cycle", nx.complete_graph(n), nx.cycle_graph(n), -1))

    return pairs


# ---------------------------------------------------------------------------
# Experiment 3: Random GNP pairs
# ---------------------------------------------------------------------------


def _generate_random_pairs(
    rng: random.Random, max_nodes: int, count: int
) -> list[tuple[str, nx.Graph, nx.Graph, int]]:
    """Generate random pairs of non-isomorphic connected graphs with same N.

    Returns list of (description, g1, g2, -1).
    """
    pairs: list[tuple[str, nx.Graph, nx.Graph, int]] = []
    attempts = 0

    while len(pairs) < count and attempts < count * 10:
        attempts += 1
        n = rng.randint(4, max_nodes)
        p1 = rng.uniform(0.2, 0.6)
        p2 = rng.uniform(0.2, 0.6)
        seed1 = rng.randint(0, 2**31)
        seed2 = rng.randint(0, 2**31)
        g1 = nx.gnp_random_graph(n, p1, seed=seed1)
        g2 = nx.gnp_random_graph(n, p2, seed=seed2)
        if nx.is_connected(g1) and nx.is_connected(g2) and not nx.is_isomorphic(g1, g2):
            pairs.append(("random_gnp", g1, g2, -1))

    return pairs


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


def _test_pair(
    pair_id: int,
    experiment: str,
    family: str,
    g1: nx.Graph,
    g2: nx.Graph,
    edit_k: int,
) -> DistancePairRecord:
    """Compute Levenshtein and GED for a graph pair."""
    adapter = NetworkXAdapter()

    t0 = time.perf_counter()

    sg1 = adapter.from_external(g1, directed=False)
    sg2 = adapter.from_external(g2, directed=False)

    c1 = canonical_string(sg1)
    c2 = canonical_string(sg2)

    lev = levenshtein(c1, c2)
    ged = _exact_ged(g1, g2)

    elapsed = time.perf_counter() - t0

    return DistancePairRecord(
        pair_id=pair_id,
        experiment=experiment,
        family=family,
        n1=g1.number_of_nodes(),
        n2=g2.number_of_nodes(),
        m1=g1.number_of_edges(),
        m2=g2.number_of_edges(),
        edit_k=edit_k,
        levenshtein_dist=lev,
        ged=ged,
        canonical_1=c1,
        canonical_2=c2,
        time_s=round(elapsed, 4),
    )


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark(
    seed: int,
    output_dir: str,
    max_nodes: int = DEFAULT_MAX_NODES,
    num_random_pairs: int = 50,
) -> BenchmarkSummary:
    """Run the full Levenshtein vs GED benchmark."""
    rng = random.Random(seed)
    summary = BenchmarkSummary()
    pair_id = 0

    print("Levenshtein vs Graph Edit Distance Benchmark")
    print(f"{'=' * 60}")
    print(f"Seed: {seed}")
    print(f"Max nodes: {max_nodes}")
    print(f"Output dir: {output_dir}")
    print()

    # ---- Experiment 1: Controlled edge edits ----
    print("[1] Controlled edge-edit experiment...")
    edge_edit_pairs = _generate_edge_edit_pairs(rng, max_nodes)
    print(f"    Generated {len(edge_edit_pairs)} pairs.")

    for i, (family, g1, g2, k) in enumerate(edge_edit_pairs):
        try:
            rec = _test_pair(pair_id, "edge_edit", family, g1, g2, k)
            summary.records.append(asdict(rec))
            pair_id += 1
        except Exception as exc:
            summary.errors.append(
                {"pair_id": pair_id, "experiment": "edge_edit", "error": str(exc)}
            )
            pair_id += 1

        if (i + 1) % max(1, len(edge_edit_pairs) // 5) == 0:
            print(f"    ... {i + 1}/{len(edge_edit_pairs)} done")

    # ---- Experiment 2: Family pairs ----
    print("\n[2] Graph family pairs (structural comparison)...")
    family_pairs = _generate_family_pairs(max_nodes)
    print(f"    Generated {len(family_pairs)} pairs.")

    for i, (family, g1, g2, k) in enumerate(family_pairs):
        try:
            rec = _test_pair(pair_id, "family_pair", family, g1, g2, k)
            summary.records.append(asdict(rec))
            pair_id += 1
        except Exception as exc:
            summary.errors.append(
                {"pair_id": pair_id, "experiment": "family_pair", "error": str(exc)}
            )
            pair_id += 1

        if (i + 1) % max(1, len(family_pairs) // 5) == 0:
            print(f"    ... {i + 1}/{len(family_pairs)} done")

    # ---- Experiment 3: Random GNP pairs ----
    print(f"\n[3] Random GNP pairs ({num_random_pairs} target)...")
    random_pairs = _generate_random_pairs(rng, max_nodes, num_random_pairs)
    print(f"    Generated {len(random_pairs)} pairs.")

    for i, (family, g1, g2, k) in enumerate(random_pairs):
        try:
            rec = _test_pair(pair_id, "random_pair", family, g1, g2, k)
            summary.records.append(asdict(rec))
            pair_id += 1
        except Exception as exc:
            summary.errors.append(
                {"pair_id": pair_id, "experiment": "random_pair", "error": str(exc)}
            )
            pair_id += 1

        if (i + 1) % max(1, len(random_pairs) // 5) == 0:
            print(f"    ... {i + 1}/{len(random_pairs)} done")

    summary.total_pairs = len(summary.records)
    summary.total_time_s = sum(r["time_s"] for r in summary.records)

    # ---- Correlation analysis ----
    print(f"\n{'=' * 60}")
    print("CORRELATION ANALYSIS")
    print(f"{'=' * 60}")

    lev_vals = [r["levenshtein_dist"] for r in summary.records]
    ged_vals = [r["ged"] for r in summary.records]

    if len(lev_vals) >= 3:
        pearson_r, pearson_p = stats.pearsonr(lev_vals, ged_vals)
        spearman_r, spearman_p = stats.spearmanr(lev_vals, ged_vals)
        print(f"All pairs ({len(lev_vals)}):")
        print(f"  Pearson:  r={pearson_r:.4f}  p={pearson_p:.2e}")
        print(f"  Spearman: r={spearman_r:.4f}  p={spearman_p:.2e}")
    else:
        pearson_r = spearman_r = pearson_p = spearman_p = float("nan")
        print("Too few data points for correlation.")

    # Per-experiment correlation
    correlations: dict[str, dict[str, float]] = {}
    for exp_name in ["edge_edit", "family_pair", "random_pair"]:
        exp_recs = [r for r in summary.records if r["experiment"] == exp_name]
        if len(exp_recs) >= 3:
            l_vals = [r["levenshtein_dist"] for r in exp_recs]
            g_vals = [r["ged"] for r in exp_recs]
            pr, pp = stats.pearsonr(l_vals, g_vals)
            sr, sp = stats.spearmanr(l_vals, g_vals)
            correlations[exp_name] = {
                "pearson_r": round(pr, 4),
                "pearson_p": round(pp, 6),
                "spearman_r": round(sr, 4),
                "spearman_p": round(sp, 6),
                "n_pairs": len(exp_recs),
            }
            print(f"\n{exp_name} ({len(exp_recs)} pairs):")
            print(f"  Pearson:  r={pr:.4f}  p={pp:.2e}")
            print(f"  Spearman: r={sr:.4f}  p={sp:.2e}")

    # ---- Edge-edit locality analysis ----
    edge_edit_recs = [r for r in summary.records if r["experiment"] == "edge_edit"]
    if edge_edit_recs:
        print(f"\n{'=' * 60}")
        print("LOCALITY: Levenshtein vs edge-edit count (k)")
        print(f"{'=' * 60}")
        for k in sorted({r["edit_k"] for r in edge_edit_recs}):
            k_recs = [r for r in edge_edit_recs if r["edit_k"] == k]
            levs = [r["levenshtein_dist"] for r in k_recs]
            geds = [r["ged"] for r in k_recs]
            print(
                f"  k={k}: n={len(k_recs):>3}  "
                f"lev_mean={sum(levs) / len(levs):>5.1f}  "
                f"ged_mean={sum(geds) / len(geds):>5.1f}  "
                f"lev_max={max(levs):>3}  ged_max={max(geds):>5.1f}"
            )

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print(
        f"Total pairs: {summary.total_pairs}, Time: {summary.total_time_s:.2f}s, "
        f"Errors: {len(summary.errors)}"
    )

    # ---- Save JSON ----
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "levenshtein_vs_ged.json")
    with open(output_path, "w") as f:
        json.dump(
            {
                "benchmark": "levenshtein_vs_ged",
                "config": {
                    "seed": seed,
                    "max_nodes": max_nodes,
                    "num_random_pairs": num_random_pairs,
                },
                "summary": {
                    "total_pairs": summary.total_pairs,
                    "total_time_s": round(summary.total_time_s, 4),
                    "errors": len(summary.errors),
                },
                "correlations": {
                    "all": {
                        "pearson_r": round(pearson_r, 4) if not math.isnan(pearson_r) else None,
                        "pearson_p": round(pearson_p, 6) if not math.isnan(pearson_p) else None,
                        "spearman_r": round(spearman_r, 4) if not math.isnan(spearman_r) else None,
                        "spearman_p": round(spearman_p, 6) if not math.isnan(spearman_p) else None,
                        "n_pairs": len(lev_vals),
                    },
                    **correlations,
                },
                "records": [
                    {k: v for k, v in r.items() if k not in ("canonical_1", "canonical_2")}
                    for r in summary.records
                ],
                "errors": summary.errors,
            },
            f,
            indent=2,
        )
    print(f"Results saved to: {output_path}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Levenshtein distance vs graph edit distance correlation."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_NODES)
    parser.add_argument("--num-random-pairs", type=int, default=50)
    args = parser.parse_args()

    summary = run_benchmark(
        seed=args.seed,
        output_dir=args.output_dir,
        max_nodes=args.max_nodes,
        num_random_pairs=args.num_random_pairs,
    )
    if summary.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
