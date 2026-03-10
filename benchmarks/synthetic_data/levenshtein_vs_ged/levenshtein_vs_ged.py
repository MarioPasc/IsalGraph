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
) -> tuple[BenchmarkSummary, list[dict[str, Any]]]:
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

    return summary, summary.records


# ---------------------------------------------------------------------------
# Parallel worker wrapper
# ---------------------------------------------------------------------------


def _parallel_test_pair(
    args_tuple: tuple[int, str, str, nx.Graph, nx.Graph, int],
) -> DistancePairRecord:
    """Worker for ProcessPoolExecutor: test a single graph pair."""
    pair_id, experiment, family, g1, g2, edit_k = args_tuple
    return _test_pair(pair_id, experiment, family, g1, g2, edit_k)


# ---------------------------------------------------------------------------
# Parallel benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark_parallel(
    seed: int,
    output_dir: str,
    max_nodes: int = DEFAULT_MAX_NODES,
    num_random_pairs: int = 50,
    n_workers: int = 4,
) -> tuple[BenchmarkSummary, list[dict[str, Any]]]:
    """Run the full Levenshtein vs GED benchmark with parallel pair evaluation.

    GED computation is the bottleneck; this distributes ``_test_pair`` calls
    across ``n_workers`` processes using ``ProcessPoolExecutor``.

    Args:
        seed: Random seed for reproducibility.
        output_dir: Directory to write JSON results.
        max_nodes: Maximum node count for generated graphs.
        num_random_pairs: Target number of random GNP pairs.
        n_workers: Number of parallel worker processes.

    Returns:
        Tuple of (summary, records) where records is ``summary.records``.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    rng = random.Random(seed)
    summary = BenchmarkSummary()

    print("Levenshtein vs Graph Edit Distance Benchmark (parallel)")
    print(f"{'=' * 60}")
    print(f"Seed: {seed}, Workers: {n_workers}")
    print(f"Max nodes: {max_nodes}")
    print(f"Output dir: {output_dir}")
    print()

    # ---- Generate all pairs upfront ----
    print("[1] Generating edge-edit pairs...")
    edge_edit_pairs = _generate_edge_edit_pairs(rng, max_nodes)
    print(f"    {len(edge_edit_pairs)} edge-edit pairs.")

    print("[2] Generating family pairs...")
    family_pairs = _generate_family_pairs(max_nodes)
    print(f"    {len(family_pairs)} family pairs.")

    print(f"[3] Generating random GNP pairs (target {num_random_pairs})...")
    random_pairs = _generate_random_pairs(rng, max_nodes, num_random_pairs)
    print(f"    {len(random_pairs)} random pairs.")

    # ---- Build task list ----
    tasks: list[tuple[int, str, str, nx.Graph, nx.Graph, int]] = []
    pair_id = 0
    for family, g1, g2, k in edge_edit_pairs:
        tasks.append((pair_id, "edge_edit", family, g1, g2, k))
        pair_id += 1
    for family, g1, g2, k in family_pairs:
        tasks.append((pair_id, "family_pair", family, g1, g2, k))
        pair_id += 1
    for family, g1, g2, k in random_pairs:
        tasks.append((pair_id, "random_pair", family, g1, g2, k))
        pair_id += 1

    # ---- Execute in parallel ----
    print(f"\nSubmitting {len(tasks)} pair evaluations to {n_workers} workers...")
    results: list[DistancePairRecord] = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_parallel_test_pair, t): t[0] for t in tasks}
        total = len(futures)
        for done_count, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
                results.append(rec)
                summary.records.append(asdict(rec))
            except Exception as exc:
                tid = futures[fut]
                summary.errors.append({"pair_id": tid, "experiment": "unknown", "error": str(exc)})
            if done_count % max(1, total // 20) == 0:
                print(f"  ... {done_count}/{total} done")

    summary.total_pairs = len(summary.records)
    summary.total_time_s = sum(r["time_s"] for r in summary.records)

    # ---- Correlation analysis (same as sequential) ----
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
                    "n_workers": n_workers,
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

    return summary, summary.records


# ---------------------------------------------------------------------------
# CSV / Figure / Table generation
# ---------------------------------------------------------------------------


def save_csv(records: list[dict[str, Any]], output_dir: str) -> str:
    """Save benchmark records as CSV (excluding canonical strings).

    Args:
        records: List of record dicts from BenchmarkSummary.records.
        output_dir: Directory to write the CSV file.

    Returns:
        Path to the saved CSV file.
    """
    import pandas as pd

    df = pd.DataFrame(records)
    # Exclude canonical strings (too long for CSV)
    drop_cols = [c for c in ("canonical_1", "canonical_2") if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "levenshtein_vs_ged.csv")
    df.to_csv(path, index=False)
    print(f"CSV saved to: {path}")
    return path


def generate_figure(records: list[dict[str, Any]], output_dir: str) -> list[str]:
    """Generate publication figure: 1x3 (scatter, locality, strip+box).

    Panel (a): Scatter of Levenshtein distance vs GED with jitter to reduce
               overplotting. OLS regression line + Pearson r annotation.
    Panel (b): Locality — mean Levenshtein vs edge-edit count k. Strip plot
               of individual points with mean ± 1 SE error bars.
    Panel (c): Strip + box overlay for Lev/GED ratio by experiment type.

    Args:
        records: List of record dicts from BenchmarkSummary.records.
        output_dir: Directory to write figure files.

    Returns:
        List of saved file paths (pdf + png).
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import matplotlib.pyplot as plt  # noqa: E402
    import numpy as np  # noqa: E402
    from plotting_styles import (  # noqa: E402
        EXPERIMENT_DISPLAY_NAMES,
        PAUL_TOL_BRIGHT,
        PLOT_SETTINGS,
        apply_ieee_style,
        get_figure_size,
        save_figure,
    )

    apply_ieee_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=get_figure_size("double", 0.5))

    rng = np.random.default_rng(42)

    # ---- Color map by experiment type ----
    exp_colors: dict[str, str] = {
        "edge_edit": PAUL_TOL_BRIGHT["blue"],
        "family_pair": PAUL_TOL_BRIGHT["red"],
        "random_pair": PAUL_TOL_BRIGHT["green"],
    }

    # ---- Panel (a): Scatter — Levenshtein vs GED with jitter ----
    all_lev = np.array([r["levenshtein_dist"] for r in records], dtype=float)
    all_ged = np.array([r["ged"] for r in records], dtype=float)

    for exp_name in ["edge_edit", "family_pair", "random_pair"]:
        mask = np.array([r["experiment"] == exp_name for r in records])
        if not mask.any():
            continue
        n_pts = int(mask.sum())
        jitter_x = rng.uniform(-0.15, 0.15, size=n_pts)
        jitter_y = rng.uniform(-0.15, 0.15, size=n_pts)
        ax1.scatter(
            all_ged[mask] + jitter_x,
            all_lev[mask] + jitter_y,
            c=exp_colors[exp_name],
            label=EXPERIMENT_DISPLAY_NAMES.get(exp_name, exp_name),
            alpha=0.5,
            s=20,
            edgecolors="none",
            zorder=2,
        )

    # OLS regression line + Pearson annotation (on un-jittered data)
    if len(all_ged) >= 3:
        from scipy import stats as sp_stats  # noqa: E402

        slope, intercept, r_value, p_value, _ = sp_stats.linregress(all_ged, all_lev)
        x_fit = np.linspace(all_ged.min(), all_ged.max(), 100)
        ax1.plot(
            x_fit,
            slope * x_fit + intercept,
            color="0.3",
            linewidth=PLOT_SETTINGS["line_width"],
            linestyle="--",
            zorder=3,
        )
        ax1.text(
            0.05,
            0.95,
            f"$r = {r_value:.3f}$\n$p = {p_value:.1e}$",
            transform=ax1.transAxes,
            fontsize=PLOT_SETTINGS["annotation_fontsize"],
            verticalalignment="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )

    ax1.set_xlabel("Graph Edit Distance")
    ax1.set_ylabel("Levenshtein Distance")
    ax1.legend(fontsize=PLOT_SETTINGS["legend_fontsize"], loc="lower right")
    ax1.grid(
        axis="y",
        alpha=PLOT_SETTINGS["grid_alpha"],
        linestyle=PLOT_SETTINGS["grid_linestyle"],
        linewidth=PLOT_SETTINGS["grid_linewidth"],
    )
    ax1.text(-0.12, 1.08, "(a)", transform=ax1.transAxes, fontsize=12, fontweight="bold")

    # ---- Panel (b): Locality — mean Lev vs k with strip + error bars ----
    edge_edit_recs = [r for r in records if r["experiment"] == "edge_edit"]
    k_values = sorted({r["edit_k"] for r in edge_edit_recs})

    if k_values:
        k_means = []
        k_se = []
        for k in k_values:
            levs = np.array(
                [float(r["levenshtein_dist"]) for r in edge_edit_recs if r["edit_k"] == k]
            )
            k_means.append(float(np.mean(levs)))
            if len(levs) >= 2:
                k_se.append(float(np.std(levs, ddof=1) / np.sqrt(len(levs))))
            else:
                k_se.append(0.0)

            # Strip plot: individual points with horizontal jitter
            strip_jitter = rng.uniform(-0.12, 0.12, size=len(levs))
            ax2.scatter(
                np.full(len(levs), k) + strip_jitter,
                levs,
                c=exp_colors["edge_edit"],
                alpha=0.3,
                s=12,
                edgecolors="none",
                zorder=1,
            )

        k_arr = np.array(k_values, dtype=float)
        k_means_arr = np.array(k_means)
        k_se_arr = np.array(k_se)

        # Mean line (thick)
        ax2.plot(
            k_arr,
            k_means_arr,
            marker="o",
            color=exp_colors["edge_edit"],
            linewidth=PLOT_SETTINGS["line_width_thick"],
            markersize=PLOT_SETTINGS["marker_size"],
            zorder=3,
        )
        # Error bars: mean ± 1 SE
        ax2.errorbar(
            k_arr,
            k_means_arr,
            yerr=k_se_arr,
            fmt="none",
            ecolor=exp_colors["edge_edit"],
            elinewidth=PLOT_SETTINGS["errorbar_linewidth"],
            capsize=PLOT_SETTINGS["errorbar_capsize"],
            capthick=PLOT_SETTINGS["errorbar_capthick"],
            zorder=2,
        )
        ax2.set_xticks(k_values)

        # Jonckheere-Terpstra trend test annotation
        try:
            from scipy.stats import jonckheere_terpstra  # noqa: E402

            groups = [
                [float(r["levenshtein_dist"]) for r in edge_edit_recs if r["edit_k"] == k]
                for k in k_values
            ]
            jt_result = jonckheere_terpstra(groups, alternative="increasing")
            ax2.text(
                0.95,
                0.05,
                f"JT $p = {jt_result.pvalue:.1e}$",
                transform=ax2.transAxes,
                fontsize=PLOT_SETTINGS["annotation_fontsize"],
                ha="right",
                va="bottom",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
            )
        except (ImportError, AttributeError):
            pass

    ax2.set_xlabel("Edge edits ($k$)")
    ax2.set_ylabel("Levenshtein Distance")
    ax2.grid(
        axis="y",
        alpha=PLOT_SETTINGS["grid_alpha"],
        linestyle=PLOT_SETTINGS["grid_linestyle"],
        linewidth=PLOT_SETTINGS["grid_linewidth"],
    )
    ax2.text(-0.12, 1.08, "(b)", transform=ax2.transAxes, fontsize=12, fontweight="bold")

    # ---- Panel (c): Strip + box overlay — Lev/GED ratio by experiment ----
    ratio_data: dict[str, list[float]] = {}
    for r in records:
        if r["ged"] > 0:
            ratio = r["levenshtein_dist"] / r["ged"]
            exp = r["experiment"]
            ratio_data.setdefault(exp, []).append(ratio)

    exp_order = ["edge_edit", "family_pair", "random_pair"]
    box_data = [ratio_data.get(e, []) for e in exp_order]
    box_labels = [EXPERIMENT_DISPLAY_NAMES.get(e, e) for e in exp_order]
    box_colors = [exp_colors.get(e, "#888888") for e in exp_order]

    # Box plot with transparent fill (no outlier markers — strip shows them)
    bp = ax3.boxplot(
        box_data,
        labels=box_labels,
        patch_artist=True,
        widths=PLOT_SETTINGS["boxplot_width"],
        showfliers=False,
        zorder=2,
    )
    for patch, color in zip(bp["boxes"], box_colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)
    for element in ("whiskers", "caps", "medians"):
        for line in bp[element]:
            line.set_color("0.3")
            line.set_linewidth(PLOT_SETTINGS["boxplot_linewidth"])

    # Strip overlay: jittered individual points
    for i, (data_pts, color) in enumerate(zip(box_data, box_colors, strict=False)):
        if not data_pts:
            continue
        pts = np.array(data_pts)
        strip_jitter = rng.uniform(-0.15, 0.15, size=len(pts))
        ax3.scatter(
            np.full(len(pts), i + 1) + strip_jitter,
            pts,
            c=color,
            alpha=0.5,
            s=15,
            edgecolors="none",
            zorder=3,
        )

    ax3.set_ylabel("Levenshtein / GED")
    ax3.grid(
        axis="y",
        alpha=PLOT_SETTINGS["grid_alpha"],
        linestyle=PLOT_SETTINGS["grid_linestyle"],
        linewidth=PLOT_SETTINGS["grid_linewidth"],
    )
    ax3.tick_params(axis="x", rotation=30)
    for label in ax3.get_xticklabels():
        label.set_fontsize(PLOT_SETTINGS["tick_labelsize"] - 1)
        label.set_ha("right")
    ax3.text(-0.12, 1.08, "(c)", transform=ax3.transAxes, fontsize=12, fontweight="bold")

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    paths = save_figure(fig, os.path.join(output_dir, "levenshtein_vs_ged_figure"))
    plt.close(fig)
    print(f"Figure saved: {paths}")
    return paths


def generate_table(records: list[dict[str, Any]], output_dir: str) -> str:
    """Generate LaTeX table with correlation statistics per experiment type.

    Columns: Experiment, N_pairs, Pearson_r, Pearson_95CI, Spearman_r,
             Mean_Lev, Mean_GED, Mean_time_s.

    Args:
        records: List of record dicts from BenchmarkSummary.records.
        output_dir: Directory to write the .tex file.

    Returns:
        Path to the saved LaTeX file.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import pandas as pd  # noqa: E402
    from plotting_styles import bootstrap_ci, save_latex_table  # noqa: E402

    rows = []
    for exp_name in ["edge_edit", "family_pair", "random_pair", "all"]:
        if exp_name == "all":
            exp_recs = records
        else:
            exp_recs = [r for r in records if r["experiment"] == exp_name]

        if len(exp_recs) < 3:
            continue

        l_vals = [float(r["levenshtein_dist"]) for r in exp_recs]
        g_vals = [float(r["ged"]) for r in exp_recs]
        times = [r["time_s"] for r in exp_recs]

        # Bootstrap CIs for Pearson correlation
        _, ci_lo, ci_hi = bootstrap_ci(l_vals, g_vals, stat_func="pearson")
        pearson_r_val = stats.pearsonr(l_vals, g_vals)[0]
        spearman_r_val = stats.spearmanr(l_vals, g_vals)[0]

        rows.append(
            {
                "Experiment": exp_name.replace("_", " ").title() if exp_name != "all" else "All",
                "N_pairs": len(exp_recs),
                "Pearson_r": f"{pearson_r_val:.3f}",
                "Pearson_95CI": f"[{ci_lo:.3f}, {ci_hi:.3f}]",
                "Spearman_r": f"{spearman_r_val:.3f}",
                "Mean_Lev": f"{sum(l_vals) / len(l_vals):.2f}",
                "Mean_GED": f"{sum(g_vals) / len(g_vals):.2f}",
                "Mean_time_s": f"{sum(times) / len(times):.3f}",
            }
        )

    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "levenshtein_vs_ged_table.tex")
    save_latex_table(
        df,
        path,
        caption=(
            "Correlation between Levenshtein distance of canonical strings "
            "and graph edit distance, by experiment type."
        ),
        label="tab:lev_vs_ged",
    )
    print(f"Table saved to: {path}")
    return path


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
    parser.add_argument(
        "--mode",
        choices=["local", "picasso"],
        default="local",
        help="Execution mode (default: local).",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = sequential).",
    )
    parser.add_argument("--csv", action="store_true", help="Save results as CSV.")
    parser.add_argument("--plot", action="store_true", help="Generate publication figure.")
    parser.add_argument("--table", action="store_true", help="Generate LaTeX table.")
    args = parser.parse_args()

    # In picasso mode, default to all outputs
    if args.mode == "picasso":
        args.csv = True
        args.plot = True
        args.table = True

    if args.n_workers > 1:
        summary, all_records = run_benchmark_parallel(
            seed=args.seed,
            output_dir=args.output_dir,
            max_nodes=args.max_nodes,
            num_random_pairs=args.num_random_pairs,
            n_workers=args.n_workers,
        )
    else:
        summary, all_records = run_benchmark(
            seed=args.seed,
            output_dir=args.output_dir,
            max_nodes=args.max_nodes,
            num_random_pairs=args.num_random_pairs,
        )

    if args.csv:
        save_csv(all_records, args.output_dir)
    if args.plot:
        generate_figure(all_records, args.output_dir)
    if args.table:
        generate_table(all_records, args.output_dir)

    if summary.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
