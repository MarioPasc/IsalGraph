"""Phase 3 -- canonical string invariance testing at scale.

Verifies two properties of the canonical string:

1. **Invariance**: For isomorphic graph pairs (created by random relabeling),
   ``canonical_string(G) == canonical_string(G')``.

2. **Discrimination**: For non-isomorphic graph pairs,
   ``canonical_string(G) != canonical_string(G')``.

Tests diverse graph families: trees, cycles, complete, Petersen, random GNP,
Barabasi-Albert, star, wheel, ladder, Watts-Strogatz.

Usage:
    python benchmarks/canonical_invariance.py --num-tests 500 --seed 42
    python benchmarks/canonical_invariance.py --num-tests 50 --seed 42 --output-dir /tmp
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from isalgraph.adapters.networkx_adapter import NetworkXAdapter
from isalgraph.core.canonical import canonical_string

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = "/media/mpascual/Sandisk2TB/research/isalgraph"
DEFAULT_NUM_TESTS = 500
DEFAULT_MAX_NODES = 10  # Canonical is expensive; keep small
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    """Result of a single canonical invariance test."""

    test_id: int
    test_type: str  # "invariance" or "discrimination"
    source: str  # "tree", "gnp", "cycle", etc.
    num_nodes: int
    num_edges: int
    canonical_1: str
    canonical_2: str
    passed: bool
    error: str = ""
    time_s: float = 0.0


@dataclass
class BenchmarkSummary:
    """Aggregate results from the full benchmark run."""

    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    total_time_s: float = 0.0
    invariance_tests: int = 0
    invariance_passed: int = 0
    discrimination_tests: int = 0
    discrimination_passed: int = 0
    results_by_source: dict[str, dict[str, int]] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Graph generation utilities
# ---------------------------------------------------------------------------


def _random_relabeling(g: nx.Graph | nx.DiGraph, rng: random.Random) -> nx.Graph | nx.DiGraph:
    """Create an isomorphic copy of g via random node relabeling.

    Args:
        g: Input NetworkX graph.
        rng: Seeded Random instance.

    Returns:
        Isomorphic copy with relabeled nodes.
    """
    nodes = list(g.nodes())
    perm = list(range(len(nodes)))
    rng.shuffle(perm)
    mapping = dict(zip(nodes, perm, strict=True))
    return nx.relabel_nodes(g, mapping)


def _generate_invariance_graphs(
    rng: random.Random, count: int, max_nodes: int
) -> list[tuple[str, nx.Graph | nx.DiGraph, nx.Graph | nx.DiGraph]]:
    """Generate pairs of isomorphic graphs for invariance testing.

    Returns list of (source_name, graph_original, graph_relabeled).
    """
    pairs: list[tuple[str, nx.Graph | nx.DiGraph, nx.Graph | nx.DiGraph]] = []

    # Allocate counts across families
    n_per_family = max(count // 8, 1)

    # Trees
    for _ in range(n_per_family):
        n = rng.randint(3, max_nodes)
        seed = rng.randint(0, 2**31)
        g = nx.random_labeled_tree(n, seed=seed)
        g2 = _random_relabeling(g, rng)
        pairs.append(("tree", g, g2))

    # Cycles
    for n in range(3, min(max_nodes + 1, 3 + n_per_family)):
        g = nx.cycle_graph(n)
        g2 = _random_relabeling(g, rng)
        pairs.append(("cycle", g, g2))

    # Complete graphs
    for n in range(3, min(max_nodes + 1, 3 + n_per_family)):
        g = nx.complete_graph(n)
        g2 = _random_relabeling(g, rng)
        pairs.append(("complete", g, g2))

    # Star graphs
    for n in range(3, min(max_nodes + 1, 3 + n_per_family)):
        g = nx.star_graph(n - 1)
        g2 = _random_relabeling(g, rng)
        pairs.append(("star", g, g2))

    # Wheel graphs
    for n in range(4, min(max_nodes + 1, 4 + n_per_family)):
        g = nx.wheel_graph(n)
        g2 = _random_relabeling(g, rng)
        pairs.append(("wheel", g, g2))

    # Petersen
    g = nx.petersen_graph()
    if g.number_of_nodes() <= max_nodes:
        g2 = _random_relabeling(g, rng)
        pairs.append(("petersen", g, g2))

    # GNP (Erdos-Renyi)
    attempts = 0
    generated = 0
    while generated < n_per_family and attempts < n_per_family * 5:
        attempts += 1
        n = rng.randint(3, max_nodes)
        p = rng.uniform(0.2, 0.6)
        seed = rng.randint(0, 2**31)
        g = nx.gnp_random_graph(n, p, seed=seed)
        if nx.is_connected(g):
            g2 = _random_relabeling(g, rng)
            pairs.append(("gnp", g, g2))
            generated += 1

    # Barabasi-Albert
    for _ in range(n_per_family):
        n = rng.randint(4, max_nodes)
        m = rng.randint(1, min(3, n - 1))
        seed = rng.randint(0, 2**31)
        g = nx.barabasi_albert_graph(n, m, seed=seed)
        g2 = _random_relabeling(g, rng)
        pairs.append(("barabasi_albert", g, g2))

    # Ladder
    for n in range(3, min(max_nodes // 2 + 1, 3 + n_per_family)):
        g = nx.ladder_graph(n)
        g2 = _random_relabeling(g, rng)
        pairs.append(("ladder", g, g2))

    return pairs


def _generate_discrimination_pairs(
    rng: random.Random, count: int, max_nodes: int
) -> list[tuple[str, nx.Graph, nx.Graph]]:
    """Generate pairs of non-isomorphic graphs for discrimination testing.

    Each pair is chosen to have the same node count but different structure.

    Returns list of (source_name, graph1, graph2).
    """
    pairs: list[tuple[str, nx.Graph, nx.Graph]] = []

    # Cycle vs Path (same node count, different structure)
    for n in range(4, min(max_nodes + 1, 4 + count // 4)):
        g1 = nx.cycle_graph(n)
        g2 = nx.path_graph(n)
        # Path is connected, cycle has one more edge
        pairs.append(("cycle_vs_path", g1, g2))

    # Star vs Path (same node count)
    for n in range(4, min(max_nodes + 1, 4 + count // 4)):
        g1 = nx.star_graph(n - 1)
        g2 = nx.path_graph(n)
        pairs.append(("star_vs_path", g1, g2))

    # Complete vs Cycle (same node count)
    for n in range(4, min(max_nodes + 1, 4 + count // 4)):
        g1 = nx.complete_graph(n)
        g2 = nx.cycle_graph(n)
        pairs.append(("complete_vs_cycle", g1, g2))

    # Random GNP pairs with different densities
    attempts = 0
    generated = 0
    while generated < count // 4 and attempts < count * 3:
        attempts += 1
        n = rng.randint(4, max_nodes)
        seed1 = rng.randint(0, 2**31)
        seed2 = rng.randint(0, 2**31)
        g1 = nx.gnp_random_graph(n, 0.3, seed=seed1)
        g2 = nx.gnp_random_graph(n, 0.5, seed=seed2)
        if nx.is_connected(g1) and nx.is_connected(g2) and not nx.is_isomorphic(g1, g2):
            pairs.append(("gnp_different", g1, g2))
            generated += 1

    return pairs


# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------


def _test_invariance(
    test_id: int,
    source_name: str,
    g_original: nx.Graph | nx.DiGraph,
    g_relabeled: nx.Graph | nx.DiGraph,
) -> TestResult:
    """Test canonical string invariance for an isomorphic pair.

    Converts both to SparseGraph, computes canonical strings, asserts equality.
    """
    t0 = time.perf_counter()
    try:
        adapter = NetworkXAdapter()
        directed = isinstance(g_original, nx.DiGraph)

        sg1 = adapter.from_external(g_original, directed=directed)
        sg2 = adapter.from_external(g_relabeled, directed=directed)

        c1 = canonical_string(sg1)
        c2 = canonical_string(sg2)

        passed = c1 == c2
        error = ""
        if not passed:
            error = f"Canonical strings differ for isomorphic graphs: '{c1}' != '{c2}'"

        elapsed = time.perf_counter() - t0
        return TestResult(
            test_id=test_id,
            test_type="invariance",
            source=source_name,
            num_nodes=sg1.node_count(),
            num_edges=sg1.logical_edge_count(),
            canonical_1=c1,
            canonical_2=c2,
            passed=passed,
            error=error,
            time_s=elapsed,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return TestResult(
            test_id=test_id,
            test_type="invariance",
            source=source_name,
            num_nodes=g_original.number_of_nodes(),
            num_edges=g_original.number_of_edges(),
            canonical_1="",
            canonical_2="",
            passed=False,
            error=f"Exception: {exc!r}",
            time_s=elapsed,
        )


def _test_discrimination(
    test_id: int,
    source_name: str,
    g1: nx.Graph,
    g2: nx.Graph,
) -> TestResult:
    """Test that non-isomorphic graphs produce different canonical strings.

    Args:
        test_id: Test identifier.
        source_name: Description of the graph pair origin.
        g1: First graph (non-isomorphic to g2).
        g2: Second graph (non-isomorphic to g1).
    """
    t0 = time.perf_counter()
    try:
        adapter = NetworkXAdapter()

        sg1 = adapter.from_external(g1, directed=False)
        sg2 = adapter.from_external(g2, directed=False)

        c1 = canonical_string(sg1)
        c2 = canonical_string(sg2)

        passed = c1 != c2
        error = ""
        if not passed:
            error = (
                f"Canonical strings EQUAL for non-isomorphic graphs: "
                f"'{c1}' (nodes={sg1.node_count()}, edges={sg1.logical_edge_count()}) vs "
                f"'{c2}' (nodes={sg2.node_count()}, edges={sg2.logical_edge_count()}). "
                f"This would mean the canonical string is NOT a complete invariant!"
            )

        elapsed = time.perf_counter() - t0
        return TestResult(
            test_id=test_id,
            test_type="discrimination",
            source=source_name,
            num_nodes=max(sg1.node_count(), sg2.node_count()),
            num_edges=max(sg1.logical_edge_count(), sg2.logical_edge_count()),
            canonical_1=c1,
            canonical_2=c2,
            passed=passed,
            error=error,
            time_s=elapsed,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return TestResult(
            test_id=test_id,
            test_type="discrimination",
            source=source_name,
            num_nodes=max(g1.number_of_nodes(), g2.number_of_nodes()),
            num_edges=max(g1.number_of_edges(), g2.number_of_edges()),
            canonical_1="",
            canonical_2="",
            passed=False,
            error=f"Exception: {exc!r}",
            time_s=elapsed,
        )


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark(
    num_tests: int,
    max_nodes: int,
    seed: int,
    output_dir: str,
) -> BenchmarkSummary:
    """Run the full Phase 3 canonical invariance benchmark.

    Args:
        num_tests: Target number of tests (split between invariance and discrimination).
        max_nodes: Maximum graph size (canonical is expensive, keep small).
        seed: Random seed for reproducibility.
        output_dir: Directory for saving results JSON.

    Returns:
        BenchmarkSummary with aggregate statistics.
    """
    rng = random.Random(seed)
    summary = BenchmarkSummary()
    all_results: list[TestResult] = []
    test_id = 0

    # Split: ~70% invariance, ~30% discrimination
    n_invariance = max(num_tests * 7 // 10, 5)
    n_discrimination = max(num_tests - n_invariance, 5)

    print("Phase 3 Canonical String Invariance Benchmark")
    print(f"{'=' * 60}")
    print(f"Total tests target: {num_tests}")
    print(f"Seed: {seed}")
    print(f"Max nodes: {max_nodes}")
    print(f"Output dir: {output_dir}")
    print()

    # ---- Part A: Invariance tests ----
    print(f"[A] Generating invariance test pairs ({n_invariance} target)...")
    inv_pairs = _generate_invariance_graphs(rng, n_invariance, max_nodes)
    print(f"    Generated {len(inv_pairs)} isomorphic pairs.")

    for i, (source_name, g1, g2) in enumerate(inv_pairs):
        result = _test_invariance(test_id, source_name, g1, g2)
        all_results.append(result)
        test_id += 1

        if (i + 1) % max(1, len(inv_pairs) // 10) == 0:
            print(f"  ... {i + 1}/{len(inv_pairs)} done")

    # ---- Part B: Discrimination tests ----
    print(f"\n[B] Generating discrimination test pairs ({n_discrimination} target)...")
    disc_pairs = _generate_discrimination_pairs(rng, n_discrimination, max_nodes)
    print(f"    Generated {len(disc_pairs)} non-isomorphic pairs.")

    for i, (source_name, g1, g2) in enumerate(disc_pairs):
        result = _test_discrimination(test_id, source_name, g1, g2)
        all_results.append(result)
        test_id += 1

        if (i + 1) % max(1, len(disc_pairs) // 10) == 0:
            print(f"  ... {i + 1}/{len(disc_pairs)} done")

    # ---- Aggregate results ----
    total_time = sum(r.time_s for r in all_results)
    summary.total_tests = len(all_results)
    summary.total_time_s = total_time

    for r in all_results:
        if r.passed:
            summary.passed += 1
        elif "Exception" in r.error:
            summary.errors += 1
        else:
            summary.failed += 1

        if r.test_type == "invariance":
            summary.invariance_tests += 1
            if r.passed:
                summary.invariance_passed += 1
        else:
            summary.discrimination_tests += 1
            if r.passed:
                summary.discrimination_passed += 1

        # Track per-source stats
        key = f"{r.test_type}/{r.source}"
        if key not in summary.results_by_source:
            summary.results_by_source[key] = {"total": 0, "passed": 0, "failed": 0}
        summary.results_by_source[key]["total"] += 1
        if r.passed:
            summary.results_by_source[key]["passed"] += 1
        else:
            summary.results_by_source[key]["failed"] += 1

        if not r.passed:
            summary.failures.append(
                {
                    "test_id": r.test_id,
                    "test_type": r.test_type,
                    "source": r.source,
                    "num_nodes": r.num_nodes,
                    "num_edges": r.num_edges,
                    "canonical_1": r.canonical_1[:200],
                    "canonical_2": r.canonical_2[:200],
                    "error": r.error,
                }
            )

    # ---- Print summary table ----
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total tests:           {summary.total_tests}")
    print(f"Passed:                {summary.passed}")
    print(f"Failed:                {summary.failed}")
    print(f"Errors:                {summary.errors}")
    pass_rate = summary.passed / max(1, summary.total_tests) * 100
    print(f"Pass rate:             {pass_rate:.1f}%")
    print(f"Total time:            {summary.total_time_s:.2f}s")
    avg_time = summary.total_time_s / max(1, summary.total_tests)
    print(f"Avg per test:          {avg_time * 1000:.1f}ms")
    print()

    inv_rate = summary.invariance_passed / max(1, summary.invariance_tests) * 100
    disc_rate = summary.discrimination_passed / max(1, summary.discrimination_tests) * 100
    print(
        f"Invariance tests:      {summary.invariance_tests} "
        f"(passed: {summary.invariance_passed}, rate: {inv_rate:.1f}%)"
    )
    print(
        f"Discrimination tests:  {summary.discrimination_tests} "
        f"(passed: {summary.discrimination_passed}, rate: {disc_rate:.1f}%)"
    )
    print()

    # Per-source breakdown
    print(f"{'Source':<35} {'Total':>6} {'Pass':>6} {'Fail':>6} {'Rate':>8}")
    print(f"{'-' * 35} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 8}")
    for source, stats in sorted(summary.results_by_source.items()):
        rate = stats["passed"] / max(1, stats["total"]) * 100
        print(
            f"{source:<35} {stats['total']:>6} {stats['passed']:>6} "
            f"{stats['failed']:>6} {rate:>7.1f}%"
        )
    print()

    # Print failures (up to 20)
    if summary.failures:
        print("FAILURES (showing up to 20):")
        print(f"{'-' * 60}")
        for f in summary.failures[:20]:
            print(
                f"  Test #{f['test_id']} [{f['test_type']}/{f['source']}] "
                f"nodes={f['num_nodes']} edges={f['num_edges']}"
            )
            print(f"    Error: {f['error'][:150]}")
            if f["canonical_1"] and f["canonical_2"]:
                print(f"    C1: {f['canonical_1'][:80]}")
                print(f"    C2: {f['canonical_2'][:80]}")
            print()

    # ---- Save results JSON ----
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "canonical_invariance_results.json")

    results_json: dict[str, Any] = {
        "benchmark": "canonical_invariance",
        "config": {
            "num_tests": num_tests,
            "max_nodes": max_nodes,
            "seed": seed,
        },
        "summary": {
            "total_tests": summary.total_tests,
            "passed": summary.passed,
            "failed": summary.failed,
            "errors": summary.errors,
            "pass_rate_pct": round(pass_rate, 2),
            "total_time_s": round(summary.total_time_s, 4),
            "avg_time_ms": round(avg_time * 1000, 2),
            "invariance_tests": summary.invariance_tests,
            "invariance_passed": summary.invariance_passed,
            "discrimination_tests": summary.discrimination_tests,
            "discrimination_passed": summary.discrimination_passed,
        },
        "results_by_source": summary.results_by_source,
        "failures": summary.failures,
    }

    with open(output_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Results saved to: {output_path}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the canonical invariance benchmark."""
    parser = argparse.ArgumentParser(
        description="Phase 3: Canonical string invariance testing for IsalGraph."
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=DEFAULT_NUM_TESTS,
        help=f"Target number of tests (default: {DEFAULT_NUM_TESTS}).",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=DEFAULT_MAX_NODES,
        help=f"Maximum nodes per graph (default: {DEFAULT_MAX_NODES}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for results JSON (default: {DEFAULT_OUTPUT_DIR}).",
    )

    args = parser.parse_args()

    summary = run_benchmark(
        num_tests=args.num_tests,
        max_nodes=args.max_nodes,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    # Exit with non-zero code if any test failed (useful for CI).
    if summary.failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
