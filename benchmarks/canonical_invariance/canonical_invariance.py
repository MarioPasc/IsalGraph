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
) -> tuple[BenchmarkSummary, list[TestResult]]:
    """Run the full Phase 3 canonical invariance benchmark.

    Args:
        num_tests: Target number of tests (split between invariance and discrimination).
        max_nodes: Maximum graph size (canonical is expensive, keep small).
        seed: Random seed for reproducibility.
        output_dir: Directory for saving results JSON.

    Returns:
        Tuple of (BenchmarkSummary, list of individual TestResults).
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

    return summary, all_results


# ---------------------------------------------------------------------------
# Parallel execution support
# ---------------------------------------------------------------------------


def _parallel_invariance(
    args_tuple: tuple[int, str, Any, Any],
) -> TestResult:
    """Worker for ProcessPoolExecutor: invariance test."""
    test_id, source_name, g1, g2 = args_tuple
    return _test_invariance(test_id, source_name, g1, g2)


def _parallel_discrimination(
    args_tuple: tuple[int, str, Any, Any],
) -> TestResult:
    """Worker for ProcessPoolExecutor: discrimination test."""
    test_id, source_name, g1, g2 = args_tuple
    return _test_discrimination(test_id, source_name, g1, g2)


def run_benchmark_parallel(
    num_tests: int,
    max_nodes: int,
    seed: int,
    output_dir: str,
    n_workers: int = 4,
) -> tuple[BenchmarkSummary, list[TestResult]]:
    """Run benchmark with ProcessPoolExecutor parallelization.

    Same logic as run_benchmark but distributes tests across workers.

    Args:
        num_tests: Target number of tests.
        max_nodes: Maximum graph size.
        seed: Random seed for reproducibility.
        output_dir: Directory for saving results JSON.
        n_workers: Number of parallel workers.

    Returns:
        Tuple of (BenchmarkSummary, list of individual TestResults).
    """
    rng = random.Random(seed)
    all_results: list[TestResult] = []
    test_id = 0

    n_invariance = max(num_tests * 7 // 10, 5)
    n_discrimination = max(num_tests - n_invariance, 5)

    print(f"Phase 3 Canonical Invariance (parallel, {n_workers} workers)")
    print(f"{'=' * 60}")
    print(f"Total tests target: {num_tests}, Seed: {seed}")
    print()

    # ---- Prepare invariance tasks ----
    inv_pairs = _generate_invariance_graphs(rng, n_invariance, max_nodes)
    inv_tasks: list[tuple[int, str, Any, Any]] = []
    for source_name, g1, g2 in inv_pairs:
        inv_tasks.append((test_id, source_name, g1, g2))
        test_id += 1

    # ---- Prepare discrimination tasks ----
    disc_pairs = _generate_discrimination_pairs(rng, n_discrimination, max_nodes)
    disc_tasks: list[tuple[int, str, Any, Any]] = []
    for source_name, g1, g2 in disc_pairs:
        disc_tasks.append((test_id, source_name, g1, g2))
        test_id += 1

    # ---- Execute in parallel ----
    from concurrent.futures import ProcessPoolExecutor, as_completed

    print(f"Submitting {len(inv_tasks)} invariance + {len(disc_tasks)} discrimination tasks...")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        inv_futures = {executor.submit(_parallel_invariance, t): t[0] for t in inv_tasks}
        disc_futures = {executor.submit(_parallel_discrimination, t): t[0] for t in disc_tasks}

        total = len(inv_futures) + len(disc_futures)
        for done_count, fut in enumerate(as_completed({**inv_futures, **disc_futures}), 1):
            all_results.append(fut.result())
            if done_count % max(1, total // 20) == 0:
                print(f"  ... {done_count}/{total} done")

    # ---- Aggregate (same as run_benchmark) ----
    summary = BenchmarkSummary()
    summary.total_tests = len(all_results)
    summary.total_time_s = sum(r.time_s for r in all_results)

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

    pass_rate = summary.passed / max(1, summary.total_tests) * 100
    print(f"\nPassed: {summary.passed}/{summary.total_tests} ({pass_rate:.1f}%)")
    print(f"Total time: {summary.total_time_s:.2f}s")

    # Save JSON
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "canonical_invariance_results.json")
    with open(output_path, "w") as f:
        json.dump(
            {
                "benchmark": "canonical_invariance",
                "config": {
                    "num_tests": num_tests,
                    "max_nodes": max_nodes,
                    "seed": seed,
                    "n_workers": n_workers,
                },
                "summary": {
                    "total_tests": summary.total_tests,
                    "passed": summary.passed,
                    "failed": summary.failed,
                    "errors": summary.errors,
                    "pass_rate_pct": round(pass_rate, 2),
                    "total_time_s": round(summary.total_time_s, 4),
                    "invariance_tests": summary.invariance_tests,
                    "invariance_passed": summary.invariance_passed,
                    "discrimination_tests": summary.discrimination_tests,
                    "discrimination_passed": summary.discrimination_passed,
                },
                "results_by_source": summary.results_by_source,
                "failures": summary.failures,
            },
            f,
            indent=2,
        )
    print(f"Results saved to: {output_path}")
    return summary, all_results


# ---------------------------------------------------------------------------
# CSV / Figure / Table generation
# ---------------------------------------------------------------------------


def save_csv(results: list[TestResult], output_dir: str) -> str:
    """Save raw results as CSV.

    Args:
        results: List of TestResult objects.
        output_dir: Directory to save the CSV file.

    Returns:
        Path to the saved CSV file.
    """
    import csv as csv_mod

    rows = []
    for r in results:
        rows.append(
            {
                "test_id": r.test_id,
                "test_type": r.test_type,
                "source": r.source,
                "num_nodes": r.num_nodes,
                "num_edges": r.num_edges,
                "canonical_1_len": len(r.canonical_1),
                "canonical_2_len": len(r.canonical_2),
                "canonical_match": r.canonical_1 == r.canonical_2,
                "passed": r.passed,
                "time_s": r.time_s,
            }
        )

    path = os.path.join(output_dir, "canonical_invariance_results.csv")
    os.makedirs(output_dir, exist_ok=True)
    fieldnames = [
        "test_id",
        "test_type",
        "source",
        "num_nodes",
        "num_edges",
        "canonical_1_len",
        "canonical_2_len",
        "canonical_match",
        "passed",
        "time_s",
    ]
    with open(path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved to: {path}")
    return path


def generate_figure(results: list[TestResult], output_dir: str) -> list[str]:
    """Generate publication figure: 1x2 (timing violin + string length box).

    Panel (a): Violin + strip plot of computation time vs num_nodes with
               exponential fit overlay. Log-scale y-axis.
    Panel (b): Box plot of canonical string length vs num_nodes, colored by
               test_type (invariance vs discrimination).

    Args:
        results: List of TestResult objects.
        output_dir: Directory for saving figures.

    Returns:
        List of saved file paths.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import matplotlib.pyplot as plt  # noqa: E402
    import numpy as np  # noqa: E402
    from plotting_styles import (  # noqa: E402
        PAUL_TOL_BRIGHT,
        PLOT_SETTINGS,
        apply_ieee_style,
        get_figure_size,
        save_figure,
    )

    apply_ieee_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=get_figure_size("double", 0.5))

    # ---- Shared data: group by num_nodes ----
    node_times: dict[int, list[float]] = {}
    node_lens_inv: dict[int, list[int]] = {}
    node_lens_disc: dict[int, list[int]] = {}
    for r in results:
        node_times.setdefault(r.num_nodes, []).append(r.time_s)
        c_len = max(len(r.canonical_1), len(r.canonical_2))
        if r.test_type == "invariance":
            node_lens_inv.setdefault(r.num_nodes, []).append(c_len)
        else:
            node_lens_disc.setdefault(r.num_nodes, []).append(c_len)

    sizes = sorted(node_times.keys())
    x_positions = np.arange(1, len(sizes) + 1)
    violin_color = PAUL_TOL_BRIGHT["blue"]

    # ================================================================
    # Panel (a): Violin + strip plot of computation time vs num_nodes
    # ================================================================
    time_data = [node_times[s] for s in sizes]

    vp = ax1.violinplot(
        time_data,
        positions=x_positions,
        showmeans=False,
        showmedians=True,
        showextrema=False,
        widths=0.7,
    )
    # Style violins
    for body in vp["bodies"]:
        body.set_facecolor(violin_color)
        body.set_edgecolor(PAUL_TOL_BRIGHT["blue"])
        body.set_alpha(0.3)
        body.set_linewidth(0.8)
    vp["cmedians"].set_color(PAUL_TOL_BRIGHT["blue"])
    vp["cmedians"].set_linewidth(1.2)

    # Strip plot: jittered individual points
    rng = np.random.default_rng(42)
    for pos, data in zip(x_positions, time_data, strict=True):
        jitter = rng.uniform(-0.15, 0.15, size=len(data))
        ax1.scatter(
            pos + jitter,
            data,
            s=PLOT_SETTINGS["scatter_size"] * 0.6,
            color=violin_color,
            alpha=0.4,
            edgecolors="none",
            zorder=3,
        )

    # Exponential fit: log(time) ~ a * N + b  =>  time ~ exp(b) * exp(a*N)
    all_nodes = np.array([r.num_nodes for r in results], dtype=float)
    all_times = np.array([r.time_s for r in results], dtype=float)
    # Filter out zero/negative times for log fit
    mask = all_times > 0
    log_times = np.log(all_times[mask])
    nodes_masked = all_nodes[mask]
    # Linear regression: log(t) = a * N + b
    a_fit, b_fit = np.polyfit(nodes_masked, log_times, 1)
    n_smooth = np.linspace(min(sizes), max(sizes), 200)
    # Map graph node values to x-axis positions for the fit line
    pos_smooth = np.interp(
        n_smooth,
        np.array(sizes, dtype=float),
        x_positions.astype(float),
    )
    t_fit = np.exp(b_fit) * np.exp(a_fit * n_smooth)
    ax1.plot(
        pos_smooth,
        t_fit,
        color=PAUL_TOL_BRIGHT["red"],
        linewidth=PLOT_SETTINGS["line_width_thick"],
        linestyle="--",
        label=f"Fit: $t \\approx {np.exp(b_fit):.1e} \\cdot e^{{{a_fit:.2f}N}}$",
        zorder=4,
    )

    ax1.set_yscale("log")
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([str(s) for s in sizes])
    ax1.set_xlabel("Number of nodes $N$")
    ax1.set_ylabel("Computation time (s)")
    ax1.yaxis.grid(
        True,
        alpha=PLOT_SETTINGS["grid_alpha"],
        linestyle=PLOT_SETTINGS["grid_linestyle"],
        linewidth=PLOT_SETTINGS["grid_linewidth"],
    )
    ax1.xaxis.grid(False)
    ax1.legend(fontsize=PLOT_SETTINGS["legend_fontsize"], loc="upper left")

    # Text box: pass count
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    ax1.text(
        0.97,
        0.03,
        f"{passed}/{total} tests passed",
        transform=ax1.transAxes,
        fontsize=PLOT_SETTINGS["annotation_fontsize"],
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )

    # Panel label
    ax1.text(
        -0.12,
        1.05,
        "(a)",
        transform=ax1.transAxes,
        fontsize=12,
        fontweight="bold",
    )

    # ================================================================
    # Panel (b): Box plot of canonical string length vs num_nodes
    #            colored by test_type
    # ================================================================
    color_inv = PAUL_TOL_BRIGHT["blue"]
    color_disc = PAUL_TOL_BRIGHT["red"]
    box_width = 0.35

    # Prepare data arrays for each test type at each node count
    inv_data = [node_lens_inv.get(s, []) for s in sizes]
    disc_data = [node_lens_disc.get(s, []) for s in sizes]

    # Draw invariance boxes (left offset)
    bp_inv = ax2.boxplot(
        inv_data,
        positions=x_positions - box_width / 2 - 0.02,
        widths=box_width,
        patch_artist=True,
        flierprops={
            "markersize": PLOT_SETTINGS["boxplot_flier_size"],
            "marker": ".",
            "markerfacecolor": color_inv,
            "markeredgecolor": "none",
            "alpha": 0.5,
        },
        medianprops={"color": "black", "linewidth": 1.0},
        whiskerprops={"linewidth": PLOT_SETTINGS["boxplot_linewidth"]},
        capprops={"linewidth": PLOT_SETTINGS["boxplot_linewidth"]},
        boxprops={"linewidth": PLOT_SETTINGS["boxplot_linewidth"]},
    )
    for patch in bp_inv["boxes"]:
        patch.set_facecolor(color_inv)
        patch.set_alpha(0.6)

    # Draw discrimination boxes (right offset)
    bp_disc = ax2.boxplot(
        disc_data,
        positions=x_positions + box_width / 2 + 0.02,
        widths=box_width,
        patch_artist=True,
        flierprops={
            "markersize": PLOT_SETTINGS["boxplot_flier_size"],
            "marker": ".",
            "markerfacecolor": color_disc,
            "markeredgecolor": "none",
            "alpha": 0.5,
        },
        medianprops={"color": "black", "linewidth": 1.0},
        whiskerprops={"linewidth": PLOT_SETTINGS["boxplot_linewidth"]},
        capprops={"linewidth": PLOT_SETTINGS["boxplot_linewidth"]},
        boxprops={"linewidth": PLOT_SETTINGS["boxplot_linewidth"]},
    )
    for patch in bp_disc["boxes"]:
        patch.set_facecolor(color_disc)
        patch.set_alpha(0.6)

    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([str(s) for s in sizes])
    ax2.set_xlabel("Number of nodes $N$")
    ax2.set_ylabel("Canonical string length")
    ax2.yaxis.grid(
        True,
        alpha=PLOT_SETTINGS["grid_alpha"],
        linestyle=PLOT_SETTINGS["grid_linestyle"],
        linewidth=PLOT_SETTINGS["grid_linewidth"],
    )
    ax2.xaxis.grid(False)

    # Legend with proxy patches
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=color_inv, alpha=0.6, edgecolor="black", linewidth=0.5, label="Invariance"),
        Patch(
            facecolor=color_disc,
            alpha=0.6,
            edgecolor="black",
            linewidth=0.5,
            label="Discrimination",
        ),
    ]
    ax2.legend(
        handles=legend_handles,
        fontsize=PLOT_SETTINGS["legend_fontsize"],
        loc="upper left",
    )

    # Panel label
    ax2.text(
        -0.12,
        1.05,
        "(b)",
        transform=ax2.transAxes,
        fontsize=12,
        fontweight="bold",
    )

    fig.tight_layout()
    paths = save_figure(fig, os.path.join(output_dir, "canonical_invariance_figure"))
    plt.close(fig)
    print(f"Figure saved: {paths}")
    return paths


def generate_table(results: list[TestResult], output_dir: str) -> str:
    """Generate LaTeX table: pass rate and timing by family and test type.

    Args:
        results: List of TestResult objects.
        output_dir: Directory for saving the table.

    Returns:
        Path to the saved .tex file.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import pandas as pd  # noqa: E402
    from plotting_styles import binomial_ci, save_latex_table  # noqa: E402

    # Group by (source, test_type)
    group_stats: dict[tuple[str, str], list[TestResult]] = {}
    for r in results:
        key = (r.source, r.test_type)
        group_stats.setdefault(key, []).append(r)

    rows = []
    for fam, tt in sorted(group_stats):
        fam_results = group_stats[(fam, tt)]
        n = len(fam_results)
        k = sum(1 for r in fam_results if r.passed)
        rate = k / n
        lo, hi = binomial_ci(k, n)
        times = [r.time_s for r in fam_results]
        max_n = max(r.num_nodes for r in fam_results)
        rows.append(
            {
                "Family": fam,
                "Test_type": tt,
                "N_tests": n,
                "Pass_rate": f"{rate * 100:.1f}\\%",
                "95\\% CI": f"[{lo * 100:.1f}, {hi * 100:.1f}]",
                "Mean_time_s": f"{sum(times) / len(times):.4f}",
                "Max_N": max_n,
            }
        )

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "canonical_invariance_table.tex")
    save_latex_table(
        df,
        path,
        caption="Canonical string invariance and discrimination pass rate by graph family.",
        label="tab:canonical_invariance",
    )
    print(f"Table saved to: {path}")
    return path


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
        summary, all_results = run_benchmark_parallel(
            num_tests=args.num_tests,
            max_nodes=args.max_nodes,
            seed=args.seed,
            output_dir=args.output_dir,
            n_workers=args.n_workers,
        )
    else:
        summary, all_results = run_benchmark(
            num_tests=args.num_tests,
            max_nodes=args.max_nodes,
            seed=args.seed,
            output_dir=args.output_dir,
        )

    if args.csv:
        save_csv(all_results, args.output_dir)
    if args.plot:
        generate_figure(all_results, args.output_dir)
    if args.table:
        generate_table(all_results, args.output_dir)

    # Exit with non-zero code if any test failed (useful for CI).
    if summary.failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
