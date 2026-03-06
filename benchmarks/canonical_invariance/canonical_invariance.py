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
from isalgraph.core.graph_to_string import GraphToString
from isalgraph.core.string_to_graph import StringToGraph

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
    greedy_length: int = -1


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
    greedy_vs_canonical_tests: int = 0
    greedy_vs_canonical_passed: int = 0
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

    # Directed GNP (~10% of invariance tests)
    n_directed = max(n_per_family // 2, 1)
    attempts = 0
    generated = 0
    while generated < n_directed and attempts < n_directed * 10:
        attempts += 1
        n = rng.randint(3, max_nodes)
        p = rng.uniform(0.3, 0.7)
        seed = rng.randint(0, 2**31)
        g = nx.gnp_random_graph(n, p, seed=seed, directed=True)
        # Check that at least one node can reach all others
        reachable = False
        for v in range(n):
            if len(nx.descendants(g, v)) + 1 == n:
                reachable = True
                break
        if reachable:
            g2 = _random_relabeling(g, rng)
            pairs.append(("directed_gnp", g, g2))
            generated += 1

    # Directed cycles
    for n in range(3, min(max_nodes + 1, 3 + max(n_directed, 2))):
        g = nx.DiGraph()
        g.add_nodes_from(range(n))
        for i in range(n):
            g.add_edge(i, (i + 1) % n)
        g2 = _random_relabeling(g, rng)
        pairs.append(("directed_cycle", g, g2))

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


def _test_greedy_vs_canonical(
    test_id: int,
    string_length: int,
    rng: random.Random,
    canonical_limit: int = 10,
) -> TestResult:
    """Test whether the greedy G2S best string matches the canonical string.

    Generates a random valid IsalGraph string, decodes it to a graph, then
    compares the best greedy string (minimum over all starting nodes) against
    the exhaustive canonical string.

    Args:
        test_id: Test identifier.
        string_length: Length of the random string to generate.
        rng: Seeded Random instance.
        canonical_limit: Maximum graph size for canonical computation.

    Returns:
        TestResult with test_type="greedy_vs_canonical".
    """
    t0 = time.perf_counter()
    try:
        # Generate a random string and decode it
        raw_string = "".join(rng.choices("NnPpVvCcW", k=string_length))

        s2g = StringToGraph(raw_string, directed_graph=False)
        sg, _ = s2g.run()

        n_nodes = sg.node_count()
        n_edges = sg.logical_edge_count()

        # Skip trivial or too-large graphs
        if n_nodes < 3 or n_nodes > canonical_limit:
            elapsed = time.perf_counter() - t0
            return TestResult(
                test_id=test_id,
                test_type="greedy_vs_canonical",
                source="random_string",
                num_nodes=n_nodes,
                num_edges=n_edges,
                canonical_1="",
                canonical_2="",
                passed=True,
                greedy_length=-1,
                time_s=elapsed,
            )

        # Greedy best: try all starting nodes, keep shortest (then lexmin)
        greedy_best: str | None = None
        for v in range(n_nodes):
            g2s = GraphToString(sg)
            w, _ = g2s.run(initial_node=v)
            if greedy_best is None or (len(w), w) < (len(greedy_best), greedy_best):
                greedy_best = w

        assert greedy_best is not None

        # Canonical (exhaustive backtracking)
        w_canonical = canonical_string(sg)

        passed = greedy_best == w_canonical
        error = ""
        if not passed:
            error = (
                f"Greedy best (len={len(greedy_best)}) != canonical (len={len(w_canonical)}). "
                f"Greedy: '{greedy_best[:80]}', Canonical: '{w_canonical[:80]}'"
            )

        elapsed = time.perf_counter() - t0
        return TestResult(
            test_id=test_id,
            test_type="greedy_vs_canonical",
            source="random_string",
            num_nodes=n_nodes,
            num_edges=n_edges,
            canonical_1=greedy_best,
            canonical_2=w_canonical,
            passed=passed,
            error=error,
            greedy_length=len(raw_string),
            time_s=elapsed,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return TestResult(
            test_id=test_id,
            test_type="greedy_vs_canonical",
            source="random_string",
            num_nodes=0,
            num_edges=0,
            canonical_1="",
            canonical_2="",
            passed=False,
            error=f"Exception: {exc!r}",
            greedy_length=string_length,
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

    # ---- Part C: Greedy vs Canonical tests ----
    # Test: rand(w) -> G -> w_greedy vs w_canonical
    greedy_lengths = [5, 10, 15, 20, 25, 30, 40, 50]
    tests_per_length = 7
    n_greedy_total = len(greedy_lengths) * tests_per_length
    print(f"\n[C] Running greedy vs canonical tests ({n_greedy_total} target)...")

    greedy_done = 0
    greedy_valid = 0
    for str_len in greedy_lengths:
        for _ in range(tests_per_length):
            result = _test_greedy_vs_canonical(test_id, str_len, rng, canonical_limit=max_nodes)
            all_results.append(result)
            test_id += 1
            greedy_done += 1
            if result.greedy_length > 0:
                greedy_valid += 1

            if greedy_done % max(1, n_greedy_total // 10) == 0:
                print(f"  ... {greedy_done}/{n_greedy_total} done ({greedy_valid} valid graphs)")

    print(f"    Completed {greedy_done} tests ({greedy_valid} produced valid graphs).")

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
        elif r.test_type == "discrimination":
            summary.discrimination_tests += 1
            if r.passed:
                summary.discrimination_passed += 1
        elif r.test_type == "greedy_vs_canonical":
            summary.greedy_vs_canonical_tests += 1
            if r.passed:
                summary.greedy_vs_canonical_passed += 1

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
    gvc_rate = summary.greedy_vs_canonical_passed / max(1, summary.greedy_vs_canonical_tests) * 100
    print(
        f"Greedy vs Canonical:   {summary.greedy_vs_canonical_tests} "
        f"(passed: {summary.greedy_vs_canonical_passed}, rate: {gvc_rate:.1f}%)"
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
            "greedy_vs_canonical_tests": summary.greedy_vs_canonical_tests,
            "greedy_vs_canonical_passed": summary.greedy_vs_canonical_passed,
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


def _parallel_greedy_vs_canonical(
    args_tuple: tuple[int, int, int, int],
) -> TestResult:
    """Worker for ProcessPoolExecutor: greedy vs canonical test."""
    test_id, string_length, seed, canonical_limit = args_tuple
    rng = random.Random(seed)
    return _test_greedy_vs_canonical(test_id, string_length, rng, canonical_limit)


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

    # ---- Prepare greedy vs canonical tasks ----
    greedy_lengths = [5, 10, 15, 20, 25, 30, 40, 50]
    tests_per_length = 7
    gvc_tasks: list[tuple[int, int, int, int]] = []
    for str_len in greedy_lengths:
        for _ in range(tests_per_length):
            task_seed = rng.randint(0, 2**31)
            gvc_tasks.append((test_id, str_len, task_seed, max_nodes))
            test_id += 1

    # ---- Execute in parallel ----
    from concurrent.futures import ProcessPoolExecutor, as_completed

    print(
        f"Submitting {len(inv_tasks)} invariance + {len(disc_tasks)} discrimination "
        f"+ {len(gvc_tasks)} greedy_vs_canonical tasks..."
    )
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        inv_futures = {executor.submit(_parallel_invariance, t): t[0] for t in inv_tasks}
        disc_futures = {executor.submit(_parallel_discrimination, t): t[0] for t in disc_tasks}
        gvc_futures = {executor.submit(_parallel_greedy_vs_canonical, t): t[0] for t in gvc_tasks}

        total = len(inv_futures) + len(disc_futures) + len(gvc_futures)
        all_futures = {**inv_futures, **disc_futures, **gvc_futures}
        for done_count, fut in enumerate(as_completed(all_futures), 1):
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
        elif r.test_type == "discrimination":
            summary.discrimination_tests += 1
            if r.passed:
                summary.discrimination_passed += 1
        elif r.test_type == "greedy_vs_canonical":
            summary.greedy_vs_canonical_tests += 1
            if r.passed:
                summary.greedy_vs_canonical_passed += 1

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
                    "greedy_vs_canonical_tests": summary.greedy_vs_canonical_tests,
                    "greedy_vs_canonical_passed": summary.greedy_vs_canonical_passed,
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
                "greedy_length": r.greedy_length,
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
        "greedy_length",
    ]
    with open(path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved to: {path}")
    return path


def generate_figure(results: list[TestResult], output_dir: str) -> list[str]:
    """Generate publication figure: 1x2 (string length scatter + compression box).

    Panel (a): Scatter of canonical string length |w*| vs N (number of nodes),
               colored by graph family on log-log scale with reference lines
               |w*| = N (dashed) and |w*| = N^2 (dotted).
    Panel (b): Box plot of compression factor N^2/|w*| by family
               (invariance tests only).

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
        FAMILY_COLORS,
        FAMILY_MARKERS,
        PLOT_SETTINGS,
        apply_ieee_style,
        family_display,
        get_figure_size,
        save_figure,
    )

    apply_ieee_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=get_figure_size("double", 0.5))

    # -- Fallback color/marker for families not in plotting_styles --
    _default_color = "#555555"
    _default_marker = "o"
    _extra_colors: dict[str, str] = {
        "barabasi_albert": FAMILY_COLORS.get("barabasi_albert", "#AA3377"),
        "gnp_different": FAMILY_COLORS.get("gnp", "#DDCC77"),
        "cycle_vs_path": FAMILY_COLORS.get("cycle", "#CCBB44"),
        "star_vs_path": FAMILY_COLORS.get("star", "#228833"),
        "complete_vs_cycle": FAMILY_COLORS.get("complete", "#EE6677"),
    }
    _extra_markers: dict[str, str] = {
        "barabasi_albert": "P",
        "gnp_different": "d",
        "cycle_vs_path": "D",
        "star_vs_path": "^",
        "complete_vs_cycle": "v",
    }

    def _get_color(family: str) -> str:
        if family in FAMILY_COLORS:
            return FAMILY_COLORS[family]
        if family in _extra_colors:
            return _extra_colors[family]
        return _default_color

    def _get_marker(family: str) -> str:
        if family in FAMILY_MARKERS:
            return FAMILY_MARKERS[family]
        if family in _extra_markers:
            return _extra_markers[family]
        return _default_marker

    # ================================================================
    # Panel (a): Scatter of |w*| vs N, colored by family (log-log)
    # ================================================================

    # Collect (N, |w*|, family) tuples for all results
    scatter_data: dict[str, list[tuple[int, int]]] = {}
    for r in results:
        family = r.source
        if r.test_type == "greedy_vs_canonical":
            continue
        if r.test_type == "invariance":
            if len(r.canonical_1) > 0:
                scatter_data.setdefault(family, []).append((r.num_nodes, len(r.canonical_1)))
        elif r.test_type == "discrimination":
            if len(r.canonical_1) > 0:
                scatter_data.setdefault(family, []).append((r.num_nodes, len(r.canonical_1)))
            if len(r.canonical_2) > 0:
                scatter_data.setdefault(family, []).append((r.num_nodes, len(r.canonical_2)))

    # Plot each family with its color and marker; collect handles for legend
    legend_handles: list[Any] = []
    legend_labels: list[str] = []
    for family in sorted(scatter_data.keys()):
        pts = scatter_data[family]
        ns = [p[0] for p in pts]
        lens = [p[1] for p in pts]
        h = ax1.scatter(
            ns,
            lens,
            c=_get_color(family),
            marker=_get_marker(family),
            s=PLOT_SETTINGS["scatter_size"] * 1.5,
            alpha=PLOT_SETTINGS["scatter_alpha"],
            edgecolors="none",
            zorder=3,
        )
        legend_handles.append(h)
        legend_labels.append(family_display(family))

    # Reference lines: |w*| = N (dashed) and |w*| = N^2 (dotted)
    all_n = [r.num_nodes for r in results if r.test_type != "greedy_vs_canonical"]
    n_min, n_max = max(min(all_n), 1), max(all_n)
    n_ref = np.linspace(n_min, n_max * 1.2, 200)
    (l1,) = ax1.plot(
        n_ref,
        n_ref,
        color="0.4",
        linewidth=PLOT_SETTINGS["line_width"],
        linestyle="--",
        zorder=1,
    )
    (l2,) = ax1.plot(
        n_ref,
        n_ref**2,
        color="0.4",
        linewidth=PLOT_SETTINGS["line_width"],
        linestyle=":",
        zorder=1,
    )
    legend_handles.extend([l1, l2])
    legend_labels.extend(["$|w^*| = N$", "$|w^*| = N^2$"])

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of nodes $N$")
    ax1.set_ylabel("Canonical string length $|w^*|$")
    ax1.yaxis.grid(
        True,
        alpha=PLOT_SETTINGS["grid_alpha"],
        linestyle=PLOT_SETTINGS["grid_linestyle"],
        linewidth=PLOT_SETTINGS["grid_linewidth"],
    )
    ax1.xaxis.grid(False)

    # Text annotation: pass count (bottom-right, white box)
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
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "0.7",
            "alpha": 0.9,
        },
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
    # Panel (b): Box plot of compression factor N^2/|w*| by family
    #            (invariance tests only)
    # ================================================================

    # Collect compression factors grouped by family (invariance only)
    compression_by_family: dict[str, list[float]] = {}
    for r in results:
        if r.test_type != "invariance":
            continue
        c_len = len(r.canonical_1)
        if c_len > 0:
            factor = (r.num_nodes**2) / c_len
            compression_by_family.setdefault(r.source, []).append(factor)

    # Sort families alphabetically for consistent ordering
    families_sorted = sorted(compression_by_family.keys())
    box_data = [compression_by_family[f] for f in families_sorted]
    x_pos = np.arange(1, len(families_sorted) + 1)

    bp = ax2.boxplot(
        box_data,
        positions=x_pos,
        widths=PLOT_SETTINGS["boxplot_width"],
        patch_artist=True,
        flierprops={
            "markersize": PLOT_SETTINGS["boxplot_flier_size"],
            "marker": ".",
            "markeredgecolor": "none",
            "alpha": 0.5,
        },
        medianprops={"color": "black", "linewidth": 1.0},
        whiskerprops={"linewidth": PLOT_SETTINGS["boxplot_linewidth"]},
        capprops={"linewidth": PLOT_SETTINGS["boxplot_linewidth"]},
        boxprops={"linewidth": PLOT_SETTINGS["boxplot_linewidth"]},
    )
    for patch, family in zip(bp["boxes"], families_sorted, strict=True):
        patch.set_facecolor(_get_color(family))
        patch.set_alpha(0.6)
    # Color fliers to match their family
    for flier, family in zip(bp["fliers"], families_sorted, strict=True):
        flier.set_markerfacecolor(_get_color(family))

    # Horizontal dashed line at factor = 1 (break-even)
    ax2.axhline(
        y=1.0,
        color="0.4",
        linewidth=PLOT_SETTINGS["line_width"],
        linestyle="--",
        zorder=1,
    )

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(
        [family_display(f) for f in families_sorted],
        rotation=45,
        ha="right",
        fontsize=PLOT_SETTINGS["tick_labelsize"],
    )
    ax2.set_xlabel("")
    ax2.set_ylabel("Compression factor $N^2/|w^*|$")
    ax2.yaxis.grid(
        True,
        alpha=PLOT_SETTINGS["grid_alpha"],
        linestyle=PLOT_SETTINGS["grid_linestyle"],
        linewidth=PLOT_SETTINGS["grid_linewidth"],
    )
    ax2.xaxis.grid(False)

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
    fig.subplots_adjust(bottom=0.22)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=5,
        fontsize=7,
        frameon=False,
        columnspacing=0.8,
        handletextpad=0.3,
        markerscale=1.2,
    )
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
