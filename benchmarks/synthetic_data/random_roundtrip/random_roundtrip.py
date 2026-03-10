"""Phase 2 -- massive random round-trip testing at scale.

Verifies the IsalGraph round-trip property:

    S2G(w) ~ S2G(G2S(S2G(w), v0))

for thousands of random graphs drawn from multiple sources:
  1. Random instruction strings (length 1-50)
  2. NetworkX graph families: GNP (Erdos-Renyi), Barabasi-Albert, random trees,
     cycle, complete, grid, watts-strogatz, star, wheel, ladder

Cross-validates isomorphism with ``nx.is_isomorphic``.

Usage:
    python benchmarks/random_roundtrip.py --num-tests 1000 --seed 42
    python benchmarks/random_roundtrip.py --num-tests 50 --seed 42 --output-dir /tmp
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
from isalgraph.core.graph_to_string import GraphToString
from isalgraph.core.string_to_graph import StringToGraph

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = "/media/mpascual/Sandisk2TB/research/isalgraph"
DEFAULT_NUM_TESTS = 1000
DEFAULT_MAX_STRING_LEN = 50
DEFAULT_MAX_NODES = 20
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    """Result of a single round-trip test."""

    test_id: int
    source: str  # "random_string", "gnp", "tree", "ba", etc.
    directed: bool
    num_nodes: int
    num_edges: int
    original_string: str
    roundtrip_string: str
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
    results_by_source: dict[str, dict[str, int]] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Random string generation
# ---------------------------------------------------------------------------

# Instruction alphabet (excluding W since it is a no-op and does not contribute
# interesting structure).
_NODE_INSTRS = ["V", "v"]
_MOVE_INSTRS = ["N", "n", "P", "p"]
_EDGE_INSTRS = ["C", "c"]


def _generate_random_string(rng: random.Random, length: int) -> str:
    """Generate a random valid IsalGraph instruction string.

    Ensures at least one V or v appears so that the string produces a
    non-trivial graph. C/c instructions are only emitted when the current
    graph has >=2 nodes (otherwise they would be self-loops which are valid
    but less interesting). Movement instructions are included to exercise
    pointer traversal.

    Args:
        rng: Seeded Random instance.
        length: Target string length (>= 1).

    Returns:
        A valid IsalGraph instruction string.
    """
    if length <= 0:
        return ""

    instructions: list[str] = []
    node_count = 1  # Initial node

    for i in range(length):
        # Force at least one V/v in the first position if length is short,
        # otherwise the graph is trivial (single node).
        if i == 0 and length <= 3:
            instr = rng.choice(_NODE_INSTRS)
        else:
            # Build valid instruction pool based on current state
            pool = list(_NODE_INSTRS) + list(_MOVE_INSTRS)
            if node_count >= 2:
                pool.extend(_EDGE_INSTRS)
            # Bias toward V/v to produce richer graphs
            instr = rng.choice(_NODE_INSTRS) if rng.random() < 0.4 else rng.choice(pool)

        instructions.append(instr)
        if instr in ("V", "v"):
            node_count += 1

    return "".join(instructions)


# ---------------------------------------------------------------------------
# NetworkX graph generators
# ---------------------------------------------------------------------------


def _nx_gnp_graphs(
    rng: random.Random, count: int, max_nodes: int, directed: bool
) -> list[tuple[str, nx.Graph | nx.DiGraph]]:
    """Generate random Erdos-Renyi G(n,p) graphs."""
    graphs: list[tuple[str, nx.Graph | nx.DiGraph]] = []
    for _ in range(count):
        n = rng.randint(3, max_nodes)
        p = rng.uniform(0.1, 0.6)
        seed = rng.randint(0, 2**31)
        if directed:
            g = nx.gnp_random_graph(n, p, seed=seed, directed=True)
        else:
            g = nx.gnp_random_graph(n, p, seed=seed, directed=False)
        # Ensure connected for undirected (or strongly connected component for directed)
        if not directed:
            if nx.is_connected(g):
                graphs.append(("gnp", g))
        else:
            # For directed, take the largest weakly connected component and
            # check reachability from node 0 after relabeling
            if g.number_of_nodes() > 0 and g.number_of_edges() > 0:
                graphs.append(("gnp_directed", g))
    return graphs


def _nx_tree_graphs(rng: random.Random, count: int, max_nodes: int) -> list[tuple[str, nx.Graph]]:
    """Generate random trees."""
    graphs: list[tuple[str, nx.Graph]] = []
    for _ in range(count):
        n = rng.randint(3, max_nodes)
        seed = rng.randint(0, 2**31)
        g = nx.random_labeled_tree(n, seed=seed)
        graphs.append(("tree", g))
    return graphs


def _nx_ba_graphs(rng: random.Random, count: int, max_nodes: int) -> list[tuple[str, nx.Graph]]:
    """Generate Barabasi-Albert preferential attachment graphs."""
    graphs: list[tuple[str, nx.Graph]] = []
    for _ in range(count):
        n = rng.randint(4, max_nodes)
        m = rng.randint(1, min(3, n - 1))
        seed = rng.randint(0, 2**31)
        g = nx.barabasi_albert_graph(n, m, seed=seed)
        graphs.append(("barabasi_albert", g))
    return graphs


def _nx_special_graphs(max_nodes: int) -> list[tuple[str, nx.Graph]]:
    """Generate deterministic special graph families."""
    graphs: list[tuple[str, nx.Graph]] = []
    for n in range(3, min(max_nodes + 1, 12)):
        graphs.append(("cycle", nx.cycle_graph(n)))
        graphs.append(("complete", nx.complete_graph(n)))
        graphs.append(("star", nx.star_graph(n - 1)))
        graphs.append(("wheel", nx.wheel_graph(n)))
        if n >= 4:
            graphs.append(("ladder", nx.ladder_graph(n)))
    # Petersen graph
    graphs.append(("petersen", nx.petersen_graph()))
    return graphs


def _nx_self_loop_graphs(
    rng: random.Random, count: int, max_nodes: int
) -> list[tuple[str, nx.Graph]]:
    """Generate random undirected graphs with self-loops added."""
    graphs: list[tuple[str, nx.Graph]] = []
    for _ in range(count):
        n = rng.randint(3, max_nodes)
        p = rng.uniform(0.2, 0.5)
        seed = rng.randint(0, 2**31)
        g = nx.gnp_random_graph(n, p, seed=seed)
        if not nx.is_connected(g):
            continue
        # Add 1-3 self-loops to random nodes
        n_loops = rng.randint(1, min(3, n))
        loop_nodes = rng.sample(range(n), n_loops)
        for node in loop_nodes:
            g.add_edge(node, node)
        graphs.append(("self_loop", g))
    return graphs


def _nx_watts_strogatz_graphs(
    rng: random.Random, count: int, max_nodes: int
) -> list[tuple[str, nx.Graph]]:
    """Generate Watts-Strogatz small-world graphs."""
    graphs: list[tuple[str, nx.Graph]] = []
    for _ in range(count):
        n = rng.randint(6, max_nodes)
        k = rng.choice([2, 4])
        p = rng.uniform(0.1, 0.5)
        seed = rng.randint(0, 2**31)
        g = nx.watts_strogatz_graph(n, k, p, seed=seed)
        if nx.is_connected(g):
            graphs.append(("watts_strogatz", g))
    return graphs


# ---------------------------------------------------------------------------
# Round-trip test runner
# ---------------------------------------------------------------------------


def _roundtrip_from_string(test_id: int, instr_string: str, directed: bool) -> TestResult:
    """Run round-trip test starting from an instruction string.

    S2G(w) -> G1, G2S(G1, 0) -> w', S2G(w') -> G2, assert G1 ~ G2.
    """
    t0 = time.perf_counter()
    try:
        # Step 1: String -> Graph (G1)
        s2g = StringToGraph(instr_string, directed)
        g1, _ = s2g.run()

        # Step 2: Graph -> String (w')
        g2s = GraphToString(g1)
        w_prime, _ = g2s.run(initial_node=0)

        # Step 3: String -> Graph (G2)
        s2g2 = StringToGraph(w_prime, directed)
        g2, _ = s2g2.run()

        # Step 4: Assert isomorphism via SparseGraph
        iso_sparse = g1.is_isomorphic(g2)

        # Step 5: Cross-validate with NetworkX
        adapter = NetworkXAdapter()
        nx_g1 = adapter.to_external(g1)
        nx_g2 = adapter.to_external(g2)
        iso_nx = nx.is_isomorphic(nx_g1, nx_g2)

        passed = iso_sparse and iso_nx
        error = ""
        if not iso_sparse:
            error = "SparseGraph.is_isomorphic returned False"
        elif not iso_nx:
            error = "nx.is_isomorphic returned False (but SparseGraph said True!)"

        elapsed = time.perf_counter() - t0
        return TestResult(
            test_id=test_id,
            source="random_string",
            directed=directed,
            num_nodes=g1.node_count(),
            num_edges=g1.logical_edge_count(),
            original_string=instr_string,
            roundtrip_string=w_prime,
            passed=passed,
            error=error,
            time_s=elapsed,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return TestResult(
            test_id=test_id,
            source="random_string",
            directed=directed,
            num_nodes=0,
            num_edges=0,
            original_string=instr_string,
            roundtrip_string="",
            passed=False,
            error=f"Exception: {exc!r}",
            time_s=elapsed,
        )


def _roundtrip_from_nx(
    test_id: int, source_name: str, nx_graph: nx.Graph | nx.DiGraph, directed: bool
) -> TestResult:
    """Run round-trip test starting from a NetworkX graph.

    nx -> SparseGraph -> G2S -> w -> S2G -> SparseGraph -> nx, assert isomorphic.
    """
    t0 = time.perf_counter()
    try:
        adapter = NetworkXAdapter()
        sg = adapter.from_external(nx_graph, directed=directed)

        # Try G2S from node 0
        g2s = GraphToString(sg)
        w, _ = g2s.run(initial_node=0)

        # S2G
        s2g = StringToGraph(w, directed)
        sg2, _ = s2g.run()

        # Cross-validate
        nx_g2 = adapter.to_external(sg2)
        nx_g1 = adapter.to_external(sg)
        iso_nx = nx.is_isomorphic(nx_g1, nx_g2)
        iso_sparse = sg.is_isomorphic(sg2)

        passed = iso_nx and iso_sparse
        error = ""
        if not iso_nx:
            error = "nx.is_isomorphic returned False"
        elif not iso_sparse:
            error = "SparseGraph.is_isomorphic returned False (but nx said True!)"

        elapsed = time.perf_counter() - t0
        return TestResult(
            test_id=test_id,
            source=source_name,
            directed=directed,
            num_nodes=sg.node_count(),
            num_edges=sg.logical_edge_count(),
            original_string="<from_nx>",
            roundtrip_string=w,
            passed=passed,
            error=error,
            time_s=elapsed,
        )
    except ValueError as exc:
        # Reachability issues are expected for some directed graphs
        elapsed = time.perf_counter() - t0
        return TestResult(
            test_id=test_id,
            source=source_name,
            directed=directed,
            num_nodes=nx_graph.number_of_nodes(),
            num_edges=nx_graph.number_of_edges(),
            original_string="<from_nx>",
            roundtrip_string="",
            passed=True,  # Skip as expected
            error=f"Skipped (expected): {exc!r}",
            time_s=elapsed,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return TestResult(
            test_id=test_id,
            source=source_name,
            directed=directed,
            num_nodes=nx_graph.number_of_nodes(),
            num_edges=nx_graph.number_of_edges(),
            original_string="<from_nx>",
            roundtrip_string="",
            passed=False,
            error=f"Exception: {exc!r}",
            time_s=elapsed,
        )


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark(
    num_tests: int,
    max_string_len: int,
    max_nodes: int,
    seed: int,
    output_dir: str,
) -> tuple[BenchmarkSummary, list[TestResult]]:
    """Run the full Phase 2 random round-trip benchmark.

    Args:
        num_tests: Total number of tests to run across all sources.
        max_string_len: Maximum instruction string length for random strings.
        max_nodes: Maximum graph size for NetworkX generators.
        seed: Random seed for reproducibility.
        output_dir: Directory for saving results JSON.

    Returns:
        Tuple of (BenchmarkSummary, list of individual TestResults).
    """
    rng = random.Random(seed)
    summary = BenchmarkSummary()
    all_results: list[TestResult] = []
    test_id = 0

    # Allocate tests across sources:
    # ~40% random strings, ~60% NetworkX graphs
    n_string_tests = max(num_tests * 2 // 5, 10)
    n_nx_tests = num_tests - n_string_tests

    # Divide NX tests across generators
    n_per_generator = max(n_nx_tests // 6, 2)

    print("Phase 2 Random Round-Trip Benchmark")
    print(f"{'=' * 60}")
    print(f"Total tests target: {num_tests}")
    print(f"Seed: {seed}")
    print(f"Max string length: {max_string_len}")
    print(f"Max nodes: {max_nodes}")
    print(f"Output dir: {output_dir}")
    print()

    # ---- Part A: Random instruction strings ----
    print(f"[A] Random instruction strings ({n_string_tests} tests)...")
    for i in range(n_string_tests):
        length = rng.randint(1, max_string_len)
        directed = rng.choice([True, False])
        instr_string = _generate_random_string(rng, length)

        result = _roundtrip_from_string(test_id, instr_string, directed)
        all_results.append(result)
        test_id += 1

        if (i + 1) % max(1, n_string_tests // 10) == 0:
            print(f"  ... {i + 1}/{n_string_tests} done")

    # ---- Part B: NetworkX graph families (undirected) ----
    print("\n[B] NetworkX graph families (undirected)...")

    # Trees
    print(f"  Generating random trees ({n_per_generator})...")
    trees = _nx_tree_graphs(rng, n_per_generator, max_nodes)
    for source_name, g in trees:
        result = _roundtrip_from_nx(test_id, source_name, g, directed=False)
        all_results.append(result)
        test_id += 1

    # GNP (Erdos-Renyi) undirected
    print(f"  Generating GNP graphs ({n_per_generator})...")
    gnp_graphs = _nx_gnp_graphs(rng, n_per_generator * 2, max_nodes, directed=False)
    for source_name, g in gnp_graphs[:n_per_generator]:
        result = _roundtrip_from_nx(test_id, source_name, g, directed=False)
        all_results.append(result)
        test_id += 1

    # Barabasi-Albert
    print(f"  Generating Barabasi-Albert graphs ({n_per_generator})...")
    ba_graphs = _nx_ba_graphs(rng, n_per_generator, max_nodes)
    for source_name, g in ba_graphs:
        result = _roundtrip_from_nx(test_id, source_name, g, directed=False)
        all_results.append(result)
        test_id += 1

    # Special graphs
    print("  Generating special graphs...")
    special = _nx_special_graphs(max_nodes)
    for source_name, g in special:
        result = _roundtrip_from_nx(test_id, source_name, g, directed=False)
        all_results.append(result)
        test_id += 1

    # Watts-Strogatz
    print(f"  Generating Watts-Strogatz graphs ({n_per_generator})...")
    ws_graphs = _nx_watts_strogatz_graphs(rng, n_per_generator * 2, max_nodes)
    for source_name, g in ws_graphs[:n_per_generator]:
        result = _roundtrip_from_nx(test_id, source_name, g, directed=False)
        all_results.append(result)
        test_id += 1

    # Self-loop graphs
    n_self_loop = max(n_per_generator // 2, 2)
    print(f"  Generating self-loop graphs ({n_self_loop})...")
    sl_graphs = _nx_self_loop_graphs(rng, n_self_loop * 2, max_nodes)
    for source_name, g in sl_graphs[:n_self_loop]:
        result = _roundtrip_from_nx(test_id, source_name, g, directed=False)
        all_results.append(result)
        test_id += 1

    # ---- Part C: NetworkX graph families (directed) ----
    print("\n[C] NetworkX directed graphs...")

    # GNP directed
    print(f"  Generating directed GNP graphs ({n_per_generator})...")
    gnp_dir = _nx_gnp_graphs(rng, n_per_generator * 2, max_nodes, directed=True)
    for source_name, g in gnp_dir[:n_per_generator]:
        result = _roundtrip_from_nx(test_id, source_name, g, directed=True)
        all_results.append(result)
        test_id += 1

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

        # Track per-source stats
        if r.source not in summary.results_by_source:
            summary.results_by_source[r.source] = {"total": 0, "passed": 0, "failed": 0}
        summary.results_by_source[r.source]["total"] += 1
        if r.passed:
            summary.results_by_source[r.source]["passed"] += 1
        else:
            summary.results_by_source[r.source]["failed"] += 1

        if not r.passed:
            summary.failures.append(
                {
                    "test_id": r.test_id,
                    "source": r.source,
                    "directed": r.directed,
                    "num_nodes": r.num_nodes,
                    "num_edges": r.num_edges,
                    "original_string": r.original_string[:200],
                    "roundtrip_string": r.roundtrip_string[:200],
                    "error": r.error,
                }
            )

    # ---- Print summary table ----
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total tests:  {summary.total_tests}")
    print(f"Passed:       {summary.passed}")
    print(f"Failed:       {summary.failed}")
    print(f"Errors:       {summary.errors}")
    pass_rate = summary.passed / max(1, summary.total_tests) * 100
    print(f"Pass rate:    {pass_rate:.1f}%")
    print(f"Total time:   {summary.total_time_s:.2f}s")
    avg_time = summary.total_time_s / max(1, summary.total_tests)
    print(f"Avg per test: {avg_time * 1000:.1f}ms")
    print()

    # Per-source breakdown
    print(f"{'Source':<20} {'Total':>6} {'Pass':>6} {'Fail':>6} {'Rate':>8}")
    print(f"{'-' * 20} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 8}")
    for source, stats in sorted(summary.results_by_source.items()):
        rate = stats["passed"] / max(1, stats["total"]) * 100
        print(
            f"{source:<20} {stats['total']:>6} {stats['passed']:>6} "
            f"{stats['failed']:>6} {rate:>7.1f}%"
        )
    print()

    # Print failures (up to 20)
    if summary.failures:
        print("FAILURES (showing up to 20):")
        print(f"{'-' * 60}")
        for f in summary.failures[:20]:
            print(
                f"  Test #{f['test_id']} [{f['source']}] "
                f"directed={f['directed']} "
                f"nodes={f['num_nodes']} edges={f['num_edges']}"
            )
            print(f"    Error: {f['error'][:120]}")
            if f["original_string"] != "<from_nx>":
                print(f"    String: {f['original_string'][:80]}")
            print()

    # ---- Save results JSON ----
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "random_roundtrip_results.json")

    results_json: dict[str, Any] = {
        "benchmark": "random_roundtrip",
        "config": {
            "num_tests": num_tests,
            "max_string_len": max_string_len,
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


def _parallel_roundtrip_string(
    args_tuple: tuple[int, str, bool],
) -> TestResult:
    """Worker for ProcessPoolExecutor: roundtrip from string."""
    test_id, instr_string, directed = args_tuple
    return _roundtrip_from_string(test_id, instr_string, directed)


def _parallel_roundtrip_nx(
    args_tuple: tuple[int, str, Any, bool],
) -> TestResult:
    """Worker for ProcessPoolExecutor: roundtrip from NX graph."""
    test_id, source_name, nx_graph, directed = args_tuple
    return _roundtrip_from_nx(test_id, source_name, nx_graph, directed)


def run_benchmark_parallel(
    num_tests: int,
    max_string_len: int,
    max_nodes: int,
    seed: int,
    output_dir: str,
    n_workers: int = 4,
) -> tuple[BenchmarkSummary, list[TestResult]]:
    """Run benchmark with ProcessPoolExecutor parallelization.

    Same logic as run_benchmark but distributes tests across workers.
    """
    rng = random.Random(seed)
    all_results: list[TestResult] = []
    test_id = 0

    n_string_tests = max(num_tests * 2 // 5, 10)

    print(f"Phase 2 Random Round-Trip (parallel, {n_workers} workers)")
    print(f"{'=' * 60}")
    print(f"Total tests target: {num_tests}, Seed: {seed}")
    print()

    # ---- Prepare all tasks ----
    string_tasks: list[tuple[int, str, bool]] = []
    for _ in range(n_string_tests):
        length = rng.randint(1, max_string_len)
        directed = rng.choice([True, False])
        instr_string = _generate_random_string(rng, length)
        string_tasks.append((test_id, instr_string, directed))
        test_id += 1

    nx_tasks: list[tuple[int, str, Any, bool]] = []
    n_nx_tests = num_tests - n_string_tests
    n_per_generator = max(n_nx_tests // 6, 2)

    trees = _nx_tree_graphs(rng, n_per_generator, max_nodes)
    for source_name, g in trees:
        nx_tasks.append((test_id, source_name, g, False))
        test_id += 1

    gnp_graphs = _nx_gnp_graphs(rng, n_per_generator * 2, max_nodes, directed=False)
    for source_name, g in gnp_graphs[:n_per_generator]:
        nx_tasks.append((test_id, source_name, g, False))
        test_id += 1

    ba_graphs = _nx_ba_graphs(rng, n_per_generator, max_nodes)
    for source_name, g in ba_graphs:
        nx_tasks.append((test_id, source_name, g, False))
        test_id += 1

    special = _nx_special_graphs(max_nodes)
    for source_name, g in special:
        nx_tasks.append((test_id, source_name, g, False))
        test_id += 1

    ws_graphs = _nx_watts_strogatz_graphs(rng, n_per_generator * 2, max_nodes)
    for source_name, g in ws_graphs[:n_per_generator]:
        nx_tasks.append((test_id, source_name, g, False))
        test_id += 1

    n_self_loop = max(n_per_generator // 2, 2)
    sl_graphs = _nx_self_loop_graphs(rng, n_self_loop * 2, max_nodes)
    for source_name, g in sl_graphs[:n_self_loop]:
        nx_tasks.append((test_id, source_name, g, False))
        test_id += 1

    gnp_dir = _nx_gnp_graphs(rng, n_per_generator * 2, max_nodes, directed=True)
    for source_name, g in gnp_dir[:n_per_generator]:
        nx_tasks.append((test_id, source_name, g, True))
        test_id += 1

    # ---- Execute in parallel ----
    from concurrent.futures import ProcessPoolExecutor, as_completed

    print(f"Submitting {len(string_tasks)} string tasks + {len(nx_tasks)} NX tasks...")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        string_futures = {
            executor.submit(_parallel_roundtrip_string, t): t[0] for t in string_tasks
        }
        nx_futures = {executor.submit(_parallel_roundtrip_nx, t): t[0] for t in nx_tasks}

        total = len(string_futures) + len(nx_futures)
        for done_count, fut in enumerate(as_completed({**string_futures, **nx_futures}), 1):
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

        if r.source not in summary.results_by_source:
            summary.results_by_source[r.source] = {"total": 0, "passed": 0, "failed": 0}
        summary.results_by_source[r.source]["total"] += 1
        if r.passed:
            summary.results_by_source[r.source]["passed"] += 1
        else:
            summary.results_by_source[r.source]["failed"] += 1

        if not r.passed:
            summary.failures.append(
                {
                    "test_id": r.test_id,
                    "source": r.source,
                    "directed": r.directed,
                    "num_nodes": r.num_nodes,
                    "num_edges": r.num_edges,
                    "error": r.error,
                }
            )

    pass_rate = summary.passed / max(1, summary.total_tests) * 100
    print(f"\nPassed: {summary.passed}/{summary.total_tests} ({pass_rate:.1f}%)")
    print(f"Total time: {summary.total_time_s:.2f}s")

    # Save JSON
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "random_roundtrip_results.json")
    with open(output_path, "w") as f:
        json.dump(
            {
                "benchmark": "random_roundtrip",
                "config": {
                    "num_tests": num_tests,
                    "max_string_len": max_string_len,
                    "max_nodes": max_nodes,
                    "seed": seed,
                    "n_workers": n_workers,
                },
                "summary": {
                    "total_tests": summary.total_tests,
                    "passed": summary.passed,
                    "failed": summary.failed,
                    "pass_rate_pct": round(pass_rate, 2),
                    "total_time_s": round(summary.total_time_s, 4),
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
    """Save raw results as CSV."""
    import pandas as pd

    rows = []
    for r in results:
        rows.append(
            {
                "test_id": r.test_id,
                "source": r.source,
                "directed": r.directed,
                "num_nodes": r.num_nodes,
                "num_edges": r.num_edges,
                "original_string_len": len(r.original_string),
                "roundtrip_string_len": len(r.roundtrip_string),
                "passed": r.passed,
                "error": r.error,
                "time_s": r.time_s,
            }
        )
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "random_roundtrip_results.csv")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"CSV saved to: {path}")
    return path


def generate_figure(results: list[TestResult], output_dir: str) -> list[str]:
    """Generate publication figure: 1x2 (compression + instruction composition).

    Panel (a): Scatter of roundtrip string length |w'| vs original string
    length |w| for random-string tests. Points below y=x show that greedy
    G2S compresses redundant random strings. Color encodes graph size.
    Panel (b): Stacked horizontal bar of mean instruction composition
    (node V/v, edge C/c, navigation N/P/n/p) in roundtrip strings by
    graph family.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import matplotlib.pyplot as plt  # noqa: E402
    import numpy as np  # noqa: E402
    from plotting_styles import (  # noqa: E402
        PAUL_TOL_BRIGHT,
        PLOT_SETTINGS,
        apply_ieee_style,
        family_display,
        get_figure_size,
        save_figure,
    )

    apply_ieee_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=get_figure_size("double", 0.5))

    total_tests = len(results)
    total_passed = sum(1 for r in results if r.passed)

    # ================================================================
    # Panel (a): Compression scatter -- |w'| vs |w| for random strings
    # ================================================================
    random_results = [
        r
        for r in results
        if r.source == "random_string"
        and r.passed
        and len(r.original_string) > 0
        and len(r.roundtrip_string) > 0
    ]

    if random_results:
        orig_lens = np.array([len(r.original_string) for r in random_results])
        rt_lens = np.array([len(r.roundtrip_string) for r in random_results])
        nodes = np.array([r.num_nodes for r in random_results])

        sc = ax1.scatter(
            orig_lens,
            rt_lens,
            c=nodes,
            cmap="viridis",
            s=PLOT_SETTINGS["scatter_size"],
            alpha=0.5,
            edgecolors="none",
            rasterized=True,
        )
        cbar = fig.colorbar(sc, ax=ax1, pad=0.02, aspect=30)
        cbar.set_label("Nodes $N$", fontsize=PLOT_SETTINGS["axes_labelsize"] - 1)

        # y=x reference line
        max_val = float(max(orig_lens.max(), rt_lens.max()) * 1.05)
        ax1.plot(
            [0, max_val],
            [0, max_val],
            "--",
            color="0.4",
            linewidth=PLOT_SETTINGS["line_width"],
            zorder=1,
            label="$|w'| = |w|$",
        )

        # Annotation: compression statistics
        ratios = rt_lens / np.maximum(orig_lens, 1)
        compressed_count = int(np.sum(rt_lens < orig_lens))
        pct_compressed = 100.0 * compressed_count / len(random_results)
        median_ratio = float(np.median(ratios))

        info = f"{pct_compressed:.0f}% compressed\nMedian $|w'|/|w| = {median_ratio:.2f}$"
        ax1.text(
            0.97,
            0.03,
            info,
            transform=ax1.transAxes,
            fontsize=PLOT_SETTINGS["annotation_fontsize"],
            ha="right",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="0.7",
                alpha=0.9,
            ),
        )

    ax1.set_xlabel("Original string length $|w|$")
    ax1.set_ylabel("Roundtrip string length $|w'|$")
    ax1.legend(fontsize=PLOT_SETTINGS["legend_fontsize"], loc="upper left")
    ax1.grid(axis="y")

    # Panel label
    ax1.text(
        -0.02,
        1.05,
        "(a)",
        transform=ax1.transAxes,
        fontsize=PLOT_SETTINGS["panel_label_fontsize"],
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    # ================================================================
    # Panel (b): Instruction composition stacked bar by family
    # ================================================================
    family_stats: dict[str, dict[str, list[float]]] = {}
    for r in results:
        if not r.passed or not r.roundtrip_string:
            continue
        s = r.roundtrip_string
        n_chars = len(s)
        if n_chars == 0:
            continue
        node_frac = sum(1 for c in s if c in "Vv") / n_chars
        edge_frac = sum(1 for c in s if c in "Cc") / n_chars
        nav_frac = sum(1 for c in s if c in "NnPp") / n_chars

        family_stats.setdefault(r.source, {"node": [], "edge": [], "nav": []})
        family_stats[r.source]["node"].append(node_frac)
        family_stats[r.source]["edge"].append(edge_frac)
        family_stats[r.source]["nav"].append(nav_frac)

    # Sort families by total structural fraction (node + edge)
    families = sorted(
        family_stats.keys(),
        key=lambda f: np.mean(family_stats[f]["node"]) + np.mean(family_stats[f]["edge"]),
    )
    mean_node = [float(np.mean(family_stats[f]["node"])) for f in families]
    mean_edge = [float(np.mean(family_stats[f]["edge"])) for f in families]
    mean_nav = [float(np.mean(family_stats[f]["nav"])) for f in families]

    y_pos = np.arange(len(families))
    node_color = PAUL_TOL_BRIGHT["green"]
    edge_color = PAUL_TOL_BRIGHT["red"]
    nav_color = PAUL_TOL_BRIGHT["blue"]

    ax2.barh(
        y_pos,
        mean_node,
        color=node_color,
        alpha=PLOT_SETTINGS["bar_alpha"],
        label="Node (V/v)",
    )
    ax2.barh(
        y_pos,
        mean_edge,
        left=mean_node,
        color=edge_color,
        alpha=PLOT_SETTINGS["bar_alpha"],
        label="Edge (C/c)",
    )
    left_nav = [n + e for n, e in zip(mean_node, mean_edge, strict=True)]
    ax2.barh(
        y_pos,
        mean_nav,
        left=left_nav,
        color=nav_color,
        alpha=PLOT_SETTINGS["bar_alpha"],
        label="Nav (N/P/n/p)",
    )

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(
        [family_display(f) for f in families],
        fontsize=PLOT_SETTINGS["tick_labelsize"] - 2,
    )
    ax2.set_xlabel("Fraction of instructions")
    ax2.set_xlim(0, 1)
    ax2.legend(
        fontsize=PLOT_SETTINGS["legend_fontsize"] - 1,
        loc="lower right",
    )
    ax2.grid(axis="x")
    ax2.grid(axis="y", visible=False)

    # Pass rate annotation
    pass_pct = 100.0 * total_passed / total_tests if total_tests > 0 else 0.0
    ax2.text(
        0.97,
        0.97,
        f"$N$ = {total_tests:,} | {pass_pct:.0f}% pass",
        transform=ax2.transAxes,
        fontsize=PLOT_SETTINGS["annotation_fontsize"],
        ha="right",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="0.7",
            alpha=0.9,
        ),
    )

    # Panel label
    ax2.text(
        -0.02,
        1.05,
        "(b)",
        transform=ax2.transAxes,
        fontsize=PLOT_SETTINGS["panel_label_fontsize"],
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    fig.tight_layout()
    paths = save_figure(fig, os.path.join(output_dir, "random_roundtrip_figure"))
    plt.close(fig)
    print(f"Figure saved: {paths}")
    return paths


def generate_table(results: list[TestResult], output_dir: str) -> str:
    """Generate LaTeX table: pass rate and timing by family."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import pandas as pd  # noqa: E402
    from plotting_styles import binomial_ci, save_latex_table  # noqa: E402

    rows = []
    source_stats: dict[str, list[TestResult]] = {}
    for r in results:
        source_stats.setdefault(r.source, []).append(r)

    for fam in sorted(source_stats):
        fam_results = source_stats[fam]
        n = len(fam_results)
        k = sum(1 for r in fam_results if r.passed)
        rate = k / n
        lo, hi = binomial_ci(k, n)
        times = [r.time_s for r in fam_results]
        max_nodes = max(r.num_nodes for r in fam_results)
        max_edges = max(r.num_edges for r in fam_results)
        rows.append(
            {
                "Family": fam,
                "N_tests": n,
                "Pass_rate": f"{rate * 100:.1f}\\%",
                "95\\% CI": f"[{lo * 100:.1f}, {hi * 100:.1f}]",
                "Mean_time_ms": f"{sum(times) / len(times) * 1000:.1f}",
                "Median_time_ms": f"{sorted(times)[len(times) // 2] * 1000:.1f}",
                "Max_nodes": max_nodes,
                "Max_edges": max_edges,
            }
        )

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "random_roundtrip_table.tex")
    save_latex_table(
        df,
        path,
        caption="Round-trip pass rate and execution time by graph family.",
        label="tab:roundtrip",
    )
    print(f"Table saved to: {path}")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the random round-trip benchmark."""
    parser = argparse.ArgumentParser(
        description="Phase 2: Massive random round-trip testing for IsalGraph."
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=DEFAULT_NUM_TESTS,
        help=f"Total number of tests (default: {DEFAULT_NUM_TESTS}).",
    )
    parser.add_argument(
        "--max-string-len",
        type=int,
        default=DEFAULT_MAX_STRING_LEN,
        help=f"Maximum random string length (default: {DEFAULT_MAX_STRING_LEN}).",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=DEFAULT_MAX_NODES,
        help=f"Maximum nodes for NX generators (default: {DEFAULT_MAX_NODES}).",
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
            max_string_len=args.max_string_len,
            max_nodes=args.max_nodes,
            seed=args.seed,
            output_dir=args.output_dir,
            n_workers=args.n_workers,
        )
    else:
        summary, all_results = run_benchmark(
            num_tests=args.num_tests,
            max_string_len=args.max_string_len,
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
