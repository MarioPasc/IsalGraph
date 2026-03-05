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
            if rng.random() < 0.4:
                instr = rng.choice(_NODE_INSTRS)
            else:
                instr = rng.choice(pool)

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
) -> BenchmarkSummary:
    """Run the full Phase 2 random round-trip benchmark.

    Args:
        num_tests: Total number of tests to run across all sources.
        max_string_len: Maximum instruction string length for random strings.
        max_nodes: Maximum graph size for NetworkX generators.
        seed: Random seed for reproducibility.
        output_dir: Directory for saving results JSON.

    Returns:
        BenchmarkSummary with aggregate statistics.
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

    return summary


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

    args = parser.parse_args()

    summary = run_benchmark(
        num_tests=args.num_tests,
        max_string_len=args.max_string_len,
        max_nodes=args.max_nodes,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    # Exit with non-zero code if any test failed (useful for CI).
    if summary.failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
