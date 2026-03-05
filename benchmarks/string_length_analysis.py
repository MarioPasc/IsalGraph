"""Empirical analysis of IsalGraph string length vs graph size and density.

Tests the central compactness claim of the preprint (Lopez-Rubio 2025,
Section 2.2). The preprint derives (Eq. 9) for the matrix-pointer algorithm:

    E[|I_G|] ~ (sqrt(pi) / (2*sqrt(2))) * N^2 * sqrt(rho)

where N = node count and rho = |E| / N^2 is edge density. For large sparse
graphs, |I_G| << N^2, achieving significant compression over the binary
adjacency representation.

Our CDLL-based algorithm uses a different instruction set ({N,n,P,p,V,v,C,c,W})
and may exhibit different scaling. This benchmark empirically characterizes:

1. **String length vs N** for fixed graph families.
2. **String length vs edge count** across families.
3. **Compression ratio**: |w| / N^2.
4. **Greedy vs canonical** string length for small graphs.

Usage:
    python benchmarks/string_length_analysis.py --seed 42
    python benchmarks/string_length_analysis.py --seed 42 --no-canonical --max-nodes 200
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

from isalgraph.adapters.networkx_adapter import NetworkXAdapter
from isalgraph.core.graph_to_string import GraphToString
from isalgraph.core.sparse_graph import SparseGraph

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = "/media/mpascual/Sandisk2TB/research/isalgraph"
DEFAULT_SEED = 42
# Graphs up to this size try all starting nodes; above, sample.
GREEDY_ALL_STARTS_LIMIT = 50
# Canonical string computation limit (exhaustive search is expensive).
CANONICAL_LIMIT = 8

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class StringLengthRecord:
    """One data point: a single graph and its string length."""

    family: str
    num_nodes: int
    num_edges: int
    density: float
    binary_length: int  # N*(N-1)/2 for undirected, N^2 for directed
    greedy_length_best: int  # min over starting nodes
    greedy_length_node0: int  # from node 0 only
    canonical_length: int  # -1 if not computed
    compression_ratio: float  # greedy_length_best / binary_length
    theoretical_prediction: float  # Eq. 9 (reference, not our algorithm)
    directed: bool
    time_s: float


@dataclass
class AnalysisSummary:
    """Aggregate results."""

    total_graphs: int = 0
    records: list[dict[str, Any]] = field(default_factory=list)
    total_time_s: float = 0.0
    errors: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# String length computation
# ---------------------------------------------------------------------------


def _greedy_string_length(sg: SparseGraph, start_node: int) -> int:
    """Compute greedy G2S string length from a single starting node.

    Returns -1 if the starting node cannot reach all other nodes.
    """
    try:
        g2s = GraphToString(sg)
        w, _ = g2s.run(initial_node=start_node)
        return len(w)
    except ValueError:
        return -1


def _greedy_min_length(sg: SparseGraph, max_starts: int | None = None) -> tuple[int, int]:
    """Compute min greedy string length over starting nodes.

    Returns:
        (min_length, length_from_node_0).
        min_length is -1 if no starting node succeeds.
    """
    n = sg.node_count()
    if n <= 1:
        return 0, 0

    len_0 = _greedy_string_length(sg, 0)

    if max_starts is None or max_starts >= n:
        starts = list(range(n))
    else:
        rng = random.Random(42)
        others = rng.sample(range(1, n), min(max_starts - 1, n - 1))
        starts = [0] + others

    best = -1
    for v in starts:
        length = len_0 if v == 0 else _greedy_string_length(sg, v)
        if length >= 0 and (best < 0 or length < best):
            best = length

    return best, len_0 if len_0 >= 0 else -1


def _theoretical_eq9(n: int, rho: float) -> float:
    """Preprint Eq. 9: E[|I_G|] ~ (sqrt(pi) / (2*sqrt(2))) * N^2 * sqrt(rho).

    This is for the matrix-pointer algorithm, not our CDLL-based one.
    Serves as a reference point.
    """
    if n <= 1 or rho <= 0:
        return 0.0
    return (math.sqrt(math.pi) / (2.0 * math.sqrt(2.0))) * n * n * math.sqrt(rho)


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------


def _generate_graph_suite(
    rng: random.Random, max_nodes: int = 100
) -> list[tuple[str, nx.Graph, bool]]:
    """Generate a diverse suite of graphs for string length analysis.

    Returns list of (family_name, nx_graph, directed).
    """
    graphs: list[tuple[str, nx.Graph, bool]] = []

    node_sizes_sparse = [n for n in [4, 6, 8, 10, 15, 20, 30, 50, 75, 100] if n <= max_nodes]
    node_sizes_dense = [n for n in [3, 4, 5, 6, 7, 8, 10, 12, 15, 20] if n <= max_nodes]

    # ---- Trees (minimum connected, density ~ 2/N) ----
    for n in node_sizes_sparse:
        seed = rng.randint(0, 2**31)
        graphs.append(("tree", nx.random_labeled_tree(n, seed=seed), False))

    # ---- Paths (linear chain) ----
    for n in node_sizes_sparse:
        graphs.append(("path", nx.path_graph(n), False))

    # ---- Stars (hub-and-spoke, same edge count as trees) ----
    for n in node_sizes_sparse:
        graphs.append(("star", nx.star_graph(n - 1), False))

    # ---- Cycles ----
    for n in node_sizes_sparse:
        graphs.append(("cycle", nx.cycle_graph(n), False))

    # ---- Complete graphs (density = 1) ----
    for n in node_sizes_dense:
        graphs.append(("complete", nx.complete_graph(n), False))

    # ---- Barabasi-Albert (scale-free, controlled density via m) ----
    for n in [n for n in node_sizes_sparse if n >= 10]:
        for m in [1, 2, 3]:
            seed = rng.randint(0, 2**31)
            graphs.append((f"ba_m{m}", nx.barabasi_albert_graph(n, m, seed=seed), False))

    # ---- Erdos-Renyi at controlled densities ----
    for n in [n for n in [10, 15, 20, 30, 50] if n <= max_nodes]:
        for p in [0.1, 0.2, 0.3, 0.5, 0.7]:
            for _attempt in range(20):
                seed = rng.randint(0, 2**31)
                g = nx.gnp_random_graph(n, p, seed=seed)
                if nx.is_connected(g):
                    graphs.append((f"gnp_p{p:.1f}", g, False))
                    break

    # ---- Watts-Strogatz small-world ----
    for n in [n for n in [10, 20, 30, 50] if n <= max_nodes]:
        for k in [4, 6]:
            if k >= n:
                continue
            seed = rng.randint(0, 2**31)
            g = nx.watts_strogatz_graph(n, k, 0.3, seed=seed)
            if nx.is_connected(g):
                graphs.append((f"ws_k{k}", g, False))

    # ---- Grids (regular, sparse, 2D lattice) ----
    for m in [3, 4, 5, 6, 7, 8]:
        if m * m > max_nodes:
            break
        g = nx.convert_node_labels_to_integers(nx.grid_2d_graph(m, m))
        graphs.append(("grid", g, False))

    # ---- Ladder ----
    for n in [n for n in [4, 6, 8, 10, 15, 20] if 2 * n <= max_nodes]:
        graphs.append(("ladder", nx.ladder_graph(n), False))

    # ---- Special graphs ----
    graphs.append(("petersen", nx.petersen_graph(), False))
    for n in [n for n in [5, 8, 10, 15, 20] if n <= max_nodes]:
        graphs.append(("wheel", nx.wheel_graph(n), False))

    return graphs


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def _analyze_graph(
    family: str,
    nx_graph: nx.Graph,
    directed: bool,
    compute_canonical: bool,
) -> StringLengthRecord:
    """Analyze string length for a single graph."""
    adapter = NetworkXAdapter()
    sg = adapter.from_external(nx_graph, directed=directed)

    n = sg.node_count()
    m = sg.logical_edge_count()
    binary_len = n * n if directed else n * (n - 1) // 2
    density = m / binary_len if binary_len > 0 else 0.0

    t0 = time.perf_counter()

    max_starts = None if n <= GREEDY_ALL_STARTS_LIMIT else 20
    greedy_best, greedy_0 = _greedy_min_length(sg, max_starts=max_starts)

    canonical_len = -1
    if compute_canonical and n <= CANONICAL_LIMIT:
        import contextlib

        from isalgraph.core.canonical import canonical_string

        with contextlib.suppress(ValueError, RuntimeError):
            canonical_len = len(canonical_string(sg))

    elapsed = time.perf_counter() - t0
    theo = _theoretical_eq9(n, density)
    compression = greedy_best / binary_len if binary_len > 0 and greedy_best >= 0 else -1.0

    return StringLengthRecord(
        family=family,
        num_nodes=n,
        num_edges=m,
        density=round(density, 6),
        binary_length=binary_len,
        greedy_length_best=greedy_best,
        greedy_length_node0=greedy_0,
        canonical_length=canonical_len,
        compression_ratio=round(compression, 6),
        theoretical_prediction=round(theo, 2),
        directed=directed,
        time_s=round(elapsed, 4),
    )


def run_analysis(
    seed: int,
    output_dir: str,
    max_nodes: int = 100,
    compute_canonical: bool = True,
) -> AnalysisSummary:
    """Run the full string length analysis."""
    rng = random.Random(seed)
    summary = AnalysisSummary()

    print("String Length Analysis Benchmark")
    print(f"{'=' * 60}")
    print(f"Seed: {seed}")
    print(f"Max nodes: {max_nodes}")
    print(f"Canonical: {'yes (N <= ' + str(CANONICAL_LIMIT) + ')' if compute_canonical else 'no'}")
    print(f"Output dir: {output_dir}")
    print()

    graphs = _generate_graph_suite(rng, max_nodes=max_nodes)
    print(f"Generated {len(graphs)} graphs.\n")

    for i, (family, nx_g, directed) in enumerate(graphs):
        try:
            rec = _analyze_graph(family, nx_g, directed, compute_canonical)
            summary.records.append(asdict(rec))
        except Exception as exc:
            summary.errors.append(
                {
                    "index": i,
                    "family": family,
                    "num_nodes": nx_g.number_of_nodes(),
                    "error": str(exc),
                }
            )

        if (i + 1) % max(1, len(graphs) // 20) == 0:
            print(f"  ... {i + 1}/{len(graphs)} done")

    summary.total_graphs = len(summary.records)
    summary.total_time_s = sum(r["time_s"] for r in summary.records)

    # ---- Print results ----
    print(f"\n{'=' * 80}")
    print(
        f"{'Family':<20} {'N':>4} {'M':>5} {'rho':>6} "
        f"{'|w|':>6} {'|w0|':>6} {'|can|':>6} {'N^2':>6} {'ratio':>7} {'Eq9':>7}"
    )
    print("-" * 80)

    for rec in summary.records:
        can_s = f"{rec['canonical_length']:>6}" if rec["canonical_length"] >= 0 else "     -"
        print(
            f"{rec['family']:<20} {rec['num_nodes']:>4} {rec['num_edges']:>5} "
            f"{rec['density']:>6.3f} "
            f"{rec['greedy_length_best']:>6} {rec['greedy_length_node0']:>6} "
            f"{can_s} "
            f"{rec['binary_length']:>6} "
            f"{rec['compression_ratio']:>7.3f} "
            f"{rec['theoretical_prediction']:>7.1f}"
        )

    # ---- Canonical vs Greedy ----
    can_recs = [r for r in summary.records if r["canonical_length"] >= 0]
    if can_recs:
        print(f"\n{'=' * 60}")
        print(f"CANONICAL vs GREEDY (N <= {CANONICAL_LIMIT})")
        print(f"{'=' * 60}")
        total_saved = 0
        for r in can_recs:
            saved = r["greedy_length_best"] - r["canonical_length"]
            total_saved += saved
            print(
                f"  {r['family']:<15} N={r['num_nodes']:>2}  "
                f"canonical={r['canonical_length']:>3}  greedy={r['greedy_length_best']:>3}  "
                f"saved={saved}"
            )
        print(f"  Total chars saved by canonical: {total_saved}")

    # ---- Summary stats ----
    ratios = [r["compression_ratio"] for r in summary.records if r["compression_ratio"] >= 0]
    print(f"\n{'=' * 60}")
    print(
        f"Total graphs: {summary.total_graphs}, Time: {summary.total_time_s:.2f}s, "
        f"Errors: {len(summary.errors)}"
    )
    if ratios:
        print(
            f"Compression ratio |w|/N^2: "
            f"mean={sum(ratios) / len(ratios):.4f}, "
            f"min={min(ratios):.4f}, max={max(ratios):.4f}"
        )

    # ---- Save JSON ----
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "string_length_analysis.json")
    with open(output_path, "w") as f:
        json.dump(
            {
                "benchmark": "string_length_analysis",
                "config": {
                    "seed": seed,
                    "max_nodes": max_nodes,
                    "compute_canonical": compute_canonical,
                },
                "summary": {
                    "total_graphs": summary.total_graphs,
                    "total_time_s": round(summary.total_time_s, 4),
                    "errors": len(summary.errors),
                },
                "records": summary.records,
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
        description="Empirical analysis of IsalGraph string length vs graph size/density."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-nodes", type=int, default=100)
    parser.add_argument("--no-canonical", action="store_true")
    args = parser.parse_args()

    summary = run_analysis(
        seed=args.seed,
        output_dir=args.output_dir,
        max_nodes=args.max_nodes,
        compute_canonical=not args.no_canonical,
    )
    if summary.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
