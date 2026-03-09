"""Neighborhood topology analysis: string-space isolation experiments.

Implements Experiments A, D, and E from the neighborhood topology
analysis document. These require the IsalGraph encoder/decoder
and operate on individual graphs.

Experiments:
    A — Isolation radius rho(G) for selected graphs
    D — Expanded neighborhood visualization (find first valid neighbors)
    E — Systematic sweep over all small non-isomorphic connected graphs

Dependencies:
    - isalgraph package (core encoder/decoder)
    - networkx
    - nauty/geng (for Experiment E graph enumeration, via networkx or subprocess)

Usage:
    python -m benchmarks.eval_neighborhood.isolation_radius \
        --experiment A --max-radius 5 --n-samples 50000
    python -m benchmarks.eval_neighborhood.isolation_radius \
        --experiment E --max-nodes 7
    python -m benchmarks.eval_neighborhood.isolation_radius \
        --experiment D --max-radius 4 --n-samples 100000

References:
    - McKay & Piperno (2014), J. Symbolic Comput. 60:94-112 (nauty).
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

# IsalGraph instruction alphabet
ALPHA = list("NnPpVvCcW")
K = len(ALPHA)  # 9


# ---------------------------------------------------------------------------
# IsalGraph interface (adapter pattern for testability)
# ---------------------------------------------------------------------------

class IsalGraphInterface:
    """Wrapper around IsalGraph encoder/decoder.

    Encapsulates all IsalGraph dependencies so the experimental logic
    is independent of the specific import paths. Adjust imports here
    if the package structure changes.
    """

    def __init__(self):
        """Initialise IsalGraph adapter and codec classes."""
        try:
            from isalgraph.adapters.networkx_adapter import NetworkXAdapter
            from isalgraph.core.graph_to_string import GraphToString
            from isalgraph.core.string_to_graph import StringToGraph
            self._adapter = NetworkXAdapter()
            self._g2s_cls = GraphToString
            self._s2g_cls = StringToGraph
            self._available = True
        except ImportError as e:
            logger.error(f"IsalGraph import failed: {e}")
            logger.error("Ensure isalgraph is installed and on PYTHONPATH.")
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def greedy_min(self, G: nx.Graph) -> Optional[str]:
        """Compute greedy-min canonical string for a NetworkX graph.

        Iterates over all starting nodes and returns the shortest string,
        breaking ties lexicographically.

        Args:
            G: NetworkX graph (undirected, connected).

        Returns:
            Greedy-min canonical string, or None on failure.
        """
        sg = self._adapter.from_external(G, directed=False)
        best = None
        for v in range(sg.node_count()):
            try:
                gts = self._g2s_cls(sg)
                s, _ = gts.run(initial_node=v)
            except Exception:
                continue
            if best is None or len(s) < len(best) or (len(s) == len(best) and s < best):
                best = s
        return best

    def decode(self, w: str) -> Optional[nx.Graph]:
        """Decode an IsalGraph string to a NetworkX graph.

        Args:
            w: IsalGraph instruction string.

        Returns:
            NetworkX graph if decoding succeeds, None otherwise.
        """
        try:
            s2g = self._s2g_cls(w, directed=False)
            sg, _ = s2g.run()
            G = self._adapter.to_external(sg)
            if G.number_of_nodes() == 0:
                return None
            return G
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Levenshtein perturbation generators
# ---------------------------------------------------------------------------

def lev1_perturbations(w: str) -> list[str]:
    """Generate all Levenshtein-1 perturbations of a string.

    Args:
        w: Input string over ALPHA.

    Returns:
        List of all strings at Levenshtein distance exactly 1 from w.
    """
    results = []
    m = len(w)

    # Substitutions: m * (K - 1) strings
    for i in range(m):
        for c in ALPHA:
            if c != w[i]:
                results.append(w[:i] + c + w[i + 1:])

    # Deletions: m strings
    for i in range(m):
        results.append(w[:i] + w[i + 1:])

    # Insertions: (m + 1) * K strings
    for i in range(m + 1):
        for c in ALPHA:
            results.append(w[:i] + c + w[i:])

    return results


def random_lev_r_perturbation(w: str, r: int, rng: random.Random) -> str:
    """Generate a random string at Levenshtein distance exactly r from w.

    Applies r sequential random Levenshtein operations. The resulting
    distance is at most r (could be less due to cancellations).

    Args:
        w: Input string.
        r: Number of operations to apply.
        rng: Random number generator.

    Returns:
        Perturbed string.
    """
    current = w
    for _ in range(r):
        m = len(current)
        op = rng.choice(["sub", "del", "ins"])

        if op == "sub" and m > 0:
            i = rng.randrange(m)
            c = rng.choice([c for c in ALPHA if c != current[i]])
            current = current[:i] + c + current[i + 1:]

        elif op == "del" and m > 1:  # don't delete to empty
            i = rng.randrange(m)
            current = current[:i] + current[i + 1:]

        elif op == "ins":
            i = rng.randrange(m + 1)
            c = rng.choice(ALPHA)
            current = current[:i] + c + current[i:]

    return current


def lev2_perturbations_sampled(
    w: str,
    n_samples: int,
    rng: random.Random,
) -> list[str]:
    """Sample strings at Levenshtein distance approximately 2 from w.

    For exact enumeration at r=2, the count is O(m^2 * K^2) which can
    be large. This function samples uniformly from 2-step perturbations.

    Args:
        w: Input string.
        n_samples: Number of samples.
        rng: Random number generator.

    Returns:
        List of sampled perturbations (may contain duplicates).
    """
    return [random_lev_r_perturbation(w, 2, rng) for _ in range(n_samples)]


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class IsolationResult:
    """Result for a single graph's isolation radius analysis."""
    graph_name: str
    n_nodes: int
    n_edges: int
    canonical_string: str
    string_length: int
    # Per-radius results
    radius_results: dict[int, dict] = field(default_factory=dict)
    # Final isolation radius (first r with valid neighbor)
    isolation_radius: Optional[int] = None
    # First valid neighbor details
    first_neighbor_string: Optional[str] = None
    first_neighbor_ged: Optional[int] = None


@dataclass
class SweepResult:
    """Summary of Experiment E systematic sweep."""
    n_nodes: int
    n_graphs: int
    n_with_radius_1: int       # graphs where rho(G) = 1
    n_with_radius_geq2: int    # graphs where rho(G) >= 2
    mean_string_length: float
    max_string_length: int
    # Details for any counterexample (rho = 1)
    counterexamples: list[dict] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Experiment A: Isolation radius for selected graphs
# ---------------------------------------------------------------------------

def experiment_a(
    isal: IsalGraphInterface,
    max_radius: int = 5,
    n_samples_per_radius: int = 50000,
    seed: int = 42,
) -> list[IsolationResult]:
    """Experiment A: Determine isolation radius for canonical graph examples.

    Tests Levenshtein-1 exhaustively, then uses Monte Carlo sampling
    for higher radii.

    Args:
        isal: IsalGraph interface.
        max_radius: Maximum radius to search.
        n_samples_per_radius: Monte Carlo samples per radius (r >= 2).
        seed: Random seed.

    Returns:
        List of IsolationResult, one per test graph.
    """
    rng = random.Random(seed)

    # Test graph set (extendable)
    test_graphs = [
        ("P3 (path)", nx.path_graph(3)),
        ("C3 (triangle)", nx.cycle_graph(3)),
        ("P4 (path)", nx.path_graph(4)),
        ("C4 (cycle)", nx.cycle_graph(4)),
        ("K4 (complete)", nx.complete_graph(4)),
        ("diamond", nx.diamond_graph()),
        ("bull", nx.bull_graph()),
        ("P5 (path)", nx.path_graph(5)),
        ("C5 (cycle)", nx.cycle_graph(5)),
        ("house", nx.house_graph()),
        ("K5 (complete)", nx.complete_graph(5)),
        ("petersen", nx.petersen_graph()),
        ("C6 (cycle)", nx.cycle_graph(6)),
        ("K3,3 (bipartite)", nx.complete_bipartite_graph(3, 3)),
        ("octahedron", nx.octahedral_graph()),
    ]

    results = []
    for name, G in test_graphs:
        logger.info(f"Processing {name} (n={G.number_of_nodes()}, m={G.number_of_edges()})")

        w = isal.greedy_min(G)
        if w is None:
            logger.warning(f"  Failed to encode {name}, skipping")
            continue

        result = IsolationResult(
            graph_name=name,
            n_nodes=G.number_of_nodes(),
            n_edges=G.number_of_edges(),
            canonical_string=w,
            string_length=len(w),
        )

        found = False
        for r in range(1, max_radius + 1):
            t0 = time.time()

            if r == 1:
                # Exhaustive enumeration
                perturbations = lev1_perturbations(w)
                n_tested = len(perturbations)
            else:
                # Monte Carlo sampling
                perturbations = [
                    random_lev_r_perturbation(w, r, rng)
                    for _ in range(n_samples_per_radius)
                ]
                n_tested = n_samples_per_radius

            n_valid = 0
            n_connected = 0
            n_connected_noniso = 0
            first_valid_neighbor = None

            for wp in perturbations:
                Gp = isal.decode(wp)
                if Gp is None:
                    continue
                n_valid += 1
                if nx.is_connected(Gp):
                    n_connected += 1
                    if not nx.is_isomorphic(Gp, G):
                        n_connected_noniso += 1
                        if first_valid_neighbor is None:
                            first_valid_neighbor = (wp, Gp)

            elapsed = time.time() - t0

            result.radius_results[r] = {
                "n_tested": n_tested,
                "n_valid_decodes": n_valid,
                "n_connected": n_connected,
                "n_connected_noniso": n_connected_noniso,
                "method": "exhaustive" if r == 1 else f"MC({n_samples_per_radius})",
                "elapsed_seconds": round(elapsed, 2),
            }

            logger.info(
                f"  r={r}: tested={n_tested}, valid={n_valid}, "
                f"connected_noniso={n_connected_noniso} ({elapsed:.1f}s)"
            )

            if n_connected_noniso > 0 and not found:
                result.isolation_radius = r
                wp_first, Gp_first = first_valid_neighbor
                result.first_neighbor_string = wp_first
                # Compute GED to original
                try:
                    ged = nx.graph_edit_distance(
                        G, Gp_first,
                        node_ins_cost=lambda _: 1,
                        node_del_cost=lambda _: 1,
                        edge_ins_cost=lambda _: 1,
                        edge_del_cost=lambda _: 1,
                        node_subst_cost=lambda _, __: 0,
                    )
                    result.first_neighbor_ged = int(ged)
                except Exception:
                    result.first_neighbor_ged = None
                found = True
                # Continue to gather statistics at higher radii

        if not found:
            result.isolation_radius = None
            logger.info(f"  No valid neighbor found up to r={max_radius}")

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Experiment E: Systematic sweep over all small graphs
# ---------------------------------------------------------------------------

def enumerate_connected_graphs(n: int) -> list[nx.Graph]:
    """Enumerate all non-isomorphic connected graphs on n nodes.

    Uses networkx's graph_atlas for small n, which contains all graphs
    up to 7 nodes. Filters for connected graphs and deduplicates by
    isomorphism.

    Args:
        n: Number of nodes.

    Returns:
        List of non-isomorphic connected NetworkX graphs.
    """
    if n < 1 or n > 7:
        raise ValueError(f"graph_atlas supports n <= 7, got n={n}")

    # graph_atlas_g() returns all graphs up to 7 nodes, indexed
    connected = []
    seen_hashes = set()

    for G in nx.graph_atlas_g():
        if G.number_of_nodes() != n:
            continue
        if not nx.is_connected(G):
            continue
        # Deduplicate by graph6 canonical form
        g6 = nx.to_graph6_bytes(G, header=False).strip()
        if g6 not in seen_hashes:
            seen_hashes.add(g6)
            connected.append(G.copy())

    return connected


def experiment_e(
    isal: IsalGraphInterface,
    max_nodes: int = 7,
) -> list[SweepResult]:
    """Experiment E: Sweep over all small non-isomorphic connected graphs.

    For each graph, tests whether any Levenshtein-1 perturbation of its
    greedy-min string produces a valid graph. Reports the fraction of
    graphs with isolation radius exactly 1 (counterexamples to universality).

    Args:
        isal: IsalGraph interface.
        max_nodes: Maximum node count (up to 7 for graph_atlas).

    Returns:
        List of SweepResult, one per node count.
    """
    results = []

    for n in range(3, max_nodes + 1):
        graphs = enumerate_connected_graphs(n)
        logger.info(f"n={n}: {len(graphs)} non-isomorphic connected graphs")

        n_radius_1 = 0
        n_radius_geq2 = 0
        string_lengths = []
        counterexamples = []

        for idx, G in enumerate(graphs):
            w = isal.greedy_min(G)
            if w is None:
                logger.warning(f"  Failed to encode graph {idx} (n={n})")
                continue

            string_lengths.append(len(w))

            # Test all Levenshtein-1 perturbations
            found_valid = False
            for wp in lev1_perturbations(w):
                Gp = isal.decode(wp)
                if Gp is not None and Gp.number_of_nodes() > 0:
                    if nx.is_connected(Gp) and not nx.is_isomorphic(Gp, G):
                        found_valid = True
                        break

            if found_valid:
                n_radius_1 += 1
                counterexamples.append({
                    "n_nodes": n,
                    "n_edges": G.number_of_edges(),
                    "string": w,
                    "string_length": len(w),
                    "graph6": nx.to_graph6_bytes(G, header=False).strip().decode(),
                    "neighbor_string": wp,
                })
                logger.info(f"  COUNTEREXAMPLE: graph {idx} (m={G.number_of_edges()}) "
                            f"w={w} → valid neighbor {wp}")
            else:
                n_radius_geq2 += 1

            if (idx + 1) % 100 == 0:
                logger.info(f"  Processed {idx+1}/{len(graphs)}")

        sweep = SweepResult(
            n_nodes=n,
            n_graphs=len(graphs),
            n_with_radius_1=n_radius_1,
            n_with_radius_geq2=n_radius_geq2,
            mean_string_length=round(float(np.mean(string_lengths)), 1) if string_lengths else 0.0,
            max_string_length=max(string_lengths) if string_lengths else 0,
            counterexamples=counterexamples,
        )

        pct = 100.0 * n_radius_geq2 / len(graphs) if graphs else 0
        logger.info(
            f"n={n}: {n_radius_geq2}/{len(graphs)} ({pct:.1f}%) have "
            f"isolation radius >= 2, {n_radius_1} counterexamples"
        )
        results.append(sweep)

    return results


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="IsalGraph string-space isolation experiments (A, D, E)"
    )
    parser.add_argument("--experiment", choices=["A", "D", "E", "all"],
                        default="all", help="Which experiment to run")
    parser.add_argument("--max-radius", type=int, default=5,
                        help="Maximum Levenshtein radius to search (Exp A, D)")
    parser.add_argument("--n-samples", type=int, default=50000,
                        help="Monte Carlo samples per radius (Exp A, D)")
    parser.add_argument("--max-nodes", type=int, default=7,
                        help="Maximum node count for graph enumeration (Exp E)")
    parser.add_argument("--output-dir", type=str, default="results/eval_neighborhood",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    isal = IsalGraphInterface()
    if not isal.available:
        logger.error("IsalGraph not available. Aborting.")
        return

    run_a = args.experiment in ("A", "D", "all")
    run_e = args.experiment in ("E", "all")

    # Experiment A (also covers D, since D extends A's results)
    if run_a:
        logger.info("=" * 60)
        logger.info("EXPERIMENT A/D: Isolation radius characterisation")
        logger.info("=" * 60)

        results_a = experiment_a(
            isal,
            max_radius=args.max_radius,
            n_samples_per_radius=args.n_samples,
            seed=args.seed,
        )

        # Save results
        out_a = output_dir / "isolation_radius.json"
        with open(out_a, "w") as f:
            json.dump([asdict(r) for r in results_a], f, indent=2)
        logger.info(f"Experiment A results → {out_a}")

        # Print summary
        print("\n" + "=" * 80)
        print("ISOLATION RADIUS SUMMARY")
        print("=" * 80)
        print(f"{'Graph':<25} {'|V|':>3} {'|E|':>3} {'|w|':>4} {'rho(G)':>7} "
              f"{'1st neighbor GED':>17}")
        print("-" * 80)
        for r in results_a:
            rho_str = str(r.isolation_radius) if r.isolation_radius else f">{args.max_radius}"
            ged_str = str(r.first_neighbor_ged) if r.first_neighbor_ged is not None else "—"
            print(f"{r.graph_name:<25} {r.n_nodes:>3} {r.n_edges:>3} "
                  f"{r.string_length:>4} {rho_str:>7} {ged_str:>17}")

    # Experiment E
    if run_e:
        logger.info("\n" + "=" * 60)
        logger.info("EXPERIMENT E: Systematic sweep over small graphs")
        logger.info("=" * 60)

        results_e = experiment_e(
            isal,
            max_nodes=args.max_nodes,
        )

        out_e = output_dir / "systematic_sweep.json"
        with open(out_e, "w") as f:
            json.dump([asdict(r) for r in results_e], f, indent=2)
        logger.info(f"Experiment E results → {out_e}")

        # Print summary
        print("\n" + "=" * 80)
        print("SYSTEMATIC SWEEP SUMMARY")
        print("=" * 80)
        print(f"{'n':>3} {'#graphs':>8} {'rho>=2':>8} {'rho=1':>6} "
              f"{'%isolated':>10} {'mean|w|':>8} {'max|w|':>7}")
        print("-" * 80)
        for r in results_e:
            pct = 100.0 * r.n_with_radius_geq2 / r.n_graphs if r.n_graphs else 0
            print(f"{r.n_nodes:>3} {r.n_graphs:>8} {r.n_with_radius_geq2:>8} "
                  f"{r.n_with_radius_1:>6} {pct:>9.1f}% "
                  f"{r.mean_string_length:>8.1f} {r.max_string_length:>7}")

        if any(r.counterexamples for r in results_e):
            print("\nCOUNTEREXAMPLES (graphs with rho = 1):")
            for r in results_e:
                for ce in r.counterexamples:
                    print(f"  n={ce['n_nodes']}, m={ce['n_edges']}, "
                          f"w={ce['string']} → neighbor={ce['neighbor_string']}")


if __name__ == "__main__":
    main()
