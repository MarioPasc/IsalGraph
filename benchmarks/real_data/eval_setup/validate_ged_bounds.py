"""Validation gate 2 -- cross-check GED bounds against exact A* on real graphs.

Runs the two bounds in :mod:`ged_bounds` against ``networkx`` exact GED on a
seeded sample of real benchmark pairs, under one cost model, and reports the
quantities the revision cites when it argues that the lower bound is the better
large-``n`` reference:

- bracket validity, ``LB <= exact <= UB``, which is a hard gate;
- the certification rate, the fraction of pairs where ``LB == UB``;
- Spearman correlation of each bound against exact GED;
- signed relative bias of each bound.

The same sample is what GEDLIB must reproduce. Run this first, record the
numbers, then run the GEDLIB side on Picasso and compare.

Requires ``torch`` for the GraphEdX loaders; use the ``isalsr`` environment.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ged_bounds import (  # noqa: E402
    UNIT_COSTS,
    EditCosts,
    bipartite_upper_bound,
    branch_lower_bound,
    exact_ged,
)

logger = logging.getLogger(__name__)

DEFAULT_SOURCE = (
    "/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/GED_PRECOMPUTED"
)


@dataclass(slots=True)
class GateReport:
    """Outcome of one validation run.

    Attributes
    ----------
    dataset : str
        Dataset the pairs were drawn from.
    n_pairs : int
        Number of pairs on which exact GED completed.
    n_violations : int
        Pairs where ``LB <= exact <= UB`` failed. Any non-zero value fails.
    certification_rate : float
        Fraction of pairs with ``LB == UB``.
    rho_exact_lb : float
        Spearman correlation of the lower bound against exact GED.
    rho_exact_ub : float
        Spearman correlation of the upper bound against exact GED.
    bias_lb : float
        Mean relative deviation of the lower bound from exact GED.
    bias_ub : float
        Mean relative deviation of the upper bound from exact GED.
    seconds : float
        Wall-clock duration.
    """

    dataset: str
    n_pairs: int
    n_violations: int
    certification_rate: float
    rho_exact_lb: float
    rho_exact_ub: float
    bias_lb: float
    bias_ub: float
    seconds: float


def load_pairs(
    dataset: str,
    source_dir: str,
    n_pairs: int,
    seed: int,
    max_nodes: int,
) -> list[tuple[nx.Graph, nx.Graph]]:
    """Draw a seeded sample of graph pairs from a GraphEdX dataset.

    Parameters
    ----------
    dataset : str
        Dataset directory name, e.g. ``'LINUX'``.
    source_dir : str
        Directory holding the GraphEdX ``.pt`` files.
    n_pairs : int
        Number of pairs to draw.
    seed : int
        Random seed.
    max_nodes : int
        Skip pairs where either graph exceeds this node count, so that exact
        GED remains tractable.

    Returns
    -------
    list of tuple
        The sampled pairs.
    """
    from graphedx_loader import load_graphedx_dataset

    result = load_graphedx_dataset(dataset, source_dir)
    graphs = [
        g
        for g in result.graphs
        if g.number_of_nodes() >= 2 and g.number_of_nodes() <= max_nodes and nx.is_connected(g)
    ]
    logger.info("%s: %d graphs usable (n <= %d, connected)", dataset, len(graphs), max_nodes)

    rng = random.Random(seed)
    pairs = []
    seen: set[tuple[int, int]] = set()
    while len(pairs) < n_pairs and len(seen) < len(graphs) * (len(graphs) - 1) // 2:
        i, j = rng.randrange(len(graphs)), rng.randrange(len(graphs))
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((graphs[key[0]], graphs[key[1]]))
    return pairs


def run_gate(
    pairs: list[tuple[nx.Graph, nx.Graph]],
    dataset: str,
    costs: EditCosts = UNIT_COSTS,
    timeout: float = 30.0,
) -> tuple[GateReport, list[dict]]:
    """Compute both bounds and exact GED over a sample and score the result.

    Parameters
    ----------
    pairs : list of tuple
        Graph pairs to evaluate.
    dataset : str
        Dataset name, recorded in the report.
    costs : EditCosts, optional
        Edit operation costs.
    timeout : float, optional
        Per-pair timeout for the exact solver.

    Returns
    -------
    tuple
        The report and the per-pair records, so that the same sample can be
        replayed against GEDLIB.
    """
    t0 = time.perf_counter()
    records: list[dict] = []
    for idx, (g1, g2) in enumerate(pairs):
        lb = branch_lower_bound(g1, g2, costs)
        ub = bipartite_upper_bound(g1, g2, costs)
        ex = exact_ged(g1, g2, costs, timeout=timeout)
        records.append(
            {
                "idx": idx,
                "n1": g1.number_of_nodes(),
                "n2": g2.number_of_nodes(),
                "m1": g1.number_of_edges(),
                "m2": g2.number_of_edges(),
                "lb": lb,
                "ub": ub,
                "exact": ex,
            }
        )
        if (idx + 1) % 50 == 0:
            logger.info("  %d/%d pairs", idx + 1, len(pairs))

    usable = [r for r in records if np.isfinite(r["exact"]) and r["exact"] > 0]
    lbs = np.array([r["lb"] for r in usable])
    ubs = np.array([r["ub"] for r in usable])
    exs = np.array([r["exact"] for r in usable])

    violations = int(((lbs > exs + 1e-9) | (ubs < exs - 1e-9)).sum())
    certified = float((np.abs(lbs - ubs) < 1e-9).mean()) if len(usable) else float("nan")

    report = GateReport(
        dataset=dataset,
        n_pairs=len(usable),
        n_violations=violations,
        certification_rate=certified,
        rho_exact_lb=float(spearmanr(exs, lbs).statistic),
        rho_exact_ub=float(spearmanr(exs, ubs).statistic),
        bias_lb=float(np.mean((lbs - exs) / exs)),
        bias_ub=float(np.mean((ubs - exs) / exs)),
        seconds=time.perf_counter() - t0,
    )
    return report, records


def main() -> int:
    """Run the gate from the command line.

    Returns
    -------
    int
        ``0`` if no bracket violation occurred, ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="LINUX")
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE)
    parser.add_argument("--n-pairs", type=int, default=400)
    parser.add_argument("--max-nodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out", default=None, help="Write per-pair records as JSON.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    pairs = load_pairs(args.dataset, args.source_dir, args.n_pairs, args.seed, args.max_nodes)
    logger.info("Sampled %d pairs from %s (seed %d)", len(pairs), args.dataset, args.seed)

    report, records = run_gate(pairs, args.dataset, timeout=args.timeout)

    logger.info("\n%s", json.dumps(asdict(report), indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({"report": asdict(report), "pairs": records}))
        logger.info("Per-pair records written to %s", args.out)

    if report.n_violations:
        logger.error("GATE FAILED: %d bracket violations", report.n_violations)
        return 1
    logger.info("GATE PASSED: 0 bracket violations on %d pairs", report.n_pairs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
