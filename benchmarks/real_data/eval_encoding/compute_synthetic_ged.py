# ruff: noqa: E402, N803, N806
"""Compute GED on synthetic random graphs for fair comparison with encoding times.

Uses the **exact same** graph families, seeds, and instances as
``eval_encoding.py`` so that the GED timing curve in
``fig_empirical_complexity`` is directly comparable to the IsalGraph
encoding curves.

Measurement methodology follows Weber et al. (2019):
- CPU time via ``time.process_time()``
- Adaptive repetitions based on probe time
- Reports median and IQR

Intermediate results are saved after each ``(family, n)`` combination
to allow resuming and incremental re-generation.

Usage:
    python -m benchmarks.real_data.eval_encoding.compute_synthetic_ged \
        --output-dir /path/to/results \
        --n-instances 5 \
        --seed 42
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import time
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from benchmarks.real_data.eval_encoding.synthetic_generator import (
    FAMILY_CONFIGS,
    generate_graph_family,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Only stochastic families (same as fig_empirical_complexity RANDOM_FAMILIES)
RANDOM_FAMILIES = ["ba_m1", "ba_m2", "gnp_03", "gnp_05"]

# Conservative GED feasibility limits per family (GED is NP-hard).
# These are upper bounds; actual limit is enforced by timeout.
GED_N_LIMITS: dict[str, int] = {
    "ba_m1": 20,  # sparse (tree-like), GED cheaper
    "ba_m2": 16,  # moderate density
    "gnp_03": 16,  # moderate density
    "gnp_05": 13,  # dense, GED expensive
}

# Uniform GED cost functions (topology-only, no node/edge labels)
_NODE_SUBST_COST = lambda _n1, _n2: 0  # noqa: E731
_NODE_DEL_COST = lambda _n: 1  # noqa: E731
_NODE_INS_COST = lambda _n: 1  # noqa: E731
_EDGE_SUBST_COST = lambda _e1, _e2: 0  # noqa: E731
_EDGE_DEL_COST = lambda _e: 1  # noqa: E731
_EDGE_INS_COST = lambda _e: 1  # noqa: E731

# Timing parameters
GED_CALL_TIMEOUT = 60.0  # seconds per nx.graph_edit_distance call
PAIR_BUDGET_S = 120.0  # max total time per pair (probe + reps)
DEFAULT_N_REPS = 25  # target reps (adapted down for slow pairs)


# ---------------------------------------------------------------------------
# GED timing
# ---------------------------------------------------------------------------


def _compute_ged(g1: nx.Graph, g2: nx.Graph) -> float:
    """Compute exact GED with uniform costs and timeout."""
    result = nx.graph_edit_distance(
        g1,
        g2,
        node_subst_cost=_NODE_SUBST_COST,
        node_del_cost=_NODE_DEL_COST,
        node_ins_cost=_NODE_INS_COST,
        edge_subst_cost=_EDGE_SUBST_COST,
        edge_del_cost=_EDGE_DEL_COST,
        edge_ins_cost=_EDGE_INS_COST,
        timeout=GED_CALL_TIMEOUT,
    )
    return float(result) if result is not None else float("inf")


def _adaptive_reps(t_probe: float, n_reps: int) -> int:
    """Choose number of timed repetitions based on probe time.

    Follows the same adaptive scheme as eval_computational.py.
    """
    if t_probe > 0.9 * GED_CALL_TIMEOUT:
        return 0  # probe only
    if t_probe > 10.0:
        return 1
    if t_probe > 1.0:
        return 3
    if t_probe > 0.1:
        return min(10, n_reps)
    return n_reps


def _time_ged_pair(
    g1: nx.Graph,
    g2: nx.Graph,
    n_reps: int = DEFAULT_N_REPS,
) -> dict[str, Any]:
    """Time GED computation for a single pair with adaptive repetitions.

    Returns:
        Dict with ged_value, ged_time_median_s, ged_time_iqr_s,
        ged_n_reps, ged_times_all_s. Returns None values on timeout.
    """
    # --- Probe (untimed for reps, but timed for adaptation) ---
    t0 = time.process_time()
    ged_val = _compute_ged(g1, g2)
    t_probe = time.process_time() - t0

    if not np.isfinite(ged_val):
        return {
            "ged_value": float("inf"),
            "ged_time_median_s": float("inf"),
            "ged_time_iqr_s": float("nan"),
            "ged_n_reps": 0,
            "ged_times_all_s": [],
        }

    actual_reps = _adaptive_reps(t_probe, n_reps)

    # Cap by pair budget
    if actual_reps > 0 and t_probe > 0:
        budget_reps = max(1, int(PAIR_BUDGET_S / t_probe))
        actual_reps = min(actual_reps, budget_reps)

    if actual_reps == 0:
        # Probe only
        return {
            "ged_value": ged_val,
            "ged_time_median_s": t_probe,
            "ged_time_iqr_s": 0.0,
            "ged_n_reps": 1,
            "ged_times_all_s": [t_probe],
        }

    # --- Timed repetitions ---
    times: list[float] = []
    for _ in range(actual_reps):
        t0 = time.process_time()
        _compute_ged(g1, g2)
        t1 = time.process_time()
        times.append(t1 - t0)

    times_arr = np.array(times)
    return {
        "ged_value": ged_val,
        "ged_time_median_s": float(np.median(times_arr)),
        "ged_time_iqr_s": float(np.percentile(times_arr, 75) - np.percentile(times_arr, 25)),
        "ged_n_reps": actual_reps,
        "ged_times_all_s": times,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def compute_synthetic_ged_times(
    output_dir: str,
    n_instances: int = 5,
    seed: int = 42,
    n_reps: int = DEFAULT_N_REPS,
) -> str:
    """Compute GED times on synthetic random graphs and save to CSV.

    For each stochastic family and node count, generates ``n_instances``
    graphs (using the same seeds as eval_encoding.py) and times GED for
    all unique pairs.

    Saves intermediate results after each ``(family, n)`` combination.

    Args:
        output_dir: Directory to save output CSV.
        n_instances: Number of graph instances per (family, n).
        seed: Base random seed (must match eval_encoding.py).
        n_reps: Target repetitions per pair.

    Returns:
        Path to the saved CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "synthetic_ged_times.csv")

    # Load existing results for resuming
    existing_keys: set[tuple[str, int, int, int]] = set()
    rows: list[dict[str, Any]] = []
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        rows = df_existing.to_dict("records")
        for r in rows:
            existing_keys.add((r["family"], r["n_nodes"], r["instance_i"], r["instance_j"]))
        logger.info("Resuming: loaded %d existing rows from %s", len(rows), csv_path)

    for family in RANDOM_FAMILIES:
        n_max = GED_N_LIMITS.get(family, 12)
        n_min = FAMILY_CONFIGS[family]["n_min"]
        timed_out_at_n = False

        for n in range(n_min, n_max + 1):
            if timed_out_at_n:
                break

            # Generate instances (same seeds as eval_encoding.py)
            graphs: list[nx.Graph] = []
            for inst in range(n_instances):
                G = generate_graph_family(family, n, seed=seed, instance=inst)
                graphs.append(G)

            # All unique pairs
            pairs = list(itertools.combinations(range(n_instances), 2))
            n_done = 0
            n_timeout = 0

            for inst_i, inst_j in pairs:
                if (family, n, inst_i, inst_j) in existing_keys:
                    n_done += 1
                    continue

                g1, g2 = graphs[inst_i], graphs[inst_j]
                logger.info(
                    "  %s n=%d pair (%d,%d) [%d nodes, %d/%d edges]",
                    family,
                    n,
                    inst_i,
                    inst_j,
                    g1.number_of_nodes(),
                    g1.number_of_edges(),
                    g2.number_of_edges(),
                )

                result = _time_ged_pair(g1, g2, n_reps=n_reps)

                # Detect timeout: inf OR near-ceiling (nx.graph_edit_distance
                # returns best upper bound on timeout, not inf)
                timeout_ceil = 0.9 * GED_CALL_TIMEOUT
                is_timeout = (
                    not np.isfinite(result["ged_time_median_s"])
                    or result["ged_time_median_s"] >= timeout_ceil
                )
                if is_timeout:
                    n_timeout += 1
                    logger.warning(
                        "  TIMEOUT: %s n=%d pair (%d,%d)",
                        family,
                        n,
                        inst_i,
                        inst_j,
                    )
                    if n_timeout >= 2:
                        # Two timeouts at this n → skip remaining pairs and n
                        timed_out_at_n = True
                        logger.warning(
                            "  Skipping %s n>=%d (too many timeouts)",
                            family,
                            n,
                        )
                        break
                    continue

                row = {
                    "family": family,
                    "n_nodes": n,
                    "n_edges_1": g1.number_of_edges(),
                    "n_edges_2": g2.number_of_edges(),
                    "instance_i": inst_i,
                    "instance_j": inst_j,
                    "ged_value": result["ged_value"],
                    "ged_time_median_s": result["ged_time_median_s"],
                    "ged_time_iqr_s": result["ged_time_iqr_s"],
                    "ged_n_reps": result["ged_n_reps"],
                    "ged_times_all_s": json.dumps(result["ged_times_all_s"]),
                }
                rows.append(row)
                existing_keys.add((family, n, inst_i, inst_j))
                n_done += 1

            # Save intermediate results after each (family, n)
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(csv_path, index=False)
                logger.info(
                    "  %s n=%d: %d pairs done, %d timeouts → saved %d total rows",
                    family,
                    n,
                    n_done,
                    n_timeout,
                    len(rows),
                )

    logger.info("Final CSV saved: %s (%d rows)", csv_path, len(rows))
    return csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute GED on synthetic random graphs (same conditions as encoding benchmark)."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save synthetic_ged_times.csv.",
    )
    parser.add_argument(
        "--n-instances",
        type=int,
        default=5,
        help="Number of graph instances per (family, n). Default: 5.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (must match eval_encoding.py). Default: 42.",
    )
    parser.add_argument(
        "--n-reps",
        type=int,
        default=DEFAULT_N_REPS,
        help="Target timing repetitions per pair. Default: 25.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    compute_synthetic_ged_times(
        output_dir=args.output_dir,
        n_instances=args.n_instances,
        seed=args.seed,
        n_reps=args.n_reps,
    )


if __name__ == "__main__":
    main()
