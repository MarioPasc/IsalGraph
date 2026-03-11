# ruff: noqa: N803, N806
"""CLI orchestrator for computational advantage analysis.

Benchmarks the IsalGraph pipeline (encode + Levenshtein) against direct
GED computation. Follows Weber et al. (2019) timing guidelines: CPU time
via process_time(), 25 repetitions, median/IQR reporting.

Usage:
    python -m benchmarks.eval_computational.eval_computational \
        --data-root data/eval \
        --source-dir data/source \
        --output-dir results/eval_computational \
        --n-timing-reps 25 --n-pairs-per-bin 50 --seed 42 \
        --csv --plot --table
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from benchmarks.eval_computational.timing_utils import (
    fit_scaling_exponent,
    get_hardware_info,
    time_function,
    time_function_batch,
)
from benchmarks.plotting_styles import (
    PAUL_TOL_BRIGHT,
    PLOT_SETTINGS,
    apply_ieee_style,
    save_figure,
    save_latex_table,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

ALL_DATASETS = ["iam_letter_low", "iam_letter_med", "iam_letter_high", "linux", "aids"]

IAM_DATASETS: dict[str, str] = {
    "iam_letter_low": "LOW",
    "iam_letter_med": "MED",
    "iam_letter_high": "HIGH",
}
GRAPHEDX_DATASETS: dict[str, str] = {
    "linux": "LINUX",
    "aids": "AIDS",
}

DATASET_DISPLAY: dict[str, str] = {
    "iam_letter_low": "IAM LOW",
    "iam_letter_med": "IAM MED",
    "iam_letter_high": "IAM HIGH",
    "linux": "LINUX",
    "aids": "AIDS",
}

SIZE_BINS = [(3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
SIZE_BIN_LABELS = ["3-4", "5-6", "7-8", "9-10", "11-12"]

DEFAULT_DATA_ROOT = "/media/mpascual/Sandisk2TB/research/isalgraph/data/eval"
DEFAULT_SOURCE_DIR = "/media/mpascual/Sandisk2TB/research/isalgraph/data/source"
DEFAULT_OUTPUT_DIR = "/media/mpascual/Sandisk2TB/research/isalgraph/results/eval_computational"

N_MAX = 12

# GED cost functions (uniform topology-only, matching ged_computer.py)
_NODE_SUBST_COST = lambda n1, n2: 0  # noqa: E731
_NODE_DEL_COST = lambda n: 1  # noqa: E731
_NODE_INS_COST = lambda n: 1  # noqa: E731
_EDGE_SUBST_COST = lambda e1, e2: 0  # noqa: E731
_EDGE_DEL_COST = lambda e: 1  # noqa: E731
_EDGE_INS_COST = lambda e: 1  # noqa: E731

AMORTIZED_N_VALUES = [10, 50, 100, 200, 500, 1000]


# =============================================================================
# Data loading
# =============================================================================


def _load_source_graphs(
    dataset: str,
    source_dir: str,
) -> tuple[list[nx.Graph], list[str], list[str]]:
    """Load source NetworkX graphs and filter by n_max.

    Args:
        dataset: Dataset name.
        source_dir: Path to source data directory.

    Returns:
        (graphs, graph_ids, labels) after filtering.
    """
    from benchmarks.eval_setup.dataset_filter import filter_graphs

    if dataset in IAM_DATASETS:
        from benchmarks.eval_setup.iam_letter_loader import load_iam_letter

        level = IAM_DATASETS[dataset]
        letter_dir = os.path.join(source_dir, "Letter")
        ds = load_iam_letter(letter_dir, level)
        graphs, graph_ids, labels = ds.graphs, ds.graph_ids, ds.labels
    elif dataset in GRAPHEDX_DATASETS:
        from benchmarks.eval_setup.graphedx_loader import load_graphedx_dataset

        gx_name = GRAPHEDX_DATASETS[dataset]
        result = load_graphedx_dataset(gx_name, source_dir)
        graphs = result.graphs
        graph_ids = result.graph_ids
        labels = [""] * len(graphs)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Filter by n_max
    fr = filter_graphs(graphs, graph_ids, n_max=N_MAX)
    kept_graphs = [graphs[i] for i in fr.kept_indices]
    kept_ids = [graph_ids[i] for i in fr.kept_indices]
    kept_labels = [labels[i] for i in fr.kept_indices]
    logger.info(
        "Loaded %s: %d -> %d graphs (n_max=%d)",
        dataset,
        len(graphs),
        len(kept_graphs),
        N_MAX,
    )
    return kept_graphs, kept_ids, kept_labels


def _load_canonical_strings(
    data_root: str,
    dataset: str,
    method: str,
) -> dict[str, dict]:
    """Load precomputed canonical strings from eval pipeline.

    Args:
        data_root: Root of eval pipeline output.
        dataset: Dataset name.
        method: 'exhaustive' or 'greedy'.

    Returns:
        Dict mapping graph_id -> {string, length, time_s}.
    """
    path = os.path.join(data_root, "canonical_strings", f"{dataset}_{method}.json")
    with open(path) as f:
        data = json.load(f)
    return data["strings"]


def _load_graph_metadata(
    data_root: str,
    dataset: str,
) -> dict:
    """Load graph metadata (node_counts, edge_counts, graph_ids).

    Args:
        data_root: Root of eval pipeline output.
        dataset: Dataset name.

    Returns:
        Dict with graph_ids, node_counts, edge_counts.
    """
    path = os.path.join(data_root, "graph_metadata", f"{dataset}.json")
    with open(path) as f:
        return json.load(f)


# =============================================================================
# Phase A: Encoding timing
# =============================================================================


def _time_encoding(
    graphs: list[nx.Graph],
    graph_ids: list[str],
    n_reps: int,
) -> pd.DataFrame:
    """Time per-graph canonical encoding (exhaustive + greedy + conversion).

    Args:
        graphs: NetworkX graphs.
        graph_ids: Graph identifiers.
        n_reps: Number of timing repetitions.

    Returns:
        DataFrame with per-graph encoding times.
    """
    from isalgraph.adapters.networkx_adapter import NetworkXAdapter
    from isalgraph.core.canonical import canonical_string
    from isalgraph.core.graph_to_string import GraphToString

    adapter = NetworkXAdapter()
    rows: list[dict] = []

    for idx, (g, gid) in enumerate(zip(graphs, graph_ids, strict=True)):
        n_nodes = g.number_of_nodes()
        n_edges = g.number_of_edges()
        density = 2 * n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0

        # Decide reps based on graph size
        actual_reps = min(n_reps, 5) if n_nodes > 10 else n_reps

        # Time conversion: nx -> SparseGraph
        conv_timing = time_function(
            adapter.from_external,
            args=(g,),
            kwargs={"directed": False},
            n_reps=actual_reps,
        )

        # Pre-convert for encoding timing
        sg = adapter.from_external(g, directed=False)

        # Time exhaustive canonical
        exh_timing: dict | None = None
        exh_string: str | None = None
        try:
            # Single probe to check feasibility
            t_probe_start = time.process_time()
            probe_result = canonical_string(sg)
            t_probe = time.process_time() - t_probe_start

            if t_probe > 60.0:
                # Too slow for repeated timing
                exh_timing = {
                    "median_s": t_probe,
                    "iqr_s": float("nan"),
                    "all_times_s": [t_probe],
                    "n_reps": 1,
                }
                exh_string = probe_result
            else:
                exh_timing = time_function(
                    canonical_string,
                    args=(sg,),
                    n_reps=actual_reps,
                )
                exh_string = exh_timing["result"]
        except (ValueError, RuntimeError) as e:
            logger.warning("Exhaustive failed for %s: %s", gid, e)

        # Time pruned exhaustive
        from isalgraph.core.canonical_pruned import pruned_canonical_string

        pe_timing: dict | None = None
        pe_string: str | None = None
        try:
            t_probe_start = time.process_time()
            probe_pe = pruned_canonical_string(sg)
            t_probe_pe = time.process_time() - t_probe_start

            if t_probe_pe > 60.0:
                pe_timing = {
                    "median_s": t_probe_pe,
                    "iqr_s": float("nan"),
                    "all_times_s": [t_probe_pe],
                    "n_reps": 1,
                }
                pe_string = probe_pe
            else:
                pe_timing = time_function(
                    pruned_canonical_string,
                    args=(sg,),
                    n_reps=actual_reps,
                )
                pe_string = pe_timing["result"]
        except (ValueError, RuntimeError) as e:
            logger.warning("Pruned exhaustive failed for %s: %s", gid, e)

        # Time greedy-min (all starting nodes)
        def _greedy_min(sg_local=sg):
            best_s = None
            best_len = float("inf")
            for v in range(sg_local.node_count()):
                try:
                    gts = GraphToString(sg_local)
                    s, _ = gts.run(initial_node=v)
                    if len(s) < best_len or (len(s) == best_len and (best_s is None or s < best_s)):
                        best_s = s
                        best_len = len(s)
                except (ValueError, RuntimeError):
                    continue
            return best_s

        greedy_timing = time_function(_greedy_min, n_reps=actual_reps)
        greedy_string = greedy_timing["result"]

        row = {
            "graph_id": gid,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "density": round(density, 4),
            "conversion_time_median_s": conv_timing["median_s"],
            "conversion_time_iqr_s": conv_timing["iqr_s"],
            "exhaustive_time_median_s": exh_timing["median_s"] if exh_timing else float("nan"),
            "exhaustive_time_iqr_s": exh_timing.get("iqr_s", float("nan"))
            if exh_timing
            else float("nan"),
            "exhaustive_string_length": len(exh_string) if exh_string else -1,
            "exhaustive_n_reps": exh_timing["n_reps"] if exh_timing else 0,
            "pruned_exhaustive_time_median_s": pe_timing["median_s"] if pe_timing else float("nan"),
            "pruned_exhaustive_time_iqr_s": pe_timing.get("iqr_s", float("nan"))
            if pe_timing
            else float("nan"),
            "pruned_exhaustive_string_length": len(pe_string) if pe_string else -1,
            "pruned_exhaustive_n_reps": pe_timing["n_reps"] if pe_timing else 0,
            "greedy_time_median_s": greedy_timing["median_s"],
            "greedy_time_iqr_s": greedy_timing["iqr_s"],
            "greedy_string_length": len(greedy_string) if greedy_string else -1,
            "greedy_n_reps": greedy_timing["n_reps"],
            "total_exhaustive_median_s": (
                conv_timing["median_s"] + exh_timing["median_s"] if exh_timing else float("nan")
            ),
            "total_pruned_exhaustive_median_s": (
                conv_timing["median_s"] + pe_timing["median_s"] if pe_timing else float("nan")
            ),
            "total_greedy_median_s": conv_timing["median_s"] + greedy_timing["median_s"],
        }
        rows.append(row)

        if (idx + 1) % max(1, len(graphs) // 10) == 0:
            logger.info("  Encoding: %d/%d graphs", idx + 1, len(graphs))

    return pd.DataFrame(rows)


# =============================================================================
# Phase B: Stratified pair sampling
# =============================================================================


def _sample_pairs(
    graph_ids: list[str],
    node_counts: list[int],
    n_pairs_per_bin: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Sample pairs stratified by max(n_i, n_j).

    Args:
        graph_ids: Graph identifiers.
        node_counts: Node counts per graph.
        n_pairs_per_bin: Target pairs per size bin.
        seed: Random seed.

    Returns:
        List of (idx_i, idx_j) pairs.
    """
    rng = np.random.default_rng(seed)
    n = len(graph_ids)
    nc = np.array(node_counts)

    sampled_pairs: list[tuple[int, int]] = []

    for lo, hi in SIZE_BINS:
        # Find all pairs where max(n_i, n_j) falls in [lo, hi]
        candidates: list[tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                max_n = max(nc[i], nc[j])
                if lo <= max_n <= hi:
                    candidates.append((i, j))

        if not candidates:
            logger.info("  Bin [%d-%d]: no pairs available", lo, hi)
            continue

        n_select = min(n_pairs_per_bin, len(candidates))
        indices = rng.choice(len(candidates), size=n_select, replace=False)
        for idx in indices:
            sampled_pairs.append(candidates[idx])

        logger.info(
            "  Bin [%d-%d]: %d candidates, sampled %d",
            lo,
            hi,
            len(candidates),
            n_select,
        )

    logger.info("Total sampled pairs: %d", len(sampled_pairs))
    return sampled_pairs


# =============================================================================
# Phase C: Levenshtein timing
# =============================================================================


def _time_levenshtein(
    pairs: list[tuple[int, int]],
    graph_ids: list[str],
    node_counts: list[int],
    canonical_strings: dict[str, dict],
    n_reps: int,
) -> pd.DataFrame:
    """Time Levenshtein distance for sampled pairs.

    Times both the C extension (python-Levenshtein) and the pure
    Python implementation from isalgraph.core.canonical.

    Args:
        pairs: Sampled (idx_i, idx_j) pairs.
        graph_ids: Graph identifiers.
        node_counts: Node counts per graph.
        canonical_strings: Dict mapping graph_id -> {string, length}.
        n_reps: Timing repetitions.

    Returns:
        DataFrame with per-pair Levenshtein times.
    """
    from isalgraph.core.canonical import levenshtein as pure_levenshtein

    try:
        import Levenshtein as Lev_C  # noqa: N811

        has_c_extension = True
    except ImportError:
        Lev_C = None
        has_c_extension = False
        logger.warning("python-Levenshtein not installed; using pure Python only")

    rows: list[dict] = []

    for pair_idx, (i, j) in enumerate(pairs):
        gid_i = graph_ids[i]
        gid_j = graph_ids[j]

        if gid_i not in canonical_strings or gid_j not in canonical_strings:
            logger.warning("Missing canonical string for pair (%s, %s)", gid_i, gid_j)
            continue

        s_i = canonical_strings[gid_i]["string"]
        s_j = canonical_strings[gid_j]["string"]
        max_n = max(node_counts[i], node_counts[j])

        # C extension timing (batch for very short strings)
        c_timing: dict | None = None
        if has_c_extension:
            c_timing = time_function_batch(
                Lev_C.distance,
                args=(s_i, s_j),
                n_reps=n_reps,
                batch_size=1000,
            )

        # Pure Python timing
        py_timing = time_function(
            pure_levenshtein,
            args=(s_i, s_j),
            n_reps=n_reps,
        )

        row = {
            "graph_i": gid_i,
            "graph_j": gid_j,
            "max_n": max_n,
            "string_len_i": len(s_i),
            "string_len_j": len(s_j),
            "levenshtein_value": py_timing["result"],
            "c_ext_time_median_s": c_timing["median_s"] if c_timing else float("nan"),
            "c_ext_time_iqr_s": c_timing["iqr_s"] if c_timing else float("nan"),
            "pure_python_time_median_s": py_timing["median_s"],
            "pure_python_time_iqr_s": py_timing["iqr_s"],
        }
        rows.append(row)

        if (pair_idx + 1) % max(1, len(pairs) // 10) == 0:
            logger.info("  Levenshtein: %d/%d pairs", pair_idx + 1, len(pairs))

    return pd.DataFrame(rows)


# =============================================================================
# Phase D: GED timing
# =============================================================================


def _time_ged(
    pairs: list[tuple[int, int]],
    graphs: list[nx.Graph],
    graph_ids: list[str],
    node_counts: list[int],
    n_reps: int,
    ged_call_timeout: float = 60.0,
    pair_budget_s: float = 120.0,
) -> pd.DataFrame:
    """Time exact GED computation for sampled pairs.

    Uses adaptive repetitions: a single untimed probe determines how many
    timed repetitions are feasible within ``pair_budget_s``.

    Probe time -> reps policy:
        < 0.1 s   : n_reps  (full)
        0.1 - 1 s : min(10, n_reps)
        1 - 10 s  : 3
        10 - 60 s : 1  (probe only)
        > 60 s    : 0  (skip, record inf)

    Args:
        pairs: Sampled (idx_i, idx_j) pairs.
        graphs: NetworkX graphs.
        graph_ids: Graph identifiers.
        node_counts: Node counts per graph.
        n_reps: Maximum timing repetitions.
        ged_call_timeout: Per-call timeout for NetworkX GED (seconds).
        pair_budget_s: Maximum total time budget per pair (seconds).

    Returns:
        DataFrame with per-pair GED times.
    """
    rows: list[dict] = []
    cumulative_s = 0.0

    for pair_idx, (i, j) in enumerate(pairs):
        g1, g2 = graphs[i], graphs[j]
        max_n = max(node_counts[i], node_counts[j])

        def _compute_ged(g1_local=g1, g2_local=g2):
            return nx.graph_edit_distance(
                g1_local,
                g2_local,
                node_subst_cost=_NODE_SUBST_COST,
                node_del_cost=_NODE_DEL_COST,
                node_ins_cost=_NODE_INS_COST,
                edge_subst_cost=_EDGE_SUBST_COST,
                edge_del_cost=_EDGE_DEL_COST,
                edge_ins_cost=_EDGE_INS_COST,
                timeout=ged_call_timeout,
            )

        # --- Probe: single untimed GED call to estimate cost ---
        t_probe_start = time.process_time()
        try:
            probe_result = _compute_ged()
        except Exception:
            logger.warning("GED probe failed for pair (%s, %s)", graph_ids[i], graph_ids[j])
            rows.append(
                {
                    "graph_i": graph_ids[i],
                    "graph_j": graph_ids[j],
                    "max_n": max_n,
                    "ged_value": float("inf"),
                    "ged_time_median_s": float("inf"),
                    "ged_time_iqr_s": float("nan"),
                    "ged_n_reps": 0,
                    "ged_times_all_s": "[]",
                }
            )
            continue
        t_probe = time.process_time() - t_probe_start

        ged_value = float(probe_result) if probe_result is not None else float("inf")

        # --- Adaptive reps based on probe time ---
        if t_probe > ged_call_timeout * 0.9:
            # Hit timeout — record probe as the only measurement
            actual_reps = 0
        elif t_probe > 10.0:
            actual_reps = 1
        elif t_probe > 1.0:
            actual_reps = 3
        elif t_probe > 0.1:
            actual_reps = min(10, n_reps)
        else:
            actual_reps = n_reps

        # Also cap by pair budget
        if actual_reps > 0 and t_probe > 0:
            max_affordable = max(1, int(pair_budget_s / t_probe))
            actual_reps = min(actual_reps, max_affordable)

        if actual_reps > 0:
            try:
                ged_timing = time_function(_compute_ged, n_reps=actual_reps, warmup=0)
                ged_value = (
                    float(ged_timing["result"]) if ged_timing["result"] is not None else ged_value
                )
            except Exception:
                logger.warning("GED timing failed for pair (%s, %s)", graph_ids[i], graph_ids[j])
                ged_timing = {
                    "median_s": t_probe,
                    "iqr_s": float("nan"),
                    "all_times_s": [t_probe],
                    "n_reps": 1,
                }
        else:
            # Only have the probe measurement
            ged_timing = {
                "median_s": t_probe,
                "iqr_s": float("nan"),
                "all_times_s": [t_probe],
                "n_reps": 1,
            }

        pair_total = t_probe + actual_reps * ged_timing["median_s"]
        cumulative_s += pair_total

        row = {
            "graph_i": graph_ids[i],
            "graph_j": graph_ids[j],
            "max_n": max_n,
            "ged_value": ged_value,
            "ged_time_median_s": ged_timing["median_s"],
            "ged_time_iqr_s": ged_timing.get("iqr_s", float("nan")),
            "ged_n_reps": ged_timing.get("n_reps", 0),
            "ged_times_all_s": json.dumps(ged_timing.get("all_times_s", [])),
        }
        rows.append(row)

        if (pair_idx + 1) % max(1, len(pairs) // 10) == 0:
            remaining = len(pairs) - pair_idx - 1
            avg_s = cumulative_s / (pair_idx + 1)
            eta_min = remaining * avg_s / 60
            logger.info(
                "  GED: %d/%d pairs (%.1fs cumulative, ~%.0f min remaining, "
                "last probe=%.3fs, reps=%d)",
                pair_idx + 1,
                len(pairs),
                cumulative_s,
                eta_min,
                t_probe,
                actual_reps,
            )

    return pd.DataFrame(rows)


# =============================================================================
# Phase E: Crossover analysis
# =============================================================================


def _analyze_crossover(
    encoding_df: pd.DataFrame,
    lev_df: pd.DataFrame,
    ged_df: pd.DataFrame,
) -> dict:
    """Find crossover point where IsalGraph becomes faster than GED.

    For each size bin:
    - T_isalgraph(n) = median_encode(n)*2 + median_lev(n)
    - T_ged(n) = median_ged(n)

    Args:
        encoding_df: Per-graph encoding times.
        lev_df: Per-pair Levenshtein times.
        ged_df: Per-pair GED times.

    Returns:
        Dict with per-bin comparisons and crossover point.
    """
    # Use exhaustive encoding if available, else greedy
    enc_col = "total_exhaustive_median_s"
    if encoding_df[enc_col].isna().all():
        enc_col = "total_greedy_median_s"

    bins_analysis: list[dict] = []
    crossover_n = None

    for (lo, hi), label in zip(SIZE_BINS, SIZE_BIN_LABELS, strict=True):
        # Encoding time for graphs in this bin
        enc_mask = (encoding_df["n_nodes"] >= lo) & (encoding_df["n_nodes"] <= hi)
        enc_median = encoding_df.loc[enc_mask, enc_col].median() if enc_mask.any() else float("nan")

        # Levenshtein time for pairs in this bin
        lev_mask = (lev_df["max_n"] >= lo) & (lev_df["max_n"] <= hi)
        # Use C extension if available
        Lev_Col = (
            "c_ext_time_median_s"
            if "c_ext_time_median_s" in lev_df.columns
            else "pure_python_time_median_s"
        )
        lev_median = lev_df.loc[lev_mask, Lev_Col].median() if lev_mask.any() else float("nan")

        # GED time for pairs in this bin
        ged_mask = (ged_df["max_n"] >= lo) & (ged_df["max_n"] <= hi)
        ged_finite = ged_df.loc[
            ged_mask & np.isfinite(ged_df["ged_time_median_s"]), "ged_time_median_s"
        ]
        ged_median = ged_finite.median() if len(ged_finite) > 0 else float("nan")

        # IsalGraph total: encode both graphs + Levenshtein
        t_isalgraph = (
            2 * enc_median + lev_median
            if np.isfinite(enc_median) and np.isfinite(lev_median)
            else float("nan")
        )
        t_ged = ged_median

        speedup = (
            t_ged / t_isalgraph
            if (np.isfinite(t_isalgraph) and np.isfinite(t_ged) and t_isalgraph > 0)
            else float("nan")
        )
        isalgraph_faster = bool(np.isfinite(speedup) and speedup > 1.0)

        if isalgraph_faster and crossover_n is None:
            crossover_n = lo

        bins_analysis.append(
            {
                "bin": label,
                "bin_lo": lo,
                "bin_hi": hi,
                "encode_median_s": float(enc_median) if np.isfinite(enc_median) else None,
                "lev_median_s": float(lev_median) if np.isfinite(lev_median) else None,
                "ged_median_s": float(ged_median) if np.isfinite(ged_median) else None,
                "t_isalgraph_s": float(t_isalgraph) if np.isfinite(t_isalgraph) else None,
                "t_ged_s": float(t_ged) if np.isfinite(t_ged) else None,
                "speedup": float(speedup) if np.isfinite(speedup) else None,
                "isalgraph_faster": isalgraph_faster,
                "n_encoding_samples": int(enc_mask.sum()),
                "n_lev_samples": int(lev_mask.sum()),
                "n_ged_samples": int(len(ged_finite)),
            }
        )

    return {
        "crossover_n": crossover_n,
        "bins": bins_analysis,
    }


# =============================================================================
# Phase F: Amortized comparison
# =============================================================================


def _compute_amortized(
    encoding_df: pd.DataFrame,
    lev_df: pd.DataFrame,
    ged_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute amortized pipeline comparison for various dataset sizes.

    Args:
        encoding_df: Per-graph encoding times.
        lev_df: Per-pair Levenshtein times.
        ged_df: Per-pair GED times.

    Returns:
        DataFrame with amortized comparison.
    """
    # Use exhaustive encoding if available
    enc_col = "total_exhaustive_median_s"
    if encoding_df[enc_col].isna().all():
        enc_col = "total_greedy_median_s"

    mean_encode = encoding_df[enc_col].median()
    Lev_Col = (
        "c_ext_time_median_s"
        if "c_ext_time_median_s" in lev_df.columns
        else "pure_python_time_median_s"
    )
    mean_lev = lev_df[Lev_Col].median()
    mean_ged = ged_df.loc[np.isfinite(ged_df["ged_time_median_s"]), "ged_time_median_s"].median()

    rows: list[dict] = []
    for n_graphs in AMORTIZED_N_VALUES:
        n_pairs = n_graphs * (n_graphs - 1) // 2
        t_encoding = n_graphs * mean_encode
        t_lev = n_pairs * mean_lev
        t_isalgraph = t_encoding + t_lev
        t_ged = n_pairs * mean_ged
        speedup = t_ged / t_isalgraph if t_isalgraph > 0 else float("inf")

        rows.append(
            {
                "n_graphs": n_graphs,
                "n_pairs": n_pairs,
                "total_encoding_time_s": float(t_encoding),
                "total_levenshtein_time_s": float(t_lev),
                "total_isalgraph_time_s": float(t_isalgraph),
                "total_ged_time_s": float(t_ged),
                "speedup": float(speedup),
                "encoding_fraction": float(t_encoding / t_isalgraph) if t_isalgraph > 0 else 0.0,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Phase G: Scaling regression
# =============================================================================


def _compute_scaling(
    encoding_df: pd.DataFrame,
    lev_df: pd.DataFrame,
    ged_df: pd.DataFrame,
) -> dict:
    """Fit scaling exponents for each operation.

    Args:
        encoding_df: Per-graph encoding times.
        lev_df: Per-pair Levenshtein times.
        ged_df: Per-pair GED times.

    Returns:
        Dict with scaling regression results.
    """
    # GED: time vs max_n
    ged_finite = ged_df[np.isfinite(ged_df["ged_time_median_s"])]
    ged_scaling = fit_scaling_exponent(
        ged_finite["max_n"].values,
        ged_finite["ged_time_median_s"].values,
    )

    # Exhaustive encoding: time vs n_nodes
    enc_valid = encoding_df[np.isfinite(encoding_df["exhaustive_time_median_s"])]
    enc_scaling = (
        fit_scaling_exponent(
            enc_valid["n_nodes"].values,
            enc_valid["exhaustive_time_median_s"].values,
        )
        if len(enc_valid) > 0
        else {"alpha": float("nan"), "n_points": 0}
    )

    # Greedy encoding: time vs n_nodes
    greedy_valid = encoding_df[np.isfinite(encoding_df["greedy_time_median_s"])]
    greedy_scaling = (
        fit_scaling_exponent(
            greedy_valid["n_nodes"].values,
            greedy_valid["greedy_time_median_s"].values,
        )
        if len(greedy_valid) > 0
        else {"alpha": float("nan"), "n_points": 0}
    )

    # Levenshtein: time vs max(string_len)
    Lev_Col = (
        "c_ext_time_median_s"
        if "c_ext_time_median_s" in lev_df.columns
        else "pure_python_time_median_s"
    )
    max_slen = np.maximum(lev_df["string_len_i"].values, lev_df["string_len_j"].values)
    lev_valid_mask = np.isfinite(lev_df[Lev_Col].values) & (max_slen > 0)
    lev_scaling = (
        fit_scaling_exponent(
            max_slen[lev_valid_mask],
            lev_df[Lev_Col].values[lev_valid_mask],
        )
        if lev_valid_mask.any()
        else {"alpha": float("nan"), "n_points": 0}
    )

    return {
        "ged_vs_max_n": ged_scaling,
        "exhaustive_encoding_vs_n": enc_scaling,
        "greedy_encoding_vs_n": greedy_scaling,
        "levenshtein_vs_max_strlen": lev_scaling,
    }


# =============================================================================
# Per-dataset analysis
# =============================================================================


def _analyze_dataset(
    dataset: str,
    graphs: list[nx.Graph],
    graph_ids: list[str],
    node_counts: list[int],
    canonical_strings: dict[str, dict],
    args: argparse.Namespace,
) -> dict:
    """Run full computational analysis for one dataset.

    Args:
        dataset: Dataset name.
        graphs: Filtered NetworkX graphs.
        graph_ids: Graph identifiers.
        node_counts: Node count per graph.
        canonical_strings: Precomputed canonical strings dict.
        args: CLI arguments.

    Returns:
        Dict with all results for this dataset.
    """
    out_raw = os.path.join(args.output_dir, "raw")
    out_stats = os.path.join(args.output_dir, "stats")
    os.makedirs(out_raw, exist_ok=True)
    os.makedirs(out_stats, exist_ok=True)

    # --- Phase A: Encoding timing (with checkpoint) ---
    enc_csv = os.path.join(out_raw, f"{dataset}_encoding_times.csv")
    if os.path.isfile(enc_csv):
        logger.info("=== %s: Phase A - LOADING checkpoint ===", dataset)
        encoding_df = pd.read_csv(enc_csv)
    else:
        logger.info("=== %s: Phase A - Encoding timing ===", dataset)
        encoding_df = _time_encoding(graphs, graph_ids, args.n_timing_reps)
        if args.csv:
            encoding_df.to_csv(enc_csv, index=False)
            logger.info("Saved encoding times CSV")

    logger.info("=== %s: Phase B - Pair sampling ===", dataset)
    pairs = _sample_pairs(
        graph_ids,
        node_counts,
        args.n_pairs_per_bin,
        args.seed,
    )

    # --- Phase C: Levenshtein timing (with checkpoint) ---
    lev_csv = os.path.join(out_raw, f"{dataset}_levenshtein_times.csv")
    if os.path.isfile(lev_csv):
        logger.info("=== %s: Phase C - LOADING checkpoint ===", dataset)
        lev_df = pd.read_csv(lev_csv)
    else:
        logger.info("=== %s: Phase C - Levenshtein timing ===", dataset)
        lev_df = _time_levenshtein(
            pairs,
            graph_ids,
            node_counts,
            canonical_strings,
            args.n_timing_reps,
        )
        if args.csv:
            lev_df.to_csv(lev_csv, index=False)
            logger.info("Saved Levenshtein times CSV")

    # --- Phase D: GED timing (with checkpoint) ---
    ged_csv = os.path.join(out_raw, f"{dataset}_ged_times.csv")
    if os.path.isfile(ged_csv):
        logger.info("=== %s: Phase D - LOADING checkpoint ===", dataset)
        ged_df = pd.read_csv(ged_csv)
    else:
        logger.info("=== %s: Phase D - GED timing ===", dataset)
        ged_df = _time_ged(
            pairs,
            graphs,
            graph_ids,
            node_counts,
            args.n_timing_reps,
        )
    if args.csv and not os.path.isfile(ged_csv):
        ged_df.to_csv(ged_csv, index=False)
        logger.info("Saved GED times CSV")

    logger.info("=== %s: Phase E - Crossover analysis ===", dataset)
    crossover = _analyze_crossover(encoding_df, lev_df, ged_df)

    logger.info("=== %s: Phase F - Amortized comparison ===", dataset)
    amortized_df = _compute_amortized(encoding_df, lev_df, ged_df)
    if args.csv:
        amortized_df.to_csv(
            os.path.join(out_raw, f"{dataset}_amortized_comparison.csv"),
            index=False,
        )

    logger.info("=== %s: Phase G - Scaling regression ===", dataset)
    scaling = _compute_scaling(encoding_df, lev_df, ged_df)

    # Summary stats
    enc_col = "total_exhaustive_median_s"
    if encoding_df[enc_col].isna().all():
        enc_col = "total_greedy_median_s"

    Lev_Col = (
        "c_ext_time_median_s"
        if "c_ext_time_median_s" in lev_df.columns
        else "pure_python_time_median_s"
    )

    ged_finite = ged_df[np.isfinite(ged_df["ged_time_median_s"])]

    stats_dict = {
        "dataset": dataset,
        "n_graphs": len(graphs),
        "mean_n_nodes": float(np.mean(node_counts)),
        "encoding_median_s": float(encoding_df[enc_col].median()),
        "encoding_iqr_s": float(
            encoding_df[enc_col].quantile(0.75) - encoding_df[enc_col].quantile(0.25)
        ),
        "levenshtein_median_s": float(lev_df[Lev_Col].median()),
        "ged_median_s": float(ged_finite["ged_time_median_s"].median())
        if len(ged_finite) > 0
        else float("nan"),
        "crossover": crossover,
        "scaling": scaling,
        "n_sampled_pairs": len(pairs),
        "n_ged_finite": len(ged_finite),
        "max_speedup": float(amortized_df["speedup"].max()),
    }

    with open(os.path.join(out_stats, f"{dataset}_timing_stats.json"), "w") as f:
        json.dump(stats_dict, f, indent=2, default=str)

    return {
        "stats": stats_dict,
        "encoding_df": encoding_df,
        "lev_df": lev_df,
        "ged_df": ged_df,
        "amortized_df": amortized_df,
        "crossover": crossover,
        "scaling": scaling,
    }


# =============================================================================
# Visualization
# =============================================================================


def _plot_main_figure(
    all_results: dict[str, dict],
    output_dir: str,
) -> None:
    """Create the 2x2 main figure.

    Panels:
    (a) Log-log scatter: per-pair time vs max(n)
    (b) Bar chart: speedup per dataset
    (c) Amortized comparison (line plot)
    (d) Stacked bar: time breakdown per size bin
    """
    apply_ieee_style()
    fig, axes = plt.subplots(2, 2, figsize=(PLOT_SETTINGS["figure_width_double"], 5.0))

    color_ged = PAUL_TOL_BRIGHT["red"]
    color_isal = PAUL_TOL_BRIGHT["blue"]
    color_enc = PAUL_TOL_BRIGHT["cyan"]
    color_lev = PAUL_TOL_BRIGHT["green"]

    # ---- Panel (a): Log-log scaling ----
    ax_a = axes[0, 0]
    for ds_name, res in all_results.items():
        ged_df = res["ged_df"]
        lev_df = res["lev_df"]
        enc_df = res["encoding_df"]

        # GED points
        ged_f = ged_df[np.isfinite(ged_df["ged_time_median_s"])]
        if len(ged_f) > 0:
            ax_a.scatter(
                ged_f["max_n"],
                ged_f["ged_time_median_s"],
                c=color_ged,
                alpha=0.3,
                s=8,
                label="GED" if ds_name == list(all_results)[0] else None,
            )

        # IsalGraph total points (encode * 2 + lev)
        Lev_Col = (
            "c_ext_time_median_s"
            if "c_ext_time_median_s" in lev_df.columns
            else "pure_python_time_median_s"
        )
        enc_col = "total_exhaustive_median_s"
        if enc_df[enc_col].isna().all():
            enc_col = "total_greedy_median_s"

        # Match pairs to encoding times
        for _, row in lev_df.iterrows():
            # Approximate encode time by mean for that size bin
            max_n = row["max_n"]
            enc_mask = (enc_df["n_nodes"] >= max_n - 1) & (enc_df["n_nodes"] <= max_n + 1)
            enc_time = enc_df.loc[enc_mask, enc_col].median() if enc_mask.any() else float("nan")
            t_total = 2 * enc_time + row[Lev_Col]
            if np.isfinite(t_total) and t_total > 0:
                ax_a.scatter(
                    max_n,
                    t_total,
                    c=color_isal,
                    alpha=0.3,
                    s=8,
                    label="IsalGraph"
                    if ds_name == list(all_results)[0] and _ == lev_df.index[0]
                    else None,
                )

    # Reference lines
    ns = np.arange(3, 13)
    t_ref = ns.astype(float)
    ax_a.plot(ns, 1e-6 * t_ref**2, ":", color="grey", alpha=0.5, label=r"$O(n^2)$")
    ax_a.plot(ns, 1e-8 * t_ref**3, "--", color="grey", alpha=0.5, label=r"$O(n^3)$")
    ax_a.plot(ns, 1e-6 * 2.0**t_ref / 1e4, "-.", color="grey", alpha=0.5, label=r"$O(2^n)$")

    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("max(n)")
    ax_a.set_ylabel("Time (s)")
    ax_a.set_title("(a) Per-pair scaling")
    ax_a.legend(fontsize=7, loc="upper left")

    # ---- Panel (b): Speedup bar chart ----
    ax_b = axes[0, 1]
    ds_names = list(all_results.keys())
    display_names = [DATASET_DISPLAY.get(d, d) for d in ds_names]

    # Compute overall median speedup per dataset
    speedups = []
    speedup_iqrs = []
    for ds_name in ds_names:
        res = all_results[ds_name]
        ged_df = res["ged_df"]
        lev_df = res["lev_df"]
        enc_df = res["encoding_df"]

        Lev_Col = (
            "c_ext_time_median_s"
            if "c_ext_time_median_s" in lev_df.columns
            else "pure_python_time_median_s"
        )
        enc_col = "total_exhaustive_median_s"
        if enc_df[enc_col].isna().all():
            enc_col = "total_greedy_median_s"

        # Per-pair speedup
        pair_speedups: list[float] = []
        for _, row in ged_df.iterrows():
            if not np.isfinite(row["ged_time_median_s"]):
                continue
            lev_match = lev_df[
                (lev_df["graph_i"] == row["graph_i"]) & (lev_df["graph_j"] == row["graph_j"])
            ]
            if lev_match.empty:
                continue
            lev_time = lev_match.iloc[0][Lev_Col]
            enc_time = enc_df.loc[
                enc_df["graph_id"].isin([row["graph_i"], row["graph_j"]]), enc_col
            ].mean()
            t_isal = 2 * enc_time + lev_time
            if np.isfinite(t_isal) and t_isal > 0:
                pair_speedups.append(row["ged_time_median_s"] / t_isal)

        if pair_speedups:
            arr = np.array(pair_speedups)
            speedups.append(float(np.median(arr)))
            speedup_iqrs.append(float(np.percentile(arr, 75) - np.percentile(arr, 25)))
        else:
            speedups.append(0.0)
            speedup_iqrs.append(0.0)

    x_pos = np.arange(len(ds_names))
    ax_b.bar(
        x_pos,
        speedups,
        yerr=speedup_iqrs,
        color=color_isal,
        alpha=0.8,
        capsize=3,
    )
    ax_b.axhline(1.0, color="grey", ls="--", lw=0.8)
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)
    ax_b.set_ylabel("Speedup (GED / IsalGraph)")
    ax_b.set_title("(b) Per-pair speedup")

    # ---- Panel (c): Amortized comparison ----
    ax_c = axes[1, 0]
    # Use first dataset as representative
    first_ds = list(all_results.keys())[0]
    amort_df = all_results[first_ds]["amortized_df"]

    ax_c.plot(
        amort_df["n_graphs"],
        amort_df["total_ged_time_s"],
        "o-",
        color=color_ged,
        ms=4,
        label="GED",
    )
    ax_c.plot(
        amort_df["n_graphs"],
        amort_df["total_isalgraph_time_s"],
        "s-",
        color=color_isal,
        ms=4,
        label="IsalGraph",
    )
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xlabel("Dataset size (N)")
    ax_c.set_ylabel("Total time (s)")
    ax_c.set_title(f"(c) Amortized ({DATASET_DISPLAY.get(first_ds, first_ds)})")
    ax_c.legend(fontsize=8)

    # ---- Panel (d): Stacked bar breakdown ----
    ax_d = axes[1, 1]
    # Aggregate across datasets per bin
    bin_enc_times: list[float] = []
    bin_lev_times: list[float] = []
    bin_ged_times: list[float] = []

    for lo, hi in SIZE_BINS:
        enc_vals: list[float] = []
        lev_vals: list[float] = []
        ged_vals: list[float] = []

        for res in all_results.values():
            enc_df = res["encoding_df"]
            lev_df = res["lev_df"]
            ged_df = res["ged_df"]

            enc_col = "total_exhaustive_median_s"
            if enc_df[enc_col].isna().all():
                enc_col = "total_greedy_median_s"
            Lev_Col = (
                "c_ext_time_median_s"
                if "c_ext_time_median_s" in lev_df.columns
                else "pure_python_time_median_s"
            )

            mask_enc = (enc_df["n_nodes"] >= lo) & (enc_df["n_nodes"] <= hi)
            mask_lev = (lev_df["max_n"] >= lo) & (lev_df["max_n"] <= hi)
            mask_ged = (ged_df["max_n"] >= lo) & (ged_df["max_n"] <= hi)

            if mask_enc.any():
                enc_vals.append(enc_df.loc[mask_enc, enc_col].median())
            if mask_lev.any():
                lev_vals.append(lev_df.loc[mask_lev, Lev_Col].median())
            ged_f = ged_df.loc[
                mask_ged & np.isfinite(ged_df["ged_time_median_s"]), "ged_time_median_s"
            ]
            if len(ged_f) > 0:
                ged_vals.append(ged_f.median())

        bin_enc_times.append(float(np.median(enc_vals)) if enc_vals else 0.0)
        bin_lev_times.append(float(np.median(lev_vals)) if lev_vals else 0.0)
        bin_ged_times.append(float(np.median(ged_vals)) if ged_vals else 0.0)

    x_bins = np.arange(len(SIZE_BINS))
    width = 0.35
    ax_d.bar(x_bins - width / 2, bin_enc_times, width * 0.5, label="Encoding", color=color_enc)
    ax_d.bar(
        x_bins - width / 2,
        bin_lev_times,
        width * 0.5,
        bottom=bin_enc_times,
        label="Levenshtein",
        color=color_lev,
    )
    ax_d.bar(x_bins + width / 2, bin_ged_times, width * 0.5, label="GED", color=color_ged)
    ax_d.set_xticks(x_bins)
    ax_d.set_xticklabels(SIZE_BIN_LABELS, fontsize=8)
    ax_d.set_xlabel("max(n) bin")
    ax_d.set_ylabel("Median time (s)")
    ax_d.set_title("(d) Time breakdown")
    ax_d.legend(fontsize=7)
    ax_d.set_yscale("log")

    fig.tight_layout()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    save_figure(fig, os.path.join(fig_dir, "computational_main_figure"))
    plt.close(fig)
    logger.info("Saved main figure")


def _plot_scaling_loglog(
    dataset: str,
    result: dict,
    output_dir: str,
) -> None:
    """Plot per-dataset log-log scaling with regression lines."""
    apply_ieee_style()
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(PLOT_SETTINGS["figure_width_single"], PLOT_SETTINGS["figure_width_single"] * 0.75),
    )

    color_ged = PAUL_TOL_BRIGHT["red"]
    color_isal = PAUL_TOL_BRIGHT["blue"]

    ged_df = result["ged_df"]
    enc_df = result["encoding_df"]

    # GED scaling
    ged_f = ged_df[np.isfinite(ged_df["ged_time_median_s"])]
    if len(ged_f) > 0:
        ax.scatter(
            ged_f["max_n"], ged_f["ged_time_median_s"], c=color_ged, s=12, alpha=0.5, label="GED"
        )

        # Regression line
        scaling = result["scaling"]["ged_vs_max_n"]
        if np.isfinite(scaling.get("alpha", float("nan"))):
            ns = np.linspace(ged_f["max_n"].min(), ged_f["max_n"].max(), 50)
            ax.plot(
                ns,
                scaling["c"] * ns ** scaling["alpha"],
                "--",
                color=color_ged,
                alpha=0.7,
                label=rf"GED: $\alpha$={scaling['alpha']:.2f}, $R^2$={scaling['r_squared']:.2f}",
            )

    # Encoding scaling
    enc_col = "exhaustive_time_median_s"
    enc_valid = enc_df[np.isfinite(enc_df[enc_col])]
    if len(enc_valid) > 0:
        ax.scatter(
            enc_valid["n_nodes"],
            enc_valid[enc_col],
            c=color_isal,
            s=12,
            alpha=0.5,
            label="Exhaustive enc.",
        )

        scaling_enc = result["scaling"]["exhaustive_encoding_vs_n"]
        if np.isfinite(scaling_enc.get("alpha", float("nan"))):
            ns = np.linspace(enc_valid["n_nodes"].min(), enc_valid["n_nodes"].max(), 50)
            ax.plot(
                ns,
                scaling_enc["c"] * ns ** scaling_enc["alpha"],
                "--",
                color=color_isal,
                alpha=0.7,
                label=rf"Enc: $\alpha$={scaling_enc['alpha']:.2f}",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Graph size (n)")
    ax.set_ylabel("Time (s)")
    ax.set_title(DATASET_DISPLAY.get(dataset, dataset))
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    save_figure(fig, os.path.join(fig_dir, f"scaling_loglog_{dataset}"))
    plt.close(fig)


# =============================================================================
# Table generation
# =============================================================================


def _generate_summary_table(
    all_results: dict[str, dict],
    output_dir: str,
) -> None:
    """Generate LaTeX summary table."""
    rows: list[dict] = []
    for ds_name, res in all_results.items():
        stats_d = res["stats"]
        scaling = res["scaling"]

        rows.append(
            {
                "Dataset": DATASET_DISPLAY.get(ds_name, ds_name),
                "N": stats_d["n_graphs"],
                "mean n": f"{stats_d['mean_n_nodes']:.1f}",
                "Encode (s)": f"{stats_d['encoding_median_s']:.4f}",
                "Lev (s)": f"{stats_d['levenshtein_median_s']:.6f}",
                "GED (s)": f"{stats_d['ged_median_s']:.4f}"
                if np.isfinite(stats_d["ged_median_s"])
                else "---",
                "Speedup": f"{stats_d['max_speedup']:.1f}x",
                "a GED": f"{scaling['ged_vs_max_n']['alpha']:.2f}"
                if np.isfinite(scaling["ged_vs_max_n"]["alpha"])
                else "---",
                "a enc": f"{scaling['exhaustive_encoding_vs_n']['alpha']:.2f}"
                if np.isfinite(scaling["exhaustive_encoding_vs_n"]["alpha"])
                else "---",
            }
        )

    df = pd.DataFrame(rows)
    table_dir = os.path.join(output_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)
    save_latex_table(
        df,
        os.path.join(table_dir, "computational_summary.tex"),
        caption="Computational advantage of IsalGraph over exact GED.",
        label="tab:computational",
    )
    logger.info("Saved summary table")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Computational advantage analysis for IsalGraph.",
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="Root directory of eval pipeline output.",
    )
    parser.add_argument(
        "--source-dir",
        default=DEFAULT_SOURCE_DIR,
        help="Root directory of source graph files.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for results.",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated dataset names (default: all).",
    )
    parser.add_argument(
        "--n-timing-reps",
        type=int,
        default=25,
        help="Repetitions per timing measurement.",
    )
    parser.add_argument(
        "--n-pairs-per-bin",
        type=int,
        default=50,
        help="Pairs per size bin for pair timing.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=["local", "picasso"],
        default="local",
        help="Execution mode.",
    )
    parser.add_argument("--csv", action="store_true", help="Save raw CSVs.")
    parser.add_argument("--plot", action="store_true", help="Generate figures.")
    parser.add_argument("--table", action="store_true", help="Generate LaTeX table.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if args.mode == "picasso":
        args.data_root = os.environ.get("ISALGRAPH_DATA_ROOT", args.data_root)
        args.source_dir = os.environ.get("ISALGRAPH_SOURCE_DIR", args.source_dir)
        args.output_dir = os.environ.get("ISALGRAPH_OUTPUT_DIR", args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    datasets = args.datasets.split(",") if args.datasets else ALL_DATASETS

    # Save hardware info
    hw_info = get_hardware_info()
    hw_info["n_timing_reps"] = args.n_timing_reps
    hw_info["n_pairs_per_bin"] = args.n_pairs_per_bin
    hw_info["seed"] = args.seed
    stats_dir = os.path.join(args.output_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    with open(os.path.join(stats_dir, "hardware_info.json"), "w") as f:
        json.dump(hw_info, f, indent=2)
    logger.info("Hardware: %s", hw_info)

    all_results: dict[str, dict] = {}

    for dataset in datasets:
        logger.info("=" * 60)
        logger.info("Processing dataset: %s", dataset)
        logger.info("=" * 60)

        t_start = time.perf_counter()

        # Load source graphs
        graphs, graph_ids, labels = _load_source_graphs(dataset, args.source_dir)
        node_counts = [g.number_of_nodes() for g in graphs]

        # Load precomputed canonical strings (use exhaustive if available)
        try:
            canonical_strings = _load_canonical_strings(args.data_root, dataset, "exhaustive")
        except FileNotFoundError:
            logger.warning("No exhaustive strings for %s, trying greedy", dataset)
            canonical_strings = _load_canonical_strings(args.data_root, dataset, "greedy")

        result = _analyze_dataset(
            dataset,
            graphs,
            graph_ids,
            node_counts,
            canonical_strings,
            args,
        )
        all_results[dataset] = result

        elapsed = time.perf_counter() - t_start
        logger.info("Dataset %s completed in %.1fs", dataset, elapsed)

    # Cross-dataset outputs
    if args.plot and all_results:
        logger.info("Generating figures...")
        _plot_main_figure(all_results, args.output_dir)
        for ds_name, res in all_results.items():
            _plot_scaling_loglog(ds_name, res, args.output_dir)

    if args.table and all_results:
        logger.info("Generating table...")
        _generate_summary_table(all_results, args.output_dir)

    # Save cross-dataset summary
    summary = {
        "datasets": list(all_results.keys()),
        "per_dataset": {ds: res["stats"] for ds, res in all_results.items()},
    }
    with open(os.path.join(stats_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("All done. Results in: %s", args.output_dir)


if __name__ == "__main__":
    main()
