"""Dual canonical string computation (exhaustive + greedy-min).

For each graph G:
  - exhaustive: canonical_string(G) from isalgraph.core.canonical
  - greedy-min: min over all starting nodes of GraphToString(G, v)
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CanonicalResult:
    """Result of dual canonical computation for one graph."""

    graph_id: str
    n_nodes: int
    n_edges: int
    density: float
    exhaustive_string: str | None
    exhaustive_length: int
    exhaustive_time_s: float
    greedy_string: str | None
    greedy_length: int
    greedy_time_s: float
    greedy_best_start_node: int
    strings_identical: bool | None
    length_gap: int | None
    levenshtein_between_methods: int | None
    speedup: float | None


def _nx_to_sparse_graph(nx_graph: nx.Graph):
    """Convert NetworkX graph to SparseGraph without pickling issues.

    Returns:
        SparseGraph instance.
    """
    from isalgraph.adapters.networkx_adapter import NetworkXAdapter

    adapter = NetworkXAdapter()
    return adapter.from_external(nx_graph, directed=False)


def _reconstruct_sparse_graph(
    n_nodes: int,
    edges: list[tuple[int, int]],
    directed: bool,
):
    """Reconstruct a SparseGraph from serialized data (for subprocess).

    Args:
        n_nodes: Number of nodes.
        edges: List of (src, tgt) edge tuples.
        directed: Whether graph is directed.

    Returns:
        SparseGraph instance.
    """
    from isalgraph.core.sparse_graph import SparseGraph

    sg = SparseGraph(max_nodes=n_nodes, directed_graph=directed)
    for _ in range(n_nodes):
        sg.add_node()
    for src, tgt in edges:
        sg.add_edge(src, tgt)
    return sg


def _serialize_nx_graph(nx_graph: nx.Graph) -> tuple[int, list[tuple[int, int]], bool]:
    """Serialize a NetworkX graph for subprocess transfer.

    Returns:
        (n_nodes, edges, directed) tuple.
    """
    n = nx_graph.number_of_nodes()
    edges = list(nx_graph.edges())
    return (n, edges, False)


def _compute_dual_worker(
    args: tuple,
) -> dict:
    """Worker for ProcessPoolExecutor.

    Args:
        args: (graph_id, n_nodes, edges, directed, timeout_s) tuple.

    Returns:
        Dict with CanonicalResult fields.
    """
    graph_id, n_nodes, edges, directed, timeout_s = args
    return _compute_dual_from_serialized(graph_id, n_nodes, edges, directed, timeout_s)


def _compute_dual_from_serialized(
    graph_id: str,
    n_nodes: int,
    edges: list[tuple[int, int]],
    directed: bool,
    timeout_s: float,
) -> dict:
    """Compute both methods from serialized graph data."""
    from isalgraph.core.canonical import canonical_string
    from isalgraph.core.canonical import levenshtein as _lev
    from isalgraph.core.graph_to_string import GraphToString

    sg = _reconstruct_sparse_graph(n_nodes, edges, directed)
    n = sg.node_count()
    n_edges = sg.logical_edge_count()
    density = n_edges / (n * (n - 1) / 2) if n > 1 else 0.0

    # ---- Exhaustive canonical ----
    exhaustive_str: str | None = None
    exhaustive_time = 0.0
    t0 = time.perf_counter()
    try:
        exhaustive_str = canonical_string(sg)
        exhaustive_time = time.perf_counter() - t0
    except (ValueError, RuntimeError) as e:
        exhaustive_time = time.perf_counter() - t0
        logging.getLogger(__name__).warning(
            "Exhaustive failed for %s (%d nodes): %s", graph_id, n, e
        )

    # ---- Greedy-min (all starting nodes) ----
    t0 = time.perf_counter()
    greedy_results: list[tuple[int, int, str]] = []
    for v in range(n):
        try:
            gts = GraphToString(sg)
            s, _ = gts.run(initial_node=v)
            greedy_results.append((v, len(s), s))
        except (ValueError, RuntimeError):
            continue
    greedy_time = time.perf_counter() - t0

    if greedy_results:
        # Sort by (length, string) to get lexmin among shortest
        greedy_results.sort(key=lambda x: (x[1], x[2]))
        best_start, best_len, best_str = greedy_results[0]
    else:
        best_start, best_len, best_str = -1, -1, None

    # ---- Comparison ----
    strings_identical = None
    length_gap = None
    lev_between = None
    speedup = None

    if exhaustive_str is not None and best_str is not None:
        strings_identical = exhaustive_str == best_str
        length_gap = best_len - len(exhaustive_str)
        lev_between = _lev(exhaustive_str, best_str)

    if greedy_time > 0:
        speedup = round(exhaustive_time / greedy_time, 1)

    return {
        "graph_id": graph_id,
        "n_nodes": n,
        "n_edges": n_edges,
        "density": round(density, 4),
        "exhaustive_string": exhaustive_str,
        "exhaustive_length": len(exhaustive_str) if exhaustive_str is not None else -1,
        "exhaustive_time_s": round(exhaustive_time, 4),
        "greedy_string": best_str,
        "greedy_length": best_len,
        "greedy_time_s": round(greedy_time, 4),
        "greedy_best_start_node": best_start,
        "strings_identical": strings_identical,
        "length_gap": length_gap,
        "levenshtein_between_methods": lev_between,
        "speedup": speedup,
    }


def compute_dual_canonical(
    nx_graph: nx.Graph,
    graph_id: str,
    timeout_s: float = 600.0,
) -> CanonicalResult:
    """Compute exhaustive canonical and greedy-min for one graph.

    Args:
        nx_graph: Input graph.
        graph_id: Graph identifier.
        timeout_s: Per-graph timeout for exhaustive search.

    Returns:
        CanonicalResult with both methods' strings and comparison.
    """
    n_nodes, edges, directed = _serialize_nx_graph(nx_graph)
    result_dict = _compute_dual_from_serialized(graph_id, n_nodes, edges, directed, timeout_s)
    return CanonicalResult(**result_dict)


def compute_all_canonical(
    graphs: list[nx.Graph],
    graph_ids: list[str],
    n_workers: int = 1,
    timeout_s: float = 600.0,
    checkpoint_path: str | None = None,
    checkpoint_interval: int = 100,
) -> list[CanonicalResult]:
    """Compute dual canonical strings for all graphs.

    Args:
        graphs: List of NetworkX graphs.
        graph_ids: Graph identifiers.
        n_workers: Number of parallel workers.
        timeout_s: Per-graph timeout for exhaustive.
        checkpoint_path: Path for checkpoint JSON file.
        checkpoint_interval: Save every N graphs.

    Returns:
        List of CanonicalResult (same order as input).
    """
    n = len(graphs)

    # Load checkpoint
    completed: dict[str, dict] = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            checkpoint_data = json.load(f)
        completed = {r["graph_id"]: r for r in checkpoint_data.get("results", [])}
        logger.info("Resumed from checkpoint: %d/%d graphs", len(completed), n)

    # Build work items
    work_items: list[tuple[int, tuple]] = []
    for i, (g, gid) in enumerate(zip(graphs, graph_ids, strict=True)):
        if gid in completed:
            continue
        n_nodes, edges, directed = _serialize_nx_graph(g)
        work_items.append((i, (gid, n_nodes, edges, directed, timeout_s)))

    logger.info("Computing dual canonical: %d remaining (of %d total)", len(work_items), n)
    t0 = time.perf_counter()

    def _save_checkpoint():
        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            with open(checkpoint_path, "w") as f:
                json.dump({"results": list(completed.values())}, f)

    if n_workers <= 1:
        for count, (_i, args) in enumerate(work_items, 1):
            result_dict = _compute_dual_worker(args)
            completed[result_dict["graph_id"]] = result_dict

            if count % checkpoint_interval == 0:
                _save_checkpoint()
            if count % max(1, len(work_items) // 20) == 0:
                logger.info(
                    "  %d/%d graphs (%.1f%%)",
                    len(completed),
                    n,
                    100.0 * len(completed) / n,
                )
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for _i, args in work_items:
                fut = executor.submit(_compute_dual_worker, args)
                futures[fut] = args[0]  # graph_id

            for done_count, fut in enumerate(as_completed(futures), 1):
                gid = futures[fut]
                try:
                    result_dict = fut.result(timeout=timeout_s + 60)
                    completed[result_dict["graph_id"]] = result_dict
                except TimeoutError:
                    logger.warning("Timeout for graph %s", gid)
                    completed[gid] = {
                        "graph_id": gid,
                        "n_nodes": -1,
                        "n_edges": -1,
                        "density": 0.0,
                        "exhaustive_string": None,
                        "exhaustive_length": -1,
                        "exhaustive_time_s": timeout_s,
                        "greedy_string": None,
                        "greedy_length": -1,
                        "greedy_time_s": 0.0,
                        "greedy_best_start_node": -1,
                        "strings_identical": None,
                        "length_gap": None,
                        "levenshtein_between_methods": None,
                        "speedup": None,
                    }
                except Exception:
                    logger.exception("Worker failed for graph %s", gid)

                if done_count % checkpoint_interval == 0:
                    _save_checkpoint()
                if done_count % max(1, len(work_items) // 20) == 0:
                    logger.info(
                        "  %d/%d graphs (%.1f%%)",
                        len(completed),
                        n,
                        100.0 * len(completed) / n,
                    )

    _save_checkpoint()

    elapsed = time.perf_counter() - t0
    logger.info("Canonical computation complete: %d graphs in %.1fs", n, elapsed)

    # Return in input order
    results: list[CanonicalResult] = []
    for gid in graph_ids:
        if gid in completed:
            results.append(CanonicalResult(**completed[gid]))
        else:
            logger.warning("Missing result for %s", gid)

    return results


def save_canonical_strings(
    results: list[CanonicalResult],
    method: str,
    dataset_name: str,
    n_max_filter: int,
    output_path: str,
) -> None:
    """Save canonical strings to JSON.

    Args:
        results: List of CanonicalResult.
        method: 'exhaustive' or 'greedy'.
        dataset_name: Dataset name.
        n_max_filter: Node-count filter used.
        output_path: Output JSON path.
    """
    is_exhaustive = method == "exhaustive"

    strings_dict: dict[str, dict] = {}
    lengths: list[int] = []
    times: list[float] = []
    n_timeout = 0

    for r in results:
        s = r.exhaustive_string if is_exhaustive else r.greedy_string
        length = r.exhaustive_length if is_exhaustive else r.greedy_length
        t = r.exhaustive_time_s if is_exhaustive else r.greedy_time_s

        if s is None:
            n_timeout += 1
            continue

        strings_dict[r.graph_id] = {
            "string": s,
            "length": length,
            "time_s": t,
        }
        lengths.append(length)
        times.append(t)

    lengths_arr = np.array(lengths) if lengths else np.array([0])
    times_arr = np.array(times) if times else np.array([0.0])

    output = {
        "dataset": dataset_name,
        "method": method,
        "n_max_filter": n_max_filter,
        "n_graphs": len(strings_dict),
        "strings": strings_dict,
        "stats": {
            "mean_length": round(float(np.mean(lengths_arr)), 1),
            "median_length": int(np.median(lengths_arr)),
            "std_length": round(float(np.std(lengths_arr)), 1),
            "max_length": int(np.max(lengths_arr)),
            "min_length": int(np.min(lengths_arr)),
            "mean_time_s": round(float(np.mean(times_arr)), 4),
            "median_time_s": round(float(np.median(times_arr)), 4),
            "max_time_s": round(float(np.max(times_arr)), 4),
            "total_time_s": round(float(np.sum(times_arr)), 2),
            "n_timeout_exclusions": n_timeout,
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(
        "Saved %s canonical strings to %s (%d graphs, %d timeouts)",
        method,
        output_path,
        len(strings_dict),
        n_timeout,
    )
