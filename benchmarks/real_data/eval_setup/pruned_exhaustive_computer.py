# ruff: noqa: E402
"""Pruned-exhaustive computation: canonical encoding with triplet pruning.

Follows the same pattern as ``greedy_single_computer.py`` for pipeline integration.
"""

from __future__ import annotations

import json
import logging
import os
import time

import numpy as np

logger = logging.getLogger(__name__)


def compute_pruned_exhaustive(
    graphs: list,
    graph_ids: list[str],
    timeout_s: float = 600.0,
) -> tuple[list[str | None], list[float]]:
    """Run pruned exhaustive canonical encoding per graph.

    For each graph, converts to SparseGraph and calls
    ``pruned_canonical_string``.

    Args:
        graphs: List of NetworkX graphs.
        graph_ids: Graph identifiers aligned with graphs.
        timeout_s: Per-graph timeout in seconds.

    Returns:
        (strings, times) lists aligned with input graphs.
    """
    from isalgraph.adapters.networkx_adapter import NetworkXAdapter
    from isalgraph.core.canonical_pruned import pruned_canonical_string

    adapter = NetworkXAdapter()

    strings: list[str | None] = []
    times: list[float] = []

    for i, (nx_graph, gid) in enumerate(zip(graphs, graph_ids, strict=True)):
        sg = adapter.from_external(nx_graph, directed=False)
        n = sg.node_count()
        if n == 0:
            strings.append(None)
            times.append(0.0)
            continue

        t0 = time.perf_counter()
        try:
            s = pruned_canonical_string(sg)
            elapsed = time.perf_counter() - t0

            if elapsed > timeout_s:
                logger.warning(
                    "Pruned exhaustive slow for %s (n=%d): %.1fs",
                    gid,
                    n,
                    elapsed,
                )

            strings.append(s)
            times.append(elapsed)
        except (ValueError, RuntimeError) as e:
            elapsed = time.perf_counter() - t0
            logger.warning("Pruned exhaustive failed for %s (n=%d): %s", gid, n, e)
            strings.append(None)
            times.append(elapsed)

        if (i + 1) % 200 == 0:
            logger.info("  Pruned exhaustive: %d/%d graphs", i + 1, len(graphs))

    return strings, times


def save_pruned_exhaustive_strings(
    graph_ids: list[str],
    strings: list[str | None],
    times: list[float],
    dataset_name: str,
    n_max_filter: int,
    output_path: str,
) -> None:
    """Save pruned-exhaustive strings in the standard JSON format.

    Args:
        graph_ids: Graph identifiers.
        strings: Encoded strings (None for failures).
        times: Encoding times per graph.
        dataset_name: Dataset name for metadata.
        n_max_filter: Node count filter used.
        output_path: Output JSON path.
    """
    strings_dict: dict[str, dict] = {}
    lengths: list[int] = []
    time_list: list[float] = []

    for gid, s, t in zip(graph_ids, strings, times, strict=True):
        if s is None:
            continue
        strings_dict[gid] = {
            "string": s,
            "length": len(s),
            "time_s": round(t, 4),
        }
        lengths.append(len(s))
        time_list.append(t)

    lengths_arr = np.array(lengths) if lengths else np.array([0])
    times_arr = np.array(time_list) if time_list else np.array([0.0])

    output = {
        "dataset": dataset_name,
        "method": "pruned_exhaustive",
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
            "n_timeout_exclusions": sum(1 for s in strings if s is None),
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info(
        "Saved pruned-exhaustive strings to %s (%d graphs)",
        output_path,
        len(strings_dict),
    )
