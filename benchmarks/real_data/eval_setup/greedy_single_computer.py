# ruff: noqa: E402
"""Greedy-rnd(v0) computation: greedy encoding from a single random starting node.

Extracted from generate_greedy_single.py for reuse in the main eval_setup pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import time

import numpy as np

logger = logging.getLogger(__name__)


def compute_greedy_single(
    graphs: list,
    graph_ids: list[str],
    seed: int,
) -> tuple[list[str | None], list[float]]:
    """Run greedy G2S from a single random starting node per graph.

    For each graph, picks a uniformly random starting node (seeded for
    reproducibility) and runs the greedy G2S algorithm once.

    Args:
        graphs: List of NetworkX graphs.
        graph_ids: Graph identifiers aligned with graphs.
        seed: Random seed for starting node selection.

    Returns:
        (strings, times) lists aligned with input graphs.
    """
    from isalgraph.adapters.networkx_adapter import NetworkXAdapter
    from isalgraph.core.graph_to_string import GraphToString

    adapter = NetworkXAdapter()
    rng = np.random.default_rng(seed)

    strings: list[str | None] = []
    times: list[float] = []

    for i, (nx_graph, gid) in enumerate(zip(graphs, graph_ids, strict=True)):
        sg = adapter.from_external(nx_graph, directed=False)
        n = sg.node_count()
        if n == 0:
            strings.append(None)
            times.append(0.0)
            continue

        # Pick a single random starting node
        v0 = int(rng.integers(0, n))

        t0 = time.perf_counter()
        try:
            gts = GraphToString(sg)
            s, _ = gts.run(initial_node=v0)
            elapsed = time.perf_counter() - t0
            strings.append(s)
            times.append(elapsed)
        except (ValueError, RuntimeError) as e:
            elapsed = time.perf_counter() - t0
            logger.warning("Encoding failed for %s (v0=%d): %s", gid, v0, e)
            strings.append(None)
            times.append(elapsed)

        if (i + 1) % 200 == 0:
            logger.info("  Encoded %d/%d graphs", i + 1, len(graphs))

    return strings, times


def save_greedy_single_strings(
    graph_ids: list[str],
    strings: list[str | None],
    times: list[float],
    dataset_name: str,
    n_max_filter: int,
    output_path: str,
) -> None:
    """Save greedy-single strings in the same JSON format as greedy/exhaustive.

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
        "method": "greedy_single",
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
    logger.info("Saved greedy-single strings to %s (%d graphs)", output_path, len(strings_dict))
