"""Trace reconstruction for the visualisation layer.

Reads a JSON trace written by :func:`isalgraph.core.trace.dump_trace` and
rebuilds the final :class:`SparseGraph`, so a caller can draw a stored
trace without re-running the algorithm that produced it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from isalgraph.core.sparse_graph import SparseGraph
from isalgraph.core.trace import AlgorithmTrace, load_trace


def graph_from_edgelist(payload: dict[str, Any]) -> SparseGraph:
    """Rebuild a :class:`SparseGraph` from a trace envelope's ``final_graph``.

    Args:
        payload: Mapping with keys ``"n_nodes"``, ``"edges"`` and
            ``"directed"``, as produced by
            :func:`isalgraph.core.trace.graph_to_dict`.

    Returns:
        A graph with ``n_nodes`` contiguous nodes and every listed edge.
    """
    n_nodes = int(payload["n_nodes"])
    directed = bool(payload.get("directed", False))
    edges = payload.get("edges", [])

    graph = SparseGraph(max(n_nodes, 1), directed)
    for _ in range(n_nodes):
        graph.add_node()
    for u, v in edges:
        graph.add_edge(int(u), int(v))
    return graph


def load_trace_for_viz(path: str | Path) -> tuple[AlgorithmTrace, SparseGraph]:
    """Load *path* and return the trace together with its final graph."""
    trace = load_trace(path)
    return trace, graph_from_edgelist(trace.final_graph)


__all__ = ["graph_from_edgelist", "load_trace_for_viz"]
