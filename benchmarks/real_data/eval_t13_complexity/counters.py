"""CLI for the T-13 operation counters (schema ``t13c.1``).

Reads a JSON Lines *spec file* describing graphs and writes one JSON Lines row
per ``(graph, encoder)`` with the realised operation counts of
:mod:`benchmarks.real_data.eval_t13_complexity.instrumented`, together with a
``parity_ok`` flag that is ``True`` only when the instrumented mirror
reproduced the frozen pure-Python reference byte for byte.

Spec-file contract
------------------
Each input row is a JSON object. Two fields are required and carry the graph
explicitly, so this module never has to import a peer's graph builder:

``n``
    Node count; nodes are ``0 .. n-1``.
``edges``
    List of ``[u, v]`` pairs.

Everything else is optional provenance, copied verbatim into the output row:
``source``, ``family``, ``n_target``, ``replicate``, ``dataset``,
``graph_index``, ``directed`` (default ``false``) and ``encoders`` (default all
three).

Usage
-----
``python -m benchmarks.eval_t13_complexity.counters --spec-file S --out O``

``--self-test K`` replaces the spec file with a deterministic pool of ``K``
connected graphs per order, so the CLI is runnable without a spec file.

Encoder values, and why there are two greedy ones
-------------------------------------------------
``encoder`` takes one of ``"greedy_single"``, ``"greedy_min"``, ``"canonical"``,
``"pruned"``. The two greedy values price different objects and the distinction
travels in the data rather than in the invocation, because a consumer asserting
``frames == m`` on the wrong one gets a wrong answer with no error:

``greedy_single``
    One greedy encode from node ``0``. ``frames == m``.
``greedy_min``
    The whole ``GreedyMinG2S`` unit -- one encode per start node, lexmin
    shortest kept. ``frames == n * m``. This is what the registered
    ``isalgraph_greedy`` arm times, so counts and timings price the same object.

``--greedy-mode {min,single,both}`` selects which greedy rows are emitted.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmarks.real_data.eval_t13_complexity.instrumented import (
    OperationCounts,
    canonical_counts,
    greedy_counts,
    greedy_min_counts,
    pruned_counts,
)
from isalgraph.core.canonical import canonical_string
from isalgraph.core.canonical_pruned import pruned_canonical_string
from isalgraph.core.graph_to_string import GraphToString
from isalgraph.core.sparse_graph import SparseGraph

SCHEMA_VERSION = "t13c.1"
ENCODERS: tuple[str, ...] = ("greedy_single", "greedy_min", "canonical", "pruned")
DEFAULT_ENCODERS: tuple[str, ...] = ("greedy_min", "canonical", "pruned")
GREEDY_BY_MODE: dict[str, tuple[str, ...]] = {
    "min": ("greedy_min",),
    "single": ("greedy_single",),
    "both": ("greedy_single", "greedy_min"),
}

LOGGER = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_ENCODERS",
    "ENCODERS",
    "GREEDY_BY_MODE",
    "SCHEMA_VERSION",
    "count_row",
    "random_connected_graphs",
    "reference_string",
    "to_sparse",
]


# ----------------------------------------------------------------------
# Graph plumbing
# ----------------------------------------------------------------------


def to_sparse(n: int, edges: Sequence[Sequence[int]], *, directed: bool = False) -> SparseGraph:
    """Build a :class:`SparseGraph` on ``0 .. n-1`` from an explicit edge list.

    Args:
        n: Node count.
        edges: Iterable of ``(u, v)`` pairs.
        directed: Whether the graph is directed.

    Returns:
        The constructed graph.
    """
    sg = SparseGraph(max_nodes=max(n, 1), directed_graph=directed)
    for _ in range(n):
        sg.add_node()
    for u, v in edges:
        sg.add_edge(int(u), int(v))
    return sg


def random_connected_graphs(
    *,
    seed: int,
    sizes: Sequence[int],
    per_size: int,
    p_min: float = 0.15,
    p_max: float = 0.95,
    max_edges: int | None = None,
) -> Iterator[tuple[int, list[tuple[int, int]]]]:
    """Yield a deterministic pool of connected simple graphs.

    Graphs are drawn as ``G(n, p)`` with ``p`` sampled uniformly from
    ``[p_min, p_max]`` and rejected unless connected. Node labels are the
    identity permutation, so the pool exercises the ``set``-iteration-order
    dependence of the greedy encoder rather than hiding it.

    Args:
        seed: Seed of the single :class:`random.Random` stream driving the pool.
        sizes: Node counts to draw.
        per_size: Accepted graphs per node count.
        p_min: Lower edge probability.
        p_max: Upper edge probability.
        max_edges: If given, reject graphs with more than this many edges --
            used to keep the exhaustive canonical arm tractable.

    Yields:
        Tuples ``(n, edges)``.
    """
    rng = random.Random(seed)
    for n in sizes:
        accepted = 0
        attempts = 0
        while accepted < per_size:
            attempts += 1
            if attempts > 400 * per_size:
                raise RuntimeError(f"could not draw {per_size} connected graphs at n={n}")
            p = rng.uniform(p_min, p_max)
            edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
            if max_edges is not None and len(edges) > max_edges:
                continue
            if not _is_connected(n, edges):
                continue
            accepted += 1
            yield n, edges


def _is_connected(n: int, edges: Sequence[tuple[int, int]]) -> bool:
    """Return whether the undirected graph on ``0 .. n-1`` is connected."""
    if n <= 1:
        return True
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == n


# ----------------------------------------------------------------------
# Reference and counted encodes
# ----------------------------------------------------------------------


def reference_string(graph: SparseGraph, encoder: str) -> str:
    """Return the frozen **pure-Python** reference string for *encoder*.

    The imports are taken from ``isalgraph.core.*`` rather than from the
    top-level package, so the C++ engine is never consulted and the mirror is
    compared against the reference it was transcribed from.

    Args:
        graph: The graph to encode.
        encoder: One of the values in :data:`ENCODERS`.

    Returns:
        The reference string.

    Raises:
        ValueError: If *encoder* is not recognised.
    """
    if encoder == "greedy_single":
        return GraphToString(graph).run(0)[0]
    if encoder == "greedy_min":
        return _greedy_min_reference(graph)
    if encoder == "canonical":
        return canonical_string(graph)
    if encoder == "pruned":
        return pruned_canonical_string(graph)
    raise ValueError(f"unknown encoder: {encoder!r}")


def _greedy_min_reference(graph: SparseGraph) -> str:
    """Pure-Python ``GreedyMinG2S``: lexmin shortest over every start node."""
    n = graph.node_count()
    if n == 0 or (n == 1 and graph.logical_edge_count() == 0):
        return ""
    results: list[tuple[int, str]] = []
    for v in range(n):
        try:
            s = GraphToString(graph).run(v)[0]
        except (ValueError, RuntimeError):
            continue
        results.append((len(s), s))
    if not results:
        raise ValueError("No starting node can reach all other nodes.")
    results.sort()
    return results[0][1]


def count_row(graph: SparseGraph, encoder: str) -> tuple[str, OperationCounts]:
    """Run the instrumented mirror for *encoder*.

    Args:
        graph: The graph to encode.
        encoder: One of the values in :data:`ENCODERS`.

    Returns:
        Tuple ``(string, counts)``.

    Raises:
        ValueError: If *encoder* is not recognised.
    """
    if encoder == "greedy_single":
        return greedy_counts(graph, 0)
    if encoder == "greedy_min":
        return greedy_min_counts(graph)
    if encoder == "canonical":
        return canonical_counts(graph)
    if encoder == "pruned":
        return pruned_counts(graph)
    raise ValueError(f"unknown encoder: {encoder!r}")


# ----------------------------------------------------------------------
# Rows
# ----------------------------------------------------------------------


def _row(spec: dict[str, Any], encoder: str) -> dict[str, Any]:
    """Build one output row for ``(spec, encoder)``."""
    n = int(spec["n"])
    edges = [(int(u), int(v)) for u, v in spec["edges"]]
    directed = bool(spec.get("directed", False))
    graph = to_sparse(n, edges, directed=directed)

    string, counts = count_row(graph, encoder)
    reference = reference_string(graph, encoder)

    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": spec.get("source"),
        "family": spec.get("family"),
        "n_target": spec.get("n_target"),
        "replicate": spec.get("replicate"),
        "dataset": spec.get("dataset"),
        "graph_index": spec.get("graph_index"),
        "n": n,
        "m": graph.logical_edge_count(),
        "encoder": encoder,
    }
    row.update(asdict(counts))
    row["parity_ok"] = string == reference
    return row


def _read_specs(path: Path) -> list[dict[str, Any]]:
    """Read a JSON Lines spec file, skipping blank lines."""
    specs: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                specs.append(json.loads(line))
    return specs


def _self_test_specs(*, seed: int, per_size: int) -> list[dict[str, Any]]:
    """Deterministic specs for ``--self-test``, sparse enough to stay tractable."""
    specs: list[dict[str, Any]] = []
    pool = random_connected_graphs(
        seed=seed, sizes=(4, 5, 6, 7), per_size=per_size, p_min=0.25, p_max=0.75
    )
    for index, (n, edges) in enumerate(pool):
        specs.append(
            {
                "source": "constructed",
                "family": "self_test_gnp",
                "n_target": n,
                "replicate": index,
                "dataset": None,
                "graph_index": index,
                "n": n,
                "edges": [list(e) for e in edges],
                "directed": False,
            }
        )
    return specs


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.eval_t13_complexity.counters",
        description=(
            "Emit T-13 operation counts (schema t13c.1) for the IsalGraph encoders, "
            "each row parity-checked against the frozen pure-Python reference."
        ),
    )
    parser.add_argument("--spec-file", type=Path, help="JSON Lines graph specs")
    parser.add_argument("--out", type=Path, required=True, help="JSON Lines output path")
    parser.add_argument(
        "--encoders",
        default=",".join(DEFAULT_ENCODERS),
        help=f"comma-separated subset of {','.join(ENCODERS)}",
    )
    parser.add_argument(
        "--self-test",
        type=int,
        metavar="K",
        help="ignore --spec-file and use K deterministic connected graphs per order",
    )
    parser.add_argument("--seed", type=int, default=13, help="seed for --self-test")
    parser.add_argument(
        "--greedy-mode",
        choices=tuple(GREEDY_BY_MODE),
        default=None,
        help=(
            "which greedy rows to emit, overriding the greedy entries of --encoders: "
            "'min' -> greedy_min (the whole GreedyMinG2S unit, frames == n * m), "
            "'single' -> greedy_single (one encode from node 0, frames == m), "
            "'both' -> both rows"
        ),
    )
    return parser


def _resolve_encoders(spec: str, greedy_mode: str | None) -> list[str]:
    """Resolve the encoder list, applying ``--greedy-mode`` if it was given."""
    encoders = [e.strip() for e in spec.split(",") if e.strip()]
    if greedy_mode is None:
        return encoders
    others = [e for e in encoders if not e.startswith("greedy")]
    return [*GREEDY_BY_MODE[greedy_mode], *others]


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv[1:]``.

    Returns:
        Process exit status; non-zero if any row failed its parity check.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    encoders = _resolve_encoders(args.encoders, args.greedy_mode)
    unknown = sorted(set(encoders) - set(ENCODERS))
    if unknown:
        LOGGER.error("unknown encoders: %s", unknown)
        return 2

    if args.self_test is not None:
        specs = _self_test_specs(seed=args.seed, per_size=args.self_test)
    elif args.spec_file is not None:
        specs = _read_specs(args.spec_file)
    else:
        LOGGER.error("one of --spec-file or --self-test is required")
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    written = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for spec in specs:
            for encoder in encoders:
                row = _row(spec, encoder)
                if not row["parity_ok"]:
                    failures += 1
                    LOGGER.error("parity failure: %s", {k: row[k] for k in ("n", "m", "encoder")})
                fh.write(json.dumps(row, sort_keys=True) + "\n")
                written += 1

    LOGGER.info("wrote %d rows to %s (%d parity failures)", written, args.out, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
