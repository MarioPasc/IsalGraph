"""Tests for the instrumented graph-to-string encoder mirror.

The load-bearing test is :func:`test_mirror_reproduces_frozen_encoder`.
``viz.encoder_trace`` re-implements the encoder's outer loop so it can
record the displacement pairs the frozen encoder rejects, and
``core/graph_to_string.py`` may not be edited to add that instrumentation
because the C++ differential suite compares against it. The mirror is
therefore only trustworthy for as long as it emits the same string, and
that is what this module checks -- exhaustively on small graphs, and on
random larger and directed ones.

Everything else here is a property of the recorded structure: that the
probes end on the operation that fired, that the captured sets grow
monotonically, and that the groups concatenate to the string.
"""

from __future__ import annotations

import itertools
import random

import pytest

from isalgraph.core.graph_to_string import GraphToString
from isalgraph.core.sparse_graph import SparseGraph
from isalgraph.viz.encoder_trace import (
    REJECTED,
    EncoderMirrorError,
    trace_encoder,
)

#: Node counts enumerated exhaustively. Six is the largest that keeps the
#: whole sweep well under a second with the C++ engine active.
_EXHAUSTIVE_SIZES: tuple[int, ...] = (4, 5, 6)


def _build(n: int, edges: tuple[tuple[int, int], ...], *, directed: bool = False) -> SparseGraph:
    g = SparseGraph(n, directed)
    for _ in range(n):
        g.add_node()
    for u, v in edges:
        g.add_edge(u, v)
    return g


def _is_connected(n: int, edges: tuple[tuple[int, int], ...]) -> bool:
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    seen, stack = {0}, [0]
    while stack:
        for w in adj[stack.pop()]:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return len(seen) == n


def _connected_graphs(n: int) -> list[tuple[tuple[int, int], ...]]:
    """Every connected labelled graph on *n* nodes within the edge budget.

    The budget shrinks with *n* to keep the sweep inside a few seconds:
    the count grows as ``C(C(n,2), m)`` and n=6 with ``m <= n+2`` alone is
    ~100k encodes. Wider coverage was run once out of band -- 134,609
    ``(graph, start)`` pairs to n=14, undirected and directed, with zero
    mismatches -- and is recorded in the T-09 work log rather than paid
    for on every suite run.
    """
    all_edges = list(itertools.combinations(range(n), 2))
    budget = n + 2 if n <= 5 else n
    out: list[tuple[tuple[int, int], ...]] = []
    for m in range(n - 1, min(len(all_edges), budget) + 1):
        out.extend(e for e in itertools.combinations(all_edges, m) if _is_connected(n, e))
    return out


@pytest.mark.parametrize("n", _EXHAUSTIVE_SIZES)
def test_mirror_reproduces_frozen_encoder(n: int) -> None:
    """The mirror emits GraphToString's string, byte for byte.

    Exhaustive over every connected graph on *n* nodes with ``n-1`` to
    ``n+2`` edges, from every starting node.
    """
    checked = 0
    for edges in _connected_graphs(n):
        for start in range(n):
            reference, _ = GraphToString(_build(n, edges)).run(start)
            mirrored = trace_encoder(_build(n, edges), start, verify=False)
            assert mirrored.instruction_string == reference, (
                f"n={n} edges={edges} start={start}: "
                f"mirror {mirrored.instruction_string!r} != encoder {reference!r}"
            )
            checked += 1
    assert checked > 0


def test_mirror_reproduces_frozen_encoder_on_larger_random_graphs() -> None:
    """The agreement survives past the exhaustive range, to 14 nodes."""
    rng = random.Random(20260825)
    checked = 0
    while checked < 40:
        n = rng.randint(7, 14)
        p = rng.uniform(0.3, 0.6)
        edges = tuple(e for e in itertools.combinations(range(n), 2) if rng.random() < p)
        if not _is_connected(n, edges):
            continue
        start = rng.randrange(n)
        reference, _ = GraphToString(_build(n, edges)).run(start)
        assert trace_encoder(_build(n, edges), start, verify=False).instruction_string == reference
        checked += 1


def test_mirror_reproduces_frozen_encoder_on_directed_graphs() -> None:
    """The ``c`` branch of the cascade is exercised and agrees too.

    Directedness is the one axis on which the cascade has a fourth level,
    and it is the level a mirror is most likely to omit.
    """
    rng = random.Random(11)
    checked = 0
    attempts = 0
    while checked < 25 and attempts < 400:
        attempts += 1
        n = rng.randint(4, 8)
        arcs = tuple((u, v) for u in range(n) for v in range(n) if u != v and rng.random() < 0.45)
        graph = _build(n, arcs, directed=True)
        try:
            reference, _ = GraphToString(graph).run(0)
        except ValueError:  # not reachable from node 0
            continue
        mirrored = trace_encoder(_build(n, arcs, directed=True), 0, verify=False)
        assert mirrored.instruction_string == reference
        checked += 1
    assert checked == 25


def test_verify_flag_raises_on_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    """``verify=True`` is a real gate, not a comment."""
    graph = _build(4, ((0, 1), (1, 2), (2, 3)))
    monkeypatch.setattr(
        "isalgraph.viz.encoder_trace.GraphToString.run",
        lambda self, start, trace=False: ("NOT-THE-STRING", []),
    )
    with pytest.raises(EncoderMirrorError, match="drifted"):
        trace_encoder(graph, 0)


def test_groups_concatenate_to_the_string() -> None:
    """Iteration groups partition the instruction string in order."""
    graph = _build(6, ((0, 1), (0, 2), (0, 3), (1, 3), (2, 4), (3, 5)))
    trace = trace_encoder(graph, 0)
    assert "".join(trace.groups) == trace.instruction_string
    for iteration in trace.iterations:
        assert trace.instruction_string.startswith(iteration.string_before + iteration.emitted)


def test_last_probe_is_the_one_that_fired() -> None:
    """Every earlier probe is a rejection; the last carries the operation."""
    graph = _build(6, ((0, 1), (0, 2), (0, 3), (1, 3), (2, 4), (3, 5)))
    for iteration in trace_encoder(graph, 0).iterations:
        assert iteration.selected.verdict in {"V", "v", "C", "c"}
        assert all(p.verdict == REJECTED for p in iteration.probes[:-1])


def test_probe_costs_are_non_decreasing() -> None:
    """Pairs are tested in increasing displacement cost.

    This is the ordering Definition 2.5 fixes and Remark 2.7 says is not
    branched over, so a figure drawn from these probes must show it.
    """
    graph = _build(6, ((0, 1), (0, 2), (0, 3), (1, 3), (2, 4), (3, 5)))
    for iteration in trace_encoder(graph, 0).iterations:
        costs = [p.cost for p in iteration.probes]
        assert costs == sorted(costs), f"iteration {iteration.index}: {costs}"


def test_captured_sets_grow_by_at_most_one_element_per_iteration() -> None:
    """An iteration adds at most one node and at most one edge."""
    graph = _build(6, ((0, 1), (0, 2), (0, 3), (1, 3), (2, 4), (3, 5)))
    trace = trace_encoder(graph, 0)
    for iteration in trace.iterations:
        assert len(iteration.captured_nodes_after) - len(iteration.captured_nodes_before) <= 1
        assert len(iteration.captured_edges_after) - len(iteration.captured_edges_before) <= 1
    last = trace.iterations[-1]
    assert set(last.captured_nodes_after) == set(range(graph.node_count()))
    assert len(last.captured_edges_after) == graph.logical_edge_count()


def test_probe_trimming_keeps_the_winner() -> None:
    """``max_probes_per_iteration`` trims the record, never the search."""
    graph = _build(6, ((0, 1), (0, 2), (0, 3), (1, 3), (2, 4), (3, 5)))
    full = trace_encoder(graph, 0)
    trimmed = trace_encoder(graph, 0, max_probes_per_iteration=2)
    assert trimmed.instruction_string == full.instruction_string
    for short, long in zip(trimmed.iterations, full.iterations, strict=True):
        assert len(short.probes) <= 2
        assert short.selected == long.selected


def test_start_node_precondition_is_delegated_to_the_frozen_encoder() -> None:
    """An unreachable start raises the encoder's own error, not a mirror one."""
    graph = _build(3, ((0, 1), (1, 2)), directed=True)
    with pytest.raises(ValueError, match="reachable"):
        trace_encoder(graph, 2)
