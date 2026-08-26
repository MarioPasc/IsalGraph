"""Instrumented mirror of the frozen IsalGraph encoder (T-13 / R3.4b).

Reviewer 3 asks, verbatim, whether the ordered displacement lists are
recomputed at each iteration or precomputed, and for pair scanning, pointer
walking, neighbour checks and canonical backtracking to be accounted for in
the theoretical complexity discussion. ``T-13-design.md`` §2.1 supplies the
derivation; this module supplies the realised operation counts that validate
it.

Design
------
``src/isalgraph/core/`` is **frozen**: it is what the C++ differential suite
compares against, so it is not instrumented in place. Instead this module is a
*mirror* -- a line-by-line transcription of the three frozen encoders with
counter increments interleaved -- exactly the device
``isalgraph/viz/encoder_trace.py`` used for T-09. The mirror shares every data
structure with the reference (``SparseGraph``, ``CircularDoublyLinkedList``,
``generate_pairs_sorted_by_sum``, ``_undo_edge``, ``_undo_node``) so that the
only difference between mirror and reference is the counting.

**Parity is the deliverable, not the counts.** Every public entry point returns
a string that must be byte-identical to the corresponding pure-Python
reference, imported from ``isalgraph.core.*`` so that the engine is never
consulted.

Counting conventions
--------------------
These are the definitions the derivation is checked against. They are fixed
here and must not drift.

``frames``
    Payload instructions (``V``/``v``/``C``/``c``) emitted. For a greedy
    encode this is one per iteration of the ``while`` loop, hence exactly
    ``m``. For the canonical arms it is summed over the whole search tree and
    over every reachable start node, and equals
    ``backtrack_nodes - search_leaves``.
``pair_trials`` / ``scan_depth_total``
    Iterations of the ``for (a, b) in pairs`` loop, including the accepted
    pair, summed over frames. The two fields are equal by construction;
    ``scan_depth_max`` is the per-frame maximum of the same quantity.
``pointer_steps``
    Unit CDLL moves, i.e. executions of the loop body inside
    ``GraphToString._move_pointer`` / ``canonical._walk``. A trial of pair
    ``(a, b)`` costs ``|a|`` steps always and ``|b|`` steps only when control
    actually reaches the secondary walk -- a ``V`` acceptance returns before
    the secondary pointer is moved, and the count reflects that.
``neighbour_checks``
    Adjacency and uninserted-neighbour tests. In the greedy encoder
    ``_find_new_neighbor`` short-circuits, so one check is counted per
    neighbour *examined*; in the canonical arms the candidate list is
    materialised, so one check is counted per neighbour of the node. The
    ``C``/``c`` guards contribute one check per membership test actually
    evaluated (Python's ``and`` short-circuits, and so does the count).
    Triplet precomputation and triplet-key comparisons in the pruned arm are
    **not** counted: they are a fixed ``O(n(n+m))`` preprocessing step plus an
    ``O(|cands|)`` filter, and folding them in would make the pruned and
    unpruned neighbour counts incomparable.
``backtrack_nodes``
    ``_step`` / ``_pruned_step`` invocations entered, including the root call
    for each start node and including terminal calls. ``0`` for greedy, which
    does not recurse.
``search_leaves``
    Invocations that reach the terminal ``nleft <= 0 and eleft <= 0`` branch,
    i.e. complete strings produced by the search. ``0`` for greedy, which
    produces its single string without a search.
``string_length``
    ``len`` of the string actually returned by the entry point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

from isalgraph.core.canonical import (
    _is_reachable,
    _primary_moves,
    _secondary_moves,
    _undo_edge,
    _undo_node,
)
from isalgraph.core.canonical_pruned import compute_structural_triplets
from isalgraph.core.cdll import CircularDoublyLinkedList
from isalgraph.core.graph_to_string import generate_pairs_sorted_by_sum
from isalgraph.core.sparse_graph import SparseGraph

__all__ = [
    "FrameRecord",
    "InstrumentationError",
    "OperationCounts",
    "canonical_counts",
    "canonical_detail",
    "greedy_counts",
    "greedy_detail",
    "greedy_min_counts",
    "pair_generation_work",
    "pruned_counts",
    "pruned_detail",
]


class InstrumentationError(Exception):
    """Raised when the instrumented mirror reaches a state the reference cannot."""


# ----------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OperationCounts:
    """Realised operation counts for one encode.

    Attributes:
        frames: Payload instructions emitted; ``== m`` for a greedy encode.
        pair_trials: ``(a, b)`` pairs examined, summed over frames.
        scan_depth_total: Same as *pair_trials*; kept separate for the
            per-frame maximum below.
        scan_depth_max: Largest per-frame pair-scan depth.
        pointer_steps: Unit CDLL moves executed while trialling and committing.
        neighbour_checks: Adjacency / uninserted-neighbour tests.
        backtrack_nodes: Recursion frames entered; ``0`` for greedy.
        search_leaves: Complete strings produced by the search; ``0`` for greedy.
        string_length: Length of the returned string.
    """

    frames: int
    pair_trials: int
    scan_depth_total: int
    scan_depth_max: int
    pointer_steps: int
    neighbour_checks: int
    backtrack_nodes: int
    search_leaves: int
    string_length: int


@dataclass(frozen=True, slots=True)
class FrameRecord:
    """Per-frame detail, used by the derivation checks of T-13 §2.1.

    Attributes:
        pair_scope: ``M``, the argument passed to
            ``generate_pairs_sorted_by_sum``; the frame therefore generates
            ``(2M + 1) ** 2`` pairs.
        pairs_generated: ``(2M + 1) ** 2``, recorded rather than recomputed.
        pair_trials: ``D_f``, the realised scan depth of this frame.
        pointer_steps: Unit CDLL moves charged to this frame.
        neighbour_checks: Adjacency tests charged to this frame.
        opcode: The payload instruction the frame emitted.
        a: Accepted primary displacement.
        b: Accepted secondary displacement.
        disp_emitted: Movement characters the frame actually appended --
            ``|a|`` for ``V``, ``|b|`` for ``v``, ``|a| + |b|`` for ``C``/``c``.
        n_cands: Uninserted neighbours found at a ``V``/``v`` frame before
            pruning (``0`` at a ``C``/``c`` frame).
        branch_factor: ``b_f``, continuations actually expanded by the frame.
        depth: Recursion depth, ``0`` at the root of a start node's search.
        start_node: The start node whose search this frame belongs to.
    """

    pair_scope: int
    pairs_generated: int
    pair_trials: int
    pointer_steps: int
    neighbour_checks: int
    opcode: str
    a: int
    b: int
    disp_emitted: int
    n_cands: int
    branch_factor: int
    depth: int
    start_node: int


@dataclass(slots=True)
class _Recorder:
    """Mutable accumulator threaded through the mirrored encoders."""

    pair_trials: int = 0
    pointer_steps: int = 0
    neighbour_checks: int = 0
    backtrack_nodes: int = 0
    search_leaves: int = 0
    frames: list[FrameRecord] = field(default_factory=list)

    def finish(self, string_length: int) -> OperationCounts:
        """Freeze the accumulator into an :class:`OperationCounts`."""
        depths = [f.pair_trials for f in self.frames]
        return OperationCounts(
            frames=len(self.frames),
            pair_trials=self.pair_trials,
            scan_depth_total=self.pair_trials,
            scan_depth_max=max(depths) if depths else 0,
            pointer_steps=self.pointer_steps,
            neighbour_checks=self.neighbour_checks,
            backtrack_nodes=self.backtrack_nodes,
            search_leaves=self.search_leaves,
            string_length=string_length,
        )


# ----------------------------------------------------------------------
# Pair generation cost
# ----------------------------------------------------------------------


class _CountingKey:
    """Tuple wrapper that tallies ``__lt__`` calls, for sort-cost measurement."""

    __slots__ = ("key", "tally")

    def __init__(self, key: tuple[int, int, tuple[int, int]], tally: list[int]) -> None:
        self.key = key
        self.tally = tally

    def __lt__(self, other: _CountingKey) -> bool:
        self.tally[0] += 1
        return self.key < other.key


@lru_cache(maxsize=256)
def pair_generation_work(m: int) -> tuple[int, int, float]:
    """Cost of one ``generate_pairs_sorted_by_sum(m)`` call.

    The sorted order depends on *m* alone and never on the graph, which is why
    the C++ engine may memoise it. This function measures the two components
    of the ``Theta(M**2 log M)`` claim of T-13 §2.1: the number of pairs built,
    and the number of key comparisons Timsort performs on exactly the sequence
    the frozen function hands it.

    Args:
        m: The argument the frozen function is called with; ``M`` in the
            derivation.

    Returns:
        Tuple ``(pairs_generated, sort_comparisons, analytic_p_log2_p)`` where
        ``pairs_generated == (2 * m + 1) ** 2``.

    Raises:
        ValueError: If *m* is not positive.
    """
    if m <= 0:
        raise ValueError("m must be a positive integer.")

    pairs: list[tuple[int, int]] = [(a, b) for a in range(-m, m + 1) for b in range(-m, m + 1)]
    tally = [0]
    keys = [_CountingKey((abs(p[0]) + abs(p[1]), abs(p[0]), p), tally) for p in pairs]
    keys.sort()

    n_pairs = len(pairs)
    return n_pairs, tally[0], n_pairs * math.log2(n_pairs)


# ----------------------------------------------------------------------
# Shared mirrored primitives
# ----------------------------------------------------------------------


def _walk_counted(
    cdll: CircularDoublyLinkedList,
    ptr: int,
    steps: int,
    rec: _Recorder,
) -> int:
    """Mirror of ``canonical._walk`` / ``GraphToString._move_pointer``.

    Both frozen implementations execute ``abs(steps)`` unit moves; the count is
    the number of loop-body executions, not the number of calls.
    """
    for _ in range(abs(steps)):
        ptr = cdll.next_node(ptr) if steps > 0 else cdll.prev_node(ptr)
    rec.pointer_steps += abs(steps)
    return ptr


def _reachable_or_raise(graph: SparseGraph, initial_node: int) -> None:
    """Mirror of ``GraphToString._check_reachability`` (not counted)."""
    n = graph.node_count()
    if n <= 1:
        return
    visited: set[int] = set()
    stack: list[int] = [initial_node]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                stack.append(neighbor)
    if len(visited) != n:
        raise ValueError(
            "GraphToString requires all nodes to be reachable from "
            f"initial_node={initial_node} via outgoing edges."
        )


# ----------------------------------------------------------------------
# Greedy mirror -- isalgraph.core.graph_to_string.GraphToString.run
# ----------------------------------------------------------------------


def greedy_detail(graph: SparseGraph, start: int) -> tuple[str, OperationCounts, list[FrameRecord]]:
    """Greedy encode from *start*, with counts and per-frame detail.

    Mirrors ``GraphToString(graph).run(start)[0]`` byte for byte.

    Args:
        graph: The graph to encode.
        start: Index of the starting node in *graph*.

    Returns:
        Tuple ``(string, counts, frames)``.

    Raises:
        ValueError: If *start* is out of range or cannot reach every node.
        InstrumentationError: If no displacement pair admits an operation,
            which the reference reports as ``RuntimeError``.
    """
    if start < 0 or start >= graph.node_count():
        raise ValueError("Initial node out of range")
    _reachable_or_raise(graph, start)

    rec = _Recorder()
    out = ""

    cdll = CircularDoublyLinkedList(graph.max_nodes())
    og = SparseGraph(graph.max_nodes(), graph.directed())
    n0 = og.add_node()
    c0 = cdll.insert_after(-1, n0)
    pri = c0
    sec = c0
    i2o: dict[int, int] = {start: n0}
    o2i: dict[int, int] = {n0: start}

    nleft = graph.node_count() - 1
    eleft = graph.logical_edge_count()

    while nleft > 0 or eleft > 0:
        scope = og.node_count()
        pairs = generate_pairs_sorted_by_sum(scope)

        p0, n0c, t0 = rec.pointer_steps, rec.neighbour_checks, rec.pair_trials
        found = False

        for a, b in pairs:
            rec.pair_trials += 1

            tp = _walk_counted(cdll, pri, a, rec)
            tp_out = cdll.get_value(tp)
            tp_in = o2i[tp_out]

            # -- V --
            if nleft > 0:
                candidate = _first_new_neighbour(graph, tp_in, i2o, rec)
                if candidate is not None:
                    new_out = og.add_node()
                    nleft -= 1
                    i2o[candidate] = new_out
                    o2i[new_out] = candidate
                    og.add_edge(tp_out, new_out)
                    eleft -= 1
                    cdll.insert_after(tp, new_out)
                    out += _primary_moves(a) + "V"
                    pri = tp
                    _push_frame(rec, scope, p0, n0c, t0, "V", a, b, abs(a), 1, 1, 0, start)
                    found = True
                    break

            ts = _walk_counted(cdll, sec, b, rec)
            ts_out = cdll.get_value(ts)
            ts_in = o2i[ts_out]

            # -- v --
            if nleft > 0:
                candidate = _first_new_neighbour(graph, ts_in, i2o, rec)
                if candidate is not None:
                    new_out = og.add_node()
                    nleft -= 1
                    i2o[candidate] = new_out
                    o2i[new_out] = candidate
                    og.add_edge(ts_out, new_out)
                    eleft -= 1
                    cdll.insert_after(ts, new_out)
                    out += _secondary_moves(b) + "v"
                    sec = ts
                    _push_frame(rec, scope, p0, n0c, t0, "v", a, b, abs(b), 1, 1, 0, start)
                    found = True
                    break

            # -- C --
            rec.neighbour_checks += 1
            if ts_in in graph.neighbors(tp_in):
                rec.neighbour_checks += 1
                if ts_out not in og.neighbors(tp_out):
                    og.add_edge(tp_out, ts_out)
                    eleft -= 1
                    out += _primary_moves(a) + _secondary_moves(b) + "C"
                    pri, sec = tp, ts
                    _push_frame(rec, scope, p0, n0c, t0, "C", a, b, abs(a) + abs(b), 0, 1, 0, start)
                    found = True
                    break

            # -- c (directed only) --
            if graph.directed():
                rec.neighbour_checks += 1
                if tp_in in graph.neighbors(ts_in):
                    rec.neighbour_checks += 1
                    if tp_out not in og.neighbors(ts_out):
                        og.add_edge(ts_out, tp_out)
                        eleft -= 1
                        out += _primary_moves(a) + _secondary_moves(b) + "c"
                        pri, sec = tp, ts
                        _push_frame(
                            rec, scope, p0, n0c, t0, "c", a, b, abs(a) + abs(b), 0, 1, 0, start
                        )
                        found = True
                        break

        if not found:
            raise InstrumentationError(
                f"greedy mirror: no valid operation found. Remaining: {nleft} nodes, {eleft} edges."
            )

    return out, rec.finish(len(out)), rec.frames


def greedy_counts(graph: SparseGraph, start: int) -> tuple[str, OperationCounts]:
    """Greedy encode from *start*, with realised operation counts.

    Args:
        graph: The graph to encode.
        start: Index of the starting node in *graph*.

    Returns:
        Tuple ``(string, counts)``. The string is byte-identical to
        ``isalgraph.core.graph_to_string.GraphToString(graph).run(start)[0]``.
    """
    string, counts, _ = greedy_detail(graph, start)
    return string, counts


def greedy_min_counts(graph: SparseGraph) -> tuple[str, OperationCounts]:
    """Greedy-min encode: the ``isalgraph_greedy`` arm's whole unit of work.

    ``GreedyMinG2S`` runs the greedy encoder from *every* start node and keeps
    the lexicographically smallest shortest string, so its cost is the sum over
    starts. This entry point exists so that counts and the timing campaign
    price the same object.

    Args:
        graph: The graph to encode.

    Returns:
        Tuple ``(string, counts)`` with counts summed over every start node
        that reaches the whole graph.

    Raises:
        ValueError: If no start node reaches every other node.
    """
    n = graph.node_count()
    rec = _Recorder()
    if n == 0 or (n == 1 and graph.logical_edge_count() == 0):
        return "", rec.finish(0)

    best: str | None = None
    for v in range(n):
        if not _is_reachable(graph, v):
            continue
        string, _, frames = greedy_detail(graph, v)
        _absorb(rec, string, frames)
        if best is None or (len(string), string) < (len(best), best):
            best = string

    if best is None:
        raise ValueError("No starting node can reach all other nodes.")
    return best, rec.finish(len(best))


def _absorb(rec: _Recorder, _string: str, frames: list[FrameRecord]) -> None:
    """Fold a per-start greedy frame list into an aggregate recorder."""
    for f in frames:
        rec.pair_trials += f.pair_trials
        rec.pointer_steps += f.pointer_steps
        rec.neighbour_checks += f.neighbour_checks
        rec.frames.append(f)


def _first_new_neighbour(
    graph: SparseGraph,
    node: int,
    i2o: dict[int, int],
    rec: _Recorder,
) -> int | None:
    """Mirror of ``GraphToString._find_new_neighbor``.

    Iterates the very same ``set`` object the reference iterates, so the
    slot-order dependence that greedy parity rests on is preserved.
    """
    for neighbor in graph.neighbors(node):
        rec.neighbour_checks += 1
        if neighbor not in i2o:
            return neighbor
    return None


def _push_frame(  # noqa: PLR0913
    rec: _Recorder,
    scope: int,
    p0: int,
    n0c: int,
    t0: int,
    opcode: str,
    a: int,
    b: int,
    disp_emitted: int,
    n_cands: int,
    branch_factor: int,
    depth: int,
    start_node: int,
) -> None:
    """Append a :class:`FrameRecord` built from the recorder deltas."""
    rec.frames.append(
        FrameRecord(
            pair_scope=scope,
            pairs_generated=(2 * scope + 1) ** 2,
            pair_trials=rec.pair_trials - t0,
            pointer_steps=rec.pointer_steps - p0,
            neighbour_checks=rec.neighbour_checks - n0c,
            opcode=opcode,
            a=a,
            b=b,
            disp_emitted=disp_emitted,
            n_cands=n_cands,
            branch_factor=branch_factor,
            depth=depth,
            start_node=start_node,
        )
    )


# ----------------------------------------------------------------------
# Canonical mirrors -- isalgraph.core.canonical / canonical_pruned
# ----------------------------------------------------------------------


@dataclass(slots=True)
class _SearchContext:
    """Immutable-per-encode context for the mirrored canonical search."""

    ig: SparseGraph
    rec: _Recorder
    triplets: list[tuple[int, int, int]] | None
    start_node: int = 0


def _canonical_search(
    graph: SparseGraph,
    *,
    pruned: bool,
) -> tuple[str, OperationCounts, list[FrameRecord]]:
    """Shared driver for the exhaustive and triplet-pruned canonical mirrors."""
    n = graph.node_count()
    rec = _Recorder()
    if n == 0 or (n == 1 and graph.logical_edge_count() == 0):
        return "", rec.finish(0), rec.frames

    triplets = compute_structural_triplets(graph) if pruned else None
    ctx = _SearchContext(ig=graph, rec=rec, triplets=triplets)

    best: str | None = None
    for v in range(n):
        if not _is_reachable(graph, v):
            continue
        ctx.start_node = v

        max_n = graph.max_nodes()
        og = SparseGraph(max_n, graph.directed())
        cdll = CircularDoublyLinkedList(max_n)
        n0 = og.add_node()
        c0 = cdll.insert_after(-1, n0)
        i2o: dict[int, int] = {v: n0}
        o2i: dict[int, int] = {n0: v}

        w = _step_counted(
            ctx,
            og,
            cdll,
            c0,
            c0,
            i2o,
            o2i,
            graph.node_count() - 1,
            graph.logical_edge_count(),
            "",
            0,
        )
        if best is None or (len(w), w) < (len(best), best):
            best = w

    if best is None:
        raise ValueError("No starting node can reach all other nodes.")
    return best, rec.finish(len(best)), rec.frames


def _step_counted(  # noqa: PLR0912, PLR0913, PLR0915
    ctx: _SearchContext,
    og: SparseGraph,
    cdll: CircularDoublyLinkedList,
    pri: int,
    sec: int,
    i2o: dict[int, int],
    o2i: dict[int, int],
    nleft: int,
    eleft: int,
    prefix: str,
    depth: int,
) -> str:
    """Mirror of ``canonical._step`` and ``canonical_pruned._pruned_step``.

    The two frozen functions differ only in the triplet filter applied to the
    candidate list; ``ctx.triplets is None`` selects the unpruned arm.
    """
    ig = ctx.ig
    rec = ctx.rec
    triplets = ctx.triplets
    rec.backtrack_nodes += 1

    if nleft <= 0 and eleft <= 0:
        rec.search_leaves += 1
        return prefix

    scope = og.node_count()
    pairs = generate_pairs_sorted_by_sum(scope)
    p0, n0c, t0 = rec.pointer_steps, rec.neighbour_checks, rec.pair_trials

    for a, b in pairs:
        rec.pair_trials += 1

        tp = _walk_counted(cdll, pri, a, rec)
        tp_out = cdll.get_value(tp)
        tp_in = o2i[tp_out]

        # -- V: primary has an uninserted neighbour --
        if nleft > 0:
            cands = _materialise_candidates(ig, tp_in, i2o, rec)
            if cands:
                expand = _prune(cands, triplets)
                mov = _primary_moves(a)
                _push_frame(
                    rec,
                    scope,
                    p0,
                    n0c,
                    t0,
                    "V",
                    a,
                    b,
                    abs(a),
                    len(cands),
                    len(expand),
                    depth,
                    ctx.start_node,
                )
                best: str | None = None
                for c in expand:
                    new_out = og.add_node()
                    i2o[c] = new_out
                    o2i[new_out] = c
                    og.add_edge(tp_out, new_out)
                    new_cdll = cdll.insert_after(tp, new_out)

                    r = _step_counted(
                        ctx,
                        og,
                        cdll,
                        tp,
                        sec,
                        i2o,
                        o2i,
                        nleft - 1,
                        eleft - 1,
                        prefix + mov + "V",
                        depth + 1,
                    )
                    if best is None or (len(r), r) < (len(best), best):
                        best = r

                    cdll.remove(new_cdll)
                    _undo_edge(og, tp_out, new_out)
                    _undo_node(og)
                    del i2o[c]
                    del o2i[new_out]

                assert best is not None
                return best

        ts = _walk_counted(cdll, sec, b, rec)
        ts_out = cdll.get_value(ts)
        ts_in = o2i[ts_out]

        # -- v: secondary has an uninserted neighbour --
        if nleft > 0:
            cands = _materialise_candidates(ig, ts_in, i2o, rec)
            if cands:
                expand = _prune(cands, triplets)
                mov = _secondary_moves(b)
                _push_frame(
                    rec,
                    scope,
                    p0,
                    n0c,
                    t0,
                    "v",
                    a,
                    b,
                    abs(b),
                    len(cands),
                    len(expand),
                    depth,
                    ctx.start_node,
                )
                best = None
                for c in expand:
                    new_out = og.add_node()
                    i2o[c] = new_out
                    o2i[new_out] = c
                    og.add_edge(ts_out, new_out)
                    new_cdll = cdll.insert_after(ts, new_out)

                    r = _step_counted(
                        ctx,
                        og,
                        cdll,
                        pri,
                        ts,
                        i2o,
                        o2i,
                        nleft - 1,
                        eleft - 1,
                        prefix + mov + "v",
                        depth + 1,
                    )
                    if best is None or (len(r), r) < (len(best), best):
                        best = r

                    cdll.remove(new_cdll)
                    _undo_edge(og, ts_out, new_out)
                    _undo_node(og)
                    del i2o[c]
                    del o2i[new_out]

                assert best is not None
                return best

        # -- C: edge primary -> secondary --
        rec.neighbour_checks += 1
        if ts_in in ig.neighbors(tp_in):
            rec.neighbour_checks += 1
            if ts_out not in og.neighbors(tp_out):
                og.add_edge(tp_out, ts_out)
                _push_frame(
                    rec,
                    scope,
                    p0,
                    n0c,
                    t0,
                    "C",
                    a,
                    b,
                    abs(a) + abs(b),
                    0,
                    1,
                    depth,
                    ctx.start_node,
                )
                r = _step_counted(
                    ctx,
                    og,
                    cdll,
                    tp,
                    ts,
                    i2o,
                    o2i,
                    nleft,
                    eleft - 1,
                    prefix + _primary_moves(a) + _secondary_moves(b) + "C",
                    depth + 1,
                )
                _undo_edge(og, tp_out, ts_out)
                return r

        # -- c: edge secondary -> primary (directed only) --
        if ig.directed():
            rec.neighbour_checks += 1
            if tp_in in ig.neighbors(ts_in):
                rec.neighbour_checks += 1
                if tp_out not in og.neighbors(ts_out):
                    og.add_edge(ts_out, tp_out)
                    _push_frame(
                        rec,
                        scope,
                        p0,
                        n0c,
                        t0,
                        "c",
                        a,
                        b,
                        abs(a) + abs(b),
                        0,
                        1,
                        depth,
                        ctx.start_node,
                    )
                    r = _step_counted(
                        ctx,
                        og,
                        cdll,
                        tp,
                        ts,
                        i2o,
                        o2i,
                        nleft,
                        eleft - 1,
                        prefix + _primary_moves(a) + _secondary_moves(b) + "c",
                        depth + 1,
                    )
                    _undo_edge(og, ts_out, tp_out)
                    return r

    raise InstrumentationError(
        f"canonical mirror: no valid operation found. Remaining: {nleft} nodes, {eleft} edges."
    )


def _materialise_candidates(
    ig: SparseGraph,
    node: int,
    i2o: dict[int, int],
    rec: _Recorder,
) -> list[int]:
    """Mirror of ``[n for n in ig.neighbors(node) if n not in i2o]``.

    The comprehension always scans the whole adjacency set, so one neighbour
    check is charged per neighbour -- unlike the greedy encoder, which
    short-circuits at the first hit. The difference is the point of the
    ``O(deg)`` versus ``O(Delta * D_f)`` split in T-13 §2.1.
    """
    out: list[int] = []
    for neighbor in ig.neighbors(node):
        rec.neighbour_checks += 1
        if neighbor not in i2o:
            out.append(neighbor)
    return out


def _prune(
    cands: list[int],
    triplets: list[tuple[int, int, int]] | None,
) -> list[int]:
    """Apply the structural-triplet filter, or pass the list through."""
    if triplets is None:
        return cands
    max_trip = max(triplets[c] for c in cands)
    return [c for c in cands if triplets[c] == max_trip]


def canonical_detail(
    graph: SparseGraph,
) -> tuple[str, OperationCounts, list[FrameRecord]]:
    """Exhaustive canonical encode with counts and per-frame detail."""
    return _canonical_search(graph, pruned=False)


def canonical_counts(graph: SparseGraph) -> tuple[str, OperationCounts]:
    """Exhaustive canonical encode with realised operation counts.

    Args:
        graph: The graph to encode.

    Returns:
        Tuple ``(string, counts)``. The string is byte-identical to
        ``isalgraph.core.canonical.canonical_string(graph)``.
    """
    string, counts, _ = _canonical_search(graph, pruned=False)
    return string, counts


def pruned_detail(
    graph: SparseGraph,
) -> tuple[str, OperationCounts, list[FrameRecord]]:
    """Triplet-pruned canonical encode with counts and per-frame detail."""
    return _canonical_search(graph, pruned=True)


def pruned_counts(graph: SparseGraph) -> tuple[str, OperationCounts]:
    """Triplet-pruned canonical encode with realised operation counts.

    Args:
        graph: The graph to encode.

    Returns:
        Tuple ``(string, counts)``. The string is byte-identical to
        ``isalgraph.core.canonical_pruned.pruned_canonical_string(graph)``.
    """
    string, counts, _ = _canonical_search(graph, pruned=True)
    return string, counts
