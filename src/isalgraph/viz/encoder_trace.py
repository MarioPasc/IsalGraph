"""Instrumented mirror of the greedy graph-to-string encoder.

Why this module exists
----------------------
``GraphToString.run_with_trace`` does not trace the encoder. It takes the
finished string and **replays** it through
:class:`~isalgraph.core.string_to_graph.StringToGraph`, so its snapshots
are interpreter states, one per emitted symbol. A figure built from them
shows a decoder running, which is what the S2G figure already shows; the
only difference is which side of the graph is drawn solid. The encoder's
actual work -- walking the displacement pairs of
:math:`\\mathcal{P}(M)` in increasing cost, rejecting the ones that admit
no operation, and running the ``V`` :math:`\\succ` ``v`` :math:`\\succ`
``C`` :math:`\\succ` ``c`` cascade at the first pair that does -- appears
nowhere in it.

``GraphToString.run(v0, trace=True)`` does record real encoder states,
one per outer-loop iteration, but not the rejected pairs inside an
iteration.

So this module re-runs the encoder loop with the rejections recorded.

Why a mirror rather than an instrumented encoder
------------------------------------------------
``core/graph_to_string.py`` is frozen: it is the reference the C++
differential suite compares against, and editing it means re-proving
parity over 3,079 graphs. A mirror costs nothing there.

The obvious hazard of a mirror is that it drifts from the algorithm it
claims to depict, and a drifted schematic is worse than no schematic.
The mitigation is the one ``search_tree`` already uses: the mirror emits
an instruction string, and a test asserts that string is byte-identical
to the frozen encoder's on a family of graphs. A mirror that has drifted
fails the suite; it cannot quietly produce a wrong figure.

Node identity
-------------
The encoder maintains two node spaces -- the input graph's and the
output graph's -- linked by ``_i2o`` / ``_o2i``. Everything this module
reports is in **input-graph ids**, because those are what the figure
draws. The output ids exist only inside the loop.

Restriction: standard library only, like the rest of ``core``-facing
code. No matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass

from isalgraph.core.cdll import CircularDoublyLinkedList
from isalgraph.core.graph_to_string import GraphToString, generate_pairs_sorted_by_sum
from isalgraph.core.sparse_graph import SparseGraph
from isalgraph.core.trace import Edge, normalise_edge
from isalgraph.types import NodeId

#: Verdict recorded for a displacement pair that admitted no operation.
REJECTED: str = "-"


class EncoderMirrorError(RuntimeError):
    """Raised when the mirror cannot reproduce the frozen encoder."""


@dataclass(frozen=True)
class PairProbe:
    """One displacement pair tested inside a single encoder iteration.

    Args:
        displacement: The ``(a, b)`` pair from
            :func:`~isalgraph.core.graph_to_string.generate_pairs_sorted_by_sum`;
            ``a`` primary steps, ``b`` secondary steps, negative meaning
            backwards.
        cost: ``|a| + |b|``, the number of movement instructions the pair
            would cost. This is the key the pair order sorts on.
        primary_node: Input-graph node under the tentative primary pointer.
        secondary_node: Input-graph node under the tentative secondary pointer.
        verdict: The instruction this pair admitted, or :data:`REJECTED`.
        reasons: One line per rejected option, in priority order, saying
            why it did not apply. Empty for the options never reached
            because a higher-priority one fired.
    """

    displacement: tuple[int, int]
    cost: int
    primary_node: NodeId
    secondary_node: NodeId
    verdict: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class EncoderIteration:
    """One pass of the encoder's outer loop.

    An iteration emits a whole *group* of symbols -- the movement
    instructions for the selected displacement, then the operation --
    which is why the encoder has fewer iterations than the string has
    symbols.

    Args:
        index: Zero-based iteration number.
        ring_before: CDLL contents in forward circular order, as
            input-graph node ids, at the top of the iteration.
        primary_before: Input-graph node under the primary pointer before.
        secondary_before: Input-graph node under the secondary pointer before.
        probes: Every pair tested, in the order tested. The last entry is
            the one that fired.
        emitted: The symbol group this iteration appended.
        string_before: The instruction string before this iteration.
        captured_nodes_before: Input-graph nodes already encoded, before.
        captured_edges_before: Input-graph edges already encoded, before.
        created_node: The input-graph node this iteration inserted, if any.
        created_edge: The input-graph edge this iteration created, if any.
        ring_after: Ring contents after, as input-graph node ids.
        primary_after: Input-graph node under the primary pointer after.
        secondary_after: Input-graph node under the secondary pointer after.
    """

    index: int
    ring_before: tuple[NodeId, ...]
    primary_before: NodeId
    secondary_before: NodeId
    probes: tuple[PairProbe, ...]
    emitted: str
    string_before: str
    captured_nodes_before: tuple[NodeId, ...]
    captured_edges_before: tuple[Edge, ...]
    created_node: NodeId | None
    created_edge: Edge | None
    ring_after: tuple[NodeId, ...]
    primary_after: NodeId
    secondary_after: NodeId

    @property
    def selected(self) -> PairProbe:
        """The probe that fired.

        Returns:
            The last recorded probe.

        Raises:
            EncoderMirrorError: If the iteration recorded no probe, which
                the encoder's own ``found`` guard makes impossible.
        """
        if not self.probes:
            raise EncoderMirrorError(f"iteration {self.index} recorded no probe")
        return self.probes[-1]

    @property
    def captured_nodes_after(self) -> tuple[NodeId, ...]:
        """Input-graph nodes encoded once this iteration has run."""
        if self.created_node is None:
            return self.captured_nodes_before
        return tuple(sorted((*self.captured_nodes_before, self.created_node)))

    @property
    def captured_edges_after(self) -> tuple[Edge, ...]:
        """Input-graph edges encoded once this iteration has run."""
        if self.created_edge is None:
            return self.captured_edges_before
        return tuple(sorted((*self.captured_edges_before, self.created_edge)))


@dataclass(frozen=True)
class EncoderTrace:
    """The full instrumented encode of one graph from one starting node.

    Args:
        graph: The input graph.
        start_node: The input-graph node the encode started from.
        instruction_string: The emitted string. Byte-identical to
            ``GraphToString(graph).run(start_node)[0]``; see
            :func:`trace_encoder`.
        iterations: One entry per outer-loop pass.
    """

    graph: SparseGraph
    start_node: NodeId
    instruction_string: str
    iterations: tuple[EncoderIteration, ...]

    def __len__(self) -> int:
        """Return the number of encoder iterations."""
        return len(self.iterations)

    @property
    def groups(self) -> tuple[str, ...]:
        """The symbol group each iteration emitted."""
        return tuple(it.emitted for it in self.iterations)


def _emit_moves(steps: int, *, primary: bool) -> str:
    """Return the movement instructions for *steps* displacements.

    Args:
        steps: Signed displacement; positive is forward.
        primary: Whether the primary pointer moves.

    Returns:
        The movement symbols, possibly empty.
    """
    if steps >= 0:
        return ("N" if primary else "n") * steps
    return ("P" if primary else "p") * (-steps)


class _MirrorState:
    """Mutable encoder state, mirroring ``GraphToString``'s private fields."""

    __slots__ = ("cdll", "i2o", "o2i", "out", "primary", "secondary")

    def __init__(self, graph: SparseGraph, start_node: NodeId) -> None:
        self.out = SparseGraph(graph.max_nodes(), graph.directed())
        self.cdll = CircularDoublyLinkedList(graph.max_nodes())
        first = self.out.add_node()
        slot = self.cdll.insert_after(-1, first)
        self.primary: int = slot
        self.secondary: int = slot
        self.i2o: dict[int, int] = {start_node: first}
        self.o2i: dict[int, int] = {first: start_node}

    def move(self, ptr: int, steps: int) -> int:
        """Walk *ptr* by *steps* CDLL positions, signed."""
        step_fn = self.cdll.next_node if steps >= 0 else self.cdll.prev_node
        for _ in range(abs(steps)):
            ptr = step_fn(ptr)
        return ptr

    def ring(self, anchor: int = 0) -> tuple[NodeId, ...]:
        """Return the ring in forward circular order, as input-graph ids."""
        size = self.cdll.size()
        if size == 0:
            return ()
        out: list[NodeId] = []
        ptr = anchor
        for _ in range(size):
            out.append(self.o2i[int(self.cdll.get_value(ptr))])
            ptr = self.cdll.next_node(ptr)
        return tuple(out)

    def input_node_at(self, ptr: int) -> NodeId:
        """Return the input-graph node the CDLL slot *ptr* carries."""
        return self.o2i[int(self.cdll.get_value(ptr))]


def _first_uninserted_neighbour(
    graph: SparseGraph,
    node: NodeId,
    i2o: dict[int, int],
) -> NodeId | None:
    """Return the first neighbour of *node* not yet in the output graph.

    Mirrors ``GraphToString._find_new_neighbor``, including its reliance
    on ``SparseGraph.neighbors`` iteration order. Invariant 7 of
    ``CLAUDE.md`` makes that order load-bearing: it is what the greedy
    encoder's output depends on, and reproducing it is the whole reason
    this helper does not sort.

    Args:
        graph: The input graph.
        node: Input-graph node whose neighbours are scanned.
        i2o: Input-to-output node map; membership means "already inserted".

    Returns:
        An input-graph node id, or ``None`` when every neighbour is in.
    """
    for neighbour in graph.neighbors(node):
        if neighbour not in i2o:
            return neighbour
    return None


def _probe_pair(  # noqa: PLR0911, PLR0913  -- one return per cascade level is the point
    graph: SparseGraph,
    state: _MirrorState,
    displacement: tuple[int, int],
    *,
    nodes_left: int,
    tentative_primary: int,
    tentative_secondary: int,
) -> PairProbe:
    """Run the ``V`` / ``v`` / ``C`` / ``c`` cascade on one displacement pair.

    The cascade is read directly off ``GraphToString.run``: each level is
    tried in priority order, and the first that applies wins the pair.
    This function decides nothing; it only records which level applied
    and why the earlier ones did not.

    Args:
        graph: The input graph.
        state: Current mirror state.
        displacement: The ``(a, b)`` pair being tested.
        nodes_left: Input-graph nodes still to be inserted.
        tentative_primary: CDLL slot the primary would move to.
        tentative_secondary: CDLL slot the secondary would move to.

    Returns:
        The probe record, with ``verdict`` naming the instruction or
        :data:`REJECTED`.
    """
    a, b = displacement
    pri_in = state.input_node_at(tentative_primary)
    sec_in = state.input_node_at(tentative_secondary)
    pri_out = int(state.cdll.get_value(tentative_primary))
    sec_out = int(state.cdll.get_value(tentative_secondary))
    reasons: list[str] = []

    def probe(verdict: str) -> PairProbe:
        return PairProbe(
            displacement=(a, b),
            cost=abs(a) + abs(b),
            primary_node=pri_in,
            secondary_node=sec_in,
            verdict=verdict,
            reasons=tuple(reasons),
        )

    if nodes_left > 0 and _first_uninserted_neighbour(graph, pri_in, state.i2o) is not None:
        return probe("V")
    reasons.append(
        "V: no nodes left" if nodes_left <= 0 else f"V: every neighbour of {pri_in} is inserted"
    )

    if nodes_left > 0 and _first_uninserted_neighbour(graph, sec_in, state.i2o) is not None:
        return probe("v")
    reasons.append(
        "v: no nodes left" if nodes_left <= 0 else f"v: every neighbour of {sec_in} is inserted"
    )

    if sec_in not in graph.neighbors(pri_in):
        reasons.append(f"C: {pri_in}-{sec_in} is not an edge of G")
    elif sec_out in state.out.neighbors(pri_out):
        reasons.append(f"C: {pri_in}-{sec_in} is already encoded")
    else:
        return probe("C")

    if graph.directed():
        if pri_in not in graph.neighbors(sec_in):
            reasons.append(f"c: {sec_in}-{pri_in} is not an arc of G")
        elif pri_out in state.out.neighbors(sec_out):
            reasons.append(f"c: {sec_in}-{pri_in} is already encoded")
        else:
            return probe("c")

    return probe(REJECTED)


def uninserted_neighbours(
    graph: SparseGraph,
    node: NodeId,
    i2o: dict[int, int],
) -> tuple[NodeId, ...]:
    """Return every neighbour of *node* not yet in the output graph.

    The greedy encoder commits to the first of these; the exhaustive and
    pruned canonicalisations branch over all of them. Per Remark 2.7 this
    is the *only* choice inside one execution -- the displacement order
    and the ``V`` / ``v`` / ``C`` / ``c`` priority are fixed -- so this
    function delimits the entire branching factor.

    Args:
        graph: The input graph.
        node: Input-graph node whose neighbours are scanned.
        i2o: Input-to-output node map.

    Returns:
        Candidates, in ``SparseGraph.neighbors`` iteration order.
    """
    return tuple(n for n in graph.neighbors(node) if n not in i2o)


def _apply(  # noqa: PLR0913  -- the commit needs the whole tentative state
    graph: SparseGraph,
    state: _MirrorState,
    probe: PairProbe,
    *,
    tentative_primary: int,
    tentative_secondary: int,
    candidate: NodeId | None = None,
) -> tuple[str, NodeId | None, Edge | None]:
    """Commit the operation *probe* selected and return what it produced.

    Args:
        graph: The input graph.
        state: Mirror state, mutated in place.
        probe: The winning probe.
        tentative_primary: CDLL slot the primary moves to.
        tentative_secondary: CDLL slot the secondary moves to.
        candidate: Which uninserted neighbour a ``V``/``v`` inserts.
            ``None`` takes the first, which is what the greedy encoder
            does; a caller reconstructing a specific execution passes the
            neighbour that execution chose.

    Returns:
        A ``(emitted_group, created_node, created_edge)`` triple, with the
        node and edge in input-graph ids.

    Raises:
        EncoderMirrorError: If *probe* carries no operation.
    """
    a, b = probe.displacement
    directed = graph.directed()
    pri_out = int(state.cdll.get_value(tentative_primary))
    sec_out = int(state.cdll.get_value(tentative_secondary))

    if probe.verdict in {"V", "v"}:
        via_primary = probe.verdict == "V"
        anchor_slot = tentative_primary if via_primary else tentative_secondary
        anchor_in = probe.primary_node if via_primary else probe.secondary_node
        anchor_out = pri_out if via_primary else sec_out

        if candidate is None:
            candidate = _first_uninserted_neighbour(graph, anchor_in, state.i2o)
        if candidate is None:  # pragma: no cover -- probe already established it exists
            raise EncoderMirrorError(f"{probe.verdict} selected but no candidate at {anchor_in}")
        new_out = state.out.add_node()
        state.i2o[candidate] = new_out
        state.o2i[new_out] = candidate
        state.out.add_edge(anchor_out, new_out)
        state.cdll.insert_after(anchor_slot, new_out)
        if via_primary:
            state.primary = tentative_primary
            emitted = _emit_moves(a, primary=True) + "V"
        else:
            state.secondary = tentative_secondary
            emitted = _emit_moves(b, primary=False) + "v"
        return emitted, candidate, normalise_edge(anchor_in, candidate, directed=directed)

    if probe.verdict in {"C", "c"}:
        if probe.verdict == "C":
            state.out.add_edge(pri_out, sec_out)
            edge = normalise_edge(probe.primary_node, probe.secondary_node, directed=directed)
        else:
            state.out.add_edge(sec_out, pri_out)
            edge = normalise_edge(probe.secondary_node, probe.primary_node, directed=directed)
        state.primary = tentative_primary
        state.secondary = tentative_secondary
        emitted = _emit_moves(a, primary=True) + _emit_moves(b, primary=False) + probe.verdict
        return emitted, None, edge

    raise EncoderMirrorError(f"cannot apply verdict {probe.verdict!r}")


def trace_encoder(
    graph: SparseGraph,
    start_node: NodeId = 0,
    *,
    max_probes_per_iteration: int | None = None,
    verify: bool = True,
) -> EncoderTrace:
    """Encode *graph* from *start_node*, recording the pair search.

    Args:
        graph: A connected graph (reachable from *start_node* when directed).
        start_node: Input-graph node to start from.
        max_probes_per_iteration: Keep at most this many probe records per
            iteration, the winning one always included. ``None`` keeps
            every probe. Only the record is trimmed; the search is not.
        verify: Compare the emitted string against the frozen encoder and
            raise on any difference. Leave this on.

    Returns:
        The instrumented trace.

    Raises:
        EncoderMirrorError: If *verify* is set and the mirror's string
            differs from ``GraphToString(graph).run(start_node)[0]``, or
            if an iteration finds no applicable operation.
        ValueError: Propagated from the frozen encoder's reachability
            check when *start_node* cannot reach every node.
    """
    # Delegate the precondition to the frozen encoder so the mirror cannot
    # accept an input the real algorithm rejects.
    reference, _ = GraphToString(graph).run(start_node)

    state = _MirrorState(graph, start_node)
    nodes_left = graph.node_count() - 1
    edges_left = graph.logical_edge_count()
    emitted_total = ""
    captured_nodes: tuple[NodeId, ...] = (start_node,)
    captured_edges: tuple[Edge, ...] = ()
    iterations: list[EncoderIteration] = []

    while nodes_left > 0 or edges_left > 0:
        ring_before = state.ring()
        primary_before = state.input_node_at(state.primary)
        secondary_before = state.input_node_at(state.secondary)
        probes: list[PairProbe] = []
        winner: PairProbe | None = None
        tent_pri = tent_sec = -1

        for displacement in generate_pairs_sorted_by_sum(state.out.node_count()):
            a, b = displacement
            tent_pri = state.move(state.primary, a)
            tent_sec = state.move(state.secondary, b)
            probe = _probe_pair(
                graph,
                state,
                displacement,
                nodes_left=nodes_left,
                tentative_primary=tent_pri,
                tentative_secondary=tent_sec,
            )
            probes.append(probe)
            if probe.verdict != REJECTED:
                winner = probe
                break

        if winner is None:
            raise EncoderMirrorError(
                f"no operation applies at iteration {len(iterations)}: "
                f"{nodes_left} nodes and {edges_left} edges remain"
            )

        group, created_node, created_edge = _apply(
            graph,
            state,
            winner,
            tentative_primary=tent_pri,
            tentative_secondary=tent_sec,
        )
        if created_node is not None:
            nodes_left -= 1
        if created_edge is not None:
            edges_left -= 1

        iterations.append(
            EncoderIteration(
                index=len(iterations),
                ring_before=ring_before,
                primary_before=primary_before,
                secondary_before=secondary_before,
                probes=_trim_probes(probes, max_probes_per_iteration),
                emitted=group,
                string_before=emitted_total,
                captured_nodes_before=captured_nodes,
                captured_edges_before=captured_edges,
                created_node=created_node,
                created_edge=created_edge,
                ring_after=state.ring(),
                primary_after=state.input_node_at(state.primary),
                secondary_after=state.input_node_at(state.secondary),
            )
        )
        emitted_total += group
        captured_nodes = iterations[-1].captured_nodes_after
        captured_edges = iterations[-1].captured_edges_after

    if verify and emitted_total != reference:
        raise EncoderMirrorError(
            "mirror drifted from the frozen encoder: "
            f"mirror emitted {emitted_total!r}, GraphToString emitted {reference!r}"
        )

    return EncoderTrace(
        graph=graph,
        start_node=start_node,
        instruction_string=emitted_total,
        iterations=tuple(iterations),
    )


def _trim_probes(probes: list[PairProbe], limit: int | None) -> tuple[PairProbe, ...]:
    """Keep the first *limit* probes plus the winner, preserving order."""
    if limit is None or len(probes) <= limit:
        return tuple(probes)
    if limit <= 1:
        return (probes[-1],)
    return (*probes[: limit - 1], probes[-1])


def _one_iteration(
    graph: SparseGraph,
    state: _MirrorState,
    *,
    nodes_left: int,
) -> tuple[list[PairProbe], int, int]:
    """Run the pair search for one iteration without committing anything.

    Args:
        graph: The input graph.
        state: Current mirror state; not modified.
        nodes_left: Input-graph nodes still to be inserted.

    Returns:
        The probes tested and the two tentative CDLL slots of the winner.

    Raises:
        EncoderMirrorError: If no pair admits an operation.
    """
    probes: list[PairProbe] = []
    for displacement in generate_pairs_sorted_by_sum(state.out.node_count()):
        a, b = displacement
        tent_pri = state.move(state.primary, a)
        tent_sec = state.move(state.secondary, b)
        probe = _probe_pair(
            graph,
            state,
            displacement,
            nodes_left=nodes_left,
            tentative_primary=tent_pri,
            tentative_secondary=tent_sec,
        )
        probes.append(probe)
        if probe.verdict != REJECTED:
            return probes, tent_pri, tent_sec
    raise EncoderMirrorError(f"no operation applies with {nodes_left} nodes left")


def _record(  # noqa: PLR0913  -- assembling a frozen record needs its fields
    *,
    index: int,
    before: _MirrorState,
    ring_before: tuple[NodeId, ...],
    primary_before: NodeId,
    secondary_before: NodeId,
    probes: tuple[PairProbe, ...],
    emitted: str,
    string_before: str,
    captured_nodes: tuple[NodeId, ...],
    captured_edges: tuple[Edge, ...],
    created_node: NodeId | None,
    created_edge: Edge | None,
) -> EncoderIteration:
    """Freeze one iteration's record after the state has been committed."""
    return EncoderIteration(
        index=index,
        ring_before=ring_before,
        primary_before=primary_before,
        secondary_before=secondary_before,
        probes=probes,
        emitted=emitted,
        string_before=string_before,
        captured_nodes_before=captured_nodes,
        captured_edges_before=captured_edges,
        created_node=created_node,
        created_edge=created_edge,
        ring_after=before.ring(),
        primary_after=before.input_node_at(before.primary),
        secondary_after=before.input_node_at(before.secondary),
    )


def _search_execution(  # noqa: PLR0913  -- a DFS frame carries the whole state
    graph: SparseGraph,
    state: _MirrorState,
    *,
    target: str,
    nodes_left: int,
    edges_left: int,
    emitted: str,
    captured_nodes: tuple[NodeId, ...],
    captured_edges: tuple[Edge, ...],
    iterations: list[EncoderIteration],
    max_probes: int | None,
) -> list[EncoderIteration] | None:
    """Depth-first search for the execution that emits *target*.

    Only the uninserted-neighbour identity at a ``V``/``v`` step is
    branched over, because that is the only thing an execution is free to
    choose (Remark 2.7). The emitted symbol group does **not** depend on
    the choice -- the displacement and the cascade level are already
    fixed by the state -- so the choice is invisible in this iteration's
    output and only steers later ones. That is exactly why the search
    needs backtracking rather than a per-step decision rule.

    Args:
        graph: The input graph.
        state: Mirror state at the top of the next iteration.
        target: The instruction string to reproduce.
        nodes_left: Nodes still to insert.
        edges_left: Edges still to encode.
        emitted: What has been emitted so far.
        captured_nodes: Input-graph nodes encoded so far.
        captured_edges: Input-graph edges encoded so far.
        iterations: Records accumulated on this branch.
        max_probes: Probe-record cap per iteration.

    Returns:
        The completed iteration list, or ``None`` if this branch cannot
        reach *target*.
    """
    from copy import deepcopy

    if nodes_left <= 0 and edges_left <= 0:
        return list(iterations) if emitted == target else None

    ring_before = state.ring()
    primary_before = state.input_node_at(state.primary)
    secondary_before = state.input_node_at(state.secondary)
    probes, tent_pri, tent_sec = _one_iteration(graph, state, nodes_left=nodes_left)
    winner = probes[-1]

    if winner.verdict in {"V", "v"}:
        anchor = winner.primary_node if winner.verdict == "V" else winner.secondary_node
        candidates = uninserted_neighbours(graph, anchor, state.i2o)
    else:
        candidates = (None,)  # type: ignore[assignment]

    for candidate in candidates:
        branch = deepcopy(state)
        group, created_node, created_edge = _apply(
            graph,
            branch,
            winner,
            tentative_primary=tent_pri,
            tentative_secondary=tent_sec,
            candidate=candidate,
        )
        if not target.startswith(emitted + group):
            continue
        record = _record(
            index=len(iterations),
            before=branch,
            ring_before=ring_before,
            primary_before=primary_before,
            secondary_before=secondary_before,
            probes=_trim_probes(list(probes), max_probes),
            emitted=group,
            string_before=emitted,
            captured_nodes=captured_nodes,
            captured_edges=captured_edges,
            created_node=created_node,
            created_edge=created_edge,
        )
        found = _search_execution(
            graph,
            branch,
            target=target,
            nodes_left=nodes_left - (1 if created_node is not None else 0),
            edges_left=edges_left - (1 if created_edge is not None else 0),
            emitted=emitted + group,
            captured_nodes=record.captured_nodes_after,
            captured_edges=record.captured_edges_after,
            iterations=[*iterations, record],
            max_probes=max_probes,
        )
        if found is not None:
            return found
    return None


def trace_execution(
    graph: SparseGraph,
    target: str,
    *,
    start_node: NodeId | None = None,
    max_probes_per_iteration: int | None = None,
) -> EncoderTrace:
    """Reconstruct the execution of ``GraphToString`` that emits *target*.

    The greedy encoder commits to the first uninserted neighbour at every
    ``V``/``v``; the canonical and pruned canonicalisations minimise over
    all of them. A string produced by either canonicalisation is
    therefore a valid ``GraphToString`` execution that greedy may never
    reach, and this function recovers it, so the pruned form can be drawn
    with the same encoder-side detail as the greedy one rather than as a
    replay.

    Args:
        graph: The input graph.
        target: The instruction string to reproduce.
        start_node: Starting node. ``None`` searches every node and takes
            the first that admits an execution emitting *target*.
        max_probes_per_iteration: Probe-record cap; see
            :func:`trace_encoder`.

    Returns:
        The instrumented trace, whose ``instruction_string`` is *target*.

    Raises:
        EncoderMirrorError: If no execution from any admissible start node
            emits *target*.
    """
    starts = range(graph.node_count()) if start_node is None else (start_node,)
    for start in starts:
        state = _MirrorState(graph, start)
        found = _search_execution(
            graph,
            state,
            target=target,
            nodes_left=graph.node_count() - 1,
            edges_left=graph.logical_edge_count(),
            emitted="",
            captured_nodes=(start,),
            captured_edges=(),
            iterations=[],
            max_probes=max_probes_per_iteration,
        )
        if found is not None:
            return EncoderTrace(
                graph=graph,
                start_node=start,
                instruction_string=target,
                iterations=tuple(found),
            )
    raise EncoderMirrorError(
        f"no execution of GraphToString emits {target!r} from "
        + ("any start node" if start_node is None else f"node {start_node}")
    )


__all__ = [
    "REJECTED",
    "EncoderIteration",
    "EncoderMirrorError",
    "EncoderTrace",
    "PairProbe",
    "trace_encoder",
    "trace_execution",
    "uninserted_neighbours",
]
