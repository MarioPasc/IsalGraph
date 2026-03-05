"""Canonical IsalGraph string computation.

The canonical string for a graph G with N nodes is defined as the
lexicographically smallest among all shortest strings producible by
the G2S algorithm from any starting node, across all possible valid
neighbor choices at V/v branch points:

    w*_G = lexmin over v in V, over all execution paths P of G2S(G, v),
           of (|P|, P)

This is a **complete graph invariant**: w*_G = w*_H if and only if G ~ H
(graph isomorphism).

The greedy GraphToString algorithm is not isomorphism-equivariant because
its neighbor iteration order (over Python sets) depends on node IDs.
To compute a true invariant, this module implements exhaustive backtracking
search over all valid neighbor choices at each V/v step.

The function also exposes an approximate graph distance metric via
Levenshtein edit distance between canonical strings.
"""

from __future__ import annotations

from isalgraph.core.cdll import CircularDoublyLinkedList
from isalgraph.core.graph_to_string import generate_pairs_sorted_by_sum
from isalgraph.core.sparse_graph import SparseGraph


def canonical_string(graph: SparseGraph) -> str:
    """Compute the canonical IsalGraph string for a graph.

    Uses exhaustive backtracking search over all valid neighbor choices
    at V/v branch points to produce a labeling-independent result.

    Args:
        graph: The graph to compute the canonical string for.
            Must be connected (undirected) or have all nodes reachable
            from at least one starting node (directed).

    Returns:
        The canonical string w*_G.

    Raises:
        ValueError: If no starting node can reach all other nodes.
    """
    n = graph.node_count()
    if n <= 1:
        return ""

    best: str | None = None
    for v in range(n):
        if not _is_reachable(graph, v):
            continue
        w = _canonical_g2s(graph, v)
        if best is None or (len(w), w) < (len(best), best):
            best = w

    if best is None:
        raise ValueError(
            "No starting node can reach all other nodes. "
            "For undirected graphs, the graph must be connected. "
            "For directed graphs, at least one node must reach all others."
        )

    return best


def graph_distance(g1: SparseGraph, g2: SparseGraph) -> int:
    """Approximate graph edit distance via Levenshtein distance between canonical strings.

    Args:
        g1: First graph.
        g2: Second graph.

    Returns:
        Levenshtein edit distance between canonical_string(g1) and canonical_string(g2).
    """
    return levenshtein(canonical_string(g1), canonical_string(g2))


def levenshtein(s: str, t: str) -> int:
    """Compute the Levenshtein edit distance between two strings.

    Uses the standard O(n*m) dynamic programming approach with O(min(n,m))
    space via a single-row optimization.

    Args:
        s: First string.
        t: Second string.

    Returns:
        The minimum number of single-character edits (insertions, deletions,
        substitutions) needed to transform s into t.
    """
    if len(s) < len(t):
        return levenshtein(t, s)

    if len(t) == 0:
        return len(s)

    prev_row = list(range(len(t) + 1))
    for i, sc in enumerate(s):
        curr_row = [i + 1]
        for j, tc in enumerate(t):
            insert = prev_row[j + 1] + 1
            delete = curr_row[j] + 1
            replace = prev_row[j] + (0 if sc == tc else 1)
            curr_row.append(min(insert, delete, replace))
        prev_row = curr_row

    return prev_row[-1]


# ------------------------------------------------------------------
# Internal: exhaustive G2S search with backtracking
# ------------------------------------------------------------------


def _is_reachable(graph: SparseGraph, start: int) -> bool:
    """Check if all nodes are reachable from *start* via outgoing edges."""
    n = graph.node_count()
    if n <= 1:
        return True
    visited: set[int] = set()
    stack: list[int] = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                stack.append(neighbor)
    return len(visited) == n


def _walk(cdll: CircularDoublyLinkedList, ptr: int, steps: int) -> int:
    """Move *ptr* through the CDLL by *steps* (positive=next, negative=prev)."""
    for _ in range(abs(steps)):
        ptr = cdll.next_node(ptr) if steps > 0 else cdll.prev_node(ptr)
    return ptr


def _primary_moves(a: int) -> str:
    """Emit N or P instructions for primary pointer displacement *a*."""
    return "N" * a if a >= 0 else "P" * (-a)


def _secondary_moves(b: int) -> str:
    """Emit n or p instructions for secondary pointer displacement *b*."""
    return "n" * b if b >= 0 else "p" * (-b)


def _undo_edge(og: SparseGraph, source: int, target: int) -> None:
    """Remove an edge that was just added (reverse of add_edge)."""
    og._adjacency[source].discard(target)  # noqa: SLF001
    og._edge_count -= 1  # noqa: SLF001
    if not og._directed_graph:  # noqa: SLF001
        og._adjacency[target].discard(source)  # noqa: SLF001
        og._edge_count -= 1  # noqa: SLF001


def _undo_node(og: SparseGraph) -> None:
    """Remove the last added node (reverse of add_node)."""
    og._node_count -= 1  # noqa: SLF001


def _canonical_g2s(input_graph: SparseGraph, start_node: int) -> str:
    """Find the shortest, then lex-smallest G2S string from *start_node*.

    Explores all possible neighbor choices at V/v branch points via
    backtracking. The greedy V>v>C>c priority and minimum-displacement
    pair ordering are preserved (they are labeling-independent).
    """
    max_n = input_graph.max_nodes()
    og = SparseGraph(max_n, input_graph.directed())
    cdll = CircularDoublyLinkedList(max_n)

    n0 = og.add_node()
    c0 = cdll.insert_after(-1, n0)
    i2o: dict[int, int] = {start_node: n0}
    o2i: dict[int, int] = {n0: start_node}

    return _step(
        input_graph,
        og,
        cdll,
        c0,
        c0,
        i2o,
        o2i,
        input_graph.node_count() - 1,
        input_graph.logical_edge_count(),
        "",
    )


def _step(  # noqa: ANN001
    ig: SparseGraph,
    og: SparseGraph,
    cdll: CircularDoublyLinkedList,
    pri: int,
    sec: int,
    i2o: dict[int, int],
    o2i: dict[int, int],
    nleft: int,
    eleft: int,
    prefix: str,
) -> str:
    """One step of the exhaustive G2S search.

    Mirrors the greedy GraphToString algorithm but branches over all
    valid neighbor choices at V/v steps. Uses in-place mutation with
    undo (backtracking) instead of deep copies for performance.
    """
    if nleft <= 0 and eleft <= 0:
        return prefix

    pairs = generate_pairs_sorted_by_sum(og.node_count())

    for a, b in pairs:
        # ---- tentative primary position ----
        tp = _walk(cdll, pri, a)
        tp_out = cdll.get_value(tp)
        tp_in = o2i[tp_out]

        # -- V: primary has uninserted neighbor --
        if nleft > 0:
            cands = [n for n in ig.neighbors(tp_in) if n not in i2o]
            if cands:
                mov = _primary_moves(a)
                best: str | None = None
                for c in cands:
                    # Forward
                    new_out = og.add_node()
                    i2o[c] = new_out
                    o2i[new_out] = c
                    og.add_edge(tp_out, new_out)
                    new_cdll = cdll.insert_after(tp, new_out)

                    r = _step(
                        ig,
                        og,
                        cdll,
                        tp,
                        sec,
                        i2o,
                        o2i,
                        nleft - 1,
                        eleft - 1,
                        prefix + mov + "V",
                    )
                    if best is None or (len(r), r) < (len(best), best):
                        best = r

                    # Backward
                    cdll.remove(new_cdll)
                    _undo_edge(og, tp_out, new_out)
                    _undo_node(og)
                    del i2o[c]
                    del o2i[new_out]

                return best  # type: ignore[return-value]

        # ---- tentative secondary position ----
        ts = _walk(cdll, sec, b)
        ts_out = cdll.get_value(ts)
        ts_in = o2i[ts_out]

        # -- v: secondary has uninserted neighbor --
        if nleft > 0:
            cands = [n for n in ig.neighbors(ts_in) if n not in i2o]
            if cands:
                mov = _secondary_moves(b)
                best = None
                for c in cands:
                    # Forward
                    new_out = og.add_node()
                    i2o[c] = new_out
                    o2i[new_out] = c
                    og.add_edge(ts_out, new_out)
                    new_cdll = cdll.insert_after(ts, new_out)

                    r = _step(
                        ig,
                        og,
                        cdll,
                        pri,
                        ts,
                        i2o,
                        o2i,
                        nleft - 1,
                        eleft - 1,
                        prefix + mov + "v",
                    )
                    if best is None or (len(r), r) < (len(best), best):
                        best = r

                    # Backward
                    cdll.remove(new_cdll)
                    _undo_edge(og, ts_out, new_out)
                    _undo_node(og)
                    del i2o[c]
                    del o2i[new_out]

                return best  # type: ignore[return-value]

        # -- C: edge primary -> secondary --
        if ts_in in ig.neighbors(tp_in) and ts_out not in og.neighbors(tp_out):
            og.add_edge(tp_out, ts_out)
            r = _step(
                ig,
                og,
                cdll,
                tp,
                ts,
                i2o,
                o2i,
                nleft,
                eleft - 1,
                prefix + _primary_moves(a) + _secondary_moves(b) + "C",
            )
            _undo_edge(og, tp_out, ts_out)
            return r

        # -- c: edge secondary -> primary (directed only) --
        if ig.directed() and tp_in in ig.neighbors(ts_in) and tp_out not in og.neighbors(ts_out):
            og.add_edge(ts_out, tp_out)
            r = _step(
                ig,
                og,
                cdll,
                tp,
                ts,
                i2o,
                o2i,
                nleft,
                eleft - 1,
                prefix + _primary_moves(a) + _secondary_moves(b) + "c",
            )
            _undo_edge(og, ts_out, tp_out)
            return r

    raise RuntimeError(
        f"Canonical G2S: no valid operation found. Remaining: {nleft} nodes, {eleft} edges."
    )
