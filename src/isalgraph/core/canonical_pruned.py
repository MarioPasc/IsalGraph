"""Pruned canonical IsalGraph string computation.

Variant of the exhaustive canonical algorithm (``canonical.py``) with
**structural triplet pruning** at V/v branch points. At each branch, only
candidates sharing the maximum lexicographic triplet (|N_1|, |N_2|, |N_3|)
are explored. Nodes with different triplets cannot be equivalent under any
automorphism, so deterministic selection among them preserves the complete
invariant property while dramatically reducing the branching factor.

The triplet for a node v is (|N_1(v)|, |N_2(v)|, |N_3(v)|) where N_k(v)
is the set of nodes at distance exactly k from v in the **input** graph.
These are precomputed once via BFS truncated at distance 3.

**Complete invariant property**: ``pruned_canonical_string(G1) ==
pruned_canonical_string(G2)`` if and only if ``G1 ~ G2`` (isomorphic).

**Proof**: At each V/v branch point, the max-triplet candidate set is
automorphism-invariant (BFS distance counts are preserved under isomorphism).
Exhaustive backtracking within this set produces a labeling-independent
result. Conversely, equal strings decode to isomorphic graphs via S2G.

**Note**: The pruned canonical string may differ from ``canonical_string(G)``
(the unpruned exhaustive version). It defines a different canonical form
that is equally valid as a complete invariant. The pruned version may
produce longer strings on some graphs because it restricts which candidates
are explored, but this does not affect its invariant properties.
"""

from __future__ import annotations

from collections import deque

from isalgraph.core.canonical import (
    _is_reachable,
    _primary_moves,
    _secondary_moves,
    _undo_edge,
    _undo_node,
    _walk,
    levenshtein,
)
from isalgraph.core.cdll import CircularDoublyLinkedList
from isalgraph.core.graph_to_string import generate_pairs_sorted_by_sum
from isalgraph.core.sparse_graph import SparseGraph

# ------------------------------------------------------------------
# Structural triplet computation
# ------------------------------------------------------------------


def _bfs_distance_counts(graph: SparseGraph, source: int) -> tuple[int, int, int]:
    """BFS from source, count nodes at distance 1, 2, 3.

    Early-stops when all reachable nodes at distance <= 3 are visited.

    Args:
        graph: The graph to search.
        source: Starting node index.

    Returns:
        Tuple (|N_1|, |N_2|, |N_3|): counts of nodes at distance 1, 2, 3.
    """
    n = graph.node_count()
    dist: list[int] = [-1] * n
    dist[source] = 0

    queue: deque[int] = deque()
    queue.append(source)

    counts = [0, 0, 0]  # index 0 -> distance 1, etc.

    while queue:
        u = queue.popleft()
        d = dist[u]
        if d >= 3:
            continue
        for v in graph.neighbors(u):
            if dist[v] == -1:
                dist[v] = d + 1
                if dist[v] <= 3:
                    counts[dist[v] - 1] += 1
                    queue.append(v)

    return (counts[0], counts[1], counts[2])


def compute_structural_triplets(graph: SparseGraph) -> list[tuple[int, int, int]]:
    """For each node v, compute (|N_1(v)|, |N_2(v)|, |N_3(v)|) via BFS.

    Complexity: O(N * (N + E)) where N = nodes, E = edges.

    Args:
        graph: The graph to compute triplets for.

    Returns:
        List of triplets indexed by node ID.
    """
    n = graph.node_count()
    return [_bfs_distance_counts(graph, v) for v in range(n)]


# ------------------------------------------------------------------
# Pruned canonical string computation
# ------------------------------------------------------------------


def pruned_canonical_string(graph: SparseGraph) -> str:
    """Compute the canonical IsalGraph string with triplet pruning.

    Same contract as ``canonical_string`` but with structural triplet
    pruning at V/v branch points. Only candidates sharing the maximum
    lexicographic triplet are explored.

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
    if n == 0:
        return ""
    if n == 1 and graph.logical_edge_count() == 0:
        return ""

    # Precompute triplets once (shared across all starting nodes)
    triplets = compute_structural_triplets(graph)

    best: str | None = None
    for v in range(n):
        if not _is_reachable(graph, v):
            continue
        w = _pruned_canonical_g2s(graph, v, triplets)
        if best is None or (len(w), w) < (len(best), best):
            best = w

    if best is None:
        raise ValueError(
            "No starting node can reach all other nodes. "
            "For undirected graphs, the graph must be connected. "
            "For directed graphs, at least one node must reach all others."
        )

    return best


def pruned_graph_distance(g1: SparseGraph, g2: SparseGraph) -> int:
    """Approximate graph edit distance via Levenshtein on pruned canonical strings.

    Args:
        g1: First graph.
        g2: Second graph.

    Returns:
        Levenshtein edit distance between pruned canonical strings.
    """
    return levenshtein(pruned_canonical_string(g1), pruned_canonical_string(g2))


# ------------------------------------------------------------------
# Internal: pruned exhaustive G2S search with backtracking
# ------------------------------------------------------------------


def _pruned_canonical_g2s(
    input_graph: SparseGraph,
    start_node: int,
    triplets: list[tuple[int, int, int]],
) -> str:
    """Find the shortest, then lex-smallest G2S string from *start_node*.

    Identical to ``_canonical_g2s`` but prunes candidates at V/v branch
    points using structural triplets.
    """
    max_n = input_graph.max_nodes()
    og = SparseGraph(max_n, input_graph.directed())
    cdll = CircularDoublyLinkedList(max_n)

    n0 = og.add_node()
    c0 = cdll.insert_after(-1, n0)
    i2o: dict[int, int] = {start_node: n0}
    o2i: dict[int, int] = {n0: start_node}

    return _pruned_step(
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
        triplets,
    )


def _pruned_step(  # noqa: ANN001
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
    triplets: list[tuple[int, int, int]],
) -> str:
    """One step of the pruned exhaustive G2S search.

    Identical to ``_step`` in ``canonical.py`` except that at V/v branch
    points, candidates are filtered to those with the maximum structural
    triplet before backtracking.
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
                # PRUNING: only keep candidates with max triplet
                max_trip = max(triplets[c] for c in cands)
                pruned = [c for c in cands if triplets[c] == max_trip]

                mov = _primary_moves(a)
                best: str | None = None
                for c in pruned:
                    # Forward
                    new_out = og.add_node()
                    i2o[c] = new_out
                    o2i[new_out] = c
                    og.add_edge(tp_out, new_out)
                    new_cdll = cdll.insert_after(tp, new_out)

                    r = _pruned_step(
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
                        triplets,
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
                # PRUNING: only keep candidates with max triplet
                max_trip = max(triplets[c] for c in cands)
                pruned = [c for c in cands if triplets[c] == max_trip]

                mov = _secondary_moves(b)
                best = None
                for c in pruned:
                    # Forward
                    new_out = og.add_node()
                    i2o[c] = new_out
                    o2i[new_out] = c
                    og.add_edge(ts_out, new_out)
                    new_cdll = cdll.insert_after(ts, new_out)

                    r = _pruned_step(
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
                        triplets,
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
            r = _pruned_step(
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
                triplets,
            )
            _undo_edge(og, tp_out, ts_out)
            return r

        # -- c: edge secondary -> primary (directed only) --
        if ig.directed() and tp_in in ig.neighbors(ts_in) and tp_out not in og.neighbors(ts_out):
            og.add_edge(ts_out, tp_out)
            r = _pruned_step(
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
                triplets,
            )
            _undo_edge(og, ts_out, tp_out)
            return r

    raise RuntimeError(
        f"Pruned canonical G2S: no valid operation found. Remaining: {nleft} nodes, {eleft} edges."
    )
