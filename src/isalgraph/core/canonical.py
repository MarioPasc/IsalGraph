"""Canonical IsalGraph string computation.

The canonical string for a graph G with N nodes is:

    w*_G = lexmin{ w in argmin_{v in V} |G2S(G, v)| }

That is: run GraphToString from every starting node v in V, collect all
resulting strings, keep only those of minimum length, and among those
select the lexicographically smallest.

This is a **complete graph invariant**: w*_G = w*_H if and only if G ~ H
(graph isomorphism).

The function also exposes an approximate graph distance metric via
Levenshtein edit distance between canonical strings.
"""

from __future__ import annotations

from isalgraph.core.graph_to_string import GraphToString
from isalgraph.core.sparse_graph import SparseGraph


def canonical_string(graph: SparseGraph) -> str:
    """Compute the canonical IsalGraph string for a graph.

    Runs GraphToString from every starting node, filters to minimum-length
    strings, and returns the lexicographically smallest.

    Args:
        graph: The graph to compute the canonical string for.
            Must be connected (undirected) or have all nodes reachable
            from every node (directed).

    Returns:
        The canonical string w*_G.

    Raises:
        ValueError: If the graph has no nodes, or if some starting node
            cannot reach all other nodes.
    """
    n = graph.node_count()
    if n == 0:
        return ""

    if n == 1:
        return ""

    strings: list[str] = []
    for v in range(n):
        gts = GraphToString(graph)
        w, _ = gts.run(v)
        strings.append(w)

    min_len = min(len(w) for w in strings)
    candidates = [w for w in strings if len(w) == min_len]
    return min(candidates)


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
