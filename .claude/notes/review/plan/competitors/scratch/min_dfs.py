"""Minimum DFS code (Yan & Han, gSpan, ICDM 2002) for a single connected graph.

Neither `LasseRegin/gSpan` nor `betterenvi/gSpan` exposes the minimum DFS code of one
graph: the first only has a greedy `is_canonical` with no tie branching, the second
buries the construction inside `gSpan._is_min`.  This module implements the standard
construction with correct tie branching, which is what an unlabelled corpus needs
(every label is equal, so every step is a tie).

Edge tuple: (i, j, l_i, l_ij, l_j) with i, j DFS discovery indices.
For unlabelled graphs every label is 0 and the tuple degenerates to (i, j).

DFS lexicographic order (Yan & Han, Def. 5), specialised to the extension step:
  * a backward extension always precedes a forward extension;
  * backward extensions are ordered by increasing target index j;
  * forward extensions all share the same new index j, so they are ordered by
    *decreasing* source index i (deepest point of the rightmost path first).
"""

from __future__ import annotations

import networkx as nx

Tuple5 = tuple[int, int, int, int, int]


def _rightmost_path(code: list[Tuple5]) -> list[int]:
    """DFS indices on the rightmost path, root first."""
    if not code:
        return []
    path: list[int] = []
    prev = -1
    for i, j, *_ in reversed(code):
        if i < j and (prev == -1 or j == prev):  # forward edge on the rightmost path
            if prev == -1:
                path.append(j)
            path.append(i)
            prev = i
    return list(reversed(path))


class MinDfsBudgetExceeded(RuntimeError):
    """The live embedding set outgrew `max_projections` before the code was complete."""

    def __init__(self, live: int, step: int, m: int):
        super().__init__(f"{live} live embeddings at edge {step}/{m}")
        self.live = live
        self.step = step
        self.m = m


class _Proj:
    """One partial embedding: DFS index -> graph vertex, plus the edges consumed."""

    __slots__ = ("v_of", "g_of", "used")

    def __init__(self, v_of: list[int], g_of: dict[int, int], used: set[frozenset[int]]):
        self.v_of = v_of  # DFS index -> graph node
        self.g_of = g_of  # graph node -> DFS index
        self.used = used  # frozenset({u, v}) of consumed graph edges


def min_dfs_code(
    G: nx.Graph,
    vlabel: dict[int, int] | None = None,
    elabel: dict[frozenset[int], int] | None = None,
    max_projections: int | None = None,
) -> list[Tuple5]:
    """Minimum DFS code of a connected graph with at least one edge.

    Parameters
    ----------
    max_projections
        Cap on the number of simultaneously live embeddings.  The construction keeps
        every embedding that realises the current minimal prefix, and on a large
        symmetric graph that set can grow until the process is killed by the OOM
        reaper -- measured on IAM Mutagenicity (``n_max = 92``).  Raising instead of
        dying is the difference between a recorded failure and a lost run.

    Raises
    ------
    ValueError
        If the graph is disconnected, or has no edge.  The DFS code is undefined
        in both cases; the caller must supply a convention.
    MinDfsBudgetExceeded
        If *max_projections* is set and the live embedding set exceeds it.
    """
    if G.number_of_edges() == 0:
        raise ValueError("DFS code undefined: no edges (isolated vertices carry no tuple)")
    if not nx.is_connected(G):
        raise ValueError("DFS code undefined: graph is disconnected")

    vl = vlabel or {v: 0 for v in G}
    el = elabel or {frozenset(e): 0 for e in G.edges()}
    m = G.number_of_edges()

    # --- root: minimal (l_u, l_uv, l_v) over every oriented edge -----------------
    roots: list[tuple[Tuple5, _Proj]] = []
    best: Tuple5 | None = None
    for u, v in G.edges():
        for a, b in ((u, v), (v, u)):
            t = (0, 1, vl[a], el[frozenset((a, b))], vl[b])
            if best is None or t < best:
                best = t
    for u, v in G.edges():
        for a, b in ((u, v), (v, u)):
            t = (0, 1, vl[a], el[frozenset((a, b))], vl[b])
            if t == best:
                roots.append((t, _Proj([a, b], {a: 0, b: 1}, {frozenset((a, b))})))

    code: list[Tuple5] = [best]  # type: ignore[list-item]
    projs = [p for _, p in roots]

    # --- grow, keeping only the embeddings that realise the minimum extension ---
    while len(code) < m:
        rmpath = _rightmost_path(code)
        rm_idx = rmpath[-1]  # rightmost vertex, DFS index
        cands: dict[Tuple5, list[_Proj]] = {}

        if max_projections is not None and len(projs) > max_projections:
            raise MinDfsBudgetExceeded(len(projs), len(code), m)

        for p in projs:
            rm_v = p.v_of[rm_idx]
            # backward extensions: rightmost vertex -> an ancestor on the rmpath
            for anc in rmpath[:-1]:
                anc_v = p.v_of[anc]
                e = frozenset((rm_v, anc_v))
                if G.has_edge(rm_v, anc_v) and e not in p.used:
                    t = (rm_idx, anc, vl[rm_v], el[e], vl[anc_v])
                    cands.setdefault(t, []).append(_Proj(p.v_of, p.g_of, p.used | {e}))
            # forward extensions: any rmpath vertex -> a fresh graph vertex
            new_idx = len(p.v_of)
            for src in rmpath:
                src_v = p.v_of[src]
                for w in G.neighbors(src_v):
                    if w in p.g_of:
                        continue
                    e = frozenset((src_v, w))
                    if e in p.used:
                        continue
                    t = (src, new_idx, vl[src_v], el[e], vl[w])
                    cands.setdefault(t, []).append(
                        _Proj(p.v_of + [w], {**p.g_of, w: new_idx}, p.used | {e})
                    )

        if not cands:
            raise RuntimeError("no extension available; graph traversal is inconsistent")
        t_min = min(cands, key=_ext_key)
        code.append(t_min)
        projs = cands[t_min]

    return code


def _ext_key(t: Tuple5) -> tuple:
    """Sort key realising the DFS lexicographic order on one extension step."""
    i, j, li, lij, lj = t
    if i > j:  # backward: smallest first, ordered by target j
        return (0, j, lij)
    return (1, -i, lij, lj)  # forward: deeper source i is smaller


# ---------------------------------------------------------------------------
# serialisation and inverse
# ---------------------------------------------------------------------------


def code_to_string(code: list[Tuple5], labelled: bool = False) -> str:
    """Compact ASCII serialisation.  Unlabelled: '0-1 1-2 2-0'."""
    if labelled:
        return " ".join(f"{i}-{j}:{li},{lij},{lj}" for i, j, li, lij, lj in code)
    return " ".join(f"{i}-{j}" for i, j, *_ in code)


def code_to_graph(code: list[Tuple5]) -> nx.Graph:
    """Inverse map: a DFS code rebuilds the graph up to isomorphism."""
    H = nx.Graph()
    for i, j, li, lij, lj in code:
        H.add_node(i, label=li)
        H.add_node(j, label=lj)
        H.add_edge(i, j, label=lij)
    return H
