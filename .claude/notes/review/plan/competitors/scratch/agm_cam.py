"""AGM canonical adjacency-matrix code (Inokuchi et al., PKDD 2000).

Definition used, following Jiang, Coenen & Zito (KER 28(1):75-105, 2013), sec. 3.1:

    "Given an adjacency matrix M of a graph g, an encoding of M can be obtained by
     the sequence obtained from concatenating the lower (or upper) triangular entries
     of M, including entries on the diagonal.  Since different permutations of the set
     of vertexes correspond to different adjacency matrices, the canonical (CAM) form
     of g is defined as the maximal (or minimal) encoding."

AGM takes the **minimum**; FFSM (Huan et al., ICDM 2003) takes the maximum.  We fix
AGM's convention and say so.  For an unlabelled simple graph the diagonal is all-zero
and every vertex label is equal, so the code reduces to the strict lower triangle read
row by row:

    code(pi) = x_{2,1} | x_{3,1} x_{3,2} | x_{4,1} x_{4,2} x_{4,3} | ...

That reading order has the prefix property -- the first k(k-1)/2 bits depend only on
the first k vertices of the permutation -- which is what makes branch and bound
possible at all.

Exact minimisation is a lex-leader canonical form and is NP-hard in general; this is a
branch-and-bound over vertex-permutation prefixes with (a) a degree-sequence-based
initial incumbent, (b) prefix pruning against the incumbent, and (c) an optional node
budget so the caller can detect the intractable cases instead of hanging.
"""

from __future__ import annotations

import itertools

import networkx as nx


class AGMBudgetExceeded(RuntimeError):
    """The branch-and-bound hit its node budget before proving optimality."""

    def __init__(self, best: str, nodes: int):
        super().__init__(f"budget of {nodes} search nodes exhausted; incumbent only")
        self.best = best
        self.nodes = nodes


def _code_from_perm(A: list[set[int]], perm: list[int]) -> str:
    bits: list[str] = []
    for k in range(1, len(perm)):
        vk = perm[k]
        for j in range(k):
            bits.append("1" if perm[j] in A[vk] else "0")
    return "".join(bits)


def agm_canonical_code(
    G: nx.Graph,
    *,
    node_budget: int = 2_000_000,
    exact: bool = True,
) -> tuple[str, int]:
    """Minimum CAM code of an unlabelled simple graph.

    Returns
    -------
    (code, nodes_expanded)

    Raises
    ------
    AGMBudgetExceeded
        If *exact* and the budget runs out before the search closes.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return "", 0
    if n == 1:
        return "", 0
    idx = {v: i for i, v in enumerate(nodes)}
    A = [set(idx[w] for w in G.neighbors(v)) for v in nodes]

    # incumbent: greedy, low-degree-first, ties by fewest links to the placed prefix
    order: list[int] = []
    placed: set[int] = set()
    for _ in range(n):
        cand = min(
            (v for v in range(n) if v not in placed),
            key=lambda v: (len(A[v] & placed), len(A[v]), v),
        )
        order.append(cand)
        placed.add(cand)
    best = _code_from_perm(A, order)

    expanded = 0
    prefix: list[int] = []
    used = [False] * n

    def rec(pos: int, code_so_far: str) -> None:
        nonlocal best, expanded
        expanded += 1
        if expanded > node_budget:
            raise AGMBudgetExceeded(best, expanded)
        if pos == n:
            if code_so_far < best:
                best = code_so_far
            return
        # candidate extensions, ordered by the bits they contribute
        cands = []
        for v in range(n):
            if used[v]:
                continue
            bits = "".join("1" if prefix[j] in A[v] else "0" for j in range(pos))
            cands.append((bits, v))
        cands.sort()
        for bits, v in cands:
            nxt = code_so_far + bits
            # prefix pruning against the incumbent
            cmp_len = len(nxt)
            if nxt > best[:cmp_len]:
                continue
            used[v] = True
            prefix.append(v)
            rec(pos + 1, nxt)
            prefix.pop()
            used[v] = False

    if not exact:
        return best, 0
    rec(0, "")
    return best, expanded


def brute_force_code(G: nx.Graph) -> str:
    """Reference implementation: minimum over every permutation.  n <= 8 only."""
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    A = [set(idx[w] for w in G.neighbors(v)) for v in nodes]
    return min(_code_from_perm(A, list(p)) for p in itertools.permutations(range(len(nodes))))


def code_to_graph(code: str, n: int) -> nx.Graph:
    """Inverse map: the code plus n rebuilds the graph exactly."""
    H = nx.Graph()
    H.add_nodes_from(range(n))
    it = iter(code)
    for k in range(1, n):
        for j in range(k):
            if next(it) == "1":
                H.add_edge(k, j)
    return H
