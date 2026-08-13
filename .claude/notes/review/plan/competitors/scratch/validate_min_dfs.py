"""Validate min_dfs.min_dfs_code against brute-force enumeration of every DFS code.

Three checks:
  V1  brute force  -- the returned code equals the lexicographic minimum over all
                      valid DFS traversals (exhaustive, n <= 6).
  V2  invariance   -- 30 random relabellings of the same graph give one code.
  V3  completeness -- two graphs share a code iff they are isomorphic
                      (over all connected graphs on n <= 6 nodes).
"""

from __future__ import annotations

import itertools
import random
import sys

import networkx as nx
from min_dfs import Tuple5, code_to_graph, min_dfs_code


# --- general DFS lexicographic order (Yan & Han, Def. 5) -----------------------
def edge_lt(e1: Tuple5, e2: Tuple5) -> bool:
    i1, j1 = e1[0], e1[1]
    i2, j2 = e2[0], e2[1]
    f1, f2 = i1 < j1, i2 < j2
    if f1 and f2:
        if j1 != j2:
            return j1 < j2
        if i1 != i2:
            return i1 > i2
        return (e1[2], e1[3], e1[4]) < (e2[2], e2[3], e2[4])
    if not f1 and not f2:
        if i1 != i2:
            return i1 < i2
        if j1 != j2:
            return j1 < j2
        return e1[3] < e2[3]
    if not f1 and f2:
        return i1 < j2
    return j1 <= i2


def code_lt(c1: list[Tuple5], c2: list[Tuple5]) -> bool:
    for a, b in zip(c1, c2):
        if a != b:
            return edge_lt(a, b)
    return len(c1) < len(c2)


def all_dfs_codes(G: nx.Graph) -> list[list[Tuple5]]:
    """Every valid DFS code of a connected unlabelled graph (exponential)."""
    m = G.number_of_edges()
    out: list[list[Tuple5]] = []

    def rmpath(code):
        if not code:
            return []
        path, prev = [], -1
        for i, j, *_ in reversed(code):
            if i < j and (prev == -1 or j == prev):
                if prev == -1:
                    path.append(j)
                path.append(i)
                prev = i
        return list(reversed(path))

    def rec(code, v_of, g_of, used):
        if len(code) == m:
            out.append(list(code))
            return
        rmp = rmpath(code)
        rm_idx = rmp[-1]
        rm_v = v_of[rm_idx]
        for anc in rmp[:-1]:
            av = v_of[anc]
            e = frozenset((rm_v, av))
            if G.has_edge(rm_v, av) and e not in used:
                rec(code + [(rm_idx, anc, 0, 0, 0)], v_of, g_of, used | {e})
        new_idx = len(v_of)
        for src in rmp:
            sv = v_of[src]
            for w in G.neighbors(sv):
                if w in g_of:
                    continue
                e = frozenset((sv, w))
                if e in used:
                    continue
                rec(code + [(src, new_idx, 0, 0, 0)], v_of + [w], {**g_of, w: new_idx}, used | {e})

    for u, v in G.edges():
        for a, b in ((u, v), (v, u)):
            rec([(0, 1, 0, 0, 0)], [a, b], {a: 0, b: 1}, {frozenset((a, b))})
    return out


def connected_graphs(n: int):
    """All connected graphs on exactly n labelled nodes, deduplicated by isomorphism."""
    nodes = list(range(n))
    pairs = list(itertools.combinations(nodes, 2))
    seen: list[nx.Graph] = []
    for mask in range(1 << len(pairs)):
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(p for k, p in enumerate(pairs) if mask >> k & 1)
        if not nx.is_connected(G):
            continue
        if any(nx.is_isomorphic(G, H) for H in seen):
            continue
        seen.append(G)
    return seen


def main() -> int:
    rng = random.Random(42)
    fails = 0

    # ---- V1 + V3 ------------------------------------------------------------
    for n in (2, 3, 4, 5, 6):
        graphs = connected_graphs(n)
        codes: dict[tuple, nx.Graph] = {}
        brute_checked = 0
        for G in graphs:
            got = min_dfs_code(G)
            if n <= 5:  # exhaustive enumeration only where cheap
                exp = min(
                    all_dfs_codes(G),
                    key=__import__("functools").cmp_to_key(
                        lambda a, b: -1 if code_lt(a, b) else (1 if code_lt(b, a) else 0)
                    ),
                )
                brute_checked += 1
                if got != exp:
                    print(f"V1 FAIL n={n} edges={sorted(G.edges())}\n  got {got}\n  exp {exp}")
                    fails += 1
            key = tuple(got)
            if key in codes and not nx.is_isomorphic(G, codes[key]):
                print(
                    f"V3 FAIL collision n={n}: {sorted(G.edges())} vs {sorted(codes[key].edges())}"
                )
                fails += 1
            codes[key] = G
            # reversibility
            if not nx.is_isomorphic(code_to_graph(got), G):
                print(f"V-REV FAIL n={n} {sorted(G.edges())}")
                fails += 1
        print(
            f"n={n}: {len(graphs)} connected iso-classes, {len(codes)} distinct codes, "
            f"{brute_checked} brute-force-checked"
        )

    # ---- V2 -----------------------------------------------------------------
    inv_fail = 0
    trials = 0
    for n in (5, 6, 7, 8, 9, 10):
        for _ in range(40):
            G = nx.gnp_random_graph(n, 0.35, seed=rng.randrange(10**9))
            if not nx.is_connected(G) or G.number_of_edges() == 0:
                continue
            base = min_dfs_code(G)
            for _ in range(30):
                perm = list(G.nodes())
                rng.shuffle(perm)
                H = nx.relabel_nodes(G, dict(zip(G.nodes(), perm)), copy=True)
                trials += 1
                if min_dfs_code(H) != base:
                    inv_fail += 1
    print(f"V2 invariance: {trials} relabellings, {inv_fail} mismatches")
    fails += inv_fail

    print("ALL OK" if fails == 0 else f"{fails} FAILURES")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
