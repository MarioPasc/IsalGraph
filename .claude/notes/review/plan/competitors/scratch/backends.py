"""One `encode(G) -> str` per competitor, plus bit accounting and inverse maps.

Scratchpad only.  This is the shape `src/isalgraph/competitors/` should take, not the
code to install there.
"""

from __future__ import annotations

import collections
import math
import subprocess
import sys

import agm_cam
import min_dfs
import networkx as nx

# --------------------------------------------------------------------------- #
# graph6 / sparse6  (McKay's formats; networkx implements both)
# --------------------------------------------------------------------------- #


def graph6(G: nx.Graph) -> str:
    """Raw graph6 on the node order networkx happens to hold."""
    H = nx.convert_node_labels_to_integers(G, ordering="sorted")
    return nx.to_graph6_bytes(H, header=False).decode().strip()


def sparse6(G: nx.Graph) -> str:
    H = nx.convert_node_labels_to_integers(G, ordering="sorted")
    return nx.to_sparse6_bytes(H, header=False).decode().strip()


# --------------------------------------------------------------------------- #
# nauty canonical labelling -> graph6
# --------------------------------------------------------------------------- #


def nauty_canon_graph6(G: nx.Graph) -> str:
    """Relabel by nauty's canonical labelling, then serialise as graph6."""
    import pynauty

    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    adj = {idx[v]: [idx[w] for w in G.neighbors(v)] for v in nodes}
    pg = pynauty.Graph(len(nodes), directed=False, adjacency_dict=adj)
    lab = pynauty.canon_label(pg)  # canonical order: lab[i] = old vertex
    pos = {old: new for new, old in enumerate(lab)}
    H = nx.Graph()
    H.add_nodes_from(range(len(nodes)))
    H.add_edges_from((pos[idx[u]], pos[idx[v]]) for u, v in G.edges())
    return nx.to_graph6_bytes(H, header=False).decode().strip()


def nauty_certificate(G: nx.Graph) -> bytes:
    import pynauty

    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    adj = {idx[v]: [idx[w] for w in G.neighbors(v)] for v in nodes}
    return pynauty.certificate(pynauty.Graph(len(nodes), directed=False, adjacency_dict=adj))


# --------------------------------------------------------------------------- #
# adjacency matrix, raw and AGM-canonical
# --------------------------------------------------------------------------- #


def adjacency_bits(G: nx.Graph) -> str:
    """Strict upper triangle, row-major, on the incident node order."""
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    A = [[0] * len(nodes) for _ in nodes]
    for u, v in G.edges():
        A[idx[u]][idx[v]] = A[idx[v]][idx[u]] = 1
    n = len(nodes)
    return "".join(str(A[i][j]) for i in range(n) for j in range(i + 1, n))


def agm_code(G: nx.Graph, budget: int = 2_000_000) -> tuple[str, int]:
    return agm_cam.agm_canonical_code(G, node_budget=budget)


# --------------------------------------------------------------------------- #
# gSpan minimum DFS code
# --------------------------------------------------------------------------- #


def mdfsc(G: nx.Graph) -> str:
    return min_dfs.code_to_string(min_dfs.min_dfs_code(G))


# --------------------------------------------------------------------------- #
# Weisfeiler-Lehman subtree feature vector (Shervashidze et al., JMLR 2011)
# --------------------------------------------------------------------------- #


def wl_features(G: nx.Graph, h: int = 3) -> dict[str, int]:
    """Multiset of WL colours over h refinement rounds.  Unlabelled: all start at ''."""
    colour = {v: "0" for v in G}
    feats: collections.Counter[str] = collections.Counter()
    feats.update(f"0:{c}" for c in colour.values())
    for it in range(1, h + 1):
        new = {}
        for v in G:
            sig = colour[v] + "|" + ",".join(sorted(colour[w] for w in G.neighbors(v)))
            new[v] = sig
        # compress to keep strings short, order-independently
        table = {s: str(i) for i, s in enumerate(sorted(set(new.values())))}
        colour = {v: table[s] for v, s in new.items()}
        feats.update(f"{it}:{new[v]}" for v in G)
    return dict(feats)


def wl_distance(fa: dict[str, int], fb: dict[str, int]) -> float:
    """Kernel distance sqrt(K(a,a) + K(b,b) - 2 K(a,b)) for the linear WL kernel.

    NOTE: WL colours are only comparable across graphs when the compression table is
    shared.  Here the round-0 and round-1..h signatures are built from the *uncompressed*
    parent signature, so they are comparable; this is the standard construction.
    """
    keys = set(fa) | set(fb)
    kaa = sum(fa.get(k, 0) ** 2 for k in keys)
    kbb = sum(fb.get(k, 0) ** 2 for k in keys)
    kab = sum(fa.get(k, 0) * fb.get(k, 0) for k in keys)
    return math.sqrt(max(kaa + kbb - 2 * kab, 0.0))


# --------------------------------------------------------------------------- #
# IsalGraph (out-of-process: it lives in the conda env, not this venv)
# --------------------------------------------------------------------------- #

ISAL_PY = "/home/mpascual/.conda/envs/isalgraph-cpp/bin/python"

_ISAL_SNIPPET = r"""
import json, sys, time
import isalgraph
from isalgraph import SparseGraph, pruned_canonical_string, canonical_string
assert isalgraph.engine() == "cpp", isalgraph.engine()
budget, payload = json.load(sys.stdin)
out = []
for n, edges in payload:
    g = SparseGraph(n, False)
    for _ in range(n):
        g.add_node()
    for u, v in edges:
        g.add_edge(u, v)
    rec = {}
    for name, fn in (("pruned", pruned_canonical_string), ("canonical", canonical_string)):
        t0 = time.perf_counter()
        try:
            rec[name] = fn(g, timeout_s=budget)
            rec[name + "_s"] = time.perf_counter() - t0
        except Exception as exc:
            rec[name] = None
            rec[name + "_s"] = time.perf_counter() - t0
            rec[name + "_err"] = type(exc).__name__
    out.append(rec)
print(json.dumps(out))
"""


def isalgraph_strings(graphs: list[nx.Graph], budget: float = 30.0) -> list[dict]:
    import json

    payload = []
    for G in graphs:
        H = nx.convert_node_labels_to_integers(G, ordering="sorted")
        payload.append([H.number_of_nodes(), [list(e) for e in H.edges()]])
    p = subprocess.run(
        [ISAL_PY, "-c", _ISAL_SNIPPET],
        input=json.dumps([budget, payload]),
        capture_output=True,
        text=True,
        check=False,
    )
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-3000:])
    return json.loads(p.stdout)


# --------------------------------------------------------------------------- #
# bit accounting -- the two conventions competitors.md §5 locks
# --------------------------------------------------------------------------- #


def entropy_bits(length: int, alphabet: int) -> float:
    return length * math.log2(alphabet)


def realised_bits(s: str | bytes) -> int:
    return len(s) * 8


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def hamming(a: str, b: str) -> int | None:
    return None if len(a) != len(b) else sum(x != y for x, y in zip(a, b))


def padded_hamming(G: nx.Graph, H: nx.Graph) -> int:
    """Embed both adjacency matrices in a common max(n1, n2) frame, compare triangles."""
    n = max(G.number_of_nodes(), H.number_of_nodes())

    def tri(K):
        nodes = list(K.nodes())
        idx = {v: i for i, v in enumerate(nodes)}
        s = set()
        for u, v in K.edges():
            a, b = sorted((idx[u], idx[v]))
            s.add((a, b))
        return s

    a, b = tri(G), tri(H)
    return len(a ^ b)


if __name__ == "__main__":
    print("module; run probe.py", file=sys.stderr)
