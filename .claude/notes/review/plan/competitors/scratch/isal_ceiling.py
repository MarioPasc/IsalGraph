"""Does `timeout_s` actually bound wall-clock?  And where does each IsalGraph variant stop?

Run with the isalgraph-cpp env python, not the scratch venv.
"""

import random
import time

import networkx as nx

import isalgraph
from isalgraph import SparseGraph, canonical_string, pruned_canonical_string


def mk(n, m, rng):
    """Random spanning tree + uniform extra edges.  Rejection sampling on G(n, m) does
    not terminate at m ~ n above n ~ 30 -- such graphs are almost surely disconnected."""
    G = nx.random_labeled_tree(n, seed=rng.randrange(10**9))
    extra = m - (n - 1)
    if extra > 0:
        non = [(u, v) for u in range(n) for v in range(u + 1, n) if not G.has_edge(u, v)]
        rng.shuffle(non)
        G.add_edges_from(non[:extra])
    assert nx.is_connected(G)
    return G


def to_sg(G):
    H = nx.convert_node_labels_to_integers(G, ordering="sorted")
    g = SparseGraph(H.number_of_nodes(), False)
    for _ in range(H.number_of_nodes()):
        g.add_node()
    for u, v in H.edges():
        g.add_edge(u, v)
    return g


assert isalgraph.engine() == "cpp"
rng = random.Random(42)
print(
    f"{'n':>4}{'m':>5}{'variant':>12}{'timeout_s':>11}{'elapsed s':>11}{'len':>7}  outcome",
    flush=True,
)
for n, m in [(20, 21), (30, 31), (50, 52), (70, 73), (98, 103), (30, 60), (50, 100)]:
    G = mk(n, m, rng)
    g = to_sg(G)
    for name, fn in (("pruned", pruned_canonical_string), ("canonical", canonical_string)):
        t0 = time.perf_counter()
        try:
            s = fn(g, timeout_s=5.0)
            dt = time.perf_counter() - t0
            flag = "  <-- OVERRAN timeout_s" if dt > 6.0 else ""
            print(f"{n:>4}{m:>5}{name:>12}{5.0:>11}{dt:>11.2f}{len(s):>7}  ok{flag}", flush=True)
        except Exception as e:
            dt = time.perf_counter() - t0
            flag = "  <-- OVERRAN timeout_s" if dt > 6.0 else ""
            print(
                f"{n:>4}{m:>5}{name:>12}{5.0:>11}{dt:>11.2f}{'-':>7}  {type(e).__name__}{flag}",
                flush=True,
            )
