"""Wave-0: does §4.1 reproduce EXACTLY once real_suite1.py's rng stream is replicated?

real_suite1.py advances one Random(42) through, per dataset:
  1. idx = rng.sample(range(N), 50)                      -- the F3 graph draw
  2. 5 encoders x 50 graphs x 20 shuffled_copy(rng)      -- 5,000 variable-length draws
  3. sub = rng.sample(range(N), 200)                     -- the rho draw
Only step 3 matters for rho, but step 2 must be consumed identically to reach it.
"""

from __future__ import annotations

import math
import random
import sys

import networkx as nx
import numpy as np
from rapidfuzz.distance import Levenshtein as RFLev
from scipy.stats import spearmanr

sys.path.insert(
    0, "/home/mpascual/research/code/IsalGraph/.claude/notes/review/plan/competitors/scratch"
)
import agm_cam  # noqa: E402
from repro_41 import (  # noqa: E402
    DATASETS,
    README,
    RHO_SAMPLE,
    SEED,
    adj_colwise,
    adj_rowmajor,
    g6,
    isal_strings,
    load,
    mdfs_tuplestring,
    nauty_g6,
    s6,
)

AGM_BUDGET = 200_000


def shuffled_copy(G: nx.Graph, rng: random.Random) -> nx.Graph:
    nodes = list(G.nodes())
    new = list(range(len(nodes)))
    rng.shuffle(new)
    mapping = dict(zip(nodes, new))
    order = list(new)
    rng.shuffle(order)
    H = nx.Graph()
    H.add_nodes_from(order)
    edges = [(mapping[u], mapping[v]) for u, v in G.edges()]
    rng.shuffle(edges)
    H.add_edges_from(edges)
    return H


# real_suite1.py's ENCODERS dict, in insertion order -- the F3 loop iterates it
ENCODERS = {
    "graph6": g6,
    "sparse6": s6,
    "nauty->graph6": nauty_g6,
    "adjacency": adj_rowmajor,
    "min-DFS (tuples)": mdfs_tuplestring,
}


def main() -> None:
    only = sys.argv[1:] or DATASETS
    rng = random.Random(SEED)
    print(f"{'dataset':<17}{'method':<14}{'README':>8}{'measured':>10}{'delta':>9}")
    for ds in DATASETS:
        graphs, ged, cert = load(ds)
        N = len(graphs)

        # --- step 1+2: consume the rng exactly as real_suite1.py's F3 block does ---
        idx = rng.sample(range(N), min(50, N))
        f3 = {}
        for k, fn in ENCODERS.items():
            inv = sum(
                len({fn(shuffled_copy(graphs[i], rng)) for _ in range(20)} | {fn(graphs[i])}) == 1
                for i in idx
            )
            f3[k] = f"{inv}/{len(idx)}"

        # --- step 3: the rho draw ---
        sub = sorted(rng.sample(range(N), min(RHO_SAMPLE, N)))
        if ds not in only:
            continue
        print(f"{ds:<17}F3: " + "  ".join(f"{k}={v}" for k, v in f3.items()))

        codes: dict[str, list] = {
            "adjacency": [adj_rowmajor(G) for G in graphs],
            "adj_colwise": [adj_colwise(G) for G in graphs],
            "graph6": [g6(G) for G in graphs],
            "sparse6": [s6(G) for G in graphs],
            "nauty": [nauty_g6(G) for G in graphs],
            "mdfs": [mdfs_tuplestring(G) for G in graphs],
        }
        agm = []
        for G in graphs:
            try:
                agm.append(agm_cam.agm_canonical_code(G, node_budget=AGM_BUDGET)[0])
            except agm_cam.AGMBudgetExceeded:
                agm.append(None)
        codes["agm"] = agm
        codes["isal"] = isal_strings(graphs)
        order = [G.number_of_nodes() for G in graphs]

        pairs = [(a, b) for ii, a in enumerate(sub) for b in sub[ii + 1 :] if cert[a, b]]
        gd = np.array([ged[a, b] for a, b in pairs])
        dn = np.array([abs(order[a] - order[b]) for a, b in pairs], dtype=float)
        rows = [("null", float(spearmanr(dn, gd).statistic))]
        for k in (
            "adjacency",
            "adj_colwise",
            "graph6",
            "sparse6",
            "nauty",
            "agm",
            "mdfs",
            "isal",
        ):
            cs = codes[k]
            if any(cs[a] is None or cs[b] is None for a, b in pairs):
                rows.append((k, math.nan))
                continue
            d = np.array([RFLev.distance(cs[a], cs[b]) for a, b in pairs], dtype=float)
            rows.append((k, float(spearmanr(d, gd).statistic)))
        for k, v in rows:
            ref = README[ds].get(k)
            delta = "" if ref is None or math.isnan(v) else f"{v - ref:+.4f}"
            refs = "-" if ref is None else f"{ref:.3f}"
            print(f"{ds:<17}{k:<14}{refs:>8}{v:>10.4f}{delta:>9}")
        print(f"{'':17}(pairs={len(pairs)})")


if __name__ == "__main__":
    main()
