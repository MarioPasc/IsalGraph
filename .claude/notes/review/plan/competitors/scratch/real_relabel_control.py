"""Is the non-canonical representations' rho structural, or an artefact of the corpus?

real_suite1.py measures rho(Levenshtein-on-adjacency, exact GED) = 0.75-0.86 -- higher
than IsalGraph on three of five datasets -- from a representation that is invariant on
0-6 of 50 graphs.  Two explanations:

  (i)  structural: the bit vector genuinely tracks GED;
  (ii) artefact: IAM and GraphEdX graphs carry a *consistent* incidental vertex order
       (drawing order, node id), so the "arbitrary" labelling is correlated across
       graphs and the distance is reading that correlation.

The test discriminates them in one line: relabel every graph independently at random and
recompute.  Under (i) rho is unchanged.  Under (ii) it collapses.  Canonical
representations are the control -- their rho must not move at all.
"""

from __future__ import annotations

import json
import random

import backends as B
import min_dfs
import networkx as nx
import numpy as np
from rapidfuzz.distance import Levenshtein as RFLev
from scipy.stats import spearmanr
from sweep import shuffled_copy

ROOT = "/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data"
GEDD = f"{ROOT}/source/GED_PRECOMPUTED/extended_merged_exact_ged/computed"
DATASETS = ["iam_letter_low", "iam_letter_high", "linux", "aids"]
RHO_SAMPLE = 200
SEED = 42


def load(ds):
    z = np.load(f"{ROOT}/exported/{ds}.npz", allow_pickle=True)
    g = np.load(f"{GEDD}/{ds}.npz", allow_pickle=True)
    off, ed = z["edge_offsets"], z["edges"]
    graphs = []
    for i, n in enumerate(z["n_nodes"]):
        G = nx.Graph()
        G.add_nodes_from(range(int(n)))
        G.add_edges_from(
            zip(ed[0, off[i] : off[i + 1]].tolist(), ed[1, off[i] : off[i + 1]].tolist())
        )
        graphs.append(G)
    return graphs, g["ged_matrix"], g["certified_mask"]


def tuple_string(G):
    return "".join(chr(0xE000 + i * 100 + j) for i, j, *_ in min_dfs.min_dfs_code(G))


ENC = {
    "adjacency": B.adjacency_bits,
    "graph6": B.graph6,
    "sparse6": B.sparse6,
    "nauty->graph6": B.nauty_canon_graph6,
    "min-DFS (tuples)": tuple_string,
}


def rho_for(graphs, pairs, gd, fn):
    cs = {i: fn(graphs[i]) for i in {x for p in pairs for x in p}}
    d = np.array([RFLev.distance(cs[a], cs[b]) for a, b in pairs], dtype=float)
    return float(spearmanr(d, gd).statistic)


def main():
    rng = random.Random(SEED)
    out = {}
    print(
        f"{'dataset':<18}{'representation':<20}{'rho as shipped':>16}"
        f"{'rho relabelled':>16}{'delta':>9}"
    )
    for ds in DATASETS:
        graphs, ged, cert = load(ds)
        N = len(graphs)
        sub = sorted(rng.sample(range(N), min(RHO_SAMPLE, N)))
        pairs = [(a, b) for ii, a in enumerate(sub) for b in sub[ii + 1 :] if cert[a, b]]
        gd = np.array([ged[a, b] for a, b in pairs])
        shuffled = {i: shuffled_copy(graphs[i], rng) for i in sub}
        sg = [shuffled.get(i, graphs[i]) for i in range(N)]
        for name, fn in ENC.items():
            a = rho_for(graphs, pairs, gd, fn)
            b = rho_for(sg, pairs, gd, fn)
            out[f"{ds}|{name}"] = {"shipped": a, "relabelled": b, "delta": b - a}
            print(f"{ds:<18}{name:<20}{a:>16.4f}{b:>16.4f}{b - a:>9.4f}")
        print()
    with open("real_relabel_control.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote real_relabel_control.json")


if __name__ == "__main__":
    main()
