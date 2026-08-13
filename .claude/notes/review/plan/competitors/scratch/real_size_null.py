"""How much of every representation's rho is just graph size?

The relabelling control (real_relabel_control.py) showed the adjacency matrix keeps
rho ~ 0.75-0.87 even after every graph is randomly relabelled, so its signal is not the
corpus's incidental vertex order.  The likely explanation is duller: Levenshtein on a
bit string of length n(n-1)/2 is dominated by the length difference, which is a monotone
function of |n1 - n2|; and under the D6 unit cost model GED >= |n1 - n2| and is largely
driven by node-count difference.

Two null models and one conditional test:

  N1  rho(|n1 - n2|, GED)                      -- node count alone
  N2  rho(|n1 - n2| + |m1 - m2|, GED)          -- node and edge count alone
  EQ  rho(representation distance, GED) restricted to EQUAL-n pairs

EQ is the sharp question: does the representation track GED among graphs of the same
size, where the trivial predictor is constant?
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

ROOT = "/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data"
GEDD = f"{ROOT}/source/GED_PRECOMPUTED/extended_merged_exact_ged/computed"
DATASETS = ["iam_letter_low", "iam_letter_med", "iam_letter_high", "linux", "aids"]
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


def main():
    rng = random.Random(SEED)
    out = {}
    for ds in DATASETS:
        graphs, ged, cert = load(ds)
        N = len(graphs)
        sub = sorted(rng.sample(range(N), min(RHO_SAMPLE, N)))
        pairs = [(a, b) for ii, a in enumerate(sub) for b in sub[ii + 1 :] if cert[a, b]]
        gd = np.array([ged[a, b] for a, b in pairs])
        ns = {i: graphs[i].number_of_nodes() for i in sub}
        ms = {i: graphs[i].number_of_edges() for i in sub}
        dn = np.array([abs(ns[a] - ns[b]) for a, b in pairs], dtype=float)
        dnm = np.array([abs(ns[a] - ns[b]) + abs(ms[a] - ms[b]) for a, b in pairs], dtype=float)
        eq = dn == 0

        isal = B.isalgraph_strings([graphs[i] for i in sub], budget=30.0)
        istr = {i: s["pruned"] for i, s in zip(sub, isal)}

        n1 = float(spearmanr(dn, gd).statistic)
        n2 = float(spearmanr(dnm, gd).statistic)
        print(
            f"\n=== {ds}  {len(pairs)} certified pairs, {int(eq.sum())} equal-n "
            f"({100 * eq.mean():.1f}%) ==="
        )
        print(f"  NULL N1  rho(|n1-n2|, GED)            = {n1:.4f}")
        print(f"  NULL N2  rho(|dn|+|dm|, GED)          = {n2:.4f}")
        print(f"  {'representation':<20}{'rho all pairs':>15}{'rho equal-n':>13}{'vs N1':>9}")
        row = {
            "null_n1": n1,
            "null_n2": n2,
            "n_pairs": len(pairs),
            "equal_n_pairs": int(eq.sum()),
            "reps": {},
        }
        for name, fn in list(ENC.items()) + [("IsalGraph pruned", None)]:
            cs = {i: fn(graphs[i]) for i in sub} if fn else istr
            d = np.array([RFLev.distance(cs[a], cs[b]) for a, b in pairs], dtype=float)
            r_all = float(spearmanr(d, gd).statistic)
            r_eq = float(spearmanr(d[eq], gd[eq]).statistic) if eq.sum() > 30 else float("nan")
            row["reps"][name] = {"rho_all": r_all, "rho_equal_n": r_eq}
            print(f"  {name:<20}{r_all:>15.4f}{r_eq:>13.4f}{r_all - n1:>9.4f}")
        out[ds] = row
        with open("real_size_null.json", "w") as fh:
            json.dump(out, fh, indent=2)
    print("\nwrote real_size_null.json")


if __name__ == "__main__":
    main()
