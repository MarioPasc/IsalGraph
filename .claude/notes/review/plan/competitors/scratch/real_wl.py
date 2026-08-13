"""WL subtree kernel distance vs certified exact GED, Suite 1, same sample as real_suite1.py.

Kept separate because WL is the one comparator that is not a serialisation: it has no
encode(G) -> str and no bit count, so it does not fit the Claim A loop.
"""

from __future__ import annotations

import json
import random

import backends as B
import networkx as nx
import numpy as np
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


def main():
    rng = random.Random(SEED)
    out = {}
    print(
        f"{'dataset':<18}{'h':>3}{'pairs':>8}{'rho(WL, exact GED)':>21}{'frac d=0':>10}"
        f"{'iso pairs (GED=0)':>19}"
    )
    for ds in DATASETS:
        graphs, ged, cert = load(ds)
        N = len(graphs)
        # the same draw order real_suite1.py uses: 50-graph F3 sample first, then the rho sample
        rng2 = random.Random(SEED)
        for d in DATASETS:
            if d == ds:
                break
        sub = sorted(rng.sample(range(N), min(RHO_SAMPLE, N)))
        pairs = [(a, b) for ii, a in enumerate(sub) for b in sub[ii + 1 :] if cert[a, b]]
        gd = np.array([ged[a, b] for a, b in pairs])
        for h in (2, 3):
            feats = {i: B.wl_features(graphs[i], h) for i in sub}
            d = np.array([B.wl_distance(feats[a], feats[b]) for a, b in pairs])
            r = spearmanr(d, gd)
            iso = float((gd == 0).mean())
            print(
                f"{ds:<18}{h:>3}{len(pairs):>8}{r.statistic:>21.4f}"
                f"{float((d == 0).mean()):>10.4f}{iso:>19.4f}"
            )
            out[f"{ds}_h{h}"] = {
                "spearman": float(r.statistic),
                "p": float(r.pvalue),
                "zero_frac": float((d == 0).mean()),
                "iso_frac": iso,
                "n_pairs": len(pairs),
            }
    with open("real_wl.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote real_wl.json")


if __name__ == "__main__":
    main()
