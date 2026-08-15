"""Quantify what the WL convention defect costs on the real cohort.

Reproduces real_wl.py's exact draw sequence, then computes rho(WL, exact GED) three ways:
  (a) scout   -- backends.wl_features, per-round per-graph colour compression
  (b) shared  -- correct WL, one shared vocabulary across the whole dataset sample
  (c) grakel  -- grakel WeisfeilerLehman(n_iter=h, VertexHistogram, normalize=False)
"""

from __future__ import annotations

import collections
import json
import math
import random

import networkx as nx
import numpy as np
from scipy.stats import spearmanr

ROOT = "/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data"
GEDD = f"{ROOT}/source/GED_PRECOMPUTED/extended_merged_exact_ged/computed"
DATASETS = ["iam_letter_low", "iam_letter_med", "iam_letter_high", "linux", "aids"]
RHO_SAMPLE = 200
SEED = 42


def load(ds: str):
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


def wl_scout(G: nx.Graph, h: int) -> dict[str, int]:
    """Verbatim copy of scratch/backends.py::wl_features."""
    colour = {v: "0" for v in G}
    feats: collections.Counter[str] = collections.Counter()
    feats.update(f"0:{c}" for c in colour.values())
    for it in range(1, h + 1):
        new = {}
        for v in G:
            sig = colour[v] + "|" + ",".join(sorted(colour[w] for w in G.neighbors(v)))
            new[v] = sig
        table = {s: str(i) for i, s in enumerate(sorted(set(new.values())))}
        colour = {v: table[s] for v, s in new.items()}
        feats.update(f"{it}:{new[v]}" for v in G)
    return dict(feats)


def wl_shared(G: nx.Graph, h: int) -> dict[str, int]:
    """Standard WL: labels stay uncompressed, so the vocabulary is global by construction."""
    colour = {v: "0" for v in G}
    feats: collections.Counter[str] = collections.Counter()
    feats.update(f"0:{c}" for c in colour.values())
    for it in range(1, h + 1):
        new = {v: colour[v] + "|" + ",".join(sorted(colour[w] for w in G.neighbors(v))) for v in G}
        colour = new
        feats.update(f"{it}:{colour[v]}" for v in G)
    return dict(feats)


def kdist(fa: dict[str, int], fb: dict[str, int]) -> float:
    keys = set(fa) | set(fb)
    kaa = sum(fa.get(k, 0) ** 2 for k in keys)
    kbb = sum(fb.get(k, 0) ** 2 for k in keys)
    kab = sum(fa.get(k, 0) * fb.get(k, 0) for k in keys)
    return math.sqrt(max(kaa + kbb - 2 * kab, 0.0))


def grakel_matrix(graphs, h: int):
    from grakel import Graph, VertexHistogram, WeisfeilerLehman

    def cv(g):
        e = {(u, v) for u, v in g.edges()} | {(v, u) for u, v in g.edges()}
        if not e:
            e = set()
        return Graph(e, node_labels={v: "0" for v in g.nodes()})

    return WeisfeilerLehman(
        n_iter=h, base_graph_kernel=VertexHistogram, normalize=False
    ).fit_transform([cv(g) for g in graphs])


def main() -> None:
    rng = random.Random(SEED)
    rows = {}
    hdr = f"{'dataset':<18}{'h':>3}{'pairs':>7}{'scout':>9}{'shared':>9}{'grakel':>9}{'README':>9}"
    print(hdr)
    readme = {
        ("iam_letter_low", 2): 0.895,
        ("iam_letter_med", 2): 0.869,
        ("iam_letter_high", 2): 0.580,
        ("linux", 2): 0.573,
        ("aids", 2): 0.459,
    }
    for ds in DATASETS:
        graphs, ged, cert = load(ds)
        N = len(graphs)
        sub = sorted(rng.sample(range(N), min(RHO_SAMPLE, N)))
        pairs = [(a, b) for ii, a in enumerate(sub) for b in sub[ii + 1 :] if cert[a, b]]
        gd = np.array([ged[a, b] for a, b in pairs])
        for h in (2, 3):
            fs = {i: wl_scout(graphs[i], h) for i in sub}
            ds_scout = np.array([kdist(fs[a], fs[b]) for a, b in pairs])
            fh = {i: wl_shared(graphs[i], h) for i in sub}
            ds_shared = np.array([kdist(fh[a], fh[b]) for a, b in pairs])
            K = grakel_matrix([graphs[i] for i in sub], h)
            idx = {g: k for k, g in enumerate(sub)}
            ds_gk = np.array(
                [
                    math.sqrt(
                        max(K[idx[a], idx[a]] + K[idx[b], idx[b]] - 2 * K[idx[a], idx[b]], 0.0)
                    )
                    for a, b in pairs
                ]
            )
            r_s = spearmanr(ds_scout, gd).statistic
            r_h = spearmanr(ds_shared, gd).statistic
            r_g = spearmanr(ds_gk, gd).statistic
            ref = readme.get((ds, h))
            print(
                f"{ds:<18}{h:>3}{len(pairs):>7}{r_s:>9.4f}{r_h:>9.4f}{r_g:>9.4f}"
                f"{(f'{ref:.3f}' if ref else '-'):>9}"
            )
            rows[f"{ds}_h{h}"] = {
                "scout": float(r_s),
                "shared": float(r_h),
                "grakel": float(r_g),
                "readme": ref,
                "n_pairs": len(pairs),
            }
    with open("wl_rho_probe.json", "w") as fp:
        json.dump(rows, fp, indent=2)
    print("\nwrote wl_rho_probe.json")


if __name__ == "__main__":
    main()
