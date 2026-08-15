"""Wave-0 reproduction probe of competitors/README.md §4.1 (rho vs certified exact GED).

Replicates real_suite1.py's exact draw sequence (rng advanced: F3 sample, then rho sample,
per dataset in DATASETS order) and recomputes every row, plus:
  - adjacency under BOTH reading orders (scout row-major vs the design's column-wise)
  - the size null
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
import min_dfs  # noqa: E402

ROOT = "/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data"
GEDD = f"{ROOT}/source/GED_PRECOMPUTED/extended_merged_exact_ged/computed"
DATASETS = ["iam_letter_low", "iam_letter_med", "iam_letter_high", "linux", "aids"]
RHO_SAMPLE = 200
SEED = 42
AGM_BUDGET = 200_000

README = {
    "iam_letter_low": dict(
        null=0.899,
        adjacency=0.873,
        graph6=0.691,
        sparse6=0.748,
        nauty=0.677,
        agm=0.911,
        mdfs=0.972,
        isal=0.925,
    ),
    "iam_letter_med": dict(
        null=0.909,
        adjacency=0.850,
        graph6=0.681,
        sparse6=0.703,
        nauty=0.663,
        agm=0.920,
        mdfs=0.965,
        isal=0.916,
    ),
    "iam_letter_high": dict(
        null=0.926,
        adjacency=0.839,
        graph6=0.670,
        sparse6=0.654,
        nauty=0.639,
        agm=0.892,
        mdfs=0.842,
        isal=0.683,
    ),
    "linux": dict(
        null=0.713,
        adjacency=0.754,
        graph6=0.507,
        sparse6=0.559,
        nauty=0.538,
        agm=0.798,
        mdfs=0.653,
        isal=0.474,
    ),
    "aids": dict(
        null=0.799,
        adjacency=0.787,
        graph6=0.456,
        sparse6=0.515,
        nauty=0.460,
        agm=None,
        mdfs=0.551,
        isal=0.255,
    ),
}


def load(ds: str):
    z = np.load(f"{ROOT}/exported/{ds}.npz", allow_pickle=True)
    g = np.load(f"{GEDD}/{ds}.npz", allow_pickle=True)
    assert (z["graph_ids"] == g["graph_ids"]).all(), ds
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


def adj_rowmajor(G: nx.Graph) -> str:
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    A = [[0] * n for _ in range(n)]
    for u, v in G.edges():
        A[idx[u]][idx[v]] = A[idx[v]][idx[u]] = 1
    return "".join(str(A[i][j]) for i in range(n) for j in range(i + 1, n))


def adj_colwise(G: nx.Graph) -> str:
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    A = [[0] * n for _ in range(n)]
    for u, v in G.edges():
        A[idx[u]][idx[v]] = A[idx[v]][idx[u]] = 1
    return "".join(str(A[i][j]) for j in range(n) for i in range(j))


def g6(G: nx.Graph) -> str:
    H = nx.convert_node_labels_to_integers(G, ordering="sorted")
    return nx.to_graph6_bytes(H, header=False).decode().strip()


def s6(G: nx.Graph) -> str:
    H = nx.convert_node_labels_to_integers(G, ordering="sorted")
    return nx.to_sparse6_bytes(H, header=False).decode().strip()


def nauty_g6(G: nx.Graph) -> str:
    import pynauty

    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    pg = pynauty.Graph(
        len(nodes),
        directed=False,
        adjacency_dict={idx[v]: [idx[w] for w in G.neighbors(v)] for v in nodes},
    )
    lab = pynauty.canon_label(pg)
    pos = {old: new for new, old in enumerate(lab)}
    H = nx.Graph()
    H.add_nodes_from(range(len(nodes)))
    H.add_edges_from((pos[idx[u]], pos[idx[v]]) for u, v in G.edges())
    return nx.to_graph6_bytes(H, header=False).decode().strip()


def mdfs_tuplestring(G: nx.Graph) -> str:
    return "".join(chr(0xE000 + i * 100 + j) for i, j, *_ in min_dfs.min_dfs_code(G))


def isal_strings(graphs: list[nx.Graph], budget: float = 30.0) -> list[str | None]:
    import isalgraph
    from isalgraph import SparseGraph, pruned_canonical_string

    assert isalgraph.engine() == "cpp", isalgraph.engine()
    out: list[str | None] = []
    for G in graphs:
        H = nx.convert_node_labels_to_integers(G, ordering="sorted")
        g = SparseGraph(H.number_of_nodes(), False)
        for _ in range(H.number_of_nodes()):
            g.add_node()
        for u, v in H.edges():
            g.add_edge(u, v)
        try:
            out.append(pruned_canonical_string(g, timeout_s=budget))
        except Exception:  # noqa: BLE001
            out.append(None)
    return out


def main() -> None:
    rng = random.Random(SEED)
    print(f"{'dataset':<17}{'method':<14}{'README':>8}{'measured':>10}{'delta':>9}")
    for ds in DATASETS:
        graphs, ged, cert = load(ds)
        N = len(graphs)
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

        rng.sample(range(N), min(50, N))  # the F3 draw, consumed to keep the stream aligned
        sub = sorted(rng.sample(range(N), min(RHO_SAMPLE, N)))
        pairs = [(a, b) for ii, a in enumerate(sub) for b in sub[ii + 1 :] if cert[a, b]]
        gd = np.array([ged[a, b] for a, b in pairs])

        dn = np.array([abs(order[a] - order[b]) for a, b in pairs], dtype=float)
        rows = [("null", float(spearmanr(dn, gd).statistic))]
        for k in ("adjacency", "adj_colwise", "graph6", "sparse6", "nauty", "agm", "mdfs", "isal"):
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
        print(f"{'':17}{'(pairs=' + str(len(pairs)) + ')':<14}")


if __name__ == "__main__":
    main()
