"""Wave-0 probe: re-verify the grakel n_iter off-by-one and the 5.830952 identity."""

from __future__ import annotations

import math
from collections import Counter

import networkx as nx


def running_example() -> tuple[nx.Graph, nx.Graph]:
    G = nx.Graph()
    G.add_nodes_from(range(6))
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (4, 5), (5, 3)])
    H = G.copy()
    H.remove_edge(0, 3)
    return G, H


def wl_features(G: nx.Graph, h: int) -> Counter[str]:
    """Our own WL: `h` counts REFINEMENT rounds; round 0 is the base histogram."""
    labels = {v: "0" for v in G.nodes()}
    feats: Counter[str] = Counter(labels.values())
    for _ in range(h):
        new = {}
        for v in G.nodes():
            sig = labels[v] + "|" + ",".join(sorted(labels[u] for u in G.neighbors(v)))
            new[v] = sig
        labels = new
        feats.update(labels.values())
    return feats


def dot(a: Counter[str], b: Counter[str]) -> float:
    return float(sum(a[k] * b[k] for k in set(a) | set(b)))


def rkhs(a: Counter[str], b: Counter[str]) -> float:
    return math.sqrt(dot(a, a) + dot(b, b) - 2 * dot(a, b))


def main() -> None:
    G, H = running_example()
    print("--- ours ---")
    for h in (1, 2, 3, 5):
        fg, fh = wl_features(G, h), wl_features(H, h)
        print(f"ours h={h}: d={rkhs(fg, fh):.6f}  nnz(G)={len(fg)} nnz(H)={len(fh)}")

    import grakel

    print("grakel __version__ =", grakel.__version__)
    from grakel import Graph, VertexHistogram, WeisfeilerLehman

    def to_grakel(g: nx.Graph) -> Graph:
        edges = {(u, v) for u, v in g.edges()} | {(v, u) for u, v in g.edges()}
        return Graph(edges, node_labels={v: "0" for v in g.nodes()})

    print("--- grakel ---")
    for k in (1, 2, 3, 4, 6):
        K = WeisfeilerLehman(
            n_iter=k, base_graph_kernel=VertexHistogram, normalize=False
        ).fit_transform([to_grakel(G), to_grakel(H)])
        d = math.sqrt(K[0, 0] + K[1, 1] - 2 * K[0, 1])
        print(f"grakel n_iter={k}: d={d:.6f}  K={K.tolist()}")

    print()
    print("sqrt(34) =", math.sqrt(34))


if __name__ == "__main__":
    main()
