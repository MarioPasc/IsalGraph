"""The running example: every representation of the same graph, side by side.

G  = 4-cycle (0-1-2-3) sharing node 3 with a triangle (3-4-5).  6 nodes, 7 edges.
G' = G under a random relabelling                     -- tests isomorphism invariance
H  = G minus edge (0,3)                               -- exact GED 1 under the D6 model
"""

from __future__ import annotations

import json
import random

import backends as B
import networkx as nx

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (4, 5), (5, 3)]


def build():
    G = nx.Graph(EDGES)
    rng = random.Random(2026)
    # a genuine relabelling: new names AND a new insertion order.  `relabel_nodes`
    # alone preserves insertion order and makes order-dependent codes look invariant.
    new = list(range(G.number_of_nodes()))
    rng.shuffle(new)
    mapping = dict(zip(sorted(G.nodes()), new))
    order = list(new)
    rng.shuffle(order)
    Gp = nx.Graph()
    Gp.add_nodes_from(order)
    e2 = [(mapping[u], mapping[v]) for u, v in G.edges()]
    rng.shuffle(e2)
    Gp.add_edges_from(e2)
    assert nx.is_isomorphic(G, Gp) and list(Gp.nodes()) != sorted(Gp.nodes())
    H = nx.Graph([e for e in EDGES if e != (3, 0)])
    return G, Gp, H


def encode_all(G):
    out = {}
    out["graph6"] = B.graph6(G)
    out["sparse6"] = B.sparse6(G)
    out["nauty_graph6"] = B.nauty_canon_graph6(G)
    out["nauty_certificate"] = B.nauty_certificate(G).hex()
    out["adjacency"] = B.adjacency_bits(G)
    code, nodes = B.agm_code(G)
    out["agm_cam"] = code
    out["agm_search_nodes"] = nodes
    out["mdfsc"] = B.mdfsc(G)
    out["wl_dim"] = len(B.wl_features(G))
    return out


def main():
    G, Gp, H = build()
    isal = B.isalgraph_strings([G, Gp, H])
    rows = {}
    for name, K, s in (("G", G, isal[0]), ("G_relabelled", Gp, isal[1]), ("H", H, isal[2])):
        r = encode_all(K)
        r["isalgraph_canonical"] = s["canonical"]
        r["isalgraph_pruned"] = s["pruned"]
        r["n"] = K.number_of_nodes()
        r["m"] = K.number_of_edges()
        rows[name] = r

    print("=" * 78)
    print("RUNNING EXAMPLE  G = C4(0,1,2,3) + K3(3,4,5), n=6 m=7")
    print("=" * 78)
    keys = [
        "graph6",
        "sparse6",
        "nauty_graph6",
        "adjacency",
        "agm_cam",
        "mdfsc",
        "isalgraph_canonical",
        "isalgraph_pruned",
    ]
    for k in keys:
        print(f"\n{k}")
        for name in ("G", "G_relabelled", "H"):
            v = rows[name][k]
            mark = ""
            if name == "G_relabelled":
                mark = "  <-- INVARIANT" if v == rows["G"][k] else "  <-- ***CHANGED***"
            print(f"  {name:<14} {v!r}{mark}")

    print(
        f"\nnauty certificate G == G' : "
        f"{rows['G']['nauty_certificate'] == rows['G_relabelled']['nauty_certificate']}"
    )
    print(f"AGM search nodes expanded (n=6): {rows['G']['agm_search_nodes']}")

    # ---- bit accounting -----------------------------------------------------
    print("\n" + "=" * 78)
    print("BIT ACCOUNTING for G (n=6, m=7)")
    print("=" * 78)
    n, m = 6, 7
    tab = [
        ("graph6", len(rows["G"]["graph6"]), 64, B.realised_bits(rows["G"]["graph6"])),
        ("sparse6", len(rows["G"]["sparse6"]), 64, B.realised_bits(rows["G"]["sparse6"])),
        (
            "nauty->graph6",
            len(rows["G"]["nauty_graph6"]),
            64,
            B.realised_bits(rows["G"]["nauty_graph6"]),
        ),
        ("adjacency", n * (n - 1) // 2, 2, ((n * (n - 1) // 2) + 7) // 8 * 8),
        ("AGM CAM", len(rows["G"]["agm_cam"]), 2, (len(rows["G"]["agm_cam"]) + 7) // 8 * 8),
        ("min-DFS (tuple)", m, n * n, m * 2 * 8),
        (
            "IsalGraph canon",
            len(rows["G"]["isalgraph_canonical"]),
            9,
            B.realised_bits(rows["G"]["isalgraph_canonical"]),
        ),
        (
            "IsalGraph pruned",
            len(rows["G"]["isalgraph_pruned"]),
            9,
            B.realised_bits(rows["G"]["isalgraph_pruned"]),
        ),
    ]
    print(f"{'method':<18} {'L':>4} {'|S|':>5} {'entropy bits':>13} {'realised bits':>14}")
    for nm, L, sig, real in tab:
        print(f"{nm:<18} {L:>4} {sig:>5} {B.entropy_bits(L, sig):>13.2f} {real:>14}")
    print(
        f"{'GED construction':<18} {'-':>4} {'-':>5} "
        f"{(n - 1 + m) + 2 * m * (n - 1).bit_length():>13.2f} {'-':>14}   "
        f"# B_GED = (N-1+M) + 2M ceil(log2 N)"
    )

    # ---- distances G vs H (true GED = 1 edge deletion) ----------------------
    print("\n" + "=" * 78)
    print("DISTANCE G vs H  (H = G minus one edge; unit-cost GED = 1)")
    print("=" * 78)
    print(f"{'representation':<18} {'Hamming':>9} {'Levenshtein':>12}")
    for k in keys:
        a, b = rows["G"][k], rows["H"][k]
        h = B.hamming(a, b)
        print(f"{k:<18} {str(h) if h is not None else 'UNDEF':>9} {B.levenshtein(a, b):>12}")
    fa, fb = B.wl_features(G), B.wl_features(H)
    print(f"{'WL(h=3) kernel d':<18} {'-':>9} {B.wl_distance(fa, fb):>12.4f}")
    print(f"{'padded Hamming':<18} {B.padded_hamming(G, H):>9} {'-':>12}")

    with open("running_example.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    print("\nwrote running_example.json")


if __name__ == "__main__":
    main()
