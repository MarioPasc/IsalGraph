"""Picasso-side verification for T-04's wave-2 smoke.

Checks, in order of what would stop the ticket:
  1. pynauty imports from a from-source build and reproduces the running example
     and the K33/prism witness byte for byte against the workstation.
  2. grakel imports under this env's numpy and reproduces the corrected identity
     grakel(n_iter=k) == ours(h=k), with ours(h=2) == 5.830952.
  3. rapidfuzz returns the SYMBOL-level Levenshtein on a tuple.
  4. The Suite-1 and Suite-2 cohorts load, with their counts.
"""

from __future__ import annotations

import collections
import math
import platform
import sys

FAIL: list[str] = []


def check(label: str, got: object, want: object) -> None:
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {label:38s} got={got!r}")
    if not ok:
        FAIL.append(f"{label}: got {got!r}, want {want!r}")


def main() -> int:
    print(f"host      {platform.node()}")
    print(f"python    {sys.version.split()[0]}")
    import networkx as nx
    import numpy as np

    print(f"networkx  {nx.__version__}   numpy {np.__version__}")

    # ---- 1. pynauty ------------------------------------------------------
    import pynauty

    g = nx.Graph()
    g.add_nodes_from(range(6))
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (4, 5), (5, 3)])
    h = g.copy()
    h.remove_edge(0, 3)

    def canon(graph: nx.Graph) -> str:
        nodes = sorted(graph.nodes())
        idx = {v: i for i, v in enumerate(nodes)}
        pg = pynauty.Graph(
            len(nodes),
            directed=False,
            adjacency_dict={idx[v]: [idx[u] for u in graph.neighbors(v)] for v in nodes},
        )
        lab = pynauty.canon_label(pg)
        pos = {old: new for new, old in enumerate(lab)}
        out = nx.Graph()
        out.add_nodes_from(range(len(nodes)))
        out.add_edges_from((pos[idx[u]], pos[idx[v]]) for u, v in graph.edges())
        # Not in a test: on EVERY encode. Inverting canon_label gives a
        # different but still deterministic labelling that PASSES F3.
        assert nx.is_isomorphic(graph, out), "canon_label inversion is backwards"
        return nx.to_graph6_bytes(out, header=False).strip().decode()

    check("nauty_graph6 G", canon(g), "E@ro")
    check("nauty_graph6 H", canon(h), "E@po")
    pg = pynauty.Graph(6, directed=False, adjacency_dict={v: list(g.neighbors(v)) for v in g})
    check("|Aut(G)|", pynauty.autgrp(pg)[1], 4.0)

    k33 = nx.convert_node_labels_to_integers(nx.complete_bipartite_graph(3, 3))
    prism = nx.Graph()
    prism.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)])
    check("nauty K33", canon(k33), "Es\\o")
    check("nauty prism", canon(prism), "E{Sw")

    # ---- 2. grakel, and the CORRECTED identity ---------------------------
    def ours(graph: nx.Graph, rounds: int) -> collections.Counter[str]:
        colour = {v: "0" for v in graph}
        feats: collections.Counter[str] = collections.Counter(colour.values())
        for _ in range(rounds):
            colour = {
                v: colour[v] + "|" + ",".join(sorted(colour[w] for w in graph.neighbors(v)))
                for v in graph
            }
            feats.update(colour.values())
        return feats

    def rkhs(a: collections.Counter[str], b: collections.Counter[str]) -> float:
        keys = set(a) | set(b)
        return math.sqrt(
            sum(a[k] ** 2 for k in keys)
            + sum(b[k] ** 2 for k in keys)
            - 2 * sum(a[k] * b[k] for k in keys)
        )

    try:
        from grakel import Graph, VertexHistogram, WeisfeilerLehman

        def cv(graph: nx.Graph) -> Graph:
            edges = {(u, v) for u, v in graph.edges()} | {(v, u) for u, v in graph.edges()}
            return Graph(edges, node_labels={v: "0" for v in graph.nodes()})

        for k in (1, 2, 3):
            mat = WeisfeilerLehman(
                n_iter=k, base_graph_kernel=VertexHistogram, normalize=False
            ).fit_transform([cv(g), cv(h)])
            d_gk = math.sqrt(mat[0, 0] + mat[1, 1] - 2 * mat[0, 1])
            d_ours = rkhs(ours(g, k), ours(h, k))
            print(
                f"{'PASS' if abs(d_gk - d_ours) < 1e-9 else 'FAIL'}  "
                f"grakel(n_iter={k}) == ours(h={k})       {d_gk:.6f} vs {d_ours:.6f}"
            )
            if abs(d_gk - d_ours) >= 1e-9:
                FAIL.append(f"WL identity at k={k}: grakel {d_gk} != ours {d_ours}")
        check("ours(h=2)", round(rkhs(ours(g, 2), ours(h, 2)), 6), 5.830952)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL  grakel unusable: {type(exc).__name__}: {exc}")
        FAIL.append(f"grakel: {exc}")

    # ---- 3. rapidfuzz symbol level ---------------------------------------
    from rapidfuzz.distance import Levenshtein as lev

    check("rapidfuzz tuple (symbol level)", lev.distance(("0-1", "1-2", "2-0"), ("0-1", "2-0")), 1)
    check("rapidfuzz str (character level)", lev.distance("0-1 1-2 2-0", "0-1 2-0"), 4)

    # ---- 4. cohorts -------------------------------------------------------
    import os

    root = os.environ.get("ISALGRAPH_COHORT_ROOT", "")
    print(f"\ncohort root {root or '<unset>'}")
    expected = {
        "iam_letter_low": 1180,
        "iam_letter_med": 1253,
        "iam_letter_high": 2059,
        "linux": 89,
        "aids": 769,
        "grec": 650,
        "aids_iam": 1811,
        "coil_del": 3900,
        "mutagenicity": 4040,
        "protein": 569,
    }
    for name, want in expected.items():
        sub = (
            "exported"
            if name in ("iam_letter_low", "iam_letter_med", "iam_letter_high", "linux", "aids")
            else "exported_suite2"
        )
        path = os.path.join(root, sub, f"{name}.npz")
        if not os.path.exists(path):
            print(f"SKIP  {name:16s} absent at {path}")
            continue
        z = np.load(path, allow_pickle=True)
        check(f"cohort {name}", int(len(z["n_nodes"])), want)

    print()
    if FAIL:
        print(f"{len(FAIL)} FAILURES:")
        for f in FAIL:
            print("  -", f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
