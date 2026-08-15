"""Wave-0 spot check: the running-example strings every agent's criterion 1 asserts."""

from __future__ import annotations

import sys

import networkx as nx

sys.path.insert(
    0, "/home/mpascual/research/code/IsalGraph/.claude/notes/review/plan/competitors/scratch"
)


def running_example() -> tuple[nx.Graph, nx.Graph]:
    G = nx.Graph()
    G.add_nodes_from(range(6))
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (4, 5), (5, 3)])
    H = G.copy()
    H.remove_edge(0, 3)
    return G, H


def triangle_upper_columnwise(G: nx.Graph) -> str:
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()), dtype=int)
    return "".join(str(int(A[i, j])) for j in range(n) for i in range(j))


def k33_prism() -> tuple[nx.Graph, nx.Graph]:
    k33 = nx.complete_bipartite_graph(3, 3)
    prism = nx.Graph()
    prism.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)])
    return nx.convert_node_labels_to_integers(k33), prism


def main() -> None:
    G, H = running_example()
    for name, g in (("G", G), ("H", H)):
        g6 = nx.to_graph6_bytes(g, header=False).strip().decode()
        s6 = nx.to_sparse6_bytes(g, header=False).strip().decode()
        print(f"{name}: adjacency={triangle_upper_columnwise(g)!r} graph6={g6!r} sparse6={s6!r}")

    print("\n--- pynauty ---")
    import pynauty

    print("pynauty", pynauty.__file__)

    def canonical_relabel(g: nx.Graph) -> nx.Graph:
        nodes = sorted(g.nodes())
        idx = {v: i for i, v in enumerate(nodes)}
        pg = pynauty.Graph(
            len(nodes),
            directed=False,
            adjacency_dict={idx[v]: [idx[u] for u in g.neighbors(v)] for v in nodes},
        )
        lab = pynauty.canon_label(pg)
        pos = {old: new for new, old in enumerate(lab)}
        out = nx.Graph()
        for new in range(len(nodes)):
            out.add_node(new)
        for u, v in g.edges():
            out.add_edge(pos[idx[u]], pos[idx[v]])
        assert nx.is_isomorphic(g, out), "canon_label inversion is backwards"
        return out

    for name, g in (("G", G), ("H", H)):
        cg = canonical_relabel(g)
        g6 = nx.to_graph6_bytes(cg, header=False).strip().decode()
        pg = pynauty.Graph(
            g.number_of_nodes(),
            directed=False,
            adjacency_dict={v: list(g.neighbors(v)) for v in g.nodes()},
        )
        gens, grpsize1, grpsize2, orbits, numorbits = pynauty.autgrp(pg)
        print(f"{name}: nauty_graph6={g6!r} |Aut|={grpsize1}e{grpsize2}")

    print("\n--- K33 / prism ---")
    k33, prism = k33_prism()
    for name, g in (("K33", k33), ("prism", prism)):
        cg = canonical_relabel(g)
        print(f"{name}: nauty_graph6={nx.to_graph6_bytes(cg, header=False).strip().decode()!r}")

    print("\n--- AGM (scout port) ---")
    try:
        import agm_cam

        for name, g in (("G", G), ("H", H), ("K33", k33), ("prism", prism)):
            fn = getattr(agm_cam, "agm_canonical_code", None) or getattr(
                agm_cam, "canonical_code", None
            )
            print(f"{name}: agm={fn(g)!r}" if fn else f"no entry point; dir={dir(agm_cam)}")
    except Exception as exc:  # noqa: BLE001
        print("agm probe failed:", type(exc).__name__, exc)

    print("\n--- min-DFS (scout port) ---")
    try:
        import min_dfs

        for name, g in (("G", G), ("H", H), ("K33", k33), ("prism", prism)):
            code = min_dfs.min_dfs_code(g)
            print(f"{name}: {' '.join(f'{i}-{j}' for i, j, *_ in code)}  ({len(code)} tuples)")
    except Exception as exc:  # noqa: BLE001
        print("min_dfs probe failed:", type(exc).__name__, exc)


if __name__ == "__main__":
    main()
