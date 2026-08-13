"""F3 (isomorphism invariance) done honestly, plus a cohort-matched scaling sweep.

The naive relabelling test is wrong: `nx.relabel_nodes(copy=True)` preserves *insertion
order*, so a representation that reads `list(G.nodes())` sees an unchanged order and
looks invariant when it is not.  Here every relabelled copy is rebuilt by inserting
nodes and edges in a freshly shuffled order.

Sizes and edge counts are the measured Suite-2 per-dataset means from
`.claude/notes/review/plan/data.md` §1.2 (T-01, 2026-08-13).  The graphs themselves are
synthetic: the IAM corpus is not on this workstation.  Downstream must re-run on the
real cohort; the shapes measured here are what the ordering conclusions rest on.
"""

from __future__ import annotations

import json
import random
import statistics
import time

import agm_cam
import backends as B
import networkx as nx

# (label, n, m) -- Suite-2 measured means, T-01
COHORT = [
    ("Letter LOW", 4, 3),
    ("Letter MED", 4, 3),
    ("Letter HIGH", 5, 5),
    ("LINUX", 9, 8),
    ("AIDS (GraphEdX)", 11, 11),
    ("GREC", 11, 12),
    ("AIDS (IAM)", 14, 15),
    ("COIL-DEL", 22, 54),
    ("Mutagenicity", 29, 30),
    ("Protein", 32, 61),
]

# ceiling sweep: n_max = 98 in Suite 2
CEILING = [
    (f"n={n}, m={int(r * n)}", n, int(r * n)) for n in (20, 30, 50, 70, 98) for r in (1.05, 2.0)
]


def random_connected(n: int, m: int, rng: random.Random) -> nx.Graph:
    """Connected graph on n nodes with exactly m edges.

    Rejection sampling on `gnm_random_graph` does NOT work here: connectivity of G(n, m)
    needs m ~ (n/2) ln n, so at the cohort's m ~ n it almost never terminates above
    n ~ 30.  Instead: draw a uniform random spanning tree (random Prufer sequence), then
    add the remaining m - (n-1) edges uniformly from the non-edges.  This is NOT uniform
    over connected graphs on m edges -- it over-weights tree-like structure -- and that
    caveat travels with every number derived from it.
    """
    m = max(m, n - 1)
    m = min(m, n * (n - 1) // 2)
    if n == 2:
        return nx.path_graph(2)
    G = (
        nx.random_labeled_tree(n, seed=rng.randrange(10**9))
        if hasattr(nx, "random_labeled_tree")
        else nx.random_tree(n, seed=rng.randrange(10**9))
    )
    extra = m - (n - 1)
    if extra > 0:
        non = [(u, v) for u in range(n) for v in range(u + 1, n) if not G.has_edge(u, v)]
        rng.shuffle(non)
        G.add_edges_from(non[:extra])
    assert nx.is_connected(G) and G.number_of_edges() == m
    return G


def shuffled_copy(G: nx.Graph, rng: random.Random) -> nx.Graph:
    """A relabelled copy with a genuinely different insertion order."""
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


# --------------------------------------------------------------------------- #
# F3
# --------------------------------------------------------------------------- #


def f3(n_graphs: int = 40, n_relab: int = 25, seed: int = 42) -> dict[str, dict]:
    rng = random.Random(seed)
    reps = {
        "graph6": B.graph6,
        "sparse6": B.sparse6,
        "nauty->graph6": B.nauty_canon_graph6,
        "adjacency": B.adjacency_bits,
        "AGM CAM": lambda G: B.agm_code(G)[0],
        "min-DFS code": B.mdfsc,
        "WL(h=3) vector": lambda G: json.dumps(sorted(B.wl_features(G).items())),
    }
    stats = {k: {"graphs": 0, "invariant": 0, "distinct_max": 0} for k in reps}
    isal_batch: list[nx.Graph] = []
    isal_group: list[int] = []
    for gi in range(n_graphs):
        n = rng.choice([5, 6, 7, 8, 9, 10, 12])
        G = random_connected(n, rng.randint(n - 1, min(2 * n, n * (n - 1) // 2)), rng)
        copies = [G] + [shuffled_copy(G, rng) for _ in range(n_relab)]
        isal_batch.extend(copies)
        isal_group.extend([gi] * len(copies))
        for name, fn in reps.items():
            codes = {fn(K) for K in copies}
            stats[name]["graphs"] += 1
            stats[name]["invariant"] += int(len(codes) == 1)
            stats[name]["distinct_max"] = max(stats[name]["distinct_max"], len(codes))
    # IsalGraph, one subprocess for the lot
    isal = B.isalgraph_strings(isal_batch)
    for key in ("canonical", "pruned"):
        name = f"IsalGraph {key}"
        stats[name] = {"graphs": 0, "invariant": 0, "distinct_max": 0}
        for gi in range(n_graphs):
            codes = {s[key] for s, g in zip(isal, isal_group) if g == gi}
            stats[name]["graphs"] += 1
            stats[name]["invariant"] += int(len(codes) == 1)
            stats[name]["distinct_max"] = max(stats[name]["distinct_max"], len(codes))
    return stats


# --------------------------------------------------------------------------- #
# scaling / bit accounting
# --------------------------------------------------------------------------- #


def bits_row(G: nx.Graph, isal: dict[str, str]) -> dict:
    n, m = G.number_of_nodes(), G.number_of_edges()
    row: dict[str, float | None] = {}
    t = {}

    t0 = time.perf_counter()
    g6 = B.graph6(G)
    t["graph6"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    s6 = B.sparse6(G)
    t["sparse6"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    ng6 = B.nauty_canon_graph6(G)
    t["nauty->graph6"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    md = B.mdfsc(G)
    t["min-DFS code"] = time.perf_counter() - t0

    row["graph6"] = len(g6) * 6.0
    row["sparse6"] = len(s6) * 6.0
    row["nauty->graph6"] = len(ng6) * 6.0
    row["adjacency"] = n * (n - 1) / 2
    row["min-DFS code"] = m * 2 * (max(n - 1, 1)).bit_length()
    row["IsalGraph canonical"] = len(isal["canonical"]) * (9**0.0) * __import__("math").log2(9)
    row["IsalGraph pruned"] = len(isal["pruned"]) * __import__("math").log2(9)
    row["GED construction"] = (n - 1 + m) + 2 * m * (max(n - 1, 1)).bit_length()

    t0 = time.perf_counter()
    try:
        code, nodes = B.agm_code(G, budget=400_000)
        row["AGM CAM"] = float(len(code))
        agm_nodes = nodes
        agm_status = "exact"
    except agm_cam.AGMBudgetExceeded as e:
        row["AGM CAM"] = None
        agm_nodes = e.nodes
        agm_status = "BUDGET EXCEEDED"
    t["AGM CAM"] = time.perf_counter() - t0
    return {
        "bits": row,
        "time": t,
        "agm_nodes": agm_nodes,
        "agm_status": agm_status,
        "n": n,
        "m": m,
        "isal_len": len(isal["pruned"]),
    }


def scaling(profiles, reps: int = 5, seed: int = 42):
    rng = random.Random(seed)
    out = []
    for label, n, m in profiles:
        graphs = [random_connected(n, m, rng) for _ in range(reps)]
        isal = B.isalgraph_strings(graphs)
        rows = [bits_row(G, s) for G, s in zip(graphs, isal)]
        agg = {"label": label, "n": n, "m": m}
        for k in rows[0]["bits"]:
            vals = [r["bits"][k] for r in rows if r["bits"][k] is not None]
            agg[k] = statistics.median(vals) if vals else None
        agg["_time"] = {k: statistics.median(r["time"][k] for r in rows) for k in rows[0]["time"]}
        agg["_agm_nodes"] = statistics.median(r["agm_nodes"] for r in rows)
        agg["_agm_status"] = (
            "exact" if all(r["agm_status"] == "exact" for r in rows) else "BUDGET EXCEEDED"
        )
        agg["_isal_len"] = statistics.median(r["isal_len"] for r in rows)
        out.append(agg)
        print(f"  done {label}", flush=True)
    return out


METHODS = [
    "graph6",
    "sparse6",
    "nauty->graph6",
    "adjacency",
    "AGM CAM",
    "min-DFS code",
    "IsalGraph canonical",
    "IsalGraph pruned",
    "GED construction",
]


def print_table(rows, title):
    print("\n" + "=" * 118)
    print(title)
    print("=" * 118)
    hdr = f"{'dataset profile':<18}{'n':>4}{'m':>5}" + "".join(f"{k[:13]:>15}" for k in METHODS)
    print(hdr)
    for r in rows:
        line = f"{r['label']:<18}{r['n']:>4}{r['m']:>5}"
        for k in METHODS:
            v = r[k]
            line += f"{'n/a':>15}" if v is None else f"{v:>15.0f}"
        print(line)
    print("\nmedian encode time (ms) and AGM search nodes")
    print(
        f"{'dataset profile':<18}{'graph6':>10}{'sparse6':>10}{'nauty':>10}"
        f"{'min-DFS':>10}{'AGM':>12}{'AGM nodes':>12}  status"
    )
    for r in rows:
        t = r["_time"]
        print(
            f"{r['label']:<18}{t['graph6'] * 1e3:>10.3f}{t['sparse6'] * 1e3:>10.3f}"
            f"{t['nauty->graph6'] * 1e3:>10.3f}{t['min-DFS code'] * 1e3:>10.3f}"
            f"{t['AGM CAM'] * 1e3:>12.2f}{r['_agm_nodes']:>12.0f}  {r['_agm_status']}"
        )


def main():
    print("### F3 -- isomorphism invariance, 40 graphs x 25 genuine relabellings")
    stats = f3()
    print(f"{'representation':<22}{'invariant graphs':>18}{'max distinct codes':>20}")
    for k, v in stats.items():
        print(f"{k:<22}{v['invariant']}/{v['graphs']:<16}{v['distinct_max']:>20}")

    print("\n### scaling -- cohort profiles")
    rows = scaling(COHORT)
    print_table(rows, "ENTROPY-BOUND BITS, median of 5 (Suite-2 measured means, T-01)")

    print("\n### scaling -- ceiling sweep to n = 98")
    rows2 = scaling(CEILING, reps=3)
    print_table(rows2, "ENTROPY-BOUND BITS, median of 3")

    with open("sweep.json", "w") as fh:
        json.dump({"f3": stats, "cohort": rows, "ceiling": rows2}, fh, indent=2)
    print("\nwrote sweep.json")


if __name__ == "__main__":
    main()
