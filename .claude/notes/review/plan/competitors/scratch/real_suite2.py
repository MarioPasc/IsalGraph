"""Suite 2 on the REAL cohort: Claim A and feasibility, per dataset, n up to 98.

No GED reference exists for Suite 2 (that is T-05's job), so there is no rho here.
What this settles is the part of the folder that does not need GED: real per-graph bit
counts under both conventions, the real AGM ceiling, and the real IsalGraph ceiling.

Graphs come from the repo's own T-01 loader, so the cohort is the locked one:
min_nodes = 2, require_connected = True, cxl enumeration (data.md §1.2, decision 27).
"""

from __future__ import annotations

import json
import statistics
import sys
import time

import networkx as nx
import numpy as np

sys.path.insert(0, "/home/mpascual/research/code/IsalGraph/benchmarks/real_data/eval_setup")

import agm_cam  # noqa: E402
import backends as B  # noqa: E402
import min_dfs  # noqa: E402
from iam_gxl_loader import load_iam_gxl  # noqa: E402

IAM_ROOT = (
    "/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/"
    "data/source/IAM_Database/extracted"
)
KEYS = ["grec", "aids_iam", "coil_del", "mutagenicity", "protein"]

AGM_BUDGET = 100_000
ISAL_BUDGET = 10.0
MAX_GRAPHS = 400  # per dataset; enough for a stable median, bounded wall-clock
SEED = 42


def to_nx(rec) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(rec.n_nodes if hasattr(rec, "n_nodes") else rec["n_nodes"]))
    edges = rec.edges if hasattr(rec, "edges") else rec["edges"]
    G.add_edges_from(edges)
    return G


def tuple_string(G):
    return "".join(chr(0xE000 + i * 100 + j) for i, j, *_ in min_dfs.min_dfs_code(G))


def bits(name, code, G):
    n, m = G.number_of_nodes(), G.number_of_edges()
    if name in ("graph6", "sparse6", "nauty->graph6"):
        return len(code) * 6.0
    if name in ("adjacency", "AGM CAM"):
        return n * (n - 1) / 2
    if name == "min-DFS (tuples)":
        return m * 2.0 * max(n - 1, 1).bit_length()
    if name == "GED construction":
        return (n - 1 + m) + 2 * m * max(n - 1, 1).bit_length()
    return len(code) * np.log2(9)


METHODS = [
    "adjacency",
    "graph6",
    "sparse6",
    "nauty->graph6",
    "AGM CAM",
    "min-DFS (tuples)",
    "IsalGraph pruned",
    "IsalGraph canonical",
    "GED construction",
]
ENC = {
    "graph6": B.graph6,
    "sparse6": B.sparse6,
    "nauty->graph6": B.nauty_canon_graph6,
    "adjacency": B.adjacency_bits,
    "min-DFS (tuples)": tuple_string,
}


def main() -> None:
    import random

    rng = random.Random(SEED)
    out = {}
    for key in KEYS:
        t0 = time.perf_counter()
        ds = load_iam_gxl(IAM_ROOT, key)
        graphs = [
            nx.convert_node_labels_to_integers(G, ordering="sorted")
            for G in ds.graphs
            if G.number_of_nodes() >= 2 and nx.is_connected(G)
        ]
        n_all = len(graphs)
        if len(graphs) > MAX_GRAPHS:
            graphs = [graphs[i] for i in sorted(rng.sample(range(len(graphs)), MAX_GRAPHS))]
        ns = [g.number_of_nodes() for g in graphs]
        ms = [g.number_of_edges() for g in graphs]
        print(
            f"\n=== {key}  retained={n_all}  sampled={len(graphs)}  "
            f"n_mean={statistics.mean(ns):.2f} n_max={max(ns)} "
            f"m_mean={statistics.mean(ms):.2f} ===",
            flush=True,
        )

        codes = {k: [] for k in METHODS}
        times = {k: 0.0 for k in METHODS}
        mdfsc_fail = agm_fail = 0
        for G in graphs:
            for name, fn in ENC.items():
                t = time.perf_counter()
                try:
                    codes[name].append(fn(G))
                except Exception:
                    codes[name].append(None)
                    if name == "min-DFS (tuples)":
                        mdfsc_fail += 1
                times[name] += time.perf_counter() - t
            t = time.perf_counter()
            try:
                codes["AGM CAM"].append(B.agm_code(G, budget=AGM_BUDGET)[0])
            except agm_cam.AGMBudgetExceeded:
                codes["AGM CAM"].append(None)
                agm_fail += 1
            times["AGM CAM"] += time.perf_counter() - t
        codes["GED construction"] = ["" for _ in graphs]

        isal = B.isalgraph_strings(graphs, budget=ISAL_BUDGET)
        codes["IsalGraph pruned"] = [s["pruned"] for s in isal]
        codes["IsalGraph canonical"] = [s["canonical"] for s in isal]
        times["IsalGraph pruned"] = sum(s["pruned_s"] for s in isal)
        times["IsalGraph canonical"] = sum(s["canonical_s"] for s in isal)
        isal_p_fail = sum(s["pruned"] is None for s in isal)
        isal_c_fail = sum(s["canonical"] is None for s in isal)

        bl = {
            k: [bits(k, c, G) if c is not None else None for c, G in zip(codes[k], graphs)]
            for k in METHODS
        }
        ref = bl["IsalGraph pruned"]
        print(
            f"{'method':<20}{'median bits':>12}{'mean bits':>11}"
            f"{'% Isal shorter':>16}{'ms/graph':>11}{'failed':>8}"
        )
        rows = {}
        for k in METHODS:
            vals = [v for v in bl[k] if v is not None]
            paired = [(a, b) for a, b in zip(bl[k], ref) if a is not None and b is not None]
            win = 100 * sum(b < a for a, b in paired) / len(paired) if paired else float("nan")
            rows[k] = {
                "median": statistics.median(vals) if vals else None,
                "mean": statistics.mean(vals) if vals else None,
                "isal_shorter_pct": win,
                "n_ok": len(vals),
                "ms_per_graph": 1e3 * times[k] / len(graphs),
            }
            med = f"{rows[k]['median']:.1f}" if vals else "FAIL"
            mean = f"{rows[k]['mean']:.1f}" if vals else "-"
            print(
                f"{k:<20}{med:>12}{mean:>11}{win:>16.1f}"
                f"{rows[k]['ms_per_graph']:>11.3f}{len(graphs) - len(vals):>8}"
            )
        print(
            f"  AGM budget failures {agm_fail}/{len(graphs)} | min-DFS failures "
            f"{mdfsc_fail}/{len(graphs)} | IsalGraph pruned {isal_p_fail} "
            f"canonical {isal_c_fail} | {time.perf_counter() - t0:.0f}s",
            flush=True,
        )

        out[key] = {
            "retained": n_all,
            "sampled": len(graphs),
            "n_mean": statistics.mean(ns),
            "n_max": max(ns),
            "m_mean": statistics.mean(ms),
            "claim_a": rows,
            "agm_fail": agm_fail,
            "mdfsc_fail": mdfsc_fail,
            "isal_pruned_fail": isal_p_fail,
            "isal_canonical_fail": isal_c_fail,
        }
        with open("real_suite2.json", "w") as fh:
            json.dump(out, fh, indent=2)
    print("\nwrote real_suite2.json")


if __name__ == "__main__":
    main()
