"""Suite 2, the three remaining datasets: COIL-DEL, Mutagenicity, Protein.

Trimmed relative to `real_suite2.py`, and the trims are deliberate rather than budgetary
shortcuts -- both remove work whose answer is already measured:

  * **AGM on a 50-graph subsample**, not 400.  GREC (76 % exact) and AIDS-IAM (82 %)
    already establish that AGM does not survive Suite 2; what is still worth having is a
    failure *rate* per dataset, and 50 graphs at 100k nodes gives that at 1/8 the cost.
  * **`canonical_string` at a 2 s budget, failure rate only, no bit-count row.**  It
    already fails 12/400 on AIDS-IAM at 10 s and T-06 will use `pruned` regardless.  Its
    bit counts here would be conditioned on the graphs fast enough to finish, which is a
    biased sample and must not be printed as a median.

`pruned`, and every competitor except AGM, still run on the full 400-graph sample.
"""

from __future__ import annotations

import json
import random
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
KEYS = ["mutagenicity", "protein"]
MDFSC_PROJECTIONS = 50_000

AGM_BUDGET = 100_000
AGM_SUBSAMPLE = 50
ISAL_BUDGET = 2.0
MAX_GRAPHS = 400
SEED = 42


def tuple_string(G):
    code = min_dfs.min_dfs_code(G, max_projections=MDFSC_PROJECTIONS)
    return "".join(chr(0xE000 + i * 100 + j) for i, j, *_ in code)


def bits(name, code, G):
    n, m = G.number_of_nodes(), G.number_of_edges()
    if name in ("graph6", "sparse6", "nauty->graph6"):
        return len(code) * 6.0
    if name == "adjacency":
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
    "min-DFS (tuples)",
    "IsalGraph pruned",
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
        for G in graphs:
            for name, fn in ENC.items():
                t = time.perf_counter()
                try:
                    codes[name].append(fn(G))
                except Exception:
                    codes[name].append(None)
                times[name] += time.perf_counter() - t
        codes["GED construction"] = ["" for _ in graphs]

        isal = B.isalgraph_strings(graphs, budget=ISAL_BUDGET)
        codes["IsalGraph pruned"] = [s["pruned"] for s in isal]
        times["IsalGraph pruned"] = sum(s["pruned_s"] for s in isal)
        isal_p_fail = sum(s["pruned"] is None for s in isal)
        isal_c_fail = sum(s["canonical"] is None for s in isal)
        t_canon = sum(s["canonical_s"] for s in isal)

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

        # AGM: failure rate on a subsample, no bit row
        sub = rng.sample(range(len(graphs)), min(AGM_SUBSAMPLE, len(graphs)))
        agm_fail = 0
        t_agm = time.perf_counter()
        for i in sub:
            try:
                B.agm_code(graphs[i], budget=AGM_BUDGET)
            except agm_cam.AGMBudgetExceeded:
                agm_fail += 1
        t_agm = time.perf_counter() - t_agm
        print(
            f"  AGM CAM: {agm_fail}/{len(sub)} budget failures "
            f"({100 * agm_fail / len(sub):.0f} %), {1e3 * t_agm / len(sub):.0f} ms/graph "
            f"[50-graph subsample, 100k-node budget]"
        )
        print(
            f"  IsalGraph canonical: {isal_c_fail}/{len(graphs)} timeouts at {ISAL_BUDGET}s, "
            f"{1e3 * t_canon / len(graphs):.0f} ms/graph  |  pruned failures {isal_p_fail}"
            f"  |  {time.perf_counter() - t0:.0f}s total",
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
            "agm_subsample": len(sub),
            "agm_ms_per_graph": 1e3 * t_agm / len(sub),
            "isal_pruned_fail": isal_p_fail,
            "isal_canonical_fail": isal_c_fail,
            "isal_canonical_budget_s": ISAL_BUDGET,
        }
        with open("real_suite2b.json", "w") as fh:
            json.dump(out, fh, indent=2)
    print("\nwrote real_suite2b.json")


if __name__ == "__main__":
    main()
