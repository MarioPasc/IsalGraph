"""Suite 1 on the REAL cohort: Claim A, F3, feasibility, and rho against certified exact GED.

Data
----
graphs      data/exported/<ds>.npz                       (CSR: n_nodes, edge_offsets, edges)
exact GED   .../extended_merged_exact_ged/computed/<ds>.npz  (ged_matrix + certified_mask,
            cost model [1,1,0,1,1,0] = D6)

`graph_ids` align between the two files -- asserted, not assumed.

rho is computed on certified-exact pairs only, over a fixed 200-graph subsample per
dataset (seed 42), which is [competitors](competitors.md) §3.1's sample design.  It is
reported as DESCRIPTIVE: T-04a's selection rule (§3.4) is F5-blind by construction and
nothing here may change which distance is chosen.
"""

from __future__ import annotations

import json
import random
import statistics
import time

import agm_cam
import backends as B
import min_dfs
import networkx as nx
import numpy as np
from rapidfuzz.distance import Levenshtein as RFLev
from scipy.stats import spearmanr
from sweep import shuffled_copy

ROOT = "/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data"
GEDD = f"{ROOT}/source/GED_PRECOMPUTED/extended_merged_exact_ged/computed"
DATASETS = ["iam_letter_low", "iam_letter_med", "iam_letter_high", "linux", "aids"]

AGM_BUDGET = 200_000
RHO_SAMPLE = 200
SEED = 42


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


def tuple_string(G):
    return "".join(chr(0xE000 + i * 100 + j) for i, j, *_ in min_dfs.min_dfs_code(G))


ENCODERS = {
    "graph6": B.graph6,
    "sparse6": B.sparse6,
    "nauty->graph6": B.nauty_canon_graph6,
    "adjacency": B.adjacency_bits,
    "min-DFS (tuples)": tuple_string,
}


def bits(name: str, code: str, G: nx.Graph) -> float:
    n, m = G.number_of_nodes(), G.number_of_edges()
    if name in ("graph6", "sparse6", "nauty->graph6"):
        return len(code) * 6.0
    if name in ("adjacency", "AGM CAM"):
        return n * (n - 1) / 2
    if name == "min-DFS (tuples)":
        return m * 2.0 * max(n - 1, 1).bit_length()
    return len(code) * np.log2(9)  # IsalGraph


METHODS = [
    "adjacency",
    "graph6",
    "sparse6",
    "nauty->graph6",
    "AGM CAM",
    "min-DFS (tuples)",
    "IsalGraph pruned",
    "IsalGraph canonical",
]


def main() -> None:
    rng = random.Random(SEED)
    out: dict[str, dict] = {}

    for ds in DATASETS:
        t_ds = time.perf_counter()
        graphs, ged, cert = load(ds)
        N = len(graphs)
        print(
            f"\n=== {ds}  N={N}  n_max={max(g.number_of_nodes() for g in graphs)} ===", flush=True
        )

        codes: dict[str, list] = {k: [] for k in METHODS}
        times: dict[str, float] = {k: 0.0 for k in METHODS}
        agm_fail = 0
        for G in graphs:
            for name, fn in ENCODERS.items():
                t0 = time.perf_counter()
                codes[name].append(fn(G))
                times[name] += time.perf_counter() - t0
            t0 = time.perf_counter()
            try:
                codes["AGM CAM"].append(B.agm_code(G, budget=AGM_BUDGET)[0])
            except agm_cam.AGMBudgetExceeded:
                codes["AGM CAM"].append(None)
                agm_fail += 1
            times["AGM CAM"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        isal = B.isalgraph_strings(graphs, budget=30.0)
        t_isal = time.perf_counter() - t0
        codes["IsalGraph pruned"] = [s["pruned"] for s in isal]
        codes["IsalGraph canonical"] = [s["canonical"] for s in isal]
        times["IsalGraph pruned"] = sum(s["pruned_s"] for s in isal)
        times["IsalGraph canonical"] = sum(s["canonical_s"] for s in isal)
        isal_fail = sum(s["canonical"] is None for s in isal)

        # ---- Claim A ------------------------------------------------------
        bl = {
            k: [bits(k, c, G) if c is not None else None for c, G in zip(codes[k], graphs)]
            for k in METHODS
        }
        ref = bl["IsalGraph pruned"]
        print(
            f"{'method':<20}{'median bits':>12}{'mean bits':>11}"
            f"{'% Isal shorter':>16}{'ms/graph':>10}{'failed':>8}"
        )
        claim_a = {}
        for k in METHODS:
            vals = [v for v in bl[k] if v is not None]
            paired = [(a, b) for a, b in zip(bl[k], ref) if a is not None and b is not None]
            win = 100 * sum(b < a for a, b in paired) / len(paired) if paired else float("nan")
            claim_a[k] = {
                "median": statistics.median(vals),
                "mean": statistics.mean(vals),
                "isal_shorter_pct": win,
                "n_ok": len(vals),
                "ms_per_graph": 1e3 * times[k] / N,
            }
            print(
                f"{k:<20}{claim_a[k]['median']:>12.1f}{claim_a[k]['mean']:>11.1f}"
                f"{win:>16.1f}{claim_a[k]['ms_per_graph']:>10.3f}{N - len(vals):>8}"
            )
        print(
            f"  AGM budget failures {agm_fail}/{N} | IsalGraph canonical failures "
            f"{isal_fail}/{N} | isal wall-clock {t_isal:.1f}s",
            flush=True,
        )

        # ---- F3 on real graphs --------------------------------------------
        idx = rng.sample(range(N), min(50, N))
        f3 = {}
        for k, fn in ENCODERS.items():
            inv = sum(
                len({fn(shuffled_copy(graphs[i], rng)) for _ in range(20)} | {fn(graphs[i])}) == 1
                for i in idx
            )
            f3[k] = f"{inv}/{len(idx)}"
        print(
            "  F3 (50 graphs x 20 relabellings): " + "  ".join(f"{k}={v}" for k, v in f3.items()),
            flush=True,
        )

        # ---- rho vs certified exact GED -----------------------------------
        sub = sorted(rng.sample(range(N), min(RHO_SAMPLE, N)))
        pairs = [(a, b) for ii, a in enumerate(sub) for b in sub[ii + 1 :] if cert[a, b]]
        gd = np.array([ged[a, b] for a, b in pairs])
        print(
            f"  rho sample: {len(sub)} graphs, {len(pairs)} certified pairs "
            f"(GED {gd.min():.0f}-{gd.max():.0f})"
        )
        rho = {}
        for k in METHODS:
            cs = codes[k]
            if any(cs[a] is None or cs[b] is None for a, b in pairs):
                rho[k] = None
                continue
            d = np.array([RFLev.distance(cs[a], cs[b]) for a, b in pairs], dtype=float)
            r = spearmanr(d, gd)
            rho[k] = {
                "spearman": float(r.statistic),
                "p": float(r.pvalue),
                "zero_frac": float((d == 0).mean()),
            }
        print(f"  {'method':<20}{'rho(Lev, exact GED)':>22}{'frac d=0':>10}")
        for k in METHODS:
            v = rho[k]
            print(
                f"  {k:<20}"
                + (
                    f"{v['spearman']:>22.4f}{v['zero_frac']:>10.4f}"
                    if v
                    else f"{'n/a':>22}{'':>10}"
                ),
                flush=True,
            )

        out[ds] = {
            "N": N,
            "claim_a": claim_a,
            "f3": f3,
            "rho": rho,
            "agm_fail": agm_fail,
            "isal_canonical_fail": isal_fail,
            "n_pairs_rho": len(pairs),
            "n_max": int(max(g.number_of_nodes() for g in graphs)),
            "seconds": time.perf_counter() - t_ds,
        }
        with open("real_suite1.json", "w") as fh:
            json.dump(out, fh, indent=2)

    print("\nwrote real_suite1.json")


if __name__ == "__main__":
    main()
