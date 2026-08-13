"""Feasibility and bit accounting across the Suite-2 cohort profiles and up to n = 98.

Every cell is attempted; a cell that fails is a result and is printed as one.
IsalGraph is given a 10 s per-graph budget; the C++ engine honours `timeout_s` and
raises `CanonicalizationTimeoutError`, which is recorded rather than swallowed.

Sizes and edge counts are the measured Suite-2 per-dataset means from
`.claude/notes/review/plan/data.md` §1.2 (T-01, 2026-08-13).  The graphs themselves are
synthetic -- the IAM corpus is not on this workstation.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import time

import agm_cam
import backends as B
import networkx as nx
from sweep import COHORT, random_connected

CEILING = [
    (f"n={n} m={int(r * n)}", n, int(r * n)) for n in (20, 30, 50, 70, 98) for r in (1.05, 2.0)
]

BUDGET_S = 10.0
AGM_NODES = 300_000


def one(G: nx.Graph, isal: dict) -> dict:
    n, m = G.number_of_nodes(), G.number_of_edges()
    logn = max(n - 1, 1).bit_length()
    bits: dict[str, float | None] = {}
    t: dict[str, float] = {}

    for key, fn in (
        ("graph6", B.graph6),
        ("sparse6", B.sparse6),
        ("nauty->graph6", B.nauty_canon_graph6),
        ("min-DFS code", B.mdfsc),
    ):
        t0 = time.perf_counter()
        s = fn(G)
        t[key] = time.perf_counter() - t0
        bits[key] = len(s) * 6.0 if key != "min-DFS code" else m * 2.0 * logn

    bits["adjacency"] = n * (n - 1) / 2
    bits["GED construction"] = (n - 1 + m) + 2 * m * logn

    t0 = time.perf_counter()
    try:
        code, nodes = B.agm_code(G, budget=AGM_NODES)
        bits["AGM CAM"] = float(len(code))
        agm = (nodes, "exact")
    except agm_cam.AGMBudgetExceeded as e:
        bits["AGM CAM"] = None
        agm = (e.nodes, "BUDGET")
    t["AGM CAM"] = time.perf_counter() - t0

    for key, tag in (("IsalGraph pruned", "pruned"), ("IsalGraph canonical", "canonical")):
        s = isal[tag]
        bits[key] = len(s) * math.log2(9) if s is not None else None
        t[key] = isal[tag + "_s"]
    return {
        "bits": bits,
        "t": t,
        "agm": agm,
        "n": n,
        "m": m,
        "isal_err": (isal.get("pruned_err"), isal.get("canonical_err")),
    }


METHODS = [
    "adjacency",
    "graph6",
    "sparse6",
    "nauty->graph6",
    "AGM CAM",
    "min-DFS code",
    "IsalGraph pruned",
    "IsalGraph canonical",
    "GED construction",
]


def run(profiles, reps: int, seed: int = 42):
    rng = random.Random(seed)
    out = []
    for label, n, m in profiles:
        graphs = [random_connected(n, m, rng) for _ in range(reps)]
        isal = B.isalgraph_strings(graphs, budget=BUDGET_S)
        rows = [one(G, s) for G, s in zip(graphs, isal)]
        agg = {"label": label, "n": n, "m": m, "reps": reps}
        for k in METHODS:
            vals = [r["bits"][k] for r in rows if r["bits"][k] is not None]
            agg[k] = statistics.median(vals) if vals else None
            agg[k + "__ok"] = f"{len(vals)}/{reps}"
        agg["_t"] = {k: statistics.median(r["t"][k] for r in rows) for k in rows[0]["t"]}
        agg["_agm_nodes"] = statistics.median(r["agm"][0] for r in rows)
        agg["_agm_ok"] = sum(r["agm"][1] == "exact" for r in rows)
        out.append(agg)
        print(
            f"  {label:<18} agm {agg['_agm_ok']}/{reps} exact, "
            f"isal pruned {agg['IsalGraph pruned__ok']}, "
            f"canon {agg['IsalGraph canonical__ok']}",
            flush=True,
        )
    return out


def table(rows, title):
    print("\n" + "=" * 132)
    print(title)
    print("=" * 132)
    print(f"{'profile':<16}{'n':>4}{'m':>5}" + "".join(f"{k[:14]:>14}" for k in METHODS))
    for r in rows:
        line = f"{r['label']:<16}{r['n']:>4}{r['m']:>5}"
        for k in METHODS:
            v = r[k]
            line += f"{'FAIL':>14}" if v is None else f"{v:>14.0f}"
        print(line)
    print(
        f"\n{'profile':<16}{'graph6 ms':>11}{'sparse6 ms':>11}{'nauty ms':>11}"
        f"{'mDFSC ms':>11}{'AGM ms':>11}{'AGMnodes':>11}{'AGMexact':>10}"
        f"{'Isal-pr ms':>12}{'Isal-cn ms':>12}{'cn ok':>8}"
    )
    for r in rows:
        t = r["_t"]
        print(
            f"{r['label']:<16}{t['graph6'] * 1e3:>11.3f}{t['sparse6'] * 1e3:>11.3f}"
            f"{t['nauty->graph6'] * 1e3:>11.3f}{t['min-DFS code'] * 1e3:>11.3f}"
            f"{t['AGM CAM'] * 1e3:>11.2f}{r['_agm_nodes']:>11.0f}"
            f"{r['_agm_ok']:>7}/{r['reps']:<2}"
            f"{t['IsalGraph pruned'] * 1e3:>12.2f}{t['IsalGraph canonical'] * 1e3:>12.2f}"
            f"{r['IsalGraph canonical__ok']:>8}"
        )


if __name__ == "__main__":
    print("### cohort profiles (Suite-2 measured means)")
    a = run(COHORT, reps=5)
    table(a, "ENTROPY-BOUND BITS, median over 5 graphs")
    print("\n### ceiling sweep")
    b = run(CEILING, reps=3)
    table(b, "ENTROPY-BOUND BITS, median over 3 graphs")
    with open("scale.json", "w") as fh:
        json.dump({"cohort": a, "ceiling": b}, fh, indent=2)
    print("\nwrote scale.json")
