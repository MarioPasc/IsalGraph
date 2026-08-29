"""Summarise a T-28 probe file into the head-to-head table the decision needs.

Prints, per (reference, view), how many (suite, dataset) records the IsalGraph
arm wins, ties or loses against the best competing representation, and whether
it clears the ``|n_i - n_j|`` size null. The WL reference is degenerate for the
``wl_subtree`` arm -- that arm's distance *is* the reference -- so it is marked
and excluded from the win count rather than silently dropped.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ARM = "isalgraph_pruned"
#: Under this reference, this arm's distance is byte-identical to the reference.
DEGENERATE = {("wl", "wl_subtree")}

REF_LABEL = {
    "exact": "GED exact",
    "lb": "GED lower bd",
    "ub": "GED upper bd",
    "wl": "WL kernel",
    "spectral": "spectral (norm L)",
    "spectral_comb": "spectral (comb L)",
    "spectral_adj": "spectral (adj)",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--view", default="all_pairs")
    ap.add_argument("--detail", action="store_true", help="print the per-dataset rows")
    args = ap.parse_args()

    rows: list[dict[str, Any]] = json.loads(args.probe.read_text())["rows"]
    rows = [r for r in rows if r["view"] == args.view]

    cells: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(dict)
    for r in rows:
        cells[(r["suite"], r["dataset"], r["reference"])][r["representation"]] = r

    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])  # win, loss, clears null, total
    detail: dict[str, list[str]] = defaultdict(list)

    for (suite, dataset, ref), by_rep in sorted(cells.items()):
        if ARM not in by_rep:
            continue
        arm_rho = by_rep[ARM]["rho"]
        null_rho = by_rep[ARM].get("size_null_rho")
        competitors = {
            k: v["rho"]
            for k, v in by_rep.items()
            if k != ARM
            and not k.startswith("isalgraph")
            and (ref, k) not in DEGENERATE
        }
        if not competitors:
            continue
        best_rep = max(competitors, key=lambda k: competitors[k])
        best = competitors[best_rep]
        t = tally[ref]
        t[3] += 1
        if arm_rho > best:
            t[0] += 1
            verdict = "WIN "
        else:
            t[1] += 1
            verdict = "loss"
        clears = null_rho is not None and arm_rho > null_rho
        if clears:
            t[2] += 1
        nl = f"{null_rho:+.4f}" if null_rho is not None else "  n/a "
        detail[ref].append(
            f"  {verdict}  {suite[-1]}/{dataset:<16} arm={arm_rho:+.4f}  "
            f"best={best:+.4f} ({best_rep:<14}) null={nl} {'clears' if clears else '  ---'}"
        )

    order = ["exact", "lb", "ub", "wl", "spectral", "spectral_comb", "spectral_adj"]
    print(f"\n=== view = {args.view} | arm = {ARM} ===")
    print(f"{'reference':<20} {'win':>4} {'loss':>5} {'clears null':>12} {'records':>8}")
    print("-" * 54)
    for ref in order:
        if ref not in tally:
            continue
        w, ls, c, n = tally[ref]
        print(f"{REF_LABEL.get(ref, ref):<20} {w:>4} {ls:>5} {c:>7}/{n:<4} {n:>8}")
    if args.detail:
        for ref in order:
            if ref in detail:
                print(f"\n--- {REF_LABEL.get(ref, ref)} ---")
                print("\n".join(detail[ref]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
