"""T-28: per-competitor head-to-head tally under every reference.

The "best competitor" summary answers *is the arm the single best representation?*
That is the strictest reading and it hides the structure a reader needs: an arm can
be second of seven on every dataset and still record zero wins. This script reports
the pairwise tally instead -- for each (reference, competitor), how many cells the
IsalGraph arm's rho exceeds that competitor's.

Point estimates only. A difference here is not a significance verdict; the paired
graph-level bootstrap in the production campaign supplies that.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ARM = "isalgraph_pruned"
DEGENERATE = {("wl", "wl_subtree")}
REF_ORDER = ["exact", "lb", "ub", "wl", "spectral", "spectral_comb", "spectral_adj", "spectral_esd"]
REF_LABEL = {
    "exact": "GED exact",
    "lb": "GED lower bd",
    "ub": "GED upper bd",
    "wl": "WL kernel",
    "spectral": "spectral norm-L",
    "spectral_comb": "spectral comb-L",
    "spectral_adj": "spectral adj",
    "spectral_esd": "spectral ESD",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--view", default="all_pairs")
    args = ap.parse_args()

    rows: list[dict[str, Any]] = json.loads(args.probe.read_text())["rows"]
    rows = [r for r in rows if r["view"] == args.view]

    cells: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(dict)
    for r in rows:
        cells[(r["suite"], r["dataset"], r["reference"])][r["representation"]] = r

    # (reference, competitor) -> [arm_higher, competitor_higher, sum_delta, n]
    tally: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for (_suite, _dataset, ref), by_rep in cells.items():
        if ARM not in by_rep:
            continue
        arm_rho = by_rep[ARM]["rho"]
        for rep, rec in by_rep.items():
            if rep == ARM or rep.startswith("isalgraph") or (ref, rep) in DEGENERATE:
                continue
            t = tally[(ref, rep)]
            t[3] += 1
            t[2] += arm_rho - rec["rho"]
            if arm_rho > rec["rho"]:
                t[0] += 1
            else:
                t[1] += 1

    competitors = sorted({c for _, c in tally})
    print(f"\n=== per-competitor head-to-head | view = {args.view} | arm = {ARM} ===")
    print("cells where IsalGraph's rho is higher, out of all cells; (mean delta)\n")
    header = f"{'reference':<18}" + "".join(f"{c[:14]:>17}" for c in competitors)
    print(header)
    print("-" * len(header))
    for ref in REF_ORDER:
        present = [(ref, c) for c in competitors if (ref, c) in tally]
        if not present:
            continue
        line = f"{REF_LABEL.get(ref, ref):<18}"
        for c in competitors:
            key = (ref, c)
            if key not in tally:
                line += f"{'-':>17}"
                continue
            w, _ls, sd, n = tally[key]
            line += f"{int(w)}/{int(n)} ({sd / n:+.3f})".rjust(17)
        print(line)

    print("\nDegenerate cells excluded:", ", ".join(f"{r}/{c}" for r, c in sorted(DEGENERATE)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
