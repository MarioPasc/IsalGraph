"""Pairwise dominance across representations on BOTH claims.

Answers: does any representation dominate another on (compactness, GED
correlation)? And is IsalGraph on the Pareto frontier?

Claim A  -- median bits per (dataset, n) stratum, entropy convention, from the
            encodings directly so competitor-vs-competitor is available.
Claim B  -- Spearman rho per (dataset, n, reference) stratum, from size_profile.
Both are compared with a sign test over strata, which is the test a reviewer
runs and the one that refuted the "unresolved" framing.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
from scipy import stats

T = Path("/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/T06")
MIN_STRATA = 20
N_FLOOR = 20  # compare above this size; below it every claim behaves differently

# Representations carrying a bit count AND an admissible distance.
BITS = [
    "isalgraph_pruned",
    "min_dfs",
    "nauty_graph6",
    "sparse6_nauty",
    "adjacency",
    "graph6",
    "sparse6",
    "agm_cam",
]
ADMISSIBLE = {
    "isalgraph_pruned",
    "agm_cam",
    "min_dfs",
    "nauty_graph6",
    "sparse6_nauty",
    "wl_subtree",
}


def claim_a_strata() -> dict[tuple[str, str, int], dict[str, float]]:
    """Median entropy bits per representation, per (suite, dataset, n) stratum."""
    out: dict[tuple[str, str, int], dict[str, float]] = collections.defaultdict(dict)
    for suite in ("suite1", "suite2"):
        d = T / "encodings" / suite
        if not d.is_dir():
            continue
        for rep in BITS:
            for path in sorted(d.glob(f"*__{rep}.npz")):
                ds = path.stem.split("__")[0]
                with np.load(path, allow_pickle=True) as z:
                    nc = np.asarray(z["node_counts"]).astype(int)
                    st = np.asarray(z["status"]).astype(str)
                    bits = np.asarray(z["entropy_bits"], dtype=float)
                ok = ((st == "ok") | (st == "censored")) & np.isfinite(bits)
                for n in np.unique(nc[ok]):
                    if n <= N_FLOOR:
                        continue
                    m = ok & (nc == n)
                    if int(m.sum()) < 8:
                        continue
                    out[(suite, ds, int(n))][rep] = float(np.median(bits[m]))
    return out


def claim_b_strata() -> dict[tuple[str, str, int, str], dict[str, float]]:
    """Spearman rho per representation, per (suite, dataset, n, reference) stratum."""
    rows = json.loads((T / "size_profile.json").read_text())["rows"]
    out: dict[tuple[str, str, int, str], dict[str, float]] = collections.defaultdict(dict)
    for r in rows:
        if r["rho"] is None or r["n"] <= N_FLOOR:
            continue
        out[(r["suite"], r["dataset"], int(r["n"]), r["reference"])][r["representation"]] = r["rho"]
    return out


def sign_matrix(strata: dict, reps: list[str], lower_is_better: bool) -> dict:
    """Pairwise sign test: does row beat column on this axis?"""
    res: dict[tuple[str, str], tuple[int, int, float]] = {}
    for a in reps:
        for b in reps:
            if a == b:
                continue
            wins = losses = 0
            for cell in strata.values():
                if a not in cell or b not in cell:
                    continue
                if cell[a] == cell[b]:
                    continue
                a_better = (cell[a] < cell[b]) if lower_is_better else (cell[a] > cell[b])
                wins += a_better
                losses += not a_better
            if wins + losses < MIN_STRATA:
                continue
            p = stats.binomtest(wins, wins + losses, 0.5).pvalue
            res[(a, b)] = (wins, losses, p)
    return res


def verdict(entry: tuple[int, int, float] | None) -> str:
    """WIN / LOSS / tie for one ordered pair."""
    if entry is None:
        return "n/a"
    w, l, p = entry
    if p >= 0.05:
        return "tie"
    return "WIN" if w > l else "LOSS"


def main() -> None:
    a = sign_matrix(claim_a_strata(), BITS, lower_is_better=True)
    b_reps = sorted({r for cell in claim_b_strata().values() for r in cell})
    b = sign_matrix(claim_b_strata(), b_reps, lower_is_better=False)

    print(f"=== n > {N_FLOOR}. Sign test over strata, alpha = 0.05 ===\n")
    print("--- Claim A (compactness): row vs column ---")
    reps = [r for r in BITS if any(k[0] == r for k in a)]
    print(f"{'':20s}" + "".join(f"{c[:11]:>12s}" for c in reps))
    for r in reps:
        print(
            f"{r:20s}"
            + "".join(f"{verdict(a.get((r, c))):>12s}" if r != c else f"{'-':>12s}" for c in reps)
        )

    print("\n--- Claim B (GED correlation): row vs column ---")
    rb = [r for r in b_reps if r in ADMISSIBLE]
    print(f"{'':20s}" + "".join(f"{c[:11]:>12s}" for c in rb))
    for r in rb:
        print(
            f"{r:20s}"
            + "".join(f"{verdict(b.get((r, c))):>12s}" if r != c else f"{'-':>12s}" for c in rb)
        )

    print("\n=== Does anything DOMINATE isalgraph_pruned (>= on both, > on one)? ===")
    me = "isalgraph_pruned"
    for c in sorted(ADMISSIBLE - {me}):
        va = verdict(a.get((me, c))) if (me, c) in a else "n/a"
        vb = verdict(b.get((me, c))) if (me, c) in b else "n/a"
        dominated = (va in ("LOSS", "tie")) and (vb in ("LOSS", "tie")) and "LOSS" in (va, vb)
        flag = "  <-- DOMINATES IsalGraph" if dominated else ""
        cross = (
            "  <-- CROSS-OVER (we win one, lose the other)" if {va, vb} == {"WIN", "LOSS"} else ""
        )
        print(f"  vs {c:18s} ClaimA={va:5s} ClaimB={vb:5s}{flag}{cross}")


if __name__ == "__main__":
    main()
