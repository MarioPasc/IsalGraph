"""Paired graph-level bootstrap CI of rho(Lev, ref) - rho(|dn|, ref).

Written standalone so it cannot share a bug with f5.py.  The point of the
pairing: both arms are recomputed on the SAME resampled graph set every
iteration, so the difference's variance excludes the between-resample
variation that is common to both.  Comparing two marginal CIs instead --
either "is the point estimate inside the other interval" or "do the
intervals overlap" -- is the wrong test and is anti-conservative in one
direction and conservative in the other.
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import rankdata

from isalgraph.competitors import datasets
from isalgraph.competitors.registry import get_backend, get_metric

ROOT = datasets.cohort_root()
OUT = "/media/mpascual/Sandisk2TB/research/isalgraph/T-04a/paired_null_ci.json"
RESAMPLES = 2000
SEED = 42


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / d) if d else float("nan")


def reference(ds: str, arm: str, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (matrix, valid_mask) restricted to the draw."""
    if arm == "exact":
        p = os.path.join(
            ROOT, "source", "GED_PRECOMPUTED", "extended_merged_exact_ged", "computed", f"{ds}.npz"
        )
        z = np.load(p, allow_pickle=True)
        return z["ged_matrix"][np.ix_(idx, idx)], z["certified_mask"][np.ix_(idx, idx)]
    key = "lb_matrix" if arm == "lb" else "ub_matrix"
    z = np.load(
        os.path.join(ROOT, "source", "APPROX_GED", arm.upper(), f"{ds}.npz"), allow_pickle=True
    )
    m = z[key][np.ix_(idx, idx)]
    return m, np.isfinite(m)


def run(ds: str, arm: str) -> dict[str, object] | None:
    cohort = datasets.load(ds)
    idx = np.array(cohort.sample(200, seed=SEED))
    graphs = [cohort.graphs[i] for i in idx]
    backend, metric = get_backend("isalgraph_pruned"), get_metric("levenshtein")

    enc: dict[int, object] = {}
    for k, g in enumerate(graphs):
        try:
            enc[k] = backend.encode(g)
        except Exception:  # noqa: BLE001 -- a failure is a datum
            pass
    order = np.array([g.number_of_nodes() for g in graphs])
    ref, valid = reference(ds, arm, idx)

    ii, jj, lev, dn, ged = [], [], [], [], []
    for a in sorted(enc):
        for b in sorted(enc):
            if a >= b or not valid[a, b] or not np.isfinite(ref[a, b]):
                continue
            ii.append(a)
            jj.append(b)
            lev.append(metric.distance(enc[a], enc[b]))
            dn.append(abs(int(order[a]) - int(order[b])))
            ged.append(float(ref[a, b]))
    if len(ged) < 3:
        return None
    ii_a, jj_a = np.array(ii), np.array(jj)
    lev_a, dn_a, ged_a = np.array(lev, float), np.array(dn, float), np.array(ged, float)

    point = spearman(lev_a, ged_a) - spearman(dn_a, ged_a)
    rng = np.random.default_rng(SEED)
    n = len(graphs)
    diffs = np.empty(RESAMPLES)
    for r in range(RESAMPLES):
        keep = np.zeros(n, bool)
        keep[np.unique(rng.integers(0, n, n))] = True  # graph-level resample
        m = keep[ii_a] & keep[jj_a]
        diffs[r] = (
            spearman(lev_a[m], ged_a[m]) - spearman(dn_a[m], ged_a[m]) if m.sum() >= 3 else np.nan
        )
    good = diffs[np.isfinite(diffs)]
    lo, hi = np.percentile(good, [2.5, 97.5])
    return {
        "dataset": ds,
        "arm": arm,
        "n_pairs": len(ged),
        "n_encoded": len(enc),
        "rho_lev": spearman(lev_a, ged_a),
        "rho_null": spearman(dn_a, ged_a),
        "paired_diff": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_resamples_used": int(good.size),
    }


if __name__ == "__main__":
    jobs = [(d, "exact") for d in datasets.SUITE1]
    jobs += [(d, a) for d in datasets.SUITE2 for a in ("lb", "ub")]
    out = []
    print(f"{'record':22s}{'rho_lev':>9s}{'rho_null':>10s}{'diff':>9s}{'95% CI':>20s}  sig?")
    for ds, arm in jobs:
        r = run(ds, arm)
        if r is None:
            continue
        out.append(r)
        print(
            f"{ds + '::' + arm:22s}{r['rho_lev']:9.3f}{r['rho_null']:10.3f}"
            f"{r['paired_diff']:+9.3f}   [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]  "
            f"{'YES' if r['excludes_zero'] else 'no'}"
        )
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "statistic": "rho(Lev,ref) - rho(|dn|,ref), paired graph-level bootstrap",
                "resamples": RESAMPLES,
                "seed": SEED,
                "records": out,
            },
            fh,
            indent=2,
        )
    print(f"\nwrote {OUT}")
