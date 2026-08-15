"""``python -m isalgraph.competitors.f5`` -- F5, and F5 only.

Spearman rho of each (representation x distance) against T-03's **certified
exact GED**, on Suite 1.

**This is reported and is not an input to selection.**  It lives in its own
entry point precisely so that ``grid.py`` -- which applies
``competitors.md`` §3.4's rule -- has no import path that could reach a GED
value.  Selecting a distance on its correlation with GED would be selecting
the baseline that makes IsalGraph look best, and decision 24's defence is
that the selection tool *could not* have done so.

**Every rho printed here carries its size null.**  ``rho(|n1 - n2|, GED)``
scores 0.71-0.93 on the five Suite-1 datasets, and IsalGraph clears it on
two of five by at most 0.03.  A rho without the null beside it hands the
first reviewer who computes it an unanswerable objection, so the null is
emitted in the same record rather than left to a later script.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from isalgraph.competitors import datasets
from isalgraph.competitors.base import VectorBackend
from isalgraph.competitors.ged_reference import load_ged
from isalgraph.competitors.registry import (
    available_backends,
    get_backend,
    get_metric,
)
from isalgraph.errors import CompetitorError


def _spearman(xs: list[float], ys: list[float]) -> tuple[float, float]:
    from scipy.stats import spearmanr

    result = spearmanr(xs, ys)
    return float(result.statistic), float(result.pvalue)


def rho_for(
    backend_name: str,
    metric_name: str,
    dataset: str,
    indices: tuple[int, ...],
) -> dict[str, Any]:
    """Spearman rho for one cell on one dataset's certified-exact pairs."""
    cohort = datasets.load(dataset)
    reference = load_ged(dataset)
    pairs = reference.certified_pairs(indices)
    if not pairs:
        return {"rho": None, "reason": "no certified-exact pairs in the sample"}

    backend = get_backend(backend_name)
    metric = get_metric(metric_name)
    graphs = {i: cohort.graphs[i] for i in indices}

    try:
        if isinstance(backend, VectorBackend):
            backend.fit(list(graphs.values()))
            encoded: dict[int, Any] = {i: backend.features(g) for i, g in graphs.items()}
        else:
            encoded = {i: backend.encode(g) for i, g in graphs.items()}
    except CompetitorError as exc:
        return {"rho": None, "reason": f"{type(exc).__name__}: {exc}"}

    distances: list[float] = []
    geds: list[float] = []
    undefined = 0
    for a, b in pairs:
        if not metric.is_defined(encoded[a], encoded[b]):
            undefined += 1
            continue
        distances.append(metric.distance(encoded[a], encoded[b]))
        geds.append(float(reference.ged[a, b]))
    if len(distances) < 3:
        return {"rho": None, "reason": "fewer than three defined pairs"}

    rho, p = _spearman(distances, geds)
    return {
        "rho": rho,
        "p": p,
        "n_pairs": len(distances),
        "n_undefined": undefined,
        "zero_frac": sum(d == 0.0 for d in distances) / len(distances),
    }


def size_null_rho(dataset: str, indices: tuple[int, ...]) -> dict[str, Any]:
    """``rho(|n1 - n2|, GED)`` -- the baseline every other row is judged against."""
    cohort = datasets.load(dataset)
    reference = load_ged(dataset)
    pairs = reference.certified_pairs(indices)
    if not pairs:
        return {"rho": None, "reason": "no certified-exact pairs"}
    orders = {i: cohort.graphs[i].number_of_nodes() for i in indices}
    diffs = [float(abs(orders[a] - orders[b])) for a, b in pairs]
    geds = [float(reference.ged[a, b]) for a, b in pairs]
    rho, p = _spearman(diffs, geds)
    equal_n = sum(orders[a] == orders[b] for a, b in pairs)
    return {
        "rho": rho,
        "p": p,
        "n_pairs": len(pairs),
        "equal_n_pairs": equal_n,
        "equal_n_frac": equal_n / len(pairs),
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(prog="python -m isalgraph.competitors.f5")
    parser.add_argument("--datasets", default=",".join(datasets.SUITE1))
    parser.add_argument("--n-graphs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", default="levenshtein")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    payload: dict[str, Any] = {
        "seed": args.seed,
        "n_graphs": args.n_graphs,
        "metric": args.metric,
        "note": (
            "DESCRIPTIVE. F5 is not an input to distance selection; grid.py "
            "applies competitors.md 3.4 and cannot import this module's data."
        ),
        "results": {},
    }
    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        cohort = datasets.load(dataset)
        indices = cohort.sample(args.n_graphs, seed=args.seed)
        row: dict[str, Any] = {"size_null": size_null_rho(dataset, indices)}
        for backend in available_backends():
            metric = "kernel" if backend == "wl_subtree" else args.metric
            row[backend] = rho_for(backend, metric, dataset, indices)
        payload["results"][dataset] = row
        null = row["size_null"]["rho"]
        print(f"\n=== {dataset}  null={null:.4f} ===")
        for name, rec in row.items():
            if name == "size_null" or rec.get("rho") is None:
                continue
            print(f"  {name:20s} {rec['rho']:>8.4f}   vs null {rec['rho'] - null:+.4f}")

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
