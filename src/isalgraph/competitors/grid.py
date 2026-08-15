"""``python -m isalgraph.competitors.grid`` -- the (representation x distance) grid.

Computes **F1, F2, F3, F4 and F6**.  It does **not** compute F5 and it
cannot: nothing in this module's import closure reaches a GED loader, and a
test asserts that.

That is the point.  Decision 24's whole defence is that T-04a's exclusion
rule could not have seen the outcome it selects on.  ``competitors.md``
§3.4 fixes the rule in advance -- *the primary distance is the cheapest that
passes F1 at 100 %, F2, F3 and F4; ties break on F6, never on F5* -- and
selecting on correlation with GED would be selecting the baseline that makes
IsalGraph look best.  Prose cannot enforce it; an import graph can.

F5 lives in ``python -m isalgraph.competitors.f5``, whose output is reported
and is **not an input to selection**.

**Every cell is attempted, and a cell that fails is a result** -- one a
reviewer would otherwise ask about.  ``padded_hamming`` x ``sparse6`` is
undefined and prints as such; that cell is one of the reasons the grid
exists.

T-04 ships this and proves it runs end to end on a 20-graph dry run.
**T-04a** runs it on the 200-graph stratified sample under its own protocol
and applies the selection rule.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx

from isalgraph.competitors import datasets, fixtures
from isalgraph.competitors.base import (
    Capability,
    Comparable,
    Encoding,
    ReprBackend,
    VectorBackend,
    table_scope_error,
)
from isalgraph.competitors.registry import (
    AnyBackend,
    AnyMetric,
    available_backends,
    available_metrics,
    get_backend,
    get_metric,
)
from isalgraph.errors import CompetitorError, DistanceUndefined

#: F2 is checked over this many random triples.
F2_TRIPLES = 5_000
#: F3 protocol, frozen.
F3_GRAPHS = 50
F3_RELABELLINGS = 20
#: F6's threshold: a distance costing more than this is unaffordable.
F6_MS_PER_PAIR_LIMIT = 1.0


@dataclass
class Cell:
    """One (representation x distance) cell's measurements."""

    backend: str
    metric: str
    #: ``None`` when the pairing is structurally impossible, e.g. a metric
    #: that consumes ``features`` against a ``ReprBackend``.
    applicable: bool = True
    reason: str | None = None
    #: F1: fraction of pairs on which the distance is computable.
    f1_defined_frac: float | None = None
    #: F2: metric axioms.  ``is_pseudometric`` is the backend's declaration;
    #: the violations are what was actually observed.
    f2_declared_pseudometric: bool | None = None
    f2_violations: dict[str, int] | None = None
    #: F3: ``k / 50`` graphs whose distance to a relabelled self is 0.
    f3_invariant: str | None = None
    #: F4: degeneracy.
    f4_zero_mass: float | None = None
    f4_coeff_variation: float | None = None
    #: F6: cost.
    f6_ms_per_pair: float | None = None
    #: Whether this cell is eligible to be a primary distance.
    passes_selection: bool = False
    excluded_because: str | None = None


def _encode_all(backend: ReprBackend, graphs: list[nx.Graph]) -> tuple[list[Encoding | None], int]:
    out: list[Encoding | None] = []
    failed = 0
    for graph in graphs:
        try:
            out.append(backend.encode(graph))
        except Exception:  # noqa: BLE001 - a failure is a datum, not a stop
            out.append(None)
            failed += 1
    return out, failed


def _applicable(backend: AnyBackend, metric: AnyMetric) -> tuple[bool, str | None]:
    """Whether this pairing can exist at all, before anything is computed."""
    is_vector = isinstance(backend, VectorBackend)
    if metric.consumes == "features" and not is_vector:
        return False, "kernel distance consumes features; this backend is a serialisation"
    if metric.consumes != "features" and is_vector:
        return False, f"{metric.name} consumes {metric.consumes}; WL emits a feature vector"
    return True, None


def _f2(metric: AnyMetric, codes: list[Comparable], rng: random.Random) -> dict[str, int]:
    """Identity, symmetry and the triangle inequality over random triples.

    A violation is **declared, not repaired**.  Comparing a metric against a
    non-metric is legitimate; comparing them without saying so is not.
    """
    violations = {"identity": 0, "symmetry": 0, "triangle": 0}
    if len(codes) < 3:
        return violations
    for _ in range(F2_TRIPLES):
        a, b, c = (codes[rng.randrange(len(codes))] for _ in range(3))
        try:
            if metric.distance(a, a) != 0.0:
                violations["identity"] += 1
            d_ab = metric.distance(a, b)
            if d_ab != metric.distance(b, a):
                violations["symmetry"] += 1
            if d_ab > metric.distance(a, c) + metric.distance(c, b) + 1e-9:
                violations["triangle"] += 1
        except DistanceUndefined:
            continue
    return violations


def measure_cell(
    backend_name: str,
    metric_name: str,
    graphs: list[nx.Graph],
    *,
    seed: int,
    suite: str | None = None,
) -> Cell:
    """Measure one cell.  Never raises on a backend or metric failure.

    Args:
        suite: ``"suite1"`` / ``"suite2"`` when every graph comes from one
            named dataset, so a printed row can be refused where it would be
            conditioned on tractability.  ``None`` for a pooled sample, where
            a ``SUITE1_ONLY`` backend simply drives F1 below 1.0 and is
            excluded from selection on the measurement itself.
    """
    cell = Cell(backend=backend_name, metric=metric_name)
    try:
        backend = get_backend(backend_name)
        metric = get_metric(metric_name)
    except CompetitorError as exc:
        cell.applicable = False
        cell.reason = f"{type(exc).__name__}: {exc}"
        return cell

    if suite is not None:
        scope = table_scope_error(backend.capabilities, suite, backend_name)
        if scope is not None:
            cell.applicable = False
            cell.reason = scope
            return cell

    ok, reason = _applicable(backend, metric)
    if not ok:
        cell.applicable = False
        cell.reason = reason
        return cell

    rng = random.Random(seed)
    if isinstance(backend, VectorBackend):
        backend.fit(graphs)
        items: list[Comparable] = [dict(backend.features(g)) for g in graphs]
    else:
        encoded, _ = _encode_all(backend, graphs)
        items = [e for e in encoded if e is not None]
    if len(items) < 2:
        cell.applicable = False
        cell.reason = "fewer than two graphs encoded"
        return cell

    pairs = [(items[i], items[j]) for i in range(len(items)) for j in range(i + 1, len(items))]
    defined = [p for p in pairs if metric.is_defined(*p)]
    cell.f1_defined_frac = len(defined) / len(pairs) if pairs else 0.0

    cell.f2_declared_pseudometric = metric.is_pseudometric
    cell.f2_violations = _f2(metric, items, rng) if not isinstance(items[0], dict) else None

    t0 = time.perf_counter()
    values = [metric.distance(*p) for p in defined]
    cell.f6_ms_per_pair = 1e3 * (time.perf_counter() - t0) / len(defined) if defined else None

    if values:
        cell.f4_zero_mass = sum(v == 0.0 for v in values) / len(values)
        mean = statistics.fmean(values)
        cell.f4_coeff_variation = statistics.pstdev(values) / mean if mean else float("inf")

    cell.f3_invariant = _f3_cell(backend, metric, graphs[:F3_GRAPHS], seed)
    _apply_selection_rule(cell, backend)
    return cell


def _f3_cell(backend: AnyBackend, metric: AnyMetric, graphs: list[nx.Graph], seed: int) -> str:
    """Distance from a graph to a genuinely relabelled copy of itself must be 0."""
    rng = random.Random(seed)
    invariant = 0
    for graph in graphs:
        try:
            copies = [fixtures.shuffled_copy(graph, rng) for _ in range(F3_RELABELLINGS)]
            if isinstance(backend, VectorBackend):
                backend.fit([graph, *copies])
                base = dict(backend.features(graph))
                distances = [metric.distance(base, dict(backend.features(c))) for c in copies]
            else:
                base_enc = backend.encode(graph)
                distances = [
                    metric.distance(base_enc, backend.encode(c))
                    for c in copies
                    if metric.is_defined(base_enc, backend.encode(c))
                ]
        except (CompetitorError, Exception):  # noqa: B014 - a failure is not invariance
            continue
        if distances and all(d == 0.0 for d in distances):
            invariant += 1
    return f"{invariant}/{len(graphs)}"


def _apply_selection_rule(cell: Cell, backend: AnyBackend) -> None:
    """``competitors.md`` §3.4, fixed in advance and applied mechanically.

    A **baseline** backend is never eligible: ``size_null`` is a descriptive
    row, not a competitor, and letting it win a selection would put it into
    the confirmatory family and break decision 23.
    """
    if Capability.BASELINE in backend.capabilities:
        cell.excluded_because = "backend carries Capability.BASELINE; never a primary distance"
        return
    reasons = []
    if cell.f1_defined_frac is None or cell.f1_defined_frac < 1.0:
        reasons.append(f"F1 = {cell.f1_defined_frac}")
    if cell.f2_violations and any(cell.f2_violations.values()):
        reasons.append(f"F2 violations {cell.f2_violations}")
    if cell.f3_invariant is not None:
        got, total = (int(x) for x in cell.f3_invariant.split("/"))
        if total and got < total:
            reasons.append(f"F3 = {cell.f3_invariant}")
    if cell.f4_zero_mass is not None and cell.f4_zero_mass > 0.5:
        reasons.append(f"F4 degenerate, zero mass {cell.f4_zero_mass:.3f}")
    if cell.f4_coeff_variation is not None and cell.f4_coeff_variation < 1e-6:
        reasons.append("F4 near-constant")
    cell.passes_selection = not reasons
    cell.excluded_because = "; ".join(reasons) or None


def select_primary(cells: list[Cell]) -> dict[str, str | None]:
    """Cheapest passing distance per representation.  **Ties break on F6.**

    Never on F5 -- this module cannot compute F5, so the rule is enforced by
    the absence of the data rather than by the discipline of the caller.
    """
    out: dict[str, str | None] = {}
    for cell in cells:
        out.setdefault(cell.backend, None)
    for backend in out:
        eligible = [c for c in cells if c.backend == backend and c.passes_selection]
        if not eligible:
            continue
        out[backend] = min(
            eligible, key=lambda c: (c.f6_ms_per_pair or float("inf"), c.metric)
        ).metric
    return out


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(prog="python -m isalgraph.competitors.grid")
    parser.add_argument(
        "--sample",
        default="dryrun-20",
        help="'dryrun-20' or 'stratified-<k>'; the latter is T-04a's protocol",
    )
    parser.add_argument("--dataset", default="iam_letter_low")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if args.sample.startswith("dryrun-"):
        k = int(args.sample.split("-")[1])
        cohort = datasets.load(args.dataset)
        graphs = [cohort.graphs[i] for i in cohort.sample(k, seed=args.seed)]
        sample_desc = {"kind": "dryrun", "dataset": args.dataset, "k": k}
        sample_suite: str | None = cohort.suite
    elif args.sample.startswith("stratified-"):
        k = int(args.sample.split("-")[1])
        picked = datasets.stratified_sample(datasets.ALL_DATASETS, k, seed=args.seed)
        graphs = [datasets.load(ds).graphs[i] for ds, idxs in picked.items() for i in idxs]
        sample_desc = {
            "kind": "stratified",
            "k": k,
            "per_dataset": {d: len(v) for d, v in picked.items()},
        }
        # Pooled across both suites, so there is no single suite to refuse a
        # row for; a SUITE1_ONLY backend shows up as F1 < 1.0 instead.
        sample_suite = None
    else:
        parser.error(f"unknown --sample {args.sample!r}")

    backends = available_backends(include_baseline=True)
    metrics = available_metrics()
    cells = [
        measure_cell(b, m, graphs, seed=args.seed, suite=sample_suite)
        for b in backends
        for m in metrics
    ]

    payload = {
        "sample": sample_desc,
        "seed": args.seed,
        "n_graphs": len(graphs),
        "backends": list(backends),
        "metrics": list(metrics),
        "cells": [asdict(c) for c in cells],
        "primary_distance": select_primary(cells),
        "f5": "NOT COMPUTED HERE, BY CONSTRUCTION -- see isalgraph.competitors.f5",
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    print(f"{'backend':20s}{'metric':17s}{'F1':>7}{'F3':>8}{'F4=0':>7}{'us/pair':>10}  verdict")
    for cell in cells:
        if not cell.applicable:
            blank = f"{'':>7}{'':>8}{'':>7}{'':>10}"
            print(f"{cell.backend:20s}{cell.metric:17s}{blank}  n/a: {cell.reason}")
            continue
        print(
            f"{cell.backend:20s}{cell.metric:17s}"
            f"{cell.f1_defined_frac or 0:>7.3f}{cell.f3_invariant or '-':>8}"
            f"{cell.f4_zero_mass or 0:>7.3f}{1e3 * (cell.f6_ms_per_pair or 0):>10.2f}  "
            + ("PASS" if cell.passes_selection else f"excluded: {cell.excluded_because}")
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
