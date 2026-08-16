"""``python -m isalgraph.competitors.f5`` -- F5, and F5 only.

Spearman rho of each representation's **selected primary distance** against
ground-truth graph edit distance.

**This is descriptive.  It is reported and it is not an input to selection.**
The primary distance is chosen by ``grid.py`` on F1-F4 with cost as the
tie-break, deliberately blind to rho; this module *consumes* that choice and
never feeds back into it.  The entry point is separate precisely so that
``grid.py`` has no import path that could reach a GED value: selecting a
distance on its correlation with GED would be selecting the baseline that
makes IsalGraph look best, and decision 24's defence is that the selection
tool *could not* have done so.  A test asserts both halves of that closure.

Three things this reports that a bare rho does not.

**Every rho carries its size null.**  ``rho(|n1 - n2|, GED)`` -- count the
nodes, subtract, no representation at all -- scores 0.71-0.93 on the five
Suite-1 datasets, and IsalGraph clears it on one of five.  A rho printed
without its null hands the first reviewer who computes it an unanswerable
objection, so the null is emitted in the same record rather than left to a
later script.

**Every rho carries a graph-level bootstrap interval.**  rho moved by up to
0.07 between two independent 200-graph draws on AIDS.  The effective sample
size is governed by *graphs*, not by pairs, so a pair-level interval is wrong
by construction; see :mod:`isalgraph.competitors.bootstrap`.  This is a
stated precondition for printing any rho in this revision.

**Every rho is reported in two views.**  ``all_pairs`` and ``equal_n``
(``n1 == n2``).  The equal-``n`` view removes the size channel entirely and
is therefore where canonicity, rather than order, has to do the work.

The Suite-2 arm reports T-05's proven bounds as **two separate records**,
``<dataset>::lb`` and ``<dataset>::ub``.  They are never averaged and never
interpolated into a midpoint (``approx_ged.md`` §4).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from isalgraph.competitors import datasets
from isalgraph.competitors.base import VectorBackend, table_scope_error
from isalgraph.competitors.bootstrap import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    ResampleIndex,
    graph_bootstrap_ci,
    make_resample_index,
    spearman,
)
from isalgraph.competitors.ged_reference import load_bounds, load_ged
from isalgraph.competitors.registry import available_backends, get_backend, get_metric
from isalgraph.errors import CompetitorError

if TYPE_CHECKING:
    import networkx as nx

#: Frozen protocol constant (design note 3.8): the per-dataset draw.
DEFAULT_N_GRAPHS = 200

#: The baseline row.  It is a registered backend and metric so that the null
#: falls out of the same encode/distance path every competitor uses, rather
#: than out of a second hand-written implementation that could drift from it.
SIZE_NULL = "size_null"

#: The two views, frozen.  ``equal_n`` is the one the paper leads with.
VIEWS: tuple[str, ...] = ("all_pairs", "equal_n")

#: Printed for a representation the grid found no admissible distance for.
NO_DISTANCE_REASON = "no admissible distance (T-04a selection)"

#: Printed for a representation absent from the grid's ``primary_distance``.
ABSENT_REASON = "absent from the grid's primary_distance map"


@dataclass(frozen=True, slots=True)
class Draw:
    """One dataset's sampled graphs, and the position map the bootstrap uses.

    Attributes:
        dataset: cohort name.
        suite: ``"suite1"`` or ``"suite2"``.
        indices: cohort indices, sorted, as ``Cohort.sample`` returned them.
        position: cohort index -> position in :attr:`indices`.  The bootstrap
            resamples **positions**, so every pair must be translated before
            it reaches :func:`~isalgraph.competitors.bootstrap.graph_bootstrap_ci`.
        n_nodes: cohort index -> node count, for the size null and the
            equal-``n`` view.
    """

    dataset: str
    suite: str
    indices: tuple[int, ...]
    position: dict[int, int]
    n_nodes: dict[int, int]


def draw_for(dataset: str, n_graphs: int, seed: int) -> Draw:
    """Sample *n_graphs* graphs from *dataset* at *seed*."""
    cohort = datasets.load(dataset)
    indices = cohort.sample(n_graphs, seed=seed)
    return Draw(
        dataset=dataset,
        suite=cohort.suite,
        indices=indices,
        position={idx: p for p, idx in enumerate(indices)},
        n_nodes={idx: int(cohort.graphs[idx].number_of_nodes()) for idx in indices},
    )


def encode_draw(backend_name: str, draw: Draw) -> tuple[dict[int, Any], int, dict[str, int]]:
    """Encode every sampled graph under *backend_name*.

    A graph the backend raises on is **excluded from that backend's pair set
    and counted**, never silently substituted.  Both the count and the
    exception names reach the record.

    Returns:
        ``(encoded, n_unencodable, errors)``.  ``encoded`` is keyed by cohort
        index and holds only the graphs that encoded; ``errors`` counts
        ``type(exc).__name__``.
    """
    cohort = datasets.load(draw.dataset)
    graphs: dict[int, nx.Graph] = {idx: cohort.graphs[idx] for idx in draw.indices}
    backend = get_backend(backend_name)
    encoded: dict[int, Any] = {}
    errors: dict[str, int] = {}

    if isinstance(backend, VectorBackend):
        # Fitted once over the whole draw: a per-batch vocabulary would make
        # the distance matrix depend on batching order.
        backend.fit(list(graphs.values()))

    for idx, graph in graphs.items():
        try:
            if isinstance(backend, VectorBackend):
                encoded[idx] = backend.features(graph)
            else:
                encoded[idx] = backend.encode(graph)
        except CompetitorError as exc:
            name = type(exc).__name__
            errors[name] = errors.get(name, 0) + 1
    return encoded, len(graphs) - len(encoded), errors


def reference_pairs(draw: Draw, reference: str) -> tuple[list[tuple[int, int]], list[float]]:
    """Ground-truth pairs and their GED, for one reference.

    Args:
        draw: the sampled graphs.
        reference: ``"exact"`` for Suite 1's certified A* GED, ``"lb"`` or
            ``"ub"`` for T-05's proven Suite-2 bounds.

    Returns:
        ``(pairs, geds)`` over the upper triangle of the draw.  Suite 1 keeps
        certified pairs only; the bounds keep finite pairs only.
    """
    if reference == "exact":
        matrix = load_ged(draw.dataset)
        pairs = matrix.certified_pairs(draw.indices)
        return pairs, [float(matrix.ged[a, b]) for a, b in pairs]
    bounds = load_bounds(draw.dataset, reference)
    pairs = bounds.finite_pairs(draw.indices)
    return pairs, [float(bounds.values[a, b]) for a, b in pairs]


def _view_mask(draw: Draw, pairs: list[tuple[int, int]], view: str) -> list[bool]:
    if view == "all_pairs":
        return [True] * len(pairs)
    if view == "equal_n":
        return [draw.n_nodes[a] == draw.n_nodes[b] for a, b in pairs]
    raise ValueError(f"unknown view {view!r}; expected one of {VIEWS}")


def _empty_record(metric: str | None, reason: str) -> dict[str, Any]:
    """A printed absence.  Every key of a full record is present and null."""
    return {
        "metric": metric,
        "rho": None,
        "p": None,
        "ci": None,
        "n_pairs": 0,
        "n_undefined": 0,
        "zero_frac": None,
        "reason": reason,
    }


def _constant_reason(distances: list[float], geds: list[float], metric_name: str) -> str | None:
    """Why Spearman is undefined here, or ``None`` when it is defined."""
    flat_distance = len(set(distances)) < 2
    flat_ged = len(set(geds)) < 2
    if not (flat_distance or flat_ged):
        return None
    if flat_distance and flat_ged:
        side = "both the distance and the reference GED are"
    elif flat_distance:
        side = f"the {metric_name!r} distance is"
    else:
        side = "the reference GED is"
    return (
        f"Spearman undefined: {side} constant over the pairs of this view, so the "
        f"rank correlation has no denominator"
    )


def rho_record(
    backend_name: str,
    metric_name: str,
    draw: Draw,
    encoded: dict[int, Any],
    pairs: list[tuple[int, int]],
    geds: list[float],
    index: ResampleIndex,
) -> dict[str, Any]:
    """One cell of one view: rho, its p-value, and its graph-level interval.

    Pairs whose distance is undefined are counted and excluded; pairs naming
    a graph this backend could not encode are excluded without being counted
    as undefined, since the count they belong in is ``n_unencodable``.
    """
    metric = get_metric(metric_name)
    distances: list[float] = []
    values: list[float] = []
    positions: list[tuple[int, int]] = []
    undefined = 0
    for (a, b), ged in zip(pairs, geds, strict=True):
        if a not in encoded or b not in encoded:
            continue
        if not metric.is_defined(encoded[a], encoded[b]):
            undefined += 1
            continue
        distances.append(float(metric.distance(encoded[a], encoded[b])))
        values.append(ged)
        positions.append((draw.position[a], draw.position[b]))

    if len(distances) < 3:
        record = _empty_record(metric_name, "fewer than three defined pairs")
        record["n_undefined"] = undefined
        return record

    # A rank correlation against a constant column is undefined, and scipy
    # returns NaN for it rather than raising.  NaN is not valid JSON, so it
    # would either abort the write or -- worse -- reach track C as the token
    # `NaN` and be read as a number.  The size null hits this **by
    # construction** in the equal-n view, where |n1 - n2| is 0 for every
    # pair: that is the whole point of the view, and it has to be printed as
    # the stated absence it is rather than as a correlation of nothing.
    constant = _constant_reason(distances, values, metric_name)
    if constant is not None:
        record = _empty_record(metric_name, constant)
        record["n_pairs"] = len(distances)
        record["n_undefined"] = undefined
        record["zero_frac"] = sum(d == 0.0 for d in distances) / len(distances)
        return record

    rho, p = spearman(distances, values)
    ci = graph_bootstrap_ci(distances, values, positions, index)
    return {
        "metric": metric_name,
        "rho": rho,
        "p": p,
        "ci": None if ci is None else [ci[0], ci[1]],
        "n_pairs": len(distances),
        "n_undefined": undefined,
        "zero_frac": sum(d == 0.0 for d in distances) / len(distances),
        "reason": None,
    }


def dataset_record(
    draw: Draw,
    reference: str,
    primary: dict[str, Any],
    reasons: dict[str, str],
    scope_reasons: dict[str, str],
    encodings: dict[str, dict[int, Any]],
    unencodable: dict[str, int | None],
    index: ResampleIndex,
) -> dict[str, Any]:
    """The full record for one (dataset, reference): both views, every row."""
    pairs, geds = reference_pairs(draw, reference)
    views: dict[str, Any] = {}
    for view in VIEWS:
        mask = _view_mask(draw, pairs, view)
        view_pairs = [pair for pair, keep in zip(pairs, mask, strict=True) if keep]
        view_geds = [ged for ged, keep in zip(geds, mask, strict=True) if keep]
        row: dict[str, Any] = {
            SIZE_NULL: rho_record(
                SIZE_NULL, SIZE_NULL, draw, encodings[SIZE_NULL], view_pairs, view_geds, index
            )
        }
        for backend_name in sorted(primary):
            metric_name = primary[backend_name]
            if metric_name is None:
                row[backend_name] = _empty_record(
                    None, reasons.get(backend_name, NO_DISTANCE_REASON)
                )
                continue
            scope = scope_reasons.get(backend_name)
            if scope is not None:
                row[backend_name] = _empty_record(str(metric_name), scope)
                continue
            row[backend_name] = rho_record(
                backend_name,
                str(metric_name),
                draw,
                encodings[backend_name],
                view_pairs,
                view_geds,
                index,
            )
        views[view] = row

    return {
        "dataset": draw.dataset,
        "suite": draw.suite,
        "reference": reference,
        "n_graphs": len(draw.indices),
        "n_unencodable": unencodable,
        "n_reference_pairs": len(pairs),
        "views": views,
    }


def _primary_map(grid: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """The grid's selection, widened to every available representation.

    A representation the grid never mentions is emitted as a printed absence
    rather than dropped, so a silently shrinking grid cannot silently shrink
    the F5 table with it -- and the two absences are told apart: the grid
    found no admissible distance, or the grid never mentioned it at all.

    Returns:
        ``(primary, reasons)`` -- the widened selection, and the reason to
        print for each representation whose distance is ``None``.
    """
    declared = dict(grid.get("primary_distance") or {})
    primary = dict(declared)
    for name in available_backends():
        primary.setdefault(name, None)
    reasons = {
        name: (NO_DISTANCE_REASON if name in declared else ABSENT_REASON)
        for name, value in primary.items()
        if value is None
    }
    return primary, reasons


def _scope_reasons(primary: dict[str, Any], suite: str) -> dict[str, str]:
    """Representations that may not contribute a printed row on *suite*.

    ``agm_cam`` and ``isalgraph_canonical`` both declare
    :attr:`~isalgraph.competitors.base.Capability.SUITE1_ONLY` and both raise
    ``SuiteScopeError`` above n = 12.  On a Suite-2 cohort they therefore
    encode only the graphs that happen to be Suite-1-sized, and a rho over
    those is a rho over the *small* half of the cohort wearing the Suite-2
    label.  Measured on a 40-graph GREC draw: 11 of 40 graphs are refused, so
    the surviving pair set is 406 of 780 -- and the missing 374 are exactly
    the pairs the representation finds hardest.

    ``base.table_scope_error`` is the frozen rule for this (design criteria 5
    and 9): running the backend to measure its ceiling is allowed, printing a
    row from whichever graphs finished is not.  The row is emitted with the
    reason instead of the number.
    """
    out: dict[str, str] = {}
    for name in primary:
        if primary[name] is None:
            continue
        reason = table_scope_error(get_backend(name).capabilities, suite, name)
        if reason is not None:
            out[name] = reason
    return out


def run(
    grid_path: str,
    *,
    names: tuple[str, ...],
    n_graphs: int = DEFAULT_N_GRAPHS,
    seed: int = DEFAULT_SEED,
    resamples: int = DEFAULT_RESAMPLES,
) -> dict[str, Any]:
    """Compute the whole F5 table.  Returns the payload of CONTRACTS §5."""
    with open(grid_path, encoding="utf-8") as handle:
        grid = json.load(handle)
    primary, reasons = _primary_map(grid)

    payload: dict[str, Any] = {
        "protocol": "T-04a-F5",
        "note": "DESCRIPTIVE. F5 is not an input to distance selection.",
        "seed": seed,
        "n_graphs": n_graphs,
        "bootstrap_resamples": resamples,
        "primary_distance_source": os.path.abspath(grid_path),
        "primary_distance": primary,
        "results": {},
    }

    for dataset in names:
        draw = draw_for(dataset, n_graphs, seed)
        scope_reasons = _scope_reasons(primary, draw.suite)
        encodings: dict[str, dict[int, Any]] = {}
        unencodable: dict[str, int | None] = {}
        errors: dict[str, dict[str, int]] = {}
        for backend_name in [SIZE_NULL, *sorted(primary)]:
            skipped = backend_name != SIZE_NULL and (
                primary.get(backend_name) is None or backend_name in scope_reasons
            )
            if skipped:
                # `None`, not 0.  Zero would assert that every graph encoded;
                # this representation was never attempted on this dataset.
                encodings[backend_name] = {}
                unencodable[backend_name] = None
                continue
            encoded, failed, failures = encode_draw(backend_name, draw)
            encodings[backend_name] = encoded
            unencodable[backend_name] = failed
            if failures:
                errors[backend_name] = failures

        index = make_resample_index(len(draw.indices), resamples=resamples, seed=seed)
        references = ("exact",) if draw.suite == "suite1" else ("lb", "ub")
        for reference in references:
            key = dataset if reference == "exact" else f"{dataset}::{reference}"
            record = dataset_record(
                draw, reference, primary, reasons, scope_reasons, encodings, unencodable, index
            )
            record["encode_errors"] = errors
            payload["results"][key] = record

    return payload


def _print_summary(payload: dict[str, Any]) -> None:
    for key, record in payload["results"].items():
        for view, row in record["views"].items():
            null = row[SIZE_NULL]["rho"]
            head = f"{key} [{view}] null="
            print(f"\n=== {head}{'n/a' if null is None else f'{null:.4f}'} ===")
            for name, cell in row.items():
                if name == SIZE_NULL:
                    continue
                if cell["rho"] is None:
                    print(f"  {name:20s} ---   {cell['reason']}")
                    continue
                gap = "" if null is None else f"   vs null {cell['rho'] - null:+.4f}"
                ci = cell["ci"]
                span = "" if ci is None else f"  CI [{ci[0]:.4f}, {ci[1]:.4f}]"
                print(f"  {name:20s} {cell['rho']:>8.4f}{gap}{span}  n={cell['n_pairs']}")


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(prog="python -m isalgraph.competitors.f5")
    parser.add_argument("--grid", required=True, help="grid.json from isalgraph.competitors.grid")
    parser.add_argument("--out", required=True)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--n-graphs",
        type=int,
        default=DEFAULT_N_GRAPHS,
        help="per-dataset draw; the frozen protocol is 200 and the flag exists for "
        "reduced demonstration runs",
    )
    parser.add_argument(
        "--datasets",
        default="",
        help="comma-separated subset; the default is every cohort present under the root",
    )
    args = parser.parse_args(argv)

    if args.datasets.strip():
        names = tuple(d.strip() for d in args.datasets.split(",") if d.strip())
    else:
        names = datasets.available_datasets()

    started = time.perf_counter()
    payload = run(
        args.grid,
        names=names,
        n_graphs=args.n_graphs,
        seed=args.seed,
        resamples=args.resamples,
    )
    payload["wall_seconds"] = time.perf_counter() - started

    _print_summary(payload)
    with open(args.out, "w", encoding="utf-8") as handle:
        # allow_nan=False: NaN and Infinity are not JSON (RFC 8259).  Python
        # writes them anyway by default, and a downstream strict parser then
        # either fails far from the cause or reads `NaN` as a number.  Fail
        # here instead.
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    print(f"\nwrote {args.out} in {payload['wall_seconds']:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
