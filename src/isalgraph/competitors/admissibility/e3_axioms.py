"""``python -m isalgraph.competitors.admissibility.e3_axioms`` -- E3.

The metric axioms, **exhaustively rather than by sampling**.

The T-04a grid checks F2 on 5,000 random triples drawn per cell.  That is a
sampling statement, and a sampling statement about a property that is either
true or false everywhere is the weakest form the evidence can take.  Here the
sweep is complete over a finite universe: every ordered triple of the 142
connected graphs on ``n <= 6``
(:func:`~isalgraph.competitors.admissibility.common.connected_atlas`), for
every (representation x its selected primary distance) cell.

**The expected violation count is exactly zero, and that is the point.**
Levenshtein is provably a metric on strings (Wagner & Fischer, *J. ACM*
21(1):168-173, 1974); the WL kernel distance is the RKHS distance
``||phi(G) - phi(H)||`` induced by a positive semi-definite linear kernel and
therefore a pseudometric, whose triangle inequality is the norm triangle
inequality.  So this module is a **correctness check on our implementation**,
not a discovery.  A nonzero count is a bug: the payload is written first, so
the evidence survives, and then :class:`~isalgraph.competitors.admissibility.
common.AdmissibilityError` is raised rather than a "finding" reported.

Three implementation notes that matter for the number to mean anything.

**Symmetry is measured, not assumed.**  ``d(a, b)`` and ``d(b, a)`` are two
separate calls into the metric.  Filling the lower triangle by mirroring the
upper one would make the symmetry check vacuous -- it would test ``numpy``.

**The triangle sweep is a loop over the apex, not a rank-3 tensor.**  For each
``c`` the whole ``(a, b)`` slab is checked at once, so the working set is
``O(n^2)`` rather than the 22.9 MB an ``n^3`` broadcast would allocate at
``n = 142``, and the same code runs unchanged at ``n = 200``.

**The tolerance is relative.**  Levenshtein returns exact integers and needs
none.  The kernel distance is ``sqrt`` of an exactly-representable integer, so
its error is bounded by half an ulp; :data:`RTOL` is nine orders of magnitude
above that, and far below any real violation, which would be at least one edit
unit.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import TYPE_CHECKING, Any

from isalgraph.competitors.admissibility import common
from isalgraph.competitors.base import VectorBackend
from isalgraph.competitors.registry import get_backend, get_metric
from isalgraph.errors import CompetitorError

if TYPE_CHECKING:
    import networkx as nx
    import numpy.typing as npt

#: Absolute slack on every axiom comparison.
ATOL = 1e-9
#: Relative slack, scaled by the magnitude of the right-hand side.  A genuine
#: violation of any distance in this pool is at least one edit unit; this is
#: nine orders of magnitude below that and nine above float ``sqrt`` error.
RTOL = 1e-9


def _atlas_profile(graphs: list[nx.Graph]) -> dict[str, Any]:
    """Node-count histogram of the atlas, so a caller can assert it is intact.

    Args:
        graphs: the atlas, as returned by :func:`common.connected_atlas`.

    Returns:
        ``n_graphs`` and ``by_n``, the OEIS A001349 counts keyed by node count
        as strings so the record round-trips through JSON unchanged.
    """
    by_n: dict[str, int] = {}
    for graph in graphs:
        key = str(graph.number_of_nodes())
        by_n[key] = by_n.get(key, 0) + 1
    return {"n_graphs": len(graphs), "by_n": by_n}


def encode_atlas(
    backend_name: str, graphs: list[nx.Graph]
) -> tuple[list[int], list[Any], dict[str, int]]:
    """Encode every atlas graph once under *backend_name*.

    A :class:`VectorBackend` is fitted over the **whole atlas** before any
    graph is transformed, matching the per-dataset fit
    ``competitors.metrics.kernel`` documents: a per-batch vocabulary would
    make the distance matrix depend on batching order, and here the "dataset"
    is the atlas.

    Args:
        backend_name: registry key.
        graphs: the atlas.

    Returns:
        ``(kept, encodings, errors)`` -- the positions that encoded, their
        encodings in the same order, and a count per exception type.  A graph
        the backend refuses is dropped from this cell and counted, never
        substituted.
    """
    backend = get_backend(backend_name)
    if isinstance(backend, VectorBackend):
        backend.fit(graphs)

    kept: list[int] = []
    encodings: list[Any] = []
    errors: dict[str, int] = {}
    for position, graph in enumerate(graphs):
        try:
            if isinstance(backend, VectorBackend):
                encodings.append(backend.features(graph))
            else:
                encodings.append(backend.encode(graph))
        except CompetitorError as exc:
            name = type(exc).__name__
            errors[name] = errors.get(name, 0) + 1
            continue
        kept.append(position)
    return kept, encodings, errors


def distance_matrix(metric_name: str, encodings: list[Any]) -> tuple[npt.NDArray[Any], int]:
    """The full ``n x n`` distance matrix, **both triangles computed**.

    ``d(a, b)`` and ``d(b, a)`` are separate calls, and the diagonal is a real
    ``d(a, a)`` call.  Mirroring instead would make :func:`check_symmetry` and
    :func:`check_identity` test ``numpy`` rather than the metric.

    Args:
        metric_name: registry key.
        encodings: one encoding per graph.

    Returns:
        ``(matrix, n_undefined)``.  A pair the metric declares undefined is
        written as ``inf``, which propagates into every axiom check as a
        detectable absence rather than as a fabricated zero.
    """
    import numpy as np

    metric = get_metric(metric_name)
    n = len(encodings)
    matrix = np.zeros((n, n), dtype=np.float64)
    undefined = 0
    for i in range(n):
        for j in range(n):
            if not metric.is_defined(encodings[i], encodings[j]):
                matrix[i, j] = math.inf
                undefined += 1
                continue
            matrix[i, j] = float(metric.distance(encodings[i], encodings[j]))
    return matrix, undefined


def check_identity(matrix: npt.NDArray[Any]) -> dict[str, Any]:
    """``d(a, a) = 0`` over every graph.

    Args:
        matrix: the full distance matrix.

    Returns:
        The count of checks, the count of violations, the worst absolute
        diagonal entry, and -- when no violation is seen -- the rule-of-three
        upper bound rather than a bare ``0``.
    """
    import numpy as np

    diagonal = np.diag(matrix)
    bad = np.abs(diagonal) > ATOL
    return _axiom_record(
        n_checks=int(diagonal.size),
        n_violations=int(bad.sum()),
        worst=float(np.abs(diagonal).max()) if diagonal.size else 0.0,
        worst_key="max_abs_self_distance",
    )


def check_symmetry(matrix: npt.NDArray[Any]) -> dict[str, Any]:
    """``d(a, b) = d(b, a)`` over every unordered pair.

    Args:
        matrix: the full distance matrix, with both triangles independently
            computed.

    Returns:
        The same record shape as :func:`check_identity`, over ``C(n, 2)``
        checks.
    """
    import numpy as np

    n = matrix.shape[0]
    upper_i, upper_j = np.triu_indices(n, 1)
    gap = np.abs(matrix[upper_i, upper_j] - matrix[upper_j, upper_i])
    scale = np.abs(matrix[upper_i, upper_j]) + np.abs(matrix[upper_j, upper_i])
    bad = gap > ATOL + RTOL * scale
    return _axiom_record(
        n_checks=int(gap.size),
        n_violations=int(bad.sum()),
        worst=float(gap.max()) if gap.size else 0.0,
        worst_key="max_asymmetry",
    )


def check_triangle(matrix: npt.NDArray[Any]) -> dict[str, Any]:
    """``d(a, b) <= d(a, c) + d(c, b)`` over **every** triple.

    One inequality per (unordered pair, distinct apex), which is exactly three
    per unordered triple: ``C(n, 2) * (n - 2) == 3 * C(n, 3)``.  Both counts
    are reported, because the protocol names the triple count and the
    rule-of-three bound is a statement about the number of *opportunities*.

    Args:
        matrix: the full distance matrix.

    Returns:
        ``n_triples``, ``n_checks``, ``n_violations``, ``worst_excess`` (the
        largest ``d(a,b) - d(a,c) - d(c,b)``, which must be ``<= 0``), and the
        rule-of-three bounds when nothing fires.
    """
    import numpy as np

    n = matrix.shape[0]
    rows = np.arange(n)[:, None]
    cols = np.arange(n)[None, :]
    upper = rows < cols

    violations = 0
    worst = -math.inf
    for c in range(n):
        # bound[a, b] = d(a, c) + d(c, b); the apex is excluded from the pair.
        bound = matrix[:, c][:, None] + matrix[c, :][None, :]
        valid = upper & (rows != c) & (cols != c)
        excess = matrix - bound
        violations += int((valid & (excess > ATOL + RTOL * np.abs(bound))).sum())
        if valid.any():
            worst = max(worst, float(excess[valid].max()))

    record = _axiom_record(
        n_checks=math.comb(n, 2) * max(n - 2, 0),
        n_violations=violations,
        worst=worst if math.isfinite(worst) else 0.0,
        worst_key="worst_excess",
    )
    record["n_triples"] = math.comb(n, 3)
    if violations == 0:
        record["rule_of_three_upper_per_triple"] = common.rule_of_three(math.comb(n, 3))
    return record


def _axiom_record(
    *, n_checks: int, n_violations: int, worst: float, worst_key: str
) -> dict[str, Any]:
    """One axiom's record.  Zero violations is never reported as a bare ``0``.

    Args:
        n_checks: trials performed.
        n_violations: trials that failed.
        worst: the extremal residual, for the diagnostic column.
        worst_key: what to call it in the record.

    Returns:
        The record.  ``rate`` is ``None`` at zero violations, where
        ``rule_of_three_upper`` carries what the sample actually licenses --
        printing ``0`` would assert impossibility from a finite sweep, even
        an exhaustive one over a *sub*-universe.
    """
    zero = n_violations == 0
    return {
        "n_checks": n_checks,
        "n_violations": n_violations,
        "rate": None if zero else n_violations / n_checks,
        "rule_of_three_upper": common.rule_of_three(n_checks) if zero else None,
        "clopper_pearson_95": list(common.clopper_pearson(n_violations, n_checks)),
        worst_key: worst,
    }


def cell_record(backend_name: str, metric_name: str, graphs: list[nx.Graph]) -> dict[str, Any]:
    """All three axioms for one (representation, distance) cell.

    Args:
        backend_name: registry key of the representation.
        metric_name: registry key of its selected primary distance.
        graphs: the atlas.

    Returns:
        The cell's record, including the encode failures that shrank it.
    """
    started = time.perf_counter()
    kept, encodings, errors = encode_atlas(backend_name, graphs)
    matrix, undefined = distance_matrix(metric_name, encodings)
    metric = get_metric(metric_name)
    return {
        "backend": backend_name,
        "metric": metric_name,
        "declared_pseudometric": bool(metric.is_pseudometric),
        "n_graphs": len(kept),
        "n_unencodable": len(graphs) - len(kept),
        "encode_errors": errors,
        "n_undefined_pairs": undefined,
        "identity": check_identity(matrix),
        "symmetry": check_symmetry(matrix),
        "triangle": check_triangle(matrix),
        "wall_seconds": time.perf_counter() - started,
    }


def violating_cells(payload: dict[str, Any]) -> list[str]:
    """Names of the cells that failed any axiom.

    Args:
        payload: a completed :func:`run` payload.

    Returns:
        ``"<backend>/<metric>/<axiom>"`` for each failure, empty when clean.
    """
    out: list[str] = []
    for key, cell in payload["cells"].items():
        for axiom in ("identity", "symmetry", "triangle"):
            if cell[axiom]["n_violations"]:
                out.append(f"{key}/{axiom}")
    return out


def run(grid_path: str, *, max_n: int = common.EXHAUSTIVE_N_TRIPLES) -> dict[str, Any]:
    """Run E3 over every cell the grid selected a primary distance for.

    Args:
        grid_path: path to ``grid_200.json``.
        max_n: largest node count in the atlas.  The frozen value is
            :data:`common.EXHAUSTIVE_N_TRIPLES`; the argument exists so the
            test suite can sweep a nine-graph universe in milliseconds.

    Returns:
        The payload, with one entry per cell under ``cells``.
    """
    graphs = common.connected_atlas(max_n)
    primary = common.primary_distances(grid_path)
    cells: dict[str, Any] = {}
    excluded: dict[str, str] = {}
    for backend_name in sorted(primary):
        metric_name = primary[backend_name]
        if metric_name is None:
            excluded[backend_name] = (
                "no admissible distance under competitors.md 3.4; there is no "
                "distance to check the axioms of"
            )
            continue
        cells[f"{backend_name}/{metric_name}"] = cell_record(backend_name, metric_name, graphs)

    return {
        "protocol_section": "4",
        "grid": grid_path,
        "max_n": max_n,
        "atlas": _atlas_profile(graphs),
        "n_triples": math.comb(len(graphs), 3),
        "n_triangle_checks": math.comb(len(graphs), 2) * max(len(graphs) - 2, 0),
        "atol": ATOL,
        "rtol": RTOL,
        "excluded": excluded,
        "cells": cells,
    }


def _print_summary(payload: dict[str, Any]) -> None:
    atlas = payload["atlas"]
    print(
        f"atlas: {atlas['n_graphs']} connected graphs on n <= {payload['max_n']} "
        f"({atlas['by_n']}); {payload['n_triples']:,} triples, "
        f"{payload['n_triangle_checks']:,} triangle checks per cell"
    )
    for key, cell in payload["cells"].items():
        parts = []
        for axiom in ("identity", "symmetry", "triangle"):
            record = cell[axiom]
            if record["n_violations"]:
                parts.append(f"{axiom}={record['n_violations']} VIOLATIONS")
            else:
                parts.append(
                    f"{axiom}=0/{record['n_checks']:,} (<= {record['rule_of_three_upper']:.2e})"
                )
        flag = " [pseudometric]" if cell["declared_pseudometric"] else ""
        print(f"  {key:36s}{flag}  " + "  ".join(parts))


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: command line, or ``None`` for ``sys.argv``.

    Returns:
        ``0``.

    Raises:
        AdmissibilityError: when any axiom fails.  The payload is written
            **before** the raise, so the evidence survives the abort: a
            violation of the triangle inequality by Levenshtein is a defect in
            our encoding or in our metric, and the protocol escalates it
            rather than reporting it as a property of the method.
    """
    parser = argparse.ArgumentParser(prog="python -m isalgraph.competitors.admissibility.e3_axioms")
    parser.add_argument("--grid", required=True, help="grid_200.json from competitors.grid")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-n", type=int, default=common.EXHAUSTIVE_N_TRIPLES)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    payload = run(args.grid, max_n=args.max_n)
    payload["wall_seconds"] = time.perf_counter() - started

    _print_summary(payload)
    common.write_result(args.out, "E3", payload)
    print(f"\nwrote {args.out} in {payload['wall_seconds']:.1f} s")

    failures = violating_cells(payload)
    if failures:
        raise common.AdmissibilityError(
            f"E3 recorded axiom violations in {len(failures)} cell(s): "
            f"{', '.join(failures)}. Levenshtein is provably a metric and the WL "
            f"kernel distance is a pseudometric, so this is a defect in our "
            f"implementation, not a property of the method. The payload was written "
            f"to {args.out} before this abort"
        )
    return 0


__all__ = [
    "ATOL",
    "RTOL",
    "cell_record",
    "check_identity",
    "check_symmetry",
    "check_triangle",
    "distance_matrix",
    "encode_atlas",
    "main",
    "run",
    "violating_cells",
]


if __name__ == "__main__":
    raise SystemExit(main())
