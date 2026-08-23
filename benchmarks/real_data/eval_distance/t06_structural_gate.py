"""Structural gate over every T-06 distance matrix, plus the graph_ids join.

Acceptance criterion 4: *"Distance matrices join T-05's on ``graph_ids``, and
every matrix is symmetric, zero-diagonal, finite --- 0 violations."*

Two halves, and the second is the one that catches real corruption. The
structural half re-uses :func:`gates.check_dense` rather than reimplementing it,
because two copies of a predicate drifting apart is exactly how an arm and its
null end up measured on different pair sets. The join half checks
``graph_ids`` **element-wise against the reference GED matrix**, never
positionally: ``aids`` is 769 graphs in Suite 1 against ``aids_graphedx``'s 819
in Suite 2 (F-12), so a positional join silently compares different graphs.

A degenerate all-zero matrix is reported but is **not** by itself a failure. GED
is legitimately 0 for isomorphic graphs --- 28.05 % of IAM Letter LOW pairs are
certified with ``LB == UB`` --- so the guard is at the matrix level
(``offdiag_zero_fraction >= 0.99``), never per pair.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from benchmarks.eval_distance import gates

LOGGER: Final = logging.getLogger(__name__)


class StructuralGateError(Exception):
    """Raised when a distance file cannot be read as a CONTRACTS section 4 matrix."""


@dataclass
class MatrixReport:
    """Structural and join outcome for one distance matrix.

    Attributes:
        suite: Suite key.
        dataset: Dataset key.
        representation: Backend name.
        metric: Distance name.
        n_graphs: Matrix side length.
        symmetric: Whether the matrix is symmetric where defined.
        zero_diagonal: Whether every diagonal entry is exactly zero.
        finite_where_defined: Whether no defined entry is non-finite.
        non_negative: Whether no defined entry is negative.
        mask_symmetric: Whether ``defined_mask`` is symmetric.
        max_asymmetry: Largest defined asymmetry.
        offdiag_zero_fraction: Fraction of defined off-diagonal cells at zero.
        degenerate: Whether that fraction reaches the silent-zero threshold.
        join_reference: Which reference the join was checked against.
        join_exact: Whether ``graph_ids`` matched element-wise.
        violations: Human-readable failures; empty means the gate passed.
    """

    suite: str
    dataset: str
    representation: str
    metric: str
    n_graphs: int
    symmetric: bool
    zero_diagonal: bool
    finite_where_defined: bool
    non_negative: bool
    mask_symmetric: bool
    max_asymmetry: float
    offdiag_zero_fraction: float
    degenerate: bool
    join_reference: str | None
    join_exact: bool | None
    violations: list[str] = field(default_factory=list)


def _reference_ids(
    suite: str, dataset: str, ged_root: Path, approx_root: Path
) -> tuple[str, np.ndarray] | None:
    """Return the reference ``graph_ids`` for one dataset.

    Args:
        suite: Suite key.
        dataset: Dataset key.
        ged_root: Suite-1 exact-GED matrices.
        approx_root: ``APPROX_GED`` root.

    Returns:
        ``(reference_name, graph_ids)``, or ``None`` when no reference exists.
    """
    path = (
        ged_root / f"{dataset}.npz" if suite == "suite1" else approx_root / "LB" / f"{dataset}.npz"
    )
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as z:
        return ("T-05 exact" if suite == "suite1" else "T-05 LB"), np.asarray(
            z["graph_ids"]
        ).astype(str)


def check_matrix(path: Path, reference: tuple[str, np.ndarray] | None) -> MatrixReport:
    """Run the structural gate and the join check on one distance matrix.

    Args:
        path: ``{dataset}__{representation}__{metric}.npz``.
        reference: ``(name, graph_ids)`` from :func:`_reference_ids`.

    Returns:
        The report.

    Raises:
        StructuralGateError: If the file lacks a required column.
    """
    stem = path.stem
    parts = stem.split("__")
    if len(parts) != 3:
        raise StructuralGateError(f"{path} is not {{dataset}}__{{representation}}__{{metric}}.npz")
    dataset, representation, metric = parts

    with np.load(path, allow_pickle=True) as z:
        missing = {"distance_matrix", "defined_mask", "graph_ids"} - set(z.files)
        if missing:
            raise StructuralGateError(f"{path} lacks {sorted(missing)}")
        matrix = np.asarray(z["distance_matrix"], dtype=np.float64)
        mask = np.asarray(z["defined_mask"], dtype=bool)
        ids = np.asarray(z["graph_ids"]).astype(str)

    report = gates.check_dense(matrix, mask)
    violations: list[str] = []
    if not report.symmetric:
        violations.append(f"asymmetric, max |d[i,j]-d[j,i]| = {report.max_asymmetry:.6g}")
    if not report.zero_diagonal:
        violations.append("diagonal is not exactly zero")
    if not report.finite_where_defined:
        violations.append("non-finite entry under defined_mask")
    if not report.non_negative:
        violations.append(f"negative entry, min = {report.min_value:.6g}")
    if not report.mask_symmetric:
        violations.append("defined_mask is not symmetric")

    degenerate = report.offdiag_zero_fraction >= gates.DEGENERATE_ZERO_FRACTION
    if degenerate:
        violations.append(
            f"off-diagonal exact-zero fraction {report.offdiag_zero_fraction:.4f} "
            f">= {gates.DEGENERATE_ZERO_FRACTION} -- the silent-zero signature"
        )

    join_name: str | None = None
    join_exact: bool | None = None
    if reference is not None:
        join_name, ref_ids = reference
        join_exact = bool(len(ids) == len(ref_ids) and np.array_equal(ids, ref_ids))
        if not join_exact:
            shared = len(set(ids) & set(ref_ids.tolist()))
            violations.append(
                f"graph_ids do not match {join_name} element-wise "
                f"({len(ids)} vs {len(ref_ids)}, {shared} shared) -- F-12"
            )

    return MatrixReport(
        suite=path.parent.name,
        dataset=dataset,
        representation=representation,
        metric=metric,
        n_graphs=report.n_graphs,
        symmetric=report.symmetric,
        zero_diagonal=report.zero_diagonal,
        finite_where_defined=report.finite_where_defined,
        non_negative=report.non_negative,
        mask_symmetric=report.mask_symmetric,
        max_asymmetry=report.max_asymmetry,
        offdiag_zero_fraction=report.offdiag_zero_fraction,
        degenerate=degenerate,
        join_reference=join_name,
        join_exact=join_exact,
        violations=violations,
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distances", type=Path, required=True, help="the distances/ tree")
    ap.add_argument("--ged-root", type=Path, required=True, help="Suite-1 exact matrices")
    ap.add_argument("--approx-root", type=Path, required=True, help="APPROX_GED root")
    ap.add_argument("--out", type=Path, required=True, help="gate_T06_structural.json")
    return ap


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector, or ``None`` for ``sys.argv``.

    Returns:
        0 when every matrix passes, 1 otherwise.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    reports: list[MatrixReport] = []
    cache: dict[tuple[str, str], tuple[str, np.ndarray] | None] = {}
    for suite in ("suite1", "suite2"):
        directory = args.distances / suite
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.npz")):
            if ".shard" in path.name:
                continue
            dataset = path.stem.split("__")[0]
            key = (suite, dataset)
            if key not in cache:
                cache[key] = _reference_ids(suite, dataset, args.ged_root, args.approx_root)
            reports.append(check_matrix(path, cache[key]))

    failed = [r for r in reports if r.violations]
    payload: dict[str, Any] = {
        "gate": "T-06 structural gate (acceptance criterion 4)",
        "n_matrices": len(reports),
        "n_violations": len(failed),
        "degenerate_zero_threshold": gates.DEGENERATE_ZERO_FRACTION,
        "passed": not failed,
        "matrices": [asdict(r) for r in reports],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    joins = sum(1 for r in reports if r.join_exact)
    no_ref = sum(1 for r in reports if r.join_reference is None)
    print(f"matrices checked        : {len(reports)}")
    print(f"graph_ids join exact    : {joins}  (no reference available: {no_ref})")
    print(f"violations              : {len(failed)}")
    for r in failed[:25]:
        for v in r.violations:
            print(f"  {r.suite}/{r.dataset}/{r.representation}/{r.metric}: {v}")
    print(f"\nGATE: {'PASS' if not failed else 'FAIL'}  -> {args.out}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
