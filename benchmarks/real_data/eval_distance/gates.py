"""The structural gate every dense distance matrix must clear.

Four properties, checked together because each catches a different way of
producing a plausible wrong matrix:

===================  ==========================================================
symmetric            a merge that assembled bands in the wrong order, or a
                     metric that is not symmetric and was assumed to be
zero diagonal        an off-by-one in the band offset
finite where defined ``inf`` or ``nan`` leaking through ``defined_mask``
non-negative         a signed subtraction where a distance was intended
===================  ==========================================================

T-05's equivalent is ``benchmarks/eval_setup/approx_ged_gates.py``.  The one
check deliberately **not** made here is per-pair ``value > 0``: a distance of
exactly 0 is legitimate for two graphs with the same encoding, and 28.05 % of
IAM Letter LOW pairs are certified at GED 0.  The campaign-level analogue --
an all-but-zero matrix -- is :func:`degenerate_zero_fraction`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmarks.eval_distance.schema import SchemaError

#: An off-diagonal exact-zero fraction at or above this is the signature of a
#: silently-empty matrix, not of a cohort full of isomorphic graphs.  Mirrors
#: the T-05 merged-matrix guard.
DEGENERATE_ZERO_FRACTION = 0.99


@dataclass(frozen=True, slots=True)
class GateReport:
    """Outcome of :func:`check_dense`.

    Attributes:
        n_graphs: matrix side length.
        symmetric: ``d == d.T`` where both entries are defined.
        zero_diagonal: every ``d[i, i]`` is exactly 0.
        finite_where_defined: no ``nan``/``inf`` under ``defined_mask``.
        non_negative: no negative entry under ``defined_mask``.
        mask_symmetric: ``defined_mask == defined_mask.T``.
        max_asymmetry: largest ``|d[i, j] - d[j, i]|`` over defined pairs.
        min_value: smallest defined entry.
        n_undefined: number of ``(i, j)`` cells with ``defined_mask`` false.
        offdiag_zero_fraction: fraction of defined off-diagonal cells at 0.
    """

    n_graphs: int
    symmetric: bool
    zero_diagonal: bool
    finite_where_defined: bool
    non_negative: bool
    mask_symmetric: bool
    max_asymmetry: float
    min_value: float
    n_undefined: int
    offdiag_zero_fraction: float

    @property
    def passed(self) -> bool:
        """Whether every structural property holds."""
        return (
            self.symmetric
            and self.zero_diagonal
            and self.finite_where_defined
            and self.non_negative
            and self.mask_symmetric
        )


def _offdiag_zero_fraction(matrix: np.ndarray, defined: np.ndarray) -> float:
    """Fraction of defined off-diagonal cells that are exactly 0."""
    n = matrix.shape[0]
    if n < 2:
        return 0.0
    offdiag = ~np.eye(n, dtype=bool)
    considered = offdiag & defined
    total = int(considered.sum())
    if total == 0:
        return 0.0
    return float(np.count_nonzero(matrix[considered] == 0.0) / total)


def check_dense(distance_matrix: np.ndarray, defined_mask: np.ndarray) -> GateReport:
    """Measure the four structural properties without raising.

    Args:
        distance_matrix: ``float64 (G, G)``.
        defined_mask: ``bool (G, G)``.

    Returns:
        The report.  Use :func:`assert_dense` to turn a failure into an error.

    Raises:
        SchemaError: when the two arrays are not both square and equal-shaped.
    """
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise SchemaError(f"distance_matrix has shape {distance_matrix.shape}, expected square")
    if defined_mask.shape != distance_matrix.shape:
        raise SchemaError(
            f"defined_mask {defined_mask.shape} does not match "
            f"distance_matrix {distance_matrix.shape}"
        )
    n = int(distance_matrix.shape[0])
    both = defined_mask & defined_mask.T
    diff = np.abs(distance_matrix - distance_matrix.T)
    asym = diff[both]
    values = distance_matrix[defined_mask]
    diagonal = np.diagonal(distance_matrix)
    return GateReport(
        n_graphs=n,
        symmetric=bool(asym.size == 0 or np.all(asym == 0.0)),
        zero_diagonal=bool(np.all(diagonal == 0.0)),
        finite_where_defined=bool(values.size == 0 or np.all(np.isfinite(values))),
        non_negative=bool(values.size == 0 or np.all(values >= 0.0)),
        mask_symmetric=bool(np.array_equal(defined_mask, defined_mask.T)),
        max_asymmetry=float(asym.max()) if asym.size else 0.0,
        min_value=float(np.nanmin(values)) if values.size else 0.0,
        n_undefined=int(np.count_nonzero(~defined_mask)),
        offdiag_zero_fraction=_offdiag_zero_fraction(distance_matrix, defined_mask),
    )


def assert_dense(distance_matrix: np.ndarray, defined_mask: np.ndarray) -> GateReport:
    """Run :func:`check_dense` and raise on any failed property.

    Returns:
        The report, when every property holds.

    Raises:
        SchemaError: naming every property that failed, with its measurement.
    """
    report = check_dense(distance_matrix, defined_mask)
    faults: list[str] = []
    if not report.symmetric:
        faults.append(f"not symmetric (max |d[i,j] - d[j,i]| = {report.max_asymmetry!r})")
    if not report.zero_diagonal:
        faults.append("diagonal is not identically zero")
    if not report.finite_where_defined:
        faults.append("non-finite entries under defined_mask")
    if not report.non_negative:
        faults.append(f"negative entries under defined_mask (min = {report.min_value!r})")
    if not report.mask_symmetric:
        faults.append("defined_mask is not symmetric")
    if faults:
        raise SchemaError(f"structural gate failed on a {report.n_graphs}-graph matrix: {faults}")
    return report


def degenerate_zero_fraction(report: GateReport) -> None:
    """Raise when a matrix is almost entirely zero off the diagonal.

    The failure mode is a driver that wrote an unfilled buffer.  It is a
    property of the **matrix**, never of a pair: two graphs with the same
    encoding are legitimately at distance 0, and a per-pair ``value > 0``
    guard aborts a correct run on every such pair.

    Raises:
        SchemaError: when the off-diagonal exact-zero fraction reaches
            :data:`DEGENERATE_ZERO_FRACTION`.
    """
    if report.n_graphs >= 2 and report.offdiag_zero_fraction >= DEGENERATE_ZERO_FRACTION:
        raise SchemaError(
            f"{report.offdiag_zero_fraction:.4f} of defined off-diagonal cells are exactly 0, "
            f"at or above the {DEGENERATE_ZERO_FRACTION} degeneracy threshold; this is the "
            f"shape of an unfilled buffer, not of a cohort of isomorphic graphs"
        )


__all__ = [
    "DEGENERATE_ZERO_FRACTION",
    "GateReport",
    "assert_dense",
    "check_dense",
    "degenerate_zero_fraction",
]
