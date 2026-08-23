"""Certified exact GED for Suite 1.  **Imported only by ``f5.py`` and ``reproduce.py``.**

Kept in its own module so that ``grid.py``'s import closure provably does not
reach it.  ``grid.py`` computes F1, F2, F3, F4 and F6; it has no code path
that reads a GED value and it cannot compute F5.  That is the cheapest way
to make decision 24 defensible to a reviewer: the selection tool could not
have seen the outcome, because it cannot load it.

Adding an import of this module to ``grid.py``, to ``datasets.py``, or to
anything either of them imports **breaks decision 24**.  A test asserts the
closure.

Source: T-03's ``extended_merged_exact_ged/computed/<ds>.npz`` under the D6
unit cost model ``[1,1,0,1,1,0]``.  ``graph_ids`` alignment against the
graph cohort is **asserted, not assumed** -- the two files are produced by
different pipelines and a silent misalignment would corrupt every rho.

**There is no exact-GED reference above n = 12.**  :func:`load_ged` therefore
covers Suite 1 only and raises above it rather than returning a partial
matrix.  Suite 2's *bounds* -- T-05's proven BRANCH_FAST lower bound and
BIPARTITE upper bound over all 21.7 M pairs -- arrive through
:func:`load_bounds`, and the two are **never** averaged into a midpoint
(``approx_ged.md`` §4's no-interpolation rule).
"""

from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import numpy.typing as npt

from isalgraph.competitors.datasets import SUITE1, cohort_root, load

#: Subdirectory of the cohort root holding the certified matrices.
GED_SUBDIR = os.path.join("source", "GED_PRECOMPUTED", "extended_merged_exact_ged", "computed")

#: Subdirectory of the cohort root holding T-05's proven bounds.
BOUNDS_SUBDIR = os.path.join("source", "APPROX_GED")

#: Which side of the bracket a request names, and where each one lives.
Which = Literal["lb", "ub"]
BOUNDS_DIR: dict[str, str] = {"lb": "LB", "ub": "UB"}
BOUNDS_KEY: dict[str, str] = {"lb": "lb_matrix", "ub": "ub_matrix"}

#: The shape of the silent-zero failure T-05 measured: a wrong GEDLIB
#: accessor returns 0.00 without raising, so an entire matrix fills with
#: zeros.  A **per-pair** ``value > 0`` guard is wrong -- GED is legitimately
#: 0 for isomorphic graphs, and 28 % of IAM Letter LOW pairs are certified at
#: exactly that.  The failure is a property of the whole matrix, so the guard
#: is too.
MAX_OFFDIAG_ZERO_FRACTION = 0.99

#: The D6 cost model these matrices were computed under:
#: ``[node_ins, node_del, node_rel, edge_ins, edge_del, edge_rel]``.
COST_MODEL: tuple[int, int, int, int, int, int] = (1, 1, 0, 1, 1, 0)


class GEDReferenceError(RuntimeError):
    """Raised when the certified GED matrix is absent or misaligned."""


@dataclass(frozen=True, slots=True)
class GEDMatrix:
    """A dataset's certified exact GED, with its certification mask."""

    dataset: str
    #: ``ged[i, j]`` -- valid only where ``certified[i, j]``.
    ged: npt.NDArray[Any]
    #: ``True`` where the A* search closed optimally.
    certified: npt.NDArray[Any]

    def certified_pairs(self, indices: tuple[int, ...]) -> list[tuple[int, int]]:
        """Upper-triangle pairs from *indices* whose GED is certified exact."""
        return [
            (a, b) for k, a in enumerate(indices) for b in indices[k + 1 :] if self.certified[a, b]
        ]


@functools.cache
def load_ged(dataset: str) -> GEDMatrix:
    """Load one dataset's certified exact GED.

    Raises:
        GEDReferenceError: if the dataset is not Suite 1, if the file is
            absent, or if ``graph_ids`` do not align with the cohort.
    """
    import numpy as np

    if dataset not in SUITE1:
        raise GEDReferenceError(
            f"{dataset!r} has no exact-GED reference: exact GED exists only for "
            f"Suite 1 (n <= 12). Suite 2 has no F5 until T-05's bounds land"
        )
    path = os.path.join(cohort_root(), GED_SUBDIR, f"{dataset}.npz")
    if not os.path.exists(path):
        raise GEDReferenceError(f"certified exact GED for {dataset!r} not found at {path}")
    data = np.load(path, allow_pickle=True)

    cohort = load(dataset)
    ids = np.asarray(data["graph_ids"]).tolist()
    if ids != list(cohort.graph_ids):
        raise GEDReferenceError(
            f"graph_ids misalign between the {dataset!r} cohort and its GED matrix "
            f"({len(cohort.graph_ids)} vs {len(ids)} entries). The two files come "
            f"from different pipelines; a silent misalignment corrupts every rho"
        )
    return GEDMatrix(dataset=dataset, ged=data["ged_matrix"], certified=data["certified_mask"])


@dataclass(frozen=True, slots=True)
class GEDBounds:
    """One side of T-05's proven GED bracket for one dataset.

    Attributes:
        dataset: provenance, carried into every record.
        which: ``"lb"`` or ``"ub"``.  **Never mixed with the other side**:
            a lower and an upper bound are two measurements, and their
            midpoint is a number nobody computed.
        values: ``values[i, j]``, the bound in the D6 unit cost model.
            Censored pairs are ``inf``, **not** ``NaN``.
        method: the GEDLIB method that produced it, from the file metadata.
        offdiag_zero_fraction: fraction of finite off-diagonal entries that
            are exactly 0.  Recorded rather than merely checked, because it
            is the diagnostic a reviewer would ask for.
    """

    dataset: str
    which: str
    values: npt.NDArray[Any]
    method: str
    offdiag_zero_fraction: float

    def finite_pairs(self, indices: tuple[int, ...]) -> list[tuple[int, int]]:
        """Upper-triangle pairs from *indices* whose bound is finite.

        **The filter is** ``np.isfinite``, **not** ``not np.isnan``.  T-05
        censors an unfinished pair with ``inf``; ``np.isnan(inf)`` is
        ``False``, so an ``isnan`` filter passes every censored pair straight
        through, and ``inf <= x`` is ``False`` without raising anything.

        A bound of exactly 0 is **kept**.  Isomorphic graphs have GED 0 and
        1 % of the Suite-2 cohort attains it; rejecting zeros would discard
        the pairs the encoding gets most obviously right.
        """
        import numpy as np

        return [
            (a, b)
            for k, a in enumerate(indices)
            for b in indices[k + 1 :]
            if np.isfinite(self.values[a, b])
        ]


@functools.cache
def load_bounds(dataset: str, which: str) -> GEDBounds:
    """Load one side of T-05's proven GED bracket.

    Args:
        dataset: a member of ``datasets.ALL_DATASETS``.  Suite-1 names are
            accepted -- the bounds exist for most of them -- but Suite 1's
            F5 arm uses :func:`load_ged`, which is exact.
        which: ``"lb"`` or ``"ub"``.

    Returns:
        The :class:`GEDBounds`.

    Raises:
        GEDReferenceError: if *which* is not a side of the bracket, if the
            file is absent, if ``graph_ids`` do not align with the cohort, if
            any finite entry is negative, or if the off-diagonal exact-zero
            fraction reaches :data:`MAX_OFFDIAG_ZERO_FRACTION`.
    """
    import numpy as np

    if which not in BOUNDS_KEY:
        raise GEDReferenceError(
            f"which={which!r} is not a side of the bracket; expected 'lb' or 'ub'. "
            f"The two sides are reported separately and are never interpolated"
        )
    path = os.path.join(cohort_root(), BOUNDS_SUBDIR, BOUNDS_DIR[which], f"{dataset}.npz")
    if not os.path.exists(path):
        raise GEDReferenceError(f"{which!r} GED bound for {dataset!r} not found at {path}")
    data = np.load(path, allow_pickle=True)

    cohort = load(dataset)
    ids = np.asarray(data["graph_ids"]).tolist()
    if ids != list(cohort.graph_ids):
        raise GEDReferenceError(
            f"graph_ids misalign between the {dataset!r} cohort and its {which!r} bound "
            f"({len(cohort.graph_ids)} vs {len(ids)} entries). The two files come from "
            f"different pipelines; joining them positionally corrupts every rho"
        )

    values = np.asarray(data[BOUNDS_KEY[which]], dtype=np.float64)
    zero_fraction = _validated_offdiag_zero_fraction(values, dataset, which)

    method = ""
    if "metadata" in data.files:
        try:
            method = str(json.loads(str(data["metadata"])).get("method", ""))
        except (ValueError, TypeError, AttributeError):  # provenance only, never load-fatal
            method = ""
    return GEDBounds(
        dataset=dataset,
        which=which,
        values=values,
        method=method,
        offdiag_zero_fraction=zero_fraction,
    )


def _validated_offdiag_zero_fraction(values: npt.NDArray[Any], dataset: str, which: str) -> float:
    """Run T-05's two matrix-level guards and return the zero fraction.

    Raises:
        GEDReferenceError: on a negative finite entry, or on a matrix that
            is almost entirely zero off the diagonal.
    """
    import numpy as np

    finite = np.isfinite(values)
    if bool((values[finite] < 0).any()):
        raise GEDReferenceError(
            f"the {which!r} bound for {dataset!r} holds a negative distance; the D6 unit "
            f"cost model cannot produce one"
        )
    offdiag = ~np.eye(values.shape[0], dtype=bool) & finite
    denominator = int(offdiag.sum())
    if denominator == 0:
        raise GEDReferenceError(
            f"the {which!r} bound for {dataset!r} has no finite off-diagonal entry: every "
            f"pair is censored"
        )
    zero_fraction = float((values[offdiag] == 0.0).sum()) / denominator
    if zero_fraction >= MAX_OFFDIAG_ZERO_FRACTION:
        raise GEDReferenceError(
            f"the {which!r} bound for {dataset!r} is {zero_fraction:.4f} exactly zero off the "
            f"diagonal, at or above the {MAX_OFFDIAG_ZERO_FRACTION} abort threshold. That is "
            f"the shape of GEDLIB's silent wrong-accessor failure, in which get_lower_bound() "
            f"on an upper-bound method returns 0.00 without raising"
        )
    return zero_fraction


__all__ = [
    "BOUNDS_DIR",
    "BOUNDS_KEY",
    "BOUNDS_SUBDIR",
    "COST_MODEL",
    "MAX_OFFDIAG_ZERO_FRACTION",
    "GEDBounds",
    "GEDMatrix",
    "GEDReferenceError",
    "Which",
    "load_bounds",
    "load_ged",
]
