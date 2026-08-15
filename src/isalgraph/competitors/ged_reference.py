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

**There is no exact-GED reference above n = 12.**  Suite 2 has no F5 at all
until T-05's bounds land, and asking for one raises rather than returning a
partial matrix.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy.typing as npt

from isalgraph.competitors.datasets import SUITE1, cohort_root, load

#: Subdirectory of the cohort root holding the certified matrices.
GED_SUBDIR = os.path.join("source", "GED_PRECOMPUTED", "extended_merged_exact_ged", "computed")

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


__all__ = ["COST_MODEL", "GEDMatrix", "GEDReferenceError", "load_ged"]
