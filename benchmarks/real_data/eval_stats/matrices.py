"""Read-only loaders for the GED and distance matrices the family consumes.

Two schemas, deliberately identical in the keys this module reads so that one
loader serves both: T-05's GED matrices (``ged_matrix``/``lb_matrix``/
``ub_matrix`` plus ``certified_mask``) and T-06's distance matrices
(``distance_matrix`` plus ``defined_mask``, CONTRACTS.md section 4).

Three traps this module exists to absorb:

* **Censored entries carry ``inf``, never ``nan``.** Filtering is by
  ``np.isfinite``, and selection by ``certified_mask``.
* **GED is legitimately 0** for isomorphic graphs --- 28.05 % of IAM Letter LOW
  pairs are certified exact at 0. Nothing here asserts a positive distance.
* **Joins are on ``graph_ids``, never positional.** Suite 1's ``aids`` has 769
  graphs and Suite 2's ``aids_graphedx`` has 819. ``graph_ids`` dtype varies
  across datasets (``<U8`` on ``iam_letter_low``, ``<U16`` on ``linux``,
  ``<U10`` on ``protein``), so every identifier is widened to ``<U16`` before
  any set operation.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from benchmarks.real_data.eval_stats.resampling import BoolArray, FloatArray

LOGGER = logging.getLogger(__name__)

#: Every graph identifier is widened to this before comparison or intersection.
ID_DTYPE = "<U16"

__all__ = ["ID_DTYPE", "MatrixBundle", "MatrixError", "align", "load_matrix", "normalise_ids"]


class MatrixError(Exception):
    """Raised when a matrix file does not satisfy the schema it claims."""


def normalise_ids(graph_ids: npt.NDArray[Any]) -> npt.NDArray[np.str_]:
    """Widen graph identifiers to a common dtype.

    Element-wise comparison handles mixed ``<U`` widths, but ``np.intersect1d``
    and friends can surprise, so every identifier is widened before any set
    operation.

    Args:
        graph_ids: Identifiers of any ``<U`` width.

    Returns:
        The same identifiers as ``<U16``.
    """
    return np.asarray(graph_ids, dtype=ID_DTYPE)


@dataclass(frozen=True)
class MatrixBundle:
    """One square matrix with the per-graph arrays needed to analyse it.

    Attributes:
        key: Dataset key.
        matrix: Square ``(G, G)`` float64 matrix.
        graph_ids: ``(G,)`` identifiers, widened to ``<U16``.
        node_counts: ``(G,)`` node counts, carried through from the cohort.
        edge_counts: ``(G,)`` edge counts, carried through from the cohort.
        mask: ``(G, G)`` usability mask --- ``certified_mask`` for a GED file,
            ``defined_mask`` for a distance file, all-``True`` when absent.
        metadata: The file's decoded ``metadata`` JSON, treated as open for
            extension.
        source: The file the bundle came from.
    """

    key: str
    matrix: FloatArray
    graph_ids: npt.NDArray[np.str_]
    node_counts: npt.NDArray[np.int32]
    edge_counts: npt.NDArray[np.int32]
    mask: BoolArray
    metadata: dict[str, Any]
    source: Path

    @property
    def n_graphs(self) -> int:
        """Number of graphs."""
        return int(self.matrix.shape[0])

    def take(self, positions: npt.NDArray[np.int64]) -> MatrixBundle:
        """Return the sub-bundle on *positions*, in the order given.

        Args:
            positions: Row indices into this bundle.

        Returns:
            A bundle over the selected graphs.
        """
        idx = np.asarray(positions, dtype=np.int64)
        return MatrixBundle(
            key=self.key,
            matrix=np.ascontiguousarray(self.matrix[np.ix_(idx, idx)]),
            graph_ids=self.graph_ids[idx],
            node_counts=self.node_counts[idx],
            edge_counts=self.edge_counts[idx],
            mask=np.ascontiguousarray(self.mask[np.ix_(idx, idx)]),
            metadata=self.metadata,
            source=self.source,
        )


def _pick(archive: Mapping[str, Any], candidates: tuple[str, ...], path: Path) -> str:
    """Return the first present key among *candidates*."""
    for name in candidates:
        if name in archive:
            return name
    raise MatrixError(f"{path} has none of {candidates}; keys are {sorted(archive)}")


def load_matrix(
    path: Path | str,
    *,
    value_key: str | None = None,
    mask_key: str | None = None,
) -> MatrixBundle:
    """Load one matrix ``.npz``, GED or distance.

    Args:
        path: The file.
        value_key: The matrix key. Defaults to the first of ``distance_matrix``,
            ``ged_matrix``, ``lb_matrix``, ``ub_matrix`` that is present.
        mask_key: The usability mask key. Defaults to the first of
            ``defined_mask``, ``certified_mask`` that is present, or all-``True``.

    Returns:
        The bundle.

    Raises:
        MatrixError: If the file is missing a required key or is not square.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise MatrixError(f"no such matrix file: {file_path}")
    with np.load(file_path, allow_pickle=False) as archive:
        keys = set(archive.files)
        chosen = value_key or _pick(
            archive, ("distance_matrix", "ged_matrix", "lb_matrix", "ub_matrix"), file_path
        )
        if chosen not in keys:
            raise MatrixError(f"{file_path} has no {chosen!r}; keys are {sorted(keys)}")
        matrix = np.ascontiguousarray(archive[chosen], dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise MatrixError(f"{file_path}:{chosen} is not square, shape {matrix.shape}")
        n = int(matrix.shape[0])

        selected_mask = mask_key
        if selected_mask is None:
            selected_mask = next((k for k in ("defined_mask", "certified_mask") if k in keys), None)
        mask = (
            np.ascontiguousarray(archive[selected_mask], dtype=bool)
            if selected_mask
            else np.ones((n, n), dtype=bool)
        )

        graph_ids = (
            normalise_ids(archive["graph_ids"]) if "graph_ids" in keys else _synthetic_ids(n)
        )
        node_counts = _optional_counts(archive, "node_counts", n)
        edge_counts = _optional_counts(archive, "edge_counts", n)
        metadata = _decode_metadata(archive, keys, file_path)

    for name, array in (("graph_ids", graph_ids), ("node_counts", node_counts)):
        if array.shape != (n,):
            raise MatrixError(f"{file_path}:{name} has shape {array.shape}, expected ({n},)")
    return MatrixBundle(
        key=file_path.stem,
        matrix=matrix,
        graph_ids=graph_ids,
        node_counts=node_counts,
        edge_counts=edge_counts,
        mask=mask,
        metadata=metadata,
        source=file_path,
    )


def _synthetic_ids(n: int) -> npt.NDArray[np.str_]:
    """Return positional identifiers for a file that carries none."""
    LOGGER.warning("matrix file has no graph_ids; falling back to positional identifiers")
    return normalise_ids(np.array([f"idx{i:011d}" for i in range(n)]))


def _optional_counts(archive: Mapping[str, Any], name: str, n: int) -> npt.NDArray[np.int32]:
    """Return an int32 per-graph array, zeros when absent."""
    if name not in archive:
        return np.zeros(n, dtype=np.int32)
    return np.ascontiguousarray(archive[name], dtype=np.int32)


def _decode_metadata(archive: Mapping[str, Any], keys: set[str], path: Path) -> dict[str, Any]:
    """Decode the 0-d JSON ``metadata`` field, tolerating extra keys."""
    if "metadata" not in keys:
        return {}
    try:
        decoded = json.loads(str(archive["metadata"]))
    except (TypeError, ValueError):
        LOGGER.warning("%s carries unparseable metadata; continuing with an empty mapping", path)
        return {}
    return dict(decoded) if isinstance(decoded, dict) else {}


def align(*bundles: MatrixBundle) -> tuple[MatrixBundle, ...]:
    """Restrict every bundle to the graphs they all share, joined on identifier.

    Positional alignment is never assumed: Suite 1 applies ``n_max = 12`` and
    therefore holds a different cohort from Suite 2 even where the dataset name
    matches.

    Args:
        *bundles: Two or more bundles.

    Returns:
        The bundles restricted to the shared identifiers, in a common order.

    Raises:
        MatrixError: If fewer than two bundles are given, if any carries
            duplicate identifiers, or if the intersection is empty.
    """
    if len(bundles) < 2:
        raise MatrixError("align needs at least two bundles")
    shared: npt.NDArray[np.str_] | None = None
    for bundle in bundles:
        unique = np.unique(bundle.graph_ids)
        if unique.size != bundle.graph_ids.size:
            raise MatrixError(f"{bundle.source} carries duplicate graph_ids")
        shared = unique if shared is None else np.intersect1d(shared, unique, assume_unique=True)
    assert shared is not None
    if shared.size == 0:
        raise MatrixError("the bundles share no graph identifier")

    aligned: list[MatrixBundle] = []
    for bundle in bundles:
        order = np.argsort(bundle.graph_ids, kind="stable")
        positions = order[np.searchsorted(bundle.graph_ids[order], shared)]
        aligned.append(bundle.take(positions.astype(np.int64)))
    return tuple(aligned)
