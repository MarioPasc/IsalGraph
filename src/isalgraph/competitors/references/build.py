"""Build and gate-check the T-28 reference distance matrices.

Loads cohort graph data from the IsalGraph archive, computes the four spectral
reference variants, copies the cached WL kernel matrix, applies structural
gates G3 and G5 from the T-28 design note §8, and writes compliant DENSE_KEYS
NPZ files.

Output layout (all under *out_root*)::

    {suite}/{dataset}__spectral.npz
    {suite}/{dataset}__spectral_comb.npz
    {suite}/{dataset}__spectral_adj.npz
    {suite}/{dataset}__spectral_esd.npz
    {suite}/{dataset}__wl.npz

Each file holds exactly DENSE_KEYS:
    distance_matrix  float64 (G, G)
    graph_ids        <U16    (G,)
    node_counts      int32   (G,)
    defined_mask     bool    (G, G)
    metadata         0-d JSON string

The ``wl`` reference is NOT recomputed.  It is the cached
``{dataset}__wl_subtree__kernel.npz`` matrix, copied byte-for-byte with a fresh
T-28 metadata block.

Design note reference: T-28, §3, §6, §8.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from isalgraph.competitors.references.spectral import (
    SpectralVariant,
    cohort_spectra,
    spectral_distance_matrix,
    spectral_esd_matrix,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "t28.0"
TICKET = "T-28"
WAVE = "2026-08-29-t28-metrics"

#: All reference keys produced by this builder, in canonical order.
REF_KEYS: tuple[str, ...] = (
    "spectral",
    "spectral_comb",
    "spectral_adj",
    "spectral_esd",
    "wl",
)

#: Map from reference key to the SpectralVariant used (ESD always uses "norm").
_SPECTRAL_VARIANT: dict[str, SpectralVariant] = {
    "spectral": "norm",
    "spectral_comb": "comb",
    "spectral_adj": "adj",
}

#: Suite-to-archive sub-directory mapping.
_SUITE_DIR: dict[str, str] = {
    "suite1": "exported",
    "suite2": "exported_suite2",
}

#: WL cache path template relative to archive root.
_WL_TEMPLATE = "data/source/T06/distances/{suite}/{dataset}__wl_subtree__kernel.npz"


# ---------------------------------------------------------------------------
# Gate error
# ---------------------------------------------------------------------------


class GateError(RuntimeError):
    """A structural gate (G3 or G5) failed for a reference matrix."""


# ---------------------------------------------------------------------------
# Cohort loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CohortData:
    """Raw graph data for one (suite, dataset) cohort."""

    graph_ids: npt.NDArray[np.str_]
    n_nodes: npt.NDArray[np.int32]
    edge_offsets: npt.NDArray[np.int64]
    edges: npt.NDArray[np.int32]


def load_cohort(archive_root: Path, suite: str, dataset: str) -> CohortData:
    """Load cohort graph topology from the archive.

    Args:
        archive_root: Root of the IsalGraph archive.
        suite: ``"suite1"`` or ``"suite2"``.
        dataset: Dataset key (e.g. ``"aids"``).

    Returns:
        :class:`CohortData` with topology arrays.

    Raises:
        FileNotFoundError: If the NPZ does not exist.
        KeyError: If *suite* is not recognised.
    """
    sub = _SUITE_DIR[suite]
    path = archive_root / "data" / sub / f"{dataset}.npz"
    with np.load(path, allow_pickle=False) as handle:
        return CohortData(
            graph_ids=handle["graph_ids"],
            n_nodes=handle["n_nodes"].astype(np.int32),
            edge_offsets=handle["edge_offsets"].astype(np.int64),
            edges=handle["edges"].astype(np.int32),
        )


# ---------------------------------------------------------------------------
# Metadata construction
# ---------------------------------------------------------------------------


def _git_head(repo: Path) -> str:
    """Return HEAD SHA for *repo*, or ``"unknown"``."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],  # noqa: S603
            capture_output=True,
            text=True,
            check=True,
            timeout=30.0,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _build_metadata(
    *,
    suite: str,
    dataset: str,
    reference: str,
    variant: str,
    n_graphs: int,
    n_max: int | None,
    off_diag_zero_fraction: float,
    code_commit: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the T-28 metadata block for one reference matrix.

    Args:
        suite: ``"suite1"`` or ``"suite2"``.
        dataset: Dataset key.
        reference: Reference key (``"spectral"``, ``"wl"``, …).
        variant: Spectral variant or ``"wl_kernel"`` for the WL copy.
        n_graphs: Cohort size.
        n_max: Maximum node count used for zero-padding, or ``None`` for ESD.
        off_diag_zero_fraction: Fraction of off-diagonal entries that are exactly
            zero (gate G5 measurement).
        code_commit: HEAD SHA of the producing checkout.
        extra: Optional additional keys appended after the frozen block.

    Returns:
        JSON-serialisable metadata dict.
    """
    meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "ticket": TICKET,
        "wave": WAVE,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "suite": suite,
        "dataset": dataset,
        "reference": reference,
        "variant": variant,
        "n_graphs": n_graphs,
        "n_max": n_max,
        "off_diag_zero_fraction": off_diag_zero_fraction,
        "code_commit": code_commit,
    }
    if extra:
        meta.update(extra)
    return meta


# ---------------------------------------------------------------------------
# Gate checks
# ---------------------------------------------------------------------------


def _off_diag_zero_fraction(dist: npt.NDArray[np.float64]) -> float:
    """Return the fraction of off-diagonal entries that are exactly zero."""
    n = dist.shape[0]
    n_off = n * (n - 1)
    if n_off == 0:
        return 0.0
    mask = ~np.eye(n, dtype=bool)
    return float(np.sum(dist[mask] == 0.0)) / n_off


def gate_check(
    distance_matrix: npt.NDArray[np.float64],
    graph_ids: npt.NDArray[np.str_],
    cohort_ids: npt.NDArray[np.str_],
    label: str,
) -> float:
    """Apply structural gates G3 and G5 from the T-28 design note §8.

    Gates:
        G3: Symmetric, zero-diagonal, finite, non-negative, graph_ids match cohort.
        G5: Off-diagonal exact-zero fraction < 0.99.

    Args:
        distance_matrix: The (G, G) distance matrix to check.
        graph_ids: IDs in the reference matrix.
        cohort_ids: IDs from the source cohort (must agree exactly with
            *graph_ids*).
        label: Human-readable identifier for error messages.

    Returns:
        Measured off-diagonal exact-zero fraction (for recording in metadata).

    Raises:
        GateError: On any gate violation.
    """
    n = distance_matrix.shape[0]

    # G3a: symmetric (after our own symmetrisation, this is exact)
    if not np.allclose(distance_matrix, distance_matrix.T, atol=1e-12, rtol=0.0):
        raise GateError(f"{label}: G3 — not symmetric")

    # G3b: zero diagonal
    diag = np.diag(distance_matrix)
    if not np.allclose(diag, 0.0, atol=1e-12):
        raise GateError(f"{label}: G3 — non-zero diagonal (max={diag.max():.3e})")

    # G3c: finite
    if not np.all(np.isfinite(distance_matrix)):
        n_bad = int(np.sum(~np.isfinite(distance_matrix)))
        raise GateError(f"{label}: G3 — {n_bad} non-finite entries")

    # G3d: non-negative
    min_val = float(distance_matrix.min())
    if min_val < 0.0:
        raise GateError(f"{label}: G3 — negative entries (min={min_val:.3e})")

    # G3e: graph_ids join
    if not np.array_equal(graph_ids, cohort_ids):
        raise GateError(
            f"{label}: G3 — graph_ids mismatch (first diff at index "
            f"{int(np.argmax(graph_ids != cohort_ids))})"
        )

    # G5: off-diagonal zero fraction
    frac = _off_diag_zero_fraction(distance_matrix)
    if n > 1 and frac >= 0.99:
        raise GateError(
            f"{label}: G5 — off-diagonal zero fraction {frac:.4f} >= 0.99 "
            "(silent-zero failure)"
        )

    logger.info("%s gates passed (off-diag zero fraction %.4f)", label, frac)
    return frac


# ---------------------------------------------------------------------------
# NPZ writing
# ---------------------------------------------------------------------------


def write_reference_npz(
    path: Path,
    *,
    distance_matrix: npt.NDArray[np.float64],
    graph_ids: npt.NDArray[np.str_],
    node_counts: npt.NDArray[np.int32],
    defined_mask: npt.NDArray[np.bool_],
    metadata: dict[str, Any],
) -> None:
    """Write a CONTRACTS §4 dense distance NPZ.

    Args:
        path: Destination ``.npz``; parents are created.
        distance_matrix: float64 (G, G).
        graph_ids: <U16 (G,).
        node_counts: int32 (G,).
        defined_mask: bool (G, G).
        metadata: Metadata dict; stored as a 0-d JSON string.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        distance_matrix=np.asarray(distance_matrix, dtype=np.float64),
        graph_ids=np.asarray(graph_ids),
        node_counts=np.asarray(node_counts, dtype=np.int32),
        defined_mask=np.asarray(defined_mask, dtype=bool),
        metadata=np.array(json.dumps(metadata, sort_keys=False)),
    )
    logger.info("wrote %s (%d x %d)", path.name, distance_matrix.shape[0], distance_matrix.shape[0])


# ---------------------------------------------------------------------------
# Per-reference builders
# ---------------------------------------------------------------------------


def _build_spectral(
    cohort: CohortData,
    ref_key: str,
    *,
    suite: str,
    dataset: str,
    code_commit: str,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.str_],
    npt.NDArray[np.int32],
    npt.NDArray[np.bool_],
    dict[str, Any],
]:
    """Compute one Euclidean spectral reference matrix.

    Args:
        cohort: Cohort topology data.
        ref_key: One of ``"spectral"``, ``"spectral_comb"``, ``"spectral_adj"``.
        suite: Suite identifier.
        dataset: Dataset key.
        code_commit: HEAD SHA.

    Returns:
        Tuple of (distance_matrix, graph_ids, node_counts, defined_mask, metadata).
    """
    variant = _SPECTRAL_VARIANT[ref_key]
    spectra = cohort_spectra(
        cohort.n_nodes, cohort.edge_offsets, cohort.edges, variant=variant
    )
    n_max = spectra.shape[1]
    dist = spectral_distance_matrix(spectra)
    del spectra  # free padded spectrum memory

    g = cohort.graph_ids
    n_graphs = dist.shape[0]
    defined = np.ones((n_graphs, n_graphs), dtype=bool)

    frac = gate_check(dist, g, g, f"{suite}/{dataset}/{ref_key}")

    meta = _build_metadata(
        suite=suite,
        dataset=dataset,
        reference=ref_key,
        variant=variant,
        n_graphs=n_graphs,
        n_max=n_max,
        off_diag_zero_fraction=frac,
        code_commit=code_commit,
    )
    return dist, g, cohort.n_nodes, defined, meta


def _build_spectral_esd(
    cohort: CohortData,
    *,
    suite: str,
    dataset: str,
    code_commit: str,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.str_],
    npt.NDArray[np.int32],
    npt.NDArray[np.bool_],
    dict[str, Any],
]:
    """Compute the 1-Wasserstein ESD reference matrix.

    Args:
        cohort: Cohort topology data.
        suite: Suite identifier.
        dataset: Dataset key.
        code_commit: HEAD SHA.

    Returns:
        Tuple of (distance_matrix, graph_ids, node_counts, defined_mask, metadata).
    """
    dist = spectral_esd_matrix(
        cohort.n_nodes, cohort.edge_offsets, cohort.edges
    )
    g = cohort.graph_ids
    n_graphs = dist.shape[0]
    defined = np.ones((n_graphs, n_graphs), dtype=bool)

    frac = gate_check(dist, g, g, f"{suite}/{dataset}/spectral_esd")

    meta = _build_metadata(
        suite=suite,
        dataset=dataset,
        reference="spectral_esd",
        variant="norm_wasserstein",
        n_graphs=n_graphs,
        n_max=None,
        off_diag_zero_fraction=frac,
        code_commit=code_commit,
    )
    return dist, g, cohort.n_nodes, defined, meta


def _build_wl(
    archive_root: Path,
    cohort: CohortData,
    *,
    suite: str,
    dataset: str,
    code_commit: str,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.str_],
    npt.NDArray[np.int32],
    npt.NDArray[np.bool_],
    dict[str, Any],
]:
    """Load the cached WL kernel matrix and produce a T-28 reference copy.

    The WL matrix is NOT recomputed — it is the
    ``{dataset}__wl_subtree__kernel.npz`` from the T-06 cache, which guarantees
    the degeneracy check (ρ(wl, wl_subtree) == 1.0 exactly) is preserved.

    Args:
        archive_root: Root of the archive.
        cohort: Cohort data (used for graph_ids join verification).
        suite: Suite identifier.
        dataset: Dataset key.
        code_commit: HEAD SHA.

    Returns:
        Tuple of (distance_matrix, graph_ids, node_counts, defined_mask, metadata).
    """
    wl_path = archive_root / _WL_TEMPLATE.format(suite=suite, dataset=dataset)
    with np.load(wl_path, allow_pickle=False) as handle:
        dist: npt.NDArray[np.float64] = handle["distance_matrix"].astype(np.float64)
        wl_ids: npt.NDArray[np.str_] = handle["graph_ids"]
        node_counts_wl: npt.NDArray[np.int32] = handle["node_counts"].astype(np.int32)
        defined: npt.NDArray[np.bool_] = handle["defined_mask"].astype(bool)
        wl_meta_raw: str = str(handle["metadata"])

    n_graphs = dist.shape[0]
    frac = gate_check(dist, wl_ids, cohort.graph_ids, f"{suite}/{dataset}/wl")

    meta = _build_metadata(
        suite=suite,
        dataset=dataset,
        reference="wl",
        variant="wl_kernel",
        n_graphs=n_graphs,
        n_max=None,
        off_diag_zero_fraction=frac,
        code_commit=code_commit,
        extra={"wl_source_metadata": wl_meta_raw},
    )
    return dist, wl_ids, node_counts_wl, defined, meta


# ---------------------------------------------------------------------------
# Cell builder
# ---------------------------------------------------------------------------


def build_cell(
    suite: str,
    dataset: str,
    archive_root: Path,
    out_root: Path,
) -> dict[str, bool]:
    """Build all five reference matrices for one (suite, dataset) cell.

    Processes reference keys in :data:`REF_KEYS` order.  Each key either
    succeeds (NPZ written, gate-checked) or fails (exception caught, status
    recorded as ``False``).

    Args:
        suite: ``"suite1"`` or ``"suite2"``.
        dataset: Dataset key.
        archive_root: Root of the IsalGraph archive.
        out_root: Root of the T-28 reference output tree.

    Returns:
        Dict mapping each reference key to ``True`` (success) or ``False``
        (failure).  Check the log for details on failures.
    """
    code_commit = _git_head(Path(__file__).resolve())
    status: dict[str, bool] = {}

    logger.info("loading cohort %s/%s", suite, dataset)
    try:
        cohort = load_cohort(archive_root, suite, dataset)
    except Exception:
        logger.exception("failed to load cohort %s/%s", suite, dataset)
        return {k: False for k in REF_KEYS}

    cell_dir = out_root / suite
    cell_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Spectral Euclidean variants
    # ------------------------------------------------------------------
    for ref_key in ("spectral", "spectral_comb", "spectral_adj"):
        out_path = cell_dir / f"{dataset}__{ref_key}.npz"
        try:
            dist, gids, nc, dmask, meta = _build_spectral(
                cohort, ref_key, suite=suite, dataset=dataset, code_commit=code_commit
            )
            write_reference_npz(
                out_path,
                distance_matrix=dist,
                graph_ids=gids,
                node_counts=nc,
                defined_mask=dmask,
                metadata=meta,
            )
            del dist
            status[ref_key] = True
        except Exception:
            logger.exception("%s/%s/%s failed", suite, dataset, ref_key)
            status[ref_key] = False

    # ------------------------------------------------------------------
    # Spectral ESD (1-Wasserstein)
    # ------------------------------------------------------------------
    out_path = cell_dir / f"{dataset}__spectral_esd.npz"
    try:
        dist, gids, nc, dmask, meta = _build_spectral_esd(
            cohort, suite=suite, dataset=dataset, code_commit=code_commit
        )
        write_reference_npz(
            out_path,
            distance_matrix=dist,
            graph_ids=gids,
            node_counts=nc,
            defined_mask=dmask,
            metadata=meta,
        )
        del dist
        status["spectral_esd"] = True
    except Exception:
        logger.exception("%s/%s/spectral_esd failed", suite, dataset)
        status["spectral_esd"] = False

    # ------------------------------------------------------------------
    # WL copy
    # ------------------------------------------------------------------
    out_path = cell_dir / f"{dataset}__wl.npz"
    try:
        dist, gids, nc, dmask, meta = _build_wl(
            archive_root, cohort, suite=suite, dataset=dataset, code_commit=code_commit
        )
        write_reference_npz(
            out_path,
            distance_matrix=dist,
            graph_ids=gids,
            node_counts=nc,
            defined_mask=dmask,
            metadata=meta,
        )
        del dist
        status["wl"] = True
    except Exception:
        logger.exception("%s/%s/wl failed", suite, dataset)
        status["wl"] = False

    return status
