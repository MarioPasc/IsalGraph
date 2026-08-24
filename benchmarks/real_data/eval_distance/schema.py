"""The two ``.npz`` schemas this track reads and writes, and their provenance.

Reads CONTRACTS §3 (encodings, produced by the encoding track), writes
CONTRACTS §4 (distance matrices, consumed by the statistics track), and stamps
CONTRACTS §5 metadata onto everything.

``isalgraph_build_hash`` and ``src_commit`` are stamped unconditionally.  They
are the only way to detect afterwards that a run picked up another branch's
``src/``, which in a git worktree is the default rather than the exception: the
``scikit-build-core`` editable finder is path-pinned to the main checkout and
outranks ``PYTHONPATH``.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "t06.1"
TICKET = "T-06"
WAVE = "2026-08-16-t06-recompute"
SEED = 42
ENCODE_BUDGET_S = 300.0

#: The checkout the ``scikit-build-core`` editable finder is pinned to.  Its
#: HEAD is what actually produced every number, whatever branch the caller is
#: standing on.
MAIN_CHECKOUT = Path("/home/mpascual/research/code/IsalGraph")

#: CONTRACTS §3, in order, as amended 2026-08-16 (``error_kind``, and
#: ``encoding`` holding a ``metadata.symbol_sep``-joined **symbol sequence**).
ENCODINGS_KEYS: tuple[str, ...] = (
    "graph_ids",
    "node_counts",
    "edge_counts",
    "encoding",
    "length",
    "error_kind",
    "entropy_bits",
    "realised_bits",
    "status",
    "fallback_used",
    "seconds",
    "metadata",
)

#: CONTRACTS §4, in order.  A dense file has exactly these keys.
DENSE_KEYS: tuple[str, ...] = (
    "distance_matrix",
    "graph_ids",
    "node_counts",
    "defined_mask",
    "metadata",
)

#: A shard adds the band bounds.  It is an intermediate, not a deliverable.
SHARD_KEYS: tuple[str, ...] = (
    "distance_band",
    "defined_band",
    "row_start",
    "row_stop",
    "n_graphs",
    "graph_ids",
    "node_counts",
    "metadata",
)

#: CONTRACTS §5, in order.
METADATA_KEYS: tuple[str, ...] = (
    "schema_version",
    "ticket",
    "wave",
    "generated_utc",
    "seed",
    "suite",
    "dataset",
    "representation",
    "metric",
    "n_graphs",
    "isalgraph_engine",
    "isalgraph_build_hash",
    "code_commit",
    "src_commit",
    "encode_budget_s",
    "notes",
)


class DistanceError(Exception):
    """Base class for every fault this package raises."""


class SchemaError(DistanceError):
    """A file does not conform to the schema it claims."""


class ShardError(DistanceError):
    """A shard set is incomplete, overlapping, or internally inconsistent."""


class MetricUnsupportedError(DistanceError):
    """The metric cannot be computed from what CONTRACTS §3 carries."""


@dataclass(frozen=True, slots=True)
class EncodingsFile:
    """One ``(suite, dataset, representation)`` encodings file, CONTRACTS §3.

    Attributes:
        path: where it came from.
        graph_ids: cohort order, ``<U16``.
        node_counts: carried through from the cohort, ``int32``.
        edge_counts: carried through from the cohort, ``int32``.
        encoding: the symbol sequence joined by ``metadata.symbol_sep``,
            ``''`` when ``status != "ok"``.
        length: **the symbol count**, ``-1`` when not encoded.  Ground truth
            for sequence length; ``len(encoding)`` is not, once the separator
            is non-empty.
        error_kind: exception class name when ``status == "error"``, else
            ``''``.
        entropy_bits: ``L log2 |Sigma|``, ``nan`` when undefined.
        realised_bits: format-defined byte length x 8, ``nan`` when undefined.
        status: ``ok`` | ``censored`` | ``fallback`` | ``error``.
        fallback_used: ``True`` iff the D14 greedy-min string was substituted.
        seconds: per-graph encode time, ``-1`` when killed.
        metadata: the parsed CONTRACTS §5 block.
    """

    path: Path
    graph_ids: np.ndarray
    node_counts: np.ndarray
    edge_counts: np.ndarray
    encoding: np.ndarray
    length: np.ndarray
    error_kind: np.ndarray
    entropy_bits: np.ndarray
    realised_bits: np.ndarray
    status: np.ndarray
    fallback_used: np.ndarray
    seconds: np.ndarray
    metadata: dict[str, Any]

    @property
    def n_graphs(self) -> int:
        """Number of graphs in cohort order."""
        return int(self.graph_ids.shape[0])


@dataclass(frozen=True, slots=True)
class DenseDistance:
    """One ``(suite, dataset, representation, metric)`` matrix, CONTRACTS §4."""

    path: Path
    distance_matrix: np.ndarray
    graph_ids: np.ndarray
    node_counts: np.ndarray
    defined_mask: np.ndarray
    metadata: dict[str, Any]

    @property
    def n_graphs(self) -> int:
        """Side length of the square matrix."""
        return int(self.graph_ids.shape[0])


@dataclass(frozen=True, slots=True)
class DistanceShard:
    """One contiguous row band of a distance matrix."""

    path: Path
    distance_band: np.ndarray
    defined_band: np.ndarray
    row_start: int
    row_stop: int
    n_graphs: int
    graph_ids: np.ndarray
    node_counts: np.ndarray
    metadata: dict[str, Any]


def git_head(repo: Path) -> str:
    """Return ``git rev-parse HEAD`` for *repo*, or ``"unknown"``.

    Args:
        repo: any path inside the working tree.

    Returns:
        The 40-character SHA, or ``"unknown"`` when git is unavailable or the
        path is not a repository.  A missing SHA degrades the provenance
        record; it must not abort a campaign that is otherwise sound.
    """
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30.0,
        )
    except (OSError, subprocess.SubprocessError):
        logger.warning("could not read git HEAD for %s", repo)
        return "unknown"
    return out.stdout.strip() or "unknown"


def engine_fields() -> tuple[str, str]:
    """Return ``(isalgraph_engine, isalgraph_build_hash)`` of the live package.

    Returns:
        The active engine name and the build hash of the extension that is
        actually importable, which in a worktree is the main checkout's.
    """
    import isalgraph

    engine = str(isalgraph.engine())
    try:
        build_hash = str(isalgraph.build_info().get("build_hash", "unknown"))
    except Exception:  # noqa: BLE001 - a pure-Python install has no build info
        build_hash = "unknown"
    return engine, build_hash


def build_metadata(
    *,
    suite: str,
    dataset: str,
    representation: str | None,
    metric: str | None,
    n_graphs: int,
    code_commit: str | None = None,
    notes: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the CONTRACTS §5 block.

    Args:
        suite: ``"suite1"`` or ``"suite2"``.
        dataset: cohort key.
        representation: backend name, or ``None`` for a derived array.
        metric: metric name, or ``None``.
        n_graphs: cohort size.
        code_commit: HEAD of the checkout producing this file.  Resolved from
            this module's own location when omitted.
        notes: free text.
        extra: additional keys appended after the frozen ones.

    Returns:
        A JSON-serialisable dict carrying every CONTRACTS §5 key.
    """
    engine, build_hash = engine_fields()
    here = Path(__file__).resolve().parent
    meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "ticket": TICKET,
        "wave": WAVE,
        "generated_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "suite": suite,
        "dataset": dataset,
        "representation": representation,
        "metric": metric,
        "n_graphs": int(n_graphs),
        "isalgraph_engine": engine,
        "isalgraph_build_hash": build_hash,
        "code_commit": code_commit if code_commit is not None else git_head(here),
        "src_commit": git_head(MAIN_CHECKOUT),
        "encode_budget_s": ENCODE_BUDGET_S,
        "notes": notes,
    }
    if extra:
        meta.update(extra)
    return meta


def _decode_metadata(raw: np.ndarray) -> dict[str, Any]:
    """Parse a 0-d ``<U...`` JSON array into a dict.

    Raises:
        SchemaError: when the payload is not a JSON object.
    """
    try:
        parsed = json.loads(str(raw))
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"metadata is not JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SchemaError(f"metadata parsed to {type(parsed).__name__}, expected an object")
    return parsed


def _encode_metadata(meta: dict[str, Any]) -> np.ndarray:
    """Render *meta* as the 0-d ``<U...`` JSON array the schema stores."""
    return np.array(json.dumps(meta, sort_keys=False))


def _require_keys(path: Path, present: set[str], expected: tuple[str, ...], what: str) -> None:
    """Raise unless *present* is exactly *expected*.

    Raises:
        SchemaError: on any missing or extra key.
    """
    missing = sorted(set(expected) - present)
    extra = sorted(present - set(expected))
    if missing or extra:
        raise SchemaError(
            f"{path} is not a {what}: missing={missing} unexpected={extra}; "
            f"expected exactly {list(expected)}"
        )


def load_encodings(path: Path) -> EncodingsFile:
    """Load a CONTRACTS §3 encodings file.

    Args:
        path: the ``.npz``.

    Returns:
        The parsed file.

    Raises:
        SchemaError: on a missing key or a length disagreement between arrays.
    """
    with np.load(path, allow_pickle=False) as handle:
        present = set(handle.files)
        _require_keys(path, present, ENCODINGS_KEYS, "CONTRACTS §3 encodings file")
        arrays = {key: handle[key] for key in ENCODINGS_KEYS if key != "metadata"}
        metadata = _decode_metadata(handle["metadata"])
    n = int(arrays["graph_ids"].shape[0])
    bad = {k: v.shape for k, v in arrays.items() if v.shape != (n,)}
    if bad:
        raise SchemaError(f"{path}: arrays disagree with graph_ids length {n}: {bad}")
    return EncodingsFile(path=path, metadata=metadata, **arrays)


def write_dense(
    path: Path,
    *,
    distance_matrix: np.ndarray,
    graph_ids: np.ndarray,
    node_counts: np.ndarray,
    defined_mask: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Write a CONTRACTS §4 dense distance file.

    Args:
        path: destination ``.npz``; parents are created.
        distance_matrix: ``float64 (G, G)``.
        graph_ids: ``<U16 (G,)``, cohort order.
        node_counts: ``int32 (G,)``.
        defined_mask: ``bool (G, G)``.
        metadata: CONTRACTS §5 block.

    Raises:
        SchemaError: on a shape or dtype violation.
    """
    n = int(graph_ids.shape[0])
    if distance_matrix.shape != (n, n) or defined_mask.shape != (n, n):
        raise SchemaError(
            f"{path}: distance_matrix {distance_matrix.shape} and defined_mask "
            f"{defined_mask.shape} must both be ({n}, {n})"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        distance_matrix=np.asarray(distance_matrix, dtype=np.float64),
        graph_ids=np.asarray(graph_ids),
        node_counts=np.asarray(node_counts, dtype=np.int32),
        defined_mask=np.asarray(defined_mask, dtype=bool),
        metadata=_encode_metadata(metadata),
    )
    logger.info("wrote dense distance matrix %s (%d x %d)", path, n, n)


def load_dense(path: Path) -> DenseDistance:
    """Load a CONTRACTS §4 dense distance file.

    Raises:
        SchemaError: on a missing or unexpected key.
    """
    with np.load(path, allow_pickle=False) as handle:
        _require_keys(path, set(handle.files), DENSE_KEYS, "CONTRACTS §4 distance file")
        payload = {key: handle[key] for key in DENSE_KEYS if key != "metadata"}
        metadata = _decode_metadata(handle["metadata"])
    return DenseDistance(path=path, metadata=metadata, **payload)


def shard_path(out_dir: Path, basename: str, chunk_index: int) -> Path:
    """Return the frozen shard filename ``{basename}.shard{K:04d}.npz``."""
    return out_dir / f"{basename}.shard{chunk_index:04d}.npz"


def write_shard(
    path: Path,
    *,
    distance_band: np.ndarray,
    defined_band: np.ndarray,
    row_start: int,
    row_stop: int,
    n_graphs: int,
    graph_ids: np.ndarray,
    node_counts: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Write one contiguous row band.

    Args:
        path: destination ``.npz``; parents are created.
        distance_band: ``float64 (row_stop - row_start, n_graphs)``.
        defined_band: ``bool``, same shape.
        row_start: first row, inclusive.
        row_stop: last row, exclusive.
        n_graphs: cohort size, i.e. the band's column count.
        graph_ids: the whole cohort's ids, so a shard is self-describing.
        node_counts: the whole cohort's node counts.
        metadata: CONTRACTS §5 block plus ``chunk_index`` and ``n_chunks``.

    Raises:
        SchemaError: on a shape violation.
    """
    height = row_stop - row_start
    if distance_band.shape != (height, n_graphs) or defined_band.shape != (height, n_graphs):
        raise SchemaError(
            f"{path}: band shapes {distance_band.shape}/{defined_band.shape} disagree "
            f"with rows [{row_start}, {row_stop}) over {n_graphs} columns"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        distance_band=np.asarray(distance_band, dtype=np.float64),
        defined_band=np.asarray(defined_band, dtype=bool),
        row_start=np.asarray(row_start, dtype=np.int64),
        row_stop=np.asarray(row_stop, dtype=np.int64),
        n_graphs=np.asarray(n_graphs, dtype=np.int64),
        graph_ids=np.asarray(graph_ids),
        node_counts=np.asarray(node_counts, dtype=np.int32),
        metadata=_encode_metadata(metadata),
    )
    logger.info("wrote shard %s rows [%d, %d) of %d", path, row_start, row_stop, n_graphs)


def load_shard(path: Path) -> DistanceShard:
    """Load one row-band shard.

    Raises:
        SchemaError: on a missing or unexpected key.
    """
    with np.load(path, allow_pickle=False) as handle:
        _require_keys(path, set(handle.files), SHARD_KEYS, "distance shard")
        metadata = _decode_metadata(handle["metadata"])
        return DistanceShard(
            path=path,
            distance_band=handle["distance_band"],
            defined_band=handle["defined_band"],
            row_start=int(handle["row_start"]),
            row_stop=int(handle["row_stop"]),
            n_graphs=int(handle["n_graphs"]),
            graph_ids=handle["graph_ids"],
            node_counts=handle["node_counts"],
            metadata=metadata,
        )


__all__ = [
    "DENSE_KEYS",
    "ENCODINGS_KEYS",
    "ENCODE_BUDGET_S",
    "METADATA_KEYS",
    "SCHEMA_VERSION",
    "SEED",
    "SHARD_KEYS",
    "TICKET",
    "WAVE",
    "DenseDistance",
    "DistanceError",
    "DistanceShard",
    "EncodingsFile",
    "MetricUnsupportedError",
    "SchemaError",
    "ShardError",
    "build_metadata",
    "engine_fields",
    "git_head",
    "load_dense",
    "load_encodings",
    "load_shard",
    "shard_path",
    "write_dense",
    "write_shard",
]
