"""Serialize the five Suite-1 datasets into one ``.npz`` each -- CONTRACT A.

Why this exists
---------------
Picasso's ``fscratch`` enforces a **file-count** quota, not only a space quota.
The IAM Letter GXL tree is 6,767 files and must never be transferred. The
GraphEdX graphs live in ``torch`` pickles, and ``torch`` is not installed on the
cluster and must never need to be. Serializing locally to one ``.npz`` per
dataset removes both problems: six files travel instead of ~6,800, and the
cluster-side reader (:func:`load_exported`) imports nothing beyond ``numpy`` and
``networkx``.

The cohort is **locked**. ``filter_graphs(min_nodes=2, require_connected=True,
n_max=12)`` over merged splits must reproduce the counts in
:data:`DATASETS` exactly. A mismatch aborts with a non-zero exit status; it is a
finding to report, never a filter to adjust.

Usage
-----
``python -m benchmarks.real_data.eval_setup.export_graphs --source DIR --out DIR``
``python -m benchmarks.real_data.eval_setup.export_graphs --verify-only``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from math import comb
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_filter import FilterResult, filter_graphs  # noqa: E402

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

#: The locked Suite-1 filter. Never parameterised -- the cohort is frozen.
FILTER_MIN_NODES = 2
FILTER_REQUIRE_CONNECTED = True
FILTER_N_MAX = 12

DEFAULT_SOURCE_DIR = "/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source"
DEFAULT_EXPORT_DIR = "/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/exported"

MANIFEST_NAME = "manifest.json"

_REPO_ROOT = Path(__file__).resolve().parents[3]


class ExportError(Exception):
    """A serialized dataset violates CONTRACT A."""


class CohortMismatchError(ExportError):
    """Observed graph or pair counts differ from the locked cohort."""


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """One Suite-1 dataset and the counts it must reproduce.

    Attributes
    ----------
    key : str
        Export key; the ``.npz`` is written as ``{key}.npz``.
    source : str
        Which loader family reads it, ``"iam_letter"`` or ``"graphedx"``.
    loader_arg : str
        Level (``LOW``/``MED``/``HIGH``) or dataset directory (``LINUX``/``AIDS``).
    expected_kept : int
        Locked graph count after the Suite-1 filter.
    expected_pairs : int
        Locked pair count; always ``C(expected_kept, 2)``.
    """

    key: str
    source: str
    loader_arg: str
    expected_kept: int
    expected_pairs: int


#: The locked cohort (CONTRACTS section 2 / T-03-design section 2).
DATASETS: dict[str, DatasetSpec] = {
    "iam_letter_low": DatasetSpec("iam_letter_low", "iam_letter", "LOW", 1180, 695610),
    "iam_letter_med": DatasetSpec("iam_letter_med", "iam_letter", "MED", 1253, 784378),
    "iam_letter_high": DatasetSpec("iam_letter_high", "iam_letter", "HIGH", 2059, 2118711),
    "linux": DatasetSpec("linux", "graphedx", "LINUX", 89, 3916),
    "aids": DatasetSpec("aids", "graphedx", "AIDS", 769, 295296),
}

TOTAL_EXPECTED_GRAPHS = 5350
TOTAL_EXPECTED_PAIRS = 3897911


@dataclass(frozen=True, slots=True)
class ExportedDataset:
    """One serialized dataset, in kept order.

    Attributes
    ----------
    key : str
        Export key, matching ``metadata["dataset"]``.
    graphs : list[nx.Graph]
        Relabelled ``0..n-1``, undirected, no self-loops, no attributes.
    graph_ids : list[str]
        Stable per-graph identifier from the source loader.
    splits : list[str]
        Origin split, retained because gate 0 needs within-split AIDS pairs and
        cannot recover them once the splits are merged.
    labels : list[str]
        Class label where the source has one, ``""`` otherwise.
    n_nodes : np.ndarray
        ``int32 (N,)`` node count per graph.
    n_edges : np.ndarray
        ``int32 (N,)`` edge count per graph.
    metadata : dict[str, object]
        Provenance and the drop breakdown; schema in CONTRACTS section 4.
    """

    key: str
    graphs: list[nx.Graph]
    graph_ids: list[str]
    splits: list[str]
    labels: list[str]
    n_nodes: np.ndarray
    n_edges: np.ndarray
    metadata: dict[str, object]


# --------------------------------------------------------------------------- #
# Array plumbing
# --------------------------------------------------------------------------- #


def _str_array(values: list[str]) -> np.ndarray:
    """Return a unicode array, with a well-defined dtype when ``values`` is empty.

    ``np.asarray([])`` yields ``float64``, which would break the CONTRACT A
    dtype for a zero-graph dataset.
    """
    if not values:
        return np.empty(0, dtype="<U1")
    return np.asarray(values, dtype=np.str_)


def _graphs_to_csr(graphs: list[nx.Graph]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten graphs into the CONTRACT A CSR layout.

    Parameters
    ----------
    graphs : list[nx.Graph]
        Graphs already relabelled ``0..n-1``.

    Returns
    -------
    tuple of np.ndarray
        ``(n_nodes int32 (N,), n_edges int32 (N,), edge_offsets int64 (N+1,),
        edges int32 (2, M))``. Edges within each graph are sorted ascending by
        ``(u, v)`` so the serialization is a deterministic function of the
        graphs, independent of NetworkX iteration order.
    """
    n = len(graphs)
    n_nodes = np.zeros(n, dtype=np.int32)
    n_edges = np.zeros(n, dtype=np.int32)
    per_graph: list[list[tuple[int, int]]] = []

    for i, g in enumerate(graphs):
        pairs = sorted((min(u, v), max(u, v)) for u, v in g.edges() if u != v)
        n_nodes[i] = g.number_of_nodes()
        n_edges[i] = len(pairs)
        per_graph.append(pairs)

    edge_offsets = np.zeros(n + 1, dtype=np.int64)
    edge_offsets[1:] = np.cumsum(n_edges, dtype=np.int64)

    total = int(edge_offsets[-1]) if n else 0
    edges = np.zeros((2, total), dtype=np.int32)
    cursor = 0
    for pairs in per_graph:
        for u, v in pairs:
            edges[0, cursor] = u
            edges[1, cursor] = v
            cursor += 1

    return n_nodes, n_edges, edge_offsets, edges


def _validate_arrays(
    graph_ids: np.ndarray,
    n_nodes: np.ndarray,
    n_edges: np.ndarray,
    edge_offsets: np.ndarray,
    edges: np.ndarray,
    splits: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Enforce every CONTRACT A array invariant.

    Applied on both write and read, so a truncated or hand-fabricated file
    fails loudly instead of silently producing wrong graphs.

    Raises
    ------
    ExportError
        On any shape, dtype, ordering or range violation.
    """
    n = int(graph_ids.shape[0])
    for name, arr in (("splits", splits), ("labels", labels)):
        if arr.shape != (n,):
            raise ExportError(f"{name} has shape {arr.shape}, expected ({n},)")
    for name, arr in (("n_nodes", n_nodes), ("n_edges", n_edges)):
        if arr.shape != (n,):
            raise ExportError(f"{name} has shape {arr.shape}, expected ({n},)")
        if np.any(arr < 0):
            raise ExportError(f"{name} contains a negative value")

    if edge_offsets.shape != (n + 1,):
        raise ExportError(f"edge_offsets has shape {edge_offsets.shape}, expected ({n + 1},)")
    if int(edge_offsets[0]) != 0:
        raise ExportError(f"edge_offsets[0] == {int(edge_offsets[0])}, expected 0")
    if np.any(np.diff(edge_offsets) < 0):
        raise ExportError("edge_offsets is not monotone non-decreasing")
    if not np.array_equal(np.diff(edge_offsets), n_edges.astype(np.int64)):
        raise ExportError("edge_offsets increments disagree with n_edges")

    total = int(n_edges.sum())
    if int(edge_offsets[-1]) != total:
        raise ExportError(f"edge_offsets[-1] == {int(edge_offsets[-1])}, expected {total}")
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ExportError(f"edges has shape {edges.shape}, expected (2, M)")
    if edges.shape[1] != total:
        raise ExportError(f"edges has {edges.shape[1]} columns, expected {total}")

    if total == 0:
        return

    u = edges[0].astype(np.int64)
    v = edges[1].astype(np.int64)
    if np.any(u >= v):
        raise ExportError("every edge must satisfy u < v (no self-loops, canonical orientation)")
    if np.any(u < 0):
        raise ExportError("edges contains a negative endpoint")

    owner = np.repeat(np.arange(n, dtype=np.int64), n_edges.astype(np.int64))
    bound = n_nodes.astype(np.int64)[owner]
    if np.any(v >= bound):
        raise ExportError("edge endpoint out of range for its graph (need u, v < n_nodes[g])")

    order = np.lexsort((v, u, owner))
    su, sv, so = u[order], v[order], owner[order]
    dup = (so[1:] == so[:-1]) & (su[1:] == su[:-1]) & (sv[1:] == sv[:-1])
    if bool(np.any(dup)):
        raise ExportError("duplicate edge within a graph")


# --------------------------------------------------------------------------- #
# CONTRACT A -- save / load
# --------------------------------------------------------------------------- #


def save_exported(dataset: ExportedDataset, path: str | Path) -> None:
    """Write one dataset to a CONTRACT A ``.npz``.

    Parameters
    ----------
    dataset : ExportedDataset
        Dataset to serialize. ``n_nodes`` and ``n_edges`` must agree with
        ``graphs``; a disagreement is a defect, not something to reconcile.
    path : str or Path
        Destination ``.npz``. Parent directories are created.

    Raises
    ------
    ExportError
        If the dataset is internally inconsistent or violates CONTRACT A.
    """
    n = len(dataset.graphs)
    if not (len(dataset.graph_ids) == len(dataset.splits) == len(dataset.labels) == n):
        raise ExportError(
            "graphs/graph_ids/splits/labels length mismatch: "
            f"{n}/{len(dataset.graph_ids)}/{len(dataset.splits)}/{len(dataset.labels)}"
        )

    n_nodes, n_edges, edge_offsets, edges = _graphs_to_csr(dataset.graphs)
    if not np.array_equal(n_nodes, dataset.n_nodes.astype(np.int32)):
        raise ExportError("dataset.n_nodes disagrees with the node counts of dataset.graphs")
    if not np.array_equal(n_edges, dataset.n_edges.astype(np.int32)):
        raise ExportError("dataset.n_edges disagrees with the edge counts of dataset.graphs")

    graph_ids = _str_array(dataset.graph_ids)
    splits = _str_array(dataset.splits)
    labels = _str_array(dataset.labels)
    _validate_arrays(graph_ids, n_nodes, n_edges, edge_offsets, edges, splits, labels)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        graph_ids=graph_ids,
        n_nodes=n_nodes,
        n_edges=n_edges,
        edge_offsets=edge_offsets,
        edges=edges,
        splits=splits,
        labels=labels,
        metadata=np.array(json.dumps(dataset.metadata, sort_keys=True)),
    )
    logger.info("Wrote %s (%d graphs, %d edges)", out, n, int(edge_offsets[-1]))


def load_exported(path: str | Path) -> ExportedDataset:
    """Read a CONTRACT A ``.npz``.

    Imports nothing beyond ``numpy``, ``networkx`` and the standard library --
    in particular **not** ``torch``, which is absent on Picasso by design.

    Parameters
    ----------
    path : str or Path
        A ``.npz`` written by :func:`save_exported`.

    Returns
    -------
    ExportedDataset
        Graphs rebuilt edge-for-edge, with ids, splits, labels and metadata.

    Raises
    ------
    ExportError
        If the file violates any CONTRACT A invariant.
    """
    with np.load(Path(path), allow_pickle=False) as handle:
        graph_ids_arr = handle["graph_ids"]
        n_nodes = handle["n_nodes"].astype(np.int32, copy=True)
        n_edges = handle["n_edges"].astype(np.int32, copy=True)
        edge_offsets = handle["edge_offsets"].astype(np.int64, copy=True)
        edges = handle["edges"].astype(np.int32, copy=True)
        splits_arr = handle["splits"]
        labels_arr = handle["labels"]
        metadata_raw = handle["metadata"]

    _validate_arrays(graph_ids_arr, n_nodes, n_edges, edge_offsets, edges, splits_arr, labels_arr)

    metadata = json.loads(str(metadata_raw.item()) if metadata_raw.ndim == 0 else str(metadata_raw))
    if not isinstance(metadata, dict):
        raise ExportError("metadata is not a JSON object")

    graphs: list[nx.Graph] = []
    for i in range(int(n_nodes.shape[0])):
        g = nx.Graph()
        g.add_nodes_from(range(int(n_nodes[i])))
        lo, hi = int(edge_offsets[i]), int(edge_offsets[i + 1])
        for k in range(lo, hi):
            g.add_edge(int(edges[0, k]), int(edges[1, k]))
        graphs.append(g)

    return ExportedDataset(
        key=str(metadata.get("dataset", Path(path).stem)),
        graphs=graphs,
        graph_ids=[str(x) for x in graph_ids_arr],
        splits=[str(x) for x in splits_arr],
        labels=[str(x) for x in labels_arr],
        n_nodes=n_nodes,
        n_edges=n_edges,
        metadata=metadata,
    )


# --------------------------------------------------------------------------- #
# Loading and filtering the sources
# --------------------------------------------------------------------------- #


def _normalise(g: nx.Graph) -> nx.Graph:
    """Return an attribute-free undirected copy relabelled ``0..n-1``.

    Node ids are remapped in sorted order, which is the identity on every
    Suite-1 source (all five already label nodes ``0..n-1``) but makes the
    guarantee explicit rather than inherited.
    """
    order = sorted(g.nodes())
    remap = {old: i for i, old in enumerate(order)}
    out = nx.Graph()
    out.add_nodes_from(range(len(order)))
    for src, dst in g.edges():
        a, b = remap[src], remap[dst]
        if a != b:
            out.add_edge(min(a, b), max(a, b))
    return out


def _load_iam_letter_level(
    source_dir: Path, level: str
) -> tuple[list[Any], list[str], list[str], list[str]]:
    """Load one IAM Letter distortion level, splits merged."""
    from iam_letter_loader import load_iam_letter

    base = source_dir / "IAM_Database" / "extracted" / "Letter"
    dataset = load_iam_letter(str(base), level)
    return dataset.graphs, dataset.graph_ids, dataset.splits, dataset.labels


def _load_graphedx(
    source_dir: Path, name: str
) -> tuple[list[Any], list[str], list[str], list[str]]:
    """Load one GraphEdX dataset, splits merged.

    ``load_graphedx_dataset`` does not return a per-graph split list; it appends
    graphs in ``SPLITS`` order and records the sizes in an insertion-ordered
    dict. The split label is reconstructed from those sizes and then
    cross-checked against the split encoded in each ``graph_id``, so a future
    reordering inside the loader fails here instead of silently mislabelling
    gate 0's within-split AIDS pairs.
    """
    from graphedx_loader import load_graphedx_dataset

    base = source_dir / "GED_PRECOMPUTED"
    dataset = load_graphedx_dataset(name, str(base))

    splits: list[str] = []
    for split, size in dataset.split_sizes.items():
        splits.extend([split] * size)
    if len(splits) != len(dataset.graphs):
        raise ExportError(
            f"{name}: split_sizes sum to {len(splits)} but {len(dataset.graphs)} graphs were loaded"
        )

    prefix = name.lower()
    for gid, split in zip(dataset.graph_ids, splits, strict=True):
        if gid.rsplit("_", 1)[0] != f"{prefix}_{split}":
            raise ExportError(f"{name}: graph id {gid!r} contradicts reconstructed split {split!r}")

    return dataset.graphs, dataset.graph_ids, splits, [""] * len(dataset.graphs)


def _load_raw(
    spec: DatasetSpec, source_dir: Path
) -> tuple[list[Any], list[str], list[str], list[str]]:
    """Dispatch to the loader for ``spec``, returning graphs, ids, splits, labels."""
    if spec.source == "iam_letter":
        return _load_iam_letter_level(source_dir, spec.loader_arg)
    if spec.source == "graphedx":
        return _load_graphedx(source_dir, spec.loader_arg)
    raise ExportError(f"unknown source family {spec.source!r} for {spec.key!r}")


def assert_cohort(spec: DatasetSpec, n_kept: int, n_pairs: int) -> None:
    """Abort unless the observed counts equal the locked cohort.

    Raises
    ------
    CohortMismatchError
        With the observed values printed beside the expected ones. The filter is
        never adjusted to make this pass.
    """
    if n_kept == spec.expected_kept and n_pairs == spec.expected_pairs:
        return
    raise CohortMismatchError(
        f"{spec.key}: observed {n_kept} graphs / {n_pairs} pairs, "
        f"expected {spec.expected_kept} graphs / {spec.expected_pairs} pairs"
    )


def _git_commit() -> str:
    """Return the repository HEAD sha, or ``"unknown"`` outside a checkout."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return proc.stdout.strip() or "unknown"


def build_exported(spec: DatasetSpec, source_dir: Path, code_commit: str) -> ExportedDataset:
    """Load, filter and package one dataset, asserting the locked cohort.

    Parameters
    ----------
    spec : DatasetSpec
        Dataset to build.
    source_dir : Path
        Root of the read-only source tree.
    code_commit : str
        Repository sha recorded in the metadata.

    Returns
    -------
    ExportedDataset
        Ready for :func:`save_exported`.

    Raises
    ------
    CohortMismatchError
        If the filtered counts differ from the locked cohort.
    """
    graphs, graph_ids, splits, labels = _load_raw(spec, source_dir)
    result: FilterResult = filter_graphs(
        graphs,
        graph_ids,
        n_max=FILTER_N_MAX,
        require_connected=FILTER_REQUIRE_CONNECTED,
        min_nodes=FILTER_MIN_NODES,
    )
    kept = result.kept_indices
    n_pairs = comb(result.n_kept, 2)
    assert_cohort(spec, result.n_kept, n_pairs)

    kept_graphs = [_normalise(graphs[i]) for i in kept]
    n_nodes = np.asarray([g.number_of_nodes() for g in kept_graphs], dtype=np.int32)
    n_edges = np.asarray([g.number_of_edges() for g in kept_graphs], dtype=np.int32)

    metadata: dict[str, object] = {
        "dataset": spec.key,
        "source": spec.source,
        "n_raw": result.n_raw,
        "n_kept": result.n_kept,
        "n_dropped_size": result.n_dropped_size,
        "n_dropped_disconnected": result.n_dropped_disconnected,
        "n_dropped_trivial": result.n_dropped_trivial,
        "n_pairs": n_pairs,
        "filter": {
            "min_nodes": FILTER_MIN_NODES,
            "require_connected": FILTER_REQUIRE_CONNECTED,
            "n_max": FILTER_N_MAX,
        },
        "splits_merged": True,
        "exported_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_commit": code_commit,
        "schema_version": SCHEMA_VERSION,
    }

    return ExportedDataset(
        key=spec.key,
        graphs=kept_graphs,
        graph_ids=[graph_ids[i] for i in kept],
        splits=[splits[i] for i in kept],
        labels=[labels[i] for i in kept],
        n_nodes=n_nodes,
        n_edges=n_edges,
        metadata=metadata,
    )


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


def _as_int(value: object, field: str) -> int:
    """Narrow a metadata value to ``int``.

    ``metadata`` is typed ``dict[str, object]`` by CONTRACT A, so every numeric
    read out of it needs an explicit narrowing step.

    Raises
    ------
    ExportError
        If the field is not an integer.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExportError(f"metadata field {field!r} is {value!r}, expected an int")
    return value


def sha256_file(path: str | Path) -> str:
    """Return the hex sha256 of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def content_sha256(dataset: ExportedDataset) -> str:
    """Return a sha256 over the array contents, independent of zip framing.

    ``np.savez_compressed`` stamps each zip member with the local time, so the
    file digest changes between two byte-identical exports. This digest is a
    deterministic function of the data alone, which is what "the export
    reproduces" has to mean.
    """
    n_nodes, n_edges, edge_offsets, edges = _graphs_to_csr(dataset.graphs)
    digest = hashlib.sha256()
    digest.update(dataset.key.encode("utf-8"))
    for values in (dataset.graph_ids, dataset.splits, dataset.labels):
        for item in values:
            digest.update(item.encode("utf-8"))
            digest.update(b"\x00")
        digest.update(b"\x01")
    for arr in (n_nodes, n_edges, edge_offsets, edges):
        digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


def write_manifest(entries: dict[str, dict[str, object]], export_dir: str | Path) -> Path:
    """Write ``manifest.json`` and return its path."""
    out = Path(export_dir) / MANIFEST_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Wrote %s (%d datasets)", out, len(entries))
    return out


def read_manifest(export_dir: str | Path) -> dict[str, dict[str, object]]:
    """Read ``manifest.json`` from ``export_dir``."""
    path = Path(export_dir) / MANIFEST_NAME
    if not path.is_file():
        raise ExportError(f"manifest not found: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ExportError(f"manifest is not a JSON object: {path}")
    return loaded


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def export_all(
    source_dir: str | Path, export_dir: str | Path, keys: list[str]
) -> dict[str, dict[str, object]]:
    """Export the requested datasets and write the manifest.

    Parameters
    ----------
    source_dir : str or Path
        Read-only source tree.
    export_dir : str or Path
        Destination directory; created if absent.
    keys : list[str]
        Subset of :data:`DATASETS` to export.

    Returns
    -------
    dict
        The manifest entries just written.

    Raises
    ------
    CohortMismatchError
        On any dataset whose counts differ from the locked cohort.
    """
    src = Path(source_dir)
    dst = Path(export_dir)
    commit = _git_commit()
    entries: dict[str, dict[str, object]] = {}
    total_graphs = 0
    total_pairs = 0

    for key in keys:
        spec = DATASETS[key]
        logger.info("--- %s (%s/%s)", key, spec.source, spec.loader_arg)
        dataset = build_exported(spec, src, commit)
        path = dst / f"{key}.npz"
        save_exported(dataset, path)

        meta = dataset.metadata
        n_kept = _as_int(meta["n_kept"], "n_kept")
        n_pairs = _as_int(meta["n_pairs"], "n_pairs")
        entries[key] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "n_kept": n_kept,
            "n_pairs": n_pairs,
            "content_sha256": content_sha256(dataset),
        }
        total_graphs += n_kept
        total_pairs += n_pairs
        logger.info(
            "    kept %d/%d  pairs %d  dropped size=%d disconnected=%d trivial=%d",
            meta["n_kept"],
            meta["n_raw"],
            meta["n_pairs"],
            meta["n_dropped_size"],
            meta["n_dropped_disconnected"],
            meta["n_dropped_trivial"],
        )

    if set(keys) == set(DATASETS):
        if total_graphs != TOTAL_EXPECTED_GRAPHS or total_pairs != TOTAL_EXPECTED_PAIRS:
            raise CohortMismatchError(
                f"totals: observed {total_graphs} graphs / {total_pairs} pairs, "
                f"expected {TOTAL_EXPECTED_GRAPHS} graphs / {TOTAL_EXPECTED_PAIRS} pairs"
            )
        logger.info("Totals: %d graphs / %d pairs -- match", total_graphs, total_pairs)

    write_manifest(entries, dst)
    return entries


def verify_exports(export_dir: str | Path, keys: list[str]) -> bool:
    """Re-read existing exports and recheck counts and checksums.

    Reports every failure rather than stopping at the first, because a partial
    transfer usually damages more than one file.

    Returns
    -------
    bool
        ``True`` when every requested dataset verified.
    """
    dst = Path(export_dir)
    manifest = read_manifest(dst)
    ok = True
    total_graphs = 0
    total_pairs = 0

    for key in keys:
        spec = DATASETS[key]
        path = dst / f"{key}.npz"
        if not path.is_file():
            logger.error("%s: missing %s", key, path)
            ok = False
            continue

        recorded = manifest.get(key)
        if recorded is None:
            logger.error("%s: absent from %s", key, MANIFEST_NAME)
            ok = False
            continue

        digest = sha256_file(path)
        size = path.stat().st_size
        if digest != recorded.get("sha256"):
            logger.error("%s: sha256 %s, manifest records %s", key, digest, recorded.get("sha256"))
            ok = False
        if size != recorded.get("bytes"):
            logger.error("%s: %d bytes, manifest records %s", key, size, recorded.get("bytes"))
            ok = False

        try:
            dataset = load_exported(path)
        except ExportError:
            logger.exception("%s: failed CONTRACT A validation", key)
            ok = False
            continue

        n_kept = len(dataset.graphs)
        n_pairs = comb(n_kept, 2)
        total_graphs += n_kept
        total_pairs += n_pairs
        try:
            assert_cohort(spec, n_kept, n_pairs)
        except CohortMismatchError as exc:
            logger.error("%s", exc)
            ok = False
            continue

        expected_content = recorded.get("content_sha256")
        if expected_content is not None:
            observed_content = content_sha256(dataset)
            if observed_content != expected_content:
                logger.error(
                    "%s: content_sha256 %s, manifest records %s",
                    key,
                    observed_content,
                    expected_content,
                )
                ok = False

        logger.info("%s: OK  %d graphs / %d pairs  %d bytes", key, n_kept, n_pairs, size)

    if set(keys) == set(DATASETS):
        if total_graphs != TOTAL_EXPECTED_GRAPHS or total_pairs != TOTAL_EXPECTED_PAIRS:
            logger.error(
                "totals: observed %d graphs / %d pairs, expected %d graphs / %d pairs",
                total_graphs,
                total_pairs,
                TOTAL_EXPECTED_GRAPHS,
                TOTAL_EXPECTED_PAIRS,
            )
            ok = False
        else:
            logger.info("Totals: %d graphs / %d pairs -- match", total_graphs, total_pairs)

    return ok


def _parse_keys(value: str) -> list[str]:
    """Resolve ``--datasets`` into an ordered list of dataset keys."""
    if value.strip().lower() == "all":
        return list(DATASETS)
    keys = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [k for k in keys if k not in DATASETS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown dataset key(s): {', '.join(unknown)}")
    return keys


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="export_graphs",
        description="Serialize the five Suite-1 datasets into one .npz each (CONTRACT A).",
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE_DIR, help="read-only source tree")
    parser.add_argument("--out", default=DEFAULT_EXPORT_DIR, help="destination directory")
    parser.add_argument(
        "--datasets",
        default="all",
        type=_parse_keys,
        help="'all' or a comma-separated subset of " + ", ".join(DATASETS),
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="re-read existing exports, recompute counts and checksums, write nothing",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a non-zero status on any cohort or format failure."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    keys: list[str] = args.datasets

    try:
        if args.verify_only:
            return 0 if verify_exports(args.out, keys) else 1
        export_all(args.source, args.out, keys)
    except CohortMismatchError as exc:
        logger.error("COHORT MISMATCH -- %s", exc)
        logger.error(
            "The filter is locked. This is a finding to report, not a parameter to adjust."
        )
        return 1
    except (ExportError, FileNotFoundError, OSError) as exc:
        logger.error("export failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
