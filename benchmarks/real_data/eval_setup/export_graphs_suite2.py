"""Serialize the ten Suite-2 datasets into one ``.npz`` each -- CONTRACTS sections 1 and 2.

Why this exists
---------------
T-05 computes a proven GED bracket over all **21,710,892** Suite-2 pairs on Picasso. The ten files
this module writes are the *only* input that reaches the cluster. The IAM GXL tree is 35,604 small
files and Picasso's ``fscratch`` enforces a **file-count** quota, so the tree is never transferred;
the GraphEdX graphs live in ``torch`` pickles and ``torch`` is not installed on the cluster and must
never need to be. Eleven files travel instead of ~35,600, and the cluster-side reader
(:func:`export_graphs.load_exported`) imports nothing beyond ``numpy`` and ``networkx``.

Relationship to :mod:`export_graphs`
------------------------------------
This module is the Suite-2 **registry and driver**; it is not a second serializer. The schema, the
CSR packing, every array invariant and the manifest all come from :mod:`export_graphs` by import, so
a future fix there applies to both suites and the two cannot drift. What differs is the cohort:

===================  ==========================  ==============================
Aspect               Suite 1 (``export_graphs``)  Suite 2 (this module)
===================  ==========================  ==============================
datasets             5                            10
``n_max``            12                           **none**
IAM enumeration      Letter only                  every IAM set, by **split index**
``labels``           empty for every dataset      **populated** where a class exists
===================  ==========================  ==============================

The cohort is **locked**. ``filter_graphs(min_nodes=2, require_connected=True, no n_max)`` over
merged splits must reproduce :data:`SUITE2_DATASETS` exactly. A mismatch aborts with a non-zero exit
status; it is a finding to report, never a filter to adjust.

Two roots, not one
------------------
LINUX and AIDS-GraphEdX are not IAM datasets. The IAM tree and the GraphEdX tree sit under different
parents and **no single source directory resolves both**, so this module takes two roots. The frozen
loaders in ``export_graphs.py:430`` and ``cohort_audit.py:254`` both assume a single
``<source>/GED_PRECOMPUTED/<NAME>`` layout that does not exist on the current tree; that is a defect
in those files, reported rather than patched, and the reason the GraphEdX split reconstruction is
replicated here instead of imported.

Decision 27 -- enumerate by split index, never by directory
-----------------------------------------------------------
``COIL-DEL/data`` ships **7,200** ``.gxl`` files; ``train.cxl``/``valid.cxl``/``test.cxl`` name
**3,900** of them (2,400 / 500 / 1,000) and the other 3,300 carry no class label. Exporting the
directory reproduces the *retracted* cohort -- 7,200 graphs and 25,916,400 pairs for this dataset
alone -- and nothing would raise. The 3,900 graphs are exactly class-balanced at 100 x 39, and that
balance is asserted here because it is what distinguishes the two enumerations by a property of the
data rather than by a count that could be reached another way.

Usage
-----
``python -m benchmarks.real_data.eval_setup.export_graphs_suite2 --verify-only``
``python -m benchmarks.real_data.eval_setup.export_graphs_suite2 --out DIR``
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from math import comb
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cohort_audit import NO_N_MAX, SUITE2_KEYS  # noqa: E402
from dataset_filter import FilterResult, filter_graphs  # noqa: E402
from export_graphs import (  # noqa: E402
    MANIFEST_NAME,
    ExportedDataset,
    ExportError,
    _git_commit,
    _normalise,
    content_sha256,
    save_exported,
    sha256_file,
    write_manifest,
)
from iam_gxl_loader import load_iam_gxl  # noqa: E402

logger = logging.getLogger(__name__)

#: Bumped independently of :data:`export_graphs.SCHEMA_VERSION`; the array schema is shared but the
#: metadata carries two Suite-2-only keys (``enumeration``, ``label_classes``).
SCHEMA_VERSION = 1

#: The locked Suite-2 filter. Never parameterised -- the cohort is frozen.
FILTER_MIN_NODES = 2
FILTER_REQUIRE_CONNECTED = True

#: Decision 27. ``"directory"`` is never used here; it exists in the loader for the audit only.
ENUMERATION = "cxl"

#: What the ``enumeration`` metadata field records, per CONTRACTS section 2. The loader's own name
#: for the same policy is ``"cxl"``; the contract asks for the descriptive name.
ENUMERATION_LABEL = "split_index"

_SANDISK = "/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph"
DEFAULT_IAM_ROOT = f"{_SANDISK}/data/source/APPROX_GED/datasets/IAM_Database/extracted"
DEFAULT_GRAPHEDX_ROOT = f"{_SANDISK}/data/source/GED_PRECOMPUTED"
DEFAULT_EXPORT_DIR = f"{_SANDISK}/data/source/APPROX_GED/exported_suite2"

#: Suite-1 census used as an external, element-wise check on loader and ordering.
DEFAULT_REFERENCE_DIR = f"{DEFAULT_GRAPHEDX_ROOT}/extended_merged_exact_ged/computed"

#: The four datasets whose Suite-2 cohort is *identical* to Suite 1, so their ``graph_ids`` must
#: reproduce the reference census element-wise. ``aids_graphedx`` is deliberately absent: Suite 1's
#: ``aids`` applies ``n_max = 12`` and keeps 769 of the same 911 graphs, so the arrays differ by
#: construction and comparing them would be a bug, not a check.
REFERENCE_KEYS: tuple[str, ...] = (
    "iam_letter_low",
    "iam_letter_med",
    "iam_letter_high",
    "linux",
)

#: COIL-DEL under the split index. Both numbers are asserted; see the module docstring.
COIL_DEL_KEY = "coil_del"
COIL_DEL_CLASSES = 100
COIL_DEL_PER_CLASS = 39


class Suite2ExportError(Exception):
    """A Suite-2 export violates CONTRACTS section 1 or 2."""


class Suite2CohortMismatchError(Suite2ExportError):
    """Observed graph, pair or class counts differ from the locked cohort."""


@dataclass(frozen=True, slots=True)
class Suite2Spec:
    """One Suite-2 dataset and the counts it must reproduce.

    Attributes
    ----------
    key : str
        Export key; the ``.npz`` is written as ``{key}.npz``.
    source : str
        Which loader family reads it, ``"iam_gxl"`` or ``"graphedx"``.
    loader_arg : str
        IAM cohort key, or the GraphEdX dataset directory (``LINUX`` / ``AIDS``).
    expected_kept : int
        Locked graph count after the Suite-2 filter.
    expected_pairs : int
        Locked pair count; always ``C(expected_kept, 2)``.
    expected_label_classes : int
        Distinct non-empty class labels **among kept graphs**. ``0`` means the source carries no
        class label at all. These are post-filter counts and differ from the raw dataset counts
        wherever the connectivity filter removes a whole class -- see :func:`assert_label_classes`.
    """

    key: str
    source: str
    loader_arg: str
    expected_kept: int
    expected_pairs: int
    expected_label_classes: int


#: The locked cohort (CONTRACTS section 1 / T-05-design section 2). Key order is the export order
#: and matches :data:`cohort_audit.SUITE2_KEYS`, which :func:`_check_registry` enforces.
SUITE2_DATASETS: dict[str, Suite2Spec] = {
    "iam_letter_low": Suite2Spec("iam_letter_low", "iam_gxl", "iam_letter_low", 1180, 695610, 9),
    "iam_letter_med": Suite2Spec("iam_letter_med", "iam_gxl", "iam_letter_med", 1253, 784378, 15),
    "iam_letter_high": Suite2Spec(
        "iam_letter_high", "iam_gxl", "iam_letter_high", 2059, 2118711, 15
    ),
    "linux": Suite2Spec("linux", "graphedx", "LINUX", 89, 3916, 0),
    "aids_graphedx": Suite2Spec("aids_graphedx", "graphedx", "AIDS", 819, 334971, 0),
    "grec": Suite2Spec("grec", "iam_gxl", "grec", 650, 210925, 17),
    "aids_iam": Suite2Spec("aids_iam", "iam_gxl", "aids_iam", 1811, 1638955, 2),
    "coil_del": Suite2Spec("coil_del", "iam_gxl", "coil_del", 3900, 7603050, 100),
    "mutagenicity": Suite2Spec("mutagenicity", "iam_gxl", "mutagenicity", 4040, 8158780, 2),
    "protein": Suite2Spec("protein", "iam_gxl", "protein", 569, 161596, 6),
}

TOTAL_EXPECTED_GRAPHS = 16370
TOTAL_EXPECTED_PAIRS = 21710892


def _check_registry() -> None:
    """Fail at import unless the registry is T-01's certified enumeration, in its order.

    Written as an explicit raise rather than ``assert`` because ``python -O`` strips assertions, and
    a registry that silently diverges from :data:`cohort_audit.SUITE2_KEYS` is precisely the failure
    this ticket cannot afford: every downstream pair index is positional.

    Raises
    ------
    Suite2ExportError
        If the keys or their order differ from ``cohort_audit.SUITE2_KEYS``, if a declared pair
        count is not ``C(kept, 2)``, or if the totals do not sum.
    """
    if tuple(SUITE2_DATASETS) != tuple(SUITE2_KEYS):
        raise Suite2ExportError(
            f"registry keys {tuple(SUITE2_DATASETS)} != "
            f"cohort_audit.SUITE2_KEYS {tuple(SUITE2_KEYS)}"
        )
    for spec in SUITE2_DATASETS.values():
        if spec.expected_pairs != comb(spec.expected_kept, 2):
            raise Suite2ExportError(
                f"{spec.key}: declared {spec.expected_pairs} pairs, "
                f"C({spec.expected_kept}, 2) = {comb(spec.expected_kept, 2)}"
            )
    total_graphs = sum(s.expected_kept for s in SUITE2_DATASETS.values())
    total_pairs = sum(s.expected_pairs for s in SUITE2_DATASETS.values())
    if total_graphs != TOTAL_EXPECTED_GRAPHS or total_pairs != TOTAL_EXPECTED_PAIRS:
        raise Suite2ExportError(
            f"registry totals {total_graphs} graphs / {total_pairs} pairs, "
            f"expected {TOTAL_EXPECTED_GRAPHS} / {TOTAL_EXPECTED_PAIRS}"
        )


_check_registry()


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _load_iam(
    iam_root: str | Path, key: str
) -> tuple[list[nx.Graph], list[str], list[str], list[str]]:
    """Load one IAM dataset by split index, splits merged.

    Parameters
    ----------
    iam_root : str or Path
        Path to ``.../IAM_Database/extracted``.
    key : str
        A key of :data:`iam_gxl_loader.IAM_DATASETS`.

    Returns
    -------
    tuple
        ``(graphs, graph_ids, splits, labels)`` in split-index order.

    Raises
    ------
    Suite2ExportError
        If any file failed to parse. A partial parse yields a plausible count that is silently
        wrong, which is the failure mode this whole module exists to prevent.
    """
    dataset = load_iam_gxl(str(iam_root), key, enumeration=ENUMERATION)
    if dataset.parse_failures:
        raise Suite2ExportError(
            f"{key}: {len(dataset.parse_failures)} file(s) failed to parse; "
            f"first: {dataset.parse_failures[0]}"
        )
    return dataset.graphs, dataset.graph_ids, dataset.splits, dataset.labels


def _load_graphedx(
    graphedx_root: str | Path, name: str
) -> tuple[list[nx.Graph], list[str], list[str], list[str]]:
    """Load one GraphEdX dataset, splits merged.

    Replicates ``export_graphs.py:416 _load_graphedx`` including its id-versus-split cross-check.
    It is replicated rather than imported because that function hardcodes
    ``<source>/GED_PRECOMPUTED/<NAME>``, a layout the current tree does not have -- the datasets are
    under ``GED_PRECOMPUTED/datasets/<NAME>``.

    ``load_graphedx_dataset`` returns no per-graph split list; it appends graphs in ``SPLITS`` order
    and records the sizes in an insertion-ordered dict. The split label is reconstructed from those
    sizes and then cross-checked against the split encoded in each ``graph_id``, so a future
    reordering inside the loader fails here instead of silently mislabelling a graph.

    Parameters
    ----------
    graphedx_root : str or Path
        Path to ``.../GED_PRECOMPUTED``. ``datasets/`` is appended here.
    name : str
        ``"LINUX"`` or ``"AIDS"``.

    Returns
    -------
    tuple
        ``(graphs, graph_ids, splits, labels)``. ``labels`` is all ``""``: GraphEdX carries no class
        label, and LINUX carries no node or edge attribute at all (T-01).

    Raises
    ------
    Suite2ExportError
        If the split sizes do not account for every graph, or an id contradicts its split.
    """
    from graphedx_loader import load_graphedx_dataset

    base = Path(graphedx_root) / "datasets"
    dataset = load_graphedx_dataset(name, str(base))

    splits: list[str] = []
    for split, size in dataset.split_sizes.items():
        splits.extend([split] * size)
    if len(splits) != len(dataset.graphs):
        raise Suite2ExportError(
            f"{name}: split_sizes sum to {len(splits)} but {len(dataset.graphs)} graphs were loaded"
        )

    prefix = name.lower()
    for gid, split in zip(dataset.graph_ids, splits, strict=True):
        if gid.rsplit("_", 1)[0] != f"{prefix}_{split}":
            raise Suite2ExportError(
                f"{name}: graph id {gid!r} contradicts reconstructed split {split!r}"
            )

    return dataset.graphs, dataset.graph_ids, splits, [""] * len(dataset.graphs)


def load_raw(
    spec: Suite2Spec, iam_root: str | Path, graphedx_root: str | Path
) -> tuple[list[nx.Graph], list[str], list[str], list[str]]:
    """Dispatch to the loader for ``spec``.

    Returns
    -------
    tuple
        ``(graphs, graph_ids, splits, labels)``, splits merged, in export order.

    Raises
    ------
    Suite2ExportError
        If the source family is unknown.
    """
    if spec.source == "iam_gxl":
        return _load_iam(iam_root, spec.loader_arg)
    if spec.source == "graphedx":
        return _load_graphedx(graphedx_root, spec.loader_arg)
    raise Suite2ExportError(f"unknown source family {spec.source!r} for {spec.key!r}")


# --------------------------------------------------------------------------- #
# Cohort assertions
# --------------------------------------------------------------------------- #


def assert_cohort(spec: Suite2Spec, n_kept: int, n_pairs: int) -> None:
    """Abort unless the observed counts equal the locked cohort.

    Raises
    ------
    Suite2CohortMismatchError
        With the observed values printed beside the expected ones. The filter is never adjusted to
        make this pass.
    """
    if n_kept == spec.expected_kept and n_pairs == spec.expected_pairs:
        return
    raise Suite2CohortMismatchError(
        f"{spec.key}: observed {n_kept} graphs / {n_pairs} pairs, "
        f"expected {spec.expected_kept} graphs / {spec.expected_pairs} pairs"
    )


def assert_label_classes(spec: Suite2Spec, labels: list[str]) -> None:
    """Abort unless the retained class count matches the locked cohort.

    The expected values are **post-filter** and differ from the raw dataset counts wherever the
    connectivity filter removes a whole class: GREC retains 17 of 22, Letter LOW 9 of 15. Asserting
    the measured value is what turns a future label regression into a failure instead of a quietly
    different table.

    Raises
    ------
    Suite2CohortMismatchError
        If the number of distinct non-empty labels differs from ``spec.expected_label_classes``.
    """
    observed = sorted({label for label in labels if label})
    if len(observed) == spec.expected_label_classes:
        return
    raise Suite2CohortMismatchError(
        f"{spec.key}: observed {len(observed)} label classes, "
        f"expected {spec.expected_label_classes}; observed classes {observed}"
    )


def assert_coil_del_balance(labels: list[str]) -> None:
    """Abort unless COIL-DEL is exactly 100 classes of exactly 39 graphs.

    This is the property that separates the split-index enumeration from the directory one by
    something other than a count. The 3,300 graphs no split lists carry no class label, so a
    directory export is both unbalanced and partly unlabelled.

    Raises
    ------
    Suite2CohortMismatchError
        On any class count other than 100, or any per-class size other than 39.
    """
    counts = Counter(labels)
    sizes = sorted(set(counts.values()))
    if len(counts) == COIL_DEL_CLASSES and sizes == [COIL_DEL_PER_CLASS]:
        return
    raise Suite2CohortMismatchError(
        f"{COIL_DEL_KEY}: expected {COIL_DEL_CLASSES} classes x {COIL_DEL_PER_CLASS} graphs, "
        f"observed {len(counts)} classes with sizes {sizes}"
    )


def check_reference_graph_ids(
    key: str, graph_ids: list[str], reference_dir: str | Path
) -> list[str]:
    """Compare ``graph_ids`` element-wise against the Suite-1 census.

    For the four datasets in :data:`REFERENCE_KEYS` the Suite-2 cohort is identical to Suite 1, so
    this is an exact, end-to-end check of the loader *and* the ordering against a census already on
    record. It costs one ``np.load`` per dataset.

    Parameters
    ----------
    key : str
        Dataset key.
    graph_ids : list[str]
        The ids this module produced, in export order.
    reference_dir : str or Path
        Directory holding ``extended_merged_exact_ged/computed/{key}.npz``.

    Returns
    -------
    list[str]
        One line per problem; empty when the arrays agree or the reference is absent.
    """
    if key not in REFERENCE_KEYS:
        return []
    path = Path(reference_dir) / f"{key}.npz"
    if not path.is_file():
        logger.warning("reference census absent, skipping graph_ids check for %s: %s", key, path)
        return []

    with np.load(path, allow_pickle=False) as handle:
        reference = handle["graph_ids"]

    mine = np.asarray(graph_ids, dtype=reference.dtype)
    if mine.shape != reference.shape:
        return [f"{key}: graph_ids shape {mine.shape}, reference {reference.shape}"]
    if not bool((mine == reference).all()):
        bad = int(np.flatnonzero(mine != reference)[0])
        return [
            f"{key}: graph_ids differ from the reference census; first at index {bad}: "
            f"{mine[bad]!r} vs {reference[bad]!r}"
        ]
    logger.info("%s: graph_ids reproduce the reference census (%d ids)", key, mine.shape[0])
    return []


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def build_exported(
    spec: Suite2Spec,
    iam_root: str | Path,
    graphedx_root: str | Path,
    code_commit: str,
    reference_dir: str | Path | None = None,
) -> ExportedDataset:
    """Load, filter and package one Suite-2 dataset, asserting the locked cohort.

    Parameters
    ----------
    spec : Suite2Spec
        Dataset to build.
    iam_root, graphedx_root : str or Path
        The two read-only source roots.
    code_commit : str
        Repository sha recorded in the metadata.
    reference_dir : str or Path, optional
        Suite-1 census directory. When given and the dataset is in :data:`REFERENCE_KEYS`, the
        ``graph_ids`` are checked element-wise against it.

    Returns
    -------
    ExportedDataset
        Ready for :func:`export_graphs.save_exported`.

    Raises
    ------
    Suite2CohortMismatchError
        If the filtered counts, the class count, the COIL-DEL balance or the reference ``graph_ids``
        disagree with the locked cohort.
    """
    graphs, graph_ids, splits, labels = load_raw(spec, iam_root, graphedx_root)
    raw_classes = sorted({label for label in labels if label})

    result: FilterResult = filter_graphs(
        graphs,
        graph_ids,
        n_max=NO_N_MAX,
        require_connected=FILTER_REQUIRE_CONNECTED,
        min_nodes=FILTER_MIN_NODES,
    )
    kept = result.kept_indices
    n_pairs = comb(result.n_kept, 2)
    assert_cohort(spec, result.n_kept, n_pairs)

    kept_ids = [graph_ids[i] for i in kept]
    kept_labels = [labels[i] for i in kept]
    assert_label_classes(spec, kept_labels)
    if spec.key == COIL_DEL_KEY:
        assert_coil_del_balance(kept_labels)

    if reference_dir is not None:
        problems = check_reference_graph_ids(spec.key, kept_ids, reference_dir)
        if problems:
            raise Suite2CohortMismatchError("; ".join(problems))

    kept_graphs = [_normalise(graphs[i]) for i in kept]
    n_nodes = np.asarray([g.number_of_nodes() for g in kept_graphs], dtype=np.int32)
    n_edges = np.asarray([g.number_of_edges() for g in kept_graphs], dtype=np.int32)

    kept_classes = sorted({label for label in kept_labels if label})
    metadata: dict[str, object] = {
        "dataset": spec.key,
        "source": spec.source,
        "n_raw": result.n_raw,
        "n_kept": result.n_kept,
        "n_dropped_min_nodes": result.n_dropped_trivial,
        "n_dropped_disconnected": result.n_dropped_disconnected,
        "n_pairs": n_pairs,
        "filter": {
            "min_nodes": FILTER_MIN_NODES,
            "require_connected": FILTER_REQUIRE_CONNECTED,
            "n_max": None,
        },
        "splits_merged": True,
        "enumeration": ENUMERATION_LABEL,
        # Recorded on the orchestrator's ruling of 2026-08-13: T-18 needs *which* classes vanish,
        # not only how many survive. The raw list is kept so the loss is readable from one file.
        "n_label_classes": len(kept_classes),
        "label_classes": kept_classes,
        "n_label_classes_raw": len(raw_classes),
        "label_classes_lost": sorted(set(raw_classes) - set(kept_classes)),
        "exported_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_commit": code_commit,
        "schema_version": SCHEMA_VERSION,
    }

    return ExportedDataset(
        key=spec.key,
        graphs=kept_graphs,
        graph_ids=kept_ids,
        splits=[splits[i] for i in kept],
        labels=kept_labels,
        n_nodes=n_nodes,
        n_edges=n_edges,
        metadata=metadata,
    )


def _manifest_entry(dataset: ExportedDataset, path: Path) -> dict[str, object]:
    """Build one manifest row for a written dataset."""
    meta = dataset.metadata
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "content_sha256": content_sha256(dataset),
        "bytes": path.stat().st_size,
        "n_graphs": len(dataset.graphs),
        "n_pairs": meta["n_pairs"],
        "n_raw": meta["n_raw"],
        "n_edges_total": int(sum(int(x) for x in dataset.n_edges)),
        "n_nodes_max": int(max((int(x) for x in dataset.n_nodes), default=0)),
        "source": meta["source"],
        "enumeration": meta["enumeration"],
        "splits": sorted(set(dataset.splits)),
        "n_label_classes": meta["n_label_classes"],
        "label_classes": meta["label_classes"],
        "n_label_classes_raw": meta["n_label_classes_raw"],
        "label_classes_lost": meta["label_classes_lost"],
        "schema_version": meta["schema_version"],
    }


def export_all(
    iam_root: str | Path,
    graphedx_root: str | Path,
    export_dir: str | Path,
    reference_dir: str | Path | None = None,
) -> dict[str, dict[str, object]]:
    """Build, assert and write all ten datasets plus ``manifest.json``.

    Parameters
    ----------
    iam_root, graphedx_root : str or Path
        The two read-only source roots.
    export_dir : str or Path
        Destination directory; created if absent.
    reference_dir : str or Path, optional
        Suite-1 census directory for the element-wise ``graph_ids`` check.

    Returns
    -------
    dict
        The manifest, keyed by dataset.

    Raises
    ------
    Suite2CohortMismatchError
        On any locked-count disagreement, before anything is written for that dataset.
    """
    code_commit = _git_commit()
    out_dir = Path(export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: dict[str, dict[str, object]] = {}
    total_graphs = total_pairs = 0
    for key, spec in SUITE2_DATASETS.items():
        dataset = build_exported(spec, iam_root, graphedx_root, code_commit, reference_dir)
        path = out_dir / f"{key}.npz"
        save_exported(dataset, path)
        entries[key] = _manifest_entry(dataset, path)
        total_graphs += len(dataset.graphs)
        total_pairs += int(entries[key]["n_pairs"])  # type: ignore[arg-type]

    if total_graphs != TOTAL_EXPECTED_GRAPHS or total_pairs != TOTAL_EXPECTED_PAIRS:
        raise Suite2CohortMismatchError(
            f"totals {total_graphs} graphs / {total_pairs} pairs, "
            f"expected {TOTAL_EXPECTED_GRAPHS} / {TOTAL_EXPECTED_PAIRS}"
        )

    entries["_totals"] = {
        "n_datasets": len(SUITE2_DATASETS),
        "n_graphs": total_graphs,
        "n_pairs": total_pairs,
        "code_commit": code_commit,
        "iam_root": str(iam_root),
        "graphedx_root": str(graphedx_root),
        "filter": {
            "min_nodes": FILTER_MIN_NODES,
            "require_connected": FILTER_REQUIRE_CONNECTED,
            "n_max": None,
        },
        "enumeration": ENUMERATION_LABEL,
        "schema_version": SCHEMA_VERSION,
    }
    write_manifest(entries, out_dir)
    return entries


def verify(
    iam_root: str | Path,
    graphedx_root: str | Path,
    reference_dir: str | Path | None = None,
) -> list[str]:
    """Reproduce every locked count without writing anything.

    Returns
    -------
    list[str]
        One line per problem; empty when all ten rows and both totals reproduce.
    """
    problems: list[str] = []
    total_graphs = total_pairs = 0

    for key, spec in SUITE2_DATASETS.items():
        try:
            graphs, graph_ids, _splits, labels = load_raw(spec, iam_root, graphedx_root)
        except Exception as exc:  # noqa: BLE001 -- a load failure is a finding, not control flow
            problems.append(f"{key}: load failed: {exc}")
            continue

        result = filter_graphs(
            graphs,
            graph_ids,
            n_max=NO_N_MAX,
            require_connected=FILTER_REQUIRE_CONNECTED,
            min_nodes=FILTER_MIN_NODES,
        )
        n_pairs = comb(result.n_kept, 2)
        kept_ids = [graph_ids[i] for i in result.kept_indices]
        kept_labels = [labels[i] for i in result.kept_indices]
        total_graphs += result.n_kept
        total_pairs += n_pairs

        try:
            assert_cohort(spec, result.n_kept, n_pairs)
        except Suite2CohortMismatchError as exc:
            problems.append(str(exc))
        try:
            assert_label_classes(spec, kept_labels)
        except Suite2CohortMismatchError as exc:
            problems.append(str(exc))
        if key == COIL_DEL_KEY:
            try:
                assert_coil_del_balance(kept_labels)
            except Suite2CohortMismatchError as exc:
                problems.append(str(exc))
        if reference_dir is not None:
            problems.extend(check_reference_graph_ids(key, kept_ids, reference_dir))

        logger.info(
            "%-16s raw=%5d kept=%5d pairs=%9d classes=%3d",
            key,
            result.n_raw,
            result.n_kept,
            n_pairs,
            len({x for x in kept_labels if x}),
        )

    if total_graphs != TOTAL_EXPECTED_GRAPHS:
        problems.append(f"total graphs {total_graphs}, locked {TOTAL_EXPECTED_GRAPHS}")
    if total_pairs != TOTAL_EXPECTED_PAIRS:
        problems.append(f"total pairs {total_pairs}, locked {TOTAL_EXPECTED_PAIRS}")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Export or verify the Suite-2 cohort. Returns a process exit status."""
    parser = argparse.ArgumentParser(description="Serialize the ten Suite-2 datasets.")
    parser.add_argument("--iam-root", default=DEFAULT_IAM_ROOT, help="IAM_Database/extracted")
    parser.add_argument("--graphedx-root", default=DEFAULT_GRAPHEDX_ROOT, help="GED_PRECOMPUTED")
    parser.add_argument("--out", default=DEFAULT_EXPORT_DIR, help="destination directory")
    parser.add_argument(
        "--reference-dir",
        default=DEFAULT_REFERENCE_DIR,
        help="Suite-1 census for the element-wise graph_ids check; '' disables it",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="reproduce every locked count and exit, writing nothing",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    reference_dir: str | None = args.reference_dir or None

    if args.verify_only:
        problems = verify(args.iam_root, args.graphedx_root, reference_dir)
        if problems:
            for line in problems:
                logger.error("%s", line)
            logger.error("Suite-2 cohort does NOT reproduce (%d problems)", len(problems))
            return 1
        logger.info(
            "Suite-2 cohort reproduces: %d datasets, %d graphs, %d pairs",
            len(SUITE2_DATASETS),
            TOTAL_EXPECTED_GRAPHS,
            TOTAL_EXPECTED_PAIRS,
        )
        return 0

    try:
        entries = export_all(args.iam_root, args.graphedx_root, args.out, reference_dir)
    except (Suite2ExportError, ExportError) as exc:
        logger.error("%s", exc)
        return 1

    totals: Any = entries["_totals"]
    logger.info(
        "Wrote %d datasets + %s to %s (%s graphs, %s pairs)",
        len(SUITE2_DATASETS),
        MANIFEST_NAME,
        args.out,
        totals["n_graphs"],
        totals["n_pairs"],
    )
    logger.debug("%s", json.dumps(totals, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
