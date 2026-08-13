"""Generic IAM Graph Database (GXL/CXL) loader for the Suite-2 cohort extension.

Covers the five datasets added for scale -- GREC, AIDS (IAM), COIL-DEL,
Mutagenicity and Protein -- plus the three IAM Letter distortion levels, so that
one code path produces every IAM row of the cohort table.

Why this module exists (T-01, 2026-08-13). The Suite-2 half of the cohort table
had **no reproducing code**: fifteen of sixteen measurement scripts are gone, and
grepping the repository for ``GREC``, ``Mutagenicity``, ``COIL-DEL`` or
``Protein`` returned nothing. Every number in that half -- 19,670 graphs,
40,024,242 pairs, ``n_max = 98`` -- was unverifiable. See
``.claude/notes/review/tasks/T-01-design.md``.

Parsing is **delegated to** :mod:`iam_letter_loader`. That is deliberate: reusing
the exact functions that produce the locked, asserted Suite-1 cohort is what makes
"Suite 1 reproduces ``export_graphs.py``" a genuine check on this module rather
than a comparison of two independent implementations that happen to agree.

Two enumeration policies, and they disagree on one dataset
----------------------------------------------------------
``"cxl"``
    The union of the split index files -- the dataset as the IAM benchmark
    defines it. Consistent with decision 3 ("merge all splits") and with
    :mod:`iam_letter_loader`.
``"directory"``
    Every ``.gxl`` file present on disk, whether or not any split lists it.

The two agree on GREC (1,100), AIDS (2,000), Mutagenicity (4,337), Protein (600)
and all three Letter levels. They **disagree on COIL-DEL**: the split files list
3,900 graphs (2,400 train / 500 valid / 1,000 test) while 7,200 ``.gxl`` files
ship in the same directory. Both figures are reported by
:mod:`cohort_audit`; the choice is a cohort decision, not a parsing detail.

Reference:
    Riesen, K. & Bunke, H. (2008). IAM Graph Database Repository for Graph Based
    Pattern Recognition and Machine Learning. SSPR/SPR 2008, LNCS 5342, 287-297.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Literal

import networkx as nx

# The sibling loaders in this directory import each other by bare top-level name,
# which is the established idiom here (see ``build_extended_merged.py``). Bootstrap
# the path so this module imports both as ``benchmarks.eval_setup.iam_gxl_loader``
# and as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Loaded through importlib rather than ``from iam_letter_loader import ...`` because
# isort/ruff hoist a plain import above the ``sys.path`` line and break it -- the same
# formatter trap the GEDLIB bindings hit (see the project CLAUDE.md).
_letter = importlib.import_module("iam_letter_loader")
parse_cxl = _letter.parse_cxl
parse_gxl = _letter.parse_gxl

logger = logging.getLogger(__name__)

Enumeration = Literal["cxl", "directory"]

#: Split index basenames probed in this order. IAM is inconsistent: Letter and
#: Mutagenicity ship ``validation.cxl`` while GREC, AIDS, COIL-DEL and Protein
#: ship ``valid.cxl``. Probing rather than hardcoding is what keeps one code path.
SPLIT_CANDIDATES: tuple[str, ...] = ("train", "valid", "validation", "val", "test")


class IAMLoaderError(Exception):
    """Raised when an IAM dataset cannot be located or parsed."""


@dataclass(frozen=True, slots=True)
class IAMSpec:
    """Where one IAM dataset lives under the extracted database root.

    Attributes
    ----------
    key : str
        Cohort key, matching the row name in the cohort table.
    subdir : str
        Path relative to ``.../IAM_Database/extracted`` holding the ``.gxl`` and
        ``.cxl`` files.
    """

    key: str
    subdir: str


#: The five datasets added for scale, plus the three Letter levels. Letter is
#: included so that Suite 1 can be re-derived through this module and checked
#: against ``export_graphs.py``'s asserted counts.
IAM_DATASETS: dict[str, IAMSpec] = {
    "grec": IAMSpec("grec", "GREC/data"),
    "aids_iam": IAMSpec("aids_iam", "AIDS/data"),
    "coil_del": IAMSpec("coil_del", "COIL-DEL/data"),
    "mutagenicity": IAMSpec("mutagenicity", "Mutagenicity/data"),
    "protein": IAMSpec("protein", "Protein/data"),
    "iam_letter_low": IAMSpec("iam_letter_low", "Letter/LOW"),
    "iam_letter_med": IAMSpec("iam_letter_med", "Letter/MED"),
    "iam_letter_high": IAMSpec("iam_letter_high", "Letter/HIGH"),
    # Rejected from the cohort (data.md section 2) but audited, because the paper
    # states a reason for each rejection and an unmeasured reason is an assertion.
    # ``Web`` is deliberately absent: it ships ``doc.*.xml`` and no ``.gxl`` at all,
    # so ``load_iam_gxl`` raising is itself the measurement behind "does not parse".
    "fingerprint": IAMSpec("fingerprint", "Fingerprint/data"),
    "coil_rag": IAMSpec("coil_rag", "COIL-RAG/data"),
}

#: Datasets considered and dropped. Audited, never in the cohort.
REJECTED_KEYS: tuple[str, ...] = ("fingerprint", "coil_rag")


@dataclass
class IAMGxlDataset:
    """One parsed IAM dataset, splits merged.

    Attributes
    ----------
    graphs : list[nx.Graph]
        Topology only -- node and edge attributes are stripped by
        :func:`iam_letter_loader.parse_gxl`. Self-loops are dropped and parallel
        edges collapse, since the target is :class:`networkx.Graph`.
    graph_ids : list[str]
        Source filename without its extension.
    labels : list[str]
        Class label from the split index, ``""`` for a graph no index lists.
    splits : list[str]
        Origin split, ``""`` for a graph no index lists.
    key : str
        Cohort key.
    enumeration : str
        Which policy produced this graph list.
    n_files_on_disk : int
        Count of ``.gxl`` files in the directory, independent of enumeration.
    n_files_indexed : int
        Count of distinct filenames named by the split index files.
    split_sizes : dict[str, int]
        Graphs per origin split, in the order the splits were read.
    node_attributes, edge_attributes : list[str]
        Attribute names observed in the first parsed file. Recorded for the
        label column AE.4b asks for (T-18); this module discards the values.
    parse_failures : list[str]
        Files that could not be parsed. Non-empty means the count is suspect.
    """

    graphs: list[nx.Graph] = field(default_factory=list)
    graph_ids: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    splits: list[str] = field(default_factory=list)
    key: str = ""
    enumeration: str = ""
    n_files_on_disk: int = 0
    n_files_indexed: int = 0
    split_sizes: dict[str, int] = field(default_factory=dict)
    node_attributes: list[str] = field(default_factory=list)
    edge_attributes: list[str] = field(default_factory=list)
    parse_failures: list[str] = field(default_factory=list)


def dataset_dir(iam_root: str, key: str) -> str:
    """Resolve the on-disk directory for a cohort key.

    Parameters
    ----------
    iam_root : str
        Path to ``.../IAM_Database/extracted``.
    key : str
        A key of :data:`IAM_DATASETS`.

    Returns
    -------
    str
        Absolute directory path.

    Raises
    ------
    IAMLoaderError
        If the key is unknown or the directory is missing.
    """
    if key not in IAM_DATASETS:
        raise IAMLoaderError(f"unknown IAM dataset key {key!r}")
    path = os.path.join(iam_root, IAM_DATASETS[key].subdir)
    if not os.path.isdir(path):
        raise IAMLoaderError(f"IAM dataset directory not found: {path}")
    return path


def read_split_index(directory: str) -> tuple[dict[str, tuple[str, str]], dict[str, int]]:
    """Read every split index file present in ``directory``.

    Parameters
    ----------
    directory : str
        Directory holding the ``.cxl`` files.

    Returns
    -------
    dict[str, tuple[str, str]]
        Filename to ``(split, class_label)``. A filename listed by more than one
        split keeps its first occurrence, matching
        :func:`iam_letter_loader.load_iam_letter`.
    dict[str, int]
        Graphs contributed per split, in read order.
    """
    index: dict[str, tuple[str, str]] = {}
    split_sizes: dict[str, int] = {}

    for split in SPLIT_CANDIDATES:
        cxl_path = os.path.join(directory, f"{split}.cxl")
        if not os.path.isfile(cxl_path):
            continue
        added = 0
        for filename, class_label in parse_cxl(cxl_path):
            if filename in index:
                continue
            index[filename] = (split, class_label)
            added += 1
        split_sizes[split] = added

    return index, split_sizes


def _observe_attributes(gxl_path: str) -> tuple[list[str], list[str]]:
    """Return the node and edge attribute names carried by one GXL file.

    The loader strips attributes; this records *which* were discarded, because
    AE.4b asks for a label-content column and the answer must be measured rather
    than recalled.
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(gxl_path).getroot()
    graph_el = root.find(".//graph")
    if graph_el is None:
        return [], []

    def names(tag: str) -> list[str]:
        seen: list[str] = []
        for element in graph_el.findall(tag):
            for attr in element.findall("attr"):
                name = attr.get("name", "")
                if name and name not in seen:
                    seen.append(name)
        return seen

    return names("node"), names("edge")


def load_iam_gxl(
    iam_root: str,
    key: str,
    enumeration: Enumeration = "cxl",
) -> IAMGxlDataset:
    """Load one IAM dataset with all splits merged.

    Parameters
    ----------
    iam_root : str
        Path to ``.../IAM_Database/extracted``.
    key : str
        A key of :data:`IAM_DATASETS`.
    enumeration : {'cxl', 'directory'}
        ``'cxl'`` takes the union of the split index files -- the dataset as the
        benchmark defines it. ``'directory'`` takes every ``.gxl`` file present.
        The two differ only on COIL-DEL; see the module docstring.

    Returns
    -------
    IAMGxlDataset
        Parsed graphs with provenance. Both file counts are always populated, so
        a caller can detect the disagreement without loading twice.

    Raises
    ------
    IAMLoaderError
        If the directory is missing or holds no ``.gxl`` files.
    """
    directory = dataset_dir(iam_root, key)

    on_disk = sorted(f for f in os.listdir(directory) if f.endswith(".gxl"))
    if not on_disk:
        raise IAMLoaderError(f"no .gxl files under {directory}")

    index, split_sizes = read_split_index(directory)

    if enumeration == "cxl":
        if not index:
            raise IAMLoaderError(f"no .cxl split index under {directory}; cannot enumerate by cxl")
        filenames = list(index)
    elif enumeration == "directory":
        filenames = on_disk
    else:
        raise IAMLoaderError(f"unknown enumeration policy {enumeration!r}")

    dataset = IAMGxlDataset(
        key=key,
        enumeration=enumeration,
        n_files_on_disk=len(on_disk),
        n_files_indexed=len(index),
        split_sizes=split_sizes,
    )

    for filename in filenames:
        gxl_path = os.path.join(directory, filename)
        if not os.path.isfile(gxl_path):
            dataset.parse_failures.append(f"{filename}: listed in a split index but absent")
            continue
        try:
            graph = parse_gxl(gxl_path)
        except Exception as exc:  # noqa: BLE001 -- a parse failure is data, not control flow
            dataset.parse_failures.append(f"{filename}: {exc}")
            continue

        split, label = index.get(filename, ("", ""))
        dataset.graphs.append(graph)
        dataset.graph_ids.append(os.path.splitext(filename)[0])
        dataset.labels.append(label)
        dataset.splits.append(split)

        if not dataset.node_attributes and not dataset.edge_attributes:
            dataset.node_attributes, dataset.edge_attributes = _observe_attributes(gxl_path)

    if dataset.parse_failures:
        logger.warning(
            "%s: %d of %d files failed to parse; the count is not trustworthy",
            key,
            len(dataset.parse_failures),
            len(filenames),
        )

    logger.info(
        "%s [%s]: %d graphs (%d on disk, %d indexed)",
        key,
        enumeration,
        len(dataset.graphs),
        dataset.n_files_on_disk,
        dataset.n_files_indexed,
    )
    return dataset
