"""Loader for the frozen T-06 cohort export (Suite 1 and Suite 2).

Reads the CSR export written by ``export_graphs_suite2.py`` and yields
``networkx.Graph`` objects plus the per-graph arrays the rest of the wave
carries through unchanged.

Two facts about the export that a caller gets wrong silently:

**Node ids are local.** Every graph's vertices are ``0 .. n_nodes-1``; they
are not offsets into a global vertex array.

**Only one orientation of each edge is stored.** ``CONTRACTS.md`` §2 states
that both orientations are stored and therefore that an ``edge_offsets`` span
is ``2 x n_edges``. Measured on all ten files that is wrong: the span is
exactly ``n_edges`` and every pair satisfies ``u < v``. A reader that follows
the contract literally and halves the span loses half of every graph without
raising. :func:`_decode_edges` therefore de-duplicates on the unordered pair
and asserts the recovered count against ``n_edges``, so it is correct under
either layout and loud under neither.

Suite 1 is Suite 2 with ``n_nodes <= 12`` applied and ``aids_graphedx``
renamed ``aids``. The two cohorts are different even where the name matches,
which is why every join in this wave is on ``graph_ids`` and never positional.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import networkx as nx

LOGGER = logging.getLogger(__name__)

#: Default location of the frozen export. Overridable for tests and for a
#: cluster run where the data lives elsewhere.
DEFAULT_COHORT_ROOT = Path(
    "/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/APPROX_GED"
)
COHORT_ROOT_ENV = "ISALGRAPH_T06_COHORT_ROOT"
EXPORT_SUBDIR = "exported_suite2"

#: Dataset keys, in the spelling and order ``CONTRACTS.md`` §2 freezes.
SUITE2_KEYS: tuple[str, ...] = (
    "iam_letter_low",
    "iam_letter_med",
    "iam_letter_high",
    "linux",
    "aids_graphedx",
    "grec",
    "aids_iam",
    "coil_del",
    "mutagenicity",
    "protein",
)
SUITE1_KEYS: tuple[str, ...] = (
    "iam_letter_low",
    "iam_letter_med",
    "iam_letter_high",
    "linux",
    "aids",
)

#: Suite 1 is Suite 2 under this node-count ceiling.
SUITE1_N_MAX = 12

#: Suite-1 key -> the export file it is derived from. Only ``aids`` differs.
SUITE1_SOURCE_KEY: dict[str, str] = {
    "iam_letter_low": "iam_letter_low",
    "iam_letter_med": "iam_letter_med",
    "iam_letter_high": "iam_letter_high",
    "linux": "linux",
    "aids": "aids_graphedx",
}

#: Frozen expected counts. A change here is a change of cohort, not a bug fix.
EXPECTED_SUITE1_COUNTS: dict[str, int] = {
    "iam_letter_low": 1180,
    "iam_letter_med": 1253,
    "iam_letter_high": 2059,
    "linux": 89,
    "aids": 769,
}
EXPECTED_SUITE2_TOTAL = 16370

SUITES: tuple[str, ...] = ("suite1", "suite2")


class CohortError(RuntimeError):
    """Raised when the export is missing, malformed, or the wrong cohort."""


def cohort_root(root: Path | str | None = None) -> Path:
    """Return the export root, honouring the environment override.

    Args:
        root: Explicit root. Wins over the environment and the default.

    Returns:
        Directory holding ``exported_suite2/``.
    """
    if root is not None:
        return Path(root)
    override = os.environ.get(COHORT_ROOT_ENV)
    return Path(override) if override else DEFAULT_COHORT_ROOT


def suite_keys(suite: str) -> tuple[str, ...]:
    """Dataset keys belonging to *suite*.

    Args:
        suite: ``"suite1"`` or ``"suite2"``.

    Returns:
        The frozen key tuple.

    Raises:
        CohortError: If *suite* is not one of the two.
    """
    if suite == "suite2":
        return SUITE2_KEYS
    if suite == "suite1":
        return SUITE1_KEYS
    raise CohortError(f"unknown suite {suite!r}; expected one of {SUITES}")


def source_key(suite: str, dataset: str) -> str:
    """Export-file basename backing ``(suite, dataset)``.

    Args:
        suite: ``"suite1"`` or ``"suite2"``.
        dataset: A key from :func:`suite_keys`.

    Returns:
        The ``.npz`` basename under ``exported_suite2/``.

    Raises:
        CohortError: If *dataset* does not belong to *suite*.
    """
    keys = suite_keys(suite)
    if dataset not in keys:
        raise CohortError(f"{dataset!r} is not a {suite} key; expected one of {keys}")
    return SUITE1_SOURCE_KEY[dataset] if suite == "suite1" else dataset


@dataclass(frozen=True, slots=True)
class Cohort:
    """One dataset of one suite, in cohort order.

    Attributes:
        suite: ``"suite1"`` or ``"suite2"``.
        dataset: The suite-local key.
        source: The export basename the rows came from.
        graph_ids: Stable identifiers, ``<U16``. Every join in the wave is on
            these and never on position.
        node_counts: ``n`` per graph.
        edge_counts: ``m`` per graph, undirected.
        splits: Original train/validation/test assignment, carried through.
        labels: Per-graph class label, ``''`` where the dataset has none.
        edge_lists: One tuple of ``(u, v)`` pairs per graph, ``u < v``.
        source_metadata: The export file's own metadata block.
    """

    suite: str
    dataset: str
    source: str
    graph_ids: np.ndarray
    node_counts: np.ndarray
    edge_counts: np.ndarray
    splits: np.ndarray
    labels: np.ndarray
    edge_lists: tuple[tuple[tuple[int, int], ...], ...]
    source_metadata: dict[str, Any]

    def __len__(self) -> int:
        """Number of graphs."""
        return int(self.graph_ids.shape[0])

    def to_networkx(self, index: int) -> nx.Graph:
        """Build the ``networkx`` graph at *index*.

        Args:
            index: Position in cohort order.

        Returns:
            An undirected graph on ``0 .. n-1`` with every isolated vertex
            present, which the competitor backends require.
        """
        import networkx as nx

        graph = nx.Graph()
        graph.add_nodes_from(range(int(self.node_counts[index])))
        graph.add_edges_from(self.edge_lists[index])
        return graph

    def iter_networkx(self, indices: Sequence[int] | None = None) -> Iterator[tuple[int, nx.Graph]]:
        """Yield ``(index, graph)`` for *indices*, or for the whole cohort.

        Args:
            indices: Positions to yield, in the given order. ``None`` yields
                every graph in cohort order.

        Yields:
            The position and the graph at it.
        """
        chosen = range(len(self)) if indices is None else indices
        for index in chosen:
            yield index, self.to_networkx(index)


def _decode_edges(
    edges: np.ndarray, offsets: np.ndarray, index: int, n_edges: int
) -> tuple[tuple[int, int], ...]:
    """Recover one graph's undirected edge set from the CSR block.

    De-duplicates on the unordered pair, so the result is the same whether the
    export stored one orientation or both. See the module docstring.

    Args:
        edges: The ``(2, E)`` endpoint array.
        offsets: The ``(G+1,)`` span boundaries.
        index: Position in cohort order.
        n_edges: The declared undirected edge count for this graph.

    Returns:
        Sorted ``(u, v)`` pairs with ``u < v``.

    Raises:
        CohortError: If the recovered count disagrees with *n_edges*, or a
            self-loop appears. Both mean the export is not what we think.
    """
    start, stop = int(offsets[index]), int(offsets[index + 1])
    span = edges[:, start:stop]
    pairs = {
        (min(int(u), int(v)), max(int(u), int(v)))
        for u, v in zip(span[0], span[1], strict=True)
        if int(u) != int(v)
    }
    if len(pairs) != n_edges:
        raise CohortError(
            f"graph {index} declares n_edges={n_edges} but its CSR span yields "
            f"{len(pairs)} distinct undirected edges over {stop - start} entries"
        )
    return tuple(sorted(pairs))


def _load_export(root: Path, key: str) -> dict[str, np.ndarray]:
    """Read one export file into a plain dict.

    Args:
        root: The export root (the parent of ``exported_suite2``).
        key: Export basename.

    Returns:
        Every array in the file.

    Raises:
        CohortError: If the file is absent or missing a required key.
    """
    path = root / EXPORT_SUBDIR / f"{key}.npz"
    if not path.is_file():
        raise CohortError(f"cohort export not found: {path}")
    with np.load(path, allow_pickle=False) as handle:
        required = ("graph_ids", "n_nodes", "n_edges", "edge_offsets", "edges", "metadata")
        missing = [name for name in required if name not in handle.files]
        if missing:
            raise CohortError(f"{path} is missing {missing}")
        return {name: handle[name] for name in handle.files}


def load_cohort(
    suite: str,
    dataset: str,
    *,
    root: Path | str | None = None,
    limit: int | None = None,
) -> Cohort:
    """Load one ``(suite, dataset)`` cohort in cohort order.

    Args:
        suite: ``"suite1"`` or ``"suite2"``.
        dataset: A key from :func:`suite_keys`.
        root: Export root; defaults to :func:`cohort_root`.
        limit: Keep only the first ``limit`` graphs after the suite filter.
            Development only -- a production campaign never passes it.

    Returns:
        The cohort.

    Raises:
        CohortError: If the export is malformed or the suite filter yields a
            count that disagrees with the frozen expectation.
    """
    key = source_key(suite, dataset)
    raw = _load_export(cohort_root(root), key)
    keep = np.arange(raw["n_nodes"].shape[0])
    if suite == "suite1":
        keep = keep[raw["n_nodes"][keep] <= SUITE1_N_MAX]
        expected = EXPECTED_SUITE1_COUNTS[dataset]
        if keep.shape[0] != expected:
            raise CohortError(
                f"suite1/{dataset} filtered to {keep.shape[0]} graphs, expected {expected}"
            )
    if limit is not None:
        keep = keep[:limit]

    edge_lists = tuple(
        _decode_edges(raw["edges"], raw["edge_offsets"], int(i), int(raw["n_edges"][i]))
        for i in keep
    )
    blank = np.array([""] * keep.shape[0])
    return Cohort(
        suite=suite,
        dataset=dataset,
        source=key,
        graph_ids=raw["graph_ids"][keep].astype("<U16"),
        node_counts=raw["n_nodes"][keep].astype(np.int32),
        edge_counts=raw["n_edges"][keep].astype(np.int32),
        splits=raw["splits"][keep] if "splits" in raw else blank,
        labels=raw["labels"][keep] if "labels" in raw else blank,
        edge_lists=edge_lists,
        source_metadata=json.loads(str(raw["metadata"])),
    )


def verify(root: Path | str | None = None) -> dict[str, Any]:
    """Recount both suites and check them against the frozen expectations.

    Args:
        root: Export root; defaults to :func:`cohort_root`.

    Returns:
        ``{"suite2": {key: count}, "suite2_total": int, "suite1": {key: count},
        "suite1_total": int}``.

    Raises:
        CohortError: On any disagreement with the frozen counts.
    """
    resolved = cohort_root(root)
    suite2 = {key: int(_load_export(resolved, key)["graph_ids"].shape[0]) for key in SUITE2_KEYS}
    total2 = sum(suite2.values())
    if total2 != EXPECTED_SUITE2_TOTAL:
        raise CohortError(f"suite2 holds {total2} graphs, expected {EXPECTED_SUITE2_TOTAL}")

    suite1: dict[str, int] = {}
    for key in SUITE1_KEYS:
        arrays = _load_export(resolved, SUITE1_SOURCE_KEY[key])
        count = int((arrays["n_nodes"] <= SUITE1_N_MAX).sum())
        if count != EXPECTED_SUITE1_COUNTS[key]:
            raise CohortError(
                f"suite1/{key} holds {count} graphs, expected {EXPECTED_SUITE1_COUNTS[key]}"
            )
        suite1[key] = count
    return {
        "suite2": suite2,
        "suite2_total": total2,
        "suite1": suite1,
        "suite1_total": sum(suite1.values()),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect the frozen T-06 cohort export.")
    parser.add_argument("--root", type=Path, default=None, help="export root")
    parser.add_argument("--verify", action="store_true", help="recount both suites and assert")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv``.

    Returns:
        Process exit status.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args(argv)
    if not args.verify:
        _build_parser().print_help()
        return 0
    counts = verify(args.root)
    LOGGER.info("Suite 2 = %d graphs over %d keys", counts["suite2_total"], len(counts["suite2"]))
    for key, value in counts["suite2"].items():
        LOGGER.info("  suite2/%-16s %6d", key, value)
    LOGGER.info("Suite 1 = %d graphs over %d keys", counts["suite1_total"], len(counts["suite1"]))
    for key, value in counts["suite1"].items():
        LOGGER.info("  suite1/%-16s %6d", key, value)
    LOGGER.info("OK")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
