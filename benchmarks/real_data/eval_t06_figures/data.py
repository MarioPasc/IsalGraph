"""Readers for the T-06 archive. No plotting, no statistics beyond aggregation.

Two sources, and they are not interchangeable:

``encodings/suite{1,2}/<dataset>__<representation>.npz``
    One row per graph: node count, edge count, both bit conventions, status.
    This is the only source with per-graph resolution, so it is what an
    absolute-scale figure and any re-paired comparison must read.

``<report>/data/*.json``
    The campaign's own reductions -- ``claim_a_strata.json``,
    ``size_profile.json``, ``rho_table.json``. A figure quoting a headline
    number reads these, never a recomputation, so the figure and the text
    cannot disagree.

**Every pooled quantity carries how many datasets produced it.** Above
``n = 66`` the cohorts thin out until a single dataset contributes, and a
median pooled over one dataset is a statement about that dataset rather than
about the cohort. :class:`Aggregate` therefore carries ``n_datasets`` and
``n_graphs`` on every point, and :func:`dataset_support` reports the same thing
per node count so a caller can set its own guard. Dropping that column is how a
composition artefact becomes a trend line.

The composition-free alternative is :func:`paired_relative_gap`, which pairs on
graph identity inside a dataset: the same graphs sit on both sides of every
ratio, so it needs no guard at all.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
from numpy.typing import NDArray

LOGGER: Final = logging.getLogger(__name__)

#: Bit conventions ``competitors/bits.py`` emits, both always together.
CONVENTIONS: Final[tuple[str, str]] = ("entropy_bits", "realised_bits")

#: Minimum graphs in a node-count stratum before it is tested. Matches
#: ``claim_a_strata.json``'s ``min_graphs_per_stratum``.
MIN_GRAPHS: Final[int] = 8


@dataclass(frozen=True, slots=True)
class Cell:
    """One ``(suite, dataset, representation)`` encoding cell.

    Attributes:
        suite: ``suite1`` or ``suite2``.
        dataset: Dataset name.
        representation: Backend name.
        graph_ids: Graph identifier per row, the key any pairing joins on.
        node_counts: Node count per row.
        edge_counts: Edge count per row.
        entropy_bits: ``L log2 |Sigma|`` per row, ``nan`` where undefined.
        realised_bits: Format-defined byte length times eight, per row.
        length: Symbol count per row -- the unit of edit, and the only field
            from which an alternative realised convention may be derived.
        status: ``ok``, ``censored`` or ``error`` per row.
        fallback_used: Whether D14 substituted a greedy-min string, per row.
    """

    suite: str
    dataset: str
    representation: str
    graph_ids: NDArray[Any]
    node_counts: NDArray[Any]
    edge_counts: NDArray[Any]
    entropy_bits: NDArray[Any]
    realised_bits: NDArray[Any]
    length: NDArray[Any]
    status: NDArray[Any]
    fallback_used: NDArray[Any]

    @property
    def usable(self) -> NDArray[np.bool_]:
        """Boolean mask of rows carrying a finite bit count.

        ``status == 'censored'`` is usable: D14 retains a censored graph with
        its greedy-min string, so it did produce an encoding. Only ``error``
        rows and non-finite bit counts are dropped.
        """
        mask = (self.status != "error") & np.isfinite(self.entropy_bits)
        return cast("NDArray[np.bool_]", mask)


@dataclass(frozen=True, slots=True)
class Aggregate:
    """One pooled point at fixed ``(representation, n)``.

    Attributes:
        representation: Backend name.
        n: Node count.
        median: Median bit count across every contributing graph.
        q1: First quartile.
        q3: Third quartile.
        n_graphs: Graphs contributing.
        n_datasets: Datasets contributing. **One means the point describes a
            single dataset, not the cohort.**
    """

    representation: str
    n: int
    median: float
    q1: float
    q3: float
    n_graphs: int
    n_datasets: int


def load_cells(root: Path) -> list[Cell]:
    """Load every encoding cell under *root*.

    Args:
        root: The ``encodings/`` directory, holding ``suite1/`` and
            ``suite2/``.

    Returns:
        One :class:`Cell` per ``.npz``, in sorted path order.

    Raises:
        FileNotFoundError: If *root* holds no ``.npz`` at all, which means
            the archive path is wrong rather than the campaign incomplete.
    """
    paths = sorted(root.glob("suite*/*.npz"))
    if not paths:
        raise FileNotFoundError(f"no encoding cells under {root}")
    cells: list[Cell] = []
    for path in paths:
        dataset, representation = path.stem.split("__", 1)
        z = np.load(path, allow_pickle=True)
        cells.append(
            Cell(
                suite=path.parent.name,
                dataset=dataset,
                representation=representation,
                graph_ids=z["graph_ids"],
                node_counts=z["node_counts"],
                edge_counts=z["edge_counts"],
                entropy_bits=z["entropy_bits"],
                realised_bits=z["realised_bits"],
                length=z["length"],
                status=z["status"],
                fallback_used=z["fallback_used"],
            )
        )
    LOGGER.info("loaded %d encoding cells from %s", len(cells), root)
    return cells


def aggregate_bits(
    cells: list[Cell],
    *,
    convention: str = "entropy_bits",
    min_graphs: int = MIN_GRAPHS,
) -> list[Aggregate]:
    """Pool bit counts across datasets at fixed ``(representation, n)``.

    Args:
        cells: Encoding cells.
        convention: ``entropy_bits`` or ``realised_bits``.
        min_graphs: Strata below this many graphs are dropped.

    Returns:
        One :class:`Aggregate` per ``(representation, n)`` that clears
        *min_graphs*.

    Raises:
        ValueError: If *convention* is not one of :data:`CONVENTIONS`.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}, got {convention!r}")
    bits: dict[tuple[str, int], list[float]] = defaultdict(list)
    datasets: dict[tuple[str, int], set[str]] = defaultdict(set)
    for cell in cells:
        keep = cell.usable
        values = getattr(cell, convention)[keep]
        for n, value in zip(cell.node_counts[keep], values, strict=True):
            bits[(cell.representation, int(n))].append(float(value))
            datasets[(cell.representation, int(n))].add(cell.dataset)
    out: list[Aggregate] = []
    for (representation, n), values in sorted(bits.items()):
        if len(values) < min_graphs:
            continue
        arr = np.asarray(values)
        out.append(
            Aggregate(
                representation=representation,
                n=n,
                median=float(np.median(arr)),
                q1=float(np.percentile(arr, 25)),
                q3=float(np.percentile(arr, 75)),
                n_graphs=len(values),
                n_datasets=len(datasets[(representation, n)]),
            )
        )
    return out


def dataset_support(cells: list[Cell], *, min_graphs: int = MIN_GRAPHS) -> dict[int, int]:
    """Return, per node count, how many datasets contribute at least one graph.

    Args:
        cells: Encoding cells.
        min_graphs: Ignored for the count itself; kept so the caller reads
            the same guard the aggregates used.

    Returns:
        ``{n: number of distinct datasets}``. A caller pooling across datasets
        uses this to decide where its pooled curve stops describing the cohort.
    """
    seen: dict[int, set[str]] = defaultdict(set)
    for cell in cells:
        if cell.representation != "adjacency":
            # One representation is enough and adjacency never refuses, so it
            # measures cohort support rather than backend coverage.
            continue
        for n in cell.node_counts:
            seen[int(n)].add(cell.dataset)
    _ = min_graphs
    return {n: len(v) for n, v in sorted(seen.items())}


def unlabelled_floor(cells: list[Cell], *, min_graphs: int = MIN_GRAPHS) -> list[tuple[int, float]]:
    """Return the information-theoretic floor per node count.

    An unlabelled simple graph on ``n`` nodes with ``m`` edges cannot be
    encoded in fewer than ``log2 |U(n, m)|`` bits, and by orbit counting
    ``|U(n, m)| >= C(T, m) / n!`` with ``T = n(n-1)/2``. The bound is a
    genuine lower bound rather than an estimate: the orbit of a symmetric
    graph is smaller than ``n!``, so dividing by ``n!`` can only understate
    the count. It is evaluated at the cohort's **median** ``m`` for each
    ``n``, so it is a floor for the typical graph at that size, not for
    every graph.

    Args:
        cells: Encoding cells; edge counts are read from the adjacency arm,
            which never refuses a graph.
        min_graphs: Node counts backed by fewer graphs are dropped. Without
            this the median edge count at the sparse tail jumps between
            single graphs and the floor draws as a picket fence.

    Returns:
        ``(n, floor_bits)`` pairs, ascending in ``n``, clipped at zero.
    """
    edges: dict[int, list[int]] = defaultdict(list)
    for cell in cells:
        if cell.representation != "adjacency":
            continue
        for n, m in zip(cell.node_counts, cell.edge_counts, strict=True):
            edges[int(n)].append(int(m))
    out: list[tuple[int, float]] = []
    for n in sorted(edges):
        if len(edges[n]) < min_graphs:
            continue
        m = int(np.median(edges[n]))
        triangle = n * (n - 1) // 2
        if not 0 <= m <= triangle:
            continue
        log2_choose = (
            math.lgamma(triangle + 1) - math.lgamma(m + 1) - math.lgamma(triangle - m + 1)
        ) / math.log(2)
        log2_factorial = math.lgamma(n + 1) / math.log(2)
        out.append((n, max(log2_choose - log2_factorial, 0.0)))
    return out


def paired_relative_gap(
    cells: list[Cell],
    *,
    reference: str,
    convention: str = "entropy_bits",
    min_graphs: int = MIN_GRAPHS,
) -> dict[str, list[tuple[int, float, int]]]:
    """Pair *reference* against every other backend and report the relative gap.

    For each graph encoded by both arms, ``100 * (1 - b_ref / b_comp)`` is the
    percentage of the competitor's message the reference saves; negative means
    the reference is longer. Pairing is on ``graph_id`` inside a dataset, so
    the quantity is immune to the composition drift that an absolute pooled
    median suffers -- the same graphs are on both sides of every ratio.

    Args:
        cells: Encoding cells.
        reference: Backend to measure from, normally
            ``design.REFERENCE_KEY``.
        convention: ``entropy_bits`` or ``realised_bits``.
        min_graphs: Strata below this many paired graphs are dropped.

    Returns:
        ``{competitor: [(n, median relative gap %, paired graphs), ...]}``,
        ascending in ``n``.

    Raises:
        ValueError: If *convention* is not one of :data:`CONVENTIONS`.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}, got {convention!r}")
    by_cell = {(c.suite, c.dataset, c.representation): c for c in cells}
    gaps: dict[tuple[str, int], list[float]] = defaultdict(list)
    for cell in cells:
        if cell.representation == reference:
            continue
        ref = by_cell.get((cell.suite, cell.dataset, reference))
        if ref is None:
            continue
        position = {gid: i for i, gid in enumerate(ref.graph_ids)}
        pairs = [(position[g], j) for j, g in enumerate(cell.graph_ids) if g in position]
        if not pairs:
            continue
        ri = np.fromiter((a for a, _ in pairs), dtype=int, count=len(pairs))
        ci = np.fromiter((b for _, b in pairs), dtype=int, count=len(pairs))
        keep = ref.usable[ri] & cell.usable[ci]
        ri, ci = ri[keep], ci[keep]
        b_ref = getattr(ref, convention)[ri]
        b_cmp = getattr(cell, convention)[ci]
        live = b_cmp > 0
        relative = 100.0 * (1.0 - b_ref[live] / b_cmp[live])
        for n, value in zip(ref.node_counts[ri][live], relative, strict=True):
            gaps[(cell.representation, int(n))].append(float(value))
    out: dict[str, list[tuple[int, float, int]]] = defaultdict(list)
    for (competitor, n), values in sorted(gaps.items()):
        if len(values) < min_graphs:
            continue
        out[competitor].append((n, float(np.median(values)), len(values)))
    return dict(out)


def load_json(path: Path) -> dict[str, Any]:
    """Load one T-06 result JSON.

    Args:
        path: The file.

    Returns:
        Its parsed contents.
    """
    return cast("dict[str, Any]", json.loads(Path(path).read_text()))


def benjamini_hochberg(p_values: list[float], q: float) -> list[bool]:
    """Return the BH rejection mask at level *q*.

    Args:
        p_values: Uncorrected p-values.
        q: False-discovery rate.

    Returns:
        One boolean per input, in input order.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    ranked = np.asarray(p_values, dtype=float)[order]
    passed = ranked <= q * (np.arange(1, m + 1) / m)
    cutoff = int(np.max(np.flatnonzero(passed)) + 1) if passed.any() else 0
    mask = np.zeros(m, dtype=bool)
    if cutoff:
        mask[order[:cutoff]] = True
    return [bool(v) for v in mask]


__all__ = [
    "CONVENTIONS",
    "MIN_GRAPHS",
    "Aggregate",
    "Cell",
    "aggregate_bits",
    "benjamini_hochberg",
    "dataset_support",
    "load_cells",
    "load_json",
    "paired_relative_gap",
    "unlabelled_floor",
]
