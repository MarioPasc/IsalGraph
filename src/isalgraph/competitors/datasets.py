"""Real-cohort loading.  **Graphs only -- this module never touches GED.**

That restriction is structural, not stylistic.  Decision 24 rests on
T-04a's exclusion rule being F5-blind *by construction*: ties break on cost,
never on correlation with GED.  Prose cannot enforce that, so the import
graph does:

- ``grid.py`` imports this module and computes F1-F4 and F6.
- ``f5.py`` imports :mod:`isalgraph.competitors.ged_reference` and is the
  only entry point that can see a GED value.
- A test asserts ``grid.py``'s import closure never reaches a GED loader.

If a GED import ever appears here, the selection tool becomes able to see
the outcome it selects on, and decision 24 stops being defensible.  Do not
add one.

Both suites use the same CSR ``.npz`` layout, produced by T-01's locked
filter (``min_nodes = 2``, ``require_connected``, cxl enumeration):
``n_nodes``, ``edge_offsets``, ``edges``, ``graph_ids``.
"""

from __future__ import annotations

import functools
import json
import os
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import networkx as nx

#: Where the exported cohorts live.  Overridable so a cluster run and a
#: workstation run read the same code.
ENV_ROOT = "ISALGRAPH_COHORT_ROOT"
DEFAULT_ROOT = "/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data"

#: Suite 1: node counts 2-12, and the only suite with certified exact GED.
SUITE1: tuple[str, ...] = (
    "iam_letter_low",
    "iam_letter_med",
    "iam_letter_high",
    "linux",
    "aids",
)

#: Suite 2: node counts to 98.  No exact-GED reference exists above n = 12.
SUITE2: tuple[str, ...] = (
    "grec",
    "aids_iam",
    "coil_del",
    "mutagenicity",
    "protein",
)

ALL_DATASETS: tuple[str, ...] = SUITE1 + SUITE2

#: Subdirectory of the root holding each suite's ``.npz`` files.
_SUBDIR = {"suite1": "exported", "suite2": "exported_suite2"}

#: ``competitors.md`` §3.1's six node-count strata, frozen.  The bins are
#: closed on both ends and the *first* matching bin wins, so they must stay
#: disjoint and ascending.  A graph with ``n < 2`` falls outside every bin;
#: T-01's locked filter (``min_nodes = 2``) means none exists, and the guard
#: is a guard rather than a case.
STRATA: tuple[tuple[int, int], ...] = ((2, 5), (6, 9), (10, 12), (13, 20), (21, 40), (41, 10**9))


class DatasetNotFoundError(FileNotFoundError):
    """Raised when a cohort file is absent, naming where it was looked for."""


@dataclass(frozen=True, slots=True)
class Cohort:
    """One dataset's graphs, in the order the exporter wrote them."""

    name: str
    suite: str
    graphs: tuple[nx.Graph, ...]
    graph_ids: tuple[object, ...]

    def __len__(self) -> int:
        return len(self.graphs)

    def sample(self, k: int, *, seed: int) -> tuple[int, ...]:
        """Indices of a ``k``-graph draw, sorted.

        A fresh ``random.Random(seed)`` per call, so a sample is a function
        of ``(dataset, k, seed)`` alone and never of what ran before it.
        The reproduction gate deliberately does **not** use this -- replaying
        the scout's stream needs the draws interleaved exactly as their
        script made them.
        """
        rng = random.Random(seed)
        return tuple(sorted(rng.sample(range(len(self.graphs)), min(k, len(self.graphs)))))


@dataclass(frozen=True, slots=True)
class SampleRecord:
    """One drawn graph, carrying everything a consumer needs without a reload.

    ``(dataset, index)`` is the address: ``load(dataset).graphs[index]``.  The
    node count, stratum and suite are carried rather than re-derived so that
    a report can group a draw without touching the cohorts, and so that the
    grid's F0 block can be split per suite from the sample alone.

    Attributes:
        dataset: a member of :data:`ALL_DATASETS`.
        index: index into ``load(dataset).graphs``.
        n_nodes: order of that graph.
        stratum: index into :data:`STRATA`.
        suite: ``"suite1"`` or ``"suite2"``, from :func:`suite_of`.
    """

    dataset: str
    index: int
    n_nodes: int
    stratum: int
    suite: str


def cohort_root() -> str:
    """Root directory for the exported cohorts."""
    return os.environ.get(ENV_ROOT, DEFAULT_ROOT)


def suite_of(dataset: str) -> str:
    """``"suite1"`` or ``"suite2"``."""
    if dataset in SUITE1:
        return "suite1"
    if dataset in SUITE2:
        return "suite2"
    raise DatasetNotFoundError(f"unknown dataset {dataset!r}; known: {list(ALL_DATASETS)}")


def _npz_path(dataset: str) -> str:
    suite = suite_of(dataset)
    return os.path.join(cohort_root(), _SUBDIR[suite], f"{dataset}.npz")


@functools.cache
def load(dataset: str) -> Cohort:
    """Load one dataset's graphs.  Cached, since every entry point reloads.

    Raises:
        DatasetNotFoundError: naming the path searched, so a missing cohort
            is one message rather than a numpy traceback.
    """
    import networkx as nx
    import numpy as np

    path = _npz_path(dataset)
    if not os.path.exists(path):
        raise DatasetNotFoundError(
            f"cohort {dataset!r} not found at {path}. Set ${ENV_ROOT} if the "
            f"exported cohorts live elsewhere"
        )
    data = np.load(path, allow_pickle=True)
    offsets, edges = data["edge_offsets"], data["edges"]
    graphs = []
    for i, n in enumerate(data["n_nodes"]):
        graph = nx.Graph()
        graph.add_nodes_from(range(int(n)))
        lo, hi = offsets[i], offsets[i + 1]
        graph.add_edges_from(zip(edges[0, lo:hi].tolist(), edges[1, lo:hi].tolist(), strict=True))
        graphs.append(graph)
    return Cohort(
        name=dataset,
        suite=suite_of(dataset),
        graphs=tuple(graphs),
        graph_ids=tuple(data["graph_ids"].tolist()),
    )


def available_datasets() -> tuple[str, ...]:
    """Datasets whose ``.npz`` is actually present under the current root."""
    return tuple(ds for ds in ALL_DATASETS if os.path.exists(_npz_path(ds)))


def manifest(suite: str) -> dict[str, Any]:
    """The exporter's manifest for a suite, for provenance in a run header."""
    path = os.path.join(cohort_root(), _SUBDIR[suite], "manifest.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return dict(json.load(handle))


def stratum_of(n_nodes: int) -> int | None:
    """Index into :data:`STRATA`, or ``None`` when *n_nodes* is outside them all.

    Args:
        n_nodes: order of the graph.

    Returns:
        The index of the first matching stratum, or ``None``.
    """
    for index, (lo, hi) in enumerate(STRATA):
        if lo <= n_nodes <= hi:
            return index
    return None


def stratum_quotas(k: int, n_populated: int) -> tuple[int, ...]:
    """``k`` split over *n_populated* strata, **remainder to the largest**.

    ``divmod(k, n_populated)`` gives a floor quota and a remainder; the
    remainder is handed to the *last* ``rem`` strata, which are the largest
    by node count.  That preserves the rationale the deleted proportional
    sampler stated: the tail, where the ``m``-scaling argument lives, must not
    be rounded away.  For ``(200, 6)`` this is ``(33, 33, 33, 33, 34, 34)``
    and for ``(50, 6)`` it is ``(8, 8, 8, 8, 9, 9)``.

    Args:
        k: total draw size.
        n_populated: number of strata with at least one member.

    Returns:
        One quota per populated stratum, in ascending stratum order.
    """
    if n_populated <= 0:
        return ()
    base, rem = divmod(k, n_populated)
    return tuple(base + (1 if i >= n_populated - rem else 0) for i in range(n_populated))


def stratified_subsample(
    records: Sequence[SampleRecord],
    k: int,
    *,
    seed: int,
    order: Sequence[str] = ALL_DATASETS,
) -> tuple[SampleRecord, ...]:
    """Draw ``k`` of *records*, balanced over :data:`STRATA`.

    The allocation arithmetic of :func:`pooled_stratified_sample`, applied to
    an existing record list rather than to the cohorts on disk.  T-04a's F3
    sub-sample ``S50`` is exactly this call on ``S200``, so the two draws
    share one implementation and cannot drift apart.

    A stratum with fewer members than its quota contributes all of them; the
    shortfall is **not** redistributed, so a draw can return fewer than ``k``
    records and the caller reports it.

    Args:
        records: the pool to draw from.
        k: draw size.
        seed: seed of the single :class:`random.Random` consumed stratum by
            stratum in ascending order.
        order: dataset ordering used to make each stratum's candidate list
            deterministic.  A dataset absent from *order* sorts last.

    Returns:
        The draw, sorted by ``(stratum, order.index(dataset), index)``.
    """
    rank = {name: i for i, name in enumerate(order)}
    unranked = len(rank)

    def key(record: SampleRecord) -> tuple[int, int]:
        return (rank.get(record.dataset, unranked), record.index)

    binned: list[list[SampleRecord]] = [[] for _ in STRATA]
    for record in records:
        binned[record.stratum].append(record)
    for bucket in binned:
        bucket.sort(key=key)

    populated = [bucket for bucket in binned if bucket]
    quotas = stratum_quotas(k, len(populated))
    rng = random.Random(seed)
    picked: list[SampleRecord] = []
    for bucket, quota in zip(populated, quotas, strict=True):
        picked.extend(rng.sample(bucket, min(quota, len(bucket))))
    picked.sort(key=lambda r: (r.stratum, *key(r)))
    return tuple(picked)


def pooled_stratified_sample(
    names: Sequence[str], k: int, *, seed: int
) -> tuple[SampleRecord, ...]:
    """``competitors.md`` §3.1's sample: **one pooled draw, stratum-balanced**.

    Every graph of *names* is pooled and binned by node count into
    :data:`STRATA`; each populated stratum then contributes its
    :func:`stratum_quotas` share.  This is *not* the deleted
    ``stratified_sample``, which drew ``k`` graphs **per dataset**
    (measured: 1,889 graphs for ``k = 200``) and allocated proportionally
    *within* a dataset, so it reproduced each dataset's own size distribution
    instead of spreading the draw over the six strata §3.1 names.

    The result is a pure function of ``(tuple(names), k, seed)`` and of
    nothing that ran before it.

    Args:
        names: datasets to pool, in the order that breaks candidate ties.
        k: draw size.
        seed: seed of the single :class:`random.Random`.

    Returns:
        The draw, sorted by ``(stratum, names.index(dataset), index)``.

    Raises:
        DatasetNotFoundError: if any of *names* is not on disk.
    """
    records: list[SampleRecord] = []
    for name in names:
        cohort = load(name)
        for index, graph in enumerate(cohort.graphs):
            n_nodes = int(graph.number_of_nodes())
            stratum = stratum_of(n_nodes)
            if stratum is None:
                continue
            records.append(
                SampleRecord(
                    dataset=name,
                    index=index,
                    n_nodes=n_nodes,
                    stratum=stratum,
                    suite=cohort.suite,
                )
            )
    return stratified_subsample(records, k, seed=seed, order=tuple(names))


__all__ = [
    "ALL_DATASETS",
    "STRATA",
    "SUITE1",
    "SUITE2",
    "Cohort",
    "DatasetNotFoundError",
    "SampleRecord",
    "available_datasets",
    "cohort_root",
    "load",
    "manifest",
    "pooled_stratified_sample",
    "stratified_subsample",
    "stratum_of",
    "stratum_quotas",
    "suite_of",
]
