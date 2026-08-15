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


def stratified_sample(datasets: Sequence[str], k: int, *, seed: int) -> dict[str, tuple[int, ...]]:
    """A ``k``-graph draw per dataset, stratified by node count.

    Strata are ``competitors.md`` §3.1's: ``[2,5]``, ``[6,9]``, ``[10,12]``,
    ``[13,20]``, ``[21,40]``, ``>40``, so the unequal-``n`` case dominates
    exactly as it does in production.  Strata are filled proportionally and
    the remainder goes to the largest strata first, which keeps the tail --
    where the ``m``-scaling argument lives -- from being rounded away.
    """
    bounds = ((2, 5), (6, 9), (10, 12), (13, 20), (21, 40), (41, 10**9))
    out: dict[str, tuple[int, ...]] = {}
    for dataset in datasets:
        cohort = load(dataset)
        rng = random.Random(seed)
        strata: list[list[int]] = [[] for _ in bounds]
        for idx, graph in enumerate(cohort.graphs):
            n = graph.number_of_nodes()
            for s, (lo, hi) in enumerate(bounds):
                if lo <= n <= hi:
                    strata[s].append(idx)
                    break
        populated = [s for s in strata if s]
        if not populated:
            out[dataset] = ()
            continue
        total = sum(len(s) for s in populated)
        picked: list[int] = []
        quotas = [min(len(s), round(k * len(s) / total)) for s in populated]
        for stratum, quota in zip(populated, quotas, strict=True):
            picked.extend(rng.sample(stratum, quota))
        # Top up from the largest strata first if rounding lost a graph.
        if len(picked) < min(k, total):
            spare = [i for s in reversed(populated) for i in s if i not in set(picked)]
            picked.extend(spare[: min(k, total) - len(picked)])
        out[dataset] = tuple(sorted(picked[:k]))
    return out


__all__ = [
    "ALL_DATASETS",
    "SUITE1",
    "SUITE2",
    "Cohort",
    "DatasetNotFoundError",
    "available_datasets",
    "cohort_root",
    "load",
    "manifest",
    "stratified_sample",
    "suite_of",
]
