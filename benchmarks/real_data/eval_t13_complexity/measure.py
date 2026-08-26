"""T-13 controlled-experiment runner: time every representation on every graph.

The instrument behind R3.7d.  The manuscript currently says a canonical search
is "exponential in the worst case" without saying *when*; T-13 replaces that
with a characterised claim -- the cost is governed by ``|Aut(G)|``, not by size
or density -- and this module produces the measurements the claim is fitted on.
One immutable :class:`~.schema.Record` per ``(graph, representation, arm)``.

**The timing rule lives here and nowhere else** (CONTRACTS §5.3).  It is not a
detail: on the real IAM cohort the marginal correlation between ``log|Aut|``
and ``log t`` is only +0.189, against +0.326 for ``log n``, and the effect
appears solely in the within-fixed-``(n, m)`` contrast (+0.655, positive in 12
of 13 strata).  An effect that small does not survive a sloppy clock.

Four decisions carry the design, each with a scar behind it.

1. **``time.process_time``, not wall clock.**  Wall clock on a shared node
   measures the neighbours.  ``process_time`` is process-wide CPU across all
   threads, which is why the child pins every thread pool to one -- a 4-thread
   engine would report four times the CPU for the same work.

2. **The budget is enforced by killing a subprocess, never by ``SIGALRM``.**
   T-05 finding 5: ``SIGALRM`` does not interrupt the C++ engine, so a
   signal-based timeout silently fails to fire and the job runs to the
   wallclock.  The parent's kill is the one mechanism that covers every
   backend.  The IsalGraph arms additionally receive the engine's own
   ``timeout_s`` so they can stop cleanly at the same budget; both paths
   produce an identical censored record.

3. **One fresh subprocess per work unit**, which is isolation rather than
   overhead.  A persistent worker would carry the engine's pair-memoisation
   cache across units, so a graph's measured cost would depend on which graphs
   preceded it in the shard -- and the shard order is a hash.  At the design
   note's 400-700 core-h estimate the ~0.6 s of interpreter startup per unit is
   under 1 % and buys a measurement that does not depend on its own ordering.

4. **Nothing is dropped.**  A backend that declines a graph is recorded
   ``unsupported``; a budget that expires is recorded ``censored`` with the
   mechanism named.  The censored cells *are* the high-``|Aut|`` cells, so
   dropping them would delete the result this ticket exists to obtain.

Sharding is a deterministic hash of the work-unit key, so a shard's membership
never depends on iteration order and a re-run of shard *K* is the same shard.
Single-threaded shards, one per dedicated core, are the measured optimum for
this workload: a prior ticket's parallelisation was *negative-scaling*
(1 worker 36 core-s, 4 -> 212, 15 -> 928, 32 -> 5,260 on identical work).

CLI::

    python -m benchmarks.eval_t13_complexity.measure \\
        --source constructed|cohort --shard K --n-shards N \\
        [--dataset D] --arms default[,no_pairs_memo,...] \\
        --budget-s 300 --seed 13 --out <path>

``families.py`` and ``symmetry.py`` belong to track A.  This module imports
them lazily and degrades explicitly: a missing ``symmetry`` nulls the nine
symmetry fields and stamps ``symmetry_available: false`` in the shard header,
and a missing ``families`` refuses ``--source constructed`` by name.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    import networkx as nx

from benchmarks.real_data.eval_t13_complexity import schema
from benchmarks.real_data.eval_t13_complexity.schema import Record

LOGGER = logging.getLogger("t13.measure")

# ---------------------------------------------------------------------------
# Frozen constants.  Every one of these is a value the analysis quotes.
# ---------------------------------------------------------------------------

#: The engine build this campaign is valid for.  A shard whose build differs is
#: **aborted, not warned**: two shards from different builds cannot be pooled,
#: and a warning in a 4,000-line SLURM log is a warning nobody reads.
EXPECTED_BUILD_HASH: Final = "298fc1188bf1b051"

#: Wall clock per ``(graph, representation, arm)``.  Matches D14 and T-06.
DEFAULT_BUDGET_S: Final = 300.0

#: Extra seconds the parent allows a child beyond the budget before killing it.
#: Covers interpreter startup and the three sub-second repeats, both of which
#: sit outside the budgeted call.  Generous on purpose: a grace that is too
#: tight turns a completed measurement into a fabricated censoring.
KILL_GRACE_S: Final = 30.0

#: A warm-up at or above this many seconds is reported alone, with
#: ``repeats = 1``.  Relative noise is negligible once a run is seconds long,
#: and three repeats of a 200 s encode would blow the budget by construction.
WARMUP_THRESHOLD_S: Final = 1.0

#: Timed runs after the warm-up when the warm-up came in under the threshold.
REPEATS: Final = 3

#: Search-based arms: a canonical form obtained by searching.  The cost law is
#: a claim about this class, which is why all eight run.
SEARCH_BASED: Final[tuple[str, ...]] = (
    "isalgraph_canonical",
    "isalgraph_exhaustive",
    "isalgraph_pruned",
    "isalgraph_greedy",
    "nauty_graph6",
    "sparse6_nauty",
    "min_dfs",
    "agm_cam",
)

#: Search-free controls.  Their cost must be Theta(n^2) or Theta(n + m) and
#: **flat in ``|Aut|``**.  Without them the cost law has no null: a rising
#: curve on the search arms alone could be any confound that tracks symmetry.
SEARCH_FREE: Final[tuple[str, ...]] = (
    "adjacency",
    "graph6",
    "sparse6",
    "wl_subtree",
    "size_null",
)

#: The thirteen registered arms, frozen.
#:
#: ``size_null`` carries ``Capability.BASELINE`` and is therefore **absent from
#: ``available_backends()``** -- it is returned only when named.  Discovering
#: the list instead of naming it would silently drop the null arm, and the
#: figure would regenerate successfully with it missing.
REPRESENTATIONS: Final[tuple[str, ...]] = SEARCH_BASED + SEARCH_FREE

#: Arms whose cost the two native toggles can actually change.  The toggles
#: gate the canonical search's pair memoisation and its branch-and-bound lower
#: bound, so an ablation of a serialisation or of the greedy encoder would
#: measure nothing and cost a full budget to find out.
ABLATABLE_REPRESENTATIONS: Final[tuple[str, ...]] = (
    "isalgraph_canonical",
    "isalgraph_exhaustive",
    "isalgraph_pruned",
)

#: Graphs per stratum carried into the ablation arms.  **Fixed here, in code,
#: before any result exists** (CONTRACTS §5.3.5), and selected by hash rank
#: within the stratum so the choice cannot depend on iteration order or on
#: which cells turned out interesting.
ABLATION_PER_STRATUM: Final = 2

#: AGM's frozen branch-and-bound node budgets, mirrored from
#: ``competitors.backends.agm``.  Suite 1 is the default there.
AGM_SEARCH_NODES_SUITE1: Final = 200_000
AGM_SEARCH_NODES_SUITE2: Final = 100_000

#: min-DFS's frozen retained-embedding cap, mirrored from
#: ``competitors.backends.min_dfs``.  A **memory** guard: the first Suite-2 run
#: was OOM-killed, not slow, so it must stay set on every call.
MIN_DFS_MAX_PROJECTIONS: Final = 50_000

#: Node count at or above which the Suite-2 AGM budget applies.  The
#: constructed grid has no suite, so the rule is expressed on ``n``, matching
#: the ``SUITE1_MAX_NODES = 12`` boundary the backends already use.
SUITE1_MAX_NODES: Final = 12

#: Environment the child runs under.  ``process_time`` sums CPU across threads,
#: so an unpinned BLAS would inflate every reading by its thread count.
#: ``ISALGRAPH_THREADS`` is set because CONTRACTS §5.3.1 names it; note that no
#: code under ``src/isalgraph/`` reads it -- the engine's own default is 1.
CHILD_THREAD_ENV: Final[Mapping[str, str]] = {
    "ISALGRAPH_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


class MeasureError(RuntimeError):
    """Base class for the runner's own faults."""


class EngineMismatchError(MeasureError):
    """The active engine is not the one this campaign was frozen against."""


class TrackAMissingError(MeasureError):
    """``families.py`` or ``symmetry.py`` is absent from this checkout."""


# ---------------------------------------------------------------------------
# Engine gate
# ---------------------------------------------------------------------------


def assert_engine(*, expected_build_hash: str = EXPECTED_BUILD_HASH) -> dict[str, Any]:
    """Abort unless the C++ engine is active at the frozen build.

    Not a warning.  A pure-Python fallback is 23x-1025x slower and would be
    reported as a timing; a different build is a different set of optimisations
    and cannot be pooled with the rest of the campaign.  Both failures are
    silent, which is exactly why T-06's headline rates were retracted as
    unprovenanced.

    Args:
        expected_build_hash: the frozen build hash.

    Returns:
        ``isalgraph.build_info()``, for the shard header.

    Raises:
        EngineMismatchError: when the engine is not ``"cpp"`` or the build hash
            differs.
    """
    import isalgraph

    engine = isalgraph.engine()
    info = dict(isalgraph.build_info())
    if engine != "cpp":
        raise EngineMismatchError(
            f"engine is {engine!r}, not 'cpp'. Every timing would be from the "
            f"pure-Python reference, which is 23x-1025x slower. If you exported "
            f"PYTHONPATH=<repo>/src, unset it: a src-first path shadows the editable "
            f"install and falls back silently"
        )
    actual = info.get("build_hash")
    if actual != expected_build_hash:
        raise EngineMismatchError(
            f"build_hash is {actual!r}, expected {expected_build_hash!r}. Shards from "
            f"two builds cannot be pooled. Rebuild the extension, or re-freeze the "
            f"campaign against this build and re-run every shard"
        )
    return info


# ---------------------------------------------------------------------------
# Representation resolution
# ---------------------------------------------------------------------------


def resolve_representations(
    names: Sequence[str] = REPRESENTATIONS,
) -> dict[str, str]:
    """Resolve every representation through the registry, reporting each.

    Args:
        names: registry keys to resolve.  Defaults to :data:`REPRESENTATIONS`.

    Returns:
        ``name -> status``, where a resolved backend reports ``"ok
        (ReprBackend)"`` or ``"ok (VectorBackend)"`` and an unresolved one
        reports ``"UNRESOLVED: <ExcType>: <message>"``.  A mapping rather than
        a raise, so one missing optional dependency does not hide the other
        twelve.
    """
    from isalgraph.competitors.base import ReprBackend
    from isalgraph.competitors.registry import get_backend

    out: dict[str, str] = {}
    for name in names:
        try:
            backend = get_backend(name)
        except Exception as exc:  # noqa: BLE001 - the report is the point
            out[name] = f"UNRESOLVED: {type(exc).__name__}: {exc}"
            continue
        kind = "ReprBackend" if isinstance(backend, ReprBackend) else "VectorBackend"
        out[name] = f"ok ({kind})"
    return out


# ---------------------------------------------------------------------------
# The timing rule.  CONTRACTS §5.3.2.  Implemented once, here.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Timing:
    """The outcome of applying the frozen timing rule to one callable.

    Attributes:
        seconds: the reported time -- the warm-up when it reached the
            threshold, otherwise the median of :data:`REPEATS` further runs.
        repeats: ``1`` or :data:`REPEATS`, matching which branch was taken.
        warmup_seconds: the warm-up on its own, kept so the analysis can check
            that the two branches agree where they overlap.
        value: whatever the callable returned on its last run.
    """

    seconds: float
    repeats: int
    warmup_seconds: float
    value: int


def timed_call(
    call: Callable[[], int],
    *,
    clock: Callable[[], float] = time.process_time,
    threshold_s: float = WARMUP_THRESHOLD_S,
    repeats: int = REPEATS,
) -> Timing:
    """Apply the frozen timing rule to *call*.

    One warm-up.  If it took ``>= threshold_s``, that single reading is the
    answer.  Otherwise *repeats* further runs and the **median** -- median, not
    mean, because a stray page fault or a scheduler preemption is a right-tail
    event and the mean would carry it into the fit.

    Args:
        call: the work to time.  Returns the encoding's length, which is
            carried through so the caller need not re-run it.
        clock: injected for testing.  Production is ``time.process_time``.
        threshold_s: warm-up duration at or above which no repeats are run.
        repeats: number of timed runs when the warm-up was fast.

    Returns:
        The :class:`Timing`.
    """
    start = clock()
    value = call()
    warmup = clock() - start
    if warmup >= threshold_s:
        return Timing(seconds=warmup, repeats=1, warmup_seconds=warmup, value=value)

    samples: list[float] = []
    for _ in range(repeats):
        start = clock()
        value = call()
        samples.append(clock() - start)
    return Timing(
        seconds=statistics.median(samples),
        repeats=repeats,
        warmup_seconds=warmup,
        value=value,
    )


# ---------------------------------------------------------------------------
# Engine ablation arms
# ---------------------------------------------------------------------------


class _NativeToggles(Protocol):
    """The four native entry points the ablation arms touch."""

    def set_pairs_memo(self, on: bool, /) -> None: ...  # noqa: D102

    def set_branch_and_bound(self, on: bool, /) -> None: ...  # noqa: D102


def arm_settings(arm: str) -> tuple[bool, bool]:
    """Return ``(pairs_memo, branch_and_bound)`` for *arm*.

    Args:
        arm: one of :data:`schema.ARMS`.

    Returns:
        The two toggle states.

    Raises:
        ValueError: on an unknown arm.
    """
    if arm not in schema.ARMS:
        raise ValueError(f"unknown arm {arm!r}; known: {list(schema.ARMS)}")
    return (
        arm not in ("no_pairs_memo", "no_pairs_memo_no_bnb"),
        arm not in ("no_bnb", "no_pairs_memo_no_bnb"),
    )


@contextlib.contextmanager
def engine_arm(arm: str, *, native: _NativeToggles | None = None) -> Iterator[None]:
    """Run the body with the engine configured for *arm*, then restore.

    **Both toggles are restored to ``True`` in a ``finally``, including when
    the timed call raises**, which it routinely does: a censored encode leaves
    the block by exception, and an ablation arm that leaked its state would
    silently ablate every subsequent unit in the process.  A fresh subprocess
    per unit makes that leak impossible in production; the ``finally`` makes it
    impossible in a test, in the CLI's single-unit mode, and in anything a
    later ticket writes.

    Args:
        arm: one of :data:`schema.ARMS`.
        native: injected for testing.  Production is
            ``isalgraph.core._native``.

    Yields:
        ``None``.

    Raises:
        ValueError: on an unknown arm.
    """
    memo, bnb = arm_settings(arm)
    if native is None:
        from isalgraph.core import _native as native_module

        native = native_module  # type: ignore[assignment]
    assert native is not None
    try:
        native.set_pairs_memo(memo)
        native.set_branch_and_bound(bnb)
        yield
    finally:
        native.set_pairs_memo(True)
        native.set_branch_and_bound(True)


# ---------------------------------------------------------------------------
# Deterministic sharding
# ---------------------------------------------------------------------------


def unit_digest(key: str) -> int:
    """Return a stable 64-bit digest of *key*.

    ``blake2b`` rather than ``hash()``: Python's string hash is salted per
    process, so a shard defined by it would differ between the launcher and the
    worker, and between two workers of the same array.

    Args:
        key: the work-unit or graph key.

    Returns:
        An integer in ``[0, 2**64)``.
    """
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "big")


def shard_of(key: str, n_shards: int) -> int:
    """Return the shard *key* belongs to.

    Args:
        key: the work-unit key.
        n_shards: total shards, at least 1.

    Returns:
        An index in ``[0, n_shards)``.

    Raises:
        ValueError: when *n_shards* is below 1.
    """
    if n_shards < 1:
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")
    return unit_digest(key) % n_shards


# ---------------------------------------------------------------------------
# Work units
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GraphSpec:
    """The address of one graph, independent of how it is measured.

    ``source`` selects which half of the address is populated: a constructed
    graph carries ``family``/``n_target``/``replicate``/``params``, a cohort
    graph carries ``dataset``/``graph_index``/``graph_id``.
    """

    source: str
    family: str | None = None
    n_target: int | None = None
    replicate: int | None = None
    params: tuple[tuple[str, int], ...] = ()
    dataset: str | None = None
    graph_index: int | None = None
    graph_id: str | None = None

    @property
    def key(self) -> str:
        """A stable string address, used for hashing and for resume."""
        params = ",".join(f"{k}={v}" for k, v in self.params)
        return (
            f"{self.source}|{self.family}|{self.n_target}|{self.replicate}|{params}"
            f"|{self.dataset}|{self.graph_index}"
        )


@dataclass(frozen=True, slots=True)
class WorkUnit:
    """One ``(graph, representation, arm)`` cell: exactly one output record."""

    graph: GraphSpec
    representation: str
    arm: str

    @property
    def key(self) -> str:
        """A stable string address for hashing, sharding and resume."""
        return f"{self.graph.key}|{self.representation}|{self.arm}"


def ablation_stratum(spec: GraphSpec, n_nodes: int) -> str:
    """Return the stratum *spec* is sampled within for the ablation arms.

    Stratifying on ``(source, family-or-dataset, n)`` keeps the ablation
    subsample spread across the whole grid rather than concentrated wherever
    the hash happened to land, which matters because the two toggles are
    expected to bite only in the high-``|Aut|`` cells.

    Args:
        spec: the graph's address.
        n_nodes: the realised order.

    Returns:
        The stratum key.
    """
    return f"{spec.source}|{spec.family or spec.dataset}|{n_nodes}"


def select_ablation_graphs(
    specs: Sequence[tuple[GraphSpec, int]],
    *,
    per_stratum: int = ABLATION_PER_STRATUM,
) -> frozenset[str]:
    """Choose the graphs that carry the ablation arms.

    The rule is fixed before any result exists: within each stratum, rank the
    graphs by :func:`unit_digest` and take the first *per_stratum*.  Rank by
    digest rather than by position so the subsample cannot move when the
    enumeration order changes, and cannot be steered by a later edit.

    Args:
        specs: ``(spec, n_nodes)`` for every graph in the grid.
        per_stratum: how many graphs each stratum contributes.

    Returns:
        The selected graph keys.
    """
    buckets: dict[str, list[tuple[int, str]]] = {}
    for spec, n_nodes in specs:
        buckets.setdefault(ablation_stratum(spec, n_nodes), []).append(
            (unit_digest(spec.key), spec.key)
        )
    chosen: set[str] = set()
    for ranked in buckets.values():
        ranked.sort()
        chosen.update(key for _digest, key in ranked[:per_stratum])
    return frozenset(chosen)


def units_for_graph(
    spec: GraphSpec,
    *,
    representations: Sequence[str],
    arms: Sequence[str],
    ablation_keys: frozenset[str],
) -> tuple[WorkUnit, ...]:
    """Return every work unit for one graph.

    ``default`` runs on every representation.  An ablation arm runs only on a
    representation the toggles can change (:data:`ABLATABLE_REPRESENTATIONS`)
    and only on a graph in the stratified subsample -- otherwise the 2x2 would
    cost four full budgets per cell to measure nothing.

    Args:
        spec: the graph's address.
        representations: registry keys.
        arms: arms requested on the command line.
        ablation_keys: output of :func:`select_ablation_graphs`.

    Returns:
        The units, in a deterministic order.
    """
    units: list[WorkUnit] = []
    for representation in representations:
        for arm in arms:
            if arm == "default":
                units.append(WorkUnit(spec, representation, arm))
                continue
            if representation not in ABLATABLE_REPRESENTATIONS:
                continue
            if spec.key not in ablation_keys:
                continue
            units.append(WorkUnit(spec, representation, arm))
    return tuple(units)


# ---------------------------------------------------------------------------
# Grid enumeration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GridEntry:
    """One graph of the grid, with the graph itself already built."""

    spec: GraphSpec
    graph: Any


def _cohort_entries(datasets: Sequence[str] | None) -> tuple[GridEntry, ...]:
    """Build the cohort half of the grid.

    Args:
        datasets: dataset names, or ``None`` for every dataset present under
            the current cohort root.

    Returns:
        One entry per graph.
    """
    from isalgraph.competitors import datasets as ds

    names = tuple(datasets) if datasets else ds.available_datasets()
    entries: list[GridEntry] = []
    for name in names:
        cohort = ds.load(name)
        for index, graph in enumerate(cohort.graphs):
            entries.append(
                GridEntry(
                    spec=GraphSpec(
                        source="cohort",
                        dataset=name,
                        graph_index=index,
                        graph_id=str(cohort.graph_ids[index]),
                    ),
                    graph=graph,
                )
            )
    return tuple(entries)


def _constructed_entries(seed: int) -> tuple[GridEntry, ...]:
    """Build the constructed half of the grid via track A's ``families``.

    Args:
        seed: the campaign seed, threaded into every random family.

    Returns:
        One entry per constructed graph.

    Raises:
        TrackAMissingError: when ``families.py`` is not in this checkout.
    """
    try:
        from benchmarks.real_data.eval_t13_complexity import families
    except ImportError as exc:  # pragma: no cover - depends on track A
        raise TrackAMissingError(
            "--source constructed needs families.py, which is track A's module and is "
            "not in this checkout. Merge track A's branch, or run --source cohort"
        ) from exc

    entries: list[GridEntry] = []
    for spec in families.enumerate_grid(
        sizes=families.SIZES, replicates=families.REPLICATES, seed=seed
    ):
        entries.append(
            GridEntry(
                spec=GraphSpec(
                    source="constructed",
                    family=spec.family,
                    n_target=spec.n,
                    replicate=spec.replicate,
                    params=tuple(spec.params),
                ),
                graph=families.build(spec, seed=seed),
            )
        )
    return tuple(entries)


def build_grid(source: str, *, datasets: Sequence[str] | None, seed: int) -> tuple[GridEntry, ...]:
    """Build every graph of the requested half of the grid.

    Args:
        source: ``"constructed"`` or ``"cohort"``.
        datasets: cohort names, ignored for the constructed source.
        seed: campaign seed.

    Returns:
        The entries.

    Raises:
        ValueError: on an unknown source.
    """
    if source == "cohort":
        return _cohort_entries(datasets)
    if source == "constructed":
        return _constructed_entries(seed)
    raise ValueError(f"unknown source {source!r}; known: {list(schema.SOURCES)}")


# ---------------------------------------------------------------------------
# Per-graph facts
# ---------------------------------------------------------------------------


def graph_properties(graph: nx.Graph) -> dict[str, Any]:
    """Return the five structural fields of the record.

    Args:
        graph: the graph.

    Returns:
        ``n``, ``m``, ``density``, ``max_degree``, ``connected``.
    """
    import networkx as nx_mod

    n = int(graph.number_of_nodes())
    m = int(graph.number_of_edges())
    degrees = [int(d) for _node, d in graph.degree()]
    return {
        "n": n,
        "m": m,
        "density": (2.0 * m) / (n * (n - 1)) if n > 1 else 0.0,
        "max_degree": max(degrees) if degrees else 0,
        "connected": bool(n > 0 and nx_mod.is_connected(graph)),
    }


def symmetry_fields(graph: nx.Graph, *, available: bool) -> dict[str, Any]:
    """Return the nine symmetry fields, or nulls when track A is absent.

    Args:
        graph: the graph.
        available: whether ``symmetry.resolution_record`` imported.

    Returns:
        A mapping over :data:`schema.SYMMETRY_FIELDS`.

    Raises:
        TrackAMissingError: when ``resolution_record`` returns a key set other
            than the nine the contract freezes.  A silently different key set
            would null a column the ``|Aut|`` regression is fitted on.
    """
    if not available:
        return dict.fromkeys(schema.SYMMETRY_FIELDS)
    from benchmarks.real_data.eval_t13_complexity import symmetry

    record = dict(symmetry.resolution_record(graph))
    if set(record) != set(schema.SYMMETRY_FIELDS):
        raise TrackAMissingError(
            f"symmetry.resolution_record returned keys {sorted(record)}, expected "
            f"{sorted(schema.SYMMETRY_FIELDS)} (CONTRACTS §2)"
        )
    return record


def symmetry_available() -> bool:
    """Whether track A's ``symmetry`` module can be imported.

    Returns:
        ``True`` when ``resolution_record`` is importable.
    """
    try:
        from benchmarks.real_data.eval_t13_complexity import symmetry
    except ImportError:
        return False
    return hasattr(symmetry, "resolution_record")


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


def budget_fields(*, n_nodes: int, budget_s: float) -> dict[str, Any]:
    """Return the fully resolved budget for a graph of *n_nodes*.

    **One fully populated budget is threaded through every backend**, which is
    what :class:`~isalgraph.competitors.base.Budget` is documented for: "only
    the fields a backend declares are read".  Each field sits at its frozen
    published value, so the object reproduces every backend's default exactly
    while making the caps explicit enough to serialise.

    Leaving a field ``None`` would not be equivalent.  ``min_dfs.encode`` reads
    ``cap = MAX_PROJECTIONS if budget is None else budget.max_projections``, so
    a budget with the field unset runs it **unbounded** -- and that cap is a
    memory guard whose first Suite-2 run was OOM-killed, not slow.

    AGM's node budget is suite-conditional and the constructed grid has no
    suite, so the rule is expressed on ``n`` at the same boundary the backends
    themselves use.

    Args:
        n_nodes: the graph's order.
        budget_s: the wall clock.

    Returns:
        ``search_nodes``, ``max_projections``, ``timeout_s``.
    """
    search_nodes = (
        AGM_SEARCH_NODES_SUITE1 if n_nodes <= SUITE1_MAX_NODES else AGM_SEARCH_NODES_SUITE2
    )
    return {
        "search_nodes": search_nodes,
        "max_projections": MIN_DFS_MAX_PROJECTIONS,
        "timeout_s": budget_s,
    }


def budget_spec(fields: Mapping[str, Any]) -> str:
    """Render a resolved budget for the record's ``budget_spec`` field.

    Args:
        fields: output of :func:`budget_fields`.

    Returns:
        ``"search_nodes=...,max_projections=...,timeout_s=..."``.
    """
    return ",".join(
        f"{key}={fields[key]}" for key in ("search_nodes", "max_projections", "timeout_s")
    )


# ---------------------------------------------------------------------------
# Child side: execute exactly one unit in this process
# ---------------------------------------------------------------------------


def _classify_exception(exc: BaseException) -> tuple[str, str]:
    """Map an encoder exception to ``(status, error_kind)``.

    The four censoring mechanisms are kept apart on purpose: a wall-clock kill
    at 300 s and a min-DFS projection cap that fires in milliseconds are both
    "the budget ran out", but pooling them puts a fabricated 300 s into a
    timing distribution the cost law is fitted on.

    Args:
        exc: the exception the encode raised.

    Returns:
        A legal ``(status, error_kind)`` pair.
    """
    from isalgraph.errors import (
        AGMBudgetExceeded,
        BackendNotFoundError,
        BackendUnavailableError,
        BudgetExceeded,
        CanonicalizationTimeoutError,
        DisconnectedGraphError,
        MinDfsBudgetExceeded,
        NotReversible,
        SuiteScopeError,
    )

    if isinstance(exc, CanonicalizationTimeoutError):
        return "censored", schema.KIND_TIMEOUT
    if isinstance(exc, MinDfsBudgetExceeded):
        return "censored", schema.KIND_MAX_PROJECTIONS
    if isinstance(exc, AGMBudgetExceeded):
        return "censored", schema.KIND_SEARCH_NODES
    if isinstance(exc, BudgetExceeded):
        # A budget class added later. Name it by its own type rather than
        # guessing a mechanism; the schema will reject the row and the reader
        # will know a mechanism is missing from CENSORING_KINDS.
        return "error", type(exc).__name__
    if isinstance(
        exc,
        SuiteScopeError
        | BackendUnavailableError
        | BackendNotFoundError
        | NotReversible
        | DisconnectedGraphError,
    ):
        return "unsupported", type(exc).__name__
    return "error", type(exc).__name__


def _make_call(representation: str, graph: nx.Graph, budget: Any) -> tuple[Callable[[], int], str]:
    """Build the timed callable for *representation* and report its fallback.

    The registry lookup and the backend's construction sit **outside** the
    returned callable: they are the harness, not the representation.  WL's
    ``fit`` sits **inside** it, because building the colour vocabulary is part
    of computing a WL representation, not part of setting up to compute one.

    Args:
        representation: registry key.
        graph: the graph to encode.
        budget: a fully populated ``Budget``.

    Returns:
        ``(call, fallback_variant)``.  The callable returns the number of
        symbols produced, which for WL is the number of distinct colours.
    """
    from isalgraph.competitors.base import ReprBackend
    from isalgraph.competitors.registry import get_backend

    backend = get_backend(representation)
    if isinstance(backend, ReprBackend):
        repr_backend = backend

        def encode_call() -> int:
            return int(repr_backend.encode(graph, budget=budget).length)

        return encode_call, str(getattr(backend, "fallback_variant", ""))

    vector_backend = backend

    def vector_call() -> int:
        # Fitting on the single graph under test is correct here and would be
        # wrong in a distance campaign: a per-batch vocabulary makes distances
        # depend on batching order, but this module computes no distances.
        vector_backend.fit([graph])
        return len(vector_backend.features(graph))

    return vector_call, ""


def execute_unit(
    *,
    graph: nx.Graph,
    representation: str,
    arm: str,
    budget_s: float,
    budget: Mapping[str, Any],
    clock: Callable[[], float] = time.process_time,
) -> dict[str, Any]:
    """Time one ``(graph, representation, arm)`` cell in this process.

    Args:
        graph: the graph.
        representation: registry key.
        arm: one of :data:`schema.ARMS`.
        budget_s: the wall clock, reported on a time-censored row.
        budget: output of :func:`budget_fields`.
        clock: injected for testing.

    Returns:
        The measurement half of a record: ``status``, ``error_kind``,
        ``seconds``, ``repeats``, ``length_chars``, ``fallback_used``.
    """
    from isalgraph.competitors.base import Budget

    budget_obj = Budget(**dict(budget))
    call, fallback_variant = _make_call(representation, graph, budget_obj)
    fallback_used: bool | None = False if fallback_variant else None

    start = clock()
    try:
        with engine_arm(arm):
            timing = timed_call(call, clock=clock)
    except Exception as exc:  # noqa: BLE001 - every failure is a recorded outcome
        elapsed = clock() - start
        status, error_kind = _classify_exception(exc)
        seconds = budget_s if error_kind in schema.TIME_CENSORING_KINDS else elapsed
        return {
            "status": status,
            "error_kind": error_kind,
            "seconds": float(seconds),
            "repeats": 0,
            "length_chars": None,
            "fallback_used": fallback_used,
            "detail": f"{type(exc).__name__}: {exc}"[:400],
        }

    return {
        "status": "ok",
        "error_kind": None,
        "seconds": float(timing.seconds),
        "repeats": int(timing.repeats),
        "length_chars": int(timing.value),
        "fallback_used": fallback_used,
        "detail": "",
    }


# ---------------------------------------------------------------------------
# Parent side: run one unit in a killable subprocess
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    """Return the repository root, resolving the ``benchmarks/`` symlink.

    ``__file__`` may arrive through ``benchmarks/eval_t13_complexity`` (the
    symlink) or through ``benchmarks/real_data/eval_t13_complexity`` (the real
    path).  Resolving first, then walking up to the ``pyproject.toml``, gives
    the same answer either way.

    Returns:
        The repository root.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def child_argv() -> list[str]:
    """Return the command that runs one unit in a fresh interpreter.

    Returns:
        The argv for :func:`run_unit`'s subprocess.
    """
    return [
        sys.executable,
        "-m",
        "benchmarks.real_data.eval_t13_complexity.measure",
        "--exec-unit",
    ]


def run_unit(
    payload: Mapping[str, Any],
    *,
    budget_s: float,
    argv: Sequence[str] | None = None,
    grace_s: float = KILL_GRACE_S,
    cwd: Path | None = None,
) -> dict[str, Any]:
    """Run one unit in a subprocess, killing it if it outlives the budget.

    **This is the budget, and it is the only mechanism that covers every
    backend.**  ``SIGALRM`` may not be used: T-05 finding 5 established that it
    does not interrupt the C++ engine, so a signal-based timeout silently fails
    to fire and the shard runs to its wallclock with no result.

    On expiry the child is terminated, then killed, and the unit is recorded
    ``status="censored"`` with ``error_kind="wallclock_kill"``,
    ``seconds = budget_s`` and ``length_chars = None``.  The parent survives.

    Args:
        payload: the child's stdin document -- the graph, the representation,
            the arm and the resolved budget.
        budget_s: the wall clock.
        argv: injected for testing.  Defaults to :func:`child_argv`.
        grace_s: seconds allowed beyond *budget_s* for interpreter startup and
            the sub-second repeats, both of which sit outside the budgeted call.
        cwd: working directory for the child.  Defaults to the repository root,
            so ``-m benchmarks...`` resolves.

    Returns:
        The measurement half of a record.
    """
    command = list(argv) if argv is not None else child_argv()
    env = dict(os.environ)
    env.update(CHILD_THREAD_ENV)
    # A src-first path shadows the editable install and drops the engine to
    # pure Python with no error, which would make every timing fiction.
    env.pop("PYTHONPATH", None)

    process = subprocess.Popen(  # noqa: S603 - argv is built here, never user text
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(cwd or _repo_root()),
        env=env,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(json.dumps(payload), timeout=budget_s + grace_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        return {
            "status": "censored",
            "error_kind": schema.KIND_WALLCLOCK,
            "seconds": float(budget_s),
            "repeats": 0,
            "length_chars": None,
            "fallback_used": None,
            "detail": f"killed after {budget_s + grace_s:.1f} s",
        }

    if process.returncode != 0 or not stdout.strip():
        return {
            "status": "error",
            "error_kind": f"ChildExit{process.returncode}",
            "seconds": 0.0,
            "repeats": 0,
            "length_chars": None,
            "fallback_used": None,
            "detail": (stderr or stdout).strip()[-400:],
        }

    result: dict[str, Any] = json.loads(stdout.splitlines()[-1])
    return result


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Provenance:
    """The seven fields every record of one shard shares."""

    run_id: str
    host: str
    engine: str
    build_hash: str
    isalgraph_version: str
    timestamp_utc: str


def assemble_record(
    *,
    provenance: Provenance,
    spec: GraphSpec,
    properties: Mapping[str, Any],
    symmetry: Mapping[str, Any],
    representation: str,
    arm: str,
    measurement: Mapping[str, Any],
    budget_s: float,
    spec_string: str,
) -> Record:
    """Build one validated :class:`~.schema.Record`.

    Args:
        provenance: the shared seven fields.
        spec: the graph's address.
        properties: output of :func:`graph_properties`.
        symmetry: output of :func:`symmetry_fields`.
        representation: registry key.
        arm: one of :data:`schema.ARMS`.
        measurement: output of :func:`run_unit` or :func:`execute_unit`.
        budget_s: the wall clock.
        spec_string: output of :func:`budget_spec`.

    Returns:
        The record.

    Raises:
        SchemaError: when the assembled row violates the frozen schema, which
            means the runner produced a combination the timing rule cannot.
    """
    mapping: dict[str, Any] = {
        "schema_version": schema.SCHEMA_VERSION,
        "run_id": provenance.run_id,
        "host": provenance.host,
        "engine": provenance.engine,
        "build_hash": provenance.build_hash,
        "isalgraph_version": provenance.isalgraph_version,
        "timestamp_utc": provenance.timestamp_utc,
        "source": spec.source,
        "family": spec.family,
        "n_target": spec.n_target,
        "replicate": spec.replicate,
        "dataset": spec.dataset,
        "graph_index": spec.graph_index,
        "graph_id": spec.graph_id,
        **{k: properties[k] for k in ("n", "m", "density", "max_degree", "connected")},
        **{k: symmetry[k] for k in schema.SYMMETRY_FIELDS},
        "representation": representation,
        "arm": arm,
        "status": measurement["status"],
        "error_kind": measurement["error_kind"],
        "seconds": measurement["seconds"],
        "repeats": measurement["repeats"],
        "budget_s": budget_s,
        "budget_spec": spec_string,
        "length_chars": measurement["length_chars"],
        "fallback_used": measurement["fallback_used"],
    }
    return schema.record_from_mapping(mapping)


# ---------------------------------------------------------------------------
# The exhaustive/canonical identity gate
# ---------------------------------------------------------------------------


def canonical_identity_violations(records: Iterable[Record]) -> tuple[str, ...]:
    """Report graphs where the two exhaustive-canonical arms disagree.

    ``isalgraph_canonical`` and ``isalgraph_exhaustive`` share one ``encode``
    path and one ``variant="canonical"``; they differ **only** in a
    ``SUITE1_ONLY`` scope guard.  So wherever the guard permits both to run --
    ``n <= 12`` -- their ``status`` and ``length_chars`` must be identical.  A
    divergence means the guard is doing something other than refusing, and the
    substitution of one arm for the other on the constructed grid stops being
    sound.

    Shards split the two arms by hash, so this is a merge-time gate; it is
    still run per shard, free, on whichever pairs happen to co-reside.

    Args:
        records: records to check.

    Returns:
        One message per violating graph, empty when the arms agree.
    """
    seen: dict[tuple[str, str], Record] = {}
    for record in records:
        if record.representation not in ("isalgraph_canonical", "isalgraph_exhaustive"):
            continue
        if record.arm != "default" or record.n > SUITE1_MAX_NODES:
            continue
        key = (
            f"{record.source}|{record.family}|{record.dataset}|{record.graph_index}"
            f"|{record.n_target}|{record.replicate}",
            record.representation,
        )
        seen[key] = record

    violations: list[str] = []
    graph_keys = {key for key, _arm in seen}
    for graph_key in sorted(graph_keys):
        left = seen.get((graph_key, "isalgraph_canonical"))
        right = seen.get((graph_key, "isalgraph_exhaustive"))
        if left is None or right is None:
            continue
        if (left.status, left.length_chars) != (right.status, right.length_chars):
            violations.append(
                f"{graph_key}: isalgraph_canonical=({left.status}, {left.length_chars}) "
                f"vs isalgraph_exhaustive=({right.status}, {right.length_chars})"
            )
    return tuple(violations)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def shard_filename(source: str, shard: int, n_shards: int) -> str:
    """Return the per-shard file name CONTRACTS §5.4 fixes.

    Args:
        source: ``"constructed"`` or ``"cohort"``.
        shard: this shard's index.
        n_shards: total shards.

    Returns:
        ``records_<source>_<shard>of<n_shards>.jsonl``.
    """
    return f"records_{source}_{shard}of{n_shards}.jsonl"


def resolve_out_path(out: Path, *, source: str, shard: int, n_shards: int) -> Path:
    """Resolve ``--out`` to a file.

    A directory (existing, or any path with no ``.jsonl`` suffix) receives the
    canonical per-shard name; an explicit ``.jsonl`` path is used verbatim, so
    a smoke run can name its own file.

    Args:
        out: the ``--out`` value.
        source: ``"constructed"`` or ``"cohort"``.
        shard: this shard's index.
        n_shards: total shards.

    Returns:
        The file to append to.
    """
    if out.suffix == ".jsonl" and not out.is_dir():
        return out
    return out / shard_filename(source, shard, n_shards)


def existing_unit_keys(path: Path) -> set[str]:
    """Return the unit keys already present in *path*, for resume.

    A shard file is append-only, so a task that was requeued mid-array picks up
    where it stopped rather than duplicating rows.  A malformed trailing line
    (the signature of a kill mid-write) is skipped, not fatal.

    Args:
        path: the shard file.

    Returns:
        The keys already recorded.
    """
    if not path.exists():
        return set()
    keys: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("record_kind") == "header":
                continue
            spec = GraphSpec(
                source=row["source"],
                family=row["family"],
                n_target=row["n_target"],
                replicate=row["replicate"],
                dataset=row["dataset"],
                graph_index=row["graph_index"],
            )
            keys.add(WorkUnit(spec, row["representation"], row["arm"]).key)
    return keys


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Everything one shard needs, parsed from the command line."""

    source: str
    shard: int
    n_shards: int
    datasets: tuple[str, ...] | None
    arms: tuple[str, ...]
    representations: tuple[str, ...]
    budget_s: float
    seed: int
    out: Path
    run_id: str


def run_shard(config: RunConfig) -> dict[str, int]:
    """Run every in-shard work unit and append its record to the shard file.

    Args:
        config: the parsed command line.

    Returns:
        ``status -> count`` over the records this call wrote.

    Raises:
        EngineMismatchError: when the engine gate fails.  Before any work.
    """
    build_info = assert_engine()
    import isalgraph

    provenance = Provenance(
        run_id=config.run_id,
        host=platform.node(),
        engine=isalgraph.engine(),
        build_hash=str(build_info["build_hash"]),
        isalgraph_version=str(getattr(isalgraph, "__version__", "unknown")),
        timestamp_utc=datetime.now(UTC).replace(microsecond=0).isoformat(),
    )

    have_symmetry = symmetry_available()
    if not have_symmetry:
        LOGGER.warning(
            "symmetry.py (track A) is absent: the nine symmetry fields will be null "
            "and no row of this shard may enter the |Aut| regression"
        )

    entries = build_grid(config.source, datasets=config.datasets, seed=config.seed)
    LOGGER.info("grid: %d graphs from source=%s", len(entries), config.source)

    properties = {entry.spec.key: graph_properties(entry.graph) for entry in entries}
    ablation_keys = select_ablation_graphs(
        [(entry.spec, properties[entry.spec.key]["n"]) for entry in entries]
    )

    path = resolve_out_path(
        config.out, source=config.source, shard=config.shard, n_shards=config.n_shards
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    done = existing_unit_keys(path)
    fresh = not path.exists() or path.stat().st_size == 0

    counts: dict[str, int] = {}
    written: list[Record] = []
    with path.open("a", encoding="utf-8") as handle:
        if fresh:
            header = schema.run_header(
                run_id=config.run_id,
                host=provenance.host,
                engine=provenance.engine,
                build_info=build_info,
                isalgraph_version=provenance.isalgraph_version,
                timestamp_utc=provenance.timestamp_utc,
                source=config.source,
                shard=config.shard,
                n_shards=config.n_shards,
                arms=config.arms,
                representations=config.representations,
                budget_s=config.budget_s,
                seed=config.seed,
                symmetry_available=have_symmetry,
            )
            handle.write(json.dumps(header, ensure_ascii=False) + "\n")
            handle.flush()

        for entry in entries:
            units = [
                unit
                for unit in units_for_graph(
                    entry.spec,
                    representations=config.representations,
                    arms=config.arms,
                    ablation_keys=ablation_keys,
                )
                if shard_of(unit.key, config.n_shards) == config.shard and unit.key not in done
            ]
            if not units:
                continue

            props = properties[entry.spec.key]
            sym = symmetry_fields(entry.graph, available=have_symmetry)
            resolved = budget_fields(n_nodes=props["n"], budget_s=config.budget_s)
            spec_string = budget_spec(resolved)
            payload_graph = {
                "n": props["n"],
                "edges": [[int(u), int(v)] for u, v in entry.graph.edges()],
            }

            for unit in units:
                measurement = run_unit(
                    {
                        "graph": payload_graph,
                        "representation": unit.representation,
                        "arm": unit.arm,
                        "budget_s": config.budget_s,
                        "budget": resolved,
                        "expected_build_hash": EXPECTED_BUILD_HASH,
                    },
                    budget_s=config.budget_s,
                )
                record = assemble_record(
                    provenance=provenance,
                    spec=entry.spec,
                    properties=props,
                    symmetry=sym,
                    representation=unit.representation,
                    arm=unit.arm,
                    measurement=measurement,
                    budget_s=config.budget_s,
                    spec_string=spec_string,
                )
                handle.write(record.to_json_line())
                handle.flush()
                written.append(record)
                counts[record.status] = counts.get(record.status, 0) + 1

    for violation in canonical_identity_violations(written):
        LOGGER.error("canonical/exhaustive identity gate: %s", violation)
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _exec_unit_main() -> int:
    """Child entry point: read one unit from stdin, write its result to stdout.

    Returns:
        Process exit status.
    """
    payload = json.loads(sys.stdin.read())
    assert_engine(expected_build_hash=payload["expected_build_hash"])

    import networkx as nx_mod

    graph = nx_mod.Graph()
    graph.add_nodes_from(range(int(payload["graph"]["n"])))
    graph.add_edges_from((int(u), int(v)) for u, v in payload["graph"]["edges"])

    result = execute_unit(
        graph=graph,
        representation=payload["representation"],
        arm=payload["arm"],
        budget_s=float(payload["budget_s"]),
        budget=payload["budget"],
    )
    sys.stdout.write(json.dumps(result) + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.eval_t13_complexity.measure",
        description="T-13 controlled-experiment runner (CONTRACTS §5).",
    )
    parser.add_argument("--source", choices=list(schema.SOURCES), default="constructed")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="cohort name; repeatable. Default: every dataset present.",
    )
    parser.add_argument(
        "--arms",
        default="default",
        help=f"comma-separated, from {list(schema.ARMS)}",
    )
    parser.add_argument(
        "--representations",
        default=",".join(REPRESENTATIONS),
        help="comma-separated registry keys. Default: the frozen thirteen.",
    )
    parser.add_argument("--budget-s", type=float, default=DEFAULT_BUDGET_S)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", type=Path, default=Path("records.jsonl"))
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--exec-unit",
        action="store_true",
        help=argparse.SUPPRESS,  # internal: one unit from stdin
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: arguments, or ``None`` for ``sys.argv[1:]``.

    Returns:
        Process exit status.
    """
    args = build_parser().parse_args(argv)
    if args.exec_unit:
        return _exec_unit_main()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    arms = tuple(a.strip() for a in str(args.arms).split(",") if a.strip())
    unknown = [a for a in arms if a not in schema.ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; known: {list(schema.ARMS)}")

    representations = tuple(r.strip() for r in str(args.representations).split(",") if r.strip())
    unresolved = {
        name: status
        for name, status in resolve_representations(representations).items()
        if not status.startswith("ok")
    }
    if unresolved:
        raise SystemExit(f"representations did not resolve: {unresolved}")

    config = RunConfig(
        source=args.source,
        shard=int(args.shard),
        n_shards=int(args.n_shards),
        datasets=tuple(args.dataset) if args.dataset else None,
        arms=arms,
        representations=representations,
        budget_s=float(args.budget_s),
        seed=int(args.seed),
        out=Path(args.out),
        run_id=str(args.run_id) or f"t13_{datetime.now(UTC):%Y%m%dT%H%M%SZ}",
    )
    counts = run_shard(config)
    LOGGER.info("shard %d/%d complete: %s", config.shard, config.n_shards, counts)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
