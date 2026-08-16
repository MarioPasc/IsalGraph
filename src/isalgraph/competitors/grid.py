"""``python -m isalgraph.competitors.grid`` -- the (representation x distance) grid.

Computes **F0, F1, F2, F3, F4 and F6**.  It does **not** compute F5 and it
cannot: nothing in this module's import closure reaches a GED loader, and a
test asserts that.

That is the point.  Decision 24's whole defence is that T-04a's exclusion
rule could not have seen the outcome it selects on.  ``competitors.md``
§3.4 fixes the rule in advance -- *the primary distance is the cheapest that
passes F1 at 100 %, F2, F3 and F4; ties break on F6, never on F5* -- and
selecting on correlation with GED would be selecting the baseline that makes
IsalGraph look best.  Prose cannot enforce it; an import graph can.

F5 lives in ``python -m isalgraph.competitors.f5``, whose output is reported
and is **not an input to selection**.

**Every cell is attempted, and a cell that fails is a result** -- one a
reviewer would otherwise ask about.  ``padded_hamming`` x ``sparse6`` is
undefined and prints as such; that cell is one of the reasons the grid
exists.  The same holds for a cell the *candidate rule* below excludes: it
is measured and printed in full, and only its eligibility is withdrawn.

Four things this module does that its T-04 ancestor did not, each repairing
a measured defect that produced a plausible number and no error
(``T-04a-design.md`` §1.3):

1. ``--sample pooled-<k>`` draws **one** stratum-balanced sample over the
   pooled cohort.  ``--sample stratified-<k>`` drew ``k`` graphs *per
   dataset* -- 1,889 for ``k = 200`` -- and is gone.
2. **F0 and F1 are split** (§3.3).  Encodability is a property of a
   *representation*; well-definedness is a property of a *distance*.  The
   ancestor discarded the encode-failure count and computed F1 over whatever
   encoded, so ``agm_cam``'s 102/200 failures were invisible.
3. **The candidate rule is on ``metric.consumes``**, not only on
   ``Capability.BASELINE`` on the backend.  ``size_null`` the *metric* is
   defined on 100 % of pairs, is a true metric, is relabelling-invariant and
   is the cheapest cell in the grid, so the ancestor's rule named *count the
   nodes and subtract* the primary distance of all eleven representations.
4. **Encoding happens once per backend** and is reused across all six
   metrics, and each F3 relabelling is encoded once rather than twice.  At
   0.53 s/graph for ``min_dfs`` that is the difference between a grid that
   runs in minutes and one that does not.

T-04 shipped the ancestor and proved it ran end to end on a 20-graph dry
run.  **T-04a** runs this on the 200-graph stratified sample under its own
protocol and applies the selection rule.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    import networkx as nx

from isalgraph.competitors import datasets, fixtures
from isalgraph.competitors.base import (
    Capability,
    Comparable,
    VectorBackend,
    table_scope_error,
)
from isalgraph.competitors.registry import (
    AnyBackend,
    AnyMetric,
    available_backends,
    available_metrics,
    get_backend,
    get_metric,
)
from isalgraph.errors import CompetitorError, DistanceUndefined

#: F2 is checked over this many random triples.
F2_TRIPLES = 5_000
#: F3 protocol, frozen (design note §3.2).
F3_GRAPHS = 50
F3_RELABELLINGS = 20
#: F6's advisory threshold.  **Advisory, never a gate**: §3.4, the operative
#: rule, breaks ties on F6 and does not exclude on it.  §3.3's "> 1 ms/pair"
#: line is reported as a flag so a reader can see the cost without the number
#: silently removing a cell.
F6_MS_PER_PAIR_ADVISORY_LIMIT = 1.0
#: Lower bound of strata 4-5.  ``f6_ms_per_pair_large`` restricts to pairs
#: where **both** graphs clear it, which is the closest available reading of
#: ``competitors.md`` §3.3's "us/pair at n-bar = 30".
LARGE_GRAPH_MIN_NODES = 21

#: A metric is a candidate primary distance iff it reads the representation.
#: ``"order"`` (``size_null``) reads only the node count; ``"text"``
#: (``levenshtein_char``) reads the character rendering, which charges four
#: edits for one deleted min-DFS tuple and is supplementary by construction.
#: Design note §3.4 and CONTRACTS §4 both state the rule as this membership.
CANDIDATE_CONSUMES: frozenset[str] = frozenset({"symbols", "frame", "features"})

#: CONTRACTS §4, verbatim.
ORDER_EXCLUSION = "baseline: consumes 'order'; not a candidate distance (competitors.md 3.2)"
BASELINE_EXCLUSION = "backend carries Capability.BASELINE; never a primary distance"


@dataclass
class Cell:
    """One (representation x distance) cell's measurements.

    Field names and semantics are frozen by CONTRACTS §3; the report reads
    them by name.
    """

    backend: str
    metric: str
    #: ``False`` when the pairing is structurally impossible, e.g. a metric
    #: that consumes ``features`` against a ``ReprBackend``.
    applicable: bool = True
    reason: str | None = None
    #: Eligible to be a **primary** distance at all.  See
    #: :func:`_candidate_status`.  An ineligible cell is still measured.
    candidate: bool = True
    #: F1: fraction of pairs **among encodable graphs** on which the
    #: distance is computable, and its denominator.
    f1_defined_frac: float | None = None
    f1_n_pairs: int | None = None
    #: F2: metric axioms.  ``is_pseudometric`` is the backend's declaration;
    #: the violations are what was actually observed.
    f2_declared_pseudometric: bool | None = None
    f2_violations: dict[str, int] | None = None
    #: F3: ``invariant / attempted`` graphs whose distance to a relabelled
    #: self is 0, and the number the backend raised on.  A skipped graph
    #: never counts as non-invariant, so ``0/0`` and ``0/50`` differ.
    f3_invariant: str | None = None
    f3_skipped: int | None = None
    #: F4: degeneracy.
    f4_zero_mass: float | None = None
    f4_coeff_variation: float | None = None
    #: F6: cost, over all defined pairs and over the large-graph pairs.
    f6_ms_per_pair: float | None = None
    f6_ms_per_pair_large: float | None = None
    f6_over_advisory_limit: bool | None = None
    passes_selection: bool = False
    excluded_because: str | None = None


@dataclass
class _Tally:
    """One F0 counter: attempted, encodable, and the exception types seen."""

    attempted: int = 0
    encodable: int = 0
    errors: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        """The JSON shape of CONTRACTS §2's ``f0`` leaf.

        ``frac`` is 0.0 when nothing was attempted.  That is a placeholder,
        not a measurement -- read ``attempted`` before reading ``frac``.
        """
        return {
            "attempted": self.attempted,
            "encodable": self.encodable,
            "frac": self.encodable / self.attempted if self.attempted else 0.0,
            "errors": dict(self.errors),
        }


@dataclass
class EncodeCache:
    """One backend's encodings of the whole sample, computed **once**.

    Reused across all six metrics.  The ancestor re-encoded per cell, which
    at ``min_dfs``'s 0.53 s/graph is a sixfold waste and, worse, let two
    cells of the same row disagree about which graphs encoded.

    Attributes:
        items: the encodable graphs' encodings, in sample order, failures
            dropped.  ``n_nodes`` is aligned with it element by element.
        f0: ``{"overall"|"suite1"|"suite2": tally}``, per CONTRACTS §2.
    """

    backend: str
    available: bool = True
    error: str | None = None
    is_vector: bool = False
    capabilities: frozenset[Capability] = frozenset()
    items: list[Comparable] = field(default_factory=list)
    n_nodes: list[int] = field(default_factory=list)
    f0: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class F3Cache:
    """One backend's F3 encodings: a base graph and its 20 relabelled copies.

    Each copy is encoded **once** and the result is shared across every
    metric.  The ancestor encoded each copy twice -- once for ``is_defined``
    and once for ``distance`` -- inside a loop that itself ran per cell.

    Attributes:
        entries: ``(base, copies)`` per graph the backend encoded.
        skipped: graphs the backend raised on.  ``attempted + skipped`` is
            the size of the F3 sample, always.
    """

    backend: str
    entries: list[tuple[Comparable, list[Comparable]]] = field(default_factory=list)
    skipped: int = 0

    @property
    def attempted(self) -> int:
        """Graphs on which F3 could actually be evaluated."""
        return len(self.entries)


def _empty_f0(suites: Sequence[str]) -> dict[str, dict[str, Any]]:
    """An F0 block for a backend that could not be constructed at all."""
    tallies = {"overall": _Tally(attempted=len(suites)), "suite1": _Tally(), "suite2": _Tally()}
    for suite in suites:
        tallies[suite].attempted += 1
    return {key: tally.as_dict() for key, tally in tallies.items()}


def encode_sample(
    backend_name: str, graphs: Sequence[nx.Graph], suites: Sequence[str]
) -> EncodeCache:
    """Encode every graph of the sample once under *backend_name*.  **F0.**

    A failure is counted by exception type and the graph is dropped; nothing
    raises.  For the one :class:`VectorBackend` the fit is over the whole
    sample, never per batch -- a per-batch fit produces a different
    vocabulary and therefore a distance matrix that depends on batching
    order.

    Args:
        backend_name: registry key.
        graphs: the sample, in draw order.
        suites: ``"suite1"``/``"suite2"`` per graph, aligned with *graphs*.

    Returns:
        The cache, with ``available = False`` and ``error`` set when the
        backend itself could not be constructed.
    """
    try:
        backend = get_backend(backend_name)
    except CompetitorError as exc:
        return EncodeCache(
            backend=backend_name,
            available=False,
            error=f"{type(exc).__name__}: {exc}",
            f0=_empty_f0(suites),
        )

    is_vector = isinstance(backend, VectorBackend)
    cache = EncodeCache(
        backend=backend_name, is_vector=is_vector, capabilities=backend.capabilities
    )
    tallies = {"overall": _Tally(), "suite1": _Tally(), "suite2": _Tally()}

    if isinstance(backend, VectorBackend):
        try:
            backend.fit(list(graphs))
        except Exception as exc:  # noqa: BLE001 - a failed fit is a datum
            for graph, suite in zip(graphs, suites, strict=True):
                del graph
                for key in ("overall", suite):
                    tallies[key].attempted += 1
                    tallies[key].errors[type(exc).__name__] += 1
            cache.f0 = {key: tally.as_dict() for key, tally in tallies.items()}
            return cache

    for graph, suite in zip(graphs, suites, strict=True):
        keys = ("overall", suite)
        for key in keys:
            tallies[key].attempted += 1
        try:
            item: Comparable = (
                dict(backend.features(graph))
                if isinstance(backend, VectorBackend)
                else backend.encode(graph)
            )
        except Exception as exc:  # noqa: BLE001 - a failure is a datum, not a stop
            for key in keys:
                tallies[key].errors[type(exc).__name__] += 1
            continue
        for key in keys:
            tallies[key].encodable += 1
        cache.items.append(item)
        cache.n_nodes.append(int(graph.number_of_nodes()))

    cache.f0 = {key: tally.as_dict() for key, tally in tallies.items()}
    return cache


def encode_f3(backend_name: str, graphs: Sequence[nx.Graph], *, seed: int) -> F3Cache:
    """Encode each F3 graph and its :data:`F3_RELABELLINGS` copies, **once**.

    Relabelling goes through :func:`fixtures.shuffled_copy`, never
    ``nx.relabel_nodes(copy=True)``, which preserves insertion order and so
    makes an order-dependent format look invariant (finding 13).

    The copies are drawn before any encode, so the RNG stream is identical
    across backends whatever any of them raises on.

    Args:
        backend_name: registry key.
        graphs: the F3 sub-sample ``S50``.
        seed: seed of the single :class:`random.Random` driving the
            relabellings.

    Returns:
        The cache.  ``attempted + skipped == len(graphs)`` always holds.
    """
    cache = F3Cache(backend=backend_name)
    try:
        backend = get_backend(backend_name)
    except CompetitorError:
        cache.skipped = len(graphs)
        return cache

    rng = random.Random(seed)
    for graph in graphs:
        copies = [fixtures.shuffled_copy(graph, rng) for _ in range(F3_RELABELLINGS)]
        try:
            if isinstance(backend, VectorBackend):
                # Refit per graph: the vocabulary must cover the copies, and
                # this instance is not the one that encoded the sample.
                backend.fit([graph, *copies])
                base: Comparable = dict(backend.features(graph))
                encoded: list[Comparable] = [dict(backend.features(c)) for c in copies]
            else:
                base = backend.encode(graph)
                encoded = [backend.encode(c) for c in copies]
        except Exception:  # noqa: BLE001 - a raise is "skipped", never "not invariant"
            cache.skipped += 1
            continue
        cache.entries.append((base, encoded))
    return cache


def _applicable(is_vector: bool, metric: AnyMetric) -> tuple[bool, str | None]:
    """Whether this pairing can exist at all, before anything is computed."""
    if metric.consumes == "features" and not is_vector:
        return False, "kernel distance consumes features; this backend is a serialisation"
    if metric.consumes != "features" and is_vector:
        return False, f"{metric.name} consumes {metric.consumes}; WL emits a feature vector"
    return True, None


def _candidate_status(
    capabilities: frozenset[Capability], metric: AnyMetric
) -> tuple[bool, str | None]:
    """CONTRACTS §4: is this cell eligible to be a **primary** distance?

    Two independent exclusions, and the metric's is checked first so that
    ``size_null`` the metric carries the same reason under every backend.

    Args:
        capabilities: the backend's declared capabilities.
        metric: the distance.

    Returns:
        ``(candidate, reason)``; *reason* is ``None`` iff *candidate*.
    """
    if metric.consumes not in CANDIDATE_CONSUMES:
        if metric.consumes == "order":
            return False, ORDER_EXCLUSION
        return False, (
            f"supplementary: consumes {metric.consumes!r}; not a candidate distance "
            f"(competitors.md 3.2)"
        )
    if Capability.BASELINE in capabilities:
        return False, BASELINE_EXCLUSION
    return True, None


def _f2(metric: AnyMetric, codes: Sequence[Comparable], rng: random.Random) -> dict[str, int]:
    """Identity, symmetry and the triangle inequality over random triples.

    A violation is **declared, not repaired**.  Comparing a metric against a
    non-metric is legitimate; comparing them without saying so is not.
    """
    violations = {"identity": 0, "symmetry": 0, "triangle": 0}
    if len(codes) < 3:
        return violations
    for _ in range(F2_TRIPLES):
        a, b, c = (codes[rng.randrange(len(codes))] for _ in range(3))
        try:
            if metric.distance(a, a) != 0.0:
                violations["identity"] += 1
            d_ab = metric.distance(a, b)
            if d_ab != metric.distance(b, a):
                violations["symmetry"] += 1
            if d_ab > metric.distance(a, c) + metric.distance(c, b) + 1e-9:
                violations["triangle"] += 1
        except DistanceUndefined:
            continue
    return violations


def _f3_from_cache(metric: AnyMetric, cache: F3Cache) -> tuple[str, int]:
    """``invariant/attempted`` and the skip count, from pre-encoded copies."""
    invariant = 0
    for base, copies in cache.entries:
        try:
            distances = [metric.distance(base, c) for c in copies if metric.is_defined(base, c)]
        except Exception:  # noqa: BLE001 - a raising distance is not invariance
            continue
        if distances and all(d == 0.0 for d in distances):
            invariant += 1
    return f"{invariant}/{cache.attempted}", cache.skipped


def measure_cell(
    backend_name: str,
    metric_name: str,
    cache: EncodeCache | Sequence[nx.Graph],
    f3cache: F3Cache | None = None,
    *,
    seed: int,
    suite: str | None = None,
    suites: Sequence[str] | None = None,
) -> Cell:
    """Measure one cell from pre-computed encodings.  Never raises.

    Args:
        backend_name: registry key of the representation.
        metric_name: registry key of the distance.
        cache: :func:`encode_sample`'s output for *backend_name*.  A plain
            graph sequence is also accepted, for a single ad-hoc cell; the
            caches are then built on the spot, which re-encodes per call and
            is why :func:`run_grid` does not do it.
        f3cache: :func:`encode_f3`'s output for *backend_name*.  Required
            unless *cache* is a graph sequence.
        seed: seed of the F2 triple sampler.
        suite: ``"suite1"``/``"suite2"`` when every graph comes from one
            named dataset, so a printed row can be refused where it would be
            conditioned on tractability.  ``None`` for a pooled sample, where
            a ``SUITE1_ONLY`` backend instead shows up as F0 < 1.0 on
            ``suite2`` and is charged there (design note §3.3).
        suites: per-graph suite for the ad-hoc path.  Defaults to *suite*
            repeated, or ``"suite1"``.

    Returns:
        The cell, with every measurable field filled in whether or not it is
        a candidate.
    """
    if not isinstance(cache, EncodeCache):
        graphs = list(cache)
        per_graph = list(suites) if suites is not None else [suite or "suite1"] * len(graphs)
        cache = encode_sample(backend_name, graphs, per_graph)
        f3cache = encode_f3(backend_name, graphs[:F3_GRAPHS], seed=seed)
    if f3cache is None:
        f3cache = F3Cache(backend=backend_name)

    cell = Cell(backend=backend_name, metric=metric_name)
    if not cache.available:
        cell.applicable = False
        cell.candidate = False
        cell.reason = cache.error
        cell.excluded_because = cache.error
        return cell
    try:
        metric = get_metric(metric_name)
    except CompetitorError as exc:
        cell.applicable = False
        cell.candidate = False
        cell.reason = f"{type(exc).__name__}: {exc}"
        cell.excluded_because = cell.reason
        return cell

    # Eligibility is fixed by the declarations alone and is recorded before
    # anything is measured, so that it cannot depend on an outcome.
    candidate, candidate_reason = _candidate_status(cache.capabilities, metric)
    cell.candidate = candidate
    cell.excluded_because = candidate_reason

    if suite is not None:
        scope = table_scope_error(cache.capabilities, suite, backend_name)
        if scope is not None:
            cell.applicable = False
            cell.candidate = False
            cell.reason = scope
            cell.excluded_because = candidate_reason or scope
            return cell

    ok, reason = _applicable(cache.is_vector, metric)
    if not ok:
        cell.applicable = False
        cell.candidate = False
        cell.reason = reason
        cell.excluded_because = candidate_reason or reason
        return cell

    items = cache.items
    if len(items) < 2:
        cell.applicable = False
        cell.candidate = False
        cell.reason = f"fewer than two graphs encoded ({len(items)}); see f0"
        cell.excluded_because = candidate_reason or cell.reason
        return cell

    index_pairs = [(i, j) for i in range(len(items)) for j in range(i + 1, len(items))]
    cell.f1_n_pairs = len(index_pairs)
    defined = [(i, j) for i, j in index_pairs if metric.is_defined(items[i], items[j])]
    cell.f1_defined_frac = len(defined) / len(index_pairs) if index_pairs else 0.0

    cell.f2_declared_pseudometric = metric.is_pseudometric
    cell.f2_violations = _f2(metric, items, random.Random(seed))

    values: list[float] = []
    if defined:
        try:
            start = time.perf_counter()
            values = [metric.distance(items[i], items[j]) for i, j in defined]
            elapsed = time.perf_counter() - start
            cell.f6_ms_per_pair = 1e3 * elapsed / len(defined)
        except Exception as exc:  # noqa: BLE001 - a raising sweep is a datum
            cell.reason = f"distance sweep raised {type(exc).__name__}: {exc}"

    large = [
        (i, j)
        for i, j in defined
        if cache.n_nodes[i] >= LARGE_GRAPH_MIN_NODES and cache.n_nodes[j] >= LARGE_GRAPH_MIN_NODES
    ]
    if large and cell.f6_ms_per_pair is not None:
        start = time.perf_counter()
        for i, j in large:
            metric.distance(items[i], items[j])
        cell.f6_ms_per_pair_large = 1e3 * (time.perf_counter() - start) / len(large)
    if cell.f6_ms_per_pair is not None:
        cell.f6_over_advisory_limit = cell.f6_ms_per_pair > F6_MS_PER_PAIR_ADVISORY_LIMIT

    if values:
        cell.f4_zero_mass = sum(v == 0.0 for v in values) / len(values)
        mean = statistics.fmean(values)
        cell.f4_coeff_variation = statistics.pstdev(values) / mean if mean else float("inf")

    cell.f3_invariant, cell.f3_skipped = _f3_from_cache(metric, f3cache)

    _apply_selection_rule(cell)
    return cell


def _apply_selection_rule(cell: Cell, backend: AnyBackend | None = None) -> None:
    """``competitors.md`` §3.4, fixed in advance and applied mechanically.

    An **ineligible** cell keeps the reason it is ineligible and is never
    marked as passing, whatever it scored.  An eligible cell passes iff every
    criterion holds; the failures are joined so a reader sees all of them at
    once rather than the first.

    **F6 is not consulted here.**  It is the tie-break in
    :func:`select_primary` and never a gate -- ``f6_over_advisory_limit`` is
    reported and does nothing.

    Args:
        cell: measured, with :attr:`Cell.candidate` and, when it is ``False``,
            :attr:`Cell.excluded_because` already set by :func:`measure_cell`.
        backend: optional, and then the ``Capability.BASELINE`` check is
            applied here instead.  The candidate rule proper needs the
            *metric* as well, so it lives in :func:`_candidate_status`; this
            argument keeps the backend-only half callable on its own.
    """
    if backend is not None and Capability.BASELINE in backend.capabilities:
        cell.candidate = False
        cell.excluded_because = BASELINE_EXCLUSION
    if not cell.candidate:
        cell.passes_selection = False
        return
    reasons = []
    if cell.f1_defined_frac is None or cell.f1_defined_frac < 1.0:
        reasons.append(f"F1 = {cell.f1_defined_frac}")
    if cell.f2_violations and any(cell.f2_violations.values()):
        reasons.append(f"F2 violations {cell.f2_violations}")
    if cell.f3_invariant is not None:
        got, total = (int(x) for x in cell.f3_invariant.split("/"))
        if total == 0:
            reasons.append(f"F3 never ran ({cell.f3_skipped} graphs skipped)")
        elif got < total:
            reasons.append(f"F3 = {cell.f3_invariant}")
    if cell.f4_zero_mass is not None and cell.f4_zero_mass > 0.5:
        reasons.append(f"F4 degenerate, zero mass {cell.f4_zero_mass:.3f}")
    if cell.f4_coeff_variation is not None and cell.f4_coeff_variation < 1e-6:
        reasons.append("F4 near-constant")
    cell.passes_selection = not reasons
    cell.excluded_because = "; ".join(reasons) or None


def select_primary(cells: Sequence[Cell]) -> dict[str, str | None]:
    """Cheapest **candidate** passing distance per representation.

    Ties break on ``(f6_ms_per_pair, metric_name)`` -- never on F5.  This
    module cannot compute F5, so the rule is enforced by the absence of the
    data rather than by the discipline of the caller.

    Args:
        cells: every measured cell.

    Returns:
        ``{backend: metric}``, ``None`` where no candidate passed.
    """
    out: dict[str, str | None] = {}
    for cell in cells:
        out.setdefault(cell.backend, None)
    for backend in out:
        eligible = [c for c in cells if c.backend == backend and c.candidate and c.passes_selection]
        if not eligible:
            continue
        out[backend] = min(
            eligible, key=lambda c: (c.f6_ms_per_pair or float("inf"), c.metric)
        ).metric
    return out


def selection_reasons(cells: Sequence[Cell], primary: dict[str, str | None]) -> dict[str, str]:
    """One sentence per representation explaining what selection did.

    Where nothing was selected, the sentence names the failing criterion for
    **every** candidate, so an empty cell in the selection table is a printed
    absence rather than a missing key.
    """
    out: dict[str, str] = {}
    for backend, chosen in primary.items():
        candidates = [c for c in cells if c.backend == backend and c.candidate]
        if chosen is not None:
            cell = next(c for c in candidates if c.metric == chosen)
            passing = sorted(c.metric for c in candidates if c.passes_selection)
            out[backend] = (
                f"cheapest candidate passing F1-F4; tie-break on F6 over {passing} "
                f"-> {chosen} at {cell.f6_ms_per_pair:.6f} ms/pair"
            )
            continue
        if not candidates:
            out[backend] = "no candidate distance exists for this representation"
            continue
        failures = "; ".join(
            f"{c.metric}: {c.excluded_because or c.reason or 'unmeasured'}"
            for c in sorted(candidates, key=lambda c: c.metric)
        )
        out[backend] = f"no candidate passed -- {failures}"
    return out


def run_grid(
    graphs: Sequence[nx.Graph],
    suites: Sequence[str],
    f3_graphs: Sequence[nx.Graph],
    *,
    seed: int,
    suite: str | None = None,
    backends: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
) -> tuple[list[Cell], dict[str, dict[str, dict[str, Any]]]]:
    """Measure every (representation x distance) cell.

    Encoding is hoisted out of the metric loop: one :func:`encode_sample` and
    one :func:`encode_f3` per backend serve all of its cells.

    Args:
        graphs: the sample.
        suites: per-graph suite, aligned with *graphs*.
        f3_graphs: the F3 sub-sample.
        seed: run seed.
        suite: single suite, or ``None`` for a pooled sample.
        backends: override the registry listing (tests).
        metrics: override the registry listing (tests).

    Returns:
        ``(cells, f0)``.
    """
    backend_names = (
        tuple(backends) if backends is not None else available_backends(include_baseline=True)
    )
    metric_names = tuple(metrics) if metrics is not None else available_metrics()
    cells: list[Cell] = []
    f0: dict[str, dict[str, dict[str, Any]]] = {}
    for backend_name in backend_names:
        cache = encode_sample(backend_name, graphs, suites)
        f0[backend_name] = cache.f0
        f3cache = encode_f3(backend_name, f3_graphs, seed=seed)
        for metric_name in metric_names:
            cells.append(
                measure_cell(backend_name, metric_name, cache, f3cache, seed=seed, suite=suite)
            )
    return cells, f0


def sample_block(
    kind: str,
    records: Sequence[datasets.SampleRecord],
    *,
    k: int,
    seed: int,
    names: Sequence[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """CONTRACTS §2's ``sample`` / ``f3_sample`` block for a draw."""
    per_stratum = Counter(record.stratum for record in records)
    per_dataset = {name: 0 for name in names}
    for record in records:
        per_dataset[record.dataset] = per_dataset.get(record.dataset, 0) + 1
    orders = [record.n_nodes for record in records]
    block: dict[str, Any] = {
        "kind": kind,
        "k": k,
        "seed": seed,
        "n_graphs": len(records),
        "datasets": list(names),
        "strata": [list(bounds) for bounds in datasets.STRATA],
        "per_stratum": {str(s): per_stratum.get(s, 0) for s in range(len(datasets.STRATA))},
        "per_dataset": per_dataset,
        "n_min": min(orders) if orders else None,
        "n_mean": round(statistics.fmean(orders), 4) if orders else None,
        "n_max": max(orders) if orders else None,
        "records": [asdict(record) for record in records],
    }
    if extra:
        block.update(extra)
    return block


def _records_for_dryrun(dataset: str, k: int, seed: int) -> tuple[datasets.SampleRecord, ...]:
    """T-04's smoke draw, expressed as :class:`~datasets.SampleRecord`s."""
    cohort = datasets.load(dataset)
    out = []
    for index in cohort.sample(k, seed=seed):
        n_nodes = int(cohort.graphs[index].number_of_nodes())
        stratum = datasets.stratum_of(n_nodes)
        if stratum is None:
            continue
        out.append(
            datasets.SampleRecord(
                dataset=dataset,
                index=index,
                n_nodes=n_nodes,
                stratum=stratum,
                suite=cohort.suite,
            )
        )
    return tuple(out)


def _print_report(cells: Sequence[Cell], f0: dict[str, dict[str, dict[str, Any]]]) -> None:
    """The console table.  Units are stated because getting them wrong is free."""
    print(f"\n{'F0 -- encodability (encodable/attempted)':60s}")
    print(f"{'backend':22s}{'overall':>14}{'suite1':>12}{'suite2':>12}  errors")
    for backend, block in sorted(f0.items()):
        overall, s1, s2 = block["overall"], block["suite1"], block["suite2"]
        errors = ", ".join(f"{k}={v}" for k, v in sorted(overall["errors"].items())) or "-"
        print(
            f"{backend:22s}"
            f"{overall['encodable']:>7}/{overall['attempted']:<6}"
            f"{s1['encodable']:>6}/{s1['attempted']:<5}"
            f"{s2['encodable']:>6}/{s2['attempted']:<5}  {errors}"
        )

    print(
        f"\n{'backend':22s}{'metric':17s}{'F1':>7}{'F3(+skip)':>14}{'F4=0':>7}"
        f"{'ms/pair':>10}{'ms/pair n>=21':>15}  verdict"
    )
    for cell in cells:
        if not cell.applicable:
            blank = f"{'':>7}{'':>14}{'':>7}{'':>10}{'':>15}"
            print(f"{cell.backend:22s}{cell.metric:17s}{blank}  n/a: {cell.reason}")
            continue
        f3 = f"{cell.f3_invariant}" + (f"+{cell.f3_skipped}s" if cell.f3_skipped else "")
        large = f"{cell.f6_ms_per_pair_large:.4f}" if cell.f6_ms_per_pair_large is not None else "-"
        verdict = "PASS" if cell.passes_selection else f"excluded: {cell.excluded_because}"
        print(
            f"{cell.backend:22s}{cell.metric:17s}"
            f"{cell.f1_defined_frac if cell.f1_defined_frac is not None else 0:>7.3f}"
            f"{f3:>14}"
            f"{cell.f4_zero_mass if cell.f4_zero_mass is not None else 0:>7.3f}"
            f"{cell.f6_ms_per_pair if cell.f6_ms_per_pair is not None else 0:>10.4f}"
            f"{large:>15}  {verdict}"
        )


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(prog="python -m isalgraph.competitors.grid")
    parser.add_argument(
        "--sample",
        default="dryrun-20",
        help="'pooled-<k>' (T-04a's protocol, one stratum-balanced draw over "
        "ALL_DATASETS) or 'dryrun-<k>' (T-04's single-dataset smoke draw)",
    )
    parser.add_argument("--dataset", default="iam_letter_low", help="dryrun only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if args.sample.startswith("dryrun-"):
        k = int(args.sample.split("-", 1)[1])
        records = _records_for_dryrun(args.dataset, k, args.seed)
        names: tuple[str, ...] = (args.dataset,)
        block = sample_block(
            "dryrun", records, k=k, seed=args.seed, names=names, extra={"dataset": args.dataset}
        )
        sample_suite: str | None = datasets.suite_of(args.dataset)
    elif args.sample.startswith("pooled-"):
        k = int(args.sample.split("-", 1)[1])
        names = datasets.ALL_DATASETS
        records = datasets.pooled_stratified_sample(names, k, seed=args.seed)
        block = sample_block("pooled_stratified", records, k=k, seed=args.seed, names=names)
        # Pooled across both suites, so there is no single suite whose row
        # could be refused; a SUITE1_ONLY backend is charged through F0 on
        # suite2 instead (design note §3.3).
        sample_suite = None
    else:
        parser.error(f"unknown --sample {args.sample!r}")

    f3_records = datasets.stratified_subsample(records, F3_GRAPHS, seed=args.seed, order=names)
    f3_block = sample_block(
        "f3_stratified_subsample", f3_records, k=F3_GRAPHS, seed=args.seed, names=names
    )

    graphs = [datasets.load(r.dataset).graphs[r.index] for r in records]
    suites = [r.suite for r in records]
    f3_graphs = [datasets.load(r.dataset).graphs[r.index] for r in f3_records]

    cells, f0 = run_grid(graphs, suites, f3_graphs, seed=args.seed, suite=sample_suite)
    primary = select_primary(cells)

    payload = {
        "protocol": "T-04a",
        "seed": args.seed,
        "n_graphs": len(graphs),
        "sample": block,
        "f3_sample": f3_block,
        "backends": sorted(f0),
        "metrics": sorted({cell.metric for cell in cells}),
        "f0": f0,
        "cells": [asdict(cell) for cell in cells],
        "primary_distance": primary,
        "selection_reason": selection_reasons(cells, primary),
        "f5": "NOT COMPUTED HERE, BY CONSTRUCTION -- see isalgraph.competitors.f5",
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    _print_report(cells, f0)
    print("\nprimary distance per representation")
    for backend in sorted(primary):
        print(f"  {backend:22s}{primary[backend] or 'NONE ADMISSIBLE'}")
    print(f"\nwrote {args.out} ({len(cells)} cells)")
    return 0


__all__ = [
    "CANDIDATE_CONSUMES",
    "Cell",
    "EncodeCache",
    "F3Cache",
    "encode_f3",
    "encode_sample",
    "main",
    "measure_cell",
    "run_grid",
    "sample_block",
    "select_primary",
    "selection_reasons",
]


if __name__ == "__main__":
    raise SystemExit(main())
