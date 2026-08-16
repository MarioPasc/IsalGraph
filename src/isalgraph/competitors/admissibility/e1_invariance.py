"""E1 -- relabelling invariance and the separation ratio ``psi``.

Protocol: ``.claude/notes/review/tasks/T-04a-admissibility-protocol.md`` §2.
Every pre-declared decision in §1 (D-A1…D-A5) is frozen and lives in
:mod:`isalgraph.competitors.admissibility.common`; nothing here re-derives an
interval, a seed or a relabelling count.

What this measures, and why it is two different things
------------------------------------------------------
For a representation ``R`` and a distance ``d``, the **self-distance** of a
graph is ``d(R(G), R(pi(G)))`` for a relabelling ``pi``.  If it is not 0 then
``d`` is not a well-defined function on isomorphism classes: its value depends
on a choice of node ordering, which is precisely what R1.2 asks about.  The
**separation ratio**

.. math::

    \\psi_R \\;=\\; \\frac{\\mathbb{E}\\,[\\,d(G, \\pi(G))\\,]}
                        {\\mathbb{E}\\,[\\,d(G, H) \\mid G \\not\\cong H\\,]}

puts that failure on a scale a reader can act on.  ``psi = 0`` iff the
representation is invariant on the cohort; ``psi ~ 1`` means the distance
between two relabellings of **one** graph is as large as the distance between
two **different** graphs, i.e. the representation is measuring node ordering
rather than structure.

Three parts, and the first is the load-bearing one
--------------------------------------------------
**Part A, the exhaustive characterisation.**  Over every connected graph on
2..7 nodes -- one per isomorphism class, 995 of them, OEIS A001349 -- decide
for each representation exactly which graphs are invariant under **every**
relabelling.  This is a decision procedure, not a sample: D-A3 says a
characterisation beats a p-value against a null nobody believes.

Two facts make it affordable and both are exact rather than approximations.
The orbit of ``G`` under ``S_n`` contains ``n!/|Aut(G)|`` distinct labelled
graphs, so **deduplicating by the permuted edge set** removes ``|Aut(G)|``-fold
repeated work without skipping a single labelled graph; and a representation
that is *not* invariant is refuted by **one** witness, so the sweep exits at
the first mismatch.  The residual cost is borne only by representations that
really are invariant, where the full sweep is exactly what certifies them.
Summed over the 853 graphs at ``n = 7`` the deduplicated sweep is 1,866,256
encodes -- the number of labelled connected graphs on seven nodes -- which
every backend currently in the pool completes.

**Part A tests label invariance, and that is deliberately the weaker test.**
A permuted graph is rebuilt with nodes inserted in ascending order, so the only
thing that differs between two permutations is the labelling.  Part B's
:func:`~isalgraph.competitors.fixtures.shuffled_copy` varies the labelling
*and* the insertion order *and* the edge order, which is strictly harder.  A
graph in Part A's invariant set may therefore still be non-invariant in Part B;
the converse cannot happen.  Part A's invariant set is an upper bound on
Part B's, and reporting both is what separates "depends on the labelling" from
"depends on how the object was built".

**Part B, the cohort statement.**  The frozen ``S200`` pooled draw plus a
seed-42 200-graph draw per dataset, :data:`common.RELABELLINGS` relabellings
each, invariance rate with an exact Clopper-Pearson interval and ``psi`` with a
graph-level percentile bootstrap.

**Part C, the paired between-representation comparison** (D-A5).  Every
representation sees the same graphs and the same relabellings -- the copies are
drawn **once per draw, before any backend runs** -- so the comparison is paired
by construction.  Wilcoxon signed-rank on the per-graph ``psi``, Holm-corrected
across the pairwise family, with the matched-pairs rank-biserial effect size.
**This family is exploratory** and is outside ``preregistration.md``'s frozen
confirmatory family: it changes neither ``N_max`` nor ``N_actual``.

The denominator is conditioned on non-isomorphism, and it is verified
--------------------------------------------------------------------
``E[d(G,H) | G !~ H]`` is taken over **distinct isomorphism classes**, settled
by an exact ``networkx`` VF2 test rather than by trusting that a draw without
repeated indices holds no repeated graphs.  It does not: 28 % of IAM Letter LOW
pairs are certified at GED 0.  A cheap ``(n, m, degree sequence)`` key buckets
the draw -- two graphs with different keys cannot be isomorphic -- and VF2
delivers the verdict inside a bucket, per the E2 rule that a certificate may
pre-filter but never decide.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import math
import os
import random
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from isalgraph.competitors import bootstrap, datasets, fixtures, registry
from isalgraph.competitors.admissibility import common
from isalgraph.competitors.base import Comparable, ReprBackend, VectorBackend

if TYPE_CHECKING:
    import networkx as nx
    import numpy.typing as npt

    from isalgraph.competitors.registry import AnyMetric

LOGGER = logging.getLogger(__name__)

#: OEIS A001349, connected graphs on ``n`` nodes up to isomorphism.  Asserted
#: rather than trusted: a wrong count means the atlas is not intact and every
#: number in Part A would be measured on the wrong population.
A001349: dict[int, int] = {2: 1, 3: 2, 4: 6, 5: 21, 6: 112, 7: 853}

#: Distance used where the T-04a grid admitted none.  The three representations
#: this applies to -- ``adjacency``, ``graph6``, ``sparse6`` -- are precisely
#: the ones E1 exists to characterise, so the substitution is recorded per row
#: rather than left to a reader to infer.
FALLBACK_METRIC = "levenshtein"

#: Deterministic ceiling on Part A's deduplicated encodes per ``(backend, n)``.
#: Machine-independent, unlike a wall-clock cap: the same run covers the same
#: graphs everywhere.  ``n = 7`` needs 1,866,256, so this does not bind for any
#: backend currently in the pool; a slower future one would report
#: ``exhaustive = False`` and name how many graphs it settled.
PART_A_MAX_ENCODES: int = 2_500_000

#: Fewer surviving pairs than this in a bootstrap replicate and it contributes
#: no ``psi``.  Matches :data:`isalgraph.competitors.bootstrap.MIN_PAIRS`.
MIN_PAIRS = bootstrap.MIN_PAIRS

#: ``--quick`` overrides.  Development only; every record carries ``quick``.
QUICK_MAX_N = 5
QUICK_GRAPHS = 25
QUICK_RELABELLINGS = 5
QUICK_RESAMPLES = 200
QUICK_DATASETS: tuple[str, ...] = ("iam_letter_low", "linux")


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExhaustiveRow:
    """Part A: one ``(backend, n)`` cell of the exhaustive characterisation.

    Attributes:
        backend: registry key.
        metric: the distance whose zero defines invariance.
        n_nodes: order of the graphs in this cell.
        n_graphs: connected graphs on *n_nodes* nodes, up to isomorphism.
        n_settled: graphs actually decided.  Below *n_graphs* only when
            :data:`PART_A_MAX_ENCODES` bound.
        n_invariant: settled graphs invariant under **every** relabelling.
        invariant_graph6: ``graph6`` certificate of each invariant graph, so a
            reader can check the characterisation rather than believe it.
        invariant_set_is_complete_graph: whether the invariant set is exactly
            ``{K_n}``.  T-04's claim for the ``n^2`` family, re-verified here
            independently of T-04's code.
        exhaustive: every distinct labelled copy of every settled graph was
            covered, or the graph was refuted by a witness.
        encodes: deduplicated encodes actually performed.
        orbit_total: distinct labelled graphs covered by the sweeps that ran to
            completion, i.e. ``sum n!/|Aut(G)|`` over the invariant graphs.
        n_skipped: graphs the backend raised on.  A raise is never counted as
            non-invariance.
        seconds: wall clock.
    """

    backend: str
    metric: str
    n_nodes: int
    n_graphs: int
    n_settled: int
    n_invariant: int
    invariant_graph6: tuple[str, ...]
    invariant_set_is_complete_graph: bool
    exhaustive: bool
    encodes: int
    orbit_total: int
    n_skipped: int
    seconds: float


@dataclass(frozen=True, slots=True)
class PsiRow:
    """Part B: one ``(backend, draw)`` cell of the cohort statement.

    Attributes:
        backend: registry key.
        metric: distance used.
        metric_is_fallback: the grid admitted no primary distance and
            :data:`FALLBACK_METRIC` stood in.  True for exactly the three
            representations E1 exists to characterise.
        draw: ``"pooled_S200"`` or a dataset name.
        n_graphs: graphs the backend encoded.
        n_skipped: graphs it raised on.
        n_self_pairs: ``n_graphs * relabellings`` self-distance trials.
        n_invariant_self_pairs: those with self-distance exactly 0.
        invariance_rate: their fraction.
        invariance_ci: exact Clopper-Pearson 95 % interval for that fraction.
        n_graphs_all_relabellings_invariant: graphs whose **every**
            relabelling gave 0.  This is the grid's F3 numerator at 50
            relabellings rather than 20, and it is a different quantity from
            *invariance_rate*: a graph with a large automorphism group scores
            self-distance 0 on a fair share of individual relabellings without
            being invariant on all of them.  Quoting one for the other is the
            mistake this field exists to make impossible.
        graphs_invariant_ci: exact interval for that per-graph fraction.
        non_invariance_rule_of_three: 95 % upper bound on the non-invariance
            rate when **zero** non-invariant events were seen, else ``None``.
            Printing ``0`` would assert impossibility from a finite sample.
        mean_self_distance: ``E[d(G, pi(G))]``, one weight per graph.
        mean_between_distance: ``E[d(G,H) | G !~ H]`` over the draw.
        psi: their ratio, or ``None`` when the denominator is 0.
        psi_ci: graph-level percentile bootstrap interval, or ``None``.
        n_between_pairs: non-isomorphic distinct pairs in the denominator.
        n_isomorphic_pairs: pairs excluded because the graphs are isomorphic.
        isomorphism_verified: the exclusion came from an exact VF2 verdict
            rather than from assuming a draw holds no repeated graphs.
        seconds: wall clock.
    """

    backend: str
    metric: str
    metric_is_fallback: bool
    draw: str
    n_graphs: int
    n_skipped: int
    n_self_pairs: int
    n_invariant_self_pairs: int
    invariance_rate: float | None
    invariance_ci: tuple[float, float] | None
    n_graphs_all_relabellings_invariant: int
    graphs_invariant_ci: tuple[float, float] | None
    non_invariance_rule_of_three: float | None
    mean_self_distance: float | None
    mean_between_distance: float | None
    psi: float | None
    psi_ci: tuple[float, float] | None
    n_between_pairs: int
    n_isomorphic_pairs: int
    isomorphism_verified: bool
    seconds: float


@dataclass(frozen=True, slots=True)
class PairedRow:
    """Part C: one Wilcoxon comparison of two representations' per-graph psi.

    Attributes:
        draw: which draw the per-graph statistics came from.
        backend_a: first representation.
        backend_b: second representation.
        n_paired: graphs both encoded.  Pairing is exact on those graphs --
            same graphs, same relabellings.
        median_psi_a: median per-graph psi for *backend_a*.
        median_psi_b: median per-graph psi for *backend_b*.
        statistic: Wilcoxon signed-rank statistic.
        p_raw: raw two-sided p-value, ``None`` when every pair is tied.
        p_holm: Holm-adjusted p-value across the pairwise family.
        rank_biserial: matched-pairs rank-biserial correlation.
        n_nonzero: non-tied pairs the test actually ran on.
    """

    draw: str
    backend_a: str
    backend_b: str
    n_paired: int
    median_psi_a: float
    median_psi_b: float
    statistic: float
    p_raw: float | None
    p_holm: float | None
    rank_biserial: float
    n_nonzero: int


@dataclass
class Cohort:
    """One draw: the graphs and the relabellings every backend shares.

    The relabelled copies are built **once** and handed to every backend.  A
    per-backend RNG would give each representation a different set of
    relabellings and silently destroy the pairing D-A5 relies on.

    Attributes:
        name: ``"pooled_S200"`` or a dataset name.
        graphs: the draw, in draw order.
        copies: ``copies[i]`` are the relabellings of ``graphs[i]``.
    """

    name: str
    graphs: list[nx.Graph] = field(default_factory=list)
    copies: list[list[nx.Graph]] = field(default_factory=list)


# --------------------------------------------------------------------------
# Part A -- exhaustive characterisation
# --------------------------------------------------------------------------


def _mapped_edges(graph: nx.Graph, mapping: dict[Any, int]) -> list[tuple[int, int]]:
    """Edges of *graph* under *mapping*, each ordered ``(min, max)``."""
    out: list[tuple[int, int]] = []
    for u, v in graph.edges():
        a, b = mapping[u], mapping[v]
        out.append((a, b) if a < b else (b, a))
    return out


def permuted_graph(graph: nx.Graph, perm: Sequence[int]) -> nx.Graph:
    """Relabel *graph* by *perm*, rebuilding in ascending insertion order.

    ``perm[i]`` is the new label of the ``i``-th node of ``sorted(graph)``.
    Nodes and edges are inserted in sorted order, so the **only** thing that
    differs between two permutations is the labelling.  That isolates label
    dependence from insertion-order dependence, which
    :func:`~isalgraph.competitors.fixtures.shuffled_copy` deliberately mixes.

    Args:
        graph: the graph to relabel.
        perm: a permutation of ``range(graph.number_of_nodes())``.

    Returns:
        The relabelled copy.
    """
    import networkx as nx

    mapping = {node: perm[i] for i, node in enumerate(sorted(graph.nodes()))}
    out = nx.Graph()
    out.add_nodes_from(sorted(mapping.values()))
    out.add_edges_from(sorted(_mapped_edges(graph, mapping)))
    return out


def _coder(
    backend_name: str, *, fit_on: Sequence[nx.Graph] | None = None
) -> Callable[[nx.Graph], Comparable]:
    """A one-argument encoder covering both backend protocols.

    A :class:`~isalgraph.competitors.base.VectorBackend` is fitted on *fit_on*,
    which must be the whole draw and never a batch.  The fit is inert with
    respect to the distance -- ``WLSubtree.features`` returns colour digests
    whatever was fitted -- but the contract is per dataset and honouring it
    costs one pass.

    Args:
        backend_name: registry key.
        fit_on: graphs to fit a vector backend on; ignored otherwise.

    Returns:
        A callable mapping a graph to its :class:`Comparable`.
    """
    backend = registry.get_backend(backend_name)
    if isinstance(backend, VectorBackend):
        vector_backend = backend
        vector_backend.fit(list(fit_on) if fit_on is not None else [])

        def encode_vector(graph: nx.Graph) -> Comparable:
            return dict(vector_backend.features(graph))

        return encode_vector

    repr_backend: ReprBackend = backend

    def encode_repr(graph: nx.Graph) -> Comparable:
        return repr_backend.encode(graph)

    return encode_repr


def _is_complete(graph: nx.Graph) -> bool:
    """Whether *graph* is ``K_n``."""
    n = int(graph.number_of_nodes())
    return int(graph.number_of_edges()) == n * (n - 1) // 2


def _graph6(graph: nx.Graph) -> str:
    """``graph6`` certificate of *graph*, so an invariant list is checkable."""
    import networkx as nx

    return str(nx.to_graph6_bytes(graph, header=False).decode("ascii").strip())


def exhaustive_invariance(
    backend_name: str,
    metric_name: str,
    *,
    max_n: int = common.EXHAUSTIVE_N_INVARIANCE,
    max_encodes: int = PART_A_MAX_ENCODES,
) -> list[ExhaustiveRow]:
    """Decide, per node count, exactly which connected graphs are invariant.

    Args:
        backend_name: registry key of the representation.
        metric_name: registry key of the distance whose zero defines
            invariance.
        max_n: largest node count, at most 7 (the atlas's range).
        max_encodes: deterministic ceiling on deduplicated encodes per node
            count.  A cell that hits it reports ``exhaustive = False``.

    Returns:
        One :class:`ExhaustiveRow` per node count in ``2..max_n``.

    Raises:
        AdmissibilityError: if the atlas's per-``n`` counts are not
            :data:`A001349`.  A wrong count means the population is wrong and
            every number below it is measured on the wrong thing.
    """
    atlas = common.connected_atlas(max_n)
    by_n: dict[int, list[nx.Graph]] = {}
    for graph in atlas:
        by_n.setdefault(graph.number_of_nodes(), []).append(graph)
    for n, expected in A001349.items():
        if n > max_n:
            continue
        got = len(by_n.get(n, []))
        if got != expected:
            raise common.AdmissibilityError(
                f"connected atlas has {got} graphs on {n} nodes, OEIS A001349 says "
                f"{expected}; the atlas is not intact and Part A would be measured "
                f"on the wrong population"
            )

    code = _coder(backend_name)
    metric = registry.get_metric(metric_name)
    return [
        _exhaustive_row(
            backend_name, metric_name, metric, code, n, by_n[n], max_encodes=max_encodes
        )
        for n in sorted(by_n)
    ]


def _orbit(graph: nx.Graph, perms: Sequence[Sequence[int]]) -> Iterator[nx.Graph]:
    """Every **distinct** labelled copy of *graph*, one per orbit member.

    ``|orbit| = n!/|Aut(G)|``.  Deduplicating by the permuted edge set is exact
    -- two permutations with the same edge set produce the same labelled graph
    and therefore the same encoding -- so the saving costs no coverage.

    Lazy on purpose: a representation that is not invariant is refuted by its
    second encoding, and building all ``n!`` copies first would spend 5,040
    graph constructions per graph to reach it.
    """
    import networkx as nx

    nodes = sorted(graph.nodes())
    seen: set[frozenset[tuple[int, int]]] = set()
    for perm in perms:
        mapping = {node: perm[i] for i, node in enumerate(nodes)}
        key = frozenset(_mapped_edges(graph, mapping))
        if key in seen:
            continue
        seen.add(key)
        copy = nx.Graph()
        copy.add_nodes_from(sorted(mapping.values()))
        copy.add_edges_from(sorted(key))
        yield copy


def _exhaustive_row(
    backend_name: str,
    metric_name: str,
    metric: AnyMetric,
    code: Callable[[nx.Graph], Comparable],
    n: int,
    graphs: Sequence[nx.Graph],
    *,
    max_encodes: int,
) -> ExhaustiveRow:
    """One ``(backend, n)`` cell.  See :func:`exhaustive_invariance`."""
    perms = list(itertools.permutations(range(n)))
    started = time.perf_counter()
    invariant: list[str] = []
    encodes = 0
    orbit_total = 0
    settled = 0
    skipped = 0
    exhaustive = True

    for graph in graphs:
        if encodes >= max_encodes:
            exhaustive = False
            break
        settled += 1
        first: Comparable | None = None
        invariant_here = True
        covered = 0
        for copy in _orbit(graph, perms):
            try:
                other = code(copy)
            except Exception:  # noqa: BLE001 - a raise is "skipped", never a result
                skipped += 1
                invariant_here = False
                break
            encodes += 1
            covered += 1
            if first is None:
                first = other
                continue
            if not metric.is_defined(first, other) or metric.distance(first, other) != 0.0:
                invariant_here = False
                break
        if invariant_here:
            orbit_total += covered
            invariant.append(_graph6(graph))

    complete = {_graph6(g) for g in graphs if _is_complete(g)}
    return ExhaustiveRow(
        backend=backend_name,
        metric=metric_name,
        n_nodes=n,
        n_graphs=len(graphs),
        n_settled=settled,
        n_invariant=len(invariant),
        invariant_graph6=tuple(sorted(invariant)),
        invariant_set_is_complete_graph=(exhaustive and set(invariant) == complete),
        exhaustive=exhaustive,
        encodes=encodes,
        orbit_total=orbit_total,
        n_skipped=skipped,
        seconds=time.perf_counter() - started,
    )


# --------------------------------------------------------------------------
# Part B -- the cohort statement
# --------------------------------------------------------------------------


def isomorphism_classes(graphs: Sequence[nx.Graph]) -> list[int]:
    """Class label per graph, decided by an exact VF2 test.

    A ``(n, m, sorted degree sequence)`` key buckets the draw; VF2 then decides
    inside a bucket against one representative per class already found.  The
    key is a **pre-filter** -- two graphs with different keys cannot be
    isomorphic -- and never the verdict.

    Args:
        graphs: the draw.

    Returns:
        ``labels`` with ``labels[i] == labels[j]`` iff ``graphs[i]`` and
        ``graphs[j]`` are isomorphic.
    """
    import networkx as nx

    buckets: dict[tuple[Any, ...], list[int]] = {}
    representatives: list[int] = []
    labels = [-1] * len(graphs)
    for i, graph in enumerate(graphs):
        key: tuple[Any, ...] = (
            graph.number_of_nodes(),
            graph.number_of_edges(),
            tuple(sorted(degree for _, degree in graph.degree())),
        )
        candidates = buckets.setdefault(key, [])
        for class_id in candidates:
            if nx.is_isomorphic(graphs[representatives[class_id]], graph):
                labels[i] = class_id
                break
        else:
            new_id = len(representatives)
            representatives.append(i)
            candidates.append(new_id)
            labels[i] = new_id
    return labels


def build_cohort(
    name: str,
    graphs: Sequence[nx.Graph],
    *,
    relabellings: int = common.RELABELLINGS,
    seed: int = common.SEED,
) -> Cohort:
    """Draw the shared relabellings once, for every backend to reuse.

    Args:
        name: draw name, carried into every row.
        graphs: the draw.
        relabellings: relabellings per graph.
        seed: seed of the single :class:`random.Random` driving them.

    Returns:
        The cohort.
    """
    rng = random.Random(seed)
    cohort = Cohort(name=name, graphs=list(graphs))
    for graph in graphs:
        cohort.copies.append([fixtures.shuffled_copy(graph, rng) for _ in range(relabellings)])
    return cohort


def _encode_cohort(
    code: Callable[[nx.Graph], Comparable], cohort: Cohort
) -> tuple[list[int], list[Comparable], list[list[Comparable]], int]:
    """Encode every base graph and its copies, dropping graphs that raise.

    A graph is kept only when its base **and every** copy encodes: a partial
    row would make the self-distance mean depend on which relabellings happened
    to survive, which is a selection effect rather than a measurement.
    """
    positions: list[int] = []
    bases: list[Comparable] = []
    copies: list[list[Comparable]] = []
    skipped = 0
    pairs = zip(cohort.graphs, cohort.copies, strict=True)
    for position, (graph, graph_copies) in enumerate(pairs):
        try:
            base = code(graph)
            encoded = [code(copy) for copy in graph_copies]
        except Exception:  # noqa: BLE001 - a raise is "skipped", never a result
            skipped += 1
            continue
        positions.append(position)
        bases.append(base)
        copies.append(encoded)
    return positions, bases, copies, skipped


def _psi_bootstrap(
    self_mean: Sequence[float],
    distances: npt.NDArray[Any],
    valid: npt.NDArray[Any],
    index: bootstrap.ResampleIndex,
    *,
    alpha: float = bootstrap.DEFAULT_ALPHA,
) -> tuple[float, float] | None:
    """Percentile bootstrap interval for ``psi``, resampling **graphs**.

    The resampling unit is the graph, not the pair: every pair shares a graph
    with ``2(k-2)`` others, so a pair-level interval would claim an
    independence the cohort does not have.  Self-pairs are dropped by the same
    construction :mod:`isalgraph.competitors.bootstrap` uses -- the diagonal of
    *valid* is never set, so a graph drawn into two slots contributes no pair
    with itself.

    Args:
        self_mean: per-graph mean self-distance, one entry per graph.
        distances: ``(k, k)`` matrix of between-graph distances.
        valid: ``(k, k)`` boolean, ``True`` where the pair is distinct and
            non-isomorphic.
        index: the shared resample matrix.
        alpha: two-sided miss rate.

    Returns:
        ``(low, high)``, or ``None`` when fewer than two replicates produced a
        finite ratio.
    """
    import numpy as np

    k = len(self_mean)
    means = np.asarray(self_mean, dtype=np.float64)
    flat_d = np.asarray(distances, dtype=np.float64).ravel()
    flat_v = np.asarray(valid, dtype=bool).ravel()
    upper_i, upper_j = np.triu_indices(k, 1)
    ratios: list[float] = []
    for r in range(index.resamples):
        drawn = index.draws[r]
        flat = drawn[upper_i].astype(np.int64) * k + drawn[upper_j].astype(np.int64)
        chosen = flat[flat_v[flat]]
        if chosen.size < MIN_PAIRS:
            continue
        denominator = float(flat_d[chosen].mean())
        if denominator <= 0.0:
            continue
        ratio = float(means[drawn].mean()) / denominator
        if math.isfinite(ratio):
            ratios.append(ratio)
    if len(ratios) < 2:
        return None
    low, high = np.percentile(
        np.asarray(ratios), [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)]
    )
    return float(low), float(high)


def cohort_psi(
    backend_name: str,
    metric_name: str,
    *,
    is_fallback: bool = False,
    cohort: Cohort,
    iso_labels: Sequence[int] | None = None,
    resamples: int = common.RESAMPLES,
    seed: int = common.SEED,
) -> tuple[PsiRow, dict[int, float]]:
    """One ``(backend, draw)`` cell, plus the per-graph psi Part C pairs on.

    Args:
        backend_name: registry key of the representation.
        metric_name: registry key of the distance.
        is_fallback: the grid admitted no primary distance for this backend.
        cohort: the draw and its shared relabellings.
        iso_labels: :func:`isomorphism_classes` output for ``cohort.graphs``.
            Computed here when ``None``; pass it in to share one verdict across
            every backend on a draw.
        resamples: bootstrap replicates.
        seed: bootstrap seed.

    Returns:
        The row, and ``{position in the draw: per-graph psi}``.
    """
    import numpy as np

    started = time.perf_counter()
    code = _coder(backend_name, fit_on=cohort.graphs)
    metric = registry.get_metric(metric_name)
    if iso_labels is None:
        iso_labels = isomorphism_classes(cohort.graphs)
    positions, bases, copies, skipped = _encode_cohort(code, cohort)
    k = len(positions)

    if k == 0:
        return (
            PsiRow(
                backend=backend_name,
                metric=metric_name,
                metric_is_fallback=is_fallback,
                draw=cohort.name,
                n_graphs=0,
                n_skipped=skipped,
                n_self_pairs=0,
                n_invariant_self_pairs=0,
                invariance_rate=None,
                invariance_ci=None,
                n_graphs_all_relabellings_invariant=0,
                graphs_invariant_ci=None,
                non_invariance_rule_of_three=None,
                mean_self_distance=None,
                mean_between_distance=None,
                psi=None,
                psi_ci=None,
                n_between_pairs=0,
                n_isomorphic_pairs=0,
                isomorphism_verified=True,
                seconds=time.perf_counter() - started,
            ),
            {},
        )

    # -- numerator: self-distances, one weight per graph --------------------
    # Invariance is a **per-graph** property and needs no second graph, so it
    # is measured whenever anything encoded.  Only psi needs a denominator and
    # therefore a pair, and it is the only thing a one-graph draw withholds.
    self_mean: list[float] = []
    n_self = 0
    n_invariant = 0
    n_graphs_invariant = 0
    for base, encoded in zip(bases, copies, strict=True):
        values = [metric.distance(base, other) for other in encoded]
        zeros = sum(1 for value in values if value == 0.0)
        n_self += len(values)
        n_invariant += zeros
        n_graphs_invariant += int(bool(values) and zeros == len(values))
        self_mean.append(sum(values) / len(values) if values else 0.0)

    # -- denominator: between-graph distances, conditioned on non-isomorphism
    labels = [iso_labels[position] for position in positions]
    distances = np.zeros((k, k), dtype=np.float64)
    valid = np.zeros((k, k), dtype=bool)
    n_between = 0
    n_isomorphic = 0
    for a in range(k):
        for b in range(a + 1, k):
            if labels[a] == labels[b]:
                n_isomorphic += 1
                continue
            value = metric.distance(bases[a], bases[b])
            distances[a, b] = distances[b, a] = value
            valid[a, b] = valid[b, a] = True
            n_between += 1

    mean_self = sum(self_mean) / k
    mean_between = float(distances[valid].mean()) if n_between else None
    psi = mean_self / mean_between if mean_between else None
    per_graph = (
        {positions[i]: self_mean[i] / mean_between for i in range(k)} if mean_between else {}
    )

    psi_ci = None
    if psi is not None:
        index = bootstrap.make_resample_index(k, resamples=resamples, seed=seed)
        psi_ci = _psi_bootstrap(self_mean, distances, valid, index)

    non_invariant = n_self - n_invariant
    row = PsiRow(
        backend=backend_name,
        metric=metric_name,
        metric_is_fallback=is_fallback,
        draw=cohort.name,
        n_graphs=k,
        n_skipped=skipped,
        n_self_pairs=n_self,
        n_invariant_self_pairs=n_invariant,
        invariance_rate=n_invariant / n_self if n_self else None,
        invariance_ci=common.clopper_pearson(n_invariant, n_self) if n_self else None,
        n_graphs_all_relabellings_invariant=n_graphs_invariant,
        graphs_invariant_ci=common.clopper_pearson(n_graphs_invariant, k),
        non_invariance_rule_of_three=(
            common.rule_of_three(n_self) if n_self and non_invariant == 0 else None
        ),
        mean_self_distance=mean_self,
        mean_between_distance=mean_between,
        psi=psi,
        psi_ci=psi_ci,
        n_between_pairs=n_between,
        n_isomorphic_pairs=n_isomorphic,
        isomorphism_verified=True,
        seconds=time.perf_counter() - started,
    )
    return row, per_graph


# --------------------------------------------------------------------------
# Part C -- the paired comparison (D-A5)
# --------------------------------------------------------------------------


def paired_comparisons(draw: str, per_graph: dict[str, dict[int, float]]) -> list[PairedRow]:
    """Wilcoxon signed-rank on per-graph psi, Holm-corrected over the family.

    Pairing is by position within the draw, which is exact: every
    representation saw the same graphs and the same relabellings.  A pair is
    restricted to the graphs **both** backends encoded, because a
    ``SUITE1_ONLY`` backend legitimately has fewer.

    A fully tied pair enters the Holm family at ``p = 1``: it is a comparison
    that was made, and dropping it would shrink ``m`` and make the surviving
    adjusted p-values smaller than the family they came from warrants.

    Args:
        draw: name of the draw, carried into every row.
        per_graph: ``{backend: {position: psi}}``.

    Returns:
        One row per unordered backend pair, with Holm-adjusted p-values.
    """
    import statistics

    names = sorted(per_graph)
    rows: list[PairedRow] = []
    raw: list[float] = []
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            shared = sorted(set(per_graph[first]) & set(per_graph[second]))
            if len(shared) < 2:
                continue
            a = [per_graph[first][p] for p in shared]
            b = [per_graph[second][p] for p in shared]
            result = common.wilcoxon_paired(a, b)
            p_raw = result["p"]
            rows.append(
                PairedRow(
                    draw=draw,
                    backend_a=first,
                    backend_b=second,
                    n_paired=len(shared),
                    median_psi_a=float(statistics.median(a)),
                    median_psi_b=float(statistics.median(b)),
                    statistic=float(result["statistic"] or 0.0),
                    p_raw=None if p_raw is None else float(p_raw),
                    p_holm=None,
                    rank_biserial=float(result["rank_biserial"] or 0.0),
                    n_nonzero=int(result["n_nonzero"] or 0),
                )
            )
            raw.append(1.0 if p_raw is None else float(p_raw))

    adjusted = common.holm(raw)
    out: list[PairedRow] = []
    for i, row in enumerate(rows):
        fields = asdict(row)
        fields["p_holm"] = None if row.p_raw is None else adjusted[i]
        out.append(PairedRow(**fields))
    return out


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def _resolve_metrics(
    grid_path: str, backends: Sequence[str], *, override: str | None = None
) -> dict[str, tuple[str, bool]]:
    """``{backend: (metric, is_fallback)}`` from the grid's primary block.

    Args:
        grid_path: ``grid_200.json``.
        backends: representations to resolve.
        override: force every backend onto one distance.  **Supplementary
            only.**  The protocol fixes :data:`FALLBACK_METRIC` where the grid
            admitted none, and the main run must be produced without this; it
            exists so the sensitivity of psi to that choice is reproducible
            from committed code rather than from a scratch script.  The record
            carries ``metric_override`` whenever it is set.

    Returns:
        The mapping.  Under an override every entry is marked a fallback,
        because none of them is the grid's selection.
    """
    if override is not None:
        return {name: (override, True) for name in backends}
    primary = common.primary_distances(grid_path)
    out: dict[str, tuple[str, bool]] = {}
    for name in backends:
        chosen = primary.get(name)
        out[name] = (chosen, False) if chosen else (FALLBACK_METRIC, True)
    return out


def _draws(names: Sequence[str], *, k: int, seed: int) -> list[tuple[str, list[nx.Graph]]]:
    """The pooled ``S200`` draw, then one seed-42 ``k``-graph draw per dataset."""
    pooled = datasets.pooled_stratified_sample(tuple(names), k, seed=seed)
    out: list[tuple[str, list[nx.Graph]]] = [
        ("pooled_S200", [datasets.load(r.dataset).graphs[r.index] for r in pooled])
    ]
    for name in names:
        cohort = datasets.load(name)
        out.append((name, [cohort.graphs[i] for i in cohort.sample(k, seed=seed)]))
    return out


def run_e1(
    grid_path: str,
    *,
    backends: Sequence[str] | None = None,
    dataset_names: Sequence[str] | None = None,
    parts: str = "ABC",
    max_n: int = common.EXHAUSTIVE_N_INVARIANCE,
    n_graphs: int = common.N_POOLED,
    relabellings: int = common.RELABELLINGS,
    resamples: int = common.RESAMPLES,
    seed: int = common.SEED,
    quick: bool = False,
    metric_override: str | None = None,
) -> dict[str, Any]:
    """Run E1 and return its record.

    Args:
        grid_path: ``grid_200.json``, read only for its primary-distance block.
        backends: representations to measure.  ``None`` uses every available
            non-baseline backend.
        dataset_names: datasets for Part B.  ``None`` uses every dataset on
            disk.
        parts: any subset of ``"ABC"``.
        max_n: Part A's largest node count.
        n_graphs: draw size for Part B.
        relabellings: relabellings per graph.
        resamples: bootstrap replicates.
        seed: the single seed for every draw, relabelling and resample.
        quick: development mode; recorded in the output.
        metric_override: supplementary sensitivity run; see
            :func:`_resolve_metrics`.  ``None`` for the protocol run.

    Returns:
        The record :func:`common.write_result` serialises.
    """
    started = time.perf_counter()
    names = tuple(backends) if backends else registry.available_backends()
    metrics = _resolve_metrics(grid_path, names, override=metric_override)
    record: dict[str, Any] = {
        "quick": quick,
        "metric_override": metric_override,
        "parts": parts,
        "backends": list(names),
        "metric_per_backend": {k: {"metric": m, "fallback": f} for k, (m, f) in metrics.items()},
        "fallback_metric": FALLBACK_METRIC,
        "grid": os.path.abspath(grid_path),
        "n_graphs_requested": n_graphs,
        "relabellings_used": relabellings,
        "resamples_used": resamples,
        "part_a_max_encodes": PART_A_MAX_ENCODES,
        "part_a_tests": (
            "label invariance under canonical ascending insertion order; strictly "
            "weaker than part B's shuffled_copy, which also varies insertion and "
            "edge order. Part A's invariant set upper-bounds part B's"
        ),
    }

    if "A" in parts:
        record["atlas_counts_expected"] = {str(n): c for n, c in A001349.items() if n <= max_n}
        rows_a: list[dict[str, Any]] = []
        for name in names:
            metric_name, _ = metrics[name]
            LOGGER.info("part A: %s under %s", name, metric_name)
            for row in exhaustive_invariance(name, metric_name, max_n=max_n):
                LOGGER.info(
                    "  n=%d invariant %d/%d complete-only=%s encodes=%d %.1fs",
                    row.n_nodes,
                    row.n_invariant,
                    row.n_settled,
                    row.invariant_set_is_complete_graph,
                    row.encodes,
                    row.seconds,
                )
                rows_a.append(asdict(row))
        record["exhaustive"] = rows_a

    if "B" in parts or "C" in parts:
        chosen = tuple(dataset_names) if dataset_names else datasets.available_datasets()
        record["datasets"] = list(chosen)
        rows_b: list[dict[str, Any]] = []
        rows_c: list[dict[str, Any]] = []
        for draw_name, graphs in _draws(chosen, k=n_graphs, seed=seed):
            LOGGER.info("part B: draw %s, %d graphs", draw_name, len(graphs))
            cohort = build_cohort(draw_name, graphs, relabellings=relabellings, seed=seed)
            iso_labels = isomorphism_classes(graphs)
            per_graph: dict[str, dict[int, float]] = {}
            for name in names:
                metric_name, is_fallback = metrics[name]
                psi_row, psi_by_graph = cohort_psi(
                    name,
                    metric_name,
                    is_fallback=is_fallback,
                    cohort=cohort,
                    iso_labels=iso_labels,
                    resamples=resamples,
                    seed=seed,
                )
                rate = psi_row.invariance_rate
                LOGGER.info(
                    "  %-20s psi=%s inv=%s (%d/%d) %.1fs",
                    name,
                    "n/a" if psi_row.psi is None else f"{psi_row.psi:.4f}",
                    "n/a" if rate is None else f"{rate:.4f}",
                    psi_row.n_invariant_self_pairs,
                    psi_row.n_self_pairs,
                    psi_row.seconds,
                )
                rows_b.append(asdict(psi_row))
                if psi_by_graph:
                    per_graph[name] = psi_by_graph
            if "C" in parts:
                rows_c.extend(asdict(p) for p in paired_comparisons(draw_name, per_graph))
        if "B" in parts:
            record["cohort"] = rows_b
        if "C" in parts:
            record["paired"] = rows_c
            record["paired_note"] = (
                "EXPLORATORY (D-A5). Outside preregistration.md's frozen confirmatory "
                "family; changes neither N_max nor N_actual"
            )

    record["wall_seconds"] = time.perf_counter() - started
    return record


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m isalgraph.competitors.admissibility.e1_invariance",
        description="E1: relabelling invariance and the separation ratio psi.",
    )
    parser.add_argument("--grid", required=True, help="grid_200.json, for the primary distances")
    parser.add_argument("--out", required=True, help="destination .json")
    parser.add_argument("--parts", default="ABC", help="any subset of ABC")
    parser.add_argument("--backends", nargs="*", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--quick", action="store_true", help="development-sized run")
    parser.add_argument(
        "--metric-override",
        default=None,
        help="SUPPLEMENTARY: force every backend onto one distance, to measure "
        "how much psi depends on the fallback the protocol fixed. Never the "
        "protocol run; the record carries metric_override",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: command line, ``None`` for ``sys.argv[1:]``.

    Returns:
        Process exit status.
    """
    args = _parse(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    kwargs: dict[str, Any] = {
        "backends": args.backends,
        "dataset_names": args.datasets,
        "parts": args.parts,
        "quick": args.quick,
        "metric_override": args.metric_override,
    }
    if args.quick:
        kwargs.update(
            max_n=QUICK_MAX_N,
            n_graphs=QUICK_GRAPHS,
            relabellings=QUICK_RELABELLINGS,
            resamples=QUICK_RESAMPLES,
        )
        if args.datasets is None:
            kwargs["dataset_names"] = QUICK_DATASETS
    record = run_e1(args.grid, **kwargs)
    common.write_result(args.out, "E1", record)
    LOGGER.info("wrote %s in %.1fs", args.out, record["wall_seconds"])
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())


__all__ = [
    "A001349",
    "FALLBACK_METRIC",
    "MIN_PAIRS",
    "PART_A_MAX_ENCODES",
    "Cohort",
    "ExhaustiveRow",
    "PairedRow",
    "PsiRow",
    "build_cohort",
    "cohort_psi",
    "exhaustive_invariance",
    "isomorphism_classes",
    "main",
    "paired_comparisons",
    "permuted_graph",
    "run_e1",
]
