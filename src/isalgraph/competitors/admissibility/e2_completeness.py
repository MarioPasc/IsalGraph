"""E2 -- completeness: is the induced distance a **metric** or a **pseudometric**?

E1 settles one direction of injectivity, ``d_R(G, π(G)) = 0`` for every
relabelling ``π``.  A representation can pass that and still fail the other:
``d_R(G, H) = 0`` while ``G ≇ H``.  On the quotient space of isomorphism
classes -- the space graph edit distance lives on -- the first failure makes
``d_R`` *not a function at all* (class I) and the second makes it a
**pseudometric** rather than a metric (class II).  Only class III supports a
claim of the form *distance zero certifies isomorphism*.

Three parts, in increasing cost:

**A. Proof by exhibition.**  ``K₃,₃`` and the triangular prism are both
connected, both 3-regular on six vertices, and not isomorphic.  1-WL cannot
tell them apart -- the colouring is constant after round 1, so refinement
never starts and no number of rounds fixes it -- while every canonical backend
separates them.  One fixture settles the qualitative claim and no statistics
are involved.

**B. The collision rate on real data.**  Per-dataset seed-42 200-graph draws,
all ten datasets, every ``C(200,2)`` pair, each representation under its
primary distance from the T-04a grid.  Every pair the representation puts at
distance zero is settled by an **exact** VF2 isomorphism test.

**C. The class table**, which is what T-17's AE.3 comparison consumes for the
*uniqueness* axis.

Two rules this module enforces mechanically:

- **Never a bare ``0``.**  Zero collisions in ``N`` trials is reported as the
  rule-of-three upper bound ``3/N``; printing ``0`` asserts impossibility from
  a finite sample.
- **VF2 is the only verdict.**  A nauty canonical certificate groups graphs so
  that the number of VF2 calls stays linear in the draw rather than quadratic,
  but every ``G ≅ H`` answer traces back to a VF2 run -- either directly on the
  pair, or through a certified representative by transitivity.

Frozen protocol, §3: ``.claude/notes/review/tasks/T-04a-admissibility-protocol.md``.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from isalgraph.competitors import datasets, fixtures
from isalgraph.competitors.admissibility import common
from isalgraph.competitors.base import Capability, Comparable, VectorBackend
from isalgraph.competitors.registry import (
    AnyMetric,
    available_backends,
    get_backend,
    get_metric,
)

if TYPE_CHECKING:
    import networkx as nx

LOGGER = logging.getLogger(__name__)

#: Default location of the T-04a grid whose ``primary_distance`` block selects
#: the distance each representation is measured under.
DEFAULT_GRID = "/media/mpascual/Sandisk2TB/research/isalgraph/T-04a/grid_200.json"

#: A distance is treated as zero at or below this.  Both admissible distances
#: are exact -- Levenshtein is integer-valued and the WL kernel is a sum of
#: products of small integers -- so the tolerance never binds.  ``near_zero``
#: in every record is the count of pairs that fell in ``(0, ZERO_TOL]``, i.e.
#: the measured proof that it never binds.
ZERO_TOL = 1e-12

#: Distance used for a representation the grid admitted none for, so that
#: part A can still report a number for the class-I family (protocol §2).
FALLBACK_METRIC = "levenshtein"

#: Datasets and draw size for ``--quick``.  A smoke path, never a result.
QUICK_DATASETS: tuple[str, ...] = ("iam_letter_low", "linux")
QUICK_K = 40

#: Collision witnesses retained per (representation, dataset).  A collision is
#: a stop-and-ask, so the first few pairs are all anyone needs to reproduce it.
MAX_WITNESSES = 20


# --------------------------------------------------------------------------
# The grid, read for what E2 is forbidden to recompute
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GridView:
    """The slice of the T-04a grid E2 reads.

    Attributes:
        path: provenance.
        primary: backend -> its primary distance, ``None`` where the grid
            admitted none.
        f3: backend -> {metric -> ``"k/n"``} over **candidate** cells only.
            The ``size_null`` metric is a descriptive null and is invariant
            for every backend, so including it would report the class-I family
            as invariant.
        class_i: backends that are not relabelling-invariant.  **Taken from
            the grid's F3, never recomputed here** -- E2 measures the other
            direction.
    """

    path: str
    primary: dict[str, str | None]
    f3: dict[str, dict[str, str]]
    class_i: frozenset[str]


def _f3_is_full(record: str) -> bool:
    """Whether an ``"k/n"`` F3 record means every relabelling was invariant."""
    k, _, n = record.partition("/")
    return k == n and n not in ("", "0")


def load_grid(path: str) -> GridView:
    """Read the grid's ``primary_distance`` and F3 blocks.

    Args:
        path: the grid JSON, normally ``grid_200.json``.

    Returns:
        The view.

    Raises:
        AdmissibilityError: if the file has no ``cells`` block.
    """
    import json

    primary = common.primary_distances(path)
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    cells = payload.get("cells")
    if not isinstance(cells, list):
        raise common.AdmissibilityError(f"{path} has no cells block")

    f3: dict[str, dict[str, str]] = {}
    for cell in cells:
        if not cell.get("candidate") or cell.get("f3_invariant") is None:
            continue
        f3.setdefault(str(cell["backend"]), {})[str(cell["metric"])] = str(cell["f3_invariant"])

    class_i = frozenset(
        name
        for name, records in f3.items()
        if records and not any(_f3_is_full(rec) for rec in records.values())
    )
    return GridView(path=path, primary=primary, f3=f3, class_i=class_i)


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------


def encode_all(
    backend_name: str, graphs: Sequence[nx.Graph]
) -> tuple[list[Comparable | None], Counter[str]]:
    """Encode every graph once, ``None`` where the backend refused.

    A refusal is a datum, not a stop: ``agm_cam`` and ``isalgraph_canonical``
    declare :attr:`Capability.SUITE1_ONLY` and raise above that scale, and
    ``min_dfs`` raises on its projection cap.  The pairs those graphs would
    have entered are excluded from the denominator and the count is reported.

    The one :class:`VectorBackend` is fitted over the **whole draw**, never per
    batch: a per-batch fit produces a different vocabulary and therefore a
    distance matrix that depends on batching order.

    Args:
        backend_name: registry key.
        graphs: the draw, in order.

    Returns:
        ``(items, failures)`` with *items* aligned to *graphs* and *failures*
        counted by exception type.
    """
    backend = get_backend(backend_name)
    failures: Counter[str] = Counter()
    if isinstance(backend, VectorBackend):
        try:
            backend.fit(list(graphs))
        except Exception as exc:  # a failed fit is a datum for the whole column
            failures[type(exc).__name__] += len(graphs)
            return [None] * len(graphs), failures

    items: list[Comparable | None] = []
    for graph in graphs:
        try:
            item: Comparable = (
                dict(backend.features(graph))
                if isinstance(backend, VectorBackend)
                else backend.encode(graph)
            )
        except Exception as exc:  # a refusal is a measured ceiling, not a stop
            failures[type(exc).__name__] += 1
            items.append(None)
            continue
        items.append(item)
    return items, failures


# --------------------------------------------------------------------------
# The isomorphism oracle: VF2 decides, nauty only groups
# --------------------------------------------------------------------------


@dataclass
class IsomorphismOracle:
    """Exact ``G ≅ H`` verdicts over one draw, with the VF2 cost bounded.

    Running VF2 on all ``C(200,2)`` pairs is the cost trap this class exists to
    avoid, and running it only on the pairs a representation put at distance
    zero is still quadratic for a representation that collides often.  So the
    draw is partitioned by a **canonical certificate** -- the ``nauty_graph6``
    encoding, itself one of the pool's declared complete invariants -- and each
    class is *certified* by VF2 against its representative, at most ``n - 1``
    calls per class.

    Every verdict then traces to VF2:

    - same class: both members were certified isomorphic to one representative,
      so they are isomorphic by transitivity;
    - different classes: VF2 runs on the pair directly.

    **The class partition is therefore never a verdict, but it is load-bearing
    for the converse direction.**  :attr:`iso_pairs` is the within-class pair
    set, so "these are all the isomorphic pairs" is conditional on the
    certificate being a *complete* invariant -- soundness is certified here,
    completeness is what part A and E1 establish independently.

    Attributes:
        graphs: the draw.
        keys: certificate per graph, ``None`` when the certifier refused.
        certificate_defects: members VF2 rejected from the class the
            certificate assigned them to.  A nonzero count means the
            certificate is not sound and is escalated by the caller through
            ``nauty_graph6``'s own collision count.
    """

    graphs: Sequence[nx.Graph]
    keys: Sequence[object | None]
    _classes: dict[int, list[int]] = field(default_factory=dict, init=False)
    _class_of: list[int] = field(default_factory=list, init=False)
    certificate_defects: list[tuple[int, int]] = field(default_factory=list, init=False)
    vf2_calls: int = field(default=0, init=False)
    vf2_seconds: float = field(default=0.0, init=False)
    vf2_slowest_s: float = field(default=0.0, init=False)
    _cache: dict[tuple[int, int], bool] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        buckets: dict[object, list[int]] = {}
        singletons: list[int] = []
        for index, key in enumerate(self.keys):
            if key is None:
                singletons.append(index)
            else:
                buckets.setdefault(key, []).append(index)

        self._class_of = [-1] * len(self.graphs)
        next_id = 0
        for members in buckets.values():
            representative = members[0]
            certified = [representative]
            for member in members[1:]:
                if self._vf2(representative, member):
                    certified.append(member)
                else:
                    # The certificate put two non-isomorphic graphs together.
                    self.certificate_defects.append((representative, member))
                    singletons.append(member)
            self._classes[next_id] = certified
            for member in certified:
                self._class_of[member] = next_id
            next_id += 1
        for index in singletons:
            self._classes[next_id] = [index]
            self._class_of[index] = next_id
            next_id += 1

    def _vf2(self, i: int, j: int) -> bool:
        """One exact VF2 run, memoised and timed."""
        import networkx as nx

        key = (i, j) if i <= j else (j, i)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        start = time.perf_counter()
        verdict = bool(nx.is_isomorphic(self.graphs[i], self.graphs[j]))
        elapsed = time.perf_counter() - start
        self.vf2_calls += 1
        self.vf2_seconds += elapsed
        self.vf2_slowest_s = max(self.vf2_slowest_s, elapsed)
        self._cache[key] = verdict
        return verdict

    def are_isomorphic(self, i: int, j: int) -> bool:
        """Whether graphs *i* and *j* are isomorphic.  **VF2 decides.**"""
        if self._class_of[i] == self._class_of[j]:
            return True
        return self._vf2(i, j)

    @property
    def n_classes(self) -> int:
        """Number of certified isomorphism classes in the draw."""
        return len(self._classes)

    def iso_pairs(self) -> list[tuple[int, int]]:
        """Every within-class pair, i.e. every isomorphic pair of the draw.

        Sound by VF2 certification; complete only if the certificate is a
        complete invariant.  See the class docstring.
        """
        out: list[tuple[int, int]] = []
        for members in self._classes.values():
            out.extend(itertools.combinations(sorted(members), 2))
        out.sort()
        return out


# --------------------------------------------------------------------------
# The two directions
# --------------------------------------------------------------------------


def zero_pairs(
    items: Sequence[Comparable | None], metric: AnyMetric
) -> tuple[list[tuple[int, int]], int, int, int]:
    """Every pair at distance zero, plus the denominators.

    Args:
        items: encodings aligned to the draw, ``None`` where the backend
            refused.
        metric: the representation's primary distance.

    Returns:
        ``(pairs, evaluated, undefined, near_zero)``.  *evaluated* counts pairs
        both of whose encodings exist and on which the metric is defined;
        *undefined* counts pairs the metric declined (F1); *near_zero* counts
        pairs in ``(0, ZERO_TOL]``, which must be 0 -- both admissible
        distances are exact, so a nonzero count means the tolerance is doing
        work it was never meant to do.
    """
    pairs: list[tuple[int, int]] = []
    evaluated = 0
    undefined = 0
    near_zero = 0
    for i, j in itertools.combinations(range(len(items)), 2):
        a, b = items[i], items[j]
        if a is None or b is None:
            continue
        if not metric.is_defined(a, b):
            undefined += 1
            continue
        evaluated += 1
        distance = metric.distance(a, b)
        if distance <= 0.0:
            pairs.append((i, j))
        elif distance <= ZERO_TOL:
            near_zero += 1
            pairs.append((i, j))
    return pairs, evaluated, undefined, near_zero


def pair_distance(
    items: Sequence[Comparable | None], metric: AnyMetric
) -> Callable[[int, int], float | None]:
    """A ``(i, j) -> d`` reader over one encoded draw.

    A module-level factory rather than a closure inside the representation
    loop, so that the metric it reads is bound at construction and cannot be
    the loop's later value.

    Args:
        items: encodings aligned to the draw, ``None`` where the backend
            refused.
        metric: the distance to read.

    Returns:
        A callable returning the distance, or ``None`` when either encoding is
        missing or the metric is undefined on the pair.
    """

    def read(i: int, j: int) -> float | None:
        a, b = items[i], items[j]
        if a is None or b is None or not metric.is_defined(a, b):
            return None
        return float(metric.distance(a, b))

    return read


def converse_check(
    name: str,
    iso_pairs: Sequence[tuple[int, int]],
    distance: Callable[[int, int], float | None],
) -> int:
    """Assert an invariant representation puts every isomorphic pair at zero.

    This is the direction that **cannot** be a property of the method: a
    representation whose encoding is a function of the isomorphism class
    assigns two isomorphic graphs the same encoding and therefore distance
    zero.  A violation is a defect in our code, so it is **raised, never
    reported** (protocol §7).

    Args:
        name: representation, for the message.
        iso_pairs: pairs a VF2 run certified isomorphic.
        distance: ``(i, j) -> d`` or ``None`` when the pair was not evaluable
            because the backend refused one of the graphs.

    Returns:
        The number of pairs actually checked.

    Raises:
        AdmissibilityError: on the first pair with ``d > ZERO_TOL``.
    """
    checked = 0
    for i, j in iso_pairs:
        value = distance(i, j)
        if value is None:
            continue
        checked += 1
        if value > ZERO_TOL:
            raise common.AdmissibilityError(
                f"{name!r} separates two graphs a VF2 run certified isomorphic: "
                f"pair ({i}, {j}) at distance {value!r}. An invariant representation "
                f"cannot do this, so it is a defect in our code and not a property "
                f"of the method -- protocol §7 escalates rather than reports it"
            )
    return checked


def _rate(events: int, trials: int) -> dict[str, Any]:
    """A proportion with its exact interval, and never a bare ``0``."""
    lo, hi = common.clopper_pearson(events, trials)
    record: dict[str, Any] = {
        "events": events,
        "trials": trials,
        "point": (events / trials) if trials else None,
        "ci95_clopper_pearson": [lo, hi],
    }
    if trials == 0:
        record["reported"] = "no trials"
    elif events == 0:
        bound = common.rule_of_three(trials)
        record["rule_of_three_upper"] = bound
        record["reported"] = f"0/{trials}; <= {bound:.3g} at 95 % (rule of three)"
    else:
        record["reported"] = f"{events}/{trials} = {events / trials:.4g} [{lo:.3g}, {hi:.3g}]"
    return record


# --------------------------------------------------------------------------
# Part A -- proof by exhibition
# --------------------------------------------------------------------------


def separation_witness(names: Sequence[str], grid: GridView) -> dict[str, Any]:
    """``K₃,₃`` versus the triangular prism, under every representation.

    Both connected, both 3-regular on six vertices, not isomorphic.  1-WL
    assigns them identical colour histograms at every round, so ``wl_subtree``
    must give distance exactly zero and every declared complete invariant must
    give a positive distance.

    Args:
        names: representations to evaluate.
        grid: supplies each one's primary distance.

    Returns:
        The record, including the fixture's own asserted properties so a
        reader never has to take "3-regular, non-isomorphic" on trust.
    """
    import networkx as nx

    left = fixtures.to_networkx(fixtures.K33)
    right = fixtures.to_networkx(fixtures.PRISM)
    degrees_left = {int(d) for _, d in left.degree()}
    degrees_right = {int(d) for _, d in right.degree()}

    record: dict[str, Any] = {
        "left": "k33",
        "right": "prism",
        "n_nodes": [left.number_of_nodes(), right.number_of_nodes()],
        "n_edges": [left.number_of_edges(), right.number_of_edges()],
        "degree_sets": [sorted(degrees_left), sorted(degrees_right)],
        "both_3_regular": degrees_left == degrees_right == {3},
        "non_isomorphic_vf2": not bool(nx.is_isomorphic(left, right)),
        "representations": {},
    }

    for name in names:
        metric_name = grid.primary.get(name) or FALLBACK_METRIC
        entry: dict[str, Any] = {
            "metric": metric_name,
            "metric_source": "grid" if grid.primary.get(name) else "fallback",
            "declared_complete_invariant": _declares_complete(name),
        }
        try:
            items, failures = encode_all(name, [left, right])
            if items[0] is None or items[1] is None:
                entry["error"] = f"encode refused: {dict(failures)}"
                entry["separates"] = None
            else:
                metric = get_metric(metric_name)
                distance = metric.distance(items[0], items[1])
                entry["distance"] = distance
                entry["separates"] = distance > ZERO_TOL
        except Exception as exc:  # a broken backend is a datum for the witness
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["separates"] = None
        record["representations"][name] = entry
    return record


def _declares_complete(name: str) -> bool:
    """Whether the backend declares :attr:`Capability.COMPLETE_INVARIANT`."""
    try:
        return Capability.COMPLETE_INVARIANT in get_backend(name).capabilities
    except Exception:  # an unconstructible backend declares nothing
        return False


# --------------------------------------------------------------------------
# Part B -- the collision rate on real data
# --------------------------------------------------------------------------


def scan_dataset(dataset: str, names: Sequence[str], grid: GridView, k: int) -> dict[str, Any]:
    """One dataset's collision scan over every invariant representation.

    Args:
        dataset: cohort name.
        names: representations to measure -- the caller has already dropped
            the class-I family, which has no admissible distance.
        grid: supplies each representation's primary distance.
        k: draw size.

    Returns:
        The dataset's record.

    Raises:
        AdmissibilityError: through :func:`converse_check`, if an invariant
            representation separates two isomorphic graphs.
    """
    cohort = datasets.load(dataset)
    indices = cohort.sample(k, seed=common.SEED)
    graphs = [cohort.graphs[i] for i in indices]
    n_graphs = len(graphs)
    LOGGER.info("%s: %d graphs drawn (seed %d)", dataset, n_graphs, common.SEED)

    certifier_items, certifier_failures = encode_all("nauty_graph6", graphs)
    keys: list[object | None] = [
        None if item is None else getattr(item, "symbols", None) for item in certifier_items
    ]
    oracle = IsomorphismOracle(graphs=graphs, keys=keys)
    iso_pairs = oracle.iso_pairs()
    LOGGER.info(
        "%s: %d isomorphism classes, %d isomorphic pairs, %d VF2 calls in %.1f s",
        dataset,
        oracle.n_classes,
        len(iso_pairs),
        oracle.vf2_calls,
        oracle.vf2_seconds,
    )

    record: dict[str, Any] = {
        "n_graphs": n_graphs,
        "n_pairs": n_graphs * (n_graphs - 1) // 2,
        "sample_indices": list(indices),
        "n_nodes": {
            "min": min(g.number_of_nodes() for g in graphs),
            "max": max(g.number_of_nodes() for g in graphs),
            "mean": sum(g.number_of_nodes() for g in graphs) / n_graphs,
        },
        "certifier": {
            "backend": "nauty_graph6",
            "failures": dict(certifier_failures),
            "n_classes": oracle.n_classes,
            "n_iso_pairs": len(iso_pairs),
            "certificate_defects": [list(p) for p in oracle.certificate_defects],
            "note": (
                "the certificate groups graphs so the VF2 cost stays linear in the "
                "draw; every G ~= H verdict is a VF2 run, directly or through a "
                "certified representative. The iso-pair set is therefore SOUND by "
                "VF2 and COMPLETE only if the certificate is a complete invariant, "
                "which part A and E1 establish independently"
            ),
        },
        "representations": {},
    }

    for name in names:
        metric_name = grid.primary[name]
        if metric_name is None:
            continue
        metric = get_metric(metric_name)
        started = time.perf_counter()
        items, failures = encode_all(name, graphs)
        pairs, evaluated, undefined, near_zero = zero_pairs(items, metric)

        collisions: list[tuple[int, int]] = []
        for i, j in pairs:
            if not oracle.are_isomorphic(i, j):
                collisions.append((i, j))

        checked = converse_check(name, iso_pairs, pair_distance(items, metric))

        record["representations"][name] = {
            "metric": metric_name,
            "declared_complete_invariant": _declares_complete(name),
            "encoded": sum(1 for item in items if item is not None),
            "encode_failures": dict(failures),
            "pairs_evaluated": evaluated,
            "pairs_undefined": undefined,
            "zero_pairs": len(pairs),
            "near_zero_nonzero": near_zero,
            "collisions": len(collisions),
            "collision_witnesses": [list(p) for p in collisions[:MAX_WITNESSES]],
            "collision_rate_among_zero": _rate(len(collisions), len(pairs)),
            "collision_rate_among_pairs": _rate(len(collisions), evaluated),
            "converse_pairs_checked": checked,
            "converse_violations": 0,
            "seconds": time.perf_counter() - started,
        }
        LOGGER.info(
            "%s / %s: %d zero pairs, %d collisions, %d converse pairs checked",
            dataset,
            name,
            len(pairs),
            len(collisions),
            checked,
        )

    record["vf2"] = {
        "calls": oracle.vf2_calls,
        "seconds": oracle.vf2_seconds,
        "slowest_call_s": oracle.vf2_slowest_s,
    }
    return record


def pool(per_dataset: dict[str, dict[str, Any]], names: Sequence[str]) -> dict[str, Any]:
    """Sum the counts over datasets and re-interval them.

    The interval is Clopper-Pearson per D-A4, and it is **anticonservative
    here**: pairs within a dataset share graphs, so they are not independent
    trials.  The caveat travels with the number rather than being left to a
    reader, and the per-dataset rates are the ones to compare.

    Args:
        per_dataset: :func:`scan_dataset` records by dataset.
        names: representations.

    Returns:
        Pooled record per representation.
    """
    out: dict[str, Any] = {}
    for name in names:
        collisions = 0
        zeros = 0
        evaluated = 0
        checked = 0
        datasets_seen = 0
        for record in per_dataset.values():
            entry = record["representations"].get(name)
            if entry is None:
                continue
            datasets_seen += 1
            collisions += int(entry["collisions"])
            zeros += int(entry["zero_pairs"])
            evaluated += int(entry["pairs_evaluated"])
            checked += int(entry["converse_pairs_checked"])
        out[name] = {
            "datasets": datasets_seen,
            "collisions": collisions,
            "zero_pairs": zeros,
            "pairs_evaluated": evaluated,
            "converse_pairs_checked": checked,
            "collision_rate_among_zero": _rate(collisions, zeros),
            "collision_rate_among_pairs": _rate(collisions, evaluated),
            "interval_caveat": (
                "pairs within a dataset share graphs, so the binomial interval "
                "understates uncertainty; D-A4 fixes Clopper-Pearson for proportions "
                "and the per-dataset rates are the comparable ones"
            ),
        }
    return out


# --------------------------------------------------------------------------
# Part C -- the class table
# --------------------------------------------------------------------------


def classify(
    names: Sequence[str],
    grid: GridView,
    witness: dict[str, Any],
    pooled: dict[str, Any],
) -> dict[str, Any]:
    """Assign each representation its class, as data rather than as prose.

    ``I`` comes from the grid's F3 and is not recomputed.  ``II`` is assigned
    on **either** a measured collision or a failure to separate the part-A
    witness -- one exhibited collision is a proof, and a sampled rate of zero
    does not overturn it.  ``III`` carries the rule-of-three bound, because a
    finite sample licenses an upper bound and not the claim of impossibility.

    Args:
        names: every representation in the pool.
        grid: the F3 source.
        witness: :func:`separation_witness`'s record.
        pooled: :func:`pool`'s record.

    Returns:
        Backend -> class record.
    """
    out: dict[str, Any] = {}
    for name in names:
        entry: dict[str, Any] = {
            "primary_distance": grid.primary.get(name),
            "grid_f3": grid.f3.get(name, {}),
            "declared_complete_invariant": _declares_complete(name),
            "separates_k33_prism": witness["representations"].get(name, {}).get("separates"),
        }
        if name in grid.class_i:
            entry["class"] = "I"
            entry["reason"] = (
                "not relabelling-invariant: the grid's F3 fails under every candidate "
                "distance, so d_R is not a function on isomorphism classes and E2's "
                "question does not arise"
            )
            out[name] = entry
            continue

        stats = pooled.get(name)
        if stats is None:
            entry["class"] = None
            entry["reason"] = "no admissible distance and no F3 record; not classified"
            out[name] = entry
            continue

        entry["collisions"] = stats["collisions"]
        entry["collision_rate_among_zero"] = stats["collision_rate_among_zero"]
        entry["collision_rate_among_pairs"] = stats["collision_rate_among_pairs"]
        exhibited = entry["separates_k33_prism"] is False
        if stats["collisions"] > 0 or exhibited:
            entry["class"] = "II"
            entry["reason"] = (
                "invariant but not injective on isomorphism classes: "
                + ("K33 and the triangular prism receive distance 0. " if exhibited else "")
                + f"{stats['collisions']} collisions over {stats['zero_pairs']} "
                f"zero-distance pairs. d_R is a PSEUDOMETRIC and is barred from any "
                f"claim that d = 0 certifies isomorphism (D-A2)"
            )
        else:
            bound = stats["collision_rate_among_zero"].get("rule_of_three_upper")
            entry["class"] = "III"
            entry["reason"] = (
                f"invariant, and no collision observed over {stats['zero_pairs']} "
                f"zero-distance pairs; the 95 % rule-of-three upper bound on the "
                f"collision rate is {bound:.3g}"
                if bound is not None
                else "invariant, and no collision observed"
            )
        out[name] = entry
    return out


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def run(*, grid_path: str, dataset_names: Sequence[str], k: int) -> dict[str, Any]:
    """Run parts A, B and C.

    Args:
        grid_path: the T-04a grid JSON.
        dataset_names: cohorts to scan.
        k: per-dataset draw size.

    Returns:
        The full E2 payload.

    Raises:
        AdmissibilityError: if an invariant representation separates two
            isomorphic graphs (the converse direction, protocol §7).
    """
    import isalgraph

    started = time.perf_counter()
    grid = load_grid(grid_path)
    pool_names = list(available_backends())

    witness = separation_witness(pool_names, grid)
    measured = [n for n in pool_names if grid.primary.get(n) is not None]
    LOGGER.info("part A done; part B over %d representations: %s", len(measured), measured)

    per_dataset: dict[str, dict[str, Any]] = {}
    for dataset in dataset_names:
        per_dataset[dataset] = scan_dataset(dataset, measured, grid, k)

    pooled = pool(per_dataset, measured)
    table = classify(pool_names, grid, witness, pooled)

    escalations: list[dict[str, Any]] = []
    for name, entry in table.items():
        if not entry.get("declared_complete_invariant"):
            continue
        if entry.get("separates_k33_prism") is False:
            escalations.append(
                {
                    "kind": "class_iii_fails_witness",
                    "backend": name,
                    "detail": "declares COMPLETE_INVARIANT yet gives K33 and the prism d = 0",
                }
            )
        if int(entry.get("collisions", 0) or 0) > 0:
            escalations.append(
                {
                    "kind": "class_iii_collision",
                    "backend": name,
                    "collisions": entry["collisions"],
                    "detail": "declares COMPLETE_INVARIANT yet collides on real data",
                }
            )
    for dataset, record in per_dataset.items():
        defects = record["certifier"]["certificate_defects"]
        if defects:
            escalations.append(
                {
                    "kind": "certificate_defect",
                    "dataset": dataset,
                    "pairs": defects,
                    "detail": "nauty canonical certificate grouped non-isomorphic graphs",
                }
            )

    return {
        "engine": isalgraph.engine(),
        "grid_path": grid_path,
        "datasets": list(dataset_names),
        "k_per_dataset": k,
        "part_a_witness": witness,
        "part_b_per_dataset": per_dataset,
        "part_b_pooled": pooled,
        "part_c_classes": table,
        "escalations": escalations,
        "wall_s": time.perf_counter() - started,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: command-line arguments, ``None`` for ``sys.argv[1:]``.

    Returns:
        Process exit status.

    Raises:
        AdmissibilityError: on a converse violation, or on a class-III
            collision **after** the payload is written, so the evidence
            survives the stop.
    """
    parser = argparse.ArgumentParser(description="E2 -- completeness of the induced distance")
    parser.add_argument("--grid", default=DEFAULT_GRID, help="T-04a grid JSON")
    parser.add_argument("--out", required=True, help="destination JSON")
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"smoke path: {QUICK_DATASETS} at k = {QUICK_K}. Never a result.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    dataset_names = QUICK_DATASETS if args.quick else datasets.ALL_DATASETS
    k = QUICK_K if args.quick else common.N_PER_DATASET
    payload = run(grid_path=str(args.grid), dataset_names=dataset_names, k=k)
    payload["quick"] = bool(args.quick)
    common.write_result(str(args.out), "E2", payload)
    LOGGER.info("wrote %s in %.1f s", args.out, payload["wall_s"])

    if payload["escalations"]:
        raise common.AdmissibilityError(
            f"E2 escalations, payload written to {args.out}: {payload['escalations']}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "DEFAULT_GRID",
    "FALLBACK_METRIC",
    "MAX_WITNESSES",
    "QUICK_DATASETS",
    "QUICK_K",
    "ZERO_TOL",
    "GridView",
    "IsomorphismOracle",
    "classify",
    "converse_check",
    "encode_all",
    "load_grid",
    "main",
    "pair_distance",
    "pool",
    "run",
    "scan_dataset",
    "separation_witness",
    "zero_pairs",
]
