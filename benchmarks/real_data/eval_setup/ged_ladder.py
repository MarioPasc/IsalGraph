"""The T-05 calibration ladder: exact GED above ``n = 12``, one rung at a time.

The bracket ``LB <= GED <= UB`` was *selected* where exact GED exists -- T-03's
census stops at ``n = 12`` -- and is *licensed* all the way to ``n = 98``. That
gap is demand AE.1, and T-27 limitation 1 states that no bake-off against ground
truth can close it because ground truth does not exist up there. **Every node the
exact solver buys narrows it**, so this module buys as many as the budget allows
and reports honestly where it stops.

Design, frozen in ``.claude/notes/review/tasks/T-05-design.md`` §6 and reproduced
here so the code cannot drift from it::

    population     all Suite-2 pairs with max(n1, n2) = n, pooled over the ten
                   datasets, one population per rung
    rungs          n = 13, 14, 15, 16, 17, 18
    pairs per rung 250, stratified by source dataset proportionally to that
                   dataset's pair mass at that n, minimum 20 per contributor
    seed           42 throughout
    exact solver   networkx.graph_edit_distance under cost model D6 [1,1,0,1,1,0]
    budget         1,200 s wall per pair
    non-completion interval-censored [LB, UB] under D11 -- never dropped, never
                   promoted to exact
    truncation     at the first rung whose certification rate falls below 25 %,
                   reported as the measured exact-GED ceiling

Three things this module refuses to do, each because the alternative fails
silently rather than loudly.

**It does not use ``ANCHOR_AWARE_GED``.** T-03-design.md amendment 2 retired it:
measured on Picasso it is non-deterministic on 14 of 15 real AIDS pairs, wrong on
4 of 18 against brute force, and it reports ``LB == UB`` -- a *false optimality
certificate*, which is worse than a wrong value because it defeats the one check
designed to catch a wrong value. :func:`~ged_backends._reject_retired` refuses to
construct a backend that names it, and nothing here names it.

**It does not read completion off the returned value.**
``nx.graph_edit_distance(timeout=t)`` returns its best-found-so-far cost when the
budget expires; it neither raises nor returns ``None`` unless *no* complete edit
path was found at all. Every "exact GED" matrix in the submitted study was
produced that way (T-03-design.md §0). Completion is decided by
:func:`ged_backends.astar_completed`, that is, by whether the search terminated
before its deadline, and this module reaches the solver only through
:class:`ged_backends.NetworkxBackend`, which already implements that rule.

**It does not tighten the recorded upper bound with the A\\* best-so-far cost.**
:class:`ged_backends.ExactPlusBoundsBackend` takes ``ub = min(GEDLIB, A* cost)``,
which is correct for a census but destroys this ticket's measurement twice over:
on a certified pair the recorded ``ub`` becomes the exact value (amendment 6
verified exactly that on 234,258 AIDS pairs), so ``rho(exact, UB)`` degenerates to
1 and the containment check ``lb <= exact <= ub`` becomes vacuous. A wall-clock
budget also makes an A\\* best-so-far cost machine-dependent, and a bound that
moves with the node it ran on is not reproducible. ``lb`` and ``ub`` here are the
raw GEDLIB bounds under their frozen options strings, for every pair, certified or
not. **The options string is part of the method name**: GEDLIB's upper bounds
change on 74-94 % of pairs between runs at library defaults (amendment 6, T-27
§4.2).

The best-so-far cost is not thrown away, though. It lands in its own
``ub_astar_bestsofar`` column, **censored pairs only**, so whoever wants the
tighter D11 interval can have it while ``ub`` keeps a single meaning. That column
is flagged machine-dependent in the metadata and no §6 analysis consumes it.

Output, one file per rung, flat arrays of length ``P``::

    ladder/rung_{n}.npz   dataset_key <U | pair_i pair_j n_max int32 |
                          exact lb ub ub_astar_bestsofar float64 |
                          certified bool | seconds float32 |
                          metadata <U (JSON, 0-d)
    ladder/manifest.json  the same summary across rungs, plus the ceiling

``exact`` is ``inf`` on a censored pair -- never ``nan``, matching T-03's census,
so a consumer selects on ``certified`` and filters with :func:`numpy.isfinite`.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

# Importable both as ``python -m benchmarks.real_data.eval_setup.ged_ladder`` from
# the repository root (the way the SLURM worker invokes it) and as a bare module
# from inside ``eval_setup/``.
if __package__:
    from .export_graphs import load_exported
    from .ged_backends import CERT_TOL, GedlibBackend, NetworkxBackend
    from .ged_bounds import GRAPHEDX_COSTS, UNIT_COSTS, EditCosts
else:  # pragma: no cover - only when run as a bare script from eval_setup/
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from export_graphs import load_exported  # noqa: E402
    from ged_backends import CERT_TOL, GedlibBackend, NetworkxBackend  # noqa: E402
    from ged_bounds import GRAPHEDX_COSTS, UNIT_COSTS, EditCosts  # noqa: E402

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_BUDGET_SECONDS",
    "DEFAULT_MIN_PER_DATASET",
    "DEFAULT_PAIRS_PER_RUNG",
    "DEFAULT_RUNGS",
    "DEFAULT_SEED",
    "DEFAULT_TRUNCATE_BELOW",
    "SCHEMA_VERSION",
    "SUITE2_KEYS",
    "LadderError",
    "PairRecord",
    "RungResult",
    "RungSample",
    "allocate_quota",
    "load_rung_npz",
    "main",
    "rung_population",
    "run_rung",
    "sample_rung",
    "solve_pair",
    "write_manifest",
    "write_rung_npz",
]


class LadderError(Exception):
    """Any violation of the frozen §6 design or of the output contract."""


SCHEMA_VERSION = "ladder-1"

DEFAULT_RUNGS: tuple[int, ...] = (13, 14, 15, 16, 17, 18)
DEFAULT_PAIRS_PER_RUNG = 250
DEFAULT_MIN_PER_DATASET = 20
DEFAULT_SEED = 42
DEFAULT_BUDGET_SECONDS = 1200.0
DEFAULT_TRUNCATE_BELOW = 0.25

#: The ten Suite-2 cohorts, in a **fixed alphabetical order**. The order is what
#: gives each dataset its ordinal, and the ordinal is what seeds its per-rung
#: generator, so this tuple is part of the sampler's reproducibility contract:
#: reordering it changes which pairs seed 42 selects.
SUITE2_KEYS: tuple[str, ...] = (
    "aids_graphedx",
    "aids_iam",
    "coil_del",
    "grec",
    "iam_letter_high",
    "iam_letter_low",
    "iam_letter_med",
    "linux",
    "mutagenicity",
    "protein",
)

#: Roles from CONTRACTS §3 / ``slurm/approx_ged/_env.sh``, verbatim. `BRANCH_FAST`
#: is the lower bound and `BIPARTITE` the loose Riesen-Bunke reference upper bound;
#: both run ``--threads 1`` because the parallelism is the process pool.
DEFAULT_LB_METHOD = "BRANCH_FAST"
DEFAULT_LB_OPTIONS = "--threads 1"
DEFAULT_UB_METHOD = "BIPARTITE"
DEFAULT_UB_OPTIONS = "--threads 1"

_INF = float("inf")

_COST_MODELS: dict[str, EditCosts] = {"unit": UNIT_COSTS, "graphedx": GRAPHEDX_COSTS}


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #


@cache
def _code_commit() -> str:
    """Return the commit that is *running*, not the one ``.git`` happens to hold.

    ``ISALGRAPH_CODE_COMMIT`` takes precedence over ``git rev-parse`` because the
    cluster checkout is populated by ``rsync``, so its ``.git`` stays pinned at
    whatever was last pulled there. Measured 2026-08-13: a banner announced
    ``d6a9f4b`` while executing code eleven commits ahead. Provenance that names
    the wrong commit is worse than none, because it looks checkable.

    **Cached, and warmed by** :func:`main` **before any pair is solved.** The
    first version of this function resolved the commit at metadata-build time,
    which is after the rung has run. My own rung-13 pilot then recorded a commit
    three commits ahead of the code that produced it, because the working tree
    moved while the ladder was solving. That is the same class of defect as the
    ``rsync``-pinned banner, arriving from the opposite direction, and a run
    spanning hours is exactly where it bites: caching pins the answer to when the
    process started, which is the last moment at which it is certainly right.

    Returns
    -------
    str
        A commit hash, or ``'unknown'`` outside a checkout.
    """
    declared = os.environ.get("ISALGRAPH_CODE_COMMIT")
    if declared:
        return declared.strip()
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:  # pragma: no cover - git absent
        return "unknown"
    return out.stdout.strip() or "unknown"


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# the population at one rung
# --------------------------------------------------------------------------- #


def rung_population(n_nodes: np.ndarray, rung: int) -> np.ndarray:
    """Enumerate one dataset's pairs whose larger graph has exactly ``rung`` nodes.

    The population of rung ``n`` is ``{(i, j) : i < j, max(n_i, n_j) = n}``. It
    splits into two disjoint blocks -- both endpoints at ``n``, and one endpoint
    at ``n`` with the other strictly below -- which is what makes it countable in
    closed form as ``C(a, 2) + a * b`` before it is enumerated.

    Parameters
    ----------
    n_nodes : numpy.ndarray
        Node count per graph, in the dataset's exported order.
    rung : int
        The rung, i.e. the exact value of ``max(n_1, n_2)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(M, 2)``, dtype ``int32``, rows ``(i, j)`` with ``i < j``, sorted
        lexicographically by ``(i, j)``. Empty with shape ``(0, 2)`` when the
        dataset contributes nothing at this rung.
    """
    counts = np.asarray(n_nodes).astype(np.int64, copy=False)
    at = np.flatnonzero(counts == int(rung)).astype(np.int64)
    below = np.flatnonzero(counts < int(rung)).astype(np.int64)

    blocks: list[np.ndarray] = []
    if at.size >= 2:
        ii, jj = np.triu_indices(at.size, k=1)
        blocks.append(np.column_stack((at[ii], at[jj])))
    if at.size and below.size:
        left = np.repeat(at, below.size)
        right = np.tile(below, at.size)
        blocks.append(np.column_stack((left, right)))
    if not blocks:
        return np.empty((0, 2), dtype=np.int32)

    pairs = np.concatenate(blocks, axis=0)
    # Canonicalise to i < j: the cross block puts the large graph first, and the
    # index below it may be either side of it in the export order.
    lo = np.minimum(pairs[:, 0], pairs[:, 1])
    hi = np.maximum(pairs[:, 0], pairs[:, 1])
    order = np.lexsort((hi, lo))
    return np.column_stack((lo[order], hi[order])).astype(np.int32, copy=False)


def rung_mass(n_nodes: np.ndarray, rung: int) -> int:
    """Count one dataset's rung population without enumerating it.

    Parameters
    ----------
    n_nodes : numpy.ndarray
        Node count per graph.
    rung : int
        The rung.

    Returns
    -------
    int
        ``C(a, 2) + a * b`` where ``a`` graphs have exactly ``rung`` nodes and
        ``b`` have fewer.
    """
    counts = np.asarray(n_nodes)
    a = int(np.count_nonzero(counts == int(rung)))
    b = int(np.count_nonzero(counts < int(rung)))
    return a * (a - 1) // 2 + a * b


# --------------------------------------------------------------------------- #
# allocation
# --------------------------------------------------------------------------- #


def allocate_quota(
    masses: dict[str, int],
    total: int,
    minimum: int,
) -> dict[str, int]:
    """Split ``total`` pairs across contributing datasets, floor first.

    §6 asks for "stratified by source dataset proportionally to that dataset's
    pair mass at that ``n``, minimum 20 per contributing dataset", which under-
    determines the order of the two operations. It is resolved here, once, and
    the resolution is part of the frozen design: **the floor is satisfied first**,
    and the residual ``total - sum(floors)`` is then allocated proportionally to
    full mass by largest remainder, capped at each dataset's available mass. The
    alternative -- proportional first, then raise anyone below the floor -- has to
    take the extra pairs back off someone, and there is no principled donor.

    A dataset with zero mass at this rung is not a contributor and receives
    nothing; it is reported as absent rather than allocated a floor it cannot
    fill.

    Parameters
    ----------
    masses : dict of str to int
        Population size per dataset at this rung. Zero-mass entries are ignored.
    total : int
        Pairs to allocate across the rung.
    minimum : int
        Per-contributor floor, itself capped by that contributor's mass.

    Returns
    -------
    dict of str to int
        Allocation per contributing dataset, keys sorted. Sums to ``total``
        unless the whole rung holds fewer than ``total`` pairs, in which case it
        sums to the rung's total mass.

    Raises
    ------
    LadderError
        If ``total`` or ``minimum`` is negative.
    """
    if total < 0 or minimum < 0:
        raise LadderError(f"total and minimum must be non-negative, got {total}, {minimum}")

    caps = {k: int(masses[k]) for k in sorted(masses) if int(masses[k]) > 0}
    if not caps:
        return {}
    if sum(caps.values()) <= total:
        # The rung is smaller than the quota: take all of it, and let the caller
        # report a short rung rather than silently resampling with replacement.
        return dict(caps)

    alloc = {k: min(minimum, caps[k]) for k in caps}
    if sum(alloc.values()) > total:
        # More contributors than the quota can floor. Not reachable with ten
        # datasets, a floor of 20 and a quota of 250, but a design change could
        # reach it, and dropping the floor loudly beats over-allocating quietly.
        logger.warning(
            "per-dataset floor %d over %d contributors exceeds the quota %d; "
            "the floor is dropped for this rung and the split is purely proportional",
            minimum,
            len(caps),
            total,
        )
        alloc = dict.fromkeys(caps, 0)

    keys = sorted(caps)
    for _ in range(len(keys) + 2):
        remaining = total - sum(alloc.values())
        if remaining <= 0:
            break
        open_keys = [k for k in keys if alloc[k] < caps[k]]
        if not open_keys:
            break
        weight = float(sum(caps[k] for k in open_keys))
        ideal = {k: remaining * caps[k] / weight for k in open_keys}
        for k in open_keys:
            alloc[k] += min(int(math.floor(ideal[k])), caps[k] - alloc[k])
        left = total - sum(alloc.values())
        if left <= 0:
            break
        # Largest fractional remainder, ties broken by key so the split does not
        # depend on dict ordering.
        order = sorted(open_keys, key=lambda k: (-(ideal[k] - math.floor(ideal[k])), k))
        progressed = False
        for k in order:
            if left == 0:
                break
            if alloc[k] < caps[k]:
                alloc[k] += 1
                left -= 1
                progressed = True
        if not progressed:
            break

    return {k: v for k, v in sorted(alloc.items()) if v > 0}


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #


def _select_indices(rng: np.random.Generator, m: int, k: int) -> np.ndarray:
    """Draw ``k`` distinct positions from ``range(m)`` without replacement.

    Implemented as "one uniform key per element, take the ``k`` smallest by a
    stable argsort" rather than through :meth:`numpy.random.Generator.choice`.
    ``choice`` and ``permutation`` are free to change their internal algorithm
    between NumPy releases; the raw ``random()`` stream of PCG64 is a documented
    stability guarantee, and a stable argsort is deterministic on top of it. The
    cost is one float64 per population element, which at the largest rung
    population measured here (177,123 pairs) is 1.4 MB.

    Parameters
    ----------
    rng : numpy.random.Generator
        Seeded generator.
    m : int
        Population size.
    k : int
        Sample size. Values at or above ``m`` return the whole population.

    Returns
    -------
    numpy.ndarray
        Ascending positions, dtype ``int64``.
    """
    if k >= m:
        return np.arange(m, dtype=np.int64)
    keys = rng.random(m)
    return np.sort(np.argsort(keys, kind="stable")[:k]).astype(np.int64, copy=False)


@dataclass(frozen=True, slots=True)
class RungSample:
    """The pairs one rung will solve, and the bookkeeping that explains them.

    Attributes
    ----------
    rung : int
        ``max(n_1, n_2)`` shared by every pair.
    dataset_key : numpy.ndarray
        Source dataset per pair, dtype ``<U``.
    pair_i, pair_j : numpy.ndarray
        Indices into that dataset's exported graph order, ``i < j``, ``int32``.
    masses : dict of str to int
        Population size per dataset at this rung, including zero-mass datasets.
    allocation : dict of str to int
        Requested pairs per contributing dataset.
    realised : dict of str to int
        Pairs actually drawn per dataset. Differs from ``allocation`` only when a
        dataset holds fewer pairs than it was allocated, which the allocator's
        caps already prevent; kept because a silent disagreement between the two
        would be exactly the kind of defect this ticket exists to catch.
    """

    rung: int
    dataset_key: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray
    masses: dict[str, int]
    allocation: dict[str, int]
    realised: dict[str, int]

    @property
    def n_pairs(self) -> int:
        """Number of sampled pairs."""
        return int(self.pair_i.shape[0])

    @property
    def is_empty(self) -> bool:
        """Whether the rung has no eligible pair in any dataset."""
        return self.n_pairs == 0


def sample_rung(
    n_nodes_by_key: dict[str, np.ndarray],
    rung: int,
    *,
    total: int = DEFAULT_PAIRS_PER_RUNG,
    minimum: int = DEFAULT_MIN_PER_DATASET,
    seed: int = DEFAULT_SEED,
) -> RungSample:
    """Draw one rung's stratified sample, reproducibly from ``seed`` alone.

    Every generator is derived as ``default_rng([seed, rung, ordinal])`` where
    ``ordinal`` is the dataset's position in :data:`SUITE2_KEYS`. Nothing is
    seeded from iteration order, wall-clock time, or a generator threaded across
    datasets, so the pairs drawn for one dataset at one rung do not move when
    another dataset's mass changes or when the rungs are run in a different
    order, or in separate processes.

    Parameters
    ----------
    n_nodes_by_key : dict of str to numpy.ndarray
        Node counts per graph, per dataset, in exported order.
    rung : int
        The rung to sample.
    total : int, optional
        Pairs per rung. Frozen at 250.
    minimum : int, optional
        Per-contributor floor. Frozen at 20.
    seed : int, optional
        Master seed. Frozen at 42.

    Returns
    -------
    RungSample
        Possibly empty, which is a reported outcome and not an error.

    Raises
    ------
    LadderError
        If a dataset key is not one of the ten Suite-2 cohorts.
    """
    unknown = sorted(set(n_nodes_by_key) - set(SUITE2_KEYS))
    if unknown:
        raise LadderError(
            f"unknown dataset key(s) {unknown}; the ordinal that seeds the sampler is "
            f"the position in SUITE2_KEYS, so a key outside it has no reproducible seed"
        )

    masses = {k: rung_mass(n_nodes_by_key[k], rung) for k in sorted(n_nodes_by_key)}
    allocation = allocate_quota(masses, total, minimum)

    keys_out: list[str] = []
    i_out: list[np.ndarray] = []
    j_out: list[np.ndarray] = []
    realised: dict[str, int] = {}

    for key in sorted(allocation):
        want = allocation[key]
        population = rung_population(n_nodes_by_key[key], rung)
        if population.shape[0] != masses[key]:
            raise LadderError(
                f"{key} rung {rung}: enumerated {population.shape[0]} pairs but the "
                f"closed-form mass is {masses[key]}; one of the two is wrong"
            )
        rng = np.random.default_rng([int(seed), int(rung), SUITE2_KEYS.index(key)])
        take = _select_indices(rng, population.shape[0], want)
        chosen = population[take]
        keys_out.extend([key] * chosen.shape[0])
        i_out.append(chosen[:, 0])
        j_out.append(chosen[:, 1])
        realised[key] = int(chosen.shape[0])

    if not keys_out:
        return RungSample(
            rung=int(rung),
            dataset_key=np.empty(0, dtype="<U1"),
            pair_i=np.empty(0, dtype=np.int32),
            pair_j=np.empty(0, dtype=np.int32),
            masses=masses,
            allocation=allocation,
            realised=realised,
        )

    return RungSample(
        rung=int(rung),
        dataset_key=np.asarray(keys_out, dtype=np.str_),
        pair_i=np.concatenate(i_out).astype(np.int32, copy=False),
        pair_j=np.concatenate(j_out).astype(np.int32, copy=False),
        masses=masses,
        allocation=allocation,
        realised=realised,
    )


# --------------------------------------------------------------------------- #
# solving one pair
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PairRecord:
    """One solved pair, in the terms the output contract records.

    Attributes
    ----------
    dataset_key : str
        Source cohort.
    pair_i, pair_j : int
        Indices into that cohort's exported graph order.
    n_max : int
        ``max(n_1, n_2)``; equals the rung by construction.
    exact : float
        The certified distance, or ``inf`` where the pair is censored. Never a
        best-so-far cost.
    lb, ub : float
        The raw bounds under their frozen options strings. Always finite.
    certified : bool
        Whether the exact search terminated before its deadline.
    seconds : float
        Wall time for the whole pair -- bounds plus exact search.
    ub_astar_bestsofar : float
        The cost of the best complete edit path A* constructed before its budget
        expired, on **censored pairs only**; ``inf`` on a certified pair and on a
        censored pair where no complete path was found. It is a valid upper
        bound -- the path exists -- and usually a tighter one than ``ub``, so it
        narrows the D11 interval for anyone who wants that. It is **not** part of
        the reproducible bracket: how far A* gets in 1,200 s is a function of the
        machine, so this column moves between nodes while ``lb`` and ``ub`` do
        not. Recorded, never consumed by a §6 analysis.
    """

    dataset_key: str
    pair_i: int
    pair_j: int
    n_max: int
    exact: float
    lb: float
    ub: float
    certified: bool
    seconds: float
    ub_astar_bestsofar: float = _INF


def solve_pair(
    g1: Any,
    g2: Any,
    *,
    bounds_backend: Any,
    exact_backend: NetworkxBackend,
    bounds_kind: str,
) -> tuple[float, float, float, bool, float, float]:
    """Bracket one pair and try to certify it, keeping the two independent.

    Parameters
    ----------
    g1, g2 : networkx.Graph
        The pair.
    bounds_backend : GedlibBackend or NetworkxBackend
        Source of ``(lb, ub)``.
    exact_backend : NetworkxBackend
        Source of the exact value. Only :meth:`~NetworkxBackend.solve_exact` is
        called, so the bounds this backend could also produce are never mixed in.
    bounds_kind : {'gedlib', 'networkx'}
        Which accessor the bounds backend exposes.

    Returns
    -------
    tuple
        ``(exact, lb, ub, certified, seconds, ub_astar_bestsofar)`` with
        ``exact = inf`` when the search did not terminate.
        ``ub_astar_bestsofar`` is the cost of the best complete edit path A*
        built before its budget expired, on censored pairs only, and ``inf``
        otherwise. It is recorded beside the bracket rather than folded into it.

    Raises
    ------
    LadderError
        If a bound is non-finite, if the bracket is inverted, or if a certified
        optimum falls outside its own bracket. The bounds and the exact value
        come from independent implementations here, so a contradiction means one
        of them is wrong and the run must stop rather than record a plausible
        number.
    """
    t0 = time.perf_counter()
    if bounds_kind == "gedlib":
        lb, ub = bounds_backend.bounds(g1, g2)
    else:
        lb, ub = bounds_backend.heuristic_bracket(g1, g2)
    lb, ub = float(lb), float(ub)

    if not (math.isfinite(lb) and math.isfinite(ub)):
        raise LadderError(
            f"non-finite bracket [{lb}, {ub}]; a censored pair must still carry two "
            "finite ends or D11 has no interval to report"
        )
    if lb > ub + CERT_TOL:
        raise LadderError(f"inverted bracket: lb {lb} exceeds ub {ub}")

    exact, best_cost, _solver_seconds, _timed_out = exact_backend.solve_exact(g1, g2)

    # Deliberately NOT ``ub = min(ub, best_cost)``. See the module docstring: the
    # A* best-so-far cost is a valid upper bound but a machine-dependent one, and
    # folding it in makes ``ub`` equal ``exact`` on every certified pair, which is
    # precisely the quantity this ladder measures. It is kept in its own column
    # instead, so the tighter D11 interval is recoverable without giving ``ub``
    # two meanings.
    if exact is not None and not (lb - CERT_TOL <= exact <= ub + CERT_TOL):
        raise LadderError(
            f"certified optimum {exact} outside its bracket [{lb}, {ub}]; the bounds "
            "and the exact solver are independent implementations, so one is wrong"
        )

    best = _INF
    if exact is None and math.isfinite(best_cost):
        best = float(best_cost)

    seconds = time.perf_counter() - t0
    return (
        float(exact) if exact is not None else _INF,
        lb,
        ub,
        exact is not None,
        seconds,
        best,
    )


# --------------------------------------------------------------------------- #
# the process pool
# --------------------------------------------------------------------------- #

_WORKER: dict[str, Any] = {}


def _build_backends(
    costs: EditCosts,
    bounds_kind: str,
    lb_method: str,
    lb_options: str,
    ub_method: str,
    ub_options: str,
    budget_seconds: float,
) -> tuple[Any, NetworkxBackend]:
    """Construct the two backends one worker needs.

    ``lb_symmetry_probes=0`` is not a tuning choice. The default 32 evaluates the
    lower bound in both orientations for the first 32 pairs *of each backend
    instance* and keeps the larger, so a pair's recorded ``lb`` would depend on
    its position in whichever process happened to take it. That is exactly the
    kind of order dependence that makes a rerun disagree with itself, so the
    probe is off and the lower bound is a function of the pair alone.

    Parameters
    ----------
    costs : EditCosts
        Cost model, D6 for production.
    bounds_kind : {'gedlib', 'networkx'}
        Bounds source. ``'networkx'`` uses ``ged_bounds``' own BRANCH/BP
        implementations, which need no compiled library; CLAUDE.md requires the
        two to agree on the same pairs, and this is the switch that lets them be
        compared.
    lb_method, lb_options, ub_method, ub_options : str
        GEDLIB method names and their verbatim options strings.
    budget_seconds : float
        Per-pair wall budget for the exact solver.

    Returns
    -------
    tuple
        ``(bounds_backend, exact_backend)``.
    """
    if bounds_kind == "gedlib":
        bounds: Any = GedlibBackend(
            costs,
            lb_method=lb_method,
            lb_options=lb_options,
            ub_method=ub_method,
            ub_options=ub_options,
            compute="both",
            lb_symmetry_probes=0,
            env_mode="per-pair",
        )
    elif bounds_kind == "networkx":
        bounds = NetworkxBackend(costs, timeout_s=budget_seconds)
    else:
        raise LadderError(f"unknown bounds kind {bounds_kind!r}; expected gedlib or networkx")
    return bounds, NetworkxBackend(costs, timeout_s=budget_seconds)


def _init_worker(
    exported_dir: str,
    keys: list[str],
    cost_model: str,
    bounds_kind: str,
    lb_method: str,
    lb_options: str,
    ub_method: str,
    ub_options: str,
    budget_seconds: float,
) -> None:
    """Load the cohorts and build the backends once per pool process."""
    costs = _COST_MODELS[cost_model]
    graphs = {k: load_exported(Path(exported_dir) / f"{k}.npz").graphs for k in keys}
    bounds, exact = _build_backends(
        costs, bounds_kind, lb_method, lb_options, ub_method, ub_options, budget_seconds
    )
    _WORKER.clear()
    _WORKER.update(graphs=graphs, bounds=bounds, exact=exact, bounds_kind=bounds_kind, rung=None)


def _solve_task(
    task: tuple[str, int, int, int],
) -> tuple[str, int, int, int, float, float, float, bool, float, float]:
    """Solve one ``(key, i, j, rung)`` task inside a pool worker."""
    key, i, j, rung = task
    graphs = _WORKER["graphs"][key]
    g1, g2 = graphs[i], graphs[j]
    exact, lb, ub, certified, seconds, best = solve_pair(
        g1,
        g2,
        bounds_backend=_WORKER["bounds"],
        exact_backend=_WORKER["exact"],
        bounds_kind=_WORKER["bounds_kind"],
    )
    return key, int(i), int(j), int(rung), exact, lb, ub, certified, seconds, best


# --------------------------------------------------------------------------- #
# one rung, end to end
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class RungResult:
    """One finished rung: its records and the summary the manifest carries."""

    rung: int
    records: list[PairRecord] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_pairs(self) -> int:
        """Number of pairs attempted at this rung."""
        return len(self.records)

    @property
    def n_certified(self) -> int:
        """Number of pairs whose exact search terminated."""
        return sum(1 for r in self.records if r.certified)

    @property
    def certification_rate(self) -> float:
        """Fraction certified. ``0.0`` on an empty rung, which truncates."""
        return self.n_certified / self.n_pairs if self.n_pairs else 0.0

    @property
    def censoring_rate(self) -> float:
        """Fraction interval-censored under D11."""
        return 1.0 - self.certification_rate if self.n_pairs else 0.0


def run_rung(
    sample: RungSample,
    graphs_by_key: dict[str, list[Any]],
    *,
    cost_model: str = "unit",
    bounds_kind: str = "gedlib",
    lb_method: str = DEFAULT_LB_METHOD,
    lb_options: str = DEFAULT_LB_OPTIONS,
    ub_method: str = DEFAULT_UB_METHOD,
    ub_options: str = DEFAULT_UB_OPTIONS,
    budget_seconds: float = DEFAULT_BUDGET_SECONDS,
    workers: int = 1,
    exported_dir: Path | None = None,
    seed: int = DEFAULT_SEED,
    progress_every: int = 25,
) -> RungResult:
    """Solve every pair of one rung and assemble its result.

    Parameters
    ----------
    sample : RungSample
        The pairs to solve.
    graphs_by_key : dict of str to list
        Graphs per dataset, in exported order. Used directly when
        ``workers == 1``; the pool reloads them per process otherwise.
    cost_model : {'unit', 'graphedx'}, optional
        ``'unit'`` is D6 and is the only production model.
    bounds_kind : {'gedlib', 'networkx'}, optional
        Source of the bracket.
    lb_method, lb_options, ub_method, ub_options : str, optional
        GEDLIB configuration. Ignored under ``bounds_kind='networkx'``.
    budget_seconds : float, optional
        Per-pair wall budget for the exact solver.
    workers : int, optional
        Pool size. ``1`` runs in-process.
    exported_dir : pathlib.Path, optional
        Required when ``workers > 1``: the pool workers reload the cohorts from
        it rather than pickling graphs through the queue.
    seed : int, optional
        Recorded in the metadata; the sample is already drawn.
    progress_every : int, optional
        Log a running certification count every this many pairs.

    Returns
    -------
    RungResult
        Records in the sample's order, plus the metadata dictionary.

    Raises
    ------
    LadderError
        If ``workers > 1`` without ``exported_dir``.
    """
    tasks = [
        (str(sample.dataset_key[t]), int(sample.pair_i[t]), int(sample.pair_j[t]), sample.rung)
        for t in range(sample.n_pairs)
    ]

    records: list[PairRecord] = []
    t_start = time.perf_counter()

    if not tasks:
        rows: list[tuple[Any, ...]] = []
    elif workers <= 1:
        costs = _COST_MODELS[cost_model]
        bounds, exact_backend = _build_backends(
            costs, bounds_kind, lb_method, lb_options, ub_method, ub_options, budget_seconds
        )
        rows = []
        for n_done, (key, i, j, rung) in enumerate(tasks, start=1):
            g1, g2 = graphs_by_key[key][i], graphs_by_key[key][j]
            rows.append(
                (key, i, j, rung)
                + solve_pair(
                    g1,
                    g2,
                    bounds_backend=bounds,
                    exact_backend=exact_backend,
                    bounds_kind=bounds_kind,
                )
            )
            if progress_every and n_done % progress_every == 0:
                done = sum(1 for r in rows if r[7])
                logger.info(
                    "rung %d: %d/%d pairs, %d certified, %.1f s elapsed",
                    sample.rung,
                    n_done,
                    len(tasks),
                    done,
                    time.perf_counter() - t_start,
                )
    else:
        if exported_dir is None:
            raise LadderError("workers > 1 requires exported_dir so the pool can reload cohorts")
        keys = sorted({t[0] for t in tasks})
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(
                str(exported_dir),
                keys,
                cost_model,
                bounds_kind,
                lb_method,
                lb_options,
                ub_method,
                ub_options,
                budget_seconds,
            ),
        ) as pool:
            rows = []
            for n_done, row in enumerate(pool.map(_solve_task, tasks, chunksize=1), start=1):
                rows.append(row)
                if progress_every and n_done % progress_every == 0:
                    done = sum(1 for r in rows if r[7])
                    logger.info(
                        "rung %d: %d/%d pairs, %d certified, %.1f s elapsed",
                        sample.rung,
                        n_done,
                        len(tasks),
                        done,
                        time.perf_counter() - t_start,
                    )

    for key, i, j, rung, exact_v, lb_v, ub_v, cert_v, secs, best_v in rows:
        records.append(
            PairRecord(
                dataset_key=key,
                pair_i=int(i),
                pair_j=int(j),
                n_max=int(rung),
                exact=float(exact_v),
                lb=float(lb_v),
                ub=float(ub_v),
                certified=bool(cert_v),
                seconds=float(secs),
                ub_astar_bestsofar=float(best_v),
            )
        )

    result = RungResult(rung=sample.rung, records=records)
    per_dataset_certified: dict[str, int] = {}
    for rec in records:
        per_dataset_certified.setdefault(rec.dataset_key, 0)
        per_dataset_certified[rec.dataset_key] += int(rec.certified)

    if bounds_kind == "gedlib":
        lb_name, lb_opt, ub_name, ub_opt = lb_method, lb_options, ub_method, ub_options
    else:
        lb_name, lb_opt = "ged_bounds.branch_lower_bound", ""
        ub_name, ub_opt = "ged_bounds.bipartite_upper_bound", ""

    result.meta = {
        "rung": int(sample.rung),
        "n_pairs": result.n_pairs,
        "n_certified": result.n_certified,
        "certification_rate": result.certification_rate,
        "censoring_rate": result.censoring_rate,
        "per_dataset_counts": dict(sorted(sample.realised.items())),
        "per_dataset_allocation": dict(sorted(sample.allocation.items())),
        "per_dataset_population": dict(sorted(sample.masses.items())),
        "per_dataset_certified": dict(sorted(per_dataset_certified.items())),
        "seed": int(seed),
        "budget_seconds": float(budget_seconds),
        "cost_model": cost_model,
        "cost_constant": list(_COST_MODELS[cost_model].as_gedlib_constant())
        if hasattr(_COST_MODELS[cost_model], "as_gedlib_constant")
        else None,
        "lb_method": lb_name,
        "lb_options": lb_opt,
        "ub_method": ub_name,
        "ub_options": ub_opt,
        "bounds_kind": bounds_kind,
        "solver": "networkx.graph_edit_distance (A*), completion by astar_completed",
        "ub_astar_bestsofar_note": (
            "Side column, censored pairs only, inf elsewhere. A valid upper bound (the "
            "edit path was constructed) but MACHINE-DEPENDENT: how far A* gets in the "
            "budget is a function of the node. It is not part of the reproducible "
            "bracket and must not enter a section 6 analysis; lb and ub are."
        ),
        "env_mode": "per-pair",
        "lb_symmetry_probes": 0,
        "workers": int(workers),
        "wall_seconds": time.perf_counter() - t_start,
        "code_commit": _code_commit(),
        "computed_utc": _utc_now(),
        "schema_version": SCHEMA_VERSION,
    }
    return result


# --------------------------------------------------------------------------- #
# output contract
# --------------------------------------------------------------------------- #


def write_rung_npz(path: Path, result: RungResult) -> None:
    """Write one rung to its ``.npz``, atomically.

    Parameters
    ----------
    path : pathlib.Path
        Destination ``ladder/rung_{n}.npz``. Parents are created.
    result : RungResult
        The finished rung.

    Raises
    ------
    LadderError
        If any invariant of the output contract is violated -- a non-finite
        bound, a censored pair carrying a finite ``exact``, a certified pair
        carrying ``inf``, or a certified value outside its bracket. The file is
        not written when a check fails.
    """
    n = result.n_pairs
    dataset_key = np.asarray([r.dataset_key for r in result.records], dtype=np.str_)
    if n == 0:
        dataset_key = np.empty(0, dtype="<U1")
    pair_i = np.asarray([r.pair_i for r in result.records], dtype=np.int32)
    pair_j = np.asarray([r.pair_j for r in result.records], dtype=np.int32)
    n_max = np.asarray([r.n_max for r in result.records], dtype=np.int32)
    exact = np.asarray([r.exact for r in result.records], dtype=np.float64)
    lb = np.asarray([r.lb for r in result.records], dtype=np.float64)
    ub = np.asarray([r.ub for r in result.records], dtype=np.float64)
    certified = np.asarray([r.certified for r in result.records], dtype=bool)
    seconds = np.asarray([r.seconds for r in result.records], dtype=np.float32)
    ub_astar = np.asarray([r.ub_astar_bestsofar for r in result.records], dtype=np.float64)

    if n:
        if not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)):
            raise LadderError("lb/ub must be finite on every pair, censored or not")
        if np.any(lb > ub + CERT_TOL):
            raise LadderError("inverted bracket in the rung arrays")
        if np.any(np.isfinite(exact[~certified])):
            raise LadderError(
                "a censored pair carries a finite exact value; a best-so-far cost is an "
                "upper bound, not a distance"
            )
        if np.any(~np.isfinite(exact[certified])):
            raise LadderError("a certified pair carries a non-finite exact value")
        cert = certified
        if np.any(exact[cert] < lb[cert] - CERT_TOL) or np.any(exact[cert] > ub[cert] + CERT_TOL):
            raise LadderError("a certified optimum falls outside its own bracket")
        if not np.all(n_max == result.rung):
            raise LadderError(f"n_max must equal the rung {result.rung} on every pair")
        # The side column is a censored-pair record only. A finite value on a
        # certified pair would mean an A* cost was kept next to a proven optimum,
        # which is the confusion the separate column exists to prevent.
        if np.any(np.isfinite(ub_astar[certified])):
            raise LadderError(
                "ub_astar_bestsofar is finite on a certified pair; it records only what "
                "a search that did NOT terminate had reached"
            )
        censored_finite = ~certified & np.isfinite(ub_astar)
        if np.any(ub_astar[censored_finite] < lb[censored_finite] - CERT_TOL):
            raise LadderError(
                "an A* best-so-far cost falls below the lower bound; a constructed edit "
                "path cannot be cheaper than a valid lower bound"
            )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The temp name must itself end in `.npz`: `savez_compressed` appends the
    # extension when the name lacks it, and would then write somewhere other than
    # where `os.replace` looks.
    tmp = path.with_name(f"{path.name}.part.{os.getpid()}.npz")
    np.savez_compressed(
        tmp,
        dataset_key=dataset_key,
        pair_i=pair_i,
        pair_j=pair_j,
        n_max=n_max,
        exact=exact,
        lb=lb,
        ub=ub,
        certified=certified,
        seconds=seconds,
        ub_astar_bestsofar=ub_astar,
        metadata=np.array(json.dumps(result.meta, sort_keys=True)),
    )
    os.replace(tmp, path)
    logger.info(
        "wrote %s (%d pairs, %d certified, %.1f %% certification)",
        path,
        n,
        result.n_certified,
        100.0 * result.certification_rate,
    )


def load_rung_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read a rung file back.

    Parameters
    ----------
    path : pathlib.Path
        A file written by :func:`write_rung_npz`.

    Returns
    -------
    tuple
        ``(arrays, metadata)``.
    """
    with np.load(Path(path), allow_pickle=False) as handle:
        arrays = {k: handle[k] for k in handle.files if k != "metadata"}
        raw = handle["metadata"]
    meta = json.loads(str(raw.item()) if raw.ndim == 0 else str(raw))
    return arrays, meta


def write_manifest(
    out_dir: Path,
    rung_metas: list[dict[str, Any]],
    *,
    ceiling: int,
    truncated_at: int | None,
    threshold: float,
    seed: int,
) -> Path:
    """Write ``ladder/manifest.json`` across the rungs that ran.

    Parameters
    ----------
    out_dir : pathlib.Path
        The ``ladder/`` directory.
    rung_metas : list of dict
        One metadata dictionary per rung, in ascending rung order.
    ceiling : int
        The highest rung whose certification rate reached ``threshold``. Falls
        back to 12 -- T-03's census ceiling -- when no ladder rung reaches it.
    truncated_at : int or None
        The first rung below ``threshold``, or ``None`` if none was.
    threshold : float
        The truncation threshold, frozen at 0.25.
    seed : int
        Master seed.

    Returns
    -------
    pathlib.Path
        The manifest path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "seed": int(seed),
        "truncate_below": float(threshold),
        "exact_ged_ceiling": int(ceiling),
        "truncated_at_rung": None if truncated_at is None else int(truncated_at),
        "rungs": rung_metas,
        "code_commit": _code_commit(),
        "computed_utc": _utc_now(),
        "note": (
            "exact_ged_ceiling is the highest rung MEASURED to certify at or above "
            "truncate_below; it is a measurement, not an assertion. A value of 12 means "
            "no rung above T-03's census reached the threshold."
        ),
    }
    path = out_dir / "manifest.json"
    tmp = path.with_suffix(f".json.part.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    logger.info("wrote %s (%d rungs, ceiling n = %d)", path, len(rung_metas), ceiling)
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_rungs(text: str) -> tuple[int, ...]:
    """Parse a comma-separated rung list, ascending and deduplicated."""
    values = sorted({int(part) for part in text.replace(" ", "").split(",") if part})
    if not values:
        raise LadderError(f"no rungs parsed from {text!r}")
    if values[0] <= 12:
        raise LadderError(
            f"rung {values[0]} is at or below T-03's exact-GED census ceiling of 12; "
            "the ladder starts one node above it"
        )
    return tuple(values)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    p = argparse.ArgumentParser(
        prog="ged_ladder",
        description="Exact GED on a size-stratified sample at each rung above n = 12.",
    )
    p.add_argument("--exported-dir", type=Path, required=True, help="Suite-2 exported_suite2/")
    p.add_argument("--out-dir", type=Path, required=True, help="destination ladder/ directory")
    p.add_argument("--rungs", type=str, default=",".join(str(n) for n in DEFAULT_RUNGS))
    p.add_argument("--pairs-per-rung", type=int, default=DEFAULT_PAIRS_PER_RUNG)
    p.add_argument("--min-per-dataset", type=int, default=DEFAULT_MIN_PER_DATASET)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--budget-seconds", type=float, default=DEFAULT_BUDGET_SECONDS)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--cost-model", choices=sorted(_COST_MODELS), default="unit")
    p.add_argument("--bounds", choices=("gedlib", "networkx"), default="gedlib")
    p.add_argument("--lb-method", default=DEFAULT_LB_METHOD)
    p.add_argument("--lb-options", default=DEFAULT_LB_OPTIONS)
    p.add_argument("--ub-method", default=DEFAULT_UB_METHOD)
    p.add_argument("--ub-options", default=DEFAULT_UB_OPTIONS)
    p.add_argument("--truncate-below", type=float, default=DEFAULT_TRUNCATE_BELOW)
    p.add_argument(
        "--no-truncate",
        action="store_true",
        help="run every requested rung even after one falls below the threshold",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="skip a rung whose .npz already exists and fold its metadata into the manifest",
    )
    p.add_argument(
        "--sample-only",
        action="store_true",
        help="draw and report the samples without solving anything",
    )
    p.add_argument(
        "--mirror-dir",
        type=Path,
        default=None,
        help="copy each finished rung here as soon as it lands (cluster checkpointing)",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def _mirror(src: Path, mirror_dir: Path) -> None:
    """Copy one finished file to a shared directory without a torn read."""
    mirror_dir.mkdir(parents=True, exist_ok=True)
    tmp = mirror_dir / f"{src.name}.part.{os.getpid()}"
    shutil.copy2(src, tmp)
    os.replace(tmp, mirror_dir / src.name)
    logger.info("mirrored %s -> %s", src.name, mirror_dir)


def main(argv: list[str] | None = None) -> int:
    """Run the ladder.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments.

    Returns
    -------
    int
        Process exit status.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    rungs = _parse_rungs(args.rungs)
    exported_dir = Path(args.exported_dir)
    out_dir = Path(args.out_dir)

    # Pin the commit BEFORE any pair is solved. A ladder runs for hours and the
    # working tree can move under it; resolving this at metadata-build time made
    # my own rung-13 pilot name a commit three ahead of the code that ran it.
    logger.info("code commit %s", _code_commit())

    present = [k for k in SUITE2_KEYS if (exported_dir / f"{k}.npz").is_file()]
    if not present:
        raise LadderError(f"no Suite-2 cohort found under {exported_dir}")
    if len(present) != len(SUITE2_KEYS):
        logger.warning(
            "only %d of the %d Suite-2 cohorts are present: %s",
            len(present),
            len(SUITE2_KEYS),
            ", ".join(present),
        )

    datasets = {k: load_exported(exported_dir / f"{k}.npz") for k in present}
    n_nodes = {k: np.asarray(d.n_nodes) for k, d in datasets.items()}
    graphs = {k: d.graphs for k, d in datasets.items()}

    rung_metas: list[dict[str, Any]] = []
    ceiling = 12
    truncated_at: int | None = None

    for rung in rungs:
        target = out_dir / f"rung_{rung}.npz"
        if args.resume and target.is_file():
            _arrays, meta = load_rung_npz(target)
            logger.info(
                "rung %d already present, resuming past it (%.1f %% certified)",
                rung,
                100.0 * float(meta.get("certification_rate", 0.0)),
            )
            rung_metas.append(meta)
            if float(meta.get("certification_rate", 0.0)) >= args.truncate_below:
                ceiling = rung
            elif truncated_at is None:
                truncated_at = rung
                if not args.no_truncate:
                    break
            continue

        sample = sample_rung(
            n_nodes,
            rung,
            total=args.pairs_per_rung,
            minimum=args.min_per_dataset,
            seed=args.seed,
        )
        logger.info(
            "rung %d: %d pairs from %d dataset(s) %s (population %d)",
            rung,
            sample.n_pairs,
            len(sample.realised),
            dict(sorted(sample.realised.items())),
            sum(sample.masses.values()),
        )
        if sample.is_empty:
            logger.warning(
                "rung %d has no eligible pair in any present cohort; it is reported empty, "
                "not skipped",
                rung,
            )

        if args.sample_only:
            result = RungResult(rung=rung, records=[])
            result.meta = {
                "rung": rung,
                "n_pairs": sample.n_pairs,
                "n_certified": 0,
                "certification_rate": 0.0,
                "censoring_rate": 0.0,
                "per_dataset_counts": dict(sorted(sample.realised.items())),
                "per_dataset_allocation": dict(sorted(sample.allocation.items())),
                "per_dataset_population": dict(sorted(sample.masses.items())),
                "seed": int(args.seed),
                "sample_only": True,
                "schema_version": SCHEMA_VERSION,
                "code_commit": _code_commit(),
                "computed_utc": _utc_now(),
            }
            sample_path = out_dir / f"sample_rung_{rung}.npz"
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                sample_path,
                dataset_key=sample.dataset_key,
                pair_i=sample.pair_i,
                pair_j=sample.pair_j,
                n_max=np.full(sample.n_pairs, rung, dtype=np.int32),
                metadata=np.array(json.dumps(result.meta, sort_keys=True)),
            )
            rung_metas.append(result.meta)
            continue

        result = run_rung(
            sample,
            graphs,
            cost_model=args.cost_model,
            bounds_kind=args.bounds,
            lb_method=args.lb_method,
            lb_options=args.lb_options,
            ub_method=args.ub_method,
            ub_options=args.ub_options,
            budget_seconds=args.budget_seconds,
            workers=args.workers,
            exported_dir=exported_dir,
            seed=args.seed,
        )
        write_rung_npz(target, result)
        if args.mirror_dir is not None:
            _mirror(target, Path(args.mirror_dir))
        rung_metas.append(result.meta)

        logger.info(
            "rung %d: %d/%d certified (%.1f %%), censoring %.1f %%, %.1f s wall",
            rung,
            result.n_certified,
            result.n_pairs,
            100.0 * result.certification_rate,
            100.0 * result.censoring_rate,
            result.meta["wall_seconds"],
        )
        if result.certification_rate >= args.truncate_below:
            ceiling = rung
        elif truncated_at is None:
            truncated_at = rung
            logger.warning(
                "rung %d certifies at %.1f %%, below the %.0f %% threshold; the measured "
                "exact-GED ceiling is n = %d",
                rung,
                100.0 * result.certification_rate,
                100.0 * args.truncate_below,
                ceiling,
            )
            if not args.no_truncate:
                break

    manifest = write_manifest(
        out_dir,
        rung_metas,
        ceiling=ceiling,
        truncated_at=truncated_at,
        threshold=args.truncate_below,
        seed=args.seed,
    )
    if args.mirror_dir is not None:
        _mirror(manifest, Path(args.mirror_dir))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
