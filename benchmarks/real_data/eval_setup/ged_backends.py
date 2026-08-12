"""GED backends -- Contract B of wave ``2026-08-12-exact-ged``.

Interchangeable ways to obtain a graph edit distance, behind one frozen result
type. Every backend returns a :class:`PairResult` whose ``exact`` field is
``None`` unless the value carries an optimality certificate.

Amendment 2 of ``T-03-design.md``: there is no GEDLIB certificate
----------------------------------------------------------------
``ANCHOR_AWARE_GED`` was measured on Picasso on 2026-08-12 and is **neither
exact nor deterministic**: 14 of 15 real AIDS pairs gave different answers
across fresh environments, 4 of 18 small pairs were wrong against exhaustive
brute force (always over, never under), and it reports ``LB == UB`` on those
wrong values -- a *false optimality certificate*, which is worse than a wrong
number because it defeats the check meant to catch one. No option restores it.
It is a randomised upper-bound heuristic wearing an exact label.

It is therefore **retired**, and :func:`_reject_retired` raises on any attempt
to name it. The guard exists because the failure mode is silent: a copied
config would otherwise reintroduce it and the pipeline would accept its
certificates without complaint.

The consequences for this module:

- :class:`GedlibBackend` is **bounds-only** -- ``BRANCH_FAST`` for ``lb``,
  ``IPFP`` for ``ub`` -- and always reports ``exact=None, certified=False``.
- Exact values come from :class:`NetworkxBackend` alone.
- :class:`ExactPlusBoundsBackend` composes the two, which is what production
  needs per pair.

How certification is decided
----------------------------
``networkx.graph_edit_distance(g1, g2, timeout=t)`` returns its *best-found-so-
far* cost when the budget expires. It does not raise, and it returns ``None``
only when no complete edit path was found at all. So the value alone cannot
distinguish an optimum from an upper bound, and no property of the value can:
the decision must come from **whether the search completed**. Elapsed solver
time is measured against the budget with a margin, and
``elapsed >= timeout - margin`` is treated as a possible cut-off, hence
uncertified -- ``exact=None``, and the pair is interval-censored on ``[lb, ub]``
under decision D11. The margin makes the test conservative in the safe
direction: it can censor a pair that in fact finished, never certify one that
did not.

An achievable cost that meets a valid lower bound is also a proof of
optimality, and wave 1 used it as a second certificate. It is **deliberately
not used here**: it would make the certification rate depend on the
correctness of our own bound implementation rather than on solver completion,
which is the class of derived certificate this amendment exists to distrust.
Gate 1 checks that bound against the exact values instead.

GEDLIB has two failure modes of its own, both silent:

1. ``get_lower_bound()`` on an upper-bound method returns ``0.00`` rather than
   raising, so a whole distance matrix fills with zeros;
2. ``gklearn.gedlib.libraries_import`` must ``dlopen()`` libdoublefann, libsvm
   and libnomad *before* ``gedlibpy_gxl`` loads.

Both are handled here: every read is range-checked, the method/accessor pairing
is validated at construction time against the capability table measured on
Picasso, and the imports go through :func:`importlib.import_module` so that a
formatter cannot reorder them.

References
----------
Blumenthal, D. B., & Gamper, J. (2018). On the exact computation of the graph
edit distance. *IEEE TKDE* 30(3), 503-516. doi:10.1109/TKDE.2017.2772243
"""

from __future__ import annotations

import importlib
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import networkx as nx

try:  # package import: python -m benchmarks.real_data.eval_setup.ged_backends
    from .ged_bounds import (
        UNIT_COSTS,
        EditCosts,
        bipartite_upper_bound,
        branch_lower_bound,
        exact_ged,
    )
except ImportError:  # pragma: no cover - bare-module import from inside eval_setup/
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ged_bounds import (  # type: ignore[no-redef]  # noqa: E402
        UNIT_COSTS,
        EditCosts,
        bipartite_upper_bound,
        branch_lower_bound,
        exact_ged,
    )

logger = logging.getLogger(__name__)

#: Tolerance at which two distances are treated as equal.
CERT_TOL = 1e-9

#: Absolute and relative safety margins subtracted from an exact solver's
#: budget before its elapsed time is compared against it. A search that
#: returned within ``timeout - margin`` is taken to have completed; anything
#: closer to the deadline is treated as a possible cut-off. Conservative by
#: construction: the error it can make is to censor a pair that finished.
TIMEOUT_MARGIN_ABS = 1e-3
TIMEOUT_MARGIN_REL = 0.01

_INF = float("inf")

#: Constant attribute value attached to every node and edge. GEDLIB's GXL
#: bindings require *string* attribute values; substitution is free under both
#: permitted cost models, so a constant label cannot influence any distance.
_CONSTANT_LABEL = "1"

#: The 21 methods verified present on Picasso 2026-08-11.
VERIFIED_METHODS = frozenset(
    {
        "BRANCH",
        "BRANCH_FAST",
        "BRANCH_TIGHT",
        "BRANCH_UNIFORM",
        "BRANCH_COMPACT",
        "PARTITION",
        "HYBRID",
        "RING",
        "ANCHOR_AWARE_GED",
        "WALKS",
        "IPFP",
        "BIPARTITE",
        "SUBGRAPH",
        "NODE",
        "RING_ML",
        "BIPARTITE_ML",
        "REFINE",
        "BP_BEAM",
        "SIMULATED_ANNEALING",
        "HED",
        "STAR",
    }
)

#: Methods whose ``get_lower_bound()`` was measured to return a real bound.
LOWER_BOUND_METHODS = frozenset({"BRANCH", "BRANCH_FAST", "BRANCH_TIGHT", "STAR"})

#: Methods whose ``get_upper_bound()`` was measured to return a real bound.
UPPER_BOUND_METHODS = frozenset({"BIPARTITE", "IPFP", "REFINE", "BP_BEAM"})

#: No GEDLIB method returns a certified optimum. ``ANCHOR_AWARE_GED`` claimed
#: to and does not (amendment 2). Kept as an empty set so that the shape of the
#: capability table still says what it says: exact is not a GEDLIB role.
EXACT_METHODS: frozenset[str] = frozenset()

#: Retired by amendment 2 after measurement, distinct from never-verified.
RETIRED_METHODS = frozenset({"ANCHOR_AWARE_GED"})

#: Never used. Returns ``LB = 0`` / ``UB = inf`` under default options and the
#: cause is undiagnosed (gedlib.md, "unresolved, do not use yet").
FORBIDDEN_METHODS = frozenset({"HED"})

#: Message shown when a retired or forbidden method is named. Written out in
#: full so that a stack trace alone explains the refusal.
_RETIRED_REASON = {
    "ANCHOR_AWARE_GED": (
        "ANCHOR_AWARE_GED is retired by T-03-design.md amendment 2 (PI-authorised, "
        "2026-08-12). Measured on Picasso: non-deterministic on 14/15 real AIDS pairs, "
        "wrong on 4/18 pairs against exhaustive brute force (always over), and it "
        "reports LB == UB on the wrong values -- a false optimality certificate. "
        "No option (--threads 1, --map-root-to-root, --search-method DFS) restores it. "
        "Exact GED now comes from networkx A* only; GEDLIB is bounds-only. "
        "No core-hour is to be spent on this method."
    ),
    "HED": (
        "HED is never used: it returns lower bound 0 and upper bound inf under "
        "default options and the cause is undiagnosed."
    ),
}


class GedBackendError(Exception):
    """Raised when a backend is misconfigured or a solver returns a bad value."""


def _reject_retired(method: str | None) -> None:
    """Refuse a method that amendment 2 retired, or that was never usable.

    The single choke point every backend routes through. A method name reaching
    GEDLIB without passing here is a bug.

    Parameters
    ----------
    method : str or None
        Requested GEDLIB method. ``None`` passes.

    Raises
    ------
    GedBackendError
        If the method is retired or forbidden, quoting the reason in full.
    """
    if method is None:
        return
    name = str(method).strip().upper()
    if name in RETIRED_METHODS or name in FORBIDDEN_METHODS:
        raise GedBackendError(_RETIRED_REASON[name])


@dataclass(frozen=True, slots=True)
class PairResult:
    """One graph pair, one backend.

    Attributes
    ----------
    lb : float
        Lower bound on the distance. Finite, and positive unless a zero
        distance is genuinely attainable for the pair.
    ub : float
        Upper bound, symmetrised over both argument orders.
    exact : float or None
        The distance, set **if and only if** ``certified``. Never a
        best-so-far value.
    certified : bool
        ``True`` iff a solver *proved* the value optimal by running to
        completion. Not inferred from the bracket -- see the note below.
    seconds : float
        Wall time spent on this pair by this backend.
    timed_out : bool
        The solver's budget expired before it proved optimality.
    method : str
        Solver identification, e.g. ``'networkx_astar+BRANCH_FAST+IPFP'``.

    Notes
    -----
    Wave 1 required ``certified == (ub - lb <= CERT_TOL)``, because the only
    exact solver then in use closed its own bracket. Amendment 2 splits the two
    apart: the exact value now comes from networkx A* while ``lb`` and ``ub``
    come from GEDLIB, so a certified pair usually has ``lb < exact < ub``.

    Collapsing the bracket onto a certified exact value would be the obvious
    alternative and is **wrong for this pipeline**: gate 3b measures bound
    quality -- rho(exact, LB), rho(exact, UB), relative bias, and the rate at
    which ``LB == UB`` -- and every one of those quantities is destroyed by
    overwriting the bounds with the exact value. The bounds are reported data,
    not scratch space.

    What survives, and is what invariant 4 of Contract B actually protects, is
    the biconditional ``certified <=> exact is not None`` plus the containment
    ``lb <= exact <= ub``. Those are enforced here, so no backend can promote a
    best-so-far cost, and a bound that contradicts a proven optimum surfaces as
    an exception rather than as a plausible number.

    Raises
    ------
    GedBackendError
        If the fields are mutually inconsistent.
    """

    lb: float
    ub: float
    exact: float | None
    certified: bool
    seconds: float
    timed_out: bool
    method: str

    def __post_init__(self) -> None:
        """Enforce the internal consistency of the result."""
        if not math.isfinite(self.lb) or self.lb < 0.0:
            raise GedBackendError(f"lb must be finite and non-negative, got {self.lb!r}")
        if not math.isfinite(self.ub) or self.ub < 0.0:
            raise GedBackendError(f"ub must be finite and non-negative, got {self.ub!r}")
        if self.lb > self.ub + CERT_TOL:
            raise GedBackendError(f"lb {self.lb} exceeds ub {self.ub}")
        if self.certified and self.exact is None:
            raise GedBackendError("certified result must carry an exact value")
        if not self.certified and self.exact is not None:
            raise GedBackendError(
                f"uncertified result must not carry exact={self.exact!r}; "
                "a best-so-far cost is an upper bound, not a distance"
            )
        if self.exact is not None and not math.isfinite(self.exact):
            raise GedBackendError(f"exact must be finite when set, got {self.exact!r}")
        if self.exact is not None and not (self.lb - CERT_TOL <= self.exact <= self.ub + CERT_TOL):
            raise GedBackendError(
                f"exact {self.exact} outside bracket [{self.lb}, {self.ub}]; a proven "
                "optimum outside its own bounds means one of the two is wrong"
            )
        if self.seconds < 0.0:
            raise GedBackendError(f"seconds must be non-negative, got {self.seconds!r}")

    @property
    def bracket_closed(self) -> bool:
        """Whether ``ub - lb`` is within :data:`CERT_TOL`.

        A closed bracket from two independently valid bounds does prove
        optimality, and gate 3b reports the rate at which GEDLIB achieves it.
        It is reported, never used to set :attr:`certified`: after amendment 2
        no GEDLIB self-report is treated as a certificate.
        """
        return (self.ub - self.lb) <= CERT_TOL


@runtime_checkable
class GedBackend(Protocol):
    """A source of graph edit distances for one pair at a time."""

    def pair(self, g1: nx.Graph, g2: nx.Graph) -> PairResult:
        """Compute bounds and, where certifiable, the exact distance."""
        ...

    @property
    def name(self) -> str:
        """Short backend identifier, used in shard metadata."""
        ...


@dataclass(slots=True)
class BackendStats:
    """Running counters for one backend instance.

    The upper-bound asymmetry rate is the reason this exists. Our own bipartite
    implementation measured differences of 12 vs 14 and 5 vs 7, with one
    orientation tighter on roughly a third of pairs; the same figure is needed
    for GEDLIB before its upper bounds can be called a distance matrix.
    """

    n_pairs: int = 0
    n_certified: int = 0
    n_timed_out: int = 0
    total_seconds: float = 0.0
    n_ub_orientations_compared: int = 0
    n_ub_asymmetric: int = 0
    max_ub_gap: float = 0.0
    n_lb_orientations_compared: int = 0
    n_lb_asymmetric: int = 0
    max_lb_gap: float = 0.0
    n_zero_values_accepted: int = 0
    # A zero LOWER bound on a non-isomorphic pair: valid but uninformative. Its rate is
    # a reported bound-quality statistic, not an error -- see GedlibBackend.bounds.
    n_trivial_lower_bounds: int = 0
    n_lb_above_exact: int = 0
    n_ub_below_exact: int = 0

    def record_ub_orientations(self, a: float, b: float) -> None:
        """Record one pair of oriented upper bounds."""
        self.n_ub_orientations_compared += 1
        gap = abs(a - b)
        if gap > CERT_TOL:
            self.n_ub_asymmetric += 1
        self.max_ub_gap = max(self.max_ub_gap, gap)

    def record_lb_orientations(self, a: float, b: float) -> None:
        """Record one pair of oriented lower bounds."""
        self.n_lb_orientations_compared += 1
        gap = abs(a - b)
        if gap > CERT_TOL:
            self.n_lb_asymmetric += 1
        self.max_lb_gap = max(self.max_lb_gap, gap)

    def record_result(self, result: PairResult) -> None:
        """Fold one finished pair into the counters."""
        self.n_pairs += 1
        self.n_certified += int(result.certified)
        self.n_timed_out += int(result.timed_out)
        self.total_seconds += result.seconds

    def as_dict(self) -> dict[str, float | int]:
        """Return the counters plus the derived rates."""
        ub_n = self.n_ub_orientations_compared
        lb_n = self.n_lb_orientations_compared
        return {
            "n_pairs": self.n_pairs,
            "n_certified": self.n_certified,
            "certification_rate": (self.n_certified / self.n_pairs) if self.n_pairs else 0.0,
            "n_timed_out": self.n_timed_out,
            "total_seconds": self.total_seconds,
            "mean_seconds": (self.total_seconds / self.n_pairs) if self.n_pairs else 0.0,
            "n_ub_orientations_compared": ub_n,
            "n_ub_asymmetric": self.n_ub_asymmetric,
            "ub_asymmetry_rate": (self.n_ub_asymmetric / ub_n) if ub_n else 0.0,
            "max_ub_gap": self.max_ub_gap,
            "n_lb_orientations_compared": lb_n,
            "n_lb_asymmetric": self.n_lb_asymmetric,
            "lb_asymmetry_rate": (self.n_lb_asymmetric / lb_n) if lb_n else 0.0,
            "max_lb_gap": self.max_lb_gap,
            "n_zero_values_accepted": self.n_zero_values_accepted,
            "n_lb_above_exact": self.n_lb_above_exact,
            "n_ub_below_exact": self.n_ub_below_exact,
        }


def _drop_isolated(g: nx.Graph) -> nx.Graph:
    """Return a copy of ``g`` without its degree-zero nodes."""
    h = g.copy()
    h.remove_nodes_from([v for v, d in g.degree() if d == 0])
    return h


def zero_distance_is_attainable(g1: nx.Graph, g2: nx.Graph, costs: EditCosts) -> bool:
    """Decide whether a distance of exactly zero is legal for this pair.

    A read of ``0.00`` from GEDLIB is the signature of the wrong accessor, so
    it must be rejected -- except where a zero-cost edit path genuinely exists.

    Under the production model that means the graphs are isomorphic. Under
    GraphEdX's model node operations are free, so adding or removing isolated
    nodes costs nothing and the graphs need only be isomorphic after their
    degree-zero nodes are dropped.

    Parameters
    ----------
    g1, g2 : networkx.Graph
        The graphs compared.
    costs : EditCosts
        The active cost model.

    Returns
    -------
    bool
        ``True`` if some edit path of cost zero exists.
    """
    if costs.node_sub > 0.0 or costs.edge_sub > 0.0:
        return g1.number_of_nodes() == 0 and g2.number_of_nodes() == 0
    a, b = g1, g2
    if costs.node_ins == 0.0 and costs.node_del == 0.0:
        a, b = _drop_isolated(g1), _drop_isolated(g2)
    if a.number_of_nodes() != b.number_of_nodes():
        return False
    if a.number_of_edges() != b.number_of_edges():
        return False
    return nx.is_isomorphic(a, b)


class StubBackend:
    """Deterministic bracket with no solver, for plumbing tests.

    The bounds are the two trivial ones: the size difference is a valid lower
    bound under both permitted cost models, and deleting every element of one
    graph then inserting every element of the other is a valid, achievable
    edit path, hence an upper bound.

    Parameters
    ----------
    costs : EditCosts, optional
        Edit operation costs.
    timeout_s : float, optional
        Accepted and recorded for signature compatibility; never used.
    seconds : float, optional
        Synthetic wall time reported for every pair, so that output is
        byte-reproducible. Defaults to ``0.0``.
    """

    __slots__ = ("_costs", "_seconds", "_timeout_s", "stats")

    def __init__(
        self,
        costs: EditCosts = UNIT_COSTS,
        *,
        timeout_s: float = 300.0,
        seconds: float = 0.0,
    ) -> None:
        self._costs = costs
        self._timeout_s = timeout_s
        self._seconds = float(seconds)
        self.stats = BackendStats()

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "stub"

    @property
    def costs(self) -> EditCosts:
        """The active cost model."""
        return self._costs

    def pair(self, g1: nx.Graph, g2: nx.Graph) -> PairResult:
        """Return the trivial bracket for one pair.

        Parameters
        ----------
        g1, g2 : networkx.Graph
            The graphs to compare.

        Returns
        -------
        PairResult
            Bounds only, certified only when the two bounds coincide.
        """
        c = self._costs
        n1, n2 = g1.number_of_nodes(), g2.number_of_nodes()
        m1, m2 = g1.number_of_edges(), g2.number_of_edges()
        node_gap = (n1 - n2) * c.node_del if n1 >= n2 else (n2 - n1) * c.node_ins
        edge_gap = (m1 - m2) * c.edge_del if m1 >= m2 else (m2 - m1) * c.edge_ins
        lb = float(node_gap + edge_gap)
        ub = float(n1 * c.node_del + n2 * c.node_ins + m1 * c.edge_del + m2 * c.edge_ins)
        certified = (ub - lb) <= CERT_TOL
        result = PairResult(
            lb=lb,
            ub=ub,
            exact=ub if certified else None,
            certified=certified,
            seconds=self._seconds,
            timed_out=False,
            method="stub",
        )
        self.stats.record_result(result)
        return result


def astar_completed(solver_seconds: float, budget: float | None) -> bool:
    """Decide whether an A* run finished rather than hit its deadline.

    ``networkx.graph_edit_distance`` prunes every branch once the deadline has
    passed and then unwinds, so a cut-off run returns *after* the budget while
    a completed run returns before it. The comparison is made against
    ``budget - margin`` so that clock granularity and the cost of the final
    unwind cannot turn a cut-off into an apparent completion.

    Parameters
    ----------
    solver_seconds : float
        Wall time of the solver call alone, excluding bound computation.
    budget : float or None
        The ``timeout`` passed to the solver. ``None`` or non-finite means the
        search was unlimited and therefore always completed.

    Returns
    -------
    bool
        ``True`` only when the search is known to have exhausted its space.
    """
    if budget is None or not math.isfinite(budget):
        return True
    margin = max(TIMEOUT_MARGIN_ABS, TIMEOUT_MARGIN_REL * float(budget))
    return solver_seconds < float(budget) - margin


class NetworkxBackend:
    """Bounds from :mod:`ged_bounds`, exact distance from NetworkX A*.

    The exact stage is the one that needs care. ``nx.graph_edit_distance``
    returns its best-found-so-far cost when ``timeout`` expires, so the value
    alone does not distinguish an optimum from an upper bound, and no test on
    the value can. Certification is decided by :func:`astar_completed`, that
    is, by whether the search finished. Failing that the cost is folded into
    ``ub``, ``exact`` stays ``None``, and the pair is interval-censored on
    ``[lb, ub]`` under D11.

    Parameters
    ----------
    costs : EditCosts, optional
        Edit operation costs.
    timeout_s : float, optional
        Per-pair budget for the exact solver. ``None`` or ``inf`` runs it to
        completion.
    """

    __slots__ = ("_costs", "_timeout_s", "stats")

    def __init__(self, costs: EditCosts = UNIT_COSTS, *, timeout_s: float | None = 300.0) -> None:
        self._costs = costs
        self._timeout_s = timeout_s
        self.stats = BackendStats()

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "networkx"

    @property
    def costs(self) -> EditCosts:
        """The active cost model."""
        return self._costs

    def heuristic_bracket(self, g1: nx.Graph, g2: nx.Graph) -> tuple[float, float]:
        """Return the untightened ``(lb, ub)`` bracket for one pair.

        Both orientations of the upper bound are evaluated and the asymmetry
        recorded, then the smaller is returned: each orientation is an
        achievable edit path, so the minimum is still an upper bound and,
        unlike either orientation alone, it is symmetric.

        Parameters
        ----------
        g1, g2 : networkx.Graph
            The graphs to compare.

        Returns
        -------
        tuple of float
            ``(lower_bound, upper_bound)``.
        """
        lb = branch_lower_bound(g1, g2, self._costs)
        ub_fwd = bipartite_upper_bound(g1, g2, self._costs, symmetrise=False)
        ub_rev = bipartite_upper_bound(g2, g1, self._costs, symmetrise=False)
        self.stats.record_ub_orientations(ub_fwd, ub_rev)
        return lb, min(ub_fwd, ub_rev)

    def solve_exact(self, g1: nx.Graph, g2: nx.Graph) -> tuple[float | None, float, float, bool]:
        """Run A* once and report what it proved, not merely what it returned.

        Parameters
        ----------
        g1, g2 : networkx.Graph
            The graphs to compare.

        Returns
        -------
        tuple
            ``(exact_or_None, best_cost, solver_seconds, timed_out)``.
            ``best_cost`` is the returned cost whether or not it was proven --
            it is a valid upper bound either way, being the cost of an edit
            path the solver actually constructed -- and is ``inf`` when no
            complete path was found. ``exact_or_None`` is set only when the
            search completed.
        """
        budget = self._timeout_s
        unlimited = budget is None or not math.isfinite(budget)
        t0 = time.perf_counter()
        value = exact_ged(g1, g2, self._costs, timeout=None if unlimited else budget)
        solver_seconds = time.perf_counter() - t0

        completed = astar_completed(solver_seconds, None if unlimited else budget)
        found = math.isfinite(value)
        proven = completed and found
        return (float(value) if proven else None), float(value), solver_seconds, not proven

    def pair(self, g1: nx.Graph, g2: nx.Graph) -> PairResult:
        """Compute the bracket and, where the search completed, the distance.

        Parameters
        ----------
        g1, g2 : networkx.Graph
            The graphs to compare.

        Returns
        -------
        PairResult
            ``exact`` is set only when A* ran to completion. A best-so-far cost
            is folded into ``ub`` and never promoted.
        """
        t0 = time.perf_counter()
        lb, ub = self.heuristic_bracket(g1, g2)
        exact, best_cost, _solver_seconds, timed_out = self.solve_exact(g1, g2)

        if math.isfinite(best_cost):
            ub = min(ub, best_cost)
        if exact is not None:
            # Both bounds and the exact value come from our own code here, so a
            # violation is a defect in `ged_bounds`, not a disagreement between
            # implementations. Count it and let gate 1 report the distribution;
            # raising would replace that distribution with one stack trace.
            if exact < lb - CERT_TOL:
                self.stats.n_lb_above_exact += 1
                logger.warning(
                    "branch_lower_bound %.6g exceeds the A* optimum %.6g; the bound is invalid",
                    lb,
                    exact,
                )
            if exact > ub + CERT_TOL:
                self.stats.n_ub_below_exact += 1
                logger.warning(
                    "bipartite_upper_bound %.6g is below the A* optimum %.6g; the bound is invalid",
                    ub,
                    exact,
                )
            # A completed search collapses the bracket: the value is the
            # distance, so both bounds are known exactly.
            lb = ub = float(exact)

        result = PairResult(
            lb=float(lb),
            ub=float(ub),
            exact=exact,
            certified=exact is not None,
            seconds=time.perf_counter() - t0,
            timed_out=bool(timed_out),
            method="networkx_astar",
        )
        self.stats.record_result(result)
        return result


class GedlibBackend:
    """GEDLIB **bounds**, with every silent trap guarded.

    Bounds only, since amendment 2. ``BRANCH_FAST`` supplies ``lb`` and
    ``IPFP`` supplies ``ub`` over both orientations; ``exact`` is always
    ``None`` and ``certified`` always ``False``. The one method that claimed to
    certify, ``ANCHOR_AWARE_GED``, was measured non-deterministic and wrong,
    and :func:`_reject_retired` refuses to construct a backend that names it.

    The environment object is built once per instance -- meaning once per
    worker process -- and reset between pairs with ``restart_env()`` rather
    than reconstructed, so that a run of hundreds of thousands of pairs neither
    pays construction cost per pair nor accumulates graphs without bound.

    Parameters
    ----------
    costs : EditCosts, optional
        Edit operation costs, passed to GEDLIB as ``CONSTANT``.
    timeout_s : float, optional
        Advisory per-pair budget. GEDLIB's solve runs inside a C++ call that
        Python cannot interrupt, so this is detected after the fact and
        recorded; it does not abort a running solve.
    exact_method : None, optional
        Retained so that a configuration copied from wave 1 raises a message
        naming amendment 2 rather than ``TypeError: unexpected keyword``. The
        only accepted value is ``None``.
    lb_method : str, optional
        Lower-bound method, read through ``get_lower_bound()``.
    ub_method : str, optional
        Upper-bound method, read through ``get_upper_bound()`` in both
        orientations.
    threads : int, optional
        Value of GEDLIB's ``--threads`` option. Defaults to 1, matching the
        one-process-per-core layout the cluster jobs use.
    lb_symmetry_probes : int, optional
        Number of leading pairs on which the lower bound is also evaluated in
        the reverse orientation, to measure rather than assume its symmetry.

    Raises
    ------
    GedBackendError
        If any method is retired, forbidden, unverified, or paired with an
        accessor it was not measured to support.
    """

    __slots__ = (
        "_costs",
        "_env",
        "_gedlib_module",
        "_heuristic_options",
        "_init_option",
        "_lb_method",
        "_lb_symmetry_probes",
        "_reset_mode",
        "_timeout_s",
        "_ub_method",
        "stats",
    )

    def __init__(
        self,
        costs: EditCosts = UNIT_COSTS,
        *,
        timeout_s: float = 300.0,
        exact_method: str | None = None,
        lb_method: str = "BRANCH_FAST",
        ub_method: str = "IPFP",
        threads: int = 1,
        lb_symmetry_probes: int = 32,
    ) -> None:
        if exact_method is not None:
            _reject_retired(exact_method)
            raise GedBackendError(
                f"GedlibBackend is bounds-only since T-03-design.md amendment 2; "
                f"exact_method={exact_method!r} is not available. GEDLIB has no method "
                "that certifies an optimum. Exact GED comes from NetworkxBackend; use "
                "ExactPlusBoundsBackend to get both in one PairResult."
            )
        _reject_retired(lb_method)
        _reject_retired(ub_method)
        _validate_method(lb_method, LOWER_BOUND_METHODS, "lower bound")
        _validate_method(ub_method, UPPER_BOUND_METHODS, "upper bound")

        self._costs = costs
        self._timeout_s = timeout_s
        self._lb_method = lb_method
        self._ub_method = ub_method
        self._lb_symmetry_probes = int(lb_symmetry_probes)
        self._init_option = "EAGER_WITHOUT_SHUFFLED_COPIES"
        self._heuristic_options = f"--threads {int(threads)}"
        self._gedlib_module: Any = None
        self._env: Any = None
        self._reset_mode = "unknown"
        self.stats = BackendStats()

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "gedlib"

    @property
    def costs(self) -> EditCosts:
        """The active cost model."""
        return self._costs

    @property
    def method(self) -> str:
        """Composite solver identification written into every result."""
        return f"{self._lb_method}+{self._ub_method}"

    def module(self) -> Any:
        """Import GEDLIB once, in the order the shared libraries require.

        Returns
        -------
        module
            ``gklearn.gedlib.gedlibpy_gxl``.

        Notes
        -----
        ``libraries_import`` ``dlopen()``s libdoublefann, libsvm and libnomad;
        importing ``gedlibpy_gxl`` first fails with
        ``libdoublefann.so.2: cannot open shared object file``. The two calls
        go through :func:`importlib.import_module` because ruff and isort
        reorder plain ``from ... import`` statements alphabetically, which
        would put ``gedlibpy_gxl`` first and break the load.
        """
        if self._gedlib_module is None:
            importlib.import_module("gklearn.gedlib.libraries_import")
            self._gedlib_module = importlib.import_module("gklearn.gedlib.gedlibpy_gxl")
        return self._gedlib_module

    def env(self) -> Any:
        """Return the process-wide ``GEDEnvGXL``, constructing it on first use.

        Returns
        -------
        object
            The environment. Never rebuilt per pair.
        """
        if self._env is None:
            self._env = self.module().GEDEnvGXL()
            self._reset_mode = "restart_env" if hasattr(self._env, "restart_env") else "rebuild"
            logger.info("GEDLIB environment built; per-pair reset via %s", self._reset_mode)
        return self._env

    def _fresh_env(self) -> Any:
        """Return the environment emptied of graphs, methods and costs."""
        env = self.env()
        if self._reset_mode == "restart_env":
            env.restart_env()
            return env
        self._env = self.module().GEDEnvGXL()
        return self._env

    def _read(self, env: Any, i: int, j: int, accessor: str, method: str, zero_ok: bool) -> float:
        """Read one bound and range-check it.

        Parameters
        ----------
        env : object
            The GEDLIB environment, with ``method`` already run on ``(i, j)``.
        i, j : int
            Graph ids inside the environment.
        accessor : {'lb', 'ub'}
            Which accessor to call.
        method : str
            Method name, for the error message.
        zero_ok : bool
            Whether a value of exactly zero is legal for this pair.

        Returns
        -------
        float
            The bound.

        Raises
        ------
        GedBackendError
            On a non-finite value, a negative value, or an illegal zero -- the
            signature of the wrong accessor, which GEDLIB does not report.
        """
        raw = env.get_lower_bound(i, j) if accessor == "lb" else env.get_upper_bound(i, j)
        value = float(raw)
        if not math.isfinite(value):
            raise GedBackendError(
                f"{method}.get_{'lower' if accessor == 'lb' else 'upper'}_bound returned "
                f"{value!r}; GEDLIB returns inf for a method that cannot produce this bound"
            )
        if value < 0.0:
            raise GedBackendError(f"{method} returned a negative bound {value!r}")
        if value == 0.0:
            if not zero_ok:
                raise GedBackendError(
                    f"{method}.get_{'lower' if accessor == 'lb' else 'upper'}_bound returned "
                    "0.00 for a pair whose distance cannot be zero; this is the wrong "
                    "accessor for the method, and GEDLIB does not raise on it"
                )
            self.stats.n_zero_values_accepted += 1
        return value

    def _run(self, env: Any, method: str, options: str, i: int, j: int) -> None:
        """Select, initialise and run one method on one ordered pair."""
        env.set_method(method, options)
        env.init_method()
        env.run_method(i, j)

    def bounds(self, g1: nx.Graph, g2: nx.Graph) -> tuple[float, float]:
        """Return the GEDLIB bracket ``(lb, ub)`` for one pair.

        Separated from :meth:`pair` so that :class:`ExactPlusBoundsBackend` can
        take the bounds without also taking a ``PairResult`` it would discard.

        Parameters
        ----------
        g1, g2 : networkx.Graph
            The graphs to compare.

        Returns
        -------
        tuple of float
            ``(lb, ub)``. The upper bound is the minimum over both
            orientations: ``IPFP`` builds an edit path from a directed
            assignment, so the two orientations differ, and each is separately
            achievable, which makes their minimum a valid bound and, unlike
            either alone, symmetric.

        Raises
        ------
        GedBackendError
            On any inconsistent read, including an inverted bracket.
        """
        zero_ok = zero_distance_is_attainable(g1, g2, self._costs)

        env = self._fresh_env()
        i0 = env.add_nx_graph(_with_string_labels(g1), "")
        i1 = env.add_nx_graph(_with_string_labels(g2), "")
        env.set_edit_cost("CONSTANT", edit_cost_constant=self._costs.as_gedlib_constant())
        env.init(init_option=self._init_option)

        # A LOWER bound of zero is always mathematically valid -- it is the trivial
        # bound, merely uninformative -- so it must not trip the zero-guard. Measured on
        # Picasso 2026-08-12, BRANCH_FAST returns 0.00 for real pairs whose true GED is
        # 2 and 6; the guard rejected those and failed gates 1 and 3 after ~50 pairs.
        #
        # The `0 < v < inf` rule the plan states exists to catch an accessor MISMATCH --
        # calling get_lower_bound() on an upper-bound method, which silently returns
        # 0.00. That can only manifest on the UPPER bound, where a zero really is
        # impossible unless a zero-cost edit path exists. The lower bound is protected
        # instead by the constructor guard, which refuses any method not in
        # LOWER_BOUND_METHODS. Zero lower bounds are counted, because their rate is the
        # bound-quality signal T-05's calibration ladder reports.
        self._run(env, self._lb_method, self._heuristic_options, i0, i1)
        lb = self._read(env, i0, i1, "lb", self._lb_method, zero_ok=True)
        if lb == 0.0 and not zero_ok:
            self.stats.n_trivial_lower_bounds += 1
        if self.stats.n_lb_orientations_compared < self._lb_symmetry_probes:
            self._run(env, self._lb_method, self._heuristic_options, i1, i0)
            lb_rev = self._read(env, i1, i0, "lb", self._lb_method, zero_ok=True)
            self.stats.record_lb_orientations(lb, lb_rev)
            lb = max(lb, lb_rev)

        self._run(env, self._ub_method, self._heuristic_options, i0, i1)
        ub_fwd = self._read(env, i0, i1, "ub", self._ub_method, zero_ok)
        self._run(env, self._ub_method, self._heuristic_options, i1, i0)
        ub_rev = self._read(env, i1, i0, "ub", self._ub_method, zero_ok)
        self.stats.record_ub_orientations(ub_fwd, ub_rev)
        ub = min(ub_fwd, ub_rev)

        if lb > ub + CERT_TOL:
            raise GedBackendError(
                f"GEDLIB bracket is inverted: {self._lb_method} lb {lb} exceeds "
                f"{self._ub_method} ub {ub}; one of the two is misconfigured"
            )
        return float(min(lb, ub)), float(ub)

    def pair(self, g1: nx.Graph, g2: nx.Graph) -> PairResult:
        """Return the GEDLIB bracket for one pair. Never an exact value.

        Parameters
        ----------
        g1, g2 : networkx.Graph
            The graphs to compare.

        Returns
        -------
        PairResult
            ``exact`` is ``None`` and ``certified`` is ``False``, always.
            GEDLIB has no method that certifies an optimum; the one that
            claimed to was measured wrong (amendment 2). Even a coincidentally
            closed bracket is left uncertified here and reported by gate 3b
            instead, so that no GEDLIB self-report can enter the census as a
            distance.

        Raises
        ------
        GedBackendError
            On any inconsistent read.
        """
        t0 = time.perf_counter()
        lb, ub = self.bounds(g1, g2)
        seconds = time.perf_counter() - t0
        result = PairResult(
            lb=lb,
            ub=ub,
            exact=None,
            certified=False,
            seconds=seconds,
            timed_out=bool(seconds > self._timeout_s),
            method=self.method,
        )
        self.stats.record_result(result)
        return result


class ExactPlusBoundsBackend:
    """Exact distance from NetworkX A*, bounds from a separate backend.

    What production needs per pair after amendment 2, in one
    :class:`PairResult`: ``exact`` proved by A*, ``lb`` and ``ub`` measured by
    GEDLIB, and ``certified`` decided by whether A* completed. The two halves
    are independent implementations, which is the point -- it is what makes
    gate 1's containment check and gate 3b's bound-quality table meaningful.

    The bounds are **not** overwritten by a certified exact value. Gate 3b
    reports rho(exact, LB), rho(exact, UB), relative bias and the ``LB == UB``
    rate, and collapsing the bracket would erase all four.

    Parameters
    ----------
    costs : EditCosts, optional
        Edit operation costs, shared by both halves.
    timeout_s : float, optional
        Per-pair budget for A*. ``None`` or ``inf`` runs it to completion.
    bounds_kind : {'gedlib', 'networkx', 'stub'}, optional
        Which backend supplies ``lb`` and ``ub``. ``'gedlib'`` is production.
        ``'networkx'`` uses the :mod:`ged_bounds` reference implementation and
        exists so the composition is testable on a machine without GEDLIB --
        the resulting bounds are *ours*, not GEDLIB's, and every report that
        carries them says so.
    bounds_options : dict, optional
        Extra keyword arguments for the bounds backend.

    Raises
    ------
    GedBackendError
        If the bounds backend is itself asked for an exact value, or names a
        retired method.
    """

    __slots__ = ("_bounds", "_bounds_kind", "_costs", "_exact", "stats")

    def __init__(
        self,
        costs: EditCosts = UNIT_COSTS,
        *,
        timeout_s: float | None = 300.0,
        bounds_kind: str = "gedlib",
        bounds_options: dict[str, Any] | None = None,
    ) -> None:
        if bounds_kind not in ("gedlib", "networkx", "stub"):
            raise GedBackendError(
                f"unknown bounds backend {bounds_kind!r}; expected gedlib, networkx or stub"
            )
        options = dict(bounds_options or {})
        options.setdefault("timeout_s", 300.0 if timeout_s is None else timeout_s)
        self._costs = costs
        self._bounds_kind = bounds_kind
        self._exact = NetworkxBackend(costs, timeout_s=timeout_s)
        self._bounds: Any = make_backend(bounds_kind, costs, **options)
        self.stats = BackendStats()

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "exact_plus_bounds"

    @property
    def costs(self) -> EditCosts:
        """The active cost model."""
        return self._costs

    @property
    def bounds_source(self) -> str:
        """Provenance label for the bounds, carried into every report.

        A gate 3b table built on ``ged_bounds`` is not a GEDLIB table, and the
        two must never be pooled. Naming the source in the result is how the
        distinction survives into the JSON.
        """
        if self._bounds_kind == "gedlib":
            return f"gedlib:{self._bounds.method}"
        if self._bounds_kind == "networkx":
            return "ged_bounds:branch_lower_bound+bipartite_upper_bound"
        return "stub:trivial_bracket"

    @property
    def method(self) -> str:
        """Composite solver identification written into every result."""
        return f"networkx_astar+{self.bounds_source}"

    def _raw_bounds(self, g1: nx.Graph, g2: nx.Graph) -> tuple[float, float]:
        """Take the bracket from the bounds backend, whichever kind it is."""
        if self._bounds_kind == "gedlib":
            return self._bounds.bounds(g1, g2)
        if self._bounds_kind == "networkx":
            return self._bounds.heuristic_bracket(g1, g2)
        res = self._bounds.pair(g1, g2)
        return float(res.lb), float(res.ub)

    def pair(self, g1: nx.Graph, g2: nx.Graph) -> PairResult:
        """Compute A* exact plus the independent bracket for one pair.

        Parameters
        ----------
        g1, g2 : networkx.Graph
            The graphs to compare.

        Returns
        -------
        PairResult
            ``exact`` set iff A* completed; ``lb``/``ub`` from the bounds
            backend, uncollapsed.

        Raises
        ------
        GedBackendError
            If a bound contradicts the proven optimum. Here the two sides are
            independent implementations, so a contradiction means one of them
            is wrong and the run must stop rather than record a plausible
            number. That is the cross-check CLAUDE.md requires of GEDLIB.
        """
        t0 = time.perf_counter()
        lb, ub = self._raw_bounds(g1, g2)
        exact, best_cost, _solver_seconds, timed_out = self._exact.solve_exact(g1, g2)

        if math.isfinite(best_cost):
            # An A* cost is the cost of a constructed edit path, hence
            # achievable, hence a valid upper bound whether or not it is
            # optimal. Keeping the tighter of the two loses nothing.
            ub = min(ub, best_cost)

        if exact is not None:
            if exact < lb - CERT_TOL:
                self.stats.n_lb_above_exact += 1
                raise GedBackendError(
                    f"{self.bounds_source} lower bound {lb} exceeds the A* optimum {exact}; "
                    "the two are independent implementations, so one of them is wrong"
                )
            if exact > ub + CERT_TOL:
                self.stats.n_ub_below_exact += 1
                raise GedBackendError(
                    f"{self.bounds_source} upper bound {ub} is below the A* optimum {exact}; "
                    "an upper bound below the optimum is not an upper bound"
                )

        result = PairResult(
            lb=float(lb),
            ub=float(ub),
            exact=exact,
            certified=exact is not None,
            seconds=time.perf_counter() - t0,
            timed_out=bool(timed_out),
            method=self.method,
        )
        self.stats.record_result(result)
        return result


def _validate_method(method: str, capable: frozenset[str], role: str) -> None:
    """Reject a method that cannot fill the role it was given.

    Parameters
    ----------
    method : str
        Requested GEDLIB method.
    capable : frozenset of str
        Methods measured to support the role.
    role : str
        Role name, for the error message.

    Raises
    ------
    GedBackendError
        If the method is retired, forbidden, unknown, or not measured capable.
    """
    _reject_retired(method)
    if method not in VERIFIED_METHODS:
        raise GedBackendError(f"{method!r} is not among the 21 methods verified on Picasso")
    if method not in capable:
        raise GedBackendError(
            f"{method!r} was not measured to produce a {role}; reading it would return "
            f"0.00 or inf without raising. Capable: {sorted(capable)}"
        )


def _with_string_labels(g: nx.Graph) -> nx.Graph:
    """Return a copy of ``g`` with a constant string label on every element.

    GEDLIB's GXL bindings require string-valued node and edge attributes.
    Substitution is free under both permitted cost models, so a constant label
    cannot change any distance.

    Parameters
    ----------
    g : networkx.Graph
        Unlabelled input graph.

    Returns
    -------
    networkx.Graph
        A labelled copy.
    """
    h = nx.Graph()
    for v in g.nodes():
        h.add_node(v, label=_CONSTANT_LABEL)
    for a, b in g.edges():
        h.add_edge(a, b, label=_CONSTANT_LABEL)
    return h


@dataclass(frozen=True, slots=True)
class BackendSpec:
    """A picklable description of a backend, for process pools.

    Attributes
    ----------
    kind : {'gedlib', 'networkx', 'exact_plus_bounds', 'stub'}
        Which backend to build.
    costs : EditCosts
        Edit operation costs.
    options : dict
        Extra keyword arguments forwarded to the constructor.
    """

    kind: str
    costs: EditCosts = UNIT_COSTS
    options: dict[str, Any] = field(default_factory=dict)

    def build(self) -> GedBackend:
        """Construct the backend this spec describes."""
        return make_backend(self.kind, self.costs, **self.options)


def make_backend(kind: str, costs: EditCosts = UNIT_COSTS, **options: Any) -> GedBackend:
    """Build a backend by name.

    Parameters
    ----------
    kind : {'gedlib', 'networkx', 'exact_plus_bounds', 'stub'}
        Backend identifier.
    costs : EditCosts, optional
        Edit operation costs.
    **options
        Forwarded to the backend constructor.

    Returns
    -------
    GedBackend
        The constructed backend.

    Raises
    ------
    GedBackendError
        If ``kind`` is unknown.
    """
    if kind == "gedlib":
        return GedlibBackend(costs, **options)
    if kind == "networkx":
        return NetworkxBackend(costs, **options)
    if kind == "exact_plus_bounds":
        return ExactPlusBoundsBackend(costs, **options)
    if kind == "stub":
        return StubBackend(costs, **options)
    raise GedBackendError(
        f"unknown backend {kind!r}; expected gedlib, networkx, exact_plus_bounds or stub"
    )
