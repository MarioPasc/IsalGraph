"""GED backends -- Contract B of wave ``2026-08-12-exact-ged``.

Three interchangeable ways to obtain a graph edit distance, behind one frozen
result type. All three return a :class:`PairResult` whose ``exact`` field is
``None`` unless the value carries an optimality certificate.

Why the certificate matters
---------------------------
``networkx.graph_edit_distance(g1, g2, timeout=t)`` returns its *best-found-so-
far* cost when the budget expires. It does not raise, and it returns ``None``
only when no complete edit path was found at all. Code that stores that value
as "exact GED" is storing an uncertified upper bound.
:class:`NetworkxBackend` detects the cut-off and refuses to promote the value.

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

#: Tolerance at which a bracket is treated as closed, hence certified.
CERT_TOL = 1e-9

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

#: Methods that return a certified optimum, reading both accessors.
EXACT_METHODS = frozenset({"ANCHOR_AWARE_GED"})

#: Never used. Returns ``LB = 0`` / ``UB = inf`` under default options and the
#: cause is undiagnosed (gedlib.md, "unresolved, do not use yet").
FORBIDDEN_METHODS = frozenset({"HED"})


class GedBackendError(Exception):
    """Raised when a backend is misconfigured or a solver returns a bad value."""


@dataclass(frozen=True, slots=True)
class PairResult:
    """One graph pair, one backend.

    Attributes
    ----------
    lb : float
        Certified lower bound. Finite, and positive unless a zero distance is
        genuinely attainable for the pair.
    ub : float
        Certified upper bound, symmetrised over both argument orders.
    exact : float or None
        The distance, set **if and only if** ``certified``. Never a
        best-so-far value.
    certified : bool
        ``True`` iff ``ub - lb <= CERT_TOL``, which certifies optimality.
    seconds : float
        Wall time spent on this pair by this backend.
    timed_out : bool
        The solver's budget expired before it proved optimality.
    method : str
        Solver identification, e.g. ``'ANCHOR_AWARE_GED+BRANCH_FAST+IPFP'``.

    Raises
    ------
    GedBackendError
        If the fields are mutually inconsistent. Validating here means every
        backend, present and future, is held to invariant 4 of Contract B.
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
        closed = (self.ub - self.lb) <= CERT_TOL
        if self.certified != closed:
            raise GedBackendError(
                f"certified={self.certified} contradicts bracket [{self.lb}, {self.ub}]"
            )
        if self.certified and self.exact is None:
            raise GedBackendError("certified result must carry an exact value")
        if not self.certified and self.exact is not None:
            raise GedBackendError(
                f"uncertified result must not carry exact={self.exact!r}; "
                "a best-so-far cost is an upper bound, not a distance"
            )
        if self.exact is not None and not (self.lb - CERT_TOL <= self.exact <= self.ub + CERT_TOL):
            raise GedBackendError(f"exact {self.exact} outside bracket [{self.lb}, {self.ub}]")
        if self.seconds < 0.0:
            raise GedBackendError(f"seconds must be non-negative, got {self.seconds!r}")


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


class NetworkxBackend:
    """Bounds from :mod:`ged_bounds`, exact distance from NetworkX A*.

    The exact stage is the one that needs care. ``nx.graph_edit_distance``
    returns its best-found-so-far cost when ``timeout`` expires, so the value
    alone does not distinguish an optimum from an upper bound. Two independent
    certificates are accepted:

    1. the search returned before its deadline, so it exhausted the space;
    2. the returned cost meets the BRANCH lower bound, which proves optimality
       whatever the solver did.

    Failing both, the cost is folded into ``ub`` and ``exact`` stays ``None``.

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

    def pair(self, g1: nx.Graph, g2: nx.Graph) -> PairResult:
        """Compute the bracket and, where certifiable, the exact distance.

        Parameters
        ----------
        g1, g2 : networkx.Graph
            The graphs to compare.

        Returns
        -------
        PairResult
            ``exact`` is set only when optimality is certified.
        """
        t0 = time.perf_counter()
        lb, ub = self.heuristic_bracket(g1, g2)

        budget = self._timeout_s
        unlimited = budget is None or not math.isfinite(budget)
        t1 = time.perf_counter()
        value = exact_ged(g1, g2, self._costs, timeout=None if unlimited else budget)
        solver_seconds = time.perf_counter() - t1

        deadline_hit = (not unlimited) and solver_seconds >= float(budget)  # type: ignore[arg-type]
        found = math.isfinite(value)
        if found:
            ub = min(ub, value)
        proven = found and (not deadline_hit or value <= lb + CERT_TOL)
        if proven:
            lb = ub = float(value)

        certified = (ub - lb) <= CERT_TOL
        result = PairResult(
            lb=float(lb),
            ub=float(ub),
            exact=float(ub) if certified else None,
            certified=certified,
            seconds=time.perf_counter() - t0,
            timed_out=bool(deadline_hit and not proven),
            method="networkx_astar",
        )
        self.stats.record_result(result)
        return result


class GedlibBackend:
    """GEDLIB bounds and exact distance, with every silent trap guarded.

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
        recorded; it does not abort a running solve. Pass
        ``exact_time_limit_s`` to ask the solver itself to stop early.
    exact_method : str or None, optional
        Certifying solver, or ``None`` to skip the exact stage and return a
        pure heuristic bracket.
    lb_method : str, optional
        Lower-bound method, read through ``get_lower_bound()``.
    ub_method : str, optional
        Upper-bound method, read through ``get_upper_bound()`` in both
        orientations.
    threads : int, optional
        Value of GEDLIB's ``--threads`` option. Defaults to 1, matching the
        one-process-per-core layout the cluster jobs use.
    exact_time_limit_s : float or None, optional
        If set, appends ``--time-limit`` to the exact method's options.
        **Unverified**: whether ``ANCHOR_AWARE_GED`` accepts that option has
        not been measured, and GEDLIB rejects unknown options, so the default
        is ``None``.
    lb_symmetry_probes : int, optional
        Number of leading pairs on which the lower bound is also evaluated in
        the reverse orientation, to measure rather than assume its symmetry.

    Raises
    ------
    GedBackendError
        If any method is forbidden, unverified, or paired with an accessor it
        was not measured to support.
    """

    __slots__ = (
        "_costs",
        "_env",
        "_exact_method",
        "_exact_options",
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
        exact_method: str | None = "ANCHOR_AWARE_GED",
        lb_method: str = "BRANCH_FAST",
        ub_method: str = "IPFP",
        threads: int = 1,
        exact_time_limit_s: float | None = None,
        lb_symmetry_probes: int = 32,
    ) -> None:
        _validate_method(lb_method, LOWER_BOUND_METHODS, "lower bound")
        _validate_method(ub_method, UPPER_BOUND_METHODS, "upper bound")
        if exact_method is not None:
            _validate_method(exact_method, EXACT_METHODS, "exact")

        self._costs = costs
        self._timeout_s = timeout_s
        self._exact_method = exact_method
        self._lb_method = lb_method
        self._ub_method = ub_method
        self._lb_symmetry_probes = int(lb_symmetry_probes)
        self._init_option = "EAGER_WITHOUT_SHUFFLED_COPIES"
        self._heuristic_options = f"--threads {int(threads)}"
        self._exact_options = self._heuristic_options
        if exact_time_limit_s is not None:
            self._exact_options += f" --time-limit {int(exact_time_limit_s)}"
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
        parts = [self._exact_method] if self._exact_method else []
        parts += [self._lb_method, self._ub_method]
        return "+".join(parts)

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

    def pair(self, g1: nx.Graph, g2: nx.Graph) -> PairResult:
        """Compute the bracket and, where certifiable, the exact distance.

        Parameters
        ----------
        g1, g2 : networkx.Graph
            The graphs to compare.

        Returns
        -------
        PairResult
            ``exact`` is set only when ``ANCHOR_AWARE_GED`` closed the bracket.

        Raises
        ------
        GedBackendError
            On any inconsistent read, including a lower bound above a proven
            optimum.
        """
        t0 = time.perf_counter()
        zero_ok = zero_distance_is_attainable(g1, g2, self._costs)

        env = self._fresh_env()
        i0 = env.add_nx_graph(_with_string_labels(g1), "")
        i1 = env.add_nx_graph(_with_string_labels(g2), "")
        env.set_edit_cost("CONSTANT", edit_cost_constant=self._costs.as_gedlib_constant())
        env.init(init_option=self._init_option)

        self._run(env, self._lb_method, self._heuristic_options, i0, i1)
        lb = self._read(env, i0, i1, "lb", self._lb_method, zero_ok)
        if self.stats.n_lb_orientations_compared < self._lb_symmetry_probes:
            self._run(env, self._lb_method, self._heuristic_options, i1, i0)
            lb_rev = self._read(env, i1, i0, "lb", self._lb_method, zero_ok)
            self.stats.record_lb_orientations(lb, lb_rev)
            lb = max(lb, lb_rev)

        self._run(env, self._ub_method, self._heuristic_options, i0, i1)
        ub_fwd = self._read(env, i0, i1, "ub", self._ub_method, zero_ok)
        self._run(env, self._ub_method, self._heuristic_options, i1, i0)
        ub_rev = self._read(env, i1, i0, "ub", self._ub_method, zero_ok)
        self.stats.record_ub_orientations(ub_fwd, ub_rev)
        ub = min(ub_fwd, ub_rev)

        timed_out = False
        if self._exact_method is not None:
            self._run(env, self._exact_method, self._exact_options, i0, i1)
            ex_lb = self._read(env, i0, i1, "lb", self._exact_method, zero_ok)
            ex_ub = self._read(env, i0, i1, "ub", self._exact_method, zero_ok)
            if ex_lb > ex_ub + CERT_TOL:
                raise GedBackendError(f"{self._exact_method} returned lb {ex_lb} above ub {ex_ub}")
            if (ex_ub - ex_lb) <= CERT_TOL:
                if ex_ub < lb - CERT_TOL:
                    raise GedBackendError(
                        f"{self._lb_method} lower bound {lb} exceeds the certified optimum "
                        f"{ex_ub}; one of the two methods is misconfigured"
                    )
                lb = ub = ex_ub
            else:
                lb = max(lb, ex_lb)
                ub = min(ub, ex_ub)
                timed_out = True

        if lb > ub + CERT_TOL:
            raise GedBackendError(f"combined bracket is inverted: lb {lb} > ub {ub}")
        lb = min(lb, ub)
        certified = (ub - lb) <= CERT_TOL
        seconds = time.perf_counter() - t0
        result = PairResult(
            lb=float(lb),
            ub=float(ub),
            exact=float(ub) if certified else None,
            certified=certified,
            seconds=seconds,
            timed_out=bool(timed_out or seconds > self._timeout_s),
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
        If the method is forbidden, unknown, or not measured capable.
    """
    if method in FORBIDDEN_METHODS:
        raise GedBackendError(
            f"{method} is never used: it returns lower bound 0 and upper bound inf "
            "under default options and the cause is undiagnosed"
        )
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
    kind : {'gedlib', 'networkx', 'stub'}
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
    kind : {'gedlib', 'networkx', 'stub'}
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
    if kind == "stub":
        return StubBackend(costs, **options)
    raise GedBackendError(f"unknown backend {kind!r}; expected gedlib, networkx or stub")
