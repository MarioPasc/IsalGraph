"""Unit tests for the Contract B GED backends.

GEDLIB exists only on Picasso, so every GEDLIB interaction here runs against a
fake module installed on :data:`sys.meta_path`. The fake reproduces the two
failure modes that do not raise -- the wrong accessor returning ``0.00`` and
``gedlibpy_gxl`` failing to load before ``libraries_import`` has ``dlopen()``ed
the shared objects -- so the call sequence, the argument values and the
accessor choice are all covered without the library.

The five invariants of CONTRACTS §5 are tested by
:class:`TestInvariant1ZeroGuard` through :class:`TestInvariant5ImportOrder`.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import math
import sys
import types
from collections.abc import Callable, Iterator
from typing import Any

import networkx as nx
import pytest

from benchmarks.eval_setup import ged_backends as ged_backends_module
from benchmarks.eval_setup.ged_backends import (
    CERT_TOL,
    FORBIDDEN_METHODS,
    RETIRED_METHODS,
    ROLE_SPECS,
    BackendSpec,
    GedBackend,
    GedBackendError,
    GedlibBackend,
    NetworkxBackend,
    PairResult,
    StubBackend,
    make_backend,
    zero_distance_is_attainable,
)
from benchmarks.eval_setup.ged_bounds import GRAPHEDX_COSTS, UNIT_COSTS, EditCosts

# --------------------------------------------------------------------------
# A fake GEDLIB, installed through the import system so that import ORDER is
# observable. Pre-seeding sys.modules would not work: import_module would
# return the cached object without ever executing anything.
# --------------------------------------------------------------------------

_LOADED: list[str] = []


class _FakeEnv:
    """Stand-in for ``GEDEnvGXL`` that records every call it receives."""

    def __init__(self, behaviour: dict[str, Any]) -> None:
        self.behaviour = behaviour
        self.calls: list[tuple[str, Any]] = []
        self.graphs: list[nx.Graph] = []
        self.edit_cost: tuple[str, list[float]] | None = None
        self.init_option: str | None = None
        self.method: str | None = None
        self.method_options: str | None = None
        self.last_run: tuple[int, int] | None = None
        self.n_restarts = 0

    def restart_env(self) -> None:
        self.n_restarts += 1
        self.calls.append(("restart_env", None))
        self.graphs = []
        self.edit_cost = None
        self.method = None

    def add_nx_graph(self, graph: nx.Graph, classe: str, **kwargs: object) -> int:
        self.calls.append(("add_nx_graph", classe))
        for _, data in graph.nodes(data=True):
            for value in data.values():
                assert isinstance(value, str), "node attributes must be strings"
        for _, _, data in graph.edges(data=True):
            for value in data.values():
                assert isinstance(value, str), "edge attributes must be strings"
        self.graphs.append(graph)
        return len(self.graphs) - 1

    def set_edit_cost(self, name: str, edit_cost_constant: list[float]) -> None:
        self.calls.append(("set_edit_cost", (name, list(edit_cost_constant))))
        self.edit_cost = (name, list(edit_cost_constant))

    def init(self, init_option: str = "EAGER_WITHOUT_SHUFFLED_COPIES") -> None:
        self.calls.append(("init", init_option))
        self.init_option = init_option

    def set_method(self, method: str, options: str = "") -> None:
        self.calls.append(("set_method", (method, options)))
        if (
            method in self.behaviour.get("reject_options", {})
            and options in self.behaviour["reject_options"][method]
        ):
            raise RuntimeError(f"Invalid option {options!r} for {method}")
        self.method = method
        self.method_options = options

    def init_method(self) -> None:
        self.calls.append(("init_method", self.method))

    def run_method(self, i: int, j: int) -> None:
        self.calls.append(("run_method", (self.method, i, j)))
        self.last_run = (i, j)

    def _value(self, accessor: str, i: int, j: int) -> float:
        table = self.behaviour["values"][self.method]
        entry = table[accessor]
        return float(entry((i, j)) if callable(entry) else entry)

    def get_lower_bound(self, i: int, j: int) -> float:
        self.calls.append(("get_lower_bound", (self.method, i, j)))
        return self._value("lb", i, j)

    def get_upper_bound(self, i: int, j: int) -> float:
        self.calls.append(("get_upper_bound", (self.method, i, j)))
        return self._value("ub", i, j)


class _FakeLoader(importlib.abc.Loader):
    """Executes the fake GEDLIB submodules and records the order."""

    def __init__(self, behaviour: dict[str, Any]) -> None:
        self.behaviour = behaviour

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> types.ModuleType | None:
        return None

    def exec_module(self, module: types.ModuleType) -> None:
        name = module.__name__
        _LOADED.append(name)
        if (
            name == "gklearn.gedlib.gedlibpy_gxl"
            and "gklearn.gedlib.libraries_import" not in _LOADED
        ):
            raise ImportError("libdoublefann.so.2: cannot open shared object file")
        if name == "gklearn.gedlib.gedlibpy_gxl":
            behaviour = self.behaviour
            module.GEDEnvGXL = lambda: _FakeEnv(behaviour)  # type: ignore[attr-defined]
            module.list_of_method_options = ["BRANCH_FAST", "IPFP", "ANCHOR_AWARE_GED"]  # type: ignore[attr-defined]
            module.list_of_edit_cost_options = ["CONSTANT"]  # type: ignore[attr-defined]


class _FakeFinder(importlib.abc.MetaPathFinder):
    """Serves the fake ``gklearn`` package tree."""

    NAMES = frozenset(
        {
            "gklearn",
            "gklearn.gedlib",
            "gklearn.gedlib.libraries_import",
            "gklearn.gedlib.gedlibpy_gxl",
        }
    )

    def __init__(self, behaviour: dict[str, Any]) -> None:
        self.loader = _FakeLoader(behaviour)

    def find_spec(
        self, fullname: str, path: object = None, target: object = None
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname not in self.NAMES:
            return None
        is_pkg = fullname in ("gklearn", "gklearn.gedlib")
        return importlib.machinery.ModuleSpec(fullname, self.loader, is_package=is_pkg)


def _default_behaviour() -> dict[str, Any]:
    """P4 vs C4: true GED 1, as measured on Picasso 2026-08-11."""
    return {
        "values": {
            "BRANCH_FAST": {"lb": 1.0, "ub": 0.0},
            "IPFP": {"lb": 0.0, "ub": 1.0},
            "ANCHOR_AWARE_GED": {"lb": 1.0, "ub": 1.0},
        }
    }


#: Signature of the fixture that installs the fake library.
_InstallFake = Callable[..., "_FakeFinder"]


@pytest.fixture
def fake_gedlib(monkeypatch: pytest.MonkeyPatch) -> Iterator[_InstallFake]:
    """Install the fake GEDLIB for the duration of one test."""

    def install(behaviour: dict[str, Any] | None = None) -> _FakeFinder:
        _LOADED.clear()
        for name in list(sys.modules):
            if name == "gklearn" or name.startswith("gklearn."):
                monkeypatch.delitem(sys.modules, name, raising=False)
        finder = _FakeFinder(behaviour or _default_behaviour())
        monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
        return finder

    yield install
    for name in list(sys.modules):
        if name == "gklearn" or name.startswith("gklearn."):
            del sys.modules[name]
    _LOADED.clear()


P4 = nx.path_graph(4)
C4 = nx.cycle_graph(4)

#: A pair whose heuristic bracket stays open (BRANCH 4, bipartite 7, true 7).
#: Needed wherever a test must reach the exact solver's certification path
#: rather than certify on the bracket alone.
OPEN_PAIR = (nx.star_graph(5), nx.cycle_graph(6))


# --------------------------------------------------------------------------


class TestPairResultContract:
    """PairResult validates the whole contract, so every backend inherits it."""

    def test_certified_result_is_accepted(self) -> None:
        r = PairResult(1.0, 1.0, 1.0, True, 0.1, False, "m")
        assert r.exact == 1.0

    def test_uncertified_result_must_not_carry_exact(self) -> None:
        """This is the ged_computer.py defect, refused at the type level."""
        with pytest.raises(GedBackendError, match="best-so-far"):
            PairResult(1.0, 5.0, 3.0, False, 0.1, False, "m")

    def test_a_certified_exact_must_lie_inside_the_bracket(self) -> None:
        """The bracket invariant survives amendment 2; the old equality one does not.

        Before, ``certified`` meant ``lb == ub`` and an open bracket with a certified
        value was a contradiction. Now ``certified`` means *A\\* ran to completion*, and
        the bounds come from a different library entirely, so an open bracket around a
        certified value is the normal case: exact=3 inside GEDLIB's [1, 5] is sound.
        What must still never happen is a certified value outside its own bracket --
        that is gate 1's violation condition, caught here at construction.
        """
        PairResult(1.0, 5.0, 3.0, True, 0.1, False, "m")  # open bracket, sound
        with pytest.raises(GedBackendError):
            PairResult(1.0, 5.0, 7.0, True, 0.1, False, "m")  # above ub
        with pytest.raises(GedBackendError):
            PairResult(4.0, 5.0, 2.0, True, 0.1, False, "m")  # below lb

    def test_certified_result_must_carry_exact(self) -> None:
        with pytest.raises(GedBackendError, match="must carry an exact value"):
            PairResult(1.0, 1.0, None, True, 0.1, False, "m")

    def test_inverted_bracket_is_refused(self) -> None:
        with pytest.raises(GedBackendError, match="exceeds ub"):
            PairResult(5.0, 1.0, None, False, 0.1, False, "m")

    @pytest.mark.parametrize("bad", [math.inf, -1.0, math.nan])
    def test_non_finite_or_negative_bounds_are_refused(self, bad: float) -> None:
        with pytest.raises(GedBackendError):
            PairResult(bad, bad, None, False, 0.1, False, "m")

    def test_exact_outside_the_bracket_is_refused(self) -> None:
        with pytest.raises(GedBackendError, match="outside bracket"):
            PairResult(2.0, 2.0, 7.0, True, 0.1, False, "m")


class TestZeroLegality:
    """A zero is legal only where a zero-cost edit path exists."""

    def test_isomorphic_graphs_may_be_zero(self) -> None:
        assert zero_distance_is_attainable(C4, nx.cycle_graph(4), UNIT_COSTS)

    def test_distinct_graphs_may_not_be_zero(self) -> None:
        assert not zero_distance_is_attainable(P4, C4, UNIT_COSTS)

    def test_isolated_nodes_are_free_under_the_graphedx_model(self) -> None:
        """Node operations cost nothing there, so an extra isolated node is free."""
        g = nx.cycle_graph(4)
        h = nx.cycle_graph(4)
        h.add_node("spare")
        assert zero_distance_is_attainable(g, h, GRAPHEDX_COSTS)
        assert not zero_distance_is_attainable(g, h, UNIT_COSTS)


class TestStubBackend:
    """The stub must satisfy the contract without running a solver."""

    def test_bounds_bracket_the_true_distance(self) -> None:
        r = StubBackend().pair(P4, C4)
        assert r.lb <= 1.0 <= r.ub
        assert r.exact is None

    def test_identical_empty_graphs_certify_at_zero(self) -> None:
        empty = nx.Graph()
        r = StubBackend().pair(empty, empty)
        assert r.certified and r.exact == 0.0

    def test_output_is_deterministic(self) -> None:
        a, b = StubBackend().pair(P4, C4), StubBackend().pair(P4, C4)
        assert a == b

    def test_satisfies_the_protocol(self) -> None:
        assert isinstance(StubBackend(), GedBackend)


class TestNetworkxBackend:
    """The exact stage must never promote a best-so-far cost."""

    def test_exact_is_certified_without_a_deadline(self) -> None:
        r = NetworkxBackend(timeout_s=None).pair(P4, C4)
        assert r.certified and r.exact == 1.0 and not r.timed_out

    def test_p4_vs_c4_matches_the_picasso_reference(self) -> None:
        assert NetworkxBackend(timeout_s=30.0).pair(P4, C4).exact == 1.0

    def test_the_open_pair_really_is_open(self) -> None:
        """Guard the three tests below against silently going vacuous.

        A pair whose heuristic bracket is already closed certifies on the
        bracket alone, whatever the exact solver did, so it cannot exercise the
        timeout path at all.
        """
        import benchmarks.eval_setup.ged_backends as mod

        lb = mod.branch_lower_bound(*OPEN_PAIR, UNIT_COSTS)
        ub = mod.bipartite_upper_bound(*OPEN_PAIR, UNIT_COSTS)
        assert ub - lb > CERT_TOL

    def test_a_cut_off_search_yields_no_exact_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A returned best-so-far cost becomes an upper bound, never a distance."""
        import benchmarks.eval_setup.ged_backends as mod

        monkeypatch.setattr(mod, "exact_ged", lambda *a, **k: _slow_value(99.0))
        r = mod.NetworkxBackend(timeout_s=0.001).pair(*OPEN_PAIR)
        assert r.exact is None
        assert r.timed_out
        assert r.ub <= 99.0

    def test_a_cut_off_search_does_not_certify_even_meeting_the_lower_bound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Certification is decided by completion, not by the value that came back.

        Reversed by design note amendment 2. Meeting the BRANCH bound does prove
        optimality mathematically, but ``nx.graph_edit_distance`` returns its
        best-so-far cost on timeout, so "the value happens to equal the bound" is
        not evidence the search finished. Trusting it is the ged_computer.py defect
        wearing a proof. The conservative reading costs a handful of pairs, which
        A* then computes properly; the permissive one silently mints exact values.
        """
        import benchmarks.eval_setup.ged_backends as mod

        lb = mod.branch_lower_bound(*OPEN_PAIR, UNIT_COSTS)
        monkeypatch.setattr(mod, "exact_ged", lambda *a, **k: _slow_value(lb))
        r = mod.NetworkxBackend(timeout_s=0.001).pair(*OPEN_PAIR)
        assert r.timed_out
        assert not r.certified
        assert r.exact is None

    def test_no_complete_path_leaves_the_bracket_open(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import benchmarks.eval_setup.ged_backends as mod

        monkeypatch.setattr(mod, "exact_ged", lambda *a, **k: _slow_value(math.inf))
        r = mod.NetworkxBackend(timeout_s=0.001).pair(*OPEN_PAIR)
        assert r.exact is None and math.isfinite(r.ub)

    def test_upper_bound_asymmetry_is_measured(self) -> None:
        backend = NetworkxBackend(timeout_s=5.0)
        for a, b in ((P4, C4), (nx.star_graph(5), nx.path_graph(6))):
            backend.pair(a, b)
        stats = backend.stats.as_dict()
        assert stats["n_ub_orientations_compared"] == 2
        assert 0.0 <= stats["ub_asymmetry_rate"] <= 1.0


def _slow_value(value: float) -> float:
    """Return ``value`` after a delay long enough to trip a 1 ms deadline."""
    import time as _time

    _time.sleep(0.01)
    return value


class TestInvariant1ZeroGuard:
    """Invariant 1 -- every GEDLIB read is asserted ``0 < value < inf``."""

    def test_wrong_accessor_zero_raises_instead_of_filling_a_matrix(
        self, fake_gedlib: _InstallFake
    ) -> None:
        """The trap can only manifest on the UPPER bound, so that is where it is tested.

        Moved off the lower bound, where the original version put it. A lower bound of
        zero is the trivial bound and is always valid: BRANCH_FAST really does return
        0.00 on real pairs with true GED 2 and 6 (measured on Picasso, 2026-08-12), and
        guarding it failed gates 1 and 3 after ~50 pairs on a correct read. An upper
        bound of zero on a non-isomorphic pair is impossible, so a zero there is the
        signature of ``get_upper_bound()`` having been called on a lower-bound method.
        """
        behaviour = _default_behaviour()
        behaviour["values"]["IPFP"]["ub"] = 0.0  # what a LB-only method returns here
        fake_gedlib(behaviour)
        with pytest.raises(GedBackendError, match="wrong accessor"):
            GedlibBackend(UNIT_COSTS).pair(P4, C4)

    def test_a_trivial_zero_lower_bound_is_recorded_not_refused(
        self, fake_gedlib: _InstallFake
    ) -> None:
        """BRANCH_FAST returning 0 is loose, not broken; the rate is a reported figure."""
        behaviour = _default_behaviour()
        behaviour["values"]["BRANCH_FAST"]["lb"] = 0.0
        fake_gedlib(behaviour)
        backend = GedlibBackend(UNIT_COSTS)
        r = backend.pair(P4, C4)  # P4 and C4 are NOT isomorphic
        assert r.lb == 0.0
        assert backend.stats.n_trivial_lower_bounds == 1

    def test_infinite_read_raises(self, fake_gedlib: _InstallFake) -> None:
        behaviour = _default_behaviour()
        behaviour["values"]["IPFP"]["ub"] = math.inf
        fake_gedlib(behaviour)
        with pytest.raises(GedBackendError, match="inf"):
            GedlibBackend(UNIT_COSTS).pair(P4, C4)

    def test_negative_read_raises(self, fake_gedlib: _InstallFake) -> None:
        behaviour = _default_behaviour()
        behaviour["values"]["BRANCH_FAST"]["lb"] = -1.0
        fake_gedlib(behaviour)
        with pytest.raises(GedBackendError, match="negative"):
            GedlibBackend(UNIT_COSTS).pair(P4, C4)

    def test_zero_is_accepted_for_isomorphic_graphs(self, fake_gedlib: _InstallFake) -> None:
        """A genuine zero must pass the zero-guard; only an UNCERTIFIED zero is the trap.

        The corpus contains isomorphic duplicates -- both IAM Letter and AIDS -- so
        zero is a legitimate distance. GEDLIB is bounds-only after amendment 2, so it
        reports the closed bracket without claiming an optimum.
        """
        behaviour = {
            "values": {
                "BRANCH_FAST": {"lb": 0.0, "ub": 0.0},
                "IPFP": {"lb": 0.0, "ub": 0.0},
            }
        }
        fake_gedlib(behaviour)
        r = GedlibBackend(UNIT_COSTS).pair(C4, nx.cycle_graph(4))
        assert (r.lb, r.ub) == (0.0, 0.0)
        assert r.exact is None
        assert not r.certified

    def test_lower_bound_above_the_upper_bound_raises(self, fake_gedlib: _InstallFake) -> None:
        """An inverted bracket means one of the two GEDLIB methods is misconfigured."""
        behaviour = _default_behaviour()
        behaviour["values"]["BRANCH_FAST"]["lb"] = 9.0
        fake_gedlib(behaviour)
        with pytest.raises(GedBackendError, match="inverted"):
            GedlibBackend(UNIT_COSTS).pair(P4, C4)


class TestInvariant2NoHed:
    """Invariant 2 -- HED is never used."""

    def test_hed_is_refused_in_every_role(self) -> None:
        assert frozenset({"HED"}) == FORBIDDEN_METHODS
        for kwargs in ({"lb_method": "HED"}, {"ub_method": "HED"}, {"exact_method": "HED"}):
            with pytest.raises(GedBackendError, match="HED is never used"):
                GedlibBackend(UNIT_COSTS, **kwargs)  # type: ignore[arg-type]

    def test_an_unverified_method_is_refused(self) -> None:
        with pytest.raises(GedBackendError, match="not among the 21"):
            GedlibBackend(UNIT_COSTS, lb_method="NOT_A_METHOD")

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"lb_method": "IPFP"}, "lower bound"),
            ({"ub_method": "BRANCH_FAST"}, "upper bound"),
            ({"exact_method": "BIPARTITE"}, "exact"),
        ],
    )
    def test_an_accessor_the_method_cannot_serve_is_refused(
        self, kwargs: dict[str, str], match: str
    ) -> None:
        """The trap is caught at construction, before a matrix fills with zeros."""
        with pytest.raises(GedBackendError, match=match):
            GedlibBackend(UNIT_COSTS, **kwargs)  # type: ignore[arg-type]


class TestInvariant3UpperBoundOrientation:
    """Invariant 3 -- upper bounds are computed both ways and minimised."""

    def test_both_orientations_are_run_and_the_minimum_taken(
        self, fake_gedlib: _InstallFake
    ) -> None:
        behaviour = _default_behaviour()
        behaviour["values"]["IPFP"]["ub"] = lambda ij: 3.0 if ij == (0, 1) else 2.0
        behaviour["values"]["ANCHOR_AWARE_GED"] = {"lb": 1.0, "ub": 2.0}
        fake_gedlib(behaviour)
        backend = GedlibBackend(UNIT_COSTS)
        result = backend.pair(P4, C4)
        env = backend.env()
        runs = [c[1] for c in env.calls if c[0] == "run_method" and c[1][0] == "IPFP"]
        assert ("IPFP", 0, 1) in runs and ("IPFP", 1, 0) in runs
        assert result.ub == 2.0

    def test_the_asymmetry_rate_is_recorded(self, fake_gedlib: _InstallFake) -> None:
        behaviour = _default_behaviour()
        behaviour["values"]["IPFP"]["ub"] = lambda ij: 3.0 if ij == (0, 1) else 2.0
        behaviour["values"]["ANCHOR_AWARE_GED"] = {"lb": 1.0, "ub": 2.0}
        fake_gedlib(behaviour)
        backend = GedlibBackend(UNIT_COSTS)
        backend.pair(P4, C4)
        stats = backend.stats.as_dict()
        assert stats["n_ub_asymmetric"] == 1
        assert stats["ub_asymmetry_rate"] == 1.0
        assert stats["max_ub_gap"] == 1.0

    def test_lower_bound_symmetry_is_measured_not_assumed(self, fake_gedlib: _InstallFake) -> None:
        fake_gedlib()
        backend = GedlibBackend(UNIT_COSTS, lb_symmetry_probes=4)
        backend.pair(P4, C4)
        assert backend.stats.as_dict()["n_lb_orientations_compared"] == 1

    def test_the_symmetry_probe_stops_after_its_budget(self, fake_gedlib: _InstallFake) -> None:
        fake_gedlib()
        backend = GedlibBackend(UNIT_COSTS, lb_symmetry_probes=0)
        backend.pair(P4, C4)
        assert backend.stats.as_dict()["n_lb_orientations_compared"] == 0


class TestInvariant4ExactOnlyWhenCertified:
    """Invariant 4 -- ``exact`` is ``None`` unless ``lb == ub``."""

    def test_a_closed_gedlib_bracket_does_not_certify(self, fake_gedlib: _InstallFake) -> None:
        """Deliberately conservative: GEDLIB never mints an exact value after amendment 2.

        ``lb == ub`` from a *sound* lower and a *sound* upper bound does determine GED,
        so this is stricter than the mathematics requires. It is stricter on purpose:
        ANCHOR_AWARE_GED was trusted on exactly that kind of self-report and turned out
        to be a randomised heuristic with a false certificate. Until gate 1 has
        validated BRANCH_FAST and IPFP on real pairs, the closed bracket is recorded as
        a statistic and the value is recomputed by A*. The measured rate is ~1.5% of
        pairs, so the wasted work is small and the failure it prevents is not.
        """
        fake_gedlib()
        r = GedlibBackend(UNIT_COSTS).pair(P4, C4)
        assert (r.lb, r.ub) == (1.0, 1.0)
        assert r.exact is None
        assert not r.certified

    def test_an_open_bracket_does_not_either(self, fake_gedlib: _InstallFake) -> None:
        behaviour = _default_behaviour()
        behaviour["values"]["IPFP"]["ub"] = 4.0
        fake_gedlib(behaviour)
        r = GedlibBackend(UNIT_COSTS).pair(P4, C4)
        assert r.exact is None and not r.certified
        assert (r.lb, r.ub) == (1.0, 4.0)

    def test_the_exact_stage_can_be_skipped(self, fake_gedlib: _InstallFake) -> None:
        """A heuristic-only bracket is what gate 2 compares against ged_bounds."""
        behaviour = _default_behaviour()
        behaviour["values"]["IPFP"]["ub"] = 4.0
        fake_gedlib(behaviour)
        backend = GedlibBackend(UNIT_COSTS, exact_method=None)
        r = backend.pair(P4, C4)
        assert r.exact is None
        assert (r.lb, r.ub) == (1.0, 4.0)
        assert "ANCHOR_AWARE_GED" not in r.method


class TestInvariant5ImportOrder:
    """Invariant 5 -- ``libraries_import`` loads before ``gedlibpy_gxl``."""

    def test_libraries_import_is_loaded_first(self, fake_gedlib: _InstallFake) -> None:
        fake_gedlib()
        GedlibBackend(UNIT_COSTS).module()
        assert _LOADED.index("gklearn.gedlib.libraries_import") < _LOADED.index(
            "gklearn.gedlib.gedlibpy_gxl"
        )

    def test_the_wrong_order_is_what_the_fake_punishes(self, fake_gedlib: _InstallFake) -> None:
        """Proof that the ordering test above is not vacuous."""
        fake_gedlib()
        with pytest.raises(ImportError, match="libdoublefann"):
            importlib.import_module("gklearn.gedlib.gedlibpy_gxl")

    def test_the_module_is_imported_once(self, fake_gedlib: _InstallFake) -> None:
        fake_gedlib()
        backend = GedlibBackend(UNIT_COSTS)
        assert backend.module() is backend.module()
        assert _LOADED.count("gklearn.gedlib.gedlibpy_gxl") == 1


class TestGedlibCallSequence:
    """The argument values and the call order the library requires."""

    def test_environment_is_built_once_and_reset_per_pair(self, fake_gedlib: _InstallFake) -> None:
        fake_gedlib()
        backend = GedlibBackend(UNIT_COSTS)
        for _ in range(3):
            backend.pair(P4, C4)
        env = backend.env()
        assert env.n_restarts == 3
        assert len(env.graphs) == 2, "graphs must not accumulate across pairs"

    def test_the_cost_vector_is_gedlibs_constant_ordering(self, fake_gedlib: _InstallFake) -> None:
        fake_gedlib()
        backend = GedlibBackend(UNIT_COSTS)
        backend.pair(P4, C4)
        assert backend.env().edit_cost == ("CONSTANT", [1.0, 1.0, 0.0, 1.0, 1.0, 0.0])

    def test_the_graphedx_cost_vector_charges_nothing_for_nodes(
        self, fake_gedlib: _InstallFake
    ) -> None:
        fake_gedlib()
        backend = GedlibBackend(GRAPHEDX_COSTS)
        backend.pair(P4, C4)
        assert backend.env().edit_cost == ("CONSTANT", [0.0, 0.0, 0.0, 1.0, 1.0, 0.0])

    def test_init_uses_the_verified_option(self, fake_gedlib: _InstallFake) -> None:
        fake_gedlib()
        backend = GedlibBackend(UNIT_COSTS)
        backend.pair(P4, C4)
        assert backend.env().init_option == "EAGER_WITHOUT_SHUFFLED_COPIES"

    def test_string_attributes_are_attached(self, fake_gedlib: _InstallFake) -> None:
        """add_nx_graph asserts this itself; reaching the end means it held."""
        fake_gedlib()
        backend = GedlibBackend(UNIT_COSTS)
        backend.pair(P4, C4)
        added = backend.env().graphs
        assert all(isinstance(d["label"], str) for _, d in added[0].nodes(data=True))

    def test_init_method_follows_every_set_method(self, fake_gedlib: _InstallFake) -> None:
        fake_gedlib()
        backend = GedlibBackend(UNIT_COSTS)
        backend.pair(P4, C4)
        seq = [c[0] for c in backend.env().calls]
        for i, name in enumerate(seq):
            if name == "set_method":
                assert seq[i + 1] == "init_method"

    def test_threads_option_is_passed(self, fake_gedlib: _InstallFake) -> None:
        fake_gedlib()
        backend = GedlibBackend(UNIT_COSTS, threads=4)
        backend.pair(P4, C4)
        opts = {c[1][1] for c in backend.env().calls if c[0] == "set_method"}
        assert opts == {"--threads 4"}

    def test_no_retired_method_ever_reaches_set_method(self, fake_gedlib: _InstallFake) -> None:
        """The load-bearing assertion after amendment 2: no core-hour goes to ANCHOR.

        Replaces a test of ``exact_time_limit_s``, which existed only to bound
        ANCHOR_AWARE_GED's runtime. The constructor guard refuses the method by name;
        this checks the other end -- that a fully exercised pair never emits it, so a
        method reintroduced through some future code path is caught at the call site
        rather than at configuration.
        """
        fake_gedlib()
        backend = GedlibBackend(UNIT_COSTS)
        backend.pair(P4, C4)
        emitted = {c[1][0] for c in backend.env().calls if c[0] == "set_method"}
        assert emitted == {"BRANCH_FAST", "IPFP"}
        assert not emitted & set(RETIRED_METHODS) and not emitted & set(FORBIDDEN_METHODS)


class TestBackendFactory:
    """The picklable spec every process pool builds its backend from."""

    @pytest.mark.parametrize("kind", ["networkx", "stub"])
    def test_make_backend_returns_the_named_backend(self, kind: str) -> None:
        assert make_backend(kind).name == kind

    def test_an_unknown_backend_is_refused(self) -> None:
        with pytest.raises(GedBackendError, match="unknown backend"):
            make_backend("gurobi")

    def test_a_spec_survives_pickling(self) -> None:
        import pickle

        spec = BackendSpec(kind="stub", costs=EditCosts(), options={"seconds": 0.5})
        restored = pickle.loads(pickle.dumps(spec))
        assert restored.build().pair(P4, C4).seconds == 0.5

    def test_the_cost_model_reaches_the_backend(self) -> None:
        backend = make_backend("stub", GRAPHEDX_COSTS)
        assert backend.costs is GRAPHEDX_COSTS  # type: ignore[attr-defined]
        assert backend.pair(P4, C4).lb == 1.0  # nodes free, one edge apart


class TestToleranceIsShared:
    """Certification uses one tolerance, exported for the gates to reuse."""

    def test_tolerance_value(self) -> None:
        assert CERT_TOL == 1e-9


# --------------------------------------------------------------------------
# T-05: the options string is part of the method name
# --------------------------------------------------------------------------


_DET_START = (
    "--threads 1 --randomness PSEUDO --initialization-method BIPARTITE --initial-solutions 1"
)
_MULTI_START = "--threads 1 --randomness PSEUDO --initial-solutions 10"


def _roles_behaviour() -> dict[str, Any]:
    """A fake carrying every method the four CONTRACTS §3 roles name."""
    return {
        "values": {
            "BRANCH_FAST": {"lb": 1.0, "ub": 0.0},
            "BIPARTITE": {"lb": 0.0, "ub": 1.0},
            "BP_BEAM": {"lb": 0.0, "ub": 1.0},
            "IPFP": {"lb": 0.0, "ub": 1.0},
        }
    }


def _options_used(env: _FakeEnv, method: str) -> set[str]:
    """Every option string ``set_method`` received for one method."""
    return {c[1][1] for c in env.calls if c[0] == "set_method" and c[1][0] == method}


class TestRoleSpecs:
    """CONTRACTS §3 is transcribed, not paraphrased."""

    def test_the_four_roles_match_the_contract_verbatim(self) -> None:
        """A single character wrong here silently changes what the paper reports.

        T-27 measured GEDLIB's upper bounds moving on 91.5-93.6 % of pairs
        between runs at library defaults, so these strings are the difference
        between a reproducible number and a random one.
        """
        assert ROLE_SPECS["lb"] == ("BRANCH_FAST", "--threads 1", "lb")
        assert ROLE_SPECS["ub"] == ("BIPARTITE", "--threads 1", "ub")
        assert ROLE_SPECS["ubs"] == ("BP_BEAM", _DET_START, "ub")
        assert ROLE_SPECS["ubt"] == ("IPFP", _MULTI_START, "ub")


class TestPerEndOptions:
    """One option string per end, because the roles do not share one."""

    def test_the_two_ends_receive_their_own_strings(self, fake_gedlib: _InstallFake) -> None:
        """The defect this whole change exists to fix.

        Before, one ``--threads {n}`` string went to both ends. ``BP_BEAM_DET``
        and ``IPFP_MS`` do not share a string with anything, so the backend
        could not express the specification the paper prints.
        """
        fake_gedlib(_roles_behaviour())
        backend = GedlibBackend(
            UNIT_COSTS,
            lb_method="BRANCH_FAST",
            lb_options="--threads 1",
            ub_method="BP_BEAM",
            ub_options=_DET_START,
        )
        backend.bounds(P4, C4)
        env = backend.env()
        assert _options_used(env, "BRANCH_FAST") == {"--threads 1"}
        assert _options_used(env, "BP_BEAM") == {_DET_START}

    def test_the_default_is_exactly_what_t03_ran(self, fake_gedlib: _InstallFake) -> None:
        """Regression guard: the closed ticket's numbers depend on this default."""
        fake_gedlib()
        backend = GedlibBackend(UNIT_COSTS)
        backend.bounds(P4, C4)
        env = backend.env()
        assert _options_used(env, "BRANCH_FAST") == {"--threads 1"}
        assert _options_used(env, "IPFP") == {"--threads 1"}
        assert backend.lb_options == "--threads 1"
        assert backend.ub_options == "--threads 1"

    def test_threads_still_supplies_both_defaults(self, fake_gedlib: _InstallFake) -> None:
        fake_gedlib()
        backend = GedlibBackend(UNIT_COSTS, threads=4)
        assert backend.lb_options == "--threads 4"
        assert backend.ub_options == "--threads 4"

    def test_the_specification_records_the_options_verbatim(
        self, fake_gedlib: _InstallFake
    ) -> None:
        """A run whose metadata omits the options string is rejected at the gate."""
        fake_gedlib(_roles_behaviour())
        backend = GedlibBackend(UNIT_COSTS, ub_method="IPFP", ub_options=_MULTI_START, compute="ub")
        spec = backend.specification()
        assert spec["ub_method"] == "IPFP"
        assert spec["ub_options"] == _MULTI_START
        assert spec["ub_accessor"] == "upper"
        assert "lb_method" not in spec  # not computed, so not claimed


class TestComputeMode:
    """A single-role campaign pays for one end only."""

    def test_compute_lb_makes_no_upper_bound_call(self, fake_gedlib: _InstallFake) -> None:
        fake_gedlib(_roles_behaviour())
        backend = GedlibBackend(UNIT_COSTS, lb_method="BRANCH_FAST", compute="lb")
        lb, ub = backend.bounds(P4, C4)
        assert lb == 1.0
        assert ub == math.inf
        env = backend.env()
        assert not [c for c in env.calls if c[0] == "get_upper_bound"]
        assert {c[1][0] for c in env.calls if c[0] == "set_method"} == {"BRANCH_FAST"}

    def test_compute_ub_makes_no_lower_bound_call(self, fake_gedlib: _InstallFake) -> None:
        fake_gedlib(_roles_behaviour())
        backend = GedlibBackend(UNIT_COSTS, ub_method="BIPARTITE", compute="ub")
        lb, ub = backend.bounds(P4, C4)
        assert lb == -math.inf
        assert ub == 1.0
        env = backend.env()
        assert not [c for c in env.calls if c[0] == "get_lower_bound"]
        assert {c[1][0] for c in env.calls if c[0] == "set_method"} == {"BIPARTITE"}

    def test_a_one_sided_result_carries_its_sentinel(self, fake_gedlib: _InstallFake) -> None:
        fake_gedlib(_roles_behaviour())
        res = GedlibBackend(UNIT_COSTS, compute="lb", lb_method="BRANCH_FAST").pair(P4, C4)
        assert res.computed == "lb"
        assert res.ub == math.inf
        assert res.certified is False and res.exact is None

    def test_the_inverted_bracket_guard_is_skipped_when_one_sided(
        self, fake_gedlib: _InstallFake
    ) -> None:
        """With one end at infinity there is nothing to cross-check."""
        fake_gedlib({"values": {"BRANCH_FAST": {"lb": 99.0, "ub": 0.0}}})
        lb, ub = GedlibBackend(UNIT_COSTS, compute="lb", lb_method="BRANCH_FAST").bounds(P4, C4)
        assert (lb, ub) == (99.0, math.inf)

    def test_an_unknown_compute_mode_is_refused(self) -> None:
        with pytest.raises(GedBackendError, match="compute must be one of"):
            GedlibBackend(UNIT_COSTS, compute="upper")  # type: ignore[arg-type]


class TestOneSidedPairResult:
    """An unevaluated end must be unmistakable, never a plausible number."""

    def test_contract_b_still_has_exactly_seven_fields(self) -> None:
        """The compute mode is derived, not stored.

        ``ged_gates`` builds its payload by iterating these slots, so adding an
        eighth would silently change what the gates are required to emit.
        """
        assert PairResult.__slots__ == (
            "lb",
            "ub",
            "exact",
            "certified",
            "seconds",
            "timed_out",
            "method",
        )

    def test_only_the_vacuous_sentinel_is_admitted_on_each_side(self) -> None:
        """+inf above and -inf below; every other non-finite value is still refused."""
        assert PairResult(1.0, math.inf, None, False, 0.1, False, "m").computed == "lb"
        assert PairResult(-math.inf, 1.0, None, False, 0.1, False, "m").computed == "ub"
        assert PairResult(1.0, 5.0, None, False, 0.1, False, "m").computed == "both"
        with pytest.raises(GedBackendError, match="lb must be finite"):
            PairResult(math.inf, math.inf, None, False, 0.1, False, "m")
        with pytest.raises(GedBackendError, match="ub must be finite"):
            PairResult(1.0, -math.inf, None, False, 0.1, False, "m")
        with pytest.raises(GedBackendError, match="lb must be finite"):
            PairResult(math.nan, 1.0, None, False, 0.1, False, "m")

    def test_a_result_that_measured_nothing_is_refused(self) -> None:
        with pytest.raises(GedBackendError, match="measured nothing"):
            PairResult(-math.inf, math.inf, None, False, 0.1, False, "m")


class TestLazyZeroGuard:
    """CONTRACTS §6.1: identical behaviour, evaluated only when it is needed."""

    def test_the_predicate_is_not_called_when_no_read_returns_zero(
        self, fake_gedlib: _InstallFake, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point: 21.7 M pairs must not each pay for an isomorphism test."""
        fake_gedlib(_roles_behaviour())
        calls: list[int] = []
        monkeypatch.setattr(
            ged_backends_module,
            "zero_distance_is_attainable",
            lambda *a, **k: calls.append(1) or True,
        )
        backend = GedlibBackend(UNIT_COSTS, lb_method="BRANCH_FAST", ub_method="BIPARTITE")
        lb, ub = backend.bounds(P4, C4)
        assert (lb, ub) == (1.0, 1.0)
        assert calls == []

    def test_the_predicate_is_called_once_when_a_read_returns_zero(
        self, fake_gedlib: _InstallFake, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both upper-bound orientations read 0.00, and one decision covers both."""
        fake_gedlib(
            {
                "values": {
                    "BRANCH_FAST": {"lb": 0.0, "ub": 0.0},
                    "BIPARTITE": {"lb": 0.0, "ub": 0.0},
                }
            }
        )
        calls: list[int] = []
        monkeypatch.setattr(
            ged_backends_module,
            "zero_distance_is_attainable",
            lambda *a, **k: calls.append(1) or True,
        )
        backend = GedlibBackend(UNIT_COSTS, lb_method="BRANCH_FAST", ub_method="BIPARTITE")
        assert backend.bounds(P4, P4) == (0.0, 0.0)
        assert len(calls) == 1

    def test_an_illegal_zero_upper_bound_is_still_rejected(self, fake_gedlib: _InstallFake) -> None:
        """Deferring the decision must not weaken it."""
        fake_gedlib(
            {
                "values": {
                    "BRANCH_FAST": {"lb": 0.0, "ub": 0.0},
                    "BIPARTITE": {"lb": 0.0, "ub": 0.0},
                }
            }
        )
        backend = GedlibBackend(UNIT_COSTS, lb_method="BRANCH_FAST", ub_method="BIPARTITE")
        with pytest.raises(GedBackendError, match="wrong accessor"):
            backend.bounds(P4, C4)  # not isomorphic, so 0.00 is impossible

    def test_the_deferred_value_equals_the_eager_one(self) -> None:
        """A pure performance change computes the same answer."""
        for g1, g2 in ((P4, C4), (P4, P4), (nx.star_graph(5), nx.cycle_graph(6))):
            assert zero_distance_is_attainable(g1, g2, UNIT_COSTS) == (
                g1.number_of_nodes() == g2.number_of_nodes()
                and g1.number_of_edges() == g2.number_of_edges()
                and nx.is_isomorphic(g1, g2)
            )


class TestAccessorProbe:
    """The check that catches a wrong accessor before 21.7 M pairs of zeros."""

    def test_the_probe_passes_on_a_correctly_configured_backend(
        self, fake_gedlib: _InstallFake
    ) -> None:
        fake_gedlib(_roles_behaviour())
        backend = GedlibBackend(UNIT_COSTS, lb_method="BRANCH_FAST", ub_method="BIPARTITE")
        assert backend.probe_accessors() == {"lb": 1.0, "ub": 1.0}

    def test_a_method_read_through_the_wrong_accessor_returns_zero_and_the_probe_fires(
        self, fake_gedlib: _InstallFake
    ) -> None:
        """GEDLIB's silent failure, reproduced exactly.

        ``get_upper_bound()`` on a lower-bound method returns ``0.00`` and
        raises nothing. The constructor cannot catch this case because the
        method table is a *measured* capability, not a static one, so the probe
        is the check that stands between it and a matrix of zeros.
        """
        fake_gedlib({"values": {"BRANCH_FAST": {"lb": 1.0, "ub": 0.0}}})
        backend = GedlibBackend(UNIT_COSTS, lb_method="BRANCH_FAST", ub_method="BIPARTITE")
        # Force the misconfiguration past the static table the way a library
        # change would: the method stays legal, its upper accessor goes dead.
        object.__setattr__(backend, "_ub_method", "BRANCH_FAST")
        with pytest.raises(GedBackendError, match="accessor probe failed"):
            backend.probe_accessors()

    def test_the_static_table_refuses_the_mismatch_at_construction(self) -> None:
        """The first line of defence, before the probe is even reached."""
        with pytest.raises(GedBackendError, match="upper bound"):
            GedlibBackend(UNIT_COSTS, ub_method="BRANCH_FAST")

    def test_an_infinite_read_fires_the_probe(self, fake_gedlib: _InstallFake) -> None:
        """HED's failure mode: get_upper_bound() returns inf, silently."""
        fake_gedlib(
            {"values": {"BRANCH_FAST": {"lb": 1.0, "ub": 0.0}, "BIPARTITE": {"ub": math.inf}}}
        )
        backend = GedlibBackend(UNIT_COSTS, ub_method="BIPARTITE", compute="ub")
        with pytest.raises(GedBackendError, match="accessor probe failed"):
            backend.probe_accessors()
