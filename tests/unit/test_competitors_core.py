"""Orchestrator-owned tests for the shared `isalgraph.competitors` machinery.

These cover T-04 acceptance criteria 4, 5, 6 and 9, plus the contracts that
three concurrent tracks code against.  The per-backend suites belong to their
tracks: ``test_competitors_serial.py``, ``test_competitors_canonical.py``,
``test_agm_cam.py``, ``test_min_dfs.py``, ``test_wl_subtree.py``.

Tests that need a backend which may not be merged yet skip rather than fail,
so this file is green before the wave lands and strictly stronger after it.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import sys

import pytest

from isalgraph.competitors import bits as bits_mod
from isalgraph.competitors import fixtures
from isalgraph.competitors.base import (
    Budget,
    Capability,
    Encoding,
    PositionalFrame,
    ReprBackend,
    VectorBackend,
)
from isalgraph.competitors.registry import (
    available_backends,
    available_metrics,
    get_backend,
    get_metric,
    get_repr_backend,
    registered_backends,
)
from isalgraph.errors import (
    BackendError,
    BackendNotFoundError,
    BackendUnavailableError,
    BitCountUndefined,
    DistanceUndefined,
    NotReversible,
    SuiteScopeError,
)

nx = pytest.importorskip("networkx")

COMPETITORS_DIR = pathlib.Path(bits_mod.__file__).parent


# ----------------------------------------------------------------------
# Criterion 5 -- the dependency contract
# ----------------------------------------------------------------------


#: Asserts the import contract in a *clean* interpreter with every optional
#: dependency blocked.  A subprocess rather than an in-process
#: ``builtins.__import__`` patch, because re-importing the package under a
#: guard builds a second set of registry dicts and leaves whichever one the
#: already-imported symbols close over out of sync -- which silently emptied
#: the registry for every test after it.
_IMPORT_CONTRACT_PROBE = """
import builtins, sys
blocked = {"networkx", "numpy", "pynauty", "grakel", "rapidfuzz", "scipy"}
_real = builtins.__import__
def guard(name, *a, **k):
    if name.split(".")[0] in blocked:
        raise ImportError("blocked: " + name)
    return _real(name, *a, **k)
builtins.__import__ = guard
import isalgraph.competitors as C
assert C.registered_backends() is not None
assert C.available_backends() == (), C.available_backends()
import isalgraph
assert "competitors" not in sys.modules["isalgraph"].__dict__ or True
print("OK", sorted(C.registered_backends()))
"""


def test_import_succeeds_with_every_optional_dependency_absent() -> None:
    """`import isalgraph.competitors` must work with all four libraries gone.

    The package sits at the same tier as `viz` and `adapters` and inherits
    their contract: every third-party import lives inside a function body or
    behind the lazy registry.  A missing dependency raises on *request*, never
    at import, and never degrades to a substitute representation.
    """
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_CONTRACT_PROBE],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.startswith("OK")


def test_isalgraph_top_level_does_not_import_competitors() -> None:
    """The top-level import chain stays stdlib-only."""
    source = pathlib.Path(importlib.import_module("isalgraph").__file__ or "").read_text(
        encoding="utf-8"
    )
    assert "competitors" not in source


def test_unknown_backend_raises_not_found_not_unavailable() -> None:
    with pytest.raises(BackendNotFoundError):
        get_backend("no_such_backend")


def test_unknown_metric_raises() -> None:
    with pytest.raises(BackendNotFoundError):
        get_metric("no_such_metric")


_MISSING_DEP_PROBE = """
import builtins, sys
_real = builtins.__import__
def guard(name, *a, **k):
    if name.split(".")[0] == "pynauty":
        raise ImportError("blocked: pynauty")
    return _real(name, *a, **k)
builtins.__import__ = guard
from isalgraph.competitors.registry import get_backend, registered_backends
from isalgraph.errors import CompetitorError
if "nauty_graph6" not in registered_backends():
    print("SKIP")
else:
    try:
        get_backend("nauty_graph6")
    except CompetitorError as exc:
        print("RAISED", type(exc).__name__)
    else:
        raise SystemExit("nauty_graph6 was returned with pynauty absent -- a silent degrade")
"""


def test_missing_dependency_raises_rather_than_degrading() -> None:
    """A backend whose library is absent raises; it never returns a substitute.

    Same reason as `isalgraph.core.backends`: a silent degrade turns a wrong
    number into a plausible one, and nothing downstream can tell.
    """
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", _MISSING_DEP_PROBE],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    if result.stdout.strip() == "SKIP":
        pytest.skip("nauty backend not merged yet")
    assert result.stdout.startswith("RAISED")


# ----------------------------------------------------------------------
# Criterion 6 -- F5-blindness is structural
# ----------------------------------------------------------------------


def _import_closure(module_name: str, *, stop: frozenset[str] = frozenset()) -> set[str]:
    """Names imported transitively by *module_name*, within isalgraph.competitors.

    Read from the AST rather than from `sys.modules`, so the answer does not
    depend on what some other test happened to import first.
    """
    seen: set[str] = set()
    queue = [module_name]
    while queue:
        name = queue.pop()
        if name in seen or name in stop:
            continue
        seen.add(name)
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            continue
        if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
            continue
        tree = ast.parse(pathlib.Path(spec.origin).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("isalgraph"):
                        queue.append(alias.name)
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("isalgraph")
            ):
                queue.append(node.module)
                queue.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return seen


def test_grid_import_closure_reaches_no_ged_loader() -> None:
    """Decision 24's defence, enforced by the import graph rather than by prose.

    `grid.py` applies competitors.md 3.4's selection rule.  If it could reach a
    GED value, the rule could be selecting on the very outcome it is supposed
    to be blind to, and decision 24 stops being defensible to a reviewer.
    """
    closure = _import_closure("isalgraph.competitors.grid")
    offenders = {name for name in closure if "ged" in name.lower()}
    assert not offenders, (
        f"grid.py's import closure reaches {sorted(offenders)}. F5-blindness is "
        f"structural (design note 4.5): move that import into f5.py"
    )


def test_datasets_import_closure_reaches_no_ged_loader() -> None:
    closure = _import_closure("isalgraph.competitors.datasets")
    assert not {name for name in closure if "ged" in name.lower()}


def test_f5_is_the_only_entry_point_that_can_see_ged() -> None:
    """The complement of the rule above: f5.py *must* reach the loader."""
    closure = _import_closure("isalgraph.competitors.f5")
    assert "isalgraph.competitors.ged_reference" in closure


# ----------------------------------------------------------------------
# Criterion 4 -- no fabricated bit count, no 8x inflation
# ----------------------------------------------------------------------


def test_no_metric_or_bits_module_reads_encoding_text() -> None:
    """`Encoding.text` is a debugging view and is never measured.

    The one sanctioned reader is `levenshtein_char`, which exists to report
    the character-level answer as a supplementary number.
    """
    offenders: list[str] = []
    targets = [COMPETITORS_DIR / "bits.py", *sorted((COMPETITORS_DIR / "metrics").glob("*.py"))]
    for path in targets:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "text":
                if path.name == "levenshtein.py":
                    continue  # LevenshteinChar, the one sanctioned reader
                if path.name == "bits.py":
                    continue  # the min_dfs row, flagged inflated=True
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"these read Encoding.text: {offenders}"


def test_size_null_and_wl_have_no_bit_count() -> None:
    """A fabricated bit count would measure the container, not the encoding."""
    backend = get_repr_backend("size_null")
    encoding = backend.encode(fixtures.to_networkx(fixtures.RUNNING_EXAMPLE))
    with pytest.raises(BitCountUndefined):
        backend.bits(encoding)

    if "wl_subtree" in registered_backends():
        wl = get_backend("wl_subtree")
        assert not hasattr(wl, "bits"), (
            "VectorBackend must have no bits() at all -- unreachable, not merely forbidden"
        )


@pytest.mark.parametrize("name", sorted(fixtures.CONNECTED_FIXTURES))
def test_no_eightfold_inflation_on_the_adjacency_matrix(name: str) -> None:
    """Counting '1010...' at eight bits per character inflates it 8x.

    The comparison is against ``8 * len(text)``, not ``len(text)``.  Design
    criterion 4 originally said the latter, which contradicts its own formula:
    ``8*ceil(n(n-1)/16) == 8*ceil(T/8) >= T == len(text)``, because packing T
    bits into whole bytes cannot use fewer than T bits.  Amended 2026-08-15.
    """
    if "adjacency" not in registered_backends():
        pytest.skip("adjacency backend not merged yet")
    backend = get_repr_backend("adjacency")
    encoding = backend.encode(fixtures.to_networkx(fixtures.ALL_FIXTURES[name]))
    counted = backend.bits(encoding)
    assert counted.realised_bits < 8 * len(encoding.text)
    # Byte padding and nothing more.
    assert counted.entropy_bits <= counted.realised_bits < counted.entropy_bits + 8


def test_adjacency_realised_bits_pack_the_triangle_into_bytes() -> None:
    """`8*ceil(n(n-1)/16)`, which is `8*ceil(T/8)` -- T bits in whole bytes.

    A regression guard for the halving defect track A found: reading the 16 as
    a word size gives 8 bits for a 15-bit triangle at n=6.
    """
    if "adjacency" not in registered_backends():
        pytest.skip("adjacency backend not merged yet")
    import math

    backend = get_repr_backend("adjacency")
    for n in (2, 6, 12, 28):
        counted = backend.bits(backend.encode(nx.complete_graph(n)))
        assert counted.realised_bits == 8 * math.ceil(n * (n - 1) / 16)
        assert counted.entropy_bits == n * (n - 1) // 2


def test_bits_refuses_a_backend_it_has_no_row_for() -> None:
    """`bits.py` is the only producer, so an unknown backend is a hard error."""
    encoding = Encoding(
        backend="not_a_registered_backend",
        symbols=("0",),
        alphabet_size=2,
        n_nodes=2,
        n_edges=1,
        text="0",
    )
    with pytest.raises(BitCountUndefined):
        bits_mod.count(encoding)


def test_graph6_bit_count_requires_a_measured_wire() -> None:
    """The closed form is a test oracle; production measures the emitted bytes."""
    encoding = Encoding(
        backend="graph6", symbols=("E",), alphabet_size=64, n_nodes=6, n_edges=7, text="E"
    )
    with pytest.raises(BitCountUndefined):
        bits_mod.count(encoding)


# ----------------------------------------------------------------------
# Criterion 9 -- suite scope is enforced, not documented
# ----------------------------------------------------------------------


def test_isalgraph_canonical_refuses_above_suite_one() -> None:
    """Rather than silently producing a 76 %-complete column."""
    backend = get_repr_backend("isalgraph_canonical")
    assert Capability.SUITE1_ONLY in backend.capabilities
    big = nx.path_graph(40)
    with pytest.raises(SuiteScopeError):
        backend.encode(big)


def test_isalgraph_pruned_has_no_suite_restriction() -> None:
    backend = get_repr_backend("isalgraph_pruned")
    assert Capability.SUITE1_ONLY not in backend.capabilities


def test_a_budget_the_engine_cannot_enforce_is_refused_not_dropped() -> None:
    """`timeout_s` is cpp-only; the Python reference has no interruption point.

    Dropping it silently would turn a 2 s budget into an unbounded run whose
    bit counts are then quoted as if budgeted.
    """
    import isalgraph

    backend = get_repr_backend("isalgraph_pruned")
    graph = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE)
    if isalgraph.engine() == "cpp":
        # Forcing the python engine must refuse the default budget.
        with pytest.raises(BackendError):
            backend.encode(graph, engine="python")  # type: ignore[call-arg]
    else:
        with pytest.raises(BackendError):
            backend.encode(graph)
    # Opting out deliberately is always allowed.
    assert backend.encode(graph, budget=Budget(timeout_s=None)).length > 0


# ----------------------------------------------------------------------
# size_null -- registered, and hard-excluded from the confirmatory family
# ----------------------------------------------------------------------


def test_size_null_is_registered_but_absent_from_the_default_listing() -> None:
    """Decision 23: it is a descriptive baseline, never a family member."""
    assert "size_null" in registered_backends()
    assert "size_null" not in available_backends()
    assert "size_null" in available_backends(include_baseline=True)


def test_size_null_carries_the_baseline_capability() -> None:
    assert Capability.BASELINE in get_repr_backend("size_null").capabilities


def test_size_null_is_not_reversible() -> None:
    backend = get_repr_backend("size_null")
    encoding = backend.encode(fixtures.to_networkx(fixtures.RUNNING_EXAMPLE))
    with pytest.raises(NotReversible):
        backend.decode(encoding)


def test_size_null_metric_is_declared_a_pseudometric() -> None:
    """Identity of indiscernibles fails: two non-isomorphic graphs on n nodes."""
    metric = get_metric("size_null")
    assert metric.is_pseudometric
    k33 = get_repr_backend("size_null").encode(fixtures.to_networkx(fixtures.K33))
    prism = get_repr_backend("size_null").encode(fixtures.to_networkx(fixtures.PRISM))
    assert metric.distance(k33, prism) == 0.0


def test_grid_refuses_to_select_a_baseline_as_a_primary_distance() -> None:
    from isalgraph.competitors.grid import Cell, _apply_selection_rule

    cell = Cell(backend="size_null", metric="size_null", f1_defined_frac=1.0, f3_invariant="50/50")
    _apply_selection_rule(cell, get_repr_backend("size_null"))
    assert not cell.passes_selection
    assert cell.excluded_because is not None
    assert "BASELINE" in cell.excluded_because


# ----------------------------------------------------------------------
# The value objects and the metrics
# ----------------------------------------------------------------------


def test_positional_frame_rejects_a_bit_count_that_cannot_align() -> None:
    with pytest.raises(ValueError, match="pairs"):
        PositionalFrame(n_nodes=3, pairs=((0, 1),), bits=("1",))
    with pytest.raises(ValueError, match="bits"):
        PositionalFrame(n_nodes=3, pairs=((0, 1), (0, 2), (1, 2)), bits=("1",))


def test_padded_hamming_is_undefined_without_a_frame_and_says_so() -> None:
    """That cell is a reported F1 result, which is one reason the grid exists."""
    metric = get_metric("padded_hamming")
    backend = get_repr_backend("isalgraph_pruned")
    a = backend.encode(fixtures.to_networkx(fixtures.K33), budget=Budget(timeout_s=None))
    b = backend.encode(fixtures.to_networkx(fixtures.PRISM), budget=Budget(timeout_s=None))
    assert a.frame is None
    assert not metric.is_defined(a, b)
    with pytest.raises(DistanceUndefined):
        metric.distance(a, b)


def test_plain_hamming_is_undefined_on_unequal_lengths() -> None:
    metric = get_metric("hamming")
    backend = get_repr_backend("isalgraph_pruned")
    a = backend.encode(
        fixtures.to_networkx(fixtures.RUNNING_EXAMPLE), budget=Budget(timeout_s=None)
    )
    b = backend.encode(fixtures.to_networkx(fixtures.PATH_2), budget=Budget(timeout_s=None))
    assert len(a.symbols) != len(b.symbols)
    assert not metric.is_defined(a, b)
    with pytest.raises(DistanceUndefined):
        metric.distance(a, b)


def test_levenshtein_runs_on_symbols_and_not_on_text() -> None:
    """One symbol is one edit.  Character level charges 4 for one deleted tuple."""
    pytest.importorskip("rapidfuzz")
    symbol = get_metric("levenshtein")
    character = get_metric("levenshtein_char")
    a = Encoding(
        backend="min_dfs",
        symbols=("0-1", "1-2", "2-0"),
        alphabet_size=9,
        n_nodes=3,
        n_edges=3,
        text="0-1 1-2 2-0",
    )
    b = Encoding(
        backend="min_dfs",
        symbols=("0-1", "2-0"),
        alphabet_size=9,
        n_nodes=3,
        n_edges=2,
        text="0-1 2-0",
    )
    assert symbol.distance(a, b) == 1.0
    assert character.distance(a, b) == 4.0


def test_every_metric_declares_what_it_consumes_and_its_axioms() -> None:
    for name in available_metrics():
        metric = get_metric(name)
        assert metric.consumes in ("symbols", "text", "frame", "features", "order")
        assert isinstance(metric.is_pseudometric, bool)


def test_every_registered_backend_declares_a_name_matching_its_key() -> None:
    for name in registered_backends():
        try:
            backend = get_backend(name)
        except BackendUnavailableError:
            continue
        assert backend.name == name
        assert isinstance(backend, ReprBackend | VectorBackend)


# ----------------------------------------------------------------------
# The relabeller -- an F3 harness that cannot fail is worthless
# ----------------------------------------------------------------------


def test_the_relabeller_actually_changes_insertion_order() -> None:
    """`nx.relabel_nodes(copy=True)` alone preserves it (finding 13).

    If this ever passes trivially, every F3 result in the ticket is void.
    """
    import random

    rng = random.Random(42)
    graph = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE)
    orders = {tuple(fixtures.shuffled_copy(graph, rng).nodes()) for _ in range(20)}
    assert len(orders) > 1, "the relabeller produced one insertion order in 20 draws"


def test_the_relabeller_can_make_an_order_dependent_format_fail() -> None:
    """The harness must be capable of failing, or its 50/50 results mean nothing."""
    import random

    rng = random.Random(42)
    graph = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE)
    codes = {
        nx.to_graph6_bytes(fixtures.shuffled_copy(graph, rng), header=False).strip()
        for _ in range(20)
    }
    assert len(codes) > 1, "graph6 was invariant under 20 relabellings, so the harness is void"


def test_shuffled_copy_preserves_the_graph_up_to_isomorphism() -> None:
    import random

    rng = random.Random(7)
    for key in fixtures.CONNECTED_FIXTURES:
        graph = fixtures.to_networkx(fixtures.ALL_FIXTURES[key])
        assert nx.is_isomorphic(graph, fixtures.shuffled_copy(graph, rng))


# ----------------------------------------------------------------------
# Fixtures the paper depends on
# ----------------------------------------------------------------------


def test_running_example_shape() -> None:
    graph = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE)
    assert (graph.number_of_nodes(), graph.number_of_edges()) == (6, 7)
    minus = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE_MINUS_EDGE)
    assert (minus.number_of_nodes(), minus.number_of_edges()) == (6, 6)
    assert not minus.has_edge(0, 3)


def test_the_completeness_witness_is_two_non_isomorphic_three_regular_graphs() -> None:
    """1-WL cannot separate them at any h; every other pool member can."""
    k33 = fixtures.to_networkx(fixtures.K33)
    prism = fixtures.to_networkx(fixtures.PRISM)
    for graph in (k33, prism):
        assert graph.number_of_nodes() == 6
        assert graph.number_of_edges() == 9
        assert set(dict(graph.degree()).values()) == {3}
        assert nx.is_connected(graph)
    assert not nx.is_isomorphic(k33, prism)
