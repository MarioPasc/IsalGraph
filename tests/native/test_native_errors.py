"""Exception parity between the two engines, and the errors.py contract.

Two distinct claims are tested here.

1. PARITY (must hold now).  On every failure path both backends raise the
   same class with the same message text, so a caller matching on either is
   backend-blind.

2. BUILTIN MIXINS (lands at integration).  `main` owns errors.py and will add
   the builtin bases -- DisconnectedGraphError(+ValueError),
   CapacityError(+RuntimeError), InvalidNodeError(+IndexError), and so on.
   Those assertions are skipped until the change lands rather than xfailed,
   because this is an unlanded dependency and not a known defect: the skips
   turn into passes the moment errors.py is updated, with no edit here.
"""

from __future__ import annotations

import pytest

pytest.importorskip("isalgraph.core._native", reason="C++ extension not built")

import graphs as G

from isalgraph import errors
from isalgraph.core import backends
from isalgraph.core.canonical import canonical_string as ref_canonical
from isalgraph.core.canonical_pruned import pruned_canonical_string as ref_pruned
from isalgraph.core.graph_to_string import GraphToString
from isalgraph.core.string_to_graph import StringToGraph

_MIXINS_LANDED = issubclass(errors.CapacityError, RuntimeError)
_needs_mixins = pytest.mark.skipif(
    not _MIXINS_LANDED,
    reason="errors.py builtin mixins land at wave integration; main owns errors.py",
)


def _disconnected(n: int = 4) -> object:
    """Two components: no starting node reaches everything."""
    return G.build(n, [(0, 1), (2, 3)])


def _directed_unreachable() -> object:
    """A directed graph where no node is the root of a spanning out-tree."""
    return G.build(3, [(1, 0), (2, 0)], directed=True)


# ----------------------------------------------------------------------
# 1. Parity
# ----------------------------------------------------------------------


@pytest.mark.parametrize("factory", [_disconnected, _directed_unreachable])
def test_disconnected_graph_parity(factory) -> None:
    g = factory()

    with pytest.raises(Exception) as ref_exc:  # noqa: B017 - type is the assertion
        ref_canonical(g)
    with pytest.raises(errors.DisconnectedGraphError) as cpp_exc:
        backends.canonical_string(g, backend="cpp")
    with pytest.raises(errors.DisconnectedGraphError) as py_exc:
        backends.canonical_string(g, backend="python")

    assert str(cpp_exc.value) == str(ref_exc.value)
    assert str(py_exc.value) == str(ref_exc.value)
    assert type(cpp_exc.value) is type(py_exc.value)


@pytest.mark.parametrize("factory", [_disconnected, _directed_unreachable])
def test_disconnected_graph_parity_pruned(factory) -> None:
    g = factory()
    with pytest.raises(Exception) as ref_exc:  # noqa: B017
        ref_pruned(g)
    with pytest.raises(errors.DisconnectedGraphError) as cpp_exc:
        backends.pruned_canonical_string(g, backend="cpp")
    with pytest.raises(errors.DisconnectedGraphError) as py_exc:
        backends.pruned_canonical_string(g, backend="python")
    assert str(cpp_exc.value) == str(ref_exc.value) == str(py_exc.value)


def test_graph_to_string_unreachable_parity() -> None:
    """The reference message embeds a Python set repr, which is why the
    reachability check runs Python-side for the cpp path too."""
    g = _disconnected()
    with pytest.raises(ValueError) as ref_exc:
        GraphToString(g).run(0)
    with pytest.raises(errors.DisconnectedGraphError) as cpp_exc:
        backends.graph_to_string(g, 0, backend="cpp")
    with pytest.raises(errors.DisconnectedGraphError) as py_exc:
        backends.graph_to_string(g, 0, backend="python")
    assert str(cpp_exc.value) == str(ref_exc.value)
    assert str(py_exc.value) == str(ref_exc.value)
    assert "Unreachable nodes:" in str(cpp_exc.value)


@pytest.mark.parametrize("bad_node", [-1, 4, 99])
def test_initial_node_out_of_range_parity(bad_node: int) -> None:
    g = G.path(4)
    with pytest.raises(ValueError) as ref_exc:
        GraphToString(g).run(bad_node)
    with pytest.raises(ValueError) as cpp_exc:
        backends.graph_to_string(g, bad_node, backend="cpp")
    with pytest.raises(ValueError) as py_exc:
        backends.graph_to_string(g, bad_node, backend="python")
    assert str(cpp_exc.value) == str(ref_exc.value) == str(py_exc.value)
    assert str(cpp_exc.value) == "Initial node out of range"


@pytest.mark.parametrize("bad", ["X", "VVQ", "hello", "V V", "NnPpVvCcWZ", "é"])
def test_invalid_instruction_parity(bad: str) -> None:
    with pytest.raises(ValueError) as ref_exc:
        StringToGraph(bad, False)
    with pytest.raises(errors.InvalidStringError) as cpp_exc:
        backends.string_to_graph(bad, False, backend="cpp")
    with pytest.raises(errors.InvalidStringError) as py_exc:
        backends.string_to_graph(bad, False, backend="python")
    assert str(cpp_exc.value) == str(ref_exc.value) == str(py_exc.value)
    assert repr(bad) in str(cpp_exc.value)


def test_capacity_overflow_parity() -> None:
    from isalgraph.core import _native as ext
    from isalgraph.core.sparse_graph import SparseGraph

    py = SparseGraph(2, False)
    py.add_node()
    py.add_node()
    with pytest.raises(RuntimeError) as ref_exc:
        py.add_node()

    cpp = ext.Cdll(1)
    cpp.insert_after(-1, 0)
    with pytest.raises(errors.CapacityError) as cpp_exc:
        cpp.insert_after(0, 1)

    assert str(ref_exc.value) == "Maximum number of nodes reached: 2"
    assert str(cpp_exc.value) == "CircularDoublyLinkedList is full"


# ----------------------------------------------------------------------
# Timeout
# ----------------------------------------------------------------------


def test_timeout_raises_the_mirrored_class() -> None:
    """A budget small enough that even a modest search cannot finish."""
    g = G.complete(9)
    with pytest.raises(errors.CanonicalizationTimeoutError):
        backends.canonical_string(g, timeout_s=1e-6, backend="cpp")


def test_generous_timeout_returns_the_same_answer() -> None:
    g = G.cycle(6)
    unbounded = backends.canonical_string(g, backend="cpp")
    bounded = backends.canonical_string(g, timeout_s=120.0, backend="cpp")
    assert bounded == unbounded == ref_canonical(g)


def test_python_backend_refuses_a_budget_it_cannot_honour() -> None:
    """Silently ignoring timeout_s would let a harness believe a run was
    bounded when it was not."""
    g = G.path(4)
    with pytest.raises(errors.BackendError, match="only supported by the 'cpp' backend"):
        backends.canonical_string(g, timeout_s=1.0, backend="python")


# ----------------------------------------------------------------------
# 2. errors.py builtin-mixin contract (lands at integration)
# ----------------------------------------------------------------------


@_needs_mixins
@pytest.mark.parametrize(
    ("name", "builtin"),
    [
        ("DisconnectedGraphError", ValueError),
        ("CanonicalizationTimeoutError", RuntimeError),
        ("EncodingStuckError", RuntimeError),
        ("CapacityError", RuntimeError),
        ("InvalidNodeError", IndexError),
        ("InvalidStringError", ValueError),
        ("BackendError", RuntimeError),
    ],
)
def test_error_class_carries_its_builtin_mixin(name: str, builtin: type) -> None:
    cls = getattr(errors, name)
    assert issubclass(cls, builtin), f"{name} must also be a {builtin.__name__}"
    assert issubclass(cls, errors.IsalGraphError)


@_needs_mixins
def test_invalid_node_error_is_not_a_value_error() -> None:
    """sparse_graph.py's bounds checks raise IndexError and six tests pin it."""
    assert not issubclass(errors.InvalidNodeError, ValueError)


@_needs_mixins
def test_encoding_error_base_carries_no_builtin() -> None:
    """A mixin on the shared base would make the RuntimeError leaves lie."""
    assert not issubclass(errors.EncodingError, ValueError)
    assert not issubclass(errors.EncodingError, RuntimeError)


@_needs_mixins
def test_legacy_value_error_contract_holds_end_to_end() -> None:
    """The two baseline tests this whole contract exists to keep green."""
    g = _disconnected()
    with pytest.raises(ValueError, match="No starting node"):
        backends.canonical_string(g, backend="cpp")
    with pytest.raises(ValueError, match="No starting node"):
        backends.pruned_canonical_string(g, backend="cpp")
