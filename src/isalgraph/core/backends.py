"""Engine-selection layer for ``isalgraph.core``.

Exposes the active computation engine (C++ extension or pure-Python
fallback), a build-info dict, a generic dispatcher, and one dispatching free
function per user-facing algorithm.

Engine resolution order
-----------------------
1. An explicit ``backend=`` keyword argument.  This always wins.
2. The ``ISALGRAPH_ENGINE`` environment variable (``"cpp"`` | ``"python"``).
3. :data:`DEFAULT_BACKEND`: ``"cpp"`` when the native extension imported,
   ``"python"`` otherwise.

The kwarg-beats-env ordering is not incidental.  IsalSR recorded a real bug
where the environment variable was consulted before the keyword argument was
honoured, so ``ISALSR_ENGINE=python`` reported ``"python"`` from ``engine()``
while still dispatching every call to C++.

Naming law
----------
``_python_<fn>`` is the reference path, ``_cpp_<fn>`` is a thin marshalling
wrapper, ``<fn>`` is the dispatcher.

Marshalling contract
--------------------
:func:`_marshal` hands each adjacency across the FFI as a ``list`` built by
``list(graph.neighbors(u))`` -- that is, in CPython's own set-iteration
order.  This is load-bearing.  ``GraphToString._find_new_neighbor`` returns
*the first* uninserted neighbour of a Python ``set[int]``, and CPython
iterates small-int sets in slot order ``i & (table_size - 1)``, not ascending
value order: ``{2, 9}`` with table size 8 yields ``9, 2``.  A C++
``std::set<int32_t>`` would yield ``2, 9`` and produce a different -- equally
valid, but different -- greedy string.  Preserving the order at the boundary
makes byte-exact greedy parity achievable at zero algorithmic cost.

Exception contract
------------------
Both paths raise the classes in :mod:`isalgraph.errors`, with identical
message text, so the differential suite can assert type *and* message parity.
The Python reference itself is frozen and still raises bare ``ValueError`` /
``RuntimeError``; the ``_python_*`` wrappers translate.
"""

from __future__ import annotations

import logging
import os
from typing import Literal, TypeVar

from isalgraph import errors
from isalgraph.core.sparse_graph import SparseGraph
from isalgraph.types import VALID_INSTRUCTIONS

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import the C++ extension -- graceful fallback to Python.
# ---------------------------------------------------------------------------
try:
    from isalgraph.core import _native as _cpp_ext  # type: ignore[attr-defined]

    _CPP_AVAILABLE: bool = True
    log.debug("isalgraph.core._native loaded (C++ engine active)")
except ImportError as _err:  # pragma: no cover - exercised by the fallback test
    _cpp_ext = None  # type: ignore[assignment]
    _CPP_AVAILABLE = False
    log.debug("isalgraph.core._native not available (%s); using Python engine", _err)

Backend = Literal["cpp", "python"]
"""Type alias for the ``backend=`` / ``ISALGRAPH_ENGINE`` values."""

DEFAULT_BACKEND: Backend = "cpp" if _CPP_AVAILABLE else "python"
"""Active default backend: ``"cpp"`` when the extension imported, else ``"python"``."""

C = TypeVar("C")

# ``EncodingStuckError`` is landed in errors.py by the integrating agent.
# Until then the historical bare ``RuntimeError`` is the correct stand-in --
# it is exactly what the frozen reference raises at those call sites, and the
# class will be a RuntimeError subclass, so this resolves identically either way.
_EncodingStuckError: type[Exception] = getattr(errors, "EncodingStuckError", RuntimeError)

_DISCONNECTED_PREFIX = "No starting node"
_STUCK_MARKER = "no valid operation found"


# ---------------------------------------------------------------------------
# Engine selection
# ---------------------------------------------------------------------------


def engine() -> Backend:
    """Return the name of the currently active engine.

    ``ISALGRAPH_ENGINE`` overrides the compiled-in default.  An explicit
    ``backend=`` argument passed to any dispatching function overrides this
    in turn; see :func:`resolve`.

    Returns:
        Either ``"cpp"`` or ``"python"``.

    Raises:
        BackendError: If ``ISALGRAPH_ENGINE=cpp`` but the extension is absent,
            or if the variable holds an unrecognised value.
    """
    override = os.environ.get("ISALGRAPH_ENGINE", "").strip().lower()
    if override == "cpp":
        if not _CPP_AVAILABLE:
            raise errors.BackendError(
                "ISALGRAPH_ENGINE=cpp requested but isalgraph.core._native could not be "
                "imported. Build the extension with: pip install -e '.[dev,native]'"
            )
        return "cpp"
    if override == "python":
        return "python"
    if override:
        raise errors.BackendError(
            f"ISALGRAPH_ENGINE={override!r} is not valid; choose 'cpp' or 'python'."
        )
    return DEFAULT_BACKEND


def build_info() -> dict[str, str]:
    """Return build metadata for the active engine.

    Under the C++ engine the values come from the compiled probe translation
    unit, so a stale or wrong-ISA ``.so`` is detectable without running any
    algorithm.  Keys are always present; an empty value means not applicable.

    Returns:
        Mapping with keys ``engine``, ``compiler``, ``cplusplus``,
        ``isa_level``, ``avx2``, ``fma``, ``avx512f``, ``ndebug``,
        ``build_hash``.
    """
    if engine() == "cpp" and _cpp_ext is not None:
        info: dict[str, str] = dict(_cpp_ext.build_info())
        info["engine"] = "cpp"
        return info
    return {
        "engine": "python",
        "compiler": "",
        "cplusplus": "",
        "isa_level": "",
        "avx2": "",
        "fma": "",
        "avx512f": "",
        "ndebug": "",
        "build_hash": "",
    }


def resolve(backend: str | None, registry: dict[str, C]) -> C:
    """Return the implementation registered under *backend*.

    Args:
        backend: Backend name.  ``None`` defers to :func:`engine`, which
            honours ``ISALGRAPH_ENGINE``.  A non-``None`` value always wins.
        registry: Per-function dispatch table.

    Returns:
        The chosen implementation.

    Raises:
        BackendError: If ``"cpp"`` is requested but the extension is absent,
            or if *backend* is not a key of *registry*.
    """
    chosen: str = engine() if backend is None else backend
    if chosen == "cpp" and not _CPP_AVAILABLE:
        # An explicit request never silently degrades.
        raise errors.BackendError(
            "backend='cpp' requested but isalgraph.core._native is not available. "
            "Run: pip install -e '.[dev,native]'"
        )
    impl = registry.get(chosen)
    if impl is None:
        valid = ", ".join(sorted(registry))
        raise errors.BackendError(f"Unknown backend {chosen!r}; valid options: {valid}")
    return impl


# ---------------------------------------------------------------------------
# Marshalling and shared validation
# ---------------------------------------------------------------------------


def _marshal(graph: SparseGraph) -> tuple[int, int, bool, int, list[list[int]]]:
    """Flatten *graph* for the FFI, preserving CPython set-iteration order.

    ``logical_edge_count`` is taken from the graph's own counter rather than
    derived from the adjacency lengths.  A self-loop in an undirected graph
    occupies one adjacency slot but increments ``_edge_count`` twice, so
    ``sum(len(adj)) // 2`` would undercount it and the encoder would stop one
    edge early.
    """
    n = graph.node_count()
    adjacency = [list(graph.neighbors(u)) for u in range(n)]
    return n, graph.max_nodes(), graph.directed(), graph.logical_edge_count(), adjacency


def _translate_encoding_error(exc: Exception) -> Exception:
    """Map a frozen-reference exception onto the :mod:`isalgraph.errors` type.

    Message text is carried through verbatim so both engines are
    indistinguishable to a caller matching on the message.
    """
    text = str(exc)
    if isinstance(exc, ValueError) and text.startswith(_DISCONNECTED_PREFIX):
        return errors.DisconnectedGraphError(text)
    if isinstance(exc, RuntimeError) and _STUCK_MARKER in text:
        return _EncodingStuckError(text)
    return exc


def _reject_python_timeout(timeout_s: float | None) -> None:
    """Refuse a wall-clock budget the Python reference cannot honour.

    Silently ignoring a budget would let an evaluation harness believe a run
    was bounded when it was not, so this is an error rather than a warning.
    """
    if timeout_s is not None:
        raise errors.BackendError(
            "timeout_s is only supported by the 'cpp' backend; the Python reference "
            "has no interruption point. Pass backend='cpp' or timeout_s=None."
        )


def _check_initial_node(graph: SparseGraph, initial_node: int) -> None:
    """Reproduce ``GraphToString.run``'s range check, message included."""
    if initial_node < 0 or initial_node >= graph.node_count():
        raise ValueError("Initial node out of range")


def _check_reachability(graph: SparseGraph, initial_node: int) -> None:
    """Reproduce ``GraphToString._check_reachability``, message included.

    Performed Python-side rather than in C++ because the message embeds a
    Python ``set`` repr, whose element order follows CPython hash-slot layout.
    Reproducing that in C++ would be fragile for no gain: the check is
    O(V + E) against an encoder that is far more expensive.
    """
    n = graph.node_count()
    if n <= 1:
        return

    visited: set[int] = set()
    stack: list[int] = [initial_node]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                stack.append(neighbor)

    if len(visited) != n:
        unreachable = set(range(n)) - visited
        raise errors.DisconnectedGraphError(
            f"GraphToString requires all nodes to be reachable from "
            f"initial_node={initial_node} via outgoing edges. "
            f"Unreachable nodes: {unreachable}. "
            f"For directed graphs, ensure all nodes are reachable "
            f"from the start node. For undirected graphs, ensure "
            f"the graph is connected."
        )


def _validate_instructions(input_string: str) -> None:
    """Reproduce ``StringToGraph.__init__``'s alphabet check, message included."""
    if not set(input_string).issubset(VALID_INSTRUCTIONS):
        raise errors.InvalidStringError(f"Invalid IsalGraph string: {input_string!r}")


# ---------------------------------------------------------------------------
# canonical_string
# ---------------------------------------------------------------------------


def _python_canonical_string(graph: SparseGraph, timeout_s: float | None, threads: int) -> str:
    from isalgraph.core.canonical import canonical_string as _ref

    _reject_python_timeout(timeout_s)
    del threads  # the reference is single-threaded by construction
    try:
        return _ref(graph)
    except (ValueError, RuntimeError) as exc:
        raise _translate_encoding_error(exc) from None


def _cpp_canonical_string(graph: SparseGraph, timeout_s: float | None, threads: int) -> str:
    assert _cpp_ext is not None
    n, max_nodes, directed, edges, adjacency = _marshal(graph)
    result: str = _cpp_ext.canonical_string(
        n, max_nodes, directed, edges, adjacency, timeout_s, threads
    )
    return result


_CANONICAL_STRING = {"python": _python_canonical_string, "cpp": _cpp_canonical_string}


def canonical_string(
    graph: SparseGraph,
    *,
    timeout_s: float | None = None,
    threads: int = 1,
    backend: str | None = None,
) -> str:
    """Compute the canonical IsalGraph string ``w*_G``.

    Args:
        graph: Graph to encode.  Must be connected (undirected) or have some
            node reaching all others (directed).
        timeout_s: Wall-clock budget in seconds.  ``None`` is unlimited.
            Supported by the ``cpp`` backend only.
        threads: Number of worker threads over the starting-node loop.  The
            default of 1 is the only value safe to assume inside a SLURM
            cgroup, where ``hardware_concurrency()`` reports the whole node.
        backend: ``"cpp"``, ``"python"``, or ``None`` to use :func:`engine`.

    Returns:
        The canonical string.

    Raises:
        DisconnectedGraphError: If no starting node reaches every other node.
        CanonicalizationTimeoutError: If *timeout_s* elapses.
        BackendError: If the requested backend is unavailable.
    """
    return resolve(backend, _CANONICAL_STRING)(graph, timeout_s, threads)


# ---------------------------------------------------------------------------
# pruned_canonical_string
# ---------------------------------------------------------------------------


def _python_pruned_canonical_string(
    graph: SparseGraph, timeout_s: float | None, threads: int
) -> str:
    from isalgraph.core.canonical_pruned import pruned_canonical_string as _ref

    _reject_python_timeout(timeout_s)
    del threads
    try:
        return _ref(graph)
    except (ValueError, RuntimeError) as exc:
        raise _translate_encoding_error(exc) from None


def _cpp_pruned_canonical_string(graph: SparseGraph, timeout_s: float | None, threads: int) -> str:
    assert _cpp_ext is not None
    n, max_nodes, directed, edges, adjacency = _marshal(graph)
    result: str = _cpp_ext.pruned_canonical_string(
        n, max_nodes, directed, edges, adjacency, timeout_s, threads
    )
    return result


_PRUNED_CANONICAL_STRING = {
    "python": _python_pruned_canonical_string,
    "cpp": _cpp_pruned_canonical_string,
}


def pruned_canonical_string(
    graph: SparseGraph,
    *,
    timeout_s: float | None = None,
    threads: int = 1,
    backend: str | None = None,
) -> str:
    """Compute the triplet-pruned canonical string.

    Same contract as :func:`canonical_string`, with V/v candidates filtered to
    those attaining the maximum structural triplet ``(|N_1|, |N_2|, |N_3|)``.

    Args:
        graph: Graph to encode.
        timeout_s: Wall-clock budget in seconds; ``cpp`` backend only.
        threads: Worker threads over the starting-node loop.
        backend: ``"cpp"``, ``"python"``, or ``None``.

    Returns:
        The pruned canonical string.

    Raises:
        DisconnectedGraphError: If no starting node reaches every other node.
        CanonicalizationTimeoutError: If *timeout_s* elapses.
        BackendError: If the requested backend is unavailable.
    """
    return resolve(backend, _PRUNED_CANONICAL_STRING)(graph, timeout_s, threads)


# ---------------------------------------------------------------------------
# levenshtein
# ---------------------------------------------------------------------------


def _python_levenshtein(s: str, t: str) -> int:
    from isalgraph.core.canonical import levenshtein as _ref

    return _ref(s, t)


def _cpp_levenshtein(s: str, t: str) -> int:
    assert _cpp_ext is not None
    result: int = _cpp_ext.levenshtein(s, t)
    return result


_LEVENSHTEIN = {"python": _python_levenshtein, "cpp": _cpp_levenshtein}


def levenshtein(s: str, t: str, *, backend: str | None = None) -> int:
    """Levenshtein edit distance between two strings.

    Args:
        s: First string.
        t: Second string.
        backend: ``"cpp"``, ``"python"``, or ``None``.

    Returns:
        Minimum number of single-character insertions, deletions and
        substitutions transforming *s* into *t*.
    """
    return resolve(backend, _LEVENSHTEIN)(s, t)


# ---------------------------------------------------------------------------
# graph_distance / pruned_graph_distance
# ---------------------------------------------------------------------------


def graph_distance(
    g1: SparseGraph,
    g2: SparseGraph,
    *,
    timeout_s: float | None = None,
    threads: int = 1,
    backend: str | None = None,
) -> int:
    """Approximate graph edit distance via Levenshtein on canonical strings.

    Args:
        g1: First graph.
        g2: Second graph.
        timeout_s: Per-canonicalisation budget; ``cpp`` backend only.
        threads: Worker threads per canonicalisation.
        backend: ``"cpp"``, ``"python"``, or ``None``.

    Returns:
        ``levenshtein(canonical_string(g1), canonical_string(g2))``.
    """
    w1 = canonical_string(g1, timeout_s=timeout_s, threads=threads, backend=backend)
    w2 = canonical_string(g2, timeout_s=timeout_s, threads=threads, backend=backend)
    return levenshtein(w1, w2, backend=backend)


def pruned_graph_distance(
    g1: SparseGraph,
    g2: SparseGraph,
    *,
    timeout_s: float | None = None,
    threads: int = 1,
    backend: str | None = None,
) -> int:
    """Approximate graph edit distance via Levenshtein on pruned canonical strings.

    Args:
        g1: First graph.
        g2: Second graph.
        timeout_s: Per-canonicalisation budget; ``cpp`` backend only.
        threads: Worker threads per canonicalisation.
        backend: ``"cpp"``, ``"python"``, or ``None``.

    Returns:
        ``levenshtein(pruned_canonical_string(g1), pruned_canonical_string(g2))``.
    """
    w1 = pruned_canonical_string(g1, timeout_s=timeout_s, threads=threads, backend=backend)
    w2 = pruned_canonical_string(g2, timeout_s=timeout_s, threads=threads, backend=backend)
    return levenshtein(w1, w2, backend=backend)


# ---------------------------------------------------------------------------
# string_to_graph
# ---------------------------------------------------------------------------


def _python_string_to_graph(input_string: str, directed: bool) -> SparseGraph:
    from isalgraph.core.string_to_graph import StringToGraph

    _validate_instructions(input_string)
    graph, _ = StringToGraph(input_string, directed).run()
    return graph


def _cpp_string_to_graph(input_string: str, directed: bool) -> SparseGraph:
    assert _cpp_ext is not None
    _validate_instructions(input_string)
    node_count, max_nodes, _directed, edges = _cpp_ext.string_to_graph(input_string, directed)

    graph = SparseGraph(max_nodes, directed)
    for _ in range(node_count):
        graph.add_node()
    # Edges are replayed in the engine's add_edge call order, which is the
    # reference's order too. That matters beyond set contents: CPython
    # resolves set slot collisions by insertion order, so identical order
    # gives adjacency sets that iterate identically as well as compare equal.
    for source, target in edges:
        graph.add_edge(source, target)
    return graph


_STRING_TO_GRAPH = {"python": _python_string_to_graph, "cpp": _cpp_string_to_graph}


def string_to_graph(
    input_string: str, directed: bool, *, backend: str | None = None
) -> SparseGraph:
    """Decode an IsalGraph instruction string into a graph.

    Args:
        input_string: String over the alphabet ``{N, n, P, p, V, v, C, c, W}``.
        directed: Whether to build a directed graph.
        backend: ``"cpp"``, ``"python"``, or ``None``.

    Returns:
        The decoded graph.

    Raises:
        InvalidStringError: If *input_string* leaves the alphabet.
    """
    return resolve(backend, _STRING_TO_GRAPH)(input_string, directed)


# ---------------------------------------------------------------------------
# graph_to_string
# ---------------------------------------------------------------------------


def _python_graph_to_string(graph: SparseGraph, initial_node: int) -> str:
    from isalgraph.core.graph_to_string import GraphToString

    try:
        output, _ = GraphToString(graph).run(initial_node)
    except ValueError as exc:
        if str(exc).startswith("GraphToString requires"):
            raise errors.DisconnectedGraphError(str(exc)) from None
        raise
    except RuntimeError as exc:
        raise _translate_encoding_error(exc) from None
    return output


def _cpp_graph_to_string(graph: SparseGraph, initial_node: int) -> str:
    assert _cpp_ext is not None
    _check_initial_node(graph, initial_node)
    _check_reachability(graph, initial_node)
    n, max_nodes, directed, edges, adjacency = _marshal(graph)
    result: str = _cpp_ext.graph_to_string(n, max_nodes, directed, edges, adjacency, initial_node)
    return result


_GRAPH_TO_STRING = {"python": _python_graph_to_string, "cpp": _cpp_graph_to_string}


def graph_to_string(graph: SparseGraph, initial_node: int, *, backend: str | None = None) -> str:
    """Greedily encode *graph* into an IsalGraph instruction string.

    Args:
        graph: Graph to encode.
        initial_node: Index of the starting node in *graph*.
        backend: ``"cpp"``, ``"python"``, or ``None``.

    Returns:
        The instruction string.

    Raises:
        ValueError: If *initial_node* is out of range.
        DisconnectedGraphError: If some node is unreachable from *initial_node*.
    """
    return resolve(backend, _GRAPH_TO_STRING)(graph, initial_node)


__all__ = [
    "Backend",
    "DEFAULT_BACKEND",
    "build_info",
    "canonical_string",
    "engine",
    "graph_distance",
    "graph_to_string",
    "levenshtein",
    "pruned_canonical_string",
    "pruned_graph_distance",
    "resolve",
    "string_to_graph",
]
