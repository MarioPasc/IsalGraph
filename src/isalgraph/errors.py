"""Custom exception hierarchy for IsalGraph.

This module is the single source of truth for IsalGraph exception types.
The native C++ engine mirrors this hierarchy in
``core/native/include/isalgraph/errors.hpp`` and maps C++ exceptions back
onto these Python classes at the binding boundary, so a caller never has to
know which backend raised.

Layout::

    IsalGraphError
    |-- CapacityError                     (also RuntimeError)
    |-- InvalidNodeError                  (also IndexError)
    |-- InvalidStringError                (also ValueError)
    |-- EncodingError
    |   |-- DisconnectedGraphError        (also ValueError)
    |   |-- CanonicalizationTimeoutError  (also RuntimeError)
    |   +-- EncodingStuckError            (also RuntimeError)
    |-- BackendError                      (also RuntimeError)
    |-- VizError
    |   |-- VizBackendNotFoundError
    |   +-- VizBackendUnavailableError
    +-- CompetitorError
        |-- BackendUnavailableError       (also ImportError)
        |-- BackendNotFoundError          (also KeyError)
        |-- BudgetExceeded                (also RuntimeError)
        |   |-- AGMBudgetExceeded
        |   +-- MinDfsBudgetExceeded
        |-- BitCountUndefined             (also TypeError)
        |-- DistanceUndefined             (also ValueError)
        |-- NotReversible                 (also TypeError)
        +-- SuiteScopeError               (also ValueError)

**Why the builtin mixins.** The pure-Python reference implementation raises
plain builtins -- ``ValueError`` for an unreachable start node, ``RuntimeError``
for a full CDLL, ``IndexError`` for a bad node id -- and roughly thirty tests
pin those types.  The native engine raises the classes above instead.  Mixing
the corresponding builtin into each class makes the two backends
indistinguishable to ``except`` clauses and to ``pytest.raises``, so switching
engines cannot change a caller's control flow.

The builtin is attached to the **leaves**, never to ``EncodingError`` itself:
its descendants straddle the split.  "No starting node reaches all others" is
historically a ``ValueError`` (``canonical.py``, ``canonical_pruned.py``,
``greedy_min.py``), while "no valid operation found" is a ``RuntimeError``
(``graph_to_string.py``, ``canonical.py``).  A ``ValueError`` on the shared base
would make the ``RuntimeError`` leaves lie about their type.

``InvalidNodeError`` is deliberately an ``IndexError`` and **not** a
``ValueError``: ``SparseGraph`` raises ``IndexError`` for out-of-range node ids
and six tests depend on it.
"""


class IsalGraphError(Exception):
    """Base exception for all IsalGraph errors."""


# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------


class CapacityError(IsalGraphError, RuntimeError):
    """Raised when a data structure exceeds its preallocated capacity."""


class InvalidNodeError(IsalGraphError, IndexError):
    """Raised when an operation references a nonexistent node."""


class InvalidStringError(IsalGraphError, ValueError):
    """Raised when an IsalGraph instruction string contains invalid characters."""


# ----------------------------------------------------------------------
# Encoding (G2S / canonicalization)
# ----------------------------------------------------------------------


class EncodingError(IsalGraphError):
    """Raised when graph-to-string encoding cannot proceed.

    Carries no builtin mixin: its subclasses straddle the ``ValueError`` /
    ``RuntimeError`` split.  Catch a subclass, or catch this together with the
    builtin you expect.
    """


class DisconnectedGraphError(EncodingError, ValueError):
    """Raised when no starting node reaches every other node.

    For undirected graphs this means the graph is disconnected.  For
    directed graphs it means no node is a root of a spanning out-tree.
    """


class CanonicalizationTimeoutError(EncodingError, RuntimeError):
    """Raised when a canonical search exceeds its allotted budget."""


class EncodingStuckError(EncodingError, RuntimeError):
    """Raised when no valid instruction exists but the encoding is incomplete.

    This indicates an algorithmic error rather than bad input: the search
    exhausted every displacement pair without finding an applicable
    ``V``/``v``/``C``/``c`` while nodes or edges remained uninserted.
    """


# ----------------------------------------------------------------------
# Backend dispatch
# ----------------------------------------------------------------------


class BackendError(IsalGraphError, RuntimeError):
    """Raised when an unknown or unusable compute backend is requested."""


# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------


class VizError(IsalGraphError):
    """Base exception for the ``isalgraph.viz`` subpackage."""


class VizBackendNotFoundError(VizError):
    """Raised when a drawing backend name is not present in the registry."""


class VizBackendUnavailableError(VizError):
    """Raised when a registered drawing backend's third-party library is missing."""


# ----------------------------------------------------------------------
# Competitor representations (isalgraph.competitors)
# ----------------------------------------------------------------------


class CompetitorError(IsalGraphError):
    """Base exception for the ``isalgraph.competitors`` subpackage."""


class BackendUnavailableError(CompetitorError, ImportError):
    """Raised when a competitor backend's third-party library is absent.

    Mirrors ``BackendError``'s contract for the C++ engine: a missing
    optional dependency is an error, never a silent degrade to a different
    representation.  ``ImportError`` is mixed in so callers who already
    guard optional imports catch it without knowing about this hierarchy.
    """


class BackendNotFoundError(CompetitorError, KeyError):
    """Raised when a competitor backend or metric name is not registered."""


class BudgetExceeded(CompetitorError, RuntimeError):
    """Raised when a bounded search exhausts its budget.

    Carries no result.  A budget that runs out must never return an
    incumbent: doing so would put a non-canonical code into a column
    headed canonical, which is the error the pool exists to expose.
    """


class AGMBudgetExceeded(BudgetExceeded):
    """Raised when the AGM canonical-code search exhausts its node budget."""


class MinDfsBudgetExceeded(BudgetExceeded):
    """Raised when the minimum-DFS-code search exhausts its projection budget.

    The budget is on *memory*, not time: the construction holds every
    embedding realising the current minimal prefix, and that set is
    worst-case exponential in the number of ties.
    """


class BitCountUndefined(CompetitorError, TypeError):
    """Raised when a bit count is requested from a representation that has none.

    A feature-vector "bit cost" would measure the choice of container
    rather than the encoding, so the correct answer is an empty cell with
    a printed reason.
    """


class DistanceUndefined(CompetitorError, ValueError):
    """Raised when a metric is not defined on the pair it was given.

    This is a **result**, not a defect: which cells of the
    (representation x distance) grid are undefined is the measurement.
    """


class NotReversible(CompetitorError, TypeError):
    """Raised when ``decode`` is called on a representation that discards structure."""


class SuiteScopeError(CompetitorError, ValueError):
    """Raised when a backend is asked for a graph outside its declared suite scope."""


__all__ = [
    "IsalGraphError",
    "CapacityError",
    "InvalidNodeError",
    "InvalidStringError",
    "EncodingError",
    "DisconnectedGraphError",
    "CanonicalizationTimeoutError",
    "EncodingStuckError",
    "BackendError",
    "VizError",
    "VizBackendNotFoundError",
    "VizBackendUnavailableError",
    "CompetitorError",
    "BackendUnavailableError",
    "BackendNotFoundError",
    "BudgetExceeded",
    "AGMBudgetExceeded",
    "MinDfsBudgetExceeded",
    "BitCountUndefined",
    "DistanceUndefined",
    "NotReversible",
    "SuiteScopeError",
]
