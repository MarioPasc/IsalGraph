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
    +-- VizError
        |-- VizBackendNotFoundError
        +-- VizBackendUnavailableError

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
]
