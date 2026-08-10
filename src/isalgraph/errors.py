"""Custom exception hierarchy for IsalGraph.

This module is the single source of truth for IsalGraph exception types.
The native C++ engine mirrors this hierarchy in
``core/_native/include/isalgraph/errors.hpp`` and maps C++ exceptions back
onto these Python classes at the binding boundary, so a caller never has to
know which backend raised.

Layout::

    IsalGraphError
    |-- CapacityError
    |-- InvalidNodeError
    |-- InvalidStringError
    |-- EncodingError
    |   |-- DisconnectedGraphError
    |   +-- CanonicalizationTimeoutError
    |-- BackendError
    +-- VizError
        |-- VizBackendNotFoundError
        +-- VizBackendUnavailableError
"""


class IsalGraphError(Exception):
    """Base exception for all IsalGraph errors."""


# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------


class CapacityError(IsalGraphError):
    """Raised when a data structure exceeds its preallocated capacity."""


class InvalidNodeError(IsalGraphError):
    """Raised when an operation references a nonexistent node."""


class InvalidStringError(IsalGraphError):
    """Raised when an IsalGraph instruction string contains invalid characters."""


# ----------------------------------------------------------------------
# Encoding (G2S / canonicalization)
# ----------------------------------------------------------------------


class EncodingError(IsalGraphError):
    """Raised when graph-to-string encoding cannot proceed."""


class DisconnectedGraphError(EncodingError):
    """Raised when no starting node reaches every other node.

    For undirected graphs this means the graph is disconnected.  For
    directed graphs it means no node is a root of a spanning out-tree.
    """


class CanonicalizationTimeoutError(EncodingError):
    """Raised when a canonical search exceeds its allotted budget."""


# ----------------------------------------------------------------------
# Backend dispatch
# ----------------------------------------------------------------------


class BackendError(IsalGraphError):
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
    "BackendError",
    "VizError",
    "VizBackendNotFoundError",
    "VizBackendUnavailableError",
]
