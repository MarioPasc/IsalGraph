"""Custom exception hierarchy for IsalGraph."""


class IsalGraphError(Exception):
    """Base exception for all IsalGraph errors."""


class CapacityError(IsalGraphError):
    """Raised when a data structure exceeds its preallocated capacity."""


class InvalidNodeError(IsalGraphError):
    """Raised when an operation references a nonexistent node."""


class InvalidStringError(IsalGraphError):
    """Raised when an IsalGraph instruction string contains invalid characters."""
