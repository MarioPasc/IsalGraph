"""Shared type aliases and dataclasses for IsalGraph."""

from __future__ import annotations

# Type alias for graph node indices (contiguous integers 0..N-1).
NodeId = int

# Type alias for CDLL internal node indices (allocated from free list).
CdllIndex = int

# Type alias for an IsalGraph instruction string.
InstructionString = str

# The IsalGraph alphabet.
VALID_INSTRUCTIONS: frozenset[str] = frozenset("NnPpVvCcW")

# Trace entry: (graph_snapshot, cdll_snapshot, primary_ptr, secondary_ptr, string_so_far)
# Used by both StringToGraph and GraphToString when trace=True.
# Defined loosely here; converters use their own concrete types.
TraceEntry = tuple[object, object, CdllIndex, CdllIndex, str]

__all__: list[str] = [
    "NodeId",
    "CdllIndex",
    "InstructionString",
    "VALID_INSTRUCTIONS",
    "TraceEntry",
]
