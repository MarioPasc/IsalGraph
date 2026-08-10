"""IsalGraph core.

Pure-Python parts have zero external dependencies.  The optional C++ engine
(``isalgraph.core._native``) is built from ``core/native/`` and is used
automatically when present.

The names re-exported here are the **dispatching** entry points from
``isalgraph.core.backends``, not the pure-Python reference functions.  They run
on whichever engine is active and accept ``backend="cpp" | "python"`` to force
one.  The reference implementations remain importable at their original paths
(``isalgraph.core.canonical``, ``isalgraph.core.canonical_pruned``) and are what
the differential test suite compares against; import them directly only when you
specifically want the Python implementation regardless of engine.
"""

from isalgraph.core.algorithms import (
    DEFAULT_ALGORITHM,
    ExhaustiveG2S,
    G2SAlgorithm,
    GreedyMinG2S,
    GreedySingleG2S,
    PrunedExhaustiveG2S,
)
from isalgraph.core.backends import (
    DEFAULT_BACKEND,
    Backend,
    build_info,
    canonical_string,
    engine,
    graph_distance,
    levenshtein,
    pruned_canonical_string,
    pruned_graph_distance,
)
from isalgraph.core.cdll import CircularDoublyLinkedList
from isalgraph.core.graph_to_string import GraphToString
from isalgraph.core.sparse_graph import SparseGraph
from isalgraph.core.string_to_graph import StringToGraph
from isalgraph.core.trace import AlgorithmTrace, StepSnapshot

__all__ = [
    # Data structures
    "CircularDoublyLinkedList",
    "SparseGraph",
    # Converters
    "StringToGraph",
    "GraphToString",
    # Execution traces
    "AlgorithmTrace",
    "StepSnapshot",
    # Canonical forms and distances (engine-dispatched)
    "canonical_string",
    "pruned_canonical_string",
    "graph_distance",
    "pruned_graph_distance",
    "levenshtein",
    # Engine selection
    "Backend",
    "DEFAULT_BACKEND",
    "engine",
    "build_info",
    # G2S algorithm strategies
    "G2SAlgorithm",
    "GreedyMinG2S",
    "ExhaustiveG2S",
    "PrunedExhaustiveG2S",
    "GreedySingleG2S",
    "DEFAULT_ALGORITHM",
]
