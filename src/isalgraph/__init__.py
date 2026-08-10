"""IsalGraph -- representation of graph structure by instruction strings.

Encodes the topology of a finite simple graph as a string over the nine-symbol
alphabet ``{N, n, P, p, V, v, C, c, W}``, reversibly and -- via the canonical
form -- as a complete isomorphism invariant.

Quick start::

    from isalgraph import SparseGraph, canonical_string, StringToGraph

    g = SparseGraph(3, directed_graph=False)
    for _ in range(3):
        g.add_node()
    g.add_edge(0, 1)
    g.add_edge(1, 2)

    w = canonical_string(g)          # runs on the C++ engine when available
    back, _ = StringToGraph(w, directed_graph=False).run()

Two things a caller should know:

* ``canonical_string`` and its siblings dispatch to the native C++ engine when
  ``isalgraph.core._native`` is importable, and to the pure-Python reference
  otherwise.  ``isalgraph.engine()`` reports which is active; the
  ``ISALGRAPH_ENGINE`` environment variable and a ``backend=`` keyword override
  it.  Both engines are byte-exact against each other.
* **The canonical string does not encode directedness.**  A single undirected
  edge and a single directed arc both canonicalise to ``"V"``.  Decoding
  therefore needs the ``directed`` flag as separate metadata, and any
  deduplication over a corpus mixing both must key on ``(directed, string)``.

Visualization lives in ``isalgraph.viz`` and is imported explicitly; it is not
pulled in here, so the root package stays free of matplotlib.
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
__version__ = "0.1.0"
