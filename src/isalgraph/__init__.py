"""IsalGraph -- Representation of graph structure by instruction strings."""

from isalgraph.core.algorithms import (
    DEFAULT_ALGORITHM,
    ExhaustiveG2S,
    G2SAlgorithm,
    GreedyMinG2S,
    GreedySingleG2S,
    PrunedExhaustiveG2S,
)
from isalgraph.core.canonical import canonical_string, graph_distance, levenshtein
from isalgraph.core.canonical_pruned import pruned_canonical_string, pruned_graph_distance
from isalgraph.core.cdll import CircularDoublyLinkedList
from isalgraph.core.graph_to_string import GraphToString
from isalgraph.core.sparse_graph import SparseGraph
from isalgraph.core.string_to_graph import StringToGraph

__all__ = [
    # Data structures
    "CircularDoublyLinkedList",
    "SparseGraph",
    # Converters
    "StringToGraph",
    "GraphToString",
    # Canonical forms and distances
    "canonical_string",
    "pruned_canonical_string",
    "graph_distance",
    "pruned_graph_distance",
    "levenshtein",
    # G2S algorithm strategies
    "G2SAlgorithm",
    "GreedyMinG2S",
    "ExhaustiveG2S",
    "PrunedExhaustiveG2S",
    "GreedySingleG2S",
    "DEFAULT_ALGORITHM",
]
__version__ = "0.1.0"
