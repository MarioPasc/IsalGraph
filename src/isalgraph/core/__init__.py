"""IsalGraph core -- zero external dependencies."""

from isalgraph.core.algorithms import (
    DEFAULT_ALGORITHM,
    ExhaustiveG2S,
    G2SAlgorithm,
    GreedyMinG2S,
    GreedySingleG2S,
)
from isalgraph.core.cdll import CircularDoublyLinkedList
from isalgraph.core.graph_to_string import GraphToString
from isalgraph.core.sparse_graph import SparseGraph
from isalgraph.core.string_to_graph import StringToGraph

__all__ = [
    "CircularDoublyLinkedList",
    "SparseGraph",
    "StringToGraph",
    "GraphToString",
    "G2SAlgorithm",
    "GreedyMinG2S",
    "ExhaustiveG2S",
    "GreedySingleG2S",
    "DEFAULT_ALGORITHM",
]
