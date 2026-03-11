"""IsalGraph G2S algorithm registry.

Provides the abstract base class and concrete implementations for
graph-to-string encoding algorithms.

Default algorithm: GreedyMinG2S (greedy over all starting nodes).
"""

from isalgraph.core.algorithms.base import G2SAlgorithm
from isalgraph.core.algorithms.exhaustive import ExhaustiveG2S
from isalgraph.core.algorithms.greedy_min import GreedyMinG2S
from isalgraph.core.algorithms.greedy_single import GreedySingleG2S

DEFAULT_ALGORITHM = GreedyMinG2S

__all__ = [
    "G2SAlgorithm",
    "GreedyMinG2S",
    "ExhaustiveG2S",
    "GreedySingleG2S",
    "DEFAULT_ALGORITHM",
]
