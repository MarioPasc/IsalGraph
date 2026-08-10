"""Visualisation subsystem for IsalGraph.

Three atomic views -- the CDLL ring, the instruction strip and the graph
-- composed into per-step columns and stacked into step figures and
round-trip collages, plus the canonical-search-space schematic.

Dependency rule: importing this package must succeed with no third-party
library installed. Every ``matplotlib`` / ``networkx`` / ``igraph``
import lives inside a function or method body, and ``Axes`` is referenced
only under ``TYPE_CHECKING``. The rule is enforced by
``tests/viz/test_import_without_matplotlib.py``, which blocks the imports
with a meta-path finder and imports the package anyway.
"""

from isalgraph.viz.base import GraphVizBackend, Position
from isalgraph.viz.registry import (
    DEFAULT_BACKEND,
    available_backends,
    get_backend,
    register_backend,
    registered_backends,
)

__all__ = [
    "DEFAULT_BACKEND",
    "GraphVizBackend",
    "Position",
    "available_backends",
    "get_backend",
    "register_backend",
    "registered_backends",
]
