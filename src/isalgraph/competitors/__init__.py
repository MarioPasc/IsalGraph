"""Competitor representations for the Pattern Recognition revision.

Serves **AE.4a** (the requirement-modal owner), AE.3, R1.1, R1.2a/b and
R3.6a: the side-by-side comparison against existing graph representations
that the Area Editor asked for in their own voice.

Eleven registered backends across three families -- the ``n^2``
serialisations (adjacency, graph6, nauty->graph6, AGM CAM), the ``m``-scaling
ones (sparse6, min-DFS, IsalGraph), and WL, which is not a serialisation at
all -- plus the size null.

**The import contract, and a test enforces it.**  This package sits at the
same tier as :mod:`isalgraph.viz` and :mod:`isalgraph.adapters`::

    isalgraph.competitors  -> optional: networkx, pynauty, grakel, rapidfuzz
    isalgraph.viz          -> optional: matplotlib, networkx, igraph
    isalgraph.adapters     -> optional: networkx, igraph, pyg
    isalgraph.core         -> stdlib only (+ the optional C++ engine)

- ``import isalgraph.competitors`` **must succeed with every optional
  dependency uninstalled.**  Each third-party import lives inside a function
  body or behind the lazy registry.
- **A missing dependency raises**
  :class:`~isalgraph.errors.BackendUnavailableError`.  It never degrades
  silently -- the same rule as :mod:`isalgraph.core.backends`, and the same
  reason: a silent degrade turns a wrong number into a plausible one.
- ``isalgraph/__init__.py`` does not import this package, so the top-level
  import chain stays stdlib-only.

Four entry points, all dispatching through the registry so that adding a
backend extends them without editing them:

===========================================  ==========================================
``python -m isalgraph.competitors.smoke``    encode a real cohort, time it, record failures
``python -m isalgraph.competitors.grid``     F1-F4 and F6.  **Never F5**
``python -m isalgraph.competitors.f5``       F5 alone, reported, not an input to selection
``python -m isalgraph.competitors.reproduce``the reproduction gate
===========================================  ==========================================
"""

from __future__ import annotations

from isalgraph.competitors.base import (
    BitCount,
    Budget,
    Capability,
    DistanceMetric,
    Encoding,
    PositionalFrame,
    ReprBackend,
    VectorBackend,
)
from isalgraph.competitors.registry import (
    available_backends,
    available_metrics,
    get_backend,
    get_metric,
    register_backend,
    register_metric,
    registered_backends,
    unavailable_backends,
)

__all__ = [
    "BitCount",
    "Budget",
    "Capability",
    "DistanceMetric",
    "Encoding",
    "PositionalFrame",
    "ReprBackend",
    "VectorBackend",
    "available_backends",
    "available_metrics",
    "get_backend",
    "get_metric",
    "register_backend",
    "register_metric",
    "registered_backends",
    "unavailable_backends",
]
