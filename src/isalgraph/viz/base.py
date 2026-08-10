"""``GraphVizBackend`` -- the abstract drawing interface.

A backend wraps one drawing strategy (raw matplotlib primitives,
NetworkX, igraph) behind a single :meth:`GraphVizBackend.draw` method.

Three contract clauses matter and are enforced by the test-suite:

1. **A backend never creates a figure.** It paints on a caller-supplied
   ``Axes``. Figure geometry is the composer's business, not the
   backend's.
2. **A backend takes a layout in and returns the layout it used.** That
   is what lets a multi-panel figure pin node coordinates across columns
   so nodes do not jump between frames.
3. **Grey masks are caller-decided.** The backend receives
   ``grayed_nodes`` / ``grayed_edges`` as sets and renders them as
   ghosts; it never works out for itself which elements are "not yet
   built", because that answer is direction-dependent (see
   :mod:`isalgraph.viz.composite`).

``matplotlib`` is referenced only under ``TYPE_CHECKING``; at runtime
``Axes`` aliases :data:`typing.Any`, so importing this module does not
require matplotlib.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeAlias

from isalgraph.core.sparse_graph import SparseGraph
from isalgraph.core.trace import Edge
from isalgraph.types import NodeId

if TYPE_CHECKING:
    from matplotlib.axes import Axes
else:
    Axes = Any

Position: TypeAlias = tuple[float, float]


class GraphVizBackend(ABC):
    """Abstract base class for graph drawing backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this backend (matches the registry key)."""
        ...

    @classmethod
    def is_available(cls) -> bool:
        """Report whether this backend's third-party library can be imported.

        The default implementation returns ``True``, which is correct for
        backends with no third-party requirement beyond matplotlib.
        Backends wrapping an optional library must override it.

        Upstream IsalHG has no such hook: its ``available_backends()``
        performs detection at *draw* time, so it happily lists backends
        whose library is not installed and only fails once a caller tries
        to draw. Declaring availability at the class level lets
        :func:`isalgraph.viz.registry.available_backends` filter honestly.
        """
        return True

    @abstractmethod
    def draw(
        self,
        graph: SparseGraph,
        ax: Axes,
        *,
        node_colors: dict[NodeId, str],
        edge_colors: dict[Edge, str],
        grayed_nodes: frozenset[NodeId] = frozenset(),
        grayed_edges: frozenset[Edge] = frozenset(),
        layout: dict[NodeId, Position] | None = None,
    ) -> dict[NodeId, Position]:
        """Draw *graph* on *ax* and return the layout used.

        Args:
            graph: The graph to draw.
            ax: Target matplotlib axes. The backend must not create one.
            node_colors: Per-node colour. Every node needs an entry.
            edge_colors: Per-edge colour, keyed by the normalised
                ``(u, v)`` tuple. Every edge needs an entry.
            grayed_nodes: Nodes to render as ghosts (reduced opacity).
            grayed_edges: Edges to render as ghosts.
            layout: When given, the backend must reuse these coordinates
                instead of computing its own. When ``None``, the backend
                computes a layout and returns it.

        Returns:
            ``{node_id: (x, y)}`` for every node drawn, so the caller can
            pin the layout for the next panel.
        """
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


__all__ = ["GraphVizBackend", "Position"]
