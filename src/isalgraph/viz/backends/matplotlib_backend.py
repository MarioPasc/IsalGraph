"""Default backend: raw matplotlib primitives.

A simple graph needs no third-party drawing library. Nodes are
``Circle`` patches, edges are ``Line2D`` segments, and directed edges get
a ``FancyArrowPatch`` head. That keeps the whole visualisation subsystem
working on a matplotlib-only install, which is why this is the default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isalgraph.core.sparse_graph import SparseGraph
from isalgraph.core.trace import Edge, graph_edges
from isalgraph.types import NodeId
from isalgraph.viz.base import GraphVizBackend, Position
from isalgraph.viz.layout import cdll_ring_positions, compact_graph_layout
from isalgraph.viz.registry import register_backend
from isalgraph.viz.style import (
    ACTIVE_ALPHA,
    DEFAULT_NODE_COLOR,
    GRAYED_ALPHA,
    GRAYED_EDGE,
    GRAYED_FACE,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
else:
    Axes = Any


class MatplotlibBackend(GraphVizBackend):
    """Draw a :class:`SparseGraph` with plain matplotlib patches."""

    def __init__(
        self,
        *,
        node_radius: float = 0.09,
        node_lw: float = 0.9,
        edge_lw: float = 1.2,
        show_labels: bool = True,
        label_fontsize: float = 7.0,
        axis_margin: float = 0.12,
    ) -> None:
        self._node_radius = node_radius
        self._node_lw = node_lw
        self._edge_lw = edge_lw
        self._show_labels = show_labels
        self._label_fontsize = label_fontsize
        self._axis_margin = axis_margin

    @property
    def name(self) -> str:
        return "matplotlib"

    @classmethod
    def is_available(cls) -> bool:
        """Report whether matplotlib imports."""
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            return False
        return True

    def _resolve_layout(
        self,
        graph: SparseGraph,
        layout: dict[NodeId, Position] | None,
    ) -> dict[NodeId, Position]:
        """Reuse *layout* if given, else compute one, falling back to a ring."""
        if layout is not None:
            return layout
        try:
            return compact_graph_layout(graph)
        except ImportError:
            # No NetworkX: a ring is a poor but honest fallback that keeps
            # the matplotlib-only install fully functional.
            return cdll_ring_positions(tuple(range(graph.node_count())))

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
        """Draw *graph* on *ax*; see :meth:`GraphVizBackend.draw`."""
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, FancyArrowPatch

        pos = self._resolve_layout(graph, layout)
        directed = graph.directed()

        for edge in graph_edges(graph):
            u, v = edge
            if u not in pos or v not in pos:
                continue
            is_grayed = edge in grayed_edges
            color = GRAYED_EDGE if is_grayed else edge_colors.get(edge, "0.5")
            alpha = GRAYED_ALPHA if is_grayed else ACTIVE_ALPHA
            if u == v:
                continue  # self-loops carry no structure in this figure family
            if directed:
                ax.add_patch(
                    FancyArrowPatch(
                        pos[u],
                        pos[v],
                        arrowstyle="-|>",
                        mutation_scale=9,
                        color=color,
                        lw=self._edge_lw,
                        alpha=alpha,
                        shrinkA=self._node_radius * 72,
                        shrinkB=self._node_radius * 72,
                        zorder=1,
                    )
                )
            else:
                ax.add_line(
                    Line2D(
                        [pos[u][0], pos[v][0]],
                        [pos[u][1], pos[v][1]],
                        color=color,
                        lw=self._edge_lw,
                        alpha=alpha,
                        zorder=1,
                        solid_capstyle="round",
                    )
                )

        for v in range(graph.node_count()):
            if v not in pos:
                continue
            x, y = pos[v]
            is_grayed = v in grayed_nodes
            face = GRAYED_FACE if is_grayed else node_colors.get(v, DEFAULT_NODE_COLOR)
            alpha = GRAYED_ALPHA if is_grayed else ACTIVE_ALPHA
            ax.add_patch(
                Circle(
                    (x, y),
                    self._node_radius,
                    facecolor=face,
                    edgecolor="0.25" if not is_grayed else GRAYED_EDGE,
                    lw=self._node_lw,
                    alpha=alpha,
                    zorder=2,
                )
            )
            if self._show_labels:
                ax.text(
                    x,
                    y,
                    str(v),
                    ha="center",
                    va="center",
                    fontsize=self._label_fontsize,
                    color="#111111" if not is_grayed else "#777777",
                    zorder=3,
                )

        _finish_axes(ax, pos, self._node_radius + self._axis_margin)
        return pos


def _finish_axes(ax: Axes, pos: dict[NodeId, Position], pad: float) -> None:
    """Set equal-aspect limits around *pos* with *pad* slack, and hide the frame."""
    if pos:
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
    else:
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


register_backend("matplotlib", MatplotlibBackend)

__all__ = ["MatplotlibBackend"]
