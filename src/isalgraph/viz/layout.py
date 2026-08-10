"""Layout helpers shared across the three views.

:func:`cdll_ring_positions` is pure ``math``. :func:`compact_graph_layout`
imports NetworkX lazily inside the function body.
"""

from __future__ import annotations

import logging
import math

from isalgraph.core.sparse_graph import SparseGraph
from isalgraph.types import NodeId
from isalgraph.viz.base import Position

logger = logging.getLogger(__name__)


def cdll_ring_positions(
    node_order: tuple[NodeId, ...],
    *,
    radius: float = 1.0,
    start_angle: float = math.pi / 2,
    clockwise: bool = True,
) -> dict[NodeId, Position]:
    """Place *node_order* evenly on a circle of the given radius.

    Args:
        node_order: Graph node ids in forward circular CDLL order.
        radius: Ring radius in axis units.
        start_angle: Angle of the first node in radians; defaults to
            ``pi / 2`` (twelve o'clock).
        clockwise: Lay the ring out clockwise when ``True``, matching the
            ``next_node`` traversal direction used elsewhere.

    Returns:
        ``{node_id: (x, y)}``. A single node is placed at the origin.
    """
    n = len(node_order)
    if n == 0:
        return {}
    if n == 1:
        return {node_order[0]: (0.0, 0.0)}
    direction = -1.0 if clockwise else 1.0
    positions: dict[NodeId, Position] = {}
    for i, v in enumerate(node_order):
        theta = start_angle + direction * (2.0 * math.pi * i / n)
        positions[v] = (radius * math.cos(theta), radius * math.sin(theta))
    return positions


def compact_graph_layout(
    graph: SparseGraph,
    *,
    seed: int = 0,
    margin: float = 0.18,
    spring_iterations: int = 80,
    spring_k: float | None = 0.9,
    fit_fraction: float = 0.78,
) -> dict[NodeId, Position]:
    """Spring-layout *graph*, normalised to the canvas, strays pulled in.

    NetworkX's ``spring_layout`` lets disconnected components drift to
    the periphery, which forces the main cluster to shrink toward the
    middle of the frame. This helper:

    1. Spring-layouts the largest connected component.
    2. Recentres and rescales it into
       ``[-fit_fraction, fit_fraction]^2``, leaving the outer band of
       the ``[-1, 1]^2`` canvas as visual padding.
    3. Parks nodes of smaller components on a vertical strip just
       outside the main bounding box, ordered by node id.

    Unlike the IsalHG original this operates on the graph directly: a
    simple graph needs no clique expansion, so the primal-graph
    construction step is gone.

    Args:
        graph: The graph to lay out.
        seed: PRNG seed for the spring layout; pinned for reproducibility.
        margin: Gap between the main bounding box and the stray strip.
        spring_iterations: Force-iteration count.
        spring_k: Optimal inter-node distance passed to
            ``nx.spring_layout``. ``None`` uses NetworkX's ``1/sqrt(n)``.
        fit_fraction: Fraction of the canvas the main component occupies.

    Returns:
        ``{node_id: (x, y)}`` covering every node of *graph*.
    """
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(range(graph.node_count()))
    for u in range(graph.node_count()):
        for v in graph.neighbors(u):
            if u != v:
                g.add_edge(u, v)

    components = sorted(nx.connected_components(g), key=len, reverse=True)
    if not components:
        return {}

    main = components[0]
    if len(main) <= 1:
        # Degenerate: no edges at all. Spread every node evenly on a ring.
        return cdll_ring_positions(tuple(range(graph.node_count())))

    raw = nx.spring_layout(
        g.subgraph(main),
        seed=seed,
        iterations=spring_iterations,
        k=spring_k,
    )
    xs = [p[0] for p in raw.values()]
    ys = [p[1] for p in raw.values()]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    span = max(max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6))
    cx, cy = (xmax + xmin) / 2.0, (ymax + ymin) / 2.0
    scale = 2.0 * fit_fraction
    positions: dict[NodeId, Position] = {
        int(v): (scale * (x - cx) / span, scale * (y - cy) / span) for v, (x, y) in raw.items()
    }

    strays = sorted(int(v) for comp in components[1:] for v in comp)
    if strays:
        strip_top, strip_bot = 0.9, -0.9
        step = (strip_top - strip_bot) / max(1, len(strays))
        strip_x = 1.0 + margin
        for i, v in enumerate(strays):
            y = strip_top - (i + 0.5) * step if len(strays) > 1 else 0.0
            positions[v] = (strip_x, y)
        logger.debug(
            "compact_graph_layout: %d main + %d stray nodes over %d component(s)",
            len(main),
            len(strays),
            len(components),
        )

    return positions


__all__ = ["cdll_ring_positions", "compact_graph_layout"]
