"""Render the CDLL ring with its two pointers.

Nodes are evenly spaced circles labelled with the *graph* node id they
carry. The primary and secondary pointers are arrows drawn from outside
the ring, coloured :data:`~isalgraph.viz.style.POINTER_PALETTE`\\ ``[0]``
and ``[1]`` -- the same two colours the instruction strip uses for its
stroke accents, which is what makes the correspondence readable without
a legend.

:func:`draw_cdll_ring` keeps the exact signature of the original
``benchmarks/real_data/eval_visualizations/cdll_drawing.py`` so the
figure scripts that already call it keep working;
:func:`draw_cdll_ring_for_snapshot` is the trace-driven entry point,
which resolves a :class:`~isalgraph.core.trace.StepSnapshot` onto it.

The original depended on numpy for ``cos``/``sin``/``pi`` alone; this
port uses :mod:`math` instead, which is numerically identical for
scalars and drops a dependency.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from isalgraph.core.trace import StepSnapshot
from isalgraph.types import NodeId
from isalgraph.viz.style import (
    NEW_ELEMENT_COLOR,
    POINTER_OVERLAP_COLOR,
    POINTER_PALETTE,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.patches import Patch
else:
    Axes = Any
    Patch = Any

#: Retained under their original names for the benchmark figure scripts.
PRIMARY_COLOR: str = POINTER_PALETTE[0]
SECONDARY_COLOR: str = POINTER_PALETTE[1]
NEW_NODE_COLOR: str = NEW_ELEMENT_COLOR
_DEFAULT_NODE_COLOR: str = "#DDDDDD"


def get_legend_handles(include_new_node: bool = False) -> list[Patch]:
    """Return legend handles for the CDLL pointer colours.

    Args:
        include_new_node: Also include the new-node swatch.

    Returns:
        Matplotlib ``Patch`` handles for ``fig.legend``.
    """
    import matplotlib.patches as mpatches

    handles = [
        mpatches.Patch(facecolor=PRIMARY_COLOR, label="π (primary)"),
        mpatches.Patch(facecolor=SECONDARY_COLOR, label="σ (secondary)"),
    ]
    if include_new_node:
        handles.append(mpatches.Patch(facecolor=NEW_NODE_COLOR, label="New node"))
    return handles


def draw_cdll_ring(
    ax: Axes,
    cdll_order: list[int],
    primary_ptr_idx: int,
    secondary_ptr_idx: int,
    *,
    new_node_payload: int | None = None,
    radius: float = 1.0,
    node_radius: float = 0.15,
    show_legend: bool = False,
) -> None:
    """Draw the CDLL as a ring with pointer arrows.

    Args:
        ax: Target matplotlib axes.
        cdll_order: Graph node payloads in CDLL traversal order.
        primary_ptr_idx: Position *within cdll_order* of the primary
            pointer. Note this is a ring position, not a CDLL slot index
            and not a graph node id.
        secondary_ptr_idx: Ring position of the secondary pointer.
        new_node_payload: Highlight this payload as newly created.
        radius: Ring radius.
        node_radius: Node circle radius.
        show_legend: Draw a per-panel legend instead of relying on a
            shared figure-level one.
    """
    from matplotlib.patches import Circle

    n = len(cdll_order)
    if n == 0:
        ax.axis("off")
        return

    angles = [math.pi / 2 - 2 * math.pi * i / n for i in range(n)]
    positions = [(radius * math.cos(a), radius * math.sin(a)) for a in angles]

    for i in range(n):
        j = (i + 1) % n
        ax.annotate(
            "",
            xy=positions[j],
            xytext=positions[i],
            arrowprops={"arrowstyle": "-", "color": "0.6", "linewidth": 0.8},
        )

    for i, (x, y) in enumerate(positions):
        payload = cdll_order[i]
        if new_node_payload is not None and payload == new_node_payload:
            color = NEW_NODE_COLOR
        elif i == primary_ptr_idx and i == secondary_ptr_idx:
            color = POINTER_OVERLAP_COLOR
        elif i == primary_ptr_idx:
            color = PRIMARY_COLOR
        elif i == secondary_ptr_idx:
            color = SECONDARY_COLOR
        else:
            color = _DEFAULT_NODE_COLOR

        ax.add_patch(
            Circle((x, y), node_radius, facecolor=color, edgecolor="0.3", linewidth=0.8, zorder=3)
        )
        ax.text(
            x, y, str(payload), ha="center", va="center", fontsize=7, fontweight="bold", zorder=4
        )

    arrow_radius = radius + 0.35
    _draw_pointer_arrow(
        ax,
        positions[primary_ptr_idx],
        angles[primary_ptr_idx],
        arrow_radius,
        node_radius,
        PRIMARY_COLOR,
        "π",
    )
    # When both pointers rest on one node, fan the secondary arrow away
    # so the two arrowheads stay distinguishable.
    secondary_angle = (
        angles[secondary_ptr_idx]
        if secondary_ptr_idx != primary_ptr_idx
        else angles[primary_ptr_idx] + 0.3
    )
    _draw_pointer_arrow(
        ax,
        positions[secondary_ptr_idx],
        secondary_angle,
        arrow_radius,
        node_radius,
        SECONDARY_COLOR,
        "σ",
    )

    if show_legend:
        ax.legend(
            handles=get_legend_handles(include_new_node=new_node_payload is not None),
            loc="lower center",
            fontsize=5,
            ncol=3,
            framealpha=0.7,
        )

    margin = arrow_radius + 0.45
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_pointer_arrow(
    ax: Axes,
    target_pos: tuple[float, float],
    angle: float,
    arrow_radius: float,
    node_radius: float,
    color: str,
    label: str,
) -> None:
    """Draw one labelled pointer arrow from outside the ring toward a node."""
    start_x = arrow_radius * math.cos(angle)
    start_y = arrow_radius * math.sin(angle)
    ax.annotate(
        "",
        xy=target_pos,
        xytext=(start_x, start_y),
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linewidth": 1.5,
            "shrinkA": 0,
            "shrinkB": node_radius * 72,  # axis units to points
        },
    )
    label_x = (arrow_radius + 0.25) * math.cos(angle)
    label_y = (arrow_radius + 0.25) * math.sin(angle)
    ax.text(
        label_x,
        label_y,
        label,
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color=color,
    )


def draw_cdll_ring_for_snapshot(
    ax: Axes,
    snapshot: StepSnapshot,
    *,
    highlight_new_node: bool = True,
    radius: float = 1.0,
    node_radius: float = 0.15,
    show_legend: bool = False,
) -> None:
    """Draw the ring for *snapshot*.

    Resolves the snapshot's ``primary_node`` / ``secondary_node`` -- which
    are graph node ids -- to their positions in ``cdll_node_order``, and
    delegates to :func:`draw_cdll_ring`.

    Args:
        ax: Target matplotlib axes.
        snapshot: The step to render.
        highlight_new_node: Colour the node created by this step, taken
            from ``snapshot.created_edge`` on a ``V``/``v`` instruction.
        radius: Ring radius.
        node_radius: Node circle radius.
        show_legend: Draw a per-panel legend.
    """
    order: list[NodeId] = list(snapshot.cdll_node_order)
    if not order:
        ax.axis("off")
        return

    index_of = {v: i for i, v in enumerate(order)}
    primary_idx = index_of.get(snapshot.primary_node, 0)
    secondary_idx = index_of.get(snapshot.secondary_node, 0)

    new_payload: int | None = None
    if highlight_new_node and snapshot.instruction in ("V", "v") and snapshot.created_edge:
        # V/v always allocates the higher-numbered endpoint of the edge
        # it creates, because ids are handed out contiguously.
        new_payload = max(snapshot.created_edge)

    draw_cdll_ring(
        ax,
        order,
        primary_idx,
        secondary_idx,
        new_node_payload=new_payload,
        radius=radius,
        node_radius=node_radius,
        show_legend=show_legend,
    )


__all__ = [
    "NEW_NODE_COLOR",
    "PRIMARY_COLOR",
    "SECONDARY_COLOR",
    "draw_cdll_ring",
    "draw_cdll_ring_for_snapshot",
    "get_legend_handles",
]
