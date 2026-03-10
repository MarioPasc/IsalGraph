"""CDLL ring drawing utility for algorithm illustration figures.

Draws the Circular Doubly Linked List as a ring of nodes with
primary (pi) and secondary (sigma) pointer arrows.
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Pointer colors (exported for shared legend use)
PRIMARY_COLOR = "#EE6677"  # Paul Tol red
SECONDARY_COLOR = "#4477AA"  # Paul Tol blue
NEW_NODE_COLOR = "#CCBB44"  # Paul Tol yellow
_DEFAULT_NODE_COLOR = "#DDDDDD"


def get_legend_handles(include_new_node: bool = False) -> list[mpatches.Patch]:
    """Return legend handles for CDLL pointer colors.

    Args:
        include_new_node: Whether to include the new-node legend entry.

    Returns:
        List of matplotlib Patch handles for use in fig.legend().
    """
    handles = [
        mpatches.Patch(facecolor=PRIMARY_COLOR, label="\u03c0 (primary)"),
        mpatches.Patch(facecolor=SECONDARY_COLOR, label="\u03c3 (secondary)"),
    ]
    if include_new_node:
        handles.append(mpatches.Patch(facecolor=NEW_NODE_COLOR, label="New node"))
    return handles


def draw_cdll_ring(
    ax: plt.Axes,
    cdll_order: list[int],
    primary_ptr_idx: int,
    secondary_ptr_idx: int,
    *,
    new_node_payload: int | None = None,
    radius: float = 1.0,
    node_radius: float = 0.15,
    show_legend: bool = False,
) -> None:
    """Draw CDLL as a circular ring with pointer arrows.

    Args:
        ax: Matplotlib axes.
        cdll_order: List of graph node payloads in CDLL traversal order.
        primary_ptr_idx: Index into cdll_order for primary pointer position.
        secondary_ptr_idx: Index into cdll_order for secondary pointer position.
        new_node_payload: If set, highlight this node payload in yellow.
        radius: Ring radius.
        node_radius: Individual node circle radius.
        show_legend: Whether to draw per-panel legend (default False; use shared).
    """
    n = len(cdll_order)
    if n == 0:
        ax.axis("off")
        return

    # Compute node positions (evenly spaced on circle, starting from top)
    angles = [np.pi / 2 - 2 * np.pi * i / n for i in range(n)]
    positions = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]

    # Draw edges (arcs between consecutive nodes)
    for i in range(n):
        j = (i + 1) % n
        ax.annotate(
            "",
            xy=positions[j],
            xytext=positions[i],
            arrowprops={
                "arrowstyle": "-",
                "color": "0.6",
                "linewidth": 0.8,
            },
        )

    # Draw nodes
    for i, (x, y) in enumerate(positions):
        payload = cdll_order[i]
        if new_node_payload is not None and payload == new_node_payload:
            color = NEW_NODE_COLOR
        elif i == primary_ptr_idx and i == secondary_ptr_idx:
            color = "#AA77BB"  # Blend for overlap
        elif i == primary_ptr_idx:
            color = PRIMARY_COLOR
        elif i == secondary_ptr_idx:
            color = SECONDARY_COLOR
        else:
            color = _DEFAULT_NODE_COLOR

        circle = plt.Circle(
            (x, y), node_radius, facecolor=color, edgecolor="0.3", linewidth=0.8, zorder=3
        )
        ax.add_patch(circle)
        ax.text(
            x,
            y,
            str(payload),
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            zorder=4,
        )

    # Draw pointer arrows from outside the ring
    arrow_radius = radius + 0.35
    _draw_pointer_arrow(
        ax,
        positions[primary_ptr_idx],
        angles[primary_ptr_idx],
        arrow_radius,
        node_radius,
        PRIMARY_COLOR,
        "\u03c0",
    )

    if secondary_ptr_idx != primary_ptr_idx:
        _draw_pointer_arrow(
            ax,
            positions[secondary_ptr_idx],
            angles[secondary_ptr_idx],
            arrow_radius,
            node_radius,
            SECONDARY_COLOR,
            "\u03c3",
        )
    else:
        # Overlap: offset secondary arrow slightly
        offset_angle = angles[primary_ptr_idx] + 0.3
        _draw_pointer_arrow(
            ax,
            positions[secondary_ptr_idx],
            offset_angle,
            arrow_radius,
            node_radius,
            SECONDARY_COLOR,
            "\u03c3",
        )

    if show_legend:
        handles = get_legend_handles(include_new_node=new_node_payload is not None)
        ax.legend(handles=handles, loc="lower center", fontsize=5, ncol=3, framealpha=0.7)

    # Margin accounts for pointer arrows + labels outside ring
    margin = arrow_radius + 0.45
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_pointer_arrow(
    ax: plt.Axes,
    target_pos: tuple[float, float],
    angle: float,
    arrow_radius: float,
    node_radius: float,
    color: str,
    label: str,
) -> None:
    """Draw a labeled pointer arrow from outside the ring toward a node."""
    start_x = arrow_radius * np.cos(angle)
    start_y = arrow_radius * np.sin(angle)

    ax.annotate(
        "",
        xy=target_pos,
        xytext=(start_x, start_y),
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linewidth": 1.5,
            "shrinkA": 0,
            "shrinkB": node_radius * 72,  # Convert to points
        },
    )
    # Label at arrow origin (offset from arrowhead start)
    label_x = (arrow_radius + 0.25) * np.cos(angle)
    label_y = (arrow_radius + 0.25) * np.sin(angle)
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
