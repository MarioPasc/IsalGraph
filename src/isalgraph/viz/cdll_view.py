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
    new_node_color: str | None = None,
    radius: float = 1.0,
    node_radius: float = 0.15,
    show_legend: bool = False,
    label_fontsize: float = 7.0,
    label_dy: float = 0.0,
    pointer_fontsize: float = 8.0,
    pointer_scale: float = 9.0,
    pointer_lw: float = 1.4,
    show_pointer_arrows: bool = True,
    arrow_gap: float = 0.60,
    label_gap: float = 0.26,
    margin_pad: float = 0.45,
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
        new_node_color: Colour for that highlight. Defaults to
            :data:`NEW_NODE_COLOR`, which the benchmark figure scripts
            already use; the worked-example figures pass the accent
            colour instead so the ring and the graph panel mark "created
            by this step" in one colour rather than two.
        radius: Ring radius.
        node_radius: Node circle radius.
        show_legend: Draw a per-panel legend instead of relying on a
            shared figure-level one.
        label_fontsize: Node-label point size.
        label_dy: Downward shift of the node label, in node-radius units.
            ``va="center"`` centres a digit on its font's full bounding
            box, descender space included, and a digit has no descender,
            so it prints visibly high inside its disc. A shift of about
            ``0.12`` puts it on the disc's optical centre. Zero by
            default, which is what every committed figure was drawn with.
        pointer_fontsize: Point size of the π and σ glyphs.
        pointer_scale: Arrowhead ``mutation_scale``, in points.
        pointer_lw: Arrow line width, in points.
        show_pointer_arrows: Draw the two pointer arrows. Turning them
            off keeps the π and σ glyphs and moves them in to where the
            arrow tails were. The glyph sits on the same radius as its
            arrow, so on a small ring the two compete for one band of
            blank margin and a glyph large enough to read overlaps the
            shaft it is labelling; dropping the arrow spends that band on
            the glyph instead. The pointed-at disc is already filled in
            the pointer's own colour, so nothing is lost but the shaft.
        arrow_gap: Distance from the node discs out to the arrow tails,
            in ring-radius units. Ignored when *show_pointer_arrows* is
            off, which has no tails.
        label_gap: Distance from the arrow tail out to the π/σ glyph.
        margin_pad: Blank ring-radius units beyond the arrow tails. Must
            exceed *label_gap*, or the glyphs fall outside the axes.

    The size and gap parameters exist because the ring's geometry scales
    with the axes while its annotations do not: point sizes are absolute,
    so the defaults -- correct on a panel-sized ring -- draw arrowheads
    wider than the node discs once the ring is dropped into a half-inch
    row. Their defaults are the sizes every committed figure was rendered
    at, so passing nothing reproduces those figures exactly.
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
            color = new_node_color if new_node_color is not None else NEW_NODE_COLOR
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
            x,
            y - label_dy * node_radius,
            str(payload),
            ha="center",
            va="center",
            fontsize=label_fontsize,
            fontweight="bold",
            zorder=4,
        )

    # The arrow tail must clear the node disc with room left over, or the
    # head is all that survives: the old offset of 0.35 axis units was
    # shorter than the point-space shrink applied at the head, so the body
    # was drawn with negative length and disappeared.
    # With the arrows off there are no tails, so the glyphs move in to
    # where the tails would have been and the whole arrow_gap band is
    # spent on the glyph instead.
    arrow_radius = radius + node_radius + (arrow_gap if show_pointer_arrows else 0.0)
    _draw_pointer_arrow(
        ax,
        angles[primary_ptr_idx],
        radius,
        arrow_radius,
        node_radius,
        PRIMARY_COLOR,
        "π",
        fontsize=pointer_fontsize,
        mutation_scale=pointer_scale,
        linewidth=pointer_lw,
        label_gap=label_gap,
        draw_arrow=show_pointer_arrows,
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
        secondary_angle,
        radius,
        arrow_radius,
        node_radius,
        SECONDARY_COLOR,
        "σ",
        fontsize=pointer_fontsize,
        mutation_scale=pointer_scale,
        linewidth=pointer_lw,
        label_gap=label_gap,
        draw_arrow=show_pointer_arrows,
    )

    if show_legend:
        ax.legend(
            handles=get_legend_handles(include_new_node=new_node_payload is not None),
            loc="lower center",
            fontsize=5,
            ncol=3,
            framealpha=0.7,
        )

    margin = arrow_radius + margin_pad
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_pointer_arrow(  # noqa: PLR0913  -- polar geometry needs its terms
    ax: Axes,
    angle: float,
    ring_radius: float,
    arrow_radius: float,
    node_radius: float,
    color: str,
    label: str,
    *,
    fontsize: float = 8.0,
    mutation_scale: float = 9.0,
    linewidth: float = 1.4,
    label_gap: float = 0.26,
    draw_arrow: bool = True,
) -> None:
    """Draw one labelled pointer arrow from outside the ring toward a node.

    Both endpoints are computed in data coordinates and no shrink is
    applied. Shrink is measured in points, so mixing it with a tail
    offset measured in axis units makes the visible body length depend on
    the rendered figure size -- which is how the body came to vanish at
    column scale while looking correct at the default size.

    Args:
        ax: Target axes.
        angle: Polar angle of the target node, in radians.
        ring_radius: Distance from the ring centre to the node centres.
        arrow_radius: Distance from the ring centre to the arrow tail.
        node_radius: Node disc radius; the head stops just outside it.
        color: Arrow and label colour.
        label: Text drawn beyond the tail.
        fontsize: Label point size.
        mutation_scale: Arrowhead size, in points.
        linewidth: Arrow line width, in points.
        label_gap: Distance from the tail out to the label.
        draw_arrow: Draw the shaft and head. When off, only the label is
            drawn, at the same place it would otherwise have gone.
    """
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    if draw_arrow:
        tip = ring_radius + node_radius * 1.10
        ax.annotate(
            "",
            xy=(tip * cos_a, tip * sin_a),
            xytext=(arrow_radius * cos_a, arrow_radius * sin_a),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "linewidth": linewidth,
                "mutation_scale": mutation_scale,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )
    label_x = (arrow_radius + label_gap) * cos_a
    label_y = (arrow_radius + label_gap) * sin_a
    ax.text(
        label_x,
        label_y,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=color,
    )


def draw_cdll_ring_for_snapshot(
    ax: Axes,
    snapshot: StepSnapshot,
    *,
    highlight_new_node: bool = True,
    new_node_color: str | None = None,
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
        new_node_color: Colour for that highlight; see
            :func:`draw_cdll_ring`.
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
        new_node_color=new_node_color,
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
