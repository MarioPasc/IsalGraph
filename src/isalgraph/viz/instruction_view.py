"""Render the instruction-string strip.

The strip is a row of cells, one per symbol of the instruction string.
Each cell carries two colour channels:

* the **fill** is :data:`~isalgraph.viz.style.INSTRUCTION_PALETTE`, so hue
  identifies the operation (move / insert / connect / no-op);
* the **stroke** is
  :func:`~isalgraph.viz.style.pointer_accent`, so the outline identifies
  which pointer acted, in the same two colours the CDLL ring uses for
  its arrows.

Cells outside the active range are drawn grey. Which range counts as
active depends on direction, and is documented on
:func:`draw_instruction_strip`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isalgraph.viz.style import (
    ACTIVE_ALPHA,
    GRAYED_ALPHA,
    GRAYED_EDGE,
    GRAYED_FACE,
    INSTRUCTION_OPERATION,
    INSTRUCTION_PALETTE,
    POINTER_PALETTE,
    color_for_instruction,
    pointer_accent,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.patches import Patch
else:
    Axes = Any
    Patch = Any


def _auto_fontsize(n_cells: int, axis_width_inches: float) -> float:
    """Pick a label size that fits the cell, clamped to ``[3.0, 7.5]`` points.

    Per-cell width in points is ``axis_width_inches * 72 / n_cells``; a
    rotated glyph needs about one character height of horizontal room.
    """
    if n_cells <= 0:
        return 7.5
    return max(3.0, min(7.5, axis_width_inches * 72.0 / n_cells * 0.80))


def draw_instruction_strip(
    ax: Axes,
    instructions: str,
    *,
    current_idx: int,
    cell_width: float = 1.05,
    cell_height: float = 1.1,
    show_labels: bool = True,
    label_rotation: float = 0.0,
    axis_width_inches: float | None = None,
    label_fontsize: float | None = None,
    direction: str = "s2g",
    accent_lw: float = 1.6,
) -> None:
    """Draw the instruction strip on *ax*.

    Args:
        ax: Target axes.
        instructions: The full instruction string.
        current_idx: Number of symbols consumed so far.
        cell_width: Cell width in axis units.
        cell_height: Cell height in axis units.
        show_labels: Draw the symbol inside each cell.
        label_rotation: Label rotation in degrees.
        axis_width_inches: Physical strip width; enables auto font sizing.
        label_fontsize: Explicit font size, overriding auto sizing.
        direction: ``"s2g"`` or ``"g2s"``. Under ``"g2s"`` the algorithm
            *emits* symbols, so cells before *current_idx* are the ones
            produced and therefore active. Under ``"s2g"`` it *consumes*
            them, so cells from *current_idx* onward are still pending
            and are the ones drawn in colour, fading as their structure
            materialises in the graph panel.
        accent_lw: Stroke width of the pointer-accent outline.
    """
    from matplotlib.patches import FancyBboxPatch

    n = len(instructions)
    if label_fontsize is None:
        label_fontsize = 7.0 if axis_width_inches is None else _auto_fontsize(n, axis_width_inches)

    if n == 0:
        ax.text(
            0.5,
            0.5,
            "(empty)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#888888",
            fontsize=7,
        )
        _strip_axes(ax, 0, cell_width, cell_height)
        return

    for i, symbol in enumerate(instructions):
        x = i * cell_width
        is_active = i < current_idx if direction == "g2s" else i >= current_idx
        face = color_for_instruction(symbol) if is_active else GRAYED_FACE
        stroke = pointer_accent(symbol) if is_active else GRAYED_EDGE
        alpha = ACTIVE_ALPHA if is_active else GRAYED_ALPHA
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.05, 0.05),
                cell_width - 0.10,
                cell_height - 0.10,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=face,
                edgecolor=stroke,
                lw=accent_lw if is_active else 0.6,
                alpha=alpha,
            )
        )
        if show_labels:
            ax.text(
                x + cell_width / 2,
                cell_height / 2,
                symbol,
                ha="center",
                va="center",
                fontsize=label_fontsize,
                fontfamily="monospace",
                color="#111111" if is_active else "#666666",
                rotation=label_rotation,
                zorder=3,
            )

    _strip_axes(ax, n, cell_width, cell_height)


def _strip_axes(ax: Axes, n: int, cell_width: float, cell_height: float) -> None:
    """Set strip limits and hide ticks and spines."""
    ax.set_xlim(-0.1, max(n, 1) * cell_width + 0.1)
    ax.set_ylim(-0.05, cell_height + 0.05)
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def instruction_legend_handles(*, include_pointers: bool = True) -> list[Any]:
    """Return legend handles explaining both colour channels.

    One patch per operation class (hue channel) plus, optionally, two
    line handles for the primary and secondary pointer accents (stroke
    channel).

    Args:
        include_pointers: Append the two pointer-accent handles.

    Returns:
        Handles suitable for ``ax.legend`` or ``fig.legend``.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch as MplPatch

    labels = {
        "move": "N P n p  move",
        "insert": "V v  insert node",
        "connect": "C c  connect",
        "noop": "W  no-op",
    }
    handles: list[Any] = []
    for op, text in labels.items():
        representative = next(k for k, v in INSTRUCTION_OPERATION.items() if v == op)
        handles.append(MplPatch(facecolor=INSTRUCTION_PALETTE[representative], label=text))
    if include_pointers:
        handles.extend(
            [
                Line2D([0], [0], color=POINTER_PALETTE[0], lw=2.0, label="π primary"),
                Line2D([0], [0], color=POINTER_PALETTE[1], lw=2.0, label="σ secondary"),
            ]
        )
    return handles


__all__ = ["draw_instruction_strip", "instruction_legend_handles"]
