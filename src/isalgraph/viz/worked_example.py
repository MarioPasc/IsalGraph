"""The two worked-example figures: S2G decoding and G2S encoding.

These are the figures the submitted manuscript lost. ``fig_algorithm_overview.pdf``
is still in the article source, commented out at ``methodology.tex:379``
with the note *"Figure commented out to meet the 35-page limit"*. This
module rebuilds it as **two** panels that answer different questions,
because in the submitted version they answered the same one.

One index for both panels
-------------------------
S2G consumes one symbol per step; G2S emits a whole *group* per pass of
its outer loop -- the movement instructions for a displacement, then the
operation. Indexing the two panels by their own natural units would give
them different column counts and nothing to compare.

Both are therefore indexed by the **encoder's group boundaries**. Column
*k* of the G2S panel is the encoder's *k*-th pass; column *k* of the S2G
panel is the decoder's state once it has consumed the same group. The two
panels then have the same number of columns, the same milestones and the
same layout, and a reader can hold one against the other.

For the running example the groups are ``V | V | V | nv | PC | PV``: six
columns, with no step omitted from either panel.

What the G2S panel shows that the S2G panel cannot
--------------------------------------------------
Inside one pass, G2S searches: it walks the displacement pairs
:math:`\\mathcal{P}(M)` in increasing cost and runs the ``V``
:math:`\\succ` ``v`` :math:`\\succ` ``C`` :math:`\\succ` ``c`` cascade at
each until one applies. How many pairs it rejected, which one won and
which cascade level it reached appear in each column's caption. Building
this panel from ``GraphToString.run_with_trace`` would have shown none of
it -- that method replays the finished string rather than tracing the
encoder, so its states are the S2G panel's with the grey mask flipped.
:mod:`isalgraph.viz.encoder_trace` exists to avoid exactly that, and is
pinned to the frozen encoder by test.

Ghosting is a conservation argument
-----------------------------------
Ink is conserved between the strip and the graph, and the direction of
the transfer is the direction of the algorithm:

* **S2G** turns a string into a graph. The strip starts solid and empties
  as symbols are consumed; the graph starts ghosted and fills in.
* **G2S** turns a graph into a string. The graph starts solid and empties
  as structure is captured; the strip starts ghosted and fills in.

So each panel shows one thing draining and the other filling, and the two
panels run in opposite directions. That is the property worth showing --
the round trip -- and a reader can check it at a glance instead of being
told. Inverting it, so both panels fill in, makes the two figures look
like the same algorithm drawn twice.

A ghost is a **white** disc with a dashed grey outline, the sibling
projects' convention: it recedes by carrying no ink rather than by
carrying grey ink, which is what survives at column scale. The element
the current step moved between the two representations carries an accent
halo in both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from isalgraph.core.sparse_graph import SparseGraph
from isalgraph.core.string_to_graph import StringToGraph
from isalgraph.core.trace import AlgorithmTrace, Edge, StepSnapshot, graph_edges
from isalgraph.types import NodeId
from isalgraph.viz.base import Position
from isalgraph.viz.cdll_view import draw_cdll_ring
from isalgraph.viz.encoder_trace import REJECTED, EncoderIteration, EncoderTrace
from isalgraph.viz.instruction_view import draw_instruction_strip
from isalgraph.viz.style import (
    ACCENT_COLOR,
    ACTIVE_ALPHA,
    GHOST_DASH,
    GHOST_EDGE_COLOR,
    GHOST_FACE,
    GHOST_TEXT_COLOR,
    POINTER_PALETTE,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.gridspec import SubplotSpec
else:
    Axes = Any
    Figure = Any
    SubplotSpec = Any


class WorkedExampleError(ValueError):
    """Raised when a worked-example figure is asked for an impossible step."""


@dataclass(frozen=True)
class WorkedExampleLayout:
    """Printed geometry, in true inches and points.

    Every number here is what lands on the page, so the figure can be
    checked against the column measure without a mental scaling factor.
    The idiom is lifted from ``IsalSR/viz/algorithm_trace.py``.

    Both figures are built from one instance, which is what makes their
    layouts identical rather than merely similar.

    Args:
        fig_width: Total width. Defaults to the IEEE text measure.
        title_height: Band above each column for its step label.
        ring_height: Band for the CDLL ring.
        strip_height: Band for the instruction strip.
        graph_height: Band for the graph panel.
        caption_height: Band below each column for its two caption lines.
        wspace: Column gap, as a fraction of the column width.
        fs_title: Column-title point size.
        fs_caption: Column-caption point size.
        node_radius: Graph node radius, in graph-panel axis units.
        ring_node_radius: CDLL node radius, in ring axis units. Sized so
            the discs print at roughly the graph's diameter, which is
            what lets the labels sit inside them at column scale.
    """

    fig_width: float = 7.0
    title_height: float = 0.22
    ring_height: float = 0.94
    strip_height: float = 0.40
    graph_height: float = 0.88
    caption_height: float = 0.40
    wspace: float = 0.14
    fs_title: float = 7.5
    fs_caption: float = 6.0
    node_radius: float = 0.17
    ring_node_radius: float = 0.26

    @property
    def row_heights(self) -> tuple[float, float, float, float]:
        """The four per-column band heights, top to bottom."""
        return (self.title_height, self.ring_height, self.strip_height, self.graph_height)

    @property
    def figsize(self) -> tuple[float, float]:
        """The ``(width, height)`` inch pair for ``plt.figure``."""
        return (self.fig_width, sum(self.row_heights) + self.caption_height)


#: Pinned coordinates for the running example. An explicit layout rather
#: than a spring layout: it is reproducible without NetworkX, it draws the
#: triangle, the path and the pendant as three visually separate features,
#: and no edge crosses another.
RUNNING_EXAMPLE_POSITIONS: dict[NodeId, Position] = {
    0: (0.00, 0.05),
    1: (0.68, 0.68),
    3: (0.68, -0.62),
    5: (1.52, -0.98),
    2: (-0.86, 0.24),
    4: (-1.62, -0.24),
}


@dataclass(frozen=True)
class ExampleColumn:
    """Everything one column of either figure draws.

    Both builders reduce their own trace to a tuple of these, and a
    single renderer draws them. The two figures are then identical in
    layout by construction, rather than by two builders being kept in
    step by hand.

    Args:
        title: Column heading.
        ring_order: CDLL contents in forward circular order.
        primary: Graph node under the primary pointer.
        secondary: Graph node under the secondary pointer.
        consumed: Instruction cells to draw solid, counted from the left.
        span: Half-open ``[lo, hi)`` range of cells this column executes.
        present_nodes: Graph nodes drawn solid.
        present_edges: Graph edges drawn solid.
        accent_nodes: Graph nodes carrying the created-this-step halo.
        accent_edges: Graph edges drawn in the accent colour.
        ring_accent: Ring payload created this step, if any.
        strip_solid_side: ``"suffix"`` for S2G, whose strip drains as
            symbols are consumed, and ``"prefix"`` for G2S, whose strip
            fills as they are emitted. Paired with the graph panel's own
            direction, this is what makes the ink conservation visible.
        caption: Up to two short lines drawn under the column.
    """

    title: str
    ring_order: tuple[NodeId, ...]
    primary: NodeId
    secondary: NodeId
    consumed: int
    span: tuple[int, int]
    present_nodes: frozenset[NodeId]
    present_edges: frozenset[Edge]
    accent_nodes: frozenset[NodeId] = frozenset()
    accent_edges: frozenset[Edge] = frozenset()
    ring_accent: NodeId | None = None
    strip_solid_side: str = "prefix"
    caption: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Atomic panels
# ---------------------------------------------------------------------------


def _resolve_positions(
    graph: SparseGraph,
    positions: dict[NodeId, Position] | None,
) -> dict[NodeId, Position]:
    """Return pinned *positions*, or compute a layout for *graph*."""
    if positions is not None:
        return positions
    from isalgraph.viz.layout import cdll_ring_positions, compact_graph_layout

    try:
        return compact_graph_layout(graph)
    except ImportError:
        return cdll_ring_positions(tuple(range(graph.node_count())))


def draw_state_graph(  # noqa: PLR0913  -- one parameter per element state
    ax: Axes,
    graph: SparseGraph,
    positions: dict[NodeId, Position],
    *,
    present_nodes: frozenset[NodeId],
    present_edges: frozenset[Edge],
    accent_nodes: frozenset[NodeId] = frozenset(),
    accent_edges: frozenset[Edge] = frozenset(),
    primary_node: NodeId | None = None,
    secondary_node: NodeId | None = None,
    node_radius: float = 0.17,
    label_fontsize: float = 6.0,
) -> None:
    """Draw one graph state with ghost, present and accent elements.

    The whole graph is always drawn; ``present_*`` decides what is solid
    and everything else is ghosted. Drawing the target structure in every
    frame is what lets a reader see where the algorithm is *going*, which
    a panel showing only what exists cannot convey.

    Args:
        ax: Target axes.
        graph: The graph whose full structure is drawn.
        positions: Node coordinates, pinned across every column.
        present_nodes: Nodes drawn solid.
        present_edges: Edges drawn solid.
        accent_nodes: Nodes carrying the created-this-step halo.
        accent_edges: Edges drawn in the accent colour.
        primary_node: Node under the primary pointer, ringed in
            ``POINTER_PALETTE[0]``. This is the link between the CDLL
            ring above and the graph below, and it is the single hardest
            correspondence for a reader to make unaided.
        secondary_node: Node under the secondary pointer.
        node_radius: Node radius in axis units.
        label_fontsize: Node-label point size.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    for edge in graph_edges(graph):
        u, v = edge
        if u == v or u not in positions or v not in positions:
            continue
        solid = edge in present_edges
        accent = edge in accent_edges
        color = ACCENT_COLOR if accent else ("#4A4A4A" if solid else GHOST_EDGE_COLOR)
        ax.add_line(
            Line2D(
                [positions[u][0], positions[v][0]],
                [positions[u][1], positions[v][1]],
                color=color,
                lw=1.7 if accent else (1.25 if solid else 0.9),
                linestyle="-" if solid else GHOST_DASH,
                alpha=ACTIVE_ALPHA if solid else 1.0,
                zorder=2 if accent else 1,
                solid_capstyle="round",
            )
        )

    for node, (x, y) in positions.items():
        solid = node in present_nodes
        if node in accent_nodes:
            ax.add_patch(
                Circle(
                    (x, y),
                    node_radius * 1.55,
                    facecolor="none",
                    edgecolor=ACCENT_COLOR,
                    lw=1.6,
                    zorder=3,
                )
            )
        for pointer, color, scale in (
            (primary_node, POINTER_PALETTE[0], 1.26),
            (secondary_node, POINTER_PALETTE[1], 1.40),
        ):
            if pointer == node:
                ax.add_patch(
                    Circle(
                        (x, y),
                        node_radius * scale,
                        facecolor="none",
                        edgecolor=color,
                        lw=1.3,
                        zorder=4,
                    )
                )
        ax.add_patch(
            Circle(
                (x, y),
                node_radius,
                facecolor="#EDF1F6" if solid else GHOST_FACE,
                edgecolor="#3C4450" if solid else GHOST_EDGE_COLOR,
                lw=1.0,
                linestyle="-" if solid else GHOST_DASH,
                zorder=5,
            )
        )
        ax.text(
            x,
            y,
            str(node),
            ha="center",
            va="center",
            fontsize=label_fontsize,
            color="#111111" if solid else GHOST_TEXT_COLOR,
            zorder=6,
        )

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    pad = node_radius * 2.4
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.axis("off")


# ---------------------------------------------------------------------------
# The shared renderer
# ---------------------------------------------------------------------------


def draw_columns(
    columns: tuple[ExampleColumn, ...],
    graph: SparseGraph,
    instructions: str,
    *,
    positions: dict[NodeId, Position] | None = None,
    layout: WorkedExampleLayout | None = None,
) -> Figure:
    """Render *columns* as one filmstrip.

    This is the only function that lays anything out, so the S2G and G2S
    figures cannot drift apart geometrically.

    Args:
        columns: The columns to draw, left to right.
        graph: The graph drawn in every column, in full.
        instructions: The complete instruction string.
        positions: Pinned node coordinates. Computed if absent.
        layout: Printed geometry.

    Returns:
        The created figure. The caller owns it and must close it.

    Raises:
        WorkedExampleError: If *columns* is empty.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    if not columns:
        raise WorkedExampleError("at least one column is required")

    lay = layout or WorkedExampleLayout()
    pos = _resolve_positions(graph, positions)
    fig = plt.figure(figsize=lay.figsize)
    gs = GridSpec(
        1,
        len(columns),
        figure=fig,
        wspace=lay.wspace,
        left=0.012,
        right=0.988,
        top=0.985,
        bottom=lay.caption_height / lay.figsize[1],
    )

    for index, column in enumerate(columns):
        sub = gs[0, index].subgridspec(4, 1, height_ratios=list(lay.row_heights), hspace=0.18)
        ax_title = fig.add_subplot(sub[0])
        ax_title.axis("off")
        ax_title.text(
            0.5,
            0.25,
            column.title,
            ha="center",
            va="center",
            fontsize=lay.fs_title,
            transform=ax_title.transAxes,
        )
        draw_cdll_ring(
            fig.add_subplot(sub[1]),
            list(column.ring_order),
            _ring_index(column.ring_order, column.primary),
            _ring_index(column.ring_order, column.secondary),
            new_node_payload=column.ring_accent,
            new_node_color=ACCENT_COLOR,
            node_radius=lay.ring_node_radius,
        )
        draw_instruction_strip(
            fig.add_subplot(sub[2]),
            instructions,
            current_idx=column.consumed,
            solid_side=column.strip_solid_side,
            executing_span=column.span,
            axis_width_inches=lay.fig_width / len(columns),
        )
        draw_state_graph(
            fig.add_subplot(sub[3]),
            graph,
            pos,
            present_nodes=column.present_nodes,
            present_edges=column.present_edges,
            accent_nodes=column.accent_nodes,
            accent_edges=column.accent_edges,
            primary_node=column.primary,
            secondary_node=column.secondary,
            node_radius=lay.node_radius,
        )
        _caption(fig, gs[0, index], column.caption, lay)

    return fig


def _ring_index(order: tuple[NodeId, ...], node: NodeId) -> int:
    """Return the ring position of *node*, or ``0`` when absent."""
    try:
        return order.index(node)
    except ValueError:
        return 0


def _caption(
    fig: Figure,
    spec: SubplotSpec,
    lines: tuple[str, ...],
    lay: WorkedExampleLayout,
) -> None:
    """Write a column's caption lines under it."""
    if not lines:
        return
    box = spec.get_position(fig)
    x = (box.x0 + box.x1) / 2.0
    line_height = (lay.fs_caption + 2.4) / 72.0 / lay.figsize[1]
    y = box.y0 - line_height * 1.2
    for line in lines:
        fig.text(x, y, line, ha="center", va="center", fontsize=lay.fs_caption, color="#333333")
        y -= line_height


# ---------------------------------------------------------------------------
# Column construction
# ---------------------------------------------------------------------------


def group_spans(groups: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    """Return the half-open cell ranges *groups* occupy in their string.

    Args:
        groups: The symbol groups, in order.

    Returns:
        One ``(lo, hi)`` pair per group.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for group in groups:
        spans.append((cursor, cursor + len(group)))
        cursor += len(group)
    return tuple(spans)


def _pointer_line(primary: NodeId, secondary: NodeId) -> str:
    """Return the caption line naming both pointer positions."""
    return f"π = {primary},   σ = {secondary}"


def _span_edges(
    snapshots: tuple[StepSnapshot, ...],
    lo: int,
    hi: int,
) -> tuple[tuple[str, Edge], ...]:
    """Return the ``(instruction, edge)`` pairs created within ``[lo, hi)``."""
    out: list[tuple[str, Edge]] = []
    for step in range(lo + 1, hi + 1):
        snap = snapshots[step]
        if snap.created_edge is not None and snap.instruction is not None:
            out.append((snap.instruction, snap.created_edge))
    return tuple(out)


def _s2g_effect(snapshots: tuple[StepSnapshot, ...], lo: int, hi: int) -> str:
    """Return the one-line effect of the symbols in ``[lo, hi)``."""
    created = _span_edges(snapshots, lo, hi)
    if not created:
        return "pointer movement only"
    instruction, edge = created[-1]
    if instruction in ("V", "v"):
        return f"add node {max(edge)}, edge {edge[0]}–{edge[1]}"
    return f"add edge {edge[0]}–{edge[1]}"


def s2g_columns(
    trace: AlgorithmTrace,
    groups: tuple[str, ...],
) -> tuple[ExampleColumn, ...]:
    """Reduce an ``"s2g"`` trace to one column per symbol group.

    Args:
        trace: The decoding trace.
        groups: The encoder's symbol groups, which set the milestones.

    Returns:
        One column per group.

    Raises:
        WorkedExampleError: If the groups do not span the whole string.
    """
    spans = group_spans(groups)
    if spans and spans[-1][1] != len(trace.instruction_string):
        raise WorkedExampleError(
            f"groups cover {spans[-1][1]} symbols but the string has "
            f"{len(trace.instruction_string)}"
        )

    columns: list[ExampleColumn] = []
    for index, (lo, hi) in enumerate(spans):
        snap = trace.snapshots[hi]
        created = _span_edges(trace.snapshots, lo, hi)
        accent_nodes = frozenset(max(edge) for instr, edge in created if instr in ("V", "v"))
        columns.append(
            ExampleColumn(
                title=f"Step {index + 1}",
                ring_order=snap.cdll_node_order,
                primary=snap.primary_node,
                secondary=snap.secondary_node,
                consumed=hi,
                span=(lo, hi),
                present_nodes=frozenset(snap.active_nodes),
                present_edges=frozenset(snap.active_edges),
                accent_nodes=accent_nodes,
                accent_edges=frozenset(edge for _, edge in created),
                ring_accent=max(accent_nodes) if accent_nodes else None,
                strip_solid_side="suffix",
                caption=(
                    _s2g_effect(trace.snapshots, lo, hi),
                    _pointer_line(snap.primary_node, snap.secondary_node),
                ),
            )
        )
    return tuple(columns)


def _g2s_search_line(iteration: EncoderIteration) -> str:
    """Return the caption line describing the pair search this pass ran.

    Names how many displacement pairs were rejected before one applied,
    the winning pair, and the cascade level it reached. That is the whole
    of what the encoder does inside a pass, and none of it is recoverable
    from the emitted string alone.
    """
    probe = iteration.selected
    a, b = probe.displacement
    rejected = sum(1 for p in iteration.probes if p.verdict == REJECTED)
    prefix = f"{rejected} rejected, " if rejected else ""
    return f"{prefix}({a:+d},{b:+d}) → {probe.verdict}"


def _g2s_effect_line(iteration: EncoderIteration) -> str:
    """Return the caption line naming what this pass captured."""
    if iteration.created_node is not None and iteration.created_edge is not None:
        u, v = iteration.created_edge
        return f"take node {iteration.created_node}, edge {u}–{v}"
    if iteration.created_edge is not None:
        u, v = iteration.created_edge
        return f"take edge {u}–{v}"
    return "nothing captured"


def g2s_columns(trace: EncoderTrace) -> tuple[ExampleColumn, ...]:
    """Reduce an encoder trace to one column per outer-loop pass.

    The graph panel shows what is **not yet** encoded, which is the
    inverse of the S2G panel's rule. G2S consumes the graph and produces
    the string, so the graph is what drains; drawing the captured part
    solid instead would make the encoder look like a decoder.

    Args:
        trace: The instrumented encoder trace.

    Returns:
        One column per iteration.
    """
    spans = group_spans(trace.groups)
    all_nodes = frozenset(range(trace.graph.node_count()))
    all_edges = frozenset(graph_edges(trace.graph))
    columns: list[ExampleColumn] = []
    for iteration, (lo, hi) in zip(trace.iterations, spans, strict=True):
        created_node = iteration.created_node
        columns.append(
            ExampleColumn(
                title=f"Step {iteration.index + 1}",
                ring_order=iteration.ring_after,
                primary=iteration.primary_after,
                secondary=iteration.secondary_after,
                consumed=hi,
                span=(lo, hi),
                present_nodes=all_nodes - frozenset(iteration.captured_nodes_after),
                present_edges=all_edges - frozenset(iteration.captured_edges_after),
                accent_nodes=frozenset() if created_node is None else frozenset({created_node}),
                accent_edges=(
                    frozenset()
                    if iteration.created_edge is None
                    else frozenset({iteration.created_edge})
                ),
                ring_accent=created_node,
                strip_solid_side="prefix",
                caption=(_g2s_search_line(iteration), _g2s_effect_line(iteration)),
            )
        )
    return tuple(columns)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def s2g_worked_example_figure(
    trace: AlgorithmTrace,
    graph: SparseGraph,
    *,
    groups: tuple[str, ...],
    positions: dict[NodeId, Position] | None = None,
    layout: WorkedExampleLayout | None = None,
) -> Figure:
    """Build the S2G worked-example filmstrip.

    Args:
        trace: An ``"s2g"`` trace.
        graph: The decoded graph, drawn in full in every column.
        groups: The encoder's symbol groups, which set the column
            boundaries so this panel lines up with the G2S one.
        positions: Pinned node coordinates. Computed if absent.
        layout: Printed geometry.

    Returns:
        The created figure. The caller owns it and must close it.
    """
    return draw_columns(
        s2g_columns(trace, groups),
        graph,
        trace.instruction_string,
        positions=positions,
        layout=layout,
    )


def g2s_worked_example_figure(
    trace: EncoderTrace,
    *,
    positions: dict[NodeId, Position] | None = None,
    layout: WorkedExampleLayout | None = None,
) -> Figure:
    """Build the G2S worked-example filmstrip.

    Args:
        trace: An instrumented encoder trace.
        positions: Pinned node coordinates. Computed if absent.
        layout: Printed geometry.

    Returns:
        The created figure. The caller owns it and must close it.
    """
    return draw_columns(
        g2s_columns(trace),
        trace.graph,
        trace.instruction_string,
        positions=positions,
        layout=layout,
    )


def decode_trace(
    instructions: str, *, directed: bool = False
) -> tuple[SparseGraph, AlgorithmTrace]:
    """Return the graph and ``"s2g"`` trace for *instructions*.

    Args:
        instructions: The instruction string to decode.
        directed: Whether to build a directed graph.

    Returns:
        The decoded graph and its trace.
    """
    return StringToGraph(instructions, directed_graph=directed).run_with_trace()


__all__ = [
    "RUNNING_EXAMPLE_POSITIONS",
    "ExampleColumn",
    "WorkedExampleError",
    "WorkedExampleLayout",
    "decode_trace",
    "draw_columns",
    "draw_state_graph",
    "g2s_columns",
    "g2s_worked_example_figure",
    "group_spans",
    "s2g_columns",
    "s2g_worked_example_figure",
]
