"""One figure for the canonical search space and the encode that walks it.

R3.7c asks for a schematic of the canonical search space showing that
*starting nodes* and *uninserted-neighbour choices* branch, while the
displacement ordering :math:`\\mathcal{P}(M)` and the priority ``V``
:math:`\\succ` ``v`` :math:`\\succ` ``C`` :math:`\\succ` ``c`` do not.
:mod:`isalgraph.viz.search_tree` draws the first half of that sentence.
The second half -- what a *forced* step is forced by -- is only visible
inside one encoder iteration, which is what
:mod:`isalgraph.viz.worked_example` shows.

This module puts both in one landscape figure, read left to right:

``(a)``
    The complete search tree, drawn with depth on the horizontal axis so
    that every step of every path is a column. One root per starting
    node; the fan-out inside a root is the uninserted-neighbour choice.
    The leaf column carries the length of the string each path emits, so
    Definition 2.6 is readable there: take the shortest, then break ties
    lexicographically. The canonical path is boxed.

``(b)``
    That box, magnified. One row per step of the canonical path, each
    row the CDLL ring, the instruction strip and the graph state, in the
    idiom of :mod:`isalgraph.viz.worked_example`. Row *k* is the step on
    edge *k* of the boxed path, and both carry the same emitted group.

``(c)``
    One row of (b), opened up. Rows are the displacement pairs in
    :math:`\\mathcal{P}(M)` order, columns are the cascade levels in
    priority order, and no cell in the grid is a choice. The two orders
    that never branch are the two axes of the panel. The row it belongs
    to is named by a star on both, not by a second magnification: the
    zoomed row is not the last one, so a cone would cross the rows under
    it.

Everything drawn comes from :func:`~isalgraph.viz.encoder_trace.trace_encoder`
and :func:`~isalgraph.viz.search_tree.enumerate_search_tree`, both of
which are held to the frozen reference implementation by test. Nothing
here decides anything about the algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isalgraph.core.trace import graph_edges
from isalgraph.viz.encoder_trace import REJECTED, EncoderIteration, EncoderTrace, trace_encoder
from isalgraph.viz.search_tree import (
    CANONICAL_HALO,
    SearchTree,
    enumerate_search_tree,
)
from isalgraph.viz.style import (
    ACCENT_COLOR,
    GHOST_EDGE_COLOR,
    GRAYED_EDGE,
    INSTRUCTION_PALETTE,
    POINTER_PALETTE,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from isalgraph.core.sparse_graph import SparseGraph
    from isalgraph.types import NodeId
    from isalgraph.viz.base import Position

#: Step index at and below which the running example's search still
#: branches. Measured, not assumed: :func:`enumerate_search_tree` reports
#: branching factor 1 at every node of depth >= 3, so the remaining steps
#: add nodes and no leaves. Used only to shade the forced columns.
DEFAULT_BRANCH_DEPTH: int = 3

#: Fill for a leaf whose string attains the minimum length.
SHORTEST_FACE: str = "#228833"

#: Fill for a leaf that is complete but longer than the minimum.
LONGER_FACE: str = "#F2F2F2"

#: Cascade levels, in the order the encoder tries them.
CASCADE_LEVELS: tuple[str, ...] = ("V", "v", "C", "c")

#: Fill of the highlighted row and of panel (c)'s box.
ZOOM_FILL: str = "#FDF6E6"

#: Branches the triplet pruning never explores. Light enough to read as
#: backdrop rather than as a fourth kind of branch.
CUT_EDGE: str = "#E0E0E0"
CUT_FACE: str = "#FCFCFC"
CUT_TEXT: str = "#C8C8C8"

#: Outline of the magnification: the box round the canonical path, the two
#: lines that open from it, and the bracket round panel (b). Deliberately
#: neutral -- the amber is the cross-reference between the highlighted row
#: and panel (c), and two accents doing two jobs read as one.
ZOOM_EDGE: str = "#1A1A1A"


class WalkthroughError(ValueError):
    """Raised when the figure cannot be assembled from the given trace."""


#: A rectangle as ``(x0, y0, x1, y1)``.
Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class WalkthroughLayout:
    """Printed geometry, in true inches and points.

    Every number is what lands on the page. The default is a *landscape*
    canvas: the manuscript is single-column ``letterpaper`` with 4.3 cm
    top and bottom margins, so a ``sidewaysfigure`` gets 7.6 in along the
    page height and 4.72 in across it. Panel (a)'s depth axis is
    horizontal, which is what buys each of the seven steps enough width
    to carry a label.

    Args:
        fig_width: Total figure width.
        fig_height: Total figure height.
        pad: Outer margin on all four sides.
        tree_width: Width of panel (a).
        key_height: Band under panel (a) holding its key.
        frustum_width: Horizontal run of the (a) to (b) magnification.
        frustum_height: Gap between panel (b) and panel (c).
        row_height: Height of one execution row in panel (b).
        cascade_height: Height of panel (c).
        cascade_inset: Blank margin inside panel (c)'s box.
        w_step: Panel (b) step-label column width.
        w_ring: Panel (b) CDLL-ring column width.
        w_strip: Panel (b) instruction-strip column width.
        w_graph: Panel (b) graph column width.
        fs_row: Row-label point size.
        fs_cascade: Panel (c) cell and row point size.
        fs_cascade_header: Panel (c) column-header point size.
        fs_small: Annotation point size.
        fs_tree: Tree label point size.
        fs_legend: Panel (a) key point size.
        node_radius: Graph node radius, in graph-panel axis units.
        ring_node_radius: CDLL node radius, in ring axis units.
        ring_arrow_gap: Ring-radius units from the discs to the arrow tails.
        ring_label_gap: Ring-radius units from the tails to the π/σ glyphs.
        ring_margin_pad: Blank ring-radius units beyond the tails.
        tree_node_points: Panel (a) marker area, in points squared.
    """

    fig_width: float = 7.20
    fig_height: float = 4.50
    pad: float = 0.04
    tree_width: float = 2.80
    key_height: float = 0.34
    frustum_width: float = 0.32
    frustum_height: float = 0.14
    row_height: float = 0.55
    cascade_height: float = 0.96
    cascade_inset: float = 0.10
    w_step: float = 0.52
    w_ring: float = 0.62
    w_strip: float = 1.60
    w_graph: float = 1.26
    fs_row: float = 5.4
    fs_cascade: float = 5.6
    fs_cascade_header: float = 6.2
    fs_small: float = 4.8
    fs_tree: float = 5.2
    fs_legend: float = 5.6
    # The CDLL disc and the graph disc are drawn on separate equal-aspect
    # axes of the same height, so equal *printed* size means equal
    # fractions of that height. ``draw_state_graph`` pads its limits by
    # 2.4 node radii and the running example spans 1.66 units in y, so
    # 2r/(1.66 + 4.8r) must equal R/(1 + R + arrow_gap + margin_pad).
    # With R = 0.42 and the gaps below, that fixes r at 0.2924. Changing
    # either radius alone breaks the match.
    node_radius: float = 0.2924
    # 0.42 is the ceiling for a six-node ring: adjacent centres are one
    # ring-radius unit apart, so discs of radius 0.5 would touch.
    ring_node_radius: float = 0.42
    ring_arrow_gap: float = 0.22
    ring_label_gap: float = 0.40
    # Comfortably above ring_label_gap: the π and σ glyphs are centred at
    # the label gap and are half a glyph tall either side of it.
    ring_margin_pad: float = 0.56

    tree_node_points: float = 70.0

    @property
    def figsize(self) -> tuple[float, float]:
        """Return the ``(width, height)`` inch pair for ``plt.figure``."""
        return (self.fig_width, self.fig_height)

    @property
    def right_x(self) -> float:
        """Left edge of the panel (b) / (c) column, in inches."""
        return self.pad + self.tree_width + self.frustum_width

    @property
    def right_width(self) -> float:
        """Width of the panel (b) / (c) column, in inches."""
        return self.fig_width - self.pad - self.right_x


# ---------------------------------------------------------------------------
# Panel (a) -- the search space
# ---------------------------------------------------------------------------


def _horizontal_layout(tree: SearchTree) -> dict[int, tuple[float, float]]:
    """Assign ``(depth, y)`` to every tree node.

    Terminal nodes are spaced evenly on y in enumeration order, with a
    gutter between starting-node subtrees. A parent sits on its *optimal*
    child when it has one and at the mean of its children otherwise, so
    the canonical path is a straight horizontal line and the box drawn
    round it is a thin band rather than a wedge covering half the tree.

    Args:
        tree: The enumerated tree.

    Returns:
        Node index to position.
    """
    pos: dict[int, tuple[float, float]] = {}
    cursor = [0.0]

    def place(idx: int) -> float:
        node = tree.nodes[idx]
        if not node.children:
            y = -cursor[0]
            cursor[0] += 1.0
        else:
            ys = [place(c) for c in node.children]
            optimal = [pos[c][1] for c in node.children if tree.nodes[c].optimal and c in pos]
            y = optimal[0] if optimal else sum(ys) / len(ys)
        pos[idx] = (float(node.depth), y)
        return y

    for root in tree.roots:
        place(root)
        cursor[0] += 0.8
    return pos


def prune_survivors(tree: SearchTree, triplets: Sequence[tuple[int, int, int]]) -> frozenset[int]:
    """Return the tree nodes the *pruned* canonicalisation ever visits.

    Triplet pruning filters the candidate set at each ``V``/``v`` branch
    to those attaining the maximum structural triplet
    :math:`(|N_1|,|N_2|,|N_3|)`. The candidate set at a branch is exactly
    that node's children in the enumerated exhaustive tree, and the pair
    loop and the cascade are untouched by pruning, so the pruned search
    is a *subtree* of the exhaustive one and can be obtained by filtering
    rather than by a second replay.

    That claim is checkable and the caller should check it: the minimum
    over the surviving leaves must equal
    :func:`~isalgraph.core.canonical_pruned.pruned_canonical_string`.

    Args:
        tree: A fully enumerated exhaustive tree.
        triplets: Structural triplet per input-graph node, from
            :func:`~isalgraph.core.canonical_pruned.compute_structural_triplets`.

    Returns:
        Indices of the nodes reachable under pruning.
    """
    keep: set[int] = set(tree.roots)
    frontier = list(tree.roots)
    while frontier:
        node = tree.nodes[frontier.pop()]
        kids = node.children
        if not kids:
            continue
        # A candidate set exists only where every child attached a node,
        # which is what a V/v branch is. Anything else -- a forced step, a
        # C/c -- is not a choice and is not filtered.
        picked = {k: st.chosen for k in kids if (st := tree.nodes[k].step) is not None}
        choices = {k: c for k, c in picked.items() if c is not None}
        if len(choices) != len(kids) or len(kids) == 1:
            survivors = list(kids)
        else:
            best = max(triplets[c] for c in choices.values())
            survivors = [k for k, c in choices.items() if triplets[c] == best]
        keep.update(survivors)
        frontier.extend(survivors)
    return frozenset(keep)


def remark_optimal(tree: SearchTree, target: str, *, survivors: frozenset[int] | None) -> None:
    """Re-point the tree's highlighted path at *target*.

    :func:`~isalgraph.viz.search_tree.enumerate_search_tree` marks the
    path realising the *exhaustive* canonical string. The pruned figure
    highlights a different leaf, so the marking has to move, and
    :attr:`SearchTree.canonical` with it.

    Args:
        tree: The enumerated tree, modified in place.
        target: The string whose path should be highlighted.
        survivors: When given, only a leaf inside this set is eligible;
            a cut leaf carrying the same string would highlight a path
            the pruned search never walks.

    Raises:
        WalkthroughError: If no eligible leaf emits *target*.
    """
    leaves = [
        n
        for n in tree.leaves()
        if n.prefix == target and (survivors is None or n.index in survivors)
    ]
    if not leaves:
        raise WalkthroughError(f"no eligible leaf emits {target!r}")
    for node in tree.nodes:
        node.optimal = False
    index: int | None = leaves[0].index
    while index is not None:
        tree.nodes[index].optimal = True
        index = tree.nodes[index].parent
    tree.canonical = target


def _label_pruning_decision(  # noqa: PLR0913 -- one parameter per drawing input
    ax: Axes,
    tree: SearchTree,
    pos: dict[int, tuple[float, float]],
    triplets: Sequence[tuple[int, int, int]],
    survivors: frozenset[int],
    label_fontsize: float,
) -> None:
    """Write the triplets at the first pruned branch on the highlighted path.

    One branch, not all of them. The rule is the same everywhere, so a
    single worked comparison states it; labelling all eight pruned
    branches of the running example would bury the tree.
    """
    on_path = sorted(
        (n for n in tree.nodes if n.optimal and len(n.children) > 1),
        key=lambda n: n.depth,
    )
    branch = next(
        (n for n in on_path if any(c not in survivors for c in n.children)),
        None,
    )
    if branch is None:
        return
    for child_idx in branch.children:
        child = tree.nodes[child_idx]
        step = child.step
        if step is None or step.chosen is None:
            continue
        kept = child_idx in survivors
        t1, t2, t3 = triplets[step.chosen]
        x0, y0 = pos[branch.index]
        x1, y1 = pos[child_idx]
        ax.text(
            (x0 + x1) / 2.0,
            y0 + (y1 - y0) * 0.5 + (0.40 if y1 >= y0 else -0.40),
            f"({t1},{t2},{t3})",
            ha="center",
            va="center",
            fontsize=label_fontsize - 0.6,
            color="#8A6D1F" if kept else CUT_TEXT,
            fontweight="bold" if kept else "normal",
            zorder=5,
        )


def _optimal_box(tree: SearchTree, pos: dict[int, tuple[float, float]]) -> Box:
    """Return the bounding box of the canonical path, in tree data units."""
    xs = [pos[n.index][0] for n in tree.nodes if n.optimal]
    ys = [pos[n.index][1] for n in tree.nodes if n.optimal]
    if not xs:  # pragma: no cover - _mark_optimal always marks a path
        raise WalkthroughError("no node is marked optimal; the tree has no canonical path")
    return (min(xs) - 0.42, min(ys) - 0.52, max(xs) + 0.42, max(ys) + 0.52)


def draw_search_space(  # noqa: PLR0913, PLR0912, PLR0915 -- one block per visual layer
    ax: Axes,
    tree: SearchTree,
    *,
    branch_depth: int = DEFAULT_BRANCH_DEPTH,
    label_fontsize: float = 5.2,
    node_points: float = 60.0,
    box_path: bool = True,
    path_labels: bool = False,
    survivors: frozenset[int] | None = None,
    triplets: Sequence[tuple[int, int, int]] | None = None,
) -> Box:
    """Draw the whole search tree with depth on the horizontal axis.

    Every step of every path is drawn; nothing is collapsed. Branch edges
    -- a ``V``/``v`` step with more than one candidate -- are solid and
    coloured by the acting pointer, and a step with a single candidate is
    dashed grey. Past *branch_depth* every path is forced, which the
    drawing shows as a band of parallel dashed rails rather than as an
    assertion.

    Args:
        ax: Target axes.
        tree: A tree enumerated deep enough that every leaf is terminal.
        branch_depth: Last step at which the search still branches. Used
            only to shade the forced columns.
        label_fontsize: Point size for node and edge labels.
        node_points: Node marker area, in points squared.
        box_path: Outline the canonical path, for the magnification.
        path_labels: Label the canonical path's edges with the symbol
            group each step emits. Off by default: the tokens duplicate
            what panel (b)'s rows already print, and the box and its two
            magnification lines carry the correspondence on their own.
        survivors: Nodes the pruned search reaches, from
            :func:`prune_survivors`. When given, everything outside the
            set is drawn as cut, so the exhaustive tree stays visible as
            the backdrop against which the pruning is legible. A tree
            drawn from the surviving nodes alone would show a search
            space, not a pruning.
        triplets: Structural triplets, used to label the first branch on
            the highlighted path with the values the cut was decided on.
            Requires *survivors*.

    Returns:
        The canonical path's box in data coordinates, for the caller to
        anchor a magnification to.

    Raises:
        WalkthroughError: If any leaf is not terminal.
    """
    from matplotlib.patches import FancyBboxPatch, Rectangle

    unfinished = [n for n in tree.leaves() if not n.terminal]
    if unfinished:
        raise WalkthroughError(
            f"{len(unfinished)} of {len(tree.leaves())} leaves are truncated; "
            "enumerate with a depth budget large enough to complete every path"
        )

    pos = _horizontal_layout(tree)
    live = [lf for lf in tree.leaves() if survivors is None or lf.index in survivors]
    shortest = min(len(leaf.prefix) for leaf in live)
    max_depth = max(node.depth for node in tree.nodes)
    ys = [p[1] for p in pos.values()]
    y_lo, y_hi = min(ys), max(ys)

    if branch_depth < max_depth:
        ax.add_patch(
            Rectangle(
                (branch_depth + 0.5, y_lo - 1.1),
                max_depth - branch_depth,
                (y_hi - y_lo) + 2.2,
                facecolor="#F5F5F5",
                edgecolor="none",
                zorder=-2,
            )
        )
        # Under the band, not over it: the (a) to (b) magnification cone
        # leaves from the top right of this panel and would cross a label
        # placed above the columns it names.
        ax.text(
            (branch_depth + 1.0 + max_depth) / 2.0,
            y_lo - 1.55,
            f"No branching past Step {branch_depth}",
            ha="center",
            va="center",
            fontsize=label_fontsize - 0.2,
            color="0.5",
        )

    for node in tree.nodes:
        if node.parent is None:
            continue
        x0, y0 = pos[node.parent]
        x1, y1 = pos[node.index]
        step = node.step
        if step is None:  # pragma: no cover - only roots have no step
            continue
        branching = step.n_candidates > 1
        cut = survivors is not None and node.index not in survivors
        if cut:
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=CUT_EDGE,
                lw=0.5,
                ls=(0, (1.0, 1.2)),
                zorder=0,
                solid_capstyle="round",
            )
            continue
        color = POINTER_PALETTE[0 if step.op in ("V", "C") else 1] if branching else GRAYED_EDGE
        if node.optimal:
            ax.plot([x0, x1], [y0, y1], color=CANONICAL_HALO, lw=3.4, zorder=0, alpha=0.85)
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=color,
            lw=1.0 if branching else 0.7,
            ls="-" if branching else (0, (1.8, 1.4)),
            zorder=1,
            solid_capstyle="round",
        )
        if path_labels and node.optimal:
            ax.text(
                (x0 + x1) / 2.0,
                y0 + (y1 - y0) / 2.0 + 0.34,
                step.segment,
                ha="center",
                va="center",
                fontsize=label_fontsize,
                fontfamily="monospace",
                fontweight="bold",
                color=INSTRUCTION_PALETTE.get(step.op, "#333333"),
                zorder=4,
            )

    for node in tree.nodes:
        x, y = pos[node.index]
        if survivors is not None and node.index not in survivors:
            ax.scatter(
                [x],
                [y],
                s=node_points * 0.55,
                facecolor=CUT_FACE,
                edgecolor=CUT_EDGE,
                linewidths=0.4,
                zorder=1,
            )
            # No label, leaf or otherwise. Labelling only the cut leaves
            # made them read as a different kind of cut node, and the
            # lengths of paths the search never walks are not the point.
            continue
        if node.parent is None:
            face, label, text_color, edge = POINTER_PALETTE[0], str(node.start_node), "#FFF", "0.3"
        elif node.terminal:
            length = len(node.prefix)
            is_shortest = length == shortest
            face = SHORTEST_FACE if is_shortest else LONGER_FACE
            label = str(length)
            text_color = "#FFFFFF" if is_shortest else "#555555"
            edge = "0.25" if is_shortest else "0.55"
            # No extra ring on the winning leaf: the gold path already
            # arrives at it, and a second gold mark on the same marker
            # reads as a different kind of leaf rather than as emphasis.
        else:
            chosen = node.step.chosen if node.step is not None else None
            face, text_color, edge = "#FFFFFF", "#222222", "0.3"
            label = "" if chosen is None else str(chosen)
        ax.scatter(
            [x],
            [y],
            s=node_points,
            facecolor=face,
            edgecolor=edge,
            linewidths=0.6,
            zorder=3,
        )
        if label:
            ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=label_fontsize,
                color=text_color,
                zorder=4,
            )

    if survivors is not None and triplets is not None:
        _label_pruning_decision(ax, tree, pos, triplets, survivors, label_fontsize)

    headers = ("Start", *(f"Step {k}" for k in range(1, max_depth + 1)))
    for depth, name in enumerate(headers):
        ax.text(
            float(depth),
            y_hi + 1.25,
            name,
            ha="center",
            va="center",
            fontsize=label_fontsize,
            color="0.35",
        )
    ax.text(
        float(max_depth),
        y_lo - 0.85,
        "$|w|$",
        ha="center",
        va="center",
        fontsize=label_fontsize,
        color="0.35",
    )

    box = _optimal_box(tree, pos)
    if box_path:
        # A small rounding radius on purpose: the two magnification lines
        # leave from this box's right-hand corners, and a generous radius
        # pulls the drawn corner away from the mathematical one the lines
        # are anchored to, so they visibly miss it.
        ax.add_patch(
            FancyBboxPatch(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                boxstyle="round,pad=0,rounding_size=0.06",
                facecolor="none",
                edgecolor=ZOOM_EDGE,
                linewidth=0.7,
                zorder=6,
            )
        )

    ax.set_xlim(-0.55, max_depth + 0.55)
    ax.set_ylim(y_lo - 2.0, y_hi + 1.75)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return box


def canonical_decision(tree: SearchTree) -> tuple[int, int, tuple[str, ...]]:
    """Return the numbers behind the leaf column, for the caption.

    The gold path is the *result* of a rule applied to the leaf column,
    not a route walked down from a root, and the rule is two-stage:
    ``argmin`` over length, then lexicographic minimum among those. On
    this example the one-stage rule happens to agree -- the shortest
    string is also the lexicographic minimum of all of them -- which it
    need not be in general, so the caption has to say which rule it is.

    Args:
        tree: A fully enumerated tree.

    Returns:
        The number of complete strings, the minimum length, and the
        strings attaining it in lexicographic order.
    """
    leaves = tree.leaves()
    shortest = min(len(leaf.prefix) for leaf in leaves)
    winners = sorted({leaf.prefix for leaf in leaves if len(leaf.prefix) == shortest})
    return len(leaves), shortest, tuple(winners)


def _search_space_key(shortest: int, *, pruned: bool = False) -> list[object]:
    """Return the key for panel (a).

    Args:
        shortest: Length of the shortest complete string, so the marker
            entry names the number the green leaves carry rather than
            asking the reader to infer it. Measured over the surviving
            leaves when the tree is pruned.
        pruned: Add the entry for branches triplet pruning removes.

    Returns:
        Handles for ``ax.legend``.
    """
    from matplotlib.lines import Line2D

    # ``V`` and ``v`` are distinct instructions -- primary and secondary
    # pointer -- so only the surrounding words are capitalised. Title-casing
    # through the maths would silently rename half the alphabet.
    return [
        Line2D([0], [0], color=POINTER_PALETTE[0], lw=1.2, label="Branch at $V$ (π)"),
        Line2D([0], [0], color=POINTER_PALETTE[1], lw=1.2, label="Branch at $v$ (σ)"),
        Line2D([0], [0], color=GRAYED_EDGE, lw=0.8, ls=(0, (1.8, 1.4)), label="Forced Step"),
        Line2D([0], [0], color=CANONICAL_HALO, lw=3.0, alpha=0.85, label="Canonical Path"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=SHORTEST_FACE,
            markeredgecolor="0.25",
            markersize=3.8,
            label=f"Shortest ($|w|={shortest}$)",
        ),
        *(
            [Line2D([0], [0], color=CUT_EDGE, lw=1.0, ls=(0, (1.0, 1.2)), label="Pruned Branch")]
            if pruned
            else []
        ),
    ]


# ---------------------------------------------------------------------------
# Panel (c) -- the two orders that never branch
# ---------------------------------------------------------------------------


def _levels_reached(reasons: tuple[str, ...]) -> frozenset[str]:
    """Return the cascade levels that were tried and failed."""
    return frozenset(reason.split(":", 1)[0].strip() for reason in reasons)


def draw_cascade_grid(  # noqa: PLR0913 -- a panel needs its type sizes
    ax: Axes,
    iteration: EncoderIteration,
    *,
    directed: bool,
    label_fontsize: float = 4.8,
    header_fontsize: float = 5.4,
    show_reason: bool = False,
) -> None:
    """Draw one iteration's pair scan as a grid.

    Rows are the displacement pairs in :math:`\\mathcal{P}(M)` order,
    which is the order ``generate_pairs_sorted_by_sum`` yields and is
    keyed on ``(|a|+|b|, |a|, (a, b))``. Columns are the cascade levels
    in priority order. Every cell is determined: a level either failed
    for a stated reason, fired, or was never reached because a
    higher-priority level fired first. Nothing in this panel is a choice,
    which is the half of R3.7c's sentence a search tree cannot show.

    Args:
        ax: Target axes.
        iteration: The iteration to open up.
        directed: Whether the graph is directed. On an undirected graph
            the ``c`` level is never tried, and a column of blank cells
            would read as "never reached" rather than "not applicable".
        label_fontsize: Point size for cell and row labels.
        header_fontsize: Point size for the column headers.
        show_reason: Print why the winning pair's higher-priority levels
            failed.
    """
    from matplotlib.patches import Rectangle

    probes = iteration.probes
    n_rows = len(probes)
    x0 = 2.03
    cell_w = 0.30
    cell_h = 0.78

    for col, level in enumerate(CASCADE_LEVELS):
        ax.text(
            x0 + (col + 0.5) * cell_w,
            0.62,
            f"${level}$",
            ha="center",
            va="center",
            fontsize=header_fontsize,
            color=INSTRUCTION_PALETTE.get(level, "#333333"),
            fontweight="bold",
        )
    ax.annotate(
        "",
        xy=(x0 + len(CASCADE_LEVELS) * cell_w, 1.26),
        xytext=(x0, 1.26),
        arrowprops={"arrowstyle": "-|>", "color": "0.45", "linewidth": 0.7},
    )
    ax.text(
        x0 + len(CASCADE_LEVELS) * cell_w / 2.0,
        1.52,
        "Priority (fixed)",
        ha="center",
        va="center",
        fontsize=label_fontsize,
        color="0.35",
    )
    ax.annotate(
        "",
        xy=(x0 - 0.16, -(n_rows - 0.35) * cell_h),
        xytext=(x0 - 0.16, 0.18),
        arrowprops={"arrowstyle": "-|>", "color": "0.45", "linewidth": 0.7},
    )
    # Header, not a rotated spine label: rotated, it lands on the pointer
    # column, and there is no width to move either of them.
    # On its own line above the priority arrow: at panel (c)'s type size
    # this label is wide enough to run into "Priority (fixed)" if the two
    # share a baseline.
    ax.text(
        0.90,
        0.62,
        r"Displacement Pairs, in $\mathcal{P}(M)$ Order (fixed)",
        ha="center",
        va="center",
        fontsize=label_fontsize,
        color="0.35",
    )

    for row, probe in enumerate(probes):
        y = -row * cell_h
        a, b = probe.displacement
        ax.text(
            0.0,
            y,
            f"$({a:+d},{b:+d})$",
            ha="left",
            va="center",
            fontsize=label_fontsize + 0.3,
            fontfamily="monospace",
        )
        ax.text(
            0.66,
            y,
            f"$|a|{{+}}|b|={probe.cost}$",
            ha="left",
            va="center",
            fontsize=label_fontsize,
            color="0.4",
        )
        ax.text(
            1.24,
            y,
            f"$\\pi\\!\\to\\!{probe.primary_node},\\ \\sigma\\!\\to\\!{probe.secondary_node}$",
            ha="left",
            va="center",
            fontsize=label_fontsize,
            color="0.4",
        )
        failed = _levels_reached(probe.reasons)
        for col, level in enumerate(CASCADE_LEVELS):
            cx = x0 + col * cell_w
            not_applicable = level == "c" and not directed
            if level == probe.verdict:
                face, mark, mark_color = "#DCEFDC", r"$\checkmark$", "#1E7A2E"
            elif level in failed:
                face, mark, mark_color = "#F4F4F4", r"$\times$", "#9A9A9A"
            elif not_applicable:
                face, mark, mark_color = "#FFFFFF", r"$-$", "#C4C4C4"
            else:
                face, mark, mark_color = "#FFFFFF", "", "#C4C4C4"
            ax.add_patch(
                Rectangle(
                    (cx + 0.02, y - cell_h * 0.36),
                    cell_w - 0.04,
                    cell_h * 0.72,
                    facecolor=face,
                    edgecolor=GHOST_EDGE_COLOR,
                    linewidth=0.5,
                    zorder=1,
                )
            )
            if mark:
                ax.text(
                    cx + cell_w / 2.0,
                    y,
                    mark,
                    ha="center",
                    va="center",
                    fontsize=label_fontsize + 0.8,
                    color=mark_color,
                    zorder=2,
                )
        right = x0 + len(CASCADE_LEVELS) * cell_w + 0.12
        if probe.verdict == REJECTED:
            ax.text(
                right,
                y,
                "No operation applies",
                ha="left",
                va="center",
                fontsize=label_fontsize,
                color="0.55",
            )
        else:
            ax.text(
                right,
                y,
                f"emit  {iteration.emitted}",
                ha="left",
                va="center",
                fontsize=label_fontsize + 0.4,
                fontfamily="monospace",
                color="#1E7A2E",
                fontweight="bold",
            )

    if show_reason and probes:
        selected = probes[-1]
        if selected.reasons:
            ax.text(
                0.0,
                -(n_rows - 1) * cell_h - 0.72,
                "; ".join(selected.reasons),
                ha="left",
                va="center",
                fontsize=label_fontsize - 0.2,
                color="0.45",
            )

    # Bounded by what is actually drawn. The old right margin reserved
    # half an inch that nothing occupied, and every glyph in the panel
    # was scaled down to pay for it.
    ax.set_xlim(-0.1, x0 + len(CASCADE_LEVELS) * cell_w + 1.07)
    ax.set_ylim(-(n_rows - 1) * cell_h - 0.45, 1.90)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _ring_pos(order: tuple[NodeId, ...], node: NodeId) -> int:
    """Return the ring position of *node*, or ``0`` when absent."""
    try:
        return order.index(node)
    except ValueError:  # pragma: no cover - the pointers are always in the ring
        return 0


def _magnify(fig: Figure, source: Box, target: Box, lay: WalkthroughLayout) -> None:
    """Open a boxed region to the right into the panel that magnifies it.

    Two straight lines run from the source's right-hand corners to the
    target's left-hand corners, and the target is bracketed on three
    sides. The bracket has no left edge: the two lines arrive exactly
    where it would be, and drawing it as well would read as a seam.

    Both boxes are in inches.

    Args:
        fig: Target figure.
        source: The small box, in inches.
        target: The magnifying panel, in inches.
        lay: Printed geometry, for the inch-to-figure conversion.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    def to_fig(x: float, y: float) -> tuple[float, float]:
        return (x / lay.fig_width, y / lay.fig_height)

    sx1, sy0, sy1 = source[2], source[1], source[3]
    tx0, ty0, tx1, ty1 = target
    for sy, ty in ((sy1, ty1), (sy0, ty0)):
        (x0, y0), (x1, y1) = to_fig(sx1, sy), to_fig(tx0, ty)
        fig.add_artist(
            Line2D(
                [x0, x1],
                [y0, y1],
                transform=fig.transFigure,
                color=ZOOM_EDGE,
                linewidth=0.7,
                zorder=4,
            )
        )

    rc = 0.10  # corner radius, inches
    verts = [
        to_fig(tx0, ty1),
        to_fig(tx1 - rc, ty1),
        to_fig(tx1, ty1),
        to_fig(tx1, ty1 - rc),
        to_fig(tx1, ty0 + rc),
        to_fig(tx1, ty0),
        to_fig(tx1 - rc, ty0),
        to_fig(tx0, ty0),
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO,
    ]
    fig.add_artist(
        PathPatch(
            Path(verts, codes),
            transform=fig.transFigure,
            facecolor="none",
            edgecolor=ZOOM_EDGE,
            linewidth=0.7,
            zorder=4,
        )
    )


def _badge(fig: Figure, x_in: float, y_in: float, lay: WalkthroughLayout) -> None:
    """Draw the mark that pairs the highlighted row with panel (c).

    A drawn marker rather than a glyph: the figure's font is whatever the
    rendering machine has installed, and a symbol that silently falls
    back to a missing-glyph box would break the only link between the two
    panels.
    """
    from matplotlib.lines import Line2D

    fig.add_artist(
        Line2D(
            [x_in / lay.fig_width],
            [y_in / lay.fig_height],
            transform=fig.transFigure,
            marker="*",
            markersize=5.0,
            markerfacecolor=ACCENT_COLOR,
            markeredgecolor="#8A6D1F",
            markeredgewidth=0.5,
            linestyle="none",
            zorder=6,
        )
    )


def _rounded_box(fig: Figure, box: Box, lay: WalkthroughLayout) -> None:
    """Outline panel (c), in the highlighted row's colour."""
    from matplotlib.patches import FancyBboxPatch

    x0, y0, x1, y1 = box
    fig.add_artist(
        FancyBboxPatch(
            (x0 / lay.fig_width, y0 / lay.fig_height),
            (x1 - x0) / lay.fig_width,
            (y1 - y0) / lay.fig_height,
            boxstyle="round,pad=0,rounding_size=0.012",
            transform=fig.transFigure,
            facecolor="none",
            edgecolor=ACCENT_COLOR,
            linewidth=0.7,
            zorder=4,
        )
    )


def _draw_execution_row(  # noqa: PLR0913 -- one parameter per column of the row
    fig: Figure,
    trace: EncoderTrace,
    iteration: EncoderIteration,
    span: tuple[int, int],
    *,
    positions: dict[NodeId, Position],
    lay: WalkthroughLayout,
    y_bottom: float,
    highlight: bool,
) -> Box:
    """Draw one execution row of panel (b) and return its box, in inches."""
    from isalgraph.viz.cdll_view import draw_cdll_ring
    from isalgraph.viz.instruction_view import draw_instruction_strip
    from isalgraph.viz.worked_example import draw_state_graph

    all_nodes = frozenset(range(trace.graph.node_count()))
    all_edges = frozenset(graph_edges(trace.graph))
    created = iteration.created_node

    def axes(x_in: float, w_in: float, pad: float = 0.0) -> Axes:
        return fig.add_axes(
            (
                x_in / lay.fig_width,
                (y_bottom + pad) / lay.fig_height,
                w_in / lay.fig_width,
                (lay.row_height - 2 * pad) / lay.fig_height,
            )
        )

    box = (lay.right_x, y_bottom, lay.right_x + lay.right_width, y_bottom + lay.row_height)
    if highlight:
        _mark_row(fig, lay, box)

    x = lay.right_x
    ax_step = axes(x, lay.w_step)
    ax_step.axis("off")
    ax_step.text(
        0.5,
        0.62,
        f"Step {iteration.index + 1}",
        ha="center",
        va="center",
        fontsize=lay.fs_row,
        color="0.35",
        transform=ax_step.transAxes,
    )
    # The same token that labels edge k of the boxed path in panel (a).
    ax_step.text(
        0.5,
        0.28,
        iteration.emitted,
        ha="center",
        va="center",
        fontsize=lay.fs_row + 0.4,
        fontfamily="monospace",
        fontweight="bold",
        transform=ax_step.transAxes,
    )

    x += lay.w_step
    draw_cdll_ring(
        axes(x, lay.w_ring, pad=0.012),
        list(iteration.ring_after),
        _ring_pos(iteration.ring_after, iteration.primary_after),
        _ring_pos(iteration.ring_after, iteration.secondary_after),
        new_node_payload=created,
        new_node_color=ACCENT_COLOR,
        node_radius=lay.ring_node_radius,
        label_fontsize=lay.fs_small,
        pointer_fontsize=lay.fs_small,
        pointer_scale=4.6,
        pointer_lw=0.8,
        # The ring is height-bound in a half-inch row, so the only way to
        # enlarge the discs is to reclaim the arrow margin around them.
        # At the defaults it is 1.05 units against a ring of radius 1.
        arrow_gap=lay.ring_arrow_gap,
        label_gap=lay.ring_label_gap,
        margin_pad=lay.ring_margin_pad,
    )

    x += lay.w_ring
    draw_instruction_strip(
        axes(x, lay.w_strip, pad=0.12),
        trace.instruction_string,
        current_idx=span[1],
        solid_side="prefix",
        executing_span=span,
        axis_width_inches=lay.w_strip,
    )

    x += lay.w_strip
    draw_state_graph(
        axes(x, lay.w_graph, pad=0.012),
        trace.graph,
        positions,
        present_nodes=all_nodes - frozenset(iteration.captured_nodes_after),
        present_edges=all_edges - frozenset(iteration.captured_edges_after),
        accent_nodes=frozenset() if created is None else frozenset({created}),
        accent_edges=(
            frozenset() if iteration.created_edge is None else frozenset({iteration.created_edge})
        ),
        primary_node=iteration.primary_after,
        secondary_node=iteration.secondary_after,
        node_radius=lay.node_radius,
        label_fontsize=lay.fs_small,
        pointer_ring_scale=1.0,
        accent_solid=True,
    )
    if highlight:
        _badge(fig, box[0] + 0.10, box[3] - 0.09, lay)
    return box


def _mark_row(fig: Figure, lay: WalkthroughLayout, box: Box) -> None:
    """Outline the row that panel (c) magnifies.

    Drawn in figure coordinates rather than inside one row axes, so the
    outline reaches across every column of the row.
    """
    from matplotlib.patches import FancyBboxPatch

    x0, y0, x1, y1 = box
    inset = 0.02
    fig.add_artist(
        FancyBboxPatch(
            (x0 / lay.fig_width, (y0 + inset) / lay.fig_height),
            (x1 - x0) / lay.fig_width,
            (y1 - y0 - 2 * inset) / lay.fig_height,
            boxstyle="round,pad=0,rounding_size=0.006",
            transform=fig.transFigure,
            facecolor=ZOOM_FILL,
            edgecolor=ACCENT_COLOR,
            linewidth=0.7,
            linestyle=(0, (2.4, 1.8)),
            zorder=-5,
        )
    )


def _graph_positions(
    graph: SparseGraph,
    positions: dict[NodeId, Position] | None,
) -> dict[NodeId, Position]:
    """Return pinned *positions*, or compute a layout for *graph*."""
    if positions is not None:
        return positions
    from isalgraph.viz.layout import cdll_ring_positions, compact_graph_layout

    try:
        return compact_graph_layout(graph)
    except ImportError:  # pragma: no cover - networkx is optional
        return cdll_ring_positions(tuple(range(graph.node_count())))


def _box_to_inches(ax: Axes, fig: Figure, box: Box, lay: WalkthroughLayout) -> Box:
    """Convert a box in *ax* data coordinates to figure inches."""
    inv = fig.transFigure.inverted()
    (fx0, fy0), (fx1, fy1) = (
        inv.transform(ax.transData.transform((box[0], box[1]))),
        inv.transform(ax.transData.transform((box[2], box[3]))),
    )
    return (
        fx0 * lay.fig_width,
        fy0 * lay.fig_height,
        fx1 * lay.fig_width,
        fy1 * lay.fig_height,
    )


def _assemble(  # noqa: PLR0913 -- the figure has three panels
    graph: SparseGraph,
    tree: SearchTree,
    trace: EncoderTrace,
    *,
    branch_depth: int,
    zoom_iteration: int | None,
    positions: dict[NodeId, Position] | None,
    layout: WalkthroughLayout | None,
    survivors: frozenset[int] | None = None,
    triplets: Sequence[tuple[int, int, int]] | None = None,
) -> Figure:
    """Lay out the three panels from a prepared tree and trace.

    Both the exhaustive and the pruned figure come through here, so they
    cannot drift apart geometrically.

    Raises:
        WalkthroughError: If the trace does not walk the highlighted path.
    """
    import matplotlib.pyplot as plt

    lay = layout or WalkthroughLayout()
    if trace.instruction_string != tree.canonical:
        raise WalkthroughError(
            f"the encode from node {trace.start_node} emits {trace.instruction_string!r}, "
            f"which is not the highlighted string {tree.canonical!r}; "
            "panel (b) would walk a path panel (a) does not box"
        )

    if zoom_iteration is None:
        zoom_iteration = max(
            range(len(trace)),
            key=lambda i: sum(1 for p in trace.iterations[i].probes if p.verdict == REJECTED),
        )

    from isalgraph.viz.worked_example import group_spans

    spans = group_spans(trace.groups)
    pos = _graph_positions(graph, positions)
    n_rows = len(trace)
    fig = plt.figure(figsize=lay.figsize)

    # ---- panel (a) -------------------------------------------------------
    tree_top = lay.fig_height - lay.pad
    tree_bottom = lay.pad + lay.key_height
    ax_tree = fig.add_axes(
        (
            lay.pad / lay.fig_width,
            tree_bottom / lay.fig_height,
            lay.tree_width / lay.fig_width,
            (tree_top - tree_bottom) / lay.fig_height,
        )
    )
    data_box = draw_search_space(
        ax_tree,
        tree,
        branch_depth=branch_depth,
        label_fontsize=lay.fs_tree,
        node_points=lay.tree_node_points,
        survivors=survivors,
        triplets=triplets,
    )

    # Confined to panel (a)'s column. Spanning the figure would run the
    # key under panel (c), which is the same page real estate.
    ax_key = fig.add_axes(
        (
            lay.pad / lay.fig_width,
            lay.pad / lay.fig_height,
            lay.tree_width / lay.fig_width,
            lay.key_height / lay.fig_height,
        )
    )
    ax_key.axis("off")
    live = [lf for lf in tree.leaves() if survivors is None or lf.index in survivors]
    handles = _search_space_key(
        min(len(leaf.prefix) for leaf in live), pruned=survivors is not None
    )
    ax_key.legend(
        handles=handles,
        loc="center",
        fontsize=lay.fs_legend,
        ncol=3,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=2.2,
        labelspacing=0.5,
        borderpad=0.0,
        mode="expand",
    )

    # ---- panel (b) -------------------------------------------------------
    rows_top = lay.fig_height - lay.pad
    row_boxes: list[Box] = []
    for index, (iteration, span) in enumerate(zip(trace.iterations, spans, strict=True)):
        row_boxes.append(
            _draw_execution_row(
                fig,
                trace,
                iteration,
                span,
                positions=pos,
                lay=lay,
                y_bottom=rows_top - (index + 1) * lay.row_height,
                highlight=index == zoom_iteration,
            )
        )
    rows_bottom = rows_top - n_rows * lay.row_height

    # ---- panel (c) -------------------------------------------------------
    # Not a second magnification: panel (c) opens up one row of (b), and a
    # cone from a row that is not the last one would have to cross the rows
    # beneath it. The star on the highlighted row and on this box is the
    # link instead, and it costs no geometry.
    cascade_top = rows_bottom - lay.frustum_height
    cascade_bottom = cascade_top - lay.cascade_height
    cascade_box = (lay.right_x, cascade_bottom, lay.fig_width - lay.pad, cascade_top)
    ax_cascade = fig.add_axes(
        (
            (lay.right_x + lay.cascade_inset) / lay.fig_width,
            cascade_bottom / lay.fig_height,
            (lay.right_width - 2 * lay.cascade_inset) / lay.fig_width,
            lay.cascade_height / lay.fig_height,
        )
    )
    draw_cascade_grid(
        ax_cascade,
        trace.iterations[zoom_iteration],
        directed=graph.directed(),
        label_fontsize=lay.fs_cascade,
        header_fontsize=lay.fs_cascade_header,
    )
    _rounded_box(fig, cascade_box, lay)
    _badge(fig, cascade_box[0] + 0.075, cascade_box[3] - 0.075, lay)

    # ---- the magnification, (a) into (b) ---------------------------------
    _magnify(
        fig,
        _box_to_inches(ax_tree, fig, data_box, lay),
        (lay.right_x, rows_bottom, lay.fig_width - lay.pad, rows_top),
        lay,
    )
    return fig


def canonical_search_walkthrough_figure(  # noqa: PLR0913 -- the figure has three panels
    graph: SparseGraph,
    *,
    start_node: NodeId = 0,
    branch_depth: int = DEFAULT_BRANCH_DEPTH,
    max_depth: int = 24,
    max_nodes: int = 20_000,
    zoom_iteration: int | None = None,
    positions: dict[NodeId, Position] | None = None,
    layout: WalkthroughLayout | None = None,
) -> Figure:
    """Build the figure for the **exhaustive** canonicalisation.

    Args:
        graph: A small graph, 6-8 nodes for legibility.
        start_node: The node the executed path starts from. It must be
            one that attains the canonical string, or panels (a) and (b)
            show different executions.
        branch_depth: Last step at which the search still branches; used
            only to shade panel (a)'s forced columns.
        max_depth: Depth budget for enumeration. Large enough that every
            path completes; the panel refuses to draw truncated leaves.
        max_nodes: Enumeration size cap.
        zoom_iteration: Index of the iteration panel (c) magnifies.
            Defaults to the iteration that rejected the most pairs, which
            is where the fixed orders do the most work.
        positions: Pinned node coordinates for the graph panels.
        layout: Printed geometry.

    Returns:
        The created figure. The caller owns it and must close it.
    """
    tree = enumerate_search_tree(graph, max_depth=max_depth, max_nodes=max_nodes, max_roots=None)
    return _assemble(
        graph,
        tree,
        trace_encoder(graph, start_node),
        branch_depth=branch_depth,
        zoom_iteration=zoom_iteration,
        positions=positions,
        layout=layout,
    )


def pruned_search_walkthrough_figure(  # noqa: PLR0913 -- the figure has three panels
    graph: SparseGraph,
    *,
    branch_depth: int = DEFAULT_BRANCH_DEPTH,
    max_depth: int = 24,
    max_nodes: int = 20_000,
    zoom_iteration: int | None = None,
    positions: dict[NodeId, Position] | None = None,
    layout: WalkthroughLayout | None = None,
    triplet_labels: bool = False,
) -> Figure:
    """Build the same figure for the **pruned** canonicalisation.

    The exhaustive tree is still drawn in full, with the branches triplet
    pruning removes shown as cut. Drawing only the surviving nodes would
    show a smaller search space and no pruning, which is the thing the
    figure exists to show.

    There is no *start_node* parameter. The pruned canonical string is
    generally emitted by no greedy run, so panel (b) is recovered with
    :func:`~isalgraph.viz.encoder_trace.trace_execution`, which finds the
    starting node itself.

    Args:
        graph: A small graph, 6-8 nodes for legibility.
        branch_depth: Last step at which the exhaustive search branches;
            used only to shade panel (a)'s forced columns.
        max_depth: Depth budget for enumeration.
        max_nodes: Enumeration size cap.
        zoom_iteration: Index of the iteration panel (c) magnifies.
        positions: Pinned node coordinates for the graph panels.
        layout: Printed geometry.
        triplet_labels: Write the deciding triplets on the first pruned
            branch of the retained path. Off by default: at this width
            they land on the neighbouring rows, and the caption states
            the same two values without the collision.

    Returns:
        The created figure. The caller owns it and must close it.

    Raises:
        WalkthroughError: If the surviving leaves do not reproduce
            :func:`~isalgraph.core.canonical_pruned.pruned_canonical_string`.
            That is the check on :func:`prune_survivors`: the filter is a
            model of the pruned search, and a model that disagrees with
            the algorithm must not be drawn.
    """
    from isalgraph.core.canonical_pruned import (
        compute_structural_triplets,
        pruned_canonical_string,
    )
    from isalgraph.viz.encoder_trace import trace_execution

    triplets = compute_structural_triplets(graph)
    target = pruned_canonical_string(graph)
    tree = enumerate_search_tree(graph, max_depth=max_depth, max_nodes=max_nodes, max_roots=None)
    survivors = prune_survivors(tree, triplets)

    live = [lf for lf in tree.leaves() if lf.index in survivors]
    if not live:  # pragma: no cover - the roots always survive
        raise WalkthroughError("pruning left no complete path")
    best = min((len(lf.prefix), lf.prefix) for lf in live)[1]
    if best != target:
        raise WalkthroughError(
            f"the pruned survivors minimise to {best!r} but "
            f"pruned_canonical_string is {target!r}; the filter in "
            "prune_survivors does not model the pruned search"
        )

    remark_optimal(tree, target, survivors=survivors)
    return _assemble(
        graph,
        tree,
        trace_execution(graph, target),
        branch_depth=branch_depth,
        zoom_iteration=zoom_iteration,
        positions=positions,
        layout=layout,
        survivors=survivors,
        triplets=triplets if triplet_labels else None,
    )


__all__ = [
    "CASCADE_LEVELS",
    "DEFAULT_BRANCH_DEPTH",
    "WalkthroughError",
    "WalkthroughLayout",
    "canonical_decision",
    "canonical_search_walkthrough_figure",
    "prune_survivors",
    "pruned_search_walkthrough_figure",
    "remark_optimal",
    "draw_cascade_grid",
    "draw_search_space",
]
