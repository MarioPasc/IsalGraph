"""Elsevier graphical abstract: the canonical search space and the round trip.

A deliberate **copy** of :mod:`isalgraph.viz.search_walkthrough`, not an
import of it. The paper figure is a frozen artifact held by test at a
7.20 x 4.50 in portrait-ish measure; a graphical abstract is locked to
Elsevier's 5:2 window and has perhaps a fifth of the type budget. Sharing
one layout dataclass between the two would mean every tweak made for the
abstract moving the submitted figure, so the geometry is forked and the
two are kept apart on purpose.

Elsevier's constraints, applied here:

* **Aspect ratio 500:200 = 2.5**, exactly. :attr:`AbstractLayout.figsize`
  is checked against it by :meth:`AbstractLayout.__post_init__`.
* **>= 1328 x 531 px at >= 300 dpi.** The default 7.50 x 3.00 in canvas
  renders to 2250 x 900 px at 300 dpi and 4500 x 1800 px at 600.
* **Times, Arial, Courier or Symbol.** :func:`apply_abstract_style` puts
  the metric-compatible Times and Courier clones that are actually
  installed (Nimbus Roman, Liberation Serif; Nimbus Mono PS, Liberation
  Mono) ahead of matplotlib's DejaVu defaults, which are neither.
* **A clear start and end.** Reading is left to right: the search space,
  then the pair of machines that walk between the two spaces.

What differs from the paper figure, beyond size:

``tree``
    Panel (a) unchanged in kind -- the whole search tree with depth on
    the horizontal axis, the canonical path boxed.

``g2s`` / ``s2g``
    Panel (b) split in two. The encode is drawn exactly as before; the
    decode is drawn in the same row idiom from
    :func:`~isalgraph.viz.worked_example.decode_trace`, so the two
    machines are visibly inverse rather than merely asserted to be.

``graph-space`` / ``instruction-space``
    Two wide, short bands above and below the pair. Four arrows close the
    loop: graph space -> G2S -> instruction space -> S2G -> graph space.
    That loop is the round-trip theorem, drawn.

Panel (c) of the paper figure -- the displacement/priority grid -- is
**dropped**. Its cells are 4.8 pt glyphs; at the 500 px width Elsevier
renders a graphical abstract at, nothing in it is legible.

The decoded graph's node ids are not the input graph's. They are put back
in correspondence by :func:`decoded_positions`, which reads the pairing
off the two traces and *checks* that it is an isomorphism before either
panel is drawn.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from isalgraph.core.trace import graph_edges
from isalgraph.viz.encoder_trace import EncoderTrace, trace_encoder
from isalgraph.viz.search_tree import (
    CANONICAL_HALO,
    SearchTree,
    enumerate_search_tree,
)
from isalgraph.viz.style import (
    ACCENT_COLOR,
    GRAYED_EDGE,
    INSTRUCTION_PALETTE,
    POINTER_PALETTE,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from isalgraph.core.sparse_graph import SparseGraph
    from isalgraph.core.trace import AlgorithmTrace, Edge
    from isalgraph.types import NodeId
    from isalgraph.viz.base import Position

#: Step index at and below which the running example's search still
#: branches. Measured, not assumed. Used only to shade the forced columns.
DEFAULT_BRANCH_DEPTH: int = 3

#: Fill for a leaf whose string attains the minimum length.
SHORTEST_FACE: str = "#228833"

#: Fill for a leaf that is complete but longer than the minimum.
LONGER_FACE: str = "#F2F2F2"

#: Branches the triplet pruning never explores.
CUT_EDGE: str = "#E0E0E0"
CUT_FACE: str = "#FCFCFC"
CUT_TEXT: str = "#C8C8C8"

#: Outline of the magnification and of the two algorithm panels.
ZOOM_EDGE: str = "#1A1A1A"

#: Fill behind the two wide bands. Light enough that the graph and the
#: instruction strip drawn on them keep their own contrast.
BAND_FACE: str = "#F4F6F8"
BAND_EDGE: str = "#C9D0D8"

#: The four stream arrows. One neutral colour for both directions: the
#: arrowheads already say which way each runs, and two accent colours
#: here would collide with the pointer palette, where red and blue
#: already mean the primary and the secondary pointer.
STREAM_COLOR: str = "#454B54"

#: Times- and Courier-metric families that are actually installed on a
#: Debian box, ahead of matplotlib's DejaVu defaults. Elsevier asks for
#: Times, Arial, Courier or Symbol; DejaVu is none of the four.
ABSTRACT_SERIF: tuple[str, ...] = (
    "Times New Roman",
    "Times",
    "Nimbus Roman",
    "Liberation Serif",
    "DejaVu Serif",
)
ABSTRACT_MONO: tuple[str, ...] = (
    "Courier New",
    "Courier",
    "Nimbus Mono PS",
    "Liberation Mono",
    "DejaVu Sans Mono",
)

#: ``draw_instruction_strip``'s own cell geometry, in its axis units.
#: Mirrored here so the instruction-space band can size its axes to keep
#: the cells square; the strip sets an "auto" aspect and will otherwise
#: stretch them to whatever box it is handed.
STRIP_CELL_WIDTH: float = 1.05
STRIP_CELL_HEIGHT: float = 1.1

#: Elsevier's ratio: a 500 wide x 200 high window on ScienceDirect.
ELSEVIER_RATIO: float = 2.5

#: Elsevier's floor, in pixels.
ELSEVIER_MIN_PIXELS: tuple[int, int] = (1328, 531)


class VisualAbstractError(ValueError):
    """Raised when the abstract cannot be assembled from the given trace."""


#: A rectangle as ``(x0, y0, x1, y1)``.
Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class AbstractLayout:
    """Printed geometry, in true inches and points.

    Every number is what lands on the page. The canvas is locked to
    Elsevier's 5:2 window; the vertical budget is then spent, top to
    bottom, on the graph band, the stream gap, the two algorithm panels,
    the second stream gap and the instruction band, with *logo_band*
    reserved blank at the foot for the laboratory mark to be dropped in
    with TikZ.

    Args:
        fig_width: Total figure width.
        fig_height: Total figure height. Must be *fig_width* / 2.5.
        pad: Outer margin on all four sides.
        logo_band: Blank strip reserved at the foot of the figure.
        tree_width: Width of the search-space panel.
        col_gap: Gap between the search-space panel and the right column.
        tree_title_height: Band above the search-space panel for its name.
        key_height: Band under the search-space panel holding its key.
        graph_band: Height of the graph-space band. The graph inside it
            keeps an equal aspect, so this -- not the band's width --
            is what sets how large the graph prints.
        instr_band: Height of the instruction-space band. The strip
            inside it is sized *from* this, so the cells stay square.
        stream_gap: Height of each of the two arrow gaps.
        box_gap: Gap between the G2S and the S2G panel.
        box_title: Title bar inside each algorithm panel.
        box_inset: Blank margin inside each algorithm panel.
        f_step: Row step-label column, as a fraction of the panel width.
        f_ring: Row CDLL-ring column, as a fraction of the panel width.
        f_strip: Row instruction-strip column, as a fraction.
        f_graph: Row graph column, as a fraction.
        fs_box_title: Point size of "Graph-To-String" / "String-To-Graph".
        fs_band_title: Point size of the two band names.
        fs_panel_title: Point size of the search-space panel name.
        fs_stream: Point size of the "Encode" and "Decode" arrow labels.
        fs_row: Row-label point size.
        fs_ring_node: CDLL disc-label point size inside a row.
        fs_graph_node: Graph node-label point size inside a row.
        fs_pointer: Point size of the pi and sigma glyphs. Held apart
            from the disc labels on purpose: the glyphs sit in the ring's
            blank margin, where there is room to grow, and they are the
            one thing that ties a row's CDLL to its graph.
        fs_band_node: Node-label point size in the graph-space band.
        label_dy: Downward shift of every node label, in node-radius
            units, correcting the optical rise ``va="center"`` gives a
            digit. See :func:`~isalgraph.viz.cdll_view.draw_cdll_ring`.
        fs_tree: Tree label point size.
        fs_legend: Search-space key point size.
        node_radius: Graph node radius, in graph-panel axis units. The
            panel is height-bound, so this is the only lever on how large
            a node prints. Its ceiling is set by the closest pair of
            pinned coordinates -- 0.881 apart on the running example, so
            discs meet at 0.44 and anything above that overlaps.
        band_node_radius: Node radius in the graph-space band. Below
            *node_radius* on purpose: the band has height the rows do not,
            so it does not have to run at the 0.44 ceiling where the
            closest pair of discs meets, and the graph reads better with
            the pair visibly apart.
        ring_node_radius: CDLL node radius, in ring axis units. Adjacent
            centres on a six-node ring of radius one are exactly one unit
            apart, so 0.5 is where the discs touch.
        ring_label_gap: Ring-radius units from the discs out to the pi
            and sigma glyphs. The pointer arrows are off here, so this
            measures from the disc rather than from an arrow tail.
        ring_margin_pad: Blank ring-radius units beyond the discs. Must
            exceed *ring_label_gap* by at least the glyph's own height,
            or the glyph is clipped by the axes.
        tree_node_points: Search-space marker area, in points squared.
        arrow_lw: Stroke width of the four stream arrows.
        arrow_head: ``mutation_scale`` of the four stream arrowheads.
        mark_executing: Give the cells a row executes the heavy stroke
            and the caret the paper figure uses. Off here: at a third of
            an inch the caret is a solid black triangle a third the
            height of the cell it points at, and the information is
            already carried by where the strip stops being solid.
        max_rows: Cap on the number of rows per panel, applied by
            subsampling when the run is longer. Four is what the 5:2
            canvas affords once the two space bands are thinned to the
            height their contents actually need: measured on the running
            example, six rows put the CDLL discs at 2.6 pt and their
            labels are unreadable even at full size. ``None`` draws every
            step. Ignored when *steps* is given.
        steps: Which encoder iterations to draw as rows, or ``None`` to
            let *max_rows* choose. Both panels use the same selection, so
            the two stay row-aligned.
    """

    fig_width: float = 7.50
    fig_height: float = 3.00
    pad: float = 0.07
    logo_band: float = 0.30
    tree_width: float = 2.02
    col_gap: float = 0.22
    tree_title_height: float = 0.17
    key_height: float = 0.28
    graph_band: float = 0.44
    instr_band: float = 0.30
    stream_gap: float = 0.19
    box_gap: float = 0.26
    box_title: float = 0.19
    box_inset: float = 0.04
    f_step: float = 0.105
    f_ring: float = 0.140
    f_strip: float = 0.545
    f_graph: float = 0.210
    fs_box_title: float = 9.0
    fs_band_title: float = 7.2
    fs_panel_title: float = 8.4
    fs_stream: float = 6.4
    fs_row: float = 5.0
    fs_ring_node: float = 3.7
    fs_graph_node: float = 4.4
    fs_pointer: float = 4.2
    fs_band_node: float = 5.0
    label_dy: float = 0.12
    fs_tree: float = 4.2
    fs_legend: float = 5.0
    node_radius: float = 0.40
    band_node_radius: float = 0.30
    ring_node_radius: float = 0.46
    ring_label_gap: float = 0.50
    ring_margin_pad: float = 0.98
    tree_node_points: float = 36.0
    arrow_lw: float = 2.2
    arrow_head: float = 11.0
    mark_executing: bool = False
    max_rows: int | None = 4
    steps: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Refuse a canvas ScienceDirect would letterbox.

        Raises:
            VisualAbstractError: If the canvas is not 5:2, or renders
                below Elsevier's pixel floor at 300 dpi.
        """
        ratio = self.fig_width / self.fig_height
        if abs(ratio - ELSEVIER_RATIO) > 1e-3:
            raise VisualAbstractError(
                f"canvas is {ratio:.3f}:1 but Elsevier renders a graphical abstract "
                f"in a 500 x 200 window, so it must be {ELSEVIER_RATIO}:1"
            )
        px = (round(self.fig_width * 300), round(self.fig_height * 300))
        if px[0] < ELSEVIER_MIN_PIXELS[0] or px[1] < ELSEVIER_MIN_PIXELS[1]:
            raise VisualAbstractError(
                f"canvas renders to {px[0]} x {px[1]} px at 300 dpi, below Elsevier's "
                f"{ELSEVIER_MIN_PIXELS[0]} x {ELSEVIER_MIN_PIXELS[1]} floor"
            )

    @property
    def figsize(self) -> tuple[float, float]:
        """Return the ``(width, height)`` inch pair for ``plt.figure``."""
        return (self.fig_width, self.fig_height)

    @property
    def content_bottom(self) -> float:
        """Bottom of everything drawn, in inches. The logo band is below."""
        return self.pad + self.logo_band

    @property
    def content_top(self) -> float:
        """Top of everything drawn, in inches."""
        return self.fig_height - self.pad

    @property
    def right_x(self) -> float:
        """Left edge of the round-trip column, in inches."""
        return self.pad + self.tree_width + self.col_gap

    @property
    def right_width(self) -> float:
        """Width of the round-trip column, in inches."""
        return self.fig_width - self.pad - self.right_x

    @property
    def box_width(self) -> float:
        """Width of one algorithm panel, in inches. Both are equal."""
        return (self.right_width - self.box_gap) / 2.0

    @property
    def box_height(self) -> float:
        """Height of one algorithm panel, in inches. Both are equal."""
        return (
            self.content_top
            - self.content_bottom
            - self.graph_band
            - self.instr_band
            - 2.0 * self.stream_gap
        )

    def row_columns(self) -> tuple[float, float, float, float]:
        """Return the four row column widths of one panel, in inches.

        Raises:
            VisualAbstractError: If the four fractions do not sum to one.
        """
        fracs = (self.f_step, self.f_ring, self.f_strip, self.f_graph)
        if abs(sum(fracs) - 1.0) > 1e-6:
            raise VisualAbstractError(
                f"row column fractions sum to {sum(fracs):.4f}, not 1.0; "
                "the four columns must tile the panel exactly"
            )
        usable = self.box_width - 2.0 * self.box_inset
        return (fracs[0] * usable, fracs[1] * usable, fracs[2] * usable, fracs[3] * usable)


def apply_abstract_style() -> None:
    """Apply the IEEE base style, then Elsevier's rules over it.

    Two overrides, both load-bearing:

    *Fonts.* :func:`~isalgraph.viz.style.apply_ieee_style` asks for Times
    New Roman and falls back to DejaVu Serif, which is not a Times. This
    puts the installed metric-compatible clones in between, so the
    rendered glyphs really are Times and Courier shapes.

    *Bounding box.* The base style sets ``savefig.bbox = "tight"``, which
    crops the canvas to its ink. On this figure that does two harmful
    things at once: it breaks the 5:2 ratio Elsevier scales the abstract
    to -- measured, the first render came out 2220 x 774 px, that is
    2.87:1 -- and it trims away :attr:`AbstractLayout.logo_band`, which
    is reserved *blank* and therefore has no ink to hold it open.
    """
    import matplotlib as mpl

    from isalgraph.viz.style import apply_ieee_style

    apply_ieee_style()
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = list(ABSTRACT_SERIF)
    mpl.rcParams["font.monospace"] = list(ABSTRACT_MONO)
    mpl.rcParams["savefig.bbox"] = None
    mpl.rcParams["savefig.pad_inches"] = 0.0


def save_abstract(
    fig: Figure,
    basepath: str | Path,
    *,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
) -> list[Path]:
    """Write the abstract and *check* what was written against Elsevier's rule.

    The canvas is saved uncropped, then the raster is reopened and its
    real pixel size measured. A ratio assertion on the layout is not
    enough: ``savefig.bbox`` lives in the rcParams, so a caller that
    forgot :func:`apply_abstract_style` would silently emit a cropped
    image that still passed every check made before the write.

    Args:
        fig: The figure to write.
        basepath: Output path without an extension.
        formats: Extensions to emit. ``png`` is the one Elsevier's pixel
            rule applies to and the only one measured.
        dpi: Raster resolution. The default of 600 puts the default
            canvas at 4500 x 1800 px, well over the floor.

    Returns:
        The list of paths written.

    Raises:
        VisualAbstractError: If the written PNG is off ratio or below
            Elsevier's pixel floor.
    """
    bp = Path(basepath)
    bp.parent.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for fmt in formats:
        path = bp.parent / f"{bp.name}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches=None, pad_inches=0.0)
        out.append(path)
        if fmt != "png":
            continue
        import matplotlib.image as mpimg

        height, width = mpimg.imread(path).shape[:2]
        ratio = width / height
        if abs(ratio - ELSEVIER_RATIO) > 5e-3:
            raise VisualAbstractError(
                f"{path.name} is {width} x {height} px, ratio {ratio:.3f}; Elsevier "
                f"scales a graphical abstract into a 500 x 200 window, so it must be "
                f"{ELSEVIER_RATIO}:1. A cropping savefig.bbox is the usual cause"
            )
        if width < ELSEVIER_MIN_PIXELS[0] or height < ELSEVIER_MIN_PIXELS[1]:
            raise VisualAbstractError(
                f"{path.name} is {width} x {height} px, below Elsevier's "
                f"{ELSEVIER_MIN_PIXELS[0]} x {ELSEVIER_MIN_PIXELS[1]} floor"
            )
    return out


# ---------------------------------------------------------------------------
# The search space
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
    to those attaining the maximum structural triplet. The candidate set
    at a branch is exactly that node's children in the enumerated
    exhaustive tree, so the pruned search is a *subtree* of the
    exhaustive one and can be obtained by filtering.

    Args:
        tree: A fully enumerated exhaustive tree.
        triplets: Structural triplet per input-graph node.

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

    Args:
        tree: The enumerated tree, modified in place.
        target: The string whose path should be highlighted.
        survivors: When given, only a leaf inside this set is eligible.

    Raises:
        VisualAbstractError: If no eligible leaf emits *target*.
    """
    leaves = [
        n
        for n in tree.leaves()
        if n.prefix == target and (survivors is None or n.index in survivors)
    ]
    if not leaves:
        raise VisualAbstractError(f"no eligible leaf emits {target!r}")
    for node in tree.nodes:
        node.optimal = False
    index: int | None = leaves[0].index
    while index is not None:
        tree.nodes[index].optimal = True
        index = tree.nodes[index].parent
    tree.canonical = target


def _optimal_box(tree: SearchTree, pos: dict[int, tuple[float, float]]) -> Box:
    """Return the bounding box of the canonical path, in tree data units.

    Raises:
        VisualAbstractError: If no node is marked optimal.
    """
    xs = [pos[n.index][0] for n in tree.nodes if n.optimal]
    ys = [pos[n.index][1] for n in tree.nodes if n.optimal]
    if not xs:  # pragma: no cover - enumeration always marks a path
        raise VisualAbstractError("no node is marked optimal; the tree has no canonical path")
    return (min(xs) - 0.42, min(ys) - 0.52, max(xs) + 0.42, max(ys) + 0.52)


def _draw_tree_edges(  # noqa: PLR0913 -- one parameter per drawing input
    ax: Axes,
    tree: SearchTree,
    pos: dict[int, tuple[float, float]],
    survivors: frozenset[int] | None,
    *,
    halo_lw: float,
) -> None:
    """Draw every edge of the tree, coloured by what kind of step it is."""
    for node in tree.nodes:
        if node.parent is None or node.step is None:
            continue
        x0, y0 = pos[node.parent]
        x1, y1 = pos[node.index]
        step = node.step
        if survivors is not None and node.index not in survivors:
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=CUT_EDGE,
                lw=0.4,
                ls=(0, (1.0, 1.2)),
                zorder=0,
                solid_capstyle="round",
            )
            continue
        branching = step.n_candidates > 1
        color = POINTER_PALETTE[0 if step.op in ("V", "C") else 1] if branching else GRAYED_EDGE
        if node.optimal:
            ax.plot([x0, x1], [y0, y1], color=CANONICAL_HALO, lw=halo_lw, zorder=0, alpha=0.9)
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=color,
            lw=0.8 if branching else 0.55,
            ls="-" if branching else (0, (1.6, 1.3)),
            zorder=1,
            solid_capstyle="round",
        )


def _draw_tree_nodes(  # noqa: PLR0913 -- one parameter per drawing input
    ax: Axes,
    tree: SearchTree,
    pos: dict[int, tuple[float, float]],
    survivors: frozenset[int] | None,
    *,
    shortest: int,
    node_points: float,
    label_fontsize: float,
    node_labels: bool,
) -> None:
    """Draw every node of the tree."""
    for node in tree.nodes:
        x, y = pos[node.index]
        if survivors is not None and node.index not in survivors:
            ax.scatter(
                [x],
                [y],
                s=node_points * 0.55,
                facecolor=CUT_FACE,
                edgecolor=CUT_EDGE,
                linewidths=0.3,
                zorder=1,
            )
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
            linewidths=0.45,
            zorder=3,
        )
        # Interior digits are the uninserted neighbour each branch picked.
        # Sub-pixel in the thumbnail, legible at full size -- which is the
        # right trade: they are the tree's only per-node content.
        if label and (node_labels or node.parent is None or node.terminal):
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


def draw_search_space(  # noqa: PLR0913 -- one parameter per visual layer
    ax: Axes,
    tree: SearchTree,
    *,
    branch_depth: int = DEFAULT_BRANCH_DEPTH,
    label_fontsize: float = 4.2,
    node_points: float = 34.0,
    node_labels: bool = True,
    survivors: frozenset[int] | None = None,
    headers: bool = True,
) -> Box:
    """Draw the whole search tree with depth on the horizontal axis.

    Every step of every path is drawn; nothing is collapsed. Branch edges
    -- a ``V``/``v`` step with more than one candidate -- are solid and
    coloured by the acting pointer, and a step with a single candidate is
    dashed grey.

    Args:
        ax: Target axes.
        tree: A tree enumerated deep enough that every leaf is terminal.
        branch_depth: Last step at which the search still branches. Used
            only to shade the forced columns.
        label_fontsize: Point size for node and edge labels.
        node_points: Node marker area, in points squared.
        node_labels: Write the chosen neighbour inside every interior
            node, not only in the root and leaf columns. On by default:
            the digits are what say *which* uninserted neighbour each
            branch took, and without them the interior of the tree is a
            lattice of blank discs.
        survivors: Nodes the pruned search reaches. When given, everything
            outside the set is drawn as cut.
        headers: Write the ``Start`` / ``Step k`` column names.

    Returns:
        The canonical path's box in data coordinates.

    Raises:
        VisualAbstractError: If any leaf is not terminal.
    """
    from matplotlib.patches import FancyBboxPatch, Rectangle

    unfinished = [n for n in tree.leaves() if not n.terminal]
    if unfinished:
        raise VisualAbstractError(
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
                facecolor="#F0F0F0",
                edgecolor="none",
                zorder=-2,
            )
        )
        ax.text(
            (branch_depth + 1.0 + max_depth) / 2.0,
            y_lo - 1.5,
            f"Forced past Step {branch_depth}",
            ha="center",
            va="center",
            fontsize=label_fontsize + 0.2,
            color="0.45",
        )

    _draw_tree_edges(ax, tree, pos, survivors, halo_lw=2.6)
    _draw_tree_nodes(
        ax,
        tree,
        pos,
        survivors,
        shortest=shortest,
        node_points=node_points,
        label_fontsize=label_fontsize,
        node_labels=node_labels,
    )

    if headers:
        names = ("Start", *(f"Step {k}" for k in range(1, max_depth + 1)))
        for depth, name in enumerate(names):
            ax.text(
                float(depth),
                y_hi + 1.2,
                name,
                ha="center",
                va="center",
                fontsize=label_fontsize + 0.2,
                color="0.4",
            )
        # No ``|w|`` label under the leaf column: at this width it lands on
        # the forced-band caption, and the key already names what the leaf
        # numbers are.

    box = _optimal_box(tree, pos)
    ax.add_patch(
        FancyBboxPatch(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            boxstyle="round,pad=0,rounding_size=0.06",
            facecolor="none",
            edgecolor=ZOOM_EDGE,
            linewidth=0.6,
            zorder=6,
        )
    )

    ax.set_xlim(-0.55, max_depth + 0.55)
    ax.set_ylim(y_lo - 1.95, y_hi + 1.7)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return box


def _search_space_key(shortest: int, *, pruned: bool = False) -> list[object]:
    """Return the key for the search-space panel.

    Four entries, not six. A graphical abstract is read at a glance, and
    the two entries dropped -- the shortest-leaf marker's numeric value
    and the forced-step rail -- are both already stated inside the panel,
    by the ``|w|`` column and by the shaded band's own caption.

    Args:
        shortest: Length of the shortest complete string.
        pruned: Add the entry for branches triplet pruning removes.

    Returns:
        Handles for ``ax.legend``.
    """
    from matplotlib.lines import Line2D

    return [
        Line2D([0], [0], color=POINTER_PALETTE[0], lw=1.1, label="Branch at $V$"),
        Line2D([0], [0], color=POINTER_PALETTE[1], lw=1.1, label="Branch at $v$"),
        Line2D([0], [0], color=CANONICAL_HALO, lw=2.6, alpha=0.9, label="Canonical Path"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=SHORTEST_FACE,
            markeredgecolor="0.25",
            markersize=3.4,
            label=f"Shortest, $|w|={shortest}$",
        ),
        *(
            [Line2D([0], [0], color=CUT_EDGE, lw=1.0, ls=(0, (1.0, 1.2)), label="Pruned Branch")]
            if pruned
            else []
        ),
    ]


# ---------------------------------------------------------------------------
# One execution row, shared by both machines
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RowState:
    """Everything one execution row draws.

    Both machines reduce their own trace to a tuple of these and one
    renderer draws them, so the G2S and the S2G panel are identical in
    layout by construction rather than by two builders being kept in step
    by hand. It is the same device
    :class:`~isalgraph.viz.worked_example.ExampleColumn` uses, restated
    for a row rather than a column.

    Args:
        index: Zero-based step number.
        token: The symbol group this step emits or consumes.
        ring_order: CDLL contents in forward circular order.
        primary: Graph node under the primary pointer.
        secondary: Graph node under the secondary pointer.
        span: Half-open ``[lo, hi)`` cell range this step covers.
        present_nodes: Graph nodes drawn solid.
        present_edges: Graph edges drawn solid.
        accent_nodes: Graph nodes carrying the created-this-step halo.
        accent_edges: Graph edges drawn in the accent colour.
        ring_accent: Ring payload created this step, if any.
        strip_solid_side: ``"prefix"`` to draw the cells before the
            cursor solid, ``"suffix"`` to draw the cells from it onward.
            This is the direction of travel: G2S *emits*, so its strip
            fills from empty; S2G *consumes*, so its strip starts whole
            and drains. Each panel's graph runs the opposite way, and the
            pair of them is the ink-conservation argument.
    """

    index: int
    token: str
    ring_order: tuple[NodeId, ...]
    primary: NodeId
    secondary: NodeId
    span: tuple[int, int]
    present_nodes: frozenset[NodeId]
    present_edges: frozenset[Edge]
    accent_nodes: frozenset[NodeId] = frozenset()
    accent_edges: frozenset[Edge] = frozenset()
    ring_accent: NodeId | None = None
    strip_solid_side: str = "prefix"


def g2s_rows(trace: EncoderTrace) -> tuple[RowState, ...]:
    """Reduce an encoder trace to one row per iteration.

    The G2S graph panel *drains*: the structure already encoded is
    ghosted, so what stays solid is what the encoder has yet to reach.
    That is the convention the merged paper figure uses and it is kept,
    because it is what makes the ink conservation against the strip
    visible.

    Args:
        trace: The instrumented encode.

    Returns:
        One row per encoder iteration.
    """
    from isalgraph.viz.worked_example import group_spans

    all_nodes = frozenset(range(trace.graph.node_count()))
    all_edges = frozenset(graph_edges(trace.graph))
    spans = group_spans(trace.groups)
    rows: list[RowState] = []
    for iteration, span in zip(trace.iterations, spans, strict=True):
        created = iteration.created_node
        rows.append(
            RowState(
                index=iteration.index,
                token=iteration.emitted,
                ring_order=iteration.ring_after,
                primary=iteration.primary_after,
                secondary=iteration.secondary_after,
                span=span,
                present_nodes=all_nodes - frozenset(iteration.captured_nodes_after),
                present_edges=all_edges - frozenset(iteration.captured_edges_after),
                accent_nodes=frozenset() if created is None else frozenset({created}),
                accent_edges=(
                    frozenset()
                    if iteration.created_edge is None
                    else frozenset({iteration.created_edge})
                ),
                ring_accent=created,
            )
        )
    return tuple(rows)


def s2g_rows(trace: AlgorithmTrace, groups: tuple[str, ...]) -> tuple[RowState, ...]:
    """Reduce a decoder trace to one row per symbol group.

    The S2G graph panel *fills* while its instruction strip *drains*:
    the string is given whole and is consumed, where G2S is handed a
    graph and emits the string one group at a time. Both reversals come
    straight off
    :func:`~isalgraph.viz.worked_example.s2g_columns`, which already
    records ``strip_solid_side = "suffix"``.

    The milestones are the encoder's own groups, so step *k* of the two
    panels covers the same cells of the same string -- though the S2G
    panel stacks its steps bottom to top, so step *k* of the two is not
    at the same height. See *reverse* in :func:`_draw_panel_rows`.

    Args:
        trace: The ``"s2g"`` trace, from
            :func:`~isalgraph.viz.worked_example.decode_trace`.
        groups: The encoder's symbol groups.

    Returns:
        One row per group.
    """
    from isalgraph.viz.worked_example import s2g_columns

    return tuple(
        RowState(
            index=index,
            token=groups[index],
            ring_order=column.ring_order,
            primary=column.primary,
            secondary=column.secondary,
            span=column.span,
            present_nodes=column.present_nodes,
            present_edges=column.present_edges,
            accent_nodes=column.accent_nodes,
            accent_edges=column.accent_edges,
            ring_accent=column.ring_accent,
            strip_solid_side=column.strip_solid_side,
        )
        for index, column in enumerate(s2g_columns(trace, groups))
    )


def decoded_positions(
    encode: EncoderTrace,
    decode: AlgorithmTrace,
    decoded: SparseGraph,
    positions: dict[NodeId, Position],
) -> dict[NodeId, Position]:
    """Carry the pinned coordinates across the round trip.

    ``S2G(w)`` rebuilds the graph with its own node numbering -- node 0 is
    the seed and every ``V``/``v`` allocates the next id -- so the decoded
    graph drawn at *positions* would be a different picture of the same
    object, and the two panels would not look like inverses even though
    they are.

    The correspondence is not searched for; it is read off the two traces.
    Both are driven by the same string, so encoder iteration *k* and
    decoder group *k* create the same node, and the seed pairs with
    decoded node ``0``.

    Args:
        encode: The instrumented encode.
        decode: The ``"s2g"`` trace of the same string.
        decoded: The graph that trace built. Passed in rather than read
            off *decode*: :class:`~isalgraph.core.trace.AlgorithmTrace`
            has no ``graph`` attribute, and a ``hasattr`` guard here made
            the edge check below skip itself in silence -- which is the
            one failure mode a check like this must not have.
        positions: Coordinates keyed by *input*-graph node id.

    Returns:
        The same coordinates, keyed by decoded node id.

    Raises:
        VisualAbstractError: If the pairing is not a bijection, or does
            not carry the input graph's edge set onto the decoded one.
            That is the check on the whole construction: a mapping read
            off two traces that disagree must not be drawn.
    """
    from isalgraph.viz.worked_example import s2g_columns

    forward: dict[NodeId, NodeId] = {encode.start_node: decode.snapshots[0].primary_node}
    columns = s2g_columns(decode, encode.groups)
    for iteration, column in zip(encode.iterations, columns, strict=True):
        made = iteration.created_node
        if made is None:
            if column.accent_nodes:
                raise VisualAbstractError(
                    f"step {iteration.index + 1} creates no node when encoding but "
                    f"creates {sorted(column.accent_nodes)} when decoding"
                )
            continue
        if len(column.accent_nodes) != 1:
            raise VisualAbstractError(
                f"step {iteration.index + 1} creates node {made} when encoding but "
                f"{len(column.accent_nodes)} nodes when decoding"
            )
        forward[made] = next(iter(column.accent_nodes))

    if len(set(forward.values())) != len(forward) or set(forward) != set(positions):
        raise VisualAbstractError(
            f"the trace pairing covers {sorted(forward)} against pinned nodes "
            f"{sorted(positions)}; it is not a bijection on the drawn graph"
        )

    directed = encode.graph.directed()
    want = {
        _orient(forward[u], forward[v], directed=directed) for u, v in graph_edges(encode.graph)
    }
    got = set(graph_edges(decoded))
    if want != got:
        raise VisualAbstractError(
            "the trace pairing does not carry the input edge set onto the decoded one: "
            f"{sorted(want - got)} missing, {sorted(got - want)} spurious"
        )
    return {forward[node]: xy for node, xy in positions.items()}


def _orient(u: NodeId, v: NodeId, *, directed: bool) -> Edge:
    """Return the edge under the same orientation :func:`graph_edges` uses."""
    from isalgraph.core.trace import normalise_edge

    return normalise_edge(u, v, directed=directed)


def _ring_pos(order: tuple[NodeId, ...], node: NodeId) -> int:
    """Return the ring position of *node*, or ``0`` when absent."""
    try:
        return order.index(node)
    except ValueError:  # pragma: no cover - the pointers are always in the ring
        return 0


def _draw_row(  # noqa: PLR0913 -- one parameter per column of the row
    fig: Figure,
    row: RowState,
    *,
    graph: SparseGraph,
    instructions: str,
    positions: dict[NodeId, Position],
    lay: AbstractLayout,
    x_left: float,
    y_bottom: float,
    height: float,
) -> None:
    """Draw one execution row: step, CDLL ring, instruction strip, graph."""
    from isalgraph.viz.cdll_view import draw_cdll_ring
    from isalgraph.viz.instruction_view import draw_instruction_strip
    from isalgraph.viz.worked_example import draw_state_graph

    w_step, w_ring, w_strip, w_graph = lay.row_columns()

    def axes(x_in: float, w_in: float, pad: float = 0.0) -> Axes:
        return fig.add_axes(
            (
                x_in / lay.fig_width,
                (y_bottom + pad) / lay.fig_height,
                w_in / lay.fig_width,
                (height - 2 * pad) / lay.fig_height,
            )
        )

    x = x_left
    ax_step = axes(x, w_step)
    ax_step.axis("off")
    # Step number and emitted group, stacked. The number is what makes a
    # subsampled panel honest: four rows numbered 1, 3, 5 and 6 say the run
    # has six steps, where four rows labelled only by their tokens would
    # read as a four-step algorithm.
    ax_step.text(
        0.60,
        0.68,
        f"Step {row.index + 1}",
        ha="center",
        va="center",
        fontsize=lay.fs_row - 0.6,
        color="0.45",
        transform=ax_step.transAxes,
    )
    ax_step.text(
        0.60,
        0.28,
        row.token,
        ha="center",
        va="center",
        fontsize=lay.fs_row + 1.2,
        fontfamily="monospace",
        fontweight="bold",
        color=INSTRUCTION_PALETTE.get(row.token[-1], "#333333"),
        transform=ax_step.transAxes,
    )

    x += w_step
    draw_cdll_ring(
        axes(x, w_ring, pad=0.002),
        list(row.ring_order),
        _ring_pos(row.ring_order, row.primary),
        _ring_pos(row.ring_order, row.secondary),
        new_node_payload=row.ring_accent,
        new_node_color=ACCENT_COLOR,
        node_radius=lay.ring_node_radius,
        label_fontsize=lay.fs_ring_node,
        label_dy=lay.label_dy,
        pointer_fontsize=lay.fs_pointer,
        show_pointer_arrows=False,
        label_gap=lay.ring_label_gap,
        margin_pad=lay.ring_margin_pad,
    )

    x += w_ring
    draw_instruction_strip(
        axes(x, w_strip, pad=0.062),
        instructions,
        current_idx=row.span[1],
        solid_side=row.strip_solid_side,
        executing_span=row.span if lay.mark_executing else None,
        axis_width_inches=w_strip,
    )

    x += w_strip
    draw_state_graph(
        axes(x, w_graph, pad=0.002),
        graph,
        positions,
        present_nodes=row.present_nodes,
        present_edges=row.present_edges,
        accent_nodes=row.accent_nodes,
        accent_edges=row.accent_edges,
        primary_node=row.primary,
        secondary_node=row.secondary,
        node_radius=lay.node_radius,
        label_fontsize=lay.fs_graph_node,
        label_dy=lay.label_dy,
        pointer_ring_scale=1.0,
        accent_solid=True,
    )


# ---------------------------------------------------------------------------
# The two spaces and the four arrows
# ---------------------------------------------------------------------------


def _band(fig: Figure, box: Box, label: str, lay: AbstractLayout) -> None:
    """Draw one wide band: a rounded plate with its name at the left."""
    from matplotlib.patches import FancyBboxPatch

    x0, y0, x1, y1 = box
    fig.add_artist(
        FancyBboxPatch(
            (x0 / lay.fig_width, y0 / lay.fig_height),
            (x1 - x0) / lay.fig_width,
            (y1 - y0) / lay.fig_height,
            boxstyle="round,pad=0,rounding_size=0.010",
            transform=fig.transFigure,
            facecolor=BAND_FACE,
            edgecolor=BAND_EDGE,
            linewidth=0.7,
            zorder=-3,
        )
    )
    fig.text(
        (x0 + 0.10) / lay.fig_width,
        (y0 + y1) / 2.0 / lay.fig_height,
        label,
        ha="left",
        va="center",
        fontsize=lay.fs_band_title,
        color="#2A2F36",
        zorder=3,
    )


def _draw_graph_space(
    fig: Figure,
    box: Box,
    lay: AbstractLayout,
    *,
    graph: SparseGraph,
    positions: dict[NodeId, Position],
) -> None:
    """Draw the finished graph, centred in the graph-space band.

    ``draw_state_graph`` sets an equal aspect on explicit limits, so the
    drawing letterboxes itself inside whatever axes it is given: the band
    can be as wide as the two machines beneath it without the graph
    stretching to fill it.
    """
    from isalgraph.viz.worked_example import draw_state_graph

    # The band is wide and short and the drawing keeps an equal aspect, so
    # it is bound by the band's *height*: every thousandth of an inch of
    # inset here comes straight off the graph.
    inset = 0.015
    ax = fig.add_axes(
        (
            (box[0] + inset) / lay.fig_width,
            (box[1] + inset) / lay.fig_height,
            (box[2] - box[0] - 2 * inset) / lay.fig_width,
            (box[3] - box[1] - 2 * inset) / lay.fig_height,
        )
    )
    ax.patch.set_alpha(0.0)
    draw_state_graph(
        ax,
        graph,
        positions,
        present_nodes=frozenset(range(graph.node_count())),
        present_edges=frozenset(graph_edges(graph)),
        node_radius=lay.band_node_radius,
        label_fontsize=lay.fs_band_node,
        label_dy=lay.label_dy,
        pointer_ring_scale=1.0,
    )


def _draw_instruction_space(fig: Figure, box: Box, lay: AbstractLayout, string: str) -> None:
    """Draw the canonical string, centred in the instruction-space band."""
    from isalgraph.viz.instruction_view import draw_instruction_strip

    # The strip's axes carries an "auto" aspect, so it stretches to fill
    # whatever box it is given and the cells go rectangular if the box
    # does. Its width is therefore *derived* from its height rather than
    # set as a fraction of the band: ``_strip_axes`` spans
    # ``n * cell_width + 0.2`` by ``cell_height + 0.1`` in data units, and
    # matching that ratio is what keeps the cells square however thin the
    # band is made.
    pad = 0.025
    height = box[3] - box[1] - 2 * pad
    cells = len(string)
    width = min(
        height * (cells * STRIP_CELL_WIDTH + 0.2) / (STRIP_CELL_HEIGHT + 0.1),
        box[2] - box[0] - 0.2,
    )
    x0 = (box[0] + box[2]) / 2.0 - width / 2.0
    ax = fig.add_axes(
        (
            x0 / lay.fig_width,
            (box[1] + pad) / lay.fig_height,
            width / lay.fig_width,
            height / lay.fig_height,
        )
    )
    ax.patch.set_alpha(0.0)
    draw_instruction_strip(
        ax,
        string,
        current_idx=len(string),
        solid_side="prefix",
        axis_width_inches=width,
    )


def _stream_arrow(  # noqa: PLR0913 -- an arrow with a label needs its side
    fig: Figure,
    x_in: float,
    y0: float,
    y1: float,
    lay: AbstractLayout,
    *,
    label: str = "",
    side: str = "left",
) -> None:
    """Draw one vertical stream arrow from ``y0`` to ``y1``, in inches.

    Args:
        fig: Target figure.
        x_in: Arrow abscissa, in inches.
        y0: Tail ordinate, in inches.
        y1: Head ordinate, in inches.
        lay: Printed geometry.
        label: Name of the stream, or empty for an unlabelled arrow.
        side: Which side of the arrow the label sits on. Set it *away*
            from the figure's centre on both streams; between them is
            where the two panels meet and the two labels would converge.
    """
    from matplotlib.patches import FancyArrowPatch

    fig.add_artist(
        FancyArrowPatch(
            (x_in / lay.fig_width, y0 / lay.fig_height),
            (x_in / lay.fig_width, y1 / lay.fig_height),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=lay.arrow_head,
            linewidth=lay.arrow_lw,
            color=STREAM_COLOR,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=5,
        )
    )
    if not label:
        return
    offset = 0.055 if side == "right" else -0.055
    fig.text(
        (x_in + offset) / lay.fig_width,
        (y0 + y1) / 2.0 / lay.fig_height,
        label,
        ha="left" if side == "right" else "right",
        va="center",
        fontsize=lay.fs_stream,
        color=STREAM_COLOR,
        zorder=5,
    )


def _algorithm_panel(fig: Figure, box: Box, title: str, lay: AbstractLayout) -> None:
    """Outline one algorithm panel and write its name in the title bar."""
    from matplotlib.patches import FancyBboxPatch

    x0, y0, x1, y1 = box
    fig.add_artist(
        FancyBboxPatch(
            (x0 / lay.fig_width, y0 / lay.fig_height),
            (x1 - x0) / lay.fig_width,
            (y1 - y0) / lay.fig_height,
            boxstyle="round,pad=0,rounding_size=0.012",
            transform=fig.transFigure,
            facecolor="#FFFFFF",
            edgecolor=ZOOM_EDGE,
            linewidth=0.8,
            zorder=-4,
        )
    )
    fig.text(
        (x0 + x1) / 2.0 / lay.fig_width,
        (y1 - lay.box_title / 2.0) / lay.fig_height,
        title,
        ha="center",
        va="center",
        fontsize=lay.fs_box_title,
        color="#101317",
        zorder=3,
    )


def _magnify(fig: Figure, source: Box, target: Box, lay: AbstractLayout) -> None:
    """Run two lines from the boxed canonical path to the panel that walks it.

    No bracket round the target. The paper figure draws one because its
    panel (b) has no outline of its own; here the G2S panel is already a
    rounded box, and a second outline over it reads as a seam.

    Both boxes are in inches.
    """
    from matplotlib.lines import Line2D

    sx1, sy0, sy1 = source[2], source[1], source[3]
    tx0, ty0, ty1 = target[0], target[1], target[3]
    for sy, ty in ((sy1, ty1), (sy0, ty0)):
        fig.add_artist(
            Line2D(
                [sx1 / lay.fig_width, tx0 / lay.fig_width],
                [sy / lay.fig_height, ty / lay.fig_height],
                transform=fig.transFigure,
                color=ZOOM_EDGE,
                linewidth=0.6,
                zorder=4,
            )
        )


def _box_to_inches(ax: Axes, fig: Figure, box: Box, lay: AbstractLayout) -> Box:
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


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Panels:
    """The five panel boxes, in inches, and the two panel centres."""

    graph: Box
    g2s: Box
    s2g: Box
    instructions: Box
    g2s_cx: float
    s2g_cx: float


def _panel_boxes(lay: AbstractLayout) -> _Panels:
    """Solve the right column's vertical stack, top to bottom."""
    top = lay.content_top
    graph_box = (lay.right_x, top - lay.graph_band, lay.right_x + lay.right_width, top)
    box_top = graph_box[1] - lay.stream_gap
    box_bottom = box_top - lay.box_height
    g2s = (lay.right_x, box_bottom, lay.right_x + lay.box_width, box_top)
    s2g_x0 = lay.right_x + lay.box_width + lay.box_gap
    s2g = (s2g_x0, box_bottom, s2g_x0 + lay.box_width, box_top)
    instr_top = box_bottom - lay.stream_gap
    instructions = (
        lay.right_x,
        instr_top - lay.instr_band,
        lay.right_x + lay.right_width,
        instr_top,
    )
    return _Panels(
        graph=graph_box,
        g2s=g2s,
        s2g=s2g,
        instructions=instructions,
        g2s_cx=(g2s[0] + g2s[2]) / 2.0,
        s2g_cx=(s2g[0] + s2g[2]) / 2.0,
    )


def select_rows(
    rows: tuple[RowState, ...],
    *,
    steps: tuple[int, ...] | None,
    max_rows: int | None,
) -> tuple[RowState, ...]:
    """Return the rows to draw, subsampling the run if it is too long.

    The subsample always keeps the first and the last step, so a panel
    still reads as a complete run from an empty state to a finished one,
    and spaces the rest evenly between them.

    Eliding steps is visible, not silent: every row is labelled with its
    own step number, and the instruction strip of a row that follows an
    elision is already solid past where the row above it stopped.

    Args:
        rows: Every row of the run.
        steps: Zero-based indices to keep. Overrides *max_rows*.
        max_rows: Cap on the number of rows, or ``None`` for no cap.

    Returns:
        The selected rows, in run order.

    Raises:
        VisualAbstractError: If an index is out of range, or *max_rows*
            is below two, which cannot hold both ends of a run.
    """
    import math

    if steps is not None:
        bad = [s for s in steps if not 0 <= s < len(rows)]
        if bad:
            raise VisualAbstractError(f"steps {bad} are outside the run's 0..{len(rows) - 1}")
        return tuple(rows[s] for s in sorted(steps))
    if max_rows is None or len(rows) <= max_rows:
        return rows
    if max_rows < 2:
        raise VisualAbstractError(f"max_rows is {max_rows}; a run needs both of its ends")
    last = len(rows) - 1
    picked = sorted({math.ceil(i * last / (max_rows - 1)) for i in range(max_rows)})
    return tuple(rows[i] for i in picked)


def _draw_panel_rows(  # noqa: PLR0913 -- one parameter per drawing input
    fig: Figure,
    rows: tuple[RowState, ...],
    box: Box,
    *,
    graph: SparseGraph,
    instructions: str,
    positions: dict[NodeId, Position],
    lay: AbstractLayout,
    reverse: bool = False,
) -> None:
    """Stack *rows* inside an algorithm panel, under its title bar.

    Args:
        fig: Target figure.
        rows: The rows to draw, in run order.
        box: The panel, in inches.
        graph: The graph the rows draw.
        instructions: The string the rows execute.
        positions: Node coordinates for the graph column.
        lay: Printed geometry.
        reverse: Stack the run bottom to top, so step one is the lowest
            row. Set on the S2G panel: its stream arrives from the
            instruction band *below* it and leaves for the graph band
            *above* it, so a top-down stack would have the reader enter
            the panel at its last step. Reversed, the two panels read as
            one continuous loop rather than as two lists.
    """
    body_top = box[3] - lay.box_title
    body_h = body_top - box[1] - lay.box_inset
    height = body_h / len(rows)
    count = len(rows)
    for index, row in enumerate(rows):
        slot = count - index if reverse else index + 1
        _draw_row(
            fig,
            row,
            graph=graph,
            instructions=instructions,
            positions=positions,
            lay=lay,
            x_left=box[0] + lay.box_inset,
            y_bottom=body_top - slot * height,
            height=height,
        )


def _assemble(  # noqa: PLR0913 -- the abstract has five panels
    graph: SparseGraph,
    tree: SearchTree,
    encode: EncoderTrace,
    *,
    branch_depth: int,
    positions: dict[NodeId, Position] | None,
    layout: AbstractLayout | None,
    survivors: frozenset[int] | None = None,
) -> Figure:
    """Lay out the five panels from a prepared tree and encode.

    Raises:
        VisualAbstractError: If the encode does not walk the highlighted
            path, so the boxed path and the G2S panel would disagree.
    """
    import matplotlib.pyplot as plt

    from isalgraph.viz.worked_example import decode_trace

    lay = layout or AbstractLayout()
    if encode.instruction_string != tree.canonical:
        raise VisualAbstractError(
            f"the encode from node {encode.start_node} emits "
            f"{encode.instruction_string!r}, which is not the highlighted string "
            f"{tree.canonical!r}; the G2S panel would walk a path the tree does not box"
        )

    string = encode.instruction_string
    pos = _graph_positions(graph, positions)
    decoded, decode = decode_trace(string, directed=graph.directed())
    pos_decoded = decoded_positions(encode, decode, decoded, pos)

    rows_g2s = select_rows(g2s_rows(encode), steps=lay.steps, max_rows=lay.max_rows)
    rows_s2g = select_rows(s2g_rows(decode, encode.groups), steps=lay.steps, max_rows=lay.max_rows)

    fig = plt.figure(figsize=lay.figsize)
    panels = _panel_boxes(lay)

    # ---- the search space ------------------------------------------------
    title_top = lay.content_top
    tree_top = title_top - lay.tree_title_height
    tree_bottom = lay.content_bottom + lay.key_height
    fig.text(
        (lay.pad + lay.tree_width / 2.0) / lay.fig_width,
        (title_top - lay.tree_title_height / 2.0) / lay.fig_height,
        "Canonical Search Space",
        ha="center",
        va="center",
        fontsize=lay.fs_panel_title,
        color="#101317",
    )
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
    )

    ax_key = fig.add_axes(
        (
            lay.pad / lay.fig_width,
            lay.content_bottom / lay.fig_height,
            lay.tree_width / lay.fig_width,
            lay.key_height / lay.fig_height,
        )
    )
    ax_key.axis("off")
    live = [lf for lf in tree.leaves() if survivors is None or lf.index in survivors]
    ax_key.legend(
        handles=_search_space_key(
            min(len(leaf.prefix) for leaf in live), pruned=survivors is not None
        ),
        loc="center",
        fontsize=lay.fs_legend,
        ncol=2,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.4,
        labelspacing=0.35,
        borderpad=0.0,
    )

    # ---- the two spaces --------------------------------------------------
    _band(fig, panels.graph, "Graph Space", lay)
    _draw_graph_space(fig, panels.graph, lay, graph=graph, positions=pos)
    _band(fig, panels.instructions, "Instruction Space", lay)
    _draw_instruction_space(fig, panels.instructions, lay, string)

    # ---- the two machines ------------------------------------------------
    _algorithm_panel(fig, panels.g2s, "Graph-To-String", lay)
    _draw_panel_rows(
        fig,
        rows_g2s,
        panels.g2s,
        graph=graph,
        instructions=string,
        positions=pos,
        lay=lay,
    )
    _algorithm_panel(fig, panels.s2g, "String-To-Graph", lay)
    _draw_panel_rows(
        fig,
        rows_s2g,
        panels.s2g,
        graph=decoded,
        instructions=string,
        positions=pos_decoded,
        lay=lay,
        reverse=True,
    )

    # ---- the round trip --------------------------------------------------
    # Down the left, up the right. Graph space -> G2S -> instruction space
    # is the encode; instruction space -> S2G -> graph space is the decode,
    # and the two together are the round-trip theorem.
    _stream_arrow(
        fig, panels.g2s_cx, panels.graph[1], panels.g2s[3], lay, label="Encode", side="left"
    )
    _stream_arrow(fig, panels.g2s_cx, panels.g2s[1], panels.instructions[3], lay)
    _stream_arrow(fig, panels.s2g_cx, panels.instructions[3], panels.s2g[1], lay)
    _stream_arrow(
        fig, panels.s2g_cx, panels.s2g[3], panels.graph[1], lay, label="Decode", side="right"
    )

    # ---- the magnification, search space into G2S ------------------------
    _magnify(fig, _box_to_inches(ax_tree, fig, data_box, lay), panels.g2s, lay)
    return fig


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


def visual_abstract_figure(  # noqa: PLR0913 -- the abstract has five panels
    graph: SparseGraph,
    *,
    start_node: NodeId = 0,
    branch_depth: int = DEFAULT_BRANCH_DEPTH,
    max_depth: int = 24,
    max_nodes: int = 20_000,
    positions: dict[NodeId, Position] | None = None,
    layout: AbstractLayout | None = None,
) -> Figure:
    """Build the graphical abstract for the **exhaustive** canonicalisation.

    Args:
        graph: A small graph, 6-8 nodes for legibility.
        start_node: The node the executed path starts from. It must be
            one that attains the canonical string, or the boxed path and
            the G2S panel show different executions.
        branch_depth: Last step at which the search still branches.
        max_depth: Depth budget for enumeration.
        max_nodes: Enumeration size cap.
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
        positions=positions,
        layout=layout,
    )


def pruned_visual_abstract_figure(
    graph: SparseGraph,
    *,
    branch_depth: int = DEFAULT_BRANCH_DEPTH,
    max_depth: int = 24,
    max_nodes: int = 20_000,
    positions: dict[NodeId, Position] | None = None,
    layout: AbstractLayout | None = None,
) -> Figure:
    """Build the same abstract for the **pruned** canonicalisation.

    Args:
        graph: A small graph, 6-8 nodes for legibility.
        branch_depth: Last step at which the exhaustive search branches.
        max_depth: Depth budget for enumeration.
        max_nodes: Enumeration size cap.
        positions: Pinned node coordinates for the graph panels.
        layout: Printed geometry.

    Returns:
        The created figure. The caller owns it and must close it.

    Raises:
        VisualAbstractError: If the surviving leaves do not reproduce
            :func:`~isalgraph.core.canonical_pruned.pruned_canonical_string`.
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
        raise VisualAbstractError("pruning left no complete path")
    best = min((len(lf.prefix), lf.prefix) for lf in live)[1]
    if best != target:
        raise VisualAbstractError(
            f"the pruned survivors minimise to {best!r} but pruned_canonical_string "
            f"is {target!r}; the filter in prune_survivors does not model the pruned search"
        )

    remark_optimal(tree, target, survivors=survivors)
    return _assemble(
        graph,
        tree,
        trace_execution(graph, target),
        branch_depth=branch_depth,
        positions=positions,
        layout=layout,
        survivors=survivors,
    )


__all__ = [
    "ABSTRACT_MONO",
    "ABSTRACT_SERIF",
    "DEFAULT_BRANCH_DEPTH",
    "ELSEVIER_MIN_PIXELS",
    "ELSEVIER_RATIO",
    "STRIP_CELL_HEIGHT",
    "STRIP_CELL_WIDTH",
    "AbstractLayout",
    "RowState",
    "VisualAbstractError",
    "apply_abstract_style",
    "decoded_positions",
    "save_abstract",
    "draw_search_space",
    "g2s_rows",
    "prune_survivors",
    "pruned_visual_abstract_figure",
    "remark_optimal",
    "s2g_rows",
    "select_rows",
    "visual_abstract_figure",
]
