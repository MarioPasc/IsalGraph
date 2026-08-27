"""Schematic of the canonical-string search space.

Requested by Reviewer 3 for Section 2.3: a small diagram separating what
the canonical search *branches over* from what it holds *fixed*.

The distinction is exact, and reading
:func:`isalgraph.core.canonical._step` is what settles it:

* **Branching.** Two things multiply the search space. First, the choice
  of starting node -- :func:`isalgraph.core.canonical.canonical_string`
  runs an independent search from every node that reaches all others,
  and keeps the ``(len, lex)``-smallest result. Second, the choice of
  *which* uninserted neighbour to attach at a ``V`` or ``v`` step:
  ``_step`` loops over the whole candidate set and recurses on each.
* **Fixed.** Displacement ordering does not branch. ``_step`` walks the
  pairs ``(a, b)`` in increasing ``|a| + |b|`` and commits to the first
  one admitting any operation -- every ``V``/``v``/``C``/``c`` arm ends
  in ``return``, so no later pair is ever examined at that depth. Nor
  does the operation priority branch: ``V`` is tried before ``v``,
  ``v`` before ``C``, ``C`` before ``c``, and the first that applies
  wins outright.

So the search tree is shallow-branching by construction: it fans out
only at starting nodes and at ``V``/``v`` candidate sets, and every other
decision is forced. That is what the figure shows -- branch edges drawn
solid and fanned, forced steps drawn as a single stem annotated with the
constraint that forced it.

The enumerator below replays the algorithm from the root along a fixed
choice sequence rather than mutating with undo. Replay costs more than
backtracking, but it needs no access to the private undo helpers in
``canonical.py``, and for the 6-8 node graphs a schematic uses the
difference is irrelevant. :func:`enumerate_search_tree` is checked
against :func:`~isalgraph.core.canonical.canonical_string` in the test
suite, so the schematic cannot drift from the algorithm it depicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from isalgraph.core.canonical import canonical_string
from isalgraph.core.cdll import CircularDoublyLinkedList
from isalgraph.core.graph_to_string import generate_pairs_sorted_by_sum
from isalgraph.core.sparse_graph import SparseGraph
from isalgraph.types import NodeId
from isalgraph.viz.style import (
    GRAYED_EDGE,
    INSTRUCTION_PALETTE,
    PATREC_TEXT_WIDTH_INCHES,
    POINTER_PALETTE,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
else:
    Axes = Any
    Figure = Any


# ---------------------------------------------------------------------------
# Replay primitives (duplicated from canonical.py, which is read-only here)
# ---------------------------------------------------------------------------


def _walk(cdll: CircularDoublyLinkedList, ptr: int, steps: int) -> int:
    """Walk *ptr* forward (``steps >= 0``) or backward through the CDLL."""
    for _ in range(abs(steps)):
        ptr = cdll.next_node(ptr) if steps >= 0 else cdll.prev_node(ptr)
    return ptr


def _primary_moves(a: int) -> str:
    """Return the ``N``/``P`` run realising a primary displacement of *a*."""
    return "N" * a if a >= 0 else "P" * (-a)


def _secondary_moves(b: int) -> str:
    """Return the ``n``/``p`` run realising a secondary displacement of *b*."""
    return "n" * b if b >= 0 else "p" * (-b)


# ---------------------------------------------------------------------------
# Tree model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchStep:
    """One committed step along a search path.

    Args:
        op: ``"V"``, ``"v"``, ``"C"`` or ``"c"``.
        displacement: The ``(a, b)`` pair the step committed to.
        segment: The instruction substring emitted, movements included.
        chosen: The input-graph node attached, for ``V``/``v`` steps.
        n_candidates: Size of the candidate set at this step. ``1`` means
            the step was forced even though it is a branch point in kind.
    """

    op: str
    displacement: tuple[int, int]
    segment: str
    chosen: int | None
    n_candidates: int


@dataclass
class SearchTreeNode:
    """A node of the enumerated search tree.

    Args:
        index: Unique id within the tree.
        parent: Parent index, or ``None`` for a root.
        start_node: The starting node whose subtree this belongs to.
        step: The step that produced this node, or ``None`` for a root.
        prefix: Full instruction string accumulated down to here.
        depth: Distance from the root, in committed steps.
        children: Child indices, filled during enumeration.
        terminal: Whether the encoding is complete at this node.
        truncated: Whether enumeration stopped here on the depth budget.
        optimal: Whether this node lies on the canonical path.
    """

    index: int
    parent: int | None
    start_node: int
    step: SearchStep | None
    prefix: str
    depth: int
    children: list[int] = field(default_factory=list)
    terminal: bool = False
    truncated: bool = False
    optimal: bool = False


@dataclass
class SearchTree:
    """An enumerated canonical-search tree.

    Args:
        nodes: All tree nodes, indexed by :attr:`SearchTreeNode.index`.
        roots: Indices of the per-starting-node roots.
        canonical: The canonical string, as returned by
            :func:`~isalgraph.core.canonical.canonical_string`.
        max_depth: The depth budget enumeration ran under.
    """

    nodes: list[SearchTreeNode]
    roots: list[int]
    canonical: str
    max_depth: int

    def children_of(self, index: int) -> list[SearchTreeNode]:
        """Return the child nodes of the node at *index*."""
        return [self.nodes[i] for i in self.nodes[index].children]

    def leaves(self) -> list[SearchTreeNode]:
        """Return every node with no children."""
        return [n for n in self.nodes if not n.children]


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


class _Replay:
    """Deterministic replay of the canonical search along fixed choices."""

    def __init__(self, ig: SparseGraph, start_node: int) -> None:
        max_n = ig.max_nodes()
        self.ig = ig
        self.og = SparseGraph(max_n, ig.directed())
        self.cdll = CircularDoublyLinkedList(max_n)
        n0 = self.og.add_node()
        c0 = self.cdll.insert_after(-1, n0)
        self.pri = c0
        self.sec = c0
        self.i2o: dict[int, int] = {start_node: n0}
        self.o2i: dict[int, int] = {n0: start_node}
        self.nleft = ig.node_count() - 1
        self.eleft = ig.logical_edge_count()
        self.prefix = ""
        self.steps: list[SearchStep] = []

    def _insert(self, op: str, ptr: int, out: int, chosen: int, disp: tuple[int, int]) -> None:
        """Apply a ``V``/``v`` step for *chosen* and record it."""
        new_out = self.og.add_node()
        self.i2o[chosen] = new_out
        self.o2i[new_out] = chosen
        self.og.add_edge(out, new_out)
        self.cdll.insert_after(ptr, new_out)
        a, b = disp
        segment = (_primary_moves(a) if op == "V" else _secondary_moves(b)) + op
        self.prefix += segment
        self.steps.append(SearchStep(op, disp, segment, chosen, 0))
        if op == "V":
            self.pri = ptr
        else:
            self.sec = ptr
        self.nleft -= 1
        self.eleft -= 1

    def _connect(self, op: str, tp: int, ts: int, disp: tuple[int, int]) -> None:
        """Apply a ``C``/``c`` step and record it."""
        a, b = disp
        tp_out, ts_out = self.cdll.get_value(tp), self.cdll.get_value(ts)
        source, target = (tp_out, ts_out) if op == "C" else (ts_out, tp_out)
        self.og.add_edge(source, target)
        segment = _primary_moves(a) + _secondary_moves(b) + op
        self.prefix += segment
        self.steps.append(SearchStep(op, disp, segment, None, 1))
        self.pri, self.sec = tp, ts
        self.eleft -= 1

    def advance(self, choices: list[int]) -> tuple[str, list[int], tuple[int, int]] | None:
        """Run until the first unresolved ``V``/``v`` branch, or to completion.

        Args:
            choices: Candidate nodes to consume, in order, at successive
                ``V``/``v`` branch points.

        Returns:
            ``(op, candidates, displacement)`` at the first branch not
            covered by *choices*, or ``None`` when the encoding completed
            or no operation applies.
        """
        consumed = 0
        while self.nleft > 0 or self.eleft > 0:
            acted = False
            for a, b in generate_pairs_sorted_by_sum(self.og.node_count()):
                tp = _walk(self.cdll, self.pri, a)
                tp_out = self.cdll.get_value(tp)
                tp_in = self.o2i[tp_out]

                if self.nleft > 0:
                    cands = [x for x in self.ig.neighbors(tp_in) if x not in self.i2o]
                    if cands:
                        if consumed >= len(choices):
                            return ("V", sorted(cands), (a, b))
                        self._insert("V", tp, tp_out, choices[consumed], (a, b))
                        self.steps[-1] = SearchStep(
                            "V", (a, b), self.steps[-1].segment, choices[consumed], len(cands)
                        )
                        consumed += 1
                        acted = True
                        break

                ts = _walk(self.cdll, self.sec, b)
                ts_out = self.cdll.get_value(ts)
                ts_in = self.o2i[ts_out]

                if self.nleft > 0:
                    cands = [x for x in self.ig.neighbors(ts_in) if x not in self.i2o]
                    if cands:
                        if consumed >= len(choices):
                            return ("v", sorted(cands), (a, b))
                        self._insert("v", ts, ts_out, choices[consumed], (a, b))
                        self.steps[-1] = SearchStep(
                            "v", (a, b), self.steps[-1].segment, choices[consumed], len(cands)
                        )
                        consumed += 1
                        acted = True
                        break

                if ts_in in self.ig.neighbors(tp_in) and ts_out not in self.og.neighbors(tp_out):
                    self._connect("C", tp, ts, (a, b))
                    acted = True
                    break

                if (
                    self.ig.directed()
                    and tp_in in self.ig.neighbors(ts_in)
                    and tp_out not in self.og.neighbors(ts_out)
                ):
                    self._connect("c", tp, ts, (a, b))
                    acted = True
                    break

            if not acted:
                break
        return None


def _canonical_from(graph: SparseGraph, start: int) -> str:
    """Return the shortest-then-lex-smallest encoding rooted at *start*.

    Enumerates leaves by replay rather than calling the private
    ``canonical._canonical_g2s``, so this module depends on nothing
    private in ``core``.
    """
    best: str | None = None
    frontier: list[list[int]] = [[]]
    while frontier:
        choices = frontier.pop()
        replay = _Replay(graph, start)
        branch = replay.advance(choices)
        if branch is None:
            if replay.nleft <= 0 and replay.eleft <= 0:  # noqa: SIM102
                if best is None or (len(replay.prefix), replay.prefix) < (len(best), best):
                    best = replay.prefix
            continue
        _op, candidates, _disp = branch
        frontier.extend([*choices, c] for c in candidates)
    return best if best is not None else ""


def _reachable(graph: SparseGraph, start: int) -> bool:
    """Whether every node is reachable from *start* via outgoing edges."""
    seen = {start}
    stack = [start]
    while stack:
        for nb in graph.neighbors(stack.pop()):
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == graph.node_count()


def enumerate_search_tree(
    graph: SparseGraph,
    *,
    max_depth: int = 3,
    max_nodes: int = 400,
    max_roots: int | None = None,
) -> SearchTree:
    """Enumerate the canonical search tree of *graph* to a depth budget.

    Args:
        graph: The graph to encode. Small graphs only; the tree is meant
            to be *read*, so 6-8 nodes and ``max_depth`` around 3 is the
            useful range.
        max_depth: Committed steps to expand before truncating a path.
        max_nodes: Hard cap on tree size, as a runaway guard.
        max_roots: Show at most this many starting nodes. The full search
            roots at *every* node that reaches all others, which for a
            seven-node graph is seven subtrees and far too dense to read;
            capping the count keeps the schematic legible. The root that
            yields the canonical string is always retained.

    Returns:
        The enumerated :class:`SearchTree`, with the canonical path
        marked via :attr:`SearchTreeNode.optimal`.

    Raises:
        ValueError: If no starting node reaches every other node.
    """
    starts = [v for v in range(graph.node_count()) if _reachable(graph, v)]
    if not starts:
        raise ValueError("no starting node reaches every other node")

    canonical = canonical_string(graph)

    if max_roots is not None and len(starts) > max_roots:
        winners = [v for v in starts if _canonical_from(graph, v) == canonical]
        keep = winners[:1]
        keep += [v for v in starts if v not in keep][: max_roots - len(keep)]
        starts = sorted(keep)
    nodes: list[SearchTreeNode] = []
    roots: list[int] = []

    # (tree index, choice sequence leading to it)
    frontier: list[tuple[int, list[int]]] = []
    for start in starts:
        idx = len(nodes)
        nodes.append(SearchTreeNode(idx, None, start, None, "", 0))
        roots.append(idx)
        frontier.append((idx, []))

    while frontier and len(nodes) < max_nodes:
        parent_idx, choices = frontier.pop(0)
        parent = nodes[parent_idx]
        if parent.depth >= max_depth:
            parent.truncated = True
            continue

        replay = _Replay(graph, parent.start_node)
        branch = replay.advance(choices)

        if branch is None:
            # No further branch: the remaining path is forced. Attach it
            # as a single chain so the figure shows forced steps as a stem.
            forced = replay.steps[len(choices) :]
            current = parent_idx
            for step in forced[: max_depth - parent.depth]:
                child_idx = len(nodes)
                nodes.append(
                    SearchTreeNode(
                        child_idx,
                        current,
                        parent.start_node,
                        step,
                        nodes[current].prefix + step.segment,
                        nodes[current].depth + 1,
                    )
                )
                nodes[current].children.append(child_idx)
                current = child_idx
            nodes[current].terminal = replay.nleft <= 0 and replay.eleft <= 0
            nodes[current].truncated = not nodes[current].terminal
            continue

        op, candidates, displacement = branch
        for cand in candidates:
            child_choices = [*choices, cand]
            child_replay = _Replay(graph, parent.start_node)
            child_replay.advance(child_choices)
            if len(child_replay.steps) <= len(choices):
                continue
            step = child_replay.steps[len(choices)]
            child_idx = len(nodes)
            nodes.append(
                SearchTreeNode(
                    child_idx,
                    parent_idx,
                    parent.start_node,
                    SearchStep(op, displacement, step.segment, cand, len(candidates)),
                    parent.prefix + step.segment,
                    parent.depth + 1,
                )
            )
            parent.children.append(child_idx)
            frontier.append((child_idx, child_choices))

    _mark_optimal(nodes, roots, canonical)
    return SearchTree(nodes=nodes, roots=roots, canonical=canonical, max_depth=max_depth)


def _mark_optimal(nodes: list[SearchTreeNode], roots: list[int], canonical: str) -> None:
    """Flag one root-to-frontier path realising *canonical*.

    Marking *every* node whose prefix is a prefix of the canonical string
    would light up several sibling branches at once, because IsalGraph
    strings are labelling-independent: at ``v0`` above, attaching
    neighbour 1 and attaching neighbour 3 both emit ``"V"`` and both lead
    to the same canonical string. That is a real property of the
    encoding, but on a schematic it reads as a drawing bug. So exactly
    one path is marked -- descending from the first viable root and, at
    each level, taking the child with the smallest candidate id among
    those still consistent with *canonical*.
    """
    if not canonical:
        return

    for root in roots:
        current = root
        path = [current]
        while True:
            viable = [
                c
                for c in nodes[current].children
                if canonical.startswith(nodes[c].prefix) and nodes[c].prefix
            ]
            if not viable:
                break

            def _candidate_rank(index: int) -> int:
                step = nodes[index].step
                return -1 if step is None or step.chosen is None else step.chosen

            current = min(viable, key=_candidate_rank)
            path.append(current)
        if len(path) > 1:
            for idx in path:
                nodes[idx].optimal = True
            return


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _layout_tree(tree: SearchTree) -> dict[int, tuple[float, float]]:
    """Assign ``(x, y)`` to every tree node; leaves evenly spaced on x."""
    pos: dict[int, tuple[float, float]] = {}
    cursor = [0.0]

    def place(idx: int) -> float:
        node = tree.nodes[idx]
        if not node.children:
            x = cursor[0]
            cursor[0] += 1.0
        else:
            xs = [place(c) for c in node.children]
            x = sum(xs) / len(xs)
        pos[idx] = (x, -float(node.depth))
        return x

    for root in tree.roots:
        place(root)
        cursor[0] += 0.6  # gutter between starting-node subtrees
    return pos


#: Underlay colour marking the canonical path. Deliberately outside the
#: branch palette: the canonical path runs *along* branch edges, so
#: highlighting it in the same red used for V branches makes the two
#: unreadable. A wide pale-gold halo beneath the edge reads as emphasis
#: rather than as a fifth edge category.
CANONICAL_HALO: str = "#DDAA33"


def draw_search_tree(
    ax: Axes,
    tree: SearchTree,
    *,
    label_fontsize: float = 6.0,
    node_points: float = 62.0,
) -> None:
    """Draw *tree* on *ax*.

    Branch edges -- a ``V``/``v`` step with more than one candidate -- are
    solid and coloured by the acting pointer. Forced steps -- a lone
    candidate, or any ``C``/``c`` -- are dashed grey. That contrast is the
    figure's whole argument: displacement ordering and the
    ``V > v > C > c`` priority never fan out.

    Nodes are drawn as scatter markers rather than ``Circle`` patches
    because the axes cannot be equal-aspect: a tree wide enough to hold
    every leaf and only three rows deep would be unreadably wide at equal
    aspect, and a ``Circle`` under unequal aspect renders as an ellipse.
    Marker area is specified in points, so it stays circular whatever the
    axes box does.

    Args:
        ax: Target matplotlib axes.
        tree: The enumerated tree.
        label_fontsize: Point size for edge and node labels.
        node_points: Node marker area, in points squared.
    """
    pos = _layout_tree(tree)

    for node in tree.nodes:
        if node.parent is None:
            continue
        x0, y0 = pos[node.parent]
        x1, y1 = pos[node.index]
        step = node.step
        assert step is not None
        branching = step.n_candidates > 1
        color = POINTER_PALETTE[0 if step.op in ("V", "C") else 1] if branching else GRAYED_EDGE
        if node.optimal:
            ax.plot([x0, x1], [y0, y1], color=CANONICAL_HALO, lw=4.5, zorder=0, alpha=0.85)
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=color,
            lw=1.3 if branching else 0.9,
            ls="-" if branching else (0, (2.2, 1.6)),
            zorder=1,
            solid_capstyle="round",
        )
        # Sibling edges share a parent, so their midpoints are half a child
        # spacing apart -- close enough that two two-character segments
        # ("pv" against "pv") printed there overlap illegibly. Siblings
        # diverge toward their children, so placing the label at 0.70 of the
        # way down separates them by 40 % more without moving it off its
        # own edge.
        label_t = 0.70
        ax.text(
            x0 + label_t * (x1 - x0),
            y0 + label_t * (y1 - y0),
            step.segment,
            fontsize=label_fontsize,
            fontfamily="monospace",
            color=INSTRUCTION_PALETTE.get(step.op, "#333333"),
            ha="center",
            va="center",
            zorder=4,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.85},
        )

    for node in tree.nodes:
        x, y = pos[node.index]
        # Every non-root node is labelled with the input-graph node its step
        # attached, leaf rows included. Leaving the leaf row blank made it
        # read as a row of empty markers -- the depth budget truncates what
        # comes *after* a node, which says nothing about how it was reached.
        chosen = node.step.chosen if node.step is not None else None
        label = "" if chosen is None else str(chosen)
        if node.parent is None:
            # Plain node ids, not "$v_k$". Every other mark in this figure
            # set names a node of G by its integer: the interior and leaf
            # labels here, the inset, and the four worked-example panels
            # (``draw_state_graph`` writes ``str(node)``). A root drawn as
            # "$v_3$" beside a graph drawn as "3" makes the reader translate
            # between two notations for one object. The start-node role is
            # already carried by the fill colour and the row label.
            face, label, text_color = POINTER_PALETTE[0], str(node.start_node), "#FFFFFF"
        elif node.terminal:
            face, text_color = "#228833", "#FFFFFF"
        elif node.truncated:
            face, text_color = "#F2F2F2", "#666666"
        else:
            face, text_color = "#FFFFFF", "#222222"
        ax.scatter(
            [x],
            [y],
            s=node_points,
            facecolor=face,
            edgecolor="0.3",
            linewidths=0.7,
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

    for depth, name in enumerate(("Start Node", "1st $V$/$v$", "2nd Step", "3rd Step")):
        if depth > tree.max_depth:
            break
        ax.text(
            min(p[0] for p in pos.values()) - 0.75,
            -float(depth),
            name,
            fontsize=6.5,
            color="0.35",
            ha="right",
            va="center",
        )

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    # The row labels are drawn inside the axes at ``min(x) - 0.75`` and
    # extend leftward, so this left margin is what has to hold them. It is
    # in data units and the widest label is "Start Node"; 2.6 held it only
    # while the figure was drawn 7 in wide.
    ax.set_xlim(min(xs) - 3.8, max(xs) + 0.5)
    ax.set_ylim(min(ys) - 0.45, max(ys) + 0.45)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def canonical_search_tree_figure(
    graph: SparseGraph,
    *,
    max_depth: int = 3,
    max_nodes: int = 400,
    max_roots: int | None = 3,
    figsize: tuple[float, float] = (PATREC_TEXT_WIDTH_INCHES, 2.6),
    title: str | None = None,
    show_graph_inset: bool = True,
    inset_positions: dict[NodeId, tuple[float, float]] | None = None,
    backend: str = "matplotlib",
) -> Figure:
    """Build the canonical-search-space schematic for *graph*.

    Args:
        graph: A small graph, 6-8 nodes for legibility.
        max_depth: Committed steps to expand before truncating.
        max_nodes: Tree-size cap.
        max_roots: Starting-node subtrees to show; see
            :func:`enumerate_search_tree`.
        figsize: Figure size in inches. The default is the *manuscript*
            text width -- ``letterpaper`` with 4.8 cm side margins leaves
            4.72 in -- so the figure is placed at ``width=\\textwidth``
            unscaled and its point sizes are the sizes that print. Rendering
            wider and letting LaTeX scale is what put 5.5 pt labels on the
            page at 3.7 pt.
        title: Figure suptitle.
        show_graph_inset: Draw the source graph as an inset panel.
        inset_positions: Pinned coordinates for the inset. When given,
            the inset is drawn in the worked-example figures' idiom so
            the same graph reads identically across the figure set.
        backend: Backend for the inset graph.

    Returns:
        The created figure. The caller owns it and must close it.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    from isalgraph.core.trace import graph_edges
    from isalgraph.viz.graph_view import draw_graph
    from isalgraph.viz.style import build_edge_palette, build_node_palette

    tree = enumerate_search_tree(
        graph, max_depth=max_depth, max_nodes=max_nodes, max_roots=max_roots
    )

    fig = plt.figure(figsize=figsize)
    # The tree spans the full width; the legend and the source graph share
    # the band beneath it, legend left and graph right. Putting the graph
    # beside the tree instead cost the leaf row a fifth of its width, and
    # the leaf row is what sets how small the labels have to be.
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[3.5, 1.0],
        height_ratios=[1.0, 0.34] if show_graph_inset else [1.0, 0.20],
        wspace=0.02,
        hspace=0.05,
        # Matplotlib reserves the top 12 % of the figure for a suptitle. With
        # no title that is a band of blank page, and page height is the
        # scarcest resource in this manuscript. Reclaim it unless a suptitle
        # was actually asked for.
        top=0.88 if title is not None else 0.985,
        bottom=0.015,
        left=0.015,
        right=0.99,
    )
    ax_tree = fig.add_subplot(gs[0, :])
    draw_search_tree(ax_tree, tree)
    # No axes title. What stood here -- "Branches: the starting node, and the
    # uninserted-neighbour choice at each V/v. Fixed (never branch):
    # displacement order |a|+|b|, and priority V > v > C > c." -- is caption
    # text, and baking it into the image renders it in the figure's font
    # rather than the document's, at whatever scale the figure is placed.
    # It belongs in the LaTeX caption; see the plan's prose.md section 10.3.

    if show_graph_inset:
        ax_inset = fig.add_subplot(gs[1, 1])
        if inset_positions is None:
            draw_graph(
                ax_inset,
                graph,
                backend=backend,
                node_colors=build_node_palette(graph.node_count()),
                edge_colors=build_edge_palette(graph_edges(graph)),
            )
        else:
            # Draw the inset in the worked-example figures' idiom and at
            # their pinned coordinates, so the running example is the same
            # picture in all three figures. The per-node identity palette
            # is the right default for an arbitrary graph and the wrong one
            # here: it makes the same graph look different in each figure.
            from isalgraph.viz.worked_example import draw_state_graph

            everything = frozenset(range(graph.node_count()))
            draw_state_graph(
                ax_inset,
                graph,
                inset_positions,
                present_nodes=everything,
                present_edges=frozenset(graph_edges(graph)),
                node_radius=0.24,
                label_fontsize=6.0,
            )
        ax_inset.set_title("Input Graph $G$", fontsize=6.5, pad=2)

    ax_legend = fig.add_subplot(gs[1, 0])
    ax_legend.axis("off")
    handles = _legend_handles(complete=any(node.terminal for node in tree.nodes))
    # Four entries in one row overflow a 4.72 in figure and are clipped at
    # both ends. Two rows of two fit with margin at every width this figure
    # is drawn at.
    ax_legend.legend(
        handles=handles,
        loc="center",
        fontsize=6.0,
        ncol=min(len(handles), 2),
        frameon=False,
        handlelength=2.0,
        columnspacing=1.6,
        labelspacing=0.5,
    )
    if title is not None:
        fig.suptitle(title, fontsize=9)
    return fig


def _legend_handles(*, complete: bool = True) -> list[Any]:
    """Return handles distinguishing branching edges from forced ones.

    Args:
        complete: Whether any drawn node finished its encoding. When the
            tree is truncated before any leaf completes, the swatch has
            nothing in the figure to point at, and a legend entry with no
            referent is worse than a missing one.

    Returns:
        Handles for ``ax.legend``.
    """
    from matplotlib.lines import Line2D

    handles = [
        # Only the leading word is capitalised. ``V`` and ``v`` are distinct
        # instructions -- primary and secondary pointer -- so title-casing
        # the whole label would silently rename half the alphabet.
        Line2D([0], [0], color=POINTER_PALETTE[0], lw=1.5, label="Branch at $V$ (π primary)"),
        Line2D([0], [0], color=POINTER_PALETTE[1], lw=1.5, label="Branch at $v$ (σ secondary)"),
        Line2D(
            [0],
            [0],
            color=GRAYED_EDGE,
            lw=1.0,
            ls=(0, (2.2, 1.6)),
            label="Forced step (no branching)",
        ),
        Line2D([0], [0], color=CANONICAL_HALO, lw=4.0, alpha=0.85, label="Canonical path $w^*_G$"),
    ]
    if complete:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#228833",
                markeredgecolor="0.3",
                markersize=6,
                label="Encoding complete",
            )
        )
    return handles


__all__ = [
    "SearchStep",
    "SearchTree",
    "SearchTreeNode",
    "canonical_search_tree_figure",
    "draw_search_tree",
    "enumerate_search_tree",
]
