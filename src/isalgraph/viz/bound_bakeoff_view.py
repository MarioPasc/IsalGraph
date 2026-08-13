"""Views for the T-27 GED bound bake-off.

Two panels per bracket end, plus the critical-difference diagram that
:doc:`T-27-design` §3.8 requires beside the Wilcoxon/Holm table:

* **(a) mean relative error against ``max(n1, n2)``** -- one line per
  method, ribbon spanning the interquartile range, one facet per
  dataset. This is the panel that answers whether a method choice made
  on one dataset transfers across graph size, which the 400-pair LINUX
  measurement in the submitted manuscript could not answer.
* **(b) forest plot** -- mean relative error with its graph-level
  bootstrap confidence interval, methods on the y-axis, grouped by
  dataset.
* **(c) critical-difference diagram** -- average Friedman rank per
  method with the Nemenyi critical difference drawn as a bar and
  statistically indistinguishable methods joined by a clique line.

Data contract
-------------
Every function here takes plain Python scalars and tuples. Nothing in
this module imports ``numpy``, ``scipy`` or ``matplotlib`` at module
scope: ``tests/viz/test_import_without_matplotlib.py`` blocks all six of
those packages and imports every file under ``isalgraph/viz`` anyway, so
a module-scope import of any of them fails the suite. The aggregation
that produces these dataclasses lives in
``benchmarks/real_data/eval_setup/ged_bakeoff_analysis.py``.

Drawing functions paint on a caller-supplied ``Axes`` and never create a
figure; only :func:`bound_bakeoff_figure` and
:func:`critical_difference_figure` do, and the caller owns what they
return.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from isalgraph.viz.style import (
    GRAYED_EDGE,
    PAUL_TOL_MUTED,
    apply_ieee_style,
    get_figure_size,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
else:  # pragma: no cover - typing-only aliases
    Axes = Any
    Figure = Any


#: Marker cycle used alongside hue, so the lines stay separable when the
#: figure is printed in greyscale. IEEE still prints some issues that way.
METHOD_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")

#: Caveat printed under every critical-difference diagram. Both halves are
#: mandated by the design note: Friedman at ``N = 5`` separates almost
#: nothing (statistics.md §4), and the five datasets are not independent
#: because IAM Letter LOW/MED/HIGH are one corpus at three distortion
#: levels (design §3.2), so the effective omnibus sample is closer to
#: three.
CD_CAVEAT: str = (
    "Friedman over N = 5 datasets; the critical difference is wide and "
    "separates little. The five datasets are not independent: IAM Letter "
    "LOW/MED/HIGH are one corpus at three distortion levels, so the vote "
    "is really 3 + 1 + 1. Descriptive aid to a selection procedure, not a "
    "hypothesis test."
)


@dataclass(frozen=True)
class ErrorCurve:
    """Mean relative error against graph size for one method.

    Attributes:
        method: GEDLIB method name, upper case.
        n_values: The ``max(n1, n2)`` bins, ascending.
        mean: Mean relative error in each bin.
        q25: First quartile of relative error in each bin.
        q75: Third quartile of relative error in each bin.
        counts: Number of pairs behind each bin.
    """

    method: str
    n_values: tuple[int, ...]
    mean: tuple[float, ...]
    q25: tuple[float, ...]
    q75: tuple[float, ...]
    counts: tuple[int, ...]

    def __post_init__(self) -> None:
        """Reject ragged curves at construction rather than at draw time."""
        k = len(self.n_values)
        if not (len(self.mean) == len(self.q25) == len(self.q75) == len(self.counts) == k):
            raise ValueError(f"ErrorCurve {self.method!r}: all sequences must share a length")


@dataclass(frozen=True)
class DatasetCurves:
    """All method curves for one dataset, for one facet of panel (a)."""

    dataset: str
    curves: tuple[ErrorCurve, ...] = ()


@dataclass(frozen=True)
class ForestEntry:
    """One row of the forest plot.

    Attributes:
        dataset: Dataset key.
        method: GEDLIB method name.
        mean: Point estimate of mean relative error.
        ci_low: Lower percentile bootstrap bound.
        ci_high: Upper percentile bootstrap bound.
        winner: Whether this method is the dataset's frozen-rule primary.
    """

    dataset: str
    method: str
    mean: float
    ci_low: float
    ci_high: float
    winner: bool = False


@dataclass(frozen=True)
class BakeoffPanels:
    """Everything one bake-off figure needs.

    Attributes:
        end: ``"lower"`` or ``"upper"``.
        dataset_curves: One entry per dataset facet of panel (a).
        forest: Rows of panel (b), in the order they should be drawn.
        methods: Method order, which fixes the colour assignment.
    """

    end: str
    dataset_curves: tuple[DatasetCurves, ...] = ()
    forest: tuple[ForestEntry, ...] = ()
    methods: tuple[str, ...] = ()


@dataclass(frozen=True)
class CriticalDifference:
    """Average-rank summary for a critical-difference diagram.

    Attributes:
        end: ``"lower"`` or ``"upper"``.
        methods: Method names.
        average_ranks: Average Friedman rank per method, same order.
        cd: The Nemenyi critical difference.
        n_datasets: Number of datasets entering the omnibus.
        friedman_p: Omnibus p-value, or ``None`` if not computed.
        cliques: Index groups whose ranks differ by less than *cd*.
    """

    end: str
    methods: tuple[str, ...]
    average_ranks: tuple[float, ...]
    cd: float
    n_datasets: int
    friedman_p: float | None = None
    cliques: tuple[tuple[int, ...], ...] = field(default_factory=tuple)


def method_palette(methods: Sequence[str]) -> dict[str, str]:
    """Return a stable ``{method: hex}`` mapping.

    Colours are assigned by position in *methods*, so the caller fixes
    the assignment once and every panel of a figure agrees. The muted
    Paul Tol set is colour-blind safe and is the palette already in the
    submitted PDF.

    Args:
        methods: Method names, in the order that fixes their colours.

    Returns:
        Mapping from method name to hex colour.
    """
    return {m: PAUL_TOL_MUTED[i % len(PAUL_TOL_MUTED)] for i, m in enumerate(methods)}


def method_marker(methods: Sequence[str], method: str) -> str:
    """Return the greyscale-fallback marker for *method*.

    Args:
        methods: The method order fixing marker assignment.
        method: The method to look up.

    Returns:
        A matplotlib marker code; ``"o"`` if *method* is not in *methods*.
    """
    order = list(methods)
    if method not in order:
        return "o"
    return METHOD_MARKERS[order.index(method) % len(METHOD_MARKERS)]


def draw_error_vs_n(
    ax: Axes,
    curves: DatasetCurves,
    *,
    methods: Sequence[str],
    palette: dict[str, str] | None = None,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
    ribbon_alpha: float = 0.18,
    min_count: int = 1,
) -> None:
    """Draw one dataset facet of panel (a) on *ax*.

    The ribbon is the interquartile range, not a confidence interval.
    That is deliberate: the spread of per-pair relative error is a
    property of the bound worth showing, and a bootstrap interval on the
    mean would be invisible at these sample sizes.

    Bins holding fewer than *min_count* pairs are dropped rather than
    plotted, so a single pair at ``n = 12`` cannot draw a spike that
    reads like a trend.

    Args:
        ax: Target matplotlib axes.
        curves: The dataset's method curves.
        methods: Method order fixing colours and markers.
        palette: Optional explicit colour mapping.
        show_ylabel: Whether to label the y-axis.
        show_xlabel: Whether to label the x-axis.
        ribbon_alpha: Opacity of the interquartile ribbon.
        min_count: Minimum pairs for a bin to be drawn.
    """
    colors = palette if palette is not None else method_palette(methods)
    for curve in curves.curves:
        keep = [k for k, c in enumerate(curve.counts) if c >= min_count]
        if not keep:
            continue
        xs = [curve.n_values[k] for k in keep]
        ys = [curve.mean[k] for k in keep]
        lo = [curve.q25[k] for k in keep]
        hi = [curve.q75[k] for k in keep]
        color = colors.get(curve.method, GRAYED_EDGE)
        ax.fill_between(xs, lo, hi, color=color, alpha=ribbon_alpha, linewidth=0.0, zorder=1)
        ax.plot(
            xs,
            ys,
            color=color,
            marker=method_marker(methods, curve.method),
            markersize=2.6,
            linewidth=1.1,
            label=curve.method,
            zorder=2,
        )
    ax.set_title(curves.dataset, pad=2.0)
    if show_xlabel:
        ax.set_xlabel(r"$\max(n_1, n_2)$")
    if show_ylabel:
        ax.set_ylabel("mean relative error")
    ax.grid(visible=True, axis="y")
    ax.margins(x=0.04)


def draw_forest(
    ax: Axes,
    entries: Sequence[ForestEntry],
    *,
    methods: Sequence[str],
    palette: dict[str, str] | None = None,
    show_xlabel: bool = True,
    row_gap: float = 0.6,
) -> None:
    """Draw panel (b), the per-dataset ranking, on *ax*.

    Rows run top to bottom in the order given, grouped by dataset with a
    gap between groups. The winner of a group -- the frozen-rule primary
    for that dataset -- is drawn with a heavier marker.

    Args:
        ax: Target matplotlib axes.
        entries: Forest rows, already ordered.
        methods: Method order fixing colours.
        palette: Optional explicit colour mapping.
        show_xlabel: Whether to label the x-axis.
        row_gap: Extra vertical space inserted between dataset groups.
    """
    colors = palette if palette is not None else method_palette(methods)
    positions: list[float] = []
    y = 0.0
    previous: str | None = None
    for entry in entries:
        if previous is not None and entry.dataset != previous:
            y += row_gap
        positions.append(y)
        previous = entry.dataset
        y += 1.0

    for pos, entry in zip(positions, entries, strict=True):
        color = colors.get(entry.method, GRAYED_EDGE)
        ax.plot(
            [entry.ci_low, entry.ci_high],
            [pos, pos],
            color=color,
            linewidth=1.4,
            solid_capstyle="butt",
            zorder=2,
        )
        ax.plot(
            [entry.mean],
            [pos],
            marker="D" if entry.winner else "o",
            markersize=4.2 if entry.winner else 3.0,
            color=color,
            markeredgecolor="black" if entry.winner else color,
            markeredgewidth=0.6 if entry.winner else 0.0,
            linestyle="none",
            zorder=3,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels([e.method for e in entries])
    ax.invert_yaxis()
    if show_xlabel:
        ax.set_xlabel("mean relative error (graph-level bootstrap 95 % CI)")
    ax.grid(visible=True, axis="x")

    # Dataset name once per group, on the right-hand edge of the axes.
    seen: set[str] = set()
    for pos, entry in zip(positions, entries, strict=True):
        if entry.dataset in seen:
            continue
        seen.add(entry.dataset)
        ax.annotate(
            entry.dataset,
            xy=(1.005, pos),
            xycoords=("axes fraction", "data"),
            fontsize=6.0,
            va="center",
            ha="left",
            color=GRAYED_EDGE,
            annotation_clip=False,
        )


def draw_critical_difference(
    ax: Axes,
    cd: CriticalDifference,
    *,
    methods: Sequence[str] | None = None,
    palette: dict[str, str] | None = None,
) -> None:
    """Draw a Demsar critical-difference diagram on *ax*.

    Rank 1 sits on the left. Methods whose average ranks differ by less
    than :attr:`CriticalDifference.cd` are joined by a clique bar, which
    is the diagram's only claim: joined methods are not separated.

    Args:
        ax: Target matplotlib axes.
        cd: The average-rank summary.
        methods: Method order fixing colours; defaults to ``cd.methods``.
        palette: Optional explicit colour mapping.
    """
    order = list(methods) if methods is not None else list(cd.methods)
    colors = palette if palette is not None else method_palette(order)
    ranks = list(cd.average_ranks)
    if not ranks:
        ax.set_axis_off()
        return

    lo = min(1.0, min(ranks) - 0.5)
    hi = max(float(len(ranks)), max(ranks) + 0.5)
    ax.set_xlim(hi, lo)  # rank 1 on the left
    ax.set_ylim(-0.4 - 0.32 * len(ranks), 1.0)
    ax.set_yticks([])
    for side in ("left", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["top"].set_visible(True)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("average Friedman rank")

    ordered = sorted(range(len(ranks)), key=lambda k: ranks[k])
    for depth, k in enumerate(ordered):
        r = ranks[k]
        y = -0.28 * (depth + 1)
        color = colors.get(cd.methods[k], GRAYED_EDGE)
        ax.plot([r, r], [0.0, y], color=color, linewidth=0.9)
        ax.plot([r, lo if depth < len(ranks) / 2 else hi], [y, y], color=color, linewidth=0.9)
        ax.annotate(
            f"{cd.methods[k]} ({r:.2f})",
            xy=(lo if depth < len(ranks) / 2 else hi, y),
            fontsize=6.0,
            va="center",
            ha="right" if depth < len(ranks) / 2 else "left",
            color=color,
        )

    for depth, clique in enumerate(cd.cliques):
        if len(clique) < 2:
            continue
        members = [ranks[k] for k in clique]
        y = 0.12 + 0.10 * depth
        ax.plot([min(members), max(members)], [y, y], color="black", linewidth=2.2, zorder=4)

    ax.annotate(
        f"CD = {cd.cd:.2f} (Nemenyi, N = {cd.n_datasets})",
        xy=(0.5, 0.98),
        xycoords="axes fraction",
        fontsize=6.0,
        ha="center",
        va="top",
    )


def bound_bakeoff_figure(
    panels: BakeoffPanels,
    *,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    min_count: int = 1,
) -> Figure:
    """Build the two-panel bake-off figure for one bracket end.

    Args:
        panels: The assembled panel data.
        figsize: Figure size in inches; defaults to IEEE text width.
        title: Optional suptitle.
        min_count: Minimum pairs for a bin of panel (a) to be drawn.

    Returns:
        The created figure. The caller owns it and must close it.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    apply_ieee_style()
    n_facets = max(1, len(panels.dataset_curves))
    size = figsize if figsize is not None else (get_figure_size("double")[0], 5.4)
    fig = plt.figure(figsize=size)
    gs = GridSpec(
        2,
        n_facets,
        figure=fig,
        height_ratios=[1.0, 1.45],
        hspace=0.52,
        wspace=0.28,
    )

    colors = method_palette(panels.methods)
    axes_a = []
    for k, curves in enumerate(panels.dataset_curves):
        ax = fig.add_subplot(gs[0, k])
        draw_error_vs_n(
            ax,
            curves,
            methods=panels.methods,
            palette=colors,
            show_ylabel=(k == 0),
            min_count=min_count,
        )
        axes_a.append(ax)

    ax_b = fig.add_subplot(gs[1, :])
    draw_forest(ax_b, panels.forest, methods=panels.methods, palette=colors)

    if axes_a:
        handles, labels = axes_a[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.045),
                ncol=min(len(labels), 4),
            )
    if title is not None:
        fig.suptitle(title)
    return fig


def critical_difference_figure(
    cd: CriticalDifference,
    *,
    figsize: tuple[float, float] | None = None,
    caption: str = CD_CAVEAT,
) -> Figure:
    """Build the critical-difference diagram for one bracket end.

    Args:
        cd: The average-rank summary.
        figsize: Figure size in inches; defaults to IEEE column width.
        caption: Caveat text drawn beneath the diagram. The default
            carries both caveats the design note requires.

    Returns:
        The created figure. The caller owns it and must close it.
    """
    import matplotlib.pyplot as plt

    apply_ieee_style()
    size = figsize if figsize is not None else (get_figure_size("double")[0], 2.6)
    fig = plt.figure(figsize=size)
    ax = fig.add_axes((0.08, 0.34, 0.84, 0.5))
    draw_critical_difference(ax, cd)
    fig.text(0.5, 0.03, caption, fontsize=5.6, ha="center", va="bottom", wrap=True)
    return fig


__all__ = [
    "CD_CAVEAT",
    "METHOD_MARKERS",
    "BakeoffPanels",
    "CriticalDifference",
    "DatasetCurves",
    "ErrorCurve",
    "ForestEntry",
    "bound_bakeoff_figure",
    "critical_difference_figure",
    "draw_critical_difference",
    "draw_error_vs_n",
    "draw_forest",
    "method_marker",
    "method_palette",
]
