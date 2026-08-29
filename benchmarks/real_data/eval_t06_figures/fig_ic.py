"""Figure 4 --- information content: what each representation costs, per size.

One panel plus an inset, and a legend that names the families rather than
listing eleven lines in an undifferentiated block.

**Main axes.** Median encoding cost per graph against ``n``, log scale.
Canonical codes -- representations whose canonical form is intrinsic to the
code rather than outsourced to an external labeling -- are drawn solid at full
weight, and the serializations dashed and half-transparent. The shaded wedge is
where the instruction string costs fewer bits than gSpan min-DFS, the only
other canonical code that reaches these sizes; the rug above marks the strata
where the paired per-stratum test rejects in its favor.

**Inset.** The same data as *coding overhead*: bits spent divided by the
information-theoretic floor, so a value of 4 reads "four times the minimum
number of bits needed to name a graph of this order and size". It flattens the
quadratic growth the main axes show, so the ordering is legible at a glance.
Log-scaled and bracketed on the measured range, 2.0x to 15x -- the floor
itself, at 1x, is off the bottom of the axis, because no series comes within
2.18x of it and keeping it in frame cost a third of the panel.

**Why the floor and not the adjacency triangle.** The inset divided by
``adjacency`` until 2026-08-26 and the denominator was very nearly one of the
numerators: graph6's length is a function of ``n`` alone, so ``nauty_graph6 /
adjacency`` sat at 1.00--1.01 at every node count above 12 -- a tautology
rendered as a curve. A per-``n`` denominator is in any case a monotone
rescaling and **cannot reorder anything at fixed** ``n``, so this choice
changes what the reader is asked, never who wins. The floor is the denominator
that makes the question well-posed: it is the only reference in the figure that
is a bound rather than another encoding.

**The floor.** ``log2 C(T, m) - log2 n!`` with ``T = n(n-1)/2`` is, by orbit
counting, a lower bound on the bits needed to name an unlabeled graph of that
order and size. Every representation sits well above it, which makes the
comparison a compression-efficiency question rather than a ranking. ``m`` is
the cohort *median* edge count at that ``n``, so the floor inherits the
cohort's own composition and is not smooth; it is a reference level, not a
model.

**Nothing here is hard-coded to a comparator set.** Series, legend groups and
the inset are all derived from whichever representations the archive holds,
intersected with the registry's ``in_figures`` flag, so a campaign that adds
or drops an arm re-renders without an edit here.

**One arm is measured and not drawn.** ``isalgraph_exhaustive`` -- the
exhaustive canonical string with a pruned fallback -- keeps its row in every
table and is absent from this figure. Above ``n ~ 28`` the D14 cascade has
substituted the pruned string for most of the cohort (96.8 % at the ``n = 40``
stratum), so its curve is the pruned curve in a second colour, and two
coincident lines read as two methods agreeing rather than as one method drawn
twice. The number it exists to report -- 114.1 bits at ``n = 20`` against the
pruned arm's 136.3 -- is a table cell, where the fallback rate can be stated
beside it. See ``design.Representation.in_figures``.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Final

import numpy as np
from matplotlib.ticker import MaxNLocator

from benchmarks.real_data.eval_t06_figures import data, design

LOGGER: Final = logging.getLogger(__name__)

#: The comparator the advantage wedge is drawn against. Resolved against what
#: the archive actually holds; the figure drops the wedge if it is absent.
WEDGE_COMPETITOR: Final[str] = "min_dfs"

#: The inset's y limits, as a multiple of the floor. Log-scaled, and bracketed
#: tightly on the measured range: over the node counts the inset covers the
#: overhead runs 2.18x (adjacency, where the cohort median edge count makes the
#: floor unusually generous) to 13.68x (nauty-graph6). Anchoring the axis at 1
#: to keep the floor in frame spent a third of the panel on empty space no
#: series ever enters, so the floor is stated in the caption instead: every
#: representation sits at least 2.2x above it everywhere.
INSET_Y_LIM: Final[tuple[float, float]] = (2.0, 15.0)

#: Ticks on that axis, chosen to read on a log scale without crowding.
INSET_Y_TICKS: Final[tuple[int, ...]] = (2, 3, 5, 10, 15)

#: This figure pools across datasets, so a stratum backed by a handful of
#: graphs moves the pooled median by more than the trend it is drawn to show.
MIN_GRAPHS: Final[int] = 20

#: Draw a marker every this many points on a dense series.
MARKEVERY: Final[int] = 7

#: Figure fraction the axes stop at. The legend boxes hang from just below it,
#: so no canvas is left empty between the x label and the first box.
AXES_BOTTOM: Final[float] = 0.30

#: Figure fractions the axes start and stop at horizontally. The legend block
#: is laid out across exactly this span, so the boxes sit under the plot rather
#: than under the canvas; ``figure`` passes the same two values to
#: ``subplots_adjust`` and the two cannot drift apart.
SUBPLOT_LEFT: Final[float] = 0.115
SUBPLOT_RIGHT: Final[float] = 0.99

#: Vertical clearance between the axes bottom and the top of the legend
#: boxes, in figure fractions. It has to exceed the x-label's own height or
#: the boxes sit on top of it.
LEGEND_DROP: Final[float] = 0.085

#: The inset starts here. Below it every ratio is dominated by the +-1 symbol
#: granularity of a four-node graph, and the asymptotic ordering the inset
#: exists to show is what the eye should land on.
INSET_MIN_N: Final[int] = 8

#: Legend groups, in draw order. Each becomes its own boxed legend, so the
#: reader can see which family a line belongs to without decoding a dash
#: pattern. A group with no present member is skipped.
LEGEND_GROUPS: Final[tuple[tuple[str, tuple[design.Family, ...]], ...]] = (
    ("Canonical Codes", (design.Family.CANONICAL_CODE,)),
    (
        "Serializations",
        (design.Family.CANONICALIZED_SERIALIZATION, design.Family.RAW_SERIALIZATION),
    ),
    ("Other", (design.Family.KERNEL, design.Family.BASELINE)),
)

#: The group the non-representation handles (significance rug, floor) join.
#: Matched by name against :data:`LEGEND_GROUPS`, so renaming a group there
#: without renaming it here silently drops those two entries.
TRAILING_GROUP: Final[str] = "Other"


def _series(cells: list[data.Cell], convention: str) -> dict[str, list[data.Aggregate]]:
    """Return the pooled per-representation series, ascending in ``n``.

    Args:
        cells: Encoding cells.
        convention: ``entropy_bits`` or ``realised_bits``.

    Returns:
        ``{representation: [Aggregate, ...]}``.
    """
    grouped: dict[str, list[data.Aggregate]] = defaultdict(list)
    for point in data.aggregate_bits(cells, convention=convention, min_graphs=MIN_GRAPHS):
        grouped[point.representation].append(point)
    return {k: sorted(v, key=lambda p: p.n) for k, v in grouped.items()}


def _advantage_wedge(ax: Any, series: dict[str, list[data.Aggregate]]) -> float:
    """Shade where the reference arm costs fewer bits than the wedge comparator.

    Carries no legend entry: the wedge is between two lines that are already
    labeled, and a third entry for the space between them is noise.

    Args:
        ax: Target axes.
        series: Pooled series per representation.

    Returns:
        Percentage of shared node counts where the reference arm is cheaper,
        or ``nan`` when the comparator is absent from this archive.
    """
    ours = {p.n: p.median for p in series.get(design.REFERENCE_KEY, [])}
    theirs = {p.n: p.median for p in series.get(WEDGE_COMPETITOR, [])}
    shared = sorted(set(ours) & set(theirs))
    if not shared:
        return float("nan")
    lower = np.array([ours[n] for n in shared])
    upper = np.array([theirs[n] for n in shared])
    ahead = lower < upper
    ax.fill_between(
        shared,
        lower,
        upper,
        where=ahead,
        color=design.BY_KEY[design.REFERENCE_KEY].colour,
        alpha=0.18,
        linewidth=0,
        zorder=1,
        interpolate=True,
    )
    return 100.0 * float(np.mean(ahead))


def _significance_rug(ax: Any, strata: dict[str, Any]) -> int:
    """Tick every node count where the paired test resolves in our favor.

    The wedge shows the *size* of the advantage from pooled medians; this shows
    where the **paired** per-stratum test, which is what the claim rests on,
    actually rejects. Two different instruments, and both belong.

    Args:
        ax: Target axes.
        strata: Parsed ``claim_a_strata.json``.

    Returns:
        How many node counts were ticked.
    """
    alpha = float(strata.get("alpha", 0.05))
    won: set[int] = set()
    for row in strata["rows"]:
        if row["representation"] != WEDGE_COMPETITOR:
            continue
        if float(row["p_entropy"]) < alpha and float(row["median_gap_entropy"]) > 0:
            won.add(int(row["n"]))
    ticks = sorted(won)
    if ticks:
        ax.plot(
            ticks,
            [0.975] * len(ticks),
            transform=ax.get_xaxis_transform(),
            marker="|",
            linestyle="none",
            markersize=3.2,
            markeredgewidth=0.8,
            alpha=0.8,
            color=design.BY_KEY[design.REFERENCE_KEY].colour,
            clip_on=True,
            zorder=8,
            label=f"Paired test rejects vs {design.BY_KEY[WEDGE_COMPETITOR].short}",
        )
    return len(ticks)


def _compression_inset(
    parent: Any,
    series: dict[str, list[data.Aggregate]],
    floor: list[tuple[int, float]],
    plt: Any,
) -> bool:
    """Draw the coding-overhead inset in the lower-right of *parent*.

    Every series is divided by the information-theoretic floor at its own node
    count, so the y axis reads as a multiple of the minimum bits needed to name
    an unlabeled graph of that order and size, and the floor is the rule at 1.

    Args:
        parent: The main axes.
        series: Pooled series per representation.
        floor: ``(n, floor bits)`` pairs.
        plt: The pyplot module.

    Returns:
        Whether the inset was drawn; it is skipped when the floor could not be
        evaluated on any node count the inset covers.
    """
    base = {n: bits for n, bits in floor if bits > 0}
    if not base:
        return False
    inset = parent.inset_axes((0.545, 0.10, 0.44, 0.40))
    drawn = False
    for rep in design.present(series):
        points = [p for p in series[rep.key] if p.n in base and p.n >= INSET_MIN_N]
        if not points:
            continue
        style = design.line_kwargs(rep)
        style["marker"] = "None"
        style["linewidth"] = (design.LW_REFERENCE if rep.is_ours else design.LW_COMPETITOR) * 0.8
        inset.plot([p.n for p in points], [p.median / base[p.n] for p in points], **style)
        drawn = True
    if not drawn:
        inset.remove()
        return False
    inset.set_yscale("log")
    inset.set_ylim(*INSET_Y_LIM)
    inset.set_yticks(list(INSET_Y_TICKS))
    inset.set_yticklabels([f"{t}$\\times$" for t in INSET_Y_TICKS])
    inset.minorticks_off()
    inset.tick_params(labelsize=design.FS_ANNOT + 0.4, length=2)
    inset.grid(True, alpha=design.GRID_ALPHA, linewidth=design.GRID_LW)
    inset.set_title(
        r"Coding Overhead  ($\times$ Floor)",
        fontsize=design.FS_ANNOT + 0.8,
        pad=2,
    )
    inset.set_xlabel("$n$", fontsize=design.FS_ANNOT + 0.8, labelpad=1)
    inset.patch.set_alpha(1.0)
    for spine in inset.spines.values():
        spine.set_linewidth(0.6)
        spine.set_visible(True)
    _ = plt
    return True


def _justify(fig: Any, legends: list[Any], y: float) -> None:
    """Spread *legends* across the axes span with equal gaps between them.

    Called after a draw, so each box is measured as it actually rendered
    rather than estimated from a character count. The first box lands flush
    on the left margin, the last flush on the right, and the free space is
    divided evenly between neighbours -- which is what puts a three-box row's
    middle group in the middle without letting it collide with a wide
    neighbour.

    Anchoring by edge (left box left-aligned, right box right-aligned, middle
    box centred) does *not* achieve this: it ignores how wide the boxes are,
    so a wide trailing box reaches back across a centred one. That is what
    this replaces.

    Args:
        fig: The figure, already drawn once.
        legends: The boxed legends, in row order.
        y: Figure-fraction height the row hangs from.
    """
    renderer = fig.canvas.get_renderer()
    canvas = float(fig.bbox.width)
    widths = [float(lg.get_window_extent(renderer).width) / canvas for lg in legends]
    span = SUBPLOT_RIGHT - SUBPLOT_LEFT
    # A negative gap means the boxes cannot fit side by side at this font
    # size. Clamping at 0 lets them touch rather than overlap, which is
    # visible in review; silently sliding them over each other is not.
    gap = max(0.0, (span - sum(widths)) / (len(legends) - 1)) if len(legends) > 1 else 0.0
    x = SUBPLOT_LEFT
    for legend, width in zip(legends, widths, strict=True):
        legend.set_bbox_to_anchor((x, y), transform=fig.transFigure)
        x += width + gap


def _grouped_legend(fig: Any, ax: Any, plt: Any, extras: list[Any]) -> None:
    """Place one boxed legend per family group beneath the axes.

    Args:
        fig: The figure.
        ax: The main axes, whose handles are harvested.
        plt: The pyplot module.
        extras: Handles that belong in the trailing group (rug, floor).
    """
    by_label = dict(zip(*reversed(ax.get_legend_handles_labels()), strict=True))
    groups: list[tuple[str, list[Any], list[str]]] = []
    for title, families in LEGEND_GROUPS:
        keys = [
            rep
            for rep in design.REPRESENTATIONS
            if rep.in_figures and rep.family in families and rep.short in by_label
        ]
        handles = [by_label[rep.short] for rep in keys]
        labels = [rep.short for rep in keys]
        if title == TRAILING_GROUP:
            handles += extras
            labels += [h.get_label() for h in extras]
        if handles:
            groups.append((title, handles, labels))
    if not groups:
        return
    # Place the boxes provisionally, draw once so each can be measured as it
    # actually rendered, then justify. Laying them out from a character count
    # is what previously left the row short of the right margin, and edge
    # anchoring is what let a wide trailing box reach back over its neighbour.
    y = AXES_BOTTOM - LEGEND_DROP
    legends = []
    for title, handles, labels in groups:
        legend = fig.legend(
            handles,
            labels,
            title=title,
            fontsize=design.FS_LEGEND,
            title_fontsize=design.FS_LEGEND,
            loc="upper left",
            bbox_to_anchor=(SUBPLOT_LEFT, y),
            frameon=True,
            framealpha=1.0,
            edgecolor="0.75",
            borderpad=0.45,
            labelspacing=0.35,
            handlelength=1.8,
        )
        legend.get_frame().set_linewidth(0.5)
        legend.get_title().set_fontweight("bold")
        fig.add_artist(legend)
        legends.append(legend)
    fig.canvas.draw()
    _justify(fig, legends, y)
    _ = plt


def figure(
    cells: list[data.Cell],
    strata: dict[str, Any],
    out: Path,
    *,
    convention: str = "entropy_bits",
) -> tuple[list[Path], dict[str, float]]:
    """Build the information-content figure.

    Args:
        cells: Encoding cells.
        strata: Parsed ``claim_a_strata.json``.
        out: Output path without an extension.
        convention: ``entropy_bits`` or ``realised_bits``.

    Returns:
        The paths written, and the quantities the caption quotes.
    """
    plt = design.style()
    series = _series(cells, convention)
    fig, ax = plt.subplots(figsize=(design.text_width() * 0.80, 4.6))

    share = _advantage_wedge(ax, series)
    for rep in design.present(series):
        points = series[rep.key]
        style = design.line_kwargs(rep)
        style["markevery"] = max(1, len(points) // MARKEVERY)
        ax.plot([p.n for p in points], [p.median for p in points], label=rep.short, **style)
    ticks = _significance_rug(ax, strata)

    floor = data.unlabeled_floor(cells, min_graphs=MIN_GRAPHS)
    if floor:
        ax.plot(
            [n for n, _ in floor],
            [b for _, b in floor],
            color=design.INK_FLOOR,
            linewidth=1.0,
            linestyle=(0, (1, 1.3)),
            zorder=2,
            label=r"Floor $\log_2\binom{T}{m}-\log_2 n!$",
        )

    reached = max(p.n for points in series.values() for p in points)
    ax.set_xlim(0, reached + 1)
    ax.set_yscale("log")
    unit = "Entropy Bound" if convention == "entropy_bits" else "Realised Bytes"
    design.finish_axes(
        ax,
        xlabel="Graph Size ($n$)",
        ylabel=f"Median Bits per Graph ({unit})",
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

    inset_drawn = _compression_inset(ax, series, floor, plt)

    labeled = dict(zip(*reversed(ax.get_legend_handles_labels()), strict=True))
    extras = [
        labeled[name]
        for name in list(labeled)
        if name.startswith("Paired test") or name.startswith("Floor")
    ]
    _grouped_legend(fig, ax, plt, extras)
    fig.subplots_adjust(left=SUBPLOT_LEFT, right=SUBPLOT_RIGHT, top=0.985, bottom=AXES_BOTTOM)
    saved = design.save(fig, out)
    plt.close(fig)
    return saved, {
        "wedge_share": share,
        "significant_strata": float(ticks),
        "inset": float(inset_drawn),
    }


def caption(summary: dict[str, float], *, convention: str) -> str:
    """Return the LaTeX caption, with every convention and guard stated.

    Args:
        summary: Quantities returned by :func:`figure`.
        convention: Convention the figure was drawn in.

    Returns:
        Caption text, ready to paste into a ``figure`` environment.
    """
    unit = r"entropy bound $L\log_2|\Sigma|$" if convention == "entropy_bits" else "realised bytes"
    wedge = (
        "The shaded wedge is where the instruction string costs fewer bits than gSpan min-DFS, "
        "the only other canonical code reaching these sizes "
        f"({summary['wedge_share']:.0f}\\,\\% of shared node counts); the ticks above mark the "
        f"{summary['significant_strata']:.0f} node counts at which the paired per-stratum test "
        "rejects in its favor. "
        if summary["wedge_share"] == summary["wedge_share"]
        else ""
    )
    inset = (
        "\\emph{Inset:} the same data as coding overhead --- bits spent divided by the floor at "
        "that $n$, so a value of 4 reads as four times the minimum. The axis is log-scaled and "
        "starts at $2\\times$: no representation comes within $2.18\\times$ of the floor at any "
        "size, so the floor itself is below the frame. A per-$n$ denominator is a monotone "
        "rescaling and cannot reorder the representations at fixed $n$; it is drawn because the "
        "ordering is otherwise unreadable under quadratic growth. "
        if summary["inset"]
        else ""
    )
    return (
        "Information content by graph size. Median encoding cost per graph under the "
        f"{unit}, pooled over datasets, log scale, restricted to node counts backed by at least "
        f"{MIN_GRAPHS} graphs. Canonical codes -- those whose canonical form is intrinsic to the "
        "code rather than produced by an external labeling -- are drawn solid at full weight; "
        "the serializations are dashed and half-transparent, and appear at equal weight in "
        f"Table~\\ref{{tab:representation-summary}}. {wedge}{inset}"
        "Dotted: $\\log_2\\binom{T}{m}-\\log_2 n!$ with $T=n(n-1)/2$ and $m$ the cohort median "
        "edge count, which by orbit counting lower-bounds the bits needed to name an unlabeled "
        "graph of that order and size; $m$ is the cohort's own median, so the floor is a "
        "reference level rather than a smooth model. The raw graph6 and sparse6 serializations "
        "are withdrawn in favor of their nauty-canonicalized forms: graph6's length is a "
        "function of $n$ alone, so canonicalizing permutes the bits without changing how many "
        "there are and its curve lay exactly under nauty-graph6's at every node count. AGM CAM "
        "is a canonical code but its scope guard stops at $n=12$, above which its branch and "
        "bound closes on a minority of real graphs and its column would be conditioned on the "
        "graphs symmetric enough to finish. The exhaustive-with-fallback arm is measured and "
        "tabulated (Table~\\ref{tab:representation-summary}) but not drawn here: above "
        "$n\\approx28$ its budget expires on most of the cohort and the substituted string is "
        "the pruned one, so its curve would duplicate the pruned curve in a second colour. "
        "Descriptive; the confirmatory cells are per (dataset, representation) and live in the "
        "pre-registered family."
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encodings", type=Path, required=True, help="encodings/ directory")
    ap.add_argument("--strata", type=Path, required=True, help="claim_a_strata.json")
    ap.add_argument("--out-dir", type=Path, required=True, help="figure output directory")
    ap.add_argument(
        "--convention", choices=data.CONVENTIONS, default="entropy_bits", help="bit convention"
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector, or ``None`` for ``sys.argv``.

    Returns:
        Process exit status.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cells = data.load_cells(args.encodings)
    strata = data.load_json(args.strata)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / "fig4_information_content"
    saved, summary = figure(cells, strata, stem, convention=args.convention)
    (args.out_dir / "fig4_information_content.caption.tex").write_text(
        caption(summary, convention=args.convention) + "\n"
    )
    LOGGER.info("fig4_information_content -> %s", ", ".join(str(p) for p in saved))
    LOGGER.info(
        "wedge %.0f%%, %d significant strata, inset=%s",
        summary["wedge_share"],
        int(summary["significant_strata"]),
        bool(summary["inset"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
