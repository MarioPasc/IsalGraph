"""Figure 4 --- information content: what each representation costs, per size.

One panel plus an inset, and a legend that names the families rather than
listing eleven lines in an undifferentiated block.

**Main axes.** Median encoding cost per graph against ``n``, log scale.
Canonical codes -- representations whose canonical form is intrinsic to the
code rather than outsourced to an external labelling -- are drawn solid at full
weight, and the serialisations dashed and half-transparent. The shaded wedge is
where the instruction string costs fewer bits than gSpan min-DFS, the only
other canonical code that reaches these sizes; the rug above marks the strata
where the paired per-stratum test rejects in its favour.

**Inset.** The same data as a compression ratio against the adjacency matrix,
which is the naive fixed ``n(n-1)/2`` encoding. This is the modern form of the
message-length ratio the original manuscript reported: ``corrections.md`` B3
requires real serialisations beside the explicit-construction reference model,
and the adjacency triangle is the one every representation can be divided by.
It flattens the quadratic growth the main axes show, so the ordering is legible
at a glance and the floor becomes a line rather than a distant curve.

**The floor.** ``log2 C(T, m) - log2 n!`` with ``T = n(n-1)/2`` is, by orbit
counting, a lower bound on the bits needed to name an unlabelled graph of that
order and size. Every representation sits well above it, which makes the
comparison a compression-efficiency question rather than a ranking.

**Nothing here is hard-coded to a comparator set.** Series, legend groups and
the inset are all derived from whichever representations the archive holds, so
a campaign that adds or drops an arm re-renders without an edit.
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

#: The denominator of the inset's compression ratio: the naive fixed-width
#: encoding of the upper triangle, which every representation can be divided
#: by and which needs no explanation to a reader.
RATIO_BASELINE: Final[str] = "adjacency"

#: This figure pools across datasets, so a stratum backed by a handful of
#: graphs moves the pooled median by more than the trend it is drawn to show.
MIN_GRAPHS: Final[int] = 20

#: Draw a marker every this many points on a dense series.
MARKEVERY: Final[int] = 7

#: Figure fraction the axes stop at. The legend boxes hang from just below it,
#: so no canvas is left empty between the x label and the first box.
AXES_BOTTOM: Final[float] = 0.30

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
    ("canonical codes", (design.Family.CANONICAL_CODE,)),
    (
        "serialisations",
        (design.Family.CANONICALISED_SERIALISATION, design.Family.RAW_SERIALISATION),
    ),
    ("other", (design.Family.KERNEL, design.Family.BASELINE)),
)


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
    labelled, and a third entry for the space between them is noise.

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
    """Tick every node count where the paired test resolves in our favour.

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
            label=f"paired test rejects vs {design.BY_KEY[WEDGE_COMPETITOR].short}",
        )
    return len(ticks)


def _compression_inset(
    parent: Any,
    series: dict[str, list[data.Aggregate]],
    floor: list[tuple[int, float]],
    plt: Any,
) -> bool:
    """Draw the compression-ratio inset in the lower-right of *parent*.

    Args:
        parent: The main axes.
        series: Pooled series per representation.
        floor: ``(n, floor bits)`` pairs.
        plt: The pyplot module.

    Returns:
        Whether the inset was drawn; it is skipped when the baseline
        representation is absent from the archive.
    """
    base = {p.n: p.median for p in series.get(RATIO_BASELINE, [])}
    if not base:
        return False
    inset = parent.inset_axes((0.545, 0.10, 0.44, 0.40))
    for rep in design.present(series):
        if rep.key == RATIO_BASELINE:
            continue
        points = [
            p
            for p in series[rep.key]
            if p.n in base and base[p.n] > 0 and p.n >= INSET_MIN_N
        ]
        if not points:
            continue
        style = design.line_kwargs(rep)
        style["marker"] = "None"
        style["linewidth"] = (design.LW_REFERENCE if rep.is_ours else design.LW_COMPETITOR) * 0.8
        inset.plot([p.n for p in points], [p.median / base[p.n] for p in points], **style)
    usable = [
        (n, b / base[n])
        for n, b in floor
        if n in base and base[n] > 0 and n >= INSET_MIN_N
    ]
    if usable:
        inset.plot(
            [n for n, _ in usable],
            [r for _, r in usable],
            color=design.INK_FLOOR,
            linewidth=0.9,
            linestyle=(0, (1, 1.3)),
            zorder=2,
        )
    inset.axhline(1.0, color=design.INK_RULE, linewidth=0.6, linestyle=":")
    inset.set_ylim(0, 1.18)
    inset.set_yticks([0, 0.5, 1.0])
    inset.tick_params(labelsize=design.FS_ANNOT + 0.4, length=2)
    inset.grid(True, alpha=design.GRID_ALPHA, linewidth=design.GRID_LW)
    inset.set_title(
        "compression ratio vs adjacency",
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
            if rep.family in families and rep.short in by_label
        ]
        handles = [by_label[rep.short] for rep in keys]
        labels = [rep.short for rep in keys]
        if title == "other":
            handles += extras
            labels += [h.get_label() for h in extras]
        if handles:
            groups.append((title, handles, labels))
    if not groups:
        return
    widths = [max(len(t), *(len(x) for x in labels)) for t, _, labels in groups]
    total = sum(widths)
    left = 0.0
    for (title, handles, labels), width in zip(groups, widths, strict=True):
        legend = fig.legend(
            handles,
            labels,
            title=title,
            fontsize=design.FS_LEGEND,
            title_fontsize=design.FS_LEGEND,
            loc="upper left",
            bbox_to_anchor=(left / total, AXES_BOTTOM - LEGEND_DROP),
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
        left += width
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

    floor = data.unlabelled_floor(cells, min_graphs=MIN_GRAPHS)
    if floor:
        ax.plot(
            [n for n, _ in floor],
            [b for _, b in floor],
            color=design.INK_FLOOR,
            linewidth=1.0,
            linestyle=(0, (1, 1.3)),
            zorder=2,
            label=r"floor $\log_2\binom{T}{m}-\log_2 n!$",
        )

    reached = max(p.n for points in series.values() for p in points)
    ax.set_xlim(0, reached + 1)
    ax.set_yscale("log")
    unit = "entropy bound" if convention == "entropy_bits" else "realised bytes"
    design.finish_axes(ax, xlabel="graph size $n$", ylabel=f"median bits per graph, {unit}")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

    inset_drawn = _compression_inset(ax, series, floor, plt)

    labelled = dict(zip(*reversed(ax.get_legend_handles_labels()), strict=True))
    extras = [
        labelled[name]
        for name in list(labelled)
        if name.startswith("paired test") or name.startswith("floor")
    ]
    _grouped_legend(fig, ax, plt, extras)
    fig.subplots_adjust(left=0.115, right=0.99, top=0.985, bottom=AXES_BOTTOM)
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
        "rejects in its favour. "
        if summary["wedge_share"] == summary["wedge_share"]
        else ""
    )
    inset = (
        "\\emph{Inset:} the same data as a compression ratio against the adjacency matrix, the "
        "naive fixed $n(n-1)/2$ encoding, with the floor on the same scale; below 1 is cheaper "
        "than writing the upper triangle out. "
        if summary["inset"]
        else ""
    )
    return (
        "Information content by graph size. Median encoding cost per graph under the "
        f"{unit}, pooled over datasets, log scale, restricted to node counts backed by at least "
        f"{MIN_GRAPHS} graphs. Canonical codes -- those whose canonical form is intrinsic to the "
        "code rather than produced by an external labelling -- are drawn solid at full weight; "
        "the serialisations are dashed and half-transparent, and appear at equal weight in "
        f"Table~\\ref{{tab:representation-summary}}. {wedge}{inset}"
        "Dotted: $\\log_2\\binom{T}{m}-\\log_2 n!$ with $T=n(n-1)/2$ and $m$ the cohort median "
        "edge count, which by orbit counting lower-bounds the bits needed to name an unlabelled "
        "graph of that order and size. graph6 and nauty-graph6 coincide by construction -- "
        "graph6's length is a function of $n$ alone, so canonicalising permutes the bits without "
        "changing how many there are. AGM CAM is a canonical code but its scope guard stops at "
        "$n=12$. Descriptive; the confirmatory cells are per (dataset, representation) and live "
        "in the pre-registered family."
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
