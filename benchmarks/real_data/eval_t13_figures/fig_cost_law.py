"""The T-13 main-text figure: what governs the canonical search's cost.

One panel per ladder.  A ladder holds ``n``, ``m`` **and the whole degree
sequence** exactly constant while ``|Aut(G)|`` falls by orders of magnitude, so
a curve that moves across a panel moves because of symmetry and nothing else.
That is the entire argument, and the panel annotation states the held-constant
invariants because without them a rising curve proves nothing.

The reader should come away with four readings, in this order:

* the **search-free** arms are flat -- they are the null (``T-13-design.md``
  2.3), and their flatness is what licenses reading any slope elsewhere as
  symmetry rather than as a confound that tracks it;
* **isalgraph_exhaustive** is flat too, because an unpruned search expands
  every branch and its branching factors come from the degree sequence, which
  the ladder pins;
* **isalgraph_pruned** rises steeply, which is Corollary 2 measured: the
  triplet key removes every branch it can distinguish, and what it cannot
  distinguish is exactly the orbits;
* **min_dfs** rises too, and censors.

**Censored points are never drawn as ordinary points.**  ``status="censored"``
means the completion time is *greater than* the plotted value, so the point is
an open marker carrying an upward arrow, and it is excluded from the per-rung
median the line traces.  A cap-censored ``min_dfs`` row can sit at four
milliseconds; drawn as a filled point it would read as the fastest measurement
in the panel when it means the encoder never finished.

**The line traces per-rung medians in rung order**, where the rung index comes
from ``params`` and never from ``log10_aut``.  Points are placed at their own
``log10_aut`` because that is the abscissa, which is not the same thing: the
ordering that defines the series is the design's, and the position is the
measurement's.
"""

from __future__ import annotations

import argparse
import logging
import math
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from benchmarks.real_data.eval_t13_figures import data, design

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from matplotlib.axes import Axes

LOGGER: Final = logging.getLogger(__name__)

#: Panels per row.  Three fits the IEEE text width without the tick labels
#: colliding; a fourth does not.
NCOLS: Final[int] = 3

#: Height of one panel, inches.
PANEL_HEIGHT_IN: Final[float] = 2.35

#: Arms whose measured variation across a ladder is quoted in the panel note.
#: These are the two the 6.3 characterisation contrasts, and quoting them from
#: the data is what turns the sentence into a measurement.
QUOTED_KEYS: Final[tuple[str, ...]] = ("isalgraph_exhaustive", "isalgraph_pruned")

#: Factor the top of each panel's log y axis is lifted by, to clear a strip
#: above the censored observations for the annotation box.
NOTE_HEADROOM: Final[float] = 200.0

#: Output stem, without an extension.
STEM: Final[str] = "fig_t13_cost_law"


class CostLawFigureError(ValueError):
    """The records cannot support the cost-law figure."""


def _rung_medians(ladder: data.Ladder, representation: str) -> tuple[list[float], list[float], int]:
    """Return per-rung ``(x, y)`` medians over completed rows, in rung order.

    Args:
        ladder: The ladder.
        representation: Backend key.

    Returns:
        ``(xs, ys, n_rungs_with_completion)`` where ``xs`` are median
        ``log10_aut`` and ``ys`` median ``seconds``, both taken over the
        **completed** rows of each rung and ordered by the rung index.
    """
    by_rung: dict[int, tuple[list[float], list[float]]] = {}
    for graph, row in ladder.series(representation):
        if not data.is_completed(row) or graph.log10_aut is None:
            continue
        xs, ys = by_rung.setdefault(graph.rung, ([], []))
        xs.append(graph.log10_aut)
        ys.append(float(row["seconds"]))
    ordered = sorted(by_rung)
    return (
        [statistics.median(by_rung[r][0]) for r in ordered],
        [statistics.median(by_rung[r][1]) for r in ordered],
        len(ordered),
    )


def _censored_points(ladder: data.Ladder, representation: str) -> tuple[list[float], list[float]]:
    """Return every censored observation of one representation on one ladder."""
    xs: list[float] = []
    ys: list[float] = []
    for graph, row in ladder.series(representation):
        if data.is_censored(row) and graph.log10_aut is not None:
            xs.append(graph.log10_aut)
            ys.append(float(row["seconds"]))
    return xs, ys


def _variation(ys: Sequence[float]) -> float | None:
    """Return ``max / min`` over *ys*, or ``None`` when it is not defined."""
    if len(ys) < 2:
        return None
    lo = min(ys)
    return max(ys) / lo if lo > 0.0 else None


def _format_variation(value: float | None) -> str:
    """Render a variation ratio for a panel note."""
    if value is None:
        return "n/a"
    if value >= 100.0:
        return f"{value:.0f}$\\times$"
    return f"{value:.2f}$\\times$"


def _panel_note(ladder: data.Ladder, variation: dict[str, float | None]) -> str:
    """Return the held-constant annotation for one panel.

    Every number is read off the records; the degree-sequence clause states a
    construction invariant that ``families.py`` enforces at build time.

    Kept to three short lines.  A five-line box at 5.6 pt covers a quarter of a
    third-width panel, and in a dense panel there is no free corner large
    enough for it -- the first draft hid the ``complete_bipartite`` series
    entirely behind its own annotation.
    """
    span = ladder.aut_span
    header = f"$n={ladder.n}$, $m={ladder.m}$, $\\Delta={ladder.max_degree}$; deg. seq. fixed"
    lines = [header]
    if span is not None:
        lines.append(f"$\\log_{{10}}|\\mathrm{{Aut}}|$ span {span:.1f}")
    quoted = [
        f"{design.BY_KEY[key].short} {_format_variation(variation[key])}"
        for key in QUOTED_KEYS
        if key in design.BY_KEY and key in variation
    ]
    if quoted:
        lines.append(", ".join(quoted))
    return "\n".join(lines)


def _draw_panel(
    ax: Axes,
    ladder: data.Ladder,
    representations: Sequence[design.Representation],
) -> dict[str, float | None]:
    """Draw one ladder into *ax* and return its per-arm variation ratios.

    Args:
        ax: The panel.
        ladder: The ladder to draw.
        representations: Registered representations, in draw order.

    Returns:
        ``{key: max/min over the per-rung medians}``, ``None`` where the ratio
        is not defined.
    """
    variation: dict[str, float | None] = {}
    for rep in representations:
        xs, ys, _ = _rung_medians(ladder, rep.key)
        variation[rep.key] = _variation(ys)
        muted = not rep.is_focus
        if xs:
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            ax.plot(
                [xs[i] for i in order],
                [ys[i] for i in order],
                label=design.label(rep),
                **design.line_kwargs(rep, muted=muted),
            )
        cx, cy = _censored_points(ladder, rep.key)
        if cx:
            ax.plot(cx, cy, **design.censored_kwargs(rep))
    ax.set_yscale("log")
    # Headroom for the annotation. The panel's upper-left is the only region a
    # cost-law panel reliably leaves empty -- the search-free arms sit at the
    # floor and the censored observations at the budget -- but only once the
    # top of the axis is lifted clear of the censored row.
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top * NOTE_HEADROOM)
    ax.set_title(ladder.title, fontsize=design.FS_TITLE)
    design.finish_axes(
        ax,
        xlabel=r"$\log_{10}|\mathrm{Aut}(G)|$",
        ylabel="seconds (process time)",
    )
    return variation


def _draw_arrows(
    ax: Axes,
    ladder: data.Ladder,
    representations: Sequence[design.Representation],
) -> int:
    """Stamp the "greater than this" arrow on every censored point of a panel.

    Drawn after the axes limits are settled, because the arrow's length is a
    fraction of the axes height and a later autoscale would leave it pointing
    at the wrong place.

    Args:
        ax: The panel.
        ladder: The ladder drawn into it.
        representations: Registered representations.

    Returns:
        The number of arrows drawn.
    """
    drawn = 0
    for rep in representations:
        cx, cy = _censored_points(ladder, rep.key)
        for x, y in zip(cx, cy):
            design.censor_arrow(ax, x, y, rep.colour)
            drawn += 1
    return drawn


def figure(
    records: data.Records,
    out: Path,
    *,
    arm: str = data.DEFAULT_ARM,
    omit: Sequence[str] = (),
) -> tuple[list[Path], dict[str, Any]]:
    """Draw the cost-law figure.

    Args:
        records: The loaded campaign.
        out: Output path without an extension.
        arm: Engine arm to read.  The ``no_bnb`` ablation is drawn by passing
            it here, which is the ``6.3`` consequence-2 check: the unpruned
            arm's flatness must survive with branch-and-bound off.
        omit: Registered arms to leave out on purpose.

    Returns:
        ``(paths_written, summary)``.  The summary carries, per ladder key,
        the measured ``max/min`` variation of every arm -- the numbers the
        caption quotes.

    Raises:
        CostLawFigureError: If no ladder carries a measured ``log10_aut``,
            which makes the abscissa undefined.
        design.UnknownRepresentationError: If the records carry a backend this
            package does not style.
    """
    all_ladders = data.ladders(records, arm=arm)
    usable = [lad for lad in all_ladders if any(g.log10_aut is not None for g in lad.graphs)]
    if not usable:
        raise CostLawFigureError(
            f"none of the {len(all_ladders)} ladder(s) in these records carries a "
            f"measured log10_aut; the cost law has no abscissa. The shards were "
            f"written with symmetry_available=false"
        )
    if len(usable) < len(all_ladders):
        LOGGER.warning(
            "%d of %d ladders have no measured log10_aut and are not drawn",
            len(all_ladders) - len(usable),
            len(all_ladders),
        )

    keys = sorted({str(r["representation"]) for lad in usable for g in lad.graphs for r in g.rows})
    representations = design.present(keys, omit=omit)
    missing = design.absent(keys)
    if missing:
        LOGGER.warning("registered arms with no data on any ladder: %s", list(missing))

    plt = design.style()
    ncols = min(NCOLS, len(usable))
    nrows = math.ceil(len(usable) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(design.text_width(), PANEL_HEIGHT_IN * nrows),
        squeeze=False,
    )
    flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    summary: dict[str, Any] = {"arm": arm, "ladders": {}, "n_censored_drawn": 0}
    for index, ladder in enumerate(usable):
        ax = flat[index]
        variation = _draw_panel(ax, ladder, representations)
        design.panel_letter(ax, chr(ord("a") + index))
        design.note_box(ax, _panel_note(ladder, variation), loc="upper left")
        summary["n_censored_drawn"] += _draw_arrows(ax, ladder, representations)
        summary["ladders"]["|".join(str(part) for part in ladder.key)] = {
            "n": ladder.n,
            "m": ladder.m,
            "max_degree": ladder.max_degree,
            "aut_span": ladder.aut_span,
            "variation": variation,
            "completion_rate": data.completion_rate([row for g in ladder.graphs for row in g.rows]),
        }
    for ax in flat[len(usable) :]:
        ax.set_visible(False)

    handles, labels = flat[0].get_legend_handles_labels()
    fig.tight_layout()
    design.shared_legend(fig, handles, labels, ncol=5, y=0.0)
    saved = design.save(fig, out)
    plt.close(fig)
    LOGGER.info("%s -> %s", out.name, ", ".join(str(p) for p in saved))
    return saved, summary


def caption(summary: dict[str, Any]) -> str:
    """Return the LaTeX caption, with every number read from *summary*.

    Args:
        summary: The mapping :func:`figure` returned.

    Returns:
        A ``\\caption{...}`` body, without the surrounding macro.
    """
    parts = [
        "Cost of a canonical encoding against the automorphism group, on ladders that "
        "hold $n$, $m$ and the entire degree sequence fixed. ",
        "Filled markers are per-rung medians over completed encodings, placed at the "
        "rung's median $\\log_{10}|\\mathrm{Aut}(G)|$ and joined in rung order; open "
        "markers with an upward arrow are right-censored observations, whose completion "
        "time is greater than the value plotted and which enter no median. ",
        "Search-free arms are drawn dashed: they are the null, and their flatness is "
        "what licenses reading a slope elsewhere as symmetry. ",
    ]
    for name, cell in sorted(summary.get("ladders", {}).items()):
        variation = cell.get("variation", {})
        quoted = ", ".join(
            f"{design.BY_KEY[key].short} {_format_variation(variation.get(key))}"
            for key in QUOTED_KEYS
            if key in variation
        )
        if quoted:
            parts.append(f"On {name}: {quoted}. ")
    parts.append(f"Arm: \\texttt{{{summary.get('arm', data.DEFAULT_ARM)}}}.")
    return "".join(parts)


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--records",
        nargs="+",
        required=True,
        help="glob(s) or paths of records_*.jsonl shards",
    )
    ap.add_argument(
        "--counters",
        nargs="+",
        default=None,
        help="accepted for CLI uniformity across this package; not read by this module",
    )
    ap.add_argument("--out-dir", type=Path, required=True, help="figure output directory")
    ap.add_argument(
        "--arm",
        default=data.DEFAULT_ARM,
        help="engine arm to draw (default, no_pairs_memo, no_bnb, no_pairs_memo_no_bnb)",
    )
    ap.add_argument(
        "--omit",
        nargs="*",
        default=(),
        help="registered arms to leave out on purpose; an unregistered arm raises",
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
    if args.counters:
        LOGGER.info("--counters is not read by fig_cost_law; ignoring %s", args.counters)
    records = data.load(args.records)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"{STEM}_{args.arm}"
    _, summary = figure(records, stem, arm=args.arm, omit=tuple(args.omit))
    (args.out_dir / f"{stem.name}.caption.tex").write_text(caption(summary) + "\n")
    LOGGER.info(
        "%d ladder(s), %d censored point(s) drawn",
        len(summary["ladders"]),
        summary["n_censored_drawn"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
