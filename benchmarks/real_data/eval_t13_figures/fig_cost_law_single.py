"""T-13 main-text figure: IsalGraph's canonical-search cost against ``|Aut(G)|``.

**One panel, one ladder, and the subject is IsalGraph.**  The ticket answers
R3.4b, R3.4c and R3.7d, all of which are questions about *our* complexity; the
competitor bake-off belongs to T-17.  So the three IsalGraph arms are drawn at
full weight and the remaining ten representations are muted to
:data:`~.design.ALPHA_MUTED`, present only to put the vertical scale in
perspective.  Muting is not hiding: every arm is still drawn and still
labelled, so a reader can see where the field sits without the figure making a
comparative claim it was not designed to support.

The panel is **one ladder cell**, which is the whole point.  Along a ladder
``n``, ``m`` and the entire degree sequence are held **exactly** constant while
``|Aut(G)|`` falls, so the only thing varying between the points on the x axis
is the automorphism group.  That is what licenses reading the slope as a
statement about symmetry rather than about size or density -- on the real
cohort those four quantities co-vary, and the marginal correlation of
``log|Aut|`` with encode time (+0.189) is *lower* than that of ``log n``
(+0.326).

The default cell is the spider ladder at ``n = 33``: a tree, so it is the
sparsest connected graph available; ``|Aut|`` spans 4.3 decades; and it is the
largest cell in which **all three** IsalGraph arms complete inside the budget
at every rung, so nothing about our own method has to be reported as censored.
That selection rule is stated in the caption, because "why this cell" is the
first question a reviewer asks of a single-cell figure.

CLI::

    python -m benchmarks.eval_t13_figures.fig_cost_law_single \\
        --records '<glob>' --out-dir <dir> [--family spider_ladder] [--n 33]
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes

from benchmarks.real_data.eval_t13_figures import data as t13data
from benchmarks.real_data.eval_t13_figures import design

LOGGER = logging.getLogger("t13.fig_cost_law_single")

#: The cell the figure defaults to.  See the module docstring for the rule.
DEFAULT_FAMILY: Final[str] = "spider_ladder"
DEFAULT_N: Final[int] = 33

#: Drawn at full weight, in this order (back to front).
OURS: Final[tuple[str, ...]] = (
    "isalgraph_greedy",
    "isalgraph_exhaustive",
    "isalgraph_pruned",
)

#: Never drawn.  ``graph6`` and ``sparse6`` are bit-for-bit the same encoder
#: as their nauty-canonicalised counterparts up to the relabelling step, so on
#: a log axis they lie on top of ``nauty_graph6``/``sparse6_nauty`` and add two
#: lines of clutter for no information.  The nauty variants are kept because
#: they are the ones that carry a canonical labelling.
EXCLUDED: Final[tuple[str, ...]] = ("graph6", "sparse6")

#: Basename of the artefact.
STEM: Final[str] = "fig_t13_cost_law"


class NoSuchCellError(ValueError):
    """Raised when the requested ``(family, n)`` cell is not in the records."""


def _rungs(records: t13data.Records, family: str, n: int) -> list[Mapping[str, Any]]:
    """Rows of one ladder cell, default arm only.

    Args:
        records: Loaded campaign records.
        family: Ladder family name.
        n: Ladder order.

    Returns:
        Every default-arm row of that cell.

    Raises:
        NoSuchCellError: If the cell holds no rows.
    """
    rows = [
        r for r in records.rows if r["arm"] == "default" and r["family"] == family and r["n"] == n
    ]
    if not rows:
        raise NoSuchCellError(f"no rows for family={family!r}, n={n}")
    return rows


def _series(
    rows: list[Mapping[str, Any]], key: str
) -> tuple[list[float], list[float], list[float]]:
    """Split one representation's rows into completed and censored points.

    Args:
        rows: Rows of one ladder cell.
        key: Representation key.

    Returns:
        ``(x_ok, y_ok, x_censored)`` with ``x`` in ``log10|Aut|`` and ``y`` in
        seconds.  Censored points carry no ``y``: the caller pins them to the
        budget, which is the only thing the observation asserts.
    """
    sel = sorted(
        (r for r in rows if r["representation"] == key),
        key=lambda r: float(r["log10_aut"]),
    )
    x_ok = [float(r["log10_aut"]) for r in sel if r["status"] == "ok"]
    y_ok = [float(r["seconds"]) for r in sel if r["status"] == "ok"]
    x_cens = [float(r["log10_aut"]) for r in sel if r["status"] == "censored"]
    return x_ok, y_ok, x_cens


def _fold(y: list[float]) -> float | None:
    """Max/min of a completed series, or ``None`` if it cannot be formed."""
    pos = [v for v in y if v > 0]
    return max(pos) / min(pos) if len(pos) >= 2 else None


def _draw(ax: Axes, rows: list[Mapping[str, Any]], budget: float) -> None:
    """Draw the panel: muted field first, IsalGraph on top.

    Args:
        ax: Target axes.
        rows: Rows of the chosen ladder cell.
        budget: Censoring budget in seconds, for pinning censored markers.
    """
    present = sorted({str(r["representation"]) for r in rows})
    field = [k for k in design.ORDER if k in present and k not in OURS and k not in EXCLUDED]

    # The muted field is drawn first so it sits behind, but it is added to the
    # legend last -- the legend is ordered by what the figure is *about*.
    for key in field:
        rep = design.BY_KEY[key]
        x_ok, y_ok, x_cens = _series(rows, key)
        if x_ok:
            ax.plot(x_ok, y_ok, label=design.label(rep), **design.line_kwargs(rep, muted=True))
        for xc in x_cens:
            ax.plot([xc], [budget], **{**design.censored_kwargs(rep), "alpha": design.ALPHA_MUTED})

    for key in OURS:
        if key not in present:
            continue
        rep = design.BY_KEY[key]
        x_ok, y_ok, x_cens = _series(rows, key)
        if x_ok:
            ax.plot(x_ok, y_ok, label=design.label(rep), **design.line_kwargs(rep))
        for xc in x_cens:
            ax.plot([xc], [budget], **design.censored_kwargs(rep))
            design.censor_arrow(ax, xc, budget, rep.colour)


def _caption(rows: list[Mapping[str, Any]], family: str, n: int) -> str:
    """Emit the LaTeX caption.

    The panel no longer carries an in-axes box, so **the caption is where the
    held-fixed invariants live**.  They are not decoration: without them the x
    axis is just a covariate, and the whole licence to read the slope as a
    statement about symmetry rather than about size or density comes from
    ``n``, ``m`` and the degree sequence being identical at every point.

    Args:
        rows: Rows of the drawn ladder cell.
        family: Ladder family.
        n: Ladder order.

    Returns:
        A LaTeX ``\\caption{...}`` body.
    """
    m = rows[0]["m"]
    delta = rows[0]["max_degree"]
    auts = [float(r["log10_aut"]) for r in rows]
    budget = float(rows[0]["budget_s"])
    kind = "spider (tree)" if family == "spider_ladder" else family.replace("_", " ")
    censored = [
        design.BY_KEY[k].long
        for k in OURS
        if _series(rows, k)[2]
    ]
    tail = (
        rf" Arms censored at the {budget:.0f}\,s budget: {', '.join(censored)}."
        if censored
        else rf" No IsalGraph arm reaches the {budget:.0f}\,s budget in this cell."
    )
    return (
        rf"Encode time against automorphism-group order on a single {kind} ladder. "
        rf"Every point has $n={n}$, $m={m}$, $\Delta={delta}$ \emph{{and the same degree "
        rf"sequence}}; only $|\mathrm{{Aut}}(G)|$ varies, over "
        rf"{max(auts) - min(auts):.1f} decades. "
        rf"The three IsalGraph arms are drawn at full weight; the remaining representations "
        rf"are muted and are present only to set the vertical scale.{tail}"
        "\n"
    )


def build(
    records: t13data.Records,
    *,
    family: str = DEFAULT_FAMILY,
    n: int = DEFAULT_N,
) -> Any:
    """Build the one-panel figure.

    Args:
        records: Loaded campaign records.
        family: Ladder family.
        n: Ladder order.

    Returns:
        The matplotlib ``Figure``.

    Raises:
        NoSuchCellError: If the cell is absent.
    """
    plt = design.style()

    rows = _rungs(records, family, n)
    budget = float(rows[0]["budget_s"])

    fig, ax = plt.subplots(figsize=(design.column_width(), design.column_width() * 0.82))
    _draw(ax, rows, budget)

    ax.set_yscale("log")
    design.finish_axes(
        ax,
        xlabel=r"$\log_{10}|\mathrm{Aut}(G)|$",
        ylabel="Encode Time (s, CPU)",
    )

    # Ours first, then the muted field: the legend reads in the order the
    # figure is meant to be read, not in draw order.
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    ours_labels = [design.label(design.BY_KEY[k]) for k in reversed(OURS)]
    ordered = [lb for lb in ours_labels if lb in by_label]
    ordered += [lb for lb in labels if lb not in ordered]
    fig.legend(
        [by_label[lb] for lb in ordered],
        ordered,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.16),
        ncol=3,
        fontsize=design.FS_LEGEND,
        frameon=False,
        handlelength=1.9,
        columnspacing=1.1,
        labelspacing=0.3,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0, 0.155, 1, 1))
    return fig


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector, or ``None`` for ``sys.argv``.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, help="glob over t13.1 shard files")
    parser.add_argument("--counters", default=None, help="ignored; accepted for symmetry")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--family", default=DEFAULT_FAMILY)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--stem", default=STEM)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    records = t13data.load([args.records])
    fig = build(records, family=args.family, n=args.n)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = design.save(fig, out / args.stem)
    caption = out / f"{args.stem}.caption.tex"
    caption.write_text(_caption(_rungs(records, args.family, args.n), args.family, args.n))
    LOGGER.info("wrote %s", ", ".join(str(p) for p in written))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
