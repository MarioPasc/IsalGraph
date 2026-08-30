"""The three size-profile figures.

All three share one x axis --- graph size ``n`` --- and one regime split:
**exact GED for ``n <= 12``, the LB/UB bracket above it**, which is where exact
computation stops being feasible. The bracket is never averaged into a midpoint
and never interpolated (``approx_ged.md`` section 4); LB and UB are drawn as two
series, and in figure 3 as the two edges of a shaded band.

Aggregation across datasets at fixed ``(representation, reference, n)`` is a
**Fisher-z weighted mean, weighted by ``n_graphs - 3``**, not by pair count. The
effective sample size of a pairwise correlation is governed by graphs, not by
the quadratically many pairs they induce; weighting by pairs would overstate
precision by roughly the same factor the pair-level bootstrap does.

**These figures are descriptive.** The Benjamini-Hochberg correction applied
here is local to each figure and is stated on it. It is not the pre-registered
F0/F1/F2 family and nothing here feeds those.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, MaxNLocator, MultipleLocator
from scipy import stats

from benchmarks.real_data.eval_t06_figures import design

LOGGER: Final = logging.getLogger(__name__)

#: Re-exported from the single design source so these figures and the
#: information-content figure cannot disagree about where the exact-GED regime
#: ends or at what level the local correction runs.
EXACT_CEILING: Final[int] = design.EXACT_CEILING

FDR_Q: Final[float] = design.FDR_Q

#: Figure fractions the axes stop at and the legend hangs from. The gap
#: between them is the x-label; anything more reads as a layout error.
AXES_BOTTOM: Final[float] = 0.135
LEGEND_Y: Final[float] = 0.055

#: The combined figure hangs its legend higher than the single-panel ones.
#: They clear one legend row under the x-label; it clears two, and at
#: :data:`LEGEND_Y` the first row falls a centimetre clear of the label with
#: nothing between them.
LEGEND_Y_COMBINED: Final[float] = 0.098

#: Figure fraction between the two combined-figure legend rows.
LEGEND_ROW_GAP: Final[float] = 0.046

#: Above this many points in one series, draw the interval as a band rather
#: than as error bars: a picket fence of caps hides the trend it qualifies.
DENSE_SERIES: Final[int] = 20

#: Draw order and display names now come from the registry. They used to be
#: duplicated here, and the colour was assigned by walking a palette over the
#: representations *present in this call* -- so a representation's colour
#: depended on which others had landed, and two figures of the same campaign
#: could give one backend two colours.
REPRESENTATION_ORDER: Final[tuple[str, ...]] = design.FIGURE_ORDER

DATASET_MARKERS: Final[tuple[str, ...]] = ("o", "s", "^", "v", "D", "P", "X", "*", "<", ">")

DISPLAY: Final[dict[str, str]] = {r.key: r.short for r in design.REPRESENTATIONS}

# ---------------------------------------------------------------------------
# Combined-figure x ticks. The halves do not share an x *scale* --- exact GED
# spans ten strata where the structural reference and the bracket span
# sixty-two --- so the most a reader can be given is a shared set of tick
# *values* wherever two panels cover the same range of n.
#
# The bracket panels and the structural panel do cover the same range, so they
# get the identical fixed set below and the eye can carry a value across the
# rule. The exact-GED panel covers 3-12, where that set has no tick at all, so
# it is stepped by 1 instead.
#
# The two cannot be unioned onto a *linear* structural panel: 3..12 is 12 % of
# an axis that runs to 76, which at 6.5 pt puts ten labels into roughly 7 mm.
# They are unioned onto a piecewise one instead --- see STRUCT_ZOOM_FRACTION.
# ---------------------------------------------------------------------------

#: Shared by panel (a) and the bracket small multiples, which span the same n.
BRACKET_TICKS: Final[tuple[int, ...]] = (15, 30, 45, 60, 75)

#: Step for the exact-GED panel, whose whole range sits below the first
#: bracket tick.
#:
#: **2, not 1.** The manuscript's text block is 4.72 in (``design.text_width``),
#: which leaves this panel 1.00 in for ten strata. Measured, a 2-digit label at
#: 5.7 pt is 0.107 in wide, so step 1 gives a *negative* 0.005 in gap between
#: "10", "11" and "12" --- it shipped that way for one render. Step 2 lands the
#: same five values the structural zoom labels, so the two windows now carry an
#: identical set.
EXACT_TICK_STEP: Final[int] = 2

# ---------------------------------------------------------------------------
# The structural panel's piecewise x scale.
#
# (a) spans 3-76 where (b)'s wide axes stop at 12, so on a linear (a) the
# exact-GED window is a 12 % sliver and the eye compares (a)'s *large*-n region
# against (b)'s small-n one. That misreading is not hypothetical: measured at
# matched n, IsalGraph leads both nauty arms below the ceiling and trails above
# it under every reference --- wl, lb and ub alike --- so the split is n, not
# the yardstick, and a figure that invites the other reading is a figure that
# will be misquoted. Giving the window real width makes the matched-n
# comparison the easy one.
#
# The cost is that (a)'s x axis is no longer linear, so slopes either side of
# the knot are not comparable and the panel must say so. The knot is drawn.
# ---------------------------------------------------------------------------

#: Node count the structural panel's scale breaks at, half a stratum above the
#: exact-GED ceiling so it falls between n = 12 and n = 13 rather than on a
#: measured point.
STRUCT_ZOOM_KNOT: Final[float] = EXACT_CEILING + 0.5

#: Share of the structural panel's width given to ``n <= STRUCT_ZOOM_KNOT``.
#: Ten strata of sixty-two, so 0.50 is a little over three times their linear
#: share. Higher squashes the tail, which is where the result lives; the tail
#: still gets 0.86 in for its 52 strata at this value.
STRUCT_ZOOM_FRACTION: Final[float] = 0.50

#: Data limits the piecewise map is pinned to. Fixed rather than taken from the
#: data so the two halves of the scale cannot drift if a stratum is added.
STRUCT_X_LIMITS: Final[tuple[float, float]] = (2.0, 78.0)

#: Labelled ticks inside the zoomed window --- the same five values the exact
#: panel labels, so one window reads across the other.
#:
#: **Every stratum 3..12 does not fit, and no zoom fraction makes it fit.** The
#: panel is 1.72 in of a 4.72 in text block; at ``fraction = 0.50`` the zoom is
#: 0.86 in over 10.5 strata, a pitch of 0.082 in, against the 0.140 in a 2-digit
#: label needs at 6.5 pt (measured, not estimated from the point size --- the
#: estimate under-predicts by a third). Step 1 would need ``fraction = 0.86``,
#: which leaves the 52-stratum tail 0.24 in.
ZOOM_TICKS: Final[tuple[int, ...]] = (4, 6, 8, 10, 12)

#: Labelled ticks outside it. :data:`BRACKET_TICKS` minus 15.
#:
#: **15 cannot be labelled on (a) at any zoom fraction or panel width.** It sits
#: 2.5 strata past the knot on the compressed side, where the pitch is
#: ``2.5 (1 - fraction) / (78 - 12.5)`` of the panel --- 0.033 in here against
#: the 0.140 in needed --- and *widening the zoom moves it closer to 12, not
#: further*, because it shrinks the compressed side's slope.
#:
#: So one of 12 and 15 goes, and it is 15. 12 is the exact-GED ceiling, it is
#: where the axis breaks, and the caption's argument is phrased as "below the
#: ceiling / above it"; 15 is an arbitrary grid value the bracket panels carry
#: anyway, and it keeps a tick mark here without a label.
STRUCT_TAIL_TICKS: Final[tuple[int, ...]] = (30, 45, 60, 75)

#: Figure fraction the "(b)" panel letter occupies to the left of its axes.
#: :func:`design.panel_letter` right-aligns at -0.02 axes fraction, so a rule
#: centred on the raw gap runs into the letter; the rule is centred on the gap
#: minus this instead.
PANEL_LETTER_WIDTH: Final[float] = 0.020

#: Legend rows below the combined figure. Each is centred independently, so a
#: seven-entry legend reads 4 over 3 rather than 4 over 3 hanging left.
LEGEND_ROWS: Final[int] = 2


@dataclass(frozen=True)
class AggregatePoint:
    """One dataset-aggregated point.

    Attributes:
        representation: Backend name.
        reference: ``exact``, ``lb`` or ``ub``.
        n: Node count defining the stratum.
        rho: Fisher-z weighted mean correlation, back-transformed.
        ci_lo: Lower 95 % bound, back-transformed.
        ci_hi: Upper 95 % bound, back-transformed.
        p_value: Two-sided p from the weighted z and its standard error.
        n_datasets: Datasets contributing.
        n_graphs: Total graphs across contributing datasets.
        n_pairs: Total pairs across contributing datasets.
    """

    representation: str
    reference: str
    n: int
    rho: float
    ci_lo: float
    ci_hi: float
    p_value: float
    n_datasets: int
    n_graphs: int
    n_pairs: int


def _regime(n: int) -> tuple[str, ...]:
    """Return the reference names that apply at node count *n*.

    Args:
        n: Node count.

    Returns:
        ``("exact",)`` at or below the ceiling, else ``("lb", "ub")``.
    """
    return ("exact",) if n <= EXACT_CEILING else ("lb", "ub")


def load_rows(path: Path, *, keep_reference: str | None = None) -> list[dict[str, Any]]:
    """Load usable stratum rows, dropping degenerate ones.

    Args:
        path: ``size_profile.json``.
        keep_reference: When ``None`` (the default, and what every published
            figure uses) each row is kept only under the GED regime that applies
            at its node count --- ``exact`` at or below the ceiling, the
            ``lb``/``ub`` bracket above it. When a name is given, that reference
            alone is kept, at **every** node count. T-28's references carry no
            bracket and no exact ceiling, so the regime filter would otherwise
            drop every one of their rows silently.

    Returns:
        Rows with a defined rho, restricted as above and to the **primary** D14
        arm.

    Raises:
        ValueError: If the profile carries an arm this function does not know
            how to reduce, which would otherwise be plotted alongside the
            primary rows as if it were more data.
    """
    payload = json.loads(path.read_text())
    out: list[dict[str, Any]] = []
    for row in payload["rows"]:
        if row["rho"] is None:
            continue
        if keep_reference is None:
            if row["reference"] not in _regime(int(row["n"])):
                continue
        elif row["reference"] != keep_reference:
            continue
        # size_profile.json can carry two arms since schema t06.size_profile.2.
        # A profile written with --arm both holds each stratum twice, and without
        # this filter every figure would silently plot the complete-case arm on
        # top of the primary one -- doubling n and biasing the curve, with
        # nothing in the output to show it.
        arm = row.get("arm", "primary")
        if arm != "primary":
            if arm != "complete_case":
                raise ValueError(f"{path}: unknown D14 arm {arm!r} in a stratum row")
            continue
        out.append(row)
    return out


def aggregate(rows: list[dict[str, Any]]) -> list[AggregatePoint]:
    """Aggregate across datasets by Fisher-z weighted mean.

    Weight is ``n_graphs - 3``, the graph-level effective sample size, never the
    pair count.

    Args:
        rows: Usable stratum rows.

    Returns:
        One point per ``(representation, reference, n)``.
    """
    buckets: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault((row["representation"], row["reference"], int(row["n"])), []).append(row)

    points: list[AggregatePoint] = []
    for (rep, ref, n), group in sorted(buckets.items()):
        z = np.arctanh(np.clip([g["rho"] for g in group], -0.999999, 0.999999))
        w = np.array([max(g["n_graphs"] - 3, 1) for g in group], dtype=float)
        z_bar = float((w * z).sum() / w.sum())
        se = float(1.0 / np.sqrt(w.sum()))
        points.append(
            AggregatePoint(
                representation=rep,
                reference=ref,
                n=n,
                rho=float(np.tanh(z_bar)),
                ci_lo=float(np.tanh(z_bar - 1.96 * se)),
                ci_hi=float(np.tanh(z_bar + 1.96 * se)),
                p_value=float(2.0 * stats.norm.sf(abs(z_bar) / se)),
                n_datasets=len(group),
                n_graphs=int(sum(g["n_graphs"] for g in group)),
                n_pairs=int(sum(g["n_pairs"] for g in group)),
            )
        )
    return points


def benjamini_hochberg(p_values: list[float], q: float = FDR_Q) -> list[bool]:
    """Return the BH rejection mask at level *q*.

    Args:
        p_values: Uncorrected two-sided p-values.
        q: False-discovery rate.

    Returns:
        One boolean per input, in input order.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    ranked = np.asarray(p_values, dtype=float)[order]
    thresholds = q * (np.arange(1, m + 1) / m)
    passed = ranked <= thresholds
    cutoff = int(np.max(np.flatnonzero(passed)) + 1) if passed.any() else 0
    mask = np.zeros(m, dtype=bool)
    if cutoff:
        mask[order[:cutoff]] = True
    return [bool(v) for v in mask]


def _style() -> Any:
    """Apply the shared IEEE style and return the pyplot module.

    Returns:
        The ``matplotlib.pyplot`` module, styled.
    """
    return design.style()


def _colours(names: tuple[str, ...]) -> dict[str, Any]:
    """Return the pinned colour per representation.

    Args:
        names: Representation names.

    Returns:
        Mapping from representation to colour. Pinned in
        :data:`design.REPRESENTATIONS`, so it does not depend on which other
        representations are present in this call.
    """
    return {name: design.BY_KEY[name].colour for name in names if name in design.BY_KEY}


def _significance_note(n_points: int) -> str:
    """Return the textbox wording for the significance marker.

    Args:
        n_points: Number of points the correction ranged over.

    Returns:
        The caption text.
    """
    return (
        "○  significant point.  A pair enters stratum $n$ only when both\n"
        "graphs have exactly $n$ nodes, so $|n_i-n_j|$ is identically 0.\n"
        f"Benjamini–Hochberg at q = {FDR_Q:g} over all {n_points} points\n"
        "in this figure, against H₀: ρ = 0.\n"
        "Bars are 95 % intervals from a Fisher-z weighted\n"
        "mean across datasets, weighted by graphs (n−3),\n"
        "not by pairs. Descriptive: not the pre-registered family."
    )


def _bracket_representations(points: list[AggregatePoint]) -> tuple[str, ...]:
    """Return the representations carrying bracket data, in draw order.

    Derived from the points rather than hard-coded, so a campaign that adds or
    drops an arm re-renders without an edit to this module.

    Args:
        points: Aggregated points.

    Returns:
        Representation keys with at least one ``lb`` or ``ub`` point.
    """
    have = {p.representation for p in points if p.reference in ("lb", "ub")}
    return tuple(r for r in design.FIGURE_ORDER if r in have)


# ---------------------------------------------------------------------------
# The undetermined-onset rule, removed 2026-08-26. Do not reinstate it.
#
# ``figure_one`` shaded its right tail and captioned it "rho not separable
# from 0 (n > onset)", where ``onset`` came from a helper returning the
# smallest node count whose interval covers zero and beyond which none
# resolves. Three reasons it is gone, in order of severity:
#
# 1. It is a data-dependent stopping rule with no multiplicity control. The
#    figure's own BH correction runs over the p-values; the onset ran over
#    the intervals and was corrected by nothing.
# 2. It is confounded with support, not with rho. Above n = 56 a stratum
#    carries 9-17 graphs from 1-2 datasets, and the Fisher-z weight is
#    ``n_graphs - 3``, so the weights there are 6-14. An interval covering
#    zero on that support is a statement about power, not about rho. Several
#    of those strata are single-dataset, so the "weighted mean across
#    datasets" is one dataset's rho wearing an aggregate's label.
# 3. It was quoted nowhere -- not in the manuscript, not in a response-letter
#    fragment, not in a ticket note. Only on the figure, which is the worst
#    place for a claim nothing else defends.
# ---------------------------------------------------------------------------


def figure_one(points: list[AggregatePoint], out: Path) -> list[str]:
    """Figure 1 --- the within-`n` collapse, per regime and per representation.

    **Left, exact GED at ``n <= 12``.** Every representation, family-emphasized:
    canonical codes solid at full weight, serializations dashed and
    half-transparent. Ground truth exists here and the head-to-head resolves, so
    the per-representation detail is the content.

    **Right, the bracket above ``n = 12``, as small multiples.** Fourteen series
    on one axes was a texture rather than a figure, and collapsing them into a
    single envelope answered that by discarding the per-representation detail.
    The grid keeps both: one small panel per representation carrying bracket
    data, each showing its own two bounds against the gray envelope of the whole
    field, plus a final panel overlaying every arm. The grid is sized so its top
    and bottom rows align with the exact-GED panel, so the two regimes read as
    one figure. Panel titles are black rather than the representation's colour:
    the two lines inside each panel already carry it, and colouring the title
    too made the grid read as six separate figures.

    **No region of the bracket is shaded.** See the note above
    :func:`figure_one` for why the undetermined-onset rule was removed.

    Args:
        points: Aggregated points.
        out: Output path without an extension.

    Returns:
        Paths written.
    """
    plt = _style()

    exact_reps = tuple(
        r
        for r in design.FIGURE_ORDER
        if any(p.representation == r and p.reference == "exact" for p in points)
    )
    bracket_reps = _bracket_representations(points)
    flags = benjamini_hochberg([p.p_value for p in points])
    significant = {(p.representation, p.reference, p.n) for p, f in zip(points, flags) if f}

    ncols = 2
    nrows = max(1, -(-(len(bracket_reps) + 1) // ncols))
    fig = plt.figure(figsize=(design.text_width(), 1.15 * nrows + 0.95))
    grid = fig.add_gridspec(
        nrows,
        1 + ncols,
        width_ratios=[1.55, *([1.0] * ncols)],
        wspace=0.13,
        hspace=0.30,
    )
    left = fig.add_subplot(grid[:, 0])

    for key in exact_reps:
        rep = design.BY_KEY.get(key)
        if rep is None:
            continue
        sel = sorted(
            (p for p in points if p.representation == key and p.reference == "exact"),
            key=lambda p: p.n,
        )
        style = design.line_kwargs(rep, "exact")
        left.errorbar(
            [p.n for p in sel],
            [p.rho for p in sel],
            yerr=[[p.rho - p.ci_lo for p in sel], [p.ci_hi - p.rho for p in sel]],
            elinewidth=0.55,
            capsize=1.4,
            label=rep.short,
            **style,
        )
        marked = [(p.n, p.rho) for p in sel if (key, "exact", p.n) in significant]
        if marked and (rep.family in design.PRIMARY_FAMILIES or rep.is_ours):
            left.scatter(
                [m[0] for m in marked],
                [m[1] for m in marked],
                s=design.MS_SIGNIFICANT,
                facecolors="none",
                edgecolors=rep.colour,
                linewidths=0.85,
                zorder=rep.zorder + 1,
            )
    left.axhline(0.0, color=design.INK_RULE, linewidth=0.6, linestyle=":")
    left.set_title(f"Exact GED  ($n \\leq {EXACT_CEILING}$)", fontsize=design.FS_TITLE, pad=3)
    design.finish_axes(
        left,
        xlabel="Graph Size ($n$)",
        ylabel=r"Spearman $\rho$ (Distance vs GED), Within Equal $n$",
    )
    left.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    left.set_ylim(-0.75, 1.05)

    envelope: dict[int, list[float]] = {}
    for p in points:
        if p.reference != "exact":
            envelope.setdefault(p.n, []).append(p.rho)
    span = sorted(envelope)

    panels = [*bracket_reps, None]
    for index, key in enumerate(panels):
        ax = fig.add_subplot(grid[index // ncols, 1 + index % ncols])
        if span:
            ax.fill_between(
                span,
                [min(envelope[n]) for n in span],
                [max(envelope[n]) for n in span],
                color="0.55",
                alpha=0.20,
                linewidth=0,
                zorder=1,
            )
        keys = bracket_reps if key is None else (key,)
        for one in keys:
            rep = design.BY_KEY[one]
            for ref, width in (("lb", 0.75), ("ub", 1.0)):
                sel = sorted(
                    (p for p in points if p.representation == one and p.reference == ref),
                    key=lambda p: p.n,
                )
                if not sel:
                    continue
                style = design.line_kwargs(rep, ref, primary=frozenset(design.Family))
                style["marker"] = "None"
                style["linewidth"] = (design.LW_REFERENCE if rep.is_ours else 0.95) * width
                if key is None and not rep.is_ours:
                    style["alpha"] = 0.55
                ax.plot([p.n for p in sel], [p.rho for p in sel], **style)
        ax.axhline(0.0, color=design.INK_RULE, linewidth=0.6, linestyle=":")
        ax.set_ylim(-0.75, 1.05)
        # Black, not the representation's colour. The panel's two lines already
        # carry that colour, and a coloured title made the grid read as six
        # separate figures rather than one small-multiples panel.
        ax.set_title(
            "Every Arm" if key is None else design.BY_KEY[key].short,
            fontsize=design.FS_TITLE - 0.7,
            pad=2,
            color="black",
        )
        ax.grid(True, alpha=design.GRID_ALPHA, linewidth=design.GRID_LW)
        ax.tick_params(labelsize=design.FS_TICK - 0.8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        if index // ncols != nrows - 1:
            ax.set_xticklabels([])
        if index % ncols != 0:
            ax.set_yticklabels([])
        if index == len(panels) - 1:
            ax.set_xlabel("Graph Size ($n$)", fontsize=design.FS_LABEL - 0.5)

    header = f"GED Bracket  ($n > {EXACT_CEILING}$)\nDashed LB, solid UB;  gray: the whole field"
    fig.text(
        0.695,
        0.995,
        header,
        ha="center",
        va="top",
        fontsize=design.FS_TITLE - 0.6,
        color="0.25",
        linespacing=1.35,
    )

    # 🔴 Two rows, not one row of seven. This is the fix for the placement defect
    # 05_results.tex:463-470 records as open, and it is the whole fix: measured at
    # the declared 4.72 in, the one-row legend is 6.83 in wide on its own and hangs
    # 1.05 in off the left edge, so `bbox_inches="tight"` writes a 6.83 in box and
    # \includegraphics scales the page copy to 69 %, landing declared 5.5-6.5 pt
    # labels at 3.8-4.5 pt. The axes span 0.35-4.67 in and the title 2.23 in, both
    # inside the block -- so the legend is the entire cause. The docstring of
    # figure_one_single_reference used to say a narrow render "also needs a
    # shorter title"; measured, the title is innocent.
    handles, labels = left.get_legend_handles_labels()
    design.shared_legend_rows(
        fig,
        handles,
        labels,
        counts=design.balanced_rows(len(labels), LEGEND_ROWS),
        y=LEGEND_Y,
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.875, bottom=AXES_BOTTOM)
    saved = [str(q) for q in design.save(fig, out)]
    plt.close(fig)
    return saved


def _single_reference_onset(points: list[AggregatePoint], reference: str) -> int | None:
    """Return the node count beyond which no interval of *reference* excludes zero.

    The bracket version of this test reads ``p.reference != "exact"``, which is
    meaningless for a reference that has no bracket. Here the whole series is one
    reference and the question is asked of it directly.

    Args:
        points: Aggregated points.
        reference: The single reference name.

    Returns:
        The onset node count, or ``None`` when every size still resolves.
    """
    sel = [p for p in points if p.reference == reference]
    resolved = {p.n for p in sel if not p.ci_lo <= 0.0 <= p.ci_hi}
    covered = sorted({p.n for p in sel if p.ci_lo <= 0.0 <= p.ci_hi})
    return next((n for n in covered if not any(m >= n for m in resolved)), None)


def figure_one_single_reference(
    points: list[AggregatePoint],
    out: Path,
    *,
    reference: str,
    ref_label: str,
    degenerate: str | None = None,
    width: float | None = None,
) -> list[str]:
    """Figure 1 for a reference that needs no approximation: one panel, one axis.

    :func:`figure_one` is split in two because graph edit distance is only exact
    to ``n = EXACT_CEILING``; above it the reference is a *bracket*, and a bracket
    cannot share an axis with an exact value without inviting the reader to read
    one as the other. That constraint is a property of GED, not of the question.

    The Weisfeiler-Lehman kernel distance is computed exactly at every size, so
    there is no bracket, no regime split and no ceiling: every representation is
    one series over the whole size range, on one axis, directly comparable.

    Intervals are drawn as error bars for short series and as a band once a
    series passes ``DENSE_SERIES`` points, following this module's convention ---
    a picket fence of caps hides the trend it qualifies.

    Args:
        points: Aggregated points; only those carrying *reference* are drawn.
        out: Output path without an extension.
        reference: The reference key, e.g. ``wl``.
        ref_label: Human-readable name for the axis and title, e.g. ``WL kernel``.
        degenerate: A representation whose own distance **is** this reference, so
            its rho is exactly 1.0 by construction. It is drawn --- hiding it
            would be a silent exclusion --- but annotated, because a flat line at
            1.0 otherwise reads as a competitor that solved the problem.
        width: Render width in inches. Defaults to ``design.text_width()``
            (7.0 in, the frozen IEEE constant). **Point sizes inside a figure
            are absolute**, so a figure rendered wider than the text block it
            is placed in has its labels scaled down on the page: the published
            ``rho_vs_size.pdf`` is 7.03 in inside a 4.72 in Pattern Recognition
            block, which lands its 5.5-6.5 pt labels at 3.7-4.4 pt. Pass 4.72
            to render at the placement width instead. The frozen constant is
            left alone --- a test pins it to the submitted PDF.

            ⚠ Passing a narrower width is NOT sufficient on its own.
            ``save_figure`` writes with ``bbox_inches='tight'``, so the output
            box is the CONTENT box: at 4.72 in the seven-column legend overflows
            and the tight box expands back out, with nothing in the output to
            say so. **This function still has that defect.** Measured on the
            current code, it declares 4.72 in and writes 6.83 in, and the
            legend alone accounts for all of it --- 6.83 in wide, spanning
            -1.05 to 5.77 while the axes span 0.35 to 4.67. The title is not
            the cause; at 2.23 in it sits inside the axes.

            The defect now has a fix, which this function has not adopted:
            :func:`design.shared_legend_rows` splits the same entries over
            centred rows, and :func:`figure_one_combined` carries the identical
            seven labels at 4.72 in and writes 4.77 in. Adopting it here means
            re-rendering a figure a test pins, so it is a deliberate deferral
            rather than an oversight.

    Returns:
        Paths written.
    """
    plt = _style()

    selected = [p for p in points if p.reference == reference]
    # FIGURE_ORDER, not ORDER: the GED figures above filter through it, and it is
    # what excludes the arms design.py marks in_figures=False. Filtering this one
    # through ORDER would let a reference figure carry an arm its GED sibling
    # drops, with nothing in either output to say the arm sets differ.
    present = tuple(r for r in design.FIGURE_ORDER if any(p.representation == r for p in selected))

    flags = benjamini_hochberg([p.p_value for p in selected])
    significant = {(p.representation, p.n) for p, f in zip(selected, flags) if f}

    figure_width = design.text_width() if width is None else width
    fig = plt.figure(figsize=(figure_width, 3.05 * figure_width / design.text_width()))
    axis = fig.add_subplot(1, 1, 1)

    for key in present:
        rep = design.BY_KEY.get(key)
        if rep is None:
            continue
        series = sorted((p for p in selected if p.representation == key), key=lambda p: p.n)
        xs = [p.n for p in series]
        ys = [p.rho for p in series]
        style = design.line_kwargs(rep, None)
        if len(series) > DENSE_SERIES:
            axis.plot(xs, ys, label=rep.short, **style)
            axis.fill_between(
                xs,
                [p.ci_lo for p in series],
                [p.ci_hi for p in series],
                color=rep.colour,
                alpha=0.13,
                linewidth=0,
                zorder=rep.zorder - 1,
            )
        else:
            axis.errorbar(
                xs,
                ys,
                yerr=[
                    [p.rho - p.ci_lo for p in series],
                    [p.ci_hi - p.rho for p in series],
                ],
                elinewidth=0.55,
                capsize=1.4,
                label=rep.short,
                **style,
            )
        marked = [(p.n, p.rho) for p in series if (key, p.n) in significant]
        if marked and (rep.family in design.PRIMARY_FAMILIES or rep.is_ours):
            axis.scatter(
                [m[0] for m in marked],
                [m[1] for m in marked],
                s=design.MS_SIGNIFICANT,
                facecolors="none",
                edgecolors=rep.colour,
                linewidths=0.85,
                zorder=rep.zorder + 1,
            )

    onset = _single_reference_onset(points, reference)
    if onset is not None and selected:
        axis.axvspan(
            onset - 0.5,
            max(p.n for p in selected) + 1,
            color="0.85",
            alpha=0.6,
            zorder=0,
            linewidth=0,
        )
    if degenerate is not None and any(p.representation == degenerate for p in selected):
        rep = design.BY_KEY.get(degenerate)
        name = rep.short if rep is not None else degenerate
        axis.annotate(
            f"{name} $\\equiv$ reference: $\\rho \\equiv 1$ by construction",
            xy=(0.985, 0.975),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=design.FS_TICK - 0.5,
            color="0.30",
        )

    axis.axhline(0.0, color=design.INK_RULE, linewidth=0.6, linestyle=":")
    title = f"{ref_label} --- exact at every size, so no bracket and no regime split"
    if onset:
        title += f";  shaded: $\\rho$ not separable from 0 ($n>{onset}$)"
    axis.set_title(title, fontsize=design.FS_TITLE - 0.6, pad=4, color="0.25")
    design.finish_axes(
        axis,
        xlabel="graph size $n$",
        ylabel=rf"Spearman $\rho$ (distance vs {ref_label}), within equal $n$",
    )
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
    axis.set_ylim(-0.75, 1.05)

    # Two rows, for the reason recorded at figure_one's legend: a one-row
    # seven-entry legend measures 6.83 in at a declared 4.72 in and drags the
    # tight bbox out with it. Measured after this change: 4.76 in.
    handles, labels = axis.get_legend_handles_labels()
    design.shared_legend_rows(
        fig,
        handles,
        labels,
        counts=design.balanced_rows(len(labels), LEGEND_ROWS),
        y=LEGEND_Y,
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.905, bottom=AXES_BOTTOM + 0.045)
    saved = [str(q) for q in design.save(fig, out)]
    plt.close(fig)
    return saved


def _draw_series(
    ax: Any,
    points: list[AggregatePoint],
    key: str,
    reference: str,
    significant: set[tuple[str, str, int]],
    *,
    label: str | None,
) -> None:
    """Draw one representation's series for one reference onto *ax*.

    Factored out of the panel builders so the combined figure cannot drift from
    the single-reference one in interval treatment, marker rule or draw order.

    Args:
        ax: Target axes.
        points: Aggregated points; filtered to *key* and *reference* here.
        key: Representation key.
        reference: Reference key.
        significant: ``(representation, reference, n)`` triples that passed the
            figure's Benjamini-Hochberg correction.
        label: Legend label, or ``None`` to draw the series without one. Only
            one panel of a combined figure may label a representation, or the
            shared legend carries it several times.
    """
    rep = design.BY_KEY.get(key)
    if rep is None:
        return
    series = sorted(
        (p for p in points if p.representation == key and p.reference == reference),
        key=lambda p: p.n,
    )
    if not series:
        return
    xs = [p.n for p in series]
    ys = [p.rho for p in series]
    style = design.line_kwargs(rep, reference if reference in design.REFERENCE_LINESTYLE else None)
    if label is not None:
        style["label"] = label

    if len(series) > DENSE_SERIES:
        ax.plot(xs, ys, **style)
        ax.fill_between(
            xs,
            [p.ci_lo for p in series],
            [p.ci_hi for p in series],
            color=rep.colour,
            alpha=design.ALPHA_BAND,
            linewidth=0,
            zorder=rep.zorder - 1,
        )
    else:
        ax.errorbar(
            xs,
            ys,
            yerr=[[p.rho - p.ci_lo for p in series], [p.ci_hi - p.rho for p in series]],
            elinewidth=0.55,
            capsize=1.4,
            **style,
        )

    marked = [(p.n, p.rho) for p in series if (key, reference, p.n) in significant]
    if marked and (rep.family in design.PRIMARY_FAMILIES or rep.is_ours):
        ax.scatter(
            [m[0] for m in marked],
            [m[1] for m in marked],
            s=design.MS_SIGNIFICANT,
            facecolors="none",
            edgecolors=rep.colour,
            linewidths=0.85,
            zorder=rep.zorder + 1,
        )


def _apply_struct_zoom(ax: Any, fraction: float) -> None:
    """Put a two-piece linear x scale on the structural panel.

    Both pieces are linear, so a slope is still a slope *within* a piece; only
    the ratio between the two pieces is manufactured. The map is continuous and
    strictly increasing at the knot, and extrapolates linearly beyond the
    limits, which matplotlib requires of a ``function`` scale.

    Args:
        ax: The structural axes.
        fraction: Share of the axis width given to ``n <= STRUCT_ZOOM_KNOT``.
            Must lie strictly inside ``(0, 1)``.

    Raises:
        ValueError: If *fraction* is not strictly between 0 and 1.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must lie in (0, 1), got {fraction}")
    lo, hi = STRUCT_X_LIMITS
    knot = STRUCT_ZOOM_KNOT
    slope_lo = fraction / (knot - lo)
    slope_hi = (1.0 - fraction) / (hi - knot)

    def forward(x: Any) -> Any:
        values = np.asarray(x, dtype=float)
        return np.where(
            values <= knot,
            (values - lo) * slope_lo,
            fraction + (values - knot) * slope_hi,
        )

    def inverse(y: Any) -> Any:
        values = np.asarray(y, dtype=float)
        return np.where(
            values <= fraction,
            lo + values / slope_lo,
            knot + (values - fraction) / slope_hi,
        )

    ax.set_xscale("function", functions=(forward, inverse))
    ax.set_xlim(lo, hi)
    # The break, drawn rather than captioned alone: it is also the node count
    # above which exact GED stops being computable, so it marks which half of
    # (b) a given part of (a) is to be read against.
    ax.axvline(
        knot,
        color=design.INK_SEPARATOR,
        linewidth=0.6,
        linestyle=(0, (2.5, 1.8)),
        zorder=1.5,
    )


def figure_one_combined(
    points: list[AggregatePoint],
    out: Path,
    *,
    reference: str = "wl",
    degenerate: str | None = "wl_subtree",
    width: float | None = None,
    emphasis: bool = False,
    struct_zoom: float | None = STRUCT_ZOOM_FRACTION,
) -> list[str]:
    """The structural reference beside graph edit distance, in one figure.

    **(a)** the structural reference, exact at every size, so one series per
    representation over the whole size range. **(b)** :func:`figure_one`
    unchanged --- exact GED at ``n <= EXACT_CEILING`` on the wide axes, the
    LB/UB bracket above it as small multiples, one per representation carrying
    bracket data plus a final panel overlaying every arm. A rule separates the
    two halves.

    The comparison the figure exists to make is that (a) and (b) hold the
    *reference* side of the correlation and nothing else: the representation
    distances behind both are the same cached T-04a matrices. A difference
    between them is a property of the yardstick.

    **The y axis is shared throughout** --- one scale, ticks on the leftmost
    axes of each half. The x axis is graph size ``n`` everywhere, but exact GED
    spans ten strata where the other references span sixty-two, so the panels
    are not on a common x scale. **That belongs in the caption**, and so does
    every other qualification: this function draws no titles, no in-axes notes
    and no significance textbox. Only the two panel letters.

    Args:
        points: Aggregated points, carrying *reference* and the GED references.
        out: Output path without an extension.
        reference: The structural reference drawn in (a).
        degenerate: A representation whose own distance **is** *reference*, so
            its rho is 1.0 by construction. It is still drawn --- hiding it
            would be a silent exclusion --- and named in the caption instead of
            on the figure.
        width: Render width in inches; defaults to the frozen 7.0 in constant.
            See :func:`figure_one_single_reference` for why a narrower value is
            not sufficient on its own.
        emphasis: Carry :data:`design.PRIMARY_FAMILIES` into the **overlay**
            bracket panel, so it foregrounds the same arms as (a), the exact
            panel and the information-content figure. Off, the overlay mutes
            everything that is not ours, which puts ``min_dfs`` and ``agm_cam``
            --- canonical codes the claim is scoped against --- behind the
            serializations they are being compared with. The per-arm small
            multiples are never muted either way: an arm is the subject of its
            own panel.
        struct_zoom: Share of panel (a)'s width given to the exact-GED window,
            or ``None`` for a linear axis. See :data:`STRUCT_ZOOM_FRACTION` for
            what the break buys and what it costs.

    Returns:
        Paths written.

    Raises:
        ValueError: If *points* carries no row for *reference*.
    """
    plt = _style()

    if not any(p.reference == reference for p in points):
        raise ValueError(
            f"no aggregated points for reference {reference!r}; "
            "the profile was probably built without it"
        )

    struct_reps = tuple(
        r
        for r in design.FIGURE_ORDER
        if any(p.representation == r and p.reference == reference for p in points)
    )
    exact_reps = tuple(
        r
        for r in design.FIGURE_ORDER
        if any(p.representation == r and p.reference == "exact" for p in points)
    )
    bracket_reps = _bracket_representations(points)

    # One correction over every point the figure draws, matching this module's
    # rule that the BH level is local to a figure. The count goes in the caption.
    drawn = [
        p
        for p in points
        if (p.reference == reference and p.representation in struct_reps)
        or (p.reference == "exact" and p.representation in exact_reps)
        or (p.reference in ("lb", "ub") and p.representation in bracket_reps)
    ]
    flags = benjamini_hochberg([p.p_value for p in drawn])
    significant = {(p.representation, p.reference, p.n) for p, f in zip(drawn, flags) if f}

    ncols = 2
    nrows = max(1, -(-(len(bracket_reps) + 1) // ncols))
    figure_width = design.text_width() if width is None else width
    fig = plt.figure(figsize=(figure_width, 1.15 * nrows + 1.05))

    # Outer split: (a) | rule | (b). The middle column is empty and exists only
    # to hold the gap the rule is drawn down.
    # The middle column was 0.26 and the gap it opened was about 8 mm, of which
    # the "(b)" letter took the right-hand third: (a), the rule and the letter
    # all crowded into one band. Widened so each of the three clearances is
    # legible on its own.
    # (a) was 1.62 against (b)'s 3.00. The zoom needs the width: ten labelled
    # strata in half of (a) only clear 6.5 pt once (a) reaches about 2.5 in.
    outer = fig.add_gridspec(1, 3, width_ratios=[2.15, 0.55, 2.70], wspace=0.0)
    ax_struct = fig.add_subplot(outer[0, 0])
    # 1.55 was enough for the five ticks MaxNLocator chose. Ten stepped by 1
    # need the width: at 1.55 the panel prints "10 11 12" as one glyph run.
    inner = outer[0, 2].subgridspec(
        nrows, 1 + ncols, width_ratios=[2.05, *([1.0] * ncols)], wspace=0.13, hspace=0.30
    )
    ax_exact = fig.add_subplot(inner[:, 0], sharey=ax_struct)

    for key in struct_reps:
        _draw_series(ax_struct, points, key, reference, significant, label=design.BY_KEY[key].short)
    for key in exact_reps:
        _draw_series(ax_exact, points, key, "exact", significant, label=None)

    # The bracket small multiples, reproducing figure_one's treatment: a grey
    # envelope of the whole field behind each panel, LB thinner than UB, and a
    # final panel overlaying every arm.
    envelope: dict[int, list[float]] = {}
    for p in points:
        if p.reference in ("lb", "ub"):
            envelope.setdefault(p.n, []).append(p.rho)
    span = sorted(envelope)

    panels: list[str | None] = [*bracket_reps, None]
    for index, key in enumerate(panels):
        ax = fig.add_subplot(inner[index // ncols, 1 + index % ncols], sharey=ax_struct)
        if span:
            ax.fill_between(
                span,
                [min(envelope[n]) for n in span],
                [max(envelope[n]) for n in span],
                color="0.55",
                alpha=0.20,
                linewidth=0,
                zorder=1,
            )
        for one in bracket_reps if key is None else (key,):
            rep = design.BY_KEY[one]
            for ref, scale in (("lb", 0.75), ("ub", 1.0)):
                sel = sorted(
                    (p for p in points if p.representation == one and p.reference == ref),
                    key=lambda p: p.n,
                )
                if not sel:
                    continue
                # Every family stays primary here so REFERENCE_LINESTYLE, not
                # SECONDARY_DASH, sets the dash: the LB/UB distinction is the
                # whole point of these panels and must survive the muting.
                style = design.line_kwargs(rep, ref, primary=frozenset(design.Family))
                style["marker"] = "None"
                style["linewidth"] = (design.LW_REFERENCE if rep.is_ours else 0.95) * scale
                lead = rep.family in design.PRIMARY_FAMILIES or rep.is_ours
                if key is None and not (lead if emphasis else rep.is_ours):
                    style["alpha"] = 0.55
                ax.plot([p.n for p in sel], [p.rho for p in sel], **style)
        ax.axhline(0.0, color=design.INK_RULE, linewidth=0.6, linestyle=":")
        ax.set_ylim(-0.75, 1.05)
        ax.grid(True, alpha=design.GRID_ALPHA, linewidth=design.GRID_LW)
        ax.tick_params(labelsize=design.FS_TICK - 1.4, labelleft=False)
        ax.xaxis.set_major_locator(FixedLocator(list(BRACKET_TICKS)))
        if index // ncols != nrows - 1:
            # Not set_xticklabels([]): against a FixedLocator that is a
            # label-count mismatch, and it silently strips the ticks too.
            ax.tick_params(labelbottom=False)

    for ax in (ax_struct, ax_exact):
        ax.axhline(0.0, color=design.INK_RULE, linewidth=0.6, linestyle=":")
        ax.set_ylim(-0.75, 1.05)
        design.finish_axes(ax)
    # (a) carries the bracket's tick values *and*, inside the zoom, a subset of
    # the exact panel's, so a reader can read one n off either half against it.
    # A locator would put a tick at 0, where there is no stratum: the smallest
    # is n = 3.
    if struct_zoom is not None:
        _apply_struct_zoom(ax_struct, struct_zoom)
        ax_struct.xaxis.set_major_locator(FixedLocator([*ZOOM_TICKS, *STRUCT_TAIL_TICKS]))
        # 15 gets a mark but no label, so the bracket panels' first tick still
        # has a position on (a) without colliding with 12.
        ax_struct.xaxis.set_minor_locator(FixedLocator([BRACKET_TICKS[0]]))
        ax_struct.tick_params(axis="x", which="minor", length=1.8)
    else:
        ax_struct.xaxis.set_major_locator(FixedLocator(list(BRACKET_TICKS)))
    ax_exact.xaxis.set_major_locator(MultipleLocator(EXACT_TICK_STEP))
    ax_exact.tick_params(axis="x", labelsize=design.FS_TICK - 0.8)
    # One y scale, one set of tick labels. Without this (b) reprints the same
    # ticks a centimetre from (a)'s and the axis reads as two axes.
    ax_exact.tick_params(labelleft=False)
    ax_struct.set_ylabel(r"Spearman $\rho$, within equal $n$", fontsize=design.FS_LABEL)

    design.panel_letter(ax_struct, "a")
    design.panel_letter(ax_exact, "b")

    # Matplotlib hands back handles in the order the artists were added, which
    # puts every band series before every errorbar series and so scrambles the
    # registry order. Re-order by FIGURE_ORDER, which also lands the four
    # foregrounded arms on the first row and the three muted ones on the second.
    by_label = dict(zip(*ax_struct.get_legend_handles_labels()[::-1]))
    ordered = [design.BY_KEY[k].short for k in struct_reps if design.BY_KEY[k].short in by_label]
    design.shared_legend_rows(
        fig,
        [by_label[name] for name in ordered],
        ordered,
        counts=design.balanced_rows(len(ordered), LEGEND_ROWS),
        y=LEGEND_Y_COMBINED,
        row_gap=LEGEND_ROW_GAP,
    )
    fig.subplots_adjust(left=0.078, right=0.992, top=0.935, bottom=AXES_BOTTOM + 0.045)
    fig.text(
        0.5,
        AXES_BOTTOM - 0.005,
        "Graph Size $n$",
        ha="center",
        va="top",
        fontsize=design.FS_LABEL,
    )

    # The rule, drawn after subplots_adjust so it spans the settled axes box.
    # Centred on the gap *minus* the panel letter, so the clearance (a)-to-rule
    # and the clearance rule-to-"(b)" come out equal rather than the letter
    # sitting on the rule.
    x_rule = 0.5 * (
        ax_struct.get_position().x1 + fig.axes[1].get_position().x0 - PANEL_LETTER_WIDTH
    )
    top = ax_struct.get_position().y1
    bottom = ax_struct.get_position().y0
    fig.add_artist(
        Line2D(
            [x_rule, x_rule],
            [bottom, top + 0.045],
            transform=fig.transFigure,
            color=design.INK_SEPARATOR,
            linewidth=0.8,
        )
    )

    saved = [str(q) for q in design.save(fig, out)]
    plt.close(fig)
    return saved


def figure_two(rows: list[dict[str, Any]], points: list[AggregatePoint], out: Path) -> list[str]:
    """Figure 2 --- one panel per representation, datasets broken out.

    Args:
        rows: Per-dataset stratum rows.
        points: Aggregated points, drawn on top and joined by lines.
        out: Output path without extension.

    Returns:
        Paths written.
    """
    plt = _style()
    reps = tuple(r for r in REPRESENTATION_ORDER if any(p.representation == r for p in points))
    colours = _colours(reps)
    datasets = sorted({r["dataset"] for r in rows})
    markers = {d: DATASET_MARKERS[i % len(DATASET_MARKERS)] for i, d in enumerate(datasets)}
    flags = benjamini_hochberg([p.p_value for p in points])
    significant = {(p.representation, p.reference, p.n) for p, f in zip(points, flags) if f}

    ncols = 2
    nrows = int(np.ceil(len(reps) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(design.text_width(), 2.5 * nrows),
        sharex=True,
        sharey=True,
    )
    flat = np.asarray(axes).reshape(-1)

    for ax, rep in zip(flat, reps):
        for row in (r for r in rows if r["representation"] == rep):
            ax.scatter(
                row["n"],
                row["rho"],
                marker=markers[row["dataset"]],
                s=13,
                color=colours[rep],
                alpha=0.32,
                linewidths=0,
                zorder=2,
                label=f"_{row['dataset']}",
            )
        for ref, ls in (("exact", "-"), ("lb", "--"), ("ub", "-")):
            sel = sorted(
                (p for p in points if p.representation == rep and p.reference == ref),
                key=lambda p: p.n,
            )
            if not sel:
                continue
            ax.errorbar(
                [p.n for p in sel],
                [p.rho for p in sel],
                yerr=[[p.rho - p.ci_lo for p in sel], [p.ci_hi - p.rho for p in sel]],
                color=colours[rep],
                linestyle=ls,
                marker="o",
                markersize=3.4,
                linewidth=1.4,
                elinewidth=0.6,
                capsize=1.5,
                zorder=4,
                alpha=1.0 if ref != "lb" else 0.7,
            )
            marked = [(p.n, p.rho) for p in sel if (rep, ref, p.n) in significant]
            if marked:
                ax.scatter(
                    [m[0] for m in marked],
                    [m[1] for m in marked],
                    s=52,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=0.8,
                    zorder=6,
                )
        ax.axhline(0.0, color="0.35", linewidth=0.6, linestyle=":")
        ax.axvline(EXACT_CEILING + 0.5, color="0.5", linewidth=0.7, linestyle="-.")
        ax.set_title(DISPLAY.get(rep, rep), fontsize=design.FS_TITLE)
        ax.grid(True, alpha=0.25, linewidth=0.4)
        ax.tick_params(labelsize=design.FS_TICK)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))

    for ax in flat[len(reps) :]:
        ax.axis("off")
    for ax in flat[max(0, len(reps) - ncols) : len(reps)]:
        ax.set_xlabel("Graph Size ($n$)", fontsize=design.FS_LABEL)
    for i in range(0, len(reps), ncols):
        flat[i].set_ylabel(r"Spearman $\rho$", fontsize=design.FS_LABEL)

    marker_handles = [
        plt.Line2D([], [], marker=markers[d], linestyle="none", color="0.35", markersize=4, label=d)
        for d in datasets
    ]
    fig.legend(
        handles=marker_handles,
        loc="upper center",
        ncol=min(len(datasets), 5),
        fontsize=design.FS_LEGEND,
        frameon=False,
        title="Dataset (faint markers); heavy line = aggregate",
        title_fontsize=design.FS_LEGEND,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Per-Dataset Spread Behind Each Aggregate Point"
        f"  (dash-dot: exact-GED ceiling at n = {EXACT_CEILING};"
        "  solid line = aggregate / UB, dashed = LB;  ○ = BH-significant)",
        fontsize=design.FS_TITLE,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.0, 1, 0.97))
    saved = [str(q) for q in design.save(fig, out)]
    plt.close(fig)
    return saved


def figure_three(rows: list[dict[str, Any]], out: Path) -> list[str]:
    """Figure 3 --- absolute distances against GED, with the bracket as a band.

    Left axis carries the representation's own distance (Levenshtein symbols, or
    the WL kernel distance); the right axis carries GED. Above the exact ceiling
    the LB and UB means bound a shaded region that **contains the true GED**.

    Args:
        rows: Per-dataset stratum rows.
        out: Output path without extension.

    Returns:
        Paths written.
    """
    plt = _style()
    reps = tuple(r for r in REPRESENTATION_ORDER if any(x["representation"] == r for x in rows))
    colours = _colours(reps)

    def weighted(values: list[tuple[float, int]]) -> float:
        total = sum(w for _, w in values)
        return sum(v * w for v, w in values) / total if total else float("nan")

    fig, ax = plt.subplots(figsize=(design.text_width(), 3.8))
    twin = ax.twinx()

    for rep in reps:
        by_n: dict[int, list[tuple[float, int]]] = {}
        for row in rows:
            if row["representation"] != rep or row["mean_distance"] is None:
                continue
            by_n.setdefault(int(row["n"]), []).append((row["mean_distance"], row["n_pairs"]))
        xs = sorted(by_n)
        if not xs:
            continue
        ax.plot(
            xs,
            [weighted(by_n[n]) for n in xs],
            color=colours[rep],
            marker="o",
            markersize=3.0,
            linewidth=1.3,
            label=DISPLAY.get(rep, rep),
            zorder=3,
        )

    ref_by_n: dict[str, dict[int, list[tuple[float, int]]]] = {"exact": {}, "lb": {}, "ub": {}}
    for row in rows:
        if row["representation"] != "isalgraph_pruned" or row["mean_reference"] is None:
            continue
        ref_by_n[row["reference"]].setdefault(int(row["n"]), []).append(
            (row["mean_reference"], row["n_pairs"])
        )

    exact_x = sorted(ref_by_n["exact"])
    if exact_x:
        twin.plot(
            exact_x,
            [weighted(ref_by_n["exact"][n]) for n in exact_x],
            color="0.15",
            linewidth=1.6,
            marker="s",
            markersize=3.2,
            label="exact GED",
            zorder=4,
        )
    band_x = sorted(set(ref_by_n["lb"]) & set(ref_by_n["ub"]))
    if band_x:
        lo = [weighted(ref_by_n["lb"][n]) for n in band_x]
        hi = [weighted(ref_by_n["ub"][n]) for n in band_x]
        twin.fill_between(
            band_x,
            lo,
            hi,
            color="0.15",
            alpha=0.16,
            linewidth=0,
            zorder=1,
            label="True GED lies in here",
        )
        twin.plot(band_x, lo, color="0.15", linewidth=0.9, linestyle="--", zorder=4)
        twin.plot(band_x, hi, color="0.15", linewidth=0.9, linestyle="-", zorder=4)

    ax.axvline(EXACT_CEILING + 0.5, color="0.5", linewidth=0.7, linestyle="-.")
    ax.set_xlabel("Graph Size ($n$)  (both graphs in the pair)", fontsize=design.FS_LABEL)
    ax.set_ylabel(
        "Mean Representation Distance  (symbols / kernel units)", fontsize=design.FS_LABEL
    )
    twin.set_ylabel("Mean GED  (unit cost model)", fontsize=design.FS_LABEL)
    ax.grid(True, alpha=0.25, linewidth=0.4)
    ax.tick_params(labelsize=design.FS_TICK)
    twin.tick_params(labelsize=design.FS_TICK)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=14))

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    fig.legend(
        h1 + h2,
        l1 + l2,
        fontsize=design.FS_LEGEND,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(len(l1 + l2), 4),
        frameon=False,
    )
    ax.set_title(
        "Absolute Scale: Representation Distance (left) Against GED (right).  "
        f"Above n = {EXACT_CEILING} the shaded band is the proven LB/UB bracket.",
        fontsize=design.FS_TITLE,
    )
    fig.tight_layout(rect=(0, 0.0, 1, 0.97))
    saved = [str(q) for q in design.save(fig, out)]
    plt.close(fig)
    return saved


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", type=Path, required=True, help="size_profile.json")
    ap.add_argument("--out-dir", type=Path, required=True, help="figure output directory")
    ap.add_argument(
        "--reference",
        default=None,
        help=(
            "draw the single-reference variant of figure 1 for this reference "
            "(e.g. 'wl') instead of the three GED figures. The reference carries "
            "no bracket, so each small multiple shows one exact series"
        ),
    )
    ap.add_argument(
        "--reference-label",
        default=None,
        help="human-readable name for --reference in the titles, e.g. 'WL kernel'",
    )
    ap.add_argument(
        "--stem",
        default=None,
        help="output basename; defaults to fig1_rho_vs_size_<reference>",
    )
    ap.add_argument(
        "--width",
        type=float,
        default=None,
        help=(
            "render width in inches; default is the frozen 7.0 in IEEE width. "
            "Pass 4.72 for the Pattern Recognition text block, so declared point "
            "sizes are the printed point sizes"
        ),
    )
    ap.add_argument(
        "--degenerate",
        default=None,
        help=(
            "a representation whose distance IS --reference, so its rho is 1.0 "
            "by construction; drawn but annotated (e.g. wl_subtree under wl)"
        ),
    )
    ap.add_argument(
        "--combined",
        action="store_true",
        help=(
            "draw --reference beside exact GED and the GED bracket in one "
            "three-panel figure with a shared y axis and one legend. Requires "
            "--reference; the profile must carry that reference AND exact/lb/ub"
        ),
    )
    ap.add_argument(
        "--emphasis",
        action="store_true",
        help=(
            "foreground design.PRIMARY_FAMILIES in the overlay bracket panel, "
            "matching the information-content figure. Without it that panel "
            "mutes every arm that is not ours, including the canonical codes"
        ),
    )
    ap.add_argument(
        "--struct-zoom",
        type=float,
        default=STRUCT_ZOOM_FRACTION,
        help=(
            "share of panel (a)'s width given to the exact-GED window n <= 12, "
            "so it can carry the exact panel's ticks. 0 restores a linear axis"
        ),
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

    if args.combined and args.reference is None:
        LOGGER.error("--combined needs --reference; there is nothing to place beside GED")
        return 2

    if args.combined:
        # Two passes, deliberately. The regime filter is what keeps an `lb` row
        # from being drawn where `exact` applies, and it drops every structural
        # row because a structural reference belongs to no GED regime. So the
        # GED side is loaded under the filter and the structural side without
        # it, and they are concatenated -- rather than loosening the filter,
        # which the three published GED figures also read.
        rows = load_rows(args.profile) + load_rows(args.profile, keep_reference=args.reference)
    else:
        rows = load_rows(args.profile, keep_reference=args.reference)
    if not rows:
        LOGGER.error("no usable rows in %s", args.profile)
        return 1
    points = aggregate(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.combined:
        stem = args.stem or f"fig1_rho_vs_size_{args.reference}_vs_ged"
        saved = figure_one_combined(
            points,
            args.out_dir / stem,
            reference=args.reference,
            degenerate=args.degenerate,
            width=args.width,
            emphasis=args.emphasis,
            struct_zoom=args.struct_zoom or None,
        )
        LOGGER.info("%s -> %s", stem, ", ".join(saved))
        LOGGER.info(
            "%d stratum rows, %d aggregate points, %s beside exact/lb/ub",
            len(rows),
            len(points),
            args.reference,
        )
        return 0

    if args.reference is not None:
        stem = args.stem or f"fig1_rho_vs_size_{args.reference}"
        label = args.reference_label or args.reference
        saved = figure_one_single_reference(
            points,
            args.out_dir / stem,
            reference=args.reference,
            ref_label=label,
            degenerate=args.degenerate,
            width=args.width,
        )
        LOGGER.info("%s -> %s", stem, ", ".join(saved))
        LOGGER.info(
            "%d stratum rows, %d aggregate points, reference=%s",
            len(rows),
            len(points),
            args.reference,
        )
        return 0

    for name, saved in (
        ("fig1_rho_vs_size", figure_one(points, args.out_dir / "fig1_rho_vs_size")),
        (
            "fig2_rho_by_representation",
            figure_two(rows, points, args.out_dir / "fig2_rho_by_representation"),
        ),
        ("fig3_absolute_scale", figure_three(rows, args.out_dir / "fig3_absolute_scale")),
    ):
        LOGGER.info("%s -> %s", name, ", ".join(saved))
    LOGGER.info("%d stratum rows, %d aggregate points", len(rows), len(points))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
