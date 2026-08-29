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
from matplotlib.ticker import MaxNLocator
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

#: Above this many points in one series, draw the interval as a band rather
#: than as error bars: a picket fence of caps hides the trend it qualifies.
DENSE_SERIES: Final[int] = 20

#: Draw order and display names now come from the registry. They used to be
#: duplicated here, and the colour was assigned by walking a palette over the
#: representations *present in this call* -- so a representation's colour
#: depended on which others had landed, and two figures of the same campaign
#: could give one backend two colours.
REPRESENTATION_ORDER: Final[tuple[str, ...]] = design.ORDER

DATASET_MARKERS: Final[tuple[str, ...]] = ("o", "s", "^", "v", "D", "P", "X", "*", "<", ">")

DISPLAY: Final[dict[str, str]] = {r.key: r.short for r in design.REPRESENTATIONS}


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
    return tuple(r for r in design.ORDER if r in have)


def _undetermined_onset(points: list[AggregatePoint]) -> int | None:
    """Return the node count beyond which no bracket interval excludes zero.

    Args:
        points: Aggregated points.

    Returns:
        The onset node count, or ``None`` when every size still resolves.
    """
    bracket = [p for p in points if p.reference != "exact"]
    resolved = {p.n for p in bracket if not p.ci_lo <= 0.0 <= p.ci_hi}
    covered = sorted({p.n for p in bracket if p.ci_lo <= 0.0 <= p.ci_hi})
    return next((n for n in covered if not any(m >= n for m in resolved)), None)


def figure_one(points: list[AggregatePoint], out: Path) -> list[str]:
    """Figure 1 --- the within-`n` collapse, per regime and per representation.

    **Left, exact GED at ``n <= 12``.** Every representation, family-emphasised:
    canonical codes solid at full weight, serialisations dashed and
    half-transparent. Ground truth exists here and the head-to-head resolves, so
    the per-representation detail is the content.

    **Right, the bracket above ``n = 12``, as small multiples.** Fourteen series
    on one axes was a texture rather than a figure, and collapsing them into a
    single envelope answered that by discarding the per-representation detail.
    The grid keeps both: one small panel per representation carrying bracket
    data, each showing its own two bounds against the grey envelope of the whole
    field, plus a final panel overlaying every arm. The grid is sized so its top
    and bottom rows align with the exact-GED panel, so the two regimes read as
    one figure.

    Args:
        points: Aggregated points.
        out: Output path without an extension.

    Returns:
        Paths written.
    """
    plt = _style()

    exact_reps = tuple(
        r
        for r in design.ORDER
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
    left.set_title(f"exact GED  ($n \\leq {EXACT_CEILING}$)", fontsize=design.FS_TITLE, pad=3)
    design.finish_axes(
        left,
        xlabel="graph size $n$",
        ylabel=r"Spearman $\rho$ (distance vs GED), within equal $n$",
    )
    left.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    left.set_ylim(-0.75, 1.05)

    envelope: dict[int, list[float]] = {}
    for p in points:
        if p.reference != "exact":
            envelope.setdefault(p.n, []).append(p.rho)
    span = sorted(envelope)
    onset = _undetermined_onset(points)

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
        if onset is not None and span:
            ax.axvspan(onset - 0.5, max(span) + 1, color="0.85", alpha=0.6, zorder=0, linewidth=0)
        ax.axhline(0.0, color=design.INK_RULE, linewidth=0.6, linestyle=":")
        ax.set_ylim(-0.75, 1.05)
        ax.set_title(
            "every arm" if key is None else design.BY_KEY[key].short,
            fontsize=design.FS_TITLE - 0.7,
            pad=2,
            color="0.15" if key is None else design.BY_KEY[key].colour,
        )
        ax.grid(True, alpha=design.GRID_ALPHA, linewidth=design.GRID_LW)
        ax.tick_params(labelsize=design.FS_TICK - 0.8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        if index // ncols != nrows - 1:
            ax.set_xticklabels([])
        if index % ncols != 0:
            ax.set_yticklabels([])
        if index == len(panels) - 1:
            ax.set_xlabel("graph size $n$", fontsize=design.FS_LABEL - 0.5)

    header = f"GED bracket  ($n > {EXACT_CEILING}$)\ndashed LB, solid UB;  grey: the whole field"
    if onset:
        header += f";  shaded: $\\rho$ not separable from 0 ($n>{onset}$)"
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

    handles, labels = left.get_legend_handles_labels()
    design.shared_legend(fig, handles, labels, ncol=7, y=LEGEND_Y)
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
            box is the CONTENT box: at 4.72 in the seven-column legend and the
            title overflow and the tight box expands back to about 7 in, with
            nothing in the output to say so. A genuine narrow render also needs
            a narrower legend and a shorter title, which is a different figure.
            Measured, not assumed: both renders came back 7.03 in wide.

    Returns:
        Paths written.
    """
    plt = _style()

    selected = [p for p in points if p.reference == reference]
    present = tuple(r for r in design.ORDER if any(p.representation == r for p in selected))

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

    handles, labels = axis.get_legend_handles_labels()
    design.shared_legend(fig, handles, labels, ncol=7, y=LEGEND_Y)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.905, bottom=AXES_BOTTOM + 0.045)
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
        ax.set_xlabel("graph size $n$", fontsize=design.FS_LABEL)
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
        title="dataset (faint markers); heavy line = aggregate",
        title_fontsize=design.FS_LEGEND,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Per-dataset spread behind each aggregate point"
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
            label="true GED lies in here",
        )
        twin.plot(band_x, lo, color="0.15", linewidth=0.9, linestyle="--", zorder=4)
        twin.plot(band_x, hi, color="0.15", linewidth=0.9, linestyle="-", zorder=4)

    ax.axvline(EXACT_CEILING + 0.5, color="0.5", linewidth=0.7, linestyle="-.")
    ax.set_xlabel("graph size $n$  (both graphs in the pair)", fontsize=design.FS_LABEL)
    ax.set_ylabel(
        "mean representation distance  (symbols / kernel units)", fontsize=design.FS_LABEL
    )
    twin.set_ylabel("mean GED  (unit cost model)", fontsize=design.FS_LABEL)
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
        "Absolute scale: representation distance (left) against GED (right).  "
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

    rows = load_rows(args.profile, keep_reference=args.reference)
    if not rows:
        LOGGER.error("no usable rows in %s", args.profile)
        return 1
    points = aggregate(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)

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
