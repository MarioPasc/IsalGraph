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

LOGGER: Final = logging.getLogger(__name__)

#: Above this node count exact GED stops being computable and the bracket takes
#: over. Matches the registry scope guard measured in T-06-design 15.1.
EXACT_CEILING: Final[int] = 12

FDR_Q: Final[float] = 0.05

#: Above this many points in one series, draw the interval as a band rather
#: than as error bars: a picket fence of caps hides the trend it qualifies.
DENSE_SERIES: Final[int] = 20

#: Drawn in this order so the reference arm is never hidden behind a comparator.
REPRESENTATION_ORDER: Final[tuple[str, ...]] = (
    "isalgraph_pruned",
    "isalgraph_canonical",
    "wl_subtree",
    "min_dfs",
    "agm_cam",
    "nauty_graph6",
    "sparse6_nauty",
)

DATASET_MARKERS: Final[tuple[str, ...]] = ("o", "s", "^", "v", "D", "P", "X", "*", "<", ">")

DISPLAY: Final[dict[str, str]] = {
    "isalgraph_pruned": "IsalGraph (pruned)",
    "isalgraph_canonical": "IsalGraph (canonical)",
    "wl_subtree": "WL subtree (kernel)",
    "min_dfs": "gSpan min-DFS",
    "agm_cam": "AGM CAM",
    "nauty_graph6": "nauty graph6",
    "sparse6_nauty": "nauty sparse6",
}


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


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Load usable stratum rows, dropping degenerate ones.

    Args:
        path: ``size_profile.json``.

    Returns:
        Rows with a defined rho, restricted to the regime that applies at their
        node count.
    """
    payload = json.loads(path.read_text())
    out: list[dict[str, Any]] = []
    for row in payload["rows"]:
        if row["rho"] is None:
            continue
        if row["reference"] not in _regime(int(row["n"])):
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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from benchmarks import plotting_styles

    plotting_styles.apply_ieee_style()
    return plt


def _colours(names: tuple[str, ...]) -> dict[str, Any]:
    """Assign a stable colour per representation.

    Args:
        names: Representation names in draw order.

    Returns:
        Mapping from representation to colour.
    """
    from benchmarks import plotting_styles

    palette = list(plotting_styles.PAUL_TOL_MUTED)
    return {name: palette[i % len(palette)] for i, name in enumerate(names)}


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


def figure_one(points: list[AggregatePoint], out: Path) -> list[str]:
    """Figure 1 --- rho against graph size, aggregated over datasets.

    Args:
        points: Aggregated points.
        out: Output path without extension.

    Returns:
        Paths written.
    """
    plt = _style()
    from benchmarks import plotting_styles

    reps = tuple(r for r in REPRESENTATION_ORDER if any(p.representation == r for p in points))
    colours = _colours(reps)
    flags = benjamini_hochberg([p.p_value for p in points])
    significant = {(p.representation, p.reference, p.n) for p, f in zip(points, flags) if f}

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(plotting_styles.IEEE_TEXT_WIDTH_INCHES, 3.6),
        sharey=True,
        gridspec_kw={"width_ratios": [1, 1.35], "wspace": 0.06},
    )
    panels = (("exact",), ("lb", "ub"))
    titles = (f"exact GED  (n ≤ {EXACT_CEILING})", f"GED bracket  (n > {EXACT_CEILING})")
    styles = {"exact": "-", "lb": "--", "ub": "-"}

    for ax, refs, title in zip(axes, panels, titles):
        for rep in reps:
            for ref in refs:
                sel = sorted(
                    (p for p in points if p.representation == rep and p.reference == ref),
                    key=lambda p: p.n,
                )
                if not sel:
                    continue
                xs = [p.n for p in sel]
                ys = [p.rho for p in sel]
                lo = [p.rho - p.ci_lo for p in sel]
                hi = [p.ci_hi - p.rho for p in sel]
                label = f"{DISPLAY.get(rep, rep)}" + (
                    "" if ref == "exact" else f" · {ref.upper()}"
                )
                alpha = 0.9 if ref != "lb" else 0.65
                if len(xs) > DENSE_SERIES:
                    # Error bars on a dense series produce a picket fence that
                    # hides the trend they exist to qualify. A translucent band
                    # carries the same interval and stays legible.
                    ax.plot(
                        xs, ys, color=colours[rep], linestyle=styles[ref],
                        marker="o", markersize=2.0, linewidth=1.0,
                        alpha=alpha, label=label,
                    )
                    ax.fill_between(
                        xs,
                        [p.ci_lo for p in sel],
                        [p.ci_hi for p in sel],
                        color=colours[rep], alpha=0.10, linewidth=0,
                    )
                else:
                    ax.errorbar(
                        xs, ys, yerr=[lo, hi],
                        color=colours[rep], linestyle=styles[ref],
                        marker="o", markersize=3.2, linewidth=1.2,
                        elinewidth=0.6, capsize=1.5, alpha=alpha, label=label,
                    )
                marked = [(p.n, p.rho) for p in sel if (rep, ref, p.n) in significant]
                if marked:
                    ax.scatter(
                        [m[0] for m in marked],
                        [m[1] for m in marked],
                        s=52,
                        facecolors="none",
                        edgecolors=colours[rep],
                        linewidths=0.9,
                        zorder=5,
                    )
        # Test emptiness BEFORE axhline, which would otherwise populate
        # ax.lines and mask a genuinely empty regime.
        drew = bool(ax.lines or ax.collections)
        ax.axhline(0.0, color="0.35", linewidth=0.6, linestyle=":")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("graph size $n$", fontsize=7.5)
        ax.grid(True, alpha=0.25, linewidth=0.4)
        ax.tick_params(labelsize=6.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
        if not drew:
            ax.text(
                0.5,
                0.5,
                "no data in this regime",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=7.5,
                color="0.45",
                style="italic",
            )
            ax.set_xticks([])
            ax.set_xlabel("")

    axes[0].set_ylabel(r"Spearman $\rho$ (distance vs GED)", fontsize=7.5)
    axes[0].text(
        0.03,
        0.03,
        _significance_note(len(points)),
        transform=axes[0].transAxes,
        fontsize=5.2,
        va="bottom",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.6",
            "linewidth": 0.5,
            "alpha": 0.94,
        },
    )
    # Harvest from whichever panel actually drew, so the legend never strands
    # itself in an empty regime, and place it OUTSIDE the axes along the bottom.
    seen: dict[str, Any] = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(label.split(" · ")[0], handle)
    if seen:
        fig.legend(
            list(seen.values()),
            list(seen.keys()),
            fontsize=6.2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.03),
            ncol=min(len(seen), 4),
            frameon=False,
        )
    fig.suptitle(
        r"Within-$n$ correlation with GED: the size channel is removed by construction",
        fontsize=9,
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0.0, 1, 0.95))
    saved = plotting_styles.save_figure(fig, str(out))
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
    from benchmarks import plotting_styles

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
        figsize=(plotting_styles.IEEE_TEXT_WIDTH_INCHES, 2.5 * nrows),
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
        ax.set_title(DISPLAY.get(rep, rep), fontsize=8)
        ax.grid(True, alpha=0.25, linewidth=0.4)
        ax.tick_params(labelsize=6.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))

    for ax in flat[len(reps) :]:
        ax.axis("off")
    for ax in flat[max(0, len(reps) - ncols) : len(reps)]:
        ax.set_xlabel("graph size $n$", fontsize=7.5)
    for i in range(0, len(reps), ncols):
        flat[i].set_ylabel(r"Spearman $\rho$", fontsize=7.5)

    marker_handles = [
        plt.Line2D([], [], marker=markers[d], linestyle="none", color="0.35", markersize=4, label=d)
        for d in datasets
    ]
    fig.legend(
        handles=marker_handles,
        loc="upper center",
        ncol=min(len(datasets), 5),
        fontsize=6.0,
        frameon=False,
        title="dataset (faint markers); heavy line = aggregate",
        title_fontsize=6.2,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Per-dataset spread behind each aggregate point"
        f"  (dash-dot: exact-GED ceiling at n = {EXACT_CEILING};"
        "  solid line = aggregate / UB, dashed = LB;  ○ = BH-significant)",
        fontsize=8,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.0, 1, 0.97))
    saved = plotting_styles.save_figure(fig, str(out))
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
    from benchmarks import plotting_styles

    reps = tuple(r for r in REPRESENTATION_ORDER if any(x["representation"] == r for x in rows))
    colours = _colours(reps)

    def weighted(values: list[tuple[float, int]]) -> float:
        total = sum(w for _, w in values)
        return sum(v * w for v, w in values) / total if total else float("nan")

    fig, ax = plt.subplots(figsize=(plotting_styles.IEEE_TEXT_WIDTH_INCHES, 3.8))
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
    ax.set_xlabel("graph size $n$  (both graphs in the pair)", fontsize=7.5)
    ax.set_ylabel("mean representation distance  (symbols / kernel units)", fontsize=7.5)
    twin.set_ylabel("mean GED  (unit cost model)", fontsize=7.5)
    ax.grid(True, alpha=0.25, linewidth=0.4)
    ax.tick_params(labelsize=6.5)
    twin.tick_params(labelsize=6.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=14))

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    fig.legend(
        h1 + h2,
        l1 + l2,
        fontsize=6.2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(len(l1 + l2), 4),
        frameon=False,
    )
    ax.set_title(
        "Absolute scale: representation distance (left) against GED (right).  "
        f"Above n = {EXACT_CEILING} the shaded band is the proven LB/UB bracket.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.0, 1, 0.97))
    saved = plotting_styles.save_figure(fig, str(out))
    plt.close(fig)
    return saved


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", type=Path, required=True, help="size_profile.json")
    ap.add_argument("--out-dir", type=Path, required=True, help="figure output directory")
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

    rows = load_rows(args.profile)
    if not rows:
        LOGGER.error("no usable rows in %s", args.profile)
        return 1
    points = aggregate(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)

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
