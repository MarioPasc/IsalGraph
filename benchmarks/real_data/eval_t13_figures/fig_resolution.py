"""Partition resolution against the invariance ceiling (Proposition 1).

``T-13-design.md`` 2.1 proves that for any node invariant ``kappa`` the
partition ``V/kappa`` is a **coarsening** of the orbit partition
``V/Aut(G)``: an automorphism is an isomorphism ``G -> G``, so
``kappa(v) = kappa(alpha(v))`` for every ``alpha`` and every ``v``.  The
immediate consequence, and the only thing this figure draws, is

    ``#classes(kappa) <= #orbits``

for every graph and every invariant.  The line ``y = x`` is therefore a
**ceiling**, not a trend line, and nothing may sit above it.  A point above it
does not indicate a noisy measurement; it refutes Proposition 1 or the code
that computed one of the two numbers.  :func:`figure` raises rather than
drawing it, because a figure that shows an impossible point is worse than no
figure -- it invites a reader to explain the outlier instead of doubting the
pipeline.

The reading the paper takes from it (``T-13-design.md`` 1.3, Corollary 3): 1-WL
already sits *on* the ceiling and the incumbent triplet key just below it, so
refining the pruning key cannot recover the remaining branching.  The fix is
explicit automorphism detection -- what nauty, bliss and Traces implement --
and not a finer invariant.  That is deliverable 5's justification.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from benchmarks.real_data.eval_t13_figures import data, design

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from matplotlib.axes import Axes

LOGGER: Final = logging.getLogger(__name__)

#: The two invariants drawn, with the registry key whose colour each borrows.
#: 1-WL borrows ``wl_subtree``'s and the incumbent pruning key borrows the
#: shipped arm's, so a reader who has seen the cost-law figure already knows
#: which curve belongs to which algorithm.
SERIES: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("n_wl_classes", "wl_subtree", "1-WL colour classes", "o"),
    ("n_triplet_classes", "isalgraph_pruned", "triplet pruning key", "s"),
)

#: Output stem, without an extension.
STEM: Final[str] = "fig_t13_resolution"

#: Panel size, inches.
PANEL_HEIGHT_IN: Final[float] = 2.7


class CeilingViolationError(ValueError):
    """A graph resolves more classes than it has orbits.

    Impossible under Proposition 1.  Either the proposition is false, or one of
    ``n_orbits`` / ``n_wl_classes`` / ``n_triplet_classes`` was computed wrong.
    Both are findings; neither is something to plot.
    """


def check_ceiling(rows: Sequence[data.GraphResolution]) -> None:
    """Raise unless every graph respects ``#classes <= #orbits``.

    Args:
        rows: Resolution records.

    Raises:
        CeilingViolationError: On the first graph that violates the ceiling,
            naming every offending field so the fault can be localised without
            re-running.
    """
    offenders: list[str] = []
    for row in rows:
        over = [
            f"{field}={getattr(row, field)}"
            for field, _, _, _ in SERIES
            if getattr(row, field) > row.n_orbits
        ]
        if over:
            offenders.append(f"{row.graph_key} (n={row.n}, n_orbits={row.n_orbits}): {over}")
    if offenders:
        raise CeilingViolationError(
            "invariance ceiling violated on "
            f"{len(offenders)} graph(s): a node invariant cannot resolve more classes "
            "than the graph has automorphism orbits (Proposition 1). "
            + "; ".join(offenders[:5])
            + ("; ..." if len(offenders) > 5 else "")
        )


def _ecdf(values: Sequence[int]) -> tuple[list[float], list[float]]:
    """Return the empirical CDF of *values* as step coordinates."""
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    return ordered, [(i + 1) / n for i in range(n)]


def _draw_scatter(ax: Axes, rows: Sequence[data.GraphResolution]) -> dict[str, Any]:
    """Draw the ceiling panel and return the on-ceiling fractions."""
    ceiling = max((r.n_orbits for r in rows), default=1)
    ax.plot(
        [0, ceiling],
        [0, ceiling],
        color=design.INK_CEILING,
        linewidth=1.0,
        linestyle="--",
        zorder=1,
        label=r"invariance ceiling $y = x$",
    )
    ax.annotate(
        "invariance ceiling\n(orbit partition)",
        xy=(ceiling * 0.72, ceiling * 0.72),
        xytext=(ceiling * 0.42, ceiling * 0.94),
        fontsize=design.FS_ANNOT,
        color=design.INK_RULE,
        ha="center",
        arrowprops={"arrowstyle": "->", "color": design.INK_RULE, "linewidth": 0.6},
    )
    out: dict[str, Any] = {}
    for field, colour_key, label, marker in SERIES:
        rep = design.BY_KEY[colour_key]
        xs = [r.n_orbits for r in rows]
        ys = [getattr(r, field) for r in rows]
        ax.plot(
            xs,
            ys,
            linestyle="none",
            marker=marker,
            markersize=design.MS_POINT,
            markerfacecolor=rep.colour,
            markeredgecolor="black",
            markeredgewidth=0.25,
            alpha=0.55,
            zorder=4,
            label=label,
        )
        on_ceiling = sum(1 for r in rows if getattr(r, field) == r.n_orbits)
        out[field] = {
            "on_ceiling": on_ceiling,
            "n": len(rows),
            "fraction": on_ceiling / len(rows) if rows else None,
        }
    design.finish_axes(
        ax,
        xlabel=r"$|V/\mathrm{Aut}(G)|$ (automorphism orbits)",
        ylabel="classes resolved by the invariant",
    )
    ax.set_title("resolution against the ceiling", fontsize=design.FS_TITLE)
    return out


def _draw_deficit(ax: Axes, rows: Sequence[data.GraphResolution]) -> dict[str, Any]:
    """Draw the deficit ECDF and return the median deficits."""
    out: dict[str, Any] = {}
    for field, colour_key, label, _ in SERIES:
        rep = design.BY_KEY[colour_key]
        deficits = [r.n_orbits - getattr(r, field) for r in rows]
        xs, ys = _ecdf(deficits)
        ax.step(
            xs,
            ys,
            where="post",
            color=rep.colour,
            linewidth=design.LW_FOCUS if colour_key == "isalgraph_pruned" else 1.2,
            label=label,
        )
        out[field] = {
            "max_deficit": max(deficits) if deficits else None,
            "zero_deficit": sum(1 for d in deficits if d == 0),
        }
    ax.set_ylim(0.0, 1.02)
    design.finish_axes(
        ax,
        xlabel=r"deficit $|V/\mathrm{Aut}| - |V/\kappa|$",
        ylabel="fraction of graphs",
    )
    ax.set_title("how far below the ceiling", fontsize=design.FS_TITLE)
    return out


def figure(rows: Sequence[data.GraphResolution], out: Path) -> tuple[list[Path], dict[str, Any]]:
    """Draw the resolution figure.

    Args:
        rows: Resolution records, one per graph.
        out: Output path without an extension.

    Returns:
        ``(paths_written, summary)``.

    Raises:
        ValueError: If *rows* is empty.
        CeilingViolationError: If any point sits above ``y = x``.  Checked
            **before** anything is drawn.
    """
    if not rows:
        raise ValueError("no graph carries a resolution record; nothing to draw")
    check_ceiling(rows)

    plt = design.style()
    fig, axes = plt.subplots(1, 2, figsize=(design.text_width(), PANEL_HEIGHT_IN), squeeze=False)
    scatter = _draw_scatter(axes[0][0], rows)
    deficit = _draw_deficit(axes[0][1], rows)
    design.panel_letter(axes[0][0], "a")
    design.panel_letter(axes[0][1], "b")

    equals = {
        "wl_equals_orbits": sum(1 for r in rows if r.wl_equals_orbits is True),
        "triplet_equals_orbits": sum(1 for r in rows if r.triplet_equals_orbits is True),
    }
    design.note_box(
        axes[0][0],
        f"$N={len(rows)}$ graphs\n"
        f"1-WL on ceiling: {scatter['n_wl_classes']['on_ceiling']}/{len(rows)}\n"
        f"triplet on ceiling: {scatter['n_triplet_classes']['on_ceiling']}/{len(rows)}",
        loc="lower right",
    )

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout()
    design.shared_legend(fig, handles, labels, ncol=3, y=0.0)
    saved = design.save(fig, out)
    plt.close(fig)
    LOGGER.info("%s -> %s", out.name, ", ".join(str(p) for p in saved))
    return saved, {"n_graphs": len(rows), "scatter": scatter, "deficit": deficit, **equals}


def caption(summary: dict[str, Any]) -> str:
    """Return the LaTeX caption, with every number read from *summary*.

    Args:
        summary: The mapping :func:`figure` returned.

    Returns:
        A ``\\caption{...}`` body, without the surrounding macro.
    """
    n = summary["n_graphs"]
    wl = summary["scatter"]["n_wl_classes"]["on_ceiling"]
    tri = summary["scatter"]["n_triplet_classes"]["on_ceiling"]
    return (
        "(a) Classes resolved by each node invariant against the number of automorphism "
        "orbits, over "
        f"{n} graphs. Proposition 1 makes $y = x$ a ceiling: a node invariant is constant "
        "on every orbit, so its partition is a coarsening of the orbit partition and no "
        "point can lie above the line. "
        f"1-WL attains the ceiling on {wl} of {n} graphs and the incumbent triplet key on "
        f"{tri} of {n}. "
        "(b) The deficit below the ceiling. Because the incumbent key is already at or "
        "near the floor of what any invariant can achieve, the residual branching is "
        "irreducible without explicit automorphism detection (Corollary 3), which is why "
        "the remedy is nauty-style individualisation--refinement and not a finer key."
    )


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
        LOGGER.info("--counters is not read by fig_resolution; ignoring %s", args.counters)
    records = data.load(args.records)
    rows = data.resolutions(records)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / STEM
    _, summary = figure(rows, stem)
    (args.out_dir / f"{STEM}.caption.tex").write_text(caption(summary) + "\n")
    LOGGER.info("%d graph(s) drawn", summary["n_graphs"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
