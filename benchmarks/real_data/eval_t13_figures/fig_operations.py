"""The four costed operations against ``n``, with the derived bounds overlaid.

Supplementary figure.  ``T-13-design.md`` 2.1 derives a per-encode bound for
each of the four operations the canonical search actually pays for, and the
``t13c.1`` counter records measure what each one realises.  Drawing them
together is what separates leg (i) of the three-way separation -- the
complexity of the algorithm *as defined* -- from leg (iii), the runtime this
implementation happens to reach on this cohort.

+---------------------+--------------------------------+
| operation           | per-encode bound (2.1)         |
+=====================+================================+
| pair scanning       | ``O(m n^2)``                   |
| pointer walking     | ``O(m n^3)`` -- the dominant   |
| neighbour checks    | ``O(m Delta)``                 |
| backtracking leaves | ``L <= n Delta^{n-1}``         |
+---------------------+--------------------------------+

**Two honest substitutions, both stated on the figure.**

1. ``Delta`` is *not* carried by the ``t13c.1`` record -- it has ``n``, ``m``
   and the counts, and nothing else.  Rather than pull the maximum degree from
   a different file and risk pairing it with the wrong graph, the two
   ``Delta``-dependent bounds are drawn at ``Delta = n - 1``, which is
   ``Delta``'s own worst case on a simple graph.  They are therefore looser
   than 2.1's bound, never tighter, and are labelled as such.
2. 2.1 bounds the **leaves** of the search, while ``backtrack_nodes`` counts
   recursion frames entered, of which there are more.  So the backtracking
   panel draws ``search_leaves`` beside ``backtrack_nodes`` and attaches the
   bound to the series it actually bounds.

``backtrack_nodes`` and ``search_leaves`` are ``0`` for a greedy encode by
construction (``instrumented.OperationCounts``), and a logarithmic axis cannot
show zero, so the greedy series are absent from that panel rather than drawn at
a fabricated floor.
"""

from __future__ import annotations

import argparse
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from benchmarks.real_data.eval_t13_figures import data, design

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes

LOGGER: Final = logging.getLogger(__name__)

#: Output stem, without an extension.
STEM: Final[str] = "fig_t13_operations"

#: Panel size, inches.
PANEL_HEIGHT_IN: Final[float] = 2.35


@dataclass(frozen=True, slots=True)
class EncoderStyle:
    """How one ``t13c.1`` encoder is drawn.

    The counter CLI prices *encoders*, which are not the same objects as the
    registered representations: ``greedy_single`` has no registry key at all.
    Colours are borrowed from the registry entry that runs the same code, so a
    reader who has seen the cost-law figure recognises the curve; the borrowing
    is explicit here rather than implied by a palette walk.

    Attributes:
        encoder: The ``t13c.1`` encoder name.
        colour_key: The registry key whose colour it borrows.
        short: Legend name.
        marker: Matplotlib marker.
        linestyle: Dash pattern.
    """

    encoder: str
    colour_key: str
    short: str
    marker: str
    linestyle: Any


#: Every encoder ``counters.ENCODERS`` can emit, in draw order.
ENCODERS: Final[tuple[EncoderStyle, ...]] = (
    EncoderStyle("canonical", "isalgraph_canonical", "canonical (exhaustive)", "o", "-"),
    EncoderStyle("pruned", "isalgraph_pruned", "pruned canonical", "s", "-"),
    EncoderStyle("greedy_min", "isalgraph_greedy", "greedy-min", "h", (0, (4, 1.4))),
    EncoderStyle("greedy_single", "isalgraph_greedy", "greedy (one start)", "x", (0, (1.6, 1.4))),
)

#: Lookup by encoder name.
ENCODER_BY_NAME: Final[dict[str, EncoderStyle]] = {e.encoder: e for e in ENCODERS}


@dataclass(frozen=True, slots=True)
class Panel:
    """One costed operation and the 2.1 bound drawn over it.

    Attributes:
        field: Primary ``t13c.1`` count.
        extra: Secondary count drawn beside it, or ``None``.
        title: Panel title.
        ylabel: Y-axis label.
        bound: Key passed to :func:`_bound_value`.
        bound_label: Legend label for the bound curve.
    """

    field: str
    extra: str | None
    title: str
    ylabel: str
    bound: str
    bound_label: str


#: The four panels, in the order 2.1 tabulates the operations.
PANELS: Final[tuple[Panel, ...]] = (
    Panel(
        field="pair_trials",
        extra=None,
        title="pair scanning",
        ylabel=r"pair trials $\sum_f D_f$",
        bound="mn2",
        bound_label=r"bound $mn^{2}$",
    ),
    Panel(
        field="pointer_steps",
        extra=None,
        title="pointer walking (dominant term)",
        ylabel="unit CDLL moves",
        bound="mn3",
        bound_label=r"bound $mn^{3}$",
    ),
    Panel(
        field="neighbour_checks",
        extra=None,
        title="neighbour checks",
        ylabel="adjacency tests",
        bound="m_delta",
        bound_label=r"bound $m\Delta$, $\Delta = n-1$",
    ),
    Panel(
        field="backtrack_nodes",
        extra="search_leaves",
        title="backtracking",
        ylabel="recursion frames / leaves",
        bound="leaves",
        bound_label=r"leaf bound $n\Delta^{\,n-1}$, $\Delta = n-1$",
    ),
)


class OperationsFigureError(ValueError):
    """The counter records cannot support the operations figure."""


def _bound_value(kind: str, n: int, m: float) -> float:
    """Return the 2.1 bound of *kind* at ``(n, m)``.

    Args:
        kind: One of ``mn2``, ``mn3``, ``m_delta``, ``leaves``.
        n: Node count.
        m: Edge count, the cohort median at this *n*.

    Returns:
        The bound.

    Raises:
        OperationsFigureError: On an unknown *kind*.
    """
    if kind == "mn2":
        return m * n**2
    if kind == "mn3":
        return m * n**3
    if kind == "m_delta":
        return m * max(n - 1, 1)
    if kind == "leaves":
        return float(n) * float(max(n - 1, 1) ** (n - 1))
    raise OperationsFigureError(f"unknown bound kind {kind!r}")


def _median_by_n(
    rows: Sequence[Mapping[str, Any]], encoder: str, field: str
) -> tuple[list[int], list[float]]:
    """Return per-``n`` medians of *field* for one encoder, ascending in ``n``.

    Non-positive counts are dropped: the y axis is logarithmic and a zero
    cannot be drawn there.  ``backtrack_nodes`` and ``search_leaves`` are zero
    for a greedy encode by construction, so those two series simply do not
    appear in the backtracking panel.
    """
    by_n: dict[int, list[float]] = {}
    for row in rows:
        if row["encoder"] != encoder:
            continue
        value = float(row[field])
        if value <= 0.0:
            continue
        by_n.setdefault(int(row["n"]), []).append(value)
    ordered = sorted(by_n)
    return ordered, [statistics.median(by_n[n]) for n in ordered]


def _median_m_by_n(rows: Sequence[Mapping[str, Any]]) -> dict[int, float]:
    """Return the median ``m`` at each ``n``, which the bounds are drawn at."""
    by_n: dict[int, list[float]] = {}
    for row in rows:
        by_n.setdefault(int(row["n"]), []).append(float(row["m"]))
    return {n: statistics.median(values) for n, values in by_n.items()}


def _draw_panel(
    ax: Axes, panel: Panel, rows: Sequence[Mapping[str, Any]], m_by_n: dict[int, float]
) -> dict[str, Any]:
    """Draw one operation panel and return what it plotted."""
    drawn: dict[str, Any] = {}
    for style in ENCODERS:
        rep = design.BY_KEY[style.colour_key]
        ns, ys = _median_by_n(rows, style.encoder, panel.field)
        if ns:
            ax.plot(
                ns,
                ys,
                color=rep.colour,
                marker=style.marker,
                markersize=design.MS_POINT,
                linestyle=style.linestyle,
                linewidth=design.LW_BACKGROUND,
                label=style.short,
                zorder=4,
            )
        drawn[f"{style.encoder}:{panel.field}"] = len(ns)
        if panel.extra is not None:
            ex_ns, ex_ys = _median_by_n(rows, style.encoder, panel.extra)
            if ex_ns:
                ax.plot(
                    ex_ns,
                    ex_ys,
                    color=rep.colour,
                    marker="None",
                    linestyle=(0, (1.0, 1.4)),
                    linewidth=design.LW_BACKGROUND,
                    alpha=design.ALPHA_MUTED,
                    zorder=3,
                    label=f"{style.short} · {panel.extra.replace('_', ' ')}",
                )
            drawn[f"{style.encoder}:{panel.extra}"] = len(ex_ns)

    bound_ns = sorted(m_by_n)
    ax.plot(
        bound_ns,
        [_bound_value(panel.bound, n, m_by_n[n]) for n in bound_ns],
        color=design.INK_RULE,
        linestyle="--",
        linewidth=1.0,
        zorder=2,
        label=panel.bound_label,
    )
    ax.set_yscale("log")
    ax.set_title(panel.title, fontsize=design.FS_TITLE)
    design.finish_axes(ax, xlabel="$n$", ylabel=panel.ylabel)
    return drawn


def figure(counters: data.CounterRecords, out: Path) -> tuple[list[Path], dict[str, Any]]:
    """Draw the operations figure.

    Args:
        counters: The loaded counter records.
        out: Output path without an extension.

    Returns:
        ``(paths_written, summary)``.

    Raises:
        OperationsFigureError: If the records carry an encoder this module does
            not style, which would otherwise vanish from every panel silently.
    """
    rows = list(counters.rows)
    unknown = sorted(set(counters.encoders) - set(ENCODER_BY_NAME))
    if unknown:
        raise OperationsFigureError(
            f"counter records carry unstyled encoder(s) {unknown}; add them to "
            f"fig_operations.ENCODERS. Known: {sorted(ENCODER_BY_NAME)}"
        )

    m_by_n = _median_m_by_n(rows)
    plt = design.style()
    fig, axes = plt.subplots(
        2, 2, figsize=(design.text_width(), PANEL_HEIGHT_IN * 2), squeeze=False
    )
    flat = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]

    summary: dict[str, Any] = {
        "n_rows": len(rows),
        "encoders": list(counters.encoders),
        "n_values": sorted(m_by_n),
        "panels": {},
    }
    handles: list[Any] = []
    labels: list[str] = []
    for index, (ax, panel) in enumerate(zip(flat, PANELS)):
        summary["panels"][panel.field] = _draw_panel(ax, panel, rows, m_by_n)
        design.panel_letter(ax, chr(ord("a") + index))
        panel_handles, panel_labels = ax.get_legend_handles_labels()
        handles.extend(panel_handles)
        labels.extend(panel_labels)

    fig.tight_layout()
    design.shared_legend(fig, handles, labels, ncol=4, y=0.0)
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
    ns = summary["n_values"]
    span = f"$n = {min(ns)}$ to ${max(ns)}$" if ns else "the counter cohort"
    return (
        "Realised operation counts against the derived per-encode bounds of "
        "Section 2.1, over "
        f"{summary['n_rows']} instrumented encodes spanning {span}. "
        "Markers are per-$n$ medians; the dashed curve is the bound, evaluated at the "
        "median $m$ of each $n$. The maximum degree is not carried by the counter record, "
        "so the two $\\Delta$-dependent bounds are drawn at $\\Delta = n - 1$ and are "
        "therefore looser than the bound they stand for, never tighter. "
        "Section 2.1 bounds the number of search \\emph{leaves}, so panel (d) draws "
        "\\texttt{search\\_leaves} beside \\texttt{backtrack\\_nodes} and attaches the "
        "bound to the series it bounds; both are zero for a greedy encode by construction "
        "and a logarithmic axis omits them."
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--counters",
        nargs="+",
        required=True,
        help="glob(s) or paths of t13c.1 counter JSONL files",
    )
    ap.add_argument(
        "--records",
        nargs="+",
        default=None,
        help="accepted for CLI uniformity across this package; not read by this module",
    )
    ap.add_argument("--out-dir", type=Path, required=True, help="figure output directory")
    ap.add_argument(
        "--allow-parity-failures",
        action="store_true",
        help="plot rows whose instrumented mirror did not reproduce the reference string",
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
    if args.records:
        LOGGER.info("--records is not read by fig_operations; ignoring %s", args.records)
    counters = data.load_counters(args.counters, strict_parity=not args.allow_parity_failures)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / STEM
    _, summary = figure(counters, stem)
    (args.out_dir / f"{STEM}.caption.tex").write_text(caption(summary) + "\n")
    LOGGER.info("%d counter row(s) drawn", summary["n_rows"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
