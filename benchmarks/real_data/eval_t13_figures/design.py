"""The single design source for every T-13 paper figure and table.

Nothing below this module may hard-code a colour, a marker, a dash pattern, a
font size, a display name, a draw order or a LaTeX macro for a representation.

**Two defects in the T-06 registry are fixed here, deliberately.**

1. ``eval_t06_figures.design.REPRESENTATIONS`` is a hand-written literal that
   nothing checks against the measurement registry.  Adding an arm to the
   campaign and forgetting to add it here produced figures that regenerated
   *successfully, with the arm absent and no error raised* -- the trap
   ``.claude/CLAUDE.md`` records.  Here the keys are imported from
   :data:`benchmarks.real_data.eval_t13_complexity.measure.REPRESENTATIONS`
   and :func:`_check_registry` runs **at import time**, so an arm added to the
   campaign and not styled here is an ``ImportError``, not a missing legend
   entry.
2. ``eval_t06_figures.design.present`` drops unknown names silently.  Here
   :func:`present` **raises**.  A caller that genuinely wants an arm left out
   must name it in ``omit``, which puts the omission in the call site where a
   reviewer can see it.

**The palette is not new.**  Every hex below is the one T-06 pinned for the
same backend, because the two ticket's figures sit in one paper and a
representation that changes colour between figures is unreadable.  Geometry and
rcParams come from ``isalgraph.viz.style`` via ``benchmarks.plotting_styles``,
which is already the single source of truth for the published palette.

**What the taxonomy draws.**  ``T-13-design.md`` 2.3 gives the search-free arms
one job: to be the null.  Their cost must be flat in ``|Aut|``, so that a
rising curve on a search arm cannot be any confound that tracks symmetry.  That
split is therefore load-bearing and is drawn (dash pattern) as well as stored.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from benchmarks.real_data.eval_t13_complexity import measure

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class RegistryError(ValueError):
    """The style registry and the measurement registry disagree."""


class UnknownRepresentationError(KeyError):
    """A figure or table was handed a representation this module does not style.

    Raised rather than swallowed.  A figure must never invent a style for a
    backend the registry does not know, and it must never quietly draw one
    fewer series than the data contains.
    """


# ---------------------------------------------------------------------------
# Type sizes.  Every figure reads these; none defines its own.
# ---------------------------------------------------------------------------

#: Suptitle across a multi-panel figure.
FS_SUPTITLE: Final[float] = 9.0
#: Per-panel title.
FS_TITLE: Final[float] = 8.0
#: Axis labels.
FS_LABEL: Final[float] = 7.5
#: Tick labels.
FS_TICK: Final[float] = 6.5
#: Legend entries.
FS_LEGEND: Final[float] = 6.4
#: In-axes annotation boxes.
FS_ANNOT: Final[float] = 5.6
#: Panel letter (a), (b), (c).
FS_PANEL: Final[float] = 8.5

#: Line weight for a background arm and for one the characterisation names.
LW_BACKGROUND: Final[float] = 1.0
LW_FOCUS: Final[float] = 1.9

#: Marker size for a completed observation and for a censored one.  The
#: censored marker is larger because it carries an arrow and must still read
#: as one glyph at column scale.
MS_POINT: Final[float] = 3.4
MS_CENSORED: Final[float] = 4.6

#: Alpha for a de-emphasised series and for a shaded region.
ALPHA_MUTED: Final[float] = 0.55
ALPHA_BAND: Final[float] = 0.13

#: Grid.
GRID_ALPHA: Final[float] = 0.25
GRID_LW: Final[float] = 0.4

#: Neutral ink for reference lines and ceilings.
INK_RULE: Final[str] = "0.35"
INK_CEILING: Final[str] = "0.20"

#: Length of the "greater than this" arrow drawn above a censored point, as a
#: fraction of the axes height.  Short on purpose: the arrow says *direction*,
#: not magnitude, and a censored observation carries no magnitude.
CENSOR_ARROW_FRACTION: Final[float] = 0.045


class SearchClass(enum.Enum):
    """Whether the representation obtains its canonical form by searching.

    The split is imported from ``measure.SEARCH_BASED`` / ``measure.SEARCH_FREE``
    rather than restated, and :func:`_check_registry` enforces the agreement.

    Attributes:
        SEARCH_BASED: A canonical form obtained by searching.  The cost law is
            a claim about this class.
        SEARCH_FREE: The null.  Cost must be ``Theta(n^2)`` or ``Theta(n + m)``
            and **flat in** ``|Aut|``.
    """

    SEARCH_BASED = "search-based"
    SEARCH_FREE = "search-free"


#: Dash pattern per search class, so the null survives a greyscale print.
SEARCH_CLASS_LINESTYLE: Final[dict[SearchClass, Any]] = {
    SearchClass.SEARCH_BASED: "-",
    SearchClass.SEARCH_FREE: (0, (1.6, 1.4)),
}


@dataclass(frozen=True, slots=True)
class Representation:
    """One series of every T-13 figure and one row of every T-13 table.

    Attributes:
        key: Backend name, exactly as ``measure.REPRESENTATIONS`` spells it.
        short: Axis-legend name.  Kept under ~16 characters.
        long: Name for a table cell or a caption.
        tex: LaTeX name, already escaped.
        colour: Pinned hex colour, identical to the T-06 registry's.
        marker: Matplotlib marker for a completed observation.
        search_class: See :class:`SearchClass`.
        is_ours: An IsalGraph arm.  Drawn with a black marker edge.
        is_focus: Named by the ``T-13-design.md`` 6.3 characterisation, so the
            reader's eye must land on it.  Drawn heavier and on top.
    """

    key: str
    short: str
    long: str
    tex: str
    colour: str
    marker: str
    search_class: SearchClass
    is_ours: bool = False
    is_focus: bool = False

    @property
    def linestyle(self) -> Any:
        """Dash pattern implied by :attr:`search_class`."""
        return SEARCH_CLASS_LINESTYLE[self.search_class]

    @property
    def linewidth(self) -> float:
        """Heavier stroke for an arm the characterisation names."""
        return LW_FOCUS if self.is_focus else LW_BACKGROUND

    @property
    def zorder(self) -> int:
        """A focus arm is never hidden behind a background one."""
        return 6 if self.is_focus else 3


#: The thirteen styled arms, in draw order.
#:
#: Colours are transcribed from ``eval_t06_figures.design.REPRESENTATIONS`` so
#: that a backend keeps one colour across the whole paper.  ``is_focus`` marks
#: exactly the five arms the 6.3 pilot table reports, which are the five the
#: main-text figure is read on.
REPRESENTATIONS: Final[tuple[Representation, ...]] = (
    Representation(
        key="isalgraph_exhaustive",
        short="IsalGraph (exh.)",
        long="IsalGraph (exhaustive canonical, pruned fallback)",
        tex=r"\textsc{IsalGraph}$^{\dagger}$",
        colour="#EE3377",
        marker="d",
        search_class=SearchClass.SEARCH_BASED,
        is_ours=True,
        is_focus=True,
    ),
    Representation(
        key="isalgraph_pruned",
        short="IsalGraph",
        long="IsalGraph (pruned canonical)",
        tex=r"\textsc{IsalGraph}",
        colour="#AA3377",
        marker="o",
        search_class=SearchClass.SEARCH_BASED,
        is_ours=True,
        is_focus=True,
    ),
    Representation(
        key="isalgraph_canonical",
        short="IsalGraph (pure exh.)",
        long="IsalGraph (exhaustive canonical, no fallback)",
        tex=r"\textsc{IsalGraph}$^{\ast}$",
        colour="#D699C2",
        marker="o",
        search_class=SearchClass.SEARCH_BASED,
        is_ours=True,
    ),
    Representation(
        key="isalgraph_greedy",
        short="IsalGraph greedy",
        long="IsalGraph (greedy-min, canonical search ablated)",
        tex=r"\textsc{IsalGraph}$_{\mathrm{greedy}}$",
        colour="#CC6677",
        marker="h",
        search_class=SearchClass.SEARCH_BASED,
        is_ours=True,
        is_focus=True,
    ),
    Representation(
        key="min_dfs",
        short="gSpan min-DFS",
        long="gSpan minimum DFS code",
        tex="gSpan min-DFS",
        colour="#117733",
        marker="s",
        search_class=SearchClass.SEARCH_BASED,
        is_focus=True,
    ),
    Representation(
        key="agm_cam",
        short="AGM CAM",
        long="AGM canonical adjacency matrix",
        tex="AGM CAM",
        colour="#44AA99",
        marker="D",
        search_class=SearchClass.SEARCH_BASED,
    ),
    Representation(
        key="nauty_graph6",
        short="nauty graph6",
        long="graph6 after a nauty canonical labelling",
        tex="nauty graph6",
        colour="#4477AA",
        marker="^",
        search_class=SearchClass.SEARCH_BASED,
    ),
    Representation(
        key="sparse6_nauty",
        short="nauty sparse6",
        long="sparse6 after a nauty canonical labelling",
        tex="nauty sparse6",
        colour="#66CCEE",
        marker="v",
        search_class=SearchClass.SEARCH_BASED,
    ),
    Representation(
        key="graph6",
        short="graph6",
        long="graph6 of the incident labelling",
        tex="graph6",
        colour="#999933",
        marker="^",
        search_class=SearchClass.SEARCH_FREE,
        is_focus=True,
    ),
    Representation(
        key="sparse6",
        short="sparse6",
        long="sparse6 of the incident labelling",
        tex="sparse6",
        colour="#DDCC77",
        marker="v",
        search_class=SearchClass.SEARCH_FREE,
    ),
    Representation(
        key="adjacency",
        short="adjacency",
        long="packed adjacency triangle",
        tex="adjacency",
        colour="#888888",
        marker="P",
        search_class=SearchClass.SEARCH_FREE,
    ),
    Representation(
        key="wl_subtree",
        short="WL subtree",
        long="Weisfeiler--Lehman subtree kernel ($h = 2$)",
        tex="WL subtree",
        colour="#EE7733",
        marker="X",
        search_class=SearchClass.SEARCH_FREE,
    ),
    Representation(
        key="size_null",
        short=r"$|n_i-n_j|$ null",
        long="descriptive size null",
        tex=r"$|n_i-n_j|$ null",
        colour="#555555",
        marker="*",
        search_class=SearchClass.SEARCH_FREE,
    ),
)

#: Lookup by key.
BY_KEY: Final[dict[str, Representation]] = {r.key: r for r in REPRESENTATIONS}

#: Draw order.  A figure filters this rather than defining its own tuple.
ORDER: Final[tuple[str, ...]] = tuple(r.key for r in REPRESENTATIONS)

#: The arm the paper ships, and the one every contrast is read against.
REFERENCE_KEY: Final[str] = "isalgraph_pruned"

#: The unpruned counterpart of :data:`REFERENCE_KEY`.  The exhaustive-versus-
#: pruned contrast is a *primary* comparison (``T-13-design.md`` 6.3
#: consequence 1), not an incidental one.
UNPRUNED_KEY: Final[str] = "isalgraph_exhaustive"

#: The parameter ``T-13-design.md`` 2.1 and 6.3 name as governing each arm's
#: cost.  **Pre-registered expectation, not a measurement.**  It is quoted in
#: captions so a reader can see what the figure was drawn to test; the figure
#: itself reports what the ladder measured, which may disagree.  Arms the
#: design note does not name are absent, and every consumer must ``.get()``.
HYPOTHESISED_DRIVER: Final[dict[str, str]] = {
    "isalgraph_exhaustive": "degree sequence",
    "isalgraph_canonical": "degree sequence",
    "isalgraph_pruned": "automorphism group",
    "isalgraph_greedy": "size only",
    "graph6": "size only",
    "sparse6": "size only",
    "adjacency": "size only",
    "wl_subtree": "size only",
    "size_null": "size only",
}

#: How a censored observation is drawn, and why.  ``status = "censored"``
#: means *the completion time is greater than this*, so the point is an open
#: marker (no fill: nothing completed) carrying an upward arrow (the true value
#: lies above).  Drawing it as an ordinary point would read as a fast
#: completion, which for a cap-censored ``min_dfs`` row -- a few milliseconds --
#: inverts the result.
CENSORED_FILLSTYLE: Final[str] = "none"

#: Display name per censoring mechanism.  ``schema`` keeps the two mechanisms
#: separable and so does the legend: a wall-clock kill at the budget and a
#: projection cap that fires in 40 ms are not the same observation.
CENSORING_DISPLAY: Final[dict[str, str]] = {
    "wallclock_kill": "wall-clock kill",
    "timeout_s": "engine timeout",
    "search_nodes": "search-node cap",
    "max_projections": "projection cap",
}


# ---------------------------------------------------------------------------
# Registry integrity.  Runs at import.
# ---------------------------------------------------------------------------


def _check_registry() -> None:
    """Raise unless this module styles exactly the campaign's thirteen arms.

    Both directions are errors, for the same reason ``schema.validate_mapping``
    rejects both a missing and an extra field: a styled arm the campaign does
    not run is dead code, and a campaign arm with no style is the silent
    omission this module exists to prevent.

    Raises:
        RegistryError: on a styled-but-unmeasured key, a measured-but-unstyled
            key, a duplicate key, or a search-class assignment that disagrees
            with ``measure.SEARCH_BASED`` / ``measure.SEARCH_FREE``.
    """
    keys = [r.key for r in REPRESENTATIONS]
    if len(keys) != len(set(keys)):
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        raise RegistryError(f"duplicate representation keys in the style registry: {duplicates}")

    styled = set(keys)
    measured = set(measure.REPRESENTATIONS)
    unstyled = sorted(measured - styled)
    unmeasured = sorted(styled - measured)
    if unstyled or unmeasured:
        raise RegistryError(
            "the T-13 style registry disagrees with measure.REPRESENTATIONS: "
            f"measured but unstyled={unstyled} (these would vanish from every figure "
            f"with no error), styled but unmeasured={unmeasured}"
        )

    expected = {
        **dict.fromkeys(measure.SEARCH_BASED, SearchClass.SEARCH_BASED),
        **dict.fromkeys(measure.SEARCH_FREE, SearchClass.SEARCH_FREE),
    }
    wrong = sorted(r.key for r in REPRESENTATIONS if r.search_class is not expected[r.key])
    if wrong:
        raise RegistryError(
            f"search_class disagrees with measure's own split for {wrong}; the "
            f"search-free arms are the null of the cost law and cannot be mislabelled"
        )


_check_registry()


# ---------------------------------------------------------------------------
# Accessors.  Figures and tables call these, never the dicts directly.
# ---------------------------------------------------------------------------


def present(keys: Iterable[str], *, omit: Iterable[str] = ()) -> tuple[Representation, ...]:
    """Return the representations for *keys*, in draw order.

    Args:
        keys: Backend names carried by the data in hand.
        omit: Names to leave out on purpose.  A name here is dropped whether or
            not it is registered, which is what makes an intentional omission
            visible at the call site.

    Returns:
        Representations ordered by :data:`ORDER`.

    Raises:
        UnknownRepresentationError: if any name in *keys* is neither registered
            nor listed in *omit*.  T-06's counterpart drops such a name
            silently; that is how an arm disappears from a figure that
            regenerates without error.
    """
    wanted = set(keys)
    omitted = set(omit)
    unknown = sorted(wanted - set(BY_KEY) - omitted)
    if unknown:
        raise UnknownRepresentationError(
            f"unregistered representation(s) {unknown}: add them to "
            f"eval_t13_figures.design.REPRESENTATIONS, or pass them in omit= to drop "
            f"them deliberately. Registered: {sorted(BY_KEY)}"
        )
    return tuple(r for r in REPRESENTATIONS if r.key in wanted and r.key not in omitted)


def absent(keys: Iterable[str]) -> tuple[str, ...]:
    """Return the registered arms that *keys* does **not** cover, in draw order.

    The other half of the silent-omission problem: :func:`present` catches a
    key with no style, this catches a style with no data.  Callers log it; a
    figure with twelve of thirteen arms is a legitimate result on a ladder
    where one backend is ``unsupported`` throughout, but it must be said.

    Args:
        keys: Backend names carried by the data in hand.

    Returns:
        The missing registered keys.
    """
    have = set(keys)
    return tuple(k for k in ORDER if k not in have)


def tex_name(key: str) -> str:
    """Return the LaTeX name for *key*.

    Args:
        key: Backend name.

    Returns:
        The registered LaTeX name.

    Raises:
        UnknownRepresentationError: if *key* is not registered.
    """
    rep = BY_KEY.get(key)
    if rep is None:
        raise UnknownRepresentationError(f"unregistered representation {key!r}")
    return rep.tex


def line_kwargs(rep: Representation, *, muted: bool = False) -> dict[str, Any]:
    """Return the ``Axes.plot`` keyword arguments for one completed series.

    Args:
        rep: The representation being drawn.
        muted: Draw the series as background: thinner, dashed, half-opaque.
            Muting is not hiding -- the series is still drawn and still
            labelled.

    Returns:
        Keyword arguments for ``Axes.plot``.
    """
    style: dict[str, Any] = {
        "color": rep.colour,
        "linestyle": SEARCH_CLASS_LINESTYLE[SearchClass.SEARCH_FREE] if muted else rep.linestyle,
        "linewidth": LW_BACKGROUND * 0.85 if muted else rep.linewidth,
        "marker": rep.marker,
        "markersize": MS_POINT,
        "alpha": ALPHA_MUTED if muted else 1.0,
        "zorder": 2 if muted else rep.zorder,
    }
    if rep.is_ours:
        style["markeredgecolor"] = "black"
        style["markeredgewidth"] = 0.35
    return style


def censored_kwargs(rep: Representation) -> dict[str, Any]:
    """Return the ``Axes.plot`` keyword arguments for a **censored** point.

    An open marker with no connecting line.  The caller adds the upward arrow
    with :func:`censor_arrow`; the two together say "greater than this", which
    is the whole content of a right-censored observation.

    Args:
        rep: The representation being drawn.

    Returns:
        Keyword arguments for ``Axes.plot``.
    """
    return {
        "color": rep.colour,
        "linestyle": "none",
        "marker": rep.marker,
        "markersize": MS_CENSORED,
        "markerfacecolor": CENSORED_FILLSTYLE,
        "markeredgecolor": rep.colour,
        "markeredgewidth": 0.9,
        "zorder": rep.zorder + 1,
        "alpha": 1.0,
    }


def label(rep: Representation) -> str:
    """Return the legend label for one series.

    Args:
        rep: The representation being drawn.

    Returns:
        The label.
    """
    return rep.short


# ---------------------------------------------------------------------------
# Figure lifecycle.  Every matplotlib import lives inside a function body.
# ---------------------------------------------------------------------------


def style() -> Any:
    """Apply the shared IEEE rcParams and return the pyplot module.

    Geometry and rcParams come straight from ``isalgraph.viz.style``, which is
    the single source of truth for the published palette;
    ``benchmarks.plotting_styles`` re-exports the same values byte for byte, so
    calling the source rather than the re-export costs nothing and cannot
    drift.

    Returns:
        ``matplotlib.pyplot``, styled.  Imported inside the function so this
        module stays importable with matplotlib absent, matching the
        ``isalgraph.viz`` contract.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from isalgraph.viz import style as viz_style

    viz_style.apply_ieee_style()
    return plt


def text_width() -> float:
    """Return the IEEE full-text-width figure width in inches."""
    from isalgraph.viz import style as viz_style

    return float(viz_style.IEEE_TEXT_WIDTH_INCHES)


def column_width() -> float:
    """Return the IEEE single-column figure width in inches."""
    from isalgraph.viz import style as viz_style

    return float(viz_style.IEEE_COLUMN_WIDTH_INCHES)


def save(fig: Figure, basepath: str | Path) -> list[Path]:
    """Write *fig* as PDF and PNG under *basepath*.

    Args:
        fig: The figure.
        basepath: Output path without an extension.

    Returns:
        The paths written, PDF first.
    """
    from isalgraph.viz import style as viz_style

    return [Path(p) for p in viz_style.save_figure(fig, str(basepath))]


def finish_axes(ax: Axes, *, xlabel: str | None = None, ylabel: str | None = None) -> None:
    """Apply the shared grid, tick and label treatment to one panel.

    Args:
        ax: The axes.
        xlabel: X label, or ``None`` to leave it.
        ylabel: Y label, or ``None`` to leave it.
    """
    ax.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LW)
    ax.tick_params(labelsize=FS_TICK)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)


def panel_letter(ax: Axes, letter: str) -> None:
    """Stamp a panel letter in the top-left corner of *ax*.

    Args:
        ax: The axes.
        letter: The letter, without parentheses.
    """
    ax.text(
        -0.02,
        1.045,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=FS_PANEL,
        fontweight="bold",
        ha="right",
        va="bottom",
    )


def note_box(ax: Axes, text: str, *, loc: str = "lower left") -> None:
    """Draw the standard in-axes annotation box.

    Args:
        ax: The axes.
        text: Box contents.
        loc: ``lower left``, ``lower right``, ``upper left`` or
            ``upper right``.

    Raises:
        ValueError: If *loc* is not one of the four supported corners.
    """
    anchors = {
        "lower left": (0.025, 0.03, "left", "bottom"),
        "lower right": (0.975, 0.03, "right", "bottom"),
        "upper left": (0.025, 0.97, "left", "top"),
        "upper right": (0.975, 0.97, "right", "top"),
    }
    if loc not in anchors:
        raise ValueError(f"loc must be one of {sorted(anchors)}, got {loc!r}")
    x, y, ha, va = anchors[loc]
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=FS_ANNOT,
        ha=ha,
        va=va,
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "white",
            "edgecolor": "0.6",
            "linewidth": 0.5,
            "alpha": 0.94,
        },
    )


def censor_arrow(ax: Axes, x: float, y: float, colour: str) -> None:
    """Draw the "greater than this" arrow above one censored point.

    The arrow is a fixed fraction of the axes height in *display* space, so it
    reads the same on a log axis wherever the point sits.  It deliberately
    carries no magnitude: a right-censored observation bounds the completion
    time from below and says nothing about how far above it lies.

    Args:
        ax: The axes, whose y scale may be log.
        x: Point abscissa in data coordinates.
        y: Point ordinate in data coordinates.
        colour: Arrow colour, the representation's own.
    """
    trans = ax.transData
    x_disp, y_disp = trans.transform((x, y))
    height = abs(ax.bbox.height) * CENSOR_ARROW_FRACTION
    x_top, y_top = trans.inverted().transform((x_disp, y_disp + height))
    ax.annotate(
        "",
        xy=(x_top, y_top),
        xytext=(x, y),
        arrowprops={
            "arrowstyle": "-|>",
            "color": colour,
            "linewidth": 0.8,
            "shrinkA": 1.0,
            "shrinkB": 0.0,
        },
        annotation_clip=False,
        zorder=8,
    )


def shared_legend(
    fig: Figure,
    handles: list[Any],
    labels: list[str],
    *,
    ncol: int = 5,
    y: float = -0.015,
) -> None:
    """Place one de-duplicated legend under a multi-panel figure.

    Args:
        fig: The figure.
        handles: Legend handles, in draw order.
        labels: Matching labels.
        ncol: Legend columns.
        y: Vertical anchor in figure coordinates.
    """
    seen: dict[str, Any] = {}
    for handle, text in zip(handles, labels):
        seen.setdefault(text, handle)
    fig.legend(
        list(seen.values()),
        list(seen),
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        fontsize=FS_LEGEND,
        frameon=False,
    )


__all__ = [
    "ALPHA_BAND",
    "ALPHA_MUTED",
    "BY_KEY",
    "CENSORED_FILLSTYLE",
    "CENSORING_DISPLAY",
    "CENSOR_ARROW_FRACTION",
    "FS_ANNOT",
    "FS_LABEL",
    "FS_LEGEND",
    "FS_PANEL",
    "FS_SUPTITLE",
    "FS_TICK",
    "FS_TITLE",
    "GRID_ALPHA",
    "GRID_LW",
    "HYPOTHESISED_DRIVER",
    "INK_CEILING",
    "INK_RULE",
    "LW_BACKGROUND",
    "LW_FOCUS",
    "MS_CENSORED",
    "MS_POINT",
    "ORDER",
    "REFERENCE_KEY",
    "REPRESENTATIONS",
    "SEARCH_CLASS_LINESTYLE",
    "UNPRUNED_KEY",
    "RegistryError",
    "Representation",
    "SearchClass",
    "UnknownRepresentationError",
    "absent",
    "censor_arrow",
    "censored_kwargs",
    "column_width",
    "finish_axes",
    "label",
    "line_kwargs",
    "note_box",
    "panel_letter",
    "present",
    "save",
    "shared_legend",
    "style",
    "tex_name",
    "text_width",
]
