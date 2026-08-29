"""The single design source for every T-06 paper figure and table.

Nothing below this module may hard-code a colour, a font size, a display
name, a draw order or a LaTeX macro for a representation. Figures import
:data:`REPRESENTATIONS` and the ``FS_*`` sizes; tables import the same
registry and :func:`tex_name`. That is the whole contract.

**Why a registry rather than a palette walk.** ``eval_size_profile/figures.py``
assigned colours with ``palette[i % len(palette)]`` over the representations
*present in the current call*, so a representation's colour depended on which
others happened to have landed. Two figures of the same campaign could give
``min_dfs`` two different colours and neither would look wrong on its own.
Colour is pinned per representation here and cannot drift.

**The taxonomy is load-bearing, not decoration.** ``T-06-FRAMING.md`` section 2
freezes the claim as *"the most compact of the canonical-code representations;
edge-list serializations beat it at scale"*. That sentence is only checkable if
the figure shows which family each row belongs to, so :class:`Family` is drawn
(marker shape, dash pattern) as well as named.

Geometry and rcParams come from ``isalgraph.viz.style`` via
``benchmarks.plotting_styles``, which is already the single source of truth for
the published palette; this module adds the T-06 layer on top and never
redefines what that one owns.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Type sizes. Every figure reads these; none defines its own.
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
#: In-axes annotation boxes. Tuned to fig1's seven-line significance note,
#: which is the longest in the family and overflows its panel above this.
FS_ANNOT: Final[float] = 5.2
#: Panel letter (a), (b), (c).
FS_PANEL: Final[float] = 8.5

#: Line weight for a competitor and for the reference arm.
LW_COMPETITOR: Final[float] = 1.1
LW_REFERENCE: Final[float] = 1.9

#: Marker area for a plain point and for the open ring marking significance.
MS_POINT: Final[float] = 3.0
MS_SIGNIFICANT: Final[float] = 46.0

#: Alpha for a confidence band and for a de-emphasized series.
ALPHA_BAND: Final[float] = 0.13
ALPHA_MUTED: Final[float] = 0.65

#: Grid.
GRID_ALPHA: Final[float] = 0.25
GRID_LW: Final[float] = 0.4

#: Neutral ink for reference lines, zero lines and regime separators.
INK_RULE: Final[str] = "0.35"
INK_SEPARATOR: Final[str] = "0.50"
INK_FLOOR: Final[str] = "0.20"

#: Shading for a region of the x axis where only one dataset contributes, so
#: the pooled curve is a composition artifact rather than a trend.
INK_THIN: Final[str] = "0.90"

#: False-discovery rate used by every descriptive figure in this package.
FDR_Q: Final[float] = 0.05

#: Above this node count exact GED stops being computable (networkx A*), and
#: the LB/UB bracket takes over. Frozen in ``T-06-design.md`` 15.1.
EXACT_CEILING: Final[int] = 12

#: The size scope the compactness claim carries. ``T-06-article-notes.md`` 5:
#: never write "IsalGraph produces shorter encodings" without it.
CLAIM_A_SCOPE_N: Final[int] = 20


class Family(enum.Enum):
    """Design point a representation belongs to.

    The comparison sentence the paper is allowed to make depends on this,
    so it is drawn as well as stored.

    Attributes:
        CANONICAL_CODE: The code *is* the canonical form -- canonicity is
            intrinsic to how the string is built (IsalGraph, gSpan min-DFS,
            AGM CAM).
        CANONICALIZED_SERIALIZATION: A standard serialization applied after
            an external canonical labeling, i.e. nauty (nauty-graph6,
            nauty-sparse6).
        RAW_SERIALIZATION: A serialization of the incident labeling, with no
            canonicalization at all (adjacency, graph6, sparse6).
        KERNEL: A feature map with no message length (WL subtree).
        BASELINE: The trivial descriptive null.
    """

    CANONICAL_CODE = "canonical code"
    CANONICALIZED_SERIALIZATION = "canonicalized serialization"
    RAW_SERIALIZATION = "raw serialization"
    KERNEL = "kernel"
    BASELINE = "baseline"


#: Dash pattern per family, so the taxonomy survives a grayscale print.
FAMILY_LINESTYLE: Final[dict[Family, Any]] = {
    Family.CANONICAL_CODE: "-",
    Family.CANONICALIZED_SERIALIZATION: (0, (5, 1.4)),
    Family.RAW_SERIALIZATION: (0, (1.6, 1.4)),
    Family.KERNEL: (0, (5, 1.2, 1, 1.2)),
    Family.BASELINE: (0, (3, 1.2, 1, 1.2)),
}


@dataclass(frozen=True, slots=True)
class Representation:
    """One row of every T-06 figure and table.

    Attributes:
        key: Backend name as it appears in every T-06 artifact.
        short: Axis-legend name. Kept under ~16 characters.
        long: Name for a table cell or a caption.
        tex: LaTeX name, already escaped.
        colour: Pinned hex colour. Never index-assigned.
        marker: Matplotlib marker.
        family: Design point, see :class:`Family`.
        is_ours: The reference arm, drawn heavier and on top.
        canonical: Isomorphic graphs receive identical encodings. Measured,
            not declared: E1 relabeling sensitivity psi = 0 exactly.
        complete: Equal encodings imply isomorphic graphs (E2).
        reversible: ``decode(encode(G))`` recovers the graph.
        handles_disconnected: Encodes disconnected graphs and isolated
            vertices without error.
        metric_admissible: Some candidate distance passed T-04a's F1-F4. The
            three that did not are k-excluded and carry no Claim B column.
        bit_countable: Has a message length at all. ``wl_subtree`` and
            ``size_null`` raise ``BitCountUndefined``.
        max_n: Largest node count the backend produced an encoding for in the
            T-06 cohorts. A competitor that refuses above a size cannot be
            beaten above it, so a pooled win rate against it is a statement
            about small graphs only.
        in_figures: Whether the arm is drawn. ``False`` keeps a row in every
            table and out of every figure -- for an arm whose *number* is a
            result but whose *curve* is not readable. It is never a way to
            hide a bad result: a comparator that loses stays drawn, and the
            only arms this is set on are our own.
    """

    key: str
    short: str
    long: str
    tex: str
    colour: str
    marker: str
    family: Family
    is_ours: bool = False
    canonical: bool = False
    complete: bool = False
    reversible: bool = False
    handles_disconnected: bool = False
    metric_admissible: bool = False
    bit_countable: bool = True
    max_n: int | None = None
    in_figures: bool = True

    @property
    def linestyle(self) -> Any:
        """Dash pattern implied by :attr:`family`."""
        return FAMILY_LINESTYLE[self.family]

    @property
    def linewidth(self) -> float:
        """Heavier stroke for the reference arm."""
        return LW_REFERENCE if self.is_ours else LW_COMPETITOR

    @property
    def zorder(self) -> int:
        """The reference arm is never hidden behind a comparator."""
        return 6 if self.is_ours else 3


#: Every representation T-06 measured, in draw order. Ours first so it is
#: painted last onto the legend and first into the reader's eye.
#:
#: ``max_n`` and the capability flags are transcribed from the campaign, not
#: asserted: capabilities from ``isalgraph.competitors.registry`` (each
#: backend's declared ``Capability`` set), ``metric_admissible`` from
#: ``competitors.md`` 9.1 -- ``k = 3``, the three excluded fail F3 at 1/50 --
#: and ``max_n`` from the encoding cells in ``data/source/T06/encodings/``.
#:
#: **Two measured representations are deliberately absent: ``graph6`` and
#: ``sparse6``, the raw incident-labeling serializations.** Their cells still
#: exist in every archive and nothing was recomputed; they are withdrawn from
#: the reported comparison in favor of their nauty-canonicalized forms.
#:
#: The reason is that one of them carries no information at all. graph6's
#: length is a function of ``n`` alone, so canonicalizing permutes the bits
#: without changing how many there are: over the T06_exhaustive cohort the
#: ``graph6`` and ``nauty_graph6`` medians agree at **every** node count
#: (36/36 at ``n = 8``, 198/198 at ``n = 20``, 786/786 at ``n = 40``,
#: 1776/1776 at ``n = 60``). Its curve was drawn exactly underneath nauty's
#: and its table row duplicated nauty's to the bit. ``sparse6`` does differ
#: -- 324 against nauty-sparse6's 336 at ``n = 40`` -- but only because the
#: incident labeling it happens to be handed compresses better than the
#: canonical one, which is a property of the input order rather than of the
#: format, and it is not invariant (``psi`` 0.54--1.15, F3 at 1/50), so no
#: correlation column was ever defined for it.
#:
#: **What this costs, stated because it is not free.** ``sparse6`` was the
#: most compact serialization in the field and it was not metric-admissible,
#: which is what let the head-to-head caption say that neither axis has a
#: single leader. With it withdrawn the most compact serialization is
#: nauty-sparse6, which *is* admissible, so that sentence is no longer true
#: and has been removed rather than reworded. The dominance nauty-sparse6
#: holds over the instruction string above ``n = 20`` now stands unmitigated
#: in both captions. It was already reported; it is simply no longer paired
#: with a consolation.
REPRESENTATIONS: Final[tuple[Representation, ...]] = (
    Representation(
        key="isalgraph_pruned",
        short="IsalGraph",
        long="IsalGraph (pruned canonical)",
        tex=r"\textsc{IsalGraph}",
        colour="#AA3377",
        marker="o",
        family=Family.CANONICAL_CODE,
        is_ours=True,
        canonical=True,
        complete=True,
        reversible=True,
        handles_disconnected=False,
        metric_admissible=True,
        max_n=96,
    ),
    Representation(
        key="isalgraph_canonical",
        short="IsalGraph (exh.)",
        long="IsalGraph (exhaustive canonical)",
        tex=r"\textsc{IsalGraph}$^{\ast}$",
        colour="#D699C2",
        marker="o",
        family=Family.CANONICAL_CODE,
        is_ours=True,
        canonical=True,
        complete=True,
        reversible=True,
        handles_disconnected=False,
        metric_admissible=True,
        max_n=12,
    ),
    Representation(
        key="isalgraph_exhaustive",
        short="IsalGraph (hyb.)",
        long="IsalGraph (exhaustive canonical, pruned fallback)",
        tex=r"\textsc{IsalGraph}$^{\dagger}$",
        # Deliberately far from the pruned arm's #AA3377. The two curves
        # coincide above n ~ 28 -- 96.8 % of the n = 40 stratum falls back to
        # the pruned string -- so a near-neighbor magenta made the pair
        # unreadable exactly where the reader needs to see them separate.
        colour="#EE3377",
        marker="d",
        family=Family.CANONICAL_CODE,
        is_ours=True,
        canonical=True,
        complete=True,
        reversible=True,
        handles_disconnected=False,
        metric_admissible=True,
        max_n=98,
        # Tabled, not drawn. The arm is the T06_exhaustive campaign's whole
        # result and its bit counts are the headline -- 114.1 at n = 20
        # against the pruned arm's 136.3 -- so the row stays. The *curve*
        # does not survive the same reading: the D14 cascade substitutes the
        # pruned string when the 30 s budget expires, at 41.2 % of the
        # n = 21-30 stratum and 96.8 % of n = 40, so above n ~ 28 this line
        # is the pruned line wearing a second colour. Two near-identical
        # curves in one figure read as a measured agreement between two
        # methods; here they are one method drawn twice, which is the more
        # misleading of the two ways to be wrong.
        in_figures=False,
    ),
    Representation(
        key="isalgraph_greedy",
        short="IsalGraph greedy",
        long="IsalGraph (greedy-min, canonical search ablated)",
        tex=r"\textsc{IsalGraph}$_{\mathrm{greedy}}$",
        colour="#CC6677",
        marker="h",
        family=Family.RAW_SERIALIZATION,
        is_ours=True,
        canonical=False,
        complete=False,
        reversible=True,
        handles_disconnected=False,
        metric_admissible=False,
        max_n=98,
    ),
    Representation(
        key="min_dfs",
        short="gSpan min-DFS",
        long="gSpan minimum DFS code",
        tex="gSpan min-DFS",
        colour="#117733",
        marker="s",
        family=Family.CANONICAL_CODE,
        canonical=True,
        complete=True,
        reversible=True,
        handles_disconnected=False,
        metric_admissible=True,
        max_n=96,
    ),
    Representation(
        key="agm_cam",
        short="AGM CAM",
        long="AGM canonical adjacency matrix",
        tex="AGM CAM",
        colour="#44AA99",
        marker="D",
        family=Family.CANONICAL_CODE,
        canonical=True,
        complete=True,
        reversible=True,
        handles_disconnected=True,
        metric_admissible=True,
        max_n=12,
    ),
    Representation(
        key="nauty_graph6",
        short="nauty graph6",
        long="nauty canonical labeling, graph6",
        tex="nauty-graph6",
        colour="#4477AA",
        marker="^",
        family=Family.CANONICALIZED_SERIALIZATION,
        canonical=True,
        complete=True,
        reversible=True,
        handles_disconnected=True,
        metric_admissible=True,
        max_n=98,
    ),
    Representation(
        key="sparse6_nauty",
        short="nauty sparse6",
        long="nauty canonical labeling, sparse6",
        tex="nauty-sparse6",
        colour="#66CCEE",
        marker="v",
        family=Family.CANONICALIZED_SERIALIZATION,
        canonical=True,
        complete=True,
        reversible=True,
        handles_disconnected=True,
        metric_admissible=True,
        max_n=98,
    ),
    Representation(
        key="adjacency",
        short="adjacency",
        long="adjacency matrix, upper triangle",
        tex="adjacency",
        colour="#888888",
        marker="P",
        family=Family.RAW_SERIALIZATION,
        reversible=True,
        handles_disconnected=True,
        metric_admissible=False,
        max_n=98,
    ),
    Representation(
        key="wl_subtree",
        short="WL subtree",
        long="Weisfeiler--Lehman subtree kernel ($h = 2$)",
        tex="WL subtree",
        colour="#EE7733",
        marker="X",
        family=Family.KERNEL,
        canonical=True,
        complete=False,
        reversible=False,
        handles_disconnected=True,
        metric_admissible=True,
        bit_countable=False,
        max_n=98,
    ),
    Representation(
        key="size_null",
        short=r"$|n_i-n_j|$ null",
        long=r"node-count difference (trivial baseline)",
        tex=r"$|n_i-n_j|$ baseline",
        colour="#555555",
        marker="*",
        family=Family.BASELINE,
        bit_countable=False,
        max_n=98,
    ),
)

BY_KEY: Final[dict[str, Representation]] = {r.key: r for r in REPRESENTATIONS}

#: Registry order, every row. **Tables iterate this.**
ORDER: Final[tuple[str, ...]] = tuple(r.key for r in REPRESENTATIONS)

#: Draw order. **Figures iterate this**, and it is :data:`ORDER` minus the
#: arms flagged ``in_figures=False``. A figure filters this rather than
#: defining its own tuple.
FIGURE_ORDER: Final[tuple[str, ...]] = tuple(r.key for r in REPRESENTATIONS if r.in_figures)

#: The reference arm every paired comparison is taken against.
REFERENCE_KEY: Final[str] = "isalgraph_pruned"


# ---------------------------------------------------------------------------
# GED reference styling. exact is not part of the bracket and is drawn as a
# different kind of object, never as a third bound.
# ---------------------------------------------------------------------------

REFERENCE_DISPLAY: Final[dict[str, str]] = {
    "exact": "exact GED",
    "lb": "LB (BRANCH-FAST)",
    "ub": "UB (IPFP)",
}

#: ``lb`` is drawn lighter than ``ub`` because every bracket-dependent verdict
#: in T-06 must show both, and a reader must be able to tell which is which
#: without the legend.
REFERENCE_ALPHA: Final[dict[str, float]] = {"exact": 1.0, "lb": ALPHA_MUTED, "ub": 0.95}
REFERENCE_LINESTYLE: Final[dict[str, Any]] = {"exact": "-", "lb": (0, (4, 1.5)), "ub": "-"}


# ---------------------------------------------------------------------------
# Accessors -- figures and tables call these, never the dicts directly.
# ---------------------------------------------------------------------------


def present(keys: object, *, drawable_only: bool = True) -> tuple[Representation, ...]:
    """Return the registered representations among *keys*, in draw order.

    Args:
        keys: Any iterable of backend names.
        drawable_only: Drop arms flagged ``in_figures=False``. This is the
            default because every caller is a figure; pass ``False`` to get
            the full registry order restricted to *keys*.

    Returns:
        Registered representations, ordered by :data:`ORDER`. Unknown names
        are dropped silently -- a figure must never invent a style for a
        backend the registry does not know, because that is how a colour
        starts drifting between figures.
    """
    have = set(keys)  # type: ignore[call-overload]
    return tuple(
        r for r in REPRESENTATIONS if r.key in have and (r.in_figures or not drawable_only)
    )


def tex_name(key: str) -> str:
    """Return the LaTeX name for *key*.

    Args:
        key: Backend name.

    Returns:
        The registered LaTeX name, or *key* verbatim when unregistered.
    """
    rep = BY_KEY.get(key)
    return rep.tex if rep else key


#: Families a figure foregrounds by default. Everything outside this set is
#: drawn dashed and half-transparent, so the eye reads the comparison the
#: sentence is scoped to -- ``T-06-FRAMING.md`` 2 scopes the compactness claim
#: to the canonical codes -- without any row leaving the figure. Muting is not
#: hiding: every representation is still drawn, still labeled, and appears at
#: equal weight in the tables.
PRIMARY_FAMILIES: Final[frozenset[Family]] = frozenset({Family.CANONICAL_CODE})

#: Alpha and dash pattern applied to a de-emphasized series.
SECONDARY_ALPHA: Final[float] = 0.5
SECONDARY_DASH: Final[Any] = (0, (4, 1.6))


def line_kwargs(
    rep: Representation,
    reference: str | None = None,
    *,
    primary: frozenset[Family] | None = None,
) -> dict[str, Any]:
    """Return the plot keyword arguments for one series.

    Args:
        rep: The representation being drawn.
        reference: ``exact``, ``lb``, ``ub``, or ``None`` when the series
            carries no GED reference (a bit-count series, for instance).
        primary: Families to foreground. Defaults to
            :data:`PRIMARY_FAMILIES`; pass an empty set to draw every series
            at full weight.

    Returns:
        Keyword arguments for ``Axes.plot`` / ``Axes.errorbar``.
    """
    foreground = PRIMARY_FAMILIES if primary is None else primary
    lead = rep.family in foreground or rep.is_ours
    style: dict[str, Any] = {
        "color": rep.colour,
        "linestyle": (
            (rep.linestyle if reference is None else REFERENCE_LINESTYLE[reference])
            if lead
            else SECONDARY_DASH
        ),
        "linewidth": rep.linewidth if lead else LW_COMPETITOR * 0.85,
        "marker": rep.marker if lead else "None",
        "markersize": MS_POINT + (0.8 if rep.is_ours else 0.0),
        "alpha": (1.0 if reference is None else REFERENCE_ALPHA[reference])
        if lead
        else SECONDARY_ALPHA,
        "zorder": rep.zorder if lead else 2,
    }
    if rep.is_ours:
        style["markeredgecolor"] = "black"
        style["markeredgewidth"] = 0.35
    return style


def label(rep: Representation, reference: str | None = None) -> str:
    """Return the legend label for one series.

    Args:
        rep: The representation being drawn.
        reference: GED reference, or ``None``.

    Returns:
        The label. The reference suffix uses a middle dot so a legend can be
        de-duplicated on ``label.split(" · ")[0]``.
    """
    if reference in (None, "exact"):
        return rep.short
    return f"{rep.short} · {reference.upper()}"


# ---------------------------------------------------------------------------
# Figure lifecycle
# ---------------------------------------------------------------------------


def style() -> Any:
    """Apply the shared IEEE rcParams and return the pyplot module.

    Returns:
        ``matplotlib.pyplot``, styled. Imported inside the function so this
        module stays importable with matplotlib absent, matching the
        ``isalgraph.viz`` contract.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from benchmarks import plotting_styles

    plotting_styles.apply_ieee_style()
    return plt


def text_width() -> float:
    """Return the manuscript's text width in inches.

    **Not** :data:`IEEE_TEXT_WIDTH_INCHES`. That constant is the 7.0 in
    two-column IEEE print area, and this manuscript is single-column
    letterpaper with 4.8 cm side margins, which leaves 4.7382 in. Point
    sizes inside a figure are absolute, so a figure rendered at 7.0 and
    placed at ``\\textwidth`` has every label scaled by 0.674 -- 5.5 pt
    labels reach the page at 3.7 pt. Rendering at the placement width is
    the only way a declared size is the printed size.
    """
    from benchmarks import plotting_styles

    return float(plotting_styles.PATREC_TEXT_WIDTH_INCHES)


def column_width() -> float:
    """Return the IEEE single-column figure width in inches."""
    from benchmarks import plotting_styles

    return float(plotting_styles.IEEE_COLUMN_WIDTH_INCHES)


def save(fig: Figure, basepath: str | Path) -> list[Path]:
    """Write *fig* as PDF and PNG under *basepath*.

    Args:
        fig: The figure.
        basepath: Output path without an extension.

    Returns:
        The paths written.
    """
    from benchmarks import plotting_styles

    return [Path(p) for p in plotting_styles.save_figure(fig, str(basepath))]


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


def shared_legend(
    fig: Figure,
    handles: list[Any],
    labels: list[str],
    *,
    ncol: int = 5,
    y: float = -0.015,
) -> None:
    """Place one legend below the whole figure.

    Args:
        fig: The figure.
        handles: Legend handles.
        labels: Legend labels, same length as *handles*.
        ncol: Maximum columns.
        y: Figure-fraction height the legend hangs from. Pass the axes'
            bottom minus the x-label height when the axes do not run to the
            canvas edge, or the gap between the two reads as an error.
    """
    fig.legend(
        handles,
        labels,
        fontsize=FS_LEGEND,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=min(len(labels), ncol),
        frameon=False,
    )


__all__ = [
    "ALPHA_BAND",
    "ALPHA_MUTED",
    "BY_KEY",
    "CLAIM_A_SCOPE_N",
    "EXACT_CEILING",
    "FDR_Q",
    "FS_ANNOT",
    "FS_LABEL",
    "FS_LEGEND",
    "FS_PANEL",
    "FS_SUPTITLE",
    "FS_TICK",
    "FS_TITLE",
    "GRID_ALPHA",
    "GRID_LW",
    "INK_FLOOR",
    "INK_RULE",
    "INK_SEPARATOR",
    "INK_THIN",
    "LW_COMPETITOR",
    "LW_REFERENCE",
    "MS_POINT",
    "MS_SIGNIFICANT",
    "ORDER",
    "PRIMARY_FAMILIES",
    "REFERENCE_ALPHA",
    "REFERENCE_DISPLAY",
    "REFERENCE_KEY",
    "REFERENCE_LINESTYLE",
    "REPRESENTATIONS",
    "SECONDARY_ALPHA",
    "SECONDARY_DASH",
    "Family",
    "Representation",
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
