"""Visual style, palettes and figure saving.

Single source of truth for colours across the three views, so that the
instruction strip, the CDLL ring and the graph panel agree by
construction rather than by coincidence.

Provenance: the palettes and rcParams are lifted from
``benchmarks/plotting_styles.py``, which already defines this paper's
visual identity (Paul Tol colour-blind-safe qualitative sets, IEEE
column widths, Type-42 embedded fonts). They are reproduced here rather
than imported because ``isalgraph.viz`` is library code and must not
depend on ``benchmarks``; the dependency runs the other way, and
``benchmarks/plotting_styles.py`` re-exports from this module.

Colour semantics
----------------
IsalGraph's alphabet has nine symbols in four case-pairs, and the fact a
reader most needs from a rendered instruction is *which pointer acted*.
Two orthogonal channels encode this:

* **Hue encodes the operation.** Movement (``N``/``P``, ``n``/``p``) is
  blue-cyan, insertion (``V``/``v``) is green, connection (``C``/``c``)
  is red-rose, and ``W`` is neutral grey. Within a pair, the lowercase
  (secondary) variant is a lightness/saturation shift of its uppercase
  partner, so the pair reads as one family.
* **A stroke accent encodes the pointer.** Every instruction cell is
  outlined in :data:`POINTER_PALETTE`\\ ``[0]`` when the primary pointer
  acted and :data:`POINTER_PALETTE`\\ ``[1]`` when the secondary acted.
  Those are the same two colours used for the CDLL ring pointer arrows
  and for the pointer highlight in the graph view.

The second channel is what makes the correspondence *mechanical*: the
red ring arrow, the red-outlined ``V`` cell and the red-ringed graph
node are the same pointer, and a reader can verify that at print size
without consulting a legend. Hue alone could not carry it, because
:data:`INSTRUCTION_PALETTE` is already published in figures built from
``benchmarks/plotting_styles.py`` and recolouring it would make the
revision look like a different paper.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from matplotlib.figure import Figure
else:
    Figure = Any


# ---------------------------------------------------------------------------
# Paul Tol qualitative palettes (colour-blind safe)
# ---------------------------------------------------------------------------

PAUL_TOL_BRIGHT: dict[str, str] = {
    "blue": "#4477AA",
    "red": "#EE6677",
    "green": "#228833",
    "yellow": "#CCBB44",
    "cyan": "#66CCEE",
    "purple": "#AA3377",
    "grey": "#BBBBBB",
}

PAUL_TOL_HIGH_CONTRAST: dict[str, str] = {
    "blue": "#004488",
    "yellow": "#DDAA33",
    "red": "#BB5566",
}

PAUL_TOL_MUTED: tuple[str, ...] = (
    "#CC6677",  # rose
    "#332288",  # indigo
    "#DDCC77",  # sand
    "#117733",  # green
    "#88CCEE",  # cyan
    "#882255",  # wine
    "#44AA99",  # teal
    "#999933",  # olive
    "#AA4499",  # purple
)


# ---------------------------------------------------------------------------
# Instruction alphabet: hue encodes the operation, case encodes the pointer
# ---------------------------------------------------------------------------

INSTRUCTION_PALETTE: dict[str, str] = {
    "N": "#4477AA",  # blue        primary move next
    "P": "#004488",  # dark blue   primary move prev
    "n": "#66CCEE",  # cyan        secondary move next
    "p": "#88CCEE",  # light cyan  secondary move prev
    "V": "#228833",  # green       primary node insertion
    "v": "#117733",  # dark green  secondary node insertion
    "C": "#EE6677",  # red         primary-to-secondary edge
    "c": "#CC6677",  # rose        secondary-to-primary edge
    "W": "#BBBBBB",  # grey        no-op
}

#: Which pointer each instruction acts on. ``None`` for ``W``.
INSTRUCTION_POINTER: dict[str, int | None] = {
    "N": 0,
    "P": 0,
    "V": 0,
    "C": 0,
    "n": 1,
    "p": 1,
    "v": 1,
    "c": 1,
    "W": None,
}

#: Coarse operation class per instruction, used for legends.
INSTRUCTION_OPERATION: dict[str, str] = {
    "N": "move",
    "P": "move",
    "n": "move",
    "p": "move",
    "V": "insert",
    "v": "insert",
    "C": "connect",
    "c": "connect",
    "W": "noop",
}

#: ``POINTER_PALETTE[0]`` is the primary pointer, ``[1]`` the secondary.
#: These two hex values are load-bearing: they tie the ring arrows, the
#: instruction-cell stroke accent and the graph-view pointer ring
#: together. They match ``PRIMARY_COLOR`` / ``SECONDARY_COLOR`` in the
#: pre-existing ``eval_visualizations/cdll_drawing.py``.
POINTER_PALETTE: tuple[str, str] = ("#EE6677", "#4477AA")

#: Colour used where both pointers rest on the same node.
POINTER_OVERLAP_COLOR: str = "#AA77BB"

#: Highlight for a node or edge created by the current step.
NEW_ELEMENT_COLOR: str = "#CCBB44"

GRAYED_FACE: str = "#DDDDDD"
GRAYED_EDGE: str = "#999999"
GRAYED_ALPHA: float = 0.25
ACTIVE_ALPHA: float = 0.85
DEFAULT_NODE_COLOR: str = PAUL_TOL_MUTED[1]


# ---------------------------------------------------------------------------
# Three element states for the worked-example figures
# ---------------------------------------------------------------------------
#
# The step figures in ``composite`` distinguish two states, built and not
# built, and render "not built" as a filled grey disc. In print at column
# scale a filled grey disc competes with the coloured ones for attention,
# which is the opposite of what a ghost should do.
#
# The worked-example figures use the sibling projects' three-state
# convention instead (``IsalSR/viz/backends/matplotlib_dag.py``):
#
# * **ghost** -- white fill, dashed grey outline, grey glyph. Structure
#   that exists in the target but has not been reached yet. It recedes
#   because it carries no ink, not because it carries grey ink.
# * **accent** -- an unfilled halo around the element the current step
#   created. Additive, so it composes with any fill.
# * **dim** -- present but no longer the subject of the step, drawn at
#   :data:`CONSUMED_ALPHA`.
#
# Ghost and dim are deliberately not the same state. Collapsing them
# loses the distinction between "not there yet" and "there, already
# handled", which is the whole content of a step figure.

#: Fill of a ghosted element: white, so the element reads as an outline.
GHOST_FACE: str = "#FFFFFF"

#: Outline of a ghosted element.
GHOST_EDGE_COLOR: str = "#B9BEC7"

#: Glyph colour inside a ghosted node.
GHOST_TEXT_COLOR: str = "#8B939E"

#: Dash pattern of a ghost outline, as a matplotlib ``(offset, onoff)``
#: tuple. Dashes carry the "not yet real" reading on their own, so the
#: ghost needs no alpha and stays crisp when the figure is scaled down.
GHOST_DASH: tuple[float, tuple[float, float]] = (0.0, (2.6, 2.0))

#: Halo colour marking the element created by the current step.
ACCENT_COLOR: str = "#D9911F"

#: Alpha for elements that are present but already handled.
CONSUMED_ALPHA: float = 0.28


# ---------------------------------------------------------------------------
# Figure geometry (IEEE two-column)
# ---------------------------------------------------------------------------

IEEE_COLUMN_WIDTH_INCHES: float = 3.39
IEEE_COLUMN_GAP_INCHES: float = 0.24
IEEE_TEXT_WIDTH_INCHES: float = 7.0
IEEE_TEXT_HEIGHT_INCHES: float = 9.0

#: Text width of the *Pattern Recognition* manuscript, in inches:
#: ``letterpaper`` (8.5 in) less the 4.8 cm side margins declared at
#: ``main.tex:11``, so 8.5 - 2 * 1.89 = 4.72.
#:
#: **A paper figure must be drawn at this width, not at**
#: :data:`IEEE_TEXT_WIDTH_INCHES`. Drawing at 7.0 and placing at
#: ``width=\textwidth`` scales the file by 0.674, and point sizes scale with
#: it: the search-space schematic's 5.5 pt labels reached the page at 3.7 pt.
#: Font sizes inside a figure are absolute, so the only way they mean what
#: they say is for the render width to equal the placement width.
PATREC_TEXT_WIDTH_INCHES: float = 4.72


#: rcParams applied by :func:`apply_ieee_style`. ``pdf.fonttype`` and
#: ``ps.fonttype`` are pinned to 42 so glyphs embed as TrueType rather
#: than Type-3, which is what most publishers require.
BASE_RCPARAMS: dict[str, Any] = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "text.usetex": False,
    "font.size": 9.0,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.0,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.fontsize": 7.0,
    "legend.frameon": False,
    "lines.linewidth": 1.2,
    "grid.alpha": 0.4,
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_ieee_style() -> None:
    """Apply the IEEE rcParams in :data:`BASE_RCPARAMS` to the active session."""
    import matplotlib.pyplot as plt  # local import: matplotlib stays optional

    # rcParams is typed as a mapping over ~300 literal key names, which no
    # dict[str, Any] can satisfy; cast once rather than annotate each entry.
    params: MutableMapping[str, Any] = cast("MutableMapping[str, Any]", plt.rcParams)
    params.update(BASE_RCPARAMS)


def get_figure_size(
    width: str = "single",
    height_ratio: float | None = None,
) -> tuple[float, float]:
    """Return an ``(width, height)`` inch pair for an IEEE figure.

    Args:
        width: ``"single"`` for one column, ``"double"`` for full text
            width.
        height_ratio: Height as a fraction of the width. Defaults to
            ``0.75``.

    Returns:
        The ``(width_inches, height_inches)`` pair.

    Raises:
        ValueError: If *width* is neither ``"single"`` nor ``"double"``.
    """
    if width == "single":
        w = IEEE_COLUMN_WIDTH_INCHES
    elif width == "double":
        w = IEEE_TEXT_WIDTH_INCHES
    else:
        raise ValueError(f"width must be 'single' or 'double', got {width!r}")
    ratio = 0.75 if height_ratio is None else height_ratio
    return (w, min(w * ratio, IEEE_TEXT_HEIGHT_INCHES))


# ---------------------------------------------------------------------------
# Palettes derived from a graph
# ---------------------------------------------------------------------------


def _hexify(rgba: tuple[float, float, float, float]) -> str:
    r, g, b = rgba[:3]
    return f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"


def _categorical_colors(n: int, base: tuple[str, ...], cmap_name: str) -> list[str]:
    """Return *n* distinguishable colours, none of them grey.

    Uses *base* while it lasts, then samples *cmap_name*. Greys are
    excluded deliberately: :data:`GRAYED_FACE` marks "not yet built" in
    every step figure, so a categorical palette containing grey (as both
    ``tab20`` and ``tab20c`` do) would render some *active* elements
    indistinguishable from ghosted ones.
    """
    import matplotlib.pyplot as plt

    if n <= 0:
        return []
    if n <= len(base):
        return list(base[:n])
    cmap = plt.get_cmap(cmap_name)
    return [_hexify(cmap(i / max(1, n - 1))) for i in range(n)]


def build_node_palette(
    n_nodes: int,
    *,
    cmap_name: str = "turbo",
) -> dict[int, str]:
    """Return a ``{node_id: hex}`` mapping over ``range(n_nodes)``.

    IsalGraph nodes are unlabelled and indistinguishable, so there is no
    semantic class to colour by; nodes are coloured by id purely so a
    reader can track one node across the panels of a step figure.

    Args:
        n_nodes: Number of contiguous node ids to cover.
        cmap_name: Continuous colormap used beyond the categorical set.

    Returns:
        Mapping from node id to hex colour.
    """
    return dict(enumerate(_categorical_colors(n_nodes, PAUL_TOL_MUTED, cmap_name)))


def build_edge_palette(
    edges: tuple[tuple[int, int], ...],
    *,
    cmap_name: str = "viridis",
) -> dict[tuple[int, int], str]:
    """Return an ``{edge: hex}`` mapping keyed by normalised edge tuples.

    Draws from a rotation of the muted palette offset from the one
    :func:`build_node_palette` starts at, so an edge and the nodes it
    joins are unlikely to share a colour.

    Args:
        edges: Normalised, sorted edge tuples.
        cmap_name: Continuous colormap used beyond the categorical set.

    Returns:
        Mapping from edge tuple to hex colour.
    """
    offset = 4
    rotated = PAUL_TOL_MUTED[offset:] + PAUL_TOL_MUTED[:offset]
    return dict(zip(edges, _categorical_colors(len(edges), rotated, cmap_name), strict=True))


def color_for_instruction(instruction: str | None) -> str:
    """Return the fill colour for one instruction cell.

    Args:
        instruction: A character of ``Sigma``, or ``None``.

    Returns:
        Hex colour; :data:`GRAYED_FACE` for ``None`` or unknown symbols.
    """
    if instruction is None:
        return GRAYED_FACE
    return INSTRUCTION_PALETTE.get(instruction, GRAYED_FACE)


def pointer_accent(instruction: str | None) -> str:
    """Return the stroke colour encoding which pointer *instruction* acts on.

    Args:
        instruction: A character of ``Sigma``, or ``None``.

    Returns:
        :data:`POINTER_PALETTE`\\ ``[0]`` for primary instructions,
        ``[1]`` for secondary ones, :data:`GRAYED_EDGE` for ``W`` and for
        unknown symbols.
    """
    if instruction is None:
        return GRAYED_EDGE
    idx = INSTRUCTION_POINTER.get(instruction)
    if idx is None:
        return GRAYED_EDGE
    return POINTER_PALETTE[idx]


def render_colored_string(
    ax: Any,  # noqa: ANN401  -- matplotlib Axes; typed loosely to stay import-free
    string: str,
    x: float,
    y: float,
    fontsize: int = 8,
    mono: bool = True,
) -> None:
    """Render an instruction string with per-character colouring.

    Each character is drawn in its :data:`INSTRUCTION_PALETTE` colour;
    characters outside the alphabet are drawn black. Advance width is
    measured from the rendered glyph, so the result is correctly kerned
    for the active font.

    Args:
        ax: Target matplotlib axes.
        string: The instruction string.
        x: Starting x coordinate, in data coordinates.
        y: y coordinate, in data coordinates.
        fontsize: Point size for the characters.
        mono: Whether to render in a monospace family.
    """
    font_props: dict[str, object] = {"fontsize": fontsize}
    if mono:
        font_props["fontfamily"] = "monospace"

    renderer = ax.figure.canvas.get_renderer()
    offset_x = x
    for ch in string:
        color = INSTRUCTION_PALETTE.get(ch, "#000000")
        txt = ax.text(offset_x, y, ch, color=color, transform=ax.transData, **font_props)
        txt.draw(renderer)
        bbox = txt.get_window_extent(renderer=renderer)
        data_bbox = ax.transData.inverted().transform(bbox)
        offset_x += data_bbox[1][0] - data_bbox[0][0]


def save_figure(
    fig: Figure,
    basepath: str | Path,
    *,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 300,
) -> list[Path]:
    """Save *fig* as ``<basepath>.<fmt>`` for every requested format.

    The parent directory is created if absent.

    The suffix is appended with string concatenation rather than
    :meth:`pathlib.Path.with_suffix`, which is the bug this fixes:
    ``with_suffix`` replaces everything after the last dot in the stem,
    so a basepath of ``run_1.5`` silently becomes ``run_1.pdf`` and two
    runs overwrite each other.

    Args:
        fig: The figure to write.
        basepath: Output path without an extension.
        formats: Extensions to emit.
        dpi: Raster resolution for bitmap formats.

    Returns:
        The list of paths written.
    """
    bp = Path(basepath)
    bp.parent.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for fmt in formats:
        p = bp.parent / f"{bp.name}.{fmt}"
        fig.savefig(p, dpi=dpi)
        out.append(p)
    return out


__all__ = [
    "ACCENT_COLOR",
    "ACTIVE_ALPHA",
    "BASE_RCPARAMS",
    "CONSUMED_ALPHA",
    "DEFAULT_NODE_COLOR",
    "GHOST_DASH",
    "GHOST_EDGE_COLOR",
    "GHOST_FACE",
    "GHOST_TEXT_COLOR",
    "GRAYED_ALPHA",
    "GRAYED_EDGE",
    "GRAYED_FACE",
    "IEEE_COLUMN_GAP_INCHES",
    "IEEE_COLUMN_WIDTH_INCHES",
    "IEEE_TEXT_HEIGHT_INCHES",
    "IEEE_TEXT_WIDTH_INCHES",
    "INSTRUCTION_OPERATION",
    "INSTRUCTION_PALETTE",
    "INSTRUCTION_POINTER",
    "NEW_ELEMENT_COLOR",
    "PATREC_TEXT_WIDTH_INCHES",
    "PAUL_TOL_BRIGHT",
    "PAUL_TOL_HIGH_CONTRAST",
    "PAUL_TOL_MUTED",
    "POINTER_OVERLAP_COLOR",
    "POINTER_PALETTE",
    "apply_ieee_style",
    "build_edge_palette",
    "build_node_palette",
    "color_for_instruction",
    "get_figure_size",
    "pointer_accent",
    "render_colored_string",
    "save_figure",
]
