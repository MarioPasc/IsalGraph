"""Compose graphical abstract from Panel A and Panel B.

Generates both the inline composite and ensures individual panels
are saved for Inkscape workflow.

Elsevier specifications:
  - Minimum: 531 x 1328 pixels (h x w)
  - Aspect ratio ~1:2.5
  - Readable at 5 x 13 cm at 96 dpi
"""

from __future__ import annotations

import argparse
import logging
import os

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from benchmarks.plotting_styles import apply_ieee_style, save_figure

from .panel_a_encoding import draw_panel_a, generate_panel_a
from .panel_b_results import draw_panel_b, generate_panel_b

logger = logging.getLogger(__name__)

# Elsevier dimensions
_FIG_WIDTH_INCHES = 4.5
_FIG_HEIGHT_INCHES = 1.8
_DPI = 300
_MIN_WIDTH_PX = 1328
_MIN_HEIGHT_PX = 531


def _verify_dimensions(path: str) -> bool:
    """Verify output meets Elsevier minimum dimensions.

    Args:
        path: Path to PNG file.

    Returns:
        True if dimensions are sufficient.
    """
    try:
        from PIL import Image

        img = Image.open(path)
        w, h = img.size
        ok = w >= _MIN_WIDTH_PX and h >= _MIN_HEIGHT_PX
        if ok:
            logger.info("Graphical abstract: %d x %d px — OK", w, h)
        else:
            logger.warning(
                "Graphical abstract: %d x %d px — BELOW minimum %d x %d",
                w,
                h,
                _MIN_WIDTH_PX,
                _MIN_HEIGHT_PX,
            )
        return ok
    except ImportError:
        logger.warning("Pillow not installed, skipping dimension verification")
        return True


def generate_composite(
    output_dir: str,
    run_dir: str | None = None,
) -> str:
    """Generate full composite graphical abstract.

    Draws both panels in a single figure at Elsevier dimensions.

    Args:
        output_dir: Directory to save output files.
        run_dir: Pipeline run directory for Panel B data.

    Returns:
        Base path of saved composite (without extension).
    """
    fig = plt.figure(figsize=(_FIG_WIDTH_INCHES, _FIG_HEIGHT_INCHES))

    gs = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[0.38, 0.62],
        wspace=0.06,
        left=0.02,
        right=0.98,
        top=0.96,
        bottom=0.04,
    )

    draw_panel_a(fig, gs[0])
    draw_panel_b(fig, gs[1], run_dir=run_dir)

    # Vertical separator between panels
    panel_a_bb = gs[0].get_position(fig)
    panel_b_bb = gs[1].get_position(fig)
    sep_x = (panel_a_bb.x1 + panel_b_bb.x0) / 2
    fig.add_artist(
        plt.Line2D(
            [sep_x, sep_x],
            [0.15, 0.85],
            transform=fig.transFigure,
            color="0.75",
            linewidth=0.5,
            zorder=10,
        )
    )

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "graphical_abstract")

    # Save in all formats
    save_figure(fig, path, formats=("pdf", "svg", "png"))

    # Also TIFF for Elsevier
    tiff_path = f"{path}.tiff"
    fig.savefig(
        tiff_path,
        format="tiff",
        dpi=_DPI,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    logger.info("TIFF saved: %s", tiff_path)

    plt.close(fig)
    logger.info("Composite graphical abstract saved: %s", path)

    # Verify dimensions
    png_path = f"{path}.png"
    if os.path.exists(png_path):
        _verify_dimensions(png_path)

    return path


def generate_all(
    output_dir: str,
    run_dir: str | None = None,
) -> None:
    """Generate individual panels, sub-elements, and composite.

    Args:
        output_dir: Directory to save all output files.
        run_dir: Pipeline run directory for Panel B data.
    """
    from .panel_a_encoding import generate_all_standalone as gen_a_parts
    from .panel_b_results import generate_all_standalone as gen_b_parts

    # Individual panels (for Inkscape workflow)
    generate_panel_a(output_dir)
    generate_panel_b(output_dir, run_dir=run_dir)

    # Individual sub-elements (for fine-grained Inkscape composition)
    gen_a_parts(output_dir)
    gen_b_parts(output_dir, run_dir=run_dir)

    # Composite
    generate_composite(output_dir, run_dir=run_dir)

    logger.info("All graphical abstract files generated in %s", output_dir)


# =========================================================================
# CLI
# =========================================================================


def main() -> None:
    """CLI entry point for graphical abstract generation."""
    parser = argparse.ArgumentParser(description="Generate graphical abstract for IsalGraph paper.")
    parser.add_argument(
        "--output-dir",
        default="paper_figures/graphical_abstract",
        help="Output directory for all figures.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Pipeline run directory (contains message_length/, computational/).",
    )
    parser.add_argument(
        "--panels-only",
        action="store_true",
        help="Generate only individual panels (skip composite).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    apply_ieee_style()

    if args.panels_only:
        generate_panel_a(args.output_dir)
        generate_panel_b(args.output_dir, run_dir=args.run_dir)
    else:
        generate_all(args.output_dir, run_dir=args.run_dir)


if __name__ == "__main__":
    main()
