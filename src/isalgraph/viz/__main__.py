"""Regenerate the committed figures: ``python -m isalgraph.viz [output_dir]``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from isalgraph.viz.figures import render_all

_DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "docs" / "figures"


def main() -> None:
    """Parse arguments and render every figure."""
    parser = argparse.ArgumentParser(description="Regenerate IsalGraph paper figures.")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=str(_DEFAULT_OUTPUT),
        help="destination directory (default: docs/figures)",
    )
    parser.add_argument(
        "--formats",
        default="png",
        help="comma-separated output formats (default: png)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    written = render_all(args.output_dir, formats=tuple(args.formats.split(",")))
    for path in written:
        logging.getLogger(__name__).info("wrote %s", path)


if __name__ == "__main__":
    main()
