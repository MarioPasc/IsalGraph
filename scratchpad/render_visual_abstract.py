"""Render the Elsevier graphical abstract, at several row budgets.

Usage::

    python render_visual_abstract.py <outdir> [variant ...]

Variants: ``full`` (every step), ``four`` (steps 1, 4, 5, 6), ``three``
(steps 1, 4, 6), ``pruned`` (the pruned tree, every step).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import isalgraph  # noqa: E402
from isalgraph.viz.figures import RUNNING_EXAMPLE_START, build_example_graph  # noqa: E402
from isalgraph.viz.visual_abstract import (  # noqa: E402
    AbstractLayout,
    apply_abstract_style,
    pruned_visual_abstract_figure,
    save_abstract,
    visual_abstract_figure,
)
from isalgraph.viz.worked_example import RUNNING_EXAMPLE_POSITIONS  # noqa: E402

VARIANTS: dict[str, dict[str, object]] = {
    "default": {},
    "three": {"max_rows": 3},
    "full": {"max_rows": None},
}


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    wanted = sys.argv[2:] or list(VARIANTS)
    out.mkdir(parents=True, exist_ok=True)

    assert isalgraph.engine() == "cpp", isalgraph.engine()
    apply_abstract_style()
    graph = build_example_graph()

    for name in wanted:
        if name == "pruned":
            lay = AbstractLayout()
            fig = pruned_visual_abstract_figure(
                graph, positions=RUNNING_EXAMPLE_POSITIONS, layout=lay
            )
        else:
            lay = AbstractLayout(**VARIANTS[name])  # type: ignore[arg-type]
            fig = visual_abstract_figure(
                graph,
                start_node=RUNNING_EXAMPLE_START,
                positions=RUNNING_EXAMPLE_POSITIONS,
                layout=lay,
            )
        for path in save_abstract(fig, out / f"fig_visual_abstract_{name}"):
            print(path)
        plt.close(fig)

    lay = AbstractLayout()
    print(
        f"canvas {lay.fig_width:.2f} x {lay.fig_height:.2f} in "
        f"= {round(lay.fig_width * 300)} x {round(lay.fig_height * 300)} px at 300 dpi, "
        f"ratio {lay.fig_width / lay.fig_height:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
