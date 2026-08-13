"""Tests for the T-27 bound bake-off views.

Two contracts dominate: the module must import with no third-party
drawing library present, and a view must paint on a caller-supplied
``Axes`` rather than create a figure. Both are the reasons
``isalgraph.viz`` exists as a package instead of as helper functions
inside a benchmark script.

Import note for worktrees. ``isalgraph`` is installed editable through a
scikit-build-core meta-path finder pinned to the main checkout, and a
``MetaPathFinder`` takes precedence over ``sys.path``, so ``PYTHONPATH``
alone cannot make a new file under a worktree's ``src/`` visible.
:func:`_ensure_module_visible` extends ``isalgraph.viz.__path__`` with
this file's own tree, which is a no-op in the merged main checkout.
"""

from __future__ import annotations

import ast
import importlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import pytest

MODULE = "isalgraph.viz.bound_bakeoff_view"
BLOCKED = ("matplotlib", "networkx", "igraph", "numpy", "scipy", "pandas")
_VIZ_DIR = Path(__file__).resolve().parents[2] / "src" / "isalgraph" / "viz"
_SOURCE = _VIZ_DIR / "bound_bakeoff_view.py"


def _ensure_module_visible() -> None:
    """Add this checkout's ``viz`` directory to the package search path."""
    package = importlib.import_module("isalgraph.viz")
    path = str(_VIZ_DIR)
    if path not in package.__path__:
        package.__path__.insert(0, path)


@pytest.fixture(scope="module")
def view() -> ModuleType:
    """Import the view module under test."""
    _ensure_module_visible()
    return importlib.import_module(MODULE)


@pytest.fixture
def axes() -> Iterator[object]:
    """Yield a caller-owned Axes, closing its figure afterwards."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        yield ax
    finally:
        plt.close(fig)


def _panels(view: ModuleType) -> object:
    """Build a small two-dataset, three-method panel set."""
    methods = ("BRANCH", "BRANCH_FAST", "HED")
    curves = tuple(
        view.DatasetCurves(
            dataset=dataset,
            curves=tuple(
                view.ErrorCurve(
                    method=method,
                    n_values=(2, 3, 4, 5),
                    mean=(0.1 * k, 0.2 * k, 0.3 * k, 0.4 * k),
                    q25=(0.0, 0.1 * k, 0.2 * k, 0.3 * k),
                    q75=(0.2 * k, 0.3 * k, 0.4 * k, 0.5 * k),
                    counts=(10, 40, 80, 5),
                )
                for k, method in enumerate(methods, start=1)
            ),
        )
        for dataset in ("linux", "iam_letter_low")
    )
    forest = tuple(
        view.ForestEntry(
            dataset=dataset,
            method=method,
            mean=0.2 * k,
            ci_low=0.2 * k - 0.05,
            ci_high=0.2 * k + 0.05,
            winner=(method == "BRANCH_FAST"),
        )
        for dataset in ("linux", "iam_letter_low")
        for k, method in enumerate(methods, start=1)
    )
    return view.BakeoffPanels(end="lower", dataset_curves=curves, forest=forest, methods=methods)


# ---------------------------------------------------------------------------
# Import isolation
# ---------------------------------------------------------------------------


class _BlockingFinder:
    """Meta-path finder that refuses to import the blocked packages."""

    def __init__(self, blocked: tuple[str, ...]) -> None:
        self._blocked = blocked

    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        """Raise for a blocked root package; defer everything else."""
        if fullname.split(".")[0] in self._blocked:
            raise ImportError(f"{fullname} is blocked by the dependency-free import test")
        return None


@contextmanager
def _blocked_imports(names: tuple[str, ...]) -> Iterator[None]:
    """Block *names* and every ``isalgraph`` module for the duration."""
    saved = dict(sys.modules)
    for mod in list(sys.modules):
        root = mod.split(".")[0]
        if (root in names or root == "isalgraph") and mod != "isalgraph.core._native":
            del sys.modules[mod]
    finder = _BlockingFinder(names)
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        sys.modules.clear()
        sys.modules.update(saved)


def test_module_imports_with_no_drawing_library_present() -> None:
    """The whole point of keeping third-party imports inside function bodies."""
    with _blocked_imports(BLOCKED):
        _ensure_module_visible()
        module = importlib.import_module(MODULE)
        assert module.method_palette(("BRANCH", "STAR"))["BRANCH"].startswith("#")
        assert "N = 5" in module.CD_CAVEAT


def test_no_module_scope_third_party_imports() -> None:
    """Static check, so a future edit cannot regress the import contract."""
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    assert not found & set(BLOCKED)


# ---------------------------------------------------------------------------
# Dataclass contracts
# ---------------------------------------------------------------------------


def test_error_curve_rejects_ragged_sequences(view: ModuleType) -> None:
    """A ragged curve would misalign the ribbon against the line silently."""
    with pytest.raises(ValueError, match="share a length"):
        view.ErrorCurve(
            method="BRANCH",
            n_values=(2, 3),
            mean=(0.1,),
            q25=(0.0,),
            q75=(0.2,),
            counts=(1,),
        )


def test_method_palette_is_stable_and_position_keyed(view: ModuleType) -> None:
    """Colour assignment follows the caller's order, so panels agree."""
    methods = ("BRANCH", "BRANCH_FAST", "STAR")
    palette = view.method_palette(methods)
    assert palette == view.method_palette(methods)
    assert len(set(palette.values())) == 3
    assert view.method_palette(("STAR",))["STAR"] == palette["BRANCH"]


def test_method_marker_cycles_and_falls_back(view: ModuleType) -> None:
    """Markers carry the series in greyscale; an unknown method is not fatal."""
    methods = ("A", "B", "C")
    assert view.method_marker(methods, "A") != view.method_marker(methods, "B")
    assert view.method_marker(methods, "ZZZ") == "o"


# ---------------------------------------------------------------------------
# Drawing -- a view paints, it never creates a figure
# ---------------------------------------------------------------------------


def test_draw_error_vs_n_paints_on_the_supplied_axes(view: ModuleType, axes: object) -> None:
    """One line and one ribbon per method, on the caller's Axes."""
    import matplotlib.pyplot as plt

    before = len(plt.get_fignums())
    panels = _panels(view)
    view.draw_error_vs_n(axes, panels.dataset_curves[0], methods=panels.methods)
    assert len(plt.get_fignums()) == before
    assert len(axes.lines) == 3
    assert len(axes.collections) == 3
    assert [line.get_label() for line in axes.lines] == list(panels.methods)


def test_draw_error_vs_n_drops_thin_bins(view: ModuleType, axes: object) -> None:
    """A bin of five pairs must not draw a spike that reads like a trend."""
    panels = _panels(view)
    view.draw_error_vs_n(axes, panels.dataset_curves[0], methods=panels.methods, min_count=20)
    assert all(len(line.get_xdata()) == 2 for line in axes.lines)


def test_draw_error_vs_n_tolerates_an_empty_dataset(view: ModuleType, axes: object) -> None:
    """A dataset with no cells must render an empty facet, not raise."""
    view.draw_error_vs_n(axes, view.DatasetCurves(dataset="empty"), methods=())
    assert len(axes.lines) == 0
    assert axes.get_title() == "empty"


def test_draw_forest_groups_by_dataset_and_marks_the_winner(view: ModuleType, axes: object) -> None:
    """Winner rows get the diamond marker; groups get a vertical gap."""
    panels = _panels(view)
    view.draw_forest(axes, panels.forest, methods=panels.methods)
    labels = [t.get_text() for t in axes.get_yticklabels()]
    assert labels == [e.method for e in panels.forest]
    markers = [line.get_marker() for line in axes.lines if line.get_linestyle() == "None"]
    assert markers.count("D") == 2
    positions = list(axes.get_yticks())
    assert positions[3] - positions[2] > positions[1] - positions[0]


def test_draw_forest_inverts_the_y_axis(view: ModuleType, axes: object) -> None:
    """Rows read top to bottom in the order given, as a forest plot should."""
    panels = _panels(view)
    view.draw_forest(axes, panels.forest, methods=panels.methods)
    bottom, top = axes.get_ylim()
    assert bottom > top


def test_draw_critical_difference_puts_rank_one_on_the_left(view: ModuleType, axes: object) -> None:
    """Demsar's convention; an inverted axis here would reverse the reading."""
    cd = view.CriticalDifference(
        end="lower",
        methods=("BRANCH", "BRANCH_FAST", "STAR"),
        average_ranks=(1.2, 1.6, 3.0),
        cd=1.0,
        n_datasets=5,
        friedman_p=0.01,
        cliques=((0, 1),),
    )
    view.draw_critical_difference(axes, cd)
    left, right = axes.get_xlim()
    assert left < right
    assert left < 1.2
    assert right > 3.0


def test_draw_critical_difference_tolerates_an_empty_summary(
    view: ModuleType, axes: object
) -> None:
    """A non-evaluable omnibus blanks the panel rather than raising."""
    cd = view.CriticalDifference(
        end="upper", methods=(), average_ranks=(), cd=float("nan"), n_datasets=0
    )
    view.draw_critical_difference(axes, cd)
    assert len(axes.lines) == 0
    assert not axes.axison


# ---------------------------------------------------------------------------
# Figure composition
# ---------------------------------------------------------------------------


def test_bound_bakeoff_figure_has_one_facet_per_dataset_plus_the_forest(
    view: ModuleType,
) -> None:
    """CONTRACTS §8: panel (a) faceted by dataset, panel (b) spanning."""
    plt = pytest.importorskip("matplotlib.pyplot")
    fig = view.bound_bakeoff_figure(_panels(view))
    try:
        assert len(fig.axes) == 3
        assert fig.axes[-1].get_xlabel().startswith("mean relative error")
        assert fig.legends
    finally:
        plt.close(fig)


def test_bound_bakeoff_figure_writes_pdf_and_png(view: ModuleType, tmp_path: Path) -> None:
    """Both formats, through ``save_figure``, so the suffix logic is shared."""
    plt = pytest.importorskip("matplotlib.pyplot")
    from isalgraph.viz.style import save_figure

    fig = view.bound_bakeoff_figure(_panels(view), title="lower bound")
    try:
        paths = save_figure(fig, tmp_path / "T27_lower_bound")
    finally:
        plt.close(fig)
    assert [p.name for p in paths] == ["T27_lower_bound.pdf", "T27_lower_bound.png"]
    assert all(p.stat().st_size > 0 for p in paths)


def test_critical_difference_figure_carries_both_caveats(view: ModuleType) -> None:
    """N = 5 and the Letter non-independence both belong in the caption."""
    plt = pytest.importorskip("matplotlib.pyplot")
    cd = view.CriticalDifference(
        end="lower",
        methods=("BRANCH", "STAR"),
        average_ranks=(1.0, 2.0),
        cd=1.5,
        n_datasets=5,
        friedman_p=0.2,
        cliques=((0, 1),),
    )
    fig = view.critical_difference_figure(cd)
    try:
        caption = " ".join(t.get_text() for t in fig.texts)
        assert "N = 5" in caption
        assert "3 + 1 + 1" in caption
        assert "not a hypothesis test" in caption
    finally:
        plt.close(fig)
