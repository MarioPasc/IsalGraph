"""``import isalgraph.viz`` must succeed with no drawing library installed.

Enforced by installing a meta-path finder that raises ``ImportError`` for
matplotlib, networkx and igraph, dropping any already-imported copies,
and then importing the package. If any module in ``isalgraph.viz`` grew a
module-scope third-party import, this fails.
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

BLOCKED = ("matplotlib", "networkx", "igraph", "numpy", "scipy", "pandas")

_VIZ_DIR = Path(__file__).resolve().parents[2] / "src" / "isalgraph" / "viz"


class _BlockingFinder:
    """Meta-path finder that refuses to import the blocked packages."""

    def __init__(self, blocked: tuple[str, ...]) -> None:
        self._blocked = blocked

    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        """Raise for a blocked root package; defer everything else."""
        root = fullname.split(".")[0]
        if root in self._blocked:
            raise ImportError(f"{root} is blocked by the dependency-free import test")
        return None


@contextmanager
def _blocked_imports(names: tuple[str, ...]) -> Iterator[None]:
    """Block *names* and every ``isalgraph`` module for the duration."""
    saved = dict(sys.modules)
    for mod in list(sys.modules):
        root = mod.split(".")[0]
        if root in names or root == "isalgraph":
            # Never evict the compiled extension. Purging it makes the next
            # `import isalgraph` re-run nanobind's module init, which registers
            # every bound type a second time and emits
            # "nanobind: type 'Cdll' was already registered!". That warning is
            # a test artifact, but it also leaves two distinct type objects for
            # the same C++ class, so isinstance() across them would fail.
            # Keeping it loaded costs the test nothing: it is a C extension and
            # cannot import matplotlib.
            if mod == "isalgraph.core._native":
                continue
            del sys.modules[mod]
    finder = _BlockingFinder(names)
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        sys.modules.clear()
        sys.modules.update(saved)


def test_blocking_finder_actually_blocks() -> None:
    """Guard the guard: the fixture must really prevent the import."""
    with _blocked_imports(BLOCKED), pytest.raises(ImportError):
        importlib.import_module("matplotlib")


def test_viz_imports_without_any_drawing_library() -> None:
    """The package and every pure module must import with the libraries gone."""
    with _blocked_imports(BLOCKED):
        viz: ModuleType = importlib.import_module("isalgraph.viz")
        assert viz.DEFAULT_BACKEND == "matplotlib"
        for name in (
            "base",
            "style",
            "layout",
            "registry",
            "graph_view",
            "cdll_view",
            "instruction_view",
            "composite",
            "trace_io",
            "alignment_view",
            "nx_view",
            "search_tree",
            "figures",
        ):
            importlib.import_module(f"isalgraph.viz.{name}")


def test_available_backends_is_empty_without_matplotlib() -> None:
    """With matplotlib gone, no backend may claim to be available.

    This is the upstream defect the ``is_available`` hook fixes: IsalHG
    detects the library only when drawing, so it would still list them.
    """
    with _blocked_imports(BLOCKED):
        registry = importlib.import_module("isalgraph.viz.registry")
        assert registry.available_backends() == ()


def _module_scope_imports(path: Path) -> set[str]:
    """Return top-level package names imported at module scope in *path*."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


@pytest.mark.parametrize(
    "path",
    sorted(_VIZ_DIR.rglob("*.py")),
    ids=lambda p: str(p.relative_to(_VIZ_DIR)),
)
def test_no_module_scope_drawing_imports(path: Path) -> None:
    """Every third-party import in ``viz`` must sit inside a function body."""
    offenders = _module_scope_imports(path) & set(BLOCKED)
    assert not offenders, f"{path.name} imports {sorted(offenders)} at module scope"
