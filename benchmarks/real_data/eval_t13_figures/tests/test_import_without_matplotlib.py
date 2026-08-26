"""Every module of the package must import with no drawing library installed.

The contract ``isalgraph.viz`` already carries, applied here: third-party
imports live inside function bodies, so ``import ...eval_t13_figures.tables``
on a machine with no matplotlib is an ordinary import and not a crash.  The
practical value is that a table can be regenerated on a cluster login node, and
that a schema test does not drag in a plotting stack.

``eval_t06_figures.fig_ic`` does **not** hold this line -- it imports
``numpy`` and ``matplotlib.ticker`` at module scope -- which is why the check is
written against this package's own files rather than assumed from the sibling.
"""

from __future__ import annotations

import ast
import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator

#: Root packages the fixture makes unimportable.
BLOCKED = ("matplotlib", "networkx", "igraph", "numpy", "scipy", "pandas")

#: Every module of this package.
MODULES = (
    "design",
    "data",
    "fig_cost_law",
    "fig_resolution",
    "fig_operations",
    "tables",
)

_PACKAGE = "benchmarks.real_data.eval_t13_figures"
_PACKAGE_DIR = Path(__file__).resolve().parent.parent


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
    """Block *names* and drop every already-imported copy for the duration."""
    saved = dict(sys.modules)
    for mod in list(sys.modules):
        root = mod.split(".")[0]
        if root in names or mod.startswith("benchmarks"):
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


def test_every_module_imports_without_matplotlib() -> None:
    """The acceptance criterion."""
    with _blocked_imports(BLOCKED):
        importlib.import_module(_PACKAGE)
        for name in MODULES:
            importlib.import_module(f"{_PACKAGE}.{name}")


def _module_scope_imports(path: Path) -> set[str]:
    """Return top-level package names imported at module scope in *path*."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
        elif isinstance(node, ast.If):
            # ``if TYPE_CHECKING:`` blocks never execute at runtime.
            continue
    return found


@pytest.mark.parametrize("name", MODULES)
def test_no_module_scope_third_party_import(name: str) -> None:
    """A static check, so the failure names the offending file directly."""
    offenders = _module_scope_imports(_PACKAGE_DIR / f"{name}.py") & set(BLOCKED)
    assert not offenders, f"{name}.py imports {sorted(offenders)} at module scope"
