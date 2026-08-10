"""Name-keyed registry for drawing backends.

Backend modules register themselves at import time via
:func:`register_backend`. :func:`get_backend` lazily imports the module
on first request, so an optional third-party library stays out of the
import path until something actually asks for it.
"""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Callable

from isalgraph.errors import VizBackendNotFoundError, VizBackendUnavailableError
from isalgraph.viz.base import GraphVizBackend

#: The backend used when a caller does not name one. Plain matplotlib
#: primitives suffice for a simple graph, so the default carries no
#: third-party requirement beyond matplotlib itself.
DEFAULT_BACKEND: str = "matplotlib"

_BACKENDS: dict[str, Callable[[], GraphVizBackend]] = {}

#: Backend name -> module that registers it on import.
_LAZY_MODULES: dict[str, str] = {
    "matplotlib": "isalgraph.viz.backends.matplotlib_backend",
    "networkx": "isalgraph.viz.backends.networkx_backend",
    "igraph": "isalgraph.viz.backends.igraph_backend",
}


def register_backend(name: str, factory: Callable[[], GraphVizBackend]) -> None:
    """Register *factory* under *name*, overwriting any prior entry."""
    _BACKENDS[name] = factory


def _ensure_imported(name: str) -> None:
    """Import the module that registers *name*, suppressing ImportError."""
    if name not in _BACKENDS and name in _LAZY_MODULES:
        with contextlib.suppress(ImportError):
            importlib.import_module(_LAZY_MODULES[name])


def get_backend(name: str, *, require_available: bool = True) -> GraphVizBackend:
    """Return a fresh instance of the backend registered under *name*.

    Args:
        name: Registry key.
        require_available: When ``True``, verify the backend's
            third-party library imports before returning it, so the
            failure surfaces here rather than mid-draw.

    Returns:
        A new backend instance.

    Raises:
        VizBackendNotFoundError: If *name* is unknown after lazy import.
        VizBackendUnavailableError: If the backend is registered but its
            library is missing and *require_available* is ``True``.
    """
    _ensure_imported(name)
    if name not in _BACKENDS:
        raise VizBackendNotFoundError(
            f"viz backend {name!r} is not registered (known: {sorted(_BACKENDS)})"
        )
    backend = _BACKENDS[name]()
    if require_available and not type(backend).is_available():
        raise VizBackendUnavailableError(
            f"viz backend {name!r} is registered but its drawing library is not installed"
        )
    return backend


def available_backends() -> tuple[str, ...]:
    """Return the sorted names of backends that can actually draw.

    Every known module is imported first, then each registered backend is
    filtered through :meth:`GraphVizBackend.is_available`. The filter is
    the fix for an upstream defect: IsalHG detects the third-party
    library only at draw time, so its ``available_backends()`` lists
    backends whose library is absent and callers discover the problem
    one traceback later.
    """
    for name in _LAZY_MODULES:
        _ensure_imported(name)
    return tuple(
        sorted(name for name, factory in _BACKENDS.items() if type(factory()).is_available())
    )


def registered_backends() -> tuple[str, ...]:
    """Return every registered name, including unavailable ones.

    Useful for diagnostics: comparing this against
    :func:`available_backends` shows which libraries are missing.
    """
    for name in _LAZY_MODULES:
        _ensure_imported(name)
    return tuple(sorted(_BACKENDS))


__all__ = [
    "DEFAULT_BACKEND",
    "available_backends",
    "get_backend",
    "register_backend",
    "registered_backends",
]
