"""Two lazy registries: one for representations, one for distances.

IsalHG's ``iso_backends/registry.py`` idiom, verbatim in shape.  Backend and
metric modules register themselves at import time; ``_LAZY_MODULES`` keeps an
optional third-party library out of the import path until something asks for
the backend that needs it.

Two registries rather than one, so that T-04a's grid is literally
``product(available_backends(), available_metrics())`` and a new cell cannot
be forgotten.
"""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Callable
from typing import Any

from isalgraph.competitors.base import (
    Capability,
    DistanceMetric,
    ReprBackend,
    VectorBackend,
)
from isalgraph.errors import BackendNotFoundError, BackendUnavailableError

AnyBackend = ReprBackend | VectorBackend
# ``Any`` rather than ``Comparable``: ``DistanceMetric`` is invariant in its
# parameter, so a registry typed over the union would reject every concrete
# metric. The narrowing that matters happens at the call site, via
# ``_applicable``, which rejects a (backend, metric) pairing before anything
# is computed.
AnyMetric = DistanceMetric[Any]

#: Factories take keyword arguments so a backend with a frozen hyperparameter
#: -- WL's ``h``, which is 2 and must not be tuned on rho -- can still be
#: instantiated at another value for a reproduction check without a mutable
#: attribute that anything else could reach.
_BACKENDS: dict[str, Callable[..., AnyBackend]] = {}
_METRICS: dict[str, Callable[..., AnyMetric]] = {}

#: Backend name -> module that registers it on import.
_LAZY_MODULES: dict[str, str] = {
    "adjacency": "isalgraph.competitors.backends.adjacency",
    "graph6": "isalgraph.competitors.backends.graph6",
    "sparse6": "isalgraph.competitors.backends.sparse6",
    "nauty_graph6": "isalgraph.competitors.backends.nauty",
    "sparse6_nauty": "isalgraph.competitors.backends.nauty",
    "agm_cam": "isalgraph.competitors.backends.agm",
    "min_dfs": "isalgraph.competitors.backends.min_dfs",
    "wl_subtree": "isalgraph.competitors.backends.wl",
    "isalgraph_pruned": "isalgraph.competitors.backends.isalgraph_ref",
    "isalgraph_canonical": "isalgraph.competitors.backends.isalgraph_ref",
    "isalgraph_exhaustive": "isalgraph.competitors.backends.isalgraph_ref",
    "size_null": "isalgraph.competitors.backends.size_null",
}

#: Metric name -> module that registers it on import.
_LAZY_METRIC_MODULES: dict[str, str] = {
    "levenshtein": "isalgraph.competitors.metrics.levenshtein",
    "levenshtein_char": "isalgraph.competitors.metrics.levenshtein",
    "hamming": "isalgraph.competitors.metrics.hamming",
    "padded_hamming": "isalgraph.competitors.metrics.hamming",
    "kernel": "isalgraph.competitors.metrics.kernel",
    "size_null": "isalgraph.competitors.metrics.size_null",
}


def register_backend(name: str, factory: Callable[..., AnyBackend]) -> None:
    """Register *factory* under *name*, overwriting any prior entry."""
    _BACKENDS[name] = factory


def register_metric(name: str, factory: Callable[..., AnyMetric]) -> None:
    """Register *factory* under *name*, overwriting any prior entry."""
    _METRICS[name] = factory


def _ensure_imported(name: str, table: dict[str, str], registry: dict[str, Any]) -> None:
    """Import the module registering *name*, suppressing ImportError.

    Suppression is correct here and only here: a backend whose optional
    dependency is missing must stay absent from ``available_*`` yet still be
    *nameable*, so that ``get_backend`` can raise
    :class:`BackendUnavailableError` with a real diagnosis.
    """
    if name not in registry and name in table:
        with contextlib.suppress(ImportError):
            importlib.import_module(table[name])


def get_backend(name: str, **kwargs: object) -> AnyBackend:
    """Return a fresh instance of the backend registered under *name*.

    ``size_null`` is returned only when named explicitly -- it never arrives
    through a listing that a caller then iterates blindly.  See
    :func:`available_backends`.

    Raises:
        BackendNotFoundError: if *name* is unknown after lazy import.
        BackendUnavailableError: if the backend is registered but its
            third-party library is missing.  **Never a silent degrade.**
    """
    _ensure_imported(name, _LAZY_MODULES, _BACKENDS)
    if name not in _BACKENDS:
        if name in _LAZY_MODULES:
            module = _LAZY_MODULES[name]
            try:
                importlib.import_module(module)
            except ImportError as exc:
                # Two very different faults arrive here as the same exception,
                # and conflating them sends the reader hunting for a missing
                # package that is installed. `No module named '<module>'` means
                # the backend module itself is absent -- in a git worktree that
                # means the editable install redirected `isalgraph` to the main
                # checkout, so the file you just wrote is not the file that ran.
                missing = getattr(exc, "name", "") or ""
                if missing == module or missing.startswith(f"{module}."):
                    raise BackendNotFoundError(
                        f"competitor backend {name!r} maps to {module!r}, which does "
                        f"not exist. If you are in a git worktree, `import isalgraph` "
                        f"resolves to the MAIN checkout unless sitecustomize.py drops "
                        f"the ScikitBuildRedirectingFinder -- your file is not the one "
                        f"being imported"
                    ) from exc
                raise BackendUnavailableError(
                    f"competitor backend {name!r} requires {missing or 'a dependency'}, "
                    f"which is not installed: {exc}"
                ) from exc
        raise BackendNotFoundError(
            f"competitor backend {name!r} is not registered (known: {sorted(_BACKENDS)})"
        )
    backend = _BACKENDS[name](**kwargs)
    if not type(backend).is_available():
        raise BackendUnavailableError(
            f"competitor backend {name!r} is registered but its third-party "
            f"library is not installed"
        )
    return backend


def get_metric(name: str, **kwargs: object) -> AnyMetric:
    """Return a fresh instance of the metric registered under *name*.

    Raises:
        BackendNotFoundError: if *name* is unknown after lazy import.
    """
    _ensure_imported(name, _LAZY_METRIC_MODULES, _METRICS)
    if name not in _METRICS:
        raise BackendNotFoundError(
            f"competitor metric {name!r} is not registered (known: {sorted(_METRICS)})"
        )
    return _METRICS[name](**kwargs)


def get_repr_backend(name: str, **kwargs: object) -> ReprBackend:
    """:func:`get_backend`, narrowed to a serialisation.

    Raises:
        BackendNotFoundError: if *name* resolves to the one
            :class:`VectorBackend`.  WL has no ``encode`` and no bit count,
            and a caller that wanted one asked the wrong question.
    """
    backend = get_backend(name, **kwargs)
    if not isinstance(backend, ReprBackend):
        raise BackendNotFoundError(
            f"{name!r} is a VectorBackend: it has no encode() and no bit count. "
            f"Use get_vector_backend()"
        )
    return backend


def get_vector_backend(name: str, **kwargs: object) -> VectorBackend:
    """:func:`get_backend`, narrowed to a feature-vector representation."""
    backend = get_backend(name, **kwargs)
    if not isinstance(backend, VectorBackend):
        raise BackendNotFoundError(f"{name!r} is a ReprBackend, not a VectorBackend")
    return backend


def _import_all(table: dict[str, str]) -> None:
    for name in tuple(table):
        with contextlib.suppress(ImportError):
            importlib.import_module(table[name])


def registered_backends() -> tuple[str, ...]:
    """Every registered backend name, including unavailable ones and baselines."""
    _import_all(_LAZY_MODULES)
    return tuple(sorted(_BACKENDS))


def available_backends(*, include_baseline: bool = False) -> tuple[str, ...]:
    """Backend names whose dependencies actually import.

    Args:
        include_baseline: when ``False`` (the default) backends carrying
            :attr:`Capability.BASELINE` are omitted.  ``size_null`` is a
            registered backend so that the null column falls out of every
            table by construction, but it is **not** a competitor and must
            not arrive in a listing something then treats as one.
    """
    _import_all(_LAZY_MODULES)
    out = []
    for name, factory in _BACKENDS.items():
        backend = factory()
        if not type(backend).is_available():
            continue
        if not include_baseline and Capability.BASELINE in backend.capabilities:
            continue
        out.append(name)
    return tuple(sorted(out))


def available_metrics() -> tuple[str, ...]:
    """Every registered metric name."""
    _import_all(_LAZY_METRIC_MODULES)
    return tuple(sorted(_METRICS))


def unavailable_backends() -> dict[str, str]:
    """Map each unimportable backend name to the reason it is unavailable.

    Diagnostics only.  A backend appearing here is why a grid column is
    blank, and the reason belongs in the printed table.
    """
    out: dict[str, str] = {}
    for name, module in _LAZY_MODULES.items():
        try:
            importlib.import_module(module)
        except ImportError as exc:
            out[name] = str(exc)
            continue
        if name in _BACKENDS and not type(_BACKENDS[name]()).is_available():
            out[name] = "registered, third-party library not importable"
    return out


def _reset_for_testing() -> None:
    """Clear both registries.  Tests only."""
    _BACKENDS.clear()
    _METRICS.clear()


__all__ = [
    "AnyBackend",
    "AnyMetric",
    "available_backends",
    "available_metrics",
    "get_backend",
    "get_metric",
    "get_repr_backend",
    "get_vector_backend",
    "register_backend",
    "register_metric",
    "registered_backends",
    "unavailable_backends",
]
