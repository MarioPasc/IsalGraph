"""Resolve the IAM and GraphEdX source roots from a single ``--source`` value.

The cohort's two source trees have been reorganised twice, and after the second
move they no longer share a parent: IAM lives under
``<source>/APPROX_GED/datasets/IAM_Database/extracted`` while GraphEdX lives
under ``<source>/GED_PRECOMPUTED/datasets``. Both ``export_graphs.py`` and
``cohort_audit.py`` were written when one ``<source>`` spanned both, so each
resolved exactly one hardcoded layout and **no value of ``--source`` could make
either module load both trees**. That left decision 22's tracked cohort
reproduction unable to re-derive the LINUX and AIDS-GraphEdX rows -- eight red
tests, discovered by T-05 (see ``T-05-design.md`` amendment 1, finding 3).

The fix is a resolver rather than a corrected constant, because a constant is
what broke twice. Each root is located by probing a short list of known layouts
for a marker directory, newest first, with an environment override ahead of all
of them. A failure names every candidate it tried, so the next reorganisation
produces a diagnosis instead of a ``FileNotFoundError`` from inside a loader.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["DataRootError", "resolve_graphedx_root", "resolve_iam_root"]

IAM_ENV = "ISALGRAPH_IAM_ROOT"
GRAPHEDX_ENV = "ISALGRAPH_GRAPHEDX_ROOT"

# Relative to ``source_dir``, newest layout first. A candidate wins when one of
# its marker directories exists, so an empty or half-populated tree is skipped
# rather than selected and then failing deeper in a loader.
_IAM_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("APPROX_GED", "datasets", "IAM_Database", "extracted"),
    ("IAM_Database", "extracted"),
    ("datasets", "IAM_Database", "extracted"),
    (),
)
_IAM_MARKERS = ("Letter", "GREC", "Protein")

_GRAPHEDX_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("GED_PRECOMPUTED", "datasets"),
    ("GED_PRECOMPUTED",),
    ("datasets",),
    (),
)
_GRAPHEDX_MARKERS = ("LINUX", "AIDS")


class DataRootError(FileNotFoundError):
    """Raised when neither the override nor any known layout locates a root."""


def _probe(
    source_dir: Path,
    candidates: tuple[tuple[str, ...], ...],
    markers: tuple[str, ...],
    env_var: str,
    what: str,
) -> Path:
    """Return the first candidate root holding one of ``markers``.

    Args:
        source_dir: The ``--source`` value.
        candidates: Path fragments relative to ``source_dir``, newest first.
        markers: Subdirectory names any one of which identifies the root.
        env_var: Environment variable that overrides the probe entirely.
        what: Human-readable tree name, used in the error message.

    Returns:
        The resolved directory.

    Raises:
        DataRootError: If the override is set but unusable, or if no candidate
            holds a marker. The message lists every path tried.
    """
    override = os.environ.get(env_var)
    if override:
        root = Path(override).expanduser()
        if any((root / marker).is_dir() for marker in markers):
            return root
        raise DataRootError(
            f"{env_var}={override!r} does not hold any of {markers} for the {what} tree"
        )

    tried: list[Path] = []
    for fragments in candidates:
        root = source_dir.joinpath(*fragments)
        tried.append(root)
        if any((root / marker).is_dir() for marker in markers):
            return root

    listing = "\n  ".join(str(path) for path in tried)
    raise DataRootError(
        f"cannot locate the {what} tree under {source_dir}.\n"
        f"Looked for any of {markers} in:\n  {listing}\n"
        f"Set {env_var} to point at it directly."
    )


def resolve_iam_root(source_dir: Path | str) -> Path:
    """Return the directory holding ``Letter/``, ``GREC/``, ``Protein/`` and friends.

    Args:
        source_dir: The ``--source`` value.

    Returns:
        The IAM ``extracted`` directory.

    Raises:
        DataRootError: If no known layout locates it.
    """
    return _probe(Path(source_dir), _IAM_CANDIDATES, _IAM_MARKERS, IAM_ENV, "IAM")


def resolve_graphedx_root(source_dir: Path | str) -> Path:
    """Return the directory holding ``LINUX/`` and ``AIDS/``.

    Args:
        source_dir: The ``--source`` value.

    Returns:
        The GraphEdX dataset directory.

    Raises:
        DataRootError: If no known layout locates it.
    """
    return _probe(
        Path(source_dir), _GRAPHEDX_CANDIDATES, _GRAPHEDX_MARKERS, GRAPHEDX_ENV, "GraphEdX"
    )
