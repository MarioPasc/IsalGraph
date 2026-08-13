"""Tests for the IAM / GraphEdX source-root resolver.

The resolver exists because the cohort's two source trees were reorganised
twice and, after the second move, stopped sharing a parent -- at which point no
single ``--source`` could make ``export_graphs.py`` or ``cohort_audit.py`` load
both, and eight real-data tests went red without anyone noticing (T-05 design
note, amendment 1, finding 3).

So the behaviour under test is not "finds the path" but "keeps finding it after
the next reorganisation, and says what it tried when it cannot".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.eval_setup.data_roots import (
    GRAPHEDX_ENV,
    IAM_ENV,
    DataRootError,
    resolve_graphedx_root,
    resolve_iam_root,
)


def _mk(root: Path, *fragments: str, marker: str) -> Path:
    """Create ``root/fragments/marker`` and return the directory above it."""
    base = root.joinpath(*fragments)
    (base / marker).mkdir(parents=True)
    return base


# --------------------------------------------------------------------------- #
# Layouts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "fragments",
    [
        ("APPROX_GED", "datasets", "IAM_Database", "extracted"),  # today
        ("IAM_Database", "extracted"),  # the layout the frozen code assumed
        ("datasets", "IAM_Database", "extracted"),
        (),  # source_dir already IS the extracted dir
    ],
)
def test_every_known_iam_layout_resolves(tmp_path: Path, fragments: tuple[str, ...]) -> None:
    """All four layouts this project has actually shipped must resolve."""
    expected = _mk(tmp_path, *fragments, marker="Letter")
    assert resolve_iam_root(tmp_path) == expected


@pytest.mark.parametrize(
    "fragments",
    [("GED_PRECOMPUTED", "datasets"), ("GED_PRECOMPUTED",), ("datasets",), ()],
)
def test_every_known_graphedx_layout_resolves(tmp_path: Path, fragments: tuple[str, ...]) -> None:
    expected = _mk(tmp_path, *fragments, marker="LINUX")
    assert resolve_graphedx_root(tmp_path) == expected


def test_the_newest_layout_wins_when_both_are_present(tmp_path: Path) -> None:
    """A leftover legacy tree must not shadow the current one.

    Both existed simultaneously during the migration, and picking the stale one
    would silently audit an out-of-date cohort.
    """
    _mk(tmp_path, "IAM_Database", "extracted", marker="Letter")
    current = _mk(tmp_path, "APPROX_GED", "datasets", "IAM_Database", "extracted", marker="Letter")
    assert resolve_iam_root(tmp_path) == current


def test_a_directory_without_its_marker_is_skipped(tmp_path: Path) -> None:
    """An empty scaffold must not be selected and then fail inside a loader."""
    (tmp_path / "APPROX_GED" / "datasets" / "IAM_Database" / "extracted").mkdir(parents=True)
    expected = _mk(tmp_path, "IAM_Database", "extracted", marker="GREC")
    assert resolve_iam_root(tmp_path) == expected


# --------------------------------------------------------------------------- #
# Override and failure
# --------------------------------------------------------------------------- #


def test_the_environment_override_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mk(tmp_path, "IAM_Database", "extracted", marker="Letter")
    elsewhere = _mk(tmp_path / "other", marker="Letter")
    monkeypatch.setenv(IAM_ENV, str(elsewhere))
    assert resolve_iam_root(tmp_path) == elsewhere


def test_a_wrong_override_fails_loudly_rather_than_falling_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Silently ignoring a set override would resolve a tree the user did not ask for."""
    _mk(tmp_path, "IAM_Database", "extracted", marker="Letter")
    monkeypatch.setenv(IAM_ENV, str(tmp_path / "nope"))
    with pytest.raises(DataRootError, match=IAM_ENV):
        resolve_iam_root(tmp_path)


def test_failure_names_every_candidate_it_tried(tmp_path: Path) -> None:
    """The next reorganisation should produce a diagnosis, not a puzzle."""
    with pytest.raises(DataRootError) as excinfo:
        resolve_graphedx_root(tmp_path)
    message = str(excinfo.value)
    assert "GED_PRECOMPUTED/datasets" in message
    assert GRAPHEDX_ENV in message
    assert "LINUX" in message


def test_the_two_roots_need_not_share_a_parent(tmp_path: Path) -> None:
    """The exact condition that broke the frozen modules.

    IAM under ``APPROX_GED/datasets`` and GraphEdX under ``GED_PRECOMPUTED``
    means no single ``<source>/<fixed fragment>`` reaches both.
    """
    iam = _mk(tmp_path, "APPROX_GED", "datasets", "IAM_Database", "extracted", marker="Letter")
    gedx = _mk(tmp_path, "GED_PRECOMPUTED", "datasets", marker="LINUX")
    assert resolve_iam_root(tmp_path) == iam
    assert resolve_graphedx_root(tmp_path) == gedx
    assert iam.parent != gedx.parent
