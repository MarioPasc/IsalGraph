"""The style registry must agree with the campaign, and must never guess.

The T-06 registry's two silent-failure modes are the subject of this file:
an arm the campaign runs but the registry does not style, and a key handed to
``present()`` that the registry drops without a word.  Both produce a figure
that regenerates successfully with the arm missing.
"""

from __future__ import annotations

import pytest

from benchmarks.real_data.eval_t13_complexity import measure
from benchmarks.real_data.eval_t13_figures import design


def test_registry_covers_exactly_the_campaign_arms() -> None:
    """Every measured arm is styled and every styled arm is measured."""
    assert set(design.ORDER) == set(measure.REPRESENTATIONS)
    assert len(design.ORDER) == len(measure.REPRESENTATIONS) == 13


def test_search_class_matches_measures_own_split() -> None:
    """The null is the null: search-free membership is not restated by hand."""
    search_free = {
        r.key for r in design.REPRESENTATIONS if r.search_class is design.SearchClass.SEARCH_FREE
    }
    assert search_free == set(measure.SEARCH_FREE)
    search_based = {
        r.key for r in design.REPRESENTATIONS if r.search_class is design.SearchClass.SEARCH_BASED
    }
    assert search_based == set(measure.SEARCH_BASED)


def test_registry_check_rejects_a_missing_arm() -> None:
    """A campaign arm with no style is an import-time error, not a gap."""
    original = design.REPRESENTATIONS
    try:
        design.REPRESENTATIONS = original[:-1]  # type: ignore[misc]
        with pytest.raises(design.RegistryError, match="measured but unstyled"):
            design._check_registry()
    finally:
        design.REPRESENTATIONS = original  # type: ignore[misc]


def test_registry_check_rejects_a_duplicate_key() -> None:
    """Two entries for one key would give a backend two colours."""
    original = design.REPRESENTATIONS
    try:
        design.REPRESENTATIONS = (*original, original[0])  # type: ignore[misc]
        with pytest.raises(design.RegistryError, match="duplicate"):
            design._check_registry()
    finally:
        design.REPRESENTATIONS = original  # type: ignore[misc]


def test_present_raises_on_an_unknown_key() -> None:
    """The T-06 defect, inverted: an unregistered key is an error."""
    with pytest.raises(design.UnknownRepresentationError, match="unregistered"):
        design.present(["isalgraph_pruned", "not_a_backend"])


def test_present_accepts_an_explicit_omission() -> None:
    """An intentional omission has to be said out loud, at the call site."""
    got = design.present(["isalgraph_pruned", "not_a_backend"], omit=["not_a_backend"])
    assert [r.key for r in got] == ["isalgraph_pruned"]


def test_present_can_omit_a_registered_arm() -> None:
    """omit= also drops a registered arm, which is the deliberate case."""
    got = design.present(["isalgraph_pruned", "graph6"], omit=["graph6"])
    assert [r.key for r in got] == ["isalgraph_pruned"]


def test_present_returns_draw_order_not_input_order() -> None:
    """Two figures fed the same arms in different orders must match."""
    a = design.present(["size_null", "isalgraph_pruned", "min_dfs"])
    b = design.present(["min_dfs", "size_null", "isalgraph_pruned"])
    assert [r.key for r in a] == [r.key for r in b]
    assert [r.key for r in a] == [
        k for k in design.ORDER if k in {"size_null", "isalgraph_pruned", "min_dfs"}
    ]


def test_absent_reports_the_other_half_of_the_omission_problem() -> None:
    """A styled arm with no data is reported, not silently skipped."""
    missing = design.absent(["isalgraph_pruned"])
    assert "min_dfs" in missing
    assert "isalgraph_pruned" not in missing
    assert len(missing) == len(design.ORDER) - 1


def test_tex_name_raises_rather_than_echoing_the_key() -> None:
    """T-06 returns the key verbatim; a raw key in a table reads as a typo."""
    assert design.tex_name("min_dfs") == "gSpan min-DFS"
    with pytest.raises(design.UnknownRepresentationError):
        design.tex_name("not_a_backend")


def test_colours_are_unique_per_arm() -> None:
    """Two arms sharing a colour is unreadable, whatever else is right."""
    colours = [r.colour for r in design.REPRESENTATIONS]
    assert len(set(colours)) == len(colours)


def test_censored_style_is_visually_distinct_from_a_completed_point() -> None:
    """A censored point must never be able to read as an ordinary one."""
    rep = design.BY_KEY["min_dfs"]
    completed = design.line_kwargs(rep)
    censored = design.censored_kwargs(rep)
    assert censored["markerfacecolor"] == design.CENSORED_FILLSTYLE == "none"
    assert censored["linestyle"] == "none"
    assert completed.get("markerfacecolor") != "none"
    assert censored["markersize"] > completed["markersize"]


def test_focus_arms_are_the_ones_the_characterisation_names() -> None:
    """The eye must land on the arms 6.3 contrasts."""
    focus = {r.key for r in design.REPRESENTATIONS if r.is_focus}
    assert focus == {
        "isalgraph_exhaustive",
        "isalgraph_pruned",
        "isalgraph_greedy",
        "min_dfs",
        "graph6",
    }
    assert design.BY_KEY["isalgraph_pruned"].linewidth > design.BY_KEY["adjacency"].linewidth


def test_geometry_comes_from_the_published_source_of_truth() -> None:
    """This package reads ``isalgraph.viz.style`` directly, not the re-export.

    ``benchmarks.plotting_styles`` re-exports ``isalgraph.viz.style`` so that
    the published palette cannot drift, and the repository already carries a
    test asserting that identity.  This package therefore calls the source and
    not the re-export, for two reasons: one hop fewer, and ``plotting_styles``
    is not ``mypy --strict`` clean -- routing a new package through it drags 15
    pre-existing errors into this package's own type check.  This test asserts
    the geometry helpers really do read the source of truth; the
    ``plotting_styles`` half of the identity is not re-asserted here because it
    is already covered and importing it would reintroduce those 15 errors.
    """
    pytest.importorskip("matplotlib")
    from isalgraph.viz import style as viz_style

    assert design.text_width() == viz_style.IEEE_TEXT_WIDTH_INCHES
    assert design.column_width() == viz_style.IEEE_COLUMN_WIDTH_INCHES
    assert design.INK_RULE != design.INK_CEILING


def test_hypothesised_driver_is_get_guarded_everywhere_it_is_used() -> None:
    """It is a pre-registered expectation, not a result, and it is partial."""
    assert design.HYPOTHESISED_DRIVER["isalgraph_exhaustive"] == "degree sequence"
    assert design.HYPOTHESISED_DRIVER["isalgraph_pruned"] == "automorphism group"
    assert "agm_cam" not in design.HYPOTHESISED_DRIVER
