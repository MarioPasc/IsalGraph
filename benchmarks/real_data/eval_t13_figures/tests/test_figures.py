"""Every figure must build from the fixture, in both formats, and the ceiling
figure must refuse to draw an impossible point.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.real_data.eval_t13_figures import (
    data,
    design,
    fig_cost_law,
    fig_operations,
    fig_resolution,
)
from benchmarks.real_data.eval_t13_figures.tests import (
    committed_counters,
    committed_records,
    write_records,
)

pytest.importorskip("matplotlib")


@pytest.fixture(scope="module")
def records() -> data.Records:
    """The committed fixture campaign."""
    return data.load([committed_records()])


@pytest.fixture(scope="module")
def counters() -> data.CounterRecords:
    """The committed counter fixture."""
    return data.load_counters([committed_counters()])


def _suffixes(paths: list[Path]) -> set[str]:
    return {p.suffix for p in paths}


# ---------------------------------------------------------------------------
# fig_cost_law
# ---------------------------------------------------------------------------


def test_cost_law_writes_pdf_and_png(records: data.Records, tmp_path: Path) -> None:
    """The main-text figure, in both formats."""
    saved, summary = fig_cost_law.figure(records, tmp_path / fig_cost_law.STEM)
    assert _suffixes(saved) == {".pdf", ".png"}
    assert all(p.exists() and p.stat().st_size > 0 for p in saved)
    assert len(summary["ladders"]) == 3


def test_cost_law_draws_every_censored_point_as_censored(
    records: data.Records, tmp_path: Path
) -> None:
    """A censored point gets an arrow; an ordinary point does not."""
    _, summary = fig_cost_law.figure(records, tmp_path / "arrows")
    n_censored = sum(1 for row in records.rows if data.is_censored(row))
    assert summary["n_censored_drawn"] == n_censored
    assert n_censored > 0


def test_cost_law_reproduces_the_pilot_contrast(records: data.Records, tmp_path: Path) -> None:
    """The whole point of the figure, read back off its own summary.

    On the ``n = 33`` spider ladder the unpruned arm must be nearly flat and
    the pruned arm must move by two orders of magnitude, which is
    ``T-13-design.md`` 6.3's sentence in numbers.
    """
    _, summary = fig_cost_law.figure(records, tmp_path / "contrast")
    cell = summary["ladders"]["spider_ladder|33|spider_k8"]
    assert cell["variation"]["isalgraph_exhaustive"] == pytest.approx(1.10, abs=0.02)
    assert cell["variation"]["isalgraph_pruned"] == pytest.approx(126.9, rel=0.02)
    assert cell["variation"]["graph6"] == pytest.approx(1.0)


def test_cost_law_excludes_censored_rows_from_the_line(
    records: data.Records, tmp_path: Path
) -> None:
    """A cap-censored 4 ms row must not become the fastest point of the curve."""
    ladder = next(
        lad for lad in data.ladders(records) if lad.key == ("spider_ladder", 33, "spider_k8")
    )
    _, ys, n_rungs = fig_cost_law._rung_medians(ladder, "min_dfs")
    assert n_rungs == 3
    assert min(ys) > 0.005


def test_cost_law_raises_when_no_ladder_has_a_measured_aut(tmp_path: Path) -> None:
    """With ``symmetry_available=false`` the cost law has no abscissa."""
    from benchmarks.real_data.eval_t13_figures.tests import build_rows

    rows = build_rows()
    for row in rows:
        row["log10_aut"] = None
    write_records(tmp_path, rows=rows)
    records = data.load([tmp_path])
    with pytest.raises(fig_cost_law.CostLawFigureError, match="no abscissa"):
        fig_cost_law.figure(records, tmp_path / "nope")


def test_cost_law_caption_quotes_only_measured_numbers(
    records: data.Records, tmp_path: Path
) -> None:
    """Every number in the caption comes out of the summary."""
    _, summary = fig_cost_law.figure(records, tmp_path / "caption")
    text = fig_cost_law.caption(summary)
    assert "spider" in text
    assert "right-censored" in text
    assert "default" in text


def test_cost_law_cli(tmp_path: Path) -> None:
    """The documented command line runs end to end."""
    out = tmp_path / "out"
    code = fig_cost_law.main(["--records", str(committed_records()), "--out-dir", str(out)])
    assert code == 0
    assert (out / f"{fig_cost_law.STEM}_default.pdf").exists()
    assert (out / f"{fig_cost_law.STEM}_default.png").exists()
    assert (out / f"{fig_cost_law.STEM}_default.caption.tex").exists()


# ---------------------------------------------------------------------------
# fig_resolution
# ---------------------------------------------------------------------------


def test_resolution_writes_pdf_and_png(records: data.Records, tmp_path: Path) -> None:
    """The resolution figure, in both formats."""
    rows = data.resolutions(records)
    saved, summary = fig_resolution.figure(rows, tmp_path / fig_resolution.STEM)
    assert _suffixes(saved) == {".pdf", ".png"}
    assert summary["n_graphs"] == len(rows)


def test_resolution_raises_above_the_invariance_ceiling(
    records: data.Records, tmp_path: Path
) -> None:
    """The acceptance criterion: a point above ``y = x`` refutes Proposition 1."""
    rows = list(data.resolutions(records))
    rows[0] = replace(rows[0], n_wl_classes=rows[0].n_orbits + 1)
    with pytest.raises(fig_resolution.CeilingViolationError, match="ceiling violated"):
        fig_resolution.figure(rows, tmp_path / "impossible")
    assert not (tmp_path / "impossible.pdf").exists()


def test_check_ceiling_names_every_offending_field(records: data.Records) -> None:
    """A violation must be localisable without a re-run."""
    rows = list(data.resolutions(records))
    rows[0] = replace(
        rows[0],
        n_wl_classes=rows[0].n_orbits + 2,
        n_triplet_classes=rows[0].n_orbits + 3,
    )
    with pytest.raises(fig_resolution.CeilingViolationError) as excinfo:
        fig_resolution.check_ceiling(rows)
    assert "n_wl_classes" in str(excinfo.value)
    assert "n_triplet_classes" in str(excinfo.value)


def test_resolution_cli(tmp_path: Path) -> None:
    """The documented command line runs end to end."""
    out = tmp_path / "out"
    code = fig_resolution.main(["--records", str(committed_records()), "--out-dir", str(out)])
    assert code == 0
    assert (out / f"{fig_resolution.STEM}.pdf").exists()
    assert (out / f"{fig_resolution.STEM}.png").exists()


# ---------------------------------------------------------------------------
# fig_operations
# ---------------------------------------------------------------------------


def test_operations_writes_pdf_and_png(counters: data.CounterRecords, tmp_path: Path) -> None:
    """The supplementary figure, in both formats."""
    saved, summary = fig_operations.figure(counters, tmp_path / fig_operations.STEM)
    assert _suffixes(saved) == {".pdf", ".png"}
    assert summary["n_rows"] == len(counters.rows)
    assert set(summary["panels"]) == {p.field for p in fig_operations.PANELS}


def test_operations_omits_greedy_from_the_backtracking_panel(
    counters: data.CounterRecords, tmp_path: Path
) -> None:
    """Greedy has zero frames by construction and a log axis cannot show zero."""
    _, summary = fig_operations.figure(counters, tmp_path / "panels")
    panel = summary["panels"]["backtrack_nodes"]
    assert panel["greedy_min:backtrack_nodes"] == 0
    assert panel["canonical:backtrack_nodes"] > 0


def test_operations_raises_on_an_unstyled_encoder(
    counters: data.CounterRecords, tmp_path: Path
) -> None:
    """An unstyled encoder would vanish from every panel with no error."""
    rows = [*counters.rows, {**dict(counters.rows[0]), "encoder": "future_encoder"}]
    mutated = data.CounterRecords(rows=tuple(rows), paths=counters.paths)
    with pytest.raises(fig_operations.OperationsFigureError, match="unstyled encoder"):
        fig_operations.figure(mutated, tmp_path / "unstyled")


def test_counter_fixture_stays_under_its_own_bounds(
    counters: data.CounterRecords,
) -> None:
    """A fixture whose curve crosses its bound reads as a refuted derivation.

    Section 2.1's bounds are asymptotic, so a *real* count may legitimately
    exceed one by a constant factor and this package does not assert
    otherwise.  The fixture is a different matter: it exists to show what the
    panel looks like, and a synthetic curve above its own bound would send a
    reader hunting for a defect in the derivation instead of reading the
    figure.
    """
    for row in counters.rows:
        n, m = int(row["n"]), float(row["m"])
        assert row["pair_trials"] <= fig_operations._bound_value("mn2", n, m)
        assert row["pointer_steps"] <= fig_operations._bound_value("mn3", n, m)
        assert row["neighbour_checks"] <= fig_operations._bound_value("m_delta", n, m)
        assert row["search_leaves"] <= fig_operations._bound_value("leaves", n, m)


def test_operations_bounds_are_the_ones_the_derivation_states() -> None:
    """``mn^2``, ``mn^3``, ``m(n-1)`` and ``n(n-1)^{n-1}``, and nothing invented."""
    assert fig_operations._bound_value("mn2", 4, 6.0) == 96.0
    assert fig_operations._bound_value("mn3", 4, 6.0) == 384.0
    assert fig_operations._bound_value("m_delta", 4, 6.0) == 18.0
    assert fig_operations._bound_value("leaves", 4, 6.0) == 4 * 27
    with pytest.raises(fig_operations.OperationsFigureError):
        fig_operations._bound_value("invented", 4, 6.0)


def test_operations_cli(tmp_path: Path) -> None:
    """The documented command line runs end to end."""
    out = tmp_path / "out"
    code = fig_operations.main(["--counters", str(committed_counters()), "--out-dir", str(out)])
    assert code == 0
    assert (out / f"{fig_operations.STEM}.pdf").exists()
    assert (out / f"{fig_operations.STEM}.png").exists()


# ---------------------------------------------------------------------------
# Cross-figure
# ---------------------------------------------------------------------------


def test_a_figure_refuses_an_unregistered_arm(records: data.Records, tmp_path: Path) -> None:
    """The T-06 trap, closed: an unknown key raises instead of vanishing."""
    rows = [dict(r) for r in records.rows]
    for row in rows:
        if row["representation"] == "graph6":
            row["representation"] = "future_backend"
    mutated = data.Records(
        rows=tuple(rows),
        headers=records.headers,
        build_hash=records.build_hash,
        run_ids=records.run_ids,
        paths=records.paths,
    )
    with pytest.raises(design.UnknownRepresentationError, match="future_backend"):
        fig_cost_law.figure(mutated, tmp_path / "unregistered")
