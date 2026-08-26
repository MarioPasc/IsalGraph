"""The LaTeX tables must compile-shape, escape their names, and say what they
used.

The scientific requirement these tests encode is the one ``T-13-design.md``
2.2 makes explicit: the fitted exponent is a property of the cohort, and the
table that reports it has to say so where a reader who quotes the number will
see it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.real_data.eval_t13_figures import data, tables
from benchmarks.real_data.eval_t13_figures.tests import committed_records


@pytest.fixture(scope="module")
def records() -> data.Records:
    """The committed fixture campaign."""
    return data.load([committed_records()])


def test_every_table_is_a_balanced_environment(records: data.Records) -> None:
    """Nothing here is worth much if the LaTeX does not close."""
    for name, text in tables.all_tables(records).items():
        assert name.endswith(".tex")
        assert text.count(r"\begin{tabular}") == text.count(r"\end{tabular}") == 1
        assert text.count(r"\toprule") == text.count(r"\bottomrule") == 1
        assert text.count(r"\begin{table") == text.count(r"\end{table")
        assert r"\label{" in text
        assert r"\caption{" in text


def test_names_with_underscores_are_escaped(records: data.Records) -> None:
    """``symmetry_ladder`` is a LaTeX error unless the underscore is escaped."""
    text = tables.ladder_spearman_table(records)
    assert "symmetry\\_ladder" in text
    assert "symmetry_ladder" not in text.replace("symmetry\\_ladder", "")


def test_spearman_table_reports_completion_beside_rho(records: data.Records) -> None:
    """A correlation over a fifth of a ladder is not a correlation over it."""
    text = tables.ladder_spearman_table(records)
    assert "completion" in text
    assert "completed" in text
    assert "censored" in text
    assert "sign test" in text


def test_spearman_recovers_the_two_governing_parameters(records: data.Records) -> None:
    """The pruned arm tracks ``|Aut|`` on the spider ladder; the unpruned does not.

    ``rho = +1`` for the pruned arm because its four rung times are strictly
    increasing in ``log10|Aut|``; the unpruned arm's times are within 10 % of
    each other and carry no such ordering.
    """
    rows, _ = tables.ladder_spearman_rows(records)
    by_key = {(row["ladder"].key, row["representation"].key): row["rho"] for row in rows}
    spider = ("spider_ladder", 33, "spider_k8")
    assert by_key[(spider, "isalgraph_pruned")] == pytest.approx(1.0)
    unpruned = by_key[(spider, "isalgraph_exhaustive")]
    assert unpruned is not None
    assert abs(unpruned) < 1.0


def test_spearman_rho_is_none_where_nothing_completed(records: data.Records) -> None:
    """No completions means no correlation, not a correlation of zero."""
    rows, _ = tables.ladder_spearman_rows(records)
    spider = ("spider_ladder", 33, "spider_k8")
    cell = next(
        row
        for row in rows
        if row["ladder"].key == spider and row["representation"].key == "isalgraph_canonical"
    )
    assert cell["rho"] is None
    assert cell["summary"].n_censored == 4


def test_exponent_table_says_it_is_a_cohort_property(records: data.Records) -> None:
    """The sentence R3.7d objects to must be refused in the table itself."""
    text = tables.scaling_exponent_table(records)
    assert "property of the measured cohort, not a complexity result" in text
    assert "completed" in text
    assert r"95\% CI" in text


def test_completion_table_names_the_mechanism(records: data.Records) -> None:
    """A wall-clock kill and a projection cap must be distinguishable."""
    text = tables.completion_table(records)
    assert "projection cap" in text
    assert "wall-clock kill" in text
    assert "not reached" in text
    assert "unsup." in text


def test_focus_only_narrows_the_table(records: data.Records) -> None:
    """The main text carries five arms; the appendix carries thirteen."""
    wide = tables.ladder_spearman_table(records)
    narrow = tables.ladder_spearman_table(records, focus_only=True)
    assert len(narrow) < len(wide)
    assert "AGM CAM" in wide
    assert "AGM CAM" not in narrow


def test_tables_cli(tmp_path: Path) -> None:
    """The documented command line runs end to end."""
    out = tmp_path / "out"
    code = tables.main(["--records", str(committed_records()), "--out-dir", str(out)])
    assert code == 0
    for name in tables.FILES.values():
        assert (out / name).exists()
        assert (out / name).read_text(encoding="utf-8").strip()
