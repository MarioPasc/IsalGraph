"""Loading, ladder ordering and the censoring rules.

Three of the package's acceptance criteria live here: a mixed-build load must
raise, rungs must order by the ``params`` index rather than by ``log10_aut``,
and no summary may pool a censored row with a completed one.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import pytest

from benchmarks.real_data.eval_t13_complexity import schema
from benchmarks.real_data.eval_t13_figures import data
from benchmarks.real_data.eval_t13_figures.tests import (
    BUDGET_S,
    HYPERCUBE_RUNGS,
    OTHER_BUILD_HASH,
    build_rows,
    committed_counters,
    committed_records,
    write_counters,
    write_records,
)


@pytest.fixture(scope="module")
def records() -> data.Records:
    """The committed fixture campaign."""
    return data.load([committed_records()])


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_keeps_every_row_and_the_header(records: data.Records) -> None:
    """Nothing is dropped: the censored cells are where the result lives."""
    assert len(records.rows) == len(build_rows())
    assert records.headers
    assert records.build_hash
    statuses = {r["status"] for r in records.rows}
    assert statuses == {"ok", "censored", "unsupported", "error"}


def test_load_rejects_a_mixed_build(tmp_path: Path) -> None:
    """Two shards from different engine builds cannot be pooled."""
    write_records(tmp_path, name="records_a.jsonl")
    write_records(tmp_path, name="records_b.jsonl", build_hash=OTHER_BUILD_HASH)
    with pytest.raises(data.MixedBuildError, match="different engine builds"):
        data.load([str(tmp_path / "records_*.jsonl")])


def test_load_accepts_two_shards_of_one_build(tmp_path: Path) -> None:
    """The same check must not fire on a legitimate multi-shard campaign."""
    write_records(tmp_path, name="records_constructed_0of2.jsonl")
    write_records(tmp_path, name="records_constructed_1of2.jsonl")
    loaded = data.load([str(tmp_path / "records_constructed_*.jsonl")])
    assert len(loaded.rows) == 2 * len(build_rows())
    assert loaded.build_hash


def test_load_rejects_a_row_that_fails_the_frozen_schema(tmp_path: Path) -> None:
    """A schema violation propagates; it is never repaired in the reader."""
    path = write_records(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[1])
    row["seconds"] = 1.0
    row["status"] = "censored"
    row["error_kind"] = schema.KIND_WALLCLOCK
    lines[1] = json.dumps(row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(schema.SchemaError):
        data.load([path])


def test_load_raises_when_nothing_matches(tmp_path: Path) -> None:
    """An empty match must never produce an empty figure with no error."""
    with pytest.raises(FileNotFoundError):
        data.load([str(tmp_path / "nothing_*.jsonl")])


def test_load_requires_a_header(tmp_path: Path) -> None:
    """A shard with no header cannot be provenanced and is refused."""
    path = write_records(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")
    with pytest.raises(data.DataError, match="no header"):
        data.load([path])


# ---------------------------------------------------------------------------
# Ladder addressing
# ---------------------------------------------------------------------------


def test_parse_params_round_trips_the_rendered_form() -> None:
    """``params`` is the only field that makes the ladder contrast possible."""
    assert data.parse_params("base=hypercube,swaps=3") == {"base": "hypercube", "swaps": "3"}
    assert data.parse_params(None) == {}
    with pytest.raises(data.DataError):
        data.parse_params("swaps")


def test_rung_index_reads_the_design_parameter_not_the_measurement() -> None:
    """``swaps`` for a symmetry ladder, ``rung`` for a spider ladder."""
    row = {"family": "symmetry_ladder", "params": "base=hypercube,swaps=4"}
    assert data.rung_index(row) == 4
    row = {"family": "spider_ladder", "params": "leg=4,legs=8,rung=2"}
    assert data.rung_index(row) == 2
    with pytest.raises(data.NotALadderError):
        data.rung_index({"family": "prism", "params": "k=3"})


def test_ladders_are_grouped_the_way_families_ladder_span_groups_them(
    records: data.Records,
) -> None:
    """Three ladders, keyed by ``(family, n, base)``."""
    keys = [lad.key for lad in data.ladders(records)]
    assert keys == [
        ("spider_ladder", 33, "spider_k8"),
        ("symmetry_ladder", 8, "complete_bipartite"),
        ("symmetry_ladder", 16, "hypercube"),
    ]


def test_rungs_order_by_the_params_index_and_not_by_log10_aut(
    records: data.Records,
) -> None:
    """The acceptance criterion, on a deliberately non-monotone ladder.

    The 4-cube ladder's ``log10_aut`` goes 2.98, 0.90, 1.20, 0.30 across swap
    counts 0, 1, 2, 4.  Ordering by ``log10_aut`` would put swap 2 before swap
    1; ordering by the design's own index does not.  Ordering by the
    measurement would make the abscissa of the primary figure decide the order
    of the correlation computed on it.
    """
    ladder = next(
        lad for lad in data.ladders(records) if lad.key == ("symmetry_ladder", 16, "hypercube")
    )
    assert ladder.rungs == (0, 1, 2, 4)
    assert all(g.log10_aut is not None for g in ladder.graphs)
    by_design = [float(g.log10_aut) for g in ladder.graphs if g.log10_aut is not None]
    assert by_design == [aut for _, aut, _, _, _ in HYPERCUBE_RUNGS]
    assert by_design != sorted(by_design)
    assert by_design != sorted(by_design, reverse=True)


def test_ladder_invariants_are_enforced(tmp_path: Path) -> None:
    """A ladder whose ``m`` moves is not a ladder and cannot carry the contrast."""
    rows = build_rows()
    for row in rows:
        if row["family"] == "spider_ladder" and "rung=3" in str(row["params"]):
            row["m"] = 31
    write_records(tmp_path, rows=rows)
    loaded = data.load([tmp_path])
    with pytest.raises(data.LadderIntegrityError, match="varies in m"):
        data.ladders(loaded)


def test_resolutions_collapse_to_one_record_per_graph(records: data.Records) -> None:
    """Symmetry fields repeat per representation; the figure needs one point."""
    rows = data.resolutions(records)
    graphs = {data.graph_identity(r) for r in records.rows}
    assert len(rows) == len(graphs)
    assert all(r.n_wl_classes <= r.n_orbits for r in rows)
    assert all(r.n_triplet_classes <= r.n_orbits for r in rows)


# ---------------------------------------------------------------------------
# Censoring
# ---------------------------------------------------------------------------


def _min_dfs_spider_rows(records: data.Records) -> list[dict[str, Any]]:
    """Every ``min_dfs`` row of the spider ladder."""
    ladder = next(
        lad for lad in data.ladders(records) if lad.key == ("spider_ladder", 33, "spider_k8")
    )
    return [dict(row) for _, row in ladder.series("min_dfs")]


def test_the_naive_pool_and_the_censoring_aware_path_disagree(
    records: data.Records,
) -> None:
    """The acceptance criterion, on the row that motivates the whole rule.

    ``min_dfs`` at the most symmetric spider rung is cap-censored at 4.1 ms:
    the encoder did **not** finish, and the true time is greater.  Pooling that
    number with the three completions makes it the fastest observation in the
    series and drags the median down; the completions-only median excludes it,
    and Kaplan--Meier treats it as the lower bound it is.
    """
    rows = _min_dfs_spider_rows(records)
    naive = statistics.median([float(r["seconds"]) for r in rows])
    completions_only = data.completions_only_median_seconds(rows)
    km, reached = data.km_median_seconds(rows)

    assert any(data.is_censored(r) for r in rows)
    assert completions_only is not None
    assert naive < completions_only
    assert reached is True
    assert km is not None
    assert km != naive
    assert data.completion_rate(rows) == 0.75


def test_km_median_is_not_reached_when_most_units_are_still_running(
    records: data.Records,
) -> None:
    """The exhaustive arm never finishes at ``n = 33``; that is the answer."""
    ladder = next(
        lad for lad in data.ladders(records) if lad.key == ("spider_ladder", 33, "spider_k8")
    )
    rows = [row for _, row in ladder.series("isalgraph_canonical")]
    summary = data.summarise_times(rows)
    assert summary.n_completed == 0
    assert summary.n_censored == 4
    assert summary.completion_rate == 0.0
    assert summary.km_median is None
    assert summary.km_median_reached is False
    assert summary.max_observed == BUDGET_S
    assert summary.completions_only_median is None


def test_unsupported_and_error_rows_never_enter_a_time_estimate(
    records: data.Records,
) -> None:
    """Neither status bounds a runtime, so neither may be counted as fast."""
    ladder = next(
        lad
        for lad in data.ladders(records)
        if lad.key == ("symmetry_ladder", 8, "complete_bipartite")
    )
    summary = data.summarise_times([row for _, row in ladder.series("agm_cam")])
    assert summary.n_unsupported == 3
    assert summary.n_observations == 0
    assert summary.completion_rate is None
    assert summary.completions_only_median is None
    assert summary.km_median is None


def test_summary_names_the_censoring_mechanism(records: data.Records) -> None:
    """A wall-clock kill and a projection cap are different observations."""
    rows = _min_dfs_spider_rows(records)
    summary = data.summarise_times(rows)
    assert summary.censoring_kinds == ((schema.KIND_MAX_PROJECTIONS, 1),)


def test_km_median_matches_a_hand_computed_case() -> None:
    """Guard the estimator itself, not only its use."""
    rows = [
        {"status": "censored", "seconds": 1.0, "error_kind": schema.KIND_WALLCLOCK},
        {"status": "ok", "seconds": 2.0, "error_kind": None},
        {"status": "ok", "seconds": 3.0, "error_kind": None},
    ]
    # S(2) = 1 - 1/2 = 0.5 over the two units still at risk, so the median is 2.
    assert data.km_median_seconds(rows) == (2.0, True)
    # The completions-only rule sees only {2, 3} and answers 2.5.
    assert data.completions_only_median_seconds(rows) == 2.5


def test_fit_is_named_for_the_subset_it_uses(records: data.Records) -> None:
    """A censored row carries no completion time and cannot enter a fit."""
    rows = [r for r in records.with_arm("default") if r["representation"] == "isalgraph_pruned"]
    fit = data.fit_power_law_completions_only(rows, n_boot=200, seed=7)
    assert fit is not None
    assert fit.n_points == sum(1 for r in rows if data.is_completed(r))
    assert fit.ci_low <= fit.alpha <= fit.ci_high


def test_fit_returns_none_when_nothing_completed(records: data.Records) -> None:
    """No completions means no exponent, not an exponent of zero."""
    rows = [
        r
        for r in records.with_arm("default")
        if r["representation"] == "isalgraph_canonical" and r["family"] == "spider_ladder"
    ]
    assert data.fit_power_law_completions_only(rows, n_boot=50) is None


# ---------------------------------------------------------------------------
# Plain-Python statistics
# ---------------------------------------------------------------------------


def test_spearman_is_tie_corrected() -> None:
    """The ``6 sum d^2`` shortcut is wrong under ties, which ladders produce."""
    assert data.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert data.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    rho = data.spearman([1, 2, 2, 3], [1, 2, 3, 4])
    assert rho is not None
    assert 0.9 < rho < 1.0


def test_spearman_returns_none_rather_than_zero_when_undefined() -> None:
    """A constant variable makes rho undefined; 0.0 would read as "unrelated"."""
    assert data.spearman([1, 1, 1], [1, 2, 3]) is None
    assert data.spearman([1, 2], [3, 4]) is None
    with pytest.raises(ValueError, match="equal lengths"):
        data.spearman([1, 2, 3], [1, 2])


def test_sign_test_matches_the_exact_binomial() -> None:
    """Six of six one way is p = 2/64."""
    test = data.sign_test([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert (test.n_positive, test.n_negative, test.n_ties) == (6, 0, 0)
    assert test.p_value == pytest.approx(2.0 / 64.0)
    tied = data.sign_test([0.0, 0.0])
    assert tied.p_value == 1.0
    assert tied.n_ties == 2


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------


def test_counter_rows_validate_against_the_dataclass(tmp_path: Path) -> None:
    """The count fields come from ``OperationCounts``, never a transcription."""
    counters = data.load_counters([committed_counters()])
    assert counters.rows
    assert set(counters.encoders) <= {"greedy_single", "greedy_min", "canonical", "pruned"}
    assert "backtrack_nodes" in data.COUNTER_COUNTS
    assert set(data.COUNTER_FIELDS) == set(counters.rows[0])


def test_counter_loader_rejects_a_parity_failure(tmp_path: Path) -> None:
    """An unverified counter is not a measurement of the shipped algorithm."""
    path = write_counters(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["parity_ok"] = False
    lines[0] = json.dumps(row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(data.ParityError, match="did not reproduce"):
        data.load_counters([path])
    assert data.load_counters([path], strict_parity=False).rows


def test_counter_loader_rejects_an_extra_field(tmp_path: Path) -> None:
    """An undeclared key comes from code that disagrees about the schema."""
    path = write_counters(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["wall_seconds"] = 1.0
    lines[0] = json.dumps(row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(data.DataError, match="extra="):
        data.load_counters([path])
