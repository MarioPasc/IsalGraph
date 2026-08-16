"""Unit tests for the T-04a audit-path reporter.

Every fixture here is hand-written against contracts sections 2, 3 and 5.  The
reporter is the audit path for the grid and the F5 run, so its tests must not
depend on either producing output first: a shared fixture would let one bug hide
the other, which is exactly what the no-import rule exists to prevent.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from isalgraph.competitors.report import (
    CELL_COLUMNS,
    CLAIM_B_COMPARATORS,
    IncompleteComparatorSetError,
    build_k_payload,
    compute_k_members,
    failing_criteria,
    main,
    recompute_primary,
    render_f5_table_md,
    render_selection_md,
    write_report,
)

# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

METRICS = ("hamming", "levenshtein", "size_null")


def _cell(backend: str, metric: str, **overrides: object) -> dict[str, Any]:
    """A passing ``Cell`` record, with the field set of contracts section 3."""
    cell: dict[str, Any] = {
        "backend": backend,
        "metric": metric,
        "applicable": True,
        "reason": None,
        "candidate": metric != "size_null",
        "f1_defined_frac": 1.0,
        "f1_n_pairs": 19900,
        "f2_declared_pseudometric": True,
        "f2_violations": {},
        "f3_invariant": "50/50",
        "f3_skipped": 0,
        "f4_zero_mass": 0.01,
        "f4_coeff_variation": 0.42,
        "f6_ms_per_pair": 0.5,
        "f6_ms_per_pair_large": 1.9,
        "f6_over_advisory_limit": False,
        "passes_selection": False,
        "excluded_because": None,
    }
    if metric == "size_null":
        cell["excluded_because"] = (
            "baseline: consumes 'order'; not a candidate distance (competitors.md 3.2)"
        )
        # It would win on F6 if it were ever eligible.
        cell["f6_ms_per_pair"] = 0.001
    cell.update(overrides)
    return cell


def _f0(
    one: tuple[int, int],
    two: tuple[int, int],
    *,
    errors1: dict[str, int] | None = None,
    errors2: dict[str, int] | None = None,
) -> dict[str, Any]:
    """One ``f0`` block from an (encodable, attempted) pair per suite."""

    def block(encodable: int, attempted: int, errs: dict[str, int]) -> dict[str, Any]:
        return {
            "attempted": attempted,
            "encodable": encodable,
            "frac": (encodable / attempted) if attempted else 0.0,
            "errors": errs,
        }

    e1, e2 = errors1 or {}, errors2 or {}
    s1 = block(one[0], one[1], e1)
    s2 = block(two[0], two[1], e2)
    overall = block(
        s1["encodable"] + s2["encodable"],
        s1["attempted"] + s2["attempted"],
        {**e1, **e2},
    )
    return {"overall": overall, "suite1": s1, "suite2": s2}


def _grid_payload() -> dict[str, Any]:
    """A grid JSON exercising every branch ``k`` and ``partial`` can take."""
    backends = [
        "graph6",
        "sparse6",
        "nauty_graph6",
        "adjacency",
        "agm_cam",
        "min_dfs",
        "wl_subtree",
        "sparse6_nauty",
        "isalgraph_canonical",
        "size_null",
    ]
    cells: list[dict[str, Any]] = []
    for backend in backends:
        for metric in METRICS:
            cells.append(_cell(backend, metric))

    def find(backend: str, metric: str) -> dict[str, Any]:
        return next(c for c in cells if c["backend"] == backend and c["metric"] == metric)

    # sparse6: a comparator with no admissible distance -- every candidate fails.
    find("sparse6", "hamming").update(
        {
            "applicable": False,
            "reason": "sparse6 has no fixed-width frame",
            "f1_defined_frac": None,
            "f1_n_pairs": None,
            "f2_declared_pseudometric": None,
            "f2_violations": None,
            "f3_invariant": None,
            "f3_skipped": None,
            "f4_zero_mass": None,
            "f4_coeff_variation": None,
            "f6_ms_per_pair": None,
            "f6_ms_per_pair_large": None,
            "f6_over_advisory_limit": None,
        }
    )
    find("sparse6", "levenshtein").update({"f2_violations": {"triangle": 3}})

    # sparse6_nauty: NOT a comparator, and equally inadmissible.
    find("sparse6_nauty", "hamming").update({"f1_defined_frac": 0.87})
    find("sparse6_nauty", "levenshtein").update({"f4_coeff_variation": 0.0})

    # size_null the backend: a baseline, never selectable.
    for metric in METRICS:
        find("size_null", metric).update(
            {"candidate": False, "excluded_because": "baseline backend (Capability.BASELINE)"}
        )

    # adjacency: a comparator whose every candidate fails, so an explicit null.
    find("adjacency", "hamming").update({"f1_defined_frac": 0.55})
    find("adjacency", "levenshtein").update({"f3_invariant": "44/50"})

    primary: dict[str, Any] = {
        "graph6": "levenshtein",
        "sparse6": None,
        "nauty_graph6": "levenshtein",
        "adjacency": None,
        "agm_cam": "levenshtein",
        "min_dfs": "levenshtein",
        "wl_subtree": "levenshtein",
        "sparse6_nauty": None,
        "isalgraph_canonical": "levenshtein",
        "size_null": None,
    }
    for backend, metric in primary.items():
        if metric is not None:
            find(backend, metric)["passes_selection"] = True

    return {
        "protocol": "T-04a",
        "seed": 42,
        "n_graphs": 200,
        "backends": backends,
        "metrics": list(METRICS),
        "f0": {
            "graph6": _f0((51, 51), (149, 149)),
            "sparse6": _f0((51, 51), (100, 149), errors2={"BitCountUndefined": 49}),
            "nauty_graph6": _f0((51, 51), (149, 149)),
            "adjacency": _f0((51, 51), (149, 149)),
            # no admissible suite-2 encoding at all: charges 10 rows
            "agm_cam": _f0((51, 51), (0, 149), errors2={"SuiteScopeError": 149}),
            # partial on suite 1: charges 5 rows, dominant error wins the tie by count
            "min_dfs": _f0((48, 51), (149, 149), errors1={"BudgetExceeded": 2, "TimeoutError": 1}),
            # zero attempts on suite 1: a property of the draw, never charged
            "wl_subtree": _f0((0, 0), (149, 149)),
            "sparse6_nauty": _f0((51, 51), (149, 149)),
            "isalgraph_canonical": _f0((51, 51), (149, 149)),
            "size_null": _f0((51, 51), (149, 149)),
        },
        "cells": cells,
        "primary_distance": primary,
        "selection_reason": {
            "sparse6": "hamming: not applicable; levenshtein: F2 violated",
            "sparse6_nauty": "hamming: F1 < 1.0; levenshtein: F4 coefficient of variation 0",
        },
        "f5": "NOT COMPUTED HERE, BY CONSTRUCTION -- see isalgraph.competitors.f5",
    }


def _f5_payload() -> dict[str, Any]:
    """An F5 JSON with a Suite-1 exact record and a Suite-2 lb/ub pair."""

    def entry(rho: float | None, ci: list[float] | None, **extra: object) -> dict[str, Any]:
        record: dict[str, Any] = {"rho": rho, "p": 0.0, "ci": ci, "n_pairs": 19900}
        record.update(extra)
        return record

    def view(g6: float, null: float, ci_g6: list[float], ci_null: list[float]) -> dict[str, Any]:
        return {
            "size_null": entry(null, ci_null),
            "graph6": entry(
                g6,
                ci_g6,
                metric="levenshtein",
                n_undefined=0,
                zero_frac=0.01,
                reason=None,
                # uncensored: the two nulls agree, so no annotation is printed
                size_null_on_my_pairs=null,
            ),
            "sparse6": entry(
                None, None, metric=None, reason="no admissible distance (T-04a selection)"
            ),
        }

    return {
        "protocol": "T-04a-F5",
        "note": "DESCRIPTIVE. F5 is not an input to distance selection.",
        "seed": 42,
        "n_graphs": 200,
        "bootstrap_resamples": 2000,
        "primary_distance": {"graph6": "levenshtein", "sparse6": None},
        "results": {
            "iam_letter_low": {
                "dataset": "iam_letter_low",
                "suite": "suite1",
                "reference": "exact",
                "n_graphs": 200,
                "n_unencodable": {"graph6": 0, "sparse6": 3},
                "views": {
                    "all_pairs": view(0.925, 0.899, [0.900, 0.940], [0.880, 0.920]),
                    "equal_n": view(0.611, 0.000, [0.560, 0.660], [0.000, 0.000]),
                },
            },
            "grec::lb": {
                "dataset": "grec",
                "suite": "suite2",
                "reference": "lb",
                "n_graphs": 200,
                "n_unencodable": {"graph6": 0},
                "views": {
                    "all_pairs": view(0.712, 0.500, [0.680, 0.740], [0.470, 0.530]),
                    "equal_n": view(0.401, 0.000, [0.360, 0.440], [0.000, 0.000]),
                },
            },
            "grec::ub": {
                "dataset": "grec",
                "suite": "suite2",
                "reference": "ub",
                "n_graphs": 200,
                "n_unencodable": {"graph6": 0},
                "views": {
                    "all_pairs": view(0.884, 0.600, [0.850, 0.910], [0.570, 0.630]),
                    "equal_n": view(0.455, 0.000, [0.410, 0.500], [0.000, 0.000]),
                },
            },
        },
    }


def _f5_censored_payload() -> dict[str, Any]:
    """Mutagenicity's real shape: two representations censored on different graphs.

    ``isalgraph_pruned`` encodes 186/200 and its null on its own 17,205 pairs is
    0.6363 against the all-pairs 0.7538; ``min_dfs`` is censored 14/200 on a
    *different* 14 graphs and its restricted null is 0.6817.  A single
    per-dataset null would be wrong for at least one of them.
    """
    return {
        "protocol": "T-04a-F5",
        "seed": 42,
        "n_graphs": 200,
        "bootstrap_resamples": 2000,
        "results": {
            "mutagenicity::ub": {
                "dataset": "mutagenicity",
                "suite": "suite2",
                "reference": "ub",
                "n_graphs": 200,
                "n_unencodable": {"isalgraph_pruned": 14, "min_dfs": 14},
                "views": {
                    "all_pairs": {
                        "size_null": {
                            "rho": 0.7538,
                            "p": 0.0,
                            "ci": [0.7300, 0.7760],
                            "n_pairs": 19900,
                        },
                        "isalgraph_pruned": {
                            "rho": 0.8318,
                            "p": 0.0,
                            "ci": [0.8100, 0.8530],
                            "n_pairs": 17205,
                            "metric": "levenshtein",
                            "size_null_on_my_pairs": 0.6363,
                            "reason": None,
                        },
                        "min_dfs": {
                            "rho": 0.7401,
                            "p": 0.0,
                            "ci": [0.7180, 0.7620],
                            "n_pairs": 17205,
                            "metric": "levenshtein",
                            "size_null_on_my_pairs": 0.6817,
                            "reason": None,
                        },
                        "graph6": {
                            "rho": 0.7100,
                            "p": 0.0,
                            "ci": [0.6800, 0.7400],
                            "n_pairs": 19900,
                            "metric": "levenshtein",
                            "reason": None,
                        },
                    }
                },
            }
        },
    }


def _paired_payload() -> dict[str, Any]:
    """The shape of ``paired_null_ci.json``, which names no representation."""
    return {
        "statistic": "rho(Lev,ref) - rho(|dn|,ref), paired graph-level bootstrap",
        "resamples": 2000,
        "seed": 42,
        "records": [
            {
                "dataset": "iam_letter_low",
                "arm": "exact",
                "n_pairs": 19900,
                "n_encoded": 200,
                "rho_lev": 0.925296021436113,
                "rho_null": 0.8990791903163676,
                "paired_diff": 0.026216831119745376,
                "ci_lo": 0.00832852179700783,
                "ci_hi": 0.04634559090351528,
                "excludes_zero": True,
                "n_resamples_used": 2000,
            },
            {
                "dataset": "iam_letter_high",
                "arm": "exact",
                "n_pairs": 19900,
                "n_encoded": 200,
                "rho_lev": 0.5,
                "rho_null": 0.7205,
                "paired_diff": -0.2205,
                "ci_lo": -0.2599,
                "ci_hi": -0.1825,
                "excludes_zero": True,
                "n_resamples_used": 2000,
            },
        ],
    }


@pytest.fixture
def grid() -> dict[str, Any]:
    return _grid_payload()


@pytest.fixture
def grid_file(tmp_path: Path, grid: dict[str, Any]) -> Path:
    path = tmp_path / "grid.json"
    path.write_text(json.dumps(grid), encoding="utf-8")
    return path


@pytest.fixture
def f5_file(tmp_path: Path) -> Path:
    path = tmp_path / "f5.json"
    path.write_text(json.dumps(_f5_payload()), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# the frozen comparator set
# --------------------------------------------------------------------------- #


def test_comparator_set_is_the_frozen_seven() -> None:
    assert CLAIM_B_COMPARATORS == (
        "graph6",
        "sparse6",
        "nauty_graph6",
        "adjacency",
        "agm_cam",
        "min_dfs",
        "wl_subtree",
    )
    for outsider in ("sparse6_nauty", "isalgraph_canonical", "isalgraph_pruned", "size_null"):
        assert outsider not in CLAIM_B_COMPARATORS


# --------------------------------------------------------------------------- #
# k
# --------------------------------------------------------------------------- #


def test_k_counts_only_comparators_with_no_admissible_distance(grid: dict[str, Any]) -> None:
    payload = build_k_payload(grid)
    assert payload["k_members"] == ["sparse6", "adjacency"]
    assert payload["k"] == 2


def test_non_comparator_without_a_distance_does_not_raise_k(grid: dict[str, Any]) -> None:
    assert grid["primary_distance"]["sparse6_nauty"] is None
    assert grid["primary_distance"]["size_null"] is None
    payload = build_k_payload(grid)
    assert "sparse6_nauty" not in payload["k_members"]
    assert "size_null" not in payload["k_members"]
    assert "isalgraph_canonical" not in payload["k_members"]


def test_a_comparator_absent_from_primary_distance_aborts(grid: dict[str, Any]) -> None:
    """The third preregistration case: not computable at all, charged -25 not -15.

    Rolling an absent comparator into ``k`` would undercharge it by 10 rows and
    mislabel a computability failure as a distance failure, so the reporter
    refuses to produce a number rather than guessing which case it is.
    """
    del grid["primary_distance"]["nauty_graph6"]
    with pytest.raises(IncompleteComparatorSetError) as excinfo:
        build_k_payload(grid)
    message = str(excinfo.value)
    assert "nauty_graph6" in message
    assert "-25, not -15" in message


def test_the_abort_never_increments_k(grid: dict[str, Any]) -> None:
    before = build_k_payload(grid)
    assert before["k"] == 2
    del grid["primary_distance"]["nauty_graph6"]
    with pytest.raises(IncompleteComparatorSetError):
        compute_k_members(grid["primary_distance"])
    with pytest.raises(IncompleteComparatorSetError):
        build_k_payload(grid)
    # no k.json is producible from an incomplete comparator set
    grid["primary_distance"]["nauty_graph6"] = "levenshtein"
    assert build_k_payload(grid)["k"] == 2


def test_an_absent_non_comparator_does_not_abort(grid: dict[str, Any]) -> None:
    del grid["primary_distance"]["isalgraph_canonical"]
    payload = build_k_payload(grid)
    assert payload["k"] == 2
    assert "isalgraph_canonical" not in payload["k_members"]


def test_the_cli_aborts_on_an_incomplete_comparator_set(
    tmp_path: Path, grid: dict[str, Any]
) -> None:
    del grid["primary_distance"]["nauty_graph6"]
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(grid), encoding="utf-8")
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as excinfo:
        main(["--grid", str(path), "--out-dir", str(out)])
    assert excinfo.value.code != 0
    assert not (out / "k.json").exists()


def test_comparator_set_is_reported_verbatim(grid: dict[str, Any]) -> None:
    assert build_k_payload(grid)["comparator_set"] == list(CLAIM_B_COMPARATORS)


# --------------------------------------------------------------------------- #
# partial computability -- the separate charge
# --------------------------------------------------------------------------- #


def test_partial_separates_computability_from_admissibility(grid: dict[str, Any]) -> None:
    payload = build_k_payload(grid)
    charged = {entry["representation"] for entry in payload["partial"]}
    # sparse6 has f0 < 1.0 on suite2 AND no admissible distance: it is already
    # charged 15 rows through k and must not be charged a second time.
    assert "sparse6" in payload["k_members"]
    assert "sparse6" not in charged
    assert charged == {"agm_cam", "min_dfs"}


def test_partial_charges_ten_for_suite2_and_five_for_suite1(grid: dict[str, Any]) -> None:
    by_rep = {e["representation"]: e for e in build_k_payload(grid)["partial"]}
    assert by_rep["agm_cam"]["suite"] == "suite2"
    assert by_rep["agm_cam"]["rows_lost"] == 10
    assert by_rep["min_dfs"]["suite"] == "suite1"
    assert by_rep["min_dfs"]["rows_lost"] == 5


def test_partial_reason_names_the_dominant_exception(grid: dict[str, Any]) -> None:
    by_rep = {e["representation"]: e for e in build_k_payload(grid)["partial"]}
    assert by_rep["agm_cam"]["reason"] == "F0 = 0.00 on suite2 (SuiteScopeError)"
    # BudgetExceeded (2) outranks TimeoutError (1).
    assert by_rep["min_dfs"]["reason"] == "F0 = 0.94 on suite1 (BudgetExceeded)"


def test_a_suite_with_zero_attempts_is_not_charged(grid: dict[str, Any]) -> None:
    assert grid["f0"]["wl_subtree"]["suite1"]["attempted"] == 0
    charged = {e["representation"] for e in build_k_payload(grid)["partial"]}
    assert "wl_subtree" not in charged


def test_partial_entry_carries_the_four_contracted_fields(grid: dict[str, Any]) -> None:
    for entry in build_k_payload(grid)["partial"]:
        assert set(entry) == {"representation", "suite", "rows_lost", "reason"}


# --------------------------------------------------------------------------- #
# the family size
# --------------------------------------------------------------------------- #


def test_n_actual_arithmetic(grid: dict[str, Any]) -> None:
    payload = build_k_payload(grid)
    rows_lost = sum(int(e["rows_lost"]) for e in payload["partial"])
    assert rows_lost == 15
    assert payload["n_actual_f2_before_d"] == 182 - 15 * payload["k"] - rows_lost
    assert payload["n_actual_f2_before_d"] == 137


def test_k_payload_has_exactly_the_contracted_keys(grid: dict[str, Any]) -> None:
    assert set(build_k_payload(grid)) == {
        "k",
        "k_members",
        "comparator_set",
        "partial",
        "n_actual_f2_before_d",
        "formula",
    }


def test_formula_states_the_arithmetic(grid: dict[str, Any]) -> None:
    formula = build_k_payload(grid)["formula"]
    assert "182 - 15k - 8d - p" in formula
    assert "182 - 15*2 - 8d - 15 = 137 - 8d" in formula


# --------------------------------------------------------------------------- #
# the selection rule, recomputed
# --------------------------------------------------------------------------- #


def test_failing_criteria_is_empty_for_a_passing_cell() -> None:
    assert failing_criteria(_cell("graph6", "levenshtein")) == []


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"f1_defined_frac": 0.9}, "F1 = 0.9000 < 1.0"),
        ({"f2_violations": {"symmetry": 2}}, "F2 violated (symmetry=2)"),
        ({"f3_invariant": "47/50"}, "F3 invariant on only 47/50 graphs"),
        ({"f4_zero_mass": 0.75}, "F4 zero-mass 0.7500 > 0.5"),
        ({"f4_coeff_variation": 0.0}, "F4 coefficient of variation"),
        ({"applicable": False, "reason": "no fixed-width frame"}, "not applicable"),
    ],
)
def test_failing_criteria_names_each_gate(overrides: dict[str, Any], fragment: str) -> None:
    fails = failing_criteria(_cell("graph6", "hamming", **overrides))
    assert any(fragment in item for item in fails), fails


def test_a_metric_consuming_order_is_never_selected() -> None:
    # size_null is the cheapest cell on F6 and would win if it were eligible.
    cells = [_cell("graph6", metric) for metric in METRICS]
    assert min(cells, key=lambda c: c["f6_ms_per_pair"])["metric"] == "size_null"
    assert recompute_primary(cells) == "hamming"


def test_recompute_returns_none_when_every_candidate_fails(grid: dict[str, Any]) -> None:
    sparse6 = [c for c in grid["cells"] if c["backend"] == "sparse6"]
    assert recompute_primary(sparse6) is None


# --------------------------------------------------------------------------- #
# supplementary_grid.csv
# --------------------------------------------------------------------------- #


def test_csv_has_one_row_per_cell_and_a_stable_header(
    tmp_path: Path, grid_file: Path, grid: dict[str, Any]
) -> None:
    write_report(grid_file, None, tmp_path / "out")
    with (tmp_path / "out" / "supplementary_grid.csv").open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert tuple(rows[0]) == CELL_COLUMNS
    assert len(rows) - 1 == len(grid["cells"])
    assert [(r[0], r[1]) for r in rows[1:]] == [(c["backend"], c["metric"]) for c in grid["cells"]]


def test_csv_reports_inapplicable_cells_with_reason_and_empty_measurements(
    tmp_path: Path, grid_file: Path
) -> None:
    write_report(grid_file, None, tmp_path / "out")
    with (tmp_path / "out" / "supplementary_grid.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row = next(r for r in rows if r["backend"] == "sparse6" and r["metric"] == "hamming")
    assert row["applicable"] == "false"
    assert row["reason"] == "sparse6 has no fixed-width frame"
    assert row["f1_defined_frac"] == ""
    assert row["f4_zero_mass"] == ""


def test_csv_serialises_the_f2_violation_table(tmp_path: Path, grid_file: Path) -> None:
    write_report(grid_file, None, tmp_path / "out")
    with (tmp_path / "out" / "supplementary_grid.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row = next(r for r in rows if r["backend"] == "sparse6" and r["metric"] == "levenshtein")
    assert json.loads(row["f2_violations"]) == {"triangle": 3}


# --------------------------------------------------------------------------- #
# selection.md
# --------------------------------------------------------------------------- #


def test_selection_md_names_the_failing_criterion_of_every_candidate(
    grid: dict[str, Any], tmp_path: Path
) -> None:
    text = render_selection_md(grid, tmp_path / "grid.json")
    assert "### `sparse6` -- none admissible" in text
    assert "not applicable" in text
    assert "F2 violated (triangle=3)" in text


def test_selection_md_marks_a_representation_absent_from_the_grid(
    grid: dict[str, Any], tmp_path: Path
) -> None:
    """An absent *non*-comparator still renders distinctly from a measured null."""
    del grid["primary_distance"]["isalgraph_canonical"]
    text = render_selection_md(grid, tmp_path / "grid.json")
    assert "*absent from grid*" in text
    assert "**none admissible**" in text


def test_selection_md_reports_every_representation(grid: dict[str, Any], tmp_path: Path) -> None:
    text = render_selection_md(grid, tmp_path / "grid.json")
    for backend in grid["backends"]:
        assert f"`{backend}`" in text


def test_selection_md_states_k_and_the_partial_charge(grid: dict[str, Any], tmp_path: Path) -> None:
    text = render_selection_md(grid, tmp_path / "grid.json")
    assert "`k` = 2" in text
    assert "F0 = 0.00 on suite2 (SuiteScopeError)" in text


# --------------------------------------------------------------------------- #
# f5_table.md
# --------------------------------------------------------------------------- #


def test_f5_table_keeps_lb_and_ub_separate_and_never_interpolates(tmp_path: Path) -> None:
    text = render_f5_table_md(_f5_payload(), tmp_path / "f5.json")
    assert "rho (lb)" in text
    assert "rho (ub)" in text
    assert "0.712" in text
    assert "0.884" in text
    # the midpoint of 0.712 and 0.884 is 0.798 and must appear nowhere
    assert "0.798" not in text


def test_f5_table_shows_the_size_null_and_the_explicit_difference(tmp_path: Path) -> None:
    text = render_f5_table_md(_f5_payload(), tmp_path / "f5.json")
    assert "size null" in text
    assert "+0.026" in text  # 0.925 - 0.899, Suite 1 all_pairs
    assert "+0.212" in text  # 0.712 - 0.500, Suite 2 lower bound


def test_f5_table_prints_a_missing_rho_as_a_stated_absence(tmp_path: Path) -> None:
    text = render_f5_table_md(_f5_payload(), tmp_path / "f5.json")
    assert "no admissible distance (T-04a selection)" in text


def test_f5_table_emits_both_views(tmp_path: Path) -> None:
    text = render_f5_table_md(_f5_payload(), tmp_path / "f5.json")
    assert "**View: `all_pairs`**" in text
    assert "**View: `equal_n`**" in text


def test_the_difference_uses_the_restricted_null_not_the_row_level_one(tmp_path: Path) -> None:
    """The whole point: a censored representation is differenced on its own pairs."""
    text = render_f5_table_md(_f5_censored_payload(), tmp_path / "f5.json")
    # 0.8318 - 0.6363 = +0.196 on its own pairs, NOT 0.8318 - 0.7538 = +0.078
    assert "+0.196" in text
    assert "+0.078" not in text


def test_a_second_censored_row_uses_its_own_null(tmp_path: Path) -> None:
    """The restricted null is per cell, never per dataset."""
    text = render_f5_table_md(_f5_censored_payload(), tmp_path / "f5.json")
    # min_dfs: 0.7401 - 0.6817 = +0.058, against isalgraph_pruned's null it would be +0.104
    assert "+0.058" in text
    assert "+0.104" not in text


def test_a_divergent_null_is_printed_not_left_to_inference(tmp_path: Path) -> None:
    text = render_f5_table_md(_f5_censored_payload(), tmp_path / "f5.json")
    assert "[null 0.7538->0.6363 on its pairs]" in text
    assert "[null 0.7538->0.6817 on its pairs]" in text


def test_an_agreeing_null_is_not_annotated(tmp_path: Path) -> None:
    """On an uncensored record the two nulls agree, so no noise is added.

    Asserted over the table rows only: the legend documents the annotation
    format and must keep saying so.
    """
    text = render_f5_table_md(_f5_payload(), tmp_path / "f5.json")
    rows = [line for line in text.splitlines() if line.startswith("|")]
    assert rows
    assert not any("on its pairs]" in line for line in rows)


def test_a_missing_restricted_null_falls_back_and_says_so(tmp_path: Path) -> None:
    payload = _f5_censored_payload()
    row = payload["results"]["mutagenicity::ub"]["views"]["all_pairs"]["graph6"]
    assert "size_null_on_my_pairs" not in row
    text = render_f5_table_md(payload, tmp_path / "f5.json")
    assert "all-pairs null; `size_null_on_my_pairs` absent" in text


def test_the_row_level_size_null_is_still_printed(tmp_path: Path) -> None:
    """Right as a standalone baseline, wrong only as a subtrahend."""
    text = render_f5_table_md(_f5_censored_payload(), tmp_path / "f5.json")
    assert "`size_null`" in text
    assert "0.754" in text  # the all-pairs null, rendered in its own row


def test_the_legend_does_not_claim_the_difference_ranks_with_rho(tmp_path: Path) -> None:
    """The subtrahend now varies by row, so the old legend claim is false."""
    text = render_f5_table_md(_f5_payload(), tmp_path / "f5.json")
    assert "ranks identically to rho" not in text.replace("**not** rank\nidentically", "")
    assert "does **not** rank" in text


def test_the_paired_null_section_is_the_stated_significance_test(tmp_path: Path) -> None:
    text = render_f5_table_md(_f5_payload(), tmp_path / "f5.json", _paired_payload())
    assert "## Paired size-null differences" in text
    assert "not the overlap of the marginal CIs" in text
    assert "+0.0262" in text
    assert "[+0.0083, +0.0463]" in text
    assert "2 of 2 exclude zero. There is no undecided record." in text


def test_the_paired_section_is_absent_without_the_flag(tmp_path: Path) -> None:
    text = render_f5_table_md(_f5_payload(), tmp_path / "f5.json")
    assert "## Paired size-null differences" not in text


def test_k_json_makes_no_reference_to_f5(tmp_path: Path, grid_file: Path, f5_file: Path) -> None:
    """k comes from the grid alone; F5 selects nothing and must not appear."""
    out = tmp_path / "out"
    write_report(grid_file, f5_file, out)
    text = (out / "k.json").read_text(encoding="utf-8").lower()
    for token in ("f5", "rho", "spearman", "correlation", "bootstrap", "null"):
        assert token not in text, token


def _table_widths(markdown: str) -> list[int]:
    """The cell count of every markdown table row, header rows included."""
    return [
        len(line.strip().strip("|").split("|"))
        for line in markdown.splitlines()
        if line.startswith("|")
    ]


@pytest.mark.parametrize("view_name", ["all_pairs", "equal_n"])
def test_f5_table_rows_are_not_split_by_an_unescaped_pipe(tmp_path: Path, view_name: str) -> None:
    """A cell holding ``|n1 - n2|`` would silently add columns to its row."""
    text = render_f5_table_md(_f5_payload(), tmp_path / "f5.json")
    block = text.split(f"**View: `{view_name}`**")[1].split("**View")[0]
    widths = _table_widths(block)
    assert widths, block
    assert len(set(widths)) == 1, widths


def test_selection_md_rows_are_not_split_by_an_unescaped_pipe(
    grid: dict[str, Any], tmp_path: Path
) -> None:
    text = render_selection_md(grid, tmp_path / "grid.json")
    for block in text.split("\n\n"):
        widths = _table_widths(block)
        if widths:
            assert len(set(widths)) == 1, block


def test_f5_table_marks_the_best_value_per_column(tmp_path: Path) -> None:
    text = render_f5_table_md(_f5_payload(), tmp_path / "f5.json")
    assert "**0.925**" in text
    assert "**0.884**" in text


# --------------------------------------------------------------------------- #
# the CLI, end to end
# --------------------------------------------------------------------------- #


def test_end_to_end_writes_four_artifacts(tmp_path: Path, grid_file: Path, f5_file: Path) -> None:
    out = tmp_path / "out"
    assert main(["--grid", str(grid_file), "--f5", str(f5_file), "--out-dir", str(out)]) == 0
    for name in ("supplementary_grid.csv", "selection.md", "k.json", "f5_table.md"):
        assert (out / name).is_file()
    payload = json.loads((out / "k.json").read_text(encoding="utf-8"))
    assert payload["k"] == 2
    assert payload["n_actual_f2_before_d"] == 137


def test_a_missing_f5_degrades_to_a_stated_absence(tmp_path: Path, grid_file: Path) -> None:
    out = tmp_path / "out"
    assert main(["--grid", str(grid_file), "--out-dir", str(out)]) == 0
    for name in ("supplementary_grid.csv", "selection.md", "k.json", "f5_table.md"):
        assert (out / name).is_file()
    text = (out / "f5_table.md").read_text(encoding="utf-8")
    assert "The F5 table is absent" in text
    assert "--f5" in text


def test_a_malformed_grid_exits_with_a_message(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--grid", str(bad), "--out-dir", str(tmp_path / "out")])
