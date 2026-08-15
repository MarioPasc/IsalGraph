"""Unit tests for the T-05 §7 bounded-GED analysis module.

The tests that matter are the ones a reviewer would ask for: that the width
formula is right where ``UB == 0``, that the bootstrap resamples graphs and not
pairs, that the frozen D15 tier assignment is a lookup rather than a
recomputation, that the density quintiles come from the pair population, and
that the strict upper triangle is what is indexed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from benchmarks.real_data.eval_setup import approx_ged_analysis as aga

# ---------------------------------------------------------------------------
# Fixtures: synthetic .npz trees in the production schema
# ---------------------------------------------------------------------------


def _write_role(
    directory: Path,
    dataset: str,
    role: str,
    matrix: np.ndarray,
    node_counts: np.ndarray,
    edge_counts: np.ndarray,
    seconds: np.ndarray | None = None,
) -> Path:
    """Write one role ``.npz`` in the production schema."""
    directory.mkdir(parents=True, exist_ok=True)
    n = node_counts.size
    if seconds is None:
        seconds = np.full((n, n), 0.001, dtype=np.float32)
        np.fill_diagonal(seconds, 0.0)
    upper = np.triu_indices(n, k=1)
    metadata = {
        "dataset": dataset,
        "role": role.lower(),
        "method": aga.ROLE_METHOD[role],
        "options_string": "--threads 1",
        "accessor": "lower" if role == "LB" else "upper",
        "cost_model": "unit",
        "code_commit": "0" * 40,
        "computed_utc": "2026-08-14T00:00:00+00:00",
        "n_shards": 1,
        "gedlib_source": "/synthetic",
        "seconds_total": float(seconds[upper].sum()),
        "mean_seconds_per_pair": float(seconds[upper].mean()),
        "schema_version": 1,
    }
    path = directory / f"{dataset}.npz"
    np.savez_compressed(
        path,
        ged_matrix=matrix.astype(np.float64),
        seconds_matrix=seconds.astype(np.float32),
        node_counts=node_counts.astype(np.int32),
        edge_counts=edge_counts.astype(np.int32),
        graph_ids=np.array([f"{dataset}_train_{i}" for i in range(n)], dtype="<U32"),
        labels=np.array([""] * n, dtype="<U1"),
        metadata=np.array(json.dumps(metadata)),
    )
    return path


def _symmetric(rng: np.random.Generator, n: int, high: int) -> np.ndarray:
    """Return a symmetric, zero-diagonal integer matrix."""
    matrix = rng.integers(0, high, size=(n, n)).astype(np.float64)
    matrix = np.triu(matrix, k=1)
    matrix = matrix + matrix.T
    return matrix


@pytest.fixture
def synthetic_tree(tmp_path: Path) -> tuple[Path, list[str]]:
    """Build a two-dataset synthetic input tree with valid brackets."""
    rng = np.random.default_rng(7)
    root = tmp_path / "APPROX_GED"
    datasets = ["linux", "protein"]
    sizes = {"linux": 12, "protein": 17}
    for dataset in datasets:
        n = sizes[dataset]
        node_counts = rng.integers(2, 14, size=n)
        edge_counts = np.array(
            [rng.integers(k - 1, max(k, (k * (k - 1)) // 2) + 1) for k in node_counts]
        )
        lb = _symmetric(rng, n, 6)
        ub = lb + _symmetric(rng, n, 4)
        ubs = lb + _symmetric(rng, n, 3)
        ubs = np.minimum(ubs, ub + _symmetric(rng, n, 2))
        _write_role(root / "LB", dataset, "LB", lb, node_counts, edge_counts)
        _write_role(root / "UB", dataset, "UB", ub, node_counts, edge_counts)
        _write_role(root / "UB_SENSITIVITY", dataset, "UBS", ubs, node_counts, edge_counts)
    return root, datasets


def _config(root: Path, out: Path, datasets: tuple[str, ...]) -> aga.AnalysisConfig:
    """Build a config over a synthetic tree."""
    return aga.AnalysisConfig(
        lb_dir=root / "LB",
        ub_dir=root / "UB",
        ubs_dir=root / "UB_SENSITIVITY",
        input_dir=root / "exported_suite2",
        out_dir=out,
        datasets=datasets,
        datasets_explicit=True,
        make_figures=False,
    )


# ---------------------------------------------------------------------------
# The width formula, including UB == 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lb", "ub", "expected"),
    [
        (0.0, 0.0, 0.0),  # the frozen rule: a closed bracket at zero, not undefined
        (0.0, 1.0, 1.0),
        (1.0, 1.0, 0.0),
        (2.0, 4.0, 0.5),
        (3.0, 4.0, 0.25),
    ],
)
def test_bracket_width_scalar_cases(lb: float, ub: float, expected: float) -> None:
    """The width formula at the boundary, including the UB == 0 case."""
    result = aga.bracket_width(np.array([lb]), np.array([ub]))
    np.testing.assert_allclose(result, [expected], rtol=0, atol=1e-12)


def test_bracket_width_zero_ub_is_zero_not_nan() -> None:
    """A whole array of UB == 0 gives zeros and no nan, inf or warning."""
    lb = np.zeros(64)
    ub = np.zeros(64)
    with np.errstate(all="raise"):
        width = aga.bracket_width(lb, ub)
    assert np.isfinite(width).all()
    assert not width.any()


def test_bracket_width_zero_ub_pairs_are_not_filtered() -> None:
    """Pairs with UB == 0 stay in the population; GED is legitimately 0."""
    lb = np.array([0.0, 0.0, 1.0])
    ub = np.array([0.0, 2.0, 1.0])
    width = aga.bracket_width(lb, ub)
    assert width.size == 3
    np.testing.assert_allclose(width, [0.0, 1.0, 0.0])


@pytest.mark.parametrize(
    ("lb", "ub", "message"),
    [
        (np.array([2.0]), np.array([1.0]), "violate LB <= UB"),
        (np.array([-1.0]), np.array([1.0]), "negative"),
        (np.array([np.nan]), np.array([1.0]), "non-finite"),
        (np.array([np.inf]), np.array([1.0]), "non-finite"),
    ],
)
def test_bracket_width_rejects_invalid_input(lb: np.ndarray, ub: np.ndarray, message: str) -> None:
    """Bad bounds raise rather than propagating silently."""
    with pytest.raises(aga.InputError, match=message):
        aga.bracket_width(lb, ub)


def test_bracket_width_shape_mismatch() -> None:
    """Mismatched shapes raise."""
    with pytest.raises(aga.InputError, match="shape mismatch"):
        aga.bracket_width(np.zeros(3), np.zeros(4))


# ---------------------------------------------------------------------------
# The absolute gap, and its divergence from the relative width
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lb", "ub", "expected"),
    [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 1.0, 0.0),
        (2.0, 4.0, 2.0),
        (46.0, 123.0, 77.0),
    ],
)
def test_absolute_gap_scalar_cases(lb: float, ub: float, expected: float) -> None:
    """The absolute gap is UB - LB in edit operations."""
    result = aga.absolute_gap(np.array([lb]), np.array([ub]))
    np.testing.assert_allclose(result, [expected], rtol=0, atol=1e-12)


def test_absolute_gap_and_width_share_their_validation() -> None:
    """Both measures reject exactly the same bad input."""
    bad = [
        (np.array([2.0]), np.array([1.0])),
        (np.array([-1.0]), np.array([1.0])),
        (np.array([np.nan]), np.array([1.0])),
        (np.zeros(3), np.zeros(4)),
    ]
    for lb, ub in bad:
        with pytest.raises(aga.InputError):
            aga.bracket_width(lb, ub)
        with pytest.raises(aga.InputError):
            aga.absolute_gap(lb, ub)


def test_absolute_gap_equals_width_times_upper_bound() -> None:
    """gap == w * UB wherever UB > 0, which is the algebraic link."""
    rng = np.random.default_rng(31)
    ub = rng.integers(1, 60, size=500).astype(np.float64)
    lb = np.floor(ub * rng.random(500))
    np.testing.assert_allclose(
        aga.absolute_gap(lb, ub), aga.bracket_width(lb, ub) * ub, rtol=1e-12, atol=1e-12
    )


def test_constant_absolute_gap_gives_a_falling_relative_width() -> None:
    """The mechanism the report states, as an executable assertion.

    A bound whose absolute gap is *constant* in ``n`` already produces a
    strictly falling relative width, because the ratio's denominator grows.
    This is why a falling relative width does not imply a tightening bound.
    """
    n = np.arange(5, 100, dtype=np.float64)
    ub = 1.5 * n  # an upper bound that grows with n
    lb = ub - 4.0  # a gap that does not
    gap = aga.absolute_gap(lb, ub)
    width = aga.bracket_width(lb, ub)
    assert aga.ols_fit(n, gap).slope == pytest.approx(0.0, abs=1e-12)
    assert aga.ols_fit(n, width).slope < 0.0


def test_relative_slope_can_oppose_the_absolute_slope() -> None:
    """A rising absolute gap can coexist with a falling relative width."""
    n = np.arange(5, 100, dtype=np.float64)
    ub = 3.0 * n
    lb = ub - (2.0 + 0.5 * n)  # the gap rises, but slower than UB does
    assert aga.ols_fit(n, aga.absolute_gap(lb, ub)).slope > 0.0
    assert aga.ols_fit(n, aga.bracket_width(lb, ub)).slope < 0.0


def test_lb_over_ub_is_redundant_with_the_relative_width() -> None:
    """LB/UB is exactly 1 - w, so it carries no independent information."""
    rng = np.random.default_rng(37)
    ub = rng.integers(1, 60, size=400).astype(np.float64)
    lb = np.floor(ub * rng.random(400))
    width = aga.bracket_width(lb, ub)
    np.testing.assert_allclose(lb / ub, 1.0 - width, rtol=1e-12, atol=1e-12)
    n = rng.integers(2, 90, size=400).astype(np.float64)
    np.testing.assert_allclose(
        aga.ols_fit(n, lb / ub).slope, -aga.ols_fit(n, width).slope, rtol=1e-9, atol=1e-12
    )


def test_absolute_gap_is_bootstrapped_at_the_same_tier(
    synthetic_tree: tuple[Path, list[str]], tmp_path: Path
) -> None:
    """Both measures get a CI, from the same graph resamples and tier."""
    root, datasets = synthetic_tree
    out = tmp_path / "abs"
    results = aga.run_analysis(_config(root, out, tuple(datasets)), command="pytest")
    for dataset in datasets:
        block = results["per_dataset"][dataset]["s71"]
        for arm in ("primary", "sensitivity", "primary_absolute", "sensitivity_absolute"):
            assert arm in block, arm
            assert block[arm]["slope_ci"]["bootstrap"]["resampling_unit"] == "graph"
            assert block[arm]["slope_ci"]["bootstrap"]["replicates"] == 2000
        assert "absolute gap" in block["primary_absolute"]["measure"]
        assert "relative width" in block["primary"]["measure"]


def test_absolute_gap_reaches_every_report_surface(
    synthetic_tree: tuple[Path, list[str]], tmp_path: Path
) -> None:
    """The gap is fitted per dataset, per density cell, per stratum and pooled."""
    root, datasets = synthetic_tree
    out = tmp_path / "surfaces"
    results = aga.run_analysis(_config(root, out, tuple(datasets)), command="pytest")
    assert "primary_absolute" in results["pooled"]["slope"]
    for cell in results["pooled"]["density_cells"]["cells"]:
        assert "primary_absolute" in cell
        assert "mean_upper_bound" in cell
    for row in results["pooled"]["size_strata"]:
        assert "gap_primary" in row
        assert "upper_bound" in row
    for row in results["pooled"]["density_strata"]:
        assert "gap_primary" in row
    profile = results["per_dataset"][datasets[0]]["s71"]["size_profile"]
    assert {"mean_gap", "mean_width", "mean_upper_bound"} <= set(profile[0])


# ---------------------------------------------------------------------------
# §7.1c gate attribution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("primary", "arm", "ratio", "fired"),
    [
        (0.7043, 0.0859, 8.199, True),
        (1.4557, 0.9444, 1.5414, True),
        (0.5, 0.5, 1.0, False),  # identical widening: the gate contributes nothing
        (0.2, 0.5, 0.4, False),  # the arm widens faster: the rule does not fire
    ],
)
def test_gate_attribution_ratio(primary: float, arm: float, ratio: float, fired: bool) -> None:
    """The ratio and the fired flag follow the frozen decision rule."""
    result = aga.gate_attribution(primary, arm)
    assert result["ratio"] == pytest.approx(ratio, rel=1e-3)
    assert result["fired"] is fired


@pytest.mark.parametrize("arm", [0.0, -0.5])
def test_gate_attribution_guards_a_non_widening_denominator(arm: float) -> None:
    """A zero or negative arm slope gives nan, never a signed infinity.

    On the full cohort an arm slope could be non-positive; a ratio against it
    is not interpretable and must not be printed as though it were.
    """
    result = aga.gate_attribution(1.0, arm)
    assert math.isnan(result["ratio"])
    assert result["fired"] is True


def test_gate_attribution_handles_nan_slopes() -> None:
    """A nan slope propagates to nan and does not fire the rule."""
    result = aga.gate_attribution(math.nan, 0.5)
    assert math.isnan(result["ratio"])
    assert result["fired"] is False


def _fake_dataset(mean_nodes: float, primary: float, arm: float) -> dict[str, object]:
    """Build the minimal per-dataset block the summary reads."""
    return {
        "mean_nodes": mean_nodes,
        "slope_role": "unconfounded",
        "s71": {"gate_attribution": aga.gate_attribution(primary, arm)},
    }


def test_gate_attribution_summary_orders_by_mean_nodes_and_detects_monotone() -> None:
    """Rows sort by mean node count and the monotone fall is detected."""
    per_dataset = {
        "big": _fake_dataset(30.0, 1.5, 1.0),  # ratio 1.5
        "small": _fake_dataset(4.0, 8.0, 1.0),  # ratio 8.0
        "mid": _fake_dataset(12.0, 2.0, 1.0),  # ratio 2.0
    }
    summary = aga.gate_attribution_summary(per_dataset, ["big", "small", "mid"])
    assert [row["dataset"] for row in summary["rows_by_mean_nodes"]] == ["small", "mid", "big"]
    assert summary["ratio_falls_monotonically_with_mean_nodes"] is True
    assert summary["n_fired"] == 3
    assert summary["fired_in_all"] is True
    assert summary["max_ratio_dataset"] == "small"
    assert summary["min_ratio_dataset"] == "big"
    assert summary["gate_share_at_min_ratio"] == pytest.approx(1.0 - 1.0 / 1.5)


def test_gate_attribution_summary_detects_a_non_monotone_cohort() -> None:
    """A cohort whose ratio does not fall monotonically is reported as such.

    This is the branch the full ten-dataset run takes, and the report must not
    claim a monotone fall that the data does not show.
    """
    per_dataset = {
        "a": _fake_dataset(4.0, 2.0, 1.0),
        "b": _fake_dataset(12.0, 8.0, 1.0),
        "c": _fake_dataset(30.0, 1.5, 1.0),
    }
    summary = aga.gate_attribution_summary(per_dataset, ["a", "b", "c"])
    assert summary["ratio_falls_monotonically_with_mean_nodes"] is False
    counterexample = summary["monotonicity_counterexample"]
    assert counterexample is not None
    assert counterexample["dataset"] == "b"
    assert counterexample["exceeds_dataset"] == "a"


def test_monotonicity_counterexample_is_none_when_monotone() -> None:
    """A monotone cohort has no counter-example to name."""
    per_dataset = {
        "a": _fake_dataset(4.0, 8.0, 1.0),
        "b": _fake_dataset(12.0, 2.0, 1.0),
        "c": _fake_dataset(30.0, 1.5, 1.0),
    }
    summary = aga.gate_attribution_summary(per_dataset, ["a", "b", "c"])
    assert summary["ratio_falls_monotonically_with_mean_nodes"] is True
    assert summary["monotonicity_counterexample"] is None


def test_monotonicity_counterexample_reports_the_largest_violation() -> None:
    """With several violations the largest rise is the one named."""
    per_dataset = {
        "a": _fake_dataset(4.0, 5.0, 1.0),
        "b": _fake_dataset(8.0, 5.5, 1.0),  # small rise
        "c": _fake_dataset(12.0, 2.0, 1.0),
        "d": _fake_dataset(30.0, 6.0, 1.0),  # large rise
    }
    summary = aga.gate_attribution_summary(per_dataset, ["a", "b", "c", "d"])
    assert summary["monotonicity_counterexample"]["dataset"] == "d"


def _role_dataset(mean_nodes: float, primary: float, arm: float, role: str) -> dict[str, object]:
    """Build a per-dataset block carrying an explicit slope role."""
    block = _fake_dataset(mean_nodes, primary, arm)
    block["slope_role"] = role
    return block


def test_gate_ratio_range_is_grouped_by_slope_role() -> None:
    """The range form groups by the role §7.1 assigns, in report order."""
    per_dataset = {
        "small_a": _role_dataset(4.07, 8.20, 1.0, "small-n constraint only"),
        "small_b": _role_dataset(8.71, 2.70, 1.0, "small-n constraint only"),
        "mid": _role_dataset(11.03, 2.28, 1.0, "intermediate"),
        "big_a": _role_dataset(14.02, 2.06, 1.0, "unconfounded"),
        "big_b": _role_dataset(31.68, 1.54, 1.0, "unconfounded"),
    }
    summary = aga.gate_attribution_summary(per_dataset, list(per_dataset))
    by_role = summary["by_slope_role"]
    assert [entry["slope_role"] for entry in by_role] == list(aga.SLOPE_ROLE_ORDER)
    small, _, large = by_role
    assert small["min_ratio"] == pytest.approx(2.70)
    assert small["max_ratio"] == pytest.approx(8.20)
    assert small["min_ratio_dataset"] == "small_b"
    assert large["min_ratio"] == pytest.approx(1.54)
    assert large["max_ratio"] == pytest.approx(2.06)
    assert summary["small_n_and_unconfounded_ranges_disjoint"] is True
    assert large["gate_share_at_min_ratio"] == pytest.approx(1.0 - 1.0 / 1.54)


def test_gate_ratio_ranges_can_overlap_and_are_reported_as_such() -> None:
    """Overlapping role ranges must not be claimed as disjoint."""
    per_dataset = {
        "small": _role_dataset(4.0, 2.0, 1.0, "small-n constraint only"),
        "big": _role_dataset(30.0, 3.0, 1.0, "unconfounded"),
    }
    summary = aga.gate_attribution_summary(per_dataset, list(per_dataset))
    assert summary["small_n_and_unconfounded_ranges_disjoint"] is False


def test_non_monotone_cohort_prints_the_range_form_and_never_the_sequence_form() -> None:
    """The report drops the ordered-sequence claim when the data loses it.

    This is the guard that matters: the eight-dataset cohort was monotone and
    the ten-dataset cohort is not, so the sequence wording must be gated on the
    flag rather than on what an earlier run happened to show.
    """
    per_dataset = {
        "small": _role_dataset(4.07, 8.20, 1.0, "small-n constraint only"),
        "coil": _role_dataset(21.54, 1.66, 1.0, "unconfounded"),
        "mutag": _role_dataset(28.53, 1.87, 1.0, "unconfounded"),
    }
    summary = aga.gate_attribution_summary(per_dataset, list(per_dataset))
    assert summary["ratio_falls_monotonically_with_mean_nodes"] is False
    rendered = "\n".join(aga._report_gate_pattern(summary))
    assert aga.MONOTONE_PHRASE not in rendered
    assert "NOT monotone" in rendered
    assert "`mutag` is the counter-example" in rendered
    assert "ratio range" in rendered
    assert "would survive replacing" in rendered
    assert "1.66x" in rendered and "8.20x" in rendered


def test_monotone_cohort_still_prints_the_sequence_sentence() -> None:
    """The monotone branch keeps its stronger wording when it is earned."""
    per_dataset = {
        "small": _role_dataset(4.0, 8.0, 1.0, "small-n constraint only"),
        "big": _role_dataset(30.0, 1.5, 1.0, "unconfounded"),
    }
    summary = aga.gate_attribution_summary(per_dataset, list(per_dataset))
    rendered = "\n".join(aga._report_gate_pattern(summary))
    assert aga.MONOTONE_PHRASE in rendered
    assert "NOT monotone" not in rendered


def test_gate_pattern_is_empty_without_any_interpretable_ratio() -> None:
    """No ratio, no points 2 and 3 -- rather than a section built on nan."""
    per_dataset = {"a": _role_dataset(4.0, 1.0, 0.0, "unconfounded")}
    summary = aga.gate_attribution_summary(per_dataset, ["a"])
    assert aga._report_gate_pattern(summary) == []


def test_gate_attribution_summary_reports_a_rule_that_did_not_fire() -> None:
    """A dataset where the arm widens faster is counted honestly."""
    per_dataset = {
        "a": _fake_dataset(4.0, 2.0, 1.0),
        "b": _fake_dataset(12.0, 0.5, 1.0),
    }
    summary = aga.gate_attribution_summary(per_dataset, ["a", "b"])
    assert summary["n_fired"] == 1
    assert summary["fired_in_all"] is False


def test_gate_attribution_summary_never_fits_a_trend() -> None:
    """The summary is descriptive: no regression is fitted to the ratio."""
    per_dataset = {"a": _fake_dataset(4.0, 8.0, 1.0), "b": _fake_dataset(30.0, 1.5, 1.0)}
    summary = aga.gate_attribution_summary(per_dataset, ["a", "b"])
    assert summary["trend_is_fitted"] is False
    assert "slope" not in summary
    assert "no regression is fitted" in summary["trend_note"]
    assert "BIPARTITE remains primary" in summary["ruling"]


def test_gate_attribution_reaches_the_report_and_both_json_files(
    synthetic_tree: tuple[Path, list[str]], tmp_path: Path
) -> None:
    """The verdict is computed once and serialised where the close can find it."""
    root, datasets = synthetic_tree
    out = tmp_path / "gate"
    results = aga.run_analysis(_config(root, out, tuple(datasets)), command="pytest")
    gate = results["pooled"]["gate_attribution"]
    assert gate["n_datasets"] == len(datasets)
    summary = json.loads((out / "data" / "summary.json").read_text())
    slopes = json.loads((out / "data" / "s71_within_dataset_slopes.json").read_text())
    assert summary["gate_attribution"]["n_fired"] == gate["n_fired"]
    assert slopes["gate_attribution"]["n_fired"] == gate["n_fired"]
    for dataset in datasets:
        assert "gate_attribution" in slopes["datasets"][dataset]
    report = (out / "REPORT.md").read_text()
    assert "VERDICT" in report
    assert f"fired in {gate['n_fired']}/{gate['n_datasets']}" in report
    assert "BIPARTITE remains primary" in report


# ---------------------------------------------------------------------------
# Strict upper triangle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 5, 11])
def test_upper_triangle_is_strict_and_row_major(n: int) -> None:
    """The extraction is k = 1 and matches an explicit double loop."""
    matrix = np.arange(n * n, dtype=np.float64).reshape(n, n)
    matrix = np.triu(matrix, 1) + np.triu(matrix, 1).T
    extracted = aga.upper_triangle(matrix)
    expected = np.array([matrix[i, j] for i in range(n) for j in range(i + 1, n)])
    assert extracted.size == n * (n - 1) // 2
    np.testing.assert_array_equal(extracted, expected)


def test_upper_triangle_excludes_the_diagonal() -> None:
    """A matrix with a poisoned diagonal never leaks it into the vector."""
    matrix = np.zeros((6, 6))
    np.fill_diagonal(matrix, 999.0)
    assert not aga.upper_triangle(matrix).any()


def test_upper_triangle_rejects_non_square() -> None:
    """A non-square input raises."""
    with pytest.raises(aga.InputError, match="square"):
        aga.upper_triangle(np.zeros((3, 4)))


def test_upper_triangle_single_graph_is_empty() -> None:
    """One graph induces no pairs."""
    assert aga.upper_triangle(np.zeros((1, 1))).size == 0


# ---------------------------------------------------------------------------
# The D15 tier assignment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset", sorted(aga.TIER3_DATASETS))
def test_tier3_is_exactly_coil_del_and_mutagenicity(dataset: str) -> None:
    """Tier 3 selects 1,000 replicates on a 2,000,000-pair subsample."""
    tier = aga.bootstrap_tier(dataset)
    assert tier.tier == 3
    assert tier.replicates == 1000
    assert tier.subsample == 2_000_000


@pytest.mark.parametrize("dataset", [k for k in aga.DATASET_KEYS if k not in aga.TIER3_DATASETS])
def test_other_eight_datasets_use_all_pairs(dataset: str) -> None:
    """The other eight run 2,000 replicates over all induced pairs."""
    tier = aga.bootstrap_tier(dataset)
    assert tier.tier in (1, 2)
    assert tier.replicates == 2000
    assert tier.subsample is None


def test_tier_assignment_covers_the_whole_cohort() -> None:
    """Exactly two of the ten datasets are tier 3."""
    tiers = {k: aga.bootstrap_tier(k).tier for k in aga.DATASET_KEYS}
    assert sum(1 for v in tiers.values() if v == 3) == 2
    assert set(aga.DATASET_KEYS) == set(tiers)


def test_tier_dict_states_its_effort() -> None:
    """Every tier description carries replicates, budget, seed and unit."""
    for dataset in aga.DATASET_KEYS:
        payload = aga.bootstrap_tier(dataset).as_dict()
        assert payload["resampling_unit"] == "graph"
        assert payload["seed"] == 42
        assert payload["replicates"] in (1000, 2000)
        assert payload["within_replicate_pairs"] in ("all", 2_000_000)


# ---------------------------------------------------------------------------
# The bootstrap resamples graphs, not pairs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_graphs", [4, 9, 30])
def test_replicate_selection_matches_the_repo_convention(n_graphs: int) -> None:
    """The seeding rule is the one ged_bakeoff_analysis already froze."""
    reference = pytest.importorskip("benchmarks.real_data.eval_setup.ged_bakeoff_analysis")
    for replicate in range(5):
        np.testing.assert_array_equal(
            aga.replicate_selection(n_graphs, 42, replicate),
            reference.replicate_selection(n_graphs, 42, replicate),
        )


@pytest.mark.parametrize("n_graphs", [5, 12, 40])
def test_resampling_unit_is_the_graph(n_graphs: int) -> None:
    """A replicate draws n_graphs graph indices with replacement.

    The unit being the graph, not the pair, is exactly what makes duplicated
    graph indices appear; a pair-level bootstrap would draw pair indices and
    could never produce a repeated *graph*.
    """
    selection = aga.replicate_selection(n_graphs, 42, 0)
    assert selection.shape == (n_graphs,)
    assert selection.min() >= 0
    assert selection.max() < n_graphs
    duplicated = any(
        np.unique(aga.replicate_selection(n_graphs, 42, r)).size < n_graphs for r in range(20)
    )
    assert duplicated, "with replacement over 20 replicates a graph must repeat"


def test_induced_pairs_carry_the_resample_multiplicity() -> None:
    """A graph drawn twice contributes its pairs twice; self-pairs are dropped."""
    selection = np.array([0, 0, 1, 2], dtype=np.int64)
    lo, hi = aga.induced_pair_slots(selection)
    pairs = sorted(zip(lo.tolist(), hi.tolist(), strict=True))
    # slot pairs: (0,0) dropped, then (0,1) x2, (0,2) x2, (1,2) x1
    assert pairs == [(0, 1), (0, 1), (0, 2), (0, 2), (1, 2)]


def test_induced_pairs_match_the_repo_convention() -> None:
    """The self-pair rule agrees with ged_bakeoff_analysis.induced_pairs."""
    reference = pytest.importorskip("benchmarks.real_data.eval_setup.ged_bakeoff_analysis")
    n_graphs = 15
    for replicate in range(4):
        selection = aga.replicate_selection(n_graphs, 42, replicate)
        lo, hi = aga.induced_pair_slots(selection)
        theirs = reference.induced_pairs(n_graphs, selection)
        assert lo.size == theirs.size


@pytest.mark.parametrize("n_graphs", [3, 6, 13])
def test_weighted_replicate_sums_equal_explicit_enumeration(n_graphs: int) -> None:
    """The quadratic form is an identity, not an approximation.

    This is the correctness proof for the fast path: it must return bit-close
    what materialising every induced pair returns, for matrices with and
    without a non-zero diagonal.
    """
    rng = np.random.default_rng(3)
    width = _symmetric(rng, n_graphs, 5) / 5.0
    n_max = aga.n_max_matrix(rng.integers(2, 40, size=n_graphs))
    matrices = {"y": width, "x": n_max, "xx": n_max * n_max, "xy": n_max * width}
    for replicate in range(6):
        selection = aga.replicate_selection(n_graphs, 42, replicate)
        reference = aga._replicate_sums_reference(matrices, selection)
        counts = np.bincount(selection, minlength=n_graphs).astype(np.float64)[:, None]
        fast = aga._replicate_sums_weighted(matrices, counts)
        for key in ("y", "x", "xx", "xy", "n"):
            np.testing.assert_allclose(
                fast[key][0], reference[key], rtol=1e-9, atol=1e-9, err_msg=key
            )


def test_bootstrap_slope_matches_a_naive_per_replicate_fit() -> None:
    """The bootstrapped slopes equal an explicit per-replicate OLS."""
    rng = np.random.default_rng(11)
    n_graphs = 10
    width = _symmetric(rng, n_graphs, 7) / 7.0
    node_counts = rng.integers(2, 30, size=n_graphs)
    n_max = aga.n_max_matrix(node_counts)
    fast = aga._bootstrap_slopes_full(width, n_max, n_graphs, 25, 42)
    for replicate in range(25):
        selection = aga.replicate_selection(n_graphs, 42, replicate)
        lo, hi = aga.induced_pair_slots(selection)
        expected = aga.ols_fit(n_max[lo, hi], width[lo, hi]).slope
        np.testing.assert_allclose(fast[replicate], expected, rtol=1e-9, atol=1e-12)


def test_bootstrap_is_reproducible_under_the_seed() -> None:
    """Two runs at the same seed give the same CI; a different seed does not."""
    rng = np.random.default_rng(5)
    n_graphs = 14
    width = _symmetric(rng, n_graphs, 5) / 5.0
    n_max = aga.n_max_matrix(rng.integers(2, 40, size=n_graphs))
    tier = aga.BootstrapTier(tier=1, replicates=64, subsample=None)
    first = aga.bootstrap_slope_ci(width, n_max, tier, seed=42)
    second = aga.bootstrap_slope_ci(width, n_max, tier, seed=42)
    third = aga.bootstrap_slope_ci(width, n_max, tier, seed=43)
    assert first["ci_low"] == second["ci_low"]
    assert first["ci_high"] == second["ci_high"]
    assert first["ci_low"] != third["ci_low"]


def test_subsampled_bootstrap_still_resamples_graphs_first() -> None:
    """Tier 3's subsample applies to induced pairs, not to the graph draw.

    The graph resample of replicate r is identical in both paths, so a tier-3
    run and a tier-1 run over the same data share their selections; only the
    within-replicate pair budget differs.
    """
    rng = np.random.default_rng(17)
    n_graphs = 20
    width = _symmetric(rng, n_graphs, 5).astype(np.float32) / 5.0
    n_max = aga.n_max_matrix(rng.integers(2, 40, size=n_graphs)).astype(np.float32)
    total_slots = n_graphs * (n_graphs - 1) // 2
    full = aga._bootstrap_slopes_subsampled(width, n_max, n_graphs, 12, total_slots, 42)
    exact = aga._bootstrap_slopes_full(
        width.astype(np.float64), n_max.astype(np.float64), n_graphs, 12, 42
    )
    # A subsample of every slot pair is the whole induced set, so the two agree.
    np.testing.assert_allclose(full, exact, rtol=1e-6, atol=1e-8)


def test_subsampled_bootstrap_honours_its_budget() -> None:
    """A budget below the population produces different, finite slopes."""
    rng = np.random.default_rng(19)
    n_graphs = 40
    width = _symmetric(rng, n_graphs, 5).astype(np.float32) / 5.0
    n_max = aga.n_max_matrix(rng.integers(2, 60, size=n_graphs)).astype(np.float32)
    small = aga._bootstrap_slopes_subsampled(width, n_max, n_graphs, 8, 50, 42)
    assert np.isfinite(small).all()
    full = aga._bootstrap_slopes_subsampled(
        width, n_max, n_graphs, 8, n_graphs * (n_graphs - 1) // 2, 42
    )
    assert not np.allclose(small, full)


# ---------------------------------------------------------------------------
# Density: the quintiles come from the pair population
# ---------------------------------------------------------------------------


def test_graph_density_formula() -> None:
    """Density is 2m / (n (n - 1)); a complete graph is 1."""
    node_counts = np.array([2, 3, 4, 5])
    edge_counts = np.array([1, 3, 6, 10])
    np.testing.assert_allclose(aga.graph_density(node_counts, edge_counts), np.ones(4))


def test_graph_density_guards_n_below_two() -> None:
    """n < 2 gives 0 rather than a division by zero."""
    with np.errstate(all="raise"):
        result = aga.graph_density(np.array([0, 1, 2]), np.array([0, 0, 1]))
    np.testing.assert_allclose(result, [0.0, 0.0, 1.0])


def test_pair_density_is_the_symmetric_mean() -> None:
    """d_pair = (d1 + d2)/2: symmetric, and equal to d on the diagonal."""
    density = np.array([0.1, 0.7, 0.4])
    matrix = aga.pair_density_matrix(density)
    np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_allclose(np.diag(matrix), density)
    assert matrix[0, 1] == pytest.approx(0.4)


def test_density_quintiles_are_computed_over_pairs_not_graphs() -> None:
    """The quintile edges are the pair-population quantiles.

    Constructed so the two populations disagree: one graph is dense and many
    are sparse, so the pair population is dominated by sparse-sparse pairs and
    the graph population is not.
    """
    density = np.array([1.0] + [0.0] * 9)
    pair_density = aga.upper_triangle(aga.pair_density_matrix(density)).astype(np.float32)
    vectors = [
        aga.PairVectors(
            dataset="synthetic",
            n_max=np.full(pair_density.size, 5, dtype=np.int16),
            size_code=np.zeros(pair_density.size, dtype=np.int8),
            width=np.zeros(pair_density.size, dtype=np.float32),
            width_sensitivity=np.zeros(pair_density.size, dtype=np.float32),
            gap=np.zeros(pair_density.size, dtype=np.float32),
            gap_sensitivity=np.zeros(pair_density.size, dtype=np.float32),
            upper_bound=np.zeros(pair_density.size, dtype=np.float32),
            density=pair_density,
            certified=np.zeros(pair_density.size, dtype=bool),
            certified_sensitivity=np.zeros(pair_density.size, dtype=bool),
        )
    ]
    pair_edges = aga.density_quintile_edges(vectors, 5)
    graph_edges = [float(v) for v in np.quantile(density, [0.2, 0.4, 0.6, 0.8])]
    # 45 pairs: 36 sparse-sparse at 0.0 and 9 involving the dense graph at 0.5,
    # so the pair population's top edge interpolates to 0.1.
    np.testing.assert_allclose(pair_edges, [0.0, 0.0, 0.0, 0.1], atol=1e-12)
    # 10 graphs: nine at 0.0 and one at 1.0, so every graph-population edge is 0.
    np.testing.assert_allclose(graph_edges, np.zeros(4), atol=1e-12)
    assert pair_edges != graph_edges, "the two populations must not coincide here"


def test_density_quintiles_partition_the_pair_population() -> None:
    """Every pair lands in exactly one of the five quintiles."""
    rng = np.random.default_rng(23)
    values = rng.random(50_000).astype(np.float32)
    vectors = [
        aga.PairVectors(
            dataset="synthetic",
            n_max=np.full(values.size, 5, dtype=np.int16),
            size_code=np.zeros(values.size, dtype=np.int8),
            width=np.zeros(values.size, dtype=np.float32),
            width_sensitivity=np.zeros(values.size, dtype=np.float32),
            gap=np.zeros(values.size, dtype=np.float32),
            gap_sensitivity=np.zeros(values.size, dtype=np.float32),
            upper_bound=np.zeros(values.size, dtype=np.float32),
            density=values,
            certified=np.zeros(values.size, dtype=bool),
            certified_sensitivity=np.zeros(values.size, dtype=bool),
        )
    ]
    edges = aga.density_quintile_edges(vectors, 5)
    codes = aga.density_codes(values, edges)
    counts = np.bincount(codes, minlength=5)
    assert counts.sum() == values.size
    assert counts.min() > 0.19 * values.size


# ---------------------------------------------------------------------------
# Size strata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_max", "label"),
    [
        (2, "2"),
        (3, "3-5"),
        (5, "3-5"),
        (6, "6-9"),
        (9, "6-9"),
        (10, "10-12"),
        (12, "10-12"),
        (13, "13-20"),
        (20, "13-20"),
        (21, "21-40"),
        (40, "21-40"),
        (41, ">40"),
        (98, ">40"),
    ],
)
def test_size_bins_are_statistics_section_8(n_max: int, label: str) -> None:
    """The strata are §8's report bins, with the added n = 2 bin."""
    code = int(aga.size_bin_codes(np.array([n_max]))[0])
    assert aga.SIZE_BIN_LABELS[code] == label


def test_size_bins_reject_n_below_two() -> None:
    """A graph below the Suite-2 filter raises rather than binning silently."""
    with pytest.raises(aga.InputError, match="below the lowest stratum edge"):
        aga.size_bin_codes(np.array([1]))


def test_size_bins_are_not_the_subsample_bins() -> None:
    """There are seven report strata, not §1.1's fourteen draw bins."""
    assert len(aga.SIZE_BIN_LABELS) == 7
    assert len(aga.SIZE_BIN_EDGES) == 7


# ---------------------------------------------------------------------------
# OLS
# ---------------------------------------------------------------------------


def test_ols_recovers_an_exact_line() -> None:
    """A noiseless line is recovered exactly with R^2 = 1."""
    x = np.arange(50, dtype=np.float64)
    fit = aga.ols_fit(x, 3.0 + 0.25 * x)
    assert fit.slope == pytest.approx(0.25)
    assert fit.intercept == pytest.approx(3.0)
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.n == 50


@pytest.mark.parametrize("x", [np.zeros(10), np.full(10, 7.0)])
def test_ols_returns_nan_on_zero_variance(x: np.ndarray) -> None:
    """A constant regressor gives nan, not a spurious slope."""
    fit = aga.ols_fit(x, np.arange(10, dtype=np.float64))
    assert math.isnan(fit.slope)


@pytest.mark.parametrize("size", [0, 1])
def test_ols_degenerate_sample_sizes(size: int) -> None:
    """Fewer than two observations gives nan and the honest count."""
    fit = aga.ols_fit(np.zeros(size), np.zeros(size))
    assert math.isnan(fit.slope)
    assert fit.n == size


def test_percentile_ci_on_all_nan_is_nan() -> None:
    """A degenerate bootstrap reports nan and zero finite replicates."""
    result = aga.percentile_ci(np.full(10, np.nan))
    assert math.isnan(result["ci_low"])
    assert result["n_finite_replicates"] == 0


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


def test_load_role_file_reads_ged_matrix_not_the_crossfill(tmp_path: Path) -> None:
    """The role's own ged_matrix is read even when lb/ub_matrix disagree."""
    node_counts = np.array([3, 4, 5])
    edge_counts = np.array([3, 4, 5])
    ged = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    path = _write_role(tmp_path / "LB", "linux", "LB", ged, node_counts, edge_counts)
    with np.load(path) as handle:
        payload = dict(handle)
    payload["lb_matrix"] = np.full((3, 3), 99.0)
    payload["ub_matrix"] = np.full((3, 3), -99.0)
    np.savez_compressed(path, **payload)
    loaded = aga.load_role_file(path, "LB")
    np.testing.assert_array_equal(loaded.ged, ged)


def test_load_role_file_rejects_a_swapped_role(tmp_path: Path) -> None:
    """A BIPARTITE file read as the lower bound raises."""
    node_counts = np.array([3, 4])
    path = _write_role(
        tmp_path / "UB", "linux", "UB", np.zeros((2, 2)), node_counts, np.array([3, 4])
    )
    with pytest.raises(aga.InputError, match="role directories may be swapped"):
        aga.load_role_file(path, "LB")


def test_load_role_file_missing_raises(tmp_path: Path) -> None:
    """An absent file names the role and the path."""
    with pytest.raises(aga.InputError, match="not found"):
        aga.load_role_file(tmp_path / "nope.npz", "LB")


# ---------------------------------------------------------------------------
# End to end on the synthetic tree
# ---------------------------------------------------------------------------


def test_end_to_end_writes_report_and_json(
    synthetic_tree: tuple[Path, list[str]], tmp_path: Path
) -> None:
    """The pipeline writes REPORT.md and one JSON per table."""
    root, datasets = synthetic_tree
    out = tmp_path / "out"
    results = aga.run_analysis(_config(root, out, tuple(datasets)), command="pytest")
    assert (out / "REPORT.md").is_file()
    for name in (
        "summary.json",
        "s71_within_dataset_slopes.json",
        "s71_size_profiles.json",
        "s71_density_cells.json",
        "s71_pooled.json",
        "s72_certification.json",
        "s73_strata.json",
        "s74_cost.json",
        "results_full.json",
    ):
        assert (out / "data" / name).is_file(), name
    assert results["cohort"]["complete"] is False
    assert set(results["datasets"]) == set(datasets)


def test_partial_cohort_is_first_class(
    synthetic_tree: tuple[Path, list[str]], tmp_path: Path
) -> None:
    """A single-dataset run succeeds and records the missing cohort members."""
    root, _ = synthetic_tree
    out = tmp_path / "one"
    config = aga.AnalysisConfig(
        lb_dir=root / "LB",
        ub_dir=root / "UB",
        ubs_dir=root / "UB_SENSITIVITY",
        input_dir=root / "exported_suite2",
        out_dir=out,
        datasets=aga.DATASET_KEYS,
        datasets_explicit=False,
        make_figures=False,
    )
    results = aga.run_analysis(config)
    assert results["datasets"] == ["linux", "protein"]
    assert "coil_del" in results["cohort"]["missing_datasets"]
    report = (out / "REPORT.md").read_text()
    assert "COHORT INCOMPLETE" in report


def test_explicitly_named_missing_dataset_raises(
    synthetic_tree: tuple[Path, list[str]], tmp_path: Path
) -> None:
    """Naming an absent dataset is an error; defaulting to all ten is not."""
    root, _ = synthetic_tree
    config = _config(root, tmp_path / "x", ("linux", "coil_del"))
    with pytest.raises(aga.InputError, match="coil_del"):
        aga.run_analysis(config)


def test_report_numbers_appear_in_the_json(
    synthetic_tree: tuple[Path, list[str]], tmp_path: Path
) -> None:
    """Every slope printed in the report is present in the JSON tree."""
    root, datasets = synthetic_tree
    out = tmp_path / "sourced"
    aga.run_analysis(_config(root, out, tuple(datasets)), command="pytest")
    report = (out / "REPORT.md").read_text()
    slopes = json.loads((out / "data" / "s71_within_dataset_slopes.json").read_text())
    for dataset, block in slopes["datasets"].items():
        value = block["primary"]["fit"]["slope"]
        if math.isfinite(value):
            assert f"{value:.5f}" in report, f"{dataset} slope missing from prose"


def test_cost_table_matches_metadata_seconds_total(
    synthetic_tree: tuple[Path, list[str]], tmp_path: Path
) -> None:
    """The strict-upper-triangle seconds sum equals the file's own total."""
    root, datasets = synthetic_tree
    out = tmp_path / "cost"
    results = aga.run_analysis(_config(root, out, tuple(datasets)), command="pytest")
    for dataset in datasets:
        for row in results["per_dataset"][dataset]["s74"]["roles"]:
            assert row["triu_sum_matches_metadata_total"] is True


def test_certification_and_width_agree(
    synthetic_tree: tuple[Path, list[str]], tmp_path: Path
) -> None:
    """The certification rate is the share of pairs with width exactly 0.

    They are two views of one distribution, which is the reason the report
    insists the two tables be read together.
    """
    root, datasets = synthetic_tree
    out = tmp_path / "cert"
    results = aga.run_analysis(_config(root, out, tuple(datasets)), command="pytest")
    for dataset in datasets:
        profile = results["per_dataset"][dataset]["s71"]["size_profile"]
        assert sum(row["n_pairs"] for row in profile) == results["per_dataset"][dataset]["n_pairs"]
        rate = results["per_dataset"][dataset]["s72"]["certification_rate_primary"]
        assert 0.0 <= rate <= 1.0


def test_secondary_cells_report_their_drops(
    synthetic_tree: tuple[Path, list[str]], tmp_path: Path
) -> None:
    """Underpopulated (dataset x quintile) cells are listed, not omitted."""
    root, datasets = synthetic_tree
    out = tmp_path / "cells"
    results = aga.run_analysis(_config(root, out, tuple(datasets)), command="pytest")
    cells = results["pooled"]["density_cells"]
    assert cells["n_cells_dropped"] > 0
    assert len(cells["dropped"]) == cells["n_cells_dropped"]
    for entry in cells["dropped"]:
        assert entry["n_pairs"] < cells["min_cell_pairs"]
        assert "reason" in entry


def test_config_rejects_unknown_dataset_key() -> None:
    """A typo in --datasets is caught before any file is opened."""
    parser = aga.build_parser()
    args = parser.parse_args(
        [
            "--lb-dir",
            "a",
            "--ub-dir",
            "b",
            "--ubs-dir",
            "c",
            "--input-dir",
            "d",
            "--out",
            "e",
            "--datasets",
            "linux,not_a_dataset",
        ]
    )
    with pytest.raises(aga.InputError, match="unknown dataset keys"):
        aga.config_from_args(args)


def test_config_orders_datasets_canonically() -> None:
    """--datasets is reordered into the cohort order, whatever the user typed."""
    parser = aga.build_parser()
    args = parser.parse_args(
        [
            "--lb-dir",
            "a",
            "--ub-dir",
            "b",
            "--ubs-dir",
            "c",
            "--input-dir",
            "d",
            "--out",
            "e",
            "--datasets",
            "protein,linux",
        ]
    )
    config = aga.config_from_args(args)
    assert config.datasets == ("linux", "protein")
    assert config.datasets_explicit is True
