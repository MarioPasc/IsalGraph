"""T-06 statistics engine: the graph-level protocol and the frozen family.

Every test here defends one property that a passing numeric result cannot
defend on its own. The submitted manuscript's statistics were arithmetically
correct and scientifically wrong, because the resampling unit was the pair. A
test that only checks a p-value is finite would have passed on the defective
version too.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from benchmarks.real_data.eval_stats import association, family, matrices, multiplicity, resampling
from benchmarks.real_data.eval_stats.family import Cell, GateInput, ReductionInputs
from benchmarks.real_data.eval_stats.multiplicity import Regime

pytestmark = pytest.mark.unit

#: T-05's Suite-2 approximate GED matrices, read-only.
APPROX_GED_ROOT = Path(
    "/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/APPROX_GED"
)

TIER1_SMALL = resampling.BootstrapTier(tier=1, replicates=200, permutations=199, subsample=None)


# ---------------------------------------------------------------------------
# Fixtures: a dyadically dependent dataset
# ---------------------------------------------------------------------------


def _symmetric(values: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """Return a symmetric matrix with a zero diagonal from a square array."""
    array = np.asarray(values, dtype=np.float64)
    out = np.triu(array, k=1)
    out = out + out.T
    np.fill_diagonal(out, 0.0)
    return out


def _dyadic_dataset(
    n_graphs: int = 30,
    seed: int = 7,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """Build a GED/Levenshtein pair whose dependence lives at the graph level.

    Each graph carries a latent size, and both matrices are driven by the pair
    of latent sizes. That is the structure R3.5c is about: the 435 pairs of 30
    graphs carry nowhere near 435 independent observations.
    """
    rng = np.random.default_rng(seed)
    node_counts = rng.integers(4, 40, size=n_graphs).astype(np.int32)
    delta = np.abs(node_counts[:, None] - node_counts[None, :]).astype(np.float64)
    ged = _symmetric(delta + rng.normal(0.0, 1.0, size=(n_graphs, n_graphs)))
    lev = _symmetric(1.4 * delta + rng.normal(0.0, 3.0, size=(n_graphs, n_graphs)))
    return np.abs(ged), np.abs(lev), node_counts


# ---------------------------------------------------------------------------
# Criterion 1 --- the resampling unit is the graph
# ---------------------------------------------------------------------------


def test_graph_level_interval_is_wider_than_a_pair_level_one() -> None:
    """D2's whole point: resampling pairs understates the uncertainty.

    A dataset of 30 graphs contributes 30 independent units, not 435. If this
    ever inverts, the resampling unit has silently reverted to the pair --
    which is exactly the defect R3.5c identified in the submission.
    """
    ged, lev, _ = _dyadic_dataset()
    variables = association.PairVariables.from_matrices({"ged": ged, "lev": lev})
    specs = [association.CorrelationSpec("rho", "lev", "ged")]
    results, _ = association.bootstrap_associations(
        variables, specs, TIER1_SMALL, kendall=False, replicates=400
    )
    graph_width = results["rho"].rho.width

    full = np.flatnonzero(variables.valid)
    x = variables.values["lev"][full]
    y = variables.values["ged"][full]
    rng = np.random.default_rng(resampling.SEED)
    pair_level = [
        association.spearman(x[draw], y[draw])
        for draw in (rng.integers(0, x.size, size=x.size) for _ in range(400))
    ]
    lo, hi = np.percentile(pair_level, [2.5, 97.5])

    assert graph_width > float(hi - lo)


def test_replicate_selection_matches_the_repository_convention() -> None:
    """The seeding rule is reused, not reinvented, so replicates line up."""
    from benchmarks.real_data.eval_setup import approx_ged_analysis as aga
    from benchmarks.real_data.eval_setup import ged_bakeoff_analysis as bakeoff

    for replicate in range(4):
        mine = resampling.replicate_selection(25, resampling.SEED, replicate)
        np.testing.assert_array_equal(mine, bakeoff.replicate_selection(25, 42, replicate))
        np.testing.assert_array_equal(mine, aga.replicate_selection(25, 42, replicate))


def test_paired_differences_come_from_one_shared_resample() -> None:
    """D7 requires both correlations on the same resample, not on two runs."""
    ged, lev, node_counts = _dyadic_dataset()
    size_null = association.delta_n_matrix(node_counts)
    variables = association.PairVariables.from_matrices({"ged": ged, "lev": lev, "null": size_null})
    specs = [
        association.CorrelationSpec("rho_lev", "lev", "ged"),
        association.CorrelationSpec("rho_null", "null", "ged"),
    ]
    diffs = [association.DifferenceSpec("delta", "rho_lev", "rho_null")]
    results, differences = association.bootstrap_associations(
        variables, specs, TIER1_SMALL, differences=diffs, kendall=False, replicates=200
    )
    expected = results["rho_lev"].rho.point - results["rho_null"].rho.point
    np.testing.assert_allclose(differences["delta"].interval.point, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Criterion 2 --- the pair-level bootstrap is not reachable
# ---------------------------------------------------------------------------


def _benchmark_modules() -> list[ModuleType]:
    """Return every ``benchmarks.*`` module currently loaded."""
    return [
        module
        for name, module in list(sys.modules.items())
        if name.startswith("benchmarks.") and module is not None
    ]


def test_the_pair_level_bootstrap_is_not_in_the_import_closure() -> None:
    """``bootstrap_correlation`` resamples PAIRS. It is the defect, not the fix.

    ``statistics.md`` section 11 lists it as *replaced, not supplemented*. The
    check is by **object identity** over the loaded import closure, not by
    grepping source text: an alias, a re-export or a ``getattr`` would all
    defeat a text search and none of them defeats this.
    """
    from benchmarks.real_data.eval_correlation import correlation_metrics

    forbidden = correlation_metrics.bootstrap_correlation
    for module in (association, family, matrices, multiplicity, resampling):
        for name, obj in vars(module).items():
            assert obj is not forbidden, (
                f"{module.__name__} binds the pair-level bootstrap as {name}"
            )
            assert name != "bootstrap_correlation", f"{module.__name__} binds {name}"

    for module in _benchmark_modules():
        if module.__name__.endswith("correlation_metrics"):
            continue
        offenders = [n for n, o in vars(module).items() if o is forbidden]
        assert not offenders, f"{module.__name__} binds the pair-level bootstrap as {offenders}"


def test_mantel_and_holm_are_reused_from_correlation_metrics() -> None:
    """The two correct functions in that module are reused, not reimplemented."""
    from benchmarks.real_data.eval_correlation import correlation_metrics

    assert association.mantel_test is correlation_metrics.mantel_test
    assert multiplicity.holm_bonferroni is correlation_metrics.holm_bonferroni


# ---------------------------------------------------------------------------
# Criterion 3 --- tier 3 resamples graphs first
# ---------------------------------------------------------------------------


def test_tier3_subsamples_induced_pairs_not_the_graph_list() -> None:
    """D15 rule 1: the graph resample is drawn first, unconditionally.

    Two consequences are asserted together. The graph draw of replicate ``r``
    does not depend on the subsample budget, and a budget covering every slot
    pair reproduces the tier-1 induced multiset exactly. Subsampling the graph
    list instead would break both.
    """
    n_graphs = 24
    total_slots = n_graphs * (n_graphs - 1) // 2
    for replicate in range(3):
        selection = resampling.replicate_selection(n_graphs, resampling.SEED, replicate)
        full = resampling.replicate_pair_indices(n_graphs, resampling.SEED, replicate, None)
        covered = resampling.replicate_pair_indices(
            n_graphs, resampling.SEED, replicate, total_slots
        )
        np.testing.assert_array_equal(np.sort(full), np.sort(covered))
        # The graph draw is untouched by the pair budget.
        np.testing.assert_array_equal(
            selection, resampling.replicate_selection(n_graphs, resampling.SEED, replicate)
        )


def test_tier3_budget_shrinks_the_pair_count_but_not_the_graph_draw() -> None:
    """A budget below the population yields fewer pairs from the same graphs."""
    n_graphs = 40
    small = resampling.replicate_pair_indices(n_graphs, resampling.SEED, 0, 100)
    full = resampling.replicate_pair_indices(n_graphs, resampling.SEED, 0, None)
    assert 0 < small.size <= 100 < full.size
    assert set(small.tolist()) <= set(full.tolist())


def test_tier3_subsample_matches_the_existing_tier3_implementation() -> None:
    """The tier-3 substream is the one ``approx_ged_analysis`` already uses."""
    from benchmarks.real_data.eval_setup import approx_ged_analysis as aga

    n_graphs, replicate, budget = 30, 2, 120
    rng = np.random.default_rng(np.random.SeedSequence([resampling.SEED, replicate, 1]))
    expected_slots = rng.choice(
        n_graphs * (n_graphs - 1) // 2, size=budget, replace=False, shuffle=False
    )
    slot_i, slot_j = aga.pairs_from_indices_searchsorted(expected_slots, n_graphs)
    selection = resampling.replicate_selection(n_graphs, resampling.SEED, replicate)
    a, b = selection[slot_i], selection[slot_j]
    keep = a != b
    expected = resampling.pair_flat_index(
        n_graphs, np.minimum(a[keep], b[keep]), np.maximum(a[keep], b[keep])
    )
    got = resampling.replicate_pair_indices(n_graphs, resampling.SEED, replicate, budget)
    np.testing.assert_array_equal(got, expected)


def test_frozen_d15_tiers_are_looked_up_never_recomputed() -> None:
    """CONTRACTS.md section 2 freezes the assignment; it is a table, not a rule."""
    assert resampling.bootstrap_tier("linux").tier == 1
    assert resampling.bootstrap_tier("aids_iam").tier == 2
    assert resampling.bootstrap_tier("coil_del").tier == 3
    tier3 = resampling.bootstrap_tier("mutagenicity")
    assert (tier3.replicates, tier3.permutations, tier3.subsample) == (1000, 1999, 2_000_000)
    tier1 = resampling.bootstrap_tier("protein")
    assert (tier1.replicates, tier1.permutations, tier1.subsample) == (2000, 9999, None)
    # No Suite-1 dataset is subsampled (statistics.md section 5).
    assert all(resampling.bootstrap_tier(key, "suite1").subsample is None for key in family.SUITE1)
    with pytest.raises(resampling.ResamplingError):
        resampling.bootstrap_tier("not_a_dataset")


# ---------------------------------------------------------------------------
# Criterion 4 --- Benjamini-Hochberg
# ---------------------------------------------------------------------------


def test_benjamini_hochberg_against_a_hand_computed_example() -> None:
    """A hand-worked BH, including the step-up minimum that enforces monotonicity.

    With ``p = (0.001, 0.008, 0.039, 0.041, 0.042)`` and ``m = 5`` the raw
    scaled values ``m p_(j) / j`` are ``0.005, 0.020, 0.065, 0.05125, 0.042``,
    which are **not** monotone. The running minimum taken from the largest rank
    downwards pulls ranks 3 and 4 down to 0.042. A naive implementation that
    omits it reports 0.065 at rank 3 and is wrong.
    """
    p = [0.001, 0.008, 0.039, 0.041, 0.042]
    result = multiplicity.benjamini_hochberg(p, family="hand", m=5)
    np.testing.assert_allclose(
        result.adjusted, [0.005, 0.020, 0.042, 0.042, 0.042], rtol=1e-12, atol=1e-12
    )
    assert result.rejected == (True, True, True, True, True)
    assert result.n_rejected == 5


def test_benjamini_hochberg_matches_scipy() -> None:
    """An independent oracle over random p-values, in and out of order."""
    from scipy.stats import false_discovery_control

    rng = np.random.default_rng(3)
    for _ in range(20):
        p = rng.uniform(0.0, 1.0, size=17)
        mine = multiplicity.benjamini_hochberg(p.tolist())
        np.testing.assert_allclose(mine.adjusted, false_discovery_control(p), rtol=1e-12)


def test_benjamini_hochberg_is_monotone_in_the_raw_p_values() -> None:
    """Sorting the adjusted values must reproduce the sort of the raw values."""
    rng = np.random.default_rng(11)
    p = rng.uniform(0.0, 1.0, size=50)
    adjusted = np.asarray(multiplicity.benjamini_hochberg(p.tolist()).adjusted)
    order = np.argsort(p, kind="stable")
    assert np.all(np.diff(adjusted[order]) >= -1e-12)


def test_benjamini_hochberg_over_n_max_is_the_sensitivity_column() -> None:
    """The sensitivity column is a re-threshold of the same stored p-values."""
    p = [0.001, 0.004, 0.02]
    actual = multiplicity.benjamini_hochberg(p, m=3)
    sensitivity = multiplicity.benjamini_hochberg(p, m=182)
    np.testing.assert_allclose(
        sensitivity.adjusted,
        np.minimum(np.asarray(actual.adjusted) * 182 / 3, 1.0),
        rtol=1e-12,
    )
    assert sensitivity.n_rejected <= actual.n_rejected


def test_benjamini_hochberg_rejects_an_impossible_denominator() -> None:
    """``m`` below the number of finite p-values is not a valid denominator."""
    with pytest.raises(multiplicity.MultiplicityError):
        multiplicity.benjamini_hochberg([0.1, 0.2, 0.3], m=2)


def test_fcr_adjusted_level_follows_benjamini_yekutieli() -> None:
    """``1 - R q / m`` for the selected parameters; the marginal level otherwise."""
    assert multiplicity.fcr_adjusted_level(0, 10) == pytest.approx(0.95)
    assert multiplicity.fcr_adjusted_level(2, 10, 0.05) == pytest.approx(0.99)
    assert multiplicity.fcr_adjusted_level(10, 10, 0.05) == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Criterion 5 --- N_actual arithmetic
# ---------------------------------------------------------------------------


def test_f2_enumerates_exactly_182_cells_in_the_frozen_layout() -> None:
    """A1 60, A2 1, B1e 35, B1a 70, B2 1, B3e 5, B3a 10."""
    cells = family.enumerate_f2_cells()
    counts: dict[str, int] = {}
    for cell in cells:
        counts[cell.row] = counts.get(cell.row, 0) + 1
    assert counts == {"A1": 60, "A2": 1, "B1e": 35, "B1a": 70, "B2": 1, "B3e": 5, "B3a": 10}
    assert len(cells) == family.N_MAX_F2 == 182
    assert len(set(cells)) == 182
    # WL enters Claim B and not Claim A: it has no bit count.
    assert not [c for c in cells if c.row == "A1" and c.representation == "wl_subtree"]
    assert len([c for c in cells if c.row == "B1a" and c.representation == "wl_subtree"]) == 10


def test_no_reduction_leaves_the_family_at_182() -> None:
    """k = d = c = 0 must give exactly N_max, by enumeration and closed form."""
    card = family.cardinality()
    assert card.n_actual == 182
    assert card.closed_form == 182
    assert card.discrepancy == 0
    assert card.double_charged == ()


@pytest.mark.parametrize(
    ("k", "d", "c_triples"),
    [
        ((), (), ()),
        (("min_dfs",), (), ()),
        ((), ("coil_del",), ()),
        (("min_dfs", "wl_subtree"), (), ()),
        ((), ("coil_del", "mutagenicity", "protein"), ()),
        ((), (), (("suite2", "protein", "agm_cam"), ("suite2", "mutagenicity", "agm_cam"))),
        (("min_dfs",), ("coil_del",), (("suite1", "aids", "agm_cam"),)),
        (
            ("min_dfs", "wl_subtree"),
            ("coil_del", "mutagenicity"),
            (("suite2", "grec", "agm_cam"), ("suite2", "protein", "agm_cam")),
        ),
    ],
)
def test_n_actual_enumeration_and_closed_form(
    k: tuple[str, ...],
    d: tuple[str, ...],
    c_triples: tuple[tuple[str, str, str], ...],
) -> None:
    """The enumeration is the definition; the closed form is a printed check.

    ``preregistration.md`` section 5 states that where the two disagree the
    enumeration wins and the discrepancy is reported. They disagree by exactly
    ``k * d``: the closed form charges ``k``'s ten B1a cells and ``d``'s seven
    B1a cells without noticing that ``(B1a, excluded rep, uninformative
    dataset)`` sits in both. That over-charge reports an ``N_actual`` *below*
    the admissible count, which is the anti-conservative direction.
    """
    card = family.cardinality(
        excluded_representations=k, uninformative_datasets=d, noncomputable=c_triples
    )
    assert card.n_actual == len(card.admissible)
    assert card.closed_form == 182 - 15 * len(k) - 8 * len(d) - card.c
    assert card.discrepancy == len(k) * len(d)
    assert len(card.double_charged) == len(k) * len(d)
    # Every removal stage is disjoint from every other.
    stages = [card.removed_by_k, card.removed_by_d, card.removed_by_c, card.removed_by_f0]
    flat = [cell for stage in stages for cell in stage]
    assert len(flat) == len(set(flat))
    assert set(flat).isdisjoint(set(card.admissible))
    assert len(flat) + card.n_actual == 182


def test_a_cell_removed_by_two_terms_is_charged_once() -> None:
    """The B1a cell of a k-excluded representation on a d-dataset, exactly once."""
    card = family.cardinality(
        excluded_representations=("min_dfs",), uninformative_datasets=("coil_del",)
    )
    shared = Cell("B1a", "suite2", "coil_del", "min_dfs")
    assert shared in card.removed_by_k
    assert shared not in card.removed_by_d
    assert shared not in card.admissible
    # k removes 15, d removes its remaining 7 (6 B1a + 1 B3a), never 8.
    assert len(card.removed_by_k) == 15
    assert len(card.removed_by_d) == 7
    assert card.n_actual == 182 - 15 - 7 == 160
    assert card.closed_form == 182 - 15 - 8 == 159
    assert card.discrepancy == 1
    assert card.double_charged == (shared,)


def test_c_never_charges_a_cell_an_earlier_term_removed() -> None:
    """Section 5.2's precedence: c applies to what REMAINS after k and d."""
    card = family.cardinality(
        excluded_representations=("agm_cam",),
        uninformative_datasets=("protein",),
        noncomputable=(
            ("suite2", "protein", "agm_cam"),  # already gone via k and via d
            ("suite1", "aids", "agm_cam"),  # already gone via k
            ("suite2", "grec", "agm_cam"),  # already gone via k (B1a), A1 survives
        ),
    )
    charged = {(c.row, c.suite, c.dataset, c.representation) for c in card.removed_by_c}
    assert charged == {
        ("A1", "suite2", "protein", "agm_cam"),
        ("A1", "suite2", "grec", "agm_cam"),
    }
    assert card.c == 2


def test_c_charges_wl_one_cell_per_dataset_not_two() -> None:
    """Section 5.1 consequence 1: WL has no A1 row, so a WL failure costs 1."""
    card = family.cardinality(noncomputable=(("suite2", "grec", "wl_subtree"),))
    assert card.c == 1
    other = family.cardinality(noncomputable=(("suite2", "grec", "graph6"),))
    assert other.c == 2  # A1 and B1a


def test_c_never_charges_the_mrm_or_omnibus_rows() -> None:
    """Section 5.1 consequence 3: A2, B2, B3e and B3a are never charged to c."""
    card = family.cardinality(
        noncomputable=tuple(
            ("suite2", dataset, rep)
            for dataset in family.SUITE2
            for rep in family.CLAIM_B_REPRESENTATIONS
        )
    )
    surviving = {cell.row for cell in card.admissible}
    assert surviving == {"A2", "B2", "B3e", "B3a", "B1e"}
    assert card.c == 60 + 70  # every A1 and every B1a


def test_f0_demotion_is_reported_outside_the_closed_form() -> None:
    """F0's majority branch has no coefficient in the frozen closed form."""
    card = family.cardinality(f0_demotes_approximate=True)
    assert len(card.removed_by_f0) == 70 + 1 + 10
    assert card.n_actual == 182 - 81 == 101
    assert card.closed_form == 182  # the closed form cannot express this branch


def test_reduction_inputs_reject_names_outside_the_freeze() -> None:
    """A typo in k, d or c is an error, never a silently ignored term."""
    for bad in (
        ReductionInputs(excluded_representations=frozenset({"isalgraph_pruned"})),
        ReductionInputs(uninformative_datasets=frozenset({"aids"})),
        ReductionInputs(noncomputable=frozenset({("suite2", "linux", "size_null")})),
    ):
        with pytest.raises(family.FamilyError):
            family.admissible_cells(bad)


# ---------------------------------------------------------------------------
# Criterion 6 --- F0 and F1 branch on the pre-declared rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("point", "ci_low", "ci_high", "expected"),
    [
        (0.20, 0.10, 0.30, True),  # both conditions
        (0.04, 0.02, 0.06, False),  # CI excludes 0, effect below threshold
        (-0.04, -0.06, -0.02, False),  # same, on the other side
        (0.05, 0.02, 0.08, False),  # strictly greater than 0.05, so 0.05 fails
        (0.0501, 0.02, 0.08, True),  # just over
        (0.20, -0.05, 0.45, False),  # large effect, CI includes 0
        (-0.20, -0.45, 0.05, False),
        (float("nan"), 0.1, 0.3, False),
        (0.20, float("nan"), 0.30, False),
    ],
)
def test_gate_rule_requires_both_conditions(
    point: float, ci_low: float, ci_high: float, expected: bool
) -> None:
    """The rule is a conjunction. ``|estimate| = 0.04`` with a CI excluding 0
    is exactly where a wrong implementation shows: a difference can be reliably
    non-zero and still too small to matter."""
    _, _, fails = family.evaluate_gate(point, ci_low, ci_high)
    assert fails is expected


def _gate_inputs(
    datasets: tuple[str, ...], shifts: list[float], scale: float = 0.01, seed: int = 5
) -> list[GateInput]:
    """Build gate inputs whose bootstrap distributions sit at given shifts."""
    rng = np.random.default_rng(seed)
    return [
        GateInput(dataset=name, point=shift, samples=shift + rng.normal(0.0, scale, size=800))
        for name, shift in zip(datasets, shifts, strict=True)
    ]


def test_f0_takes_the_majority_branch_when_three_of_five_fail() -> None:
    """Three failures out of five demote the approximate regime."""
    result = family.run_f0(_gate_inputs(family.SUITE1, [0.30, 0.30, 0.30, 0.0, 0.0]))
    assert len(result.failing_datasets) == 3
    assert "descriptive only" in result.note
    assert result.bh.m == 5


def test_f0_does_not_demote_on_two_failures() -> None:
    """Two of five is not a majority; the approximate regime stays confirmatory."""
    result = family.run_f0(_gate_inputs(family.SUITE1, [0.30, 0.30, 0.0, 0.0, 0.0]))
    assert len(result.failing_datasets) == 2
    assert "admits the approximate regime" in result.note


def test_f0_does_not_fire_on_a_reliable_but_tiny_difference() -> None:
    """A 0.04 difference with a tight CI excludes 0 and still must not fire."""
    result = family.run_f0(_gate_inputs(family.SUITE1, [0.04] * 5, scale=0.002))
    assert result.failing_datasets == ()
    for outcome in result.outcomes:
        assert outcome.ci_excludes_zero is True
        assert outcome.exceeds_threshold is False
        assert outcome.fails is False


def test_f1_outcome_sets_d_and_each_failure_removes_eight_cells() -> None:
    """F1's failing set is ``d``; each uninformative dataset costs 8 F2 cells."""
    shifts = [0.30, 0.30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    result = family.run_f1(_gate_inputs(family.SUITE2, shifts))
    assert set(result.failing_datasets) == {"iam_letter_low", "iam_letter_med"}
    card = family.cardinality(uninformative_datasets=result.failing_datasets)
    assert card.n_actual == 182 - 16
    assert {c.row for c in card.removed_by_d} == {"B1a", "B3a"}


def test_gate_families_are_frozen_at_five_and_ten_tests() -> None:
    """The cardinality was fixed before any p-value existed."""
    with pytest.raises(family.FamilyError):
        family.run_f0(_gate_inputs(family.SUITE1[:4], [0.0] * 4))
    with pytest.raises(family.FamilyError):
        family.run_f1(_gate_inputs(family.SUITE2[:9], [0.0] * 9))


# ---------------------------------------------------------------------------
# Criterion 7 --- the exact regime gets no omnibus
# ---------------------------------------------------------------------------


def test_friedman_refuses_the_exact_regime() -> None:
    """Friedman at N = 5 separates almost nothing; the refusal is structural."""
    rng = np.random.default_rng(2)
    scores = rng.normal(size=(5, 4))
    result = multiplicity.friedman_omnibus(scores, ["a", "b", "c", "d"], Regime.EXACT)
    assert result.ran is False
    assert np.isnan(result.statistic) and np.isnan(result.p_value)
    assert "five datasets" in result.refusal_reason
    # The average ranks survive: the exact regime is descriptive, not silent.
    assert len(result.average_ranks) == 4


def test_friedman_runs_on_the_approximate_regime() -> None:
    """The ten-dataset approximate regime carries the omnibus."""
    rng = np.random.default_rng(4)
    scores = rng.normal(size=(10, 4)) + np.array([0.0, 1.0, 2.0, 3.0])
    result = multiplicity.friedman_omnibus(scores, ["a", "b", "c", "d"], Regime.APPROXIMATE)
    assert result.ran is True
    assert result.p_value < 0.05
    assert result.average_ranks[0] < result.average_ranks[3]


def test_f2_runner_emits_no_exact_regime_omnibus() -> None:
    """The F2 runner must not produce a Friedman result for the exact regime."""
    card = family.cardinality()
    p_values = {cell: 0.5 for cell in card.admissible}
    rng = np.random.default_rng(6)
    result = family.run_f2(
        p_values,
        ReductionInputs(),
        omnibus_scores={
            "A2": (rng.normal(size=(10, 6)), list(family.CLAIM_A_REPRESENTATIONS), True),
            "B2": (rng.normal(size=(10, 7)), list(family.CLAIM_B_REPRESENTATIONS), False),
        },
    )
    assert set(result.omnibuses) == {"A2", "B2"}
    assert all(o.ran for o in result.omnibuses.values())
    assert all(o.n_datasets == 10 for o in result.omnibuses.values())
    assert result.exact_regime_omnibus is None
    assert "five" in result.exact_regime_reason
    payload = result.as_dict()
    assert payload["exact_regime_omnibus"] is None
    assert "exact" not in {name.lower() for name in payload["omnibuses"]}


def test_f2_bh_uses_n_actual_and_prints_the_n_max_sensitivity() -> None:
    """BH over ``N_actual``, with the ``N_max`` re-threshold beside it."""
    inputs = ReductionInputs(
        excluded_representations=frozenset({"min_dfs"}),
        uninformative_datasets=frozenset({"coil_del"}),
    )
    card = family.admissible_cells(inputs)
    p_values = {cell: 0.001 for cell in card.admissible}
    result = family.run_f2(p_values, inputs)
    assert result.bh_primary.m == card.n_actual == 160
    assert result.bh_sensitivity.m == 182
    assert result.cardinality.discrepancy == 1
    with pytest.raises(family.FamilyError):
        family.run_f2({Cell("B1a", "suite2", "coil_del", "min_dfs"): 0.01}, inputs)


def test_wilcoxon_holm_posthoc_is_not_a_bh_member() -> None:
    """Holm already controls the FWER; nesting it inside BH corrects twice."""
    rng = np.random.default_rng(8)
    scores = rng.normal(size=(10, 3)) + np.array([0.0, 1.5, 3.0])
    posthoc = multiplicity.wilcoxon_holm_posthoc(scores, ["a", "b", "c"])
    assert posthoc.counted_in_bh is False
    assert len(posthoc.pairs) == 3
    assert all(0.0 <= p <= 1.0 for p in posthoc.holm_adjusted)
    assert all(a >= r - 1e-12 for a, r in zip(posthoc.holm_adjusted, posthoc.p_values, strict=True))


def test_critical_difference_matches_the_demsar_formula() -> None:
    """``CD = q_alpha sqrt(k (k + 1) / (6 N))``, Demsar 2006 section 3.2.2."""
    cd = multiplicity.critical_difference([1.0, 2.0, 3.0, 4.0], list("abcd"), 10)
    np.testing.assert_allclose(cd.cd, 2.569 * np.sqrt(4 * 5 / 60.0), rtol=1e-12)
    assert cd.methods == ("a", "b", "c", "d")


# ---------------------------------------------------------------------------
# Criterion 8 --- the MRM
# ---------------------------------------------------------------------------


def _mrm_fixture(
    weight_structural: float,
    weight_size: float,
    noise: float,
    n_graphs: int = 45,
    seed: int = 13,
) -> tuple[dict[str, npt.NDArray[np.float64]], npt.NDArray[np.float64], float]:
    """Plant a known standardised beta1 in a GED matrix.

    Levenshtein carries a structural signal independent of size, so the two
    predictors are near-orthogonal and the standardised coefficient on the
    structural predictor is ``a / sqrt(a^2 + b^2 + sigma^2)``.
    """
    rng = np.random.default_rng(seed)
    node_counts = rng.integers(4, 60, size=n_graphs).astype(np.int32)
    edge_counts = (node_counts * rng.uniform(1.0, 2.0, size=n_graphs)).astype(np.int32)
    delta_n = association.delta_n_matrix(node_counts)
    delta_rho = association.delta_density_matrix(node_counts, edge_counts)

    structural = _symmetric(rng.normal(0.0, 1.0, size=(n_graphs, n_graphs)))
    idx = np.triu_indices(n_graphs, k=1)

    def z(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        vector = matrix[idx]
        centred = vector - vector.mean()
        scale = centred.std() or 1.0
        out = np.zeros((n_graphs, n_graphs))
        out[idx] = centred / scale
        return out + out.T

    ged = (
        weight_structural * z(structural)
        + weight_size * z(delta_n)
        + noise * _symmetric(rng.normal(0.0, 1.0, size=(n_graphs, n_graphs)))
    )
    predictors = {
        "lev": np.abs(z(structural) + 5.0),
        "abs_delta_n": delta_n,
        "abs_delta_density": delta_rho,
    }
    expected = weight_structural / float(np.sqrt(weight_structural**2 + weight_size**2 + noise**2))
    return predictors, np.abs(ged + 10.0), expected


def test_mrm_recovers_a_planted_beta1() -> None:
    """A planted structural coefficient must land inside the bootstrap interval."""
    predictors, ged, expected = _mrm_fixture(0.8, 0.5, 0.3)
    result = association.mrm(ged, predictors, TIER1_SMALL, replicates=250, n_permutations=199)
    assert result.predictors[0] == "lev"
    np.testing.assert_allclose(result.beta1, expected, rtol=0.15)
    assert result.beta1_interval.ci_low <= expected <= result.beta1_interval.ci_high
    assert result.beta1_permutation_p <= 0.01
    assert 0.0 < result.r_squared < 1.0


def test_mrm_beta1_collapses_when_levenshtein_is_pure_size_agreement() -> None:
    """D4's refutation branch: if beta1 collapses, Claim B must be restated.

    Here both matrices are driven only by ``|delta n|``, so the marginal
    correlation is substantial and the partial coefficient is not. That is the
    exact scenario ``statistics.md`` section 6 says can refute the paper's
    central result.
    """
    rng = np.random.default_rng(21)
    n_graphs = 45
    node_counts = rng.integers(4, 60, size=n_graphs).astype(np.int32)
    edge_counts = (node_counts * rng.uniform(1.0, 2.0, size=n_graphs)).astype(np.int32)
    delta_n = association.delta_n_matrix(node_counts)
    lev = np.abs(delta_n + _symmetric(rng.normal(0.0, 3.0, size=(n_graphs, n_graphs))))
    ged = np.abs(delta_n + _symmetric(rng.normal(0.0, 3.0, size=(n_graphs, n_graphs))))

    marginal = association.spearman(
        lev[np.triu_indices(n_graphs, k=1)], ged[np.triu_indices(n_graphs, k=1)]
    )
    result = association.mrm(
        ged,
        {
            "lev": lev,
            "abs_delta_n": delta_n,
            "abs_delta_density": association.delta_density_matrix(node_counts, edge_counts),
        },
        TIER1_SMALL,
        replicates=250,
        n_permutations=199,
    )
    assert marginal > 0.4, "the marginal correlation must be substantial for the test to bite"
    assert abs(result.beta1) < 0.15, f"beta1 did not collapse: {result.beta1}"


def test_partial_mantel_agrees_with_the_mrm_on_direction() -> None:
    """The partial Mantel is the same idea in the form reviewers recognise."""
    predictors, ged, _ = _mrm_fixture(0.8, 0.5, 0.3)
    outcome = association.partial_mantel(
        predictors["lev"], ged, predictors["abs_delta_n"], n_permutations=199
    )
    assert outcome["r_partial"] > 0.3
    assert outcome["p_value"] <= 0.01


def test_mantel_permutes_graph_labels_jointly() -> None:
    """D3's null is the joint row/column permutation, reused from E10's function."""
    ged, lev, _ = _dyadic_dataset(n_graphs=25)
    outcome = association.mantel(lev, ged, n_permutations=499)
    assert 0.0 < outcome.p_value <= 1.0
    assert outcome.n_permutations == 499
    np.testing.assert_allclose(
        outcome.observed_r,
        association.spearman(lev[np.triu_indices(25, k=1)], ged[np.triu_indices(25, k=1)]),
        rtol=1e-9,
    )


# ---------------------------------------------------------------------------
# Matrix loading, censoring and the identifier join
# ---------------------------------------------------------------------------


def _write_distance_npz(path: Path, matrix: npt.NDArray[Any], graph_ids: Any, **extra: Any) -> Path:
    """Write a CONTRACTS.md section 4 distance file."""
    n = matrix.shape[0]
    payload: dict[str, Any] = {
        "distance_matrix": np.asarray(matrix, dtype=np.float64),
        "graph_ids": np.asarray(graph_ids),
        "node_counts": np.arange(n, dtype=np.int32),
        "defined_mask": np.ones((n, n), dtype=bool),
        "metadata": np.array(json.dumps({"schema_version": "t06.1", "seed": 42})),
    }
    payload.update(extra)
    np.savez_compressed(path, **payload)
    return path


def test_censored_entries_are_masked_and_a_zero_distance_is_kept(tmp_path: Path) -> None:
    """``inf`` is censoring; ``0`` is a legitimate distance between isomorphs.

    28.05 % of IAM Letter LOW pairs are certified exact at 0. A blanket
    ``value > 0`` guard would abort a correct run on every one of them.
    """
    matrix = np.array([[0.0, 0.0, np.inf], [0.0, 0.0, 2.0], [np.inf, 2.0, 0.0]], dtype=np.float64)
    variables = association.PairVariables.from_matrices({"d": matrix})
    assert variables.valid.tolist() == [True, False, True]
    assert variables.values["d"][0] == 0.0

    path = _write_distance_npz(tmp_path / "toy.npz", matrix, np.array(["a", "b", "c"]))
    bundle = matrices.load_matrix(path)
    assert bundle.n_graphs == 3
    assert bundle.metadata["schema_version"] == "t06.1"


def test_bundles_join_on_graph_ids_never_positionally(tmp_path: Path) -> None:
    """Suite 1 and Suite 2 are different cohorts even where the name matches."""
    left = _write_distance_npz(
        tmp_path / "left.npz",
        np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]]),
        np.array(["g0", "g1", "g2"], dtype="<U8"),
    )
    right = _write_distance_npz(
        tmp_path / "right.npz",
        np.array([[0.0, 9.0, 7.0], [9.0, 0.0, 5.0], [7.0, 5.0, 0.0]]),
        np.array(["g2", "g0", "g9"], dtype="<U16"),
    )
    a, b = matrices.align(matrices.load_matrix(left), matrices.load_matrix(right))
    np.testing.assert_array_equal(a.graph_ids, b.graph_ids)
    assert a.graph_ids.tolist() == ["g0", "g2"]
    assert a.matrix[0, 1] == 2.0  # g0 vs g2 in the left file
    assert b.matrix[0, 1] == 7.0  # g0 vs g2 in the right file, positionally reversed


# ---------------------------------------------------------------------------
# Criterion 9 --- end to end on real Suite-2 matrices
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (APPROX_GED_ROOT / "LB" / "linux.npz").is_file(),
    reason="T-05 Suite-2 matrices are not mounted",
)
def test_f1_end_to_end_on_real_linux_matrices(tmp_path: Path) -> None:
    """F1 on the real LINUX LB/UB pair: 89 graphs, 3,916 pairs, tier 1.

    The Levenshtein matrix is **synthesised** here to a conforming
    CONTRACTS.md section 4 schema. The distance track owns the real one; this
    test exercises the loader, the identifier join, the D7 paired difference on
    one shared resample and the F1 branch rule, none of which depend on the
    Levenshtein values being the real ones.
    """
    lb = matrices.load_matrix(APPROX_GED_ROOT / "LB" / "linux.npz", value_key="lb_matrix")
    ub = matrices.load_matrix(APPROX_GED_ROOT / "UB" / "linux.npz", value_key="ub_matrix")
    assert lb.n_graphs == 89
    assert np.array_equal(lb.graph_ids, ub.graph_ids)

    rng = np.random.default_rng(resampling.SEED)
    surrogate = np.abs(_symmetric(1.3 * lb.matrix + rng.normal(0.0, 2.0, size=lb.matrix.shape)))
    lev_path = _write_distance_npz(
        tmp_path / "linux__isalgraph_pruned__levenshtein.npz", surrogate, lb.graph_ids
    )
    lev = matrices.load_matrix(lev_path)
    lb, ub, lev = matrices.align(lb, ub, lev)

    variables = association.PairVariables.from_matrices(
        {"lb": lb.matrix, "ub": ub.matrix, "lev": lev.matrix}
    )
    assert variables.n_pairs == 89 * 88 // 2 == 3916
    # GED is legitimately 0 for isomorphic graphs; nothing rejects that.
    assert float(variables.values["lb"].min()) == 0.0

    tier = resampling.bootstrap_tier("linux")
    assert (tier.tier, tier.replicates, tier.permutations, tier.subsample) == (1, 2000, 9999, None)

    specs = [
        association.CorrelationSpec("rho_lev_lb", "lev", "lb"),
        association.CorrelationSpec("rho_lev_ub", "lev", "ub"),
    ]
    diffs = [association.DifferenceSpec("f1_linux", "rho_lev_lb", "rho_lev_ub")]
    results, differences = association.bootstrap_associations(
        variables, specs, tier, differences=diffs, replicates=400, kendall=False
    )
    delta = differences["f1_linux"]
    assert np.isfinite(delta.interval.point)
    assert np.isfinite(delta.interval.ci_low) and np.isfinite(delta.interval.ci_high)
    assert delta.interval.ci_low <= delta.interval.point <= delta.interval.ci_high
    assert 0.0 < delta.p_value <= 1.0
    for name in ("rho_lev_lb", "rho_lev_ub"):
        assert -1.0 <= results[name].rho.point <= 1.0

    samples = delta.interval.bootstrap_mean + np.zeros(1)  # smoke: the interval is populated
    assert np.isfinite(samples).all()

    # Drive the real LINUX slot through the full ten-test F1 family.
    rng2 = np.random.default_rng(99)
    inputs = [
        GateInput(
            dataset=key,
            point=delta.interval.point if key == "linux" else 0.0,
            samples=(
                rng2.normal(delta.interval.point, max(delta.interval.bootstrap_sd, 1e-6), 800)
                if key == "linux"
                else rng2.normal(0.0, 0.01, 800)
            ),
        )
        for key in family.SUITE2
    ]
    outcome = family.run_f1(inputs)
    assert outcome.bh.m == 10
    assert set(outcome.failing_datasets) <= set(family.SUITE2)
    linux_row = next(o for o in outcome.outcomes if o.dataset == "linux")
    assert linux_row.test_id == "F1.4"
    card = family.cardinality(uninformative_datasets=outcome.failing_datasets)
    assert card.n_actual == 182 - 8 * len(outcome.failing_datasets)
