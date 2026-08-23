"""E3 (metric axioms) and E4 (the F5-blindness trap) of the T-04a annex.

Every fixture is **synthetic** or comes from ``networkx``'s graph atlas.  The
real cohort lives on an external drive, so a suite that needed it would stop
running exactly when the drive is unmounted -- which is when a regression gets
in.  Nothing here touches the drive.

Four of these tests exist because the experiment's headline number is a
**zero** or a **narrow interval**, and both are values that a broken
implementation produces just as readily as a correct one:

- A triangle-inequality sweep that never fires is indistinguishable from a
  sweep that cannot fire.  :func:`test_triangle_detects_a_real_violation` and
  its siblings feed each checker a matrix that violates exactly one axiom.
- The sweep is only exhaustive if it visits every triple.  The count is
  asserted against ``math.comb`` rather than against a literal copied from
  prose -- **the frozen protocol §4 and the track brief both state
  C(142,3) = 470,660, which is arithmetically wrong; the value is 467,180.**
  See :func:`test_atlas_triple_count_is_exhaustive`.
- A "paired" bootstrap that quietly draws twice is still a bootstrap and still
  prints an interval; it just prints the wrong one, roughly twice as wide.
  :func:`test_paired_interval_is_narrower_than_unpaired` measures the gap on
  correlated arms, and :func:`test_same_index_reaches_both_arms` asserts the
  mechanism rather than the symptom.
- A marginal interval computed here has to be the same statistic as the one
  in the F5 table, or the two tables cannot be read side by side.
  :func:`test_replicate_percentiles_match_graph_bootstrap_ci` pins it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    import numpy.typing as npt

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
nx = pytest.importorskip("networkx")
pytest.importorskip("rapidfuzz")

from isalgraph.competitors.admissibility import common, e3_axioms, e4_trap  # noqa: E402
from isalgraph.competitors.bootstrap import (  # noqa: E402
    MIN_PAIRS,
    ResampleIndex,
    graph_bootstrap_ci,
    make_resample_index,
    spearman,
)

# --------------------------------------------------------------------------
# The universe E3 sweeps
# --------------------------------------------------------------------------

#: OEIS A001349, connected graphs on n nodes, for n = 2..6.
A001349 = {2: 1, 3: 2, 4: 6, 5: 21, 6: 112}


def test_atlas_is_142_connected_graphs_on_at_most_six_nodes() -> None:
    graphs = common.connected_atlas(common.EXHAUSTIVE_N_TRIPLES)
    assert common.EXHAUSTIVE_N_TRIPLES == 6
    assert len(graphs) == 142
    assert all(nx.is_connected(g) for g in graphs)
    assert all(2 <= g.number_of_nodes() <= 6 for g in graphs)

    histogram: dict[int, int] = {}
    for graph in graphs:
        histogram[graph.number_of_nodes()] = histogram.get(graph.number_of_nodes(), 0) + 1
    assert histogram == A001349
    assert sum(A001349.values()) == 142


def test_atlas_triple_count_is_exhaustive() -> None:
    """C(142, 3) = 467,180 -- **not** the 470,660 the protocol prose states.

    The count is derived, never copied: ``142 * 141 * 140 / 6 = 467,180``.
    The identity below is the reason the derived value can be trusted -- the
    per-apex check count and three times the triple count agree, which they
    would not if either expression were wrong.
    """
    n = len(common.connected_atlas(common.EXHAUSTIVE_N_TRIPLES))
    assert n == 142
    assert math.comb(n, 3) == 467_180
    assert math.comb(n, 2) * (n - 2) == 3 * math.comb(n, 3) == 1_401_540


def test_e3_record_reports_the_derived_triple_count(grid_file: Path) -> None:
    payload = e3_axioms.run(str(grid_file), max_n=common.EXHAUSTIVE_N_TRIPLES)
    assert payload["atlas"]["n_graphs"] == 142
    assert payload["atlas"]["by_n"] == {str(k): v for k, v in A001349.items()}
    assert payload["n_triples"] == math.comb(142, 3) == 467_180
    assert payload["n_triangle_checks"] == 1_401_540
    for cell in payload["cells"].values():
        assert cell["triangle"]["n_triples"] == 467_180
        assert cell["triangle"]["n_checks"] == 1_401_540


# --------------------------------------------------------------------------
# E3: the checkers must be able to fail
# --------------------------------------------------------------------------


def test_triangle_detects_a_real_violation() -> None:
    # d(a,c) = 10 but a-b-c costs 1 + 1: one violation, at apex b.
    matrix = np.array([[0.0, 1.0, 10.0], [1.0, 0.0, 1.0], [10.0, 1.0, 0.0]])
    record = e3_axioms.check_triangle(matrix)
    assert record["n_triples"] == 1
    assert record["n_checks"] == 3
    assert record["n_violations"] == 1
    assert record["worst_excess"] == pytest.approx(8.0)
    assert record["rate"] == pytest.approx(1 / 3)
    assert record["rule_of_three_upper"] is None


def test_triangle_passes_a_genuine_metric() -> None:
    matrix = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    record = e3_axioms.check_triangle(matrix)
    assert record["n_violations"] == 0
    assert record["worst_excess"] <= 0.0


def test_symmetry_detects_an_asymmetric_matrix() -> None:
    matrix = np.array([[0.0, 1.0, 2.0], [3.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    record = e3_axioms.check_symmetry(matrix)
    assert record["n_checks"] == 3
    assert record["n_violations"] == 1
    assert record["max_asymmetry"] == pytest.approx(2.0)


def test_identity_detects_a_nonzero_diagonal() -> None:
    matrix = np.array([[0.5, 1.0], [1.0, 0.0]])
    record = e3_axioms.check_identity(matrix)
    assert record["n_checks"] == 2
    assert record["n_violations"] == 1
    assert record["max_abs_self_distance"] == pytest.approx(0.5)


def test_zero_violations_is_never_reported_as_a_bare_zero() -> None:
    """0/N is an upper bound, not a rate.  Protocol D-A4."""
    matrix = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    for record in (
        e3_axioms.check_identity(matrix),
        e3_axioms.check_symmetry(matrix),
        e3_axioms.check_triangle(matrix),
    ):
        assert record["n_violations"] == 0
        assert record["rate"] is None
        assert record["rule_of_three_upper"] == pytest.approx(
            common.rule_of_three(record["n_checks"])
        )
        assert record["clopper_pearson_95"][0] == 0.0


def test_distance_matrix_computes_both_triangles() -> None:
    """Symmetry must be measured, not mirrored.

    A matrix filled by mirroring would make :func:`check_symmetry` a test of
    ``numpy``.  The metric is called ``n^2`` times, so a genuinely asymmetric
    metric would surface.
    """
    calls: list[tuple[int, int]] = []

    class Counting:
        name = "counting"
        is_pseudometric = False

        def is_defined(self, a: object, b: object) -> bool:
            return True

        def distance(self, a: object, b: object) -> float:
            calls.append((int(a), int(b)))  # type: ignore[arg-type]
            return float(abs(int(a) - int(b)))  # type: ignore[arg-type]

    from isalgraph.competitors import registry

    registry.register_metric("counting", Counting)
    try:
        matrix, undefined = e3_axioms.distance_matrix("counting", [0, 1, 2])
    finally:
        registry._METRICS.pop("counting", None)

    assert undefined == 0
    assert len(calls) == 9
    assert (0, 2) in calls and (2, 0) in calls
    assert matrix[0, 2] == matrix[2, 0] == 2.0


def test_e3_end_to_end_on_a_small_atlas(grid_file: Path) -> None:
    """Every cell, over the nine connected graphs on n <= 4.  Zero violations."""
    payload = e3_axioms.run(str(grid_file), max_n=4)
    assert payload["atlas"]["n_graphs"] == 9
    assert set(payload["cells"]) == {
        "agm_cam/levenshtein",
        "isalgraph_canonical/levenshtein",
        "isalgraph_pruned/levenshtein",
        "min_dfs/levenshtein",
        "nauty_graph6/levenshtein",
        "sparse6_nauty/levenshtein",
        "wl_subtree/kernel",
    }
    assert set(payload["excluded"]) == {"adjacency", "graph6", "size_null", "sparse6"}
    assert e3_axioms.violating_cells(payload) == []
    assert payload["cells"]["wl_subtree/kernel"]["declared_pseudometric"] is True
    assert payload["cells"]["isalgraph_pruned/levenshtein"]["declared_pseudometric"] is False


def test_violating_cells_names_the_failure() -> None:
    payload = {
        "cells": {
            "a/levenshtein": {
                "identity": {"n_violations": 0},
                "symmetry": {"n_violations": 0},
                "triangle": {"n_violations": 7},
            }
        }
    }
    assert e3_axioms.violating_cells(payload) == ["a/levenshtein/triangle"]


# --------------------------------------------------------------------------
# E4: the bootstrap must actually be paired
# --------------------------------------------------------------------------


def _correlated_arms(
    n_graphs: int = 60, *, seed: int = 7
) -> tuple[list[float], list[float], list[float], list[tuple[int, int]]]:
    """Two representations whose rhos move together, plus a shared pair set.

    Arm B is arm A plus a small independent perturbation, so the two rhos are
    strongly positively correlated across resamples.  That is the regime in
    which pairing buys the most, and it is the regime the real comparison sits
    in: both arms score the *same* graphs against the *same* GED.
    """
    rng = np.random.default_rng(seed)
    pairs = [(a, b) for a in range(n_graphs) for b in range(a + 1, n_graphs)]
    ged = rng.integers(0, 20, size=len(pairs)).astype(float)
    arm_a = ged + rng.normal(0.0, 4.0, size=len(pairs))
    arm_b = arm_a + rng.normal(0.0, 1.0, size=len(pairs))
    return list(arm_a), list(arm_b), list(ged), pairs


def test_replicate_rhos_is_deterministic_in_the_index() -> None:
    arm_a, _, ged, pairs = _correlated_arms()
    index = make_resample_index(60, resamples=200, seed=42)
    first = e4_trap.replicate_rhos(arm_a, ged, pairs, index)
    second = e4_trap.replicate_rhos(arm_a, ged, pairs, index)
    np.testing.assert_array_equal(first, second)


def test_replicate_percentiles_match_graph_bootstrap_ci() -> None:
    """The marginal interval here is the F5 table's interval, exactly."""
    arm_a, _, ged, pairs = _correlated_arms()
    index = make_resample_index(60, resamples=300, seed=42)
    mine = e4_trap.percentile_ci(e4_trap.replicate_rhos(arm_a, ged, pairs, index))
    theirs = graph_bootstrap_ci(arm_a, ged, pairs, index)
    assert mine is not None and theirs is not None
    assert mine[0] == pytest.approx(theirs[0])
    assert mine[1] == pytest.approx(theirs[1])


def test_same_index_reaches_both_arms() -> None:
    """The mechanism, not the symptom: one index object, two arms.

    ``paired_comparison`` is handed a single :class:`ResampleIndex` and must
    pass that object -- not a fresh draw -- into both arms' replicate loops.
    The spy records every index it is called with.
    """
    arm_a, arm_b, ged, pairs = _correlated_arms()
    index = make_resample_index(60, resamples=120, seed=42)
    positions = list(pairs)
    series = {
        "challenger": (arm_a, ged, positions, pairs),
        "reference": (arm_b, ged, positions, pairs),
    }

    seen: list[int] = []
    original = e4_trap.replicate_rhos

    def spy(
        x: list[float],
        y: list[float],
        pair_index: list[tuple[int, int]],
        idx: ResampleIndex,
        *,
        min_pairs: int = MIN_PAIRS,
    ) -> npt.NDArray[Any]:
        seen.append(id(idx))
        return original(x, y, pair_index, idx, min_pairs=min_pairs)

    e4_trap.replicate_rhos = spy  # type: ignore[assignment]
    try:
        record = e4_trap.paired_comparison(
            "challenger", "reference", series, index, unpaired_seed=43
        )
    finally:
        e4_trap.replicate_rhos = original  # type: ignore[assignment]

    # Three calls: both paired arms on `index`, plus the reference arm again
    # on the deliberately independent index behind `ci_unpaired`.
    assert len(seen) == 3
    assert seen[:2] == [id(index), id(index)]
    assert seen[2] != id(index)
    assert record["n_pairs_common"] == len(pairs)


def test_paired_interval_is_narrower_than_unpaired() -> None:
    """Pairing removes the shared draw noise; unpaired adds a second variance."""
    arm_a, arm_b, ged, pairs = _correlated_arms()
    index = make_resample_index(60, resamples=800, seed=42)
    positions = list(pairs)
    series = {
        "challenger": (arm_a, ged, positions, pairs),
        "reference": (arm_b, ged, positions, pairs),
    }
    record = e4_trap.paired_comparison("challenger", "reference", series, index, unpaired_seed=43)

    paired = record["ci"]
    unpaired = record["ci_unpaired"]
    assert paired is not None and unpaired is not None
    assert (paired[1] - paired[0]) < (unpaired[1] - unpaired[0])
    # The arms here are nearly identical, so the gap is not marginal.
    assert (unpaired[1] - unpaired[0]) > 3.0 * (paired[1] - paired[0])


def test_paired_difference_point_estimate_is_the_rho_gap() -> None:
    arm_a, arm_b, ged, pairs = _correlated_arms()
    index = make_resample_index(60, resamples=100, seed=42)
    positions = list(pairs)
    series = {
        "challenger": (arm_a, ged, positions, pairs),
        "reference": (arm_b, ged, positions, pairs),
    }
    record = e4_trap.paired_comparison("challenger", "reference", series, index, unpaired_seed=43)
    rho_a, _ = spearman(arm_a, ged)
    rho_b, _ = spearman(arm_b, ged)
    assert record["difference"] == pytest.approx(rho_a - rho_b)
    assert record["rho_challenger"] == pytest.approx(rho_a)
    assert record["rho_reference"] == pytest.approx(rho_b)


def test_unpaired_seed_must_differ_from_the_shared_index() -> None:
    arm_a, arm_b, ged, pairs = _correlated_arms(20)
    index = make_resample_index(20, resamples=50, seed=42)
    positions = list(pairs)
    series = {
        "challenger": (arm_a, ged, positions, pairs),
        "reference": (arm_b, ged, positions, pairs),
    }
    with pytest.raises(ValueError, match="different from the shared"):
        e4_trap.paired_comparison("challenger", "reference", series, index, unpaired_seed=42)


def test_paired_comparison_restricts_to_the_common_pair_set() -> None:
    """A pair only one arm can score is dropped, not silently misaligned."""
    arm_a, arm_b, ged, pairs = _correlated_arms(30)
    index = make_resample_index(30, resamples=50, seed=42)
    keep = list(range(len(pairs) - 5))
    series = {
        "challenger": (arm_a, ged, list(pairs), list(pairs)),
        "reference": (
            [arm_b[k] for k in keep],
            [ged[k] for k in keep],
            [pairs[k] for k in keep],
            [pairs[k] for k in keep],
        ),
    }
    record = e4_trap.paired_comparison("challenger", "reference", series, index, unpaired_seed=43)
    assert record["n_pairs_common"] == len(pairs) - 5
    assert record["n_pairs_challenger"] == len(pairs)
    assert record["n_pairs_reference"] == len(pairs) - 5
    assert record["difference"] is not None


def test_bootstrap_p_is_never_exactly_zero() -> None:
    """2,000 replicates resolve to 1/2001, and the code says so."""
    all_positive = np.full(2000, 0.3)
    p = e4_trap.bootstrap_p(all_positive)
    assert p is not None
    assert p == pytest.approx(2.0 / 2001.0)
    assert p > 0.0

    balanced = np.concatenate([np.full(1000, -0.1), np.full(1000, 0.1)])
    assert e4_trap.bootstrap_p(balanced) == pytest.approx(1.0)
    assert e4_trap.bootstrap_p(np.array([np.nan])) is None


def test_percentile_ci_declines_to_invent_an_interval() -> None:
    assert e4_trap.percentile_ci(np.array([np.nan, np.nan])) is None
    assert e4_trap.percentile_ci(np.array([0.5, np.nan])) is None
    ci = e4_trap.percentile_ci(np.array([0.0, 1.0]))
    assert ci is not None and ci[0] < ci[1]


def test_replicate_rhos_rejects_ragged_inputs() -> None:
    index = make_resample_index(5, resamples=10, seed=42)
    with pytest.raises(ValueError, match="agree in length"):
        e4_trap.replicate_rhos([1.0, 2.0], [1.0], [(0, 1), (1, 2)], index)


# --------------------------------------------------------------------------
# E4: the psi join and the joint reading
# --------------------------------------------------------------------------


def test_load_psi_reads_protocol_section_two_shape(tmp_path: Path) -> None:
    """The shape E1 is specified to emit: per dataset, per representation."""
    payload = {
        "experiment": "E1",
        "results": {
            "iam_letter_low": {
                "adjacency": {
                    "invariance_rate": 0.0,
                    "separation_ratio": {"point": 0.97, "ci": [0.94, 0.99]},
                },
                "isalgraph_pruned": {"invariance_rate": 1.0, "separation_ratio": 0.0},
            },
            "linux": {"adjacency": {"psi": 0.88}},
        },
        "pooled": {"wl_subtree": {"psi": 0.0}},
    }
    path = tmp_path / "e1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    table = e4_trap.load_psi(str(path))
    assert table[("iam_letter_low", "adjacency")] == pytest.approx(0.97)
    assert table[("iam_letter_low", "isalgraph_pruned")] == pytest.approx(0.0)
    assert table[("linux", "adjacency")] == pytest.approx(0.88)
    assert table[("pooled", "wl_subtree")] == pytest.approx(0.0)

    assert e4_trap.psi_for(table, "linux", "adjacency") == (pytest.approx(0.88), "linux")
    # A pooled value is a fallback and is labelled as one, never printed as
    # though it were per-dataset.
    value, scope = e4_trap.psi_for(table, "aids", "wl_subtree")
    assert value == pytest.approx(0.0)
    assert scope == "pooled"
    assert e4_trap.psi_for(table, "aids", "min_dfs") == (None, "absent")


def test_load_psi_rejects_unreadable_input(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(common.AdmissibilityError, match="cannot read E1"):
        e4_trap.load_psi(str(path))


def test_reading_labels_the_trap() -> None:
    assert "TRAP" in e4_trap.reading(0.85, 0.9)
    assert e4_trap.reading(0.85, 0.0) == "a good, well-defined graph distance"
    assert e4_trap.reading(0.2, 0.0) == "well-defined and weak"
    assert e4_trap.reading(0.2, 0.9) == "neither well-defined nor strong"
    assert e4_trap.reading(None, 0.0) == "rho undefined"
    assert "psi absent" in e4_trap.reading(0.9, None)


def test_thresholds_are_declared_not_derived() -> None:
    """The cut points exist in source, so no cut point is chosen after a number."""
    assert e4_trap.PSI_INVARIANT_MAX == 0.05
    assert e4_trap.RHO_HIGH_MIN == 0.50


# --------------------------------------------------------------------------
# E4: the representation set is the point of the experiment
# --------------------------------------------------------------------------


def test_e4_measures_the_three_representations_the_grid_excluded(grid_file: Path) -> None:
    under, admitted = e4_trap.representations(str(grid_file))
    assert set(e4_trap.EXCLUDED_UNDER) == {"adjacency", "graph6", "sparse6"}
    for name in e4_trap.EXCLUDED_UNDER:
        assert under[name] == "levenshtein"
        assert admitted[name] is False
    assert under["isalgraph_pruned"] == "levenshtein"
    assert under["wl_subtree"] == "kernel"
    assert admitted["isalgraph_pruned"] is True
    # The descriptive baseline is not a representation and F5 already prints it.
    assert "size_null" not in under
    assert e4_trap.REFERENCE_ARM == "isalgraph_pruned"


def test_e4_refuses_suite_two(grid_file: Path) -> None:
    """There is no certified exact GED above n = 12, so there is no rho."""
    with pytest.raises(common.AdmissibilityError, match="Suite 1 only"):
        e4_trap.run(str(grid_file), names=("grec",))


def test_holm_is_applied_across_the_five_datasets() -> None:
    per_dataset = {
        "a": {"k::all_pairs": {"p": 0.01}},
        "b": {"k::all_pairs": {"p": 0.04}},
        "c": {"k::all_pairs": {"p": 0.20}},
        "d": {"k::all_pairs": {"p": None, "difference": None}},
    }
    out = e4_trap._holm_across_datasets(per_dataset, ("a", "b", "c", "d"))
    block = out["k::all_pairs"]
    assert block["n_tested"] == 3
    expected = common.holm([0.01, 0.04, 0.20])
    assert block["datasets"]["a"]["p_holm"] == pytest.approx(expected[0])
    assert block["datasets"]["b"]["p_holm"] == pytest.approx(expected[1])
    assert block["datasets"]["c"]["p_holm"] == pytest.approx(expected[2])
    assert block["datasets"]["d"]["p_holm"] is None


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def grid_file(tmp_path: Path) -> Path:
    """A grid payload carrying T-04a's frozen selection, verbatim.

    Written out rather than read from the drive: the selection is what the
    tests depend on, and it is frozen, so a copy is a fixture rather than a
    duplicate of a moving target.
    """
    payload = {
        "protocol": "T-04a",
        "primary_distance": {
            "adjacency": None,
            "agm_cam": "levenshtein",
            "graph6": None,
            "isalgraph_canonical": "levenshtein",
            "isalgraph_pruned": "levenshtein",
            "min_dfs": "levenshtein",
            "nauty_graph6": "levenshtein",
            "size_null": None,
            "sparse6": None,
            "sparse6_nauty": "levenshtein",
            "wl_subtree": "kernel",
        },
    }
    path = tmp_path / "grid_200.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
