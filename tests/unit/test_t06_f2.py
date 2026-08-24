"""Tests for the F2 driver.

Two of these are anti-regression guards rather than ordinary unit tests, and
they are here because the failures they catch have each already happened once in
this ticket.

``test_n_actual_is_79`` pins the BH denominator. Six separate reductions have
been proposed that would shrink ``N_actual``, and every one of them was in the
anti-conservative direction: a smaller denominator lowers the burden on every
surviving test. The code was never wrong, which is exactly why the guard belongs
in the test suite rather than in a document.

``test_the_reference_arm_is_never_charged_to_c`` pins the one exemption that a
plausible-looking simplification would delete. ``isalgraph_pruned`` sits below
the 99 % completion threshold on Mutagenicity because 101 graphs censored; D14
retains those graphs with their greedy-min string, so they *did* produce an
encoding, and charging them to ``c`` would remove F2 cells on evidence we hold.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from benchmarks.real_data.eval_stats import t06_f2, t06_f2_inputs
from benchmarks.real_data.eval_stats.family import Cell, ReductionInputs, cardinality

#: The frozen terms of design note 18.7, which F2 ran under.
FROZEN_C: frozenset[tuple[str, str, str]] = frozenset(
    {
        ("suite2", "aids_graphedx", "agm_cam"),
        ("suite2", "aids_iam", "agm_cam"),
        ("suite2", "coil_del", "agm_cam"),
        ("suite2", "grec", "agm_cam"),
        ("suite2", "mutagenicity", "agm_cam"),
        ("suite2", "mutagenicity", "min_dfs"),
        ("suite2", "protein", "agm_cam"),
    }
)


def _loaded_benchmark_modules() -> list[ModuleType]:
    """Return every loaded ``benchmarks.*`` module."""
    return [
        module
        for name, module in list(sys.modules.items())
        if name.startswith("benchmarks.") and module is not None
    ]


# ---------------------------------------------------------------------------
# A8 --- the resampling unit is the graph
# ---------------------------------------------------------------------------


def test_the_pair_level_bootstrap_is_not_reachable_from_the_f2_driver() -> None:
    """A8. ``bootstrap_correlation`` resamples pairs and must not be reachable.

    Checked by **object identity** over the loaded closure, not by grepping: an
    alias, a re-export or a ``getattr`` would each defeat a text search and none
    of them defeats this. Importing the driver at module scope is what puts it
    into ``sys.modules`` for the sweep.
    """
    from benchmarks.real_data.eval_correlation import correlation_metrics

    forbidden = correlation_metrics.bootstrap_correlation
    for module in (t06_f2, t06_f2_inputs):
        offenders = [name for name, obj in vars(module).items() if obj is forbidden]
        assert not offenders, f"{module.__name__} binds the pair-level bootstrap as {offenders}"

    assert any(m.__name__.endswith("t06_f2") for m in _loaded_benchmark_modules())


def test_the_driver_declares_the_graph_as_its_resampling_unit() -> None:
    """Every emitted tier block must say the unit was the graph, not the pair."""
    from benchmarks.real_data.eval_stats.resampling import bootstrap_tier

    assert bootstrap_tier("protein", "suite2").as_dict()["resampling_unit"] == "graph"


# ---------------------------------------------------------------------------
# The BH denominator
# ---------------------------------------------------------------------------


def test_n_actual_is_79_under_the_frozen_branch() -> None:
    """``N_actual`` is 79, by enumeration, with the closed form agreeing.

    Design note 18.7: F0's majority branch fired, so the 81 approximate-regime
    cells are descriptive, ``d`` is not applied, and ``k`` removes only its 5
    B1e cells per representation. ``101 - 5*3 - 7 = 79``.
    """
    card = cardinality(
        excluded_representations=t06_f2.EXCLUDED_REPRESENTATIONS,
        uninformative_datasets=(),
        noncomputable=FROZEN_C,
        f0_demotes_approximate=True,
    )
    assert card.n_actual == 79
    assert card.closed_form == 79
    assert card.discrepancy == 0
    assert (card.k, card.d, card.c) == (3, 0, 7)
    assert len(card.removed_by_f0) == 81


def test_the_admissible_rows_are_the_exact_regime_and_claim_a() -> None:
    """Only A1, A2, B1e and B3e survive; every approximate row is demoted."""
    card = cardinality(
        excluded_representations=t06_f2.EXCLUDED_REPRESENTATIONS,
        noncomputable=FROZEN_C,
        f0_demotes_approximate=True,
    )
    counts: dict[str, int] = {}
    for cell in card.admissible:
        counts[cell.row] = counts.get(cell.row, 0) + 1
    assert counts == {"A1": 53, "A2": 1, "B1e": 20, "B3e": 5}
    assert not {cell.row for cell in card.admissible} & {"B1a", "B2", "B3a"}


def test_d_may_not_be_applied_once_f0_has_demoted() -> None:
    """Applying ``d`` after the demotion would charge the same cells twice."""
    with pytest.raises(t06_f2_inputs.F2InputError, match="not applied"):
        t06_f2_inputs.build_reduction_inputs(
            _completion_file(),
            excluded_representations=frozenset(),
            f0_demotes_approximate=True,
            uninformative_datasets=frozenset({"grec"}),
        )


# ---------------------------------------------------------------------------
# ``c`` and the reference-arm exemption
# ---------------------------------------------------------------------------


def _completion_file() -> Path:
    """Return a completion-rates path fixture written on demand."""
    import json
    import tempfile

    rows = [
        {
            "suite": "suite2",
            "dataset": "mutagenicity",
            "representation": "isalgraph_pruned",
            "rate": 0.975,
        },
        {"suite": "suite2", "dataset": "mutagenicity", "representation": "min_dfs", "rate": 0.9478},
        {"suite": "suite2", "dataset": "protein", "representation": "agm_cam", "rate": 0.0615},
        {"suite": "suite2", "dataset": "grec", "representation": "nauty_graph6", "rate": 1.0},
    ]
    path = Path(tempfile.mkdtemp()) / "completion_rates.json"
    path.write_text(json.dumps({"rows": rows}))
    return path


def test_the_reference_arm_is_never_charged_to_c() -> None:
    """``preregistration`` 5.1 consequence 2: D14 governs the reference arm.

    It sits below the threshold on Mutagenicity only because 101 graphs
    censored, and a censored graph is retained with its greedy-min string --- so
    it did produce an encoding, which is what the criterion asks for.
    """
    triples, failing = t06_f2_inputs.noncomputable_triples(_completion_file())
    assert ("suite2", "mutagenicity", "isalgraph_pruned") not in triples
    assert triples == {
        ("suite2", "mutagenicity", "min_dfs"),
        ("suite2", "protein", "agm_cam"),
    }
    # The exempted row is still reported, so the exemption is visible.
    assert any(row["representation"] == "isalgraph_pruned" for row in failing)


# ---------------------------------------------------------------------------
# A1 --- the intersection-union test
# ---------------------------------------------------------------------------


class _Enc:
    """Minimal stand-in for :class:`t06_f2_inputs.ArmEncodings`."""

    def __init__(self, ids: list[str], entropy: list[float], realised: list[float]) -> None:
        self.graph_ids = np.array(ids)
        self.bits = {
            "entropy_bits": np.array(entropy, dtype=float),
            "realised_bits": np.array(realised, dtype=float),
        }

    def usable(self, arm: str) -> np.ndarray:
        """Every graph is usable in this fixture."""
        del arm
        return np.ones(self.graph_ids.size, dtype=bool)


def test_the_iut_takes_the_larger_marginal_p() -> None:
    """Design note 18.8: ``p = max``, so no primary convention is named.

    The fixture makes the two conventions disagree in strength on purpose: the
    entropy column separates cleanly and the realised column does not. An
    implementation that quietly picked the entropy convention would return the
    small p; the IUT must return the large one.
    """
    ids = [f"g{i}" for i in range(12)]
    reference = _Enc(ids, [10.0] * 12, [10.0] * 12)
    competitor = _Enc(ids, [20.0] * 12, [10.0, 11.0] * 6)
    record = t06_f2._claim_a_cell(reference, competitor, "grec", "graph6", "primary")

    assert record is not None
    assert record.iut_p == max(record.marginal_p.values())
    assert record.iut_p >= record.marginal_p["entropy_bits"]
    assert set(record.marginal_p) == {"entropy_bits", "realised_bits"}


def test_the_iut_flags_a_direction_disagreement_rather_than_absorbing_it() -> None:
    """Where the conventions disagree in sign, the cell says so."""
    ids = [f"g{i}" for i in range(10)]
    reference = _Enc(ids, [10.0] * 10, [30.0] * 10)
    competitor = _Enc(ids, [20.0] * 10, [10.0] * 10)
    record = t06_f2._claim_a_cell(reference, competitor, "grec", "graph6", "primary")

    assert record is not None
    assert record.discordant
    assert record.median_difference["entropy_bits"] > 0
    assert record.median_difference["realised_bits"] < 0


# ---------------------------------------------------------------------------
# Views and mask grouping
# ---------------------------------------------------------------------------


def test_equal_n_mask_selects_exactly_the_same_size_pairs() -> None:
    """``equal_n`` is derived, never materialised as an artifact."""
    mask = t06_f2.equal_n_mask(np.array([3, 3, 5, 7, 5]))
    assert mask.shape == (5, 5)
    assert mask[0, 1] and mask[2, 4]
    assert not mask[0, 2] and not mask[3, 4]
    assert mask.diagonal().all()


def test_an_unknown_view_is_an_error_not_a_silent_all_pairs() -> None:
    """A typo in a view name must fail loudly rather than widen the pair set."""
    with pytest.raises(t06_f2.F2DriverError, match="unknown view"):
        t06_f2._view_mask("equal-n", np.array([1, 2, 3]))


def _arm(name: str, defined: np.ndarray) -> t06_f2_inputs.ArmMatrices:
    """Build a comparator arm carrying *defined*."""
    n = defined.shape[0]
    return t06_f2_inputs.ArmMatrices(
        representation=name,
        metric="levenshtein",
        distance=np.zeros((n, n)),
        defined=defined,
        size_null=np.zeros((n, n)),
        graph_ids=np.array([f"g{i}" for i in range(n)]),
        node_counts=np.ones(n, dtype=np.int64),
    )


def test_arms_sharing_a_defined_mask_share_one_resample() -> None:
    """Equal masks give an equal valid pair set, so one bootstrap serves them.

    Arms with different coverage must NOT be pooled: ``agm_cam`` completes on
    6 % of Protein, and pooling it would drag every other arm down to its pairs.
    """
    full = np.ones((4, 4), dtype=bool)
    sparse = np.zeros((4, 4), dtype=bool)
    sparse[:2, :2] = True
    groups = t06_f2._group_by_mask([_arm("a", full), _arm("b", full.copy()), _arm("c", sparse)])

    assert len(groups) == 2
    sizes = sorted(len(group.comparators) for group in groups)
    assert sizes == [1, 2]
    lonely = next(g for g in groups if len(g.comparators) == 1)
    assert lonely.comparators[0].representation == "c"


# ---------------------------------------------------------------------------
# Loading contracts
# ---------------------------------------------------------------------------


def test_an_ambiguous_primary_distance_is_an_error(tmp_path: Path) -> None:
    """The primary distance is read off the tree and must be unambiguous."""
    root = tmp_path / "suite1"
    root.mkdir()
    for metric in ("levenshtein", "kernel"):
        (root / f"grec__agm_cam__{metric}.npz").write_bytes(b"")
    (root / "grec__agm_cam__size_null.npz").write_bytes(b"")

    with pytest.raises(t06_f2_inputs.F2InputError, match="unambiguous"):
        t06_f2_inputs.primary_metric(tmp_path, "suite1", "grec", "agm_cam")


def test_a_missing_cell_is_none_rather_than_an_error(tmp_path: Path) -> None:
    """A representation with no distance on a dataset is absent, not broken."""
    (tmp_path / "suite1").mkdir()
    assert t06_f2_inputs.primary_metric(tmp_path, "suite1", "grec", "agm_cam") is None


def test_the_p_value_map_never_names_an_inadmissible_cell() -> None:
    """A p-value on a demoted cell would smuggle it back into the family."""
    inputs, _ = t06_f2_inputs.build_reduction_inputs(
        _completion_file(),
        excluded_representations=t06_f2.EXCLUDED_REPRESENTATIONS,
        f0_demotes_approximate=True,
    )
    rho_rows = [
        {
            "row": "B1a",
            "view": "all_pairs",
            "p_value": 0.01,
            "dataset": "grec",
            "representation": "min_dfs",
        }
    ]
    values = t06_f2.assemble_p_values([], rho_rows, {}, inputs, view="all_pairs")
    assert Cell("B1a", "suite2", "grec", "min_dfs") not in values
    assert not values


# ---------------------------------------------------------------------------
# One arm record per cell
# ---------------------------------------------------------------------------


def _row(representation: str, n_pairs: int, rho: float) -> dict[str, object]:
    """Build a minimal rho row for the dedup tests."""
    return {
        "suite": "suite2",
        "dataset": "protein",
        "representation": representation,
        "reference": "lb",
        "view": "all_pairs",
        "n_pairs": n_pairs,
        "rho": {"point": rho},
    }


def test_dedup_keeps_the_widest_pair_set_not_the_first() -> None:
    """The arm's own pairs win, whatever order the groups were emitted in.

    Partials written before the arm was emitted once per cell carry one arm
    record per mask group. On Protein the ``agm_cam`` group holds 595 pairs
    against 161,596, so "keep the first" makes the surviving value depend on
    emission order --- and a 595-pair record can stand in for the headline
    number with nothing failing.
    """
    narrow = _row("isalgraph_pruned", 595, 0.5531)
    wide = _row("isalgraph_pruned", 161_596, 0.7321)

    for ordering in ([narrow, wide], [wide, narrow]):
        kept = t06_f2.dedup_rho_rows(ordering)
        assert len(kept) == 1
        assert kept[0]["n_pairs"] == 161_596
        assert kept[0]["rho"]["point"] == pytest.approx(0.7321)


def test_dedup_is_a_no_op_on_well_formed_rows() -> None:
    """Distinct cells all survive; the rule only collapses true duplicates."""
    rows = [_row("isalgraph_pruned", 100, 0.5), _row("wl_subtree", 100, 0.6)]
    assert len(t06_f2.dedup_rho_rows(rows)) == 2


def test_only_one_group_may_emit_the_arm_record() -> None:
    """``emit_arm`` defaults to False so a group cannot claim the arm by accident."""
    import inspect

    signature = inspect.signature(t06_f2.run_correlation_group)
    assert signature.parameters["emit_arm"].default is False


# ---------------------------------------------------------------------------
# The BH denominator survives an incomplete run
# ---------------------------------------------------------------------------


def test_bh_denominator_is_the_admissible_count_not_the_computed_count() -> None:
    """Uncomputed is not inadmissible.

    A campaign can be cut short, and the tempting "fix" is to run BH over the
    cells that actually carry a p-value. That shrinks the denominator, weakens
    the correction on every surviving test, and is the anti-conservative
    direction --- the same pattern six earlier reductions took, and the only one
    that would be a deliberate choice rather than an inherited defect.

    The property is asserted here rather than left as a rule to remember,
    because a rule someone has to remember is one someone can help past.
    """
    from benchmarks.real_data.eval_stats.family import run_f2

    inputs = ReductionInputs(
        excluded_representations=t06_f2.EXCLUDED_REPRESENTATIONS,
        noncomputable=FROZEN_C,
        f0_demotes_approximate=True,
    )
    card = cardinality(
        excluded_representations=t06_f2.EXCLUDED_REPRESENTATIONS,
        noncomputable=FROZEN_C,
        f0_demotes_approximate=True,
    )
    # Only three of the seventy-nine cells carry a value, as if the campaign
    # had been stopped early.
    partial = {cell: 0.01 for cell in card.admissible[:3]}
    result = run_f2(partial, inputs)

    assert result.bh_primary.m == 79
    assert result.cardinality.n_actual == 79
    assert len(result.cells) == 3
    assert result.bh_sensitivity.m == 182


# ---------------------------------------------------------------------------
# Completion is counted from artifacts, never from progress signals
# ---------------------------------------------------------------------------


def test_a_log_line_is_never_counted_as_a_landed_cell(tmp_path: Path) -> None:
    """A finished ``MRM@`` line in a log does not make a cell complete.

    Three times in one session an observer counted a progress signal instead of
    an artifact and reported a plausible, wrong number. The handoff's rule
    already covered it --- *confirm a run against its output file, never against
    a process list* --- so what was missing was not the rule but a mechanism.
    """
    partials = tmp_path / "f2_partials"
    partials.mkdir()
    (partials / "suite1__linux.json").write_text("{}")

    logs = tmp_path / "logs"
    logs.mkdir()
    # A shard whose MRM has finished but whose partial has not been written.
    (logs / "f2_suite2__mutagenicity.log").write_text(
        "INFO suite2/mutagenicity all_pairs group[abc123] 3 arms in 10.0 s\n"
        "INFO suite2/mutagenicity MRM@lb beta1=+0.1000 p=0.00010 in 100.0 s\n"
    )

    report = t06_f2.campaign_status([partials], logs)

    assert report["n_landed"] == 1
    assert report["landed"] == ["suite1__linux"]
    assert "suite2__mutagenicity" in report["in_flight"]
    assert "suite2__mutagenicity" not in report["landed"]
    assert "1 MRM(s) reported, NOT landed" in report["in_flight"]["suite2__mutagenicity"]


def test_status_never_double_counts_a_landed_cell_that_also_has_a_log(
    tmp_path: Path,
) -> None:
    """A cell with both a partial and a log counts once, as landed."""
    partials = tmp_path / "f2_partials"
    partials.mkdir()
    (partials / "suite2__grec.json").write_text("{}")
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "f2_suite2__grec.log").write_text("INFO suite2/grec MRM@lb beta1=+0.2 p=0.0001\n")

    report = t06_f2.campaign_status([partials], logs)

    assert report["n_landed"] == 1
    assert report["in_flight"] == {}


def test_status_reports_unknown_rather_than_not_started_without_logs(
    tmp_path: Path,
) -> None:
    """Without a readable log root the in-flight state is UNKNOWN, not 'not started'.

    This is the over-accepting-consumer defect reappearing inside the tool
    written to prevent it: an earlier version reported every unlanded cell as
    ``not started`` when no log root was given. That is a claim the function
    cannot support, and a dangerous one --- "not started" on a shard 90 %
    through its MRM reads as a dead shard, and the correct response to a dead
    shard is to relaunch it.
    """
    partials = tmp_path / "f2_partials"
    partials.mkdir()
    (partials / "suite1__linux.json").write_text("{}")

    blind = t06_f2.campaign_status([partials], None)
    assert blind["in_flight_state_known"] is False
    assert blind["not_started"] == []
    assert "suite2__mutagenicity" in blind["unknown_state"]

    empty_logs = tmp_path / "logs"
    empty_logs.mkdir()
    still_blind = t06_f2.campaign_status([partials], empty_logs)
    assert still_blind["in_flight_state_known"] is False
    assert still_blind["not_started"] == []


def test_status_claims_not_started_only_when_it_can_see_logs(tmp_path: Path) -> None:
    """With logs readable, a cell absent from them really has not started."""
    partials = tmp_path / "f2_partials"
    partials.mkdir()
    (partials / "suite1__linux.json").write_text("{}")
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "f2_suite2__mutagenicity.log").write_text("INFO group[abc] 1 arms in 1.0 s\n")

    report = t06_f2.campaign_status([partials], logs)

    assert report["in_flight_state_known"] is True
    assert report["unknown_state"] == []
    assert "suite2__mutagenicity" in report["in_flight"]
    assert "suite2__grec" in report["not_started"]


# ---------------------------------------------------------------------------
# beta1 cannot be rendered without its size coefficient
# ---------------------------------------------------------------------------


def test_beta1_cannot_render_without_beta_size() -> None:
    """The rule 'beta1 never travels without beta_size' is a mechanism, not a habit.

    It was stated twice, after two separate corrections, and both times held only
    because someone remembered. A significant beta1 that is one-fifth the size of
    the confound it competes with is a real finding AND a modest one; a reader
    shown only the first half draws the wrong conclusion, which is what happened
    when a log-sourced beta1 set briefly read as a clean win.
    """
    from benchmarks.real_data.eval_stats import t06_decision_summary as summary

    measured = summary.render_beta1(
        {"beta_levenshtein": 0.2494, "beta_delta_n": 0.8161}
    )
    assert "+0.2494" in measured
    assert "β_size" in measured and "+0.8161" in measured

    # A log line carries beta1 and no size coefficient.
    unmeasured = summary.render_beta1(
        {"beta_levenshtein": 0.0976, "beta_delta_n": float("nan")}
    )
    assert "+0.0976" in unmeasured
    assert "UNMEASURED" in unmeasured
    # The bare number must never be the whole cell.
    assert unmeasured.strip() != "+0.0976"


def test_log_parsing_accepts_both_beta1_formats(tmp_path: Path) -> None:
    """The producer now emits beta_size; shards launched before it did not.

    A guard installed at the producer only helps output produced after it. The
    consumer therefore has to read both, and the older bare-beta1 line must come
    back as UNMEASURED rather than fail to parse -- a log that silently stops
    matching loses the fit entirely, which is worse than showing it qualified.
    """
    from benchmarks.real_data.eval_stats import t06_decision_summary as summary

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "f2_suite2__mutagenicity.log").write_text(
        "INFO suite2/mutagenicity  MRM@lb  beta1=+0.5229 p=0.00050 in 3452.7 s\n"
    )
    (logs / "f2_suite2__grec.log").write_text(
        "INFO suite2/grec  MRM@ub  beta1=+0.4475 beta_size=+0.6531 p=0.00010 in 12.0 s\n"
    )

    rows = {r["dataset"]: r for r in summary.mrm_table([tmp_path / "none"], logs=logs)}

    assert len(rows) == 2, "an unguarded log line must still parse"
    assert not np.isfinite(rows["mutagenicity"]["beta_delta_n"])
    assert "UNMEASURED" in summary.render_beta1(rows["mutagenicity"])
    assert rows["grec"]["beta_delta_n"] == pytest.approx(0.6531)
    assert "UNMEASURED" not in summary.render_beta1(rows["grec"])


# ---------------------------------------------------------------------------
# D4 predictor collinearity
# ---------------------------------------------------------------------------


def test_collinearity_flags_the_case_the_coefficients_hide() -> None:
    """VIF must be read off the design matrix, not guessed from the betas.

    ``coil_del``/ub announces its collinearity (beta_lev = +1.49 with beta_size
    negative). ``aids_iam``/lb does not: +0.10 against +0.91 at R2 = 0.998 looks
    like the cleanest fit in the set and has VIF 15.3. A heuristic over stored
    coefficients catches the harmless case and misses the harmful one, which is
    why the report is computed from the predictors themselves.
    """
    rng = np.random.default_rng(42)
    n = 5000
    independent = np.column_stack(
        [rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)]
    )
    base = rng.normal(size=n)
    collinear = np.column_stack(
        [base, base + 0.05 * rng.normal(size=n), rng.normal(size=n)]
    )

    def max_vif(design: np.ndarray) -> float:
        worst = 0.0
        for j in range(design.shape[1]):
            others = [k for k in range(design.shape[1]) if k != j]
            matrix = np.column_stack([np.ones(design.shape[0]), design[:, others]])
            beta, *_ = np.linalg.lstsq(matrix, design[:, j], rcond=None)
            residual = design[:, j] - matrix @ beta
            total = ((design[:, j] - design[:, j].mean()) ** 2).sum()
            worst = max(worst, 1.0 / max(1.0 - (1.0 - (residual @ residual) / total), 1e-12))
        return worst

    assert max_vif(independent) < 10.0, "independent predictors must be identifiable"
    assert max_vif(collinear) > 10.0, "r ~ 0.999 predictors must not be"


def test_collinearity_report_shape(tmp_path: Path) -> None:
    """The emitted report carries what a reader needs to re-derive the verdict."""
    from benchmarks.real_data.eval_stats import t06_f2 as driver

    assert hasattr(driver, "collinearity_report")
    # The threshold is the conventional one and must not drift silently.
    source = Path(driver.__file__).read_text()
    assert "worst <= 10.0" in source
    assert '"threshold": 10.0' in source
