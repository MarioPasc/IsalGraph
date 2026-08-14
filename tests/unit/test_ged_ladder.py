"""Unit tests for the T-05 calibration ladder.

The centre of gravity is :func:`test_timeout_is_censored_never_promoted`. Every
"exact GED" matrix in the submitted study was produced by a call that returns its
best-found-so-far cost when the budget expires (T-03-design.md §0), and no test
on the returned *value* can tell that apart from a proven optimum. The test below
induces the timeout deliberately and asserts what is recorded, so the defect
cannot come back silently.

The bounds run through ``ged_bounds``' own BRANCH/BP implementations
(``bounds_kind='networkx'``) wherever the assertion is about the ladder rather
than about GEDLIB, so the suite needs no compiled library. One test opts into
GEDLIB and skips when it is absent; it is the CLAUDE.md cross-check that the two
implementations agree on the same pairs.
"""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from benchmarks.real_data.eval_setup.ged_bounds import UNIT_COSTS
from benchmarks.real_data.eval_setup.ged_ladder import (
    DEFAULT_MIN_PER_DATASET,
    DEFAULT_PAIRS_PER_RUNG,
    DEFAULT_RUNGS,
    DEFAULT_SEED,
    DEFAULT_TRUNCATE_BELOW,
    SUITE2_KEYS,
    LadderError,
    PairRecord,
    RungResult,
    _build_backends,
    allocate_quota,
    load_rung_npz,
    run_rung,
    rung_mass,
    rung_population,
    sample_rung,
    solve_pair,
    write_manifest,
    write_rung_npz,
)


def _has_gedlib() -> bool:
    """Whether the compiled GEDLIB bindings can be imported here."""
    try:
        importlib.import_module("gklearn.gedlib.libraries_import")
        importlib.import_module("gklearn.gedlib.gedlibpy_gxl")
    except Exception:  # pragma: no cover - depends on the machine
        return False
    return True


# --------------------------------------------------------------------------- #
# a synthetic cohort, small enough that A* finishes and big enough to stratify
# --------------------------------------------------------------------------- #


@pytest.fixture
def toy_counts() -> dict[str, np.ndarray]:
    """Node counts for four synthetic cohorts with deliberately unequal mass."""
    return {
        "coil_del": np.array([13] * 6 + [5] * 40, dtype=np.int32),
        "grec": np.array([13] * 3 + [4] * 10, dtype=np.int32),
        "linux": np.array([13, 13, 3], dtype=np.int32),
        "protein": np.array([7, 8, 9], dtype=np.int32),  # nothing at rung 13
    }


def _path_graph(n: int) -> nx.Graph:
    """A path on ``n`` nodes, cheap for A* and non-trivial for the bounds."""
    return nx.path_graph(n)


def _rows(pair_i: np.ndarray, pair_j: np.ndarray) -> set[tuple[int, int]]:
    """Collect ``(i, j)`` pairs as a set, for subset assertions."""
    return set(zip(pair_i.tolist(), pair_j.tolist(), strict=True))


# --------------------------------------------------------------------------- #
# population
# --------------------------------------------------------------------------- #


def test_rung_population_matches_closed_form(toy_counts: dict[str, np.ndarray]) -> None:
    """Enumeration and the ``C(a,2) + a*b`` count agree, per dataset per rung."""
    for key, counts in toy_counts.items():
        for rung in (4, 5, 7, 13):
            pairs = rung_population(counts, rung)
            assert pairs.shape[0] == rung_mass(counts, rung), (key, rung)
            if pairs.size:
                assert np.all(pairs[:, 0] < pairs[:, 1])
                sizes = counts[pairs]
                assert np.all(sizes.max(axis=1) == rung)


def test_rung_population_is_sorted_and_unique(toy_counts: dict[str, np.ndarray]) -> None:
    """The population order is canonical, so the sample does not depend on it."""
    pairs = rung_population(toy_counts["coil_del"], 13)
    as_rows = [tuple(row) for row in pairs.tolist()]
    assert as_rows == sorted(as_rows)
    assert len(set(as_rows)) == len(as_rows)


# --------------------------------------------------------------------------- #
# allocation
# --------------------------------------------------------------------------- #


def test_allocation_hits_the_total_and_the_floor() -> None:
    """Every contributor clears the floor and the split sums to the quota."""
    masses = {"a": 100_000, "b": 30_000, "c": 500, "d": 40}
    alloc = allocate_quota(masses, 250, 20)
    assert sum(alloc.values()) == 250
    assert set(alloc) == set(masses)
    assert min(alloc.values()) >= 20
    # The largest mass still takes the largest share.
    assert max(alloc, key=lambda k: alloc[k]) == "a"


def test_allocation_is_proportional_beyond_the_floor() -> None:
    """Above the floor the split tracks pair mass, not dataset count."""
    masses = {"big": 900_000, "small": 100_000}
    alloc = allocate_quota(masses, 250, 20)
    assert sum(alloc.values()) == 250
    # 20 + 20 floored, 210 split 9:1 -> ~209 / ~41.
    assert alloc["big"] > 5 * alloc["small"]


def test_allocation_never_exceeds_available_mass() -> None:
    """A dataset holding 5 pairs is allocated 5, not its floor of 20."""
    masses = {"tiny": 5, "big": 100_000}
    alloc = allocate_quota(masses, 250, 20)
    assert alloc["tiny"] == 5
    assert sum(alloc.values()) == 250


def test_allocation_ignores_zero_mass_datasets() -> None:
    """A dataset with no pair at this rung is not a contributor."""
    alloc = allocate_quota({"a": 0, "b": 1000}, 250, 20)
    assert set(alloc) == {"b"}


def test_allocation_takes_the_whole_rung_when_it_is_smaller_than_the_quota() -> None:
    """A short rung is reported short rather than resampled with replacement."""
    masses = {"a": 7, "b": 3}
    assert allocate_quota(masses, 250, 20) == masses


def test_allocation_rejects_negative_arguments() -> None:
    """Nonsense parameters raise rather than silently allocating nothing."""
    with pytest.raises(LadderError):
        allocate_quota({"a": 10}, -1, 20)


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #


def test_sampler_is_reproducible_from_the_seed_alone(toy_counts: dict[str, np.ndarray]) -> None:
    """Two independent draws at seed 42 are element-wise identical."""
    a = sample_rung(toy_counts, 13, total=20, minimum=2, seed=DEFAULT_SEED)
    b = sample_rung(toy_counts, 13, total=20, minimum=2, seed=DEFAULT_SEED)
    assert np.array_equal(a.dataset_key, b.dataset_key)
    assert np.array_equal(a.pair_i, b.pair_i)
    assert np.array_equal(a.pair_j, b.pair_j)
    assert a.realised == b.realised


def test_sampler_moves_with_the_seed(toy_counts: dict[str, np.ndarray]) -> None:
    """A different seed draws different pairs; the seed is doing work."""
    a = sample_rung(toy_counts, 13, total=20, minimum=2, seed=42)
    b = sample_rung(toy_counts, 13, total=20, minimum=2, seed=43)
    assert not (np.array_equal(a.pair_i, b.pair_i) and np.array_equal(a.pair_j, b.pair_j))


def test_sampler_does_not_couple_datasets(toy_counts: dict[str, np.ndarray]) -> None:
    """One dataset's draw depends on its own allocation and on nothing else.

    Each generator is seeded on ``[seed, rung, ordinal]``, never threaded across
    datasets, so removing a cohort cannot reshuffle the ones that remain. What it
    *can* do is redistribute the quota, and the surviving cohorts then draw more
    pairs. The invariant is therefore nesting, not equality: the smaller draw is
    a subset of the larger one, with the same pairs at the front. Equality would
    be the wrong assertion -- it would only hold when both allocations happen to
    coincide, which hides the coupling this test exists to rule out.
    """
    subset = {k: v for k, v in toy_counts.items() if k != "grec"}
    full = sample_rung(toy_counts, 13, total=100, minimum=2, seed=DEFAULT_SEED)
    part = sample_rung(subset, 13, total=100, minimum=2, seed=DEFAULT_SEED)
    for key in ("coil_del", "linux"):
        m_full = full.dataset_key == key
        m_part = part.dataset_key == key
        assert int(m_part.sum()) >= int(m_full.sum()), key
        rows_full = _rows(full.pair_i[m_full], full.pair_j[m_full])
        rows_part = _rows(part.pair_i[m_part], part.pair_j[m_part])
        assert rows_full <= rows_part, key

    # And with the allocation pinned, the draw really is identical.
    pinned_full = sample_rung(
        {"coil_del": toy_counts["coil_del"]}, 13, total=17, minimum=1, seed=DEFAULT_SEED
    )
    pinned_part = sample_rung(
        {"coil_del": toy_counts["coil_del"], "linux": toy_counts["linux"]},
        13,
        total=18,
        minimum=1,
        seed=DEFAULT_SEED,
    )
    m = pinned_part.dataset_key == "coil_del"
    assert int(m.sum()) == pinned_full.n_pairs
    assert np.array_equal(pinned_full.pair_i, pinned_part.pair_i[m])
    assert np.array_equal(pinned_full.pair_j, pinned_part.pair_j[m])


def test_reduced_sample_nests_inside_the_full_one(toy_counts: dict[str, np.ndarray]) -> None:
    """A smaller quota draws a subset, so a pilot slice informs the full run.

    Selection is "smallest ``k`` uniform keys", and the keys depend only on
    ``[seed, rung, ordinal]``, so the ``k'``-pair draw is a prefix of the
    ``k``-pair draw for ``k' < k`` within each dataset.
    """
    big = sample_rung(toy_counts, 13, total=40, minimum=4, seed=DEFAULT_SEED)
    small = sample_rung(toy_counts, 13, total=12, minimum=1, seed=DEFAULT_SEED)
    big_set = set(
        zip(big.dataset_key.tolist(), big.pair_i.tolist(), big.pair_j.tolist(), strict=True)
    )
    small_set = set(
        zip(small.dataset_key.tolist(), small.pair_i.tolist(), small.pair_j.tolist(), strict=True)
    )
    assert small_set <= big_set


def test_sampler_honours_the_floor_on_real_shaped_masses(
    toy_counts: dict[str, np.ndarray],
) -> None:
    """Every contributing dataset reaches 20, or its whole population."""
    counts = {
        "coil_del": np.array([13] * 60 + [5] * 400, dtype=np.int32),
        "grec": np.array([13] * 30 + [4] * 100, dtype=np.int32),
        "linux": np.array([13] * 8 + [3] * 8, dtype=np.int32),
    }
    sample = sample_rung(counts, 13, total=DEFAULT_PAIRS_PER_RUNG, minimum=DEFAULT_MIN_PER_DATASET)
    assert sample.n_pairs == DEFAULT_PAIRS_PER_RUNG
    for key, taken in sample.realised.items():
        assert taken >= min(DEFAULT_MIN_PER_DATASET, sample.masses[key]), key


def test_empty_rung_is_reported_not_skipped(toy_counts: dict[str, np.ndarray]) -> None:
    """A rung with no eligible pair yields an empty sample, not an exception."""
    sample = sample_rung(toy_counts, 17, total=250, minimum=20)
    assert sample.is_empty
    assert sample.n_pairs == 0
    assert sample.realised == {}
    assert sample.allocation == {}
    assert all(m == 0 for m in sample.masses.values())


def test_empty_rung_writes_a_valid_file(tmp_path: Path) -> None:
    """An empty rung still produces a schema-valid ``.npz``."""
    result = RungResult(rung=17, records=[], meta={"rung": 17, "n_pairs": 0})
    target = tmp_path / "rung_17.npz"
    write_rung_npz(target, result)
    arrays, meta = load_rung_npz(target)
    assert arrays["pair_i"].shape == (0,)
    assert arrays["exact"].shape == (0,)
    assert meta["rung"] == 17


def test_sampler_rejects_a_key_outside_suite2() -> None:
    """An unknown cohort has no reproducible ordinal, so it is refused."""
    with pytest.raises(LadderError, match="SUITE2_KEYS"):
        sample_rung({"not_a_cohort": np.array([13, 13, 4])}, 13)


def test_suite2_key_order_is_the_seed_contract() -> None:
    """The ordinal list is fixed; reordering it silently changes the sample."""
    assert tuple(sorted(SUITE2_KEYS)) == SUITE2_KEYS
    assert len(SUITE2_KEYS) == 10


# --------------------------------------------------------------------------- #
# censoring -- the important one
# --------------------------------------------------------------------------- #


def test_timeout_is_censored_never_promoted() -> None:
    """A pair whose search did not terminate is censored, not recorded exact.

    ``nx.graph_edit_distance(timeout=t)`` returns its best-so-far cost when the
    budget expires, so the returned value alone cannot distinguish a proven
    optimum from an upper bound. With a budget of 1 ms the search cannot
    complete, and the record must show ``certified=False``, ``exact=inf`` and a
    finite ``lb <= ub``.
    """
    g1, g2 = _path_graph(9), nx.cycle_graph(9)
    bounds, exact_backend = _build_backends(
        UNIT_COSTS, "networkx", "BRANCH_FAST", "", "BIPARTITE", "", 0.001
    )
    exact, lb, ub, certified, seconds = solve_pair(
        g1, g2, bounds_backend=bounds, exact_backend=exact_backend, bounds_kind="networkx"
    )
    assert certified is False
    assert exact == math.inf
    assert math.isfinite(lb) and math.isfinite(ub)
    assert lb <= ub
    assert seconds >= 0.0


def test_write_refuses_a_censored_pair_carrying_a_finite_exact(tmp_path: Path) -> None:
    """The output contract rejects a best-so-far cost promoted to a distance."""
    result = RungResult(
        rung=13,
        records=[
            PairRecord(
                dataset_key="linux",
                pair_i=0,
                pair_j=1,
                n_max=13,
                exact=7.0,  # a value with certified=False is exactly the defect
                lb=3.0,
                ub=11.0,
                certified=False,
                seconds=1.0,
            )
        ],
    )
    with pytest.raises(LadderError, match="censored pair carries a finite exact"):
        write_rung_npz(tmp_path / "rung_13.npz", result)


def test_write_refuses_a_certified_pair_carrying_inf(tmp_path: Path) -> None:
    """The biconditional runs both ways: certified implies a finite value."""
    result = RungResult(
        rung=13,
        records=[
            PairRecord("linux", 0, 1, 13, math.inf, 3.0, 11.0, True, 1.0),
        ],
    )
    with pytest.raises(LadderError, match="certified pair carries a non-finite"):
        write_rung_npz(tmp_path / "rung_13.npz", result)


def test_write_refuses_a_non_finite_bound(tmp_path: Path) -> None:
    """A censored pair must still carry two finite ends, or D11 has no interval."""
    result = RungResult(
        rung=13,
        records=[PairRecord("linux", 0, 1, 13, math.inf, 3.0, math.inf, False, 1.0)],
    )
    with pytest.raises(LadderError, match="must be finite"):
        write_rung_npz(tmp_path / "rung_13.npz", result)


# --------------------------------------------------------------------------- #
# bracket containment
# --------------------------------------------------------------------------- #


def test_certified_pairs_are_contained_in_their_bracket() -> None:
    """``lb <= exact <= ub`` on every certified pair, tolerance 1e-9."""
    counts = {
        "linux": np.array([7] * 5 + [4] * 6, dtype=np.int32),
    }
    graphs = {
        "linux": [_path_graph(7), nx.cycle_graph(7), nx.star_graph(6), nx.wheel_graph(7)]
        + [nx.path_graph(7)]
        + [nx.path_graph(4), nx.cycle_graph(4), nx.star_graph(3)]
        + [nx.path_graph(4)] * 3
    }
    sample = sample_rung(counts, 7, total=12, minimum=2)
    result = run_rung(
        sample,
        graphs,
        bounds_kind="networkx",
        budget_seconds=120.0,
        workers=1,
        progress_every=0,
    )
    assert result.n_pairs > 0
    assert result.n_certified > 0, "the toy cohort must certify, or the test proves nothing"
    for rec in result.records:
        if rec.certified:
            assert rec.lb - 1e-9 <= rec.exact <= rec.ub + 1e-9
        else:
            assert rec.exact == math.inf
        assert math.isfinite(rec.lb) and math.isfinite(rec.ub)
        assert rec.lb <= rec.ub + 1e-9


def test_write_refuses_a_certified_value_outside_its_bracket(tmp_path: Path) -> None:
    """A proven optimum outside its own bounds means one of the two is wrong."""
    result = RungResult(
        rung=13,
        records=[PairRecord("linux", 0, 1, 13, 12.0, 3.0, 11.0, True, 1.0)],
    )
    with pytest.raises(LadderError, match="outside its own bracket"):
        write_rung_npz(tmp_path / "rung_13.npz", result)


def test_solve_pair_raises_on_a_contradicted_optimum() -> None:
    """An independent bound that contradicts A* stops the run, not the pair."""

    class _WrongBounds:
        """A bounds backend whose lower bound is deliberately impossible."""

        def heuristic_bracket(self, g1, g2):  # noqa: ANN001, ANN202, D102
            return 10_000.0, 10_001.0

    _bounds, exact_backend = _build_backends(
        UNIT_COSTS, "networkx", "BRANCH_FAST", "", "BIPARTITE", "", 60.0
    )
    with pytest.raises(LadderError, match="outside its bracket"):
        solve_pair(
            _path_graph(4),
            nx.cycle_graph(4),
            bounds_backend=_WrongBounds(),
            exact_backend=exact_backend,
            bounds_kind="networkx",
        )


# --------------------------------------------------------------------------- #
# truncation
# --------------------------------------------------------------------------- #


def _result_with_rate(rung: int, n: int, certified: int) -> RungResult:
    """Build a rung whose certification rate is exactly ``certified / n``."""
    records = [
        PairRecord(
            "linux",
            i,
            i + 1,
            rung,
            5.0 if i < certified else math.inf,
            1.0,
            9.0,
            i < certified,
            0.5,
        )
        for i in range(n)
    ]
    result = RungResult(rung=rung, records=records)
    result.meta = {
        "rung": rung,
        "n_pairs": result.n_pairs,
        "n_certified": result.n_certified,
        "certification_rate": result.certification_rate,
        "censoring_rate": result.censoring_rate,
    }
    return result


def test_certification_and_censoring_rates_are_complements() -> None:
    """The two rates sum to one, so neither can be quoted without the other."""
    result = _result_with_rate(13, 200, 63)
    assert result.certification_rate == pytest.approx(0.315)
    assert result.certification_rate + result.censoring_rate == pytest.approx(1.0)


def test_truncation_threshold_selects_the_ceiling() -> None:
    """The ceiling is the last rung at or above 25 %; below it the ladder stops."""
    above = _result_with_rate(13, 100, 30)
    at = _result_with_rate(14, 100, 25)
    below = _result_with_rate(15, 100, 24)
    assert above.certification_rate >= DEFAULT_TRUNCATE_BELOW
    assert at.certification_rate >= DEFAULT_TRUNCATE_BELOW, "the threshold is inclusive"
    assert below.certification_rate < DEFAULT_TRUNCATE_BELOW


def test_empty_rung_truncates() -> None:
    """A rung with nothing to certify certifies at 0 %, which is below 25 %."""
    empty = RungResult(rung=19, records=[])
    assert empty.certification_rate == 0.0
    assert empty.certification_rate < DEFAULT_TRUNCATE_BELOW


def test_manifest_records_the_ceiling_as_a_measurement(tmp_path: Path) -> None:
    """The manifest carries the ceiling, the truncating rung and the threshold."""
    metas = [_result_with_rate(13, 100, 40).meta, _result_with_rate(14, 100, 10).meta]
    path = write_manifest(tmp_path, metas, ceiling=13, truncated_at=14, threshold=0.25, seed=42)
    payload = json.loads(path.read_text())
    assert payload["exact_ged_ceiling"] == 13
    assert payload["truncated_at_rung"] == 14
    assert payload["truncate_below"] == 0.25
    assert payload["seed"] == 42
    assert len(payload["rungs"]) == 2


def test_manifest_falls_back_to_twelve_when_no_rung_certifies(tmp_path: Path) -> None:
    """No rung above T-03's census reaching 25 % leaves the ceiling at n = 12."""
    metas = [_result_with_rate(13, 100, 5).meta]
    path = write_manifest(tmp_path, metas, ceiling=12, truncated_at=13, threshold=0.25, seed=42)
    assert json.loads(path.read_text())["exact_ged_ceiling"] == 12


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #


def test_rung_file_carries_the_contracted_keys_and_dtypes(tmp_path: Path) -> None:
    """The eight arrays and the metadata JSON, with the contracted dtypes."""
    result = _result_with_rate(13, 4, 2)
    result.meta.update(
        {
            "per_dataset_counts": {"linux": 4},
            "seed": 42,
            "budget_seconds": 1200.0,
            "cost_model": "unit",
            "lb_method": "BRANCH_FAST",
            "lb_options": "--threads 1",
            "ub_method": "BIPARTITE",
            "ub_options": "--threads 1",
            "solver": "networkx.graph_edit_distance",
            "code_commit": "deadbeef",
            "computed_utc": "2026-08-14T00:00:00Z",
            "schema_version": "ladder-1",
        }
    )
    target = tmp_path / "rung_13.npz"
    write_rung_npz(target, result)

    arrays, meta = load_rung_npz(target)
    assert set(arrays) == {
        "dataset_key",
        "pair_i",
        "pair_j",
        "n_max",
        "exact",
        "lb",
        "ub",
        "certified",
        "seconds",
    }
    assert arrays["pair_i"].dtype == np.int32
    assert arrays["pair_j"].dtype == np.int32
    assert arrays["n_max"].dtype == np.int32
    assert arrays["exact"].dtype == np.float64
    assert arrays["lb"].dtype == np.float64
    assert arrays["ub"].dtype == np.float64
    assert arrays["certified"].dtype == bool
    assert arrays["seconds"].dtype == np.float32
    assert arrays["dataset_key"].dtype.kind == "U"
    for key in (
        "rung",
        "n_pairs",
        "n_certified",
        "certification_rate",
        "censoring_rate",
        "per_dataset_counts",
        "seed",
        "budget_seconds",
        "cost_model",
        "lb_method",
        "lb_options",
        "ub_method",
        "ub_options",
        "solver",
        "code_commit",
        "computed_utc",
        "schema_version",
    ):
        assert key in meta, key


def test_censored_pairs_carry_inf_not_nan(tmp_path: Path) -> None:
    """T-03's census uses ``inf``; a consumer filters with ``np.isfinite``."""
    target = tmp_path / "rung_13.npz"
    write_rung_npz(target, _result_with_rate(13, 6, 2))
    arrays, _meta = load_rung_npz(target)
    censored = ~arrays["certified"]
    assert not np.any(np.isnan(arrays["exact"]))
    assert np.all(np.isinf(arrays["exact"][censored]))


def test_write_refuses_a_rung_mismatch(tmp_path: Path) -> None:
    """``n_max`` must equal the rung on every pair, by construction."""
    result = RungResult(rung=13, records=[PairRecord("linux", 0, 1, 14, 5.0, 1.0, 9.0, True, 0.5)])
    with pytest.raises(LadderError, match="n_max must equal the rung"):
        write_rung_npz(tmp_path / "rung_13.npz", result)


# --------------------------------------------------------------------------- #
# the retired solver, and the two implementations that must agree
# --------------------------------------------------------------------------- #


def test_anchor_aware_ged_is_unreachable() -> None:
    """The retired false-certificate solver cannot be selected, even by name.

    T-03-design.md amendment 2: measured non-deterministic on 14 of 15 real AIDS
    pairs, wrong on 4 of 18 against brute force, and it reports ``LB == UB``.
    """
    from benchmarks.real_data.eval_setup.ged_backends import RETIRED_METHODS, GedBackendError

    assert "ANCHOR_AWARE_GED" in RETIRED_METHODS
    with pytest.raises(GedBackendError):
        _build_backends(
            UNIT_COSTS,
            "gedlib",
            "ANCHOR_AWARE_GED",
            "--threads 1",
            "BIPARTITE",
            "--threads 1",
            60.0,
        )


def test_default_rungs_start_above_the_census_ceiling() -> None:
    """The ladder starts at 13, one node above T-03's exact-GED ceiling."""
    assert DEFAULT_RUNGS[0] == 13
    assert DEFAULT_RUNGS == (13, 14, 15, 16, 17, 18)


@pytest.mark.skipif(not _has_gedlib(), reason="compiled GEDLIB bindings not importable")
def test_gedlib_and_ged_bounds_agree_as_bounds() -> None:
    """Both bracket the same optimum -- CLAUDE.md's mandatory cross-check.

    The two implementations are not required to return the same numbers: BP and
    BIPARTITE are different heuristics. They are required to be *valid*, that is,
    to bracket the A* optimum, and a disagreement about that is a bug in one of
    them.
    """
    pairs = [
        (nx.path_graph(4), nx.cycle_graph(4)),
        (nx.star_graph(4), nx.path_graph(5)),
        (nx.cycle_graph(6), nx.wheel_graph(6)),
    ]
    gl_bounds, exact_backend = _build_backends(
        UNIT_COSTS, "gedlib", "BRANCH_FAST", "--threads 1", "BIPARTITE", "--threads 1", 120.0
    )
    nx_bounds, _ = _build_backends(
        UNIT_COSTS, "networkx", "BRANCH_FAST", "", "BIPARTITE", "", 120.0
    )
    for g1, g2 in pairs:
        e_gl, lb_gl, ub_gl, cert_gl, _ = solve_pair(
            g1,
            g2,
            bounds_backend=gl_bounds,
            exact_backend=exact_backend,
            bounds_kind="gedlib",
        )
        e_nx, lb_nx, ub_nx, cert_nx, _ = solve_pair(
            g1,
            g2,
            bounds_backend=nx_bounds,
            exact_backend=exact_backend,
            bounds_kind="networkx",
        )
        assert cert_gl and cert_nx
        assert e_gl == pytest.approx(e_nx)
        assert lb_gl - 1e-9 <= e_gl <= ub_gl + 1e-9
        assert lb_nx - 1e-9 <= e_nx <= ub_nx + 1e-9
