"""Unit tests for the T-05 cross-fill step.

Cross-fill is the only step that sees all three role campaigns at once, and it
is where ``certified_mask`` comes from. Two things are therefore worth more
scrutiny than the rest: that it never sources a certificate from a backend's
self-report (CONTRACTS §4.1), and that it never attaches a bound from one pair
to another. The second is why the ``graph_ids`` check is a refusal rather than
a warning -- a cohort mismatch produces values that all look plausible.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.eval_setup.approx_ged_crossfill import (
    CROSSFILL_KEYS,
    OUTPUT_KEYS,
    CrossfillError,
    certified_mask,
    crossfill,
    main,
)

N = 5


def _symmetric(values: np.ndarray) -> np.ndarray:
    """Return a symmetric matrix with a zero diagonal."""
    m = np.triu(values, 1)
    m = m + m.T
    return m


def _role_file(
    path: Path,
    *,
    values: np.ndarray,
    key: str = "linux",
    ids: np.ndarray | None = None,
    role: str = "lb",
) -> None:
    """Write one CONTRACTS §4 role file.

    Args:
        path: Destination ``.npz``.
        values: ``(N, N)`` role values, becoming ``ged_matrix``.
        key: Dataset key.
        ids: Optional explicit ``graph_ids``.
        role: Role label for the metadata.
    """
    n = values.shape[0]
    graph_ids = (
        ids if ids is not None else np.array([f"{key}_{t:04d}" for t in range(n)], dtype=str)
    )
    np.savez_compressed(
        path,
        ged_matrix=values.astype(np.float64),
        lb_matrix=np.zeros((n, n), dtype=np.float64),
        ub_matrix=np.full((n, n), np.inf, dtype=np.float64),
        certified_mask=np.zeros((n, n), dtype=np.bool_),
        seconds_matrix=np.full((n, n), 0.25, dtype=np.float32),
        node_counts=np.arange(2, 2 + n, dtype=np.int32),
        edge_counts=np.arange(1, 1 + n, dtype=np.int32),
        graph_ids=graph_ids,
        labels=np.array([""] * n, dtype=str),
        metadata=np.array(json.dumps({"dataset": key, "role": role})),
    )


@pytest.fixture()
def role_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A consistent LB / UB / UB_SENSITIVITY triple over the same cohort."""
    lb = _symmetric(np.full((N, N), 2.0))
    ub = _symmetric(np.full((N, N), 5.0))
    ubs = _symmetric(np.full((N, N), 6.0))
    # One pair where the bracket closes, so certification is non-trivial.
    lb[0, 1] = lb[1, 0] = 5.0
    paths = (tmp_path / "LB.npz", tmp_path / "UB.npz", tmp_path / "UBS.npz")
    _role_file(paths[0], values=lb, role="lb")
    _role_file(paths[1], values=ub, role="ub")
    _role_file(paths[2], values=ubs, role="ubs")
    return paths


# --------------------------------------------------------------------------- #
# certified_mask is derived, never self-reported
# --------------------------------------------------------------------------- #


class TestCertifiedMask:
    """CONTRACTS §4.1."""

    def test_the_mask_is_where_the_two_bounds_meet(self) -> None:
        lb = np.array([[0.0, 3.0], [3.0, 0.0]])
        ub = np.array([[0.0, 3.0], [3.0, 0.0]])
        assert certified_mask(lb, ub).all()

    def test_an_open_bracket_is_not_certified(self) -> None:
        lb = np.array([[0.0, 2.0], [2.0, 0.0]])
        ub = np.array([[0.0, 5.0], [5.0, 0.0]])
        mask = certified_mask(lb, ub)
        assert not mask[0, 1] and not mask[1, 0]
        assert mask[0, 0] and mask[1, 1]

    def test_the_diagonal_is_set_not_measured(self) -> None:
        """A graph's distance to itself is zero by definition; no solver is asked."""
        lb = np.full((3, 3), 1.0)
        ub = np.full((3, 3), 9.0)
        assert np.all(np.diag(certified_mask(lb, ub)))

    def test_no_backend_field_can_reach_the_mask(self, role_files: tuple[Path, ...]) -> None:
        """ANCHOR_AWARE_GED issued a false optimality certificate; nothing self-reports here.

        Every input file carries ``certified_mask`` all-False. The output mask is
        non-trivially True on the closed pair, so it cannot have been copied
        from any input.
        """
        lb_p, ub_p, ubs_p = role_files
        crossfill(lb_path=lb_p, ub_path=ub_p, ubs_path=ubs_p)
        with np.load(lb_p) as data:
            mask = np.asarray(data["certified_mask"])
        assert mask[0, 1] and mask[1, 0]
        assert not mask[0, 2]


# --------------------------------------------------------------------------- #
# What cross-fill writes, and what it must not touch
# --------------------------------------------------------------------------- #


class TestCrossfillWrites:
    """The bracket goes into all three files; each file's own measurement stays."""

    def test_all_three_files_receive_the_same_bracket(
        self, role_files: tuple[Path, Path, Path]
    ) -> None:
        lb_p, ub_p, ubs_p = role_files
        report = crossfill(lb_path=lb_p, ub_path=ub_p, ubs_path=ubs_p)
        assert len(report.written) == 3
        arrays = [dict(np.load(p)) for p in (lb_p, ub_p, ubs_p)]
        for key in CROSSFILL_KEYS:
            for other in arrays[1:]:
                assert np.array_equal(arrays[0][key], other[key]), key

    def test_ged_matrix_and_seconds_matrix_are_never_touched(
        self, role_files: tuple[Path, Path, Path]
    ) -> None:
        """Overwriting ged_matrix would silently make UB_SENSITIVITY a copy of UB."""
        lb_p, ub_p, ubs_p = role_files
        before = {p: dict(np.load(p)) for p in (lb_p, ub_p, ubs_p)}
        crossfill(lb_path=lb_p, ub_path=ub_p, ubs_path=ubs_p)
        for path, original in before.items():
            with np.load(path) as after:
                assert np.array_equal(after["ged_matrix"], original["ged_matrix"])
                assert np.array_equal(after["seconds_matrix"], original["seconds_matrix"])
        # And the three roles still hold three different values.
        with np.load(ub_p) as a, np.load(ubs_p) as b:
            assert not np.array_equal(a["ged_matrix"], b["ged_matrix"])

    def test_the_output_carries_the_ten_contract_keys(
        self, role_files: tuple[Path, Path, Path]
    ) -> None:
        lb_p, ub_p, ubs_p = role_files
        crossfill(lb_path=lb_p, ub_path=ub_p, ubs_path=ubs_p)
        with np.load(lb_p) as data:
            assert set(data.files) == set(OUTPUT_KEYS)
            assert data["lb_matrix"].dtype == np.float64
            assert data["ub_matrix"].dtype == np.float64
            assert data["certified_mask"].dtype == np.bool_
            assert data["seconds_matrix"].dtype == np.float32
            assert data["node_counts"].dtype == np.int32
            assert data["edge_counts"].dtype == np.int32

    def test_the_bounds_come_from_the_right_files(
        self, role_files: tuple[Path, Path, Path]
    ) -> None:
        lb_p, ub_p, ubs_p = role_files
        with np.load(lb_p) as a, np.load(ub_p) as b:
            expected_lb = np.asarray(a["ged_matrix"]).copy()
            expected_ub = np.asarray(b["ged_matrix"]).copy()
        crossfill(lb_path=lb_p, ub_path=ub_p, ubs_path=ubs_p)
        with np.load(ubs_p) as data:
            assert np.array_equal(data["lb_matrix"], expected_lb)
            assert np.array_equal(data["ub_matrix"], expected_ub)

    def test_the_metadata_records_the_certification_rate(
        self, role_files: tuple[Path, Path, Path]
    ) -> None:
        lb_p, ub_p, ubs_p = role_files
        report = crossfill(lb_path=lb_p, ub_path=ub_p, ubs_path=ubs_p)
        with np.load(lb_p) as data:
            meta = json.loads(str(data["metadata"]))
        assert meta["role"] == "lb"  # pre-existing field survives
        assert meta["n_certified"] == report.n_certified
        assert meta["certification_rate"] == pytest.approx(report.certification_rate)
        assert "crossfilled_utc" in meta

    def test_the_sensitivity_arm_is_optional(self, role_files: tuple[Path, Path, Path]) -> None:
        lb_p, ub_p, _ubs_p = role_files
        report = crossfill(lb_path=lb_p, ub_path=ub_p)
        assert len(report.written) == 2

    def test_a_dry_run_writes_nothing(self, role_files: tuple[Path, Path, Path]) -> None:
        lb_p, ub_p, ubs_p = role_files
        before = lb_p.read_bytes()
        report = crossfill(lb_path=lb_p, ub_path=ub_p, ubs_path=ubs_p, dry_run=True)
        assert report.written == []
        assert lb_p.read_bytes() == before


class TestIdempotence:
    """Running it twice must be indistinguishable from running it once."""

    def test_a_second_run_reproduces_the_arrays(self, role_files: tuple[Path, Path, Path]) -> None:
        lb_p, ub_p, ubs_p = role_files
        crossfill(lb_path=lb_p, ub_path=ub_p, ubs_path=ubs_p)
        after_first = {k: v.copy() for k, v in dict(np.load(lb_p)).items() if k != "metadata"}
        report = crossfill(lb_path=lb_p, ub_path=ub_p, ubs_path=ubs_p)
        with np.load(lb_p) as data:
            for key, value in after_first.items():
                assert np.array_equal(data[key], value), key
        assert report.n_certified == 1


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


class TestRefusals:
    """Every one of these produces plausible-looking numbers if it is allowed through."""

    def test_disagreeing_graph_ids_are_refused(self, tmp_path: Path) -> None:
        """Pair indices are positions in this order; a mismatch misattributes every bound."""
        lb = _symmetric(np.full((N, N), 2.0))
        ub = _symmetric(np.full((N, N), 5.0))
        _role_file(tmp_path / "LB.npz", values=lb)
        _role_file(
            tmp_path / "UB.npz",
            values=ub,
            ids=np.array([f"other_{t:04d}" for t in range(N)], dtype=str),
        )
        with pytest.raises(CrossfillError, match="graph_ids disagree"):
            crossfill(lb_path=tmp_path / "LB.npz", ub_path=tmp_path / "UB.npz")

    def test_a_different_cohort_size_is_refused(self, tmp_path: Path) -> None:
        _role_file(tmp_path / "LB.npz", values=_symmetric(np.full((N, N), 2.0)))
        _role_file(tmp_path / "UB.npz", values=_symmetric(np.full((N + 1, N + 1), 5.0)))
        with pytest.raises(CrossfillError, match="not the same cohort"):
            crossfill(lb_path=tmp_path / "LB.npz", ub_path=tmp_path / "UB.npz")

    def test_an_inverted_bracket_is_refused(self, tmp_path: Path) -> None:
        """Two independent campaigns; lb > ub means one of them is wrong."""
        _role_file(tmp_path / "LB.npz", values=_symmetric(np.full((N, N), 9.0)))
        _role_file(tmp_path / "UB.npz", values=_symmetric(np.full((N, N), 5.0)))
        with pytest.raises(CrossfillError, match="lb > ub"):
            crossfill(lb_path=tmp_path / "LB.npz", ub_path=tmp_path / "UB.npz")

    def test_an_unfinished_campaign_is_refused(self, tmp_path: Path) -> None:
        """inf in a bound matrix means the campaign did not produce that pair."""
        lb = _symmetric(np.full((N, N), 2.0))
        lb[0, 1] = lb[1, 0] = np.inf
        _role_file(tmp_path / "LB.npz", values=lb)
        _role_file(tmp_path / "UB.npz", values=_symmetric(np.full((N, N), 5.0)))
        with pytest.raises(CrossfillError, match="did not finish"):
            crossfill(lb_path=tmp_path / "LB.npz", ub_path=tmp_path / "UB.npz")

    def test_a_missing_file_is_refused(self, tmp_path: Path) -> None:
        _role_file(tmp_path / "LB.npz", values=_symmetric(np.full((N, N), 2.0)))
        with pytest.raises(CrossfillError, match="does not exist"):
            crossfill(lb_path=tmp_path / "LB.npz", ub_path=tmp_path / "absent.npz")

    def test_a_file_that_is_not_a_role_file_is_refused(self, tmp_path: Path) -> None:
        np.savez_compressed(tmp_path / "junk.npz", something=np.zeros(3))
        _role_file(tmp_path / "UB.npz", values=_symmetric(np.full((N, N), 5.0)))
        with pytest.raises(CrossfillError, match="missing 'ged_matrix'"):
            crossfill(lb_path=tmp_path / "junk.npz", ub_path=tmp_path / "UB.npz")

    def test_a_refused_crossfill_leaves_every_file_untouched(self, tmp_path: Path) -> None:
        """A partial rewrite would be indistinguishable from a correct one."""
        _role_file(tmp_path / "LB.npz", values=_symmetric(np.full((N, N), 9.0)))
        _role_file(tmp_path / "UB.npz", values=_symmetric(np.full((N, N), 5.0)))
        before = (tmp_path / "LB.npz").read_bytes()
        with pytest.raises(CrossfillError):
            crossfill(lb_path=tmp_path / "LB.npz", ub_path=tmp_path / "UB.npz")
        assert (tmp_path / "LB.npz").read_bytes() == before


class TestCli:
    """The entry point the launcher calls."""

    def test_the_cli_crossfills_and_returns_zero(self, role_files: tuple[Path, Path, Path]) -> None:
        lb_p, ub_p, ubs_p = role_files
        code = main(
            ["--lb", str(lb_p), "--ub", str(ub_p), "--ubs", str(ubs_p), "--log-level", "WARNING"]
        )
        assert code == 0
        with np.load(ubs_p) as data:
            assert np.asarray(data["certified_mask"])[0, 1]

    def test_the_cli_returns_one_on_a_refusal(self, tmp_path: Path) -> None:
        _role_file(tmp_path / "LB.npz", values=_symmetric(np.full((N, N), 9.0)))
        _role_file(tmp_path / "UB.npz", values=_symmetric(np.full((N, N), 5.0)))
        code = main(
            [
                "--lb",
                str(tmp_path / "LB.npz"),
                "--ub",
                str(tmp_path / "UB.npz"),
                "--log-level",
                "CRITICAL",
            ]
        )
        assert code == 1
