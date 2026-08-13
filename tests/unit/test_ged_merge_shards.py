"""Tests for the shard merge and gate 4 (T-03, CONTRACTS sections 6 and 7).

The merge is the last point at which a corrupt GED matrix can be caught before it
becomes a number in a journal table, so its assertions are tested as carefully as
the arithmetic they guard:

* **Coverage.** A pair absent from every shard fails the merge. Silently writing
  ``inf`` for it would be indistinguishable from a legitimately censored pair.
* **Consistency.** Two shards that disagree about the same pair fail. This is how
  stage-1 reuse is verified: the stage-2 census seeds from stage-1, both shards
  land in the merge directory, and agreement on the overlap is the check.
* **Gate 4.** Symmetry, a zero diagonal, ``lb <= ged <= ub``, and censored entries
  that really are flagged as censored.
* **Downstream compatibility.** The six legacy keys keep their names and dtypes so
  ``eval_correlation.py``, ``method_comparator.py``, ``dataset_filter.py`` and
  ``validator.py`` consume the output unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.eval_setup.ged_merge_shards import (
    MergeError,
    collect_shards,
    gate4,
    main,
    merge_shards,
)
from benchmarks.eval_setup.ged_pair_index import n_pairs, pairs_from_indices

N_GRAPHS = 9
TOTAL = n_pairs(N_GRAPHS)  # 36


def _write_cohort(path: Path, n: int = N_GRAPHS, key: str = "aids") -> None:
    """Write a CONTRACT A file carrying the per-graph arrays CONTRACT D needs.

    Args:
        path: Destination ``.npz``.
        n: Number of graphs.
        key: Dataset key.
    """
    np.savez_compressed(
        path,
        graph_ids=np.array([f"{key}_{t:03d}" for t in range(n)], dtype=str),
        n_nodes=np.arange(2, 2 + n, dtype=np.int32),
        n_edges=np.arange(1, 1 + n, dtype=np.int32),
        edge_offsets=np.zeros(n + 1, dtype=np.int64),
        edges=np.zeros((2, 0), dtype=np.int32),
        labels=np.array([f"c{t % 3}" for t in range(n)], dtype=str),
        metadata=np.array(
            json.dumps(
                {
                    "dataset": key,
                    "source": "synthetic",
                    "filter": {"min_nodes": 2, "require_connected": True, "n_max": 12},
                    "n_dropped_size": 3,
                    "n_dropped_disconnected": 1,
                    "n_dropped_trivial": 0,
                }
            )
        ),
    )


def _rows(pairs: np.ndarray, *, censor: set[int] | None = None) -> dict[str, np.ndarray]:
    """Build deterministic shard arrays for a set of pair indices.

    Args:
        pairs: Linear pair indices.
        censor: Indices to mark interval-censored instead of certified.

    Returns:
        The six CONTRACT C arrays.
    """
    censor = censor or set()
    ged, lb, ub, cert = [], [], [], []
    for k in pairs:
        base = float(int(k) % 7 + 1)
        if int(k) in censor:
            ged.append(float("inf"))
            lb.append(base)
            ub.append(base + 4.0)
            cert.append(False)
        else:
            ged.append(base)
            lb.append(base)
            ub.append(base)
            cert.append(True)
    return {
        "pair_index": np.asarray(pairs, dtype=np.int64),
        "ged": np.asarray(ged, dtype=np.float64),
        "lb": np.asarray(lb, dtype=np.float64),
        "ub": np.asarray(ub, dtype=np.float64),
        "certified": np.asarray(cert, dtype=np.bool_),
        "seconds": np.full(len(pairs), 0.5, dtype=np.float32),
    }


def _write_shard(
    path: Path, pairs: np.ndarray, *, censor: set[int] | None = None, **meta: object
) -> None:
    """Write one CONTRACT C shard.

    Args:
        path: Destination ``.npz``.
        pairs: Pair indices.
        censor: Indices to censor.
        **meta: Extra fields for the shard metadata.
    """
    payload: dict[str, np.ndarray | np.generic] = dict(_rows(pairs, censor=censor))
    payload["meta"] = np.array(json.dumps({"cost_model": "unit", "backend_name": "stub", **meta}))
    np.savez_compressed(path, **payload)


@pytest.fixture()
def shard_dir(tmp_path: Path) -> Path:
    """A directory holding a cohort file and three complete, disjoint shards."""
    _write_cohort(tmp_path / "aids.npz")
    for t, chunk in enumerate(np.array_split(np.arange(TOTAL, dtype=np.int64), 3)):
        _write_shard(tmp_path / f"aids_c{t:04d}.npz", chunk, censor={5, 17})
    return tmp_path


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def test_checkpoints_and_pair_lists_are_not_mistaken_for_shards(shard_dir: Path) -> None:
    """A checkpoint is legitimately shard-shaped; merging one would double-count."""
    _write_shard(shard_dir / "aids_c0000.ckpt.npz", np.array([0, 1], dtype=np.int64))
    np.savez_compressed(shard_dir / "aids_pair_list.npz", pair_index=np.array([0, 1]))
    found = collect_shards(shard_dir, "aids", exclude=set())
    assert [p.name for p in found] == ["aids_c0000.npz", "aids_c0001.npz", "aids_c0002.npz"]


def test_an_empty_shard_directory_is_an_error(tmp_path: Path) -> None:
    """Merging nothing must not produce an all-inf matrix."""
    with pytest.raises(MergeError, match="no CONTRACT C shards"):
        collect_shards(tmp_path, "aids", exclude=set())


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #


def test_merge_writes_contract_d_with_every_required_key(shard_dir: Path) -> None:
    """Six legacy keys with their original dtypes, plus the four D11/D12 additions."""
    out = shard_dir / "merged.npz"
    report, shards = merge_shards(shard_dir=shard_dir, key="aids", n_graphs=N_GRAPHS, out=out)
    assert report.passed
    assert len(shards) == 3
    with np.load(out) as data:
        assert data["ged_matrix"].shape == (N_GRAPHS, N_GRAPHS)
        assert data["ged_matrix"].dtype == np.float64
        assert data["node_counts"].dtype == np.int32
        assert data["edge_counts"].dtype == np.int32
        assert data["graph_ids"].size == N_GRAPHS
        assert data["labels"].size == N_GRAPHS
        assert data["lb_matrix"].dtype == np.float64
        assert data["ub_matrix"].dtype == np.float64
        assert data["certified_mask"].dtype == np.bool_
        assert data["seconds_matrix"].dtype == np.float32
        meta = json.loads(str(data["metadata"]))
    for legacy in (
        "dataset",
        "ged_method",
        "ged_cost_function",
        "source",
        "n_graphs",
        "n_valid_pairs",
        "n_max_filter",
        "n_dropped",
    ):
        assert legacy in meta, f"downstream readers expect {legacy}"
    assert meta["n_max_filter"] == 12
    assert meta["n_dropped"] == 4
    assert meta["n_censored"] == 2
    assert meta["n_valid_pairs"] == TOTAL - 2
    assert meta["gate4"]["passed"] is True
    # n_zero_offdiag_certified is a REPORTED quantity, not a diagnostic: it is the
    # "GED > 0" rung of the per-dataset pair-accounting ladder
    # (raw -> connected -> GED-available -> GED > 0 -> Lev > 0 -> analysed), so it
    # must reach the merged metadata rather than existing only in a log line.
    assert "n_zero_offdiag_certified" in meta
    assert meta["n_zero_offdiag_certified"] == 0
    assert meta["n_certified"] + meta["n_censored"] == TOTAL


def test_the_matrix_places_every_pair_at_the_index_it_came_from(shard_dir: Path) -> None:
    """Row k of the shards lands at (i, j) with index_of_pair(i, j, N) == k.

    This is the merge-side half of the index correctness argument: the runner
    computes the pair the index names, and the merge writes it where the index
    says. Either half being wrong transposes the matrix invisibly.
    """
    out = shard_dir / "merged.npz"
    merge_shards(shard_dir=shard_dir, key="aids", n_graphs=N_GRAPHS, out=out)
    expected = _rows(np.arange(TOTAL, dtype=np.int64), censor={5, 17})
    i, j = pairs_from_indices(np.arange(TOTAL, dtype=np.int64), N_GRAPHS)
    with np.load(out) as data:
        ged = data["ged_matrix"]
        lb = data["lb_matrix"]
    assert np.array_equal(ged[i, j], expected["ged"])
    assert np.array_equal(ged[j, i], expected["ged"])
    assert np.array_equal(lb[i, j], expected["lb"])


def test_a_missing_pair_fails_the_merge(tmp_path: Path) -> None:
    """An incomplete matrix is refused rather than padded with inf."""
    _write_cohort(tmp_path / "aids.npz")
    _write_shard(tmp_path / "aids_c0000.npz", np.arange(TOTAL - 1, dtype=np.int64))
    with pytest.raises(MergeError, match="absent from every shard"):
        merge_shards(shard_dir=tmp_path, key="aids", n_graphs=N_GRAPHS, out=tmp_path / "m.npz")


def test_overlapping_shards_that_agree_are_accepted_and_counted(tmp_path: Path) -> None:
    """Stage-2 seeded from stage 1 overlaps stage 1; identical values are the check."""
    _write_cohort(tmp_path / "aids.npz")
    _write_shard(tmp_path / "aids_c0000.npz", np.arange(0, 20, dtype=np.int64))
    _write_shard(tmp_path / "aids_c0001.npz", np.arange(10, TOTAL, dtype=np.int64))
    out = tmp_path / "m.npz"
    report, _ = merge_shards(shard_dir=tmp_path, key="aids", n_graphs=N_GRAPHS, out=out)
    assert report.passed
    with np.load(out) as data:
        assert json.loads(str(data["metadata"]))["n_duplicate_pairs"] == 10


def test_overlapping_shards_that_disagree_fail_hard(tmp_path: Path) -> None:
    """A disagreement means the two stages computed different answers. That is fatal."""
    _write_cohort(tmp_path / "aids.npz")
    _write_shard(tmp_path / "aids_c0000.npz", np.arange(0, 20, dtype=np.int64))
    rows = _rows(np.arange(10, TOTAL, dtype=np.int64))
    rows["ged"][0] += 1.0  # pair 10 disagrees with the first shard
    rows["lb"][0] += 1.0
    rows["ub"][0] += 1.0
    payload: dict[str, np.ndarray | np.generic] = dict(rows)
    payload["meta"] = np.array(json.dumps({"cost_model": "unit"}))
    np.savez_compressed(tmp_path / "aids_c0001.npz", **payload)
    with pytest.raises(MergeError, match="disagree between shards"):
        merge_shards(shard_dir=tmp_path, key="aids", n_graphs=N_GRAPHS, out=tmp_path / "m.npz")


def test_censored_pairs_agreeing_across_shards_are_not_read_as_a_conflict(
    tmp_path: Path,
) -> None:
    """inf == inf: a censored pair present in two shards must not trip the check."""
    _write_cohort(tmp_path / "aids.npz")
    _write_shard(tmp_path / "aids_c0000.npz", np.arange(0, 20, dtype=np.int64), censor={5, 12})
    _write_shard(tmp_path / "aids_c0001.npz", np.arange(10, TOTAL, dtype=np.int64), censor={12})
    report, _ = merge_shards(
        shard_dir=tmp_path, key="aids", n_graphs=N_GRAPHS, out=tmp_path / "m.npz"
    )
    assert report.passed
    assert report.n_censored == 2


def test_shards_mixing_cost_models_are_refused(tmp_path: Path) -> None:
    """Unit and GraphEdX costs are not comparable; merging them would be meaningless."""
    _write_cohort(tmp_path / "aids.npz")
    _write_shard(tmp_path / "aids_c0000.npz", np.arange(0, 20, dtype=np.int64))
    payload: dict[str, np.ndarray | np.generic] = dict(_rows(np.arange(20, TOTAL, dtype=np.int64)))
    payload["meta"] = np.array(json.dumps({"cost_model": "graphedx"}))
    np.savez_compressed(tmp_path / "aids_c0001.npz", **payload)
    with pytest.raises(MergeError, match="mix cost models"):
        merge_shards(shard_dir=tmp_path, key="aids", n_graphs=N_GRAPHS, out=tmp_path / "m.npz")


def test_a_graph_count_mismatch_stops_the_merge(shard_dir: Path) -> None:
    """--n-graphs is the cohort guard; a mismatch means the wrong cohort file."""
    with pytest.raises(MergeError, match="n-graphs"):
        merge_shards(
            shard_dir=shard_dir, key="aids", n_graphs=N_GRAPHS + 1, out=shard_dir / "m.npz"
        )


def test_a_missing_cohort_file_is_explained_not_guessed(tmp_path: Path) -> None:
    """CONTRACT D cannot be written without the per-graph arrays; say so."""
    _write_shard(tmp_path / "aids_c0000.npz", np.arange(TOTAL, dtype=np.int64))
    with pytest.raises(MergeError, match="CONTRACT A"):
        merge_shards(shard_dir=tmp_path, key="aids", n_graphs=N_GRAPHS, out=tmp_path / "m.npz")


# --------------------------------------------------------------------------- #
# Gate 4
# --------------------------------------------------------------------------- #


def _clean_matrices(n: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a small, gate-4-clean set of matrices.

    Args:
        n: Matrix order.

    Returns:
        ``(ged, lb, ub, certified)``.
    """
    base = np.arange(1.0, n * n + 1).reshape(n, n)
    ged = np.triu(base, 1)
    ged = ged + ged.T
    lb = ged.copy()
    ub = ged.copy()
    cert = np.ones((n, n), dtype=bool)
    return ged, lb, ub, cert


def test_gate4_passes_a_clean_matrix() -> None:
    """The happy path, so the failure cases below mean something."""
    assert gate4(*_clean_matrices()).passed


def test_gate4_accepts_a_properly_flagged_censored_entry() -> None:
    """D11: inf is legal when the pair is uncertified and carries a finite bracket."""
    ged, lb, ub, cert = _clean_matrices()
    ged[0, 1] = ged[1, 0] = np.inf
    cert[0, 1] = cert[1, 0] = False
    ub[0, 1] = ub[1, 0] = lb[0, 1] + 5.0
    ub[1, 0] = ub[0, 1]
    rep = gate4(ged, lb, ub, cert)
    assert rep.passed
    assert rep.n_censored == 1


@pytest.mark.parametrize(
    "corrupt",
    ["asymmetric", "nonzero_diagonal", "inf_bound", "outside_bracket", "certified_inf"],
)
def test_gate4_catches_each_structural_defect(corrupt: str) -> None:
    """Every clause of CONTRACTS section 7 rejects the defect it exists for."""
    ged, lb, ub, cert = _clean_matrices()
    if corrupt == "asymmetric":
        ged[0, 1] = 99.0
    elif corrupt == "nonzero_diagonal":
        ged[2, 2] = 0.5
    elif corrupt == "inf_bound":
        ub[0, 1] = ub[1, 0] = np.inf
    elif corrupt == "outside_bracket":
        ub[0, 1] = ub[1, 0] = ged[0, 1] - 1.0
    elif corrupt == "certified_inf":
        ged[0, 1] = ged[1, 0] = np.inf
    rep = gate4(ged, lb, ub, cert)
    assert not rep.passed
    assert rep.violations


def test_gate4_rejects_an_uncertified_zero_the_silently_zero_filled_matrix() -> None:
    """A zero from calling get_lower_bound() on an upper-bound method must fail.

    CONTRACT B section 5 invariant 1: GEDLIB returns 0.00 rather than raising, and a
    whole matrix can fill with zeros with nothing complaining.
    """
    ged, lb, ub, cert = _clean_matrices()
    ged[0, 1] = ged[1, 0] = 0.0
    lb[0, 1] = lb[1, 0] = 0.0
    cert[0, 1] = cert[1, 0] = False
    rep = gate4(ged, lb, ub, cert)
    assert not rep.passed
    assert any("not certified" in v for v in rep.violations)


def test_gate4_accepts_a_certified_zero_but_counts_it_and_can_reject_it() -> None:
    """An isomorphic pair genuinely has GED 0; --strict-nonzero enforces the literal text."""
    ged, lb, ub, cert = _clean_matrices()
    ged[0, 1] = ged[1, 0] = 0.0
    lb[0, 1] = lb[1, 0] = 0.0
    ub[0, 1] = ub[1, 0] = 0.0
    rep = gate4(ged, lb, ub, cert)
    assert rep.passed
    assert rep.n_zero_offdiag == 1
    strict = gate4(ged, lb, ub, cert, strict_nonzero=True)
    assert not strict.passed


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_merges_and_deletes_the_shards_only_after_passing(shard_dir: Path) -> None:
    """--delete-shards runs last, so a failed gate leaves the inputs recoverable."""
    out = shard_dir / "aids_merged.npz"
    code = main(
        [
            "--shards",
            str(shard_dir),
            "--key",
            "aids",
            "--n-graphs",
            str(N_GRAPHS),
            "--out",
            str(out),
            "--delete-shards",
            "--log-level",
            "WARNING",
        ]
    )
    assert code == 0
    assert out.is_file()
    assert not list(shard_dir.glob("aids_c*.npz"))


def test_cli_leaves_the_shards_alone_when_the_merge_fails(tmp_path: Path) -> None:
    """A failed merge must not destroy the only copy of a hundred core-hours of work."""
    _write_cohort(tmp_path / "aids.npz")
    _write_shard(tmp_path / "aids_c0000.npz", np.arange(TOTAL - 2, dtype=np.int64))
    code = main(
        [
            "--shards",
            str(tmp_path),
            "--key",
            "aids",
            "--n-graphs",
            str(N_GRAPHS),
            "--out",
            str(tmp_path / "m.npz"),
            "--delete-shards",
            "--log-level",
            "CRITICAL",
        ]
    )
    assert code == 1
    assert (tmp_path / "aids_c0000.npz").is_file()
    assert not (tmp_path / "m.npz").exists()


# --------------------------------------------------------------------------- #
# T-05: role files, and the zero-fraction gate
# --------------------------------------------------------------------------- #


def _bound_rows(pairs: np.ndarray, *, value: float, side: str) -> dict[str, np.ndarray]:
    """Shard arrays as a one-sided role campaign writes them.

    Args:
        pairs: Linear pair indices.
        value: The bound this role computed for every pair.
        side: ``'lb'`` or ``'ub'``.

    Returns:
        The six CONTRACT C arrays, with the unevaluated end at its sentinel.
    """
    m = len(pairs)
    lb = np.full(m, value if side == "lb" else -np.inf, dtype=np.float64)
    ub = np.full(m, value if side == "ub" else np.inf, dtype=np.float64)
    return {
        "pair_index": np.asarray(pairs, dtype=np.int64),
        "ged": np.full(m, np.inf, dtype=np.float64),
        "lb": lb,
        "ub": ub,
        "certified": np.zeros(m, dtype=np.bool_),
        "seconds": np.full(m, 0.25, dtype=np.float32),
    }


def _write_bound_shard(path: Path, pairs: np.ndarray, *, value: float, side: str) -> None:
    """Write one CONTRACT C shard from a one-sided role campaign.

    Args:
        path: Destination ``.npz``.
        pairs: Pair indices.
        value: The bound for every pair.
        side: ``'lb'`` or ``'ub'``.
    """
    payload: dict[str, np.ndarray | np.generic] = dict(_bound_rows(pairs, value=value, side=side))
    payload["meta"] = np.array(
        json.dumps(
            {
                "cost_model": "unit",
                "backend_name": "gedlib",
                "compute": side,
                f"{side}_method": "BRANCH_FAST" if side == "lb" else "BIPARTITE",
                f"{side}_options": "--threads 1",
            }
        )
    )
    np.savez_compressed(path, **payload)


class TestGedFrom:
    """CONTRACTS section 7: which array becomes ged_matrix."""

    def test_the_default_is_still_exact(self, shard_dir: Path) -> None:
        """T-03's merge is untouched."""
        out = shard_dir / "m.npz"
        merge_shards(shard_dir=shard_dir, key="aids", n_graphs=N_GRAPHS, out=out)
        with np.load(out) as data:
            meta = json.loads(str(data["metadata"]))
        assert meta["ged_from"] == "exact"
        assert meta["accessor"] == "exact"

    def test_ged_from_lb_reports_the_lower_bound_as_the_value(self, tmp_path: Path) -> None:
        """A role file's ged_matrix is that role's own number (CONTRACTS section 4)."""
        _write_cohort(tmp_path / "linux.npz", key="linux")
        _write_bound_shard(
            tmp_path / "linux_c0000.npz", np.arange(TOTAL, dtype=np.int64), value=3.0, side="lb"
        )
        out = tmp_path / "LB.npz"
        report, _ = merge_shards(
            shard_dir=tmp_path,
            key="linux",
            n_graphs=N_GRAPHS,
            out=out,
            ged_from="lb",
            role="lb",
            seconds_role="lb",
        )
        assert report.passed
        with np.load(out) as data:
            ged = np.asarray(data["ged_matrix"])
            lb = np.asarray(data["lb_matrix"])
            meta = json.loads(str(data["metadata"]))
        off = ~np.eye(N_GRAPHS, dtype=bool)
        assert np.array_equal(ged[off], lb[off])
        assert np.all(np.diag(ged) == 0.0)
        assert meta["role"] == "lb"
        assert meta["method"] == "BRANCH_FAST"
        assert meta["options_string"] == "--threads 1"
        assert meta["accessor"] == "lower"
        assert meta["compute"] == "lb"

    def test_an_unknown_ged_from_is_refused(self, shard_dir: Path) -> None:
        with pytest.raises(MergeError, match="--ged-from must be"):
            merge_shards(
                shard_dir=shard_dir,
                key="aids",
                n_graphs=N_GRAPHS,
                out=shard_dir / "m.npz",
                ged_from="upper",
            )

    def test_shards_mixing_compute_modes_are_refused(self, tmp_path: Path) -> None:
        """Half a matrix at its sentinel and half measured, with nothing to say which."""
        _write_cohort(tmp_path / "linux.npz", key="linux")
        chunks = np.array_split(np.arange(TOTAL, dtype=np.int64), 2)
        _write_bound_shard(tmp_path / "linux_c0000.npz", chunks[0], value=3.0, side="lb")
        payload: dict[str, np.ndarray | np.generic] = dict(_rows(chunks[1]))
        payload["meta"] = np.array(json.dumps({"cost_model": "unit", "compute": "both"}))
        np.savez_compressed(tmp_path / "linux_c0001.npz", **payload)
        with pytest.raises(MergeError, match="mix compute modes"):
            merge_shards(
                shard_dir=tmp_path,
                key="linux",
                n_graphs=N_GRAPHS,
                out=tmp_path / "m.npz",
                ged_from="lb",
            )


class TestZeroFractionGate:
    """The shape of the silent-all-zeros failure a wrong accessor produces."""

    def test_an_all_zero_matrix_fails_the_gate(self) -> None:
        """get_lower_bound() on an upper-bound method returns 0.00 and raises nothing."""
        n = 6
        ged = np.zeros((n, n), dtype=np.float64)
        lb = np.zeros((n, n), dtype=np.float64)
        ub = np.zeros((n, n), dtype=np.float64)
        cert = np.ones((n, n), dtype=np.bool_)
        rep = gate4(ged, lb, ub, cert, ged_from="lb")
        assert not rep.passed
        assert rep.zero_offdiag_fraction == 1.0
        assert any("wrong accessor" in v for v in rep.violations)

    def test_the_fraction_is_recorded_even_when_the_gate_passes(self) -> None:
        ged, lb, ub, cert = _clean_matrices()
        rep = gate4(ged, lb, ub, cert)
        assert rep.passed
        assert rep.as_dict()["zero_offdiag_fraction"] == rep.zero_offdiag_fraction

    def test_a_zero_bound_needs_no_certificate(self) -> None:
        """BRANCH_FAST returns the trivial lower bound 0 on real pairs of distance 2 and 6.

        Under ``ged_from='exact'`` that same matrix is a claim of isomorphism and
        is refused without a certificate; under ``ged_from='lb'`` it is a valid,
        merely uninformative bound.
        """
        n = 6
        ged = np.full((n, n), 4.0, dtype=np.float64)
        np.fill_diagonal(ged, 0.0)
        ged[0, 1] = ged[1, 0] = 0.0
        lb = ged.copy()
        ub = np.full((n, n), 9.0, dtype=np.float64)
        np.fill_diagonal(ub, 0.0)
        cert = np.zeros((n, n), dtype=np.bool_)
        np.fill_diagonal(cert, True)
        assert gate4(ged, lb, ub, cert, ged_from="lb").passed
        rep = gate4(ged, lb, ub, cert, ged_from="exact")
        assert not rep.passed
        assert any("silently-zero-filled" in v for v in rep.violations)


class TestOrientationSummary:
    """The asymmetry statistic, aggregated and never densified."""

    def test_the_summary_is_absent_when_no_shard_recorded_orientations(
        self, shard_dir: Path
    ) -> None:
        out = shard_dir / "m.npz"
        merge_shards(shard_dir=shard_dir, key="aids", n_graphs=N_GRAPHS, out=out)
        with np.load(out) as data:
            meta = json.loads(str(data["metadata"]))
        assert meta["ub_orientation"] == {}

    def test_the_summary_is_pooled_across_shards_without_a_new_matrix(self, tmp_path: Path) -> None:
        """Four numbers in the metadata, not two more (N, N) arrays."""
        _write_cohort(tmp_path / "linux.npz", key="linux")
        pairs = np.arange(TOTAL, dtype=np.int64)
        for t, chunk in enumerate(np.array_split(pairs, 2)):
            payload: dict[str, np.ndarray | np.generic] = dict(_rows(chunk))
            # fwd exceeds rev by 2 on every pair, so the reverse is always tighter.
            payload["ub_fwd"] = payload["ub"] + 2.0
            payload["ub_rev"] = payload["ub"].copy()
            payload["meta"] = np.array(json.dumps({"cost_model": "unit"}))
            np.savez_compressed(tmp_path / f"linux_c{t:04d}.npz", **payload)
        out = tmp_path / "m.npz"
        merge_shards(shard_dir=tmp_path, key="linux", n_graphs=N_GRAPHS, out=out)
        with np.load(out) as data:
            meta = json.loads(str(data["metadata"]))
            assert "ub_fwd_matrix" not in data
            assert "ub_rev_matrix" not in data
            assert set(data.files) == {
                "ged_matrix",
                "lb_matrix",
                "ub_matrix",
                "certified_mask",
                "seconds_matrix",
                "node_counts",
                "edge_counts",
                "graph_ids",
                "labels",
                "metadata",
            }
        summary = meta["ub_orientation"]
        assert summary["n_pairs"] == TOTAL
        assert summary["asymmetry_rate"] == 1.0
        assert summary["mean_abs_gap"] == 2.0
        assert summary["max_abs_gap"] == 2.0
        assert summary["reverse_tighter_rate"] == 1.0


class TestOneSidedGate:
    """A single-role merge leaves the other end at its sentinel."""

    def test_an_infinite_ub_matrix_is_accepted_when_only_lb_was_computed(self) -> None:
        n = 5
        lb = np.full((n, n), 2.0, dtype=np.float64)
        np.fill_diagonal(lb, 0.0)
        ub = np.full((n, n), np.inf, dtype=np.float64)
        np.fill_diagonal(ub, 0.0)
        ged = lb.copy()
        cert = np.zeros((n, n), dtype=np.bool_)
        np.fill_diagonal(cert, True)
        assert gate4(ged, lb, ub, cert, ged_from="lb", computed="lb").passed
        # The default still demands both ends finite.
        assert not gate4(ged, lb, ub, cert, ged_from="lb").passed

    def test_an_unknown_computed_mode_is_refused(self) -> None:
        ged, lb, ub, cert = _clean_matrices()
        with pytest.raises(MergeError, match="computed must be"):
            gate4(ged, lb, ub, cert, computed="lower")
