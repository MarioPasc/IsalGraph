"""Unit tests for the ``ubt`` subsample join.

The failure this file exists to prevent is a *silent partial join*. The output
is a flat list whose length depends on realised bin populations, so nothing
downstream knows how many rows it should have; a join that quietly dropped a
dataset would produce a shorter file in which every row was individually
correct. Hence the two-directional exactness check, and hence most of the tests
below.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.eval_setup.approx_ged_subsample_merge import (
    SUBSAMPLE_KEYS,
    SubsampleMergeError,
    collect_subsample_shards,
    main,
    merge_subsample,
)
from benchmarks.eval_setup.ged_pair_index import index_of_pair

DET = "--threads 1 --randomness PSEUDO --initial-solutions 10"


def _write_shard(
    path: Path,
    *,
    key: str,
    n_graphs: int,
    pairs: list[tuple[int, int]],
    values: list[float],
) -> None:
    """Write a CONTRACT C shard for one dataset's slice of the subsample.

    Args:
        path: Destination ``.npz``.
        key: Dataset key.
        n_graphs: Cohort size, needed to invert the linear index.
        pairs: ``(i, j)`` graph-index pairs.
        values: One upper bound per pair.
    """
    idx = np.array([index_of_pair(i, j, n_graphs) for i, j in pairs], dtype=np.int64)
    m = len(pairs)
    np.savez_compressed(
        path,
        pair_index=idx,
        ged=np.full(m, np.inf, dtype=np.float64),
        lb=np.full(m, -np.inf, dtype=np.float64),
        ub=np.asarray(values, dtype=np.float64),
        certified=np.zeros(m, dtype=np.bool_),
        seconds=np.full(m, 0.5, dtype=np.float32),
        meta=np.array(
            json.dumps(
                {
                    "dataset": key,
                    "n_graphs": n_graphs,
                    "cost_model": "unit",
                    "compute": "ub",
                    "ub_method": "IPFP",
                    "ub_options": DET,
                    "role": "ubt",
                }
            )
        ),
    )


def _write_pair_list(path: Path, rows: list[tuple[str, int, int, int, int]]) -> None:
    """Write a §5 subsample pair list.

    Args:
        path: Destination ``.npz``.
        rows: ``(dataset_key, i, j, n_max, bin_index)`` per row, in sample order.
    """
    np.savez_compressed(
        path,
        dataset_key=np.array([r[0] for r in rows], dtype=str),
        pair_i=np.array([r[1] for r in rows], dtype=np.int32),
        pair_j=np.array([r[2] for r in rows], dtype=np.int32),
        n_max=np.array([r[3] for r in rows], dtype=np.int32),
        bin_index=np.array([r[4] for r in rows], dtype=np.int8),
        metadata=np.array(json.dumps({"bin_edges": [2, 4, 6, 8], "seed": 42, "n_per_bin": 2000})),
    )


@pytest.fixture()
def campaign(tmp_path: Path) -> tuple[Path, Path]:
    """Two datasets' shards plus the pair list that names exactly their pairs."""
    shards = tmp_path / "shards"
    shards.mkdir()
    _write_shard(
        shards / "linux_c0000.npz",
        key="linux",
        n_graphs=6,
        pairs=[(0, 1), (2, 3)],
        values=[5.0, 7.0],
    )
    _write_shard(
        shards / "grec_c0000.npz",
        key="grec",
        n_graphs=5,
        pairs=[(1, 4)],
        values=[9.0],
    )
    listed = tmp_path / "subsample_pairs.npz"
    # Deliberately not grouped by dataset: row order is the sampler's, and the
    # output must preserve it exactly.
    _write_pair_list(
        listed,
        [("grec", 1, 4, 7, 2), ("linux", 0, 1, 5, 1), ("linux", 2, 3, 9, 3)],
    )
    return shards, listed


class TestJoin:
    """The happy path, and the row order that makes it reproducible."""

    def test_the_output_carries_the_ten_flat_keys(
        self, campaign: tuple[Path, Path], tmp_path: Path
    ) -> None:
        shards, listed = campaign
        out = tmp_path / "subsample.npz"
        merge_subsample(shard_dir=shards, pair_list=listed, out=out)
        with np.load(out, allow_pickle=False) as data:
            assert set(data.files) == set(SUBSAMPLE_KEYS)
            assert data["pair_i"].dtype == np.int32
            assert data["pair_j"].dtype == np.int32
            assert data["n_max"].dtype == np.int32
            assert data["bin_index"].dtype == np.int8
            assert data["value"].dtype == np.float64
            assert data["seconds"].dtype == np.float32

    def test_row_order_is_the_pair_lists(self, campaign: tuple[Path, Path], tmp_path: Path) -> None:
        """Reproducible from seed 42 alone means the sampler's order survives."""
        shards, listed = campaign
        out = tmp_path / "subsample.npz"
        merge_subsample(shard_dir=shards, pair_list=listed, out=out)
        with np.load(out, allow_pickle=False) as data:
            assert np.asarray(data["dataset_key"]).astype(str).tolist() == [
                "grec",
                "linux",
                "linux",
            ]
            assert data["value"].tolist() == [9.0, 5.0, 7.0]
            assert data["n_max"].tolist() == [7, 5, 9]
            assert data["bin_index"].tolist() == [2, 1, 3]

    def test_no_dense_matrix_is_written(self, campaign: tuple[Path, Path], tmp_path: Path) -> None:
        shards, listed = campaign
        out = tmp_path / "subsample.npz"
        merge_subsample(shard_dir=shards, pair_list=listed, out=out)
        with np.load(out, allow_pickle=False) as data:
            for name in data.files:
                assert np.asarray(data[name]).ndim <= 1, name

    def test_the_metadata_records_the_specification_and_the_strata(
        self, campaign: tuple[Path, Path], tmp_path: Path
    ) -> None:
        shards, listed = campaign
        out = tmp_path / "subsample.npz"
        meta = merge_subsample(shard_dir=shards, pair_list=listed, out=out, role="ubt")
        assert meta["role"] == "ubt"
        assert meta["method"] == "IPFP"
        assert meta["options_string"] == DET
        assert meta["accessor"] == "upper"
        assert meta["cost_model"] == "unit"
        assert meta["bin_edges"] == [2, 4, 6, 8]
        assert meta["seed"] == 42
        assert meta["realised_per_bin"] == {"1": 1, "2": 1, "3": 1}
        assert meta["n_pairs"] == 3
        assert meta["n_datasets"] == 2
        assert meta["schema_version"] == 1
        assert "code_commit" in meta and "computed_utc" in meta

    def test_an_explicit_method_overrides_the_shards(
        self, campaign: tuple[Path, Path], tmp_path: Path
    ) -> None:
        shards, listed = campaign
        meta = merge_subsample(
            shard_dir=shards,
            pair_list=listed,
            out=tmp_path / "s.npz",
            method="REFINE",
            options="--threads 1",
        )
        assert meta["method"] == "REFINE"
        assert meta["options_string"] == "--threads 1"

    def test_the_orientation_columns_are_nan_and_the_metadata_says_why(
        self, campaign: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """CONTRACT C carries only the symmetrised ub; fabricating the two halves would lie."""
        shards, listed = campaign
        out = tmp_path / "s.npz"
        meta = merge_subsample(shard_dir=shards, pair_list=listed, out=out)
        with np.load(out, allow_pickle=False) as data:
            assert np.isnan(data["value_fwd"]).all()
            assert np.isnan(data["value_rev"]).all()
        assert "not retained" in meta["orientation_detail"]


class TestJoinExactness:
    """A partial join is undetectable downstream, so it is a hard failure."""

    def test_a_listed_pair_with_no_computed_value_is_refused(
        self, campaign: tuple[Path, Path], tmp_path: Path
    ) -> None:
        shards, listed = campaign
        _write_pair_list(
            listed,
            [
                ("grec", 1, 4, 7, 2),
                ("linux", 0, 1, 5, 1),
                ("linux", 2, 3, 9, 3),
                ("linux", 0, 5, 6, 1),
            ],
        )
        with pytest.raises(SubsampleMergeError, match="1 listed pairs have no computed value"):
            merge_subsample(shard_dir=shards, pair_list=listed, out=tmp_path / "s.npz")

    def test_a_computed_pair_absent_from_the_list_is_refused(
        self, campaign: tuple[Path, Path], tmp_path: Path
    ) -> None:
        shards, listed = campaign
        _write_pair_list(listed, [("grec", 1, 4, 7, 2), ("linux", 0, 1, 5, 1)])
        with pytest.raises(SubsampleMergeError, match="1 computed pairs are not on the list"):
            merge_subsample(shard_dir=shards, pair_list=listed, out=tmp_path / "s.npz")

    def test_a_whole_missing_dataset_is_refused(
        self, campaign: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """The shape of a shard directory that a failed array task left short."""
        shards, listed = campaign
        (shards / "grec_c0000.npz").unlink()
        with pytest.raises(SubsampleMergeError, match="the join is not exact"):
            merge_subsample(shard_dir=shards, pair_list=listed, out=tmp_path / "s.npz")

    def test_nothing_is_written_when_the_join_is_refused(
        self, campaign: tuple[Path, Path], tmp_path: Path
    ) -> None:
        shards, listed = campaign
        (shards / "grec_c0000.npz").unlink()
        out = tmp_path / "s.npz"
        with pytest.raises(SubsampleMergeError):
            merge_subsample(shard_dir=shards, pair_list=listed, out=out)
        assert not out.exists()


class TestRefusals:
    """Everything else that must stop rather than produce a plausible file."""

    def test_a_shard_without_its_cohort_size_is_refused(self, tmp_path: Path) -> None:
        """A linear pair index means nothing without the cohort it indexes."""
        shards = tmp_path / "s"
        shards.mkdir()
        np.savez_compressed(
            shards / "x_c0000.npz",
            pair_index=np.array([0], dtype=np.int64),
            ged=np.array([np.inf]),
            lb=np.array([-np.inf]),
            ub=np.array([3.0]),
            certified=np.array([False]),
            seconds=np.array([0.1], dtype=np.float32),
            meta=np.array(json.dumps({"dataset": "linux"})),
        )
        listed = tmp_path / "p.npz"
        _write_pair_list(listed, [("linux", 0, 1, 5, 1)])
        with pytest.raises(SubsampleMergeError, match="n_graphs"):
            merge_subsample(shard_dir=shards, pair_list=listed, out=tmp_path / "o.npz")

    def test_an_infinite_value_is_refused(self, tmp_path: Path) -> None:
        """inf is the signature of a method that does not set the upper end."""
        shards = tmp_path / "s"
        shards.mkdir()
        _write_shard(
            shards / "linux_c0000.npz", key="linux", n_graphs=6, pairs=[(0, 1)], values=[np.inf]
        )
        listed = tmp_path / "p.npz"
        _write_pair_list(listed, [("linux", 0, 1, 5, 1)])
        with pytest.raises(SubsampleMergeError, match="not finite"):
            merge_subsample(shard_dir=shards, pair_list=listed, out=tmp_path / "o.npz")

    def test_an_empty_shard_directory_is_refused(self, tmp_path: Path) -> None:
        shards = tmp_path / "s"
        shards.mkdir()
        listed = tmp_path / "p.npz"
        _write_pair_list(listed, [("linux", 0, 1, 5, 1)])
        with pytest.raises(SubsampleMergeError, match="no CONTRACT C shards"):
            merge_subsample(shard_dir=shards, pair_list=listed, out=tmp_path / "o.npz")

    def test_a_missing_pair_list_is_refused(self, campaign: tuple[Path, Path]) -> None:
        shards, listed = campaign
        with pytest.raises(SubsampleMergeError, match="does not exist"):
            merge_subsample(
                shard_dir=shards,
                pair_list=listed.parent / "absent.npz",
                out=listed.parent / "o.npz",
            )

    def test_a_file_that_is_not_a_pair_list_is_refused(
        self, campaign: tuple[Path, Path], tmp_path: Path
    ) -> None:
        shards, _listed = campaign
        junk = tmp_path / "junk.npz"
        np.savez_compressed(junk, something=np.zeros(3))
        with pytest.raises(SubsampleMergeError, match="missing 'dataset_key'"):
            merge_subsample(shard_dir=shards, pair_list=junk, out=tmp_path / "o.npz")

    def test_checkpoints_are_not_mistaken_for_shards(
        self, campaign: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """A checkpoint is legitimately shard-shaped; joining one would double-count."""
        shards, listed = campaign
        _write_shard(
            shards / "linux_c0000.ckpt.npz",
            key="linux",
            n_graphs=6,
            pairs=[(0, 1)],
            values=[99.0],
        )
        assert all(not p.name.endswith(".ckpt.npz") for p in collect_subsample_shards(shards))
        out = tmp_path / "s.npz"
        merge_subsample(shard_dir=shards, pair_list=listed, out=out)
        with np.load(out, allow_pickle=False) as data:
            assert 99.0 not in data["value"].tolist()


class TestCli:
    """The entry point the launcher calls."""

    def test_the_cli_writes_the_file_and_returns_zero(
        self, campaign: tuple[Path, Path], tmp_path: Path
    ) -> None:
        shards, listed = campaign
        out = tmp_path / "subsample.npz"
        code = main(
            [
                "--shards",
                str(shards),
                "--pair-list",
                str(listed),
                "--out",
                str(out),
                "--role",
                "ubt",
                "--method",
                "IPFP",
                "--options",
                DET,
                "--log-level",
                "WARNING",
            ]
        )
        assert code == 0
        with np.load(out, allow_pickle=False) as data:
            assert data["value"].size == 3

    def test_the_cli_returns_one_on_an_incomplete_join(
        self, campaign: tuple[Path, Path], tmp_path: Path
    ) -> None:
        shards, listed = campaign
        (shards / "grec_c0000.npz").unlink()
        code = main(
            [
                "--shards",
                str(shards),
                "--pair-list",
                str(listed),
                "--out",
                str(tmp_path / "s.npz"),
                "--log-level",
                "CRITICAL",
            ]
        )
        assert code == 1
