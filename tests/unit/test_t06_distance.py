"""T-06 distance track: the nine acceptance criteria, plus the §3.1 symbol rule.

Real-data tests read the exported Suite-2 cohort and **synthesise a conforming
CONTRACTS §3 encodings file** from it, because the encoding track's own output
did not exist when these were written.  Building the fixture is the only place
this file touches a cohort; the driver under test never does, which is what
keeps the two ownership sets disjoint.

Per CONTRACTS §1.2, ``import isalgraph`` resolves to the **main checkout**, not
to this worktree.  No test here asserts a numeric value a competitor backend
produced.  Every assertion is an invariant -- symmetry, zero diagonal,
agreement with the pre-existing oracle, agreement between a sharded and an
unsharded run -- or a property of this track's own code.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmarks.eval_distance import distance_merge, distance_runner, masks, size_null
from benchmarks.eval_distance.bands import RowBand, band_for, split_bands, verify_tiling
from benchmarks.eval_distance.gates import assert_dense, check_dense, degenerate_zero_fraction
from benchmarks.eval_distance.schema import (
    DENSE_KEYS,
    METADATA_KEYS,
    MetricUnsupportedError,
    SchemaError,
    ShardError,
    build_metadata,
    load_dense,
    load_shard,
    shard_path,
)

COHORT_ROOT = Path(
    "/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/APPROX_GED"
    "/exported_suite2"
)
UNIT_SEP = distance_runner.UNIT_SEPARATOR

requires_cohort = pytest.mark.skipif(
    not COHORT_ROOT.is_dir(), reason="the exported Suite-2 cohort is not mounted"
)


# --------------------------------------------------------------------------
# Fixture construction: real graphs -> real encodings -> a conforming §3 file
# --------------------------------------------------------------------------


def _load_cohort_graphs(dataset: str, limit: int) -> tuple[list[Any], np.ndarray]:
    """Rebuild up to *limit* networkx graphs from the exported CSR cohort."""
    nx = pytest.importorskip("networkx")
    with np.load(COHORT_ROOT / f"{dataset}.npz", allow_pickle=False) as handle:
        node_counts = handle["n_nodes"]
        offsets = handle["edge_offsets"]
        edges = handle["edges"]
        graph_ids = handle["graph_ids"]
    take = min(limit, int(node_counts.shape[0]))
    graphs = []
    for i in range(take):
        graph = nx.Graph()
        graph.add_nodes_from(range(int(node_counts[i])))
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        graph.add_edges_from(zip(edges[0, lo:hi].tolist(), edges[1, lo:hi].tolist(), strict=True))
        graphs.append(graph)
    return graphs, np.asarray(graph_ids[:take], dtype="<U16")


def _write_encodings_npz(
    path: Path,
    *,
    graph_ids: np.ndarray,
    node_counts: list[int],
    edge_counts: list[int],
    symbols: list[tuple[str, ...]],
    separator: str,
    dataset: str,
    representation: str,
    status: list[str] | None = None,
) -> Path:
    """Write a CONTRACTS §3-conforming encodings file."""
    n = len(symbols)
    states = status if status is not None else ["ok"] * n
    encodings = [separator.join(item) for item in symbols]
    lengths = [len(item) if states[i] != "error" else -1 for i, item in enumerate(symbols)]
    meta = build_metadata(
        suite="suite2",
        dataset=dataset,
        representation=representation,
        metric=None,
        n_graphs=n,
        notes="synthesised by tests/unit/test_t06_distance.py from real cohort graphs",
        extra={"symbol_sep": separator},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        graph_ids=np.asarray(graph_ids, dtype="<U16"),
        node_counts=np.asarray(node_counts, dtype=np.int32),
        edge_counts=np.asarray(edge_counts, dtype=np.int32),
        encoding=np.asarray(encodings, dtype=np.str_),
        length=np.asarray(lengths, dtype=np.int32),
        error_kind=np.asarray(["" for _ in range(n)], dtype="<U32"),
        entropy_bits=np.full(n, np.nan, dtype=np.float64),
        realised_bits=np.full(n, np.nan, dtype=np.float64),
        status=np.asarray(states, dtype="<U12"),
        fallback_used=np.zeros(n, dtype=bool),
        seconds=np.zeros(n, dtype=np.float32),
        metadata=np.array(json.dumps(meta)),
    )
    return path


def _real_encodings(
    tmp_path: Path, dataset: str, representation: str, limit: int = 120
) -> tuple[Path, list[tuple[str, ...]], list[str]]:
    """Encode real cohort graphs and write them as a §3 file.

    Returns:
        ``(path, symbol tuples, Encoding.text renderings)``.
    """
    competitors = pytest.importorskip("isalgraph.competitors")
    graphs, graph_ids = _load_cohort_graphs(dataset, limit)
    backend = competitors.get_backend(representation)
    encoded = [backend.encode(graph) for graph in graphs]
    symbols = [enc.symbols for enc in encoded]
    multi_char = any(len(symbol) != 1 for item in symbols for symbol in item)
    separator = UNIT_SEP if multi_char else ""
    path = _write_encodings_npz(
        tmp_path / f"{dataset}__{representation}.npz",
        graph_ids=graph_ids,
        node_counts=[enc.n_nodes for enc in encoded],
        edge_counts=[enc.n_edges for enc in encoded],
        symbols=symbols,
        separator=separator,
        dataset=dataset,
        representation=representation,
    )
    return path, symbols, [enc.text for enc in encoded]


def _synthetic_encodings(tmp_path: Path, alphabet: str = "abcd", n: int = 24) -> Path:
    """A deterministic §3 file needing neither the cohort nor a backend."""
    rng = np.random.default_rng(42)
    symbols = [
        tuple(rng.choice(list(alphabet), size=int(rng.integers(1, 9))).tolist()) for _ in range(n)
    ]
    return _write_encodings_npz(
        tmp_path / "synthetic__toy.npz",
        graph_ids=np.asarray([f"g{i:04d}" for i in range(n)], dtype="<U16"),
        node_counts=[int(rng.integers(2, 12)) for _ in range(n)],
        edge_counts=[int(rng.integers(1, 20)) for _ in range(n)],
        symbols=symbols,
        separator="",
        dataset="synthetic",
        representation="toy",
    )


def _run(path: Path, metric: str, out: Path, **kwargs: object) -> Path:
    """Invoke the runner in-process."""
    config = distance_runner.RunnerConfig(encodings=path, metric=metric, out_dir=out, **kwargs)
    return distance_runner.run(config)


# --------------------------------------------------------------------------
# Criterion 1 -- schema conformance
# --------------------------------------------------------------------------


def test_dense_file_carries_exactly_the_contract_keys(tmp_path: Path) -> None:
    source = _synthetic_encodings(tmp_path)
    _run(source, "levenshtein", tmp_path / "out")
    dense = tmp_path / "out" / "synthetic__toy__levenshtein.npz"
    with np.load(dense, allow_pickle=False) as handle:
        assert set(handle.files) == set(DENSE_KEYS)
        assert handle["distance_matrix"].dtype == np.float64
        assert handle["defined_mask"].dtype == np.bool_
        assert handle["node_counts"].dtype == np.int32
        assert handle["graph_ids"].dtype.kind == "U"
        assert handle["metadata"].shape == ()
    loaded = load_dense(dense)
    n = loaded.n_graphs
    assert loaded.distance_matrix.shape == (n, n)
    assert loaded.defined_mask.shape == (n, n)


def test_metadata_carries_every_contract_key_including_provenance(tmp_path: Path) -> None:
    source = _synthetic_encodings(tmp_path)
    _run(source, "levenshtein", tmp_path / "out")
    meta = load_dense(tmp_path / "out" / "synthetic__toy__levenshtein.npz").metadata
    assert set(METADATA_KEYS) <= set(meta)
    assert meta["isalgraph_build_hash"] and meta["isalgraph_build_hash"] != "unknown"
    assert meta["src_commit"] and meta["src_commit"] != "unknown"
    assert meta["code_commit"] and meta["code_commit"] != "unknown"
    assert meta["schema_version"] == "t06.1"
    assert meta["ticket"] == "T-06"
    assert meta["seed"] == 42
    assert meta["metric"] == "levenshtein"


@requires_cohort
def test_metadata_conformance_on_a_real_emitted_file(tmp_path: Path) -> None:
    source, _, _ = _real_encodings(tmp_path, "iam_letter_low", "isalgraph_pruned")
    _run(source, "levenshtein", tmp_path / "out")
    dense = tmp_path / "out" / "iam_letter_low__isalgraph_pruned__levenshtein.npz"
    with np.load(dense, allow_pickle=False) as handle:
        assert set(handle.files) == set(DENSE_KEYS)
    meta = load_dense(dense).metadata
    assert set(METADATA_KEYS) <= set(meta)
    assert meta["isalgraph_engine"] == "cpp"
    assert meta["dataset"] == "iam_letter_low"
    assert meta["representation"] == "isalgraph_pruned"


# --------------------------------------------------------------------------
# Criterion 2 -- structural gate
# --------------------------------------------------------------------------


@requires_cohort
@pytest.mark.parametrize("representation", ["isalgraph_pruned", "graph6", "adjacency"])
def test_structural_gate_holds_on_real_matrices(tmp_path: Path, representation: str) -> None:
    source, _, _ = _real_encodings(tmp_path, "iam_letter_low", representation)
    out = tmp_path / f"out_{representation}"
    _run(source, "levenshtein", out)
    loaded = load_dense(out / f"iam_letter_low__{representation}__levenshtein.npz")
    matrix, mask = loaded.distance_matrix, loaded.defined_mask
    assert np.array_equal(matrix, matrix.T)
    assert np.all(np.diagonal(matrix) == 0.0)
    assert np.all(np.isfinite(matrix[mask]))
    assert np.all(matrix[mask] >= 0.0)
    report = check_dense(matrix, mask)
    assert report.passed


@pytest.mark.parametrize(
    ("matrix", "fault"),
    [
        (np.array([[0.0, 1.0], [2.0, 0.0]]), "not symmetric"),
        (np.array([[1.0, 1.0], [1.0, 1.0]]), "diagonal"),
        (np.array([[0.0, np.inf], [np.inf, 0.0]]), "non-finite"),
        (np.array([[0.0, -1.0], [-1.0, 0.0]]), "negative"),
    ],
)
def test_structural_gate_rejects_each_fault(matrix: np.ndarray, fault: str) -> None:
    mask = np.ones(matrix.shape, dtype=bool)
    with pytest.raises(SchemaError, match=fault):
        assert_dense(matrix, mask)


def test_degeneracy_guard_fires_only_on_an_all_zero_matrix() -> None:
    zeros = np.zeros((10, 10))
    ones = np.ones((10, 10)) - np.eye(10)
    mask = np.ones((10, 10), dtype=bool)
    with pytest.raises(SchemaError, match="degeneracy threshold"):
        degenerate_zero_fraction(check_dense(zeros, mask))
    degenerate_zero_fraction(check_dense(ones, mask))


# --------------------------------------------------------------------------
# Criterion 3 -- differential against the pre-existing oracle
# --------------------------------------------------------------------------


@requires_cohort
@pytest.mark.parametrize("representation", ["isalgraph_pruned", "graph6"])
def test_levenshtein_matches_the_existing_oracle_exactly(
    tmp_path: Path, representation: str
) -> None:
    from benchmarks.eval_setup.levenshtein_computer import compute_levenshtein_matrix

    source, _, texts = _real_encodings(tmp_path, "iam_letter_low", representation, limit=200)
    out = tmp_path / "out"
    _run(source, "levenshtein", out)
    mine = load_dense(out / f"iam_letter_low__{representation}__levenshtein.npz")
    oracle = compute_levenshtein_matrix(
        list(texts), [str(g) for g in mine.graph_ids], method="oracle"
    )
    assert mine.defined_mask.all()
    assert np.array_equal(mine.distance_matrix.astype(np.int64), oracle.astype(np.int64))


# --------------------------------------------------------------------------
# Criterion 4 -- sharding is exact
# --------------------------------------------------------------------------


@requires_cohort
def test_seven_shards_merge_to_the_unsharded_matrix(tmp_path: Path) -> None:
    source, _, _ = _real_encodings(tmp_path, "iam_letter_low", "isalgraph_pruned", limit=150)
    single = tmp_path / "single"
    _run(source, "levenshtein", single)
    sharded = tmp_path / "sharded"
    for k in range(7):
        _run(source, "levenshtein", sharded, chunk_index=k, n_chunks=7)
    basename = "iam_letter_low__isalgraph_pruned__levenshtein"
    distance_merge.run(sharded, basename, sharded)
    a = load_dense(single / f"{basename}.npz")
    b = load_dense(sharded / f"{basename}.npz")
    assert np.array_equal(a.distance_matrix, b.distance_matrix)
    assert np.array_equal(a.defined_mask, b.defined_mask)
    assert np.array_equal(a.graph_ids, b.graph_ids)


def test_row_bands_tile_the_interval_for_every_chunk_count() -> None:
    for n_graphs in (0, 1, 5, 150, 4041):
        for n_chunks in (1, 2, 7, 64):
            bands = split_bands(n_graphs, n_chunks)
            assert len(bands) == n_chunks
            verify_tiling(bands, n_graphs)
            assert sum(band.height for band in bands) == n_graphs
            assert max(band.height for band in bands) - min(band.height for band in bands) <= 1


def test_verify_tiling_names_a_gap_and_an_overlap() -> None:
    with pytest.raises(ShardError, match="gap"):
        verify_tiling([RowBand(0, 0, 3), RowBand(1, 5, 10)], 10)
    with pytest.raises(ShardError, match="overlap"):
        verify_tiling([RowBand(0, 0, 6), RowBand(1, 3, 10)], 10)
    with pytest.raises(ShardError, match="missing"):
        verify_tiling([RowBand(0, 0, 6)], 10)
    with pytest.raises(ShardError, match="outside"):
        band_for(10, 3, 3)


# --------------------------------------------------------------------------
# Criterion 5 -- the merge refuses partial input
# --------------------------------------------------------------------------


def test_merge_raises_when_a_shard_is_missing(tmp_path: Path) -> None:
    source = _synthetic_encodings(tmp_path)
    sharded = tmp_path / "sharded"
    for k in range(5):
        _run(source, "levenshtein", sharded, chunk_index=k, n_chunks=5)
    basename = "synthetic__toy__levenshtein"
    shard_path(sharded, basename, 2).unlink()
    with pytest.raises(ShardError, match="incomplete"):
        distance_merge.run(sharded, basename, tmp_path / "merged")
    assert not (tmp_path / "merged" / f"{basename}.npz").exists()


def test_merge_raises_on_an_empty_shard_directory(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(ShardError, match="no shard"):
        distance_merge.run(tmp_path / "empty", "nothing__here__levenshtein", tmp_path)


def test_merge_cli_returns_nonzero_on_a_missing_shard(tmp_path: Path) -> None:
    source = _synthetic_encodings(tmp_path)
    sharded = tmp_path / "sharded"
    for k in range(3):
        _run(source, "levenshtein", sharded, chunk_index=k, n_chunks=3)
    shard_path(sharded, "synthetic__toy__levenshtein", 1).unlink()
    code = distance_merge.main(
        [
            "--shard-dir",
            str(sharded),
            "--basename",
            "synthetic__toy__levenshtein",
            "--out",
            str(tmp_path / "merged"),
        ]
    )
    assert code == 1


def test_merge_rejects_shards_from_two_different_matrices(tmp_path: Path) -> None:
    source = _synthetic_encodings(tmp_path)
    sharded = tmp_path / "sharded"
    _run(source, "levenshtein", sharded, chunk_index=0, n_chunks=2)
    _run(source, "levenshtein", sharded, chunk_index=1, n_chunks=2)
    shard = load_shard(shard_path(sharded, "synthetic__toy__levenshtein", 1))
    tampered = dict(shard.metadata)
    tampered["dataset"] = "somewhere_else"
    np.savez_compressed(
        shard_path(sharded, "synthetic__toy__levenshtein", 1),
        distance_band=shard.distance_band,
        defined_band=shard.defined_band,
        row_start=np.asarray(shard.row_start, dtype=np.int64),
        row_stop=np.asarray(shard.row_stop, dtype=np.int64),
        n_graphs=np.asarray(shard.n_graphs, dtype=np.int64),
        graph_ids=shard.graph_ids,
        node_counts=shard.node_counts,
        metadata=np.array(json.dumps(tampered)),
    )
    with pytest.raises(ShardError, match="different matrices"):
        distance_merge.run(sharded, "synthetic__toy__levenshtein", tmp_path / "merged")


# --------------------------------------------------------------------------
# Criterion 6 -- the size null
# --------------------------------------------------------------------------


@requires_cohort
def test_size_null_is_the_absolute_node_count_difference(tmp_path: Path) -> None:
    source, _, _ = _real_encodings(tmp_path, "iam_letter_low", "isalgraph_pruned")
    out = size_null.run(source, tmp_path / "out")
    assert out.name == "iam_letter_low__size_null.npz"
    loaded = load_dense(out)
    counts = loaded.node_counts.astype(np.int64)
    expected = np.abs(counts[:, None] - counts[None, :]).astype(np.float64)
    assert np.array_equal(loaded.distance_matrix, expected)
    assert np.array_equal(loaded.distance_matrix, loaded.distance_matrix.T)
    assert np.all(np.diagonal(loaded.distance_matrix) == 0.0)
    assert loaded.defined_mask.all()


def test_size_null_file_has_the_same_schema_as_any_distance_file(tmp_path: Path) -> None:
    source = _synthetic_encodings(tmp_path)
    out = size_null.run(source, tmp_path / "out")
    with np.load(out, allow_pickle=False) as handle:
        assert set(handle.files) == set(DENSE_KEYS)
    meta = load_dense(out).metadata
    assert set(METADATA_KEYS) <= set(meta)
    assert meta["representation"] == "size_null"


# --------------------------------------------------------------------------
# Criterion 7 -- undefined pairs are nan and masked, never 0.0
# --------------------------------------------------------------------------


@requires_cohort
def test_hamming_on_unequal_lengths_is_nan_and_unmasked(tmp_path: Path) -> None:
    source, symbols, _ = _real_encodings(tmp_path, "iam_letter_low", "graph6", limit=150)
    lengths = np.array([len(item) for item in symbols])
    assert len(set(lengths.tolist())) > 1, "fixture must contain unequal lengths"
    out = tmp_path / "out"
    _run(source, "hamming", out)
    loaded = load_dense(out / "iam_letter_low__graph6__hamming.npz")
    expected = lengths[:, None] == lengths[None, :]
    assert np.array_equal(loaded.defined_mask, expected)
    undefined = ~loaded.defined_mask
    assert undefined.any()
    assert np.all(np.isnan(loaded.distance_matrix[undefined]))
    assert not np.any(loaded.distance_matrix[undefined] == 0.0)


def test_a_row_with_status_error_is_undefined_off_the_diagonal(tmp_path: Path) -> None:
    n = 8
    symbols = [tuple("abc"[: 1 + i % 3]) for i in range(n)]
    status = ["ok"] * n
    status[3] = "error"
    symbols[3] = ()
    path = _write_encodings_npz(
        tmp_path / "toy__rep.npz",
        graph_ids=np.asarray([f"g{i}" for i in range(n)], dtype="<U16"),
        node_counts=[3] * n,
        edge_counts=[2] * n,
        symbols=symbols,
        separator="",
        dataset="toy",
        representation="rep",
        status=status,
    )
    _run(path, "levenshtein", tmp_path / "out")
    loaded = load_dense(tmp_path / "out" / "toy__rep__levenshtein.npz")
    off = ~np.eye(n, dtype=bool)
    assert not loaded.defined_mask[3][off[3]].any()
    assert np.all(np.isnan(loaded.distance_matrix[3][off[3]]))
    assert loaded.defined_mask[3, 3]
    assert loaded.distance_matrix[3, 3] == 0.0
    assert loaded.defined_mask[np.ix_([0, 1, 2], [0, 1, 2])].all()


def test_a_metric_needing_a_frame_or_features_is_refused(tmp_path: Path) -> None:
    source = _synthetic_encodings(tmp_path)
    for metric in ("padded_hamming", "kernel"):
        with pytest.raises(MetricUnsupportedError, match="does not carry"):
            _run(source, metric, tmp_path / "out")


# --------------------------------------------------------------------------
# CONTRACTS §3.1 -- symbols, not characters
# --------------------------------------------------------------------------


@requires_cohort
def test_min_dfs_symbol_level_differs_from_character_level(tmp_path: Path) -> None:
    """The §3.1 regression guard: one deleted DFS tuple is one edit, not four."""
    rapidfuzz = pytest.importorskip("rapidfuzz")
    source, symbols, texts = _real_encodings(tmp_path, "iam_letter_low", "min_dfs", limit=80)
    assert any(len(symbol) != 1 for item in symbols for symbol in item)
    out = tmp_path / "out"
    _run(source, "levenshtein", out)
    loaded = load_dense(out / "iam_letter_low__min_dfs__levenshtein.npz")

    scorer = rapidfuzz.distance.Levenshtein.distance
    n = len(symbols)
    reference = np.array(
        [[float(scorer(list(symbols[i]), list(symbols[j]))) for j in range(n)] for i in range(n)]
    )
    assert np.array_equal(loaded.distance_matrix, reference)

    char_level = np.array([[float(scorer(texts[i], texts[j])) for j in range(n)] for i in range(n)])
    assert np.any(char_level != reference), "symbol and character levels must differ on min_dfs"
    differing = np.count_nonzero(char_level != reference)
    assert differing > 0
    assert np.all(char_level >= reference)


@requires_cohort
def test_character_level_metric_is_refused_when_symbols_are_joined(tmp_path: Path) -> None:
    source, _, _ = _real_encodings(tmp_path, "iam_letter_low", "min_dfs", limit=40)
    with pytest.raises(MetricUnsupportedError, match="not the text"):
        _run(source, "levenshtein_char", tmp_path / "out")


def test_symbol_separator_resolution_order() -> None:
    resolve = distance_runner.resolve_symbol_separator
    assert resolve("min_dfs", {"symbol_sep": ""}, None) == ""
    assert resolve("min_dfs", {}, None) == UNIT_SEP
    assert resolve("graph6", {}, None) == ""
    assert resolve("graph6", {"symbol_sep": UNIT_SEP}, None) == UNIT_SEP
    assert resolve("min_dfs", {"symbol_sep": UNIT_SEP}, " ") == " "


def test_a_wrong_separator_raises_rather_than_producing_a_plausible_number(
    tmp_path: Path,
) -> None:
    symbols = [("0-1", "1-2"), ("0-1", "1-2", "2-3")]
    path = _write_encodings_npz(
        tmp_path / "toy__min_dfs.npz",
        graph_ids=np.asarray(["g0", "g1"], dtype="<U16"),
        node_counts=[3, 4],
        edge_counts=[2, 3],
        symbols=symbols,
        separator=UNIT_SEP,
        dataset="toy",
        representation="min_dfs",
    )
    with pytest.raises(SchemaError, match="separator is wrong"):
        _run(path, "levenshtein", tmp_path / "out", symbol_sep="")
    _run(path, "levenshtein", tmp_path / "warned", symbol_sep="", on_length_mismatch="warn")
    meta = load_dense(tmp_path / "warned" / "toy__min_dfs__levenshtein.npz").metadata
    assert meta["symbol_length_matches_npz_length"] is False


# --------------------------------------------------------------------------
# Criterion 8 -- small end to end on real data, through the CLI
# --------------------------------------------------------------------------


@requires_cohort
def test_end_to_end_through_the_cli_on_real_data(tmp_path: Path) -> None:
    source, _, _ = _real_encodings(tmp_path, "iam_letter_low", "isalgraph_pruned", limit=200)
    out = tmp_path / "out"
    repo = Path(__file__).resolve().parents[2]
    for k in range(3):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.eval_distance.distance_runner",
                "--encodings",
                str(source),
                "--metric",
                "levenshtein",
                "--out",
                str(out),
                "--chunk-index",
                str(k),
                "--n-chunks",
                "3",
            ],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
    basename = "iam_letter_low__isalgraph_pruned__levenshtein"
    merged = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.eval_distance.distance_merge",
            "--shard-dir",
            str(out),
            "--basename",
            basename,
            "--out",
            str(out),
            "--expect-chunks",
            "3",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert merged.returncode == 0, merged.stderr[-2000:]
    loaded = load_dense(out / f"{basename}.npz")
    assert loaded.n_graphs == 200
    assert check_dense(loaded.distance_matrix, loaded.defined_mask).passed
    assert loaded.metadata["isalgraph_engine"] == "cpp"
    null = size_null.run(source, out)
    assert load_dense(null).n_graphs == loaded.n_graphs


# --------------------------------------------------------------------------
# masks.py -- the helpers the statistics track consumes
# --------------------------------------------------------------------------


def test_equal_n_mask_is_computed_and_symmetric() -> None:
    counts = np.array([2, 3, 3, 5])
    mask = masks.equal_n_mask(counts)
    assert mask.dtype == np.bool_
    assert np.array_equal(mask, mask.T)
    assert mask.diagonal().all()
    assert mask[1, 2] and not mask[0, 1]


def test_upper_triangle_is_strict_and_indexable() -> None:
    matrix = np.arange(16, dtype=np.float64).reshape(4, 4)
    values, rows, cols = masks.upper_triangle(matrix)
    assert values.size == 6
    assert np.all(rows < cols)
    assert np.array_equal(values, matrix[rows, cols])
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 1] = mask[2, 3] = True
    values, rows, cols = masks.upper_triangle(matrix, mask=mask)
    assert values.size == 2
    assert set(zip(rows.tolist(), cols.tolist(), strict=True)) == {(0, 1), (2, 3)}


def test_paired_upper_triangle_reads_identical_positions() -> None:
    left = np.arange(9, dtype=np.float64).reshape(3, 3)
    right = left * 10.0
    lv, rv, rows, cols = masks.paired_upper_triangle(left, right)
    assert np.array_equal(rv, lv * 10.0)
    assert np.all(rows < cols)
    with pytest.raises(ValueError, match="differ in shape"):
        masks.paired_upper_triangle(left, np.zeros((4, 4)))


def test_upper_triangle_rejects_a_non_square_matrix() -> None:
    with pytest.raises(ValueError, match="expected square"):
        masks.upper_triangle(np.zeros((2, 3)))
