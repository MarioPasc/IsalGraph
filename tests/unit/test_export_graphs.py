"""Unit tests for export_graphs -- CONTRACT A, the exported dataset file.

Two agents consume this format from fabricated fixtures, so the invariants
tested here are the ones they rely on: the CSR offsets, the ``u < v`` edge
orientation, the metadata schema, and round-trip identity edge-for-edge. The
cohort assertions are tested separately because a wrong count must stop the
pipeline rather than propagate into a GED matrix.

The real-data tests are marked ``integration`` and skip when the Sandisk source
tree is absent, so the suite passes on a machine without it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from collections.abc import Callable
from math import comb
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from benchmarks.eval_setup.export_graphs import (
    DATASETS,
    DEFAULT_EXPORT_DIR,
    DEFAULT_SOURCE_DIR,
    MANIFEST_NAME,
    SCHEMA_VERSION,
    TOTAL_EXPECTED_GRAPHS,
    TOTAL_EXPECTED_PAIRS,
    CohortMismatchError,
    DatasetSpec,
    ExportedDataset,
    ExportError,
    assert_cohort,
    build_parser,
    content_sha256,
    export_all,
    load_exported,
    main,
    read_manifest,
    save_exported,
    verify_exports,
    write_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

_SOURCE_PRESENT = Path(DEFAULT_SOURCE_DIR).is_dir()
requires_source = pytest.mark.skipif(
    not _SOURCE_PRESENT, reason=f"source tree absent: {DEFAULT_SOURCE_DIR}"
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _metadata(key: str, n_kept: int, **overrides: object) -> dict[str, object]:
    """Return a CONTRACTS section 4 metadata block."""
    meta: dict[str, object] = {
        "dataset": key,
        "source": "graphedx",
        "n_raw": n_kept + 3,
        "n_kept": n_kept,
        "n_dropped_size": 1,
        "n_dropped_disconnected": 1,
        "n_dropped_trivial": 1,
        "n_pairs": comb(n_kept, 2),
        "filter": {"min_nodes": 2, "require_connected": True, "n_max": 12},
        "splits_merged": True,
        "exported_utc": "2026-08-12T00:00:00Z",
        "code_commit": "0" * 40,
        "schema_version": SCHEMA_VERSION,
    }
    meta.update(overrides)
    return meta


def _dataset(
    graphs: list[nx.Graph],
    key: str = "fixture",
    *,
    graph_ids: list[str] | None = None,
    splits: list[str] | None = None,
    labels: list[str] | None = None,
    n_nodes: np.ndarray | None = None,
    n_edges: np.ndarray | None = None,
) -> ExportedDataset:
    """Build an ExportedDataset with arrays derived from ``graphs``."""
    n = len(graphs)
    return ExportedDataset(
        key=key,
        graphs=graphs,
        graph_ids=graph_ids if graph_ids is not None else [f"g{i:03d}" for i in range(n)],
        splits=splits if splits is not None else ["train"] * n,
        labels=labels if labels is not None else [""] * n,
        n_nodes=(
            n_nodes
            if n_nodes is not None
            else np.asarray([g.number_of_nodes() for g in graphs], dtype=np.int32)
        ),
        n_edges=(
            n_edges
            if n_edges is not None
            else np.asarray([g.number_of_edges() for g in graphs], dtype=np.int32)
        ),
        metadata=_metadata(key, n),
    )


def _sample_graphs() -> list[nx.Graph]:
    """A mix covering empty-edge, single-edge, path, cycle and complete graphs."""
    empty2 = nx.Graph()
    empty2.add_nodes_from(range(2))  # 2 nodes, no edges
    single = nx.Graph()
    single.add_nodes_from(range(2))
    single.add_edge(0, 1)  # exactly one edge
    return [empty2, single, nx.path_graph(4), nx.cycle_graph(5), nx.complete_graph(4)]


# --------------------------------------------------------------------------- #
# Round-trip identity
# --------------------------------------------------------------------------- #


def test_round_trip_reproduces_every_graph_edge_for_edge(tmp_path: Path) -> None:
    """save then load reproduces graphs, ids, splits and labels exactly."""
    graphs = _sample_graphs()
    original = _dataset(
        graphs,
        key="rt",
        graph_ids=[f"rt_{i}" for i in range(len(graphs))],
        splits=["train", "val", "test", "train", "val"],
        labels=["A", "B", "", "A", "Z"],
    )
    path = tmp_path / "rt.npz"
    save_exported(original, path)
    loaded = load_exported(path)

    assert loaded.key == "rt"
    assert loaded.graph_ids == original.graph_ids
    assert loaded.splits == original.splits
    assert loaded.labels == original.labels
    np.testing.assert_array_equal(loaded.n_nodes, original.n_nodes)
    np.testing.assert_array_equal(loaded.n_edges, original.n_edges)

    assert len(loaded.graphs) == len(graphs)
    for got, want in zip(loaded.graphs, graphs, strict=True):
        assert sorted(got.nodes()) == sorted(want.nodes())
        assert {frozenset(e) for e in got.edges()} == {frozenset(e) for e in want.edges()}


def test_round_trip_is_idempotent(tmp_path: Path) -> None:
    """A reloaded dataset re-saves to identical content."""
    original = _dataset(_sample_graphs(), key="idem")
    first, second = tmp_path / "a.npz", tmp_path / "b.npz"
    save_exported(original, first)
    reloaded = load_exported(first)
    save_exported(reloaded, second)
    assert content_sha256(reloaded) == content_sha256(load_exported(second))


def test_empty_dataset_round_trips(tmp_path: Path) -> None:
    """Zero graphs must not collapse the string dtypes to float64."""
    path = tmp_path / "empty.npz"
    save_exported(_dataset([], key="empty"), path)
    loaded = load_exported(path)
    assert loaded.graphs == []
    assert loaded.graph_ids == []
    with np.load(path) as handle:
        assert handle["graph_ids"].dtype.kind == "U"
        assert handle["edge_offsets"].tolist() == [0]
        assert handle["edges"].shape == (2, 0)


def test_single_edge_and_empty_edge_graphs(tmp_path: Path) -> None:
    """A 2-node edgeless graph and a 2-node single-edge graph survive intact."""
    edgeless = nx.Graph()
    edgeless.add_nodes_from(range(2))
    one = nx.Graph()
    one.add_nodes_from(range(2))
    one.add_edge(0, 1)

    path = tmp_path / "tiny.npz"
    save_exported(_dataset([edgeless, one], key="tiny"), path)
    loaded = load_exported(path)

    assert loaded.n_nodes.tolist() == [2, 2]
    assert loaded.n_edges.tolist() == [0, 1]
    assert loaded.graphs[0].number_of_edges() == 0
    assert list(loaded.graphs[1].edges()) == [(0, 1)]


# --------------------------------------------------------------------------- #
# CSR and edge invariants
# --------------------------------------------------------------------------- #


def test_csr_offset_invariants(tmp_path: Path) -> None:
    """edge_offsets[0] == 0, [-1] == n_edges.sum(), monotone non-decreasing."""
    path = tmp_path / "csr.npz"
    save_exported(_dataset(_sample_graphs(), key="csr"), path)
    with np.load(path) as handle:
        offsets = handle["edge_offsets"]
        n_edges = handle["n_edges"]
        edges = handle["edges"]
        n_nodes = handle["n_nodes"]

    assert offsets.dtype == np.int64
    assert n_edges.dtype == np.int32
    assert n_nodes.dtype == np.int32
    assert edges.dtype == np.int32
    assert offsets.shape == (len(n_edges) + 1,)
    assert int(offsets[0]) == 0
    assert int(offsets[-1]) == int(n_edges.sum())
    assert np.all(np.diff(offsets) >= 0)
    np.testing.assert_array_equal(np.diff(offsets), n_edges.astype(np.int64))
    assert edges.shape == (2, int(n_edges.sum()))


def test_every_edge_is_ordered_and_in_range(tmp_path: Path) -> None:
    """u < v for every edge, and both endpoints lie inside their own graph."""
    path = tmp_path / "bounds.npz"
    save_exported(_dataset(_sample_graphs(), key="bounds"), path)
    with np.load(path) as handle:
        offsets, edges, n_nodes = handle["edge_offsets"], handle["edges"], handle["n_nodes"]

    for i in range(len(n_nodes)):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        for k in range(lo, hi):
            u, v = int(edges[0, k]), int(edges[1, k])
            assert 0 <= u < v < int(n_nodes[i])


def test_edges_are_sorted_within_each_graph(tmp_path: Path) -> None:
    """Deterministic ordering, so the serialization does not depend on nx internals."""
    path = tmp_path / "sorted.npz"
    save_exported(_dataset(_sample_graphs(), key="sorted"), path)
    with np.load(path) as handle:
        offsets, edges = handle["edge_offsets"], handle["edges"]
    for i in range(len(offsets) - 1):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        pairs = [(int(edges[0, k]), int(edges[1, k])) for k in range(lo, hi)]
        assert pairs == sorted(pairs)


# --------------------------------------------------------------------------- #
# Defensive validation
# --------------------------------------------------------------------------- #


def test_save_rejects_inconsistent_node_counts(tmp_path: Path) -> None:
    graphs = _sample_graphs()
    bad = np.asarray([g.number_of_nodes() + 1 for g in graphs], dtype=np.int32)
    with pytest.raises(ExportError, match="n_nodes"):
        save_exported(_dataset(graphs, n_nodes=bad), tmp_path / "bad.npz")


def test_save_rejects_inconsistent_edge_counts(tmp_path: Path) -> None:
    graphs = _sample_graphs()
    bad = np.asarray([g.number_of_edges() + 1 for g in graphs], dtype=np.int32)
    with pytest.raises(ExportError, match="n_edges"):
        save_exported(_dataset(graphs, n_edges=bad), tmp_path / "bad.npz")


def test_save_rejects_length_mismatch(tmp_path: Path) -> None:
    graphs = _sample_graphs()
    with pytest.raises(ExportError, match="length mismatch"):
        save_exported(_dataset(graphs, splits=["train"]), tmp_path / "bad.npz")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.__setitem__("edge_offsets", np.array([1, 1, 2], dtype=np.int64)), r"\[0\]"),
        (lambda d: d.__setitem__("edges", np.array([[1], [0]], dtype=np.int32)), "u < v"),
        (lambda d: d.__setitem__("n_edges", np.array([0, 5], dtype=np.int32)), "disagree"),
    ],
)
def test_load_rejects_corrupt_files(
    tmp_path: Path, mutate: Callable[[dict[str, np.ndarray]], None], message: str
) -> None:
    """A hand-damaged npz fails loudly instead of yielding wrong graphs."""
    one = nx.Graph()
    one.add_nodes_from(range(2))
    one.add_edge(0, 1)
    edgeless = nx.Graph()
    edgeless.add_nodes_from(range(2))

    good = tmp_path / "good.npz"
    save_exported(_dataset([edgeless, one], key="corrupt"), good)
    with np.load(good) as handle:
        payload = {k: handle[k] for k in handle.files}

    mutate(payload)
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, **payload)
    with pytest.raises(ExportError, match=message):
        load_exported(bad)


def test_load_rejects_out_of_range_endpoint(tmp_path: Path) -> None:
    """An endpoint beyond its own graph's node count is rejected."""
    one = nx.Graph()
    one.add_nodes_from(range(2))
    one.add_edge(0, 1)
    good = tmp_path / "good.npz"
    save_exported(_dataset([one], key="oor"), good)
    with np.load(good) as handle:
        payload = {k: handle[k] for k in handle.files}
    payload["edges"] = np.array([[0], [7]], dtype=np.int32)
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, **payload)
    with pytest.raises(ExportError, match="out of range"):
        load_exported(bad)


def test_load_rejects_duplicate_edge(tmp_path: Path) -> None:
    """A duplicated edge would silently collapse when rebuilding the graph."""
    two = nx.Graph()
    two.add_nodes_from(range(3))
    two.add_edges_from([(0, 1), (1, 2)])
    good = tmp_path / "good.npz"
    save_exported(_dataset([two], key="dup"), good)
    with np.load(good) as handle:
        payload = {k: handle[k] for k in handle.files}
    payload["edges"] = np.array([[0, 0], [1, 1]], dtype=np.int32)
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, **payload)
    with pytest.raises(ExportError, match="duplicate edge"):
        load_exported(bad)


# --------------------------------------------------------------------------- #
# Metadata schema
# --------------------------------------------------------------------------- #

_REQUIRED_METADATA_FIELDS = (
    "dataset",
    "source",
    "n_raw",
    "n_kept",
    "n_dropped_size",
    "n_dropped_disconnected",
    "n_dropped_trivial",
    "n_pairs",
    "filter",
    "splits_merged",
    "exported_utc",
    "code_commit",
    "schema_version",
)


def test_metadata_round_trips_with_every_contract_field(tmp_path: Path) -> None:
    """metadata survives as JSON and carries the full CONTRACTS section 4 schema."""
    graphs = _sample_graphs()
    original = _dataset(graphs, key="meta")
    path = tmp_path / "meta.npz"
    save_exported(original, path)
    loaded = load_exported(path)

    assert loaded.metadata == original.metadata
    for field in _REQUIRED_METADATA_FIELDS:
        assert field in loaded.metadata, f"missing metadata field {field}"
    assert loaded.metadata["filter"] == {"min_nodes": 2, "require_connected": True, "n_max": 12}
    assert loaded.metadata["splits_merged"] is True
    assert loaded.metadata["schema_version"] == SCHEMA_VERSION
    assert loaded.metadata["n_pairs"] == comb(len(graphs), 2)


def test_metadata_is_stored_as_a_zero_dim_unicode_array(tmp_path: Path) -> None:
    path = tmp_path / "meta.npz"
    save_exported(_dataset(_sample_graphs(), key="meta"), path)
    with np.load(path) as handle:
        arr = handle["metadata"]
    assert arr.ndim == 0
    assert arr.dtype.kind == "U"
    assert json.loads(str(arr.item()))["dataset"] == "meta"


# --------------------------------------------------------------------------- #
# Cohort assertions
# --------------------------------------------------------------------------- #


def test_locked_cohort_table_matches_the_contract() -> None:
    """The five specs and their totals are the numbers in CONTRACTS section 2."""
    assert {k: (s.expected_kept, s.expected_pairs) for k, s in DATASETS.items()} == {
        "iam_letter_low": (1180, 695610),
        "iam_letter_med": (1253, 784378),
        "iam_letter_high": (2059, 2118711),
        "linux": (89, 3916),
        "aids": (769, 295296),
    }
    assert sum(s.expected_kept for s in DATASETS.values()) == TOTAL_EXPECTED_GRAPHS == 5350
    assert sum(s.expected_pairs for s in DATASETS.values()) == TOTAL_EXPECTED_PAIRS == 3897911
    for spec in DATASETS.values():
        assert spec.expected_pairs == comb(spec.expected_kept, 2)


def test_assert_cohort_accepts_the_locked_counts() -> None:
    spec = DATASETS["aids"]
    assert_cohort(spec, 769, 295296)


@pytest.mark.parametrize(("kept", "pairs"), [(768, 295296), (769, 295295), (0, 0)])
def test_assert_cohort_raises_on_mismatch(kept: int, pairs: int) -> None:
    spec = DATASETS["aids"]
    with pytest.raises(CohortMismatchError) as excinfo:
        assert_cohort(spec, kept, pairs)
    message = str(excinfo.value)
    assert str(kept) in message and "769" in message  # observed beside expected


def test_main_exits_non_zero_on_cohort_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wrong count aborts the CLI; the filter is never adjusted to make it pass."""
    import benchmarks.eval_setup.export_graphs as mod

    def fake_load_raw(
        spec: DatasetSpec, source_dir: Path
    ) -> tuple[list[nx.Graph], list[str], list[str], list[str]]:
        g = nx.path_graph(3)
        return [g, g], ["a", "b"], ["train", "test"], ["", ""]

    monkeypatch.setattr(mod, "_load_raw", fake_load_raw)
    status = main(["--source", str(tmp_path), "--out", str(tmp_path), "--datasets", "linux"])
    assert status == 1
    assert not (tmp_path / "linux.npz").exists()


def test_main_exits_non_zero_when_source_missing(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    status = main(["--source", str(missing), "--out", str(tmp_path), "--datasets", "linux"])
    assert status == 1


# --------------------------------------------------------------------------- #
# Manifest and CLI plumbing
# --------------------------------------------------------------------------- #


def test_manifest_round_trips(tmp_path: Path) -> None:
    entries = {"linux": {"sha256": "ab", "bytes": 3, "n_kept": 89, "n_pairs": 3916}}
    write_manifest(entries, tmp_path)
    assert (tmp_path / MANIFEST_NAME).is_file()
    assert read_manifest(tmp_path) == entries


def test_read_manifest_raises_when_absent(tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="manifest not found"):
        read_manifest(tmp_path)


def test_parser_rejects_unknown_dataset_key() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--datasets", "not_a_dataset"])


def test_parser_expands_all_to_the_five_keys() -> None:
    args = build_parser().parse_args([])
    assert args.datasets == list(DATASETS)
    assert len(args.datasets) == 5


def test_content_sha256_is_stable_and_data_dependent() -> None:
    """The content digest is reproducible, unlike the zip-framed file digest."""
    base = _dataset(_sample_graphs(), key="cs")
    assert content_sha256(base) == content_sha256(_dataset(_sample_graphs(), key="cs"))
    altered = _sample_graphs()
    altered[2].add_edge(0, 3)
    assert content_sha256(base) != content_sha256(_dataset(altered, key="cs"))


def test_verify_reports_a_tampered_file(tmp_path: Path) -> None:
    """verify_exports fails when the bytes on disk stop matching the manifest."""
    path = tmp_path / "linux.npz"
    dataset = _dataset(_sample_graphs(), key="linux")
    save_exported(dataset, path)
    write_manifest(
        {
            "linux": {
                "sha256": "0" * 64,
                "bytes": path.stat().st_size,
                "n_kept": 89,
                "n_pairs": 3916,
            }
        },
        tmp_path,
    )
    assert verify_exports(tmp_path, ["linux"]) is False


# --------------------------------------------------------------------------- #
# load_exported must not need torch
# --------------------------------------------------------------------------- #

_NO_TORCH_PROBE = textwrap.dedent(
    """
    import sys
    from importlib.abc import MetaPathFinder


    class BlockTorch(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "torch" or fullname.startswith("torch."):
                raise ImportError("torch is blocked: Picasso has no torch, by design")
            return None


    sys.meta_path.insert(0, BlockTorch())
    sys.path.insert(0, {repo!r})

    from benchmarks.eval_setup.export_graphs import load_exported

    dataset = load_exported({npz!r})
    assert "torch" not in sys.modules, "importing export_graphs pulled in torch"
    print(len(dataset.graphs), dataset.graphs[-1].number_of_edges())
    """
)


def test_load_exported_works_without_torch(tmp_path: Path) -> None:
    """Picasso has no torch; the read path must never need it, even transitively."""
    path = tmp_path / "notorch.npz"
    graphs = _sample_graphs()
    save_exported(_dataset(graphs, key="notorch"), path)

    script = tmp_path / "probe.py"
    script.write_text(_NO_TORCH_PROBE.format(repo=str(REPO_ROOT), npz=str(path)), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=180
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert proc.stdout.split() == [str(len(graphs)), str(graphs[-1].number_of_edges())]


# --------------------------------------------------------------------------- #
# Real data
# --------------------------------------------------------------------------- #


@pytest.mark.integration
@requires_source
@pytest.mark.parametrize("key", list(DATASETS))
def test_real_export_reproduces_the_locked_cohort(tmp_path: Path, key: str) -> None:
    """The locked counts are reproduced from the source tree, per dataset."""
    spec = DATASETS[key]
    entries = export_all(DEFAULT_SOURCE_DIR, tmp_path, [key])

    assert entries[key]["n_kept"] == spec.expected_kept
    assert entries[key]["n_pairs"] == spec.expected_pairs

    loaded = load_exported(tmp_path / f"{key}.npz")
    assert len(loaded.graphs) == spec.expected_kept
    assert comb(len(loaded.graphs), 2) == spec.expected_pairs
    assert loaded.metadata["dataset"] == key
    assert loaded.metadata["n_kept"] == spec.expected_kept
    assert set(loaded.splits) <= {"train", "val", "test", "validation"}

    for graph, n_nodes in zip(loaded.graphs, loaded.n_nodes, strict=True):
        assert 2 <= int(n_nodes) <= 12
        assert nx.is_connected(graph)
        assert sorted(graph.nodes()) == list(range(int(n_nodes)))


@pytest.mark.integration
@requires_source
def test_real_export_all_five_totals(tmp_path: Path) -> None:
    """All five together must total 5,350 graphs and 3,897,911 pairs."""
    entries = export_all(DEFAULT_SOURCE_DIR, tmp_path, list(DATASETS))
    assert sum(int(e["n_kept"]) for e in entries.values()) == TOTAL_EXPECTED_GRAPHS
    assert sum(int(e["n_pairs"]) for e in entries.values()) == TOTAL_EXPECTED_PAIRS
    assert verify_exports(tmp_path, list(DATASETS)) is True


@pytest.mark.integration
@requires_source
def test_real_export_is_deterministic(tmp_path: Path) -> None:
    """Two exports of LINUX agree on content, so the digest certifies the data."""
    first = export_all(DEFAULT_SOURCE_DIR, tmp_path / "a", ["linux"])
    second = export_all(DEFAULT_SOURCE_DIR, tmp_path / "b", ["linux"])
    assert first["linux"]["content_sha256"] == second["linux"]["content_sha256"]


@pytest.mark.integration
@requires_source
def test_real_aids_retains_within_split_structure(tmp_path: Path) -> None:
    """Gate 0 needs within-split AIDS pairs, which merging would otherwise erase."""
    export_all(DEFAULT_SOURCE_DIR, tmp_path, ["aids"])
    loaded = load_exported(tmp_path / "aids.npz")
    assert set(loaded.splits) == {"train", "val", "test"}
    for graph_id, split in zip(loaded.graph_ids, loaded.splits, strict=True):
        assert graph_id.startswith(f"aids_{split}_")


@pytest.mark.integration
@pytest.mark.skipif(
    not (Path(DEFAULT_EXPORT_DIR) / MANIFEST_NAME).is_file(),
    reason=f"no export present at {DEFAULT_EXPORT_DIR}",
)
def test_committed_export_still_verifies() -> None:
    """The export the cluster will consume still matches its manifest."""
    assert verify_exports(DEFAULT_EXPORT_DIR, list(DATASETS)) is True
