"""Unit tests for the IAM GXL/CXL loader (T-01).

The loader exists because the Suite-2 cohort had no reproducing code. These tests
cover the behaviours that decide a printed number:

* the **two enumeration policies** and the case where they disagree -- the real
  COIL-DEL situation, where 7,200 files ship but the split index lists 3,900;
* IAM's inconsistent split-file naming (``valid.cxl`` vs ``validation.cxl``),
  which silently drops a third of a dataset if only one spelling is probed;
* a graph listed by two splits counted **once**, since splits are merged;
* a parse failure recorded rather than raised, so one bad file cannot make a
  whole dataset vanish without a trace.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.eval_setup.iam_gxl_loader import (
    IAMLoaderError,
    load_iam_gxl,
    read_split_index,
)

pytestmark = pytest.mark.unit


def _gxl(nodes: int, edges: list[tuple[int, int]], node_attrs: tuple[str, ...] = ()) -> str:
    """Render a minimal IAM-style GXL document."""
    attr_xml = "".join(f'<attr name="{a}"><int>1</int></attr>' for a in node_attrs)
    node_xml = "".join(f'<node id="_{i}">{attr_xml}</node>' for i in range(nodes))
    edge_xml = "".join(f'<edge from="_{u}" to="_{v}"></edge>' for u, v in edges)
    return (
        '<?xml version="1.0"?><gxl>'
        f'<graph id="g" edgemode="undirected">{node_xml}{edge_xml}</graph>'
        "</gxl>"
    )


def _cxl(entries: list[tuple[str, str]]) -> str:
    """Render a minimal IAM-style CXL index."""
    prints = "".join(f'<print file="{f}" class="{c}"/>' for f, c in entries)
    return f'<?xml version="1.0"?><GraphCollection>{prints}</GraphCollection>'


@pytest.fixture()
def grec_dir(tmp_path: Path) -> Path:
    """A GREC-shaped directory: four files on disk, three named by the index.

    Mirrors COIL-DEL, where the split index covers a strict subset of the files.
    """
    data = tmp_path / "GREC" / "data"
    data.mkdir(parents=True)

    (data / "a.gxl").write_text(_gxl(3, [(0, 1), (1, 2)], ("x", "y")))
    (data / "b.gxl").write_text(_gxl(4, [(0, 1), (1, 2), (2, 3), (3, 0)]))
    (data / "c.gxl").write_text(_gxl(2, [(0, 1)]))
    (data / "orphan.gxl").write_text(_gxl(5, [(0, 1), (1, 2), (2, 3), (3, 4)]))

    (data / "train.cxl").write_text(_cxl([("a.gxl", "A"), ("b.gxl", "B")]))
    (data / "test.cxl").write_text(_cxl([("c.gxl", "C")]))
    return tmp_path


def test_cxl_enumeration_ignores_unindexed_files(grec_dir: Path) -> None:
    ds = load_iam_gxl(str(grec_dir), "grec", enumeration="cxl")

    assert len(ds.graphs) == 3
    assert set(ds.graph_ids) == {"a", "b", "c"}
    assert ds.n_files_on_disk == 4
    assert ds.n_files_indexed == 3


def test_directory_enumeration_takes_every_file(grec_dir: Path) -> None:
    ds = load_iam_gxl(str(grec_dir), "grec", enumeration="directory")

    assert len(ds.graphs) == 4
    assert "orphan" in ds.graph_ids
    # Both counts are reported under either policy, so a caller can detect the
    # divergence without loading the dataset twice.
    assert (ds.n_files_on_disk, ds.n_files_indexed) == (4, 3)


def test_unindexed_file_carries_no_split_or_label(grec_dir: Path) -> None:
    ds = load_iam_gxl(str(grec_dir), "grec", enumeration="directory")
    idx = ds.graph_ids.index("orphan")

    assert ds.splits[idx] == ""
    assert ds.labels[idx] == ""


def test_labels_and_splits_come_from_the_index(grec_dir: Path) -> None:
    ds = load_iam_gxl(str(grec_dir), "grec", enumeration="cxl")
    by_id = dict(zip(ds.graph_ids, zip(ds.labels, ds.splits, strict=True), strict=True))

    assert by_id["a"] == ("A", "train")
    assert by_id["c"] == ("C", "test")
    assert ds.split_sizes == {"train": 2, "test": 1}


def test_topology_only_and_attributes_recorded(grec_dir: Path) -> None:
    ds = load_iam_gxl(str(grec_dir), "grec", enumeration="cxl")
    graph = ds.graphs[ds.graph_ids.index("a")]

    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2
    assert all(not data for _, data in graph.nodes(data=True))
    # The values are discarded; which attributes were discarded is the measurement
    # AE.4b's label column needs.
    assert ds.node_attributes == ["x", "y"]


@pytest.mark.parametrize("spelling", ["valid", "validation"])
def test_both_validation_spellings_are_read(tmp_path: Path, spelling: str) -> None:
    """IAM ships ``valid.cxl`` for GREC and ``validation.cxl`` for Mutagenicity."""
    data = tmp_path / "GREC" / "data"
    data.mkdir(parents=True)
    (data / "v.gxl").write_text(_gxl(2, [(0, 1)]))
    (data / f"{spelling}.cxl").write_text(_cxl([("v.gxl", "V")]))

    index, sizes = read_split_index(str(data))

    assert index == {"v.gxl": (spelling, "V")}
    assert sizes == {spelling: 1}


def test_graph_listed_by_two_splits_is_counted_once(tmp_path: Path) -> None:
    data = tmp_path / "GREC" / "data"
    data.mkdir(parents=True)
    (data / "dup.gxl").write_text(_gxl(2, [(0, 1)]))
    (data / "train.cxl").write_text(_cxl([("dup.gxl", "A")]))
    (data / "test.cxl").write_text(_cxl([("dup.gxl", "A")]))

    ds = load_iam_gxl(str(tmp_path), "grec", enumeration="cxl")

    assert len(ds.graphs) == 1
    assert ds.splits == ["train"]
    assert ds.split_sizes == {"train": 1, "test": 0}


def test_parse_failure_is_recorded_not_raised(tmp_path: Path) -> None:
    data = tmp_path / "GREC" / "data"
    data.mkdir(parents=True)
    (data / "good.gxl").write_text(_gxl(2, [(0, 1)]))
    (data / "bad.gxl").write_text("<gxl><graph><node id=")
    (data / "train.cxl").write_text(_cxl([("good.gxl", "A"), ("bad.gxl", "B")]))

    ds = load_iam_gxl(str(tmp_path), "grec", enumeration="cxl")

    assert len(ds.graphs) == 1
    assert len(ds.parse_failures) == 1
    assert ds.parse_failures[0].startswith("bad.gxl")


def test_indexed_but_missing_file_is_recorded(tmp_path: Path) -> None:
    data = tmp_path / "GREC" / "data"
    data.mkdir(parents=True)
    (data / "present.gxl").write_text(_gxl(2, [(0, 1)]))
    (data / "train.cxl").write_text(_cxl([("present.gxl", "A"), ("ghost.gxl", "B")]))

    ds = load_iam_gxl(str(tmp_path), "grec", enumeration="cxl")

    assert len(ds.graphs) == 1
    assert any("ghost.gxl" in failure for failure in ds.parse_failures)


def test_unknown_key_raises(tmp_path: Path) -> None:
    with pytest.raises(IAMLoaderError, match="unknown IAM dataset key"):
        load_iam_gxl(str(tmp_path), "not_a_dataset")


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(IAMLoaderError, match="directory not found"):
        load_iam_gxl(str(tmp_path), "grec")


def test_directory_without_gxl_raises(tmp_path: Path) -> None:
    (tmp_path / "GREC" / "data").mkdir(parents=True)

    with pytest.raises(IAMLoaderError, match="no .gxl files"):
        load_iam_gxl(str(tmp_path), "grec")


def test_cxl_enumeration_without_an_index_raises(tmp_path: Path) -> None:
    """Falling back to the directory here would silently change the cohort."""
    data = tmp_path / "GREC" / "data"
    data.mkdir(parents=True)
    (data / "a.gxl").write_text(_gxl(2, [(0, 1)]))

    with pytest.raises(IAMLoaderError, match="cannot enumerate by cxl"):
        load_iam_gxl(str(tmp_path), "grec", enumeration="cxl")


def test_unknown_enumeration_policy_raises(grec_dir: Path) -> None:
    with pytest.raises(IAMLoaderError, match="unknown enumeration policy"):
        load_iam_gxl(str(grec_dir), "grec", enumeration="whatever")  # type: ignore[arg-type]
