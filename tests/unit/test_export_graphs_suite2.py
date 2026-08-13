"""Unit tests for export_graphs_suite2 -- the ten Suite-2 datasets, CONTRACTS sections 1 and 2.

The ten files this module tests are the *only* input that reaches Picasso, and a cohort that differs
by one graph from T-01's certified enumeration produces wrong numbers everywhere downstream without
raising anything. So the tests here are weighted towards the failures that are silent:

* the ten locked counts, which are the primary check;
* COIL-DEL's split-index enumeration against the directory one, which differ by 3,300 graphs and
  18,313,350 pairs and which nothing else distinguishes;
* the ``graph_ids`` of the four datasets whose Suite-2 cohort equals Suite 1, checked element-wise
  against a census already on record;
* export order, which every downstream pair index depends on positionally.

Real-data tests are marked ``integration`` and skip when the Sandisk tree is absent, so the suite
passes on a machine without it.
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from benchmarks.eval_setup.cohort_audit import SUITE2_KEYS
from benchmarks.eval_setup.export_graphs import load_exported
from benchmarks.eval_setup.export_graphs_suite2 import (
    COIL_DEL_CLASSES,
    COIL_DEL_PER_CLASS,
    DEFAULT_EXPORT_DIR,
    DEFAULT_GRAPHEDX_ROOT,
    DEFAULT_IAM_ROOT,
    DEFAULT_REFERENCE_DIR,
    ENUMERATION_LABEL,
    FILTER_MIN_NODES,
    REFERENCE_KEYS,
    SUITE2_DATASETS,
    TOTAL_EXPECTED_GRAPHS,
    TOTAL_EXPECTED_PAIRS,
    Suite2CohortMismatchError,
    assert_cohort,
    assert_coil_del_balance,
    assert_label_classes,
    check_reference_graph_ids,
    export_all,
    load_raw,
    main,
    verify,
)

_IAM_PRESENT = Path(DEFAULT_IAM_ROOT).is_dir()
_GRAPHEDX_PRESENT = Path(DEFAULT_GRAPHEDX_ROOT).is_dir()
requires_source = pytest.mark.skipif(
    not (_IAM_PRESENT and _GRAPHEDX_PRESENT),
    reason=f"source tree absent: {DEFAULT_IAM_ROOT} / {DEFAULT_GRAPHEDX_ROOT}",
)

#: CONTRACTS section 2 requires exactly these; the two ``label_class`` pairs were added by the
#: orchestrator's ruling of 2026-08-13 so T-18 can see which classes the filter removes.
EXPECTED_METADATA_KEYS = {
    "dataset",
    "source",
    "n_raw",
    "n_kept",
    "n_dropped_min_nodes",
    "n_dropped_disconnected",
    "n_pairs",
    "filter",
    "splits_merged",
    "enumeration",
    "n_label_classes",
    "label_classes",
    "n_label_classes_raw",
    "label_classes_lost",
    "exported_utc",
    "code_commit",
    "schema_version",
}


# --------------------------------------------------------------------------- #
# Registry -- no data required
# --------------------------------------------------------------------------- #


def test_registry_is_t01_certified_enumeration_in_order() -> None:
    """The registry must be ``cohort_audit.SUITE2_KEYS``, same keys *and* same order.

    Order is not cosmetic: pair indices are positional, so a reordered registry silently maps every
    downstream result onto the wrong dataset.
    """
    assert tuple(SUITE2_DATASETS) == tuple(SUITE2_KEYS)


def test_registry_totals_match_the_locked_cohort() -> None:
    assert sum(s.expected_kept for s in SUITE2_DATASETS.values()) == TOTAL_EXPECTED_GRAPHS
    assert sum(s.expected_pairs for s in SUITE2_DATASETS.values()) == TOTAL_EXPECTED_PAIRS
    assert TOTAL_EXPECTED_GRAPHS == 16370
    assert TOTAL_EXPECTED_PAIRS == 21710892


@pytest.mark.parametrize("key", list(SUITE2_DATASETS))
def test_declared_pairs_are_the_binomial_of_declared_graphs(key: str) -> None:
    spec = SUITE2_DATASETS[key]
    assert spec.expected_pairs == comb(spec.expected_kept, 2)


def test_coil_del_is_the_split_index_count_not_the_directory_count() -> None:
    """3,900 is the split index; 7,200 is the directory. The registry must hold the former."""
    assert SUITE2_DATASETS["coil_del"].expected_kept == 3900
    assert SUITE2_DATASETS["coil_del"].expected_kept != 7200
    assert SUITE2_DATASETS["coil_del"].expected_pairs == comb(3900, 2) == 7603050


def test_aids_graphedx_is_the_n_max_free_cohort() -> None:
    """819, not Suite 1's 769. The two differ only by ``n_max = 12`` and must never be confused."""
    assert SUITE2_DATASETS["aids_graphedx"].expected_kept == 819
    assert SUITE2_DATASETS["aids_graphedx"].expected_kept != 769
    assert "aids_graphedx" in SUITE2_DATASETS
    assert "aids" not in SUITE2_DATASETS


def test_rejected_datasets_are_absent() -> None:
    for key in ("coil_rag", "fingerprint", "web"):
        assert key not in SUITE2_DATASETS


# --------------------------------------------------------------------------- #
# Assertions -- fabricated inputs
# --------------------------------------------------------------------------- #


def test_assert_cohort_accepts_the_locked_counts() -> None:
    spec = SUITE2_DATASETS["grec"]
    assert_cohort(spec, spec.expected_kept, spec.expected_pairs)


@pytest.mark.parametrize("delta", [-1, 1])
def test_assert_cohort_rejects_an_off_by_one_graph_count(delta: int) -> None:
    spec = SUITE2_DATASETS["grec"]
    with pytest.raises(Suite2CohortMismatchError, match="grec"):
        assert_cohort(spec, spec.expected_kept + delta, spec.expected_pairs)


def test_assert_label_classes_counts_only_non_empty_labels() -> None:
    spec = SUITE2_DATASETS["mutagenicity"]
    assert_label_classes(spec, ["mutagen", "nonmutagen", "mutagen", ""])
    with pytest.raises(Suite2CohortMismatchError, match="label classes"):
        assert_label_classes(spec, ["mutagen", "mutagen"])


def test_assert_label_classes_accepts_an_unlabelled_dataset() -> None:
    """GraphEdX has no class label; ``expected_label_classes`` is 0 and every label is ``''``."""
    spec = SUITE2_DATASETS["linux"]
    assert spec.expected_label_classes == 0
    assert_label_classes(spec, [""] * 89)


def test_coil_del_balance_accepts_100_classes_of_39() -> None:
    labels = [str(c) for c in range(1, COIL_DEL_CLASSES + 1) for _ in range(COIL_DEL_PER_CLASS)]
    assert len(labels) == 3900
    assert_coil_del_balance(labels)


def test_coil_del_balance_rejects_the_directory_enumeration_shape() -> None:
    """The directory adds 3,300 unlabelled graphs, so the balance breaks even if a count matched."""
    labels = [str(c) for c in range(1, COIL_DEL_CLASSES + 1) for _ in range(COIL_DEL_PER_CLASS)]
    labels += [""] * 3300
    with pytest.raises(Suite2CohortMismatchError, match="coil_del"):
        assert_coil_del_balance(labels)


def test_coil_del_balance_rejects_an_unbalanced_class() -> None:
    labels = [str(c) for c in range(1, COIL_DEL_CLASSES + 1) for _ in range(COIL_DEL_PER_CLASS)]
    labels.append("1")
    with pytest.raises(Suite2CohortMismatchError, match="sizes"):
        assert_coil_del_balance(labels)


# --------------------------------------------------------------------------- #
# Reference census check -- fabricated npz
# --------------------------------------------------------------------------- #


def _write_reference(tmp_path: Path, key: str, ids: list[str]) -> Path:
    np.savez_compressed(tmp_path / f"{key}.npz", graph_ids=np.asarray(ids, dtype=np.str_))
    return tmp_path


def test_reference_check_passes_on_an_identical_array(tmp_path: Path) -> None:
    ids = ["AP1_0001", "AP1_0002", "ZP1_0149"]
    _write_reference(tmp_path, "iam_letter_low", ids)
    assert check_reference_graph_ids("iam_letter_low", ids, tmp_path) == []


def test_reference_check_catches_a_reordering(tmp_path: Path) -> None:
    """A permutation preserves every count and breaks every downstream pair index."""
    ids = ["AP1_0001", "AP1_0002", "ZP1_0149"]
    _write_reference(tmp_path, "iam_letter_low", ids)
    problems = check_reference_graph_ids("iam_letter_low", [ids[1], ids[0], ids[2]], tmp_path)
    assert len(problems) == 1
    assert "index 0" in problems[0]


def test_reference_check_catches_a_length_change(tmp_path: Path) -> None:
    _write_reference(tmp_path, "linux", ["linux_train_0000", "linux_train_0001"])
    problems = check_reference_graph_ids("linux", ["linux_train_0000"], tmp_path)
    assert len(problems) == 1
    assert "shape" in problems[0]


def test_reference_check_skips_datasets_with_no_census(tmp_path: Path) -> None:
    """``aids_graphedx`` is deliberately outside REFERENCE_KEYS -- Suite 1 kept 769 of the 819."""
    assert "aids_graphedx" not in REFERENCE_KEYS
    assert check_reference_graph_ids("aids_graphedx", ["x"], tmp_path) == []


def test_reference_check_is_quiet_when_the_census_file_is_absent(tmp_path: Path) -> None:
    assert check_reference_graph_ids("linux", ["a", "b"], tmp_path) == []


# --------------------------------------------------------------------------- #
# Real data
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def loaded() -> dict[str, tuple[int, int, int, list[str], list[str]]]:
    """Load, filter and summarise all ten datasets once for the whole module.

    Returns
    -------
    dict
        ``key -> (n_raw, n_kept, n_pairs, kept_ids, kept_labels)``.
    """
    if not (_IAM_PRESENT and _GRAPHEDX_PRESENT):
        pytest.skip("source tree absent")

    from benchmarks.eval_setup.dataset_filter import filter_graphs
    from benchmarks.eval_setup.export_graphs_suite2 import (
        FILTER_REQUIRE_CONNECTED,
        NO_N_MAX,
    )

    out: dict[str, tuple[int, int, int, list[str], list[str]]] = {}
    for key, spec in SUITE2_DATASETS.items():
        graphs, ids, _splits, labels = load_raw(spec, DEFAULT_IAM_ROOT, DEFAULT_GRAPHEDX_ROOT)
        result = filter_graphs(
            graphs,
            ids,
            n_max=NO_N_MAX,
            require_connected=FILTER_REQUIRE_CONNECTED,
            min_nodes=FILTER_MIN_NODES,
        )
        out[key] = (
            result.n_raw,
            result.n_kept,
            comb(result.n_kept, 2),
            [ids[i] for i in result.kept_indices],
            [labels[i] for i in result.kept_indices],
        )
    return out


@pytest.mark.integration
@requires_source
@pytest.mark.parametrize("key", list(SUITE2_DATASETS))
def test_locked_graph_and_pair_counts_reproduce(
    key: str, loaded: dict[str, tuple[int, int, int, list[str], list[str]]]
) -> None:
    """Each of the ten CONTRACTS section 1 rows, graphs and pairs."""
    spec = SUITE2_DATASETS[key]
    _n_raw, n_kept, n_pairs, _ids, _labels = loaded[key]
    assert n_kept == spec.expected_kept
    assert n_pairs == spec.expected_pairs


@pytest.mark.integration
@requires_source
def test_locked_totals_reproduce(
    loaded: dict[str, tuple[int, int, int, list[str], list[str]]],
) -> None:
    assert sum(v[1] for v in loaded.values()) == TOTAL_EXPECTED_GRAPHS
    assert sum(v[2] for v in loaded.values()) == TOTAL_EXPECTED_PAIRS


@pytest.mark.integration
@requires_source
@pytest.mark.parametrize("key", list(REFERENCE_KEYS))
def test_graph_ids_reproduce_the_suite1_census(
    key: str, loaded: dict[str, tuple[int, int, int, list[str], list[str]]]
) -> None:
    """Element-wise against ``extended_merged_exact_ged/computed/{key}.npz``.

    An exact end-to-end check of the loader *and* the export order against a census on record. It
    catches reorderings that every count in the cohort table would pass.
    """
    if not Path(DEFAULT_REFERENCE_DIR).is_dir():
        pytest.skip(f"reference census absent: {DEFAULT_REFERENCE_DIR}")
    _n_raw, _n_kept, _n_pairs, ids, _labels = loaded[key]
    assert check_reference_graph_ids(key, ids, DEFAULT_REFERENCE_DIR) == []


@pytest.mark.integration
@requires_source
def test_label_classes_are_populated_as_measured(
    loaded: dict[str, tuple[int, int, int, list[str], list[str]]],
) -> None:
    """Post-filter class counts, which differ from the raw dataset counts.

    GREC retains 17 of 22 and Letter LOW 9 of 15 because the connectivity filter removes whole
    classes. Asserting the measured value is what makes a future label regression fail.
    """
    observed = {key: len({x for x in v[4] if x}) for key, v in loaded.items()}
    assert observed == {
        "iam_letter_low": 9,
        "iam_letter_med": 15,
        "iam_letter_high": 15,
        "linux": 0,
        "aids_graphedx": 0,
        "grec": 17,
        "aids_iam": 2,
        "coil_del": 100,
        "mutagenicity": 2,
        "protein": 6,
    }


@pytest.mark.integration
@requires_source
def test_coil_del_split_index_is_balanced_and_the_directory_is_not() -> None:
    """The two enumerations, side by side. This is the retraction guard.

    Split index: 3,900 graphs, 100 x 39, every graph labelled. Directory: 7,200 graphs, 3,300 of
    them carrying no class label at all. Nothing else in the pipeline separates the two.
    """
    from benchmarks.eval_setup.iam_gxl_loader import load_iam_gxl

    by_index = load_iam_gxl(DEFAULT_IAM_ROOT, "coil_del", enumeration="cxl")
    assert len(by_index.graphs) == 3900
    assert_coil_del_balance(by_index.labels)
    assert all(by_index.labels)

    by_directory = load_iam_gxl(DEFAULT_IAM_ROOT, "coil_del", enumeration="directory")
    assert len(by_directory.graphs) == 7200
    assert sum(1 for label in by_directory.labels if not label) == 3300
    with pytest.raises(Suite2CohortMismatchError):
        assert_coil_del_balance(by_directory.labels)

    assert comb(7200, 2) - comb(3900, 2) == 18313350


@pytest.mark.integration
@requires_source
def test_no_n_max_keeps_graphs_suite1_would_drop(
    loaded: dict[str, tuple[int, int, int, list[str], list[str]]],
) -> None:
    """Suite 2 has no node ceiling: ``aids_graphedx`` keeps 819 where Suite 1 kept 769."""
    assert loaded["aids_graphedx"][1] == 819
    assert loaded["aids_graphedx"][0] == 911
    # Datasets whose largest graph exceeds Suite 1's n_max = 12 exist only because there is no cap.
    assert loaded["mutagenicity"][1] == 4040


@pytest.mark.integration
@requires_source
def test_verify_reports_no_problems() -> None:
    assert verify(DEFAULT_IAM_ROOT, DEFAULT_GRAPHEDX_ROOT, DEFAULT_REFERENCE_DIR) == []


@pytest.mark.integration
@requires_source
def test_main_verify_only_exits_zero() -> None:
    assert main(["--verify-only", "--reference-dir", DEFAULT_REFERENCE_DIR]) == 0


@pytest.mark.integration
@requires_source
def test_main_exits_non_zero_on_a_corrupted_expectation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wrong locked count must stop the pipeline, not be reconciled by adjusting the filter."""
    import dataclasses

    from benchmarks.eval_setup import export_graphs_suite2 as module

    corrupted = dict(module.SUITE2_DATASETS)
    corrupted["coil_del"] = dataclasses.replace(corrupted["coil_del"], expected_kept=7200)
    monkeypatch.setattr(module, "SUITE2_DATASETS", corrupted)
    assert module.main(["--verify-only", "--reference-dir", ""]) == 1


@pytest.mark.integration
@requires_source
def test_export_is_deterministic_across_two_runs(tmp_path: Path) -> None:
    """Two exports must agree on content. ``content_sha256`` covers ids, splits, labels and CSR."""
    first = export_all(DEFAULT_IAM_ROOT, DEFAULT_GRAPHEDX_ROOT, tmp_path / "a", None)
    second = export_all(DEFAULT_IAM_ROOT, DEFAULT_GRAPHEDX_ROOT, tmp_path / "b", None)
    for key in SUITE2_DATASETS:
        assert first[key]["content_sha256"] == second[key]["content_sha256"], key


@pytest.mark.integration
@requires_source
@pytest.mark.skipif(
    not Path(DEFAULT_EXPORT_DIR).is_dir(), reason=f"export absent: {DEFAULT_EXPORT_DIR}"
)
@pytest.mark.parametrize("key", list(SUITE2_DATASETS))
def test_written_files_load_and_carry_the_contract_schema(key: str) -> None:
    """``load_exported`` must read every file unchanged, with the CONTRACTS section 2 metadata."""
    path = Path(DEFAULT_EXPORT_DIR) / f"{key}.npz"
    dataset = load_exported(path)
    spec = SUITE2_DATASETS[key]

    assert len(dataset.graphs) == spec.expected_kept
    assert len(dataset.graph_ids) == spec.expected_kept
    assert len(dataset.splits) == spec.expected_kept
    assert len(dataset.labels) == spec.expected_kept
    assert dataset.n_nodes.dtype == np.int32
    assert dataset.n_edges.dtype == np.int32
    assert all(isinstance(g, nx.Graph) for g in dataset.graphs)

    meta = dataset.metadata
    assert set(meta) == EXPECTED_METADATA_KEYS
    assert meta["dataset"] == key
    assert meta["n_kept"] == spec.expected_kept
    assert meta["n_pairs"] == spec.expected_pairs
    assert meta["enumeration"] == ENUMERATION_LABEL
    assert meta["splits_merged"] is True
    assert meta["filter"] == {"min_nodes": 2, "require_connected": True, "n_max": None}
    assert meta["n_label_classes"] == spec.expected_label_classes
    assert len(meta["label_classes"]) == spec.expected_label_classes


@pytest.mark.integration
@requires_source
@pytest.mark.skipif(
    not Path(DEFAULT_EXPORT_DIR).is_dir(), reason=f"export absent: {DEFAULT_EXPORT_DIR}"
)
def test_written_manifest_totals_match_the_locked_cohort() -> None:
    manifest = json.loads((Path(DEFAULT_EXPORT_DIR) / "manifest.json").read_text())
    assert manifest["_totals"]["n_graphs"] == TOTAL_EXPECTED_GRAPHS
    assert manifest["_totals"]["n_pairs"] == TOTAL_EXPECTED_PAIRS
    assert manifest["_totals"]["n_datasets"] == 10
    for key in SUITE2_DATASETS:
        assert manifest[key]["n_graphs"] == SUITE2_DATASETS[key].expected_kept
        assert "label_classes" in manifest[key]
        assert "label_classes_lost" in manifest[key]


@pytest.mark.integration
@requires_source
@pytest.mark.skipif(
    not Path(DEFAULT_EXPORT_DIR).is_dir(), reason=f"export absent: {DEFAULT_EXPORT_DIR}"
)
def test_written_graphs_are_relabelled_and_connected() -> None:
    """Spot check on the two extremes of the cohort: the smallest and the largest dataset."""
    for key in ("linux", "mutagenicity"):
        dataset = load_exported(Path(DEFAULT_EXPORT_DIR) / f"{key}.npz")
        for graph, n in zip(dataset.graphs, dataset.n_nodes, strict=True):
            assert set(graph.nodes()) == set(range(int(n)))
            assert graph.number_of_nodes() >= FILTER_MIN_NODES
        assert nx.is_connected(dataset.graphs[0])
