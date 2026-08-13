"""Tests for the T-05 independent validation gates and the Picasso launcher.

Every gate test that can run on **real recorded data** does. Building a
contract-shaped role file out of T-27's recorded ``BRANCH_FAST`` and
``BIPARTITE`` census and asserting the gate passes is worth more than any
synthetic fixture: it exercises the same pair ordering, the same value
distribution and the same integer-stored-as-float representation the campaign
will produce. Each of those tests is then run again against a copy with one
entry perturbed, because a gate that cannot fail is not a gate.

The tests skip when the Sandisk reference tree is not mounted, so the suite
still passes on a machine without it. They were exercised with it present.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.real_data.eval_setup import approx_ged_gates as gates  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "slurm" / "approx_ged" / "launcher.sh"

SANDISK = Path("/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph")
T27_CELLS = SANDISK / "results/reports/T-27-ged-bound-bakeoff/data/cells"
T27_INDEX = SANDISK / "results/reports/T-27-ged-bound-bakeoff/data/index"
EXACT_DIR = SANDISK / "data/source/GED_PRECOMPUTED/extended_merged_exact_ged/computed"
EXPORTED = SANDISK / "data/exported"

needs_reference = pytest.mark.skipif(
    not (T27_CELLS.exists() and EXACT_DIR.exists()),
    reason="Sandisk reference tree (T-27 cells / T-03 census) not mounted",
)


def _gedlib_available() -> bool:
    """Report whether the in-place GEDLIB build is importable.

    Returns
    -------
    bool
        True if ``gklearn.gedlib`` loads.
    """
    import importlib

    try:
        importlib.import_module("gklearn.gedlib.libraries_import")
        importlib.import_module("gklearn.gedlib.gedlibpy_gxl")
    except Exception:  # noqa: BLE001 - any import failure means unavailable
        return False
    return True


needs_gedlib = pytest.mark.skipif(
    not _gedlib_available(), reason="GEDLIB (gklearn.gedlib) not importable"
)


# --------------------------------------------------------------------------- fixtures


def _square(values: np.ndarray, n: int) -> np.ndarray:
    """Expand a flat upper triangle into a dense symmetric matrix.

    Parameters
    ----------
    values : ndarray
        Flat values in ``numpy.triu_indices(n, k=1)`` order.
    n : int
        Matrix side.

    Returns
    -------
    ndarray of float64
        The symmetric matrix with a zero diagonal.
    """
    rows, cols = np.triu_indices(n, k=1)
    matrix = np.zeros((n, n), dtype=np.float64)
    matrix[rows, cols] = values
    matrix[cols, rows] = values
    return matrix


def _metadata(dataset: str, role: str, method: str, options: str, n: int, p: int) -> str:
    """Build a CONTRACTS §4-complete metadata JSON string.

    Parameters
    ----------
    dataset, role, method, options : str
        Provenance fields.
    n, p : int
        Graph and pair counts.

    Returns
    -------
    str
        JSON text.
    """
    return json.dumps(
        {
            "dataset": dataset,
            "role": role,
            "method": method,
            "options_string": options,
            "accessor": "lower" if role == "lb" else "upper",
            "cost_model": [1, 1, 0, 1, 1, 0],
            "n_graphs": n,
            "n_pairs": p,
            "n_zero_offdiag": 0,
            "n_certified": 0,
            "certification_rate": 0.0,
            "seconds_total": 0.0,
            "mean_seconds_per_pair": 0.0,
            "filter": {"min_nodes": 2, "require_connected": True, "n_max": None},
            "splits_merged": True,
            "gedlib_source": "jajupmochi/graphkit-learn",
            "code_commit": "test",
            "computed_utc": "2026-08-13T00:00:00Z",
            "schema_version": 1,
            "slurm_job_id": "test",
        }
    )


def _role_file(
    dataset: str,
    lb_values: np.ndarray,
    ub_values: np.ndarray,
    own: str,
    exact_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Assemble a CONTRACTS §4-shaped role file from recorded values.

    Parameters
    ----------
    dataset : str
        Dataset key.
    lb_values, ub_values : ndarray
        Flat upper-triangle lower and upper bounds.
    own : {"lb", "ub"}
        Which of the two becomes ``ged_matrix``, i.e. this file's own role.
    exact_arrays : dict
        T-03's computed file, source of the cohort columns.

    Returns
    -------
    dict
        Arrays ready for ``numpy.savez_compressed``.
    """
    n = int(exact_arrays["graph_ids"].shape[0])
    lb_matrix = _square(lb_values, n)
    ub_matrix = _square(ub_values, n)
    certified = np.abs(lb_matrix - ub_matrix) <= 1e-9
    np.fill_diagonal(certified, True)
    method, options = (
        ("BRANCH_FAST", "--threads 1") if own == "lb" else ("BIPARTITE", "--threads 1")
    )
    return {
        "ged_matrix": lb_matrix if own == "lb" else ub_matrix,
        "lb_matrix": lb_matrix,
        "ub_matrix": ub_matrix,
        "certified_mask": certified,
        "seconds_matrix": np.zeros((n, n), dtype=np.float32),
        "node_counts": exact_arrays["node_counts"].astype(np.int32),
        "edge_counts": exact_arrays["edge_counts"].astype(np.int32),
        "graph_ids": exact_arrays["graph_ids"],
        "labels": exact_arrays["labels"],
        "metadata": np.array(_metadata(dataset, own, method, options, n, int(lb_values.size))),
    }


def _campaign(tmp_path: Path, datasets: tuple[str, ...]) -> Path:
    """Materialise a whole campaign from T-27's and T-03's recorded files.

    Parameters
    ----------
    tmp_path : Path
        Root to write under.
    datasets : tuple of str
        Datasets to materialise.

    Returns
    -------
    Path
        The campaign root, containing ``LB/`` and ``UB/``.
    """
    root = tmp_path / "campaign"
    (root / "LB").mkdir(parents=True, exist_ok=True)
    (root / "UB").mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        with np.load(EXACT_DIR / f"{dataset}.npz", allow_pickle=False) as handle:
            exact_arrays = {k: handle[k] for k in handle.files}
        with np.load(T27_CELLS / f"{dataset}__BRANCH_FAST.npz") as handle:
            lb_values = handle["value"]
        with np.load(T27_CELLS / f"{dataset}__BIPARTITE.npz") as handle:
            ub_values = handle["value"]
        np.savez_compressed(
            root / "LB" / f"{dataset}.npz",
            **_role_file(dataset, lb_values, ub_values, "lb", exact_arrays),
        )
        np.savez_compressed(
            root / "UB" / f"{dataset}.npz",
            **_role_file(dataset, lb_values, ub_values, "ub", exact_arrays),
        )
    return root


def _rewrite(path: Path, **changes: np.ndarray) -> None:
    """Rewrite an ``.npz`` with some arrays replaced.

    Parameters
    ----------
    path : Path
        File to rewrite in place.
    **changes : ndarray
        Keys to replace. A key mapped to ``None`` is dropped.
    """
    with np.load(path, allow_pickle=False) as handle:
        arrays = {k: handle[k] for k in handle.files}
    for key, value in changes.items():
        if value is None:
            arrays.pop(key, None)
        else:
            arrays[key] = value
    np.savez_compressed(path, **arrays)


def _run(out: Path, gate: str, root: Path, **extra: str) -> tuple[int, dict[str, Any]]:
    """Run one gate and return its exit code and JSON record.

    Parameters
    ----------
    out : Path
        Report directory.
    gate : str
        Gate id.
    root : Path
        Campaign root.
    **extra : str
        Extra command-line flags, as ``flag_name=value``.

    Returns
    -------
    rc, record : int, dict
        Exit code and the parsed JSON record.
    """
    argv = [
        "--gate",
        gate,
        "--lb-dir",
        str(root / "LB"),
        "--ub-dir",
        str(root / "UB"),
        "--out",
        str(out),
    ]
    for key, value in extra.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    rc = gates.main(argv)
    record = json.loads((out / f"gate_{gate}.json").read_text())
    return rc, record


# --------------------------------------------------------------------------- G2


@needs_reference
def test_g2_passes_on_t27_recorded_values(tmp_path: Path) -> None:
    """G2 passes when the campaign reproduces T-27's census exactly."""
    root = _campaign(tmp_path, ("linux",))
    rc, record = _run(
        tmp_path / "r",
        "G2",
        root,
        datasets="linux",
        t27_cells=T27_CELLS,
        t27_index=T27_INDEX,
        exact_dir=EXACT_DIR,
    )
    assert rc == 0, record["notes"]
    assert record["passed"] is True
    assert record["n_violations"] == 0
    # both cells, all 3,916 LINUX pairs
    assert record["n_compared"] == 2 * 3916
    assert record["per_dataset"]["linux"]["precondition"].startswith("graph order matches")
    assert set(record["per_dataset"]["linux"]["graph_order_references"]) == {"t27_index", "exact"}


@needs_reference
def test_g2_fails_on_a_single_perturbed_entry(tmp_path: Path) -> None:
    """One changed bound in 3,916 pairs is caught, and its pair index is named."""
    root = _campaign(tmp_path, ("linux",))
    path = root / "LB" / "linux.npz"
    with np.load(path, allow_pickle=False) as handle:
        matrix = handle["ged_matrix"].copy()
    # Perturb symmetrically: the ONLY thing that should fail is the value
    # comparison, not the symmetry check G4 would otherwise also raise.
    matrix[3, 7] += 1.0
    matrix[7, 3] += 1.0
    _rewrite(path, ged_matrix=matrix)

    rc, record = _run(
        tmp_path / "r",
        "G2",
        root,
        datasets="linux",
        t27_cells=T27_CELLS,
        t27_index=T27_INDEX,
        exact_dir=EXACT_DIR,
    )
    assert rc == 1
    assert record["n_violations"] == 1
    violation = record["violations"][0]
    assert violation["pair"] == [3, 7]
    assert violation["dataset"] == "linux"
    assert violation["difference"] == pytest.approx(1.0)
    assert record["per_dataset"]["linux"]["BRANCH_FAST"]["n_mismatched"] == 1
    # the untouched cell still reconciles
    assert record["per_dataset"]["linux"]["BIPARTITE"]["n_mismatched"] == 0


@needs_reference
def test_g2_reports_graph_order_before_comparing_values(tmp_path: Path) -> None:
    """A misordered cohort reports as graph order, not as 3,916 wrong bounds."""
    root = _campaign(tmp_path, ("linux",))
    with np.load(root / "LB" / "linux.npz", allow_pickle=False) as handle:
        ids = handle["graph_ids"].copy()
    ids[0], ids[1] = ids[1], ids[0]
    _rewrite(root / "LB" / "linux.npz", graph_ids=ids)

    rc, record = _run(
        tmp_path / "r",
        "G2",
        root,
        datasets="linux",
        t27_cells=T27_CELLS,
        t27_index=T27_INDEX,
        exact_dir=EXACT_DIR,
    )
    assert rc == 1
    assert any("graph order differs" in note for note in record["notes"])
    assert record["per_dataset"]["linux"]["precondition"] == "FAILED: graph order"
    # The precondition short-circuits: no value comparison was attempted.
    assert record["n_compared"] == 0
    assert record["violations"] == []


@needs_reference
def test_g2_cannot_pass_without_a_graph_order_reference(tmp_path: Path) -> None:
    """With no reference cohort the precondition is unevaluable, so G2 fails."""
    root = _campaign(tmp_path, ("linux",))
    rc, record = _run(tmp_path / "r", "G2", root, datasets="linux", t27_cells=T27_CELLS)
    assert rc == 1
    assert any("cannot be evaluated" in note for note in record["notes"])


@needs_reference
def test_g2_skips_datasets_whose_cohort_differs(tmp_path: Path) -> None:
    """Non-G2 datasets are named as skipped rather than silently ignored."""
    root = _campaign(tmp_path, ("linux",))
    _, record = _run(
        tmp_path / "r",
        "G2",
        root,
        datasets="linux,coil_del,aids_graphedx",
        t27_cells=T27_CELLS,
        t27_index=T27_INDEX,
        exact_dir=EXACT_DIR,
    )
    note = next(n for n in record["notes"] if "not compared" in n)
    assert "coil_del" in note and "aids_graphedx" in note


@needs_reference
@pytest.mark.slow
def test_g2_full_coverage_is_3_602_615_pairs(tmp_path: Path) -> None:
    """All four G2 datasets together cover exactly the documented pair count."""
    root = _campaign(tmp_path, gates.G2_DATASETS)
    rc, record = _run(
        tmp_path / "r",
        "G2",
        root,
        datasets=",".join(gates.G2_DATASETS),
        t27_cells=T27_CELLS,
        t27_index=T27_INDEX,
        exact_dir=EXACT_DIR,
    )
    assert rc == 0, record["notes"]
    coverage = record["per_dataset"]["_coverage"]
    assert coverage["pairs_per_cell"] == gates.G2_FULL_PAIRS == 3_602_615
    assert coverage["complete"] is True
    assert record["n_compared"] == 2 * 3_602_615


# --------------------------------------------------------------------------- G3


@needs_reference
def test_g3_passes_on_the_real_bracket(tmp_path: Path) -> None:
    """BRANCH_FAST <= BIPARTITE, and both bracket T-03's certified exact values."""
    root = _campaign(tmp_path, ("linux",))
    rc, record = _run(tmp_path / "r", "G3", root, datasets="linux", exact_dir=EXACT_DIR)
    assert rc == 0, record["notes"]
    detail = record["per_dataset"]["linux"]
    assert detail["bracket"]["n_pairs"] == 3916
    assert detail["bracket"]["n_inverted"] == 0
    # LINUX has 46 censored pairs in T-03's census; certified selects the other 3,870.
    assert detail["exact"]["n_certified"] == 3870
    assert detail["exact"]["n_lb_above_exact"] == 0
    assert detail["exact"]["n_ub_below_exact"] == 0


@needs_reference
def test_g3_catches_an_inverted_bracket(tmp_path: Path) -> None:
    """lb > ub on one pair is caught and the pair is named."""
    root = _campaign(tmp_path, ("linux",))
    with np.load(root / "LB" / "linux.npz", allow_pickle=False) as handle:
        lb = handle["lb_matrix"].copy()
    lb[5, 11] += 100.0
    lb[11, 5] += 100.0
    _rewrite(root / "LB" / "linux.npz", lb_matrix=lb)

    rc, record = _run(tmp_path / "r", "G3", root, datasets="linux", exact_dir=EXACT_DIR)
    assert rc == 1
    inverted = [v for v in record["violations"] if v["comparison"] == "lb <= ub"]
    assert len(inverted) == 1
    assert inverted[0]["pair"] == [5, 11]


@needs_reference
def test_g3_catches_an_upper_bound_below_exact(tmp_path: Path) -> None:
    """ub < exact on a certified pair breaks the only claim the paper rests on."""
    root = _campaign(tmp_path, ("linux",))
    with np.load(EXACT_DIR / "linux.npz", allow_pickle=False) as handle:
        certified = handle["certified_mask"]
        exact = handle["ged_matrix"]
    rows, cols = np.triu_indices(certified.shape[0], k=1)
    sel = np.flatnonzero(certified[rows, cols] & (exact[rows, cols] > 0))
    i, j = int(rows[sel[0]]), int(cols[sel[0]])

    with np.load(root / "UB" / "linux.npz", allow_pickle=False) as handle:
        ub = handle["ub_matrix"].copy()
    ub[i, j] = ub[j, i] = 0.0
    _rewrite(root / "UB" / "linux.npz", ub_matrix=ub)

    rc, record = _run(tmp_path / "r", "G3", root, datasets="linux", exact_dir=EXACT_DIR)
    assert rc == 1
    below = [v for v in record["violations"] if v["comparison"] == "exact <= ub"]
    assert below and below[0]["pair"] == [i, j]


@needs_reference
def test_g3_refuses_to_compare_a_different_cohort_positionally(tmp_path: Path) -> None:
    """aids_graphedx (819) must never be compared by index to aids (769)."""
    root = _campaign(tmp_path, ("linux",))
    with np.load(root / "LB" / "linux.npz", allow_pickle=False) as handle:
        ids = handle["graph_ids"].copy()
    ids[:] = np.array([f"other_{k}" for k in range(ids.shape[0])], dtype=ids.dtype)
    _rewrite(root / "LB" / "linux.npz", graph_ids=ids)

    rc, record = _run(tmp_path / "r", "G3", root, datasets="linux", exact_dir=EXACT_DIR)
    # The bracket arm still runs and passes; only the exact arm is skipped.
    assert rc == 0
    assert record["per_dataset"]["linux"]["exact"].startswith("SKIPPED")


@needs_reference
def test_g3_without_exact_dir_runs_only_the_bracket(tmp_path: Path) -> None:
    """The missing reference is recorded, not silently treated as a pass."""
    root = _campaign(tmp_path, ("linux",))
    rc, record = _run(tmp_path / "r", "G3", root, datasets="linux")
    assert rc == 0
    assert any("only lb <= ub" in note for note in record["notes"])


# --------------------------------------------------------------------------- G4


@needs_reference
def test_g4_passes_on_a_wellformed_file(tmp_path: Path) -> None:
    """A contract-shaped file built from recorded values passes every check."""
    root = _campaign(tmp_path, ("linux",))
    rc, record = _run(tmp_path / "r", "G4", root, datasets="linux")
    assert rc == 0, record["notes"]
    detail = record["per_dataset"]["lb:linux"]
    assert detail["missing_keys"] == [] and detail["wrong_dtypes"] == []
    assert detail["ged_matrix"]["max_asymmetry"] == 0.0
    assert detail["ged_matrix"]["n_nonfinite"] == 0
    assert detail["certified_diagonal_all_true"] is True


@needs_reference
def test_g4_accepts_an_all_empty_labels_column(tmp_path: Path) -> None:
    """LINUX and AIDS-GraphEdX legitimately ship no class labels.

    G4 checks that ``labels`` is present with the stated dtype and asserts
    nothing about its contents. The Suite-2 class counts were raw dataset
    counts, not post-filter counts, so a non-emptiness check would fail on
    correct data.
    """
    root = _campaign(tmp_path, ("linux",))
    rc, record = _run(tmp_path / "r", "G4", root, datasets="linux")
    assert rc == 0
    assert record["per_dataset"]["lb:linux"]["labels_all_empty"] is True


@needs_reference
def test_g4_catches_an_all_zero_matrix(tmp_path: Path) -> None:
    """The GEDLIB wrong-accessor signature: 0.00 everywhere, no exception."""
    root = _campaign(tmp_path, ("linux",))
    with np.load(root / "LB" / "linux.npz", allow_pickle=False) as handle:
        shape = handle["ged_matrix"].shape
    _rewrite(root / "LB" / "linux.npz", ged_matrix=np.zeros(shape, dtype=np.float64))

    rc, record = _run(tmp_path / "r", "G4", root, datasets="linux")
    assert rc == 1
    assert record["per_dataset"]["lb:linux"]["ged_matrix"]["offdiag_zero_fraction"] == 1.0
    assert any("wrong accessor" in note for note in record["notes"])


@needs_reference
def test_g4_catches_an_asymmetric_matrix(tmp_path: Path) -> None:
    """An upper bound filled in one orientation is not a distance matrix."""
    root = _campaign(tmp_path, ("linux",))
    with np.load(root / "UB" / "linux.npz", allow_pickle=False) as handle:
        matrix = handle["ged_matrix"].copy()
    matrix[2, 9] += 3.0  # one side only
    _rewrite(root / "UB" / "linux.npz", ged_matrix=matrix)

    rc, record = _run(tmp_path / "r", "G4", root, datasets="linux")
    assert rc == 1
    assert record["per_dataset"]["ub:linux"]["ged_matrix"]["max_asymmetry"] == 3.0
    assert any("not symmetric" in note for note in record["notes"])


@needs_reference
def test_g4_catches_a_nonzero_diagonal(tmp_path: Path) -> None:
    """d(G, G) = 0 is not negotiable."""
    root = _campaign(tmp_path, ("linux",))
    with np.load(root / "LB" / "linux.npz", allow_pickle=False) as handle:
        matrix = handle["ged_matrix"].copy()
    matrix[4, 4] = 2.0
    _rewrite(root / "LB" / "linux.npz", ged_matrix=matrix)

    rc, record = _run(tmp_path / "r", "G4", root, datasets="linux")
    assert rc == 1
    assert any("diagonal is not zero" in note for note in record["notes"])


@needs_reference
def test_g4_catches_a_missing_key(tmp_path: Path) -> None:
    """All ten CONTRACTS §4 keys, or the file is not readable by one loader."""
    root = _campaign(tmp_path, ("linux",))
    _rewrite(root / "LB" / "linux.npz", seconds_matrix=None)

    rc, record = _run(tmp_path / "r", "G4", root, datasets="linux")
    assert rc == 1
    assert record["per_dataset"]["lb:linux"]["missing_keys"] == ["seconds_matrix"]


@needs_reference
def test_g4_catches_a_wrong_dtype(tmp_path: Path) -> None:
    """float32 where the contract says float64 breaks the shared loader."""
    root = _campaign(tmp_path, ("linux",))
    with np.load(root / "LB" / "linux.npz", allow_pickle=False) as handle:
        matrix = handle["ged_matrix"].astype(np.float32)
    _rewrite(root / "LB" / "linux.npz", ged_matrix=matrix)

    rc, record = _run(tmp_path / "r", "G4", root, datasets="linux")
    assert rc == 1
    assert any("float32 != float64" in w for w in record["per_dataset"]["lb:linux"]["wrong_dtypes"])


@needs_reference
def test_g4_catches_a_false_certified_diagonal(tmp_path: Path) -> None:
    """certified_mask must be True on the diagonal: GED(G, G) = 0 is proven."""
    root = _campaign(tmp_path, ("linux",))
    with np.load(root / "LB" / "linux.npz", allow_pickle=False) as handle:
        mask = handle["certified_mask"].copy()
    mask[6, 6] = False
    _rewrite(root / "LB" / "linux.npz", certified_mask=mask)

    rc, record = _run(tmp_path / "r", "G4", root, datasets="linux")
    assert rc == 1
    assert any("diagonal is not all True" in note for note in record["notes"])


@needs_reference
def test_g4_catches_a_self_reported_certified_mask(tmp_path: Path) -> None:
    """certified_mask is derived from lb == ub, never sourced from a backend."""
    root = _campaign(tmp_path, ("linux",))
    with np.load(root / "LB" / "linux.npz", allow_pickle=False) as handle:
        mask = handle["certified_mask"].copy()
    mask[1, 2] = mask[2, 1] = not bool(mask[1, 2])
    _rewrite(root / "LB" / "linux.npz", certified_mask=mask)

    rc, record = _run(tmp_path / "r", "G4", root, datasets="linux")
    assert rc == 1
    assert record["per_dataset"]["lb:linux"]["certified_mask_disagreements"] == 2


@needs_reference
def test_g4_catches_an_empty_options_string(tmp_path: Path) -> None:
    """The options string is part of the method name (T-27 §4.2)."""
    root = _campaign(tmp_path, ("linux",))
    with np.load(root / "LB" / "linux.npz", allow_pickle=False) as handle:
        metadata = json.loads(str(handle["metadata"]))
    metadata["options_string"] = ""
    _rewrite(root / "LB" / "linux.npz", metadata=np.array(json.dumps(metadata)))

    rc, record = _run(tmp_path / "r", "G4", root, datasets="linux")
    assert rc == 1
    assert any("options_string is empty" in note for note in record["notes"])


@needs_reference
def test_g4_catches_a_missing_file(tmp_path: Path) -> None:
    """A dataset that never landed is a failure, not an absence."""
    root = _campaign(tmp_path, ("linux",))
    rc, record = _run(tmp_path / "r", "G4", root, datasets="linux,mutagenicity")
    assert rc == 1
    assert record["per_dataset"]["lb:mutagenicity"]["error"] == "missing"


# --------------------------------------------------------------------------- reads


def test_all_zero_reads_are_the_wrong_accessor_signature() -> None:
    """get_lower_bound() on an upper-bound method returns 0.00 and never raises."""
    problems = gates._assert_reads_sane(np.zeros(500), "probe")
    assert any("wrong-accessor signature" in p for p in problems)


def test_legitimate_zeros_are_accepted() -> None:
    """A per-pair ``value > 0`` assertion would fail on correct data.

    GED is legitimately zero for isomorphic graphs. T-03's recorded
    ``iam_letter_low`` census has 215,968 exactly-zero off-diagonal entries out
    of 1,391,220 -- 15.5 %. The check is therefore "not identically zero across
    the block", not "positive on every pair".
    """
    values = np.zeros(1000)
    values[:845] = 3.0  # 15.5 % zeros, as Letter LOW really has
    assert gates._assert_reads_sane(values, "probe") == []


def test_non_finite_and_negative_reads_are_rejected() -> None:
    """HED returns get_upper_bound() = inf without raising."""
    assert any(
        "non-finite" in p for p in gates._assert_reads_sane(np.array([1.0, np.inf, 2.0]), "probe")
    )
    assert any(
        "negative" in p for p in gates._assert_reads_sane(np.array([1.0, -2.0, 3.0]), "probe")
    )


def test_graphs_from_export_reads_the_csr_schema() -> None:
    """CONTRACTS §2's CSR export round-trips into NetworkX graphs."""
    pytest.importorskip("networkx")
    arrays = {
        "n_nodes": np.array([3, 2], dtype=np.int32),
        "edge_offsets": np.array([0, 2, 3], dtype=np.int64),
        "edges": np.array([[0, 1, 0], [1, 2, 1]], dtype=np.int32),
    }
    graphs = gates._graphs_from_export(arrays)
    assert [g.number_of_nodes() for g in graphs] == [3, 2]
    assert [g.number_of_edges() for g in graphs] == [2, 1]
    assert sorted(graphs[0].edges()) == [(0, 1), (1, 2)]
    # GEDLIB's GXL bindings require string attributes.
    assert all(isinstance(v, str) for _, v in graphs[0].nodes(data="l"))


# --------------------------------------------------------------------------- lb


@needs_reference
@needs_gedlib
def test_lb_consistency_reproduces_t27_branch_fast(tmp_path: Path) -> None:
    """Recomputing BRANCH_FAST from the export reproduces the recorded census.

    End-to-end against the real solver: the gate rebuilds LINUX's graphs from
    the CONTRACTS §2 CSR export, runs BRANCH_FAST through GEDLIB directly under
    cost model D6, and compares to T-27's recorded values at exact equality.
    """
    root = _campaign(tmp_path, ("linux",))
    rc, record = _run(
        tmp_path / "r",
        "lb-consistency",
        root,
        datasets="linux",
        input_dir=EXPORTED,
        sample_size=400,
        seed=42,
    )
    assert rc == 0, record["notes"]
    detail = record["per_dataset"]["linux"]
    assert detail["n_sampled"] == 400
    assert detail["n_mismatched"] == 0
    assert record["tolerance"] == "exact equality"


@needs_reference
@needs_gedlib
def test_lb_consistency_catches_a_perturbed_lower_bound(tmp_path: Path) -> None:
    """A campaign whose LB drifted from BRANCH_FAST is caught by resampling."""
    root = _campaign(tmp_path, ("linux",))
    with np.load(root / "LB" / "linux.npz", allow_pickle=False) as handle:
        matrix = handle["ged_matrix"].copy()
    matrix += 1.0  # every off-diagonal bound is now wrong
    np.fill_diagonal(matrix, 0.0)
    _rewrite(root / "LB" / "linux.npz", ged_matrix=matrix)

    rc, record = _run(
        tmp_path / "r",
        "lb-consistency",
        root,
        datasets="linux",
        input_dir=EXPORTED,
        sample_size=200,
        seed=42,
    )
    assert rc == 1
    assert record["per_dataset"]["linux"]["n_mismatched"] == 200
    assert record["violations"][0]["difference"] == pytest.approx(1.0)


@needs_reference
@needs_gedlib
def test_lb_consistency_draw_is_reproducible_from_the_seed(tmp_path: Path) -> None:
    """The sampled pairs depend on the seed alone."""
    root = _campaign(tmp_path, ("linux",))
    first = _run(
        tmp_path / "a",
        "lb-consistency",
        root,
        datasets="linux",
        input_dir=EXPORTED,
        sample_size=100,
        seed=42,
    )[1]
    second = _run(
        tmp_path / "b",
        "lb-consistency",
        root,
        datasets="linux",
        input_dir=EXPORTED,
        sample_size=100,
        seed=42,
    )[1]
    assert first["per_dataset"] == second["per_dataset"]


# --------------------------------------------------------------------------- records


@needs_reference
def test_every_gate_record_names_its_tolerance_and_why(tmp_path: Path) -> None:
    """A verdict without its tolerance is not auditable."""
    root = _campaign(tmp_path, ("linux",))
    for gate in ("G2", "G3", "G4"):
        extra = {"datasets": "linux"}
        if gate == "G2":
            extra |= {"t27_cells": T27_CELLS, "exact_dir": EXACT_DIR}
        if gate == "G3":
            extra |= {"exact_dir": EXACT_DIR}
        _, record = _run(tmp_path / gate, gate, root, **extra)
        assert record["tolerance"], gate
        assert len(record["tolerance_rationale"]) > 80, gate
        assert record["schema_version"] == gates.SCHEMA_VERSION
        assert "environment" in record and "invocation" in record


def test_unevaluable_gate_fails_rather_than_passing_vacuously(tmp_path: Path) -> None:
    """A gate that cannot run must not report success."""
    rc = gates.main(["--gate", "G2", "--out", str(tmp_path)])
    assert rc == 1
    record = json.loads((tmp_path / "gate_G2.json").read_text())
    assert record["passed"] is False
    assert any("could not evaluate" in note for note in record["notes"])


# --------------------------------------------------------------------------- launcher


def _launcher(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the launcher and capture its output.

    Parameters
    ----------
    *args : str
        Launcher flags.

    Returns
    -------
    subprocess.CompletedProcess
        The finished process.
    """
    return subprocess.run(
        ["bash", str(LAUNCHER), *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
    )


def test_launcher_dry_run_issues_no_sbatch() -> None:
    """--dry-run prints every sbatch line it would issue, and issues none."""
    result = _launcher("--dry-run", "--stage", "all")
    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    for name in ("aged-lb", "aged-ub", "aged-ubs", "aged-ubt", "aged-crossfill"):
        assert f"--job-name={name}" in combined, name
    assert combined.count("[DRY-RUN] sbatch") == 5
    # Resource flags resolved, not templated.
    assert "--constraint=sr" in combined
    assert "--account=tic_163_uma" in combined
    assert "--dependency=afterok:" in combined


def test_launcher_resolves_the_documented_core_counts() -> None:
    """At the CONTRACTS §3 projected rates the sizing is 1 / 2 / 9 / 31 cores."""
    result = _launcher("--dry-run", "--stage", "all")
    combined = result.stdout + result.stderr
    for name, cores, wall in (
        ("aged-lb", 1, "0-12:00:00"),
        ("aged-ub", 2, "0-12:00:00"),
        ("aged-ubs", 9, "0-12:00:00"),
        ("aged-ubt", 31, "1-00:00:00"),
    ):
        assert f"{name}: {cores} cores" in combined, name
        assert f"--job-name={name} --account=tic_163_uma --time={wall}" in combined


def test_launcher_refuses_a_job_under_the_two_hour_floor() -> None:
    """A projection under FLOOR_SECONDS is refused, not submitted short."""
    result = _launcher("--dry-run", "--stage", "lb", "--rate-lb", "0.00001")
    assert result.returncode == 3
    assert "under the 7200s floor" in result.stderr
    assert "Submitting short is not one of them" in result.stderr
    # It refused BEFORE emitting an sbatch line for that job.
    assert "--job-name=aged-lb " not in (result.stdout + result.stderr)


def test_launcher_probe_stage_refuses_and_submits_nothing() -> None:
    """T-05-design §5: a standalone probe job would itself violate the floor."""
    result = _launcher("--dry-run", "--stage", "probe")
    assert result.returncode == 3
    assert "[DRY-RUN] sbatch" not in result.stdout
    assert "under the 7200s floor" in result.stderr


def test_launcher_group_merges_roles_into_one_job() -> None:
    """--group makes 'merge the role into an adjacent job' executable."""
    result = _launcher("--dry-run", "--stage", "all", "--group", "lb,ub")
    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert "--job-name=aged-lb-ub" in combined
    assert "ROLES=lb:ub" in combined  # colon-separated: --export splits on commas
    assert "--job-name=aged-lb " not in combined
    assert combined.count("[DRY-RUN] sbatch") == 4  # one fewer job than ungrouped
    # core-seconds add: 12,240 + 30,240 = 42,480 -> floor(42480/10800) = 3
    assert "aged-lb-ub: 3 cores" in combined


def test_launcher_warns_when_sizing_from_the_flat_projection() -> None:
    """An n-bar projection under-estimates on a cohort reaching n = 98."""
    result = _launcher("--dry-run", "--stage", "lb")
    combined = result.stdout + result.stderr
    assert "WARNING: sizing WITHOUT the per-bin table" in combined
    assert "LOWER bound on true cost" in combined
    assert "evidence=lb:projected" in combined


def test_launcher_rejects_an_unknown_stage() -> None:
    """A typo in --stage must not silently run everything."""
    result = _launcher("--dry-run", "--stage", "lbb")
    assert result.returncode == 2
    assert "bad --stage" in result.stderr


def test_workers_carry_no_sbatch_header() -> None:
    """Every #SBATCH flag lives on the launcher's command line."""
    for worker in sorted((REPO_ROOT / "slurm" / "approx_ged").glob("worker_*.sh")):
        assert "#SBATCH" not in worker.read_text(), worker.name


def test_launcher_does_not_use_the_bash_builtin_GROUPS() -> None:
    """`GROUPS` is a bash builtin array; assigning to it fails with rc=1.

    Under ``set -e`` that kills the launcher during flag parsing, or leaves the
    array empty so ``--group`` is silently ignored and four jobs are submitted
    where two were asked for. Measured on bash 5.2.15.
    """
    text = LAUNCHER.read_text()
    # A bare GROUPS assignment, not preceded by an identifier character -- so
    # ROLE_GROUPS=() and ROLE_GROUPS+=() do not match.
    bare = re.findall(r"(?<![A-Za-z0-9_])GROUPS\+?=", text)
    assert bare == [], f"bare GROUPS assignment(s): {bare}"
    assert "ROLE_GROUPS=()" in text
