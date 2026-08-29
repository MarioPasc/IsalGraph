"""T-28: the alternative similarity references must stay OUT of the frozen family.

``N_actual = 79`` and the Benjamini-Hochberg correction over it are pre-registered.
Family membership was decided by ``comparator.representation`` alone, so before
T-28 added its guard, **every reference added later would have entered the family
as a ``B1a`` row and inflated the cardinality with no error raised**. These tests
pin the guard, the regime label, and the structural gates on the new reference
matrices.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.real_data.eval_stats import t06_f2, t06_f2_inputs

T28_KEYS = ("wl", "spectral", "spectral_comb", "spectral_adj", "spectral_esd")


def _arm(name: str, n: int, rng: np.random.Generator) -> t06_f2_inputs.ArmMatrices:
    """Build an arm carrying a symmetric, zero-diagonal random distance."""
    raw = rng.random((n, n))
    distance = np.triu(raw, 1)
    distance = distance + distance.T
    return t06_f2_inputs.ArmMatrices(
        representation=name,
        metric="levenshtein",
        distance=distance,
        defined=np.ones((n, n), dtype=bool),
        size_null=np.zeros((n, n)),
        graph_ids=np.array([f"g{i}" for i in range(n)]),
        node_counts=np.arange(1, n + 1, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def test_only_the_ged_references_may_carry_a_confirmatory_cell() -> None:
    """The frozen set is exactly the three GED references, and nothing else."""
    assert frozenset({"exact", "lb", "ub"}) == t06_f2.CONFIRMATORY_REFERENCES
    for key in T28_KEYS:
        assert key not in t06_f2.CONFIRMATORY_REFERENCES


@pytest.mark.parametrize(
    ("reference", "expected"),
    [("exact", "exact"), ("lb", "approximate"), ("ub", "approximate")]
    + [(key, "structural") for key in T28_KEYS],
)
def test_the_regime_label_separates_the_three_reference_families(
    reference: str, expected: str
) -> None:
    """A T-28 reference is neither exact nor bracketed, so it never merges."""
    assert t06_f2._reference_regime(reference) == expected


def test_a_t28_reference_never_produces_a_family_row() -> None:
    """The end-to-end guard: same comparator, two references, one family row.

    ``min_dfs`` is a frozen family comparator, so under ``exact`` it must yield
    ``in_family`` with row ``B1e``. Under ``wl`` -- the *same* comparator -- it
    must yield no row at all, or ``N_actual`` silently leaves 79.
    """
    rng = np.random.default_rng(0)
    n = 12
    arm = _arm("isalgraph_pruned", n, rng)
    comparator = _arm("min_dfs", n, rng)
    reference = _arm("ref", n, rng).distance

    records = t06_f2.run_correlation_group(
        suite="suite1",
        dataset="linux",  # a real key: bootstrap_tier only knows the frozen D15 table
        view="all_pairs",
        arm=arm,
        group=t06_f2.CorrelationGroup(digest="d0", comparators=[comparator]),
        references={"exact": reference, "wl": reference.copy()},
        replicates=8,
    )

    by_reference = {r.reference: r for r in records if r.representation == "min_dfs"}
    assert by_reference["exact"].in_family is True
    assert by_reference["exact"].row == "B1e"
    assert by_reference["wl"].in_family is False
    assert by_reference["wl"].row is None
    assert by_reference["wl"].regime == "structural"


# ---------------------------------------------------------------------------
# Loading and the structural gates
# ---------------------------------------------------------------------------


def _write_reference(path: Path, matrix: np.ndarray, ids: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        distance_matrix=matrix,
        graph_ids=ids,
        node_counts=np.ones(len(ids), dtype=np.int32),
        defined_mask=np.ones(matrix.shape, dtype=bool),
        metadata=json.dumps({"ticket": "T-28"}),
    )


def test_an_unset_root_reproduces_t06_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gate G1: with no tree configured, nothing new is loaded."""
    monkeypatch.setattr(t06_f2_inputs, "T28_REFERENCE_ROOT", "")
    ids = np.array(["g0", "g1", "g2"])
    assert t06_f2_inputs._load_t28_references("suite1", "aids", ids) == {}


def test_a_configured_root_is_loaded_and_keyed_by_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reference token after the double underscore becomes the key."""
    ids = np.array(["g0", "g1", "g2"])
    matrix = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    _write_reference(tmp_path / "suite1" / "aids__spectral.npz", matrix, ids)
    monkeypatch.setattr(t06_f2_inputs, "T28_REFERENCE_ROOT", str(tmp_path))

    loaded = t06_f2_inputs._load_t28_references("suite1", "aids", ids)

    assert set(loaded) == {"spectral"}
    np.testing.assert_allclose(loaded["spectral"], matrix)


def test_an_almost_all_zero_reference_aborts_rather_than_propagating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The silent-zero failure shape must abort the campaign, not fill a table.

    A per-pair ``value > 0`` rule is wrong -- GED is legitimately 0 for
    isomorphic graphs -- so the guard is on the matrix-wide off-diagonal zero
    fraction, exactly as the GEDLIB correction of 2026-08-15 prescribes.
    """
    ids = np.array([f"g{i}" for i in range(20)])
    matrix = np.zeros((20, 20))
    matrix[0, 1] = matrix[1, 0] = 5.0
    _write_reference(tmp_path / "suite1" / "aids__spectral.npz", matrix, ids)
    monkeypatch.setattr(t06_f2_inputs, "T28_REFERENCE_ROOT", str(tmp_path))

    with pytest.raises(ValueError, match="silent-zero failure shape"):
        t06_f2_inputs._load_t28_references("suite1", "aids", ids)


def test_a_legitimate_zero_does_not_trip_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Isomorphic graphs give distance 0 and must survive; only 0.99+ aborts."""
    ids = np.array([f"g{i}" for i in range(10)])
    rng = np.random.default_rng(1)
    raw = np.triu(rng.random((10, 10)) + 0.1, 1)
    matrix = raw + raw.T
    matrix[0, 1] = matrix[1, 0] = 0.0  # a genuinely isomorphic pair
    _write_reference(tmp_path / "suite1" / "aids__spectral.npz", matrix, ids)
    monkeypatch.setattr(t06_f2_inputs, "T28_REFERENCE_ROOT", str(tmp_path))

    loaded = t06_f2_inputs._load_t28_references("suite1", "aids", ids)

    assert loaded["spectral"][0, 1] == 0.0


def test_a_non_finite_reference_aborts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``HED`` returns ``inf`` without raising; a reference matrix must not."""
    ids = np.array(["g0", "g1", "g2"])
    matrix = np.array([[0.0, 1.0, np.inf], [1.0, 0.0, 3.0], [np.inf, 3.0, 0.0]])
    _write_reference(tmp_path / "suite1" / "aids__spectral.npz", matrix, ids)
    monkeypatch.setattr(t06_f2_inputs, "T28_REFERENCE_ROOT", str(tmp_path))

    with pytest.raises(ValueError, match="non-finite"):
        t06_f2_inputs._load_t28_references("suite1", "aids", ids)
