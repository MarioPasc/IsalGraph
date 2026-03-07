"""Validate metric properties of GED and Levenshtein matrices.

A distance function d(x,y) must satisfy:
1. d(x,x) = 0 (identity)
2. d(x,y) = d(y,x) (symmetry)
3. d(x,y) >= 0 (non-negativity)
4. d(x,z) <= d(x,y) + d(y,z) (triangle inequality)
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.eval_validation.conftest import DATASETS, METHODS, EvalData

pytestmark = pytest.mark.eval_validation

TRIANGLE_SAMPLE_SIZE = 5000


class TestGEDMetricProperties:
    """GED matrices must satisfy metric axioms."""

    @pytest.mark.parametrize("dataset", DATASETS)
    def test_diagonal_zero(self, eval_data: EvalData, dataset: str):
        if dataset not in eval_data.ged_matrices:
            pytest.skip(f"No GED matrix for {dataset}")
        ged = eval_data.ged_matrices[dataset]
        diag = np.diag(ged)
        assert np.allclose(diag, 0.0), f"Non-zero diagonal entries: {diag[diag != 0]}"

    @pytest.mark.parametrize("dataset", DATASETS)
    def test_symmetry(self, eval_data: EvalData, dataset: str):
        if dataset not in eval_data.ged_matrices:
            pytest.skip(f"No GED matrix for {dataset}")
        ged = eval_data.ged_matrices[dataset]
        finite = np.isfinite(ged) & np.isfinite(ged.T)
        diff = np.abs(ged[finite] - ged.T[finite])
        max_diff = float(diff.max()) if diff.size > 0 else 0.0
        assert max_diff < 1e-10, f"Max asymmetry: {max_diff}"

    @pytest.mark.parametrize("dataset", DATASETS)
    def test_non_negative(self, eval_data: EvalData, dataset: str):
        if dataset not in eval_data.ged_matrices:
            pytest.skip(f"No GED matrix for {dataset}")
        ged = eval_data.ged_matrices[dataset]
        finite = ged[np.isfinite(ged)]
        n_neg = int((finite < -1e-10).sum())
        assert n_neg == 0, f"{n_neg} negative GED entries"

    @pytest.mark.parametrize("dataset", DATASETS)
    def test_triangle_inequality(self, eval_data: EvalData, dataset: str):
        if dataset not in eval_data.ged_matrices:
            pytest.skip(f"No GED matrix for {dataset}")
        ged = eval_data.ged_matrices[dataset]
        n = ged.shape[0]
        if n < 3:
            pytest.skip("Fewer than 3 graphs")

        rng = np.random.default_rng(42)
        violations = 0
        n_checked = 0
        for _ in range(TRIANGLE_SAMPLE_SIZE):
            i, j, k = rng.choice(n, size=3, replace=False)
            if np.isfinite(ged[i, j]) and np.isfinite(ged[j, k]) and np.isfinite(ged[i, k]):
                if ged[i, k] > ged[i, j] + ged[j, k] + 1e-10:
                    violations += 1
                n_checked += 1

        assert violations == 0, (
            f"{violations}/{n_checked} triangle inequality violations "
            f"(sampled {TRIANGLE_SAMPLE_SIZE} triples)"
        )


class TestLevenshteinMetricProperties:
    """Levenshtein matrices must satisfy metric axioms."""

    @pytest.mark.parametrize("dataset", DATASETS)
    @pytest.mark.parametrize("method", METHODS)
    def test_diagonal_zero(self, eval_data: EvalData, dataset: str, method: str):
        key = (dataset, method)
        if key not in eval_data.levenshtein_matrices:
            pytest.skip(f"No Levenshtein matrix for {dataset}/{method}")
        lev = eval_data.levenshtein_matrices[key]
        diag = np.diag(lev)
        valid_diag = diag[diag >= 0]
        assert np.all(valid_diag == 0), f"Non-zero diagonal: {valid_diag[valid_diag != 0]}"

    @pytest.mark.parametrize("dataset", DATASETS)
    @pytest.mark.parametrize("method", METHODS)
    def test_symmetry(self, eval_data: EvalData, dataset: str, method: str):
        key = (dataset, method)
        if key not in eval_data.levenshtein_matrices:
            pytest.skip(f"No Levenshtein matrix for {dataset}/{method}")
        lev = eval_data.levenshtein_matrices[key]
        valid = (lev >= 0) & (lev.T >= 0)
        assert np.array_equal(lev[valid], lev.T[valid]), "Levenshtein matrix not symmetric"

    @pytest.mark.parametrize("dataset", DATASETS)
    @pytest.mark.parametrize("method", METHODS)
    def test_non_negative(self, eval_data: EvalData, dataset: str, method: str):
        key = (dataset, method)
        if key not in eval_data.levenshtein_matrices:
            pytest.skip(f"No Levenshtein matrix for {dataset}/{method}")
        lev = eval_data.levenshtein_matrices[key]
        # -1 is sentinel for missing; all valid entries must be >= 0
        bad = lev[(lev < 0) & (lev != -1)]
        assert len(bad) == 0, f"Unexpected negative entries: {bad[:10]}"

    @pytest.mark.parametrize("dataset", DATASETS)
    @pytest.mark.parametrize("method", METHODS)
    def test_triangle_inequality(self, eval_data: EvalData, dataset: str, method: str):
        key = (dataset, method)
        if key not in eval_data.levenshtein_matrices:
            pytest.skip(f"No Levenshtein matrix for {dataset}/{method}")
        lev = eval_data.levenshtein_matrices[key]
        n = lev.shape[0]
        if n < 3:
            pytest.skip("Fewer than 3 graphs")

        rng = np.random.default_rng(42)
        violations = 0
        n_checked = 0
        for _ in range(TRIANGLE_SAMPLE_SIZE):
            i, j, k = rng.choice(n, size=3, replace=False)
            if lev[i, j] >= 0 and lev[j, k] >= 0 and lev[i, k] >= 0:
                if lev[i, k] > lev[i, j] + lev[j, k]:
                    violations += 1
                n_checked += 1

        assert violations == 0, (
            f"{violations}/{n_checked} triangle inequality violations "
            f"(sampled {TRIANGLE_SAMPLE_SIZE} triples)"
        )
