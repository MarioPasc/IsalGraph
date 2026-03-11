"""Tests for WL kernel distance properties on real evaluation data.

Validates that the WL kernel matrix is PSD and the derived distance
matrix satisfies metric axioms on the datasets produced by eval_setup.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.eval_validation.conftest import DATASETS


@pytest.mark.eval_validation
class TestWLKernelPositivity:
    """Verify WL kernel matrix properties (PSD, symmetry)."""

    @pytest.mark.parametrize("dataset", DATASETS)
    def test_kernel_diagonal_nonneg(self, eval_data, dataset):
        if dataset not in eval_data.wl_kernel_matrices:
            pytest.skip(f"No WL kernel for {dataset}")
        K = eval_data.wl_kernel_matrices[dataset]
        diag = np.diag(K)
        assert np.all(diag >= -1e-10), f"Negative diagonal entries: {diag[diag < -1e-10]}"

    @pytest.mark.parametrize("dataset", DATASETS)
    def test_kernel_symmetric(self, eval_data, dataset):
        if dataset not in eval_data.wl_kernel_matrices:
            pytest.skip(f"No WL kernel for {dataset}")
        K = eval_data.wl_kernel_matrices[dataset]
        np.testing.assert_allclose(K, K.T, atol=1e-10, err_msg="Kernel matrix not symmetric")


@pytest.mark.eval_validation
class TestWLDistanceMetricAxioms:
    """Verify WL distance matrix satisfies metric axioms."""

    @pytest.mark.parametrize("dataset", DATASETS)
    def test_diagonal_zero(self, eval_data, dataset):
        if dataset not in eval_data.wl_distance_matrices:
            pytest.skip(f"No WL distance for {dataset}")
        D = eval_data.wl_distance_matrices[dataset]
        diag = np.diag(D)
        np.testing.assert_allclose(diag, 0.0, atol=1e-10, err_msg="Non-zero diagonal")

    @pytest.mark.parametrize("dataset", DATASETS)
    def test_symmetry(self, eval_data, dataset):
        if dataset not in eval_data.wl_distance_matrices:
            pytest.skip(f"No WL distance for {dataset}")
        D = eval_data.wl_distance_matrices[dataset]
        np.testing.assert_allclose(D, D.T, atol=1e-10, err_msg="Distance matrix not symmetric")

    @pytest.mark.parametrize("dataset", DATASETS)
    def test_non_negative(self, eval_data, dataset):
        if dataset not in eval_data.wl_distance_matrices:
            pytest.skip(f"No WL distance for {dataset}")
        D = eval_data.wl_distance_matrices[dataset]
        assert np.all(D >= -1e-10), f"Negative distances found: min={D.min()}"

    @pytest.mark.parametrize("dataset", DATASETS)
    def test_triangle_inequality(self, eval_data, dataset):
        """Sample 5000 triples and check triangle inequality."""
        if dataset not in eval_data.wl_distance_matrices:
            pytest.skip(f"No WL distance for {dataset}")
        D = eval_data.wl_distance_matrices[dataset]
        n = D.shape[0]
        if n < 3:
            pytest.skip("Too few graphs for triangle inequality test")

        rng = np.random.default_rng(42)
        n_check = min(5000, n * (n - 1) * (n - 2) // 6)
        n_violations = 0

        for _ in range(n_check):
            i, j, k = rng.choice(n, size=3, replace=False)
            # d(i,k) <= d(i,j) + d(j,k) + tolerance
            if D[i, k] > D[i, j] + D[j, k] + 1e-8:
                n_violations += 1

        assert n_violations == 0, f"{n_violations}/{n_check} triangle inequality violations"


@pytest.mark.eval_validation
class TestWLEffectiveDimensionality:
    """Verify WL kernel has nontrivial structure."""

    @pytest.mark.parametrize("dataset", DATASETS)
    def test_nontrivial_rank(self, eval_data, dataset):
        """Kernel matrix should have rank > 1 (not all graphs identical)."""
        if dataset not in eval_data.wl_kernel_matrices:
            pytest.skip(f"No WL kernel for {dataset}")
        K = eval_data.wl_kernel_matrices[dataset]
        eigenvalues = np.linalg.eigvalsh(K)
        max_eig = eigenvalues[-1]
        # Count eigenvalues > 1% of max
        n_significant = int(np.sum(eigenvalues > 0.01 * max_eig))
        assert n_significant > 1, f"Effective rank is {n_significant}, expected > 1"

    @pytest.mark.parametrize("dataset", DATASETS)
    def test_distance_variance(self, eval_data, dataset):
        """Distance matrix should have nonzero variance."""
        if dataset not in eval_data.wl_distance_matrices:
            pytest.skip(f"No WL distance for {dataset}")
        D = eval_data.wl_distance_matrices[dataset]
        n = D.shape[0]
        triu_idx = np.triu_indices(n, k=1)
        d_upper = D[triu_idx]
        assert np.var(d_upper) > 0, "All WL distances are identical (zero variance)"


@pytest.mark.eval_validation
class TestWLCorrelationSanity:
    """Verify WL distance shows nontrivial correlation with GED."""

    @pytest.mark.parametrize("dataset", DATASETS)
    def test_nontrivial_ged_correlation(self, eval_data, dataset):
        """Spearman(d_WL, GED) should be > 0.1."""
        if dataset not in eval_data.wl_distance_matrices:
            pytest.skip(f"No WL distance for {dataset}")
        if dataset not in eval_data.ged_matrices:
            pytest.skip(f"No GED matrix for {dataset}")

        from scipy import stats

        D_wl = eval_data.wl_distance_matrices[dataset]
        D_ged = eval_data.ged_matrices[dataset]
        n = D_wl.shape[0]
        triu_idx = np.triu_indices(n, k=1)

        v_wl = D_wl[triu_idx].astype(np.float64)
        v_ged = D_ged[triu_idx].astype(np.float64)

        # Mask invalid entries
        valid = np.isfinite(v_wl) & np.isfinite(v_ged) & (v_ged >= 0) & (v_wl >= 0)
        if np.sum(valid) < 10:
            pytest.skip("Too few valid pairs")

        rho, _ = stats.spearmanr(v_wl[valid], v_ged[valid])
        assert rho > 0.1, f"WL-GED Spearman rho = {rho:.4f}, expected > 0.1"
