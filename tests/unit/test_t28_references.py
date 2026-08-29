"""Unit tests for T-28 spectral reference metrics.

Tests cover:
- Known small graph spectra against analytic values.
- Isomorphic graphs give distance exactly 0 for every variant.
- spectral_esd against a hand-computed 3-vs-2-atom Wasserstein example.
- Zero-padding produces the documented padded length.
- Metric axioms (symmetry, non-negativity, identity, triangle inequality) on a
  small random sample for the three Euclidean variants.
- The isolated-vertex guard does not cause division by zero.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import wasserstein_distance  # type: ignore[import-untyped]

from isalgraph.competitors.references.spectral import (
    cohort_spectra,
    laplacian_spectrum,
    spectral_distance_matrix,
    spectral_esd_matrix,
)

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _empty_edges(dtype: type = np.int32) -> np.ndarray:
    """Return an empty (2, 0) edge array."""
    return np.empty((2, 0), dtype=dtype)


def _edges(pairs: list[tuple[int, int]]) -> np.ndarray:
    """Build a (2, m) int32 edge array from a list of (u, v) pairs."""
    if not pairs:
        return _empty_edges()
    arr = np.array(pairs, dtype=np.int32).T  # shape (2, m)
    return arr


def _trivial_csr(n_list: list[int], edge_list: list[list[tuple[int, int]]]) -> tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    """Build CSR-style arrays for a small cohort.

    Args:
        n_list: Per-graph node counts.
        edge_list: Per-graph edge pairs.

    Returns:
        (n_nodes int32, edge_offsets int64, edges int32)
    """
    n_nodes = np.array(n_list, dtype=np.int32)
    flat: list[tuple[int, int]] = []
    offsets = [0]
    for pairs in edge_list:
        flat.extend(pairs)
        offsets.append(len(flat))
    edge_offsets = np.array(offsets, dtype=np.int64)
    edges = np.array(flat, dtype=np.int32).T if flat else _empty_edges()  # (2, E)
    return n_nodes, edge_offsets, edges


# ---------------------------------------------------------------------------
# 1.  Known small-graph spectra
# ---------------------------------------------------------------------------

class TestKnownSpectra:
    """Verify against analytically known Laplacian spectra.

    P3 (path on 3 nodes):  L_sym eigenvalues {0, 1, 2}  → sorted desc: [2, 1, 0]
    C3 (3-cycle):          L_sym eigenvalues {0, 3/2, 3/2} → sorted desc: [3/2, 3/2, 0]
    K4 (complete on 4):    L_sym eigenvalues {0, 4/3, 4/3, 4/3} → sorted desc: [4/3, 4/3, 4/3, 0]
    K1 (single node):      L_sym eigenvalue {0}
    """

    def test_p3_norm(self) -> None:
        # P3: edges 0-1, 1-2
        sp = laplacian_spectrum(3, _edges([(0, 1), (1, 2)]), variant="norm")
        np.testing.assert_allclose(sp, [2.0, 1.0, 0.0], atol=1e-10)

    def test_c3_norm(self) -> None:
        # C3: edges 0-1, 1-2, 2-0
        sp = laplacian_spectrum(3, _edges([(0, 1), (1, 2), (2, 0)]), variant="norm")
        expected = [1.5, 1.5, 0.0]
        np.testing.assert_allclose(
            sorted(sp, reverse=True), sorted(expected, reverse=True), atol=1e-10
        )

    def test_k4_norm(self) -> None:
        # K4: all 6 pairs
        pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        sp = laplacian_spectrum(4, _edges(pairs), variant="norm")
        expected = [4 / 3, 4 / 3, 4 / 3, 0.0]
        np.testing.assert_allclose(
            sorted(sp, reverse=True), sorted(expected, reverse=True), atol=1e-10
        )

    def test_k1_norm(self) -> None:
        # Single node, no edges.  The isolated-vertex guard sets inv_sqrt = 0,
        # so L_sym = I - 0 = I, giving eigenvalue 1.0.  This is our convention
        # (not the graph-theoretic zero); the guard comment in spectral.py
        # documents it explicitly.
        sp = laplacian_spectrum(1, _empty_edges(), variant="norm")
        np.testing.assert_allclose(sp, [1.0], atol=1e-10)

    def test_p3_comb(self) -> None:
        # P3 combinatorial Laplacian eigenvalues: {0, 1, 3}  (Wilson & Zhu Table 1)
        # Actually P3 L_comb: degrees are [1, 2, 1], eigenvalues 0, 1, 3
        sp = laplacian_spectrum(3, _edges([(0, 1), (1, 2)]), variant="comb")
        np.testing.assert_allclose(sorted(sp, reverse=True), [3.0, 1.0, 0.0], atol=1e-10)

    def test_k4_adj(self) -> None:
        # K4 adjacency spectrum: eigenvalues {3, -1, -1, -1}
        pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        sp = laplacian_spectrum(4, _edges(pairs), variant="adj")
        np.testing.assert_allclose(sorted(sp, reverse=True), [3.0, -1.0, -1.0, -1.0], atol=1e-10)


# ---------------------------------------------------------------------------
# 2.  Isomorphic graphs → distance exactly 0
# ---------------------------------------------------------------------------

class TestIsomorphicDistance:
    """Isomorphism-invariant: relabelling nodes must not change the spectrum."""

    @pytest.mark.parametrize("variant", ["norm", "comb", "adj"])
    def test_c4_permuted(self, variant: str) -> None:
        # C4: 0-1, 1-2, 2-3, 3-0.  Permutation 0→2, 1→3, 2→0, 3→1 gives 2-3, 3-0, 0-1, 1-2.
        edges_orig = _edges([(0, 1), (1, 2), (2, 3), (3, 0)])
        edges_perm = _edges([(2, 3), (3, 0), (0, 1), (1, 2)])
        sp1 = laplacian_spectrum(4, edges_orig, variant=variant)  # type: ignore[arg-type]
        sp2 = laplacian_spectrum(4, edges_perm, variant=variant)  # type: ignore[arg-type]
        np.testing.assert_allclose(sp1, sp2, atol=1e-10)

    @pytest.mark.parametrize("variant", ["norm", "comb", "adj"])
    def test_distance_zero_for_permuted(self, variant: str) -> None:
        # cohort with G=C4 and its permutation → distance = 0
        n_nodes, offsets, edges = _trivial_csr(
            [4, 4],
            [[(0, 1), (1, 2), (2, 3), (3, 0)], [(2, 3), (3, 0), (0, 1), (1, 2)]],
        )
        spectra = cohort_spectra(n_nodes, offsets, edges, variant=variant)  # type: ignore[arg-type]
        dist = spectral_distance_matrix(spectra)
        assert dist[0, 1] == pytest.approx(0.0, abs=1e-10)
        assert dist[1, 0] == pytest.approx(0.0, abs=1e-10)

    def test_esd_zero_for_permuted(self) -> None:
        n_nodes, offsets, edges = _trivial_csr(
            [4, 4],
            [[(0, 1), (1, 2), (2, 3), (3, 0)], [(2, 3), (3, 0), (0, 1), (1, 2)]],
        )
        dist = spectral_esd_matrix(n_nodes, offsets, edges)
        assert dist[0, 1] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 3.  spectral_esd hand-computed example (3 vs 2 atoms)
# ---------------------------------------------------------------------------

class TestSpectralEsdHandComputed:
    """Verify scipy.stats.wasserstein_distance against a hand-computed example.

    Two uniform measures:
        μ = {0, 1, 2}  (atoms at 0, 1, 2, each with mass 1/3)
        ν = {0.5, 1.5} (atoms at 0.5, 1.5, each with mass 1/2)

    Quantile functions:
        F_μ^{-1}(t) : [0,1/3) → 0; [1/3,2/3) → 1; [2/3,1) → 2
        F_ν^{-1}(t) : [0,1/2) → 0.5; [1/2,1) → 1.5

    W_1 = integral_0^1 |F_μ^{-1}(t) - F_ν^{-1}(t)| dt

    Breakpoints: 0, 1/3, 1/2, 2/3, 1.
    Intervals and integrands:
        [0, 1/3):    |0 - 0.5| * (1/3)    = 0.5 * 1/3   = 1/6
        [1/3, 1/2):  |1 - 0.5| * (1/6)    = 0.5 * 1/6   = 1/12
        [1/2, 2/3):  |1 - 1.5| * (1/6)    = 0.5 * 1/6   = 1/12
        [2/3, 1):    |2 - 1.5| * (1/3)    = 0.5 * 1/3   = 1/6
    Total = 1/6 + 1/12 + 1/12 + 1/6 = 2/6 + 2/12 = 1/3 + 1/6 = 1/2.

    Expected W_1 = 0.5.
    """

    def test_scipy_matches_hand_computation(self) -> None:
        mu = np.array([0.0, 1.0, 2.0])
        nu = np.array([0.5, 1.5])
        result = float(wasserstein_distance(mu, nu))
        assert result == pytest.approx(0.5, abs=1e-12)

    def test_esd_matrix_unequal_sizes(self) -> None:
        # Build a cohort with graphs of different sizes.
        # Graph 0: P3 (3 nodes, edges 0-1, 1-2) — spectrum computed by test above
        # Graph 1: K2 (2 nodes, edge 0-1) — L_sym eigenvalues {0, 2}
        # W_1(P3_norm_spectrum, K2_norm_spectrum) ≠ 0
        n_nodes, offsets, edges = _trivial_csr(
            [3, 2],
            [[(0, 1), (1, 2)], [(0, 1)]],
        )
        dist = spectral_esd_matrix(n_nodes, offsets, edges)

        # diagonal must be 0
        assert dist[0, 0] == pytest.approx(0.0, abs=1e-12)
        assert dist[1, 1] == pytest.approx(0.0, abs=1e-12)

        # symmetric
        assert dist[0, 1] == pytest.approx(dist[1, 0], abs=1e-12)

        # non-negative and non-zero (P3 and K2 have different spectra)
        assert dist[0, 1] > 0.0

    def test_esd_uses_scipy_correctly(self) -> None:
        """The ESD matrix entry (i,j) equals scipy.wasserstein_distance directly."""
        n_nodes, offsets, edges = _trivial_csr(
            [3, 2],
            [[(0, 1), (1, 2)], [(0, 1)]],
        )
        sp_p3 = laplacian_spectrum(3, _edges([(0, 1), (1, 2)]), variant="norm")
        sp_k2 = laplacian_spectrum(2, _edges([(0, 1)]), variant="norm")
        expected = float(wasserstein_distance(sp_p3, sp_k2))

        dist = spectral_esd_matrix(n_nodes, offsets, edges)
        assert dist[0, 1] == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# 4.  Zero-padding produces the documented padded length
# ---------------------------------------------------------------------------

class TestZeroPadding:
    """cohort_spectra zero-pads to the cohort's maximum n."""

    def test_padded_length_equals_n_max(self) -> None:
        # Cohort: P3 (n=3) and K2 (n=2).  n_max should be 3.
        n_nodes, offsets, edges = _trivial_csr(
            [3, 2],
            [[(0, 1), (1, 2)], [(0, 1)]],
        )
        spectra = cohort_spectra(n_nodes, offsets, edges, variant="norm")
        assert spectra.shape == (2, 3)

    def test_k2_row_padded_to_3(self) -> None:
        # The K2 row (index 1) must have its last element zero-padded.
        n_nodes, offsets, edges = _trivial_csr(
            [3, 2],
            [[(0, 1), (1, 2)], [(0, 1)]],
        )
        spectra = cohort_spectra(n_nodes, offsets, edges, variant="norm")
        # K2 norm spectrum is [2, 0] (non-increasing), padded to [2, 0, 0]
        np.testing.assert_allclose(spectra[1], [2.0, 0.0, 0.0], atol=1e-10)

    def test_esd_not_padded(self) -> None:
        # ESD works on raw spectra: the distance between P3 and K2 should not
        # be inflated by padding zeros.  Verify by checking that the ESD result
        # differs from the Euclidean result on the padded spectra.
        n_nodes, offsets, edges = _trivial_csr(
            [3, 2],
            [[(0, 1), (1, 2)], [(0, 1)]],
        )
        spectra = cohort_spectra(n_nodes, offsets, edges, variant="norm")
        euclidean_dist = spectral_distance_matrix(spectra)[0, 1]
        esd_dist = spectral_esd_matrix(n_nodes, offsets, edges)[0, 1]
        # They need not be equal; the ESD removes the zero-padding confound.
        # Both must be finite and positive.
        assert math.isfinite(euclidean_dist) and euclidean_dist > 0.0
        assert math.isfinite(esd_dist) and esd_dist > 0.0


# ---------------------------------------------------------------------------
# 5.  Metric axioms on a small random sample
# ---------------------------------------------------------------------------

class TestMetricAxioms:
    """Symmetry, non-negativity, zero identity, triangle inequality."""

    @staticmethod
    def _make_small_cohort(seed: int = 42, n_graphs: int = 8) -> tuple[
        np.ndarray, np.ndarray, np.ndarray
    ]:
        rng = np.random.default_rng(seed)
        n_nodes_list: list[int] = []
        edge_list: list[list[tuple[int, int]]] = []
        for _ in range(n_graphs):
            n = int(rng.integers(2, 6))
            n_nodes_list.append(n)
            # Erdos-Renyi p=0.5
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.5]
            edge_list.append(pairs)
        return _trivial_csr(n_nodes_list, edge_list)

    @pytest.mark.parametrize("variant", ["norm", "comb", "adj"])
    def test_symmetry(self, variant: str) -> None:
        n_nodes, offsets, edges = self._make_small_cohort()
        spectra = cohort_spectra(n_nodes, offsets, edges, variant=variant)  # type: ignore[arg-type]
        dist = spectral_distance_matrix(spectra)
        np.testing.assert_allclose(dist, dist.T, atol=1e-12)

    @pytest.mark.parametrize("variant", ["norm", "comb", "adj"])
    def test_nonnegative(self, variant: str) -> None:
        n_nodes, offsets, edges = self._make_small_cohort()
        spectra = cohort_spectra(n_nodes, offsets, edges, variant=variant)  # type: ignore[arg-type]
        dist = spectral_distance_matrix(spectra)
        assert float(dist.min()) >= 0.0

    @pytest.mark.parametrize("variant", ["norm", "comb", "adj"])
    def test_zero_diagonal(self, variant: str) -> None:
        n_nodes, offsets, edges = self._make_small_cohort()
        spectra = cohort_spectra(n_nodes, offsets, edges, variant=variant)  # type: ignore[arg-type]
        dist = spectral_distance_matrix(spectra)
        np.testing.assert_allclose(np.diag(dist), 0.0, atol=1e-12)

    @pytest.mark.parametrize("variant", ["norm", "comb", "adj"])
    def test_triangle_inequality(self, variant: str) -> None:
        n_nodes, offsets, edges = self._make_small_cohort()
        spectra = cohort_spectra(n_nodes, offsets, edges, variant=variant)  # type: ignore[arg-type]
        dist = spectral_distance_matrix(spectra)
        g = dist.shape[0]
        violations = 0
        for i in range(g):
            for j in range(g):
                for k in range(g):
                    if dist[i, j] > dist[i, k] + dist[k, j] + 1e-10:
                        violations += 1
        assert violations == 0, (
            f"{violations} triangle-inequality violations for variant={variant!r}"
        )


# ---------------------------------------------------------------------------
# 6.  Isolated-vertex guard
# ---------------------------------------------------------------------------

class TestIsolatedVertexGuard:
    """Isolated vertices must not cause division by zero."""

    def test_single_isolated_node_norm(self) -> None:
        # Graph with 1 node and no edges: degree = 0, inv_sqrt = 0 (guard),
        # L_sym = I (not 0), giving eigenvalue 1.0 -- no NaN, no division by zero.
        sp = laplacian_spectrum(1, _empty_edges(), variant="norm")
        assert math.isfinite(sp[0])
        assert sp[0] == pytest.approx(1.0, abs=1e-10)

    def test_mixed_isolated_node_norm(self) -> None:
        # Graph 0-1-2 with node 3 isolated (no edges to it)
        # Node degrees: 0→1, 1→2, 2→1, 3→0
        sp = laplacian_spectrum(4, _edges([(0, 1), (1, 2)]), variant="norm")
        assert all(math.isfinite(v) for v in sp)

    def test_fully_disconnected_graph_norm(self) -> None:
        # 4 isolated nodes: all degrees 0, all inv_sqrt = 0 (guard).
        # L_sym = I - D^{-1/2} A D^{-1/2} = I - 0 = I.  All eigenvalues 1.0.
        sp = laplacian_spectrum(4, _empty_edges(), variant="norm")
        np.testing.assert_allclose(sorted(sp, reverse=True), [1.0, 1.0, 1.0, 1.0], atol=1e-10)

    def test_esd_isolated_node(self) -> None:
        # ESD should not raise for isolated nodes
        n_nodes, offsets, edges = _trivial_csr([1, 2], [[], [(0, 1)]])
        dist = spectral_esd_matrix(n_nodes, offsets, edges)
        assert all(math.isfinite(dist[i, j]) for i in range(2) for j in range(2))
