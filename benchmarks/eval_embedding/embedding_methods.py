"""Pure computational functions for MDS embedding and Procrustes analysis.

No I/O, no logging side effects. Uses only numpy + scipy.

Mathematical variable names (D, B, H, V, W, X) follow standard MDS notation.

References:
    - Borg & Groenen (2005). Modern Multidimensional Scaling. Springer.
    - De Leeuw (1977). Applications of convex analysis to MDS. (SMACOF)
    - Gower (1975). Generalized Procrustes analysis. Psychometrika.
"""
# ruff: noqa: N803, N806

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.spatial import procrustes as scipy_procrustes
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr

# =============================================================================
# Result dataclasses
# =============================================================================


@dataclass
class CMDSResult:
    """Result of classical MDS eigenanalysis."""

    eigenvalues: np.ndarray
    nev_ratio: float
    n_positive: int
    n_negative: int
    cumulative_variance: np.ndarray


@dataclass
class SMACOFResult:
    """Result of SMACOF MDS embedding."""

    coords: np.ndarray
    stress_1: float
    n_iterations: int
    converged: bool


@dataclass
class ProcrustesResult:
    """Result of Procrustes permutation test."""

    m_squared: float
    p_value: float
    n_permutations: int
    n_more_extreme: int


@dataclass
class ShepardResult:
    """Result of Shepard diagram analysis."""

    original_distances: np.ndarray
    embedded_distances: np.ndarray
    r_squared: float
    monotonic_r_squared: float


# =============================================================================
# Finite submatrix extraction
# =============================================================================


def find_finite_submatrix(D: np.ndarray) -> np.ndarray:
    """Greedy removal of rows/cols with most inf until all-finite.

    At each step, removes the row/col with the most non-finite entries.
    Continues until all remaining entries are finite.

    Args:
        D: Square distance matrix, possibly containing inf/nan.

    Returns:
        Index array of rows/cols to keep.
    """
    n = D.shape[0]
    keep = np.ones(n, dtype=bool)

    while True:
        active = np.where(keep)[0]
        sub = D[np.ix_(active, active)]
        non_finite = ~np.isfinite(sub)
        if not non_finite.any():
            break
        # Count non-finite per row
        counts = non_finite.sum(axis=1)
        worst = int(np.argmax(counts))
        if counts[worst] == 0:
            break
        keep[active[worst]] = False

    return np.where(keep)[0]


# =============================================================================
# Classical MDS
# =============================================================================


def classical_mds_eigenanalysis(D: np.ndarray) -> CMDSResult:
    """Classical MDS eigenanalysis on a distance matrix.

    Computes B = -0.5 * H @ D^2 @ H and eigendecomposes.
    Returns full spectrum and negative eigenvalue ratio (NEV).

    The NEV ratio quantifies non-Euclideanity:
    NEV = sum(|lambda_i|, lambda_i < 0) / sum(|lambda_i|)

    Args:
        D: Square distance matrix (all entries must be finite).

    Returns:
        CMDSResult with eigenvalues, NEV ratio, and variance explained.
    """
    n = D.shape[0]
    D_sq = D**2
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D_sq @ H

    eigenvalues = np.linalg.eigvalsh(B)
    eigenvalues = eigenvalues[::-1]  # Descending order

    n_positive = int(np.sum(eigenvalues > 0))
    n_negative = int(np.sum(eigenvalues < 0))

    abs_eigs = np.abs(eigenvalues)
    total = abs_eigs.sum()
    nev_ratio = float(abs_eigs[eigenvalues < 0].sum() / total) if total > 0 else 0.0

    # Cumulative variance from positive eigenvalues
    pos_eigs = np.maximum(eigenvalues, 0)
    pos_total = pos_eigs.sum()
    cumulative_variance = (
        np.cumsum(pos_eigs) / pos_total if pos_total > 0 else np.zeros_like(pos_eigs)
    )

    return CMDSResult(
        eigenvalues=eigenvalues,
        nev_ratio=nev_ratio,
        n_positive=n_positive,
        n_negative=n_negative,
        cumulative_variance=cumulative_variance,
    )


def classical_mds_embed(D: np.ndarray, n_components: int) -> np.ndarray:
    """Classical MDS embedding using top-k positive eigenvalues.

    Args:
        D: Square distance matrix (all entries must be finite).
        n_components: Target dimensionality.

    Returns:
        Coordinate matrix of shape (n, n_components).
    """
    n = D.shape[0]
    D_sq = D**2
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D_sq @ H

    # eigh returns ascending order; we want top-k
    eigenvalues, eigenvectors = eigh(B)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Take top n_components, clamp negative to zero
    k = min(n_components, len(eigenvalues))
    eigs = np.maximum(eigenvalues[:k], 0)
    vecs = eigenvectors[:, :k]

    coords = vecs * np.sqrt(eigs)[np.newaxis, :]

    # Pad with zeros if fewer positive eigenvalues than requested
    if k < n_components:
        coords = np.hstack([coords, np.zeros((n, n_components - k))])

    return coords


# =============================================================================
# SMACOF
# =============================================================================


def _smacof_single_run(
    D: np.ndarray,
    n_components: int,
    max_iter: int,
    eps: float,
    init: np.ndarray,
    weights: np.ndarray | None,
) -> tuple[np.ndarray, float, int, bool]:
    """Single SMACOF run with Guttman transform.

    Implements the iterative majorization algorithm:
    X^(k+1) = V^+ @ B(X^k) @ X^k

    Args:
        D: Square distance matrix (inf entries should be pre-replaced).
        n_components: Embedding dimensionality.
        max_iter: Maximum iterations.
        eps: Relative convergence tolerance on stress.
        init: Initial coordinates (n, n_components).
        weights: Weight matrix (same shape as D). None means uniform.

    Returns:
        (coords, stress_1, n_iterations, converged)
    """
    n = D.shape[0]
    X = init.copy()  # noqa: N806

    W = np.ones_like(D) if weights is None else weights.copy()  # noqa: N806

    # Zero diagonal weights
    np.fill_diagonal(W, 0.0)

    # Precompute V = diag(W @ 1) - W and its pseudoinverse
    V = np.diag(W.sum(axis=1)) - W
    V_pinv = np.linalg.pinv(V)

    stress_prev = np.inf
    converged = False

    for iteration in range(max_iter):
        # Pairwise distances in current embedding
        D_X = squareform(pdist(X))

        # Compute stress-1
        stress_1 = kruskal_stress_1(D, X, weights=W)

        # Check convergence
        if stress_prev != np.inf:
            rel_change = abs(stress_prev - stress_1) / (stress_prev + 1e-12)
            if rel_change < eps:
                converged = True
                return X, stress_1, iteration + 1, converged

        stress_prev = stress_1

        # Build B(X) matrix
        B = np.zeros((n, n))
        safe_D_X = np.where(D_X > 1e-12, D_X, 1e-12)
        B = -W * D / safe_D_X
        np.fill_diagonal(B, 0.0)
        np.fill_diagonal(B, -B.sum(axis=1))

        # Guttman transform
        X = V_pinv @ B @ X

    # Final stress
    stress_1 = kruskal_stress_1(D, X, weights=W)
    return X, stress_1, max_iter, converged


def smacof(
    D: np.ndarray,
    n_components: int = 2,
    max_iter: int = 300,
    eps: float = 1e-6,
    n_init: int = 4,
    seed: int = 42,
    weights: np.ndarray | None = None,
) -> SMACOFResult:
    """SMACOF MDS with multiple restarts.

    First run uses cMDS initialization; remaining runs use random init.
    Returns the embedding with lowest stress.

    Args:
        D: Square distance matrix. Inf entries should be pre-replaced
            (e.g., with max_finite * 2) and masked via weights.
        n_components: Target dimensionality.
        max_iter: Maximum SMACOF iterations per run.
        eps: Relative stress convergence tolerance.
        n_init: Number of restarts.
        seed: Random seed.
        weights: Weight matrix (w_ij=0 for inf pairs). None means uniform.

    Returns:
        SMACOFResult with best embedding.
    """
    rng = np.random.default_rng(seed)
    n = D.shape[0]

    best_stress = np.inf
    best_coords = None
    best_iters = 0
    best_converged = False

    for run_idx in range(n_init):
        if run_idx == 0:
            # cMDS initialization
            init = classical_mds_embed(D, n_components)
        else:
            # Random initialization
            init = rng.standard_normal((n, n_components))

        coords, stress, n_iters, conv = _smacof_single_run(
            D, n_components, max_iter, eps, init, weights
        )

        if stress < best_stress:
            best_stress = stress
            best_coords = coords
            best_iters = n_iters
            best_converged = conv

    return SMACOFResult(
        coords=best_coords,  # type: ignore[arg-type]
        stress_1=best_stress,
        n_iterations=best_iters,
        converged=best_converged,
    )


# =============================================================================
# Stress measure
# =============================================================================


def kruskal_stress_1(
    D: np.ndarray,
    coords: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Kruskal's Stress-1: sqrt(sum(w*(delta-d)^2) / sum(w*d^2)).

    Computed over the upper triangle only.

    Args:
        D: Original distance matrix.
        coords: Embedded coordinates.
        weights: Weight matrix (same shape as D). None means uniform.

    Returns:
        Stress-1 value.
    """
    n = D.shape[0]
    triu_idx = np.triu_indices(n, k=1)

    delta = D[triu_idx]
    d_emb = squareform(pdist(coords))
    d = d_emb[triu_idx]

    w = weights[triu_idx] if weights is not None else np.ones_like(delta)

    numerator = np.sum(w * (delta - d) ** 2)
    denominator = np.sum(w * d**2)

    if denominator < 1e-12:
        return 0.0
    return float(np.sqrt(numerator / denominator))


# =============================================================================
# Procrustes analysis
# =============================================================================


def procrustes_permutation_test(
    coords_1: np.ndarray,
    coords_2: np.ndarray,
    n_permutations: int = 9999,
    seed: int = 42,
) -> ProcrustesResult:
    """Procrustes analysis with row-permutation null distribution.

    Uses scipy.spatial.procrustes for alignment, then builds a null
    distribution by randomly permuting rows of coords_2.

    p-value = (n_more_extreme + 1) / (n_permutations + 1)

    Args:
        coords_1: First coordinate matrix (n, d).
        coords_2: Second coordinate matrix (n, d).
        n_permutations: Number of random permutations for null.
        seed: Random seed.

    Returns:
        ProcrustesResult with m^2 and permutation p-value.
    """
    rng = np.random.default_rng(seed)

    # Observed Procrustes disparity (m^2)
    _, _, observed_m2 = scipy_procrustes(coords_1, coords_2)

    # Permutation null
    n_more_extreme = 0
    for _ in range(n_permutations):
        perm = rng.permutation(coords_2.shape[0])
        _, _, perm_m2 = scipy_procrustes(coords_1, coords_2[perm])
        if perm_m2 <= observed_m2:
            n_more_extreme += 1

    p_value = (n_more_extreme + 1) / (n_permutations + 1)

    return ProcrustesResult(
        m_squared=float(observed_m2),
        p_value=p_value,
        n_permutations=n_permutations,
        n_more_extreme=n_more_extreme,
    )


# =============================================================================
# Shepard diagram
# =============================================================================


def shepard_data(
    D: np.ndarray,
    coords: np.ndarray,
    weights: np.ndarray | None = None,
) -> ShepardResult:
    """Compute Shepard diagram data: original vs embedded distances.

    Returns upper-triangle pairs with R^2 (Pearson) and monotonic R^2
    (Spearman squared).

    Args:
        D: Original distance matrix.
        coords: Embedded coordinates.
        weights: Weight matrix. If provided, only includes pairs with w > 0.

    Returns:
        ShepardResult with distance arrays and R^2 values.
    """
    n = D.shape[0]
    triu_idx = np.triu_indices(n, k=1)

    orig = D[triu_idx].astype(np.float64)
    emb = squareform(pdist(coords))[triu_idx].astype(np.float64)

    if weights is not None:
        w = weights[triu_idx]
        mask = w > 0
        orig = orig[mask]
        emb = emb[mask]

    # Filter non-finite
    valid = np.isfinite(orig) & np.isfinite(emb)
    orig = orig[valid]
    emb = emb[valid]

    if len(orig) < 3:
        return ShepardResult(
            original_distances=orig,
            embedded_distances=emb,
            r_squared=0.0,
            monotonic_r_squared=0.0,
        )

    r, _ = pearsonr(orig, emb)
    rho, _ = spearmanr(orig, emb)

    return ShepardResult(
        original_distances=orig,
        embedded_distances=emb,
        r_squared=float(r**2),
        monotonic_r_squared=float(rho**2),
    )
