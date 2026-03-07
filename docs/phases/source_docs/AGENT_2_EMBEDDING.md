# Agent 2: Embedding Quality Analysis

**Priority**: Secondary benchmark — supports the correlation claim with geometric evidence.
**Dependencies**: Agent 0 (SETUP) must complete first.
**Parallelizable with**: Agents 1, 3, 4.
**Estimated local time**: ~20 min.
**Estimated Picasso time**: ~10 min.

PLAN ALREADY CREATED: `~/.claude/plans/immutable-sniffing-moonbeam.md`
---

## 1. Scientific Context

### 1.1 Research Question

If Levenshtein distance is a good proxy for GED, then the **geometry of the Levenshtein distance space** should resemble the **geometry of the GED space**. This benchmark tests whether the two spaces have similar topology by embedding both into Euclidean space and comparing the resulting configurations.

### 1.2 Theoretical Background

**Levenshtein distance is a metric** (satisfies non-negativity, identity of indiscernibles, symmetry, triangle inequality) **but is NOT isometrically embeddable in L₂** (Bourgain, 1985). The double-centered matrix will have negative eigenvalues, meaning it is not a Euclidean Distance Matrix (EDM). Bourgain's theorem guarantees O(log n) distortion in the worst case, but practical distortion may be much lower.

**Metric MDS (SMACOF)** — Scaling by Majorizing a Complicated Function (de Leeuw, 1977; Borg & Groenen, 2005) — directly minimizes raw stress:

$$\sigma(X) = \sum_{i<j} w_{ij} (\delta_{ij} - d_{ij}(X))^2$$

where δ_{ij} are the original distances and d_{ij}(X) are the Euclidean distances in the embedding. SMACOF has guaranteed monotone convergence and runs in O(n²T) per iteration.

**Kruskal's Stress-1** assesses embedding quality:

$$\text{Stress-1} = \sqrt{\frac{\sum_{i<j} (\delta_{ij} - d_{ij})^2}{\sum_{i<j} d_{ij}^2}}$$

Interpretation (Kruskal, 1964): < 0.05 excellent, 0.05–0.10 good, 0.10–0.20 fair, > 0.20 poor.

**Classical MDS (Torgerson, 1952)** provides initialization and an embeddability diagnostic. By double-centering the squared distance matrix and computing eigendecomposition, we get eigenvalues λ₁ ≥ λ₂ ≥ ... . For a true EDM, all eigenvalues are non-negative. The **negative eigenvalue ratio** (NEV ratio) quantifies non-Euclideanity:

$$\text{NEV ratio} = \frac{\sum_{i: \lambda_i < 0} |\lambda_i|}{\sum_{i} |\lambda_i|}$$

Lower is better. NEV ratio = 0 means the distance is perfectly Euclidean.

**Procrustes analysis (PROTEST)** (Peres-Neto & Jackson, 2001, *Oecologia* 129:169–178) directly compares two MDS configurations by finding the optimal rotation, reflection, translation, and isotropic scaling that minimizes:

$$m^2 = 1 - \frac{(\text{tr}(R \cdot X_2^T X_1))^2}{\text{tr}(X_1^T X_1) \cdot \text{tr}(X_2^T X_2)}$$

where R is the optimal rotation matrix. Peres-Neto & Jackson showed PROTEST is **more powerful** than the Mantel test for detecting matrix associations. Available in `scipy.spatial.procrustes()`.

### 1.3 Hypotheses

**H₁ (Low distortion)**: MDS embeddings of Levenshtein matrices achieve Stress-1 < 0.20 in 2D and < 0.10 in 5D.

**H₂ (Geometric agreement)**: Procrustes analysis between MDS(Levenshtein) and MDS(GED) yields m² significantly lower than random (p < 0.001 via permutation).

**H₃ (Shepard fidelity)**: Shepard diagrams show monotone relationship between original and embedded distances for both metrics.

### 1.4 What we explicitly do NOT do

**t-SNE and UMAP are NOT suitable** for this analysis. t-SNE (van der Maaten & Hinton, 2008) minimizes KL divergence between probability distributions, sacrificing absolute distance information. UMAP (McInnes et al., 2018) prioritizes topological neighborhoods over metric distances. Both are visualization tools, not distance-preserving embeddings. Using them would be methodologically incorrect for our purpose.

**Isomap** (Tenenbaum et al., 2000) estimates geodesic distances on manifolds — but Levenshtein distances don't arise from a continuous manifold, making the geodesic estimation step pointless.

---

## 2. Input Data

From Agent 0:

| File | Content |
|------|---------|
| `ged_matrices/{dataset}.npz` | GED distance matrix |
| `levenshtein_matrices/{dataset}.npz` | Levenshtein distance matrix |
| `graph_metadata/{dataset}.json` | Class labels for coloring |

---

## 3. Output Specification

### 3.1 Directory Structure

```
benchmarks/eval_embedding/
    __init__.py
    eval_embedding.py           # Main orchestrator
    embedding_methods.py        # MDS, Procrustes, stress computation
    README.md

results/eval_embedding/
    raw/
        {dataset}_cmds_eigenvalues.npz       # Full eigenspectrum from cMDS
        {dataset}_smacof_coords_{dim}d.npz   # SMACOF coordinates
        {dataset}_stress_by_dim.json         # Stress-1 for each dimensionality
        {dataset}_procrustes_residuals.npz   # Per-point residuals
        {dataset}_shepard_data.npz           # (original_dist, embedded_dist) pairs
    stats/
        {dataset}_embedding_stats.json       # All statistics
        summary.json                         # Cross-dataset summary
    figures/
        embedding_main_figure.pdf            # Main paper figure
        shepard_{dataset}.pdf                # Shepard diagrams
        scree_{dataset}.pdf                  # Scree plots
        mds_scatter_{dataset}.pdf            # 2D MDS colored by class
    tables/
        embedding_summary.tex                # LaTeX table
```

### 3.2 Raw Data — cMDS Eigenvalues

Save the full eigenspectrum for each distance matrix (GED and Levenshtein). This enables downstream analysis of embeddability without recomputation.

```
{dataset}_cmds_eigenvalues.npz:
    eigenvalues_ged: float64 [N]        # Sorted descending
    eigenvalues_lev: float64 [N]        # Sorted descending
    nev_ratio_ged: float64              # Negative eigenvalue ratio
    nev_ratio_lev: float64
    explained_variance_ged: float64 [N] # Cumulative proportion of positive eigenvalues
    explained_variance_lev: float64 [N]
```

### 3.3 Statistical Results JSON

```json
{
    "dataset": "iam_letter_low",
    "n_graphs": 750,
    "cmds": {
        "nev_ratio_ged": 0.02,
        "nev_ratio_lev": 0.08,
        "n_positive_eigenvalues_ged": 740,
        "n_positive_eigenvalues_lev": 700,
        "variance_2d_ged": 0.45,
        "variance_2d_lev": 0.38
    },
    "smacof": {
        "stress_by_dim": {
            "2": 0.18,
            "3": 0.12,
            "5": 0.07,
            "10": 0.03
        },
        "stress_by_dim_ged": {
            "2": 0.15,
            "3": 0.10,
            "5": 0.06,
            "10": 0.02
        }
    },
    "procrustes": {
        "m_squared_2d": 0.12,
        "m_squared_5d": 0.06,
        "p_value_2d": 0.0001,
        "p_value_5d": 0.0001,
        "n_permutations": 9999,
        "mean_per_point_residual_2d": 0.08,
        "max_per_point_residual_2d": 0.35
    },
    "shepard": {
        "r_squared_lev_2d": 0.82,
        "r_squared_ged_2d": 0.88,
        "r_squared_lev_5d": 0.95,
        "r_squared_ged_5d": 0.97
    }
}
```

---

## 4. Implementation Plan

### 4.1 Module Structure

```
benchmarks/eval_embedding/
    __init__.py
    eval_embedding.py        # CLI orchestrator
    embedding_methods.py     # All embedding computations
    README.md
```

### 4.2 Core Functions (`embedding_methods.py`)

```python
"""Embedding quality analysis for Levenshtein and GED distance matrices.

References:
    - de Leeuw (1977). SMACOF algorithm.
    - Borg & Groenen (2005). Modern Multidimensional Scaling. Springer.
    - Kruskal (1964). Psychometrika 29:1-27.
    - Peres-Neto & Jackson (2001). Oecologia 129:169-178.
"""

import numpy as np
from sklearn.manifold import MDS
from scipy.spatial import procrustes
from scipy.spatial.distance import squareform, pdist


def classical_mds_eigenvalues(D: np.ndarray) -> dict:
    """Compute classical MDS eigendecomposition.
    
    Performs double-centering on the squared distance matrix
    and returns the full eigenspectrum.
    
    Args:
        D: Symmetric distance matrix (N×N).
        
    Returns:
        Dict with eigenvalues (sorted descending), NEV ratio,
        cumulative explained variance.
    """
    N = D.shape[0]
    # Replace inf with max finite value for computation
    D_clean = D.copy()
    finite_mask = np.isfinite(D_clean)
    max_val = D_clean[finite_mask].max() if finite_mask.any() else 1.0
    D_clean[~finite_mask] = max_val * 2  # Replace inf
    
    # Double centering: B = -0.5 * J * D² * J where J = I - 1/n * 11^T
    D_sq = D_clean ** 2
    H = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * H @ D_sq @ H
    
    eigenvalues = np.linalg.eigvalsh(B)[::-1]  # Sort descending
    
    pos_eigenvalues = eigenvalues[eigenvalues > 0]
    neg_eigenvalues = eigenvalues[eigenvalues < 0]
    
    nev_ratio = np.abs(neg_eigenvalues).sum() / np.abs(eigenvalues).sum()
    cumulative_var = np.cumsum(pos_eigenvalues) / pos_eigenvalues.sum()
    
    return {
        "eigenvalues": eigenvalues,
        "nev_ratio": float(nev_ratio),
        "n_positive": len(pos_eigenvalues),
        "n_negative": len(neg_eigenvalues),
        "cumulative_variance": cumulative_var,
    }


def smacof_embedding(
    D: np.ndarray,
    n_components: int = 2,
    max_iter: int = 300,
    n_init: int = 4,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """Metric MDS embedding via SMACOF.
    
    Uses scikit-learn's MDS with dissimilarity='precomputed'.
    
    Args:
        D: Symmetric distance matrix.
        n_components: Embedding dimensionality.
        max_iter: Max SMACOF iterations.
        n_init: Number of random restarts.
        seed: Random seed.
        
    Returns:
        Tuple of (embedded_coords [N, n_components], stress_1).
    """
    # Handle inf: replace with large finite value
    D_clean = D.copy()
    finite_mask = np.isfinite(D_clean)
    max_val = D_clean[finite_mask].max() if finite_mask.any() else 1.0
    D_clean[~finite_mask] = max_val * 2
    
    mds = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        max_iter=max_iter,
        n_init=n_init,
        random_state=seed,
        normalized_stress=True,  # Returns Stress-1
    )
    coords = mds.fit_transform(D_clean)
    stress_1 = mds.stress_  # This is normalized (Stress-1)
    
    return coords, float(stress_1)


def kruskal_stress_1(D_original: np.ndarray, coords: np.ndarray) -> float:
    """Compute Kruskal's Stress-1 from original distances and embedded coords.
    
    Stress-1 = sqrt(sum((delta_ij - d_ij)^2) / sum(d_ij^2))
    
    Note: scikit-learn's MDS.stress_ may use a different normalization.
    We compute the standard Kruskal formula independently.
    """
    D_embedded = squareform(pdist(coords))
    D_orig_clean = D_original.copy()
    finite = np.isfinite(D_orig_clean)
    mask = np.triu(finite, k=1)
    
    delta = D_orig_clean[mask]
    d = D_embedded[mask]
    
    numerator = np.sum((delta - d) ** 2)
    denominator = np.sum(d ** 2)
    
    return float(np.sqrt(numerator / denominator)) if denominator > 0 else float("inf")


def procrustes_analysis(
    coords_1: np.ndarray,
    coords_2: np.ndarray,
    n_permutations: int = 9999,
    seed: int = 42,
) -> dict:
    """Procrustes analysis comparing two point configurations.
    
    Finds optimal rotation, reflection, translation, and scaling
    to minimize sum of squared residuals.
    
    Args:
        coords_1: First configuration [N, d].
        coords_2: Second configuration [N, d].
        n_permutations: For permutation p-value.
        seed: Random seed.
        
    Returns:
        Dict with m_squared, disparity, per_point_residuals, p_value.
    """
    # Center and scale
    mtx1, mtx2, disparity = procrustes(coords_1, coords_2)
    
    # Per-point residuals
    residuals = np.sqrt(np.sum((mtx1 - mtx2) ** 2, axis=1))
    
    # Permutation test for significance
    rng = np.random.default_rng(seed)
    perm_disparities = []
    for _ in range(n_permutations):
        perm_idx = rng.permutation(len(coords_2))
        _, _, perm_disp = procrustes(coords_1, coords_2[perm_idx])
        perm_disparities.append(perm_disp)
    
    p_value = (np.sum(np.array(perm_disparities) <= disparity) + 1) / (n_permutations + 1)
    
    return {
        "m_squared": float(disparity),
        "p_value": float(p_value),
        "per_point_residuals": residuals,
        "mean_residual": float(residuals.mean()),
        "max_residual": float(residuals.max()),
    }


def shepard_data(D_original: np.ndarray, coords: np.ndarray) -> dict:
    """Generate Shepard diagram data.
    
    Returns paired (original_distance, embedded_distance) for all valid pairs.
    Used for plotting and for R² computation.
    """
    D_embedded = squareform(pdist(coords))
    mask = np.triu(np.isfinite(D_original), k=1)
    
    original = D_original[mask]
    embedded = D_embedded[mask]
    
    # R² via correlation
    correlation = np.corrcoef(original, embedded)[0, 1]
    r_squared = correlation ** 2
    
    return {
        "original_distances": original,
        "embedded_distances": embedded,
        "r_squared": float(r_squared),
    }
```

### 4.3 Analysis Pipeline

Per dataset:

1. **Load distance matrices** (GED and Levenshtein).
2. **Handle inf entries**: For LINUX (test-test pairs are inf) and ALKANE (train-test are inf), restrict to valid submatrix. Report how many graphs/pairs are used.
3. **Classical MDS eigenanalysis** on both matrices. Save full eigenspectra.
4. **SMACOF embedding** at dimensions d ∈ {2, 3, 5, 10} for both matrices.
5. **Compute Stress-1** (Kruskal formula) for each embedding.
6. **Procrustes analysis** comparing MDS(Levenshtein, d) vs MDS(GED, d) for d ∈ {2, 5}.
7. **Shepard diagrams** for both metrics at d=2 and d=5.
8. **Save everything**.

### 4.4 Visualization

#### Main Figure (`embedding_main_figure.pdf`)

**Layout**: 2 rows × 3 columns (6 panels), full-page.

| Panel | Content |
|-------|---------|
| (a) | 2D MDS scatter of Levenshtein distances, colored by class label (IAM Letter LOW). |
| (b) | 2D MDS scatter of GED distances, same dataset, same coloring. |
| (c) | Scree plot: eigenvalues from cMDS for both GED (blue) and Levenshtein (orange), with 0-line. |
| (d) | Stress-1 vs dimension for both metrics across IAM Letter LOW. Dashed lines at 0.05, 0.10, 0.20. |
| (e) | Shepard diagram: original Levenshtein vs embedded Levenshtein at d=2, with isotonic regression. |
| (f) | Shepard diagram: original GED vs embedded GED at d=2. |

#### Scree Plots (`scree_{dataset}.pdf`)

One per dataset. Shows eigenvalue magnitude (y-axis, log scale) vs eigenvalue index (x-axis). Mark the zero crossing clearly. Overlay GED and Levenshtein eigenspectra.

#### 2D MDS Scatter (`mds_scatter_{dataset}.pdf`)

Side-by-side: MDS(Levenshtein) and MDS(GED), both colored by class label. Same aspect ratio, same point size. This is the visual evidence the advisor wants: "similar graphs cluster together in string space."

### 4.5 Handling Large N

For IAM Letter (~750 graphs), MDS is O(n²) per iteration — manageable. For LINUX (1000 graphs), also manageable. SMACOF with 300 iterations and 4 restarts takes ~30s for N=1000.

If eigendecomposition becomes a bottleneck (O(n³)), use `scipy.linalg.eigvalsh` which is optimized for symmetric matrices.

---

## 5. CLI Interface

```bash
python -m benchmarks.eval_embedding.eval_embedding \
    --data-root data/eval \
    --output-dir results/eval_embedding \
    --dimensions 2,3,5,10 \
    --seed 42 \
    --n-procrustes-perms 9999 \
    --csv --plot --table
```

---

## 6. Local Testing Plan

### 6.1 Smoke Test

```python
def test_cmds_on_euclidean():
    # Create 20 points in 3D, compute Euclidean distance matrix
    # cMDS should have 3 positive eigenvalues, rest ≈ 0
    # NEV ratio ≈ 0
    ...

def test_smacof_recovers_known():
    # Create 2D point cloud, compute distance matrix
    # SMACOF at d=2 should recover points (up to rotation/reflection)
    # Stress-1 ≈ 0
    ...

def test_procrustes_identical():
    # Two identical configurations: m² = 0
    ...
```

---

## 7. Acceptance Criteria

1. ✅ cMDS eigenvalue files exist for all 5 datasets.
2. ✅ SMACOF coordinates saved for d ∈ {2, 3, 5, 10} for all datasets.
3. ✅ Stress-1 values computed and saved.
4. ✅ Procrustes analysis completed for d ∈ {2, 5}.
5. ✅ Main figure generated in PDF.
6. ✅ Scree plots and MDS scatter plots generated.
7. ✅ Summary LaTeX table generated.
8. ✅ Code passes `ruff check`.
