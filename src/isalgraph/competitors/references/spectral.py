"""Spectral reference metrics for the T-28 fidelity re-analysis.

Implements four variants of spectral graph distance, as specified in the T-28
design note §3.  The primary variant (``spectral``) uses the symmetric
normalised Laplacian with zero-padded Euclidean distance.  The
``spectral_esd`` variant applies 1-Wasserstein between the empirical spectral
distributions, which is size-controlled by construction and must not be
zero-padded.

Reference:
    Wilson, R.C. and Zhu, P. (2008). A study of graph spectra for comparing
    graphs and trees. *Pattern Recognition* 41(9):2833-2841.
    DOI 10.1016/j.patcog.2008.03.011
"""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.stats import wasserstein_distance

__all__ = [
    "SpectralVariant",
    "laplacian_spectrum",
    "cohort_spectra",
    "spectral_distance_matrix",
    "spectral_esd_matrix",
]

#: Which Laplacian (or adjacency matrix) the spectrum is taken from.
#:
#: ``norm`` -- symmetric normalised Laplacian ``L_sym = I - D^-1/2 A D^-1/2``.
#:             PRIMARY. Spectrum confined to ``[0, 2]``.
#: ``comb`` -- combinatorial Laplacian ``L = D - A``. Sensitivity arm.
#: ``adj``  -- adjacency matrix. Second sensitivity arm.
SpectralVariant: TypeAlias = Literal["norm", "comb", "adj"]

#: Degree threshold below which a vertex is considered isolated.  An isolated
#: vertex contributes zero rows/columns in D^-1/2, never a division by zero.
_DEGREE_EPS: float = 1e-12


def laplacian_spectrum(
    n: int,
    edges: npt.NDArray[np.integer[Any]],
    *,
    variant: SpectralVariant = "norm",
) -> npt.NDArray[np.float64]:
    """Return the sorted spectrum of one graph's Laplacian (or adjacency matrix).

    Eigenvalues are returned in **non-increasing** order so that the structural
    zero eigenvalue and any zero padding sit at the tail.

    Args:
        n: Number of nodes.  Nodes are labelled ``0..n-1``.
        edges: Integer array of shape ``(2, m)`` holding undirected edge
            endpoints.  Parallel edges and self-loops are ignored via the
            adjacency fill (duplicates overwrite with the same value).
        variant: Which matrix to diagonalise.

    Returns:
        Array of shape ``(n,)`` with eigenvalues sorted non-increasing.

    Raises:
        ValueError: If *variant* is not a recognised key.
    """
    adjacency = np.zeros((n, n), dtype=np.float64)
    if edges.size:
        rows = edges[0].astype(np.intp)
        cols = edges[1].astype(np.intp)
        adjacency[rows, cols] = 1.0
        adjacency[cols, rows] = 1.0
    np.fill_diagonal(adjacency, 0.0)

    if variant == "adj":
        matrix: npt.NDArray[np.float64] = adjacency
    elif variant == "comb":
        matrix = np.diag(adjacency.sum(axis=1)) - adjacency
    elif variant == "norm":
        degrees = adjacency.sum(axis=1)
        # Isolated vertices (degree 0) get inv_sqrt = 0, so their rows/columns
        # in the normalised Laplacian are zero rather than NaN.
        inv_sqrt = np.where(
            degrees > _DEGREE_EPS,
            1.0 / np.sqrt(np.maximum(degrees, _DEGREE_EPS)),
            0.0,
        )
        matrix = np.eye(n, dtype=np.float64) - (
            inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
        )
    else:  # pragma: no cover
        raise ValueError(f"unknown spectral variant {variant!r}")

    values: npt.NDArray[np.float64] = np.linalg.eigvalsh(matrix)
    return np.sort(values)[::-1].astype(np.float64, copy=False)


def cohort_spectra(
    n_nodes: npt.NDArray[np.integer[Any]],
    edge_offsets: npt.NDArray[np.integer[Any]],
    edges: npt.NDArray[np.integer[Any]],
    *,
    variant: SpectralVariant = "norm",
) -> npt.NDArray[np.float64]:
    """Return zero-padded spectra for every graph in a cohort.

    Each row holds a non-increasing spectrum zero-padded to the cohort's
    maximum node count ``n_max``.  The structural zero eigenvalue and the
    padding zeros therefore sit at the tail and are indistinguishable, which is
    intentional: padding turns the Euclidean distance between rows into a pure
    spectral distance without truncation artefacts.

    Args:
        n_nodes: Integer array of shape ``(G,)`` with per-graph node counts.
        edge_offsets: Integer array of shape ``(G+1,)`` s.t. graph ``i`` has
            edges ``edges[:, edge_offsets[i]:edge_offsets[i+1]]``.
        edges: Integer array of shape ``(2, E)`` holding edge endpoints.
        variant: Laplacian variant to diagonalise.

    Returns:
        Float64 array of shape ``(G, n_max)``.
    """
    n_graphs = int(n_nodes.shape[0])
    n_max = int(n_nodes.max()) if n_graphs else 0
    out = np.zeros((n_graphs, n_max), dtype=np.float64)
    for i in range(n_graphs):
        n = int(n_nodes[i])
        lo = int(edge_offsets[i])
        hi = int(edge_offsets[i + 1])
        out[i, :n] = laplacian_spectrum(n, edges[:, lo:hi], variant=variant)
    return out


def spectral_distance_matrix(
    spectra: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Euclidean distance matrix between zero-padded spectra.

    Uses the identity ``||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>`` to avoid
    materialising G^2 difference vectors.  Floating-point residuals that fall
    slightly below zero are clamped before the square root.

    Args:
        spectra: Float64 array of shape ``(G, n_max)`` from
            :func:`cohort_spectra`.

    Returns:
        Symmetric float64 array of shape ``(G, G)`` with zero diagonal.
    """
    sq_norms = np.einsum("ij,ij->i", spectra, spectra)
    gram = spectra @ spectra.T
    squared = sq_norms[:, None] + sq_norms[None, :] - 2.0 * gram
    np.maximum(squared, 0.0, out=squared)
    distances = np.sqrt(squared)
    np.fill_diagonal(distances, 0.0)
    # Enforce exact symmetry; floating-point can produce single-ulp asymmetry.
    return np.asarray((distances + distances.T) * 0.5, dtype=np.float64)


def spectral_esd_matrix(
    n_nodes: npt.NDArray[np.integer[Any]],
    edge_offsets: npt.NDArray[np.integer[Any]],
    edges: npt.NDArray[np.integer[Any]],
) -> npt.NDArray[np.float64]:
    """1-Wasserstein distance matrix between L_sym empirical spectral distributions.

    Each graph's spectrum is treated as a discrete uniform measure on ``n``
    atoms, so the distance is size-controlled by construction and must NOT be
    zero-padded.  Uses :func:`scipy.stats.wasserstein_distance`, which computes
    the exact 1-Wasserstein for 1D distributions from sample arrays via the
    integral of the absolute difference of CDFs.

    For unequal atom counts ``n`` and ``m``, the closed-form is the L1 distance
    between quantile functions evaluated over the merged sorted support.
    ``scipy.stats.wasserstein_distance`` implements precisely this; see
    test_t28_references.py for a hand-computed verification on a 3-vs-2-atom
    example.

    Args:
        n_nodes: Integer array of shape ``(G,)`` with per-graph node counts.
        edge_offsets: Integer array of shape ``(G+1,)`` in CSR format.
        edges: Integer array of shape ``(2, E)``.

    Returns:
        Symmetric float64 array of shape ``(G, G)`` with zero diagonal.
    """
    n_graphs = int(n_nodes.shape[0])
    raw: list[npt.NDArray[np.float64]] = []
    for i in range(n_graphs):
        n = int(n_nodes[i])
        lo = int(edge_offsets[i])
        hi = int(edge_offsets[i + 1])
        raw.append(laplacian_spectrum(n, edges[:, lo:hi], variant="norm"))

    out = np.zeros((n_graphs, n_graphs), dtype=np.float64)
    for i in range(n_graphs):
        for j in range(i + 1, n_graphs):
            d = float(wasserstein_distance(raw[i], raw[j]))
            out[i, j] = d
            out[j, i] = d
    return out
