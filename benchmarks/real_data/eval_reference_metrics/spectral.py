"""Spectral lambda-distance reference metric for the T-28 fidelity re-analysis.

Implements the Euclidean distance between sorted Laplacian eigenvalue vectors of
two graphs, zero-padded to a common length. The primary variant uses the
symmetric normalised Laplacian, whose spectrum is confined to ``[0, 2]`` and is
therefore far less size-dominated than the unnormalised one -- the confound that
dominates the graph-edit-distance reference on these cohorts.

Reference:
    Wilson, R.C. and Zhu, P. (2008). A study of graph spectra for comparing
    graphs and trees. *Pattern Recognition* 41(9):2833-2841.
    DOI 10.1016/j.patcog.2008.03.011
"""

from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

__all__ = [
    "SpectralVariant",
    "laplacian_spectrum",
    "cohort_spectra",
    "spectral_distance_matrix",
]

#: Which Laplacian the spectrum is taken from.
#:
#: ``norm``  -- symmetric normalised Laplacian ``L_sym = I - D^-1/2 A D^-1/2``.
#:              PRIMARY. Spectrum in ``[0, 2]``, weakly size-dependent.
#: ``comb``  -- combinatorial Laplacian ``L = D - A``. Disclosed sensitivity arm.
#: ``adj``   -- adjacency spectrum. Second sensitivity arm.
SpectralVariant: TypeAlias = Literal["norm", "comb", "adj"]

#: Degrees at or below this are treated as isolated when forming ``D^-1/2``.
_DEGREE_EPS = 1e-12


def laplacian_spectrum(
    n: int,
    edges: npt.NDArray[np.integer],
    *,
    variant: SpectralVariant = "norm",
) -> npt.NDArray[np.float64]:
    """Return the sorted spectrum of one graph's Laplacian.

    Eigenvalues are returned in **non-increasing** order, so that the zeros
    that pad a shorter spectrum sit at the tail alongside the structural zero
    eigenvalue every Laplacian carries.

    Args:
        n: Node count. Nodes are ``0..n-1``.
        edges: Array of shape ``(2, m)`` holding undirected edge endpoints.
        variant: Which matrix to take the spectrum of.

    Returns:
        Array of shape ``(n,)``, sorted non-increasing.

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
        matrix = adjacency
    elif variant == "comb":
        matrix = np.diag(adjacency.sum(axis=1)) - adjacency
    elif variant == "norm":
        degrees = adjacency.sum(axis=1)
        # An isolated vertex contributes a zero row and column rather than a
        # division by zero; the cohorts are connected, so this is a guard.
        inv_sqrt = np.where(degrees > _DEGREE_EPS, 1.0 / np.sqrt(np.maximum(degrees, _DEGREE_EPS)), 0.0)
        matrix = np.eye(n, dtype=np.float64) - (inv_sqrt[:, None] * adjacency * inv_sqrt[None, :])
    else:  # pragma: no cover - guarded by the Literal
        raise ValueError(f"unknown spectral variant: {variant!r}")

    values = np.linalg.eigvalsh(matrix)
    return np.sort(values)[::-1].astype(np.float64, copy=False)


def cohort_spectra(
    n_nodes: npt.NDArray[np.integer],
    edge_offsets: npt.NDArray[np.integer],
    edges: npt.NDArray[np.integer],
    *,
    variant: SpectralVariant = "norm",
) -> npt.NDArray[np.float64]:
    """Compute zero-padded spectra for a whole CSR-packed cohort.

    Args:
        n_nodes: Node count per graph, shape ``(G,)``.
        edge_offsets: CSR offsets into *edges*, shape ``(G + 1,)``.
        edges: Endpoint array of shape ``(2, E)``.
        variant: Which matrix to take spectra of.

    Returns:
        Array of shape ``(G, n_max)``, each row a non-increasing spectrum
        right-padded with zeros to the cohort's maximum node count.
    """
    n_graphs = int(n_nodes.shape[0])
    n_max = int(n_nodes.max()) if n_graphs else 0
    out = np.zeros((n_graphs, n_max), dtype=np.float64)
    for i in range(n_graphs):
        n = int(n_nodes[i])
        lo, hi = int(edge_offsets[i]), int(edge_offsets[i + 1])
        out[i, :n] = laplacian_spectrum(n, edges[:, lo:hi], variant=variant)
    return out


def spectral_distance_matrix(
    spectra: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return the dense Euclidean distance matrix between padded spectra.

    Args:
        spectra: Array of shape ``(G, n_max)`` from :func:`cohort_spectra`.

    Returns:
        Symmetric ``(G, G)`` array of float64 distances with a zero diagonal.
    """
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b, then clamp fp-negative residue.
    sq_norms = np.einsum("ij,ij->i", spectra, spectra)
    gram = spectra @ spectra.T
    squared = sq_norms[:, None] + sq_norms[None, :] - 2.0 * gram
    np.maximum(squared, 0.0, out=squared)
    distances = np.sqrt(squared)
    np.fill_diagonal(distances, 0.0)
    # Enforce exact symmetry; the einsum route can differ in the last ulp.
    return (distances + distances.T) * 0.5
