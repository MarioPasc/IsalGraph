"""Production reference distance matrices for the T-28 fidelity re-analysis.

Provides four spectral reference variants (§3 of the T-28 design note) and a
thin wrapper that copies the cached WL kernel matrix into the T-28 reference
tree.  All outputs conform to the CONTRACTS §4 dense NPZ schema
(``DENSE_KEYS``).

Modules:
    spectral  -- spectral distance computation (Euclidean + 1-Wasserstein).
    build     -- cohort loading, gate checking, and NPZ writing for all 15
                 (suite, dataset) cells.
"""

from __future__ import annotations

from isalgraph.competitors.references import build, spectral

__all__ = ["build", "spectral"]
