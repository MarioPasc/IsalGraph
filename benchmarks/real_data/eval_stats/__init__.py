"""T-06 statistics engine: the graph-level protocol D1-D15 and the frozen family.

The submitted manuscript computed significance with an asymptotic Spearman test
over pair counts, treating ``d(G1, G2)`` and ``d(G1, G3)`` as independent when
they share ``G1``. Reviewer 3 identified this (R3.5c). This package replaces
that machinery: **all uncertainty comes from a graph-level cluster bootstrap
(D2) and all significance from a Mantel permutation test (D3)**, so the
resampling unit is the graph and never the pair.

Modules:
    :mod:`resampling`
        D2 graph-level cluster bootstrap, D7 paired differences, D15 tiers.
    :mod:`association`
        D1 Spearman / Kendall, D3 Mantel, D4 multiple regression on matrices.
    :mod:`multiplicity`
        D8 Friedman + Wilcoxon-Holm + critical difference, D9 Benjamini-Hochberg.
    :mod:`family`
        The frozen F0 / F1 / F2 confirmatory family and ``N_actual``.
    :mod:`matrices`
        Read-only loaders for the T-05 GED matrices and T-06 distance matrices.

``benchmarks.real_data.eval_correlation.correlation_metrics.bootstrap_correlation``
resamples **pairs** and is the defect this package replaces. It is never
imported here, and ``tests/unit/test_t06_stats.py`` asserts so by inspecting the
import closure by object identity.
"""

from __future__ import annotations

__all__ = ["association", "family", "matrices", "multiplicity", "resampling"]
