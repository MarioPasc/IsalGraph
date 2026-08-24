"""Spearman rho against GED, stratified by graph size ``n``.

Answers a question the per-dataset tables cannot: **where in the size range
does a representation's distance track GED, and where does it stop?**

The stratification is by *equal* ``n`` --- a pair enters stratum ``n`` only when
``n_i == n_j == n``. That is deliberate and it is the whole point of the view:
within a stratum the size null ``|n_i - n_j|`` is **identically zero**, so its
rank correlation is undefined and there is nothing to subtract. Raw rho inside a
stratum is therefore the structural signal, free of the size channel that
dominates the pooled ``all_pairs`` comparison.

Reference GED follows the exact-computability ceiling: ``exact`` where it exists
(Suite 1), and the ``lb``/``ub`` bracket reported as **two separate series**,
never averaged and never interpolated (``approx_ged.md`` section 4).

**Descriptive.** These strata are not a pre-registered family. Nothing here is
an input to F0, F1 or F2, and the multiplicity correction applied in the figures
is local to the figure and stated on it.
"""

from __future__ import annotations
