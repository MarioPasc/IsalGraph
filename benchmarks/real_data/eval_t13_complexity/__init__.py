"""T-13: the controlled complexity experiment (R3.7d).

Replaces the manuscript's unqualified "exponential worst case" with a
characterised one: the cost of a canonical search is governed by ``|Aut(G)|``,
not by size or density.  On the real IAM cohort that cannot be shown, because
``n``, ``m``, density and ``|Aut|`` co-vary; only a within-fixed-``(n, m)``
contrast recovers the effect.  So the experiment is factorial and controlled,
on constructed base graphs, and it runs **every** registered representation --
the search-based canonical forms because the claim is about canonical forms as
a class, and the search-free serialisations as the null arm whose cost must be
flat in ``|Aut|``.

===================  =====================================================
``families``         constructed base graphs with closed-form ``|Aut|``
``symmetry``         ``log10|Aut|``, orbits, 1-WL and triplet partitions
``instrumented``     an instrumented mirror of the frozen reference
``counters``         CLI: operation counts per (graph, encoder)
``schema``           the frozen ``t13.1`` measurement record
``measure``          CLI: the timing runner and its SLURM fan-out
===================  =====================================================

**Exports nothing on purpose.**  Every consumer imports the submodule it needs,
so that a missing optional dependency in one module cannot make the package
unimportable for the others -- ``schema`` in particular must stay readable by an
analysis process with no C++ engine and no ``networkx``.
"""

from __future__ import annotations

__all__: list[str] = []
