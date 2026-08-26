"""T-13: the characterised worst case of the IsalGraph canonical search (R3.7d).

The ticket replaces the manuscript's unqualified "exponential worst case" with a
statement about *what* drives the cost: ``|Aut(G)|``, not order or density.  On
the real IAM cohort that claim cannot be isolated, because ``n``, ``m``, density
and ``|Aut|`` co-vary; the primary evidence is therefore a controlled experiment
on constructed base graphs where one factor moves at a time.

Modules:
    families: the constructed graph families with closed-form ``|Aut|``, and the
        ``symmetry_ladder`` matched design that holds ``(n, m)`` and the whole
        degree sequence fixed while ``|Aut|`` falls.
    symmetry: the partition-resolution toolkit -- ``|Aut|``, orbits, 1-WL, the
        incumbent triplet key, and the exact refinement test that Proposition 1
        is checked with.
    instrumented, counters: the operation counters (owner B).
    schema, measure: the record schema and the measurement CLI (owner C).

This package deliberately exports nothing: every consumer imports the module it
needs by name, so that a missing peer module is an ``ImportError`` at the call
site rather than a silent absence at package import.
"""

from __future__ import annotations

__all__: list[str] = []
