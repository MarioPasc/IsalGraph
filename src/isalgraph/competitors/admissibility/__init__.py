"""Metric-admissibility experiments E1-E4 for the T-04a annex.

The T-04a grid decided **which** representations admit a distance under
``competitors.md`` §3.4.  This package establishes **why**, and separates two
failure modes that are routinely conflated:

======  ==============================  =========================  =====================
class   ``d(G, π(G)) = 0``?             ``d = 0 ⇒ G ≅ H``?         on isomorphism classes
======  ==============================  =========================  =====================
I       **no**                          --                         not a well-defined map
II      yes                             **no**                     a pseudometric
III     yes                             yes                        a metric
======  ==============================  =========================  =====================

Class I *over*-discriminates -- it separates graphs that are the same.  Class
II *under*-discriminates -- it identifies graphs that are different.  They are
opposite defects and an argument that excludes one does not touch the other.

Frozen protocol, including every pre-declared decision:
``.claude/notes/review/tasks/T-04a-admissibility-protocol.md``.
"""

from __future__ import annotations

__all__ = ["common"]
