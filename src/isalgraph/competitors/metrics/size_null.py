"""``|n1 - n2|`` -- the distance on the trivial representation.

Count the nodes, subtract.  No representation at all.

Against T-03's certified exact GED it scores **0.899 / 0.909 / 0.926 /
0.713 / 0.799** on the five Suite-1 datasets, and IsalGraph clears it on two
of five by at most 0.03.  Every printed rho needs this column beside it, or
the first reviewer to compute it has an unanswerable objection.

It is registered as a metric, and its representation as a backend, so that
the null column falls out of every table **by construction**.  A column
produced by a separate script is a column a later ticket can forget -- which
is exactly how this stayed unowned until the scout tripped over it.

**A pseudometric, declared**: two non-isomorphic graphs on ``n`` nodes get
distance 0.
"""

from __future__ import annotations

from isalgraph.competitors.base import DistanceMetric, Encoding
from isalgraph.competitors.registry import register_metric


class SizeNullDistance(DistanceMetric[Encoding]):
    """``|n1 - n2|``.  Reads node counts; never touches the symbols."""

    name = "size_null"
    consumes = "order"
    is_pseudometric = True

    def is_defined(self, a: Encoding, b: Encoding) -> bool:
        """Always defined: every graph has a node count."""
        return True

    def distance(self, a: Encoding, b: Encoding) -> float:
        return float(abs(a.n_nodes - b.n_nodes))


register_metric("size_null", SizeNullDistance)

__all__ = ["SizeNullDistance"]
