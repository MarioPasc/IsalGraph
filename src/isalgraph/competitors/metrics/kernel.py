"""RKHS distance for a :class:`VectorBackend`.

``d(G,H) = sqrt(K(G,G) + K(H,H) - 2 K(G,H))`` with ``K`` the linear kernel
on the feature multisets.

**Declared a pseudometric.**  Identity of indiscernibles fails: ``K_{3,3}``
and the triangular prism are not isomorphic and get distance exactly
``0.0000``.  ``competitors.md`` §3.3 F2 requires the declaration rather than
inference, and that failure is the whole reason WL is in the pool.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from isalgraph.competitors.base import DistanceMetric
from isalgraph.competitors.registry import register_metric

Features = Mapping[str, int]


def linear_kernel(a: Features, b: Features) -> float:
    """Dot product of two feature multisets over their shared vocabulary.

    The vocabulary is shared because the backend fitted it **per dataset**.
    A per-batch fit produces a different vocabulary and therefore different
    distances, which makes the distance matrix depend on batching order.
    """
    if len(a) > len(b):
        a, b = b, a
    return float(sum(count * b.get(key, 0) for key, count in a.items()))


class KernelDistance(DistanceMetric[Features]):
    """RKHS distance induced by :func:`linear_kernel`."""

    name = "kernel"
    consumes = "features"
    is_pseudometric = True

    def is_defined(self, a: Features, b: Features) -> bool:
        """Always defined on two fitted feature maps."""
        return True

    def distance(self, a: Features, b: Features) -> float:
        squared = linear_kernel(a, a) + linear_kernel(b, b) - 2.0 * linear_kernel(a, b)
        # Clamp only the floating-point epsilon below zero, never a real
        # negative: a genuinely negative Gram entry means the kernel is not
        # positive semi-definite and must surface, not be hidden.
        if squared < -1e-9:
            raise ValueError(
                f"linear WL kernel produced a negative squared distance ({squared!r}); "
                f"the Gram matrix is not PSD and the feature vocabulary is suspect"
            )
        return math.sqrt(max(squared, 0.0))


register_metric("kernel", KernelDistance)

__all__ = ["KernelDistance", "linear_kernel"]
