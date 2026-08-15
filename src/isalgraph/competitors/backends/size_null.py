"""The trivial representation: ``encode(G) = n``.

Registered as a backend **against ``competitors/README.md`` §6's advice**,
which says to port it as an analysis script.  PI-approved 2026-08-14.

``|n1 - n2|`` *is* a distance on the representation "the node count", so
registering it makes the null column fall out of every rho table and out of
T-04a's grid by construction.  A printed rho without its null becomes
*unreachable* rather than merely discouraged, and finding 1's own text is
"every printed rho needs the null beside it".  A column produced by a
separate script is a column a later ticket can forget.

**Two guards, both mandatory.**

1. It carries :attr:`Capability.BASELINE`.  ``grid.py`` refuses to select a
   ``BASELINE`` backend as any representation's primary distance, and
   ``available_backends()`` omits it unless asked.
2. **It is outside the frozen confirmatory family.**  It is not a Claim-A
   serialisation and not one of the seven Claim-B comparators.  It changes
   neither ``N_max`` nor ``N_actual``.  It is a descriptive baseline row.
   **Adding it to a family breaks decision 23 -- stop.**
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx

from isalgraph.competitors.base import Budget, Capability, Encoding, ReprBackend
from isalgraph.competitors.registry import register_backend
from isalgraph.errors import NotReversible


class SizeNullBackend(ReprBackend):
    """Encodes a graph as its node count and nothing else."""

    name = "size_null"
    capabilities = frozenset({Capability.BASELINE, Capability.HANDLES_DISCONNECTED})

    def encode(self, graph: nx.Graph, *, budget: Budget | None = None) -> Encoding:
        """Return the node count as a one-symbol encoding.

        ``symbols`` deliberately holds one entry: the representation has a
        single degree of freedom, and an edit distance over it is not
        meaningful.  The ``size_null`` metric reads ``n_nodes`` directly.
        """
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        return Encoding(
            backend=self.name,
            symbols=(str(n),),
            alphabet_size=max(n, 1),
            n_nodes=n,
            n_edges=m,
            text=str(n),
        )

    def decode(self, encoding: Encoding) -> nx.Graph:
        """Always raises: the node count discards every edge."""
        raise NotReversible("size_null discards all structure; that is what makes it a null")


register_backend("size_null", SizeNullBackend)

__all__ = ["SizeNullBackend"]
