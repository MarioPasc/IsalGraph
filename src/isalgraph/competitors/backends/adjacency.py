"""The raw adjacency matrix -- the ``n^2`` reference point, and the reading order.

**This module owns the reading order for the whole ``n^2`` family.**
``adjacency``, ``graph6``, ``nauty_graph6`` and ``agm_cam`` must all produce
the *same bit sequence* for the same labelling, and they do so by importing
:func:`normalised` and :func:`upper_triangle_columnwise` from here rather
than by each re-deriving the triangle.  That is what keeps
``competitors/README.md`` §2's four-member-family argument true in code
rather than in prose.

**The order is the strict upper triangle read COLUMN-WISE**::

    a(0,1)  a(0,2) a(1,2)  a(0,3) a(1,3) a(2,3)  ...

Row-major would break the correspondence with graph6's payload for no
benefit.  Verified, not assumed: graph6 ``'ElCW'`` unpacks to
``'101101000100011'``, which is exactly the column-wise triangle of the
running example ``C4(0,1,2,3) + K3(3,4,5)``.

The matrix is **not** isomorphism-invariant and it is **not** a complete
invariant.  Both are the finding, not a defect: it is in the pool as the
control every reviewer already has in mind, and as the denominator for
Claim A.  It handles disconnected graphs and isolated vertices without
ceremony, which IsalGraph does not.

.. warning::

   ``Encoding.text`` -- the ``'101101...'`` rendering -- is a **debugging
   view**.  Counting it as eight bits per character inflates the adjacency
   matrix eightfold and hands IsalGraph a baseline it beats for free.  The
   bit count comes from :mod:`isalgraph.competitors.bits` and from nowhere
   else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx

from isalgraph.competitors.base import (
    Budget,
    Capability,
    Encoding,
    PositionalFrame,
    ReprBackend,
)
from isalgraph.competitors.registry import register_backend

#: ``|Sigma|`` for the triangle: the symbols *are* bits.
ALPHABET_SIZE = 2


def normalised(graph: nx.Graph) -> nx.Graph:
    """A fresh graph on ``0..n-1``, labelled by ``sorted(graph.nodes())``.

    **This is the one normalisation the whole ``n^2`` family shares**, and it
    is deliberately stronger than the ``nx.convert_node_labels_to_integers(
    G, ordering="sorted")`` that ``competitors/graph6.md`` §7 prescribes.

    That call relabels the node *values* but leaves the graph's **insertion
    order** untouched, and ``nx.to_graph6_bytes`` then re-derives its own
    labels from the insertion order (``readwrite/graph6.py``: ``H =
    convert_node_labels_to_integers(G)``, default ordering).  Measured on
    2026-08-15 over 300 graphs whose insertion order was scrambled while
    their labels were held fixed, the two disagree on **260**.  Following
    §7 literally therefore yields a backend that is deterministic on a
    given *(labelling, insertion order)* pair rather than on a given
    labelling.

    Rebuilding the graph with nodes added in ascending order makes the
    subsequent ``networkx`` conversion the identity, so ``graph6`` and
    ``sparse6`` -- which reach their labels by two *different* code paths
    inside ``networkx`` -- serialise the same labelling as this triangle.

    Args:
        graph: any ``networkx.Graph``; node labels need only be sortable.

    Returns:
        A new ``networkx.Graph`` with nodes ``0..n-1`` inserted in ascending
        order, self-loops dropped.

    Note:
        This is **determinism, not isomorphism invariance**.  Relabelling
        the input changes the output, which is exactly the F3 failure these
        three backends are in the pool to demonstrate.
    """
    import networkx as nx

    mapping = {old: new for new, old in enumerate(sorted(graph.nodes()))}
    out = nx.Graph()
    out.add_nodes_from(range(len(mapping)))
    out.add_edges_from((mapping[u], mapping[v]) for u, v in graph.edges() if u != v)
    return out


def upper_triangle_columnwise(
    graph: nx.Graph,
) -> tuple[tuple[tuple[int, int], ...], tuple[str, ...]]:
    """Strict upper triangle of *graph*, read column-wise.

    *graph* must already be :func:`normalised`, i.e. its nodes are exactly
    ``0..n-1``.

    Args:
        graph: a normalised ``networkx.Graph``.

    Returns:
        ``(pairs, bits)``: the ``(i, j)`` cells with ``i < j`` in
        ``a(0,1) a(0,2) a(1,2) a(0,3) ...`` order, and the ``'0'``/``'1'``
        entry for each.  Both have length ``n(n-1)/2``.
    """
    n = graph.number_of_nodes()
    present = {(u, v) if u < v else (v, u) for u, v in graph.edges()}
    pairs: list[tuple[int, int]] = []
    bits: list[str] = []
    for j in range(1, n):
        for i in range(j):
            pairs.append((i, j))
            bits.append("1" if (i, j) in present else "0")
    return tuple(pairs), tuple(bits)


def serialise(graph: nx.Graph) -> Encoding:
    """Encode *graph* as its column-wise strict upper triangle.

    Args:
        graph: a ``networkx.Graph``.

    Returns:
        An :class:`Encoding` whose ``symbols`` are one triangle bit each and
        whose ``frame`` carries the same bits against their ``(i, j)`` cells.

    Note:
        ``wire`` is deliberately ``None``.  The adjacency matrix has no
        format-defined serialisation to measure -- ``bits.py`` derives its
        realised cost from ``n`` alone -- and populating ``wire`` with an
        invented packing is exactly what ``adjacency-matrix.md`` §4 forbids.
    """
    normal = normalised(graph)
    n = normal.number_of_nodes()
    pairs, bits = upper_triangle_columnwise(normal)
    return Encoding(
        backend=AdjacencyBackend.name,
        symbols=bits,
        alphabet_size=ALPHABET_SIZE,
        n_nodes=n,
        n_edges=normal.number_of_edges(),
        text="".join(bits),
        wire=None,
        payload_bits=n * (n - 1) // 2,
        frame=PositionalFrame(n_nodes=n, pairs=pairs, bits=bits),
    )


class AdjacencyBackend(ReprBackend):
    """The strict upper triangle, column-wise, one symbol per bit."""

    name = "adjacency"
    capabilities = frozenset(
        {
            Capability.POSITIONAL_FRAME,
            Capability.REVERSIBLE,
            Capability.HANDLES_DISCONNECTED,
        }
    )

    def encode(self, graph: nx.Graph, *, budget: Budget | None = None) -> Encoding:
        """Encode *graph*.  ``budget`` is ignored: there is no search.

        Args:
            graph: a ``networkx.Graph``.
            budget: unused; the encode is ``Theta(n^2)`` and cannot fail.

        Returns:
            The encoding.
        """
        return serialise(graph)

    def decode(self, encoding: Encoding) -> nx.Graph:
        """Rebuild the labelled graph from the triangle.

        Exactly reversible -- not merely up to isomorphism -- to the
        :func:`normalised` labelling, which is the input labelling whenever
        the input nodes are ``0..n-1``.

        Args:
            encoding: an encoding produced by this backend.

        Returns:
            The graph.

        Raises:
            ValueError: if *encoding* did not come from this backend.
        """
        import networkx as nx

        if encoding.backend != self.name:
            raise ValueError(f"encoding is from {encoding.backend!r}, not {self.name!r}")
        frame = encoding.frame
        pairs: tuple[tuple[int, int], ...]
        if frame is not None:
            pairs, bits = frame.pairs, frame.bits
        else:  # pragma: no cover - serialise always populates the frame
            n = encoding.n_nodes
            pairs = tuple((i, j) for j in range(1, n) for i in range(j))
            bits = encoding.symbols
        out = nx.Graph()
        out.add_nodes_from(range(encoding.n_nodes))
        out.add_edges_from(pair for pair, bit in zip(pairs, bits, strict=True) if bit == "1")
        return out

    @classmethod
    def is_available(cls) -> bool:
        """Whether ``networkx`` imports."""
        try:
            import networkx  # noqa: F401
        except ImportError:
            return False
        return True


register_backend("adjacency", AdjacencyBackend)

__all__ = [
    "ALPHABET_SIZE",
    "AdjacencyBackend",
    "normalised",
    "serialise",
    "upper_triangle_columnwise",
]
