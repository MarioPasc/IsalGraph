"""sparse6 -- McKay's edge-list serialisation.  **IsalGraph's only rival on bits.**

sparse6 is ``':'`` + ``N(n)`` + a bit stream of ``(b_i, x_i)`` groups, where
``b_i`` is one flag bit and ``x_i`` is a ``k = ceil(log2 n)``-bit vertex
index.  Decoding walks a cursor upward, incrementing it on ``b_i = 1``, and
emits ``{x_i, v}``.  Its length scales with ``m log n`` rather than ``n^2``,
which makes it the one competitor whose compactness is a genuine rival to
IsalGraph's -- it wins at ``m/n ~ 2`` and loses at ``m/n ~ 1``.

Three properties this module has to get right.

**The ``':'`` prefix is framing, not payload.**  It is excluded from
``entropy_bits`` (``6 len(wire) - 6``) and included in ``realised_bits``
(``8 len(wire)``).  That is decided once, in ``bits.py``'s table, and never
again in a script.  ``symbols`` likewise excludes it, per CONTRACTS §9.

**There is no positional frame, and therefore no padded Hamming.**  sparse6
is not a positional bit vector, so there is nothing to pad into.  This
backend does **not** declare :attr:`Capability.POSITIONAL_FRAME` and leaves
``Encoding.frame`` unset, which makes ``padded_hamming`` report *undefined*.
That ``undefined`` is a **result** that goes in the supplementary grid, not
an error to work around.

**Length varies with ``m``, so plain Hamming is defined on only 30.8 % of
one-edit pairs** -- the concrete case the T-04a grid was written to catch.

.. warning::

   ``networkx`` emits sparse6 with ``k = ceil(log2 n)``, and the spec has an
   off-by-one special case when ``n`` is a power of two.  Suite 2 contains
   graphs at ``n = 16, 32, 64``.  :meth:`Sparse6Backend.decode` is asserted
   round-trip-exact in the test suite at each of those sizes rather than any
   length formula being trusted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx

from isalgraph.competitors.backends.adjacency import normalised
from isalgraph.competitors.base import Budget, Capability, Encoding, ReprBackend
from isalgraph.competitors.registry import register_backend

#: ``|Sigma|``: the printable ASCII range 63..126 that sparse6 writes into.
ALPHABET_SIZE = 64


def serialise(graph: nx.Graph) -> Encoding:
    """Encode *graph* as sparse6.

    **This is the one cross-track entry point of wave 1.**  Agent B imports
    it to register ``sparse6_nauty``: it canonicalises the labelling with
    ``pynauty`` and then calls this function, so the two rows differ in
    exactly one step and in nothing else.  The signature is frozen in
    CONTRACTS §4 and may not be changed without the orchestrator.

    Args:
        graph: a ``networkx.Graph``.

    Returns:
        An :class:`Encoding` whose ``wire`` is exactly what ``networkx``
        emitted with the trailing newline stripped, whose ``symbols`` are one
        ASCII byte each **excluding the leading** ``':'``, whose ``text``
        includes it, and whose ``frame`` is ``None``.
    """
    import networkx as nx

    normal = normalised(graph)
    # Strip the trailing newline: to_sparse6_bytes appends one, and leaving
    # it in silently costs eight realised bits per graph.
    wire = nx.to_sparse6_bytes(normal, header=False).rstrip(b"\n")
    text = wire.decode("ascii")
    return Encoding(
        backend=Sparse6Backend.name,
        # ':' is framing.  It is in `text` and in `wire`, and therefore in
        # realised_bits, but it is not a unit of edit and not payload.
        symbols=tuple(text[1:]) if text.startswith(":") else tuple(text),
        alphabet_size=ALPHABET_SIZE,
        n_nodes=normal.number_of_nodes(),
        n_edges=normal.number_of_edges(),
        text=text,
        wire=wire,
        payload_bits=None,
        frame=None,
    )


class Sparse6Backend(ReprBackend):
    """``nx.to_sparse6_bytes(G, header=False)``, one ASCII byte per symbol."""

    name = "sparse6"
    capabilities = frozenset(
        {
            Capability.REVERSIBLE,
            Capability.HANDLES_DISCONNECTED,
        }
    )

    def encode(self, graph: nx.Graph, *, budget: Budget | None = None) -> Encoding:
        """Encode *graph*.  ``budget`` is ignored: there is no search.

        Args:
            graph: a ``networkx.Graph``.
            budget: unused.

        Returns:
            The encoding.
        """
        return serialise(graph)

    def decode(self, encoding: Encoding) -> nx.Graph:
        """Rebuild the labelled graph with ``nx.from_sparse6_bytes``.

        Exactly reversible -- not merely up to isomorphism -- to the
        normalised labelling.

        Args:
            encoding: an encoding produced by this backend.

        Returns:
            The graph.

        Raises:
            ValueError: if *encoding* is from another backend or carries no
                ``wire``.
        """
        import networkx as nx

        if encoding.backend != self.name:
            raise ValueError(f"encoding is from {encoding.backend!r}, not {self.name!r}")
        if encoding.wire is None:
            raise ValueError("sparse6 encoding carries no wire to decode")
        return nx.Graph(nx.from_sparse6_bytes(encoding.wire))

    @classmethod
    def is_available(cls) -> bool:
        """Whether ``networkx`` imports."""
        try:
            import networkx  # noqa: F401
        except ImportError:
            return False
        return True


register_backend("sparse6", Sparse6Backend)

__all__ = ["ALPHABET_SIZE", "Sparse6Backend", "serialise"]
