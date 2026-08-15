"""graph6 -- McKay's six-bit-packed adjacency serialisation.  **The negative control.**

graph6 is ``N(n)`` followed by ``R(x)``, where ``x`` is the strict upper
triangle of the adjacency matrix read **column-wise** and packed six bits to
a printable ASCII byte at offset 63.  It is a published, implemented,
exactly reversible serialisation, which is what R3.6a asked to see beside
the author-defined ``B_GED`` construction.

**graph6 serialises the labelling it is handed.  It performs no
canonicalisation, and its F3 failure is the finding rather than a defect in
this module.**  Its job in the pool is to isolate one variable: graph6 and
``nauty_graph6`` are byte-identical in format and differ only in *which*
labelling is serialised, so putting them side by side answers R1.2's
uniqueness axis by subtraction instead of by assertion.

Two traps this module exists to close.

**Determinism is not invariance.**  Node labels are normalised through
:func:`~isalgraph.competitors.backends.adjacency.normalised` before
serialisation, so the backend is deterministic *on a given input
labelling*.  It is not, and does not claim to be, invariant under
relabelling.

**``n > 62`` uses the four-byte ``N(n)`` header, and that branch is live.**
Suite 2 reaches ``n = 98``.  The closed form ``1 + ceil(n(n-1)/12)`` is
correct only for ``n <= 62`` -- measured, it under-counts by exactly three
bytes at ``n = 63``, ``64`` and ``98`` -- so it survives here only as a test
oracle inside its range.  ``entropy_bits`` and ``realised_bits`` are counted
from the bytes ``networkx`` emitted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx

from isalgraph.competitors.backends.adjacency import normalised, upper_triangle_columnwise
from isalgraph.competitors.base import (
    Budget,
    Capability,
    Encoding,
    PositionalFrame,
    ReprBackend,
)
from isalgraph.competitors.registry import register_backend

#: ``|Sigma|``: the printable ASCII range 63..126 that graph6 writes into.
ALPHABET_SIZE = 64

#: Byte 126 (``'~'``) introduces the extended ``N(n)`` forms.
_TILDE = 0x7E


def header_length(wire: bytes) -> int:
    """Number of leading bytes ``N(n)`` occupies in *wire*.

    ``N(n)`` is one byte for ``n <= 62``, four bytes (``'~'`` plus three)
    for ``63 <= n <= 258047``, and eight (``'~~'`` plus six) above that.

    Args:
        wire: a graph6 payload with no ``>>graph6<<`` header.

    Returns:
        1, 4 or 8.
    """
    if not wire or wire[0] != _TILDE:
        return 1
    if len(wire) > 1 and wire[1] != _TILDE:
        return 4
    return 8


def unpack_payload(wire: bytes, n_nodes: int) -> tuple[str, ...]:
    """The ``n(n-1)/2`` triangle bits carried by *wire*.

    This is the assertion that keeps the family argument executable: the
    result must equal ``adjacency.serialise(G).symbols`` bit for bit.

    Args:
        wire: a graph6 payload with no ``>>graph6<<`` header and no
            trailing newline.
        n_nodes: order of the graph, used to trim the six-bit padding.

    Returns:
        ``'0'``/``'1'`` per triangle cell, column-wise, length
        ``n(n-1)/2``.
    """
    body = wire[header_length(wire) :]
    bits = "".join(format(byte - 63, "06b") for byte in body)
    return tuple(bits[: n_nodes * (n_nodes - 1) // 2])


def serialise(graph: nx.Graph) -> Encoding:
    """Encode *graph* as graph6.

    Args:
        graph: a ``networkx.Graph``.

    Returns:
        An :class:`Encoding` whose ``symbols`` are one ASCII byte each, whose
        ``wire`` is exactly what ``networkx`` emitted with the trailing
        newline stripped, and whose ``frame`` carries the **unpacked triangle
        bits** -- not the bytes.  ``padded_hamming`` reads ``frame.bits``,
        and a graph6 symbol is a six-bit byte while a frame entry is one
        triangle cell, so zipping ``pairs`` against ``symbols`` would compare
        a byte to a cell.

    Note:
        ``payload_bits`` is recorded separately from ``8 * len(wire)``.
        Claim A's two conventions need both and neither is recoverable from
        the other after the fact.
    """
    import networkx as nx

    normal = normalised(graph)
    n = normal.number_of_nodes()
    # Strip the trailing newline: to_graph6_bytes appends one, and leaving
    # it in silently costs eight realised bits per graph.
    wire = nx.to_graph6_bytes(normal, header=False).rstrip(b"\n")
    pairs, _ = upper_triangle_columnwise(normal)
    payload = unpack_payload(wire, n)
    text = wire.decode("ascii")
    return Encoding(
        backend=Graph6Backend.name,
        symbols=tuple(text),
        alphabet_size=ALPHABET_SIZE,
        n_nodes=n,
        n_edges=normal.number_of_edges(),
        text=text,
        wire=wire,
        payload_bits=n * (n - 1) // 2,
        frame=PositionalFrame(n_nodes=n, pairs=pairs, bits=payload),
    )


class Graph6Backend(ReprBackend):
    """``nx.to_graph6_bytes(G, header=False)``, one ASCII byte per symbol."""

    name = "graph6"
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
            budget: unused.

        Returns:
            The encoding.
        """
        return serialise(graph)

    def decode(self, encoding: Encoding) -> nx.Graph:
        """Rebuild the labelled graph with ``nx.from_graph6_bytes``.

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
            raise ValueError("graph6 encoding carries no wire to decode")
        return nx.Graph(nx.from_graph6_bytes(encoding.wire))

    @classmethod
    def is_available(cls) -> bool:
        """Whether ``networkx`` imports."""
        try:
            import networkx  # noqa: F401
        except ImportError:
            return False
        return True


register_backend("graph6", Graph6Backend)

__all__ = [
    "ALPHABET_SIZE",
    "Graph6Backend",
    "header_length",
    "serialise",
    "unpack_payload",
]
