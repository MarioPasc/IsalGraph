"""nauty canonical labelling -> graph6, and the canonicalised sparse6 variant.

Two registered backends, both built on one public function,
:func:`canonical_relabel`:

- ``nauty_graph6`` is **the pool's most important control**.  It is graph6
  with exactly one variable changed -- the labelling -- so the bit count is
  identical to graph6 *by construction* and every difference between the
  two columns is attributable to canonicalisation alone.  Holding the
  format fixed, that single substitution moves F3 from ``4/50`` to
  ``50/50`` and equal-``n`` rho from 0.539 to 0.974 on Letter LOW
  (``competitors/nauty.md`` §3).
- ``sparse6_nauty`` is ``sparse6.serialise(canonical_relabel(G))``.  It
  removes the objection that we compared a canonical method against a
  non-canonical one on the compactness axis.  It is **supplementary, not a
  family member**: ``preregistration.md`` §4.1 freezes the comparator sets
  at 6 and 7, and decision 23 freezes ``N_max``.

Cite: McKay & Piperno, *Practical graph isomorphism, II*, J. Symb. Comput.
60:94-112, 2014, doi:10.1016/j.jsc.2013.09.003.  For the exponential worst
case: Miyazaki, *The complexity of McKay's canonical labeling algorithm*,
Groups and Computation II, DIMACS 28:239-256, 1997.

The ``canon_label`` inversion, and what actually catches it
----------------------------------------------------------

``pynauty.canon_label`` returns, **for each new position, the old vertex**.
To relabel you need the inverse, ``{old: new for new, old in
enumerate(lab)}``.  Using ``lab`` directly as ``old -> new`` produces a
different but still deterministic labelling.

``competitors/nauty.md`` §1 and §7 instruct: *"Assert
``nx.is_isomorphic(G, relabelled)`` on every encode"*, on the grounds that
the inverted labelling *"will pass an invariance test and be wrong"*.
**Both halves of that were measured on 2026-08-15 and both are false**, so
this module states what the guards can and cannot do rather than inheriting
a guard that cannot fail:

1. **The inverted labelling does not pass F3 -- it fails it loudly.**  Over
   20 genuine relabellings the inverted code took 15 / 19 / 5 / 13 distinct
   values on the running example, ``G - (0,3)``, ``K_{3,3}`` and the prism,
   and 30 of 30 random ``n = 8`` graphs were non-invariant.  Reason:
   ``lab_{G'} = pi_G^{-1} tau`` for ``G' = G^tau``, so the wrong-direction
   image is ``G^{tau pi_G^{-1} tau}``, which depends on ``tau``.
2. **``nx.is_isomorphic`` cannot catch it, ever.**  Any bijective
   relabelling of ``G`` is isomorphic to ``G`` by construction, so the
   assertion is satisfied for *every* permutation, correct or inverted.  It
   was True on 100 % of the inverted cases above.

What the isomorphism check *does* catch is an index-mapping fault -- the
realistic bug, since ``pynauty`` requires vertices ``0..n-1`` while a
``networkx`` graph may carry arbitrary labels.  So both guards are kept and
both are honest about their job: :func:`canonical_relabel` always verifies
that the map is a bijection onto ``range(n)`` and that no two edges
collided (together an ``O(n + m)`` *proof* of isomorphism), and
additionally runs ``nx.is_isomorphic`` when ``verify=True``, which is the
default.

``verify=False`` exists for the language-matched Fig. 2 timing mode:
``nx.is_isomorphic`` costs 6.7 ms at ``n = 96`` against 0.33 ms for the
whole relabelling, a 20x tax, and ``nauty.md`` §2's published encode cost
of 0.042-0.351 ms is the unverified figure.

``pynauty.certificate()`` is **not** a substitute for the graph6 route in a
comparison table: it is a padded machine-word bit matrix, so its length is
a function of the word size, not of the graph.  It is exposed here for
isomorphism assertions only.
"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

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
from isalgraph.errors import CompetitorError

#: graph6's printable-ASCII alphabet: bytes 63..126 inclusive.
GRAPH6_ALPHABET_SIZE = 64

#: sparse6 uses the same 64-character alphabet plus the ``':'`` framing byte,
#: which ``bits.py`` excludes from the entropy bound.
SPARSE6_ALPHABET_SIZE = 64


def _pynauty() -> ModuleType:
    """Import ``pynauty`` lazily.

    Third-party imports live inside function bodies so that
    ``import isalgraph.competitors`` succeeds with ``pynauty`` uninstalled.
    """
    import pynauty

    module: ModuleType = pynauty
    return module


def _sparse6_serialise() -> Callable[[nx.Graph], Encoding]:
    """Resolve agent A's ``sparse6.serialise``, CONTRACTS.md §4's one cross-edge.

    Reached through :func:`importlib.import_module` and a ``cast`` rather
    than a ``from ... import``, because in an isolated wave-1 worktree agent
    A's module does not exist and a static import makes ``mypy --strict``
    fail on a file that is correct.  The cast writes out the frozen
    signature, and ``test_competitors_canonical.py`` asserts the real
    function matches it once the branches merge.
    """
    module = importlib.import_module("isalgraph.competitors.backends.sparse6")
    return cast("Callable[[nx.Graph], Encoding]", module.serialise)


def _to_pynauty(graph: nx.Graph) -> tuple[Any, list[Any]]:
    """Build a ``pynauty.Graph`` and return it with its vertex order.

    ``pynauty`` indexes vertices ``0..n-1``; a ``networkx`` graph may carry
    any hashable label.  The returned list is the index -> label map, and
    getting it wrong is the fault the isomorphism guard exists to catch.
    """
    pynauty = _pynauty()
    nodes = list(graph.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    adjacency = {i: [index[w] for w in graph.neighbors(v)] for i, v in enumerate(nodes)}
    return pynauty.Graph(len(nodes), directed=False, adjacency_dict=adjacency), nodes


def canonical_relabel(graph: nx.Graph, *, verify: bool = True) -> nx.Graph:
    """Relabel *graph* onto ``0..n-1`` under nauty's canonical labelling.

    Isomorphic inputs produce the **identical** output graph, node ids
    included, which is what makes the graph6 serialisation of the result a
    complete invariant.

    Args:
        graph: a ``networkx.Graph``.  Node labels may be arbitrary.
        verify: additionally run ``nx.is_isomorphic``.  The ``O(n + m)``
            bijection and edge-count checks run unconditionally and already
            *prove* isomorphism; this adds an independent oracle at a
            measured 20x cost at ``n = 96``.  See the module docstring for
            why it cannot catch the ``canon_label`` inversion.

    Returns:
        A fresh graph on ``range(n)``.

    Raises:
        CompetitorError: if the canonical labelling is not a bijection onto
            ``range(n)``, if two edges collided, or if *verify* is set and
            the result is not isomorphic to the input.
    """
    import networkx as nx

    pynauty = _pynauty()
    nodes = list(graph.nodes())
    n = len(nodes)
    out = nx.Graph()
    out.add_nodes_from(range(n))
    if n == 0:
        return out

    index = {v: i for i, v in enumerate(nodes)}
    adjacency = {i: [index[w] for w in graph.neighbors(v)] for i, v in enumerate(nodes)}
    labelling = pynauty.canon_label(pynauty.Graph(n, directed=False, adjacency_dict=adjacency))
    # THE INVERSION.  `labelling[i]` is the OLD vertex at NEW position `i`;
    # relabelling needs the other direction.
    position = {old: new for new, old in enumerate(labelling)}

    if sorted(position) != list(range(n)) or sorted(position.values()) != list(range(n)):
        raise CompetitorError(
            f"pynauty.canon_label returned {labelling!r}, which is not a permutation "
            f"of range({n}); the canonical relabelling would not be a bijection"
        )
    out.add_edges_from((position[index[u]], position[index[v]]) for u, v in graph.edges())
    if out.number_of_edges() != graph.number_of_edges():
        raise CompetitorError(
            f"canonical relabelling collapsed {graph.number_of_edges()} edges to "
            f"{out.number_of_edges()}; the index map is wrong"
        )
    if verify and not nx.is_isomorphic(graph, out):
        raise CompetitorError(
            "the canonically relabelled graph is not isomorphic to its input; the "
            "networkx-label to pynauty-index map is wrong"
        )
    return out


def certificate(graph: nx.Graph) -> bytes:
    """``pynauty.certificate``: equal iff isomorphic.

    **For isomorphism assertions only.**  It is a padded machine-word bit
    matrix, so its length is a function of the word size rather than of the
    graph, and it must never appear in a compactness table.
    """
    pynauty = _pynauty()
    pg, _nodes = _to_pynauty(graph)
    result: bytes = pynauty.certificate(pg)
    return result


def automorphism_group_size(graph: nx.Graph) -> float:
    """``|Aut(G)|``, from ``pynauty.autgrp``'s (mantissa, exponent) pair.

    Free once this backend exists, and T-13 needs it for the complexity
    section's worst case (``corrections.md`` §5).  The running example
    ``C4(0,1,2,3) + K3(3,4,5)`` gives ``4.0``.

    A ``float`` because ``|Aut(G)|`` reaches ``n!`` and nauty reports it in
    scientific notation; exactness is not available from that API.
    """
    pynauty = _pynauty()
    pg, _nodes = _to_pynauty(graph)
    _generators, mantissa, exponent, _orbits, _n_orbits = pynauty.autgrp(pg)
    return float(mantissa) * 10.0 ** int(exponent)


def automorphism_orbits(graph: nx.Graph) -> tuple[int, ...]:
    """Vertex orbits under ``Aut(G)``, in the graph's own node order.

    Exposed because orbit pruning is the one real optimisation available to
    the AGM search (``agm.md`` §2.3).  **It is deliberately not wired into
    ``agm_cam``**: it changes the number of search nodes expanded, and the
    frozen 200k/100k budgets are the values behind published failure rates.
    """
    pynauty = _pynauty()
    pg, _nodes = _to_pynauty(graph)
    _generators, _mantissa, _exponent, orbits, _n_orbits = pynauty.autgrp(pg)
    return tuple(int(o) for o in orbits)


def upper_triangle_pairs(n: int) -> tuple[tuple[int, int], ...]:
    """``(i, j)``, ``i < j``, strict upper triangle read **column-wise**.

    ``(0,1) (0,2) (1,2) (0,3) (1,3) (2,3) ...`` -- graph6's payload order,
    the adjacency backend's symbol order, and AGM's code order, which is the
    same sequence read as the strict lower triangle row by row.
    """
    return tuple((i, j) for j in range(1, n) for i in range(j))


def graph6_prefix_bytes(n: int) -> int:
    """Length of graph6's ``N(n)`` prefix.

    One byte to ``n = 62``, four to ``n = 258047``, eight above.  **The
    branch is live**: Suite 2 reaches ``n = 98``, which is why the payload
    is unpacked from a measured offset rather than assumed to start at
    byte 1.
    """
    if n <= 62:
        return 1
    if n <= 258047:
        return 4
    return 8


def graph6_payload_bits(wire: bytes, n: int) -> tuple[str, ...]:
    """Unpack graph6's payload into the ``n(n-1)/2`` triangle bits.

    Each payload byte carries six bits, most significant first, as
    ``byte - 63``.  The last byte is zero-padded; the padding is dropped.
    """
    payload = wire[graph6_prefix_bytes(n) :]
    bits = "".join(format(byte - 63, "06b") for byte in payload)
    return tuple(bits[: n * (n - 1) // 2])


class NautyGraph6Backend(ReprBackend):
    """nauty's canonical labelling, serialised through graph6.

    Identical bit count to ``graph6`` by construction -- canonicalisation
    permutes bits without changing how many there are -- so the two must be
    printed as **one** Claim A row with a footnote naming both, not as two
    rows carrying the same numbers (``nauty.md`` §4).
    """

    name = "nauty_graph6"
    capabilities = frozenset(
        {
            Capability.POSITIONAL_FRAME,
            Capability.CANONICAL,
            Capability.COMPLETE_INVARIANT,
            Capability.REVERSIBLE,
            Capability.HANDLES_DISCONNECTED,
        }
    )

    def __init__(self, *, verify: bool = True) -> None:
        """Args:
        verify: passed to :func:`canonical_relabel`.  ``False`` drops the
            ``nx.is_isomorphic`` call for the language-matched timing mode.
        """
        self.verify = verify

    def encode(self, graph: nx.Graph, *, budget: Budget | None = None) -> Encoding:
        """Canonically relabel, then serialise through graph6.

        *budget* is accepted and ignored: nauty's worst case is exponential
        (Miyazaki 1997) but was not observed on this cohort, and a bound we
        never hit would be a bound whose failure rate we cannot report.
        """
        import networkx as nx

        canonical = canonical_relabel(graph, verify=self.verify)
        n = canonical.number_of_nodes()
        wire = nx.to_graph6_bytes(canonical, header=False).strip()
        text = wire.decode("ascii")
        triangle = n * (n - 1) // 2
        return Encoding(
            backend=self.name,
            symbols=tuple(text),
            alphabet_size=GRAPH6_ALPHABET_SIZE,
            n_nodes=n,
            n_edges=canonical.number_of_edges(),
            text=text,
            wire=wire,
            payload_bits=triangle,
            frame=PositionalFrame(
                n_nodes=n,
                pairs=upper_triangle_pairs(n),
                bits=graph6_payload_bits(wire, n),
            ),
        )

    def decode(self, encoding: Encoding) -> nx.Graph:
        """Rebuild the canonically labelled graph from its graph6 bytes."""
        import networkx as nx

        if encoding.wire is None:
            raise CompetitorError(f"{self.name!r} encoding carries no wire bytes to decode")
        return nx.from_graph6_bytes(encoding.wire)

    @classmethod
    def is_available(cls) -> bool:
        """``pynauty`` and ``networkx`` must both import."""
        try:
            import networkx  # noqa: F401
            import pynauty  # noqa: F401
        except ImportError:
            return False
        return True


class Sparse6NautyBackend(ReprBackend):
    """``sparse6.serialise(canonical_relabel(G))``.  Supplementary.

    One line of substance.  It removes the objection that the compactness
    axis compared a canonical method against a non-canonical one, and it
    costs nothing beyond the relabelling that ``nauty_graph6`` already does
    (``sparse6.md`` §3).

    **Not a family member.**  ``preregistration.md`` §4.1 fixes the Claim-A
    set at 6 and the Claim-B set at 7 and decision 23 freezes ``N_max``;
    this backend is reported in the supplementary grid only.  An agent that
    adds it to a confirmatory family has broken decision 23.
    """

    name = "sparse6_nauty"
    capabilities = frozenset(
        {
            Capability.CANONICAL,
            Capability.COMPLETE_INVARIANT,
            Capability.REVERSIBLE,
            Capability.HANDLES_DISCONNECTED,
        }
    )

    def __init__(self, *, verify: bool = True) -> None:
        self.verify = verify

    def encode(self, graph: nx.Graph, *, budget: Budget | None = None) -> Encoding:
        """Canonically relabel, then hand the result to agent A's serialiser.

        The import is inside the function so that registration succeeds even
        where ``backends/sparse6.py`` is absent -- which is the state of an
        isolated wave-1 worktree.  Only invocation fails there, and it fails
        with a diagnosis.
        """
        encoding = _sparse6_serialise()(canonical_relabel(graph, verify=self.verify))
        return dataclasses.replace(encoding, backend=self.name)

    def decode(self, encoding: Encoding) -> nx.Graph:
        """Rebuild the canonically labelled graph from its sparse6 bytes."""
        import networkx as nx

        if encoding.wire is None:
            raise CompetitorError(f"{self.name!r} encoding carries no wire bytes to decode")
        return nx.from_sparse6_bytes(encoding.wire)

    @classmethod
    def is_available(cls) -> bool:
        """``pynauty``, ``networkx`` and agent A's ``sparse6`` module.

        The sibling module is part of the availability contract because a
        backend that registers and then raises ``ModuleNotFoundError`` on
        first use is exactly the silent degrade the registry exists to
        prevent.  In the merged tree it is always present.
        """
        try:
            import networkx  # noqa: F401
            import pynauty  # noqa: F401

            importlib.import_module("isalgraph.competitors.backends.sparse6")
        except ImportError:
            return False
        return True


register_backend("nauty_graph6", NautyGraph6Backend)
register_backend("sparse6_nauty", Sparse6NautyBackend)

__all__ = [
    "GRAPH6_ALPHABET_SIZE",
    "SPARSE6_ALPHABET_SIZE",
    "NautyGraph6Backend",
    "Sparse6NautyBackend",
    "automorphism_group_size",
    "automorphism_orbits",
    "canonical_relabel",
    "certificate",
    "graph6_payload_bits",
    "graph6_prefix_bytes",
    "upper_triangle_pairs",
]
