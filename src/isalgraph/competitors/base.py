"""Protocols and value objects for the competitor representations.

Four of the seven documented traps in ``competitors/README.md`` §6 are the
same mistake: **a string was measured with the wrong unit**.  Counting
``'1010...'`` at eight bits per character inflates the adjacency matrix
eightfold; running Levenshtein on ``'0-1 1-2 2-0'`` charges four edits for
one deleted DFS tuple, a twofold difference.  Both are invisible and both
produce a plausible number.

A ``str`` return type cannot prevent either, so :meth:`ReprBackend.encode`
returns an :class:`Encoding` whose fields make the unit explicit and make
the wrong unit unreachable rather than merely discouraged.

This module imports nothing outside the standard library at runtime.
``networkx`` is an optional dependency of this subpackage and must not
appear in its import path, so its types arrive under ``TYPE_CHECKING`` and
``from __future__ import annotations`` keeps them as strings at runtime.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

from isalgraph.errors import BitCountUndefined, NotReversible

if TYPE_CHECKING:
    import networkx as nx

if TYPE_CHECKING:
    pass

#: What a metric reads out of an encoding.  Declared per metric so the grid
#: can reject an impossible cell before computing it.
Consumes = Literal["symbols", "text", "frame", "features", "order"]


class Capability(enum.Enum):
    """Properties a backend declares about itself.

    Declared, never inferred.  ``competitors.md`` §3.3 F2 requires the
    pseudometric declaration for the same reason: a property discovered by
    sampling is a property that can be wrong on the pair nobody sampled.
    """

    #: The encoding is a positional bit vector, so ``padded_hamming`` is
    #: principled rather than improvised.
    POSITIONAL_FRAME = "positional_frame"
    #: Isomorphic graphs receive identical encodings.
    CANONICAL = "canonical"
    #: Equal encodings imply isomorphic graphs.
    COMPLETE_INVARIANT = "complete_invariant"
    #: ``decode(encode(G))`` recovers the graph.
    REVERSIBLE = "reversible"
    #: Disconnected graphs and isolated vertices encode without error.
    HANDLES_DISCONNECTED = "handles_disconnected"
    #: Computable on Suite 1 only; raises above that scale rather than
    #: silently producing a partially complete column.
    SUITE1_ONLY = "suite1_only"
    #: A descriptive null.  ``grid.py`` refuses to select one as any
    #: representation's primary distance, and it is outside the frozen
    #: confirmatory family (decision 23).
    BASELINE = "baseline"


@dataclass(frozen=True, slots=True)
class PositionalFrame:
    """The index pairs a positional code occupies, under its own labelling.

    ``pairs[k]`` is the ``(i, j)`` cell that symbol ``k`` reports, with
    ``i < j``.  The frame is expressed in the **backend's own** labelling,
    so a canonical backend pads its canonical frame and a non-canonical one
    pads its incident frame, automatically.

    That is the whole point of carrying a frame rather than re-deriving the
    triangle from the graph: ``scratch/backends.py::padded_hamming`` takes
    two *graphs* and builds both triangles from the incident node order, so
    it cannot express the canonical-frame rule at all.

    **The frame carries its own bits, and this is not redundant.**  The edit
    unit and the positional unit differ for half the pool: graph6's
    ``symbols`` are six-bit ASCII *bytes* while its frame is the unpacked
    *bit* triangle underneath them.  Zipping ``pairs`` against ``symbols``
    would silently compare a byte against a cell for exactly those backends,
    so ``bits`` is stored explicitly and ``padded_hamming`` reads only this
    object.  For the adjacency matrix the two coincide; for graph6,
    nauty->graph6 and AGM they do not.

    Attributes:
        n_nodes: order of the graph this frame describes.
        pairs: ``(i, j)`` with ``i < j``, in the order the format emits them
            -- strict upper triangle, **column-wise**.
        bits: ``'0'``/``'1'`` per entry of *pairs*, same length, same order.
    """

    n_nodes: int
    pairs: tuple[tuple[int, int], ...]
    bits: tuple[str, ...]

    def __post_init__(self) -> None:
        expected = self.n_nodes * (self.n_nodes - 1) // 2
        if len(self.pairs) != expected:
            raise ValueError(
                f"frame has {len(self.pairs)} pairs, expected {expected} for n={self.n_nodes}"
            )
        if len(self.bits) != len(self.pairs):
            raise ValueError(
                f"frame has {len(self.bits)} bits for {len(self.pairs)} pairs; "
                f"padded_hamming reads them positionally and cannot align a mismatch"
            )


@dataclass(frozen=True, slots=True)
class BitCount:
    """Both bit conventions ``competitors.md`` §5 locks, always together.

    Attributes:
        entropy_bits: ``L log2 |Sigma|`` for a string representation,
            ``n(n-1)/2`` for a raw bit vector.  The like-for-like measure of
            encoding efficiency.
        realised_bits: the serialised length as the format defines it --
            what a practitioner actually stores.
        payload_bits: the format-defined payload where it differs from the
            wire, e.g. graph6's ``n(n-1)/2`` inside its six-bit ASCII.
        inflated: ``True`` when ``realised_bits`` is a character rendering
            rather than a real serialisation.  min-DFS is the only such
            case, and the flag is what stops it being quoted unlabelled.
    """

    backend: str
    entropy_bits: float
    realised_bits: int
    payload_bits: int | None = None
    inflated: bool = False


@dataclass(frozen=True, slots=True)
class Encoding:
    """One graph's encoding under one backend.

    Attributes:
        backend: provenance, carried into every record.
        symbols: **the comparison unit.**  One entry is one edit.  For
            min-DFS one symbol is one DFS tuple; for graph6 one symbol is
            one ASCII byte; for the adjacency matrix one symbol is one bit.
            Every edit distance runs on this and on nothing else.
        alphabet_size: ``|Sigma|`` for the entropy bound.  May depend on
            ``n`` -- min-DFS's alphabet grows as ``O(n^2)``.
        wire: the realised serialisation, exactly as the format emitted it.
            **Measured, never computed**: graph6's closed form
            ``1 + ceil(n(n-1)/12)`` is correct only for ``n <= 62`` and
            Suite 2 reaches ``n = 98``.
        payload_bits: format-defined payload where it differs from the wire.
        frame: present **iff** the code is a positional bit vector.  Its
            absence is what makes ``padded_hamming`` undefined on sparse6,
            min-DFS, IsalGraph and WL -- a reported F1 result, not an error.
        text: **figures and debugging only, never measured.**  A test
            asserts that nothing under ``metrics/`` or ``bits.py``
            references it, except the one sanctioned reader
            ``levenshtein_char``.
    """

    backend: str
    symbols: tuple[str, ...]
    alphabet_size: int
    n_nodes: int
    n_edges: int
    text: str
    wire: bytes | None = None
    payload_bits: int | None = None
    frame: PositionalFrame | None = None

    @property
    def length(self) -> int:
        """Number of symbols, i.e. the unit-of-edit length."""
        return len(self.symbols)


@dataclass(frozen=True, slots=True)
class Budget:
    """A bound a backend must respect, and raise on rather than degrade.

    ``None`` means unbounded.  Only the fields a backend declares are read;
    passing a budget a backend ignores is not an error, so one budget object
    can be threaded through a heterogeneous grid.
    """

    #: AGM: branch-and-bound search nodes.  200_000 Suite 1, 100_000 Suite 2.
    search_nodes: int | None = None
    #: min-DFS: retained embeddings.  50_000, frozen -- a MEMORY cap, since
    #: the first Suite-2 run was OOM-killed rather than slow.
    max_projections: int | None = None
    #: IsalGraph: wall clock in seconds.
    timeout_s: float | None = None


class ReprBackend(ABC):
    """A graph serialisation that produces a comparable symbol sequence."""

    #: Registry key.  Also the ``Encoding.backend`` provenance string.
    name: str = ""
    #: Declared properties.  See :class:`Capability`.
    capabilities: frozenset[Capability] = frozenset()

    @abstractmethod
    def encode(self, graph: nx.Graph, *, budget: Budget | None = None) -> Encoding:
        """Encode *graph*.

        Args:
            graph: a ``networkx.Graph``.
            budget: bound to respect.  ``None`` uses the backend's frozen
                default, which is the value behind its published failure
                rate -- do not change one without changing that number.

        Returns:
            The encoding.

        Raises:
            BudgetExceeded: when the budget runs out.  **Never returns an
                incumbent.**
            SuiteScopeError: when the backend declares ``SUITE1_ONLY`` and
                the graph is outside that scope.
        """

    def decode(self, encoding: Encoding) -> nx.Graph:
        """Reconstruct a graph from *encoding*.

        Raises:
            NotReversible: unless the backend declares
                :attr:`Capability.REVERSIBLE`.
        """
        raise NotReversible(f"backend {self.name!r} is not reversible")

    def bits(self, encoding: Encoding) -> BitCount:
        """Return both bit conventions for *encoding*.

        Delegates to :mod:`isalgraph.competitors.bits`, which is the only
        module permitted to produce a :class:`BitCount`.  Overriding this is
        how a backend gets a wrong bit count into a paper table, so do not.
        """
        from isalgraph.competitors import bits as _bits

        return _bits.count(encoding)

    @classmethod
    def is_available(cls) -> bool:
        """Whether this backend's third-party dependency imports."""
        return True

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r}>"


class VectorBackend(ABC):
    """A feature-vector representation.  WL only.  **Not** a ``ReprBackend``.

    A separate protocol rather than a ``ReprBackend`` with a raising
    ``bits()``, per ``wl-subtree-kernel.md`` §7: fabricating a WL bit count
    is the failure mode to avoid, and a separate type makes it *unreachable*
    rather than merely forbidden.  There is deliberately no ``bits`` method
    here -- the Claim A cell is empty with the reason printed.
    """

    name: str = ""
    capabilities: frozenset[Capability] = frozenset()

    @abstractmethod
    def fit(self, graphs: Sequence[nx.Graph]) -> None:
        """Build the colour vocabulary from *graphs*.

        **Per dataset, never per batch.**  Fitting on a subset produces a
        different vocabulary and therefore different distances, which makes
        the distance matrix depend on batching order -- silent corruption of
        the same family as GEDLIB's ``get_lower_bound()`` trap.
        """

    @abstractmethod
    def features(self, graph: nx.Graph) -> Mapping[str, int]:
        """Return the fitted feature multiset for *graph*."""

    def bit_count_reason(self) -> str:
        """Why this representation has no bit count, for the empty cell."""
        return (
            "no bit count: a feature-vector cost (dimension x counter width) would "
            "measure the choice of container, not the encoding"
        )

    @classmethod
    def is_available(cls) -> bool:
        """Whether this backend's third-party dependency imports."""
        return True

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r}>"


#: What a metric compares: an :class:`Encoding` for every serialisation, or a
#: fitted feature multiset for the one :class:`VectorBackend`.
Comparable = Encoding | Mapping[str, int]

#: A metric is generic in what it compares, so ``Levenshtein`` can state that
#: it takes an :class:`Encoding` without violating the Liskov substitution
#: principle against a base declared over the union.
ComparableT = TypeVar("ComparableT", bound=Comparable)


class DistanceMetric(ABC, Generic[ComparableT]):
    """A distance over encodings, with its definedness and axioms declared."""

    #: Registry key.
    name: str = ""
    #: What this metric reads.  The grid rejects a cell whose backend cannot
    #: supply it, before computing anything.
    consumes: Consumes = "symbols"
    #: **Declared, per F2, never inferred.**  Comparing a metric against a
    #: non-metric is legitimate and must be stated.
    is_pseudometric: bool = False

    @abstractmethod
    def is_defined(self, a: ComparableT, b: ComparableT) -> bool:
        """Whether :meth:`distance` is computable on this pair.  This is F1."""

    @abstractmethod
    def distance(self, a: ComparableT, b: ComparableT) -> float:
        """Distance between two encodings.

        Precondition: :meth:`is_defined`.

        Raises:
            DistanceUndefined: when the precondition fails.  That is a
                **result**, not a defect.
        """

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r}>"


__all__ = [
    "BitCount",
    "BitCountUndefined",
    "Budget",
    "Capability",
    "Comparable",
    "ComparableT",
    "Consumes",
    "DistanceMetric",
    "Encoding",
    "PositionalFrame",
    "ReprBackend",
    "VectorBackend",
]
