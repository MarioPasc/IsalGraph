"""The only module that produces a :class:`BitCount`.

Both conventions ``competitors.md`` §5 locks are always emitted together:
the **entropy bound** (``L log2 |Sigma|``, the like-for-like measure of
encoding efficiency) and the **realised bytes** (what a practitioner
stores).  They answer different questions and the paper must say which is
which.

Two rules that this module exists to enforce:

**Measure the wire, never a closed form.**  graph6's
``1 + ceil(n(n-1)/12)`` is correct only for ``n <= 62``; above that ``N(n)``
is four bytes and Suite 2 reaches ``n = 98``, so the branch is live.  The
closed forms live in the test suite as oracles inside their valid range;
production code counts the bytes ``networkx`` actually emitted.

**Never ``len(text) * 8``.**  ``'101001...'`` is a debugging view.  Counting
it at eight bits per character inflates the adjacency matrix eightfold and
hands us a baseline we beat for free.  Nothing here reads
:attr:`Encoding.text`, and a test asserts it.
"""

from __future__ import annotations

import math

from isalgraph.competitors.base import BitCount, Encoding
from isalgraph.errors import BitCountUndefined

#: Backends whose realised length is a character rendering rather than a
#: serialisation.  Their ``realised_bits`` is an inflated number and is
#: flagged so it cannot be quoted unlabelled.
_INFLATED = frozenset({"min_dfs"})

#: Backends that have no bit count at all.
_UNDEFINED = frozenset({"wl_subtree", "size_null"})


def _packed_bits(n_bits: int) -> int:
    """Bits used when ``n_bits`` raw bits are packed into whole bytes.

    The frozen formula for the adjacency triangle is ``8 * ceil(n(n-1)/16)``.
    Since ``n(n-1) = 2T`` for a triangle of ``T = n(n-1)/2`` bits, that is
    ``8 * ceil(T/8)`` -- **T bits packed into bytes**, and nothing else.

    The first draft read the formula's denominator as a word size and computed
    ``8 * ceil(T/16)``, halving every adjacency and AGM realised count: 8 bits
    for a 15-bit triangle at ``n = 6``, 2,384 where 4,760 is right at
    ``n = 98``.  ``entropy_bits`` was unaffected, so Claim A's headline table
    never moved -- but the realised-bytes column, which is the one a
    practitioner actually stores, was wrong by 2x.  Found by track A, 2026-08-15.
    """
    return 8 * math.ceil(n_bits / 8)


def count(encoding: Encoding) -> BitCount:
    """Return both bit conventions for *encoding*.

    Raises:
        BitCountUndefined: for ``wl_subtree`` and ``size_null``, which have
            no message length.  The Claim A cell is empty and the reason is
            printed; a fabricated number would be worse than a blank.
    """
    name = encoding.backend
    if name in _UNDEFINED:
        raise BitCountUndefined(
            f"{name!r} has no message length: it is not a serialisation, so any "
            f"bit count would measure the container rather than the encoding"
        )

    n, m = encoding.n_nodes, encoding.n_edges
    triangle = n * (n - 1) // 2

    if name in ("adjacency", "agm_cam"):
        return BitCount(
            backend=name,
            entropy_bits=float(triangle),
            realised_bits=_packed_bits(triangle),
            payload_bits=triangle,
        )

    if name in ("graph6", "nauty_graph6"):
        wire = _require_wire(encoding)
        return BitCount(
            backend=name,
            entropy_bits=6.0 * len(wire),
            realised_bits=8 * len(wire),
            payload_bits=triangle,
        )

    if name in ("sparse6", "sparse6_nauty"):
        wire = _require_wire(encoding)
        # The ':' prefix is framing, not payload: excluded from the entropy
        # bound, included in the realised bytes.  Decided once, here.
        return BitCount(
            backend=name,
            entropy_bits=6.0 * len(wire) - 6.0,
            realised_bits=8 * len(wire),
        )

    if name == "min_dfs":
        # A fixed-width upper bound, and a reviewer can say so: DFS indices
        # are not uniform on [0, n).  It is not tightened because the same
        # convention is applied to B_GED's 2M ceil(log2 N) endpoint
        # addressing, and tightening one and not the other is exactly the
        # asymmetry R3.6a objects to.  Consistency is the defence.
        width = 2 * max(n - 1, 1).bit_length()
        return BitCount(
            backend=name,
            entropy_bits=float(m * width),
            realised_bits=8 * len(encoding.text),
            inflated=True,
        )

    if name.startswith("isalgraph_"):
        length = encoding.length
        return BitCount(
            backend=name,
            entropy_bits=length * math.log2(encoding.alphabet_size),
            realised_bits=8 * length,
        )

    raise BitCountUndefined(
        f"no bit convention is registered for backend {name!r}; add a row to "
        f"bits.py rather than overriding ReprBackend.bits()"
    )


def _require_wire(encoding: Encoding) -> bytes:
    if encoding.wire is None:
        raise BitCountUndefined(
            f"backend {encoding.backend!r} must populate Encoding.wire: its bit "
            f"count is measured from the emitted bytes, never from a closed form"
        )
    return encoding.wire


def entropy_bits(length: int, alphabet_size: int) -> float:
    """``L log2 |Sigma|``.  Exposed for test oracles, not for backends."""
    return length * math.log2(alphabet_size)


__all__ = ["BitCount", "count", "entropy_bits"]
