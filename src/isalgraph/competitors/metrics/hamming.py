"""Hamming distance, plain and padded.

**Plain Hamming is undefined for most pairs**, and that is the measurement,
not a nuisance.  graph6 encodes ``n`` in its header and packs ``n(n-1)/2``
bits, so graphs with different node counts give strings of different length;
node counts run 2-12 in Suite 1 and 2-98 in Suite 2, so equal-``n`` pairs are
a small minority.  On sparse6 the length varies with ``m`` and plain Hamming
is defined on only **30.8 %** of one-edit pairs.  Reporting "Hamming on
graph6 correlates poorly" without that denominator would record an artefact
of undefinedness as a finding.

**Padded Hamming reads the frame, not the string.**  The frame is the
triangle as index pairs *under the backend's own labelling*, so a canonical
backend pads its canonical frame and a non-canonical one pads its incident
frame, automatically.  ``scratch/backends.py::padded_hamming`` takes two
*graphs* and rebuilds both triangles from the incident node order, so it
cannot express that rule at all -- which is why it is on the do-not-port
list.
"""

from __future__ import annotations

from isalgraph.competitors.base import DistanceMetric, Encoding
from isalgraph.competitors.registry import register_metric
from isalgraph.errors import DistanceUndefined


class Hamming(DistanceMetric[Encoding]):
    """Plain Hamming over :attr:`Encoding.symbols`.  Equal length only."""

    name = "hamming"
    consumes = "symbols"
    is_pseudometric = False

    def is_defined(self, a: Encoding, b: Encoding) -> bool:
        return len(a.symbols) == len(b.symbols)

    def distance(self, a: Encoding, b: Encoding) -> float:
        if not self.is_defined(a, b):
            raise DistanceUndefined(
                f"hamming is undefined on lengths {len(a.symbols)} and "
                f"{len(b.symbols)}; this is an F1 result, not an error"
            )
        return float(sum(x != y for x, y in zip(a.symbols, b.symbols, strict=True)))


class PaddedHamming(DistanceMetric[Encoding]):
    """Hamming over a common ``max(n1, n2)`` triangle.

    Both codes are embedded in the frame of the larger graph and compared
    cell by cell.  A cell absent from the smaller graph reads 0, which is
    exactly the node insertion D6 charges -- that coincidence is the
    justification, and whether it survives contact with data is what the
    grid measures.

    **Requires both encodings to carry a frame.**  sparse6, min-DFS,
    IsalGraph and WL have none, so this metric is *undefined* there and the
    cell is printed as such.  That cell is one of the reasons the grid
    exists.
    """

    name = "padded_hamming"
    consumes = "frame"
    is_pseudometric = False

    def is_defined(self, a: Encoding, b: Encoding) -> bool:
        return a.frame is not None and b.frame is not None

    def distance(self, a: Encoding, b: Encoding) -> float:
        if not self.is_defined(a, b):
            missing = [e.backend for e in (a, b) if e.frame is None]
            raise DistanceUndefined(
                f"padded_hamming needs a positional frame; {missing} have none. "
                f"This is an F1 result and is reported, not repaired"
            )
        if a.frame is None or b.frame is None:  # pragma: no cover - narrowed above
            raise DistanceUndefined("padded_hamming needs a positional frame")
        # frame.bits, never .symbols: for graph6 a symbol is a six-bit ASCII
        # byte while a frame entry is one triangle cell, and zipping the two
        # would compare a byte against a cell without erroring.
        cells_a = dict(zip(a.frame.pairs, a.frame.bits, strict=True))
        cells_b = dict(zip(b.frame.pairs, b.frame.bits, strict=True))
        n = max(a.frame.n_nodes, b.frame.n_nodes)
        total = 0
        for j in range(n):
            for i in range(j):
                if cells_a.get((i, j), "0") != cells_b.get((i, j), "0"):
                    total += 1
        return float(total)


register_metric("hamming", Hamming)
register_metric("padded_hamming", PaddedHamming)

__all__ = ["Hamming", "PaddedHamming"]
