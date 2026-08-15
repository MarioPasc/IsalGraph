"""Edit distance, at the symbol level (primary) and the character level.

The distinction is not cosmetic.  Running Levenshtein on min-DFS's character
rendering ``'0-1 1-2 2-0'`` charges **four** edits for one deleted tuple
(``' 5-2'``); at the tuple level it charges **one**, which is the
semantically correct unit and the like-for-like comparison against
IsalGraph, whose symbols are also single operations.  Measured, the gap is
twofold.

``rapidfuzz.distance.Levenshtein.distance`` accepts ``tuple[str, ...]`` and
returns the symbol-level answer directly -- 1 where the character-level
answer is 4 -- so no interning and no extra dependency are needed.
"""

from __future__ import annotations

from types import ModuleType

from isalgraph.competitors.base import DistanceMetric, Encoding
from isalgraph.competitors.registry import register_metric
from isalgraph.errors import BackendUnavailableError


def _rapidfuzz() -> ModuleType:
    try:
        from rapidfuzz.distance import Levenshtein as _lev  # noqa: N813
    except ImportError as exc:  # pragma: no cover - exercised via sys.modules patching
        raise BackendUnavailableError(f"the levenshtein metric requires rapidfuzz: {exc}") from exc
    return _lev


class Levenshtein(DistanceMetric[Encoding]):
    """Symbol-level edit distance.  **The primary edit distance.**

    Runs on :attr:`Encoding.symbols` and on nothing else.  One symbol is one
    edit: one DFS tuple for min-DFS, one ASCII byte for graph6, one triangle
    bit for the adjacency matrix, one instruction for IsalGraph.
    """

    name = "levenshtein"
    consumes = "symbols"
    is_pseudometric = False

    def is_defined(self, a: Encoding, b: Encoding) -> bool:
        """Always defined: edit distance needs no length agreement."""
        return True

    def distance(self, a: Encoding, b: Encoding) -> float:
        return float(_rapidfuzz().distance(a.symbols, b.symbols))


class LevenshteinChar(DistanceMetric[Encoding]):
    """Character-level edit distance.  **Supplementary only.**

    The one sanctioned reader of :attr:`Encoding.text`.  It exists so the
    supplementary grid can *report* the twofold gap against the tuple-level
    answer rather than leave a reader to wonder which convention produced a
    number.  It is never a primary distance.
    """

    name = "levenshtein_char"
    consumes = "text"
    is_pseudometric = False

    def is_defined(self, a: Encoding, b: Encoding) -> bool:
        return True

    def distance(self, a: Encoding, b: Encoding) -> float:
        return float(_rapidfuzz().distance(a.text, b.text))


register_metric("levenshtein", Levenshtein)
register_metric("levenshtein_char", LevenshteinChar)

__all__ = ["Levenshtein", "LevenshteinChar"]
