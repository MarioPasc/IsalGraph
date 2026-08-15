"""Distance metrics for the (representation x distance) grid.

Nothing is imported here: the registry imports each module lazily on first
request, so a missing ``rapidfuzz`` does not break ``import
isalgraph.competitors``.

**No module in this package may read** :attr:`Encoding.text` **except**
:class:`~isalgraph.competitors.metrics.levenshtein.LevenshteinChar`, which
exists precisely to report the character-level answer as a supplementary
number.  A test enforces it.
"""

from __future__ import annotations

__all__: list[str] = []
