"""Backend implementations.  Nothing is imported here.

The registry imports each module lazily on first request, so a missing
``pynauty`` or ``grakel`` does not break ``import isalgraph.competitors``.
Every module in this package registers itself at import time by calling
:func:`~isalgraph.competitors.registry.register_backend`.
"""

from __future__ import annotations

__all__: list[str] = []
