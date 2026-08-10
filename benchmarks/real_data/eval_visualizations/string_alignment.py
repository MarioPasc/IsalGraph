"""Compatibility shim -- the implementation now lives in ``isalgraph.viz``.

See :mod:`isalgraph.viz.alignment_view`. Behaviour is unchanged; new
figure code should import from ``isalgraph.viz`` directly.
"""

from __future__ import annotations

from isalgraph.viz.alignment_view import (
    _OP_ALPHA,
    _OP_COLORS,
    draw_alignment,
    levenshtein_alignment,
)

__all__ = ["_OP_ALPHA", "_OP_COLORS", "draw_alignment", "levenshtein_alignment"]
